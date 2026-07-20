"""CPU-side block-W8 conversion and references for the fused decode MoE path."""

from dataclasses import dataclass
import math
import os

import torch
import torch.nn.functional as F


BLOCK_SIZE = 128
LEGACY_E4M3_MAX = 240.0
INT8_MAX = 127.0
ROW_FP8_PROJECTION_CHOICES = ("all", "gate_up", "down", "none")
MOE_ROUNDING_STAGES = frozenset(
    ("affinity_bf16", "activation_bf16", "local_output_bf16")
)


def _byte_values(raw):
    if raw.element_size() != 1:
        raise TypeError(f"FP8 storage must contain one-byte elements, got {raw.dtype}")
    return raw.contiguous().view(torch.uint8)


def decode_e4m3fn(raw, *, reject_nan=True):
    """Decode raw safetensors F8_E4M3 (OCP E4M3FN) bytes to float32.

    This is implemented from the bit layout and does not depend on a PyTorch
    float8 dtype. E4M3FN uses exponent 15 for finite values through 448; only
    exponent=15,mantissa=7 is NaN.
    """
    bits = _byte_values(raw)
    sign = (bits >> 7).to(torch.bool)
    exponent = ((bits >> 3) & 0x0F).to(torch.int32)
    mantissa = (bits & 0x07).to(torch.int32)
    is_nan = (exponent == 0x0F) & (mantissa == 0x07)
    if reject_nan and bool(is_nan.any()):
        count = int(is_nan.sum())
        raise ValueError(f"F8_E4M3 tensor contains {count} NaN encoding(s)")

    exp_f = exponent.to(torch.float32)
    mant_f = mantissa.to(torch.float32)
    subnormal = mant_f * (2.0 ** -9)
    normal = (1.0 + mant_f / 8.0) * torch.pow(2.0, exp_f - 7.0)
    value = torch.where(exponent == 0, subnormal, normal)
    value = torch.where(sign, -value, value)
    if not reject_nan:
        value = torch.where(is_nan, torch.full_like(value, float("nan")), value)
    return value


def _legacy_positive_values(device):
    codes = torch.arange(0x78, dtype=torch.uint8, device=device)
    exponent = ((codes >> 3) & 0x0F).to(torch.int32)
    mantissa = (codes & 0x07).to(torch.float32)
    subnormal = mantissa * (2.0 ** -9)
    normal = (1.0 + mantissa / 8.0) * torch.pow(
        2.0, exponent.to(torch.float32) - 7.0
    )
    return codes, torch.where(exponent == 0, subnormal, normal)


def validate_legacy_e4m3_bytes(raw):
    """Reject bytes that legacy Trn2 E4M3 interprets as Inf/NaN."""
    bits = _byte_values(raw)
    unsafe = ((bits >> 3) & 0x0F) == 0x0F
    if bool(unsafe.any()):
        count = int(unsafe.sum())
        raise ValueError(f"legacy E4M3 tensor contains {count} unsafe encoding(s)")


def encode_legacy_e4m3(values):
    """Round float values to finite Trn2 legacy E4M3 and return int8 bytes."""
    if not torch.is_floating_point(values):
        values = values.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("cannot encode non-finite values as legacy E4M3")

    magnitude = values.abs().float().clamp_max(LEGACY_E4M3_MAX)
    codes, positive = _legacy_positive_values(magnitude.device)
    midpoints = (positive[:-1] + positive[1:]) * 0.5
    selected = torch.bucketize(magnitude.reshape(-1), midpoints, right=False)

    # bucketize chooses the lower code at a midpoint. E4M3 uses round-to-nearest
    # ties-to-even, so move to the upper code when the lower significand is odd.
    if selected.numel():
        lower = selected.clamp_max(midpoints.numel() - 1)
        tied = (
            (selected < positive.numel() - 1)
            & (magnitude.reshape(-1) == midpoints.index_select(0, lower))
        )
        lower_is_odd = (selected & 1) != 0
        selected = selected + (tied & lower_is_odd).to(selected.dtype)

    encoded = codes.index_select(0, selected).reshape(values.shape)
    sign = torch.signbit(values) & (magnitude != 0)
    encoded = encoded | (sign.to(torch.uint8) << 7)
    validate_legacy_e4m3_bytes(encoded)
    return encoded.contiguous().view(torch.int8)


def decode_legacy_e4m3(raw):
    """Decode finite legacy E4M3 bytes, rejecting its reserved exponent."""
    validate_legacy_e4m3_bytes(raw)
    bits = _byte_values(raw)
    sign = (bits >> 7).to(torch.bool)
    exponent = ((bits >> 3) & 0x0F).to(torch.int32)
    mantissa = (bits & 0x07).to(torch.float32)
    subnormal = mantissa * (2.0 ** -9)
    normal = (1.0 + mantissa / 8.0) * torch.pow(
        2.0, exponent.to(torch.float32) - 7.0
    )
    value = torch.where(exponent == 0, subnormal, normal)
    return torch.where(sign, -value, value)


def _scale_grid_shape(shape, block_size):
    rows, cols = shape[-2:]
    return (
        math.ceil(rows / block_size),
        math.ceil(cols / block_size),
    )


def expand_block_scales(scales, weight_shape, block_size=BLOCK_SIZE):
    """Expand [...,row_blocks,col_blocks] scales to a weight tensor shape."""
    expected = (*weight_shape[:-2], *_scale_grid_shape(weight_shape, block_size))
    if tuple(scales.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(scales.shape)} does not match expected {expected}"
        )
    expanded = scales.repeat_interleave(block_size, dim=-2).repeat_interleave(
        block_size, dim=-1
    )
    return expanded[..., : weight_shape[-2], : weight_shape[-1]]


def dequantize_official_fp8(raw, scales, block_size=BLOCK_SIZE):
    """Reconstruct an official block-scaled E4M3FN tensor as float32."""
    if scales.dtype != torch.bfloat16:
        raise TypeError(f"official weight_scale_inv must be BF16, got {scales.dtype}")
    decoded = decode_e4m3fn(raw)
    return decoded * expand_block_scales(
        scales.float(), tuple(raw.shape), block_size
    )


@dataclass
class QuantizationStats:
    dot: float = 0.0
    source_sq: float = 0.0
    target_sq: float = 0.0
    error_sq: float = 0.0
    count: int = 0
    exact_count: int = 0
    block_count: int = 0
    shifted_block_count: int = 0
    clipped_count: int = 0

    def update(self, source, target):
        source = source.double()
        target = target.double()
        self.dot += float((source * target).sum())
        self.source_sq += float(source.square().sum())
        self.target_sq += float(target.square().sum())
        self.error_sq += float((source - target).square().sum())
        self.count += source.numel()
        self.exact_count += int((source == target).sum())

    def merge(self, other):
        self.dot += other.dot
        self.source_sq += other.source_sq
        self.target_sq += other.target_sq
        self.error_sq += other.error_sq
        self.count += other.count
        self.exact_count += other.exact_count
        self.block_count += other.block_count
        self.shifted_block_count += other.shifted_block_count
        self.clipped_count += other.clipped_count
        return self

    @property
    def cosine(self):
        denom = math.sqrt(self.source_sq * self.target_sq)
        return self.dot / denom if denom else 1.0

    @property
    def normalized_rmse(self):
        return math.sqrt(self.error_sq / max(self.source_sq, 1e-30))

    @property
    def exact_fraction(self):
        return self.exact_count / self.count if self.count else 1.0

    @property
    def shifted_block_fraction(self):
        if not self.block_count:
            return 0.0
        return self.shifted_block_count / self.block_count


def requantize_official_fp8(
    raw,
    source_scales,
    mode,
    block_size=BLOCK_SIZE,
):
    """Convert official E4M3FN blocks to legacy E4M3 or symmetric INT8 blocks.

    Returns `(bytes, bf16_scales, stats)`. The output scale grid has the same
    orientation as the source weight matrix and is consumed directly by NKI.
    """
    if mode not in ("fp8", "int8"):
        raise ValueError(f"mode must be 'fp8' or 'int8', got {mode!r}")
    if source_scales.dtype != torch.bfloat16:
        raise TypeError(
            f"official weight_scale_inv must be BF16, got {source_scales.dtype}"
        )
    expected = (*raw.shape[:-2], *_scale_grid_shape(raw.shape, block_size))
    if tuple(source_scales.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(source_scales.shape)} does not match expected {expected}"
        )

    result = torch.empty_like(raw, dtype=torch.int8)
    result_scales = torch.empty_like(source_scales, dtype=torch.bfloat16)
    stats = QuantizationStats()
    leading_shape = raw.shape[:-2]
    leading_count = math.prod(leading_shape) if leading_shape else 1
    raw_flat = raw.reshape(leading_count, *raw.shape[-2:])
    source_scale_flat = source_scales.reshape(
        leading_count, *source_scales.shape[-2:]
    )
    result_flat = result.reshape(leading_count, *raw.shape[-2:])
    result_scale_flat = result_scales.reshape(
        leading_count, *source_scales.shape[-2:]
    )
    rows, cols = raw.shape[-2:]
    limit = LEGACY_E4M3_MAX if mode == "fp8" else INT8_MAX

    for leading in range(leading_count):
        for row_block, row_start in enumerate(range(0, rows, block_size)):
            row_end = min(row_start + block_size, rows)
            for col_block, col_start in enumerate(range(0, cols, block_size)):
                col_end = min(col_start + block_size, cols)
                source = decode_e4m3fn(
                    raw_flat[leading, row_start:row_end, col_start:col_end]
                )
                source = source * source_scale_flat[
                    leading, row_block, col_block
                ].float()
                scale = source.abs().amax().clamp_min(1e-12) / limit
                stored_scale = scale.to(torch.bfloat16)
                normalized = source / stored_scale.float()
                if mode == "fp8":
                    quantized = encode_legacy_e4m3(normalized)
                    reconstructed = (
                        decode_legacy_e4m3(quantized) * stored_scale.float()
                    )
                else:
                    quantized = (
                        normalized.round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
                    )
                    reconstructed = quantized.float() * stored_scale.float()
                result_flat[
                    leading, row_start:row_end, col_start:col_end
                ] = quantized
                result_scale_flat[leading, row_block, col_block] = stored_scale
                stats.update(source, reconstructed)

    return result, result_scales, stats


def requantize_official_fp8_pow2(
    raw,
    source_scales,
    block_size=BLOCK_SIZE,
):
    """Convert E4M3FN blocks to native E4M3 with an exact scale exponent shift.

    A source block containing any E4M3FN-only finite code is divided by two
    while its external BF16 scale is doubled. This maps values 256..448 to
    native E4M3 values 128..224. All values except the smallest normal and
    subnormal codes remain exactly representable. Blocks without extended
    codes retain their bytes and scale bit-for-bit.
    """
    if source_scales.dtype != torch.bfloat16:
        raise TypeError(
            f"official weight_scale_inv must be BF16, got {source_scales.dtype}"
        )
    expected = (*raw.shape[:-2], *_scale_grid_shape(raw.shape, block_size))
    if tuple(source_scales.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(source_scales.shape)} does not match expected {expected}"
        )

    result = torch.empty_like(raw, dtype=torch.int8)
    result_scales = torch.empty_like(source_scales)
    stats = QuantizationStats()
    leading_shape = raw.shape[:-2]
    leading_count = math.prod(leading_shape) if leading_shape else 1
    raw_flat = raw.reshape(leading_count, *raw.shape[-2:])
    source_scale_flat = source_scales.reshape(
        leading_count, *source_scales.shape[-2:]
    )
    result_flat = result.reshape(leading_count, *raw.shape[-2:])
    result_scale_flat = result_scales.reshape(
        leading_count, *source_scales.shape[-2:]
    )
    rows, cols = raw.shape[-2:]

    for leading in range(leading_count):
        for row_block, row_start in enumerate(range(0, rows, block_size)):
            row_end = min(row_start + block_size, rows)
            for col_block, col_start in enumerate(range(0, cols, block_size)):
                col_end = min(col_start + block_size, cols)
                source_raw = raw_flat[
                    leading, row_start:row_end, col_start:col_end
                ]
                source_normalized = decode_e4m3fn(source_raw)
                source_scale = source_scale_flat[
                    leading, row_block, col_block
                ]
                source_scale_f32 = source_scale.float()
                if not bool(torch.isfinite(source_scale_f32)):
                    raise ValueError(
                        "official weight_scale_inv contains a non-finite value"
                    )

                magnitude = _byte_values(source_raw) & 0x7F
                shifted = bool((magnitude >= 0x78).any())
                stats.block_count += 1
                if shifted:
                    stats.shifted_block_count += 1
                    shifted_values = source_normalized * 0.5
                    stats.clipped_count += int(
                        (shifted_values.abs() > LEGACY_E4M3_MAX).sum()
                    )
                    quantized = encode_legacy_e4m3(shifted_values)
                    stored_scale = (source_scale_f32 * 2.0).to(
                        torch.bfloat16
                    )
                    if (
                        not bool(torch.isfinite(stored_scale.float()))
                        or bool(
                            stored_scale.float() * 0.5
                            != source_scale_f32
                        )
                    ):
                        raise ValueError(
                            "doubling weight_scale_inv is not exactly "
                            "representable as finite BF16"
                        )
                else:
                    quantized = _byte_values(source_raw).contiguous().view(
                        torch.int8
                    )
                    stored_scale = source_scale

                validate_legacy_e4m3_bytes(quantized)
                source = source_normalized * source_scale_f32
                reconstructed = (
                    decode_legacy_e4m3(quantized) * stored_scale.float()
                )
                result_flat[
                    leading, row_start:row_end, col_start:col_end
                ] = quantized
                result_scale_flat[
                    leading, row_block, col_block
                ] = stored_scale
                stats.update(source, reconstructed)

    if stats.clipped_count:
        raise RuntimeError(
            f"power-of-two E4M3 conversion clipped {stats.clipped_count} values"
        )
    return result, result_scales, stats


def requantize_official_fp8_row(
    raw,
    source_scales,
    scale_factors=(0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0),
):
    """Requantize to legacy E4M3 with an MSE-selected scale per output row."""
    source = dequantize_official_fp8(raw, source_scales)
    absmax_scale = (
        source.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        / LEGACY_E4M3_MAX
    )
    best_error = torch.full(
        source.shape[:-1], float("inf"), dtype=torch.float32
    )
    best_scale = torch.empty_like(absmax_scale)
    best_quantized = torch.empty_like(raw, dtype=torch.int8)
    for factor in scale_factors:
        candidate_scale = absmax_scale * factor
        candidate = encode_legacy_e4m3(source / candidate_scale)
        candidate_reconstructed = (
            decode_legacy_e4m3(candidate) * candidate_scale
        )
        error = (source - candidate_reconstructed).square().sum(dim=-1)
        improved = error < best_error
        best_error = torch.where(improved, error, best_error)
        best_scale = torch.where(
            improved.unsqueeze(-1), candidate_scale, best_scale
        )
        best_quantized = torch.where(
            improved.unsqueeze(-1), candidate, best_quantized
        )

    reconstructed = decode_legacy_e4m3(best_quantized) * best_scale
    stats = QuantizationStats()
    stats.update(source, reconstructed)
    return (
        best_quantized,
        best_scale.squeeze(-1).contiguous(),
        stats,
    )


def retain_official_fp8_row_bf16(raw, source_scales):
    """Retain one projection in BF16 while preserving the row-scaled layout."""
    source = dequantize_official_fp8(raw, source_scales)
    retained = source.to(torch.bfloat16)
    scales = torch.ones(source.shape[:-1], dtype=torch.float32)
    stats = QuantizationStats()
    stats.update(source, retained.float())
    return retained, scales, stats


def retain_official_fp8(raw, source_scales, block_size=BLOCK_SIZE):
    """Validate and retain official E4M3FN bytes and block scales exactly."""
    if source_scales.dtype != torch.bfloat16:
        raise TypeError(
            f"official weight_scale_inv must be BF16, got {source_scales.dtype}"
        )
    expected = (*raw.shape[:-2], *_scale_grid_shape(raw.shape, block_size))
    if tuple(source_scales.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(source_scales.shape)} does not match expected {expected}"
        )
    source = dequantize_official_fp8(raw, source_scales, block_size)
    stats = QuantizationStats()
    stats.update(source, source)
    return (
        raw.contiguous().view(torch.int8),
        source_scales.contiguous(),
        stats,
    )


def split_official_fp8(raw, source_scales, block_size=BLOCK_SIZE):
    """Split E4M3FN into two finite legacy-E4M3 planes exactly.

    Codes through magnitude 0x77 are shared by E4M3FN and legacy E4M3. The
    E4M3FN-only finite values 0x78..0x7e are represented as signed 240 in the
    base plane plus a signed residual in the second plane. Every residual is
    itself exactly representable by legacy E4M3, so both planes can be consumed
    directly by TensorE with the original block scale.
    """
    if source_scales.dtype != torch.bfloat16:
        raise TypeError(
            f"official weight_scale_inv must be BF16, got {source_scales.dtype}"
        )
    expected = (*raw.shape[:-2], *_scale_grid_shape(raw.shape, block_size))
    if tuple(source_scales.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(source_scales.shape)} does not match expected {expected}"
        )

    bits = _byte_values(raw)
    source = decode_e4m3fn(bits)
    magnitude = bits & 0x7F
    base_magnitude = torch.minimum(
        magnitude, torch.tensor(0x77, dtype=torch.uint8, device=bits.device)
    )
    base_bits = base_magnitude | (bits & 0x80)
    base = base_bits.contiguous().view(torch.int8)
    residual = encode_legacy_e4m3(source - decode_legacy_e4m3(base))
    reconstructed = decode_legacy_e4m3(base) + decode_legacy_e4m3(residual)
    if not torch.equal(reconstructed, source):
        raise RuntimeError("legacy FP8 planes did not reconstruct E4M3FN exactly")

    stats = QuantizationStats()
    stats.update(source, reconstructed)
    return base, residual, source_scales.contiguous(), stats


def pack_coalesced_block_scales(gate_up_scales, down_scales):
    """Pack block scales in projection, contraction, output-block order.

    Input grids retain the official projection orientation:
    gate/up `[E,2,I/128,H/128]` and down `[E,H/128,I/128]`.
    Each scalar is repeated over the NKI partition dimension so one expert's
    complete tables can be loaded and converted to FP32 once.
    """
    if gate_up_scales.ndim != 4 or gate_up_scales.shape[1] != 2:
        raise ValueError(
            "gate/up block scales must have shape [E,2,I/128,H/128]"
        )
    if down_scales.ndim != 3:
        raise ValueError(
            "down block scales must have shape [E,H/128,I/128]"
        )
    experts, _, intermediate_blocks, hidden_blocks = gate_up_scales.shape
    if tuple(down_scales.shape) != (
        experts,
        hidden_blocks,
        intermediate_blocks,
    ):
        raise ValueError("gate/up and down block-scale grids are inconsistent")
    if gate_up_scales.dtype != torch.bfloat16:
        raise TypeError("gate/up block scales must be BF16")
    if down_scales.dtype != torch.bfloat16:
        raise TypeError("down block scales must be BF16")

    gate_up_flat = (
        gate_up_scales.permute(0, 1, 3, 2)
        .reshape(experts, -1)
        .contiguous()
    )
    down_flat = (
        down_scales.transpose(1, 2)
        .reshape(experts, -1)
        .contiguous()
    )
    return (
        gate_up_flat.unsqueeze(1)
        .expand(experts, BLOCK_SIZE, gate_up_flat.shape[1])
        .contiguous(),
        down_flat.unsqueeze(1)
        .expand(experts, BLOCK_SIZE, down_flat.shape[1])
        .contiguous(),
    )


def unpack_coalesced_block_scales(
    gate_up_scales,
    down_scales,
    hidden_size,
    intermediate_size,
):
    """Recover NKI-oriented scale grids from coalesced repeated tables."""
    if hidden_size % BLOCK_SIZE or intermediate_size % BLOCK_SIZE:
        raise ValueError("hidden and intermediate dimensions must be block aligned")
    hidden_blocks = hidden_size // BLOCK_SIZE
    intermediate_blocks = intermediate_size // BLOCK_SIZE
    experts = gate_up_scales.shape[0]
    expected_gate_up = (
        experts,
        BLOCK_SIZE,
        2 * hidden_blocks * intermediate_blocks,
    )
    expected_down = (
        experts,
        BLOCK_SIZE,
        intermediate_blocks * hidden_blocks,
    )
    if tuple(gate_up_scales.shape) != expected_gate_up:
        raise ValueError(
            f"gate/up scale table {tuple(gate_up_scales.shape)} does not "
            f"match expected {expected_gate_up}"
        )
    if tuple(down_scales.shape) != expected_down:
        raise ValueError(
            f"down scale table {tuple(down_scales.shape)} does not "
            f"match expected {expected_down}"
        )
    if not torch.equal(
        gate_up_scales,
        gate_up_scales[:, :1, :].expand_as(gate_up_scales),
    ):
        raise ValueError("gate/up scale table rows must be identical")
    if not torch.equal(
        down_scales,
        down_scales[:, :1, :].expand_as(down_scales),
    ):
        raise ValueError("down scale table rows must be identical")

    gate_up_grid = gate_up_scales[:, 0, :].reshape(
        experts,
        2,
        hidden_blocks,
        intermediate_blocks,
    )
    down_grid = down_scales[:, 0, :].reshape(
        experts,
        intermediate_blocks,
        hidden_blocks,
    )
    return gate_up_grid, down_grid


class OfficialFP8ExpertReader:
    """Lazy reader for only the routed experts in an official FP8 checkpoint."""

    def __init__(self, checkpoint):
        from st_reader import build_weight_map

        self.checkpoint = checkpoint
        self.weight_map = build_weight_map(checkpoint)
        sample = next(
            (
                key
                for key in self.weight_map
                if "layers.0.input_layernorm.weight" in key
            ),
            None,
        )
        if sample is None:
            raise KeyError(
                "official FP8 checkpoint has no layer-0 input layernorm key"
            )
        self.prefix = sample[: sample.index("layers.0.input_layernorm.weight")]
        self._handles = {}

    def _get(self, key, expected_dtype):
        from st_reader import SafeReader

        if key not in self.weight_map:
            raise KeyError(f"official FP8 checkpoint is missing {key!r}")
        filename = self.weight_map[key]
        if filename not in self._handles:
            self._handles[filename] = SafeReader(
                os.path.join(self.checkpoint, filename)
            )
        reader = self._handles[filename]
        actual_dtype = reader.get_dtype(key)
        if actual_dtype != expected_dtype:
            raise TypeError(
                f"{key} must have safetensors dtype {expected_dtype}, "
                f"found {actual_dtype}"
            )
        return reader.get_tensor(key)

    def load_layer(
        self,
        layer,
        expert_start,
        expert_end,
        mode,
        fp8_impl="dual",
    ):
        """Return one layer in the NKI layouts plus aggregate quality metrics."""
        if mode == "fp8" and fp8_impl not in (
            "dual",
            "block_pow2",
            "block_pow2_coalesced",
        ):
            raise ValueError(
                "fp8_impl must be 'dual', 'block_pow2', or "
                "'block_pow2_coalesced' for block FP8"
            )
        coalesced = (
            mode == "fp8" and fp8_impl == "block_pow2_coalesced"
        )
        gate_up_weights = []
        gate_up_residuals = []
        gate_up_scales = []
        down_weights = []
        down_residuals = []
        down_scales = []
        aggregate = QuantizationStats()
        layer_prefix = f"{self.prefix}layers.{layer}.mlp.experts."

        for expert in range(expert_start, expert_end):
            expert_prefix = f"{layer_prefix}{expert}."
            projections = []
            residual_projections = []
            projection_scales = []
            for projection in ("gate_proj", "up_proj"):
                weight_key = expert_prefix + projection + ".weight"
                raw = self._get(weight_key, "F8_E4M3")
                source_scales = self._get(
                    weight_key + "_scale_inv", "BF16"
                )
                if mode == "fp8":
                    if fp8_impl == "dual":
                        converted, residual, scales, stats = (
                            split_official_fp8(raw, source_scales)
                        )
                    else:
                        converted, scales, stats = (
                            requantize_official_fp8_pow2(
                                raw, source_scales
                            )
                        )
                else:
                    converted, scales, stats = requantize_official_fp8(
                        raw, source_scales, mode
                    )
                quality_mode = (
                    f"{mode}/{fp8_impl}" if mode == "fp8" else mode
                )
                self._check_quality(weight_key, quality_mode, stats)
                aggregate.merge(stats)
                projections.append(converted.transpose(0, 1).contiguous())
                if mode == "fp8" and fp8_impl == "dual":
                    residual_projections.append(
                        residual.transpose(0, 1).contiguous()
                    )
                projection_scales.append(scales)
            gate_up_weights.append(
                torch.stack(projections, dim=1 if coalesced else 0)
            )
            if mode == "fp8" and fp8_impl == "dual":
                gate_up_residuals.append(
                    torch.stack(residual_projections, dim=0)
                )
            gate_up_scales.append(torch.stack(projection_scales, dim=0))

            weight_key = expert_prefix + "down_proj.weight"
            raw = self._get(weight_key, "F8_E4M3")
            source_scales = self._get(
                weight_key + "_scale_inv", "BF16"
            )
            if mode == "fp8":
                if fp8_impl == "dual":
                    converted, residual, scales, stats = split_official_fp8(
                        raw, source_scales
                    )
                else:
                    converted, scales, stats = (
                        requantize_official_fp8_pow2(
                            raw, source_scales
                        )
                    )
            else:
                converted, scales, stats = requantize_official_fp8(
                    raw, source_scales, mode
                )
            quality_mode = f"{mode}/{fp8_impl}" if mode == "fp8" else mode
            self._check_quality(weight_key, quality_mode, stats)
            aggregate.merge(stats)
            down_weights.append(converted.transpose(0, 1).contiguous())
            if mode == "fp8" and fp8_impl == "dual":
                down_residuals.append(
                    residual.transpose(0, 1).contiguous()
                )
            down_scales.append(scales)

        gate_up_scale_grid = torch.stack(gate_up_scales, dim=0)
        down_scale_grid = torch.stack(down_scales, dim=0)
        if coalesced:
            gate_up_scale_table, down_scale_table = (
                pack_coalesced_block_scales(
                    gate_up_scale_grid, down_scale_grid
                )
            )
            converted = {
                "w8_gate_up": torch.stack(gate_up_weights, dim=0),
                "w8_gate_up_scale": gate_up_scale_table,
                "w8_down": torch.stack(down_weights, dim=0),
                "w8_down_scale": down_scale_table,
            }
        else:
            converted = {
                "w8_gate_up": torch.stack(gate_up_weights, dim=0),
                # NKI block scales are partition-broadcast operands. Pre-broadcast
                # the grids in HBM to avoid stream shuffles per GEMM tile.
                "w8_gate_up_scale": gate_up_scale_grid.unsqueeze(-1)
                .expand(*gate_up_scale_grid.shape, BLOCK_SIZE)
                .contiguous(),
                "w8_down": torch.stack(down_weights, dim=0),
                "w8_down_scale": down_scale_grid.unsqueeze(-1)
                .expand(*down_scale_grid.shape, BLOCK_SIZE)
                .contiguous(),
            }
        if mode == "fp8" and fp8_impl == "dual":
            converted.update(
                {
                    "w8_gate_up_residual": torch.stack(
                        gate_up_residuals, dim=0
                    ),
                    "w8_down_residual": torch.stack(
                        down_residuals, dim=0
                    ),
                }
            )
        return converted, aggregate

    def load_layer_bf16(self, layer, expert_start, expert_end):
        """Build the exact official-FP8-dequantized BF16 expert reference."""
        gate_up = []
        down = []
        layer_prefix = f"{self.prefix}layers.{layer}.mlp.experts."
        for expert in range(expert_start, expert_end):
            expert_prefix = f"{layer_prefix}{expert}."
            projections = []
            for projection in ("gate_proj", "up_proj"):
                weight_key = expert_prefix + projection + ".weight"
                projections.append(
                    dequantize_official_fp8(
                        self._get(weight_key, "F8_E4M3"),
                        self._get(weight_key + "_scale_inv", "BF16"),
                    ).to(torch.bfloat16)
                )
            gate_up.append(torch.cat(projections, dim=0))
            weight_key = expert_prefix + "down_proj.weight"
            down.append(
                dequantize_official_fp8(
                    self._get(weight_key, "F8_E4M3"),
                    self._get(weight_key + "_scale_inv", "BF16"),
                ).to(torch.bfloat16)
            )
        return {
            "gate_up": torch.stack(gate_up, dim=0),
            "down": torch.stack(down, dim=0),
        }

    def load_layer_row_fp8(
        self,
        layer,
        expert_start,
        expert_end,
        fp8_projections="all",
    ):
        """Build a row-scaled nkilib layout with selected FP8 projections."""
        if fp8_projections not in ROW_FP8_PROJECTION_CHOICES:
            raise ValueError(
                "fp8_projections must be one of "
                + ", ".join(ROW_FP8_PROJECTION_CHOICES)
            )
        gate_up = []
        gate_up_scales = []
        down = []
        down_scales = []
        aggregate = QuantizationStats()
        layer_prefix = f"{self.prefix}layers.{layer}.mlp.experts."

        for expert in range(expert_start, expert_end):
            expert_prefix = f"{layer_prefix}{expert}."
            projections = []
            projection_scales = []
            for projection in ("gate_proj", "up_proj"):
                weight_key = expert_prefix + projection + ".weight"
                raw = self._get(weight_key, "F8_E4M3")
                source_scales = self._get(
                    weight_key + "_scale_inv", "BF16"
                )
                if fp8_projections in ("all", "gate_up"):
                    quantized, scales, stats = requantize_official_fp8_row(
                        raw, source_scales
                    )
                else:
                    quantized, scales, stats = retain_official_fp8_row_bf16(
                        raw, source_scales
                    )
                aggregate.merge(stats)
                projections.append(quantized)
                projection_scales.append(scales)
            gate_up.append(
                torch.stack(projections, dim=0)
                .permute(2, 0, 1)
                .contiguous()
            )
            gate_up_scales.append(
                torch.stack(projection_scales, dim=0)
            )

            weight_key = expert_prefix + "down_proj.weight"
            raw = self._get(weight_key, "F8_E4M3")
            source_scales = self._get(
                weight_key + "_scale_inv", "BF16"
            )
            if fp8_projections in ("all", "down"):
                quantized, scales, stats = requantize_official_fp8_row(
                    raw, source_scales
                )
            else:
                quantized, scales, stats = retain_official_fp8_row_bf16(
                    raw, source_scales
                )
            aggregate.merge(stats)
            down.append(quantized.transpose(0, 1).contiguous())
            down_scales.append(scales)

        return {
            "row_gate_up": torch.stack(gate_up, dim=0),
            "row_gate_up_scale": torch.stack(
                gate_up_scales, dim=0
            ),
            "row_down": torch.stack(down, dim=0),
            "row_down_scale": torch.stack(down_scales, dim=0),
        }, aggregate

    @staticmethod
    def _check_quality(key, mode, stats):
        if stats.cosine < 0.9995 or stats.normalized_rmse > 0.035:
            raise ValueError(
                f"{key} failed the {mode} quality gate: "
                f"cosine={stats.cosine:.7f}, "
                f"nrmse={stats.normalized_rmse:.5%}"
            )

    def close(self):
        for reader in self._handles.values():
            reader.close()
        self._handles.clear()


def dequantize_w8(weight, scales, mode, block_size=BLOCK_SIZE):
    """Dequantize a converted legacy-FP8/INT8 weight for CPU validation."""
    if scales.ndim == weight.ndim + 1 and scales.shape[-1] == block_size:
        scales = scales[..., 0]
    if mode == "fp8":
        decoded = decode_e4m3fn(weight)
    elif mode == "int8":
        decoded = weight.float()
    else:
        raise ValueError(f"unknown W8 mode {mode!r}")
    return decoded * expand_block_scales(scales.float(), weight.shape, block_size)


def dequantize_fp8_planes(base, residual, scales, block_size=BLOCK_SIZE):
    """Reconstruct an exact two-plane legacy-FP8 weight."""
    if base.shape != residual.shape:
        raise ValueError("FP8 base and residual plane shapes must match")
    if scales.ndim == base.ndim + 1 and scales.shape[-1] == block_size:
        scales = scales[..., 0]
    decoded = decode_legacy_e4m3(base) + decode_legacy_e4m3(residual)
    return decoded * expand_block_scales(
        scales.float(), base.shape, block_size
    )


def fused_moe_row_fp8_cpu(
    hidden,
    gate_up,
    down,
    gate_up_scales,
    down_scales,
    affinities,
    *,
    activation_bf16=True,
    affinity_bf16=True,
):
    """CPU reference for nkilib's row-scaled all-expert FP8 layout."""
    experts, hidden_size, projections, intermediate = gate_up.shape
    if projections != 2:
        raise ValueError("gate_up must contain gate and up projections")
    if tuple(down.shape) != (experts, intermediate, hidden_size):
        raise ValueError("gate_up/down row-FP8 layouts are inconsistent")
    if tuple(gate_up_scales.shape) != (experts, 2, intermediate):
        raise ValueError("gate/up row scale shape does not match weights")
    if tuple(down_scales.shape) != (experts, hidden_size):
        raise ValueError("down row scale shape does not match weights")
    if tuple(affinities.shape) != (hidden.shape[0], experts):
        raise ValueError("affinity shape does not match tokens and experts")

    hidden_f32 = hidden.float()
    output = torch.zeros(
        hidden.shape[0], hidden_size, dtype=torch.float32, device=hidden.device
    )

    def row_weight(weight, scales):
        if weight.dtype == torch.int8:
            values = decode_legacy_e4m3(weight)
        elif weight.dtype == torch.bfloat16:
            values = weight.float()
        else:
            raise TypeError(
                "row-scaled MoE weights must use INT8 FP8 storage or BF16, "
                f"got {weight.dtype}"
            )
        return values * scales.float().unsqueeze(0)

    for expert in range(experts):
        gate_weight = row_weight(
            gate_up[expert, :, 0, :],
            gate_up_scales[expert, 0],
        )
        up_weight = row_weight(
            gate_up[expert, :, 1, :],
            gate_up_scales[expert, 1],
        )
        down_weight = row_weight(
            down[expert],
            down_scales[expert],
        )
        activated = F.silu(hidden_f32 @ gate_weight)
        activated *= hidden_f32 @ up_weight
        if activation_bf16:
            activated = activated.to(torch.bfloat16).float()
        expert_output = activated @ down_weight
        affinity = affinities[:, expert : expert + 1].float()
        if affinity_bf16:
            affinity = affinity.to(torch.bfloat16).float()
        output += expert_output * affinity
    return output


def build_local_affinities(top_ids, top_weights, expert_start, local_experts):
    """Sum global top-k routes into a dense local affinity matrix.

    `scatter_add_` deliberately preserves duplicate-route semantics and makes
    the result independent of top-k slot ordering.
    """
    local = top_ids.to(torch.long) - expert_start
    on_rank = (local >= 0) & (local < local_experts)
    safe = local.clamp(0, local_experts - 1)
    affinities = torch.zeros(
        top_ids.shape[0],
        local_experts,
        dtype=top_weights.dtype,
        device=top_weights.device,
    )
    affinities.scatter_add_(1, safe, top_weights * on_rank.to(top_weights.dtype))
    return affinities


def fused_moe_cpu(
    hidden,
    gate_up,
    down,
    gate_up_scales,
    down_scales,
    affinities,
    mode,
    rounding=(),
    gate_up_residual=None,
    down_residual=None,
):
    """CPU reference for the local routed part of the fused all-expert kernel.

    Kernel layouts are gate_up `[E,2,H,I]`, down `[E,I,H]`, with pre-broadcast
    scales `[E,2,I/128,H/128,128]` and `[E,H/128,I/128,128]`.
    """
    rounding = frozenset(rounding)
    unknown = rounding - MOE_ROUNDING_STAGES
    if unknown:
        raise ValueError(
            "unknown MoE rounding stages: " + ", ".join(sorted(unknown))
        )
    experts, _, hidden_size, intermediate = gate_up.shape
    if tuple(down.shape) != (experts, intermediate, hidden_size):
        raise ValueError("gate_up/down kernel layouts are inconsistent")
    if (gate_up_residual is None) != (down_residual is None):
        raise ValueError("both FP8 residual planes must be provided together")
    if gate_up_residual is not None:
        if mode != "fp8":
            raise ValueError("residual planes are supported only for FP8")
        if gate_up_residual.shape != gate_up.shape:
            raise ValueError("gate/up residual plane shape does not match")
        if down_residual.shape != down.shape:
            raise ValueError("down residual plane shape does not match")
    output = torch.zeros(
        hidden.shape[0], hidden_size, dtype=torch.float32, device=hidden.device
    )
    for expert in range(experts):
        if gate_up_residual is None:
            gate = dequantize_w8(
                gate_up[expert, 0].transpose(0, 1),
                gate_up_scales[expert, 0],
                mode,
            )
            up = dequantize_w8(
                gate_up[expert, 1].transpose(0, 1),
                gate_up_scales[expert, 1],
                mode,
            )
            down_weight = dequantize_w8(
                down[expert].transpose(0, 1),
                down_scales[expert],
                mode,
            )
        else:
            gate = dequantize_fp8_planes(
                gate_up[expert, 0].transpose(0, 1),
                gate_up_residual[expert, 0].transpose(0, 1),
                gate_up_scales[expert, 0],
            )
            up = dequantize_fp8_planes(
                gate_up[expert, 1].transpose(0, 1),
                gate_up_residual[expert, 1].transpose(0, 1),
                gate_up_scales[expert, 1],
            )
            down_weight = dequantize_fp8_planes(
                down[expert].transpose(0, 1),
                down_residual[expert].transpose(0, 1),
                down_scales[expert],
            )
        activated = F.silu(F.linear(hidden.float(), gate)) * F.linear(
            hidden.float(), up
        )
        if "activation_bf16" in rounding:
            activated = activated.to(torch.bfloat16).float()
        expert_output = F.linear(activated, down_weight)
        affinity = affinities[:, expert : expert + 1].float()
        if "affinity_bf16" in rounding:
            affinity = affinity.to(torch.bfloat16).float()
        output += expert_output * affinity
    if "local_output_bf16" in rounding:
        output = output.to(torch.bfloat16).float()
    return output


def fused_moe_block_coalesced_cpu(
    hidden,
    gate_up,
    down,
    gate_up_scales,
    down_scales,
    affinities,
    rounding=(),
):
    """CPU reference for `[E,H,2,I]` coalesced block-power-of-two FP8."""
    rounding = frozenset(rounding)
    unknown = rounding - MOE_ROUNDING_STAGES
    if unknown:
        raise ValueError(
            "unknown MoE rounding stages: " + ", ".join(sorted(unknown))
        )
    experts, hidden_size, projections, intermediate = gate_up.shape
    if projections != 2:
        raise ValueError("gate_up must contain gate and up projections")
    if tuple(down.shape) != (experts, intermediate, hidden_size):
        raise ValueError("coalesced gate_up/down layouts are inconsistent")
    if tuple(affinities.shape) != (hidden.shape[0], experts):
        raise ValueError("affinity shape does not match tokens and experts")
    gate_up_grid, down_grid = unpack_coalesced_block_scales(
        gate_up_scales,
        down_scales,
        hidden_size,
        intermediate,
    )

    output = torch.zeros(
        hidden.shape[0],
        hidden_size,
        dtype=torch.float32,
        device=hidden.device,
    )
    for expert in range(experts):
        gate = dequantize_w8(
            gate_up[expert, :, 0, :],
            gate_up_grid[expert, 0],
            "fp8",
        ).transpose(0, 1)
        up = dequantize_w8(
            gate_up[expert, :, 1, :],
            gate_up_grid[expert, 1],
            "fp8",
        ).transpose(0, 1)
        down_weight = dequantize_w8(
            down[expert],
            down_grid[expert],
            "fp8",
        ).transpose(0, 1)
        activated = F.silu(F.linear(hidden.float(), gate)) * F.linear(
            hidden.float(), up
        )
        if "activation_bf16" in rounding:
            activated = activated.to(torch.bfloat16).float()
        affinity = affinities[:, expert : expert + 1].float()
        if "affinity_bf16" in rounding:
            affinity = affinity.to(torch.bfloat16).float()
        output += F.linear(activated, down_weight) * affinity
    if "local_output_bf16" in rounding:
        output = output.to(torch.bfloat16).float()
    return output
