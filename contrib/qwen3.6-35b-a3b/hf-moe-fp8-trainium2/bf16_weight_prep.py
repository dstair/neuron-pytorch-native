"""Direct BF16-to-FP8 weight preparation for the Qwen routed-expert kernels.

This module never reads or reconstructs an FP8 checkpoint. It quantizes the
BF16 expert tensors directly into Trn2 legacy-E4M3 bytes stored as ``torch.int8``.

Two layouts are provided:

* ``row``: one FP32 scale per output row, for nkilib ``moe_tkg`` row
  quantization and the BS=1 profile.
* ``output_block``: one BF16 scale per 128 output rows, repeated across the
  contraction-block axis and packed for
  ``moe_w8::fused_fp8_block_coalesced_ob``.

The BF16 checkpoint layout is:

* gate/up: ``[experts, 2 * intermediate, hidden]``
* down: ``[experts, hidden, intermediate]``

The returned kernel weight layout is:

* gate/up: ``[experts, hidden, 2, intermediate]``
* down: ``[experts, intermediate, hidden]``
"""

import argparse
import os
from dataclasses import dataclass

import torch

from moe_w8 import (
    BLOCK_SIZE,
    LEGACY_E4M3_MAX,
    QuantizationStats,
    assert_output_block_scale_grid,
    decode_legacy_e4m3,
    encode_legacy_e4m3,
    pack_coalesced_block_scales,
)


LAYOUTS = ("row", "output_block")


@dataclass
class BF16QuantizationReport:
    """Per-projection direct-BF16 quantization statistics."""

    gate: QuantizationStats
    up: QuantizationStats
    down: QuantizationStats

    @property
    def aggregate(self):
        result = QuantizationStats()
        result.merge(self.gate)
        result.merge(self.up)
        result.merge(self.down)
        return result

    @property
    def relative_l2(self):
        return {
            "gate": self.gate.normalized_rmse,
            "up": self.up.normalized_rmse,
            "down": self.down.normalized_rmse,
        }

    def assert_quality(self, max_relative_l2=0.035):
        failures = {
            name: error
            for name, error in self.relative_l2.items()
            if error > max_relative_l2
        }
        if failures:
            detail = ", ".join(
                f"{name}={error:.5%}" for name, error in failures.items()
            )
            raise ValueError(
                "direct BF16-to-FP8 quality gate failed "
                f"(maximum {max_relative_l2:.5%}): {detail}"
            )


def _validate_matrix(weight):
    if weight.dtype != torch.bfloat16:
        raise TypeError(
            f"direct FP8 quantization requires BF16 input, got {weight.dtype}"
        )
    if weight.ndim != 2:
        raise ValueError(
            f"weight must be a two-dimensional [output,input] matrix, "
            f"got shape {tuple(weight.shape)}"
        )
    if weight.shape[0] == 0 or weight.shape[1] == 0:
        raise ValueError(f"weight dimensions must be nonzero, got {tuple(weight.shape)}")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("BF16 weight contains non-finite values")


def quantize_bf16_rows(weight):
    """Quantize ``[output,input]`` BF16 weights with one scale per output row.

    Returns legacy-E4M3 bytes as INT8, FP32 scales shaped ``[output]``, and
    quantization statistics against the original BF16 values.
    """
    _validate_matrix(weight)
    source = weight.float()
    scales = (
        source.abs().amax(dim=1).clamp_min(1e-12) / LEGACY_E4M3_MAX
    )
    quantized = encode_legacy_e4m3(source / scales[:, None])
    reconstructed = decode_legacy_e4m3(quantized) * scales[:, None]
    stats = QuantizationStats()
    stats.block_count = weight.shape[0]
    stats.update(source, reconstructed)
    return quantized, scales.contiguous(), stats


def quantize_bf16_output_blocks(weight, block_size=BLOCK_SIZE):
    """Quantize BF16 weights with one scale per output-row block.

    The returned scale grid is ``[output/block, input/block]``. Each row of
    that grid contains one repeated value, making the scale independent of the
    contraction block. This is the precondition for accumulating the complete
    contraction in PSUM and applying the scale once.
    """
    _validate_matrix(weight)
    if block_size != BLOCK_SIZE:
        raise ValueError(
            f"the coalesced kernel requires block_size={BLOCK_SIZE}, "
            f"got {block_size}"
        )
    output_size, input_size = weight.shape
    if output_size % block_size or input_size % block_size:
        raise ValueError(
            "output-block quantization requires both matrix dimensions to be "
            f"divisible by {block_size}, got {tuple(weight.shape)}"
        )

    output_blocks = output_size // block_size
    input_blocks = input_size // block_size
    quantized = torch.empty_like(weight, dtype=torch.int8)
    scale_grid = torch.empty(
        output_blocks,
        input_blocks,
        dtype=torch.bfloat16,
        device=weight.device,
    )
    stats = QuantizationStats()

    for output_block in range(output_blocks):
        row_start = output_block * block_size
        row_end = row_start + block_size
        source = weight[row_start:row_end].float()
        scale = (
            source.abs().amax().clamp_min(1e-12) / LEGACY_E4M3_MAX
        ).to(torch.bfloat16)
        scale_f32 = scale.float()
        encoded = encode_legacy_e4m3(source / scale_f32)
        reconstructed = decode_legacy_e4m3(encoded) * scale_f32
        quantized[row_start:row_end] = encoded
        scale_grid[output_block] = scale
        stats.block_count += input_blocks
        stats.update(source, reconstructed)

    return quantized, scale_grid, stats


def _validate_expert_tensors(gate_up_bf16, down_bf16):
    if gate_up_bf16.dtype != torch.bfloat16:
        raise TypeError(
            f"gate/up experts must be BF16, got {gate_up_bf16.dtype}"
        )
    if down_bf16.dtype != torch.bfloat16:
        raise TypeError(f"down experts must be BF16, got {down_bf16.dtype}")
    if gate_up_bf16.ndim != 3 or down_bf16.ndim != 3:
        raise ValueError(
            "expected gate/up [E,2I,H] and down [E,H,I], got "
            f"{tuple(gate_up_bf16.shape)} and {tuple(down_bf16.shape)}"
        )

    experts, twice_intermediate, hidden = gate_up_bf16.shape
    experts_down, hidden_down, intermediate = down_bf16.shape
    if (
        experts != experts_down
        or hidden != hidden_down
        or twice_intermediate != 2 * intermediate
    ):
        raise ValueError(
            "inconsistent gate/up and down expert shapes: "
            f"{tuple(gate_up_bf16.shape)} and {tuple(down_bf16.shape)}"
        )
    return experts, hidden, intermediate


def prepare_bf16_expert_tensors(
    gate_up_bf16,
    down_bf16,
    *,
    layout="output_block",
    check_quality=True,
):
    """Convert BF16 expert tensors into an existing MoE kernel operand layout.

    ``layout="output_block"`` returns ``w8_*`` operands for
    ``fused_fp8_block_coalesced_ob``. ``layout="row"`` returns ``row_*``
    operands for nkilib ``moe_tkg`` row quantization.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    experts, hidden, intermediate = _validate_expert_tensors(
        gate_up_bf16, down_bf16
    )

    gate_weights = []
    up_weights = []
    gate_scales = []
    up_scales = []
    down_weights = []
    down_scales = []
    gate_stats = QuantizationStats()
    up_stats = QuantizationStats()
    down_stats = QuantizationStats()
    quantize = (
        quantize_bf16_rows
        if layout == "row"
        else quantize_bf16_output_blocks
    )

    for expert in range(experts):
        gate = gate_up_bf16[expert, :intermediate, :]
        up = gate_up_bf16[expert, intermediate:, :]
        down = down_bf16[expert]

        gate_q, gate_s, gate_stat = quantize(gate)
        up_q, up_s, up_stat = quantize(up)
        down_q, down_s, down_stat = quantize(down)
        gate_stats.merge(gate_stat)
        up_stats.merge(up_stat)
        down_stats.merge(down_stat)

        gate_weights.append(gate_q.transpose(0, 1).contiguous())
        up_weights.append(up_q.transpose(0, 1).contiguous())
        down_weights.append(down_q.transpose(0, 1).contiguous())
        gate_scales.append(gate_s)
        up_scales.append(up_s)
        down_scales.append(down_s)

    report = BF16QuantizationReport(
        gate=gate_stats,
        up=up_stats,
        down=down_stats,
    )
    if check_quality:
        report.assert_quality()

    gate_up_weights = torch.stack(
        [
            torch.stack(
                (gate_weights[expert], up_weights[expert]),
                dim=1,
            )
            for expert in range(experts)
        ],
        dim=0,
    )
    down_weight_tensor = torch.stack(down_weights, dim=0)

    if layout == "row":
        return {
            "row_gate_up": gate_up_weights,
            "row_gate_up_scale": torch.stack(
                [
                    torch.stack(
                        (gate_scales[expert], up_scales[expert]),
                        dim=0,
                    )
                    for expert in range(experts)
                ],
                dim=0,
            ).float(),
            "row_down": down_weight_tensor,
            "row_down_scale": torch.stack(down_scales, dim=0).float(),
        }, report

    gate_up_scale_grid = torch.stack(
        [
            torch.stack(
                (gate_scales[expert], up_scales[expert]),
                dim=0,
            )
            for expert in range(experts)
        ],
        dim=0,
    )
    down_scale_grid = torch.stack(down_scales, dim=0)
    assert_output_block_scale_grid(gate_up_scale_grid, down_scale_grid)
    gate_up_scale_table, down_scale_table = pack_coalesced_block_scales(
        gate_up_scale_grid,
        down_scale_grid,
    )
    return {
        "w8_gate_up": gate_up_weights,
        "w8_gate_up_scale": gate_up_scale_table,
        "w8_down": down_weight_tensor,
        "w8_down_scale": down_scale_table,
    }, report


class BF16ExpertReader:
    """Read Qwen BF16 expert tensors and quantize selected experts directly."""

    def __init__(self, checkpoint):
        from st_reader import build_weight_map

        self.checkpoint = checkpoint
        self.weight_map = build_weight_map(checkpoint)
        marker = "layers.0.input_layernorm.weight"
        samples = [
            key
            for key in self.weight_map
            if key.endswith(marker)
            and not key.startswith("mtp.")
            and ".mtp." not in key
        ]
        if len(samples) != 1:
            raise KeyError(
                f"expected exactly one decoder-stack {marker!r} key in the "
                f"BF16 checkpoint, found {samples}"
            )
        self.prefix = samples[0][: -len(marker)]
        self._handles = {}

    def _get(self, key):
        from st_reader import SafeReader

        if key not in self.weight_map:
            raise KeyError(f"BF16 checkpoint is missing {key!r}")
        filename = self.weight_map[key]
        if filename not in self._handles:
            self._handles[filename] = SafeReader(
                os.path.join(self.checkpoint, filename)
            )
        reader = self._handles[filename]
        actual_dtype = reader.get_dtype(key)
        if actual_dtype != "BF16":
            raise TypeError(
                f"{key} must have safetensors dtype BF16, found {actual_dtype}"
            )
        return reader.get_tensor(key)

    def load_layer(
        self,
        layer,
        expert_start,
        expert_end,
        *,
        layout="output_block",
        check_quality=True,
    ):
        """Load experts ``[expert_start, expert_end)`` and quantize from BF16."""
        if expert_start < 0 or expert_end <= expert_start:
            raise ValueError(
                f"invalid expert range [{expert_start}, {expert_end})"
            )
        layer_prefix = f"{self.prefix}layers.{layer}.mlp.experts."
        packed_gate_up_key = layer_prefix + "gate_up_proj"
        packed_down_key = layer_prefix + "down_proj"

        if (
            packed_gate_up_key in self.weight_map
            and packed_down_key in self.weight_map
        ):
            gate_up_all = self._get(packed_gate_up_key)
            down_all = self._get(packed_down_key)
            if expert_end > gate_up_all.shape[0] or expert_end > down_all.shape[0]:
                raise ValueError(
                    f"expert range [{expert_start}, {expert_end}) exceeds "
                    f"checkpoint expert count {gate_up_all.shape[0]}"
                )
            gate_up = gate_up_all[expert_start:expert_end].clone()
            down = down_all[expert_start:expert_end].clone()
        else:
            gate_up_parts = []
            down_parts = []
            for expert in range(expert_start, expert_end):
                expert_prefix = layer_prefix + f"{expert}."
                gate = self._get(expert_prefix + "gate_proj.weight")
                up = self._get(expert_prefix + "up_proj.weight")
                down = self._get(expert_prefix + "down_proj.weight")
                gate_up_parts.append(torch.cat((gate, up), dim=0))
                down_parts.append(down)
            gate_up = torch.stack(gate_up_parts, dim=0)
            down = torch.stack(down_parts, dim=0)

        return prepare_bf16_expert_tensors(
            gate_up,
            down,
            layout=layout,
            check_quality=check_quality,
        )

    def close(self):
        for reader in self._handles.values():
            reader.close()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Qwen routed experts directly from a BF16 checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-start", type=int, default=0)
    parser.add_argument("--expert-end", type=int, default=32)
    parser.add_argument("--layout", choices=LAYOUTS, default="output_block")
    parser.add_argument(
        "--output",
        help="optional torch.save destination for the prepared operands",
    )
    args = parser.parse_args()

    reader = BF16ExpertReader(args.checkpoint)
    try:
        operands, report = reader.load_layer(
            args.layer,
            args.expert_start,
            args.expert_end,
            layout=args.layout,
        )
    finally:
        reader.close()

    print(f"layout={args.layout}")
    for name, tensor in operands.items():
        print(f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
    for name, error in report.relative_l2.items():
        print(f"{name} relative_l2={error:.5%}")
    if args.output:
        torch.save(operands, args.output)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
