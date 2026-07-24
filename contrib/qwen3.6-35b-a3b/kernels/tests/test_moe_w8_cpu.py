#!/usr/bin/env python3
"""CPU tests for official-FP8 conversion and fused-W8 routing semantics."""

import json
import os
import struct
import sys

import pytest
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from moe_w8 import (  # noqa: E402
    build_local_affinities,
    decode_e4m3fn,
    decode_legacy_e4m3,
    dequantize_fp8_planes,
    dequantize_official_fp8,
    dequantize_w8,
    encode_legacy_e4m3,
    expand_block_scales,
    fused_moe_block_coalesced_cpu,
    fused_moe_cpu,
    fused_moe_row_fp8_cpu,
    OfficialFP8ExpertReader,
    pack_coalesced_block_scales,
    QuantizationStats,
    requantize_official_fp8,
    requantize_official_fp8_pow2,
    requantize_official_fp8_row,
    retain_official_fp8,
    split_official_fp8,
    unpack_coalesced_block_scales,
    validate_legacy_e4m3_bytes,
)
from st_reader import SafeReader  # noqa: E402
import static_decode_35b as S  # noqa: E402


def _write_safetensor(path, name, dtype, shape, payload):
    header = {
        name: {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [0, len(payload)],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(payload)


def _write_tensors(path, tensors):
    header = {}
    payload = bytearray()
    for name, dtype, shape, raw in tensors:
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(payload)


def _bf16_bytes(tensor):
    return tensor.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().tobytes()


def test_safe_reader_returns_raw_fp8_bytes(tmp_path):
    path = tmp_path / "weights.safetensors"
    payload = bytes([0x00, 0x38, 0x78, 0xFE])
    _write_safetensor(path, "expert.weight", "F8_E4M3", (2, 2), payload)
    reader = SafeReader(path)
    try:
        tensor = reader.get_tensor("expert.weight")
        assert reader.get_dtype("expert.weight") == "F8_E4M3"
        assert tensor.dtype == torch.uint8
        assert tensor.tolist() == [[0x00, 0x38], [0x78, 0xFE]]
    finally:
        reader.close()


def test_exact_e4m3fn_decoding_including_ocp_only_codes():
    raw = torch.tensor(
        [0x00, 0x01, 0x07, 0x08, 0x38, 0x77, 0x78, 0x7E, 0xB8],
        dtype=torch.uint8,
    )
    expected = torch.tensor(
        [0.0, 2.0**-9, 7.0 * 2.0**-9, 2.0**-6, 1.0, 240.0, 256.0, 448.0, -1.0]
    )
    torch.testing.assert_close(decode_e4m3fn(raw), expected, rtol=0, atol=0)


def test_e4m3fn_nan_and_legacy_unsafe_codes_are_rejected():
    with pytest.raises(ValueError, match="NaN encoding"):
        decode_e4m3fn(torch.tensor([0x7F], dtype=torch.uint8))
    with pytest.raises(ValueError, match="unsafe encoding"):
        validate_legacy_e4m3_bytes(torch.tensor([0x78], dtype=torch.uint8))
    with pytest.raises(ValueError, match="non-finite"):
        encode_legacy_e4m3(torch.tensor([float("inf")]))


def test_legacy_encoding_is_finite_and_rounds_ties_to_even():
    values = torch.tensor(
        [0.0, 1.0, -1.0, 240.0, 1000.0, 1.0625, 1.1875],
        dtype=torch.float32,
    )
    encoded = encode_legacy_e4m3(values)
    validate_legacy_e4m3_bytes(encoded)
    expected = torch.tensor([0.0, 1.0, -1.0, 240.0, 240.0, 1.0, 1.25])
    torch.testing.assert_close(decode_legacy_e4m3(encoded), expected, rtol=0, atol=0)


def test_two_legacy_planes_exactly_reconstruct_every_finite_e4m3fn_code():
    codes = torch.arange(256, dtype=torch.uint8)
    finite = (codes & 0x7F) != 0x7F
    raw = codes[finite].reshape(2, 127)
    scales = torch.ones(1, 1, dtype=torch.bfloat16)
    base, residual, retained_scales, stats = split_official_fp8(
        raw, scales, block_size=128
    )

    validate_legacy_e4m3_bytes(base)
    validate_legacy_e4m3_bytes(residual)
    reconstructed = (
        decode_legacy_e4m3(base) + decode_legacy_e4m3(residual)
    )
    torch.testing.assert_close(
        reconstructed, decode_e4m3fn(raw), rtol=0, atol=0
    )
    assert torch.equal(retained_scales, scales)
    assert stats.cosine == 1.0
    assert stats.normalized_rmse == 0.0
    assert int((residual.view(torch.uint8) != 0).sum()) == 14


def test_pow2_conversion_handles_every_finite_e4m3fn_code():
    codes = torch.arange(256, dtype=torch.uint8)
    finite = (codes & 0x7F) != 0x7F
    raw = codes[finite].reshape(2, 127)
    scales = torch.ones(1, 1, dtype=torch.bfloat16)

    converted, converted_scales, stats = requantize_official_fp8_pow2(
        raw, scales, block_size=128
    )
    reconstructed = dequantize_w8(
        converted, converted_scales, "fp8", block_size=128
    )
    expected = dequantize_official_fp8(raw, scales, block_size=128)

    validate_legacy_e4m3_bytes(converted)
    assert torch.equal(
        converted_scales, torch.full_like(converted_scales, 2.0)
    )
    assert stats.block_count == 1
    assert stats.shifted_block_count == 1
    assert stats.clipped_count == 0
    assert stats.exact_fraction > 0.90
    assert stats.cosine >= 0.9995
    assert stats.normalized_rmse <= 0.035
    high = (raw & 0x7F) >= 0x78
    torch.testing.assert_close(
        reconstructed[high], expected[high], rtol=0, atol=0
    )


def test_pow2_conversion_only_shifts_blocks_with_extended_codes():
    raw = torch.tensor(
        [
            [0x38, 0x40, 0x78, 0x01],
            [0xB8, 0xC0, 0xFE, 0x81],
        ],
        dtype=torch.uint8,
    )
    scales = torch.tensor([[0.25, 0.5]], dtype=torch.bfloat16)

    converted, converted_scales, stats = requantize_official_fp8_pow2(
        raw, scales, block_size=2
    )
    reconstructed = dequantize_w8(
        converted, converted_scales, "fp8", block_size=2
    )
    expected = dequantize_official_fp8(raw, scales, block_size=2)

    assert torch.equal(converted[:, :2].view(torch.uint8), raw[:, :2])
    assert converted_scales.tolist() == [[0.25, 1.0]]
    assert stats.block_count == 2
    assert stats.shifted_block_count == 1
    torch.testing.assert_close(
        reconstructed[:, :2], expected[:, :2], rtol=0, atol=0
    )
    torch.testing.assert_close(
        reconstructed[0, 2:3], expected[0, 2:3], rtol=0, atol=0
    )
    torch.testing.assert_close(
        reconstructed[1, 2:3], expected[1, 2:3], rtol=0, atol=0
    )


def test_pow2_conversion_rejects_nan_and_bf16_scale_overflow():
    scale = torch.ones(1, 1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="NaN encoding"):
        requantize_official_fp8_pow2(
            torch.tensor([[0x7F]], dtype=torch.uint8),
            scale,
            block_size=1,
        )

    max_bf16 = torch.tensor(
        [[torch.finfo(torch.bfloat16).max]], dtype=torch.bfloat16
    )
    with pytest.raises(ValueError, match="exactly representable"):
        requantize_official_fp8_pow2(
            torch.tensor([[0x78]], dtype=torch.uint8),
            max_bf16,
            block_size=1,
        )


@pytest.mark.parametrize("mode", ["fp8", "int8"])
def test_block_scales_and_requantization(mode):
    block_size = 2
    source_values = torch.tensor(
        [
            [1.0, 2.0, 16.0, 32.0],
            [3.0, 4.0, 48.0, 64.0],
            [0.5, 1.0, 4.0, 8.0],
            [1.5, 2.0, 12.0, 16.0],
        ]
    )
    source_bytes = encode_legacy_e4m3(source_values)
    source_scales = torch.tensor(
        [[1.0, 2.0], [0.5, 4.0]], dtype=torch.bfloat16
    )
    official = dequantize_official_fp8(
        source_bytes.view(torch.uint8), source_scales, block_size
    )
    converted, scales, stats = requantize_official_fp8(
        source_bytes.view(torch.uint8), source_scales, mode, block_size
    )
    reconstructed = dequantize_w8(converted, scales, mode, block_size)
    assert scales.shape == (2, 2)
    assert stats.cosine >= 0.9995
    assert stats.normalized_rmse <= 0.035
    torch.testing.assert_close(
        reconstructed, official, rtol=0.08 if mode == "fp8" else 0.02, atol=0.05
    )


def test_expand_block_scales_rejects_wrong_grid():
    with pytest.raises(ValueError, match="does not match"):
        expand_block_scales(torch.ones(1, 2), (4, 4), block_size=2)


def test_row_fp8_requantization_uses_one_scale_per_output():
    source = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [16.0, 32.0, 48.0, 64.0]]
    )
    raw = encode_legacy_e4m3(source).view(torch.uint8)
    source_scales = torch.ones(1, 1, dtype=torch.bfloat16)
    quantized, scales, stats = requantize_official_fp8_row(
        raw, source_scales
    )
    reconstructed = decode_legacy_e4m3(quantized) * scales[:, None]

    assert scales.shape == (2,)
    assert stats.cosine >= 0.9995
    assert stats.normalized_rmse <= 0.035
    torch.testing.assert_close(reconstructed, source, rtol=0.04, atol=0.01)


def _write_official_expert_checkpoint(tmp_path):
    prefix = "model.language_model."
    tensors = [
        (
            prefix + "layers.0.input_layernorm.weight",
            "BF16",
            (128,),
            _bf16_bytes(torch.ones(128)),
        ),
        (
            prefix + "layers.0.self_attn.q_proj.weight",
            "F32",
            (1,),
            torch.ones(1).numpy().tobytes(),
        ),
    ]
    expected = {}
    for expert in range(2):
        ep = prefix + f"layers.0.mlp.experts.{expert}."
        for projection_index, projection in enumerate(
            ("gate_proj", "up_proj", "down_proj")
        ):
            shape = (256, 128) if projection == "down_proj" else (128, 256)
            scale_shape = (2, 1) if projection == "down_proj" else (1, 2)
            # 0x78 is finite E4M3FN 256 but unsafe to bitcast as legacy E4M3.
            raw = torch.empty(shape, dtype=torch.uint8)
            scales = torch.empty(scale_shape, dtype=torch.bfloat16)
            for row_block in range(scale_shape[0]):
                for col_block in range(scale_shape[1]):
                    raw[
                        row_block * 128 : (row_block + 1) * 128,
                        col_block * 128 : (col_block + 1) * 128,
                    ] = 0x78 - expert - projection_index - row_block - col_block
                    scales[row_block, col_block] = (
                        0.01
                        + expert * 0.01
                        + projection_index * 0.002
                        + row_block * 0.003
                        + col_block * 0.004
                    )
            tensors.append(
                (ep + projection + ".weight", "F8_E4M3", raw.shape, bytes(raw.flatten()))
            )
            tensors.append(
                (
                    ep + projection + ".weight_scale_inv",
                    "BF16",
                    scales.shape,
                    _bf16_bytes(scales),
                )
            )
            expected[expert, projection] = dequantize_official_fp8(raw, scales)
    _write_tensors(tmp_path / "model.safetensors", tensors)
    return expected


@pytest.mark.parametrize("mode", ["fp8", "int8"])
def test_official_reader_loads_only_expert_weights_and_scales(tmp_path, mode):
    expected = _write_official_expert_checkpoint(tmp_path)

    reader = OfficialFP8ExpertReader(tmp_path)
    try:
        converted, stats = reader.load_layer(0, 0, 2, mode)
    finally:
        reader.close()
    assert converted["w8_gate_up"].shape == (2, 2, 256, 128)
    assert converted["w8_gate_up_scale"].shape == (2, 2, 1, 2, 128)
    assert converted["w8_down"].shape == (2, 128, 256)
    assert converted["w8_down_scale"].shape == (2, 2, 1, 128)
    if mode == "fp8":
        assert converted["w8_gate_up_residual"].shape == (
            2, 2, 256, 128
        )
        assert converted["w8_down_residual"].shape == (2, 128, 256)
        validate_legacy_e4m3_bytes(converted["w8_gate_up"])
        validate_legacy_e4m3_bytes(converted["w8_gate_up_residual"])
        # The official-only 0x78 code is represented by base 0x77 plus 16.
        assert bool(
            converted["w8_gate_up"].view(torch.uint8).eq(0x77).any()
        )
        assert bool(
            converted["w8_gate_up_residual"].view(torch.uint8).ne(0).any()
        )
        reconstructed = dequantize_fp8_planes(
            converted["w8_gate_up"][0, 0].transpose(0, 1),
            converted["w8_gate_up_residual"][0, 0].transpose(0, 1),
            converted["w8_gate_up_scale"][0, 0],
        )
        torch.testing.assert_close(
            reconstructed, expected[0, "gate_proj"], rtol=0, atol=0
        )
    assert stats.cosine >= 0.9995
    assert stats.normalized_rmse <= 0.035


def test_official_reader_builds_single_plane_pow2_fp8(tmp_path):
    expected = _write_official_expert_checkpoint(tmp_path)
    reader = OfficialFP8ExpertReader(tmp_path)
    try:
        converted, stats = reader.load_layer(
            0, 0, 2, "fp8", fp8_impl="block_pow2"
        )
    finally:
        reader.close()

    assert "w8_gate_up_residual" not in converted
    assert "w8_down_residual" not in converted
    validate_legacy_e4m3_bytes(converted["w8_gate_up"])
    validate_legacy_e4m3_bytes(converted["w8_down"])
    reconstructed = dequantize_w8(
        converted["w8_gate_up"][0, 0].transpose(0, 1),
        converted["w8_gate_up_scale"][0, 0],
        "fp8",
    )
    torch.testing.assert_close(
        reconstructed, expected[0, "gate_proj"], rtol=0, atol=0
    )
    assert stats.shifted_block_count > 0
    assert stats.clipped_count == 0
    assert stats.cosine >= 0.9995
    assert stats.normalized_rmse <= 0.035


def test_coalesced_scale_packing_order_and_round_trip():
    gate_up_grid = torch.arange(
        1, 1 + 2 * 2 * 3, dtype=torch.float32
    ).reshape(1, 2, 2, 3).to(torch.bfloat16)
    down_grid = torch.arange(
        21, 21 + 3 * 2, dtype=torch.float32
    ).reshape(1, 3, 2).to(torch.bfloat16)

    gate_up_table, down_table = pack_coalesced_block_scales(
        gate_up_grid, down_grid
    )

    expected_gate_up = torch.tensor(
        [
            1,
            4,
            2,
            5,
            3,
            6,
            7,
            10,
            8,
            11,
            9,
            12,
        ],
        dtype=torch.bfloat16,
    )
    expected_down = torch.tensor(
        [21, 23, 25, 22, 24, 26],
        dtype=torch.bfloat16,
    )
    assert gate_up_table.shape == (1, 128, 12)
    assert down_table.shape == (1, 128, 6)
    assert torch.equal(gate_up_table[0, 0], expected_gate_up)
    assert torch.equal(down_table[0, 0], expected_down)
    assert torch.equal(
        gate_up_table, gate_up_table[:, :1].expand_as(gate_up_table)
    )
    assert torch.equal(
        down_table, down_table[:, :1].expand_as(down_table)
    )

    gate_up_nki, down_nki = unpack_coalesced_block_scales(
        gate_up_table,
        down_table,
        hidden_size=3 * 128,
        intermediate_size=2 * 128,
    )
    assert torch.equal(
        gate_up_nki,
        gate_up_grid.permute(0, 1, 3, 2),
    )
    assert torch.equal(
        down_nki,
        down_grid.transpose(1, 2),
    )


def test_coalesced_scale_unpack_rejects_non_repeated_rows():
    gate_up = torch.ones(1, 128, 2, dtype=torch.bfloat16)
    down = torch.ones(1, 128, 1, dtype=torch.bfloat16)
    gate_up[0, 17, 0] = 2
    with pytest.raises(ValueError, match="rows must be identical"):
        unpack_coalesced_block_scales(
            gate_up,
            down,
            hidden_size=128,
            intermediate_size=128,
        )


def test_official_reader_builds_coalesced_pow2_layout(tmp_path):
    expected = _write_official_expert_checkpoint(tmp_path)
    reader = OfficialFP8ExpertReader(tmp_path)
    try:
        converted, stats = reader.load_layer(
            0,
            0,
            2,
            "fp8",
            fp8_impl="block_pow2_coalesced",
        )
    finally:
        reader.close()

    assert converted["w8_gate_up"].shape == (2, 256, 2, 128)
    assert converted["w8_gate_up_scale"].shape == (2, 128, 4)
    assert converted["w8_down"].shape == (2, 128, 256)
    assert converted["w8_down_scale"].shape == (2, 128, 2)
    validate_legacy_e4m3_bytes(converted["w8_gate_up"])
    validate_legacy_e4m3_bytes(converted["w8_down"])

    gate_up_grid, down_grid = unpack_coalesced_block_scales(
        converted["w8_gate_up_scale"],
        converted["w8_down_scale"],
        hidden_size=256,
        intermediate_size=128,
    )
    gate = dequantize_w8(
        converted["w8_gate_up"][0, :, 0, :],
        gate_up_grid[0, 0],
        "fp8",
    ).transpose(0, 1)
    down = dequantize_w8(
        converted["w8_down"][0],
        down_grid[0],
        "fp8",
    ).transpose(0, 1)
    torch.testing.assert_close(
        gate, expected[0, "gate_proj"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        down, expected[0, "down_proj"], rtol=0, atol=0
    )
    assert stats.shifted_block_count > 0
    assert stats.clipped_count == 0


def test_retain_official_fp8_is_exact_and_validates_scales():
    raw = torch.tensor([[0x38, 0x78], [0xB8, 0xFE]], dtype=torch.uint8)
    scales = torch.tensor([[0.25]], dtype=torch.bfloat16)
    stored, retained_scales, stats = retain_official_fp8(
        raw, scales, block_size=2
    )
    assert torch.equal(stored.view(torch.uint8), raw)
    assert torch.equal(retained_scales, scales)
    assert stats.cosine == 1.0
    assert stats.normalized_rmse == 0.0


def test_two_plane_dequantization_preserves_block_scales():
    raw = torch.tensor(
        [[0x38, 0x78], [0xB8, 0xFE]], dtype=torch.uint8
    )
    scales = torch.tensor([[0.25]], dtype=torch.bfloat16)
    base, residual, retained_scales, _ = split_official_fp8(
        raw, scales, block_size=2
    )
    reconstructed = dequantize_fp8_planes(
        base, residual, retained_scales, block_size=2
    )
    expected = dequantize_official_fp8(raw, scales, block_size=2)
    torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)


def test_official_reader_builds_exact_bf16_reference_layout(tmp_path):
    expected = _write_official_expert_checkpoint(tmp_path)
    reader = OfficialFP8ExpertReader(tmp_path)
    try:
        loaded = reader.load_layer_bf16(0, 0, 2)
    finally:
        reader.close()

    expected_gate_up = torch.stack(
        [
            torch.cat(
                [
                    expected[expert, "gate_proj"],
                    expected[expert, "up_proj"],
                ],
                dim=0,
            ).to(torch.bfloat16)
            for expert in range(2)
        ]
    )
    expected_down = torch.stack(
        [
            expected[expert, "down_proj"].to(torch.bfloat16)
            for expert in range(2)
        ]
    )
    assert torch.equal(loaded["gate_up"], expected_gate_up)
    assert torch.equal(loaded["down"], expected_down)


def test_official_reader_builds_row_fp8_nkilib_layout(tmp_path):
    expected = _write_official_expert_checkpoint(tmp_path)
    reader = OfficialFP8ExpertReader(tmp_path)
    try:
        loaded, stats = reader.load_layer_row_fp8(0, 0, 2)
    finally:
        reader.close()

    assert loaded["row_gate_up"].shape == (2, 256, 2, 128)
    assert loaded["row_gate_up_scale"].shape == (2, 2, 128)
    assert loaded["row_down"].shape == (2, 128, 256)
    assert loaded["row_down_scale"].shape == (2, 256)
    gate = (
        decode_legacy_e4m3(
            loaded["row_gate_up"][0, :, 0].transpose(0, 1)
        )
        * loaded["row_gate_up_scale"][0, 0, :, None]
    )
    torch.testing.assert_close(
        gate, expected[0, "gate_proj"], rtol=0.04, atol=0.01
    )
    assert stats.cosine >= 0.9995
    assert stats.normalized_rmse <= 0.035


@pytest.mark.parametrize(
    ("fp8_projections", "gate_up_dtype", "down_dtype"),
    [
        ("all", torch.int8, torch.int8),
        ("gate_up", torch.int8, torch.bfloat16),
        ("down", torch.bfloat16, torch.int8),
        ("none", torch.bfloat16, torch.bfloat16),
    ],
)
def test_official_reader_builds_mixed_row_fp8_layout(
    tmp_path,
    fp8_projections,
    gate_up_dtype,
    down_dtype,
):
    expected = _write_official_expert_checkpoint(tmp_path)
    reader = OfficialFP8ExpertReader(tmp_path)
    try:
        loaded, stats = reader.load_layer_row_fp8(
            0, 0, 2, fp8_projections=fp8_projections
        )
    finally:
        reader.close()

    assert loaded["row_gate_up"].dtype == gate_up_dtype
    assert loaded["row_down"].dtype == down_dtype
    if gate_up_dtype == torch.bfloat16:
        expected_gate_up = torch.stack(
            [
                torch.stack(
                    [
                        expected[expert, "gate_proj"],
                        expected[expert, "up_proj"],
                    ]
                )
                .permute(2, 0, 1)
                .to(torch.bfloat16)
                for expert in range(2)
            ]
        )
        assert torch.equal(loaded["row_gate_up"], expected_gate_up)
        assert torch.equal(
            loaded["row_gate_up_scale"],
            torch.ones_like(loaded["row_gate_up_scale"]),
        )
    if down_dtype == torch.bfloat16:
        expected_down = torch.stack(
            [
                expected[expert, "down_proj"]
                .transpose(0, 1)
                .to(torch.bfloat16)
                for expert in range(2)
            ]
        )
        assert torch.equal(loaded["row_down"], expected_down)
        assert torch.equal(
            loaded["row_down_scale"],
            torch.ones_like(loaded["row_down_scale"]),
        )
    assert stats.cosine >= 0.9995
    assert stats.normalized_rmse <= 0.035


def test_official_reader_rejects_unknown_row_fp8_projection_mode(tmp_path):
    _write_official_expert_checkpoint(tmp_path)
    reader = OfficialFP8ExpertReader(tmp_path)
    try:
        with pytest.raises(ValueError, match="fp8_projections"):
            reader.load_layer_row_fp8(
                0, 0, 1, fp8_projections="gate_only"
            )
    finally:
        reader.close()


def test_layer_range_loads_row_fp8_only_for_selected_layers(monkeypatch):
    calls = []
    row_weights = {"row_gate_up": object()}
    bf16_weights = {"gate_up": object(), "down": object()}

    class Reader:
        def load_layer_row_fp8(
            self,
            layer,
            expert_start,
            expert_end,
            fp8_projections,
        ):
            calls.append(
                (
                    "row",
                    layer,
                    expert_start,
                    expert_end,
                    fp8_projections,
                )
            )
            return row_weights, QuantizationStats()

        def load_layer_bf16(self, layer, expert_start, expert_end):
            calls.append(("bf16", layer, expert_start, expert_end))
            return bf16_weights

        def load_layer(self, *args):
            raise AssertionError("dual-plane loader should not be called")

    monkeypatch.setattr(S, "USE_MOE_FUSED_W8_ROW_FP8", True)
    monkeypatch.setattr(S, "MOE_FUSED_W8_FP8_LAYER_START", 1)
    monkeypatch.setattr(S, "MOE_FUSED_W8_FP8_LAYER_LIMIT", 3)
    monkeypatch.setattr(S, "MOE_FUSED_W8_FP8_PROJECTIONS", "down")

    skipped, skipped_stats = S._load_fused_w8_expert_layer(
        Reader(), 0, 8, 16, "fp8", "row"
    )
    selected, selected_stats = S._load_fused_w8_expert_layer(
        Reader(), 1, 8, 16, "fp8", "row"
    )

    assert skipped is bf16_weights
    assert skipped_stats is None
    assert selected is row_weights
    assert isinstance(selected_stats, QuantizationStats)
    assert calls == [
        ("bf16", 0, 8, 16),
        ("row", 1, 8, 16, "down"),
    ]


def test_block_pow2_loader_dispatches_single_plane_conversion():
    calls = []
    weights = {"w8_gate_up": object()}
    stats = QuantizationStats()

    class Reader:
        def load_layer(
            self,
            layer,
            expert_start,
            expert_end,
            mode,
            fp8_impl,
        ):
            calls.append(
                (layer, expert_start, expert_end, mode, fp8_impl)
            )
            return weights, stats

    loaded, loaded_stats = S._load_fused_w8_expert_layer(
        Reader(), 3, 96, 128, "fp8", "block_pow2"
    )

    assert loaded is weights
    assert loaded_stats is stats
    assert calls == [(3, 96, 128, "fp8", "block_pow2")]


def test_block_pow2_coalesced_loader_dispatches_separate_conversion():
    calls = []
    weights = {"w8_gate_up": object()}
    stats = QuantizationStats()

    class Reader:
        def load_layer(
            self,
            layer,
            expert_start,
            expert_end,
            mode,
            fp8_impl,
        ):
            calls.append(
                (layer, expert_start, expert_end, mode, fp8_impl)
            )
            return weights, stats

    loaded, loaded_stats = S._load_fused_w8_expert_layer(
        Reader(),
        4,
        64,
        96,
        "fp8",
        "block_pow2_coalesced",
    )

    assert loaded is weights
    assert loaded_stats is stats
    assert calls == [
        (4, 64, 96, "fp8", "block_pow2_coalesced")
    ]


def test_layer_range_dispatches_skipped_layers_to_baseline_moe(monkeypatch):
    calls = []

    def fake_fused(self, layer, x2d, lead):
        calls.append(("row", layer))
        return torch.full((*lead, S.D.HIDDEN), 7.0, dtype=x2d.dtype)

    def fake_baseline(hidden, *args, **kwargs):
        calls.append(("bf16", int(args[0][0, 0].item())))
        output = torch.ones_like(hidden.float())
        return output, output * 2.0

    monkeypatch.setattr(S, "USE_MOE_FUSED_W8", True)
    monkeypatch.setattr(S, "USE_MOE_FUSED_W8_ROW_FP8", True)
    monkeypatch.setattr(S, "MOE_FUSED_W8_FP8_LAYER_START", 1)
    monkeypatch.setattr(S, "MOE_FUSED_W8_FP8_LAYER_LIMIT", 2)
    monkeypatch.setattr(S, "USE_MOE_CTE", False)
    monkeypatch.setattr(S, "USE_MOE_NKILIB", False)
    monkeypatch.setattr(S, "USE_MOE_FP8", False)
    monkeypatch.setattr(S.StaticDecode35B, "_moe_fused_w8", fake_fused)
    monkeypatch.setattr(S, "moe_forward", fake_baseline)

    model = object.__new__(S.StaticDecode35B)
    torch.nn.Module.__init__(model)
    model.batch_size = 1
    model.e_lo = 0
    model.e_hi = 1
    model.tp_group = [0]
    for name, value in (
        ("l0_router", 10.0),
        ("l0_gate_up", 11.0),
        ("l0_down", 12.0),
        ("l0_sh_gate", 13.0),
        ("l0_sh_up", 14.0),
        ("l0_sh_down", 15.0),
        ("l0_sh_sigmoid", 16.0),
    ):
        setattr(model, name, torch.full((1, 1), value))

    hidden = torch.zeros(1, 1, S.D.HIDDEN, dtype=torch.bfloat16)
    skipped = model._moe(0, hidden)
    selected = model._moe(1, hidden)

    assert calls == [("bf16", 10), ("row", 1)]
    torch.testing.assert_close(
        skipped, torch.full_like(hidden, 3.0)
    )
    torch.testing.assert_close(
        selected, torch.full_like(hidden, 7.0)
    )


def test_row_fp8_moe_reference_handles_affinity_and_activation_rounding():
    torch.manual_seed(17)
    experts, tokens, hidden_size, intermediate = 2, 3, 128, 64
    gate_up = encode_legacy_e4m3(
        torch.randn(experts, hidden_size, 2, intermediate) * 0.25
    )
    down = encode_legacy_e4m3(
        torch.randn(experts, intermediate, hidden_size) * 0.25
    )
    gate_up_scales = torch.rand(experts, 2, intermediate) * 0.01
    down_scales = torch.rand(experts, hidden_size) * 0.01
    hidden = torch.randn(tokens, hidden_size).to(torch.bfloat16)
    affinities = torch.tensor(
        [[0.25, 0.75], [1.0, 0.0], [0.4, 0.6]], dtype=torch.bfloat16
    )

    actual = fused_moe_row_fp8_cpu(
        hidden,
        gate_up,
        down,
        gate_up_scales,
        down_scales,
        affinities,
    )
    expected = torch.zeros_like(actual)
    for expert in range(experts):
        gate = (
            decode_legacy_e4m3(gate_up[expert, :, 0])
            * gate_up_scales[expert, 0].unsqueeze(0)
        )
        up = (
            decode_legacy_e4m3(gate_up[expert, :, 1])
            * gate_up_scales[expert, 1].unsqueeze(0)
        )
        down_weight = (
            decode_legacy_e4m3(down[expert])
            * down_scales[expert].unsqueeze(0)
        )
        activated = (
            F.silu(hidden.float() @ gate)
            * (hidden.float() @ up)
        ).to(torch.bfloat16).float()
        expected += (
            activated @ down_weight
        ) * affinities[:, expert : expert + 1].float()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("fp8_projections", ["gate_up", "down"])
def test_row_fp8_moe_reference_supports_mixed_bf16_projections(
    fp8_projections,
):
    torch.manual_seed(19)
    experts, tokens, hidden_size, intermediate = 2, 3, 128, 64
    gate_up_f32 = torch.randn(
        experts, hidden_size, 2, intermediate
    ) * 0.01
    down_f32 = torch.randn(
        experts, intermediate, hidden_size
    ) * 0.01
    if fp8_projections == "gate_up":
        gate_up = encode_legacy_e4m3(gate_up_f32)
        down = down_f32.to(torch.bfloat16)
    else:
        gate_up = gate_up_f32.to(torch.bfloat16)
        down = encode_legacy_e4m3(down_f32)
    gate_up_scales = torch.ones(experts, 2, intermediate)
    down_scales = torch.ones(experts, hidden_size)
    hidden = torch.randn(tokens, hidden_size).to(torch.bfloat16)
    affinities = torch.rand(tokens, experts).to(torch.bfloat16)

    actual = fused_moe_row_fp8_cpu(
        hidden,
        gate_up,
        down,
        gate_up_scales,
        down_scales,
        affinities,
    )
    assert torch.isfinite(actual).all()
    assert actual.shape == (tokens, hidden_size)


def test_local_routes_none_all_duplicates_and_ordering():
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    no_local = build_local_affinities(
        torch.tensor([[1, 2, 3, 4]]), weights, expert_start=8, local_experts=4
    )
    assert torch.equal(no_local, torch.zeros_like(no_local))

    all_local = build_local_affinities(
        torch.tensor([[8, 9, 10, 11]]), weights, expert_start=8, local_experts=4
    )
    torch.testing.assert_close(all_local, weights)

    duplicated = build_local_affinities(
        torch.tensor([[9, 8, 9, 11]]), weights, expert_start=8, local_experts=4
    )
    reordered = build_local_affinities(
        torch.tensor([[11, 9, 8, 9]]),
        torch.tensor([[0.4, 0.3, 0.2, 0.1]]),
        expert_start=8,
        local_experts=4,
    )
    expected = torch.tensor([[0.2, 0.4, 0.0, 0.4]])
    torch.testing.assert_close(duplicated, expected)
    torch.testing.assert_close(reordered, expected)


@pytest.mark.parametrize("mode", ["fp8", "int8"])
def test_quantized_moe_reference_handles_no_and_all_local_routes(mode):
    torch.manual_seed(7)
    experts, tokens, hidden_size, intermediate = 2, 3, 128, 128
    if mode == "fp8":
        gate_up = encode_legacy_e4m3(
            torch.randn(experts, 2, hidden_size, intermediate) * 0.25
        )
        down = encode_legacy_e4m3(
            torch.randn(experts, intermediate, hidden_size) * 0.25
        )
    else:
        gate_up = torch.randint(
            -8, 9, (experts, 2, hidden_size, intermediate), dtype=torch.int8
        )
        down = torch.randint(
            -8, 9, (experts, intermediate, hidden_size), dtype=torch.int8
        )
    gate_up_scales = torch.full(
        (experts, 2, 1, 1, 128), 0.01, dtype=torch.bfloat16
    )
    down_scales = torch.full((experts, 1, 1, 128), 0.01, dtype=torch.bfloat16)
    hidden = torch.randn(tokens, hidden_size)

    no_routes = fused_moe_cpu(
        hidden,
        gate_up,
        down,
        gate_up_scales,
        down_scales,
        torch.zeros(tokens, experts),
        mode,
    )
    assert torch.equal(no_routes, torch.zeros_like(no_routes))

    affinities = torch.tensor([[0.25, 0.75], [1.0, 0.0], [0.4, 0.6]])
    combined = fused_moe_cpu(
        hidden,
        gate_up,
        down,
        gate_up_scales,
        down_scales,
        affinities,
        mode,
    )
    expected = torch.zeros_like(combined)
    for expert in range(experts):
        one = torch.zeros_like(affinities)
        one[:, expert] = affinities[:, expert]
        expected += fused_moe_cpu(
            hidden,
            gate_up,
            down,
            gate_up_scales,
            down_scales,
            one,
            mode,
        )
    torch.testing.assert_close(combined, expected, rtol=1e-5, atol=1e-5)


def test_coalesced_moe_reference_handles_no_all_and_duplicate_routes():
    torch.manual_seed(23)
    experts, tokens, hidden_size, intermediate = 2, 3, 128, 128
    gate_up = encode_legacy_e4m3(
        torch.randn(experts, hidden_size, 2, intermediate) * 0.25
    )
    down = encode_legacy_e4m3(
        torch.randn(experts, intermediate, hidden_size) * 0.25
    )
    gate_up_scales, down_scales = pack_coalesced_block_scales(
        torch.full(
            (experts, 2, 1, 1), 0.01, dtype=torch.bfloat16
        ),
        torch.full(
            (experts, 1, 1), 0.01, dtype=torch.bfloat16
        ),
    )
    hidden = torch.randn(tokens, hidden_size).to(torch.bfloat16)
    no_routes = fused_moe_block_coalesced_cpu(
        hidden,
        gate_up,
        down,
        gate_up_scales,
        down_scales,
        torch.zeros(tokens, experts),
    )
    assert torch.equal(no_routes, torch.zeros_like(no_routes))

    top_ids = torch.tensor(
        [[1, 0, 1], [0, 0, 0], [1, 1, 0]]
    )
    top_weights = torch.tensor(
        [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.4, 0.1, 0.5]]
    )
    affinities = build_local_affinities(
        top_ids,
        top_weights,
        expert_start=0,
        local_experts=experts,
    )
    combined = fused_moe_block_coalesced_cpu(
        hidden,
        gate_up,
        down,
        gate_up_scales,
        down_scales,
        affinities,
    )
    expected = torch.zeros_like(combined)
    for expert in range(experts):
        one = torch.zeros_like(affinities)
        one[:, expert] = affinities[:, expert]
        expected += fused_moe_block_coalesced_cpu(
            hidden,
            gate_up,
            down,
            gate_up_scales,
            down_scales,
            one,
        )
    torch.testing.assert_close(combined, expected, rtol=1e-5, atol=1e-5)


def test_quantized_moe_reference_precision_controls_are_independent():
    torch.manual_seed(11)
    experts, tokens, hidden_size, intermediate = 1, 4, 128, 128
    gate_up = encode_legacy_e4m3(
        torch.randn(experts, 2, hidden_size, intermediate) * 0.25
    )
    down = encode_legacy_e4m3(
        torch.randn(experts, intermediate, hidden_size) * 0.25
    )
    gate_up_scales = torch.full(
        (experts, 2, 1, 1, 128), 0.01, dtype=torch.bfloat16
    )
    down_scales = torch.full(
        (experts, 1, 1, 128), 0.01, dtype=torch.bfloat16
    )
    hidden = torch.randn(tokens, hidden_size).to(torch.bfloat16)
    affinities = torch.tensor(
        [[0.12345], [0.23456], [0.34567], [0.45678]],
        dtype=torch.float32,
    )
    args = (
        hidden,
        gate_up,
        down,
        gate_up_scales,
        down_scales,
        affinities,
        "fp8",
    )
    full = fused_moe_cpu(*args)
    affinity = fused_moe_cpu(*args, rounding=("affinity_bf16",))
    activation = fused_moe_cpu(*args, rounding=("activation_bf16",))
    local_output = fused_moe_cpu(
        *args, rounding=("local_output_bf16",)
    )
    all_reduced = fused_moe_cpu(
        *args,
        rounding=(
            "affinity_bf16",
            "activation_bf16",
            "local_output_bf16",
        ),
    )

    assert not torch.equal(full, affinity)
    assert not torch.equal(full, activation)
    assert torch.equal(local_output, full.to(torch.bfloat16).float())
    assert not torch.equal(all_reduced, full)


def test_quantized_moe_reference_rejects_unknown_rounding_stage():
    args = make_minimal_fused_case()
    with pytest.raises(ValueError, match="unknown MoE rounding"):
        fused_moe_cpu(*args, rounding=("not_a_stage",))


def make_minimal_fused_case():
    hidden = torch.ones(1, 128, dtype=torch.bfloat16)
    gate_up = encode_legacy_e4m3(torch.ones(1, 2, 128, 128))
    down = encode_legacy_e4m3(torch.ones(1, 128, 128))
    gate_up_scales = torch.ones(1, 2, 1, 1, 128, dtype=torch.bfloat16)
    down_scales = torch.ones(1, 1, 1, 128, dtype=torch.bfloat16)
    affinities = torch.ones(1, 1)
    return (
        hidden,
        gate_up,
        down,
        gate_up_scales,
        down_scales,
        affinities,
        "fp8",
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
