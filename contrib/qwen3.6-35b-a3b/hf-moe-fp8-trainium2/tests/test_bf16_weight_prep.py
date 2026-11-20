#!/usr/bin/env python3
"""CPU checks for direct BF16-to-FP8 expert weight preparation."""

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import bf16_weight_prep as bp  # noqa: E402
import moe_w8  # noqa: E402


def _make_weights(experts=2, hidden=512, intermediate=512):
    generator = torch.Generator().manual_seed(17)
    gate_up = (
        torch.randn(
            experts,
            2 * intermediate,
            hidden,
            generator=generator,
        )
        * 0.02
    ).bfloat16()
    down = (
        torch.randn(
            experts,
            hidden,
            intermediate,
            generator=generator,
        )
        * 0.02
    ).bfloat16()
    return gate_up, down


def test_output_block_matrix_quantization():
    weight = _make_weights(experts=1)[0][0, :512]
    quantized, scales, stats = bp.quantize_bf16_output_blocks(weight)
    assert quantized.shape == weight.shape
    assert quantized.dtype == torch.int8
    assert scales.shape == (4, 4)
    assert scales.dtype == torch.bfloat16
    assert torch.equal(scales, scales[:, :1].expand_as(scales))
    assert stats.normalized_rmse < 0.035


def test_row_matrix_quantization():
    weight = _make_weights(experts=1)[0][0, :512]
    quantized, scales, stats = bp.quantize_bf16_rows(weight)
    assert quantized.shape == weight.shape
    assert quantized.dtype == torch.int8
    assert scales.shape == (512,)
    assert scales.dtype == torch.float32
    reconstructed = (
        moe_w8.decode_legacy_e4m3(quantized) * scales[:, None]
    )
    relative_l2 = float(
        (reconstructed.double() - weight.double()).norm()
        / weight.double().norm().clamp_min(1e-30)
    )
    assert abs(relative_l2 - stats.normalized_rmse) < 1e-7
    assert relative_l2 < 0.035


def test_prepare_output_block_operands():
    gate_up, down = _make_weights()
    operands, report = bp.prepare_bf16_expert_tensors(
        gate_up,
        down,
        layout="output_block",
    )
    assert operands["w8_gate_up"].shape == (2, 512, 2, 512)
    assert operands["w8_down"].shape == (2, 512, 512)
    assert operands["w8_gate_up_scale"].shape == (2, 128, 32)
    assert operands["w8_down_scale"].shape == (2, 128, 16)
    assert all(error < 0.035 for error in report.relative_l2.values())


def test_prepare_row_operands():
    gate_up, down = _make_weights()
    operands, report = bp.prepare_bf16_expert_tensors(
        gate_up,
        down,
        layout="row",
    )
    assert operands["row_gate_up"].shape == (2, 512, 2, 512)
    assert operands["row_gate_up_scale"].shape == (2, 2, 512)
    assert operands["row_gate_up_scale"].dtype == torch.float32
    assert operands["row_down"].shape == (2, 512, 512)
    assert operands["row_down_scale"].shape == (2, 512)
    assert all(error < 0.035 for error in report.relative_l2.values())


def test_rejects_non_bf16_source():
    try:
        bp.quantize_bf16_rows(torch.zeros(128, 128))
        raise AssertionError("FP32 source was accepted")
    except TypeError:
        pass


if __name__ == "__main__":
    test_output_block_matrix_quantization()
    test_row_matrix_quantization()
    test_prepare_output_block_operands()
    test_prepare_row_operands()
    test_rejects_non_bf16_source()
    print("direct BF16 weight-prep checks: PASS")
