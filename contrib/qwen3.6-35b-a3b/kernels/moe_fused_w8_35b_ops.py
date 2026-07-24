"""PyTorch custom-op registration for fused high-batch block-W8 MoE."""

import torch
from torch_neuronx import nki_op

from moe_fused_w8_35b import (
    nki_moe_fused_w8_fp8,
    nki_moe_fused_w8_fp8_block_coalesced,
    nki_moe_fused_w8_fp8_block_coalesced_ob,
    nki_moe_fused_w8_fp8_dual,
    nki_moe_fused_w8_fp8_native,
    nki_moe_fused_w8_fp8_token_stationary,
    nki_moe_fused_w8_int8,
)


@nki_op("moe_w8::fused_fp8", mutates_args={})
def fused_fp8(
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_scales: torch.Tensor,
    affinities: torch.Tensor,
) -> torch.Tensor:
    return nki_moe_fused_w8_fp8(
        hidden, gate_up, down, gate_up_scales, down_scales, affinities
    )


@nki_op("moe_w8::fused_fp8_dual", mutates_args={})
def fused_fp8_dual(
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    gate_up_residual: torch.Tensor,
    down: torch.Tensor,
    down_residual: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_scales: torch.Tensor,
    affinities: torch.Tensor,
) -> torch.Tensor:
    return nki_moe_fused_w8_fp8_dual(
        hidden,
        gate_up,
        gate_up_residual,
        down,
        down_residual,
        gate_up_scales,
        down_scales,
        affinities,
    )


@nki_op("moe_w8::fused_fp8_native", mutates_args={})
def fused_fp8_native(
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_scales: torch.Tensor,
    affinities: torch.Tensor,
) -> torch.Tensor:
    return nki_moe_fused_w8_fp8_native(
        hidden, gate_up, down, gate_up_scales, down_scales, affinities
    )


@nki_op("moe_w8::fused_fp8_block_coalesced", mutates_args={})
def fused_fp8_block_coalesced(
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_scales: torch.Tensor,
    affinities: torch.Tensor,
) -> torch.Tensor:
    return nki_moe_fused_w8_fp8_block_coalesced(
        hidden, gate_up, down, gate_up_scales, down_scales, affinities
    )


@nki_op("moe_w8::fused_fp8_block_coalesced_ob", mutates_args={})
def fused_fp8_block_coalesced_ob(
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_scales: torch.Tensor,
    affinities: torch.Tensor,
) -> torch.Tensor:
    return nki_moe_fused_w8_fp8_block_coalesced_ob(
        hidden, gate_up, down, gate_up_scales, down_scales, affinities
    )


@nki_op("moe_w8::fused_fp8_token_stationary", mutates_args={})
def fused_fp8_token_stationary(
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_scales: torch.Tensor,
    affinities: torch.Tensor,
) -> torch.Tensor:
    return nki_moe_fused_w8_fp8_token_stationary(
        hidden, gate_up, down, gate_up_scales, down_scales, affinities
    )


@nki_op("moe_w8::fused_int8", mutates_args={})
def fused_int8(
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_scales: torch.Tensor,
    affinities: torch.Tensor,
) -> torch.Tensor:
    return nki_moe_fused_w8_int8(
        hidden, gate_up, down, gate_up_scales, down_scales, affinities
    )
