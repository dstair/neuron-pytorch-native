"""Custom-op registration for row-scaled FP8/BF16 MoE TKG."""

import torch
from torch_neuronx import nki_op

from moe_tkg_row_fp8_35b import nki_moe_tkg_row_fp8


@nki_op("moe_w8::tkg_row_fp8", mutates_args={})
def tkg_row_fp8(
    hidden: torch.Tensor,
    gate_up_i8: torch.Tensor,
    down_i8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_scales: torch.Tensor,
    affinities: torch.Tensor,
    expert_index: torch.Tensor,
    rank_id: torch.Tensor,
) -> torch.Tensor:
    return nki_moe_tkg_row_fp8(
        hidden,
        gate_up_i8,
        down_i8,
        gate_up_scales,
        down_scales,
        affinities,
        expert_index,
        rank_id,
    )
