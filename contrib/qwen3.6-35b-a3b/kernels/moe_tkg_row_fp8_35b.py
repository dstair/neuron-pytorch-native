"""Legacy-FP8 storage adapter for nkilib's all-expert MoE TKG kernel."""

import nki
import nki.language as nl

from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
try:
    from nkilib.core.mlp.mlp_parameters import (
        ROW_FP8_INT8_STORAGE_SUPPORTED,
    )
except ImportError as exc:
    raise RuntimeError(
        "row-FP8 MoE requires deploy/nkilib_row_fp8_int8_storage.patch"
    ) from exc
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

if not ROW_FP8_INT8_STORAGE_SUPPORTED:
    raise RuntimeError("nkilib row-FP8 INT8 storage support is disabled")


@nki.jit
def nki_moe_tkg_row_fp8(
    hidden,
    gate_up_i8,
    down_i8,
    gate_up_scales,
    down_scales,
    affinities,
    expert_index,
    rank_id,
):
    """Run row-scaled legacy E4M3, optionally mixed with BF16 projections."""
    return moe_tkg(
        hidden_input=hidden,
        expert_gate_up_weights=gate_up_i8,
        expert_down_weights=down_i8,
        expert_affinities=affinities,
        expert_index=expert_index,
        is_all_expert=True,
        rank_id=rank_id,
        expert_gate_up_weights_scale=gate_up_scales,
        expert_down_weights_scale=down_scales,
        mask_unselected_experts=False,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        activation_fn=ActFnType.SiLU,
        output_dtype=nl.bfloat16,
    )
