"""Graph-safe LNC2 wrapper for nkilib's context-encoding MoE kernel."""

import nki
import nki.language as nl

from nkilib.core.moe.moe_cte.bwmm_shard_on_I import (
    blockwise_mm_baseline_shard_intermediate_hybrid,
)
from nkilib.core.moe.moe_cte.moe_cte_utils import SkipMode
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

_moe_cte_hybrid_impl = blockwise_mm_baseline_shard_intermediate_hybrid.func


@nki.jit
def nki_moe_cte_35b(
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,
    down_proj_weight,
    token_position_to_id,
    block_to_expert,
    conditions,
    block_size: int,
):
    """Expose only tensors and integers to Dynamo; keep NKI config internal."""
    return _moe_cte_hybrid_impl(
        conditions=conditions,
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        block_size=block_size,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        num_static_block=0,
        activation_function=ActFnType.SiLU,
        skip_dma=SkipMode(False, False),
        compute_dtype=nl.bfloat16,
        is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    )
