"""FP8xFP8 context-encoding MoE prefill kernel (Trainium2 double_row 2x path).

Goal: run the prefill MoE experts as native FP8xFP8 matmuls to (a) halve resident
expert HBM (enabling higher prefill batch sizes) and (b) capture Trainium2's 2x FP8
Tensor-Engine throughput (158 vs 79 BF16 TFLOPS via the "double_row" contraction).

Design: nkilib's shared `compute_one_block` already implements the double_row FP8xFP8
path, gated on `is_block_quant` (256x256 block quant, see bwmm_shard_on_I.py). We drive
it through the **hybrid** entry `blockwise_mm_baseline_shard_intermediate_hybrid`, which
(unlike the baseline) processes blocks with a dynamic loop over `conditions` — one block
at a time — so it scales to any block count (the baseline lays out a [num_blocks]-partition
buffer that overflows the 128 SBUF-partition limit at BS=16, where max_blocks=288). The
hybrid originally hardcoded is_block_quant=False; a one-line nkilib patch
(patches/nkilib-blockquant-hybrid.patch) parameterizes it, defaulting False so the BF16
path is unchanged. The existing LNC=1 patch already makes the hybrid's dynamic loop +
barriers safe at NUM_SHARDS==1 (the BF16 routed path uses this same loop at
num_static_block=0), and padded blocks are skipped via `conditions` (no expert-sentinel
OOB, so no routing clamp is needed).

Weights are stored as nl.int8 holding legacy-E4M3 bytes and reinterpreted to
nl.float8_e4m3 here: torch has no legacy-e4m3 dtype and neuronx-cc rejects F8E4M3FN
operands on TRN2 (NCC_EVRF051), so the HLO custom-call operand must be int8; the .view
is an in-kernel reinterpret (mirrors moe_fused_w8_35b.py:_load_native_fp8_tile). Block
scales are 256x256-block dequant tables built by moe_w8.quantize_{gate_up,down}_256block:
gate_up [E,H//256,2,I//256,128], down [E,I//256,H//256,128].
"""

import nki
import nki.language as nl

from nkilib.core.moe.moe_cte.bwmm_shard_on_I import (
    blockwise_mm_baseline_shard_intermediate_hybrid,
)
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from nkilib.core.utils.kernel_assert import kernel_assert

# Reuse the validated route-packing preamble and constants from the BF16 wrapper so the
# FP8 path packs routes identically (only the matmul dtype/scale path differs). Sibling
# import (kernels/ is on sys.path), matching static_decode_35b.py's convention.
from moe_cte_35b import (
    _DIRECT_ROUTE_MAX_ASSIGNMENTS,
    _max_packed_blocks,
    _pack_local_routes_impl,
    SkipMode,
)

_moe_cte_fp8_hybrid_impl = blockwise_mm_baseline_shard_intermediate_hybrid.func


@nki.jit
def nki_moe_cte_fp8_routed_35b(
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,   # [E, H, 2, I_TP], nl.int8 holding legacy-E4M3 bytes
    down_proj_weight,      # [E, I_TP, H],    nl.int8 holding legacy-E4M3 bytes
    gate_up_proj_scale,    # [E, H//256, 2, I//256, 128]  256x256-block dequant scale
    down_proj_scale,       # [E, I//256, H//256, 128]
    expert_indices,
    expert_lo: int,
    block_size: int,
):
    """Pack routes, then run the CTE MoE as FP8xFP8 block-quant double_row (2x)."""
    # Weights/scales/affinities are padded to E+1 experts (the extra dummy expert E is
    # all-zero), so the padding-block sentinel (== E) gathers the dummy row in-bounds
    # instead of overflowing the [E,...] tables -- no routing clamp needed. The REAL
    # local-expert count for the route packer is therefore shape[0] - 1.
    num_experts_padded, _, _ = down_proj_weight.shape
    num_local_experts = num_experts_padded - 1
    kernel_assert(
        num_local_experts in (32, 64),
        f"routed CTE requires 32 or 64 local experts, found {num_local_experts}",
    )
    T, K = expert_indices.shape
    assignments = T * K
    max_blocks = _max_packed_blocks(assignments, num_local_experts, block_size)
    packed_len = max_blocks * block_size
    shard_id = nl.program_id(axis=0)
    num_shards = nl.num_programs(axes=0)
    kernel_assert(
        num_shards in (1, 2),
        f"routed CTE requires LNC1 or LNC2, found {num_shards} shards",
    )

    # --- Route-packing preamble (identical to the BF16 routed wrapper) ---
    token_position_0 = nl.ndarray(
        (packed_len,), dtype=nl.int32, buffer=nl.shared_hbm,
        name="fp8_route_pack_token_position_shard_0",
    )
    block_to_expert_0 = nl.ndarray(
        (max_blocks, 1), dtype=nl.int32, buffer=nl.shared_hbm,
        name="fp8_route_pack_block_expert_shard_0",
    )
    conditions_0 = nl.ndarray(
        (max_blocks + 1,), dtype=nl.int32, buffer=nl.shared_hbm,
        name="fp8_route_pack_conditions_shard_0",
    )
    if num_shards == 1:
        metadata_tensors = (token_position_0, block_to_expert_0, conditions_0)
    else:
        token_position_1 = nl.ndarray(
            (packed_len,), dtype=nl.int32, buffer=nl.shared_hbm,
            name="fp8_route_pack_token_position_shard_1",
        )
        block_to_expert_1 = nl.ndarray(
            (max_blocks, 1), dtype=nl.int32, buffer=nl.shared_hbm,
            name="fp8_route_pack_block_expert_shard_1",
        )
        conditions_1 = nl.ndarray(
            (max_blocks + 1,), dtype=nl.int32, buffer=nl.shared_hbm,
            name="fp8_route_pack_conditions_shard_1",
        )
        if shard_id == 0:
            metadata_tensors = (token_position_0, block_to_expert_0, conditions_0)
        else:
            metadata_tensors = (token_position_1, block_to_expert_1, conditions_1)

    token_position_to_id, block_to_expert, conditions = _pack_local_routes_impl(
        expert_indices,
        expert_lo,
        num_local_experts,
        block_size,
        nl.shared_hbm,
        metadata_tensors=metadata_tensors,
        scatter_barrier=assignments <= _DIRECT_ROUTE_MAX_ASSIGNMENTS,
    )

    # int8 (legacy-E4M3) -> nki float8_e4m3 view; HLO operand stays int8 (legal on TRN2).
    gate_up_proj_weight_fp8 = gate_up_proj_weight.view(nl.float8_e4m3)
    down_proj_weight_fp8 = down_proj_weight.view(nl.float8_e4m3)

    # Hybrid entry: is_block_quant=True selects the double_row 2x path in the shared
    # compute_one_block; num_static_block=0 routes every block through the dynamic loop
    # (padded blocks skipped via `conditions`), so any block count fits.
    return _moe_cte_fp8_hybrid_impl(
        conditions=conditions,
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight_fp8,
        down_proj_weight=down_proj_weight_fp8,
        block_size=block_size,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        num_static_block=0,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        is_block_quant=True,
        activation_function=ActFnType.SiLU,
        skip_dma=SkipMode(False, False),
        compute_dtype=nl.bfloat16,
        is_tensor_update_accumulating=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    )
