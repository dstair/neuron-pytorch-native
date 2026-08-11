"""FP8xFP8 context-encoding MoE prefill kernel (Trainium2 double_row 2x path).

Goal: run the prefill MoE experts as native FP8xFP8 matmuls to (a) halve resident
expert HBM (enabling higher prefill batch sizes) and (b) capture Trainium2's 2x FP8
Tensor-Engine throughput (158 vs 79 BF16 TFLOPS via the "double_row" contraction).

Design (thin wrapper, NOT a from-scratch kernel): nkilib's shared blockwise-MoE
`compute_one_block` already implements the double_row FP8xFP8 path, gated on
`is_block_quant` (256x256 block quant, see bwmm_shard_on_I.py). The prefill `_hybrid`
entry hardcodes `is_block_quant=False`, but the sibling BASELINE entry
`blockwise_mm_baseline_shard_intermediate` exposes `is_block_quant` directly AND takes
the same routed metadata (token_position_to_id / block_to_expert). The existing LNC=1
patch (patches/nkilib-lnc1-moe-cte.patch) already relaxes the baseline + shared helpers
for NUM_SHARDS==1, so TP=8/LNC=1 is supported. We therefore just pack routes (reusing
moe_cte_35b's packer) and call the baseline with is_block_quant=True + 256x256 block
scales. Weights come from OfficialFP8ExpertReader (int8-stored legacy-E4M3); scales are
the 256x256-block tables produced by the CPU packer (see moe_w8 / moe_cte_adapter).

Tradeoff vs the hybrid: the baseline processes padded blocks too (no dynamic-loop skip),
so there is some wasted compute on padding. Acceptable for v1; if padding overhead is
material at BS=16, fork the hybrid driver to pass is_block_quant=True instead.

STATUS: DEVICE-PENDING. NKI does not trace/compile off-device, so this has not been
run yet. Items to confirm on the first trn2 compile (Step 5 of the plan):
  - weight tensor dtype the baseline expects for block-quant (float8_e4m3 dtype tensor
    vs int8-stored + in-kernel .view) and how gate_up_proj_scale / down_proj_scale must
    be shaped/reshaped (bwmm_shard_on_I.py:1104-1128 reshapes gup scale to
    [E, H/256 * 2 * I/256 * 128]; down scale loaded near :2087).
  - that block-quant double_row actually runs at NUM_SHARDS==1 (the LNC=1 patch was
    validated on the BF16 non-block path; block-quant at 1 shard is unproven).
  - activation numerics: block-quant casts hidden->FP8 with NO per-token scale
    (bwmm_shard_on_I.py:1095); measure cosine vs BF16 CTE, add activation scaling only
    if it fails the gate.
"""

import nki
import nki.language as nl

from nkilib.core.moe.moe_cte.bwmm_shard_on_I import (
    blockwise_mm_baseline_shard_intermediate,
)
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from nkilib.core.utils.kernel_assert import kernel_assert

# Reuse the validated route-packing preamble and constants from the BF16 wrapper so
# the FP8 path packs routes identically (only the matmul dtype/scale path differs).
# Sibling import (kernels/ is on sys.path), matching static_decode_35b.py's convention.
from moe_cte_35b import (
    _DIRECT_ROUTE_MAX_ASSIGNMENTS,
    _max_packed_blocks,
    _pack_local_routes_impl,
    SkipMode,
)

_moe_cte_baseline_impl = blockwise_mm_baseline_shard_intermediate.func


@nki.jit
def nki_moe_cte_fp8_routed_35b(
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,   # [E, H, 2, I_TP], nl.int8 holding legacy-E4M3 bytes
    down_proj_weight,      # [E, I_TP, H],    nl.int8 holding legacy-E4M3 bytes
    gate_up_proj_scale,    # 256x256-block scale table (gate/up), see module docstring
    down_proj_scale,       # 256x256-block scale table (down)
    expert_indices,
    expert_lo: int,
    block_size: int,
):
    """Pack routes, then run the CTE MoE as FP8xFP8 (block-quant double_row).

    Mirrors nki_moe_cte_routed_35b's packing preamble verbatim, then dispatches to the
    baseline shard-on-I kernel with is_block_quant=True instead of the BF16 hybrid.
    """
    num_local_experts, _, _ = down_proj_weight.shape
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

    token_position_to_id, block_to_expert, _conditions = _pack_local_routes_impl(
        expert_indices,
        expert_lo,
        num_local_experts,
        block_size,
        nl.shared_hbm,
        metadata_tensors=metadata_tensors,
        scatter_barrier=assignments <= _DIRECT_ROUTE_MAX_ASSIGNMENTS,
    )

    # Weights arrive as nl.int8 holding legacy-E4M3 bytes and are reinterpreted to
    # nki's legacy float8_e4m3 here. torch has no legacy-e4m3 dtype and neuronx-cc
    # rejects F8E4M3FN operands on TRN2 (NCC_EVRF051), so the HLO custom-call operand
    # must be int8 (legal); the .view is an in-kernel reinterpret that makes the
    # baseline's DMA + double_row matmul treat them as FP8. Mirrors the int8-store +
    # .view(nl.float8_e4m3) pattern in moe_fused_w8_35b.py:_load_native_fp8_tile.
    gate_up_proj_weight_fp8 = gate_up_proj_weight.view(nl.float8_e4m3)
    down_proj_weight_fp8 = down_proj_weight.view(nl.float8_e4m3)

    # --- FP8xFP8 block-quant double_row MoE (baseline entry; padded blocks included) ---
    # The baseline processes all N blocks; padded tokens map to id=T (dummy row), so the
    # result is correct. is_block_quant=True selects the double_row 2x path in the shared
    # compute_one_block; the block scales are applied post-accumulation per 256x256 block.
    return _moe_cte_baseline_impl(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight_fp8,
        down_proj_weight=down_proj_weight_fp8,
        block_size=block_size,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        is_block_quant=True,
        activation_function=ActFnType.SiLU,
        skip_dma=SkipMode(False, False),
        compute_dtype=nl.bfloat16,
        is_tensor_update_accumulating=True,
        # Match the BF16 routed wrapper's affinity handling (POST_SCALE), not the
        # baseline's PRE_SCALE default.
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    )
