"""Graph-safe LNC1/LNC2 wrappers for nkilib's context-encoding MoE kernel."""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.moe.moe_cte.bwmm_shard_on_I import (
    blockwise_mm_baseline_shard_intermediate_hybrid,
)
from nkilib.core.moe.moe_cte.moe_cte_utils import SkipMode, stream_shuffle_broadcast
from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
from nkilib.core.utils.kernel_assert import kernel_assert

_moe_cte_hybrid_impl = blockwise_mm_baseline_shard_intermediate_hybrid.func

_TOP_K = 8
_ROUTE_TILE = 2048
_ASSIGNMENT_SLICE = 128
_DIRECT_ROUTE_MAX_ASSIGNMENTS = 16384

# Lanes per `nonzero_with_count` call in the direct route packer, and the partition
# stride between them. 4 lanes at a 32-partition stride looks like it wastes half the
# GpSimd hardware -- the ISA has 8 cores at a 16-partition stride (partitions
# 0,16,...,112; nkilib core/subkernels/find_nonzero_indices.py: _NUM_GPSIMD_CORES = 8,
# _PARTITIONS_PER_GPSIMD = 16) -- but 8 lanes at stride 16 is NOT expressible:
#   MEASURED 2026-08-21: (8, 16) fails BIR verification with
#   "Invalid access of 1 partitions starting at partition 16"
#   on the per-lane route_pack_direct_match tensor_scalar.
# A single-partition SBUF access must begin on a 32-partition QUADRANT boundary
# (_QUADRANT_SIZE = 32), so the odd cores at 16/48/80/112 are unreachable directly --
# which is exactly why find_nonzero_indices carries a quad_mask stream shuffle for
# them. Using all 8 cores therefore costs extra shuffles on both the input and the
# result, against a ~2.1%-of-region ceiling. Not worth it; keep (4, 32).
_ROUTE_LANES = 4
_ROUTE_LANE_STRIDE = 32

# Route-packer overrides, read once at import. This module is NKI-traced, so the
# `os` module must NOT stay in its namespace -- the specializer walks module
# globals and rejects it ("NKI classes must inherit from either Enum or
# NKIObject"). Keep only plain str/bool constants.
import os as _os  # noqa: E402

_SCATTER_BARRIER_OVERRIDE = _os.environ.get("MOE_CTE_SCATTER_BARRIER", "")
_FORCE_TILED_ROUTE = _os.environ.get("MOE_CTE_FORCE_TILED", "") == "1"
# Diagnostic ONLY: neutralize one tiled-exclusive construct to find which one
# deadlocks the fused TP=8 graph. Output is deliberately WRONG under any
# ablation -- the only question asked is "does it still hang".
_ABLATE = _os.environ.get("MOE_CTE_ABLATE", "")
# MOE_CTE_FP8_OUTPUT_BLOCK (read in static_decode_35b.py) reduces the HOST scale
# grid over the contraction axis. MOE_CTE_FP8_OB_HOIST then lets the KERNEL hoist
# the scale out of the contraction loop and PSUM-accumulate. Keeping them separate
# enables the numerics-first staging step: grid on + hoist OFF produces the new
# numerics at the OLD op count, so accuracy is gated before any kernel risk.
# Default: the hoist follows the grid.
_FP8_OUTPUT_BLOCK_QUANT = _os.environ.get("MOE_CTE_FP8_OUTPUT_BLOCK", "0") == "1"
_FP8_OUTPUT_BLOCK_HOIST = (
    _os.environ.get("MOE_CTE_FP8_OB_HOIST", "1" if _FP8_OUTPUT_BLOCK_QUANT else "0")
    == "1"
)
if _FP8_OUTPUT_BLOCK_HOIST and not _FP8_OUTPUT_BLOCK_QUANT:
    raise RuntimeError(
        "MOE_CTE_FP8_OB_HOIST requires MOE_CTE_FP8_OUTPUT_BLOCK=1: the kernel "
        "hoist is only correct when the host scale grid is constant along the "
        "contraction axis"
    )
if _SCATTER_BARRIER_OVERRIDE != "" or _FORCE_TILED_ROUTE:
    # Positive evidence in the log: a hang under an override must not be
    # confused with an override that never reached the container.
    print(
        f"[moe-cte-route] scatter_barrier_override="
        f"{_SCATTER_BARRIER_OVERRIDE or 'none'} force_tiled={_FORCE_TILED_ROUTE}",
        flush=True,
    )
del _os


def _scatter_barrier_default(num_assignments: int) -> bool:
    """Per-slice scatter barriers: on for the direct packer, off for the tiled scan.

    ``MOE_CTE_SCATTER_BARRIER`` (1/0) overrides the size-derived choice, which
    is what separates "the barrier deadlocks" from "the token count deadlocks"
    at a fixed call size.
    """
    if _SCATTER_BARRIER_OVERRIDE != "":
        return _SCATTER_BARRIER_OVERRIDE == "1"
    return num_assignments <= _DIRECT_ROUTE_MAX_ASSIGNMENTS


def _max_packed_blocks(
    num_assignments: int, num_local_experts: int, block_size: int
) -> int:
    """Maximum blocks for the local experts and one out-of-rank dummy group."""
    return (num_assignments - num_local_experts + block_size - 1) // block_size + num_local_experts


def _init_route_metadata(
    token_position_to_id,
    block_to_expert,
    conditions,
    num_tokens: int,
    num_local_experts: int,
    sbm: SbufManager,
    writer_id: int,
    num_writers: int,
):
    """Initialize fixed-size metadata with the CTE padding token/expert values."""
    sbm.open_scope(name="route_pack_metadata_init")
    packed_len = token_position_to_id.shape[0]
    packed_cols = packed_len // 128
    kernel_assert(
        packed_cols % num_writers == 0,
        f"packed metadata columns ({packed_cols}) must divide {num_writers} writers",
    )
    packed_cols_per_writer = packed_cols // num_writers
    packed_col_lo = writer_id * packed_cols_per_writer
    packed_col_hi = packed_col_lo + packed_cols_per_writer
    packed_init = sbm.alloc_stack(
        (128, packed_cols_per_writer),
        dtype=nl.int32,
        name="route_pack_packed_init",
    )
    nisa.memset(
        dst=packed_init,
        value=num_tokens,
        name="route_pack_init_token_positions",
    )
    nisa.dma_copy(
        dst=token_position_to_id.reshape((128, packed_cols))[
            :, packed_col_lo:packed_col_hi
        ],
        src=packed_init,
        name="route_pack_store_token_positions_init",
    )

    num_blocks = block_to_expert.shape[0]
    block_lo = writer_id * num_blocks // num_writers
    block_hi = (writer_id + 1) * num_blocks // num_writers
    # block_to_expert is [num_blocks, 1] (partition-major over blocks), so its init
    # must be tiled to <=128 partitions: at high batch num_blocks exceeds the SBUF
    # 128-partition limit (e.g. BS=16 -> 288 blocks) and a single [num_blocks,1] memset
    # trips "memset dst partition dimension exceeds maximum 128". Compile-time loop;
    # one iteration at low batch (BS=2 -> 64 blocks), so BF16 behavior is unchanged.
    block_init = sbm.alloc_stack(
        (128, 1),
        dtype=nl.int32,
        name="route_pack_block_init",
    )
    # Plain `for x in range(...)` with a SIMPLE loop variable: NKI's loop frontend
    # rejects a tuple loop var (e.g. `for i, x in enumerate(...)`) with "expecting
    # simple variable". Slice bounds are precomputed simple variables for the same
    # reason (an arithmetic expression as a bound is also rejected). Unique op names
    # are derived from the (compile-time-constant) loop offset.
    for _blk_lo in range(block_lo, block_hi, 128):
        _blk_hi = min(_blk_lo + 128, block_hi)
        _blk_n = _blk_hi - _blk_lo
        nisa.memset(
            dst=block_init[0:_blk_n],
            value=num_local_experts,
            name=f"route_pack_init_block_experts_{_blk_lo}",
        )
        nisa.dma_copy(
            dst=block_to_expert[_blk_lo:_blk_hi, :],
            src=block_init[0:_blk_n],
            name=f"route_pack_store_block_experts_init_{_blk_lo}",
        )

    num_conditions = num_blocks + 1
    condition_lo = writer_id * num_conditions // num_writers
    condition_hi = (writer_id + 1) * num_conditions // num_writers
    condition_init = sbm.alloc_stack(
        (1, condition_hi - condition_lo),
        dtype=nl.int32,
        name="route_pack_condition_init",
    )
    nisa.memset(
        dst=condition_init,
        value=0,
        name="route_pack_init_conditions",
    )
    nisa.dma_copy(
        dst=conditions.reshape((1, num_conditions))[
            0:1, condition_lo:condition_hi
        ],
        src=condition_init,
        name="route_pack_store_conditions_init",
    )
    sbm.close_scope()


def _pack_local_routes_impl(
    expert_indices,
    expert_lo: int,
    num_local_experts: int,
    block_size: int,
    metadata_buffer,
    metadata_tensors=None,
    writer_id: int = 0,
    num_writers: int = 1,
    scatter_barrier: bool = False,
):
    """Build stable, expert-grouped CTE metadata in two linear route passes."""
    T, K = expert_indices.shape
    assignments = T * K
    num_shards = nl.num_programs(axes=0)
    kernel_assert(K == _TOP_K, f"route packer requires top-k {_TOP_K}, found {K}")
    kernel_assert(
        num_shards in (1, 2),
        f"route packer requires LNC1 or LNC2, found {num_shards} shards",
    )
    kernel_assert(
        block_size == 256 or block_size == 512,
        f"route packer supports block size 256 or 512, found {block_size}",
    )
    kernel_assert(
        assignments % _ROUTE_TILE == 0,
        f"route assignments ({assignments}) must be divisible by {_ROUTE_TILE}",
    )
    kernel_assert(
        num_local_experts in (32, 64),
        f"route packer supports 32 or 64 local experts, found {num_local_experts}",
    )
    max_blocks = _max_packed_blocks(assignments, num_local_experts, block_size)
    packed_len = max_blocks * block_size
    if metadata_tensors is None:
        token_position_to_id = nl.ndarray(
            (packed_len,),
            dtype=nl.int32,
            buffer=metadata_buffer,
            name="route_pack_token_position_to_id",
        )
        block_to_expert = nl.ndarray(
            (max_blocks, 1),
            dtype=nl.int32,
            buffer=metadata_buffer,
            name="route_pack_block_to_expert",
        )
        conditions = nl.ndarray(
            (max_blocks + 1,),
            dtype=nl.int32,
            buffer=metadata_buffer,
            name="route_pack_conditions",
        )
    else:
        token_position_to_id, block_to_expert, conditions = metadata_tensors

    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size)
    sbm.open_scope(name="route_pack")
    _init_route_metadata(
        token_position_to_id,
        block_to_expert,
        conditions,
        T,
        num_local_experts,
        sbm,
        writer_id,
        num_writers,
    )

    routes_hbm = expert_indices.reshape((1, assignments))
    expert_keys = sbm.alloc_stack(
        (num_local_experts, _ROUTE_TILE),
        dtype=nl.float32,
        name="route_pack_expert_keys",
    )
    nisa.iota(
        dst=expert_keys,
        pattern=[[0, _ROUTE_TILE]],
        offset=expert_lo,
        channel_multiplier=1,
        name="route_pack_expert_iota",
    )

    # Pass 1: count assignments for each local expert.
    counts = sbm.alloc_stack(
        (num_local_experts, 1),
        dtype=nl.float32,
        name="route_pack_counts",
    )
    nisa.memset(dst=counts, value=0, name="route_pack_zero_counts")
    for tile_idx in nl.sequential_range(assignments // _ROUTE_TILE):
        sbm.open_scope(name=f"route_pack_count_tile_{tile_idx}")
        route_values = sbm.alloc_stack(
            (num_local_experts, _ROUTE_TILE),
            dtype=nl.float32,
            name=f"route_pack_count_routes_t{tile_idx}",
        )
        nisa.dma_copy(
            dst=route_values[0:1, :],
            src=routes_hbm[0:1, tile_idx * _ROUTE_TILE : (tile_idx + 1) * _ROUTE_TILE],
            name=f"route_pack_count_load_t{tile_idx}",
        )
        stream_shuffle_broadcast(route_values[0:1, :], route_values)

        matches = sbm.alloc_stack(
            (num_local_experts, _ROUTE_TILE),
            dtype=nl.float32,
            name=f"route_pack_count_matches_t{tile_idx}",
        )
        nisa.tensor_tensor(
            dst=matches,
            data1=route_values,
            data2=expert_keys,
            op=nl.equal,
            name=f"route_pack_count_equal_t{tile_idx}",
        )
        tile_counts = sbm.alloc_stack(
            (num_local_experts, 1),
            dtype=nl.float32,
            name=f"route_pack_tile_counts_t{tile_idx}",
        )
        nisa.tensor_reduce(
            dst=tile_counts,
            data=matches,
            op=nl.add,
            axis=1,
            name=f"route_pack_count_reduce_t{tile_idx}",
        )
        nisa.tensor_tensor(
            dst=counts,
            data1=counts,
            data2=tile_counts,
            op=nl.add,
            name=f"route_pack_count_accumulate_t{tile_idx}",
        )
        sbm.close_scope()

    # Compute ceil(count / block_size), exclusive starts, and inclusive ends.
    counts_psum = nl.ndarray((1, num_local_experts), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(
        dst=counts_psum,
        data=counts,
        name="route_pack_counts_transpose",
    )
    counts_row = sbm.alloc_stack(
        (1, num_local_experts),
        dtype=nl.int32,
        name="route_pack_counts_row",
    )
    nisa.tensor_copy(dst=counts_row, src=counts_psum, name="route_pack_counts_cast")
    rounded_counts = sbm.alloc_stack(
        (1, num_local_experts),
        dtype=nl.int32,
        name="route_pack_rounded_counts",
    )
    nisa.tensor_scalar(
        dst=rounded_counts,
        data=counts_row,
        op0=nl.add,
        operand0=block_size - 1,
        name="route_pack_count_round_up",
    )
    blocks_per_expert = sbm.alloc_stack(
        (1, num_local_experts),
        dtype=nl.int32,
        name="route_pack_blocks_per_expert",
    )
    nisa.tensor_scalar(
        dst=blocks_per_expert,
        data=rounded_counts,
        op0=nl.right_shift,
        operand0=8 if block_size == 256 else 9,
        name="route_pack_count_to_blocks",
    )
    scan_ones = sbm.alloc_stack(
        (1, num_local_experts),
        dtype=nl.int32,
        name="route_pack_block_scan_ones",
    )
    nisa.memset(dst=scan_ones, value=1, name="route_pack_init_block_scan")
    scan_zero = sbm.alloc_stack(
        (1, 1),
        dtype=nl.int32,
        name="route_pack_block_scan_zero",
    )
    nisa.memset(dst=scan_zero, value=0, name="route_pack_init_block_carry")
    block_ends = sbm.alloc_stack(
        (1, num_local_experts),
        dtype=nl.int32,
        name="route_pack_block_ends",
    )
    nisa.tensor_tensor_scan(
        dst=block_ends,
        data0=scan_ones,
        data1=blocks_per_expert,
        initial=scan_zero,
        op0=nl.multiply,
        op1=nl.add,
        name="route_pack_block_prefix",
    )
    block_starts = sbm.alloc_stack(
        (1, num_local_experts),
        dtype=nl.int32,
        name="route_pack_block_starts",
    )
    nisa.tensor_tensor(
        dst=block_starts,
        data1=block_ends,
        data2=blocks_per_expert,
        op=nl.subtract,
        name="route_pack_block_starts_subtract",
    )

    # Materialize block expert IDs and the dynamic active-block conditions.
    #
    # Every buffer here is PARTITION-MAJOR over blocks, so all of them are capped
    # at the 128-partition SBUF limit. Above 6,144 tokens per MoE call max_blocks
    # exceeds 128 and the first op to be validated fails with
    #   "iota dst partition dimension 160 exceeds maximum 128"
    # (MEASURED 2026-08-22 at BS=8/chunk1024, max_blocks=160). So tile over blocks
    # in <=128-partition chunks, exactly as _init_route_metadata already does for
    # block_to_expert. Compile-time loop with a single iteration at the block
    # counts that worked before, so those configs are unchanged.
    #
    # The condition ROW is (1, max_blocks) -- free dimension, no partition limit --
    # so it is allocated once outside the loop, filled per tile, and stored once.
    condition_row = sbm.alloc_stack(
        (1, max_blocks),
        dtype=nl.int32,
        name="route_pack_condition_row",
    )
    for _blk_lo in range(0, max_blocks, 128):
        _blk_hi = min(_blk_lo + 128, max_blocks)
        _blk_n = _blk_hi - _blk_lo
        sbm.open_scope(name=f"route_pack_block_meta_{_blk_lo}")
        block_ids = sbm.alloc_stack(
            (_blk_n, num_local_experts),
            dtype=nl.int32,
            name=f"route_pack_block_ids_{_blk_lo}",
        )
        # offset carries the tile base so block ids stay GLOBAL across tiles.
        nisa.iota(
            dst=block_ids,
            pattern=[[0, num_local_experts]],
            offset=_blk_lo,
            channel_multiplier=1,
            name=f"route_pack_block_iota_{_blk_lo}",
        )
        ends_broadcast = sbm.alloc_stack(
            (_blk_n, num_local_experts),
            dtype=nl.int32,
            name=f"route_pack_ends_broadcast_{_blk_lo}",
        )
        stream_shuffle_broadcast(block_ends, ends_broadcast)
        ended_experts = sbm.alloc_stack(
            (_blk_n, num_local_experts),
            dtype=nl.int32,
            name=f"route_pack_ended_experts_{_blk_lo}",
        )
        nisa.tensor_tensor(
            dst=ended_experts,
            data1=block_ids,
            data2=ends_broadcast,
            op=nl.greater_equal,
            name=f"route_pack_block_expert_compare_{_blk_lo}",
        )
        block_experts = sbm.alloc_stack(
            (_blk_n, 1),
            dtype=nl.int32,
            name=f"route_pack_block_experts_{_blk_lo}",
        )
        nisa.tensor_reduce(
            dst=block_experts,
            data=ended_experts,
            op=nl.add,
            axis=1,
            name=f"route_pack_block_expert_reduce_{_blk_lo}",
        )
        # Intersect this tile with the writer's block range (num_writers > 1 at
        # LNC=2); both bounds are compile-time ints, so this is a Python test.
        _w_lo = max(_blk_lo, writer_id * max_blocks // num_writers)
        _w_hi = min(_blk_hi, (writer_id + 1) * max_blocks // num_writers)
        if _w_hi > _w_lo:
            nisa.dma_copy(
                dst=block_to_expert[_w_lo:_w_hi, :],
                src=block_experts[_w_lo - _blk_lo : _w_hi - _blk_lo, :],
                name=f"route_pack_store_block_experts_{_blk_lo}",
            )

        block_id_column = sbm.alloc_stack(
            (_blk_n, 1),
            dtype=nl.int32,
            name=f"route_pack_block_id_column_{_blk_lo}",
        )
        nisa.iota(
            dst=block_id_column,
            pattern=[[0, 1]],
            offset=_blk_lo,
            channel_multiplier=1,
            name=f"route_pack_condition_iota_{_blk_lo}",
        )
        active_conditions = sbm.alloc_stack(
            (_blk_n, 1),
            dtype=nl.int32,
            name=f"route_pack_active_conditions_{_blk_lo}",
        )
        active_block_count = sbm.alloc_stack(
            (_blk_n, 1),
            dtype=nl.int32,
            name=f"route_pack_active_block_count_{_blk_lo}",
        )
        stream_shuffle_broadcast(
            block_ends[0:1, num_local_experts - 1 : num_local_experts],
            active_block_count,
        )
        nisa.tensor_tensor(
            dst=active_conditions,
            data1=block_id_column,
            data2=active_block_count,
            op=nl.less,
            name=f"route_pack_active_compare_{_blk_lo}",
        )
        condition_psum = nl.ndarray((1, _blk_n), dtype=nl.float32, buffer=nl.psum)
        condition_float = sbm.alloc_stack(
            (_blk_n, 1),
            dtype=nl.float32,
            name=f"route_pack_condition_float_{_blk_lo}",
        )
        nisa.tensor_copy(
            dst=condition_float,
            src=active_conditions,
            name=f"route_pack_condition_to_float_{_blk_lo}",
        )
        nisa.nc_transpose(
            dst=condition_psum,
            data=condition_float,
            name=f"route_pack_condition_transpose_{_blk_lo}",
        )
        nisa.tensor_copy(
            dst=condition_row[0:1, _blk_lo:_blk_hi],
            src=condition_psum,
            name=f"route_pack_condition_to_int_{_blk_lo}",
        )
        sbm.close_scope()

    condition_write_lo = writer_id * max_blocks // num_writers
    condition_write_hi = (writer_id + 1) * max_blocks // num_writers
    nisa.dma_copy(
        dst=conditions.reshape((1, max_blocks + 1))[
            0:1, condition_write_lo:condition_write_hi
        ],
        src=condition_row[0:1, condition_write_lo:condition_write_hi],
        name="route_pack_store_conditions",
    )

    # The direct four-expert nonzero path keeps a [pmax, assignments] result
    # in SBUF. It is the fastest route packer through BS=2 (16,384
    # assignments), but cannot fit at BS=4. Larger calls use the tiled stable
    # scan below, which keeps route working sets bounded by _ROUTE_TILE.
    if (
        scatter_barrier
        and assignments <= _DIRECT_ROUTE_MAX_ASSIGNMENTS
        and not _FORCE_TILED_ROUTE
    ):
        kernel_assert(
            num_writers == 1,
            "direct route packing requires one writer per metadata tensor",
        )
        kernel_assert(
            num_local_experts % _ROUTE_LANES == 0,
            f"direct route packing needs num_local_experts divisible by "
            f"{_ROUTE_LANES} lanes, found {num_local_experts}",
        )
        sbm.open_scope(name="route_pack_direct_store")
        nonzero_input = sbm.alloc_stack(
            (nl.tile_size.pmax, assignments),
            dtype=nl.int32,
            name="route_pack_nonzero_input",
            align=32,
        )
        routed_indices = sbm.alloc_stack(
            (nl.tile_size.pmax, assignments + 1),
            dtype=nl.int32,
            name="route_pack_nonzero_indices",
            align=32,
        )
        packed_row = token_position_to_id.reshape((1, packed_len))
        write_offset = sbm.alloc_stack(
            (1, 1),
            dtype=nl.int32,
            name="route_pack_direct_write_offset",
        )
        source_offset = sbm.alloc_stack(
            (1, 1),
            dtype=nl.uint32,
            name="route_pack_direct_source_offset",
            align=32,
        )
        route_block = sbm.alloc_stack(
            (1, block_size),
            dtype=nl.int32,
            name="route_pack_direct_block",
        )
        expert_block_count = sbm.alloc_stack(
            (1, 1),
            dtype=nl.int32,
            name="route_pack_direct_expert_blocks",
        )
        nisa.memset(
            dst=write_offset,
            value=0,
            name="route_pack_zero_direct_write_offset",
        )

        for expert_batch in nl.sequential_range(num_local_experts // _ROUTE_LANES):
            for expert_lane in range(_ROUTE_LANES):
                expert_idx = expert_batch * _ROUTE_LANES + expert_lane
                gpsimd_partition = expert_lane * _ROUTE_LANE_STRIDE
                nisa.dma_copy(
                    dst=nonzero_input[
                        gpsimd_partition : gpsimd_partition + 1, :
                    ],
                    src=routes_hbm,
                    name=f"route_pack_load_routes_e{expert_idx}",
                )
                nisa.tensor_scalar(
                    dst=nonzero_input[
                        gpsimd_partition : gpsimd_partition + 1, :
                    ],
                    data=nonzero_input[
                        gpsimd_partition : gpsimd_partition + 1, :
                    ],
                    op0=nl.equal,
                    operand0=expert_lo + expert_idx,
                    name=f"route_pack_direct_match_e{expert_idx}",
                )
            nisa.nonzero_with_count(
                dst=routed_indices,
                src=nonzero_input,
                index_offset=0,
                padding_val=assignments,
                name=f"route_pack_direct_nonzero_b{expert_batch}",
            )
            # Lever 4a: assignment index -> token id is the SAME constant shift for
            # every lane, and a 1-partition tensor_scalar over a [1, assignments]
            # row uses 1 of 128 Vector partitions (8.68 us to move 16,384 int32).
            # One op over the full partition dim costs about the same as one lane
            # and replaces all of them. Partitions outside the lane stride hold
            # nonzero_with_count padding that nothing reads (the only consumer is
            # the per-lane `route_pack_direct_select_block_e*` gather below, at
            # offset gpsimd_partition * (assignments + 1)), so shifting them is inert.
            nisa.tensor_scalar(
                dst=routed_indices[0 : nl.tile_size.pmax, 0:assignments],
                data=routed_indices[0 : nl.tile_size.pmax, 0:assignments],
                op0=nl.right_shift,
                operand0=3,
                name=f"route_pack_direct_assignment_to_token_b{expert_batch}",
            )
            for expert_lane in range(_ROUTE_LANES):
                expert_idx = expert_batch * _ROUTE_LANES + expert_lane
                gpsimd_partition = expert_lane * _ROUTE_LANE_STRIDE
                nisa.tensor_copy(
                    dst=expert_block_count,
                    src=blocks_per_expert[0:1, expert_idx : expert_idx + 1],
                    name=f"route_pack_direct_block_count_e{expert_idx}",
                )
                block_count_reg = nisa.register_alloc()
                nisa.register_load(
                    dst=block_count_reg,
                    src=expert_block_count,
                )
                nisa.memset(
                    dst=source_offset,
                    value=0,
                    name=f"route_pack_direct_zero_source_offset_e{expert_idx}",
                )
                for _ in nl.dynamic_range(block_count_reg):
                    nisa.tensor_copy(
                        dst=route_block,
                        src=routed_indices.ap(
                            pattern=[[assignments + 1, 1], [1, block_size]],
                            offset=gpsimd_partition * (assignments + 1),
                            scalar_offset=source_offset,
                            indirect_dim=1,
                        ),
                        name=f"route_pack_direct_select_block_e{expert_idx}",
                    )
                    nisa.dma_copy(
                        dst=packed_row.ap(
                            pattern=[[packed_len, 1], [1, block_size]],
                            offset=0,
                            scalar_offset=write_offset,
                            indirect_dim=1,
                        ),
                        src=route_block,
                        name=f"route_pack_direct_store_e{expert_idx}",
                    )
                    nisa.tensor_scalar(
                        dst=source_offset,
                        data=source_offset,
                        op0=nl.add,
                        operand0=block_size,
                        name=f"route_pack_direct_advance_source_e{expert_idx}",
                    )
                    nisa.tensor_scalar(
                        dst=write_offset,
                        data=write_offset,
                        op0=nl.add,
                        operand0=block_size,
                        name=f"route_pack_direct_advance_write_e{expert_idx}",
                    )

        sbm.close_scope()
        sbm.close_scope()
        return token_position_to_id, block_to_expert, conditions

    block_starts_float = sbm.alloc_stack(
        (1, num_local_experts),
        dtype=nl.float32,
        name="route_pack_block_starts_float",
    )
    nisa.tensor_copy(
        dst=block_starts_float,
        src=block_starts,
        name="route_pack_block_starts_cast",
    )
    block_offsets_psum = nl.ndarray(
        (num_local_experts, 1),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    nisa.nc_transpose(
        dst=block_offsets_psum,
        data=block_starts_float,
        name="route_pack_block_starts_transpose",
    )
    block_offsets = sbm.alloc_stack(
        (num_local_experts, 1),
        dtype=nl.float32,
        name="route_pack_block_offsets",
    )
    nisa.tensor_copy(
        dst=block_offsets,
        src=block_offsets_psum,
        name="route_pack_block_offsets_copy",
    )
    nisa.tensor_scalar(
        dst=block_offsets,
        data=block_offsets,
        op0=nl.multiply,
        operand0=block_size,
        name="route_pack_block_offsets_scale",
    )

    # Pass 2: stable per-expert ordinals, then unique indirect token-ID stores.
    carry = sbm.alloc_stack(
        (num_local_experts, 1),
        dtype=nl.float32,
        name="route_pack_ordinal_carry",
    )
    nisa.memset(dst=carry, value=0, name="route_pack_zero_ordinal_carry")
    ordinal_ones = sbm.alloc_stack(
        (num_local_experts, _ROUTE_TILE),
        dtype=nl.float32,
        name="route_pack_ordinal_ones",
    )
    nisa.memset(dst=ordinal_ones, value=1, name="route_pack_init_ordinal_scan")
    packed_2d = token_position_to_id.reshape((max_blocks * block_size, 1))

    # Per-assignment "pick my expert's destination out of this row" used to be a
    # nisa.local_gather. That is a GpSimd op whose result is broadcast to all 16
    # connected partitions of a core, which is why a diagonal mask used to live
    # here to undo the broadcast.
    #
    # MEASURED 2026-08-21: that local_gather is what DEADLOCKED the fused TP=8
    # graph. Ablation ladder at 4L with the tiled path forced, everything else
    # identical: nothing ablated -> hang; the indirect scatter ablated -> hang;
    # the whole slice body ablated -> completes; local_gather alone ablated ->
    # COMPLETES. It is the only tiled-exclusive GpSimd op (the direct packer uses
    # nonzero_with_count), and it hangs at ANY size -- it reproduces with a single
    # 2,048-assignment tile, so this was never about token count or volume.
    #
    # Replacement needs no gather at all: `matches` is ALREADY the one-hot we were
    # reconstructing (matches[e, a] == 1 iff assignment a routes to local expert e,
    # all-zero when it routes off-rank). Transposing it alongside `destinations`
    # turns the select into a masked row reduce -- one nc_transpose, one multiply
    # and one reduce, all on engines the working direct path already exercises.

    for tile_idx in nl.sequential_range(assignments // _ROUTE_TILE):
        sbm.open_scope(name=f"route_pack_scan_tile_{tile_idx}")
        route_values = sbm.alloc_stack(
            (num_local_experts, _ROUTE_TILE),
            dtype=nl.float32,
            name=f"route_pack_scan_routes_t{tile_idx}",
        )
        nisa.dma_copy(
            dst=route_values[0:1, :],
            src=routes_hbm[0:1, tile_idx * _ROUTE_TILE : (tile_idx + 1) * _ROUTE_TILE],
            name=f"route_pack_scan_load_t{tile_idx}",
        )
        stream_shuffle_broadcast(route_values[0:1, :], route_values)

        matches = sbm.alloc_stack(
            (num_local_experts, _ROUTE_TILE),
            dtype=nl.float32,
            name=f"route_pack_scan_matches_t{tile_idx}",
        )
        nisa.tensor_tensor(
            dst=matches,
            data1=route_values,
            data2=expert_keys,
            op=nl.equal,
            name=f"route_pack_scan_equal_t{tile_idx}",
        )
        inclusive = sbm.alloc_stack(
            (num_local_experts, _ROUTE_TILE),
            dtype=nl.float32,
            name=f"route_pack_inclusive_ordinals_t{tile_idx}",
        )
        nisa.tensor_tensor_scan(
            dst=inclusive,
            data0=ordinal_ones,
            data1=matches,
            initial=carry,
            op0=nl.multiply,
            op1=nl.add,
            name=f"route_pack_ordinal_scan_t{tile_idx}",
        )
        nisa.tensor_copy(
            dst=carry,
            src=inclusive[:, _ROUTE_TILE - 1 : _ROUTE_TILE],
            name=f"route_pack_update_ordinal_carry_t{tile_idx}",
        )
        destinations = sbm.alloc_stack(
            (num_local_experts, _ROUTE_TILE),
            dtype=nl.float32,
            name=f"route_pack_destinations_t{tile_idx}",
        )
        nisa.tensor_tensor(
            dst=destinations,
            data1=inclusive,
            data2=matches,
            op=nl.subtract,
            name=f"route_pack_zero_based_ordinals_t{tile_idx}",
        )
        nisa.tensor_scalar(
            dst=destinations,
            data=destinations,
            op0=nl.add,
            operand0=block_offsets,
            name=f"route_pack_add_block_offsets_t{tile_idx}",
        )

        # SEQUENTIAL, not affine: the slices are NOT independent. Every slice's
        # lane 0 can be forced to the shared padding slot
        # (max_blocks * block_size - 1) and every non-store is aimed one past the
        # end, so declaring independence hands the compiler a false licence to
        # overlap 16 slices' worth of indirect (DGE) scatters into the same
        # shared_hbm buffer. The per-slice `core_barrier` below was the intended
        # serialization, and it is illegal at LNC=1 (`core_barrier() requires LNC
        # degree >= 2`), so at NUM_SHARDS==1 nothing ordered them at all -- which
        # is why this path deadlocked at ANY token count while the direct packer
        # (a few contiguous scalar_offset block copies) was always fine.
        # Diagnostic "slices": skip the entire per-slice body. Pass 1 still writes
        # block_to_expert and conditions, so the MoE's block-loop trip count is
        # unchanged and only the token IDs are wrong -- a fair test of whether the
        # deadlock lives in the slice body or in the per-tile code above it.
        slice_tripcount = 0 if _ABLATE == "slices" else _ROUTE_TILE // _ASSIGNMENT_SLICE
        for slice_idx in nl.sequential_range(slice_tripcount):
            sbm.open_scope(name=f"route_pack_slice_{tile_idx}_{slice_idx}")
            slice_lo = slice_idx * _ASSIGNMENT_SLICE
            slice_hi = slice_lo + _ASSIGNMENT_SLICE

            destination_psum = nl.ndarray(
                (_ASSIGNMENT_SLICE, num_local_experts),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            nisa.nc_transpose(
                dst=destination_psum,
                data=destinations[:, slice_lo:slice_hi],
                name=f"route_pack_destination_transpose_t{tile_idx}_s{slice_idx}",
            )
            destination_rows = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, num_local_experts),
                dtype=nl.float32,
                name=f"route_pack_destination_rows_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_copy(
                dst=destination_rows,
                src=destination_psum,
                name=f"route_pack_destination_copy_t{tile_idx}_s{slice_idx}",
            )

            match_psum = nl.ndarray(
                (_ASSIGNMENT_SLICE, num_local_experts),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            nisa.nc_transpose(
                dst=match_psum,
                data=matches[:, slice_lo:slice_hi],
                name=f"route_pack_match_transpose_t{tile_idx}_s{slice_idx}",
            )
            match_rows = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, num_local_experts),
                dtype=nl.float32,
                name=f"route_pack_match_rows_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_copy(
                dst=match_rows,
                src=match_psum,
                name=f"route_pack_match_copy_t{tile_idx}_s{slice_idx}",
            )

            route_psum = nl.ndarray(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            nisa.nc_transpose(
                dst=route_psum,
                data=route_values[0:1, slice_lo:slice_hi],
                name=f"route_pack_route_transpose_t{tile_idx}_s{slice_idx}",
            )
            route_column = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.int32,
                name=f"route_pack_route_column_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_copy(
                dst=route_column,
                src=route_psum,
                name=f"route_pack_route_cast_t{tile_idx}_s{slice_idx}",
            )
            masked_destinations = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, num_local_experts),
                dtype=nl.float32,
                name=f"route_pack_masked_destinations_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_tensor(
                dst=masked_destinations,
                data1=destination_rows,
                data2=match_rows,
                op=nl.multiply,
                name=f"route_pack_mask_destination_row_t{tile_idx}_s{slice_idx}",
            )
            selected_destination = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.float32,
                name=f"route_pack_selected_destination_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_reduce(
                dst=selected_destination,
                data=masked_destinations,
                op=nl.add,
                axis=1,
                name=f"route_pack_select_ordinal_t{tile_idx}_s{slice_idx}",
            )

            in_local_range = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.int32,
                name=f"route_pack_in_local_range_t{tile_idx}_s{slice_idx}",
            )
            above_lo = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.int32,
                name=f"route_pack_above_lo_t{tile_idx}_s{slice_idx}",
            )
            below_hi = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.int32,
                name=f"route_pack_below_hi_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_scalar(
                dst=above_lo,
                data=route_column,
                op0=nl.greater_equal,
                operand0=expert_lo,
                name=f"route_pack_local_lower_compare_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_scalar(
                dst=below_hi,
                data=route_column,
                op0=nl.less,
                operand0=expert_lo + num_local_experts,
                name=f"route_pack_local_upper_compare_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_tensor(
                dst=in_local_range,
                data1=above_lo,
                data2=below_hi,
                op=nl.multiply,
                name=f"route_pack_local_range_t{tile_idx}_s{slice_idx}",
            )
            destination_above_lo = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.int32,
                name=f"route_pack_destination_above_lo_t{tile_idx}_s{slice_idx}",
            )
            destination_below_hi = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.int32,
                name=f"route_pack_destination_below_hi_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_scalar(
                dst=destination_above_lo,
                data=selected_destination,
                op0=nl.greater_equal,
                operand0=writer_id * max_blocks * block_size // num_writers,
                name=f"route_pack_destination_lower_compare_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_scalar(
                dst=destination_below_hi,
                data=selected_destination,
                op0=nl.less,
                operand0=(writer_id + 1) * max_blocks * block_size // num_writers,
                name=f"route_pack_destination_upper_compare_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_tensor(
                dst=in_local_range,
                data1=in_local_range,
                data2=destination_above_lo,
                op=nl.multiply,
                name=f"route_pack_destination_lower_mask_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_tensor(
                dst=in_local_range,
                data1=in_local_range,
                data2=destination_below_hi,
                op=nl.multiply,
                name=f"route_pack_destination_owner_t{tile_idx}_s{slice_idx}",
            )
            non_store = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.int32,
                name=f"route_pack_non_store_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_scalar(
                dst=non_store,
                data=in_local_range,
                op0=nl.subtract,
                operand0=1,
                reverse0=True,
                name=f"route_pack_non_store_mask_t{tile_idx}_s{slice_idx}",
            )
            # Every non-owned assignment is redirected to the VALID padding slot
            # carrying the padding token, so the scatter never emits an
            # out-of-range descriptor.
            #
            # The original aimed non-owned lanes one element PAST the buffer and
            # relied on oob_mode.skip to drop them, keeping a single in-range
            # descriptor alive per slice (lane 0 forced to the padding slot).
            # That makes almost every descriptor of almost every slice an OOB
            # event -- the 1,667 vector-DGE OOBs seen on this path -- and each
            # indirect-DMA event costs a notification. The direct packer emits
            # none, which is why only this path deadlocked, structurally, at any
            # token count (it reproduces at 1 tile / 256 tokens).
            #
            # Writing the padding token to the padding slot is idempotent:
            # _init_route_metadata already seeds the buffer with it, so the
            # many-writers-one-address WAW is benign.
            pad_term = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.int32,
                name=f"route_pack_pad_term_t{tile_idx}_s{slice_idx}",
            )
            destination_index = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.int32,
                name=f"route_pack_destination_index_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_copy(
                dst=destination_index,
                src=selected_destination,
                name=f"route_pack_destination_cast_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_tensor(
                dst=destination_index,
                data1=destination_index,
                data2=in_local_range,
                op=nl.multiply,
                name=f"route_pack_keep_route_destination_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_scalar(
                dst=pad_term,
                data=non_store,
                op0=nl.multiply,
                operand0=max_blocks * block_size - 1,
                name=f"route_pack_padding_destination_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_tensor(
                dst=destination_index,
                data1=destination_index,
                data2=pad_term,
                op=nl.add,
                name=f"route_pack_select_padding_destination_t{tile_idx}_s{slice_idx}",
            )

            token_ids = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.int32,
                name=f"route_pack_token_ids_t{tile_idx}_s{slice_idx}",
            )
            nisa.iota(
                dst=token_ids,
                pattern=[[0, 1]],
                offset=tile_idx * _ROUTE_TILE + slice_lo,
                channel_multiplier=1,
                name=f"route_pack_assignment_iota_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_scalar(
                dst=token_ids,
                data=token_ids,
                op0=nl.right_shift,
                operand0=3,
                name=f"route_pack_assignment_to_token_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_tensor(
                dst=token_ids,
                data1=token_ids,
                data2=in_local_range,
                op=nl.multiply,
                name=f"route_pack_keep_assignment_token_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_scalar(
                dst=pad_term,
                data=non_store,
                op0=nl.multiply,
                operand0=T,
                name=f"route_pack_padding_token_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_tensor(
                dst=token_ids,
                data1=token_ids,
                data2=pad_term,
                op=nl.add,
                name=f"route_pack_select_padding_token_t{tile_idx}_s{slice_idx}",
            )
            # Diagnostic: "scatter" keeps all the compute but writes contiguously,
            # dropping the only tiled-exclusive DMA (a vector_offset indirect
            # scatter; the direct packer uses contiguous scalar_offset copies).
            if _ABLATE == "scatter":
                scatter_dst = packed_2d[0:_ASSIGNMENT_SLICE, 0:1]
            else:
                scatter_dst = packed_2d.ap(
                    pattern=[[1, _ASSIGNMENT_SLICE], [1, 1]],
                    offset=0,
                    vector_offset=destination_index,
                    indirect_dim=0,
                )
            nisa.dma_copy(
                dst=scatter_dst,
                src=token_ids,
                # error, not skip: with non-owned lanes redirected to the padding
                # slot there is no longer any legitimate out-of-range descriptor,
                # so a fault here is a real bug rather than routine traffic.
                oob_mode=nisa.oob_mode.error,
                name=f"route_pack_scatter_token_ids_t{tile_idx}_s{slice_idx}",
            )
            if scatter_barrier:
                if num_shards == 1:
                    nisa.core_barrier(
                        expert_indices,
                        (0),
                        engine=nisa.engine.dma,
                        name=f"route_pack_scatter_barrier_t{tile_idx}_s{slice_idx}",
                    )
                else:
                    nisa.core_barrier(
                        expert_indices,
                        (0, 1),
                        engine=nisa.engine.dma,
                        name=f"route_pack_scatter_barrier_t{tile_idx}_s{slice_idx}",
                    )
            sbm.close_scope()
        sbm.close_scope()

    sbm.close_scope()
    return token_position_to_id, block_to_expert, conditions


@nki.jit
def nki_pack_local_routes_35b(
    expert_indices,
    expert_lo: int,
    num_local_experts: int,
    block_size: int,
):
    """Standalone route packer for device correctness and isolated profiling."""
    T, K = expert_indices.shape
    return _pack_local_routes_impl(
        expert_indices,
        expert_lo,
        num_local_experts,
        block_size,
        nl.shared_hbm,
        # Mirror the production wrappers instead of hardcoding True. Hardcoding it
        # made this entry unable to exercise the TILED path at all on LNC=1: the
        # tiled scan's per-slice core_barrier is illegal there
        # ("core_barrier() requires LNC degree >= 2"), so the gate silently only
        # ever tested the direct packer.
        scatter_barrier=_scatter_barrier_default(T * K),
    )


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


@nki.jit
def nki_moe_cte_routed_35b(
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,
    down_proj_weight,
    expert_indices,
    expert_lo: int,
    block_size: int,
):
    """Pack routes into internal metadata and immediately execute CTE MoE."""
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
    token_position_0 = nl.ndarray(
        (packed_len,),
        dtype=nl.int32,
        buffer=nl.shared_hbm,
        name="route_pack_token_position_shard_0",
    )
    block_to_expert_0 = nl.ndarray(
        (max_blocks, 1),
        dtype=nl.int32,
        buffer=nl.shared_hbm,
        name="route_pack_block_expert_shard_0",
    )
    conditions_0 = nl.ndarray(
        (max_blocks + 1,),
        dtype=nl.int32,
        buffer=nl.shared_hbm,
        name="route_pack_conditions_shard_0",
    )
    if num_shards == 1:
        metadata_tensors = (
            token_position_0,
            block_to_expert_0,
            conditions_0,
        )
    else:
        token_position_1 = nl.ndarray(
            (packed_len,),
            dtype=nl.int32,
            buffer=nl.shared_hbm,
            name="route_pack_token_position_shard_1",
        )
        block_to_expert_1 = nl.ndarray(
            (max_blocks, 1),
            dtype=nl.int32,
            buffer=nl.shared_hbm,
            name="route_pack_block_expert_shard_1",
        )
        conditions_1 = nl.ndarray(
            (max_blocks + 1,),
            dtype=nl.int32,
            buffer=nl.shared_hbm,
            name="route_pack_conditions_shard_1",
        )
        if shard_id == 0:
            metadata_tensors = (
                token_position_0,
                block_to_expert_0,
                conditions_0,
            )
        else:
            metadata_tensors = (
                token_position_1,
                block_to_expert_1,
                conditions_1,
            )
    token_position_to_id, block_to_expert, conditions = _pack_local_routes_impl(
        expert_indices,
        expert_lo,
        num_local_experts,
        block_size,
        nl.shared_hbm,
        metadata_tensors=metadata_tensors,
        # The direct BS=1/2 path needs its established barrier behavior.
        # At BS=4 the bounded tiled scan owns every non-padding destination
        # uniquely; keeping its per-slice core barriers can cross-synchronize
        # with a different TP rank's adjacent custom call.
        # MOE_CTE_SCATTER_BARRIER overrides the size-derived choice so the
        # barrier can be isolated from the packer path at a fixed token count.
        scatter_barrier=_scatter_barrier_default(assignments),
    )
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
