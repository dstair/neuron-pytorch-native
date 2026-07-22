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
    block_init = sbm.alloc_stack(
        (block_hi - block_lo, 1),
        dtype=nl.int32,
        name="route_pack_block_init",
    )
    nisa.memset(
        dst=block_init,
        value=num_local_experts,
        name="route_pack_init_block_experts",
    )
    nisa.dma_copy(
        dst=block_to_expert[block_lo:block_hi, :],
        src=block_init,
        name="route_pack_store_block_experts_init",
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
    block_ids = sbm.alloc_stack(
        (max_blocks, num_local_experts),
        dtype=nl.int32,
        name="route_pack_block_ids",
    )
    nisa.iota(
        dst=block_ids,
        pattern=[[0, num_local_experts]],
        offset=0,
        channel_multiplier=1,
        name="route_pack_block_iota",
    )
    ends_broadcast = sbm.alloc_stack(
        (max_blocks, num_local_experts),
        dtype=nl.int32,
        name="route_pack_ends_broadcast",
    )
    stream_shuffle_broadcast(block_ends, ends_broadcast)
    ended_experts = sbm.alloc_stack(
        (max_blocks, num_local_experts),
        dtype=nl.int32,
        name="route_pack_ended_experts",
    )
    nisa.tensor_tensor(
        dst=ended_experts,
        data1=block_ids,
        data2=ends_broadcast,
        op=nl.greater_equal,
        name="route_pack_block_expert_compare",
    )
    block_experts = sbm.alloc_stack(
        (max_blocks, 1),
        dtype=nl.int32,
        name="route_pack_block_experts",
    )
    nisa.tensor_reduce(
        dst=block_experts,
        data=ended_experts,
        op=nl.add,
        axis=1,
        name="route_pack_block_expert_reduce",
    )
    block_write_lo = writer_id * max_blocks // num_writers
    block_write_hi = (writer_id + 1) * max_blocks // num_writers
    nisa.dma_copy(
        dst=block_to_expert[block_write_lo:block_write_hi, :],
        src=block_experts[block_write_lo:block_write_hi, :],
        name="route_pack_store_block_experts",
    )

    block_id_column = sbm.alloc_stack(
        (max_blocks, 1),
        dtype=nl.int32,
        name="route_pack_block_id_column",
    )
    nisa.iota(
        dst=block_id_column,
        pattern=[[0, 1]],
        offset=0,
        channel_multiplier=1,
        name="route_pack_condition_iota",
    )
    active_conditions = sbm.alloc_stack(
        (max_blocks, 1),
        dtype=nl.int32,
        name="route_pack_active_conditions",
    )
    active_block_count = sbm.alloc_stack(
        (max_blocks, 1),
        dtype=nl.int32,
        name="route_pack_active_block_count",
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
        name="route_pack_active_compare",
    )
    condition_psum = nl.ndarray((1, max_blocks), dtype=nl.float32, buffer=nl.psum)
    condition_float = sbm.alloc_stack(
        (max_blocks, 1),
        dtype=nl.float32,
        name="route_pack_condition_float",
    )
    nisa.tensor_copy(
        dst=condition_float,
        src=active_conditions,
        name="route_pack_condition_to_float",
    )
    nisa.nc_transpose(
        dst=condition_psum,
        data=condition_float,
        name="route_pack_condition_transpose",
    )
    condition_row = sbm.alloc_stack(
        (1, max_blocks),
        dtype=nl.int32,
        name="route_pack_condition_row",
    )
    nisa.tensor_copy(
        dst=condition_row,
        src=condition_psum,
        name="route_pack_condition_to_int",
    )
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
    if scatter_barrier and assignments <= _DIRECT_ROUTE_MAX_ASSIGNMENTS:
        kernel_assert(
            num_writers == 1,
            "direct route packing requires one writer per metadata tensor",
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

        for expert_batch in nl.sequential_range(num_local_experts // 4):
            for expert_lane in range(4):
                expert_idx = expert_batch * 4 + expert_lane
                gpsimd_partition = expert_lane * 32
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
            for expert_lane in range(4):
                expert_idx = expert_batch * 4 + expert_lane
                gpsimd_partition = expert_lane * 32
                nisa.tensor_scalar(
                    dst=routed_indices[
                        gpsimd_partition : gpsimd_partition + 1, 0:assignments
                    ],
                    data=routed_indices[
                        gpsimd_partition : gpsimd_partition + 1, 0:assignments
                    ],
                    op0=nl.right_shift,
                    operand0=3,
                    name=f"route_pack_direct_assignment_to_token_e{expert_idx}",
                )
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

    # local_gather returns every core's 16 indices to all 16 connected
    # partitions. Select the diagonal to recover one value per assignment.
    gather_partition_lanes = sbm.alloc_stack(
        (_ASSIGNMENT_SLICE, 16),
        dtype=nl.int32,
        name="route_pack_gather_partition_lanes",
    )
    nisa.iota(
        dst=gather_partition_lanes,
        pattern=[[0, 16]],
        offset=0,
        channel_multiplier=1,
        name="route_pack_gather_partition_iota",
    )
    nisa.tensor_scalar(
        dst=gather_partition_lanes,
        data=gather_partition_lanes,
        op0=nl.bitwise_and,
        operand0=15,
        name="route_pack_gather_partition_mod",
    )
    gather_free_lanes = sbm.alloc_stack(
        (_ASSIGNMENT_SLICE, 16),
        dtype=nl.int32,
        name="route_pack_gather_free_lanes",
    )
    nisa.iota(
        dst=gather_free_lanes,
        pattern=[[1, 16]],
        offset=0,
        channel_multiplier=0,
        name="route_pack_gather_free_iota",
    )
    gather_diagonal = sbm.alloc_stack(
        (_ASSIGNMENT_SLICE, 16),
        dtype=nl.float32,
        name="route_pack_gather_diagonal",
    )
    nisa.tensor_tensor(
        dst=gather_diagonal,
        data1=gather_partition_lanes,
        data2=gather_free_lanes,
        op=nl.equal,
        name="route_pack_gather_diagonal_compare",
    )
    assignment_lanes = sbm.alloc_stack(
        (_ASSIGNMENT_SLICE, 1),
        dtype=nl.int32,
        name="route_pack_assignment_lanes",
    )
    nisa.iota(
        dst=assignment_lanes,
        pattern=[[0, 1]],
        offset=0,
        channel_multiplier=1,
        name="route_pack_assignment_lane_iota",
    )
    first_assignment_lane = sbm.alloc_stack(
        (_ASSIGNMENT_SLICE, 1),
        dtype=nl.int32,
        name="route_pack_first_assignment_lane",
    )
    nisa.tensor_scalar(
        dst=first_assignment_lane,
        data=assignment_lanes,
        op0=nl.equal,
        operand0=0,
        name="route_pack_first_assignment_compare",
    )

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

        for slice_idx in nl.affine_range(_ROUTE_TILE // _ASSIGNMENT_SLICE):
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
            local_index = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.int32,
                name=f"route_pack_local_index_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_scalar(
                dst=local_index,
                data=route_column,
                op0=nl.subtract,
                operand0=expert_lo,
                op1=nl.maximum,
                operand1=0,
                name=f"route_pack_local_index_lower_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_scalar(
                dst=local_index,
                data=local_index,
                op0=nl.minimum,
                operand0=num_local_experts - 1,
                name=f"route_pack_local_index_upper_t{tile_idx}_s{slice_idx}",
            )
            local_index_u16 = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.uint16,
                name=f"route_pack_local_index_u16_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_copy(
                dst=local_index_u16,
                src=local_index,
                name=f"route_pack_local_index_cast_t{tile_idx}_s{slice_idx}",
            )
            gathered_destinations = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 16),
                dtype=nl.float32,
                name=f"route_pack_gathered_destinations_t{tile_idx}_s{slice_idx}",
            )
            nisa.local_gather(
                dst=gathered_destinations,
                src_buffer=destination_rows,
                index=local_index_u16,
                num_elem_per_idx=1,
                num_valid_indices=_ASSIGNMENT_SLICE // 8,
                name=f"route_pack_select_ordinal_t{tile_idx}_s{slice_idx}",
            )
            diagonal_destinations = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 16),
                dtype=nl.float32,
                name=f"route_pack_diagonal_destinations_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_tensor(
                dst=diagonal_destinations,
                data1=gathered_destinations,
                data2=gather_diagonal,
                op=nl.multiply,
                name=f"route_pack_mask_gather_diagonal_t{tile_idx}_s{slice_idx}",
            )
            selected_destination = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.float32,
                name=f"route_pack_selected_destination_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_reduce(
                dst=selected_destination,
                data=diagonal_destinations,
                op=nl.add,
                axis=1,
                name=f"route_pack_reduce_gather_diagonal_t{tile_idx}_s{slice_idx}",
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
            force_padding = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.int32,
                name=f"route_pack_force_padding_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_tensor(
                dst=force_padding,
                data1=non_store,
                data2=first_assignment_lane,
                op=nl.multiply,
                name=f"route_pack_force_padding_lane_t{tile_idx}_s{slice_idx}",
            )
            store_mask = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.int32,
                name=f"route_pack_store_mask_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_tensor(
                dst=store_mask,
                data1=in_local_range,
                data2=force_padding,
                op=nl.add,
                name=f"route_pack_store_or_padding_t{tile_idx}_s{slice_idx}",
            )
            invalid_offset = sbm.alloc_stack(
                (_ASSIGNMENT_SLICE, 1),
                dtype=nl.int32,
                name=f"route_pack_invalid_offset_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_scalar(
                dst=invalid_offset,
                data=store_mask,
                op0=nl.subtract,
                operand0=1,
                reverse0=True,
                op1=nl.multiply,
                operand1=max_blocks * block_size,
                name=f"route_pack_invalid_destination_t{tile_idx}_s{slice_idx}",
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
            nisa.tensor_scalar(
                dst=store_mask,
                data=force_padding,
                op0=nl.subtract,
                operand0=1,
                reverse0=True,
                name=f"route_pack_not_forced_padding_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_tensor(
                dst=destination_index,
                data1=destination_index,
                data2=store_mask,
                op=nl.multiply,
                name=f"route_pack_keep_route_destination_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_scalar(
                dst=non_store,
                data=force_padding,
                op0=nl.multiply,
                operand0=max_blocks * block_size - 1,
                name=f"route_pack_padding_destination_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_tensor(
                dst=destination_index,
                data1=destination_index,
                data2=non_store,
                op=nl.add,
                name=f"route_pack_select_padding_destination_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_tensor(
                dst=destination_index,
                data1=destination_index,
                data2=invalid_offset,
                op=nl.add,
                name=f"route_pack_mask_destination_t{tile_idx}_s{slice_idx}",
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
                data2=store_mask,
                op=nl.multiply,
                name=f"route_pack_keep_assignment_token_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_scalar(
                dst=non_store,
                data=force_padding,
                op0=nl.multiply,
                operand0=T,
                name=f"route_pack_padding_token_t{tile_idx}_s{slice_idx}",
            )
            nisa.tensor_tensor(
                dst=token_ids,
                data1=token_ids,
                data2=non_store,
                op=nl.add,
                name=f"route_pack_select_padding_token_t{tile_idx}_s{slice_idx}",
            )
            nisa.dma_copy(
                dst=packed_2d.ap(
                    pattern=[[1, _ASSIGNMENT_SLICE], [1, 1]],
                    offset=0,
                    vector_offset=destination_index,
                    indirect_dim=0,
                ),
                src=token_ids,
                oob_mode=nisa.oob_mode.skip,
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
    return _pack_local_routes_impl(
        expert_indices,
        expert_lo,
        num_local_experts,
        block_size,
        nl.shared_hbm,
        scatter_barrier=True,
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
        scatter_barrier=assignments <= _DIRECT_ROUTE_MAX_ASSIGNMENTS,
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
