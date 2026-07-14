"""Fixed-shape routing metadata for nkilib's context-encoding MoE kernel."""

import math

import torch
import torch.nn.functional as F


def max_packed_blocks(num_assignments: int, num_local_experts: int, block_size: int) -> int:
    """Maximum blocks after grouping assignments into local experts plus a dummy."""
    num_groups = num_local_experts + 1
    if num_assignments < num_groups:
        return num_assignments
    return math.ceil((num_assignments - (num_groups - 1)) / block_size) + num_groups - 1


def pack_local_routes(
    expert_indices: torch.Tensor,
    expert_lo: int,
    num_local_experts: int,
    block_size: int,
):
    """Pack global top-k routes into fixed-size, local-expert blocks.

    Out-of-rank assignments are placed in a final dummy group. Since groups are
    ordered, all local blocks precede dummy blocks; ``conditions.sum()`` is
    therefore the exact dynamic-loop trip count for ``moe_cte``.

    Returns:
        token_position_to_id: int32 [N * block_size], padded with token id T.
        block_to_expert: int32 [N, 1], dummy/trailing blocks use expert E.
        conditions: int32 [N + 1], one for each real local block.
    """
    if expert_indices.ndim != 2:
        raise ValueError("expert_indices must have shape [T, K]")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    T, K = expert_indices.shape
    E = num_local_experts
    assignments = T * K
    max_blocks = max_packed_blocks(assignments, E, block_size)

    global_flat = expert_indices.reshape(-1).to(torch.int64)
    local_flat = global_flat - expert_lo
    is_local = (local_flat >= 0) & (local_flat < E)
    group = torch.where(is_local, local_flat, torch.full_like(local_flat, E))

    group_hot = F.one_hot(group, E + 1).to(torch.int32)
    ordinal = (group_hot.cumsum(dim=0) - 1).gather(1, group[:, None]).squeeze(1)
    counts = group_hot.sum(dim=0)
    blocks_per_group = torch.div(counts + block_size - 1, block_size, rounding_mode="floor")
    block_ends = blocks_per_group.cumsum(dim=0)
    block_starts = block_ends - blocks_per_group

    destinations = block_starts.gather(0, group) * block_size + ordinal
    token_ids = torch.arange(T, dtype=torch.int32, device=expert_indices.device).repeat_interleave(K)
    token_ids = torch.where(is_local, token_ids, torch.full_like(token_ids, T))

    token_position_to_id = torch.full(
        (max_blocks * block_size,), T, dtype=torch.int32, device=expert_indices.device
    )
    token_position_to_id.scatter_(0, destinations.to(torch.int64), token_ids)

    block_ids = torch.arange(max_blocks, dtype=torch.int32, device=expert_indices.device)
    block_to_expert = (block_ids[:, None] >= block_ends[None, :]).sum(dim=1).clamp_max(E)
    local_block_count = block_ends[E - 1]
    conditions = torch.cat(
        [(block_ids < local_block_count).to(torch.int32), torch.zeros(1, dtype=torch.int32, device=block_ids.device)]
    )
    return token_position_to_id, block_to_expert.to(torch.int32)[:, None], conditions
