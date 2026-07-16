#!/usr/bin/env python3
"""CPU algebra checks for decode-only tensor parallelism within experts."""

import torch
import torch.nn.functional as F


HIDDEN = 17
INTERMEDIATE = 12
NUM_EXPERTS = 16
TOP_K = 8
WORLD_SIZE = 4


def expert(hidden, gate_up, down, expert_id):
    gu = F.linear(hidden, gate_up[expert_id])
    gate, up = gu.chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down[expert_id])


def full_route(hidden, gate_up, down, selected, affinities):
    output = torch.zeros_like(hidden)
    for expert_id, affinity in zip(selected, affinities):
        output += expert(hidden, gate_up, down, expert_id) * affinity
    return output


def expert_parallel_route(
    hidden, gate_up, down, selected, affinities
):
    experts_per_rank = NUM_EXPERTS // WORLD_SIZE
    partials = []
    ownership = []
    for rank in range(WORLD_SIZE):
        expert_lo = rank * experts_per_rank
        expert_hi = expert_lo + experts_per_rank
        partial = torch.zeros_like(hidden)
        local_count = 0
        for expert_id, affinity in zip(selected, affinities):
            if expert_lo <= expert_id < expert_hi:
                partial += (
                    expert(hidden, gate_up, down, expert_id) * affinity
                )
                local_count += 1
        partials.append(partial)
        ownership.append(local_count)
    return torch.stack(partials).sum(dim=0), ownership


def tp_expert_route(hidden, gate_up, down, selected, affinities):
    width = INTERMEDIATE // WORLD_SIZE
    partials = []
    for rank in range(WORLD_SIZE):
        start = rank * width
        end = start + width
        gate = gate_up[:, start:end]
        up = gate_up[:, INTERMEDIATE + start:INTERMEDIATE + end]
        gate_up_shard = torch.cat([gate, up], dim=1)
        down_shard = down[:, :, start:end]

        partial = torch.zeros_like(hidden)
        for expert_id, affinity in zip(selected, affinities):
            partial += (
                expert(
                    hidden,
                    gate_up_shard,
                    down_shard,
                    expert_id,
                )
                * affinity
            )
        partials.append(partial)
    return torch.stack(partials).sum(dim=0)


def check_case(name, selected, expected_ownership):
    generator = torch.Generator().manual_seed(20260716)
    hidden = torch.randn(1, HIDDEN, generator=generator)
    gate_up = torch.randn(
        NUM_EXPERTS,
        2 * INTERMEDIATE,
        HIDDEN,
        generator=generator,
    )
    down = torch.randn(
        NUM_EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        generator=generator,
    )
    affinities = torch.tensor(
        [0.20, 0.18, 0.16, 0.14, 0.11, 0.09, 0.07, 0.05]
    )

    reference = full_route(
        hidden, gate_up, down, selected, affinities
    )
    expert_parallel, ownership = expert_parallel_route(
        hidden, gate_up, down, selected, affinities
    )
    tp_expert = tp_expert_route(
        hidden, gate_up, down, selected, affinities
    )

    assert ownership == expected_ownership, (name, ownership)
    torch.testing.assert_close(
        expert_parallel, reference, rtol=2e-6, atol=2e-5
    )
    torch.testing.assert_close(
        tp_expert, reference, rtol=2e-6, atol=2e-5
    )


def main():
    cases = [
        (
            "no_local_routes_on_rank_1",
            [0, 1, 2, 3, 12, 13, 14, 15],
            [4, 0, 0, 4],
        ),
        (
            "all_local_routes_on_rank_1",
            [4, 5, 6, 7, 4, 5, 6, 7],
            [0, 8, 0, 0],
        ),
        (
            "duplicate_experts",
            [0, 4, 4, 8, 12, 4, 8, 0],
            [2, 3, 2, 1],
        ),
        (
            "interleaved_slot_order",
            [15, 0, 11, 4, 14, 1, 10, 5],
            [2, 2, 2, 2],
        ),
    ]
    for case in cases:
        check_case(*case)
    print("PASS: decode TP-expert ownership, duplicates, and ordering")


if __name__ == "__main__":
    main()
