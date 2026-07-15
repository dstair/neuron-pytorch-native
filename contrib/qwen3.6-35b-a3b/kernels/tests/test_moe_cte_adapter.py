#!/usr/bin/env python3
"""CPU tests for the runtime routing metadata consumed by nkilib moe_cte."""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from moe_cte_adapter import max_packed_blocks, pack_local_routes


def expected_local_routes(indices, expert_lo, num_local_experts):
    expected = {}
    for token, row in enumerate(indices.tolist()):
        for expert in row:
            local = expert - expert_lo
            if 0 <= local < num_local_experts:
                expected.setdefault(local, []).append(token)
    return expected


def check_case(indices, expert_lo, num_local_experts, block_size):
    packed, block_experts, conditions = pack_local_routes(
        indices, expert_lo, num_local_experts, block_size
    )
    N = max_packed_blocks(indices.numel(), num_local_experts, block_size)
    assert packed.shape == (N * block_size,)
    assert block_experts.shape == (N, 1)
    assert conditions.shape == (N + 1,)
    assert conditions[-1].item() == 0

    actual = {}
    active = int(conditions.sum())
    blocks = packed.reshape(N, block_size)
    for block_idx in range(active):
        expert = int(block_experts[block_idx])
        assert 0 <= expert < num_local_experts
        actual.setdefault(expert, []).extend(int(t) for t in blocks[block_idx] if int(t) < indices.shape[0])

    assert actual == expected_local_routes(indices, expert_lo, num_local_experts)
    assert torch.all(block_experts[active:] == num_local_experts)
    assert torch.all((conditions == 0) | (conditions == 1))


def main():
    # Balanced routes, no local routes, and heavily skewed routes exercise the
    # block-boundary and terminal dummy-group behavior.
    check_case(
        torch.tensor([[0, 4], [1, 5], [2, 6], [3, 7]], dtype=torch.int32),
        expert_lo=0,
        num_local_experts=4,
        block_size=2,
    )
    check_case(
        torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32),
        expert_lo=8,
        num_local_experts=4,
        block_size=4,
    )
    check_case(
        torch.tensor([[9, 9], [9, 10], [9, 15], [9, 0], [9, 17]], dtype=torch.int32),
        expert_lo=8,
        num_local_experts=8,
        block_size=4,
    )

    torch.manual_seed(0)
    random_routes = torch.stack([torch.randperm(256)[:8] for _ in range(2048)]).to(torch.int32)
    check_case(random_routes, expert_lo=64, num_local_experts=64, block_size=512)

    # Exact block boundaries and duplicate assignments are important because the
    # NKI packer uses this implementation as its byte-for-byte oracle.
    boundary_routes = torch.full((1024, 8), 255, dtype=torch.int32)
    boundary_flat = boundary_routes.reshape(-1)
    cursor = 0
    for expert, count in enumerate((255, 256, 257, 511, 512, 513)):
        boundary_flat[cursor : cursor + count] = 64 + expert
        cursor += count
    check_case(boundary_routes, expert_lo=64, num_local_experts=64, block_size=256)
    check_case(boundary_routes, expert_lo=64, num_local_experts=64, block_size=512)

    duplicate_routes = torch.full((1024, 8), 127, dtype=torch.int32)
    duplicate_routes[:, :] = 73
    check_case(duplicate_routes, expert_lo=64, num_local_experts=64, block_size=256)
    print("moe_cte routing adapter: all CPU checks passed")


if __name__ == "__main__":
    main()
