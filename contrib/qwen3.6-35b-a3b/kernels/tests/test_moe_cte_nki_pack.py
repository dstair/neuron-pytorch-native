#!/usr/bin/env python3
"""Exact NKI route-packer checks and optional fused CTE device equivalence."""

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "kernels")]

from moe_cte_adapter import pack_local_routes
from moe_cte_35b import (  # noqa: E402
    nki_moe_cte_35b,
    nki_moe_cte_routed_35b,
    nki_pack_local_routes_35b,
)
from topology_35b import LNC_DEGREE


def route_cases(tokens: int, expert_lo: int, num_local_experts: int):
    assignments = tokens * 8
    outside = 0 if expert_lo else 255
    generator = torch.Generator().manual_seed(1000 + tokens + expert_lo)

    yield "random", torch.randint(
        256, (tokens, 8), generator=generator, dtype=torch.int32
    )
    yield "no_local", torch.full((tokens, 8), outside, dtype=torch.int32)
    yield "one_expert_skew", torch.full(
        (tokens, 8),
        expert_lo + min(17, num_local_experts - 1),
        dtype=torch.int32,
    )

    empty_experts = torch.full((tokens, 8), outside, dtype=torch.int32)
    empty_flat = empty_experts.reshape(-1)
    sparse_experts = (
        expert_lo,
        expert_lo + min(9, num_local_experts - 1),
        expert_lo + num_local_experts - 1,
    )
    for assignment in range(assignments):
        if assignment % 5:
            empty_flat[assignment] = sparse_experts[assignment % len(sparse_experts)]
    yield "empty_experts", empty_experts

    duplicates = torch.full((tokens, 8), outside, dtype=torch.int32)
    duplicate_ids = (
        torch.arange(tokens, dtype=torch.int32) % num_local_experts + expert_lo
    ).unsqueeze(1)
    duplicates[:] = duplicate_ids
    yield "duplicate_routes", duplicates

    boundaries = torch.full((tokens, 8), outside, dtype=torch.int32)
    boundary_flat = boundaries.reshape(-1)
    cursor = 0
    for expert, count in enumerate((255, 256, 257, 511, 512, 513)):
        if cursor + count > assignments:
            break
        boundary_flat[cursor : cursor + count] = expert_lo + expert
        cursor += count
    yield "block_boundaries", boundaries


def assert_metadata_equal(actual, expected, context: str):
    for name, got, want in zip(
        ("token_position_to_id", "block_to_expert", "conditions"),
        actual,
        expected,
    ):
        got_cpu = torch.as_tensor(np.asarray(got)).cpu()
        want_cpu = want.cpu()
        if not torch.equal(got_cpu, want_cpu):
            mismatch = (got_cpu != want_cpu).reshape(-1).nonzero()[0].item()
            raise AssertionError(
                f"{context} {name} differs at flat index {mismatch}: "
                f"got={got_cpu.reshape(-1)[mismatch].item()} "
                f"expected={want_cpu.reshape(-1)[mismatch].item()}"
            )


def run_metadata_matrix(backend: str):
    num_local_experts = 32 if LNC_DEGREE == 1 else 64
    if backend == "simulation":
        import nki

        def invoke(routes, expert_lo, block_size):
            return nki.simulate(nki_pack_local_routes_35b)[1](
                routes.numpy(), expert_lo, num_local_experts, block_size
            )

    else:
        from torch_neuronx import wrap_nki

        device = torch.device("neuron:0")
        kernel = wrap_nki(nki_pack_local_routes_35b)[LNC_DEGREE]

        def invoke(routes, expert_lo, block_size):
            outputs = kernel(
                routes.to(device),
                expert_lo,
                num_local_experts,
                block_size,
            )
            torch.neuron.synchronize()
            return tuple(output.cpu() for output in outputs)

    checked = 0
    for tokens in (1024, 2048):
        for expert_lo in range(0, 256, num_local_experts):
            for block_size in (256, 512):
                for case_name, routes in route_cases(
                    tokens, expert_lo, num_local_experts
                ):
                    expected = pack_local_routes(
                        routes, expert_lo, num_local_experts, block_size
                    )
                    actual = invoke(routes, expert_lo, block_size)
                    context = (
                        f"case={case_name} T={tokens} expert_lo={expert_lo} "
                        f"block_size={block_size}"
                    )
                    assert_metadata_equal(actual, expected, context)
                    checked += 1
                    print(f"PASS {context}")
    print(
        f"NKI route packer: {checked} exact metadata cases passed "
        f"via {backend}, LNC={LNC_DEGREE}, local_experts={num_local_experts}"
    )


def run_fused_equivalence():
    """Compare the fused route+CTE call with precomputed metadata on Trn2."""
    from torch_neuronx import wrap_nki

    tokens = 256
    hidden_size = 512
    intermediate_size = 512
    num_local_experts = 32 if LNC_DEGREE == 1 else 64
    expert_lo = num_local_experts
    block_size = 256
    generator = torch.Generator().manual_seed(37)

    routes = (
        torch.arange(tokens * 8, dtype=torch.int32).reshape(tokens, 8) % 8
        + expert_lo
    )
    metadata = pack_local_routes(
        routes, expert_lo, num_local_experts, block_size
    )
    hidden = torch.randn(
        tokens + 1, hidden_size, generator=generator, dtype=torch.bfloat16
    )
    affinities = torch.randn(
        (tokens + 1) * num_local_experts,
        1,
        generator=generator,
        dtype=torch.bfloat16,
    )
    gate_up = torch.randn(
        num_local_experts,
        hidden_size,
        2,
        intermediate_size,
        generator=generator,
        dtype=torch.bfloat16,
    )
    down = torch.randn(
        num_local_experts,
        intermediate_size,
        hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
    )

    device = torch.device("neuron:0")
    hidden_d, affinities_d, gate_up_d, down_d, routes_d = (
        tensor.to(device)
        for tensor in (hidden, affinities, gate_up, down, routes)
    )
    metadata_d = tuple(tensor.to(device) for tensor in metadata)
    precomputed = wrap_nki(nki_moe_cte_35b)[LNC_DEGREE]
    fused = wrap_nki(nki_moe_cte_routed_35b)[LNC_DEGREE]

    expected = precomputed(
        hidden_d,
        affinities_d,
        gate_up_d,
        down_d,
        *metadata_d,
        block_size,
    )
    torch.neuron.synchronize()
    expected_cpu = expected.cpu()
    print("precomputed metadata CTE path completed")

    for iteration in range(3):
        actual = fused(
            hidden_d,
            affinities_d,
            gate_up_d,
            down_d,
            routes_d,
            expert_lo,
            block_size,
        )
        torch.neuron.synchronize()
        torch.testing.assert_close(actual.cpu(), expected_cpu, rtol=0, atol=0)
        print(
            f"fused routed CTE iteration {iteration}: "
            "exact match with precomputed metadata path"
        )


def run_distributed_equivalence(
    mode: str,
    iterations: int,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    block_size: int,
    routes_from_topk: bool,
    routes_dir: str = None,
    inputs_dir: str = None,
):
    """Exercise route packing and CTE concurrently on every TP rank."""
    import torch.distributed as dist
    from torch_neuronx import wrap_nki

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (4, 8):
        raise RuntimeError(
            f"distributed route-pack check requires world size 4 or 8, found {world_size}"
        )
    expected_world_size = 8 if LNC_DEGREE == 1 else 4
    if world_size != expected_world_size:
        raise RuntimeError(
            f"LNC={LNC_DEGREE} route-pack check expects TP={expected_world_size}, "
            f"found TP={world_size}"
        )

    num_local_experts = 256 // world_size
    expert_lo = rank * num_local_experts
    generator = torch.Generator().manual_seed(3700 + rank)
    route_logits = None
    captured_inputs = None
    if inputs_dir:
        captured_inputs = torch.load(
            pathlib.Path(inputs_dir) / f"inputs_rank{rank}_call0.pt",
            map_location="cpu",
            weights_only=True,
        )
        expert_lo = int(captured_inputs["expert_lo"])
        captured_block_size = int(captured_inputs["block_size"])
        if captured_block_size != block_size:
            raise ValueError(
                f"rank {rank} captured block size {captured_block_size} "
                f"does not match requested {block_size}"
            )
        routes = captured_inputs["routes"].to(torch.int32)
        hidden = captured_inputs["hidden"]
        affinities = captured_inputs["affinities"]
        gate_up = captured_inputs["gate_up"]
        down = captured_inputs["down"]
        if routes.shape != (tokens, 8):
            raise ValueError(
                f"rank {rank} captured routes have shape {tuple(routes.shape)}, "
                f"expected {(tokens, 8)}"
            )
    elif routes_dir:
        routes = torch.load(
            pathlib.Path(routes_dir) / f"routes_rank{rank}_call0.pt",
            map_location="cpu",
            weights_only=True,
        ).to(torch.int32)
        if routes.shape != (tokens, 8):
            raise ValueError(
                f"rank {rank} captured routes have shape {tuple(routes.shape)}, "
                f"expected {(tokens, 8)}"
            )
    else:
        route_logits = torch.randn(
            tokens, 256, generator=generator, dtype=torch.float32
        )
        routes = torch.topk(route_logits, 8, dim=-1).indices.to(torch.int32)
    if captured_inputs is not None:
        captured_experts = down.shape[0]
        if captured_experts != num_local_experts:
            raise ValueError(
                f"rank {rank} captured {captured_experts} local experts, "
                f"expected {num_local_experts}"
            )
    metadata = pack_local_routes(
        routes, expert_lo, num_local_experts, block_size
    )
    if captured_inputs is None:
        hidden = torch.randn(
            tokens + 1, hidden_size, generator=generator, dtype=torch.bfloat16
        )
        affinities = torch.randn(
            (tokens + 1) * num_local_experts,
            1,
            generator=generator,
            dtype=torch.bfloat16,
        )
        gate_up = torch.randn(
            num_local_experts,
            hidden_size,
            2,
            intermediate_size,
            generator=generator,
            dtype=torch.bfloat16,
        )
        down = torch.randn(
            num_local_experts,
            intermediate_size,
            hidden_size,
            generator=generator,
            dtype=torch.bfloat16,
        )

    device = torch.neuron.current_device()
    hidden_d, affinities_d, gate_up_d, down_d = (
        tensor.to(device)
        for tensor in (hidden, affinities, gate_up, down)
    )
    if routes_from_topk:
        if route_logits is None:
            raise ValueError("--routes-from-topk cannot be used with --routes-dir")
        routes_d = torch.topk(route_logits.to(device), 8, dim=-1).indices.to(
            torch.int32
        )
    else:
        routes_d = routes.to(device)
    metadata_d = tuple(tensor.to(device) for tensor in metadata)
    pack = wrap_nki(nki_pack_local_routes_35b)[LNC_DEGREE]
    precomputed = wrap_nki(nki_moe_cte_35b)[LNC_DEGREE]
    fused = wrap_nki(nki_moe_cte_routed_35b)[LNC_DEGREE]

    expected_cpu = None
    if mode in ("precomputed", "both"):
        expected = precomputed(
            hidden_d,
            affinities_d,
            gate_up_d,
            down_d,
            *metadata_d,
            block_size,
        )
        torch.neuron.synchronize()
        expected_cpu = expected.cpu()
        print(f"rank={rank} precomputed CTE completed", flush=True)
        dist.barrier()

    for iteration in range(iterations):
        if mode == "pack":
            actual = pack(
                routes_d,
                expert_lo,
                num_local_experts,
                block_size,
            )
            torch.neuron.synchronize()
            assert_metadata_equal(
                tuple(output.cpu() for output in actual),
                metadata,
                f"rank={rank} iteration={iteration}",
            )
        elif mode == "precomputed":
            actual = precomputed(
                hidden_d,
                affinities_d,
                gate_up_d,
                down_d,
                *metadata_d,
                block_size,
            )
            torch.neuron.synchronize()
            torch.testing.assert_close(actual.cpu(), expected_cpu, rtol=0, atol=0)
        else:
            actual = fused(
                hidden_d,
                affinities_d,
                gate_up_d,
                down_d,
                routes_d,
                expert_lo,
                block_size,
            )
            torch.neuron.synchronize()
            if expected_cpu is not None:
                torch.testing.assert_close(
                    actual.cpu(), expected_cpu, rtol=0, atol=0
                )
        print(f"rank={rank} {mode} iteration={iteration} completed", flush=True)
        dist.barrier()

    if rank == 0:
        print(f"distributed {mode}: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("simulation", "device"), default="simulation"
    )
    parser.add_argument(
        "--fused",
        action="store_true",
        help="also run fused CTE output equivalence (device backend only)",
    )
    parser.add_argument(
        "--distributed",
        choices=("pack", "precomputed", "fused", "both"),
        help="run only the selected concurrent TP=4/LNC2 or TP=8/LNC1 check",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--block-size", type=int, choices=(256, 512), default=256)
    parser.add_argument("--routes-from-topk", action="store_true")
    parser.add_argument("--routes-dir")
    parser.add_argument(
        "--inputs-dir",
        help="replay exact CTE inputs captured by test_bucketed_prefill_batched_device.py",
    )
    args = parser.parse_args()
    if args.fused and args.backend != "device":
        parser.error("--fused requires --backend device")
    if args.distributed and args.backend != "device":
        parser.error("--distributed requires --backend device")

    if args.distributed:
        run_distributed_equivalence(
            args.distributed,
            args.iterations,
            args.tokens,
            args.hidden_size,
            args.intermediate_size,
            args.block_size,
            args.routes_from_topk,
            args.routes_dir,
            args.inputs_dir,
        )
        return

    run_metadata_matrix(args.backend)
    if args.fused:
        run_fused_equivalence()


if __name__ == "__main__":
    main()
