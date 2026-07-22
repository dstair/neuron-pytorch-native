#!/usr/bin/env python3
"""Production-shape correctness and latency gate for row-scaled FP8 MoE TKG."""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F


KERNELS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(KERNELS)
sys.path.insert(0, KERNELS)
sys.path.insert(0, ROOT)

from moe_w8 import (  # noqa: E402
    fused_moe_row_fp8_cpu,
    OfficialFP8ExpertReader,
)


def metrics(actual, expected):
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    diff = actual_f32 - expected_f32
    cosine = F.cosine_similarity(
        actual_f32.reshape(-1), expected_f32.reshape(-1), dim=0
    ).item()
    nrmse = (
        torch.linalg.vector_norm(diff)
        / torch.linalg.vector_norm(expected_f32).clamp_min(1e-12)
    ).item()
    return cosine, nrmse, diff.abs().max().item()


def official_reference(hidden, weights, affinities):
    output = torch.zeros(
        hidden.shape[0], hidden.shape[1], dtype=torch.float32
    )
    for expert in range(weights["gate_up"].shape[0]):
        gate, up = weights["gate_up"][expert].chunk(2, dim=0)
        activated = F.silu(hidden.float() @ gate.float().transpose(0, 1))
        activated *= hidden.float() @ up.float().transpose(0, 1)
        activated = activated.to(torch.bfloat16).float()
        expert_output = activated @ weights["down"][expert].float().transpose(
            0, 1
        )
        output += (
            expert_output
            * affinities[:, expert : expert + 1].to(torch.bfloat16).float()
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--expert-count", type=int, default=32)
    parser.add_argument(
        "--fp8-projections",
        choices=("all", "gate_up", "down", "none"),
        default="all",
    )
    parser.add_argument("--benchmark-iters", type=int, default=20)
    args = parser.parse_args()

    import torch_neuronx  # noqa: F401
    import moe_tkg_row_fp8_35b_ops  # noqa: F401

    torch.manual_seed(1234 + args.batch_size)
    reader = OfficialFP8ExpertReader(args.expert_model_path)
    try:
        expert_end = args.expert + args.expert_count
        converted, stats = reader.load_layer_row_fp8(
            args.layer,
            args.expert,
            expert_end,
            fp8_projections=args.fp8_projections,
        )
        official = reader.load_layer_bf16(
            args.layer, args.expert, expert_end
        )
    finally:
        reader.close()

    hidden_size = converted["row_gate_up"].shape[1]
    hidden = torch.randn(args.batch_size, hidden_size).to(torch.bfloat16)
    selected = torch.randint(
        0, args.expert_count, (args.batch_size, min(8, args.expert_count))
    )
    selected_weights = torch.rand_like(selected, dtype=torch.float32)
    selected_weights /= selected_weights.sum(dim=1, keepdim=True)
    affinities = torch.zeros(
        args.batch_size, args.expert_count, dtype=torch.float32
    )
    affinities.scatter_add_(1, selected, selected_weights)
    affinities = affinities.to(torch.bfloat16)

    expected = fused_moe_row_fp8_cpu(
        hidden,
        converted["row_gate_up"],
        converted["row_down"],
        converted["row_gate_up_scale"],
        converted["row_down_scale"],
        affinities,
    )
    source_expected = official_reference(hidden, official, affinities)
    source_cosine, source_nrmse, source_max = metrics(
        expected, source_expected
    )
    print(
        f"source: weight_cosine={stats.cosine:.7f} "
        f"weight_nrmse={stats.normalized_rmse:.5%} "
        f"fp8_projections={args.fp8_projections} "
        f"output_cosine={source_cosine:.7f} "
        f"output_nrmse={source_nrmse:.5%} max_abs={source_max:.6f}"
    )

    device = torch.neuron.current_device()
    expert_index = selected.to(torch.int32)
    rank_id = torch.zeros((1, 1), dtype=torch.uint32)
    device_args = tuple(
        tensor.to(device)
        for tensor in (
            hidden,
            converted["row_gate_up"],
            converted["row_down"],
            converted["row_gate_up_scale"],
            converted["row_down_scale"],
            affinities,
            expert_index,
            rank_id,
        )
    )
    op = torch.ops.moe_w8.tkg_row_fp8
    actual = op(*device_args).cpu()
    cosine, nrmse, max_abs = metrics(actual, expected)
    print(
        f"kernel: BS={args.batch_size} "
        f"E={args.expert_count} fp8_projections={args.fp8_projections} "
        f"cosine={cosine:.7f} "
        f"nrmse={nrmse:.5%} max_abs={max_abs:.6f}"
    )
    assert cosine >= 0.999
    assert nrmse <= 0.05

    if args.benchmark_iters:
        for _ in range(3):
            benchmark_output = op(*device_args)
        torch.neuron.synchronize()
        start = time.perf_counter()
        for _ in range(args.benchmark_iters):
            benchmark_output = op(*device_args)
        torch.neuron.synchronize()
        elapsed = time.perf_counter() - start
        print(
            f"BENCHMARK: {elapsed * 1000 / args.benchmark_iters:.3f} "
            f"ms/call over {args.benchmark_iters} iterations"
        )
        del benchmark_output


if __name__ == "__main__":
    main()
