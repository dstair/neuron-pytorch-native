#!/usr/bin/env python3
"""Evaluate one shared legacy-E4M3 scale per 128x512 weight slab."""

import argparse
from collections import Counter
import os
import sys

import torch
import torch.nn.functional as F


KERNELS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(KERNELS)
sys.path.insert(0, ROOT)

from moe_w8 import (  # noqa: E402
    OfficialFP8ExpertReader,
    QuantizationStats,
    decode_legacy_e4m3,
    dequantize_official_fp8,
    encode_legacy_e4m3,
)


FACTORS = tuple(round(0.70 + index * 0.05, 2) for index in range(7))
K_TILE = 128
N_TILE = 512


def quantize_group512(weight):
    """Quantize NKI-layout [K,N] weights with one BF16 scale per tile."""
    if weight.ndim != 2:
        raise ValueError("group512 input must be a matrix")
    rows, cols = weight.shape
    if rows % K_TILE or cols % N_TILE:
        raise ValueError(
            f"group512 requires multiples of {K_TILE}x{N_TILE}, "
            f"got {tuple(weight.shape)}"
        )

    quantized = torch.empty_like(weight, dtype=torch.int8)
    scales = torch.empty(
        rows // K_TILE,
        cols // N_TILE,
        dtype=torch.bfloat16,
    )
    stats = QuantizationStats()
    factor_counts = Counter()

    for row_block, row_start in enumerate(range(0, rows, K_TILE)):
        for col_block, col_start in enumerate(range(0, cols, N_TILE)):
            source = weight[
                row_start : row_start + K_TILE,
                col_start : col_start + N_TILE,
            ].float()
            absmax_scale = (
                source.abs().amax().clamp_min(1e-12) / 240.0
            )
            best_error = float("inf")
            best_factor = None
            best_scale = None
            best_quantized = None
            best_reconstructed = None
            for factor in FACTORS:
                candidate_scale = (
                    absmax_scale * factor
                ).to(torch.bfloat16)
                candidate = encode_legacy_e4m3(
                    source / candidate_scale.float()
                )
                reconstructed = (
                    decode_legacy_e4m3(candidate)
                    * candidate_scale.float()
                )
                error = float((source - reconstructed).square().sum())
                if error < best_error:
                    best_error = error
                    best_factor = factor
                    best_scale = candidate_scale
                    best_quantized = candidate
                    best_reconstructed = reconstructed

            quantized[
                row_start : row_start + K_TILE,
                col_start : col_start + N_TILE,
            ] = best_quantized
            scales[row_block, col_block] = best_scale
            stats.update(source, best_reconstructed)
            stats.block_count += 1
            factor_counts[best_factor] += 1

    return quantized, scales, stats, factor_counts


def dequantize_group512(weight, scales):
    expanded = scales.float().repeat_interleave(
        K_TILE, dim=0
    ).repeat_interleave(N_TILE, dim=1)
    return decode_legacy_e4m3(weight) * expanded


def load_projection(reader, layer, expert, projection):
    key = (
        f"{reader.prefix}layers.{layer}.mlp.experts."
        f"{expert}.{projection}.weight"
    )
    return dequantize_official_fp8(
        reader._get(key, "F8_E4M3"),
        reader._get(key + "_scale_inv", "BF16"),
    )


def quality(actual, expected):
    actual = actual.double().reshape(-1)
    expected = expected.double().reshape(-1)
    dot = float((actual * expected).sum())
    denominator = float(
        torch.linalg.vector_norm(actual)
        * torch.linalg.vector_norm(expected)
    )
    cosine = dot / denominator if denominator else 1.0
    nrmse = float(
        torch.linalg.vector_norm(actual - expected)
        / torch.linalg.vector_norm(expected).clamp_min(1e-30)
    )
    return cosine, nrmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-model-path", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert-start", type=int, default=0)
    parser.add_argument("--expert-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.layer < 0:
        parser.error("--layer must be non-negative")
    if args.expert_start < 0 or args.expert_count < 1:
        parser.error("--expert-start must be non-negative and count positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    torch.manual_seed(20260720)
    hidden = None
    affinities = None
    exact_output = None
    group_output = None
    aggregate = QuantizationStats()
    factor_counts = Counter()

    reader = OfficialFP8ExpertReader(args.expert_model_path)
    try:
        for local_expert in range(args.expert_count):
            expert = args.expert_start + local_expert
            gate_source = load_projection(
                reader, args.layer, expert, "gate_proj"
            )
            up_source = load_projection(
                reader, args.layer, expert, "up_proj"
            )
            down_source = load_projection(
                reader, args.layer, expert, "down_proj"
            )

            gate_q, gate_scale, gate_stats, gate_factors = (
                quantize_group512(gate_source.transpose(0, 1))
            )
            up_q, up_scale, up_stats, up_factors = (
                quantize_group512(up_source.transpose(0, 1))
            )
            down_q, down_scale, down_stats, down_factors = (
                quantize_group512(down_source.transpose(0, 1))
            )
            for stats in (gate_stats, up_stats, down_stats):
                aggregate.merge(stats)
            for counts in (gate_factors, up_factors, down_factors):
                factor_counts.update(counts)

            if hidden is None:
                hidden_size = gate_source.shape[1]
                hidden = torch.randn(
                    args.batch_size, hidden_size
                ).to(torch.bfloat16)
                selected = torch.randint(
                    0,
                    args.expert_count,
                    (
                        args.batch_size,
                        min(8, args.expert_count),
                    ),
                )
                route_weights = torch.rand_like(
                    selected, dtype=torch.float32
                )
                route_weights /= route_weights.sum(
                    dim=1, keepdim=True
                )
                affinities = torch.zeros(
                    args.batch_size,
                    args.expert_count,
                    dtype=torch.float32,
                )
                affinities.scatter_add_(
                    1, selected, route_weights
                )
                exact_output = torch.zeros(
                    args.batch_size,
                    hidden_size,
                    dtype=torch.float32,
                )
                group_output = torch.zeros_like(exact_output)

            gate_group = dequantize_group512(
                gate_q, gate_scale
            ).transpose(0, 1)
            up_group = dequantize_group512(
                up_q, up_scale
            ).transpose(0, 1)
            down_group = dequantize_group512(
                down_q, down_scale
            ).transpose(0, 1)

            exact_activated = F.silu(
                F.linear(hidden.float(), gate_source)
            ) * F.linear(hidden.float(), up_source)
            group_activated = F.silu(
                F.linear(hidden.float(), gate_group)
            ) * F.linear(hidden.float(), up_group)
            exact_activated = exact_activated.to(
                torch.bfloat16
            ).float()
            group_activated = group_activated.to(
                torch.bfloat16
            ).float()
            affinity = affinities[
                :, local_expert : local_expert + 1
            ]
            exact_output += (
                F.linear(exact_activated, down_source)
                * affinity
            )
            group_output += (
                F.linear(group_activated, down_group)
                * affinity
            )
    finally:
        reader.close()

    output_cosine, output_nrmse = quality(
        group_output, exact_output
    )
    factor_summary = ", ".join(
        f"{factor:.2f}:{factor_counts[factor]}"
        for factor in FACTORS
    )
    print(
        "128x512 shared-scale weights: "
        f"cosine={aggregate.cosine:.9f} "
        f"nrmse={aggregate.normalized_rmse:.7%}"
    )
    print(f"selected factors: {factor_summary}")
    print(
        "128x512 shared-scale CPU MoE: "
        f"cosine={output_cosine:.9f} "
        f"nrmse={output_nrmse:.7%}"
    )

    passed = (
        aggregate.cosine >= 0.9995
        and aggregate.normalized_rmse <= 0.035
        and output_cosine >= 0.9995
        and output_nrmse <= 0.035
    )
    if not passed:
        raise SystemExit(
            "REJECTED: block_group512 missed the weight or CPU MoE gate"
        )
    print("ACCEPTED: block_group512 passed both quality gates")


if __name__ == "__main__":
    main()
