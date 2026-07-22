#!/usr/bin/env python3
"""Measure block-power-of-two FP8 quality on official expert weights."""

import argparse
import os
import sys


KERNELS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(KERNELS)
sys.path.insert(0, ROOT)

from moe_w8 import OfficialFP8ExpertReader, QuantizationStats  # noqa: E402


def parse_layers(value):
    layers = sorted({int(item) for item in value.split(",")})
    if not layers or layers[0] < 0:
        raise argparse.ArgumentTypeError("layers must be non-negative integers")
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-model-path", required=True)
    parser.add_argument("--layers", type=parse_layers, default=[0])
    parser.add_argument("--expert-start", type=int, default=0)
    parser.add_argument("--expert-end", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=32)
    args = parser.parse_args()

    if not 0 <= args.expert_start < args.expert_end:
        parser.error("--expert-start must be less than --expert-end")
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")

    aggregate = QuantizationStats()
    reader = OfficialFP8ExpertReader(args.expert_model_path)
    try:
        for layer in args.layers:
            layer_stats = QuantizationStats()
            for start in range(
                args.expert_start, args.expert_end, args.shard_size
            ):
                end = min(start + args.shard_size, args.expert_end)
                converted, stats = reader.load_layer(
                    layer,
                    start,
                    end,
                    "fp8",
                    fp8_impl="block_pow2",
                )
                del converted
                layer_stats.merge(stats)
                print(
                    f"layer={layer} experts=[{start},{end}): "
                    f"cosine={stats.cosine:.9f} "
                    f"nrmse={stats.normalized_rmse:.7%} "
                    f"shifted_blocks={stats.shifted_block_fraction:.2%} "
                    f"exact_values={stats.exact_fraction:.2%} "
                    f"clipped={stats.clipped_count}"
                )
            aggregate.merge(layer_stats)
            print(
                f"layer={layer} aggregate: "
                f"cosine={layer_stats.cosine:.9f} "
                f"nrmse={layer_stats.normalized_rmse:.7%}"
            )
    finally:
        reader.close()

    print(
        "all requested weights: "
        f"cosine={aggregate.cosine:.9f} "
        f"nrmse={aggregate.normalized_rmse:.7%} "
        f"shifted_blocks={aggregate.shifted_block_fraction:.2%} "
        f"exact_values={aggregate.exact_fraction:.2%} "
        f"clipped={aggregate.clipped_count}"
    )
    if aggregate.cosine < 0.9995 or aggregate.normalized_rmse > 0.035:
        raise SystemExit("FAIL: power-of-two conversion missed the quality gate")


if __name__ == "__main__":
    main()
