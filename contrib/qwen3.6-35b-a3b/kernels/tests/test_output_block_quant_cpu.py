"""CPU numerics gate for output-block FP8 CTE scales (no device needed).

Phase 1.1 of the FP8 prefill Vector-tax work: the per-256-H-chunk block scale is
what forces a Vector scale-add per contraction chunk in nkilib's block-quant path
(bwmm_shard_on_I.py:1364-1390 gate_up, :2133-2140 down) -- 26% of prefill Vector
time. Reducing the scale grid over the CONTRACTION-block axis makes the scale
h-independent so the chunks can accumulate in one PSUM tile instead.

This script answers the accuracy question on REAL weights before any kernel
change or device compile: it quantizes real expert tensors under both grids and
reports dequant cosine / nrmse / E4M3 saturation.

Usage (in the DLC, model mounted at /models/Qwen3.5-35B-A3B):
    python3 kernels/tests/test_output_block_quant_cpu.py \
        --model /models/Qwen3.5-35B-A3B --layers 0,19,39
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from moe_w8 import quantize_down_256block, quantize_gate_up_256block  # noqa: E402


def _load_shard_map(model_dir):
    for name in ("model.safetensors.index.json", "model.safetensors-index.json"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            with open(path) as handle:
                return json.load(handle)["weight_map"]
    raise FileNotFoundError(f"no safetensors index in {model_dir}")


def _get(model_dir, shard_map, key):
    from safetensors import safe_open

    shard = shard_map[key]
    with safe_open(os.path.join(model_dir, shard), framework="pt") as handle:
        return handle.get_tensor(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", default="0")
    parser.add_argument("--experts", type=int, default=32,
                        help="local experts per rank (256 global / TP=8)")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    args = parser.parse_args()

    shard_map = _load_shard_map(args.model)
    # This checkpoint nests the decoder under model.language_model; discover the
    # prefix rather than hardcoding it.
    stem = next(
        k.split("layers.")[0]
        for k in shard_map
        if k.endswith("layers.0.mlp.experts.gate_up_proj") and not k.startswith("mtp.")
    )
    failures = []
    for layer in [int(x) for x in args.layers.split(",")]:
        prefix = f"{stem}layers.{layer}.mlp.experts."
        gu = _get(args.model, shard_map, prefix + "gate_up_proj")[: args.experts]
        dn = _get(args.model, shard_map, prefix + "down_proj")[: args.experts]
        Ec, two_i, HH = gu.shape
        II = two_i // 2
        # Exactly static_decode_35b.py's FP8 CTE layout.
        gate_up_k = gu.reshape(Ec, 2, II, HH).permute(0, 3, 1, 2).contiguous()
        down_k = dn.permute(0, 2, 1).contiguous()
        print(f"\nlayer {layer}: gate_up {tuple(gate_up_k.shape)} "
              f"down {tuple(down_k.shape)}")
        for which, tensor, fn in (
            ("gate_up", gate_up_k, quantize_gate_up_256block),
            ("down", down_k, quantize_down_256block),
        ):
            row = {}
            for grid in (False, True):
                _, _, st = fn(tensor, output_block=grid)
                row["output_block" if grid else "full"] = st
            full, ob = row["full"], row["output_block"]
            verdict = "PASS" if ob.cosine >= args.min_cosine else "FAIL"
            if ob.clipped_count:
                verdict = "FAIL(clipped)"
            print(
                f"  {which:8s} full: cosine={full.cosine:.7f} "
                f"nrmse={full.normalized_rmse:.5f} scales={full.block_count}\n"
                f"  {which:8s} ob:   cosine={ob.cosine:.7f} "
                f"nrmse={ob.normalized_rmse:.5f} scales={ob.block_count} "
                f"clipped={ob.clipped_count}  [{verdict}]"
            )
            if verdict != "PASS":
                failures.append(f"l{layer}.{which}")

    if failures:
        print(f"\nGATE FAILED for: {', '.join(failures)}")
        return 1
    print(f"\nGATE PASSED (cosine >= {args.min_cosine}, no saturation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
