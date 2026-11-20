#!/usr/bin/env python3
"""Runnable example: official FP8 checkpoint -> fused-MoE kernel operands.

Shows the offline (CPU) weight prep for `torch.ops.moe_w8.fused_fp8_block_coalesced_ob`,
for one MoE layer and one tensor-parallel rank's local experts, then builds the
routing `affinities` from a (synthetic) router output. With `--run-kernel` on a
Trainium2 device it also calls the kernel end-to-end.

Usage
-----
Weight prep only (CPU; needs the official FP8 checkpoint + this repo on PYTHONPATH):

    python3 examples/prepare_fp8_moe_weights.py \
        --checkpoint /path/to/Qwen3.5-35B-A3B-FP8 \
        --layer 0 --world-size 8 --rank 0

End-to-end on a Trn2 device (inside the PyTorch-Native DLC):

    python3 examples/prepare_fp8_moe_weights.py \
        --checkpoint /path/to/Qwen3.5-35B-A3B-FP8 \
        --layer 0 --world-size 8 --rank 0 --run-kernel --batch 128

The kernel computes only this rank's local experts (expert-parallel); in a real
serving loop you all-reduce the [B, H] output across the TP group and then add the
shared expert. See the README "FP8 MoE decode kernel" section and INTEGRATION notes.
"""
import argparse
import os
import sys

import torch

# repo root holds moe_w8.py / st_reader.py; kernels/ holds the op registration.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "kernels"))

from moe_w8 import OfficialFP8ExpertReader, build_local_affinities  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="official FP8 checkpoint dir")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=8, help="TP degree (experts shard across ranks)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--run-kernel", action="store_true", help="call the op on a Trainium device")
    ap.add_argument("--batch", type=int, default=128, choices=(32, 64, 128))
    args = ap.parse_args()

    local_experts = args.num_experts // args.world_size
    e_start = args.rank * local_experts
    e_end = e_start + local_experts
    print(f"TP={args.world_size} rank={args.rank}: local experts [{e_start}, {e_end}) "
          f"of {args.num_experts}, layer {args.layer}")

    # --- 1) Offline weight prep (CPU): official E4M3FN + block scales -> kernel operands ---
    reader = OfficialFP8ExpertReader(args.checkpoint)
    w, stats = reader.load_layer(
        args.layer, e_start, e_end, mode="fp8", fp8_impl="block_ob_coalesced"
    )
    print("\nkernel operands (block_ob_coalesced):")
    for k in ("w8_gate_up", "w8_gate_up_scale", "w8_down", "w8_down_scale"):
        t = w[k]
        print(f"  {k:20s} shape={tuple(t.shape)} dtype={t.dtype}")
    print(f"  requantization quality: weight cosine={stats.cosine:.6f}, "
          f"NRMSE={stats.normalized_rmse * 100:.3f}%")
    H = w["w8_gate_up"].shape[1]        # [E, H, 2, I]
    I = w["w8_gate_up"].shape[3]
    print(f"  (E={w['w8_gate_up'].shape[0]}, H={H}, I={I})")

    # --- 2) Routing: (synthetic) router output -> per-token local-expert affinities ---
    torch.manual_seed(0)
    B = args.batch
    router_logits = torch.randn(B, args.num_experts)
    routing = torch.softmax(router_logits, dim=1, dtype=torch.float)
    top_w, top_id = torch.topk(routing, args.top_k, dim=-1)
    top_w = top_w / top_w.sum(dim=-1, keepdim=True)          # norm_topk_prob
    affinities = build_local_affinities(top_id, top_w, e_start, local_experts)
    on_rank = (affinities > 0).sum().item()
    print(f"\naffinities: shape={tuple(affinities.shape)} dtype={affinities.dtype} "
          f"({on_rank} of {B * args.top_k} routed slots land on this rank)")

    # --- 3) Optional: run the kernel on device ---
    if args.run_kernel:
        import moe_fused_w8_35b_ops  # noqa: F401  registers torch.ops.moe_w8.*
        dev = torch.device("neuron")
        hidden = torch.randn(B, H, dtype=torch.bfloat16).to(dev)
        out = torch.ops.moe_w8.fused_fp8_block_coalesced_ob(
            hidden,
            w["w8_gate_up"].to(dev),
            w["w8_down"].to(dev),
            w["w8_gate_up_scale"].to(dev),
            w["w8_down_scale"].to(dev),
            affinities.to(dev),
        )
        out = out.cpu()
        print(f"\nkernel output: shape={tuple(out.shape)} dtype={out.dtype} "
              f"(local-expert routed sum, PRE all-reduce) finite={bool(torch.isfinite(out).all())}")
    else:
        print("\n(pass --run-kernel on a Trn2 device to call the op end-to-end)")

    print("\nDONE")


if __name__ == "__main__":
    main()
