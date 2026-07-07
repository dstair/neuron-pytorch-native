#!/usr/bin/env python3
"""
Verify the TRUE-SPARSE MoE path == the masked-dense path (the validated oracle),
on real layer-0 weights, across world_size sharding. CPU, no device.

Sparse (MOE_SPARSE=1) gathers only the top-K experts per token; masked-dense
computes all E local experts and masks. They must produce identical routed
output (summed across ranks for expert-parallel). Run:
    python3 kernels/tests/test_moe_sparse_eq.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)

import model_dims as D


def run_moe(x, w, e_lo, e_hi, sparse):
    """Call moe_forward with PRE-SLICED local experts + GLOBAL offset e_lo
    (matches the harness convention: weights are this rank's rows, routing maps
    global top-k ids via local = sel - e_lo)."""
    import importlib
    os.environ["MOE_SPARSE"] = "1" if sparse else "0"
    import static_decode_35b as S
    importlib.reload(S)              # pick up the env flag at import time
    gup = w["gate_up"][e_lo:e_hi]    # local rows for this rank
    dn = w["down"][e_lo:e_hi]
    routed, shared = S.moe_forward(
        x, w["router"], gup, dn, e_lo, e_hi,
        w["sh_gate"], w["sh_up"], w["sh_down"], w["sh_sigmoid"])
    return routed, shared


def main():
    ckpt = os.environ.get("QWEN35_MODEL_PATH",
                          "/models/Qwen3.5-35B-A3B")
    D.load_from_config(os.path.join(ckpt, "config.json"))
    # Load real layer-0 packed MoE weights via the oracle's loader.
    sys.path.insert(0, HERE)
    from test_moe_oracle_cpu import load_layer_moe_weights
    raw = load_layer_moe_weights(ckpt, 0)
    w = {
        "router": raw["router"], "gate_up": raw["gate_up_proj"], "down": raw["down_proj"],
        "sh_gate": raw["sh_gate"], "sh_up": raw["sh_up"], "sh_down": raw["sh_down"],
        "sh_sigmoid": raw["sh_sigmoid_gate"],
    }
    torch.manual_seed(0)
    T = 4
    x = torch.randn(T, D.HIDDEN) * 0.5

    for W in (1, 4):
        per = D.NUM_EXPERTS // W
        # masked-dense full result = sum of per-rank partials
        md = torch.zeros(T, D.HIDDEN)
        sp = torch.zeros(T, D.HIDDEN)
        for r in range(W):
            lo, hi = r * per, (r + 1) * per
            md += run_moe(x, w, lo, hi, sparse=False)[0]
            sp += run_moe(x, w, lo, hi, sparse=True)[0]
        d = (md - sp).abs().max().item()
        rel = d / (md.abs().max().item() + 1e-9)
        print(f"world_size={W}: routed max_abs_diff={d:.3e} rel={rel:.3e}  "
              f"{'PASS' if rel < 1e-5 else 'FAIL'}")

    print("done")


if __name__ == "__main__":
    main()
