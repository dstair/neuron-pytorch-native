#!/usr/bin/env python3
"""
Validate the FP8 expert path (MOE_FP8=1 + MOE_SPARSE=1) vs the bf16 sparse path,
on real layer-0 weights. FP8 is lossy (e4m3, ~2 mantissa bits) so we expect a
small bounded error, NOT bit-exact — check relative error is in the expected
quant-noise band (a few %), and cosine similarity is ~1.

Run: python3 kernels/tests/test_moe_fp8_eq.py
"""
import importlib
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)
import model_dims as D


def run_bf16(x, w, e_lo, e_hi):
    os.environ["MOE_SPARSE"] = "1"; os.environ["MOE_FP8"] = "0"
    import static_decode_35b as S
    importlib.reload(S)
    routed, _ = S.moe_forward(
        x, w["router"], w["gate_up"][e_lo:e_hi], w["down"][e_lo:e_hi], e_lo, e_hi,
        w["sh_gate"], w["sh_up"], w["sh_down"], w["sh_sigmoid"])
    return routed


def run_fp8_ref(x, w, e_lo, e_hi):
    """CPU reference for the FP8 path. The NKI kernel runs device-only, so here
    we replicate its MATH: quantize (per-channel e4m3 + scale), dequant, matmul.
    Validates the quantize/transpose/scale layout; the kernel's nc_matmul FP8 is
    bit-equivalent to dequant-then-matmul up to accumulation order."""
    import static_decode_35b as S
    import model_dims as D
    import torch.nn.functional as F
    E = e_hi - e_lo
    xf = x.float()
    logits = F.linear(xf, w["router"].float()); rw = F.softmax(logits, 1, dtype=torch.float)
    rw_top, sel = torch.topk(rw, D.TOP_K, -1); rw_top = rw_top / rw_top.sum(-1, keepdim=True)
    gu_i8T, gu_s = S.quantize_experts_fp8(w["gate_up"][e_lo:e_hi])   # [E,H,2I],[E,2I,1]
    dn_i8T, dn_s = S.quantize_experts_fp8(w["down"][e_lo:e_hi])      # [E,I,H],[E,H,1]
    # dequant back to bf16-equivalent weights in [E,OUT,IN]
    gup = (gu_i8T.view(torch.float8_e4m3fn).float().transpose(1, 2) * gu_s)  # [E,2I,H]
    dn = (dn_i8T.view(torch.float8_e4m3fn).float().transpose(1, 2) * dn_s)   # [E,H,I]
    I = gup.shape[1] // 2
    local = (sel - e_lo); is_local = ((sel >= e_lo) & (sel < e_hi)).float()
    idx = local.clamp(0, E - 1).reshape(-1)
    x_sel = xf.unsqueeze(1).expand(x.shape[0], D.TOP_K, gup.shape[2]).reshape(-1, gup.shape[2])
    gu = torch.bmm(x_sel.unsqueeze(1), gup.index_select(0, idx).transpose(1, 2)).squeeze(1)
    hh = F.silu(gu[:, :I]) * gu[:, I:]
    y = torch.bmm(hh.unsqueeze(1), dn.index_select(0, idx).transpose(1, 2)).squeeze(1)
    ww = (rw_top * is_local).reshape(-1, 1)
    return (y * ww).reshape(x.shape[0], D.TOP_K, gup.shape[2]).sum(1)


def main():
    ckpt = os.environ.get("QWEN35_MODEL_PATH", "/models/Qwen3.5-35B-A3B")
    D.load_from_config(os.path.join(ckpt, "config.json"))
    from test_moe_oracle_cpu import load_layer_moe_weights
    raw = load_layer_moe_weights(ckpt, 0)
    w = {"router": raw["router"], "gate_up": raw["gate_up_proj"], "down": raw["down_proj"],
         "sh_gate": raw["sh_gate"], "sh_up": raw["sh_up"], "sh_down": raw["sh_down"],
         "sh_sigmoid": raw["sh_sigmoid_gate"]}
    torch.manual_seed(0)
    T = 4
    x = torch.randn(T, D.HIDDEN) * 0.5

    for W in (1, 4):
        per = D.NUM_EXPERTS // W
        bf16 = torch.zeros(T, D.HIDDEN); fp8 = torch.zeros(T, D.HIDDEN)
        for r in range(W):
            lo, hi = r * per, (r + 1) * per
            bf16 += run_bf16(x, w, lo, hi)
            fp8 += run_fp8_ref(x, w, lo, hi)
        rel = (bf16 - fp8).abs().max().item() / (bf16.abs().max().item() + 1e-9)
        cos = F.cosine_similarity(bf16.reshape(-1), fp8.reshape(-1), dim=0).item()
        # e4m3 per-channel quant: expect rel error a few % and cosine > 0.99
        ok = rel < 0.10 and cos > 0.99
        print(f"world_size={W}: fp8-vs-bf16 rel={rel:.3e} cosine={cos:.6f}  "
              f"{'PASS' if ok else 'FAIL'}")
    print("done")


if __name__ == "__main__":
    main()
