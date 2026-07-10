#!/usr/bin/env python3
"""On-device validation: flash GQA causal-prefill kernel vs pure-torch reference.

Mirrors _gqa_prefill's grouped causal attention (single KV head, Q_HEADS queries,
head_dim=256, scale 1/sqrt(256)). Checks cosine sim + max abs diff across a few S,
including a non-512-multiple S to exercise the ragged tail and straddling causal block.

Run in DLC:  python kernels/tests/test_flash_prefill_device.py
"""
import os
import sys
import math
import torch
import torch.nn.functional as F
import torch_neuronx  # noqa: F401

# kernels/ dir (parent of tests/) holds gqa_flash_prefill_35b + _ops
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gqa_flash_prefill_35b_ops  # noqa: E402  registers torch.ops.gqa35b.flash_prefill

Q_HEADS = int(os.environ.get("GQA_Q_HEADS", "4"))
HEAD_DIM = 256


def ref_causal(q, k, v):
    # q [H,S,D], k/v [S,D] -> [H,S,D]  (single KV head shared by all Q heads)
    H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    for h in range(H):
        sc = (q[h].float() @ k.float().t()) * scale        # [S,S]
        sc = sc.masked_fill(mask, float("-inf"))
        p = F.softmax(sc, dim=-1)
        out[h] = (p @ v.float()).to(q.dtype)
    return out


def main():
    dev = "privateuseone:0"
    torch.manual_seed(0)
    for S in [512, 640, 1024]:
        q = (torch.randn(Q_HEADS, S, HEAD_DIM) * 0.3).float()
        k = (torch.randn(S, HEAD_DIM) * 0.3).float()
        v = (torch.randn(S, HEAD_DIM) * 0.3).float()
        ref = ref_causal(q, k, v)
        out = torch.ops.gqa35b.flash_prefill(q.to(dev), k.to(dev), v.to(dev)).cpu().float()
        cos = F.cosine_similarity(ref.reshape(-1), out.reshape(-1), dim=0).item()
        maxd = (ref - out).abs().max().item()
        rel = maxd / (ref.abs().max().item() + 1e-9)
        ok = cos > 0.9999 and rel < 0.02
        print(f"[flash-prefill] S={S:5d}  cos={cos:.6f}  maxdiff={maxd:.4e}  rel={rel:.4e}  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
