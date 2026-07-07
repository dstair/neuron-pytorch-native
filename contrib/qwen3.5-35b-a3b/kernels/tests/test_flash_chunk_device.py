#!/usr/bin/env python3
"""On-device validation: CHUNKED flash prefill (one reusable NEFF, dynamic q_base)
stitched across chunks == a single full-prompt causal attention.

Simulates the bucketed prefill loop: full prompt of S tokens split into CHUNK-sized
pieces. A fixed KMAX KV buffer is zero-initialized; each chunk writes its own K/V
into [q_base:q_base+CHUNK] then attends via the chunk kernel with runtime q_base.
The concatenated per-chunk outputs must equal ref_causal over the whole prompt.

Run in DLC:  python kernels/tests/test_flash_chunk_device.py
"""
import os
import sys
import math
import torch
import torch.nn.functional as F
import torch_neuronx  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gqa_flash_prefill_35b_ops  # noqa: E402  registers torch.ops.gqa35b.flash_prefill_chunk

Q_HEADS = int(os.environ.get("GQA_Q_HEADS", "4"))
HEAD_DIM = 256


def ref_causal(q, k, v):
    H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    for h in range(H):
        sc = (q[h].float() @ k.float().t()) * scale
        sc = sc.masked_fill(mask, float("-inf"))
        out[h] = (F.softmax(sc, dim=-1) @ v.float()).to(q.dtype)
    return out


def main():
    dev = "privateuseone:0"
    torch.manual_seed(0)
    # (S, KMAX, CHUNK): last two mirror the harness configs that FAILED coherence —
    # KMAX=2048 with only [0:S] filled (zero tail), CHUNK=512 (2 chunks) & CHUNK=1024 (1 chunk).
    for S, KMAX, CHUNK in [(1024, 1024, 512), (1536, 2048, 512),
                           (1024, 2048, 512), (1024, 2048, 1024)]:
        q = (torch.randn(Q_HEADS, S, HEAD_DIM) * 0.3).float()
        k = (torch.randn(S, HEAD_DIM) * 0.3).float()
        v = (torch.randn(S, HEAD_DIM) * 0.3).float()
        ref = ref_causal(q, k, v)

        # simulate the bucketed loop
        kbuf = torch.zeros(KMAX, HEAD_DIM)
        vbuf = torch.zeros(KMAX, HEAD_DIM)
        outs = []
        n_chunks = (S + CHUNK - 1) // CHUNK
        for c in range(n_chunks):
            cs = c * CHUNK
            ce = min(cs + CHUNK, S)
            csz = ce - cs
            # pad chunk queries to CHUNK
            qc = torch.zeros(Q_HEADS, CHUNK, HEAD_DIM)
            qc[:, :csz] = q[:, cs:ce]
            # write this chunk's K/V into the buffer
            kbuf[cs:ce] = k[cs:ce]
            vbuf[cs:ce] = v[cs:ce]
            qb = torch.tensor([[float(cs)]], dtype=torch.float32)
            oc = torch.ops.gqa35b.flash_prefill_chunk(
                qc.to(dev), kbuf.to(dev), vbuf.to(dev), qb.to(dev)).cpu().float()
            outs.append(oc[:, :csz])
        out = torch.cat(outs, dim=1)   # [H,S,D]

        cos = F.cosine_similarity(ref.reshape(-1), out.reshape(-1), dim=0).item()
        maxd = (ref - out).abs().max().item()
        rel = maxd / (ref.abs().max().item() + 1e-9)
        ok = cos > 0.9999 and rel < 0.02
        print(f"[flash-chunk] S={S:5d} KMAX={KMAX:5d} CHUNK={CHUNK}  "
              f"cos={cos:.6f}  maxdiff={maxd:.4e}  rel={rel:.4e}  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
