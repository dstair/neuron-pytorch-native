"""Validate nki_gqa_tail vs the host oracle (gqa_tail_ref) on the NKI simulator (CPU).
Catches math errors cheaply before any device compile."""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gqa_tail_ref import gqa_tail_ref, HEAD_DIM, Q_HEADS, ROPE_DIM


def run(B=8, S=128, pos=100, seed=0):
    torch.manual_seed(seed)
    query = torch.randn(B, Q_HEADS, HEAD_DIM)
    gate = torch.randn(B, Q_HEADS, HEAD_DIM)
    q_norm = torch.randn(HEAD_DIM) * 0.1
    cos = torch.randn(ROPE_DIM); sin = torch.randn(ROPE_DIM)
    cached_k = torch.randn(B, S, HEAD_DIM)
    cached_v = torch.randn(B, S, HEAD_DIM)
    mask = (torch.arange(S) <= pos).float()

    ref = gqa_tail_ref(query, gate, q_norm, cos, sin, cached_k, cached_v, mask)  # [B,1536]

    # flatten to kernel layout: query/gate [B*6,256]; cached_k/v [B*S,256]; mask [1,S]
    import nki
    from gqa_tail import nki_gqa_tail
    np_args = [
        query.reshape(B * Q_HEADS, HEAD_DIM).numpy(),
        gate.reshape(B * Q_HEADS, HEAD_DIM).numpy(),
        q_norm.reshape(1, HEAD_DIM).numpy(), cos.reshape(1, ROPE_DIM).numpy(), sin.reshape(1, ROPE_DIM).numpy(),
        cached_k.reshape(B * S, HEAD_DIM).numpy(),
        cached_v.reshape(B * S, HEAD_DIM).numpy(),
        mask.reshape(1, S).numpy(),
    ]
    res = nki.simulate(nki_gqa_tail)(*np_args)
    out = torch.from_numpy(np.asarray(res)).float()  # [B*6,256]
    out = out.reshape(B, Q_HEADS * HEAD_DIM)

    d = (out - ref).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(out.reshape(-1), ref.reshape(-1), dim=0).item()
    print(f"B={B} S={S} pos={pos}: max_diff={d:.3e} cos={cos_sim:.6f}")
    ok = d < 1e-2
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--S", type=int, default=128)
    p.add_argument("--pos", type=int, default=100)
    a = p.parse_args()
    run(B=a.B, S=a.S, pos=a.pos)
