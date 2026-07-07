"""Standalone test: nki_deltanet_chunked_prefill_v2 vs the HF-exact oracle.

Runs on the trn2 box (device='neuron'). Builds multi-head inputs, runs the NKI
kernel, and compares output + final state against ref_chunk_single_head (which
itself matches chunked_prefill.neuron_chunk_gated_delta_rule to 1e-7).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # kernels/ for bare-name imports
import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deltanet_chunked_v2_ref import build_constants, ref_chunk_single_head

K_DIM = 128
V_DIM = 128
V_HEADS = 12


def run(C=16, S=64, seed=0, realistic_gates=True):
    torch.manual_seed(seed)
    H = V_HEADS
    # head-major rows: row h*S + t
    q = torch.randn(H * S, K_DIM)
    k = torch.randn(H * S, K_DIM)
    v = torch.randn(H * S, V_DIM)
    if realistic_gates:
        # Real DeltaNet gates: g = -exp(A_log)*softplus(a+dt) is strongly
        # NEGATIVE (g_cum reaches ~-200 over 64 tokens). This exercises the
        # decay-mask exp-before-mask NaN path that g*0.1 never triggers.
        A_log = torch.randn(H, 1)
        dt = torch.randn(H, 1) * 0.1
        a = torch.randn(H * S, 1)
        g = ((-torch.exp(A_log)).repeat_interleave(S, dim=0)
             * torch.nn.functional.softplus(a + dt.repeat_interleave(S, dim=0)))
    else:
        g = torch.randn(H * S, 1) * 0.1
    beta = torch.sigmoid(torch.randn(H * S, 1))
    state = torch.randn(H * K_DIM, V_DIM) * 0.01

    m_incl, m_strict, eye = build_constants(C)

    # ---- oracle (per head), kernel L2-normalizes internally so pass raw q,k ----
    ref_out = torch.zeros(H * S, V_DIM)
    ref_state = torch.zeros(H * K_DIM, V_DIM)
    for h in range(H):
        qh = F.normalize(q[h * S:(h + 1) * S], p=2, dim=-1)
        kh = F.normalize(k[h * S:(h + 1) * S], p=2, dim=-1)
        vh = v[h * S:(h + 1) * S]
        gh = g[h * S:(h + 1) * S]
        bh = beta[h * S:(h + 1) * S]
        sh = state[h * K_DIM:(h + 1) * K_DIM]
        oh, nsh = ref_chunk_single_head(sh, qh, kh, vh, gh, bh, C, m_incl, m_strict, eye)
        ref_out[h * S:(h + 1) * S] = oh
        ref_state[h * K_DIM:(h + 1) * K_DIM] = nsh

    # ---- Run the NKI kernel. Default: nki.simulate (CPU, numpy — exercises the
    # exact kernel logic without XLA device lowering, ideal for numerical
    # validation). --device drives it through wrap_nki on XLA instead. ----
    import nki
    from deltanet_chunked_v2 import nki_deltanet_chunked_prefill_v2
    import numpy as np
    np_args = [t.numpy() for t in (state, q, k, v, g, beta, m_incl, m_strict, eye)]
    res = nki.simulate(nki_deltanet_chunked_prefill_v2)(*np_args)
    out_np, ns_np = res
    out = torch.from_numpy(np.asarray(out_np)).float()
    ns = torch.from_numpy(np.asarray(ns_np)).float()

    od = (out - ref_out).abs().max().item()
    sd = (ns - ref_state).abs().max().item()
    ocos = F.cosine_similarity(out.reshape(-1), ref_out.reshape(-1), dim=0).item()
    print(f"C={C} S={S} H={H}: out_max_diff={od:.3e} state_max_diff={sd:.3e} out_cos={ocos:.6f}")
    ok = od < 1e-2 and sd < 1e-2
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--C", type=int, default=16)
    p.add_argument("--S", type=int, default=64)
    a = p.parse_args()
    run(C=a.C, S=a.S)
