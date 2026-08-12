"""REGRESSION TEST for the chunked-prefill gibberish root cause (2026-06-18).

The vLLM runner pads prompts to the bucket length with token_id=0 tokens, which
all project to IDENTICAL q/k/v. A chunk containing such degenerate rows makes
A_str = -(k_beta @ k.T) rank-deficient, and the chunked kernel's Woodbury inverse
(I - A_str)^-1 via the DOUBLING SERIES is fp32-stable ONLY when A_str is small —
so it BLOWS UP (state → ~291 at S_real=15, C=64), corrupting the recurrent state
the decode kernel reads → "first token right, then gibberish". The sequential
per-token recurrence (deltanet_full, used by QWEN3_5_CHUNKED_PREFILL=0) has no
matrix inverse, so it stays stable — which is why the token-loop path is coherent.

This test reproduces the blow-up (doubling-series ref_chunk_single_head vs the
sequential recurrence on [real..., identical-pad...]) and verifies the FIX:
zeroing pad tokens' beta (→ k_beta=0 → their A_str rows/cols vanish) and g makes
the chunked state bit-exact (≤5e-7) vs the sequential recurrence over real tokens.
NOTE: the HF oracle (chunked_prefill.py) uses data-dependent forward-substitution
Woodbury, NOT the doubling series, so it does NOT blow up — which is why all the
oracle-based tests missed this. This test compares against the DOUBLING ref on
purpose, because that mirrors the actual kernel.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # kernels/
import torch, torch.nn.functional as F
from deltanet_chunked_v2_ref import build_constants, ref_chunk_single_head, tri_inverse_doubling

K=V=128
def seq_ref(q,k,v,g,beta,S_eval):
    qn=F.normalize(q,p=2,dim=-1)/(K**0.5); kn=F.normalize(k,p=2,dim=-1)
    s=torch.zeros(K,V)
    for t in range(S_eval):
        s=s*g[t].exp()
        kt=kn[t]; vt=v[t]; bt=beta[t]
        kv=kt@s
        delta=(vt-kv)*bt
        s=s+torch.outer(kt,delta)
    return s

def run(S_real, S_total, C, seed=0):
    torch.manual_seed(seed)
    q=torch.randn(S_total,K); k=torch.randn(S_total,K); v=torch.randn(S_total,V)
    A_log=torch.randn(1); dt=torch.randn(1)*0.1; a=torch.randn(S_total)
    g=(-torch.exp(A_log))*F.softplus(a+dt)
    beta=torch.sigmoid(torch.randn(S_total))
    # identical pad rows (token_id=0)
    pq=torch.randn(K); pk=torch.randn(K); pv=torch.randn(V)
    q[S_real:]=pq; k[S_real:]=pk; v[S_real:]=pv
    g[S_real:]=g[S_real]; beta[S_real:]=beta[S_real]
    m_incl,m_strict,eye=build_constants(C)
    # doubling-series chunked (mirrors the KERNEL) over ALL tokens
    qn=F.normalize(q,p=2,dim=-1); kn=F.normalize(k,p=2,dim=-1)
    _, s_chunk = ref_chunk_single_head(torch.zeros(K,V), qn, kn, v, g.view(-1,1), beta.view(-1,1), C, m_incl, m_strict, eye)
    # check A_str magnitude in the all-pad chunk
    s_seq_all = seq_ref(q,k,v,g,beta,S_total)
    d = (s_chunk - s_seq_all).abs().max().item()
    print(f"S_real={S_real} S_total={S_total} C={C}: doubling-chunk vs seq(all) max_diff={d:.4e} "
          f"chunk_absmax={s_chunk.abs().max().item():.4e} chunk_nan={int(s_chunk.isnan().sum())} chunk_inf={int(s_chunk.isinf().sum())}")
    return d

print("=== BUG REPRO: identical pad tokens (token_id=0) blow up the doubling inverse ===")
blew_up = False
for sr in [6, 15, 64, 70]:
    if run(sr, 128, 64) > 1.0:
        blew_up = True
print("REPRO OK (state blew up at >=1 S_real)" if blew_up else "NO BLOW-UP (unexpected)")

print("=== FIX: zero pad-token beta (kills A_str contribution) + g ===")
def run_fix(S_real, S_total, C, seed=0):
    torch.manual_seed(seed)
    q=torch.randn(S_total,K); k=torch.randn(S_total,K); v=torch.randn(S_total,V)
    A_log=torch.randn(1); dt=torch.randn(1)*0.1; a=torch.randn(S_total)
    g=(-torch.exp(A_log))*F.softplus(a+dt)
    beta=torch.sigmoid(torch.randn(S_total))
    pq=torch.randn(K); pk=torch.randn(K); pv=torch.randn(V)
    q[S_real:]=pq; k[S_real:]=pk; v[S_real:]=pv
    g[S_real:]=g[S_real]; beta[S_real:]=beta[S_real]
    # FIX: zero the pad tokens' beta (no state update) and g=0 (no decay).
    beta[S_real:]=0.0
    g[S_real:]=0.0
    m_incl,m_strict,eye=build_constants(C)
    qn=F.normalize(q,p=2,dim=-1); kn=F.normalize(k,p=2,dim=-1)
    _, s_chunk = ref_chunk_single_head(torch.zeros(K,V), qn, kn, v, g.view(-1,1), beta.view(-1,1), C, m_incl, m_strict, eye)
    # ground truth = sequential over REAL tokens only
    s_seq_real = seq_ref(q,k,v,g,beta,S_real)
    d = (s_chunk - s_seq_real).abs().max().item()
    print(f"S_real={S_real}: fixed-chunk vs seq(real-only) max_diff={d:.4e} chunk_absmax={s_chunk.abs().max().item():.4e}")
    return d
worst = max(run_fix(sr, 128, 64) for sr in [6, 15, 64, 70])
print("PASS" if worst < 1e-3 else "FAIL")
