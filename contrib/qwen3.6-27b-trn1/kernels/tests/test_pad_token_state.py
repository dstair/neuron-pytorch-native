"""Does trailing PAD-token padding (the vLLM runner pattern) corrupt the chunked
DeltaNet prefill state vs the sequential per-token recurrence?

The runner appends pad tokens (token_id=0) on the RIGHT of the prompt up to the
bucket length: [real_0..real_{S-1}, pad, pad, ...]. It assumes "causal attention
masks padding" — TRUE for the GQA layers, but DeltaNet is a RECURRENCE with NO
token mask, so every prefill path folds the pad tokens into the persisted
recurrent_state. The decode kernel then reads that state for token S (the real
continuation). If the chunked rule folds pad tokens differently from the proven
sequential recurrence (deltanet_full steps each token), the persisted state at
the real boundary diverges -> "first token right, then gibberish".

This test feeds IDENTICAL (real + pad) frontend tensors to:
  (A) neuron_chunk_gated_delta_rule  (the chunked oracle == kernel math)
  (B) a plain sequential gated-delta-rule loop (== deltanet_full token loop)
and compares the recurrent state AT THE LAST REAL TOKEN (position S-1), which is
what decode actually continues from.

Pad-token frontend values are NOT zero: token_id=0 still projects through the
in-proj + conv to nonzero q/k/v/g/beta. We emulate that by giving pad rows their
own (small) random projections — the realistic case.
"""
import os, sys
import torch
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))  # qwen3_6/ for chunked_prefill
from chunked_prefill import neuron_chunk_gated_delta_rule

K = V = 128
H = 12


def seq_gated_delta_rule(q, k, v, g, beta, S_eval):
    """Sequential reference (mirrors deltanet_full's per-token recurrence).
    q,k,v [H,T,*]; g,beta [H,T]. L2-norm q,k per the kernel. Returns state [H,K,V]
    AFTER processing the first S_eval tokens (the real prompt only)."""
    qn = F.normalize(q, p=2, dim=-1) / (K ** 0.5)
    kn = F.normalize(k, p=2, dim=-1)
    state = torch.zeros(H, K, V)
    for t in range(S_eval):
        gt = g[:, t].exp().view(H, 1, 1)              # decay
        state = state * gt
        kt = kn[:, t]                                 # [H,K]
        vt = v[:, t]                                  # [H,V]
        bt = beta[:, t].view(H, 1)                    # [H,1]
        kv = torch.einsum('hk,hkv->hv', kt, state)    # state.T @ k  [H,V]
        delta = (vt - kv) * bt                        # [H,V]
        state = state + torch.einsum('hk,hv->hkv', kt, delta)
    return state


def chunked_state(q, k, v, g, beta, C, S_total):
    """Chunked oracle state after ALL S_total tokens (real+pad)."""
    # oracle wants [B, S, H, *]; our tensors are [H, S, *] -> [1, S, H, *].
    qd = q.transpose(0, 1).unsqueeze(0)
    kd = k.transpose(0, 1).unsqueeze(0)
    vd = v.transpose(0, 1).unsqueeze(0)
    gd = g.transpose(0, 1).unsqueeze(0)
    bd = beta.transpose(0, 1).unsqueeze(0)
    _, fstate = neuron_chunk_gated_delta_rule(
        qd, kd, vd, g=gd, beta=bd,
        chunk_size=C, initial_state=torch.zeros(1, H, K, V),
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    return fstate.view(H, K, V)


def run(S_real=6, S_total=128, C=64, seed=0, pad_scale=1.0, identical_pads=True):
    torch.manual_seed(seed)
    # frontend tensors [H, S_total, *]; first S_real are "real", rest are pad.
    q = torch.randn(H, S_total, K)
    k = torch.randn(H, S_total, K)
    v = torch.randn(H, S_total, V)
    # realistic gates: g = -exp(A_log)*softplus(a+dt) (strongly negative)
    A_log = torch.randn(H, 1)
    dt = torch.randn(H, 1) * 0.1
    a = torch.randn(H, S_total)
    g = (-torch.exp(A_log)) * F.softplus(a + dt)
    beta = torch.sigmoid(torch.randn(H, S_total))
    if identical_pads:
        # REAL runner behaviour: pad tokens are ALL token_id=0 -> they project to
        # the SAME q/k/v/g/beta values. A chunk full of identical k-rows makes
        # A_str = -(k_beta @ k.T) rank-deficient/correlated -> large entries ->
        # the doubling-series (I-A_str)^-1 can blow up (kernel comment: stable
        # ONLY when A_str small). The sequential recurrence has no inverse -> stays
        # stable. This is the suspected chunked-vs-tokenloop divergence.
        pad_q = (torch.randn(H, 1, K) * pad_scale).expand(H, S_total - S_real, K)
        pad_k = (torch.randn(H, 1, K) * pad_scale).expand(H, S_total - S_real, K)
        pad_v = (torch.randn(H, 1, V) * pad_scale).expand(H, S_total - S_real, V)
        q[:, S_real:] = pad_q
        k[:, S_real:] = pad_k
        v[:, S_real:] = pad_v
        # pad gates also identical per head
        g[:, S_real:] = g[:, S_real:S_real + 1]
        beta[:, S_real:] = beta[:, S_real:S_real + 1]
    elif pad_scale != 1.0:
        q[:, S_real:] *= pad_scale
        k[:, S_real:] *= pad_scale
        v[:, S_real:] *= pad_scale

    # (B) sequential, REAL tokens only — the ground-truth state decode should see
    seq_real = seq_gated_delta_rule(q, k, v, g, beta, S_real)
    # (A) chunked over ALL tokens (real+pad), as the model runs it
    chunk_all = chunked_state(q, k, v, g, beta, C, S_total)
    # (B') sequential over ALL tokens (what the token-loop path actually persists)
    seq_all = seq_gated_delta_rule(q, k, v, g, beta, S_total)

    d_chunk_vs_seqreal = (chunk_all - seq_real).abs().max().item()
    d_seqall_vs_seqreal = (seq_all - seq_real).abs().max().item()
    d_chunk_vs_seqall = (chunk_all - seq_all).abs().max().item()
    print(f"S_real={S_real} S_total={S_total} C={C} pad_scale={pad_scale}")
    print(f"  chunked(all)  vs seq(real-only) : {d_chunk_vs_seqreal:.3e}  "
          f"<- does pad corrupt the chunked state at the real boundary?")
    print(f"  seq(all)      vs seq(real-only) : {d_seqall_vs_seqreal:.3e}  "
          f"<- does pad corrupt the SEQUENTIAL (token-loop) state too?")
    print(f"  chunked(all)  vs seq(all)       : {d_chunk_vs_seqall:.3e}  "
          f"<- do the two PATHS agree on the full padded sequence?")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--S_real", type=int, default=6)
    p.add_argument("--S_total", type=int, default=128)
    p.add_argument("--C", type=int, default=64)
    p.add_argument("--pad_scale", type=float, default=1.0)
    a = p.parse_args()
    run(S_real=a.S_real, S_total=a.S_total, C=a.C, pad_scale=a.pad_scale)
