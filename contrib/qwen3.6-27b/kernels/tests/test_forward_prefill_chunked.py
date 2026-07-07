"""CPU harness for the chunked forward_prefill ASSEMBLY (task A4).

The chunked NKI kernel (deltanet_chunked_v2) only does L2norm + the chunked
recurrence and returns RAW output. Wiring it into Qwen3_5LinearAttention.forward_prefill
means lifting the rest of the DeltaNet front-end OUT into vectorized PyTorch:

    project → causal depthwise conv1d + SiLU → split q/k/v → GQA-expand q,k 3x
    → gates (g, beta) → [head-major reshape] → recurrence → [un-reshape]
    → RMSNormGated(z) → out_proj

This harness validates that assembly + layout glue on CPU against the HF-exact
oracle (chunked_prefill.neuron_chunk_gated_delta_rule), with NO container / NO
device compile. The kernel's recurrence itself is already validated separately
(test_deltanet_chunked_v2.py, 3e-5 vs oracle); here we stand the kernel in with
ref_chunk_single_head (the proven CPU mirror) and check the GLUE around it:
conv, split, GQA expansion, gate formula, head-major transpose, gated norm.

Per-core dims (TP=4): k_heads=4, v_heads=12, k_dim=v_dim=128, conv_kernel=4,
qkv_dim=2560, head_group=3.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # kernels/ for bare-name imports
import os, sys, math
import torch
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))           # kernels/tests
sys.path.insert(0, os.path.dirname(_here))                   # kernels/ (deltanet_chunked_v2_ref)
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))  # qwen3_6/ (chunked_prefill)
from deltanet_chunked_v2_ref import build_constants, ref_chunk_single_head
from chunked_prefill import neuron_chunk_gated_delta_rule

# ── per-core DeltaNet dims (TP=4) ────────────────────────────────────────────
K_HEADS = 4
V_HEADS = 12
K_DIM = 128
V_DIM = 128
CONV_K = 4
HEAD_GROUP = V_HEADS // K_HEADS  # 3
QKV_DIM = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM  # 2560
RMS_EPS = 1e-6


# ── front-end shared by both the OLD token-loop and the NEW chunked path ─────
def deltanet_frontend(mixed_qkv, a_out, b_out, A_log, dt_bias, conv_weight, conv_state0):
    """mixed_qkv [S,QKV_DIM] (pre-conv); a_out/b_out [S,V_HEADS]; returns the
    per-token q,k,v (raw, GQA-expanded to V_HEADS) and gates g,beta, plus the
    conv_state to persist. Mirrors the decode kernel's phase-1/phase-2 exactly."""
    S = mixed_qkv.shape[0]
    # causal depthwise conv1d + SiLU over the QKV channels.
    # conv_state0 is the 3 tokens BEFORE this prompt (zeros at prefill start).
    x = mixed_qkv.transpose(0, 1).unsqueeze(0)              # [1, QKV_DIM, S]
    left = conv_state0.transpose(0, 1).unsqueeze(0)        # [1, QKV_DIM, 3]
    x_pad = torch.cat([left, x], dim=-1)                   # [1, QKV_DIM, S+3]
    w = conv_weight.unsqueeze(1)                            # [QKV_DIM, 1, 4]
    conv = F.conv1d(x_pad, w, groups=QKV_DIM)              # [1, QKV_DIM, S]
    conv = F.silu(conv).squeeze(0).transpose(0, 1)        # [S, QKV_DIM]

    # conv_state to persist = last 3 columns of the PRE-conv qkv sequence.
    new_conv_state = mixed_qkv[-(CONV_K - 1):].clone()     # [3, QKV_DIM]

    # split q | k | v  (per-core widths 512 | 512 | 1536)
    qd = K_HEADS * K_DIM
    q = conv[:, 0:qd].reshape(S, K_HEADS, K_DIM)
    k = conv[:, qd:2 * qd].reshape(S, K_HEADS, K_DIM)
    v = conv[:, 2 * qd:].reshape(S, V_HEADS, V_DIM)

    # GQA expand: each k-head serves HEAD_GROUP v-heads.
    q = q.repeat_interleave(HEAD_GROUP, dim=1)             # [S, V_HEADS, K_DIM]
    k = k.repeat_interleave(HEAD_GROUP, dim=1)             # [S, V_HEADS, K_DIM]

    # gates (per value-head): g = -exp(A_log)*softplus(a+dt); beta = sigmoid(b)
    g = (-torch.exp(A_log)) * F.softplus(a_out + dt_bias)  # [S, V_HEADS]
    beta = torch.sigmoid(b_out)                            # [S, V_HEADS]
    return q, k, v, g, beta, new_conv_state


# ── NEW chunked prefill (the model.py rewrite, in pure torch for CPU validation)
def chunked_prefill_assembly(mixed_qkv, a_out, b_out, z, A_log, dt_bias,
                             conv_weight, norm_weight, conv_state0, C):
    """The assembly that will go into forward_prefill. Uses ref_chunk_single_head
    (the CPU stand-in for the NKI kernel) per head, with head-major reshape."""
    S = mixed_qkv.shape[0]
    q, k, v, g, beta, new_cs = deltanet_frontend(
        mixed_qkv, a_out, b_out, A_log, dt_bias, conv_weight, conv_state0)

    m_incl, m_strict, eye = build_constants(C)
    state0 = torch.zeros(K_DIM, V_DIM)  # prefill starts from zero recurrent state

    # PARTIAL-CHUNK FIX (mirrors the chunked-prefill rule's own internal pad):
    # the kernel/ref process num_chunks = S // C chunks (floor) and do NOT pad,
    # so S < C runs ZERO chunks (state stays zero -> decode gibberish) and a
    # non-multiple S drops its final partial chunk. Pad S up to a multiple of C
    # with ZEROS: pad tokens get beta=0 (-> v_new=0, no state contribution) and
    # g=0 (-> no spurious decay), so they are exact no-ops. Slice output to [:S].
    S_pad = ((S + C - 1) // C) * C
    if S_pad != S:
        pad = S_pad - S
        q = F.pad(q, (0, 0, 0, 0, 0, pad))       # [S,H,K] -> pad seq dim 0
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))             # [S,H]   -> pad seq dim 0
        beta = F.pad(beta, (0, 0, 0, pad))

    # kernel expects RAW q,k (it L2-normalizes internally); ref_chunk_single_head
    # does NOT normalize, so normalize here to stand in for the kernel.
    raw_out = torch.zeros(S_pad, V_HEADS, V_DIM)
    new_state = torch.zeros(V_HEADS * K_DIM, V_DIM)
    for h in range(V_HEADS):
        qh = F.normalize(q[:, h], p=2, dim=-1)
        kh = F.normalize(k[:, h], p=2, dim=-1)
        oh, nsh = ref_chunk_single_head(
            state0, qh, kh, v[:, h], g[:, h:h + 1], beta[:, h:h + 1],
            C, m_incl, m_strict, eye)
        raw_out[:, h] = oh
        new_state[h * K_DIM:(h + 1) * K_DIM] = nsh
    raw_out = raw_out[:S]  # drop pad tokens' outputs

    # RMSNormGated(raw_out, z), per value-head, then flatten.
    gated = _rms_norm_gated(raw_out, z, norm_weight, RMS_EPS)   # [S, V_HEADS, V_DIM]
    return gated.reshape(S, V_HEADS * V_DIM), new_state, new_cs


def _rms_norm_gated(x, gate, weight, eps):
    xf = x.float()
    norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * norm) * F.silu(gate.float())


# ── ground-truth oracle path (HF-exact chunked rule) ─────────────────────────
def oracle_prefill(mixed_qkv, a_out, b_out, z, A_log, dt_bias,
                   conv_weight, norm_weight, conv_state0, C):
    S = mixed_qkv.shape[0]
    q, k, v, g, beta, oracle_cs = deltanet_frontend(
        mixed_qkv, a_out, b_out, A_log, dt_bias, conv_weight, conv_state0)
    # oracle expects [B,S,H,*]; B=1, H=V_HEADS. use_qk_l2norm_in_kernel=True so it
    # L2-normalizes q,k itself (matching the NKI kernel's internal normalization).
    out, fstate = neuron_chunk_gated_delta_rule(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        g=g.unsqueeze(0), beta=beta.unsqueeze(0),
        chunk_size=C, initial_state=torch.zeros(1, V_HEADS, K_DIM, V_DIM),
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    raw_out = out.view(S, V_HEADS, V_DIM)
    gated = _rms_norm_gated(raw_out, z, norm_weight, RMS_EPS)
    # final recurrent state [B,H,K,V] -> head-major [V_HEADS*K_DIM, V_DIM] (the
    # layout forward_prefill persists into self.recurrent_state[0]).
    oracle_state = fstate.view(V_HEADS, K_DIM, V_DIM).reshape(V_HEADS * K_DIM, V_DIM)
    return gated.reshape(S, V_HEADS * V_DIM), oracle_state, oracle_cs


def run(S=128, C=64, seed=0):
    torch.manual_seed(seed)
    mixed_qkv = torch.randn(S, QKV_DIM)
    a_out = torch.randn(S, V_HEADS) * 0.1
    b_out = torch.randn(S, V_HEADS)
    z = torch.randn(S, V_HEADS, V_DIM)
    A_log = torch.randn(V_HEADS)
    dt_bias = torch.randn(V_HEADS) * 0.1
    conv_weight = torch.randn(QKV_DIM, CONV_K) * 0.2
    norm_weight = torch.randn(V_DIM) * 0.1
    conv_state0 = torch.zeros(CONV_K - 1, QKV_DIM)

    mine, mine_state, mine_cs = chunked_prefill_assembly(
        mixed_qkv, a_out, b_out, z, A_log, dt_bias,
        conv_weight, norm_weight, conv_state0, C)
    ref, ref_state, ref_cs = oracle_prefill(
        mixed_qkv, a_out, b_out, z, A_log, dt_bias,
        conv_weight, norm_weight, conv_state0, C)

    d = (mine - ref).abs().max().item()
    cos = F.cosine_similarity(mine.reshape(-1), ref.reshape(-1), dim=0).item()
    # persisted recurrent state (consumed by the DECODE kernel) — NOT checked
    # before; the prefill->decode state handoff is the prime gibberish suspect.
    sd = (mine_state - ref_state).abs().max().item()
    scos = F.cosine_similarity(mine_state.reshape(-1), ref_state.reshape(-1), dim=0).item()
    # persisted conv_state (last ck-1 PRE-conv qkv tokens) — also decode-consumed.
    csd = (mine_cs - ref_cs).abs().max().item()
    print(f"S={S} C={C}: out max_diff={d:.3e} cos={cos:.6f} | "
          f"STATE max_diff={sd:.3e} cos={scos:.6f} | conv_state max_diff={csd:.3e}")
    ok = d < 1e-3 and sd < 1e-3 and csd < 1e-3
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--S", type=int, default=128)
    p.add_argument("--C", type=int, default=64)
    a = p.parse_args()
    run(S=a.S, C=a.C)
