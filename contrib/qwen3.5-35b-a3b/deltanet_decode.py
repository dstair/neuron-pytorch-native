#!/usr/bin/env python3
"""
Pure-torch DeltaNet single-token decode recurrence (gated delta rule).

Correctness-first path for the 35B bring-up: avoids retuning the 27B NKI
deltanet kernels (which hardcode 4 k-heads / 12 v-heads per core — the 35B is
4/8 at TP=4) by expressing the single-token recurrent step in plain torch. It
has NO data-dependent control flow, so it compiles under
torch.compile(fullgraph=True, backend="neuron"). The NKI kernel is a Task-5
perf lever, added back once the model is correct.

The step is the chunk recurrence specialized to chunk_size=1, which is validated
against HF's torch_chunk_gated_delta_rule to ~1e-10 (see chunked_prefill.py).
This module's test cross-checks it against neuron_chunk_gated_delta_rule(C=1).

State layout matches the 27B harness: state[H, K, V] per (batch,layer).
"""
import torch
import torch.nn.functional as F


def deltanet_recurrent_step(state, q, k, v, g, beta,
                            use_qk_l2norm=True):
    """One decode token through the gated delta rule.

    Args (per batch row; H = value heads after k->v expansion):
        state: [H, K, V] f32   recurrent state (modified out-of-place, returned)
        q,k:   [H, K]    query/key (RAW — L2norm + 1/sqrt(K) scale applied here)
        v:     [H, V]
        g:     [H]       log-decay (already = -exp(A_log)*softplus(a+dt_bias))
        beta:  [H]       update gate (already sigmoid'd)
    Returns:
        out:        [H, V]      attention output (pre gated-norm)
        new_state:  [H, K, V]   updated recurrent state
    """
    K = q.shape[-1]
    if use_qk_l2norm:
        q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
        k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
    else:
        q = q.float()
        k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()
    state = state.float()

    import os as _os
    if _os.environ.get("DN_PASSTHROUGH", "0") == "1":
        # DIAGNOSTIC: skip the recurrent einsums (return v, decayed state).
        # Isolates whether the pure-torch DeltaNet recurrence is the neuronx-cc
        # PGTiling trigger. NOT numerically correct.
        return v, state * g.exp().unsqueeze(-1).unsqueeze(-1)

    q = q * (1.0 / (K ** 0.5))                     # query scale
    expg = g.exp()                                 # [H]

    # Specialization of the chunk recurrence to a single token (C=1):
    #   v_new   = beta*v - (beta*expg) * (k·state)        [H, V]
    #   out     = expg*(q·state) + (q·k) * v_new          [H, V]
    #   state'  = expg*state + kᵀ·v_new                   [H, K, V]
    k_state = torch.einsum("hk,hkv->hv", k, state)         # [H, V]
    v_beta = v * beta.unsqueeze(-1)                        # [H, V]
    v_new = v_beta - (beta * expg).unsqueeze(-1) * k_state # [H, V]

    q_state = torch.einsum("hk,hkv->hv", q, state)         # [H, V]
    qk = (q * k).sum(-1, keepdim=True)                     # [H, 1]
    out = expg.unsqueeze(-1) * q_state + qk * v_new        # [H, V]

    new_state = expg.unsqueeze(-1).unsqueeze(-1) * state \
        + torch.einsum("hk,hv->hkv", k, v_new)             # [H, K, V]

    return out, new_state


# ─── Validation against the proven chunked reference ─────────────────────────
if __name__ == "__main__":
    import sys, os
    torch.manual_seed(0)
    # reuse the validated chunk kernel from the 27B dir
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qwen3_6"))
    from chunked_prefill import neuron_chunk_gated_delta_rule

    H, K, V = 8, 128, 128       # 35B: 8 v-heads/core at TP=4
    n_steps = 6

    # random per-step inputs
    state0 = torch.randn(H, K, V) * 0.01
    qs = torch.randn(n_steps, H, K)
    ks = torch.randn(n_steps, H, K)
    vs = torch.randn(n_steps, H, V)
    gs = torch.randn(n_steps, H) * 0.1
    betas = torch.sigmoid(torch.randn(n_steps, H))

    # (A) sequential single-step recurrence
    st = state0.clone()
    outs = []
    for t in range(n_steps):
        o, st = deltanet_recurrent_step(st, qs[t], ks[t], vs[t], gs[t], betas[t],
                                        use_qk_l2norm=True)
        outs.append(o)
    out_seq = torch.stack(outs, 0)          # [T, H, V]
    state_seq = st

    # (B) chunked reference over the whole sequence (C=1), same init state.
    # neuron_chunk_gated_delta_rule expects [B,S,H,D]; B=1.
    q_in = qs.unsqueeze(0)                   # [1,T,H,K]
    k_in = ks.unsqueeze(0)
    v_in = vs.unsqueeze(0)
    g_in = gs.unsqueeze(0)                   # [1,T,H]
    beta_in = betas.unsqueeze(0)
    ref_out, ref_state = neuron_chunk_gated_delta_rule(
        q_in, k_in, v_in, g=g_in, beta=beta_in,
        chunk_size=1, initial_state=state0.unsqueeze(0),
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    ref_out = ref_out.squeeze(0)             # [T,H,V]
    ref_state = ref_state.squeeze(0)         # [H,K,V]

    od = (out_seq - ref_out).abs().max().item()
    sd = (state_seq - ref_state).abs().max().item()
    print(f"out max_diff   = {od:.3e}")
    print(f"state max_diff = {sd:.3e}")
    print("PASS" if (od < 1e-4 and sd < 1e-4) else "FAIL")
