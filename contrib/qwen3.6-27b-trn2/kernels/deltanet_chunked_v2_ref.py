"""
Host-side reference + constant builders for the chunked DeltaNet prefill NKI
kernel (deltanet_chunked_v2). Single source of truth for:

  - the [C,C] constant matrices the kernel needs (no iota in this NKI build),
  - a NumPy/torch reference that mirrors the kernel's math EXACTLY (same
    matmul order, same masks, same doubling-series inverse), so we can debug
    the kernel against a CPU oracle before touching Neuron.

The doubling-series inverse identity (the crux):
  For strictly-lower-triangular (hence nilpotent, T^C = 0) T with C a power of
  two,  (I - T)^{-1} = (I+T)(I+T^2)(I+T^4)...(I+T^{C/2}).
  Proof sketch: product telescopes to  sum_{k=0}^{C-1} T^k = (I-T)^{-1}.

This matches chunked_prefill.neuron_chunk_gated_delta_rule's Woodbury loop,
which builds the same (I - A_str)^{-1} by forward substitution.
"""
import torch


def build_constants(C: int, device="cpu", dtype=torch.float32):
    """Return the [C,C] constants the kernel consumes.

    m_incl:   1.0 where i >= j (lower triangle incl diagonal). Doubles as the
              cumsum operator:  cumsum(g)[i] = sum_{j<=i} g[j] = (m_incl @ g)[i].
    m_strict: 1.0 where i  > j (strict lower triangle). Keeps the strict-lower
              part of A (reference masked_fill(triu(0)) -> keep strict lower).
    eye:      identity.
    """
    idx = torch.arange(C, device=device)
    i = idx.view(C, 1)
    j = idx.view(1, C)
    m_incl = (i >= j).to(dtype)
    m_strict = (i > j).to(dtype)
    eye = torch.eye(C, device=device, dtype=dtype)
    return m_incl, m_strict, eye


def tri_inverse_doubling(T, eye):
    """(I - T)^{-1} for strictly-lower nilpotent T, via the doubling product.
    C must be a power of two. Mirrors the kernel's loop exactly."""
    C = T.shape[-1]
    assert (C & (C - 1)) == 0, "C must be a power of two for the doubling series"
    acc = eye + T                      # (I + T)
    Tp = T @ T                         # T^2
    k = 2
    while k < C:
        acc = acc @ (eye + Tp)         # multiply in (I + T^{k})
        Tp = Tp @ Tp                   # T^{2k}
        k *= 2
    return acc


def ref_chunk_single_head(state, q, k, v, g, beta, C, m_incl, m_strict, eye):
    """Mirror of the per-head math, one head. Shapes:
       state [K,V]; q/k [S,K]; v [S,V]; g/beta [S,1].  S multiple of C.
    Returns output [S,V], new_state [K,V].  Uses the SAME op order the kernel
    will use so divergences localize to a single step."""
    K = q.shape[1]
    V = v.shape[1]
    S = q.shape[0]
    num_chunks = S // C
    scale = 1.0 / (K ** 0.5)
    q = q * scale
    s = state.clone()
    out = torch.zeros(S, V, dtype=torch.float32)
    for ci in range(num_chunks):
        cs, ce = ci * C, ci * C + C
        q_c, k_c, v_c = q[cs:ce], k[cs:ce], v[cs:ce]
        g_c, b_c = g[cs:ce], beta[cs:ce]              # [C,1]

        g_cum = m_incl @ g_c                          # [C,1] cumsum
        # decay mask dm[a,b] = exp(g_cum[a]-g_cum[b]) for a>=b else 0.
        # MUST mask BEFORE exp: in the upper triangle (a<b) gdiff>0 and, with
        # real-model gates (g strongly negative), exp(+large)=inf → inf*0=NaN.
        # Zeroing gdiff in the upper tri first makes those entries exp(0)=1,
        # then the final *m_incl drops them. Exact for the lower tri (g_cum is
        # monotone non-increasing since g<=0, so gdiff<=0 there).
        gdiff = g_cum @ torch.ones(1, C) - torch.ones(C, 1) @ g_cum.transpose(0, 1)
        dm = torch.exp(gdiff * m_incl) * m_incl       # [C,C]

        v_beta = v_c * b_c                            # [C,V]
        k_beta = k_c * b_c                            # [C,K]

        # A_str = -(k_beta @ k_c.T) * dm, keep strict lower
        A_str = -(k_beta @ k_c.transpose(0, 1)) * dm
        A_str = A_str * m_strict
        A = tri_inverse_doubling(A_str, eye)          # (I - A_str)^{-1}

        v_corrected = A @ v_beta                       # [C,V]
        k_cumdecay = A @ (k_beta * torch.exp(g_cum))   # [C,K]

        intra = (q_c @ k_c.transpose(0, 1)) * dm
        intra = intra * m_incl                         # keep lower incl diag
        v_prime = k_cumdecay @ s                       # [C,V]
        v_new = v_corrected - v_prime
        attn_inter = (q_c * torch.exp(g_cum)) @ s      # [C,V]
        out[cs:ce] = attn_inter + intra @ v_new

        total_decay = g_cum[-1]                        # scalar [1]
        s = s * torch.exp(total_decay)
        k_weighted = k_c * torch.exp(total_decay - g_cum)   # [C,K]
        s = s + k_weighted.transpose(0, 1) @ v_new          # [K,V]
    return out, s


if __name__ == "__main__":
    # Cross-check this mirror against chunked_prefill (the HF-exact oracle).
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chunked_prefill import neuron_chunk_gated_delta_rule

    import torch.nn.functional as F
    torch.manual_seed(0)
    C = 16
    H, S, K, V = 1, 64, 128, 128
    # Real model L2-normalizes q,k per-head (decode kernel does this); the
    # doubling-series inverse is only fp32-stable when A_str is small, which
    # holds precisely because q,k are normalized. Mirror that here.
    q = F.normalize(torch.randn(S, K), p=2, dim=-1)
    k = F.normalize(torch.randn(S, K), p=2, dim=-1)
    v = torch.randn(S, V)
    g = torch.randn(S, 1) * 0.1
    beta = torch.sigmoid(torch.randn(S, 1))
    st = torch.randn(K, V) * 0.01

    m_incl, m_strict, eye = build_constants(C)
    out, ns = ref_chunk_single_head(st, q, k, v, g, beta, C, m_incl, m_strict, eye)

    # oracle expects [B,S,H,*]; build B=H=1. NOTE chunked_prefill applies the
    # 1/sqrt(K) scale internally, so pass UNSCALED q (our ref scales internally too).
    qd = q.view(1, S, 1, K); kd = k.view(1, S, 1, K); vd = v.view(1, S, 1, V)
    gd = g.view(1, S, 1); bd = beta.view(1, S, 1)
    o_ref, s_ref = neuron_chunk_gated_delta_rule(
        qd.clone(), kd.clone(), vd.clone(), g=gd.clone(), beta=bd.clone(),
        chunk_size=C, initial_state=st.view(1, 1, K, V).clone(),
        output_final_state=True, use_qk_l2norm_in_kernel=False,
    )
    o_ref = o_ref.view(S, V); s_ref = s_ref.view(K, V)
    od = (out - o_ref).abs().max().item()
    sd = (ns - s_ref).abs().max().item()
    print(f"mirror-vs-oracle  out_max_diff={od:.2e}  state_max_diff={sd:.2e}")
    print("PASS" if od < 1e-4 and sd < 1e-4 else "FAIL")
