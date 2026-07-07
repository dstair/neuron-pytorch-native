"""
Chunked DeltaNet prefill NKI kernel (v2) — ONE @nki.jit call per layer.

Replaces the per-token Python loop in forward_prefill (which unrolls to ~49k FX
nodes / layer) with a single custom call that processes the whole sequence in
chunks of C, using the closed-form chunked gated-delta-rule.

Math + op order validated on CPU against the HF-exact oracle in
deltanet_chunked_v2_ref.py (out/state diff ~1e-7). Crux: the Woodbury delta-rule
correction A = (I - A_str)^{-1} for strictly-lower nilpotent A_str is built with a
DOUBLING PRODUCT (I+T)(I+T^2)... = log2(C) static matmuls — no data-dependent
slicing. Stable in fp32 ONLY because q,k are L2-normalized internally (done here).

This kernel takes ALREADY-PROJECTED, ALREADY-L2-NORMED-OR-NOT q/k/v + gates. To
keep it self-contained and match the reference exactly, it L2-normalizes q,k and
applies the 1/sqrt(K) q-scale internally. Gates g (log-decay) and beta are passed
precomputed per (token, head) — same as the decode kernel computes them.

Layout (per TP core; flattened head-major like deltanet_full / deltanet_chunked):
  state:   [V_HEADS*K_DIM, V_DIM]            f32   in
  q,k:     [V_HEADS*S, K_DIM]                f32   (head-major: rows h*S+t)
  v:       [V_HEADS*S, V_DIM]                f32
  g:       [V_HEADS*S, 1]                    f32   log-decay per (h,t)
  beta:    [V_HEADS*S, 1]                    f32
  m_incl:  [C, C]  f32  (i>=j) — also the cumsum operator
  m_strict:[C, C]  f32  (i> j)
  eye:     [C, C]  f32

Returns:
  output:    [V_HEADS*S, V_DIM]  f32   (ungated raw delta-rule output)
  new_state: [V_HEADS*K_DIM, V_DIM] f32

NOTE: this returns the RAW attention output (pre RMSNormGated / pre out_proj), so
it can be validated directly against ref_chunk_single_head. Gating + out_proj are
applied outside (Phase B will fuse them).
"""
import math
import os as _os
import nki
import nki.isa as nisa
import nki.language as nl

# 35B-A3B per-core (TP=4): DeltaNet has 16 K-heads / 32 V-heads globally; per core
# V_HEADS=8 (32/4), K_HEADS=4 (caller expands k-heads -> v-heads before calling).
# 27B used V_HEADS=12. K_DIM/V_DIM=128 identical. Env-overridable for other TP.
K_DIM = 128
V_DIM = 128
V_HEADS = int(_os.environ.get("DN_V_HEADS", "8"))
RMS_EPS = 1e-6


@nki.jit
def nki_deltanet_chunked_prefill_v2(
    state,      # [V_HEADS*K_DIM, V_DIM] f32
    query,      # [V_HEADS*S, K_DIM] f32
    key,        # [V_HEADS*S, K_DIM] f32
    value,      # [V_HEADS*S, V_DIM] f32
    g,          # [V_HEADS*S, 1] f32
    beta,       # [V_HEADS*S, 1] f32
    m_incl,     # [C, C] f32
    m_strict,   # [C, C] f32
    eye,        # [C, C] f32
):
    num_heads = V_HEADS
    total_rows = query.shape[0]
    S = total_rows // num_heads
    C = m_incl.shape[0]
    num_chunks = S // C
    Q_SCALE = 1.0 / math.sqrt(K_DIM)

    out = nl.ndarray((total_rows, V_DIM), dtype=nl.float32, buffer=nl.shared_hbm)
    new_state = nl.ndarray((num_heads * K_DIM, V_DIM), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- constants resident in SBUF (reused across all heads/chunks) ----
    m_incl_s = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=m_incl_s, src=m_incl[0:C, 0:C])
    m_strict_s = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=m_strict_s, src=m_strict[0:C, 0:C])
    eye_s = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=eye_s, src=eye[0:C, 0:C])
    eps1 = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps1, value=RMS_EPS)
    zC1 = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zC1, value=0.0)
    zCC = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zCC, value=0.0)
    onesC = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=onesC, value=1.0)

    for h in nl.sequential_range(num_heads):
        # state for this head: [K_DIM, V_DIM]
        s = nl.ndarray((K_DIM, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s, src=state[h * K_DIM:(h + 1) * K_DIM, 0:V_DIM])
        row0 = h * S
        for c in nl.sequential_range(num_chunks):
            base = row0 + c * C
            _chunk(
                s, out, base,
                query, key, value, g, beta,
                m_incl_s, m_strict_s, eye_s,
                eps1, zC1, zCC, onesC, C, Q_SCALE,
            )
        nisa.dma_copy(dst=new_state[h * K_DIM:(h + 1) * K_DIM, 0:V_DIM], src=s)

    return out, new_state


def _mm(stat, mov, M, N):
    """nc_matmul wrapper: dst[M,N] = sum_P stat[P,M] * mov[P,N] = stat.T @ mov.
    Returns an SBUF tile."""
    p = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=p, stationary=stat, moving=mov)
    o = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src=p)
    return o


def _T(x, P, F):
    """Transpose [P,F] -> [F,P] via nc_transpose."""
    p = nl.ndarray((F, P), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=p, data=x[0:P, 0:F])
    o = nl.ndarray((F, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src=p)
    return o


def _l2norm_rows(x, C, D, eps1, zC1):
    """Per-row (per-partition) L2 normalize x[C,D] over the free axis D.

    MUST match torch F.normalize(p=2, eps=1e-6) = x / max(||x||, eps), NOT the
    old x / sqrt(ss + eps). For near-zero rows (conv+SiLU can produce them) the
    two differ by ~1000x: old gave x/sqrt(1e-6)=x/1e-3, F.normalize gives
    x/max(~0,1e-6)=x/1e-6. That mismatch = the real-model coherence bug (10%
    near-zero rows → cos 0.95/layer → layer-5 cliff). Floor semantics fixes it."""
    ss = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    sq = nl.ndarray((C, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq,
                    op=nl.square, data=x, bias=zC1, scale=1.0,
                    reduce_op=nl.add, reduce_res=ss, reduce_cmd=nisa.reduce_cmd.reset_reduce)
    # norm = sqrt(ss); denom = max(norm, eps); rinv = 1/denom  (F.normalize floor)
    norm = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=norm, op=nl.sqrt, data=ss, bias=zC1, scale=1.0)
    denom = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=denom, data=norm, op0=nl.maximum, operand0=RMS_EPS)
    rinv = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rinv, op=nl.reciprocal, data=denom, bias=zC1, scale=1.0)
    o = nl.ndarray((C, D), dtype=nl.float32, buffer=nl.sbuf)
    # per-partition scalar broadcast over free axis
    nisa.tensor_scalar(dst=o, data=x, op0=nl.multiply, operand0=rinv)
    return o


def _chunk(s, out, base, query, key, value, g, beta,
           m_incl, m_strict, eye, eps1, zC1, zCC, onesC, C, Q_SCALE):
    """One chunk for one head. Mutates s (state) in place; writes out[base:base+C].
    Mirrors deltanet_chunked_v2_ref.ref_chunk_single_head exactly."""
    # ---- load chunk tiles [C, *] ----
    q_in = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_in, src=query[base:base + C, 0:K_DIM])
    k_in = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_in, src=key[base:base + C, 0:K_DIM])
    v_c = nl.ndarray((C, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_c, src=value[base:base + C, 0:V_DIM])
    g_c = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_c, src=g[base:base + C, 0:1])
    b_c = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_c, src=beta[base:base + C, 0:1])

    # ---- L2 normalize q,k; scale q ----
    k_c = _l2norm_rows(k_in, C, K_DIM, eps1, zC1)
    qn = _l2norm_rows(q_in, C, K_DIM, eps1, zC1)
    qS = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=qS, data=qn, op0=nl.multiply, operand0=Q_SCALE)

    # ---- g_cum = m_incl @ g  (= (m_incl^T)^T @ g via _mm(m_incl^T, g)) ----
    mi_T = _T(m_incl, C, C)                      # [C,C]
    g_cum = _mm(mi_T, g_c, C, 1)                 # [C,1]
    # gdiff[a,b] = g_cum[a]-g_cum[b]; via outer with ones
    g_cum_row = _T(g_cum, C, 1)                  # [1,C]
    gA = _mm(g_cum_row, onesC, C, C)             # col broadcast -> [C,C], gA[a,b]=g_cum[a]
    gB = _mm(onesC, g_cum_row, C, C)             # row broadcast -> [C,C], gB[a,b]=g_cum[b]
    gdiff = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gdiff, data1=gA, data2=gB, op=nl.subtract)
    # MUST mask gdiff to the lower tri BEFORE exp. With real-model gates (g
    # strongly negative), the upper tri has gdiff>0 → exp(+large)=inf → inf*0=NaN
    # when we then multiply by m_incl. Zeroing the upper tri of gdiff first makes
    # those exp(0)=1, harmlessly dropped by the final *m_incl. Exact for the lower
    # tri (g_cum monotone non-increasing since g<=0 ⇒ gdiff<=0 there).
    nisa.tensor_tensor(dst=gdiff, data1=gdiff, data2=m_incl, op=nl.multiply)
    dm = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=dm, op=nl.exp, data=gdiff, bias=zC1, scale=1.0)
    nisa.tensor_tensor(dst=dm, data1=dm, data2=m_incl, op=nl.multiply)  # mask lower

    # ---- v_beta, k_beta ----
    v_beta = nl.ndarray((C, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=v_beta, data=v_c, op0=nl.multiply, operand0=b_c)
    k_beta = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=k_beta, data=k_c, op0=nl.multiply, operand0=b_c)

    # ---- A_str = -(k_beta @ k_c.T) * dm * m_strict ----
    kb_T = _T(k_beta, C, K_DIM)                  # [K,C]
    kc_T = _T(k_c, C, K_DIM)                     # [K,C]
    kk = _mm(kb_T, kc_T, C, C)                   # k_beta@k_c.T [C,C]
    A_str = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=A_str, data1=kk, data2=dm, op=nl.multiply)
    nisa.tensor_scalar(dst=A_str, data=A_str, op0=nl.multiply, operand0=-1.0)
    nisa.tensor_tensor(dst=A_str, data1=A_str, data2=m_strict, op=nl.multiply)

    # ---- A = (I - A_str)^{-1} via doubling product ----
    A = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=A, data1=eye, data2=A_str, op=nl.add)   # I + T
    Tp = _mm(_T(A_str, C, C), A_str, C, C)                         # T^2
    kk2 = 2
    while kk2 < C:
        eyeTp = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=eyeTp, data1=eye, data2=Tp, op=nl.add)
        A = _mm(_T(A, C, C), eyeTp, C, C)                          # A @ (I+Tp)
        Tp = _mm(_T(Tp, C, C), Tp, C, C)                           # Tp^2
        kk2 *= 2

    # ---- v_corrected = A @ v_beta ; k_cumdecay = A @ (k_beta*exp(g_cum)) ----
    A_T = _T(A, C, C)
    v_corr = _mm(A_T, v_beta, C, V_DIM)          # [C,V]
    expg = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=expg, op=nl.exp, data=g_cum, bias=zC1, scale=1.0)
    kbexp = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=kbexp, data=k_beta, op0=nl.multiply, operand0=expg)
    k_cumdecay = _mm(A_T, kbexp, C, K_DIM)       # [C,K]

    # ---- intra = (qS @ k_c.T) * dm * m_incl ----
    qS_T = _T(qS, C, K_DIM)                       # [K,C]
    qk = _mm(qS_T, kc_T, C, C)                    # qS@k_c.T [C,C]
    intra = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=intra, data1=qk, data2=dm, op=nl.multiply)
    nisa.tensor_tensor(dst=intra, data1=intra, data2=m_incl, op=nl.multiply)

    # ---- v_new = v_corr - k_cumdecay @ s ----
    kcd_T = _T(k_cumdecay, C, K_DIM)              # [K,C]
    vprime = _mm(kcd_T, s, C, V_DIM)              # k_cumdecay@s [C,V]
    v_new = nl.ndarray((C, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=v_new, data1=v_corr, data2=vprime, op=nl.subtract)

    # ---- out = (qS*exp(g_cum)) @ s + intra @ v_new ----
    qSe = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=qSe, data=qS, op0=nl.multiply, operand0=expg)
    qSe_T = _T(qSe, C, K_DIM)                      # [K,C]
    attn_inter = _mm(qSe_T, s, C, V_DIM)           # [C,V]
    intra_T = _T(intra, C, C)
    o_chunk = _mm(intra_T, v_new, C, V_DIM)        # intra@v_new [C,V]
    nisa.tensor_tensor(dst=o_chunk, data1=o_chunk, data2=attn_inter, op=nl.add)
    nisa.dma_copy(dst=out[base:base + C, 0:V_DIM], src=o_chunk)

    # ---- state update: s = s*exp(g_cum[-1]) + (k_c*exp(g_cum[-1]-g_cum)).T @ v_new ----
    # total_decay = g_cum[C-1] (scalar). Extract via a MATMUL PICKOFF using eye's
    # last column (one-hot at row C-1): td = sum_p eye[p,C-1]*g_cum[p] = g_cum[C-1].
    # A direct `tensor_copy(src=g_cum[C-1:C, ...])` reads from partition C-1 into
    # partition 0 — a cross-partition copy that PASSES the CPU simulator but FAILS
    # the on-device ISA check (NCC_IXCG864). The matmul keeps everything aligned.
    eye_last = eye[0:C, C - 1:C]                   # [C,1] one-hot column at row C-1
    td = _mm(eye_last, g_cum, 1, 1)                # [1,1] = g_cum[C-1]
    exp_td = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    z11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=z11, value=0.0)
    nisa.activation(dst=exp_td, op=nl.exp, data=td, bias=z11, scale=1.0)
    # decay state: s *= exp_td (scalar). Use tensor_scalar with [1,1]? needs per-partition.
    # Broadcast exp_td[1,1] -> [K,1] via matmul ones[1,K]
    onesK = nl.ndarray((1, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=onesK, value=1.0)
    exp_td_col = _mm(onesK, exp_td, K_DIM, 1)      # [K,1]
    nisa.tensor_scalar(dst=s, data=s, op0=nl.multiply, operand0=exp_td_col)
    # k_weighted = k_c * exp(td - g_cum). Compute exp(td - g_cum) DIRECTLY — the
    # argument is <=0 (td=g_cum[-1] is the most negative cumsum), so bounded.
    # The old form exp_td * exp(-g_cum) overflowed: exp(-g_cum)=exp(+236)=inf with
    # real-model gates, then inf*exp_td(~0)=NaN. Broadcast td[1,1]->[C,1] via
    # matmul, subtract g_cum, exp.
    onesC_col = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=onesC_col, value=1.0)
    td_C = _mm(onesC_col, td, C, 1)                # [C,1] broadcast of td scalar
    tdmg_arg = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tdmg_arg, data1=td_C, data2=g_cum, op=nl.subtract)  # td - g_cum
    tdmg = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=tdmg, op=nl.exp, data=tdmg_arg, bias=zC1, scale=1.0)
    k_w = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=k_w, data=k_c, op0=nl.multiply, operand0=tdmg)
    # s += k_w.T @ v_new : stat=k_w[C,K], mov=v_new[C,V] -> [K,V]
    upd = _mm(k_w, v_new, K_DIM, V_DIM)
    nisa.tensor_tensor(dst=s, data1=s, data2=upd, op=nl.add)
