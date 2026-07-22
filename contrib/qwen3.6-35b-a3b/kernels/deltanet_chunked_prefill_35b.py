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

Layout (per TP core; flattened batch/head-major):
  state:   [B, V_HEADS*K_DIM, V_DIM]         f32   in
  q,k:     [B*V_HEADS*S, K_DIM]              f32   rows (b*H+h)*S+t
  v:       [B*V_HEADS*S, V_DIM]              f32
  g:       [B*V_HEADS*S, 1]                  f32   log-decay per (b,h,t)
  beta:    [B*V_HEADS*S, 1]                  f32
  m_incl:  [C, C]  f32  (i>=j) — also the cumsum operator
  m_strict:[C, C]  f32  (i> j)
  eye:     [C, C]  f32

Returns:
  output:    [B*V_HEADS*S, V_DIM]             f32   ungated raw output
  new_state: [B, V_HEADS*K_DIM, V_DIM]        f32

For even batch sizes with C16, DN_PAIRED_BATCH=1 packs adjacent independent C16
token blocks into block-diagonal C32 solves while retaining private recurrent
states. Other shapes use the original sequential-stream path.

NOTE: this returns the RAW attention output (pre RMSNormGated / pre out_proj), so
it can be validated directly against ref_chunk_single_head. Gating + out_proj are
applied outside (Phase B will fuse them).
"""
import math
import os as _os
import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.allocator import BufferManager

# 35B-A3B per-core (TP=4): DeltaNet has 16 K-heads / 32 V-heads globally; per core
# V_HEADS=8 (32/4), K_HEADS=4 (caller expands k-heads -> v-heads before calling).
# 27B used V_HEADS=12. K_DIM/V_DIM=128 identical. Env-overridable for other TP.
K_DIM = 128
V_DIM = 128
V_HEADS = int(_os.environ.get("DN_V_HEADS", "8"))
RMS_EPS = 1e-6
STABLE_C32 = _os.environ.get("DN_STABLE_C32", "1") == "1"
PAIRED_BATCH = _os.environ.get("DN_PAIRED_BATCH", "0") == "1"


@nki.jit
def nki_deltanet_chunked_prefill_v2(
    state,      # [B, V_HEADS*K_DIM, V_DIM] f32
    query,      # [B*V_HEADS*S, K_DIM] f32, batch/head-major
    key,        # [B*V_HEADS*S, K_DIM] f32, batch/head-major
    value,      # [B*V_HEADS*S, V_DIM] f32, batch/head-major
    g,          # [B*V_HEADS*S, 1] f32
    beta,       # [B*V_HEADS*S, 1] f32
    m_incl,     # [C, C] f32
    m_strict,   # [C, C] f32
    eye,        # [C, C] f32
):
    batch_size = state.shape[0]
    num_streams = batch_size * V_HEADS
    total_rows = query.shape[0]
    S = total_rows // num_streams
    C = m_incl.shape[0]
    num_chunks = S // C
    Q_SCALE = 1.0 / math.sqrt(K_DIM)

    out = nl.ndarray((total_rows, V_DIM), dtype=nl.float32, buffer=nl.shared_hbm)
    new_state = nl.ndarray(
        (batch_size, V_HEADS * K_DIM, V_DIM),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )

    # C32's direct block solve stages masks per chunk to avoid carrying full
    # padded tiles across the nested head/chunk sequential loops.
    if C == 32 and STABLE_C32:
        m_incl_s = m_incl
        m_strict_s = m_strict
        eye_s = eye
    else:
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
    if C == 32 and STABLE_C32:
        zCC = zC1
        # Reused sequentially by each C32 chunk so the factor scope contains
        # only C16 RHS tiles.
        c32_key_scratch = nl.ndarray(
            (C, K_DIM), dtype=nl.float32, buffer=nl.shared_hbm
        )
        c32_q_scratch = nl.ndarray(
            (C, K_DIM), dtype=nl.float32, buffer=nl.shared_hbm
        )
        c32_vprime_scratch = nl.ndarray(
            (C, V_DIM), dtype=nl.float32, buffer=nl.shared_hbm
        )
        c32_inverse_scratch = nl.ndarray(
            (C, C), dtype=nl.float32, buffer=nl.shared_hbm
        )
        c32_vcorr_scratch = nl.ndarray(
            (C, V_DIM), dtype=nl.float32, buffer=nl.shared_hbm
        )
        c32_state_scratch = nl.ndarray(
            (K_DIM, V_DIM), dtype=nl.float32, buffer=nl.shared_hbm
        )
        c32_row_scratch = nl.ndarray(
            (C, C), dtype=nl.float32, buffer=nl.shared_hbm
        )
    else:
        zCC = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zCC, value=0.0)
        c32_key_scratch = None
        c32_q_scratch = None
        c32_vprime_scratch = None
        c32_inverse_scratch = None
        c32_vcorr_scratch = None
        c32_state_scratch = None
        c32_row_scratch = None
    onesC = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=onesC, value=1.0)

    if PAIRED_BATCH and batch_size % 2 == 0 and C == 16:
        pair_rows = 2 * C
        pair_m_incl = nl.ndarray(
            (pair_rows, pair_rows), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=pair_m_incl, value=0.0)
        nisa.dma_copy(
            dst=pair_m_incl[0:C, 0:C], src=m_incl[0:C, 0:C]
        )
        nisa.dma_copy(
            dst=pair_m_incl[C:pair_rows, C:pair_rows],
            src=m_incl[0:C, 0:C],
        )
        pair_m_strict = nl.ndarray(
            (pair_rows, pair_rows), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=pair_m_strict, value=0.0)
        nisa.dma_copy(
            dst=pair_m_strict[0:C, 0:C], src=m_strict[0:C, 0:C]
        )
        nisa.dma_copy(
            dst=pair_m_strict[C:pair_rows, C:pair_rows],
            src=m_strict[0:C, 0:C],
        )
        pair_eye = nl.ndarray(
            (pair_rows, pair_rows), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=pair_eye, value=0.0)
        nisa.dma_copy(dst=pair_eye[0:C, 0:C], src=eye[0:C, 0:C])
        nisa.dma_copy(
            dst=pair_eye[C:pair_rows, C:pair_rows],
            src=eye[0:C, 0:C],
        )
        pair_eps = nl.ndarray(
            (pair_rows, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=pair_eps, value=RMS_EPS)
        pair_zero = nl.ndarray(
            (pair_rows, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=pair_zero, value=0.0)
        pair_ones = nl.ndarray(
            (1, pair_rows), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=pair_ones, value=1.0)
        pair_key_scratch = nl.ndarray(
            (pair_rows, K_DIM), dtype=nl.float32, buffer=nl.shared_hbm
        )
        pair_gcum_scratch = nl.ndarray(
            (pair_rows, 1), dtype=nl.float32, buffer=nl.shared_hbm
        )
        pair_vcorr_scratch = nl.ndarray(
            (pair_rows, V_DIM), dtype=nl.float32, buffer=nl.shared_hbm
        )
        pair_kcd_scratch = nl.ndarray(
            (pair_rows, K_DIM), dtype=nl.float32, buffer=nl.shared_hbm
        )
        pair_qse_scratch = nl.ndarray(
            (pair_rows, K_DIM), dtype=nl.float32, buffer=nl.shared_hbm
        )
        pair_intra_scratch = nl.ndarray(
            (pair_rows, pair_rows), dtype=nl.float32, buffer=nl.shared_hbm
        )
        # Each pair iteration retains two independent recurrent states. The
        # token-local solve is block diagonal across their C16 row banks.
        for head in nl.sequential_range(V_HEADS):
            for pair in nl.sequential_range(batch_size // 2):
                batch0 = pair * 2
                batch1 = batch0 + 1
                s0 = nl.ndarray(
                    (K_DIM, V_DIM), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=s0,
                    src=state[
                        batch0, head * K_DIM:(head + 1) * K_DIM, 0:V_DIM
                    ],
                )
                s1 = nl.ndarray(
                    (K_DIM, V_DIM), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=s1,
                    src=state[
                        batch1, head * K_DIM:(head + 1) * K_DIM, 0:V_DIM
                    ],
                )
                row0 = (batch0 * V_HEADS + head) * S
                row1 = (batch1 * V_HEADS + head) * S
                for c in nl.sequential_range(num_chunks):
                    base0 = row0 + c * C
                    base1 = row1 + c * C
                    _chunk_pair_c16(
                        s0, s1, out, base0, base1,
                        query, key, value, g, beta,
                        pair_m_incl, pair_m_strict, pair_eye, eye_s,
                        pair_eps, pair_zero, pair_ones,
                        pair_key_scratch, pair_gcum_scratch,
                        pair_vcorr_scratch, pair_kcd_scratch,
                        pair_qse_scratch, pair_intra_scratch,
                        zC1, onesC, C, Q_SCALE,
                    )
                nisa.dma_copy(
                    dst=new_state[
                        batch0, head * K_DIM:(head + 1) * K_DIM, 0:V_DIM
                    ],
                    src=s0,
                )
                nisa.dma_copy(
                    dst=new_state[
                        batch1, head * K_DIM:(head + 1) * K_DIM, 0:V_DIM
                    ],
                    src=s1,
                )
    else:
        for stream in nl.sequential_range(num_streams):
            batch = stream // V_HEADS
            head = stream % V_HEADS
            s = nl.ndarray((K_DIM, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=s,
                src=state[
                    batch, head * K_DIM:(head + 1) * K_DIM, 0:V_DIM
                ],
            )
            row0 = stream * S
            for c in nl.sequential_range(num_chunks):
                base = row0 + c * C
                _chunk(
                    s, out, base,
                    query, key, value, g, beta,
                    m_incl_s, m_strict_s, eye_s,
                    eps1, zC1, zCC, onesC, c32_key_scratch,
                    c32_q_scratch, c32_vprime_scratch,
                    c32_inverse_scratch, c32_vcorr_scratch,
                    c32_state_scratch, c32_row_scratch, C, Q_SCALE,
                )
            nisa.dma_copy(
                dst=new_state[
                    batch, head * K_DIM:(head + 1) * K_DIM, 0:V_DIM
                ],
                src=s,
            )

    return out, new_state


def _chunk_pair_c16(
    s0, s1, out, base0, base1, query, key, value, g, beta,
    m_incl, m_strict, pair_eye, local_eye, eps, zero, pair_ones,
    key_scratch, gcum_scratch, vcorr_scratch, kcd_scratch,
    qse_scratch, intra_scratch, local_zero, local_ones, C, Q_SCALE,
):
    """Run two independent C16 chunks as one block-diagonal row tile."""
    P = 2 * C

    q_in = nl.ndarray((P, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_in[0:C, 0:K_DIM], src=query[base0:base0 + C, 0:K_DIM])
    nisa.dma_copy(dst=q_in[C:P, 0:K_DIM], src=query[base1:base1 + C, 0:K_DIM])
    k_in = nl.ndarray((P, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_in[0:C, 0:K_DIM], src=key[base0:base0 + C, 0:K_DIM])
    nisa.dma_copy(dst=k_in[C:P, 0:K_DIM], src=key[base1:base1 + C, 0:K_DIM])
    v_c = nl.ndarray((P, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_c[0:C, 0:V_DIM], src=value[base0:base0 + C, 0:V_DIM])
    nisa.dma_copy(dst=v_c[C:P, 0:V_DIM], src=value[base1:base1 + C, 0:V_DIM])
    g_c = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_c[0:C, 0:1], src=g[base0:base0 + C, 0:1])
    nisa.dma_copy(dst=g_c[C:P, 0:1], src=g[base1:base1 + C, 0:1])
    b_c = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_c[0:C, 0:1], src=beta[base0:base0 + C, 0:1])
    nisa.dma_copy(dst=b_c[C:P, 0:1], src=beta[base1:base1 + C, 0:1])

    k_c = _l2norm_rows(k_in, P, K_DIM, eps, zero)
    qn = _l2norm_rows(q_in, P, K_DIM, eps, zero)
    qS = nl.ndarray((P, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=qS, data=qn, op0=nl.multiply, operand0=Q_SCALE)
    nisa.dma_copy(dst=key_scratch, src=k_c)

    mi_T = _T(m_incl, P, P)
    g_cum = _mm(mi_T, g_c, P, 1)
    nisa.dma_copy(dst=gcum_scratch, src=g_cum)
    g_cum_row = _T(g_cum, P, 1)
    gA = _mm(g_cum_row, pair_ones, P, P)
    gB = _mm(pair_ones, g_cum_row, P, P)
    gdiff = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gdiff, data1=gA, data2=gB, op=nl.subtract)
    nisa.tensor_tensor(dst=gdiff, data1=gdiff, data2=m_incl, op=nl.multiply)
    dm = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=dm, op=nl.exp, data=gdiff, bias=zero, scale=1.0)
    nisa.tensor_tensor(dst=dm, data1=dm, data2=m_incl, op=nl.multiply)

    v_beta = nl.ndarray((P, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=v_beta, data=v_c, op0=nl.multiply, operand0=b_c)
    k_beta = nl.ndarray((P, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=k_beta, data=k_c, op0=nl.multiply, operand0=b_c)
    kb_T = _T(k_beta, P, K_DIM)
    kc_T = _T(k_c, P, K_DIM)
    kk = _mm(kb_T, kc_T, P, P)
    A_str = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=A_str, data1=kk, data2=dm, op=nl.multiply)
    nisa.tensor_scalar(dst=A_str, data=A_str, op0=nl.multiply, operand0=-1.0)
    nisa.tensor_tensor(dst=A_str, data1=A_str, data2=m_strict, op=nl.multiply)

    # Each diagonal block is C16 nilpotent, so only the C16 doubling depth is
    # required even though both blocks occupy a P=32 tile.
    A = _tri_inverse_doubling(A_str, pair_eye, P, C)
    A_T = _T(A, P, P)
    v_corr = _mm(A_T, v_beta, P, V_DIM)
    nisa.dma_copy(dst=vcorr_scratch, src=v_corr)
    expg = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=expg, op=nl.exp, data=g_cum, bias=zero, scale=1.0)
    kbexp = nl.ndarray((P, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=kbexp, data=k_beta, op0=nl.multiply, operand0=expg)
    k_cumdecay = _mm(A_T, kbexp, P, K_DIM)
    nisa.dma_copy(dst=kcd_scratch, src=k_cumdecay)

    qS_T = _T(qS, P, K_DIM)
    qk = _mm(qS_T, kc_T, P, P)
    intra = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=intra, data1=qk, data2=dm, op=nl.multiply)
    nisa.tensor_tensor(dst=intra, data1=intra, data2=m_incl, op=nl.multiply)
    nisa.dma_copy(dst=intra_scratch, src=intra)
    qSe = nl.ndarray((P, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=qSe, data=qS, op0=nl.multiply, operand0=expg)
    nisa.dma_copy(dst=qse_scratch, src=qSe)

    onesK = nl.ndarray((1, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=onesK, value=1.0)
    z11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=z11, value=0.0)
    _finish_paired_bank(
        s0, out, base0, 0,
        kcd_scratch, qse_scratch, intra_scratch,
        key_scratch, gcum_scratch, vcorr_scratch,
        local_eye, local_zero, local_ones, onesK, z11, C,
    )
    _finish_paired_bank(
        s1, out, base1, C,
        kcd_scratch, qse_scratch, intra_scratch,
        key_scratch, gcum_scratch, vcorr_scratch,
        local_eye, local_zero, local_ones, onesK, z11, C,
    )


def _finish_paired_bank(
    s, out, base, row,
    kcd_scratch, qse_scratch, intra_scratch,
    key_scratch, gcum_scratch, vcorr_scratch,
    eye, zC1, onesC, onesK, z11, C,
):
    """Apply one packed block to its private recurrent state."""
    v_corr = nl.ndarray((C, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=v_corr, src=vcorr_scratch[row:row + C, 0:V_DIM]
    )
    k_cumdecay = nl.ndarray(
        (C, K_DIM), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.dma_copy(
        dst=k_cumdecay, src=kcd_scratch[row:row + C, 0:K_DIM]
    )
    kcd_T = _T(k_cumdecay, C, K_DIM)
    vprime = _mm(kcd_T, s, C, V_DIM)
    v_new = nl.ndarray((C, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=v_new, data1=v_corr, data2=vprime, op=nl.subtract)

    qSe = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=qSe, src=qse_scratch[row:row + C, 0:K_DIM])
    qSe_T = _T(qSe, C, K_DIM)
    attn_inter = _mm(qSe_T, s, C, V_DIM)
    intra = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=intra, src=intra_scratch[row:row + C, row:row + C]
    )
    intra_T = _T(intra, C, C)
    o_chunk = _mm(intra_T, v_new, C, V_DIM)
    nisa.tensor_tensor(
        dst=o_chunk, data1=o_chunk, data2=attn_inter, op=nl.add
    )
    nisa.dma_copy(dst=out[base:base + C, 0:V_DIM], src=o_chunk)

    g_cum = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_cum, src=gcum_scratch[row:row + C, 0:1])
    eye_last = eye[0:C, C - 1:C]
    td = _mm(eye_last, g_cum, 1, 1)
    exp_td = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=exp_td, op=nl.exp, data=td, bias=z11, scale=1.0)
    exp_td_col = _mm(onesK, exp_td, K_DIM, 1)
    nisa.tensor_scalar(dst=s, data=s, op0=nl.multiply, operand0=exp_td_col)

    td_C = _mm(onesC, td, C, 1)
    tdmg_arg = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=tdmg_arg, data1=td_C, data2=g_cum, op=nl.subtract
    )
    tdmg = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=tdmg, op=nl.exp, data=tdmg_arg, bias=zC1, scale=1.0)
    k_c = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_c, src=key_scratch[row:row + C, 0:K_DIM])
    k_w = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=k_w, data=k_c, op0=nl.multiply, operand0=tdmg)
    upd = _mm(k_w, v_new, K_DIM, V_DIM)
    nisa.tensor_tensor(dst=s, data1=s, data2=upd, op=nl.add)


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


def _mm_to(stat, mov, dst, M, N):
    """nc_matmul into an explicitly allocated SBUF destination."""
    p = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=p, stationary=stat, moving=mov)
    nisa.tensor_copy(dst=dst, src=p)
    return dst


def _T_to(x, P, F, dst):
    """Transpose [P,F] -> [F,P] into an explicitly allocated SBUF destination."""
    p = nl.ndarray((F, P), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=p, data=x[0:P, 0:F])
    nisa.tensor_copy(dst=dst, src=p)
    return dst


def _c32_tile(sbm, rows, cols, base_partition=0):
    """Allocate an FP32 tile managed by the scoped stable-C32 allocator."""
    return sbm.alloc_stack(
        (rows, cols), dtype=nl.float32, buffer=nl.sbuf,
        base_partition=base_partition, align=64,
    )


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


def _tri_inverse_doubling(T, eye, C, span):
    """Build (I-T)^-1 for a nilpotent block whose width is ``span``."""
    A = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=A, data1=eye, data2=T, op=nl.add)
    Tp = _mm(_T(T, C, C), T, C, C)
    power = 2
    while power < span:
        eyeTp = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=eyeTp, data1=eye, data2=Tp, op=nl.add)
        A = _mm(_T(A, C, C), eyeTp, C, C)
        power *= 2
        # The final square is T^span and cannot contribute to the inverse.
        if power < span:
            Tp = _mm(_T(Tp, C, C), Tp, C, C)
    return A


def _tri_inverse_doubling_to(T, eye, dst, span, sbm):
    """Build a C16 triangular inverse using scoped, reusable SBUF scratch."""
    sbm.open_scope(name="c32_inverse")
    with nl.no_reorder():
        t_trans = _c32_tile(sbm, span, span)
        power_term = _c32_tile(sbm, span, span)
        eye_power = _c32_tile(sbm, span, span)
        a_trans = _c32_tile(sbm, span, span)

        nisa.tensor_tensor(dst=dst, data1=eye, data2=T, op=nl.add)
        _T_to(T, span, span, t_trans)
        _mm_to(t_trans, T, power_term, span, span)

        power = 2
        while power < span:
            nisa.tensor_tensor(
                dst=eye_power, data1=eye, data2=power_term, op=nl.add
            )
            _T_to(dst, span, span, a_trans)
            _mm_to(a_trans, eye_power, dst, span, span)
            power *= 2
            if power < span:
                _T_to(power_term, span, span, t_trans)
                _mm_to(t_trans, power_term, power_term, span, span)
    sbm.close_scope()
    return dst


def _tri_inverse_forward_from_hbm_to(
        strict_hbm, eye, dst, row_scratch_hbm, C, sbm):
    """Build the C32 inverse in the CPU oracle's row-update order."""
    sbm.open_scope(name="c32_forward_inverse_rows")
    with nl.no_reorder():
        nisa.dma_copy(dst=dst, src=eye[0:C, 0:C])
        for row in range(1, C):
            strict_row = _c32_tile(sbm, 1, C)
            row_t = _c32_tile(sbm, C, 1)
            row_update = _c32_tile(sbm, 1, C)
            nisa.memset(dst=strict_row, value=0.0)
            nisa.dma_copy(
                dst=strict_row,
                src=strict_hbm[row:row + 1, 0:C],
            )
            _T_to(strict_row, 1, C, row_t)
            _mm_to(
                row_t, dst, row_update, 1, C
            )
            # ScalarE requires same-start SBUF partitions. DMA through HBM
            # repositions the one-row result at its logical matrix row.
            nisa.dma_copy(
                dst=row_scratch_hbm[row:row + 1, 0:C], src=row_update
            )
            nisa.dma_copy(
                dst=dst[row:row + 1, 0:C],
                src=row_scratch_hbm[row:row + 1, 0:C],
            )
            nisa.dma_copy(
                dst=dst[row:row + 1, row:row + 1],
                src=eye[row:row + 1, row:row + 1],
            )
    sbm.close_scope()
    return dst


def _chunk(s, out, base, query, key, value, g, beta,
           m_incl, m_strict, eye, eps1, zC1, zCC, onesC, c32_key_scratch,
           c32_q_scratch, c32_vprime_scratch, c32_inverse_scratch,
           c32_vcorr_scratch, c32_state_scratch, c32_row_scratch, C, Q_SCALE):
    """One chunk for one head. Mutates s (state) in place; writes out[base:base+C].
    Mirrors deltanet_chunked_v2_ref.ref_chunk_single_head exactly."""
    if C == 32 and STABLE_C32:
        # The factor solve needs only diagonal C16 mask blocks. Deferring the
        # full C32 lower-triangle copy shortens the live range that otherwise
        # collides with the Cx128 kernel schedule.
        half = C // 2
        m_incl00 = nl.ndarray((half, half), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m_incl00, src=m_incl[0:half, 0:half])
        m_incl11 = nl.ndarray((half, half), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m_incl11, src=m_incl[half:C, half:C])
        m_strict00 = nl.ndarray((half, half), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m_strict00, src=m_strict[0:half, 0:half])
        m_strict11 = nl.ndarray((half, half), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m_strict11, src=m_strict[half:C, half:C])
        eye16 = nl.ndarray((half, half), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=eye16, src=eye[0:half, 0:half])

    # ---- load chunk tiles [C, *] ----
    q_in = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_in, src=query[base:base + C, 0:K_DIM])
    k_in = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_in, src=key[base:base + C, 0:K_DIM])
    if C == 32 and STABLE_C32:
        # The factor phase does not depend on values. Load each RHS half only
        # after B00/B11/BXB are available so no C32x128 value tile overlaps
        # the scoped inverse buffers.
        v_c = None
    else:
        v_c = nl.ndarray((C, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_c, src=value[base:base + C, 0:V_DIM])
    if C == 32 and STABLE_C32:
        g0_in = nl.ndarray((half, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=g0_in, src=g[base:base + half, 0:1])
        g1_in = nl.ndarray((half, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=g1_in, src=g[base + half:base + C, 0:1])
    else:
        g_c = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=g_c, src=g[base:base + C, 0:1])
    if C == 32 and STABLE_C32:
        b_c = None
    else:
        b_c = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_c, src=beta[base:base + C, 0:1])

    # ---- L2 normalize q,k; scale q ----
    k_c = _l2norm_rows(k_in, C, K_DIM, eps1, zC1)
    qn = _l2norm_rows(q_in, C, K_DIM, eps1, zC1)
    qS = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=qS, data=qn, op0=nl.multiply, operand0=Q_SCALE)
    if C == 32 and STABLE_C32:
        # Full normalized q/k tiles are needed only after the factor solve.
        # Move them out of SBUF before the scoped C16 factor/RHS phases.
        nisa.dma_copy(dst=c32_key_scratch, src=k_c)
        nisa.dma_copy(dst=c32_q_scratch, src=qS)

    # ---- g_cum = m_incl @ g  (= (m_incl^T)^T @ g via _mm(m_incl^T, g)) ----
    if C == 32 and STABLE_C32:
        ones16 = onesC[0:1, 0:half]
        mi0_T = _T(m_incl00, half, half)
        g0 = _mm(mi0_T, g0_in, half, 1)

        mi1_T = _T(m_incl11, half, half)
        g1 = _mm(mi1_T, g1_in, half, 1)

        td0 = _mm(eye16[0:half, half - 1:half], g0, 1, 1)
        td0_rows = _mm(ones16, td0, half, 1)
        nisa.tensor_tensor(dst=g1, data1=g1, data2=td0_rows, op=nl.add)
    else:
        mi_T = _T(m_incl, C, C)                  # [C,C]
        g_cum = _mm(mi_T, g_c, C, 1)             # [C,1]
        # gdiff[a,b] = g_cum[a]-g_cum[b]; via outer with ones
        g_cum_row = _T(g_cum, C, 1)              # [1,C]
        gA = _mm(g_cum_row, onesC, C, C)         # col broadcast
        gB = _mm(onesC, g_cum_row, C, C)         # row broadcast
        gdiff = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gdiff, data1=gA, data2=gB, op=nl.subtract)
        # MUST mask gdiff to the lower tri BEFORE exp. With real-model gates
        # the upper tri can overflow before a later multiply by zero.
        nisa.tensor_tensor(dst=gdiff, data1=gdiff, data2=m_incl, op=nl.multiply)
        dm = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=dm, op=nl.exp, data=gdiff, bias=zC1, scale=1.0)
        nisa.tensor_tensor(dst=dm, data1=dm, data2=m_incl, op=nl.multiply)

    # ---- A = (I - A_str)^-1 ----
    if C == 32 and STABLE_C32:
        # Factor the lower-triangular system into two C16 diagonal systems:
        # T = D + X, where D is block diagonal and X is only lower-left.
        # B=(I-D)^-1 and X*B*X=0, so (I-T)^-1 = B + B*X*B exactly.
        # Keep every factor input C16-wide; the full normalized key remains in
        # HBM until the factor scope has closed.
        c32_sbm = BufferManager(
            0,
            nl.tile_size.total_available_sbuf_size,
            use_auto_alloc=True,
        )
        c32_sbm.open_scope(name="c32_factor")
        nisa.dma_copy(
            dst=c32_inverse_scratch, src=m_strict[0:C, 0:C]
        )
        k0_factor = nl.ndarray((half, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=k0_factor, src=c32_key_scratch[0:half, 0:K_DIM]
        )
        k1_factor = nl.ndarray((half, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=k1_factor, src=c32_key_scratch[half:C, 0:K_DIM]
        )
        b0_factor = nl.ndarray((half, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=b0_factor, src=beta[base:base + half, 0:1])
        b1_factor = nl.ndarray((half, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=b1_factor, src=beta[base + half:base + C, 0:1]
        )
        kb0_factor = nl.ndarray(
            (half, K_DIM), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=kb0_factor, data=k0_factor, op0=nl.multiply,
            operand0=b0_factor,
        )
        kb1_factor = nl.ndarray(
            (half, K_DIM), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=kb1_factor, data=k1_factor, op0=nl.multiply,
            operand0=b1_factor,
        )
        kb0_T = _T(kb0_factor, half, K_DIM)
        kb1_T = _T(kb1_factor, half, K_DIM)
        k0_factor_T = _T(k0_factor, half, K_DIM)
        k1_factor_T = _T(k1_factor, half, K_DIM)
        T00 = _mm(kb0_T, k0_factor_T, half, half)
        T11_raw = _mm(kb1_T, k1_factor_T, half, half)
        X_raw = _mm(kb1_T, k0_factor_T, half, half)
        T11 = nl.ndarray((half, half), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=T11, src=T11_raw)
        X = _c32_tile(c32_sbm, half, half)
        nisa.tensor_copy(dst=X, src=X_raw)
        g0_row = _T(g0, half, 1)
        g1_row = _T(g1, half, 1)

        dm00_a = _mm(g0_row, ones16, half, half)
        dm00_b = _mm(ones16, g0_row, half, half)
        dm00 = nl.ndarray((half, half), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dm00, data1=dm00_a, data2=dm00_b, op=nl.subtract)
        nisa.tensor_tensor(
            dst=dm00, data1=dm00, data2=m_incl00, op=nl.multiply
        )
        nisa.activation(
            dst=dm00, op=nl.exp, data=dm00, bias=zC1[0:half, 0:1], scale=1.0
        )
        nisa.tensor_tensor(
            dst=dm00, data1=dm00, data2=m_incl00, op=nl.multiply
        )
        nisa.tensor_tensor(dst=T00, data1=T00, data2=dm00, op=nl.multiply)
        nisa.tensor_scalar(dst=T00, data=T00, op0=nl.multiply, operand0=-1.0)
        nisa.tensor_tensor(
            dst=T00, data1=T00, data2=m_strict00, op=nl.multiply
        )

        dm11_a = _mm(g1_row, ones16, half, half)
        dm11_b = _mm(ones16, g1_row, half, half)
        dm11 = nl.ndarray((half, half), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dm11, data1=dm11_a, data2=dm11_b, op=nl.subtract)
        nisa.tensor_tensor(dst=dm11, data1=dm11, data2=m_incl11, op=nl.multiply)
        nisa.activation(
            dst=dm11, op=nl.exp, data=dm11, bias=zC1[0:half, 0:1], scale=1.0
        )
        nisa.tensor_tensor(dst=dm11, data1=dm11, data2=m_incl11, op=nl.multiply)
        nisa.tensor_tensor(dst=T11, data1=T11, data2=dm11, op=nl.multiply)
        nisa.tensor_scalar(dst=T11, data=T11, op0=nl.multiply, operand0=-1.0)
        nisa.tensor_tensor(dst=T11, data1=T11, data2=m_strict11, op=nl.multiply)

        # Every row in the lower-left block is strictly below every column,
        # so its m_strict entries are all one.
        dm21_a = _mm(g1_row, ones16, half, half)
        dm21_b = _mm(ones16, g0_row, half, half)
        dm21 = nl.ndarray((half, half), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dm21, data1=dm21_a, data2=dm21_b, op=nl.subtract)
        nisa.activation(
            dst=dm21, op=nl.exp, data=dm21, bias=zC1[0:half, 0:1], scale=1.0
        )
        nisa.tensor_tensor(dst=X, data1=X, data2=dm21, op=nl.multiply)
        nisa.tensor_scalar(dst=X, data=X, op0=nl.multiply, operand0=-1.0)

        # Stage only the strict lower matrix in HBM. The upper block is never
        # read by the row-wise inverse, so it does not need initialization.
        nisa.dma_copy(
            dst=c32_inverse_scratch[0:half, 0:half], src=T00
        )
        nisa.dma_copy(
            dst=c32_inverse_scratch[half:C, half:C], src=T11
        )
        nisa.dma_copy(
            dst=c32_inverse_scratch[half:C, 0:half], src=X
        )
        c32_sbm.close_scope()

        # The block-factorized solve is algebraically correct but its changed
        # association loses material bits on the captured rank-2 state. Build
        # the inverse in the CPU oracle's ordered forward-substitution form.
        c32_sbm.open_scope(name="c32_forward_inverse")
        with nl.no_reorder():
            inverse_c = _c32_tile(c32_sbm, C, C)
            _tri_inverse_forward_from_hbm_to(
                c32_inverse_scratch, eye, inverse_c, c32_row_scratch,
                C, c32_sbm,
            )
            nisa.dma_copy(dst=c32_inverse_scratch, src=inverse_c)
        c32_sbm.close_scope()
    else:
        v_beta = nl.ndarray((C, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=v_beta, data=v_c, op0=nl.multiply, operand0=b_c)
        k_beta = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=k_beta, data=k_c, op0=nl.multiply, operand0=b_c)
        kb_T = _T(k_beta, C, K_DIM)              # [K,C]
        kc_T = _T(k_c, C, K_DIM)                 # [K,C]
        kk = _mm(kb_T, kc_T, C, C)               # k_beta@k_c.T [C,C]
        A_str = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=A_str, data1=kk, data2=dm, op=nl.multiply)
        nisa.tensor_scalar(dst=A_str, data=A_str, op0=nl.multiply, operand0=-1.0)
        nisa.tensor_tensor(dst=A_str, data1=A_str, data2=m_strict, op=nl.multiply)
        A = _tri_inverse_doubling(A_str, eye, C, C)
        A_T = _T(A, C, C)

    if C == 32 and STABLE_C32:
        m_incl_full = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m_incl_full, src=m_incl[0:C, 0:C])
        m_incl = m_incl_full
        g_c = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=g_c, src=g[base:base + C, 0:1])
        mi_T = _T(m_incl, C, C)
        g_cum = _mm(mi_T, g_c, C, 1)
        expg = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=expg, op=nl.exp, data=g_cum, bias=zC1, scale=1.0)

        # The full C32 RHS tile exists only after the inverse scope has
        # closed. Both RHS applications reload the HBM-staged inverse and
        # release all Cx128 SBUF storage before the next phase.
        c32_sbm.open_scope(name="c32_value_rhs")
        with nl.no_reorder():
            inverse_c = _c32_tile(c32_sbm, C, C)
            inverse_t = _c32_tile(c32_sbm, C, C)
            beta_rhs = _c32_tile(c32_sbm, C, 1)
            nisa.dma_copy(dst=inverse_c, src=c32_inverse_scratch)
            _T_to(inverse_c, C, C, inverse_t)
            nisa.dma_copy(
                dst=beta_rhs, src=beta[base:base + C, 0:1]
            )
            for v_base in range(0, V_DIM, 64):
                c32_sbm.open_scope(name="c32_value_rhs_tile")
                value_rhs = _c32_tile(c32_sbm, C, 64)
                v_corr = _c32_tile(c32_sbm, C, 64)
                nisa.dma_copy(
                    dst=value_rhs,
                    src=value[base:base + C, v_base:v_base + 64],
                )
                nisa.tensor_scalar(
                    dst=value_rhs, data=value_rhs, op0=nl.multiply,
                    operand0=beta_rhs,
                )
                _mm_to(inverse_t, value_rhs, v_corr, C, 64)
                nisa.dma_copy(
                    dst=c32_vcorr_scratch[0:C, v_base:v_base + 64],
                    src=v_corr,
                )
                c32_sbm.close_scope()
        c32_sbm.close_scope()

        nisa.dma_copy(dst=c32_state_scratch, src=s)
        c32_sbm.open_scope(name="c32_key_rhs")
        with nl.no_reorder():
            inverse_c = _c32_tile(c32_sbm, C, C)
            inverse_t = _c32_tile(c32_sbm, C, C)
            beta_rhs = _c32_tile(c32_sbm, C, 1)
            nisa.dma_copy(dst=inverse_c, src=c32_inverse_scratch)
            _T_to(inverse_c, C, C, inverse_t)
            nisa.dma_copy(
                dst=beta_rhs, src=beta[base:base + C, 0:1]
            )
            for v_base in range(0, V_DIM, 64):
                c32_sbm.open_scope(name="c32_key_rhs_value_tile")
                vprime = _c32_tile(c32_sbm, C, 64)
                nisa.memset(dst=vprime, value=0.0)
                for k_base in range(0, K_DIM, 64):
                    c32_sbm.open_scope(name="c32_key_rhs_key_tile")
                    k_rhs = _c32_tile(c32_sbm, C, 64)
                    k_cumdecay = _c32_tile(c32_sbm, C, 64)
                    k_cumdecay_t = _c32_tile(c32_sbm, 64, C)
                    state_tile = _c32_tile(c32_sbm, 64, 64)
                    vprime_part = _c32_tile(c32_sbm, C, 64)
                    nisa.dma_copy(
                        dst=k_rhs,
                        src=c32_key_scratch[0:C, k_base:k_base + 64],
                    )
                    nisa.dma_copy(
                        dst=state_tile,
                        src=c32_state_scratch[
                            k_base:k_base + 64, v_base:v_base + 64
                        ],
                    )
                    nisa.tensor_scalar(
                        dst=k_rhs, data=k_rhs, op0=nl.multiply,
                        operand0=beta_rhs,
                    )
                    nisa.tensor_scalar(
                        dst=k_rhs, data=k_rhs, op0=nl.multiply,
                        operand0=expg,
                    )
                    _mm_to(inverse_t, k_rhs, k_cumdecay, C, 64)
                    _T_to(k_cumdecay, C, 64, k_cumdecay_t)
                    _mm_to(k_cumdecay_t, state_tile, vprime_part, C, 64)
                    nisa.tensor_tensor(
                        dst=vprime, data1=vprime, data2=vprime_part,
                        op=nl.add,
                    )
                    c32_sbm.close_scope()
                nisa.dma_copy(
                    dst=c32_vprime_scratch[0:C, v_base:v_base + 64],
                    src=vprime,
                )
                c32_sbm.close_scope()
        c32_sbm.close_scope()
        # Rehydrate full normalized q/k tiles only after both C32 RHS scopes
        # have gone.
        k_c = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_c, src=c32_key_scratch)
        qS = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=qS, src=c32_q_scratch)
    else:
        # ---- v_corrected = A @ v_beta ; k_cumdecay = A @ (k_beta*exp(g_cum)) ----
        v_corr = _mm(A_T, v_beta, C, V_DIM)      # [C,V]
        expg = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=expg, op=nl.exp, data=g_cum, bias=zC1, scale=1.0)
        kbexp = nl.ndarray((C, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=kbexp, data=k_beta, op0=nl.multiply, operand0=expg)
        k_cumdecay = _mm(A_T, kbexp, C, K_DIM)   # [C,K]

    if C == 32 and STABLE_C32:
        # dm is not needed between the inverse and intra-block attention.
        # Rebuild it here so the first C32 decay tile can be released while
        # the block-factorized inverse is live.
        g_cum_row = _T(g_cum, C, 1)
        gA = _mm(g_cum_row, onesC, C, C)
        gB = _mm(onesC, g_cum_row, C, C)
        gdiff = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gdiff, data1=gA, data2=gB, op=nl.subtract)
        nisa.tensor_tensor(dst=gdiff, data1=gdiff, data2=m_incl, op=nl.multiply)
        dm = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=dm, op=nl.exp, data=gdiff, bias=zC1, scale=1.0)
        nisa.tensor_tensor(dst=dm, data1=dm, data2=m_incl, op=nl.multiply)

    # ---- intra = (qS @ k_c.T) * dm * m_incl ----
    if C == 32 and STABLE_C32:
        kc_T = _T(k_c, C, K_DIM)
    qS_T = _T(qS, C, K_DIM)                       # [K,C]
    qk = _mm(qS_T, kc_T, C, C)                    # qS@k_c.T [C,C]
    intra = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=intra, data1=qk, data2=dm, op=nl.multiply)
    nisa.tensor_tensor(dst=intra, data1=intra, data2=m_incl, op=nl.multiply)

    # ---- v_new = v_corr - k_cumdecay @ s ----
    if C == 32 and STABLE_C32:
        # Reload both RHS products only after their C32 scopes have closed.
        v_corr = nl.ndarray((C, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_corr, src=c32_vcorr_scratch)
        vprime = nl.ndarray((C, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=vprime, src=c32_vprime_scratch)
    else:
        kcd_T = _T(k_cumdecay, C, K_DIM)          # [K,C]
        vprime = _mm(kcd_T, s, C, V_DIM)          # k_cumdecay@s [C,V]
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
    if C == 32 and STABLE_C32:
        eye_full = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=eye_full, src=eye[0:C, 0:C])
        eye = eye_full
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
