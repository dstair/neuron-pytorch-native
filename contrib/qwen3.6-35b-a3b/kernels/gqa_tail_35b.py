"""GQA-tail mega-kernel (Phase 1, BS>1 throughput). ONE @nki.jit per GQA layer that
collapses the attention-tail glue cluster — per-head q RMSNorm + partial-64 RoPE +
scaled scores + masked softmax + weighted-V + sigmoid output-gate — into a single
custom call, removing the ~12 inter-op EVENT_SEMAPHORE barriers/layer that the BS=8
profile (decv4bs8) showed dominate the critical path.

The default variant receives an already-updated cache. The stateful variant
receives the full flattened BF16 cache buffers plus a static layer index,
performs attention over the prior cache plus current rows, then writes only
those current rows through aliased inputs. o_proj stays F.linear.

Math mirrors gqa_tail_ref.gqa_tail_ref (validated vs _gqa_layer to 3.9e-7). Per b:
  qn = (1+q_norm) * rms_norm(query[b])             [6,256]  free-axis
  qr = partial_rope(qn, cos, sin)                  [6,256]  rotary on [:64]
  scores = (qr @ cached_k[b].T) / sqrt(256)        [6,S]
  p = softmax_masked(scores, mask)                 [6,S]    (exp*mask / sum)
  o = p @ cached_v[b]                              [6,256]
  attn_out[b] = o * sigmoid(gate[b])               [6,256] -> row b*6+h

Layout (per TP core): Q_HEADS=6 on partition; HEAD_DIM=256 tiled 2x128 for the
contraction; seq S tiled by 128 (K) / 512 (matmul moving free). All head vectors
that must index the partition dim stay <=128 by the 2x128 d-split.
"""
import math
import os as _os
import nki
import nki.isa as nisa
import nki.language as nl

# 35B-A3B per-core (TP=4): Q_HEADS=4 (16/4). KV heads (2) are REPLICATED across
# cores → 1 KV head/core, so all Q_HEADS attend to the single cached_k[b] — exactly
# this kernel's assumption. Only Q_HEADS differs from the 27B (6→4); HEAD_DIM 256 /
# ROPE_DIM 64 are identical. Q_HEADS env-overridable for other TP degrees.
HEAD_DIM = 256
Q_HEADS = int(_os.environ.get("GQA_Q_HEADS", "4"))
ROPE_DIM = 64
HALF_ROPE = 32          # ROPE_DIM // 2
RMS_EPS = 1e-6
DT = 128                # d-tile (HEAD_DIM = 2*DT)
NMAX = 512              # matmul moving free max (psum_fmax)


def _mm(stat, mov, M, N):
    """dst[M,N] = stat[K,M].T @ mov[K,N]."""
    p = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=p, stationary=stat, moving=mov)
    o = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src=p)
    return o


def _nki_gqa_tail_impl(
    query,      # [B*Q_HEADS, HEAD_DIM] f32  (projected, pre-norm)
    gate,       # [B*Q_HEADS, HEAD_DIM] f32  (raw)
    q_norm,     # [1, HEAD_DIM] f32
    cos,        # [1, ROPE_DIM] f32
    sin,        # [1, ROPE_DIM] f32
    cached_k,   # [G*B*S, HEAD_DIM] bf16/f32
    cached_v,   # [G*B*S, HEAD_DIM] bf16/f32
    mask,       # [1, S] f32  (1.0 valid / 0.0 masked)
    key=None,   # optional [B, HEAD_DIM] bf16
    value=None, # optional [B, HEAD_DIM] bf16
    position=None,  # optional [1, 1] int32
    layer_index=0,
    cache_bf16=False,
):
    B = query.shape[0] // Q_HEADS
    S = mask.shape[1]
    QSCALE = 1.0 / math.sqrt(HEAD_DIM)
    num_n = (S + NMAX - 1) // NMAX     # score N-blocks
    num_k = (S + DT - 1) // DT         # weighted-V K-tiles (128)

    out = nl.ndarray((B * Q_HEADS, HEAD_DIM), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- shared constants resident in SBUF ----
    z6 = nl.ndarray((Q_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=z6, value=0.0)
    # DEVICE CONSTRAINT (not caught by sim): tensor_tensor requires MATCHING partition
    # counts — a [1,D] row is NOT auto-broadcast across the 6 head-partitions. So we
    # row-broadcast every [1,D] constant to [Q_HEADS,D] ONCE via the ones-column matmul
    # idiom (_mm(ones6[1,6], row[1,D]) = [6,D]), then all per-b ops use 6-partition operands.
    ones6 = nl.ndarray((1, Q_HEADS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones6, value=1.0)
    qn_w1 = nl.ndarray((1, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)   # (1+q_norm) row
    nisa.dma_copy(dst=qn_w1, src=q_norm[0:1, 0:HEAD_DIM])
    nisa.tensor_scalar(dst=qn_w1, data=qn_w1, op0=nl.add, operand0=1.0)
    qn_w = _mm(ones6, qn_w1, Q_HEADS, HEAD_DIM)                           # [6,256]
    cos1 = nl.ndarray((1, ROPE_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos1, src=cos[0:1, 0:ROPE_DIM])
    cos_r = _mm(ones6, cos1, Q_HEADS, ROPE_DIM)                           # [6,64]
    sin1 = nl.ndarray((1, ROPE_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sin1, src=sin[0:1, 0:ROPE_DIM])
    sin_r = _mm(ones6, sin1, Q_HEADS, ROPE_DIM)                           # [6,64]
    # additive mask bias row: (mask-1)*1e9  (0 where valid, -1e9 where masked)
    neg_row = nl.ndarray((1, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=neg_row, src=mask[0:1, 0:S])
    nisa.tensor_scalar(dst=neg_row, data=neg_row, op0=nl.subtract, operand0=1.0)   # mask-1
    nisa.tensor_scalar(dst=neg_row, data=neg_row, op0=nl.multiply, operand0=1e9)    # *(1e9) -> 0 / -1e9
    if position is not None:
        pos = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pos, src=position)
    # NOTE: do NOT broadcast to [6,S] in one matmul — at max_seq=2048 the moving free
    # dim exceeds the 512 matmul limit. Broadcast per-N-block (<=512) inside the loop.

    for b in nl.sequential_range(B):
        qrow = b * Q_HEADS
        krow = (layer_index * B + b) * S

        # ---- load q[6,256], RMSNorm over free axis, *(1+w) ----
        q_in = nl.ndarray((Q_HEADS, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_in, src=query[qrow:qrow + Q_HEADS, 0:HEAD_DIM])
        ss = nl.ndarray((Q_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        sq = nl.ndarray((Q_HEADS, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=q_in, bias=z6, scale=1.0,
                        reduce_op=nl.add, reduce_res=ss, reduce_cmd=nisa.reduce_cmd.reset_reduce)
        rinv = nl.ndarray((Q_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rinv, op=nl.rsqrt, data=ss, bias=z6, scale=1.0 / HEAD_DIM)
        qn = nl.ndarray((Q_HEADS, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=qn, data=q_in, op0=nl.multiply, operand0=rinv)   # per-partition scale
        nisa.tensor_tensor(dst=qn, data1=qn, data2=qn_w, op=nl.multiply)        # *(1+w) (free broadcast [1,256])

        # ---- partial RoPE on first ROPE_DIM cols (rotate_half within [:64]) ----
        # qr[:, :32]  = qn[:, :32]*cos[:32] - qn[:, 32:64]*sin[:32]
        # qr[:, 32:64]= qn[:, 32:64]*cos[32:64] + qn[:, :32]*sin[32:64]
        # qr[:, 64:]  = qn[:, 64:]   (pass-through)
        qr = nl.ndarray((Q_HEADS, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qr[0:Q_HEADS, ROPE_DIM:HEAD_DIM], src=qn[0:Q_HEADS, ROPE_DIM:HEAD_DIM])
        # lo half
        t_a = nl.ndarray((Q_HEADS, HALF_ROPE), dtype=nl.float32, buffer=nl.sbuf)
        t_b = nl.ndarray((Q_HEADS, HALF_ROPE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t_a, data1=qn[0:Q_HEADS, 0:HALF_ROPE], data2=cos_r[0:Q_HEADS, 0:HALF_ROPE], op=nl.multiply)
        nisa.tensor_tensor(dst=t_b, data1=qn[0:Q_HEADS, HALF_ROPE:ROPE_DIM], data2=sin_r[0:Q_HEADS, 0:HALF_ROPE], op=nl.multiply)
        nisa.tensor_tensor(dst=qr[0:Q_HEADS, 0:HALF_ROPE], data1=t_a, data2=t_b, op=nl.subtract)
        # hi half
        t_c = nl.ndarray((Q_HEADS, HALF_ROPE), dtype=nl.float32, buffer=nl.sbuf)
        t_d = nl.ndarray((Q_HEADS, HALF_ROPE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t_c, data1=qn[0:Q_HEADS, HALF_ROPE:ROPE_DIM], data2=cos_r[0:Q_HEADS, HALF_ROPE:ROPE_DIM], op=nl.multiply)
        nisa.tensor_tensor(dst=t_d, data1=qn[0:Q_HEADS, 0:HALF_ROPE], data2=sin_r[0:Q_HEADS, HALF_ROPE:ROPE_DIM], op=nl.multiply)
        nisa.tensor_tensor(dst=qr[0:Q_HEADS, HALF_ROPE:ROPE_DIM], data1=t_c, data2=t_d, op=nl.add)

        # ---- qr -> d-partition tiles [DT,6] x2 (transpose [6,128]->[128,6]) ----
        qrT0 = nl.ndarray((DT, Q_HEADS), dtype=nl.float32, buffer=nl.sbuf)
        qrT1 = nl.ndarray((DT, Q_HEADS), dtype=nl.float32, buffer=nl.sbuf)
        p0 = nl.ndarray((DT, Q_HEADS), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=p0, data=qr[0:Q_HEADS, 0:DT])
        nisa.tensor_copy(dst=qrT0, src=p0)
        p1 = nl.ndarray((DT, Q_HEADS), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=p1, data=qr[0:Q_HEADS, DT:HEAD_DIM])
        nisa.tensor_copy(dst=qrT1, src=p1)

        if position is not None:
            current_key_b = nl.ndarray(
                (1, HEAD_DIM), dtype=key.dtype, buffer=nl.sbuf
            )
            current_value_b = nl.ndarray(
                (1, HEAD_DIM), dtype=value.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=current_key_b, src=key[b:b + 1, 0:HEAD_DIM]
            )
            nisa.dma_copy(
                dst=current_value_b, src=value[b:b + 1, 0:HEAD_DIM]
            )
            current_key = nl.ndarray(
                (1, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf
            )
            current_value = nl.ndarray(
                (1, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=current_key, src=current_key_b)
            nisa.tensor_copy(dst=current_value, src=current_value_b)

            current_kT0 = nl.ndarray(
                (DT, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            current_kT1 = nl.ndarray(
                (DT, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            current_kp0 = nl.ndarray(
                (DT, 1), dtype=nl.float32, buffer=nl.psum
            )
            current_kp1 = nl.ndarray(
                (DT, 1), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(
                dst=current_kp0, data=current_key[0:1, 0:DT]
            )
            nisa.nc_transpose(
                dst=current_kp1, data=current_key[0:1, DT:HEAD_DIM]
            )
            nisa.tensor_copy(dst=current_kT0, src=current_kp0)
            nisa.tensor_copy(dst=current_kT1, src=current_kp1)
            current_score0 = _mm(
                qrT0, current_kT0, Q_HEADS, 1
            )
            current_score1 = _mm(
                qrT1, current_kT1, Q_HEADS, 1
            )
            current_score = nl.ndarray(
                (Q_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=current_score,
                data1=current_score0,
                data2=current_score1,
                op=nl.add,
            )
            nisa.tensor_scalar(
                dst=current_score,
                data=current_score,
                op0=nl.multiply,
                operand0=QSCALE,
            )

        # ---- cached_k[b] -> d-partition [DT,S] x2 via dma_transpose ----
        ckT0 = nl.ndarray((DT, S), dtype=nl.float32, buffer=nl.sbuf)
        ckT1 = nl.ndarray((DT, S), dtype=nl.float32, buffer=nl.sbuf)
        if cache_bf16:
            ckB0 = nl.ndarray((DT, S), dtype=cached_k.dtype, buffer=nl.sbuf)
            ckB1 = nl.ndarray((DT, S), dtype=cached_k.dtype, buffer=nl.sbuf)
            nisa.dma_transpose(dst=ckB0, src=cached_k[krow:krow + S, 0:DT])
            nisa.dma_transpose(dst=ckB1, src=cached_k[krow:krow + S, DT:HEAD_DIM])
            nisa.tensor_copy(dst=ckT0, src=ckB0)
            nisa.tensor_copy(dst=ckT1, src=ckB1)
        else:
            nisa.dma_transpose(dst=ckT0, src=cached_k[krow:krow + S, 0:DT])
            nisa.dma_transpose(dst=ckT1, src=cached_k[krow:krow + S, DT:HEAD_DIM])

        # ---- scores[6,S] = (qr @ ck.T) * QSCALE, in N-blocks of <=512 ----
        scores = nl.ndarray((Q_HEADS, S), dtype=nl.float32, buffer=nl.sbuf)
        for nt in nl.affine_range(num_n):
            ns = nt * NMAX
            nsz = min(NMAX, S - ns)
            acc0 = nl.ndarray((Q_HEADS, NMAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=acc0[0:Q_HEADS, 0:nsz], stationary=qrT0, moving=ckT0[0:DT, ns:ns + nsz])
            acc0s = nl.ndarray((Q_HEADS, NMAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=acc0s[0:Q_HEADS, 0:nsz], src=acc0[0:Q_HEADS, 0:nsz])
            acc1 = nl.ndarray((Q_HEADS, NMAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=acc1[0:Q_HEADS, 0:nsz], stationary=qrT1, moving=ckT1[0:DT, ns:ns + nsz])
            accsum = nl.ndarray((Q_HEADS, NMAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=accsum[0:Q_HEADS, 0:nsz], data1=acc0s[0:Q_HEADS, 0:nsz], data2=acc1[0:Q_HEADS, 0:nsz], op=nl.add)
            sc = nl.ndarray((Q_HEADS, NMAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=sc[0:Q_HEADS, 0:nsz], data=accsum[0:Q_HEADS, 0:nsz],
                               op0=nl.multiply, operand0=QSCALE)
            # add the causal mask bias for this block (broadcast neg_row slice to 6 parts)
            negb = _mm(ones6, neg_row[0:1, ns:ns + nsz], Q_HEADS, nsz)   # [6,nsz], <=512
            nisa.tensor_tensor(dst=scores[0:Q_HEADS, ns:ns + nsz], data1=sc[0:Q_HEADS, 0:nsz], data2=negb[0:Q_HEADS, 0:nsz], op=nl.add)

        # ---- masked softmax over free axis S ----
        smax = nl.ndarray((Q_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=smax, data=scores, op=nl.maximum, axis=1)
        if position is not None:
            softmax_max = nl.ndarray(
                (Q_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=softmax_max,
                data1=smax,
                data2=current_score,
                op=nl.maximum,
            )
        else:
            softmax_max = smax
        nsmax = nl.ndarray((Q_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=nsmax, data=softmax_max,
            op0=nl.multiply, operand0=-1.0,
        )
        p = nl.ndarray((Q_HEADS, S), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=p, op=nl.exp, data=scores, bias=nsmax, scale=1.0)   # exp(scores - max); masked cols already -1e9 -> exp~0
        psum_ = nl.ndarray((Q_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=psum_, data=p, op=nl.add, axis=1)
        if position is not None:
            current_p = nl.ndarray(
                (Q_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.activation(
                dst=current_p,
                op=nl.exp,
                data=current_score,
                bias=nsmax,
                scale=1.0,
            )
            psum_total = nl.ndarray(
                (Q_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=psum_total,
                data1=psum_,
                data2=current_p,
                op=nl.add,
            )
        else:
            psum_total = psum_
        rsum = nl.ndarray((Q_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=rsum, op=nl.reciprocal,
            data=psum_total, bias=z6, scale=1.0,
        )

        # ---- weighted-V o[6,256] = p @ cv ; accumulate over S in 128-tiles ----
        o = nl.ndarray((Q_HEADS, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=o, value=0.0)
        for kt in nl.affine_range(num_k):
            ks = kt * DT
            ksz = min(DT, S - ks)
            # p column tile [6, ksz] -> [ksz, 6] (seq on partition) for stationary
            pT = nl.ndarray((DT, Q_HEADS), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=pT[0:ksz, 0:Q_HEADS], data=p[0:Q_HEADS, ks:ks + ksz])
            pT_s = nl.ndarray((DT, Q_HEADS), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pT_s[0:ksz, 0:Q_HEADS], src=pT[0:ksz, 0:Q_HEADS])
            cv_t = nl.ndarray((DT, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
            if cache_bf16:
                cv_b = nl.ndarray((DT, HEAD_DIM), dtype=cached_v.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=cv_b[0:ksz, 0:HEAD_DIM], src=cached_v[krow + ks:krow + ks + ksz, 0:HEAD_DIM])
                nisa.tensor_copy(dst=cv_t[0:ksz, 0:HEAD_DIM], src=cv_b[0:ksz, 0:HEAD_DIM])
            else:
                nisa.dma_copy(dst=cv_t[0:ksz, 0:HEAD_DIM], src=cached_v[krow + ks:krow + ks + ksz, 0:HEAD_DIM])
            ovk = _mm(pT_s[0:ksz, 0:Q_HEADS], cv_t[0:ksz, 0:HEAD_DIM], Q_HEADS, HEAD_DIM)  # [6,256]
            nisa.tensor_tensor(dst=o, data1=o, data2=ovk, op=nl.add)
        if position is not None:
            current_pT = nl.ndarray(
                (1, Q_HEADS), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(dst=current_pT, data=current_p)
            current_pT_s = nl.ndarray(
                (1, Q_HEADS), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=current_pT_s, src=current_pT)
            current_o = _mm(
                current_pT_s, current_value, Q_HEADS, HEAD_DIM
            )
            nisa.tensor_tensor(
                dst=o, data1=o, data2=current_o, op=nl.add
            )
        nisa.tensor_scalar(dst=o, data=o, op0=nl.multiply, operand0=rsum)       # /sum (per-partition)

        # ---- output gate: o * sigmoid(gate[b]) ----
        g_in = nl.ndarray((Q_HEADS, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=g_in, src=gate[qrow:qrow + Q_HEADS, 0:HEAD_DIM])
        gs = nl.ndarray((Q_HEADS, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=gs, op=nl.sigmoid, data=g_in, bias=z6, scale=1.0)
        nisa.tensor_tensor(dst=o, data1=o, data2=gs, op=nl.multiply)
        nisa.dma_copy(dst=out[qrow:qrow + Q_HEADS, 0:HEAD_DIM], src=o)

        if position is not None:
            cache_row = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=cache_row, data=pos, op0=nl.add, operand0=krow,
            )
            nisa.dma_copy(
                dst=cached_k.ap(
                    [[HEAD_DIM, 1], [1, HEAD_DIM]], scalar_offset=cache_row
                ),
                src=key[b:b + 1, 0:HEAD_DIM],
            )
            nisa.dma_copy(
                dst=cached_v.ap(
                    [[HEAD_DIM, 1], [1, HEAD_DIM]], scalar_offset=cache_row
                ),
                src=value[b:b + 1, 0:HEAD_DIM],
            )

    return out


@nki.jit
def nki_gqa_tail(
    query, gate, q_norm, cos, sin, cached_k, cached_v, mask,
):
    return _nki_gqa_tail_impl(
        query, gate, q_norm, cos, sin, cached_k, cached_v, mask,
    )


@nki.jit
def nki_gqa_tail_stateful(
    query, gate, q_norm, cos, sin, cached_k, cached_v, mask,
    key, value, position, layer_index,
):
    out = _nki_gqa_tail_impl(
        query, gate, q_norm, cos, sin, cached_k, cached_v, mask,
        key, value, position, layer_index=layer_index, cache_bf16=True,
    )
    return out
