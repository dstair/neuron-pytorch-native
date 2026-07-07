"""Flash causal-attention PREFILL kernel for GQA head_dim=256 (35B-A3B).

Purpose: replace the pure-torch `_gqa_prefill` full [S,S] causal attention (which
OOMs at S>~2k and is slow) with a flash-style, memory-flat-in-S kernel. Algorithm
ported from PR#60 nki_flash_attn_d256 (aws-neuron/neuronx-distributed-inference)
but rewritten in the bare-`nki` dst-style ISA API to match our other kernels
(gqa_tail_35b / deltanet_full_batched_35b) and drop into our torch_neuronx.nki_op
harness. Causal masking uses nisa.affine_select (validated in probe_affine_causal).

Prefill case is BS=1 with the 2 KV heads REPLICATED across TP cores -> exactly
1 KV head/core, so all Q_HEADS attend the single cached K/V. PR#60's bs*kv_heads
grid collapses to 1 program; we loop over Q_HEADS and query-tiles internally.

Inputs (per TP core, RoPE ALREADY applied, f32 or bf16):
  q : [Q_HEADS, S, HEAD_DIM]   query (post q-norm + rope)
  k : [S, HEAD_DIM]            key   (post k-norm + rope) — single KV head
  v : [S, HEAD_DIM]            value — single KV head
Returns:
  o : [Q_HEADS, S, HEAD_DIM]   attention output (pre output-gate, pre o_proj)

Math per (head h, query row i):  o_i = softmax_j( q_i·k_j / sqrt(d)  s.t. j<=i ) · v_j
Flash online-softmax over key-blocks: running max m, running denom l, running
output accumulator o_acc rescaled by exp(m_prev - m_cur) each block. Head_dim 256
tiled 2x128 for the QK contraction (like gqa_tail). No [S,S] tensor is materialized.

Sequence tiling: query rows by BP=128 (partition), key blocks by BF=512 (moving
free / psum_fmax). Whole key-blocks entirely in the future are skipped (causal);
the one straddling block gets a per-element affine_select causal mask.
"""
import math
import os as _os
import nki
import nki.isa as nisa
import nki.language as nl

HEAD_DIM = 256
Q_HEADS = int(_os.environ.get("GQA_Q_HEADS", "4"))
DT = 128                 # d-tile (HEAD_DIM = 2*DT)
BP = 128                 # query-row tile (partition)
BF = 512                 # key-block (moving free / psum fmax)
NEG_INF = -30000.0
RMS_EPS = 1e-6           # unused here (norm done in torch), kept for parity notes


def _copy_ps(src_ps, M, N):
    """PSUM [M,N] -> fresh SBUF [M,N] f32."""
    o = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src=src_ps)
    return o


@nki.jit
def nki_gqa_flash_prefill_chunk(q, k, v, q_base):
    """Chunked/bucketed flash prefill — ONE reusable NEFF for every prompt chunk.

    q     : [Q_HEADS, CHUNK, HEAD_DIM]  queries for this chunk (post norm+rope)
    k, v  : [KMAX, HEAD_DIM]            FULL fixed-size KV buffer (max_seq); rows
                                        [0 : q_base+CHUNK) are valid, rest are zeros.
    q_base: [1, 1] int32/f32            runtime scalar = global index of this chunk's
                                        first query row (0, CHUNK, 2*CHUNK, ...).
    Returns o [Q_HEADS, CHUNK, HEAD_DIM].

    Causality across chunks is handled ENTIRELY by the dynamic mask
    keep <=> (q_base + qi*BP + p) >= (k0 + f). Keys not yet written (index >
    q_base+CHUNK-1) are always in the future of every query row in this chunk, so
    the same mask also excludes the still-zero tail of the KMAX buffer — no separate
    validity check needed. Fixed CHUNK/KMAX shapes => single compile, reused N times.
    """
    H = q.shape[0]
    CHUNK = q.shape[1]
    KMAX = k.shape[0]
    QSCALE = 1.0 / math.sqrt(HEAD_DIM)
    n_q_tiles = (CHUNK + BP - 1) // BP
    n_kv_tiles = (KMAX + BF - 1) // BF

    out = nl.ndarray((H, CHUNK, HEAD_DIM), dtype=q.dtype, buffer=nl.shared_hbm)

    # broadcast the runtime q_base scalar to a [BP,1] column (device needs matching
    # partitions for the per-partition add); reused for every q-tile via +qi*BP.
    qb1 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=qb1, src=q_base[0:1, 0:1])
    onesBP = nl.ndarray((1, BP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=onesBP, value=1.0)
    qb_ps = nl.ndarray((BP, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qb_ps, stationary=onesBP, moving=qb1)
    qb_col = nl.ndarray((BP, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qb_col, src=qb_ps)   # [BP,1] all = q_base

    # Pre-transpose the full K buffer to d-partition tiles [DT, KMAX] x2 (shared).
    kT0 = nl.ndarray((DT, KMAX), dtype=nl.float32, buffer=nl.sbuf)
    kT1 = nl.ndarray((DT, KMAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_transpose(dst=kT0, src=k[0:KMAX, 0:DT])
    nisa.dma_transpose(dst=kT1, src=k[0:KMAX, DT:HEAD_DIM])

    for h in nl.affine_range(H):
        for qi in nl.sequential_range(n_q_tiles):
            q0 = qi * BP                       # LOCAL row offset within the chunk
            qsz = min(BP, CHUNK - q0)

            q_in = nl.ndarray((BP, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_in[0:qsz, 0:HEAD_DIM], src=q[h, q0:q0 + qsz, 0:HEAD_DIM])
            qT0 = nl.ndarray((DT, BP), dtype=nl.float32, buffer=nl.sbuf)
            qT1 = nl.ndarray((DT, BP), dtype=nl.float32, buffer=nl.sbuf)
            pqt0 = nl.ndarray((DT, BP), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=pqt0[0:DT, 0:qsz], data=q_in[0:qsz, 0:DT])
            nisa.tensor_copy(dst=qT0[0:DT, 0:qsz], src=pqt0[0:DT, 0:qsz])
            pqt1 = nl.ndarray((DT, BP), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=pqt1[0:DT, 0:qsz], data=q_in[0:qsz, DT:HEAD_DIM])
            nisa.tensor_copy(dst=qT1[0:DT, 0:qsz], src=pqt1[0:DT, 0:qsz])

            # global start of this q-tile = q_base + q0  -> per-partition column
            qglob = nl.ndarray((BP, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=qglob[0:qsz, 0:1], data=qb_col[0:qsz, 0:1],
                               op0=nl.add, operand0=float(q0))

            o_acc = nl.zeros((BP, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
            m_acc = nl.full((BP, 1), fill_value=NEG_INF, dtype=nl.float32, buffer=nl.sbuf)
            l_acc = nl.zeros((BP, 1), dtype=nl.float32, buffer=nl.sbuf)

            for kvi in nl.sequential_range(n_kv_tiles):
                k0 = kvi * BF
                ksz = min(BF, KMAX - k0)

                sc_ps = nl.ndarray((BP, BF), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=sc_ps[0:qsz, 0:ksz],
                               stationary=qT0[0:DT, 0:qsz], moving=kT0[0:DT, k0:k0 + ksz])
                sc0 = _copy_ps(sc_ps[0:qsz, 0:ksz], qsz, ksz)
                sc_ps2 = nl.ndarray((BP, BF), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=sc_ps2[0:qsz, 0:ksz],
                               stationary=qT1[0:DT, 0:qsz], moving=kT1[0:DT, k0:k0 + ksz])
                scores = nl.ndarray((BP, BF), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=scores[0:qsz, 0:ksz], data1=sc0[0:qsz, 0:ksz],
                                   data2=sc_ps2[0:qsz, 0:ksz], op=nl.add)
                nisa.tensor_scalar(dst=scores[0:qsz, 0:ksz], data=scores[0:qsz, 0:ksz],
                                   op0=nl.multiply, operand0=QSCALE)

                # ---- DYNAMIC causal mask: keep where (q_base+q0+p) >= (k0+f) ----
                # D[p,f] = p - f - k0 (static via iota); Dq = D + (q_base+q0);
                # bias = (Dq < 0) * NEG_INF (masks future keys AND unwritten tail).
                Dstat = nl.ndarray((BP, BF), dtype=nl.float32, buffer=nl.sbuf)
                nisa.iota(dst=Dstat[0:qsz, 0:ksz], pattern=[[-1, ksz]],
                          channel_multiplier=1, offset=-k0)
                Dq = nl.ndarray((BP, BF), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=Dq[0:qsz, 0:ksz], data=Dstat[0:qsz, 0:ksz],
                                   op0=nl.add, operand0=qglob[0:qsz, 0:1])
                mflag = nl.ndarray((BP, BF), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=mflag[0:qsz, 0:ksz], data=Dq[0:qsz, 0:ksz],
                                   op0=nl.less, operand0=0.0)
                nisa.tensor_scalar(dst=mflag[0:qsz, 0:ksz], data=mflag[0:qsz, 0:ksz],
                                   op0=nl.multiply, operand0=NEG_INF)
                nisa.tensor_tensor(dst=scores[0:qsz, 0:ksz], data1=scores[0:qsz, 0:ksz],
                                   data2=mflag[0:qsz, 0:ksz], op=nl.add)

                # ---- online softmax update ----
                blk_max = nl.ndarray((BP, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=blk_max[0:qsz, 0:1], data=scores[0:qsz, 0:ksz],
                                   op=nl.maximum, axis=1)
                m_prev = nl.ndarray((BP, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=m_prev[0:qsz, 0:1], src=m_acc[0:qsz, 0:1])
                nisa.tensor_tensor(dst=m_acc[0:qsz, 0:1], data1=m_prev[0:qsz, 0:1],
                                   data2=blk_max[0:qsz, 0:1], op=nl.maximum)
                neg_mcur = nl.ndarray((BP, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=neg_mcur[0:qsz, 0:1], data=m_acc[0:qsz, 0:1],
                                   op0=nl.multiply, operand0=-1.0)
                alpha = nl.ndarray((BP, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=alpha[0:qsz, 0:1], op=nl.exp, data=m_prev[0:qsz, 0:1],
                                bias=neg_mcur[0:qsz, 0:1], scale=1.0)
                p = nl.ndarray((BP, BF), dtype=nl.float32, buffer=nl.sbuf)
                p_sum = nl.ndarray((BP, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=p[0:qsz, 0:ksz], op=nl.exp, data=scores[0:qsz, 0:ksz],
                                bias=neg_mcur[0:qsz, 0:1], scale=1.0,
                                reduce_op=nl.add, reduce_res=p_sum[0:qsz, 0:1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
                nisa.tensor_tensor(dst=l_acc[0:qsz, 0:1], data1=l_acc[0:qsz, 0:1],
                                   data2=alpha[0:qsz, 0:1], op=nl.multiply)
                nisa.tensor_tensor(dst=l_acc[0:qsz, 0:1], data1=l_acc[0:qsz, 0:1],
                                   data2=p_sum[0:qsz, 0:1], op=nl.add)
                nisa.tensor_scalar(dst=o_acc[0:qsz, 0:HEAD_DIM], data=o_acc[0:qsz, 0:HEAD_DIM],
                                   op0=nl.multiply, operand0=alpha[0:qsz, 0:1])
                n_ksub = (ksz + DT - 1) // DT
                for kt in nl.affine_range(n_ksub):
                    ks = kt * DT
                    kssz = min(DT, ksz - ks)
                    pT = nl.ndarray((DT, BP), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(dst=pT[0:kssz, 0:qsz], data=p[0:qsz, ks:ks + kssz])
                    pT_s = nl.ndarray((DT, BP), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=pT_s[0:kssz, 0:qsz], src=pT[0:kssz, 0:qsz])
                    v_blk = nl.ndarray((DT, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(dst=v_blk[0:kssz, 0:HEAD_DIM],
                                  src=v[k0 + ks:k0 + ks + kssz, 0:HEAD_DIM])
                    pv_ps = nl.ndarray((BP, HEAD_DIM), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=pv_ps[0:qsz, 0:HEAD_DIM],
                                   stationary=pT_s[0:kssz, 0:qsz], moving=v_blk[0:kssz, 0:HEAD_DIM])
                    nisa.tensor_tensor(dst=o_acc[0:qsz, 0:HEAD_DIM], data1=o_acc[0:qsz, 0:HEAD_DIM],
                                       data2=pv_ps[0:qsz, 0:HEAD_DIM], op=nl.add)

            rinv = nl.ndarray((BP, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rinv[0:qsz, 0:1], data=l_acc[0:qsz, 0:1],
                               op0=nl.add, operand0=1e-20)
            nisa.activation(dst=rinv[0:qsz, 0:1], op=nl.reciprocal, data=rinv[0:qsz, 0:1], scale=1.0)
            o_fin = nl.ndarray((BP, HEAD_DIM), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=o_fin[0:qsz, 0:HEAD_DIM], data=o_acc[0:qsz, 0:HEAD_DIM],
                               op0=nl.multiply, operand0=rinv[0:qsz, 0:1])
            nisa.dma_copy(dst=out[h, q0:q0 + qsz, 0:HEAD_DIM], src=o_fin[0:qsz, 0:HEAD_DIM])

    return out


@nki.jit
def nki_gqa_flash_prefill(q, k, v):
    H = q.shape[0]                 # Q_HEADS
    S = q.shape[1]
    QSCALE = 1.0 / math.sqrt(HEAD_DIM)
    n_q_tiles = (S + BP - 1) // BP
    n_kv_tiles = (S + BF - 1) // BF

    out = nl.ndarray((H, S, HEAD_DIM), dtype=q.dtype, buffer=nl.shared_hbm)

    # ---- Pre-transpose K to d-partition tiles [DT, S] x2 (shared by all heads) ----
    # kT0/kT1[d, j] hold the two 128-halves of head_dim on the partition axis so the
    # QK matmul is stationary=qT[DT,rows], moving=kT[DT,keys] -> [rows,keys].
    kT0 = nl.ndarray((DT, S), dtype=nl.float32, buffer=nl.sbuf)
    kT1 = nl.ndarray((DT, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_transpose(dst=kT0, src=k[0:S, 0:DT])
    nisa.dma_transpose(dst=kT1, src=k[0:S, DT:HEAD_DIM])

    for h in nl.affine_range(H):
        for qi in nl.sequential_range(n_q_tiles):
            q0 = qi * BP
            qsz = min(BP, S - q0)

            # ---- load q tile [qsz, 256], transpose to d-partition [DT, qsz] x2 ----
            q_in = nl.ndarray((BP, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_in[0:qsz, 0:HEAD_DIM], src=q[h, q0:q0 + qsz, 0:HEAD_DIM])
            qT0 = nl.ndarray((DT, BP), dtype=nl.float32, buffer=nl.sbuf)
            qT1 = nl.ndarray((DT, BP), dtype=nl.float32, buffer=nl.sbuf)
            pqt0 = nl.ndarray((DT, BP), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=pqt0[0:DT, 0:qsz], data=q_in[0:qsz, 0:DT])
            nisa.tensor_copy(dst=qT0[0:DT, 0:qsz], src=pqt0[0:DT, 0:qsz])
            pqt1 = nl.ndarray((DT, BP), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=pqt1[0:DT, 0:qsz], data=q_in[0:qsz, DT:HEAD_DIM])
            nisa.tensor_copy(dst=qT1[0:DT, 0:qsz], src=pqt1[0:DT, 0:qsz])

            # ---- flash accumulators over query rows [qsz] ----
            o_acc = nl.zeros((BP, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
            m_acc = nl.full((BP, 1), fill_value=NEG_INF, dtype=nl.float32, buffer=nl.sbuf)
            l_acc = nl.zeros((BP, 1), dtype=nl.float32, buffer=nl.sbuf)

            for kvi in nl.sequential_range(n_kv_tiles):
                k0 = kvi * BF
                ksz = min(BF, S - k0)
                # causal: whole key-block strictly after the last query row -> skip
                if k0 > (q0 + qsz - 1):
                    continue

                # ---- scores[qsz, ksz] = (qT0.T@kT0 + qT1.T@kT1) * QSCALE ----
                sc_ps = nl.ndarray((BP, BF), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=sc_ps[0:qsz, 0:ksz],
                               stationary=qT0[0:DT, 0:qsz], moving=kT0[0:DT, k0:k0 + ksz])
                sc0 = _copy_ps(sc_ps[0:qsz, 0:ksz], qsz, ksz)
                sc_ps2 = nl.ndarray((BP, BF), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=sc_ps2[0:qsz, 0:ksz],
                               stationary=qT1[0:DT, 0:qsz], moving=kT1[0:DT, k0:k0 + ksz])
                scores = nl.ndarray((BP, BF), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=scores[0:qsz, 0:ksz], data1=sc0[0:qsz, 0:ksz],
                                   data2=sc_ps2[0:qsz, 0:ksz], op=nl.add)
                nisa.tensor_scalar(dst=scores[0:qsz, 0:ksz], data=scores[0:qsz, 0:ksz],
                                   op0=nl.multiply, operand0=QSCALE)

                # ---- causal mask (only needed on the straddling block) ----
                # keep where (q0 + p) >= (k0 + f)  <=>  1*p + (-1)*f + (q0-k0) >= 0
                if (k0 + ksz - 1) > q0:      # block contains future keys for some rows
                    masked = nl.ndarray((BP, BF), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.affine_select(dst=masked[0:qsz, 0:ksz], pattern=[[-1, ksz]],
                                       channel_multiplier=1, on_true_tile=scores[0:qsz, 0:ksz],
                                       on_false_value=NEG_INF, cmp_op=nl.greater_equal,
                                       offset=int(q0 - k0))
                    scores = masked

                # ---- online softmax update ----
                blk_max = nl.ndarray((BP, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=blk_max[0:qsz, 0:1], data=scores[0:qsz, 0:ksz],
                                   op=nl.maximum, axis=1)
                m_prev = nl.ndarray((BP, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=m_prev[0:qsz, 0:1], src=m_acc[0:qsz, 0:1])
                nisa.tensor_tensor(dst=m_acc[0:qsz, 0:1], data1=m_prev[0:qsz, 0:1],
                                   data2=blk_max[0:qsz, 0:1], op=nl.maximum)   # m_cur

                # alpha = exp(m_prev - m_cur); neg m_cur bias, scale on m_prev
                neg_mcur = nl.ndarray((BP, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=neg_mcur[0:qsz, 0:1], data=m_acc[0:qsz, 0:1],
                                   op0=nl.multiply, operand0=-1.0)
                alpha = nl.ndarray((BP, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=alpha[0:qsz, 0:1], op=nl.exp, data=m_prev[0:qsz, 0:1],
                                bias=neg_mcur[0:qsz, 0:1], scale=1.0)

                # p = exp(scores - m_cur), row-sum -> p_sum
                p = nl.ndarray((BP, BF), dtype=nl.float32, buffer=nl.sbuf)
                p_sum = nl.ndarray((BP, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=p[0:qsz, 0:ksz], op=nl.exp, data=scores[0:qsz, 0:ksz],
                                bias=neg_mcur[0:qsz, 0:1], scale=1.0,
                                reduce_op=nl.add, reduce_res=p_sum[0:qsz, 0:1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)

                # l_acc = l_acc*alpha + p_sum
                nisa.tensor_tensor(dst=l_acc[0:qsz, 0:1], data1=l_acc[0:qsz, 0:1],
                                   data2=alpha[0:qsz, 0:1], op=nl.multiply)
                nisa.tensor_tensor(dst=l_acc[0:qsz, 0:1], data1=l_acc[0:qsz, 0:1],
                                   data2=p_sum[0:qsz, 0:1], op=nl.add)

                # o_acc = o_acc*alpha + p @ v_block
                nisa.tensor_scalar(dst=o_acc[0:qsz, 0:HEAD_DIM], data=o_acc[0:qsz, 0:HEAD_DIM],
                                   op0=nl.multiply, operand0=alpha[0:qsz, 0:1])
                # p@v: need p as [key, row] stationary -> transpose p in 128-tiles;
                # v_block [key, 256] moving. Accumulate over key sub-tiles of 128.
                n_ksub = (ksz + DT - 1) // DT
                for kt in nl.affine_range(n_ksub):
                    ks = kt * DT
                    kssz = min(DT, ksz - ks)
                    pT = nl.ndarray((DT, BP), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(dst=pT[0:kssz, 0:qsz], data=p[0:qsz, ks:ks + kssz])
                    pT_s = nl.ndarray((DT, BP), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=pT_s[0:kssz, 0:qsz], src=pT[0:kssz, 0:qsz])
                    v_blk = nl.ndarray((DT, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(dst=v_blk[0:kssz, 0:HEAD_DIM],
                                  src=v[k0 + ks:k0 + ks + kssz, 0:HEAD_DIM])
                    pv_ps = nl.ndarray((BP, HEAD_DIM), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=pv_ps[0:qsz, 0:HEAD_DIM],
                                   stationary=pT_s[0:kssz, 0:qsz], moving=v_blk[0:kssz, 0:HEAD_DIM])
                    nisa.tensor_tensor(dst=o_acc[0:qsz, 0:HEAD_DIM], data1=o_acc[0:qsz, 0:HEAD_DIM],
                                       data2=pv_ps[0:qsz, 0:HEAD_DIM], op=nl.add)

            # ---- normalize by running denom and store ----
            rinv = nl.ndarray((BP, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=rinv[0:qsz, 0:1], data=l_acc[0:qsz, 0:1],
                               op0=nl.add, operand0=1e-20)
            nisa.activation(dst=rinv[0:qsz, 0:1], op=nl.reciprocal, data=rinv[0:qsz, 0:1], scale=1.0)
            o_fin = nl.ndarray((BP, HEAD_DIM), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=o_fin[0:qsz, 0:HEAD_DIM], data=o_acc[0:qsz, 0:HEAD_DIM],
                               op0=nl.multiply, operand0=rinv[0:qsz, 0:1])
            nisa.dma_copy(dst=out[h, q0:q0 + qsz, 0:HEAD_DIM], src=o_fin[0:qsz, 0:HEAD_DIM])

    return out
