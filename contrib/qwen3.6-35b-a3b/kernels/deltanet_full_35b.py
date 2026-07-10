"""
DeltaNet "full" NKI kernel — fuses the entire DeltaNet inner block.

Replaces ~30 PyTorch ops in `_deltanet_layer()` with ONE @nki_op:
  1. Conv state update + depthwise conv + bias + SiLU (phase 1)
  2. L2 normalize q,k per-head + Q-scale (phase 2)
  3. Gates: g = -exp(A_log) * softplus(a + dt_bias);  beta = sigmoid(b)
  4. Recurrence: state *= exp(g); kv = state.T @ k; delta = (v - kv) * beta;
                 state += outer(k, delta); attn = state.T @ q
  5. RMSNormGated: gated = silu(z) * norm_w * rsqrt(mean(attn^2) + eps) * attn

Layout (TP=4, per-core):
  state:         [V_HEADS*K_DIM=1536, V_DIM=128] f32  in/out
  mixed_qkv:     [QKV_DIM=2560]      bf16
  conv_state:    [QKV_DIM, 3]        bf16  in/out
  conv_weight:   [QKV_DIM, 4]        f32
  conv_bias:     [QKV_DIM]           f32
  a_out, b_out, A_log, dt_bias: [V_HEADS=12] f32
  z:             [V_HEADS, V_DIM]    bf16
  norm_weight:   [V_DIM]             f32

Returns:
  new_state:        [V_HEADS*K_DIM, V_DIM]  f32
  new_conv_state:   [QKV_DIM, 3]            bf16
  output:           [V_HEADS, V_DIM]        bf16   gated, ready for out_proj

Key design decisions (vs broken `deltanet_fused_inner.py`):
  - qkv_act laid out [PMAX, NUM_TILES] (column slots), not [QKV_DIM, 1]
    — avoids SBUF partition limit. See feedback-nki-sbuf-partitions.
  - Outer affine over K_HEADS, inner affine over HEAD_GROUP — inner index
    `h = kh*HEAD_GROUP + ig` is affine, so compiler can vectorize/pipeline
    without per-head scalar pickoffs from a [V_HEADS, 1] gate buffer.
  - Per-head gate values loaded directly from HBM (`a_out[h:h+1]`,
    `b_out[h:h+1]`, ...) inside the inner loop — same pattern as the
    proven recurrent_v2 kernel. Avoids the [V_HEADS, 1] -> [1,1] pickoff
    that broke the original all-in-one fuse.
"""
import math
import nki
import nki.isa as nisa
import nki.language as nl


K_DIM = 128
V_DIM = 128
K_HEADS = 4
V_HEADS = 12
HEAD_GROUP = V_HEADS // K_HEADS  # 3
QKV_DIM = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM  # 2560
CONV_KERNEL = 4
RMS_EPS = 1e-6


@nki.jit
def nki_deltanet_full(
    state,         # [V_HEADS*K_DIM, V_DIM] f32
    mixed_qkv,     # [QKV_DIM] bf16
    conv_state,    # [QKV_DIM, 3] bf16
    conv_weight,   # [QKV_DIM, 4] f32
    conv_bias,     # [QKV_DIM] f32
    a_out,         # [V_HEADS] f32
    b_out,         # [V_HEADS] f32
    z,             # [V_HEADS, V_DIM] bf16
    A_log,         # [V_HEADS] f32
    dt_bias,       # [V_HEADS] f32
    norm_weight,   # [V_DIM] f32
):
    new_state = nl.ndarray((V_HEADS * K_DIM, V_DIM), dtype=state.dtype, buffer=nl.shared_hbm)
    new_conv_state = nl.ndarray((QKV_DIM, 3), dtype=conv_state.dtype, buffer=nl.shared_hbm)
    output = nl.ndarray((V_HEADS, V_DIM), dtype=z.dtype, buffer=nl.shared_hbm)
    # Opt 3 scratch: silu(z) precomputed in HBM so per-head DMA back into
    # SBUF works — same pattern recurrent_v2 uses for its g/beta scalars.
    silu_z_hbm = nl.ndarray((V_HEADS, V_DIM), dtype=nl.float32, buffer=nl.shared_hbm)
    # Opt 5 scratch: exp_g and beta precomputed in HBM. Per-head DMA back.
    exp_g_hbm = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    beta_hbm = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    PMAX = nl.tile_size.pmax  # 128
    NUM_QKV_TILES = QKV_DIM // PMAX  # 20
    Q_SCALE = 1.0 / math.sqrt(K_DIM)

    # =========================================================================
    # PHASE 1 — conv state update + depthwise conv + SiLU
    # qkv_act laid out [PMAX, NUM_QKV_TILES]. Slot t holds channels
    # [t*PMAX : (t+1)*PMAX]. Phase 3 reads slots:
    #   q for k-head kh:    slot kh                  (0..3)
    #   k for k-head kh:    slot K_HEADS + kh        (4..7)
    #   v for v-head h:     slot 2*K_HEADS + h       (8..19)
    # =========================================================================
    qkv_act = nl.ndarray((PMAX, NUM_QKV_TILES), dtype=nl.float32, buffer=nl.sbuf)
    zb_p = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zb_p, value=0.0)

    for t in nl.affine_range(NUM_QKV_TILES):
        ch_start = t * PMAX

        cs_bf = nl.ndarray((PMAX, 3), dtype=conv_state.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=cs_bf, src=conv_state[ch_start:ch_start + PMAX, 0:3])
        cs_f = nl.ndarray((PMAX, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=cs_f, op=nl.copy, data=cs_bf, bias=zb_p, scale=1.0)

        nq_bf = nl.ndarray((PMAX, 1), dtype=mixed_qkv.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=nq_bf, src=mixed_qkv[ch_start:ch_start + PMAX])
        nq_f = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=nq_f, op=nl.copy, data=nq_bf, bias=zb_p, scale=1.0)

        conv_in = nl.ndarray((PMAX, CONV_KERNEL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=conv_in[0:PMAX, 0:3], src=cs_f)
        nisa.tensor_copy(dst=conv_in[0:PMAX, 3:4], src=nq_f)

        # Write back updated conv_state (drop oldest column)
        ncs = nl.ndarray((PMAX, 3), dtype=conv_state.dtype, buffer=nl.sbuf)
        nisa.activation(dst=ncs, op=nl.copy, data=conv_in[0:PMAX, 1:4], bias=zb_p, scale=1.0)
        nisa.dma_copy(dst=new_conv_state[ch_start:ch_start + PMAX, 0:3], src=ncs)

        cw = nl.ndarray((PMAX, CONV_KERNEL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cw, src=conv_weight[ch_start:ch_start + PMAX, 0:CONV_KERNEL])
        prod = nl.ndarray((PMAX, CONV_KERNEL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=conv_in, data2=cw, op=nl.multiply)

        conv_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=prod, op=nl.copy, data=prod, bias=zb_p, scale=1.0,
            reduce_op=nl.add, reduce_res=conv_sum,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )

        # Opt 2: fold bias-add + silu into one activation call.
        # activation computes op(scale*data + bias); silu(conv_sum + cb).
        cb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cb, src=conv_bias[ch_start:ch_start + PMAX])
        act = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=act, op=nl.silu, data=conv_sum, bias=cb, scale=1.0)
        nisa.tensor_copy(dst=qkv_act[0:PMAX, t:t + 1], src=act)

    # =========================================================================
    # Persistent helpers reused across the V_HEADS loop
    # =========================================================================
    eps_tile = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_tile, value=RMS_EPS)
    zb1 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zb1, value=0.0)
    zk_row = nl.ndarray((1, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zk_row, value=0.0)
    zv_row = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zv_row, value=0.0)
    z_kv = nl.ndarray((K_DIM, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=z_kv, value=0.0)
    z_v1 = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=z_v1, value=0.0)
    ones_k = nl.ndarray((1, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_k, value=1.0)
    ones_v = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_v, value=1.0)

    # Load norm_weight once as a row [1, V_DIM]
    nw_row = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nw_row, src=norm_weight[0:V_DIM])

    # Opt 3: hoist silu(z) out of the V_HEADS loop.
    # Compute silu(z[V_HEADS, V_DIM]) once, write to HBM scratch, then
    # per-head DMA back inside the loop. HBM round-trip (vs SBUF pickoff)
    # avoids the [V_HEADS, V_DIM] partition-slice failure pattern.
    z_bf_full = nl.ndarray((V_HEADS, V_DIM), dtype=z.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=z_bf_full, src=z[0:V_HEADS, 0:V_DIM])
    zb_v_row = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zb_v_row, value=0.0)
    silu_z_sbuf = nl.ndarray((V_HEADS, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=silu_z_sbuf, op=nl.silu, data=z_bf_full, bias=zb_v_row, scale=1.0,
    )
    nisa.dma_copy(dst=silu_z_hbm[0:V_HEADS, 0:V_DIM], src=silu_z_sbuf)

    # Opt 5: hoist gate computation out of the V_HEADS loop.
    # exp_g[h] = exp(-exp(A_log[h]) * softplus(a[h] + dt_bias[h]))
    # beta[h]  = sigmoid(b[h])
    # All [V_HEADS, 1] tiles — V_HEADS=12 fits one tile. Saves 8 ScalarE
    # ops per inner iter * 12 iters = 96 ops total.
    a_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_full, src=a_out[0:V_HEADS])
    dt_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=dt_full, src=dt_bias[0:V_HEADS])
    Al_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Al_full, src=A_log[0:V_HEADS])
    b_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_full, src=b_out[0:V_HEADS])

    # softplus(a + dt) — fold the add via activation bias.
    sp_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=sp_full, op=nl.softplus, data=a_full, bias=dt_full, scale=1.0,
    )
    # exp(A_log)
    expA_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=expA_full, op=nl.exp, data=Al_full, bias=zb_v_row, scale=1.0,
    )
    # pos = sp * expA
    pos_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=pos_full, data1=sp_full, data2=expA_full, op=nl.multiply)
    # exp_g = exp(-pos)
    exp_g_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_g_full, op=nl.exp, data=pos_full, bias=zb_v_row, scale=-1.0,
    )
    nisa.dma_copy(dst=exp_g_hbm[0:V_HEADS, 0:1], src=exp_g_full)
    # beta = sigmoid(b)
    beta_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=beta_full, op=nl.sigmoid, data=b_full, bias=zb_v_row, scale=1.0,
    )
    nisa.dma_copy(dst=beta_hbm[0:V_HEADS, 0:1], src=beta_full)

    # =========================================================================
    # PHASE 2+3 — outer over K_HEADS, inner over HEAD_GROUP.
    # For each k-head kh, compute q_row/k_row once (with L2 norm + Q-scale).
    # Then for each ig in 0..HEAD_GROUP-1, the v-head is h = kh*HEAD_GROUP + ig.
    # That inner head index runs the recurrence and RMSNormGated.
    # =========================================================================
    for kh in nl.affine_range(K_HEADS):
        # ----- q for k-head kh: qkv_act[:, kh:kh+1] -> normed + Q_SCALE -----
        # Opt 4: use nc_matmul(q.T @ q) for the L2 sum-of-squares to avoid
        # the col->row transpose round-trip. q_col stays in column form
        # throughout — the recurrent matmul uses stationary=s_new with
        # moving=q_col directly. (Transposes saved: 2 per K_HEADS = 8 total.)
        q_col = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_col, src=qkv_act[0:K_DIM, kh:kh + 1])

        # qsum = q.T @ q : [1,1] in PSUM. Both operands are [K_DIM, 1] cols;
        # nc_matmul treats stationary K dim as the contraction dim, giving
        # stationary[K,M].T @ moving[K,N] -> [M, N] = [1, 1].
        qsum_p = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=qsum_p, stationary=q_col, moving=q_col)
        qsum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qsum, src=qsum_p)
        # Opt 1: fold eps-add + rsqrt
        qrinv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=qrinv, op=nl.rsqrt, data=qsum, bias=eps_tile, scale=1.0)
        # Broadcast qrinv [1,1] -> [K_DIM, 1] via matmul with ones_k[1,K]
        # (already used elsewhere in this kernel for exp_g/beta broadcasts).
        qrinv_p = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=qrinv_p, stationary=ones_k, moving=qrinv)
        qrinv_vec = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qrinv_vec, src=qrinv_p)
        # q_colS = q_col * qrinv_vec * Q_SCALE — combine norm and scale.
        # scalar_tensor_tensor: dst = op1(op0(scale * data, operand0), operand1).
        # Use scale=Q_SCALE so the multiply by 1/sqrt(K_DIM) folds in for free.
        q_pre = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=q_pre, data=q_col,
            op0=nl.multiply, operand0=qrinv_vec,
            op1=nl.add, operand1=z_v1,  # [K_DIM,1] zeros, K_DIM==V_DIM
        )
        q_colS = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
        zb_k1 = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zb_k1, value=0.0)
        nisa.activation(dst=q_colS, op=nl.copy, data=q_pre, bias=zb_k1, scale=Q_SCALE)

        # ----- k for k-head kh: qkv_act[:, K_HEADS+kh] -> normed -----
        # Opt 4: keep k in column form. Same nc_matmul(k.T @ k) trick for
        # sum-of-squares as for q. k_colN is the column form used directly
        # as moving operand by the recurrent kv_mem matmul (no transpose).
        k_slot = K_HEADS + kh
        k_col_in = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_col_in, src=qkv_act[0:K_DIM, k_slot:k_slot + 1])

        ksum_p = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ksum_p, stationary=k_col_in, moving=k_col_in)
        ksum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ksum, src=ksum_p)
        # Opt 1: fold eps-add + rsqrt
        krinv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=krinv, op=nl.rsqrt, data=ksum, bias=eps_tile, scale=1.0)
        # Broadcast krinv -> [K_DIM, 1] then scale k_col_in.
        krinv_p = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=krinv_p, stationary=ones_k, moving=krinv)
        krinv_vec = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=krinv_vec, src=krinv_p)
        k_colN = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=k_colN, data=k_col_in,
            op0=nl.multiply, operand0=krinv_vec,
            op1=nl.add, operand1=z_v1,  # [K_DIM,1] zeros (K_DIM==V_DIM)
        )
        # k_normed [1, K_DIM] is still needed for the outer matmul stationary
        # later in the inner loop (k_normed.T @ delta_row -> [K, V]).
        # Compute it once via a single transpose here — saves N-1 transposes
        # vs doing one per inner iter. (4 transposes total vs 12 before.)
        knrow_p = nl.ndarray((1, K_DIM), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=knrow_p, data=k_colN)
        k_normed = nl.ndarray((1, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_normed, src=knrow_p)

        # ----- inner over HEAD_GROUP v-heads sharing this k-head -----
        for ig in nl.affine_range(HEAD_GROUP):
            h = kh * HEAD_GROUP + ig

            # Opt 5: gate scalars (exp_g, beta) precomputed in HBM —
            # just DMA back the per-head [1,1] slice.
            exp_g = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=exp_g, src=exp_g_hbm[h:h + 1, 0:1])
            beta_s = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=beta_s, src=beta_hbm[h:h + 1, 0:1])

            # ----- Recurrence -----
            # Load state[h*K_DIM:(h+1)*K_DIM, :]
            s = nl.ndarray((K_DIM, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s, src=state[h * K_DIM:(h + 1) * K_DIM, 0:V_DIM])

            # Broadcast exp_g [1,1] -> [K_DIM, 1] via matmul: ones_k[1,K].T @ exp_g[1,1]
            decay_p = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=decay_p, stationary=ones_k, moving=exp_g)
            decay_vec = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=decay_vec, src=decay_p)

            # s_dec = s * decay_vec (broadcast along V dim)
            s_dec = nl.ndarray((K_DIM, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=s_dec, data=s,
                op0=nl.multiply, operand0=decay_vec,
                op1=nl.add, operand1=z_kv,
            )

            # Get k_col (K_DIM,1) for matmul moving
            # Opt 4: k_colN is already in column form from outer K_HEADS loop;
            # skip the per-iter transpose. (Saves 12 transposes per call.)

            # kv_mem = s_dec.T @ k_colN -> [V_DIM, 1]
            kv_p = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=kv_p, stationary=s_dec, moving=k_colN)
            kv_mem = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=kv_mem, src=kv_p)

            # Load v from qkv_act slot 2*K_HEADS + h, transpose to [V_DIM, 1]
            v_slot = 2 * K_HEADS + h
            v_col_in = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=v_col_in, src=qkv_act[0:V_DIM, v_slot:v_slot + 1])
            # Already a column

            # diff = v - kv_mem
            diff = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=diff, data1=v_col_in, data2=kv_mem, op=nl.subtract)

            # Broadcast beta to [V_DIM, 1]
            beta_p = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=beta_p, stationary=ones_v, moving=beta_s)
            beta_vec = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=beta_vec, src=beta_p)

            # delta = diff * beta_vec
            delta_col = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=delta_col, data=diff,
                op0=nl.multiply, operand0=beta_vec,
                op1=nl.add, operand1=z_v1,
            )

            # outer = k_normed[1,K].T @ delta_row[1,V] -> [K, V]
            d_p = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=d_p, data=delta_col)
            delta_row = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=delta_row, src=d_p)

            outer_p = nl.ndarray((K_DIM, V_DIM), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=outer_p, stationary=k_normed, moving=delta_row)
            outer_t = nl.ndarray((K_DIM, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=outer_t, src=outer_p)

            s_new = nl.ndarray((K_DIM, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=s_new, data1=s_dec, data2=outer_t, op=nl.add)

            # Store new state
            nisa.dma_copy(dst=new_state[h * K_DIM:(h + 1) * K_DIM, 0:V_DIM], src=s_new)

            # attn = s_new.T @ q_colS -> [V_DIM, 1]
            attn_p = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=attn_p, stationary=s_new, moving=q_colS)
            attn_col = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=attn_col, src=attn_p)

            # Opt 6: compute attn^2 sum on TensorE via attn.T @ attn while
            # attn is still in column form — saves a ScalarE square+reduce.
            asum_p = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=asum_p, stationary=attn_col, moving=attn_col)
            asum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=asum, src=asum_p)

            # transpose to row [1, V_DIM]
            attn_rowp = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=attn_rowp, data=attn_col)
            attn_row = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=attn_row, src=attn_rowp)

            # ----- RMSNormGated -----
            # rms = attn * rsqrt(mean(attn^2) + eps); out = silu(z) * norm_w * rms
            # Opt 1: fold mean-divide + eps-add + rsqrt into one op:
            # rms_inv = rsqrt(asum * (1/V_DIM) + eps_tile)
            rms_inv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=rms_inv, op=nl.rsqrt, data=asum,
                bias=eps_tile, scale=1.0 / V_DIM,
            )

            # attn_norm = attn_row * rms_inv (broadcast)
            attn_norm = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=attn_norm, data=attn_row,
                op0=nl.multiply, operand0=rms_inv,
                op1=nl.add, operand1=zv_row,
            )
            # weighted = attn_norm * norm_weight
            weighted = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=weighted, data1=attn_norm, data2=nw_row, op=nl.multiply)

            # Opt 3: silu(z[h]) precomputed in HBM — DMA back the slice.
            sz = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=sz, src=silu_z_hbm[h:h + 1, 0:V_DIM])

            # gated_f32 = weighted * sz
            gated_f = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=gated_f, data1=weighted, data2=sz, op=nl.multiply)
            # cast to bf16 + DMA out
            gated_bf = nl.ndarray((1, V_DIM), dtype=z.dtype, buffer=nl.sbuf)
            nisa.activation(dst=gated_bf, op=nl.copy, data=gated_f, bias=zb1, scale=1.0)
            nisa.dma_copy(dst=output[h:h + 1, 0:V_DIM], src=gated_bf)

    return new_state, new_conv_state, output
