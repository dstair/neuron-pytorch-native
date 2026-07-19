"""
DeltaNet "full" NKI kernel — BATCHED via outer affine_range(B).

Identical fused inner block as deltanet_full.py (conv+silu, L2norm+Qscale,
gates, recurrence, RMSNormGated), but wraps the whole body in an outer loop
over batch elements so a single kernel call (one custom-call boundary)
processes all B sequences. This is the "batch-as-heads" idea applied to the
full fused kernel: the per-element Python loop in batch_decode (B custom-calls
per layer) collapses to ONE call.

Batch layout (B sequences, weights SHARED across batch):
  state:        [B*V_HEADS*K_DIM, V_DIM] f32   in   (head row for (b,h) = (b*V_HEADS+h)*K_DIM)
  mixed_qkv:    [B*QKV_DIM]              bf16  in
  conv_state:   [B*QKV_DIM, 3]           bf16  in
  conv_weight:  [QKV_DIM, 4]             f32   in   SHARED (weight)
  conv_bias:    [QKV_DIM]                f32   in   SHARED (weight)
  a_out:        [B*V_HEADS]              f32   in
  b_out:        [B*V_HEADS]              f32   in
  z:            [B*V_HEADS, V_DIM]       bf16  in
  A_log:        [V_HEADS]                f32   in   SHARED (weight)
  dt_bias:      [V_HEADS]                f32   in   SHARED (weight)
  norm_weight:  [V_DIM]                  f32   in   SHARED (weight)

Returns:
  new_state:      [B*V_HEADS*K_DIM, V_DIM] f32
  new_conv_state: [B*QKV_DIM, 3]           bf16
  output:         [B*V_HEADS, V_DIM]       bf16  gated, ready for out_proj
"""
import math
import os as _os
import nki
import nki.isa as nisa
import nki.language as nl


# 35B-A3B per-core head counts (TP=4): K_HEADS=4 (16/4), V_HEADS=8 (32/4).
# Same kernel body as the 27B deltanet_full_batched; only these constants differ
# (V_HEADS 12->8, HEAD_GROUP 3->2, QKV_DIM 2560->2048). Overridable via env for
# other TP degrees. Constraints (all hold): K_DIM=V_DIM=PMAX=128,
# QKV_DIM % 128 == 0 (2048=16*128), V_HEADS % K_HEADS == 0 (8%4==0).
K_DIM = 128
V_DIM = 128
K_HEADS = int(_os.environ.get("DN_K_HEADS", "4"))
V_HEADS = int(_os.environ.get("DN_V_HEADS", "8"))
HEAD_GROUP = V_HEADS // K_HEADS  # 2
QKV_DIM = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM  # 2048
CONV_KERNEL = 4
RMS_EPS = 1e-6


@nki.jit
def nki_deltanet_full_batched(
    state,         # [B*V_HEADS*K_DIM, V_DIM] f32
    mixed_qkv,     # [B*QKV_DIM] bf16
    conv_state,    # [B*QKV_DIM, 3] bf16
    conv_weight,   # [QKV_DIM, 4] f32 (shared)
    conv_bias,     # [QKV_DIM] f32 (shared)
    a_out,         # [B*V_HEADS] f32
    b_out,         # [B*V_HEADS] f32
    z,             # [B*V_HEADS, V_DIM] bf16
    A_log,         # [V_HEADS] f32 (shared)
    dt_bias,       # [V_HEADS] f32 (shared)
    norm_weight,   # [V_DIM] f32 (shared)
):
    # Derive batch size from the state row count (static at trace time).
    B = state.shape[0] // (V_HEADS * K_DIM)

    new_state = nl.ndarray((B * V_HEADS * K_DIM, V_DIM), dtype=state.dtype, buffer=nl.shared_hbm)
    new_conv_state = nl.ndarray((B * QKV_DIM, 3), dtype=conv_state.dtype, buffer=nl.shared_hbm)
    output = nl.ndarray((B * V_HEADS, V_DIM), dtype=z.dtype, buffer=nl.shared_hbm)
    # Per-batch HBM scratch (sized B*V_HEADS so b slices are disjoint).
    silu_z_hbm = nl.ndarray((B * V_HEADS, V_DIM), dtype=nl.float32, buffer=nl.shared_hbm)
    exp_g_hbm = nl.ndarray((B * V_HEADS, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    beta_hbm = nl.ndarray((B * V_HEADS, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    PMAX = nl.tile_size.pmax  # 128
    NUM_QKV_TILES = QKV_DIM // PMAX  # 20
    Q_SCALE = 1.0 / math.sqrt(K_DIM)

    # =========================================================================
    # Read-only helpers — hoisted ABOVE the batch loop (shared across all b).
    # =========================================================================
    eps_tile = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_tile, value=RMS_EPS)
    zb1 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zb1, value=0.0)
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
    # norm_weight is shared — load once as [1, V_DIM]
    nw_row = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nw_row, src=norm_weight[0:V_DIM])
    # Shared gate weights loaded once: A_log, dt_bias as [V_HEADS, 1]
    dt_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=dt_full, src=dt_bias[0:V_HEADS])
    Al_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Al_full, src=A_log[0:V_HEADS])
    zb_v_row = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zb_v_row, value=0.0)
    # exp(A_log) is batch-independent — compute once.
    expA_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=expA_full, op=nl.exp, data=Al_full, bias=zb_v_row, scale=1.0)

    # =========================================================================
    # BATCH LOOP — independent per b, so affine_range (pipelined).
    # =========================================================================
    for b in nl.affine_range(B):
        qkv_base = b * QKV_DIM
        head_base = b * V_HEADS  # head index offset for (b, h)

        # ---------------------------------------------------------------------
        # PHASE 1 — conv state update + depthwise conv + SiLU (this b)
        # ---------------------------------------------------------------------
        qkv_act = nl.ndarray((PMAX, NUM_QKV_TILES), dtype=nl.float32, buffer=nl.sbuf)
        zb_p = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zb_p, value=0.0)

        for t in nl.affine_range(NUM_QKV_TILES):
            ch = qkv_base + t * PMAX

            cs_bf = nl.ndarray((PMAX, 3), dtype=conv_state.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=cs_bf, src=conv_state[ch:ch + PMAX, 0:3])
            cs_f = nl.ndarray((PMAX, 3), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=cs_f, op=nl.copy, data=cs_bf, bias=zb_p, scale=1.0)

            nq_bf = nl.ndarray((PMAX, 1), dtype=mixed_qkv.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=nq_bf, src=mixed_qkv[ch:ch + PMAX])
            nq_f = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=nq_f, op=nl.copy, data=nq_bf, bias=zb_p, scale=1.0)

            conv_in = nl.ndarray((PMAX, CONV_KERNEL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=conv_in[0:PMAX, 0:3], src=cs_f)
            nisa.tensor_copy(dst=conv_in[0:PMAX, 3:4], src=nq_f)

            # Write back updated conv_state (drop oldest column)
            ncs = nl.ndarray((PMAX, 3), dtype=conv_state.dtype, buffer=nl.sbuf)
            nisa.activation(dst=ncs, op=nl.copy, data=conv_in[0:PMAX, 1:4], bias=zb_p, scale=1.0)
            nisa.dma_copy(dst=new_conv_state[ch:ch + PMAX, 0:3], src=ncs)

            # conv_weight / conv_bias are SHARED — index by local channel only.
            cw_ch = t * PMAX
            cw = nl.ndarray((PMAX, CONV_KERNEL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=cw, src=conv_weight[cw_ch:cw_ch + PMAX, 0:CONV_KERNEL])
            prod = nl.ndarray((PMAX, CONV_KERNEL), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prod, data1=conv_in, data2=cw, op=nl.multiply)

            conv_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=prod, op=nl.copy, data=prod, bias=zb_p, scale=1.0,
                reduce_op=nl.add, reduce_res=conv_sum,
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )

            cb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=cb, src=conv_bias[cw_ch:cw_ch + PMAX])
            act = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=act, op=nl.silu, data=conv_sum, bias=cb, scale=1.0)
            nisa.tensor_copy(dst=qkv_act[0:PMAX, t:t + 1], src=act)

        # ---------------------------------------------------------------------
        # silu(z) for this b -> HBM scratch slice [head_base:head_base+V_HEADS]
        # ---------------------------------------------------------------------
        z_bf_full = nl.ndarray((V_HEADS, V_DIM), dtype=z.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=z_bf_full, src=z[head_base:head_base + V_HEADS, 0:V_DIM])
        silu_z_sbuf = nl.ndarray((V_HEADS, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=silu_z_sbuf, op=nl.silu, data=z_bf_full, bias=zb_v_row, scale=1.0)
        nisa.dma_copy(dst=silu_z_hbm[head_base:head_base + V_HEADS, 0:V_DIM], src=silu_z_sbuf)

        # ---------------------------------------------------------------------
        # Gates for this b: exp_g = exp(-exp(A_log)*softplus(a+dt)); beta=sig(b)
        # A_log/dt shared (expA_full hoisted); a_out/b_out are per-(b,h).
        # ---------------------------------------------------------------------
        a_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_full, src=a_out[head_base:head_base + V_HEADS])
        b_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_full, src=b_out[head_base:head_base + V_HEADS])

        sp_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sp_full, op=nl.softplus, data=a_full, bias=dt_full, scale=1.0)
        pos_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=pos_full, data1=sp_full, data2=expA_full, op=nl.multiply)
        exp_g_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_g_full, op=nl.exp, data=pos_full, bias=zb_v_row, scale=-1.0)
        nisa.dma_copy(dst=exp_g_hbm[head_base:head_base + V_HEADS, 0:1], src=exp_g_full)
        beta_full = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=beta_full, op=nl.sigmoid, data=b_full, bias=zb_v_row, scale=1.0)
        nisa.dma_copy(dst=beta_hbm[head_base:head_base + V_HEADS, 0:1], src=beta_full)

        # ---------------------------------------------------------------------
        # PHASE 2+3 — outer K_HEADS, inner HEAD_GROUP (this b)
        # ---------------------------------------------------------------------
        for kh in nl.affine_range(K_HEADS):
            q_col = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=q_col, src=qkv_act[0:K_DIM, kh:kh + 1])

            qsum_p = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=qsum_p, stationary=q_col, moving=q_col)
            qsum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qsum, src=qsum_p)
            qrinv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=qrinv, op=nl.rsqrt, data=qsum, bias=eps_tile, scale=1.0)
            qrinv_p = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=qrinv_p, stationary=ones_k, moving=qrinv)
            qrinv_vec = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qrinv_vec, src=qrinv_p)
            q_pre = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=q_pre, data=q_col,
                op0=nl.multiply, operand0=qrinv_vec,
                op1=nl.add, operand1=z_v1,
            )
            q_colS = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            zb_k1 = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=zb_k1, value=0.0)
            nisa.activation(dst=q_colS, op=nl.copy, data=q_pre, bias=zb_k1, scale=Q_SCALE)

            k_slot = K_HEADS + kh
            k_col_in = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_col_in, src=qkv_act[0:K_DIM, k_slot:k_slot + 1])

            ksum_p = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ksum_p, stationary=k_col_in, moving=k_col_in)
            ksum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ksum, src=ksum_p)
            krinv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=krinv, op=nl.rsqrt, data=ksum, bias=eps_tile, scale=1.0)
            krinv_p = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=krinv_p, stationary=ones_k, moving=krinv)
            krinv_vec = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=krinv_vec, src=krinv_p)
            k_colN = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=k_colN, data=k_col_in,
                op0=nl.multiply, operand0=krinv_vec,
                op1=nl.add, operand1=z_v1,
            )
            knrow_p = nl.ndarray((1, K_DIM), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=knrow_p, data=k_colN)
            k_normed = nl.ndarray((1, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_normed, src=knrow_p)

            for ig in nl.affine_range(HEAD_GROUP):
                h = kh * HEAD_GROUP + ig
                gh = head_base + h  # global (b,h) head index

                exp_g = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=exp_g, src=exp_g_hbm[gh:gh + 1, 0:1])
                beta_s = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=beta_s, src=beta_hbm[gh:gh + 1, 0:1])

                s = nl.ndarray((K_DIM, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=s, src=state[gh * K_DIM:(gh + 1) * K_DIM, 0:V_DIM])

                decay_p = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=decay_p, stationary=ones_k, moving=exp_g)
                decay_vec = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=decay_vec, src=decay_p)

                # F1: In-place state decay. s := s * decay_vec instead of
                # allocating a fresh s_dec buffer. Bit-identical; saves 1
                # (K_DIM, V_DIM) SBUF alloc per (b, kh, ig) per token.
                # Pattern from HF Hub v2.0-task007.
                # (Original v1 note: drops the wasted `+ z_kv` add-with-zero
                # the old scalar_tensor_tensor carried.)
                nisa.tensor_scalar(
                    dst=s, data=s, op0=nl.multiply, operand0=decay_vec,
                )

                kv_p = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=kv_p, stationary=s, moving=k_colN)

                v_slot = 2 * K_HEADS + h
                v_col_in = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=v_col_in, src=qkv_act[0:V_DIM, v_slot:v_slot + 1])

                # F2: PSUM read fusion. Read kv_p directly as the second
                # operand of tensor_tensor, saving a (V_DIM, 1) SBUF alloc
                # and a tensor_copy per (b, kh, ig) per token.
                # Pattern from HF Hub v2.0-task007.
                diff = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=diff, data1=v_col_in, data2=kv_p, op=nl.subtract)

                beta_p = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=beta_p, stationary=ones_v, moving=beta_s)
                beta_vec = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=beta_vec, src=beta_p)

                # delta = diff * beta_vec. Plain tensor_scalar (drops the +z_v1
                # add-with-zero of the old scalar_tensor_tensor).
                delta_col = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=delta_col, data=diff, op0=nl.multiply, operand0=beta_vec,
                )

                d_p = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=d_p, data=delta_col)
                delta_row = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=delta_row, src=d_p)

                outer_p = nl.ndarray((K_DIM, V_DIM), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=outer_p, stationary=k_normed, moving=delta_row)
                outer_t = nl.ndarray((K_DIM, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=outer_t, src=outer_p)

                # F3: In-place state update. s := s + outer_t (where s is
                # the decayed state from F1). Saves a (K_DIM, V_DIM) SBUF
                # alloc per (b, kh, ig) per token. Pattern from HF Hub
                # v2.0-task007.
                nisa.tensor_tensor(dst=s, data1=s, data2=outer_t, op=nl.add)

                nisa.dma_copy(dst=new_state[gh * K_DIM:(gh + 1) * K_DIM, 0:V_DIM], src=s)

                attn_p = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=attn_p, stationary=s, moving=q_colS)
                attn_col = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=attn_col, src=attn_p)

                asum_p = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=asum_p, stationary=attn_col, moving=attn_col)
                asum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=asum, src=asum_p)

                attn_rowp = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=attn_rowp, data=attn_col)
                attn_row = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=attn_row, src=attn_rowp)

                rms_inv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    dst=rms_inv, op=nl.rsqrt, data=asum,
                    bias=eps_tile, scale=1.0 / V_DIM,
                )
                attn_norm = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(
                    dst=attn_norm, data=attn_row,
                    op0=nl.multiply, operand0=rms_inv,
                    op1=nl.add, operand1=zv_row,
                )
                weighted = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=weighted, data1=attn_norm, data2=nw_row, op=nl.multiply)

                sz = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=sz, src=silu_z_hbm[gh:gh + 1, 0:V_DIM])

                gated_f = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=gated_f, data1=weighted, data2=sz, op=nl.multiply)
                gated_bf = nl.ndarray((1, V_DIM), dtype=z.dtype, buffer=nl.sbuf)
                nisa.activation(dst=gated_bf, op=nl.copy, data=gated_f, bias=zb1, scale=1.0)
                nisa.dma_copy(dst=output[gh:gh + 1, 0:V_DIM], src=gated_bf)

    return new_state, new_conv_state, output
