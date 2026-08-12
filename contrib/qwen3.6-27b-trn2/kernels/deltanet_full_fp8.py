"""
DeltaNet "full + FP8 in_proj" all-in-one NKI kernel.

Extends deltanet_full.py to also fuse the four input projection GEMMs
(in_proj_qkv, in_proj_z, in_proj_a, in_proj_b) into the same kernel,
with FP8 W8A16 weights. This:

  1. Halves weight bandwidth for the 4 in_proj GEMMs (the dominant cost
     of decode per the E2E profile: 2.5 GB/step weight reads at 84% MBU).
  2. Reduces graph nodes from 5 separate ops to 1 nki_op per deltanet
     layer (×48 layers = 240 fewer custom-calls), avoiding the
     compile-time blowup we hit with per-Linear FP8 nki_ops.

Layout (TP=4 per-core):
  Inputs (HBM):
    x:              [1, HIDDEN=5120]      bf16  — input activation
    state:          [V_HEADS*K_DIM=1536, V_DIM=128] f32  in/out
    conv_state:     [QKV_DIM=2560, 3]     bf16  in/out
    conv_weight:    [QKV_DIM, 4]          f32
    conv_bias:      [QKV_DIM]             f32
    A_log, dt_bias: [V_HEADS=12]          f32
    norm_weight:    [V_DIM]               f32
    qkv_w_T_i8:     [HIDDEN, QKV_DIM=2560] int8   (FP8 e4m3 bytes, [K, N])
    qkv_s:          [QKV_DIM, 1]          f32
    z_w_T_i8:       [HIDDEN, 1536]        int8
    z_s:            [1536, 1]             f32
    a_w_T_i8:       [HIDDEN, V_HEADS=12]  int8
    a_s:            [V_HEADS, 1]          f32
    b_w_T_i8:       [HIDDEN, V_HEADS=12]  int8
    b_s:            [V_HEADS, 1]          f32

  Outputs (HBM):
    new_state:      [V_HEADS*K_DIM, V_DIM] f32
    new_conv_state: [QKV_DIM, 3]           bf16
    output:         [V_HEADS, V_DIM]       bf16   gated, ready for out_proj

Memory thrift:
  - x is loaded ONCE as a [PMAX, NUM_X_TILES] SBUF buffer (5120/128 = 40
    K-tiles), reused across all 4 in_proj GEMMs.
  - mixed_qkv is computed directly into the existing qkv_act SBUF buffer
    (same [PMAX, NUM_QKV_TILES] layout phase 3 already reads from).
    The conv path then operates on these tiles in-place rather than
    DMA-loading from HBM.
  - z, a, b results are kept in SBUF — no HBM round-trip.

FP8 reinterpretation: same trick as fp8_matmul.py — weights are passed
as int8 (bytes valid in legacy-e4m3 since FP8_MAX=240 quantization), and
the SBUF tile is loaded as int8 then `.view(nl.float8_e4m3)` for matmul.
"""
import math
import nki
import nki.isa as nisa
import nki.language as nl


HIDDEN = 5120
K_DIM = 128
V_DIM = 128
K_HEADS = 4
V_HEADS = 12
HEAD_GROUP = V_HEADS // K_HEADS  # 3
QKV_DIM = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM  # 2560
Z_DIM = V_HEADS * V_DIM                          # 1536
CONV_KERNEL = 4
RMS_EPS = 1e-6
PMAX = 128


def _div_ceil(n, d):
    return (n + d - 1) // d


@nki.jit
def nki_deltanet_full_fp8(
    x,             # [1, HIDDEN] bf16
    state,         # [V_HEADS*K_DIM, V_DIM] f32
    conv_state,    # [QKV_DIM, 3] bf16
    conv_weight,   # [QKV_DIM, 4] f32
    conv_bias,     # [QKV_DIM] f32
    A_log,         # [V_HEADS] f32
    dt_bias,       # [V_HEADS] f32
    norm_weight,   # [V_DIM] f32
    qkv_w_T_i8,    # [HIDDEN, QKV_DIM] int8 (fp8 bytes)
    qkv_s,         # [QKV_DIM, 1] f32
    z_w_T_i8,      # [HIDDEN, Z_DIM] int8
    z_s,           # [Z_DIM, 1] f32
    a_w_T_i8,      # [HIDDEN, V_HEADS] int8
    a_s,           # [V_HEADS, 1] f32
    b_w_T_i8,      # [HIDDEN, V_HEADS] int8
    b_s,           # [V_HEADS, 1] f32
):
    new_state = nl.ndarray((V_HEADS * K_DIM, V_DIM), dtype=state.dtype, buffer=nl.shared_hbm)
    new_conv_state = nl.ndarray((QKV_DIM, 3), dtype=conv_state.dtype, buffer=nl.shared_hbm)
    output = nl.ndarray((V_HEADS, V_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    silu_z_hbm = nl.ndarray((V_HEADS, V_DIM), dtype=nl.float32, buffer=nl.shared_hbm)
    exp_g_hbm = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    beta_hbm = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    # z stays in HBM scratch — building it row-by-row in SBUF triggers the
    # [V_HEADS, V_DIM] partition-slice failure that broke fused_inner. Same
    # pattern as silu_z_hbm: build per-row, DMA out, then materialize as a
    # full [V_HEADS, V_DIM] SBUF tile in one DMA-in for silu(z).
    z_hbm = nl.ndarray((V_HEADS, V_DIM), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    NUM_X_TILES = HIDDEN // PMAX        # 40
    NUM_QKV_TILES = QKV_DIM // PMAX     # 20
    NUM_Z_TILES = Z_DIM // PMAX         # 12
    Q_SCALE = 1.0 / math.sqrt(K_DIM)

    # =========================================================================
    # PHASE 0a — load x[1, HIDDEN] into [PMAX, NUM_X_TILES] tiled SBUF.
    # x is the moving operand for every in_proj matmul. We load it once
    # transposed onto the K partition dim, so each of the 40 K-tiles can
    # be moved into a single nc_matmul call.
    # =========================================================================
    # Stage 1: load [1, HIDDEN] as a row, transpose to columns.
    x_row = nl.ndarray((1, HIDDEN), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_row, src=x[0:1, 0:HIDDEN])
    # Stage 2: chop into NUM_X_TILES columns of PMAX each, transposing each
    # so the K dim sits on partitions. After this, x_tiles[t][:, 0:1] is
    # the t-th K-tile of x as a [PMAX, 1] column.
    x_tiles = nl.ndarray((PMAX, NUM_X_TILES), dtype=x.dtype, buffer=nl.sbuf)
    for t in nl.affine_range(NUM_X_TILES):
        k_start = t * PMAX
        # Transpose [1, PMAX] -> [PMAX, 1]
        chunk = nl.ndarray((1, PMAX), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=chunk, src=x_row[0:1, k_start:k_start + PMAX])
        chunk_T_p = nl.ndarray((PMAX, 1), dtype=x.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=chunk_T_p, data=chunk)
        nisa.tensor_copy(dst=x_tiles[0:PMAX, t:t + 1], src=chunk_T_p)

    # =========================================================================
    # Helper: FP8 matmul fused with per-channel scale.
    # Computes y[N_tile, 1] = sum_k (w_fp8[K_tile, N_tile].T @ x[K_tile, 1])
    #   then multiply by scale[N_tile, 1].
    # Returns f32 SBUF [N_tile, 1].
    #
    # Implemented INLINE per call site below (NKI doesn't allow Python helpers
    # that allocate SBUF tiles to be reused cleanly — the allocator gets
    # confused). Pattern is repeated for each of the 4 GEMMs.
    # =========================================================================

    # Persistent scratch for the f32 zero-add operand.
    zb_p = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zb_p, value=0.0)

    # =========================================================================
    # PHASE 0b — in_proj_qkv: mixed_qkv = x @ qkv_w.T  -> qkv_act tiles
    # qkv_w_T_i8 is [HIDDEN, QKV_DIM] = [K, N]. NUM_QKV_TILES N-tiles.
    # We compute mixed_qkv[N_tile=128, 1] f32, multiply by scale, then run
    # the conv pipeline on it (conv expects bf16-cast values; we cast).
    # The conv result lands in qkv_act tile slot t.
    # =========================================================================
    qkv_act = nl.ndarray((PMAX, NUM_QKV_TILES), dtype=nl.float32, buffer=nl.sbuf)

    for n_idx in nl.affine_range(NUM_QKV_TILES):
        n_start = n_idx * PMAX

        # PSUM accumulator: [N_tile=PMAX, 1]
        accum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.psum)
        for k_idx in nl.affine_range(NUM_X_TILES):
            k_start = k_idx * PMAX
            # Load weight tile [K_tile=PMAX, N_tile=PMAX] as int8 then view as fp8.
            w_i8 = nl.ndarray((PMAX, PMAX), dtype=nl.int8, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=w_i8,
                src=qkv_w_T_i8[k_start:k_start + PMAX, n_start:n_start + PMAX],
            )
            w_fp8 = w_i8.view(nl.float8_e4m3)
            # x tile is already in SBUF as a [PMAX, 1] column at x_tiles[:, k_idx].
            nisa.nc_matmul(
                dst=accum,
                stationary=w_fp8,
                moving=x_tiles[0:PMAX, k_idx:k_idx + 1],
            )

        # Apply per-channel scale and convert. Load qkv_s tile [PMAX, 1].
        sc = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sc, src=qkv_s[n_start:n_start + PMAX, 0:1])
        # mixed_qkv_tile = accum * sc (broadcast along free=1 trivially).
        mixed_qkv_f = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=mixed_qkv_f, data=accum,
            op0=nl.multiply, operand0=sc,
            op1=nl.add, operand1=zb_p,
        )
        # Cast to bf16 for the conv path. (The original kernel reads conv_state
        # as bf16 and adds the new mixed_qkv col as bf16; we mirror that.)
        mixed_qkv_bf = nl.ndarray((PMAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(
            dst=mixed_qkv_bf, op=nl.copy, data=mixed_qkv_f, bias=zb_p, scale=1.0,
        )

        # === PHASE 1 (conv state update + depthwise conv + SiLU) ===
        # Same as deltanet_full.py phase 1, but with mixed_qkv_bf already
        # in SBUF (saves a DMA load from HBM).
        cs_bf = nl.ndarray((PMAX, 3), dtype=conv_state.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=cs_bf, src=conv_state[n_start:n_start + PMAX, 0:3])
        cs_f = nl.ndarray((PMAX, 3), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=cs_f, op=nl.copy, data=cs_bf, bias=zb_p, scale=1.0)

        # Cast mixed_qkv_bf back to f32 for conv math (same as original kernel).
        nq_f = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=nq_f, op=nl.copy, data=mixed_qkv_bf, bias=zb_p, scale=1.0)

        conv_in = nl.ndarray((PMAX, CONV_KERNEL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=conv_in[0:PMAX, 0:3], src=cs_f)
        nisa.tensor_copy(dst=conv_in[0:PMAX, 3:4], src=nq_f)

        ncs = nl.ndarray((PMAX, 3), dtype=conv_state.dtype, buffer=nl.sbuf)
        nisa.activation(dst=ncs, op=nl.copy, data=conv_in[0:PMAX, 1:4], bias=zb_p, scale=1.0)
        nisa.dma_copy(dst=new_conv_state[n_start:n_start + PMAX, 0:3], src=ncs)

        cw = nl.ndarray((PMAX, CONV_KERNEL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=cw, src=conv_weight[n_start:n_start + PMAX, 0:CONV_KERNEL],
        )
        prod = nl.ndarray((PMAX, CONV_KERNEL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=conv_in, data2=cw, op=nl.multiply)
        conv_sum = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=prod, op=nl.copy, data=prod, bias=zb_p, scale=1.0,
            reduce_op=nl.add, reduce_res=conv_sum,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )
        cb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cb, src=conv_bias[n_start:n_start + PMAX])
        act = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=act, op=nl.silu, data=conv_sum, bias=cb, scale=1.0)
        nisa.tensor_copy(dst=qkv_act[0:PMAX, n_idx:n_idx + 1], src=act)

    # =========================================================================
    # PHASE 0c — in_proj_z: z = x @ z_w.T -> z_sbuf [V_HEADS, V_DIM] bf16
    # z_w_T_i8 is [HIDDEN, 1536]. We produce z directly in V_HEADS-partitioned
    # layout (12 v-heads × 128 V_DIM fits one tile). Each of NUM_Z_TILES=12
    # N-tiles produces a [PMAX=128, 1] column — we then transpose each into
    # one row of z_sbuf[h, :].
    # =========================================================================
    for n_idx in nl.affine_range(NUM_Z_TILES):
        n_start = n_idx * PMAX
        accum_z = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.psum)
        for k_idx in nl.affine_range(NUM_X_TILES):
            k_start = k_idx * PMAX
            w_i8 = nl.ndarray((PMAX, PMAX), dtype=nl.int8, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=w_i8,
                src=z_w_T_i8[k_start:k_start + PMAX, n_start:n_start + PMAX],
            )
            w_fp8 = w_i8.view(nl.float8_e4m3)
            nisa.nc_matmul(
                dst=accum_z,
                stationary=w_fp8,
                moving=x_tiles[0:PMAX, k_idx:k_idx + 1],
            )

        sc = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sc, src=z_s[n_start:n_start + PMAX, 0:1])
        z_f = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=z_f, data=accum_z,
            op0=nl.multiply, operand0=sc,
            op1=nl.add, operand1=zb_p,
        )
        # z layout: NUM_Z_TILES=12 corresponds 1:1 with V_HEADS=12 (each head
        # has V_DIM=PMAX=128 channels). Transpose [PMAX, 1] -> [1, PMAX]
        # and write directly to z_hbm row n_idx (no V_HEADS-partitioned SBUF
        # buffer — see comment on z_hbm declaration).
        z_row_p = nl.ndarray((1, PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=z_row_p, data=z_f)
        z_row_bf = nl.ndarray((1, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        zb_1 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zb_1, value=0.0)
        nisa.activation(
            dst=z_row_bf, op=nl.copy, data=z_row_p, bias=zb_1, scale=1.0,
        )
        nisa.dma_copy(
            dst=z_hbm[n_idx:n_idx + 1, 0:V_DIM], src=z_row_bf,
        )

    # =========================================================================
    # PHASE 0d — in_proj_a, in_proj_b: a_out, b_out = x @ a/b_w.T
    # Both produce [V_HEADS=12, 1] f32. Single N-tile each (12 < PMAX), but
    # we need to allocate the matmul stationary as [PMAX, PMAX] so we use
    # the [PMAX, V_HEADS] partial slice.
    # =========================================================================
    # Allocate a_full / b_full with PMAX partitions (same as accum_a/b) so
    # scalar_tensor_tensor's base-partition equality check passes. Only
    # the [0:V_HEADS, :] slice carries data; the rest is unused.
    a_full = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    b_full = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)

    accum_a = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.psum)
    accum_b = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.psum)
    for k_idx in nl.affine_range(NUM_X_TILES):
        k_start = k_idx * PMAX
        wa_i8 = nl.ndarray((PMAX, PMAX), dtype=nl.int8, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=wa_i8[0:PMAX, 0:V_HEADS],
            src=a_w_T_i8[k_start:k_start + PMAX, 0:V_HEADS],
        )
        wa_fp8 = wa_i8.view(nl.float8_e4m3)
        nisa.nc_matmul(
            dst=accum_a[0:V_HEADS, 0:1],
            stationary=wa_fp8[0:PMAX, 0:V_HEADS],
            moving=x_tiles[0:PMAX, k_idx:k_idx + 1],
        )

        wb_i8 = nl.ndarray((PMAX, PMAX), dtype=nl.int8, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=wb_i8[0:PMAX, 0:V_HEADS],
            src=b_w_T_i8[k_start:k_start + PMAX, 0:V_HEADS],
        )
        wb_fp8 = wb_i8.view(nl.float8_e4m3)
        nisa.nc_matmul(
            dst=accum_b[0:V_HEADS, 0:1],
            stationary=wb_fp8[0:PMAX, 0:V_HEADS],
            moving=x_tiles[0:PMAX, k_idx:k_idx + 1],
        )

    # Scales also use PMAX partitions so all SB ops in the multiply share a
    # base partition.
    a_sc = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_sc[0:V_HEADS, 0:1], src=a_s[0:V_HEADS, 0:1])
    b_sc = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_sc[0:V_HEADS, 0:1], src=b_s[0:V_HEADS, 0:1])

    zb_v = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zb_v, value=0.0)
    nisa.scalar_tensor_tensor(
        dst=a_full[0:V_HEADS, 0:1], data=accum_a[0:V_HEADS, 0:1],
        op0=nl.multiply, operand0=a_sc[0:V_HEADS, 0:1],
        op1=nl.add, operand1=zb_v[0:V_HEADS, 0:1],
    )
    nisa.scalar_tensor_tensor(
        dst=b_full[0:V_HEADS, 0:1], data=accum_b[0:V_HEADS, 0:1],
        op0=nl.multiply, operand0=b_sc[0:V_HEADS, 0:1],
        op1=nl.add, operand1=zb_v[0:V_HEADS, 0:1],
    )

    # =========================================================================
    # PHASE 2 helpers (same as deltanet_full)
    # =========================================================================
    eps_tile = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_tile, value=RMS_EPS)
    zb1 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zb1, value=0.0)
    z_kv = nl.ndarray((K_DIM, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=z_kv, value=0.0)
    z_v1 = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=z_v1, value=0.0)
    ones_k = nl.ndarray((1, K_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_k, value=1.0)
    ones_v = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_v, value=1.0)

    nw_row = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nw_row, src=norm_weight[0:V_DIM])

    # silu(z): DMA z back from HBM in one shot, apply silu, DMA out for
    # later per-head pickoff.
    z_sbuf = nl.ndarray((V_HEADS, V_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=z_sbuf, src=z_hbm[0:V_HEADS, 0:V_DIM])
    zb_v_row = nl.ndarray((V_HEADS, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zb_v_row, value=0.0)
    silu_z_sbuf = nl.ndarray((V_HEADS, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=silu_z_sbuf, op=nl.silu, data=z_sbuf, bias=zb_v_row, scale=1.0,
    )
    nisa.dma_copy(dst=silu_z_hbm[0:V_HEADS, 0:V_DIM], src=silu_z_sbuf)

    # gate computation. a_full / b_full are PMAX-partitioned (see above),
    # so dt_full / Al_full / etc. allocate same way for base-partition match.
    dt_full = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=dt_full[0:V_HEADS, 0:1], src=dt_bias[0:V_HEADS])
    Al_full = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Al_full[0:V_HEADS, 0:1], src=A_log[0:V_HEADS])
    zb_p1 = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zb_p1, value=0.0)

    sp_full = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=sp_full[0:V_HEADS, 0:1], op=nl.softplus,
        data=a_full[0:V_HEADS, 0:1], bias=dt_full[0:V_HEADS, 0:1], scale=1.0,
    )
    expA_full = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=expA_full[0:V_HEADS, 0:1], op=nl.exp,
        data=Al_full[0:V_HEADS, 0:1], bias=zb_p1[0:V_HEADS, 0:1], scale=1.0,
    )
    pos_full = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=pos_full[0:V_HEADS, 0:1],
        data1=sp_full[0:V_HEADS, 0:1], data2=expA_full[0:V_HEADS, 0:1],
        op=nl.multiply,
    )
    exp_g_full = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=exp_g_full[0:V_HEADS, 0:1], op=nl.exp,
        data=pos_full[0:V_HEADS, 0:1], bias=zb_p1[0:V_HEADS, 0:1], scale=-1.0,
    )
    nisa.dma_copy(dst=exp_g_hbm[0:V_HEADS, 0:1], src=exp_g_full[0:V_HEADS, 0:1])
    beta_full = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=beta_full[0:V_HEADS, 0:1], op=nl.sigmoid,
        data=b_full[0:V_HEADS, 0:1], bias=zb_p1[0:V_HEADS, 0:1], scale=1.0,
    )
    nisa.dma_copy(dst=beta_hbm[0:V_HEADS, 0:1], src=beta_full[0:V_HEADS, 0:1])

    # =========================================================================
    # PHASE 2+3 — outer over K_HEADS, inner over HEAD_GROUP.
    # Identical to deltanet_full body — just operates on the qkv_act tiles
    # we computed above instead of those from HBM-loaded mixed_qkv.
    # =========================================================================
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

            exp_g = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=exp_g, src=exp_g_hbm[h:h + 1, 0:1])
            beta_s = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=beta_s, src=beta_hbm[h:h + 1, 0:1])

            s = nl.ndarray((K_DIM, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s, src=state[h * K_DIM:(h + 1) * K_DIM, 0:V_DIM])

            decay_p = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=decay_p, stationary=ones_k, moving=exp_g)
            decay_vec = nl.ndarray((K_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=decay_vec, src=decay_p)

            s_dec = nl.ndarray((K_DIM, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=s_dec, data=s,
                op0=nl.multiply, operand0=decay_vec,
                op1=nl.add, operand1=z_kv,
            )

            kv_p = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=kv_p, stationary=s_dec, moving=k_colN)
            kv_mem = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=kv_mem, src=kv_p)

            v_slot = 2 * K_HEADS + h
            v_col_in = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=v_col_in, src=qkv_act[0:V_DIM, v_slot:v_slot + 1])

            diff = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=diff, data1=v_col_in, data2=kv_mem, op=nl.subtract)

            beta_p = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=beta_p, stationary=ones_v, moving=beta_s)
            beta_vec = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=beta_vec, src=beta_p)

            delta_col = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=delta_col, data=diff,
                op0=nl.multiply, operand0=beta_vec,
                op1=nl.add, operand1=z_v1,
            )

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

            nisa.dma_copy(dst=new_state[h * K_DIM:(h + 1) * K_DIM, 0:V_DIM], src=s_new)

            attn_p = nl.ndarray((V_DIM, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=attn_p, stationary=s_new, moving=q_colS)
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
            zv_row = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=zv_row, value=0.0)
            nisa.scalar_tensor_tensor(
                dst=attn_norm, data=attn_row,
                op0=nl.multiply, operand0=rms_inv,
                op1=nl.add, operand1=zv_row,
            )
            weighted = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=weighted, data1=attn_norm, data2=nw_row, op=nl.multiply)

            sz = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=sz, src=silu_z_hbm[h:h + 1, 0:V_DIM])

            gated_f = nl.ndarray((1, V_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=gated_f, data1=weighted, data2=sz, op=nl.multiply)
            gated_bf = nl.ndarray((1, V_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=gated_bf, op=nl.copy, data=gated_f, bias=zb1, scale=1.0)
            nisa.dma_copy(dst=output[h:h + 1, 0:V_DIM], src=gated_bf)

    return new_state, new_conv_state, output
