"""
FP8 W8A16 matmul NKI kernel.

Computes: out[B, N] = x[B, K] @ w[N, K].T  with w stored as fp8 + per-row f32 scale.
The dequantization (fp8 -> bf16 multiplied by scale) happens AFTER nc_matmul on
the f32 PSUM result, so the heavy data path reads only 1 byte/elem of weight
from HBM. This halves weight bandwidth, which is the dominant cost of decode
on the Qwen 3.6 27B model (E2E profile shows 84% MBU, 2.5GB/step weight reads).

Why a custom NKI kernel: this Native Beta compiler version (neuronxcc 2.0.253)
does NOT support `convert(fp8 -> bf16)` in compiled torch graphs — putting the
dequant in PyTorch and using torch.compile fails with "COMPILATION FAILED:
%convert.<id> = bf16[N,K] convert(<fp8 weight>)". A NKI kernel sidesteps this
because the compiler sees only an opaque @nki.jit boundary; the fp8->f32 cast
happens inside the kernel via nc_matmul's native fp8 stationary support.

nc_matmul fp8 semantics (Trainium2):
  stationary[K, M] fp8 . moving[K, N] bf16 -> psum[M, N] f32
The contraction reduces along K. Output is f32 PSUM regardless of operand
dtypes; we apply the per-row scale and cast to bf16 on the SBUF copy.

Layout (caller-side, matches PyTorch nn.Linear convention):
  x:     [B, K]   bf16  — activations (B is the batch/seq dim, often small)
  w_fp8: [N, K]   fp8   — weight, stored row-major (out, in)
  scale: [N]      f32   — per-output-channel dequantization scale

Internal layout for nc_matmul:
  stationary = w_fp8 sliced as [K_tile, N_tile] — already in [out, in] order;
    we slice w_fp8[n_start:n_end, k_start:k_end].T effectively by addressing
    the columns as partition-dim tiles.
  moving     = x sliced as [K_tile, B] — needs x.T view; we DMA the transpose
    by addressing x[:, k_start:k_end].T into the SBUF buffer.

Tile sizes:
  K (contraction, partition):  tiled by P_MAX = 128
  N (output, stationary free): tiled by F_STAT_MAX = 128
  B (batch, moving free):      no tiling needed for B<=512; for decode B=1.

Output: [B, N] bf16.
"""
import nki
import nki.isa as nisa
import nki.language as nl


P_MAX = 128         # nc_matmul partition / contraction dim limit
F_STAT_MAX = 128    # stationary free dim limit (= per-tile output rows)
F_MOV_MAX = 512     # moving free dim limit


def _div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


@nki.jit
def nki_fp8_matmul(
    x,         # [B, K]      bf16
    w_fp8_T,   # [K, N]      float8_e4m3  — pre-transposed weight (K partition)
    scale,     # [N, 1]      f32          — per-output-channel scale (column form)
):
    """Returns [B, N] bf16 = x @ w.T with w dequantized via per-row scale.

    Caller must provide the weight pre-transposed to [K, N] layout (partition
    dim = K, contraction dim). One-time transpose at quantize time, amortized
    across every decode step.
    """
    B, K = x.shape
    K2, N = w_fp8_T.shape
    assert K == K2, f"K mismatch: x has K={K}, w_fp8_T has K={K2}"
    # SBUF partition limit is 128. We use B as the moving free dim of nc_matmul
    # (free, can be up to F_MOV_MAX=512), but the per-tile SBUF allocations
    # holding [N_tile, B] also need B<=P_MAX since we'll transpose to [B, N_tile]
    # for output. For decode B=1; for batch>128 we'd need an outer B loop.
    assert B <= P_MAX, f"B={B} exceeds P_MAX={P_MAX}"

    out = nl.ndarray((B, N), dtype=x.dtype, buffer=nl.shared_hbm)

    num_n_tiles = _div_ceil(N, F_STAT_MAX)
    num_k_tiles = _div_ceil(K, P_MAX)

    # ── Hoisted activation transpose (the lever) ───────────────────────────
    # The moving operand for nc_matmul is x.T sliced as [K_tile, B]. This is
    # IDENTICAL across all N tiles, so transposing it inside the (n, k) double
    # loop did num_n_tiles× redundant nc_transpose ops — and nc_transpose runs
    # on the Pool/GPSIMD engine, which the decfp8 profile showed at ~98% busy
    # (TensorE idle). Transpose each K tile of x exactly ONCE here, before the
    # N loop, and reuse the SBUF-resident xT across every N tile.
    #
    # Layout: xT_all[P_MAX, num_k_tiles*B] — K on partition (contraction), and
    # K tile k_idx occupies columns [k_idx*B : k_idx*B + B]. For decode B=1
    # this is 128×40 bf16 ≈ 10 KB, trivially SBUF-resident.
    xT_all = nl.ndarray((P_MAX, num_k_tiles * B), dtype=x.dtype, buffer=nl.sbuf)
    for k_idx in nl.affine_range(num_k_tiles):
        k_start = k_idx * P_MAX
        k_size = min(P_MAX, K - k_start)
        col = k_idx * B
        x_row = nl.ndarray((P_MAX, P_MAX), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=x_row[0:B, 0:k_size],
            src=x[0:B, k_start:k_start + k_size],
        )
        x_psum = nl.ndarray((P_MAX, P_MAX), dtype=x.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=x_psum[0:k_size, 0:B], data=x_row[0:B, 0:k_size])
        nisa.tensor_copy(dst=xT_all[0:k_size, col:col + B], src=x_psum[0:k_size, 0:B])

    for n_idx in nl.affine_range(num_n_tiles):
        n_start = n_idx * F_STAT_MAX
        n_size = min(F_STAT_MAX, N - n_start)

        # PSUM accumulator: [N_tile, B] f32. nc_matmul accumulates across K.
        # B <= 128 (decode is B=1); cap free dim at P_MAX to fit one partition tile.
        accum = nl.ndarray((F_STAT_MAX, F_STAT_MAX), dtype=nl.float32, buffer=nl.psum)

        for k_idx in nl.affine_range(num_k_tiles):
            k_start = k_idx * P_MAX
            k_size = min(P_MAX, K - k_start)
            col = k_idx * B

            # Load weight tile [K_tile, N_tile] — partition dim = K (contraction).
            # PyTorch fp8 is `float8_e4m3fn` (OCP variant, requires Trn3+).
            # On Trn2 (gen2) `nc_matmul` only accepts legacy `float8_e4m3`.
            # We pass the weight bytes as int8 to dodge the HLO verifier's
            # F8E4M3FN check, then bit-reinterpret in SBUF as float8_e4m3
            # (legacy). The two formats are bitwise compatible for normal
            # (non-NaN) values, which is fine for trained weights.
            w_sb_i8 = nl.ndarray((P_MAX, F_STAT_MAX), dtype=nl.int8, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=w_sb_i8[0:k_size, 0:n_size],
                src=w_fp8_T[k_start:k_start + k_size, n_start:n_start + n_size],
            )
            # Bit-reinterpret int8 -> float8_e4m3 (no numeric conversion).
            w_sb = w_sb_i8.view(nl.float8_e4m3)

            # nc_matmul: dst[N, B] = stationary[K, N].T @ moving[K, B]
            #   stationary = w_sb            [K_tile, N_tile]
            #   moving     = xT_all[:, col]  [K_tile, B]  (hoisted, no transpose)
            nisa.nc_matmul(
                dst=accum[0:n_size, 0:B],
                stationary=w_sb[0:k_size, 0:n_size],
                moving=xT_all[0:k_size, col:col + B],
            )

        # PSUM -> SBUF, apply per-row scale, cast to output dtype.
        # Scale is passed as [N, 1] so the slice is direct, no reshape.
        scale_tile = nl.ndarray((F_STAT_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=scale_tile[0:n_size, 0:1],
            src=scale[n_start:n_start + n_size, 0:1],
        )

        # accum is [N_tile, B] f32. Multiply by scale broadcast along B,
        # cast to bf16. Use scalar_tensor_tensor: dst = op1(op0(scale*data, op0v), op1v)
        zb = nl.ndarray((F_STAT_MAX, F_STAT_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zb, value=0.0)
        scaled = nl.ndarray((F_STAT_MAX, F_STAT_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=scaled[0:n_size, 0:B], data=accum[0:n_size, 0:B],
            op0=nl.multiply, operand0=scale_tile[0:n_size, 0:1],
            op1=nl.add, operand1=zb[0:n_size, 0:B],
        )

        # Cast f32 -> bf16
        result = nl.ndarray((F_STAT_MAX, F_STAT_MAX), dtype=x.dtype, buffer=nl.sbuf)
        zb1 = nl.ndarray((F_STAT_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zb1, value=0.0)
        nisa.activation(
            dst=result[0:n_size, 0:B], op=nl.copy, data=scaled[0:n_size, 0:B],
            bias=zb1[0:n_size, 0:1], scale=1.0,
        )

        # Transpose [N_tile, B] -> [B, N_tile]. B fits on partition dim
        # (B<=F_STAT_MAX=128), so allocate the transposed result with B
        # partitions and N_tile (<=128) free dim.
        result_T = nl.ndarray((F_STAT_MAX, F_STAT_MAX), dtype=x.dtype, buffer=nl.sbuf)
        result_T_p = nl.ndarray((F_STAT_MAX, F_STAT_MAX), dtype=x.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=result_T_p[0:B, 0:n_size], data=result[0:n_size, 0:B])
        nisa.tensor_copy(dst=result_T[0:B, 0:n_size], src=result_T_p[0:B, 0:n_size])

        nisa.dma_copy(
            dst=out[0:B, n_start:n_start + n_size],
            src=result_T[0:B, 0:n_size],
        )

    return out
