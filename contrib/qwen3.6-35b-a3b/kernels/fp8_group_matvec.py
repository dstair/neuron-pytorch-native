"""
FP8 grouped matrix-vector kernel for MoE expert GEMMs (decode, BS-token-slot).

Computes, for each of G groups independently:
    out[g, :] = ( w_fp8[g] @ x[g] ) * scale[g]        # [OUT] = [OUT,IN]·[IN]
where w_fp8[g] is one expert's FP8 (legacy e4m3) weight matrix and scale[g] is
its per-output-channel dequant scale. G = T*TOP_K gathered (token,slot) pairs.

Why a kernel (not in-graph dequant): neuronx-cc rejects an in-graph
`int8.view(float8_e4m3fn).float()` (f8e4m3fn->f32 bitcast-convert COMPILATION
FAILED). The FP8 must stay behind the @nki.jit boundary, where nc_matmul
natively consumes an FP8 *stationary* operand and the per-channel scale is
applied on the PSUM->SBUF copy — so the dequant is FUSED into the matmul, never
a separate GPSIMD pass (the 27B's 98%-GPSIMD regression, see
[[project_qwen36_fp8_kernel_profile]]).

Weight bytes are passed as int8 (the e4m3 bytes) to dodge the HLO F8E4M3FN
verifier; bit-reinterpreted in SBUF as nl.float8_e4m3 (legacy — bitwise
compatible with e4m3fn for normal/trained values).

Layout per group g:
    x[g]      : [IN]            bf16   (one token's activation)
    w_i8[g]   : [OUT, IN]       int8   (fp8 e4m3 bytes, [out,in] order)
    scale[g]  : [OUT]           f32    (per-output-channel)
  -> out[g]   : [OUT]           f32

nc_matmul wants stationary [K, M] and moving [K, N] with K on the partition
(contraction) dim. Here K=IN (contraction), M=OUT, N=1 (single token). We pass
the weight as w_i8[g] viewed [OUT, IN] then use IN on partition by transposing —
but to avoid a transpose per group we DMA the weight tile with IN on partition
directly (the caller provides w pre-transposed to [G, IN, OUT]).
"""
import nki
import nki.isa as nisa
import nki.language as nl

PMAX = 128          # partition / contraction tile
FSTAT = 128         # stationary free (OUT) tile
RMS = 1e-6


def _div_ceil(a, b):
    return (a + b - 1) // b


@nki.jit
def nki_fp8_group_matvec(
    x,          # [G, IN]        bf16   gathered activations (one token/group)
    w_i8_T,     # [G, IN, OUT]   int8   fp8 e4m3 bytes, PRE-TRANSPOSED to [IN,OUT]
    scale,      # [G, OUT, 1]    f32    per-output-channel dequant scale (column)
):
    """out[G, OUT] f32 = (x[g] @ w[g]) * scale[g] per group, w dequantized."""
    G, IN = x.shape
    G2, IN2, OUT = w_i8_T.shape
    out = nl.ndarray((G, OUT), dtype=nl.float32, buffer=nl.shared_hbm)

    num_k = _div_ceil(IN, PMAX)
    num_n = _div_ceil(OUT, FSTAT)

    for g in nl.affine_range(G):
        # Load x[g] and transpose each K-tile to [K_tile, 1] (contraction on part).
        xT = nl.ndarray((PMAX, num_k), dtype=x.dtype, buffer=nl.sbuf)
        for kt in nl.affine_range(num_k):
            ks = kt * PMAX
            ksz = min(PMAX, IN - ks)
            xr = nl.ndarray((1, PMAX), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xr[0:1, 0:ksz], src=x[g:g + 1, ks:ks + ksz])
            xp = nl.ndarray((PMAX, 1), dtype=x.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=xp[0:ksz, 0:1], data=xr[0:1, 0:ksz])
            nisa.tensor_copy(dst=xT[0:ksz, kt:kt + 1], src=xp[0:ksz, 0:1])

        for nt in nl.affine_range(num_n):
            ns = nt * FSTAT
            nsz = min(FSTAT, OUT - ns)
            accum = nl.ndarray((FSTAT, 1), dtype=nl.float32, buffer=nl.psum)
            for kt in nl.affine_range(num_k):
                ks = kt * PMAX
                ksz = min(PMAX, IN - ks)
                # weight tile [K_tile, N_tile] from pre-transposed [IN, OUT]
                w_i8 = nl.ndarray((PMAX, FSTAT), dtype=nl.int8, buffer=nl.sbuf)
                nisa.dma_copy(dst=w_i8[0:ksz, 0:nsz],
                              src=w_i8_T[g, ks:ks + ksz, ns:ns + nsz])
                w_f8 = w_i8.view(nl.float8_e4m3)          # bit-reinterpret (no convert)
                # nc_matmul: dst[N,1] = stationary[K,N].T @ moving[K,1], FP8 stationary
                nisa.nc_matmul(dst=accum[0:nsz, 0:1],
                               stationary=w_f8[0:ksz, 0:nsz],
                               moving=xT[0:ksz, kt:kt + 1])
            # dequant: apply per-output-channel scale on the PSUM->SBUF copy
            sc = nl.ndarray((FSTAT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=sc[0:nsz, 0:1], src=scale[g, ns:ns + nsz, 0:1])
            zb = nl.ndarray((FSTAT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=zb, value=0.0)
            res = nl.ndarray((FSTAT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=res[0:nsz, 0:1], data=accum[0:nsz, 0:1],
                op0=nl.multiply, operand0=sc[0:nsz, 0:1],
                op1=nl.add, operand1=zb[0:nsz, 0:1])
            # write [N_tile,1] -> out[g, ns:ns+nsz] via transpose to a row
            rp = nl.ndarray((1, FSTAT), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=rp[0:1, 0:nsz], data=res[0:nsz, 0:1])
            rr = nl.ndarray((1, FSTAT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=rr[0:1, 0:nsz], src=rp[0:1, 0:nsz])
            nisa.dma_copy(dst=out[g:g + 1, ns:ns + nsz], src=rr[0:1, 0:nsz])

    return out
