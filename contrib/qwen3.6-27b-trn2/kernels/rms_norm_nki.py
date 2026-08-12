"""
Free-axis RMSNorm NKI kernel — kills the avoidable on-chip norm collective.

Production decode emits ~145-274 CollectiveCompute `_reduce` ops with
replica_groups [[0,1]], is_local=True. These are the compiler splitting the
mean(x^2) reduction over HIDDEN=5120 across the 2 physical cores of one logical
NeuronCore. They are AVOIDABLE: after each sublayer's TP all-reduce the hidden
state is fully replicated per rank, so the norm needs no cross-device or
cross-core communication.

This kernel forces the layout the compiler won't pick on its own: the 5120
hidden elements live on the FREE axis of a SINGLE partition, so sum-of-squares
is a free-axis reduce (reduce_op=add inside one activation) with no partition
split and therefore no collective.

Computes Qwen3.5 residual RMSNorm:
    out = x * rsqrt(mean(x^2) + eps) * (1 + weight)

Shapes (decode = single token):
    x:      [H]   any float dtype   hidden vector for one token
    weight: [H]   f32               norm weight (residual: 1+weight applied)
    out:    [H]   x.dtype

Free-axis reduce pattern mirrors the gated-norm in kernels/deltanet_full.py
(sum-of-squares -> fold mean+eps+rsqrt into one nl.rsqrt activation).
"""
import nki
import nki.isa as nisa
import nki.language as nl

RMS_EPS = 1e-6
# Free-axis tile width. Keeps each engine instruction within a safe free span
# and lets the reduce accumulate across tiles via reduce_cmd. 5120 % 512 == 0.
FTILE = 512


@nki.jit
def nki_rms_norm(x, weight):
    H = x.shape[0]
    out = nl.ndarray((H,), dtype=x.dtype, buffer=nl.shared_hbm)

    zb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zb, value=0.0)
    eps_tile = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_tile, value=RMS_EPS)

    # Load x onto a single partition, hidden on the free axis: [1, H].
    x_in = nl.ndarray((1, H), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_in, src=x[0:H])
    x_row = nl.ndarray((1, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=x_row, op=nl.copy, data=x_in, bias=zb, scale=1.0)

    ntiles = H // FTILE
    rem = H - ntiles * FTILE

    # Sum of squares over the free axis, accumulated across tiles.
    ssum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    sq = nl.ndarray((1, FTILE), dtype=nl.float32, buffer=nl.sbuf)
    for t in nl.affine_range(ntiles):
        cmd = nisa.reduce_cmd.reset_reduce if t == 0 else nisa.reduce_cmd.reduce
        nisa.activation(
            dst=sq, op=nl.square, data=x_row[0:1, t * FTILE:(t + 1) * FTILE],
            bias=zb, scale=1.0,
            reduce_op=nl.add, reduce_res=ssum, reduce_cmd=cmd,
        )
    if rem > 0:
        sqr = nl.ndarray((1, rem), dtype=nl.float32, buffer=nl.sbuf)
        cmd = nisa.reduce_cmd.reset_reduce if ntiles == 0 else nisa.reduce_cmd.reduce
        nisa.activation(
            dst=sqr, op=nl.square, data=x_row[0:1, ntiles * FTILE:H],
            bias=zb, scale=1.0,
            reduce_op=nl.add, reduce_res=ssum, reduce_cmd=cmd,
        )

    # rms_inv = rsqrt(ssum / H + eps)  — fold mean-divide + eps-add + rsqrt.
    rms_inv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rms_inv, op=nl.rsqrt, data=ssum, bias=eps_tile, scale=1.0 / H)

    # out = (x * rms_inv) * (1 + weight), tiled over the free axis.
    for t in nl.affine_range(ntiles):
        sl = slice(t * FTILE, (t + 1) * FTILE)
        _norm_tile(x_row, weight, rms_inv, zb, out, sl, FTILE, x.dtype)
    if rem > 0:
        sl = slice(ntiles * FTILE, H)
        _norm_tile(x_row, weight, rms_inv, zb, out, sl, rem, x.dtype)

    return out


def _norm_tile(x_row, weight, rms_inv, zb, out, sl, width, odtype):
    # normed = x * rms_inv  (per-partition scalar broadcast over free axis)
    normed = nl.ndarray((1, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=normed, op=nl.copy, data=x_row[0:1, sl], bias=zb, scale=rms_inv)

    # (1 + weight)
    w_in = nl.ndarray((1, width), dtype=weight.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_in, src=weight[sl])
    w_f = nl.ndarray((1, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=w_f, op=nl.copy, data=w_in, bias=zb, scale=1.0)
    w1 = nl.ndarray((1, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=w1, data=w_f, op0=nl.add, operand0=1.0)

    # out = normed * (1 + weight)
    out_f = nl.ndarray((1, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out_f, data1=normed, data2=w1, op=nl.multiply)
    out_o = nl.ndarray((1, width), dtype=odtype, buffer=nl.sbuf)
    nisa.activation(dst=out_o, op=nl.copy, data=out_f, bias=zb, scale=1.0)
    nisa.dma_copy(dst=out[sl], src=out_o)
