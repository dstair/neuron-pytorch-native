"""Dynamic-offset partial RoPE and KV-cache update for bucketed GQA prefill."""

import nki
import nki.isa as nisa
import nki.language as nl


TILE = 128
HEAD_DIM = 256
ROPE_DIM = 64
HALF_ROPE = ROPE_DIM // 2


def _to_f32(src, rows, cols):
    loaded = nl.ndarray((rows, cols), dtype=src.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=loaded, src=src)
    out = nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out, src=loaded)
    return out


def _rotate_tile(src, cos, sin, rows):
    """Apply rotate-half RoPE to [rows, 256], returning f32 SBUF."""
    src_f32 = _to_f32(src, rows, HEAD_DIM)
    out = nl.ndarray((rows, HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)

    neg_hi = nl.ndarray((rows, HALF_ROPE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=neg_hi,
        data=src_f32[:, HALF_ROPE:ROPE_DIM],
        op0=nl.multiply,
        operand0=-1.0,
    )
    rot_lo = nl.ndarray((rows, HALF_ROPE), dtype=nl.float32, buffer=nl.sbuf)
    rot_hi = nl.ndarray((rows, HALF_ROPE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=rot_lo,
        data1=src_f32[:, :HALF_ROPE],
        data2=cos[:, :HALF_ROPE],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=rot_hi,
        data1=src_f32[:, HALF_ROPE:ROPE_DIM],
        data2=cos[:, HALF_ROPE:ROPE_DIM],
        op=nl.multiply,
    )
    lo_sin = nl.ndarray((rows, HALF_ROPE), dtype=nl.float32, buffer=nl.sbuf)
    hi_sin = nl.ndarray((rows, HALF_ROPE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=lo_sin,
        data1=neg_hi,
        data2=sin[:, :HALF_ROPE],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=hi_sin,
        data1=src_f32[:, :HALF_ROPE],
        data2=sin[:, HALF_ROPE:ROPE_DIM],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=out[:, :HALF_ROPE],
        data1=rot_lo,
        data2=lo_sin,
        op=nl.add,
    )
    nisa.tensor_tensor(
        dst=out[:, HALF_ROPE:ROPE_DIM],
        data1=rot_hi,
        data2=hi_sin,
        op=nl.add,
    )
    nisa.tensor_copy(dst=out[:, ROPE_DIM:HEAD_DIM], src=src_f32[:, ROPE_DIM:HEAD_DIM])
    return out


@nki.jit
def nki_gqa_rope_kv_dynamic(
    query,
    key,
    value,
    rope_cos,
    rope_sin,
    kv_key,
    kv_value,
    q_base,
):
    """Rotate Q/K and write K/V at runtime-selected contiguous cache rows.

    Shapes:
      query: [B, Q_HEADS, CHUNK, 256] bf16
      key/value: [B, CHUNK, 256] bf16
      rope_cos/rope_sin: [KMAX, 64] f32
      kv_key/kv_value: [B*KMAX, 256] bf16, mutated in place
      q_base: [1, 1] int32
    """
    batch_size = query.shape[0]
    q_heads = query.shape[1]
    chunk = query.shape[2]
    kmax = kv_key.shape[0] // batch_size
    assert chunk % TILE == 0, "GQA dynamic RoPE/KV requires CHUNK divisible by 128"

    query_out = nl.ndarray(
        (batch_size, q_heads, chunk, HEAD_DIM),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )
    key_out = nl.ndarray(
        (batch_size, chunk, HEAD_DIM),
        dtype=key.dtype,
        buffer=nl.shared_hbm,
    )
    base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=base, src=q_base)

    for tile_idx in nl.sequential_range(chunk // TILE):
        row = tile_idx * TILE
        tile_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=tile_base,
            data=base,
            op0=nl.add,
            operand0=row,
        )

        cos = nl.ndarray((TILE, ROPE_DIM), dtype=nl.float32, buffer=nl.sbuf)
        sin = nl.ndarray((TILE, ROPE_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=cos,
            src=rope_cos.ap(
                [[ROPE_DIM, TILE], [1, ROPE_DIM]], scalar_offset=tile_base
            ),
        )
        nisa.dma_copy(
            dst=sin,
            src=rope_sin.ap(
                [[ROPE_DIM, TILE], [1, ROPE_DIM]], scalar_offset=tile_base
            ),
        )

        # Keep the batch base static. Combining an affine batch index with the
        # runtime row offset makes the driver conservatively bound the scalar
        # DMA against the full flattened address range and reject the NEFF.
        for batch in range(batch_size):
            cache_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=cache_base,
                data=tile_base,
                op0=nl.add,
                operand0=batch * kmax,
            )
            for head in nl.affine_range(q_heads):
                q_rot = _rotate_tile(
                    query[batch, head, row : row + TILE, :], cos, sin, TILE
                )
                nisa.dma_copy(
                    dst=query_out[batch, head, row : row + TILE, :],
                    src=q_rot,
                )

            key_rot = _rotate_tile(
                key[batch, row : row + TILE, :], cos, sin, TILE
            )
            key_store = nl.ndarray((TILE, HEAD_DIM), dtype=kv_key.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=key_store, src=key_rot)
            nisa.dma_copy(dst=key_out[batch, row : row + TILE, :], src=key_store)
            nisa.dma_copy(
                dst=kv_key.ap(
                    [[HEAD_DIM, TILE], [1, HEAD_DIM]], scalar_offset=cache_base
                ),
                src=key_store,
            )
            nisa.dma_copy(
                dst=kv_value.ap(
                    [[HEAD_DIM, TILE], [1, HEAD_DIM]], scalar_offset=cache_base
                ),
                src=value[batch, row : row + TILE, :],
            )

    return query_out, key_out, kv_key, kv_value
