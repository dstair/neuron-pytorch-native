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
    group_index=0,
    num_groups=1,
):
    """Rotate Q/K and write K/V at runtime-selected contiguous cache rows.

    Shapes:
      query: [B, Q_HEADS, CHUNK, 256] bf16
      key/value: [B, CHUNK, 256] bf16
      rope_cos/rope_sin: [KMAX, 64] f32
      kv_key/kv_value: [NUM_GROUPS*B*KMAX, 256] bf16, mutated in place
      q_base: [1, 1] int32
      group_index/num_groups: static ints selecting this layer's cache slab

    kv_key/kv_value are the WHOLE flattened cache for every layer, with this
    layer picked out by the static group_index -- the same convention as
    gqa35b::tail_stateful (gqa_tail_35b.py:107).

    They are RETURNED as well as mutated, which is what makes the write-back
    correct rather than lucky: NKI emits `input_output_aliases` only for inputs a
    kernel returns, and that map is what torch-neuronx turns into the
    `ctx.replace`/`commit_update`/`sync` that models the mutation
    (torch_neuronx/nki_hop.py:391-398). Without it `mutates_args` reaches only the
    PyTorch-level schema and the write survives or not depending on whether the
    backend happens to reorder it. Measured, kernels/tests/probe_kv_alias_f4.py:
    aliased 3/3 writes land where unaliased lands 0/3, at equal HBM and equal
    wall time. An aliased return is the same buffer, so it does NOT materialize
    the cache -- that cost applies to an unaliased output, which is what the
    2026-07 version of this comment was actually about.
    """
    batch_size = query.shape[0]
    q_heads = query.shape[1]
    chunk = query.shape[2]
    kmax = kv_key.shape[0] // (batch_size * num_groups)
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

        # Keep the batch and group bases static. Combining an affine batch index
        # with the runtime row offset makes the driver conservatively bound the
        # scalar DMA against the full flattened address range and reject the NEFF.
        for batch in range(batch_size):
            cache_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=cache_base,
                data=tile_base,
                op0=nl.add,
                operand0=(group_index * batch_size + batch) * kmax,
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

    # RETURN the caches. This is what declares the aliasing (see the docstring):
    # returning a mutated input is the only way to get NKI to report
    # input_output_aliases, and without that map the in-place write has no
    # representation in the emitted graph at all.
    #
    # This is NOT the "materialize both full caches" cost the pre-2026-08-05
    # comment here warned about. That cost is real for an *unaliased* output,
    # which allocates a fresh [B*KMAX, 256] buffer and copies into it. An aliased
    # output is by definition the same buffer as the input: no allocation, no
    # copy. probe_kv_alias_f4.py measures it at equal HBM and equal wall time.
    #
    # Callers should read the cache back through THESE returns rather than
    # through their own view of the buffer -- an in-graph read of a mutated
    # buffer is not ordered against the mutation, but a read of the op's own
    # return value is.
    return query_out, key_out, kv_key, kv_value
