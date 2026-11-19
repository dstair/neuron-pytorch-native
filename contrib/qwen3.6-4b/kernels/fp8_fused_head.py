"""Fused Trn2 FP8 vocabulary projection, scale, and greedy argmax.

The kernel scans 512 vocabulary rows at a time.  Logits remain on chip; only
one int32 token ID per batch row is returned to HBM.
"""

import nki
import nki.isa as nisa
import nki.language as nl


TILE = 128
WIDTH = 512


def _broadcast_row(source, partitions, width):
    result = nl.ndarray((TILE, width), dtype=source.dtype, buffer=nl.sbuf)
    shuffle_mask = [0] * 32
    for start in range(0, partitions, 32):
        nisa.nc_stream_shuffle(
            src=source,
            dst=result[start : start + 32, :],
            shuffle_mask=shuffle_mask,
        )
    return result


@nki.jit
def nki_fp8_fused_head_argmax(
    hidden,       # [B, H] bf16
    weight_i8,    # [H, V] int8 containing legacy-compatible FP8 bytes
    scales,       # [1, V] f32
):
    batch, hidden_size = hidden.shape
    hidden_size_w, vocab = weight_i8.shape
    assert batch in (32, 64, 128)
    assert hidden_size == hidden_size_w
    assert hidden_size % TILE == 0
    assert vocab % WIDTH == 0

    hidden_tiles = hidden_size // TILE
    vocab_slabs = vocab // WIDTH
    output = nl.ndarray((batch, 1), dtype=nl.int32, buffer=nl.shared_hbm)

    hidden_sb = nl.ndarray(
        (TILE, hidden_tiles, batch), dtype=hidden.dtype, buffer=nl.sbuf
    )
    for hidden_tile in nl.affine_range(hidden_tiles):
        hidden_start = hidden_tile * TILE
        hidden_row = nl.ndarray(
            (TILE, TILE), dtype=hidden.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=hidden_row[0:batch, :],
            src=hidden[:, hidden_start : hidden_start + TILE],
        )
        hidden_psum = nl.ndarray(
            (TILE, batch), dtype=hidden.dtype, buffer=nl.psum
        )
        nisa.nc_transpose(
            dst=hidden_psum,
            data=hidden_row[0:batch, :],
        )
        nisa.tensor_copy(
            dst=hidden_sb[:, hidden_tile, :],
            src=hidden_psum,
        )

    best_max = nl.ndarray(
        (TILE, 1), dtype=hidden.dtype, buffer=nl.sbuf
    )
    best_encoded = nl.ndarray(
        (TILE, 1), dtype=nl.int32, buffer=nl.sbuf
    )
    nisa.memset(dst=best_max[0:batch, :], value=-1.0e30)
    nisa.memset(dst=best_encoded[0:batch, :], value=0)

    weight_slot = nl.ndarray(
        (TILE, WIDTH), dtype=nl.int8, buffer=nl.sbuf
    )
    for slab in nl.sequential_range(vocab_slabs):
        vocab_start = slab * WIDTH
        logits_psum = nl.ndarray(
            (TILE, WIDTH), dtype=nl.float32, buffer=nl.psum
        )
        for hidden_tile in nl.sequential_range(hidden_tiles):
            hidden_start = hidden_tile * TILE
            nisa.dma_copy(
                dst=weight_slot,
                src=weight_i8[
                    hidden_start : hidden_start + TILE,
                    vocab_start : vocab_start + WIDTH,
                ],
            )
            nisa.nc_matmul(
                dst=logits_psum[0:batch, :],
                stationary=hidden_sb[:, hidden_tile, :],
                moving=weight_slot.view(nl.float8_e4m3),
            )

        scale_row = nl.ndarray(
            (1, WIDTH), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=scale_row,
            src=scales[0:1, vocab_start : vocab_start + WIDTH],
        )
        scale = _broadcast_row(scale_row, batch, WIDTH)
        scaled = nl.ndarray(
            (TILE, WIDTH), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=scaled[0:batch, :],
            data1=logits_psum[0:batch, :],
            data2=scale[0:batch, :],
            op=nl.multiply,
        )

        # Match the existing head's f32 -> bf16 boundary before argmax.
        logits = nl.ndarray(
            (TILE, WIDTH), dtype=hidden.dtype, buffer=nl.sbuf
        )
        nisa.activation(
            dst=logits[0:batch, :],
            op=nl.copy,
            data=scaled[0:batch, :],
            scale=1.0,
        )
        slab_max = nl.ndarray(
            (TILE, 1), dtype=hidden.dtype, buffer=nl.sbuf
        )
        nisa.tensor_reduce(
            dst=slab_max[0:batch, :],
            data=logits[0:batch, :],
            op=nl.maximum,
            axis=1,
        )
        slab_max_broadcast = nl.ndarray(
            (TILE, WIDTH), dtype=hidden.dtype, buffer=nl.sbuf
        )
        nisa.activation(
            dst=slab_max_broadcast[0:batch, :],
            op=nl.copy,
            data=logits[0:batch, :],
            bias=slab_max[0:batch, :],
            scale=0.0,
        )

        winner_mask = nl.ndarray(
            (TILE, WIDTH), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=winner_mask[0:batch, :],
            data1=logits[0:batch, :],
            data2=slab_max_broadcast[0:batch, :],
            op=nl.equal,
        )
        encoded_ids = nl.ndarray(
            (TILE, WIDTH), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.iota(
            dst=encoded_ids,
            pattern=[[-1, WIDTH]],
            offset=vocab - vocab_start,
            channel_multiplier=0,
        )
        encoded_winners = nl.ndarray(
            (TILE, WIDTH), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=encoded_winners[0:batch, :],
            data1=winner_mask[0:batch, :],
            data2=encoded_ids[0:batch, :],
            op=nl.multiply,
        )
        slab_encoded = nl.ndarray(
            (TILE, 1), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.tensor_reduce(
            dst=slab_encoded[0:batch, :],
            data=encoded_winners[0:batch, :],
            op=nl.maximum,
            axis=1,
        )

        new_max = nl.ndarray(
            (TILE, 1), dtype=hidden.dtype, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=new_max[0:batch, :],
            data1=best_max[0:batch, :],
            data2=slab_max[0:batch, :],
            op=nl.maximum,
        )
        old_wins = nl.ndarray(
            (TILE, 1), dtype=nl.int32, buffer=nl.sbuf
        )
        new_wins = nl.ndarray(
            (TILE, 1), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=old_wins[0:batch, :],
            data1=best_max[0:batch, :],
            data2=new_max[0:batch, :],
            op=nl.equal,
        )
        nisa.tensor_tensor(
            dst=new_wins[0:batch, :],
            data1=slab_max[0:batch, :],
            data2=new_max[0:batch, :],
            op=nl.equal,
        )
        old_encoded = nl.ndarray(
            (TILE, 1), dtype=nl.int32, buffer=nl.sbuf
        )
        new_encoded = nl.ndarray(
            (TILE, 1), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=old_encoded[0:batch, :],
            data1=best_encoded[0:batch, :],
            data2=old_wins[0:batch, :],
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=new_encoded[0:batch, :],
            data1=slab_encoded[0:batch, :],
            data2=new_wins[0:batch, :],
            op=nl.multiply,
        )
        merged_encoded = nl.ndarray(
            (TILE, 1), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=merged_encoded[0:batch, :],
            data1=old_encoded[0:batch, :],
            data2=new_encoded[0:batch, :],
            op=nl.maximum,
        )
        nisa.tensor_copy(
            dst=best_max[0:batch, :],
            src=new_max[0:batch, :],
        )
        nisa.tensor_copy(
            dst=best_encoded[0:batch, :],
            src=merged_encoded[0:batch, :],
        )

    token_id = nl.ndarray(
        (TILE, 1), dtype=nl.int32, buffer=nl.sbuf
    )
    nisa.tensor_scalar(
        dst=token_id[0:batch, :],
        data=best_encoded[0:batch, :],
        op0=nl.multiply,
        operand0=-1,
        op1=nl.add,
        operand1=vocab,
    )
    nisa.dma_copy(dst=output, src=token_id[0:batch, :])
    return output
