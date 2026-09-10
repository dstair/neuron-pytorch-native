"""Fused Trn2 FP8 gate/up/SwiGLU/down kernel for Qwen3.5-4B decode.

The TP=1 decode winner has B=32, H=2560, and I=9216.  Gate and up are packed
as [H, 2, I] so one 512-wide DMA supplies both projections.  The complete
SwiGLU intermediate remains in SBUF and feeds the down projection directly;
only the final [B, H] result is written to HBM.
"""

import nki
import nki.isa as nisa
import nki.language as nl


TILE = 128
WIDTH = 512


def _broadcast_row(source, partitions, width):
    """Broadcast source[0, :] to `partitions` SBUF partitions."""
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
def nki_fp8_fused_mlp(
    hidden,          # [B, H] bf16
    gate_up_i8,      # [H, 2, I] int8 containing legacy-compatible FP8 bytes
    gate_up_scales,  # [2, I] f32, one scale per output channel
    down_i8,         # [I, H] int8 containing legacy-compatible FP8 bytes
    down_scales,     # [1, H] f32, one scale per output channel
):
    batch, hidden_size = hidden.shape
    hidden_size_w, projections, intermediate = gate_up_i8.shape
    intermediate_d, hidden_size_d = down_i8.shape
    assert batch in (32, 64, 128)
    assert hidden_size == hidden_size_w == hidden_size_d
    assert intermediate == intermediate_d
    assert projections == 2
    assert hidden_size % WIDTH == 0
    assert intermediate % WIDTH == 0

    hidden_tiles = hidden_size // TILE
    intermediate_tiles = intermediate // TILE
    intermediate_slabs = intermediate // WIDTH
    hidden_slabs = hidden_size // WIDTH

    output = nl.ndarray(
        (batch, hidden_size), dtype=hidden.dtype, buffer=nl.shared_hbm
    )

    # Transpose the activation once.  Every gate/up output slab reuses it.
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

    # [I, B] in tile-major form.  This is about 576 KiB at B=32.
    activated = nl.ndarray(
        (TILE, intermediate_tiles, batch),
        dtype=hidden.dtype,
        buffer=nl.sbuf,
    )
    zero_bias = nl.ndarray(
        (TILE, 1), dtype=hidden.dtype, buffer=nl.sbuf
    )
    nisa.memset(dst=zero_bias, value=0.0)

    gate_up_slot = nl.ndarray(
        (TILE, 2, WIDTH), dtype=nl.int8, buffer=nl.sbuf
    )

    for slab in nl.affine_range(intermediate_slabs):
        output_start = slab * WIDTH
        gate_psum = nl.ndarray(
            (TILE, WIDTH), dtype=nl.float32, buffer=nl.psum
        )
        up_psum = nl.ndarray(
            (TILE, WIDTH), dtype=nl.float32, buffer=nl.psum
        )

        for hidden_tile in nl.sequential_range(hidden_tiles):
            hidden_start = hidden_tile * TILE
            nisa.dma_copy(
                dst=gate_up_slot,
                src=gate_up_i8[
                    hidden_start : hidden_start + TILE,
                    :,
                    output_start : output_start + WIDTH,
                ],
            )
            native_weight = gate_up_slot.view(nl.float8_e4m3)
            nisa.nc_matmul(
                dst=gate_psum[0:batch, :],
                stationary=hidden_sb[:, hidden_tile, :],
                moving=native_weight[:, 0, :],
            )
            nisa.nc_matmul(
                dst=up_psum[0:batch, :],
                stationary=hidden_sb[:, hidden_tile, :],
                moving=native_weight[:, 1, :],
            )

        gate_scale_row = nl.ndarray(
            (1, WIDTH), dtype=nl.float32, buffer=nl.sbuf
        )
        up_scale_row = nl.ndarray(
            (1, WIDTH), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=gate_scale_row,
            src=gate_up_scales[
                0:1, output_start : output_start + WIDTH
            ],
        )
        nisa.dma_copy(
            dst=up_scale_row,
            src=gate_up_scales[
                1:2, output_start : output_start + WIDTH
            ],
        )
        gate_scale = _broadcast_row(gate_scale_row, batch, WIDTH)
        up_scale = _broadcast_row(up_scale_row, batch, WIDTH)

        gate_scaled = nl.ndarray(
            (TILE, WIDTH), dtype=nl.float32, buffer=nl.sbuf
        )
        up_scaled = nl.ndarray(
            (TILE, WIDTH), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=gate_scaled[0:batch, :],
            data1=gate_psum[0:batch, :],
            data2=gate_scale[0:batch, :],
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=up_scaled[0:batch, :],
            data1=up_psum[0:batch, :],
            data2=up_scale[0:batch, :],
            op=nl.multiply,
        )

        # Match the existing three-op path's BF16 boundaries.
        gate_bf16 = nl.ndarray(
            (TILE, WIDTH), dtype=hidden.dtype, buffer=nl.sbuf
        )
        up_bf16 = nl.ndarray(
            (TILE, WIDTH), dtype=hidden.dtype, buffer=nl.sbuf
        )
        nisa.activation(
            dst=gate_bf16[0:batch, :],
            op=nl.copy,
            data=gate_scaled[0:batch, :],
            scale=1.0,
        )
        nisa.activation(
            dst=up_bf16[0:batch, :],
            op=nl.copy,
            data=up_scaled[0:batch, :],
            scale=1.0,
        )
        silu_gate = nl.ndarray(
            (TILE, WIDTH), dtype=hidden.dtype, buffer=nl.sbuf
        )
        nisa.activation(
            dst=silu_gate[0:batch, :],
            op=nl.silu,
            data=gate_bf16[0:batch, :],
            bias=zero_bias[0:batch, :],
            scale=1.0,
        )
        product = nl.ndarray(
            (TILE, WIDTH), dtype=hidden.dtype, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=product[0:batch, :],
            data1=silu_gate[0:batch, :],
            data2=up_bf16[0:batch, :],
            op=nl.multiply,
        )

        for slab_tile in nl.affine_range(WIDTH // TILE):
            tile_start = slab_tile * TILE
            intermediate_tile = slab * (WIDTH // TILE) + slab_tile
            activated_psum = nl.ndarray(
                (TILE, batch), dtype=hidden.dtype, buffer=nl.psum
            )
            nisa.nc_transpose(
                dst=activated_psum,
                data=product[
                    0:batch, tile_start : tile_start + TILE
                ],
            )
            nisa.tensor_copy(
                dst=activated[:, intermediate_tile, :],
                src=activated_psum,
            )

    down_slot = nl.ndarray(
        (TILE, WIDTH), dtype=nl.int8, buffer=nl.sbuf
    )
    for slab in nl.affine_range(hidden_slabs):
        output_start = slab * WIDTH
        down_psum = nl.ndarray(
            (TILE, WIDTH), dtype=nl.float32, buffer=nl.psum
        )
        for intermediate_tile in nl.sequential_range(intermediate_tiles):
            intermediate_start = intermediate_tile * TILE
            nisa.dma_copy(
                dst=down_slot,
                src=down_i8[
                    intermediate_start : intermediate_start + TILE,
                    output_start : output_start + WIDTH,
                ],
            )
            nisa.nc_matmul(
                dst=down_psum[0:batch, :],
                stationary=activated[:, intermediate_tile, :],
                moving=down_slot.view(nl.float8_e4m3),
            )

        down_scale_row = nl.ndarray(
            (1, WIDTH), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=down_scale_row,
            src=down_scales[
                0:1, output_start : output_start + WIDTH
            ],
        )
        down_scale = _broadcast_row(down_scale_row, batch, WIDTH)
        down_scaled = nl.ndarray(
            (TILE, WIDTH), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=down_scaled[0:batch, :],
            data1=down_psum[0:batch, :],
            data2=down_scale[0:batch, :],
            op=nl.multiply,
        )
        down_bf16 = nl.ndarray(
            (TILE, WIDTH), dtype=hidden.dtype, buffer=nl.sbuf
        )
        nisa.activation(
            dst=down_bf16[0:batch, :],
            op=nl.copy,
            data=down_scaled[0:batch, :],
            scale=1.0,
        )
        nisa.dma_copy(
            dst=output[:, output_start : output_start + WIDTH],
            src=down_bf16[0:batch, :],
        )

    return output
