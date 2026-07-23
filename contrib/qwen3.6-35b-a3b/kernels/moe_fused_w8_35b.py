"""Static high-batch all-expert MoE with 128x128 block-W8 weights.

The router and shared expert remain in the surrounding PyTorch graph. This
kernel fuses the local routed experts:

    gate/up GEMM -> SwiGLU -> down GEMM -> affinity scale -> expert sum

It never materializes `[experts, batch, hidden]`. For each 128-token tile it
keeps one expert's gate/up/activated intermediate and the final routed
accumulator in SBUF. Weight bytes are converted tile-by-tile:

* FP8: official E4M3FN bytes are decoded tile-by-tile to BF16 because TensorE
  does not accept E4M3FN as an `nc_matmul` stationary dtype. Codes below 0x78
  use the legacy-E4M3 converter; the finite E4M3FN-only codes 0x78..0x7e are
  patched exactly.
* Dual FP8: exact E4M3FN values are represented by finite legacy-E4M3 base and
  residual planes. TensorE consumes both planes natively and accumulates them
  into one FP32 PSUM, avoiding tile-local vector decoding.
* INT8: int8 storage is numerically converted to BF16 in SBUF.
"""

import nki
import nki.isa as nisa
import nki.language as nl


TILE = 128
COALESCED_WIDTH = 512


def _load_weight_tile(source, indices, use_fp8):
    weight_i8 = nl.ndarray((TILE, TILE), dtype=nl.int8, buffer=nl.sbuf)
    nisa.dma_copy(dst=weight_i8, src=source[indices])
    if not use_fp8:
        weight_bf16 = nl.ndarray(
            (TILE, TILE), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        nisa.activation(
            dst=weight_bf16, op=nl.copy, data=weight_i8, scale=1.0
        )
        return weight_bf16

    magnitude_i8 = nl.ndarray(
        (TILE, TILE), dtype=nl.int8, buffer=nl.sbuf
    )
    nisa.tensor_scalar(
        dst=magnitude_i8,
        data=weight_i8,
        op0=nl.bitwise_and,
        operand0=0x7F,
    )
    safe_magnitude_i8 = nl.ndarray(
        (TILE, TILE), dtype=nl.int8, buffer=nl.sbuf
    )
    nisa.tensor_scalar(
        dst=safe_magnitude_i8,
        data=magnitude_i8,
        op0=nl.minimum,
        operand0=0x77,
    )
    sign_i8 = nl.ndarray((TILE, TILE), dtype=nl.int8, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=sign_i8,
        data=weight_i8,
        op0=nl.bitwise_and,
        operand0=-128,
    )
    safe_i8 = nl.ndarray((TILE, TILE), dtype=nl.int8, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=safe_i8,
        data1=safe_magnitude_i8,
        data2=sign_i8,
        op=nl.bitwise_or,
    )
    safe_bf16 = nl.ndarray(
        (TILE, TILE), dtype=nl.bfloat16, buffer=nl.sbuf
    )
    nisa.activation(
        dst=safe_bf16,
        op=nl.copy,
        data=safe_i8.view(nl.float8_e4m3),
        scale=1.0,
    )

    magnitude = nl.ndarray(
        (TILE, TILE), dtype=nl.bfloat16, buffer=nl.sbuf
    )
    nisa.activation(
        dst=magnitude, op=nl.copy, data=magnitude_i8, scale=1.0
    )
    high_mask = nl.ndarray(
        (TILE, TILE), dtype=nl.bfloat16, buffer=nl.sbuf
    )
    nisa.tensor_scalar(
        dst=high_mask,
        data=magnitude,
        op0=nl.greater_equal,
        operand0=0x78,
    )
    sign = nl.ndarray((TILE, TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=sign, op=nl.copy, data=sign_i8, scale=1.0)
    nisa.tensor_scalar(
        dst=sign,
        data=sign,
        op0=nl.multiply,
        operand0=1.0 / 64.0,
        op1=nl.add,
        operand1=1.0,
    )
    high_value = nl.ndarray(
        (TILE, TILE), dtype=nl.bfloat16, buffer=nl.sbuf
    )
    nisa.tensor_scalar(
        dst=high_value,
        data=magnitude,
        op0=nl.multiply,
        operand0=32.0,
        op1=nl.subtract,
        operand1=3584.0,
    )
    signed_high = nl.ndarray(
        (TILE, TILE), dtype=nl.bfloat16, buffer=nl.sbuf
    )
    nisa.tensor_tensor(
        dst=signed_high,
        data1=high_value,
        data2=sign,
        op=nl.multiply,
    )
    correction = nl.ndarray(
        (TILE, TILE), dtype=nl.bfloat16, buffer=nl.sbuf
    )
    nisa.tensor_tensor(
        dst=correction,
        data1=signed_high,
        data2=safe_bf16,
        op=nl.subtract,
    )
    nisa.tensor_tensor(
        dst=correction,
        data1=correction,
        data2=high_mask,
        op=nl.multiply,
    )
    weight_bf16 = nl.ndarray(
        (TILE, TILE), dtype=nl.bfloat16, buffer=nl.sbuf
    )
    nisa.tensor_tensor(
        dst=weight_bf16,
        data1=safe_bf16,
        data2=correction,
        op=nl.add,
    )
    return weight_bf16


def _load_native_fp8_tile(source, indices):
    weight_i8 = nl.ndarray((TILE, TILE), dtype=nl.int8, buffer=nl.sbuf)
    nisa.dma_copy(dst=weight_i8, src=source[indices])
    return weight_i8.view(nl.float8_e4m3)


def _load_scale(source):
    stored = nl.ndarray((TILE, 1), dtype=source.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=stored[:, 0], src=source)
    scale = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=scale, op=nl.copy, data=stored, scale=1.0)
    return scale


def _broadcast_partition0(source, free_size):
    broadcast = nl.ndarray(
        (TILE, free_size), dtype=source.dtype, buffer=nl.sbuf
    )
    shuffle_mask = [0] * 32
    for partition_start in range(0, TILE, 32):
        nisa.nc_stream_shuffle(
            src=source,
            dst=broadcast[partition_start : partition_start + 32, :],
            shuffle_mask=shuffle_mask,
        )
    return broadcast


def _load_coalesced_scale_table(source, expert):
    """Load and convert one expert's repeated BF16 scale table once."""
    table_size = source.shape[2]
    stored = nl.ndarray(
        (TILE, table_size), dtype=source.dtype, buffer=nl.sbuf
    )
    nisa.dma_copy(dst=stored, src=source[expert, :, :])
    converted = nl.ndarray(
        (TILE, table_size), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.activation(
        dst=converted, op=nl.copy, data=stored, scale=1.0
    )
    return converted


def _gate_or_up_projection(
    hidden_sb,
    weights,
    residual_weights,
    scales,
    expert,
    projection,
    token_size,
    hidden_tiles,
    intermediate_tiles,
    use_fp8,
    tokens_stationary,
    native_fp8,
    dual_plane,
):
    projected = nl.ndarray(
        (TILE, intermediate_tiles, token_size),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.memset(dst=projected, value=0.0)

    for output_tile in nl.affine_range(intermediate_tiles):
        output_start = output_tile * TILE
        projected_psum_shape = (
            (token_size, TILE)
            if tokens_stationary
            else (TILE, token_size)
        )
        projected_psum = nl.ndarray(
            projected_psum_shape, dtype=nl.float32, buffer=nl.psum
        )
        for input_tile in nl.sequential_range(hidden_tiles):
            input_start = input_tile * TILE
            indices = (
                expert,
                projection,
                nl.ds(input_start, TILE),
                nl.ds(output_start, TILE),
            )
            scale = _load_scale(
                scales[
                    expert,
                    projection,
                    output_tile,
                    input_tile,
                    :,
                ]
            )
            # Reusing one PSUM destination forms a TensorE accumulation group
            # across the full contraction dimension.
            if native_fp8:
                base = _load_native_fp8_tile(weights, indices)
                scaled_hidden = nl.ndarray(
                    (TILE, token_size),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                )
                nisa.tensor_scalar(
                    dst=scaled_hidden,
                    data=hidden_sb[:, input_tile, :],
                    op0=nl.multiply,
                    operand0=scale,
                )
                nisa.nc_matmul(
                    dst=projected_psum,
                    stationary=base,
                    moving=scaled_hidden,
                )
                if dual_plane:
                    residual = _load_native_fp8_tile(
                        residual_weights, indices
                    )
                    nisa.nc_matmul(
                        dst=projected_psum,
                        stationary=residual,
                        moving=scaled_hidden,
                    )
            elif tokens_stationary:
                weight = _load_weight_tile(weights, indices, use_fp8)
                scaled_weight = nl.ndarray(
                    (TILE, TILE), dtype=nl.bfloat16, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    dst=scaled_weight,
                    data=weight,
                    op0=nl.multiply,
                    operand0=scale,
                )
                nisa.nc_matmul(
                    dst=projected_psum,
                    stationary=hidden_sb[:, input_tile, :],
                    moving=scaled_weight,
                )
            else:
                weight = _load_weight_tile(weights, indices, use_fp8)
                scaled_weight = nl.ndarray(
                    (TILE, TILE), dtype=nl.bfloat16, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    dst=scaled_weight,
                    data=weight,
                    op0=nl.multiply,
                    operand0=scale,
                )
                nisa.nc_matmul(
                    dst=projected_psum,
                    stationary=scaled_weight,
                    moving=hidden_sb[:, input_tile, :],
                )
        if tokens_stationary:
            projected_row = nl.ndarray(
                (TILE, TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(
                dst=projected_row[:token_size, :],
                src=projected_psum,
            )
            projected_transposed_psum = nl.ndarray(
                (TILE, token_size), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(
                dst=projected_transposed_psum,
                data=projected_row[:token_size, :],
            )
            nisa.tensor_copy(
                dst=projected[:, output_tile, :],
                src=projected_transposed_psum,
            )
        else:
            nisa.tensor_copy(
                dst=projected[:, output_tile, :],
                src=projected_psum,
            )
    return projected


def _moe_fused_w8_block_coalesced(
    hidden,
    gate_up,
    down,
    gate_up_scales,
    down_scales,
    affinities,
    coarse_scale=False,
):
    """Coalesced native-E4M3 MoE retaining exact 128x128 block scales.

    ``coarse_scale`` (Reduction B1) requires the scale table to be
    input-independent (one scale per 128-output-block, repeated across input
    tiles — produced by moe_w8.requantize_official_fp8_output_block). It then
    accumulates the whole contraction natively in PSUM and post-scales each
    output block once, instead of the per-128x128-block Vector scale-adds. It is
    ONLY correct on such coarse weights, so it defaults OFF: the standard
    per-block path (block_pow2_coalesced) is unchanged.

    The ring-buffer and TensorE column-packing schedule is adapted from
    nki-library commit 7a5b6f9 (`moe_tkg/mlp_tkg_*_projection.py`).
    Weight slabs are wider than scale blocks, so each 128-column PSUM partial
    is scaled in SBUF before contraction-block accumulation.
    """
    tokens, hidden_size = hidden.shape
    experts, hidden_size_w, projections, intermediate = gate_up.shape
    experts_d, intermediate_d, hidden_size_d = down.shape
    assert tokens in (32, 64, 128)
    assert projections == 2
    assert hidden_size == hidden_size_w == hidden_size_d
    assert experts == experts_d
    assert intermediate == intermediate_d
    assert hidden_size % COALESCED_WIDTH == 0
    assert intermediate == COALESCED_WIDTH
    assert affinities.shape == (tokens, experts)

    hidden_tiles = hidden_size // TILE
    intermediate_tiles = intermediate // TILE
    assert gate_up_scales.shape == (
        experts,
        TILE,
        2 * hidden_tiles * intermediate_tiles,
    )
    assert down_scales.shape == (
        experts,
        TILE,
        intermediate_tiles * hidden_tiles,
    )

    if tokens == 32:
        column_packing = 4
    elif tokens == 64:
        column_packing = 2
    else:
        column_packing = 1
    assert hidden_tiles % column_packing == 0
    assert intermediate_tiles % column_packing == 0

    output = nl.ndarray(
        (tokens, hidden_size), dtype=nl.float32, buffer=nl.shared_hbm
    )

    hidden_sb = nl.ndarray(
        (TILE, hidden_tiles, tokens),
        dtype=hidden.dtype,
        buffer=nl.sbuf,
    )
    for hidden_tile in nl.affine_range(hidden_tiles):
        hidden_start = hidden_tile * TILE
        hidden_row = nl.ndarray(
            (TILE, TILE), dtype=hidden.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=hidden_row[:tokens, :],
            src=hidden[:, hidden_start : hidden_start + TILE],
        )
        hidden_transposed_psum = nl.ndarray(
            (TILE, tokens), dtype=hidden.dtype, buffer=nl.psum
        )
        nisa.nc_transpose(
            dst=hidden_transposed_psum,
            data=hidden_row[:tokens, :],
        )
        nisa.tensor_copy(
            dst=hidden_sb[:, hidden_tile, :],
            src=hidden_transposed_psum,
        )

    routed = nl.ndarray(
        (tokens, hidden_size), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.memset(dst=routed, value=0.0)
    expert_output = nl.ndarray(
        (tokens, hidden_size), dtype=nl.float32, buffer=nl.sbuf
    )

    # Two rotating slots break load/use anti-dependencies while keeping the
    # expert scratch bounded. Gate and up need independent PSUM streams.
    weight_slots = [
        nl.ndarray(
            (TILE, 2, COALESCED_WIDTH),
            dtype=nl.int8,
            buffer=nl.sbuf,
        ),
        nl.ndarray(
            (TILE, 2, COALESCED_WIDTH),
            dtype=nl.int8,
            buffer=nl.sbuf,
        ),
    ]
    # PSUMs are allocated FRESH per input_group (in the loops below) instead of
    # reused rotating slots: a matmul into a fresh nl.psum ndarray overwrites
    # (starts the accumulation group), so no per-iteration memset is needed —
    # matching _gate_or_up_projection. (Reused slots require zeroing because the
    # matmul continues-accumulates onto stale data; confirmed by iso test.)

    for expert in nl.sequential_range(experts):
        gate_scale_table = _load_coalesced_scale_table(
            gate_up_scales, expert
        )
        down_scale_table = _load_coalesced_scale_table(
            down_scales, expert
        )
        nisa.memset(dst=expert_output, value=0.0)

        affinity = nl.ndarray(
            (tokens, 1), dtype=affinities.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=affinity,
            src=affinities[:, expert : expert + 1],
        )
        affinity_f32 = nl.ndarray(
            (tokens, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.activation(
            dst=affinity_f32,
            op=nl.copy,
            data=affinity,
            scale=1.0,
        )

        gate = nl.ndarray(
            (tokens, intermediate), dtype=nl.float32, buffer=nl.sbuf
        )
        up = nl.ndarray(
            (tokens, intermediate), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=gate, value=0.0)
        nisa.memset(dst=up, value=0.0)
        partial = nl.ndarray(
            (tokens, COALESCED_WIDTH),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )

        gate_groups = hidden_tiles // column_packing
        if column_packing == 1 and coarse_scale:
            # Reduction B1: the coarse output-block scale is input-independent
            # (s[i,o] == s[o]), so sum_i(partial_i * s[o]) == s[o] * sum_i
            # partial_i. Accumulate the whole hidden contraction natively in
            # PSUM (a matmul into a fresh nl.psum starts the accumulation group;
            # subsequent matmuls accumulate), then post-scale each output block
            # ONCE — replacing hidden_tiles*intermediate_tiles narrow Vector
            # scale-adds with intermediate_tiles wide post-scales per projection.
            gate_psum = nl.ndarray(
                (TILE, COALESCED_WIDTH), dtype=nl.float32, buffer=nl.psum
            )
            up_psum = nl.ndarray(
                (TILE, COALESCED_WIDTH), dtype=nl.float32, buffer=nl.psum
            )
            for input_tile in range(hidden_tiles):
                input_start = input_tile * TILE
                weight_slot = input_tile % 2
                nisa.dma_copy(
                    dst=weight_slots[weight_slot],
                    src=gate_up[
                        expert,
                        nl.ds(input_start, TILE),
                        :,
                        :,
                    ],
                )
                native_weight = weight_slots[weight_slot].view(
                    nl.float8_e4m3
                )
                nisa.nc_matmul(
                    dst=gate_psum,
                    stationary=hidden_sb[:, input_tile, :],
                    moving=native_weight[:, 0, :],
                )
                nisa.nc_matmul(
                    dst=up_psum,
                    stationary=hidden_sb[:, input_tile, :],
                    moving=native_weight[:, 1, :],
                )
            for output_block in range(intermediate_tiles):
                output_start = output_block * TILE
                gate_scale_index = output_block
                up_scale_index = (
                    hidden_tiles * intermediate_tiles + output_block
                )
                nisa.tensor_scalar(
                    dst=gate[:, nl.ds(output_start, TILE)],
                    data=gate_psum[:tokens, nl.ds(output_start, TILE)],
                    op0=nl.multiply,
                    operand0=gate_scale_table[
                        :tokens,
                        gate_scale_index : gate_scale_index + 1,
                    ],
                )
                nisa.tensor_scalar(
                    dst=up[:, nl.ds(output_start, TILE)],
                    data=up_psum[:tokens, nl.ds(output_start, TILE)],
                    op0=nl.multiply,
                    operand0=gate_scale_table[
                        :tokens,
                        up_scale_index : up_scale_index + 1,
                    ],
                )
        else:
          for input_group in range(gate_groups):
            gate_psum = nl.ndarray(
                (TILE, COALESCED_WIDTH), dtype=nl.float32, buffer=nl.psum
            )
            up_psum = nl.ndarray(
                (TILE, COALESCED_WIDTH), dtype=nl.float32, buffer=nl.psum
            )
            for packed_index in range(column_packing):
                input_tile = (
                    input_group * column_packing + packed_index
                )
                input_start = input_tile * TILE
                weight_slot = packed_index % 2
                nisa.dma_copy(
                    dst=weight_slots[weight_slot],
                    src=gate_up[
                        expert,
                        nl.ds(input_start, TILE),
                        :,
                        :,
                    ],
                )
                packed_row = packed_index * tokens
                native_weight = weight_slots[weight_slot].view(
                    nl.float8_e4m3
                )
                nisa.nc_matmul(
                    dst=gate_psum[
                        nl.ds(packed_row, tokens), :
                    ],
                    stationary=hidden_sb[:, input_tile, :],
                    moving=native_weight[:, 0, :],
                    tile_position=(0, packed_row),
                    tile_size=(TILE, tokens),
                )
                nisa.nc_matmul(
                    dst=up_psum[
                        nl.ds(packed_row, tokens), :
                    ],
                    stationary=hidden_sb[:, input_tile, :],
                    moving=native_weight[:, 1, :],
                    tile_position=(0, packed_row),
                    tile_size=(TILE, tokens),
                )

            for packed_index in range(column_packing):
                input_tile = (
                    input_group * column_packing + packed_index
                )
                packed_row = packed_index * tokens
                nisa.activation(
                    dst=partial,
                    op=nl.copy,
                    data=gate_psum[
                        nl.ds(packed_row, tokens), :
                    ],
                    scale=1.0,
                )
                for output_block in range(intermediate_tiles):
                    output_start = output_block * TILE
                    scale_index = (
                        input_tile * intermediate_tiles + output_block
                    )
                    partial_block = partial[
                        :, nl.ds(output_start, TILE)
                    ]
                    gate_block = gate[
                        :, nl.ds(output_start, TILE)
                    ]
                    nisa.scalar_tensor_tensor(
                        data=partial_block,
                        op0=nl.multiply,
                        operand0=gate_scale_table[
                            :tokens,
                            scale_index : scale_index + 1,
                        ],
                        op1=nl.add,
                        operand1=gate_block,
                        dst=gate_block,
                    )

                nisa.activation(
                    dst=partial,
                    op=nl.copy,
                    data=up_psum[
                        nl.ds(packed_row, tokens), :
                    ],
                    scale=1.0,
                )
                for output_block in range(intermediate_tiles):
                    output_start = output_block * TILE
                    scale_index = (
                        hidden_tiles * intermediate_tiles
                        + input_tile * intermediate_tiles
                        + output_block
                    )
                    partial_block = partial[
                        :, nl.ds(output_start, TILE)
                    ]
                    up_block = up[
                        :, nl.ds(output_start, TILE)
                    ]
                    nisa.scalar_tensor_tensor(
                        data=partial_block,
                        op0=nl.multiply,
                        operand0=gate_scale_table[
                            :tokens,
                            scale_index : scale_index + 1,
                        ],
                        op1=nl.add,
                        operand1=up_block,
                        dst=up_block,
                    )

        zero_bias = nl.ndarray(
            (tokens, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=zero_bias, value=0.0)
        silu_gate = nl.ndarray(
            (tokens, intermediate), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.activation(
            dst=silu_gate,
            op=nl.silu,
            data=gate,
            bias=zero_bias,
            scale=1.0,
        )
        product = nl.ndarray(
            (tokens, intermediate), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=product,
            data1=silu_gate,
            data2=up,
            op=nl.multiply,
        )
        activated_row = nl.ndarray(
            (tokens, intermediate),
            dtype=hidden.dtype,
            buffer=nl.sbuf,
        )
        nisa.activation(
            dst=activated_row,
            op=nl.copy,
            data=product,
            scale=1.0,
        )
        activated = nl.ndarray(
            (TILE, intermediate_tiles, tokens),
            dtype=hidden.dtype,
            buffer=nl.sbuf,
        )
        for intermediate_tile in nl.affine_range(intermediate_tiles):
            intermediate_start = intermediate_tile * TILE
            activated_psum = nl.ndarray(
                (TILE, tokens), dtype=hidden.dtype, buffer=nl.psum
            )
            nisa.nc_transpose(
                dst=activated_psum,
                data=activated_row[
                    :, nl.ds(intermediate_start, TILE)
                ],
            )
            nisa.tensor_copy(
                dst=activated[:, intermediate_tile, :],
                src=activated_psum,
            )

        hidden_slabs = hidden_size // COALESCED_WIDTH
        down_groups = intermediate_tiles // column_packing
        for hidden_slab in nl.affine_range(hidden_slabs):
            hidden_start = hidden_slab * COALESCED_WIDTH
            if column_packing == 1 and coarse_scale:
                # Reduction B1 (see gate/up): accumulate the intermediate
                # contraction for this hidden slab natively in PSUM, then
                # post-scale each 128-wide output block ONCE with s[o].
                down_psum = nl.ndarray(
                    (TILE, COALESCED_WIDTH), dtype=nl.float32, buffer=nl.psum
                )
                for intermediate_tile in range(intermediate_tiles):
                    intermediate_start = intermediate_tile * TILE
                    weight_slot = intermediate_tile % 2
                    nisa.dma_copy(
                        dst=weight_slots[weight_slot][:, 0, :],
                        src=down[
                            expert,
                            nl.ds(intermediate_start, TILE),
                            nl.ds(hidden_start, COALESCED_WIDTH),
                        ],
                    )
                    native_weight = weight_slots[weight_slot][
                        :, 0, :
                    ].view(nl.float8_e4m3)
                    nisa.nc_matmul(
                        dst=down_psum,
                        stationary=activated[:, intermediate_tile, :],
                        moving=native_weight,
                    )
                for slab_block in range(COALESCED_WIDTH // TILE):
                    output_block = (
                        hidden_slab * (COALESCED_WIDTH // TILE) + slab_block
                    )
                    output_start = slab_block * TILE
                    scale_index = output_block
                    expert_start = hidden_start + output_start
                    nisa.tensor_scalar(
                        dst=expert_output[:, nl.ds(expert_start, TILE)],
                        data=down_psum[:tokens, nl.ds(output_start, TILE)],
                        op0=nl.multiply,
                        operand0=down_scale_table[
                            :tokens,
                            scale_index : scale_index + 1,
                        ],
                    )
                # column_packing==1 is a compile-time constant, so this branch
                # is taken uniformly for every slab (safe inside affine_range).
                continue
            for input_group in range(down_groups):
                down_psum = nl.ndarray(
                    (TILE, COALESCED_WIDTH), dtype=nl.float32, buffer=nl.psum
                )
                for packed_index in range(column_packing):
                    intermediate_tile = (
                        input_group * column_packing + packed_index
                    )
                    intermediate_start = intermediate_tile * TILE
                    weight_slot = packed_index % 2
                    nisa.dma_copy(
                        dst=weight_slots[weight_slot][:, 0, :],
                        src=down[
                            expert,
                            nl.ds(intermediate_start, TILE),
                            nl.ds(hidden_start, COALESCED_WIDTH),
                        ],
                    )
                    packed_row = packed_index * tokens
                    native_weight = weight_slots[weight_slot][
                        :, 0, :
                    ].view(nl.float8_e4m3)
                    nisa.nc_matmul(
                        dst=down_psum[
                            nl.ds(packed_row, tokens), :
                        ],
                        stationary=activated[
                            :, intermediate_tile, :
                        ],
                        moving=native_weight,
                        tile_position=(0, packed_row),
                        tile_size=(TILE, tokens),
                    )

                for packed_index in range(column_packing):
                    intermediate_tile = (
                        input_group * column_packing + packed_index
                    )
                    packed_row = packed_index * tokens
                    nisa.activation(
                        dst=partial,
                        op=nl.copy,
                        data=down_psum[
                            nl.ds(packed_row, tokens), :
                        ],
                        scale=1.0,
                    )
                    for slab_block in range(
                        COALESCED_WIDTH // TILE
                    ):
                        output_block = (
                            hidden_slab * (COALESCED_WIDTH // TILE)
                            + slab_block
                        )
                        output_start = slab_block * TILE
                        scale_index = (
                            intermediate_tile * hidden_tiles
                            + output_block
                        )
                        partial_block = partial[
                            :, nl.ds(output_start, TILE)
                        ]
                        expert_start = hidden_start + output_start
                        expert_block = expert_output[
                            :, nl.ds(expert_start, TILE)
                        ]
                        nisa.scalar_tensor_tensor(
                            dst=expert_block,
                            data=partial_block,
                            op0=nl.multiply,
                            operand0=down_scale_table[
                                :tokens,
                                scale_index : scale_index + 1,
                            ],
                            op1=nl.add,
                            operand1=expert_block,
                        )

        for hidden_tile in nl.affine_range(hidden_tiles):
            hidden_start = hidden_tile * TILE
            routed_block = routed[:, nl.ds(hidden_start, TILE)]
            nisa.scalar_tensor_tensor(
                data=expert_output[:, nl.ds(hidden_start, TILE)],
                op0=nl.multiply,
                operand0=affinity_f32[:, 0],
                op1=nl.add,
                operand1=routed_block,
                dst=routed_block,
            )

    nisa.dma_copy(dst=output, src=routed)
    return output


def _moe_fused_w8(
    hidden,
    gate_up,
    gate_up_residual,
    down,
    down_residual,
    gate_up_scales,
    down_scales,
    affinities,
    use_fp8,
    tokens_stationary,
    native_fp8,
    dual_plane,
):
    tokens, hidden_size = hidden.shape
    experts, projections, hidden_size_w, intermediate = gate_up.shape
    experts_d, intermediate_d, hidden_size_d = down.shape
    assert tokens in (32, 64, 128, 256)
    assert projections == 2
    assert hidden_size == hidden_size_w == hidden_size_d
    assert experts == experts_d
    assert intermediate == intermediate_d
    assert hidden_size % TILE == 0
    assert intermediate % TILE == 0
    assert affinities.shape == (tokens, experts)
    if native_fp8:
        assert use_fp8
        assert not tokens_stationary
    if dual_plane:
        assert native_fp8
        assert gate_up_residual.shape == gate_up.shape
        assert down_residual.shape == down.shape

    hidden_tiles = hidden_size // TILE
    intermediate_tiles = intermediate // TILE
    assert gate_up_scales.shape == (
        experts,
        2,
        intermediate_tiles,
        hidden_tiles,
        TILE,
    )
    assert down_scales.shape == (
        experts,
        hidden_tiles,
        intermediate_tiles,
        TILE,
    )

    output = nl.ndarray(
        (tokens, hidden_size), dtype=nl.float32, buffer=nl.shared_hbm
    )
    token_tiles = (tokens + TILE - 1) // TILE

    for token_tile in nl.sequential_range(token_tiles):
        token_start = token_tile * TILE
        token_size = min(TILE, tokens - token_start)
        # Preload/transposed hidden once and reuse it for every local expert.
        hidden_sb = nl.ndarray(
            (TILE, hidden_tiles, token_size),
            dtype=hidden.dtype,
            buffer=nl.sbuf,
        )
        for hidden_tile in nl.affine_range(hidden_tiles):
            hidden_start = hidden_tile * TILE
            hidden_row = nl.ndarray(
                (TILE, TILE), dtype=hidden.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=hidden_row[:token_size, :],
                src=hidden[
                    token_start : token_start + token_size,
                    hidden_start : hidden_start + TILE,
                ],
            )
            hidden_transposed_psum = nl.ndarray(
                (TILE, TILE), dtype=hidden.dtype, buffer=nl.psum
            )
            nisa.nc_transpose(
                dst=hidden_transposed_psum[:, :token_size],
                data=hidden_row[:token_size, :],
            )
            nisa.tensor_copy(
                dst=hidden_sb[:, hidden_tile, :],
                src=hidden_transposed_psum[:, :token_size],
            )

        routed = nl.ndarray(
            (TILE, hidden_tiles, token_size),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        nisa.memset(dst=routed, value=0.0)
        zero_bias = nl.ndarray((TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zero_bias, value=0.0)

        for expert in nl.sequential_range(experts):
            affinity_col = nl.ndarray(
                (TILE, 1), dtype=affinities.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=affinity_col[:token_size, :],
                src=affinities[
                    token_start : token_start + token_size,
                    expert : expert + 1,
                ],
            )
            affinity_psum = nl.ndarray(
                (1, TILE), dtype=affinities.dtype, buffer=nl.psum
            )
            nisa.nc_transpose(
                dst=affinity_psum[:, :token_size],
                data=affinity_col[:token_size, :],
            )
            affinity = nl.ndarray(
                (1, token_size), dtype=affinities.dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=affinity, src=affinity_psum[:, :token_size])
            affinity_broadcast = _broadcast_partition0(affinity, token_size)

            gate = _gate_or_up_projection(
                hidden_sb,
                gate_up,
                gate_up_residual,
                gate_up_scales,
                expert,
                0,
                token_size,
                hidden_tiles,
                intermediate_tiles,
                use_fp8,
                tokens_stationary,
                native_fp8,
                dual_plane,
            )
            up = _gate_or_up_projection(
                hidden_sb,
                gate_up,
                gate_up_residual,
                gate_up_scales,
                expert,
                1,
                token_size,
                hidden_tiles,
                intermediate_tiles,
                use_fp8,
                tokens_stationary,
                native_fp8,
                dual_plane,
            )

            activated = nl.ndarray(
                (TILE, intermediate_tiles, token_size),
                dtype=hidden.dtype,
                buffer=nl.sbuf,
            )
            for intermediate_tile in nl.affine_range(intermediate_tiles):
                silu_gate = nl.ndarray(
                    (TILE, token_size), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.activation(
                    dst=silu_gate,
                    op=nl.silu,
                    data=gate[:, intermediate_tile, :],
                    bias=zero_bias,
                    scale=1.0,
                )
                product = nl.ndarray(
                    (TILE, token_size), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_tensor(
                    dst=product,
                    data1=silu_gate,
                    data2=up[:, intermediate_tile, :],
                    op=nl.multiply,
                )
                nisa.activation(
                    dst=activated[:, intermediate_tile, :],
                    op=nl.copy,
                    data=product,
                    scale=1.0,
                )

            for hidden_tile in nl.affine_range(hidden_tiles):
                hidden_start = hidden_tile * TILE
                expert_output_psum_shape = (
                    (token_size, TILE)
                    if tokens_stationary
                    else (TILE, token_size)
                )
                expert_output_psum = nl.ndarray(
                    expert_output_psum_shape,
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                for intermediate_tile in nl.sequential_range(intermediate_tiles):
                    intermediate_start = intermediate_tile * TILE
                    indices = (
                        expert,
                        nl.ds(intermediate_start, TILE),
                        nl.ds(hidden_start, TILE),
                    )
                    scale = _load_scale(
                        down_scales[
                            expert,
                            hidden_tile,
                            intermediate_tile,
                            :,
                        ]
                    )
                    if native_fp8:
                        base = _load_native_fp8_tile(down, indices)
                        scaled_activated = nl.ndarray(
                            (TILE, token_size),
                            dtype=nl.bfloat16,
                            buffer=nl.sbuf,
                        )
                        nisa.tensor_scalar(
                            dst=scaled_activated,
                            data=activated[:, intermediate_tile, :],
                            op0=nl.multiply,
                            operand0=scale,
                        )
                        nisa.nc_matmul(
                            dst=expert_output_psum,
                            stationary=base,
                            moving=scaled_activated,
                        )
                        if dual_plane:
                            residual = _load_native_fp8_tile(
                                down_residual, indices
                            )
                            nisa.nc_matmul(
                                dst=expert_output_psum,
                                stationary=residual,
                                moving=scaled_activated,
                            )
                    elif tokens_stationary:
                        weight = _load_weight_tile(
                            down, indices, use_fp8
                        )
                        scaled_weight = nl.ndarray(
                            (TILE, TILE),
                            dtype=nl.bfloat16,
                            buffer=nl.sbuf,
                        )
                        nisa.tensor_scalar(
                            dst=scaled_weight,
                            data=weight,
                            op0=nl.multiply,
                            operand0=scale,
                        )
                        nisa.nc_matmul(
                            dst=expert_output_psum,
                            stationary=activated[:, intermediate_tile, :],
                            moving=scaled_weight,
                        )
                    else:
                        weight = _load_weight_tile(
                            down, indices, use_fp8
                        )
                        scaled_weight = nl.ndarray(
                            (TILE, TILE),
                            dtype=nl.bfloat16,
                            buffer=nl.sbuf,
                        )
                        nisa.tensor_scalar(
                            dst=scaled_weight,
                            data=weight,
                            op0=nl.multiply,
                            operand0=scale,
                        )
                        nisa.nc_matmul(
                            dst=expert_output_psum,
                            stationary=scaled_weight,
                            moving=activated[:, intermediate_tile, :],
                        )
                expert_output = nl.ndarray(
                    (TILE, token_size), dtype=nl.float32, buffer=nl.sbuf
                )
                if tokens_stationary:
                    expert_output_row = nl.ndarray(
                        (TILE, TILE), dtype=nl.float32, buffer=nl.sbuf
                    )
                    nisa.tensor_copy(
                        dst=expert_output_row[:token_size, :],
                        src=expert_output_psum,
                    )
                    expert_output_transposed_psum = nl.ndarray(
                        (TILE, token_size),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    nisa.nc_transpose(
                        dst=expert_output_transposed_psum,
                        data=expert_output_row[:token_size, :],
                    )
                    nisa.tensor_copy(
                        dst=expert_output,
                        src=expert_output_transposed_psum,
                    )
                else:
                    nisa.tensor_copy(
                        dst=expert_output,
                        src=expert_output_psum,
                    )
                weighted = nl.ndarray(
                    (TILE, token_size), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_tensor(
                    dst=weighted,
                    data1=expert_output,
                    data2=affinity_broadcast,
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=routed[:, hidden_tile, :],
                    data1=weighted,
                    data2=routed[:, hidden_tile, :],
                    op=nl.add,
                )

        # Preserve the rank-local FP32 sum through the TP all-reduce. The caller
        # applies the model's BF16 boundary only after adding the shared expert.
        for hidden_tile in nl.affine_range(hidden_tiles):
            hidden_start = hidden_tile * TILE
            routed_transposed_psum = nl.ndarray(
                (TILE, TILE), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(
                dst=routed_transposed_psum[:token_size, :],
                data=routed[:, hidden_tile, :],
            )
            routed_transposed = nl.ndarray(
                (TILE, TILE), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(
                dst=routed_transposed[:token_size, :],
                src=routed_transposed_psum[:token_size, :],
            )
            nisa.dma_copy(
                dst=output[
                    token_start : token_start + token_size,
                    hidden_start : hidden_start + TILE,
                ],
                src=routed_transposed[:token_size, :],
            )

    return output


@nki.jit
def nki_moe_fused_w8_fp8(
    hidden,
    gate_up,
    down,
    gate_up_scales,
    down_scales,
    affinities,
):
    return _moe_fused_w8(
        hidden,
        gate_up,
        gate_up,
        down,
        down,
        gate_up_scales,
        down_scales,
        affinities,
        True,
        False,
        False,
        False,
    )


@nki.jit
def nki_moe_fused_w8_fp8_native(
    hidden,
    gate_up,
    down,
    gate_up_scales,
    down_scales,
    affinities,
):
    return _moe_fused_w8(
        hidden,
        gate_up,
        gate_up,
        down,
        down,
        gate_up_scales,
        down_scales,
        affinities,
        True,
        False,
        True,
        False,
    )


@nki.jit
def nki_moe_fused_w8_fp8_block_coalesced(
    hidden,
    gate_up,
    down,
    gate_up_scales,
    down_scales,
    affinities,
):
    return _moe_fused_w8_block_coalesced(
        hidden,
        gate_up,
        down,
        gate_up_scales,
        down_scales,
        affinities,
    )


@nki.jit
def nki_moe_fused_w8_fp8_block_coalesced_ob(
    hidden,
    gate_up,
    down,
    gate_up_scales,
    down_scales,
    affinities,
):
    # Reduction B1: input-independent per-output-block scales, PSUM-accumulate.
    return _moe_fused_w8_block_coalesced(
        hidden,
        gate_up,
        down,
        gate_up_scales,
        down_scales,
        affinities,
        coarse_scale=True,
    )


@nki.jit
def nki_moe_fused_w8_fp8_dual(
    hidden,
    gate_up,
    gate_up_residual,
    down,
    down_residual,
    gate_up_scales,
    down_scales,
    affinities,
):
    return _moe_fused_w8(
        hidden,
        gate_up,
        gate_up_residual,
        down,
        down_residual,
        gate_up_scales,
        down_scales,
        affinities,
        True,
        False,
        True,
        True,
    )


@nki.jit
def nki_moe_fused_w8_fp8_token_stationary(
    hidden,
    gate_up,
    down,
    gate_up_scales,
    down_scales,
    affinities,
):
    return _moe_fused_w8(
        hidden,
        gate_up,
        gate_up,
        down,
        down,
        gate_up_scales,
        down_scales,
        affinities,
        True,
        True,
        False,
        False,
    )


@nki.jit
def nki_moe_fused_w8_int8(
    hidden,
    gate_up,
    down,
    gate_up_scales,
    down_scales,
    affinities,
):
    return _moe_fused_w8(
        hidden,
        gate_up,
        gate_up,
        down,
        down,
        gate_up_scales,
        down_scales,
        affinities,
        False,
        False,
        False,
        False,
    )
