# FP8 scale layouts, shapes, and gates

Reference for converting an official FP8 checkpoint into something a Trainium kernel can
consume. Implementations live in `contrib/qwen3.6-35b-a3b/moe_w8.py`.

## Source format

An official FP8 checkpoint stores, per weight:

- `...weight` — dtype `F8_E4M3` (E4M3**FN**, max 448.0)
- `...weight_scale_inv` — dtype `BF16`, shape = the weight's trailing 2 dims divided by 128,
  i.e. **one scale per 128×128 block**

Assert both dtypes when reading. A BF16 assumption that silently accepts F32 will produce
plausible garbage.

## Conversions

### pow2 exponent shift — `requantize_official_fp8_pow2` (exact)

For each 128×128 block: if any byte's magnitude is ≥ `0x78` (an E4M3FN-only code), halve the
values and **double** the BF16 scale; otherwise keep the bytes and scale **bit-for-bit**.

- Maps E4M3FN 256..448 onto legacy-E4M3 128..224.
- All values except the smallest normals/subnormals remain exactly representable.
- Verify the doubled scale is exactly representable in BF16 (`stored*0.5 == source`) and raise
  if not — silent inexactness here is unrecoverable later.
- Preserves the official 128×128 grid, so this is the layout to use when faithfulness matters.

### dual plane — `split_official_fp8`

Splits each value into a primary and a residual FP8 plane; the kernel runs two matmuls and adds.
Most accurate, 2× weight bytes, 2× the matmuls. Use only when accuracy is the binding constraint.

### output-block — `requantize_official_fp8_output_block`

One scale per **output** block rather than per contraction block. Enables PSUM accumulation
across the whole contraction with a single scale applied at the end — the Vector-tax remedy in
SKILL.md. Coarser; requires a cosine gate.

### row — `requantize_official_fp8_row`

One scale per output channel. Coarsest; rejected for this model. `retain_official_fp8_row_bf16`
keeps BF16 for layers outside a configured FP8 range.

## Kernel-facing tensor shapes

Non-coalesced block layouts (the token-tiled kernel family), with `TILE = BLOCK_SIZE = 128`:

```
w8_gate_up        [E, 2, H, I]                      int8 legacy-E4M3
w8_down           [E, I, H]                          int8 legacy-E4M3
w8_gate_up_scale  [E, 2, I/128, H/128, 128]          BF16
w8_down_scale     [E, H/128, I/128, 128]             BF16
```

The trailing `128` is a **partition pre-broadcast**: NKI block scales are partition-broadcast
operands, so the grid is expanded in HBM (`grid.unsqueeze(-1).expand(..., 128).contiguous()`)
to avoid a stream shuffle per GEMM tile. Cost is negligible (~1.5 MB/layer) and it must match
the kernel's asserts exactly.

256×256 block-quant layouts for the double_row CTE path:

```
gate_up_proj_scale [E, H/256, 2, I/256, 128]
down_proj_scale    [E, I/256, H/256, 128]
```

Coalesced layouts differ again (`pack_coalesced_block_scales`, and the weight stack uses
`dim=1` rather than `dim=0`). **Verify the layout the impl expects rather than assuming the
family shares one** — an impl↔layout mismatch produces finite, wrong numbers, not an error.

## Padding hazard: the expert sentinel

Kernels that gather weights/scales by a per-block expert id will index **past** an `[E, ...]`
tensor when padded blocks carry the sentinel `== E`. Two fixes:

- **E+1 dummy-expert padding** — pad weights, scales *and* affinities to `E+1`; the wrapper
  derives the real `E = shape[0] - 1`. Padding blocks then read a zero dummy in bounds.
- **Clamp** the block→expert map to `E-1` (simpler, but computes padded blocks, and doing the
  clamp with per-rank device work can desynchronize a multi-rank collective schedule — prefer
  clamping on the host if you go this way).

Note a "skip padded blocks" mechanism does **not** save you if the kernel gathers weights
*before* the skip decision.

## Quality gates

Print at load and gate on them; do not accept a conversion silently.

| stat | meaning | gate used here |
|---|---|---|
| `cosine` | dequant vs original | ≥ 0.999 (measured 0.9996) |
| `normalized_rmse` | relative error | report |
| `shifted_block_fraction` | blocks needing the pow2 shift | report (pow2 only) |
| `exact_fraction` | values exactly representable | report |
| `clipped_count` | values clipped at legacy max 240.0 | **must be 0**, else raise |

Device-level: a standalone kernel test against the block-quant torch reference gave
**cosine 0.999259**; a 4-layer end-to-end check gave **identical top-5 tokens** with norm
945.56 vs 944.18 (0.15%). That pair — unit cosine plus end-to-end coherence — is the gate to
reproduce for any new quantization path.

## Reference implementations to compare against

Build the CPU/torch reference with the **same** FP8 weights and FP32 activations, so the only
difference under test is the activation cast. That is what isolated our activation-quant error
to "numerically fine" rather than leaving it confounded with weight error.
