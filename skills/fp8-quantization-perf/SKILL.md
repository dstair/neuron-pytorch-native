---
name: fp8-quantization-perf
description: |
  Make FP8 / int8 quantized kernels work and go fast on Trainium2. Use when working on FP8,
  E4M3, E5M2, MXFP8, block quantization, weight_scale_inv, dequant scales, double_row / 2x
  matmul, "quantize the experts", "FP8 is slower than BF16", or when you hit
  "NCC_EVRF051", a rejected float8_e4m3fn operand, or are deciding between row / block /
  output-block / dual-plane scale layouts. Explains why FP8 buys CAPACITY not SPEED here,
  and how to remove the block-dequant Vector tax. For routing/blocking see moe-kernel-perf.
---

# FP8 on Trainium2

**Headline, measured repeatedly: on this hardware FP8 is a capacity lever, not a speed lever.**
Plan around memory, and treat any throughput gain as unproven upside to be profiled.

- Memory: **5.08 vs 8.94 GB/core** at 40 layers, bit-identical outputs. Real, and it fits
  configurations BF16 cannot.
- Speed: a ~**7% tax** vs BF16 at the identical configuration, mechanically explained below.
- The 2× Tensor-engine FP8 rate is real (158 vs 79 TFLOPS) and irrelevant while MFU ≈ 10% —
  see the three refutations in `nki-perf-patterns/references/refuted-levers.md`.

## Getting FP8 to compile at all (version-bound: SDK 2.32 / TRN2)

`torch.float8_e4m3fn` is rejected by **both** NKI and `neuronx-cc` on TRN2 (`NCC_EVRF051`), and
the `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` flag is **not forwarded** by this driver.

The workaround, used by every working kernel here:

```python
# store weights as int8 carrying legacy-E4M3 bytes ...
self.register_buffer(f"l{i}_w8_gate_up", int8_tensor)
# ... and reinterpret inside the NKI kernel
w = weights_int8.view(nl.float8_e4m3)
```

The HLO operand stays `int8`, so nothing downstream sees the rejected dtype. Note **legacy
E4M3** (max 240.0) is a different encoding from E4M3FN (max 448.0) — converting between them is
what the scale machinery below is for.

## The 2× double_row path — exact requirements

You get the 2× only for a true double-pumped matmul:

- **both** operands E4M3/E5M2 — a BF16 × FP8 mixed matmul gets **nothing**
- contraction laid out partition = 128 × free = 2
- FP32 accumulate, `perf_mode="double_row"`
- block quantization at **256×256** granularity (256 of contraction per double-row step)

Consequence: the pre-existing "FP8 decode kernel" was BF16 × FP8 mixed and never had the 2×.
Check the operand dtypes before believing a kernel is on the fast path.

MX / microscaling (E8M0) is a different format and the nkilib MX CTE path is **TRN3-only**
(`NUM_SHARDS == 2`); the newer MXFP8 kernels are training-**backward** only. Do not plan a TRN2
prefill around them.

## Scale layouts — pick deliberately

An official FP8 checkpoint ships `weight.weight_scale_inv` as a BF16 grid, one scale per
128×128 block. Four conversions, in increasing order of numeric damage:

| layout | what it does | numerics | use when |
|---|---|---|---|
| **pow2 exponent shift** | divide any block containing an E4M3FN-only code by 2, double its BF16 scale | **exact**; bit-for-bit for untouched blocks; only smallest normals/subnormals change | you must stay faithful to the official grid |
| **dual plane** | split each value into a primary + residual FP8 plane | most accurate, 2× the weight bytes and 2× the matmuls | accuracy is the binding constraint |
| **output-block** | one scale per *output* block instead of per contraction block | coarser — needs a cosine gate | you want the Vector tax gone (below) |
| **row** | one scale per output channel | coarsest | rejected for this model |

Layout details, tensor shapes and the quality-stat gates are in `references/scale-layouts.md`.

Always print the conversion's quality stats (cosine, normalized RMSE, shifted-block fraction,
clipped count) at load and gate on them. Ours: dequant cosine **0.9996**, and a device unit
test at **0.999259** against the block-quant reference.

## The block-dequant Vector tax — the main FP8 speed lever

**Mechanism.** With per-contraction-block scales, the H contraction must be broken into
256-element chunks, and each chunk's PSUM result must be multiplied by *its* block scale before
accumulating. That is one Vector `tensor_scalar` / `scalar_tensor_tensor` per chunk per token
tile. BF16 pays none of it. **This is the 7% tax.**

Measured in a 40-layer BS=16 prefill (`bwmm_shard_on_I.py:1384`, `:2133`, `:2140`):

- 62,304 Vector instructions, **38.15 ms = 26% of all Vector time**
- Vector was the **top engine** (40.6% active)
- ≈ 5.8 s of a 43.5 s prefill

**Remedy — and it is already implemented here for decode:** convert to **output-block** scales
and accumulate all contraction chunks in PSUM, scaling once at the end
(`requantize_official_fp8_output_block` + `nki_moe_fused_w8_fp8_block_coalesced_ob`, described
as "Reduction B1: input-independent per-output-block scales, PSUM-accumulate"). At H=2048 with
256-wide chunks this removes **7/8** of the scale ops.

Two honest caveats before you promise a number:

1. Coarser scales are **not** bit-identical to the official grid. Pre-register a cosine /
   coherence gate.
2. The pipeline ceiling may not move much: with Vector at 116.5 ms and GpSimd at 114.1 ms, a
   25% Vector cut buys ~2 ms of `perfect_pipeline`. The real hope is shortening serial
   dependency chains in a workload that is 79% semaphore wait. Make that argument explicitly
   or measure it — do not quote the engine saving as throughput.

The same finding holds in decode: the coalesced FP8 MoE is **Vector-bound**, and the thing to
optimize is the scale-adds, not the PSUM copies (partition packing there was a dead end).

## Activation quantization

Dynamic per-token FP8 activation quant was the accuracy risk we expected and it was **fine**:
a device test isolating the activation cast gave cosine 0.999259 with FP8 weights and FP32
reference activations. Default to per-token absmax; gate at per-layer cosine ≥ 0.990 plus a
coherence check against BF16.

## Capacity arithmetic worth reusing

Per rank, one MoE layer at H=2048 / I=512 / 32 local experts: `2·H·I + I·H` = 3.15 M params per
expert → ~100 MB per rank per layer in FP8, ~4 GB across 40 layers. That is the whole FP8 win,
and it is what let BS=16 load where BF16 OOMs.

FP8 **KV cache** was tried and failed — do not assume the weight result transfers to KV.
