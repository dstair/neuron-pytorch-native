# Refuted levers — measured, did not pay

Check this before proposing an optimization. Several of these were proposed two or three times
by successive sessions that hadn't seen the measurement. Each entry: what was tried, the
number, and *why* it failed — the mechanism matters more than the verdict, because it tells you
when the verdict might not apply.

Hardware context: trn2, TP=8/LNC=1, Qwen3.6-35B-A3B prefill/decode unless stated.

## FP8 as a *speed* lever — refuted three times

| attempt | result | mechanism |
|---|---|---|
| FP8 experts to exploit 158 vs 79 TFLOPS | ~0 | MoE was 0-FLOP GpSimd (semaphore/DMA/gather); the expert GEMM already overlapped on Tensor |
| Expert-weight DMA as the wall | gate failed 3× | `weight_queue_bytes` = 2.05 GB of 32.18 GB HBM = **6.4%** of traffic at MBU 8.6–12.2%. Pre-registered gate was >15–20% |
| FP8 CTE prefill kernel, profiled | Tensor only 40% | Vector 64.6% and GpSimd 53.2% bound it; matmul was not the wall |

FP8 **is** a real capacity lever (5.08 vs 8.94 GB/core at 40L, bit-identical). It is not a
speed lever while MFU sits near 10%. See `fp8-quantization-perf`.

## Batch as a prefill throughput lever — refuted (capacity only)

At a fixed per-MoE-call token count, reshaping batch × chunk is **break-even**:
BS=4/chunk512 = 3,865.9 tok/s vs BS=2/chunk1024 = 3,920.3 (−1.4%, noise).

Worse if you shrink the chunk to raise batch: BS=16/chunk128 (79 chunks vs 20) = 3,495.9 tok/s,
**−14.7%** — per-chunk fixed costs (attention setup, per-chunk×layer collectives and launches)
scale with chunk count and outweigh the batch parallelism.

Best achieved with the MoE call decoupled from the bucket: BS=4/chunk1024 = 3,846.2 (−1.9%),
BS=16/chunk256 = 3,675.1 (−6.2%). MFU did improve (8.2% → 11.9%), which is exactly what
"latency-bound" predicts.

**CORRECTED 2026-08-22 — "throughput is monotone in bucket size" is FALSE outside 128–1024.**
That rule was measured at small buckets where per-chunk fixed costs dominate. Bucket size also
sets the **intra-chunk attention work, which grows LINEARLY with the bucket**: summed over
chunks, within-chunk causal attention is `n_chunks · chunk²/2 = S·chunk/2`, while the
cross-chunk term is ~`S²/2` and bucket-independent. So 256→1024 is 4× the intra-chunk work and
1024→2048 is 8×. Measured at 40L/seq10k with tokens-per-call held at 6,144:
BS=6/chunk1024 (10 chunks) = **4,383.9** vs BS=3/chunk2048 (5 chunks) = **4,187.5 (−4.5%)**.
Halving the chunk count LOST. There is an optimum bucket (~1024 here), not a monotone trend.

Note you cannot vary bucket independently of batch at fixed tokens-per-call, since
`tokens/call = BS × bucket` — so that A/B moves both, and "flat in batch" and "monotone in
bucket" cannot both be read off it.

## Compiler optimization level — no effect

O1 / O2 / O3 on FP8 prefill: 35,388 / 34,603 / 34,831 tok/s — flat within ±2%, **O1 marginally
best**, while compile time grows ~3/5/7 min at 4 layers. Confirmed independently at 40 layers.
O2/O3 also perturb low-order FP sums (reassociation) without changing tokens.

Do not sweep optlevel again for throughput. `--optlevel 1` is also the fix for a decode
host-OOM, so it is the default for a reason.

## Vector PSUM→SBUF copy rerouting — negative

Prefill was Vector-bound via `_mm`/`_T` PSUM→SBUF copies, so the copies were rerouted off
Vector: **−6.4%**. Do not land. The real lever there was the serialization gap.

Superseded anyway by a topology change (LNC=2 → LNC=1) that cut Vector 73.34 → 32.47 ms
(−56%) and moved it from #1 to #3 engine — a reminder that engine rankings are configuration
dependent, so re-profile after any topology change instead of trusting an old attribution.

## Partition packing of the coalesced FP8 MoE — dead end

Confirmed dead end; the win in that kernel was optimizing the **scale-adds**, not the PSUM
copies.

## `--graph-splits` for decode — no-op

A no-op under `DECODE_FULLGRAPH=1`. The real lever against the compiler host-OOM was
`--optlevel 1`.

## Swap / paging for a decode host-OOM — never the lever

Repeatedly attempted; `--optlevel 1` is the fix.

## "Tokens per MoE call" as the CTE hang variable — refuted, and the real cause found

The whole ">64 blocks" → ">2,048 tokens/call" chain was a **correlation**. Exceeding
`_DIRECT_ROUTE_MAX_ASSIGNMENTS` (16,384 = 2,048 tokens at K=8) switches our route packer from
the direct `nonzero_with_count` path to a tiled stable scan, and **the tiled scan deadlocks at
any size** — forcing it at a known-good 2,048 tokens hangs with the identical `NRT status 5`
signature. Single-variable: `core_barrier` lives only inside the tiled scan, so the working
direct path is barrier-free too.

Consequences: the deadlock is **ours** (`kernels/moe_cte_35b.py`), not nkilib's — the drafted
escalation was withdrawn. And it retro-explains why control-flow rewrites, graph splits, and
isolated kernel repros all came back null: none of them touched the packer.

Also refuted along the way: forcing `scatter_barrier=True` on the tiled path is **impossible**,
not merely unhelpful — `core_barrier() requires LNC degree >= 2`.

**FIXED 2026-08-21** — the deadlock was `nisa.local_gather` in the tiled slice body, the only
tiled-exclusive GpSimd op. Replaced by a masked reduce over the routing `matches` one-hot:
bit-identical, and >2,048 tokens/call now runs (BS=3 = 3,072 and BS=4 = 4,096 both complete).
Refuted on the way there, so don't re-try any of them: `affine_range`→`sequential_range` on the
slice loop, metadata corruption (packer gate is 192/192 bit-exact), DMA volume (it hangs at ONE
2,048-assignment tile), and the `oob_mode.skip` OOB-notification storm.

## Control-flow rewrites to fix a hang — all three refuted

`nl.dynamic_range` (deprecated) → fully-static `sequential_range` unroll → `nl.fori_loop`
(new MLIR tracer). All hung identically at the same threshold. The static unroll also compiled
far slower and would blow instruction count at 40 layers.

The `fori_loop` migration was landed anyway as **deprecation cleanup** (bit-identical,
65.7 ms vs 67.1 ms at 4L = noise) — but it fixes nothing. Don't re-attempt as a hang fix.

## Chasing a runtime OOB by clamping gather indices — no effect

Clamped the index tensor of **every** `vector_offset` indirect DMA on the path, individually
and all together: the OOB count stayed at **exactly 1667**, unchanged. Then proved via the
allocation manifest that every tensor was correctly sized and every gather already in bounds —
which is *why* clamping was a no-op.

Lesson: before clamping, prove the index is actually out of range. And the OOB turned out not
to be the blocker at all — the hang was, and it reproduced with **zero** OOBs in BF16.

## `NEURON_SCRATCHPAD_PAGE_SIZE=512` to clear a hang — no effect

Tried against the block-count hang; no change. (Page size 256 remains the working default.)

## Tiled convolution in DeltaNet decode — sign flips with batch size

`DN_TILED_CONV`: **+16.8% at BS=8, −37% at BS=16.** A kernel-level win can invert with shape.
Always re-measure at the batch you ship, and never generalize a single-batch measurement.
The accompanying microbenchmark was dispatch-bound and did not predict either result.


## Unconditional partition-tiling of a section that fits — costs 3.0%

Not a lever that failed; a *robustness* change that quietly cost throughput, which is the same
trap from the other direction. Tiling the CTE route packer's partition-major block-metadata
section to ≤128 partitions lifted a real compile ceiling (BS=8 / `max_blocks=160` had failed
with `iota dst partition dimension 160 exceeds maximum 128`) and was shipped as "buys ~zero
throughput". Nobody tested whether it **cost** any.

At the shipping config the loop runs one iteration and is bit-identical. It was still
**−3.0%**: 4,370.8 → 4,244.9 tok/s at BS=6/chunk1024/40L/seq10k, two tiled runs at 0.026%
spread, identical fingerprint. Cause is SBUF allocation only — an added alloc scope, a hoisted
live range, and renamed buffers changing allocation order.

Fix is conditional emission (`> 128` only), not reverting the ceiling fix. See Pattern 6.

**The transferable claim: "bit-identical" says nothing about performance.** And a change
justified as robustness still needs a throughput A/B at the config you ship, because "should be
free" is a prediction, and this workload has falsified that prediction at ±1–2% several times.

## Extrapolating a saturating curve past its last point — burned twice in one session

2026-08-22, tokens-per-MoE-call at 40L/seq10k: 2,048 → 4,096 = **+3.3%**; 4,096 → 6,144 =
**+2.2%**; 6,144 → 8,192 = **−2.6%**. I predicted +1.5% for the last step from the trend and had
the **sign** wrong. Padding was not the cause (it *improves*, 25% → 20%); the likely cause is the
tiled packer's own O(assignments) cost and the doubled activation working set.

Same session, same mistake shape: the 4-layer A/B put a lever at +11% where 40 layers said
+3.3%. **A monotone-looking trend over two points predicts nothing about the third.** Measure the
config you intend to ship.
