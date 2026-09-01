# FP8 on Trainium2 — what it buys, what it costs, and the one optimization that matters

Deck outline. 9 slides. Audience: teams considering or already running FP8 on trn2 who would
rather not repeat six weeks of trial and error.

Measurements are from Qwen3.6-35B-A3B (35B MoE + DeltaNet/GQA hybrid), 40 layers, seq 10k,
TP=8 / LNC=1, SDK 2.32. Cite the config whenever you quote a number — several of these
reverse with configuration.

---

## Slide 1 — TL;DR: three numbers and a warning

```
   FP8 on trn2, measured end to end on a 40-layer model:

   ┌─────────────────────────────────────────────────────────────────────┐
   │  MEMORY   8.94 → 5.08 GB/core     -44%   ✓ real, and BIT-IDENTICAL  │
   │  SPEED    4,227 → 3,920 tok/s     -7.3%  ✗ a tax, not a win         │
   │  CAUSE    of that 7%: ~26% of Vector time in block-dequant scaling  │
   │                       ...which is removable. Slide 6.               │
   └─────────────────────────────────────────────────────────────────────┘

   The 2x FP8 matmul rate?  Bought us nothing. Three times. Slide 2.
```

**Plan FP8 as a capacity feature.** It fits configurations BF16 cannot (BS=16 at 40 layers
loads in FP8 and OOMs in BF16). Treat any throughput gain as unproven upside.

What this deck saves you, roughly in order of pain:
- 3 separate attempts to get speed out of the 2× matmul rate (slide 2)
- a dtype that no toolchain component accepts, plus a workaround flag that isn't wired up (slide 4)
- 4 scale layouts, only one of which is numerically exact (slide 5)
- a profiling method that silently under-reports the thing you're optimizing (slide 7)
- ~15× wrong timings from debug flags, and stale-cache runs that measure nothing (slide 9)

---

## Slide 2 — Expectation setting: the 2× that isn't

```
   Trn2 Tensor Engine peak            Where a 40L FP8 prefill actually sits
  ┌──────────────────────────┐
  │ FP8   158 TFLOPS ███████ │        MFU   ██                    11.9%
  │ BF16   79 TFLOPS ███     │        MBU   █                      9.4%
  └──────────────────────────┘        wait  ████████████████████  79%
        "free 2x"                            └─ *SEMAPHORE ops, all engines

   Doubling the peak doubles a number nobody is spending.
```

Three independent attempts to cash the 2×, and why each failed:

| attempt | result | mechanism |
|---|---|---|
| FP8 experts for the 2× GEMM | ~0 | MoE was 0-FLOP GpSimd (semaphore/DMA/gather); the GEMM already overlapped |
| "expert-weight DMA is the wall" | gate failed 3× | weight traffic = 2.05 of 32.18 GB HBM = **6.4%**, at MBU 8.6–12.2%. Pre-registered gate was >15–20% |
| FP8 CTE kernel, profiled in isolation | Tensor only 40% | Vector 64.6% / GpSimd 53.2% bound it |

**Transferable rule:** pre-register the gate that would justify the work, *before* measuring.
"Expert-weight DMA > 15% of wall" failed by 3× and saved building the kernel. A gate invented
afterwards is not a gate.

**Check your own workload first:** if MFU and MBU are both under ~15%, you are latency-bound and
a faster matmul buys ~0. That is a 10-minute measurement (slide 7).

---

## Slide 3 — What FP8 actually buys: capacity

```
   HBM per core, 40 layers, experts resident
   BF16  ████████████████████████  8.94 GB    BS=16 → does not fit
   FP8   ██████████████            5.08 GB    BS=16 → loads, identical tokens
                                   └── -44%, and the output is BIT-IDENTICAL

   the arithmetic, per rank (H=2048, I=512, 32 local experts of 256):
     per expert   2·H·I + I·H       = 3.15 M params
     per layer    × 32 experts      = ~100 MB in FP8
     × 40 layers                    = ~4.0 GB  ←  the entire win
```

This is what unlocked configurations BF16 could not reach. Use it for:
- higher batch, longer context, or more resident layers
- fitting a model that otherwise needs another node

**Caveat that cost us a day:** the weights fitting does not mean the *run* fits. At BS=16 ×
chunk1024 the module loaded at 5.08 GB/core and then failed with
`Failed to allocate DEVICE memory (95079808 bytes)` — the **instruction/staging buffer**, not
weights. Cap in-flight activations (batch × sequence chunk), not just weight bytes.

---

## Slide 4 — Making it compile at all (trn2-specific, version-bound)

```
   checkpoint             host prep                  inside the NKI kernel
  ┌───────────────┐   ┌──────────────────┐   ┌──────────────────────────────────┐
  │ F8_E4M3(FN)   │──▶│ requantize_*_pow2│──▶│ w = w_int8.view(nl.float8_e4m3)  │
  │   max 448     │   │  → int8 bytes    │   │ nisa.nc_matmul(..., FP32 accum)  │
  │ scale_inv BF16│   │  + BF16 scales   │   └──────────────────────────────────┘
  └───────────────┘   └──────────────────┘
                        ▲ legacy E4M3 max = 240, not 448
                          → halve the block, double its scale (exactly)

   ✗ torch.float8_e4m3fn as an operand → NCC_EVRF051
     rejected by NKI *and* by neuronx-cc on TRN2
   ✗ --experimental-unsafe-fp8e4m3fn-as-fp8e4m3 → not forwarded by this driver
   ✓ store int8, reinterpret in-kernel: the HLO operand stays int8
```

**The 2× double_row path has exact requirements.** Miss any one and you silently get the 1× path:

```
   both operands E4M3/E5M2      ← a BF16 × FP8 mixed matmul gets NOTHING
   contraction: partition=128 × free=2
   FP32 accumulate, perf_mode="double_row"
   block quantization at 256×256 granularity
```

We had a shipping "FP8 kernel" that was BF16 × FP8 mixed and never had the 2× at all. **Check
operand dtypes before believing a kernel is on the fast path.**

MX / microscaling (E8M0) is a *different format*: the nkilib MX CTE path is TRN3-only, and the
newer MXFP8 kernels are training-backward only. Do not plan a TRN2 inference path around them.

---

## Slide 5 — Choose your scale layout deliberately

```
                    ┌─ must stay faithful to the official grid?
                    │      └─ YES → pow2 exponent shift        (EXACT)
   official         │
   128×128    ──────┼─ accuracy is the binding constraint?
   scale grid       │      └─ YES → dual plane      (2× bytes, 2× matmuls)
                    │
                    └─ want the Vector tax gone? (slide 6)
                           └─ YES → output-block    (coarser, needs a cosine gate)

                    ( row / per-output-channel = coarsest. We rejected it. )
```

| layout | numerics | cost |
|---|---|---|
| **pow2 exponent shift** | **exact**; bit-for-bit for untouched blocks | none |
| **dual plane** | best | 2× weight bytes, 2× matmuls |
| **output-block** | coarser — gate it | removes 7/8 of the dequant ops |
| row | coarsest | rejected here |

**Gate every conversion, and print the stats at load:** cosine, normalized RMSE, shifted-block
fraction, and **clipped count must be 0** or raise. Ours: dequant cosine 0.9996; device unit test
0.999259 against the block-quant reference; 4-layer end-to-end gave **identical top-5 tokens**
with norm 945.56 vs 944.18 (0.15%).

That pair — a unit cosine *plus* an end-to-end coherence check — is the gate to reproduce. A
matching generation hash is **not** a gate: ours agreed across a real defect *and* across its fix.

---

## Slide 6 — THE optimization: where block-quant scaling costs you

Per-**contraction-block** scales force the contraction into chunks, and every chunk's PSUM result
must be scaled *before* it can be accumulated. That is one Vector op per chunk. BF16 pays none of
it. **This is the 7% tax.**

```
BEFORE  — one scale per contraction chunk        H = 2048, in 8 chunks of 256

  chunk      [0:256]   [256:512]    ...     [1792:2048]
               │           │                     │
  PSUM      ┌──▼──┐     ┌──▼──┐               ┌──▼──┐
            │ mm  │     │ mm  │      ...      │ mm  │      8 matmuls
            └──┬──┘     └──┬──┘               └──┬──┘
  VECTOR     ×s0 +       ×s1 +        ...      ×s7 +       8 Vector scale-adds  ✗
               └───────────┴─────────────────────┘
                          SBUF accumulator

AFTER   — one scale per OUTPUT block             (requantize_official_fp8_output_block)

  PSUM      ┌─────┐     ┌─────┐               ┌─────┐
            │ mm  │────▶│ mm  │──── ... ────▶│ mm  │      same 8 matmuls,
            └─────┘     └─────┘               └──┬──┘      accumulating IN PSUM
  VECTOR                                        ×S         1 Vector scale-add   ✓
                                                 ▼
                                               SBUF

            removes  H/256 − 1  =  7 of 8  Vector ops
```

Measured, 40-layer BS=16 prefill (`bwmm_shard_on_I.py:1384`, `:2133`, `:2140`):

```
   62,304 Vector instructions   38.15 ms   =  26% of ALL Vector time
   Vector was the TOP engine     40.6% active
   ≈ 5.8 s of a 43.5 s prefill
```

**This is already implemented and shipped in our decode path** —
`requantize_official_fp8_output_block` + `nki_moe_fused_w8_fp8_block_coalesced_ob`, described in
code as *"Reduction B1: input-independent per-output-block scales, PSUM-accumulate."* Port it,
don't rediscover it.

**Generalizes past FP8:** any time a chunked contraction needs a scalar multiply per chunk before
accumulating, ask whether the multiplier can be made per-output-block instead. Then PSUM does the
accumulation for free.

---

## Slide 7 — How to find your own levers (and the trap that fakes the answer)

```
  compile cache            capture (collective replay)        parquet             answer
  ─────────────            ───────────────────────────        ───────             ──────
  neff_cache/*.neff   ──▶  neuron-explorer capture       ──▶  Instruction    ──▶  std(duration) ≈ 0
   pick by SIZE:            -r 8 -i 0 --single-io              OpcodeSummary       → FIXED-COST op
   size classes =           --profile-nth-exec=2                                  → the lever is
   regions × ranks          DGE notifs OFF at 40 layers                             COUNT, not size
```

**The one-line trick that found the biggest items:** group instructions by opcode and look at the
**standard deviation** of duration. Near-zero std across unrelated call sites = a fixed-overhead
op whose cost is set purely by how many times you issue it.

```
   EMBEDDING_UPDATE     10,879 ns ± 6 ns       ← 10% of a prefill wall, 4 per block
   NONZERO_WITH_COUNT   90,818 ns ± 541 ns     ← identical across 3 unrelated kernels
   ACT_TABLE_LOAD        1,283 ns ± 0 ns       ← activation-table reload
```

⚠ **The trap.** `capture` replays with **synthetic inputs**, so any loop whose trip count comes
from data (an MoE block loop driven by routing) executes an arbitrary, usually much smaller number
of iterations. In our 4-region capture the MoE appeared in **one** region: 29,920 instructions at
one source line in region A, **0** in region B. Summing the regions understated the MoE ~4× and
would have sent us optimizing the wrong subsystem.

```
   ALWAYS run the reconciliation gate:
     (per-region time) × (regions) × (steps)  ≈  measured wall ?

     4 × rA  = 44.8 s  vs  43.54 s measured   → 103%  ✓ valid unit
     sum of 4 regions = 35.2 s                →  81%  ✗ looked plausible, was wrong
```

Corollary: a capture that does **not** reproduce a data-dependent bug is not evidence the bug
isn't there. Synthetic routing gave 0 out-of-bounds events for a fault that fires every real run.

---

## Slide 8 — Bounding the payoff honestly (do this before you promise anything)

```
  rA engine ACTIVE  (of a 286.8 ms region)
  Vector  ████████████████████████  116.5   ← #1; 26% of it is FP8 scale-adds
  GpSimd  ███████████████████████   114.1   ← the floor a Vector-only fix reaches
  DMA     ███████████████████████   113.0
  Tensor  ███████████████████        95.4
  Scalar  ██████████                 50.5
  Sync    ████████                   38.6

  perfect_pipeline = max(active) = 116.5      total_time = 286.8
  serialization gap = 170.3 ms = 59%          *SEMAPHORE ops = 79% of the step

  cut Vector by 26%  →  ceiling moves 116.5 → 114.1  ≈  2 ms of 286.8
```

**So is slide 6 worth doing?** The engine-time argument alone says ~1%. The real case is that
removing 54,000 serialized Vector ops shortens dependency chains in a workload that is 79%
waiting — a *different* argument, which has to be measured, not asserted.

Say this out loud to your stakeholders. Three things to carry:

- **Cutting the top engine only helps down to the second engine.** Check the gap first.
- **A 59% serialization gap means engine savings convert to wall-clock only partly.**
- **Report what you measured, not what the arithmetic promised.** We published a −14.7% and a
  flat result rather than dressing them up; that is why the numbers here are trustworthy.

---

## Slide 9 — Don't re-chase these · and the order to work in

**Refuted, with numbers — check before proposing:**

| lever | result |
|---|---|
| FP8 for speed | refuted 3× (slide 2) |
| Batch for prefill throughput | break-even at fixed tokens/call; **−14.7%** if you shrink the chunk to raise batch |
| `--optlevel` O2/O3 | flat ±2%, O1 marginally best, 2× the compile time |
| Rerouting Vector PSUM→SBUF copies | **−6.4%** |
| Partition packing the coalesced FP8 MoE | dead end — the win was the scale-adds |
| FP8 **KV cache** | failed — the weight result does not transfer |
| Rewriting a runtime loop to fix a hang | 3 rewrites, all hung identically |

**Landmines that produce fake numbers:**

```
  ✗ NEURON_LAUNCH_BLOCKING / DGE notifications while timing  → ~15× slower; we published
                                                                two wrong results from this
  ✗ non-sudo rm of a root-owned compile cache  → fails SILENTLY, serves a stale NEFF
                                                 (grep the log for "0 compile activity")
  ✗ trusting the docker exit code  → teardown SIGABRTs *after* printing valid results;
                                     read your own marker instead
  ✗ aws ec2 stop-instances on the box  → /mnt/nvme is instance store. REBOOT, never stop.
```

**Recommended order for a new project:**

```
  1. Measure MFU / MBU / semaphore share first.        (10 min — slide 7)
     Both under ~15%? → FP8 is a CAPACITY project. Set expectations now.
  2. Size the memory win with the parameter arithmetic. (slide 3)
  3. Get it compiling: int8 + .view(), pow2 scales.     (slide 4-5)
  4. Gate numerics: unit cosine + end-to-end coherence. (slide 5)
  5. Profile, find YOUR fixed-cost ops and dequant tax. (slide 6-7)
  6. Bound the payoff before building.                  (slide 8)
```

**Reusable assets** (this repo, `skills/`): `neuron-model-profiling` (capture drivers +
`profile_queries.py`, which runs the whole battery including fixed-cost detection),
`fp8-quantization-perf`, `moe-kernel-perf`, `nki-perf-patterns` (full refuted ledger),
`trainium-run-discipline`.

---

### If you need to cut to 7 slides
Merge 2 into 1 (the TL;DR already carries "the 2× is not real"), and fold 8 into 6 as a closing
"what this is worth" box. Slides 4, 5, 6, 7 are the load-bearing ones — 6 and 7 are the reason to
give the talk at all.

### If you have 12 and an audience that will follow you into the MoE
Add: the ~32× FLOP cost of a masked-dense prefill bypass; the route packer as a first-class cost
(~11% of the step, with 8 × 90 µs fixed-cost calls per MoE invocation); and the co-varying
threshold case study — "it hangs above 64 blocks" was refuted by two 20-minute runs, and the
replacement explanation was *also* confounded. That last one is the best methodology story we
have, but it is an MoE talk, not an FP8 talk.
