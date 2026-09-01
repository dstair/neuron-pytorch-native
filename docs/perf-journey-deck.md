# 259 → 4,098 tok/s prefill, 15 → 681 tok/s decode

## What actually moved the needle on a 35B MoE on one Trainium2

Deck outline. 9 slides. Audience: teams bringing a large model up on Trainium who want to know
which levers pay and in what order.

All numbers measured on `trn2.3xlarge`, Qwen3.6-35B-A3B (256 experts, top-8, ~3B active of 35B)
with a DeltaNet + GQA hybrid backbone. Prefill = N=20,000 prompt, aggregate tok/s. Decode =
synchronized 30–50 iteration timer. Source: `contrib/qwen3.6-35b-a3b/BENCHMARK.md`.

---

## Slide 1 — Where we started, where we landed

```
                        START                 FINISH            multiplier
   PREFILL     eager, no kernels  29 tok/s              →   4,098 tok/s      140×
               first bucketed path 259 tok/s            →   4,098 tok/s     15.8×
   DECODE      BS=1 masked-dense   15 tok/s             →      49 tok/s      3.2×
               high-batch path                          →     681 tok/s     45×  (vs BS=1 start)

   Nothing here is a rewrite in a faster language. It is:
     6 kernel/graph changes, 1 topology change, 1 aliasing bug found twice,
     1 capacity unlock, and 2 free compiler upgrades.
```

**Pick your headline honestly.** "A few hundred → 4,000" is the **15.8×** story and it is the
defensible one for a talk: 259 tok/s was the first path that completed a 20k prompt at all.
The 140× includes the eager baseline (68 s to prefill 2,000 tokens), which nobody would ship.

Two structural lessons up front, because they recur on every slide:

- **The biggest wins were not arithmetic.** They were *graph shape*, *data movement*, and
  *serialization*. The single largest single-lever win (+96%) removed no math at all.
- **Two of the top wins came from the same root cause found twice** — once as a correctness bug,
  once as a throughput bug. Slide 8.

---

## Slide 2 — The prefill ladder

```
  tok/s (N=20,000 prompt, aggregate)
       0        1000        2000        3000        4000
       ├─────────┼───────────┼───────────┼───────────┤
    29 ▏                                               eager, no kernels        68 s / 2k tok
   259 ██                                              + bucketing + flash-GQA
       │                                                 & chunked DeltaNet kernels    8.8×
   958 ███████                                         + compiled buckets, CTE MoE     3.7×
  1120 ████████▌                                       + CTE-GQA (used-KV-prefix)     +17%
  1483 ███████████▌                                    + NKI route packer in-kernel   +29%
  2090 ████████████████▎                               + BS=2 homogeneous batching     +41%
  2277 █████████████████▋                              + DeltaNet C32 (stable inverse) +8.5%
  2776 █████████████████████▌                          + C32 n=4 pack, transpose-once +22%
  3457 ██████████████████████████▊                     + TP=8/LNC=1, vocab-sharded  +24.5%
  3645 ████████████████████████████▎                   + beta-4 container (FREE)      +5.4%
  4002 ███████████████████████████████                 + one KV buffer per GQA layer  +9.8%
  4098 ███████████████████████████████▊                + beta-5 container (FREE)      +2.4%
```

No single lever dominates. **Six of the eleven steps are between +8% and +30%** — the result is
compounding, which means the discipline to keep a clean A/B harness matters more than any one idea.

---

## Slide 3 — The decode ladder runs on two tracks

```
  BS=1 LATENCY track (seq=2048)          BS=128 THROUGHPUT track (40L, seq=256)
  ───────────────────────────            ────────────────────────────────────────
  15.1 ▏    masked-dense MoE              294 ███      full graph + vocab-sharded head
  30.0 ██   + true-sparse MoE    2.0×     304 ███      + direct recurrent-state output
  30.5 ██   + DeltaNet micro-opt          321 ███▎     + stateful K/V cache
  40.9 ███  + GQATAIL           +34%      322 ███▎     + FP8 experts → BS=128 FITS
  43.2 ███  + DNBATCHED_V2                632 ██████▌  + one state buffer per layer +96%
  48.9 ████ + TP-within-experts +13%      681 ███████  + beta-5 container          +7.8%
            └─ 3.24× total                             └─ 2.1× on top of the batch path
```

**The two tracks want opposite things.** True-sparse MoE wins only at BS≤4 (it gathers `T·K`
experts, and `T·K ≥ 64` once batch grows); above that, masked-dense grouped GEMMs amortize better.
`MOE_DECODE_TP` is BS=1-only by construction.

**Don't over-read a sparse MoE win.** True-sparse gave ~2×, not ~8×, because the expert GEMMs are
only about half the step — DeltaNet, GQA, projections, norms and all-reduces are the rest. Amdahl
applies to the thing you are proudest of.

---

## Slide 4 — Levers 1–2: stop being eager, and stop reading KV you don't need

```
  (1) EAGER → BUCKETED + KERNELIZED                        29 → 259 tok/s    8.8×
      fixed-shape sequence buckets, compiled once, reused per chunk
      → an eager 20k prefill is not slow, it is a different order of magnitude

  (2) COMPILED BUCKETS + CONTEXT-ENCODING ATTENTION       259 → 1120 tok/s   4.3×

      local fixed-KMAX flash kernel        CTE attention (visits used KV prefix only)
      ┌──────────────────────────┐         ┌──────────────────────────┐
      │ K/V: ████████████████████│ 20k     │ K/V: ████░░░░░░░░░░░░░░░░│ used prefix
      │ every chunk reads ALL    │         │ chunk n reads only 0..n  │
      └──────────────────────────┘         └──────────────────────────┘
        11.66 – 11.69 ms / call       →      0.77 – 0.81 ms / call     ≈ 15× ON THE KERNEL
                                            (at full depth: 958 → 1120 tok/s)
```

**Why the 15× kernel win is only +17% end to end:** attention was not the whole step. This is the
recurring shape of the whole project — measure the *step*, not the kernel, before promising.

Also on this rung: it took fixing two kernel bugs (pad-token DeltaNet state corruption, and an
L2-norm epsilon mismatch on near-zero rows) before *any* compiled number was trustworthy.
Correctness gates first, or you optimize noise.

---

## Slide 5 — Lever 3: a harmless-looking torch op cost 24.9 GB of HBM

The MoE routing metadata was built in torch, inside the compiled graph.

```
  BEFORE   sel ──▶ one_hot() ──▶ cumsum(dim=0) ──▶ … ──▶ CTE MoE
                                    │
                                    └─ Neuron lowers this to HLO `reduce-window`,
                                       backed by TensorE MATMUL + LDWEIGHTS

     matched 10-layer segment:   route scan   61.3 ms (BS=1)  →  258.3 ms (BS=2)
                                 route HBM    24.9 GB         →  105.7 GB
     → the metadata scan and DeltaNet together explained 99% of a 2.5× BS=2 regression

  AFTER    sel ──▶ [ NKI stable compaction  ──▶  CTE MoE ]   one custom call

           route HBM    24.94 GB  ─────────────▶  45.9 MB     (543×)
           total HBM    31.96 GB  ─────────────▶   3.30 GB
           `reduce-window` instruction: gone
           1,151 → 1,483 tok/s   (+29%)
```

**Transferable:** profile your *bookkeeping*, not just your math. `one_hot().cumsum()` is three
words of Python and was the single largest HBM consumer in the graph. Nothing about the source
hints at `reduce-window` on the Tensor engine — only the profile showed it.

Gate used: 96 exact metadata cases across both token counts, both block sizes, all four TP expert
ranges, plus distributed output matching the precomputed path exactly.

---

## Slide 6 — Levers 4–5: batch the prompts, then fix the chunk/partition geometry

```
  (4) HOMOGENEOUS BATCHING  BS=1 → BS=2                  1483 → 2090 tok/s   +41%
      independent DeltaNet / conv / KV state per prompt, ONE custom call per layer
      gate: 4-layer BS=2 vs independent BS=1 runs, cosine ≥ 0.999936 on logits + all states

  (5) DELTANET CHUNK GEOMETRY                            2090 → 2776 tok/s   +33%

      C16 → C32: halves the chunk count                          +8.5%  → 2277
        ⚠ the naive full-32 inverse OVERFLOWS on near-1-decay streams.
          Required a stable inverse: split the 32×32 into two 16×16 diagonal
          blocks + a coupling term (a Horner series is stable but ~4× costlier).

      then pack 4 independent C32 streams block-diagonally into the partition dim:

        C=32 uses 32 of 128 partitions        pack n=4 → P = 4×32 = 128
        ┌────┬────────────────────┐           ┌────┬────┬────┬────┐
        │ s0 │      IDLE 96       │           │ s0 │ s1 │ s2 │ s3 │
        └────┴────────────────────┘           └────┴────┴────┴────┘
                                              +12.5% at n=2, +19.4% at n=4, bit-identical
                                              CAP: P ≤ 128 → n ≤ 4. More streams do NOT help.
```

**Two lessons.** A faster numerical scheme can be arithmetically unusable — budget for the stable
variant. And partition-dim occupancy is a first-class resource: a kernel using 32 of 128 lanes is
leaving 4× on the table regardless of how good its math is.

---

## Slide 7 — Lever 6: the topology change, and the misdiagnosis that hid it

```
   TP=4 / LNC=2                          TP=8 / LNC=1
   4 logical cores, 24 GB HBM each       8 logical cores, ~12 GB each
        2,775.6 tok/s                         3,456.8 tok/s      +24.5%

   Blocked for weeks by "LNC=1 doesn't work on this model."
   Reality: a 30-line shard-count guard. The actual wall was HBM per rank:

     two REPLICATED [V, H] vocab tensors, ~1.02 GB each
       lm_head  ████  1.02 GB  ──shard at LOAD──▶  █  0.13 GB
       embed    ████  1.02 GB  ──shard at LOAD──▶  █  0.13 GB
     + scratchpad page size 256  →  40 layers fit in ~12 GB/rank

   Bonus, visible only in the profile:
     LNC=2  tensor_engine_instruction_time = 100.1% of total_time  ← Tensor WAS the critical path
     LNC=1  max engine instruction share    =  71.3%               ← no engine owns it any more
```

**Transferable:** when a configuration is written off as "doesn't work", check whether that is a
measurement or a memory. This one was inherited as folklore and cost weeks. Re-test cheap
configuration claims before you build around them.

Note the profile consequence: the topology change *moved the bottleneck*, so every engine-level
attribution taken before it was stale. Re-profile after any topology change.

---

## Slide 8 — Levers 7–8: the same aliasing bug, twice, worth +9.8% and +96%

One shared base tensor, mutated once per layer inside a traced graph:

```
      layer 0  ─┐
      layer 1  ─┼──▶   [ NUM_LAYERS, B, … ]        every write aliases ONE base
        …       │
      layer 29 ─┘

  PREFILL — GQA KV cache                  DECODE — DeltaNet recurrent state
  ────────────────────────                ────────────────────────────────────
  writes SILENTLY LOST                    writes kept, but SERIALIZED
  occupancy [98304, 0, 0, …, 0]           ~30 write-after-write dependencies
  9 of 10 layers attended over ZEROS      chained through a single buffer
  output stayed finite and plausible      output was always correct
  ~60% of KV writes dropped               just slow

  FIX (identical in both cases): one distinct tensor per mutation, held in a list,
       stacked at return.  Use .clone() — NOT .contiguous():
       a dim-0 slice of a contiguous tensor is ALREADY contiguous, so .contiguous()
       returns the same storage and all N "buffers" still share one base.

  → +9.8% prefill (3,645 → 4,002) AND correct     → +96.3% decode (322 → 632), bit-identical
```

**This is the most transferable slide in the deck.** The same structural mistake presents as a
silent correctness bug in one place and a 2× throughput loss in another. Neither is visible in the
output values.

**So gate on occupancy, not plausibility.** A matching greedy generation hash agreed across the
*defect* and across the *fix* — it is a floor, not proof. What caught it: asserting every buffer
you expect to be written has non-zero rows.

---

## Slide 9 — Levers 9–10, and the order to work in

```
  (9) FP8 EXPERTS = CAPACITY, NOT SPEED
      BF16 decode ceiling was BS=32 (BS=64 OOM'd on the next 240 MB state tensor).
      FP8 experts made BS=128 fit → that unlocked the whole high-batch track.
      On prefill, FP8 costs ~7% and buys 8.94 → 5.08 GB/core.  See the FP8 deck.

 (10) FREE COMPILER WINS — re-benchmark on every new container
      beta-4  3,457 → 3,645 prefill  (+5.4%)   identical source, bit-identical output
      beta-5  4,002 → 4,098 prefill  (+2.4%)   and decode 632 → 681 (+7.8%)
      ~16% of the final prefill number came from doing nothing but pulling an image.
```

**Recommended order for a new model bring-up:**

```
  1. Correctness gates + a clean A/B harness FIRST.        (two kernel bugs blocked slide 4)
  2. Get out of eager: fixed-shape buckets, compiled.      biggest single multiplier
  3. Profile the STEP. Attribute to engines and ops.       expect bookkeeping, not math
  4. Fix data movement before arithmetic.                  543× on route HBM (slide 5)
  5. Check geometry: chunk counts, partition occupancy.    32 of 128 lanes is 4× left over
  6. Re-test inherited "that doesn't work" claims.         slide 7 cost weeks
  7. Audit every in-place mutation for shared storage.     slide 8, twice
  8. Quantize for CAPACITY to unlock batch, not for speed. slide 9
  9. Re-benchmark on each new container.                   free, bit-identical
```

**What did NOT work** (so you can skip it): batch for prefill throughput (break-even at fixed
tokens/call, −14.7% if you shrink the chunk to raise batch), `--optlevel` O2/O3 (flat ±2%),
rerouting Vector PSUM→SBUF copies (−6.4%), FP8 for matmul speed (refuted 3×), FP8 KV cache. The
full ledger with mechanisms is in `skills/nki-perf-patterns/references/refuted-levers.md`.

---

### If you need 6 slides
Keep 1, 2, 5, 8, and merge 4+6 into "graph shape and geometry". Slides 5 and 8 are the ones
people can act on tomorrow; 2 and 3 are the credibility.

### Caveat to state out loud
Every number is one model, one SDK, one instance type. The *mechanisms* transfer — eager overhead,
op lowering, partition occupancy, aliasing, capacity-driven batch — the *percentages* do not.
Slide 3's two-track structure and `DN_TILED_CONV` (+16.8% at BS=8, **−37%** at BS=16) are the
reminders that a lever can invert with shape.
