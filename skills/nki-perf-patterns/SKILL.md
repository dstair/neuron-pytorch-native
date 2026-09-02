---
name: nki-perf-patterns
description: |
  Turn a Trainium profile into a ranked list of fixes, and apply the kernel-level
  optimizations that actually paid off. Use when asked to "optimize this kernel", "make it
  faster", "find low hanging fruit", "what should I fix first", "is this at its floor", or
  when you have engine/op attribution and need to know what to do about it. Covers the
  remedies for fixed-cost ops, wasted engine width, tiny-matmul / LDWEIGHTS-bound loops,
  PSUM accumulation vs per-chunk scaling, partition-dim packing, and a ledger of levers
  that were MEASURED AND REFUTED so they are not re-chased. For FP8/quantization specifics
  see fp8-quantization-perf; for MoE routing/blocking see moe-kernel-perf.
---

# From attribution to a fix

Read `neuron-model-profiling` first — the numbers this skill acts on come from there, and its
traps will otherwise hand you a confident wrong answer.

## Rule 0: bound the payoff before you build

Compute the ceiling first, in one line, and write it down:

- **Serialization gap.** If `total_time − max(engine active)` is ~60% of the step, you are
  latency-bound. Removing X ms of engine work does **not** buy X ms of wall time. Say what it
  might buy and why.
- **Next-engine floor.** Cutting the top engine only helps down to the second engine. Ours:
  Vector 116.5 ms with GpSimd at 114.1 → a 25% Vector cut moved the pipeline ceiling ~2 ms.
  The remaining hope is shortening serial dependency chains, which is a *different* argument
  and must be made explicitly.
- **MFU/MBU.** Both under ~15% means neither compute nor bandwidth is the wall; a 2× faster
  matmul buys ~0. This is how three separate FP8-as-speed proposals were correctly killed.
- **Pre-register a gate.** Write the threshold that would make the lever worth building
  *before* measuring ("expert-weight DMA > 15–20% of wall"). One of ours failed by 3×, which
  saved building it. A gate you invent afterwards is not a gate.

## Pattern 1 — fixed-cost ops: reduce the COUNT, not the size

**Detect:** group instructions by opcode and look at the *standard deviation* of duration.
Near-zero std over hundreds of calls, especially across unrelated call sites, means a
fixed-overhead op. (`profile_queries.py` section 3 does this automatically.)

Measured on trn2, all with coefficient of variation < 1%:

| op | fixed cost | note |
|---|---|---|
| `EMBEDDING_UPDATE` | **10,879 ns ± 6 ns** | indirect scatter-accumulate; 10% of a prefill wall at 4 per block |
| `NONZERO_WITH_COUNT` | **90,818 ns ± 541 ns** | identical across three unrelated kernels; std as low as **2 ns** at one site |
| `ACT_TABLE_LOAD` | **1,283 ns, std 0** | activation-table reload |

**Remedy:** the only lever is issuing fewer. Restructure the call: batch N logical items into
one invocation, hoist it out of a loop, or avoid needing it. Do **not** try to make the data
smaller — it will not move.

For `ACT_TABLE_LOAD` specifically: it fires when the activation function changes. Group
same-function activations so tables stay resident.

## Pattern 2 — wasted engine width (the 1-partition op)

**Detect:** compute time-per-element and compare with the engine's width (128 lanes). A
`tensor_scalar` over a `[1, 16384]` row took **8.68 µs** — about 1/90 of what the same element
count costs spread across partitions.

**Remedy:** reshape to `[128, N/128]`, or fold several logical rows into one strided op with a
per-partition operand (`operand0` may be a `[P,1]` tensor, so P items can share one
instruction). Check first whether the 1-partition layout is *semantically required* by a
consumer (a per-partition compaction op, for instance) — if so, the win is batching the
producers, not reshaping.

## Pattern 3 — scale/accumulate ordering: use PSUM, not per-chunk Vector

When a matmul's contraction is split into chunks and each chunk's PSUM result needs a scalar
multiply before accumulation, you pay one Vector op per chunk. If the multiplier can be made
**per output block** instead of per contraction chunk, all chunks accumulate in PSUM and you
scale **once**.

Measured: the per-chunk form was 26% of all Vector time in a 40-layer prefill, and Vector was
the top engine. At H=2048 with 256-wide chunks, moving to output-block scaling removes 7/8 of
those ops. See `fp8-quantization-perf` for the quantization-layout version of this (it is the
same trick, already shipped in our decode kernel).

## Pattern 4 — tiny matmuls: check whether the stationary operand is reusable

**Detect:** `LDWEIGHTS` count ≈ `MATMUL` count (ours: 275,372 vs 276,011) means one stationary
load per matmul — zero reuse — and a sub-microsecond average (0.38 µs) means the pipe is
dominated by setup.

**Then ask what the stationary operand is:**

- **Weights** → reuse is available. Hold the stationary resident and stream more moving columns
  through it (more tokens, more batch). Big win.
- **Data × data** (both operands derived from activations, e.g. `_mm(k_w, v_new)` in a
  recurrence) → **there is nothing to amortize; this is the floor.** Batch cannot widen it,
  because each batch element needs its own stationary. Say so and stop.

Ours was the second case, which is why DeltaNet — the single largest component at 546 ms of
instruction time — is *not* a lever. Confirming this took one look at the source line and saved
an optimization attempt.

## Pattern 5 — partition-dim packing for small-contraction kernels

A contraction of C=32 uses 32 of 128 partitions. Packing N independent streams block-diagonally
into P = N×32 recovers the width: measured **+12.5% at N=2, +19.4% at N=4**, bit-identical.

The cap is hardware: P ≤ 128 → N ≤ 4 at C=32. Having *more* independent streams available (e.g.
from a larger batch) does **not** allow a wider pack — don't propose N=8 at higher batch.

## Pattern 6 — tile every partition-major buffer to ≤128, but tile CONDITIONALLY

Any `[num_items, ...]` SBUF buffer where `num_items` is a problem-size count will compile at
small scale and fail at large with `memset dst partition dimension N exceeds maximum 128`.
Tile the initialization and give each chunk a unique op `name=`. Assume this bug exists in any
path that has only ever run at one size.

**Emit the tiling only when `num_items > 128`.** A tiling loop that runs a single iteration is
bit-identical but **not free**: measured **−3.0%** of a 40-layer prefill wall clock
(4,370.8 → 4,244.9 tok/s at BS=6, run-to-run spread 0.026%, identical fingerprint) from tiling
one route-packer metadata section whose loop ran exactly once. Three mechanisms, all SBUF
allocation, none visible in the op semantics:

- the added `sbm.open_scope`/`close_scope` pair,
- hoisting a buffer's allocation above the loop, lengthening its live range,
- renaming every buffer (`foo` → `foo_0`), which changes allocation order.

So structure it as `tiled = n > 128; tile = 128 if tiled else n`, keep the original op `name=`s
and the original allocation *positions* on the untiled path, and pass the tile base as `offset`
so the tiled path stays correct. Guard it with a comment: this looks like duplication a future
reader will want to collapse.

Generalizes past tiling: **"bit-identical" is a numerics claim, not a performance claim.** This
workload has repeatedly shown ±1–2% from allocation-order changes alone. A/B any restructuring
you believe is a no-op, at the config you ship.

## Correctness hazards that masquerade as performance work

- **Mutating one tensor twice in a traced graph loses writes** — silently, finitely, plausibly.
  Measured: with one shared cache tensor, only the first group's writes survived
  (`per_group_nz=[98304,0,0,...]`) and nine layers attended over zeros. Use one distinct
  tensor per mutation; `.clone()` not `.contiguous()` (a dim-0 slice of a contiguous tensor is
  already contiguous, so `.contiguous()` returns the same storage and changes nothing). Gate on
  buffer **occupancy**, not just finiteness.
- Declare aliasing by returning the mutated buffer; a kernel that mutates in place and doesn't
  return it may lose the functionalization dependency that orders it.

## Before proposing anything: check the refuted ledger

`references/refuted-levers.md` lists levers that were built or measured and **did not pay**,
with the numbers. Several were proposed more than once. Check it first.
