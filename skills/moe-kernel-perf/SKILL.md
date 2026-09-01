---
name: moe-kernel-perf
description: |
  Make a Mixture-of-Experts layer fast and correct on Trainium. Use when working on MoE,
  experts, expert-parallel, top-k routing, the CTE / blockwise / moe_tkg / masked-dense MoE
  kernels, a route packer, block_size / max_blocks / token_position_to_id / block_to_expert /
  conditions, or when an MoE call hangs or OOBs above some batch or token count. Covers which
  MoE kernel to use for prefill vs decode, the blocking arithmetic, the fixed-cost packer and
  output-scatter overheads, why masked-dense costs ~32x routed FLOPs at prefill, and padding
  sentinel hazards. For FP8 scale layouts and the block-dequant tax see fp8-quantization-perf.
---

# MoE performance on Trainium

## Pick the right kernel

| kernel family | shape it serves | notes |
|---|---|---|
| **CTE / blockwise** (`moe_cte`, nkilib `blockwise_mm_*`) | prefill, many tokens | routed: computes only real assignments, packed into blocks. The only sane prefill choice at scale |
| **TKG** (`moe_tkg`) | decode, T small | maps tokens onto the partition dim → **T ≤ 128**; has a selective mode for T<16 |
| **token-tiled all-expert** (`_moe_fused_w8`) | decode, high batch | dense over local experts; see the 32× warning below |
| **masked-dense** (torch `moe_forward`) | reference / oracle | builds `[E, T, ...]`; OOMs at prefill T unless chunked |

A prefill kernel generally **cannot** meta-specialize T=1, so decode must use a different path;
budget for two MoE implementations sharing one set of packed weights.

## Blocking arithmetic (get this right before anything else)

```
assignments  = tokens_per_call * K                 # K = top-k, global
max_blocks   = ceil((assignments - E) / block_size) + E      # E = local experts
packed_len   = max_blocks * block_size
```

- The `+E` term is per-expert boundary padding and **dominates** at small block counts: at
  2048 tokens, K=8, E=32, block=512 you get 32 + 32 = 64 blocks, i.e. half the blocks are
  boundary padding.
- `block_size` must be a multiple of 256 (nkilib asserts `B % 256 == 0`). Smaller block_size
  means **less** padding waste. Our packer only accepts 256 or 512; a log2 shift constant and
  one assert are what limit it.
- Blocks ≥ 1024 use a **different** nkilib compute function (`compute_one_block_dropping`, which
  tiles by `MAX_BLOCK_TILE_SIZE = 1024`), so raising block_size past 512 is not a free knob.
- `tokens_per_call` is what actually matters, and it is **not** the sequence chunk: it is
  `batch × chunk` if the MoE sees the whole flattened chunk. Chunk the MoE call itself to
  control it (see below).
- Real executed blocks ≈ `on_rank_assignments / block_size + partial`, which is far fewer than
  `max_blocks`. Anything sized by `max_blocks` is worst-case; anything counted at runtime
  (`conditions`) is data-dependent — a distinction that matters enormously when profiling
  (see `neuron-model-profiling`'s synthetic-input trap).

### Decouple the MoE call size from the bucket chunk

If a wall forces a cap on tokens per MoE call, **cap the MoE call, not the bucket**. MoE is
per-token independent, so chunk the kernel invocation and keep the sequence chunk large:

```python
cap = int(os.environ.get("MOE_MAX_TOKENS", "0"))
if cap > 0 and T > cap:
    parts = [self._moe(x2d[cs:min(cs+cap, T)]) for cs in range(0, T, cap)]
    return torch.cat(parts, dim=0).reshape(*lead, HIDDEN)
```

Shrinking the *bucket* instead costs 14.7% (79 chunks vs 20: per-chunk attention setup and
per-chunk×layer collectives scale with chunk count). Chunking only the MoE was flat (−1.9%).

Keep one all-reduce per layer: chunk the kernel calls, concatenate, **then** reduce and add the
shared expert once — not once per chunk.

## The overheads nobody budgets for

Measured on a 40-layer BS=16 FP8 prefill, per representative region:

| item | cost | mechanism |
|---|---|---|
| **output scatter** (`EMBEDDING_UPDATE`) | **10% of total wall** | fixed **10,879 ns ± 6 ns** per call, **4 per block** (one per 128-token tile of a 512-token block, `NUM_B_TILES_SHARDED = B/TILE_SIZE`). A [128, H] indirect read-modify-write at ~94 GB/s effective |
| **route packer** | ~11% of region time | of which `NONZERO_WITH_COUNT` at fixed **90,818 ns**, issued `num_local_experts // 4` = **8 times per MoE call** |
| packer match/shift | 8.68 µs each | `tensor_scalar` over a `[1, assignments]` row — 1 of 128 Vector partitions |

The packer is ~19% of the nkilib MoE compute it feeds. **Routing metadata is a first-class cost,
not bookkeeping** — profile it explicitly instead of assuming the GEMMs dominate.

Both `EMBEDDING_UPDATE` and `NONZERO_WITH_COUNT` are **fixed-cost** ops: the only lever is
issuing fewer (see `nki-perf-patterns` pattern 1). For the scatter that means coalescing (needs
sorted routing) or accumulating top-k in SBUF.

### `nonzero_with_count`: the ISA gives you 8 lanes, not 4 and not 32

Before restructuring a packer around it, know the hardware shape (source:
`nkilib/core/subkernels/find_nonzero_indices.py`, which is in the nkilib checkout):

```
_NUM_GPSIMD_CORES      = 8    # 8 cores run in parallel -> 8 columns per call, MAX
_PARTITIONS_PER_GPSIMD = 16   # cores read partitions 0, 16, 32, ..., 112
_QUADRANT_SIZE         = 32   # 4 quadrants of 32; 2 GpSimd cores per quadrant
```

- **8 columns per invocation is the ceiling.** With E local experts the floor is `E/8` calls — for
  E=32 that is **4 calls, not 1**. Don't plan on collapsing to a single call.
- Even cores write results at partitions **0/32/64/96** (directly readable); odd cores write at
  **16/48/80/112** and need a **stream shuffle** (`quad_mask = [16] + [255]*31`) to extract.
- **That shuffle is MANDATORY, not an optimization — a 1-partition SBUF access must begin on a
  32-partition quadrant boundary.** MEASURED 2026-08-21: setting the packer's lane stride to 16
  to use all 8 cores fails BIR verification outright —
  `Invalid access of 1 partitions starting at partition 16`, on the per-lane
  `tensor_scalar` that builds the match mask. So partitions 16/48/80/112 are simply not
  addressable by a Vector op; the odd cores are reachable *only* through the shuffle, on both
  the input and the result. Against a ~2%-of-region ceiling that did not pay, so a
  **32-partition, 4-lane stride is the right design** — it is not the oversight it looks like.
- Corollary: don't read `_PARTITIONS_PER_GPSIMD = 16` as "you may address any 16-aligned
  partition". It describes the GpSimd engine's internal core mapping, not what the other
  engines can address.
- **`find_nonzero_indices` is a complete subkernel** for exactly this ("indices of nonzero elements
  along T, for each column; up to 65,536 tokens and 128 columns"). `core/moe/moe_tkg/all_expert_impl.py`
  is a second in-repo consumer of the raw ISA. Evaluate reuse before hand-rolling.

Packer fixed costs are **per call**, so larger MoE calls amortize them better — an argument for
raising tokens-per-call if a wall permits.

## Masked-dense at prefill costs ~32× routed FLOPs — do the arithmetic first

A dense/all-expert kernel loops every local expert and weights by affinity. Expected on-rank
experts per token is `K · E_local / E_global` — with K=8, 32 local of 256 global, that is **1**.
So dense does ~**32×** the useful work.

To break even against a routed kernel measured at 1.3% MFU, the dense kernel would need ~42%
MFU. Nothing in this codebase is close. **Compute this ratio before proposing a dense bypass**;
it is a capacity/robustness workaround, never a throughput one.

Dense kernels also often hard-code token counts (`assert tokens in (32,64,128,256)`) — check
before assuming a decode kernel can serve prefill shapes.

## Correctness hazards

- **Padding sentinel**: padded blocks carry `block_to_expert == E`, which indexes past `[E,...]`
  weight/scale/affinity tensors. Fix with E+1 dummy-expert padding (pad affinities too) or a
  host-side clamp. A `conditions`-based "skip padded blocks" mechanism does **not** protect
  gathers that happen *before* the skip.
- **Expert-parallel semantics**: all-reduce the routed partials, then add the shared expert
  **once, after** the reduce (it is replicated, so adding before double-counts by the world size).
- **Baseline vs hybrid entry**: a baseline blockwise entry may lay out a per-block SBUF buffer
  with `partition = num_blocks`, which fails past 128 blocks
  (`memset dst partition dimension 288 exceeds maximum 128`). The hybrid/dynamic-loop entry
  avoids it. Prefer the entry the production BF16 path uses.
- **Barriers that switch off with size**: a conditional like
  `scatter_barrier = assignments <= LIMIT` makes "small works, large hangs" a *synchronization*
  story, not a capacity one. See the case study in `references/cte-blocking.md` — we spent
  weeks on the wrong variable.
- **A size threshold that also switches PACKER PATH is the thing to test first.** RESOLVED
  2026-08-21: our CTE "hangs above 2,048 tokens per MoE call" was never a token, block, or
  `packed_len` limit — exceeding `_DIRECT_ROUTE_MAX_ASSIGNMENTS` switches the packer from the
  direct `nonzero_with_count` path to a **tiled stable scan**, and *the tiled scan deadlocks at
  any size*. Forcing the tiled path at a known-good 2,048 tokens hangs identically. Test the
  path, not the quantity: add an env override that selects each packer at a **fixed** size.
- **FIXED: the culprit was `nisa.local_gather`, and the cure was already in the kernel.** It was
  the only *tiled-exclusive GpSimd op* (the working direct path uses `nonzero_with_count`). Its
  result is broadcast to all 16 connected partitions of a core, so it dragged a whole
  diagonal-mask apparatus along just to undo that broadcast. Replacing it with a masked row
  reduce over the routing `matches` matrix — which is **already the one-hot** the gather was
  reconstructing — fixed the deadlock, was bit-identical, and *deleted* code. >2,048 tokens/call
  now runs (3,072 and 4,096 both complete). Before reaching for `local_gather` to do a
  per-partition select, check whether a comparison you already computed IS the one-hot.
- **Ablate against the known-good path, and bisect by halves.** Five plausible hypotheses
  (`affine_range` independence, corrupt metadata, DMA volume, `oob_mode.skip` OOB storms,
  `core_barrier`) were each measured and refuted at ~6 min per run. What converged in three runs
  was: ablate the whole suspect region (completes) → ablate one half (hangs) → ablate the other
  half (completes). And cross ops off for free by asking which ones the *working* path also
  executes: `stream_shuffle_broadcast` and `tensor_tensor_scan` live in shared pass-1 code, so
  they never needed a run.
- **`tensor_scalar`'s per-partition `operand0` must be float32.** An int32 `[P,1]` operand is
  rejected (`'operand0' must be float32, got 'i32'`).
- **Buffer names and op names share ONE namespace.** Reusing a name for both an
  `sbm.alloc_stack` and a `nisa.*` op gives `error: duplicate op name` — and the standalone
  kernel gate compiled it anyway, only the fused Dynamo path rejected it, so the gate is not a
  name-collision check.
- **`nisa.core_barrier` requires LNC degree >= 2** — it is the *only* barrier/sync primitive in
  `nki.isa`. An `if num_shards == 1: core_barrier(x, (0))` branch is dead, uncompilable code at
  LNC=1 (`error: assertion failed: core_barrier() requires LNC degree >= 2`). So an LNC=1 kernel
  cannot express inter-slice ordering this way at all — if a scatter into `nl.shared_hbm` needs
  it, the design needs rethinking, not a flag.
- **A wrapper family can have more than one call site.** Our FP8 CTE wrapper carries its *own*
  copy of the route-packing preamble (`moe_cte_fp8_35b.py`) alongside the BF16 one
  (`moe_cte_35b.py`). Editing one and running the other silently changes nothing — always print
  the flag you think you set, from inside the traced module, and grep it in the run log.

## Real routing is wildly imbalanced — synthetic routing proves nothing

Measured per-expert on-rank assignment counts within one layer: **5 … 1119**; another layer
**0 … 3974** with several experts completely unused. Eight synthetic distributions (uniform,
zipf-skew, all-to-one, all-off-rank, one-hot, real top-k) failed to reproduce a fault that fires
on every real run.

If you need real routing in a harness, get it by graph surgery: add a debug mode that runs
embed → attention → router → top-k, writes `sel` to a device buffer, and **skips** the MoE so
the graph completes; save to `.npy` and feed it in. Remember that skipping the MoE changes every
downstream layer's input, so that trick is valid for extracting routing, not for localizing a
downstream fault.
