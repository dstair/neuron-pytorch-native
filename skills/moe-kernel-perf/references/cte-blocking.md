# CTE blocking, the route packer, and a worked threshold case study

## Route packer anatomy (`kernels/moe_cte_35b.py`)

The packer turns global top-k `sel` into the three metadata tensors the blockwise kernel needs:
`token_position_to_id` (the packed assignment list), `block_to_expert`, and `conditions` (which
blocks are real). Two implementations:

| path | selected when | cost structure |
|---|---|---|
| **direct four-expert nonzero** | `scatter_barrier and assignments <= _DIRECT_ROUTE_MAX_ASSIGNMENTS` (16384) | fastest; keeps a `[128, assignments]` int32 result in SBUF |
| **tiled stable scan** | otherwise | bounded working set (`_ROUTE_TILE = 2048`); required above the SBUF limit |

The direct path allocates **two** `[128, assignments]` int32 buffers (`nonzero_input`,
`routed_indices`) = **16 MB of SBUF** at assignments=16384, against 24 MB total. That is why the
cap exists and why it cannot simply be raised.

Its loop is `for expert_batch in range(num_local_experts // 4): for expert_lane in range(4)`,
with lanes at a **32-partition stride** (partitions 0, 32, 64, 96). Per MoE call that yields:

- 8 × `nonzero_with_count` — fixed **90.8 µs** each ⇒ ~727 µs/call, ~39% of packer cost
- 32 × equality-match `tensor_scalar` on a `[1, assignments]` row — 8.68 µs each
- 32 × right-shift `tensor_scalar` (assignment id → token id), same 1-partition layout
- a `dynamic_range` block-select/store loop per expert

Only 4 of 128 partitions are ever populated, so the 8 MB buffers are ~3% utilized per op. If the
lane stride can be tightened, 8 nonzero calls collapse toward 1 — worth ~4% of a prefill.

## Worked case study: "it hangs above 64 blocks" — and why that was wrong twice

A CTE MoE call deadlocked (`NRT status 5`, 30 s timeout, collective completed) above some size.
The received wisdom was "more than 64 blocks per call hangs". Eliminated first, at real cost:
loop primitive (three rewrites), graph splits, and isolation (clean standalone at 112 blocks,
single-core **and** full TP=8 with real routing).

### Step 1 — enumerate everything that co-varies

All prior data used `block_size = 512`, where **three** quantities move together. Setting
`block_size = 256` (already supported, zero code) decouples them:

| tokens/call | block_size | blocks | packed_len | result |
|---|---|---|---|---|
| 1536 | 512 | 56 | 28672 | runs |
| 2048 | 512 | 64 | 32768 | runs (baseline) |
| **2048** | **256** | **96** | **24576** | **runs, bit-identical** |
| **2560** | **256** | **112** | **28672** | **hangs** |
| **3072** | **256** | **128** | **32768** | **hangs** |
| 3072 | 512 | 80 | 40960 | hangs |
| 5120 | 512 | 112 | 57344 | hangs |

Now apply the separation test — **a candidate variable must not overlap between the working and
failing sets**:

- **block count**: runs span 56–96, hangs span 80–128 → **overlap → refuted.** 96 blocks runs.
- **packed_len**: 32768 appears on **both** sides, and 28672 hangs while 32768 runs →
  **refuted in both directions.**
- **tokens per call**: runs ≤ 2048, hangs ≥ 2560 → separates cleanly.

Two experiments (~20 min each at 4 layers, zero code) refuted the load-bearing premise of weeks
of work. The lesson generalizes: **the quantity people name is rarely the one that separates.**

### Step 2 — the miss, and the correction

"Tokens per call" separates the sets, but so does something else nobody enumerated: with
`K = 8`, 2048 tokens is **exactly 16384 assignments**, i.e. `_DIRECT_ROUTE_MAX_ASSIGNMENTS`. And:

```python
scatter_barrier = assignments <= _DIRECT_ROUTE_MAX_ASSIGNMENTS   # moe_cte_35b.py:1294
```

So at the same threshold, **three** things flip at once:

1. tokens per MoE call crosses 2048,
2. the packer switches direct → tiled stable scan,
3. `scatter_barrier` switches **True → False**, disabling per-slice core barriers in the scatter.

Every working config is on the direct packer **with** barriers; every hanging config is on the
tiled scan **without** them. The existing code comment even flags the hazard:

> At BS=4 the bounded tiled scan owns every non-padding destination uniquely; keeping its
> per-slice core barriers can cross-synchronize with a different TP rank's adjacent custom call.

A size-conditional barrier plus a TP=8 hang whose collective completed is a far better fit for
the evidence than an intrinsic token-count limit — and it explains the standalone isolation
result, because a standalone kernel has no *adjacent custom calls* to cross-synchronize with.

**Status: hypothesis, not yet tested.** The decisive experiments, both one-liners at 4 layers:

- force `scatter_barrier=True` on the tiled path at 3072 tokens → isolates the **barrier**
- force the tiled path at 2048 tokens with barriers on → isolates the **path**

The direct path cannot be pushed above 16384 assignments (SBUF), so the confound can only be
broken downward.

### Method, generalized

The full checklist is in `trainium-run-discipline/references/isolating-thresholds.md`. The two
rules that mattered here:

- Enumerate **code paths and configuration branches** as variables, not just numeric quantities.
  Numbers are easy to list and were not enough; a boolean derived from the same number was the
  real suspect.
- Prefer the zero-code knob that decouples variables over building a workaround. The workaround
  under consideration (a dense bypass) would have been a ~32× FLOP regression.

## Blocking-related compile cascades

Taking the CTE block-quant path from BS=2 to higher batch cleared, in order: `memset` partition
288 > 128 (tile the block-metadata init to ≤128); duplicate op names (make `name=` unique per
chunk); "expecting simple variable" from a tuple loop variable (`enumerate` → `range`); an
arithmetic slice bound (precompute a plain name); then a gather OOB from the padding sentinel.

Expect a **cascade** when a path meets a scale it has never run at, and iterate at 4 layers
(block counts depend on batch × chunk, not layer count).
