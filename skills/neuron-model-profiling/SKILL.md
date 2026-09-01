---
name: neuron-model-profiling
description: |
  Profile a FULL multi-layer model or multi-rank (TP=N) graph on Trainium and attribute
  time to engines, opcodes, and source lines. Use when asked to "profile the model",
  "profile prefill/decode", "why is this slow", "find the bottleneck", "which engine is
  bound", "attribute time to ops", "find low hanging fruit", or when you have a
  torch-neuronx compile cache and need to know WHICH NEFF to capture. Covers
  neuron-explorer collective replay across ranks, region-NEFF identification in a
  neff_cache, the synthetic-input trap that silently under-reports runtime loops, the
  wall-clock reconciliation gate, and the parquet queries that find fixed-cost ops,
  wasted engine width, and stall sites.
  For a SINGLE NKI kernel on one core use neuron-nki-profiling instead.
---

# Profiling a full model on Trainium

Scope: many layers, many ranks, graphs you did not compile by hand. For one kernel on one
core, use `neuron-nki-profiling` — it is simpler and correct for that case.

**This skill overrides one default in `neuron-nki-profiling`:** that skill captures with
`--enable-dge-notifs`. At full-model scale you must turn DGE notifications **off** — see
"Flags that are wrong at scale" below.

## Workflow

### 1. Find the NEFF you actually want

A bucketed / graph-split model compiles many NEFFs into a cache. You want the big ones.

```bash
sudo find "$CACHE/neff_cache" -name '*.neff' -printf '%s %p\n' | sort -rn | head -40
```

- Use `-printf '%s %p'`. `find -size` works in k-blocks and will not separate these.
- Group by size. Sizes cluster into **classes = one distinct graph × one per rank**. With
  `--prefill-splits 4` at TP=8 you should see 32 large NEFFs in 4 classes of 8.
- Take one representative per class and name them `rA`, `rB`, … Copy them out of the cache
  (`sudo cp` then `chown`, since a `--privileged` container makes them root-owned).
- Helper NEFFs (tens–hundreds of KB) are individual NKI kernels, not the model. Ignore them
  unless that is what you want.

### 2. Capture with collective replay

`scripts/capture_region.sh <tag>` wraps this:

```bash
neuron-explorer capture -n /work/$TAG.neff -s /work/$TAG.ntff \
  --collectives-worker-count 8 -r 8 -i 0 --single-io \
  --profile-nth-exec=2 --ignore-exec-errors
```

- `--collectives-worker-count 8 -r 8 -i 0` replays the collectives with 8 workers and
  profiles worker 0. Without it, any graph containing an all-reduce will not run.
- Run **inside the DLC** with `-e NEURON_LOGICAL_NC_CONFIG=1`, else you get
  "Logical NC not available 8/4".
- Output lands at `<tag>_rank_0_exec_2.ntff` (the `_exec_2` suffix comes from
  `--profile-nth-exec=2`), not the name you passed to `-s`.
- Direct runtime inspection (`neuron-explorer inspect`, `NEURON_RT_INSPECT_*`) captures the
  REAL run but emits **NTFF v129**, which neither host `neuron-profile` 2.31 nor container
  `neuron-explorer` 2.32 can parse (they support v1–7). It is a dead end today
  **(version-bound)** — use capture-replay.

### 3. Engine-level summary

`scripts/summarize.sh <tag>` →

```bash
neuron-explorer view --output-format summary-json -n /work/$TAG.neff \
  -s /work/${TAG}_rank_0_exec_2.ntff > summ_$TAG.json
```

`summary-json` prints to **stdout**; `--output-file` is rejected for this format
**(version-bound)**. `view` on a ~600 MB–1 GB NTFF takes minutes — empty output is not failure.

### 4. Source- and op-level attribution

`scripts/ingest.sh <tag>` →

```bash
neuron-explorer view --ingest-only --data-path /work/data --display-name $TAG \
  -n /work/$TAG.neff -s /work/${TAG}_rank_0_exec_2.ntff
```

- `--ingest-only` **requires `--display-name`** or it exits `fatal: Missing --display-name`.
- Parquet lands in `<data-path>/profiles/global/<display-name>@latest/*.parquet`.
- The tables you will use: `Summary`, `OpcodeSummary`, `Instruction` (has
  `nki_source_location` per instruction), `ActiveTime`, `DmaUsage`, `Throttle`.
- The host may have no pandas. Use a venv (on our box: `/mnt/nvme/profile-venv/bin/python`).

Then run `scripts/profile_queries.py` for the standard battery (engine table, top
(engine, opcode), fixed-cost detection, per-source rollup, component rollup, region diff).

For source lines to render in the interactive UI, see `references/interpreting-metrics.md`
(the `--source-code-root` confinement gotcha will silently blank them otherwise).

## ⚠️ The trap that will invalidate your analysis

**`capture` replays the NEFF with synthetic inputs.** Any loop whose trip count is decided at
runtime from data — an MoE block loop driven by a routing `conditions` vector, a dynamic
`while`, `nl.fori_loop` over a register — executes an arbitrary and usually *much smaller*
number of iterations than in the real run.

Observed: capturing 4 region NEFFs of the same model, the MoE compute appeared in **one** of
them. `bwmm_shard_on_I.py:1384` had 29,920 instructions in region A and **0** in region B.
Summing the four regions understated the MoE by roughly 4×, and would have sent us optimizing
DeltaNet instead.

**Always run the reconciliation gate before you trust any number:**

```
(per-region time) × (regions) × (steps per inference) ≈ measured wall time?
```

Our case: `4 × rA(286.83 ms) × 39.1 steps = 44.8 s` vs a measured 43.54 s (103%) → rA is a
valid unit. The 4-region sum gave 35.15 s (81%), which looked plausible and was wrong. If it
does not reconcile, your unit is wrong — find the region where the data-dependent work
actually ran, or accept that you can only rank, not budget.

Corollary: **a capture that does not reproduce a data-dependent bug is not evidence the bug
isn't there.** Synthetic routing produced 0 out-of-bounds events for a fault that fires every
time on real data.

## Flags that are wrong at scale

| flag | single kernel | full model |
|---|---|---|
| `--enable-dge-notifs` / `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1` | recommended (fills `DmaPacket*`) | **OFF** — one host notification per indirect DMA overflows the queue at 40 layers (`NRT status 1204 NQ overflow`) |
| `NEURON_LAUNCH_BLOCKING=1` | fine for debugging | never when timing |
| `--single-io` | optional | yes — avoids materializing every input |

Cost of DGE-off: `DmaPacket` tables are incomplete. Use `Summary`'s `hbm_read_bytes` /
`hbm_write_bytes` for aggregate traffic and never the packet byte total as a bandwidth bound.

Debug flags together measured **~15×** slower than clean flags on this workload. Any timing
taken with them is not a number — see `trainium-run-discipline`.

## Reading the output

Full detail in `references/interpreting-metrics.md`. The three things people get wrong:

1. **`*_engine_active_time_percent` overlaps across engines** — they sum well past 100%. The
   critical-path metric is `<engine>_engine_instruction_time / total_time`. If one engine is
   at ~100%, that instruction stream *is* the critical path; if the max is 65–70%, no engine
   owns it and you are latency-bound.
2. `perfect_pipeline = max(per-engine active time)`; `serialization_gap = total_time −
   perfect_pipeline`. A 60% gap means engine-time savings convert to wall-clock only partly —
   bound your claims accordingly.
3. `evt_wait_time_ns` double-counts (it summed to 390 ms inside a 105 ms window). Use it to
   **rank** stall owners, never as absolute time.

## Files

- `scripts/capture_region.sh`, `scripts/summarize.sh`, `scripts/ingest.sh` — the three driver
  steps, parameterized by tag; edit `W`/`IMAGE` at the top for a new box.
- `scripts/profile_queries.py` — the standard analysis battery over the parquet.
- `references/interpreting-metrics.md` — metric semantics, the UI recipe, known tool quirks.
