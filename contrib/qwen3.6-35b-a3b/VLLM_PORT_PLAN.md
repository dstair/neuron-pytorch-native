# Plan — vLLM serving for Qwen3.6-35B-A3B

Companion to [VLLM_NEURON_ASSESSMENT.md](VLLM_NEURON_ASSESSMENT.md), which analysed the
existing port in depth. This document is the *plan*: what to do, in what order, at what
cost, and what to decide first.

---

## 0. Bottom line

**There is no "add vLLM support" work.** A complete, device-validated vLLM-Neuron port of
this exact model already exists — `htokoyo/vllm-neuron`, branch `add-qwen36-moe`, tip
`65ef8b7`: 30 files, +8,090 lines, GSM8K-CoT 93.0%, 22.57 → 123.78 output tok/s across
concurrency 1 → 8 on `trn2.48xlarge` at TP8/EP8.

So the real question is not "how do we build it" but **"which of four different goals do we
actually want"**, because they are separate projects with different blockers:

| # | Goal | Blocker | Size |
|---|---|---|---|
| A | Reproduce their baseline as a reference point | needs a `trn2.48xlarge` | ~1 day given HW |
| B | Serve under vLLM on our own trn2.3xlarge, TP=4/EP=4 | environment lift (0.16→0.21); HBM risk | ~1 week |
| C | Get the port onto the current release line so it is not stranded | 0.21 → 0.24 forward-port | 1–2 weeks |
| D | Two-way technique trade with the port (no serving) | none — startable today | days, incremental |

**Recommendation: B is the vLLM route to take** (no 48xlarge, no LNC port), with D running
alongside it since D needs no device at all. A is dropped — we set our own trn2.3xlarge
baseline instead of reproducing theirs. C matters only if the port is to be upstreamed.

---

## 1. What changed since the assessment was written

Three of its facts are now stale. Two blockers dissolved; one new one appeared.

| Assessment claim | Current state |
|---|---|
| §4c disk: `/mnt/nvme` has **3.2 GB free of 430 GB**, ~40 GB must be reclaimed | **RESOLVED — 361 GB free.** The box was wiped and re-bootstrapped 2026-09-01, so the old scratch is gone. Not a blocker. |
| Checkpoint is not a blocker (ours serves unchanged) | **Still true.** `/mnt/nvme/Qwen3.5-35B-A3B` (67 GB BF16) is present; same `qwen3_5_moe` arch. No re-download. |
| §4b host stack is "a generation behind" (vLLM 0.16 vs 0.21 required) | **Understated, and now the main code risk — see §3.** The port's base release is itself superseded. |

### New finding: the port is stranded off the release line

The port branches from `release-0.21.0.1.0.0` (5 ahead / 4 behind). The current release
branch, `release-0.24.0.1.1.0`, has **no merge-base with it at all** — unrelated histories —
and does **not** contain `vllm_neuron/model/qwen3_5_moe/`. Its model packages are
`gpt_oss`, `llama3`, `qwen3`, `qwen3_vl`, `synthetic`.

So this port is not "in vllm-neuron"; it is a feature branch against a release that has since
been replaced. Anyone adopting it inherits a forward-port.

**Checked, because it would have changed the plan:** whether upstream gained native hybrid
support in the meantime, which would have obsoleted the port's shared-file changes. It did
not — files containing `IsHybrid` / `HasInnerState` / `get_mamba_state_shape` / `MambaSpec`:
**7 on `add-qwen36-moe`, 0 on `release-0.21`, 0 on `release-0.24`.** The port is still the
only hybrid-model support in that tree, so all 7 touch-points remain necessary.

---

## 2. Goal B — serve on our own trn2.3xlarge at TP=4/EP=4 (viable; memory is the risk)

**Corrected 2026-09-02.** An earlier reading of this called goal B infeasible on topology
grounds, citing the port's `_gdn_wrapped[2](...)` as "the LNC=2 variant indexed by a literal".
That is wrong in an important way: `wrap_nki` returns a caller whose `[...]` is an **SPMD
launch grid**, and `grid=[2]` is *correct* at LNC=2 — which is the trn2.3xlarge default. Their
code expects exactly the topology our box gives by default.

| route | LNC work needed |
|---|---|
| TP=8/EP=8 on our box | LNC=1 → every `[2]` grid literal is wrong (**9+ sites** in `model.py` alone, plus intra-kernel assumptions about 2 program instances). Real work; still not recommended. |
| **TP=4/EP=4 on our box** | **none** — `grid=[2]` at LNC=2 is what they wrote for. |

So the blocker is not topology, it is **HBM**, and it is a genuine risk rather than a wall.
Expert weights dominate: 32.2 B expert params (256 experts × 3.15 M × 40 layers) at BF16 =
64.4 GB, so **EP=4 puts 16.1 GB/rank of expert weights alone** on a 24 GB/rank budget, versus
8.05 GB/rank at their validated EP=8. Adding the non-expert share lands at roughly
**17.5–19.5 GB/rank**, leaving ~4.5–6.5 GB for the KV pool, GDN state and activations.

Two things say it may fit: their KV utilization at `max_model_len=1024` peaked at **≤1.5%**, so
the pool is small; and `max_num_seqs=1` is a valid starting point. One thing says caution:
**their own dense 27B at TP=4 hit an HBM allocation failure above concurrency 2 at a lighter
13.5 GB/rank.** Expect concurrency to be the binding limit, not whether it loads.

**This is now the recommended vLLM route**, since it needs no `trn2.48xlarge` and no LNC port.
Establish our own trn2.3xlarge baseline rather than reproducing theirs. Sequence: load at
`max_num_seqs=1` first (does it fit at all), then walk concurrency up until it OOMs and record
that ceiling as the result. Gate on their GSM8K recipe, not on stock logit thresholds.

**The real remaining cost is the environment**, not the model code — see §3 and §4b of
[VLLM_NEURON_ASSESSMENT.md](VLLM_NEURON_ASSESSMENT.md): vLLM 0.16 installed vs **0.21**
required, `transformers` 4.57.6 vs **≥5.5.1**, Neuron 2.26 vs **2.31**. The plugin interfaces
moved between those versions, so this is a stack lift, not a pip bump.

## 3. Goal C — forward-port 0.21 → 0.24, sized from the actual diffs

The port's own footprint in shared files, against how far those files moved between releases:

| shared file | port adds | 0.21 → 0.24 drift | rebase difficulty |
|---|---|---|---|
| `model/registry.py` | +3 | 2 lines | trivial |
| `model/kv_cache.py` | +19 | 0 lines | trivial |
| `vllm/attention/attn.py` | +14 | 0 lines | trivial |
| `vllm/platform.py` | +98 / −5 | 244 lines | moderate |
| `vllm/core/scheduler.py` | +243 | 285 lines | moderate |
| `vllm/worker/neuron_worker.py` | +95 / −4 | 473 lines | moderate |
| `vllm/worker/neuron_model_runner.py` | **+667 / −13** | **~2,000 lines** (8,252 → 9,086) | **the whole cost** |

Everything except `neuron_model_runner.py` is a day's work. That one file is the project: the
port threads hybrid state allocation, binding, and slot management through a runner that has
since changed by ~2,000 lines.

The 22 *new* files (model package, 8 NKI kernels, APC sidecar, docs, example) carry no merge
risk — they are additive and the kernels are self-contained NKI.

**Sequencing that de-risks it:** port the new files first and get an import-clean tree on
0.24, then rebase the six easy touch-points, then do `neuron_model_runner.py` last with the
port's own GSM8K gate as the acceptance test. Do not attempt a single `git rebase` — the
histories are unrelated, so it would be a merge of two independent trees.

**Open question to resolve before starting:** whether the port's author is already doing
this. Ask before duplicating 1–2 weeks of work; `add-qwen36-moe` may simply be awaiting a
rebase they intend to do.

---

## 4. Goal D — the two-way trade (start here)

Both directions are evidence-backed in the assessment and neither needs a `trn2.48xlarge`.

### 4a. Give: the stable block-diagonal inverse → unlocks their chunked GDN prefill

Their chunked GDN prefill is present but **disabled**. The assessment's §7 argues the cause
is a numerical scheme we already replaced: full-width nilpotent doubling at their default
C=64. Measured on CPU in fp32 against a forward-substitution oracle:

| C | their accumulate-form err | our block-16 err | finite? |
|---:|---:|---:|:---:|
| 16 | 5.3e-05 | 5.3e-05 | yes |
| 32 | **2.00** | 5.3e-05 | yes |
| **64** ← their default | **1.07e+09** | 1.08e-04 | yes |
| 128 ← their clamp ceiling | **1.74e+26** | 2.65e-04 | yes |

**It fails silently** — finite at every width, so a NaN gate never fires. This is why it is
worth sending: their own note attributes the chunked path's divergence to "bf16 noise
perturbing MoE routing", which is what a finite-but-garbage GDN output looks like
downstream, and rounding noise does not produce 1e+09 relative error.

Why this is the top item: chunked-off pins them to the sequential scan → bounded to
≤512-token segments → caps `max_model_len` at 1024. That single ceiling is what makes their
numbers and ours incomparable. Fixing the inverse potentially lifts all of it.

**Deliverable:** `_tri_inverse_blockdiag` (`kernels/deltanet_chunked_prefill_35b.py:908`) plus
`kernels/tests/test_dn_block_inverse_stability.py`, which runs in pure PyTorch with no device
and shows both the failure and the fix. ~40–60 lines of NKI on their side, reusing their
existing `_transpose_pe` / `nc_matmul` helpers. Their validation is one GSM8K run with
`VLLM_GDN_FORCE_CHUNKED=1`.

**Send it as a hypothesis with a test attached, not a bug report.** It is a CPU reproduction
on a synthetic near-worst-case matrix plus our own device experience at C=32 — it predicts
their behaviour, it does not measure it.

### 4b. Take: `paged_kv_gather.py` addressing techniques → our profiled #1 bottleneck

Three items, all pure data movement (`dma_copy` is a byte mover, dtype follows the pool, no
arithmetic changes), so all three are expected **bit-identical** — throughput with no quality
risk. They target the LNC=1 profile's top cost, `rope_kv_dynamic` returning the whole
20,000-token KV cache (25% of instruction time, 21.7% of HBM):

1. **In-place alias** — scatter into the pool and *return it*, so the compiler auto-aliases
   and persists with no fresh alloc or copy. First check whether our in-flight `group_index`
   path in `gqa_rope_kv_35b.py` actually gets that alias or silently pays a full-cache
   materialization.
2. **32-bit indirect-address overflow — a general fix where we have a workaround.** Keep the
   pool 2-D, free-reshape, and use a per-token slot index so the wide multiply happens in the
   DMA engine's >32-bit space, keeping the static `.ap offset=` at zero. Our stashed
   `gqa_rope_kv` work is fighting this same failure class.
3. **`.ap` indirect gather** — expresses "gather only the blocks you need" where torch
   fancy-indexing on a page-strided view is rejected outright.

### 4c. Take (larger, separate project): slot-indexed decode state

Their `[N_SLOTS, H, kd, vd]` paged state with a runtime slot index is the prerequisite for
continuous batching, which we do not have at all. The `.ap` + `PAD_SLOT_ID=-1` +
`oob_mode.skip` idiom is liftable — its only vLLM dependency is a `[B,1]` int32 tensor of
slot indices. What is *not* liftable is everything around it: pool allocation, slot lifecycle,
and the admission gate (`scheduler.py::_pool_admission_ok`, +243 lines).

**Adopt the kernel idiom; treat continuous batching as a separate project.** Note our fused
decode kernel vs their split conv/state kernels makes this a real port, not a copy.

---

## 5. Do not do

- **Do not port their GDN prefill kernels to us.** Sequential is O(T) capped at 512; the
  chunked draft has the broken inverse. We are ahead on both.
- **Do not adopt their split conv/state decode kernels for speed.** We fuse; theirs would
  regress fixed-batch throughput. The value there is continuous batching (§4c), not speed.
- **Do not adopt `VLLM_MOE_TKG_ROUTER_FP32`.** It is a correctness flag for their stack; fp32
  router is slower than bf16 — a trade in the wrong direction for us.
- **Do not compare their tok/s to ours as a ranking.** Their 123.78 tok/s is continuous
  batching at concurrency 8 on **16 chips**; our 681.3 tok/s is fixed BS=128 on **1 chip**.
  Different denominators, batching disciplines, and context budgets. Our 4,370.6 prefill
  headline has no counterpart in their port at all — it is a 20,000-token measurement and
  they are bounded to 1,024.
- **Do not try to reproduce their TP8/EP8 48xlarge numbers.** We set our own trn2.3xlarge
  baseline at TP=4/EP=4 instead; the two are not comparable and do not need to be.

---

## 6. Immediate next actions

1. **Ask the port's author** whether a 0.24 forward-port is already planned (§3). One
   message; it can save 1–2 weeks.
2. **Send §4a** — the inverse finding plus the standalone CPU test. Highest unlock-to-effort
   ratio on either side, and it costs us nothing but the write-up.
3. **Start §4b item 1** — check whether our `group_index` KV write gets the free alias or
   pays a materialization. Pure local work against a profile we already have.
4. **Decide on `trn2.48xlarge` provisioning** before scheduling goal A (§2).

Nothing in 1–3 needs the Trn2, so none of it competes with the prefill queue on the current
capacity block.
