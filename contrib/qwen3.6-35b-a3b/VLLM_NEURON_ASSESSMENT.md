# vLLM-Neuron Qwen3.6-35B-A3B — port assessment and GDN kernel trade

Assessment of the vLLM-Neuron port of Qwen3.6-35B-A3B against this repo's
PyTorch-Native implementation, and of the Gated DeltaNet (GDN) kernels on both
sides.

**Source pinned:** vLLM-Neuron branch `origin/add-qwen36-moe`, tip commit
**`65ef8b7`**. Nothing on that branch was modified or checked out; all
inspection was read-only git plumbing. Paths below of the form
`vllm_neuron/...` or `Qwen3.6-35B-A3B/...` refer to that commit.

Peer documents: [BENCHMARK.md](BENCHMARK.md), [PREFILL_RECIPE.md](PREFILL_RECIPE.md),
[DECODE_RECIPE.md](DECODE_RECIPE.md).

---

## 1. Summary

The question this started from was what a vLLM-Neuron port of Qwen3.6-35B-A3B
would look like and what performance baseline it would give us. The answer:
**the port already exists, in-tree, and its authors validated it on hardware.**
There is no porting work to do. What is left is a *reproduce-and-compare* job and
a genuine two-way kernel trade.

Three things to take away:

1. **Their port is complete and device-validated** — `trn2.48xlarge`, TP8/EP8,
   BF16: GSM8K-CoT exact-match **93.0%**, output throughput **22.57 → 123.78
   tok/s** across concurrency 1 → 8 (§3).

2. **We cannot run it on the hardware we have**, for three independent reasons —
   topology (they need TP8, our box gives 4 logical cores), a host stack a
   generation behind, and disk (§4). This is a hardware-availability gap, not a
   code gap. §10 is a copy-paste recipe for whoever gets a `trn2.48xlarge`.

3. **The high-value finding: their chunked GDN prefill is disabled by a numerical
   failure we already solved.** Their draft chunked kernel inverts the intra-chunk
   matrix with full-width nilpotent doubling at default C=64. That is the same
   Neumann series this repo already found unusable, and it is unusable *in fp32*,
   *silently* — the result stays finite at every chunk width, so a NaN gate never
   fires (§7). Our shipped `_tri_inverse_blockdiag` is a direct fix. This is the
   strongest item to hand back to them (§9).

A note on what this document does **not** claim: their numbers and ours are not
comparable as a ranking. Their validated regime tops out at `max_model_len=1024`
with ≤512-token GDN prefill segments, so **our 20,000-token prefill headline has
no counterpart in their port at all** (§5).

---

## 2. What is on `origin/add-qwen36-moe`

30 files changed, +8,090 / −32 against the merge-base with
`origin/release-0.21.0.1.0.0`. Grouped by purpose:

**Model package** — `vllm_neuron/model/qwen3_5_moe/`
| File | Lines | Role |
|---|---:|---|
| `model.py` | 3,618 | the whole decoder: GDN layer, GQA layer, MoE, 6 GDN algorithm variants behind env-flag dispatch |
| `config.py` | 218 | Neuron config plumbing, bucket/segment sizing |
| `weight_loaders_bf16.py` | 196 | BF16 checkpoint → sharded device weights |
| `factory.py` | 81 | model construction |
| `README.md` | 36 | package-level notes |
| `__init__.py` | 6 | |

**NKI kernels** — `vllm_neuron/functional/`
| File | Lines | Role | Enabled? |
|---|---:|---|---|
| `gated_delta_rule_seq.py` | 201 | sequential per-token GDN prefill scan (`nl.sequential_range(T)`) | **yes** — the only supported GDN prefill |
| `gated_delta_rule.py` | 471 | chunked GDN prefill | **no** — draft, see §7 |
| `gdn_state_update.py` | 183 | slot-indexed decode recurrent-state update, indirect DMA | yes |
| `gdn_conv_update.py` | 170 | slot-indexed decode conv-state update | yes |
| `gdn_state_update_compact.py` | 86 | compact variant | |
| `gdn_conv_update_compact.py` | 82 | compact variant | |
| `paged_kv_gather.py` | 450 | block-indexed unified KV gather for the GQA layers | yes (`VLLM_UNIFIED_KV_GATHER=1`) |
| `slot_indirect_probe.py` | 113 | `.ap` indirect-DMA probe / idiom reference | |

**Hybrid-cache sidecar** — `vllm_neuron/vllm/worker/neuron_mamba_apc.py` (430 lines),
the automatic-prefix-caching path for the recurrent state.

**Shared-framework touch-points (7 files)** — `model/registry.py` (+3, registers
`Qwen3_5MoeForCausalLM`), `model/kv_cache.py` (+19, hybrid KV spec),
`vllm/attention/attn.py` (+14), `vllm/platform.py` (+103),
`vllm/core/scheduler.py` (+243, pool-aware admission gate),
`vllm/worker/neuron_model_runner.py` (+680),
`vllm/worker/neuron_worker.py` (+99). Plus one unrelated-looking tweak to
`functional/attention/attention_decode_mask.py` (+34/−…).

**Docs and example** — `Qwen3.6-35B-A3B/README.md` (432 lines; the port's single
source of truth, and the only file in that directory),
`docs/model-recipes/qwen3-6-moe.md`, `docs/tutorials/tutorial-qwen3-6-moe.md`,
`examples/vllm_neuron/models/qwen3_5_moe/run.py` (101 lines, standalone runnable).

**Registration path.** vLLM discovers the platform through the
`vllm.platform_plugins` entry point (`neuron = "vllm_neuron:register"`); the model
is then resolved from `registry.py`'s `get_models()` as `Qwen3_5MoeForCausalLM`.
Because the HF checkpoint declares the multimodal class
`Qwen3_5MoeForConditionalGeneration`, serving **requires** an explicit
`--hf-overrides '{"architectures":["Qwen3_5MoeForCausalLM"]}'` to select the
text-only decoder — see §10.

**Sibling branch.** `origin/add-qwen36-27b` carries the *dense* 27B variant, built
on the same base commits, swapping `qwen3_5_moe/` for `qwen3_5_dense/`. Its TP=4
experience is cited in §4 as the closest evidence for what a TP=4 run costs.

---

## 3. Their measured baseline

All figures below are quoted from `Qwen3.6-35B-A3B/README.md` at commit `65ef8b7`
and are that port's own measurements. Configuration for every row:
`trn2.48xlarge` (16 chips / 64 logical NeuronCores at LNC2), **TP8 / EP8**, BF16,
greedy on-device sampling, Neuron 2.31 stack (vLLM 0.21 / vllm-neuron
0.21.0.1.0.0), `max_model_len=1024`, `kv_segment_size_buckets:[512]`.

**Accuracy** — GSM8K-CoT, exact-match, 100 questions, 4-shot
(`--apply_chat_template --fewshot_as_multiturn`), `max_tokens 448`,
`enable_thinking: false`, `num_concurrent 1`:

| Metric | Value |
|---|:---:|
| GSM8K-CoT exact-match, flexible-extract (BS=1) | **93.0%** ± 2.6% |
| GSM8K-CoT exact-match, strict-match (BS=1) | **92.0%** ± 2.7% |

The ± figures are lm_eval's standard error; 100 questions is a subset, not the
full split.

**Throughput** — `vllm bench serve` (online rows) / `vllm bench throughput`
(offline row), random dataset, 256-token input / 128-token output
(`--random-range-ratio 0`), `--ignore-eos`, 24 prompts per online concurrency
level and 16 offline. All requests completed (0 failed):

| Configuration | Concurrency | Output throughput | Mean TPOT | Mean TTFT |
|---|:---:|:---:|:---:|:---:|
| online | 1 | 22.57 tok/s | 42.18 ms | 314 ms |
| online | 4 | 74.37 tok/s | 48.10 ms | 775 ms |
| online | 8 | **123.78 tok/s** | 54.01 ms | 1411 ms |
| offline (`bench throughput`) | 1 | 22.45 tok/s (67.36 total tok/s) | — | — |

Output throughput scales 5.5× from concurrency 1 → 8 while TPOT rises only 28%.

**What they explicitly did not verify** (their own list, condensed): any TP/EP
degree other than TP8/EP8; any quantization other than BF16; any sequence length
above `max_model_len=1024`; APC beyond one functional check (an 817-token shared
prefix at BS=4 — no accuracy or throughput measurement); vision input; and
**chunked GDN prefill**, which is present in the tree but not enabled on this
stack.

They also flag a gating trap worth carrying over: **do not gate this model with
the generic logit-validation script's stock thresholds.** Those thresholds sit
below the model's BF16 noise floor — the CPU BF16 baseline's own L-inf error
against FP32 is 0.03–0.11, against the script's static 0.011 top-5 tolerance — so
`run_logit_validation_offline.py` reports FAILED even when every prompt's greedy
argmax stays within the script's own 1-ULP acceptance threshold. Gate on GSM8K
instead. (Our repo reaches the same conclusion by a different route: our prefill
gate is a fingerprint + top-5 + capture-replay cosine + real-prompt coherence
stack, not a raw logit tolerance.)

---

## 4. Reproduction requirements, and why it is blocked here

Three independent blockers. Each would have to be cleared; none is a code change.

### 4a. Topology — the binding one

Their recipe is TP=8. The **trn2.3xlarge** used for this assessment is
**1 chip / 8 physical NeuronCores / 96 GB**, which
`neuron-ls` presents as **4 logical cores × 24 GB at LNC=2**. Their validation
host is a **trn2.48xlarge** — 16 chips / 64 logical cores.

The LNC=1 trick this repo uses to turn one chip into 8 logical cores (see
`PREFILL_RECIPE.md` §3c) **does not transfer to their tree**:

- Their GDN prefill host wrappers call `_gdn_wrapped[2](...)` — the LNC=**2**
  variant, indexed by a literal, in both `_chunk_gated_delta_rule_nki` and
  `_segmented_gated_delta_rule_nki`.
- `gated_delta_rule_seq.py`'s docstring records the same: `wrap_nki` "(LNC=2)".
- Their `rmsnorm_router_topk_tkg` and `moe_block_tkg` paths document LNC=2
  sharding requirements.

So the most that could run here is **TP=4 / EP=4**, a configuration they never
validated, at roughly **19.5 GB against a 24 GB/rank budget**. That margin is
thin, and the nearest evidence is discouraging: their own **27B dense** TP=4 run
hit an HBM allocation failure above concurrency 2 at a *lighter* 13.5 GB/rank.

### 4b. Host stack — a generation behind

| Component | Installed here | Branch requires |
|---|---|---|
| vLLM | 0.16.0 | **0.21.0** |
| vllm-neuron | 0.5.0 | **0.21.0.1.0.0** |
| transformers | 4.57.6 | **≥ 5.5.1** |
| Neuron / neuronx-cc | 2.26 | **2.31** |

This is not a patch-level bump; the plugin interfaces (`HasInnerState`,
`IsHybrid`, `get_mamba_state_shape_from_config`, `bind_mamba_state`) moved between
these versions.

### 4c. Disk

The instance-store volume was nearly full during the assessment. Roughly
**40 GB** would need reclaiming for a NEFF/NKI compile cache
(`VLLM_CACHE_ROOT`).

**One thing that is *not* a blocker:** the checkpoint. Our existing
`QWEN35_MODEL_DIR` BF16 checkpoint serves their port unchanged — same
architecture (`qwen3_5_moe`), and their own README lists Qwen3.5-35B-A3B as
"BF16 (same arch)". No 70 GB re-download.

**What would unblock it:** a `trn2.48xlarge`; a Neuron 2.31 / vLLM 0.21
environment; and ~40 GB reclaimed on the instance-store volume. With those
three, §10 runs as-is.

---

## 5. Metric comparability — read this before comparing any number

Their figures and ours measure different things in different regimes. The
honest summary is that **there is currently no apples-to-apples pair.**

| Axis | Their validated regime | Ours |
|---|---|---|
| Context length | `max_model_len=1024` | **20,000-token** prefill; 2,048 decode |
| GDN prefill | sequential per-token scan, ≤512-token segments | chunked C32, 4-stream packed, full 20k |
| Batching | continuous, concurrency 1–8 | fixed batch (BS=2 prefill, BS=1/32/128 decode) |
| Metric | output tok/s (decode-side, 256-in/128-out) | agg. prompt tok/s (prefill); output tok/s (decode) |
| Precision | BF16 only | BF16 + FP8 experts (decode) |
| Topology | TP8/EP8 on 16 chips | TP4/LNC2 or TP8/LNC1 on **1** chip |

Two specific traps:

- **Our prefill headline has no counterpart.** 3,456.8 aggregate prompt tok/s is
  measured over a 20,000-token prompt. Their port is bounded to
  `max_model_len=1024` and their GDN prefill to ≤512-token segments, so that
  measurement cannot be taken on their stack without a new bucket recipe and a
  cold recompile. Their TTFT figures (314 ms at concurrency 1) are for 256-token
  prompts.
- **Their 123.78 tok/s is not our 320.6 tok/s.** Theirs is continuous batching at
  concurrency 8 on 16 chips; ours is a fixed BS=32 full-graph decode on 1 chip.
  Different denominators (chips), different batching disciplines, different
  context budgets.

**What an apples-to-apples run would need**, in order of cost: (a) same hardware —
both on `trn2.48xlarge`, or their port down to 1 chip, which §4a says is not
currently possible; (b) same context and shape — either their recipe extended past
1024 (which needs §7 fixed, or a longer segmented recipe), or ours cut to
256-in/128-out; (c) same batching discipline — ours has no continuous batching at
all (§9b), so the only common ground today is fixed-batch, which throws away
their main advantage.

Until then: treat their numbers as **the vLLM-integration baseline** (what you
get with a scheduler, continuous batching, APC, and an OpenAI endpoint) and ours
as **the kernel-ceiling baseline** (what the hardware yields on a frozen shape).
Both are useful; neither is the other's ranking.

---

## 6. GDN kernel comparison

### 6a. Prefill

| | **Ours** — `kernels/deltanet_chunked_prefill_35b.py` | **Theirs (enabled)** — `functional/gated_delta_rule_seq.py` | **Theirs (draft)** — `functional/gated_delta_rule.py` |
|---|---|---|---|
| Algorithm | chunked delta rule | sequential per-token scan | chunked delta rule |
| Loop construct | static unroll over chunks | `nl.sequential_range(T)` | static unroll over chunks |
| Chunk / segment | **C=32** (C=16 fallback) | ≤ **512**-token segments | C=**64** default, clamped to [1,128] |
| Intra-chunk inverse | `_tri_inverse_blockdiag` — two 16-wide diagonal blocks + coupling term | n/a (no chunk matrix) | full-width nilpotent doubling, `n_double = ceil(log2 C)` |
| Stream packing | `_chunk_pack4` — 4 streams into one P=128 block-diagonal tile | none | none |
| Intermediates | SBUF-resident; transpose-once finish | — | HBM round-trips |
| Normalization | **inside the kernel** (`_l2norm_rows`, `1/sqrt(K)`) | host prologue | host prologue |
| Context ceiling | 20,000 tokens, validated | 1,024 validated (segment ≤512) | untested |
| Status | **shipped**, 3,456.8 agg prompt tok/s | **shipped**, the only supported GDN prefill | **draft, disabled** (§7) |

The shapes are close enough that the comparison is meaningful: both chunked
kernels compute the same quantity in the same order (Gram matrix → strictly-lower
mask → `(I-M)^{-1}` → intra terms → inter-chunk `v' = w @ S_prev` → state update).
They diverge on exactly one design choice, the inverse — and that choice is why
theirs is off (§7).

Their sequential kernel is a genuinely different trade: no chunk matrix means no
inverse and no stability question at all, at the cost of a `sequential_range(T)`
scan whose instruction count is linear in tokens. That is why it is bounded to
≤512-token segments — the graph gets impractical beyond that. It is the right
choice for a 1,024-token recipe and the wrong one for 20,000.

### 6b. Decode

| | **Ours** — `kernels/deltanet_full_batched_v2_35b.py` | **Theirs** — `functional/gdn_state_update.py` + `gdn_conv_update.py` |
|---|---|---|
| State layout | `[B*V_HEADS*K_DIM, V_DIM]` — flat, contiguous, static batch | `[N_SLOTS, H, kd, vd]` — paged pool, slot-indexed |
| Addressing | compile-time batch offsets | **runtime** slot index via `.ap(..., scalar_offset=slot, indirect_dim=0)` |
| Slot source | n/a | `block_table[:, 0]`, `[B,1]` int32 |
| Padding lanes | n/a | `PAD_SLOT_ID=-1` → huge uint32 → `oob_mode.skip` makes the DMA inert |
| Free-axis tiling | conv-state tiling (`DN_TILED_CONV`) | `F_TILE=8192` on the baseline pool copy (`H*kd*vd` fp32 = 256 KB > ~192 KB SBUF) |
| Reductions | `nc_matmul` contractions on `kd` | `nc_matmul` contractions on `kd`, q/k as `[kd,1]` stationary |
| Batching | fixed / static | continuous |
| Fusion | conv + recurrence + gated RMSNorm in one kernel, optional direct state-out | conv and state updates are separate kernels |
| Best measured | **442.1 tok/s @ BS=128** (FP8 experts) | 123.78 tok/s @ concurrency 8 (BF16; see §5) |

The decode comparison runs the other way from prefill: **their state addressing is
the thing we lack.** Ours is faster on a frozen batch; theirs can serve a queue.
See §9b.

---

## 7. Finding — full-width nilpotent doubling fails from C≥32, silently

**This is the item worth sending back to them.**

### The claim

`gated_delta_rule.py` Step 8 forms `(I - M)^{-1}` for the intra-chunk matrix by
accumulate-form nilpotent doubling:

```python
# inv = I; m_pow = M
# repeat n_double:  inv = inv + m_pow @ inv ;  m_pow = m_pow @ m_pow
nisa.tensor_copy(dst=inv_cc, src=eye_sb)
nisa.tensor_copy(dst=mpow_cc, src=M_cc)
for _d in range(n_double):                         # static unroll
    ...
    nisa.nc_matmul(dst=p_term, stationary=mpow_t, moving=inv_cc)
    nisa.tensor_tensor(dst=inv_cc, data1=inv_cc, data2=term_cc, op=nl.add)
    nisa.nc_matmul(dst=p_sq, stationary=mpow_t, moving=mpow_cc)
```

with the host computing `n_double = max(1, (C - 1).bit_length())` — i.e.
`ceil(log2 C)` — and `VLLM_GDN_CHUNK_SIZE` defaulting to **64** (clamped to
[1,128]).

`M` is strictly lower triangular, hence nilpotent (`M^C = 0`), so the doubling is
*exact in exact arithmetic*. Expanding it:

```
inv_0 = I,                              m_0 = M
inv_1 = I + M,                          m_1 = M^2
inv_2 = (I + M^2)(I + M) = Σ_{j<4} M^j, m_2 = M^4
inv_3 = (I + M^4)(I + M^2)(I + M) = Σ_{j<8} M^j
```

That is the same Neumann series as the product form
`Π_k (I + M^(2^k)) = Σ_{j<2^n} M^j` — which is the form **this repo already tried
and rejected**: "the naïve full-32 doubling overflows on near-1-decay streams →
NaN at bs2 (root-caused to layer 18 / near-1 `T` entries); **unusable**"
(`PREFILL_RECIPE.md` §6).

### The measurement

`kernels/tests/test_dn_block_inverse_stability.py::test_doubling_degrades_with_chunk_size`
reproduces this, in **fp32**, on a near-worst-case chunk matrix (gate −0.01 →
near-1 decay; keys 0.99-correlated onto a shared direction — the regime that
root-caused to layer 18). Error is max-abs against a scalar forward-substitution
oracle. Verified output:

| C | `n_double` | accumulate-form err (theirs) | product-form err | **block-16 err (ours)** | peak ‖M^(2^k)‖ | finite? |
|---:|---:|---:|---:|---:|---:|:---:|
| 16 | 4 | 5.299e-05 | 1.223e-04 | 5.299e-05 | 1.147e+03 | yes |
| 32 | 5 | **2.000** | 1.001 | **5.299e-05** | 1.729e+07 | yes |
| **64** ← their default | 6 | **1.074e+09** | 2.684e+08 | **1.082e-04** | 5.665e+15 | yes |
| 128 ← their clamp ceiling | 7 | **1.741e+26** | 2.321e+26 | **2.653e-04** | 8.782e+32 | yes |

Three things to read off this:

1. **C=16 is fine; C≥32 is not.** This is exactly why paired C16 was our reliable
   prefill baseline until the stable C32 inverse landed.
2. **At their default C=64 the inverse is meaningless** — nine orders of magnitude
   of error. The mechanism is plain in the last column: the intermediate powers
   reach 5.7e+15, past the fp32 mantissa, and the alternating series then cancels
   catastrophically. It is *not* a bf16 problem; fp32 does not save it.
3. **The failure is silent.** `finite=True` at every width. There is no NaN and no
   inf to gate on. This is precisely the trap our own C32 work hit —
   "catastrophic rank-2 errors masked by RMSNorm finiteness" — and it is why a
   finiteness check is not a sufficient correctness gate for this kernel.

The accumulate form is *slightly worse* than the product form at C=32 and C=64
(2.00 vs 1.00; 1.07e+09 vs 2.68e+08). Both are unusable; the distinction does not
matter practically.

### Why this probably explains their symptom

`model.py` attributes the chunked path's divergence to bf16 noise perturbing MoE
top-8 routing. That is a plausible *downstream* description — a corrupted GDN
output does perturb routing — but the evidence above says the inverse scheme is
the upstream cause, and that it fails in fp32 too. Their own note that the chunked
path "perturbs MoE routing" is consistent with a silently-wrong (finite, garbage)
GDN output rather than with rounding noise: rounding noise does not produce
1e+09 relative error.

### The fix we already ship

`_tri_inverse_blockdiag` (`deltanet_chunked_prefill_35b.py:908`) splits the 32×32
chunk matrix into two 16×16 diagonal blocks — where doubling *is* accurate, per
the C=16 row — inverts each by doubling, and couples them with a single term
`X_10 = D_1 M_10 D_0`. The last column of the table shows the same idea holds at
C=64 and C=128 with block forward substitution over 16-wide blocks
(`inverse_blocked` in the test), staying at ~1e-4 where full-width doubling is at
1e+09 and 1e+26.

Cost, measured on our stack: the block-diagonal inverse is essentially free
relative to the alternatives — a stable Horner series was ~4× costlier (−2.9%
throughput) and was rejected for that reason; block-diagonal doubling is both
stable and cheap.

### Honest caveats

- **This is a CPU fp32 reproduction on a synthetic near-worst-case matrix.** It
  predicts their device behaviour; it does not prove it. Their real chunk matrices
  at C=64 may be better conditioned than this construction.
- What raises confidence above "synthetic": we hit the *same* failure on the
  *same* algorithm on *real weights* on device at C=32, root-caused to a specific
  layer, and fixed it by exactly this change. And their independently-observed
  symptom (chunked path perturbs routing) is what a silent wrong-inverse looks
  like.
- What would settle it in an hour on their hardware: swap Step 8 for a 16-wide
  block-diagonal inverse and re-run their GSM8K gate with
  `VLLM_GDN_FORCE_CHUNKED=1`. If accuracy recovers, the diagnosis holds.

---

## 8. Seam analysis

### 8a. Our chunked prefill → their tree

Better than expected. The math and layouts line up; the work is plumbing plus one
real numerical hazard.

**What maps cleanly:**

| Concern | Theirs | Ours | Bridge |
|---|---|---|---|
| q/k/v layout | `[H, T_pad, D]` fp32, head-major, B==1 (`sq = lambda x: x[0]`) | `[B*V_HEADS*S, K_DIM]` 2-D, row = `h*S + t` | a `.reshape(H*T, D)` — same memory order |
| state | `[H, kd, vd]` | `[B, V_HEADS*K_DIM, V_DIM]` | reshape `[H,kd,vd]` → `[1, H*kd, vd]` |
| initial state | APC seed `initial_state[0]`, `[1,H,Dk,Dv]` | `state` is already an **input** | direct — no new code path needed |
| gate | `g_h` `[H, T_pad]` log-decay | `g` `[B*V_HEADS*S, 1]` | reshape |
| mask tiles | `eye`, `tril_incl`, `strict_lower` `[C,C]` fp32, host-built | `eye`, `m_incl`, `m_strict` `[C,C]` | same three tiles, different names |
| head count | TP8 rank → 2 K-heads / **4 V-heads** | `DN_PACK_N=4` wants V_HEADS % 4 == 0 | **exactly one pack** — the packing maps with no remainder |
| padding | host pads `T` to a multiple of `C` | kernel requires `S % C == 0` | keep their host pad |
| sequence parallel | all-gather to full T before the scan, reduce-scatter after | expects full-T contiguous per head | already handled on their side |

The head-count coincidence is worth noting: at TP8 their per-rank DeltaNet width
is 4 V-heads, which is precisely `DN_PACK_N=4` → P=128, the pack width that gave
us +19.4%. It should still be confirmed against `_chunk_pack4`'s named-scalar
`base0..base3` offsets, which are hard-coded 4-way (the NKI index specializer
rejects a slice offset sourced from a list of symints, hence the explicit
unrolling).

**What does not map — three items, in order of hazard:**

1. **⚠️ Double normalization, and an eps-semantics conflict.** Their host prologue
   L2-normalizes q and k and applies `1/sqrt(D)` before the kernel call. **Our
   kernel does both internally** (`_l2norm_rows` on `q_in`/`k_in`, `Q_SCALE =
   1.0/math.sqrt(K_DIM)`) and documents that callers pass **raw** projections.
   Dropping our kernel in behind their prologue normalizes twice.

   Worse, the two use **different eps semantics**, and each tree has a comment
   defending its choice against the other:

   - **Theirs** (`model.py:1192`): `x * rsqrt(x.pow(2).sum(-1) + eps)` — eps
     *inside* the radicand — "to match HF (modeling:231-232 …) and the
     device-proven ancestor …, NOT F.normalize (which clamps the norm to >= eps —
     a different eps semantics)."
   - **Ours** (`_l2norm_rows`, and `static_decode_35b.py:1988`): `F.normalize`
     semantics, `x / max(‖x‖, eps)` — "MUST match torch F.normalize … NOT the old
     x / sqrt(ss + eps). For near-zero rows (conv+SiLU can produce them) the two
     differ by ~1000x … That mismatch = the real-model coherence bug (10%
     near-zero rows → cos 0.95/layer → layer-5 cliff)."

   Both cite device evidence. They cannot both be right for the same model, and
   this is unresolved: HF upstream uses the eps-inside form, so **ours is the
   form that diverges from HF** — yet we measured a real coherence cliff with the
   HF-equivalent form and fixed it by switching. Possible reconciliations: the
   near-zero rows depend on conv implementation/dtype and only arise in our path;
   or one tree has a latent bug elsewhere that the eps choice happens to mask.
   Either way, a transplant must decide this deliberately rather than inherit it,
   and no existing regression on either side would catch getting it wrong. Cheap
   experiment: their tree can A/B the two eps forms on its already-validated
   sequential path in ~5 lines, and GSM8K would show the difference.

2. **Invocation convention.** Ours is registered as a torch custom op
   (`@nki_op("deltanet35b::chunked_prefill")` via `torch_neuronx.nki_op`, in
   `deltanet_chunked_prefill_35b_ops.py`); theirs goes through `wrap_nki` as an
   NKI-HOP indexed by LNC (`_gdn_wrapped[2]`). Re-wrapping is mechanical but is
   also where the LNC=2 assumption lives (§4a).

3. **Configuration by module-level env globals.** Our kernel reads `V_HEADS`,
   `PACK_N`, `CHUNK_SIZE`, `STABLE_C32`, `PAIRED_BATCH`, `STREAM_WINDOW` from the
   environment **at import time** (`kernels/deltanet_chunked_prefill_35b.py:51-74`).
   Their tree configures per-model from `config.py` and per-rank at construction.
   These need reconciling: an import-time global cannot express "4 V-heads at TP8,
   8 at TP4".

4. **Beta pre-multiplication (minor).** They compute `k_beta = key*beta` and
   `v_beta = value*beta` on the host and pass both; ours takes raw `beta [.,1]`
   and applies it internally. Drop their precompute.

5. **State write-back (minor).** They write the returned state into the paged pool
   at slot `block_table[:,0]` via `_seed_state(self.recurrent_state, last_state[:1],
   self._rec_page_stride, self._rec_raw_slab, self._rec_raw_off)`. Our kernel
   returns `new_state` as a plain tensor; their existing epilogue handles the rest
   unchanged.

**Bottom line:** the transplant is plumbing (items 2–5) plus one decision that
must be made consciously (item 1). It is not an algorithm port.

### 8b. Their paged decode state → our repo

This is the direction where they have something we don't: **slot-indexed state,
which is the prerequisite for continuous batching.**

What would have to change on our side:

- **State layout.** Ours is `[B*V_HEADS*K_DIM, V_DIM]`, flat and contiguous, with
  batch offsets resolved at compile time — a static-batch layout by construction.
  Theirs is `[N_SLOTS, H, kd, vd]` with a runtime slot index. Moving to a pool
  means every DeltaNet decode state access in `deltanet_full_batched_v2_35b.py`
  becomes an indirect access, and `static_decode_35b.py`'s state ownership (it
  holds state as aliased module buffers, e.g. `GQA_STATEFUL_KV`,
  `DN_DIRECT_STATE_OUT`) would need a pool allocator behind it.
- **Padding semantics.** Their `PAD_SLOT_ID=-1` → huge uint32 → `oob_mode.skip`
  idiom makes an inactive lane's gather *and* scatter inert, against a pool
  pre-copied to the output. That trick is worth adopting verbatim; it is what lets
  a partially-full batch run without masking arithmetic.
- **Free-axis tiling.** Their `F_TILE=8192` exists because a full head's
  `H*kd*vd` fp32 slab (256 KB) exceeds ~192 KB of SBUF. Our fused decode kernel
  would hit the same wall on a pool-wide copy.

**Is the `.ap` pattern reusable standalone?** Mostly yes — this is the encouraging
part. The kernel's dependency on vLLM is only the `idx_hbm` `[B,1]` int32 tensor
of slot indices; everything else is self-contained NKI. Their own docstring notes
the idiom is the shipping one used by MoE-permute and deformable-attention
scatter, and `slot_indirect_probe.py` exists as a standalone reference for it. So
the indirect-DMA mechanism can be lifted without taking vLLM's block-table
machinery; what *is* entangled with vLLM is everything around it — who allocates
slots, who frees them, and the admission gate
(`vllm/core/scheduler.py::_pool_admission_ok`, +243 lines) that throttles
concurrency to what the pool can hold. Adopting the kernel is tractable; adopting
continuous batching is a scheduler project.

---

## 9. Ranked opportunities

Strongest first. Each is evidence-backed above; sizes are rough.

### 9a. Give them the stable block-diagonal inverse — small change, large unlock

**Evidence:** §7. Their chunked prefill is disabled; the cause is very likely a
numerical scheme we have already replaced on device.

**Why it is the top item:** it is the highest ratio of unlock to effort on either
side of this comparison. Their chunked path being off is what pins them to the
sequential scan, which is what bounds them to ≤512-token segments, which is what
caps `max_model_len` at 1024 — the single constraint that makes §5's
comparability problem unfixable. Fixing the inverse potentially lifts all of it.

**Size:** replacing Step 8 with a 16-wide block-diagonal inverse is on the order
of 40–60 lines of NKI in `gated_delta_rule.py`, reusing their existing
`_transpose_pe` / `nc_matmul` helpers. Validation is their existing GSM8K gate
with `VLLM_GDN_FORCE_CHUNKED=1`.

**Deliverable:** `_tri_inverse_blockdiag`
(`kernels/deltanet_chunked_prefill_35b.py:908`) plus the parameterized test
(`kernels/tests/test_dn_block_inverse_stability.py`), which runs in pure PyTorch
with no device and demonstrates the failure and the fix at C ∈ {16,32,64,128}.

**Caveat to send with it:** §7's "Honest caveats" — this is a prediction from a
CPU reproduction plus our own device experience, not a measurement on their stack.
Send it as a hypothesis with a one-hour test attached, not as a bug report.

### 9b. Take their slot-indexed decode state as our route to continuous batching

**Evidence:** §6b, §8b. We have no continuous batching; they do, and the kernel
mechanism is largely liftable.

**Why it matters:** this is a *capability* gap, not a throughput gap. Our 442.1
tok/s @ BS=128 is a fixed-batch number; it does not describe what we would serve
under a real request queue with mixed sequence lengths. Slot-indexed state is the
prerequisite for finding out.

**Size:** the kernel piece is moderate (adapt `gdn_state_update.py`'s `.ap` +
`oob_mode.skip` idiom to our fused decode kernel — theirs splits conv and state
into separate kernels, ours fuses, so this is a real port not a copy). The
surrounding piece is large: pool allocation, slot lifecycle, and an admission gate.
Recommend adopting the kernel idiom first and treating continuous batching as a
separate project.

### 9c. Reproduce their baseline on a `trn2.48xlarge`

**Evidence:** §3, §4, §10.

**Why third:** it is pure execution — the recipe is frozen and copy-paste (§10) —
but it is gated on hardware we do not have, and what it buys is a *reference
point*, not a capability. Worth doing when a 48xlarge is available; not worth
contorting the TP=4 configuration to approximate (§4a: never validated by them,
19.5 of 24 GB/rank, and their 27B dense TP=4 already failed above concurrency 2 at
a lighter footprint).

**Size:** ~1 day given hardware, most of it cold compile.

### 9d. Lower-priority, noted for completeness

- **Their gating discipline.** Their "gate on GSM8K, not on stock logit
  thresholds" finding (§3) is a real trap with a quantified noise floor
  (BF16-vs-FP32 L-inf 0.03–0.11 against a 0.011 tolerance). Our gate stack is
  already stronger, but the BF16 noise-floor figure is a useful reference number.
- **Resolve the l2norm eps conflict** (§8a item 1). Independently of any
  transplant, one of the two trees is diverging from HF, and ours is the one that
  looks divergent on paper while having device evidence for its choice. Cheap to
  test on their validated path; worth knowing.
### 9e. Components worth adopting — quality-neutral perf, from `paged_kv_gather.py`

Unlike the GDN kernels (§6, §7 — we are ahead on both), `paged_kv_gather.py`
(450 lines) is not an algorithm we already beat. It is **addressing and lowering
technique**, all of it pure data movement (`dma_copy` is a byte mover; dtype
follows the pool; no arithmetic changes), so all three items below are expected
**bit-identical** in output — the class of change that buys throughput without
touching quality. And they target our profiled #1 LNC=1 bottleneck:
`rope_kv_dynamic` returning the whole 20,000-token KV cache (25% of instruction
time, 21.7% of HBM — see the LNC=1 prefill profile). Ranked by how cleanly each
maps to that bottleneck:

1. **In-place alias — persist the KV write without materializing the cache.**
   `paged_kv_write` scatters into the pool and *returns it*, so the compiler
   auto-aliases (`operand_output_aliases={0:0}`) and persists to the
   manager-visible slab with **no fresh alloc or copy**. This is the same
   mechanism our in-flight `group_index` change in `gqa_rope_kv_35b.py` is
   reaching for — that diff's own comment: a per-layer slice's in-place write "is
   only carried back if the tensor is also returned as an op output (which costs a
   full-cache materialization)." First check: confirm our `group_index` path
   actually gets the free alias rather than silently paying the copy; their
   docstring asserts theirs does not.

2. **32-bit indirect-address overflow — a general fix, not a cap.** A flat
   `block*page_stride` int32 offset overflows once `num_blocks*page_stride ≥
   2^31`; neuronx-cc rejects it (`dst_indirect_max_index exceeds 32-bit range`),
   which had forced a `num_blocks ≤ 16383` cap that **halved their usable KV
   cache**. Fix: keep the pool 2-D, free-reshape to `[num_blocks*n_slots,
   head_dim]`, and use a per-token slot index (`slot = block_id*n_slots +
   col_slot + pos`) so the wide multiply happens in the DMA engine's >32-bit
   space — keeping the static `.ap offset=` at zero, because a nonzero static
   offset is added in 32-bit and overflows. This is the highest-signal item: our
   stashed `gqa_rope_kv` work is fighting the same failure class — "combining an
   affine batch index with the runtime row offset makes the driver conservatively
   bound the scalar DMA against the full flattened address range and reject the
   NEFF" — and they have a documented general answer where we have a workaround.

3. **`.ap` indirect gather — lowers where torch fancy-index doesn't.**
   `K_blocks = k_src[flat_idx]` on a page-strided `as_strided` view is rejected —
   "Detected non-contiguous slicing for requested Device Tensor." Expressing the
   same gather as `.ap` indirect DMA with an explicit page stride
   (`vector_offset=block_ids, indirect_dim=0`) compiles, because the stride is a
   pattern literal handed to the DMA engine rather than a property of a strided
   tensor. This is what makes "gather only the blocks you need" possible instead
   of returning the whole cache — the direct attack on the bottleneck.

Tracked as tasks #5 (item 1), #6 (item 2), #7 (item 3).

**Also worth a look — `attention_decode_mask.py` (a bug fix, not a perf win).**
They replaced a hardcoded `lnc = 2` with `lnc = 2 if s_prior % (2*P_MAX) == 0
else 1`, because hardcoding it computes the wrong `sprior_n_prgs` for any
`s_prior` that is a multiple of `P_MAX` but not `2*P_MAX` (e.g. block_size
896 = 7×128): the resize is skipped, `block_len` stays oversized, and the
downstream un-shuffle reshape gets a non-permutation index layout and throws.
This applies only if we hit nkilib's `attention_tkg` decode-mask path — we use
our own `GQATAIL` kernel, so possibly not — but hardcoded-`lnc` assumptions in
nkilib are exactly what bit us before at LNC=1, so it is worth knowing the fix
exists.

**Not worth adopting, to be explicit:** their GDN prefill kernels (§6a — O(T)
sequential capped at 512, or the draft chunked with the broken inverse); their
split conv/state decode kernels (§6b — we fuse, theirs would regress fixed-batch
throughput; the value there is continuous batching per §9b, not speed); and
`VLLM_MOE_TKG_ROUTER_FP32` (a correctness flag — fp32 router is slower than bf16,
a trade in the wrong direction for us).

**Larger lever, separate project — `neuron_mamba_apc.py`** (430 lines, prefix
caching) is a big win with zero quality cost *when prompts share prefixes* — high
ceiling for 20k-context work — but they validated it functionally only (one
817-token prefix at BS=4, no accuracy or throughput number) and it is substantial.
Per §9b, continuous batching is a scheduler project; APC rides on the same
machinery.

**Calibration.** None of the above is measured on our stack; the ranking is by fit
to the profiled bottleneck, not observed gain. The KV-gather items are the ones
most likely to hold up, because the failure modes they document are ones we have
independently hit.

---

## 10. Appendix — ready-to-run recipe

Their published recipe, verbatim, with local adaptation notes. This is the target
to reproduce (§9c) once §4's blockers clear. Their commands are quoted from
`Qwen3.6-35B-A3B/README.md` at `65ef8b7`.

### Environment

```bash
neuron-ls   # expect 16 NeuronCores on trn2.48xlarge

export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_NEURON_COMPILATION_TIMEOUT=1200
export VLLM_CACHE_ROOT=/path/to/scratch/vllm_cache   # large volume, NOT /

# All three are default-OFF and load-bearing:
export VLLM_GDN_SEQ_NKI=1          # bounded-graph GDN prefill (required for seq >= 512)
export VLLM_UNIFIED_KV_GATHER=1    # block-indexed unified KV gather (full-attn layers)
export VLLM_MOE_TKG_ROUTER_FP32=1  # fp32 decode router for correct top-8 selection
```

Two traps they call out explicitly:

- **Core selection:** use `NEURON_VISIBLE_DEVICES` (e.g. `0-7`). Do **not** use
  `NEURON_RT_VISIBLE_CORES` for a multi-process TP serve — it hard-errors with
  "cannot be used with multi-processing execution".
- **Compiler flags:** leave `NEURON_CC_FLAGS` **unset**. Setting it *replaces*
  rather than appends to the framework's flag string and drops flags the model
  requires (e.g. `--enable-verifier=false`). Relocate caches with
  `VLLM_CACHE_ROOT` instead.

### Checkpoint

```bash
hf download Qwen/Qwen3.6-35B-A3B --local-dir /path/to/Qwen3.6-35B-A3B
```

> **Local adaptation:** not needed here — our existing
> `QWEN35_MODEL_DIR` BF16 checkpoint serves this port unchanged (same
> `qwen3_5_moe` architecture; their README lists Qwen3.5-35B-A3B as "BF16 (same
> arch)"). Point `vllm serve` at it directly. Note `huggingface-cli` is removed on
> the pinned `huggingface_hub` 1.x — use `hf download`; set
> `HF_XET_HIGH_PERFORMANCE=1` for faster transfers.

### Online serve

`--hf-overrides` and `--limit-mm-per-prompt` are **mandatory** — omitting the
`Qwen3_5MoeForCausalLM` override forces the multimodal config path (fp32 SSM cache
→ block size 384 → compile-time OOM).

```bash
vllm serve /path/to/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-model-len 1024 \
    --max-num-batched-tokens 512 \
    --max-num-seqs 1 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --hf-overrides '{"architectures":["Qwen3_5MoeForCausalLM"]}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "bf16",
            "ep_degree": 8,
            "on_device_sampling_config": {"all_greedy": "true"},
            "kv_segment_size_buckets": [512],
            "num_batched_tokens_buckets": [512],
            "num_seqs_buckets": [1]
        }
    }'
```

> **Bucket sizing:** `kv_segment_size_buckets` and `num_batched_tokens_buckets`
> must be **≤ 512** so no 1024-extent prefill graph is traced — this is the
> sequential-GDN bound from §6a. Each bucket adds compile time and changing any
> bucket forces a full cold recompile. Freeze one recipe; do not sweep.

For their concurrency-4/8 throughput rows, keep `--max-model-len 1024` with
`kv_segment_size_buckets:[512]` and raise the batch — `--max-num-seqs 8` plus
`num_seqs_buckets: [8]`.

### Offline

```python
import os
os.environ.setdefault("VLLM_GDN_SEQ_NKI", "1")
os.environ.setdefault("VLLM_UNIFIED_KV_GATHER", "1")
os.environ.setdefault("VLLM_MOE_TKG_ROUTER_FP32", "1")

from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Qwen3.6-35B-A3B",
    tensor_parallel_size=8,
    enable_expert_parallel=True,
    max_model_len=1024,
    max_num_batched_tokens=512,
    max_num_seqs=1,
    limit_mm_per_prompt={"image": 0, "video": 0},
    hf_overrides={"architectures": ["Qwen3_5MoeForCausalLM"]},
    additional_config={
        "neuron_config": {
            "quantization": "bf16",
            "ep_degree": 8,
            "on_device_sampling_config": {"all_greedy": "true"},
            "kv_segment_size_buckets": [512],
            "num_batched_tokens_buckets": [512],
            "num_seqs_buckets": [1],
        }
    },
)
out = llm.generate(["What is the capital of France?"],
                   SamplingParams(max_tokens=200, temperature=0.0))
print(out[0].outputs[0].text)
```

`examples/vllm_neuron/models/qwen3_5_moe/run.py` is the same thing ready to run
(it sets the three flags via `os.environ.setdefault`).

### Benchmark (reproduces §3's throughput table)

`vllm bench serve` against the running server for the online rows, `vllm bench
throughput` for the offline row: random dataset, 256-token input / 128-token
output (`--random-range-ratio 0`), `--ignore-eos`, 24 prompts per online
concurrency level, 16 offline.

### Accuracy gate (reproduces §3's 93.0%)

`lm_eval --tasks gsm8k_cot` against the server's OpenAI-compatible endpoint: 100
questions, 4-shot (`--apply_chat_template --fewshot_as_multiturn`),
`max_tokens 448`, `max_length 1024`, `num_concurrent 1`, and
`chat_template_kwargs: {"enable_thinking": false}` so the generation budget goes
to the answer rather than a truncated `<think>` block.

Two failure modes they flag: pass the value of `--served-model-name` as lm_eval's
`model=` argument (not the checkpoint path) or every request 404s; and keep
few-shot prompt + generation inside the 1,024-token window or requests are
rejected for exceeding `max_model_len`.

**Do not** gate on `run_logit_validation_offline.py`'s stock thresholds — see §3.
