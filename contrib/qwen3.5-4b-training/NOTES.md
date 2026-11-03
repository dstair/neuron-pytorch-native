# Lab notebook

Measured findings, things that broke, and levers tried and refuted. Newest first within a
section. Anything asserted here has a command next to it.

## Eval methodology rework: honest denominators, CIs, truncation, paired tests

Four harness fixes in `sqlft/eval_sql.py`, each with a self-test gate and a measured
effect on the reported numbers:

1. **The bind quarantine is now output-independent.** `schema_unusable` used to be marked
   only when the model's output PARSED, so a worse model quarantined fewer rows and the
   bind denominator moved with the model (observed: 27 gold / 26 model, then 100 vs 99
   after the first fix). The quarantine is now computed from the schema alone, BEFORE the
   parse -- and before the empty-output early return, which was the last hole: an untuned
   model emits comment-only output that `extract_sql` reduces to `""`, and those rows
   skipped the schema check entirely (found when the baseline still reported 37 instead
   of 39). Verified: `evaluate()` reports exactly 39 quarantined on these 128 rows for
   ANY output (`''`, `'SELEC broken'`, `'SELECT 1'`). Also quarantined now: the **10** eval
   rows that ship no schema at all (~8%; there is nothing to bind against). `n_bindable`
   is 89 for gold, baseline and fine-tuned alike.
2. **Wilson 95% CIs on both rates.** parse 0.883 at n=128 is [0.816, 0.928] -- the point
   estimate alone always overstated its precision. Every `evaluate()` result carries
   `parse_ci95` / `bind_ci95`.
3. **Truncation is its own stage.** Generation reports `finished` (EOS inside the cap);
   a parse failure on an unfinished row is a decode-cap casualty, not a syntax error.
   Measured: the fine-tuned model's "15 parse errors" decompose into **14 truncations +
   1 genuine parse error**. And 17/128 GOLD queries exceed the 48-token cap, so those
   rows are unwinnable by construction (`n_gold_over_cap` is reported).
4. **The EOS confound is a number now, and paired tests replace point-estimate
   comparisons.** `first_stmt_parse_rate` grades only the first statement: baseline
   0.695 vs parse_rate 0.539 -- 16 points of the baseline's "failure" is not stopping,
   not bad SQL (fine-tuned: 0.930 vs 0.883). Every run stores `per_row` outcomes +
   generated SQL; `python3 -m sqlft.eval_sql --compare A.json B.json` runs McNemar's
   exact test on the discordant rows. Baseline vs fine-tuned: 3 vs 47 discordant,
   p < 0.001. Same model twice: 0 discordant -- decode is deterministic given a warm
   compile cache, which is also the test's own sanity check.

Final v5 numbers, 128 rows, cached decode, max_new 48:

| model | parse [CI] | first_stmt | bind [CI] (of 89) | stages |
|---|---|---|---|---|
| gold ceiling | 0.984 | -- | -- | 2 parse errors |
| untuned | 0.539 [0.453, 0.623] | 0.695 | 0.315 [0.228, 0.417] | ok 28, trunc 59, bind_err 20 |
| + 1 epoch LoRA | 0.883 [0.816, 0.928] | 0.930 | 0.652 [0.548, 0.743] | ok 58, trunc 14, bind_err 23, parse_err 1 |

Sources: `runs/lab01_baseline_128_v5.json`, `runs/lab07_eval_ft_128_v5.json`.

Caution reading across harness versions: pre-v5 numbers counted truncations as parse
errors and used floating bind denominators, so e.g. the historical 0.906/0.898/0.883
fine-tuned parse rates and the 0.570 baseline are not exactly comparable to v5's. The
v5 numbers are the canonical ones going forward.

## Recurrent-cache decode: eval 31.4 → 10.4 min (3.0×), logit-parity gated

The O(n²) re-forward eval is replaced by `sqlft/infer.py`: prefill once through the kernel
repo's INFERENCE kernels (nkilib `gdn_cte`, state out), then single-token steps
(`deltanet_tkg_batched_bh`, state in/out) for the 24 GDN layers, with a fixed-size
preallocated KV buffer for the 8 full-attention layers -- every decode step is the SAME
graph, so it compiles once. `NeuronGatedDeltaNet.forward` (v15) binds onto the stock HF
modules (identical weight layout); LoRA is merged+unwrapped first (`lora.merge_and_unwrap_lora`)
because the v15 fused in-proj reads `.weight`. The transformers 5.22 cache API does not
match what v15 expects, so `infer.DecodeCache` is a 40-line object implementing BOTH
interfaces (v15's `layers[i].conv_states/recurrent_states` + `update_*_state`, and HF
attention's `update(k, v, layer_idx)` returning full static buffers; the additive mask
decides visibility, which is what keeps shapes static).

Validation (`labs/lab09_cached_decode_parity.py`, teacher-forced logits vs the training-kernel
re-forward, 8 rows x 11 steps): **top-1 agreement 88/88 steps, worst cos 0.99881**. The cos
band is 0.998, not lab08's 0.999: that band was calibrated across layouts of ONE kernel path,
this one compares across kernels (split16 training vs nkilib gdn_cte + tkg), whose floor is
higher. The one sub-0.999 step recovers fully the next step (0.99881 → 0.99984), which
rules out state corruption -- corruption does not recover.

End to end, 128 rows, ft_epoch1 adapter, max_new 48, fresh box (compiles included):

| decode | time | parse_rate | stages |
|---|---:|---:|---|
| re-forward, padded bs=8 | ~50 min | 0.906 | ok 61 / parse_err 12 |
| re-forward, bucketed bs=1 | 31.4 min | 0.898 | ok 60 / parse_err 13 |
| **cached recurrent** | **10.4 min** | **0.883** | ok 61 / parse_err 15 |

Sources: `runs/lab07_eval_ft_128{,_fast,_cached}.{json,log}`. The 0.898 → 0.883 move is 2
queries of 128 -- inside the ±5pt binomial noise at n=128, and the same magnitude as the
1-query move the decode LAYOUT change caused (0.906 → 0.898), which lab08 showed is
greedy-chaos noise, not a model difference. `ok` went 60 → 61: rows flip in both
directions. Steady state after compiles is ~4 s/prompt (vs ~12): prefill ~1 s + ~24 tokens
x ~0.1 s/step.

Also measured here, confirming a known harness caveat: `schema_unusable` dropped 26 → 23
because more rows failed parse BEFORE reaching schema creation, so fewer got quarantined --
the bind denominator depends on the model's parse success. That is the denominator
inconsistency flagged for the harness rework; read bind_rate across runs with care until
it is fixed.

## Upstream kernel v15: still not accurate enough here; `nki-v15-split16` is the default

v15 (`jburtoft/qwen35-gdn-deltanet-neuron-kernels`, commit `36d9feb8`; no v15 tag) carries
upstream's own fix for the v3.0 failure since v8: row-block forward substitution at `BLOCK=32`.
v15's headline prefill fix (nkilib `gdn_cte`) is inference-only; the training kernels are
unchanged since v14.

**In situ it is close but not exact.** `labs/lab02_gdn_in_situ.py --layers 32 --seq 512
--variant nki-v15`, loader seed 1: 22 of 24 GDN calls at cos ≥ 1.00002, but **call 19 0.9836
and call 20 0.9910**. The same batch under `nki-patched` matches every call (19: 1.000042,
20: 1.000025), so it is v15, not the batch.

**Isolated on call 19's captured inputs** (`--dump 19`, then
`labs/lab03_patched_kernel_parity.py --grads --repro /work/runs/gdn_repro_call19.pt`):

| build | fwd | dq | dk | dv | dg | dbeta |
|---|---:|---:|---:|---:|---:|---:|
| v3.0 | −0.077006 (\|max\| 5.4e10) | 0.039 | −0.004 | 0.034 | 0.031 | −0.079 |
| v3.0 + blockdiag | 1.000047 | 1.000049 | 1.000091 | 1.000058 | 1.000000 | 1.000000 |
| v15 | 0.997881 | 0.998370 | 0.992111 | **0.925385** | 0.986994 | **0.943209** |
| v15 + split16 | 1.000047 | 1.000049 | 1.000091 | 1.000058 | 1.000000 | 1.000000 |

The gradients are the problem: the backward consumes the forward's saved `N_stack`, so a 0.998
forward becomes 0.925 on `dv`. Upstream's own validation is on Qwen3.5-2B/0.8B; this is 4B.

On call 5 (this box's data order) **all four builds pass**, v3.0 included — so the original
layer-5 capture from the old box is not the discriminating case any more; call 19 is. The
`gdn_patched` gate now uses it.

**Refuted: `BLOCK=16`.** The obvious one-constant change does not compile:
`'nisa.tensor_copy' op Expected 'src' partition start in {0, 32, 64, 96} for a 16-partition
access, but got 16`. Upstream hit the same wall on SDK 2.31 and it is still there on 2.33.

**The fix that works** (`sqlft/kernel_patch/apply_v15_split16.py`): keep 32-row blocks, but
invert each 32×32 diagonal block with masks instead of partition slices — `D` = the two 16×16
diagonal blocks, `X` = the lower-left 16×16 corner, doubling on `D` only (3 rounds; powers of a
block-diagonal matrix stay block-diagonal, so every squaring is effectively 16-wide), then the
exact merge `(I-A)⁻¹ = D⁻¹ + D⁻¹XD⁻¹`. Masks come from `nisa.iota` + arithmetic, like upstream's
`emit_tri_masks`. In situ, all 24 calls ≥ 1.000019.

**Speed**, 32L LoRA, seq 512, single core, public stack, same batches (`runs/kab_*.log`):

| build | warm s/step | tok/s |
|---|---:|---:|
| `nki-patched` | 1.174 | 436.2 |
| `nki-v15` | 1.171 | 437.2 (one 1.31 s outlier step) |
| **`nki-v15-split16`** | **1.148** | **445.9** |

2.2% over `nki-patched`. Small, because the GDN kernel is a minority of the step; v14's
backward speedup shows up mostly inside the kernel. ALL GATES PASS on the new default
(`gates-split16.log`), including `neumann` now that `gdn_repro_call5.pt` exists on this box.

Worth reporting upstream: call-19 capture + `apply_v15_split16.py` is a complete bug report, and
the split is a drop-in for the diagonal-block solve in both the training forward and (likely)
the legacy prefill path, which uses the same `BLOCK=32` solver.

## Public beta stack (Neuron SDK 2.33.0) — 1.7× faster than beta6, same numerics

`STACK=public` (now the default) is `public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04`,
pinned by digest `sha256:b4648437…`. Torch 2.14.0, torch-neuronx 2.14.3, nki 0.7.0, neuronx-cc
2.28.7631, Python 3.14 in the system interpreter (no `/opt/torch-neuronx/.venv`). The derived image
also picked up transformers 5.20.2 and duckdb 2.0.0. `STACK=beta6` keeps the old build, and each
stack has its own NEFF cache (`cache-$STACK`).

A/B on the same box, same batches, 32L LoRA, `--gdn nki-patched`, single
core, `LNC=2`:

| stack | warm s/step | tok/s | cold compile | step-0 loss |
|---|---:|---:|---:|---:|
| beta6 | 1.986 | 257.8 | 341.9 s | 0.9673 |
| **public** | **1.171** | **437.1** | **60.7 s** | 0.9673 |

Step-0 loss is identical and steps 1–5 agree to 1–3% (bf16 noise). All gates pass on public except
`neumann`, which needs `runs/gdn_repro_call5.pt` — a capture that was never copied to this box, not
a regression. Sources: `runs/gate_train_32L_patched.log`, `runs/b6_train_32L_patched.log`.

Full-FT FSDP2 world=4 on public: **1034.5 agg tok/s**, 1.980 s/step, 16.65 GiB/rank
(`runs/lab05_public_fsdp2_fullft_w4.json`) — against 577.1 on beta6, but that beta6 number is from
the old box, so treat the 1.8× as indicative.

The step-0 loss here (0.967) differs from the 0.50 recorded below because this box's data order
differs; both stacks agree with each other, which is the comparison that matters.

## Tensor parallel: runs, and loses on this model at this size

`labs/run_lab05_tp.sh 32 512 6`, full-FT, lr 1e-5, world=4, `LNC=2`, public stack:

| config | dp × tp | warm s/step | agg tok/s | peak HBM/rank |
|---|---|---:|---:|---:|
| FSDP2 | 4 × 1 | 1.980 | **1034.5** | **16.65 GiB** |
| FSDP2 + TP | 2 × 2 | 2.199 | 465.6 | 18.13 GiB |
| FSDP2 + TP | 1 × 4 | — | — | **OOM** |

TP is worse on **both** axes, and the reason is structural rather than a backend problem:

- **Only half the model is TP-sharded.** `apply_tensor_parallel` covers the MLPs and the 8
  full-attention layers. The 24 GatedDeltaNet layers and the 0.636 B tied embedding stay
  replicated across the tp group (sharding GDN needs the conv groups and the kernel's head
  layout resharded too). With tp=2 the dp degree halves, so FSDP shards that replicated
  remainder 2 ways instead of 4 — peak memory goes **up**, not down.
- **tp=4 leaves dp=1**, so FSDP shards nothing: the replicated ~1.6 B params carry full
  bf16 weights + grads + fp32 AdamW on every rank, which does not fit 24 GiB.
- **Throughput halves** because each tp pair processes one batch where FSDP2 processes two, and
  pays an all-reduce per MLP / attention block on top. Per dp replica: 232.8 tok/s for the tp=2
  pair against 258.6 per core under FSDP2.

So for a 4.2 B model on one chip, FSDP2 alone is the right answer. TP earns its keep when one
layer's weights or activations do not fit a core, or when dp alone cannot scale further — neither
is true here.

Correctness: forward and backward run; step-0 loss 0.9764 under tp=2 vs 0.9673 unsharded on the
same batch (0.9%, from the row-parallel bf16 all-reduce changing summation order). Later steps are
not comparable because dp=2 vs dp=4 trains on different data. A forward-only logit parity test
against the unsharded model is the proper check and has not been written.

Two code changes were needed:

1. **Clip across meshes.** TP params are 2D `(dp, tp)` DTensors, everything else is 1D `(dp,)`;
   stock `clip_grad_norm_` dies with `All operands in aten.stack.default must have the same mesh`.
   `sqlft.parallel.clip_grad_norm_` takes one norm per mesh, `full_tensor()`s each, and combines.
2. **Seed and count by dp rank.** `lab04` seeded the loader by global rank, which would have fed
   the two halves of a tp group *different* batches — silently wrong, not a crash. It now seeds by
   `rank // tp`, and aggregate tok/s counts dp replicas, not ranks.

LoRA + TP is refused explicitly: the TP plan targets `nn.Linear` leaves that LoRA has already
wrapped, and `lora_B` would need the matching shard.

## Environment reconciliation (beta3-era recipe → beta6 container)

The earlier bring-up of this model ran on a **beta3** container (torch 2.11, nki 0.4). This
box has **beta6** (torch 2.14.0, torch-neuronx 2.14.3, neuronx-cc 2.0.404830, nki 0.7.0b1).
Four of the inherited environment settings were wrong; `labs/_probe_flags.py` reads the
build's own sources to check which env vars it actually honours.

| setting | beta3 | beta6 | resolution |
|---|---|---|---|
| `TORCH_NEURONX_ENABLE_HOST_CC=1` | worked | **hard error** unless `TORCH_NEURONX_ENABLE_STREAMS=1` too | dropped; revisit as a pair for multi-rank, with a before/after number |
| `TORCH_NEURONX_ENABLE_ASYNC_NRT=1` | a flag | **absent from the sources** | dropped — it was silently doing nothing |
| `NEURON_RT_VIRTUAL_CORE_SIZE` | selected LNC | **absent from the sources** | use `NEURON_LOGICAL_NC_CONFIG` |
| mount host `/opt/aws/neuron` | required | **breaks the container** | do not mount; see below |

### Do not mount the host's `/opt/aws/neuron` into the beta6 container

Every older deploy script in this repo does. On beta6 it produces:

```
ImportError: /workspace/torch_neuron_eager/torch_neuronx/_C.so:
             undefined symbol: nrta_tensor_cast, version NRT_3.0.0
```

The container bundles a **newer** NRT than the host's runtime-lib 2.34.10, so overmounting
the host libs downgrades it below what `_C.so` needs. `--privileged` alone is enough to get
the device in. `MOUNT_HOST_NEURON=1` restores the old behaviour if a future image needs it.

### Host driver is older than the container wants (non-fatal)

```
WARN NRT:nrta_tensor_write  needs asynchronous IO support in the Neuron Driver ...
                            Falling back to runtime threaded asynchronous implementation.
```

`aws-neuronx-dkms` on the host is 2.30.2.0. This affects host↔device transfer throughput
only, and training is device-resident, so it is noise for our numbers — but it does mean
**H2D/D2H timings measured here are pessimistic**. Do not quote them.

### A venv cannot inherit the container's torch; use a derived image

`python3 -m venv --system-site-packages` off `/opt/torch-neuronx/.venv` inherits the **base
interpreter's** site-packages, not the venv's, so torch vanishes:
`ModuleNotFoundError: No module named 'torch'`. And `pip install --target` re-resolves
dependencies and drags in a second CPU-only torch wheel that shadows torch-neuronx.
`env/Dockerfile` installs straight into the container's venv instead, where pip sees torch
2.14 as already satisfied.

## GDN kernel

### ✅ FIXED: the GDN kernel bug was the fp32 Neumann instability this repo already knew about

**Root cause.** `deltanet_fused_chunked_fwd_batched` builds the intra-chunk triangular inverse
with full-width Neumann power-doubling at **C=128**:

```
N = (I + A)(I + A^2)(I + A^4) ... (I + A^64)
```

`A` is strictly lower triangular, hence nilpotent, so this is exact in exact arithmetic — and
6 rounds genuinely reaches A^127, so it is **not** a truncation bug. It is a **stability** bug:
chaining 6 squarings in fp32 loses the cancellation the series depends on.

This repo had already hit, root-caused and fixed the identical failure in the **27B/35B prefill**
kernels — which is why those ship **C=32** with `_tri_inverse_blockdiag`
(`contrib/qwen3.6-35b-a3b/kernels/deltanet_chunked_prefill_35b.py`,
`VLLM_NEURON_ASSESSMENT.md` §7, `kernels/tests/test_dn_block_inverse_stability.py`). The
trainable kernel runs C=128, the worst cell in that table.

`labs/lab03_neumann_stability.py` reproduces it on the **real** Qwen3.5-4B chunk matrices from
the first diverging layer — max-abs error of `(I-A)^-1` vs a float64 forward-substitution oracle:

| C | full-width doubling | block-16 factorization |
|---:|---:|---:|
| 16 | 7.15e-07 | 7.15e-07 |
| 32 | 2.31e-05 | 7.82e-07 |
| 64 | 7.89e-03 | 1.16e-06 |
| **128** ← the kernel | **1.08e+03** | **1.29e-06** |

One honest difference from the repo's synthetic worst case: there the intermediate powers reach
8.8e+32 and overflow the mantissa; on real 4B data peak ‖A^(2^k)‖ is only **3.26**, so this is
pure accumulated cancellation, not overflow. Same conclusion, milder mechanism — and it means a
magnitude-based guard would not have caught it either.

This also explains why every sweep I ran missed it. Real `g` has **mean −0.19** (near-1 decay,
weakly damped); my synthetic `-rand()*g_scale` has mean −0.5·g_scale. More-negative `g` means
more damping, hence a *smaller* A — so **raising `g_scale` made the kernel more accurate**, which
is why that sweep never broke. Real data is the weakly-damped, badly-conditioned case, and it
needs real `k` too because `A` is built from `k_beta @ kᵀ`: real keys are correlated with small
norms, unlike iid `randn`.

**The fix** (`sqlft/kernel_patch/apply_blockdiag_fix.py`) generalises `_tri_inverse_blockdiag`
— which is 2-way and C=32 only — to a recursive 2-way split down to 16-wide diagonal blocks.
At each level `A = D + X` with `D` the diagonal half-blocks and `X` the strictly lower-left
half-block; `((I-D)^-1 X)^2 = 0`, so exactly

```
(I - A)^-1 = (I-D)^-1 + (I-D)^-1 X (I-D)^-1
```

Recursing 16 → 32 → 64 → 128 keeps every squaring 16-wide, so intermediates never grow. Cost is
comparable — 3 squarings for the 16-wide doubling plus 2 matmuls per merge × 3 merges, against
the original 6 squarings + 6 accumulations. The backward kernel consumes the forward's saved
`N_stack` rather than recomputing the inverse, so fixing the forward fixes all six gradients.

**Verified.** `labs/lab03_patched_kernel_parity.py`, upstream vs patched vs HF reference:

| case | build | cos vs ref | \|max\| |
|---|---|---:|---:|
| synthetic | upstream | 1.000125 | 0.08294 |
| synthetic | patched | 1.000125 | 0.08294 |
| **real (diverging layer)** | upstream | **−0.000591** | **7092** |
| **real (diverging layer)** | patched | **1.000289** | 0.05839 |

`labs/lab02_gdn_in_situ.py --patched`: **all 24 GDN layers** match the reference (cos ≈ 1.0003),
against upstream breaking at call 5 and reaching inf by call 17.

End-to-end, full 32L LoRA, seq 512, bs 1, single core — the patched kernel is correct **and**
faster than the reference it replaces:

| GDN path | warm s/step | tok/s | step 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|---:|---:|
| `torch` reference | ~3.40 | ~151 | 0.5031 | 1.0049 | 0.8884 | 1.8015 |
| `nki-patched` | **1.96** | **~260** | 0.5106 | 1.0076 | 0.8930 | 1.7991 |
| `nki` upstream | — | — | NaN | | | |

The loss trajectories agree to ~0.5%, which is the end-to-end confirmation that matters.
**1.7× faster than the correct reference**, so the performance lever is recovered.

Worth reporting upstream: the kernel is published as validated at "cos_sim = 1.000000 across
22+ configurations", and that is true and insufficient — see the testing-gap note below.

### The original diagnosis (kept for the reasoning trail)

**Resolved — see the section above for the root cause and fix.** Use `--gdn nki-patched`.
`--gdn nki` is upstream and remains broken; `--gdn torch` is the correct-but-slow reference.
What follows is the diagnostic trail that led there, kept because the sequence of wrong guesses
is the useful part.

What happens, at full 32 layers (`labs/lab02_gdn_in_situ.py` runs both implementations on the
model's own tensors, per GDN call):

| GDN call | reference \|max\| | NKI \|max\| | cos |
|---|---|---|---|
| 0–4 | ~3e-2 | ~3e-2 | ≈1.000 |
| **5** | 5.8e-2 | **7.1e+3** | **−0.0006** |
| 9 | 2.8e-2 | 1.6e+07 | 0.0005 |
| 11 | 4.0e-2 | 1.6e+10 | 0.0004 |
| 15 / 16 | ~1e-1 | 9.7e+24 / 5.9e+31 | 0.000 |
| 17+ | 4.6e-2 | inf | nan |

From the 6th GatedDeltaNet layer the output is **uncorrelated with the truth**, on finite
inputs, then compounds to inf and the loss goes NaN. Only call 5 is diagnostic; everything
after it is fed corrupted hidden states.

End-to-end, same seed-0 batch, 32 layers, LoRA:

| GDN path | step 0 loss | steps 1–3 |
|---|---|---|
| `torch` (HF reference) | **0.5031** | 1.00, 0.89, 1.80 — all finite |
| `nki` | 1.0299 | **NaN** |

Note the kernel is wrong even where it is finite — 1.03 vs 0.50 on the identical batch. "It
produced a plausible-looking loss" was not evidence of anything.

**Why the synthetic parity test missed it.** Everything I swept was the wrong axis:

| swept | result |
|---|---|
| shape: T ∈ {256, 512, 1024} × H ∈ {2, 8, 32} | cos ≥ 0.999999 everywhere |
| log-decay magnitude: `g_scale` ∈ {1, 4, 16, 64, 256} | cos ≥ 0.999999 everywhere |

It is the input **distribution**, and `labs/lab02_gdn_repro.py` pins it down by ablating the
captured real tensors one at a time:

| configuration | cos | verdict |
|---|---|---|
| all real | −0.000591 | DIVERGES |
| all synthetic | 1.000243 | OK |
| real, but **k** ← synthetic | 1.000221 | **OK** |
| real, but **g** ← synthetic | 1.000258 | **OK** |
| real, but q ← synthetic | 0.003236 | DIVERGES |
| real, but v ← synthetic | 0.000574 | DIVERGES |
| real, but beta ← synthetic | 0.023348 | DIVERGES |
| synthetic, but any ONE tensor ← real | ≈1.0 | OK (all five) |

So it needs **real `k` and real `g` together**, and no single real tensor is sufficient.
Mechanism, not fully confirmed: the chunked gated-delta-rule forms a block inverse of a matrix
built from `K`, and real `k` has very small row norms (std 0.07) while real `g` spans
[−15.04, −6e−6]. That combination appears to make the kernel's inversion unstable where the
reference's is not.

Tested and **refuted** as the mechanism: exact-zero rows from padding. Real q/k/v are ~29.5%
exactly zero (HF's `apply_mask_to_padding_states` zeroes padded positions), which looked like
the obvious culprit — but running with `--packed` (0% padding, attention mask all ones) still
NaNs at step 0. Do not re-chase padding.

Reproducer for a bug report: `runs/gdn_repro_call5.pt` (25 MB, the exact q/k/v/g/beta) plus
`labs/lab02_gdn_repro.py`.

**Consequence for earlier numbers.** Every throughput figure measured before this was on the
NKI path. They remain valid as *throughput* (same shapes, same FLOPs) but any **loss** value
from an 8-layer NKI run is not convergence evidence: 8 layers = 6 GDN layers = calls 0–5, so
call 5 was already corrupt. The results table now carries torch-path numbers.

### Parity on synthetic inputs (necessary, and nowhere near sufficient)

The kernels-hub repo documents itself against "PyTorch Native Beta 3 (nki 0.4.0) or Beta 4
(nki 0.5.0), SDK 2.29–2.31". This box is nki 0.7.0b1 / torch 2.14, so this was the largest
risk in the plan. It compiles and runs correctly:

```
$ ./env/container.sh exec python3 -m sqlft.gdn --parity --device neuron --seq 256 --heads 2
fwd   cos=1.000000  max|d|=9.872e-08
bwd   dq    cos=1.000002   dk cos=1.000004   dv cos=1.000004
bwd   dg    cos=1.000000   dbeta cos=1.000000
GDN PARITY OK
```

Backward parity matters more than forward here: a forward-only check would have passed even
if the VJP were wrong, and the whole point of this kernel is that it trains.

The lesson, recorded because it cost most of a day: **cos = 1.000000 on synthetic tensors of
the right shape says almost nothing.** The first real test of a kernel swap should be
in-situ against the reference on the model's own activations, which is ~40 lines
(`labs/lab02_gdn_in_situ.py`) and would have caught this immediately.

### Swap at the narrowest seam: one module-level function

`Qwen3_5GatedDeltaNet.forward` does the projections, causal conv1d, beta/g gating, GQA head
expansion, gated RMSNorm and output projection itself, and calls out **exactly once** for
the recurrence — `torch_chunk_gated_delta_rule(q, k, v, g=…, beta=…, use_qk_l2norm_in_kernel=True)`.
Replacing that single module-level function leaves every other line of HF's layer alone, so
checkpoints load unmodified and LoRA targets keep working. Replacing the whole module (which
the kernel repo's `layers.py` does for *inference*, via `KernelConfig`) would have meant
re-deriving the weight layout.

`use_qk_l2norm_in_kernel=True` means HF passes **unnormalised** q/k and expects the callee to
do the l2-norm and the `1/sqrt(head_dim)` scale — which is exactly what the kernel's
`forward` does, so the contract lines up with no adapter math.

Constraints inherited from the kernel: `head_dim` must be 128 (Qwen3.5's GDN heads are 128 —
its *GQA* heads are 256, which is a different code path), sequence length must be a multiple
of 128, and internals are fp32.

## Dataset

### seq_len is a direct trade against dataset coverage

`python3 -m sqlft.data --stats` over 4000 train rows — prompt+completion tokens:
mean 388, p50 354, p90 676, p95 794, p99 967, max 1271.

| seq_len | rows kept |
|---|---|
| 512 | 75.7% |
| 768 | 94.2% |
| 1024 | **99.8%** |
| 2048 | 100% |

Rows over budget are **dropped, not truncated**: truncating cuts the tail off the SQL and
teaches the model to emit unparseable fragments, which is the exact metric we are moving.

This sets up the central tension of the memory labs — seq 1024 is the length we want for
coverage, and it is the length the earlier bring-up reports as OOM on one core even under
LoRA. Activation checkpointing and packing are the levers.

Padding waste at seq 1024 on a median row is **37.2%** of positions. `PackedSqlDataset`
takes that to 0% at the cost of examples bleeding across block boundaries.

### The metric has a ceiling of 0.987, not 1.0

**2.78% of the dataset's gold SQL does not parse in DuckDB 1.5.5** (694/25000). Real
examples: `COPY … (FORMAT CSV, DELIMITER=',' …)` (invalid option syntax), an unbalanced
subquery paren, and one row whose "SQL" is `import ibis; duckdb_conn = ibis.duckdb.connect(…)`.
So a model that reproduced the targets perfectly would score ~0.987. Quoting a model number
without that context invites the wrong conclusion.

Train rows with unparseable gold are filtered out (`filter_parseable`); eval rows keep their
noise so the reported ceiling stays honest.

### `extract_statements` is not purely syntactic

The first version of the harness counted every DuckDB exception as a syntax error and scored
gold at 0.983 — the self-test correctly refused to certify it. Cause:
`PRAGMA drop_fts_index('artist')` and `DROP EXTENSION tpch` raise `CatalogException` /
`NotImplementedException`, not `ParserException`. Those are well-formed SQL referring to
absent objects. Only `ParserException` counts against `parse_rate` now.

Rejected alternative: `json_serialize_sql()` is a clean pure-parser check but **SELECT-only**,
and this dataset is full of `PRAGMA`, `CREATE`, `COPY`.

### ~25% of dataset schemas do not create

369/1500 sampled `CREATE TABLE` blocks fail — mostly foreign keys referencing tables the
generator never emitted, plus duplicate column names. Those rows can never bind no matter
what the model emits, so they are quarantined out of the `bind_rate` denominator instead of
being charged to the model. On a 300-row eval sample that is 64 rows; `bind_rate` goes from
a misleading 0.530 to 0.674 of 236 genuinely bindable rows.

## Memory

`python3 -m sqlft.model --report`, from the config rather than from a measurement:

```
params total         4.206 B   (embedding 0.636 B = 15.1%, tied with the LM head)
bf16 weights          7.83 GiB
+ bf16 grads          7.83 GiB
+ AdamW fp32 m,v     31.34 GiB
= full-FT state      47.00 GiB   vs 24 GiB per logical core at LNC=2
```

So full fine-tuning cannot run on one core, and FSDP is the *enabler* rather than a
speedup. Independently confirms the earlier bring-up's ~50 GiB estimate.

Still to measure: the **248,320-vocab logits tensor**. At bs1 × seq512 that is one
127 M-element tensor — 508 MiB in fp32 — before the cross-entropy's own temporaries. Chunked
cross-entropy is the expected fix; lab 02 has to show the number first.

## Collectives: beta6 FIXES the `world>=8` blocker

The prior bring-up hit a hard wall at 8 ranks on beta3 —
`ENC:enc_init_comm ... no_hier no_mesh replica-group: [0..7]` → `Failed to execute the
device barrier 2` — after which several container collective flags were tried and all
failed. On beta6 it simply works:

```
$ LNC=2 WORLD=4 ./env/container.sh exec torchrun --nproc_per_node 4 labs/lab_collectives.py
world=4  LNC=2  barrier ok  all_reduce ok (=10)  all_gather ok  reduce_scatter ok
2D mesh (dp=2, tp=2) sub-group all_reduce ok           COLLECTIVES OK world=4

$ LNC=1 WORLD=8 ./env/container.sh exec torchrun --nproc_per_node 8 labs/lab_collectives.py
world=8  LNC=1  barrier ok  all_reduce ok (=36)  all_gather ok  reduce_scatter ok
2D mesh (dp=4, tp=2) sub-group all_reduce ok           COLLECTIVES OK world=8
```

So the full plan is available on this box: 8-rank FSDP at LNC=1, and a 2D `(dp, tp)` mesh at
either LNC. `reduce_scatter` and `all_gather` are checked specifically because they are what
FSDP's backward and forward respectively depend on — a barrier-only smoke test would have
passed on beta3 too.

Note `torch.distributed.reduce_scatter_tensor` is deprecated in torch 2.14 in favour of
`reduce_scatter_single`; harmless for now, worth switching before it is removed.

## FSDP needs `HOST_CC` on beta6 — and I had dropped that flag as dead

Both FSDP1 and FSDP2 fail at their first forward with:

```
NMGR:dlr_build_and_load_cc_resources  Failed to build and load collectives resources
    for module.neff, exec_id 469
RuntimeError: NRT Execution error occurred on Neuron for operation=allgather
    "NRT EXECUTION FAILED: NRT model scheduling failed, Non-specific failure"
```

`labs/run_fsdp_triage.sh` ran the three-way experiment (world=4, 8L, LoRA):

| hypothesis | HOST_CC | result |
|---|---|---|
| A — FSDP2 + host collectives | 1 | **OK**, 0.928 s/step, 2207.4 agg tok/s |
| B — FSDP1, plain c10d collectives | 0 | **FAILED**, same `collectives resources` error |
| C — FSDP1 + host collectives | 1 | **OK**, 1.363 s/step, 1502.5 agg tok/s |

So it is not an FSDP2/DTensor bug and not a missing collective: bare eager `all_gather` and
`reduce_scatter` pass at world 4 *and* 8. FSDP's collectives simply cannot be carried inside
the NEFF on this build; they have to go through the host path.

Worth recording honestly: I had *removed* `TORCH_NEURONX_ENABLE_HOST_CC` from `run_env.sh`
as an inherited beta3 flag, on the reasoning that it hard-errors without
`TORCH_NEURONX_ENABLE_STREAMS=1` and should not be enabled blind. Deferring it was right —
it does need the pair — but it is load-bearing, not dead. It is now opt-in via `HOST_CC=1`
and set automatically for FSDP strategies in the lab runners. DDP does not need it and does
not get it, so the DDP number is not carrying extra machinery.

## Parallelism regime: DDP wins throughput for LoRA, FSDP2 wins memory

Full 32 layers, `--gdn torch`, world=4, seq 512, bs 1/rank, LNC=2:

| strategy | mode | warm s/step | agg tok/s | peak HBM/rank | loss (6 steps) |
|---|---|---|---|---|---|
| DDP | LoRA | 4.018 | **509.7** | 15.67 GiB | 0.503 → 0.411 |
| FSDP2 | LoRA | 4.856 | 421.8 | **12.04 GiB** | 0.503 → 0.416 |
| FSDP2 | full-FT | 5.786 | 353.9 | 17.75 GiB | 0.503 → 4.152 (diverged, see below) |

For LoRA: DDP is **1.21x** faster, FSDP2 uses **23% less memory** (3.6 GiB/rank). That is a
genuine trade, so the rule is "DDP unless memory is binding", not "DDP always".

Per-rank this is ~127 tok/s against the earlier bring-up's 83 tok/s single-core full-32L LoRA
on a trn2.48xlarge — but that was the NKI path on beta3 and ours is the torch reference on
beta6, so the comparison is only indicative.

### Correcting two things the 8-layer runs appeared to show

Earlier 8-layer numbers (and the 1.39x DDP margin, and "FSDP1 is 1.5x slower") were measured
on the **NKI path, which is numerically broken past 6 GDN layers**. They stand as throughput
measurements and are worthless as anything else. Two specific retractions:

1. **"FSDP2 saves no memory for LoRA."** At 8 layers peak was identical (5.74 vs 5.71 GiB), and
   I read that as "sharding a frozen base buys nothing". At 32 layers it is 15.67 vs 12.04 GiB.
   The 8-layer model's weights are small enough that activations dominate; the full model's
   7.83 GiB of weights do not, and FSDP shards them. The truncated model was hiding the effect
   the lab exists to show.
2. **The loss values (12.09 → 8.82 etc.) were on a corrupted model** and are not convergence
   evidence. On the correct path a fresh 32-layer model starts at **0.503**, not 12 — it is a
   well-pretrained LM and SQL text is predictable. A "loss of 12" should have been a red flag
   for a pretrained 4B model rather than something I tabulated.

### Full fine-tune at lr 1e-4 diverges

0.503 → 4.152 over 6 steps: loss going UP eightfold is divergence, not noise. 1e-4 is a LoRA
learning rate — the adapter starts at zero and has to move — and is far too hot for updating
all 4.206 B pretrained weights directly. Re-running at 1e-5.

This is also why the run gate now compares the mean of the first half of steps against the
second half instead of `losses[-1] < losses[0]`: at bs=1 a single batch moves the loss more
than a few steps of training do, so first-vs-last both false-alarms and false-passes. NaN
remains a hard per-step failure.

## Full-FT on one core OOMs exactly where predicted — and the OOM does not fail fast

Full 32L, all 4.206 B params trainable, seq 512, bs 1, single logical core at LNC=2:

```
total_hbm          = 25769803776   (24.00 GiB -- one logical core)
reserved_bytes.peak= 24121442304   (22.46 GiB)
allocated_bytes.peak=22622125824   (21.07 GiB)
requested_size     =    47185920   (0.04 GiB  <- the allocation that failed)
num_alloc_retries=73 num_ooms=73 largest_cached_free=19922944
TDRV: LNC size is 2. Neuron Cores using this HBM: NC 4 and NC 5
```

It filled a 24 GiB core and died asking for 45 MB more. `sqlft/memory.py` predicts 47 GiB of
state for this configuration (7.83 weights + 7.83 grads + 31.34 fp32 AdamW), so the ceiling
is hit well before activations matter. Also confirms the LNC=2 pairing: one logical core =
NC 4 + NC 5 sharing 24 GiB.

**Operational hazard.** The allocator does NOT fail fast. It retries with progressively
smaller requests (48 MB → 2 MB → …) and the driver writes **three** dump files per failure
(`neuron_mem_table_*`, `nrt_mem_log_*`, `neuron_mem_log_*`) into `/tmp`, which is our cache
dir on `/mnt/nvme`. One unattended run produced **669 dump files / ~400 MB** in a few minutes
while making no progress, and kept the device held. `run_fullft_capability.sh` now caps the
OOM run with `timeout --signal=KILL` and sweeps the dumps; `container.sh clean-cache` sweeps
them too.

Second gotcha found doing that: `timeout` kills the **docker CLI**, not the container it
started. The container keeps running and keeps the device, and the next run fails with a
confusing allocation error instead of an obvious "device busy". Always `docker kill` the
container by name after a timeout — the runner does this now.

Also note the severity labelling is misleading: `TDRV:log_dev_mem_usage_table` and
`dml_dump` / `dml_driver_dump` log at **ERROR** level. They are the memory *report*, not
extra failures.

## Baseline

Untuned Qwen3.5-4B, full 32L, **`--gdn torch`**, cache-free greedy decode, 32 eval rows,
seq 1024, bs 8, 64 new tokens: **parse_rate 0.625**, bind_rate 0.250 of 28 bindable.
Stages: 7 ok, 9 bind_error, 12 parse_error, 4 schema_unusable.
Gold on the same rows: parse_rate 1.000, bind_rate 0.731 of 26.

**Retracted: the first baseline said 0.062.** It was measured with `nki_gdn=True`, so it
described the corrupted model, not Qwen3.5-4B — it produced things like
`-- 'recital!!!!!!!!!!!!!!!!`. Ten times too low, and it would have made any fine-tune look
far better than it is. `labs/lab01_baseline_eval.py` now defaults to `--gdn torch`.
General lesson: baseline on the exact path you will train on, and be suspicious of a
baseline that looks conveniently terrible.

The real failure mode is **not stopping**. The model already writes correct SQL —
`SELECT * FROM read_parquet('input.parquet');` is verbatim the gold answer, and the ENUM
example is right too — then continues with `\n\n-- Question: ...`. So a large share of the
0.625 → 1.000 gap is teaching EOS after the statement, which is precisely what the
completion-only loss mask trains. Two consequences worth stating before quoting any
improvement: the win should be cheap, and `parse_rate` is partly measuring **formatting
discipline** rather than SQL competence.

Cost: ~45 min wall for 32 prompts. The decode is O(n^2) (no KV cache) and the torch GDN
reference is much slower than the kernel: 4 batches x up to 64 tokens = up to 256 full
32-layer forwards. `greedy_generate` now prints progress — a silent 40-minute run is
indistinguishable from a hang, which cost a diagnostic detour.

### Cache-free greedy decode, and why

`eval_sql.greedy_generate` re-runs the whole prefix for every new token instead of using a
KV cache. The trainable NKI GDN kernel takes a chunk of >=128 tokens and has no single-token
decode path, which is what `use_cache=True` needs. Re-forwarding is O(n^2) but runs the exact
graph we trained, so eval cannot silently diverge from training. The kernel repo ships
separate inference kernels (`deltanet_tkg_batched`, `deltanet_recurrent_fwd_state`) — wiring
those would be a speed optimisation, not a correctness one, and is not on the critical path.

## The fine-tune: it works, and the metric hits its ceiling

One full epoch, full 32L, LoRA r=16, patched NKI kernel, DDP world=4, effective batch 16
sequences / 8192 tokens, lr 1e-4 warmup+cosine → 1e-5:

```
loss 1.1541 -> 0.2662   1299.8 agg tok/s   8.99M tokens   116.6 min   peak 14.67 GiB/rank
```

Before/after on the **same 128 held-out rows** (`runs/lab01_baseline_128.json`,
`runs/lab07_eval_ft_128.json`):

| model | parse_rate | bind_rate | fully correct | parse errors |
|---|---:|---:|---:|---:|
| gold (ceiling) | 0.984 | — | — | 2 |
| untuned | 0.570 | 0.252 (of 111) | 28 | 55 |
| fine-tuned | **0.906** | **0.598** (of 102) | **61** | **12** |

**81% of the achievable gap closed** — (0.906 − 0.570) / (0.984 − 0.570) — in one epoch.

**A methodology mistake I nearly shipped.** The first eval used 32 rows and scored a perfect
`parse_rate = 1.000`, exactly matching gold's ceiling on those rows, with `bind_rate` landing on
gold's 0.731 too. It was small-sample luck: 128 rows gives 0.906 against a 0.984 ceiling. Worse,
I briefly had a 32-row baseline next to a 128-row after-number, which is not a comparison at all
— hence the baseline re-run at 128. Two lessons: a metric that lands exactly on its ceiling
deserves suspicion rather than a headline, and before/after must share the row set.

Two things worth separating in the result:

1. **A real dialect gain.** Asked for milliseconds since epoch, the untuned model wrote
   `SELECT EXTRACT(EPOCH FROM timestamp) * 1000` — valid generic SQL, wrong for the task; the
   tuned model writes `SELECT epoch_ms(timestamp '...')`, DuckDB's actual function. That is
   learned dialect, not formatting.
2. **A large formatting gain.** The untuned model frequently emitted *correct* SQL and then
   kept going (`...;\n\n-- Question: ...`), which fails the parser. Teaching EOS is exactly
   what the completion-only loss mask does, and the eval logs quantify it: across all 128
   baseline rows the untuned model hit EOS **zero** times (`done=0/8` in every batch, so no
   batch ever exited early), while the tuned model finished most batches by token ~28. So
   `parse_rate` is partly a stopping metric — say so when quoting it.

`n_bindable` differs between the two runs (111 untuned vs 102 tuned) and that is not a bug:
`schema_unusable` is only reachable once the SQL parses, so the baseline's 55 parse failures
masked rows whose schemas were unusable anyway. It does mean the two `bind_rate` figures have
slightly different denominators; the fully-correct count (28 → 61) is the cleaner comparison.

### Gradient accumulation with DDP needs `no_sync()`

1299.8 agg tok/s here against lab04's 987.2 for the same DDP LoRA config. The difference is
`model.no_sync()` around every non-final micro-batch: without it DDP all-reduces the adapter
gradients on *each* of the 4 backward passes instead of once per optimizer step, paying 4x the
communication for one step's worth of progress.

### Eval decode: 1.6x end-to-end, and batching was making it WORSE

Eval had become the bottleneck — ~50 min for 128 rows against ~2 min of equivalent training
compute. Three changes, in `sqlft/eval_sql.py::greedy_generate` (the original is kept as
`greedy_generate_padded`, reference only):

1. **Bucketing.** The old loop always ran a full `seq_len=1024`-wide forward even when content
   was ~350 tokens. Now the tensor is the smallest multiple of 128 that holds the content
   (128 must stay the unit — the GDN kernel requires it). All 8 buckets get exercised across a
   128-row eval, so 8 one-time compiles.
2. **`logits_to_keep=1`.** Content is RIGHT-aligned so every row's newest token lands on the
   final column, which is what makes `logits_to_keep` usable at all — it slices
   `hidden_states[:, -n:, :]`, and our frontier previously sat mid-tensor. At bs 8 × seq 1024 ×
   vocab 248,320 the fp32 logits are **8.1 GB**; now ~8 MB. This is the `sqlft/memory.py`
   hotspot showing up in practice.
3. **Mask-aware `position_ids`.** The model defaults to `position_ids = arange(seq_len)`, which
   is *not* mask-aware, so left padding shifts each row's real tokens. Latent bug in the old
   path; `position_ids_from_mask` fixes it and `lab08` shows the fast path is batch-invariant.

**Batching is counterproductive here**, which was the surprise. Per prompt, warm, 32L, 24 new
tokens:

| | per prompt |
|---|---:|
| **fast bs=1** | **11.0 s** |
| fast bs=8 | 18.5 s |
| padded bs=1 | 13.2 s |
| padded bs=8 | 41.1 s |

Cause: the GDN kernel's outer loop is `for sample_idx in nl.sequential_range(BH)` — it
serialises over batch × heads, so the 24 linear-attention layers get **no** parallelism from
batching, while activations, masks and logits all grow with it. `batch_size` now defaults to 1
in `evaluate_model` and both eval labs. I had been running bs=8 on the assumption batching
helps; on this kernel it costs 1.7x.

End to end on the same 128 rows: **~50 min → 31.4 min** (1.6x; ~2x in steady state, 23.4 →
12 s/prompt, the rest being the 8 bucket compiles).

**The metric moved by one query: 0.906 → 0.898** (ok 61 → 60, parse_error 12 → 13). Expected
and worth stating: decoded output is *not* bit-identical between the layouts, because greedy
argmax over bf16 logits is chaotic — a near-tie flips and the rest of the query diverges.
Measured, 1 of 8 prompts diverged that way, into an equally valid query
(`w.Attendees FROM wedding w` vs `Attendees FROM wedding`).

### Testing a generation rewrite: compare logits, not strings

The first version of `lab08` gated on decoded-string equality and "failed" on 1/8 prompts. That
test was wrong, not the code: greedy decode amplifies any float difference into a completely
different output. The invariant that actually matters is tested with **teacher forcing** — feed
both layouts the identical token sequence and compare next-token logits:

```
step 0..7   cos 0.99986-0.99993   max|d| ~0.16   top-1 agree 8/8   top-2 margin 0.94-9.1
```

Top-1 agrees everywhere, with margins far above the noise. A second wrong turn: I first gated
on `cos > 0.9999`, which fails — for bf16 logits of magnitude ~10 over a 248,320-wide vocab,
`max|d| ~0.16` *is* the noise floor. The gate is now top-1 agreement (what greedy decode
actually consumes) with `cos > 0.999` as a sanity band.

Also measured and worth noting: the old path showed **no** batch dependence on this sample
(criterion C, 0/8), despite passing non-mask-aware position ids. Plausibly because Qwen3.5 uses
partial RoPE (25% of head_dim) and only 8 of 32 layers are attention at all — the other 24 are
linear-attention with no positional encoding. So that bug was real but latent, not biting yet.

## Open / not yet measured

- **The actual SQL fine-tune.** Everything is in place: correct path, validated metric, honest
  baseline (0.625), DDP at 509.7 agg tok/s. One epoch is 23.7k rows; at ~4 s/step with
  effective batch 4 that is ~6.6 h, so gradient accumulation and a longer run are next.
- **Tensor parallelism has not been run.** `apply_tensor_parallel` is written and the 2D
  `(dp, tp)` mesh passes the collectives gate, but no model has trained through it.
- **Pipeline parallelism not started.** `torch.distributed.pipelining` on the neuron backend
  is still unexplored.
- **Activation checkpointing and gradient accumulation are coded and self-tested but not
  benchmarked.** The interesting number is whether activation checkpointing buys seq 1024 at
  bs > 1, since seq 1024 is what keeps 99.8% of the dataset.
- **The 248,320-vocab logits hotspot is still only arithmetic**, not a measurement. Chunked
  cross-entropy is the expected fix.
- **`torch.compile(backend="neuron")` on the full training step** — everything so far is eager.
- **Report the kernel fix upstream.** `sqlft/kernel_patch/apply_blockdiag_fix.py` plus
  `runs/gdn_repro_call5.pt` is a complete bug report. The right upstream change is to apply the
  block factorization in the kernel itself rather than carrying a patch here. Also worth
  flagging: the published validation is "cos_sim = 1.000000 across 22+ configurations", which is
  true, was never run inside a model, and never at H=32.
- **`dq` parity is 0.9977 on real tensors** against ≈1.0002 for the other gradients. Above any
  reasonable threshold and far from upstream's garbage, but it is the one residual worth a look —
  `dq` is the gradient that flows most directly through the reconstructed `N`.
- **Re-test the 16-wide block choice.** The fix uses `INV_BLOCK = 16` because that is the width
  the 27B/35B path validated. 32 would halve the merge count if it is accurate enough here; the
  table says C=32 doubling error is 2.31e-05, so probably yes, and it is a one-constant change.
- ~~**Eval could go much faster still.**~~ DONE (see top of this file): recurrent-cache
  decode via the kernel repo's inference kernels, 31.4 → 10.4 min end to end, logit-parity
  gated by lab09. Lever (b) from the old note stands: the GDN kernel's `sequential_range(BH)`
  loop is why batching does not help; parallelising over heads would make batched eval
  worthwhile and would speed up training too.
- Cold compile was 331.7 s for the first 4-layer LoRA graph but only 10.9 s for a 32-layer
  torch-path graph once the NEFF cache was warm. Still, keep `main_first()` for multi-rank
  fan-out: the run-discipline skill records 8 parallel `neuronx-cc` compiles exhausting a
  124 GB host, and this box has exactly 124 GB and 12 vCPU.

## Gates

`labs/run_gates.sh` runs all of them, cheapest first, each gating on its own marker:
`lora_tests, data, eval_sql, ckpt, kernel_patch, neumann, device, gdn_parity, gdn_patched,
train_4L, train_32L_patched, eval_fast, collectives_w4`. Currently **ALL GATES PASS**.

`kernel_patch` is a gate rather than a setup step on purpose: it re-applies the source rewrite
against the pinned upstream every run, so an upstream change breaks the build loudly instead of
silently reverting the numerics fix.

## Refuted / do not re-chase

Measured here and found NOT to be the answer:

- **GDN kernel divergence is not a shape problem.** Parity holds at every `T ∈ {256, 512, 1024}`
  × `H ∈ {2, 8, 32}`.
- **Nor a log-decay magnitude problem.** `g_scale ∈ {1, 4, 16, 64, 256}` all pass.
- **Nor caused by zero rows from padding.** Real q/k/v are ~29.5% exactly zero, which looked
  like the obvious culprit; running `--packed` (0% padding, mask all ones) still NaNs at step 0.
- **The NaN is not FSDP, not multi-rank, and not full-FT.** 32L LoRA on a *single core* NaNs at
  step 1. Do not re-triage the parallelism layer for this.
- **Not an optimizer or gradient-clipping problem.** The forward loss is already NaN before
  `backward()` is called.

Inherited from the earlier bring-up, still worth not repeating: at `world>=8` on beta3,
`NEURON_RT_DBG_*_CHANNEL_BUFFER_SIZE=256MB`, `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, and
`NEURON_SKIP_EFA_AFFINITY=1 FI_PROVIDER=efa` all failed to fix the collective mesh. Moot now —
beta6 works at world 8 with no flags.
