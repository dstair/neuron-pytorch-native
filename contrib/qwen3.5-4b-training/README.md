# Qwen3.5-4B SQL fine-tune — a native-PyTorch training curriculum on Trainium2

Fine-tune `Qwen/Qwen3.5-4B` on [`motherduckdb/duckdb-text2sql-25k`](https://huggingface.co/datasets/motherduckdb/duckdb-text2sql-25k)
to emit syntactically valid DuckDB SQL, using the SQL task as a vehicle for learning
**distributed training in native PyTorch** — `torch.device("neuron")`, `loss.backward()`,
`optimizer.step()`. No XLA, no NxDI, no HF `Trainer`, no `accelerate`, no PEFT.

Everything here is written to be read: one small library (`sqlft/`), one runnable script per
rubric item (`labs/`), and a measured number for every claim.

## What this is for

| Rubric item | Where | Status |
|---|---|---|
| `DataLoader`, `collate_fn`, packing, label masking | [`sqlft/data.py`](sqlft/data.py) | done, self-tested |
| `torch.nn` + autograd — LoRA written by hand, no PEFT | [`sqlft/lora.py`](sqlft/lora.py), [`tests/test_lora.py`](sqlft/tests/test_lora.py) | done, 11 tests inc. fp64 gradcheck |
| Memory accounting from first principles | [`sqlft/memory.py`](sqlft/memory.py) | budget done; logits/CE hotspot not yet measured |
| The OOM cliff, measured | [`labs/run_fullft_capability.sh`](labs/run_fullft_capability.sh) | done — single-core full-FT OOMs |
| Activation ckpt vs grad accumulation vs state ckpt | [`sqlft/ckpt.py`](sqlft/ckpt.py) | code + self-test done; not yet benchmarked |
| Collectives on the neuron backend | [`labs/lab_collectives.py`](labs/lab_collectives.py) | done, green at world 4 and 8 |
| Data parallel (DDP) vs sharded (FSDP2 / ZeRO-3) | [`sqlft/parallel.py`](sqlft/parallel.py), [`labs/lab04_ddp_vs_fsdp.py`](labs/lab04_ddp_vs_fsdp.py) | done, measured at 32L |
| Tensor parallel (DTensor), 2D `(dp, tp)` mesh | `sqlft/parallel.py::apply_tensor_parallel`, [`labs/run_lab05_tp.sh`](labs/run_lab05_tp.sh) | done, full-FT — runs, loses to FSDP2 here (see NOTES) |
| Pipeline parallel (`torch.distributed.pipelining`) | — | **not started** (spike) |
| Baseline / after parse rate, the actual fine-tune | [`labs/lab01_baseline_eval.py`](labs/lab01_baseline_eval.py), [`labs/lab07_finetune.py`](labs/lab07_finetune.py), [`sqlft/eval_sql.py`](sqlft/eval_sql.py) | full-epoch run done |
| Distributed sampling, LR schedule, adapter save/load | `sqlft/data.py`, `sqlft/lora.py`, `labs/lab07_finetune.py` | done |

`sqlft/gdn.py` is the one Trainium-specific piece: it swaps Qwen3.5's GatedDeltaNet recurrence
for a [trainable NKI kernel](https://huggingface.co/kernels/jburtoft/qwen35-gdn-deltanet-neuron-kernels).
**v3.0 of that kernel is numerically wrong on this model**, and v15 is close but still 6–7% off
on two gradients at one layer — see "The GDN kernel bug, and the fix" under Results. The
default, `--gdn nki-v15-split16`, is v15 with a 16-wide diagonal solve and matches the
reference everywhere; `labs/lab02_gdn_in_situ.py` and `labs/lab03_patched_kernel_parity.py`
are the evidence.

## The model, and why it is awkward

Qwen3.5-4B is a **hybrid**: of its 32 decoder layers, 24 are GatedDeltaNet linear attention
and 8 are full GQA attention (every 4th layer, indices 3/7/11/…). Measured from the config:

| | |
|---|---|
| params | 4.206 B (0.636 B of that is the embedding, tied with the LM head) |
| hidden / intermediate | 2560 / 9216 |
| vocab | **248,320** — big enough that the logits tensor is its own memory problem |
| GDN heads | 32 v-heads / 16 k-heads, `head_dim=128` |
| GQA heads | 16 Q / 4 KV, `head_dim=256` |

Two consequences that shape every lab:

**Full fine-tuning does not fit on one core.** 4.206 B params in bf16 is 7.83 GiB of
weights, 7.83 GiB of gradients, and 31.34 GiB of fp32 AdamW moments — **47 GiB** before a
single activation, against **24 GiB per logical core**. FSDP is not a speedup here, it is
the only way full-FT runs at all. (`python3 -m sqlft.model --report`)

**Truncating the stack changes which layers exist.** `--layers 3` contains no
full-attention layer, so no `q_proj` to attach LoRA to. `apply_lora` intersects its target
list with the modules actually present rather than trusting a fixed list.

## Hardware

One `trn2.3xlarge`: 1 Trainium2 chip, 96 GB HBM, 12 vCPU, 124 GB host RAM.

| LNC | logical cores | HBM/core | use |
|---|---|---|---|
| `LNC=2` (default) | 4 | 24 GB | correctness, LoRA, full-FT under FSDP — the proven regime |
| `LNC=1` | 8 | 12 GB | scaling labs, where rank *count* is the point |

`/mnt/nvme` is **instance store**. A reboot preserves it; a stop/start destroys the weights,
the compile caches and every artifact. `env/setup_box.sh` is the recovery path — keep it working.

## Setup

```bash
./env/remote.sh status          # state, IP (rediscovered from the instance ID), neuron-ls, disk
./env/remote.sh push            # rsync this package to /mnt/nvme/qwen35-training/src
./env/remote.sh ssh /mnt/nvme/qwen35-training/src/env/setup_box.sh all
```

`setup_box.sh all` builds the derived container image (beta6 DLC + transformers / datasets /
duckdb / kernels), downloads the 8.8 GB checkpoint, the dataset, and the pinned GDN kernel,
then asserts everything imports.

## Running

Everything runs inside the container via `env/container.sh`, which supplies the device, the
work volume and the clean environment:

```bash
R=/mnt/nvme/qwen35-training/src

# All gates, cheapest first. Run this after changing anything in sqlft/.
./env/remote.sh ssh "$R/labs/run_gates.sh"
#   lora_tests  data  eval_sql  ckpt  device  gdn_parity  train_4L  collectives_w4
#   -> ALL GATES PASS

# train
./env/remote.sh ssh "$R/env/container.sh exec python3 -m sqlft.train \
    --layers 32 --seq 512 --bs 1 --steps 6 --lora --tag my_run"

# the parallelism matrix (world=4, full 32 layers)
./env/remote.sh ssh "$R/labs/run_lab04_matrix.sh 4 32 512 6"

./env/remote.sh pull            # bring runs/*.json back down
```

Each run writes `runs/<tag>.json`; the results table below is built from those files.

### Run discipline

These are not optional — each cost real hours to learn (see
[`skills/trainium-run-discipline`](../../skills/trainium-run-discipline)):

- **Gate on the success marker** (`TRAIN OK`, `SMOKE OK`, `GDN PARITY OK`), never on the
  exit code. The beta containers SIGABRT during teardown *after* printing valid results.
- **`env/container.sh clean-cache` before any number you will quote.** The cache is
  root-owned; a non-sudo `rm -rf` fails silently and serves you the previous NEFF.
- **No debug flags in a timed run.** `run_env.sh` is the clean environment and contains no
  switch to make it dirty.
- **Never `stop-instances`.** Reboot only, and ask first.

## Results

Measured on the box above, bf16, eager (no `torch.compile`), `LNC=2`, single core unless
stated. Cold-compile time is listed separately from warm step time because the first step
is 100–1000× the rest.

All at the **full 32 layers**, seq 512, bs 1/rank, `LNC=2`, gated on their own `LAB04 OK` /
`TRAIN OK` marker. Both GDN paths shown: `nki-patched` is the NKI kernel with our
block-factorized inverse fix, `torch` is HF's reference.

| config | world | `torch` agg tok/s | **`nki-patched`** agg tok/s | speedup | peak HBM/rank |
|---|---|---:|---:|---:|---:|
| LoRA, **DDP** | 4 | 509.7 | **987.2** | **1.94×** | 15.67 GiB |
| LoRA, FSDP2 | 4 | 421.8 | **725.7** | 1.72× | **12.04 GiB** |
| full-FT, FSDP2 | 4 | 353.9 | **577.1** | 1.63× | 17.75 GiB |
| full-FT, single core | 1 | — | — | — | **OOM at 24 GiB** |

Warm s/step on the patched path: 2.075 (DDP LoRA), 2.822 (FSDP2 LoRA), 3.549 (FSDP2 full-FT).
Cold compile 75–83 s. Sources: `runs/lab04_*_w4.json`, `runs/lab04_fullft_1core.log`.

### The full fine-tune run

`labs/lab07_finetune.py`, one full epoch over the 23,383 filtered train rows:

| | |
|---|---|
| config | full 32L, LoRA r=16, `--gdn nki-patched`, DDP world=4, `LNC=2` |
| batching | bs 1/rank × grad-accum 4 × 4 ranks = **16 sequences / 8192 tokens per step** |
| schedule | 1097 steps, lr 1e-4, 20-step warmup, cosine → 1e-5 |
| **loss** | **1.1541 → 0.2662** |
| throughput | **1299.8 agg tok/s**, 6.28 s/step, peak 14.67 GiB/rank |
| total | 8.99 M tokens, **116.6 min** wall |
| adapter | 130.0 MB (32.46 M trainable params, fp32) |

Source: `runs/ft_epoch1.json`. Gradient accumulation with DDP `no_sync()` on the non-final
micro-batches is why this beats the lab04 DDP number (1300 vs 987 tok/s) — the all-reduce is
paid once per optimizer step instead of once per micro-batch.

Loss behaviour (measured on the reference path, and reproduced to ~0.5% on the patched one):
a fresh 32-layer model starts at **≈0.50** on this data. Full-FT at lr 1e-4 **diverges**
(0.503 → 4.152 in 6 steps); at lr 1e-5 it is stable. LoRA at lr 1e-4 is fine.

**Two regimes, two right answers.** For LoRA, DDP is **1.21× faster** but FSDP2 uses **23%
less memory** (12.04 vs 15.67 GiB/rank) — a real trade, so the rule is "DDP unless memory is
binding", not "DDP always". For full fine-tuning, FSDP2 is not an optimisation at all: single
core dies filling a 24 GiB core and asking for 45 MB more, exactly as `sqlft/memory.py`
predicts from 47 GiB of weights+grads+AdamW state. Sharding the optimizer moments across 4
ranks is the only reason the run exists.

**Full fine-tuning needs a much smaller learning rate than LoRA.** At lr 1e-4 — a perfectly
normal LoRA rate, since the adapter starts at zero and has to move — the loss goes *up*
eightfold (0.503 → 4.152). That is divergence, not noise. 1e-5 is stable.

A fresh 32-layer model starts at **loss ≈ 0.50** on this data, not ~12: it is a well-pretrained
LM and SQL text is highly predictable. Any "loss of 12" here means the model is broken (a
truncated stack, or the NKI kernel).

### Eval decode speed, and why `batch_size=1` is fastest

Eval was the bottleneck (~50 min for 128 rows against ~2 min of equivalent training compute).
`sqlft/eval_sql.py::greedy_generate` now (a) sizes the tensor to the smallest 128-multiple
bucket that holds the content instead of always 1024 wide, (b) right-aligns content so the
newest token is on the final column, which makes `logits_to_keep=1` valid — dropping the fp32
logits from **8.1 GB** to ~8 MB at bs 8 × seq 1024 × 248,320 vocab — and (c) passes mask-aware
`position_ids`, which the model does not derive itself.

**Batching makes this slower**, per prompt, warm, 32L, 24 new tokens:

| | per prompt |
|---|---:|
| **fast bs=1** | **11.0 s** |
| fast bs=8 | 18.5 s |
| padded bs=1 | 13.2 s |
| padded bs=8 | 41.1 s |

The GDN kernel loops `for sample_idx in nl.sequential_range(BH)` — it serialises over
batch × heads, so the 24 linear-attention layers gain nothing from batching while everything
else grows. `batch_size` defaults to 1 now. End to end: **~50 → 31.4 min**.

Verified by `labs/lab08_eval_speed.py`, which compares **logits under teacher forcing** rather
than decoded strings (greedy decode is chaotic — a bf16 near-tie flips and the rest of the query
diverges): top-1 agrees at every step, top-2 margins 0.94–9.1, and the fast path is
batch-invariant where the old one was not guaranteed to be.

**Recurrent-cache decode** (`sqlft/infer.py`) then removes the O(n²) entirely: prefill once
through the kernel repo's inference kernels (nkilib `gdn_cte`, recurrent state out), then
single-token steps (`deltanet_tkg_batched_bh`) for the 24 GDN layers, with a static
preallocated KV buffer for the 8 attention layers so every decode step is the same compiled
graph. Validated by `labs/lab09_cached_decode_parity.py` under the same teacher-forcing
standard (top-1 agreement 88/88 steps, worst cos 0.99881 across kernels). End to end the
128-row eval went **~50 → 31.4 → 10.4 min**, with parse_rate 0.906 → 0.898 → 0.883 — the
last two moves are 1 and 2 queries of 128, inside greedy-chaos/binomial noise at n=128.
`lab07_eval_finetuned.py --decode cached` selects it.

### The LoRA sweep: what buys accuracy, and what doesn't

Eight 1-epoch configs (`labs/run_sweep.sh`), each graded on motherduck parse/bind AND
BIRD exec accuracy. Graphs in [`graphs/`](graphs) (`plot_sweep.py` regenerates them).

| config | params | epoch wall | tok/s | loss | parse | bind | exec |
|---|---:|---:|---:|---:|---:|---:|---:|
| **r=4 α=8** | **8.1 M** | 81.5 min | 1878 | 0.231 | 0.961 | 0.697 | 0.180 |
| r=8 α=16 | 16.2 M | 81.8 | 1871 | 0.202 | 0.961 | 0.674 | 0.156 |
| r=16 α=32 (ft_epoch1, old stack) | 32.5 M | 116.6* | 1300 | 0.266 | 0.969 | 0.708 | 0.219 |
| r=32 α=64 | 64.9 M | 81.0 | 1890 | 0.163 | **1.000** | 0.663 | 0.156 |
| r=64 α=128 | 129.8 M | 82.4 | 1859 | 0.158 | 0.992 | 0.663 | 0.156 |
| lr 3e-5 (r=16) | 32.5 M | 80.3 | 1896 | 0.293 | 0.984 | 0.674 | 0.156 |
| lr 3e-4 (r=16) | 32.5 M | 79.2 | 1903 | 0.162 | 0.984 | 0.629 | 0.164 |
| mixers r=16 (no MLP) | **14.4 M** | **72.9** | **2084** | 0.233 | 0.969 | 0.640 | 0.188 |

*ft_epoch1 trained on the old box (beta6 container); the same recipe is ~81 min on this
box's public stack, so wall times are not comparable across that row.

**The most efficient config is r=4 α=8, and the runner-up is mixers r=16.** The findings,
qualified by n=128 CIs (±5pt parse, ±7pt exec):

- **Wall time barely moves across configs (73–82 min, ±13%).** The adapter is a tiny
  share of the 4.2 B forward/backward, so "training time vs accuracy" is NOT the binding
  tradeoff at a fixed epoch budget — the axes that matter are params and metric choice.
- **Rank doesn't buy accuracy here.** Parse/bind are flat r=4 → r=64; r=32 even hits
  parse 1.000 with bind *unchanged* (0.663) — perfect syntax, same semantics. Capacity
  past r=4 is wasted on this data scale (and would matter only with more/longer training).
- **lr 1e-4 is the sweet spot.** 3e-4 doesn't diverge (LoRA's tolerance, contrasted with
  full-FT diverging at 1e-4) but bind degrades to 0.629; 3e-5 under-converges (loss curve
  sits above the pack).
- **The MLP adapters earn little.** mixers drops 56% of the parameters, runs the fastest
  epoch (72.9 min, 2084 tok/s), and lands within noise of all-targets on parse
  (0.969) and exec (0.188).
- **More steps ≠ monotonic gains.** r=8 checkpoints: 0.969 @400 → 0.984 @800 → 0.961
  @1097 — inside the ±4pt noise band; single-epoch LoRA saturates early.
- **The motherduck metric has hit its ceiling as a discriminator** (r=32: 1.000). BIRD
  exec accuracy is now the only metric with headroom to rank configs — and all config
  differences there are inside its CI. Ranking configs properly means a bigger exec
  sample, not a better motherduck number.

### The GDN kernel bug, and the fix

Upstream `deltanet_fused_chunked_fwd_batched` passes synthetic parity **exactly** (cos =
1.000000 fwd and bwd, every shape to T=1024/H=32, every log-decay magnitude) and is still
unusable: on the model's own activations it diverges at the 6th GatedDeltaNet layer (\|max\|
7.1e+3 vs the reference's 5.8e-2, cos **−0.0006**) and compounds to inf by the 17th.

**Root cause.** It builds the intra-chunk triangular inverse by full-width Neumann
power-doubling at C=128. That is exact in exact arithmetic — 6 rounds do reach A^127, so it is
not a truncation bug — but chaining 6 squarings in fp32 destroys the cancellation the series
relies on. This repo had already hit and fixed the same failure in the **27B/35B prefill**
kernels, which is why those ship **C=32** with `_tri_inverse_blockdiag`
(`contrib/qwen3.6-35b-a3b/`, `VLLM_NEURON_ASSESSMENT.md` §7).

Measured on real Qwen3.5-4B chunk matrices, max-abs error of `(I-A)^-1` vs a float64 oracle
(`labs/lab03_neumann_stability.py`):

| C | full-width doubling | block-16 factorization |
|---:|---:|---:|
| 16 | 7.15e-07 | 7.15e-07 |
| 32 | 2.31e-05 | 7.82e-07 |
| 64 | 7.89e-03 | 1.16e-06 |
| **128** ← the kernel | **1.08e+03** | **1.29e-06** |

**The fix** — `sqlft/kernel_patch/apply_blockdiag_fix.py` generalises `_tri_inverse_blockdiag`
(2-way, C=32 only) to a recursive 2-way split down to 16-wide diagonal blocks, so every squaring
stays 16-wide. Applied as an anchored source rewrite into a sibling kernel tree, leaving the
upstream download pristine and failing loudly if upstream changes:

```bash
./env/container.sh exec python3 -m sqlft.kernel_patch.apply_blockdiag_fix
./env/container.sh exec python3 labs/lab03_patched_kernel_parity.py --grads
```

**Verified**: on the real diverging-layer tensors, cos **−0.000591 → 1.000289**; all 24 GDN
layers match in situ; all 5 gradients match (dq 0.9977, dk/dv ≈1.0002, dg/dbeta 1.000000); and
the 32-layer loss trajectory tracks the reference to ~0.5% while running **1.6–1.9× faster**
than it. `--gdn nki` is kept only to reproduce the failure.

**Upstream v15, and why it is patched too.** Upstream fixed the same failure in v8 with row-block
forward substitution at `BLOCK=32`, and v14 sped up the backward. On Qwen3.5-4B that is still
not enough: at GDN layer 19 (`runs/gdn_repro_call19.pt`) v15's forward is cos 0.9979 and its
`dv` / `dbeta` gradients are **0.925 / 0.943**. Narrowing to `BLOCK=16` does not compile (a
16-partition access must start at partition 0/32/64/96), so
`sqlft/kernel_patch/apply_v15_split16.py` keeps 32-row blocks and inverts each 32×32 diagonal
block by a mask-based 2-way split — doubling only the two 16-wide halves, then the exact merge
`D⁻¹ + D⁻¹XD⁻¹`. Layer 19, vs the reference:

| build | fwd | dq | dk | dv | dg | dbeta |
|---|---:|---:|---:|---:|---:|---:|
| v3.0 | −0.077 | 0.039 | −0.004 | 0.034 | 0.031 | −0.079 |
| v3.0 + blockdiag (`nki-patched`) | 1.000047 | 1.000049 | 1.000091 | 1.000058 | 1.000000 | 1.000000 |
| v15 | 0.997881 | 0.998370 | 0.992111 | **0.925385** | 0.986994 | **0.943209** |
| **v15 + split16** (default) | **1.000047** | **1.000049** | **1.000091** | **1.000058** | **1.000000** | **1.000000** |

All 24 GDN layers match in situ (worst cos 1.000019), and it is the fastest build: 32L LoRA
single core **1.148 s/step** against 1.174 for `nki-patched` and 1.171 for plain v15. Use
`--gdn nki-v15-split16`; it is the default everywhere.

Reference points from the earlier bring-up of this model on a `trn2.48xlarge` (different box,
beta3 container, NKI path — indicative only): full-32L LoRA single core 6.16 s/step / 83 tok/s;
full-FT FSDP=4 26.83 s/step; full-FT single core OOM.

### FSDP on beta6 requires the host-collectives flag pair

Both FSDP1 and FSDP2 fail without it, with an error that looks like a hardware problem:

```
NMGR:dlr_build_and_load_cc_resources  Failed to build and load collectives resources
    for module.neff, exec_id 469
RuntimeError: NRT Execution error occurred on Neuron for operation=allgather
    "NRT EXECUTION FAILED: NRT model scheduling failed, Non-specific failure"
```

Bare eager `all_gather` / `reduce_scatter` pass at world 4 and 8, so this is not a collective
support gap — it is that FSDP's collectives cannot be carried inside the NEFF. Setting
`HOST_CC=1` (which `run_env.sh` expands to `TORCH_NEURONX_ENABLE_HOST_CC=1` +
`TORCH_NEURONX_ENABLE_STREAMS=1`, required as a pair) fixes both. `labs/run_fsdp_triage.sh`
is the three-run experiment that isolated it; DDP does not need the flag and runs without it.

### SQL metric

`parse_rate` — fraction of generated statements DuckDB's parser accepts — is the headline,
because "syntactically correct SQL" is literally what it measures. `bind_rate` additionally
requires every referenced table and column to exist in the row's schema.

Calibrations that the harness self-test enforces before it will report anything:

- **The ceiling is not 1.0.** 2.78% of the dataset's own gold SQL does not parse in DuckDB
  1.5.5 — malformed `COPY` options, unbalanced parens, and one row that is Python. Gold
  scores `parse_rate = 0.984` on the 128-row eval set. Train rows with unparseable gold are
  dropped (`filter_parseable`); eval rows keep their noise so the ceiling stays honest.
- **~30% of eval rows can never bind** — 29 whose schemas do not create (foreign keys
  referencing tables that were never emitted) plus 10 that ship no schema at all. They are
  quarantined out of the `bind_rate` denominator, decided **from the schema alone**, so the
  denominator (89 of these 128 rows) is identical for every model and every run.
- **Truncation is not a syntax error.** A generation that hits the decode cap without EOS
  and then fails to parse is reported as `truncated`, its own stage; `n_gold_over_cap`
  reports how many gold queries exceed the cap and are unwinnable by construction.
- **Every rate carries a Wilson 95% CI**, and every run stores per-row outcomes + generated
  SQL so two runs can be compared **paired** (McNemar's exact test) rather than by point
  estimates: `python3 -m sqlft.eval_sql --compare runA.json runB.json`.

Before/after on the **same 128 held-out rows**, same prompt format, same decode settings
(cached decode, max_new 128 — every gold query fits, so no row is unwinnable):

| model | parse_rate [CI] | first_stmt | bind_rate [CI] (of 89) | stage breakdown |
|---|---:|---:|---:|---|
| gold (ceiling) | **0.984** | — | — | 2 parse errors |
| Qwen3.5-4B untuned | 0.586 [0.499, 0.668] | 0.812 | 0.292 [0.208, 0.394] | ok 26, truncated 51, bind_err 30 |
| **+ 1 epoch LoRA fine-tune** | **0.969** [0.922, 0.988] | **0.969** | **0.708** [0.606, 0.792] | **ok 63, truncated 3, bind_err 25, parse_err 1** |

Paired (McNemar), before vs after: 1 vs 50 discordant on parses, **p < 0.001**; 5 vs 42 on
binds, **p < 0.001**. Sources: `runs/lab01_baseline_128_mn128.json`,
`runs/lab07_eval_ft_128_mn128.json`.

### Execution accuracy: queries that actually run

Parse and bind are still proxies. The real question — *does it return the right answer?* —
needs data, which this dataset doesn't ship. [`contrib/bird-duckdb`](../../bird-duckdb)
builds a verified DuckDB-dialect execution benchmark from BIRD dev (1,491 questions with
real databases, gold translated and machine-verified against the SQLite originals), and
`labs/lab10_bird_exec_eval.py` grades result-set equality on a stratified 128-question
subset, same prompt format and cached decode as above:

| model | exec_accuracy [CI] | parse_rate | stage breakdown |
|---|---:|---:|---|
| Qwen3.5-4B untuned | 0.070 [0.037, 0.128] | 0.719 | ok 9, wrong 47, exec_error 36, truncated 36 |
| **+ 1 epoch LoRA fine-tune** | **0.219** [0.156, 0.298] | **0.961** | **ok 28, wrong 41, exec_error 54, truncated 2** |

Paired (McNemar): 5 vs 24 discordant, **p = 0.0005**. Sources: `runs/lab10_base_128.json`,
`runs/lab10_ft_128.json`.

The honest read: **0.961 parses but only 0.219 answers correctly.** BIRD is a much harder
benchmark than the fine-tune's own distribution (multi-table joins over real databases,
external-knowledge columns), and the residual failure is dominated by *schema-fidelity*
errors — inventing column names, wrong tables, made-up functions — i.e. the thing bind_rate
approximates, made exact. That gap is the target for the next fine-tune.

**The decode cap matters more than it looks.** At max_new 48 the fine-tuned model scored
0.883 with 14 truncations; 96 scores 0.961 (p = 0.006, paired) and 128 scores 0.969 — and
at 128 `n_gold_over_cap` is 0, so the cap no longer confounds the metric at all (max gold
is 124). One subtlety found by auditing rows: gold length bounds *matching gold*, not
*answering well* — two gold > 96 rows already passed at cap 96 because the model wrote a
terser query. Residual failures: 3 degenerate `'SELECT'`-only outputs and 1 genuine
syntax error.

> Historical note: pre-v5 numbers (parse 0.570 baseline, 0.906/0.898 fine-tuned) counted
> truncations as parse errors and used bind denominators that moved with the model. They
> are not exactly comparable to the v5 table; both live in `runs/` for reference.

**parse_rate 0.586 → 0.969** against a ceiling of 0.984 — **96% of the achievable gap closed**
in one epoch. `bind_rate` 2.4×, fully-correct queries 26 → 63, and the residual failures
are 3 degenerate outputs and **one genuine parse error** in 128 generations.

> A smaller n=32 run scored a perfect 1.000 and I nearly reported that. It was small-sample
> luck; 128 rows gives ~0.9. Both numbers are in `runs/`. The 32-row pair is also *not*
> comparable to a 128-row pair, which is why the baseline was re-run at 128 — and why every
> rate now carries a Wilson interval and every comparison is paired.

The improvement is not only formatting. The model learned DuckDB dialect:

```
Q    : What is the number of milliseconds since the epoch for a given timestamp?
gold : SELECT epoch_ms(TIMESTAMP '1992-03-22 00:00:00')
before: SELECT EXTRACT(EPOCH FROM timestamp) * 1000;        <- generic SQL, wrong function
after : SELECT epoch_ms(timestamp '2022-11-07 08:43:04');   <- DuckDB's epoch_ms

Q    : Load all columns and rows from the 'input.parquet' Parquet file.
before: SELECT * FROM read_parquet('input.parquet');\n\n-- Question: Load all columns ...
after : SELECT * FROM read_parquet('input.parquet');        <- and it stops
```

The same eval on the faster decode paths (`--batch-size 1` bucketed, then `--decode cached`):
**0.898 in 31.4 min**, then **0.883 in 10.4 min** — one and two queries different out of 128.
Decoded output is not bit-identical across decode layouts because greedy argmax over bf16
logits is chaotic; the layouts are verified logit-equivalent instead (see below).

Two caveats to carry with the numbers. Part of the gain is learning to emit **EOS** —
`parse_rate` rewards stopping as much as syntax, and now that is measured rather than
asserted: the untuned model's `first_stmt_parse_rate` is 0.805 against its 0.602 full-output
rate, so ~20 points of the baseline failure is *not stopping*, not bad SQL (fine-tuned:
0.961 vs 0.961 — the fine-tune closed the ramble gap entirely). And `bind_rate` is capped
well below 1.0 by dataset artifacts — queries reading CSV files that do not exist,
duplicate column names — not by the model; ~30% of eval rows are quarantined out of its
denominator entirely.

The untuned model already writes decent SQL. Its problem is **not stopping** — it emits the
right statement and then keeps going with more prose:

```
Q    : Create a new enum type 'performancetype' with five possible values: ...
gold : CREATE TYPE performance_type AS ENUM ('concert', 'recital', 'opera', 'ballet', 'musical');
model: CREATE TYPE performancetype AS ENUM ('concert', 'recital', 'opera', 'ballet', 'musical');   <- correct

Q    : Load all columns and rows from the 'input.parquet' Parquet file.
gold : SELECT * FROM read_parquet('input.parquet');
model: SELECT * FROM read_parquet('input.parquet');\n\n-- Question: Load all columns and rows ...
```

So most of the 0.625 → 1.000 gap is teaching the model to emit EOS after the statement, which
is exactly what our completion-only loss mask trains. That makes it a cheap win, and it also
means **parse_rate is partly a formatting metric, not purely a SQL-competence one** — worth
saying out loud before quoting an improvement.

> **Retracted:** an earlier baseline of **0.062** was measured with `--gdn nki`, i.e. on the
> numerically corrupted model (see the kernel warning above). It reported gibberish like
> `-- 'recital!!!!!!!!!!!!` and would have flattered any subsequent fine-tune by ~10x. Always
> baseline on the path you will actually train on.

## Layout

```
env/     remote.sh        box access: IP discovery, ssh, push/pull
         container.sh     launch the DLC with device + volumes + clean env
         run_env.sh       the clean NEURON_* flags (and the HOST_CC opt-in)
         Dockerfile       beta6 DLC + transformers/datasets/duckdb/kernels
         setup_box.sh     image, weights, dataset, kernel, check

sqlft/   device.py        neuron device registration, sync, Timer, memory probe
         data.py          Dataset/DataLoader, masking, packing, gold-SQL filter
         lora.py          hand-written LoRALinear + apply_lora
         gdn.py           the NKI GatedDeltaNet swap (+ its parity test)
         model.py         load Qwen3.5-4B text tower, truncate, LoRA, bf16
         memory.py        byte budget from first principles
         ckpt.py          activation ckpt / grad accumulation / DCP save-resume
         parallel.py      process group, meshes, DDP/FSDP1/FSDP2/TP
         train.py         the training loop
         eval_sql.py      DuckDB parse+bind rate, cache-free greedy decode; Wilson CIs,
                          truncation stage, per-row records, McNemar run comparison
         bird.py          BIRD-for-DuckDB questions, native materialization, exec grading
         infer.py         recurrent-cache decode (inference GDN kernels + static KV cache)
         tests/           CPU-only correctness tests

labs/    lab_smoke.py             device + autograd gate
         lab_collectives.py       collectives gate at world 2/4/8 + 2D mesh
         lab01_baseline_eval.py   the "before" parse rate
         lab02_find_nan_layer.py  per-layer activation magnitudes
         lab02_gdn_in_situ.py     kernel vs reference on the model's own tensors
         lab02_gdn_repro.py       ablate the captured inputs to isolate the bug
         lab04_ddp_vs_fsdp.py     the parallelism regime comparison
         lab09_cached_decode_parity.py  cached vs re-forward decode, teacher-forced logits
         lab10_bird_exec_eval.py  execution accuracy on real BIRD databases
         run_*.sh                 serial runners for each matrix / triage

runs/    gitignored: logs, result json, captured reproducers
```

`NOTES.md` is the lab notebook — measured numbers, things that broke, and levers that were
tried and refuted, so they are not re-chased.

## Stack

Default (`STACK=public`): the public-beta TorchNeuron DLC
`public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04` (digest
`sha256:b4648437…`): torch **2.14.0**, torch-neuronx 2.14.3, neuronx-cc 2.28.7631, nki 0.7.0,
Python 3.14. Plus transformers 5.20.2, duckdb 2.0.0 from `env/Dockerfile`. Measured **1.7×**
faster per step than beta6 on the same box with identical losses (NOTES.md).

`STACK=beta6`: the pre-GA DLC (not public) behind
every number measured before 2026-10-21 — torch 2.14.0, torch-neuronx 2.14.3, neuronx-cc
2.0.404830, nki 0.7.0b1, transformers 5.17.0, duckdb 1.5.5.

`torch.device("neuron")` — no `torch_xla`, no `neuronx_distributed`.

## License

Apache-2.0. Qwen3.5-4B weights © Alibaba (Apache-2.0). The dataset is CC-BY-SA-4.0.
