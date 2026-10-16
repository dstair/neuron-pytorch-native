# Qwen3-ASR-0.6B on AWS Neuron (PyTorch-native)

Runs [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) speech-to-text on Trainium2 with
`torch.compile(backend="neuron")` — PyTorch-native (`torch_neuronx`), not NxDI / torch-xla. The task is English
transcription (language forced via `apply_transcription_request(language="English")`), greedy decoding. The
model is a static-shape re-implementation of the HF audio-encoder + Qwen3 text-decoder forward passes that
reuses the HF weights unmodified; output is token-for-token identical to HF `generate()` on CPU in fp32 for
9/10 of a 10-sample `librispeech_asr_dummy` set (the 10th diverges by one token at a documented fp32 near-tie,
see below). What is novel here (beyond "port it and compile it"): a decode-step fix for a tied-`lm_head`
weight-transpose/spill defect that cost **85x** of the step's device time, an in-graph-argmax miscompile found
and fixed, and three further rounds of profile-guided optimization that cut the decode step from 655 ms to
4.2-4.7 ms/token (**~150x**) — most of it DMA-bound and documented with NEFF/NTFF profiles, not guesses.

## Results (trn2.3xlarge, 1 of 4 logical NeuronCores, LNC=2, bf16, batch 1)

| | Value |
|---|---|
| Accuracy | 10/10 token-exact vs HF CPU fp32 `generate()` on `librispeech_asr_dummy` (first 10 samples); WER 4.33% (= CPU fp32 WER). Extended to 32 samples for the batch sweep: 30/32 token-exact, WER 4.23-4.40% across prefixes. |
| Encoder + projector, per utterance (bucketed 8/16/32 mel chunks ≈ 8-32 s audio) | 10.8-11.3 ms (C=8/16), 18.7-19.1 ms (C=32) |
| Prefill (bucketed 128/256/512 tokens) | 8.0-11.6 ms |
| Decode | **4.2-4.7 ms/token** (`step_amort`, K=4 tokens/graph-call), down from 655 ms/token in the first working compiled port (≈150x) |
| End-to-end per utterance (encoder+prefill+greedy decode, 13-88 generated tokens) | 100-137 ms median (sweep-size-dependent — see batch sweep); CPU fp32 HF `generate()` reference: 1027-1521 ms median over the same prefixes (≈10-13x) |
| Throughput | ~187-206 tokens/s, ~6.4-10.1 utterances/s, sequential on 1 NeuronCore (this port has no in-graph batching — see "Known limits") |
| Cold compile | ~459 s (7.7 min) for one (encoder, prefill, decode) bucket triple from an empty NEFF cache; graphs are cached in `$TORCH_NEURONX_NEFF_CACHE_DIR` |

[EXPERIMENTS.md](EXPERIMENTS.md) records every attempt in order: the initial port, the lm_head/argmax device
bugs and fixes, three profile→improve rounds (with NEFF/NTFF evidence), and the rejected variants with their
numbers (int8/fp8 weight quantization, in-place KV-cache mutation, split-softmax "late" KV writes, larger K
unrolls) — kept here so the next person doesn't re-spend the hours it took to rule them out.

## Why not HF `generate()`

Eager `generate()` on the Neuron device actually **works** for this model (unlike some other HF ports where
`aten::index` lowers to a built-in NKI kernel that only compiles for LNC=1): a diagnostic run (`eager_test.py`,
no longer included in this folder) matched the CPU fp32 reference token-for-token. But it compiles every op
individually (176.6 s for sample 0's first call) and the warm call is still 3.65 s — slower than the entire
compiled pipeline below including its own compiles. `run.py` instead builds three fixed-shape graphs with
`torch.compile(backend="neuron", fullgraph=True, dynamic=False)`:

| Graph | Shapes | When |
|---|---|---|
| `Encoder` | mel features `[1,128,C*100]` + valid-frame count → projected audio embeds `[C*13, 1024]`, bucketed `C ∈ {8,16,32}` | once per utterance |
| `Prefill` | token ids `[1,P]` (right-padded) + audio embeds + last-token index → first generated token, static K/V caches `[28,8,768,128]`, bucketed `P ∈ {128,256,512}` | once per utterance |
| `DecodeK` | token, position, K/V caches → **K=4** next tokens (fed back in-graph via embedding gather) + updated caches | once per 4 generated tokens |

Design points, each motivated by a measured profile or an ablation (full detail in EXPERIMENTS.md):
- **Encoder packing**: HF packs valid post-CNN frames with `nonzero`/`index_select` and splits into windows
  with a Python loop over `cu_seqlens`. At batch 1 the valid frames are a flat prefix of the padded layout, so
  this port keeps a fixed `[C*13]` padded layout and uses one block-diagonal additive mask
  (`window = p // 104`, key valid iff `k < n_valid`) instead — mathematically identical at batch 1, and it
  compiles to a static shape.
- **Audio-placeholder splice**: HF's `masked_scatter` becomes `torch.where(ids==AUDIO_TOK, audio shifted by the
  fixed template offset, text_embeds)` — same result, static shape.
- **K/V cache**: `DynamicCache` → static `[28,8,768,128]` bf16 tensors threaded through the graph functionally.
  The decode slot write is `torch.where(arange==pos, new, old)`, not `index_copy_`: an in-place/graph-input-
  mutation variant was tried twice (round 1's `QASR_KV=inplace`, round 3's `QASR_KV=lateK`) and was **slower**
  both times (+2.5 ms/step and +18%/token respectively) despite moving fewer bytes — graph-input mutation
  overhead through this compiler outweighs the DMA it saves.
- **lm_head**: the tied `lm_head` as `nn.Linear` over the `[151936,1024]` embedding weight lowered to a
  **per-call DMA transpose of the whole weight** (1.3M DMA instructions, 857 ms of the first working decode
  step). Fix: store a pre-transposed `[1024,V]` buffer split into 8 contiguous column blocks (`logits = cat([h
  @ block_i for i in range(8)])`); one big matmul still made the compiler re-tile/spill the weight, 8 medium
  ones don't. This alone took decode from 655 ms to ~6 ms/step.
- **In-graph argmax**: after the lm_head fix, `torch.argmax` over the `[1,151936]` bf16 logits returned
  `INT_MAX` on-device (a compiler miscompile, not narrowed further) and corrupted the next step's embedding
  gather. Fix: upcast logits to fp32, then `max` + masked-`min` (`greedy_argmax` in `qasr_port.py`).
- **Weight layout**: every decoder-layer weight is pre-transposed to `[in,out]` (`x @ Wt` instead of
  `nn.Linear`'s `x @ W.T`, which cost a Tensor-engine transpose/retile per call) and Q/K/V and gate/up are
  fused into single matmuls. Confirmed by profile: Tensor-engine `TRANSPOSE` flops fell 84% (182→30 GFLOP) and
  weight HBM traffic dropped from "read twice + spilled" to "read exactly once, at the bf16 floor".
- **GQA**: query heads are folded into the row dimension (`q.reshape(1, n_kv, n_rep*T, hd)`) instead of
  materializing a repeated K/V (`expand().reshape()`) — avoids duplicating K/V bytes 2x (n_rep=2) every layer.
- **K=4 token unroll (`DecodeK`)**: profiling showed decode is memory/DMA-bound (88-91% DMA-active, <11%
  serialization gap) with weight reads already at the bf16 read-once floor — so the only lever left was fewer,
  bigger graph calls. `DecodeK` unrolls 4 greedy steps (argmax token fed back via an in-graph embedding gather)
  in one compiled call, amortizing the ~1 ms/call host+runtime overhead over 4 tokens. This also had a genuine
  device-side bonus: the per-layer full-cache `torch.where` rewrite got ~70% cheaper per token when the
  compiler could schedule it across the unrolled steps. K=6/K=8 were tried and rejected (net neutral to
  negative on the actual 10-sample set once wasted post-EOS steps are counted, and ~1.4-2.1x longer to compile).
- **What did *not* work** (kept as negative results, not re-attempted): int8/e5m2/fp8 weight-only quantization
  (either slower on-device due to the dequant cost, or broke token-exactness even on CPU fp32 — this model is
  numerically tight: one sample's bf16 divergence traces to an fp32 top-2 logit margin of 0.004), per-layer
  K/V cache lists instead of one stacked tensor, and a "write the K/V cache once per K-step call" variant —
  all measured slower than the simple functional `torch.where` rewrite on this compiler.

## Files

| File | What |
|---|---|
| `qasr_port.py` | The model: `Encoder`, `Prefill`, `Decode`/`DecodeK`/`DecodeKLate`, all the kept-and-rejected variants behind `QASR_*` env switches (documented inline and in EXPERIMENTS.md) |
| `run.py` | CLI: builds the three compiled graphs, runs the accuracy+latency loop over N samples, writes `artifacts/run_<tag>.json` and a `SUMMARY` line |
| `cpu_ref.py` | HF `generate()` CPU fp32 reference (writes `ref/ref.json`, needed by `run.py`'s accuracy check); also has `load_samples()`/`wer()` helpers `run.py` imports |
| `batch_sweep.sh` | Runs `run.py` for a list of N values, one process per N (see "Known limits" for what "batch size" means here) |
| `env.sh` | Neuron env for LNC=2 (source before every run); also sets `QASR_MODEL` |
| `ref/ref.json` | CPU fp32 reference for 32 `librispeech_asr_dummy` samples (ids, generated ids, text, WER, cpu time) — committed so the accuracy/latency commands below don't need a CPU-only pass first |
| `artifacts/run_bs*.json` | Per-batch-size sweep output, as committed/validated (see table below) |
| `EXPERIMENTS.md` | Every attempt, in order, with before/after numbers: the port, the two device bugs, three profile→improve rounds, and this documentation stage's batch sweep |

## Requirements

A Neuron venv with torch-neuronx 2.14+ and neuronx-cc (validated versions, from `host_neuron_versions.txt` /
`pip list` on the box this was built on):

```
torch                 2.14.0+cpu
torch-neuronx         2.14.3.0.2017+e7b12f7a4.dev
neuronx-cc            2.0.408210.0a0+56671ef7
nki                   0.7.0b1+35892045307.ga8db8bc7
transformers          5.17.0
datasets              4.7.0
jiwer                 4.0.0
soundfile             0.14.0
aws-neuronx-dkms            2.x.12078.0 (host driver)
aws-neuronx-runtime-lib      2.x.81379.0-d6fc24e5b
aws-neuronx-tools            2.0.30074.0-532c67072
```

```bash
pip install "transformers>=5.13" datasets soundfile jiwer   # neuron deps (torch-neuronx, neuronx-cc) via the Neuron venv
```

## Usage

```bash
export NEURON_VENV=/path/to/neuron/venv              # env.sh activates it; its bin/ must hold neuronx-cc
export ASR_WORK=/path/to/writable/workdir             # defaults to this directory
export QASR_MODEL=Qwen/Qwen3-ASR-0.6B-hf              # or a local snapshot dir (what this run used)
source env.sh

python cpu_ref.py 32                                  # 1. CPU fp32 reference -> ref/ref.json (32 samples; ships pre-generated)
python run.py -n 10                                   # 2. Neuron bf16, compiled: accuracy (token-exact+WER) + latency, 10 samples
python run.py -n 3 --device cpu --no-compile --dtype fp32   # same code path on CPU (fast sanity check, no device needed)
./batch_sweep.sh 1 2 4 8 16 32                         # 3. batch sweep (see "Known limits": sequential, not in-graph batched)
```

### Fastest / latest config

The defaults in `qasr_port.py`/`run.py` already **are** the fastest config found (every round's winning
variant is the default; losers are switches, default off): `QASR_LMSPLIT=8 QASR_GQA=group QASR_WT=tf
QASR_KV=where QASR_MULTI=4`. Reproduce it with:

```bash
source env.sh
python run.py -n 10                      # accuracy + per-phase latency, prints a SUMMARY line
```

Expect `SUMMARY ... token_exact=10/10 WER=0.0433 ... decode_step_ms=... step_amort_ms≈4.2-4.7 e2e_ms≈120-140`.

## Batch sweep (as committed; `./batch_sweep.sh 1 2 4 8 16 32`)

**Read this with the caveat above**: `Encoder`/`Prefill`/`DecodeK` all hardcode a leading batch dim of 1 (no
batch axis anywhere in `qasr_port.py`), unlike this repo's `whisper-large-v3` contribution, which decodes B
sequences in lockstep with a real batch dimension. So batch size here means **N independent utterances
transcribed sequentially** through the same batch=1 graphs in one process — the sweep measures whether
dataset-level throughput changes with N (it shouldn't, and doesn't), not a batching speedup.

| N (sequential utts) | token-exact | WER (over that N-prefix) | median e2e/utt (ms) | median step_amort (ms/tok) | median encoder (ms) | median prefill (ms) | throughput (utt/s) | throughput (tok/s) | compile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1/1 | 0.0588 | 117.0 | 4.47 | 10.70 | 8.02 | 8.55 | 196.7 | warm cache (see below) |
| 2 | 2/2 | 0.0370 | 98.9 | 4.66 | 10.65 | 7.91 | 10.11 | 187.1 | warm cache |
| 4 | 4/4 | 0.0120 | 136.9 | 4.46 | 11.25 | 9.84 | 7.55 | 198.1 | warm cache |
| 8 | 8/8 | 0.0350 | 127.3 | 4.36 | 11.20 | 11.56 | 6.37 | 206.2 | warm cache |
| 16 | 15/16 | 0.0388 | 123.3 | 4.37 | 11.11 | 10.00 | 6.82 | 200.9 | warm cache |
| 32 | 30/32 | 0.0423 | 99.6 | 4.34 | 10.85 | 8.16 | 8.46 | 198.1 | warm cache |

"warm cache": these six runs reused the NEFF cache already populated by the optimization rounds, so no new
`neuronx-cc` compile happened (every utterance's (encoder-chunk, prefill-length) bucket pair was already
compiled); only dynamo retracing + NEFF lookup is paid on the first sample of each process (0.1-33 s depending
on whether that bucket pair was first in this specific process). From an **empty** cache, one (C=8, P=128,
K=4-decode) bucket triple cold-compiles in **~459 s** (measured directly for this report, see EXPERIMENTS.md).

**Why throughput is flat**: it isn't a batch-size effect at all — every row runs the identical single-sequence
pipeline once per utterance; what varies row-to-row is which utterances (and therefore which length buckets
and generated-token counts) fall into that N-prefix. Throughput stays in a 187-206 tok/s / 6.4-10.1 utt/s band
regardless of N because there is no batching dimension to exploit. See "Known limits" for what it would take
to change that.

## Known limits / next steps

- **No in-graph batching.** All three graphs hardcode batch=1 (`Encoder`'s windowed-attention packing assumed
  "batch 1 ⇒ valid frames are a flat prefix", `Prefill`'s audio-placeholder splice and `DecodeK`'s caches have
  no batch axis). Extending this the way `whisper-large-v3`'s `transcribe_batch()` does (B sequences in
  lockstep, sharing position) is the most impactful remaining lever for throughput, since the decode step is
  memory-bound on weight reads that would be shared across a real batch.
- **Weight-only quantization doesn't help on this stack**: int8 was slower (dequant overhead > bytes saved,
  2.39 vs 1.98 ms in a 28-block microbench), `float8_e4m3fn`/`_fnuz` don't compile/run on this `neuronx-cc`,
  and `float8_e5m2` (the only faster variant) breaks token-exactness even in CPU fp32 (6-9/10 depending on
  which weight groups are quantized) because this model is already numerically tight in bf16 (one sample's
  divergence traces to an fp32 top-2 logit margin of 0.004).
- **The ~160-180 MB/token "unattributed spill" bucket** (10-13% of decode's DMA bytes, confirmed present in
  three successive profiles) was never localized to a specific source line; two targeted rewrites aimed at it
  (per-layer cache lists, "write cache once per K-step call") both made things slower rather than confirming or
  fixing it. Worth one more profiling pass with explicit per-call-site naming before attempting a fourth fix.
- **Larger `DecodeK` unrolls (K=6, K=8)** were net-neutral-to-negative on this (mostly short, 13-88-token)
  utterance set because wasted post-EOS steps (up to K-1 per utterance) cancel the per-token device-time gain;
  they would likely win on longer-form audio/transcripts.
- No language auto-detection (language is forced to English via the processor call), no timestamps, no
  streaming/partial decoding.
