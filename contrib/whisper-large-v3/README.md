# Whisper large-v3 on AWS Neuron (PyTorch-native)

Runs [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) speech-to-text on Trainium2 with
`torch.compile(backend="neuron")`. The model is a static-shape re-implementation of the HF decoding loop that reuses
the HF weights. Output is token-for-token identical to HF `generate()` on CPU in fp32.

## Results (trn2.3xlarge, 1 logical NeuronCore, bf16, batch 1)

| | Value |
|---|---|
| Accuracy | 10/10 token-exact vs HF CPU fp32 on `librispeech_asr_dummy`; WER 3.86% (same as CPU) |
| Encoder + cross-attention K/V, per 30 s window | 99 ms |
| Decode | **7.4 ms/token** (from 32.2 ms in the first working port, 4.3×) |
| Throughput | 3.1 utterances/s, 23× real time (BS=1); peak 3.8 utt/s at BS=16 |
| Cold compile | ~4 min at BS=1; graphs are cached in `$TORCH_NEURONX_NEFF_CACHE_DIR` |

[EXPERIMENTS.md](EXPERIMENTS.md) records every optimization round: profiles, what each change did, and the
changes that were rejected.

## Why not HF `generate()`

- With the model on the Neuron device, the eager `aten::index` calls inside `generate()`'s host bookkeeping lower
  to torch-neuronx's bounds-check NKI kernel. That kernel compiles for LNC=1 and fails on an LNC=2 trn2
  (`NCC_EVRF066`), even with `NEURON_LOGICAL_NC_CONFIG=2` set.
- With the model on CPU and only `forward` compiled, HF's `StaticCache` ends up mixing devices.

Instead, `whisper_static.py` builds three fixed-shape graphs:

| Graph | Shapes | When |
|---|---|---|
| `encoder` | `feats [B,128,3000]` → `[B,1500,1280]` | once per utterance |
| `cross_kv` | per-layer K `[B,H,D,1500]` (pre-transposed), V `[B,H,1500,D]` | once per utterance |
| `step` | `tok [B]`, `pos [1]`, per-layer caches → next token `[B]` | once per decode step |

Key design points, all motivated by profiles (see EXPERIMENTS.md):
- **Caches:** one self-attention cache tensor per layer, with K stored pre-transposed `[B,H,D,S]`. The slot is
  written with a `torch.where` blend. `index_copy_` compiles but is ~3 ms/step slower.
- **Weights:** decoder weights are stored pre-transposed (`LinT`), and self-attention Q/K/V is fused into one
  matmul (`FusedQKV`).
- **Greedy loop on device:** token embedding, the suppress-token rules and argmax all run in-graph. The host only
  enqueues steps and reads tokens back every 8 steps for the EOS check.
- **Batching:** `transcribe_batch()` decodes B utterances in lockstep. They share the prompt, so every sequence is
  at the same position.
- **Static shapes:** `torch.compile(..., dynamic=False)` is required. Without it, a second shape retraces with
  symbolic sizes and the Neuron backend fails on `sym_sum`.

## Files

| File | What |
|---|---|
| `whisper_static.py` | The model: `StaticWhisper`, `transcribe()` / `transcribe_batch()`; CLI = accuracy + latency run |
| `cpu_ref.py` | HF `generate()` CPU fp32 reference (writes `ref/whisper_float32.json`, needed by the checks) |
| `whisper_batch.py`, `sweep.sh` | Batch-size sweep (one process per BS) |
| `whisper_step_timing.py` | Decode step timing: sync every step vs flush every 8 vs fully async |
| `ablate.sh` | Step timing with round-2 choices switched via `ASR_ABLATE` |
| `whisper_profile.py` | Dumps NEFFs (runtime inspect + debug info) for `neuron-explorer` |
| `profiling/` | `ingest.sh` (NEFF+NTFF → parquet), `bounds.py` (performance bounds), `attrib.py` / `deep.py` (per-tensor, per-module, per-source-line, DMA-queue breakdowns) |
| `probe_index_ops.py` | Checks that in-graph gather / `index_copy_` / argmax compile and match CPU |
| `env.sh` | Neuron env for LNC=2 (source before every run) |

## Requirements

A Neuron venv with torch-neuronx 2.14+ and neuronx-cc (validated: torch 2.14.0, torch-neuronx 2.14.3,
neuronx-cc 2.0.408210, nki 0.7.0b1), plus:

```bash
pip install "transformers>=5.13" datasets soundfile librosa jiwer   # duckdb pyarrow pandas for profiling/
```

## Usage

```bash
export NEURON_VENV=/path/to/neuron/venv          # env.sh activates it; its bin/ must hold neuronx-cc
export WHISPER_MODEL=openai/whisper-large-v3     # or a local snapshot directory
source env.sh

python cpu_ref.py -n 10                          # 1. CPU fp32 reference -> ref/whisper_float32.json
python whisper_static.py -n 10                   # 2. Neuron bf16: per-utterance latency, token-exact, WER
python whisper_static.py -n 3 --device cpu --dtype float32 --no-compile   # same code path on CPU (fast check)
python whisper_batch.py --bs 8                   # one batch-size point; ./sweep.sh 1 2 4 8 16 32 for the sweep
python whisper_step_timing.py                    # decode-step host vs device split
```

Programmatic:

```python
import torch
from transformers import WhisperProcessor
from whisper_static import MODEL, StaticWhisper

proc = WhisperProcessor.from_pretrained(MODEL)
model = StaticWhisper(MODEL, torch.bfloat16, "neuron", compile_=True)
feats = proc(audio_16khz, sampling_rate=16000, return_tensors="pt").input_features   # [1,128,3000]
ids, timings = model.transcribe(feats)
print(proc.decode(ids, skip_special_tokens=True))
```

English transcription without timestamps only: the prompt is fixed to `<|en|><|transcribe|><|notimestamps|>`.
Audio over 30 s needs chunking by the caller.

## Profiling

```bash
# Compile into a separate cache: a NEFF cache hit reuses NEFFs built without debug info (no op attribution).
TORCH_NEURONX_NEFF_CACHE_DIR=$ASR_WORK/neff-cache-debug ASR_BS=1 python whisper_profile.py
NEFF=$(ls profiles/whisper/neffs/*/*/neff_<step-id>_vnc_0.neff)   # the step is the mid-size NEFF
NEURON_RT_VISIBLE_CORES=0 NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1 \
  neuron-explorer capture -n $NEFF -s cap/p.ntff --profile-nth-exec=2      # writes cap/p_exec_2.ntff
profiling/ingest.sh $NEFF cap/p_exec_2.ntff step
python profiling/bounds.py step; python profiling/attrib.py step; python profiling/deep.py step
```

## Known limits / next steps

- **Batching barely helps** (BS=16 is 1.2× BS=1). Each sequence adds ~460 MB of DMA traffic per step:
  246 MB of cross-attention K/V and ~180 MB of self-cache work at the full 448 slots. Levers, not yet tried:
  self-cache length buckets, fp8 cross-attention K/V, and data parallelism across the 4 logical cores.
- **The encoder** (~120 ms per utterance, no batching gain) moves 27 GB of DMA traffic for 1.3 GB of necessary
  bytes (21× reloads).
- No timestamps, language detection, or temperature fallback.
