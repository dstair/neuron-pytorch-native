# Whisper large-v3 on Trainium2 (PyTorch-native) — experiment log

Record of what was tried, what it did, and why the code looks the way it does. Newest entries at the bottom.
Hardware: trn2.3xlarge, LNC=2, 1 logical NeuronCore per run, bf16.
Stack: torch 2.14.0, torch-neuronx 2.14.3, neuronx-cc 2.0.408210, nki 0.7.0b1, transformers 5.17.
Accuracy reference: CPU fp32 HF `generate()` on 10 `hf-internal-testing/librispeech_asr_dummy` samples (`cpu_ref.py` → `ref/`).

## Setup findings

- **HF `generate()` on the Neuron device doesn't work.** Eager `aten::index` inside generate's host bookkeeping
  lowers to torch-neuronx's `_mask_bounds_check_kernel`, compiled LNC=1 → `NCC_EVRF066` on this LNC=2 box even with
  `NEURON_LOGICAL_NC_CONFIG=2`. Keeping the model on CPU and compiling only `forward` instead makes `StaticCache`
  mix devices (`index_copy_` cache on `neuron:0`, `cache_position` on CPU).
  → Static-shape ports: the host does embedding lookup, suppress rules, and argmax; no gather/index ops in graphs;
  cache writes via `torch.where` against `arange == pos`.
- **`torch.compile(..., dynamic=False)` is required.** A second bucket shape otherwise retraces with symbolic
  shapes and the Neuron backend fails: `Unimplemented call_function: sym_sum`.
- Env: `env.sh` (`NEURON_LOGICAL_NC_CONFIG=2`, scratchpad 1024, venv on PATH, otherwise `posix_spawn` fails).

## Experiments (`whisper_static.py`)

### Baseline — static port (2026-10-14)
Encoder graph + cross-K/V graph (once per utterance) + T=1 decoder step over a preallocated 448-slot cache.
- Accuracy: 10/10 token-exact vs CPU fp32, WER 3.86% (= CPU).
- Encoder 137.5 ms per 30 s window (94 ms device). Decode **32.2 ms/token** (device 32.9 ms in profile; host
  loop overhead only ~3.5 ms — `whisper_step_timing.py`: 32.4 full loop vs 28.8 resident/async).
- Cold compile ~4 min.

### Profile of the decoder step (2026-10-14)
DMA-bound: 26.2 ms DMA active, 6.77 GB moved per step vs 2.0 GB necessary, 258 GB/s achieved (435 peak).
TensorE active only 6.2 ms. Three causes, by bytes:

1. **KV-cache write re-reads the whole cache (~2.4 GB/step).** Self-attn K and V caches (36.7 MB each, one
   `[32,1,20,448,64]` tensor) are each read 1.21 GB per step, 33×, because writing layer *i* via
   `k_cache[i].copy_(torch.where(...))` lowers to every layer touching the entire 32-layer cache.
   Fix: one cache tensor per layer, written only at the current slot.
2. **K transposed on the fly (~2.4 GB/step).** ~3 M tiny DMA transposes (~939 B packets) turn K into Kᵀ for
   `q @ Kᵀ`, for both the self-attn and cross-attn caches. That's also why DMA only reaches 258 GB/s.
   Fix: store K as `[H, D, S]`, with the cross-K/V graph emitting that layout once per utterance.
3. **Weights transposed every step.** 99% of TensorE work is transposes: 243 GFLOP vs 1.9 GFLOP of real matmul —
   every `nn.Linear` weight is re-transposed on each token. ~18 ms of TensorE time, though not yet on the
   critical path.
   Fix: pre-transpose the weights once at load.

Side finding — encoder: 94 ms device, 27 GB DMA vs 1.3 GB necessary (21× reloads). PR #105's encoder megakernel
targets exactly this.

### Fixes 1+2+3 together (2026-10-15)
Implemented in one change: per-layer caches (`new_cache()`), K stored `[B,H,D,S]` for self- and cross-attn
(`_kT`, manual `q @ kT` attention instead of SDPA), `LinT` pre-transposed `[in,out]` decoder weights + lm_head.
CPU fp32: 3/3 token-exact (unchanged numerics).

**Result: DONE — all three fixes landed. Decode 32.2 → 11.1 ms/token wall (2.9×), 32.9 → 9.0 ms device step (3.6×).**
Accuracy unchanged: 10/10 token-exact vs CPU fp32, WER 3.86%.

| Metric (decoder step) | Baseline | Fixes 1+2+3 |
|---|---|---|
| Wall, full host loop | 32.2 ms/token | **11.1 ms/token** |
| Wall, resident inputs, async (device throughput) | 28.8 ms | 7.9 ms |
| Device step (profile `total_time`) | 32.9 ms | **9.0 ms** |
| DMA bytes moved / necessary | 6.77 GB / 2.0 GB | **2.00 GB / 1.99 GB** (excess gone) |
| DMA active / achieved BW | 26.2 ms / 258 GB/s | 7.1 ms / 281 GB/s |
| TensorE transpose FLOPs vs real matmul | 243 GF vs 1.9 GF | 14 GF vs 1.9 GF |
| Encoder + cross-K/V stage, per utterance | 137.5 ms | 99 ms (cross-K/V layout also cheaper) |

Per fix: (1) self-attn caches now read/written once per layer (the 33× re-read is gone); (2) no more
`d2dtranspose` DMAs for Kᵀ; (3) TensorE transposes down 17×. The bytes moved now match the necessary bytes, so the
remaining time is pure bandwidth/scheduling, not wasted traffic.

What's left in the 11.1 ms (profile `cap_1084467210067063`, ingested as `w1_step`, op-level
attribution present — compiled with `XLA_IR_DEBUG/XLA_HLO_DEBUG/NEURON_FRAMEWORK_DEBUG`):
- **DMA efficiency:** 7.1 ms active vs 4.6 ms at peak for the same 2.0 GB (281 vs 435 GB/s). Largest single
  tensor is the lm_head `[1280, 51866]` (133 MB, 6.6% of traffic).
- **Engine serialization:** 1.9 ms (9.0 total vs 7.1 DMA-bound).
- **Host per-token sync:** ~3 ms (11.0 ms full loop vs 7.9 resident/async) — the D2H of logits + argmax on the host
  blocks the next step from being enqueued.

### Round 2 — proposed (2026-10-15, from `w1_step` profile via `deep.py`)
Remaining 11.1 ms/token = ~9.0 ms device step + ~3 ms host sync per token (some overlap). Device step: DMA-bound
(7.1 ms active), all 2.0 GB of weights/caches go through the GpSimd software-DGE queue (`qGpSimdDynamic`) as
759 k packets averaging 2.6 KB; GpSimd is 77% busy, mostly generating those descriptors.

A. **Keep the greedy loop on the device** (expected ~2.5–3 ms/token). Evidence: full host loop 11.0 ms vs the same
   graph back-to-back with resident inputs 7.9 ms — the per-token D2H of logits + host argmax + re-enqueue costs ~3 ms.
   Change: suppress rules as additive bias inputs, argmax in-graph, next-token embedding in-graph (try gather first;
   fall back to one-hot @ embedding, which costs +133 MB/step), token and position stay on device; the host enqueues
   steps without syncing and reads tokens back every 8 steps for the EOS check (≤7 wasted steps per utterance).
B. **Write one cache slot instead of re-blending the whole cache** (expected ~0.5–1 ms). Evidence: the
   `torch.where` blend still rewrites every layer's full cache each step: 73 MB HBM writes + 37 MB reads, ~1.3 ms
   Vector + ~0.9 ms GpSimd in `select` ops, plus most of the 4.9 ms of software DMA in unattributed instructions.
   Change: `index_copy_` at `pos` (in-graph scatter, 164 KB/step); must check it doesn't pull in the LNC=1
   bounds-check kernel that broke eager `aten::index`.
C. **Fuse self-attn Q/K/V into one `[1280, 3840]` matmul** (expected ~0.3–0.8 ms). Evidence: each of q/k/v reads
   105 MB/step in ~2.4 KB packets; fewer, wider matmuls mean fewer descriptors for GpSimd and longer DMA runs
   (2–4 KB packets run at 17.9 GB/s per packet vs 20.6 for 4–16 KB). Cross-attn q can't join (different input).
D. Later: the vocab projection `[1280, 51866]` (133 MB, 6.6% of bytes) runs as a serial tail after layer 32
   (~1.3 ms of TensorE busy time in the last 0.5 ms).

### Round 2 — tried A+B+C (2026-10-15)
Probe first (`probe_index_ops.py`): in-graph `F.embedding`, `index_copy_` at a device `pos`, and `argmax` all
compile and match CPU — only *eager* `aten::index` hits the LNC=1 bounds-check bug. All three implemented together;
CPU fp32 3/3 token-exact, Neuron 10/10 token-exact, WER 3.86%.

**Result as built: 11.1 → 10.46 ms/token (only −6%).** Timing split (`whisper_step_timing.py`, 64 steps):
sync every step 30.2 | flush every 8 10.36 | fully async 9.98 ms/step → A removed the host sync as intended
(flush-8 ≈ async), but the device step got *slower*: 7.9 → 10.0 ms async. Profile `w2step`
(`cap_562235213420811`): device 10.4 ms (was 9.0), DMA active unchanged at 7.1 ms and bytes slightly lower
(1.93 GB), but engine serialization gap 1.9 → 3.3 ms. B did remove the `select` work (Vector 1.26 → 0.03 ms) and
the 73 MB of cache writes (→ 0.2 MB).

**2b — tried, no effect:** guessed the scatter's HBM round trip sat on the critical path (attention read the
cache after writing it, ×32 layers), so attended over the old cache + the new k/v directly and wrote the slot
afterwards. 10.6 ms/step — not the cause.

**Ablation** (`ablate.sh`, `ASR_ABLATE=` switches in `whisper_static.py`; device throughput, fully async).
At the time `where` was the switch (index_copy_ was the default); the shipped default is the where-blend and
the switch is now `ASR_ABLATE=index`:

| Variant | ms/step |
|---|---|
| Round 2 as built (A+B+C) | 10.34 |
| without C (separate q/k/v) | 10.30 |
| **without B (where-blend cache write)** — shipped default | **7.08** |
| without in-graph argmax | 9.95 |
| all three off (≈ round 1 + in-graph embedding) | 7.96 (= round 1's 7.9 → in-graph embedding gather is free) |

Conclusions:
- **B rejected:** the `index_copy_` scatter costs ~3.3 ms/step on device despite moving ~110 MB less — far more
  than the where-blend it replaced. Kept behind `ASR_ABLATE=index`.
- **C kept, no measurable effect on its own** (10.30 vs 10.34); harmless, fewer ops.
- **A kept:** removes the ~3 ms host sync per token; in-graph argmax + suppress costs ~0.4 ms of device time.
- Final config = A + C + where-blend write.

**Result: DONE — round 2 (A + C, B rejected). Decode 11.1 → 7.42 ms/token wall (1.5×); collectively
32.2 → 7.42 ms/token since the baseline (4.3×).** Accuracy unchanged: 10/10 token-exact, WER 3.86%.
Longest sample (88 tokens) decodes in 672 ms (round 1: 966 ms).

| Metric (decoder step) | Baseline | Round 1 | Round 2 |
|---|---|---|---|
| Wall per token (`transcribe`, 10 samples) | 32.2 ms | 11.1 ms | **7.42 ms** |
| Host sync cost per token | ~3.5 ms | ~3 ms | ~0.3 ms (flush every 8) |
| Device step (profile `total_time`) | 32.9 ms | 9.0 ms | **8.0 ms** |
| DMA active / achieved BW | 26.2 ms / 258 GB/s | 7.1 ms / 281 GB/s | 6.2 ms / 323 GB/s |
| Engine serialization gap | — | 1.9 ms | 1.8 ms |

Profile `w2final` (`cap_629968501572712`, op-level attribution; compiled into a separate
`neff-cache-debug` so the debug flags actually took effect). Note the profiled step (8.0 ms) is a bit
slower than steady-state wall (7.4 ms) — profiling overhead. What's left:
- DMA-bound: 6.2 ms active vs 4.6 ms at peak for the same 2.0 GB.
- The where-blend cache write is back: `select` ops ~1.3 ms Vector + ~1.0 ms GpSimd, 73 MB written per step.
- In-graph argmax/suppress runs as a serial tail after the vocab projection (~0.4 ms in the ablation).

### Batch sweep (2026-10-15, round-2 config)
`transcribe_batch()`: B utterances decoded in lockstep (same 4-token prompt → same position for every sequence), stops
when every sequence has emitted EOS — short utterances ride along until the longest finishes. CPU fp32 B=3
token-exact. `whisper_batch.py --bs B` (one process per BS, own compiled graphs; `sweep.sh`), librispeech dummy,
≥ 32 timed utterances after a warm-up batch. Token-exact only checkable on the 10 referenced samples (BS ≥ 16 times
samples 16+, hence 0/0); WER covers all.

| BS | ms/step | ms/step per seq | tokens/s | utt/s | RTFx | encoder ms/batch | WER | token-exact | compile |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7.38 | 7.38 | 105 | 3.11 | 23.0 | 100 | 3.66% | 9/9 | cached |
| 2 | 11.1 | 5.55 | 109 | 3.15 | 23.0 | 211 | 3.50% | 8/8 | 2.6 min |
| 4 | 15.8 | 3.95 | 120 | 3.40 | 23.3 | 448 | 3.92% | 6/6 | 3.1 min |
| 8 | 25.4 | 3.18 | 124 | 3.61 | 21.9 | 914 | 3.70% | 2/2 | 4.3 min |
| 16 | 45.2 | 2.82 | 129 | **3.80** | 20.7 | 1905 | 3.80% | 0/0 | 6.5 min |
| 32 | 103.5 | 3.24 | 99 | 3.20 | 18.9 | 3893 | 3.49% | 0/0 | 12.1 min |

**Result: batching barely helps — peak 3.8 utt/s at BS=16, only 1.2× BS=1; BS=32 regresses.**
- The encoder doesn't batch at all: ~120 ms per utterance at every BS (≈ 40% of wall time at BS=8–16).
- The decode step grows ~2.5–3 ms per extra sequence even though the weights are shared; if the step stayed
  weight-bandwidth-bound it would grow only ~0.3 ms/seq (each sequence adds 73 MB self-cache + 246 MB cross-cache).
- Accuracy is stable across BS (WER 3.5–3.9%, all referenced samples token-exact).

**BS=8 step profile** (`w2bs8`, debug-info compile, `ASR_BS=8 whisper_profile.py`): device 29.0 ms, 4.8 GB moved
(= necessary) at 215 GB/s; GpSimd is the bottleneck at 85% busy (software-DGE descriptor generation), DMA 77%.
Each extra sequence adds ~430–460 MB of traffic per step, which at 215 GB/s is ~2.1 ms — matching the measured
2.5–3 ms/seq:

| Traffic per step at BS=8 | Bytes | Scales with BS? |
|---|---|---|
| Cross-attn K/V reads (32 layers × `[8,20,1500,64]` × 2) | ~1.97 GB (246 MB/seq) | yes |
| Decoder weights incl. the 133 MB vocab projection | ~2.0 GB | no (shared) |
| Self-attn cache: where-blend read 294 MB + write 587 MB, attention reads ~585 MB | ~1.46 GB (~180 MB/seq) | yes |

Levers (not yet tried): (1) self-cache length buckets — the cache is processed at all 448 slots while these
utterances average ~30 tokens, so ~93% of the ~180 MB/seq is padding; (2) fp8 cross-attn K/V (halves 246 MB/seq,
accuracy risk — the 1500-frame window can't be truncated without changing outputs); (3) the encoder's 21× DMA
reloads (~120 ms/utt, ~40% of wall at BS 8–16); (4) DP=4 across the box's 4 logical NeuronCores (sweep used one).
