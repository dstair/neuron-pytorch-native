# Pika-H3-Flex on Trainium2

Port of [Pika-Labs/Pika-H3-Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex)
to AWS Trainium (trn2), on branch `pika-h3-flex`.

**Current state: all customer modes run end-to-end on a trn2.48xlarge** — T2V,
first-frame i2v (FL2VA), image-reference (ref2va), lip-sync (audio reference),
portrait, and 15s long-form, all with decoded 32 kHz stereo audio.

## Start here

- **[docs/STATUS.md](docs/STATUS.md)** — customer asks vs evidence, measured
  numbers, request formats, known limitations. The running ledger.
- **[docs/NEXT_SESSION_PLAN.md](docs/NEXT_SESSION_PLAN.md)** — resume plan /
  session log with hard-won lessons.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — model component findings.
- [deploy/RESTORE.md](deploy/RESTORE.md) — rebuild a fresh trn2.48xl from the
  S3 mirror (~15 min).

## Measured performance

Warm (graph-cached) requests, 1344×768 unless noted, Pika 6-NFE schedule.
"Cold" adds a one-time per-shape NEFF load (~90–225 s) — eliminated from the
request path by the boot warmup (`warmup_num_frames` + `text_seq_bucket`).

| Workload | Config | text enc | denoise | VAE dec | **e2e warm** |
|---|---|---|---|---|---|
| T2V 124 f (5.2 s) | TP=4 (trn2.3xl) | 4.3 s | 221 s (36.8/step) | 20.7 s | **246 s** |
| T2V 124 f + audio | TP=8 (2 devices) | 2.6 s | 66.3 s (9.0/step) | 13.1 s | **82.4 s** |
| T2V 124 f, 3-step | TP=8 | ~5 s | ~35 s | ~13 s | **~55 s** |
| T2V 124 f, 2-step | TP=8 | ~5 s | ~25 s | ~13 s | **~45 s** |
| FL2VA 124 f + audio | TP=8 | 2.9 s | 63.8 s | ~13 s | **83.5 s** (keyframe enc 3.5 s) |
| T2V 362 f (15.1 s) + audio | TP=8×CP=8 (16 devices) | 8.7 s | ~40 s (6.4/step) | ~90–100 s | **144 s** |
| FL2VA 362 f + audio | TP=8×CP=8 | ~9 s | ~45 s | ~90–100 s | **~160 s** (keyframe CPU enc ~42 s at CP=8) |
| T2V 768×1344 portrait | TP=8 | ~5 s | ~66 s | ~13 s | **~85 s** |

**Multi-engine serving (short-form)**: 2× TP=8 engines on core groups 0-7/8-15
run at **81.6 s each concurrently — zero contention** (distinct
`NEURON_RT_ROOT_COMM_ID`, separate cache dirs). Ceiling is 8× TP=8 on 64 cores
≈ **~350 req/hr** short-form (host-RAM-bound; ~300 GB per 2 engines at load).
15s uses all 64 cores for one engine (no multi-engine at that config).

Step counts are per-request (`num_inference_steps`: 1/2/3/6; knots subsample
the trained 6-knot grid). 2/3-step outputs verified coherent.

## What runs where

| Work | trn2.48xlarge (64 cores) | trn2.3xlarge (4 cores) |
|---|---|---|
| 15 s (362 f), any mode | ✅ TP=8×CP=8 (the only config that fits; see limits below) | ❌ OOMs (34.6 GB > 24 GB/core at 108k rows) |
| TP=8 speed (82 s vs 246 s) | ✅ | n/a (TP=4 max) |
| Multi-engine serving | ✅ 2–8× TP=8 | ❌ single engine only |
| Development / correctness at 124 f | ✅ | ✅ (T2V/FL2VA/ref2va/lip-sync, TP=4, 246 s) |
| FL2VA/ref2va at ≤124 f | ✅ | ✅ (large 2048-edge references encode ~17 min CPU here) |
| Unit tests, layout math, patch-chain dev | ✅ | ✅ (also works on any CPU host) |

## Hard limits discovered (why configs look like they do)

- DiT block graph OOMs past ~60k rows/rank (34.6 GB at 108k) → CP for 15 s.
- `attention_cte` materializes an S_q×S_k bf16 tile → uint32 overflow past
  ~46k×46k → CP=8 (13.5k rows/rank) clears both this and the OOM. CP=2 failed.
- NKI ring-attention kernel doesn't survive dynamo fake-tensor tracing →
  all-gather CP fallback (`apply_cp_fallback.py`).
- No-swap at TP=8 OOMs (DiT 8.3 + VAE 4.9 + text enc 8 GB > 24 GB/core);
  the swap layout stays. CP=8 swap needs ~216 GB of host-maps disk.
- Device VAE-encode inside the swap overflows the executor queue past
  ~1.4 Mpx; larger conditioning images use the upstream CPU path.
- Distributed VAE executor is collective: every VAE rank must call in
  lockstep; conditioning encodes broadcast from the VAE ranks at CP>1.

## Layout

| Path | Contents |
|---|---|
| `docs/` | STATUS (ledger), NEXT_SESSION_PLAN (resume), ARCHITECTURE, FL2VA_INTEGRATION |
| `deploy/` | `apply_*.py` patch chain (apply in numbered order in RESTORE.md), stage configs, `box.sh`/`box48.sh` SSH helpers, bootstrap/restore scripts |
| `run_fl2va.py`, `run_ref2va.py` | Mode runners (T2V uses the plugin's `examples/minimax_h3/run.py`) |
| `patches/` | LoRA-at-load + renoise-sampling patches, `plugin_pika_lora.py` |
| `test_fl2va*.py` | Unit tests (layout math, frozen anchors, T2V bit-equivalence) |

**WIP**: `deploy/apply_video_refs.py` — ref2va video references (motion/camera,
with soundtrack). Chain-validated (all 16 patches apply + parse) but **never
run on device**. Video vision-tower blocks are unsupported by the Neuron
encoder (labels only). This is the top resume item.

The prior MiniMax-H3 project (cache/FP8/sparsity work, benchmark harnesses)
lives on the `minimax-h3` branch under `contrib/minimax-h3/` — reference only.
