# Pika-H3-Flex FL2VA Implementation Status

**Last updated: 2026-11-05 (marathon session, ~04:00 UTC)**

## Customer asks vs status

| Ask | Status |
|---|---|
| LoRA on base H3 at load | DONE (both adapters, TP-sharded merge) |
| first-frame-to-video-and-audio | DONE — FL2VA first-frame verified (PSNR 35.5 dB anchor) + audio decode (32 kHz stereo WAV) |
| ref2va (image references) | DONE — layout + encode verified; subject carries through |
| first-frame-and-audio (lip-sync) | DONE — audio reference rows, envelope corr 0.920 to the supplied track |
| 1344x768 | DONE |
| 768x1344 portrait | compiling at this writing |
| up to 15 s (362 frames) | pending |
| steps 2/3/6 | DONE — 3-step verified coherent; 2-step pending spot-check; per-request `num_inference_steps`, knots subsample the trained grid |
| speed | TP=8: 8.9s/step warm T2V; keyframe device encode 10s (was 120s CPU); see tuning below |

## Session 3 additions (2026-11-05)

- **Audio decode** (`apply_audio_decode.py`): audio rows -> (2,C,T) -> denorm -> audio VAE
  on host -> 32 kHz stereo WAV at `$MINIMAX_H3_OUT/request_NNN.wav`. Rank-0 WAV write;
  audio VAE loads on all ranks (needed for reference encode).
- **Step counts** (`apply_step_count.py`): `_pika_step_knots(num_steps)` — 6: full grid,
  3: [1000,500,100,0], 2: [1000,250,0], 1: [1000,0]; `pika_denoising_step_list` overrides.
- **Ref2VA** (`apply_ref2va.py`): ref2va layout (image refs at tier short edge on the
  shared rotary clock, +1.0 per image; audio refs clean t=1.0, 4-entry timestep plan),
  image normalize (2048 short edge, 32-round), audio-reference encode (resample 32k,
  stereo upmix, truncate to video duration, posterior **mode**).
- **Device VAE encode** (`apply_device_vae_encode.py`): compiled encoder graph inside the
  decode swap; 120s CPU -> ~10s device for a canvas-size keyframe.
- **Encode routing** (`apply_encode_fallback.py`, `apply_encode_broadcast.py`,
  `apply_encode_paths.py`): small images -> device (all ranks lockstep, distributed
  executor is collective!); large images -> upstream tiled CPU encode per rank, threads
  capped at cores/ranks; audio ref -> rank-0 + cpu_group broadcast.
- **Config knobs**: `compile_vae_encoder`, `reference_short_edge` (2048 released),
  `device_encode_max_pixels` (1.4M), `pika_denoising_step_list`.

## Hard-won lessons (session 3)

- **The distributed VAE executor is collective** — every rank must call encode/decode in
  lockstep; rank-0-only hangs the group (gloo "connection closed by peer").
- **Executor queue full** = too many compiled-graph submissions without a drain; hits the
  tiled device encode past ~1.4 Mpx inside the swap. CPU fallback for big references.
- **192-thread MKL spike OOM-kills workers** — cap encode threads; the workers otherwise
  run 1 intra-op thread, so CPU encode needs an explicit (bounded) budget.
- **No-swap TP=8 OOMs**: DiT(8.3)+VAE(4.9)+encoder(8) GB/rank resident > 24 GB. The swap
  layout stays.
- Apply scripts: compute the output BEFORE truncating the target file.
- s5cmd drops execute bits; mirror them from the source's `find -perm -u+x` list.

This document tracks the FL2VA (first/last-frame image-to-video) implementation progress.
Prior session: T2V working end-to-end (285s, 12.6s/step).

## trn2.48xlarge TP=8 bring-up (DONE, 2026-11-04)

New box: trn2.48xlarge `i-0a981ba5acfbaef04`, ap-south-2b (capacity reservation
`cr-049744aca3bf47932`), 16 devices / 64 logical cores (LNC2), 2 TB RAM, 500 GB root.
CF stack `pika-h3-trn2-48xl` from `dev/trainium-model-dev/cloudformation/trn_capacity_block.yml`
(VolumeSize bumped to 500 GB). SSH helper: `deploy/box48.sh run '<cmd>'` (key
`~/.ssh/trn2-ap-south-2-20261104`). SG `sg-09f409131909c07f4`: SSH allowed from corp prefix
70.232.80.0/25 + workstation egress 99.87.8.17/32.

**TP=8 T2V works** (`pika_h3_flex_stage_tp8.yaml`, TP=8/ring=1/VAE-PP=8 on devices 0-7):
- Warm request: text encode 2.9s, **denoise 53.4s (8.9s/step)**, VAE 12.6s, **total 68.9s**
  (vs 246s warm at TP=4 on the 3xl — 4.1x per-step speedup from 2x cores; TP=8 also has
  more HBM headroom per rank). Output video coherent, matches prompt.
- First (cold) request 773s — all TP=8 graphs compile fresh; host-weight maps rebuild for
  r*of8 (TP=4 maps don't transfer).
- Run dir on box: `/data/omni29/runs/pika_t2v_tp8`.

**Data plane (s5cmd, much faster than rsync):** cross-region rsync ran at ~13 MB/s; S3 via
s5cmd (`--numworkers 64`) did ~435 MB/s up / ~1.1 GB/s down. Bucket
`s3://pika-h3-transfer-933591302813-apse4` (ap-southeast-4) with a bucket policy granting
the two instance roles; MiniMax-H3 (144 GB), Pika-H3-Flex (2.8 GB), venv (12 GB), plugin
(3.3 GB) all mirrored there. **s5cmd does not preserve execute bits** — mirror them with the
source's `find -perm -u+x` list (see session history) or venv binaries break.

Lessons: kill-detached-jobs carefully (a killed rsync left a 0-byte pipeline file); corp
egress IP differs per destination region (check from the instance side with `ss -tn`);
ap-south-2 has no corp managed prefix list.

## FL2VA bring-up (DONE, 2026-11-05, trn2.48xl TP=8)

First-frame FL2VA runs end-to-end on the 48xl: `/data/omni29/runs/pika_fl2va_first`.
Keyframe = frame 0 of the T2V output; same prompt/seed. `FL2VA: 1 keyframe(s) -> 1008
condition rows (encode ~120s CPU per rank)`, denoise compiled a new FL2VA-shaped graph
(seq +1008 rows, 3-entry timestep), total 736s cold. **Output frame 0 anchors on the
keyframe (MAE 3.08, PSNR 35.5 dB — the expected t=0.999-noise + VAE-roundtrip delta)**
and the video develops coherently (perch -> wings -> flight). Runner:
`/data/omni29/fl2va/run_fl2va.py` (also in the repo).

Warm FL2VA timing not yet measured (engine shutdown after one request; expect
~keyframe-encode 120s + denoise ~60s + VAE ~13s once graphs are cached).

### Remaining FL2VA follow-ups

- **Vision-tower conditioning** (quality gap): keyframes currently condition only via
  VAE-anchor rows; the reference also prepends `"<Picture i>: " + vision block` to the
  text presentation (Qwen3-VL vision tower, hidden_states[50], vision rows tagged 0).
  Text encoder port is text-only today.
- **last-frame anchor test** (`--last-frame`), and **first+last** together.
- **Device-side VAE encode** (compile_encoder / swap) to replace the 120s CPU encode.
- **Warm-timing run** for FL2VA.

### Files Created

| File | Purpose | Status |
|------|---------|--------|
| `patches/pika_fl2va.patch` | Main pipeline patch | Ready to apply |
| `pipeline_fl2va.py` | Standalone utilities | Reference only |
| `run_fl2va.py` | Test runner | Ready to deploy |
| `test_fl2va.py` | Unit tests | Passing locally |
| `deploy/pika_h3_flex_fl2va_stage.yaml` | Stage config | Ready to deploy |
| `docs/FL2VA_INTEGRATION.md` | Integration guide | Complete |

### Next Steps (On Box)

1. ~~Deploy the patch~~ — DONE (via `deploy/apply_fl2va.py`, applied on both boxes)
2. ~~Test FL2VA end-to-end~~ — DONE (first-frame, TP=8, anchors verified)
3. **Add vision tower encoding** (future):
   - Current implementation does NOT encode keyframes through Qwen3-VL
   - Text encoder doesn't see the keyframe image
   - May produce suboptimal conditioning quality

### Implementation Details

#### Key Changes in Patch

1. **VAE encoder compilation** - `compile_vae_encoder: true` in stage config
2. **FL2VA layout** - New `_build_fl2va_layout()` function with keyframe anchors
3. **Keyframe encoding** - `_prepare_keyframe_latents()` and `_noise_keyframe_latents()`
4. **Frozen anchors** - Modified `_denoise_pika()` to only update generated rows
5. **Request handling** - `forward()` now accepts `multi_modal_data.keyframes`

#### Request Format

```python
request = {
    "prompt": "A parrot flying...",
    "multi_modal_data": {
        "keyframes": [PIL.Image, ...],  # first-frame, optionally last-frame
    },
}
```

#### Known Limitation: Vision Tower

The current implementation does NOT encode keyframes through Qwen3-VL's vision tower. This means:
- Text encoder doesn't see keyframe image content
- Conditioning relies only on VAE latents, not vision embeddings
- May produce lower quality results than full reference

Full FL2VA would require:
- Extending `NeuronQwen3VLTextEncoder` to handle vision inputs
- Building presentation: `"<Picture i>: " + vision_block`
- This is significant work; current approach is a "fast path"

## Previous Session Notes (T2V)

**T2V working (2026-11-04):**
- Warm request: text encode 83.2s, denoise 181.0s (12.6s/step), VAE 21.0s
- Total 285.5s, exit 0, repeat deterministic
- Output video coherent and high-quality

**Box status:**
- trn2.3xlarge `i-0498e6292f78f9774`
- Disk at 99% (8.2 GB free) - no re-downloads!
- Use symlinks for shared weights

**Hard-won lessons:**
- VAE encoder runs in FP32 (BF16 produces NaN)
- Generator draw order matters for reproducibility
- Use `index_select` not strided slices for Neuron
- Frozen anchors need to be noised *before* generating target noise
