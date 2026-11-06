# Resume prompt — Pika-H3-Flex on Trainium (paste into a fresh session)

Read `contrib/pika-h3-flex/README.md` and `docs/STATUS.md` first (branch
`pika-h3-flex`). Everything below is captured there; this prompt is the
short version.

## Context

We ported Pika-H3-Flex (LoRA adapters on MiniMax-H3) to AWS Trainium via the
Neuron vLLM-Omni plugin. **All customer modes work end-to-end on a
trn2.48xlarge**: T2V, first-frame i2v (FL2VA), image-reference (ref2va),
lip-sync (audio reference), portrait, 15 s long-form — all with decoded 32 kHz
stereo audio. Warm numbers: T2V 124 f 82.4 s at TP=8; 15 s 144 s at TP=8×CP=8;
2× TP=8 engines serve concurrently with zero contention.

## Environment options (pick based on what's available)

### A. trn2.48xlarge (needed for 15 s, TP=8 speed, multi-engine)

- If a capacity reservation exists, bring the box up with
  `dev/trainium-model-dev/cloudformation/trn_capacity_block.yml` (VolumeSize
  200→500 GB), then restore from the S3 mirror in ~15 min:
  `contrib/pika-h3-flex/deploy/RESTORE.md` (+ `restore_48xl.sh`). The mirror
  has weights, the fully-patched plugin (`plugin-48xl-patched-v2/`), venv,
  graph caches, configs. Add the new stack's IAM role to the bucket policy
  (a deleted stack's role makes put-bucket-policy fail).
- SSH helper: `contrib/pika-h3-flex/deploy/box48.sh` (update the IP; add your
  egress IP to the SG — the corp prefix list doesn't exist in ap-south-2).
- s5cmd drops execute bits — after any restore, mirror them:
  `cd /data/omni29/venv && chmod +x bin/*`, and re-apply the source's
  `find . -type f -perm -u+x` list (neuronxcc/starfish/bin included).

### B. trn2.3xlarge (dev/correctness at ≤124 f)

- The existing 3xl may still be up: `contrib/pika-h3-flex/deploy/box.sh run
  'pgrep -af run.py'`. It has the session-3 plugin state (FL2VA + audio
  decode + step counts + ref2va; NOT the session-4 patches: warmup, text
  bucket, vision conditioning, CP fallback, vae-rank encode).
- To refresh it, sync `plugin-48xl-patched-v2/` from S3 (or rsync the patched
  file) — the pipeline is one file:
  `vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py`
  (plus `diffusion/attention/cp_attention.py` for the CP fallback).
- TP=4 config: `deploy/pika_h3_flex_stage.yaml` (3xl paths: /mnt/nvme host maps).
- NOT for 15 s (OOMs past ~60k rows/rank; TP=4 has no CP headroom).

## Patch chain (if applying to a fresh plugin checkout)

Order matters; all are anchor-based and idempotent-guarded:
`apply_fl2va → apply_audio_decode → apply_step_count → apply_ref2va →
apply_device_vae_encode → apply_devenc_fix → apply_ref_short_edge →
apply_encode_fallback → apply_encode_broadcast → apply_encode_paths →
apply_warmup → apply_text_bucket → apply_step0_profile →
apply_vision_conditioning → apply_vae_rank_encode` (+ `apply_cp_fallback`
on `cp_attention.py`). The whole chain is validated to apply and parse
(16 patches). `apply_video_refs.py` applies but is **untested on device**.

## State of the work / next tasks (priority)

1. **Video references (WIP, top item)** — `deploy/apply_video_refs.py`:
   ref2va video refs (resample 24 fps → snap to 17n+5 → per-clip VAE encode;
   soundtrack rows pack before video rows, rotary advances by
   `max(audio_latents, sequential_span)`; `"<Video k>: "` labels, audio label
   first). Never run on device. Test: image+audio refs worked; use a generated
   mp4 as the motion reference. Note: the Neuron encoder rejects video
   vision-tower blocks (`video_token_id`) — labels only, quality gap vs the
   reference.
2. **Warm-timing polish**: 15 s VAE decode (~90–100 s) is the residual
   bottleneck (vae_pp=64 was *worse*, 117 s; kept at 16).
3. **Multi-engine scale-out**: validated at 2× TP=8; try 4× (host RAM watch:
   ~300 GB per 2 engines at load; 2 TB box).
4. **ref2va at 15 s** (same math as FL2VA@15s, needs CP=8 compiles).
5. **Quality check vs the official Pika runtime** (HF showcase assets).

## Hard-won lessons (don't relearn)

- Engine workers survive client death mid-compile — `pgrep -af "run.py|neuronx-cc"`
  before relaunching; kill by process group.
- The distributed VAE executor is collective (all VAE ranks lockstep).
- Neuron tensors can't cast on-device; normalize on host, move after.
- 192-thread MKL spikes OOM-kill workers; cap encode threads (world-relative).
- Broadcast headers need 6 slots for 5-D latents (5 slots = crash).
- CP=8 swap maps need ~216 GB disk; swap=25 + CPU keyframe encode (no encoder
  NEFF) is the working 15 s FL2VA combo.
- The per-shape NEFF load (~225 s) is per prompt-token-count: text-bucket
  padding + boot warmup moves it off the request path.
