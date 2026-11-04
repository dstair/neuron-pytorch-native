# Pika-H3-Flex on Trainium — NEXT SESSION resume plan

**Read this first when resuming.** It is the current state + the concrete plan for
the next phase (FL2VA). Branch: `pika-h3-flex`. Older history: `RESUME_PLAN.md`
(progress log) and `ARCHITECTURE.md` (model breakdown). Prior project: `minimax-h3`
branch (do not modify).

## 1. Current state — T2V WORKS on device (2026-11-04)

Pika-H3-Flex text-to-video+audio runs end-to-end on the trn2.3xlarge with the LoRA
adapter merged and the 6-NFE renoise schedule. Evidence in `results/`:

- Warm request: text encode 83.2s, **denoise 181.0s (12.6s/step steady-state)**,
  VAE 21.0s, total 285.5s, exit 0, repeat deterministic, latents finite.
- Output video is a coherent, high-quality parrot clip matching the prompt
  (`results/pika_t2v_contact_sheet.png`). Sigmas match the reference grid exactly
  (`results/t2v_warm_summary.json`).
- Run dir on box: `/data/omni29/runs/pika_t2v_20261104`.

**Confirmed foundations (do not re-verify):** the DiT (33B, 50 layers), text
encoder (Qwen3-VL-32B **incl. vision tower**), VAE, and audio VAE are **bit-identical
to MiniMax-H3** (ranged-read hash-verified). Only the LoRA adapters differ.

## 2. Environment and paths

- Box: trn2.3xlarge `i-0498e6292f78f9774`, ap-southeast-4, TP4/CP1/LNC2.
  SSH helper: `contrib/pika-h3-flex/deploy/box.sh run '<cmd>'` (needs
  `~/.ssh/trn2-ap-southeast-4-20261001`). Do NOT stop the instance; check for
  active jobs first: `deploy/box.sh run 'pgrep -af "run.py|neuronx-cc"'`.
- Env: `source /data/omni29/env29.sh` (venv `/data/omni29/venv`): vllm 0.29,
  Lite 2.13, neuronx-cc 2.28.7631 (SDK 2.33), diffusers 0.40.0, torch 2.13.
  **diffusers 0.40 lacks the Pika pipeline and the renoise scheduler — both are
  implemented manually; do not import `PikaH3FlexModularPipeline` or rely on the
  installed scheduler's `step()`.**
- Model dirs:
  - `/data/models/Pika-H3-Flex/` — downloaded small files (adapter, schedulers,
    tokenizer, processor, configs, manifests, LICENSE, SHA256SUMS).
  - `/data/models/Pika-H3-Flex-Neuron/` — the dir the pipeline loads: symlinks
    shared weights from MiniMax-H3, keeps Pika scheduler/audio_scheduler/tokenizer/
    processor/fl2va. Rebuild script is inline in RESUME_PLAN progress log.
  - `/data/models/MiniMax-H3/` — the actual shared weights (do not modify).
- Plugin (Neuron vLLM-Omni): `/data/omni29/plugin` (private, base a2614a2 + our
  patches). Backups under `/data/omni29/backups/`.
- Runner: `/data/omni29/run29.sh NAME CMD...` -> `/data/omni29/runs/NAME/run.log`
  ending `=== exit N`. Detached: `setsid nohup ... < /dev/null &`.

## 3. What is built (do not rebuild)

| Piece | Location | Notes |
|---|---|---|
| LoRA-at-load | plugin `pika_lora.py`, patch `patches/pika_lora.patch` | merges rank-128 adapter into 6 projections x 52 blocks, per-rank, fp32 deltas. Config `pika_lora_path`. Validated tp=1 CPU + TP4 sharding. |
| Renoise sampling | plugin `pipeline_minimax_h3.py::_denoise_pika`, patch `patches/pika_renoise_sampling.patch` | 6-NFE renoise, config `pika_renoise`. `_pika_shifted_sigmas` matches diffusers main. |
| Stage config | `deploy/pika_h3_flex_stage.yaml` (on box at `/data/omni29/`) | `pika_lora_path` + `pika_renoise: true` on top of the single-device config. |
| Host-weight maps | `/mnt/nvme/minimax_h3/*Pika-H3-Flex-Neuron*` | HARDLINKS to MiniMax-H3 maps (encoder bit-identical) — reused ~67 GB, no extra space. Reuse for future runs; only `.tmp`/`.lock` partials need cleanup. |
| Unit tests | plugin `test/unit/test_minimax_h3_pika_{lora,renoise}.py` | 10 passed on box. |

T2V reproduce command:
```bash
RUN=/data/omni29/runs/pika_t2v_new
setsid nohup /data/omni29/run29.sh pika_t2v_new env MINIMAX_H3_CONST_MAX_CP1=0 \
  MINIMAX_H3_DUMP_BLOCK_INPUTS= MINIMAX_H3_DUMP_LATENTS_DIR=$RUN/latents \
  WORKLOAD_OUTPUT_RW=$RUN python examples/minimax_h3/run.py \
  --model-path /data/models/Pika-H3-Flex-Neuron \
  --stage-config /data/omni29/pika_h3_flex_stage.yaml \
  --num-inference-steps 6 --seed 42 --repeat 2 --output $RUN/out.mp4 \
  > $RUN/run_outer.log 2>&1 < /dev/null &
```

## 4. NEXT PHASE: FL2VA (first/last-frame image-to-video) — the main task

The customer's model is primarily image-to-video (`pipeline_tag: image-to-video`).
FL2VA adds keyframe conditioning on top of the working T2V path. Reference:
diffusers main `src/diffusers/modular_pipelines/pika_h3_flex/` (I have a local copy
at `/tmp/pika-pipeline/` on the workstation — re-fetch if gone).

### What FL2VA needs (all confirmed from the reference)

1. **Packed layout** — `[text | keyframe conditions | target audio | target video]`
   (`before_denoise.py::build_packed_sequence(keyframe_anchors)`):
   - `keyframe_anchors`: tuple of `"first"` and/or `"last"`. Condition rows =
     `len(anchors) * rows_per_frame`, placed right after text.
   - Anchor RoPE time: `"first"` → `anchor_time = num_text_tokens`; `"last"` →
     `num_text_tokens + spans.sum() - _ROPE_FRAME_RESCALE` (numpy pairwise sum —
     replicate exactly; note the comment about summation order differing from ref2va).
   - Token tags: keyframe vision-block rows are tagged **0 (video)**, not text.
   - Returns `num_condition_video_rows` (leading video rows that are conditioning).

2. **Keyframe encoding** (`encoders.py::encode_vae_condition`): VAE-encode the
   keyframe image to latents with the video VAE, using ImageNet `pixel_mean=(0.485,
   0.456,0.406)` / `pixel_std=(0.229,0.224,0.225)`, sample the posterior under
   **`keyframe_encode_seed = 42`** (fixed, independent of the request generator),
   and round to float16 before normalizing. Our pipeline currently only has the VAE
   **decoder**; the VAE **encoder** path must be added (MiniMax-H3's VAE has it).

3. **Noise augmentation**: anchors are noised to **`keyframe_noise_aug = 0.999`**
   via `scheduler.scale_noise(condition, 0.999, noise)` (rectified-flow:
   `x_t = t*x0 + (1-t)*noise`). Not clean t=1.0 — the model was trained with
   slightly-noised anchors.

4. **Anchors in the loop**: `latents = cat([condition_rows, latents])`; the renoise
   step only writes rows `[num_condition_video_rows:]` — anchors survive untouched.
   `row_timestep_plan` pins condition rows at `max(video_t, keyframe_noise_aug)`.

5. **Vision conditioning in the text encoder** (`encoders.py`): the Qwen3-VL
   **vision tower** encodes the keyframe image (`pixel_values`, `image_grid_thw`);
   the keyframe prepends a label + vision block; `mm_token_type_ids` (0 text, 1
   image, 2 video) drive Qwen's mrope. Conditioning reads `hidden_states[50]` (not
   final). The vision weights are already on the box (verified in the encoder).

### Suggested implementation order

1. VAE-encode keyframe image -> conditioning latents (host or device VAE encoder),
   with the exact ImageNet normalization + seed-42 posterior + float16 rounding.
2. Packed layout with keyframe anchors + position ids (replicate the numpy pairwise
   sum for the `"last"` anchor time).
3. Vision-tower encoding for the keyframe (Qwen3-VL, hidden_states[50]) — the
   encoder port currently does text-only; add the vision path.
4. Extend `_denoise_pika` to write only generated rows (anchors frozen) and pin
   anchor timesteps at `keyframe_noise_aug`.
5. Bring-up: one keyframe image + prompt, 6-step, compare against the official Pika
   FL2VA output if obtainable; else check temporal coherence to the keyframe.

## 5. Other follow-ups (after FL2VA)

- **Step-0 warmup (~105s/process):** first block_modulations / DiT executor load.
  Profile whether it's AdaLN host compute or NEFF load; consider caching the 6
  steps' block_modulations across requests (fixed schedule).
- **Ref2VA:** `transformer_ref/` + `ref2va/adapter.safetensors` + vision encoder +
  multimodal reference rows (audio reference rows too). Defer until FL2VA works.
- **Quality gate:** compare against the official Pika runtime outputs (the HF repo
  has showcase assets + prompts under `assets/showcases/`).

## 6. Hard-won lessons (do not re-learn)

- **Shared weights are reused via symlinks/hardlinks — do not re-download** the
  DiT/encoder/VAE. Disk is tight (~27 GB free); keep it that way.
- **Do not import the Pika pipeline or installed scheduler for sampling** — 0.40
  can't do the shift-applied grid or renoise; both are manual in `_denoise_pika`.
- **Host-weight maps are tagged by model basename.** Reuse via hardlink when weights
  are identical (that's how the Pika map issue was solved). Clean stray `*.tmp` /
  `*.lock` partials before relaunching after an ENOSPC failure.
- **Fence timing with an output probe** (`probe.cpu()`); `torch.neuron.synchronize()`
  does not drain the compiled Executor queue.
- **No eager device collectives** on neuron tensors in this Lite stack (fails);
  use in-graph collectives or `tp_group.cpu_group` for CPU tensors.
- **LNC2 internal collectives need `NEURON_RT_ROOT_COMM_ID`** set before Lite init.
- **Detached runs:** put `source env29.sh` INSIDE the `setsid bash -c` command; never
  `pkill -f <pattern>` that can match your own launch command string.
- **Dense repeats are bit-identical**, so any divergence is attributable to a change.
  Compare latents + encoded video (SSIM/PSNR) + a contact sheet before claiming quality.
- The full Neuron/compile/HBM lesson list is in `RESUME_PLAN.md` section 5.

## 7. First actions checklist

- [ ] Confirm box idle: `deploy/box.sh run 'pgrep -af "run.py|neuronx-cc"; df -h /data'`
- [ ] Re-read this plan + `ARCHITECTURE.md` + the FL2VA reference files
      (`before_denoise.py` build_packed_sequence, `encoders.py` encode_vae_condition).
- [ ] Add the VAE-encoder path for keyframe images (exact normalization + seed 42 +
      float16 rounding).
- [ ] Implement the keyframe packed layout + anchor position ids (numpy pairwise sum).
- [ ] Add vision-tower conditioning to the text encoder (hidden_states[50]).
- [ ] Extend `_denoise_pika` for frozen anchors + noise_aug 0.999.
- [ ] FL2VA bring-up on device, one keyframe + prompt, check coherence to keyframe.
