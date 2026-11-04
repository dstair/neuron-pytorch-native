# Pika-H3-Flex on Trainium — resume plan / handoff

**Read this first if resuming with fresh context.** It captures the goal, the
corrections to initial assumptions, the environment, the hard-won lessons from
the MiniMax-H3 work, and a phased port plan. Branch: `pika-h3-flex`.
Prior project (do not modify): branch `minimax-h3`, `contrib/minimax-h3/`.

## 1. Goal

Port https://huggingface.co/Pika-Labs/Pika-H3-Flex to AWS Trainium (trn2),
starting with the simplest task (T2V or FL2VA) at the 6-NFE sampling contract,
with quality and speed measured against a reference. License: MiniMax H3
Community License (same as MiniMax-H3) — confirm the customer's use is covered.

## 1.5 Verified findings (2026-11-04) — port is much lighter than expected

**Every shared weight is bit-identical to MiniMax-H3 (already on the box).**
Hash-verified with ranged reads (no full downloads), `/data/omni29/pika_port/verify_out.txt`:

| Component | Result | Action |
|---|---|---|
| transformer (DiT, 33B) | 4/4 tensors MATCH | reuse `/data/models/MiniMax-H3/transformer` |
| text_encoder (Qwen3-VL-32B **incl. vision tower**) | 3/3 MATCH | reuse `/data/models/MiniMax-H3/text_encoder` (vision tower present — usable for FL2VA image conditioning) |
| vae | 4/4 MATCH | reuse `/data/models/MiniMax-H3/vae` |
| audio_vae | 3/3 MATCH | reuse `/data/models/MiniMax-H3/audio_vae` |

Only the **~2.8 GB LoRA adapter + small configs** were downloaded to
`/data/models/Pika-H3-Flex/` (scheduler, audio_scheduler, tokenizer, processor,
configs, manifests, README, LICENSE, SHA256SUMS). **No disk reclaim needed to start.**

**Adapter (`fl2va/adapter.safetensors`, sha256 verified 8ed6115d…):** standard PEFT
LoRA, 624 tensors = (50 DiT blocks + 2 refiner blocks) x 6 modules x (lora_A,lora_B).
Manifest `fl2va/manifest.json`: rank 128, alpha 128 (**scale = alpha/rank = 1.0**),
target modules `to_q, to_k, to_v, to_out.0, ff.net.0.proj, ff.net.2`, base
`MiniMaxAI/MiniMax-H3` rev 42ed227e, knots `[1000,750,500,250,100,40,0]`,
video_shift 12, audio_shift 3, method `renoise`, trunk `transformer`.
Merge formula per module: `W_merged = W_base + lora_B @ lora_A`.

**Disk constraint revised:** 30 GB free is enough because the DiT/encoder/VAE are
reused. An offline merged 33B checkpoint would NOT fit — so apply the LoRA **at
load time in the plugin** (per-rank from adapter slices, no merged checkpoint).

## 2. Corrections to the initial brief

- **The DiT is NOT smaller.** It is the same 50-layer / 33B MiniMax-H3 DiT
  (`MiniMaxH3Transformer3DModel`, hidden 5376, 56 heads, ffn 14336). It fits
  exactly like MiniMax-H3, already proven at TP4 / 768p on one trn2.3xlarge.
- **"Far fewer" = 6 sampling steps (6 NFE), not fewer layers** — knots
  `1000/750/500/250/100/40/0`, video shift 12, audio shift 3. ~8x fewer denoise
  forwards than the base 50-step run.
- **It is LoRA adapters on the MiniMax-H3 base** (`fl2va/adapter.safetensors`,
  `ref2va/adapter.safetensors`, rank 128 / fused-QKV 384, strength 1.0). Merge
  adapters into the TP-sharded weights on the host at load (no runtime LoRA cost).

See `docs/ARCHITECTURE.md` for the full component breakdown.

## 3. Environment (same box as the MiniMax-H3 work)

- Box: trn2.3xlarge `i-0498e6292f78f9774`, ap-southeast-4, TP4/CP1/LNC2.
  SSH helper: `deploy/box.sh run '<cmd>'` (needs the local key
  `~/.ssh/trn2-ap-southeast-4-20261001`). Do NOT stop the instance; check for
  active jobs first: `deploy/box.sh run 'pgrep -af "run.py|neuronx-cc"'`.
- Runtime env on box: `source /data/omni29/env29.sh` (venv `/data/omni29/venv`):
  vllm 0.29.0, vllm-neuron 0.29.0.1.2.0, vllm-omni 0.29.0rc1, Lite 2.13,
  neuronx-cc 2.28.7631 (SDK 2.33), **diffusers 0.40.0**, torch 2.13.
  NOTE: the Pika pipeline likely needs diffusers >= 0.42.dev or the Pika
  inference package — our diffusers is 0.40, so we must vendor/adapt the
  pipeline rather than import `PikaH3FlexModularPipeline` directly.
- Plugin (Neuron vLLM-Omni) on box: `/data/omni29/plugin` (private
  aws-neuron/private-vllm-omni-neuron-staging, base `a2614a2`, archive, no .git,
  plus our experiment patches). Backups under `/data/omni29/backups/`.
- Weights: MiniMax-H3 at `/data/models/MiniMax-H3` (135 GB, verified). **Pika-H3-Flex
  weights NOT yet downloaded.** **Disk is the immediate blocker: only ~30 GB free
  on `/data`** (2026-11-04) — reclaim before any download. Consumers: `/data/omni29/nvme`
  64 G (host mmap weights, in use), `/data/omni29/row_padding_study` 5.1 G (block
  captures, archivable), runs 231 M. The full Pika repo (2x 33B DiT + Qwen3-VL
  vision encoder + VAE + adapters) will NOT fit alongside MiniMax-H3 without
  reclaiming; download SELECTIVELY and/or free space first (see phase 1).
- Runner: `/data/omni29/run29.sh NAME CMD...` -> `/data/omni29/runs/NAME/run.log`
  ending `=== exit N`. Launch detached with `setsid nohup ... < /dev/null &`.

## 4. What already exists and is reusable

- **Neuron MiniMax-H3 DiT** (`vllm_omni_neuron/diffusion/models/minimax_h3/`):
  TP-sharded `MiniMaxH3Transformer3DModel` + `MiniMaxH3SwiGLU`, attention with
  QK RMSNorm + partial RoPE (rope_by_matmul), per-block compiled stages, host
  AdaLN. Proven: 768p/124f t2va dense ~755s/request (denoise ~640s).
- Single-device stage config: `examples/minimax_h3/minimax_h3_stage_single_device.yaml`
  (TP4, adaln_on_host, compile_per_block, rope_by_matmul, attention_heads_per_call=2,
  text_encoder_stream_layers=4, vae_swap_dit_blocks=25, vae_fused_vit,
  host_weights_dir=/mnt/nvme/minimax_h3).
- Video VAE (fused ViT decoder) and text encoder (Qwen3-VL text, streamed) ports.
- On the `minimax-h3` branch: benchmark harnesses (block replay, compare tools),
  CPU cache implementation, FP8 kernel + studies. Useful as patterns; not required.

## 5. Hard-won lessons (avoid re-learning these)

1. **HBM is 24 GB/core**; the single-device fit is tight. Measure with
   `neuron-monitor` (1-second SAMPLED runtime HBM — not an exact allocator peak;
   this Lite build has no supported peak counter). Other DiT blocks/encoders/VAE
   are not resident in isolated block tests.
2. **NCC_EXSP001 (110 GB estimate)**: a standalone small graph containing a TP
   `all_reduce` trips the compiler's static memory-pressure estimate. The SAME
   all_reduce compiles fine inside the full block. Keep TP reductions in-graph.
3. **No eager device collectives** in this Lite stack (`dist.all_reduce` and the
   GroupCoordinator API both fail eagerly on neuron tensors). Do device
   collectives in-graph, or CPU tensors over `tp_group.cpu_group` outside timing.
4. **Fence timing with an output probe** (`probe.cpu()` per call).
   `torch.neuron.synchronize()` alone does NOT drain the compiled Executor queue
   (caused dispatch-biased timings and `Execution Queue Full`).
5. **CPU-group `dist.all_reduce` works as a fence**; `dist.barrier()` probes an
   unavailable PrivateUse1 hook and fails — do not use it.
6. **LNC2 internal collectives need `NEURON_RT_ROOT_COMM_ID`** set before Lite
   init, even for a single logical core.
7. **FP8 findings**: FFN FP8 works (~1.48x FFN, ~9% denoise) but causes visible
   trajectory divergence end-to-end (SSIM 0.60, latent 0.75) — do NOT treat
   component microbenchmarks as quality evidence. QKV FP8 is ~3x slower in-block
   (unresolved). See `minimax-h3` branch `fp8/` for details before any FP8 reuse.
8. **Compile flags**: production block uses `--model-type=transformer
   --auto-cast=none -O2 --hbm-scratchpad-page-size=2048`. Small standalone graphs
   may need the scratchpad-page flag dropped. `-O1` deadlocks the DiT NEFF.
9. **Compile-cache footgun**: identical module content hits a cached
   compile/failure even after changing compiler args. Vary the graph or clear the
   cache to re-test.
10. **Detached runs**: put `source env29.sh` INSIDE the `setsid bash -c` command,
    and never `pkill -f <pattern>` that can match your own launch command string.
11. **Quality gates**: dense repeats are bit-identical, so any divergence is
    attributable to the change. Compare latents + encoded video (SSIM/PSNR) +
    decoded audio, and always look at a contact sheet before claiming quality.

## 6. Phased port plan

1. **Reclaim disk + weights + reference.** First free space on `/data` (archive
   `/data/omni29/row_padding_study` captures and stale run artifacts to S3/host,
   keeping only what current work needs). Then confirm whether Pika's `transformer/`
   is bit-identical to our existing MiniMax-H3 DiT — tensor naming is already
   confirmed IDENTICAL (638 tensors, layers 0-49, same names), so hash one tensor
   per shard group (e.g. `transformer_blocks.0.ff.net.2.weight`) on both sides.
   If identical, reuse `/data/models/MiniMax-H3/transformer` and download ONLY:
   `fl2va/adapter.safetensors` (start with FL2VA), `vae/`, `audio_vae/`,
   `scheduler/`, `audio_scheduler/`, `tokenizer/`, `processor/`, and the
   `text_encoder/` (Qwen3-VL vision — large; check if we can stream it like the
   MiniMax-H3 encoder). Defer `transformer_ref/` and `ref2va/` until FL2VA works.
   Verify with SHA256SUMS. Record the pinned revision (`release.json`).
2. **LoRA merge (host).** Write a script to apply `fl2va/adapter.safetensors`
   (rank 128/384, strength 1.0) into the TP-sharded DiT weights at load, or
   produce a merged checkpoint offline. Validate merged weights on CPU vs the
   reference formula before any device run.
3. **Pipeline (minimal T2V/FL2VA).** Adapt the existing
   `NeuronMiniMaxH3Pipeline`: (a) 6-step scheduler contract (knots/shifts);
   (b) FL2VA image conditioning inputs (first/last frame) — this is the main new
   work; (c) reuse the proven single-device stage config. Keep t2va working first
   (it shares the layout/scheduler), then add image conditioning.
4. **Text encoder.** The MiniMax-H3 port streams a text-only Qwen3-VL. Pika
   needs the full Qwen3-VL (vision tower) for image inputs. For T2V (text only),
   the existing encoder may suffice — start there; add vision only for FL2VA/ref2va.
5. **DiT bring-up.** Load merged DiT with the existing Neuron transformer at TP4,
   run 6-step denoise, compare against a CPU/reference 6-step trajectory.
6. **Quality + speed gates.** Dense baseline (CPU or GPU reference) vs Neuron:
   latents, encoded video (SSIM/PSNR + contact sheet), decoded audio. Measure
   warm request and denoise time. THEN consider optimizations (the 6-step model
   is already ~8x fewer forwards than base; do NOT apply FP8/caching until the
   baseline is validated).
7. **FL2VA, then Ref2VA** (with `transformer_ref` + vision encoder) as follow-ups.

## 7. First actions checklist

- [ ] Confirm customer use is covered by the MiniMax H3 Community License.
- [ ] `deploy/box.sh run 'df -h /data; du -sh /data/models/*'` — free space.
- [ ] Resolve pinned revision; download minimal set (phase 1) to `/data/models/Pika-H3-Flex`.
- [ ] Verify Pika `transformer/` vs `/data/models/MiniMax-H3/transformer`
      (hash a few tensors) to decide reuse vs download.
- [ ] Vendor/locate `PikaH3FlexModularPipeline` semantics (diffusers 0.42.dev or
      Pika inference repo) to mirror the 6-step contract and FL2VA conditioning.

## 8. Do NOT (from prior project's scars)

- Do not enable FP8 or cross-step caching until the 6-step baseline is validated;
  both need their own quality gates, and caching is pointless on a 6-step model
  (fewer steps than its warmup).
- Do not assume "fewer layers" — the DiT is the full 33B; plan memory accordingly.
- Do not treat component microbenchmarks or attention-output error as
  end-to-end quality evidence.
- Do not modify the `minimax-h3` branch or the unrelated dirty Qwen files in the
  working tree.

## 9. Progress log

### 2026-11-04 — weights verified + LoRA merged + sampling contract understood

- **All shared weights bit-identical to MiniMax-H3** (DiT, text encoder incl. vision,
  VAE, audio_vae) — reuse `/data/models/MiniMax-H3/*`, no 130 GB download. Only
  `fl2va/adapter.safetensors` (2.77 GB, sha256 verified) + small configs downloaded
  to `/data/models/Pika-H3-Flex/`.
- **LoRA-at-load implemented** (`patches/pika_lora.patch`, module
  `plugin_pika_lora.py`): merges rank-128 (scale 1.0) adapter into all 6 projections
  of 50 DiT + 2 refiner blocks, per-rank from adapter slices, fp32 deltas. Validated:
  CPU unit test (tp=1, 8 passed) and TP4 fused-QKV sharding orientation (rank r owns
  output rows [r*1792:(r+1)*1792], param cols = base[rows].T). Applied to box plugin;
  config `pika_lora_path` (default ""). Backup `/data/omni29/backups/pre_pika_lora_20261104/`.
- **Sampling contract (diffusers main `pika_h3_flex`, scheduler
  `scheduling_minimax_h3`):** the DiT/encoder/VAE/denoiser-loop structure is the SAME
  as MiniMax-H3; the differences are (1) LoRA and (2) the 6-NFE renoise schedule:
  - base grid `[1.0, 0.75, 0.5, 0.25, 0.1, 0.04, 0.0]` (knots/1000), pushed through
    shift via `sigmas = shift*base/(1+(shift-1)*base)` with shift=12 (video) / 3
    (audio) because Pika's scheduler configs set `shift_provided_sigmas: true`.
    Effective video sigmas ≈ [1, .973, .923, .8, .571, .333, 0], audio ≈
    [1, .9, .75, .5, .25, .111, 0]. DiT conditions on `t = 1 - sigma` per modality
    (our pipeline already builds dual video/audio timesteps).
  - `stochastic_sampling: true` (renoise) per step: `x0 = latents + sigma_i * v`
    (fp32, data-ward velocity), then `latents = (1-sigma_{i+1})*x0 + sigma_{i+1}*fresh_noise(generator)`.
    Last step (sigma_next=0) returns x0. Conditioning/anchor rows are never overwritten.
  - **Installed diffusers 0.40 does NOT implement this**: its `set_timesteps(sigmas=)`
    uses sigmas verbatim (no shift) and its `step()` is Euler (no generator/renoise).
    So compute the shifted grid AND the renoise step MANUALLY in the pipeline
    (~20 lines host-side); do not rely on the installed scheduler's step.
- **Pipeline source:** diffusers main `src/diffusers/modular_pipelines/pika_h3_flex/`
  (~4750 lines) is the reference; T2V reduces to our existing t2va loop + the above.
- Next: implement Pika renoise sampling in `NeuronMiniMaxH3Pipeline` (config-gated),
  stage yaml with `pika_lora_path` + 6-step contract, T2V bring-up on device.

### 2026-11-04 (later) — T2V bring-up WORKS on device

- **Pika T2V runs end-to-end on Trainium** (`/data/omni29/runs/pika_t2v_20261104`):
  LoRA merged on all 4 ranks (52 modules each), 6-step renoise, seed 42, 768p/124f.
  Warm request 2: text encode 83.2s, **denoise 181.0s**, VAE 21.0s, total 285.5s, exit 0.
  Output video is a coherent, high-quality parrot clip matching the prompt
  (`results/pika_t2v_contact_sheet.png`). Latents finite, repeat deterministic.
- **Steady-state DiT = 12.6s/step** (matches MiniMax-H3). Step 0 carries ~105s
  one-time per-process warmup (first block_modulations / DiT executor), not a
  per-step cost. Sigmas match the reference grid exactly
  (`results/t2v_warm_summary.json`).
- **Disk fix:** the Pika text-encoder host-weight map tried to write ~50 GB of NEW
  maps (tagged by model basename). Since the encoder is bit-identical to MiniMax-H3,
  **hardlinked the existing MiniMax-H3 maps (text encoder, VAE, DiT swap blocks)
  under the Pika tag** — reused ~67 GB with zero extra space. The failed run left
  ~49 GB of `.tmp` partial writes; deleted. Model dir built at
  `/data/models/Pika-H3-Flex-Neuron` (symlinks shared weights, keeps Pika
  scheduler/audio_scheduler/tokenizer/processor/fl2va).
- Code: `patches/pika_lora.patch` (LoRA-at-load), `patches/pika_renoise_sampling.patch`
  (6-NFE renoise in the pipeline, `_denoise_pika`), `deploy/pika_h3_flex_stage.yaml`.
  Unit tests: 10 passed on the live box (sigma grid, renoise formula, LoRA merge).

**Open follow-ups:** (1) investigate/eliminate the ~105s step-0 warmup (likely
first block_modulations or DiT executor load per process); (2) FL2VA keyframe
conditioning (needs the vision tower + anchor rows — encoder has the vision
weights); (3) Ref2VA (transformer_ref + vision encoder); (4) stronger quality gate
vs the official Pika runtime if available.
