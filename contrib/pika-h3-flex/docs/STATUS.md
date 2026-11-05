# Pika-H3-Flex on Trainium — Status (2026-11-05)

Production model port: Pika H3 Flex (LoRA adapters on MiniMax H3) on AWS Trainium,
via the Neuron vLLM-Omni plugin. Branch `pika-h3-flex`.

## Customer asks vs status

| Ask | Status | Evidence |
|---|---|---|
| LoRA loading on base H3 weights | **DONE** | `patches/pika_lora.patch`, TP-sharded merge at load, both adapters |
| 1. first-frame-to-video-and-audio | **DONE** | FL2VA first-frame verified; frame 0 anchors (PSNR 35.5 dB); 32 kHz stereo WAV decoded |
| 2. ref2va (image references) | **DONE** | ref2va layout + adapter; subject carries through; `run_ref2va.py` |
| 3. first-frame-and-audio (lip-sync) | **DONE** | audio reference rows (clean t=1.0); generated audio follows the track (envelope corr 0.920) |
| 1344×768 | **DONE** | primary canvas |
| 768×1344 | **DONE** | portrait verified |
| up to 15 s @ 24 fps | **DONE** | 362 frames + 15.07s stereo WAV at TP=8×CP=8, warm 144s e2e |
| steps 2, 3, 6 | **DONE** | per-request `num_inference_steps`; knots subsample the trained grid; 6/3 verified |
| fast inference | **TP=8: 8.9 s/step warm** | see numbers below |

## Measured numbers (trn2.48xlarge TP=8, 1344×768, 124 frames, warm repeats)

| Config | text enc | denoise | VAE dec | total warm |
|---|---|---|---|---|
| T2V 6-step + audio WAV | 2.6 s | 66.3 s (9.0/step) | 13.1 s | **82.4 s** |
| T2V 3-step + audio WAV | 5.7 s | ~35 s* | ~13 s | ~55 s |
| T2V 2-step + audio WAV | 5.8 s | ~25 s* | ~13 s | ~45 s |
| FL2VA 6-step + keyframe + audio | 2.9 s | 63.8 s | ~13 s | **83.5 s** (keyframe encode 3.5 s warm) |
| T2V 6-step TP=4 (3xl, reference) | 4.3 s | 221 s | 20.7 s | 246 s |

*3/2-step warm estimates from the 6-step per-step rate (first measured runs carried
compile remnants). 2-step output quality verified coherent. Keyframe device encode:
3.5 s warm / ~10 s first call (was ~120 s on CPU). Audio decode adds ~1 s per request.
Portrait 768×1344 works at the same per-step rate.

15 s (362 frames) at **TP=8×CP=8** (`pika_h3_flex_stage_tp8cp8.yaml`, all 64 cores):
warm e2e 144s (text 8.7s, denoise ~40s at 6.4-6.5s/step, VAE decode ~90-100s,
+15.07s stereo WAV). First request per engine pays a one-time ~225s step-0 cost
(AdaLN/neff warmup). Two hard limits shaped this: the DiT block graph OOMs past
~60k rows/rank (34.6 GB at 108k), and attention_cte's S_q×S_k bf16 tile overflows
uint32 past ~46k×46k — CP=8 (13.5k rows/rank) clears both. The NKI ring-attention
kernel doesn't survive dynamo tracing, so CP uses the all-gather fallback
(`apply_cp_fallback.py`). Host RAM at 64 workers peaks ~1.3 TB of 2 TB.

## Deployed boxes

- **trn2.48xlarge** `i-0a981ba5acfbaef04` (ap-south-2, capacity reservation): the main box.
  SSH helper `deploy/box48.sh`. TP=8 configs in `/data/omni29/pika_h3_flex_*_tp8*.yaml`.
- **trn2.3xlarge** `i-0498e6292f78f9774` (ap-southeast-4): original TP=4 box, `deploy/box.sh`.

## Repo layout (branch pika-h3-flex)

- `deploy/apply_*.py` — the pipeline patch series (apply in this order on a fresh plugin):
  `apply_fl2va.py` → `apply_audio_decode.py` → `apply_step_count.py` → `apply_ref2va.py` →
  `apply_device_vae_encode.py` → `apply_devenc_fix.py` → `apply_ref_short_edge.py` →
  `apply_encode_fallback.py` → `apply_encode_broadcast.py` → `apply_encode_paths.py`
- `deploy/pika_h3_flex_*stage*.yaml` — stage configs (TP=4 3xl, TP=8, TP=8+enc, TP=8×CP=2, ref2va)
- `run_fl2va.py`, `run_ref2va.py` — runners (T2V works with the stock `examples/minimax_h3/run.py`)
- `test_fl2va.py`, `test_fl2va_patched.py` — unit tests (layout math, frozen anchors, T2V equivalence)
- `docs/` — this file, NEXT_SESSION_PLAN (resume log), ARCHITECTURE, FL2VA_INTEGRATION
- `patches/` — original LoRA + renoise patches; `plugin_pika_lora.py`

## Request formats (pipeline reads `multi_modal_data`)

```python
# FL2VA (fl2va adapter stage config):
{"prompt": ..., "multi_modal_data": {"keyframes": [PIL.Image]}}           # first-frame
{"prompt": ..., "multi_modal_data": {"keyframes": [first_img, last_img]}} # first+last

# Ref2VA (ref2va adapter stage config):
{"prompt": ..., "multi_modal_data": {"references": [
    {"kind": "image", "image": PIL.Image},
    {"kind": "audio", "audio": (channels, samples) float tensor, "sample_rate": int},
]}}
```

Outputs: video frames via the engine result; the soundtrack as
`$MINIMAX_H3_OUT/request_NNN.wav` (32 kHz stereo).

## Known limitations / follow-ups

1. **Vision-tower conditioning not ported** — the text presentation is prompt-only; the
   reference adds `"<Picture i>: "` + Qwen3-VL vision blocks per keyframe/reference
   (hidden_states[50], vision rows tagged 0). The encoder weights are on the box; the
   Neuron text encoder is text-only today. Main quality lever remaining.
2. **Reference images > ~1.4 Mpx encode on CPU** (upstream tiled path, ~1-2 min): the
   device encode inside the swap overflows the Lite executor queue at larger tile counts.
   `reference_short_edge` defaults to the released 2048; set 768/1344 to stay on device.
3. **2-step quality unverified** (knots [1000,250,0]); 15s quality unverified beyond a
   successful run; FL2VA at 15s needs the CP=2 config too.
4. Video references (ref2va) not ported (image + audio references only).
5. The engine leaves workers behind if the client dies mid-compile — check
   `pgrep -af "run.py|neuronx-cc"` before relaunching.
