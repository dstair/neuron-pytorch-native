# FL2VA Integration Guide

This document describes how to apply FL2VA support to the Neuron MiniMax-H3 pipeline.

## Files to Deploy

1. `patches/pika_fl2va.patch` - Main patch to `pipeline_minimax_h3.py`
2. `pipeline_fl2va.py` - Standalone FL2VA utilities (optional, for reference)
3. `run_fl2va.py` - Test runner script
4. `deploy/pika_h3_flex_fl2va_stage.yaml` - Stage config with VAE encoder

## Apply Patch

On the box:

```bash
cd /data/omni29/plugin

# Backup current pipeline
cp vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py \
   vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py.bak

# Apply FL2VA patch
patch -p1 < /path/to/pika_fl2va.patch
```

## Manual Changes (if patch doesn't apply cleanly)

The patch adds:
1. `import numpy as np` at top
2. `import MINIMAX_H3_TEXT_TAG` in diffusers imports
3. FL2VA constants after `MAX_DURATION_S`
4. `self._compile_vae_encoder` and `self._num_condition_video_rows` in `__init__`
5. VAE encoder compilation in `load_weights()`
6. New methods:
   - `_encode_vae_condition()`
   - `_spatial_position_grid()`
   - `_temporal_position_grid()`
   - `_frame_position_grid()`
   - `_build_fl2va_layout()`
   - `_prepare_keyframe_latents()`
   - `_noise_keyframe_latents()`
7. Modified `_denoise_pika()` to handle frozen anchors
8. Modified `forward()` to handle FL2VA requests

## Key Changes

### 1. VAE Encoder Compilation

The VAE encoder must be compiled for keyframe encoding:

```yaml
model_config:
  compile_vae_encoder: true
```

### 2. Frozen Anchor Handling

In `_denoise_pika()`, generated rows are updated but condition rows are frozen:

```python
if self._num_condition_video_rows > 0:
    gen_slice = slice(self._num_condition_video_rows, None)
    video_rows[gen_slice] = (1.0 - sig_v[i + 1]) * x0_v[gen_slice] + sig_v[i + 1] * noise_v[gen_slice]
else:
    video_rows = (1.0 - sig_v[i + 1]) * x0_v + sig_v[i + 1] * noise_v
```

### 3. Request Format

FL2VA requests pass keyframes via `multi_modal_data`:

```python
request = {
    "prompt": "...",
    "multi_modal_data": {
        "keyframes": [PIL.Image, ...],  # first-frame, optionally last-frame
    },
}
```

### 4. Keyframe Encoding

Keyframes are:
1. Resized to target canvas (PIL LANCZOS)
2. ImageNet normalized
3. VAE encoded (spatial encoder alone for single frame)
4. Sampled under fixed seed=42
5. Rounded to float16
6. Noised to t=0.999

## Testing

```bash
# Create a test keyframe (or use an existing image)
convert -size 1344x768 xc:blue test_keyframe.png

# Run FL2VA
python /data/omni29/plugin/contrib/pika-h3-flex/run_fl2va.py \
    --stage-config /data/omni29/pika_h3_flex_fl2va_stage.yaml \
    --first-frame test_keyframe.png \
    --prompt "A blue sky with clouds" \
    --output fl2va_test.mp4
```

## Known Limitations

1. **Vision Tower**: The current implementation does NOT encode keyframes through Qwen3-VL's
   vision tower for text conditioning. This means the text encoder doesn't see the keyframe
   image, which may produce suboptimal results. Full FL2VA requires:
   - Extending `NeuronQwen3VLTextEncoder` to handle vision inputs
   - Building the keyframe presentation: "<Picture i>: " + vision block
   - This is a significant extension; the current implementation works for basic keyframe
     conditioning but may not match the full reference quality

2. **Generator Order**: Keyframe noise is drawn from the request's generator BEFORE the
   generated video/audio noise - this order must match for reproducibility

## Next Steps

1. Test VAE encoder compilation separately
2. Test keyframe encoding (VAE encode only)
3. Test full FL2VA pipeline
4. Add vision tower encoding for keyframes
