# Pika-H3-Flex FL2VA Implementation Status

**Last updated: 2026-11-04 (session continuation)**

This document tracks the FL2VA (first/last-frame image-to-video) implementation progress.
Prior session: T2V working end-to-end (285s, 12.6s/step).

## Current Status: FL2VA Implementation In Progress

### Completed

1. **Reference code analyzed** - `contrib/pika-h3-flex/reference_pipeline/` contains the diffusers main reference
2. **FL2VA layout logic tested** - Position ids, token tags, and anchor placement verified
3. **Patch created** - `patches/pika_fl2va.patch` ready to apply
4. **Test runner created** - `run_fl2va.py` for on-box testing
5. **Stage config created** - `deploy/pika_h3_flex_fl2va_stage.yaml` with VAE encoder enabled
6. **Integration guide** - `docs/FL2VA_INTEGRATION.md` documents the changes

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

1. **Deploy the patch**:
   ```bash
   cd /data/omni29/plugin
   patch -p1 < /path/to/pika_fl2va.patch
   ```

2. **Test VAE encoder compilation**:
   - Verify encoder graph compiles
   - Test keyframe encoding with dummy image

3. **Test FL2VA end-to-end**:
   - Create test keyframe image
   - Run `run_fl2va.py` with first-frame anchor
   - Verify output video starts on keyframe

4. **Add vision tower encoding** (future):
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
