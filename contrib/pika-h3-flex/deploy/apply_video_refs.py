#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Ref2VA video references (motion/camera reference, optionally with its soundtrack).

Per diffusers main's ref2va blocks:
- **Normalize** (`_normalize_video_reference`): uint8 THWC, resampled to 24 fps by
  drop/duplicate (`slots = floor(arange(n) * 24/fps + 0.5)`, repeated by diff), truncated
  to the generated frame count, resized to the canvas of its OWN aspect ratio
  (short edge 768, area-capped, 32-multiple, LANCZOS).
- **Encode** (`_encode_video_reference`): snap DOWN to `17n+5`, encode per 17-frame clip
  through the upstream encoder modules, concat moments, drop `token_drop` trailing
  latents, seed-42 posterior sample, fp16 rounding, bf16 policy.
- **Layout**: a video reference's soundtrack rows pack immediately BEFORE its video rows
  and share their rotary origin; the clock advances by
  `max(audio_latents, sequential_frame_span)` — the sequential sum, not the pairwise one
  (they differ in the last ulp from 16 latent frames; the reference keeps both).
- **Presentation**: `"<Video k>: "` label; a soundtracked video adds `"<Audio j>: "`
  BEFORE it. Vision-tower video blocks are NOT supported by the Neuron encoder
  (documented limitation; image vision blocks work).

Usage:
    python apply_video_refs.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys

VIDEO_METHODS = '''
    def _normalize_video_reference(self, frames, fps: float, num_frames: int):
        """Resample/truncate/resize a video reference per the released recipe."""
        from PIL import Image

        frames = np.asarray(frames)
        if frames.dtype != np.uint8:
            frames = (frames * 255.0).round().clip(0, 255).astype(np.uint8)
        if frames.ndim != 4 or frames.shape[3] != 3:
            raise ValueError(f"A reference video must be (num_frames, h, w, 3), got {tuple(frames.shape)}.")
        if fps <= 0:
            raise ValueError(f"A reference video must have a positive frame rate, got {fps}.")
        if fps != MINIMAX_H3_FPS:
            scale = MINIMAX_H3_FPS / fps
            slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
            frames = np.repeat(frames, np.diff(slots, append=int(np.floor(frames.shape[0] * scale + 0.5))), axis=0)
        frames = frames[:num_frames]
        # The canvas of its own aspect ratio, same rule as the target (short edge 768, area-capped).
        src_h, src_w = frames.shape[1], frames.shape[2]
        ratio = src_w / src_h
        if ratio >= 1.0:
            width, height = 768.0 * ratio, 768.0
        else:
            width, height = 768.0, 768.0 / ratio
        area = width * height
        max_pixels = 768 * 1344
        if area > max_pixels:
            scale = (max_pixels / area) ** 0.5
            width, height = width * scale, height * scale
        multiple = VAE_SPATIAL_COMPRESSION * 2
        target_h = max(multiple, round(height / multiple) * multiple)
        target_w = max(multiple, round(width / multiple) * multiple)
        if (target_h, target_w) != (src_h, src_w):
            frames = np.stack(
                [
                    np.asarray(
                        Image.fromarray(frame).resize((target_w, target_h), Image.Resampling.LANCZOS)
                    )
                    for frame in frames
                ]
            )
        # A whole VAE chunk after the first is the minimum for a video reference.
        if frames.shape[0] < VAE_FRAMES_PER_CHUNK + VAE_LATENTS_PER_CHUNK:
            raise ValueError(
                f"A video reference must hold at least {VAE_FRAMES_PER_CHUNK + VAE_LATENTS_PER_CHUNK} "
                f"frames at {MINIMAX_H3_FPS} fps once resampled, got {frames.shape[0]}."
            )
        return frames

    def _encode_video_reference(self, frames: np.ndarray) -> torch.Tensor:
        """VAE-encode normalized reference frames to conditioning latents.

        Multi-frame variant of ``_encode_vae_condition``: snap down to ``17n+5``, encode
        per 17-frame clip (upstream encoder modules directly, so no distributed/device
        dispatch), concat moments, drop ``token_drop`` trailing latent frames.
        """
        vae = self.vae
        latents = None
        if vae is not None:
            num_frames = (
                max(1, (frames.shape[0] - VAE_LATENTS_PER_CHUNK) // VAE_FRAMES_PER_CHUNK) * VAE_FRAMES_PER_CHUNK
                + VAE_LATENTS_PER_CHUNK
            )
            pixels = torch.from_numpy(frames[:num_frames].copy()).permute(3, 0, 1, 2)[None]
            pixel_mean = torch.tensor(PIXEL_MEAN).view(1, -1, 1, 1, 1)
            pixel_std = torch.tensor(PIXEL_STD).view(1, -1, 1, 1, 1)
            pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
            clip = VAE_FRAMES_PER_CHUNK
            ranks = dist.get_world_size() if dist.is_initialized() else 1
            previous = torch.get_num_threads()
            torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) // max(1, ranks // 4)))
            try:
                moments = torch.cat(
                    [
                        vae.quant_conv(vae.encoder(pixels[:, :, i * clip : (i + 1) * clip]))
                        for i in range(pixels.shape[2] // clip)
                    ],
                    dim=2,
                )
            finally:
                torch.set_num_threads(previous)
            token_drop = int(getattr(vae.config, "token_drop", 0))
            if token_drop > 0:
                moments = moments[:, :, :-token_drop]
            from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

            posterior = DiagonalGaussianDistribution(moments)
            latents = posterior.sample(generator=torch.Generator().manual_seed(KEYFRAME_ENCODE_SEED))
            latents = latents.to(torch.float16).float().cpu()
            latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
            latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
            latents = (latents - latents_mean) / latents_std
        return self._conditioning_broadcast(latents)

'''

OLD_LAYOUT_LOOP = '''        for kind in self._ref2va_kinds:
            if kind == "image":
                num_latent_frames_, ref_h, ref_w = next(visual_geometry)
                num_video_rows = num_latent_frames_ * (ref_h // patch_h) * (ref_w // patch_w)
                rows = slice(cursor, cursor + num_video_rows)
                cursor = rows.stop
                video_indices.append(torch.arange(rows.start, rows.stop))
                frame_grid, _ = self._frame_position_grid(ref_h, ref_w, patch_h, patch_w)
                position_ids[rows, 0] = rotary_time
                position_ids[rows, 1:] = frame_grid
                # An image is a single frame and takes a single integer rotary slot.
                rotary_time += 1.0
            elif kind == "audio":
                num_audio_rows = next(audio_row_counts)
                reference_audio_latents_ = num_audio_rows // audio_channels
                rows = slice(cursor, cursor + num_audio_rows)
                cursor = rows.stop
                audio_indices.append(torch.arange(rows.start, rows.stop))
                self._fill_audio_positions(
                    position_ids, rows, reference_audio_latents_, rotary_time, target_width_grid, audio_channels
                )
                rotary_time += float(reference_audio_latents_)
            else:
                raise ValueError(f"ref2va references must be 'image' or 'audio', got {kind!r}")
'''

NEW_LAYOUT_LOOP = '''        video_has_audio = iter(self._ref2va_video_has_audio)
        for kind in self._ref2va_kinds:
            if kind == "image":
                num_latent_frames_, ref_h, ref_w = next(visual_geometry)
                num_video_rows = num_latent_frames_ * (ref_h // patch_h) * (ref_w // patch_w)
                rows = slice(cursor, cursor + num_video_rows)
                cursor = rows.stop
                video_indices.append(torch.arange(rows.start, rows.stop))
                frame_grid, _ = self._frame_position_grid(ref_h, ref_w, patch_h, patch_w)
                position_ids[rows, 0] = rotary_time
                position_ids[rows, 1:] = frame_grid
                # An image is a single frame and takes a single integer rotary slot.
                rotary_time += 1.0
            elif kind == "audio":
                num_audio_rows = next(audio_row_counts)
                reference_audio_latents_ = num_audio_rows // audio_channels
                rows = slice(cursor, cursor + num_audio_rows)
                cursor = rows.stop
                audio_indices.append(torch.arange(rows.start, rows.stop))
                self._fill_audio_positions(
                    position_ids, rows, reference_audio_latents_, rotary_time, target_width_grid, audio_channels
                )
                rotary_time += float(reference_audio_latents_)
            elif kind == "video":
                # A video reference's soundtrack rows pack immediately before its video
                # rows and share their rotary origin.
                num_audio_rows = next(audio_row_counts) if next(video_has_audio) else 0
                reference_audio_latents_ = num_audio_rows // audio_channels
                num_latent_frames_, ref_h, ref_w = next(visual_geometry)
                num_video_rows = num_latent_frames_ * (ref_h // patch_h) * (ref_w // patch_w)
                audio_rows = slice(cursor, cursor + num_audio_rows)
                video_rows = slice(audio_rows.stop, audio_rows.stop + num_video_rows)
                cursor = video_rows.stop
                if num_audio_rows:
                    audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
                video_indices.append(torch.arange(video_rows.start, video_rows.stop))
                frame_grid, width_grid = self._frame_position_grid(ref_h, ref_w, patch_h, patch_w)
                if num_audio_rows:
                    self._fill_audio_positions(
                        position_ids, audio_rows, reference_audio_latents_, rotary_time, width_grid, audio_channels
                    )
                frame_time = self._temporal_position_grid(num_latent_frames_, rotary_time)
                position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
                position_ids[video_rows, 1:] = frame_grid.repeat(num_latent_frames_, 1)
                # Sequential float64 sum of the frame spans (NOT the pairwise sum).
                video_span = sum(
                    _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
                    for i in range(num_latent_frames_)
                )
                rotary_time += max(float(reference_audio_latents_), video_span)
            else:
                raise ValueError(f"ref2va references must be 'image', 'audio' or 'video', got {kind!r}")
'''

OLD_REF_LOOP = '''                elif entry["kind"] == "audio":
                    max_samples = round(num_frames / MINIMAX_H3_FPS * 32000)
                    audio_reference_latents.append(
                        self._encode_audio_reference(
                            entry["audio"], int(entry.get("sample_rate") or 32000), max_samples
                        )
                    )
                else:
                    raise ValueError(f"ref2va references must be 'image' or 'audio', got {entry['kind']!r}")
'''

NEW_REF_LOOP = '''                elif entry["kind"] == "audio":
                    max_samples = round(num_frames / MINIMAX_H3_FPS * 32000)
                    audio_reference_latents.append(
                        self._encode_audio_reference(
                            entry["audio"], int(entry.get("sample_rate") or 32000), max_samples
                        )
                    )
                elif entry["kind"] == "video":
                    frames = self._normalize_video_reference(
                        entry["frames"], float(entry.get("fps") or MINIMAX_H3_FPS), num_frames
                    )
                    latents = self._encode_video_reference(frames)
                    reference_latents.append(latents.to(torch.bfloat16).to(torch.float32))
                    if entry.get("audio") is not None:
                        max_samples = round(num_frames / MINIMAX_H3_FPS * 32000)
                        audio_reference_latents.append(
                            self._encode_audio_reference(
                                entry["audio"], int(entry.get("sample_rate") or 32000), max_samples
                            )
                        )
                        self._ref2va_video_has_audio.append(True)
                    else:
                        self._ref2va_video_has_audio.append(False)
                else:
                    raise ValueError(f"ref2va references must be 'image', 'audio' or 'video', got {entry['kind']!r}")
'''

OLD_KINDS = '''            self._ref2va_kinds = [entry["kind"] for entry in references]
            reference_latents = []
            audio_reference_latents = []
'''

NEW_KINDS = '''            self._ref2va_kinds = [entry["kind"] for entry in references]
            self._ref2va_video_has_audio = []
            reference_latents = []
            audio_reference_latents = []
'''

OLD_PRESENTATION = '''        elif references:
            ref_images = [entry["image"] for entry in references if entry["kind"] == "image"]
            ref_audio_count = sum(1 for entry in references if entry["kind"] == "audio")
            prompt_embeds, text_token_tags = self._encode_prompt_with_vision(
                prompt, ref_images, audio_count=ref_audio_count
            )
'''

NEW_PRESENTATION = '''        elif references:
            ref_images = [entry["image"] for entry in references if entry["kind"] == "image"]
            presentation = [
                "video+audio" if entry["kind"] == "video" and entry.get("audio") is not None else entry["kind"]
                for entry in references
            ]
            prompt_embeds, text_token_tags = self._encode_prompt_with_vision(
                prompt, ref_images, presentation=presentation
            )
'''

OLD_VISION_LABELS = '''        token_ids, token_tags = [], []
        merge = processor.image_processor.merge_size**2
        for index, _ in enumerate(audio_count and range(audio_count) or []):
            label_ids = tokenizer(f"<Audio {index + 1}>: ", add_special_tokens=False)["input_ids"]
            token_ids += label_ids
            token_tags += [text_tag] * len(label_ids)
        for index in range(len(images)):
            num_image_tokens = int(vision_inputs["image_grid_thw"][index].prod()) // merge
            label_ids = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
            vision_ids = (
                [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            token_ids += label_ids + vision_ids
            token_tags += [text_tag] * len(label_ids) + [video_tag] * len(vision_ids)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [text_tag] * len(prompt_ids)
'''

NEW_VISION_LABELS = '''        token_ids, token_tags = [], []
        merge = processor.image_processor.merge_size**2
        if presentation is None:
            presentation = ["image"] * len(images)
        # Labels follow the references' packed order; a soundtracked video's audio label
        # precedes its video label. Vision blocks attach to the image entries in order.
        counts = {"image": 0, "audio": 0, "video": 0}
        image_index = 0
        num_image_tokens_list = (
            [int(grid.prod()) // merge for grid in vision_inputs["image_grid_thw"]] if images else []
        )

        def emit_text(value):
            ids = tokenizer(value, add_special_tokens=False)["input_ids"]
            token_ids.extend(ids)
            token_tags.extend([text_tag] * len(ids))

        for entry_kind in presentation:
            if entry_kind == "image":
                counts["image"] += 1
                emit_text(f"<Picture {counts['image']}>: ")
                vision_ids = (
                    [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                    + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens_list[image_index]
                    + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                )
                token_ids.extend(vision_ids)
                token_tags.extend([video_tag] * len(vision_ids))
                image_index += 1
            elif entry_kind == "audio":
                counts["audio"] += 1
                emit_text(f"<Audio {counts['audio']}>: ")
            elif entry_kind == "video":
                counts["video"] += 1
                emit_text(f"<Video {counts['video']}>: ")
            elif entry_kind == "video+audio":
                counts["audio"] += 1
                emit_text(f"<Audio {counts['audio']}>: ")
                counts["video"] += 1
                emit_text(f"<Video {counts['video']}>: ")
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [text_tag] * len(prompt_ids)
'''

OLD_VISION_SIG = "    def _encode_prompt_with_vision(self, prompt: str, images: list, audio_count: int = 0):"
NEW_VISION_SIG = "    def _encode_prompt_with_vision(self, prompt: str, images: list, audio_count: int = 0, presentation: list | None = None):"


def apply(src: str) -> str:
    if "_encode_video_reference" in src:
        raise SystemExit("video refs already applied")

    anchor = "    def _encode_vae_condition(self, pixels: torch.Tensor) -> torch.Tensor:"
    assert anchor in src, "method anchor not found"
    src = src.replace(anchor, VIDEO_METHODS + anchor, 1)

    assert OLD_LAYOUT_LOOP in src, "layout loop anchor not found"
    src = src.replace(OLD_LAYOUT_LOOP, NEW_LAYOUT_LOOP, 1)
    assert OLD_REF_LOOP in src, "ref loop anchor not found"
    src = src.replace(OLD_REF_LOOP, NEW_REF_LOOP, 1)
    assert OLD_KINDS in src, "kinds anchor not found"
    src = src.replace(OLD_KINDS, NEW_KINDS, 1)
    assert OLD_PRESENTATION in src, "presentation anchor not found"
    src = src.replace(OLD_PRESENTATION, NEW_PRESENTATION, 1)
    assert OLD_VISION_SIG in src, "vision sig anchor not found"
    src = src.replace(OLD_VISION_SIG, NEW_VISION_SIG, 1)
    assert OLD_VISION_LABELS in src, "vision labels anchor not found"
    src = src.replace(OLD_VISION_LABELS, NEW_VISION_LABELS, 1)
    return src


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    out = apply(src)
    with open(path, "w") as f:
        f.write(out)
    print(f"Applied video refs to {path}")


if __name__ == "__main__":
    main()
