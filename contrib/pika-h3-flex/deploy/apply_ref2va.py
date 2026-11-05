#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add Ref2VA (multimodal reference-to-video) support to pipeline_minimax_h3.py.

Covers the customer's two reference modes:
- **Image references** (subject/style/scene): normalized to the tier's reference short
  edge (2048 at the 768 tier, 32-multiple rounded), VAE-encoded at that resolution,
  noised to t=0.999, packed as leading video rows on the shared rotary clock (+1.0 per
  image).
- **Audio references** (lip-sync track): resampled to 32 kHz, upmixed to stereo,
  truncated to the video duration, encoded through the audio VAE (posterior *mode*,
  clean), packed as leading audio rows pinned at t=1.0.

The layout matches diffusers main ``PikaH3FlexRef2VAPrepareLayoutStep.build_ref2va_packed_sequence``
for image and audio references (video references are a follow-up). The ref2va LoRA
adapter loads via the existing ``pika_lora_path`` (point it at ``ref2va/adapter.safetensors``).

Request shape:
    {"prompt": ..., "multi_modal_data": {"references": [
        {"kind": "image", "image": PIL.Image},
        {"kind": "audio", "audio": (channels, samples) tensor, "sample_rate": int},
    ]}}

Vision-tower presentation ("<Picture i>: " blocks) is NOT included — same documented
quality gap as FL2VA.

Usage:
    python apply_ref2va.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys

REF2VA_METHODS = '''
    @staticmethod
    def _fill_audio_positions(position_ids, rows, num_audio_latents, rotary_time, width_grid, audio_channels):
        """One channel-major audio block on the shared rotary clock (reference's helper)."""
        time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
        position_ids[rows, 0] = time.repeat(audio_channels)
        position_ids[rows, 2] = torch.cat(
            [
                torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
                torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
            ]
        )

    def _build_ref2va_layout(
        self,
        text_token_tags: torch.Tensor,
        reference_geometries: list,
        audio_reference_latents: list,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        patch_size: tuple,
    ):
        """The ``[text | reference blocks | target audio | target video]`` ref2va layout.

        Matches diffusers main ``build_ref2va_packed_sequence`` for image and audio
        references. ``reference_geometries``: per visual reference ``(frames, h, w)`` of
        its encoded latents (an image reference is one latent frame at its normalized
        resolution). ``audio_reference_latents``: per audio-bearing reference the
        ``(rows, C)`` packed tensor (its row count drives the block geometry).

        Returns the layout tuple plus ``(num_reference_video_rows, num_reference_audio_rows)``.
        """
        _, patch_h, patch_w = patch_size
        audio_channels = MINIMAX_H3_AUDIO_CHANNELS
        num_text_tokens = text_token_tags.shape[0]
        num_target_video_rows = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
        num_target_audio_rows = num_audio_latents * audio_channels

        num_reference_video_rows = sum(
            frames * (h // patch_h) * (w // patch_w) for frames, h, w in reference_geometries
        )
        num_reference_audio_rows = sum(rows.shape[0] for rows in audio_reference_latents)
        sequence_length = (
            num_text_tokens
            + num_reference_video_rows
            + num_reference_audio_rows
            + num_target_audio_rows
            + num_target_video_rows
        )

        position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
        position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
        target_frame_grid, target_width_grid = self._frame_position_grid(latent_height, latent_width, patch_h, patch_w)

        video_indices, audio_indices = [], []
        cursor = num_text_tokens
        rotary_time = float(num_text_tokens)
        visual_geometry = iter(reference_geometries)
        audio_row_counts = iter(rows.shape[0] for rows in audio_reference_latents)
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
            else:
                raise ValueError(f"ref2va references must be 'image' or 'audio', got {kind!r}")

        audio_start = cursor
        video_start = audio_start + num_target_audio_rows
        self._fill_audio_positions(
            position_ids, slice(audio_start, video_start), num_audio_latents, rotary_time, target_width_grid, audio_channels
        )
        frame_time = self._temporal_position_grid(num_latent_frames, rotary_time)
        position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
        position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

        video_indices = torch.cat(video_indices + [torch.arange(video_start, sequence_length)])
        audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
        text_indices = torch.arange(num_text_tokens)

        token_tags = torch.empty(sequence_length, dtype=torch.long)
        token_tags[text_indices] = text_token_tags.to(torch.long)
        token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
        token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

        return (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            num_reference_video_rows,
            num_reference_audio_rows,
        )

    def _normalize_image_reference(self, image, multiple: int, reference_short_edge: int):
        """Put a reference image on the tier's reference short edge (2048 at 768p), 32-rounded.

        Matches the reference setup step: upscale included, no area cap, 1:4..4:1 aspects.
        """
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        image = image.convert("RGB")
        width, height = image.size
        if width > 4 * height or height > 4 * width:
            raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")
        scale = reference_short_edge / min(width, height)
        target_height = max(multiple, round(height * scale / multiple) * multiple)
        target_width = max(multiple, round(width * scale / multiple) * multiple)
        if image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return image

    def _encode_audio_reference(self, waveform: torch.Tensor, sample_rate: int, max_samples: int) -> torch.Tensor:
        """Encode a reference soundtrack to clean channel-major conditioning rows.

        Matches the reference: resample to the audio VAE's rate, upmix mono to stereo,
        truncate to the generated duration, encode, take the posterior *mode* (never
        sampled), normalize, pack channel-major.
        """
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
        if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
            raise ValueError(f"A reference soundtrack must be (channels, samples), got {tuple(waveform.shape)}.")
        if waveform.shape[0] != 2:
            waveform = waveform.expand(2, -1)
        if sample_rate != 32000:
            import torchaudio

            waveform = torchaudio.transforms.Resample(sample_rate, 32000)(waveform)
        waveform = waveform[:, :max_samples].contiguous()

        posterior = self.audio_vae.encode(waveform[:, None], return_dict=False)[0]
        latents = posterior.mode().float().cpu().transpose(1, 2)
        mean = torch.tensor(self.audio_vae.config.latents_mean).view(1, 1, -1)
        std = torch.tensor(self.audio_vae.config.latents_std).view(1, 1, -1)
        normalized = (latents - mean) / std
        return normalized.reshape(-1, self.transformer.config.audio_in_channels)

'''

OLD_TIMESTEP_PLAN = '''        n_cond = self._num_condition_video_rows
        timestep_indices = (token_tags == MINIMAX_H3_AUDIO_TAG).long()
        if n_cond > 0:
            timestep_indices = timestep_indices.clone()
            timestep_indices[video_indices[:n_cond]] = 2
'''

NEW_TIMESTEP_PLAN = '''        n_cond = self._num_condition_video_rows
        n_cond_a = self._num_condition_audio_rows
        timestep_indices = (token_tags == MINIMAX_H3_AUDIO_TAG).long()
        if n_cond > 0 or n_cond_a > 0:
            timestep_indices = timestep_indices.clone()
            if n_cond > 0:
                timestep_indices[video_indices[:n_cond]] = 2
            if n_cond_a > 0:
                # Audio-reference rows sit after the video-condition entry when there is one.
                timestep_indices[audio_indices[:n_cond_a]] = 2 + (1 if n_cond > 0 else 0)
'''

OLD_STEP_T = '''            t_video = 1.0 - sig_v[i]
            if n_cond > 0:
                t_cond = torch.clamp(t_video, min=KEYFRAME_NOISE_AUG)
                timestep = torch.stack([t_video, 1.0 - sig_a[i], t_cond]).to(self.device)
            else:
                timestep = torch.stack([t_video, 1.0 - sig_a[i]]).to(self.device)
'''

NEW_STEP_T = '''            t_video = 1.0 - sig_v[i]
            if n_cond > 0 or n_cond_a > 0:
                # Reference audio rows are clean: pinned at t = 1.0 (condition_audio_timestep).
                entries = [t_video, 1.0 - sig_a[i]]
                if n_cond > 0:
                    entries.append(torch.clamp(t_video, min=KEYFRAME_NOISE_AUG))
                if n_cond_a > 0:
                    entries.append(torch.ones((), dtype=t_video.dtype))
                timestep = torch.stack(entries).to(self.device)
            else:
                timestep = torch.stack([t_video, 1.0 - sig_a[i]]).to(self.device)
'''

OLD_AUDIO_RENOISE = '''            noise_a = torch.randn(
                audio_rows.shape, generator=generator, dtype=torch.float32
            )
            video_rows[gen] = (1.0 - sig_v[i + 1]) * x0_v + sig_v[i + 1] * noise_v
            audio_rows = (1.0 - sig_a[i + 1]) * x0_a + sig_a[i + 1] * noise_a
'''

NEW_AUDIO_RENOISE = '''            gen_a = slice(self._num_condition_audio_rows, None)
            x0_a = audio_rows[gen_a] + sig_a[i] * audio_noise_pred.cpu()[0].float()[gen_a]
            noise_a = torch.randn(
                audio_rows[gen_a].shape, generator=generator, dtype=torch.float32
            )
            video_rows[gen] = (1.0 - sig_v[i + 1]) * x0_v + sig_v[i + 1] * noise_v
            audio_rows[gen_a] = (1.0 - sig_a[i + 1]) * x0_a + sig_a[i + 1] * noise_a
'''

OLD_X0A = '''            x0_v = video_rows[gen] + sig_v[i] * noise_pred.cpu()[0].float()[gen]
            x0_a = audio_rows + sig_a[i] * audio_noise_pred.cpu()[0].float()
'''

NEW_X0A = '''            x0_v = video_rows[gen] + sig_v[i] * noise_pred.cpu()[0].float()[gen]
'''

OLD_FORWARD_REFS = '''        params = req.sampling_params_list[0]
        raw_prompt = req.prompts[0]
        keyframes = None
        keyframe_anchors = ()
        if isinstance(raw_prompt, str):
            prompt = raw_prompt
        else:
            mm_data = raw_prompt.get("multi_modal_data") or {}
            keyframes = mm_data.get("keyframes") or None
'''

NEW_FORWARD_REFS = '''        params = req.sampling_params_list[0]
        raw_prompt = req.prompts[0]
        keyframes = None
        keyframe_anchors = ()
        references = None
        if isinstance(raw_prompt, str):
            prompt = raw_prompt
        else:
            mm_data = raw_prompt.get("multi_modal_data") or {}
            keyframes = mm_data.get("keyframes") or None
            references = mm_data.get("references") or None
'''

OLD_LAYOUT2 = '''        self._fl2va_height, self._fl2va_width = height, width
        layout = self._build_fl2va_layout(
            torch.full((prompt_embeds.shape[1],), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            patch_size,
            keyframe_anchors=keyframe_anchors,
        )
'''

NEW_LAYOUT2 = '''        self._fl2va_height, self._fl2va_width = height, width
        text_token_tags = torch.full((prompt_embeds.shape[1],), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
        if references:
            # Ref2VA: encode visual references (their own resolutions drive the layout),
            # encode audio references, then build the reference layout.
            t0 = time.perf_counter()
            self._ref2va_kinds = [entry["kind"] for entry in references]
            reference_latents = []
            audio_reference_latents = []
            for entry in references:
                if entry["kind"] == "image":
                    image = self._normalize_image_reference(
                        entry["image"], VAE_SPATIAL_COMPRESSION * patch_size[2], 2048
                    )
                    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None, :, None]
                    encoder_device = next(self.vae.encoder.parameters()).device
                    latents = self._encode_vae_condition(pixels.to(encoder_device))
                    reference_latents.append(latents.to(torch.bfloat16).to(torch.float32))
                elif entry["kind"] == "audio":
                    max_samples = round(num_frames / MINIMAX_H3_FPS * 32000)
                    audio_reference_latents.append(
                        self._encode_audio_reference(
                            entry["audio"], int(entry.get("sample_rate") or 32000), max_samples
                        )
                    )
                else:
                    raise ValueError(f"ref2va references must be 'image' or 'audio', got {entry['kind']!r}")
            reference_geometries = [tuple(latents.shape[2:5]) for latents in reference_latents]
            layout = self._build_ref2va_layout(
                text_token_tags,
                reference_geometries,
                audio_reference_latents,
                num_latent_frames,
                latent_height,
                latent_width,
                num_audio_latents,
                patch_size,
            )
            logger.info(
                "Ref2VA: %d image + %d audio reference(s) -> %d video + %d audio condition rows (encode %.2fs)",
                len(reference_latents),
                len(audio_reference_latents),
                layout[5],
                layout[6],
                time.perf_counter() - t0,
            )
            self._ref2va_visual_latents = reference_latents
            self._ref2va_audio_latents = audio_reference_latents
        else:
            self._ref2va_kinds = None
            layout = self._build_fl2va_layout(
                text_token_tags,
                num_latent_frames,
                latent_height,
                latent_width,
                num_audio_latents,
                patch_size,
                keyframe_anchors=keyframe_anchors,
            )
'''

OLD_COND_PREP = '''        generator = (req.collate_request_generators(1, None) or [None])[0]
        condition_rows = None
        if keyframes:
            t0 = time.perf_counter()
            condition_latents = self._prepare_keyframe_latents(keyframes)
            condition_rows = self._noise_keyframe_latents(condition_latents, generator, patch_size)
            self._num_condition_video_rows = condition_rows.shape[0]
            logger.info(
                "FL2VA: %d keyframe(s) -> %d condition rows (encode %.2fs)",
                len(keyframes),
                self._num_condition_video_rows,
                time.perf_counter() - t0,
            )
        else:
            self._num_condition_video_rows = 0
'''

NEW_COND_PREP = '''        generator = (req.collate_request_generators(1, None) or [None])[0]
        condition_rows = None
        audio_condition_rows = None
        self._num_condition_audio_rows = 0
        if references:
            # Draw one noise per visual reference, in packed order, before the generated noise.
            condition_rows = self._noise_keyframe_latents(self._ref2va_visual_latents, generator, patch_size)
            self._num_condition_video_rows = condition_rows.shape[0] if condition_rows is not None else 0
            if self._ref2va_audio_latents:
                audio_condition_rows = torch.cat(self._ref2va_audio_latents)
                self._num_condition_audio_rows = audio_condition_rows.shape[0]
        elif keyframes:
            t0 = time.perf_counter()
            condition_latents = self._prepare_keyframe_latents(keyframes)
            condition_rows = self._noise_keyframe_latents(condition_latents, generator, patch_size)
            self._num_condition_video_rows = condition_rows.shape[0]
            logger.info(
                "FL2VA: %d keyframe(s) -> %d condition rows (encode %.2fs)",
                len(keyframes),
                self._num_condition_video_rows,
                time.perf_counter() - t0,
            )
        else:
            self._num_condition_video_rows = 0
'''

OLD_DENOISE2 = '''        t0 = time.perf_counter()
        video_rows = patchify_video_latents(latents, patch_size)
        if condition_rows is not None:
            video_rows = torch.cat([condition_rows, video_rows])
'''

NEW_DENOISE2 = '''        t0 = time.perf_counter()
        video_rows = patchify_video_latents(latents, patch_size)
        if condition_rows is not None:
            video_rows = torch.cat([condition_rows, video_rows])
        if audio_condition_rows is not None:
            audio_rows = torch.cat([audio_condition_rows, audio_rows])
'''

OLD_STRIP = '''        # FL2VA: strip the frozen anchors; only the generated rows are unpacked and decoded.
        if condition_rows is not None:
            video_rows = video_rows[self._num_condition_video_rows :]
            self._num_condition_video_rows = 0
'''

NEW_STRIP = '''        # FL2VA/Ref2VA: strip the frozen conditioning; only the generated rows are decoded.
        if condition_rows is not None:
            video_rows = video_rows[self._num_condition_video_rows :]
            self._num_condition_video_rows = 0
        if audio_condition_rows is not None:
            audio_rows = audio_rows[self._num_condition_audio_rows :]
            self._num_condition_audio_rows = 0
'''


def apply(src: str) -> str:
    if "_build_ref2va_layout" in src:
        raise SystemExit("ref2va already applied; refusing to re-apply")

    # The audio VAE must load on every rank for ref2va (every rank encodes the
    # conditioning); the WAV write stays rank-0-only.
    old = "        if self._decode_audio and self.is_output_rank:\n"
    assert old in src, "audio load gate anchor"
    src = src.replace(old, "        if self._decode_audio:\n", 1)
    old = '''        directory = os.environ.get("MINIMAX_H3_OUT") or os.environ.get("WORKLOAD_OUTPUT_RW")
        if not directory:
            return None
'''
    assert old in src, "wav gate anchor"
    src = src.replace(old, '''        directory = os.environ.get("MINIMAX_H3_OUT") or os.environ.get("WORKLOAD_OUTPUT_RW")
        if not directory or not self.is_output_rank:
            return None
''', 1)

    anchor = "    def block_modulations(self, timestep: torch.Tensor)"
    assert anchor in src
    src = src.replace(anchor, REF2VA_METHODS + anchor, 1)

    assert OLD_TIMESTEP_PLAN in src, "timestep plan anchor"
    src = src.replace(OLD_TIMESTEP_PLAN, NEW_TIMESTEP_PLAN, 1)
    assert OLD_STEP_T in src, "step t anchor"
    src = src.replace(OLD_STEP_T, NEW_STEP_T, 1)
    assert OLD_X0A in src, "x0a anchor"
    src = src.replace(OLD_X0A, NEW_X0A, 1)
    assert OLD_AUDIO_RENOISE in src, "audio renoise anchor"
    src = src.replace(OLD_AUDIO_RENOISE, NEW_AUDIO_RENOISE, 1)
    assert OLD_FORWARD_REFS in src, "forward refs anchor"
    src = src.replace(OLD_FORWARD_REFS, NEW_FORWARD_REFS, 1)
    assert OLD_LAYOUT2 in src, "layout anchor"
    src = src.replace(OLD_LAYOUT2, NEW_LAYOUT2, 1)
    assert OLD_COND_PREP in src, "cond prep anchor"
    src = src.replace(OLD_COND_PREP, NEW_COND_PREP, 1)
    assert OLD_DENOISE2 in src, "denoise anchor"
    src = src.replace(OLD_DENOISE2, NEW_DENOISE2, 1)
    assert OLD_STRIP in src, "strip anchor"
    src = src.replace(OLD_STRIP, NEW_STRIP, 1)

    return src


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    backup = path + ".pre_ref2va"
    with open(backup, "w") as f:
        f.write(src)
    with open(path, "w") as f:
        f.write(apply(src))
    print(f"Applied ref2va to {path} (backup: {backup})")


if __name__ == "__main__":
    main()
