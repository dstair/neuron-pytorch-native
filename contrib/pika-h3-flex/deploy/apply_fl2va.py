#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply FL2VA (first/last-frame image-to-video) support to pipeline_minimax_h3.py.

Anchor-string based edits (robust to line drift); makes a ``.pre_fl2va`` backup.

What this adds, per diffusers main's ``modular_pipelines/pika_h3_flex``:

1. Keyframe VAE encoding (``_encode_vae_condition``): ImageNet normalization, seed-42
   posterior sample, float16 rounding. Runs on the VAE's current device — the host when
   ``vae_swap_dit_blocks`` keeps the VAE off the accelerator (uncompiled upstream path,
   FP32 encoder, exact recipe; device-side encode is a later optimization).
2. FL2VA packed layout (``_build_fl2va_layout``): ``[text | keyframes | audio | video]``
   with anchor RoPE times ("first" at num_text_tokens, "last" via numpy pairwise sum).
3. Frozen anchors in ``_denoise_pika``: the DiT sees condition + generated rows; only
   generated rows are renoised. Anchor rows are pinned at
   ``max(video_t, keyframe_noise_aug)`` via a third timestep entry (the reference's
   ``row_timestep_plan``). T2V (no anchors) is bit-identical to before.
4. ``forward()`` parses ``multi_modal_data.keyframes`` (first image -> "first" anchor,
   second -> "last"), noises anchors BEFORE the generated noise (reference draw order),
   concatenates them in front of the generated rows, and strips them before decode.

NOT included (follow-up): vision-tower conditioning of the keyframes in the text
encoder (the "<Picture i>: " presentation + Qwen3-VL vision blocks). The text segment
is prompt-only, so the DiT sees the keyframes through the VAE-anchor rows alone.

Usage on box:
    python apply_fl2va.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys


FL2VA_CONSTANTS = '''
# FL2VA (Pika first/last-frame image-to-video) constants.
KEYFRAME_NOISE_AUG = 0.999
KEYFRAME_ENCODE_SEED = 42
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32
'''

FL2VA_INIT = '''        # FL2VA: number of leading video rows that are frozen keyframe conditioning.
        self._num_condition_video_rows = 0
'''

FL2VA_METHODS = '''
    def _encode_vae_condition(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode one keyframe image into normalized conditioning latents.

        The released recipe (diffusers main ``encode_vae_condition``): ImageNet-normalize
        the uint8 pixels, VAE-encode (the spatial encoder alone for a single frame),
        sample the posterior under a fixed seed independent of the request generator,
        and round to float16 before normalizing. Runs on the VAE's current device (the
        host when ``vae_swap_dit_blocks`` keeps it off the accelerator): the uncompiled
        upstream path in FP32, which the encoder requires (BF16 moments go NaN).

        Args:
            pixels: ``(1, 3, 1, H, W)`` uint8 tensor on the VAE encoder's device.

        Returns:
            ``(1, C, 1, H//16, W//16)`` float32 CPU tensor, normalized.
        """
        vae = self.vae
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(PIXEL_MEAN, device=pixels.device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(PIXEL_STD, device=pixels.device).view(1, -1, 1, 1, 1)

        pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
        posterior = vae.encode(pixels, return_dict=False)[0]
        latents = posterior.sample(generator=torch.Generator().manual_seed(KEYFRAME_ENCODE_SEED))
        latents = latents.to(torch.float16).float().cpu()
        return (latents - latents_mean) / latents_std

    @staticmethod
    def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
        ratio = dim / sqrt_area
        left = (1.0 - ratio) / 2.0
        grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
        return torch.from_numpy(grid).to(torch.float64)

    @staticmethod
    def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
        spans = torch.tensor(
            [
                _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
                for i in range(num_latent_frames)
            ],
            dtype=torch.float64,
        )
        return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])

    def _frame_position_grid(self, latent_height, latent_width, patch_h, patch_w):
        sqrt_area = float(np.sqrt(latent_height * latent_width))
        height_grid = self._spatial_position_grid(latent_height, patch_h, sqrt_area)
        width_grid = self._spatial_position_grid(latent_width, patch_w, sqrt_area)
        grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
        return torch.stack([g.reshape(-1) for g in grids], dim=-1), width_grid

    def _build_fl2va_layout(
        self,
        text_token_tags: torch.Tensor,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        patch_size: tuple,
        keyframe_anchors: tuple = (),
    ):
        """The ``[text | keyframe conditions | target audio | target video]`` layout.

        Matches diffusers main ``PikaH3FlexPrepareLayoutStep.build_packed_sequence``;
        with empty ``keyframe_anchors`` this is the T2V layout.
        """
        _, patch_h, patch_w = patch_size
        rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
        num_text_tokens = text_token_tags.shape[0]
        num_condition_rows = len(keyframe_anchors) * rows_per_frame
        num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
        num_video_rows = num_latent_frames * rows_per_frame
        sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

        condition_start = num_text_tokens
        audio_start = condition_start + num_condition_rows
        video_start = audio_start + num_audio_rows

        position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
        position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

        frame_grid, width_grid = self._frame_position_grid(latent_height, latent_width, patch_h, patch_w)

        for index, anchor in enumerate(keyframe_anchors):
            if anchor == "first":
                anchor_time = float(num_text_tokens)
            elif anchor == "last":
                # Summed by numpy's pairwise summation, as the reference computes this anchor.
                spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
                for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
                    spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[offset]
                anchor_time = float(num_text_tokens) + float(spans.sum()) - _ROPE_FRAME_RESCALE
            else:
                raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}")
            rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
            position_ids[rows, 0] = anchor_time
            position_ids[rows, 1:] = frame_grid

        audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
        position_ids[audio_start:video_start, 0] = audio_time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
        position_ids[audio_start:video_start, 2] = torch.cat(
            [
                torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
                torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
            ]
        )

        video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
        video_position_ids[:, :, 0] = self._temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
        video_position_ids[:, :, 1:] = frame_grid[None]
        position_ids[video_start:] = video_position_ids.reshape(-1, 3)

        video_indices = torch.cat(
            [torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)]
        )
        audio_indices = torch.arange(audio_start, video_start)
        text_indices = torch.arange(num_text_tokens)

        token_tags = torch.empty(sequence_length, dtype=torch.long)
        token_tags[text_indices] = text_token_tags.to(torch.long)
        token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
        token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

        # Same 7-field shape as MiniMaxH3PrepareLayoutStep.build_packed_sequence: FL2VA has no
        # audio conditioning rows (that is ref2va), so the last count is 0.
        return (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            num_condition_rows,
            0,
        )

    def _prepare_keyframe_latents(self, keyframes: list) -> list:
        """VAE-encode keyframe images to normalized conditioning latents, one per keyframe."""
        from PIL import Image

        height = self._fl2va_height
        width = self._fl2va_width
        encoder_device = next(self.vae.encoder.parameters()).device
        condition_latents = []
        for image in keyframes:
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            pixels = torch.from_numpy(np.array(image)).to(encoder_device).permute(2, 0, 1)[None, :, None]
            latents = self._encode_vae_condition(pixels)
            # The fl2va adapter's bf16_conditioning_v1 policy.
            condition_latents.append(latents.to(torch.bfloat16).to(torch.float32))
        return condition_latents

    def _noise_keyframe_latents(self, condition_latents: list, generator, patch_size: tuple):
        """Noise keyframe latents to the conditioning level and pack into rows.

        Anchors are noised to ``keyframe_noise_aug`` (0.999, not clean t=1.0), one draw
        per condition from the request's generator, *before* the generated rows' noise —
        the draw order is part of what the generator reproduces.
        """
        packed = []
        for latents in condition_latents:
            noise = randn_tensor(latents.shape, generator=generator, dtype=torch.float32)
            noised = self.scheduler.scale_noise(latents, KEYFRAME_NOISE_AUG, noise)
            packed.append(patchify_video_latents(noised, patch_size))
        return torch.cat(packed) if packed else None

'''

OLD_RENOISE = '''            # Renoise (stochastic_sampling): x0 = latents + sigma_i * v, then re-noise to sigma_{i+1}.
            x0_v = video_rows + sig_v[i] * noise_pred.cpu()[0].float()
            x0_a = audio_rows + sig_a[i] * audio_noise_pred.cpu()[0].float()
            noise_v = torch.randn(
                video_rows.shape, generator=generator, dtype=torch.float32
            )
            noise_a = torch.randn(
                audio_rows.shape, generator=generator, dtype=torch.float32
            )
            video_rows = (1.0 - sig_v[i + 1]) * x0_v + sig_v[i + 1] * noise_v
            audio_rows = (1.0 - sig_a[i + 1]) * x0_a + sig_a[i + 1] * noise_a
'''

NEW_RENOISE = '''            # Renoise (stochastic_sampling): x0 = latents + sigma_i * v, then re-noise to sigma_{i+1}.
            # FL2VA: only the generated rows are written; the frozen anchors ride through, and
            # their noise_pred entries are discarded, as the reference's scheduler step does.
            gen = slice(self._num_condition_video_rows, None)
            x0_v = video_rows[gen] + sig_v[i] * noise_pred.cpu()[0].float()[gen]
            x0_a = audio_rows + sig_a[i] * audio_noise_pred.cpu()[0].float()
            noise_v = torch.randn(
                video_rows[gen].shape, generator=generator, dtype=torch.float32
            )
            noise_a = torch.randn(
                audio_rows.shape, generator=generator, dtype=torch.float32
            )
            video_rows[gen] = (1.0 - sig_v[i + 1]) * x0_v + sig_v[i + 1] * noise_v
            audio_rows = (1.0 - sig_a[i + 1]) * x0_a + sig_a[i + 1] * noise_a
'''

OLD_TIMESTEP = '''        dit_inputs = {
            "encoder_hidden_states": prompt_embeds.to(self.dtype).to(self.device),
            "timestep_indices": (token_tags == MINIMAX_H3_AUDIO_TAG).long().to(self.device),
            "token_tags": token_tags.to(self.device),
            "position_ids": position_ids.float().to(self.device),
            "video_indices": video_indices.to(self.device),
            "audio_indices": audio_indices.to(self.device),
            "text_indices": text_indices.to(self.device),
        }
        step_seconds = []
        for i in range(sig_v.numel() - 1):
            t0 = time.perf_counter()
            timestep = torch.stack([1.0 - sig_v[i], 1.0 - sig_a[i]]).to(self.device)
'''

NEW_TIMESTEP = '''        # FL2VA: the reference's row_timestep_plan pins keyframe condition rows at
        # max(video_t, keyframe_noise_aug) — a third timestep entry; generated video rows
        # keep the video schedule, audio rows the audio schedule, text rows the video's.
        n_cond = self._num_condition_video_rows
        timestep_indices = (token_tags == MINIMAX_H3_AUDIO_TAG).long()
        if n_cond > 0:
            timestep_indices = timestep_indices.clone()
            timestep_indices[video_indices[:n_cond]] = 2
        dit_inputs = {
            "encoder_hidden_states": prompt_embeds.to(self.dtype).to(self.device),
            "timestep_indices": timestep_indices.to(self.device),
            "token_tags": token_tags.to(self.device),
            "position_ids": position_ids.float().to(self.device),
            "video_indices": video_indices.to(self.device),
            "audio_indices": audio_indices.to(self.device),
            "text_indices": text_indices.to(self.device),
        }
        step_seconds = []
        for i in range(sig_v.numel() - 1):
            t0 = time.perf_counter()
            t_video = 1.0 - sig_v[i]
            if n_cond > 0:
                t_cond = torch.clamp(t_video, min=KEYFRAME_NOISE_AUG)
                timestep = torch.stack([t_video, 1.0 - sig_a[i], t_cond]).to(self.device)
            else:
                timestep = torch.stack([t_video, 1.0 - sig_a[i]]).to(self.device)
'''

OLD_FORWARD_PROMPT = '''        params = req.sampling_params_list[0]
        raw_prompt = req.prompts[0]
        if isinstance(raw_prompt, str):
            prompt = raw_prompt
        else:
            if raw_prompt.get("multi_modal_data"):
                raise NotImplementedError("MiniMax-H3 on Neuron supports text-to-video only")
            prompt = raw_prompt.get("prompt")
'''

NEW_FORWARD_PROMPT = '''        params = req.sampling_params_list[0]
        raw_prompt = req.prompts[0]
        keyframes = None
        keyframe_anchors = ()
        if isinstance(raw_prompt, str):
            prompt = raw_prompt
        else:
            mm_data = raw_prompt.get("multi_modal_data") or {}
            keyframes = mm_data.get("keyframes") or None
            if keyframes:
                # Packed order: the first image anchors the first frame, a second the last.
                keyframe_anchors = tuple(
                    "first" if i == 0 else "last" for i in range(len(keyframes))
                )
            prompt = raw_prompt.get("prompt")
'''

OLD_LAYOUT = '''        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            torch.full((prompt_embeds.shape[1],), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            patch_size,
            MINIMAX_H3_AUDIO_CHANNELS,
            MINIMAX_H3_AUDIO_TAG,
            MINIMAX_H3_VIDEO_TAG,
        )

        # The reference's draw order on the request's generator: the video noise as a latent
        # tensor, then the audio noise in row layout.
        generator = (req.collate_request_generators(1, None) or [None])[0]
'''

NEW_LAYOUT = '''        self._fl2va_height, self._fl2va_width = height, width
        layout = self._build_fl2va_layout(
            torch.full((prompt_embeds.shape[1],), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            patch_size,
            keyframe_anchors=keyframe_anchors,
        )

        # The reference's draw order on the request's generator: any conditioning noise first
        # (one draw per keyframe), then the video noise as a latent tensor, then the audio
        # noise in row layout.
        generator = (req.collate_request_generators(1, None) or [None])[0]
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

OLD_DENOISE_CALL = '''        t0 = time.perf_counter()
        video_rows, step_seconds = self.denoise(
            prompt_embeds,
            layout,
            patchify_video_latents(latents, patch_size),
            audio_rows,
            num_steps,
            generator=generator,
        )
        denoise_seconds = time.perf_counter() - t0
'''

NEW_DENOISE_CALL = '''        t0 = time.perf_counter()
        video_rows = patchify_video_latents(latents, patch_size)
        if condition_rows is not None:
            video_rows = torch.cat([condition_rows, video_rows])
        video_rows, step_seconds = self.denoise(
            prompt_embeds,
            layout,
            video_rows,
            audio_rows,
            num_steps,
            generator=generator,
        )
        # FL2VA: strip the frozen anchors; only the generated rows are unpacked and decoded.
        if condition_rows is not None:
            video_rows = video_rows[self._num_condition_video_rows :]
            self._num_condition_video_rows = 0
        denoise_seconds = time.perf_counter() - t0
'''


def apply(src: str) -> str:
    if "KEYFRAME_NOISE_AUG" in src:
        raise SystemExit("FL2VA already applied (KEYFRAME_NOISE_AUG present); refusing to re-apply")

    # 1. numpy import.
    anchor = "import torch\nimport torch.distributed as dist"
    assert anchor in src, "torch import anchor not found"
    src = src.replace(anchor, "import numpy as np\nimport torch\nimport torch.distributed as dist", 1)

    # 2. FL2VA constants after the duration bounds.
    anchor = "MIN_DURATION_S = 5.0\nMAX_DURATION_S = 15.0\n"
    assert anchor in src, "duration anchor not found"
    src = src.replace(anchor, anchor + FL2VA_CONSTANTS, 1)

    # 3. __init__ state after _pika_renoise.
    anchor = '        self._pika_renoise = bool(model_config.pop("pika_renoise", False))\n'
    assert anchor in src, "_pika_renoise anchor not found"
    src = src.replace(anchor, anchor + FL2VA_INIT, 1)

    # 4. FL2VA methods before block_modulations.
    anchor = "    def block_modulations(self, timestep: torch.Tensor)"
    assert anchor in src, "block_modulations anchor not found"
    src = src.replace(anchor, FL2VA_METHODS + anchor, 1)

    # 5. _denoise_pika: timestep plan + generated-only renoise.
    assert OLD_TIMESTEP in src, "timestep anchor not found"
    src = src.replace(OLD_TIMESTEP, NEW_TIMESTEP, 1)
    assert OLD_RENOISE in src, "renoise anchor not found"
    src = src.replace(OLD_RENOISE, NEW_RENOISE, 1)

    # 6. forward(): keyframe parsing, FL2VA layout, condition prep, concat/strip.
    assert OLD_FORWARD_PROMPT in src, "forward prompt anchor not found"
    src = src.replace(OLD_FORWARD_PROMPT, NEW_FORWARD_PROMPT, 1)
    assert OLD_LAYOUT in src, "layout anchor not found"
    src = src.replace(OLD_LAYOUT, NEW_LAYOUT, 1)
    assert OLD_DENOISE_CALL in src, "denoise call anchor not found"
    src = src.replace(OLD_DENOISE_CALL, NEW_DENOISE_CALL, 1)

    return src


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    backup = path + ".pre_fl2va"
    with open(backup, "w") as f:
        f.write(src)
    with open(path, "w") as f:
        f.write(apply(src))
    print(f"Applied FL2VA to {path} (backup: {backup})")


if __name__ == "__main__":
    main()
