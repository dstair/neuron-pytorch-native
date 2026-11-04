# SPDX-License-Identifier: Apache-2.0
"""Pika-H3-Flex FL2VA (first/last-frame image-to-video) extensions for Neuron pipeline.

This module provides FL2VA support as a drop-in extension to the existing
NeuronMiniMaxH3Pipeline. Key additions:

1. VAE encoding of keyframe images (ImageNet normalize + seed-42 posterior)
2. FL2VA packed sequence layout (keyframe anchors)
3. Vision-tower encoding for keyframes (Qwen3-VL)
4. Frozen anchor rows during denoising (keyframe_noise_aug=0.999)

Reference: diffusers main's modular_pipelines/pika_h3_flex/
"""

import logging
import time

import numpy as np
import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    audio_latent_num_frames,
    video_latent_num_frames,
)
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

logger = logging.getLogger(__name__)

# Constants from the reference
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)

KEYFRAME_NOISE_AUG = 0.999
KEYFRAME_ENCODE_SEED = 42

VAE_FRAMES_PER_CHUNK = 17
VAE_LATENTS_PER_CHUNK = 5
VAE_SPATIAL_COMPRESSION = 16


def encode_vae_condition(vae, pixels: torch.Tensor) -> torch.Tensor:
    """Encode a keyframe image to normalized conditioning latents.

    Exact recipe from Pika reference:
    1. ImageNet normalize
    2. VAE encode (spatial encoder only for single frame)
    3. Sample posterior with fixed seed=42
    4. Round to float16 before normalizing

    Args:
        vae: AutoencoderKLMiniMaxH3
        pixels: (1, 3, 1, H, W) uint8 tensor on VAE device

    Returns:
        (1, C, 1, H/16, W/16) float32 CPU tensor, normalized
    """
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    pixel_mean = torch.tensor(PIXEL_MEAN, device=pixels.device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD, device=pixels.device).view(1, -1, 1, 1, 1)

    # ImageNet normalize
    pixels_norm = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std

    # Encode + sample posterior
    posterior = vae.encode(pixels_norm, return_dict=False)[0]
    latents = posterior.sample(generator=torch.Generator().manual_seed(KEYFRAME_ENCODE_SEED))

    # Round to float16 (~11 bits) per fl2va adapter's bf16_conditioning_v1 policy
    latents = latents.to(torch.float16).float().cpu()

    return (latents - latents_mean) / latents_std


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """Aspect-normalized spatial rotary axis (numpy linspace, endpoint=False)."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    """Rotary time of every latent frame, starting at origin.

    Non-uniform: 5/3 * (1, 4, 4, 4, 4) pattern.
    """
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
            for i in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The (h, w) rotary coordinates of one latent frame."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=-1), width_grid


def build_fl2va_layout(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    keyframe_anchors: tuple[str, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build FL2VA packed sequence layout [text | keyframes | audio | video].

    Matches PikaH3FlexPrepareLayoutStep.build_packed_sequence from the reference.

    Args:
        text_token_tags: (num_text_tokens,) modality tags for text rows
        num_latent_frames: Number of generated video latent frames
        latent_height: Latent height
        latent_width: Latent width
        num_audio_latents: Audio latents per channel
        patch_size: Transformer (t, h, w) patch
        keyframe_anchors: Tuple of "first" and/or "last" anchor positions

    Returns:
        (position_ids, token_tags, video_indices, audio_indices, text_indices,
         num_condition_rows)
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

    # Build (t, h, w) rotary grid in float64
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    # Place keyframe anchors
    for idx, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            # Numpy pairwise sum for exact match (spans.sum() in float64)
            spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
            for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
                spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[offset]
            anchor_time = float(num_text_tokens) + float(spans.sum()) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"keyframe anchor must be 'first' or 'last', got {anchor!r}")

        rows = slice(condition_start + idx * rows_per_frame, condition_start + (idx + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    # Audio rows: channel-major, no height, pinned to width extremes
    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )

    # Target video rows
    video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    # Row indices and modality tags
    video_indices = torch.cat(
        [torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)]
    )
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

    return position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows


def prepare_keyframe_latents(
    vae,
    keyframes: list[Image.Image],
    target_height: int,
    target_width: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Encode keyframe images to normalized conditioning latents.

    Args:
        vae: Video VAE
        keyframes: PIL Images (in packed order: first, then last)
        target_height, target_width: Canvas dimensions
        device: VAE device

    Returns:
        List of (1, C, 1, H/16, W/16) normalized latent tensors
    """
    condition_latents = []
    for image in keyframes:
        # Resize to canvas
        if image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # (H, W, 3) uint8 -> (1, 3, 1, H, W)
        pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None, :, None]
        latents = encode_vae_condition(vae, pixels)
        # bf16 per fl2va policy
        latents = latents.to(torch.bfloat16).to(torch.float32)
        condition_latents.append(latents)

    return condition_latents


def noise_keyframe_latents(
    scheduler,
    condition_latents: list[torch.Tensor],
    generator: torch.Generator | None,
    patch_size: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Noise keyframe latents to conditioning level and pack into rows.

    Each anchor noised to t=0.999, drawn from request generator (one draw per condition).
    """
    packed = []
    for latents in condition_latents:
        noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=torch.float32)
        noised = scheduler.scale_noise(latents.to(device), KEYFRAME_NOISE_AUG, noise)
        packed.append(patchify_video_latents(noised, patch_size))

    return torch.cat(packed) if packed else torch.empty(0, dtype=torch.float32)


class FL2VAExtensions:
    """FL2VA extensions that can be mixed into NeuronMiniMaxH3Pipeline.

    Usage:
        # In forward(), check for keyframes
        if keyframes:
            self._num_condition_video_rows = ...
            condition_rows = noise_keyframe_latents(...)
            video_rows = cat([condition_rows, video_rows])
            # In denoise loop, only update generated rows
    """

    @staticmethod
    def is_fl2va_request(req) -> bool:
        """Check if request has keyframe conditioning."""
        if isinstance(req.prompts[0], str):
            return False
        data = req.prompts[0]
        if isinstance(data, dict):
            return bool(data.get("multi_modal_data", {}).get("keyframes"))
        return False

    @staticmethod
    def get_keyframes(req) -> tuple[list[Image.Image], tuple[str, ...]]:
        """Extract keyframes and anchors from request.

        Returns:
            (keyframes_list, keyframe_anchors_tuple)
        """
        data = req.prompts[0]
        mm_data = data.get("multi_modal_data", {})
        keyframes = mm_data.get("keyframes", [])
        anchors = mm_data.get("keyframe_anchors", ())

        # Default: interpret keyframes in order (first, then last)
        if not anchors and keyframes:
            anchors = tuple("first" if i == 0 else "last" for i in range(len(keyframes)))

        return keyframes, anchors
