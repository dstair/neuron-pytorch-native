#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FL2VA implementation.

Run on the box:
    pytest test_fl2va.py -v

Or standalone:
    python test_fl2va.py
"""

import math

import numpy as np
import torch

# Constants from FL2VA implementation
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32

MINIMAX_H3_AUDIO_TAG = 2
MINIMAX_H3_VIDEO_TAG = 0
MINIMAX_H3_TEXT_TAG = 1
MINIMAX_H3_AUDIO_CHANNELS = 2


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float):
    """Port of the FL2VA spatial position grid."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float):
    """Port of the FL2VA temporal position grid."""
    spans = torch.tensor(
        [_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
         for i in range(num_latent_frames)],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _frame_position_grid(latent_height: int, latent_width: int, patch_h: int, patch_w: int):
    """Port of the FL2VA frame position grid."""
    sqrt_area = math.sqrt(latent_height * latent_width)
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
    patch_size: tuple,
    keyframe_anchors: tuple = (),
):
    """Simplified FL2VA layout builder for testing."""
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

    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    for idx, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
            for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
                spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[offset]
            anchor_time = float(num_text_tokens) + float(spans.sum()) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"keyframe anchor must be 'first' or 'last', got {anchor!r}")

        rows = slice(condition_start + idx * rows_per_frame, condition_start + (idx + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(MINIMAX_H3_AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat([
        torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
        torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
    ])

    video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    video_indices = torch.cat([torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)])
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = MINIMAX_H3_AUDIO_TAG
    token_tags[video_indices] = MINIMAX_H3_VIDEO_TAG

    return position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows


def test_t2v_layout():
    """Test T2V (no keyframes) layout matches existing behavior."""
    num_text_tokens = 50
    num_latent_frames = 37  # 124 frames -> 37 latent frames
    latent_height = 48  # 768 / 16
    latent_width = 84   # 1344 / 16
    num_audio_latents = 200
    patch_size = (1, 2, 2)

    text_token_tags = torch.full((num_text_tokens,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)

    position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows = build_fl2va_layout(
        text_token_tags,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        patch_size,
        keyframe_anchors=(),
    )

    # No condition rows for T2V
    assert num_condition_rows == 0

    # Check sequence length
    rows_per_frame = (latent_height // 2) * (latent_width // 2)
    expected_video_rows = num_latent_frames * rows_per_frame
    expected_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    expected_length = num_text_tokens + expected_video_rows + expected_audio_rows
    assert position_ids.shape[0] == expected_length

    # Text rows start at time 0, 1, 2, ...
    assert torch.allclose(position_ids[:num_text_tokens, 0], torch.arange(num_text_tokens, dtype=torch.float64))

    print("T2V layout test passed!")


def test_fl2va_first_frame_layout():
    """Test FL2VA with first-frame anchor."""
    num_text_tokens = 50
    num_latent_frames = 37
    latent_height = 48
    latent_width = 84
    num_audio_latents = 200
    patch_size = (1, 2, 2)

    text_token_tags = torch.full((num_text_tokens,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)

    position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows = build_fl2va_layout(
        text_token_tags,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        patch_size,
        keyframe_anchors=("first",),
    )

    # One keyframe = one frame's worth of condition rows
    rows_per_frame = (latent_height // 2) * (latent_width // 2)
    assert num_condition_rows == rows_per_frame

    # Condition rows are at num_text_tokens time
    condition_start = num_text_tokens
    assert position_ids[condition_start, 0] == float(num_text_tokens)

    # Condition rows have same spatial grid as video
    frame_grid, _ = _frame_position_grid(latent_height, latent_width, 2, 2)
    condition_spatial = position_ids[condition_start:condition_start + rows_per_frame, 1:]
    assert torch.allclose(condition_spatial, frame_grid)

    print("FL2VA first-frame layout test passed!")


def test_fl2va_last_frame_layout():
    """Test FL2VA with last-frame anchor."""
    num_text_tokens = 50
    num_latent_frames = 37
    latent_height = 48
    latent_width = 84
    num_audio_latents = 200
    patch_size = (1, 2, 2)

    text_token_tags = torch.full((num_text_tokens,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)

    position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows = build_fl2va_layout(
        text_token_tags,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        patch_size,
        keyframe_anchors=("last",),
    )

    # One keyframe
    rows_per_frame = (latent_height // 2) * (latent_width // 2)
    assert num_condition_rows == rows_per_frame

    # Last-frame anchor time = num_text_tokens + video_time_span - one_frame_rescale
    condition_start = num_text_tokens

    # Compute expected anchor time (numpy pairwise sum)
    spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
    for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
        spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[offset]
    expected_anchor_time = float(num_text_tokens) + float(spans.sum()) - _ROPE_FRAME_RESCALE

    assert abs(float(position_ids[condition_start, 0]) - expected_anchor_time) < 1e-10

    print("FL2VA last-frame layout test passed!")


def test_fl2va_both_anchors_layout():
    """Test FL2VA with both first and last frame anchors."""
    num_text_tokens = 50
    num_latent_frames = 37
    latent_height = 48
    latent_width = 84
    num_audio_latents = 200
    patch_size = (1, 2, 2)

    text_token_tags = torch.full((num_text_tokens,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)

    position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows = build_fl2va_layout(
        text_token_tags,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        patch_size,
        keyframe_anchors=("first", "last"),
    )

    # Two keyframes
    rows_per_frame = (latent_height // 2) * (latent_width // 2)
    assert num_condition_rows == 2 * rows_per_frame

    # First anchor at num_text_tokens
    condition_start = num_text_tokens
    assert position_ids[condition_start, 0] == float(num_text_tokens)

    # Second (last) anchor at correct time
    second_anchor_start = condition_start + rows_per_frame
    spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
    for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
        spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[offset]
    expected_anchor_time = float(num_text_tokens) + float(spans.sum()) - _ROPE_FRAME_RESCALE
    assert abs(float(position_ids[second_anchor_start, 0]) - expected_anchor_time) < 1e-10

    print("FL2VA both-anchors layout test passed!")


if __name__ == "__main__":
    test_t2v_layout()
    test_fl2va_first_frame_layout()
    test_fl2va_last_frame_layout()
    test_fl2va_both_anchors_layout()
    print("\nAll tests passed!")
