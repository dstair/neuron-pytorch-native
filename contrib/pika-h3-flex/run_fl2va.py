#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pika-H3-Flex FL2VA (image-to-video) test runner.

This script tests the FL2VA (first/last-frame image-to-video) implementation.

Usage (on the box):
    cd /data/omni29/plugin
    source /data/omni29/env29.sh

    # Single first-frame keyframe
    python contrib/pika-h3-flex/run_fl2va.py \
        --stage-config /data/omni29/pika_h3_flex_stage_fl2va.yaml \
        --first-frame /data/omni29/test_keyframe_first.png \
        --prompt "A parrot flying towards a glass tower" \
        --output /data/omni29/runs/fl2va_test/out.mp4

The stage config needs compile_vae_encoder: true for FL2VA to work.
"""

import argparse
import os
import sys
import time

import torch
from PIL import Image

# Bootstrap must come before vllm imports
import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip

from diffusers.utils import export_to_video
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

parser = argparse.ArgumentParser(description="Pika-H3-Flex FL2VA")
parser.add_argument("--model-path", type=str, default="/data/models/Pika-H3-Flex-Neuron")
parser.add_argument("--stage-config", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--first-frame", type=str, help="Path to first-frame keyframe")
parser.add_argument("--last-frame", type=str, help="Path to last-frame keyframe")
parser.add_argument("--height", type=int, default=768)
parser.add_argument("--width", type=int, default=1344)
parser.add_argument("--num-frames", type=int, default=124)
parser.add_argument("--num-inference-steps", type=int, default=6, help="Pika uses 6 NFE")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, default="fl2va_output.mp4")
parser.add_argument("--repeat", type=int, default=1)
args = parser.parse_args()


def main():
    # Collect keyframes
    keyframes = []
    if args.first_frame:
        keyframes.append(Image.open(args.first_frame).convert("RGB"))
    if args.last_frame:
        keyframes.append(Image.open(args.last_frame).convert("RGB"))

    if not keyframes:
        print("ERROR: Must provide --first-frame and/or --last-frame", file=sys.stderr)
        sys.exit(1)

    print(f"FL2VA test: {len(keyframes)} keyframe(s)")
    print(f"Canvas: {args.width}x{args.height}, {args.num_frames} frames, {args.num_inference_steps} steps")
    print(f"Prompt: {args.prompt[:80]}...")

    # Load engine
    omni = Omni(
        model=args.model_path,
        deploy_config=args.stage_config,
        stage_init_timeout=3600,
        init_timeout=3600,
    )

    try:
        for request_idx in range(args.repeat):
            start = time.perf_counter()

            # Build request with keyframes
            request = {
                "prompt": args.prompt,
                "multi_modal_data": {
                    "keyframes": keyframes,
                },
            }

            params = OmniDiffusionSamplingParams(
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed + request_idx,
            )

            # Generate
            result = omni.generate(request, params)[0]
            video = result.images[0]  # (1, T, H, W, C) uint8

            elapsed = time.perf_counter() - start
            print(f"Request {request_idx + 1}/{args.repeat}: {elapsed:.2f}s")

            if request_idx == 0:
                # Save first result
                frames = [Image.fromarray(frame) for frame in video[0]]
                export_to_video(frames, args.output, fps=24)
                print(f"Saved {video.shape[1]} frames to {args.output}")

    finally:
        omni.shutdown()


if __name__ == "__main__":
    main()
