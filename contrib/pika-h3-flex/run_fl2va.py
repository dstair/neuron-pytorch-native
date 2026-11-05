#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pika-H3-Flex FL2VA (first/last-frame image-to-video) runner for the Neuron pipeline.

The pipeline (FL2VA-patched) reads keyframes from the request's
``multi_modal_data.keyframes``: the first image anchors the first video frame, a
second image anchors the last. Anchors are VAE-encoded (ImageNet norm, seed-42
posterior, fp16 rounding), noised to t=0.999, and frozen through the 6-NFE renoise
loop.

Usage (on the box):
    python run_fl2va.py \
        --stage-config /data/omni29/pika_h3_flex_stage_tp8.yaml \
        --first-frame keyframe.png \
        --prompt "A sun conure parrot springs off a pale perch ..." \
        --output fl2va_out.mp4
"""

import argparse
import os
import sys
import time

import yaml

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
from diffusers.utils import export_to_video
from PIL import Image
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from vllm_omni_neuron import env_profiles

parser = argparse.ArgumentParser(description="Pika-H3-Flex FL2VA on Neuron")
parser.add_argument("--model-path", type=str, default="/data/models/Pika-H3-Flex-Neuron")
parser.add_argument("--stage-config", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--first-frame", type=str, default=None, help="first-frame keyframe image")
parser.add_argument("--last-frame", type=str, default=None, help="last-frame keyframe image")
parser.add_argument("--height", type=int, default=768)
parser.add_argument("--width", type=int, default=1344)
parser.add_argument("--num-frames", type=int, default=124)
parser.add_argument("--num-inference-steps", type=int, default=6, help="Pika uses a fixed 6-NFE schedule")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, default="fl2va_output.mp4")
parser.add_argument("--repeat", type=int, default=1)
args = parser.parse_args()

with open(args.stage_config) as handle:
    _stage = yaml.safe_load(handle)["stages"][0]
env_profiles.apply(
    env_profiles.WAN22_T2V,
    env_profiles.thread_limits(_stage["engine_args"]["parallel_config"]["tensor_parallel_size"]),
)


def main():
    keyframes = []
    if args.first_frame:
        keyframes.append(Image.open(args.first_frame).convert("RGB"))
    if args.last_frame:
        keyframes.append(Image.open(args.last_frame).convert("RGB"))
    if not keyframes:
        print("ERROR: provide --first-frame and/or --last-frame", file=sys.stderr)
        sys.exit(1)

    print(
        f"FL2VA: {len(keyframes)} keyframe(s) ({'first' if args.first_frame else ''}"
        f"{'+last' if args.last_frame else ''}), {args.width}x{args.height}, "
        f"{args.num_frames} frames, seed {args.seed}"
    )

    omni = Omni(
        model=args.model_path,
        deploy_config=args.stage_config,
        stage_init_timeout=3600,
        init_timeout=3600,
    )
    params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
    )
    try:
        for request in range(args.repeat):
            start = time.perf_counter()
            video = omni.generate(
                {"prompt": args.prompt, "multi_modal_data": {"keyframes": keyframes}}, params
            )[0].images[0]
            print(f"request {request + 1}/{args.repeat}: {time.perf_counter() - start:.2f} s")
        export_to_video([Image.fromarray(frame) for frame in video[0]], args.output, fps=24)
        print(f"Saved {video.shape[1]} frames of {video.shape[3]}x{video.shape[2]} to {args.output}")
    finally:
        omni.shutdown()


if __name__ == "__main__":
    main()
