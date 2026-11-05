#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pika-H3-Flex Ref2VA (reference-to-video+audio) runner for the Neuron pipeline.

References are passed via the request's ``multi_modal_data.references``:
  --image subject.png (repeatable, up to 9)   image references (subject/style/scene)
  --audio-track voice.wav (repeatable)        soundtrack references (lip-sync conditioning)

Image references are normalized to a 2048 short edge and VAE-encoded; soundtracks are
resampled to 32 kHz, upmixed to stereo, truncated to the video duration, and encoded
through the audio VAE (clean, t=1.0).

Usage (on the box):
    python run_ref2va.py \
        --stage-config /data/omni29/pika_h3_flex_ref2va_stage_tp8.yaml \
        --image subject.png --prompt "..." --output ref2va_out.mp4
    python run_ref2va.py \
        --stage-config /data/omni29/pika_h3_flex_ref2va_stage_tp8.yaml \
        --image first_frame.png --audio-track voice.wav --prompt "..." --output lipsync.mp4
"""

import argparse
import os
import sys
import time
import wave

import numpy as np
import yaml

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
import torch
from diffusers.utils import export_to_video
from PIL import Image
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from vllm_omni_neuron import env_profiles

parser = argparse.ArgumentParser(description="Pika-H3-Flex Ref2VA on Neuron")
parser.add_argument("--model-path", type=str, default="/data/models/Pika-H3-Flex-Neuron")
parser.add_argument("--stage-config", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--image", type=str, action="append", default=[], help="image reference (repeatable)")
parser.add_argument("--audio-track", type=str, action="append", default=[], help="WAV soundtrack reference (repeatable)")
parser.add_argument("--height", type=int, default=768)
parser.add_argument("--width", type=int, default=1344)
parser.add_argument("--num-frames", type=int, default=124)
parser.add_argument("--num-inference-steps", type=int, default=6)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, default="ref2va_output.mp4")
parser.add_argument("--repeat", type=int, default=1)
args = parser.parse_args()

with open(args.stage_config) as handle:
    _stage = yaml.safe_load(handle)["stages"][0]
env_profiles.apply(
    env_profiles.WAN22_T2V,
    env_profiles.thread_limits(_stage["engine_args"]["parallel_config"]["tensor_parallel_size"]),
)


def load_wav(path: str):
    with wave.open(path, "rb") as handle:
        channels, width, rate, nframes = (
            handle.getnchannels(),
            handle.getsampwidth(),
            handle.getframerate(),
            handle.getnframes(),
        )
        if width != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported, got sampwidth {width}")
        pcm = np.frombuffer(handle.readframes(nframes), dtype="<i2").astype(np.float32) / 32768.0
    waveform = torch.from_numpy(pcm.reshape(-1, channels).T.copy())
    return waveform, rate


def main():
    references = []
    for path in args.image:
        references.append({"kind": "image", "image": Image.open(path).convert("RGB")})
    for path in args.audio_track:
        waveform, rate = load_wav(path)
        references.append({"kind": "audio", "audio": waveform, "sample_rate": rate})
    if not references:
        print("ERROR: provide --image and/or --audio-track", file=sys.stderr)
        sys.exit(1)

    kinds = [entry["kind"] for entry in references]
    print(f"Ref2VA: references {kinds}, {args.width}x{args.height}, {args.num_frames} frames, seed {args.seed}")

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
                {"prompt": args.prompt, "multi_modal_data": {"references": references}}, params
            )[0].images[0]
            print(f"request {request + 1}/{args.repeat}: {time.perf_counter() - start:.2f} s")
        export_to_video([Image.fromarray(frame) for frame in video[0]], args.output, fps=24)
        print(f"Saved {video.shape[1]} frames of {video.shape[3]}x{video.shape[2]} to {args.output}")
    finally:
        omni.shutdown()


if __name__ == "__main__":
    main()
