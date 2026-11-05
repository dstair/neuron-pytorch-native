#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pipeline-side production-shape warmup at engine boot.

The 15s CP=8 profile showed the first request's step 0 costs ~225s — the per-shape
NEFF/executor load — while later steps run ~6s. The engine's own dummy warmup is fixed at
512x512, which warms none of the production shapes. This patch runs one full synthetic
forward (T2V, seed 0) at the configured canvas/frame count at the end of
``pipeline.compile()`` — weights resident, graphs wrapped — so NEFF compile/load and the
audio/swap paths all warm before the engine serves.

Config (model_config):
  warmup_num_frames: 362   # 0 (default) disables; snapped to 17n+5 like a real request
  warmup_canvas: [768, 1344]  # optional [height, width]

Usage:
    python apply_warmup.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys

WARMUP_METHOD = '''
    def _warmup_forward(self):
        """One synthetic T2V request at the configured shape, run on every rank at boot.

        Exercises text encode, the DiT graphs at the production sequence shape (the
        ~225 s NEFF load), the decode swap and the audio decode, so the first served
        request runs at steady-state. Off when ``warmup_num_frames`` is 0.
        """
        if self._warmup_num_frames <= 0:
            return
        import types

        t0 = time.perf_counter()
        height, width = self._warmup_canvas
        params = types.SimpleNamespace(
            height=height,
            width=width,
            num_frames=self._warmup_num_frames,
            num_inference_steps=None,
            num_outputs_per_prompt=1,
            seed=0,
        )
        req = types.SimpleNamespace(
            num_reqs=1,
            sampling_params_list=[params],
            prompts=["warmup"],
            collate_request_generators=lambda n, g: [None],
        )
        logger.info(
            "Warmup: synthetic %dx%dx%d forward (NEFF compile/load before serving)...",
            height,
            width,
            self._warmup_num_frames,
        )
        self.forward(req)
        logger.info("Warmup: done in %.1fs", time.perf_counter() - t0)

'''

OLD_COMPILE = '''    def compile(self, *args, **kwargs):
        """Compile the text encoder, VAE and DiT for Neuron."""
        self.compile_text_encoder(*args, **kwargs)
        self.compile_vae(*args, **kwargs)
        self.compile_transformer(*args, **kwargs)
        return self
'''

NEW_COMPILE = '''    def compile(self, *args, **kwargs):
        """Compile the text encoder, VAE and DiT for Neuron."""
        self.compile_text_encoder(*args, **kwargs)
        self.compile_vae(*args, **kwargs)
        self.compile_transformer(*args, **kwargs)
        self._warmup_forward()
        return self
'''


def apply(src: str) -> str:
    if "_warmup_forward" in src:
        raise SystemExit("warmup already applied")

    anchor = '        self._device_encode_max_pixels = int(model_config.pop("device_encode_max_pixels", 1400 * 1400))\n'
    assert anchor in src, "init anchor not found"
    src = src.replace(
        anchor,
        anchor + '        # Boot-time warmup: one synthetic forward at this frame count (0 = off).\n'
        '        self._warmup_num_frames = int(model_config.pop("warmup_num_frames", 0) or 0)\n'
        '        self._warmup_canvas = tuple(model_config.pop("warmup_canvas", [768, 1344]))\n',
        1,
    )

    anchor = "    def compile(self, *args, **kwargs):"
    assert anchor in src
    src = src.replace(anchor, WARMUP_METHOD + anchor, 1)
    assert OLD_COMPILE in src, "compile anchor not found"
    src = src.replace(OLD_COMPILE, NEW_COMPILE, 1)
    return src


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    out = apply(src)
    with open(path, "w") as f:
        f.write(out)
    print(f"Applied warmup to {path}")


if __name__ == "__main__":
    main()
