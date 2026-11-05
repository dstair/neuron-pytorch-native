#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pixel-area-gated CPU fallback for the device VAE encode.

The device encode inside the decode swap overflows the Lite executor queue
("Execution Queue Full") past roughly the canvas tile count (~1 Mpx). Conditioning
images larger than ``device_encode_max_pixels`` (default 1.4 Mpx, between the 768p
keyframe's 1.03 Mpx and a 2352x1344 reference's 3.15 Mpx) encode on the host instead,
with the rank's share of the machine's cores (the workers otherwise run one intra-op
thread, which is what made the 2048-short-edge CPU encode take ~17 min).

Usage:
    python apply_encode_fallback.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys

OLD = '''        pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
        if self._compile_vae_encoder:
            # Device-side encode: the compiled encoder graph needs the module resident;
            # with the swap config the VAE steps in (DiT's last blocks step out), as for
            # the decode. Without a swap it is already on the device.
            if self._vae_swap_dit_blocks:
                with self._swap_for_decode():
                    posterior = vae.encode(pixels.to(self.device), return_dict=False)[0]
            else:
                posterior = vae.encode(pixels.to(self.device), return_dict=False)[0]
        else:
            posterior = vae.encode(pixels, return_dict=False)[0]
        latents = posterior.sample(generator=torch.Generator().manual_seed(KEYFRAME_ENCODE_SEED))
'''

NEW = '''        pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
        num_pixels = pixels.shape[-1] * pixels.shape[-2]
        if self._compile_vae_encoder and num_pixels <= self._device_encode_max_pixels:
            # Device-side encode: the compiled encoder graph needs the module resident;
            # with the swap config the VAE steps in (DiT's last blocks step out), as for
            # the decode. Without a swap it is already on the device.
            if self._vae_swap_dit_blocks:
                with self._swap_for_decode():
                    posterior = vae.encode(pixels.to(self.device), return_dict=False)[0]
            else:
                posterior = vae.encode(pixels.to(self.device), return_dict=False)[0]
        else:
            # Host encode: past the executor-queue ceiling the device encode fails, so
            # big references run on CPU with the rank's share of the cores (the workers
            # otherwise run one intra-op thread).
            import vllm_omni.diffusion.distributed.utils as _dist_utils

            ranks = max(1, getattr(getattr(self, "od_config", None), "parallel_config", None) and self.od_config.parallel_config.tensor_parallel_size or 1)
            budget = max(1, len(os.sched_getaffinity(0)) // ranks)
            previous = torch.get_num_threads()
            torch.set_num_threads(budget)
            try:
                posterior = vae.encode(pixels, return_dict=False)[0]
            finally:
                torch.set_num_threads(previous)
        latents = posterior.sample(generator=torch.Generator().manual_seed(KEYFRAME_ENCODE_SEED))
'''


def apply(src: str) -> str:
    if "_device_encode_max_pixels" in src:
        raise SystemExit("encode fallback already applied")

    anchor = '        self._reference_short_edge = int(model_config.pop("reference_short_edge", 2048))\n'
    assert anchor in src, "init anchor not found"
    src = src.replace(
        anchor,
        anchor + '        # Past this many pixels the conditioning image encodes on the host (the device\n'
        '        # encode inside the swap overflows the executor queue).\n'
        '        self._device_encode_max_pixels = int(model_config.pop("device_encode_max_pixels", 1400 * 1400))\n',
        1,
    )
    assert OLD in src, "encode anchor not found"
    src = src.replace(OLD, NEW, 1)
    return src


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    out = apply(src)
    with open(path, "w") as f:
        f.write(out)
    print(f"Applied encode fallback to {path}")


if __name__ == "__main__":
    main()
