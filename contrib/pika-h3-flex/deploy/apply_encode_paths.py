#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Final conditioning-encode routing: per-rank upstream-CPU fallback, device for small.

Routing in ``_encode_vae_condition``:
- small images (<= device_encode_max_pixels): device encode inside the decode swap,
  ALL ranks in lockstep (the distributed VAE executor is collective — rank-0-only hangs
  it, which is what killed the lip-sync run);
- large images: every rank runs the *upstream* tiled CPU encode (bypassing the
  Neuron/distributed ``_encode_clip`` overrides, so there are no collectives), threads
  capped at cores//ranks so 8 concurrent encodes can't OOM the host.

The audio-reference encode keeps the rank-0 + cpu_group broadcast (the audio VAE is not
distributed, so a single encode is safe and it is tiny).

Usage:
    python apply_encode_paths.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys

OLD = '''        latents = None
        if self._tp_is_src_rank():
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
                # big references run on CPU with the machine's cores (one encode, then
                # broadcast, so the full thread budget is safe to use).
                # 192 threads x MKL buffers OOM-killed rank 0 with 8 workers resident; cap it.
                torch.set_num_threads(min(32, max(1, len(os.sched_getaffinity(0)))))
                try:
                    posterior = vae.encode(pixels, return_dict=False)[0]
                finally:
                    torch.set_num_threads(1)
            latents = posterior.sample(generator=torch.Generator().manual_seed(KEYFRAME_ENCODE_SEED))
            latents = latents.to(torch.float16).float().cpu()
        latents = self._tp_broadcast(latents)
        return (latents - latents_mean) / latents_std
'''

NEW = '''        num_pixels = pixels.shape[-1] * pixels.shape[-2]
        if self._compile_vae_encoder and num_pixels <= self._device_encode_max_pixels:
            # Device-side encode: the compiled encoder graph needs the module resident;
            # with the swap config the VAE steps in (DiT's last blocks step out), as for
            # the decode. Every rank encodes in lockstep: the distributed VAE executor
            # is collective, and a subset of ranks calling it hangs the group.
            if self._vae_swap_dit_blocks:
                with self._swap_for_decode():
                    posterior = vae.encode(pixels.to(self.device), return_dict=False)[0]
            else:
                posterior = vae.encode(pixels.to(self.device), return_dict=False)[0]
        else:
            # Host encode of a big reference: run upstream's tiled CPU encode on every
            # rank (deterministic), bypassing the Neuron/distributed ``_encode_clip``
            # overrides so there are no collectives to lockstep. Threads are capped at
            # the rank's share of the cores so concurrent encodes can't OOM the host.
            from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
                AutoencoderKLMiniMaxH3 as _UpstreamVAE,
            )
            from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

            ranks = max(1, self.od_config.parallel_config.tensor_parallel_size)
            previous = torch.get_num_threads()
            torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) // ranks))
            try:
                moments = _UpstreamVAE._encode(vae, pixels)
            finally:
                torch.set_num_threads(previous)
            posterior = DiagonalGaussianDistribution(moments)
        latents = posterior.sample(generator=torch.Generator().manual_seed(KEYFRAME_ENCODE_SEED))
        latents = latents.to(torch.float16).float().cpu()
        return (latents - latents_mean) / latents_std
'''


def apply(src: str) -> str:
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
    print(f"Applied encode paths to {path}")


if __name__ == "__main__":
    main()
