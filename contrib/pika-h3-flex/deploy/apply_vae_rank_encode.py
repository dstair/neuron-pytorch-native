#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fix keyframe/reference VAE encode at CP>1 (VAE exists only on the first vae_pp ranks).

At TP=8xCP=8 (64 ranks, vae_patch_parallel_size=16), ranks 16+ hold no VAE and crashed in
``_encode_vae_condition`` ('NoneType' has no attribute 'config'). Now:

- VAE ranks (0..vae_pp-1) run the encode in lockstep (the distributed executor is
  collective within that group), normalize the latents there;
- rank 0 then broadcasts the normalized latents to every rank over a dedicated world gloo
  group created at init (CPU tensors cannot use the device collectives).

Usage:
    python apply_vae_rank_encode.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys

GROUP_INIT = '''        # World CPU group for broadcasting conditioning latents from the VAE ranks
        # (ranks >= vae_patch_parallel_size hold no VAE; all ranks call new_group).
        self._world_cpu_group = (
            dist.new_group(ranks=list(range(dist.get_world_size())), backend="gloo")
            if dist.is_initialized() and dist.get_world_size() > 1
            else None
        )
'''

BCAST_METHOD = '''
    def _conditioning_broadcast(self, tensor) -> torch.Tensor:
        """Broadcast normalized conditioning latents from rank 0 over the world gloo group."""
        group = self._world_cpu_group
        if group is None:
            return tensor
        header = torch.zeros(6, dtype=torch.long)
        if dist.get_rank() == 0:
            header[0] = tensor.ndim
            header[1 : 1 + tensor.ndim] = torch.tensor(tensor.shape, dtype=torch.long)
        dist.broadcast(header, src=0, group=group)
        if dist.get_rank() != 0:
            ndim = int(header[0])
            tensor = torch.empty(tuple(header[1 : 1 + ndim].tolist()), dtype=torch.float32)
        dist.broadcast(tensor, src=0, group=group)
        return tensor

'''

OLD_ENCODE_HEAD = '''        vae = self.vae
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(PIXEL_MEAN, device=pixels.device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(PIXEL_STD, device=pixels.device).view(1, -1, 1, 1, 1)
'''

NEW_ENCODE_HEAD = '''        vae = self.vae
        if vae is not None:
            latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
            latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(PIXEL_MEAN, device=pixels.device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(PIXEL_STD, device=pixels.device).view(1, -1, 1, 1, 1)
'''

OLD_ENCODE_TAIL = '''        num_pixels = pixels.shape[-1] * pixels.shape[-2]
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

NEW_ENCODE_TAIL = '''        latents = None
        if vae is not None:
            num_pixels = pixels.shape[-1] * pixels.shape[-2]
            if self._compile_vae_encoder and num_pixels <= self._device_encode_max_pixels:
                # Device-side encode: the compiled encoder graph needs the module resident;
                # with the swap config the VAE steps in (DiT's last blocks step out), as for
                # the decode. Every VAE rank encodes in lockstep: the distributed VAE
                # executor is collective, and a subset of ranks calling it hangs the group.
                if self._vae_swap_dit_blocks:
                    with self._swap_for_decode():
                        posterior = vae.encode(pixels.to(self.device), return_dict=False)[0]
                else:
                    posterior = vae.encode(pixels.to(self.device), return_dict=False)[0]
            else:
                # Host encode of a big reference: upstream's tiled CPU encode, bypassing
                # the Neuron/distributed ``_encode_clip`` overrides so there are no
                # collectives to lockstep. Threads are capped at the rank's share of the
                # cores so concurrent encodes can't OOM the host.
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
            latents = (latents - latents_mean) / latents_std
        # Ranks without a VAE (>= vae_patch_parallel_size) receive the normalized latents.
        return self._conditioning_broadcast(latents)
'''


def apply(src: str) -> str:
    if "_conditioning_broadcast" in src:
        raise SystemExit("vae rank encode already applied")

    anchor = '        self._text_seq_bucket = int(model_config.pop("text_seq_bucket", 0) or 0)\n'
    assert anchor in src, "init anchor not found"
    src = src.replace(anchor, anchor + GROUP_INIT, 1)

    anchor = "    def _encode_vae_condition(self, pixels: torch.Tensor) -> torch.Tensor:"
    assert anchor in src, "method anchor not found"
    src = src.replace(anchor, BCAST_METHOD + anchor, 1)

    assert OLD_ENCODE_HEAD in src, "encode head anchor not found"
    src = src.replace(OLD_ENCODE_HEAD, NEW_ENCODE_HEAD, 1)
    assert OLD_ENCODE_TAIL in src, "encode tail anchor not found"
    src = src.replace(OLD_ENCODE_TAIL, NEW_ENCODE_TAIL, 1)
    return src


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    out = apply(src)
    with open(path, "w") as f:
        f.write(out)
    print(f"Applied vae rank encode to {path}")


if __name__ == "__main__":
    main()
