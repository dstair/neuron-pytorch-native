#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Encode conditioning once on TP rank 0 and broadcast, instead of per-rank encodes.

The conditioning encodes (keyframe/reference VAE encode, audio-reference audio-VAE
encode) are deterministic (seed-42 posterior sample / posterior mode), so every TP rank
encoding its own copy just multiplies host memory and CPU pressure — 8 ranks x threaded
FP32 conv encodes OOM-killed a worker. Rank 0 encodes (device path when small enough,
host otherwise), then the latents go out over ``get_tp_group().cpu_group`` (shape header
+ data, so non-encoding ranks don't derive shapes).

Usage:
    python apply_encode_broadcast.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys

HELPERS = '''
    def _tp_broadcast(self, tensor) -> torch.Tensor:
        """Broadcast ``tensor`` from TP rank 0 over the CPU group (shape header + data).

        Non-source ranks pass None and receive the tensor. No-op at TP=1.
        """
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group().cpu_group
        if group is None or dist.get_world_size(group) == 1:
            return tensor
        header = torch.zeros(5, dtype=torch.long)  # [ndim, d0..d3]
        if dist.get_rank(group) == 0:
            header[0] = tensor.ndim
            header[1 : 1 + tensor.ndim] = torch.tensor(tensor.shape, dtype=torch.long)
        dist.broadcast(header, src=0, group=group)
        if dist.get_rank(group) != 0:
            ndim = int(header[0])
            tensor = torch.empty(tuple(header[1 : 1 + ndim].tolist()), dtype=torch.float32)
        dist.broadcast(tensor, src=0, group=group)
        return tensor

    def _tp_is_src_rank(self) -> bool:
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group().cpu_group
        return group is None or dist.get_world_size(group) == 1 or dist.get_rank(group) == 0

'''

OLD_VAE_ENC = '''        num_pixels = pixels.shape[-1] * pixels.shape[-2]
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
        latents = latents.to(torch.float16).float().cpu()
        return (latents - latents_mean) / latents_std
'''

NEW_VAE_ENC = '''        latents = None
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

OLD_AUDIO_ENC = '''        posterior = self.audio_vae.encode(waveform[:, None], return_dict=False)[0]
        latents = posterior.mode().float().cpu().transpose(1, 2)
        mean = torch.tensor(self.audio_vae.config.latents_mean).view(1, 1, -1)
        std = torch.tensor(self.audio_vae.config.latents_std).view(1, 1, -1)
        normalized = (latents - mean) / std
        return normalized.reshape(-1, self.transformer.config.audio_in_channels)
'''

NEW_AUDIO_ENC = '''        latents = None
        if self._tp_is_src_rank():
            posterior = self.audio_vae.encode(waveform[:, None], return_dict=False)[0]
            latents = posterior.mode().float().cpu().transpose(1, 2).contiguous()
        latents = self._tp_broadcast(latents)
        mean = torch.tensor(self.audio_vae.config.latents_mean).view(1, 1, -1)
        std = torch.tensor(self.audio_vae.config.latents_std).view(1, 1, -1)
        normalized = (latents - mean) / std
        return normalized.reshape(-1, self.transformer.config.audio_in_channels)
'''


def apply(src: str) -> str:
    if "_tp_broadcast" in src:
        raise SystemExit("encode broadcast already applied")

    anchor = "    def _encode_vae_condition(self, pixels: torch.Tensor) -> torch.Tensor:"
    assert anchor in src, "encode anchor not found"
    src = src.replace(anchor, HELPERS + anchor, 1)
    assert OLD_VAE_ENC in src, "vae enc anchor not found"
    src = src.replace(OLD_VAE_ENC, NEW_VAE_ENC, 1)
    assert OLD_AUDIO_ENC in src, "audio enc anchor not found"
    src = src.replace(OLD_AUDIO_ENC, NEW_AUDIO_ENC, 1)
    return src


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    out = apply(src)
    with open(path, "w") as f:
        f.write(out)
    print(f"Applied encode broadcast to {path}")


if __name__ == "__main__":
    main()
