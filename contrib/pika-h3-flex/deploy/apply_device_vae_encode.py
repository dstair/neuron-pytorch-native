#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device-side VAE encode for keyframe/reference conditioning.

The CPU encode of a conditioning image is the FL2VA/Ref2VA latency floor (~120s for a
768p keyframe, many minutes for a 2048-short-edge reference, per rank). This moves it
onto the accelerator:

- ``compile_vae_encoder: true`` in the stage config wraps the encoder as one compiled
  graph (the Neuron VAE's existing ``compile(compile_encoder=True)``).
- ``_encode_vae_condition`` runs inside the same DiT-blocks-out/VAE-in swap the decoder
  uses (``vae_swap_dit_blocks``), or directly when the VAE is already resident
  (no-swap config), with pixels on the device.

Usage:
    python apply_device_vae_encode.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys

OLD_COMPILE_VAE = '''    def compile_vae(self, *args, **kwargs):
        if self.vae is None:
            return
        options = {**kwargs.pop("options", {}), "model_name": "minimax_h3_vae"}
        options["compiler_args"] = VAE_COMPILER_ARGS
        # Decoder tiles use one fixed spatial shape.
        kwargs.setdefault("fullgraph", True)
        self.vae.compile(*args, options=options, **kwargs)
'''

NEW_COMPILE_VAE = '''    def compile_vae(self, *args, **kwargs):
        if self.vae is None:
            return
        options = {**kwargs.pop("options", {}), "model_name": "minimax_h3_vae"}
        options["compiler_args"] = VAE_COMPILER_ARGS
        # Decoder tiles use one fixed spatial shape.
        kwargs.setdefault("fullgraph", True)
        # Keyframe/reference conditioning: compile the encoder graph too.
        if self._compile_vae_encoder:
            kwargs["compile_encoder"] = True
        self.vae.compile(*args, options=options, **kwargs)
'''

OLD_ENCODE = '''        pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
        posterior = vae.encode(pixels, return_dict=False)[0]
        latents = posterior.sample(generator=torch.Generator().manual_seed(KEYFRAME_ENCODE_SEED))
'''

NEW_ENCODE = '''        pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
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


def apply(src: str) -> str:
    if "_compile_vae_encoder" in src:
        raise SystemExit("device vae encode already applied; refusing to re-apply")

    anchor = '        self._pika_step_list = model_config.pop("pika_denoising_step_list", None) or None\n'
    assert anchor in src, "init anchor not found"
    src = src.replace(
        anchor,
        anchor + '        # Device-side VAE encode for keyframe/reference conditioning (compile the\n'
        '        # encoder graph; runs inside the decode swap or directly when resident).\n'
        '        self._compile_vae_encoder = bool(model_config.pop("compile_vae_encoder", False))\n',
        1,
    )

    assert OLD_COMPILE_VAE in src, "compile_vae anchor not found"
    src = src.replace(OLD_COMPILE_VAE, NEW_COMPILE_VAE, 1)
    assert OLD_ENCODE in src, "encode anchor not found"
    src = src.replace(OLD_ENCODE, NEW_ENCODE, 1)
    return src


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    backup = path + ".pre_devenc"
    with open(backup, "w") as f:
        f.write(src)
    with open(path, "w") as f:
        f.write(apply(src))
    print(f"Applied device vae encode to {path} (backup: {backup})")


if __name__ == "__main__":
    main()
