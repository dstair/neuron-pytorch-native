#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fix: normalize conditioning pixels on the host, move to the device after.

Neuron tensors don't support on-device dtype casts ("A host-to-device copy cannot also
cast"), and with the no-swap config the VAE encoder lives on the device, so the uint8
pixels must be normalized on the host first. Applies to both the FL2VA keyframe path and
the Ref2VA image path.

Usage:
    python apply_devenc_fix.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys

OLD_PREP = '''        height = self._fl2va_height
        width = self._fl2va_width
        encoder_device = next(self.vae.encoder.parameters()).device
        condition_latents = []
        for image in keyframes:
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            pixels = torch.from_numpy(np.array(image)).to(encoder_device).permute(2, 0, 1)[None, :, None]
            latents = self._encode_vae_condition(pixels)
'''

NEW_PREP = '''        height = self._fl2va_height
        width = self._fl2va_width
        # Pixels stay on the host through normalization (device tensors can't cast);
        # _encode_vae_condition moves the float32 result to the encode device itself.
        condition_latents = []
        for image in keyframes:
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None, :, None]
            latents = self._encode_vae_condition(pixels)
'''

OLD_REF = '''                    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None, :, None]
                    encoder_device = next(self.vae.encoder.parameters()).device
                    latents = self._encode_vae_condition(pixels.to(encoder_device))
'''

NEW_REF = '''                    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None, :, None]
                    latents = self._encode_vae_condition(pixels)
'''


def apply(src: str) -> str:
    assert OLD_PREP in src, "prep anchor not found"
    src = src.replace(OLD_PREP, NEW_PREP, 1)
    assert OLD_REF in src, "ref anchor not found"
    src = src.replace(OLD_REF, NEW_REF, 1)
    return src


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    out = apply(src)
    with open(path, "w") as f:
        f.write(out)
    print(f"Applied devenc fix to {path}")


if __name__ == "__main__":
    main()
