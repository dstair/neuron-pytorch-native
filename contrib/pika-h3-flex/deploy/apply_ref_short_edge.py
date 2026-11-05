#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Make the ref2va image-reference short edge configurable.

The released model encodes image references at a 2048 short edge. On Neuron the device
encode of a 3584x2048 reference inside the decode swap overflows the Lite executor queue
("Failed to schedule neff execution: Execution Queue Full"), so the short edge becomes a
stage-config knob: ``reference_short_edge`` (default 2048, the released value). 1344 keeps
the encode in the proven keyframe tile-count regime.

Usage:
    python apply_ref_short_edge.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys


def apply(src: str) -> str:
    if "_reference_short_edge" in src:
        raise SystemExit("reference_short_edge already applied")

    anchor = '        self._compile_vae_encoder = bool(model_config.pop("compile_vae_encoder", False))\n'
    assert anchor in src, "init anchor not found"
    src = src.replace(
        anchor,
        anchor + '        # Ref2VA image-reference short edge (released: 2048). Lower it if the device\n'
        '        # encode of a large reference overflows the executor queue.\n'
        '        self._reference_short_edge = int(model_config.pop("reference_short_edge", 2048))\n',
        1,
    )

    old = "                        entry[\"image\"], VAE_SPATIAL_COMPRESSION * patch_size[2], 2048\n"
    assert old in src, "normalize anchor not found"
    src = src.replace(old, "                        entry[\"image\"], VAE_SPATIAL_COMPRESSION * patch_size[2], self._reference_short_edge\n", 1)
    return src


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    out = apply(src)
    with open(path, "w") as f:
        f.write(out)
    print(f"Applied reference_short_edge to {path}")


if __name__ == "__main__":
    main()
