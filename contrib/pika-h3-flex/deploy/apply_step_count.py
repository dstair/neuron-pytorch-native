#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Configurable Pika renoise step counts (1/2/3/6) for pipeline_minimax_h3.py.

The released adapter was distilled on the 6-knot grid 1000/750/500/250/100/40/0. Shorter
schedules take every n-th knot of it, so each step still lands on a trained sigma:
3 steps -> [1000, 500, 100, 0]; 2 -> [1000, 250, 0]; 1 -> [1000, 0]. The stage config's
``pika_denoising_step_list`` overrides outright. ``num_inference_steps`` on the request
selects per request; unset stays at 6 for pika_renoise.

Usage:
    python apply_step_count.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys

KNOTS_METHOD = '''
    def _pika_step_knots(self, num_steps: int) -> list:
        """Knots (1000-scale, descending to 0) for a Pika renoise schedule of ``num_steps``.

        Shorter schedules subsample the released 6-step grid so every step lands on a
        trained sigma. ``pika_denoising_step_list`` in the stage config overrides.
        """
        if self._pika_step_list:
            return list(self._pika_step_list)
        base = [1000, 750, 500, 250, 100, 40, 0]
        if num_steps == 6:
            return base
        if num_steps == 3:
            return base[::2]
        if num_steps == 2:
            return base[::3]
        if num_steps == 1:
            return [1000, 0]
        raise ValueError(f"Pika renoise supports 1/2/3/6 steps, got {num_steps}")

'''


def apply(src: str) -> str:
    if "_pika_step_knots" in src:
        raise SystemExit("step count already applied; refusing to re-apply")

    # 1. __init__ override list.
    anchor = '        self._pika_renoise = bool(model_config.pop("pika_renoise", False))\n'
    assert anchor in src
    src = src.replace(
        anchor,
        anchor + '        # Pika renoise schedule override (list of 1000-scale knots); default 6-step grid.\n'
        '        self._pika_step_list = model_config.pop("pika_denoising_step_list", None) or None\n',
        1,
    )

    # 2. _pika_step_knots before _pika_shifted_sigmas.
    anchor = "    @staticmethod\n    def _pika_shifted_sigmas"
    assert anchor in src
    src = src.replace(anchor, KNOTS_METHOD + anchor, 1)

    # 3. _pika_shifted_sigmas takes knots (signature + body).
    old_sig = "    def _pika_shifted_sigmas(shift: float) -> torch.Tensor:"
    assert old_sig in src
    src = src.replace(old_sig, "    def _pika_shifted_sigmas(shift: float, knots=None) -> torch.Tensor:", 1)
    old = '''        base = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.1, 0.04, 0.0], dtype=torch.float32)
        return shift * base / (1.0 + (shift - 1.0) * base)'''
    new = '''        if knots is None:
            knots = [1000, 750, 500, 250, 100, 40, 0]
        base = torch.tensor([k / 1000.0 for k in knots], dtype=torch.float32)
        return shift * base / (1.0 + (shift - 1.0) * base)'''
    assert old in src
    src = src.replace(old, new, 1)

    # 4. _denoise_pika signature + knots.
    old = "    def _denoise_pika(self, prompt_embeds, layout, video_rows, audio_rows, generator):"
    assert old in src
    src = src.replace(old, "    def _denoise_pika(self, prompt_embeds, layout, video_rows, audio_rows, generator, num_steps=6):", 1)
    old = '''        sig_v = self._pika_shifted_sigmas(float(self.scheduler.config.shift))
        sig_a = self._pika_shifted_sigmas(float(self.audio_scheduler.config.shift))'''
    new = '''        knots = self._pika_step_knots(num_steps)
        sig_v = self._pika_shifted_sigmas(float(self.scheduler.config.shift), knots)
        sig_a = self._pika_shifted_sigmas(float(self.audio_scheduler.config.shift), knots)'''
    assert old in src
    src = src.replace(old, new, 1)

    # 5. Dispatcher passes num_steps.
    old = "            return self._denoise_pika(prompt_embeds, layout, video_rows, audio_rows, generator)"
    assert old in src
    src = src.replace(old, "            return self._denoise_pika(prompt_embeds, layout, video_rows, audio_rows, generator, num_steps=num_steps)", 1)

    # 6. forward(): pika defaults to 6 steps when unset.
    old = "        num_steps = params.num_inference_steps or DEFAULT_NUM_STEPS\n"
    assert old in src
    src = src.replace(old, "        num_steps = params.num_inference_steps or (6 if self._pika_renoise else DEFAULT_NUM_STEPS)\n", 1)

    return src


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    backup = path + ".pre_steps"
    with open(backup, "w") as f:
        f.write(src)
    with open(path, "w") as f:
        f.write(apply(src))
    print(f"Applied step count to {path} (backup: {backup})")


if __name__ == "__main__":
    main()
