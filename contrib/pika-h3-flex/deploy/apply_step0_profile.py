#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile the pika renoise step-0 warmup: time AdaLN vs transformer forward separately.

Adds a one-line-per-step log to _denoise_pika:
    pika step i: adaln X.XXs, forward Y.YYs

Usage:
    python apply_step0_profile.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys

OLD = '''            block_modulations = tuple(
                m.to(self.device) for m in self.block_modulations(timestep)
            )
            noise_pred, audio_noise_pred = self.transformer(
                hidden_states=video_rows[None].to(self.dtype).to(self.device),
                audio_hidden_states=audio_rows[None].to(self.dtype).to(self.device),
                timestep=timestep,
                block_modulations=block_modulations,
                return_dict=False,
                **dit_inputs,
            )
            # Renoise (stochastic_sampling): x0 = latents + sigma_i * v, then re-noise to sigma_{i+1}.
'''

NEW = '''            t_adaln = time.perf_counter()
            block_modulations = tuple(
                m.to(self.device) for m in self.block_modulations(timestep)
            )
            adaln_seconds = time.perf_counter() - t_adaln
            t_fwd = time.perf_counter()
            noise_pred, audio_noise_pred = self.transformer(
                hidden_states=video_rows[None].to(self.dtype).to(self.device),
                audio_hidden_states=audio_rows[None].to(self.dtype).to(self.device),
                timestep=timestep,
                block_modulations=block_modulations,
                return_dict=False,
                **dit_inputs,
            )
            fwd_seconds = time.perf_counter() - t_fwd
            if self.is_output_rank:
                logger.info(
                    "pika step %d: adaln %.2fs, forward %.2fs", i, adaln_seconds, fwd_seconds
                )
            # Renoise (stochastic_sampling): x0 = latents + sigma_i * v, then re-noise to sigma_{i+1}.
'''


def apply(src: str) -> str:
    if "adaln_seconds" in src:
        raise SystemExit("step0 profile already applied")
    assert OLD in src, "step anchor not found"
    return src.replace(OLD, NEW, 1)


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    out = apply(src)
    with open(path, "w") as f:
        f.write(out)
    print(f"Applied step0 profile to {path}")


if __name__ == "__main__":
    main()
