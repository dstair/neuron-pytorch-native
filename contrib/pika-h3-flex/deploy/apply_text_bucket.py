#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pad the packed text segment to a fixed bucket so every prompt shares one DiT shape.

The Neuron DiT compiles a static graph per packed-sequence length, which varies with the
prompt's token count: every new length pays a graph compile (~20 min cold) and every boot
pays a ~225 s NEFF load per shape. Padding the text rows to ``text_seq_bucket`` (default
64, 0 = off) makes the sequence length prompt-independent, so one warmup covers all
requests and a length is only ever compiled once.

Pad rows go after the real text: token tag = text (1), position ids continue arange,
embeddings zero. The DiT has no attention mask (full self-attention), so padded output
differs slightly from unpadded — A/B-checked; deterministic per prompt either way.

Usage:
    python apply_text_bucket.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys

OLD = '''        t0 = time.perf_counter()
        prompt_embeds = self.encode_prompt(prompt)
        text_encode_seconds = time.perf_counter() - t0

        self._fl2va_height, self._fl2va_width = height, width
        text_token_tags = torch.full((prompt_embeds.shape[1],), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
'''

NEW = '''        t0 = time.perf_counter()
        prompt_embeds = self.encode_prompt(prompt)
        text_encode_seconds = time.perf_counter() - t0

        self._fl2va_height, self._fl2va_width = height, width
        num_text = prompt_embeds.shape[1]
        if self._text_seq_bucket:
            # Pad the text segment to the bucket so every prompt shares one DiT graph
            # shape (pad rows: text tag, zero embeddings, continuing position ids).
            num_text = -(-num_text // self._text_seq_bucket) * self._text_seq_bucket
            if num_text != prompt_embeds.shape[1]:
                prompt_embeds = torch.cat(
                    [prompt_embeds, prompt_embeds.new_zeros(1, num_text - prompt_embeds.shape[1], prompt_embeds.shape[2])],
                    dim=1,
                )
        text_token_tags = torch.full((num_text,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
'''


def apply(src: str) -> str:
    if "_text_seq_bucket" in src:
        raise SystemExit("text bucket already applied")

    anchor = '        self._warmup_canvas = tuple(model_config.pop("warmup_canvas", [768, 1344]))\n'
    assert anchor in src, "init anchor not found"
    src = src.replace(
        anchor,
        anchor + '        # Pad the packed text segment to a multiple of this many rows (0 = off).\n'
        '        self._text_seq_bucket = int(model_config.pop("text_seq_bucket", 0) or 0)\n',
        1,
    )
    assert OLD in src, "layout anchor not found"
    src = src.replace(OLD, NEW, 1)
    return src


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    out = apply(src)
    with open(path, "w") as f:
        f.write(out)
    print(f"Applied text bucket to {path}")


if __name__ == "__main__":
    main()
