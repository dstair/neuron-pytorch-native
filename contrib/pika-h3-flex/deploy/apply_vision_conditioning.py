#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Vision-tower conditioning for keyframes/references in the text encoder.

The Neuron Qwen3-VL already implements the vision tower + splice
(``forward(..., pixel_values, image_grid_thw)``); this wires the Pika presentation into
the pipeline: per visual condition a ``"<Picture i>: "`` label and a vision block
(``<|vision_start|> <|image_pad|>*N <|vision_end|>``), then the prompt verbatim — with the
vision rows tagged **video** (0) in ``text_token_tags``, as the released model's AdaLN
keys off. Audio references get a text-only ``"<Audio j>: "`` label.

The presentation runs through the Qwen3-VL image processor at the image's own resolution
(merge_size**2 = 4 rows per merged block), padded to the text bucket like the T2V path.

Usage:
    python apply_vision_conditioning.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py
"""

import sys

PROCESSOR_INIT = '''        # The Qwen3-VL processor drives the vision-block presentation of keyframes and
        # image references (image_processor -> pixel_values / image_grid_thw).
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model, subfolder="processor", local_files_only=local_files_only
        )
'''

VISION_METHOD = '''
    def _encode_prompt_with_vision(self, prompt: str, images: list, audio_count: int = 0):
        """Encode the FL2VA/Ref2VA presentation: ``<Picture i>: `` + vision block per
        image (and ``<Audio j>: `` labels), then the prompt verbatim.

        Returns ``(prompt_embeds, text_token_tags)`` — the vision rows carry the video
        tag, everything else the text tag.
        """
        tokenizer, processor = self.tokenizer, self.processor
        text_tag, video_tag = MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG

        vision_inputs = {}
        if images:
            vision = processor.image_processor(images=images, return_tensors="pt")
            vision_inputs = {
                "pixel_values": vision["pixel_values"],
                "image_grid_thw": vision["image_grid_thw"],
            }
        token_ids, token_tags = [], []
        merge = processor.image_processor.merge_size**2
        for index, _ in enumerate(audio_count and range(audio_count) or []):
            label_ids = tokenizer(f"<Audio {index + 1}>: ", add_special_tokens=False)["input_ids"]
            token_ids += label_ids
            token_tags += [text_tag] * len(label_ids)
        for index in range(len(images)):
            num_image_tokens = int(vision_inputs["image_grid_thw"][index].prod()) // merge
            label_ids = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
            vision_ids = (
                [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            token_ids += label_ids + vision_ids
            token_tags += [text_tag] * len(label_ids) + [video_tag] * len(vision_ids)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [text_tag] * len(prompt_ids)

        text_inputs = tokenizer.pad(
            {"input_ids": [token_ids]},
            pad_to_multiple_of=MINIMAX_H3_TEXT_SEQ_BUCKET,
            padding_side="right",
            return_attention_mask=True,
            return_tensors="pt",
        )
        embeds = self.text_encoder(
            text_inputs["input_ids"],
            text_inputs["attention_mask"],
            pixel_values=vision_inputs.get("pixel_values"),
            image_grid_thw=vision_inputs.get("image_grid_thw"),
        ).last_hidden_state.cpu()
        return embeds, torch.tensor(token_tags, dtype=torch.long)

'''

OLD_ENCODE = '''        t0 = time.perf_counter()
        prompt_embeds = self.encode_prompt(prompt)
        text_encode_seconds = time.perf_counter() - t0

        self._fl2va_height, self._fl2va_width = height, width
        num_text = prompt_embeds.shape[1]
'''

NEW_ENCODE = '''        t0 = time.perf_counter()
        if keyframes:
            prompt_embeds, text_token_tags = self._encode_prompt_with_vision(prompt, keyframes)
        elif references:
            ref_images = [entry["image"] for entry in references if entry["kind"] == "image"]
            ref_audio_count = sum(1 for entry in references if entry["kind"] == "audio")
            prompt_embeds, text_token_tags = self._encode_prompt_with_vision(
                prompt, ref_images, audio_count=ref_audio_count
            )
        else:
            prompt_embeds = self.encode_prompt(prompt)
            text_token_tags = None
        text_encode_seconds = time.perf_counter() - t0

        self._fl2va_height, self._fl2va_width = height, width
        num_text = prompt_embeds.shape[1]
'''

OLD_TAGS = '''        text_token_tags = torch.full((num_text,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
'''

NEW_TAGS = '''        if text_token_tags is None:
            text_token_tags = torch.full((num_text,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
        elif text_token_tags.shape[0] != num_text:
            # The bucket padding extends the text segment; pads carry the text tag.
            text_token_tags = torch.cat(
                [
                    text_token_tags,
                    torch.full((num_text - text_token_tags.shape[0],), MINIMAX_H3_TEXT_TAG, dtype=torch.long),
                ]
            )
'''


def apply(src: str) -> str:
    if "_encode_prompt_with_vision" in src:
        raise SystemExit("vision conditioning already applied")

    # Processor in __init__ (after the tokenizer load).
    anchor = '''        self.tokenizer = AutoTokenizer.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only
        )
'''
    assert anchor in src, "tokenizer anchor not found"
    src = src.replace(anchor, anchor + PROCESSOR_INIT, 1)

    anchor = "    def _encode_vae_condition(self, pixels: torch.Tensor) -> torch.Tensor:"
    assert anchor in src, "method anchor not found"
    src = src.replace(anchor, VISION_METHOD + anchor, 1)

    assert OLD_ENCODE in src, "encode anchor not found"
    src = src.replace(OLD_ENCODE, NEW_ENCODE, 1)
    assert OLD_TAGS in src, "tags anchor not found"
    src = src.replace(OLD_TAGS, NEW_TAGS, 1)
    return src


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    out = apply(src)
    with open(path, "w") as f:
        f.write(out)
    print(f"Applied vision conditioning to {path}")


if __name__ == "__main__":
    main()
