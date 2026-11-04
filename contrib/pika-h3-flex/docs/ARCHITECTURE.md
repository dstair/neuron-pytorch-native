# Pika-H3-Flex: architecture findings (from the HF repo, 2026-11-04)

Source: https://huggingface.co/Pika-Labs/Pika-H3-Flex
License: **MiniMax H3 Community License Agreement** (same as MiniMax-H3 — Model
Derivative by Mellis, Inc. 2026). Read LICENSE/NOTICE before use/redistribution.

## Correction to the "fewer layers" assumption

The DiT config (`transformer/config.json` and `transformer_ref/config.json`) is:

```
_class_name: MiniMaxH3Transformer3DModel
num_layers: 50, hidden_size: 5376, num_attention_heads: 56, attention_head_dim: 128
ffn_dim: 14336, num_refiner_layers: 2, in_channels: 24, audio_in_channels: 32
patch_size: [1,2,2], text_dim: 5120, rope_freq_dim: 16, qk_norm_eps/norm_eps: 1e-5
```

**This is architecturally identical to MiniMax-H3's 33B DiT** (50 layers, not
fewer). The transformer is split into 7 shards (vs MiniMax-H3's 14) — packaging
only. "Far fewer" applies to the **sampling schedule (6 NFE), not the model size.**

Consequence: it does NOT fit "more easily" by layer count — it fits exactly like
MiniMax-H3, which we already proved fits at TP4 / 768p on a single trn2.3xlarge.
The benefit is ~8x fewer denoise forwards (6 steps vs 50).

## Components (modular_model_index.json)

| Component | Class | Notes |
|---|---|---|
| pipeline | `PikaH3FlexModularPipeline` + `PikaH3FlexBlocks` | diffusers modular; likely needs diffusers >= 0.42.dev or the Pika inference package — our box has diffusers 0.40 |
| transformer | `MiniMaxH3Transformer3DModel` | FL2VA base DiT (same 33B as MiniMax-H3) |
| transformer_ref | `MiniMaxH3Transformer3DModel` | Ref2VA base DiT (same config) |
| text_encoder | `Qwen3VLForConditionalGeneration` | **full Qwen3-VL WITH vision tower** (text 64 layers hidden 5120, vision depth 27) — image conditioning; differs from MiniMax-H3's text-only encoder use |
| vae | `AutoencoderKLMiniMaxH3` | same 36-layer ViT video decoder as MiniMax-H3 |
| audio_vae | `AutoencoderKLMiniMaxH3Audio` | same BigVGAN decoder as MiniMax-H3 |
| scheduler / audio_scheduler | `MiniMaxH3Scheduler` | same scheduler class |
| tokenizer / processor | Qwen2TokenizerFast / Qwen3VLProcessor | |

## LoRA adapters (the actual Pika contribution)

- `fl2va/adapter.safetensors` — first/last-frame-to-video+audio
- `ref2va/adapter.safetensors` — multimodal reference-to-video+audio
- Rank 128; fused Q/K/V LoRA rank 384; applied at model strength 1.0.
- Base components "require the matching Pika adapter" (release notice). The base
  transformer weights are the MiniMax-H3 base; the adapter is what makes it Pika.
- There is a separate ComfyUI BF16 conversion under `comfy/loras/` — do NOT
  substitute; the Pika inference package uses `fl2va/adapter.safetensors` /
  `ref2va/adapter.safetensors` pinned by `release.json`.

## Sampling contract

- **6 NFE**, knots `1000/750/500/250/100/40/0`, video shift 12, audio shift 3,
  fresh intermediate noise. Choosing "6 steps" in a generic sampler does NOT
  reproduce this; use the release manifest (`fl2va/manifest.json`,
  `ref2va/manifest.json`).

## Pipelines / tasks

- T2V (text-to-video+audio)
- FL2VA (first/last-frame image-to-video+audio)
- Ref2VA (multimodal reference-to-video+audio)

## Why this is a good fit for our existing work

The DiT is literally the same architecture as the MiniMax-H3 we already ported to
Trainium (`MiniMaxH3Transformer3DModel`, identical config). The genuinely new work
is: the Pika pipeline (image/reference conditioning), the full Qwen3-VL vision
encoder, LoRA application, and the 6-step sampling contract. The VAE, audio VAE,
scheduler, and DiT body are shared with what we have.
