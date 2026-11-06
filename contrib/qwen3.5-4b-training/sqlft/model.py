"""Load Qwen3.5-4B for training: layer truncation, GDN kernel swap, LoRA, bf16.

Rubric coverage: nn.Module surgery on a real 4B model, and the parameter/dtype bookkeeping
that FSDP and the optimizer both depend on.

    ./env/container.sh exec python3 -m sqlft.model --layers 4 --report
"""

from __future__ import annotations

import argparse
import pathlib

import torch
import torch.nn as nn

MODEL_DIR = pathlib.Path("/work/models/Qwen3.5-4B")

# Qwen3.5-4B is a hybrid stack: every 4th layer (index 3, 7, 11, ...) is full_attention,
# the rest are linear_attention (GatedDeltaNet). Truncating to N layers therefore changes
# WHICH families are present -- --layers 3 has no full-attention layer at all.
FULL_ATTENTION_PERIOD = 4


def load_config(model_dir: pathlib.Path = MODEL_DIR, layers: int | None = None):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(model_dir))
    text = getattr(cfg, "text_config", cfg)
    if layers is not None:
        if layers < 1 or layers > text.num_hidden_layers:
            raise ValueError(f"--layers must be in 1..{text.num_hidden_layers}")
        # layer_types must be truncated in lockstep or HF builds the wrong module per layer.
        text.layer_types = list(text.layer_types)[:layers]
        text.num_hidden_layers = layers
    return cfg


def describe_stack(model: nn.Module) -> dict[str, int]:
    kinds: dict[str, int] = {}
    for layer in model.model.layers:
        kind = "full_attention" if hasattr(layer, "self_attn") else "linear_attention"
        kinds[kind] = kinds.get(kind, 0) + 1
    return kinds


def load_model(
    model_dir: pathlib.Path = MODEL_DIR,
    *,
    layers: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    # The GatedDeltaNet recurrence: "torch" (HF's reference) or a sqlft.gdn.KERNEL_ROOTS key.
    # "nki-v15-split16" is the fast correct choice; plain "nki-v15" is 6-7% off on some
    # gradients here, and "nki" (upstream v3.0) is numerically WRONG on this model's real
    # activations from the 6th GDN layer on. See NOTES.md.
    gdn: str = "torch",
    lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    lora_targets=None,  # iterable of leaf names, or a sqlft.lora.TARGET_PRESETS key
    device: str | torch.device | None = "neuron",
    gradient_checkpointing: bool = False,
    verbose: bool = True,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    # Install the kernel BEFORE constructing the model: the swap is on a module-level
    # function, so it must be in place before any layer captures a reference to it.
    if gdn != "torch":
        from sqlft.gdn import install_nki_gdn

        install_nki_gdn(gdn, verbose=verbose)

    cfg = load_config(model_dir, layers)
    # The checkpoint's top-level arch is Qwen3_5ForConditionalGeneration (it has a vision
    # tower). AutoModelForCausalLM gives us Qwen3_5ForCausalLM -- the text tower only --
    # which is all a SQL fine-tune needs and saves loading the vision encoder entirely.
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), config=cfg, dtype=dtype, low_cpu_mem_usage=True
    )
    model.config.use_cache = False  # incompatible with training; also allocates KV cache

    if lora:
        from sqlft.lora import TARGET_PRESETS, apply_lora

        targets = TARGET_PRESETS.get(lora_targets, lora_targets) \
            if isinstance(lora_targets, str) else lora_targets
        kwargs = {} if targets is None else {"targets": targets}
        apply_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
                   verbose=verbose, **kwargs)
        # PEFT-style adapters are often created in fp32 while the base is bf16. Ours are
        # created in the base dtype on purpose -- FSDP's flat parameter requires a UNIFORM
        # dtype and otherwise fails with "Must flatten tensors with uniform dtype".
        for n, p in model.named_parameters():
            if p.requires_grad and p.dtype != dtype:
                raise AssertionError(f"{n} is {p.dtype}, expected {dtype} (FSDP needs uniform dtype)")

    if gradient_checkpointing:
        from sqlft.ckpt import enable_activation_checkpointing

        enable_activation_checkpointing(model, verbose=verbose)

    if device is not None:
        from sqlft.device import get_device, sync

        dev = get_device(str(device))
        model.to(dev)
        sync(dev)

    model.train()

    if verbose:
        kinds = describe_stack(model)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"model: {type(model).__name__}  layers={len(model.model.layers)} {kinds}")
        print(f"params: {total / 1e9:.3f}B total, {trainable / 1e6:.2f}M trainable "
              f"({100 * trainable / total:.3f}%), dtype={dtype}")
    return model


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# --------------------------------------------------------------------------------------


def _report(args) -> int:
    """Build on the meta device (no weights, no memory) and print the parameter budget."""
    from transformers import AutoModelForCausalLM

    cfg = load_config(MODEL_DIR, args.layers)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)

    text = getattr(cfg, "text_config", cfg)
    kinds = describe_stack(model)
    total = sum(p.numel() for p in model.parameters())
    embed = model.model.embed_tokens.weight.numel()

    print(f"layers={len(model.model.layers)} {kinds}")
    print(f"vocab={text.vocab_size} hidden={text.hidden_size} tied_embeddings={text.tie_word_embeddings}")
    print(f"params total      {total / 1e9:8.3f} B")
    print(f"  of which embed  {embed / 1e9:8.3f} B  ({100 * embed / total:.1f}%)")
    print(f"bf16 weights      {2 * total / 2**30:8.2f} GiB")
    print(f"+ bf16 grads      {2 * total / 2**30:8.2f} GiB")
    print(f"+ AdamW fp32 m,v  {8 * total / 2**30:8.2f} GiB")
    print(f"= full-FT state   {12 * total / 2**30:8.2f} GiB   (vs 24 GiB per logical core at LNC=2)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    raise SystemExit(_report(a))
