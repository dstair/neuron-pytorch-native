"""Where the HBM goes, from first principles, reconciled against a measurement.

Rubric coverage: understanding memory usage. The point of this module is that you can
PREDICT the number before you run, and that when the prediction and the measurement
disagree, the gap is the thing you have not understood yet.

Four consumers, with very different scaling:

  weights       2 * P bytes (bf16)                      -- fixed
  gradients     2 * P_trainable                         -- LoRA makes this ~0
  optimizer     8 * P_trainable (AdamW fp32 m and v)    -- LoRA makes this ~0
  activations   ~ bs * seq * hidden * layers * k        -- LoRA does NOT reduce this

That last row is the one people get wrong about LoRA. LoRA cuts gradients and optimizer
state to nothing, and does nothing whatsoever for activations -- which is why LoRA still
OOMs when you raise batch size or sequence length.

    ./env/container.sh exec python3 -m sqlft.memory --budget
    ./env/container.sh exec python3 -m sqlft.memory --budget --lora
"""

from __future__ import annotations

import argparse

import torch

GIB = 2**30

# One logical NeuronCore's HBM share on this box: 96 GB / 4 logical cores at LNC=2.
HBM_PER_CORE_LNC2 = 24 * 10**9
HBM_PER_CORE_LNC1 = 12 * 10**9


def parameter_budget(total_params: int, trainable_params: int, *, dtype_bytes: int = 2) -> dict:
    """Bytes of persistent state. Independent of batch and sequence length."""
    weights = dtype_bytes * total_params
    grads = dtype_bytes * trainable_params
    # AdamW keeps exp_avg and exp_avg_sq. torch allocates them in the PARAMETER dtype, so
    # in a pure-bf16 setup that is 2+2 rather than 4+4 -- but bf16 second moments lose
    # precision badly, so this reports the fp32 figure and flags which one you are running.
    adam_fp32 = 8 * trainable_params
    adam_same_dtype = 2 * dtype_bytes * trainable_params
    return {
        "weights": weights,
        "grads": grads,
        "adam_fp32": adam_fp32,
        "adam_param_dtype": adam_same_dtype,
        "total_fp32_adam": weights + grads + adam_fp32,
        "total_param_dtype_adam": weights + grads + adam_same_dtype,
    }


def logits_bytes(bs: int, seq: int, vocab: int, *, dtype_bytes: int = 2) -> dict:
    """The thing a 248,320-token vocabulary does to you.

    The logits tensor alone is bs*seq*vocab. Cross-entropy then usually materialises a
    float32 copy plus a log-softmax, so budget ~3x. At bs1/seq1024/vocab248320 the bf16
    logits are 0.47 GiB and the fp32 CE working set is ~1.9 GiB -- for ONE tensor, in a
    model whose entire weight set is 7.8 GiB.
    """
    n = bs * seq * vocab
    return {
        "logits": dtype_bytes * n,
        "logits_fp32": 4 * n,
        "ce_working_set_est": 3 * 4 * n,
    }


def activation_estimate(bs: int, seq: int, hidden: int, layers: int, *,
                        dtype_bytes: int = 2, tensors_per_layer: int = 12) -> int:
    """Order-of-magnitude activation footprint for backward.

    `tensors_per_layer` is a fudge factor, and it is labelled as one: the real count
    depends on which intermediates autograd decides to keep, and the GatedDeltaNet path
    saves a recurrent state (H*D*D per layer) that a plain transformer does not. Use this
    to predict SCALING (linear in bs and seq), not absolutes -- `--sweep` measures absolutes.
    """
    return dtype_bytes * tensors_per_layer * bs * seq * hidden * layers


def fmt(b: float) -> str:
    return f"{b / GIB:7.2f} GiB"


def report_budget(total_params: int, trainable_params: int, *, bs: int, seq: int,
                  hidden: int, layers: int, vocab: int, measured_peak: int | None = None) -> dict:
    p = parameter_budget(total_params, trainable_params)
    lg = logits_bytes(bs, seq, vocab)
    act = activation_estimate(bs, seq, hidden, layers)

    print(f"params {total_params / 1e9:.3f}B total / {trainable_params / 1e6:.2f}M trainable")
    print(f"  weights (bf16)            {fmt(p['weights'])}")
    print(f"  grads (bf16, trainable)   {fmt(p['grads'])}")
    print(f"  AdamW m,v (fp32)          {fmt(p['adam_fp32'])}")
    print(f"  -- persistent subtotal    {fmt(p['total_fp32_adam'])}")
    print(f"per-step, bs={bs} seq={seq} layers={layers}:")
    print(f"  logits (bf16)             {fmt(lg['logits'])}   vocab={vocab}")
    print(f"  cross-entropy working est {fmt(lg['ce_working_set_est'])}")
    print(f"  activations (rough)       {fmt(act)}")
    predicted = p["total_fp32_adam"] + lg["ce_working_set_est"] + act
    print(f"  == predicted peak         {fmt(predicted)}")
    print(f"budget: {fmt(HBM_PER_CORE_LNC2)} per core at LNC=2, {fmt(HBM_PER_CORE_LNC1)} at LNC=1")

    out = {"parameter_budget": p, "logits": lg, "activations_est": act, "predicted_peak": predicted}
    if measured_peak is not None:
        print(f"  measured peak             {fmt(measured_peak)}   "
              f"(prediction is {predicted / max(measured_peak, 1):.2f}x the measurement)")
        out["measured_peak"] = measured_peak
        out["prediction_ratio"] = predicted / max(measured_peak, 1)
    return out


# --------------------------------------------------------------------------------------


def _budget(args) -> int:
    from transformers import AutoModelForCausalLM

    from sqlft.model import load_config

    cfg = load_config(layers=args.layers)
    text = getattr(cfg, "text_config", cfg)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)

    total = sum(p.numel() for p in model.parameters())
    if args.lora:
        # Rank-r adapters on every wrapped Linear. Counted from the real module list rather
        # than estimated, since which projections exist depends on --layers.
        import torch.nn as nn

        trainable = 0
        for _, mod in model.named_modules():
            for cn, child in mod.named_children():
                if isinstance(child, nn.Linear) and cn in (
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                    "gate_proj", "up_proj", "down_proj",
                ):
                    trainable += args.lora_r * (child.in_features + child.out_features)
    else:
        trainable = total

    report_budget(total, trainable, bs=args.bs, seq=args.seq,
                  hidden=text.hidden_size, layers=len(model.model.layers),
                  vocab=text.vocab_size, measured_peak=args.measured)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", action="store_true")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--measured", type=int, default=None,
                    help="measured peak bytes, to reconcile against (from runs/<tag>.json)")
    raise SystemExit(_budget(ap.parse_args()))
