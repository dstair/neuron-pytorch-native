"""lab09: cached recurrent decode vs full re-forward -- teacher-forced logit parity.

The cached decode path (sqlft/infer.py) is a SECOND implementation of the model: nkilib
gdn_cte prefill + deltanet_tkg decode for the GDN layers, and a static-buffer KV cache for
the attention layers, where the current eval reuses the training graph. As lab08 showed,
decoded STRINGS are the wrong thing to compare (greedy argmax over bf16 logits is chaotic;
a near-tie flips and the rest of the query diverges). The invariant that matters is the
next-token logit under teacher forcing:

  reference: one full forward over [prompt + gold[:T-1]] with the training kernel
             (nki-v15-split16), exactly what the reported eval numbers use
  candidate: prefill(prompt) once, then T-1 single-token cached steps feeding gold

Gate: top-1 agreement at every DECIDABLE step, with cos > 0.998 as a sanity band.

Two calibrations, both measured:

- cos band 0.998, not lab08's 0.999: that band was calibrated across LAYOUTS of one kernel
  path; this lab compares across KERNELS (training split16 vs nkilib gdn_cte + tkg), whose
  floor is higher. Measured on 8 rows x 11 steps: 87/88 steps cos >= 0.99927, one isolated
  near-tie step at 0.99881 that recovers fully the next step (0.99984) -- a bf16 near-tie,
  not state corruption (corruption does not recover).

- "decidable": a step whose reference top-2 margin is below ~1 bf16 ulp at logit scale
  (|logits| ~ 10-30 -> ulp ~ 0.06-0.25) is a coin flip by definition -- the two layouts of
  the SAME model also flip it (lab08). Such steps are reported but cannot gate. Measured:
  8 rows x 11 steps had exactly one sub-0.1-margin step (0.00), and it is the only step
  whose top-1 ever disagrees.

    ./env/container.sh exec python3 labs/lab09_cached_decode_parity.py
    ./env/container.sh exec python3 labs/lab09_cached_decode_parity.py \
        --adapter /work/runs/ft_epoch1/adapter_final.pt --rows 8 --steps 16
"""

import argparse
import pathlib
import time

import torch

from sqlft import data, infer
from sqlft.device import get_device, sync
from sqlft.eval_sql import greedy_generate, position_ids_from_mask
from sqlft.lora import load_lora, merge_and_unwrap_lora
from sqlft.model import load_model


@torch.no_grad()
def _reference_logits(model, tokenizer, prompt_ids, gold_ids, T, seq_len, dev):
    """Next-token logits for gold[0..T-1], from one full re-forward (the eval's math)."""
    content = prompt_ids + gold_ids[: T - 1]
    W = min(seq_len, -(-len(content) // 128) * 128)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((1, W), pad_id, dtype=torch.long)
    mask = torch.zeros((1, W), dtype=torch.long)
    ids[0, W - len(content):] = torch.tensor(content)
    mask[0, W - len(content):] = 1
    out = model(
        input_ids=ids.to(dev),
        attention_mask=mask.to(dev),
        position_ids=position_ids_from_mask(mask).to(dev),
        logits_to_keep=T,
    )
    sync(dev)
    return out.logits[0].float().cpu()  # (T, vocab)


@torch.no_grad()
def _cached_logits(model, tokenizer, prompt_ids, gold_ids, T, seq_len, dev):
    """The same logits from prefill + T-1 cached single-token steps."""
    text = model.model
    L = len(prompt_ids)
    W = min(seq_len, -(-L // 128) * 128)
    pad = W - L
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    dt = next(model.parameters()).dtype

    cache = infer.DecodeCache(model, seq_len, dev, dt)
    row = [pad_id] * pad + prompt_ids
    ids_t = torch.tensor([row], device=dev)
    mask2d = torch.tensor([[0] * pad + [1] * L], device=dev)
    pos = (mask2d.long().cumsum(-1) - 1).clamp(min=0)
    attn_mask = infer._prefill_mask(L, W, seq_len, dev, dt)
    logits = infer._forward_last(model, ids_t, attn_mask, pos, cache,
                                 input_mask=mask2d.unsqueeze(-1).to(dt))
    cache._prefilled = True
    cache.cursor = W
    out = [logits[0, -1].float().cpu()]

    for t in range(T - 1):
        ids_t = torch.tensor([[gold_ids[t]]], device=dev)
        pos = torch.tensor([[L + t]], device=dev)
        attn_mask = infer._decode_mask(pad, cache.cursor, seq_len, dev, dt)
        logits = infer._forward_last(model, ids_t, attn_mask, pos, cache)
        out.append(logits[0, -1].float().cpu())
        cache.cursor += 1
    sync(dev)
    return torch.stack(out)  # (T, vocab)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=pathlib.Path, default=None,
                    help="optional LoRA adapter; omit for the base model")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    args = ap.parse_args()

    tok = data.get_tokenizer()
    _, evl = data.split_rows(data.load_rows())
    rows = list(evl)[: args.rows]
    dev = get_device("neuron")

    model = load_model(layers=32, gdn="nki-v15-split16",
                       lora=args.adapter is not None,
                       lora_r=args.lora_r, lora_alpha=args.lora_alpha, device="neuron")
    if args.adapter is not None:
        n = load_lora(model, args.adapter)
        print(f"loaded {n} adapter tensors from {args.adapter}")
    model.eval()

    # ---- reference logits first: the stock HF forward with the training kernel ----
    prompts = [data.render_prompt(r) for r in rows]
    prompt_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    gold_ids = [tok(r.query, add_special_tokens=False)["input_ids"] for r in rows]
    T = min(args.steps, min(len(g) for g in gold_ids))

    refs = [_reference_logits(model, tok, p, g, T, args.seq, dev)
            for p, g in zip(prompt_ids, gold_ids)]

    # Old-path timing on a couple of rows while the model is still in eval config.
    t0 = time.perf_counter()
    greedy_generate(model, tok, prompts[:2], max_new_tokens=T, seq_len=args.seq,
                    device="neuron", batch_size=1, progress=False)
    sync(dev)
    old_per_prompt = (time.perf_counter() - t0) / 2

    # ---- candidate: swap in the inference kernels and merge LoRA away ----
    if args.adapter is not None:
        print(f"merged+unwrapped {merge_and_unwrap_lora(model)} LoRA modules")
    infer.install_inference_gdn(model)

    # Warm pass first: the parity numbers should not pay one-time graph compiles, and
    # neither should the timing.
    t0 = time.perf_counter()
    cands = [_cached_logits(model, tok, p, g, T, args.seq, dev)
             for p, g in zip(prompt_ids, gold_ids)]
    cached_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    cands = [_cached_logits(model, tok, p, g, T, args.seq, dev)
             for p, g in zip(prompt_ids, gold_ids)]
    cached_warm = time.perf_counter() - t0

    # ---- compare, step by step ----
    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    # bf16 ulp at |logit| ~ 10-30 is 0.06-0.25: below TIE_MARGIN the top-2 is numerically
    # indistinguishable and any decode layout may flip it. Such steps cannot gate.
    TIE_MARGIN = 0.1

    ok = True
    for r, (ref, cand) in enumerate(zip(refs, cands)):
        agree = ties = 0
        worst_cos, max_d = 1.0, 0.0
        flips = []
        for s in range(T):
            a, b = ref[s], cand[s]
            margin = float(a.topk(2).values[0] - a.topk(2).values[1])
            c = cos(a, b)
            same = int(a.argmax() == b.argmax())
            worst_cos = min(worst_cos, c)
            max_d = max(max_d, float((a - b).abs().max()))
            if margin < TIE_MARGIN:
                ties += 1
                if not same:
                    flips.append(f"{s}(tie,cos {c:.5f})")
                continue
            agree += same
            if not same:
                flips.append(f"{s}(margin {margin:.2f},cos {c:.5f})")
        decidable = T - ties
        status = "ok" if agree == decidable and worst_cos > 0.998 else "FAIL"
        ok &= status == "ok"
        print(f"row {r}: top-1 {agree}/{decidable} decidable ({ties} ties)  "
              f"worst cos {worst_cos:.5f}  max|d| {max_d:.3f}  [{status}]")
        if flips:
            print(f"   disagreements at steps: {', '.join(flips)}")

    new_per_prompt = cached_warm / len(rows)
    print(f"\ntiming: re-forward {old_per_prompt:.1f} s/prompt  "
          f"cached {new_per_prompt:.1f} s/prompt warm ({cached_cold / len(rows):.1f} cold)  "
          f"({old_per_prompt / new_per_prompt:.1f}x)")
    print("LAB09 OK" if ok else "LAB09 FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
