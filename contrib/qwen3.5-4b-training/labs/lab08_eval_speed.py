"""Validate and time the fast decode. Correctness first, speed second.

A generation rewrite that changes outputs invalidates every eval number. But comparing decoded
STRINGS is the wrong test: greedy decode is chaotic, so a single near-tie flipped by ordinary
float noise cascades into a completely different (and equally valid) query. Measured: 1 of 8
prompts diverged that way between the two paths, at a token where the top-2 logits were within
noise.

So the invariant is tested directly, with teacher forcing -- feed BOTH layouts the identical
token sequence and compare the next-token logits. No chaotic divergence to confuse it:

  A  logits: padded-1024 vs bucketed+right-aligned, same content, every decode position.
     Reports cosine, max abs difference, top-1 agreement, and the top-2 margin wherever top-1
     disagrees -- a disagreement only matters if the margin is bigger than the noise.
  B  fast(bs=1) vs fast(bs=N) decoded strings -- must MATCH. Batch invariance: a prompt's
     output must not depend on who it was batched with.
  C  padded(bs=1) vs padded(bs=N) -- the old path passes no mask-aware position_ids, so left
     padding shifts each row's RoPE phases. Reported for information; on this sample it shows
     no difference, plausibly because Qwen3.5 uses partial RoPE and only 8 of 32 layers are
     attention at all.

    ./env/container.sh exec python3 labs/lab08_eval_speed.py --limit 8
"""

import argparse
import json
import pathlib
import time

from sqlft.gdn import GDN_CHOICES, GDN_HELP

RUNS = pathlib.Path("/work/runs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8, help="prompts (kept small; bs=1 is slow)")
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--logit-steps", type=int, default=8,
                    help="teacher-forced positions to compare logits at")
    ap.add_argument("--skip-slow", action="store_true",
                    help="run only the logit check, skipping the bs=1 timing runs")
    ap.add_argument("--adapter", type=pathlib.Path, default=None)
    ap.add_argument("--gdn", choices=GDN_CHOICES, default="nki-v15-split16")
    ap.add_argument("--tag", default="lab08_eval_speed")
    args = ap.parse_args()

    from sqlft import data, eval_sql
    from sqlft.model import load_model

    tok = data.get_tokenizer()
    _, evl = data.split_rows(data.load_rows())
    rows = evl[: args.limit]
    prompts = [data.render_prompt(r) for r in rows]

    model = load_model(
        layers=args.layers,
        gdn=args.gdn,
        lora=args.adapter is not None,
        device="neuron",
    )
    if args.adapter is not None:
        from sqlft.lora import load_lora

        load_lora(model, args.adapter)
    model.eval()

    def run(fn, bs, label):
        t0 = time.perf_counter()
        outs = fn(model, tok, prompts, max_new_tokens=args.max_new, seq_len=args.seq,
                  batch_size=bs, progress=False)
        dt = time.perf_counter() - t0
        print(f"  {label:<22} {dt:7.1f}s  ({dt / len(prompts):5.1f}s/prompt)", flush=True)
        return outs, dt

    print(f"{len(prompts)} prompts, max_new={args.max_new}, seq={args.seq}, gdn={args.gdn}\n")

    # ---- A: teacher-forced logit equivalence of the two layouts -------------------------
    import torch

    from sqlft.eval_sql import position_ids_from_mask

    tok_ids = tok(prompts[0], add_special_tokens=False)["input_ids"]
    dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def logits_padded(content):
        """Old layout: full seq_len wide, content left-aligned, right-padded, no position_ids."""
        ids = torch.full((1, args.seq), pad_id, dtype=torch.long)
        mask = torch.zeros((1, args.seq), dtype=torch.long)
        ids[0, : len(content)] = torch.tensor(content)
        mask[0, : len(content)] = 1
        out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev))
        return out.logits[0, len(content) - 1, :].float().cpu()

    def logits_bucketed(content):
        """New layout: smallest 128-multiple bucket, right-aligned, mask-aware positions."""
        bucket = -(-len(content) // 128) * 128
        ids = torch.full((1, bucket), pad_id, dtype=torch.long)
        mask = torch.zeros((1, bucket), dtype=torch.long)
        ids[0, bucket - len(content) :] = torch.tensor(content)
        mask[0, bucket - len(content) :] = 1
        out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                    position_ids=position_ids_from_mask(mask).to(dev), logits_to_keep=1)
        return out.logits[0, -1, :].float().cpu()

    print("A  teacher-forced next-token logits, padded-1024 vs bucketed:")
    content = list(tok_ids)
    a_rows, a_top1_disagree, a_min_margin = [], 0, float("inf")
    with torch.no_grad():
        for t in range(args.logit_steps):
            lp, lb = logits_padded(content), logits_bucketed(content)
            c = torch.nn.functional.cosine_similarity(lp, lb, dim=0).item()
            md = (lp - lb).abs().max().item()
            t1p, t1b = int(lp.argmax()), int(lb.argmax())
            top2 = torch.topk(lp, 2).values
            margin = float(top2[0] - top2[1])
            agree = t1p == t1b
            a_top1_disagree += (not agree)
            if not agree:
                a_min_margin = min(a_min_margin, margin)
            a_rows.append({"step": t, "cos": c, "max_abs": md, "top1_agree": agree,
                           "top2_margin": margin})
            print(f"   step {t:2d}  cos {c:.6f}  max|d| {md:.3e}  top1 "
                  f"{'agree' if agree else 'DIFFER'}  top2-margin {margin:.4f}", flush=True)
            content.append(t1b)   # teacher-force on the new path's choice

    # Greedy decode depends ONLY on the argmax, so top-1 agreement is the criterion that
    # matters. cos is a sanity band, not the gate: the model runs in bf16 and these logits are
    # O(10) over a 248,320-wide vocab, so max|d| ~0.16 and cos ~0.9999 is the noise floor, not
    # a discrepancy. An earlier version gated on cos > 0.9999 and failed on exactly that.
    a_cos_ok = all(r["cos"] > 0.999 for r in a_rows)
    a_ok = a_cos_ok and a_top1_disagree == 0
    print(f"   -> top-1 agrees at every step: "
          f"{'PASS' if a_top1_disagree == 0 else f'FAIL ({a_top1_disagree} differ)'}"
          + (f", smallest disagreeing margin {a_min_margin:.4f}" if a_top1_disagree else "")
          + f";  cos > 0.999: {'yes' if a_cos_ok else 'NO'}")

    if args.skip_slow:
        # Gate mode: the logit equivalence is the correctness claim; the four timing runs take
        # ~15 minutes and only establish a speedup that is already recorded.
        print("\n(--skip-slow: skipping the timing runs)")
        result = {"tag": args.tag, "args": vars(args), "A_logit_steps": a_rows,
                  "A_logits_equivalent": a_ok, "A_top1_disagreements": a_top1_disagree}
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / f"{args.tag}.json").write_text(json.dumps(result, indent=2, default=str))
        print("\nEVAL SPEED OK" if a_ok else "\nEVAL SPEED FAILED")
        return 0 if a_ok else 1

    print()
    fast1, t_fast1 = run(eval_sql.greedy_generate, 1, "fast   bs=1")
    fastN, t_fastN = run(eval_sql.greedy_generate, args.batch_size, f"fast   bs={args.batch_size}")
    pad1, t_pad1 = run(eval_sql.greedy_generate_padded, 1, "padded bs=1")
    padN, t_padN = run(eval_sql.greedy_generate_padded, args.batch_size,
                       f"padded bs={args.batch_size}")

    def diff(a, b):
        return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]

    str_bad = diff(pad1, fast1)
    b_bad = diff(fast1, fastN)
    c_bad = diff(pad1, padN)

    print(f"\nB fast(bs=1)    vs fast(bs={args.batch_size})   : {len(b_bad)} of {len(prompts)} differ"
          f"   {'PASS' if not b_bad else 'FAIL'}   (batch invariance)")
    print(f"C padded(bs=1)  vs padded(bs={args.batch_size}) : {len(c_bad)} of {len(prompts)} differ"
          f"   {'(old path IS batch-dependent)' if c_bad else '(no difference on this sample)'}")
    print(f"   decoded-string divergence padded(bs=1) vs fast(bs=1): "
          f"{len(str_bad)} of {len(prompts)} -- informational, see the note in the docstring")

    for i in (str_bad + b_bad)[:2]:
        print(f"\n  decoded difference at prompt {i}:")
        print(f"    padded bs1: {pad1[i][:100]!r}")
        print(f"    fast   bs1: {fast1[i][:100]!r}")
        print(f"    fast   bsN: {fastN[i][:100]!r}")

    speedup_same_bs = t_padN / max(t_fastN, 1e-9)
    print(f"\nspeedup at bs={args.batch_size}: {speedup_same_bs:.2f}x "
          f"({t_padN:.1f}s -> {t_fastN:.1f}s)")

    result = {
        "tag": args.tag, "args": vars(args), "n_prompts": len(prompts),
        "t_fast_bs1": t_fast1, "t_fast_bsN": t_fastN,
        "t_padded_bs1": t_pad1, "t_padded_bsN": t_padN,
        "speedup_at_bsN": speedup_same_bs,
        "A_logit_steps": a_rows,
        "A_logits_equivalent": a_ok,
        "A_top1_disagreements": a_top1_disagree,
        "B_fast1_vs_fastN_mismatches": len(b_bad),
        "C_padded1_vs_paddedN_mismatches": len(c_bad),
        "decoded_divergence_padded1_vs_fast1": len(str_bad),
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{args.tag}.json").write_text(json.dumps(result, indent=2, default=str))

    # Gate on the logit equivalence and on batch invariance -- not on decoded-string equality,
    # which greedy decode makes chaotic.
    ok = a_ok and not b_bad
    print("\nEVAL SPEED OK" if ok else "\nEVAL SPEED FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
