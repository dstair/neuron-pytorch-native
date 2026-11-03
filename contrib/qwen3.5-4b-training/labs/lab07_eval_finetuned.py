"""Parse rate after fine-tuning, on the same eval rows as the baseline.

Loads the base model, applies LoRA with the same geometry, loads the trained adapter, and
re-runs the identical eval as `labs/lab01_baseline_eval.py` so the before/after is
apples-to-apples: same rows, same prompt format, same decode settings, same GDN path.

    ./env/container.sh exec python3 labs/lab07_eval_finetuned.py \
        --adapter /work/runs/ft_v1/adapter_final.pt --limit 32
"""

import argparse
import json
import pathlib

from sqlft import data, eval_sql
from sqlft.lora import load_lora
from sqlft.model import load_model

from sqlft.gdn import GDN_CHOICES, GDN_HELP

RUNS = pathlib.Path("/work/runs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=pathlib.Path, required=True)
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="1 is fastest: the GDN kernel serialises over batch x heads")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--gdn", choices=GDN_CHOICES, default="nki-v15-split16")
    ap.add_argument("--decode", choices=("reforward", "cached"), default="reforward",
                    help="cached = recurrent/KV-cache decode (sqlft.infer; validated by "
                         "lab09). ~5x faster end to end; logits verified equivalent.")
    ap.add_argument("--tag", default="lab07_eval_finetuned")
    args = ap.parse_args()

    tok = data.get_tokenizer()
    _, evl = data.split_rows(data.load_rows())
    rows = evl[: args.limit]

    ceiling = eval_sql.gold_ceiling(rows)
    print(f"gold ceiling on these {len(rows)} rows: parse_rate={ceiling['parse_rate']:.3f}")

    model = load_model(
        layers=args.layers,
        gdn=args.gdn,
        lora=True, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        device="neuron",
    )
    n = load_lora(model, args.adapter)
    print(f"loaded {n} adapter tensors from {args.adapter}")
    model.eval()

    if args.decode == "cached":
        from sqlft import infer
        from sqlft.lora import merge_and_unwrap_lora

        print(f"merged+unwrapped {merge_and_unwrap_lora(model)} LoRA modules")
        infer.install_inference_gdn(model)
        res = infer.evaluate_model_cached(
            model, tok, rows,
            limit=args.limit, max_new_tokens=args.max_new, seq_len=args.seq,
        )
    else:
        res = eval_sql.evaluate_model(
            model, tok, rows,
            limit=args.limit, max_new_tokens=args.max_new,
            seq_len=args.seq, batch_size=args.batch_size,
        )
    res["gold_ceiling"] = ceiling
    res["adapter"] = str(args.adapter)
    res["args"] = vars(args)

    print("\n--- sample generations ---")
    for s in res["samples"][:5]:
        print(f"Q    : {s['question'][:90]}")
        print(f"gold : {s['gold'][:90]}")
        print(f"model: {s['generated'][:90]!r}\n")

    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{args.tag}.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"wrote {RUNS / f'{args.tag}.json'}")
    print("FINETUNE EVAL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
