"""Baseline: what parse rate does the UNTUNED Qwen3.5-4B get on our prompt format?

This is the "before" half of the only number that answers the project's actual question.
Also establishes the gold ceiling on the same eval rows, so the after-number can be read
against both a floor and a ceiling rather than against 1.0.

    ./env/container.sh exec python3 labs/lab01_baseline_eval.py --limit 32
"""

import argparse
import json
import pathlib

from sqlft import data, eval_sql
from sqlft.model import load_model

from sqlft.gdn import GDN_CHOICES, GDN_HELP

RUNS = pathlib.Path("/work/runs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=128,
                    help="128 covers every gold query in the eval set (max 124), so no row "
                         "is unwinnable by construction (see NOTES.md)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="1 is fastest: the GDN kernel serialises over batch x heads")
    # nki-v15-split16 is correct AND fastest, so it is the right default. Plain `nki` corrupts
    # the forward past ~6 GDN layers and would describe a broken model, not Qwen3.5-4B.
    ap.add_argument("--gdn", choices=GDN_CHOICES, default="nki-v15-split16")
    ap.add_argument("--decode", choices=("reforward", "cached"), default="reforward",
                    help="cached = recurrent/KV-cache decode (sqlft.infer; validated by lab09)")
    ap.add_argument("--tag", default="lab01_baseline")
    args = ap.parse_args()

    tok = data.get_tokenizer()
    _, evl = data.split_rows(data.load_rows())
    rows = evl[: args.limit]

    ceiling = eval_sql.gold_ceiling(rows)
    print(f"gold ceiling on these {len(rows)} rows: parse_rate={ceiling['parse_rate']:.3f}")

    model = load_model(layers=args.layers, gdn=args.gdn,
                       lora=False, device="neuron")
    model.eval()

    if args.decode == "cached":
        from sqlft import infer

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
    res["args"] = vars(args)

    print("\n--- sample generations ---")
    for s in res["samples"][:3]:
        print(f"Q    : {s['question'][:90]}")
        print(f"gold : {s['gold'][:90]}")
        print(f"model: {s['generated'][:90]!r}\n")

    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{args.tag}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {RUNS / f'{args.tag}.json'}")
    print("BASELINE EVAL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
