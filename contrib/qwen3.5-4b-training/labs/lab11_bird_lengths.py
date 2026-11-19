"""lab11: how long are BIRD prompts, and what does a seq budget cost?

The BIRD prompt carries the database's full CREATE TABLE DDL, so length is a property of
the DATABASE more than the question. Measures prompt / gold-completion token lengths for
the dev (eval) and train splits, and for each seq budget reports:

  * fit      -- prompt + completion + EOS fits with no truncation (training keeps it whole)
  * eval_cut -- the eval's cached decode would left-truncate the prompt (it keeps the last
                seq - 128 tokens, sqlft.infer.cached_generate)

CPU only:

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 ./env/container.sh exec python3 labs/lab11_bird_lengths.py
"""

import argparse
import json
import pathlib
import statistics

from sqlft import bird, data

BIRD = pathlib.Path("/work/bird")
SPLITS = {
    # split: (questions json, db dir, gold-SQL field)
    "dev": (BIRD / "bird_duckdb_dev.json", BIRD / "dev_databases", "SQL_duckdb"),
    "train": (BIRD / "train" / "train.json", BIRD / "train" / "train_databases", "SQL"),
}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, action="append", default=None)
    args = ap.parse_args()
    seqs = args.seq or [1024, 2048, 3072, 4096]
    tok = data.get_tokenizer()
    out = {}

    for split, (qpath, db_dir, gold_key) in SPLITS.items():
        qs = json.loads(qpath.read_text())
        ddl_len: dict[str, int] = {}
        p_lens, c_lens = [], []
        ddl_cache: dict[str, str] = {}
        for q in qs:
            db = q["db_id"]
            if db not in ddl_cache:
                ddl_cache[db] = bird.schema_ddl(str(db_dir / db / f"{db}.sqlite"))
                ddl_len[db] = len(tok(ddl_cache[db], add_special_tokens=False)["input_ids"])
            prompt = bird.render_prompt(q["question"], q.get("evidence", ""), ddl_cache[db])
            p_lens.append(len(tok(prompt, add_special_tokens=False)["input_ids"]))
            c_lens.append(len(tok(q[gold_key].strip(), add_special_tokens=False)["input_ids"]) + 1)

        tot = [p + c for p, c in zip(p_lens, c_lens)]
        n = len(qs)
        print(f"\n== {split}: {n} questions over {len(ddl_cache)} databases")
        for name, xs in (("prompt", p_lens), ("completion", c_lens), ("total", tot)):
            print(f"  {name:10}  mean {statistics.mean(xs):6.0f}  p50 {pct(xs, .5):5d}  "
                  f"p90 {pct(xs, .9):5d}  p99 {pct(xs, .99):5d}  max {max(xs):5d}")
        rows = {}
        for s in seqs:
            fit = sum(t <= s for t in tot) / n
            eval_cut = sum(p > s - 128 for p in p_lens) / n
            rows[s] = {"fit": fit, "eval_cut": eval_cut}
            print(f"  seq {s:5d}: fit {100 * fit:5.1f}%   eval would truncate "
                  f"{100 * eval_cut:5.1f}%")
        big = sorted(ddl_len.items(), key=lambda kv: -kv[1])
        print("  largest schemas (DDL tokens): "
              + ", ".join(f"{db} {k}" for db, k in big[:8]))
        out[split] = {
            "n": n, "n_db": len(ddl_cache),
            "prompt": p_lens, "completion": c_lens, "by_seq": rows,
            "ddl_tokens": ddl_len,
        }

    dst = pathlib.Path("/work/runs/lab11_bird_lengths.json")
    dst.write_text(json.dumps(out))
    print(f"\nwrote {dst}")
    print("BIRD LENGTHS OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
