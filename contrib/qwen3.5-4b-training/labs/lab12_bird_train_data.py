"""lab12: gate the BIRD-train fine-tuning JSONL before spending chip hours on it.

Three properties, each a hard assert:

  1. PROMPT PARITY. For every row, the training prompt (sqlft.data.render_prompt over the
     JSONL Row) is byte-identical to the prompt the exec eval builds (sqlft.bird.
     render_prompt over the same question/evidence/schema). A format drift here silently
     trains the model on a distribution the eval never shows it.
  2. NO LEAKAGE. No training database is a dev (eval) database, and no training question
     text appears in dev.
  3. COVERAGE at the training seq: how many rows survive filter_parseable and the
     drop-if-too-long policy, and which databases lose rows.

CPU only:

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 ./env/container.sh exec python3 labs/lab12_bird_train_data.py
"""

import argparse
import collections
import json
import pathlib

from sqlft import bird, data

BIRD = pathlib.Path("/work/bird")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=pathlib.Path, default=BIRD / "bird_train_train.jsonl")
    ap.add_argument("--verified", type=pathlib.Path, default=BIRD / "bird_duckdb_train.json")
    ap.add_argument("--seq", type=int, default=2048)
    args = ap.parse_args()

    rows = data.load_rows(args.jsonl)
    verified = json.loads(args.verified.read_text())
    assert len(rows) == len(verified), f"{len(rows)} JSONL rows vs {len(verified)} verified"

    # 1. prompt parity, every row
    ddl_cache: dict[str, str] = {}
    for row, q in zip(rows, verified):
        db = q["db_id"]
        if db not in ddl_cache:
            ddl_cache[db] = bird.schema_ddl(
                str(BIRD / "train" / "train_databases" / db / f"{db}.sqlite"))
        want = bird.render_prompt(q["question"], q.get("evidence", ""), ddl_cache[db])
        got = data.render_prompt(row)
        assert got == want, f"prompt drift on {db}:\n--- train\n{got[-300:]}\n--- eval\n{want[-300:]}"
        assert row.query.strip() == q["SQL_duckdb"].strip()
    print(f"parity: {len(rows)}/{len(rows)} training prompts byte-identical to the eval's")

    # 2. leakage
    dev = bird.load_questions()
    dev_dbs = {q["db_id"] for q in dev}
    train_dbs = {q["db_id"] for q in verified}
    assert not (dev_dbs & train_dbs), f"db overlap: {sorted(dev_dbs & train_dbs)}"
    dev_q = {q["question"].strip().lower() for q in dev}
    dup = [q["question"] for q in verified if q["question"].strip().lower() in dev_q]
    assert not dup, f"{len(dup)} train questions appear in dev, e.g. {dup[:2]}"
    print(f"leakage: 0 shared dbs ({len(train_dbs)} train / {len(dev_dbs)} dev), "
          f"0 shared questions")

    # 3. coverage at seq
    tok = data.get_tokenizer()
    kept_parse = data.filter_parseable(rows, report=True)
    ds = data.SqlDataset(kept_parse, tok, args.seq, report=True)
    lost = collections.Counter()
    for r in kept_parse:
        n = (len(tok(data.render_prompt(r), add_special_tokens=False)["input_ids"])
             + len(tok(data.render_completion(r), add_special_tokens=False)["input_ids"]) + 1)
        if n > args.seq:
            lost[r.category] += 1
    print(f"seq {args.seq}: {len(ds)} trainable rows; dropped as too long by db: "
          + ", ".join(f"{db} {n}" for db, n in lost.most_common(10)))
    lens = sorted(len(ex[0]) for ex in ds.examples)
    print(f"trainable token lengths: p50 {lens[len(lens) // 2]}  "
          f"p90 {lens[int(.9 * len(lens))]}  max {lens[-1]}  "
          f"padding waste at seq {args.seq}: "
          f"{100 * (1 - sum(lens) / (len(lens) * args.seq)):.1f}%")
    out = {"n_rows": len(rows), "n_parseable": len(kept_parse), "n_trainable": len(ds),
           "seq": args.seq, "dropped_by_db": dict(lost)}
    pathlib.Path("/work/runs/lab12_bird_train_data.json").write_text(json.dumps(out, indent=1))
    print("BIRD TRAIN DATA OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
