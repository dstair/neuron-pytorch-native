"""lab10: execution accuracy on BIRD-for-DuckDB -- queries that actually run.

The missing metric in the stack: parse_rate and bind_rate say the SQL is well-formed and
refers to real things; this says whether it RETURNS THE RIGHT ANSWER. Model SQL and the
verified DuckDB gold both execute against the real BIRD database; result sets are compared
with the same float-tolerant multiset comparator that verified the dataset.

Prompt format is the fine-tune's own (-- DuckDB SQL generation + schema DDL + question,
plus BIRD's external-knowledge evidence line). Decode is the cached recurrent path.

    ./env/container.sh exec python3 labs/lab10_bird_exec_eval.py \
        --adapter /work/runs/ft_epoch1/adapter_final.pt --limit 128 --tag lab10_ft_128
    ./env/container.sh exec python3 labs/lab10_bird_exec_eval.py --limit 128 --tag lab10_base_128
"""

import argparse
import collections
import json
import pathlib

from sqlft import bird, data, infer
from sqlft.eval_sql import _wilson, check_sql, extract_sql
from sqlft.lora import load_lora, merge_and_unwrap_lora
from sqlft.model import load_model

RUNS = pathlib.Path("/work/runs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=pathlib.Path, default=None,
                    help="LoRA adapter; omit for the base model")
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-targets", choices=("all", "mixers"), default="all",
                    help="must match the adapter's training targets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="lab10_bird_exec")
    args = ap.parse_args()

    questions = bird.stratified_subset(bird.load_questions(), args.limit, seed=args.seed)
    ddl_by_db = {}
    cons = {}
    for q in questions:
        db = q["db_id"]
        if db not in ddl_by_db:
            path = str(bird.DB_DIR / db / f"{db}.sqlite")
            ddl_by_db[db] = bird.schema_ddl(path)
            cons[db] = bird.duckdb_native(path)

    prompts = [bird.render_prompt(q["question"], q.get("evidence", ""), ddl_by_db[q["db_id"]])
               for q in questions]
    tok = data.get_tokenizer()
    long_prompts = sum(
        len(tok(p, add_special_tokens=False)["input_ids"]) > args.seq - args.max_new - 128
        for p in prompts)
    if long_prompts:
        print(f"note: {long_prompts} prompts exceed the seq budget and will truncate")

    model = load_model(layers=32, gdn="nki-v15-split16",
                       lora=args.adapter is not None,
                       lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                       lora_targets=args.lora_targets, device="neuron")
    if args.adapter is not None:
        n = load_lora(model, args.adapter)
        print(f"loaded {n} adapter tensors from {args.adapter}")
        print(f"merged+unwrapped {merge_and_unwrap_lora(model)} LoRA modules")
    model.eval()
    infer.install_inference_gdn(model)

    gens, finished = infer.cached_generate(model, tok, prompts,
                                           max_new_tokens=args.max_new, seq_len=args.seq)

    per_row = []
    for q, prompt, gen, fin in zip(questions, prompts, gens, finished):
        sql = extract_sql(gen)
        con = cons[q["db_id"]]
        verdict = check_sql(sql)  # parse stage only, no row schema here
        out = {"question_id": q["question_id"], "db_id": q["db_id"],
               "parses": verdict.parses, "truncated": not fin,
               "generated": gen}
        if verdict.parses:
            out.update(bird.grade_one(con, sql, q["SQL_duckdb"]))
        else:
            out["stage"] = "truncated" if not fin else "parse_error"
            out["error"] = verdict.error
        per_row.append(out)

    n = len(per_row)
    stages = collections.Counter(r["stage"] for r in per_row)
    n_exec = stages["ok"]
    n_parse = sum(r["parses"] for r in per_row)
    by_db: dict[str, collections.Counter] = {}
    for r in per_row:
        by_db.setdefault(r["db_id"], collections.Counter())[r["stage"]] += 1

    result = {
        "n": n,
        "exec_accuracy": n_exec / n,
        "exec_ci95": _wilson(n_exec, n),
        "parse_rate": n_parse / n,
        "stages": dict(stages),
        "by_db": {db: dict(c) for db, c in sorted(by_db.items())},
        "adapter": str(args.adapter) if args.adapter else None,
        "args": vars(args),
        "per_row": per_row,
    }
    lo, hi = result["exec_ci95"]
    print(f"\nn={n}  exec_accuracy={result['exec_accuracy']:.3f} [{lo:.3f}, {hi:.3f}]  "
          f"parse_rate={result['parse_rate']:.3f}")
    print(f"stages: {dict(stages)}")
    for db, c in sorted(by_db.items()):
        total = sum(c.values())
        print(f"  {db:26} {c.get('ok', 0)}/{total} exec")

    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{args.tag}.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {RUNS / f'{args.tag}.json'}")
    print("BIRD EXEC EVAL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
