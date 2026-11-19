"""lab13: what are the exec-eval failures, structurally? Decides where the next lift is.

Re-executes the model SQL and verified gold for every non-ok row of a lab10 run and
classifies the miss by result SHAPE (no model, CPU only):

  extra_columns_contains_gold  model returned gold's columns plus more (BIRD grades the
                               exact column set, so this is a format miss, not semantics)
  missing_columns / extra_columns_other
  row_count_differs / superset_rows    right columns, wrong filter / DISTINCT / LIMIT
  same_shape_values_differ     right shape, wrong values (aggregation, join, evidence)
  model_empty                  model query returned nothing
  exec_error buckets           first word of the DuckDB error (Binder / Parser / ...)

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 ./env/container.sh exec \
        python3 labs/lab13_bird_error_analysis.py --run /work/runs/bird2k_bird_r16.json
"""

import argparse
import collections
import itertools
import json
import pathlib
import re

from sqlft import bird
from sqlft.eval_sql import extract_sql


def classify(ref, cand, ordered: bool) -> str:
    if bird.results_match(ref, cand, ordered):
        return "matches_on_rerun"
    if not cand:
        return "model_empty"
    if not ref:
        return "gold_empty"
    gc, mc = len(ref[0]), len(cand[0])
    if mc > gc:
        if mc <= 6 and any(
                bird.results_match(ref, [tuple(row[i] for i in idx) for row in cand], ordered)
                for idx in itertools.permutations(range(mc), gc)):
            return "extra_columns_contains_gold"
        return "extra_columns_other"
    if mc < gc:
        return "missing_columns"
    if len(ref) != len(cand):
        if set(map(tuple, cand)) >= set(map(tuple, ref)):
            return "superset_rows"
        return "row_count_differs"
    if ordered and bird.results_match(ref, cand, False):
        return "order_only"
    return "same_shape_values_differ"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--examples", type=int, default=2)
    args = ap.parse_args()

    gold = {q["question_id"]: q for q in bird.load_questions()}
    run = json.loads(args.run.read_text())
    cons: dict = {}
    cls = collections.Counter()
    examples = collections.defaultdict(list)

    for r in run["per_row"]:
        stage = r["stage"]
        if stage == "ok":
            continue
        q = gold[r["question_id"]]
        sql = extract_sql(r["generated"])
        if stage == "exec_error":
            k = "exec_error:" + (re.match(r"(\w+)", r.get("error", "?")) or [None, "?"])[1]
        elif stage != "wrong":
            k = stage
        else:
            db = r["db_id"]
            if db not in cons:
                cons[db] = bird.duckdb_native(str(bird.DB_DIR / db / f"{db}.sqlite"))
            try:
                ref = bird._run_watchdog(cons[db], q["SQL_duckdb"])
                cand = bird._run_watchdog(cons[db], sql)
                k = "wrong:" + classify(ref, cand, bool(re.search(r"\border\s+by\b",
                                                                  q["SQL_duckdb"], re.I)))
            except Exception:
                k = "wrong:rerun_error"
        cls[k] += 1
        if len(examples[k]) < args.examples:
            examples[k].append({"qid": r["question_id"], "q": q["question"],
                                "evidence": q.get("evidence", ""),
                                "gold": q["SQL_duckdb"], "model": sql,
                                "error": r.get("error", "")})

    n = len(run["per_row"])
    miss = sum(cls.values())
    print(f"{args.run.name}: n={n}, ok={n - miss}, misses={miss}")
    for k, v in cls.most_common():
        print(f"  {v:4d}  {100 * v / n:5.1f}% of all  {k}")
    print()
    for k, exs in examples.items():
        for e in exs:
            print(f"[{k}] q{e['qid']}: {e['q'][:110]}")
            print(f"   evidence: {e['evidence'][:110]}")
            print(f"   gold : {e['gold'][:170]}")
            print(f"   model: {e['model'][:170]}")
            if e["error"]:
                print(f"   error: {e['error'][:120]}")
        print()
    out = pathlib.Path("/work/runs") / f"lab13_{args.run.stem}.json"
    out.write_text(json.dumps({"counts": dict(cls), "examples": examples}, indent=1))
    print(f"wrote {out}")
    print("BIRD ERROR ANALYSIS OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
