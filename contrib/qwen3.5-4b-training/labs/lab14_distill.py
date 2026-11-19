"""Distillation pilot: teacher reasoning traces for BIRD-train, filtered by execution.

Two halves, because the teacher and the databases live in different places:

  gen     (anywhere with Bedrock creds; no torch/duckdb needed)
          Ask a reasoning teacher (DeepSeek-R1 on Bedrock; gpt-oss-120b fallback) to
          solve BIRD-train questions from the SAME prompt the student sees, with a short
          DuckDB-dialect preamble. The teacher never sees the gold SQL -- the trace must be
          a real solution, not a rationalisation. Resumable: rows already in --out are skipped.

  verify  (on the box, in the container: needs sqlft.bird + the train databases)
          Execute each teacher SQL against the native DuckDB database and compare result
          sets with the verified gold (sqlft.bird.grade_one, the eval's own comparator).
          Only `ok` rows are distillation candidates. Reports pass rate by database and
          trace lengths in STUDENT tokens (what seq budget the traces would need).

    python3 labs/lab14_distill.py gen --data /tmp/bird_train_train.jsonl \
        --out runs/distill_r1_pilot.jsonl --n 100 --model r1
    ./env/container.sh exec python3 labs/lab14_distill.py verify \
        --data /work/bird/bird_train_train.jsonl --gen /work/runs/distill_r1_pilot.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

MODELS = {
    "r1": "us.deepseek.r1-v1:0",
    "gpt-oss-120b": "openai.gpt-oss-120b-1:0",
}

PREAMBLE = (
    "You are an expert data engineer writing SQL for DuckDB (not SQLite). Write ONE DuckDB "
    "SQL query that answers the question over the schema below.\n"
    "- Use only tables and columns that exist in the schema.\n"
    "- Quote identifiers that contain spaces or special characters with double quotes, "
    "never backticks.\n"
    "- The Evidence line, when present, is external knowledge that maps the question's "
    "wording onto columns and values; follow it.\n"
    "- Return exactly the columns the question asks for, nothing extra.\n"
    "Think it through, then give the final query in a single ```sql block and nothing "
    "after it.\n\n"
)


def render_student_prompt(row: dict) -> str:
    """Byte-identical to sqlft.data.render_prompt (and so to the lab10 eval prompt)."""
    return (
        "-- DuckDB SQL generation\n"
        "-- Schema:\n"
        f"{row['schema'].strip()}\n"
        f"-- Question: {row['prompt'].strip()}\n"
        "-- Answer:\n"
    )


def extract_final_sql(text: str) -> str:
    """Last ```sql fenced block (the teacher's final answer); else last fenced block."""
    blocks = re.findall(r"```(?:sql|SQL)?\s*\n(.*?)```", text, re.S)
    return blocks[-1].strip() if blocks else text.strip()


def load_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path)]


def stratified(rows: list[dict], n: int, seed: int) -> list[int]:
    """Round-robin across databases so a 100-row pilot touches every db (69 of them)."""
    by_db: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_db[r["category"]].append(i)
    rng = random.Random(seed)
    for v in by_db.values():
        rng.shuffle(v)
    dbs = sorted(by_db)
    rng.shuffle(dbs)
    out: list[int] = []
    while len(out) < n and any(by_db.values()):
        for d in dbs:
            if by_db[d] and len(out) < n:
                out.append(by_db[d].pop())
    return out


# ------------------------------------------------------------------------------ gen

FEEDBACK_TAIL = (
    "\n\nA previous attempt at this question produced the query below. It may be wrong. "
    "Re-read the question, the evidence and the schema carefully, check the previous "
    "attempt against them, and write the correct query. Solve the question directly: in "
    "your reasoning, do not refer to the previous attempt or to any feedback; reason as if "
    "you were solving it for the first time.\n\n"
    "Previous query:\n```sql\n{sql}\n```\nIt returned: {result}\n"
)

RATIONALIZE_TAIL = (
    "\n\nFor reference, a verified correct DuckDB query for this question is:\n"
    "```sql\n{gold}\n```\n"
    "Write out the reasoning that a careful engineer would use to arrive at a correct "
    "query from the question, evidence and schema alone. Your reasoning must stand on its "
    "own: do NOT mention the reference query, a hint, an expected answer, or that you were "
    "given anything. Then give your final query in a single ```sql block."
)


def build_message(mode: str, row: dict, prev: dict | None) -> str:
    base = PREAMBLE + render_student_prompt(row)
    if mode == "plain":
        return base
    if mode == "feedback":
        res = prev.get("teacher", {})
        result = (f"ERROR: {res['error']}" if "error" in res else
                  f"{res.get('n_rows')} rows x {res.get('n_cols')} cols; first rows: "
                  f"{json.dumps(res.get('head', [])[:5])}")
        return base + FEEDBACK_TAIL.format(sql=prev.get("teacher_sql", ""), result=result[:1200])
    if mode == "rationalize":
        return base + RATIONALIZE_TAIL.format(gold=row["query"].strip())
    raise ValueError(mode)




def cmd_gen(a) -> int:
    import boto3
    from botocore.config import Config

    rows = load_rows(a.data)
    if a.idx_file:
        idxs = [int(x) for x in pathlib.Path(a.idx_file).read_text().split()]
    else:
        idxs = stratified(rows, a.n, a.seed) if a.n else list(range(len(rows)))
    prev = {}  # feedback mode: the teacher's earlier attempt, from an audit file
    if a.mode == "feedback":
        prev = {json.loads(l)["idx"]: json.loads(l) for l in open(a.prev_audit) if l.strip()}
    out = pathlib.Path(a.out)
    done = set()
    if out.exists():
        done = {(json.loads(l)["idx"], json.loads(l).get("sample", 0))
                for l in out.open() if l.strip()}
    todo = [(i, k) for i in idxs for k in range(a.samples) if (i, k) not in done]
    print(f"{len(idxs)} rows x {a.samples} samples, {len(done)} already done, {len(todo)} to "
          f"generate with {a.model} ({MODELS[a.model]}), mode={a.mode}, "
          f"effort={a.reasoning_effort or 'default'}")

    client = boto3.client(
        "bedrock-runtime", region_name=a.region,
        config=Config(read_timeout=900, connect_timeout=30,
                      retries={"max_attempts": 10, "mode": "adaptive"}))
    lock = threading.Lock()
    fh = out.open("a")

    extra = ({"additionalModelRequestFields": {"reasoning_effort": a.reasoning_effort}}
             if a.reasoning_effort else {})

    def one(job) -> dict:
        i, k = job
        row = rows[i]
        msg = build_message(a.mode, row, prev.get(i))
        rec = {"idx": i, "sample": k, "db": row["category"], "model": a.model,
               "mode": a.mode, "effort": a.reasoning_effort}
        t = time.time()
        for attempt in range(4):
            try:
                r = client.converse(
                    modelId=MODELS[a.model],
                    messages=[{"role": "user", "content": [{"text": msg}]}],
                    inferenceConfig={"maxTokens": a.max_tokens, "temperature": a.temperature},
                    **extra)
                break
            except Exception as e:  # throttling beyond botocore's retries, transient 5xx
                if attempt == 3:
                    rec.update(error=f"{type(e).__name__}: {str(e)[:300]}",
                               latency_s=round(time.time() - t, 1))
                    return rec
                time.sleep(10 * (attempt + 1))
        reasoning, answer = "", ""
        for block in r["output"]["message"]["content"]:
            if "reasoningContent" in block:
                reasoning += block["reasoningContent"].get("reasoningText", {}).get("text", "")
            elif "text" in block:
                answer += block["text"]
        rec.update(reasoning=reasoning, answer=answer, sql=extract_final_sql(answer),
                   stop=r.get("stopReason"), usage=r.get("usage"),
                   latency_s=round(time.time() - t, 1))
        return rec

    n_ok = n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, j) for j in todo]
        for k, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            n_err += "error" in rec
            n_ok += "error" not in rec
            if k % 10 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)}  ok {n_ok}  err {n_err}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    fh.close()
    summarize_gen(out)
    return 0


def summarize_gen(path: pathlib.Path) -> None:
    recs = [json.loads(l) for l in path.open() if l.strip()]
    good = [r for r in recs if "error" not in r]
    if not good:
        print("no successful generations")
        return
    outt = sorted(r["usage"]["outputTokens"] for r in good)
    inn = [r["usage"]["inputTokens"] for r in good]
    lat = sorted(r["latency_s"] for r in good)
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]
    stops = collections.Counter(r.get("stop") for r in good)
    print(f"generations: {len(good)} ok, {len(recs) - len(good)} errors; stop reasons {dict(stops)}")
    print(f"teacher output tokens  p50 {q(outt, .5)}  p90 {q(outt, .9)}  max {outt[-1]}")
    print(f"latency s              p50 {q(lat, .5)}  p90 {q(lat, .9)}")
    print(f"tokens in/out total    {sum(inn):,} / {sum(outt):,}")


# --------------------------------------------------------------------------- verify


def cmd_verify(a) -> int:
    from sqlft import bird

    rows = load_rows(a.data)
    recs = [json.loads(l) for l in open(a.gen) if l.strip()]
    db_dir = pathlib.Path(a.db_dir)
    cons: dict = {}
    stages = collections.Counter()
    by_db: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    out_recs = []
    for k, rec in enumerate(recs, 1):
        if "error" in rec:
            rec["stage"] = "gen_error"
        else:
            db = rec["db"]
            if db not in cons:
                cons[db] = bird.duckdb_native(str(db_dir / db / f"{db}.sqlite"))
            g = bird.grade_one(cons[db], rec["sql"], rows[rec["idx"]]["query"])
            rec["stage"] = g["stage"]
            if "error" in g:
                rec["grade_error"] = g["error"]
        stages[rec["stage"]] += 1
        by_db[rec["db"]][0] += rec["stage"] == "ok"
        by_db[rec["db"]][1] += 1
        out_recs.append(rec)
        if k % 50 == 0:
            print(f"  graded {k}/{len(recs)}", flush=True)

    n = len(recs)
    ok = stages["ok"]
    print(f"\nteacher exec accuracy on BIRD-train: {ok}/{n} = {ok / n:.3f}")
    print(f"stages: {dict(stages)}")

    # Trace length in STUDENT tokens: what a reasoning-format training row would cost.
    try:
        from transformers import AutoTokenizer
        import os
        tok = AutoTokenizer.from_pretrained(os.environ.get("SQLFT_MODEL", "/work/models/Qwen3.5-4B"))
        lens = []
        for rec in out_recs:
            if rec["stage"] != "ok":
                continue
            p = len(tok(render_student_prompt(rows[rec["idx"]]))["input_ids"])
            t = len(tok(rec.get("reasoning", ""))["input_ids"])
            s = len(tok(rec["sql"])["input_ids"])
            rec["student_tokens"] = {"prompt": p, "trace": t, "sql": s}
            lens.append((p, t, s))
        if lens:
            q = lambda v, pc: sorted(v)[min(len(v) - 1, int(pc * len(v)))]
            tr = [t for _, t, _ in lens]
            tot = [p + t + s for p, t, s in lens]
            print(f"student-token trace length (ok rows)  p50 {q(tr, .5)}  p90 {q(tr, .9)}  max {max(tr)}")
            print(f"prompt+trace+sql                      p50 {q(tot, .5)}  p90 {q(tot, .9)}  max {max(tot)}")
            for budget in (1664, 2048, 3072, 4096, 6144):
                print(f"  fits seq {budget:>5}: {sum(x <= budget for x in tot) / len(tot):.0%}")
    except Exception as e:
        print(f"(student-token lengths skipped: {type(e).__name__}: {e})")

    out = pathlib.Path(a.gen).with_suffix(".verified.jsonl")
    with out.open("w") as fh:
        for rec in out_recs:
            fh.write(json.dumps(rec) + "\n")
    summary = {"n": n, "ok": ok, "exec_accuracy": ok / n, "stages": dict(stages),
               "by_db": {d: {"ok": v[0], "n": v[1]} for d, v in sorted(by_db.items())}}
    pathlib.Path(a.gen).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {out}")
    print("DISTILL VERIFY OK")
    return 0


# ---------------------------------------------------------------------------- audit


def _preview(rows, cap: int = 8) -> dict:
    def cell(v):
        s = repr(v) if not isinstance(v, str) else v
        return s if len(s) <= 60 else s[:57] + "..."
    return {"n_rows": len(rows), "n_cols": len(rows[0]) if rows else 0,
            "head": [[cell(v) for v in r] for r in rows[:cap]]}


def cmd_audit(a) -> int:
    """Gold-answer audit, execution half (box). For every row: run gold and the teacher
    SQL, record whether they agree, and keep small result previews so the judge (local)
    can see what each query actually returns, not just its text."""
    from sqlft import bird

    rows = load_rows(a.data)
    gen_recs = [json.loads(l) for l in open(a.gen) if l.strip()]
    gen = {g["idx"]: g for g in gen_recs}
    db_dir = pathlib.Path(a.db_dir)
    cons: dict = {}
    out = pathlib.Path(a.out)
    stages = collections.Counter()
    with out.open("w") as fh:
        jobs = ([(g["idx"], g) for g in gen_recs] if a.only_gen
                else [(i, gen.get(i)) for i in range(len(rows))])
        for n_done, (i, g) in enumerate(jobs):
            row = rows[i]
            db = row["category"]
            if db not in cons:
                cons[db] = bird.duckdb_native(str(db_dir / db / f"{db}.sqlite"))
            con = cons[db]
            rec = {"idx": i, "db": db}
            if g is not None:
                for key in ("sample", "mode", "effort"):
                    if key in g:
                        rec[key] = g[key]
            ordered = bool(re.search(r"\border\s+by\b", row["query"], re.I))
            try:
                ref = bird._run_watchdog(con, row["query"])
                rec["gold"] = _preview(ref)
            except Exception as e:
                ref = None
                rec["gold"] = {"error": str(e).splitlines()[0][:200]}
            if g is None or "error" in g:
                rec["stage"] = "no_teacher"
            else:
                rec["teacher_sql"] = g["sql"]
                try:
                    cand = bird._run_watchdog(con, g["sql"])
                    rec["teacher"] = _preview(cand)
                    if ref is None:
                        rec["stage"] = "gold_error"
                    elif len(cand) > bird.MODEL_SQL_ROW_CAP:
                        rec["stage"] = "wrong"
                    else:
                        rec["stage"] = ("ok" if bird.results_match(ref, cand, ordered)
                                        else "wrong")
                except Exception as e:
                    rec["teacher"] = {"error": str(e).splitlines()[0][:200]}
                    rec["stage"] = "gold_error" if ref is None else "exec_error"
            stages[rec["stage"]] += 1
            fh.write(json.dumps(rec) + "\n")
            if (n_done + 1) % 500 == 0:
                print(f"  audited {n_done + 1}/{len(jobs)}  {dict(stages)}", flush=True)
    print(f"stages: {dict(stages)}")
    print(f"wrote {out}")
    print("DISTILL AUDIT OK")
    return 0


# ---------------------------------------------------------------------------- judge

JUDGE_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

JUDGE_PROMPT = """You are auditing a text-to-SQL benchmark (BIRD, run on DuckDB). Two SQL queries
were written for the same question. They return DIFFERENT results. Decide, for each
query independently, whether it correctly answers the question.

Rules:
- The Evidence line is the benchmark's official external knowledge. A query that
  contradicts the evidence is incorrect, unless the evidence itself contradicts the
  question (then say so).
- Judge semantics: right tables, joins, filters, aggregation, and the columns the question
  asks for. Do NOT penalize harmless differences: column aliases, column order, an extra
  id column alongside the asked-for value, or DISTINCT when duplicates are meaningless.
  DO penalize missing asked-for columns, wrong filters, wrong aggregation, and
  returning many columns when the question asks for one value.
- If the question is genuinely ambiguous so that both readings are defensible, use
  "ambiguous" for both.

### Schema
{schema}

### Question
{question}

### Query A
{sql_a}

Result A: {res_a}

### Query B
{sql_b}

Result B: {res_b}

Briefly analyze both queries (at most ~150 words), then end your reply with a JSON object
on its own, as the LAST thing in your reply:
{{"A": "correct|incorrect|ambiguous", "B": "correct|incorrect|ambiguous",
  "flaw_A": "<=20 words or empty", "flaw_B": "<=20 words or empty",
  "category": "one of: evidence_contradiction, wrong_filter, wrong_join, wrong_aggregation,
  wrong_columns, wrong_table, ambiguous_question, both_wrong, other"}}
"""


def _fmt_res(p: dict) -> str:
    if "error" in p:
        return f"ERROR: {p['error']}"
    s = f"{p['n_rows']} rows x {p['n_cols']} cols; first rows: {json.dumps(p['head'])}"
    return s[:1500]


def cmd_judge(a) -> int:
    """Gold-answer audit, judging half (local, Bedrock). Only rows where teacher and gold
    disagree are judged. Query order is randomized per row and the labels are blind, so
    the judge does not know which query is the benchmark's gold."""
    import boto3
    from botocore.config import Config

    rows = load_rows(a.data)
    audit = [json.loads(l) for l in open(a.audit) if l.strip()]
    todo = [r for r in audit if r["stage"] in ("wrong", "exec_error", "gold_error")]
    if a.limit:
        todo = todo[: a.limit]
    out = pathlib.Path(a.out)
    done = {json.loads(l)["idx"] for l in out.open()} if out.exists() else set()
    todo = [r for r in todo if r["idx"] not in done]
    print(f"{len(audit)} audited, {len(todo)} disagreements to judge with {a.judge_model}")
    client = boto3.client("bedrock-runtime", region_name=a.region,
                          config=Config(read_timeout=300,
                                        retries={"max_attempts": 10, "mode": "adaptive"}))
    lock = threading.Lock()
    fh = out.open("a")

    def one(rec: dict) -> dict:
        row = rows[rec["idx"]]
        rng = random.Random(rec["idx"])
        gold_first = rng.random() < 0.5
        g = (row["query"], _fmt_res(rec["gold"]))
        t = (rec.get("teacher_sql", ""), _fmt_res(rec.get("teacher", {"error": "none"})))
        (sa, ra), (sb, rb) = (g, t) if gold_first else (t, g)
        msg = JUDGE_PROMPT.format(schema=row["schema"].strip(), question=row["prompt"].strip(),
                                  sql_a=sa, res_a=ra, sql_b=sb, res_b=rb)
        res = {"idx": rec["idx"], "db": rec["db"], "stage": rec["stage"],
               "gold_first": gold_first}
        try:
            r = client.converse(modelId=a.judge_model,
                                messages=[{"role": "user", "content": [{"text": msg}]}],
                                inferenceConfig={"maxTokens": 1500, "temperature": 0})
            txt = r["output"]["message"]["content"][0]["text"]
            objs = re.findall(r"\{[^{}]*\}", txt, re.S)
            if not objs:
                raise ValueError(f"no JSON in judge reply: {txt[-200:]!r}")
            v = json.loads(objs[-1])
            res["analysis"] = txt[: txt.rfind(objs[-1])].strip()[-1500:]
            ga, ta = ("A", "B") if gold_first else ("B", "A")
            res.update(gold=v[ga], teacher=v[ta], flaw_gold=v.get(f"flaw_{ga}", ""),
                       flaw_teacher=v.get(f"flaw_{ta}", ""), category=v.get("category"),
                       usage=r.get("usage"))
        except Exception as e:
            res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return res

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for k, f in enumerate(as_completed([ex.submit(one, r) for r in todo]), 1):
            res = f.result()
            with lock:
                fh.write(json.dumps(res) + "\n")
                fh.flush()
            if k % 100 == 0 or k == len(todo):
                print(f"  judged {k}/{len(todo)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    fh.close()
    summarize_judge(audit, out)
    return 0


def summarize_judge(audit: list[dict], path: pathlib.Path) -> None:
    js = [json.loads(l) for l in path.open() if l.strip()]
    js = [j for j in js if "error" not in j]
    n = len(audit)
    agree = sum(r["stage"] == "ok" for r in audit)
    verdict = collections.Counter((j["gold"], j["teacher"]) for j in js)
    gold_bad = sum(j["gold"] == "incorrect" for j in js)
    gold_amb = sum(j["gold"] == "ambiguous" for j in js)
    print(f"\nrows {n}: teacher agrees with gold {agree} ({agree / n:.1%}); judged "
          f"{len(js)} disagreements")
    print("verdicts (gold, teacher):")
    for (g, t), c in verdict.most_common():
        print(f"  gold {g:<10} teacher {t:<10} {c:>5}")
    print(f"gold judged INCORRECT: {gold_bad} = {gold_bad / n:.1%} of all rows; "
          f"ambiguous: {gold_amb} = {gold_amb / n:.1%}")
    cats = collections.Counter(j["category"] for j in js if j["gold"] == "incorrect")
    print(f"gold-incorrect categories: {dict(cats.most_common())}")


# ---------------------------------------------------------------------------- scrub

SCRUB_MODEL = "us.anthropic.claude-opus-4-6-v1"

SCRUB_HIDDEN = {
    "feedback": "an earlier (wrong) attempt at the query and the result that attempt returned "
                "when executed",
    "rationalize": "a verified reference query for the question",
}

SCRUB_PROMPT = """You are editing a reasoning trace for a text-to-SQL training dataset. The trace
was written by a model that could see {hidden}. The student model trained on this trace will
see ONLY the schema, the question and the evidence line below. Rewrite the trace so it is a
faithful first-time solve from that information alone.

Rules:
1. Remove every mention of: a previous/earlier attempt, feedback, a reference/given/provided
   query, a hint, an expected answer, or what any query returned when executed.
2. Where the trace reasons FROM that hidden information (e.g. "the earlier query returned 0
   rows, so the value must be capitalized"), replace it with reasoning justified by the
   schema, question or evidence alone, or delete it. Never invent facts about the data that
   the schema/question/evidence do not show (no claimed sample values, row counts, or
   formats).
3. Keep the original's voice: terse first-person working notes. Same length or shorter.
4. The trace must arrive at exactly the final query given below.
5. Do not launder hidden knowledge. If the final query uses an exact stored value, string
   casing, format or table choice that the trace could only know from the hidden
   information, do NOT justify it as a "convention", "standard", "typical" or "well-known"
   fact. Either (a) the query's own construct already hedges it (LOWER/ILIKE/LIKE '%..%',
   TRY_CAST) and the trace may say the exact form is unknown so it hedges, or (b) the value
   is given in the question/evidence, or (c) it is not justifiable: then use rule 6.
6. If the final query cannot be honestly justified without the hidden information, do not
   rewrite: reply with exactly one line
   UNREPAIRABLE: <short reason>

### Schema
{schema}

### Question
{question}

### Final query (must be the conclusion)
```sql
{sql}
```

### Original trace
<trace>
{trace}
</trace>

Reply with the rewritten trace wrapped in <trace>...</trace> and nothing else, or the single
UNREPAIRABLE line.
"""

LEAK_RE = re.compile(
    r"reference (query|sql|solution|answer)|the reference\b|given (query|answer|solution|sql)|"
    r"provided (query|answer|solution|sql)|\bhint|given the (query|reference|correct)|"
    r"expected (answer|query|result|output)|verified|previous (attempt|query)|prior attempt|"
    r"earlier (attempt|query)|the attempt|feedback|returned \d+ rows|it returned|was wrong",
    re.I)


VALIDATE_PROMPT = """You are checking a reasoning trace in a text-to-SQL training set. The
trace will be used to teach a model that sees ONLY the schema, question and evidence below.
Fail the trace if ANY of these hold:
- it mentions a previous attempt, feedback, a reference/given query, a hint, an expected
  answer, or results of executing a query;
- it states a specific fact about the stored data (an exact value, its casing or format,
  malformed examples, row counts, which of two similar tables holds the data) that is not
  given in the schema, question or evidence and is not genuinely common knowledge. Hedging
  ("the exact casing is unknown, so match case-insensitively") is fine; asserting is not;
- a step is a leap: the conclusion does not follow from what the trace says.
Otherwise pass it. Wording, style and minor redundancy do not matter.

### Schema
{schema}

### Question
{question}

### Trace
<trace>
{trace}
</trace>

### Final query
```sql
{sql}
```

End your reply with a JSON object on its own line:
{{"verdict": "pass|fail", "reason": "<=25 words"}}
"""


def validate_trace(client, model: str, row: dict, trace: str, sql: str) -> dict:
    msg = VALIDATE_PROMPT.format(schema=row["schema"].strip(), question=row["prompt"].strip(),
                                 trace=trace, sql=sql.strip())
    try:
        r = client.converse(modelId=model,
                            messages=[{"role": "user", "content": [{"text": msg}]}],
                            inferenceConfig={"maxTokens": 800, "temperature": 0})
        txt = r["output"]["message"]["content"][0]["text"]
        objs = re.findall(r"\{[^{}]*\}", txt, re.S)
        v = json.loads(objs[-1]) if objs else {"verdict": "fail", "reason": "no JSON"}
        v["usage"] = r.get("usage")
        return v
    except Exception as e:
        return {"verdict": "error", "reason": f"{type(e).__name__}: {str(e)[:150]}"}


def cmd_scrub(a) -> int:
    """Rewrite contaminated traces (feedback / rationalize modes) into clean first-time
    solves. One trace per row: among that row's gold-matching samples, the shortest."""
    import boto3
    from botocore.config import Config

    rows = load_rows(a.data)
    audit = {(r["idx"], r.get("sample", 0)): r
             for r in (json.loads(l) for l in open(a.audit) if l.strip())}
    best: dict[int, dict] = {}
    for g in (json.loads(l) for l in open(a.gen) if l.strip()):
        if audit.get((g["idx"], g.get("sample", 0)), {}).get("stage") != "ok":
            continue
        if g["idx"] not in best or len(g["reasoning"]) < len(best[g["idx"]]["reasoning"]):
            best[g["idx"]] = g
    todo = sorted(best.values(), key=lambda g: g["idx"])
    if a.limit:
        rng = random.Random(a.seed)
        todo = sorted(rng.sample(todo, min(a.limit, len(todo))), key=lambda g: g["idx"])
    out = pathlib.Path(a.out)
    done = {json.loads(l)["idx"] for l in out.open()} if out.exists() else set()
    todo = [g for g in todo if g["idx"] not in done]
    print(f"{len(best)} rows with a gold-matching sample; scrubbing {len(todo)}")
    client = boto3.client("bedrock-runtime", region_name=a.region,
                          config=Config(read_timeout=600,
                                        retries={"max_attempts": 10, "mode": "adaptive"}))
    lock = threading.Lock()
    fh = out.open("a")

    def one(g: dict) -> dict:
        row = rows[g["idx"]]
        msg = SCRUB_PROMPT.format(hidden=SCRUB_HIDDEN[g["mode"]], schema=row["schema"].strip(),
                                  question=row["prompt"].strip(), sql=g["sql"].strip(),
                                  trace=g["reasoning"].strip())
        rec = {"idx": g["idx"], "sample": g.get("sample", 0), "mode": g["mode"],
               "sql": g["sql"], "orig_reasoning": g["reasoning"]}
        try:
            r = client.converse(modelId=a.scrub_model,
                                messages=[{"role": "user", "content": [{"text": msg}]}],
                                inferenceConfig={"maxTokens": 6000, "temperature": 0})
            txt = r["output"]["message"]["content"][0]["text"].strip()
            rec["usage"] = r.get("usage")
            m = re.search(r"<trace>(.*?)</trace>", txt, re.S)
            un = re.search(r"UNREPAIRABLE:\s*(.*)", txt)
            if un and not m:
                rec.update(status="unrepairable", reason=un.group(1).strip()[:300])
            else:
                trace = (m.group(1) if m else txt).strip()
                leak = LEAK_RE.search(trace)
                rec.update(reasoning=trace,
                           status="leak" if leak else "clean",
                           leak=leak.group(0) if leak else None,
                           len_ratio=round(len(trace) / max(1, len(g["reasoning"])), 2))
                if not leak:
                    v = validate_trace(client, a.scrub_model, row, trace, g["sql"])
                    rec["validator"] = v
                    if v.get("verdict") != "pass":
                        rec["status"] = "validator_fail"
        except Exception as e:
            rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:200]}")
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for k, f in enumerate(as_completed([ex.submit(one, g) for g in todo]), 1):
            rec = f.result()
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            if k % 25 == 0 or k == len(todo):
                print(f"  scrubbed {k}/{len(todo)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    fh.close()
    recs = [json.loads(l) for l in out.open()]
    print(f"status: {dict(collections.Counter(r['status'] for r in recs))}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--data", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=100, help="stratified sample size; 0 = all rows")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--model", choices=sorted(MODELS), default="r1")
    g.add_argument("--region", default="us-east-1")
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--max-tokens", type=int, default=8192)
    g.add_argument("--temperature", type=float, default=0.6,
                   help="DeepSeek's recommended R1 sampling temperature")
    g.add_argument("--idx-file", help="whitespace-separated row indices to generate for")
    g.add_argument("--samples", type=int, default=1, help="independent samples per row")
    g.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                   help="gpt-oss reasoning_effort (additionalModelRequestFields)")
    g.add_argument("--mode", choices=["plain", "feedback", "rationalize"], default="plain",
                   help="plain: no hints. feedback: show the teacher's earlier attempt and "
                        "its result (never the gold). rationalize: show the gold (STaR); the "
                        "trace is filtered for leakage downstream")
    g.add_argument("--prev-audit", help="feedback mode: audit jsonl with the earlier attempt")
    v = sub.add_parser("verify")
    v.add_argument("--data", required=True)
    v.add_argument("--gen", required=True)
    v.add_argument("--db-dir", default="/work/bird/train/train_databases")
    au = sub.add_parser("audit")
    au.add_argument("--data", required=True)
    au.add_argument("--gen", required=True)
    au.add_argument("--out", required=True)
    au.add_argument("--db-dir", default="/work/bird/train/train_databases")
    au.add_argument("--only-gen", action="store_true", help="audit only rows in --gen")
    j = sub.add_parser("judge")
    j.add_argument("--data", required=True)
    j.add_argument("--audit", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--judge-model", default=JUDGE_MODEL)
    j.add_argument("--region", default="us-east-1")
    j.add_argument("--workers", type=int, default=8)
    j.add_argument("--limit", type=int, default=0)
    sc = sub.add_parser("scrub")
    sc.add_argument("--data", required=True)
    sc.add_argument("--gen", required=True)
    sc.add_argument("--audit", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--scrub-model", default=SCRUB_MODEL)
    sc.add_argument("--region", default="us-east-1")
    sc.add_argument("--workers", type=int, default=8)
    sc.add_argument("--limit", type=int, default=0, help="random pilot of N rows")
    sc.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    return {"gen": cmd_gen, "verify": cmd_verify, "audit": cmd_audit,
            "judge": cmd_judge, "scrub": cmd_scrub}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
