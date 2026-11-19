"""lab15: build the reasoning-trace fine-tune set (Tier A) in the Qwen3.5 chat format.

Input
  /work/runs/trace_pool_train.jsonl  (built from the lab14 outputs; every trace is
      gpt-oss-120b's own, unedited, generated WITHOUT hints; tier A = its SQL's result
      matched the verified gold on the real database)
  /work/bird/bird_train_train.jsonl  (schema / question+evidence / gold, row-aligned by idx)

Output
  /work/bird/bird_train_traces_tierA.jsonl: sqlft.data rows with prompt_text /
  completion_text pre-rendered:
      prompt_text      chat template, user = sqlft.chat.PREAMBLE + plain eval prompt,
                       assistant opened with `<think>\\n` (thinking on)
      completion_text  {trace}\\n</think>\\n\\n```sql\\n{teacher sql}\\n```   (+ eos in the loader)
  The target SQL is the TEACHER's (the trace's own conclusion); for tier A its result
  equals the gold's, so this loses nothing and keeps trace and answer consistent.

Gates (all asserted):
  - sqlft.chat.PREAMBLE == labs/lab14_distill.PREAMBLE (the conditioning the trace was
    generated under)
  - the training prompt for a row is byte-identical to what lab10 --format chat-think
    renders for the same question/evidence/schema
  - no train/dev overlap (databases, question text)

    ./env/container.sh exec python3 labs/lab15_trace_data.py [--tiers A] [--sources first_pass,s1_resample]
"""

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import lab14_distill  # noqa: E402
from sqlft import bird, chat, data  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/work/runs/trace_pool_train.jsonl")
    ap.add_argument("--train", default="/work/bird/bird_train_train.jsonl")
    ap.add_argument("--out", default="/work/bird/bird_train_traces_tierA.jsonl")
    ap.add_argument("--tiers", default="A")
    ap.add_argument("--sources", default="first_pass,s1_resample")
    args = ap.parse_args()

    assert chat.PREAMBLE == lab14_distill.PREAMBLE, "teacher/student preamble drift"
    tok = data.get_tokenizer()
    train = [json.loads(l) for l in open(args.train)]
    tiers, sources = set(args.tiers.split(",")), set(args.sources.split(","))
    pool = [json.loads(l) for l in open(args.pool)]
    pool = [p for p in pool if p["tier"] in tiers and p["source"] in sources]

    dev = bird.load_questions()
    dev_dbs = {q["db_id"] for q in dev}
    dev_qs = {q["question"].strip() for q in dev}

    lens, rows, by_src = [], [], collections.Counter()
    for p in pool:
        t = train[p["idx"]]
        assert t["category"] not in dev_dbs, f"dev database in train: {t['category']}"
        assert t["prompt"].split("\n-- Evidence: ")[0].strip() not in dev_qs, "dev question in train"
        plain = lab14_distill.render_student_prompt(t)
        prompt_text = chat.render_chat_prompt(tok, plain, thinking=True)
        completion = chat.render_think_completion(p["reasoning"], p["sql"])
        n_p = len(tok(prompt_text, add_special_tokens=False)["input_ids"])
        n_c = len(tok(completion, add_special_tokens=False)["input_ids"]) + 1  # + eos
        lens.append((n_p, n_c, p["source"]))
        by_src[p["source"]] += 1
        rows.append({"prompt": t["prompt"], "query": p["sql"].strip(), "schema": t["schema"],
                     "category": t["category"], "prompt_text": prompt_text,
                     "completion_text": completion, "idx": p["idx"], "source": p["source"]})

    # Train/eval prompt parity: lab10 renders bird.render_prompt(question, evidence, ddl);
    # lab14 rendered from the train jsonl. Check one row per database rebuilds identically
    # from bird.render_prompt with the jsonl's own schema.
    seen = set()
    for r in rows:
        if r["category"] in seen:
            continue
        seen.add(r["category"])
        q, _, ev = r["prompt"].partition("\n-- Evidence: ")
        eval_style = chat.render_chat_prompt(
            tok, bird.render_prompt(q, ev, r["schema"]), thinking=True)
        assert eval_style == r["prompt_text"], f"train/eval prompt drift on {r['category']}"

    q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))]
    tot = [a + b for a, b, _ in lens]
    print(f"rows {len(rows)} by source {dict(by_src)}  (parity checked on {len(seen)} dbs)")
    for src in sorted(by_src):
        c = [b for a, b, s in lens if s == src]
        print(f"  {src:12} completion tokens p50 {q(c, .5)}  p90 {q(c, .9)}  max {max(c)}")
    print(f"prompt+completion p50 {q(tot, .5)}  p90 {q(tot, .9)}  max {max(tot)}")
    for seq in (1664, 2048, 2560, 3072):
        k = sum(x <= seq for x in tot)
        ks = {s: sum(a + b <= seq for a, b, s2 in lens if s2 == s) for s in by_src}
        print(f"  fits seq {seq}: {k}/{len(tot)} ({k / len(tot):.1%})  by source {ks}")

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}")
    print("TRACE DATA OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
