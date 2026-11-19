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

from sqlft import bird, chat, data, infer
from sqlft.eval_sql import _wilson, check_sql, extract_sql
from sqlft.lora import load_lora, merge_and_unwrap_lora
from sqlft.model import load_model

RUNS = pathlib.Path("/work/runs")


def summarize(per_row: list[dict], args, tag: str) -> dict:
    """Aggregate per-row outcomes, print the report, write runs/<tag>.json."""
    n = len(per_row)
    stages = collections.Counter(r["stage"] for r in per_row)
    n_exec = stages["ok"]
    n_parse = sum(r["parses"] for r in per_row)
    n_cut = sum(r["prompt_cut"] for r in per_row)
    by_db: dict[str, collections.Counter] = {}
    for r in per_row:
        by_db.setdefault(r["db_id"], collections.Counter())[r["stage"]] += 1

    result = {
        "n": n,
        "exec_accuracy": n_exec / n,
        "exec_ci95": _wilson(n_exec, n),
        "parse_rate": n_parse / n,
        "n_prompt_cut": n_cut,
        "exec_accuracy_uncut": (
            sum(r["exec_ok"] for r in per_row if not r["prompt_cut"]) / max(1, n - n_cut)),
        "stages": dict(stages),
        "by_db": {db: dict(c) for db, c in sorted(by_db.items())},
        "adapter": str(args.adapter) if args.adapter else None,
        "args": vars(args),
        "per_row": per_row,
    }
    if any("think_tokens" in r for r in per_row):
        tt = sorted(r["think_tokens"] for r in per_row)
        result["think_tokens_p50"] = tt[len(tt) // 2]
        result["think_tokens_p90"] = tt[int(0.9 * (len(tt) - 1))]
        result["think_unclosed"] = sum(not r["think_closed"] for r in per_row)
        print(f"thinking tokens p50 {result['think_tokens_p50']}  p90 "
              f"{result['think_tokens_p90']}  unclosed (budget hit) {result['think_unclosed']}/{n}")
    lo, hi = result["exec_ci95"]
    print(f"\nn={n}  exec_accuracy={result['exec_accuracy']:.3f} [{lo:.3f}, {hi:.3f}]  "
          f"parse_rate={result['parse_rate']:.3f}")
    print(f"stages: {dict(stages)}")
    print(f"prompt-truncated rows: {n_cut}  "
          f"exec_accuracy on untruncated rows: {result['exec_accuracy_uncut']:.3f}")
    for db, c in sorted(by_db.items()):
        total = sum(c.values())
        print(f"  {db:26} {c.get('ok', 0)}/{total} exec")

    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{tag}.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {RUNS / f'{tag}.json'}")
    return result


def merge_shards(args) -> int:
    """Re-interleave <tag>_s{k}of{n}.json back into the unsharded row order.

    Shard k holds questions[k::n], so row i of the full eval is shard i % n, row i // n.
    The merged per_row is therefore row-aligned with an unsharded run of the same --limit
    and --seed, which is what the paired test requires.
    """
    n = args.merge_shards
    shards = [json.loads((RUNS / f"{args.tag}_s{k}of{n}.json").read_text())["per_row"]
              for k in range(n)]
    total = sum(len(s) for s in shards)
    per_row = [shards[i % n][i // n] for i in range(total)]
    expect = [q["question_id"] for q in
              bird.stratified_subset(bird.load_questions(), args.limit, seed=args.seed)]
    assert [r["question_id"] for r in per_row] == expect, "merged order != unsharded order"
    summarize(per_row, args, args.tag)
    print("BIRD EXEC EVAL OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=pathlib.Path, default=None,
                    help="LoRA adapter; omit for the base model")
    ap.add_argument("--limit", type=int, default=128,
                    help="stratified subset size; >= 1493 evaluates the full dev set")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-targets", choices=("all", "mixers"), default="all",
                    help="must match the adapter's training targets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=("plain", "chat-think", "chat-nothink"),
                    default="plain",
                    help="plain: the fine-tune's '-- DuckDB SQL generation' completion format "
                         "(thinking bypassed). chat-*: Qwen3.5 chat template with the lab14 "
                         "teacher instructions, thinking on/off (sqlft.chat)")
    ap.add_argument("--tag", default="lab10_bird_exec")
    ap.add_argument("--warmup-only", action="store_true",
                    help="compile every prompt-length bucket (one synthetic prompt per "
                         "128-token bucket) and exit. Run ONCE, alone, before launching "
                         "evals in parallel: concurrent processes that hit a new bucket "
                         "at the same moment race on the NKI kernel cache (NCC_EVRF059)")
    ap.add_argument("--warmup-max", type=int, default=0,
                    help="warm buckets only up to this prompt length (0 = seq - 128). On the "
                         "9B at seq 4096, single-core prefill OOMs above ~2.9k tokens; every "
                         "dev prompt is <= ~2.5k tokens even in chat format")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="evaluate questions[shard::num_shards] only (one shard per core); "
                         "writes <tag>_s{shard}of{n}.json -- merge with --merge-shards n")
    ap.add_argument("--merge-shards", type=int, default=0,
                    help="no model: interleave <tag>_s*of{n}.json into <tag>.json")
    args = ap.parse_args()
    if args.merge_shards:
        return merge_shards(args)

    questions = bird.stratified_subset(bird.load_questions(), args.limit, seed=args.seed)
    questions = questions[args.shard::args.num_shards]
    ddl_by_db = {}
    cons = {}
    for q in questions:
        db = q["db_id"]
        if db not in ddl_by_db:
            path = str(bird.DB_DIR / db / f"{db}.sqlite")
            ddl_by_db[db] = bird.schema_ddl(path)
            cons[db] = bird.duckdb_native(path)

    tok = data.get_tokenizer()
    plain = [bird.render_prompt(q["question"], q.get("evidence", ""), ddl_by_db[q["db_id"]])
             for q in questions]
    thinking = args.format == "chat-think"
    prompts = (plain if args.format == "plain" else
               [chat.render_chat_prompt(tok, p, thinking=thinking) for p in plain])
    stop_ids = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    # cached_generate keeps a prompt's LAST seq - 128 tokens, so anything longer loses
    # the head of its schema DDL. Recorded per row: a truncated-schema miss is a context
    # budget casualty, not (only) a model error.
    prompt_tokens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    prompt_cut = [t > args.seq - 128 for t in prompt_tokens]
    if any(prompt_cut):
        print(f"note: {sum(prompt_cut)}/{len(prompts)} prompts exceed seq-128 = "
              f"{args.seq - 128} tokens and will be left-truncated (schema head lost)")

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

    if args.warmup_only:
        # One prompt that lands exactly in each bucket W = 128, 256, ..., seq - 128 (the
        # longest prompt cached_generate keeps). Shapes, not content, decide the compiles.
        filler = tok("SELECT 1;", add_special_tokens=False)["input_ids"]
        warm = []
        top = args.warmup_max or (args.seq - 128)
        for W in range(128, min(top, args.seq - 128) + 1, 128):
            # W - 64: centred in the bucket, so re-tokenizing the decoded text (merges
            # can shift the count by a few tokens) cannot spill into a neighbour.
            ids = (filler * (W // len(filler) + 1))[: W - 64]
            warm.append(tok.decode(ids))
        _, _ = infer.cached_generate(model, tok, warm, max_new_tokens=4, seq_len=args.seq)
        print(f"warmed {len(warm)} buckets at seq {args.seq}")
        print("BIRD WARMUP OK")
        return 0

    gens, finished = infer.cached_generate(model, tok, prompts,
                                           max_new_tokens=args.max_new, seq_len=args.seq,
                                           stop_ids=stop_ids)

    per_row = []
    for q, gen, fin, n_tok, cut in zip(questions, gens, finished, prompt_tokens, prompt_cut):
        think, answer, closed = chat.split_thinking(gen, thinking=thinking)
        sql = extract_sql(answer)
        con = cons[q["db_id"]]
        verdict = check_sql(sql)  # parse stage only, no row schema here
        out = {"question_id": q["question_id"], "db_id": q["db_id"],
               "parses": verdict.parses, "truncated": not fin,
               "prompt_tokens": n_tok, "prompt_cut": cut,
               "generated": gen}
        if thinking:
            out["think_tokens"] = len(tok(think, add_special_tokens=False)["input_ids"])
            out["think_closed"] = closed
        if thinking and not closed:
            out["stage"] = "truncated"
            out["error"] = "thinking budget exhausted before </think>"
        elif verdict.parses:
            out.update(bird.grade_one(con, sql, q["SQL_duckdb"]))
        else:
            out["stage"] = "truncated" if not fin else "parse_error"
            out["error"] = verdict.error
        # The paired-test key (sqlft.eval_sql --compare --key exec_ok).
        out["exec_ok"] = out["stage"] == "ok"
        per_row.append(out)

    summarize(per_row, args, args.tag if args.num_shards == 1 else
              f"{args.tag}_s{args.shard}of{args.num_shards}")
    print("BIRD EXEC EVAL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
