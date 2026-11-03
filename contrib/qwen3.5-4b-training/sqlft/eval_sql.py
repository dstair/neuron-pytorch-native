"""Does the model emit syntactically correct DuckDB SQL? Two graded checks.

PARSE RATE (the headline metric -- "syntactically correct"):
    `duckdb.connect().extract_statements(sql)` runs DuckDB's real parser and nothing else.
    It raises ParserException on a syntax error and is happy about unknown tables, so it
    measures syntax alone. It also handles every statement kind in this dataset (PRAGMA,
    CREATE, SELECT, ...), unlike `json_serialize_sql`, which is SELECT-only.

BIND RATE (a strictly harder bonus):
    Create the row's own CREATE TABLE schema in an in-memory database, then EXPLAIN the
    query. This additionally requires that every table and column referenced exists. A
    query can parse but not bind -- that gap is the interesting part of the report.

Deliberately NOT execution accuracy: the dataset ships no data, so there is nothing to
compare result sets against. Saying "syntactically correct" and measuring exactly that.

    ./env/container.sh exec python3 -m sqlft.eval_sql --self-test
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import re

import torch

RUNS = pathlib.Path("/work/runs")


@dataclasses.dataclass
class Verdict:
    parses: bool
    binds: bool
    error: str = ""
    # True when the DATASET's own schema will not create, so this row could never bind no
    # matter what the model emits. Decided from the schema ALONE -- independent of whether
    # the model's output parses -- so the bind denominator is the same set of rows for
    # every model and every run.
    schema_unusable: bool = False
    # True when generation hit the token cap without emitting EOS. A truncation that fails
    # to parse ("syntax error at end of input") is a decode-cap casualty, not a syntax
    # failure, so it is reported as its own stage.
    truncated: bool = False

    @property
    def stage(self) -> str:
        if not self.parses:
            return "truncated" if self.truncated else "parse_error"
        if self.schema_unusable:
            return "schema_unusable"
        return "ok" if self.binds else "bind_error"


def _first_line(exc: BaseException) -> str:
    return str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__


def _is_syntax_error(exc: BaseException) -> bool:
    """Only a ParserException means "not syntactically correct".

    `extract_statements` is not purely syntactic: for some statements -- notably
    `PRAGMA drop_fts_index('x')` and `DROP EXTENSION tpch` -- it resolves against the
    catalog and raises CatalogException / NotImplementedException. Measured on this dataset,
    treating every exception as a syntax error understated the gold parse rate. Those
    statements ARE well-formed SQL; they just refer to things that do not exist here.
    """
    import duckdb

    return isinstance(exc, duckdb.ParserException)


def check_sql(sql: str, schema: str | None = None, *, finished: bool = True) -> Verdict:
    import duckdb

    sql = sql.strip()

    # Schema usability is a property of the DATASET ROW, decided once up front and
    # independent of whether the model's SQL parses -- INCLUDING the degenerate case of
    # an empty generation, which must NOT return early before this check (an untuned
    # model emits comment-only output that extracts to empty; those rows still need
    # their quarantine counted or the bind denominator changes with the model).
    schema_unusable = False
    schema_err = ""
    con = duckdb.connect()
    try:
        if schema is None or not schema.strip():
            # ~8% of eval rows ship NO schema. There is nothing to bind against, so the
            # row is unanswerable at the bind stage -- quarantine it like any other
            # broken schema rather than charging a bind_error to the model.
            schema_unusable, schema_err = True, "dataset schema: none supplied"
        else:
            try:
                con.execute(schema)
            except Exception as e:
                schema_unusable, schema_err = True, f"dataset schema: {_first_line(e)}"

        if not sql:
            return Verdict(False, False, "empty output", schema_unusable,
                           truncated=not finished)

        try:
            stmts = con.extract_statements(sql)
        except Exception as e:
            if _is_syntax_error(e):
                return Verdict(False, False, _first_line(e), schema_unusable,
                               truncated=not finished)
            # Parsed fine; the failure is semantic (unknown catalog object, unimplemented).
            return Verdict(True, False, _first_line(e), schema_unusable)

        if schema_unusable:
            return Verdict(True, False, schema_err, schema_unusable=True)

        try:
            for stmt in stmts:
                con.execute("EXPLAIN " + stmt.query.rstrip().rstrip(";"))
        except Exception as e:
            return Verdict(True, False, _first_line(e))
        return Verdict(True, True)
    finally:
        con.close()


# The model may wrap SQL in a fence, add prose, or ramble past the statement. Extracting
# is part of the harness, not the metric -- we grade the SQL the model meant to emit.
_FENCE = re.compile(r"```(?:sql)?\s*(.+?)```", re.S | re.I)


def extract_sql(text: str) -> str:
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    # Our prompt format is line-comment delimited; drop any trailing commentary the model
    # starts after finishing the statement.
    text = text.split("-- DuckDB SQL")[0]
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("--")]
    return "\n".join(lines).strip()


def extract_first_statement(text: str) -> str:
    """Just the first statement, up to and including the first `;`.

    The untuned model's dominant failure is not stopping: it emits a VALID statement and
    then rambles (a second question, a second query). `parse_rate` grades the whole output;
    `first_stmt_parse_rate` grades only the first statement, which separates "cannot write
    SQL" from "has not learned to stop" -- the EOS confound, made a number.
    """
    sql = extract_sql(text)
    head, sep, _ = sql.partition(";")
    return head + sep


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """95% Wilson score interval for a binomial proportion.

    The reporting companion to n=128: a point estimate of 0.90 without its interval reads
    as far more precise than it is (here, roughly +-5 points). Wilson behaves well at the
    high rates this harness reports, where the normal approximation does not.
    """
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def evaluate(pairs: list[tuple], *, verbose: bool = False) -> dict:
    """pairs = [(generated_sql, schema_or_None), ...] or
              [(generated_sql, schema_or_None, finished_bool), ...]

    `finished` is False when the decode hit the token cap without EOS; a parse failure on
    such a row is reported as `truncated`, not `parse_error`. 2-tuples mean finished=True
    (e.g. the gold ceiling, where there is no decode at all).
    """
    triples = [(p[0], p[1], p[2] if len(p) > 2 else True) for p in pairs]
    verdicts = [check_sql(extract_sql(sql), schema, finished=fin)
                for sql, schema, fin in triples]
    # "Would the first statement alone have parsed?" -- the EOS-confound probe. Graded
    # without a schema, so it is purely about syntax.
    first_parses = [
        v.parses or check_sql(extract_first_statement(sql)).parses
        for v, (sql, _s, _f) in zip(verdicts, triples)
    ]
    n = len(verdicts) or 1
    stages: dict[str, int] = {}
    for v in verdicts:
        stages[v.stage] = stages.get(v.stage, 0) + 1

    # bind_rate is over rows whose schema actually creates; anything else is unanswerable.
    # The quarantine is decided from the schema alone, so this set is IDENTICAL for every
    # model and every run on the same eval rows.
    bindable = [v for v in verdicts if not v.schema_unusable]
    n_parse = sum(v.parses for v in verdicts)
    n_bind = sum(v.binds for v in bindable)
    result = {
        "n": len(verdicts),
        "parse_rate": n_parse / n,
        "parse_ci95": _wilson(n_parse, len(verdicts)),
        "first_stmt_parse_rate": sum(first_parses) / n,
        "bind_rate": (n_bind / len(bindable)) if bindable else None,
        "bind_ci95": _wilson(n_bind, len(bindable)) if bindable else (None, None),
        "n_bindable": len(bindable),
        "stages": stages,
        "sample_errors": [v.error for v in verdicts
                          if v.error and not v.schema_unusable][:5],
        # Per-row outcomes, so two runs on the same rows can be compared PAIRED
        # (compare_runs) instead of by their point estimates alone.
        "per_row": [
            {"stage": v.stage, "parses": v.parses, "binds": v.binds,
             "schema_unusable": v.schema_unusable,
             "first_stmt_parses": fp, "truncated": v.truncated, "error": v.error,
             "generated": sql}
            for v, fp, (sql, _s, _f) in zip(verdicts, first_parses, triples)
        ],
    }
    if verbose:
        lo, hi = result["parse_ci95"]
        br = "n/a" if result["bind_rate"] is None else (
            f"{result['bind_rate']:.3f} [{result['bind_ci95'][0]:.3f}, "
            f"{result['bind_ci95'][1]:.3f}]")
        print(f"n={result['n']}  parse_rate={result['parse_rate']:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"first_stmt={result['first_stmt_parse_rate']:.3f}  "
              f"bind_rate={br} (of {result['n_bindable']} bindable)  {result['stages']}")
        for e in result["sample_errors"]:
            print(f"   e.g. {e}")
    return result


def gold_ceiling(rows, limit: int | None = None) -> dict:
    """What would a model that reproduced the gold SQL exactly score?

    2.78% of this dataset's gold queries do not parse in DuckDB 1.5.5 (malformed synthetic
    SQL, and one row that is Python rather than SQL). So parse_rate has a ceiling below
    1.0, and quoting a model number without it invites the wrong conclusion.
    """
    rows = list(rows)[:limit] if limit else list(rows)
    return evaluate([(r.query, r.schema) for r in rows])


# ------------------------------------------------------------------ generation


def position_ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Mask-aware position ids: every row's first REAL token is position 0.

    Necessary because the model defaults to `position_ids = arange(seq_len)`, which is not
    mask-aware. With left padding that shifts each row's real tokens by however much padding
    it happens to carry, so a short prompt in a batch of long ones gets its RoPE phases
    offset and generates differently than it would alone. `lab08_eval_speed.py` demonstrates
    that batch-size dependence and that this removes it.
    """
    return (mask.long().cumsum(-1) - 1).clamp(min=0)


@torch.no_grad()
def greedy_generate(model, tokenizer, prompts: list[str], max_new_tokens: int = 96,
                    seq_len: int = 1024, device="neuron", batch_size: int = 8,
                    progress: bool = True, chunk: int = 128,
                    return_finished: bool = False) -> list[str]:
    """Batched greedy decode, no KV cache, but only as wide as it needs to be.

    Why no cache: the trainable NKI GDN kernel takes a full chunk of >=128 tokens and has no
    single-token decode path, and `use_cache=True` needs one. Re-running the prefix is O(n^2)
    but reuses the exact graph we trained with, so eval cannot silently diverge from training.

    Two things make it far cheaper than the naive version (kept below as
    `greedy_generate_padded`):

    1. **Bucketing.** The old loop always ran a full `seq_len`-wide forward even when content
       was ~400 tokens. Here the tensor is the smallest multiple of `chunk` that holds the
       content, so a 350-token prompt computes 384 columns instead of 1024. `chunk` must stay
       128 -- the GDN kernel requires it.
    2. **`logits_to_keep=1`.** Content is RIGHT-aligned so every row's newest token lands on
       the final column; the model then computes logits for that one position instead of all
       of them. At bs 8 x seq 1024 the fp32 logits over a 248,320 vocab are **8.1 GB**; this
       makes them ~8 MB.

    Rows that already emitted EOS keep being extended (their output is truncated at the first
    EOS when decoding). Keeping all rows the same length is what lets them share a single
    right edge, which is what makes `logits_to_keep=1` valid for the whole batch.
    """
    import time

    from sqlft.device import get_device, sync

    dev = get_device(str(device))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos = tokenizer.eos_token_id
    outputs: list[str] = []
    finished: list[bool] = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    t_start = time.perf_counter()

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        seqs = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in batch_prompts]
        gen: list[list[int]] = [[] for _ in seqs]
        width = max(len(s) for s in seqs)
        done = torch.zeros(len(seqs), dtype=torch.bool)

        for t in range(max_new_tokens):
            content_len = width + t            # longest row; all rows share the right edge
            if content_len >= seq_len:
                break
            bucket = min(seq_len, -(-content_len // chunk) * chunk)

            ids = torch.full((len(seqs), bucket), pad, dtype=torch.long)
            mask = torch.zeros((len(seqs), bucket), dtype=torch.long)
            for i, (s, g) in enumerate(zip(seqs, gen)):
                row = s + g
                ids[i, bucket - len(row) : bucket] = torch.tensor(row, dtype=torch.long)
                mask[i, bucket - len(row) : bucket] = 1

            out = model(
                input_ids=ids.to(dev),
                attention_mask=mask.to(dev),
                position_ids=position_ids_from_mask(mask).to(dev),
                logits_to_keep=1,
            )
            nxt = out.logits[:, -1, :].argmax(-1).cpu()
            sync(dev)

            for i in range(len(seqs)):
                gen[i].append(int(nxt[i]))
            done |= nxt == eos

            if progress and (t % 16 == 0 or bool(done.all())):
                el = time.perf_counter() - t_start
                print(f"  gen batch {start // batch_size + 1}/{n_batches} "
                      f"token {t + 1}/{max_new_tokens}  done={int(done.sum())}/{len(seqs)}  "
                      f"bucket={bucket}  elapsed {el / 60:.1f} min", flush=True)
            if bool(done.all()):
                break

        for g in gen:
            finished.append(eos in g)
            if eos in g:
                g = g[: g.index(eos)]
            outputs.append(tokenizer.decode(g, skip_special_tokens=True))
    return (outputs, finished) if return_finished else outputs


@torch.no_grad()
def greedy_generate_padded(model, tokenizer, prompts: list[str], max_new_tokens: int = 96,
                           seq_len: int = 1024, device="neuron", batch_size: int = 8,
                           progress: bool = True) -> list[str]:
    """The original full-width decode. Kept ONLY as a reference for `lab08_eval_speed.py`.

    Always runs a `seq_len`-wide forward and materialises logits for every position, and does
    not pass mask-aware position ids -- so its output depends on batch composition. Do not use
    it for reported numbers.
    """
    import time

    from sqlft.device import get_device, sync

    dev = get_device(str(device))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos = tokenizer.eos_token_id
    outputs: list[str] = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    t_start = time.perf_counter()

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        enc = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in chunk]
        # Left-pad so every row's generation frontier is the same column.
        width = max(len(e) for e in enc)
        ids = torch.full((len(chunk), seq_len), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), seq_len), dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, width - len(e) : width] = torch.tensor(e)
            mask[i, width - len(e) : width] = 1

        done = torch.zeros(len(chunk), dtype=torch.bool)
        for t in range(max_new_tokens):
            pos = width + t
            if pos >= seq_len:
                break
            logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
            nxt = logits[:, pos - 1, :].argmax(-1).cpu()
            sync(dev)
            nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
            ids[:, pos] = nxt
            mask[:, pos] = (~done).long()
            done |= nxt == eos
            # This loop is O(n^2) with no KV cache: 32 layers x up to max_new_tokens forwards
            # per batch. At ~9 s/forward that is over half an hour for 32 prompts, so it must
            # say something -- a silent 40-minute run is indistinguishable from a hang.
            if progress and (t % 16 == 0 or bool(done.all())):
                el = time.perf_counter() - t_start
                print(f"  gen batch {start // batch_size + 1}/{n_batches} "
                      f"token {t + 1}/{max_new_tokens}  done={int(done.sum())}/{len(chunk)}  "
                      f"elapsed {el / 60:.1f} min", flush=True)
            if bool(done.all()):
                break

        for i, e in enumerate(enc):
            gen = ids[i, width:]
            gen = gen[gen != pad]
            outputs.append(tokenizer.decode(gen, skip_special_tokens=True))
    return outputs


def gold_over_cap(rows, tokenizer, max_new_tokens: int) -> int:
    """Rows whose GOLD query is longer than the decode cap -- unwinnable by construction,
    because the model cannot finish a correct answer either. Reported so a `truncated`
    stage can be read against how many rows were even winnable.
    """
    n = 0
    for r in rows:
        if len(tokenizer(r.query, add_special_tokens=False)["input_ids"]) > max_new_tokens:
            n += 1
    return n


def evaluate_model(model, tokenizer, rows, *, limit: int = 100, max_new_tokens: int = 96,
                   seq_len: int = 1024, device="neuron", batch_size: int = 1,
                   tag: str | None = None) -> dict:
    """batch_size defaults to 1 on purpose -- bigger batches are SLOWER here.

    The GDN kernel's outer loop is `for sample_idx in nl.sequential_range(BH)`: it serialises
    over batch x heads, so the 24 linear-attention layers get no parallelism from batching at
    all, while everything else (activations, the causal mask, the logits) grows with it.
    Measured per prompt, warm, 32L, 24 new tokens: fast bs=1 11.0 s, fast bs=8 18.5 s.
    """
    from sqlft.data import render_prompt

    rows = list(rows)[:limit]
    prompts = [render_prompt(r) for r in rows]
    gens, finished = greedy_generate(model, tokenizer, prompts,
                                     max_new_tokens=max_new_tokens, seq_len=seq_len,
                                     device=device, batch_size=batch_size,
                                     return_finished=True)
    result = evaluate([(g, r.schema, f) for g, r, f in zip(gens, rows, finished)],
                      verbose=True)
    result["n_gold_over_cap"] = gold_over_cap(rows, tokenizer, max_new_tokens)
    if result["n_gold_over_cap"]:
        print(f"note: {result['n_gold_over_cap']} gold queries exceed the {max_new_tokens}-token "
              f"decode cap (unwinnable rows)")
    result["samples"] = [
        {"question": r.prompt, "gold": r.query, "generated": g}
        for r, g in list(zip(rows, gens))[:5]
    ]
    if tag:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / f"{tag}.json").write_text(json.dumps(result, indent=2))
        print(f"wrote {RUNS / f'{tag}.json'}")
    return result


# --------------------------------------------------------------------------------------


def _binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value (no scipy on this box)."""
    from math import comb

    def pmf(i):
        return comb(n, i) * p**i * (1 - p) ** (n - i)

    pk = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= pk + 1e-15))


def compare_runs(path_a, path_b, *, key: str = "parses") -> dict:
    """PAIRED comparison of two eval runs on the same rows (McNemar's exact test).

    Point estimates move by a query or two between identical-quality runs (greedy decode
    is chaotic), so 'run B scored 0.883 vs run A's 0.898' is not evidence of anything on
    its own. The paired test asks the right question: of the rows where the two runs
    DISAGREE, is the split consistent with a coin flip? Only a large, lopsided discordant
    count is a real difference.

    `key` is the per-row boolean to compare: "parses" or "binds".
    """
    a = json.loads(pathlib.Path(path_a).read_text())
    b = json.loads(pathlib.Path(path_b).read_text())
    ra, rb = a.get("per_row"), b.get("per_row")
    if not ra or not rb:
        raise ValueError("both runs need per_row records (re-run with the current harness)")
    if len(ra) != len(rb):
        raise ValueError(f"different row counts ({len(ra)} vs {len(rb)}); not the same eval set")

    a_only = sum(1 for x, y in zip(ra, rb) if x[key] and not y[key])
    b_only = sum(1 for x, y in zip(ra, rb) if y[key] and not x[key])
    n_disc = a_only + b_only
    result = {
        "key": key,
        "n": len(ra),
        "a_rate": sum(x[key] for x in ra) / len(ra),
        "b_rate": sum(x[key] for x in rb) / len(rb),
        "a_only": a_only,
        "b_only": b_only,
        "p_value": _binomial_two_sided(min(a_only, b_only), n_disc) if n_disc else 1.0,
        "flipped_rows": [i for i, (x, y) in enumerate(zip(ra, rb)) if x[key] != y[key]],
    }
    print(f"{path_a.name} vs {path_b.name}  ({key})")
    print(f"  rates: {result['a_rate']:.3f} vs {result['b_rate']:.3f}  "
          f"discordant: {a_only} vs {b_only} (of {result['n']})  "
          f"McNemar p = {result['p_value']:.3f}")
    print("  -> " + ("no significant difference (p >= 0.05)"
                       if result["p_value"] >= 0.05 else "SIGNIFICANT difference (p < 0.05)"))
    return result


# --------------------------------------------------------------------------------------


def _self_test() -> int:
    """Validate the HARNESS before trusting it about the model.

    If ground-truth SQL does not score ~100% parse, the harness is broken, not the model.
    """
    from sqlft.data import load_rows, split_rows

    rows = load_rows()
    _, evl = split_rows(rows)
    gold = evl[:300]

    res = evaluate([(r.query, r.schema) for r in gold], verbose=True)
    # Not 1.0: ~2.8% of this dataset's gold SQL is genuinely unparseable (see gold_ceiling).
    # 0.95 is the "harness is sane" floor, not a claim the data is clean.
    assert res["parse_rate"] > 0.95, f"gold SQL only parses at {res['parse_rate']:.3f} -- harness bug"
    print(f"gold parse_rate {res['parse_rate']:.3f}  <- this is the CEILING for any model")

    # Known-bad output must be rejected, or the metric is vacuous.
    bad = [
        ("SELEC * FROM t", None),
        ("SELECT FROM WHERE", None),
        ("", None),
        ("I think the answer is to select everything.", None),
    ]
    bad_res = evaluate(bad)
    assert bad_res["parse_rate"] == 0.0, f"garbage scored {bad_res['parse_rate']} parse rate"
    print(f"garbage parse_rate {bad_res['parse_rate']:.3f} (must be 0.0)")

    # Fenced / chatty output must still be graded on the SQL inside it.
    fenced = "Sure!\n```sql\nSELECT 1;\n```\nHope that helps."
    assert check_sql(extract_sql(fenced)).parses, "fence extraction failed"
    print("fence + prose extraction: ok")

    # Parse-vs-bind must actually be distinguishable, or bind_rate is meaningless.
    schema = 'CREATE TABLE t ("a" INTEGER);'
    assert check_sql("SELECT a FROM t", schema).stage == "ok"
    assert check_sql("SELECT nope FROM t", schema).stage == "bind_error"
    assert check_sql("SELEC a FROM t", schema).stage == "parse_error"
    print("parse_error / bind_error / ok are distinguishable")

    # An uncreatable dataset schema must be quarantined, not charged to the model --
    # whether or not the model's SQL parses (the quarantine is a property of the ROW).
    broken_schema = 'CREATE TABLE a ("x" INT, FOREIGN KEY ("x") REFERENCES nosuch("y"));'
    v = check_sql("SELECT x FROM a", broken_schema)
    assert v.schema_unusable and v.stage == "schema_unusable", v
    v2 = check_sql("SELEC x FROM a", broken_schema)
    assert v2.schema_unusable and v2.stage == "parse_error", v2
    res = evaluate([("SELEC x FROM a", broken_schema), ("SELECT 1", "CREATE TABLE t (x INT)")])
    assert res["n_bindable"] == 1, res["n_bindable"]  # quarantined despite the parse error
    v3 = check_sql("SELECT 1", None)
    assert v3.schema_unusable, v3  # no schema at all: unanswerable, not a model failure
    # ...including an empty generation (comment-only model output extracts to "").
    v4 = check_sql("", broken_schema)
    assert v4.schema_unusable and v4.stage == "parse_error", v4
    res = evaluate([("-- just a comment", broken_schema), ("SELECT 1", "CREATE TABLE t (x INT)")])
    assert res["n_bindable"] == 1, res["n_bindable"]
    print("uncreatable/missing dataset schema is quarantined regardless of model output, "
          "even an empty one")

    # A decode-cap casualty is reported as truncated, not as a syntax failure.
    v = check_sql("SELECT * FROM t WHERE x = 'abc", finished=False)
    assert v.stage == "truncated", v
    v = check_sql("SELECT * FROM t WHERE x = 'abc", finished=True)
    assert v.stage == "parse_error", v
    print("truncation is its own stage, not a parse error")

    # The EOS-confound probe: a valid statement followed by rambling fails parse_rate but
    # must pass first_stmt_parse_rate.
    rambled = "SELECT 1; SELECT FROM WHERE garbage"
    res = evaluate([(rambled, None)])
    assert res["parse_rate"] == 0.0 and res["first_stmt_parse_rate"] == 1.0, res
    print("first-statement probe separates 'cannot write SQL' from 'has not learned to stop'")

    # The CI is honest: at n=128 a 0.90 rate has an interval several points wide, and
    # every run carries per-row records for paired comparison.
    res = evaluate([("SELECT 1", None)] * 115 + [("SELEC", None)] * 13)
    lo, hi = res["parse_ci95"]
    assert lo < 0.898 < hi and hi - lo > 0.03, res["parse_ci95"]
    assert len(res["per_row"]) == 128
    assert res["per_row"][0]["stage"] == "schema_unusable"  # SELECT 1, no schema to bind
    assert res["per_row"][-1]["stage"] == "parse_error"      # SELEC
    print(f"Wilson CI on 115/128 parses: [{lo:.3f}, {hi:.3f}]; per_row records present")

    # Well-formed SQL that merely refers to absent catalog objects must NOT count as a
    # syntax error -- this is the bug the gold parse rate exposed.
    v = check_sql("PRAGMA drop_fts_index('artist');")
    assert v.parses, f"catalog error misclassified as syntax error: {v}"
    print("catalog/unimplemented errors are not counted as syntax errors")

    print("EVAL SELF-TEST OK")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A.json", "RUN_B.json"),
                    help="paired (McNemar) comparison of two eval run files")
    ap.add_argument("--key", choices=("parses", "binds"), default="parses")
    args = ap.parse_args()
    if args.compare:
        raise SystemExit(0 if compare_runs(*args.compare, key=args.key)["p_value"] >= 0
                         else 1)
    raise SystemExit(_self_test())
