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
    # matter what the model emits. Excluded from the bind denominator.
    schema_unusable: bool = False

    @property
    def stage(self) -> str:
        if not self.parses:
            return "parse_error"
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


def check_sql(sql: str, schema: str | None = None) -> Verdict:
    import duckdb

    sql = sql.strip()
    if not sql:
        return Verdict(False, False, "empty output")

    con = duckdb.connect()
    try:
        stmts = con.extract_statements(sql)
    except Exception as e:
        if _is_syntax_error(e):
            return Verdict(False, False, _first_line(e))
        # Parsed fine; the failure is semantic (unknown catalog object, unimplemented).
        return Verdict(True, False, _first_line(e))

    if schema is None:
        return Verdict(True, False, "no schema supplied; bind not attempted")

    try:
        # Trusted dataset input. ~25% of these schemas do not create (FKs referencing
        # tables that were never emitted), so a failure here is NOT the model's fault.
        con.execute(schema)
    except Exception as e:
        return Verdict(True, False, f"dataset schema: {_first_line(e)}", schema_unusable=True)

    try:
        for stmt in stmts:
            con.execute("EXPLAIN " + stmt.query.rstrip().rstrip(";"))
    except Exception as e:
        return Verdict(True, False, _first_line(e))
    return Verdict(True, True)


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


def evaluate(pairs: list[tuple[str, str | None]], *, verbose: bool = False) -> dict:
    """pairs = [(generated_sql, schema_or_None), ...]"""
    verdicts = [check_sql(extract_sql(sql), schema) for sql, schema in pairs]
    n = len(verdicts) or 1
    stages: dict[str, int] = {}
    for v in verdicts:
        stages[v.stage] = stages.get(v.stage, 0) + 1

    # bind_rate is over rows whose schema actually creates; anything else is unanswerable.
    bindable = [v for v in verdicts if not v.schema_unusable]
    result = {
        "n": len(verdicts),
        "parse_rate": sum(v.parses for v in verdicts) / n,
        "bind_rate": (sum(v.binds for v in bindable) / len(bindable)) if bindable else None,
        "n_bindable": len(bindable),
        "stages": stages,
        "sample_errors": [v.error for v in verdicts
                          if v.error and not v.schema_unusable][:5],
    }
    if verbose:
        br = "n/a" if result["bind_rate"] is None else f"{result['bind_rate']:.3f}"
        print(f"n={result['n']}  parse_rate={result['parse_rate']:.3f}  "
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
                    progress: bool = True, chunk: int = 128) -> list[str]:
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
            if eos in g:
                g = g[: g.index(eos)]
            outputs.append(tokenizer.decode(g, skip_special_tokens=True))
    return outputs


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
    gens = greedy_generate(model, tokenizer, prompts, max_new_tokens=max_new_tokens,
                           seq_len=seq_len, device=device, batch_size=batch_size)
    result = evaluate([(g, r.schema) for g, r in zip(gens, rows)], verbose=True)
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

    # An uncreatable dataset schema must be quarantined, not charged to the model.
    broken_schema = 'CREATE TABLE a ("x" INT, FOREIGN KEY ("x") REFERENCES nosuch("y"));'
    v = check_sql("SELECT x FROM a", broken_schema)
    assert v.schema_unusable and v.stage == "schema_unusable", v
    print("uncreatable dataset schema is quarantined, not counted as a model failure")

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
    ap.parse_args()
    raise SystemExit(_self_test())
