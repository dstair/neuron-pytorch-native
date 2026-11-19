"""BIRD-for-DuckDB eval support: verified-gold questions, native materialization, and
result-set comparison -- the pieces the exec-accuracy lab needs on the box.

The comparator and materializer are copied from contrib/bird-duckdb/build.py (the dataset
builder) so this package stays self-contained under /work/src on the box. Keep them in
sync: any fix there is a fix here.

Dataset: bird_duckdb_dev.json (1491 records, verified DuckDB gold in SQL_duckdb).
Databases: the original BIRD .sqlite files, materialized to native DuckDB at eval time.
"""

from __future__ import annotations

import datetime
import json
import math
import pathlib
import re
import sqlite3

import duckdb

BIRD_ROOT = pathlib.Path("/work/bird")
DATASET_JSON = BIRD_ROOT / "bird_duckdb_dev.json"
DB_DIR = BIRD_ROOT / "dev_databases"


def load_questions(path: pathlib.Path = DATASET_JSON) -> list[dict]:
    return json.loads(pathlib.Path(path).read_text())


def stratified_subset(questions: list[dict], n: int, seed: int = 0) -> list[dict]:
    """A stable db-proportional subset: round-robin over per-db shuffles."""
    import random

    by_db: dict[str, list[dict]] = {}
    for q in questions:
        by_db.setdefault(q["db_id"], []).append(q)
    rng = random.Random(seed)
    for rows in by_db.values():
        rng.shuffle(rows)
    out, i = [], 0
    dbs = sorted(by_db)
    while len(out) < n and any(by_db.values()):
        db = dbs[i % len(dbs)]
        if by_db[db]:
            out.append(by_db[db].pop())
        i += 1
    return out


def schema_ddl(sqlite_path: str) -> str:
    """The CREATE TABLE statements the prompt shows as the schema."""
    lit = sqlite3.connect(sqlite_path)
    ddl = [
        r[0] for r in lit.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")
        if r[0]
    ]
    lit.close()
    return "\n".join(ddl)


def render_prompt(question: str, evidence: str, ddl: str) -> str:
    """The motherduckdb fine-tune format, plus BIRD's external-knowledge evidence line."""
    prompt = (
        "-- DuckDB SQL generation\n"
        "-- Schema:\n"
        f"{ddl.strip()}\n"
        f"-- Question: {question.strip()}\n"
    )
    if evidence and evidence.strip():
        prompt += f"-- Evidence: {evidence.strip()}\n"
    return prompt + "-- Answer:\n"


# ---------------------------------------------------------------------- comparison
# (copied from contrib/bird-duckdb/build.py)


def canon(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return ("num", float(v))
    if isinstance(v, (int, float)):
        return ("num", float(v))
    if isinstance(v, (bytes, bytearray, memoryview)):
        return ("bytes", bytes(v))
    if isinstance(v, datetime.datetime):
        return ("str", v.strftime("%Y-%m-%d %H:%M:%S"))
    if isinstance(v, datetime.date):
        return ("str", v.isoformat())
    s = str(v).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.0$", s):
        s = s[:-2]
    try:
        return ("num", float(s))
    except ValueError:
        return ("str", s)


def results_match(ref, cand, ordered: bool, rel_tol: float = 1e-6) -> bool:
    if len(ref) != len(cand):
        return False
    ref = [tuple(canon(v) for v in row) for row in ref]
    cand = [tuple(canon(v) for v in row) for row in cand]

    def cell_eq(a, b):
        if a is None or b is None:
            return a is None and b is None
        if a[0] == b[0] == "num":
            if a[1] != a[1] and b[1] != b[1]:
                return True
            return math.isclose(a[1], b[1], rel_tol=rel_tol, abs_tol=1e-9)
        return a == b

    def row_eq(a, b):
        return len(a) == len(b) and all(cell_eq(x, y) for x, y in zip(a, b))

    def multiset_eq() -> bool:
        ref_s = sorted(ref, key=repr)
        cand_s = sorted(cand, key=repr)
        return all(row_eq(a, b) for a, b in zip(ref_s, cand_s))

    if ordered:
        return all(row_eq(a, b) for a, b in zip(ref, cand)) or multiset_eq()
    return multiset_eq()


# ---------------------------------------------------------------------- materialization
# (copied from contrib/bird-duckdb/build.py: typed scan, all-varchar+TRY_CAST fallback,
# REAL columns always via the fallback to avoid f32 rounding)

_DECL_MAP = (("INT", "DOUBLE"), ("REAL", "DOUBLE"), ("FLOA", "DOUBLE"),
             ("DOUB", "DOUBLE"), ("NUM", "DOUBLE"), ("DEC", "DOUBLE"))


def table_list(sqlite_path: str):
    lit = sqlite3.connect(sqlite_path)
    out = [
        (r[0], [(c[1], c[2].upper()) for c in lit.execute(f'PRAGMA table_info("{r[0]}")')])
        for r in lit.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    ]
    lit.close()
    return out


def duckdb_native(sqlite_path: str) -> duckdb.DuckDBPyConnection:
    """Native DuckDB for one database.

    Prefers the prebuilt `<db>.duckdb` file (the dataset's own artifact, read-only) when
    present next to the .sqlite source. Otherwise materializes: typed scan first
    (content-sniffed types -- this is what lets date-ish columns come through as DATE so
    the verified gold's date functions run); REAL columns are CAST to DOUBLE at CTAS
    time; a scan failure falls back to all-varchar + TRY_CAST.
    Must stay in sync with build.py's duckdb_native: the dataset's SQL_duckdb was
    verified against THAT materialization, and it must run identically here.
    """
    prebuilt = pathlib.Path(sqlite_path).with_suffix(".duckdb")
    if prebuilt.exists():
        return duckdb.connect(str(prebuilt), read_only=True)
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    for t, cols in table_list(sqlite_path):
        try:
            exprs = ", ".join(
                f'CAST("{n}" AS DOUBLE) AS "{n}"' if d.startswith(("REAL", "FLOA", "DOUB"))
                else f'"{n}"' for n, d in cols)
            con.execute(f'CREATE TABLE "{t}" AS SELECT {exprs} FROM sqlite_scan(?, ?)',
                        [sqlite_path, t])
            continue
        except Exception:
            pass
        con.execute("SET sqlite_all_varchar=true")
        exprs = []
        for name, decl in cols:
            target = next((t2 for k, t2 in _DECL_MAP if decl.startswith(k)), None)
            exprs.append(f'TRY_CAST("{name}" AS {target}) AS "{name}"'
                         if target else f'"{name}"')
        con.execute(f'CREATE TABLE "{t}" AS SELECT {", ".join(exprs)} '
                    f'FROM sqlite_scan(?, ?)', [sqlite_path, t])
        con.execute("SET sqlite_all_varchar=false")
    return con


MODEL_SQL_TIMEOUT_S = 30.0
MODEL_SQL_ROW_CAP = 200_000


def _run_watchdog(con, sql: str, timeout: float | None = None, cap: int | None = None):
    """Execute with an interrupt watchdog and a row cap.

    Model SQL is untrusted: an accidental cross join over european_football_2 runs for
    hours (or exhausts host RAM) and would stall the whole eval after generation. A
    watchdog interrupt surfaces as an exception, graded as `timeout`.
    """
    import threading

    timeout = MODEL_SQL_TIMEOUT_S if timeout is None else timeout
    cap = MODEL_SQL_ROW_CAP if cap is None else cap
    t = threading.Timer(timeout, con.interrupt)
    t.start()
    try:
        rows = con.execute(sql).fetchmany(cap + 1)
    finally:
        t.cancel()
    return rows


def grade_one(con: duckdb.DuckDBPyConnection, model_sql: str, gold_sql: str) -> dict:
    """Execute model SQL and verified gold on one database; compare result sets.

    Returns stage: ok / exec_error (model SQL would not run) / timeout (model SQL ran past
    the watchdog) / wrong (ran, differed).
    """
    ordered = bool(re.search(r"\border\s+by\b", gold_sql, re.I))
    try:
        ref = con.execute(gold_sql).fetchall()
    except Exception as e:
        return {"stage": "gold_error", "error": str(e).splitlines()[0][:150]}
    try:
        cand = _run_watchdog(con, model_sql)
    except Exception as e:
        msg = str(e).splitlines()[0][:150]
        return {"stage": "timeout" if "nterrupt" in msg else "exec_error", "error": msg}
    if len(cand) > MODEL_SQL_ROW_CAP:
        return {"stage": "wrong", "error": f"model returned > {MODEL_SQL_ROW_CAP} rows"}
    if results_match(ref, cand, ordered):
        return {"stage": "ok"}
    return {"stage": "wrong",
            "error": f"result mismatch (gold {len(ref)} rows, model {len(cand)} rows)"}
