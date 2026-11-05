"""Prototype: BIRD-for-DuckDB steps 1-3 on one database (debit_card_specializing).

Step 1: load the SQLite database into NATIVE DuckDB tables (sqlite scanner for import,
then CREATE TABLE AS so execution semantics are DuckDB's own).

Step 2: run every gold query twice -- once against SQLite itself (the reference), once
against DuckDB as pass-through tier 0 (raw) and tier 1 (backticks -> double quotes).

Step 3: verify result sets match: ordered when the query has ORDER BY, multiset otherwise,
with float tolerance and type-canonicalized values. Report pass rates per tier and the
failure taxonomy -- this is the measured answer to "how big is the dialect gap".
"""

import json
import math
import re
import sqlite3
from collections import Counter

import duckdb

import sys
DB_ID = sys.argv[1] if len(sys.argv) > 1 else "debit_card_specializing"
DB = f"dev_databases/{DB_ID}/{DB_ID}.sqlite"
DEV = "dev_20240627/dev.json"


def canon(v):
    """One canonical Python value per cell, tolerant of SQLite/DuckDB type differences."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        f = float(v)
        return ("num", f)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return ("bytes", bytes(v))
    s = str(v).strip()
    # SQLite happily stores numbers as text; unify "1", "1.0", 1, 1.0.
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
            return math.isclose(a[1], b[1], rel_tol=rel_tol, abs_tol=1e-9)
        return a == b

    def row_eq(a, b):
        return len(a) == len(b) and all(cell_eq(x, y) for x, y in zip(a, b))

    if ordered:
        return all(row_eq(a, b) for a, b in zip(ref, cand))
    # Multiset: sort both by repr for a stable pairing, then compare PAIRWISE with the
    # tolerant row_eq -- sorting reprs and string-comparing would flag float-accumulation
    # noise (459.9562642871061 vs ...0896) as a mismatch.
    ref_s, cand_s = sorted(map(repr, ref)), sorted(map(repr, cand))
    return all(row_eq(a, b) for a, b in zip(
        [eval(r) for r in ref_s], [eval(c) for c in cand_s]))


def duckdb_native(sqlite_path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    tables = [
        r[0]
        for r in sqlite3.connect(sqlite_path)
        .execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        .fetchall()
    ]
    for t in tables:
        con.execute(
            f'CREATE TABLE "{t}" AS SELECT * FROM sqlite_scan(?, ?)', [sqlite_path, t]
        )
    return con


def tier1(sql: str) -> str:
    # SQLite's backtick identifier quoting -> DuckDB double quotes.
    return sql.replace("`", '"')


def make_tier2(text_cols: set[str]):
    """Rule-based translation, schema-aware where it must be.

    - IIF(a, b, c) -> IF(a, b, c): DuckDB has IF, not IIF.
    - `col BETWEEN <int> AND <int>` on a TEXT column -> quote the literals. SQLite
      coerces silently; DuckDB (>=0.10) refuses mixed-type BETWEEN. Quoting is only
      correct for text columns (string ordering!), so this needs the schema.
    """
    between = re.compile(r"(\b\w+(?:\.\w+)?)\s+BETWEEN\s+(\d+)\s+AND\s+(\d+)", re.I)

    def tier2(sql: str) -> str:
        sql = re.sub(r"\bIIF\(", "IF(", sql, flags=re.I)

        def quote(m):
            col = m.group(1).split(".")[-1]
            return m.group(0) if col not in text_cols else (
                f"{m.group(1)} BETWEEN '{m.group(2)}' AND '{m.group(3)}'")

        return between.sub(quote, sql)

    return tier2


def text_columns(sqlite_path: str) -> set[str]:
    lit = sqlite3.connect(sqlite_path)
    cols = set()
    for (t,) in lit.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        for r in lit.execute(f'PRAGMA table_info("{t}")'):
            if "CHAR" in r[2].upper() or "TEXT" in r[2].upper():
                cols.add(r[1])
    return cols


def typed_columns(sqlite_path: str, kinds: tuple[str, ...]) -> set[str]:
    lit = sqlite3.connect(sqlite_path)
    cols = set()
    for (t,) in lit.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        for r in lit.execute(f'PRAGMA table_info("{t}")'):
            if any(k in r[2].upper() for k in kinds):
                cols.add(r[1])
    return cols


def make_tier3(date_cols: set[str], hand: dict[int, str]):
    """The remaining three semantic rules, plus the per-query hand-fix list.

    - `x / y` -> `x / NULLIF(y, 0)`: SQLite division by zero yields NULL (and SUM/WHERE
      drop those rows); DuckDB yields inf, which poisons aggregates and admits extra rows.
      NULLIF reproduces SQLite semantics exactly.
    - `date_col LIKE 'p%'` -> `CAST(date_col AS VARCHAR) LIKE 'p%'`: the column is a real
      DATE in DuckDB (SQLite stores it as text), and ~~(DATE, STRING) does not bind.
    - Bare columns in a GROUP BY query (SQLite picks an arbitrary row) need ANY_VALUE;
      that one needs real parsing, so it lives in the reviewed hand-fix list.
    """
    div = re.compile(r"(\w+\.\w+)\s*/\s*(\w+\.\w+)(?!\s*[,)]?\s*NULLIF)")
    like = re.compile(r"(\w+(?:\.\w+)?)\s+LIKE\s+('[^']*')", re.I)

    def tier3(sql: str, qid: int) -> str:
        if qid in hand:
            return hand[qid]
        sql = div.sub(r"\1 / NULLIF(\2, 0)", sql)

        def cast_like(m):
            col = m.group(1).split(".")[-1]
            return m.group(0) if col not in date_cols else (
                f"CAST({m.group(1)} AS VARCHAR) LIKE {m.group(2)}")

        return like.sub(cast_like, sql)

    return tier3


def main() -> None:
    dev = json.load(open(DEV))
    rows = [d for d in dev if d["db_id"] == DB_ID]
    print(f"{len(rows)} questions on debit_card_specializing")

    lit = sqlite3.connect(DB)
    tier2 = make_tier2(text_columns(DB))
    # The one bare-column GROUP BY in this db: SQLite serves an arbitrary row's Segment;
    # ANY_VALUE is the faithful translation. Datasets keep this list reviewed and short.
    tier3 = make_tier3(
        typed_columns(DB, ("DATE",)),
        hand={1496: "SELECT ANY_VALUE(T1.Segment) FROM customers AS T1 INNER JOIN "
                    "yearmonth AS T2 ON T1.CustomerID = T2.CustomerID "
                    "WHERE T2.Date = '201309' GROUP BY T1.CustomerID "
                    "ORDER BY SUM(T2.Consumption) DESC LIMIT 1"},
    )
    con = duckdb_native(DB)
    n_tables = con.execute("SHOW TABLES").fetchall()
    print(f"loaded {len(n_tables)} tables into native DuckDB")

    stats = Counter()
    failures: list[dict] = []
    for d in rows:
        qid, sql = d["question_id"], d["SQL"]
        ordered = bool(re.search(r"\border\s+by\b", sql, re.I))

        try:
            ref = lit.execute(sql).fetchall()
        except Exception as e:
            stats["sqlite_gold_error"] += 1
            failures.append({"qid": qid, "tier": "sqlite", "error": str(e)[:120], "sql": sql[:160]})
            continue

        outcome = None
        tiers = ((0, lambda s: s), (1, tier1), (2, tier2), (3, lambda s: tier3(s, qid)))
        for tier, fn in tiers:
            try:
                cand = con.execute(fn(sql)).fetchall()
            except Exception as e:
                outcome = outcome or {
                    "qid": qid, "tier": tier, "kind": "duck_error",
                    "error": str(e).splitlines()[0][:120], "sql": sql[:160],
                }
                continue
            if results_match(ref, cand, ordered):
                stats[f"pass_tier{tier}"] += 1
                outcome = None
                break
            if outcome is None:
                outcome = {
                    "qid": qid, "tier": tier, "kind": "mismatch",
                    "error": f"result mismatch (ref {len(ref)} rows, duck {len(cand)} rows)",
                    "sql": sql[:160],
                    "ref": ref[:3], "cand": cand[:3],
                }
        if outcome:
            stats["fail_" + outcome["kind"]] += 1
            failures.append(outcome)

    total = len(rows)
    passed = sum(stats[f"pass_tier{t}"] for t in range(4))
    print(f"\npass tier0 (raw pass-through): {stats['pass_tier0']}/{total} "
          f"({100 * stats['pass_tier0'] / total:.1f}%)")
    print(f"pass tier1 (backtick->quote):  {stats['pass_tier1']}/{total} "
          f"({100 * stats['pass_tier1'] / total:.1f}%)")
    print(f"pass tier2 (+IIF->IF, BETWEEN): {stats['pass_tier2']}/{total} "
          f"({100 * stats['pass_tier2'] / total:.1f}%)")
    print(f"pass tier3 (+NULLIF, DATE-LIKE, hand): {stats['pass_tier3']}/{total} "
          f"({100 * stats['pass_tier3'] / total:.1f}%)")
    print(f"TOTAL verified:                {passed}/{total} ({100 * passed / total:.1f}%)")
    print(f"sqlite gold itself errored:    {stats['sqlite_gold_error']}")
    print(f"duck errors:                   {stats['fail_duck_error']}")
    print(f"result mismatches:             {stats['fail_mismatch']}")

    kinds = Counter()
    for f in failures:
        if f["tier"] == "sqlite":
            kinds["sqlite gold error"] += 1
        elif f["kind"] == "duck_error":
            msg = f["error"]
            m = re.search(r"(?:function|Function).*?(\w+)\s*(?:does not exist|not found)", msg)
            kinds[f"duck error: {m.group(0)[:60] if m else msg.split('(')[0][:60]}"] += 1
        else:
            kinds["result mismatch"] += 1
    print("\nfailure taxonomy:")
    for k, c in kinds.most_common():
        print(f"  {c:3}  {k}")
    print("\nsample failures:")
    for f in failures[:8]:
        print(f"  qid {f['qid']} [{f['kind']}@{f['tier']}] {f['error']}")
        print(f"      {f['sql']}")
        if "ref" in f:
            print(f"      ref : {f['ref']}")
            print(f"      duck: {f['cand']}")


if __name__ == "__main__":
    main()
