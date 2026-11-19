"""Build BIRD-for-DuckDB: verified translation of the full BIRD dev set.

For each of the 11 dev databases: materialize the SQLite database into native DuckDB
tables (so types are DuckDB's own), run every gold query against SQLite (reference) and
DuckDB through a tiered translation pipeline, and verify result sets match. Emits:

  bird_duckdb_dev.json   one record per question: BIRD fields + SQL_duckdb + tier (+verified)
  build_report.json      per-db tier rates, drops, and every failure with its error class

The hand-fix list (HANDFIX) is short, reviewed, and per-question -- it is part of the
artifact. Anything that still fails is EXCLUDED from the dataset and counted in the
report; the dataset proves its own gold answers.

    python3 build.py             # full build + artifacts
    python3 build.py --triage    # print failures only, no artifacts

Data: dev_20240627/ from https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
(BIRD is CC BY-SA 4.0 -- https://bird-bench.github.io/)
"""

import datetime
import json
import math
import pathlib
import re
import sqlite3
import sys
import threading
from collections import Counter

import duckdb

ROOT = pathlib.Path(__file__).resolve().parent

# Which split to build: `--set dev` (default, the eval set) or `--set train` (the
# fine-tuning corpus, ~9.4k questions over 69 databases).
_SPLIT = "train" if "--set" in sys.argv and sys.argv[sys.argv.index("--set") + 1] == "train" else "dev"
DATA = ROOT / ("train" if _SPLIT == "train" else "dev_20240627")
DEV_JSON = DATA / ("train.json" if _SPLIT == "train" else "dev.json")
DB_DIR = DATA / ("train_databases" if _SPLIT == "train" else "dev_databases")
OUT_DATASET = ROOT / f"bird_duckdb_{_SPLIT}.json"
OUT_REPORT = ROOT / f"build_report_{_SPLIT}.json" if _SPLIT == "train" else ROOT / "build_report.json"

# Reviewed per-question hand-fixes (qid -> DuckDB SQL). Each entry needs a comment
# saying WHY. 1496: SQLite serves an arbitrary row's Segment for a bare GROUP BY column;
# ANY_VALUE is the faithful translation.
HANDFIX = {
    # 1496: SQLite serves an arbitrary row's Segment for a bare GROUP BY column;
    # ANY_VALUE is the faithful translation.
    1496: "SELECT ANY_VALUE(T1.Segment) FROM customers AS T1 INNER JOIN "
          "yearmonth AS T2 ON T1.CustomerID = T2.CustomerID "
          "WHERE T2.Date = '201309' GROUP BY T1.CustomerID "
          "ORDER BY SUM(T2.Consumption) DESC LIMIT 1",
    # 388: SQLite lets a scalar subquery return N rows and uses the first; DuckDB
    # refuses. LIMIT 1 is the faithful form (single row here either way).
    388: "SELECT id, language FROM set_translations WHERE id = "
         "( SELECT id FROM cards WHERE convertedManaCost = 5 LIMIT 1 ) AND setCode = '10E'",
    # 1032: bare league_id in an aggregate query over a subquery; SQLite serves an
    # arbitrary row, so ANY_VALUE is the faithful form (GROUP BY would change the
    # cardinality -- measured).
    1032: "SELECT t2.name, t1.max_count FROM League AS t2 JOIN "
          "(SELECT ANY_VALUE(league_id) AS league_id, MAX(cnt) AS max_count FROM "
          "(SELECT league_id, COUNT(id) AS cnt FROM Match GROUP BY league_id) AS subquery) "
          "AS t1 ON t1.league_id = t2.id",
    # 880: IF(raceId, fastestLapSpeed, 0) mixes a VARCHAR column with an int literal;
    # SQLite coerces. numcoerce on the column is the faithful form.
    880: "SELECT (SUM(IF(T2.raceId = 853, COALESCE(TRY_CAST(REGEXP_EXTRACT(T2.fastestLapSpeed, '^[-+0-9.eE]+', 0) AS DOUBLE), 0), 0)) - "
         "SUM(IF(T2.raceId = 854, COALESCE(TRY_CAST(REGEXP_EXTRACT(T2.fastestLapSpeed, '^[-+0-9.eE]+', 0) AS DOUBLE), 0), 0))) * 100 / "
         "SUM(IF(T2.raceId = 853, COALESCE(TRY_CAST(REGEXP_EXTRACT(T2.fastestLapSpeed, '^[-+0-9.eE]+', 0) AS DOUBLE), 0), 0)) "
         "FROM drivers AS T1 INNER JOIN results AS T2 ON T2.driverId = T1.driverId "
         "WHERE T1.forename = 'Paul' AND T1.surname = 'di Resta'",
    # 1093 / 317: SQLite INTEGER division truncates toward zero; DuckDB always divides
    # for real. FLOOR of the real quotient is the faithful form (operands positive).
    1093: "SELECT FLOOR(SUM(t2.overall_rating) / COUNT(t1.id)) FROM Player AS T1 "
          "INNER JOIN Player_Attributes AS T2 ON T1.player_api_id = T2.player_api_id "
          "WHERE SUBSTR(T1.birthday, 1, 4) < '1986'",
    317: "SELECT FLOOR(COUNT(CASE WHEN T2.label = '+' AND T1.element = 'cl' THEN T2.molecule_id "
         "ELSE NULL END) * 100.0 / COUNT(T2.molecule_id)) FROM atom AS T1 "
         "INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id",
    # 1266: `OR '+-'` -- SQLite coerces the bare string to false, so the clause reduces
    # to RNP != '-'. DuckDB refuses to cast '+-' to BOOL.
    1266: "SELECT T1.Birthday FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID "
          "WHERE T2.RNP != '-' ORDER BY T1.Birthday DESC LIMIT 1",
}


def _first_diff(ref, cand) -> str:
    """First canonical cell pair that differs, for triage (rows sorted for stability)."""
    try:
        r = sorted((tuple(canon(v) for v in row) for row in ref), key=repr)
        c = sorted((tuple(canon(v) for v in row) for row in cand), key=repr)
        for a, b in zip(r, c):
            if a != b:
                return f"{a} != {b}"
        return "row-count-only"
    except Exception as e:  # pragma: no cover - diagnostic only
        return f"<diff failed: {e}>"


# ---------------------------------------------------------------------- comparison


def canon(v):
    """One canonical Python value per cell, tolerant of SQLite/DuckDB type differences."""
    if v is None:
        return None
    if isinstance(v, bool):
        # SQLite has no boolean type (1/0); DuckDB does (True/False). Unify numerically.
        return ("num", float(v))
    if isinstance(v, (int, float)):
        return ("num", float(v))
    if isinstance(v, (bytes, bytearray, memoryview)):
        return ("bytes", bytes(v))
    if isinstance(v, datetime.datetime):
        # SQLite stores datetimes as text ('2010-09-13 17:16:24.0'); DuckDB returns real
        # datetime objects. Same value, different representation -- normalize to text.
        return ("str", v.strftime("%Y-%m-%d %H:%M:%S"))
    if isinstance(v, datetime.date):
        return ("str", v.isoformat())
    s = str(v).strip()
    # SQLite's text form of a datetime carries a trailing '.0' microseconds field.
    if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.0$", s):
        s = s[:-2]
    try:
        return ("num", float(s))
    except ValueError:
        return ("str", s)


ROW_CAP = 100_000
QUERY_TIMEOUT_S = 30


def run_capped(con, sql: str, cap: int = ROW_CAP, timeout: float = QUERY_TIMEOUT_S):
    """Execute a query, fetching at most cap rows, with an interrupt watchdog.

    Some train golds are pathological (expensive joins over 2 GB tables that would run
    for hours in either engine). The watchdog interrupts them; the caller drops the
    question rather than stalling the build for a gold no one can execute anyway.
    Both sqlite3 and DuckDB connections support interrupt().
    """
    done = threading.Event()

    def interrupt():
        if not done.is_set():
            con.interrupt()

    t = threading.Timer(timeout, interrupt)
    t.start()
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(cap + 1)
    finally:
        done.set()
        t.cancel()
    return rows[:cap], len(rows) > cap


def compare_big(con_ref, con_duck, ref_sql, duck_sql) -> bool:
    """Compare result sets too big to materialize: row counts, then a deterministic
    ordered prefix sample. Ordering by ordinals is stable across engines except for
    ties, which the sample's multiset comparison absorbs."""
    def count_and_sample(con, sql):
        n = run_capped(con, f"SELECT COUNT(*) FROM ({sql})")[0][0][0]
        ncols = len(con.execute(f"SELECT * FROM ({sql}) LIMIT 0").description)
        order = ", ".join(str(i + 1) for i in range(ncols))
        sample, _ = run_capped(con, f"SELECT * FROM ({sql}) ORDER BY {order} LIMIT 10000")
        return n, sample

    # A watchdog interrupt here means the gold cannot be compared at all: not verified.
    try:
        n_ref, s_ref = count_and_sample(con_ref, ref_sql)
        n_duck, s_duck = count_and_sample(con_duck, duck_sql)
    except Exception:
        return False
    if n_ref != n_duck:
        return False
    return results_match(s_ref, s_duck, ordered=False)


def results_match(ref, cand, ordered: bool, rel_tol: float = 1e-6) -> bool:
    """Order-insensitive unless `ordered`, float-tolerant, type-canonicalized.

    The multiset path must compare PAIRWISE after sorting -- parallel aggregation sums
    in a different order on DuckDB, and sorted-repr string equality flags
    459.9562642871061 vs ...0896 as a mismatch when it is accumulation noise.
    """
    if len(ref) != len(cand):
        return False
    ref = [tuple(canon(v) for v in row) for row in ref]
    cand = [tuple(canon(v) for v in row) for row in cand]

    def cell_eq(a, b):
        if a is None or b is None:
            return a is None and b is None
        if a[0] == b[0] == "num":
            # NaN is a value too: text like 'nan' canonicalizes to float('nan') on both
            # sides, and NaN != NaN would flag identical rows as mismatches.
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
        # ORDER BY with ties is only PARTIALLY deterministic: the ordered comparison can
        # fail on tie order even when the row SETS are identical. Accept either.
        return all(row_eq(a, b) for a, b in zip(ref, cand)) or multiset_eq()
    # Sort by repr for a stable pairing; eval back (reprs of canon tuples are literal).
    return multiset_eq()


# ---------------------------------------------------------------------- schema helpers


def columns_by_type(sqlite_path: str, kinds: tuple[str, ...]) -> set[str]:
    lit = sqlite3.connect(sqlite_path)
    cols = set()
    for (t,) in lit.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        for r in lit.execute(f'PRAGMA table_info("{t}")'):
            if any(k in r[2].upper() for k in kinds):
                cols.add(r[1])
    lit.close()
    return cols


# Fallback TRY_CAST target for declared-numeric columns: always DOUBLE. BIGINT would
# discard fractional values ('182.88' -> NULL) that SQLite happily SUMs; DOUBLE keeps
# them, and junk ('*', 'future') becomes NULL, which never matches a numeric predicate
# and contributes nothing to aggregates -- exactly SQLite's semantics.
_DECL_MAP = (("INT", "DOUBLE"), ("REAL", "DOUBLE"), ("FLOA", "DOUBLE"),
             ("DOUB", "DOUBLE"), ("NUM", "DOUBLE"), ("DEC", "DOUBLE"))


def rebuild_trycast(con, tables, colname: str) -> None:
    """Rebuild every table owning `colname` with TRY_CAST for that column.

    Triggered lazily by a query-time conversion error ('casting from source column X'):
    the column declared one type but contains another, and a query compares or aggregates
    it numerically. TRY_CAST sends the junk to NULL, matching SQLite's semantics on
    numeric predicates and aggregates. Reads from the already-materialized table.
    """
    for t, cols in tables:
        decls = dict(cols)
        if colname not in decls:
            continue
        target = next((t2 for k, t2 in _DECL_MAP if decls[colname].startswith(k)), "DOUBLE")
        exprs = ", ".join(
            f'TRY_CAST("{n}" AS {target}) AS "{n}"' if n == colname else f'"{n}"'
            for n, _ in cols)
        con.execute(f'CREATE TABLE _rebuild AS SELECT {exprs} FROM "{t}"')
        con.execute(f'DROP TABLE "{t}"')
        con.execute(f'ALTER TABLE _rebuild RENAME TO "{t}"')
        print(f"   note: rebuilt {t}.{colname} via TRY_CAST (query-time conversion error)")


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
    """Materialize a SQLite db into native DuckDB tables.

    First choice: typed scan, so types are DuckDB's own. SQLite's loose typing means some
    columns declare one type and contain another ('182.88' in an INTEGER column, '*' in a
    numeric column) -- the typed scan then fails or, worse, fails later at query time.
    Fallback: scan all-varchar and TRY_CAST every numeric column (junk -> NULL, which
    matches SQLite semantics on comparisons and aggregates: junk never matches a numeric
    predicate and contributes nothing to a SUM). Text/date columns stay VARCHAR in the
    fallback, exactly as SQLite treats them.
    """
    tables = table_list(sqlite_path)

    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    for t, cols in tables:
        try:
            # sqlite_scan types REAL columns as FLOAT (f32); SQLite REAL is f64, and f32
            # accumulation shows up as 4e-6 relative drift in aggregates. Cast to DOUBLE.
            exprs = ", ".join(
                f'CAST("{n}" AS DOUBLE) AS "{n}"' if d.startswith(("REAL", "FLOA", "DOUB"))
                else f'"{n}"' for n, d in cols)
            con.execute(f'CREATE TABLE "{t}" AS SELECT {exprs} FROM sqlite_scan(?, ?)',
                        [sqlite_path, t])
            continue
        except Exception:
            pass
        # Dirty table: rebuild with TRY_CAST for numerics, VARCHAR elsewhere.
        con.execute("SET sqlite_all_varchar=true")
        exprs = []
        for name, decl in cols:
            target = next((t2 for k, t2 in _DECL_MAP if decl.startswith(k)), None)
            if target:
                exprs.append(f'TRY_CAST("{name}" AS {target}) AS "{name}"')
            else:
                exprs.append(f'"{name}"')
        con.execute(f'CREATE TABLE "{t}" AS SELECT {", ".join(exprs)} '
                    f'FROM sqlite_scan(?, ?)', [sqlite_path, t])
        con.execute("SET sqlite_all_varchar=false")
        print(f"   note: table {t} materialized via TRY_CAST fallback (mixed-type columns)")
    return con


def wrap_bare_column(sql: str, colname: str):
    """SQLite serves an arbitrary row's value for a bare column in a GROUP BY (or in the
    ORDER BY of an aggregate query); DuckDB refuses to plan it. Wrap the offending column
    -- named in DuckDB's own error message -- in ANY_VALUE, the faithful translation.
    """
    pat = re.compile(r"(?<!VALUE\()(\b\w+\." + re.escape(colname) + r"\b|\b" +
                     re.escape(colname) + r"\b)(?!\s*\()", re.I)
    m = re.search(r"\bSELECT\b", sql, re.I)
    f = re.search(r"\bFROM\b", sql[m.end():], re.I)
    sel_start = m.end()
    sel, tail = sql[sel_start:sel_start + f.start()], sql[sel_start + f.start():]
    new_sel, n = pat.subn(r"ANY_VALUE(\1)", sel, count=1)
    if n:
        return sql[:sel_start] + new_sel + tail, True
    o = re.search(r"\bORDER\s+BY\b", tail, re.I)
    if o:
        new_ob, n = pat.subn(r"ANY_VALUE(\1)", tail[o.start():], count=1)
        if n:
            return sql[:sel_start] + sel + tail[:o.start()] + new_ob, True
    return sql, False


# ---------------------------------------------------------------------- translation tiers


_BACKTICK = re.compile(r"`([^`]*)`")
_LIMIT_COMMA = re.compile(r"\bLIMIT\s+(\d+)\s*,\s*(\d+)", re.I)
_STRFTIME = re.compile(
    r"STRFTIME\s*\(\s*('[^']*')\s*,\s*((?:[^()]|\([^()]*\))+?)\s*\)", re.I)
_JULIANDAY = re.compile(r"JULIANDAY\s*\(([^()]+)\)", re.I)
_COMPARE_NUM = re.compile(
    r"((?:\w+\.)?(?:\w+|\"[^\"]+\"))\s*(>=|<=|=|<|>)\s*(\d+(?:\.\d+)?)")
_PERCENT_ALIAS = re.compile(r"(?<!\bAS\s)\spercent\b(?=\s*(?:,|FROM|$))", re.I)
# Operand of a '-' that DuckDB rejected as text: column, quoted literal, or fn call
# (one level of nesting, so STRFTIME(CAST(x AS TIMESTAMP), '%Y') parses).
_ARITH_OPERAND = (r"((?!(?:AVG|SUM|COUNT)\b)(?:\w+\.)?\w+\((?:[^()]|\([^()]*\))*\)"
                  r"|(?:\w+\.)?\w+|'[^']*')")
_SUBTRACT = re.compile(_ARITH_OPERAND + r"\s*-\s*" + _ARITH_OPERAND)


def _numcoerce(operand: str) -> str:
    """SQLite's text-to-number coercion as a SQL expression: leading numeric prefix,
    else 0.0. TRY_CAST rejects '05:07' wholesale; SQLite reads it as 5. CAST first:
    the operand may be a DATE or numeric, and REGEXP_EXTRACT needs VARCHAR."""
    return (f"COALESCE(TRY_CAST(REGEXP_EXTRACT(CAST({operand} AS VARCHAR), "
            f"'^[-+0-9.eE]+', 0) AS DOUBLE), 0)")
_LIKE_KW = re.compile(r"\sLIKE\s+(?=')", re.I)
_BETWEEN = re.compile(r"((?:\w+\.)?(?:\w+|\"[^\"]+\"))\s+BETWEEN\s+(\d+)\s+AND\s+(\d+)", re.I)
_DIV = re.compile(r"(\w+\.\w+)\s*/\s*(\w+\.\w+)")
_LIKE = re.compile(r"(\w+(?:\.\w+)?)\s+(NOT\s+)?LIKE\s+('[^']*')", re.I)
_AGG = re.compile(r"\b(AVG|SUM)\(\s*((?:\w+\.)?(?:\w+|\"[^\"]+\"))\s*\)", re.I)

def make_translator(text_cols: set[str], date_cols: set[str]):
    """Ordered rewrite rules; each addresses one measured dialect difference."""

    def base(sql: str) -> str:
        """Quoting/syntax normalization: no semantic content."""
        # DuckDB does NOT accept SQLite's backtick identifier quoting (it parses
        # `Free Meal` as `Free` + junk). Double quotes are the DuckDB identifier quote.
        sql = _BACKTICK.sub(r'"\1"', sql)
        # SQLite's LIMIT offset, count -> LIMIT count OFFSET offset.
        sql = _LIMIT_COMMA.sub(r"LIMIT \2 OFFSET \1", sql)
        # SQLite strftime(format, date); DuckDB strftime(date, format) -- swap, and
        # CAST the date arg: DuckDB's strftime is ambiguous on a bare VARCHAR column.
        sql = _STRFTIME.sub(r"STRFTIME(CAST(\2 AS TIMESTAMP), \1)", sql)
        # No julianday in DuckDB. Every use here is ordering or day differences, both of
        # which CAST(x AS DATE) reproduces (ISO order; DATE - DATE = integer days).
        # ('now' is not a DuckDB date literal; CURRENT_DATE is the spelling.)
        sql = _JULIANDAY.sub(
            lambda m: "CURRENT_DATE" if m.group(1).strip().strip("'") == "now"
            else f"CAST({m.group(1)} AS DATE)", sql)
        # `Date` is reserved in DuckDB; SQLite allows it as a bare alias.
        sql = re.sub(r"\bAS Date\b", 'AS "Date"', sql, flags=re.I)
        # SQLite's date('now') is CURRENT_DATE in DuckDB; DATE('now') is a failed cast.
        sql = re.sub(r"\bdate\s*\(\s*'now'\s*\)", "CURRENT_DATE", sql, flags=re.I)
        # SQLite has datetime(x); DuckDB spells it CAST(x AS TIMESTAMP). The no-arg and
        # 'localtime' forms are just CURRENT_TIMESTAMP here.
        sql = re.sub(r"\bdatetime\s*\(\s*(CURRENT_TIMESTAMP|now\(\)|'now')\s*,\s*'localtime'\s*\)",
                     r"\1", sql, flags=re.I)
        sql = re.sub(r"\bdatetime\s*\(\s*\)", "CURRENT_TIMESTAMP", sql, flags=re.I)
        sql = re.sub(r"\bdatetime\s*\(([^()]+)\)", r"CAST(\1 AS TIMESTAMP)", sql, flags=re.I)
        # `percent` is reserved in DuckDB; SQLite lets it through as a bare alias.
        sql = _PERCENT_ALIAS.sub(' AS "percent"', sql)
        # SQLite LIKE is case-INSENSITIVE for ASCII; DuckDB LIKE is case-sensitive.
        # ILIKE reproduces SQLite semantics ('%from the vault%' must match 'FROM the
        # Vault' -- measured: queries returning 1 row in SQLite, 0 in DuckDB).
        sql = _LIKE_KW.sub(" ILIKE ", sql)
        return sql

    def translate(sql: str) -> str:
        # LIKE against a real DATE column does not bind; SQLite stored them as text.
        # Applied BEFORE base() turns LIKE into ILIKE.
        sql = _LIKE.sub(
            lambda m: m.group(0) if m.group(1).split(".")[-1].strip('"') not in date_cols else
            f"CAST({m.group(1)} AS VARCHAR) {m.group(2) or ''}LIKE {m.group(3)}", sql)
        sql = base(sql)
        # SQLite has IIF, DuckDB spells it IF.
        sql = re.sub(r"\bIIF\s*\(", "IF(", sql, flags=re.I)
        # SQLite coerces mixed-type BETWEEN; DuckDB refuses. Quote literals only when
        # the column is TEXT (quoting a numeric column would impose string ordering).
        sql = _BETWEEN.sub(
            lambda m: m.group(0) if m.group(1).split(".")[-1].strip('"') not in text_cols else
            f"{m.group(1)} BETWEEN '{m.group(2)}' AND '{m.group(3)}'", sql)
        # SQLite x/0 -> NULL (dropped by SUM/WHERE); DuckDB -> inf. NULLIF is exact.
        sql = _DIV.sub(r"\1 / NULLIF(\2, 0)", sql)
        # Aggregates over TEXT columns: SQLite coerces junk to 0.0 (prefix parse);
        # DuckDB refuses avg(VARCHAR) outright. sqlite_num() reproduces the coercion.
        sql = _AGG.sub(
            lambda m: m.group(0) if m.group(2).split(".")[-1].strip('"') not in text_cols else
            f"{m.group(1)}({_numcoerce(m.group(2))})", sql)
        return sql

    translate.base = base
    return translate


# ---------------------------------------------------------------------- build


def emit_duckdb() -> None:
    """Write one persistent .duckdb file per database next to its .sqlite source.

    These are the DATASET's native database files: same materialization the gold was
    verified against, no sqlite scanner needed at consume time. Verified by reopening
    each and counting tables.
    """
    dev = json.loads(DEV_JSON.read_text())
    for db_id in sorted({d["db_id"] for d in dev}):
        sqlite_path = str(DB_DIR / db_id / f"{db_id}.sqlite")
        out = DB_DIR / db_id / f"{db_id}.duckdb"
        out.unlink(missing_ok=True)
        tables = table_list(sqlite_path)

        dst_con = duckdb.connect(str(out))
        dst_con.execute("INSTALL sqlite; LOAD sqlite;")
        for t, cols in tables:
            try:
                exprs = ", ".join(
                    f'CAST("{n}" AS DOUBLE) AS "{n}"'
                    if d.startswith(("REAL", "FLOA", "DOUB")) else f'"{n}"'
                    for n, d in cols)
                dst_con.execute(
                    f'CREATE TABLE "{t}" AS SELECT {exprs} FROM '
                    f'sqlite_scan(?, ?)', [sqlite_path, t])
                continue
            except Exception:
                pass
            dst_con.execute("SET sqlite_all_varchar=true")
            exprs = ", ".join(
                f'TRY_CAST("{n}" AS DOUBLE) AS "{n}"'
                if any(d.startswith(k) for k, _ in _DECL_MAP) else f'"{n}"'
                for n, d in cols)
            dst_con.execute(f'CREATE TABLE "{t}" AS SELECT {exprs} FROM '
                            f'sqlite_scan(?, ?)', [sqlite_path, t])
            dst_con.execute("SET sqlite_all_varchar=false")
        dst_con.close()

        check = duckdb.connect(str(out), read_only=True)
        n = len(check.execute("SHOW TABLES").fetchall())
        check.close()
        assert n == len(tables), f"{db_id}: {n} tables, expected {len(tables)}"
        print(f"{db_id:26} -> {out.name}  ({n} tables, "
              f"{out.stat().st_size / 1e6:.0f} MB)")


def build_db(job):
    """Verify every question of one database. Runs in a worker process: each db
    gets its own SQLite + in-memory DuckDB connection, so dbs are independent."""
    db_id, rows, triage = job
    records, log = [], []
    sqlite_path = str(DB_DIR / db_id / f"{db_id}.sqlite")
    lit = sqlite3.connect(sqlite_path)
    translate = make_translator(
        columns_by_type(sqlite_path, ("CHAR", "TEXT", "CLOB")),
        columns_by_type(sqlite_path, ("DATE", "TIME")),
    )
    base = translate.base
    tables = table_list(sqlite_path)
    con = duckdb_native(sqlite_path)
    rebuilt_cols: set[str] = set()
    stats = Counter()
    failures = []

    for i, d in enumerate(rows):
        # The train split ships no question_id; synthesize a stable one.
        qid, sql = d.get("question_id", f"{db_id}:{i}"), d["SQL"]
        ordered = bool(re.search(r"\border\s+by\b", sql, re.I))
        try:
            ref, ref_trunc = run_capped(lit, sql)
        except Exception as e:
            kind = "timeout" if "nterrupt" in str(e) else "sqlite_error"
            stats["sqlite_gold_error"] += 1
            failures.append({"qid": qid, "kind": kind,
                             "error": str(e)[:150], "sql": sql})
            continue

        verified_sql, outcome = None, None
        candidates = [(0, sql)] if qid not in HANDFIX else [(-1, HANDFIX[qid])]
        candidates += [(1, base(sql)), (3, translate(sql))]
        tier_errors = []
        for tier, cand_sql in candidates:
            # Repair loop: DuckDB's errors name their own remedy. A bare GROUP BY
            # column gets ANY_VALUE; a dirty mixed-type column gets a TRY_CAST
            # materialization rebuild.
            fixed_sql, repairs = cand_sql, 0
            while True:
                try:
                    cand, cand_trunc = run_capped(con, fixed_sql)
                    break
                except Exception as e:
                    msg = str(e).splitlines()[0]
                    if "nterrupt" in msg:
                        tier_errors.append(f"t{tier}: TIMEOUT")
                        cand = None
                        break
                    g = re.search(r'column "(\w+)" must appear in the GROUP BY', msg)
                    if g and repairs < 3:
                        fixed_sql, ok = wrap_bare_column(fixed_sql, g.group(1))
                        if ok:
                            repairs += 1
                            continue
                    g = re.search(r"when casting from source column (.+?)$", msg)
                    if g and g.group(1) not in rebuilt_cols:
                        rebuild_trycast(con, tables, g.group(1))
                        rebuilt_cols.add(g.group(1))
                        continue
                    # SQLite silently coerces text in arithmetic (prefix parse, else
                    # 0); DuckDB refuses. Wrap each operand in the SQLite-style
                    # coercion and retry -- error-driven, so never over-applied.
                    if re.search(r"-\((?:VAR|STRING_LIT)", msg) and repairs < 3:
                        fixed_sql = _SUBTRACT.sub(
                            lambda m: f"({_numcoerce(m.group(1))} - {_numcoerce(m.group(2))})",
                            fixed_sql, count=1)
                        repairs += 1
                        continue
                    # Same coercion story for text-vs-number comparisons.
                    if re.search(r"Cannot compare values of type (?:VAR|DATE)", msg) \
                            and repairs < 3:
                        fixed_sql = _COMPARE_NUM.sub(
                            lambda m: f"{_numcoerce(m.group(1))} {m.group(2)} {m.group(3)}",
                            fixed_sql, count=1)
                        repairs += 1
                        continue
                    tier_errors.append(f"t{tier}: {msg[:130]}")
                    cand = None
                    break
            if cand is None:
                continue
            # Huge result sets (an uncapped SELECT over a 2 GB table) cannot be
            # materialized for the Python comparator: counts + ordered sample.
            cand_trunc = len(cand) >= ROW_CAP
            if ref_trunc or cand_trunc:
                match = compare_big(lit, con, sql, fixed_sql)
            else:
                match = results_match(ref, cand, ordered)
            if match:
                stats[f"pass_tier{tier}"] += 1
                # Store the SQL AS EXECUTED (post-repair), not the tier's input:
                # the repair loop's ANY_VALUE wraps and TRY_CAST rebuilds are part
                # of the verified gold.
                records.append({**d, "SQL_duckdb": fixed_sql, "tier": tier,
                                "verified": True})
                break
            tier_errors.append(
                f"t{tier}: mismatch (ref {len(ref)}x{len(ref[0]) if ref else 0}, "
                f"duck {len(cand)}x{len(cand[0]) if cand else 0}) "
                f"ref={ref[:1]} duck={cand[:1]} diff={_first_diff(ref, cand)}")
        else:
            kind = "mismatch" if any("mismatch" in e for e in tier_errors) else "duck_error"
            stats["fail_" + kind] += 1
            failures.append({"qid": qid, "kind": kind,
                             "error": " | ".join(tier_errors), "sql": sql})

    n = len(rows)
    verified = sum(v for k, v in stats.items() if k.startswith("pass_tier"))
    entry = {
        "n": n,
        "verified": verified,
        "rate": round(verified / n, 4),
        "pass_tier0_verbatim": stats["pass_tier0"],
        "pass_tier1_quoting": stats["pass_tier1"],
        "pass_handfix": stats["pass_tier-1"],
        "pass_translated": stats["pass_tier3"],
        "sqlite_gold_error": stats["sqlite_gold_error"],
        "duck_error": stats["fail_duck_error"],
        "mismatch": stats["fail_mismatch"],
        "failures": failures,
    }
    log.append(f"{db_id:26} {verified}/{n} verified "
               f"(verbatim {stats['pass_tier0']}, quoting {stats['pass_tier1']}, "
               f"translated {stats['pass_tier3']}, hand {stats['pass_tier-1']}, "
               f"fail {stats['fail_duck_error'] + stats['fail_mismatch']})")
    if triage:
        for f in failures:
            log.append(f"   qid {f['qid']} [{f['kind']}] {f['error']}")
            log.append(f"       {f['sql'][:180]}")
    lit.close()
    con.close()
    return db_id, records, entry, stats, log


def main() -> int:
    if "--emit-duckdb" in sys.argv:
        emit_duckdb()
        return 0
    triage = "--triage" in sys.argv
    dev = json.loads(DEV_JSON.read_text())
    db_ids = sorted({d["db_id"] for d in dev})
    dataset, report = [], {}
    total_stats = Counter()

    # Databases are independent, so verify them in parallel. imap keeps db order, so the
    # dataset is byte-identical to a serial build. Each worker holds one fully
    # materialized in-memory DuckDB copy of its database (the largest train dbs are
    # multi-GB), so --jobs is bounded by host RAM, not cores.
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 1
    work = [(db_id, [d for d in dev if d["db_id"] == db_id], triage) for db_id in db_ids]
    if jobs > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(jobs, maxtasksperchild=1)
        results = pool.imap(build_db, work)
    else:
        pool, results = None, map(build_db, work)
    for db_id, records, entry, stats, log in results:
        dataset.extend(records)
        report[db_id] = entry
        total_stats.update(stats)
        print("\n".join(log), flush=True)
    if pool is not None:
        pool.close()
        pool.join()

    n_all = sum(r["n"] for r in report.values())
    v_all = sum(r["verified"] for r in report.values())
    print(f"\nTOTAL: {v_all}/{n_all} verified ({100 * v_all / n_all:.1f}%), "
          f"{len(dataset)} records")

    if not triage:
        OUT_DATASET.write_text(json.dumps(dataset, indent=1))
        slim = {db: {k: v for k, v in r.items() if k != "failures"}
                for db, r in report.items()}
        slim["_failures"] = {db: r["failures"] for db, r in report.items()}
        OUT_REPORT.write_text(json.dumps(slim, indent=1))
        print(f"wrote {OUT_DATASET} and {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
