# BIRD-for-DuckDB: a verified DuckDB-dialect execution benchmark

No DuckDB-dialect text-to-SQL benchmark with real data exists (BIRD/Spider are SQLite-flavored;
motherduckdb/duckdb-text2sql-25k is DuckDB dialect but ships no data). This builds one by
**verified translation** of the BIRD dev set: keep BIRD's real databases and questions,
translate gold SQL only where the dialects genuinely differ, and prove every translated gold
by executing it against DuckDB and comparing result sets with the SQLite original.

BIRD is CC BY-SA 4.0 (https://bird-bench.github.io/); a derivative is shareable with
attribution. Source data: `dev_20240627/` from
https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip (symlinked or copied here, gitignored).

## Result: 1491/1534 questions verified (97.2%)

```
california_schools         83/89      european_football_2        121/129
card_games                 187/191    financial                  106/106
codebase_community         183/186    formula_1                  168/174
debit_card_specializing    64/64      student_club               154/158
superhero                  129/129    thrombosis_prediction      152/163
                                      toxicology                 143/145
```

```bash
pip install duckdb
python3 build.py             # full build -> bird_duckdb_dev.json + build_report.json
python3 build.py --triage    # failures only, no artifacts
```

Artifacts:

- **`bird_duckdb_dev.json`** (~1 MB): 1491 records, BIRD fields + `SQL_duckdb` (the verified
  DuckDB gold) + `tier` (how it was produced: 0 = verbatim, 1 = quoting/syntax only,
  3 = rule-translated, -1 = reviewed hand-fix).
- **`build_report.json`**: per-db tier rates plus every dropped question with its full
  error trail.

Tier mix, measured: ~84% of gold runs **verbatim**, ~9% needs only quoting/syntax
normalization (backticks, `LIMIT a,b`, strftime arg order, `ILIKE`, `AS "percent"`),
~3% needs semantic rules (IIF→IF, BETWEEN literal quoting, NULLIF division, text→number
coercion), 8 queries carry a **reviewed hand-fix**, and **43 were dropped** (2.8%).

## The drops, and why

Every drop is counted and listed in `build_report.json`. Classes:

1. **Non-deterministic gold (~2/3 of drops).** `ORDER BY` on a non-unique key plus `LIMIT`
   (the winner is a tie), or a bare GROUP BY column where SQLite serves an arbitrary row.
   These golds are flaky in *any* engine — they flip between runs of the same verifier.
2. **Junk gold.** `'2011-02-23' - '2010-09-01' = 1` in SQLite (leading-year text
   arithmetic), `OR '+-'` as a boolean, `datetime() - birthday > 35`.
3. **Hard semantic gaps (a handful).** Mixed-type columns where the text values are
   reachable data (`frameVersion = 'future'` vs junk-as-NULL), lap-time strings
   ('05:07') used numerically in ways SQLite's prefix coercion handles but a faithful
   translation can't reach without a full parser.

## The measured dialect boundary (what translation actually costs)

Syntax/quoting (mechanical): backtick identifiers (**not** accepted by DuckDB when they
contain spaces/parens), `LIMIT offset, count`, `STRFTIME(fmt, date)` → `STRFTIME(date, fmt)`
argument order, SQLite's case-insensitive `LIKE` → `ILIKE`, reserved-word aliases
(`percent`, `Date`).

Semantic (SQLite leniency DuckDB refuses): `IIF` → `IF`; mixed-type `BETWEEN`; division
by zero (NULL vs `inf` → `NULLIF`); text in arithmetic and aggregates (SQLite
leading-prefix coercion → `REGEXP_EXTRACT` + `TRY_CAST` + `COALESCE`); bare GROUP BY
columns → `ANY_VALUE`; SQLite integer division → `FLOOR`.

Materialization choices that matter: native tables via `sqlite_scan` + `CREATE TABLE AS`;
REAL columns always through the all-varchar + `TRY_CAST AS DOUBLE` path (the typed scan's
f32 rounds values before a cast can recover them); query-time conversion errors lazily
rebuild the offending column via `TRY_CAST`.

Comparator calibrations: float-tolerant *pairwise* multiset equality (parallel aggregation
order makes 459.9562642871061 vs ...0896 routine, not a mismatch); `ORDER BY` queries get
an ordered comparison with a multiset fallback (ties are only partially deterministic);
`datetime` objects normalize to text; NaN equals NaN; booleans unify with 1/0.

## Next steps

- [x] Prototype on one database (64/64)
- [x] Full build: 1491/1534 verified
- [ ] Wire `execution_accuracy` into the qwen3.5-4b-training eval harness
      (`labs/lab10_bird_exec_eval.py` in that package)
- [ ] Ship native `.duckdb` database files alongside the dataset json
