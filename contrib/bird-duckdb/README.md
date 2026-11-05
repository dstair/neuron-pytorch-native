# BIRD-for-DuckDB: a verified DuckDB-dialect execution benchmark

No DuckDB-dialect text-to-SQL benchmark with real data exists (BIRD/Spider are SQLite-flavored;
motherduckdb/duckdb-text2sql-25k is DuckDB dialect but ships no data). This builds one by
**verified translation** of the BIRD dev set: keep BIRD's real databases and questions,
translate gold SQL only where the dialects genuinely differ, and prove every translated gold
by executing it against DuckDB and comparing result sets with the SQLite original.

BIRD is CC BY-SA 4.0; a derivative is shareable with attribution.

## Status: prototype validated on one database (debit_card_specializing, 64 questions)

```
pass tier0 (raw pass-through):            46/64 (71.9%)
pass tier1 (backtick->quote):              0/64 (0.0%)
pass tier2 (+IIF->IF, BETWEEN):           14/64 (21.9%)
pass tier3 (+NULLIF, DATE-LIKE, hand):     4/64 (6.2%)
TOTAL verified:                           64/64 (100.0%)
```

So the measured dialect gap on this database: **~72% of gold runs verbatim, ~94% with two
schema-aware rules, 100% with three more targeted fixes.** Run it:

```bash
pip install duckdb
# dev_20240627/ from https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
python3 prototype.py [db_id]     # default debit_card_specializing
```

## The dialect gap, measured (the real answer to "SQLite vs DuckDB")

1. **`IIF` → `IF`** (12/64 here). SQLite has `IIF`; DuckDB spells it `IF`. Mechanical.
2. **Loose vs strict typing in `BETWEEN`** (a few). SQLite coerces `date_text BETWEEN 201101
   AND 201112`; DuckDB ≥0.10 refuses mixed-type BETWEEN. Fix is schema-aware: quote the
   literals only when the column is TEXT (string ordering would corrupt numeric columns).
3. **Division by zero: NULL vs inf** (2/64). SQLite `x/0` → NULL (and SUM/WHERE silently
   drop those rows); DuckDB → `inf`, which poisons aggregates and admits extra rows.
   `x / NULLIF(y, 0)` reproduces SQLite semantics exactly.
4. **`LIKE` on DATE columns** (1/64). SQLite stores dates as text; DuckDB has real DATEs and
   `~~(DATE, STRING)` doesn't bind. `CAST(col AS VARCHAR) LIKE ...`.
5. **Bare columns in GROUP BY** (1/64). SQLite serves an arbitrary row's value; DuckDB
   (Postgres semantics) requires the column grouped or aggregated. `ANY_VALUE(col)` is the
   faithful translation, but detecting it needs real parsing — it lives in a short,
   reviewed per-query hand-fix list.

Things that turned out NOT to be problems, measured:

- **Backticks.** DuckDB accepts SQLite's backtick identifier quoting natively; the
  tier-1 rewrite rule was never needed.
- **Result mismatches are float noise, not dialect.** All initial "mismatches" were
  parallel-aggregation accumulation-order differences (e.g. 459.9562642871061 vs ...0896).
  The comparator must compare multisets with a float-tolerant *pairwise* row equality
  (sort by repr, zip, rel_tol 1e-6) — sorting reprs and string-comparing flags noise as
  failure.
- **SQLite gold errors: 0** on this database.

## Design decisions for the full build

- **Native tables, not the scanner.** Data is imported via `sqlite_scan` then
  `CREATE TABLE AS` materialized, so execution semantics (types!) are DuckDB's own — which
  is what surfaces and lets us fix differences like DATE-vs-text once, in the open.
- **Every gold is verified against the SQLite original**, order-insensitive unless the
  query has ORDER BY, float-tolerant. Whatever cannot be auto-verified is dropped or
  hand-fixed *and counted* — the dataset proves its own gold answers.
- **The translation tiers and the hand-fix list are part of the artifact.** They are the
  measured dialect boundary, and consumers can re-run verification.

## Next steps

- [ ] Run all 11 dev databases; collect per-db tier rates and the full hand-fix list
- [ ] Emit the dataset: questions + DuckDB gold + native DuckDB database files + verification report
- [ ] Wire `execution_accuracy` into the qwen3.5-4b-training eval harness
      (gold must self-score ~100% first — the same calibration discipline as the parse ceiling)
