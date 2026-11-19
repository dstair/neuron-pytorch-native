"""Convert a verified bird_duckdb_{dev,train}.json into the motherduck training JSONL
shape (prompt / query / schema / category) that sqlft.data.load_rows consumes.

The Evidence line is baked into `prompt` as:

    <question>
    -- Evidence: <evidence>

so sqlft.data.render_prompt produces exactly the lab10 eval-time prompt:

    -- DuckDB SQL generation
    -- Schema:
    <DDL>
    -- Question: <question>
    -- Evidence: <evidence>
    -- Answer:

Train/eval prompt-format parity is the point: the model must see at training time
precisely what it sees at eval time.

    python3 to_training_jsonl.py [--set train|dev]
"""

import json
import pathlib
import sqlite3
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SPLIT = "train" if "--set" in sys.argv and sys.argv[sys.argv.index("--set") + 1] == "train" else "dev"
DATA = ROOT / ("train" if SPLIT == "train" else "dev_20240627")
DB_DIR = DATA / ("train_databases" if SPLIT == "train" else "dev_databases")
IN = ROOT / f"bird_duckdb_{SPLIT}.json"
OUT = ROOT / f"bird_{SPLIT}_train.jsonl"


def schema_ddl(sqlite_path: str) -> str:
    lit = sqlite3.connect(sqlite_path)
    ddl = [
        r[0] for r in lit.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")
        if r[0]
    ]
    lit.close()
    return "\n".join(ddl)


def main() -> None:
    rows = json.loads(IN.read_text())
    ddl_by_db: dict[str, str] = {}
    written = 0
    with OUT.open("w") as fh:
        for d in rows:
            db = d["db_id"]
            if db not in ddl_by_db:
                ddl_by_db[db] = schema_ddl(str(DB_DIR / db / f"{db}.sqlite"))
            prompt = d["question"].strip()
            if d.get("evidence", "").strip():
                prompt += "\n-- Evidence: " + d["evidence"].strip()
            fh.write(json.dumps({
                "prompt": prompt,
                "query": d["SQL_duckdb"].strip(),
                "schema": ddl_by_db[db],
                "category": db,
            }) + "\n")
            written += 1
    print(f"wrote {written} rows -> {OUT}")


if __name__ == "__main__":
    main()
