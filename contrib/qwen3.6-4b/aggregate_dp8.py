#!/usr/bin/env python3
"""Aggregate synchronized per-replica benchmark JSON files."""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no result files match {args.pattern!r}")
    rows = [json.load(open(path)) for path in paths]
    expected = rows[0]["dp"]
    if len(rows) != expected:
        raise SystemExit(f"found {len(rows)} replicas, expected {expected}")
    replica_ids = sorted(row["replica_id"] for row in rows)
    if replica_ids != list(range(expected)):
        raise SystemExit(f"unexpected replica ids: {replica_ids}")
    for key in ("fused_fp8_mlp", "fused_fp8_head"):
        values = {row.get(key, False) for row in rows}
        if len(values) != 1:
            raise SystemExit(f"inconsistent {key} values: {sorted(values)}")

    throughputs = [row["throughput_tok_s"] for row in rows]
    tpots = [row["tpot_ms"] for row in rows]
    tp = rows[0]["tp"]
    result = {
        "model": rows[0]["model"],
        "instance": rows[0]["instance"],
        "topology": rows[0]["topology"],
        "dp": expected,
        "tp": tp,
        "batch_size_per_replica": rows[0]["batch_size"],
        "aggregate_batch_size": rows[0]["aggregate_batch_size"],
        "max_seq_len": rows[0]["max_seq_len"],
        "dtype": rows[0]["dtype"],
        "fused_fp8_mlp": rows[0].get("fused_fp8_mlp", False),
        "fused_fp8_head": rows[0].get("fused_fp8_head", False),
        "aggregate_throughput_tok_s": sum(throughputs),
        "mean_replica_throughput_tok_s": statistics.mean(throughputs),
        "min_replica_throughput_tok_s": min(throughputs),
        "max_replica_throughput_tok_s": max(throughputs),
        "mean_tpot_ms": statistics.mean(tpots),
        "max_tpot_ms": max(tpots),
        "module_gib_per_replica": rows[0]["module_gib_per_replica"],
        "replica_results": [
            {
                "replica_id": row["replica_id"],
                "throughput_tok_s": row["throughput_tok_s"],
                "tpot_ms": row["tpot_ms"],
                "source": os.path.basename(path),
            }
            for row, path in zip(rows, paths)
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
