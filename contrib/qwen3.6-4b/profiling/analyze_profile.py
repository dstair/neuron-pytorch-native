#!/usr/bin/env python3
"""Analyze a neuron-explorer parquet profile for Qwen3.5-4B decode."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def interval_merge_ns(
    starts: np.ndarray,
    ends: np.ndarray,
    t0: int,
    t1: int,
) -> int:
    if len(starts) == 0:
        return 0
    starts = np.maximum(starts.astype(np.int64), t0)
    ends = np.minimum(ends.astype(np.int64), t1)
    valid = starts < ends
    starts, ends = starts[valid], ends[valid]
    if len(starts) == 0:
        return 0
    order = np.argsort(starts)
    total = 0
    current_end = 0
    for start, end in zip(starts[order], ends[order]):
        if start >= current_end:
            total += int(end - start)
            current_end = int(end)
        elif end > current_end:
            total += int(end - current_end)
            current_end = int(end)
    return total


def short_source(value: object) -> str:
    if not isinstance(value, str) or not value:
        return "?"
    return value.rsplit("/", 1)[-1]


def extract_matmul_dims(operands: object) -> tuple[int | None, int | None, int | None]:
    if not isinstance(operands, str):
        return None, None, None
    km = re.search(r"(\d+)\*(\d+)\s*$", operands)
    if not km:
        return None, None, None
    k, m = int(km.group(1)), int(km.group(2))
    src = re.search(r"src=\S+\[[\d,\-]+\]\[(\d+),", operands)
    n = int(src.group(1)) if src else None
    return k, m, n


def load(path: Path, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path / f"{name}.parquet", columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    root = args.parquet_dir
    summary = load(root, "Summary").iloc[0]
    metadata = load(root, "Metadata").iloc[0]
    tensors = load(root, "TensorInfo")
    active = load(root, "ActiveTime")
    dma = load(
        root,
        "DmaPacket",
        ["start_ts", "end_ts", "transfer_bytes", "throughput", "queue_type"],
    )
    dma_agg = load(
        root,
        "DmaPacketAggregated",
        [
            "id",
            "start_ts",
            "end_ts",
            "transfer_bytes",
            "source",
            "dest",
            "variable",
            "is_transpose_mode",
            "write_num_sbuf_partitions",
            "read_num_sbuf_partitions",
        ],
    )
    inst = load(
        root,
        "Instruction",
        [
            "id",
            "start_ts",
            "end_ts",
            "duration_ns",
            "evt_wait_time_ns",
            "engine",
            "opcode",
            "operands",
            "bir_debug_info_source_location",
            "nki_source_location",
            "hbm_read_bytes",
            "hbm_write_bytes",
            "tensor_instruction_type",
            "adjusted_flops",
        ],
    )
    flow = load(
        root,
        "Flow",
        ["in_id", "out_id", "in_table", "out_table", "dependency_type"],
    )

    kernel_dma = dma[
        (dma["queue_type"] != "instruction") & (dma["transfer_bytes"] > 4)
    ].copy()
    t0 = int(kernel_dma["start_ts"].min())
    t1 = int(kernel_dma["end_ts"].max())
    total_time = float(summary["total_time"])
    total_ns = total_time * 1e9

    dma_peak = float(metadata["dma_ddr_bandwidth"])
    te_peak = (
        float(metadata["tensor_engine_num_rows"])
        * float(metadata["tensor_engine_num_cols"])
        * 2.0
        * float(metadata["tensor_engine_clock_freq"])
        * 1e9
    )
    dma_active = float(summary["dma_active_time"])
    transfer_bytes = int(kernel_dma["transfer_bytes"].sum())
    necessary_bytes = int(
        tensors[tensors["type"].isin(["IN", "OUT"])]["size"].sum()
    )
    te_active = float(summary["tensor_engine_active_time"])
    hardware_flops = float(summary["adjusted_hardware_flops"])
    transpose_flops = float(summary["adjusted_transpose_flops"])

    memory_ideal = transfer_bytes / dma_peak
    memory_no_reloads = necessary_bytes / dma_peak
    compute_ideal = hardware_flops / te_peak
    compute_useful = (hardware_flops - transpose_flops) / te_peak
    engine_times = {
        "GpSimd": float(summary["gpsimd_engine_active_time"]),
        "Tensor": te_active,
        "DMA": dma_active,
        "Scalar": float(summary["scalar_engine_active_time"]),
        "Vector": float(summary["vector_engine_active_time"]),
        "Sync": float(summary["sync_engine_active_time"]),
    }
    bottleneck = max(engine_times, key=engine_times.get)
    perfect_pipeline = engine_times[bottleneck]

    bounds = {
        "total_ms": total_time * 1e3,
        "memory_bound_ms": dma_active * 1e3,
        "memory_bound_ideal_ms": memory_ideal * 1e3,
        "memory_bound_ideal_no_reloads_ms": memory_no_reloads * 1e3,
        "compute_bound_ms": te_active * 1e3,
        "compute_bound_ideal_flops_ms": compute_ideal * 1e3,
        "compute_bound_ideal_useful_flops_ms": compute_useful * 1e3,
        "perfect_pipeline_ms": perfect_pipeline * 1e3,
        "bottleneck_engine": bottleneck,
        "engine_active_ms": {key: value * 1e3 for key, value in engine_times.items()},
        "gaps_ms": {
            "dma_idle": (total_time - dma_active) * 1e3,
            "dma_inefficiency": (dma_active - memory_ideal) * 1e3,
            "excess_hbm_traffic_at_peak": (memory_ideal - memory_no_reloads)
            * 1e3,
            "te_idle": (total_time - te_active) * 1e3,
            "te_underutilization": (te_active - compute_ideal) * 1e3,
            "te_transposes_at_peak": (compute_ideal - compute_useful) * 1e3,
            "serialization": (total_time - perfect_pipeline) * 1e3,
        },
    }

    size_bins = [-1, 64, 256, 1024, 4096, 16384, np.inf]
    size_labels = ["<=64", "65-256", "257-1K", "1K-4K", "4K-16K", ">16K"]
    packet_sizes = pd.cut(
        kernel_dma["transfer_bytes"],
        bins=size_bins,
        labels=size_labels,
    )
    packet_breakdown = []
    for label in size_labels:
        rows = kernel_dma[packet_sizes == label]
        packet_breakdown.append(
            {
                "bucket_bytes": label,
                "packets": int(len(rows)),
                "bytes": int(rows["transfer_bytes"].sum()),
                "percent_bytes": float(
                    rows["transfer_bytes"].sum() / transfer_bytes * 100
                ),
            }
        )

    dma_agg_bytes = int(dma_agg["transfer_bytes"].sum())
    dma_analysis = {
        "active_ms": dma_active * 1e3,
        "transfer_bytes": transfer_bytes,
        "necessary_bytes": necessary_bytes,
        "excess_ratio": transfer_bytes / necessary_bytes,
        "achieved_gb_s": transfer_bytes / dma_active / 1e9,
        "peak_gb_s": dma_peak / 1e9,
        "achieved_percent_peak": transfer_bytes / dma_active / dma_peak * 100,
        "packet_count": int(len(kernel_dma)),
        "average_packet_bytes": float(kernel_dma["transfer_bytes"].mean()),
        "packet_size_breakdown": packet_breakdown,
        "queue_breakdown": [
            {
                "queue_type": str(queue),
                "packets": int(len(rows)),
                "bytes": int(rows["transfer_bytes"].sum()),
            }
            for queue, rows in kernel_dma.groupby("queue_type")
        ],
        "dpa_bytes": dma_agg_bytes,
        "dpa_coverage_percent": dma_agg_bytes / transfer_bytes * 100,
        "summary_hbm_bytes": int(summary["hbm_read_bytes"] + summary["hbm_write_bytes"]),
        "summary_spill_bytes": int(
            summary["spill_save_bytes"] + summary["spill_reload_bytes"]
        ),
    }

    virtual_save = dma_agg[
        dma_agg["dest"].fillna("").str.contains("VIRTUAL")
    ]["transfer_bytes"].sum()
    virtual_reload = dma_agg[
        dma_agg["source"].fillna("").str.contains("VIRTUAL")
    ]["transfer_bytes"].sum()
    tensor_sizes = tensors.set_index("variable_name")["size"].to_dict()
    tensor_types = tensors.set_index("variable_name")["type"].to_dict()
    by_variable = (
        dma_agg.groupby("variable", dropna=False)["transfer_bytes"]
        .sum()
        .sort_values(ascending=False)
    )
    variable_rows = []
    reload_excess = 0.0
    for variable, size in by_variable.head(30).items():
        tensor_size = float(tensor_sizes.get(variable, 0.0) or 0.0)
        tensor_type = tensor_types.get(variable)
        repeat = float(size / tensor_size) if tensor_size else None
        if tensor_type == "IN" and repeat is not None and repeat > 1.0:
            reload_excess += float(size - tensor_size)
        variable_rows.append(
            {
                "variable": str(variable),
                "type": tensor_type,
                "tensor_bytes": tensor_size,
                "transfer_bytes": int(size),
                "repeat": repeat,
            }
        )
    reload_analysis = {
        "virtual_spill_save_bytes": int(virtual_save),
        "virtual_spill_reload_bytes": int(virtual_reload),
        "top_variables": variable_rows,
        "top30_input_reload_excess_bytes": reload_excess,
    }

    source_column = "bir_debug_info_source_location"
    source_coverage = {
        "instruction_rows": int(len(inst)),
        "with_bir_source": int(inst[source_column].notna().sum()),
        "with_nki_source": int(inst["nki_source_location"].notna().sum()),
    }

    matmuls = inst[
        (inst["engine"] == "Tensor")
        & (inst["opcode"] == "MATMUL")
        & (inst["start_ts"] >= t0)
        & (inst["start_ts"] < t1)
    ].copy()
    transposes = matmuls[matmuls["tensor_instruction_type"] == "TRANSPOSE"].copy()
    regular = matmuls[matmuls["tensor_instruction_type"] == "REGULAR"].copy()
    achieved_te = hardware_flops / te_active

    def matmul_sources(frame: pd.DataFrame, limit: int = 20) -> list[dict]:
        rows = []
        frame = frame.copy()
        frame["source"] = frame[source_column].map(short_source)
        for source, group in frame.groupby("source"):
            first = group.iloc[0]
            k, m, n = extract_matmul_dims(first["operands"])
            rows.append(
                {
                    "source": source,
                    "count": int(len(group)),
                    "adjusted_flops": float(group["adjusted_flops"].fillna(0).sum()),
                    "k": k,
                    "m": m,
                    "n": n,
                }
            )
        return sorted(rows, key=lambda row: row["adjusted_flops"], reverse=True)[
            :limit
        ]

    te_analysis = {
        "peak_tflops": te_peak / 1e12,
        "achieved_tflops_while_active": achieved_te / 1e12,
        "active_utilization_percent": achieved_te / te_peak * 100,
        "hardware_flops": hardware_flops,
        "transpose_flops": transpose_flops,
        "transpose_percent": transpose_flops / hardware_flops * 100,
        "regular_matmuls": int(len(regular)),
        "transpose_matmuls": int(len(transposes)),
        "top_matmul_sources": matmul_sources(matmuls),
        "top_transpose_sources": matmul_sources(transposes),
    }

    gpsimd = inst[inst["engine"] == "GpSimd"].copy()
    gpsimd["source"] = gpsimd[source_column].map(short_source)
    gpsimd_opcode = (
        gpsimd.groupby("opcode")
        .agg(
            count=("id", "size"),
            summed_duration_ns=("duration_ns", "sum"),
            summed_event_wait_ns=("evt_wait_time_ns", "sum"),
        )
        .sort_values("summed_duration_ns", ascending=False)
    )
    gpsimd_sources = []
    for source, group in gpsimd[gpsimd["source"] != "?"].groupby("source"):
        active_ns = interval_merge_ns(
            group["start_ts"].to_numpy(),
            group["end_ts"].to_numpy(),
            int(inst["start_ts"].min()),
            int(inst["end_ts"].max()),
        )
        gpsimd_sources.append(
            {
                "source": source,
                "instructions": int(len(group)),
                "merged_active_ms": active_ns / 1e6,
                "summed_duration_ms": float(group["duration_ns"].sum() / 1e6),
                "summed_event_wait_ms": float(
                    group["evt_wait_time_ns"].fillna(0).sum() / 1e6
                ),
            }
        )
    gpsimd_analysis = {
        "active_ms": engine_times["GpSimd"] * 1e3,
        "top_opcodes": [
            {
                "opcode": str(index),
                **{key: float(value) for key, value in row.items()},
            }
            for index, row in gpsimd_opcode.head(20).iterrows()
        ],
        "top_sources": sorted(
            gpsimd_sources,
            key=lambda row: row["merged_active_ms"],
            reverse=True,
        )[:20],
    }

    inst_to_dma = flow[
        (flow["in_table"].str.lower() == "instruction")
        & (flow["out_table"].str.lower() == "dmapacketaggregated")
    ][["in_id", "out_id"]]
    source_dma = inst_to_dma.merge(
        inst[
            [
                "id",
                source_column,
                "hbm_read_bytes",
                "hbm_write_bytes",
            ]
        ].rename(columns={"id": "in_id"}),
        on="in_id",
        how="inner",
    ).merge(
        dma_agg[
            [
                "id",
                "transfer_bytes",
                "source",
                "dest",
                "variable",
                "is_transpose_mode",
                "write_num_sbuf_partitions",
                "read_num_sbuf_partitions",
            ]
        ].rename(columns={"id": "out_id"}),
        on="out_id",
        how="inner",
    )
    source_dma["short_source"] = source_dma[source_column].map(short_source)
    attributed_dma_bytes = int(source_dma["transfer_bytes"].sum())
    source_dma_rows = []
    for source, group in source_dma[source_dma["short_source"] != "?"].groupby(
        "short_source"
    ):
        source_dma_rows.append(
            {
                "source": source,
                "transfers": int(len(group)),
                "bytes": int(group["transfer_bytes"].sum()),
                "median_transfer_bytes": float(group["transfer_bytes"].median()),
            }
        )
    dma_analysis["source_attributed_bytes"] = attributed_dma_bytes
    dma_analysis["source_attribution_coverage_percent"] = (
        attributed_dma_bytes / transfer_bytes * 100
    )
    dma_analysis["top_sources"] = sorted(
        source_dma_rows,
        key=lambda row: row["bytes"],
        reverse=True,
    )[:20]

    spill_reloads = dma_agg[
        dma_agg["source"].fillna("").str.contains("VIRTUAL")
    ][["id", "transfer_bytes"]]
    spill_flow = inst_to_dma[inst_to_dma["out_id"].isin(set(spill_reloads["id"]))]
    spill_sources = spill_flow.merge(
        inst[["id", source_column]].rename(columns={"id": "in_id"}),
        on="in_id",
        how="inner",
    ).merge(
        spill_reloads.rename(columns={"id": "out_id"}),
        on="out_id",
        how="inner",
    )
    spill_sources["source"] = spill_sources[source_column].map(short_source)
    spill_by_source = (
        spill_sources[spill_sources["source"] != "?"]
        .groupby("source")["transfer_bytes"]
        .agg(["count", "sum"])
        .sort_values("sum", ascending=False)
    )
    reload_analysis["top_spill_trigger_sources"] = [
        {
            "source": str(index),
            "reloads": int(row["count"]),
            "spill_round_trip_bytes": int(row["sum"] * 2),
        }
        for index, row in spill_by_source.head(20).iterrows()
    ]

    result = {
        "profile": os.path.basename(root),
        "quality": {
            **source_coverage,
            "source_coverage_percent": source_coverage["with_bir_source"]
            / source_coverage["instruction_rows"]
            * 100,
            "dpa_coverage_percent": dma_analysis["dpa_coverage_percent"],
        },
        "bounds": bounds,
        "dma": dma_analysis,
        "reloads_and_spills": reload_analysis,
        "tensor_engine": te_analysis,
        "gpsimd": gpsimd_analysis,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
