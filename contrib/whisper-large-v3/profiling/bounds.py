"""Performance bounds + hotspot breakdown for an ingested neuron-explorer profile.

Follows neuron-nki-profile-querying references/performance-bounds.md (memory / compute / pipeline families).
usage: python bounds.py <display-name> [<display-name> ...]
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

NE = os.path.join(os.environ.get("ASR_WORK", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                  "profiles", "ne-data", "profiles", "global")


def interval_merge_ns(starts, ends, t0, t1):
    if len(starts) == 0:
        return 0
    starts, ends = np.maximum(starts, t0), np.minimum(ends, t1)
    ok = starts < ends
    starts, ends = starts[ok], ends[ok]
    order = np.argsort(starts)
    total, cur = 0, 0
    for s, e in zip(starts[order], ends[order]):
        if s >= cur:
            total += e - s
            cur = e
        elif e > cur:
            total += e - cur
            cur = e
    return total


def load(name, table):
    p = f"{NE}/{name}@latest/{table}.parquet"
    return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()


def analyze(name):
    md = load(name, "Metadata").iloc[0]
    pk, agg, act = load(name, "DmaPacket"), load(name, "DmaPacketAggregated"), load(name, "ActiveTime")
    inst, tens = load(name, "Instruction"), load(name, "TensorInfo")
    dma_bw = md["dma_ddr_bandwidth"]
    te_peak = md["tensor_engine_num_rows"] * md["tensor_engine_num_cols"] * 2.0 * md["tensor_engine_clock_freq"] * 1e9
    src = agg if len(agg) else pk
    t0 = int(min(src["start_ts"].min(), inst["start_ts"].min()))
    t1 = int(max(src["end_ts"].max(), (inst["start_ts"] + inst["duration_ns"]).max()))
    total = (t1 - t0) / 1e9

    mem = interval_merge_ns(pk["start_ts"].values, pk["end_ts"].values, t0, t1) / 1e9 if len(pk) else float("nan")
    kp = pk[(pk["queue_type"] != "instruction") & (pk["transfer_bytes"] > 4)] if len(pk) else pk
    dma_bytes = kp["transfer_bytes"].sum() if len(kp) else float("nan")
    need = tens[tens["type"].isin(["IN", "OUT"])]["size"].sum() if len(tens) else float("nan")

    def eng(e):
        a = act[act["engine"].str.lower() == e]
        return interval_merge_ns(a["start_ts"].values, a["end_ts"].values, t0, t1) / 1e9

    mm = inst[(inst["engine"].str.lower() == "tensor") & (inst["opcode"] == "MATMUL")]
    hw = mm["adjusted_flops"].sum() if "adjusted_flops" in mm else float("nan")
    tr = mm[mm.get("tensor_instruction_type", pd.Series(dtype=str)) == "TRANSPOSE"]["adjusted_flops"].sum()
    engines = {"Tensor": eng("tensor"), "Vector": eng("vector"), "Scalar": eng("scalar"),
               "GpSimd": eng("gpsimd"), "DMA": mem}
    bn = max(engines, key=lambda k: engines[k] if engines[k] == engines[k] else -1)

    ms = lambda s: f"{s*1e3:9.3f} ms"  # noqa: E731
    print(f"\n===== {name}  total_time {ms(total)}   bottleneck {bn}")
    print("engine active:", "  ".join(f"{k} {v*1e3:.3f}ms ({v/total*100:.0f}%)" for k, v in engines.items()))
    print(f"memory : bound {ms(mem)} | ideal {ms(dma_bytes/dma_bw)} ({dma_bytes/1e6:.1f} MB moved, "
          f"{dma_bytes/max(mem,1e-12)/1e9:.0f} GB/s achieved vs {dma_bw/1e9:.0f} peak) | "
          f"no-reloads {ms(need/dma_bw)} ({need/1e6:.1f} MB IN+OUT)")
    print(f"compute: bound {ms(engines['Tensor'])} | ideal flops {ms(hw/te_peak)} ({hw/1e9:.2f} GF) | "
          f"useful {ms((hw-tr)/te_peak)} (transposes {tr/max(hw,1)*100:.0f}%)")
    print(f"pipeline: perfect {ms(engines[bn])}  serialization gap {ms(total-engines[bn])}")

    con = duckdb.connect()
    con.register("inst", inst)
    print("\n-- top engine/opcode by summed duration")
    print(con.execute("""SELECT engine, opcode, COUNT(*) n, ROUND(SUM(duration_ns)/1e6,3) ms,
                          ROUND(SUM(evt_wait_time_ns)/1e6,3) wait_ms FROM inst
                          GROUP BY 1,2 ORDER BY ms DESC LIMIT 12""").df().to_string(index=False))
    if len(pk):
        con.register("pk", pk)
        print("\n-- DMA packets by size bucket")
        print(con.execute("""SELECT CASE WHEN transfer_bytes<512 THEN '<512B' WHEN transfer_bytes<2048 THEN '<2KB'
                              WHEN transfer_bytes<32768 THEN '<32KB' ELSE '>=32KB' END sz, COUNT(*) n,
                              ROUND(SUM(transfer_bytes)/1e6,2) MB, ROUND(SUM(end_ts-start_ts)/1e6,3) pkt_ms
                              FROM pk WHERE queue_type!='instruction' GROUP BY 1 ORDER BY MB DESC""").df().to_string(index=False))
    if len(tens):
        print("\n-- largest IN/OUT tensors")
        t = tens[tens["type"].isin(["IN", "OUT"])].sort_values("size", ascending=False).head(8)
        cols = [c for c in ["name", "type", "size", "shape", "dtype"] if c in t.columns]
        print(t[cols].to_string(index=False))
    return dict(name=name, total=total, **engines)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        analyze(n)
