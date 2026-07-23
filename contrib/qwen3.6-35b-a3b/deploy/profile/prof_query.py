#!/usr/bin/env python3
"""Phase D: full-graph profile attribution for tiled BS=128 FP8.
Reads ingested parquet (data-path/<profile>@latest/) + prints engine active
times, perfect-pipeline / serialization gap, source attribution, HBM traffic.
Usage: python3 prof_query.py <parquet_dir>   (dir containing *.parquet)
"""
import sys, glob, os
import pandas as pd, numpy as np

D = sys.argv[1]
def load(name):
    p = os.path.join(D, f"{name}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None

active = load("ActiveTime")
inst   = load("Instruction")
tensors= load("TensorInfo")
meta   = load("Metadata")
summ   = load("Summary")

print("=== available parquet tables ===")
for f in sorted(glob.glob(os.path.join(D, "*.parquet"))):
    try:
        n = pd.read_parquet(f).shape
    except Exception as e:
        n = f"err:{e}"
    print(f"  {os.path.basename(f):32s} {n}")

def interval_merge_ns(starts, ends, t0, t1):
    if len(starts) == 0: return 0
    starts = np.maximum(starts, t0); ends = np.minimum(ends, t1)
    valid = starts < ends; starts, ends = starts[valid], ends[valid]
    order = np.argsort(starts); total, cur = 0, 0
    for s, e in zip(starts[order], ends[order]):
        if s >= cur: total += e - s; cur = e
        elif e > cur: total += e - cur; cur = e
    return total

if active is not None:
    print("\n=== ActiveTime columns ===", list(active.columns))
    ecol = 'engine' if 'engine' in active.columns else active.columns[0]
    scol = 'start_ts' if 'start_ts' in active.columns else None
    ecol2= 'end_ts' if 'end_ts' in active.columns else None
    engines = active[ecol].unique()
    print("engines:", list(engines))
    if scol and ecol2:
        t0 = int(active[scol].min()); t1 = int(active[ecol2].max())
        total = (t1 - t0)/1e6  # -> ms (ns/1e6)
        etimes = {}
        for eng in engines:
            e = active[active[ecol]==eng]
            etimes[eng] = interval_merge_ns(e[scol].values, e[ecol2].values, t0, t1)/1e6
        print(f"\n=== engine active times (ms), total_window={total:.3f} ms ===")
        for eng,v in sorted(etimes.items(), key=lambda x:-x[1]):
            print(f"  {eng:10s} {v:9.3f} ms  ({100*v/total:5.2f}%)")
        pp = max(etimes.values()); bn = max(etimes, key=etimes.get)
        print(f"\n  perfect_pipeline (max engine) = {pp:.3f} ms  [bottleneck: {bn}]")
        print(f"  serialization_gap = total - pp = {total-pp:.3f} ms ({100*(total-pp)/total:.2f}%)")

# Source attribution
if inst is not None:
    print("\n=== Instruction columns ===", list(inst.columns))
    srccols = [c for c in inst.columns if any(k in c.lower() for k in
               ('source','file','line','func','name','kernel','op','loc'))]
    print("candidate source/attribution columns:", srccols)
    durcol = next((c for c in inst.columns if 'dur' in c.lower()), None)
    for sc in srccols:
        if durcol and inst[sc].nunique() < 200:
            g = inst.groupby(sc)[durcol].sum().sort_values(ascending=False)
            print(f"\n--- top by {sc} ({durcol} sum) ---")
            print(g.head(15).to_string())

if summ is not None:
    print("\n=== Summary (HBM / latency / flops keys) ===")
    s = summ.iloc[0] if hasattr(summ,'iloc') else summ
    for k in summ.columns:
        if any(t in k.lower() for t in ('hbm','byte','latency','total_time','flops','active_time')):
            print(f"  {k}: {s[k]}")
