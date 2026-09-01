#!/usr/bin/env python3
"""Standard analysis battery over ingested neuron-explorer parquet.

    python3 profile_queries.py --data /mnt/nvme/lnc1-work/profile/data \
        --tags rA,rB,rC,rD --steps 39.1 --measured-s 43.54

Host often has no pandas -- use a venv (ours: /mnt/nvme/profile-venv/bin/python).

Sections, in the order you should read them:
  0 RECONCILE   does (region time x regions x steps) match the measured wall? If not, every
                per-region number below is suspect. See section 6 for why.
  1 ENGINES     per-region and aggregate engine active/instruction time, MFU/MBU, gap.
  2 OPCODES     top (engine, opcode) by active time -- "what is the machine doing".
  3 FIXEDCOST   ops whose duration is CONSTANT regardless of size. The single highest-yield
                query here: for these the ONLY lever is issuing fewer of them.
  4 SOURCES     per-source-line rollup + per-opcode breakdown of the top suspects, so you can
                tell a compute hog from a stall site from a call site.
  5 COMPONENTS  roll sources up into subsystems via --components patterns.
  6 REGIONDIFF  region A vs B per (src,opcode): exposes data-dependent runtime loops that
                capture-replay under-executed.
"""
import argparse
import json
import os

import pandas as pd

BOOKKEEPING = ["EVENT_SEMAPHORE", "EVENT_SEMAPHORE_RANGE_CLEAR", "DRAIN", "NOTIFY", "SEMAPHORE_OP"]
ENGINES = ["vector", "gpsimd", "tensor", "scalar", "sync"]
DEFAULT_COMPONENTS = "moe_cte:moe_cte_35b|bwmm:bwmm_shard_on_I|attn:attention_cte|deltanet:deltanet|transpose_helper:private_nkl/transpose|model_py:static_decode"
FMT = lambda v: "%.2f" % v  # noqa: E731


def load_instructions(data, tags, cols):
    frames = []
    for t in tags:
        p = "%s/profiles/global/%s@latest/Instruction.parquet" % (data, t)
        if not os.path.exists(p):
            print("  !! missing %s" % p)
            continue
        frames.append(pd.read_parquet(p, columns=cols).assign(region=t))
    if not frames:
        raise SystemExit("no Instruction.parquet found -- did ingest.sh run?")
    ins = pd.concat(frames, ignore_index=True)
    ins["ms"] = ins["duration_ns"] / 1e6
    ins["src"] = ins["nki_source_location"].fillna("").replace("", "<unattributed>")
    return ins


def summaries(w, tags):
    out = {}
    for t in tags:
        p = "%s/summ_%s.json" % (w, t)
        if os.path.exists(p):
            out[t] = list(json.load(open(p)).values())[0]
    return out


def sec_engines(summ, tags):
    print("=" * 78)
    print("1 ENGINES  (active% overlaps across engines and sums >100%;")
    print("            the critical-path metric is instruction_time / total_time)")
    hdr = "%-6s %9s |" % ("region", "total_ms")
    for e in ENGINES:
        hdr += "%14s" % e[:6]
    print(hdr + "%14s |%7s%7s" % ("dma", "MFU%", "MBU%"))
    agg = {}
    for t in tags:
        d = summ.get(t)
        if not d:
            continue
        T = d["total_time"] * 1000
        line = "%-6s %9.2f |" % (t, T)
        for e in ENGINES:
            a = d["%s_engine_active_time" % e] * 1000
            line += "%8.1f(%4.1f)" % (a, d["%s_engine_active_time_percent" % e] * 100)
            agg.setdefault(e, [0.0, 0.0])
            agg[e][0] += a
            agg[e][1] += d["%s_engine_instruction_time" % e] * 1000
        line += "%8.1f(%4.1f) |%7.2f%7.2f" % (
            d["dma_active_time"] * 1000, d["dma_active_time_percent"] * 100,
            d["mfu_estimated_percent"] * 100, d["mbu_estimated_percent"] * 100)
        agg.setdefault("dma", [0.0, 0.0])
        agg["dma"][0] += d["dma_active_time"] * 1000
        agg.setdefault("total", [0.0, 0.0])
        agg["total"][0] += T
        print(line)
    T = agg.get("total", [0])[0]
    if T:
        print("\n  aggregate over %d regions: total_time %.2f ms" % (len(tags), T))
        for e in ENGINES:
            print("    %-7s active %8.2f (%5.1f%%)  instruction %8.2f (%5.1f%% of total)"
                  % (e, agg[e][0], agg[e][0] / T * 100, agg[e][1], agg[e][1] / T * 100))
        pp = max([agg[e][0] for e in ENGINES] + [agg["dma"][0]])
        print("    perfect_pipeline %.2f -> serialization_gap %.2f ms (%.1f%%)"
              % (pp, T - pp, (T - pp) / T * 100))
        top = max(ENGINES, key=lambda e: agg[e][1])
        share = agg[top][1] / T * 100
        verdict = ("%s's instruction stream IS the critical path" % top if share > 90
                   else "no engine owns the critical path -> latency/dependency bound")
        print("    -> max instruction share = %s %.1f%%: %s" % (top, share, verdict))
    return T


def sec_reconcile(summ, tags, steps, measured_s):
    print("=" * 78)
    print("0 RECONCILE")
    if not steps or not measured_s:
        print("  (skipped -- pass --steps and --measured-s to enable this gate)")
        return
    tot = sum(summ[t]["total_time"] for t in tags if t in summ) * 1000
    print("  sum of %d regions      : %8.2f ms/step -> %6.2f s   (%.0f%% of measured)"
          % (len(tags), tot, tot * steps / 1000, tot * steps / 1000 / measured_s * 100))
    for t in tags:
        if t not in summ:
            continue
        one = summ[t]["total_time"] * 1000 * len(tags)
        print("  %s x %d regions%s: %8.2f ms/step -> %6.2f s   (%.0f%% of measured)"
              % (t, len(tags), " " * 7, one, one * steps / 1000, one * steps / 1000 / measured_s * 100))
    print("  measured wall          : %6.2f s" % measured_s)
    print("  -> USE the row nearest 100%. If the SUM is far low and ONE region is near 100%,")
    print("     a data-dependent runtime loop under-executed in the other regions (see sec 6).")


def sec_opcodes(data, tags, T):
    print("=" * 78)
    print("2 OPCODES  top (engine, opcode) by active time")
    frames = []
    for t in tags:
        p = "%s/profiles/global/%s@latest/OpcodeSummary.parquet" % (data, t)
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
    if not frames:
        print("  (no OpcodeSummary)")
        return
    op = pd.concat(frames, ignore_index=True)
    g = (op.groupby(["engine", "opcode"], as_index=False)
           .agg(instr=("instruction_count", "sum"),
                instr_ms=("instruction_time_ns", lambda s: s.sum() / 1e6),
                active_ms=("active_time_ns", lambda s: s.sum() / 1e6)))
    g["avg_us"] = g["instr_ms"] * 1000 / g["instr"]
    if T:
        g["pct"] = g["active_ms"] / T * 100
    print(g.sort_values("active_ms", ascending=False).head(22).to_string(index=False, float_format=FMT))


def sec_fixedcost(ins, T):
    print("=" * 78)
    print("3 FIXEDCOST  ops with (near-)CONSTANT duration -> cost = COUNT x fixed, not size.")
    print("             Lever: issue fewer of them. Fix the call structure, not the data.")
    g = (ins.groupby(["engine", "opcode"], as_index=False)
           .agg(instr=("ms", "size"), ms=("ms", "sum"),
                avg_us=("duration_ns", "mean"), std_ns=("duration_ns", "std"),
                min_ns=("duration_ns", "min"), max_ns=("duration_ns", "max")))
    g["avg_us"] = g["avg_us"] / 1000.0
    g["cv_pct"] = g["std_ns"] / (g["avg_us"] * 1000.0) * 100.0
    if T:
        g["pct"] = g["ms"] / T * 100
    fixed = g[(g["cv_pct"] < 2.0) & (g["instr"] > 20) & (g["avg_us"] > 1.0)]
    print("  -- FIXED (coefficient of variation < 2%, avg > 1us):")
    if len(fixed):
        print(fixed.sort_values("ms", ascending=False).head(12).to_string(index=False, float_format=FMT))
    else:
        print("     none found")
    print("  -- all ops > 1us avg, by total time (for context):")
    print(g[g.avg_us > 1.0].sort_values("ms", ascending=False).head(12)
           .to_string(index=False, float_format=FMT))


def sec_sources(ins, T):
    print("=" * 78)
    print("4 SOURCES  per-source instruction time (overlaps engines -> RANKING ONLY)")
    s = (ins.groupby("src", as_index=False).agg(instr=("ms", "size"), ms=("ms", "sum")))
    s["avg_us"] = s["ms"] * 1000 / s["instr"]
    s["pct_attr"] = s["ms"] / s["ms"].sum() * 100
    top = s.sort_values("ms", ascending=False).head(20)
    print(top.to_string(index=False, float_format=FMT))
    print("\n  -- opcode breakdown of the top 5 (is it compute, a stall, or a call site?)")
    for src in top.head(6)["src"]:
        if src == "<unattributed>":
            continue
        sub = ins[ins.src == src]
        by = (sub.groupby(["engine", "opcode"], as_index=False)
                 .agg(instr=("ms", "size"), ms=("ms", "sum")).sort_values("ms", ascending=False))
        sem = sub[sub.opcode.isin(BOOKKEEPING)].ms.sum()
        tag = "  <-- MOSTLY STALL/WAIT" if sem > 0.5 * sub.ms.sum() else ""
        print("   %s   total %.2f ms (wait %.0f%%)%s" % (src[-72:], sub.ms.sum(),
                                                         sem / max(sub.ms.sum(), 1e-9) * 100, tag))
        print(by.head(4).to_string(index=False, float_format=FMT))
    print("\n  -- top (src, engine, opcode) EXCLUDING semaphore/drain bookkeeping")
    nb = ins[~ins.opcode.isin(BOOKKEEPING)]
    r = (nb.groupby(["src", "engine", "opcode"], as_index=False)
           .agg(instr=("ms", "size"), ms=("ms", "sum")))
    r["avg_us"] = r["ms"] * 1000 / r["instr"]
    print(r.sort_values("ms", ascending=False).head(15).to_string(index=False, float_format=FMT))
    print("\n  -- all *SEMAPHORE* share (if ~75%+ of the step, you are latency-bound)")
    sem = ins[ins.opcode.str.contains("SEMAPHORE", na=False)]
    if T:
        print("     %.2f ms = %.1f%% of total_time (overlapping across engines)"
              % (sem.ms.sum(), sem.ms.sum() / T * 100))


def sec_components(ins, spec):
    print("=" * 78)
    print("5 COMPONENTS  subsystem rollup (instruction time; overlaps -> RANKING ONLY)")
    for item in spec.split("|"):
        if ":" not in item:
            continue
        label, pat = item.split(":", 1)
        sub = ins[ins.src.str.contains(pat, na=False, regex=True)]
        if not len(sub):
            continue
        by = sub.groupby("engine")["ms"].sum().sort_values(ascending=False)
        print("  %-18s %8.2f ms (%7d instrs) | %s"
              % (label, sub.ms.sum(), len(sub),
                 "  ".join("%s=%.1f" % (k, v) for k, v in by.items())))


def sec_regiondiff(ins, tags):
    if len(tags) < 2:
        return
    print("=" * 78)
    a, b = tags[0], tags[1]
    print("6 REGIONDIFF  %s minus %s -- large one-sided rows mean a DATA-DEPENDENT runtime" % (a, b))
    print("              loop ran in one capture and not the other (synthetic-input trap).")
    ga = (ins[ins.region == a].groupby(["src", "engine", "opcode"], as_index=False)
            .agg(iA=("ms", "size"), msA=("ms", "sum")))
    gb = (ins[ins.region == b].groupby(["src", "engine", "opcode"], as_index=False)
            .agg(iB=("ms", "size"), msB=("ms", "sum")))
    m = ga.merge(gb, on=["src", "engine", "opcode"], how="outer").fillna(0)
    m["d_ms"] = m["msA"] - m["msB"]
    print(m.sort_values("d_ms", ascending=False).head(12).to_string(index=False, float_format=FMT))
    onesided = m[(m.iB == 0) & (m.iA > 100)]
    if len(onesided):
        print("  !! %d (src,opcode) pairs present in %s and ABSENT in %s (%.1f ms) --"
              % (len(onesided), a, b, onesided.msA.sum()))
        print("     do NOT sum regions; pick the region where the work actually ran.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the --data-path given to ingest.sh")
    ap.add_argument("--summ-dir", default=None, help="dir holding summ_<tag>.json (default: --data/..)")
    ap.add_argument("--tags", default="rA,rB,rC,rD")
    ap.add_argument("--steps", type=float, default=None, help="inference steps per measured run")
    ap.add_argument("--measured-s", type=float, default=None, help="measured wall seconds")
    ap.add_argument("--components", default=DEFAULT_COMPONENTS, help="label:regex|label:regex")
    a = ap.parse_args()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 74)
    tags = [t for t in a.tags.split(",") if t]
    w = a.summ_dir or os.path.dirname(a.data.rstrip("/")) or "."

    summ = summaries(w, tags)
    sec_reconcile(summ, tags, a.steps, a.measured_s)
    T = sec_engines(summ, tags)

    ins = load_instructions(a.data, tags, ["engine", "opcode", "duration_ns",
                                          "nki_source_location", "hbm_read_bytes",
                                          "hbm_write_bytes", "adjusted_flops"])
    print("\nloaded %d instructions across %s" % (len(ins), tags))
    sec_opcodes(a.data, tags, T)
    sec_fixedcost(ins, T)
    sec_sources(ins, T)
    sec_components(ins, a.components)
    sec_regiondiff(ins, tags)


if __name__ == "__main__":
    main()
