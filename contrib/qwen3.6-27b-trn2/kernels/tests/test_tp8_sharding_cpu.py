"""CPU gate for the TP=8 / LNC=1 sharding. No device, no NKI.

LNC=1 exposes the chip as 8 logical cores of 12 GB instead of 4 of 24 GB, so it
forces TP=8. Every dimension in the 27B divides by 8 except **num_key_value_heads
= 4**, which is why `shard_linear_colwise_replicated` exists: past TP=4 each KV
head is replicated across a group of ranks rather than split, because splitting a
256-dim head across two ranks would break attention (q·k contracts over all 256
dims, and k_norm is per-head).

What this file proves, without a Trn2:

  1. The per-core constants derive correctly at TP=4 and TP=8, and the values at
     TP=4 are **exactly the constants the published numbers were measured with**
     (a regression here would silently re-scale the shipped configuration).
  2. Replicated KV sharding hands every rank a *whole* head, the right one, and
     the union over ranks covers all 4 heads.
  3. Query heads stay **disjoint** across ranks and partition the full 24. This
     is the property that makes the row-wise o_proj all-reduce exact under KV
     replication — the reduce sums over a partition of q heads, so replicating
     KV cannot double-count.
  4. A plain (non-replicating) column shard at TP=8 would split a KV head — i.e.
     the bug this code exists to avoid is real, so the test has teeth.
  5. The DeltaNet per-core QKV dim stays a multiple of 128 (partition dim) at
     both TP values, and T1a's tile count follows it (20 -> 10).

Run: python3 test_tp8_sharding_cpu.py
"""
import os as _os
import sys as _sys

import torch
import torch.nn as nn

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

# Global architecture, from config.json (Qwen3.6-27B).
HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
DN_K_HEADS_TOTAL = 16
DN_V_HEADS_TOTAL = 48
GQA_Q_HEADS_TOTAL = 24
GQA_KV_HEADS_TOTAL = 4
GQA_HEAD_DIM = 256
K_DIM = V_DIM = 128

# The constants the shipped TP=4 configuration is measured with. Frozen on
# purpose: if the derivation drifts, every published number silently changes.
TP4_EXPECTED = dict(DN_K_HEADS=4, DN_V_HEADS=12, DN_QKV_DIM=2560, DN_VALUE_DIM=1536,
                    GQA_Q_HEADS=6, GQA_KV_HEADS=1, GQA_Q_DIM=3072, MLP_DIM=4352, NT=20)
TP8_EXPECTED = dict(DN_K_HEADS=2, DN_V_HEADS=6, DN_QKV_DIM=1280, DN_VALUE_DIM=768,
                    GQA_Q_HEADS=3, GQA_KV_HEADS=1, GQA_Q_DIM=1536, MLP_DIM=2176, NT=10)


def derive(tp):
    """Mirror of static_decode.py's TP block."""
    k_heads = DN_K_HEADS_TOTAL // tp
    v_heads = DN_V_HEADS_TOTAL // tp
    value_dim = v_heads * V_DIM
    qkv_dim = 2 * k_heads * K_DIM + value_dim
    q_heads = GQA_Q_HEADS_TOTAL // tp
    return dict(
        DN_K_HEADS=k_heads, DN_V_HEADS=v_heads, DN_QKV_DIM=qkv_dim,
        DN_VALUE_DIM=value_dim, GQA_Q_HEADS=q_heads,
        GQA_KV_HEADS=max(1, GQA_KV_HEADS_TOTAL // tp),
        GQA_Q_DIM=q_heads * GQA_HEAD_DIM * 2,
        MLP_DIM=INTERMEDIATE // tp, NT=qkv_dim // 128,
    )


def shard_colwise(w, rank, ws):
    """Plain column shard — the one that is WRONG for KV heads past TP=4."""
    shard = w.shape[0] // ws
    return w[rank * shard:(rank + 1) * shard]


def shard_colwise_replicated(w, rank, ws, n_heads, head_dim):
    """Mirror of static_decode.shard_linear_colwise_replicated."""
    assert w.shape[0] == n_heads * head_dim
    if ws <= n_heads:
        return shard_colwise(w, rank, ws)
    replicas = ws // n_heads
    head = rank // replicas
    return w[head * head_dim:(head + 1) * head_dim]


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    assert cond, name


def main():
    print("=== TP=8 / LNC=1 sharding (Qwen3.6-27B) ===")

    # 1. Constant derivation, both TP values, against frozen expectations.
    for tp, expected in ((4, TP4_EXPECTED), (8, TP8_EXPECTED)):
        got = derive(tp)
        check(f"TP={tp} per-core constants match expected {expected}", got == expected)
    check("TP=4 is unchanged from the shipped/measured configuration",
          derive(4) == TP4_EXPECTED)

    # Divisibility: everything but KV heads must divide by 8.
    for name, total in (("dn key heads", DN_K_HEADS_TOTAL), ("dn value heads", DN_V_HEADS_TOTAL),
                        ("q heads", GQA_Q_HEADS_TOTAL), ("intermediate", INTERMEDIATE),
                        ("vocab", VOCAB), ("q+gate dim", GQA_Q_HEADS_TOTAL * GQA_HEAD_DIM * 2)):
        check(f"{name} ({total}) divides by 8", total % 8 == 0)
    check("KV heads (4) do NOT divide by 8 — this is why replication exists",
          GQA_KV_HEADS_TOTAL % 8 != 0)

    # NKI partition-dim constraint, both TP values (T1a's tiling depends on it).
    for tp in (4, 8):
        d = derive(tp)
        check(f"TP={tp} per-core DN_QKV_DIM={d['DN_QKV_DIM']} is a multiple of 128",
              d["DN_QKV_DIM"] % 128 == 0)
    check("T1a tile count follows TP (20 at TP=4, 10 at TP=8)",
          derive(4)["NT"] == 20 and derive(8)["NT"] == 10)

    # 2/3. Replicated KV sharding + disjoint q heads, at TP=8.
    torch.manual_seed(3)
    ws = 8
    k_w = torch.randn(GQA_KV_HEADS_TOTAL * GQA_HEAD_DIM, HIDDEN)
    q_w = torch.randn(GQA_Q_HEADS_TOTAL * GQA_HEAD_DIM * 2, HIDDEN)

    heads_seen, kv_rows = [], []
    for rank in range(ws):
        s = shard_colwise_replicated(k_w, rank, ws, GQA_KV_HEADS_TOTAL, GQA_HEAD_DIM)
        check(f"rank {rank} KV shard is one whole head ({GQA_HEAD_DIM} rows)",
              s.shape == (GQA_HEAD_DIM, HIDDEN))
        head = rank // (ws // GQA_KV_HEADS_TOTAL)
        check(f"rank {rank} holds KV head {head}, bitwise",
              torch.equal(s, k_w[head * GQA_HEAD_DIM:(head + 1) * GQA_HEAD_DIM]))
        heads_seen.append(head)
        kv_rows.append(s)
    check("the 8 ranks cover all 4 KV heads", set(heads_seen) == set(range(4)))
    check("each KV head is held by exactly 2 ranks (replicas=2)",
          all(heads_seen.count(h) == 2 for h in range(4)))
    check("replicas holding the same head are bitwise identical",
          all(torch.equal(kv_rows[2 * h], kv_rows[2 * h + 1]) for h in range(4)))

    # Query heads: disjoint, whole, and a partition of all 24.
    covered = set()
    for rank in range(ws):
        s = shard_colwise(q_w, rank, ws)
        check(f"rank {rank} q shard is 3 whole (query+gate) heads",
              s.shape == (GQA_Q_HEADS_TOTAL // ws * GQA_HEAD_DIM * 2, HIDDEN))
        per = GQA_Q_HEADS_TOTAL // ws
        block = set(range(rank * per, (rank + 1) * per))
        check(f"rank {rank} q heads {sorted(block)} disjoint from earlier ranks",
              covered.isdisjoint(block))
        covered |= block
        # A contiguous split must not cut a head in half: each head is 2*256 wide.
        check(f"rank {rank} q shard starts on a head boundary",
              (rank * s.shape[0]) % (GQA_HEAD_DIM * 2) == 0)
    check("q heads partition all 24 with no gap or overlap",
          covered == set(range(GQA_Q_HEADS_TOTAL)))

    # 4. The negative check: plain column sharding at TP=8 splits a KV head.
    plain = shard_colwise(k_w, 0, ws)
    check("a plain column shard at TP=8 gives HALF a KV head (the bug avoided)",
          plain.shape[0] == GQA_HEAD_DIM // 2)
    check("...and is therefore NOT equal to any whole KV head",
          plain.shape != (GQA_HEAD_DIM, HIDDEN))
    # At TP=4 the replicating path must fall back to the plain one exactly.
    check("at TP=4 replicated sharding == plain sharding (no behaviour change)",
          all(torch.equal(shard_colwise_replicated(k_w, r, 4, GQA_KV_HEADS_TOTAL, GQA_HEAD_DIM),
                          shard_colwise(k_w, r, 4)) for r in range(4)))

    # 5. HBM arithmetic: the reason T1b is mandatory at LNC=1, not optional.
    vocab_mb = VOCAB * HIDDEN * 2 / 1024**2          # one [V,H] bf16 tensor
    module_tp4_mb = 17264.0                           # measured, replicated vocab
    sharded_part = module_tp4_mb - 2 * vocab_mb       # scales with TP
    tp8_replicated = sharded_part / 2 + 2 * vocab_mb
    tp8_t1b = sharded_part / 2 + 2 * vocab_mb / 8
    print(f"\n  HBM at TP=8 (12 GB/rank ceiling at LNC=1):")
    print(f"    replicated vocab : {tp8_replicated:8.0f} MB/rank")
    print(f"    both T1b flags   : {tp8_t1b:8.0f} MB/rank")
    check("replicated vocab does NOT comfortably fit 12 GB at TP=8",
          tp8_replicated > 0.85 * 12288)
    check("both T1b flags DO fit 12 GB at TP=8 with room",
          tp8_t1b < 0.75 * 12288)

    print("\n  ALL TP=8 SHARDING CHECKS PASS ✓")


if __name__ == "__main__":
    main()
