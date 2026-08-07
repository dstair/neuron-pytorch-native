"""CPU unit gate for shard_gqa_kv — GQA KV-head replication at TP=8.

Runs on the box host venv (imports static_decode, which registers the NKI ops);
pure CPU, NO Neuron device. It pins the correctness of the KV-head replication
that TP=8 relies on: the model has only 4 GQA KV heads, so for TP > 4 each KV
head must be REPLICATED across rep = TP // 4 ranks, and every rank must hold
exactly the single KV head that its contiguous query-head slice attends to.

A mis-routing here is the class of bug the on-device --gate-tp8-gqa check and the
coherence run would otherwise surface as incoherent tokens; this catches it
deterministically without a device.

Invariants checked (world_size in {1,2,4,8}):
  1. Head mapping: rank r holds KV head r // rep   (rep = max(1, TP//NUM_KV)).
  2. Rep-group bit-identity: ranks sharing a KV head get byte-identical weights.
  3. Forward consistency: the rank's sharded k_proj(x) equals the reference full
     model's KV head (r // rep) for random x.
  4. Query/KV alignment: rank r's query heads [r*qpr : (r+1)*qpr] all attend to
     KV head r // rep (GQA head q attends to KV head q // (NUM_Q // NUM_KV)),
     i.e. the head the rank actually holds — so replication is self-consistent.
"""
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))          # kernels/tests
_KERNELS = _os.path.dirname(_HERE)                            # kernels/
_REPO = _os.path.dirname(_KERNELS)                            # repo root (has static_decode.py)
_sys.path.insert(0, _KERNELS)
_sys.path.insert(0, _REPO)

import torch
import torch.nn as nn

from static_decode import shard_gqa_kv, shard_linear_colwise

HEAD_DIM = 256
NUM_KV = 4
NUM_Q = 24
HIDDEN = 5120


def _make_kv_linear():
    """A [NUM_KV*HEAD_DIM, HIDDEN] projection whose per-head row blocks are
    distinguishable, so weight equality is a meaningful signal."""
    torch.manual_seed(0)
    lin = nn.Linear(HIDDEN, NUM_KV * HEAD_DIM, bias=False)
    with torch.no_grad():
        for h in range(NUM_KV):
            lin.weight[h * HEAD_DIM:(h + 1) * HEAD_DIM] = (
                torch.randn(HEAD_DIM, HIDDEN) + 100.0 * h
            )
    return lin


def _shard_one(orig_w, rank, world_size):
    lin = nn.Linear(HIDDEN, NUM_KV * HEAD_DIM, bias=False)
    lin.weight = nn.Parameter(orig_w.clone())
    shard_gqa_kv(lin, rank, world_size, HEAD_DIM)
    return lin


def test_replication(world_size):
    orig = _make_kv_linear()
    orig_w = orig.weight.data.clone()
    rep = max(1, world_size // NUM_KV)
    x = torch.randn(8, HIDDEN)  # a few tokens for the forward-consistency check

    per_rank_w = []
    for rank in range(world_size):
        lin = _shard_one(orig_w, rank, world_size)
        per_rank_w.append(lin.weight.data)

        # (1) head mapping + out_features
        if world_size > NUM_KV:
            expected_head = rank // rep
            assert lin.out_features == HEAD_DIM, (
                f"TP={world_size} rank {rank}: out_features {lin.out_features} != {HEAD_DIM}"
            )
        else:
            # plain contiguous colwise: rank r owns the r-th 1/world_size slice
            expected_head = rank * (NUM_KV // world_size)
            assert lin.out_features == NUM_KV * HEAD_DIM // world_size
        exp_w = orig_w[expected_head * HEAD_DIM:(expected_head + 1) * HEAD_DIM]
        # for world_size < NUM_KV a rank owns >1 head; compare the leading head only
        assert torch.equal(lin.weight.data[:HEAD_DIM], exp_w[:HEAD_DIM]), (
            f"TP={world_size} rank {rank}: weight != KV head {expected_head}"
        )

        # (3) forward consistency vs the reference full-model head
        ref_head = expected_head
        ref_out = x @ orig_w[ref_head * HEAD_DIM:(ref_head + 1) * HEAD_DIM].T
        shard_out = x @ lin.weight.data[:HEAD_DIM].T
        assert torch.equal(ref_out, shard_out), (
            f"TP={world_size} rank {rank}: sharded k_proj(x) != reference head {ref_head}"
        )

    # (2) rep-group bit-identity (only meaningful when replicating)
    if world_size > NUM_KV:
        for g in range(NUM_KV):
            members = [r for r in range(world_size) if r // rep == g]
            base = per_rank_w[members[0]]
            for r in members[1:]:
                assert torch.equal(base, per_rank_w[r]), (
                    f"TP={world_size} rep-group {g}: rank {r} != rank {members[0]}"
                )

    # (4) query/KV alignment: every query head a rank owns attends to the KV
    #     head that rank actually holds.
    qpr = NUM_Q // world_size            # query heads per rank
    q_per_kv = NUM_Q // NUM_KV           # GQA group size (=6)
    for rank in range(world_size):
        held_kv = rank // rep if world_size > NUM_KV else rank * (NUM_KV // world_size)
        for qh in range(rank * qpr, (rank + 1) * qpr):
            attends = qh // q_per_kv
            if world_size > NUM_KV:
                assert attends == held_kv, (
                    f"TP={world_size} rank {rank} q-head {qh} attends KV {attends}, holds {held_kv}"
                )
    return rep


if __name__ == "__main__":
    for ws in (1, 2, 4, 8):
        rep = test_replication(ws)
        print(f"  TP={ws}: replication PASS (rep={rep}, "
              f"{'replicated' if ws > NUM_KV else 'sharded'})")
    print("ALL PASS ✓  shard_gqa_kv KV-head replication verified")
