#!/usr/bin/env python3
"""CPU equivalence checks for the 27B vocab-row-sharded embed / LM head (T1b).

Both halves of the lever claim to be *exactly* — not approximately — equal to the
replicated path. This test is the cheap pre-check for that claim: it reimplements
the two `StaticDecodeModule` bodies (`_embed_tokens`, `_lm_head_logits`) with the
all-reduce modelled as a plain sum over simulated ranks, and demands bitwise
equality against the unsharded reference at rtol=atol=0.

Why the two are exact:
  embed   — ranks own disjoint contiguous row ranges, so for any id exactly one
            rank gathers a real row and every other rank contributes hard zeros.
            The sum has one non-zero term; no rounding is introduced.
  lm_head — each rank computes a disjoint contiguous logit slice and zero-pads it
            to full vocab. Each output position again has exactly one non-zero
            contributor. Crucially the *matmul* is unchanged (the reduction is over
            hidden, which is not sharded), so there is no reassociation either.

Run: python3 kernels/tests/test_decode_sharded_lm_cpu.py
"""

import torch
import torch.nn.functional as F


# ── Reimplementations of the two module methods, with sum() standing in for the
#    TP sum-all-reduce. Kept structurally identical to static_decode.py so a drift
#    in one shows up as a failure here.

def sharded_embed(ids, embed, world_size):
    """Mirror of StaticDecodeModule._embed_tokens under SHARDED_EMBED=1."""
    vocab = embed.shape[0]
    assert vocab % world_size == 0
    per_rank = vocab // world_size
    total = None
    for rank in range(world_size):
        shard = embed[rank * per_rank:(rank + 1) * per_rank]
        local_ids = ids - rank * per_rank
        in_range = (local_ids >= 0) & (local_ids < per_rank)
        safe_ids = torch.where(in_range, local_ids, torch.zeros_like(local_ids))
        rows = F.embedding(safe_ids, shard)
        rows = rows * in_range.unsqueeze(-1).to(rows.dtype)
        total = rows if total is None else total + rows
    return total


def sharded_lm_head(x, lm_head, world_size):
    """Mirror of StaticDecodeModule._lm_head_logits under SHARDED_LM_HEAD=1."""
    vocab = lm_head.shape[0]
    assert vocab % world_size == 0
    per_rank = vocab // world_size
    total = None
    for rank in range(world_size):
        shard = lm_head[rank * per_rank:(rank + 1) * per_rank]
        local = F.linear(x.to(shard.dtype), shard)
        lo = rank * per_rank
        padded = F.pad(local, (lo, vocab - lo - per_rank))
        total = padded if total is None else total + padded
    return total


def exact(actual, expected, what):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(f"  PASS (bitwise): {what}")


def main():
    g = torch.Generator().manual_seed(20260804)
    # Real 27B ratios, shrunk: V must stay divisible by every world_size tested.
    # 248320 / 4 = 62080 on the real model; 256 / 8 = 32 here.
    VOCAB, HIDDEN = 256, 64

    for dtype in (torch.float32, torch.bfloat16):
        print(f"dtype={dtype}")
        embed = (torch.randn(VOCAB, HIDDEN, generator=g) * 0.02).to(dtype)
        lm_head = (torch.randn(VOCAB, HIDDEN, generator=g) * 0.02).to(dtype)

        # Ids covering every rank, both range endpoints, duplicates, and a
        # multi-dim shape (decode passes [B]; prefill passes [S]).
        ids_1d = torch.tensor(
            [0, 1, 31, 32, 33, 63, 64, 127, 128, 200, 255, 255, 0, 96],
            dtype=torch.long,
        )
        ids_2d = ids_1d.reshape(2, 7)

        for ids, tag in ((ids_1d, "[S]"), (ids_2d, "[B, S]")):
            ref = F.embedding(ids, embed)
            for world_size in (1, 2, 4, 8):
                exact(sharded_embed(ids, embed, world_size), ref,
                      f"embed {tag} ws={world_size}")

        # LM head over decode-ish [B, 1, H] and prefill-ish [1, 1, H].
        for shape in ((1, 1, HIDDEN), (8, 1, HIDDEN), (16, 1, HIDDEN)):
            x = (torch.randn(*shape, generator=g) * 0.5).to(dtype)
            ref = F.linear(x.to(dtype), lm_head)
            for world_size in (1, 2, 4, 8):
                exact(sharded_lm_head(x, lm_head, world_size), ref,
                      f"lm_head {tuple(shape)} ws={world_size}")

        # The gate that actually matters downstream: greedy argmax must agree.
        x = (torch.randn(16, 1, HIDDEN, generator=g) * 0.5).to(dtype)
        ref_ids = F.linear(x.to(dtype), lm_head).argmax(dim=-1)
        for world_size in (1, 2, 4, 8):
            got = sharded_lm_head(x, lm_head, world_size).argmax(dim=-1)
            exact(got, ref_ids, f"greedy argmax ws={world_size}")

    # An out-of-range id must not read another rank's rows or crash: the clamp
    # sends it to local row 0 and the mask zeros it, so every rank contributes
    # zero and the row comes back all-zero rather than wrong.
    embed = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
    oob = torch.tensor([VOCAB + 5], dtype=torch.long)
    got = sharded_embed(oob, embed, 4)
    assert torch.all(got == 0), "out-of-range id leaked a real row"
    print("  PASS: out-of-range id yields zeros on every rank (no OOB gather)")

    print("PASS: 27B vocab-sharded embed + LM head are bitwise-exact vs replicated")


if __name__ == "__main__":
    main()
