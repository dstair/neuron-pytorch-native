#!/usr/bin/env python3
"""CPU checks for the vocab-sharded greedy argmax reduction."""

import torch


def sharded_top1(logits, world_size):
    batch, vocab = logits.shape
    assert vocab % world_size == 0
    shard = vocab // world_size
    maxima = []
    global_ids = []
    for rank in range(world_size):
        local = logits[:, rank * shard:(rank + 1) * shard]
        local_max, local_id = local.max(dim=-1)
        maxima.append(local_max)
        global_ids.append(local_id.to(torch.int32) + rank * shard)

    maxima = torch.stack(maxima)
    global_ids = torch.stack(global_ids)
    global_max = maxima.max(dim=0).values
    invalid = torch.full_like(global_ids, -vocab)
    neg_winner_ids = torch.where(
        maxima == global_max.unsqueeze(0), -global_ids, invalid
    )
    return -neg_winner_ids.max(dim=0).values.to(torch.long)


def main():
    generator = torch.Generator().manual_seed(20260717)
    logits = torch.randn(37, 256, generator=generator)

    # Force winners onto every rank and exercise first-index tie breaking both
    # within one shard and across different shards.
    for rank in range(8):
        logits[rank, rank * 32 + 11] = 100 + rank
    logits[8, 7] = 200
    logits[8, 13] = 200
    logits[9, 63] = 300
    logits[9, 192] = 300
    logits[10] = 0

    expected = logits.argmax(dim=-1)
    for world_size in (1, 2, 4, 8):
        actual = sharded_top1(logits, world_size)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    print("PASS: sharded LM top-1 ownership, ties, batches, and ordering")


if __name__ == "__main__":
    main()
