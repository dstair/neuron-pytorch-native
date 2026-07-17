#!/usr/bin/env python3
"""Verify that a compiled Native graph can persist an aliased K/V-style buffer."""

import argparse
import time

import nki
import nki.isa as nisa
import nki.language as nl
import torch
import torch.nn as nn
from torch_neuronx import nki_op


HEAD_DIM = 256


@nki.jit
def nki_write_cache_rows(cache, rows, position):
    batch = rows.shape[0]
    sequence = cache.shape[0] // batch

    pos = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos, src=position)

    for batch_idx in nl.sequential_range(batch):
        cache_row = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=cache_row,
            data=pos,
            op0=nl.add,
            operand0=batch_idx * sequence,
        )
        nisa.dma_copy(
            dst=cache.ap(
                [[HEAD_DIM, 1], [1, HEAD_DIM]],
                scalar_offset=cache_row,
            ),
            src=rows[batch_idx : batch_idx + 1, 0:HEAD_DIM],
        )

    ack = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    ack_sbuf = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ack_sbuf, value=1.0)
    nisa.dma_copy(dst=ack, src=ack_sbuf)
    return ack


@nki_op("qwen35b_test::write_cache_rows", mutates_args={"cache"})
def write_cache_rows(
    cache: torch.Tensor,
    rows: torch.Tensor,
    position: torch.Tensor,
) -> torch.Tensor:
    return nki_write_cache_rows(cache, rows, position)


class StatefulCacheProbe(nn.Module):
    def __init__(self, batch_size, sequence):
        super().__init__()
        self.batch_size = batch_size
        self.sequence = sequence
        self.register_buffer(
            "cache",
            torch.zeros(
                batch_size * sequence,
                HEAD_DIM,
                dtype=torch.bfloat16,
            ),
            persistent=False,
        )

    def forward(self, rows, position):
        ack = write_cache_rows(self.cache, rows, position)
        probe = self.cache.reshape(
            self.batch_size, self.sequence, HEAD_DIM
        )[:, 0]
        return probe, ack


def synchronize():
    torch.neuron.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--bench-iters", type=int, default=100)
    args = parser.parse_args()

    device = torch.neuron.current_device()
    model = StatefulCacheProbe(args.batch_size, args.max_seq_len).to(device)
    compiled = torch.compile(
        model, backend="neuron", fullgraph=True, dynamic=False
    )

    rows0 = torch.full(
        (args.batch_size, HEAD_DIM),
        1.25,
        dtype=torch.bfloat16,
        device=device,
    )
    rows1 = torch.full(
        (args.batch_size, HEAD_DIM),
        -0.75,
        dtype=torch.bfloat16,
        device=device,
    )
    pos0 = torch.tensor([[0]], dtype=torch.int32, device=device)
    pos1 = torch.tensor([[1]], dtype=torch.int32, device=device)

    probe0, _ = compiled(rows0, pos0)
    synchronize()
    probe1, _ = compiled(rows1, pos1)
    synchronize()

    expected0 = rows0.cpu()
    torch.testing.assert_close(probe0.cpu(), expected0, rtol=0, atol=0)
    torch.testing.assert_close(probe1.cpu(), expected0, rtol=0, atol=0)

    cache = model.cache.reshape(
        args.batch_size, args.max_seq_len, HEAD_DIM
    ).cpu()
    torch.testing.assert_close(cache[:, 0], expected0, rtol=0, atol=0)
    torch.testing.assert_close(cache[:, 1], rows1.cpu(), rtol=0, atol=0)
    assert torch.count_nonzero(cache[:, 2:]) == 0

    for _ in range(3):
        compiled(rows1, pos1)
    synchronize()
    start = time.time()
    for _ in range(args.bench_iters):
        compiled(rows1, pos1)
    synchronize()
    elapsed = time.time() - start
    print(
        f"PASS stateful alias B={args.batch_size} S={args.max_seq_len}: "
        f"{elapsed * 1000.0 / args.bench_iters:.4f} ms/call"
    )


if __name__ == "__main__":
    main()
