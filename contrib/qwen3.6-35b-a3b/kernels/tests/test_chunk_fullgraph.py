#!/usr/bin/env python3
"""Minimal repro: does the chunk flash kernel trip [NCC_IMGN901] MacroGeneration
'Can only vectorize loop or free axes' when compiled inside torch.compile(fullgraph)?

The standalone test_flash_chunk passes because it calls the nki_op EAGERLY (each call
traced alone). The bucketed harness wraps the kernel in torch.compile(backend=neuron,
fullgraph=True) — a single custom-call node in a big XLA graph. This isolates whether
the KERNEL itself (not DeltaNet/MoE) reproduces the vectorizer bug under fullgraph.

Single core, tiny — compiles in a few minutes.
Run in DLC:  python kernels/tests/test_chunk_fullgraph.py
"""
import os
import sys
import torch
import torch_neuronx  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "kernels"))
import gqa_flash_prefill_35b_ops  # noqa: E402  registers gqa35b::flash_prefill_chunk

Q_HEADS = int(os.environ.get("GQA_Q_HEADS", "4"))
HEAD_DIM = 256
CHUNK = 512
KMAX = 1024


def fn(q, k, v, qb):
    # trivial surrounding op so it's a real graph, then the kernel custom-call
    o = torch.ops.gqa35b.flash_prefill_chunk(q, k, v, qb)
    return o * 1.0


def main():
    dev = "privateuseone:0"
    cfn = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
    q = (torch.randn(Q_HEADS, CHUNK, HEAD_DIM) * 0.3).float().to(dev)
    k = (torch.randn(KMAX, HEAD_DIM) * 0.3).float().to(dev)
    v = (torch.randn(KMAX, HEAD_DIM) * 0.3).float().to(dev)
    qb = torch.tensor([[0.0]], dtype=torch.float32).to(dev)
    out = cfn(q, k, v, qb)
    o = out.cpu().float()
    print(f"[chunk-fullgraph] COMPILED+RAN ok, shape={tuple(o.shape)} "
          f"finite={bool(torch.isfinite(o).all())} norm={o.norm():.3e}", flush=True)


if __name__ == "__main__":
    main()
