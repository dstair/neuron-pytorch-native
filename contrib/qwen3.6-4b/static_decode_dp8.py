#!/usr/bin/env python3
"""Independent TP=1 decode replicas for Qwen/Qwen3.5-4B.

Each process owns a complete FP8 model on one logical NeuronCore. Processes
share no model-parallel collectives; a small filesystem barrier aligns their
timed windows so aggregate throughput can be summed across replica JSON files.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time


# These must be set before static_decode imports and registers the NKI kernels.
os.environ.setdefault("DN_K_HEADS", "16")
os.environ.setdefault("DN_V_HEADS", "32")
os.environ.setdefault("GQA_Q_HEADS", "4")
os.environ.setdefault("DN_TILED_CONV", "1")

import torch

import model_dims as D
from static_decode import (
    TILED_PMAX,
    StaticDecode4B,
    _state_occupancy,
    load_sharded_weights,
)


def file_barrier(
    root: str | None,
    phase: str,
    replica_id: int,
    replica_count: int,
    timeout: float,
) -> None:
    if replica_count == 1:
        return
    if not root:
        raise ValueError("--barrier-dir is required when --replica-count > 1")
    os.makedirs(root, exist_ok=True)
    marker = os.path.join(root, f"{phase}.{replica_id}")
    with open(marker, "w") as f:
        f.write(f"{os.getpid()}\n")
    deadline = time.time() + timeout
    pattern = os.path.join(root, f"{phase}.*")
    while len(glob.glob(pattern)) < replica_count:
        if time.time() > deadline:
            found = sorted(os.path.basename(p) for p in glob.glob(pattern))
            raise TimeoutError(
                f"barrier {phase!r} timed out: found {found}, "
                f"expected {replica_count} replicas"
            )
        time.sleep(0.05)


def kv_occupancy(module: StaticDecode4B) -> list[int]:
    counts = []
    local_kv_heads = module.td["gqa_kv_heads"]
    for gi in range(D.NUM_GQA):
        for kvh in range(local_kv_heads):
            suffix = str(gi) if local_kv_heads == 1 else f"{gi}_{kvh}"
            cache = getattr(module, f"kv_k{suffix}").detach().cpu()
            counts.append(
                int((cache.reshape(-1, cache.shape[-1]) != 0).any(-1).sum())
            )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/models/Qwen3.5-4B")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--bench-iters", type=int, default=20)
    parser.add_argument(
        "--fp8-scope",
        choices=("all", "mlp-head"),
        default="all",
    )
    parser.add_argument("--replica-id", type=int, default=0)
    parser.add_argument("--replica-count", type=int, default=1)
    parser.add_argument("--lnc", type=int, choices=(1, 2), default=1)
    parser.add_argument("--barrier-dir")
    parser.add_argument("--barrier-timeout", type=float, default=900.0)
    parser.add_argument("--output-json")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--fused-fp8-mlp", action="store_true")
    parser.add_argument("--fused-fp8-head", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.replica_id < args.replica_count:
        raise ValueError("--replica-id must be in [0, --replica-count)")
    if args.batch_size > 128:
        raise ValueError("the FP8 kernel currently supports batch sizes <= 128")
    if args.max_seq_len < 2:
        raise ValueError("--max-seq-len must be at least 2")

    import torch_neuronx  # noqa: F401

    D.tp_dims(1)
    device = torch.neuron.current_device()
    print(
        f"Qwen3.5-4B DP replica {args.replica_id}/{args.replica_count}: "
        f"TP=1 BS={args.batch_size} S={args.max_seq_len} "
        f"dtype=fp8-{args.fp8_scope}",
        flush=True,
    )

    load_start = time.time()
    weights = load_sharded_weights(args.model_path, rank=0, world_size=1)
    module = StaticDecode4B(
        weights,
        args.max_seq_len,
        world_size=1,
        rank=0,
        batch_size=args.batch_size,
        fp8_scope=args.fp8_scope,
        fused_fp8_mlp=args.fused_fp8_mlp,
        fused_fp8_head=args.fused_fp8_head,
    )
    del weights
    module = module.to(device).eval()
    module_bytes = sum(b.numel() * b.element_size() for b in module.buffers())
    print(
        f"replica {args.replica_id}: module loaded in "
        f"{time.time() - load_start:.1f}s ({module_bytes / 2**30:.2f} GiB)",
        flush=True,
    )

    compiled = torch.compile(
        module,
        backend="neuron",
        fullgraph=True,
        dynamic=False,
    )
    batch = args.batch_size
    states = torch.zeros(
        D.NUM_DELTANET,
        batch,
        D.DN_VALUE_DIM,
        D.DN_V_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    conv = torch.zeros(
        D.NUM_DELTANET,
        batch,
        TILED_PMAX,
        D.DN_QKV_DIM // TILED_PMAX,
        D.DN_CONV_KERNEL - 1,
        dtype=torch.bfloat16,
        device=device,
    )
    input_ids = (
        (
            torch.arange(batch, device=device, dtype=torch.long)
            + args.replica_id * batch
        )
        % 1024
    ) + 1
    position = torch.tensor(1, device=device, dtype=torch.long)

    compile_start = time.time()
    with torch.no_grad():
        next_id, states, conv = compiled(input_ids, position, states, conv)
        torch.neuron.synchronize()
    compile_seconds = time.time() - compile_start
    first_ids = next_id.detach().cpu().tolist()
    print(
        f"replica {args.replica_id}: first step/cache load "
        f"{compile_seconds:.1f}s",
        flush=True,
    )

    if args.compile_only:
        return

    with torch.no_grad():
        for _ in range(args.warmup):
            next_id, states, conv = compiled(input_ids, position, states, conv)
        torch.neuron.synchronize()

    file_barrier(
        args.barrier_dir,
        "ready",
        args.replica_id,
        args.replica_count,
        args.barrier_timeout,
    )
    start = time.time()
    with torch.no_grad():
        for _ in range(args.bench_iters):
            next_id, states, conv = compiled(input_ids, position, states, conv)
        torch.neuron.synchronize()
    elapsed = time.time() - start
    file_barrier(
        args.barrier_dir,
        "finished",
        args.replica_id,
        args.replica_count,
        args.barrier_timeout,
    )

    tpot_ms = elapsed * 1000.0 / args.bench_iters
    throughput = batch * args.bench_iters / elapsed
    state_counts = _state_occupancy(states.detach().cpu())
    if any(v == 0 for v in state_counts):
        raise RuntimeError(f"empty DeltaNet recurrent state: {state_counts}")
    kv_counts = kv_occupancy(module)
    if any(v == 0 for v in kv_counts):
        raise RuntimeError(f"empty GQA KV cache: {kv_counts}")

    result = {
        "model": "Qwen/Qwen3.5-4B",
        "instance": "trn2.3xlarge",
        "topology": f"DP{args.replica_count}-TP1-LNC{args.lnc}",
        "dp": args.replica_count,
        "tp": 1,
        "replica_id": args.replica_id,
        "batch_size": batch,
        "aggregate_batch_size": batch * args.replica_count,
        "max_seq_len": args.max_seq_len,
        "dtype": f"fp8-{args.fp8_scope}",
        "fused_fp8_mlp": args.fused_fp8_mlp,
        "fused_fp8_head": args.fused_fp8_head,
        "compile_seconds": compile_seconds,
        "module_gib_per_replica": module_bytes / 2**30,
        "tpot_ms": tpot_ms,
        "throughput_tok_s": throughput,
        "first_ids": first_ids,
        "final_ids": next_id.detach().cpu().tolist(),
        "dn_nonzero_rows": state_counts,
        "kv_nonzero_rows": kv_counts,
    }
    print(
        f"BENCH replica={args.replica_id} BS={batch} "
        f"seq={args.max_seq_len}: TPOT {tpot_ms:.2f} ms/tok | "
        f"throughput {throughput:.1f} tok/s",
        flush=True,
    )
    print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
