#!/usr/bin/env python3
"""Four-layer real-weight TP=4 decode capture and baseline comparison."""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def compare(reference_dir, candidate_dir):
    names = ("logits", "deltanet", "conv", "kv_k", "kv_v")
    for rank in range(4):
        reference = torch.load(
            os.path.join(reference_dir, f"rank{rank}.pt"),
            map_location="cpu",
            weights_only=True,
        )
        candidate = torch.load(
            os.path.join(candidate_dir, f"rank{rank}.pt"),
            map_location="cpu",
            weights_only=True,
        )
        for name in names:
            expected = reference[name].float()
            actual = candidate[name].float()
            diff = actual - expected
            expected_flat = expected.reshape(-1)
            actual_flat = actual.reshape(-1)
            cosine = F.cosine_similarity(
                actual_flat, expected_flat, dim=0
            ).item()
            max_abs = diff.abs().max().item()
            rel_l2 = (
                torch.linalg.vector_norm(diff)
                / torch.linalg.vector_norm(expected).clamp_min(1e-12)
            ).item()
            print(
                f"rank={rank} {name}: cosine={cosine:.8f} "
                f"max_abs={max_abs:.6e} rel_l2={rel_l2:.6e}"
            )
            assert cosine > 0.999, (rank, name)
            assert rel_l2 < 0.02, (rank, name)

        ref_top = torch.topk(reference["logits"][0].float(), 5).indices
        new_top = torch.topk(candidate["logits"][0].float(), 5).indices
        assert torch.equal(ref_top, new_top), (rank, ref_top, new_top)
    print("PASS: four-layer logits and all carried states match")


def capture(model_path, output_dir, bench_iters=0):
    import torch_neuronx  # noqa: F401

    import model_dims as D
    import static_decode_35b as S

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4
    device = torch.neuron.current_device()

    D.load_from_config(os.path.join(model_path, "config.json"))
    D.NUM_LAYERS = 4
    D.NUM_GQA = 1
    D.NUM_DELTANET = 3
    weights = S.load_sharded_weights(
        model_path, rank, world_size, num_layers=D.NUM_LAYERS
    )
    model = S.StaticDecode35B(
        weights, max_seq_len=16, world_size=world_size, batch_size=1, rank=rank
    ).to(device).eval()
    del weights
    model.setup_segments(1, compile_each=True)

    td = D.tp_dims(world_size)
    vh = td["dn_v_heads"]
    qkv_dim = (
        2 * td["dn_k_heads"] * D.DN_K_DIM + vh * D.DN_V_DIM
    )
    dn = torch.zeros(
        D.NUM_DELTANET,
        1,
        vh * D.DN_K_DIM,
        D.DN_V_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    conv = torch.zeros(
        D.NUM_DELTANET,
        1,
        qkv_dim,
        D.DN_CONV_KERNEL - 1,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_k = torch.zeros(
        D.NUM_GQA,
        1,
        1,
        16,
        D.GQA_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_v = torch.zeros_like(kv_k)
    token = torch.tensor([100], dtype=torch.long, device=device)
    position = torch.tensor(5, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(token, position, dn, conv, kv_k, kv_v)
        torch.neuron.synchronize()
    names = ("logits", "deltanet", "conv", "kv_k", "kv_v")
    captured = {
        name: tensor.cpu() for name, tensor in zip(names, outputs)
    }
    os.makedirs(output_dir, exist_ok=True)
    torch.save(captured, os.path.join(output_dir, f"rank{rank}.pt"))
    dist.barrier()
    if rank == 0:
        top = torch.topk(captured["logits"][0].float(), 5).indices
        print(f"captured top5={[int(value) for value in top]}")
    if bench_iters:
        with torch.no_grad():
            for _ in range(3):
                outputs = model(token, position, *outputs[1:])
            torch.neuron.synchronize()
            started = time.perf_counter()
            for _ in range(bench_iters):
                outputs = model(token, position, *outputs[1:])
            torch.neuron.synchronize()
            elapsed = time.perf_counter() - started
        dist.barrier()
        if rank == 0:
            latency_ms = elapsed * 1000 / bench_iters
            print(
                f"four-layer latency={latency_ms:.3f} ms "
                f"({bench_iters} synchronized iterations)"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", default="/models/Qwen3.5-35B-A3B"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--bench-iters", type=int, default=0)
    parser.add_argument("--compare", nargs=2, metavar=("REFERENCE", "CANDIDATE"))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if not args.output_dir:
        parser.error("--output-dir is required for device capture")
    capture(args.model_path, args.output_dir, args.bench_iters)


if __name__ == "__main__":
    main()
