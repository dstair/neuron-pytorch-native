"""
Preprocess DeepSeek V4-Flash: dequantize FP4/FP8 -> BF16, shard for TP=64.

Reads the HF checkpoint (FP4+FP8 mixed), dequantizes everything to BF16,
shards weights for tensor parallelism, and saves per-rank safetensors files.

Usage:
    python preprocess_weights.py \
        --input-path /scratch/DeepSeek-V4-Flash \
        --output-path /scratch/DeepSeek-V4-Flash-BF16-TP64 \
        --tp-degree 64
"""

import argparse
import gc
import json
import os
import time

import torch
from safetensors.torch import load_file, save_file

# FP4 E2M1 lookup table: 4-bit index -> float value
FP4_E2M1_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

FP4_BLOCK_SIZE = 32
FP8_BLOCK_SIZE = 128


def dequant_fp4(w, s, out_f, in_f):
    """Dequantize FP4 (packed int8) to BF16."""
    raw = w.view(torch.uint8)
    low = (raw & 0x0F).long()
    high = ((raw >> 4) & 0x0F).long()
    vals = torch.stack([FP4_E2M1_LUT[low], FP4_E2M1_LUT[high]], dim=-1)
    unpacked = vals.reshape(out_f, in_f)
    scales = s.to(torch.float32).repeat_interleave(FP4_BLOCK_SIZE, dim=1)[:, :in_f]
    return (unpacked * scales).to(torch.bfloat16)


def dequant_fp8(w, s):
    """Dequantize FP8 (e4m3fn) with block-wise e8m0 scales to BF16."""
    M, N = w.shape
    scales = s.to(torch.float32)
    scales = scales.repeat_interleave(FP8_BLOCK_SIZE, dim=0)
    scales = scales.repeat_interleave(FP8_BLOCK_SIZE, dim=1)
    return (w.to(torch.float32) * scales[:M, :N]).to(torch.bfloat16)


def dequant_weight(sd, key):
    """Dequantize a weight tensor in-place, removing its scale key."""
    w = sd[key]
    sk = key.replace(".weight", ".scale")
    s = sd.get(sk)

    if s is None or w.dtype in (torch.bfloat16, torch.float32, torch.float16):
        return w.to(torch.bfloat16) if w.is_floating_point() else w

    if w.dtype in (torch.int8, torch.uint8):
        # FP4 packed as int8
        out_f = w.shape[0]
        in_f = w.shape[1] * 2
        result = dequant_fp4(w, s, out_f, in_f)
    elif w.dtype == torch.float8_e4m3fn:
        result = dequant_fp8(w, s)
    elif hasattr(torch, 'float4_e2m1fn_x2') and w.dtype == torch.float4_e2m1fn_x2:
        out_f = w.shape[0]
        in_f = w.shape[1] * 2
        result = dequant_fp4(w, s, out_f, in_f)
    else:
        result = w.to(torch.bfloat16)

    sd.pop(sk, None)
    return result


def shard_for_tp(key, tensor, rank, tp, config):
    """Shard a tensor for the given TP rank. Returns (new_key, shard) or None to skip."""
    n_heads = config["num_attention_heads"]
    n_experts = config["n_routed_experts"]
    o_groups = config["o_groups"]
    o_lora_rank = config["o_lora_rank"]
    head_dim = config["head_dim"]

    # Expert weights: partition by expert index
    if ".experts." in key and ".shared_experts." not in key:
        parts = key.split(".")
        for i, p in enumerate(parts):
            if p == "experts" and i + 1 < len(parts) and parts[i + 1].isdigit():
                global_idx = int(parts[i + 1])
                experts_per_rank = n_experts // tp
                if global_idx // experts_per_rank != rank:
                    return None  # not this rank's expert
                local_idx = global_idx % experts_per_rank
                parts[i + 1] = str(local_idx)
                return ".".join(parts), tensor
        return key, tensor

    # Q projection output: shard heads (ColwiseParallel)
    if key.endswith(".wq_b.weight"):
        # weight shape: (n_heads * head_dim, q_lora_rank) — shard dim 0
        total = tensor.shape[0]
        chunk = total // tp
        return key, tensor[rank * chunk : (rank + 1) * chunk]

    # Output projection wo_a: replicate (grouped structure doesn't split cleanly when TP > o_groups)
    if key.endswith(".wo_a.weight"):
        return key, tensor

    # Output projection wo_b: shard input dim (RowwiseParallel)
    if key.endswith(".wo_b.weight"):
        # weight shape: (dim, o_groups * o_lora_rank) — shard dim 1
        total = tensor.shape[1]
        chunk = total // tp
        return key, tensor[:, rank * chunk : (rank + 1) * chunk]

    # Everything else: replicate
    return key, tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--tp-degree", type=int, default=64)
    args = parser.parse_args()

    tp = args.tp_degree
    os.makedirs(args.output_path, exist_ok=True)

    with open(os.path.join(args.input_path, "config.json")) as f:
        config = json.load(f)

    # Copy config
    with open(os.path.join(args.output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load safetensors index
    index_path = os.path.join(args.input_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))

    print(f"Config: {config['num_hidden_layers']} layers, {config['num_attention_heads']} heads, "
          f"{config['n_routed_experts']} experts, TP={tp}")
    print(f"Shards to process: {len(shard_files)}")

    # Initialize per-rank accumulators
    rank_tensors = [{} for _ in range(tp)]
    total_keys = 0
    t0 = time.time()

    for si, shard_file in enumerate(shard_files):
        st = time.time()
        sd = load_file(os.path.join(args.input_path, shard_file))
        print(f"[{si+1}/{len(shard_files)}] {shard_file}: {len(sd)} keys, "
              f"loaded in {time.time()-st:.1f}s")

        # Dequantize all weights in this shard
        keys = list(sd.keys())
        for key in keys:
            if key.endswith(".scale"):
                continue  # handled by dequant_weight
            if key.endswith(".weight") or key in sd:
                tensor = dequant_weight(sd, key) if key.endswith(".weight") else sd[key]
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                    tensor = tensor.to(torch.bfloat16)

                # Shard for each TP rank
                for rank in range(tp):
                    result = shard_for_tp(key, tensor, rank, tp, config)
                    if result is not None:
                        new_key, shard = result
                        rank_tensors[rank][new_key] = shard.clone()

                total_keys += 1
                sd.pop(key, None)

        # Also handle any remaining non-.weight keys (biases, norms, etc.)
        for key in list(sd.keys()):
            if key.endswith(".scale"):
                continue
            tensor = sd[key]
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                tensor = tensor.to(torch.bfloat16)
            for rank in range(tp):
                result = shard_for_tp(key, tensor, rank, tp, config)
                if result is not None:
                    new_key, shard = result
                    rank_tensors[rank][new_key] = shard.clone()
            total_keys += 1

        del sd
        gc.collect()

    print(f"\nDequantized {total_keys} keys in {time.time()-t0:.0f}s")

    # Save per-rank shards
    print(f"Saving {tp} rank shards...")
    for rank in range(tp):
        out_file = os.path.join(args.output_path, f"rank_{rank:03d}.safetensors")
        save_file(rank_tensors[rank], out_file)
        size_mb = os.path.getsize(out_file) / 1024 / 1024
        if rank % 8 == 0 or rank == tp - 1:
            print(f"  rank {rank}: {len(rank_tensors[rank])} keys, {size_mb:.0f} MB")
        rank_tensors[rank] = {}  # free memory
        gc.collect()

    print(f"\nDone. Output: {args.output_path}")
    print(f"Upload to S3: s5cmd sync {args.output_path}/ s3://${S3_MODEL_BUCKET}/deepseek-v4/bf16-tp64/")


if __name__ == "__main__":
    main()
