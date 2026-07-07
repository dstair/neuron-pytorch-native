"""
Preprocess DeepSeek V4-Flash: dequantize FP4/FP8 -> BF16, shard for TP=4/EP=16.

Attention weights sharded by TP=4 (within device).
Expert weights sharded by EP=16 (across devices).
Other weights replicated.

Each rank file is named rank_TTT_EEE.safetensors (tp_rank, ep_rank).

Usage:
    python preprocess_weights_hybrid.py \
        --input-path /scratch/DeepSeek-V4-Flash \
        --output-path /scratch/DeepSeek-V4-Flash-BF16-TP4EP16
"""

import argparse
import gc
import json
import os
import time

import torch
from safetensors.torch import load_file, save_file

# FP4 E2M1 lookup table
FP4_E2M1_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)
FP4_BLOCK_SIZE = 32
FP8_BLOCK_SIZE = 128


def dequant_fp4(w, s, out_f, in_f):
    raw = w.view(torch.uint8)
    low = (raw & 0x0F).long()
    high = ((raw >> 4) & 0x0F).long()
    vals = torch.stack([FP4_E2M1_LUT[low], FP4_E2M1_LUT[high]], dim=-1)
    unpacked = vals.reshape(out_f, in_f)
    scales = s.to(torch.float32).repeat_interleave(FP4_BLOCK_SIZE, dim=1)[:, :in_f]
    return (unpacked * scales).to(torch.bfloat16)


def dequant_fp8(w, s):
    M, N = w.shape
    scales = s.to(torch.float32)
    scales = scales.repeat_interleave(FP8_BLOCK_SIZE, dim=0)
    scales = scales.repeat_interleave(FP8_BLOCK_SIZE, dim=1)
    return (w.to(torch.float32) * scales[:M, :N]).to(torch.bfloat16)


def dequant_weight(sd, key):
    w = sd[key]
    sk = key.replace(".weight", ".scale")
    s = sd.get(sk)
    if s is None or w.dtype in (torch.bfloat16, torch.float32, torch.float16):
        return w.to(torch.bfloat16) if w.is_floating_point() else w
    if w.dtype in (torch.int8, torch.uint8):
        result = dequant_fp4(w, s, w.shape[0], w.shape[1] * 2)
    elif w.dtype == torch.float8_e4m3fn:
        result = dequant_fp8(w, s)
    elif hasattr(torch, 'float4_e2m1fn_x2') and w.dtype == torch.float4_e2m1fn_x2:
        result = dequant_fp4(w, s, w.shape[0], w.shape[1] * 2)
    else:
        result = w.to(torch.bfloat16)
    sd.pop(sk, None)
    return result


TP_SIZE = 4
EP_SIZE = 16
WORLD_SIZE = TP_SIZE * EP_SIZE  # 64


def shard_hybrid(key, tensor, tp_rank, ep_rank, config):
    """Shard for TP=4/EP=16 hybrid. Returns (new_key, shard) or None to skip."""
    n_heads = config["num_attention_heads"]
    n_experts = config["n_routed_experts"]
    o_groups = config["o_groups"]
    o_lora_rank = config["o_lora_rank"]

    # Expert weights: partition by EP rank (16 experts per device)
    if ".experts." in key and ".shared_experts." not in key:
        parts = key.split(".")
        for i, p in enumerate(parts):
            if p == "experts" and i + 1 < len(parts) and parts[i + 1].isdigit():
                global_idx = int(parts[i + 1])
                experts_per_ep = n_experts // EP_SIZE  # 256/16 = 16
                if global_idx // experts_per_ep != ep_rank:
                    return None
                local_idx = global_idx % experts_per_ep
                parts[i + 1] = str(local_idx)
                return ".".join(parts), tensor
        return key, tensor

    # Q projection: shard heads by TP=4 (64 heads / 4 = 16 heads per rank)
    if key.endswith(".wq_b.weight"):
        total = tensor.shape[0]
        chunk = total // TP_SIZE
        return key, tensor[tp_rank * chunk : (tp_rank + 1) * chunk]

    # wo_a: replicate (same as before — grouped structure)
    if key.endswith(".wo_a.weight"):
        return key, tensor

    # wo_b: shard input dim by TP=4
    if key.endswith(".wo_b.weight"):
        total = tensor.shape[1]
        chunk = total // TP_SIZE
        return key, tensor[:, tp_rank * chunk : (tp_rank + 1) * chunk].contiguous()

    # Everything else: replicate
    return key, tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    with open(os.path.join(args.input_path, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(args.output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    index_path = os.path.join(args.input_path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))

    print(f"TP={TP_SIZE}, EP={EP_SIZE}, world={WORLD_SIZE}")
    print(f"Shards: {len(shard_files)}, processing by EP rank (16 passes)")
    t0 = time.time()

    # Process by EP rank: 16 passes, each producing 4 rank files (one per TP rank)
    for ep_r in range(EP_SIZE):
        rank_tensors = [{} for _ in range(TP_SIZE)]

        for si, shard_file in enumerate(shard_files):
            sd = load_file(os.path.join(args.input_path, shard_file))
            if ep_r == 0:
                print(f"[{si+1}/{len(shard_files)}] {shard_file}: {len(sd)} keys")

            for key in list(sd.keys()):
                if key.endswith(".scale"):
                    continue
                tensor = dequant_weight(sd, key) if key.endswith(".weight") else sd[key]
                if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                    tensor = tensor.to(torch.bfloat16)
                for tp_r in range(TP_SIZE):
                    result = shard_hybrid(key, tensor, tp_r, ep_r, config)
                    if result is not None:
                        new_key, shard = result
                        rank_tensors[tp_r][new_key] = shard if tp_r == 0 else shard.clone()
                sd.pop(key, None)

            del sd
            gc.collect()

        for tp_r in range(TP_SIZE):
            global_rank = ep_r * TP_SIZE + tp_r
            out_file = os.path.join(args.output_path, f"rank_{global_rank:03d}.safetensors")
            save_file(rank_tensors[tp_r], out_file)
            size_mb = os.path.getsize(out_file) / 1024 / 1024
            if global_rank % 8 == 0 or global_rank == WORLD_SIZE - 1:
                print(f"  rank {global_rank} (tp={tp_r},ep={ep_r}): "
                      f"{len(rank_tensors[tp_r])} keys, {size_mb:.0f} MB")
            rank_tensors[tp_r] = {}
        del rank_tensors
        gc.collect()
        print(f"  EP rank {ep_r} done ({time.time()-t0:.0f}s)")

    print(f"\nDone in {time.time()-t0:.0f}s. Output: {args.output_path}")


if __name__ == "__main__":
    main()
