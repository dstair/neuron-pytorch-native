#!/usr/bin/env python3
"""Per-layer prefill divergence: pure-torch prefill vs kernel prefill, SAME real
weights, layer by layer. Pinpoints which layer + which kernel (flash-GQA vs
DeltaNet-chunked) first diverges from the coherent pure-torch baseline.

Runs the prefill layer loop MANUALLY (mirrors StaticDecode35B.prefill) twice —
once with USE_GQA_FLASH_PREFILL/USE_DN_CHUNK_NKI = False (pure torch), once True —
recording hidden after each layer. Prints cos/maxdiff per layer for the FIRST
divergence. No bucketing here (isolate kernels from bucketing).

Run: torchrun --nproc-per-node=4 kernels/tests/test_prefill_layerdiff.py --model-path ... --seq 128
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "kernels"))
# import both kernel ops so torch.ops.* are registered regardless of flag state
os.environ.setdefault("GQA_FLASH_PREFILL", "1")
os.environ.setdefault("DN_CHUNK_NKI", "1")
import static_decode_35b as S
import model_dims as D


def run_prefill_capture(mod, ids, S_len, use_flash, use_dnchunk):
    """Manual copy of StaticDecode35B.prefill that captures hidden after each
    layer. Toggles the kernel-path module globals for this run."""
    S.USE_GQA_FLASH_PREFILL = use_flash
    S.USE_DN_CHUNK_NKI = use_dnchunk
    td = D.tp_dims(mod.world_size); vh = td["dn_v_heads"]; KD, VD = D.DN_K_DIM, D.DN_V_DIM
    qkv_dim = 2 * td["dn_k_heads"] * KD + vh * VD
    nkv = mod.nkv
    dev = ids.device
    dn = torch.zeros(D.NUM_DELTANET, 1, vh * KD, VD, device=dev)
    cv = torch.zeros(D.NUM_DELTANET, 1, qkv_dim, D.DN_CONV_KERNEL - 1, device=dev)
    kk = torch.zeros(D.NUM_GQA, 1, nkv, mod.max_seq_len, D.GQA_HEAD_DIM, device=dev)
    vv = torch.zeros(D.NUM_GQA, 1, nkv, mod.max_seq_len, D.GQA_HEAD_DIM, device=dev)

    hidden = F.embedding(ids, mod.embed).unsqueeze(0).float()
    caps = []
    for i in range(D.NUM_LAYERS):
        normed = S.rms_norm(hidden, getattr(mod, f"l{i}_input_norm"))
        if D.layer_type(i) == "deltanet":
            hidden = hidden + mod._deltanet_prefill(i, normed, dn, cv)
        else:
            hidden = hidden + mod._gqa_prefill(i, normed, S_len, kk, vv)
        normed = S.rms_norm(hidden, getattr(mod, f"l{i}_post_norm"))
        hidden = hidden + mod._moe(i, normed)
        caps.append(hidden[0].float().cpu().clone())   # [S,H] after layer i
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/models/Qwen3.5-35B-A3B")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--num-layers", type=int, default=6)
    args = ap.parse_args()

    import torch_neuronx  # noqa: F401
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank(); ws = dist.get_world_size()
    device = torch.neuron.current_device()

    D.load_from_config(os.path.join(args.model_path, "config.json"))
    D.NUM_LAYERS = args.num_layers
    D.NUM_GQA = sum(1 for i in range(D.NUM_LAYERS) if D.layer_type(i) == "gqa")
    D.NUM_DELTANET = D.NUM_LAYERS - D.NUM_GQA
    weights = S.load_sharded_weights(args.model_path, rank, ws, num_layers=D.NUM_LAYERS)
    mod = S.StaticDecode35B(weights, args.max_seq_len, ws, batch_size=1, rank=rank).to(device).eval()

    ids = torch.tensor([760, 6511, 314, 9338, 369], dtype=torch.long, device=device)
    ids = ids.repeat((args.seq + 4) // 5)[:args.seq]   # tile prompt to seq len

    base = run_prefill_capture(mod, ids, args.seq, use_flash=False, use_dnchunk=False)
    kern = run_prefill_capture(mod, ids, args.seq, use_flash=True, use_dnchunk=True)

    if rank == 0:
        print(f"[layerdiff] seq={args.seq} — pure-torch vs kernels, per layer:")
        print(f"{'layer':>5} {'type':>9} {'cos':>10} {'maxdiff':>12} {'reldiff':>10}")
        for i, (a, b) in enumerate(zip(base, kern)):
            cos = F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()
            md = (a - b).abs().max().item()
            rel = md / (a.abs().max().item() + 1e-9)
            flag = "  <== FIRST DIVERGE" if (cos < 0.999 and all(
                F.cosine_similarity(base[j].reshape(-1), kern[j].reshape(-1), dim=0).item() >= 0.999
                for j in range(i))) else ""
            print(f"{i:>5} {D.layer_type(i):>9} {cos:>10.6f} {md:>12.4e} {rel:>10.4e}{flag}")


if __name__ == "__main__":
    main()
