#!/usr/bin/env python3
"""Direct isolation: _gqa_prefill (eager, flash_prefill) vs _gqa_prefill_chunk
(bucketed, flash_prefill_chunk over scattered KV buffer) for ONE GQA layer, SAME
input. Removes DeltaNet/MoE/multi-layer confounders. Localizes the bucketed-vs-eager
GQA coherence gap (cos 0.39 at chunk=S).

Run in DLC (torchrun):
  torchrun --nproc-per-node=4 kernels/tests/test_gqa_chunk_iso.py --model-path /models/Qwen3.5-35B-A3B
"""
import os
import sys
import argparse
import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "kernels"))
os.environ.setdefault("GQA_FLASH_PREFILL", "1")
import static_decode_35b as S
import model_dims as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/models/Qwen3.5-35B-A3B")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args()

    import torch_neuronx  # noqa: F401
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank(); ws = dist.get_world_size()
    device = torch.neuron.current_device()

    D.load_from_config(os.path.join(args.model_path, "config.json"))
    # keep only enough layers that layer index 3 (first GQA) exists
    D.NUM_LAYERS = 4
    D.NUM_GQA = sum(1 for i in range(D.NUM_LAYERS) if D.layer_type(i) == "gqa")
    D.NUM_DELTANET = D.NUM_LAYERS - D.NUM_GQA
    weights = S.load_sharded_weights(args.model_path, rank, ws, num_layers=D.NUM_LAYERS)
    mod = S.StaticDecode35B(weights, args.max_seq_len, ws, batch_size=1, rank=rank).to(device).eval()

    gi_layer = 3  # first GQA layer
    Sq = args.seq
    H = D.HIDDEN
    nkv = max(1, D.GQA_KV_HEADS // ws)

    torch.manual_seed(0)
    x = (torch.randn(1, Sq, H, device=device) * 0.2).float()

    # _gqa_prefill_chunk with PURE-TORCH attention (GQA_FLASH_PREFILL off) = reference
    S.USE_GQA_FLASH_PREFILL = False
    kk1 = torch.zeros(D.NUM_GQA, 1, nkv, args.max_seq_len, D.GQA_HEAD_DIM, device=device)
    vv1 = torch.zeros(D.NUM_GQA, 1, nkv, args.max_seq_len, D.GQA_HEAD_DIM, device=device)
    out_ref = mod._gqa_prefill_chunk(gi_layer, x, 0, Sq, kk1, vv1)      # [1,Sq,H]

    # _gqa_prefill_chunk with FLASH kernel — SAME inputs, q_base=0, chunk=Sq.
    S.USE_GQA_FLASH_PREFILL = True
    kk2 = torch.zeros(D.NUM_GQA, 1, nkv, args.max_seq_len, D.GQA_HEAD_DIM, device=device)
    vv2 = torch.zeros(D.NUM_GQA, 1, nkv, args.max_seq_len, D.GQA_HEAD_DIM, device=device)
    out_flash = mod._gqa_prefill_chunk(gi_layer, x, 0, Sq, kk2, vv2)  # [1,Sq,H]

    a = out_ref[0].float().cpu(); b = out_flash[0].float().cpu()
    if rank == 0:
        cos = torch.nn.functional.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()
        maxd = (a - b).abs().max().item()
        kd = (kk1.float().cpu() - kk2.float().cpu()).abs().max().item()
        print(f"[gqa-iso] flash vs pure-torch, seq={Sq}: out_cos={cos:.6f} out_maxdiff={maxd:.4e} "
              f"kvbuf_maxdiff={kd:.4e} {'PASS' if cos > 0.9999 else 'FAIL'}", flush=True)
        # per-position cos: does flash diverge only at certain rows (e.g. row 0, tail)?
        pc = torch.nn.functional.cosine_similarity(a, b, dim=1)   # [Sq]
        worst = pc.argmin().item()
        print(f"  per-pos cos: row0={pc[0]:.6f} rowmid={pc[Sq//2]:.6f} rowlast={pc[-1]:.6f} "
              f"WORST row{worst}={pc[worst]:.6f}", flush=True)


if __name__ == "__main__":
    main()
