#!/usr/bin/env python3
"""Isolation: _deltanet_prefill EAGER vs torch.compile'd, SAME input, ONE layer.
Localizes whether the DeltaNet chunked-prefill NKI kernel + in-place state mutation
is miscompiled under torch.compile(backend=neuron) — the suspected source of the
bucketed-vs-eager cos~0.39 catastrophe (GQA-chunk compiled was nearly exact).

Run: torchrun --nproc-per-node=4 kernels/tests/test_dn_prefill_iso.py --model-path ...
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
os.environ.setdefault("DN_CHUNK_NKI", "1")
import static_decode_35b as S
import model_dims as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/models/Qwen3.5-35B-A3B")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args()

    import torch_neuronx  # noqa: F401
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank(); ws = dist.get_world_size()
    device = torch.neuron.current_device()

    D.load_from_config(os.path.join(args.model_path, "config.json"))
    D.NUM_LAYERS = 4
    D.NUM_GQA = sum(1 for i in range(D.NUM_LAYERS) if D.layer_type(i) == "gqa")
    D.NUM_DELTANET = D.NUM_LAYERS - D.NUM_GQA
    weights = S.load_sharded_weights(args.model_path, rank, ws, num_layers=D.NUM_LAYERS)
    mod = S.StaticDecode35B(weights, args.max_seq_len, ws, batch_size=1, rank=rank).to(device).eval()

    td = D.tp_dims(ws); vh = td["dn_v_heads"]; KD, VD = D.DN_K_DIM, D.DN_V_DIM
    qkv_dim = 2 * td["dn_k_heads"] * KD + vh * VD
    Sq = args.seq; H = D.HIDDEN
    torch.manual_seed(0)
    x = (torch.randn(1, Sq, H, device=device) * 0.2).float()

    def fresh():
        dn = torch.zeros(D.NUM_DELTANET, 1, vh * KD, VD, device=device)
        cv = torch.zeros(D.NUM_DELTANET, 1, qkv_dim, D.DN_CONV_KERNEL - 1, device=device)
        return dn, cv

    i = 0  # layer 0 is DeltaNet
    dn1, cv1 = fresh()
    out_e = mod._deltanet_prefill(i, x, dn1, cv1)
    dn2, cv2 = fresh()
    fn = torch.compile(mod._deltanet_prefill, backend="neuron", fullgraph=True, dynamic=False)
    out_c = fn(i, x, dn2, cv2)

    a = out_e[0].float().cpu(); b = out_c[0].float().cpu()
    if rank == 0:
        cos = torch.nn.functional.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()
        od = (a - b).abs().max().item()
        sd = (dn1.float().cpu() - dn2.float().cpu()).abs().max().item()
        print(f"[dn-iso] seq={Sq}: out_cos={cos:.6f} out_maxdiff={od:.4e} state_maxdiff={sd:.4e} "
              f"{'PASS' if cos > 0.9999 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
