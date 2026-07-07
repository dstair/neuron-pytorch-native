#!/usr/bin/env python3
"""On-device coherence: bucketed prefill == eager flash prefill (same last-token
logits), at small S where BOTH fit/compile. Validates the full chunked orchestration
(state carry, dynamic-offset KV writes, rope positions, chunk kernel stitching)
against the already-validated single-shot flash prefill path.

Runs the real 40-layer model (few layers via --num-layers for speed) on device,
TP=world_size. Compares argmax + top-5 of the final-token logits.

Run in DLC (torchrun):
  torchrun --nproc-per-node=4 kernels/tests/test_bucketed_prefill_device.py \
      --model-path /models/Qwen3.5-35B-A3B --num-layers 8 --seq 1024 --chunk 512
"""
import os
import sys
import argparse
import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))     # examples/Qwen3.6-35B-A3B
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "kernels"))

os.environ.setdefault("GQA_FLASH_PREFILL", "1")
import static_decode_35b as S
import model_dims as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/models/Qwen3.5-35B-A3B")
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=2048)
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

    td = D.tp_dims(ws); vh = td["dn_v_heads"]; KD, VD = D.DN_K_DIM, D.DN_V_DIM
    qkv_dim = 2 * td["dn_k_heads"] * KD + vh * VD
    nkv = max(1, D.GQA_KV_HEADS // ws)

    def fresh_state():
        dn = torch.zeros(D.NUM_DELTANET, 1, vh * KD, VD, device=device)
        cv = torch.zeros(D.NUM_DELTANET, 1, qkv_dim, D.DN_CONV_KERNEL - 1, device=device)
        kk = torch.zeros(D.NUM_GQA, 1, nkv, args.max_seq_len, D.GQA_HEAD_DIM, device=device)
        vv = torch.zeros(D.NUM_GQA, 1, nkv, args.max_seq_len, D.GQA_HEAD_DIM, device=device)
        return dn, cv, kk, vv

    do_eager = os.environ.get("BUCKET_TEST_EAGER", "1") == "1"
    ids = (torch.arange(args.seq, device=device) * 7 + 3) % D.VOCAB

    compile_chunk = os.environ.get("BUCKET_COMPILE", "1") == "1"
    dn, cv, kk, vv = fresh_state()
    log_buck, *_ = mod.prefill_bucketed(ids, dn, cv, kk, vv, chunk=args.chunk,
                                        compile_chunk=compile_chunk)
    lb = log_buck[0].float().cpu()
    if rank == 0:
        print(f"[bucketed] S={args.seq} chunk={args.chunk} NL={args.num_layers}: "
              f"argmax={int(lb.argmax())} finite={bool(torch.isfinite(lb).all())} "
              f"norm={lb.norm():.3e} top5={lb.topk(5).indices.tolist()}", flush=True)

    if do_eager:
        dn, cv, kk, vv = fresh_state()
        log_eager, *_ = mod.prefill(ids, dn, cv, kk, vv)
        le = log_eager[0].float().cpu()
        if rank == 0:
            ae = int(le.argmax()); ab = int(lb.argmax())
            t5e = set(le.topk(5).indices.tolist()); t5b = set(lb.topk(5).indices.tolist())
            maxd = (le - lb).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(le, lb, dim=0).item()
            ok = (ae == ab) and (t5e == t5b)
            print(f"[coherence] argmax eager={ae} bucket={ab} match={ae==ab} | "
                  f"top5match={t5e==t5b} | cos={cos:.6f} maxdiff={maxd:.3e}  "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
