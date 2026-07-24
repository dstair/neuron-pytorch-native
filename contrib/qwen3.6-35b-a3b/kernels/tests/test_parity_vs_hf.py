#!/usr/bin/env python3
"""
End-to-end parity: our static_decode_35b harness vs HF Qwen3_5Moe reference.

Runs on CPU (single process, world_size=1) against the tiny-random debug model
so a full HF forward is cheap. Requires transformers with the Qwen3_5Moe class
(5.5.0+) — run inside the Native DLC on the box, or any env with that build.

Checks:
  1. Prefill last-token logits: our prefill vs HF forward over the same prompt.
  2. Greedy continuation: our decode loop vs HF greedy — token-id match.

Usage:
    python3 kernels/tests/test_parity_vs_hf.py \
        --model-path /models/tiny-qwen36-moe [--prompt-len 8] [--gen 6]
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))           # examples/Qwen3.6-35B-A3B
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "..", "qwen3_6"))  # chunked_prefill

import model_dims as D
import static_decode_35b as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/models/tiny-qwen36-moe")
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--gen", type=int, default=6)
    ap.add_argument("--max-seq-len", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg_path = os.path.join(args.model_path, "config.json")
    D.load_from_config(cfg_path)
    device = torch.device("cpu")
    ws = 1

    # ── our harness (world_size=1, CPU) ──
    weights = S.load_sharded_weights(args.model_path, 0, ws, num_layers=D.NUM_LAYERS)
    mod = S.StaticDecode35B(weights, args.max_seq_len, ws, batch_size=1).to(device).eval()
    del weights

    # ── HF reference ──
    # Import the qwen3_5_moe classes directly (the local transformers checkout
    # may not register them in the Auto* mapping). Build the text model.
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeConfig)
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration)
    hf_cfg = Qwen3_5MoeConfig.from_pretrained(args.model_path)
    hf = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model_path, config=hf_cfg, torch_dtype=torch.float32).eval()

    prompt = torch.randint(1, D.VOCAB - 1, (args.prompt_len,), generator=torch.manual_seed(1))
    ids = prompt.tolist()

    # HF greedy reference
    hf_ids = list(ids)
    with torch.no_grad():
        for _ in range(args.gen):
            out = hf(torch.tensor([hf_ids]))
            nxt = int(out.logits[0, -1].argmax())
            hf_ids.append(nxt)
    hf_gen = hf_ids[args.prompt_len:]

    # Our harness: prefill + decode
    KD, VD = D.DN_K_DIM, D.DN_V_DIM
    td = D.tp_dims(ws); vh = td["dn_v_heads"]
    qkv_dim = 2 * td["dn_k_heads"] * KD + vh * VD
    nkv = max(1, D.GQA_KV_HEADS // ws)
    dt = torch.float32
    dn = torch.zeros(D.NUM_DELTANET, 1, vh * KD, VD, dtype=dt)
    cv = torch.zeros(D.NUM_DELTANET, 1, qkv_dim, D.DN_CONV_KERNEL - 1, dtype=dt)
    kvk = torch.zeros(D.NUM_GQA, 1, nkv, args.max_seq_len, D.GQA_HEAD_DIM, dtype=dt)
    kvv = torch.zeros(D.NUM_GQA, 1, nkv, args.max_seq_len, D.GQA_HEAD_DIM, dtype=dt)

    with torch.no_grad():
        in_t = torch.tensor(ids, dtype=torch.long)
        logits, dn, cv, kvk, kvv = mod.prefill(in_t, dn, cv, kvk, kvv)
        # parity on prefill last-token logits
        hf_pref = hf(torch.tensor([ids])).logits[0, -1].float()
        our_pref = logits[0].float()
        ld = (our_pref - hf_pref).abs().max().item()
        top_match = int(our_pref.argmax()) == int(hf_pref.argmax())
        print(f"prefill logits max_abs_diff = {ld:.3e}  top1_match={top_match}")

        our_gen = []
        nid = our_pref.argmax().reshape(1).to(torch.long)
        our_gen.append(int(nid))
        pos = torch.tensor(len(ids), dtype=torch.long)
        one = torch.tensor(1, dtype=torch.long)
        for _ in range(args.gen - 1):
            logits, dn, cv, kvk, kvv = mod(nid, pos, dn, cv, kvk, kvv)
            nid = logits.argmax(-1).to(torch.long)
            our_gen.append(int(nid))
            pos = pos + one

    print(f"HF  greedy: {hf_gen}")
    print(f"our greedy: {our_gen}")
    match = sum(a == b for a, b in zip(hf_gen, our_gen))
    print(f"token match: {match}/{len(hf_gen)}")
    ok = (match == len(hf_gen))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
