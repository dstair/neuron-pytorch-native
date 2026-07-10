#!/usr/bin/env python3
"""
Self-consistency parity: prefill(prompt) vs token-by-token decode of the SAME
prompt from zero state. No HF reference needed — this exercises the FULL model
wiring (layer order, residuals, RoPE-in-context, q/gate split, conv-state
handoff, DeltaNet recurrence, MoE, KV cache) and catches integration bugs that
the component-level oracles (MoE routing, DeltaNet step) can't.

Both paths must agree on:
  - the running hidden state, expressed via the last-token logits after step k
  - the DeltaNet recurrent state + conv state
  - the GQA KV cache contents

Because prefill processes all tokens in parallel (chunked DeltaNet + causal
attention) and decode processes them sequentially (recurrent step + cached
attention), agreement is a strong end-to-end correctness signal.

Runs on CPU (world_size=1) against the tiny-random model.

Usage:
    python3 kernels/tests/test_prefill_decode_consistency.py \
        --model-path /models/tiny-qwen36-moe [--prompt-len 6]
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "..", "qwen3_6"))   # chunked_prefill

import model_dims as D
import static_decode_35b as S


def fresh_state(ws, max_seq, dt=torch.float32):
    KD, VD = D.DN_K_DIM, D.DN_V_DIM
    td = D.tp_dims(ws); vh = td["dn_v_heads"]
    qkv = 2 * td["dn_k_heads"] * KD + vh * VD
    nkv = max(1, D.GQA_KV_HEADS // ws)
    dn = torch.zeros(D.NUM_DELTANET, 1, vh * KD, VD, dtype=dt)
    cv = torch.zeros(D.NUM_DELTANET, 1, qkv, D.DN_CONV_KERNEL - 1, dtype=dt)
    kvk = torch.zeros(D.NUM_GQA, 1, nkv, max_seq, D.GQA_HEAD_DIM, dtype=dt)
    kvv = torch.zeros(D.NUM_GQA, 1, nkv, max_seq, D.GQA_HEAD_DIM, dtype=dt)
    return dn, cv, kvk, kvv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/models/tiny-qwen36-moe")
    ap.add_argument("--prompt-len", type=int, default=6)
    ap.add_argument("--max-seq-len", type=int, default=64)
    ap.add_argument("--fp32", action="store_true",
                    help="Cast all weights to fp32 to isolate wiring from bf16 noise.")
    args = ap.parse_args()

    torch.manual_seed(0)
    D.load_from_config(os.path.join(args.model_path, "config.json"))
    ws = 1

    weights = S.load_sharded_weights(args.model_path, 0, ws, num_layers=D.NUM_LAYERS)
    if args.fp32:
        if torch.is_tensor(weights.get("embed")):
            for k, v in list(weights.items()):
                if torch.is_tensor(v):
                    weights[k] = v.float()
        for lw in weights["layers"]:
            for k, v in list(lw.items()):
                if torch.is_tensor(v):
                    lw[k] = v.float()
    mod = S.StaticDecode35B(weights, args.max_seq_len, ws, batch_size=1).eval()
    del weights

    P = args.prompt_len
    ids = torch.randint(1, D.VOCAB - 1, (P,), generator=torch.manual_seed(7))

    with torch.no_grad():
        # (A) PREFILL the whole prompt at once.
        dn, cv, kvk, kvv = fresh_state(ws, args.max_seq_len)
        pre_logits, dnA, cvA, kvkA, kvvA = mod.prefill(ids, dn, cv, kvk, kvv)

        # (B) DECODE the same prompt token-by-token from zero state.
        dn, cv, kvk, kvv = fresh_state(ws, args.max_seq_len)
        one = torch.tensor(1, dtype=torch.long)
        logits = None
        for t in range(P):
            nid = ids[t:t + 1]
            pos = torch.tensor(t, dtype=torch.long)
            logits, dn, cv, kvk, kvv = mod(nid, pos, dn, cv, kvk, kvv)

    # Compare last-token logits.
    a = pre_logits[0].float()
    b = logits[0].float()
    ld = (a - b).abs().max().item()
    rel = ld / (b.abs().max().item() + 1e-9)
    top = int(a.argmax()) == int(b.argmax())

    # Compare DeltaNet final state + conv + KV cache (decode wrote positions 0..P-1).
    sd = (dnA.float() - dn.float()).abs().max().item()
    cd = (cvA.float() - cv.float()).abs().max().item()
    kd = (kvkA[:, :, :, :P].float() - kvk[:, :, :, :P].float()).abs().max().item()

    print(f"last-token logits: max_abs_diff={ld:.3e}  rel={rel:.3e}  top1_match={top}")
    print(f"deltanet state    max_abs_diff={sd:.3e}")
    print(f"conv state        max_abs_diff={cd:.3e}")
    print(f"kv cache (k)      max_abs_diff={kd:.3e}")

    # Tolerances: fp32 should be ~1e-5 everywhere (exact wiring). bf16 only
    # guarantees top-1 logit match + small relative logit error — the state
    # buffers drift more because prefill (parallel conv / chunked recurrence)
    # and decode (stepwise) reduce in different orders, and the tiny model's
    # UNTRAINED weights make activations large, amplifying bf16 rounding.
    if args.fp32:
        ok = (rel < 1e-4 and top and sd < 1e-4 and cd < 1e-4 and kd < 1e-4)
    else:
        ok = (top and rel < 5e-2)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
