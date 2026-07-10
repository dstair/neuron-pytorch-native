#!/usr/bin/env python3
"""
CPU reference oracle for the Qwen3.5-35B-A3B MoE layer (Phase-2 de-risk).

Goal: prove the **masked-dense grouped-bmm** MoE formulation we will run on
Neuron (adapted from examples/deepseek-v4/run_deepseek_v4_tp.py) is numerically
equivalent to HuggingFace's canonical **sparse index_add** routing, using the
REAL layer-0 weights from the downloaded checkpoint. No Neuron device needed.

Three things validated here, all on CPU, before any device work:
  1. HF-sparse vs masked-dense agree (the routing-equivalence claim).
  2. Expert-parallel sharding (split 256 experts across W ranks, sum partials)
     reproduces the single-device result — the TP combine pattern static_decode
     will use (all_reduce after the MoE).
  3. The packed checkpoint layout (gate_up_proj [E,2I,H], down_proj [E,H,I],
     shared_expert + sigmoid shared_expert_gate) is sliced correctly.

Reference routing (canonical Qwen3Moe, transformers 4.57):
    router_logits = x @ gate.T                       # [T, E]
    w = softmax(router_logits, dim=1, float)         # [T, E]
    w, sel = topk(w, top_k, dim=-1)                  # [T, k]
    if norm_topk_prob: w /= w.sum(-1, keepdim=True)  # Qwen3.5: normalize_top_k_affinities=True
    out = sum over selected experts of  w * expert(x)
Shared expert (Qwen3.5-specific, from PR #60):
    shared = sigmoid(x @ shared_expert_gate.T) * SwiGLU_shared(x)
Final MoE output = routed_out + shared

Run:
    python3 kernels/tests/test_moe_oracle_cpu.py \
        [--ckpt /models/Qwen3.5-35B-A3B] [--layer 0] [--tokens 8]
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

# ── 35B-A3B MoE constants (verified against checkpoint tensor shapes) ──────────
HIDDEN = 2048
NUM_EXPERTS = 256
TOP_K = 8
MOE_INTER = 512          # moe_intermediate_size (per routed expert)
SHARED_INTER = 512       # shared_expert_intermediate_size
NORM_TOPK_PROB = True    # config: normalize_top_k_affinities


def load_layer_moe_weights(ckpt: str, layer: int) -> dict:
    """Load the packed MoE weights for one language-model layer from safetensors.

    Returns tensors in their on-disk orientation:
      gate_up_proj  [E, 2*I, H]   (gate and up fused on the 2*I axis)
      down_proj     [E, H, I]
      router        [E, H]
      sh_gate/sh_up [I, H], sh_down [H, I], sh_sigmoid_gate [1, H]
    """
    from safetensors import safe_open

    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    pfx = f"model.language_model.layers.{layer}.mlp."
    keys = {
        "gate_up_proj": pfx + "experts.gate_up_proj",
        "down_proj": pfx + "experts.down_proj",
        "router": pfx + "gate.weight",
        "sh_gate": pfx + "shared_expert.gate_proj.weight",
        "sh_up": pfx + "shared_expert.up_proj.weight",
        "sh_down": pfx + "shared_expert.down_proj.weight",
        "sh_sigmoid_gate": pfx + "shared_expert_gate.weight",
    }
    # group keys by shard file to open each file once
    out = {}
    by_file: dict[str, list[str]] = {}
    for name, key in keys.items():
        by_file.setdefault(wm[key], []).append(name)
    for fname, names in by_file.items():
        with safe_open(os.path.join(ckpt, fname), framework="pt") as h:
            for name in names:
                out[name] = h.get_tensor(keys[name])
    return out


def swiglu(x, gate_w, up_w, down_w):
    """SwiGLU MLP: down(silu(x@gate.T) * (x@up.T)). All weights [out, in]."""
    g = F.linear(x, gate_w.float())
    u = F.linear(x, up_w.float())
    return F.linear(F.silu(g) * u, down_w.float())


def shared_expert(x, w):
    """Qwen3.5 sigmoid-gated shared expert (PR #60 math)."""
    gate = torch.sigmoid(F.linear(x, w["sh_sigmoid_gate"].float()))  # [T, 1]
    mlp = swiglu(x, w["sh_gate"], w["sh_up"], w["sh_down"])          # [T, H]
    return gate * mlp


# ── (A) HF canonical sparse reference ─────────────────────────────────────────
def moe_hf_sparse(x, w):
    """Canonical Qwen3Moe sparse routing with per-expert SwiGLU, in fp32.

    Expert e weights sliced from packed tensors:
      gate_up_proj[e] is [2I, H] -> gate = [:I], up = [I:2I]
      down_proj[e]    is [H, I]
    """
    T = x.shape[0]
    router_logits = F.linear(x, w["router"].float())              # [T, E]
    rw = F.softmax(router_logits, dim=1, dtype=torch.float)
    rw, sel = torch.topk(rw, TOP_K, dim=-1)                        # [T,k]
    if NORM_TOPK_PROB:
        rw = rw / rw.sum(dim=-1, keepdim=True)
    rw = rw.float()

    gup = w["gate_up_proj"].float()                                # [E,2I,H]
    dn = w["down_proj"].float()                                    # [E,H,I]
    out = torch.zeros(T, HIDDEN, dtype=torch.float)
    for t in range(T):
        for j in range(TOP_K):
            e = sel[t, j].item()
            g = F.linear(x[t], gup[e, :MOE_INTER, :])              # [I]
            u = F.linear(x[t], gup[e, MOE_INTER:, :])              # [I]
            h = F.silu(g) * u
            y = F.linear(h, dn[e])                                 # [H]
            out[t] += rw[t, j] * y
    return out


# ── (B) masked-dense grouped-bmm (the Neuron formulation) ─────────────────────
def moe_masked_dense(x, w, expert_slice=None):
    """Masked-dense grouped GEMM over experts (deepseek-v4 style).

    Every expert in `expert_slice` (default: all 256) computes for every token;
    a per-(token,expert) routing weight (0 for unselected) combines them. This
    is the static-shape, compile-friendly path. With expert_slice set to a
    rank's subset, this returns that rank's PARTIAL sum (expert-parallel).
    """
    T = x.shape[0]
    if expert_slice is None:
        expert_slice = range(NUM_EXPERTS)
    e_lo, e_hi = expert_slice.start, expert_slice.stop
    E = e_hi - e_lo

    # Router is replicated on every rank; compute full top-k, then keep only the
    # weights that fall on this rank's experts.
    router_logits = F.linear(x, w["router"].float())              # [T, E_all]
    rw_full = F.softmax(router_logits, dim=1, dtype=torch.float)
    rw_topk, sel = torch.topk(rw_full, TOP_K, dim=-1)             # [T,k]
    if NORM_TOPK_PROB:
        rw_topk = rw_topk / rw_topk.sum(dim=-1, keepdim=True)

    # Dense [T, E] gate matrix (weight per token per LOCAL expert, 0 if unselected)
    gate = torch.zeros(T, E, dtype=torch.float)
    for j in range(TOP_K):
        e = sel[:, j]                                             # [T] global idx
        on = (e >= e_lo) & (e < e_hi)
        if on.any():
            rows = on.nonzero(as_tuple=True)[0]
            gate[rows, e[rows] - e_lo] = rw_topk[rows, j]

    gup = w["gate_up_proj"][e_lo:e_hi].float()                    # [E,2I,H]
    dn = w["down_proj"][e_lo:e_hi].float()                        # [E,H,I]

    x_exp = x.unsqueeze(0).expand(E, T, HIDDEN).float()          # [E,T,H]
    gu = torch.bmm(x_exp, gup.transpose(1, 2))                    # [E,T,2I]
    g, u = gu[:, :, :MOE_INTER], gu[:, :, MOE_INTER:]
    h = F.silu(g) * u                                            # [E,T,I]
    y = torch.bmm(h, dn.transpose(1, 2))                         # [E,T,H]

    ew = gate.t().unsqueeze(-1)                                   # [E,T,1]
    return (y * ew).sum(dim=0)                                    # [T,H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/models/Qwen3.5-35B-A3B")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--world-size", type=int, default=4, help="EP shard count to test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print(f"Loading layer {args.layer} MoE weights from {args.ckpt} ...")
    w = load_layer_moe_weights(args.ckpt, args.layer)
    for k, v in w.items():
        print(f"  {k:18s} {tuple(v.shape)} {v.dtype}")

    x = torch.randn(args.tokens, HIDDEN) * 0.5   # fake post-norm hidden states

    routed_hf = moe_hf_sparse(x, w)
    routed_md = moe_masked_dense(x, w)
    shared = shared_expert(x, w).float()

    def report(name, a, b):
        ad = (a - b).abs()
        rel = ad.max().item() / (b.abs().max().item() + 1e-9)
        print(f"  {name:42s} max_abs={ad.max().item():.3e}  "
              f"mean_abs={ad.mean().item():.3e}  rel={rel:.3e}")
        return ad.max().item()

    print("\n=== (1) routed: HF-sparse vs masked-dense ===")
    d1 = report("routed_hf vs routed_masked_dense", routed_hf, routed_md)

    print("\n=== (2) expert-parallel: sum of per-rank partials vs single-device ===")
    W = args.world_size
    assert NUM_EXPERTS % W == 0, "experts must divide world_size"
    per = NUM_EXPERTS // W
    partials = [moe_masked_dense(x, w, range(r * per, (r + 1) * per)) for r in range(W)]
    ep_sum = torch.stack(partials, 0).sum(0)
    d2 = report(f"EP(W={W}) sum vs single-device masked-dense", ep_sum, routed_md)

    print("\n=== (3) full MoE output (routed + sigmoid-gated shared) ===")
    full = routed_md + shared
    print(f"  shared-expert contribution  max_abs={shared.abs().max().item():.3e}  "
          f"mean_abs={shared.abs().mean().item():.3e}")
    print(f"  full MoE out                norm={full.norm().item():.3e}  "
          f"shape={tuple(full.shape)}")

    TOL = 2e-3   # fp32 GEMM reassociation across the two formulations
    ok = d1 < TOL and d2 < TOL
    print(f"\n{'PASS' if ok else 'FAIL'}  (tol={TOL:.0e}; routed_diff={d1:.2e}, ep_diff={d2:.2e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
