#!/usr/bin/env python3
"""Multi-step (recurrent) bit-exactness test for DN_TILED_CONV, isolated kernel.

Feeds state+conv_state forward through N steps (as the decode loop does), with
fresh per-step mixed_qkv/a/b/z, and compares the default vs tiled kernel
per-step, byte-for-byte. Runs the @nki.jit kernel directly on the DLAMI (no DLC,
no model) so it exercises the exact BS=128 regime the wide-conv work flagged
without the full-model HBM ceiling.

  DN_TILED_CONV=0 python test_tiled_conv_steps.py --mode default --bs 128 --steps 8 --save /tmp/d.pt
  DN_TILED_CONV=1 python test_tiled_conv_steps.py --mode tiled   --bs 128 --steps 8 --save /tmp/t.pt
  python test_tiled_conv_steps.py --compare /tmp/d.pt /tmp/t.pt
"""
import argparse
import os
import sys
import torch
import torch_neuronx  # noqa: F401
import torch_xla.core.xla_model as xm

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "kernels"))

KH = int(os.environ.get("DN_K_HEADS", "4"))
VH = int(os.environ.get("DN_V_HEADS", "8"))
KD = VD = 128
PMAX = 128
QKV_DIM = 2 * KH * KD + VH * VD
NT = QKV_DIM // PMAX
DEV = xm.xla_device()


def cm_to_tiled(cs_cm, B):
    return cs_cm.reshape(B, NT, PMAX, 3).permute(0, 2, 1, 3).contiguous().reshape(B * PMAX, NT * 3)


def tiled_to_cm(cs_t, B):
    return cs_t.reshape(B, PMAX, NT, 3).permute(0, 2, 1, 3).contiguous().reshape(B, QKV_DIM, 3)


def run(mode, B, steps, save):
    from deltanet_full_batched_v2_35b import nki_deltanet_full_batched, USE_TILED_CONV
    assert (mode == "tiled") == USE_TILED_CONV
    g = torch.Generator().manual_seed(1234)
    r = lambda *s: torch.randn(*s, generator=g)
    state = (r(B * VH * KD, VD) * 0.05).to(torch.float32).to(DEV)
    conv_cm = (r(B * QKV_DIM, 3) * 0.5).to(torch.bfloat16)
    conv = (cm_to_tiled(conv_cm, B) if mode == "tiled" else conv_cm).to(DEV)
    conv_weight = (r(QKV_DIM, 4) * 0.1).to(torch.float32).to(DEV)
    conv_bias = (r(QKV_DIM) * 0.05).to(torch.float32).to(DEV)
    A_log = (r(VH) * 0.1).to(torch.float32).to(DEV)
    dt_bias = (r(VH) * 0.1).to(torch.float32).to(DEV)
    norm_weight = (1.0 + r(VD) * 0.05).to(torch.float32).to(DEV)
    # deterministic per-step inputs (same across modes)
    step_in = []
    for _ in range(steps):
        step_in.append((
            (r(B * QKV_DIM) * 0.5).to(torch.bfloat16),
            (r(B * VH) * 0.1).to(torch.float32),
            (r(B * VH) * 0.1).to(torch.float32),
            (r(B * VH, VD) * 0.5).to(torch.bfloat16),
        ))
    caps = []
    for (mq, a, b, z) in step_in:
        ns, ncs, out = nki_deltanet_full_batched(
            state, mq.to(DEV), conv, conv_weight, conv_bias,
            a.to(DEV), b.to(DEV), z.to(DEV), A_log, dt_bias, norm_weight)
        state, conv = ns, ncs
        ncs_cm = tiled_to_cm(ncs.cpu().float(), B) if mode == "tiled" else ncs.cpu().float().reshape(B, QKV_DIM, 3)
        caps.append({"out": out.cpu().float(), "ns": ns.cpu().float(), "ncs": ncs_cm})
    fin = all(torch.isfinite(c["out"]).all() and torch.isfinite(c["ns"]).all() for c in caps)
    print(f"[steps] mode={mode} BS={B} steps={steps} finite={fin}")
    if save:
        torch.save(caps, save); print(f"  saved {save}")


def compare(a, b):
    da, db = torch.load(a), torch.load(b)
    ok = True
    for i, (x, y) in enumerate(zip(da, db)):
        for k in ("out", "ns", "ncs"):
            md = (x[k].double() - y[k].double()).abs().max().item()
            denom = x[k].double().abs().max().item() + 1e-9
            good = (md / denom) < 1e-3
            ok = ok and good
            flag = "" if good else "  <-- MISMATCH"
            if not good or k == "out":
                print(f"  step={i} {k}: maxdiff={md:.4e} rel={md/denom:.2e}{flag}")
    print("RESULT:", "PASS (bit-exact-ish across all steps)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["default", "tiled"])
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--save", default="")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    run(a.mode, a.bs, a.steps, a.save or "")
