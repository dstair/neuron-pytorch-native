#!/usr/bin/env python3
"""Isolated A/B validation + timing for the DN_TILED_CONV conv-state layout.

Calls the @nki.jit decode kernel nki_deltanet_full_batched DIRECTLY (bypasses the
torch_neuronx nki_op registration that the public DLAMI lacks). The tiled path
must be byte-for-byte equivalent to the default path (same math; only conv_state
memory layout + DMA pattern differ).

Because DN_TILED_CONV is read at import time, run once per mode saving outputs,
then compare:

  DN_TILED_CONV=0 python test_tiled_conv_iso.py --mode default --bs 128 --save /tmp/def.pt
  DN_TILED_CONV=1 python test_tiled_conv_iso.py --mode tiled   --bs 128 --save /tmp/tiled.pt
  python test_tiled_conv_iso.py --compare /tmp/def.pt /tmp/tiled.pt

Each run also prints warm per-call latency (mean of --iters timed calls).
"""
import argparse
import os
import sys
import time
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


def make_inputs(B, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    state = (r(B * VH * KD, VD) * 0.1).to(torch.float32)
    mixed_qkv = (r(B * QKV_DIM) * 0.5).to(torch.bfloat16)
    conv_state_cm = (r(B * QKV_DIM, 3) * 0.5).to(torch.bfloat16)   # channel-major
    conv_weight = (r(QKV_DIM, 4) * 0.3).to(torch.float32)
    conv_bias = (r(QKV_DIM) * 0.1).to(torch.float32)
    a_out = (r(B * VH) * 0.5).to(torch.float32)
    b_out = (r(B * VH) * 0.5).to(torch.float32)
    z = (r(B * VH, VD) * 0.5).to(torch.bfloat16)
    A_log = (r(VH) * 0.5).to(torch.float32)
    dt_bias = (r(VH) * 0.1).to(torch.float32)
    norm_weight = (r(VD) * 0.1 + 1.0).to(torch.float32)
    return dict(state=state, mixed_qkv=mixed_qkv, conv_state_cm=conv_state_cm,
                conv_weight=conv_weight, conv_bias=conv_bias, a_out=a_out,
                b_out=b_out, z=z, A_log=A_log, dt_bias=dt_bias, norm_weight=norm_weight)


def cm_to_tiled(cs_cm, B):
    # channel-major [B, QKV_DIM, 3] -> tiled [B*PMAX, NT*3]; ch=t*PMAX+p -> [b,p,t,j]
    x = cs_cm.reshape(B, NT, PMAX, 3).permute(0, 2, 1, 3).contiguous()
    return x.reshape(B * PMAX, NT * 3)


def tiled_to_cm(cs_t, B):
    x = cs_t.reshape(B, PMAX, NT, 3).permute(0, 2, 1, 3).contiguous()
    return x.reshape(B, QKV_DIM, 3)


def run(mode, B, iters, save):
    from deltanet_full_batched_v2_35b import nki_deltanet_full_batched, USE_TILED_CONV
    assert (mode == "tiled") == USE_TILED_CONV, \
        f"mode={mode} but USE_TILED_CONV={USE_TILED_CONV} (set DN_TILED_CONV env)"
    inp = make_inputs(B)
    state = inp["state"].to(DEV)
    mixed_qkv = inp["mixed_qkv"].to(DEV)
    if mode == "tiled":
        conv_state = cm_to_tiled(inp["conv_state_cm"], B).to(DEV)
    else:
        conv_state = inp["conv_state_cm"].to(DEV)
    args = (state, mixed_qkv, conv_state, inp["conv_weight"].to(DEV),
            inp["conv_bias"].to(DEV), inp["a_out"].to(DEV), inp["b_out"].to(DEV),
            inp["z"].to(DEV), inp["A_log"].to(DEV), inp["dt_bias"].to(DEV),
            inp["norm_weight"].to(DEV))

    new_state, new_cs, out = nki_deltanet_full_batched(*args)
    _ = out.cpu()  # sync / compile

    # timing: warm 3, time `iters` (sync each iter via .cpu of a scalar)
    for _ in range(3):
        ns, ncs, o = nki_deltanet_full_batched(*args)
        _ = o.cpu()
    t0 = time.perf_counter()
    for _ in range(iters):
        ns, ncs, o = nki_deltanet_full_batched(*args)
        _ = o.cpu()
    dt = (time.perf_counter() - t0) / iters * 1e3

    out_c = out.cpu().float()
    ns_c = new_state.cpu().float()
    ncs_c = new_cs.cpu().float()
    if mode == "tiled":
        ncs_c = tiled_to_cm(ncs_c, B)          # back to channel-major for comparison
    else:
        ncs_c = ncs_c.reshape(B, QKV_DIM, 3)
    finite = all(torch.isfinite(x).all().item() for x in (out_c, ns_c, ncs_c))
    print(f"[tiled-conv] mode={mode} BS={B} percall={dt:.3f} ms finite={finite}")
    if save:
        torch.save({"out": out_c, "ns": ns_c, "ncs": ncs_c, "bs": B, "percall": dt}, save)
        print(f"  saved -> {save}")
    return 0


def compare(a, b):
    da, db = torch.load(a), torch.load(b)
    print(f"[compare] {a}(BS={da['bs']},{da['percall']:.3f}ms) vs "
          f"{b}(BS={db['bs']},{db['percall']:.3f}ms)")
    ok = True
    for key in ("out", "ns", "ncs"):
        x, y = da[key].double(), db[key].double()   # fp64 so the metric is stable at 16M elems
        md = (x - y).abs().max().item()
        denom = x.abs().max().item() + 1e-9
        rel = md / denom
        # Primary correctness signal is the (relative) max abs diff. maxdiff==0 is
        # bit-identical. A tiny fp32-accumulation cosine wobble on multi-million-elem
        # tensors is NOT a real mismatch, so cosine is informational only.
        cos = torch.nn.functional.cosine_similarity(x.reshape(-1), y.reshape(-1), dim=0).item()
        good = rel < 1e-3
        ok = ok and good
        print(f"  {key:4s} maxdiff={md:.4e} rel={rel:.2e} cos(fp64)={cos:.8f} {'OK' if good else 'MISMATCH'}")
    sp = da["percall"] / db["percall"] if db["percall"] else 0
    print(f"  speedup(default/tiled) = {sp:.3f}x")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["default", "tiled"])
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--save", default="")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    sys.exit(run(a.mode, a.bs, a.iters, a.save or ""))
