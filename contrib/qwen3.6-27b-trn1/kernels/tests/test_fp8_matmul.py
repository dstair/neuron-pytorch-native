"""Standalone correctness + benchmark for nki_fp8_matmul."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # kernels/ for bare-name imports
import os
# Trn2 only supports legacy float8_e4m3 (not OCP e4m3fn). Bit layouts differ
# only in NaN encoding; for trained weights they're equivalent. This flag
# tells neuron-cc to bitcast F8E4M3FN -> F8E4M3 on input.
# (No special compiler flags needed — weight bytes pass through as int8
# and are reinterpreted as nl.float8_e4m3 inside the kernel.)

import math
import time
import torch

from fp8_matmul import nki_fp8_matmul

# 240.0 is the max e4m3 LEGACY representable value (exponent < 0xF).
# The OCP e4m3fn extends the range up to 448.0 by reusing the all-1s exponent
# for finite values, but Trn2 nc_matmul only supports the legacy format that
# treats those bit patterns as NaN/Inf. Clamp to 240 so quantized values
# stay in the legacy-safe range.
FP8_MAX = 240.0


def quantize(w_bf):
    absmax = w_bf.abs().amax(dim=-1).float().clamp_min(1e-12)
    scale = absmax / FP8_MAX
    inv = (1.0 / scale).to(w_bf.dtype).unsqueeze(-1)
    w_q = (w_bf * inv).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return w_q, scale


def cpu_ref(x, w_fp8, scale):
    """Dequantize + matmul on CPU."""
    w_bf = w_fp8.to(x.dtype) * scale.to(x.dtype).unsqueeze(-1)  # [N, K]
    return torch.nn.functional.linear(x, w_bf)


def run_case(B, K, N, label):
    torch.manual_seed(0)
    device = torch.device("neuron")
    x = torch.randn(B, K, dtype=torch.bfloat16) * 0.5
    w = torch.randn(N, K, dtype=torch.bfloat16) * 0.05
    w_fp8, scale = quantize(w)

    print(f"\n=== {label}: x[{B},{K}] @ w[{N},{K}].T ===")
    ref = cpu_ref(x, w_fp8, scale)

    # Kernel takes the weight pre-transposed to [K, N] layout, scale [N, 1].
    # PyTorch fp8 (e4m3fn) is rejected by the Trn2 HLO verifier. View the
    # weight bytes as int8 (same bit layout) — the kernel reinterprets in
    # SBUF as nl.float8_e4m3 (legacy), which IS valid for Trn2 nc_matmul.
    w_fp8_T = w_fp8.t().contiguous().view(torch.int8)
    scale_2d = scale.unsqueeze(-1).contiguous()  # [N, 1]
    out = nki_fp8_matmul(x.to(device), w_fp8_T.to(device), scale_2d.to(device)).cpu()
    diff = (out.float() - ref.float()).abs()
    rel = diff / (ref.float().abs().clamp_min(1e-3))
    print(f"  max_abs_diff = {diff.max().item():.4f}")
    print(f"  max_rel_diff = {rel.max().item():.4f}")
    print(f"  mean_abs_diff = {diff.mean().item():.5f}")
    # Loose-ish tolerance: bf16 + fp8 + accumulated error.
    ok = diff.max().item() < 0.5
    print(f"  {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok, x.to(device), w_fp8.to(device), scale.to(device)


if __name__ == "__main__":
    # Cases representative of Qwen 3.6 27B per-core shapes (TP=4):
    # MLP gate/up: x[1,5120] @ w[4352,5120].T
    # MLP down:    x[1,4352] @ w[5120,4352].T
    # GQA q:       x[1,5120] @ w[3072,5120].T
    # GQA kv:      x[1,5120] @ w[256,5120].T
    # GQA o:       x[1,1536] @ w[5120,1536].T
    # DN qkv:      x[1,5120] @ w[2560,5120].T
    # DN out:      x[1,1536] @ w[5120,1536].T
    # lm_head:     x[1,5120] @ w[62080,5120].T  (vocab/4 per core)

    all_ok = True
    cases = [
        (1, 5120, 4352, "MLP gate/up"),
        (1, 4352, 5120, "MLP down"),
        (1, 5120, 3072, "GQA q"),
        (1, 5120, 256, "GQA kv"),
        (1, 1536, 5120, "GQA o / DN out"),
        (1, 5120, 2560, "DN qkv"),
        (1, 5120, 62080, "lm_head (vocab/4)"),
    ]
    last = None
    for b, k, n, lbl in cases:
        ok, *args = run_case(b, k, n, lbl)
        all_ok = all_ok and ok
        if lbl == "MLP gate/up":
            last = (b, k, n, args)

    if last:
        b, k, n, (xd, wd, sd) = last
        wdT = wd.t().contiguous().view(torch.int8)
        sd2 = sd.unsqueeze(-1).contiguous()
        print(f"\n=== Bench: {b}x{k}x{n} ({100} iters) ===")
        for _ in range(3):
            nki_fp8_matmul(xd, wdT, sd2)
        N_ITER = 100
        t0 = time.time()
        for _ in range(N_ITER):
            nki_fp8_matmul(xd, wdT, sd2)
        elapsed = (time.time() - t0) / N_ITER * 1000
        print(f"  fp8_matmul: {elapsed:.3f} ms/call")

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
