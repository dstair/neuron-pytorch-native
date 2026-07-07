"""Bit-exact correctness test: nki_rms_norm vs reference Qwen3.5 RMSNorm.

Reference (static_decode.rms_norm):
    out = (1 + weight.f32) * (x.f32 * rsqrt(mean(x.f32^2) + eps))   -> x.dtype

Invokes the @nki.jit kernel directly on torch.device("neuron"), mirroring
test_deltanet_full.py (the torch.ops/XLA path needs torch_neuronx.pyhlo which
isn't present in this image).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # kernels/ for bare-name imports
import torch
from rms_norm_nki import nki_rms_norm

RMS_EPS = 1e-6
H = 5120


def ref_rms_norm(x, weight):
    x_f32 = x.float()
    norm = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return ((1.0 + weight.float()) * norm).to(x.dtype)


def run(device, dtype, seed):
    torch.manual_seed(seed)
    x = torch.randn(H, dtype=dtype)
    w = torch.randn(H, dtype=torch.float32) * 0.1  # weight near 0 (residual norm)

    ref = ref_rms_norm(x, w)
    got = nki_rms_norm(x.to(device), w.to(device)).cpu()

    ref_f = ref.float()
    got_f = got.float()
    max_diff = (ref_f - got_f).abs().max().item()
    rel = max_diff / (ref_f.abs().max().item() + 1e-12)
    print(f"  dtype={dtype} seed={seed}: max_diff={max_diff:.3e} rel={rel:.3e}")
    return max_diff


if __name__ == "__main__":
    device = torch.device("neuron")
    print("nki_rms_norm vs reference RMSNorm (H=%d)" % H)
    worst = 0.0
    for dt in (torch.bfloat16, torch.float32):
        for s in (0, 1, 7):
            worst = max(worst, run(device, dt, s))
    # bf16 has ~7.8e-3 ULP at unit scale; f32 should be ~1e-6.
    tol = 5e-2
    print(f"WORST max_diff={worst:.3e}  tol={tol}")
    print("PASS" if worst < tol else "FAIL")
