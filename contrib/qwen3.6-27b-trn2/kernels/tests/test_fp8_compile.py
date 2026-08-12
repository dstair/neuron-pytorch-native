"""Minimal test: does torch.compile(backend='neuron') support fp8 operands?
Tiny model: 1 linear layer, weights stored as fp8 + per-row scale, dequantized
inline before matmul. No TP, no distributed.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # kernels/ for bare-name imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

FP8_E4M3_MAX = 448.0


def quantize(w_bf):
    absmax = w_bf.abs().amax(dim=-1).float().clamp_min(1e-12)
    scale = absmax / FP8_E4M3_MAX
    inv = (1.0 / scale).to(w_bf.dtype).unsqueeze(-1)
    w_q = (w_bf * inv).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return w_q, scale


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.randn(256, 128, dtype=torch.bfloat16)
        w_q, scale = quantize(w)
        self.register_buffer("w_q", w_q)
        self.register_buffer("scale", scale)

    def forward(self, x):
        # x [..., 128] bf16
        w = self.w_q.to(x.dtype) * self.scale.to(x.dtype).unsqueeze(-1)
        return F.linear(x, w)


if __name__ == "__main__":
    import torch_neuronx  # noqa: F401
    device = torch.device("neuron")
    m = Net().to(device).eval()
    x = torch.randn(1, 128, dtype=torch.bfloat16, device=device)

    print("=== Eager ===")
    with torch.no_grad():
        y = m(x)
        print("  output shape:", y.shape, "dtype:", y.dtype)
        print("  output[0, :5]:", y[0, :5].cpu().tolist())

    print("\n=== Compile ===")
    cm = torch.compile(m, backend="neuron", fullgraph=True, dynamic=False)
    with torch.no_grad():
        y2 = cm(x)
        print("  output shape:", y2.shape, "dtype:", y2.dtype)
        print("  output[0, :5]:", y2[0, :5].cpu().tolist())
        max_diff = (y - y2).abs().max().item()
        print(f"  eager-vs-compile max_diff: {max_diff:.6f}")
    print("PASS" if max_diff < 1e-2 else "FAIL")
