#!/usr/bin/env python3
"""Device smoke/parity tests for the fused dense FP8 kernels."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.dirname(HERE)
sys.path.insert(0, KERNELS)

import fp8_fused_head_ops  # noqa: E402,F401
import fp8_fused_mlp_ops  # noqa: E402,F401
import fp8_matmul_ops  # noqa: E402,F401


FP8_MAX = 240.0


def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    absmax = weight.abs().amax(dim=-1).float().clamp_min(1e-12)
    scale = absmax / FP8_MAX
    q = (
        (weight * (1.0 / scale).to(weight.dtype).unsqueeze(1))
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
    )
    return q.t().contiguous().view(torch.int8), scale.unsqueeze(1)


def timed(label: str, fn):
    start = time.time()
    value = fn()
    torch.neuron.synchronize()
    print(f"{label}: {time.time() - start:.3f}s", flush=True)
    return value


def test_mlp(device, batch: int, hidden: int, intermediate: int) -> None:
    torch.manual_seed(17)
    x = (torch.randn(batch, hidden) * 0.2).to(torch.bfloat16)
    gate = (torch.randn(intermediate, hidden) * 0.02).to(torch.bfloat16)
    up = (torch.randn(intermediate, hidden) * 0.02).to(torch.bfloat16)
    down = (torch.randn(hidden, intermediate) * 0.02).to(torch.bfloat16)
    gate_q, gate_s = quantize(gate)
    up_q, up_s = quantize(up)
    down_q, down_s = quantize(down)

    x_d = x.to(device)
    gate_q_d = gate_q.to(device)
    up_q_d = up_q.to(device)
    down_q_d = down_q.to(device)
    gate_s_d = gate_s.to(device)
    up_s_d = up_s.to(device)
    down_s_d = down_s.to(device)
    packed_q_d = torch.stack((gate_q, up_q), dim=1).contiguous().to(device)
    packed_s_d = torch.cat((gate_s.t(), up_s.t()), dim=0).contiguous().to(device)

    def generic():
        gate_out = torch.ops.qwen4b_fp8.matmul(x_d, gate_q_d, gate_s_d)
        up_out = torch.ops.qwen4b_fp8.matmul(x_d, up_q_d, up_s_d)
        activated = F.silu(gate_out) * up_out
        return torch.ops.qwen4b_fp8.matmul(
            activated, down_q_d, down_s_d
        )

    expected = timed("generic MLP", generic)
    actual = timed(
        "fused MLP",
        lambda: torch.ops.qwen4b_fp8.fused_mlp(
            x_d,
            packed_q_d,
            packed_s_d,
            down_q_d,
            down_s_d.t().contiguous(),
        ),
    )
    expected_cpu = expected.cpu()
    actual_cpu = actual.cpu()
    diff = (expected_cpu.float() - actual_cpu.float()).abs()
    print(
        f"MLP parity: max_abs={diff.max().item():.6g} "
        f"mean_abs={diff.mean().item():.6g} "
        f"exact={torch.equal(expected_cpu, actual_cpu)}",
        flush=True,
    )
    if diff.max().item() > 0.25:
        raise AssertionError("fused MLP differs materially from generic FP8 path")


def test_head(device, batch: int, hidden: int, vocab: int) -> None:
    torch.manual_seed(23)
    x = (torch.randn(batch, hidden) * 0.2).to(torch.bfloat16)
    weight = (torch.randn(vocab, hidden) * 0.02).to(torch.bfloat16)
    q, scale = quantize(weight)
    x_d = x.to(device)
    q_d = q.to(device)
    scale_d = scale.to(device)

    expected = timed(
        "generic head",
        lambda: torch.ops.qwen4b_fp8.matmul(x_d, q_d, scale_d).argmax(
            dim=-1
        ),
    )
    actual = timed(
        "fused head",
        lambda: torch.ops.qwen4b_fp8.fused_head_argmax(
            x_d,
            q_d,
            scale_d.t().contiguous(),
        ).squeeze(1),
    )
    expected_cpu = expected.cpu().to(torch.int32)
    actual_cpu = actual.cpu().to(torch.int32)
    print(
        f"head parity: matches="
        f"{int((expected_cpu == actual_cpu).sum())}/{batch}",
        flush=True,
    )
    if not torch.equal(expected_cpu, actual_cpu):
        raise AssertionError(
            f"fused head IDs differ: expected={expected_cpu.tolist()} "
            f"actual={actual_cpu.tolist()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mlp", "head", "both"), default="both")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=4096)
    args = parser.parse_args()

    import torch_neuronx  # noqa: F401

    device = torch.neuron.current_device()
    if args.mode in ("mlp", "both"):
        test_mlp(device, args.batch, args.hidden, args.intermediate)
    if args.mode in ("head", "both"):
        test_head(device, args.batch, args.hidden, args.vocab)


if __name__ == "__main__":
    main()
