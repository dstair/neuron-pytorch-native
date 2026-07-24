#!/usr/bin/env python3
"""Compare batched DeltaNet decode kernel variants on deterministic inputs."""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torch_neuronx  # noqa: F401


HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "kernels"))

import deltanet_full_batched_35b_ops  # noqa: E402,F401


K_DIM = 128
V_DIM = 128
K_HEADS = int(os.environ.get("DN_K_HEADS", "2"))
V_HEADS = int(os.environ.get("DN_V_HEADS", "4"))
QKV_DIM = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM
USE_DIRECT_STATE_OUT = os.environ.get("DN_DIRECT_STATE_OUT", "0") == "1"


def _inputs(batch_size, steps):
    generator = torch.Generator().manual_seed(20260721)
    state = (
        torch.randn(
            batch_size * V_HEADS * K_DIM,
            V_DIM,
            generator=generator,
        )
        * 0.05
    ).to(torch.bfloat16)
    conv = (
        torch.randn(
            batch_size * QKV_DIM,
            3,
            generator=generator,
        )
        * 0.5
    ).to(torch.bfloat16)
    conv_weight = (
        torch.randn(QKV_DIM, 4, generator=generator) * 0.1
    ).float()
    conv_bias = (
        torch.randn(QKV_DIM, generator=generator) * 0.05
    ).float()
    a_log = (
        torch.randn(V_HEADS, generator=generator) * 0.1
    ).float()
    dt_bias = (
        torch.randn(V_HEADS, generator=generator) * 0.1
    ).float()
    norm_weight = (
        1.0 + torch.randn(V_DIM, generator=generator) * 0.05
    ).float()
    step_inputs = []
    for _ in range(steps):
        step_inputs.append(
            (
                (
                    torch.randn(
                        batch_size * QKV_DIM,
                        generator=generator,
                    )
                    * 0.5
                ).to(torch.bfloat16),
                (
                    torch.randn(
                        batch_size * V_HEADS,
                        generator=generator,
                    )
                    * 0.1
                ).float(),
                (
                    torch.randn(
                        batch_size * V_HEADS,
                        generator=generator,
                    )
                    * 0.1
                ).float(),
                (
                    torch.randn(
                        batch_size * V_HEADS,
                        V_DIM,
                        generator=generator,
                    )
                    * 0.5
                ).to(torch.bfloat16),
            )
        )
    return (
        state,
        conv,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        norm_weight,
        step_inputs,
    )


def capture(output_dir, batch_size, steps):
    device = torch.neuron.current_device()
    (
        state,
        conv,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        norm_weight,
        step_inputs,
    ) = _inputs(batch_size, steps)
    state = state.to(device)
    conv = conv.to(device)
    shared = tuple(
        value.to(device)
        for value in (
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            norm_weight,
        )
    )

    captures = []
    with torch.no_grad():
        for mixed_qkv, a_out, b_out, z in step_inputs:
            inputs = (
                state,
                mixed_qkv.to(device),
                conv,
                shared[0],
                shared[1],
                a_out.to(device),
                b_out.to(device),
                z.to(device),
                shared[2],
                shared[3],
                shared[4],
            )
            if USE_DIRECT_STATE_OUT:
                state_out = torch.empty_like(state)
                conv_out = torch.empty_like(conv)
                state, conv, output = (
                    torch.ops.deltanet35b.full_batched_direct(
                        *inputs,
                        state_out,
                        conv_out,
                    )
                )
            else:
                state, conv, output = torch.ops.deltanet35b.full_batched(
                    *inputs
                )
            torch.neuron.synchronize()
            captures.append(
                {
                    "state": state.cpu(),
                    "conv": conv.cpu(),
                    "output": output.cpu(),
                }
            )

    os.makedirs(output_dir, exist_ok=True)
    torch.save(captures, os.path.join(output_dir, "capture.pt"))
    print(
        f"captured B={batch_size} steps={steps} "
        f"wide={os.environ.get('DN_WIDE_CONV', '0')} "
        f"direct_state_out={int(USE_DIRECT_STATE_OUT)}"
    )


def compare(reference_dir, candidate_dir):
    reference = torch.load(
        os.path.join(reference_dir, "capture.pt"),
        map_location="cpu",
        weights_only=True,
    )
    candidate = torch.load(
        os.path.join(candidate_dir, "capture.pt"),
        map_location="cpu",
        weights_only=True,
    )
    if len(reference) != len(candidate):
        raise AssertionError("capture step counts differ")

    for step, (expected_step, actual_step) in enumerate(
        zip(reference, candidate)
    ):
        for name in ("state", "conv", "output"):
            expected = expected_step[name]
            actual = actual_step[name]
            diff = actual.float() - expected.float()
            expected_norm = torch.linalg.vector_norm(expected.float())
            rel_l2 = float(
                torch.linalg.vector_norm(diff)
                / expected_norm.clamp_min(1e-30)
            )
            cosine = float(
                F.cosine_similarity(
                    actual.float().reshape(-1),
                    expected.float().reshape(-1),
                    dim=0,
                )
            )
            mismatches = int((actual != expected).sum())
            print(
                f"step={step} {name}: exact={mismatches == 0} "
                f"mismatches={mismatches} cosine={cosine:.8f} "
                f"max_abs={float(diff.abs().max()):.6e} "
                f"rel_l2={rel_l2:.6e}"
            )
            assert mismatches == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("REFERENCE", "CANDIDATE"),
    )
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if not args.output_dir:
        parser.error("--output-dir is required")
    capture(args.output_dir, args.batch_size, args.steps)


if __name__ == "__main__":
    main()
