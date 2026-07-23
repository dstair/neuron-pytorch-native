#!/usr/bin/env python3
"""Isolated fused-W8 MoE device equivalence for supported decode batches.

Run inside the PyTorch Native DLC on one NeuronCore:

  python3 test_moe_fused_w8_device.py --mode fp8 --batch-sizes 32,64,128

Use `--local-experts 32` for the production TP=8 shape after the small smoke.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F


KERNELS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(KERNELS)
sys.path.insert(0, KERNELS)
sys.path.insert(0, ROOT)

from moe_w8 import (  # noqa: E402
    build_local_affinities,
    decode_legacy_e4m3,
    encode_legacy_e4m3,
    fused_moe_block_coalesced_cpu,
    fused_moe_cpu,
    OfficialFP8ExpertReader,
    pack_coalesced_block_scales,
)


def make_case(batch, experts, hidden_size, intermediate, mode):
    torch.manual_seed(1234 + batch)
    weight_shape = (experts, 2, hidden_size, intermediate)
    down_shape = (experts, intermediate, hidden_size)
    if mode == "fp8":
        gate_up = encode_legacy_e4m3(torch.randn(weight_shape) * 0.25)
        down = encode_legacy_e4m3(torch.randn(down_shape) * 0.25)
        gate_up_residual = torch.zeros_like(gate_up)
        down_residual = torch.zeros_like(down)
    else:
        gate_up = torch.randint(-32, 33, weight_shape, dtype=torch.int8)
        down = torch.randint(-32, 33, down_shape, dtype=torch.int8)
    gate_up_scale = (
        torch.rand(
            experts, 2, intermediate // 128, hidden_size // 128
        )
        .mul_(0.01)
        .add_(0.002)
        .to(torch.bfloat16)
        .unsqueeze(-1)
        .expand(-1, -1, -1, -1, 128)
        .contiguous()
    )
    down_scale = (
        torch.rand(
            experts, hidden_size // 128, intermediate // 128
        )
        .mul_(0.01)
        .add_(0.002)
        .to(torch.bfloat16)
        .unsqueeze(-1)
        .expand(-1, -1, -1, 128)
        .contiguous()
    )
    hidden = torch.randn(batch, hidden_size).to(torch.bfloat16)
    affinities = torch.zeros(batch, experts, dtype=torch.float32)
    selected = torch.randint(0, experts, (batch, min(8, experts)))
    values = torch.rand_like(selected, dtype=torch.float32)
    values = values / values.sum(dim=1, keepdim=True)
    affinities.scatter_add_(1, selected, values)
    if mode == "fp8":
        return (
            hidden,
            gate_up,
            gate_up_residual,
            down,
            down_residual,
            gate_up_scale,
            down_scale,
            affinities,
        )
    return hidden, gate_up, down, gate_up_scale, down_scale, affinities


def make_coalesced_case(case):
    """Repack a synthetic dual-plane case for the coalesced single plane."""
    (
        hidden,
        gate_up,
        _gate_up_residual,
        down,
        _down_residual,
        gate_up_scales,
        down_scales,
        affinities,
    ) = case
    gate_up_scale_table, down_scale_table = (
        pack_coalesced_block_scales(
            gate_up_scales[..., 0],
            down_scales[..., 0],
        )
    )
    return (
        hidden,
        gate_up.permute(0, 2, 1, 3).contiguous(),
        down,
        gate_up_scale_table,
        down_scale_table,
        affinities,
    )


def make_coalesced_ob_case(case):
    """Coalesced case with input-independent (per-output-block) scales.

    Reduction B1 requires s[i,o] == s[o]: the kernel post-scales the whole
    PSUM-accumulated contraction with input-block 0's scale. Force the synthetic
    grids constant across the input-block axis (hidden blocks for gate/up,
    intermediate blocks for down) so the coalesced CPU reference — which applies
    the per-block grid — agrees with the kernel.
    """
    (
        hidden,
        gate_up,
        _gate_up_residual,
        down,
        _down_residual,
        gate_up_scales,
        down_scales,
        affinities,
    ) = case
    # gate/up grid [E,2,I/128,H/128,128]; input axis is H/128 (dim 3).
    gate_up_ob = gate_up_scales[:, :, :, :1, :].expand_as(gate_up_scales)
    # down grid [E,H/128,I/128,128]; input axis is I/128 (dim 2).
    down_ob = down_scales[:, :, :1, :].expand_as(down_scales)
    gate_up_scale_table, down_scale_table = (
        pack_coalesced_block_scales(
            gate_up_ob[..., 0].contiguous(),
            down_ob[..., 0].contiguous(),
        )
    )
    return (
        hidden,
        gate_up.permute(0, 2, 1, 3).contiguous(),
        down,
        gate_up_scale_table,
        down_scale_table,
        affinities,
    )


def make_official_case(
    batch,
    checkpoint,
    layer,
    expert,
    expert_count,
    mode,
    fp8_impl,
    analyze_row=False,
):
    torch.manual_seed(1234 + batch)
    reader = OfficialFP8ExpertReader(checkpoint)
    try:
        expert_end = expert + expert_count
        converted, stats = reader.load_layer(
            layer,
            expert,
            expert_end,
            mode,
            fp8_impl=fp8_impl,
        )
        reference = reader.load_layer_bf16(layer, expert, expert_end)
        row_converted = None
        row_stats = None
        if analyze_row:
            row_converted, row_stats = reader.load_layer_row_fp8(
                layer, expert, expert_end
            )
    finally:
        reader.close()
    hidden_size = (
        converted["w8_gate_up"].shape[1]
        if fp8_impl in ("block_pow2_coalesced", "block_ob_coalesced")
        else converted["w8_gate_up"].shape[2]
    )
    hidden = torch.randn(batch, hidden_size).to(torch.bfloat16)
    selected = torch.randint(
        0, expert_count, (batch, min(8, expert_count))
    )
    affinity_values = torch.rand_like(selected, dtype=torch.float32)
    affinity_values /= affinity_values.sum(dim=1, keepdim=True)
    affinities = torch.zeros(batch, expert_count, dtype=torch.float32)
    affinities.scatter_add_(1, selected, affinity_values)
    if mode == "fp8" and fp8_impl == "dual":
        case = (
            hidden,
            converted["w8_gate_up"],
            converted["w8_gate_up_residual"],
            converted["w8_down"],
            converted["w8_down_residual"],
            converted["w8_gate_up_scale"],
            converted["w8_down_scale"],
            affinities,
        )
    else:
        case = (
            hidden,
            converted["w8_gate_up"],
            converted["w8_down"],
            converted["w8_gate_up_scale"],
            converted["w8_down_scale"],
            affinities,
        )

    official_output = torch.zeros(
        batch, hidden_size, dtype=torch.float32
    )
    for local_expert in range(expert_count):
        gate, up = reference["gate_up"][local_expert].chunk(2, dim=0)
        activated = F.silu(F.linear(hidden.float(), gate.float()))
        activated *= F.linear(hidden.float(), up.float())
        expert_output = F.linear(
            activated, reference["down"][local_expert].float()
        )
        official_output += (
            expert_output * affinities[:, local_expert : local_expert + 1].float()
        )
    quantized_output = cpu_reference(case, mode)
    diff = quantized_output - official_output
    cosine = F.cosine_similarity(
        quantized_output.reshape(-1), official_output.reshape(-1), dim=0
    ).item()
    normalized_rmse = (
        torch.linalg.vector_norm(diff)
        / torch.linalg.vector_norm(official_output).clamp_min(1e-12)
    ).item()
    mode_label = f"{mode}/{fp8_impl}" if mode == "fp8" else mode
    print(
        f"official layer={layer} experts=[{expert},{expert_end}) -> "
        f"{mode_label}: "
        f"weight_cosine={stats.cosine:.7f} "
        f"weight_nrmse={stats.normalized_rmse:.5%} "
        f"output_cosine={cosine:.7f} output_nrmse={normalized_rmse:.5%}"
    )
    if mode == "fp8" and fp8_impl in (
        "block_pow2",
        "block_pow2_coalesced",
    ):
        print(
            "official block-power-of-two: "
            f"shifted_blocks={stats.shifted_block_fraction:.2%} "
            f"exact_values={stats.exact_fraction:.2%} "
            f"clipped={stats.clipped_count}"
        )
        assert stats.cosine >= 0.9995
        assert stats.normalized_rmse <= 0.035
        assert cosine >= 0.9995
        assert normalized_rmse <= 0.035
    if mode == "fp8" and fp8_impl == "dual":
        base_output = cpu_reference(case, mode, include_residual=False)
        base_diff = base_output - official_output
        base_cosine = F.cosine_similarity(
            base_output.reshape(-1), official_output.reshape(-1), dim=0
        ).item()
        base_nrmse = (
            torch.linalg.vector_norm(base_diff)
            / torch.linalg.vector_norm(official_output).clamp_min(1e-12)
        ).item()
        print(
            "official clipped-base only: "
            f"output_cosine={base_cosine:.7f} "
            f"output_nrmse={base_nrmse:.5%}"
        )
    if row_converted is not None:
        row_output = torch.zeros_like(official_output)
        for local_expert in range(expert_count):
            gate_up = row_converted["row_gate_up"][local_expert]
            gate_up_scale = row_converted[
                "row_gate_up_scale"
            ][local_expert]
            gate = (
                decode_legacy_e4m3(gate_up[:, 0].transpose(0, 1))
                * gate_up_scale[0, :, None]
            )
            up = (
                decode_legacy_e4m3(gate_up[:, 1].transpose(0, 1))
                * gate_up_scale[1, :, None]
            )
            down = (
                decode_legacy_e4m3(
                    row_converted["row_down"][local_expert].transpose(0, 1)
                )
                * row_converted["row_down_scale"][
                    local_expert, :, None
                ]
            )
            activated = F.silu(F.linear(hidden.float(), gate))
            activated *= F.linear(hidden.float(), up)
            row_output += (
                F.linear(activated, down)
                * affinities[:, local_expert : local_expert + 1]
            )
        row_diff = row_output - official_output
        row_cosine = F.cosine_similarity(
            row_output.reshape(-1), official_output.reshape(-1), dim=0
        ).item()
        row_nrmse = (
            torch.linalg.vector_norm(row_diff)
            / torch.linalg.vector_norm(official_output).clamp_min(1e-12)
        ).item()
        print(
            "official row-legacy FP8: "
            f"weight_cosine={row_stats.cosine:.7f} "
            f"weight_nrmse={row_stats.normalized_rmse:.5%} "
            f"output_cosine={row_cosine:.7f} "
            f"output_nrmse={row_nrmse:.5%}"
        )
    return case


def make_replay_case(
    capture_dir,
    rank,
    checkpoint,
    layer,
    expert_count,
    mode,
    fp8_impl,
):
    capture = torch.load(
        os.path.join(capture_dir, f"rank{rank}.pt"),
        map_location="cpu",
        weights_only=True,
    )
    diagnostic_layers = capture["diagnostic_layers"].tolist()
    if layer not in diagnostic_layers:
        raise ValueError(
            f"capture does not contain layer {layer}; found {diagnostic_layers}"
        )
    index = diagnostic_layers.index(layer)
    hidden = capture["diag_moe_input"][index].to(torch.bfloat16)
    top_ids = capture["diag_router_top_ids"][index]
    top_weights = capture["diag_router_top_weights"][index].float()
    expert = rank * expert_count
    affinities = build_local_affinities(
        top_ids, top_weights, expert, expert_count
    )
    reader = OfficialFP8ExpertReader(checkpoint)
    try:
        converted, _ = reader.load_layer(
            layer,
            expert,
            expert + expert_count,
            mode,
            fp8_impl=fp8_impl,
        )
    finally:
        reader.close()
    if mode == "fp8" and fp8_impl == "dual":
        case = (
            hidden,
            converted["w8_gate_up"],
            converted["w8_gate_up_residual"],
            converted["w8_down"],
            converted["w8_down_residual"],
            converted["w8_gate_up_scale"],
            converted["w8_down_scale"],
            affinities,
        )
    else:
        case = (
            hidden,
            converted["w8_gate_up"],
            converted["w8_down"],
            converted["w8_gate_up_scale"],
            converted["w8_down_scale"],
            affinities,
        )
    return case, capture["diag_local_routed"][index].float()


def cpu_reference(case, mode, include_residual=True):
    if (
        mode == "fp8"
        and len(case) == 6
        and case[1].ndim == 4
        and case[1].shape[2] == 2
    ):
        return fused_moe_block_coalesced_cpu(
            *case, rounding=("activation_bf16",)
        )
    if mode == "fp8" and len(case) == 8:
        (
            hidden,
            gate_up,
            gate_up_residual,
            down,
            down_residual,
            gate_up_scales,
            down_scales,
            affinities,
        ) = case
        residual_args = {}
        if include_residual:
            residual_args = {
                "gate_up_residual": gate_up_residual,
                "down_residual": down_residual,
            }
        return fused_moe_cpu(
            hidden,
            gate_up,
            down,
            gate_up_scales,
            down_scales,
            affinities,
            mode,
            rounding=("activation_bf16",),
            **residual_args,
        )
    return fused_moe_cpu(
        *case, mode, rounding=("activation_bf16",)
    )


def kernel_case(
    case,
    mode,
    fp8_impl,
    drop_residual,
    affinities=None,
):
    if affinities is None:
        affinities = case[-1]
    if mode == "fp8" and len(case) == 8 and (
        drop_residual or fp8_impl == "block_pow2"
    ):
        return (
            case[0],
            case[1],
            case[3],
            case[5],
            case[6],
            affinities,
        )
    return (*case[:-1], affinities)


def report_reduction(name, actual, expected):
    diff = actual.float() - expected.float()
    cosine = F.cosine_similarity(
        actual.float().reshape(-1),
        expected.float().reshape(-1),
        dim=0,
    ).item()
    normalized_rmse = (
        torch.linalg.vector_norm(diff)
        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    ).item()
    bf16_mismatches = int(
        (actual.to(torch.bfloat16) != expected.to(torch.bfloat16)).sum()
    )
    print(
        f"reduction={name}: cosine={cosine:.9f} "
        f"nrmse={normalized_rmse:.8%} "
        f"max_abs={diff.abs().max().item():.9e} "
        f"bf16_mismatches={bf16_mismatches}"
    )


def pairwise_sum(values):
    current = list(values.unbind(0))
    while len(current) > 1:
        current = [
            current[index] + current[index + 1]
            for index in range(0, len(current), 2)
        ]
    return current[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fp8", "int8"), required=True)
    parser.add_argument(
        "--fp8-impl",
        choices=(
            "dual",
            "block_pow2",
            "block_pow2_coalesced",
            "block_ob_coalesced",
        ),
        default="dual",
    )
    parser.add_argument("--layout", choices=("weight", "token"), default="weight")
    parser.add_argument("--batch-sizes", default="32,64,128,256")
    parser.add_argument("--local-experts", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--expert-model-path")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int)
    parser.add_argument("--expert-count", type=int, default=1)
    parser.add_argument("--benchmark-iters", type=int, default=0)
    parser.add_argument(
        "--drop-residual",
        action="store_true",
        help="Benchmark the clipped native base plane without correction.",
    )
    parser.add_argument(
        "--analyze-row",
        action="store_true",
        help="Also report legacy-FP8 per-output-row source quality.",
    )
    parser.add_argument(
        "--input-capture",
        help="Replay the selected layer from a diagnostic reference capture.",
    )
    parser.add_argument(
        "--tp-rank",
        type=int,
        default=int(os.environ.get("RANK", "0")),
    )
    parser.add_argument(
        "--analyze-expert-reduction",
        action="store_true",
        help="Replay each expert separately and compare summation orders.",
    )
    args = parser.parse_args()

    import torch_neuronx  # noqa: F401
    import moe_fused_w8_35b_ops  # noqa: F401

    device = torch.neuron.current_device()
    if args.layout == "token":
        parser.error("exact dual-plane FP8 requires --layout weight")
    if args.drop_residual and (
        args.mode != "fp8" or args.fp8_impl != "dual"
    ):
        parser.error(
            "--drop-residual requires --mode fp8 --fp8-impl dual"
        )
    if args.mode == "fp8":
        if args.fp8_impl == "block_ob_coalesced":
            op = torch.ops.moe_w8.fused_fp8_block_coalesced_ob
        elif args.fp8_impl == "block_pow2_coalesced":
            op = torch.ops.moe_w8.fused_fp8_block_coalesced
        elif args.fp8_impl == "block_pow2" or args.drop_residual:
            op = torch.ops.moe_w8.fused_fp8_native
        else:
            op = torch.ops.moe_w8.fused_fp8_dual
    else:
        op = torch.ops.moe_w8.fused_int8
    batches = [int(value) for value in args.batch_sizes.split(",")]
    if args.input_capture:
        if not args.expert_model_path:
            parser.error("--input-capture requires --expert-model-path")
        replay_case, replay_expected = make_replay_case(
            args.input_capture,
            args.tp_rank,
            args.expert_model_path,
            args.layer,
            args.expert_count,
            args.mode,
            args.fp8_impl,
        )
        batches = [replay_case[0].shape[0]]

    for batch in batches:
        reference_expected = None
        if args.input_capture:
            case = replay_case
            reference_expected = replay_expected
        elif args.expert_model_path:
            case = make_official_case(
                batch,
                args.expert_model_path,
                args.layer,
                args.expert or 0,
                args.expert_count,
                args.mode,
                args.fp8_impl,
                analyze_row=args.analyze_row,
            )
        else:
            case = make_case(
                batch,
                args.local_experts,
                args.hidden_size,
                args.intermediate_size,
                args.mode,
            )
            if (
                args.mode == "fp8"
                and args.fp8_impl == "block_pow2_coalesced"
            ):
                case = make_coalesced_case(case)
            elif (
                args.mode == "fp8"
                and args.fp8_impl == "block_ob_coalesced"
            ):
                case = make_coalesced_ob_case(case)
        expected = cpu_reference(
            case, args.mode, include_residual=not args.drop_residual
        )
        op_case = kernel_case(
            case,
            args.mode,
            args.fp8_impl,
            args.drop_residual,
        )
        actual = op(*(tensor.to(device) for tensor in op_case)).cpu().float()
        diff = actual - expected
        cosine = F.cosine_similarity(
            actual.reshape(-1), expected.reshape(-1), dim=0
        ).item()
        normalized_rmse = (
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(expected).clamp_min(1e-12)
        ).item()
        mode_label = (
            f"{args.mode}/{args.fp8_impl}"
            if args.mode == "fp8"
            else args.mode
        )
        print(
            f"mode={mode_label} layout={args.layout} "
            f"BS={batch} E={case[1].shape[0]}: "
            f"cosine={cosine:.7f} nrmse={normalized_rmse:.5%} "
            f"max_abs={diff.abs().max().item():.6f}"
        )
        if args.mode == "fp8" and args.fp8_impl in (
            "block_pow2",
            "block_pow2_coalesced",
            "block_ob_coalesced",
        ):
            assert cosine >= 0.9999
            assert normalized_rmse <= 0.01
        else:
            assert cosine >= 0.999
            assert normalized_rmse <= 0.03

        if args.benchmark_iters:
            device_case = tuple(tensor.to(device) for tensor in op_case)
            for _ in range(3):
                benchmark_output = op(*device_case)
            torch.neuron.synchronize()
            start = time.perf_counter()
            for _ in range(args.benchmark_iters):
                benchmark_output = op(*device_case)
            torch.neuron.synchronize()
            elapsed = time.perf_counter() - start
            print(
                f"BENCHMARK: {elapsed * 1000 / args.benchmark_iters:.3f} "
                f"ms/call over {args.benchmark_iters} iterations"
            )
            del benchmark_output

        if reference_expected is not None:
            reference_diff = actual - reference_expected
            reference_cosine = F.cosine_similarity(
                actual.reshape(-1),
                reference_expected.reshape(-1),
                dim=0,
            ).item()
            reference_nrmse = (
                torch.linalg.vector_norm(reference_diff)
                / torch.linalg.vector_norm(reference_expected).clamp_min(1e-12)
            ).item()
            reference_bf16_mismatches = int(
                (
                    actual.to(torch.bfloat16)
                    != reference_expected.to(torch.bfloat16)
                ).sum()
            )
            print(
                f"replay rank={args.tp_rank} layer={args.layer}: "
                f"cosine={reference_cosine:.9f} "
                f"nrmse={reference_nrmse:.8%} "
                f"max_abs={reference_diff.abs().max().item():.9e} "
                f"bf16_mismatches={reference_bf16_mismatches}"
            )
            assert reference_cosine >= 0.999
            assert reference_nrmse <= 0.05

        if args.analyze_expert_reduction:
            if reference_expected is None:
                parser.error(
                    "--analyze-expert-reduction requires --input-capture"
                )
            expert_outputs = []
            for expert in range(case[1].shape[0]):
                expert_affinities = torch.zeros_like(case[-1], device=device)
                expert_affinities[:, expert] = case[-1][:, expert].to(device)
                expert_outputs.append(
                    op(
                        *(
                            tensor.to(device)
                            for tensor in kernel_case(
                                case,
                                args.mode,
                                args.fp8_impl,
                                args.drop_residual,
                                expert_affinities,
                            )
                        ),
                    ).cpu()
                )
            expert_outputs = torch.stack(expert_outputs).float()
            sequential = torch.zeros_like(expert_outputs[0])
            for expert_output in expert_outputs:
                sequential += expert_output
            report_reduction("kernel-full", actual, reference_expected)
            report_reduction("sequential", sequential, reference_expected)
            report_reduction(
                "pairwise-tree",
                pairwise_sum(expert_outputs),
                reference_expected,
            )
            report_reduction(
                "torch-sum",
                expert_outputs.sum(dim=0),
                reference_expected,
            )

        zero_affinities = torch.zeros_like(case[-1], device=device)
        zero = op(
            *(
                tensor.to(device)
                for tensor in kernel_case(
                    case,
                    args.mode,
                    args.fp8_impl,
                    args.drop_residual,
                    zero_affinities,
                )
            )
        ).cpu()
        assert torch.equal(zero, torch.zeros_like(zero))
    print("PASS: fused W8 device outputs match the quantized CPU reference")


if __name__ == "__main__":
    main()
