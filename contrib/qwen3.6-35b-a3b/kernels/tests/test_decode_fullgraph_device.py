#!/usr/bin/env python3
"""Compare segmented and full-graph vocab-sharded real-weight decode."""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


NAMES = ("logits", "next_id", "deltanet", "conv", "kv_k", "kv_v")
DIAGNOSTIC_NAMES = (
    "diag_layer_input",
    "diag_attention_output",
    "diag_post_attention_hidden",
    "diag_moe_input",
    "diag_router_top_ids",
    "diag_router_top_weights",
    "diag_local_routed",
    "diag_global_routed",
    "diag_shared_output",
    "diag_layer_output",
)


def _load_capture(directory, rank):
    return torch.load(
        os.path.join(directory, f"rank{rank}.pt"),
        map_location="cpu",
        weights_only=True,
    )


def _combine_reference_chunks(directories, rank):
    chunks = [_load_capture(directory, rank) for directory in directories]
    combined = {}
    for name in chunks[0]:
        values = [chunk[name] for chunk in chunks]
        if name == "diagnostic_layers":
            if not all(torch.equal(values[0], value) for value in values[1:]):
                raise ValueError("reference chunks use different diagnostic layers")
            combined[name] = values[0]
        elif name in ("logits", "next_id"):
            combined[name] = torch.cat(values, dim=0)
        elif name in NAMES[2:] or name in DIAGNOSTIC_NAMES:
            combined[name] = torch.cat(values, dim=1)
        else:
            raise KeyError(f"unknown capture tensor {name!r}")
    return combined


def _route_membership_mismatches(actual_ids, expected_ids):
    if actual_ids.shape != expected_ids.shape:
        raise ValueError(
            "route ID shapes differ: "
            f"{tuple(actual_ids.shape)} vs {tuple(expected_ids.shape)}"
        )
    actual_sorted = actual_ids.sort(dim=-1).values
    expected_sorted = expected_ids.sort(dim=-1).values
    return int((actual_sorted != expected_sorted).sum())


def _weights_in_expert_order(route_ids, route_weights):
    if route_ids.shape != route_weights.shape:
        raise ValueError(
            "route ID/weight shapes differ: "
            f"{tuple(route_ids.shape)} vs {tuple(route_weights.shape)}"
        )
    order = route_ids.argsort(dim=-1)
    return route_weights.gather(-1, order)


def _global_logits(captures, candidate_global_width=None):
    first = captures[0]["logits"].float()
    if candidate_global_width is not None and first.shape[-1] == candidate_global_width:
        return first
    return torch.cat([capture["logits"].float() for capture in captures], dim=-1)


def _near_tie_mismatches(
    reference_captures,
    candidate_captures,
    tolerance,
):
    candidate_logits = _global_logits(candidate_captures)
    reference_logits = _global_logits(
        reference_captures,
        candidate_global_width=candidate_logits.shape[-1],
    )
    reference_ids = reference_captures[0]["next_id"]
    candidate_ids = candidate_captures[0]["next_id"]
    mismatches = (candidate_ids != reference_ids).nonzero().reshape(-1)
    rejected = []
    details = []
    for row_tensor in mismatches:
        row = int(row_tensor)
        reference_id = int(reference_ids[row])
        candidate_id = int(candidate_ids[row])
        reference_margin = float(
            reference_logits[row, reference_id]
            - reference_logits[row, candidate_id]
        )
        candidate_margin = float(
            candidate_logits[row, candidate_id]
            - candidate_logits[row, reference_id]
        )
        accepted = (
            abs(reference_margin) <= tolerance
            and abs(candidate_margin) <= tolerance
        )
        if not accepted:
            rejected.append(row)
        details.append(
            (
                row,
                reference_id,
                candidate_id,
                reference_margin,
                candidate_margin,
                accepted,
            )
        )
    return rejected, details


def test_near_tie_mismatches_accepts_only_mutual_near_ties():
    reference_logits = torch.tensor([[1.0, 0.0, 0.9375, -1.0]])
    reference_captures = [
        {
            "logits": reference_logits,
            "next_id": torch.tensor([0]),
        }
        for _ in range(2)
    ]
    candidate_captures = [
        {
            "logits": torch.tensor([[0.9375, 0.0]]),
            "next_id": torch.tensor([2]),
        },
        {
            "logits": torch.tensor([[1.0, -1.0]]),
            "next_id": torch.tensor([2]),
        },
    ]

    rejected, details = _near_tie_mismatches(
        reference_captures,
        candidate_captures,
        tolerance=0.125,
    )

    assert rejected == []
    assert details == [(0, 0, 2, 0.0625, 0.0625, True)]


def test_near_tie_mismatches_rejects_a_meaningful_margin():
    reference_captures = [
        {
            "logits": torch.tensor([[1.0, 0.0]]),
            "next_id": torch.tensor([0]),
        },
        {
            "logits": torch.tensor([[0.75, -1.0]]),
            "next_id": torch.tensor([0]),
        },
    ]
    candidate_captures = [
        {
            "logits": torch.tensor([[0.75, 0.0]]),
            "next_id": torch.tensor([2]),
        },
        {
            "logits": torch.tensor([[1.0, -1.0]]),
            "next_id": torch.tensor([2]),
        },
    ]

    rejected, details = _near_tie_mismatches(
        reference_captures,
        candidate_captures,
        tolerance=0.125,
    )

    assert rejected == [0]
    assert details == [(0, 0, 2, 0.25, 0.25, False)]


def compare(
    reference_dir,
    candidate_dir,
    world_size,
    quantized=False,
    report_only=False,
    reference_chunks=None,
    diagnostic_per_layer=False,
    greedy_tie_atol=0.0,
):
    reference_captures = [
        (
            _combine_reference_chunks(reference_chunks, rank)
            if reference_chunks
            else _load_capture(reference_dir, rank)
        )
        for rank in range(world_size)
    ]
    candidate_captures = [
        _load_capture(candidate_dir, rank) for rank in range(world_size)
    ]
    rejected_id_mismatches, id_mismatch_details = _near_tie_mismatches(
        reference_captures,
        candidate_captures,
        greedy_tie_atol,
    )
    for (
        row,
        reference_id,
        candidate_id,
        reference_margin,
        candidate_margin,
        accepted,
    ) in id_mismatch_details:
        print(
            f"next_id row={row} reference={reference_id} "
            f"candidate={candidate_id} "
            f"reference_margin={reference_margin:.6f} "
            f"candidate_margin={candidate_margin:.6f} "
            f"near_tie={accepted}"
        )

    for rank in range(world_size):
        reference = reference_captures[rank]
        candidate = candidate_captures[rank]

        reference_logits = reference["logits"].float()
        candidate_logits = candidate["logits"].float()
        if reference_logits.shape == candidate_logits.shape:
            expected_logits = reference_logits
        else:
            shard = reference_logits.shape[-1] // world_size
            expected_logits = reference_logits[
                :, rank * shard:(rank + 1) * shard
            ]
        if quantized:
            diff = candidate_logits - expected_logits
            cosine = F.cosine_similarity(
                candidate_logits.reshape(-1),
                expected_logits.reshape(-1),
                dim=0,
            ).item()
            rel_l2 = (
                torch.linalg.vector_norm(diff)
                / torch.linalg.vector_norm(expected_logits).clamp_min(1e-12)
            ).item()
            print(
                f"rank={rank} logits: cosine={cosine:.8f} "
                f"max_abs={diff.abs().max().item():.6e} rel_l2={rel_l2:.6e}"
            )
            if not report_only:
                assert cosine >= 0.999
                assert rel_l2 <= 0.05
        else:
            torch.testing.assert_close(
                candidate_logits, expected_logits, rtol=2e-2, atol=5e-2
            )
        ids_equal = torch.equal(candidate["next_id"], reference["next_id"])
        if not ids_equal:
            mismatch = (
                candidate["next_id"] != reference["next_id"]
            ).nonzero().reshape(-1)
            print(
                f"rank={rank} next_id mismatches={mismatch.tolist()} "
                f"reference={reference['next_id'][mismatch].tolist()} "
                f"candidate={candidate['next_id'][mismatch].tolist()}"
            )
        if not report_only:
            assert not rejected_id_mismatches

        for name in NAMES[2:]:
            expected = reference[name].float()
            actual = candidate[name].float()
            if actual.numel() == 0:
                assert actual.shape == expected.shape
                print(f"rank={rank} {name}: empty (shape={tuple(actual.shape)})")
                continue
            diff = actual - expected
            cosine = F.cosine_similarity(
                actual.reshape(-1), expected.reshape(-1), dim=0
            ).item()
            rel_l2 = (
                torch.linalg.vector_norm(diff)
                / torch.linalg.vector_norm(expected).clamp_min(1e-12)
            ).item()
            print(
                f"rank={rank} {name}: cosine={cosine:.8f} "
                f"max_abs={diff.abs().max().item():.6e} rel_l2={rel_l2:.6e}"
            )
            if quantized:
                if torch.linalg.vector_norm(expected) == 0:
                    if not report_only:
                        assert torch.linalg.vector_norm(actual) == 0
                else:
                    if not report_only:
                        assert cosine >= 0.999
                        assert rel_l2 <= 0.05
            else:
                torch.testing.assert_close(
                    candidate[name], reference[name], rtol=0, atol=0
                )

        reference_diagnostics = [
            name for name in DIAGNOSTIC_NAMES if name in reference
        ]
        candidate_diagnostics = [
            name for name in DIAGNOSTIC_NAMES if name in candidate
        ]
        if reference_diagnostics != candidate_diagnostics:
            raise AssertionError(
                "reference and candidate diagnostic tensors differ: "
                f"{reference_diagnostics} vs {candidate_diagnostics}"
            )
        if reference_diagnostics:
            if not torch.equal(
                reference["diagnostic_layers"],
                candidate["diagnostic_layers"],
            ):
                raise AssertionError("diagnostic layer selections differ")
            for name in reference_diagnostics:
                expected = reference[name]
                actual = candidate[name]
                if diagnostic_per_layer:
                    layers = reference["diagnostic_layers"].tolist()
                    for layer_index, layer in enumerate(layers):
                        expected_layer = expected[layer_index]
                        actual_layer = actual[layer_index]
                        if name == "diag_router_top_ids":
                            order_mismatches = int(
                                (actual_layer != expected_layer).sum()
                            )
                            membership_mismatches = (
                                _route_membership_mismatches(
                                    actual_layer, expected_layer
                                )
                            )
                            print(
                                f"rank={rank} layer={layer} {name}: "
                                f"same_membership={membership_mismatches == 0} "
                                f"membership_mismatches={membership_mismatches} "
                                f"order_mismatches={order_mismatches}"
                            )
                            if not report_only:
                                assert membership_mismatches == 0
                            continue
                        if name == "diag_router_top_weights":
                            expected_layer = _weights_in_expert_order(
                                reference["diag_router_top_ids"][layer_index],
                                expected_layer,
                            )
                            actual_layer = _weights_in_expert_order(
                                candidate["diag_router_top_ids"][layer_index],
                                actual_layer,
                            )
                        expected_f = expected_layer.float()
                        actual_f = actual_layer.float()
                        diff = actual_f - expected_f
                        expected_norm = torch.linalg.vector_norm(expected_f)
                        actual_norm = torch.linalg.vector_norm(actual_f)
                        if expected_norm == 0:
                            cosine = 1.0 if actual_norm == 0 else 0.0
                            rel_l2 = 0.0 if actual_norm == 0 else float("inf")
                        else:
                            cosine = F.cosine_similarity(
                                actual_f.reshape(-1),
                                expected_f.reshape(-1),
                                dim=0,
                            ).item()
                            rel_l2 = (
                                torch.linalg.vector_norm(diff) / expected_norm
                            ).item()
                        bf16_mismatches = ""
                        if torch.is_floating_point(expected):
                            mismatch_count = int(
                                (
                                    actual_layer.to(torch.bfloat16)
                                    != expected_layer.to(torch.bfloat16)
                                ).sum()
                            )
                            bf16_mismatches = (
                                f" bf16_mismatches={mismatch_count}"
                            )
                        print(
                            f"rank={rank} layer={layer} {name}: "
                            f"cosine={cosine:.8f} "
                            f"max_abs={diff.abs().max().item():.6e} "
                            f"rel_l2={rel_l2:.6e}{bf16_mismatches}"
                        )
                        if not report_only:
                            if quantized:
                                assert cosine >= 0.999
                                assert rel_l2 <= 0.05
                            else:
                                torch.testing.assert_close(
                                    actual_layer,
                                    expected_layer,
                                    rtol=0,
                                    atol=0,
                                )
                    continue
                if name == "diag_router_top_ids":
                    order_mismatches = int((actual != expected).sum())
                    membership_mismatches = _route_membership_mismatches(
                        actual, expected
                    )
                    print(
                        f"rank={rank} {name}: "
                        f"same_membership={membership_mismatches == 0} "
                        f"membership_mismatches={membership_mismatches} "
                        f"order_mismatches={order_mismatches}"
                    )
                    if not report_only:
                        assert membership_mismatches == 0
                    continue
                if name == "diag_router_top_weights":
                    expected = _weights_in_expert_order(
                        reference["diag_router_top_ids"], expected
                    )
                    actual = _weights_in_expert_order(
                        candidate["diag_router_top_ids"], actual
                    )
                expected_f = expected.float()
                actual_f = actual.float()
                diff = actual_f - expected_f
                expected_norm = torch.linalg.vector_norm(expected_f)
                actual_norm = torch.linalg.vector_norm(actual_f)
                if expected_norm == 0:
                    cosine = 1.0 if actual_norm == 0 else 0.0
                    rel_l2 = 0.0 if actual_norm == 0 else float("inf")
                else:
                    cosine = F.cosine_similarity(
                        actual_f.reshape(-1),
                        expected_f.reshape(-1),
                        dim=0,
                    ).item()
                    rel_l2 = (
                        torch.linalg.vector_norm(diff) / expected_norm
                    ).item()
                print(
                    f"rank={rank} {name}: cosine={cosine:.8f} "
                    f"max_abs={diff.abs().max().item():.6e} "
                    f"rel_l2={rel_l2:.6e}"
                )
                if quantized and not report_only:
                    assert cosine >= 0.999
                    assert rel_l2 <= 0.05

    if report_only:
        print("REPORT: completed comparison without enforcing gates")
    else:
        mode = "meet quantized gates" if quantized else "match"
        print(f"PASS: local logits, greedy IDs, and all carried states {mode}")


def capture(args):
    import torch_neuronx  # noqa: F401

    import model_dims as D
    import static_decode_35b as S

    if S.USE_MOE_FUSED_W8 or S.USE_MOE_OFFICIAL_FP8_REFERENCE:
        if args.mode != "sharded":
            raise ValueError("official FP8 expert captures require --mode sharded")
        if not args.expert_model_path:
            raise ValueError(
                "official FP8 expert captures require --expert-model-path or "
                "QWEN35_FP8_MODEL_PATH"
            )

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.neuron.current_device()

    D.load_from_config(os.path.join(args.model_path, "config.json"))
    D.NUM_LAYERS = args.num_layers
    D.NUM_GQA = sum(
        1 for layer in range(D.NUM_LAYERS) if D.layer_type(layer) == "gqa"
    )
    D.NUM_DELTANET = D.NUM_LAYERS - D.NUM_GQA
    weights = S.load_sharded_weights(
        args.model_path,
        rank,
        world_size,
        num_layers=D.NUM_LAYERS,
        expert_ckpt=args.expert_model_path,
    )
    model = S.StaticDecode35B(
        weights,
        max_seq_len=args.max_seq_len,
        world_size=world_size,
        batch_size=args.batch_size,
        rank=rank,
    ).to(device).eval()
    del weights

    diagnostic_layers = ()
    if args.diagnostic_layer:
        if args.diagnostic_layer == "all":
            diagnostic_layers = tuple(range(D.NUM_LAYERS))
        else:
            diagnostic_layers = tuple(
                sorted(
                    {
                        int(value)
                        for value in args.diagnostic_layer.split(",")
                        if value
                    }
                )
            )
        model.set_diagnostic_layers(diagnostic_layers)

    if args.mode == "segmented":
        if diagnostic_layers:
            raise ValueError("layer diagnostics require --mode sharded")
        model.setup_segments(1, compile_each=True)

        def step(token, position, *state):
            outputs = model(token, position, *state)
            return outputs[0], outputs[0].argmax(-1), *outputs[1:]
    else:
        assert S.USE_DECODE_FULLGRAPH
        assert S.USE_DECODE_SHARDED_LM_HEAD
        model.setup_segments(1, compile_each=False)
        if S.USE_GQA_STATEFUL_KV:
            compiled_target = (
                model.decode_step_diagnostics
                if diagnostic_layers
                else model.decode_step_stateful
            )
            compiled_step = torch.compile(
                compiled_target, backend="neuron", fullgraph=True, dynamic=False
            )

            def step(token, position, *state):
                outputs = compiled_step(token, position, state[0], state[1])
                return (
                    *outputs[:4],
                    model.decode_kv_k,
                    model.decode_kv_v,
                    *outputs[4:],
                )
        else:
            if diagnostic_layers:
                raise ValueError(
                    "layer diagnostics require GQA_STATEFUL_KV=1"
                )
            step = torch.compile(
                model.decode_step, backend="neuron", fullgraph=True, dynamic=False
            )

    td = D.tp_dims(world_size)
    vh = td["dn_v_heads"]
    qkv_dim = 2 * td["dn_k_heads"] * D.DN_K_DIM + vh * D.DN_V_DIM
    initial_capture = None
    initial_token_capture = None
    if args.initial_state_dir:
        initial_capture = _load_capture(args.initial_state_dir, rank)
        state_cpu = tuple(
            initial_capture[name]
            for name in ("deltanet", "conv", "kv_k", "kv_v")
        )
    else:
        state_cpu = (
            torch.zeros(
                D.NUM_DELTANET,
                args.batch_size,
                vh * D.DN_K_DIM,
                D.DN_V_DIM,
                dtype=torch.bfloat16,
            ),
            torch.zeros(
                D.NUM_DELTANET,
                args.batch_size,
                qkv_dim,
                D.DN_CONV_KERNEL - 1,
                dtype=torch.bfloat16,
            ),
            torch.zeros(
                D.NUM_GQA,
                args.batch_size,
                max(1, D.GQA_KV_HEADS // world_size),
                args.max_seq_len,
                D.GQA_HEAD_DIM,
                dtype=torch.bfloat16,
            ),
        )
        state_cpu = (
            state_cpu[0],
            state_cpu[1],
            state_cpu[2],
            torch.zeros_like(state_cpu[2]),
        )
    if args.initial_token_dir:
        initial_token_capture = _load_capture(args.initial_token_dir, rank)
    state = tuple(tensor.to(device) for tensor in state_cpu)
    if S.USE_GQA_STATEFUL_KV:
        model.decode_kv_k.copy_(state[2])
        model.decode_kv_v.copy_(state[3])
        state = (
            state[0],
            state[1],
            model.decode_kv_k,
            model.decode_kv_v,
        )
    if initial_token_capture is not None:
        token = initial_token_capture["next_id"].to(device)
    elif initial_capture is not None:
        token = initial_capture["next_id"].to(device)
    else:
        token = (
            (torch.arange(args.batch_size, dtype=torch.long) + args.batch_offset)
            * 7919
            + 100
        ).remainder(D.VOCAB).to(device)

    cross_target_compile = (
        os.environ.get("CROSS_TARGET_COMPILE_ONLY", "0") == "1"
    )
    marker_dir = os.environ.get(
        "CROSS_TARGET_MARKER_DIR", "/tmp/cross-target-compile"
    )
    marker_timeout = int(
        os.environ.get("CROSS_TARGET_MARKER_TIMEOUT_SECONDS", "7200")
    )
    compile_concurrency = int(
        os.environ.get("CROSS_TARGET_COMPILE_CONCURRENCY", str(world_size))
    )
    if not 1 <= compile_concurrency <= world_size:
        raise ValueError(
            "CROSS_TARGET_COMPILE_CONCURRENCY must be in "
            f"[1, {world_size}], got {compile_concurrency}"
        )
    if cross_target_compile and compile_concurrency < world_size:
        os.makedirs(marker_dir, exist_ok=True)
        wave_start = (rank // compile_concurrency) * compile_concurrency
        deadline = time.time() + marker_timeout
        while len(os.listdir(marker_dir)) < wave_start:
            if time.time() >= deadline:
                raise TimeoutError(
                    f"rank {rank} timed out waiting for cross-target "
                    f"compile wave {wave_start // compile_concurrency}"
                )
            time.sleep(1)

    with torch.no_grad():
        outputs = None
        step_captures = []
        try:
            for position_value in range(
                args.start_position,
                args.start_position + args.capture_steps,
            ):
                position = torch.tensor(
                    position_value, dtype=torch.long
                ).to(device)
                outputs = step(token, position, *state)
                token = outputs[1]
                state = outputs[2:6]
                if args.capture_every_step:
                    torch.neuron.synchronize()
                    output_names = NAMES + (
                        DIAGNOSTIC_NAMES if diagnostic_layers else ()
                    )
                    step_captures.append(
                        {
                            name: tensor.cpu()
                            for name, tensor in zip(output_names, outputs)
                        }
                    )
            torch.neuron.synchronize()
        except RuntimeError as exc:
            expected_cross_target_failure = (
                cross_target_compile and "Invalid NEFF" in str(exc)
            )
            if not expected_cross_target_failure:
                raise
            os.makedirs(marker_dir, exist_ok=True)
            open(os.path.join(marker_dir, f"rank{rank}.done"), "w").close()
            deadline = time.time() + marker_timeout
            while len(os.listdir(marker_dir)) < world_size:
                if time.time() >= deadline:
                    raise TimeoutError(
                        "timed out waiting for rank compile markers after "
                        f"{marker_timeout}s"
                    )
                time.sleep(1)
            print(f"rank={rank} compiled; skipped incompatible target load")
            return

    output_names = NAMES + (
        DIAGNOSTIC_NAMES if diagnostic_layers else ()
    )
    captured = {
        name: tensor.cpu() for name, tensor in zip(output_names, outputs)
    }
    if diagnostic_layers:
        captured["diagnostic_layers"] = torch.tensor(
            diagnostic_layers, dtype=torch.int32
        )
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(captured, os.path.join(args.output_dir, f"rank{rank}.pt"))
    for step_index, step_capture in enumerate(step_captures):
        if diagnostic_layers:
            step_capture["diagnostic_layers"] = captured[
                "diagnostic_layers"
            ]
        torch.save(
            step_capture,
            os.path.join(
                args.output_dir, f"rank{rank}.step{step_index}.pt"
            ),
        )

    if args.benchmark_iters:
        benchmark_token = token
        benchmark_state = state
        with torch.no_grad():
            for _ in range(3):
                benchmark_outputs = step(
                    benchmark_token,
                    position,
                    *benchmark_state,
                )
                benchmark_token = benchmark_outputs[1]
                benchmark_state = benchmark_outputs[2:6]
            torch.neuron.synchronize()
            dist.barrier()
            benchmark_start = time.perf_counter()
            for _ in range(args.benchmark_iters):
                benchmark_outputs = step(
                    benchmark_token,
                    position,
                    *benchmark_state,
                )
                benchmark_token = benchmark_outputs[1]
                benchmark_state = benchmark_outputs[2:6]
            torch.neuron.synchronize()
            dist.barrier()
        benchmark_seconds = time.perf_counter() - benchmark_start
        if rank == 0:
            tpot_ms = benchmark_seconds * 1000 / args.benchmark_iters
            throughput = args.batch_size * 1000 / tpot_ms
            print(
                f"BENCH BS={args.batch_size} seq={args.max_seq_len}: "
                f"TPOT {tpot_ms:.2f} ms/tok "
                f"(synced, {args.benchmark_iters} iter) | "
                f"throughput {throughput:.1f} tok/s"
            )

    dist.barrier()
    if rank == 0:
        print(
            f"captured mode={args.mode} next_ids="
            f"{captured['next_id'][:8].tolist()}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", default="/models/Qwen3.5-35B-A3B"
    )
    parser.add_argument(
        "--expert-model-path",
        default=os.environ.get("QWEN35_FP8_MODEL_PATH"),
    )
    parser.add_argument("--mode", choices=("segmented", "sharded"))
    parser.add_argument("--output-dir")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--capture-steps", type=int, default=2)
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=0,
        help="Run this many synchronized decode iterations after saving capture.",
    )
    parser.add_argument("--start-position", type=int, default=5)
    parser.add_argument(
        "--initial-state-dir",
        help="Restart from rank-local states and next_id in a prior capture.",
    )
    parser.add_argument(
        "--initial-token-dir",
        help="Override the restart token from another rank-local capture.",
    )
    parser.add_argument(
        "--capture-every-step",
        action="store_true",
        help="Also save rankR.stepN.pt after each decode step.",
    )
    parser.add_argument(
        "--batch-offset",
        type=int,
        default=0,
        help="Offset deterministic token rows for chunked high-batch references.",
    )
    parser.add_argument(
        "--diagnostic-layer",
        help="Comma-separated layer indices, or 'all', to emit stage tensors.",
    )
    parser.add_argument("--compare", nargs=2, metavar=("REFERENCE", "CANDIDATE"))
    parser.add_argument(
        "--compare-chunks",
        nargs="+",
        metavar="PATH",
        help="Compare CANDIDATE followed by two or more BS32 reference directories.",
    )
    parser.add_argument(
        "--quantized-compare",
        action="store_true",
        help="Compare W8 capture against official-FP8-dequantized BF16 reference.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print every comparison metric without enforcing correctness gates.",
    )
    parser.add_argument(
        "--diagnostic-per-layer",
        action="store_true",
        help="Report each selected diagnostic layer instead of one aggregate.",
    )
    parser.add_argument(
        "--greedy-tie-atol",
        type=float,
        default=0.0,
        help="Allow changed greedy IDs only when both logits prefer their "
        "choice over the other by no more than this absolute tolerance.",
    )
    args = parser.parse_args()
    if args.quantized_compare and not args.compare:
        if not args.compare_chunks:
            parser.error(
                "--quantized-compare requires --compare or --compare-chunks"
            )
    if args.compare and args.compare_chunks:
        parser.error("--compare and --compare-chunks are mutually exclusive")
    if args.compare:
        compare(
            *args.compare,
            args.world_size,
            quantized=args.quantized_compare,
            report_only=args.report_only,
            diagnostic_per_layer=args.diagnostic_per_layer,
            greedy_tie_atol=args.greedy_tie_atol,
        )
        return
    if args.compare_chunks:
        if len(args.compare_chunks) < 3:
            parser.error(
                "--compare-chunks requires CANDIDATE and at least two references"
            )
        compare(
            None,
            args.compare_chunks[0],
            args.world_size,
            quantized=args.quantized_compare,
            report_only=args.report_only,
            reference_chunks=args.compare_chunks[1:],
            diagnostic_per_layer=args.diagnostic_per_layer,
            greedy_tie_atol=args.greedy_tie_atol,
        )
        return
    if not args.mode or not args.output_dir:
        parser.error("--mode and --output-dir are required for capture")
    capture(args)


if __name__ == "__main__":
    main()
