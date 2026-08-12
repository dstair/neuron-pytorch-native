#!/usr/bin/env python3
"""Validate compiled decode_step convergence from a snapshot or real prefill."""

import math
import os
import sys
import time
from pathlib import Path


sys.path.insert(0, "/opt/xc/pysite")
sys.path.insert(
    0,
    os.environ.get(
        "Q27_REPO",
        os.path.dirname(os.path.abspath(__file__)),
    ),
)

# Eight ranks sharing one cold cache serialize on compile locks. This must run
# before importing torch-neuronx through static_decode.
_local_rank = os.environ.get("LOCAL_RANK", "0")
for _key in ("NEURON_COMPILE_CACHE_URL", "TORCH_NEURONX_NEFF_CACHE_DIR"):
    if os.environ.get(_key):
        os.environ[_key] = os.environ[_key].rstrip("/") + f"/r{_local_rank}"

import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend  # noqa: F401

import static_decode as sd
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


GOLDEN = [
    6312, 13, 271, 760, 3788, 314, 3712, 369, 1048,
    2894, 1521, 424, 8346, 586, 310, 3418, 279,
]
PROMPT = (
    "Geography, history, and civics are common subjects taught in schools all "
    "around the world, from small village classrooms to large universities in "
    "major cities. Students spend many years learning about the seven "
    "continents, the great oceans and seas, towering mountain ranges, long "
    "winding rivers, vast deserts, and the capital cities of dozens upon "
    "dozens of different countries across every region of the globe. They also "
    "study the important historical figures, famous inventors, influential "
    "writers, and the many political and cultural events that shaped nations "
    "over the long course of the centuries. For instance, essentially every "
    "schoolchild growing up in the United States eventually learns the well "
    "known fact that the country formally declared its independence from Great "
    "Britain in the summer of the year 1776, and that the very first President "
    "of the United States, the celebrated general who bravely led the "
    "Continental Army through the long years of the Revolutionary War and who "
    "later presided over the Constitutional Convention in Philadelphia, was a "
    "man named George"
)


def _neff_count() -> int:
    cache = os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR")
    if not cache:
        return -1
    return sum(1 for _ in Path(cache).rglob("*.neff"))


def _wait_for_neff_plateau(
    stable_polls: int = 3,
    poll_interval_s: float = 0.25,
    timeout_s: float = 120.0,
) -> int:
    """Wait until the rank-local cache count is unchanged for stable_polls."""
    count = _neff_count()
    if count < 0:
        raise RuntimeError(
            "TORCH_NEURONX_NEFF_CACHE_DIR must be set for convergence gating."
        )

    stable = 0
    deadline = time.monotonic() + timeout_s
    while stable < stable_polls:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"NEFF cache did not stabilize within {timeout_s:.1f}s."
            )
        time.sleep(poll_interval_s)
        current = _neff_count()
        if current == count:
            stable += 1
        else:
            count = current
            stable = 0
    return count


def _empty_states(device: torch.device, max_seq_len: int, batch_size: int):
    dtype = torch.bfloat16
    dn_dtype = torch.float32 if sd.USE_DN_F32_STATE else dtype
    dn_states = torch.zeros(
        sd.NUM_DELTANET,
        batch_size,
        sd.DN_V_HEADS * sd.DN_K_DIM,
        sd.DN_V_DIM,
        dtype=dn_dtype,
        device=device,
    )
    conv_states = torch.zeros(
        sd.NUM_DELTANET,
        batch_size,
        sd.DN_QKV_DIM,
        sd.DN_CONV_KERNEL - 1,
        dtype=dtype,
        device=device,
    )
    kv_k = torch.zeros(
        sd.NUM_GQA,
        batch_size,
        max_seq_len,
        sd.GQA_HEAD_DIM,
        dtype=dtype,
        device=device,
    )
    kv_v = torch.zeros_like(kv_k)
    return dn_states, conv_states, kv_k, kv_v


def _repeat_snapshot_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[1] != 1:
        raise RuntimeError(
            f"Expected a BS=1 snapshot tensor, got shape {tuple(tensor.shape)}."
        )
    repeats = [1] * tensor.ndim
    repeats[1] = batch_size
    return tensor.repeat(*repeats)


def main() -> None:
    if not sd.USE_DECODE_COMPILE_STEP:
        raise SystemExit("Set DECODE_COMPILE_STEP=1.")

    dist.init_process_group(backend="xla")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise SystemExit(
            f"dev_stepcompile.py is a TP8 acceptance gate; got TP={world_size}."
        )
    device = xm.xla_device()
    model_path = os.environ["QWEN27_MODEL_PATH"]
    ngen = int(os.environ.get("NGEN", "16"))
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    if batch_size < 1:
        raise SystemExit(f"BATCH_SIZE must be positive; got {batch_size}.")
    fullgraph = os.environ.get("DECODE_STEP_FULLGRAPH", "1") == "1"
    if (
        batch_size > 1
        and sd.USE_DECODE_VIA_CHUNKED
        and (fullgraph or sd.DECODE_STEP_BREAK_EVERY == 0)
    ):
        raise SystemExit(
            "BATCH_SIZE>1 with DECODE_VIA_CHUNKED=1 requires "
            "DECODE_STEP_FULLGRAPH=0 and DECODE_STEP_BREAK_EVERY>0; "
            "the BS>1 acceptance path uses bounded compiled spans."
        )
    resume_snapshot = os.environ.get("RESUME_PREFILL_SNAPSHOT", "1") == "1"
    max_seq_len = 512

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=False)[-128:]
    seq_len = len(prompt_ids)
    if seq_len != 128:
        raise RuntimeError(f"George prompt must contain 128 tokens; got {seq_len}.")
    if ngen + 1 > len(GOLDEN):
        raise RuntimeError(
            f"NGEN={ngen} requires at least {ngen + 1} golden tokens; "
            f"have {len(GOLDEN)}."
        )
    if not resume_snapshot and not sd.USE_PREFILL_SEGMENTED:
        raise SystemExit(
            "RESUME_PREFILL_SNAPSHOT=0 requires PREFILL_SEGMENTED=1."
        )
    if not resume_snapshot and batch_size != 1:
        raise SystemExit(
            "BATCH_SIZE>1 requires RESUME_PREFILL_SNAPSHOT=1; prefill is BS=1."
        )

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    sd.shard_model(model, rank, world_size)
    weights = sd.extract_weights(model)
    sd.apply_vocab_sharding(weights, rank, world_size)
    module = sd.StaticDecodeModule(
        weights, max_seq_len, world_size, batch_size=batch_size, rank=rank
    ).to(device).eval()
    expected_prepacked = 3 * sd.NUM_LAYERS + 1
    actual_prepacked = len(module._prepacked_linear_w)
    if sd.USE_BF16_PREPACKED_LINEAR:
        assert actual_prepacked == expected_prepacked
    else:
        assert actual_prepacked == 0
    del model, weights
    dist.barrier()

    if resume_snapshot:
        snapshot_template = os.environ.get("PREFILL_SNAPSHOT_PATH")
        if not snapshot_template:
            raise SystemExit(
                "RESUME_PREFILL_SNAPSHOT=1 requires PREFILL_SNAPSHOT_PATH. "
                "Set it in the gitignored .env file; {rank} is supported."
            )
        snapshot_path = snapshot_template.format(rank=rank)
        snapshot = torch.load(snapshot_path, map_location="cpu")
        dn_states = _repeat_snapshot_batch(snapshot["dn"], batch_size).to(device)
        conv_states = _repeat_snapshot_batch(snapshot["cv"], batch_size).to(device)
        kv_k = _repeat_snapshot_batch(snapshot["kk"], batch_size).to(device)
        kv_v = _repeat_snapshot_batch(snapshot["vv"], batch_size).to(device)
        first = int(snapshot["first"])
    else:
        dn_states, conv_states, kv_k, kv_v = _empty_states(
            device, max_seq_len, batch_size
        )
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            (
                prefill_logits,
                dn_states,
                conv_states,
                kv_k,
                kv_v,
            ) = module.prefill(
                input_ids, dn_states, conv_states, kv_k, kv_v
            )
            first_tensor = prefill_logits[0].max(dim=-1).indices.to(torch.long)
            xm.wait_device_ops()
            first = int(first_tensor.cpu())

    import torch._dynamo as dynamo

    dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, 256)
    decode_step = torch.compile(
        module.decode_step,
        backend=sd._COMPILE_BACKEND,
        fullgraph=fullgraph,
        dynamic=False,
    )

    if rank == 0:
        print(
            f"trial fullgraph={fullgraph} "
            f"break_every={sd.DECODE_STEP_BREAK_EVERY} ngen={ngen} "
            f"batch_size={batch_size} "
            f"bf16_prepacked_linear={sd.USE_BF16_PREPACKED_LINEAR} "
            f"dnbatched_v2={os.environ.get('DNBATCHED_V2', '0') == '1'} "
            f"gqa_tail={sd.USE_GQA_TAIL} "
            f"prepacked_tensors={actual_prepacked} "
            f"resume_prefill_snapshot={resume_snapshot}",
            flush=True,
        )
        print(
            f"prefill first={first} {tokenizer.decode([first])!r}",
            flush=True,
        )

    token = torch.full(
        (batch_size,), first, dtype=torch.long, device=device
    )
    position = torch.tensor(seq_len, dtype=torch.long, device=device)
    generated_by_row = [[first] for _ in range(batch_size)]
    times_ms = []
    neff_deltas = []
    selector_matches = []

    with torch.no_grad():
        for step in range(ngen):
            before = _neff_count()
            if before < 0:
                raise RuntimeError(
                    "TORCH_NEURONX_NEFF_CACHE_DIR must be set."
                )
            started = time.time()
            if step == 0 and not fullgraph and sd.DECODE_STEP_BREAK_EVERY > 0:
                # openxla specializes one Dynamo resume graph only on the
                # second entry. Inputs remain unchanged because decode_step
                # clones all state, so discard this compile probe and run the
                # real first token below from the same snapshot state.
                probe = decode_step(
                    token, position, dn_states, conv_states, kv_k, kv_v
                )
                xm.wait_device_ops()
                probe[1].detach().cpu()
                probe[2].detach().cpu()
            (
                logits,
                token,
                position,
                dn_states,
                conv_states,
                kv_k,
                kv_v,
            ) = decode_step(
                token, position, dn_states, conv_states, kv_k, kv_v
            )
            xm.wait_device_ops()
            if step == 0:
                # With graph breaks, next_position can be a trailing lazy
                # subgraph whose first consumer is otherwise the next decode.
                # Materialize it before declaring the cold-step cache stable.
                position.detach().cpu()
            elapsed_ms = (time.time() - started) * 1000.0
            token_host = token.detach().cpu()
            logits_argmax = logits.detach().cpu().float().argmax(dim=-1)
            if step == 0:
                after = _wait_for_neff_plateau()
                # Keep compile skew out of the first warm-step timing. Every
                # rank has independently observed a stable cache before release.
                dist.barrier()
            else:
                after = _neff_count()

            for row in range(batch_size):
                generated_by_row[row].append(int(token_host[row]))
            selector_matches.append(torch.equal(token_host, logits_argmax))
            times_ms.append(elapsed_ms)
            neff_deltas.append(after - before)

            if rank == 0:
                print(
                    f"step={step} token={int(token_host[0])} "
                    f"logits_argmax={int(logits_argmax[0])} "
                    f"selector_match={selector_matches[-1]} "
                    f"ms={elapsed_ms:.1f} neff_delta={neff_deltas[-1]} "
                    f"neff_total={after}",
                    flush=True,
                )

    generated = generated_by_row[0]
    expected = GOLDEN[: len(generated)]
    coherent = all(row == expected for row in generated_by_row)
    selector_agreement = all(selector_matches)
    converged = all(delta == 0 for delta in neff_deltas[1:])
    plateau_step = next(
        (
            step
            for step in range(len(neff_deltas))
            if all(delta == 0 for delta in neff_deltas[step:])
        ),
        len(neff_deltas),
    )
    steady_times = [
        elapsed
        for step, (elapsed, delta) in enumerate(zip(times_ms, neff_deltas))
        if step > 0 and delta == 0
    ]
    warm_ms = (
        sum(steady_times) / len(steady_times)
        if steady_times
        else float("nan")
    )

    generated_tensor = torch.tensor(
        generated_by_row, dtype=torch.float32, device=device
    ).unsqueeze(0)
    all_generated = xm.all_gather(generated_tensor, dim=0)
    local_metrics = torch.tensor(
        [
            float(coherent),
            float(selector_agreement),
            float(converged),
            float(plateau_step),
            warm_ms,
        ],
        dtype=torch.float32,
        device=device,
    ).reshape(1, -1)
    all_metrics = xm.all_gather(local_metrics, dim=0)
    xm.mark_step()
    all_generated_host = all_generated.cpu().to(torch.long)
    all_metrics_host = all_metrics.cpu()

    rank_coherent = all_metrics_host[:, 0].to(torch.bool)
    rank_selector = all_metrics_host[:, 1].to(torch.bool)
    rank_converged = all_metrics_host[:, 2].to(torch.bool)
    rank_plateau = all_metrics_host[:, 3].to(torch.long)
    rank_warm_ms = all_metrics_host[:, 4]
    cross_rank_tokens = all(
        torch.equal(all_generated_host[r], all_generated_host[0])
        for r in range(1, world_size)
    )
    batch_rows_identical = all(
        row == generated_by_row[0] for row in generated_by_row[1:]
    )
    all_coherent = (
        bool(rank_coherent.all())
        and cross_rank_tokens
        and batch_rows_identical
    )
    all_selector = bool(rank_selector.all())
    all_converged = bool(rank_converged.all())
    worst_plateau = int(rank_plateau.max())
    worst_warm_ms = float(rank_warm_ms.max())
    throughput = batch_size * 1000.0 / worst_warm_ms
    all_tpot_finite = all(math.isfinite(float(v)) for v in rank_warm_ms)
    passed = (
        all_coherent
        and all_selector
        and all_converged
        and worst_plateau <= 1
        and all_tpot_finite
    )

    if rank == 0:
        print(f"generated={generated}", flush=True)
        print(
            f"continuation={tokenizer.decode(generated)!r}",
            flush=True,
        )
        print(
            f"RESULT pass={passed} coherent_all_ranks={all_coherent} "
            f"selector_agreement_all_ranks={all_selector} "
            f"converged_after_step0_all_ranks={all_converged} "
            f"compile_plateau_step_worst_rank={worst_plateau} "
            f"warm_tpot_ms_worst_rank={worst_warm_ms:.1f} "
            f"throughput_tok_s={throughput:.1f} "
            f"warm_tpot_finite_all_ranks={all_tpot_finite} "
            f"cold_step_ms_rank0={times_ms[0]:.1f}",
            flush=True,
        )
        for r in range(world_size):
            print(
                f"rank={r} coherent={bool(rank_coherent[r])} "
                f"selector_agreement={bool(rank_selector[r])} "
                f"converged_after_step0={bool(rank_converged[r])} "
                f"compile_plateau_step={int(rank_plateau[r])} "
                f"warm_tpot_ms={float(rank_warm_ms[r]):.1f}",
                flush=True,
            )

    dist.barrier()
    dist.destroy_process_group()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
