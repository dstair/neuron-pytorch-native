"""The training loop. Explicit forward / backward / step -- no HF Trainer, no accelerate.

Rubric coverage: autograd end to end on a real model, and the one place where every other
lesson (dataloader, LoRA, activation checkpointing, accumulation, parallelism) plugs in.

    ./env/container.sh exec python3 -m sqlft.train --layers 4 --seq 512 --bs 1 --steps 8 --lora

Prints `TRAIN OK` on success. Gate on that marker, not on the exit code: these beta
containers SIGABRT during teardown after printing perfectly good results.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import torch

from sqlft.gdn import GDN_CHOICES, GDN_HELP

RUNS = pathlib.Path("/work/runs")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None, help="truncate the stack (None = all 32)")
    ap.add_argument("--seq", type=int, default=512, help="must be a multiple of 128")
    ap.add_argument("--bs", type=int, default=1, help="micro-batch size")
    ap.add_argument("--steps", type=int, default=8, help="optimizer steps")
    ap.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2, help="steps excluded from the timing average")

    ap.add_argument("--lora", action="store_true", help="LoRA instead of full fine-tune")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)

    ap.add_argument("--gdn", choices=GDN_CHOICES, default="torch", help=GDN_HELP)
    ap.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing")
    ap.add_argument("--packed", action="store_true", help="pack examples, no padding waste")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-filter", action="store_true",
                    help="keep the ~2.8%% of train rows whose gold SQL does not parse")

    ap.add_argument("--device", default="neuron")
    ap.add_argument("--tag", default=None, help="name for runs/<tag>.json")
    return ap


def train(args) -> dict:
    from sqlft import data
    from sqlft.device import Timer, device_memory_bytes, get_device, set_dynamo_limits, sync
    from sqlft.model import load_model, trainable_parameters

    set_dynamo_limits()
    dev = get_device(args.device)

    tok = data.get_tokenizer()
    rows, _ = data.split_rows(data.load_rows())
    if not args.no_filter:
        rows = data.filter_parseable(rows, report=True)
    loader = data.build_loader(
        rows, tok, args.seq, args.bs,
        packed=args.packed, num_workers=args.workers, report=True,
    )
    batches = data.infinite(loader)

    model = load_model(
        layers=args.layers,
        gdn=args.gdn,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        gradient_checkpointing=args.grad_ckpt,
        device=dev,
    )

    params = trainable_parameters(model)
    if not params:
        raise RuntimeError("no trainable parameters -- did apply_lora freeze everything?")
    # foreach=False: the fused/foreach multi-tensor paths are the ones most likely to hit an
    # unimplemented op on a young backend, and the fallback is silent.
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)

    losses: list[float] = []
    step_times: list[float] = []
    tokens_per_step = args.bs * args.seq * args.grad_accum

    for step in range(args.steps):
        with Timer(device=dev) as t:
            opt.zero_grad(set_to_none=True)
            micro_losses = []
            for _ in range(args.grad_accum):
                batch = {k: v.to(dev) for k, v in next(batches).items()}
                out = model(**batch)
                # Divided by grad_accum so the effective LR does not scale with accumulation.
                (out.loss / args.grad_accum).backward()
                micro_losses.append(out.loss.detach())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            loss = float(torch.stack(micro_losses).mean().float().cpu())

        losses.append(loss)
        step_times.append(t.seconds)
        mem = device_memory_bytes()
        mem_s = f"  peak {mem['peak'] / 2**30:.2f} GiB" if mem else ""
        print(f"step {step:3d}  loss {loss:8.4f}  {t.seconds:7.3f}s  "
              f"{tokens_per_step / t.seconds:8.1f} tok/s{mem_s}", flush=True)

        if loss != loss:  # NaN
            raise RuntimeError(f"loss went NaN at step {step}")

    # Trend over halves, not first-vs-last. At bs=1 a single batch swings the loss by more
    # than a few steps of training move it, so `losses[-1] < losses[0]` fails on healthy runs
    # and passes on unhealthy ones. NaN is still a hard failure, checked per step above.
    half = max(1, len(losses) // 2)
    first_half = statistics.mean(losses[:half])
    second_half = statistics.mean(losses[-half:])

    warm = step_times[args.warmup:] or step_times
    result = {
        "tag": args.tag or "train",
        "args": vars(args),
        "losses": losses,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_first_half_mean": first_half,
        "loss_second_half_mean": second_half,
        "loss_decreased": second_half < first_half,
        "compile_s": step_times[0],
        "warm_s_per_step": statistics.mean(warm),
        "warm_tok_s": tokens_per_step / statistics.mean(warm),
        "tokens_per_step": tokens_per_step,
        "peak_hbm_bytes": device_memory_bytes().get("peak"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    sync(dev)
    print(f"\ncompile(step 0) {result['compile_s']:.1f}s   "
          f"warm {result['warm_s_per_step']:.3f}s/step   "
          f"{result['warm_tok_s']:.1f} tok/s")
    print(f"loss {result['loss_first']:.4f} -> {result['loss_last']:.4f}   "
          f"mean first half {first_half:.4f} -> second half {second_half:.4f}"
          f"  ({'decreasing' if result['loss_decreased'] else 'NOT DECREASING'})")

    RUNS.mkdir(parents=True, exist_ok=True)
    out_path = RUNS / f"{result['tag']}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {out_path}")

    if not result["loss_decreased"]:
        # A warning, not a failure: over a handful of bs=1 steps this is weak evidence either
        # way. The hard gates are the per-step NaN check and the parity/self-tests.
        print("WARNING: loss did not trend down over this run -- too few steps to conclude, "
              "but do not quote it as convergence evidence")
    return result


if __name__ == "__main__":
    train(build_argparser().parse_args())
    print("TRAIN OK")
