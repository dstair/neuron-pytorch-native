"""The actual fine-tune: full 32-layer Qwen3.5-4B on duckdb-text2sql, DDP + LoRA.

This is the run everything else was scaffolding for. Uses the configuration the labs
established as best:

  * full 32 layers, `--gdn nki-v15-split16` (upstream v15 with a 16-wide diagonal solve:
    correct, and the fastest build; `nki-patched` produced the published epoch)
  * DDP, not FSDP -- for LoRA the base is frozen, so sharding it costs an all-gather per
    layer per forward and buys nothing except memory (lab04)
  * gradient accumulation, because bs=1 x 4 ranks is far too noisy to read a loss curve from
  * warmup + cosine decay; lr 1e-4 is fine for LoRA (it diverges for full-FT, see NOTES)
  * DistributedSampler, so one epoch covers the data once instead of ranks overlapping

    LNC=2 WORLD=4 ./env/container.sh exec torchrun --nproc_per_node 4 \
        labs/lab07_finetune.py --steps 600 --grad-accum 4 --tag ft_v1

Prints `FINETUNE OK`. Writes the adapter to runs/<tag>/adapter_final.pt and a loss history to
runs/<tag>.json.
"""

import argparse
import json
import math
import pathlib
import statistics
import time

import torch

from sqlft.gdn import GDN_CHOICES, GDN_HELP

RUNS = pathlib.Path("/work/runs")


def lr_at(step: int, total: int, base: float, warmup: int, floor_frac: float = 0.1) -> float:
    """Linear warmup then cosine decay to floor_frac * base.

    Warmup matters more than usual here: LoRA's B starts at zero, so the first steps see an
    adapter that contributes nothing and a gradient that is entirely a function of how wrong
    the base is -- a large first step is mostly noise.
    """
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=1, help="micro-batch per rank")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600, help="optimizer steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-targets", choices=("all", "mixers"), default="all",
                    help="mixers = attention + GDN projections only, no MLP")
    ap.add_argument("--gdn", choices=GDN_CHOICES, default="nki-v15-split16")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--packed", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="lab07_finetune")
    args = ap.parse_args()

    from sqlft import data
    from sqlft.device import Timer, device_memory_bytes, get_device, set_dynamo_limits, sync
    from sqlft.lora import save_lora
    from sqlft.model import load_model, trainable_parameters
    from sqlft.parallel import apply_ddp, init_distributed, is_main, main_first

    rank, world, _ = init_distributed()
    set_dynamo_limits()
    dev = get_device("neuron")
    out_dir = RUNS / args.tag

    tok = data.get_tokenizer()
    train_rows, eval_rows = data.split_rows(data.load_rows())
    train_rows = data.filter_parseable(train_rows, report=is_main())
    loader = data.build_loader(
        train_rows, tok, args.seq, args.bs,
        packed=args.packed, num_workers=args.workers,
        distributed=True, seed=0, report=is_main(),
    )
    epoch_steps = max(1, len(loader) // args.grad_accum)
    if is_main():
        tokens_per_step = args.bs * args.seq * args.grad_accum * world
        print(f"train rows {len(train_rows)}  batches/rank/epoch {len(loader)}  "
              f"optimizer steps/epoch {epoch_steps}")
        print(f"effective batch {args.bs * args.grad_accum * world} sequences = "
              f"{tokens_per_step} tokens/step; {args.steps} steps = "
              f"{args.steps * tokens_per_step / 1e6:.2f}M tokens "
              f"({args.steps / epoch_steps:.2f} epochs)", flush=True)

    with main_first():
        model = load_model(
            layers=args.layers,
            gdn=args.gdn,
            lora=True, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_targets=args.lora_targets,
            gradient_checkpointing=args.grad_ckpt,
            device=dev, verbose=is_main(),
        )
    inner = model
    model = apply_ddp(model)

    params = trainable_parameters(model)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
                            foreach=False)

    history: list[dict] = []
    window: list[float] = []
    times: list[float] = []
    batches = data.infinite(loader)
    tokens_per_step = args.bs * args.seq * args.grad_accum * world
    t0 = time.perf_counter()

    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for gp in opt.param_groups:
            gp["lr"] = lr

        with Timer(device=dev) as t:
            opt.zero_grad(set_to_none=True)
            micro = []
            for micro_i in range(args.grad_accum):
                batch = {k: v.to(dev) for k, v in next(batches).items()}
                # DDP all-reduces on every backward by default; suppress it for all but the
                # last micro-batch or we pay grad_accum x the communication for one step.
                last = micro_i == args.grad_accum - 1
                ctx = model.no_sync() if (not last and hasattr(model, "no_sync")) else None
                if ctx is not None:
                    with ctx:
                        out = model(**batch)
                        (out.loss / args.grad_accum).backward()
                else:
                    out = model(**batch)
                    (out.loss / args.grad_accum).backward()
                micro.append(out.loss.detach())
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            loss = float(torch.stack(micro).mean().float().cpu())

        if loss != loss:
            raise RuntimeError(f"loss went NaN at step {step}")
        window.append(loss)
        times.append(t.seconds)

        if is_main() and (step % args.log_every == 0 or step == args.steps - 1):
            mean = statistics.mean(window)
            mem = device_memory_bytes()
            elapsed = time.perf_counter() - t0
            eta = elapsed / (step + 1) * (args.steps - step - 1)
            print(f"step {step:5d}/{args.steps}  loss {mean:7.4f}  lr {lr:.2e}  "
                  f"|g| {float(gnorm.float().cpu()):6.3f}  {t.seconds:5.2f}s  "
                  f"{tokens_per_step / t.seconds:7.1f} tok/s  "
                  f"peak {mem.get('peak', 0) / 2**30:5.2f} GiB  "
                  f"eta {eta / 60:5.1f}m", flush=True)
            history.append({"step": step, "loss_mean": mean, "lr": lr,
                            "s_per_step": t.seconds})
            window = []

        if is_main() and args.save_every and (step + 1) % args.save_every == 0:
            n = save_lora(inner, out_dir / f"adapter_step{step + 1}.pt")
            print(f"  saved adapter ({n / 1e6:.1f} MB) at step {step + 1}", flush=True)

    sync(dev)
    if is_main():
        size = save_lora(inner, out_dir / "adapter_final.pt")
        warm = times[5:] or times
        result = {
            "tag": args.tag, "args": vars(args), "world": world,
            "history": history,
            "loss_first_logged": history[0]["loss_mean"] if history else None,
            "loss_last_logged": history[-1]["loss_mean"] if history else None,
            "warm_s_per_step": statistics.mean(warm),
            "agg_tok_s": tokens_per_step / statistics.mean(warm),
            "tokens_seen": args.steps * tokens_per_step,
            "epochs": args.steps / epoch_steps,
            "peak_hbm_bytes_per_rank": device_memory_bytes().get("peak"),
            "adapter": str(out_dir / "adapter_final.pt"),
            "adapter_bytes": size,
            "wall_s": time.perf_counter() - t0,
        }
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / f"{args.tag}.json").write_text(json.dumps(result, indent=2, default=str))
        print(f"\nloss {result['loss_first_logged']:.4f} -> {result['loss_last_logged']:.4f}"
              f"   {result['agg_tok_s']:.1f} agg tok/s"
              f"   {result['tokens_seen'] / 1e6:.2f}M tokens"
              f"   {result['wall_s'] / 60:.1f} min")
        print(f"adapter -> {result['adapter']} ({size / 1e6:.1f} MB)")
        print("FINETUNE OK")

    torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
