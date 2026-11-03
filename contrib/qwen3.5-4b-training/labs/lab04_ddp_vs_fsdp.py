"""DDP vs FSDP2: the regime lesson. Which one is right depends on WHAT you are training.

The single most useful finding from the earlier bring-up of this model, and the thing this
lab exists to reproduce first-hand:

  * LoRA + FSDP FULL_SHARD was measurably SLOWER than DDP (119.9 vs 150.2 tok/s at world=4).
    FSDP shards a frozen 4.2 B base that it must then all-gather on every forward, while the
    optimizer state it exists to shard is ~3 M parameters. Pure communication overhead.
  * Full fine-tune + FSDP was the ONLY configuration that ran at all. 47 GiB of
    weights+grads+AdamW against 24 GiB per core: single-core full-FT OOMs. Here FSDP is not
    an optimisation, it is the enabler.

So: LoRA -> DDP. Full fine-tune -> FSDP. Getting this backwards costs ~25% throughput in one
direction and the entire run in the other.

    # run the matrix (needs labs/lab_collectives.py to pass at this world size first)
    LNC=2 WORLD=4 ./env/container.sh exec torchrun --nproc_per_node 4 \
        labs/lab04_ddp_vs_fsdp.py --strategy ddp   --lora    --layers 8 --seq 512 --steps 6
    LNC=2 WORLD=4 ./env/container.sh exec torchrun --nproc_per_node 4 \
        labs/lab04_ddp_vs_fsdp.py --strategy fsdp2 --lora    --layers 8 --seq 512 --steps 6
    LNC=2 WORLD=4 ./env/container.sh exec torchrun --nproc_per_node 4 \
        labs/lab04_ddp_vs_fsdp.py --strategy fsdp2 --full-ft --layers 8 --seq 512 --steps 6

Prints `LAB04 OK` on success.
"""

import argparse
import json
import pathlib
import statistics

import torch

from sqlft.gdn import GDN_CHOICES, GDN_HELP

RUNS = pathlib.Path("/work/runs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True,
                    choices=("single", "ddp", "fsdp1", "fsdp1_noshard", "fsdp2", "fsdp2_tp"))
    ap.add_argument("--tp", type=int, default=2, help="tensor-parallel width for fsdp2_tp")
    # No default: an implicit truncation is how you accidentally report an 8-layer number as
    # a full-model one. Callers must say what they mean, as sqlft.train does.
    ap.add_argument("--layers", type=int, required=True,
                    help="number of decoder layers; 32 = the full model")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--gdn", choices=GDN_CHOICES, default="torch", help=GDN_HELP)
    ap.add_argument("--check-finite", action="store_true",
                    help="localise NaN: check the loss before backward and the grads after")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--lora", action="store_true")
    group.add_argument("--full-ft", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    from sqlft import data
    from sqlft.device import Timer, device_memory_bytes, get_device, set_dynamo_limits, sync
    from sqlft.model import load_model
    from sqlft.parallel import apply_strategy, clip_grad_norm_, init_distributed, is_main, main_first

    rank, world, _ = init_distributed()
    set_dynamo_limits()
    dev = get_device("neuron")

    tp = args.tp if "tp" in args.strategy else 1
    if tp > 1 and args.lora:
        # The TP plan targets nn.Linear leaves, and LoRA has already replaced them with
        # LoRALinear wrappers whose lora_B would also need sharding to match the local output.
        raise SystemExit("tensor parallel is full-FT only here: LoRA + TP is not implemented")
    # build_mesh lays the (dp, tp) mesh out row-major, so the tp ranks of one dp replica are
    # contiguous and rank // tp is the dp index.
    dp_rank, dp_world = rank // tp, world // tp

    tok = data.get_tokenizer()
    rows, _ = data.split_rows(data.load_rows())
    rows = data.filter_parseable(rows, report=is_main())
    # Each dp replica draws a different shard of the data -- that IS data parallelism. Seeding
    # by dp rank is the cheap way to get it without a DistributedSampler; the sampler is the
    # right answer for a real epoch (no overlap, reproducible), and lab00 covers that. Ranks
    # inside one tp group MUST see the same batch: they each hold a slice of the same layers.
    loader = data.build_loader(rows, tok, args.seq, args.bs, num_workers=1, seed=dp_rank)
    batches = data.infinite(loader)

    # Rank 0 compiles into the shared NEFF cache first. Eight ranks entering neuronx-cc at
    # once exhausted a 124 GB host in the prior art, and this box has exactly 124 GB.
    with main_first():
        model = load_model(
            layers=args.layers, gdn=args.gdn,
            lora=args.lora, gradient_checkpointing=args.grad_ckpt,
            device=dev, verbose=is_main(),
        )

    model, mesh = apply_strategy(model, args.strategy, world, args.tp)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)

    losses, times = [], []
    # AGGREGATE across dp replicas. tp ranks share one batch, so counting them would multiply
    # the throughput by tp for no extra data processed.
    tokens_per_step = args.bs * args.seq * dp_world

    for step in range(args.steps):
        with Timer(device=dev) as t:
            opt.zero_grad(set_to_none=True)
            batch = {k: v.to(dev) for k, v in next(batches).items()}
            out = model(**batch)

            if args.check_finite:
                # Stage 1: is the FORWARD already broken? Distinguishes "the model cannot
                # even evaluate in bf16" from "the gradients blow up".
                lv = float(out.loss.detach().float().cpu())
                print(f"[rank{rank}] step {step} forward loss = {lv}"
                      f"{'  <-- NOT FINITE' if lv != lv or abs(lv) == float('inf') else ''}",
                      flush=True)

            out.loss.backward()

            if args.check_finite:
                # Stage 2: which parameters' gradients are non-finite? Naming them is the
                # whole point -- "somewhere in 4.2B params" is not actionable.
                bad = []
                for n, p in model.named_parameters():
                    if p.grad is None or not p.requires_grad:
                        continue
                    g = p.grad
                    g = g.to_local() if hasattr(g, "to_local") else g
                    if not bool(torch.isfinite(g).all().cpu()):
                        bad.append(n)
                print(f"[rank{rank}] step {step} non-finite grads: {len(bad)}"
                      + (f"  first: {bad[:6]}" if bad else ""), flush=True)

            # clip_grad_norm_ on an already-inf gradient produces NaN everywhere rather than
            # saving the step, so with --check-finite we want to see the state before it runs.
            gnorm = clip_grad_norm_(params, 1.0)
            if args.check_finite:
                print(f"[rank{rank}] step {step} grad-norm before clip = "
                      f"{float(gnorm.float().cpu())}", flush=True)
            opt.step()
            loss = float(out.loss.detach().float().cpu())
        losses.append(loss)
        times.append(t.seconds)
        if is_main():
            mem = device_memory_bytes()
            mem_s = f"  peak {mem['peak'] / 2**30:.2f} GiB/rank" if mem else ""
            print(f"step {step:3d}  loss {loss:8.4f}  {t.seconds:7.3f}s  "
                  f"{tokens_per_step / t.seconds:8.1f} agg tok/s{mem_s}", flush=True)
        if loss != loss:
            raise RuntimeError(f"loss went NaN at step {step}")

    warm = times[args.warmup:] or times
    result = {
        "tag": args.tag or f"lab04_{args.strategy}_{'lora' if args.lora else 'fullft'}_w{world}",
        "strategy": args.strategy, "world": world, "tp": tp, "dp": dp_world,
        "mode": "lora" if args.lora else "full_ft",
        "args": vars(args),
        "losses": losses,
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "compile_s": times[0],
        "warm_s_per_step": statistics.mean(warm),
        "agg_tok_s": tokens_per_step / statistics.mean(warm),
        "peak_hbm_bytes_per_rank": device_memory_bytes().get("peak"),
    }

    sync(dev)
    if is_main():
        print(f"\n{result['strategy']}/{result['mode']} world={world}: "
              f"compile {result['compile_s']:.1f}s  warm {result['warm_s_per_step']:.3f}s/step  "
              f"{result['agg_tok_s']:.1f} agg tok/s")
        print(f"loss {result['loss_first']:.4f} -> {result['loss_last']:.4f}")
        RUNS.mkdir(parents=True, exist_ok=True)
        out_path = RUNS / (result["tag"] + ".json")
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"wrote {out_path}")
        print("LAB04 OK")

    torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
