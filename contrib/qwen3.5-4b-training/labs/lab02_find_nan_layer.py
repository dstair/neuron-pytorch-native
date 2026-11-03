"""Where do the activations blow up? Forward hooks on all 32 layers, one NaN-ing batch.

32-layer forward is finite on the seed-0 first batch (loss 1.0299) and NaN on others, at
8 layers everything is fine, and it happens identically on 1 rank and 4. So it is depth x
data, in the forward. This walks every decoder layer and reports the magnitude of its output
so the blow-up can be located rather than guessed at.

    ./env/container.sh exec python3 labs/lab02_find_nan_layer.py --layers 32 --seq 512
    ./env/container.sh exec python3 labs/lab02_find_nan_layer.py --layers 32 --gdn torch
"""

import argparse

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--gdn", choices=("nki", "torch"), default="nki")
    ap.add_argument("--batches", type=int, default=4, help="how many batches to scan")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sqlft import data
    from sqlft.device import get_device, sync
    from sqlft.model import load_model

    dev = get_device("neuron")
    tok = data.get_tokenizer()
    rows, _ = data.split_rows(data.load_rows())
    rows = data.filter_parseable(rows)
    loader = data.build_loader(rows, tok, args.seq, 1, num_workers=0, seed=args.seed)

    model = load_model(layers=args.layers, gdn=args.gdn, lora=True, device=dev)
    model.eval()

    stats: list[tuple[int, str, float, bool]] = []

    def hook(idx, kind):
        def fn(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.detach().float()
            finite = bool(torch.isfinite(h).all().cpu())
            amax = float(h.abs().amax().cpu()) if finite else float("inf")
            stats.append((idx, kind, amax, finite))
        return fn

    handles = []
    for i, layer in enumerate(model.model.layers):
        kind = "full_attn" if hasattr(layer, "self_attn") else "gdn"
        handles.append(layer.register_forward_hook(hook(i, kind)))

    for b, batch in enumerate(loader):
        if b >= args.batches:
            break
        stats.clear()
        batch = {k: v.to(dev) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        sync(dev)
        loss = float(out.loss.float().cpu())
        n_real = int(batch["attention_mask"].sum().cpu())
        pad = args.seq - n_real

        first_bad = next((s for s in stats if not s[3]), None)
        print(f"\nbatch {b}: loss={loss}  content={n_real} tok  padding={pad} tok "
              f"({100 * pad / args.seq:.0f}%)")
        if first_bad is None:
            peak = max(stats, key=lambda s: s[2])
            print(f"  all {len(stats)} layers finite; largest |activation| "
                  f"{peak[2]:.3e} at layer {peak[0]} ({peak[1]})")
        else:
            print(f"  FIRST NON-FINITE at layer {first_bad[0]} ({first_bad[1]})")
            # The few layers before it are where the magnitude ramps.
            lo = max(0, first_bad[0] - 4)
            for idx, kind, amax, finite in stats[lo : first_bad[0] + 1]:
                flag = "" if finite else "   <-- NON-FINITE"
                print(f"    layer {idx:2d} {kind:9s} |max| {amax:.4e}{flag}")

    for h in handles:
        h.remove()
    print("\nNAN PROBE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
