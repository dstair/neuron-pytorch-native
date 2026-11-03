"""In-situ GDN parity: compare the NKI kernel against the reference on the model's OWN tensors.

Synthetic parity passes everywhere -- every (T, H) up to 1024x32, and every log-decay
magnitude up to g_scale=256 -- yet the 32-layer model gives loss 1.0299 with the kernel and
0.5031 with the reference on the same batch, then NaNs. So the synthetic inputs are simply not
like the real ones in whatever way matters.

This stops guessing which axis to sweep. It intercepts every `torch_chunk_gated_delta_rule`
call the real model makes, runs BOTH implementations on those exact tensors, and reports the
divergence per layer. The first layer where cosine similarity falls away is the answer, and the
captured inputs become a minimal reproducer.

    ./env/container.sh exec python3 labs/lab02_gdn_in_situ.py --layers 32 --seq 512
    ./env/container.sh exec python3 labs/lab02_gdn_in_situ.py --layers 32 --seq 512 --dump 0
"""

import argparse
import pathlib

import torch

RUNS = pathlib.Path("/work/runs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1, help="loader seed; 1 = a rank that NaN'd")
    ap.add_argument("--dump", type=int, default=None,
                    help="save the inputs of this GDN call index to runs/ as a reproducer")
    ap.add_argument("--variant", default="nki-v15-split16",
                    help="kernel build: nki (upstream v3.0), nki-patched, nki-v15 or nki-v15-split16")
    args = ap.parse_args()

    import transformers.models.qwen3_5.modeling_qwen3_5 as hf

    from sqlft import data
    from sqlft.device import get_device, sync
    from sqlft.gdn import make_nki_chunk_fn
    from sqlft.model import load_model

    reference = hf.torch_chunk_gated_delta_rule
    nki = make_nki_chunk_fn(args.variant)
    print(f"kernel build: {args.variant}")
    rows_out = []

    def both(query, key, value, g, beta, initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kw):
        idx = len(rows_out)

        # Reference on CPU: it is the oracle, and it does not lower on Neuron anyway.
        cpu = lambda t: None if t is None else t.detach().float().cpu()
        ref_out, _ = reference(
            cpu(query), cpu(key), cpu(value), g=cpu(g), beta=cpu(beta),
            initial_state=cpu(initial_state), output_final_state=False,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        nki_out, nki_state = nki(
            query, key, value, g, beta, initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        sync(query.device)

        a = ref_out.float().flatten()
        b = nki_out.detach().float().cpu().flatten()
        b_finite = bool(torch.isfinite(b).all())
        cos = (torch.nn.functional.cosine_similarity(a, b, dim=0).item()
               if b_finite else float("nan"))

        rows_out.append({
            "call": idx,
            "in_finite": all(bool(torch.isfinite(cpu(t)).all())
                             for t in (query, key, value, g, beta) if t is not None),
            "out_finite": b_finite,
            "cos": cos,
            "ref_absmax": float(a.abs().amax()),
            "nki_absmax": float(b.abs().amax()) if b_finite else float("inf"),
            "g_min": float(cpu(g).min()), "g_max": float(cpu(g).max()),
            "beta_min": float(cpu(beta).min()), "beta_max": float(cpu(beta).max()),
            "qkv_absmax": max(float(cpu(t).abs().amax()) for t in (query, key, value)),
        })

        if args.dump is not None and idx == args.dump:
            RUNS.mkdir(parents=True, exist_ok=True)
            p = RUNS / f"gdn_repro_call{idx}.pt"
            torch.save({n: cpu(t) for n, t in
                        (("q", query), ("k", key), ("v", value), ("g", g), ("beta", beta))}, p)
            print(f"  dumped reproducer -> {p}", flush=True)

        # Return the NKI result so the model behaves exactly as it does in a real run.
        return nki_out, nki_state

    hf.torch_chunk_gated_delta_rule = both

    dev = get_device("neuron")
    tok = data.get_tokenizer()
    rows, _ = data.split_rows(data.load_rows())
    rows = data.filter_parseable(rows)
    loader = data.build_loader(rows, tok, args.seq, 1, num_workers=0, seed=args.seed)

    model = load_model(layers=args.layers, gdn="torch", lora=True, device=dev)
    model.eval()

    batch = {k: v.to(dev) for k, v in next(iter(loader)).items()}
    with torch.no_grad():
        out = model(**batch)
    sync(dev)
    print(f"\nloss = {float(out.loss.float().cpu())}")

    print(f"\n{'call':>4} {'in_fin':>6} {'out_fin':>7} {'cos':>10} "
          f"{'ref|max|':>10} {'nki|max|':>10} {'g_min':>9} {'qkv|max|':>9}")
    for r in rows_out:
        print(f"{r['call']:>4} {str(r['in_finite']):>6} {str(r['out_finite']):>7} "
              f"{r['cos']:>10.6f} {r['ref_absmax']:>10.3e} {r['nki_absmax']:>10.3e} "
              f"{r['g_min']:>9.2f} {r['qkv_absmax']:>9.3e}")

    bad = next((r for r in rows_out if not r["out_finite"] or not (r["cos"] > 0.99)), None)
    if bad:
        print(f"\nFIRST DIVERGENCE at GDN call {bad['call']}: cos={bad['cos']:.6f} "
              f"out_finite={bad['out_finite']}")
        print(f"  its inputs: g in [{bad['g_min']:.2f}, {bad['g_max']:.2f}]  "
              f"beta in [{bad['beta_min']:.3f}, {bad['beta_max']:.3f}]  "
              f"|qkv|max={bad['qkv_absmax']:.3e}  inputs_finite={bad['in_finite']}")
    else:
        print("\nno divergence found -- every GDN call matched the reference")

    print("IN SITU OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
