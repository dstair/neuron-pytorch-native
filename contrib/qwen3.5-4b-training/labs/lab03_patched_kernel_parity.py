"""Do the fixed kernels match the oracle? Upstream v3.0 vs our patch vs upstream v15.

Runs all three kernel builds in one process against HF's reference, on synthetic inputs AND on
the real captured tensors from the layer where v3.0 first diverges. The real-tensor column is
the one that matters: v3.0 already passes the synthetic one. The gate passes only if BOTH the
patched build and v15 match, forward and (with --grads) all five gradients.

    ./env/container.sh exec python3 labs/lab03_patched_kernel_parity.py
"""

import argparse
import pathlib

import torch

REPRO = pathlib.Path("/work/runs/gdn_repro_call5.pt")


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", type=pathlib.Path, default=REPRO)
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--grads", action="store_true", help="also compare backward")
    args = ap.parse_args()

    import transformers.models.qwen3_5.modeling_qwen3_5 as hf

    from sqlft.device import get_device, sync
    from sqlft.gdn import make_nki_chunk_fn

    dev = get_device(args.device)
    builds = {
        "v3.0": make_nki_chunk_fn("nki"),
        "patched": make_nki_chunk_fn("nki-patched"),
        "v15": make_nki_chunk_fn("nki-v15"),
        "v15-split16": make_nki_chunk_fn("nki-v15-split16"),
    }
    # The builds this gate certifies. v3.0 and plain v15 are printed as contrasts: v3.0 is
    # wrong outright, and v15's 32-wide diagonal solve loses 6-7% on dv/dbeta at GDN layer 19.
    gated = ("patched", "v15-split16")

    cases: dict[str, dict[str, torch.Tensor]] = {}

    torch.manual_seed(0)
    B, T, H, D = 1, 512, 32, 128
    cases["synthetic"] = {
        "q": torch.randn(B, T, H, D), "k": torch.randn(B, T, H, D),
        "v": torch.randn(B, T, H, D),
        "g": -torch.rand(B, T, H), "beta": torch.rand(B, T, H),
    }
    if args.repro.exists():
        cases["real (layer that diverges)"] = torch.load(
            args.repro, map_location="cpu", weights_only=True
        )

    def run(fn, inputs, target):
        names = ("q", "k", "v", "g", "beta")
        ten = [inputs[n].detach().clone().float().to(target).requires_grad_(args.grads)
               for n in names]
        out, _ = fn(ten[0], ten[1], ten[2], g=ten[3], beta=ten[4],
                    initial_state=None, output_final_state=False,
                    use_qk_l2norm_in_kernel=True)
        grads = None
        if args.grads:
            (out.float() ** 2).mean().backward()
            grads = [t.grad.detach().float().cpu() for t in ten]
        sync(target)
        return out.detach().float().cpu(), grads

    print(f"{'case':<28} {'build':<10} {'cos vs ref':>12} {'|max|':>12} {'finite':>7}")
    ok = True
    for label, inputs in cases.items():
        ref, ref_g = run(hf.torch_chunk_gated_delta_rule, inputs, torch.device("cpu"))
        print(f"{label:<28} {'reference':<10} {'--':>12} {float(ref.abs().max()):>12.4g} "
              f"{'yes':>7}")
        for build, fn in builds.items():
            out, gr = run(fn, inputs, dev)
            finite = bool(torch.isfinite(out).all())
            c = cos(ref, out) if finite else float("nan")
            print(f"{'':<28} {build:<10} {c:>12.6f} "
                  f"{(float(out.abs().max()) if finite else float('inf')):>12.4g} "
                  f"{('yes' if finite else 'NO'):>7}")
            gate = build in gated
            if gate:
                ok &= finite and c > 0.999
            # Gradients for every finite build, so the contrasts are visible too; only the
            # gated builds can fail the run.
            if args.grads and finite and gr is not None and ref_g is not None:
                for n, a, b in zip(("dq", "dk", "dv", "dg", "dbeta"), ref_g, gr):
                    gc = cos(a, b)
                    if gate:
                        ok &= gc > 0.99
                    print(f"{'':<28} {'  ' + n:<10} {gc:>12.6f}")

    print("\nPATCHED KERNEL PARITY OK" if ok else "\nPATCHED KERNEL PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
