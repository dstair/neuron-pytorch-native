"""Minimal reproducer + ablation for the NKI GatedDeltaNet kernel divergence.

Loads the real (q, k, v, g, beta) captured from the 6th GDN layer of Qwen3.5-4B -- the first
call where the kernel departs from the reference -- and swaps one tensor at a time for a
synthetic one. Synthetic inputs pass parity at every shape and every log-decay magnitude, so
whichever single substitution FIXES the divergence identifies the property the kernel cannot
handle.

    ./env/container.sh exec python3 labs/lab02_gdn_in_situ.py --dump 5     # capture first
    ./env/container.sh exec python3 labs/lab02_gdn_repro.py
"""

import argparse
import pathlib

import torch

REPRO = pathlib.Path("/work/runs/gdn_repro_call5.pt")


def describe(name: str, t: torch.Tensor) -> str:
    f = t.float()
    return (f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"min={f.min():.4g} max={f.max():.4g} "
            f"absmax={f.abs().max():.4g} mean={f.mean():.4g} std={f.std():.4g} "
            f"zeros={int((f == 0).sum())}/{f.numel()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", type=pathlib.Path, default=REPRO)
    ap.add_argument("--device", default="neuron")
    args = ap.parse_args()

    import transformers.models.qwen3_5.modeling_qwen3_5 as hf

    from sqlft.device import get_device, sync
    from sqlft.gdn import make_nki_chunk_fn

    dev = get_device(args.device)
    real = torch.load(args.repro, map_location="cpu", weights_only=True)
    # The ablation isolates the v3.0 failure, so it must run the v3.0 build explicitly.
    nki = make_nki_chunk_fn("nki")

    print("--- captured real inputs (6th GDN layer, the first that diverges) ---")
    for n in ("q", "k", "v", "g", "beta"):
        print("  " + describe(n, real[n]))

    B, T, H, D = real["q"].shape
    torch.manual_seed(0)
    synth = {
        "q": torch.randn(B, T, H, D), "k": torch.randn(B, T, H, D), "v": torch.randn(B, T, H, D),
        # Matched to the real log-decay range so the ablation isolates SHAPE/DISTRIBUTION of
        # each tensor, not merely its scale.
        "g": real["g"].float().min() * torch.rand(B, T, H),
        "beta": torch.rand(B, T, H),
    }

    def run(inputs: dict) -> tuple[float, float, bool]:
        ref_out, _ = hf.torch_chunk_gated_delta_rule(
            inputs["q"].float(), inputs["k"].float(), inputs["v"].float(),
            g=inputs["g"].float(), beta=inputs["beta"].float(),
            initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
        )
        nki_out, _ = nki(
            inputs["q"].to(dev), inputs["k"].to(dev), inputs["v"].to(dev),
            g=inputs["g"].to(dev), beta=inputs["beta"].to(dev),
            initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
        )
        sync(dev)
        a = ref_out.float().flatten()
        b = nki_out.detach().float().cpu().flatten()
        ok = bool(torch.isfinite(b).all())
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item() if ok else float("nan")
        return cos, (float(b.abs().max()) if ok else float("inf")), ok

    print(f"\n{'configuration':<34} {'cos':>10} {'nki|max|':>12}  verdict")
    baselines = [("all real (the failing case)", dict(real)),
                 ("all synthetic (known good)", dict(synth))]
    ablations = [(f"real, but {n} <- synthetic", {**real, n: synth[n]})
                 for n in ("q", "k", "v", "g", "beta")]
    # And the converse: synthetic everywhere except one real tensor. If exactly one of these
    # BREAKS, that tensor alone is sufficient to trigger the bug.
    converse = [(f"synthetic, but {n} <- real", {**synth, n: real[n]})
                for n in ("q", "k", "v", "g", "beta")]

    for label, inputs in baselines + ablations + converse:
        cos, amax, ok = run(inputs)
        verdict = "OK" if (ok and cos > 0.99) else ("DIVERGES" if ok else "NON-FINITE")
        print(f"{label:<34} {cos:>10.6f} {amax:>12.4g}  {verdict}")

    print("\nREPRO OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
