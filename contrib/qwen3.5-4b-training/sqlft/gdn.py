"""Swap Qwen3.5's GatedDeltaNet recurrence for the trainable NKI kernel.

The 24 linear-attention layers are the hard part of this model on any accelerator: the
stock `torch_chunk_gated_delta_rule` is a long chain of small ops that is slow in eager and
(per the prior bring-up) cannot lower under `torch.compile(backend="neuron")` at all.

We swap at the narrowest possible seam. HF's `Qwen3_5GatedDeltaNet.forward` does all of
the projections, the causal conv1d, the beta/g gating, the GQA head expansion, the gated
RMSNorm and the output projection itself, and calls out exactly once for the recurrence:

    core_attn_out, last_state = torch_chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=...,
        output_final_state=..., use_qk_l2norm_in_kernel=True, ...)

So we replace that single module-level function and leave every other line of HF's layer
untouched. Nothing about the parameters changes, so checkpoints load unmodified and LoRA
targets keep working.

`use_qk_l2norm_in_kernel=True` means HF hands over UNNORMALISED q/k and expects the callee
to do the l2-norm and the 1/sqrt(head_dim) scale -- which is exactly what the kernel's
`NeuronGatedDeltaNetTrainableBatchedCompilable.forward` does, so the contract lines up.

    ./env/container.sh exec python3 -m sqlft.gdn --parity
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import torch

# One entry per kernel build; the key is the `--gdn` value every script accepts.
KERNEL_ROOTS = {
    # Upstream v3.0: numerically WRONG beyond ~6 GDN layers on this model. Reproduction only.
    "nki": pathlib.Path("/work/kernels/qwen35-deltanet/build/torch-neuron"),
    # v3.0 with the full-width Neumann inverse replaced by a block-factorized one.
    # Produced by sqlft/kernel_patch/apply_blockdiag_fix.py; see NOTES.md.
    "nki-patched": pathlib.Path("/work/kernels/qwen35-deltanet-blockdiag/torch-neuron"),
    # Upstream v15 (commit 36d9feb8). The training forward has carried upstream's own fix for
    # the same failure since v8 (row-block forward substitution, BLOCK=32), and v14 made the
    # backward 1.09-1.17x faster. Still not accurate enough on this model: at GDN layer 19
    # dv and dbeta are 6-7% off (cos 0.925 / 0.943). Contrast only.
    "nki-v15": pathlib.Path("/work/kernels/qwen35-deltanet-v15/build/torch-neuron"),
    # THE DEFAULT. v15 with each 32-wide diagonal-block inverse split into two 16-wide halves
    # plus an exact merge -- matches the reference on all 24 GDN layers, forward and every
    # gradient. Produced by sqlft/kernel_patch/apply_v15_split16.py; see NOTES.md.
    "nki-v15-split16": pathlib.Path("/work/kernels/qwen35-deltanet-v15-split16/torch-neuron"),
}
KERNEL_ROOT = KERNEL_ROOTS["nki"]
GDN_CHOICES = ("torch", *KERNEL_ROOTS)
GDN_HELP = ("torch = HF's reference recurrence (correct, slow); nki-v15-split16 = upstream v15 "
            "with a 16-wide diagonal solve (correct AND fastest, the default for training); "
            "nki-v15 = upstream v15 as published (6-7%% off on some gradients here); "
            "nki-patched = v3.0 plus our block-factorized inverse (correct); "
            "nki = upstream v3.0, numerically wrong beyond ~6 GDN layers -- reproduction only")

# The kernel's training path fixes the chunk at 128 and hard-codes head_dim=128.
KERNEL_CHUNK = 128
KERNEL_HEAD_DIM = 128

_kernel_pkg = None
_original_chunk_fn = None


def load_kernel_package(root: pathlib.Path = KERNEL_ROOT, name: str | None = None):
    """Import the kernels-hub repo as a package.

    Done by hand rather than via `kernels.get_kernel()` so the revision is whatever
    setup_box.sh pinned on disk (auditable, and works with HF_HUB_OFFLINE=1). The
    directory is literally named `torch-neuron`, which is not a legal module name, so we
    register it under a synthetic name -- required for its `from .x import y` to resolve.
    """
    global _kernel_pkg
    if not root.exists():
        raise FileNotFoundError(f"kernel not found at {root}; run ./env/setup_box.sh kernel")

    # Distinct module name per root, so every kernel build can coexist in one process --
    # which is exactly what a before/after parity test needs.
    if name is None:
        variant = next((v for v, r in KERNEL_ROOTS.items() if r == root), root.parent.name)
        name = "qwen35_deltanet_kernel_" + variant.replace("-", "_")
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if _kernel_pkg is None:
        _kernel_pkg = mod
    return mod


def make_nki_chunk_fn(variant: str = "nki-v15-split16"):
    """Build a drop-in replacement for `torch_chunk_gated_delta_rule` from one kernel build.

    `variant` is a KERNEL_ROOTS key. All three builds expose the same training class with the
    same (B, T, H, D) contract, so only the root differs.
    """
    if variant not in KERNEL_ROOTS:
        raise ValueError(f"unknown kernel variant {variant!r}; pick from {tuple(KERNEL_ROOTS)}")
    pkg = load_kernel_package(KERNEL_ROOTS[variant])
    training = importlib.import_module(f"{pkg.__name__}.training")
    kernel = training.NeuronGatedDeltaNetTrainableBatchedCompilable(chunk_size=KERNEL_CHUNK)

    def nki_chunk_gated_delta_rule(
        query, key, value, g, beta,
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs,
    ):
        if cu_seqlens is not None:
            raise NotImplementedError("variable-length (cu_seqlens) packing is not supported by this kernel")
        if not use_qk_l2norm_in_kernel:
            raise NotImplementedError(
                "the NKI kernel always l2-normalises q/k internally; HF must pass raw q/k"
            )
        B, T, H, D = query.shape
        if D != KERNEL_HEAD_DIM:
            raise ValueError(f"kernel hard-codes head_dim={KERNEL_HEAD_DIM}, got {D}")
        if T % KERNEL_CHUNK:
            raise ValueError(f"seq_len must be a multiple of {KERNEL_CHUNK}, got {T}")

        # The kernel's internals are fp32 (matmul operands, state accumulation, PSUM).
        out, final_state = kernel(
            query.float(), key.float(), value.float(),
            g.float(), beta.float(),
            initial_state=None if initial_state is None else initial_state.float(),
        )
        # Hand back the model's own dtype; HF reshapes this straight into the gated norm.
        return out.to(value.dtype), final_state

    nki_chunk_gated_delta_rule._kernel_pkg = pkg
    return nki_chunk_gated_delta_rule


def install_nki_gdn(variant: str = "nki-v15-split16", verbose: bool = True) -> None:
    """Monkeypatch transformers' recurrence. Idempotent.

    !! `variant="nki"` (upstream v3.0) is KNOWN BROKEN on this model. Synthetic parity is
    exact (cos=1.000000 at every shape up to T=1024/H=32 and every log-decay magnitude), but
    on the model's real activations it diverges from the reference at the 6th GatedDeltaNet
    layer -- |max| 7.1e3 against the reference's 5.8e-2, cosine similarity -0.0006 -- and
    compounds to inf by the 17th, making the loss NaN. `labs/lab02_gdn_in_situ.py` is the
    diagnostic and dumps a reproducer. `nki-patched` and `nki-v15` both fix it.
    """
    global _original_chunk_fn
    import transformers.models.qwen3_5.modeling_qwen3_5 as hf

    if _original_chunk_fn is not None:
        return
    _original_chunk_fn = hf.torch_chunk_gated_delta_rule
    hf.torch_chunk_gated_delta_rule = make_nki_chunk_fn(variant)
    if verbose:
        print(f"GDN: torch_chunk_gated_delta_rule -> NKI fused chunked kernel "
              f"(chunk={KERNEL_CHUNK}, fwd+bwd autograd, {variant})")
        if variant == "nki":
            print("GDN: !! upstream v3.0 is numerically WRONG beyond ~6 GDN layers on real "
                  "activations -- speed measurements only, do not train on it")


def uninstall_nki_gdn() -> None:
    global _original_chunk_fn
    if _original_chunk_fn is None:
        return
    import transformers.models.qwen3_5.modeling_qwen3_5 as hf

    hf.torch_chunk_gated_delta_rule = _original_chunk_fn
    _original_chunk_fn = None


# --------------------------------------------------------------------------------------


def _parity(B: int = 1, T: int = 256, H: int = 4, D: int = 128, device: str = "neuron",
            g_scale: float = 1.0, variant: str = "nki-v15-split16") -> int:
    """Forward AND backward parity of the NKI kernel against HF's torch reference.

    Run on CPU first (`--device cpu`) to validate the wiring without a device compile;
    the CPU run exercises the same autograd.Function but through its torch fallback.
    """
    import transformers.models.qwen3_5.modeling_qwen3_5 as hf

    from sqlft.device import get_device, sync

    torch.manual_seed(0)
    dev = get_device(device)

    # Inputs live on CPU; each run moves its own copies. The HF reference always stays on
    # CPU -- it is the oracle, and (per the prior bring-up) it does not lower on Neuron
    # anyway, so running it on device would measure a fallback, not a reference.
    # g is the RAW LOG-DECAY, and its magnitude is the thing that breaks chunked kernels.
    # In the real model g = -A_log.exp() * softplus(a + dt_bias) with A in [0.01, 16], so it
    # reaches far below the -1..0 that a naive `-rand()` produces. Chunked delta-rule maths
    # forms exp(g_i - g_j) terms; once consecutive g are very negative, the reciprocal
    # direction overflows to inf and the whole layer becomes NaN. `--g-scale` sweeps that.
    base = {
        "q": torch.randn(B, T, H, D), "k": torch.randn(B, T, H, D), "v": torch.randn(B, T, H, D),
        "g": -torch.rand(B, T, H) * g_scale, "beta": torch.rand(B, T, H),
    }

    def run(fn, target):
        args = [base[n].detach().clone().to(target).requires_grad_(True)
                for n in ("q", "k", "v", "g", "beta")]
        out, _ = fn(args[0], args[1], args[2], g=args[3], beta=args[4],
                    initial_state=None, output_final_state=False,
                    use_qk_l2norm_in_kernel=True)
        loss = (out.float() ** 2).mean()
        loss.backward()
        sync(target)
        # Pull to CPU before comparing: comparisons on device would themselves compile.
        return (out.detach().float().cpu(),
                [a.grad.detach().float().cpu() for a in args],
                loss.item())

    ref_out, ref_grads, ref_loss = run(hf.torch_chunk_gated_delta_rule, torch.device("cpu"))
    nki_out, nki_grads, nki_loss = run(make_nki_chunk_fn(variant), dev)

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

    print(f"variant={variant}  device={device}  B={B} T={T} H={H} D={D}  g_scale={g_scale}  "
          f"g_min={float(base['g'].min()):.2f}")
    print(f"loss  ref={ref_loss:.6e}  nki={nki_loss:.6e}")
    if nki_loss != nki_loss:
        print("      nki loss is NaN while the reference is finite -> kernel numerics")
    print(f"fwd   cos={cos(ref_out, nki_out):.6f}  max|d|={(ref_out - nki_out).abs().max():.3e}")
    ok = cos(ref_out, nki_out) > 0.999
    for name, r, n in zip(("dq", "dk", "dv", "dg", "dbeta"), ref_grads, nki_grads):
        c = cos(r, n)
        ok &= c > 0.99
        print(f"bwd   {name:5s} cos={c:.6f}  max|d|={(r - n).abs().max():.3e}")
    print("GDN PARITY OK" if ok else "GDN PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--g-scale", type=float, default=1.0,
                    help="scale the log-decay magnitude toward what the real model produces")
    ap.add_argument("--variant", choices=tuple(KERNEL_ROOTS), default="nki-v15-split16")
    args = ap.parse_args()
    raise SystemExit(_parity(T=args.seq, H=args.heads, device=args.device,
                             g_scale=args.g_scale, variant=args.variant))
