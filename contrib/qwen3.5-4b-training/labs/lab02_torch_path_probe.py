"""What does `--gdn torch` actually execute, and where?

The torch path is NOT torch.compile -- nothing in this project compiles the training step. It
is HF's reference `torch_chunk_gated_delta_rule` decomposed into ordinary aten ops, dispatched
one at a time onto the neuron device. Two things worth knowing rather than assuming:

  1. That reference calls `torch.linalg.solve_triangular`. If the backend does not implement it,
     `TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1` sends it to the HOST -- meaning the
     "correct" path is partly running on CPU, which would explain a lot of its cost.
  2. HF's default chunk_size is 64; the NKI kernel is fixed at 128. Different chunking is a
     different (still correct) algorithm, and a different amount of work.

    ./env/container.sh exec python3 labs/lab02_torch_path_probe.py
"""

import torch

from sqlft.device import get_device, sync


def where(t: torch.Tensor) -> str:
    return f"{t.device}"


def main() -> int:
    dev = get_device("neuron")
    print(f"torch {torch.__version__}  device {dev}")

    # A unit-lower-triangular system of the shape the GDN reference actually solves.
    n = 128
    a = torch.eye(n) + torch.randn(n, n).tril(-1) * 0.01
    b = torch.randn(n, 16)

    ref = torch.linalg.solve_triangular(a, b, upper=False, unitriangular=True)

    print("\n-- torch.linalg.solve_triangular on the neuron device --")
    try:
        out = torch.linalg.solve_triangular(
            a.to(dev), b.to(dev), upper=False, unitriangular=True
        )
        sync(dev)
        print(f"  ran. result device = {where(out)}")
        err = (out.float().cpu() - ref).abs().max().item()
        print(f"  max |device - cpu| = {err:.3e}")
        print("  NOTE: a result that comes back on 'neuron' may still have been computed on the")
        print("        host and copied back -- the fallback is transparent by design.")
    except Exception as e:
        print(f"  RAISED: {type(e).__name__}: {str(e).splitlines()[0][:160]}")

    # Which branch of the reference do we take? `is_torchdynamo_exporting()` picks the explicit
    # forward-substitution inverse instead of the triangular solve when tracing for export.
    from transformers.utils import is_torchdynamo_exporting

    print(f"\nis_torchdynamo_exporting() = {is_torchdynamo_exporting()}"
          "  (False => the solve_triangular branch is the one that runs)")

    # HF's default chunk size, for the record.
    import inspect

    import transformers.models.qwen3_5.modeling_qwen3_5 as hf

    sig = inspect.signature(hf.torch_chunk_gated_delta_rule)
    print(f"HF reference chunk_size default = {sig.parameters['chunk_size'].default}"
          "   (NKI kernel is hard-fixed at 128)")

    print("\nTORCH PATH PROBE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
