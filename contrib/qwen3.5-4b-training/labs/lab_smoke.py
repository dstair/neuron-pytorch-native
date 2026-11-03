"""Device smoke gate: does torch.device("neuron") compute, and does autograd work on it?

Prints SMOKE OK as its success marker. Gate on that marker, not on the exit code --
the beta containers SIGABRT during teardown *after* printing valid results.

    ./env/container.sh exec python3 labs/lab_smoke.py
"""

import torch
import torch_neuronx  # registers the `neuron` device (PrivateUse1)


def main() -> int:
    print("torch        :", torch.__version__)
    print("torch_neuronx:", torch_neuronx.__version__)
    print("platform     :", torch_neuronx.utils.get_platform_target())

    dev = torch.device("neuron")

    # 1. forward: a matmul whose answer we know exactly.
    a = torch.ones(256, 512, dtype=torch.bfloat16, device=dev)
    b = torch.full((512, 128), 2.0, dtype=torch.bfloat16, device=dev)
    out = (a @ b).float().cpu()
    torch_neuronx.synchronize()
    assert torch.isfinite(out).all(), "matmul produced non-finite values"
    assert torch.allclose(out, torch.full_like(out, 1024.0)), f"wrong matmul: {out[0, :4]}"
    print(f"matmul       : ok, all {out.numel()} elems == 1024.0")

    # 2. backward: d/dw of sum((x@w)) is a column-sum of x, broadcast. Cheapest possible
    #    proof that the autograd graph actually lowers and runs on device.
    x = torch.randn(64, 32, dtype=torch.float32, device=dev)
    w = torch.randn(32, 16, dtype=torch.float32, device=dev, requires_grad=True)
    (x @ w).sum().backward()
    torch_neuronx.synchronize()
    grad = w.grad.cpu()
    expect = x.cpu().sum(0, keepdim=True).T.expand(32, 16)
    assert torch.isfinite(grad).all(), "grad is non-finite"
    assert torch.allclose(grad, expect, atol=1e-3), f"grad mismatch, max err {(grad - expect).abs().max()}"
    print("autograd     : ok, dL/dw matches the analytic column-sum")

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
