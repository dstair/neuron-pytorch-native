"""Device helpers. Import this before touching torch.device("neuron").

`neuron` is a PrivateUse1 device registered as a side effect of importing torch_neuronx.
Without that import you get the confusing:

    RuntimeError: Expected one of cpu, cuda, ... privateuseone device type
                  at start of device string: neuron

which reads like a typo rather than a missing import. Everything here funnels through one
place so that failure can only happen once.
"""

from __future__ import annotations

import time

import torch

_torch_neuronx = None


def _tnx():
    global _torch_neuronx
    if _torch_neuronx is None:
        import torch_neuronx  # noqa: PLC0415  -- the import IS the registration

        _torch_neuronx = torch_neuronx
    return _torch_neuronx


def get_device(name: str = "neuron") -> torch.device:
    if name.startswith("neuron"):
        _tnx()
    return torch.device(name)


def sync(device: torch.device | str = "neuron") -> None:
    """Barrier before any timing.

    The native beta executes EAGERLY -- there is no xm.mark_step to force a flush -- so an
    un-synchronised timer measures dispatch, not execution. This is the single easiest way
    to publish a wrong number on this stack.
    """
    if str(device).startswith("neuron"):
        _tnx().synchronize()


class Timer:
    """Wall-clock around a synchronised region.

        with Timer("step") as t: ...
        print(t.seconds)
    """

    def __init__(self, label: str = "", device: torch.device | str = "neuron", quiet: bool = True):
        self.label, self.device, self.quiet = label, device, quiet
        self.seconds = 0.0

    def __enter__(self) -> "Timer":
        sync(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        sync(self.device)
        self.seconds = time.perf_counter() - self._t0
        if not self.quiet:
            print(f"{self.label}: {self.seconds:.3f}s")


def device_memory_bytes() -> dict[str, int]:
    """Per-device HBM in use, if this build exposes it. Empty dict when it does not."""
    tnx = _tnx()
    for attr in ("memory_allocated", "max_memory_allocated"):
        if not hasattr(tnx, attr):
            return {}
    return {
        "allocated": int(tnx.memory_allocated()),
        "peak": int(tnx.max_memory_allocated()),
    }


def set_dynamo_limits(recompile_limit: int = 256) -> None:
    """The kernel README measures up to 1.44x silent degradation at the default limit of 8,
    because varying sequence length blows past it and dynamo falls back to eager."""
    torch._dynamo.config.recompile_limit = recompile_limit
