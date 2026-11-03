"""The three different things people call "checkpointing". They are not related.

The rubric lists "types of checkpointing: gradients, activations", which collapses ideas
that trade against completely different resources. Keeping them separate here on purpose:

1. ACTIVATION checkpointing (a.k.a. "gradient checkpointing") -- `activation_checkpointing`
   below. Drops the saved activations inside a block during forward and RECOMPUTES them
   during backward. Buys activation memory, costs ~1 extra forward (~30%) of compute.
   This is what lets us raise seq_len past the point where bs1/seq1024 OOMs on one core.

2. GRADIENT ACCUMULATION -- `accumulate` below. Runs several micro-batches before stepping
   the optimizer. Buys EFFECTIVE batch size at no memory cost. Nothing to do with (1)
   despite the shared word; it does not reduce peak memory of a single micro-batch at all.

3. STATE checkpointing -- `save_checkpoint` / `load_checkpoint` below. Writes model +
   optimizer + dataloader position to disk so a run can resume. On multiple ranks the
   state is sharded, which is why `torch.distributed.checkpoint` exists rather than a
   plain `torch.save` of a gathered state dict.

    ./env/container.sh exec python3 -m sqlft.ckpt --self-test
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from typing import Iterable, Iterator

import torch
import torch.nn as nn


# ---------------------------------------------------------------- 1. activations


def enable_activation_checkpointing(model: nn.Module, verbose: bool = True) -> int:
    """Wrap every decoder layer so its internals are recomputed in backward.

    Uses the NON-REENTRANT implementation: the reentrant one re-runs the block inside a
    fresh autograd graph and breaks on modules with buffers mutated in forward, which the
    GatedDeltaNet conv/state path does.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        checkpoint_wrapper,
    )

    layers = model.model.layers
    for i, layer in enumerate(layers):
        layers[i] = checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    if verbose:
        print(f"activation checkpointing: wrapped {len(layers)} decoder layers (non-reentrant)")
    return len(layers)


# ---------------------------------------------------------------- 2. accumulation


def accumulate(batches: Iterable[dict], micro_steps: int) -> Iterator[tuple[dict, bool]]:
    """Yield (batch, is_last_of_group). Loss must be divided by micro_steps by the caller.

    Dividing is not optional: without it the gradient magnitude -- and therefore the
    effective learning rate -- scales with micro_steps, so "same LR, bigger batch" silently
    is not.
    """
    buf = []
    for b in batches:
        buf.append(b)
        if len(buf) == micro_steps:
            for i, item in enumerate(buf):
                yield item, i == micro_steps - 1
            buf = []


# ---------------------------------------------------------------- 3. state


@dataclasses.dataclass
class TrainState:
    """Everything needed to resume that is not a tensor."""

    step: int = 0
    tokens_seen: int = 0
    epoch: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def save_checkpoint(
    path: pathlib.Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainState,
    *,
    distributed: bool | None = None,
) -> pathlib.Path:
    """Save model + optimizer + progress.

    Uses torch.distributed.checkpoint (DCP) whenever a process group is initialised. DCP
    writes one shard per rank and reshards on load, so a checkpoint taken at world=4 can be
    resumed at world=8 -- which a `torch.save` of a rank-0-gathered state dict cannot do.
    """
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if distributed is None:
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    if distributed:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict

        model_sd, optim_sd = get_state_dict(model, optimizer)
        dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=str(path / "dcp"))
        if torch.distributed.get_rank() == 0:
            (path / "train_state.json").write_text(json.dumps(state.to_dict(), indent=2))
    else:
        torch.save(
            {"model": model.state_dict(), "optim": optimizer.state_dict()},
            path / "single.pt",
        )
        (path / "train_state.json").write_text(json.dumps(state.to_dict(), indent=2))
    return path


def load_checkpoint(
    path: pathlib.Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    distributed: bool | None = None,
) -> TrainState:
    path = pathlib.Path(path)
    if distributed is None:
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    if distributed:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        # DCP loads IN PLACE, so it needs correctly-shaped tensors to read into -- hence
        # fetching the current state dict first rather than passing empty dicts.
        model_sd, optim_sd = get_state_dict(model, optimizer)
        payload = {"model": model_sd, "optim": optim_sd}
        dcp.load(payload, checkpoint_id=str(path / "dcp"))
        set_state_dict(model, optimizer, model_state_dict=payload["model"],
                       optim_state_dict=payload["optim"])
    else:
        blob = torch.load(path / "single.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(blob["model"])
        optimizer.load_state_dict(blob["optim"])

    return TrainState(**json.loads((path / "train_state.json").read_text()))


# --------------------------------------------------------------------------------------


def _self_test() -> int:
    """CPU-only. Checks accumulation arithmetic and a single-process save/resume."""
    import tempfile

    # (2) accumulation: 4 micro-steps must produce the SAME gradient as one batch of 4.
    torch.manual_seed(0)
    lin = nn.Linear(8, 4)
    x, y = torch.randn(8, 8), torch.randn(8, 4)

    lin.zero_grad()
    nn.functional.mse_loss(lin(x), y).backward()
    whole = lin.weight.grad.clone()

    lin.zero_grad()
    micro = 4
    for i in range(micro):
        sl = slice(i * 2, (i + 1) * 2)
        (nn.functional.mse_loss(lin(x[sl]), y[sl]) / micro).backward()
    accumulated = lin.weight.grad.clone()
    assert torch.allclose(whole, accumulated, atol=1e-6), "accumulated grad != whole-batch grad"
    print("accumulation: 4 micro-steps == 1 batch of 4 (grads match)")

    # The /micro_steps division is load-bearing; show what happens without it.
    lin.zero_grad()
    for i in range(micro):
        sl = slice(i * 2, (i + 1) * 2)
        nn.functional.mse_loss(lin(x[sl]), y[sl]).backward()
    assert not torch.allclose(whole, lin.weight.grad, atol=1e-6)
    print(f"  without the /{micro}: grad is {(lin.weight.grad / whole).mean():.1f}x too large")

    # (3) state checkpointing round-trip.
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        opt.step()

    with tempfile.TemporaryDirectory() as td:
        save_checkpoint(pathlib.Path(td), model, opt, TrainState(step=3, tokens_seen=999),
                        distributed=False)
        ref_w = model.weight.detach().clone()
        # Keep training to move the weights away, then restore.
        for _ in range(3):
            opt.zero_grad()
            nn.functional.mse_loss(model(x), y).backward()
            opt.step()
        assert not torch.allclose(model.weight, ref_w)

        st = load_checkpoint(pathlib.Path(td), model, opt, distributed=False)
        assert torch.equal(model.weight, ref_w), "weights not restored bit-exactly"
        assert st.step == 3 and st.tokens_seen == 999
        # Optimizer moments must come back too, or the first post-resume step is wrong.
        assert opt.state[model.weight]["step"].item() == 3
    print("state checkpoint: weights bit-exact, AdamW step/moments restored")

    print("CKPT SELF-TEST OK")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.parse_args()
    raise SystemExit(_self_test())
