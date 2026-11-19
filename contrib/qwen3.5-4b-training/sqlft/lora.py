"""LoRA written by hand on torch.nn -- no PEFT.

Rubric coverage: torch.nn.Module composition, Parameter registration, requires_grad /
autograd graph shape, and why the adapter is cheap in *optimizer* memory but not in
*activation* memory.

The whole idea: freeze W (d_out x d_in) and learn a low-rank correction

    y = x W^T + (alpha / r) * (x A^T) B^T      A: (r, d_in)   B: (d_out, r)

B starts at zero, so the adapted model is exactly the base model at step 0 -- a property
we assert in the tests, because getting it wrong gives you a silently different starting
point and an unexplainable loss spike.

    python3 -m pytest sqlft/tests/test_lora.py -q
"""

from __future__ import annotations

import math
from typing import Iterable, Iterator

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable rank-r update."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        dev, dt = base.weight.device, base.weight.dtype
        # Registered as Parameters on `self`, so they appear in .parameters() and get
        # picked up by the optimizer, FSDP flattening and state_dict automatically.
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        # Kaiming on A, zeros on B => delta_W == 0 at init.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self._merged:
            return out
        # Two small matmuls instead of one big one: r*(d_in + d_out) work rather than
        # d_in*d_out. Note x is retained for BOTH branches -- LoRA saves optimizer state,
        # not activations, which is why it still OOMs at long sequence length.
        lo = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scaling * lo

    @torch.no_grad()
    def merge(self) -> None:
        """Fold BA into the base weight. For inference/eval: removes the extra matmuls."""
        if self._merged:
            return
        self.base.weight.add_(self.scaling * (self.lora_B @ self.lora_A).to(self.base.weight.dtype))
        self._merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self._merged:
            return
        self.base.weight.sub_(self.scaling * (self.lora_B @ self.lora_A).to(self.base.weight.dtype))
        self._merged = False

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:g}, merged={self._merged}"


# Candidate leaf names across BOTH attention families in Qwen3.5's hybrid stack.
#   full_attention   layers: q_proj / k_proj / v_proj / o_proj
#   linear_attention layers: in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj
# plus the shared MLP. `apply_lora` intersects this with what the model ACTUALLY has --
# under --layers N truncation whole families disappear (layers 0-2 are linear_attention,
# so --layers 2 exposes no q_proj at all) and targeting a missing name is a hard error.
DEFAULT_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
    "gate_proj", "up_proj", "down_proj",
)

# Everything EXCEPT the MLP: the attention family (q/k/v/o) and the GDN projections.
# The MLP is 56% of the adapter parameters (intermediate 9216 vs hidden 2560), so this
# preset is the parameter-efficiency question in the sweep.
MIXER_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
)

TARGET_PRESETS = {"all": DEFAULT_TARGETS, "mixers": MIXER_TARGETS}


def _named_linears(model: nn.Module) -> Iterator[tuple[str, nn.Module, str, nn.Linear]]:
    for name, mod in model.named_modules():
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                yield f"{name}.{child_name}" if name else child_name, mod, child_name, child


def apply_lora(
    model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    targets: Iterable[str] = DEFAULT_TARGETS,
    *,
    verbose: bool = True,
) -> dict[str, int]:
    """Freeze everything, then swap matching nn.Linear leaves for LoRALinear.

    Returns a summary dict. Freezing happens FIRST so that anything we fail to wrap stays
    frozen -- the failure mode is an under-trained model, never an accidental full fine-tune.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    wanted = set(targets)
    present = {leaf.rsplit(".", 1)[-1] for leaf, *_ in _named_linears(model)}
    missing = wanted - present
    effective = wanted & present
    if not effective:
        raise ValueError(f"none of the LoRA targets exist in this model; present leaves: {sorted(present)}")

    wrapped = 0
    for path, parent, child_name, lin in list(_named_linears(model)):
        if child_name in effective:
            setattr(parent, child_name, LoRALinear(lin, r=r, alpha=alpha, dropout=dropout))
            wrapped += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    summary = {
        "wrapped": wrapped,
        "trainable": trainable,
        "total": total,
        "targets_used": len(effective),
        "targets_absent": len(missing),
    }
    if verbose:
        print(f"LoRA r={r} alpha={alpha}: wrapped {wrapped} Linear layers "
              f"({len(effective)} target names; {sorted(missing) or 'none'} absent)")
        print(f"  trainable {trainable / 1e6:.2f}M / {total / 1e9:.3f}B "
              f"= {100 * trainable / total:.3f}%")
    return summary


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Just the adapter -- a few MB, versus 8.4 GB for the base weights."""
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def save_lora(model: nn.Module, path, *, write: bool = True) -> int:
    """Write the adapter to disk. Returns bytes written.

    Moved to CPU and fp32: a checkpoint that can only be loaded back onto the same device
    type, or that silently carries bf16 rounding, is a checkpoint that bites later.
    """
    import pathlib

    path = pathlib.Path(path)
    sd = gather_lora_state_dict(model)
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, path)
        return path.stat().st_size
    return 0


def gather_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """The adapter as full, CPU, fp32 tensors -- also under FSDP2.

    FSDP2 holds every parameter as a sharded DTensor; `.to("cpu")` on one keeps it a
    DTensor shard, so a rank-0-only save would write 1/world of each adapter matrix (or,
    with `full_tensor()`, hang: the gather is a collective). Under FSDP2 EVERY rank must
    call this; only one should then write (`save_lora(..., write=is_main())`).
    """
    from torch.distributed.tensor import DTensor

    out = {}
    for k, v in lora_state_dict(model).items():
        v = v.detach()
        if isinstance(v, DTensor):
            v = v.full_tensor()
        out[k] = v.to("cpu", torch.float32)
    return out


def merge_and_unwrap_lora(model: nn.Module) -> int:
    """Fold every adapter into its base weight and restore plain nn.Linear modules.

    For inference only: after unwrapping there is nothing trainable to save, so call
    `save_lora` first if the adapter matters. Required by the cached-decode path, whose
    fused in-projection reads `.weight` directly and so cannot see through a LoRALinear.
    Returns the number of modules unwrapped.
    """
    n = 0
    for _name, mod in model.named_modules():
        for child_name, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                child.merge()
                setattr(mod, child_name, child.base)
                n += 1
    return n


def load_lora(model: nn.Module, path, *, strict: bool = True) -> int:
    """Load an adapter into a model that already has LoRA applied. Returns tensors loaded."""
    import pathlib

    sd = torch.load(pathlib.Path(path), map_location="cpu", weights_only=True)
    own = lora_state_dict(model)
    missing = set(own) - set(sd)
    unexpected = set(sd) - set(own)
    if strict and (missing or unexpected):
        raise KeyError(
            f"adapter does not match this model: {len(missing)} missing, "
            f"{len(unexpected)} unexpected (e.g. {sorted(missing or unexpected)[:3]}). "
            "Same --layers, --lora-r and target list are required."
        )
    # load_state_dict with strict=False so the frozen base is left alone.
    target = {k: v for k, v in sd.items() if k in own}
    by_name = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in target.items():
            by_name[k].copy_(v.to(by_name[k].dtype))
    return len(target)
