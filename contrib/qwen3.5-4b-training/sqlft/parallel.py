"""Parallelism: process-group setup, device meshes, and the four sharding strategies.

Rubric coverage: the "types of parallelism" list. What each one actually shards:

    DDP                 replicate the model on every rank, all-reduce GRADIENTS.
                        Shards: nothing. Buys: throughput (more data per step).
    FSDP2 / ZeRO-3      shard PARAMETERS, GRADIENTS and OPTIMIZER STATE across ranks;
                        all-gather params just in time for each layer's forward, then free.
                        Buys: the ability to train a model whose state does not fit.
    Tensor parallel     shard individual WEIGHT MATRICES across ranks; every rank sees all
                        the data and holds a slice of each layer. Communication is inside
                        the forward (per layer), not at the step boundary.
    Pipeline parallel   shard LAYERS across ranks; micro-batches flow through stages.
                        Buys: depth beyond one device, at the cost of bubble time.

And the one people forget, which this stack actually has a kernel for:

    Context/sequence    shard the SEQUENCE dimension; ring attention passes k/v around.
                        (torch_neuronx ships `distributed/tensor/cp_ring_handler.py` and a
                        TORCH_NEURONX_ENABLE_NKI_RING_ATTENTION flag -- unexplored here.)

The sharp lesson from the earlier bring-up, which lab04 reproduces on purpose: for LoRA,
FSDP `FULL_SHARD` is the WRONG TOOL and measurably slower than DDP. It shards a frozen base
it must then all-gather every forward, while the optimizer state it exists to shard is
~3 M parameters. For full fine-tuning the same flag is the only reason the run fits at all.
"""

from __future__ import annotations

import contextlib
import os

import torch
import torch.nn as nn


def init_distributed() -> tuple[int, int, int]:
    """Set up the neuron process group. Returns (rank, world, local_rank).

    `import torch_neuronx.distributed` is what registers the "neuron" c10d backend and its
    collectives. Importing only `torch_neuronx` leaves init_process_group(backend="neuron")
    to fail later with an opaque `ENC ... no_mesh`.
    """
    import torch.distributed as dist
    import torch_neuronx  # noqa: F401
    import torch_neuronx.distributed  # noqa: F401

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if not dist.is_initialized():
        # The backend reads LOCAL_RANK and sets NEURON_RT_VISIBLE_CORES itself, one logical
        # core per rank, and pins NUMA. Do not set visible cores by hand.
        dist.init_process_group(backend="neuron")
    return rank, world, local


def is_main() -> bool:
    import torch.distributed as dist

    return not dist.is_initialized() or dist.get_rank() == 0


@contextlib.contextmanager
def main_first():
    """Let rank 0 run the body alone, then the rest.

    Used to serialise the cold compile. The run-discipline notes record 8 ranks entering
    neuronx-cc simultaneously exhausting a 124 GB host -- and this box has exactly 124 GB
    and 12 vCPU. Rank 0 compiles into the shared NEFF cache, everyone else hits it.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        yield
        return
    if dist.get_rank() != 0:
        dist.barrier()
    yield
    if dist.get_rank() == 0:
        dist.barrier()


def decoder_layer_classes(model: nn.Module) -> set[type]:
    """The wrap unit for FSDP. Returned as a SET because a hybrid stack can hold more than
    one decoder class -- Qwen3.5 happens to use one class for both attention families, but
    hard-coding that is how you silently wrap nothing on the next model."""
    return {type(layer) for layer in model.model.layers}


def apply_ddp(model: nn.Module) -> nn.Module:
    """Replicate; all-reduce gradients at the step boundary.

    The right tool for LoRA: the base is frozen, so the only gradients on the wire are the
    few MB of adapter. `gradient_as_bucket_view` avoids a second copy of each bucket.
    """
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(model, gradient_as_bucket_view=True)


def apply_fsdp2(model: nn.Module, mesh=None, *, reshard_after_forward: bool = True) -> nn.Module:
    """FSDP2 (`fully_shard`) -- shard params, grads and optimizer state.

    Applied per decoder layer BEFORE the root, which is what lets each layer's parameters be
    freed after its own forward instead of the whole model staying gathered. Wrapping only
    the root would shard the state but keep peak memory at full-model size.

    `reshard_after_forward=False` trades memory for one less all-gather in backward; leave it
    True while memory is the binding constraint, which at 47 GiB of state it is.
    """
    from torch.distributed.fsdp import fully_shard

    kwargs = {"reshard_after_forward": reshard_after_forward}
    if mesh is not None:
        kwargs["mesh"] = mesh
    for layer in model.model.layers:
        fully_shard(layer, **kwargs)
    fully_shard(model, **kwargs)
    return model


def apply_fsdp1(model: nn.Module, *, sharding: str = "FULL_SHARD") -> nn.Module:
    """FSDP1 (`FullyShardedDataParallel`) -- the flat-parameter implementation.

    Kept alongside FSDP2 because they fail differently on this backend: FSDP2's `fully_shard`
    routes its allgather through DTensor functional collectives that get traced into the NEFF,
    and beta6 dies with `Failed to build and load collectives resources for module.neff`.
    FSDP1 issues plain c10d collectives, which the eager smoke test shows do work. The prior
    bring-up used FSDP1 successfully, so this is the known-good path, not a curiosity.

    `use_orig_params=True` is required with LoRA: the flat parameter has to hold a mix of
    frozen base and trainable adapter tensors, and without it FSDP refuses.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    return FullyShardedDataParallel(
        model,
        sharding_strategy=getattr(ShardingStrategy, sharding),
        auto_wrap_policy=ModuleWrapPolicy(decoder_layer_classes(model)),
        device_id=torch.device("neuron"),
        use_orig_params=True,
    )


def apply_tensor_parallel(model: nn.Module, tp_mesh) -> nn.Module:
    """DTensor tensor parallelism on the MLP and the full-attention projections.

    Colwise then Rowwise is the standard pair: `gate_proj`/`up_proj` shard their OUTPUT
    dimension so each rank computes a slice of the hidden activation, and `down_proj` shards
    its INPUT dimension so the matmul reduces across ranks -- one all-reduce per MLP instead
    of one per projection.

    The GatedDeltaNet layers are deliberately left alone: their `in_proj_qkv` output is
    consumed by a grouped `conv1d` and then by a kernel with a hard-coded head layout, so
    sharding the projection without also resharding the conv groups and the kernel's head
    dimension would be wrong rather than just slow.
    """
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    for layer in model.model.layers:
        plan = {
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
        }
        if hasattr(layer, "self_attn"):
            plan.update({
                "self_attn.q_proj": ColwiseParallel(),
                "self_attn.k_proj": ColwiseParallel(),
                "self_attn.v_proj": ColwiseParallel(),
                "self_attn.o_proj": RowwiseParallel(),
            })
        parallelize_module(layer, tp_mesh, plan)
    return model


def clip_grad_norm_(params, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    """`torch.nn.utils.clip_grad_norm_` that tolerates gradients on DIFFERENT meshes.

    Under fsdp2_tp the TP-sharded projections are 2D (dp, tp) DTensors while everything TP
    leaves alone -- the GatedDeltaNet layers, norms, the embedding -- is a 1D (dp,) DTensor.
    The stock clip stacks every per-tensor norm into one tensor and dies with
    `All operands in aten.stack.default must have the same mesh`. So: one norm per mesh,
    each reduced to a plain tensor with full_tensor(), then combined. Identical to the stock
    function when every gradient shares a mesh.
    """
    from torch.distributed.tensor import DTensor

    params = [p for p in params if p.grad is not None]
    groups: dict = {}
    for p in params:
        key = p.grad.device_mesh if isinstance(p.grad, DTensor) else None
        groups.setdefault(key, []).append(p.grad)

    parts = []
    for grads in groups.values():
        n = torch.nn.utils.get_total_norm(grads, norm_type, foreach=False)
        if isinstance(n, DTensor):
            n = n.full_tensor()
        parts.append(n.float() ** norm_type)
    total = torch.stack(parts).sum() ** (1.0 / norm_type)
    torch.nn.utils.clip_grads_with_norm_(params, max_norm, total, foreach=False)
    return total


def build_mesh(world: int, tp: int = 1):
    """A 1D (dp,) or 2D (dp, tp) mesh over the neuron device type."""
    from torch.distributed.device_mesh import init_device_mesh

    if world % tp:
        raise ValueError(f"world={world} is not divisible by tp={tp}")
    if tp == 1:
        return init_device_mesh("neuron", (world,), mesh_dim_names=("dp",))
    return init_device_mesh("neuron", (world // tp, tp), mesh_dim_names=("dp", "tp"))


STRATEGIES = ("single", "ddp", "fsdp1", "fsdp1_noshard", "fsdp2", "fsdp2_tp")


def apply_strategy(model: nn.Module, strategy: str, world: int, tp: int = 1):
    """Dispatch. Returns (model, mesh_or_None)."""
    if strategy == "single":
        return model, None
    if strategy == "ddp":
        return apply_ddp(model), None
    if strategy == "fsdp1":
        return apply_fsdp1(model, sharding="FULL_SHARD"), None
    if strategy == "fsdp1_noshard":
        # NO_SHARD is DDP wearing an FSDP costume: replicate, all-reduce grads. Included so
        # the LoRA regime comparison is apples-to-apples inside one container.
        return apply_fsdp1(model, sharding="NO_SHARD"), None
    if strategy == "fsdp2":
        mesh = build_mesh(world, 1)
        return apply_fsdp2(model, mesh["dp"]), mesh
    if strategy == "fsdp2_tp":
        mesh = build_mesh(world, tp)
        apply_tensor_parallel(model, mesh["tp"])
        return apply_fsdp2(model, mesh["dp"]), mesh
    raise ValueError(f"unknown strategy {strategy!r}; pick from {STRATEGIES}")
