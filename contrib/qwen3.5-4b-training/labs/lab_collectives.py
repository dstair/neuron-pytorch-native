"""The gate every multi-rank lab depends on: do collectives work, and at what world size?

The earlier bring-up of this model hit a hard wall at `world>=8` on the beta3 container:

    ENC:enc_init_comm ... failed (2) to init a collective algorithm.
        reason: no_hier no_mesh replica-group: [0,1,2,3,4,5,6,7]
    RuntimeError: Failed to execute the device barrier 2

...while world 2 and 4 were solid. Several container collective flags were tried and all
failed; the conclusion was to retest on a newer build rather than keep brute-forcing. This
box has beta6, so this script is that retest. Run it BEFORE planning anything at 8 ranks.

    LNC=2 WORLD=4 ./env/container.sh exec torchrun --nproc_per_node 4 labs/lab_collectives.py
    LNC=1 WORLD=8 ./env/container.sh exec torchrun --nproc_per_node 8 labs/lab_collectives.py

Prints `COLLECTIVES OK world=N` on success.
"""

import os
import sys

import torch
import torch.distributed as dist

# Order matters. `torch_neuronx.distributed` is what REGISTERS the "neuron" c10d backend and
# its collectives; importing only `torch_neuronx` leaves init_process_group(backend="neuron")
# to fail later with an opaque `ENC ... no_mesh`. Independently hit by two teams.
import torch_neuronx  # noqa: F401
import torch_neuronx.distributed  # noqa: F401


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])

    # The neuron backend auto-maps each rank to one logical NeuronCore by reading LOCAL_RANK
    # and setting NEURON_RT_VISIBLE_CORES itself, so do NOT pin cores by hand here.
    dist.init_process_group(backend="neuron")
    dev = torch.device("neuron")

    if rank == 0:
        print(f"world={world}  LNC={os.environ.get('NEURON_LOGICAL_NC_CONFIG')}  "
              f"backend={dist.get_backend()}", flush=True)

    # 1. barrier -- this is where the beta3 mesh failure surfaced.
    dist.barrier()
    if rank == 0:
        print("barrier          ok", flush=True)

    # 2. all_reduce with a rank-dependent value, so a no-op would be detectable.
    t = torch.full((1024,), float(rank + 1), dtype=torch.float32, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch_neuronx.synchronize()
    expect = world * (world + 1) / 2
    got = float(t[0].cpu())
    assert abs(got - expect) < 1e-3, f"all_reduce: got {got}, expected {expect}"
    assert torch.isfinite(t.cpu()).all()
    if rank == 0:
        print(f"all_reduce SUM   ok  (= {got:.0f}, expected {expect:.0f})", flush=True)

    # 3. all_gather -- FSDP's forward depends on this one specifically.
    src = torch.full((256,), float(rank), dtype=torch.float32, device=dev)
    gathered = [torch.empty_like(src) for _ in range(world)]
    dist.all_gather(gathered, src)
    torch_neuronx.synchronize()
    for i, g in enumerate(gathered):
        assert abs(float(g[0].cpu()) - i) < 1e-3, f"all_gather slot {i} = {float(g[0].cpu())}"
    if rank == 0:
        print("all_gather       ok", flush=True)

    # 4. reduce_scatter -- FSDP's backward depends on this one.
    inp = torch.arange(world * 128, dtype=torch.float32, device=dev)
    out = torch.empty(128, dtype=torch.float32, device=dev)
    dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
    torch_neuronx.synchronize()
    # Every rank contributes the same input, so each shard is world * its own slice.
    expect0 = world * float(rank * 128)
    assert abs(float(out[0].cpu()) - expect0) < 1e-3, f"reduce_scatter: {float(out[0].cpu())} vs {expect0}"
    if rank == 0:
        print("reduce_scatter   ok", flush=True)

    # 5. a 2D device mesh -- what composing FSDP with tensor parallel needs.
    if world >= 4:
        from torch.distributed.device_mesh import init_device_mesh

        dp, tp = world // 2, 2
        mesh = init_device_mesh("neuron", (dp, tp), mesh_dim_names=("dp", "tp"))
        dist.all_reduce(t.clone(), group=mesh["tp"].get_group())
        torch_neuronx.synchronize()
        if rank == 0:
            print(f"2D mesh (dp={dp}, tp={tp}) sub-group all_reduce   ok", flush=True)

    dist.barrier()
    if rank == 0:
        print(f"COLLECTIVES OK world={world}", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Print the full traceback from every rank: the informative error is often on a
        # non-zero rank and torchrun truncates by default.
        import traceback

        print(f"RANK {os.environ.get('RANK')} FAILED", flush=True)
        traceback.print_exc()
        sys.exit(1)
