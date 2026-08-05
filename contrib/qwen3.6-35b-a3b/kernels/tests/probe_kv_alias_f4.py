"""Does DECLARING the input/output alias make N in-place writes per graph land?

This is the F4 question, isolated. Everything about RoPE, GQA and dynamic offsets
is stripped out: the kernel here does nothing but DMA one tile into a cache slab,
because the variable under test is the graph-boundary alias, not the arithmetic.

WHY THIS IS THE RIGHT LEVER (traced through the stack 2026-08-05):

  torch_neuronx/nki_hop.py:451 hardcodes `operand_output_aliases={}` in
  `NKIHOPCaller.__call__` -- but that value is DEAD. Every impl that matters
  overwrites it from the kernel's own compile result:

    nki_hop.py:263  meta impl          operand_output_aliases = dconfig....
    nki_hop.py:338  proxy impl         node_args[...] = dconfig....
    nki_hop.py:378  functionalize      operand_output_aliases=dconfig....

  and `dconfig.operand_output_aliases` (nki_kernel.py:172-178) is inverted from
  `result.input_output_aliases`, which the NKI compiler populates only for inputs
  the kernel RETURNS. Our shipped kernel deliberately does not return the caches
  (gqa_rope_kv_35b.py:191 "Do NOT return kv_key/kv_value"), so the alias map is
  empty, so `ctx.replace`/`commit_update`/`sync` at nki_hop.py:391-398 never runs
  for them. `mutates_args` on the `nki_op` decorator only shapes the PyTorch-level
  schema; it does not reach the emitted custom call. The write-back is therefore
  unmodelled -- which is exactly why it survives sometimes and not others, and why
  the survivor count was never predictable.

  So the fix for the fragility is NOT a torch-side shim. It is one line in the
  kernel: return the mutated input.

THE COST QUESTION, which is the whole reason this was not done in 2026-07:
  The shipped comment asserts returning the cache "makes the backend materialize
  both full caches on every call -- 1.74 GB of HBM traffic per 10-layer region".
  That is true of an UNALIASED output. An ALIASED output is by definition the same
  buffer as the input, so there should be no fresh allocation and no copy. Arm C
  measures this rather than assuming it, because if the materialization is real the
  alias is not worth having and the per-layer buffers stay as the fix.

ARMS
  0  the alias map the compiler actually emitted, per kernel  -> direct evidence
  A  no-alias  (shipped form) + N distinct views of one base  -> expect writes LOST
  B  aliased   (returns cache) + N distinct views of one base  -> the F4 hypothesis
  C  aliased, HBM + wall cost vs A on the per-layer form       -> is the alias free?

Arm 0 matters independently of the occupancy arms: if the aliased kernel does not
produce a non-empty `operand_output_aliases`, then B landing or not landing says
nothing about aliasing and we are back to measuring luck.

Nothing here asserts. This is a measurement: the shipped path is already gated by
test_gqa_rope_kv_multicall_probe.py, and this file exists to decide whether F4 is
worth implementing at all.
"""

import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.dirname(HERE)
sys.path.insert(0, KERNELS)
sys.path.insert(0, HERE)

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402
from torch_neuronx import nki_op  # noqa: E402

TILE = 128
HEAD_DIM = 256
ROWS = 1024          # cache rows per group
G = 3                # mutations per graph
B = 1


def _write_body(cache, src, group_index, num_groups):
    """DMA `src` [TILE, HEAD_DIM] into this group's first TILE rows of `cache`.

    `cache` is [num_groups*ROWS, HEAD_DIM]; the slab base is group_index*rows,
    matching the shipped kernel's (group_index*B + b)*kmax convention at B=1.
    """
    rows = cache.shape[0] // num_groups
    base = group_index * rows
    loaded = nl.ndarray((TILE, HEAD_DIM), dtype=src.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=loaded, src=src)
    # Static slice, not `.ap(scalar_offset=...)`: group_index is a compile-time
    # constant here, and `.ap` rejects a Python int ("scalar_offset must be
    # NkiTensor or VirtualRegister"). The runtime-offset form is what the shipped
    # kernel needs; it is irrelevant to the aliasing question under test.
    nisa.dma_copy(dst=cache[base : base + TILE, :], src=loaded)
    return loaded


@nki.jit
def nki_kv_write_noalias(cache, src, group_index=0, num_groups=1):
    """Shipped shape: mutate `cache` in place, return something else entirely."""
    loaded = _write_body(cache, src, group_index, num_groups)
    receipt = nl.ndarray((TILE, HEAD_DIM), dtype=src.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=receipt, src=loaded)
    return receipt


@nki.jit
def nki_kv_write_aliased(cache, src, group_index=0, num_groups=1):
    """F4 shape: same write, but `cache` is also returned so the NKI compiler
    emits input_output_aliases and the HOP declares the mutation."""
    loaded = _write_body(cache, src, group_index, num_groups)
    receipt = nl.ndarray((TILE, HEAD_DIM), dtype=src.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=receipt, src=loaded)
    return receipt, cache


@nki_op("f4probe::kv_write_noalias", mutates_args={"cache"})
def kv_write_noalias(
    cache: torch.Tensor,
    src: torch.Tensor,
    group_index: int,
    num_groups: int,
) -> torch.Tensor:
    return nki_kv_write_noalias(
        cache, src, group_index=group_index, num_groups=num_groups
    )


@nki_op("f4probe::kv_write_aliased", mutates_args={"cache"})
def kv_write_aliased(
    cache: torch.Tensor,
    src: torch.Tensor,
    group_index: int,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return nki_kv_write_aliased(
        cache, src, group_index=group_index, num_groups=num_groups
    )


def _build_shared(op, aliased):
    """N mutations through N DISTINCT VIEWS of one base -- the form that loses
    writes today. The cache is never a graph output on the no-alias side; on the
    aliased side the op's own returned cache is dropped on the floor, so any
    write-back that shows up on the host came through the alias."""

    def body(cache, src):
        outs = []
        for gi in range(G):
            view = cache[gi]                       # [ROWS, HEAD_DIM], distinct view
            res = op(view, src, 0, 1)
            outs.append(res[0] if aliased else res)
        return tuple(outs)

    return torch.compile(body, backend="neuron", fullgraph=True, dynamic=False)


def _build_per_layer(op, aliased):
    """The shipped fix: one distinct tensor per mutation. Used by arm C to price
    the alias against the form we actually ship."""

    def body(src, *bufs):
        outs = []
        for gi in range(G):
            res = op(bufs[gi], src, 0, 1)
            outs.append(res[0] if aliased else res)
        return tuple(outs)

    return torch.compile(body, backend="neuron", fullgraph=True, dynamic=False)


_ALIAS_LOG: dict[str, dict] = {}


def _install_alias_tap():
    """Record `dconfig.operand_output_aliases` per kernel as tracing produces it.

    Tapping `get_dumped_config` rather than calling `dump_config` ourselves avoids
    having to reconstruct kernel arg names, and reports exactly the map the HOP
    dispatch layer will use (nki_hop.py:263/338/378 all read this same object).
    """
    import torch_neuronx.nki_hop as H

    inner = H.get_dumped_config

    def tapped(kernel_idx, grid, args, arg_names, constant_args_key):
        dconfig = inner(kernel_idx, grid, args, arg_names, constant_args_key)
        name = H.kernel_registry.get_kernel(kernel_idx).__name__
        _ALIAS_LOG[name] = dict(dconfig.operand_output_aliases)
        return dconfig

    H.get_dumped_config = tapped
    # The impls captured `get_dumped_config` by module attribute lookup, so
    # rebinding the module attribute is enough; no need to re-register impls.
    return inner


def _report_alias_map():
    if not _ALIAS_LOG:
        print("  (no dumped configs were observed -- the tap did not fire, so treat "
              "the occupancy arms below as uninterpreted)")
        return
    for name, aliases in sorted(_ALIAS_LOG.items()):
        shape = aliases if aliases else "{} (EMPTY -- write-back is unmodelled)"
        print(f"  {name}: operand_output_aliases={shape}")


def _src():
    torch.manual_seed(11)
    return torch.randn(TILE, HEAD_DIM, dtype=torch.bfloat16).to("neuron")


def _hbm_gb():
    """Device memory in use, if the runtime will tell us. Best-effort."""
    try:
        return torch.neuron.memory_allocated() / 2**30
    except Exception:
        return None


def _run_shared(label, op, aliased):
    src = _src()
    cache = torch.zeros(G, ROWS, HEAD_DIM, dtype=torch.bfloat16, device="neuron")
    run = _build_shared(op, aliased)
    run(cache, src)
    torch.neuron.synchronize()
    host = cache.cpu()
    per_group = [int(torch.count_nonzero(host[gi])) for gi in range(G)]
    landed = sum(1 for n in per_group if n)
    print(f"  {label}: {landed}/{G} writes reached the caller, "
          f"per_group_nz={per_group}")
    del cache
    return landed


def _run_per_layer(label, op, aliased, iters=20):
    src = _src()
    bufs = [
        torch.zeros(ROWS, HEAD_DIM, dtype=torch.bfloat16, device="neuron")
        for _ in range(G)
    ]
    run = _build_per_layer(op, aliased)
    run(src, *bufs)                                  # compile + warm
    torch.neuron.synchronize()
    before = _hbm_gb()
    t0 = time.time()
    for _ in range(iters):
        run(src, *bufs)
    torch.neuron.synchronize()
    dt = (time.time() - t0) / iters * 1e3
    after = _hbm_gb()
    landed = sum(1 for b in bufs if int(torch.count_nonzero(b.cpu())))
    hbm = "" if after is None else f", hbm={after:.3f} GB (delta {after - before:+.3f})"
    print(f"  {label}: {landed}/{G} landed, {dt:.3f} ms/call{hbm}")
    for b in bufs:
        del b
    return landed, dt


def main():
    print("F4 probe: does declaring the input/output alias model the write-back?")
    print(f"  (G={G} mutations per graph, cache {ROWS}x{HEAD_DIM} bf16 per group)")
    _install_alias_tap()

    print("\nARM A -- no alias (shipped kernel shape), N distinct views of one base")
    a = _run_shared("no-alias  shared-base", torch.ops.f4probe.kv_write_noalias, False)

    print("\nARM B -- ALIASED (kernel returns the cache), N distinct views of one base")
    b = _run_shared("aliased   shared-base", torch.ops.f4probe.kv_write_aliased, True)

    print("\nARM C -- cost of the alias on the form we ship (per-layer buffers)")
    _, dt_a = _run_per_layer("no-alias  per-layer", torch.ops.f4probe.kv_write_noalias, False)
    _, dt_b = _run_per_layer("aliased   per-layer", torch.ops.f4probe.kv_write_aliased, True)
    overhead = (dt_b / dt_a - 1.0) * 100.0 if dt_a else float("nan")
    print(f"  alias overhead on the shipped form: {overhead:+.1f}%")

    print("\nARM 0 -- the alias map the NKI compiler actually emitted")
    _report_alias_map()

    print("\nVERDICT")
    aliased_map = _ALIAS_LOG.get("nki_kv_write_aliased")
    if aliased_map is not None and not aliased_map:
        print("  STOP: the aliased kernel compiled to an EMPTY alias map, so "
              "returning a mutated input is not enough to make NKI emit "
              "input_output_aliases. F4 needs a different mechanism (ask the "
              "compiler team how to declare an in-place output); the arms below "
              "are measuring luck, not aliasing.")
    if b >= G and a < G:
        print(f"  F4 CONFIRMED: the alias fixes it ({b}/{G} vs {a}/{G} unaliased).")
        print(f"  Returning the mutated cache is a real structural fix, priced at "
              f"{overhead:+.1f}% on the shipped path.")
        if overhead > 5.0:
            print("  ...but it is NOT free -- check whether the backend is copying "
                  "the cache rather than aliasing it before adopting (that is the "
                  "materialization the 2026-07 comment warned about).")
        else:
            print("  Overhead is in the noise, which is what a true alias should "
                  "cost: same buffer in and out, no fresh allocation.")
    elif b >= G and a >= G:
        print(f"  INCONCLUSIVE: both arms landed {b}/{G}. This probe scale does not "
              f"discriminate -- G={G} in a graph this small is exactly the regime "
              f"where the unaliased form has been seen to work by luck. Raise G or "
              f"re-run the question inside the 40-layer prefill.")
    elif b < G:
        print(f"  F4 REFUTED at probe scale: the alias did NOT make the writes land "
              f"({b}/{G}). Either NKI is not emitting input_output_aliases for a "
              f"returned input, or the alias does not imply write-back. Dump "
              f"dconfig.operand_output_aliases before going further.")


if __name__ == "__main__":
    main()
