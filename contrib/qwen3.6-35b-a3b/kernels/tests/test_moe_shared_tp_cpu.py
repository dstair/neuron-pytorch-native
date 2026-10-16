#!/usr/bin/env python3
"""CPU algebra checks for sharding the SHARED expert over TP (MOE_SHARED_TP).

REFUTED ON DEVICE 2026-09-02 -- the MOE_SHARED_TP flag has been REVERTED and no longer
exists. At 40L/seq10k it measured -0.28% throughput (gate was >= +1.5%) and cosine
0.9991249 (gate was >= 0.9999), while saving 0.22 GB/core. See
skills/nki-perf-patterns/references/refuted-levers.md.

This file is kept deliberately, because it is the evidence that the ALGEBRA is exact
(2e-7 relative in fp32) -- which is what makes the 4% relative-L2 divergence measured
on device a DEVICE/precision property rather than a math error. Do not re-derive the
algebra from scratch; read the ledger entry first.

The shared expert is replicated on every rank today, costing about as much per rank
as the entire expert-parallel routed computation. Sharding it over its intermediate
dim is exact and needs no new collective, but only if the per-rank partial is folded
in BEFORE the all-reduce -- the gate `sg` is a replicated [T,1] scalar, so

    sum_r (sg * sh_r) == sg * sum_r sh_r == sg * sh.

These checks pin the three things that could silently break that:
  1. the intermediate-dim split reconstructs the full shared output exactly;
  2. folding pre-reduce equals adding post-reduce (to fp32 tolerance);
  3. adding the partial POST-reduce -- the bug the runtime guard exists to prevent --
     is wrong by the expected factor, so the guard is testing something real.

No device. Small dims, exact arithmetic in float64 where it matters.
"""

import torch
import torch.nn.functional as F


HIDDEN = 17
SHARED_INTER = 12          # stands in for shared_expert_intermediate_size = 512
WORLD_SIZE = 4             # SHARED_INTER % WORLD_SIZE == 0
TOKENS = 5


def shared_expert(x, sh_gate, sh_up, sh_down):
    """Reference: the full, replicated shared expert."""
    return F.linear(F.silu(F.linear(x, sh_gate)) * F.linear(x, sh_up), sh_down)


def shared_expert_shard(x, sh_gate, sh_up, sh_down, rank, world_size=WORLD_SIZE):
    """One rank's partial, using the same slicing the loader applies.

    colwise on gate/up ([I,H] -> [I/ws,H]), rowwise on down ([H,I] -> [H,I/ws]).
    silu and the elementwise product are per-intermediate-unit, so slicing the
    intermediate dim commutes with them -- that is why this is exact.
    """
    width = sh_gate.shape[0] // world_size
    i0 = rank * width
    i1 = i0 + width
    g = F.linear(x, sh_gate[i0:i1])
    u = F.linear(x, sh_up[i0:i1])
    return F.linear(F.silu(g) * u, sh_down[:, i0:i1])


def check_shard_reconstructs(dtype, tol):
    """1. sum_r sh_r == sh, exactly, over the intermediate-dim split."""
    torch.manual_seed(0)
    x = torch.randn(TOKENS, HIDDEN, dtype=dtype)
    sh_gate = torch.randn(SHARED_INTER, HIDDEN, dtype=dtype)
    sh_up = torch.randn(SHARED_INTER, HIDDEN, dtype=dtype)
    sh_down = torch.randn(HIDDEN, SHARED_INTER, dtype=dtype)

    full = shared_expert(x, sh_gate, sh_up, sh_down)
    summed = sum(
        shared_expert_shard(x, sh_gate, sh_up, sh_down, r) for r in range(WORLD_SIZE)
    )
    # RELATIVE error: absolute error scales with output magnitude, so an absolute
    # gate silently becomes a magnitude test. fp32 eps is 1.19e-7 and these are
    # ~12-term dot products, so ~sqrt(n)*eps is the floor to expect.
    err = ((full - summed).norm() / full.norm()).item()
    assert err <= tol, f"shard sum != full: rel err {err:.3e} > {tol:.1e} ({dtype})"
    print(f"PASS shard_reconstructs   dtype={str(dtype):16s} rel_err={err:.3e}")


def check_gate_commutes_with_reduce(dtype, tol):
    """2. Folding sg * sh_r pre-reduce == adding sg * sh post-reduce.

    Also models the routed term, so this is the real expression the kernel builds:
        sum_r (routed_r + sg * sh_r)   vs   sum_r routed_r + sg * sh
    """
    torch.manual_seed(1)
    x = torch.randn(TOKENS, HIDDEN, dtype=dtype)
    sh_gate = torch.randn(SHARED_INTER, HIDDEN, dtype=dtype)
    sh_up = torch.randn(SHARED_INTER, HIDDEN, dtype=dtype)
    sh_down = torch.randn(HIDDEN, SHARED_INTER, dtype=dtype)
    sh_sigmoid = torch.randn(1, HIDDEN, dtype=dtype)
    routed_parts = [torch.randn(TOKENS, HIDDEN, dtype=dtype) for _ in range(WORLD_SIZE)]

    # sg is computed from replicated x and replicated sh_sigmoid, so it is the SAME
    # on every rank. That is the property the fold depends on.
    sg = torch.sigmoid(F.linear(x, sh_sigmoid))
    assert sg.shape == (TOKENS, 1), f"gate must be [T,1], got {tuple(sg.shape)}"

    post = sum(routed_parts) + sg * shared_expert(x, sh_gate, sh_up, sh_down)
    pre = sum(
        routed_parts[r] + sg * shared_expert_shard(x, sh_gate, sh_up, sh_down, r)
        for r in range(WORLD_SIZE)
    )
    err = ((post - pre).norm() / post.norm()).item()
    assert err <= tol, f"pre-reduce fold != post-reduce add: {err:.3e} > {tol:.1e}"
    print(f"PASS gate_commutes        dtype={str(dtype):16s} rel_err={err:.3e}")


def check_post_reduce_add_is_wrong():
    """3. The guarded mistake really is wrong.

    If a variant keeps its post-reduce `+ sg * sh` but the weights were sharded at
    load time, it adds ONE rank's partial instead of the whole shared expert. Assert
    that this is materially wrong, so the runtime guard in _moe is load-bearing
    rather than defensive decoration.
    """
    torch.manual_seed(2)
    x = torch.randn(TOKENS, HIDDEN, dtype=torch.float64)
    sh_gate = torch.randn(SHARED_INTER, HIDDEN, dtype=torch.float64)
    sh_up = torch.randn(SHARED_INTER, HIDDEN, dtype=torch.float64)
    sh_down = torch.randn(HIDDEN, SHARED_INTER, dtype=torch.float64)

    full = shared_expert(x, sh_gate, sh_up, sh_down)
    rank0 = shared_expert_shard(x, sh_gate, sh_up, sh_down, 0)
    rel = ((full - rank0).norm() / full.norm()).item()
    assert rel > 0.1, (
        "post-reduce add with sharded weights should be badly wrong, but the "
        f"relative error is only {rel:.3e} -- this test is not exercising the hazard"
    )
    print(f"PASS post_reduce_is_wrong rel_err={rel:.3f} (guard is load-bearing)")


def check_divisibility_requirement():
    """SHARED_INTER must divide TP -- the loader raises otherwise. Real config is
    shared_expert_intermediate_size=512, so TP in {1,2,4,8,...} all divide it."""
    for ws in (1, 2, 4, 8, 16, 32):
        assert 512 % ws == 0, f"real SHARED_INTER=512 not divisible by TP={ws}"
    assert SHARED_INTER % WORLD_SIZE == 0
    print("PASS divisibility        512 divisible by TP in {1,2,4,8,16,32}")


def main():
    # fp64 is the algebra check; fp32 is the dtype the kernel actually reduces in.
    check_shard_reconstructs(torch.float64, 1e-14)
    check_shard_reconstructs(torch.float32, 1e-6)
    check_gate_commutes_with_reduce(torch.float64, 1e-14)
    check_gate_commutes_with_reduce(torch.float32, 1e-6)
    check_post_reduce_add_is_wrong()
    check_divisibility_requirement()
    print("\nPASS: MOE_SHARED_TP algebra — intermediate-dim shard is exact, "
          "the replicated gate commutes with the reduce, and the post-reduce "
          "add is genuinely wrong")


if __name__ == "__main__":
    main()
