"""Several in-place KV writes in ONE graph: which survive, and can they be read?

This is the axis that broke. The earlier version of this probe could not detect
it, for two independent reasons, both worth stating because either alone makes an
in-place-write probe useless:

  1. It returned `kvk[gi, :, 0]` as a graph output for every layer. Exporting the
     cache gives each read its own output buffer, so the reads were correct
     whether or not the *caller's* buffer was ever updated -- and not exporting
     the cache is the entire point of the op. See the header of
     test_gqa_rope_kv_alias_probe.py.

  2. Its host-side assertions were ONE-SIDED. It checked that unowned groups and
     the row tail were zero, i.e. only the ABSENCE of writes. It never asserted
     that the groups it wrote actually contained anything, so a wholly dropped
     write left the slab zero, satisfied the "no spill past active rows" check,
     and went green. A probe that can only detect writing too much cannot detect
     writing nothing.

Both are fixed here: the cache is an input only, and every owned group must be
present with exact values.

WHAT IS ACTUALLY BROKEN (measured 2026-08-05, iso2/iso3 on trn2 -- the earlier
one-line summary "only the FIRST in-place mutation of a buffer per graph is
carried back" was too generous, and the claim that "a sliced view does not alias"
was wrong outright):

  * ONE mutation of a buffer per graph always lands -- whole-tensor reshape or
    sliced view, either way. The pre-2026-07-30 prefill path mutated sliced views
    and populated the cache 100%.
  * N mutations through N DISTINCT VIEWS OF ONE BASE tensor lose writes. Measured
    here: 3 distinct slices of one shared buffer -> 0 of 3 landed, not "the first
    one". At 40 layers with --prefill-splits 4 the same form left the cache 40%
    populated (one surviving write per compiled segment) with finite, plausible
    output.
  * N mutations through the IDENTICAL whole-tensor view (the kernel doing the
    per-group offset arithmetic instead) land, 3 of 3, at this scale. That is a
    floor, not a guarantee: the decode path used exactly this form and still lost
    nine of ten writes at 40 layers -- see test_gqa_tail_stateful_probe.py.
  * READING a mutated buffer back in the same graph is NOT ordered against the
    mutation. Byte-identical bodies differing only in graph-OUTPUT ORDER give
    opposite answers: with the read emitted before the op's own return value it
    sees the pre-mutation zeros; emitted after, it sees the write. Production
    (`k_filled = kv_k[gi][:, 0]` in `_gqa_prefill_chunk`) is in the working order
    and is bit-exact against the path that consumed the op's returned cache --
    but it is not robust, so it is characterized below rather than trusted.

So the only form the shipped code may use is ONE DISTINCT TENSOR PER MUTATION,
which is why there is one cache buffer per GQA layer.

Arms:

  CONTRACT (hard assert) -- per-layer buffers, one per GQA layer, each passed
    whole with group_index=0/num_groups=1. This is the shipped path. Every
    layer's write must be present on the host, at two chunk offsets, so the
    second invocation must also still see the first's rows. The in-graph read is
    emitted in production's order and must see its own write.

  PLATFORM CHARACTERIZATION (measured, reported, never fails) -- the two forms
    above that lose writes or reads. Reported rather than asserted because
    asserting "this must stay broken" turns a platform fix into a red test; but
    if the numbers move, the messages say what to re-check.

The characterization arms are what give the contract arm its meaning: if they
stop discriminating, this file has lost its power and says so out loud.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.dirname(HERE)
sys.path.insert(0, KERNELS)

import gqa_rope_kv_35b_ops  # noqa: E402,F401


CHUNK = 128
KMAX = 4096
Q_HEADS = 2
HEAD_DIM = 256
ROPE_DIM = 64
B = 2
G = 10  # D.NUM_GQA at 40 layers
# The GQA layers of one --prefill-splits 4 segment (layers 10-19 -> gi 2,3,4).
GROUPS = (2, 3, 4)
N = len(GROUPS)
BASES = (0, CHUNK)


def _build_per_layer(digest_last=True):
    """Shipped form: N separate buffers, N calls, cache never an output.

    `digest_last` places the in-graph cache read AFTER the op's own return value
    in the output tuple, which is the order production emits it in and the only
    order in which the read is observed to see the write.
    """

    def body(q, k, v, c, s, base, *bufs):
        outs = []
        for slot in range(N):
            kvk, kvv = bufs[2 * slot], bufs[2 * slot + 1]
            # Whole-tensor reshape of THIS layer's own buffer; the kernel's slab
            # base is (group_index * B + b) * kmax, so a per-layer buffer is the
            # group_index=0, num_groups=1 case.
            ck = kvk.reshape(B * KMAX, HEAD_DIM)
            cv = kvv.reshape(B * KMAX, HEAD_DIM)
            _, key_out = torch.ops.gqa35b.rope_kv_dynamic(
                q, k, v, c, s, ck, cv, base, 0, 1
            )
            # In-graph read, reduced to a row digest so the cache is not exported.
            digest = kvk[:, 0].float().sum(dim=-1)            # [B, KMAX]
            outs.extend((key_out, digest) if digest_last else (digest, key_out))
        return tuple(outs)

    return torch.compile(body, backend="neuron", fullgraph=True, dynamic=False)


def _build_shared_slices():
    """Known-bad form 1: ONE base tensor, N calls, a DISTINCT SLICE each.

    This is what prefill did before 2026-08-05 (`kv_k[gi, :, 0].reshape(...)`).
    """

    def body(q, k, v, c, s, kvk, kvv, base):
        outs = []
        for gi in range(N):
            ck = kvk[gi, :, 0].reshape(B * KMAX, HEAD_DIM)
            cv = kvv[gi, :, 0].reshape(B * KMAX, HEAD_DIM)
            _, key_out = torch.ops.gqa35b.rope_kv_dynamic(
                q, k, v, c, s, ck, cv, base, 0, 1
            )
            outs.append(key_out)
        return tuple(outs)

    return torch.compile(body, backend="neuron", fullgraph=True, dynamic=False)


def _build_shared_whole():
    """Known-bad-at-scale form 2: ONE base tensor, N calls, the IDENTICAL
    whole-tensor view each time, with the kernel doing the per-group offset.

    Lands at this scale. Included because decode used exactly this form and still
    lost nine of ten writes in the real 40-layer graph.
    """

    def body(q, k, v, c, s, kvk, kvv, base):
        outs = []
        for gi in GROUPS:
            ck = kvk.reshape(G * B * KMAX, HEAD_DIM)
            cv = kvv.reshape(G * B * KMAX, HEAD_DIM)
            _, key_out = torch.ops.gqa35b.rope_kv_dynamic(
                q, k, v, c, s, ck, cv, base, gi, G
            )
            outs.append(key_out)
        return tuple(outs)

    return torch.compile(body, backend="neuron", fullgraph=True, dynamic=False)


def _inputs():
    torch.manual_seed(13)
    query = torch.randn(B, Q_HEADS, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
    key = torch.randn(B, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
    value = torch.randn(B, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
    inv = 1.0 / (10_000_000.0 ** (torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM))
    freqs = torch.outer(torch.arange(KMAX).float(), inv)
    rope = torch.cat([freqs, freqs], dim=-1)
    return (
        query.to("neuron"), key.to("neuron"), value.to("neuron"),
        rope.cos().to("neuron"), rope.sin().to("neuron"), value,
    )


def _new_bufs():
    bufs = []
    for _ in range(N):
        bufs.append(torch.zeros(B, 1, KMAX, HEAD_DIM, dtype=torch.bfloat16, device="neuron"))
        bufs.append(torch.zeros(B, 1, KMAX, HEAD_DIM, dtype=torch.bfloat16, device="neuron"))
    return bufs


def _contract_per_layer(failures):
    qn, kn, vn, cn, sn, value = _inputs()
    bufs = _new_bufs()
    run = _build_per_layer(digest_last=True)

    key_by_base = {}
    for base in BASES:
        base_t = torch.tensor([[base]], dtype=torch.int32, device="neuron")
        res = run(qn, kn, vn, cn, sn, base_t, *bufs)
        torch.neuron.synchronize()
        for slot, gi in enumerate(GROUPS):
            key_out, digest = res[2 * slot], res[2 * slot + 1]
            key_by_base[(base, gi)] = key_out.cpu()
            rows = slice(base, base + CHUNK)
            if int(torch.count_nonzero(digest.cpu()[:, rows])) == 0:
                failures.append(
                    f"base={base} gi={gi}: in-graph read saw zeros on the rows it "
                    f"had just written (read not ordered after the mutation)"
                )

    # THE ASSERTION THAT MATTERS, and the one the old probe was missing: every
    # owned layer must be PRESENT on the host, for every chunk written.
    for slot, gi in enumerate(GROUPS):
        host_k = bufs[2 * slot][:, 0].cpu()
        host_v = bufs[2 * slot + 1][:, 0].cpu()
        for base in BASES:
            rows = slice(base, base + CHUNK)
            for b in range(B):
                got_k, want_k = host_k[b, rows], key_by_base[(base, gi)][b]
                if not torch.equal(got_k, want_k):
                    nz = int(torch.count_nonzero(got_k))
                    failures.append(
                        f"gi={gi} base={base} b={b}: kv_k write MISSING from the "
                        f"caller's buffer (nonzero={nz}/{got_k.numel()})"
                    )
                got_v = host_v[b, rows]
                if not torch.equal(got_v, value[b]):
                    nz = int(torch.count_nonzero(got_v))
                    failures.append(
                        f"gi={gi} base={base} b={b}: kv_v write MISSING from the "
                        f"caller's buffer (nonzero={nz}/{got_v.numel()})"
                    )
        # ...and must not have run past the rows it owns.
        tail = int(torch.count_nonzero(host_k[:, 2 * CHUNK :]))
        if tail:
            failures.append(f"gi={gi}: wrote {tail} elems past the active rows")

    landed = sum(
        1 for slot in range(N) if int(torch.count_nonzero(bufs[2 * slot].cpu()))
    )
    print(f"  CONTRACT per-layer buffers: {landed}/{N} writes landed across "
          f"{len(BASES)} chunks")
    for b in bufs:
        del b
    return landed


def _characterize_read_order():
    """Same body, read emitted BEFORE the op's return value instead of after."""
    qn, kn, vn, cn, sn, _ = _inputs()
    bufs = _new_bufs()
    run = _build_per_layer(digest_last=False)
    base_t = torch.tensor([[0]], dtype=torch.int32, device="neuron")
    res = run(qn, kn, vn, cn, sn, base_t, *bufs)
    torch.neuron.synchronize()
    seen = sum(
        1 for slot in range(N)
        if int(torch.count_nonzero(res[2 * slot].cpu()[:, 0:CHUNK]))
    )
    wrote = sum(
        1 for slot in range(N) if int(torch.count_nonzero(bufs[2 * slot].cpu()))
    )
    if seen == 0:
        print(f"  PLATFORM read-before-return-value: {seen}/{N} in-graph reads saw "
              f"their own write (writes still landed: {wrote}/{N}) -- reading a "
              f"mutated buffer in-graph is NOT ordered against the mutation; "
              f"production relies on the working order, so keep the occupancy gate")
    else:
        print(f"  PLATFORM read-before-return-value: {seen}/{N} in-graph reads saw "
              f"their own write -- ordering may now be modelled; re-read the "
              f"aliasing rules in AGENT.md before relying on it")
    for b in bufs:
        del b
    return seen


def _characterize_shared(label, build, which, msg_ok, msg_bad):
    qn, kn, vn, cn, sn, _ = _inputs()
    kv_k = torch.zeros(G, B, 1, KMAX, HEAD_DIM, dtype=torch.bfloat16, device="neuron")
    kv_v = torch.zeros_like(kv_k)
    run = build()
    for base in BASES:
        base_t = torch.tensor([[base]], dtype=torch.int32, device="neuron")
        run(qn, kn, vn, cn, sn, kv_k, kv_v, base_t)
    torch.neuron.synchronize()
    host = kv_k.cpu()
    per_group = [int(torch.count_nonzero(host[gi])) for gi in which]
    landed = sum(1 for n in per_group if n)
    print(f"  PLATFORM {label}: {landed}/{len(which)} groups written, "
          f"per_group_nz={per_group}")
    print(f"           {msg_ok if landed else msg_bad}")
    del kv_k, kv_v
    return landed


def main():
    print("rope_kv_dynamic multicall probe (cache is an input only; presence asserted)")
    failures = []
    per_layer = _contract_per_layer(failures)
    read_order = _characterize_read_order()
    slices = _characterize_shared(
        f"{N} DISTINCT SLICES of one shared buffer",
        _build_shared_slices,
        range(N),                      # this arm writes groups 0..N-1
        "writes now survive N distinct views of one base -- the per-layer buffers "
        "may be removable, but confirm with a 40-layer PREFILL_KV_MAP=1 run first",
        "all writes LOST (expected; this is the form that left the 40-layer cache "
        "40% populated) -- one distinct tensor per mutation",
    )
    whole = _characterize_shared(
        f"{N} calls through the IDENTICAL whole-tensor view",
        _build_shared_whole,
        GROUPS,                        # this arm writes the real GQA group ids
        "lands at this scale, as expected -- NOT a clearance: decode used this "
        "form and still lost 9 of 10 writes at 40 layers",
        "writes LOST -- stronger than previously measured; update AGENT.md",
    )

    if failures:
        for f in failures:
            print(f"FAIL {f}")
        raise SystemExit(1)

    if slices >= N and whole >= N and read_order >= N:
        print(
            "  NOTE: every characterization arm now passes. Either this probe has "
            "stopped being able to distinguish the shipped form from the broken "
            "ones, or the platform has changed. Confirm with a 40-layer "
            "PREFILL_KV_MAP=1 run before concluding the per-layer buffers are "
            "removable."
        )
    else:
        print(f"  (discriminating: per-layer={per_layer}/{N} vs "
              f"distinct-slices={slices}, read-before-return={read_order}/{N} "
              f"-- the per-layer buffers are load-bearing)")
    print(f"PASS: {N} in-place calls per graph, one buffer each, all writes "
          f"reached the caller across {len(BASES)} chunks")


if __name__ == "__main__":
    main()
