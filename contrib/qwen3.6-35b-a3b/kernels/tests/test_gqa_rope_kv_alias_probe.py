"""Does rope_kv_dynamic's in-place KV write reach the CALLER's buffer?

`rope_kv_dynamic` mutates the KV cache in place and does NOT return it, because
returning it re-exports the whole [B*max_seq_len, HD] buffer as a graph output
(1.74 GB of HBM traffic per 10-layer prefill region to publish 1024 changed
rows). That only works if the tensor handed to the op aliases the real buffer
*and* the mutation is carried back out of the traced graph.

READ THIS BEFORE EDITING -- the earlier version of this probe passed on a build
that silently lost 60% of its KV writes. It did so because it returned
`kvk[gi, :, 0]` as a graph output and asserted on that. Exporting the cache
gives the read its own output buffer, so the assertion passed whether or not the
*caller's* tensor was ever updated -- and exporting it is precisely what
production does not do. A probe for an in-place write must therefore:

  1. compile a graph whose outputs do NOT include the cache, exactly as
     production does (`_gqa_prefill_chunk` consumes its cache read internally
     and returns only attention results), and
  2. assert on the buffer read back from the HOST after synchronize.

Anything else validates a graph that is not shipped. The in-graph read is still
exercised -- it is reduced to a small row digest rather than exported whole --
so a read hoisted above its own write still shows up, as a zero digest row.

Scope: ONE mutating call per graph. That is all this file covers, and one
mutation per buffer always survives on this stack. The number of mutations per
graph is the axis that actually broke, and it lives in
test_gqa_rope_kv_multicall_probe.py -- do not treat this file as sufficient.

Cases, split by what they are for:

  CONTRACT (hard assert, this is the shipped path)
    per-layer buffer, num_groups=1/group_index=0, at the small readable scale
    and at production's max_seq_len=20480.

  CONTRACT (hard assert, offset arithmetic)
    shared [NUM_GQA, ...] buffer with num_groups=10/group_index=9, ONE call, so
    the write is safe. This is the largest runtime DMA offset the kernel is ever
    asked for -- flat row 409,471, element offset 1.05e8, byte offset 2.1e8 --
    which per-layer buffers no longer generate but the kernel still supports.

  PLATFORM CHARACTERIZATION (measured and reported, never fails)
    the same single mutation handed over as a *sliced* view
    (`kv_k[gi, :, 0].reshape(...)`) rather than a whole-tensor reshape.

CORRECTION 2026-08-05: an earlier version of this arm asserted that a sliced view
"does not alias, so its write is lost". That is wrong, and it was never measured.
A single sliced mutation lands, and always did -- the pre-2026-07-30 prefill path
mutated sliced views and populated the cache 100%. What loses writes is mutating
SEVERAL DISTINCT VIEWS OF ONE BASE tensor in one graph (measured: 3 of 3 lost),
which is a property of the sharing, not of the slicing. That belongs to the
multicall file, and it is why the shipped code uses one distinct buffer per GQA
layer rather than one slice per GQA layer.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.dirname(HERE)
sys.path.insert(0, KERNELS)

import gqa_rope_kv_35b_ops  # noqa: E402,F401


CHUNK = 128
Q_HEADS = 2
HEAD_DIM = 256
ROPE_DIM = 64
B = 2

# (label, kmax, num_groups, group_index)
#   num_groups == 1 is the shipped per-layer-buffer form.
#   num_groups == 10 exercises the kernel's slab arithmetic at the largest
#   offset it was ever driven with; safe to assert because it is one call.
CONTRACT_CASES = (
    ("per-layer, small", 512, 1, 0),
    ("per-layer, production kmax", 20480, 1, 0),
    ("shared slab, top group, max offset", 20480, 10, 9),
)


def _build_contract(kmax, num_groups, group_index):
    """Graph shaped like production: the cache is an INPUT ONLY, never an output."""

    def body(q, k, v, c, s, kvk, kvv, base):
        # Whole-tensor reshape -- this is what aliases the buffer.
        ck = kvk.reshape(num_groups * B * kmax, HEAD_DIM)
        cv = kvv.reshape(num_groups * B * kmax, HEAD_DIM)
        q_out, key_out = torch.ops.gqa35b.rope_kv_dynamic(
            q, k, v, c, s, ck, cv, base, group_index, num_groups
        )
        # The in-graph read that production feeds into attention. Reduced to a
        # per-row digest so the cache is not exported: a read hoisted above its
        # own write leaves the digest zero on the rows it should have seen.
        k_seen = kvk[group_index, :, 0]                      # [B, kmax, HD]
        digest = k_seen.float().sum(dim=-1)                  # [B, kmax]
        return q_out, key_out, digest

    return torch.compile(body, backend="neuron", fullgraph=True, dynamic=False)


def _build_sliced(kmax, num_groups, group_index):
    """The known-bad form: hand the op a SLICED view instead of the whole tensor."""

    def body(q, k, v, c, s, kvk, kvv, base):
        ck = kvk[group_index, :, 0].reshape(B * kmax, HEAD_DIM)
        cv = kvv[group_index, :, 0].reshape(B * kmax, HEAD_DIM)
        q_out, key_out = torch.ops.gqa35b.rope_kv_dynamic(
            q, k, v, c, s, ck, cv, base, 0, 1
        )
        return q_out, key_out

    return torch.compile(body, backend="neuron", fullgraph=True, dynamic=False)


def _rope_tables(kmax):
    inv = 1.0 / (10_000_000.0 ** (torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM))
    freqs = torch.outer(torch.arange(kmax).float(), inv)
    rope = torch.cat([freqs, freqs], dim=-1)
    return rope.cos(), rope.sin()


def _inputs(kmax, seed=11):
    torch.manual_seed(seed)
    query = torch.randn(B, Q_HEADS, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
    key = torch.randn(B, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
    value = torch.randn(B, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
    cos, sin = _rope_tables(kmax)
    return (
        query.to("neuron"), key.to("neuron"), value.to("neuron"),
        cos.to("neuron"), sin.to("neuron"), value,
    )


def _check_contract(label, kmax, num_groups, group_index, failures):
    qn, kn, vn, cn, sn, value = _inputs(kmax)
    # Bottom of the slab and the last tile that still fits: the smallest and
    # largest runtime offsets production generates.
    bases = (0, kmax - CHUNK)
    tag = f"{label} (kmax={kmax},g={num_groups},gi={group_index})"

    kv_k = torch.zeros(
        num_groups, B, 1, kmax, HEAD_DIM, dtype=torch.bfloat16, device="neuron"
    )
    kv_v = torch.zeros_like(kv_k)
    run = _build_contract(kmax, num_groups, group_index)

    for base in bases:
        base_t = torch.tensor([[base]], dtype=torch.int32, device="neuron")
        _, key_out, digest = run(qn, kn, vn, cn, sn, kv_k, kv_v, base_t)
        torch.neuron.synchronize()
        key_out_c = key_out.cpu()
        rows = slice(base, base + CHUNK)

        # (a) THE ASSERTION THAT MATTERS: the caller's own buffer, on the host.
        # Value rows are copied verbatim by the kernel, so this is exact and a
        # lost write shows up as zeros.
        host_k = kv_k[group_index, :, 0, rows].cpu()
        host_v = kv_v[group_index, :, 0, rows].cpu()
        for b in range(B):
            if not torch.equal(host_k[b], key_out_c[b]):
                nz = int(torch.count_nonzero(host_k[b]))
                failures.append(
                    f"{tag}/base={base}: caller's kv_k[b={b}] did not receive the "
                    f"write (nonzero={nz}/{host_k[b].numel()})"
                )
            if not torch.equal(host_v[b], value[b]):
                nz = int(torch.count_nonzero(host_v[b]))
                failures.append(
                    f"{tag}/base={base}: caller's kv_v[b={b}] did not receive the "
                    f"write (nonzero={nz}/{host_v[b].numel()})"
                )

        # (b) the in-graph read saw its own write -- catches a hoisted read
        # without exporting the cache.
        seen_rows = digest.cpu()[:, rows]
        for b in range(B):
            if int(torch.count_nonzero(seen_rows[b])) == 0:
                failures.append(
                    f"{tag}/base={base}: in-graph read of b={b} saw zeros on the "
                    f"rows it had just written (read hoisted above the write?)"
                )

    # (c) the write stayed inside its own slab and its own rows.
    for other in (group_index - 1, group_index + 1):
        if not 0 <= other < num_groups:
            continue
        spill = int(torch.count_nonzero(kv_k[other].cpu()))
        if spill:
            failures.append(f"{tag}: wrote {spill} elems into group {other}")
    gap = kv_k[group_index, :, 0, CHUNK : bases[1]].cpu()
    if int(torch.count_nonzero(gap)):
        failures.append(f"{tag}: wrote past the active rows")

    print(f"  CONTRACT {tag}: ok")
    del kv_k, kv_v


def _characterize_sliced(kmax=512):
    """Measure, do not assert: how a SINGLE mutation of a sliced view behaves.

    Expected to land (2026-08-05). If it ever stops landing, the shipped
    whole-tensor form is still safe, but `_gqa_prefill_chunk`'s neighbours and the
    27B port should be re-checked for slice-shaped cache arguments.
    """
    qn, kn, vn, cn, sn, _ = _inputs(kmax)
    kv_k = torch.zeros(1, B, 1, kmax, HEAD_DIM, dtype=torch.bfloat16, device="neuron")
    kv_v = torch.zeros_like(kv_k)
    run = _build_sliced(kmax, 1, 0)
    base_t = torch.tensor([[0]], dtype=torch.int32, device="neuron")
    run(qn, kn, vn, cn, sn, kv_k, kv_v, base_t)
    torch.neuron.synchronize()
    landed = int(torch.count_nonzero(kv_k.cpu()))
    if landed:
        print(f"  PLATFORM single sliced-view arg: write LANDED ({landed} elems, "
              "expected) -- slicing alone is not the hazard; sharing one base "
              "across several mutations is (test_gqa_rope_kv_multicall_probe.py)")
    else:
        print("  PLATFORM single sliced-view arg: write LOST -- this is a CHANGE "
              "from 2026-08-05, when a single sliced mutation landed. The shipped "
              "whole-tensor form is unaffected, but re-check AGENT.md.")
    del kv_k, kv_v


def main():
    failures = []
    print("rope_kv_dynamic alias probe (cache is an input only, asserts on the host)")
    for label, kmax, g, gi in CONTRACT_CASES:
        _check_contract(label, kmax, g, gi, failures)
    _characterize_sliced()

    if failures:
        for f in failures:
            print(f"FAIL {f}")
        raise SystemExit(1)
    print("PASS: in-place write reaches the caller's buffer without returning the cache")
    print("NOTE: one mutation per graph only -- see test_gqa_rope_kv_multicall_probe.py")


if __name__ == "__main__":
    main()
