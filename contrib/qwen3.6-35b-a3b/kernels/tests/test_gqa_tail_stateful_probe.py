"""Do all ten decode GQA layers' in-place KV writes reach the caller's buffers?

The decode analog of test_gqa_rope_kv_multicall_probe.py, and the file whose
absence let the worst instance of this defect live longest. `tail_stateful`
appends the current K/V row to the cache in place and returns only attention
output; under `DECODE_FULLGRAPH=1` all ten GQA layers are traced into ONE graph,
and on this stack only the FIRST in-place mutation of a given buffer per graph is
carried back to the caller. Against a single shared [NUM_GQA, ...] cache that
meant nine of ten layers attended over all-zero K/V from ad342fbf (2026-07-17)
until 2026-08-05 -- `per_group_nz=[98304,0,0,0,0,0,0,0,0,0]` -- while producing
finite, plausible tokens and an unchanged `gen hash`. There was no probe for it
at all. This is that probe.

Read the header of test_gqa_rope_kv_alias_probe.py for why an in-place-write
probe must (a) keep the cache out of the graph's outputs, as production does, and
(b) assert on the buffer read back from the HOST. Both hold here. There is also a
free cross-invocation witness in this op: `tail_stateful` reads the cache
internally, and over an all-zero cache weighted-V -- and therefore the returned
attention -- is exactly zero, so a nonzero return at position >= 1 cannot happen
unless an EARLIER INVOCATION's write landed. Note what that does and does not
cover: it proves write-back between graph executions, not read-after-write
ordering inside one graph. Intra-graph ordering is characterized in
test_gqa_rope_kv_multicall_probe.py, where it is measured to depend on graph
output order.

Arms:

  CONTRACT (hard assert) -- ten per-layer buffers, ten calls in one graph, each
    with layer_index=0, driven over three sequential positions exactly as decode
    does. Every layer's every appended row must be present on the host and equal
    to the key/value it was handed (the kernel DMAs both verbatim, so this is an
    exact comparison and a lost write shows up as zeros). Keys differ per
    (layer, position) so a row landing in the wrong slot is not mistaken for a
    correct one.

  PLATFORM CHARACTERIZATION (measured, reported, never fails) -- one shared
    [NUM_GQA, ...] buffer, ten calls, two ways of handing it over:
      * the IDENTICAL whole-tensor view with layer_index=gi -- what decode
        actually did before the fix. HONEST LIMITATION: this form LANDS all ten
        at this scale, so the arm does not reproduce the production defect. The
        real 40-layer graph also carries MoE, DeltaNet and collectives at
        B=128, and there it kept only group 0. Treat a passing result as a
        floor, never as clearance -- the gate is a DECODE_KV_MAP=1 run.
      * N DISTINCT SLICES of the one base, which does lose writes here, giving
        the file one discriminating negative control.
    Reported rather than asserted because pinning a defect as "must stay broken"
    turns a platform fix into a red test.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.dirname(HERE)
sys.path.insert(0, KERNELS)

# Must be set BEFORE importing the kernel: Q_HEADS is a module constant read from
# the environment at import time. 2 is the TP=8/LNC=1 decode value.
os.environ.setdefault("GQA_Q_HEADS", "2")

import gqa_tail_35b  # noqa: E402
import gqa_tail_35b_ops  # noqa: E402,F401


Q_HEADS = gqa_tail_35b.Q_HEADS
HEAD_DIM = gqa_tail_35b.HEAD_DIM
ROPE_DIM = gqa_tail_35b.ROPE_DIM
S = 256          # production decode --max-seq-len
B = 2            # enough to exercise the per-b row base; production is 128
G = 10           # D.NUM_GQA at 40 layers -- all ten share one graph
POSITIONS = (0, 1, 2)   # sequential decode steps, as production drives it


def _build_per_layer():
    """Shipped form: G separate buffers, G calls, cache never an output."""

    def body(q, gate, qn, cos, sin, mask, keys, values, pos, *bufs):
        outs = []
        for slot in range(G):
            kvk, kvv = bufs[2 * slot], bufs[2 * slot + 1]
            # Whole-tensor reshape of THIS layer's own buffer. The kernel's row
            # base is (layer_index * B + b) * S, so a per-layer buffer is the
            # layer_index=0 case -- same as static_decode_35b.py.
            ck = kvk.reshape(B * S, HEAD_DIM)
            cv = kvv.reshape(B * S, HEAD_DIM)
            # [0] = attn_out. The op also returns the two caches (aliased, which
            # is what declares the mutation), but appending those would export the
            # cache as a graph output and destroy this probe's whole premise --
            # each read would get its own buffer and look correct regardless.
            outs.append(
                torch.ops.gqa35b.tail_stateful(
                    q, gate, qn, cos, sin, ck, cv, mask,
                    keys[slot], values[slot], pos, 0,
                )[0]
            )
        return tuple(outs)

    return torch.compile(body, backend="neuron", fullgraph=True, dynamic=False)


def _build_shared_whole():
    """The pre-fix decode form: ONE [G,...] buffer, G calls, the IDENTICAL
    whole-tensor view each time, with the kernel doing the per-layer offset.

    Lost 9 of 10 writes in the real 40-layer graph; lands at this scale.
    """

    def body(q, gate, qn, cos, sin, mask, keys, values, pos, kvk, kvv):
        outs = []
        for gi in range(G):
            ck = kvk.reshape(G * B * S, HEAD_DIM)
            cv = kvv.reshape(G * B * S, HEAD_DIM)
            # [0] = attn_out; see _build_per_layer on why the caches are dropped.
            outs.append(
                torch.ops.gqa35b.tail_stateful(
                    q, gate, qn, cos, sin, ck, cv, mask,
                    keys[gi], values[gi], pos, gi,
                )[0]
            )
        return tuple(outs)

    return torch.compile(body, backend="neuron", fullgraph=True, dynamic=False)


def _build_shared_slices():
    """ONE [G,...] buffer, G calls, a DISTINCT SLICE each. Loses writes even at
    this scale, so it is the file's negative control."""

    def body(q, gate, qn, cos, sin, mask, keys, values, pos, kvk, kvv):
        outs = []
        for gi in range(G):
            ck = kvk[gi].reshape(B * S, HEAD_DIM)
            cv = kvv[gi].reshape(B * S, HEAD_DIM)
            # [0] = attn_out; see _build_per_layer on why the caches are dropped.
            outs.append(
                torch.ops.gqa35b.tail_stateful(
                    q, gate, qn, cos, sin, ck, cv, mask,
                    keys[gi], values[gi], pos, 0,
                )[0]
            )
        return tuple(outs)

    return torch.compile(body, backend="neuron", fullgraph=True, dynamic=False)


def _static_inputs():
    """The per-call arguments that do not vary with position."""
    torch.manual_seed(17)
    query = torch.randn(B * Q_HEADS, HEAD_DIM)
    gate = torch.randn(B * Q_HEADS, HEAD_DIM)
    q_norm = torch.randn(1, HEAD_DIM) * 0.02
    cos = torch.ones(1, ROPE_DIM) * 0.6
    sin = torch.ones(1, ROPE_DIM) * 0.8
    return tuple(t.to("neuron") for t in (query, gate, q_norm, cos, sin))


def _kv_for_position(p):
    """Distinct K/V per (layer, position) so a misplaced row is not mistaken for
    a correct one. bf16 randn is nonzero everywhere, so a dropped write (zeros)
    is unambiguous."""
    torch.manual_seed(1000 + p)
    keys = torch.randn(G, B, HEAD_DIM, dtype=torch.bfloat16)
    values = torch.randn(G, B, HEAD_DIM, dtype=torch.bfloat16)
    return keys, values


def _mask_for_position(p):
    # Production stateful mask: rows strictly below the current position are the
    # ones written by earlier steps. At p=0 nothing is valid, exactly as on the
    # first decode step.
    return (torch.arange(S) < p).float().reshape(1, S).to("neuron")


def _contract_per_layer(failures):
    q, gate, qn, cos, sin = _static_inputs()
    bufs = []
    for _ in range(G):
        bufs.append(torch.zeros(B, 1, S, HEAD_DIM, dtype=torch.bfloat16, device="neuron"))
        bufs.append(torch.zeros(B, 1, S, HEAD_DIM, dtype=torch.bfloat16, device="neuron"))
    run = _build_per_layer()

    expected = {}      # (p, slot) -> (keys_host[slot], values_host[slot])
    for p in POSITIONS:
        keys, values = _kv_for_position(p)
        expected[p] = (keys, values)
        pos_t = torch.tensor([[p]], dtype=torch.int32, device="neuron")
        outs = run(
            q, gate, qn, cos, sin, _mask_for_position(p),
            keys.to("neuron"), values.to("neuron"), pos_t, *bufs,
        )
        torch.neuron.synchronize()

        for slot in range(G):
            attn = outs[slot].cpu()
            if not bool(torch.isfinite(attn).all()):
                failures.append(f"p={p} layer={slot}: attention output not finite")
            # In-graph witness, free in this op: weighted-V over an all-zero
            # cache is exactly zero, so a nonzero return at p>=1 proves an
            # earlier step's write was visible to this graph.
            if p >= 1 and int(torch.count_nonzero(attn)) == 0:
                failures.append(
                    f"p={p} layer={slot}: attention returned all zeros -- the "
                    f"cache it read was empty, so an earlier in-place write was "
                    f"not visible in-graph"
                )

    # THE ASSERTION THAT MATTERS: every layer's every appended row, present in
    # the CALLER's buffer, read back from the host.
    for slot in range(G):
        host_k = bufs[2 * slot][:, 0].cpu()      # [B, S, HD]
        host_v = bufs[2 * slot + 1][:, 0].cpu()
        for p in POSITIONS:
            keys, values = expected[p]
            for b in range(B):
                got_k, want_k = host_k[b, p], keys[slot, b]
                if not torch.equal(got_k, want_k):
                    nz = int(torch.count_nonzero(got_k))
                    failures.append(
                        f"layer={slot} p={p} b={b}: cached_k row MISSING from the "
                        f"caller's buffer (nonzero={nz}/{got_k.numel()})"
                    )
                got_v, want_v = host_v[b, p], values[slot, b]
                if not torch.equal(got_v, want_v):
                    nz = int(torch.count_nonzero(got_v))
                    failures.append(
                        f"layer={slot} p={p} b={b}: cached_v row MISSING from the "
                        f"caller's buffer (nonzero={nz}/{got_v.numel()})"
                    )
        # ...and wrote nowhere else.
        tail = int(torch.count_nonzero(host_k[:, len(POSITIONS):]))
        if tail:
            failures.append(
                f"layer={slot}: wrote {tail} elems past position {len(POSITIONS) - 1}"
            )

    per_layer_nz = [int(torch.count_nonzero(bufs[2 * i].cpu())) for i in range(G)]
    landed = sum(1 for n in per_layer_nz if n)
    print(f"  CONTRACT per-layer buffers: {landed}/{G} layers written over "
          f"{len(POSITIONS)} positions, per_layer_nz={per_layer_nz}")
    for b in bufs:
        del b
    return landed


def _characterize_shared(label, build, note):
    q, gate, qn, cos, sin = _static_inputs()
    kv_k = torch.zeros(G, B, 1, S, HEAD_DIM, dtype=torch.bfloat16, device="neuron")
    kv_v = torch.zeros_like(kv_k)
    run = build()
    for p in POSITIONS:
        keys, values = _kv_for_position(p)
        pos_t = torch.tensor([[p]], dtype=torch.int32, device="neuron")
        run(
            q, gate, qn, cos, sin, _mask_for_position(p),
            keys.to("neuron"), values.to("neuron"), pos_t, kv_k, kv_v,
        )
    torch.neuron.synchronize()
    host = kv_k.cpu()
    per_group = [int(torch.count_nonzero(host[gi])) for gi in range(G)]
    landed = sum(1 for n in per_group if n)
    print(f"  PLATFORM shared buffer, {label}: {landed}/{G} layers written, "
          f"per_group_nz={per_group}")
    print(f"           {note}")
    del kv_k, kv_v
    return landed


def main():
    print(f"tail_stateful multicall probe (G={G}, B={B}, S={S}, "
          f"Q_HEADS={Q_HEADS}; cache is an input only, presence asserted)")
    failures = []
    per_layer = _contract_per_layer(failures)
    whole = _characterize_shared(
        "IDENTICAL whole-tensor view (the pre-fix decode form)",
        _build_shared_whole,
        "landing here does NOT clear this form: at 40 layers / BS=128 it kept only "
        "group 0 (per_group_nz=[98304,0,...,0]). The gate is DECODE_KV_MAP=1.",
    )
    slices = _characterize_shared(
        f"{G} DISTINCT SLICES of the one base",
        _build_shared_slices,
        "expected to lose writes -- this is the arm that discriminates at probe "
        "scale; if it lands, the file has stopped being able to tell the forms "
        "apart and the DECODE_KV_MAP=1 gate is the only remaining check.",
    )

    if failures:
        for f in failures:
            print(f"FAIL {f}")
        raise SystemExit(1)

    if slices >= G:
        print(
            "  NOTE: both shared-buffer arms landed ALL their writes, so this probe "
            "no longer distinguishes the shipped form from either broken one. Do "
            "not conclude the per-layer buffers are removable from this file -- "
            "run a 40-layer DECODE_KV_MAP=1 decode and require all ten groups "
            "non-zero."
        )
    else:
        print(f"  (discriminating: per-layer={per_layer}/{G} vs "
              f"distinct-slices={slices}/{G}; whole-view={whole}/{G} does not "
              f"discriminate at this scale -- the per-layer buffers are load-bearing)")
    print(f"PASS: {G} stateful KV appends per graph, one buffer each, every row "
          f"reached the caller across {len(POSITIONS)} positions")


if __name__ == "__main__":
    main()
