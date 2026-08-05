"""Are several rope_kv_dynamic calls on ONE cache ordered correctly in one graph?

`rope_kv_dynamic` mutates the KV cache in place and does not return it, so the
only thing that orders a layer's cache read after its own cache write is the
op's `mutates_args` declaration -- there is no data dependency through a
returned tensor to enforce it (see test_gqa_rope_kv_alias_probe.py for why the
return was dropped).

test_gqa_rope_kv_alias_probe.py covers one write/read pair per graph. Production
does more than that: at 40 layers with --prefill-splits 4 each compiled segment
spans 10 layers and so contains 2-3 GQA layers, i.e. 2-3 mutating calls against
the *same* kv_k/kv_v buffer, each followed by its own read. That is what this
probe covers, at two chunk offsets so the second invocation must observe the
first one's rows as well.

The failure it is built to catch is a read hoisted above its own write: the
layer would attend over the previous chunks but not the rows it just wrote, a
silent accuracy loss rather than a crash. Value rows are copied verbatim by the
kernel, so `torch.equal` against the input is exact and a hoisted read shows up
as zeros.
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


def _build():
    def body(q, k, v, c, s, kvk, kvv, base):
        ck = kvk.reshape(G * B * KMAX, HEAD_DIM)
        cv = kvv.reshape(G * B * KMAX, HEAD_DIM)
        outs = []
        for gi in GROUPS:
            _, key_out = torch.ops.gqa35b.rope_kv_dynamic(
                q, k, v, c, s, ck, cv, base, gi, G
            )
            # Exactly what _gqa_prefill_chunk does: read this layer's slab back
            # off the base buffer straight after its own mutating call.
            outs.extend([key_out, kvk[gi, :, 0], kvv[gi, :, 0]])
        return tuple(outs)

    return torch.compile(body, backend="neuron", fullgraph=True, dynamic=False)


def main():
    torch.manual_seed(13)
    query = torch.randn(B, Q_HEADS, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
    key = torch.randn(B, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
    value = torch.randn(B, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
    inv = 1.0 / (10_000_000.0 ** (torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM))
    freqs = torch.outer(torch.arange(KMAX).float(), inv)
    rope = torch.cat([freqs, freqs], dim=-1)
    cos, sin = rope.cos(), rope.sin()

    qn, kn, vn = query.to("neuron"), key.to("neuron"), value.to("neuron")
    cn, sn = cos.to("neuron"), sin.to("neuron")
    kv_k = torch.zeros(G, B, 1, KMAX, HEAD_DIM, dtype=torch.bfloat16, device="neuron")
    kv_v = torch.zeros_like(kv_k)
    run = _build()

    failures = []
    # Two consecutive chunks, as prefill does. The second invocation's in-graph
    # read must show both chunks: its own rows AND the previous chunk's.
    key_by_base = {}
    for chunk_idx, base in enumerate((0, CHUNK)):
        base_t = torch.tensor([[base]], dtype=torch.int32, device="neuron")
        res = run(qn, kn, vn, cn, sn, kv_k, kv_v, base_t)
        torch.neuron.synchronize()
        for slot, gi in enumerate(GROUPS):
            key_out, k_seen, v_seen = res[3 * slot : 3 * slot + 3]
            key_by_base[(base, gi)] = key_out.cpu()
            # every chunk written so far must be visible to this layer's read
            for prev, pbase in enumerate((0, CHUNK)[: chunk_idx + 1]):
                rows = slice(pbase, pbase + CHUNK)
                k_want = key_by_base[(pbase, gi)]
                k_got = k_seen[:, rows].cpu()
                v_got = v_seen[:, rows].cpu()
                for b in range(B):
                    if not torch.equal(k_got[b], k_want[b]):
                        nz = int(torch.count_nonzero(k_got[b]))
                        failures.append(
                            f"base={base} gi={gi}: in-graph k missing chunk {prev} "
                            f"(nonzero={nz}/{k_got[b].numel()})"
                        )
                    if not torch.equal(v_got[b], value[b]):
                        nz = int(torch.count_nonzero(v_got[b]))
                        failures.append(
                            f"base={base} gi={gi}: in-graph v missing chunk {prev} "
                            f"(nonzero={nz}/{v_got[b].numel()})"
                        )

    # The three calls must not have written into each other's slabs, nor into
    # any layer that this segment does not own.
    for other in range(G):
        if other in GROUPS:
            continue
        spill = int(torch.count_nonzero(kv_k[other].cpu()))
        if spill:
            failures.append(f"wrote {spill} elems into unowned group {other}")
    for gi in GROUPS:
        tail = int(torch.count_nonzero(kv_k[gi, :, 0, 2 * CHUNK :].cpu()))
        if tail:
            failures.append(f"gi={gi}: wrote {tail} elems past the active rows")

    if failures:
        for f in failures:
            print(f"FAIL {f}")
        raise SystemExit(1)
    print(
        f"rope_kv_dynamic multicall probe: {len(GROUPS)} in-place calls per graph "
        "ordered correctly across 2 chunks"
    )


if __name__ == "__main__":
    main()
