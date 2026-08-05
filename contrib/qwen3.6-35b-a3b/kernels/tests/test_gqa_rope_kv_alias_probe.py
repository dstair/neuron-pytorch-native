"""Does an in-graph read see rope_kv_dynamic's in-place write to the KV cache?

`rope_kv_dynamic` mutates the KV cache in place and does NOT return it, because
returning it re-exports the whole [B*max_seq_len, HD] buffer as a graph output
(1.74 GB of HBM traffic per 10-layer prefill region to publish 1024 changed
rows). That only works if the tensor handed to the op aliases the real buffer.

Measured on device: it does NOT if the argument is a *sliced* view. Passing
`kv_k[gi, :, 0].reshape(B * KMAX, HD)` and dropping the return loses the write
entirely -- the cache reads back all zeros, both in-graph and on the host, i.e.
silently wrong rather than merely slow. A whole-tensor `reshape` aliases the
buffer itself and the write lands, so the caller passes the entire flattened
cache and the kernel selects the layer with a static group_index (the same
convention as gqa35b::tail_stateful, gqa_tail_35b.py:107).

This probe pins that contract down: the shipped convention must be observable
both through a fresh slice off the base buffer and through the flat view that
was passed in. Value rows are copied verbatim by the kernel, so every check is
exact (`torch.equal`) -- a lost write shows up as zeros.

It runs at two scales. The small one is readable; the second is the production
40-layer prefill geometry (NUM_GQA=10, max_seq_len=20480, top group, last full
tile), where the runtime scalar DMA offset reaches flat row 409,471 -- element
offset 1.05e8, byte offset 2.1e8. Offset arithmetic that is fine at the small
scale can still truncate there, so the big case is the one that matters.
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

# (num_groups, kmax, group_index) -- write to an interior/top group so the
# neighbours must stay zero.
CONFIGS = (
    (3, 512, 1),
    (10, 20480, 9),  # production 40-layer prefill: D.NUM_GQA=10, max_seq_len=20480
)


def _build(read_back, g, kmax, gi):
    def body(q, k, v, c, s, kvk, kvv, base):
        ck = kvk.reshape(g * B * kmax, HEAD_DIM)
        cv = kvv.reshape(g * B * kmax, HEAD_DIM)
        out, key_out = torch.ops.gqa35b.rope_kv_dynamic(
            q, k, v, c, s, ck, cv, base, gi, g
        )
        if read_back == "base_slice":
            # What the prefill caller does: index the layer off the base buffer.
            return out, key_out, kvk[gi, :, 0], kvv[gi, :, 0]
        # Read through the same flat view that was passed to the op.
        return (
            out,
            key_out,
            ck.reshape(g, B, kmax, HEAD_DIM)[gi],
            cv.reshape(g, B, kmax, HEAD_DIM)[gi],
        )

    return torch.compile(body, backend="neuron", fullgraph=True, dynamic=False)


def _rope_tables(kmax):
    inv = 1.0 / (10_000_000.0 ** (torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM))
    freqs = torch.outer(torch.arange(kmax).float(), inv)
    rope = torch.cat([freqs, freqs], dim=-1)
    return rope.cos(), rope.sin()


def _check(g, kmax, gi, failures):
    torch.manual_seed(11)
    query = torch.randn(B, Q_HEADS, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
    key = torch.randn(B, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
    value = torch.randn(B, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
    cos, sin = _rope_tables(kmax)

    qn, kn, vn = query.to("neuron"), key.to("neuron"), value.to("neuron")
    cn, sn = cos.to("neuron"), sin.to("neuron")
    # Probe the bottom of the slab and the last tile that still fits, i.e. the
    # largest runtime offset production ever generates.
    bases = (0, kmax - CHUNK)
    tag = f"g={g},kmax={kmax},gi={gi}"

    for read_back in ("base_slice", "flat_view"):
        kv_k = torch.zeros(
            g, B, 1, kmax, HEAD_DIM, dtype=torch.bfloat16, device="neuron"
        )
        kv_v = torch.zeros_like(kv_k)
        run = _build(read_back, g, kmax, gi)
        for base in bases:
            base_t = torch.tensor([[base]], dtype=torch.int32, device="neuron")
            _, key_out, k_seen, v_seen = run(qn, kn, vn, cn, sn, kv_k, kv_v, base_t)
            torch.neuron.synchronize()
            key_out_c = key_out.cpu()
            rows = slice(base, base + CHUNK)

            # (a) the in-graph read -- what the prefill attention consumer sees
            for name, seen, want in (
                ("k", k_seen, key_out_c),
                ("v", v_seen, value),
            ):
                for b in range(B):
                    got = seen[b, rows].cpu()
                    if not torch.equal(got, want[b]):
                        failures.append(
                            f"{tag}/{read_back}/base={base}: in-graph {name}[b={b}] "
                            f"!= written rows (nonzero={int(torch.count_nonzero(got))}"
                            f"/{got.numel()})"
                        )
            # (b) the persistent buffer, read from the host after synchronize
            host_k = kv_k[gi, :, 0, rows].cpu()
            host_v = kv_v[gi, :, 0, rows].cpu()
            for b in range(B):
                if not torch.equal(host_k[b], key_out_c[b]):
                    failures.append(f"{tag}/{read_back}/base={base}: host kv_k[b={b}] stale")
                if not torch.equal(host_v[b], value[b]):
                    failures.append(f"{tag}/{read_back}/base={base}: host kv_v[b={b}] stale")

        # (c) the static group_index must confine the write to its own slab --
        # check the neighbours, which is where an off-by-one slab stride lands.
        for other in (gi - 1, gi + 1):
            if not 0 <= other < g:
                continue
            spill = int(torch.count_nonzero(kv_k[other].cpu()))
            if spill:
                failures.append(f"{tag}/{read_back}: wrote {spill} elems into group {other}")
        # (d) and to the two written row ranges only
        gap = kv_k[gi, :, 0, CHUNK : bases[1]].cpu()
        if int(torch.count_nonzero(gap)):
            failures.append(f"{tag}/{read_back}: wrote past the active rows")
        print(f"{tag}/{read_back}: checked")
        del kv_k, kv_v


def main():
    failures = []
    for g, kmax, gi in CONFIGS:
        _check(g, kmax, gi, failures)

    if failures:
        for f in failures:
            print(f"FAIL {f}")
        raise SystemExit(1)
    print("rope_kv_dynamic alias probe: in-place write observable without returning the cache")


if __name__ == "__main__":
    main()
