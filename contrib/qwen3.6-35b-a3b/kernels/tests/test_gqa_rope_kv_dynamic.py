"""Device validation for reusable runtime-offset GQA RoPE/KV updates."""

import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.dirname(HERE)
sys.path.insert(0, KERNELS)

import gqa_rope_kv_35b_ops  # noqa: E402,F401


CHUNK = 128
KMAX = 512
Q_HEADS = 4
HEAD_DIM = 256
ROPE_DIM = 64


def rotate_half(x):
    return torch.cat((-x[..., ROPE_DIM // 2 : ROPE_DIM], x[..., : ROPE_DIM // 2]), dim=-1)


def reference(query, key, cos, sin):
    q_rot = query[..., :ROPE_DIM].float()
    k_rot = key[..., :ROPE_DIM].float()
    q = torch.cat(
        [q_rot * cos + rotate_half(q_rot) * sin, query[..., ROPE_DIM:].float()],
        dim=-1,
    )
    key_cos = cos[:, 0]
    key_sin = sin[:, 0]
    k = torch.cat(
        [
            k_rot * key_cos + rotate_half(k_rot) * key_sin,
            key[..., ROPE_DIM:].float(),
        ],
        dim=-1,
    )
    return q, k


def main():
    torch.manual_seed(7)
    query = torch.randn(1, Q_HEADS, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
    key = torch.randn(1, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
    value = torch.randn(1, CHUNK, HEAD_DIM, dtype=torch.bfloat16)
    inv = 1.0 / (10_000_000.0 ** (torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM))
    freqs = torch.outer(torch.arange(KMAX).float(), inv)
    rope = torch.cat([freqs, freqs], dim=-1)
    cos, sin = rope.cos(), rope.sin()

    @torch.compile(backend="neuron", fullgraph=True, dynamic=False)
    def run(q, k, v, c, s, kc, vc, base):
        out, key_out = torch.ops.gqa35b.rope_kv_dynamic(
            q, k, v, c, s, kc, vc, base, 0, 1
        )
        # The kernel mutates kc/vc in place and does NOT return them. Mirror the
        # production caller (static_decode_35b.py `_gqa_prefill_chunk`) and read
        # them back inside the same graph: this is the ordering gate, i.e. that
        # functionalization sequences these reads after the mutating op.
        return out, key_out, kc.reshape(1, KMAX, HEAD_DIM), vc.reshape(1, KMAX, HEAD_DIM)

    qn = query.to("neuron")
    kn = key.to("neuron")
    vn = value.to("neuron")
    cn = cos.to("neuron")
    sn = sin.to("neuron")
    kc = torch.zeros(KMAX, HEAD_DIM, dtype=torch.bfloat16, device="neuron")
    vc = torch.zeros_like(kc)

    for base in (0, 256):
        # kc/vc are deliberately NOT rebound from the return value -- the kernel
        # mutates these exact tensors, so the same two buffers accumulate both
        # runtime offsets across iterations.
        out, key_out, kc_graph, vc_graph = run(
            qn,
            kn,
            vn,
            cn,
            sn,
            kc,
            vc,
            torch.tensor([[base]], dtype=torch.int32, device="neuron"),
        )
        torch.neuron.synchronize()
        q_ref, k_ref = reference(
            query[0].transpose(0, 1),
            key[0],
            cos[base : base + CHUNK, None, :],
            sin[base : base + CHUNK, None, :],
        )
        q_ref = q_ref.transpose(0, 1)
        q_actual = out[0].cpu()
        assert torch.allclose(q_actual, q_ref, rtol=2e-2, atol=2e-2)
        assert torch.allclose(
            key_out[0].cpu().float(),
            k_ref.float(),
            rtol=2e-2,
            atol=2e-2,
        )
        # Two independent views of the mutation: read back inside the graph
        # (kc_graph/vc_graph, what the prefill consumer actually sees) and from
        # the host after synchronize (kc/vc, the persistent cache buffers).
        for k_seen, v_seen in (
            (kc_graph[0].cpu(), vc_graph[0].cpu()),
            (kc.cpu(), vc.cpu()),
        ):
            assert torch.allclose(
                k_seen[base : base + CHUNK].float(),
                k_ref.float(),
                rtol=2e-2,
                atol=2e-2,
            )
            assert torch.equal(v_seen[base : base + CHUNK], value[0])

    assert torch.count_nonzero(kc.cpu()[128:256]) == 0
    assert torch.count_nonzero(kc.cpu()[384:]) == 0
    print("dynamic GQA RoPE/KV: two runtime offsets passed with one compiled graph")


if __name__ == "__main__":
    main()
