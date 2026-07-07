#!/usr/bin/env python3
"""De-risk nisa.affine_select for causal masking (bare-nki dialect).

Builds a causal-masked score tile the way the flash prefill kernel will: keep
qk[p, f] where (q_offset + p) >= (k_offset + f), else NEG_INF. Verifies the
affine predicate (channel_multiplier on partition, pattern step on free, offset,
cmp_op=greater_equal) matches a numpy causal mask.

Run in DLC:  python kernels/tests/probe_affine_causal.py   (needs neuron device)
"""
import numpy as np
import torch
import torch_neuronx  # noqa: F401
import nki
import nki.isa as nisa
import nki.language as nl

P = 8      # query rows in tile (partition)
F = 16     # key cols in block (free)
Q_OFF = 4  # global offset of this query tile
K_OFF = 0  # global offset of this key block
NEG = -30000.0


@nki.jit
def causal_tile(x, q_off, k_off):
    # x: [P, F] f32 input scores. Returns masked copy in HBM.
    out = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)
    xs = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x[0:P, 0:F])
    ms = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    # predicate value = 1*p + (-1)*f + (q_off - k_off); keep where >= 0
    nisa.affine_select(
        dst=ms,
        pattern=[[-1, F]],            # free axis: step -1 over F elements
        channel_multiplier=1,          # partition axis coefficient +1
        on_true_tile=xs,
        on_false_value=NEG,
        cmp_op=nl.greater_equal,
        offset=int(q_off - k_off),
    )
    nisa.dma_copy(dst=out[0:P, 0:F], src=ms)
    return out


def main():
    dev = "privateuseone:0"
    x = torch.arange(P * F, dtype=torch.float32).reshape(P, F)
    out = causal_tile[1](x.to(dev), Q_OFF, K_OFF)
    got = out.cpu().float().numpy()

    # numpy reference: keep where (Q_OFF+p) >= (K_OFF+f)
    ref = x.numpy().copy()
    for p in range(P):
        for f in range(F):
            if (Q_OFF + p) >= (K_OFF + f):
                pass
            else:
                ref[p, f] = NEG
    ok = np.allclose(got, ref)
    print(f"[probe] affine causal match={ok}")
    if not ok:
        print("GOT:\n", got.astype(int))
        print("REF:\n", ref.astype(int))
    else:
        print("kept-mask (1=kept):\n", (got > NEG + 1).astype(int))


if __name__ == "__main__":
    main()
