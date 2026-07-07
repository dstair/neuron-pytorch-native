#!/usr/bin/env python3
"""De-risk a DYNAMIC-offset causal mask (runtime q_base scalar), needed for
chunked/bucketed prefill: one NEFF handles every chunk, the causal boundary
shifts at runtime via q_base instead of being baked per chunk.

Builds, for a score tile x[P,F] whose global query rows start at q_base and whose
key cols start at k0: keep x[p,f] where (q_base + p) >= (k0 + f), else NEG_INF.
Uses nisa.iota to make the static tile D[p,f] = p - f (channel_multiplier=1,
pattern=[[-1,F]]), broadcasts the runtime scalar q_base to a [P,1] column via the
ones-matmul idiom, and forms the additive causal bias with tensor ops only.

Run in DLC:  python kernels/tests/probe_dyn_causal.py
"""
import numpy as np
import torch
import torch_neuronx  # noqa: F401
import nki
import nki.isa as nisa
import nki.language as nl

P = 8
F = 16
K0 = 0            # static key-block start (compile-time loop index in the real kernel)
NEG = -30000.0


@nki.jit
def dyn_causal(x, q_base):
    # x: [P, F] scores; q_base: [1,1] f32 runtime scalar (global start of query rows)
    out = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)
    xs = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs, src=x[0:P, 0:F])

    # D[p,f] = p - f  (static): channel_multiplier=1 on partition, step -1 on free
    D = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=D, pattern=[[-1, F]], channel_multiplier=1, offset=K0 * -1 + 0)
    # NOTE offset here folds the static k0: value = p - f - k0. Keep where value + q_base >= 0.

    # broadcast q_base [1,1] -> [P,1] via ones-matmul (device needs matching partitions)
    qb1 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=qb1, src=q_base[0:1, 0:1])
    onesP = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=onesP, value=1.0)
    qb_ps = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=qb_ps, stationary=onesP, moving=qb1)   # [P,1] all = q_base
    qb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qb, src=qb_ps)

    # Dq = D + q_base  (per-partition add of the [P,1] column, free-broadcast)
    Dq = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=Dq, data=D, op0=nl.add, operand0=qb)

    # bias = (Dq < 0) * NEG   ; keep where Dq >= 0
    masked_flag = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=masked_flag, data=Dq, op0=nl.less, operand0=0.0)   # 1.0 where masked
    nisa.tensor_scalar(dst=masked_flag, data=masked_flag, op0=nl.multiply, operand0=NEG)
    res = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=res, data1=xs, data2=masked_flag, op=nl.add)
    nisa.dma_copy(dst=out[0:P, 0:F], src=res)
    return out


def main():
    dev = "privateuseone:0"
    for q_base in [0, 4, 100]:
        x = torch.arange(P * F, dtype=torch.float32).reshape(P, F)
        qb = torch.tensor([[float(q_base)]], dtype=torch.float32)
        out = dyn_causal[1](x.to(dev), qb.to(dev)).cpu().float().numpy()
        ref = x.numpy().copy()
        for p in range(P):
            for f in range(F):
                if (q_base + p) >= (K0 + f):
                    pass
                else:
                    ref[p, f] = ref[p, f] + NEG
        ok = np.allclose(out, ref, atol=1e-3)
        print(f"[dyn-causal] q_base={q_base:4d}  match={ok}")
        if not ok:
            print(" GOT:\n", out.astype(int))
            print(" REF:\n", ref.astype(int))


if __name__ == "__main__":
    main()
