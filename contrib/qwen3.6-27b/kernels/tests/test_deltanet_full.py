"""Standalone correctness + benchmark for nki_deltanet_full."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # kernels/ for bare-name imports
import math
import time
import torch
import torch.nn.functional as F

from deltanet_full import nki_deltanet_full

K_DIM = 128
V_DIM = 128
K_HEADS = 4
V_HEADS = 12
HEAD_GROUP = V_HEADS // K_HEADS
QKV_DIM = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM
RMS_EPS = 1e-6


def cpu_reference(state, mixed_qkv, conv_state, conv_weight, conv_bias,
                  a_out, b_out, z, A_log, dt_bias, norm_weight):
    conv_in = torch.cat(
        [conv_state.float(), mixed_qkv.float().unsqueeze(-1)], dim=-1,
    )
    new_cs = conv_in[:, 1:].to(conv_state.dtype)
    a = F.silu((conv_in * conv_weight.float()).sum(-1) + conv_bias.float())

    q = a[: K_HEADS * K_DIM].reshape(K_HEADS, K_DIM)
    k = a[K_HEADS * K_DIM : 2 * K_HEADS * K_DIM].reshape(K_HEADS, K_DIM)
    v = a[2 * K_HEADS * K_DIM :].reshape(V_HEADS, V_DIM)

    q = q / (q.pow(2).sum(-1, keepdim=True) + RMS_EPS).sqrt() * (1 / math.sqrt(K_DIM))
    k = k / (k.pow(2).sum(-1, keepdim=True) + RMS_EPS).sqrt()
    q = q.repeat_interleave(HEAD_GROUP, dim=0)
    k = k.repeat_interleave(HEAD_GROUP, dim=0)

    g = -torch.exp(A_log.float()) * F.softplus(a_out.float() + dt_bias.float())
    beta = torch.sigmoid(b_out.float())

    s = state.float().reshape(V_HEADS, K_DIM, V_DIM).clone()
    out_f = torch.empty(V_HEADS, V_DIM, dtype=torch.float32)
    for h in range(V_HEADS):
        s[h] = s[h] * torch.exp(g[h])
        kv = (s[h].T @ k[h]).reshape(V_DIM)
        delta = (v[h] - kv) * beta[h]
        s[h] = s[h] + torch.outer(k[h], delta)
        out_f[h] = s[h].T @ q[h]
    new_state = s.reshape(V_HEADS * K_DIM, V_DIM)

    # RMSNormGated: out = norm_weight * silu(z) * rsqrt(mean(out^2)+eps) * out
    var = out_f.pow(2).mean(-1, keepdim=True)
    norm = out_f * torch.rsqrt(var + RMS_EPS)
    gated = norm * norm_weight.float() * F.silu(z.float())
    return new_state, new_cs, gated.to(z.dtype)


if __name__ == "__main__":
    torch.manual_seed(11)
    device = torch.device("neuron")

    inputs = dict(
        state=torch.randn(V_HEADS * K_DIM, V_DIM, dtype=torch.float32) * 0.1,
        mixed_qkv=torch.randn(QKV_DIM, dtype=torch.bfloat16),
        conv_state=torch.randn(QKV_DIM, 3, dtype=torch.bfloat16),
        conv_weight=torch.randn(QKV_DIM, 4, dtype=torch.float32) * 0.5,
        conv_bias=torch.randn(QKV_DIM, dtype=torch.float32) * 0.1,
        a_out=torch.randn(V_HEADS, dtype=torch.float32),
        b_out=torch.randn(V_HEADS, dtype=torch.float32),
        z=torch.randn(V_HEADS, V_DIM, dtype=torch.bfloat16),
        A_log=torch.randn(V_HEADS, dtype=torch.float32) * 0.5,
        dt_bias=torch.randn(V_HEADS, dtype=torch.float32),
        norm_weight=torch.randn(V_DIM, dtype=torch.float32) * 0.1 + 1.0,
    )

    print("=== nki_deltanet_full ===")
    rs, rcs, ro = cpu_reference(**inputs)

    ns, ncs, no = nki_deltanet_full(
        **{k: v.to(device) for k, v in inputs.items()},
    )
    ns, ncs, no = (t.cpu() for t in (ns, ncs, no))

    sd = (ns.float() - rs.float()).abs().max().item()
    cd = (ncs.float() - rcs.float()).abs().max().item()
    od = (no.float() - ro.float()).abs().max().item()
    print(f"  state      max_diff = {sd:.6f}")
    print(f"  conv_state max_diff = {cd:.6f}")
    print(f"  output     max_diff = {od:.6f}")
    assert sd < 1e-2 and cd < 1e-2 and od < 5e-2, f"FAIL sd={sd} cd={cd} od={od}"
    print("  PASS ✓")

    print("\n=== Benchmark ===")
    di = {k: v.to(device) for k, v in inputs.items()}
    for _ in range(3):
        nki_deltanet_full(**di)
    N = 100
    t0 = time.time()
    for _ in range(N):
        nki_deltanet_full(**di)
    print(f"  deltanet_full: {(time.time() - t0) / N * 1000:.3f} ms/call (avg of {N})")
