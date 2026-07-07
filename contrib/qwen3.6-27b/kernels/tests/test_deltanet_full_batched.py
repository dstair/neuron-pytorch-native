"""Standalone correctness + benchmark for nki_deltanet_full_batched (device).

Safety net for the DMA-coalescing rework: the batched kernel MUST match the
single-batch cpu_reference applied per row. Set KERNEL_MODULE env to test a
variant (default deltanet_full_batched; set to deltanet_full_batched_v2 etc.).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # kernels/ for bare-name imports
import math
import os
import time
import torch
import torch.nn.functional as F

K_DIM = 128
V_DIM = 128
K_HEADS = 4
V_HEADS = 12
HEAD_GROUP = V_HEADS // K_HEADS
QKV_DIM = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM
RMS_EPS = 1e-6


def cpu_reference(state, mixed_qkv, conv_state, conv_weight, conv_bias,
                  a_out, b_out, z, A_log, dt_bias, norm_weight):
    """Single-row reference (mirrors test_deltanet_full)."""
    conv_in = torch.cat([conv_state.float(), mixed_qkv.float().unsqueeze(-1)], dim=-1)
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
    var = out_f.pow(2).mean(-1, keepdim=True)
    norm = out_f * torch.rsqrt(var + RMS_EPS)
    gated = norm * norm_weight.float() * F.silu(z.float())
    return new_state, new_cs, gated.to(z.dtype)


if __name__ == "__main__":
    mod_name = os.environ.get("KERNEL_MODULE", "deltanet_full_batched")
    kmod = __import__(mod_name)
    kernel = kmod.nki_deltanet_full_batched
    B = int(os.environ.get("BS", "8"))
    print(f"=== {mod_name}.nki_deltanet_full_batched  B={B} ===")
    torch.manual_seed(11)
    device = torch.device("neuron")

    # Per-row random inputs; shared weights identical across rows.
    state = torch.randn(B, V_HEADS * K_DIM, V_DIM, dtype=torch.float32) * 0.1
    mixed_qkv = torch.randn(B, QKV_DIM, dtype=torch.bfloat16)
    conv_state = torch.randn(B, QKV_DIM, 3, dtype=torch.bfloat16)
    a_out = torch.randn(B, V_HEADS, dtype=torch.float32)
    b_out = torch.randn(B, V_HEADS, dtype=torch.float32)
    z = torch.randn(B, V_HEADS, V_DIM, dtype=torch.bfloat16)
    conv_weight = torch.randn(QKV_DIM, 4, dtype=torch.float32) * 0.5
    conv_bias = torch.randn(QKV_DIM, dtype=torch.float32) * 0.1
    A_log = torch.randn(V_HEADS, dtype=torch.float32) * 0.5
    dt_bias = torch.randn(V_HEADS, dtype=torch.float32)
    norm_weight = torch.randn(V_DIM, dtype=torch.float32) * 0.1 + 1.0

    # Reference per row.
    rs_l, rcs_l, ro_l = [], [], []
    for bi in range(B):
        rs, rcs, ro = cpu_reference(
            state[bi], mixed_qkv[bi], conv_state[bi], conv_weight, conv_bias,
            a_out[bi], b_out[bi], z[bi], A_log, dt_bias, norm_weight)
        rs_l.append(rs); rcs_l.append(rcs); ro_l.append(ro)
    rs = torch.stack(rs_l).reshape(B * V_HEADS * K_DIM, V_DIM)
    rcs = torch.stack(rcs_l).reshape(B * QKV_DIM, 3)
    ro = torch.stack(ro_l).reshape(B * V_HEADS, V_DIM)

    # Kernel: flatten batch into dim 0 (matches static_decode wiring).
    ns, ncs, no = kernel(
        state.reshape(B * V_HEADS * K_DIM, V_DIM).to(device),
        mixed_qkv.reshape(B * QKV_DIM).to(device),
        conv_state.reshape(B * QKV_DIM, 3).to(device),
        conv_weight.to(device), conv_bias.to(device),
        a_out.reshape(B * V_HEADS).to(device), b_out.reshape(B * V_HEADS).to(device),
        z.reshape(B * V_HEADS, V_DIM).to(device),
        A_log.to(device), dt_bias.to(device), norm_weight.to(device),
    )
    ns, ncs, no = (t.cpu() for t in (ns, ncs, no))

    sd = (ns.float() - rs.float()).abs().max().item()
    cd = (ncs.float() - rcs.float()).abs().max().item()
    od = (no.float() - ro.float()).abs().max().item()
    print(f"  state      max_diff = {sd:.6f}")
    print(f"  conv_state max_diff = {cd:.6f}")
    print(f"  output     max_diff = {od:.6f}")
    ok = sd < 1e-2 and cd < 1e-2 and od < 5e-2
    print("  PASS ✓" if ok else f"  FAIL sd={sd} cd={cd} od={od}")
    assert ok

    di = dict(
        state=state.reshape(B * V_HEADS * K_DIM, V_DIM).to(device),
        mixed_qkv=mixed_qkv.reshape(B * QKV_DIM).to(device),
        conv_state=conv_state.reshape(B * QKV_DIM, 3).to(device),
        conv_weight=conv_weight.to(device), conv_bias=conv_bias.to(device),
        a_out=a_out.reshape(B * V_HEADS).to(device), b_out=b_out.reshape(B * V_HEADS).to(device),
        z=z.reshape(B * V_HEADS, V_DIM).to(device),
        A_log=A_log.to(device), dt_bias=dt_bias.to(device), norm_weight=norm_weight.to(device),
    )
    for _ in range(3):
        kernel(**di)
    N = 50
    t0 = time.time()
    for _ in range(N):
        kernel(**di)
    print(f"  {mod_name}: {(time.time() - t0) / N * 1000:.3f} ms/call (B={B}, avg {N})")
