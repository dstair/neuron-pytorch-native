"""Standalone correctness + benchmark for nki_deltanet_full_fp8."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # kernels/ for bare-name imports
import math
import time
import torch
import torch.nn.functional as F

from deltanet_full_fp8 import nki_deltanet_full_fp8

HIDDEN = 5120
K_DIM = V_DIM = 128
K_HEADS = 4
V_HEADS = 12
HEAD_GROUP = V_HEADS // K_HEADS
QKV_DIM = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM  # 2560
Z_DIM = V_HEADS * V_DIM  # 1536
RMS_EPS = 1e-6
FP8_MAX = 240.0


def quantize(w_bf):
    """[N, K] bf16 -> ([K, N] int8 fp8 bytes, [N, 1] f32 scale)."""
    absmax = w_bf.abs().amax(dim=-1).float().clamp_min(1e-12)
    scale = absmax / FP8_MAX
    inv = (1.0 / scale).to(w_bf.dtype).unsqueeze(-1)
    w_q = (w_bf * inv).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return w_q.t().contiguous().view(torch.int8), scale.unsqueeze(-1).contiguous()


def dequant(w_T_i8, scale):
    """Inverse: [K, N] int8 + [N, 1] f32 -> [N, K] bf16."""
    w_fp8 = w_T_i8.view(torch.float8_e4m3fn)
    w = w_fp8.t().contiguous().to(torch.bfloat16) * scale.to(torch.bfloat16)
    return w


def cpu_reference(
    x, state, conv_state, conv_weight, conv_bias, A_log, dt_bias, norm_weight,
    qkv_w, z_w, a_w, b_w,
):
    """End-to-end reference using bf16 weights (post-dequant)."""
    # 1) GEMMs
    mixed_qkv = F.linear(x, qkv_w).squeeze(0)        # [QKV_DIM]
    z = F.linear(x, z_w).squeeze(0).reshape(V_HEADS, V_DIM)  # [12, 128]
    a_out = F.linear(x, a_w).squeeze(0).float()      # [V_HEADS]
    b_out = F.linear(x, b_w).squeeze(0).float()      # [V_HEADS]

    # 2) Conv state update
    conv_in = torch.cat(
        [conv_state.float(), mixed_qkv.float().unsqueeze(-1)], dim=-1,
    )
    new_cs = conv_in[:, 1:].to(conv_state.dtype)
    a = F.silu((conv_in * conv_weight.float()).sum(-1) + conv_bias.float())

    # 3) Split + norm
    q = a[: K_HEADS * K_DIM].reshape(K_HEADS, K_DIM)
    k = a[K_HEADS * K_DIM : 2 * K_HEADS * K_DIM].reshape(K_HEADS, K_DIM)
    v = a[2 * K_HEADS * K_DIM :].reshape(V_HEADS, V_DIM)
    q = q / (q.pow(2).sum(-1, keepdim=True) + RMS_EPS).sqrt() * (1 / math.sqrt(K_DIM))
    k = k / (k.pow(2).sum(-1, keepdim=True) + RMS_EPS).sqrt()
    q = q.repeat_interleave(HEAD_GROUP, dim=0)  # [12, 128]
    k = k.repeat_interleave(HEAD_GROUP, dim=0)

    # 4) Gates
    g = -torch.exp(A_log.float()) * F.softplus(a_out + dt_bias.float())
    beta = torch.sigmoid(b_out)

    # 5) Recurrence
    s = state.float().reshape(V_HEADS, K_DIM, V_DIM).clone()
    out_f = torch.empty(V_HEADS, V_DIM, dtype=torch.float32)
    for h in range(V_HEADS):
        s[h] = s[h] * torch.exp(g[h])
        kv = (s[h].T @ k[h]).reshape(V_DIM)
        delta = (v[h] - kv) * beta[h]
        s[h] = s[h] + torch.outer(k[h], delta)
        out_f[h] = s[h].T @ q[h]
    new_state = s.reshape(V_HEADS * K_DIM, V_DIM)

    # 6) RMSNormGated
    var = out_f.pow(2).mean(-1, keepdim=True)
    norm = out_f * torch.rsqrt(var + RMS_EPS)
    gated = norm * norm_weight.float() * F.silu(z.float())
    return new_state, new_cs, gated.to(torch.bfloat16)


if __name__ == "__main__":
    torch.manual_seed(123)
    device = torch.device("neuron")

    # Generate weights & quantize
    qkv_w = torch.randn(QKV_DIM, HIDDEN, dtype=torch.bfloat16) * 0.05
    z_w   = torch.randn(Z_DIM,   HIDDEN, dtype=torch.bfloat16) * 0.05
    a_w   = torch.randn(V_HEADS, HIDDEN, dtype=torch.bfloat16) * 0.05
    b_w   = torch.randn(V_HEADS, HIDDEN, dtype=torch.bfloat16) * 0.05
    qkv_T_i8, qkv_s = quantize(qkv_w)
    z_T_i8,   z_s   = quantize(z_w)
    a_T_i8,   a_s   = quantize(a_w)
    b_T_i8,   b_s   = quantize(b_w)

    # CPU reference uses dequantized weights so we compare apples-to-apples.
    qkv_w_dq = dequant(qkv_T_i8, qkv_s)
    z_w_dq   = dequant(z_T_i8,   z_s)
    a_w_dq   = dequant(a_T_i8,   a_s)
    b_w_dq   = dequant(b_T_i8,   b_s)

    # Other inputs
    x = torch.randn(1, HIDDEN, dtype=torch.bfloat16) * 0.5
    state = torch.randn(V_HEADS * K_DIM, V_DIM, dtype=torch.float32) * 0.1
    conv_state = torch.randn(QKV_DIM, 3, dtype=torch.bfloat16)
    conv_weight = torch.randn(QKV_DIM, 4, dtype=torch.float32) * 0.5
    conv_bias = torch.randn(QKV_DIM, dtype=torch.float32) * 0.1
    A_log = torch.randn(V_HEADS, dtype=torch.float32) * 0.5
    dt_bias = torch.randn(V_HEADS, dtype=torch.float32)
    norm_weight = torch.randn(V_DIM, dtype=torch.float32) * 0.1 + 1.0

    print("=== nki_deltanet_full_fp8 ===")
    rs, rcs, ro = cpu_reference(
        x, state, conv_state, conv_weight, conv_bias, A_log, dt_bias, norm_weight,
        qkv_w_dq, z_w_dq, a_w_dq, b_w_dq,
    )

    args = dict(
        x=x.to(device),
        state=state.to(device),
        conv_state=conv_state.to(device),
        conv_weight=conv_weight.to(device),
        conv_bias=conv_bias.to(device),
        A_log=A_log.to(device),
        dt_bias=dt_bias.to(device),
        norm_weight=norm_weight.to(device),
        qkv_w_T_i8=qkv_T_i8.to(device),
        qkv_s=qkv_s.to(device),
        z_w_T_i8=z_T_i8.to(device),
        z_s=z_s.to(device),
        a_w_T_i8=a_T_i8.to(device),
        a_s=a_s.to(device),
        b_w_T_i8=b_T_i8.to(device),
        b_s=b_s.to(device),
    )

    ns, ncs, no = nki_deltanet_full_fp8(**args)
    ns, ncs, no = (t.cpu() for t in (ns, ncs, no))

    sd = (ns.float() - rs.float()).abs().max().item()
    cd = (ncs.float() - rcs.float()).abs().max().item()
    od = (no.float() - ro.float()).abs().max().item()
    print(f"  state      max_diff = {sd:.6f}")
    print(f"  conv_state max_diff = {cd:.6f}")
    print(f"  output     max_diff = {od:.6f}")
    # Looser tolerance: includes fp8 quant error + bf16 matmul error
    # accumulating across 4 GEMMs + recurrence + RMSNormGated.
    assert sd < 0.5 and cd < 0.5 and od < 0.5, f"FAIL sd={sd} cd={cd} od={od}"
    print("  PASS ✓")

    print("\n=== Benchmark ===")
    for _ in range(3):
        nki_deltanet_full_fp8(**args)
    N = 100
    t0 = time.time()
    for _ in range(N):
        nki_deltanet_full_fp8(**args)
    print(f"  deltanet_full_fp8: {(time.time() - t0) / N * 1000:.3f} ms/call")
