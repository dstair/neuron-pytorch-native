#!/usr/bin/env python3
"""On-device validation: ported 35B chunked-prefill NKI kernel (V_HEADS=8) vs the
pure-torch neuron_chunk_gated_delta_rule reference, at the 35B DeltaNet head config.

The kernel L2-normalizes q,k + applies 1/sqrt(K) q-scale internally and is
batch/head-major (row (b*H+h)*S+t). The torch ref takes [B,S,H,D] and (here)
pre-normalized q,k. We feed both the SAME raw q,k,v,g,beta and compare output
and final state.

Run the paired path in DLC:
  BATCH_SIZE=2 DN_PAIRED_BATCH=1 python kernels/tests/test_dn_chunk_device.py
"""
import os
import sys
import torch
import torch.nn.functional as F
import torch_neuronx  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "kernels"))
import deltanet_chunked_prefill_35b_ops  # noqa: E402  registers deltanet35b::chunked_prefill
from chunked_prefill import neuron_chunk_gated_delta_rule  # noqa: E402

VH = int(os.environ.get("DN_V_HEADS", "8"))
KD = VD = 128
C = int(os.environ.get("CHUNK_SIZE", "64"))


def main():
    dev = "privateuseone:0"
    torch.manual_seed(0)
    B = int(os.environ.get("BATCH_SIZE", "1"))
    S = int(os.environ.get("S", "512"))
    assert S % C == 0

    # G_SCALE sweeps the log-decay magnitude: the model's real g = -A_log.exp()*
    # softplus(...) is much more negative than the original test's -rand*0.5. The
    # Woodbury doubling-series is only conditionally stable, so strongly-negative g
    # (fast decay) may be what breaks the kernel on real weights.
    gscale = float(os.environ.get("G_SCALE", "0.5"))
    zero_frac = float(os.environ.get("ZERO_ROWS", "0.0"))  # fraction of near-zero q/k rows
    q = (torch.randn(B, S, VH, KD) * 0.3).float()
    k = (torch.randn(B, S, VH, KD) * 0.3).float()
    v = (torch.randn(B, S, VH, VD) * 0.3).float()
    if zero_frac > 0:
        # emulate conv+SiLU producing near-zero q/k rows — the suspected trigger of
        # the L2-norm eps-semantics divergence (kernel x/sqrt(ss+eps) vs torch x/max(norm,eps)).
        m = (torch.rand(B, S, VH, 1) < zero_frac).float()
        q = q * (1 - m) + q * m * 1e-4
        k = k * (1 - m) + k * m * 1e-4
    g = (-torch.rand(B, S, VH) * gscale).float()    # log-decay <= 0
    beta = torch.rand(B, S, VH).float()

    # ---- torch reference (pre-normalize q,k; kernel does it internally) ----
    qn = F.normalize(q, p=2, dim=-1, eps=1e-6)
    kn = F.normalize(k, p=2, dim=-1, eps=1e-6)
    init0 = torch.zeros(B, VH, KD, VD)
    ref_out, ref_state = neuron_chunk_gated_delta_rule(
        qn, kn, v, g=g, beta=beta, chunk_size=C,
        initial_state=init0, output_final_state=True, use_qk_l2norm_in_kernel=False)

    # ---- NKI kernel (raw q,k head-major; normalizes internally) ----
    q_hm = q.permute(0, 2, 1, 3).reshape(B * VH * S, KD).contiguous()
    k_hm = k.permute(0, 2, 1, 3).reshape(B * VH * S, KD).contiguous()
    v_hm = v.permute(0, 2, 1, 3).reshape(B * VH * S, VD).contiguous()
    g_hm = g.permute(0, 2, 1).reshape(B * VH * S, 1).contiguous()
    b_hm = beta.permute(0, 2, 1).reshape(B * VH * S, 1).contiguous()
    state_in = torch.zeros(B, VH * KD, VD).float()
    _idx = torch.arange(C); mi = (_idx.view(C, 1) >= _idx.view(1, C)).float()
    ms = (_idx.view(C, 1) > _idx.view(1, C)).float(); ey = torch.eye(C).float()
    out_hm, new_state = torch.ops.deltanet35b.chunked_prefill(
        state_in.to(dev), q_hm.to(dev), k_hm.to(dev), v_hm.to(dev),
        g_hm.to(dev), b_hm.to(dev), mi.to(dev), ms.to(dev), ey.to(dev))
    out = out_hm.reshape(B, VH, S, VD).permute(0, 2, 1, 3).cpu().float()
    st = new_state.reshape(B, VH, KD, VD).cpu().float()

    od = (ref_out - out).abs().max().item()
    sd = (ref_state - st).abs().max().item()
    ocos = F.cosine_similarity(ref_out.reshape(-1), out.reshape(-1), dim=0).item()
    ok = ocos > 0.9999 and od < 1e-2
    print(f"[dn-chunk] B={B} VH={VH} S={S} C={C} G_SCALE={gscale}: out_cos={ocos:.6f} "
          f"out_maxdiff={od:.4e} state_maxdiff={sd:.4e}  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
