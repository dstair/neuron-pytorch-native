#!/usr/bin/env python3
"""Isolated DeltaNet chunked-prefill validation that calls the @nki.jit kernel
DIRECTLY (bypasses the torch_neuronx `nki_op` torch.ops registration, which the
public DLAMI Neuron SDK does not expose). Same math/gates as
test_dn_chunk_device.py.

Env: CHUNK_SIZE (16/32/64), S, G_SCALE, ZERO_ROWS, DN_V_HEADS,
     DN_STABLE_C32 (1 default), DN_PAIRED_BATCH.
Run inside the Neuron venv:  python kernels/tests/test_dn_chunk_direct.py
"""
import os
import sys
import torch
import torch.nn.functional as F
import torch_neuronx  # noqa: F401  (needed to register the privateuseone device)

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "kernels"))
from deltanet_chunked_prefill_35b import nki_deltanet_chunked_prefill_v2  # noqa: E402
from chunked_prefill import neuron_chunk_gated_delta_rule  # noqa: E402

VH = int(os.environ.get("DN_V_HEADS", "8"))
KD = VD = 128
C = int(os.environ.get("CHUNK_SIZE", "16"))


def main():
    dev = "privateuseone:0"
    torch.manual_seed(0)
    S = int(os.environ.get("S", "512"))
    assert S % C == 0
    gscale = float(os.environ.get("G_SCALE", "0.5"))
    zero_frac = float(os.environ.get("ZERO_ROWS", "0.0"))
    q = (torch.randn(S, VH, KD) * 0.3).float()
    k = (torch.randn(S, VH, KD) * 0.3).float()
    v = (torch.randn(S, VH, VD) * 0.3).float()
    if zero_frac > 0:
        m = (torch.rand(S, VH, 1) < zero_frac).float()
        q = q * (1 - m) + q * m * 1e-4
        k = k * (1 - m) + k * m * 1e-4
    g = (-torch.rand(S, VH) * gscale).float()
    beta = torch.rand(S, VH).float()

    qn = F.normalize(q, p=2, dim=-1, eps=1e-6)
    kn = F.normalize(k, p=2, dim=-1, eps=1e-6)
    init0 = torch.zeros(1, VH, KD, VD)
    ref_out, ref_state = neuron_chunk_gated_delta_rule(
        qn.unsqueeze(0), kn.unsqueeze(0), v.unsqueeze(0),
        g=g.unsqueeze(0), beta=beta.unsqueeze(0), chunk_size=C,
        initial_state=init0, output_final_state=True, use_qk_l2norm_in_kernel=False)
    ref_out = ref_out.squeeze(0)
    ref_state = ref_state.squeeze(0)

    # head-major raw q,k (kernel normalizes internally); state [B=1,VH*KD,VD]
    q_hm = q.transpose(0, 1).reshape(VH * S, KD).contiguous()
    k_hm = k.transpose(0, 1).reshape(VH * S, KD).contiguous()
    v_hm = v.transpose(0, 1).reshape(VH * S, VD).contiguous()
    g_hm = g.transpose(0, 1).reshape(VH * S, 1).contiguous()
    b_hm = beta.transpose(0, 1).reshape(VH * S, 1).contiguous()
    state_in = torch.zeros(1, VH * KD, VD).float()
    _idx = torch.arange(C)
    mi = (_idx.view(C, 1) >= _idx.view(1, C)).float()
    ms = (_idx.view(C, 1) > _idx.view(1, C)).float()
    ey = torch.eye(C).float()

    out_hm, new_state = nki_deltanet_chunked_prefill_v2(
        state_in.to(dev), q_hm.to(dev), k_hm.to(dev), v_hm.to(dev),
        g_hm.to(dev), b_hm.to(dev), mi.to(dev), ms.to(dev), ey.to(dev))
    out = out_hm.reshape(VH, S, VD).transpose(0, 1).cpu().float()
    st = new_state.reshape(VH, KD, VD).cpu().float()

    finite = torch.isfinite(out).all().item() and torch.isfinite(st).all().item()
    od = (ref_out - out).abs().max().item()
    sd = (ref_state - st).abs().max().item()
    ocos = F.cosine_similarity(ref_out.reshape(-1), out.reshape(-1), dim=0).item()
    ok = finite and ocos > 0.9999 and od < 1e-2
    print(f"[dn-chunk-direct] VH={VH} S={S} C={C} G_SCALE={gscale} "
          f"ZERO={zero_frac} STABLE_C32={os.environ.get('DN_STABLE_C32','1')} "
          f"PAIRED={os.environ.get('DN_PAIRED_BATCH','0')}: "
          f"finite={finite} out_cos={ocos:.6f} out_maxdiff={od:.4e} "
          f"state_maxdiff={sd:.4e}  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
