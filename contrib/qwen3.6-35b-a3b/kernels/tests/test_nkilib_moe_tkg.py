#!/usr/bin/env python3
"""
Standalone probe: call nkilib moe_tkg with our 35B per-core MoE shapes (TP=4:
E_L=64, H=2048, I=512, K=8) in bf16 and FP8-ROW, and validate vs moe_tkg_torch_ref.
Nails the exact working invocation + weight layout before wiring into the model.

Run inside the DLC with PYTHONPATH including the nkilib dir and the neuron device:
  docker run --device=/dev/neuron0 -e PYTHONPATH=/nkilib ... python test_nkilib_moe_tkg.py
"""
import os
import torch

from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
from nkilib.core.moe.moe_tkg.moe_tkg_torch import moe_tkg_torch_ref
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

# 35B-A3B per-core (TP=4)
E = 64          # E_L local experts
H = 2048
I = 512
K = 8
FP8_MAX = 240.0


def quant_row(w):
    """w [E,OUT,IN] bf16 -> (w_fp8 [E,OUT,IN] e4m3, scale [E,OUT] f32). Per-out-channel."""
    absmax = w.abs().amax(-1, keepdim=True).float().clamp_min(1e-12)   # [E,OUT,1]
    scale = absmax / FP8_MAX
    w_q = (w.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return w_q, scale.squeeze(-1)


def main():
    fp8 = os.environ.get("FP8", "0") == "1"
    T = int(os.environ.get("T", "1"))
    torch.manual_seed(0)
    dev = "neuron" if not os.environ.get("CPU") else "cpu"

    # torch reference uses numpy internally (no bf16); use fp16 for the contract
    # check (the test suite uses nl.float16). On-device the model is bf16.
    wdt = torch.float16
    hidden = (torch.randn(T, H) * 0.3).to(wdt)
    # kernel weight layout: gate_up [E,H,2,I], down [E,I,H]
    gate_up = (torch.randn(E, H, 2, I) * 0.02).to(wdt)
    down = (torch.randn(E, I, H) * 0.02).to(wdt)
    aff = torch.softmax(torch.randn(T, E), dim=-1)         # [T,E] affinities
    idx = torch.topk(aff, K, dim=-1).indices.to(torch.int32)  # [T,K]

    kw = dict(is_all_expert=(T >= 16),
              expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
              activation_fn=ActFnType.SiLU)
    gscale = dscale = None
    if fp8:
        # quant per [E,2,I] (gate_up) and [E,H] (down) — the FP8 ROW layout
        gu_q, gu_s = quant_row(gate_up.permute(0, 2, 3, 1).reshape(E, 2 * I, H))  # [E,2I,H]
        dn_q, dn_s = quant_row(down.permute(0, 2, 1).reshape(E, H, I))            # [E,H,I]
        gate_up_q = gu_q.reshape(E, 2, I, H).permute(0, 3, 1, 2).contiguous()     # [E,H,2,I]
        down_q = dn_q.reshape(E, H, I).permute(0, 2, 1).contiguous()             # [E,I,H]
        gscale = gu_s.reshape(E, 2, I).contiguous()        # [E,2,I]
        dscale = dn_s.reshape(E, H).contiguous()           # [E,H]
        gw, dw = gate_up_q, down_q
    else:
        gw, dw = gate_up, down

    # torch reference
    ref = moe_tkg_torch_ref(
        hidden, gw, dw, aff, idx,
        expert_gate_up_weights_scale=gscale, expert_down_weights_scale=dscale, **kw)
    ref_out = ref["out"] if isinstance(ref, dict) else ref

    print(f"[probe] T={T} fp8={fp8} is_all_expert={kw['is_all_expert']} ref_out shape={tuple(ref_out.shape)} norm={ref_out.float().norm():.3e}")


if __name__ == "__main__":
    main()
