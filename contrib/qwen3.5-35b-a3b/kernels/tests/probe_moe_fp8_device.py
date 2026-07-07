#!/usr/bin/env python3
"""On-device isolation probe for the FP8 moe_tkg compile failure.

Compiles ONLY moe_tkg in the exact failing config (T=5, is_all_expert=False, FP8-ROW,
E_L=64,H=2048,I=512,K=8) so the inner NKI/neuronx-cc diagnostic surfaces synchronously
(NEURON_LAUNCH_BLOCKING=1) instead of being swallowed by the 40-layer XLA custom-call.

Run inside the DLC (single rank is enough to reproduce the kernel compile):
  NEURON_LAUNCH_BLOCKING=1 PYTHONPATH=/nkilib python probe_moe_fp8_device.py
"""
import os
import torch
import torch_neuronx  # noqa: F401  (registers the neuron device)

from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg as _moe
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

E, H, I, K = 64, 2048, 512, 8
FP8_MAX = 240.0


def quant_row(w):
    """w [E,OUT,IN] -> (e4m3fn [E,OUT,IN], scale [E,OUT]) per-out-channel."""
    absmax = w.abs().amax(-1, keepdim=True).float().clamp_min(1e-12)
    scale = absmax / FP8_MAX
    wq = (w.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return wq, scale.squeeze(-1)


def main():
    T = int(os.environ.get("T", "5"))
    torch.manual_seed(0)
    dev = "cpu" if os.environ.get("CPU") else "privateuseone:0"

    hidden = (torch.randn(T, H) * 0.3).to(torch.bfloat16)
    gate_up = (torch.randn(E, H, 2, I) * 0.02)           # [E,H,2,I]
    down = (torch.randn(E, I, H) * 0.02)                 # [E,I,H]
    aff = torch.softmax(torch.randn(T, E), dim=-1)
    idx = torch.topk(aff, K, dim=-1).indices.to(torch.int32)

    # FP8 ROW quant in the kernel's expected layouts
    gu_q, gu_s = quant_row(gate_up.permute(0, 2, 3, 1).reshape(E, 2 * I, H))   # [E,2I,H]
    dn_q, dn_s = quant_row(down.permute(0, 2, 1).reshape(E, H, I))             # [E,H,I]
    gate_up_q = gu_q.reshape(E, 2, I, H).permute(0, 3, 1, 2).contiguous()      # [E,H,2,I]
    down_q = dn_q.reshape(E, H, I).permute(0, 2, 1).contiguous()              # [E,I,H]
    gscale = gu_s.reshape(E, 2, I).contiguous()
    dscale = dn_s.reshape(E, H).contiguous()

    is_all = (T >= 16)
    args = [hidden, gate_up_q, down_q, aff, idx]
    kw = dict(expert_gate_up_weights_scale=gscale, expert_down_weights_scale=dscale,
              is_all_expert=is_all,
              expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
              activation_fn=ActFnType.SiLU)
    if is_all:
        # dense path wants global affinities [T,E_all]==[T,E here] + rank_id; mask unselected
        kw["rank_id"] = torch.tensor([[0]], dtype=torch.int32)
        kw["mask_unselected_experts"] = True

    args = [a.to(dev) for a in args]
    kw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in kw.items()}
    print(f"[probe] T={T} dev={dev} FP8 selective is_all_expert=False -> compiling moe_tkg ...",
          flush=True)
    out = _moe(*args, **kw)
    out = out["out"] if isinstance(out, dict) else out
    print(f"[probe] OK out shape={tuple(out.shape)} norm={out.float().norm():.3e}", flush=True)


if __name__ == "__main__":
    main()
