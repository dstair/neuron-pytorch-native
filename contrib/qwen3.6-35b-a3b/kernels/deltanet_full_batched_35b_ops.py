"""@nki_op registration for the 35B-A3B batched DeltaNet kernel.

Identical signature to the 27B deltanet::full_batched, registered under a
distinct op name so both can coexist. Head counts (V_HEADS=8) come from
the kernel module's constants (env-overridable).

Variant selector (same signature):
  DNBATCHED_V2=1 -> v2: DMA-coalesced rework (hoisted conv weights, direct-DMA
                        conv_in, transposed-SBUF gates) + the tensor_scalar
                        decay/delta micro-opt. BANKED BEST on the 27B (+5.8% BS=8).
  default        -> v1: original validated kernel (also carries the micro-opt).
"""
import os
import torch
from torch_neuronx import nki_op
if os.environ.get("DNBATCHED_V2", "0") == "1":
    from deltanet_full_batched_v2_35b import nki_deltanet_full_batched
else:
    from deltanet_full_batched_35b import nki_deltanet_full_batched


@nki_op("deltanet35b::full_batched", mutates_args={})
def deltanet35b_full_batched(
    state: torch.Tensor,         # [B*V_HEADS*K_DIM, V_DIM] f32
    mixed_qkv: torch.Tensor,     # [B*QKV_DIM] bf16
    conv_state: torch.Tensor,    # [B*QKV_DIM, 3] bf16
    conv_weight: torch.Tensor,   # [QKV_DIM, 4] f32 (shared)
    conv_bias: torch.Tensor,     # [QKV_DIM] f32 (shared)
    a_out: torch.Tensor,         # [B*V_HEADS] f32
    b_out: torch.Tensor,         # [B*V_HEADS] f32
    z: torch.Tensor,             # [B*V_HEADS, V_DIM] bf16
    A_log: torch.Tensor,         # [V_HEADS] f32 (shared)
    dt_bias: torch.Tensor,       # [V_HEADS] f32 (shared)
    norm_weight: torch.Tensor,   # [V_DIM] f32 (shared)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return nki_deltanet_full_batched(
        state, mixed_qkv, conv_state, conv_weight, conv_bias,
        a_out, b_out, z, A_log, dt_bias, norm_weight,
    )
