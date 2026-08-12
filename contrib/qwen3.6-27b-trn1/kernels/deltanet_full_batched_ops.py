"""@nki_op registration for the batched all-in-one DeltaNet kernel.

Single custom-call processes all B sequences (vs per-element loop = B calls).
Weights (conv_weight/conv_bias/A_log/dt_bias/norm_weight) are SHARED across
the batch; the batched tensors carry the batch axis flattened into dim 0.
"""
import os
import torch
from torch_neuronx import nki_op
# Kernel-variant selector (all share the exact same signature):
#   DNBATCHED_V2=1 -> v2: DMA-coalesced rework (conv-weight hoist + copy-activation
#                         removal + transposed-gates). BANKED BEST: +5.8% @ BS=8.
#   default        -> v1: original validated kernel
# NOTE: v3 (nc_stream_shuffle silu_z) and v4 (all PR opts) were RULED OUT (flat /
# slight regress) and moved to legacy/kernels/. See [[project-qwen36-native-baseline]].
if os.environ.get("DNBATCHED_V2", "0") == "1":
    from deltanet_full_batched_v2 import nki_deltanet_full_batched
else:
    from deltanet_full_batched import nki_deltanet_full_batched


@nki_op("deltanet::full_batched", mutates_args={})
def deltanet_full_batched(
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
    """Returns (new_state[B*V_HEADS*K_DIM,V_DIM], new_conv_state[B*QKV_DIM,3],
    gated_output[B*V_HEADS, V_DIM] bf16)."""
    return nki_deltanet_full_batched(
        state, mixed_qkv, conv_state, conv_weight, conv_bias,
        a_out, b_out, z, A_log, dt_bias, norm_weight,
    )
