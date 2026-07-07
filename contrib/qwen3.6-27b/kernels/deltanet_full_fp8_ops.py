"""@nki_op registration for the all-in-one DeltaNet + FP8 in_proj kernel."""
import torch
from torch_neuronx import nki_op
from deltanet_full_fp8 import nki_deltanet_full_fp8


@nki_op("deltanet::full_fp8", mutates_args={})
def deltanet_full_fp8(
    x: torch.Tensor,             # [1, HIDDEN] bf16
    state: torch.Tensor,         # [V_HEADS*K_DIM, V_DIM] f32
    conv_state: torch.Tensor,    # [QKV_DIM, 3] bf16
    conv_weight: torch.Tensor,   # [QKV_DIM, 4] f32
    conv_bias: torch.Tensor,     # [QKV_DIM] f32
    A_log: torch.Tensor,         # [V_HEADS] f32
    dt_bias: torch.Tensor,       # [V_HEADS] f32
    norm_weight: torch.Tensor,   # [V_DIM] f32
    qkv_w_T_i8: torch.Tensor,    # [HIDDEN, QKV_DIM] int8 (fp8 bytes)
    qkv_s: torch.Tensor,         # [QKV_DIM, 1] f32
    z_w_T_i8: torch.Tensor,      # [HIDDEN, Z_DIM=1536] int8
    z_s: torch.Tensor,           # [Z_DIM, 1] f32
    a_w_T_i8: torch.Tensor,      # [HIDDEN, V_HEADS] int8
    a_s: torch.Tensor,           # [V_HEADS, 1] f32
    b_w_T_i8: torch.Tensor,      # [HIDDEN, V_HEADS] int8
    b_s: torch.Tensor,           # [V_HEADS, 1] f32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (new_state, new_conv_state, gated_output)."""
    return nki_deltanet_full_fp8(
        x, state, conv_state, conv_weight, conv_bias,
        A_log, dt_bias, norm_weight,
        qkv_w_T_i8, qkv_s, z_w_T_i8, z_s,
        a_w_T_i8, a_s, b_w_T_i8, b_s,
    )
