"""@nki_op registration for the all-in-one DeltaNet kernel."""
import torch
from torch_neuronx import nki_op
from deltanet_full import nki_deltanet_full


@nki_op("deltanet::full", mutates_args={})
def deltanet_full(
    state: torch.Tensor,         # [V_HEADS*K_DIM=1536, V_DIM=128] f32
    mixed_qkv: torch.Tensor,     # [QKV_DIM=2560] bf16
    conv_state: torch.Tensor,    # [QKV_DIM, 3] bf16
    conv_weight: torch.Tensor,   # [QKV_DIM, 4] f32
    conv_bias: torch.Tensor,     # [QKV_DIM] f32
    a_out: torch.Tensor,         # [V_HEADS=12] f32
    b_out: torch.Tensor,         # [V_HEADS] f32
    z: torch.Tensor,             # [V_HEADS, V_DIM] bf16
    A_log: torch.Tensor,         # [V_HEADS] f32
    dt_bias: torch.Tensor,       # [V_HEADS] f32
    norm_weight: torch.Tensor,   # [V_DIM] f32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (new_state, new_conv_state, gated_output[V_HEADS, V_DIM] bf16)."""
    return nki_deltanet_full(
        state, mixed_qkv, conv_state, conv_weight, conv_bias,
        a_out, b_out, z, A_log, dt_bias, norm_weight,
    )
