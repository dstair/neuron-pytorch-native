"""PyTorch custom-op registration for the fused Trn2 FP8 MLP."""

import torch
from torch_neuronx import nki_op

from fp8_fused_mlp import nki_fp8_fused_mlp


@nki_op("qwen4b_fp8::fused_mlp", mutates_args={})
def qwen4b_fp8_fused_mlp(
    hidden: torch.Tensor,
    gate_up_i8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_i8: torch.Tensor,
    down_scales: torch.Tensor,
) -> torch.Tensor:
    return nki_fp8_fused_mlp(
        hidden,
        gate_up_i8,
        gate_up_scales,
        down_i8,
        down_scales,
    )
