"""PyTorch custom-op registration for fused FP8 vocabulary argmax."""

import torch
from torch_neuronx import nki_op

from fp8_fused_head import nki_fp8_fused_head_argmax


@nki_op("qwen4b_fp8::fused_head_argmax", mutates_args={})
def qwen4b_fp8_fused_head_argmax(
    hidden: torch.Tensor,
    weight_i8: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    return nki_fp8_fused_head_argmax(hidden, weight_i8, scales)
