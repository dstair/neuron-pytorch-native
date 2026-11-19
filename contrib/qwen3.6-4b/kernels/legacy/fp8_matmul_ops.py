"""PyTorch custom-op registration for the generic Trn2 FP8 W8A16 matmul."""

import torch
from torch_neuronx import nki_op

from fp8_matmul import nki_fp8_matmul


@nki_op("qwen4b_fp8::matmul", mutates_args={})
def qwen4b_fp8_matmul(
    x: torch.Tensor,
    weight_fp8_t: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return nki_fp8_matmul(x, weight_fp8_t, scale)
