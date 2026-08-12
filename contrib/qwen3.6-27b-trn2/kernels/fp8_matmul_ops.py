"""@nki_op registration for the FP8 W8A16 matmul kernel.

Exposes torch.ops.fp8.matmul so static_decode.py can call it as a graph node
under torch.compile(fullgraph=True, backend="neuron"). The kernel itself is
opaque to the compiler — it sees only an AwsNeuronCustomNativeKernel
custom-call, so no F8E4M3FN convert ops show up at the HLO level (which is
the issue that breaks the F.linear-with-cast path on Trn2).
"""
import torch
from torch_neuronx import nki_op
from fp8_matmul import nki_fp8_matmul


@nki_op("fp8::matmul", mutates_args={})
def fp8_matmul(
    x: torch.Tensor,           # [B, K] bf16
    w_fp8_T_i8: torch.Tensor,  # [K, N] int8 (fp8 bytes; reinterpreted in kernel)
    scale: torch.Tensor,       # [N, 1] f32 (per-output-channel dequant scale)
) -> torch.Tensor:
    """Returns [B, N] bf16 = x @ w.T."""
    return nki_fp8_matmul(x, w_fp8_T_i8, scale)
