"""@nki_op registration for the FP8 grouped matvec (MoE expert GEMMs)."""
import torch
from torch_neuronx import nki_op
from fp8_group_matvec import nki_fp8_group_matvec


@nki_op("fp8moe::group_matvec", mutates_args={})
def fp8_group_matvec(
    x: torch.Tensor,        # [G, IN]      bf16
    w_i8_T: torch.Tensor,   # [G, IN, OUT] int8 (fp8 e4m3 bytes, pre-transposed)
    scale: torch.Tensor,    # [G, OUT]     f32
) -> torch.Tensor:
    """Returns [G, OUT] f32 = (x[g] @ w[g]) * scale[g], w FP8-dequantized in-kernel."""
    return nki_fp8_group_matvec(x, w_i8_T, scale)
