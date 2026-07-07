"""@nki_op registration for the free-axis RMSNorm kernel."""
import torch
from torch_neuronx import nki_op
from rms_norm_nki import nki_rms_norm


@nki_op("normfuse::rms_norm", mutates_args={})
def rms_norm(
    x: torch.Tensor,       # [H] hidden vector for one token
    weight: torch.Tensor,  # [H] f32 norm weight (residual: 1+weight applied)
) -> torch.Tensor:
    """Returns RMSNorm(x) = x * rsqrt(mean(x^2)+eps) * (1+weight), [H] x.dtype."""
    return nki_rms_norm(x, weight)
