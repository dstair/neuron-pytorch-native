"""torch.ops registration for the fused block-scaled FP8 MoE kernel.

Registers `torch.ops.moe_w8.fused_fp8_block_coalesced_ob` (and the non-`ob`
variant) so the kernel drops into a `torch.compile(fullgraph=True,
backend="neuron")` decode graph without a graph break. The compiler sees an
opaque custom call, so the E4M3FN weight bytes never surface as an unsupported
HLO `convert`.
"""
import torch
from torch_neuronx import nki_op

from .nki_kernels.moe_fused_w8 import (
    nki_moe_fused_w8_fp8_block_coalesced_ob,
    nki_moe_fused_w8_fp8_block_coalesced,
)


@nki_op("moe_w8::fused_fp8_block_coalesced_ob", mutates_args={})
def fused_fp8_block_coalesced_ob(
    hidden: torch.Tensor,          # [B, H] bf16,  B in {32, 64, 128}
    gate_up: torch.Tensor,         # [E, H, 2, I] int8 (E4M3 bytes), I == 512
    down: torch.Tensor,            # [E, I, H] int8 (E4M3 bytes)
    gate_up_scales: torch.Tensor,  # [E, 128, 2*(H/128)*(I/128)] bf16
    down_scales: torch.Tensor,     # [E, 128, (I/128)*(H/128)] bf16
    affinities: torch.Tensor,      # [B, E] f32  (per-token local-expert routing weights)
) -> torch.Tensor:                 # [B, H] f32  (local-expert routed sum, pre all-reduce)
    return nki_moe_fused_w8_fp8_block_coalesced_ob(
        hidden, gate_up, down, gate_up_scales, down_scales, affinities
    )


@nki_op("moe_w8::fused_fp8_block_coalesced", mutates_args={})
def fused_fp8_block_coalesced(
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_scales: torch.Tensor,
    affinities: torch.Tensor,
) -> torch.Tensor:
    return nki_moe_fused_w8_fp8_block_coalesced(
        hidden, gate_up, down, gate_up_scales, down_scales, affinities
    )
