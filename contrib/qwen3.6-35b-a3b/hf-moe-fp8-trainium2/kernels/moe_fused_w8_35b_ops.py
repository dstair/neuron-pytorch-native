"""PyTorch custom-op registration for the fused block-scaled FP8 MoE decode kernels.

Both ops drop into a `torch.compile(fullgraph=True, backend="neuron")` decode graph
without a graph break. Compile with ``dynamic=False`` (or `torch._dynamo.mark_static`
the batch dim): NKI kernels reject symbolic shapes, so a second batch size otherwise
fails with `NkiTensor.__init__ does not accept dynamic shape dims`.

* `moe_w8::fused_fp8_block_coalesced` -- default. One scale per 128x128 block; pair
  with `moe_w8` weights prepared with ``fp8_impl="block_pow2_coalesced"`` (lossless
  against the official checkpoint).
* `moe_w8::fused_fp8_block_coalesced_ob` -- opt-in. One scale per 128 output rows,
  accumulated in PSUM and scaled once; ~2x faster at B=128 only (same speed at
  B=32/64). Pair with ``fp8_impl="block_ob_coalesced"``; costs ~4% rel-L2 per MoE layer.
"""
import torch
from torch_neuronx import nki_op

from moe_fused_w8_35b import (
    nki_moe_fused_w8_fp8_block_coalesced,
    nki_moe_fused_w8_fp8_block_coalesced_ob,
)


@nki_op("moe_w8::fused_fp8_block_coalesced", mutates_args={})
def fused_fp8_block_coalesced(
    hidden: torch.Tensor,          # [B, H] bf16,  B in {32, 64, 128}
    gate_up: torch.Tensor,         # [E, H, 2, I] int8 (legacy E4M3 bytes), I == 512
    down: torch.Tensor,            # [E, I, H] int8 (legacy E4M3 bytes)
    gate_up_scales: torch.Tensor,  # [E, 128, 2*(H/128)*(I/128)] bf16
    down_scales: torch.Tensor,     # [E, 128, (I/128)*(H/128)] bf16
    affinities: torch.Tensor,      # [B, E] f32  (per-token local-expert routing weights)
) -> torch.Tensor:                 # [B, H] f32  (local-expert routed sum, pre all-reduce)
    return nki_moe_fused_w8_fp8_block_coalesced(
        hidden, gate_up, down, gate_up_scales, down_scales, affinities
    )


@nki_op("moe_w8::fused_fp8_block_coalesced_ob", mutates_args={})
def fused_fp8_block_coalesced_ob(
    hidden: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_scales: torch.Tensor,
    affinities: torch.Tensor,
) -> torch.Tensor:
    return nki_moe_fused_w8_fp8_block_coalesced_ob(
        hidden, gate_up, down, gate_up_scales, down_scales, affinities
    )
