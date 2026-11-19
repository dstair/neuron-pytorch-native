"""qwen35-moe-fp8-block-coalesced — fused block-scaled FP8 MoE for Qwen3.5-MoE on Trn2.

High-batch decode kernel for the 256-expert / top-8 MoE of Qwen3.5-MoE
(published as Qwen3.6-35B-A3B). FP8 W8A16 on the routed experts is a
**capacity + high-batch-throughput** lever: it halves the expert weight
footprint (~16 -> 8 GB/core), which is what lets high batch sizes fit at all.

Measured (trn2.3xlarge, TP=8/LNC=1, 40 layers, seq=256, beta-5 DLC):
  - FP8 `block_ob_coalesced`: 681.3 tok/s @ BS=128
  - BF16 full-graph:          320.6 tok/s @ BS=32  (BS>=64 OOMs — cannot load)
BF16 is faster per-token at low batch; FP8 wins aggregate throughput by
unlocking a batch size bf16 cannot reach on one device.

## Public API

Raw kernels (advanced / serving integrators):
  from kernels import get_kernel
  k = get_kernel("<org>/qwen35-moe-fp8-block-coalesced")
  out = k.fused_fp8_block_coalesced_ob(hidden, gate_up, down,
                                       gate_up_scales, down_scales, affinities)

Offline weight prep (CPU, once at load): `weight_prep` — the block-scaled FP8
conversion + coalesced packing (see README "Weight prep").

This is a SERVING-PATH kernel, not a drop-in KernelConfig layer: it computes
only the local-rank routed experts (expert parallel), requires a
`torch.compile(fullgraph=True, backend="neuron")` decode graph, one static
compile per batch shape (32/64/128), and the router + shared expert to run in
the surrounding graph. See README for the integration contract.
"""

from .ops import (  # registers torch.ops.moe_w8.*
    fused_fp8_block_coalesced_ob,
    fused_fp8_block_coalesced,
)
from .nki_kernels import (
    nki_moe_fused_w8_fp8_block_coalesced_ob,
    nki_moe_fused_w8_fp8_block_coalesced,
)
from . import weight_prep

__all__ = [
    "fused_fp8_block_coalesced_ob",
    "fused_fp8_block_coalesced",
    "nki_moe_fused_w8_fp8_block_coalesced_ob",
    "nki_moe_fused_w8_fp8_block_coalesced",
    "weight_prep",
]
