"""Fused block-scaled FP8 MoE NKI kernel for Qwen3.5-MoE on AWS Trainium2.

Exposes the high-batch, all-local-expert decode kernel. The `_ob`
(output-block / "Reduction B1") variant is the fastest: it accumulates the
whole contraction in PSUM and post-scales once per 128-output-row block with an
input-independent scale, which is exact because the weight prep produces one
absmax scale per output-row block spanning the full input width.

The kernel fuses, per local expert: gate/up GEMM -> SwiGLU -> down GEMM ->
affinity scale -> expert-sum, converting official E4M3FN weight bytes to BF16
tile-by-tile inside the kernel (TensorE does not accept E4M3FN as a stationary
dtype). Router and shared expert stay in the surrounding graph.
"""

from .moe_fused_w8 import (
    nki_moe_fused_w8_fp8_block_coalesced_ob,
    nki_moe_fused_w8_fp8_block_coalesced,
)

__all__ = [
    "nki_moe_fused_w8_fp8_block_coalesced_ob",
    "nki_moe_fused_w8_fp8_block_coalesced",
]
