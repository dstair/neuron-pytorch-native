"""@nki_op registration for the 35B-A3B GQA-tail mega-kernel.

Q_HEADS comes from gqa_tail_35b's module constant (env-overridable). The
stateful variant additionally writes the current K/V rows into aliased base
cache arguments inside the same custom call.
"""
import torch
from torch_neuronx import nki_op
from gqa_tail_35b import nki_gqa_tail, nki_gqa_tail_stateful


@nki_op("gqa35b::tail", mutates_args={})
def gqa35b_tail(
    query: torch.Tensor,     # [B*Q_HEADS, HEAD_DIM] f32 (projected, pre-norm)
    gate: torch.Tensor,      # [B*Q_HEADS, HEAD_DIM] f32 (raw)
    q_norm: torch.Tensor,    # [1, HEAD_DIM] f32
    cos: torch.Tensor,       # [1, ROPE_DIM] f32
    sin: torch.Tensor,       # [1, ROPE_DIM] f32
    cached_k: torch.Tensor,  # [B*S, HEAD_DIM] f32 (already KV-written)
    cached_v: torch.Tensor,  # [B*S, HEAD_DIM] f32
    mask: torch.Tensor,      # [1, S] f32
) -> torch.Tensor:
    """Returns attn_out [B*Q_HEADS, HEAD_DIM] f32 = o * sigmoid(gate), pre-o_proj."""
    return nki_gqa_tail(query, gate, q_norm, cos, sin, cached_k, cached_v, mask)


@nki_op(
    "gqa35b::tail_stateful",
    mutates_args={"cached_k", "cached_v"},
)
def gqa35b_tail_stateful(
    query: torch.Tensor,
    gate: torch.Tensor,
    q_norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cached_k: torch.Tensor,  # [G*B*S, HEAD_DIM] bf16, mutated
    cached_v: torch.Tensor,  # [G*B*S, HEAD_DIM] bf16, mutated
    mask: torch.Tensor,
    key: torch.Tensor,       # [B, HEAD_DIM] bf16
    value: torch.Tensor,     # [B, HEAD_DIM] bf16
    position: torch.Tensor,  # [1, 1] int32
    layer_index: int,
) -> torch.Tensor:
    return nki_gqa_tail_stateful(
        query, gate, q_norm, cos, sin, cached_k, cached_v, mask,
        key, value, position, layer_index,
    )
