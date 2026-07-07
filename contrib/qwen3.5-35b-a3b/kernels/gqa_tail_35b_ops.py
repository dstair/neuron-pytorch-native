"""@nki_op registration for the 35B-A3B GQA-tail mega-kernel.

Identical signature to the 27B gqa::tail, registered under a distinct op name.
Q_HEADS=4 (TP=4) comes from gqa_tail_35b's module constant (env-overridable).
Folds q RMSNorm + partial-64 RoPE + scaled scores + masked softmax + weighted-V
+ sigmoid output-gate into ONE custom call/layer. k-side norm/rope + KV-cache
write stay in torch; o_proj stays F.linear.
"""
import torch
from torch_neuronx import nki_op
from gqa_tail_35b import nki_gqa_tail


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
