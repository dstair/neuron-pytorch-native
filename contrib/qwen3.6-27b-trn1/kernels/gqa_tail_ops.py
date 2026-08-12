"""@nki_op registration for the GQA-tail mega-kernel (Phase 1 BS>1 throughput).

Collapses the GQA attention-tail glue cluster (q RMSNorm + partial-64 RoPE + scaled
scores + masked softmax + weighted-V + output-gate) into ONE custom call/layer,
removing the ~12 inter-op EVENT_SEMAPHORE barriers/layer that dominate the BS=8
critical path (see project-qwen36-native-baseline Phase 0). k-side norm/rope + the
KV-cache write stay in torch; o_proj stays F.linear. See gqa_tail.py for the math.
"""
import torch
from torch_neuronx import nki_op
from gqa_tail import nki_gqa_tail


@nki_op("gqa::tail", mutates_args={})
def gqa_tail(
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
