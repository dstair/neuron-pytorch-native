"""@nki_op registration for the flash GQA causal-prefill kernel.

Wraps nki_gqa_flash_prefill so it can be called as torch.ops.gqa35b.flash_prefill
from static_decode_35b._gqa_prefill (behind env GQA_FLASH_PREFILL=1).
RoPE-applied q/k and v come in torch layout; kernel returns [Q_HEADS,S,HEAD_DIM].
"""
import torch
from torch_neuronx import nki_op
from gqa_flash_prefill_35b import nki_gqa_flash_prefill, nki_gqa_flash_prefill_chunk


@nki_op("gqa35b::flash_prefill", mutates_args={})
def gqa35b_flash_prefill(
    q: torch.Tensor,   # [Q_HEADS, S, HEAD_DIM]  (post q-norm + rope)
    k: torch.Tensor,   # [S, HEAD_DIM]           (post k-norm + rope, single KV head)
    v: torch.Tensor,   # [S, HEAD_DIM]
) -> torch.Tensor:
    """Returns attn output [Q_HEADS, S, HEAD_DIM] (pre output-gate, pre o_proj)."""
    return nki_gqa_flash_prefill(q, k, v)


@nki_op("gqa35b::flash_prefill_chunk", mutates_args={})
def gqa35b_flash_prefill_chunk(
    q: torch.Tensor,       # [Q_HEADS, CHUNK, HEAD_DIM] (post q-norm + rope)
    k: torch.Tensor,       # [KMAX, HEAD_DIM]  full fixed KV buffer (valid rows [0:q_base+CHUNK))
    v: torch.Tensor,       # [KMAX, HEAD_DIM]
    q_base: torch.Tensor,  # [1,1] f32 runtime scalar = global start row of this chunk
) -> torch.Tensor:
    """Bucketed prefill: one reusable NEFF. Returns [Q_HEADS, CHUNK, HEAD_DIM]."""
    return nki_gqa_flash_prefill_chunk(q, k, v, q_base)
