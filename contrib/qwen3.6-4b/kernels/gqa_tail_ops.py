"""PyTorch custom-op registration for the TP=4 Qwen3.5-4B GQA-tail kernel."""

import torch
from torch_neuronx import nki_op

from gqa_tail import nki_gqa_tail_stateful


@nki_op(
    "gqa4b::tail_stateful",
    mutates_args={"cached_k", "cached_v"},
)
def gqa4b_tail_stateful(
    query: torch.Tensor,
    gate: torch.Tensor,
    q_norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cached_k: torch.Tensor,
    cached_v: torch.Tensor,
    mask: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position: torch.Tensor,
    layer_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return nki_gqa_tail_stateful(
        query,
        gate,
        q_norm,
        cos,
        sin,
        cached_k,
        cached_v,
        mask,
        key,
        value,
        position,
        layer_index,
    )
