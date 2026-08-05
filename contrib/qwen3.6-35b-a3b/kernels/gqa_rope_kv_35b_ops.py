"""PyTorch registration for dynamic-offset GQA RoPE and KV-cache writes."""

import torch
from torch_neuronx import nki_op

from gqa_rope_kv_35b import nki_gqa_rope_kv_dynamic


@nki_op("gqa35b::rope_kv_dynamic", mutates_args={"kv_key", "kv_value"})
def gqa35b_rope_kv_dynamic(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    kv_key: torch.Tensor,
    kv_value: torch.Tensor,
    q_base: torch.Tensor,
    group_index: int,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (query_out, key_out). kv_key/kv_value are the whole flattened
    KV cache, mutated in place and deliberately NOT returned -- returning them
    costs a full-cache materialization at the graph boundary. Read them back
    through the caller's own view of the buffer instead."""
    return nki_gqa_rope_kv_dynamic(
        query,
        key,
        value,
        rope_cos,
        rope_sin,
        kv_key,
        kv_value,
        q_base,
        group_index=group_index,
        num_groups=num_groups,
    )
