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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (query_out, key_out, kv_key, kv_value).

    kv_key/kv_value are the whole flattened KV cache, mutated in place AND
    returned. The return is not a copy -- it is what makes NKI emit
    `input_output_aliases`, which is the only thing that gives the in-place write
    a representation in the emitted graph; `mutates_args` below reaches only the
    PyTorch-level schema. Read the cache back through these returns, not through
    the caller's own view, so the read is ordered against the write. Measured in
    kernels/tests/probe_kv_alias_f4.py."""
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
