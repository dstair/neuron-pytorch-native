"""PyTorch registration for nkilib context-encoding GQA prefill."""

import torch
from torch_neuronx import nki_op, wrap_nki

from gqa_cte_35b import nki_gqa_cte_prefill


_gqa_cte_lnc2 = wrap_nki(nki_gqa_cte_prefill)[2]


@nki_op("gqa35b::cte_prefill", mutates_args={})
def gqa35b_cte_prefill(
    query: torch.Tensor,
    key_active: torch.Tensor,
    value_active: torch.Tensor,
    key_prior: torch.Tensor,
    value_prior: torch.Tensor,
    prior_used_len: torch.Tensor,
) -> torch.Tensor:
    return _gqa_cte_lnc2(
        query,
        key_active,
        value_active,
        key_prior,
        value_prior,
        prior_used_len,
    )
