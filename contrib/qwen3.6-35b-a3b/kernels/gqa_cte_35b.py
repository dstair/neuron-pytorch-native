"""Graph-safe Qwen 35B wrapper for nkilib context-encoding attention."""

import nki
import nki.language as nl

from nkilib.core.attention.attention_cte import attention_cte


_attention_cte_impl = attention_cte.func


@nki.jit
def nki_gqa_cte_prefill(
    query,
    key_active,
    value_active,
    key_prior,
    value_prior,
    prior_used_len,
):
    """Attend a pre-scaled active block over a runtime-sized fixed prior cache."""
    return _attention_cte_impl(
        q=query,
        k=key_active,
        v=value_active,
        scale=1.0,
        causal_mask=True,
        k_prior=key_prior,
        v_prior=value_prior,
        prior_used_len=prior_used_len,
        tp_q=True,
        tp_k=False,
        tp_out=False,
        cache_softmax=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
    )
