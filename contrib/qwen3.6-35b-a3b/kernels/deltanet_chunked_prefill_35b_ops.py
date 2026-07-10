"""@nki_op registration for the chunked DeltaNet *prefill* kernel.

Wraps nki_deltanet_chunked_prefill_v2 (deltanet_chunked_v2.py) so the whole-
sequence chunked gated-delta-rule recurrence is ONE custom call per layer inside
a torch.compile(fullgraph=True) graph — replacing the per-token / per-chunk torch
loop in chunked_prefill.neuron_chunk_gated_delta_rule, whose data-dependent
Woodbury slice (chunked_prefill.py:135 `A[...,i,:i]=`) makes neuronx-cc fail with
"Failed to merge transformation chain" and is why prefill ran eager-only.

Math validated on CPU sim (out/state diff ~4e-7 vs HF-exact oracle) + assembly
glue 2.8e-6; see [[project-qwen36-chunked-prefill]]. The kernel L2-normalizes q,k
and applies the 1/sqrt(K) q-scale INTERNALLY — callers pass RAW projected q,k.

Layout (per TP core, head-major: row h*S+t):
  state:  [V_HEADS*K_DIM, V_DIM]  f32
  q,k:    [V_HEADS*S, K_DIM]      f32 (RAW, un-normed)
  v:      [V_HEADS*S, V_DIM]      f32
  g,beta: [V_HEADS*S, 1]          f32
  m_incl/m_strict/eye: [C, C]     f32 (host constants; no iota in this NKI build)
Returns (output[V_HEADS*S,V_DIM] f32 raw pre-gate, new_state[V_HEADS*K_DIM,V_DIM] f32).
"""
import torch
from torch_neuronx import nki_op
from deltanet_chunked_prefill_35b import nki_deltanet_chunked_prefill_v2


@nki_op("deltanet35b::chunked_prefill", mutates_args={})
def deltanet_chunked_prefill(
    state: torch.Tensor,      # [V_HEADS*K_DIM, V_DIM] f32
    query: torch.Tensor,      # [V_HEADS*S, K_DIM] f32 (raw)
    key: torch.Tensor,        # [V_HEADS*S, K_DIM] f32 (raw)
    value: torch.Tensor,      # [V_HEADS*S, V_DIM] f32
    g: torch.Tensor,          # [V_HEADS*S, 1] f32
    beta: torch.Tensor,       # [V_HEADS*S, 1] f32
    m_incl: torch.Tensor,     # [C, C] f32
    m_strict: torch.Tensor,   # [C, C] f32
    eye: torch.Tensor,        # [C, C] f32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (output[V_HEADS*S, V_DIM] f32 raw, new_state[V_HEADS*K_DIM, V_DIM] f32)."""
    return nki_deltanet_chunked_prefill_v2(
        state, query, key, value, g, beta, m_incl, m_strict, eye,
    )
