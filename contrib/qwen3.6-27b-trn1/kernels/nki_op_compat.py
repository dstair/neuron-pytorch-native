"""torch-neuronx **2.9** compatibility shim for the 2.11 ``nki_op`` decorator.

The 27B kernel ``*_ops.py`` files register their NKI kernels with
``from torch_neuronx import nki_op`` / ``@nki_op("ns::name", mutates_args={})``.
``nki_op`` exists in torch-neuronx **2.11** (``nki_hop.py``) but is **absent in
2.9** (which only ships ``nki_jit``). Importing this module installs a drop-in
``torch_neuronx.nki_op`` so the kernel files import unmodified under 2.9's
torch_xla / PJRT / openxla stack.

Why this can't be a verbatim copy of 2.11's ``nki_op``:
    2.11 registers the op body ``fn`` *itself* as the Fake/meta kernel. That
    works there because 2.11's ``nki.jit`` returns fake outputs under
    ``FakeTensorMode``. Under 2.9 the ``@nki.jit`` kernel is fake-unaware and
    actually tries to compile/run on a ``FakeTensor`` during Dynamo shape
    propagation → ``RuntimeError: ... not an XLA tensor``. So the compile path
    (``torch.compile(backend="openxla")``) needs a *real* fake per op that
    returns correctly-shaped/typed empties **without** invoking the kernel.

All 27B op outputs are expressible from their inputs (``empty_like`` of an input
tensor, or a ``[V_HEADS, V_DIM]`` combination), so the fakes below are exact.
The op itself must be registered specifically for XLA. A default composite
implementation is decomposed by AOT functionalization; substituting the fake
there records empty placeholders instead of the NKI call and silently corrupts
compiled execution. The eager / lazy-XLA path never consults these fakes.

Install is a no-op on 2.11 (``torch_neuronx.nki_op`` already present); callers
should guard with ``if not hasattr(torch_neuronx, "nki_op"): import nki_op_compat``.
"""
import torch
import torch_neuronx
from torch.library import custom_op, infer_schema

# --- per-op fake (meta) kernels, keyed by the "{ns}::{name}" op qualname ---
# Each receives the same positional args as the real op (as FakeTensors) and
# returns empties of the exact output shape/dtype. Shapes cross-checked against
# the "Returns ..." docstrings in each kernels/*_ops.py.
_FAKES = {
    # (state, query, key, value, g, beta, m_incl, m_strict, eye)
    # -> (output[V_HEADS*S, V_DIM] f32, new_state[V_HEADS*K_DIM, V_DIM] f32)
    "deltanet::chunked_prefill": lambda state, query, key, value, *r: (
        torch.empty_like(value), torch.empty_like(state)),
    # (state, mixed_qkv, conv_state, conv_weight, conv_bias, a_out, b_out, z, ...)
    # -> (new_state, new_conv_state, gated_output[B*V_HEADS, V_DIM] bf16 == z)
    "deltanet::full_batched": lambda state, mixed_qkv, conv_state, cw, cb, a, b, z, *r: (
        torch.empty_like(state), torch.empty_like(conv_state), torch.empty_like(z)),
    # (state, mixed_qkv, conv_state, conv_weight, conv_bias, a_out, b_out, z, ...)
    # -> (new_state, new_conv_state, gated_output[V_HEADS, V_DIM] bf16 == z)
    "deltanet::full": lambda state, mixed_qkv, conv_state, cw, cb, a, b, z, *r: (
        torch.empty_like(state), torch.empty_like(conv_state), torch.empty_like(z)),
    # (x, state, conv_state, conv_weight, conv_bias, A_log, dt_bias, norm_weight, ...)
    # -> (new_state, new_conv_state, gated_output[V_HEADS, V_DIM] bf16)
    "deltanet::full_fp8": lambda x, state, conv_state, cw, cb, A_log, dt_bias, norm_weight, *r: (
        torch.empty_like(state), torch.empty_like(conv_state),
        torch.empty((A_log.shape[0], norm_weight.shape[0]), dtype=torch.bfloat16,
                    device=state.device)),
    # (x[B,K], w_fp8_T_i8[K,N], scale) -> [B, N] bf16
    "fp8::matmul": lambda x, w, scale, *r: torch.empty(
        (x.shape[0], w.shape[1]), dtype=torch.bfloat16, device=x.device),
    # (query, gate, q_norm, cos, sin, cached_k, cached_v, mask) -> attn_out == query shape f32
    "gqa::tail": lambda query, *r: torch.empty_like(query),
    # (x[H], weight[H]) -> RMSNorm(x), same shape/dtype as x
    "normfuse::rms_norm": lambda x, *r: torch.empty_like(x),
}


def nki_op(name, fn=None, mutates_args={}):
    """2.9 drop-in for torch-neuronx 2.11's ``torch_neuronx.nki_op``.

    Registers ``fn`` as an XLA-specific torch custom op (so
    torch.compile/openxla preserves an opaque NKI call), with a correct
    Fake/Meta kernel from ``_FAKES`` for shape propagation.
    """
    def dec(fn):
        def backend_fn(*args, **kwargs):
            return fn(*args, **kwargs)

        result = custom_op(
            name, backend_fn, mutates_args=mutates_args,
            device_types="xla",
            schema=infer_schema(fn, mutates_args=mutates_args),
        )
        fake = _FAKES.get(name, fn)  # fall back to fn (2.11 behavior; lazy-only)
        result.register_fake(fake)
        return result

    return dec if fn is None else dec(fn)


# Install on import (idempotent; no-op if 2.11 already provides nki_op).
if not hasattr(torch_neuronx, "nki_op"):
    torch_neuronx.nki_op = nki_op
