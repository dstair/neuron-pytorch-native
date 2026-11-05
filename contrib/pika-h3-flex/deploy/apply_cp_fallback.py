#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fix CP>1 compile: trace the all-gather attention fallback, not the NKI ring kernel.

``cp_self_attention``'s capability check (``can_run_kernel``) answers True for dynamo's
FakeTensors (they claim a neuron device), so the ring-attention NKI kernel gets traced —
and its mlir tracer raises "RuntimeError when making fake tensor call" (the kernel body
descends into emit_matmul). The module already documents the all-gather path as the
fake-tensor fallback; this makes the check actually detect tracing.

The compiled graph then uses all-gather K/V + local flash attention for CP>1: Q/FFN
activations stay sequence-sharded (that is what fixes the 362-frame TP=8 OOM), K/V are
gathered (a modest, constant memory add).

Usage:
    python apply_cp_fallback.py /data/omni29/plugin/vllm_omni_neuron/diffusion/attention/cp_attention.py
"""

import sys

OLD = '''def _can_use_wan_ring_attention_kernel(value: torch.Tensor) -> bool:
    """Whether the ring-attention kernel can run. It has no shape limits, only this check."""
    if not can_run_kernel(value):
        return record_kernel_choice("wan_ring_attention", False, device_reason(value))
    return record_kernel_choice("wan_ring_attention", True, "eligible")
'''

NEW = '''def _can_use_wan_ring_attention_kernel(value: torch.Tensor) -> bool:
    """Whether the ring-attention kernel can run. It has no shape limits, only this check."""
    if torch.compiler.is_compiling():
        # The NKI ring kernel does not survive dynamo fake-tensor tracing (the wrapper
        # descends into the kernel body and the mlir tracer raises). Trace the all-gather
        # fallback instead; the compiled graph keeps it at runtime.
        return record_kernel_choice("wan_ring_attention", False, "dynamo fake-tensor tracing")
    if not can_run_kernel(value):
        return record_kernel_choice("wan_ring_attention", False, device_reason(value))
    return record_kernel_choice("wan_ring_attention", True, "eligible")
'''


def apply(src: str) -> str:
    if "dynamo fake-tensor tracing" in src:
        raise SystemExit("cp fallback already applied")
    assert OLD in src, "capability check anchor not found"
    return src.replace(OLD, NEW, 1)


def main():
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    out = apply(src)
    with open(path, "w") as f:
        f.write(out)
    print(f"Applied cp fallback to {path}")


if __name__ == "__main__":
    main()
