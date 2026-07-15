# Debug tools

Reusable isolation, capture, and numerical-diagnostic scripts live here. They
are not part of the runtime path and may require a Trainium device, model
weights, or previously captured tensors.

Scripts in this directory report evidence for manual interpretation. Repeatable
checks with explicit pass/fail assertions belong in `kernels/tests/`.

## Layout

- `deltanet/`: DeltaNet compile-isolation, real-input, and capture diagnostics.
