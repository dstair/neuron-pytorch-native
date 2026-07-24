"""Shared logical-NeuronCore selection for Qwen 35B NKI wrappers."""

import os


def _read_lnc_degree() -> int:
    raw = os.environ.get(
        "QWEN35_LNC",
        os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"),
    )
    try:
        degree = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"QWEN35_LNC must be 1 or 2, found {raw!r}") from exc
    if degree not in (1, 2):
        raise RuntimeError(f"QWEN35_LNC must be 1 or 2, found {degree}")
    return degree


LNC_DEGREE = _read_lnc_degree()
