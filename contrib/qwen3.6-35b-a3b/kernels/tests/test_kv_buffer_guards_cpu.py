"""Do the in-place-KV guardrails actually FIRE on the forms that lose writes?

CPU-only, no device, no weights, ~1 s. This is the test for the two assertions
that make `static_decode_35b.py` fail loudly instead of silently computing over an
empty KV cache:

  * `_assert_distinct_storage` -- precondition at the hand-off into a traced graph.
  * `_assert_kv_occupancy`     -- postcondition after prefill/decode.

Why this file has to exist: a guardrail that does not fire is worse than none,
because it is read as coverage. The device probes
(`test_gqa_rope_kv_multicall_probe.py`, `test_gqa_tail_stateful_probe.py`) measure
what the PLATFORM does; this measures whether the HARNESS notices. Both mistakes
that shipped in 2026-08 were of the second kind -- checks that could not fail.

The storage check is the one that closes the probes' structural blind spot: the
identical-whole-view form lands 10/10 at probe scale and still lost 9 of 10 writes
in the real 40-layer graph, so no probe can catch a regression into it. Storage
identity can, and does -- see `test_whole_view_repeated` and
`test_contiguous_on_dim0_slice` below, the latter being the trap that looks like a
copy and is not.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "qwen3.6-35b-a3b"))
sys.path.insert(0, ROOT)

import static_decode_35b as M  # noqa: E402

G, B, S, HD = 4, 2, 8, 16


def _base():
    return torch.zeros(G, B, 1, S, HD)


def _expect_raise(fn, needle, label):
    try:
        fn()
    except RuntimeError as e:
        if needle not in str(e):
            return f"{label}: raised, but message lacks {needle!r}: {e}"
        return None
    return f"{label}: DID NOT RAISE -- the guardrail is not covering this form"


def test_separate_tensors_pass():
    """The shipped form: one distinct allocation per mutation."""
    bufs = [torch.zeros(B, 1, S, HD) for _ in range(G)]
    n = M._assert_distinct_storage(bufs, "test")
    assert n == G, n
    return None


def test_clone_of_slices_pass():
    """The shipped form as written in `prefill_bucketed`."""
    base = _base()
    bufs = [base[gi].clone() for gi in range(G)]
    assert M._assert_distinct_storage(bufs, "test") == G
    return None


def test_distinct_slices_raise():
    """The pre-2026-08-05 prefill form: N distinct views of one base."""
    base = _base()
    bufs = [base[gi] for gi in range(G)]
    return _expect_raise(
        lambda: M._assert_distinct_storage(bufs, "test"),
        "SHARE STORAGE", "distinct slices of one base",
    )


def test_whole_view_repeated():
    """The pre-fix DECODE form. No probe can catch a regression into this one --
    it lands all its writes at probe scale -- so the storage check is the only
    mechanism that will."""
    base = _base()
    bufs = [base.reshape(G * B * S, HD) for _ in range(G)]
    return _expect_raise(
        lambda: M._assert_distinct_storage(bufs, "test"),
        "SHARE STORAGE", "repeated identical whole-tensor view",
    )


def test_contiguous_on_dim0_slice():
    """The trap: `.contiguous()` on an already-contiguous dim-0 slice returns the
    VIEW, not a copy, so ten "buffers" still share one storage. Reads as a copy at
    a glance, which is exactly why it needs a test."""
    base = _base()
    bufs = [base[gi].contiguous() for gi in range(G)]
    assert bufs[0].untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    return _expect_raise(
        lambda: M._assert_distinct_storage(bufs, "test"),
        "SHARE STORAGE", ".contiguous() on a dim-0 slice",
    )


def test_non_tensor_raises():
    try:
        M._assert_distinct_storage([torch.zeros(2), None], "test")
    except TypeError:
        return None
    return "non-tensor entry: DID NOT RAISE"


def test_meta_tensors_skip():
    """Inside a traced region there is no storage to compare; the check must not
    false-fire (all meta tensors can report data_ptr 0)."""
    bufs = [torch.zeros(B, S, HD, device="meta") for _ in range(G)]
    M._assert_distinct_storage(bufs, "test")
    return None


def test_occupancy_full_passes():
    bufs = [torch.ones(B, 1, S, HD) for _ in range(G)]
    rows = M._assert_kv_occupancy(bufs, "test")
    assert rows == [B * S] * G, rows
    return None


def test_occupancy_partial_raises():
    """The production failure: some groups written, the rest silently empty."""
    bufs = [torch.ones(B, 1, S, HD) for _ in range(G)]
    bufs[1].zero_()
    bufs[3].zero_()
    return _expect_raise(
        lambda: M._assert_kv_occupancy(bufs, "test"),
        "DROPPED", "partially populated cache",
    )


def test_occupancy_single_tensor_layout():
    """`_state_parts` accepts the non-dynamic single-tensor layout too."""
    M._assert_kv_occupancy(torch.ones(G, B, 1, S, HD), "test")
    return _expect_raise(
        lambda: M._assert_kv_occupancy(torch.zeros(G, B, 1, S, HD), "test"),
        "DROPPED", "all-zero single-tensor cache",
    )


def test_occupancy_row_granularity():
    """A group with one written row passes; a group with none does not. Decode
    appends one row per step, so row granularity is the meaningful unit."""
    bufs = [torch.zeros(B, 1, S, HD) for _ in range(G)]
    for t in bufs:
        t[:, 0, 0] = 1.0                      # exactly one row per batch item
    rows = M._assert_kv_occupancy(bufs, "test")
    assert rows == [B] * G, rows
    return None


def test_occupancy_escape_hatch():
    os.environ["KV_OCCUPANCY_CHECK"] = "0"
    try:
        M._assert_kv_occupancy([torch.zeros(B, 1, S, HD)], "test")
    finally:
        os.environ["KV_OCCUPANCY_CHECK"] = "1"
    return None


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for t in tests:
        try:
            msg = t()
        except AssertionError as e:
            msg = f"{t.__name__}: assertion failed: {e}"
        if msg:
            failures.append(msg)
        else:
            print(f"  ok  {t.__name__}")
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        raise SystemExit(1)
    print(f"PASS: {len(tests)} guardrail tests -- both assertions fire on every "
          f"storage-sharing form and on a partially-written cache")


if __name__ == "__main__":
    main()
