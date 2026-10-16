#!/usr/bin/env python3
"""Row-coverage proof for the chunked DeltaNet prefill output buffer.

`nki_deltanet_chunked_prefill_v2` allocates `out` as [total_rows, V_DIM] in
nl.shared_hbm and writes it in exactly one pattern:

    S          = total_rows // num_streams          # FLOOR
    num_chunks = S // C                             # FLOOR
    for stream in range(num_streams):
        row0 = stream * S
        for c in range(num_chunks):
            out[row0 + c*C : row0 + c*C + C] = ...

Both divisions floor, so a remainder in either leaves rows of `out` never written.
shared_hbm is not zeroed, so those rows are returned as whatever was already in HBM:
finite, ordinary in magnitude, and fed straight into the residual stream. No
downstream check catches it -- finiteness passes and the values look plausible.

This test does NOT import the kernel (that needs neuronxcc/nki). It models the
indexing arithmetic exactly and proves:
  1. which rows go unwritten, for each kind of remainder;
  2. that `total_rows % num_streams == 0 and S % C == 0` is exactly the condition for
     full coverage -- i.e. the two asserts added to the kernel are necessary AND
     sufficient, neither too weak nor too strong;
  3. that the shipping configurations are covered (so the bug is latent, not active).
"""


def written_rows(total_rows, num_streams, C):
    """Exactly the set of `out` rows the kernel writes, per the pattern above."""
    S = total_rows // num_streams
    num_chunks = S // C
    rows = set()
    for stream in range(num_streams):
        row0 = stream * S
        for c in range(num_chunks):
            base = row0 + c * C
            rows.update(range(base, base + C))
    return rows


def unwritten(total_rows, num_streams, C):
    return sorted(set(range(total_rows)) - written_rows(total_rows, num_streams, C))


def guard_passes(total_rows, num_streams, C):
    """The condition the kernel now asserts."""
    if total_rows % num_streams:
        return False
    return (total_rows // num_streams) % C == 0


def check_full_coverage_when_divisible():
    # batch 6 * V_HEADS 4 = 24 streams, S = 1024, C = 32 -> exact
    for num_streams, S, C in ((24, 1024, 32), (24, 1024, 16), (8, 512, 32), (4, 64, 16)):
        total = num_streams * S
        miss = unwritten(total, num_streams, C)
        assert not miss, f"expected full coverage for {(total, num_streams, C)}"
        assert guard_passes(total, num_streams, C)
    print("PASS full_coverage        divisible shapes write every row")


def check_tail_unwritten_when_S_not_multiple_of_C():
    # S = 1000 is not a multiple of C = 32 -> 1000 % 32 = 8 tail rows PER STREAM
    num_streams, S, C = 4, 1000, 32
    total = num_streams * S
    miss = unwritten(total, num_streams, C)
    assert len(miss) == num_streams * (S % C), (
        f"expected {num_streams * (S % C)} unwritten rows, got {len(miss)}"
    )
    # and they are the tail of each stream, not a contiguous block at the end
    per_stream_tail = [s * S + (S // C) * C for s in range(num_streams)]
    for t in per_stream_tail:
        assert t in miss, f"row {t} should be unwritten"
    assert not guard_passes(total, num_streams, C), "guard must reject this shape"
    print(f"PASS tail_per_stream      S%C!=0 leaves {len(miss)} rows unwritten "
          f"({S % C} per stream, interior to the buffer)")


def check_tail_unwritten_when_rows_not_multiple_of_streams():
    # total_rows not divisible by num_streams -> trailing rows unwritten
    num_streams, C = 4, 32
    total = 4 * 1024 + 5          # 5 stray rows
    miss = unwritten(total, num_streams, C)
    assert len(miss) >= 5, f"expected at least the 5 stray rows, got {len(miss)}"
    assert total - 1 in miss, "the very last row must be unwritten"
    assert not guard_passes(total, num_streams, C), "guard must reject this shape"
    print(f"PASS trailing_rows        rows%streams!=0 leaves {len(miss)} rows unwritten")


def check_guard_is_necessary_and_sufficient():
    """Sweep a grid: guard_passes(...) must be TRUE exactly when coverage is full.

    This is the part that makes the two asserts trustworthy -- a guard that is merely
    sufficient would reject working shapes, and one that is merely necessary would let
    a silently-wrong shape through.
    """
    checked = 0
    for num_streams in (1, 2, 3, 4, 8, 24):
        for C in (16, 32):
            for S in range(C - 2, 3 * C + 3):
                for extra in (0, 1, C - 1):
                    total = num_streams * S + extra
                    if total <= 0:
                        continue
                    full = not unwritten(total, num_streams, C)
                    ok = guard_passes(total, num_streams, C)
                    assert full == ok, (
                        f"guard disagrees with coverage at total_rows={total} "
                        f"num_streams={num_streams} C={C}: full={full} guard={ok}"
                    )
                    checked += 1
    print(f"PASS guard_exact          necessary AND sufficient over {checked} shapes")


def check_shipping_configs_are_safe():
    """The bug is LATENT: every shipping config divides cleanly."""
    # BS x V_HEADS streams, bucket-chunk 1024, C in {16,32}
    for bs in (1, 2, 3, 4, 6, 8, 16):
        for v_heads in (4, 8):
            for C in (16, 32):
                num_streams = bs * v_heads
                total = num_streams * 1024
                assert guard_passes(total, num_streams, C), (
                    f"shipping config BS={bs} V_HEADS={v_heads} C={C} should pass"
                )
    print("PASS shipping_safe        bucket 1024 x C in {16,32} always divides")


def main():
    check_full_coverage_when_divisible()
    check_tail_unwritten_when_S_not_multiple_of_C()
    check_tail_unwritten_when_rows_not_multiple_of_streams()
    check_guard_is_necessary_and_sufficient()
    check_shipping_configs_are_safe()
    print("\nPASS: chunked-prefill row coverage — the floor divisions do leave rows "
          "of an unzeroed shared_hbm buffer unwritten, and the asserted condition is "
          "exactly the condition for full coverage")


if __name__ == "__main__":
    main()
