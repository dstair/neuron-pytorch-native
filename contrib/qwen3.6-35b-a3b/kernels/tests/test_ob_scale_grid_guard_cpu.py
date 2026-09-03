#!/usr/bin/env python3
"""Negative test for the block_ob_coalesced scale-grid guard (Phase 1.2).

The B1 kernel (`nki_moe_fused_w8_fp8_block_coalesced_ob`) reads its scale at
`gate_scale_index = output_block` with the input tile hard-wired to 0, which is only
valid when the scale grid is independent of the input block:

    sum_i (partial_i * s[o]) == s[o] * sum_i partial_i

`requantize_official_fp8_output_block` produces such a grid, and deliberately gives it
the SAME SHAPE as the per-128x128 grid from `requantize_official_fp8_pow2` so packing
and kernel indexing work unchanged. That shape identity is exactly why no assertion
downstream could catch a mix-up: feeding B1 a `block_pow2_coalesced` weight set
compiles, runs, and returns finite, ordinary-looking, WRONG numbers.

This test builds both grids with the REAL producers and checks that
`assert_output_block_scale_grid`:
  1. accepts the output-block grid;
  2. REJECTS the per-128x128 grid;
  3. would not have been caught by a shape check (the two shapes are identical);
  4. rejects a grid that differs in only one position (no averaging/tolerance).

Pure CPU, no device, no checkpoint.
"""

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]

from moe_w8 import (  # noqa: E402
    BLOCK_SIZE,
    assert_output_block_scale_grid,
    encode_legacy_e4m3,
    requantize_official_fp8_output_block,
    requantize_official_fp8_pow2,
)

EXPERTS = 2
INTERMEDIATE = 2 * BLOCK_SIZE      # 2 output blocks
HIDDEN = 3 * BLOCK_SIZE            # 3 input (contraction) blocks


def _official_fp8_weight(rows, cols, seed):
    """A plausible official checkpoint tensor: legacy-E4M3 bytes + BF16 block scales."""
    g = torch.Generator().manual_seed(seed)
    dense = torch.randn(rows, cols, generator=g)
    row_blocks = rows // BLOCK_SIZE
    col_blocks = cols // BLOCK_SIZE
    scales = torch.empty(row_blocks, col_blocks)
    raw = torch.empty(rows, cols, dtype=torch.int8)
    for rb in range(row_blocks):
        for cb in range(col_blocks):
            r0, c0 = rb * BLOCK_SIZE, cb * BLOCK_SIZE
            blk = dense[r0:r0 + BLOCK_SIZE, c0:c0 + BLOCK_SIZE]
            # deliberately different per (rb, cb) so a per-block grid really varies
            s = (blk.abs().amax() / 448.0).clamp_min(1e-12) * (1.0 + 0.05 * cb)
            s16 = s.to(torch.bfloat16)
            scales[rb, cb] = s16
            raw[r0:r0 + BLOCK_SIZE, c0:c0 + BLOCK_SIZE] = encode_legacy_e4m3(
                blk / s16.float()
            )
    return raw, scales.to(torch.bfloat16)


def _grids(kind):
    """(gate_up_grid [E,2,I/128,H/128], down_grid [E,H/128,I/128]) via real producers."""
    requant = {
        "output_block": requantize_official_fp8_output_block,
        "per_block": requantize_official_fp8_pow2,
    }[kind]
    gate_up, down = [], []
    for e in range(EXPERTS):
        proj = []
        for p in range(2):                      # gate_proj, up_proj: [I, H]
            raw, src = _official_fp8_weight(INTERMEDIATE, HIDDEN, seed=100 * e + p)
            _, scales, _ = requant(raw, src)
            proj.append(scales)
        gate_up.append(torch.stack(proj, dim=0))
        raw, src = _official_fp8_weight(HIDDEN, INTERMEDIATE, seed=999 + e)  # down: [H, I]
        _, scales, _ = requant(raw, src)
        down.append(scales)
    return torch.stack(gate_up, 0), torch.stack(down, 0)


def check_accepts_output_block():
    gu, dn = _grids("output_block")
    assert_output_block_scale_grid(gu, dn)            # must not raise
    # and it really is constant along the contraction axis
    assert torch.equal(gu, gu[..., :1].expand_as(gu))
    assert torch.equal(dn, dn[..., :1].expand_as(dn))
    print(f"PASS accepts_output_block  gate_up{tuple(gu.shape)} down{tuple(dn.shape)}")


def check_rejects_per_block():
    gu, dn = _grids("per_block")
    try:
        assert_output_block_scale_grid(gu, dn)
    except ValueError as e:
        msg = str(e)
        assert "input-block-independent" in msg, f"unexpected message: {msg}"
        assert "silently mis-scale" in msg, "message must name the consequence"
        print("PASS rejects_per_block     ValueError raised, names the consequence")
        return
    raise AssertionError(
        "per-128x128 grid was ACCEPTED -- the guard does not work, and this is the "
        "exact configuration that silently returns wrong numbers"
    )


def check_shape_check_would_not_have_caught_it():
    """The whole reason a property check is needed: the shapes are identical."""
    gu_ob, dn_ob = _grids("output_block")
    gu_pb, dn_pb = _grids("per_block")
    assert gu_ob.shape == gu_pb.shape, "expected identical gate/up grid shapes"
    assert dn_ob.shape == dn_pb.shape, "expected identical down grid shapes"
    assert gu_ob.dtype == gu_pb.dtype, "expected identical dtypes too"
    print(f"PASS shape_indistinct      both kinds are {tuple(gu_ob.shape)} "
          f"{gu_ob.dtype} — no shape or dtype check could separate them")


def check_rejects_single_element_difference():
    """No tolerance, no averaging: one wrong scale is a rejection."""
    gu, dn = _grids("output_block")
    gu_bad = gu.clone()
    gu_bad[0, 0, 0, -1] = (gu_bad[0, 0, 0, -1].float() * 1.5).to(gu_bad.dtype)
    try:
        assert_output_block_scale_grid(gu_bad, dn)
    except ValueError:
        print("PASS rejects_one_element   a single differing scale is rejected")
        return
    raise AssertionError("a single differing scale was accepted")


def check_single_input_block_is_allowed():
    """H/128 == 1 makes input-independence vacuous; must not false-positive."""
    gu = torch.rand(1, 2, 2, 1, dtype=torch.bfloat16)
    dn = torch.rand(1, 2, 1, dtype=torch.bfloat16)
    assert_output_block_scale_grid(gu, dn)
    print("PASS single_input_block    a 1-wide contraction axis is trivially constant")


def main():
    check_accepts_output_block()
    check_rejects_per_block()
    check_shape_check_would_not_have_caught_it()
    check_rejects_single_element_difference()
    check_single_input_block_is_allowed()
    print("\nPASS: block_ob_coalesced scale-grid guard — accepts the coarse grid, "
          "rejects the per-block grid that would silently mis-scale, and does so on a "
          "property no shape or dtype check could see")


if __name__ == "__main__":
    main()
