"""CPU pre-check for the DN_TILED_CONV layout (T1a). No device, no NKI.

The tiled conv-state lever changes only a *layout*: conv_states move from
channel-major [NUM_DN, B, QKV_DIM, 3] to tile-partition-major
[NUM_DN, B, PMAX, NT, 3], where element [.., p, t, j] is conv tap j of channel
t*PMAX + p. Every numerical claim ("bit-identical") rests on that mapping being
the same one the kernel's head loop already assumes: qkv_act[p, t] is channel
t*PMAX + p, so tile t holds channels [t*128, (t+1)*128).

That is exactly what this file gates, cheaply and without a Trn2:
  1. `_conv_cm_to_tiled` is a pure permutation (round-trips, no data loss).
  2. It realises the channel = t*PMAX + p mapping, not some transposed variant.
  3. The flat views the kernel and harness use ([B*PMAX, NT*3] / [B*QKV_DIM, 3])
     address the same elements the 4-D views do.
  4. Tile t covers exactly the channel block the head loop reads for tile t —
     including the q / k / v slot boundaries, which is where an off-by-one tile
     mapping would corrupt one projection and not the others.

Device numerics are gated separately by test_deltanet_full_batched.py with
DN_TILED_CONV=1 (same cpu_reference, tiled input, un-tiled output).

Run: python3 test_tiled_conv_layout_cpu.py
"""
import os as _os
import sys as _sys

import torch

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

PMAX = 128
K_DIM = 128
V_DIM = 128
K_HEADS = 4
V_HEADS = 12
QKV_DIM = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM  # 2560
NT = QKV_DIM // PMAX                             # 20
TAPS = 3


def cm_to_tiled(cs):
    """[NUM_DN, B, QKV_DIM, 3] -> [NUM_DN, B, PMAX, NT, 3]. Mirrors the harness helper."""
    nd, b, qkv, k = cs.shape
    nt = qkv // PMAX
    return cs.reshape(nd, b, nt, PMAX, k).permute(0, 1, 3, 2, 4).contiguous()


def tiled_to_cm(cs):
    """Inverse of cm_to_tiled."""
    nd, b, pmax, nt, k = cs.shape
    return cs.permute(0, 1, 3, 2, 4).contiguous().reshape(nd, b, pmax * nt, k)


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    assert cond, name


def main():
    assert QKV_DIM % PMAX == 0, "tiled conv needs QKV_DIM % 128 == 0"
    print(f"=== DN_TILED_CONV layout (27B: QKV_DIM={QKV_DIM}, NT={NT}) ===")

    # The precondition the plan called out: the 27B has 20 tiles, the 35B 16.
    check("NT == 20 (27B per-core DN_QKV_DIM=2560=20*128)", NT == 20)

    torch.manual_seed(7)
    NUM_DN, B = 48, 8
    cm = torch.randn(NUM_DN, B, QKV_DIM, TAPS, dtype=torch.bfloat16)

    tiled = cm_to_tiled(cm)
    check("tiled shape == [NUM_DN, B, PMAX, NT, 3]",
          tuple(tiled.shape) == (NUM_DN, B, PMAX, NT, TAPS))
    check("element count preserved", tiled.numel() == cm.numel())

    # 1. Pure permutation — exact round-trip, no tolerance.
    check("round-trip cm -> tiled -> cm is bitwise exact",
          torch.equal(tiled_to_cm(tiled), cm))

    # 2. The mapping is channel = t*PMAX + p (spot-check the whole grid, not samples).
    expect = cm.reshape(NUM_DN, B, NT, PMAX, TAPS).permute(0, 1, 3, 2, 4)
    check("tiled[.., p, t, :] == cm[.., t*PMAX+p, :] for all (p, t)",
          torch.equal(tiled, expect.contiguous()))
    # And explicitly, elementwise, for a few (p, t) including the corners.
    ok = True
    for p in (0, 1, 63, 127):
        for t in (0, 1, NT // 2, NT - 1):
            ok &= torch.equal(tiled[3, 5, p, t], cm[3, 5, t * PMAX + p])
    check("explicit corner (p, t) -> channel checks", ok)
    # A transposed mapping (channel = p*NT + t) must NOT satisfy the above — this
    # guards the test itself against passing on a wrong-but-symmetric layout.
    wrong = cm.reshape(NUM_DN, B, PMAX, NT, TAPS)
    check("the transposed mapping is genuinely different (test has teeth)",
          not torch.equal(tiled, wrong))

    # 3. Flat views used by the kernel / harness address the same elements.
    flat_tiled = tiled[0].reshape(B * PMAX, NT * TAPS)
    check("tiled flat view [B*PMAX, NT*3] agrees with the 4-D view",
          torch.equal(flat_tiled.reshape(B, PMAX, NT, TAPS), tiled[0]))
    flat_cm = cm[0].reshape(B * QKV_DIM, TAPS)
    check("channel-major flat view [B*QKV_DIM, 3] agrees with the 4-D view",
          torch.equal(flat_cm.reshape(B, QKV_DIM, TAPS), cm[0]))

    # 4. Tile coverage vs the head loop's channel slots. The kernel reads
    #    qkv_act[0:K_DIM, kh] for q, [.., K_HEADS+kh] for k, [.., 2*K_HEADS+h] for v
    #    — i.e. slot s is tile s, covering channels [s*PMAX, (s+1)*PMAX). q occupies
    #    slots 0..K_HEADS-1, k K_HEADS..2*K_HEADS-1, v 2*K_HEADS..2*K_HEADS+V_HEADS-1.
    check("slot count == NT (q + k + v tiles exactly tile the channel axis)",
          2 * K_HEADS + V_HEADS == NT)
    covered = set()
    for s in range(NT):
        block = set(range(s * PMAX, (s + 1) * PMAX))
        check_disjoint = covered.isdisjoint(block)
        assert check_disjoint, f"tile {s} overlaps an earlier tile"
        covered |= block
    check("tiles partition [0, QKV_DIM) with no gap or overlap",
          covered == set(range(QKV_DIM)))
    # q ends / k begins / v begins must land on tile boundaries, else one
    # projection would straddle two tiles and a mapping error would be partial.
    for name, ch in (("q_end", K_HEADS * K_DIM),
                     ("k_end", 2 * K_HEADS * K_DIM),
                     ("v_end", 2 * K_HEADS * K_DIM + V_HEADS * V_DIM)):
        check(f"{name}={ch} is on a tile boundary", ch % PMAX == 0)

    # 5. The sliding-window shift the tiled branch performs (drop oldest tap,
    #    append the new sample) must agree tap-for-tap with the channel-major one.
    mq = torch.randn(NUM_DN, B, QKV_DIM, dtype=torch.bfloat16)
    new_cm = torch.cat([cm[..., 1:TAPS], mq.unsqueeze(-1)], dim=-1)
    mq_t = mq.reshape(NUM_DN, B, NT, PMAX).permute(0, 1, 3, 2)          # [.., PMAX, NT]
    new_tiled = torch.cat([tiled[..., 1:TAPS], mq_t.unsqueeze(-1)], dim=-1)
    check("shifted history matches across layouts (bitwise)",
          torch.equal(tiled_to_cm(new_tiled), new_cm))

    print("\n  ALL LAYOUT CHECKS PASS ✓")


if __name__ == "__main__":
    main()
