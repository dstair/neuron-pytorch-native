"""Make upstream v15's training-forward diagonal-block inverse square only 16-wide blocks.

WHY
---
v8 fixed the training forward upstream by replacing the global Neumann product with row-block
forward substitution, BLOCK=32: each 32x32 diagonal block A_ii is inverted by power-doubling
up to A_ii^31, and the off-diagonal blocks are substituted exactly. On Qwen3.5-4B that is not
enough. On the inputs of GDN layer 19 (`labs/lab02_gdn_in_situ.py --dump 19`), vs HF:

    build            fwd cos     dq        dk        dv        dg        dbeta
    v15 (BLOCK=32)   0.997881    0.998370  0.992111  0.925385  0.986994  0.943209
    v3.0 + blockdiag 1.000047    1.000049  1.000091  1.000058  1.000000  1.000000

The blockdiag patch only ever squares inside 16-wide blocks, and the stability table in
NOTES.md shows doubling losing accuracy between 16 and 32 wide. Simply setting BLOCK=16 does
not compile: a 16-partition access must start at partition 0/32/64/96, and the second 16-row
block starts at 16 (`'nisa.tensor_copy' op Expected 'src' partition start in {0, 32, 64, 96}`).
Upstream records the same wall on SDK 2.31.

WHAT
----
Keep BLOCK=32 rows, but invert each 32x32 A_ii by a 2-way split done with MASKS rather than
partition slices. A_ii = D + X with D the two 16x16 diagonal blocks and X the lower-left
16x16 corner. Powers of a block-diagonal matrix stay block-diagonal, so doubling D (3 rounds,
up to D^15) only ever squares 16-wide blocks. ((I-D)^-1 X)^2 = 0, so exactly

    (I - A_ii)^-1 = (I-D)^-1 + (I-D)^-1 X (I-D)^-1

Cost per diagonal block: one doubling round fewer, plus 2 matmuls and a transpose for the
merge. The backward consumes the forward's saved N_stack, so a more accurate forward fixes the
gradients too.

HOW
---
Copies v15 to a sibling directory and rewrites it, asserting each anchor matches exactly once
so an upstream change fails loudly instead of silently mis-patching:

    python3 -m sqlft.kernel_patch.apply_v15_split16
"""

from __future__ import annotations

import pathlib
import shutil

UPSTREAM = pathlib.Path("/work/kernels/qwen35-deltanet-v15/build/torch-neuron")
PATCHED = pathlib.Path("/work/kernels/qwen35-deltanet-v15-split16/torch-neuron")
FWD = "nki_kernels/deltanet_fused_chunked_fwd_batched.py"

ANCHOR_BLOCK = "    BLOCK = 32\n    N_BLOCKS = P_MAX // BLOCK\n    BLOCK_ROUNDS = 4\n"

ANCHOR_MASKS = "    Lmask, Lmask_d, eye = emit_tri_masks(P_MAX)\n"
REPLACE_MASKS = ANCHOR_MASKS + """
    # split16 patch: masks splitting a BLOCK x BLOCK diagonal block into its two HALF-wide
    # diagonal blocks (BDm) and the lower-left / upper-right HALF x HALF corners (XMm, XMTm).
    # Built from iota + arithmetic like emit_tri_masks: exact 0.0 / 1.0, loop-invariant.
    HALF = BLOCK // 2
    SPLIT_ROUNDS = 3  # log2(HALF) - 1: (I+D)(I+D^2)(I+D^4)(I+D^8) reaches D^15
    sp_col = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=sp_col, pattern=[[1, BLOCK]], offset=0, channel_multiplier=0)
    sp_delta = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=sp_delta, pattern=[[1, BLOCK]], offset=0, channel_multiplier=-1)
    sp_row = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sp_row, data1=sp_col, data2=sp_delta, op=nl.subtract)
    sp_rowhi = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sp_rowhi, data=sp_row, op0=nl.greater_equal,
                       operand0=float(HALF), engine=nisa.vector_engine)
    sp_colhi = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sp_colhi, data=sp_col, op0=nl.greater_equal,
                       operand0=float(HALF), engine=nisa.vector_engine)
    sp_both = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sp_both, data1=sp_rowhi, data2=sp_colhi, op=nl.multiply)
    XMm = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=XMm, data1=sp_rowhi, data2=sp_both, op=nl.subtract)
    XMTm = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=XMTm, data1=sp_colhi, data2=sp_both, op=nl.subtract)
    sp_off = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sp_off, data1=XMm, data2=XMTm, op=nl.add)
    sp_negoff = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sp_negoff, data=sp_off, op0=nl.multiply, operand0=-1.0,
                       engine=nisa.vector_engine)
    BDm = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=BDm, data=sp_negoff, op0=nl.add, operand0=1.0,
                       engine=nisa.vector_engine)
"""

# The whole diagonal-block inverse: from loading A_ii to the end of the doubling loop.
ANCHOR_DIAG_START = "                Aii = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)\n"
ANCHOR_DIAG_END = "                    nisa.tensor_copy(dst=Dinv, src=nxt)\n"
REPLACE_DIAG = """                Aii = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=Aii, src=A_mat[r0 : r0 + BLOCK, r0 : r0 + BLOCK])
                Aii_T = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
                AiiT_psum = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=AiiT_psum, data=Aii)
                nisa.tensor_copy(dst=Aii_T, src=AiiT_psum)

                # split16: A_ii = D + X. Doubling runs on D only (BDm is symmetric, so it
                # masks A_ii^T into D^T too); X enters once, in the exact merge below.
                Apow = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=Apow, data1=Aii, data2=BDm, op=nl.multiply)
                Apow_T = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=Apow_T, data1=Aii_T, data2=BDm, op=nl.multiply)
                X_T = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=X_T, data1=Aii_T, data2=XMTm, op=nl.multiply)
                Dinv = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=Dinv, data1=eye_b, data2=Apow, op=nl.add)

                # Same transpose-free doubling as upstream, on the block-diagonal D.
                for _r in nl.static_range(SPLIT_ROUNDS):
                    sq = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=sq, stationary=Apow_T, moving=Apow)
                    sqT = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=sqT, stationary=Apow, moving=Apow_T)
                    nisa.tensor_copy(dst=Apow, src=sq)
                    nisa.tensor_copy(dst=Apow_T, src=sqT)
                    IpA_T = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=IpA_T, data1=eye_b, data2=Apow_T, op=nl.add)
                    nxt = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=nxt, stationary=IpA_T, moving=Dinv)
                    nisa.tensor_copy(dst=Dinv, src=nxt)

                # Exact merge: (I - A_ii)^-1 = Dinv + Dinv @ X @ Dinv.
                # nc_matmul(stationary=S, moving=M) = S^T @ M.
                XD_psum = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=XD_psum, stationary=X_T, moving=Dinv)
                XD = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=XD, src=XD_psum)
                D16T_psum = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=D16T_psum, data=Dinv)
                D16_T = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=D16_T, src=D16T_psum)
                DXD_psum = nl.ndarray((BLOCK, BLOCK), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=DXD_psum, stationary=D16_T, moving=XD)
                nisa.tensor_tensor(dst=Dinv, data1=Dinv, data2=DXD_psum, op=nl.add)
"""


def patch_source(text: str) -> str:
    for name, anchor in (("block constants", ANCHOR_BLOCK), ("masks", ANCHOR_MASKS),
                         ("diag start", ANCHOR_DIAG_START), ("diag end", ANCHOR_DIAG_END)):
        n = text.count(anchor)
        if n != 1:
            raise SystemExit(f"anchor {name!r} found {n} times, expected 1; refusing to patch")
    text = text.replace(ANCHOR_MASKS, REPLACE_MASKS)
    start = text.index(ANCHOR_DIAG_START)
    end = text.index(ANCHOR_DIAG_END, start) + len(ANCHOR_DIAG_END)
    return text[:start] + REPLACE_DIAG + text[end:]


def main() -> int:
    text = patch_source((UPSTREAM / FWD).read_text())

    if PATCHED.exists():
        shutil.rmtree(PATCHED)
    PATCHED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(UPSTREAM, PATCHED)
    for pycache in PATCHED.rglob("__pycache__"):
        shutil.rmtree(pycache)
    (PATCHED / FWD).write_text(text)

    assert "BLOCK_ROUNDS)" not in text, "upstream 32-wide doubling loop still present"
    assert text.count("nl.static_range(SPLIT_ROUNDS)") == 1, "split doubling not installed"
    assert "stationary=D16_T, moving=XD" in text, "merge not installed"
    print(f"patched kernel written to {PATCHED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
