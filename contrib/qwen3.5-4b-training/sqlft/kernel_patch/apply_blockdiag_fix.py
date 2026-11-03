"""Patch the trainable GDN NKI kernel to use a numerically stable chunk inverse.

WHY
---
`deltanet_fused_chunked_fwd_batched` builds the intra-chunk triangular inverse with
full-width Neumann power-doubling at C=128:

    N = (I + A)(I + A^2)(I + A^4) ... (I + A^64)

`A` is strictly lower triangular hence nilpotent, so this is exact in exact arithmetic. In
fp32 it is not. Measured on the REAL Qwen3.5-4B chunk matrices from the first diverging layer
(`labs/lab03_neumann_stability.py`), max-abs error of (I-A)^-1 against a float64
forward-substitution oracle:

    C      full-width doubling        block-16 (this patch)
    16              7.15e-07                     7.15e-07
    32              2.31e-05                     7.82e-07
    64              7.89e-03                     1.16e-06
    128             1.08e+03                     1.29e-06     <-- the kernel's C

This repo already hit, root-caused and fixed the same failure in the 27B/35B PREFILL kernels
(`contrib/qwen3.6-35b-a3b/kernels/deltanet_chunked_prefill_35b.py::_tri_inverse_blockdiag`,
`VLLM_NEURON_ASSESSMENT.md` §7) and consequently ships those at C=32. The trainable kernel
runs C=128, the worst cell in the table.

WHAT
----
Replace the full-width doubling with a recursive 2-way block factorisation down to 16-wide
diagonal blocks. At each level A = D + X, D the two diagonal half-blocks, X the strictly
lower-left half-block. ((I-D)^-1 X)^2 = 0 for a 2-way split, so exactly

    (I - A)^-1 = (I-D)^-1 + (I-D)^-1 X (I-D)^-1

Every squaring then happens only inside a 16-wide block -- the width the C16 path already uses
successfully -- so the intermediate powers never grow. Cost is comparable: 3 squarings for the
16-wide doubling plus 2 matmuls per merge x 3 merges, against 6 squarings + 6 accumulations.

The backward kernel consumes the forward's saved `N_stack` rather than recomputing the
inverse, so fixing the forward fixes all six gradients.

HOW
---
Copies the upstream kernel package to a sibling directory and rewrites it, asserting every
anchor matches exactly once so an upstream change fails loudly instead of silently
mis-patching. Usage:

    python3 -m sqlft.kernel_patch.apply_blockdiag_fix            # patch
    python3 -m sqlft.kernel_patch.apply_blockdiag_fix --verify   # check only
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

UPSTREAM = pathlib.Path("/work/kernels/qwen35-deltanet/build/torch-neuron")
PATCHED = pathlib.Path("/work/kernels/qwen35-deltanet-blockdiag/torch-neuron")

FWD = "nki_kernels/deltanet_fused_chunked_fwd_batched.py"
AUTOGRAD = "gdn_autograd_batched.py"


# --------------------------------------------------------------------- fwd kernel edits

# 1. New numpy mask builders, appended after the existing ones.
ANCHOR_MAKERS = '''def _make_identity():
    """Identity matrix (128x128) as numpy constant."""
    return np.eye(CHUNK_SIZE, dtype=np.float32)
'''

REPLACE_MAKERS = ANCHOR_MAKERS + '''

# --- block-factorized inverse masks (numerical-stability fix) -------------------
# Selector masks for the recursive 2-way split. bd16 picks the 16-wide diagonal
# blocks; x32/x64/x128 pick the strictly lower-left half-block inside each 32/64/128
# block, i.e. the only part of A that is NOT block-diagonal at the previous level.
INV_BLOCK = 16
INV_WIDTHS = (32, 64, 128)


def _block_same(width):
    idx = np.arange(CHUNK_SIZE)
    return (idx[:, None] // width) == (idx[None, :] // width)


def _make_inv_masks():
    """(4, 128, 128) float32: [bd16, x32, x64, x128]."""
    bd16 = _block_same(INV_BLOCK)
    masks = [bd16.astype(np.float32)]
    for width in INV_WIDTHS:
        # same `width` block, but different `width // 2` block => the lower-left coupling
        masks.append((_block_same(width) & ~_block_same(width // 2)).astype(np.float32))
    return np.stack(masks).astype(np.float32)
'''

# 2. Kernel signature: one extra argument carrying the packed masks.
ANCHOR_SIG = """    lower_mask_diag: nl.ndarray,  # (128, 128) float32 -- lower tri with diag
):"""

REPLACE_SIG = """    lower_mask_diag: nl.ndarray,  # (128, 128) float32 -- lower tri with diag
    inv_masks: nl.ndarray,  # (4, 128, 128) float32 -- [bd16, x32, x64, x128] block selectors
):"""

# 3. Load the new masks into SBUF alongside the existing constants.
ANCHOR_LOAD = """    Lmask_d = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Lmask_d, src=lower_mask_diag)
"""

REPLACE_LOAD = ANCHOR_LOAD + """
    # Block selectors for the stable inverse: [bd16, x32, x64, x128].
    BDmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=BDmask, src=inv_masks[0, 0:P_MAX, 0:P_MAX])
    Xmask32 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Xmask32, src=inv_masks[1, 0:P_MAX, 0:P_MAX])
    Xmask64 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Xmask64, src=inv_masks[2, 0:P_MAX, 0:P_MAX])
    Xmask128 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Xmask128, src=inv_masks[3, 0:P_MAX, 0:P_MAX])
"""

# 4. The actual fix: replace the full-width doubling with the block factorisation.
ANCHOR_NEUMANN = """            # ============================================================
            # Neumann power-doubling: N = (I+A)(I+A^2)...(I+A^{64})
            # 6 rounds → resolves rank up to 2^6 = 64 (sufficient for chunk=128)
            # ============================================================
            P_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=P_acc, data1=eye, data2=A_mat, op=nl.add)

            A_pow = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=A_pow, src=A_mat)

            for _round in nl.sequential_range(6):
                # A_pow = A_pow^2: transpose A_pow, then matmul
                Ap_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=Ap_T_psum, data=A_pow)
                Ap_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=Ap_T, src=Ap_T_psum)

                Ap_sq_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=Ap_sq_psum, stationary=Ap_T, moving=A_pow)
                nisa.tensor_copy(dst=A_pow, src=Ap_sq_psum)

                # P_acc = (I + A_pow) @ P_acc: transpose IpA, then matmul
                IpA = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=IpA, data1=eye, data2=A_pow, op=nl.add)

                IpA_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=IpA_T_psum, data=IpA)
                IpA_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=IpA_T, src=IpA_T_psum)

                Pacc_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=Pacc_psum, stationary=IpA_T, moving=P_acc)
                nisa.tensor_copy(dst=P_acc, src=Pacc_psum)
"""

REPLACE_NEUMANN = """            # ============================================================
            # Block-factorized triangular inverse (numerical-stability fix).
            #
            # The previous code did full-width Neumann doubling to A^127. That is
            # exact in exact arithmetic but loses everything in fp32 at C=128:
            # measured error 1.08e+03 vs a float64 oracle on real Qwen3.5-4B chunk
            # matrices, against 1.29e-06 for the factorization below. Same failure
            # this repo already fixed for the 27B/35B prefill kernels, which is why
            # those ship C=32 (see VLLM_NEURON_ASSESSMENT.md Sec 7).
            #
            # A = D + X per level, D the diagonal half-blocks, X the lower-left
            # half-block. ((I-D)^-1 X)^2 = 0, so exactly
            #     (I - A)^-1 = (I-D)^-1 + (I-D)^-1 X (I-D)^-1
            # Recursing 16 -> 32 -> 64 -> 128 keeps every squaring 16-wide.
            # ============================================================
            # Step 1: D = A restricted to the 16-wide diagonal blocks.
            D_mat = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=D_mat, data1=A_mat, data2=BDmask, op=nl.multiply)

            # Step 2: exact inverse of (I - D) by doubling to D^15. D's blocks are
            # 16-nilpotent so D^16 = 0 and 3 squarings suffice; every power stays
            # inside a 16-wide block, which is why the intermediates stay bounded.
            P_acc = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=P_acc, data1=eye, data2=D_mat, op=nl.add)

            A_pow = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=A_pow, src=D_mat)

            for _round in nl.sequential_range(3):
                Ap_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=Ap_T_psum, data=A_pow)
                Ap_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=Ap_T, src=Ap_T_psum)

                Ap_sq_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=Ap_sq_psum, stationary=Ap_T, moving=A_pow)
                nisa.tensor_copy(dst=A_pow, src=Ap_sq_psum)

                IpA = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=IpA, data1=eye, data2=A_pow, op=nl.add)

                IpA_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=IpA_T_psum, data=IpA)
                IpA_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=IpA_T, src=IpA_T_psum)

                Pacc_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=Pacc_psum, stationary=IpA_T, moving=P_acc)
                nisa.tensor_copy(dst=P_acc, src=Pacc_psum)

            # Step 3: merge 16 -> 32 -> 64 -> 128. Unrolled at trace time because
            # each level uses a different selector mask.
            for _Xmask in (Xmask32, Xmask64, Xmask128):
                X_lvl = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=X_lvl, data1=A_mat, data2=_Xmask, op=nl.multiply)

                # XB = X @ P_acc
                X_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=X_T_psum, data=X_lvl)
                X_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=X_T, src=X_T_psum)

                XB_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=XB_psum, stationary=X_T, moving=P_acc)
                XB = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=XB, src=XB_psum)

                # coupling = P_acc @ XB
                B_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=B_T_psum, data=P_acc)
                B_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=B_T, src=B_T_psum)

                cpl_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=cpl_psum, stationary=B_T, moving=XB)
                cpl = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=cpl, src=cpl_psum)

                nisa.tensor_tensor(dst=P_acc, data1=P_acc, data2=cpl, op=nl.add)
"""


# --------------------------------------------------------------------- autograd edits

ANCHOR_AG_IMPORT = """from .nki_kernels.deltanet_fused_chunked_fwd_batched import (
    _make_identity,
    _make_lower_mask,
    _make_lower_mask_diag,
)

_CONST_IDENTITY = _make_identity()
_CONST_LOWER_MASK = _make_lower_mask()
_CONST_LOWER_MASK_DIAG = _make_lower_mask_diag()
"""

REPLACE_AG_IMPORT = """from .nki_kernels.deltanet_fused_chunked_fwd_batched import (
    _make_identity,
    _make_inv_masks,
    _make_lower_mask,
    _make_lower_mask_diag,
)

_CONST_IDENTITY = _make_identity()
_CONST_LOWER_MASK = _make_lower_mask()
_CONST_LOWER_MASK_DIAG = _make_lower_mask_diag()
_CONST_INV_MASKS = _make_inv_masks()
"""

ANCHOR_AG_CONST = """        _get_constants._cache[key] = (
            torch.from_numpy(_CONST_IDENTITY).to(device),
            torch.from_numpy(_CONST_LOWER_MASK).to(device),
            torch.from_numpy(_CONST_LOWER_MASK_DIAG).to(device),
        )"""

REPLACE_AG_CONST = """        _get_constants._cache[key] = (
            torch.from_numpy(_CONST_IDENTITY).to(device),
            torch.from_numpy(_CONST_LOWER_MASK).to(device),
            torch.from_numpy(_CONST_LOWER_MASK_DIAG).to(device),
            torch.from_numpy(_CONST_INV_MASKS).to(device),
        )"""

# Appears TWICE: once in forward, once in backward. Both must unpack the 4-tuple, even
# though the backward kernel's own signature is unchanged (it reuses the forward's N_stack).
ANCHOR_AG_UNPACK = "        iden, lm, lmd = _get_constants(device)"
REPLACE_AG_UNPACK = "        iden, lm, lmd, invm = _get_constants(device)"

# Device call site: forward the new mask tensor.
ANCHOR_AG_CALL = """            output, final_state, N_stack, w_stack = deltanet_fused_chunked_fwd_batched(
                q, k, v, g, beta, initial_state, lm, iden, lmd
            )"""
REPLACE_AG_CALL = """            output, final_state, N_stack, w_stack = deltanet_fused_chunked_fwd_batched(
                q, k, v, g, beta, initial_state, lm, iden, lmd, invm
            )"""

# CPU/simulator call site, used by nki.simulate. Must stay in lockstep with the device path
# or the simulator silently tests a different kernel than the one that runs.
ANCHOR_AG_SIM = """            iden_np = _to_np(iden); lm_np = _to_np(lm); lmd_np = _to_np(lmd)
            out_np, fs_np, N_np, w_np = nki.simulate(deltanet_fused_chunked_fwd_batched)(
                q_np, k_np, v_np, g_np, beta_np, init_np, lm_np, iden_np, lmd_np
            )"""
REPLACE_AG_SIM = """            iden_np = _to_np(iden); lm_np = _to_np(lm); lmd_np = _to_np(lmd)
            invm_np = _to_np(invm)
            out_np, fs_np, N_np, w_np = nki.simulate(deltanet_fused_chunked_fwd_batched)(
                q_np, k_np, v_np, g_np, beta_np, init_np, lm_np, iden_np, lmd_np, invm_np
            )"""


# (name, anchor, replacement, expected_occurrences)
EDITS = {
    FWD: [
        ("mask builders", ANCHOR_MAKERS, REPLACE_MAKERS, 1),
        ("kernel signature", ANCHOR_SIG, REPLACE_SIG, 1),
        ("SBUF mask load", ANCHOR_LOAD, REPLACE_LOAD, 1),
        ("neumann -> blockdiag", ANCHOR_NEUMANN, REPLACE_NEUMANN, 1),
    ],
    AUTOGRAD: [
        ("autograd imports", ANCHOR_AG_IMPORT, REPLACE_AG_IMPORT, 1),
        ("autograd constants", ANCHOR_AG_CONST, REPLACE_AG_CONST, 1),
        ("constants unpack (fwd+bwd)", ANCHOR_AG_UNPACK, REPLACE_AG_UNPACK, 2),
        ("device call site", ANCHOR_AG_CALL, REPLACE_AG_CALL, 1),
        ("simulator call site", ANCHOR_AG_SIM, REPLACE_AG_SIM, 1),
    ],
}


def verify(root: pathlib.Path) -> list[str]:
    """Report anchors whose occurrence count does not match expectations."""
    problems = []
    for rel, edits in EDITS.items():
        path = root / rel
        if not path.exists():
            problems.append(f"{rel}: missing")
            continue
        text = path.read_text()
        for name, anchor, _, expect in edits:
            n = text.count(anchor)
            if n != expect:
                problems.append(f"{rel}: anchor {name!r} found {n} times, expected {expect}")
    return problems


def apply(upstream: pathlib.Path, patched: pathlib.Path) -> int:
    problems = verify(upstream)
    if problems:
        print("upstream does not match the expected source; refusing to patch:")
        for p in problems:
            print(f"  {p}")
        return 1

    if patched.exists():
        shutil.rmtree(patched)
    patched.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(upstream, patched)
    # Stale bytecode from the upstream copy would shadow the patched sources.
    for pycache in patched.rglob("__pycache__"):
        shutil.rmtree(pycache)

    for rel, edits in EDITS.items():
        path = patched / rel
        text = path.read_text()
        for name, anchor, replacement, expect in edits:
            assert text.count(anchor) == expect, f"{rel}: {name}"
            text = text.replace(anchor, replacement)
            print(f"  patched {rel}: {name} ({expect}x)")
        path.write_text(text)

    # Post-conditions: the pieces that must line up or the kernel fails at trace time with a
    # confusing arity error rather than an obvious one.
    fwd = (patched / FWD).read_text()
    ag = (patched / AUTOGRAD).read_text()
    assert "inv_masks: nl.ndarray" in fwd, "signature not extended"
    assert "_make_inv_masks" in fwd and "_make_inv_masks" in ag, "mask builder not wired"
    assert "sequential_range(6)" not in fwd, "full-width doubling still present"
    assert "sequential_range(3)" in fwd, "16-wide doubling not installed"
    assert "BDmask" in fwd and "Xmask128" in fwd, "block selectors not loaded into SBUF"
    # Be specific: "lmd, invm" also occurs in the two `_get_constants` unpacks, so count the
    # call sites themselves rather than the substring.
    assert ag.count("initial_state, lm, iden, lmd, invm") == 1, "device call site not updated"
    assert ag.count("init_np, lm_np, iden_np, lmd_np, invm_np") == 1, "sim call site not updated"
    assert ag.count("iden, lm, lmd, invm = _get_constants(device)") == 2, "unpacks not updated"
    print(f"\npatched kernel written to {patched}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=pathlib.Path, default=UPSTREAM)
    ap.add_argument("--patched", type=pathlib.Path, default=PATCHED)
    ap.add_argument("--verify", action="store_true", help="check anchors only")
    args = ap.parse_args()

    if args.verify:
        problems = verify(args.upstream)
        for p in problems:
            print(p)
        print("ANCHORS OK" if not problems else "ANCHORS MISMATCH")
        return 1 if problems else 0
    return apply(args.upstream, args.patched)


if __name__ == "__main__":
    raise SystemExit(main())
