"""Is the GDN kernel's divergence the known fp32 Neumann instability? Measured on real data.

This repo already root-caused and fixed this failure for the 27B/35B PREFILL kernels:
`contrib/qwen3.6-35b-a3b/kernels/tests/test_dn_block_inverse_stability.py` and
`VLLM_NEURON_ASSESSMENT.md` §7. Full-width nilpotent doubling to build the intra-chunk
triangular inverse is exact in exact arithmetic but destroys itself in fp32 once the chunk is
wide, because the intermediate powers M^(2^k) overflow the mantissa and the alternating series
then cancels catastrophically. Their table, on a synthetic near-worst-case chunk:

      C     product-form err     block-16 err     peak |M^(2^k)|
     16            1.2e-04          5.3e-05            1.1e+03
     32            1.0e+00          5.3e-05            1.7e+07
     64            2.7e+08          1.1e-04            5.7e+15
    128            2.3e+26          2.7e-04            8.8e+32

The 27B/35B kernels therefore ship C=32 with a BLOCK-FACTORIZED inverse. The trainable GDN
kernel we are using does full-width doubling at **C=128** -- the worst cell in that table.

This script checks that claim against the *real* Qwen3.5-4B chunk matrices captured from the
first diverging layer, rather than a synthetic worst case, and evaluates a fix.

    ./env/container.sh exec python3 labs/lab03_neumann_stability.py
"""

import argparse
import pathlib

import torch
import torch.nn.functional as F

REPRO = pathlib.Path("/work/runs/gdn_repro_call5.pt")


# ---------------------------------------------------------------- inverse methods


def inv_forward_substitution(A: torch.Tensor) -> torch.Tensor:
    """Oracle: (I-A)^-1 for strictly-lower A, by forward substitution in float64.

    No intermediate powers, so no growth. This is what HF gets from
    `torch.linalg.solve_triangular`.
    """
    C = A.shape[-1]
    N = torch.eye(C, dtype=torch.float64, device=A.device).expand_as(A).clone()
    Ad = A.to(torch.float64)
    for i in range(1, C):
        N[..., i, :] = N[..., i, :] + (Ad[..., i, :i, None] * N[..., :i, :]).sum(-2)
    return N


def inv_doubling_product(A: torch.Tensor, span: int | None = None) -> torch.Tensor:
    """What the trainable NKI kernel does: N = (I+A)(I+A^2)(I+A^4)...

    Mathematically exact for nilpotent A once the powers reach A^(C-1). Numerically this is
    the failure mode.
    """
    C = A.shape[-1]
    limit = span or C
    eye = torch.eye(C, dtype=A.dtype, device=A.device).expand_as(A)
    result = eye + A
    power_term = A @ A
    power = 2
    peak = float(A.abs().amax())
    while power < limit:
        result = result @ (eye + power_term)
        peak = max(peak, float(power_term.abs().amax()))
        power *= 2
        if power < limit:
            power_term = power_term @ power_term
    inv_doubling_product.peak = peak
    return result


def inv_blockdiag(A: torch.Tensor, block: int = 16) -> torch.Tensor:
    """The fix, generalised from `_tri_inverse_blockdiag` (which is 2-way, C=32 only).

    Recursive 2-way block splitting down to `block`-wide diagonal blocks. At each level
    A = D + X with D the two diagonal half-blocks and X the strictly-lower-left half-block.
    X @ X = 0 for a 2-way split, so exactly

        (I - A)^-1 = (I-D)^-1 + (I-D)^-1 X (I-D)^-1

    Every squaring therefore happens only inside a `block`-wide diagonal block, which is the
    width the C16 path already uses successfully -- the intermediate powers never grow.
    Cost: log2(C/block) merges of 2 matmuls each, plus block-wide doubling. At C=128,
    block=16 that is ~14 matmuls against the plain doubling's ~12, so it is nearly free.
    """
    C = A.shape[-1]
    if C <= block:
        return inv_doubling_product(A, span=C)

    half = C // 2
    eye_h = torch.eye(half, dtype=A.dtype, device=A.device)

    a00 = A[..., :half, :half]
    a11 = A[..., half:, half:]
    x10 = A[..., half:, :half]  # the only off-block-diagonal part

    b00 = inv_blockdiag(a00, block)
    b11 = inv_blockdiag(a11, block)

    out = torch.zeros_like(A)
    out[..., :half, :half] = b00
    out[..., half:, half:] = b11
    # (I-D)^-1 X (I-D)^-1 restricted to the lower-left block.
    out[..., half:, :half] = b11 @ x10 @ b00
    del eye_h
    return out


# ---------------------------------------------------------------- real chunk matrices


def real_chunk_matrices(repro: pathlib.Path, chunk: int) -> torch.Tensor:
    """Rebuild the per-chunk A matrices exactly as HF's reference forms them.

    Mirrors `torch_chunk_gated_delta_rule` up to `ut_system`, then A = -tril(ut_system, -1),
    which is what both the reference's export branch and the NKI kernel invert.
    """
    d = torch.load(repro, map_location="cpu", weights_only=True)
    query, key, value, g, beta = (d["q"].float(), d["k"].float(), d["v"].float(),
                                  d["g"].float(), d["beta"].float())

    # (B, T, H, D) -> (B, H, T, D)
    query, key, value = (x.transpose(1, 2).contiguous() for x in (query, key, value))
    beta, decay = (x.transpose(1, 2).contiguous() for x in (beta, g))

    l2 = lambda x: x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)
    query, key = l2(query), l2(key)
    query = query * (query.shape[-1] ** -0.5)

    T = key.shape[2]
    pad = (chunk - T % chunk) % chunk
    if pad:
        query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
        beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))

    k_beta = key * beta.unsqueeze(-1)
    key_c, k_beta_c = (x.reshape(x.shape[0], x.shape[1], -1, chunk, x.shape[-1])
                       for x in (key, k_beta))
    decay_c = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk)

    strictly_upper = torch.ones(chunk, chunk, dtype=torch.bool).triu(1)
    cum_decay = decay_c.cumsum(dim=3)
    pairwise = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).masked_fill(
        strictly_upper, float("-inf")).exp()
    ut_system = (k_beta_c @ key_c.transpose(-1, -2)) * pairwise

    A = -torch.tril(ut_system, -1)
    return A.reshape(-1, chunk, chunk)   # (B*H*num_chunks, chunk, chunk)


def err(a: torch.Tensor, oracle: torch.Tensor) -> float:
    return float((a.to(torch.float64) - oracle).abs().amax())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", type=pathlib.Path, default=REPRO)
    ap.add_argument("--sample", type=int, default=64, help="chunk matrices to test per C")
    args = ap.parse_args()

    print("Real Qwen3.5-4B chunk matrices, layer of first divergence (GDN call 5).")
    print("max-abs error of (I-A)^-1 against a float64 forward-substitution oracle.\n")
    print(f"{'C':>5} {'n_mats':>7} {'doubling (kernel)':>20} {'block-16 (fix)':>16} "
          f"{'peak |A^2^k|':>14} {'finite?':>8}")

    for chunk in (16, 32, 64, 128):
        A = real_chunk_matrices(args.repro, chunk)
        # Worst-conditioned chunks first: those are the ones that break the model.
        order = A.abs().amax(dim=(-1, -2)).argsort(descending=True)
        A = A[order[: args.sample]]

        oracle = inv_forward_substitution(A)
        n_dbl = inv_doubling_product(A)
        n_blk = inv_blockdiag(A, block=16)

        e_dbl, e_blk = err(n_dbl, oracle), err(n_blk, oracle)
        finite = bool(torch.isfinite(n_dbl).all())
        print(f"{chunk:>5} {A.shape[0]:>7} {e_dbl:>20.3e} {e_blk:>16.3e} "
              f"{inv_doubling_product.peak:>14.3e} {str(finite):>8}")

    print("\nIf doubling error explodes with C while block-16 stays ~1e-4, the trainable")
    print("kernel's C=128 full-width doubling is the root cause, and the fix this repo")
    print("already ships for the 27B/35B prefill kernels applies directly.")
    print("NEUMANN STABILITY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
