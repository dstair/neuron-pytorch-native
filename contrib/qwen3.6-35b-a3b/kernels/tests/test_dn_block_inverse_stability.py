#!/usr/bin/env python3
"""Regression for the stable C32 block-factorized triangular inverse.

Two tests:

* `test_block_inverse_stability` — the shipped C=32 path (`_tri_inverse_blockdiag`
  in `deltanet_chunked_prefill_35b.py`) against a forward-substitution reference.
* `test_doubling_degrades_with_chunk_size` — the same failure as a function of
  chunk width, for BOTH doubling formulations: the product form this repo tried
  first, and the accumulate form used by the draft chunked GDN prefill kernel in
  the vLLM-Neuron port (`vllm_neuron/functional/gated_delta_rule.py` Step 8, at
  `origin/add-qwen36-moe` commit 65ef8b7). The two are the same Neumann series:

      product:    inv = (I + M^4)(I + M^2)(I + M) = sum_{j<8} M^j
      accumulate: inv = I; inv += M@inv; inv += M^2@inv; inv += M^4@inv

  Both are exact in exact arithmetic (M is nilpotent, M^C = 0) and both lose
  everything to cancellation once the intermediate powers overflow the fp32
  mantissa. The point of the parameterization is that the result stays FINITE at
  every width — so a NaN/inf gate does not catch it. See VLLM_NEURON_ASSESSMENT.md
  §7.
"""

import torch
import torch.nn.functional as F

# Sub-block width used by the shipped `_tri_inverse_blockdiag`.
BLOCK = 16


def inverse_doubling(t, span=None):
    size = t.shape[0]
    limit = span or size
    eye = torch.eye(size, dtype=t.dtype)
    result = eye + t
    power_term = t @ t
    power = 2
    while power < limit:
        result = result @ (eye + power_term)
        power *= 2
        if power < limit:
            power_term = power_term @ power_term
    return result


def inverse_block_c32(t):
    size = t.shape[0]
    diagonal = t.clone()
    diagonal[size // 2 :, : size // 2] = 0
    cross = t - diagonal
    block_inverse = inverse_doubling(diagonal, size // 2)
    return block_inverse + block_inverse @ cross @ block_inverse


def inverse_block_c32_partitioned(t):
    half = t.shape[0] // 2
    b00 = inverse_doubling(t[:half, :half], half)
    b11 = inverse_doubling(t[half:, half:], half)
    result = torch.zeros_like(t)
    result[:half, :half] = b00
    result[half:, half:] = b11
    result[half:, :half] = b11 @ t[half:, :half] @ b00
    return result


def inverse_forward_transposed(t):
    size = t.shape[0]
    result_t = (torch.eye(size, dtype=t.dtype) + t).T.clone()
    for row in range(1, size):
        result_t[:row, row] = (
            result_t[:row, :row] @ t[row, :row].unsqueeze(1)
        ).squeeze(1)
    return result_t.T


def inverse_doubling_accumulate(t, n_double=None):
    """Accumulate-form doubling, as used by the vLLM-Neuron draft chunked kernel.

    `inv = I; m = M; repeat n_double: inv = inv + m @ inv; m = m @ m`, with
    `n_double = ceil(log2 C)` exactly as that kernel's host dispatch computes it
    (`n_double = max(1, (C - 1).bit_length())`). Returns `(inv, peak)` where
    `peak` is the largest absolute intermediate-power entry seen — the quantity
    that overflows the fp32 mantissa and destroys the result by cancellation.
    """
    size = t.shape[0]
    if n_double is None:
        n_double = max(1, (size - 1).bit_length())
    eye = torch.eye(size, dtype=t.dtype)
    inverse = eye.clone()
    power_term = t.clone()
    peak = power_term.abs().max().item()
    for _ in range(n_double):
        inverse = inverse + power_term @ inverse
        power_term = power_term @ power_term
        peak = max(peak, power_term.abs().max().item())
    return inverse, peak


def inverse_blocked(t, block=BLOCK):
    """(I - M)^-1 by block forward substitution over `block`-wide diagonal blocks.

    Each diagonal block is inverted by doubling at width `block` (stable there);
    the strictly-lower off-diagonal blocks are filled by block forward
    substitution, X_ij = D_i @ sum_{k=j}^{i-1} M_ik @ X_kj.

    At C=32 with block=16 this reduces exactly to `inverse_block_c32_partitioned`
    (two 16x16 diagonal inverses plus one coupling term), which is what
    `_tri_inverse_blockdiag` ships. Wider C needs the full nested substitution
    below, which the shipped kernel does NOT implement — this generalization
    exists to show the block approach still holds at C=64/128, not to claim the
    kernel already supports those widths.
    """
    size = t.shape[0]
    num_blocks = size // block
    inverse = torch.zeros_like(t)
    diagonal_inverses = []
    for i in range(num_blocks):
        span = slice(i * block, (i + 1) * block)
        block_inverse, _ = inverse_doubling_accumulate(t[span, span])
        diagonal_inverses.append(block_inverse)
        inverse[span, span] = block_inverse
    for j in range(num_blocks):
        col = slice(j * block, (j + 1) * block)
        for i in range(j + 1, num_blocks):
            row = slice(i * block, (i + 1) * block)
            accumulator = torch.zeros((block, block), dtype=t.dtype)
            for k in range(j, i):
                mid = slice(k * block, (k + 1) * block)
                accumulator = accumulator + t[row, mid] @ inverse[mid, col]
            inverse[row, col] = diagonal_inverses[i] @ accumulator
    return inverse


def chunk_matrix(size, key_dim=128, seed=0):
    """A near-worst-case DeltaNet chunk matrix: near-1 decay (gate -0.01) and
    highly correlated keys (0.99 shared direction). This is the regime that
    root-caused to layer 18 on the real model and took full-32 doubling to NaN
    at bs2 (PREFILL_RECIPE.md §6)."""
    torch.manual_seed(seed)
    base = F.normalize(torch.randn(key_dim), dim=0)
    key = F.normalize(0.99 * base + 0.01 * torch.randn(size, key_dim), dim=-1)
    beta = torch.full((size, 1), 0.9)
    gate = torch.full((size, 1), -0.01)
    gate_cumulative = gate.cumsum(0)
    decay = torch.exp(gate_cumulative - gate_cumulative.T).tril()
    return (-((key * beta) @ key.T) * decay).tril(-1)


def inverse_reference(t):
    """(I - M)^-1 by scalar forward substitution — the accuracy oracle."""
    size = t.shape[0]
    forward = t.clone()
    for row in range(1, size):
        forward[row, :row] += forward[row, :row] @ forward[:row, :row]
    return torch.eye(size, dtype=t.dtype) + forward


def test_doubling_degrades_with_chunk_size():
    """Full-width doubling is accurate at C=16 and useless from C=32 up, in fp32,
    while staying finite throughout. Block-16 factorization holds at every width.

    C=64 is the vLLM-Neuron draft kernel's default (`VLLM_GDN_CHUNK_SIZE=64`);
    C=128 is its clamp ceiling.
    """
    for size in (16, 32, 64, 128):
        t = chunk_matrix(size)
        reference = inverse_reference(t)
        accumulate, peak = inverse_doubling_accumulate(t)
        accumulate_error = (accumulate - reference).abs().max().item()
        product_error = (inverse_doubling(t) - reference).abs().max().item()
        blocked_error = (inverse_blocked(t) - reference).abs().max().item()
        print(
            f"C={size:3d} n_double={max(1, (size - 1).bit_length())} "
            f"accumulate_error={accumulate_error:.3e} "
            f"product_error={product_error:.3e} "
            f"blocked_error={blocked_error:.3e} "
            f"peak_power={peak:.3e} finite={torch.isfinite(accumulate).all().item()}"
        )
        # The failure is silent: no NaN/inf to gate on at any width.
        assert torch.isfinite(accumulate).all()
        # Block-16 factorization is accurate at every width.
        assert blocked_error < 1e-3, f"C={size} blocked_error={blocked_error}"
        if size <= 16:
            # C=16 is safe under full-width doubling — this is why paired C16 was
            # the reliable prefill baseline before the stable C32 path landed.
            assert accumulate_error < 1e-3, f"C={size} accumulate_error={accumulate_error}"
        else:
            # C>=32 loses the result entirely, in fp32, in both formulations.
            assert accumulate_error > 0.5, f"C={size} accumulate_error={accumulate_error}"
            assert product_error > 0.5, f"C={size} product_error={product_error}"


def test_block_inverse_stability():
    size, key_dim = 32, 128
    t = chunk_matrix(size, key_dim)
    reference = inverse_reference(t)

    doubling_error = (inverse_doubling(t) - reference).abs().max().item()
    block_error = (inverse_block_c32(t) - reference).abs().max().item()
    partitioned_error = (
        inverse_block_c32_partitioned(t) - reference
    ).abs().max().item()
    forward_transposed_error = (
        inverse_forward_transposed(t) - reference
    ).abs().max().item()
    rhs = torch.randn(size, key_dim)
    direct_rhs = torch.empty_like(rhs)
    block_inverse = inverse_block_c32_partitioned(t)
    direct_rhs[: size // 2] = block_inverse[: size // 2, : size // 2] @ rhs[: size // 2]
    direct_rhs[size // 2 :] = (
        block_inverse[size // 2 :, : size // 2] @ rhs[: size // 2]
        + block_inverse[size // 2 :, size // 2 :] @ rhs[size // 2 :]
    )
    direct_rhs_error = (direct_rhs - reference @ rhs).abs().max().item()
    print(
        f"doubling_error={doubling_error:.6e} "
        f"block_error={block_error:.6e} "
        f"partitioned_error={partitioned_error:.6e} "
        f"forward_transposed_error={forward_transposed_error:.6e} "
        f"direct_rhs_error={direct_rhs_error:.6e}"
    )
    assert doubling_error > 0.5
    assert block_error < 1e-3
    assert partitioned_error < 1e-3
    assert forward_transposed_error < 1e-5
    assert direct_rhs_error < 1e-3


if __name__ == "__main__":
    test_block_inverse_stability()
    test_doubling_degrades_with_chunk_size()
