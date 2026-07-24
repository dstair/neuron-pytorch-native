#!/usr/bin/env python3
"""Regression for the stable C32 block-factorized triangular inverse."""

import torch
import torch.nn.functional as F


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


def test_block_inverse_stability():
    torch.manual_seed(0)
    size, key_dim = 32, 128
    base = F.normalize(torch.randn(key_dim), dim=0)
    key = F.normalize(0.99 * base + 0.01 * torch.randn(size, key_dim), dim=-1)
    beta = torch.full((size, 1), 0.9)
    gate = torch.full((size, 1), -0.01)
    gate_cumulative = gate.cumsum(0)
    decay = torch.exp(gate_cumulative - gate_cumulative.T).tril()
    t = (-((key * beta) @ key.T) * decay).tril(-1)

    reference = torch.eye(size, dtype=torch.float32)
    forward = t.clone()
    for row in range(1, size):
        forward[row, :row] += forward[row, :row] @ forward[:row, :row]
    reference += forward

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
