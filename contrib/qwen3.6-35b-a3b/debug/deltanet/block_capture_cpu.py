#!/usr/bin/env python3
"""Diagnose the C32 block-factorized DeltaNet solve on an immutable capture."""

import argparse
import math

import torch
import torch.nn.functional as F


def forward_inverse(strict_lower):
    result = strict_lower.clone()
    for row in range(1, strict_lower.shape[0]):
        result[row, :row] += result[row, :row] @ result[:row, :row]
    return result + torch.eye(
        strict_lower.shape[0], dtype=strict_lower.dtype
    )


def forward_solve(strict_lower, rhs):
    result = rhs.clone()
    for row in range(1, strict_lower.shape[0]):
        result[row] += strict_lower[row, :row] @ result[:row]
    return result


def forward_inverse_transposed(strict_lower):
    size = strict_lower.shape[0]
    result_t = (
        torch.eye(size, dtype=strict_lower.dtype) + strict_lower
    ).T.clone()
    for row in range(1, size):
        result_t[:row, row] = (
            result_t[:row, :row]
            @ strict_lower[row, :row].unsqueeze(1)
        ).squeeze(1)
    return result_t.T


def block_inverse(strict_lower):
    half = strict_lower.shape[0] // 2
    b00 = forward_inverse(strict_lower[:half, :half])
    b11 = forward_inverse(strict_lower[half:, half:])
    cross = b11 @ strict_lower[half:, :half] @ b00
    return b00, b11, cross


def apply_block_inverse(b00, b11, cross, rhs):
    half = rhs.shape[0] // 2
    result = torch.empty_like(rhs)
    result[:half] = b00 @ rhs[:half]
    result[half:] = b11 @ rhs[half:] + cross @ rhs[:half]
    return result


def diagnostic_terms(state, query, key, value, gate, beta, m_incl, m_strict):
    chunk, key_dim = query.shape
    query = F.normalize(query.float(), dim=-1, eps=1e-6) / math.sqrt(key_dim)
    key = F.normalize(key.float(), dim=-1, eps=1e-6)
    gate_cumulative = m_incl @ gate.float()
    decay = torch.exp(
        (gate_cumulative - gate_cumulative.T) * m_incl
    ) * m_incl
    strict = (-(key * beta.float()) @ key.T * decay) * m_strict
    value_beta = value.float() * beta.float()
    key_rhs = key * beta.float() * torch.exp(gate_cumulative)
    intra = ((query @ key.T) * decay) * m_incl
    direct_rhs = value_beta - key_rhs @ state.float()
    return strict, value_beta, key_rhs, direct_rhs, intra, query, gate_cumulative


def diagnose_capture(capture, seq):
    key_dim = capture["query"].shape[1]
    heads = capture["state"].shape[0] // key_dim
    full_seq = capture["query"].shape[0] // heads
    worst = {
        name: (0.0, -1, 0.0, 0.0)
        for name in (
            "block_separate",
            "block_direct_rhs",
            "forward_separate",
            "forward_direct_rhs",
            "forward_transposed_separate",
        )
    }

    for head in range(heads):
        rows = slice(head * full_seq, head * full_seq + seq)
        state_rows = slice(head * key_dim, (head + 1) * key_dim)
        state = capture["state"][state_rows].float()
        strict, value_beta, key_rhs, direct_rhs, intra, query, gate_cumulative = (
            diagnostic_terms(
                state,
                capture["query"][rows],
                capture["key"][rows],
                capture["value"][rows],
                capture["g"][rows],
                capture["beta"][rows],
                capture["m_incl"],
                capture["m_strict"],
            )
        )
        reference_inverse = forward_inverse(strict)
        forward_transposed_inverse = forward_inverse_transposed(strict)
        reference_value_new = (
            reference_inverse @ value_beta
            - (reference_inverse @ key_rhs) @ state
        )
        b00, b11, cross = block_inverse(strict)
        candidates = {
            "block_separate": (
                apply_block_inverse(b00, b11, cross, value_beta)
                - apply_block_inverse(b00, b11, cross, key_rhs) @ state
            ),
            "block_direct_rhs": apply_block_inverse(
                b00, b11, cross, direct_rhs
            ),
            "forward_separate": (
                forward_solve(strict, value_beta)
                - forward_solve(strict, key_rhs) @ state
            ),
            "forward_direct_rhs": forward_solve(strict, direct_rhs),
            "forward_transposed_separate": (
                forward_transposed_inverse @ value_beta
                - (forward_transposed_inverse @ key_rhs) @ state
            ),
        }
        for name, value_new in candidates.items():
            output = (
                (query * torch.exp(gate_cumulative)) @ state
                + intra @ value_new
            )
            reference_output = (
                (query * torch.exp(gate_cumulative)) @ state
                + intra @ reference_value_new
            )
            error = (output - reference_output).abs().max().item()
            if error > worst[name][0]:
                worst[name] = (
                    error,
                    head,
                    state.abs().max().item(),
                    reference_value_new.abs().max().item(),
                )

    for name, (error, head, state_abs, value_new_abs) in worst.items():
        print(
            f"{name}: output_max={error:.6e} head={head} "
            f"state_abs={state_abs:.6e} ref_vnew_abs={value_new_abs:.6e}"
        )


def reference_and_block(state, query, key, value, gate, beta, m_incl, m_strict):
    chunk, key_dim = query.shape
    half = chunk // 2
    query = F.normalize(query.float(), dim=-1, eps=1e-6) / math.sqrt(key_dim)
    key = F.normalize(key.float(), dim=-1, eps=1e-6)
    gate_cumulative = m_incl @ gate.float()
    decay = torch.exp(
        (gate_cumulative - gate_cumulative.T) * m_incl
    ) * m_incl
    strict = (-(key * beta.float()) @ key.T * decay) * m_strict

    reference_inverse = forward_inverse(strict)
    b00, b11, cross = block_inverse(strict)
    value_beta = value.float() * beta.float()
    key_rhs = key * beta.float() * torch.exp(gate_cumulative)
    value_reference = reference_inverse @ value_beta
    key_reference = reference_inverse @ key_rhs
    value_block = apply_block_inverse(b00, b11, cross, value_beta)
    key_block = apply_block_inverse(b00, b11, cross, key_rhs)

    intra = ((query @ key.T) * decay) * m_incl
    value_new_reference = value_reference - key_reference @ state.float()
    value_new_block = value_block - key_block @ state.float()
    output_reference = (
        (query * torch.exp(gate_cumulative)) @ state.float()
        + intra @ value_new_reference
    )
    output_block = (
        (query * torch.exp(gate_cumulative)) @ state.float()
        + intra @ value_new_block
    )
    total_decay = gate_cumulative[-1]
    state_reference = state.float() * torch.exp(total_decay)
    state_block = state_reference.clone()
    weighted_key = key * torch.exp(total_decay - gate_cumulative)
    state_reference += weighted_key.T @ value_new_reference
    state_block += weighted_key.T @ value_new_block
    return {
        "value": (value_block, value_reference),
        "key": (key_block, key_reference),
        "output": (output_block, output_reference),
        "state": (state_block, state_reference),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()
    capture = torch.load(args.capture, map_location="cpu", weights_only=True)
    key_dim = capture["query"].shape[1]
    heads = capture["state"].shape[0] // key_dim
    full_seq = capture["query"].shape[0] // heads
    seq = args.seq_len or full_seq
    if not 0 < seq <= full_seq:
        raise ValueError(f"seq_len must be in [1, {full_seq}], got {seq}")
    if args.diagnose:
        diagnose_capture(capture, seq)
    outputs = {name: [] for name in ("value", "key", "output", "state")}

    for head in range(heads):
        rows = slice(head * full_seq, head * full_seq + seq)
        state_rows = slice(head * key_dim, (head + 1) * key_dim)
        result = reference_and_block(
            capture["state"][state_rows],
            capture["query"][rows],
            capture["key"][rows],
            capture["value"][rows],
            capture["g"][rows],
            capture["beta"][rows],
            capture["m_incl"],
            capture["m_strict"],
        )
        for name in outputs:
            actual, expected = result[name]
            outputs[name].append((actual - expected).abs().max().item())

    print(
        " ".join(
            f"{name}_max={max(errors):.6e}"
            for name, errors in outputs.items()
        )
    )


if __name__ == "__main__":
    main()
