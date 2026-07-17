#!/usr/bin/env python3
"""CPU oracle for append-after-attention stateful K/V decode."""

import math

import torch


def cloned_cache_attention(query, key, value, cache_k, cache_v, layer, position):
    updated_k = cache_k.clone()
    updated_v = cache_v.clone()
    updated_k[layer, :, position] = key
    updated_v[layer, :, position] = value
    keys = updated_k[layer, :, : position + 1].float()
    values = updated_v[layer, :, : position + 1].float()
    scores = torch.matmul(query.float(), keys.transpose(1, 2))
    weights = torch.softmax(scores / math.sqrt(query.shape[-1]), dim=-1)
    return torch.matmul(weights, values), updated_k, updated_v


def append_after_attention(query, key, value, cache_k, cache_v, layer, position):
    prior_k = cache_k[layer, :, :position].float()
    prior_v = cache_v[layer, :, :position].float()
    current_score = torch.matmul(
        query.float(), key.float().unsqueeze(-1)
    )
    if position:
        prior_scores = torch.matmul(
            query.float(), prior_k.transpose(1, 2)
        )
        scores = torch.cat((prior_scores, current_score), dim=-1)
        values = torch.cat((prior_v, value.float().unsqueeze(1)), dim=1)
    else:
        scores = current_score
        values = value.float().unsqueeze(1)
    weights = torch.softmax(scores / math.sqrt(query.shape[-1]), dim=-1)
    output = torch.matmul(weights, values)
    cache_k[layer, :, position] = key
    cache_v[layer, :, position] = value
    return output


def run_case(batch_size, sequence, position):
    torch.manual_seed(1000 + batch_size + sequence + position)
    layers, query_heads, head_dim = 3, 2, 16
    layer = 1
    query = torch.randn(batch_size, query_heads, head_dim)
    key = torch.randn(batch_size, head_dim).to(torch.bfloat16)
    value = torch.randn(batch_size, head_dim).to(torch.bfloat16)
    cache_k = torch.randn(
        layers, batch_size, sequence, head_dim, dtype=torch.bfloat16
    )
    cache_v = torch.randn_like(cache_k)
    original_k = cache_k.clone()
    original_v = cache_v.clone()

    expected, expected_k, expected_v = cloned_cache_attention(
        query, key, value, cache_k, cache_v, layer, position
    )
    actual = append_after_attention(
        query, key, value, cache_k, cache_v, layer, position
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(cache_k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(cache_v, expected_v, rtol=0, atol=0)
    torch.testing.assert_close(cache_k[:layer], original_k[:layer], rtol=0, atol=0)
    torch.testing.assert_close(cache_k[layer + 1 :], original_k[layer + 1 :], rtol=0, atol=0)
    torch.testing.assert_close(cache_v[:layer], original_v[:layer], rtol=0, atol=0)
    torch.testing.assert_close(cache_v[layer + 1 :], original_v[layer + 1 :], rtol=0, atol=0)
    if position + 1 < sequence:
        torch.testing.assert_close(
            cache_k[layer, :, position + 1 :],
            original_k[layer, :, position + 1 :],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            cache_v[layer, :, position + 1 :],
            original_v[layer, :, position + 1 :],
            rtol=0,
            atol=0,
        )


def test_stateful_cache_positions():
    for batch_size in (1, 8, 64):
        for sequence, position in ((1, 0), (17, 7), (32, 31)):
            run_case(batch_size, sequence, position)


if __name__ == "__main__":
    test_stateful_cache_positions()
    print("PASS: append-after-attention matches cloned K/V cache")
