#!/usr/bin/env python3
"""Device correctness test for the nkilib prefix-cache GQA prefill path."""

import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.dirname(HERE)
sys.path.insert(0, KERNELS)

import gqa_cte_35b_ops  # noqa: E402,F401


def main():
    torch.manual_seed(11)
    heads = 4
    active = 512
    prior = 2048
    dim = 256
    q = torch.randn(heads, active, dim, dtype=torch.bfloat16)
    ka = torch.randn(1, dim, active, dtype=torch.bfloat16)
    va = torch.randn(1, active, dim, dtype=torch.bfloat16)
    kp = torch.randn(1, dim, prior, dtype=torch.bfloat16)
    vp = torch.randn(1, prior, dim, dtype=torch.bfloat16)

    for used in (0, 1024):
        used_t = torch.tensor([used], dtype=torch.int32)
        actual = torch.ops.gqa35b.cte_prefill(
            (q.float() / math.sqrt(dim)).to(torch.bfloat16).to("neuron"),
            ka.to("neuron"),
            va.to("neuron"),
            kp.to("neuron"),
            vp.to("neuron"),
            used_t.to("neuron"),
        ).cpu().float()

        keys = torch.cat([kp[:, :, :used], ka], dim=2).float()
        values = torch.cat([vp[:, :used], va], dim=1).float()
        scores = torch.matmul(q.float(), keys) / math.sqrt(dim)
        causal = torch.arange(used + active)[None, :] <= (
            used + torch.arange(active)[:, None]
        )
        scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
        expected = torch.matmul(torch.softmax(scores, dim=-1), values.float())

        cosine = torch.nn.functional.cosine_similarity(
            actual.reshape(-1), expected.reshape(-1), dim=0
        )
        max_diff = (actual - expected).abs().max()
        print(
            f"prior_used={used} cosine={float(cosine):.6f} "
            f"max_diff={float(max_diff):.4e}"
        )
        assert float(cosine) > 0.999


if __name__ == "__main__":
    main()
