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
from topology_35b import LNC_DEGREE  # noqa: E402


def main():
    torch.manual_seed(11)
    batch = 2
    heads = 2 if LNC_DEGREE == 1 else 4
    active = 512
    prior = 2048
    dim = 256
    q = torch.randn(batch, heads, active, dim, dtype=torch.bfloat16)
    ka = torch.randn(batch, dim, active, dtype=torch.bfloat16)
    va = torch.randn(batch, active, dim, dtype=torch.bfloat16)
    kp = torch.randn(batch, dim, prior, dtype=torch.bfloat16)
    vp = torch.randn(batch, prior, dim, dtype=torch.bfloat16)

    for used in (0, 1024):
        used_t = torch.tensor([used], dtype=torch.int32)
        actual = torch.ops.gqa35b.cte_prefill(
            (q.reshape(batch * heads, active, dim).float() / math.sqrt(dim))
            .to(torch.bfloat16)
            .to("neuron"),
            ka.to("neuron"),
            va.to("neuron"),
            kp.to("neuron"),
            vp.to("neuron"),
            used_t.to("neuron"),
        ).cpu().float().reshape(batch, heads, active, dim)

        keys = torch.cat([kp[:, :, :used], ka], dim=2).float()
        values = torch.cat([vp[:, :used], va], dim=1).float()
        scores = torch.matmul(q.float(), keys.unsqueeze(1)) / math.sqrt(dim)
        causal = torch.arange(used + active)[None, :] <= (
            used + torch.arange(active)[:, None]
        )
        scores = scores.masked_fill(
            ~causal.reshape(1, 1, active, used + active),
            float("-inf"),
        )
        expected = torch.matmul(
            torch.softmax(scores, dim=-1),
            values.float().unsqueeze(1),
        )

        cosine = torch.nn.functional.cosine_similarity(
            actual.reshape(-1), expected.reshape(-1), dim=0
        )
        max_diff = (actual - expected).abs().max()
        print(
            f"LNC={LNC_DEGREE} prior_used={used} cosine={float(cosine):.6f} "
            f"max_diff={float(max_diff):.4e}"
        )
        assert float(cosine) > 0.999


if __name__ == "__main__":
    main()
