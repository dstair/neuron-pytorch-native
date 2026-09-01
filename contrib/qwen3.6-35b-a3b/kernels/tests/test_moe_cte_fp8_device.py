#!/usr/bin/env python3
"""Device equivalence test for the FP8xFP8 CTE prefill MoE kernel.

Step 3 of the FP8 prefill plan. Builds synthetic experts, quantizes them to
256x256-block FP8 (the layout nkilib's is_block_quant double_row path consumes),
runs nki_moe_cte_fp8_routed_35b on the Trn2, and compares against nkilib's own
torch reference (moe_cte_torch_ref, is_block_quant=True).

The reference matmuls FP8 weights against FULL-PRECISION activations, whereas the
kernel casts activations to FP8 (bare cast, no per-token scale — see
bwmm_shard_on_I.py:1095). So the reported cosine reflects the *total* FP8 error
(256x256 weight blocks + activation cast). A high cosine means the activation cast
is tolerable; a low one localizes the loss to activations (the known risk).

Run inside the DLC on the Trn2 with NEURON_LOGICAL_NC_CONFIG=1 and the LNC=1
nkilib patch applied (see run_test_fp8_cte.sh). Single process / single logical
core — this validates kernel compile + block-quant correctness, not TP collectives.
"""

import argparse
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
# kernels/ FIRST, matching static_decode_35b.py. With ROOT first, any stale
# top-level copy of a kernel module SHADOWS kernels/<name>.py -- the trn2 box
# carries 23 such pre-August leftovers, and this test silently validated a
# 5-week-old moe_cte_35b.py because of it (2026-08-22).
sys.path[:0] = [str(ROOT / "kernels"), str(ROOT)]

from moe_cte_adapter import pack_local_routes  # noqa: E402
from moe_cte_fp8_35b import nki_moe_cte_fp8_routed_35b  # noqa: E402
from moe_w8 import (  # noqa: E402
    LEGACY_E4M3_MAX,
    decode_legacy_e4m3,
    encode_legacy_e4m3,
)
from topology_35b import LNC_DEGREE  # noqa: E402
from nkilib.core.moe.moe_cte.moe_cte_torch import moe_cte_torch_ref  # noqa: E402
from nkilib.core.moe.moe_cte.bwmm_func import BWMMFunc  # noqa: E402
from nkilib.core.utils.common_types import (  # noqa: E402
    ActFnType,
    ExpertAffinityScaleMode,
)

# Weights are stored as nl.int8 holding LEGACY-E4M3 bytes: torch has no legacy-e4m3
# dtype, and neuronx-cc rejects F8E4M3FN operands on TRN2, so the kernel takes int8 and
# .view()s to nl.float8_e4m3 internally (mirrors the decode moe_fused_w8 path). Quantize
# against the legacy max.
E4M3_MAX = LEGACY_E4M3_MAX
BQ = 256


def quantize_gate_up_256(w_bf16):
    """[E,H,2,I] bf16 -> (int8 [E,H,2,I], scale [E,H//256,2,I//256,128]).

    int8 holds legacy-E4M3 bytes; scale is the per-256x256 (H-block, gate/up, I-block)
    dequant factor (true ~= decode_legacy_e4m3(int8) * scale), broadcast over the 128
    partition rows the kernel loads. Layout matches moe_cte_torch_ref's gup_scale
    [e,:,:,:,0] indexing and bwmm_shard_on_I's gup reshape.
    """
    E, H, two, I = w_bf16.shape
    assert two == 2 and H % BQ == 0 and I % BQ == 0
    HB, IB = H // BQ, I // BQ
    w = w_bf16.float()
    i8 = torch.empty(E, H, 2, I, dtype=torch.int8)
    scale = torch.empty(E, HB, 2, IB, 128, dtype=torch.float32)
    for e in range(E):
        for g in range(2):
            for hb in range(HB):
                for ib in range(IB):
                    blk = w[e, hb * BQ:(hb + 1) * BQ, g, ib * BQ:(ib + 1) * BQ]
                    s = blk.abs().amax().clamp_min(1e-12) / E4M3_MAX
                    i8[e, hb * BQ:(hb + 1) * BQ, g, ib * BQ:(ib + 1) * BQ] = (
                        encode_legacy_e4m3(blk / s)
                    )
                    scale[e, hb, g, ib, :] = s
    return i8, scale


def quantize_down_256(w_bf16):
    """[E,I,H] bf16 -> (int8 [E,I,H], scale [E,I//256,H//256,128])."""
    E, I, H = w_bf16.shape
    assert I % BQ == 0 and H % BQ == 0
    IB, HB = I // BQ, H // BQ
    w = w_bf16.float()
    i8 = torch.empty(E, I, H, dtype=torch.int8)
    scale = torch.empty(E, IB, HB, 128, dtype=torch.float32)
    for e in range(E):
        for ib in range(IB):
            for hb in range(HB):
                blk = w[e, ib * BQ:(ib + 1) * BQ, hb * BQ:(hb + 1) * BQ]
                s = blk.abs().amax().clamp_min(1e-12) / E4M3_MAX
                i8[e, ib * BQ:(ib + 1) * BQ, hb * BQ:(hb + 1) * BQ] = (
                    encode_legacy_e4m3(blk / s)
                )
                scale[e, ib, hb, :] = s
    return i8, scale


def run(tokens, hidden_size, intermediate_size, block_size, min_cosine):
    num_local_experts = 32 if LNC_DEGREE == 1 else 64
    expert_lo = num_local_experts
    gen = torch.Generator().manual_seed(41)

    routes = (
        torch.arange(tokens * 8, dtype=torch.int32).reshape(tokens, 8) % 8
        + expert_lo
    )
    metadata = pack_local_routes(
        routes, expert_lo, num_local_experts, block_size
    )
    hidden = torch.randn(
        tokens + 1, hidden_size, generator=gen, dtype=torch.bfloat16
    ) * 0.5
    affinities = torch.rand(
        (tokens + 1) * num_local_experts, 1, generator=gen, dtype=torch.bfloat16
    )
    # Mask the padding token (id == tokens): its affinities are 0 in the real model.
    # Zeroing makes padded blocks contribute 0 in BOTH the kernel and the reference.
    affinities.view(tokens + 1, num_local_experts)[tokens, :] = 0.0
    gate_up = torch.randn(
        num_local_experts, hidden_size, 2, intermediate_size,
        generator=gen, dtype=torch.bfloat16,
    ) * 0.05
    down = torch.randn(
        num_local_experts, intermediate_size, hidden_size,
        generator=gen, dtype=torch.bfloat16,
    ) * 0.05

    gate_up_i8, gate_up_scale = quantize_gate_up_256(gate_up)
    down_i8, down_scale = quantize_down_256(down)
    # The reference matmuls FP8 *values*; recover them from the int8 legacy-e4m3 bytes
    # (exactly what the kernel's .view(nl.float8_e4m3) yields), so ref and kernel agree.
    gate_up_ref = decode_legacy_e4m3(gate_up_i8).float()
    down_ref = decode_legacy_e4m3(down_i8).float()

    # The reference indexes affinities by block_to_expert BEFORE its expert>=E skip,
    # so clamp the padding-block sentinel (== num_local_experts) to a valid local id
    # for the reference call. Padded blocks still contribute 0 via the zeroed padding
    # affinity above. The kernel takes routes and handles the sentinel internally.
    block_to_expert_ref = metadata[1].clone()
    block_to_expert_ref[block_to_expert_ref >= num_local_experts] = 0

    # CPU reference: nkilib's own torch ref, is_block_quant=True. Uses the same
    # FP8 weights + block scales (dequantized in-ref), full-precision activations.
    ref = moe_cte_torch_ref(
        hidden_states=hidden,
        expert_affinities_masked=affinities,
        gate_up_proj_weight=gate_up_ref,
        down_proj_weight=down_ref,
        token_position_to_id=metadata[0],
        block_to_expert=block_to_expert_ref,
        block_size=block_size,
        bwmm_func=BWMMFunc.SHARD_ON_INTERMEDIATE,
        lnc_degree=LNC_DEGREE,
        conditions=metadata[2],
        gate_up_proj_scale=gate_up_scale,
        down_proj_scale=down_scale,
        is_block_quant=True,
        activation_function=ActFnType.SiLU,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    )["output"].float()

    from torch_neuronx import wrap_nki

    device = torch.device("neuron:0")
    fused = wrap_nki(nki_moe_cte_fp8_routed_35b)[LNC_DEGREE]
    out = fused(
        hidden.to(device),
        affinities.to(device),
        gate_up_i8.to(device),
        down_i8.to(device),
        gate_up_scale.to(device),
        down_scale.to(device),
        routes.to(device),
        expert_lo,
        block_size,
    )
    torch.neuron.synchronize()
    got = out.cpu().float()

    finite = bool(torch.isfinite(got).all())
    cos = torch.nn.functional.cosine_similarity(
        got.flatten().double(), ref.flatten().double(), dim=0
    ).item()
    rel = (got - ref).norm().item() / ref.norm().clamp_min(1e-12).item()
    print(
        f"FP8 CTE device vs torch_ref: cosine={cos:.6f} rel_l2={rel:.4f} "
        f"finite={finite} | T={tokens} H={hidden_size} I={intermediate_size} "
        f"block={block_size} E={num_local_experts} LNC={LNC_DEGREE}"
    )
    assert finite, "non-finite kernel output"
    assert cos >= min_cosine, f"cosine {cos:.6f} < {min_cosine}"
    print("PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--block-size", type=int, choices=(256, 512), default=256)
    parser.add_argument("--min-cosine", type=float, default=0.95)
    args = parser.parse_args()
    run(
        args.tokens,
        args.hidden_size,
        args.intermediate_size,
        args.block_size,
        args.min_cosine,
    )


if __name__ == "__main__":
    main()
