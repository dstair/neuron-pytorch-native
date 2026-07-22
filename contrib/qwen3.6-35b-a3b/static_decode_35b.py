#!/usr/bin/env python3
"""
Qwen3.5-35B-A3B (MoE) — static decode/prefill for torch.compile(fullgraph=True,
backend="neuron"). PyTorch Native Beta.

Correctness-first bring-up. Reuses the proven 27B scaffolding (RoPE, RMSNorm,
functional all-reduce, compile/main loop) from examples/qwen3_6/static_decode.py
but with the 35B architecture:
  - 40 layers = [DeltaNet×3, GQA×1]×10, hidden 2048
  - DeltaNet decode: pure-torch recurrent step (deltanet_decode.py), NO NKI kernel
    yet (kernel head-constants are 27B-specific; perf lever for Task 5).
  - GQA: 16 Q / 2 KV heads, head_dim 256, sigmoid output gate, partial RoPE.
    KV heads (2) don't divide TP=4 → KV heads are REPLICATED across cores.
  - MoE: 256 experts top-8 + sigmoid-gated shared expert, masked-dense grouped
    bmm (validated in kernels/tests/test_moe_oracle_cpu.py), expert-parallel.

Usage (on the trn2 box, inside the Native DLC):
    torchrun --nproc-per-node=4 static_decode_35b.py \
        --max-seq-len 20000 --num-tokens 16 [--tiny --num-layers 4] [--skip-prefill]

Dims come from model_dims.py. This file is intentionally separate from the 27B.
"""
import os
import sys
import time
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed._functional_collectives import all_reduce as _functional_all_reduce

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels"))
import model_dims as D
from deltanet_decode import deltanet_recurrent_step
from moe_w8 import (
    build_local_affinities,
    OfficialFP8ExpertReader,
    QuantizationStats,
    ROW_FP8_PROJECTION_CHOICES,
)
from topology_35b import LNC_DEGREE

MODEL_PATH = os.environ.get("QWEN35_MODEL_PATH", "/models/Qwen3.5-35B-A3B")
EXPERT_MODEL_PATH = os.environ.get("QWEN35_FP8_MODEL_PATH")

# DeltaNet decode via the NKI kernel (env DN_NKI=1). Default OFF = pure-torch
# recurrence (the CPU-validated correctness oracle). The kernel makes DeltaNet
# opaque to neuronx-cc's tiler so the full 40-layer graph compiles (pure-torch
# einsum recurrence trips a PGTiling assertion past ~20 layers).
USE_DN_NKI = os.environ.get("DN_NKI", "0") == "1"
USE_DN_DIRECT_STATE_OUT = os.environ.get("DN_DIRECT_STATE_OUT", "0") == "1"
if USE_DN_DIRECT_STATE_OUT and not USE_DN_NKI:
    raise RuntimeError("DN_DIRECT_STATE_OUT=1 requires DN_NKI=1")
if USE_DN_DIRECT_STATE_OUT and os.environ.get("DNBATCHED_V2", "0") != "1":
    raise RuntimeError("DN_DIRECT_STATE_OUT=1 requires DNBATCHED_V2=1")
if USE_DN_NKI:
    import deltanet_full_batched_35b_ops  # registers torch.ops.deltanet35b.full_batched

# True-sparse MoE dispatch (env MOE_SPARSE=1). Gathers only the selected experts
# per token (index_select, static shape) vs the default masked-dense path that
# computes all E local experts. ~8x expert-FLOP/HBM win at BS=1. Default OFF =
# masked-dense (the CPU-validated oracle). Numerically identical (validated).
USE_MOE_SPARSE = os.environ.get("MOE_SPARSE", "0") == "1"

# BS=1 decode-only tensor parallelism within every routed expert. Each rank
# stores all expert IDs but only one world_size shard of the intermediate
# width, so the fixed top-8 gather reads one quarter of each expert at TP=4.
USE_MOE_DECODE_TP = os.environ.get("MOE_DECODE_TP", "0") == "1"
if USE_MOE_DECODE_TP and not USE_MOE_SPARSE:
    raise RuntimeError("MOE_DECODE_TP=1 requires MOE_SPARSE=1")

# GQA-tail mega-kernel (env GQATAIL=1). ONE custom call/layer folds q RMSNorm +
# partial-64 RoPE + scaled scores + masked softmax + weighted-V + output-gate,
# collapsing ~12 inter-op barriers. k-side norm/rope + KV write stay torch; o_proj
# stays F.linear. Ported from the 27B (Q_HEADS 6→4). Default OFF = torch path.
USE_GQA_TAIL = os.environ.get("GQATAIL", "0") == "1"
if USE_GQA_TAIL:
    import gqa_tail_35b_ops  # registers torch.ops.gqa35b.tail
USE_GQA_STATEFUL_KV = os.environ.get("GQA_STATEFUL_KV", "0") == "1"
if USE_GQA_STATEFUL_KV and not USE_GQA_TAIL:
    raise RuntimeError("GQA_STATEFUL_KV=1 requires GQATAIL=1")

# Flash causal-attention PREFILL kernel (env GQA_FLASH_PREFILL=1). Replaces the
# pure-torch full [S,S] causal attention in _gqa_prefill (which OOMs at S>~2k)
# with a flash, memory-flat-in-S kernel. Ported from PR#60 nki_flash_attn_d256.
# Requires NKV=1/core (KV replicated), which holds at TP=4. Default OFF.
USE_GQA_FLASH_PREFILL = os.environ.get("GQA_FLASH_PREFILL", "0") == "1"
if USE_GQA_FLASH_PREFILL:
    import gqa_flash_prefill_35b_ops  # registers torch.ops.gqa35b.flash_prefill

# Production nkilib context-encoding attention with fixed prior-cache storage and
# runtime prior_used_len. Unlike the local flash kernel, it only computes over
# the used prefix plus the active chunk.
USE_GQA_CTE_PREFILL = os.environ.get("GQA_CTE_PREFILL", "0") == "1"
if USE_GQA_CTE_PREFILL:
    if USE_GQA_FLASH_PREFILL:
        raise RuntimeError("GQA_CTE_PREFILL and GQA_FLASH_PREFILL are mutually exclusive")
    import gqa_cte_35b_ops  # registers torch.ops.gqa35b.cte_prefill

# Runtime-offset partial RoPE + aliased KV-cache writes. This removes q_base
# specialization from compiled bucket graphs; it requires the flash chunk kernel,
# which already consumes the same runtime scalar for its causal mask.
USE_GQA_DYNAMIC_ROPE_KV = os.environ.get("GQA_DYNAMIC_ROPE_KV", "0") == "1"
if USE_GQA_DYNAMIC_ROPE_KV:
    if not (USE_GQA_FLASH_PREFILL or USE_GQA_CTE_PREFILL):
        raise RuntimeError(
            "GQA_DYNAMIC_ROPE_KV requires GQA_FLASH_PREFILL=1 or GQA_CTE_PREFILL=1"
        )
    import gqa_rope_kv_35b_ops  # registers torch.ops.gqa35b.rope_kv_dynamic
if USE_GQA_CTE_PREFILL and not USE_GQA_DYNAMIC_ROPE_KV:
    raise RuntimeError("GQA_CTE_PREFILL requires GQA_DYNAMIC_ROPE_KV=1")

# NKI chunked-prefill DeltaNet kernel (env DN_CHUNK_NKI=1). ONE @nki.jit call/layer
# for the whole-sequence chunked gated-delta-rule (Woodbury doubling), replacing the
# pure-torch chunked_prefill loop that is compile-hostile under torch.compile
# (NCC_IMGN901 vectorizer bug). Ported from 27B deltanet_chunked_v2 (V_HEADS 12->8).
# Takes initial state, returns final state -> composes with bucketed prefill. Default OFF.
# C=64 is unstable on deep real-weight prefixes even though random kernel tests
# pass. C=32 is finite at S=2048 but fails at layer 18 around token 10752.
# C=16 is finite through two full 40-layer S=20000 passes.
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "16"))
USE_DN_CHUNK_NKI = os.environ.get("DN_CHUNK_NKI", "0") == "1"
if USE_DN_CHUNK_NKI:
    import deltanet_chunked_prefill_35b_ops  # registers torch.ops.deltanet35b.chunked_prefill

# Eager-only diagnostic for preserving the exact recurrent-kernel inputs at a
# long-context failure. Keep disabled for compiled production runs.
DN_CAPTURE_DIR = os.environ.get("DN_CAPTURE_DIR", "")
DN_CAPTURE_LAYER = int(os.environ.get("DN_CAPTURE_LAYER", "-1"))
DN_CAPTURE_CHUNK = int(os.environ.get("DN_CAPTURE_CHUNK", "-1"))

# FP8 expert weights (env MOE_FP8=1, requires MOE_SPARSE=1). The MoE expert
# GEMMs are ~90% of the BS=1 decode HBM read, and the step is 95% DMA-bound
# (profile: arith_intensity 0.999, MBU 78%). Storing experts as FP8 (int8 bytes)
# HALVES the gathered weight read → ~1.8x BS=1 ceiling AND ~16→8 GB/core resident.
# CRITICAL (27B lesson [[project_qwen36_fp8_kernel_profile]]): dequant must NOT
# be a separate GPSIMD pass. Here we gather only the top-8 selected experts'
# int8 weights, then dequant that SMALL [T*K,...] slice once (per-channel f32
# scale) — not all 64, not per-Linear — so the dequant is a single cheap
# multiply on the gathered slice, fused with the gather. Default OFF = bf16.
USE_MOE_FP8 = os.environ.get("MOE_FP8", "0") == "1"
if USE_MOE_DECODE_TP and USE_MOE_FP8:
    raise RuntimeError("MOE_DECODE_TP supports bf16 expert weights only")
FP8_E4M3_MAX = 240.0  # legacy e4m3 max (Trn2 nc_matmul format); OCP fn extends to 448

# Production nkilib MoE TKG kernel (env MOE_NKILIB=1). ONE fused @nki.jit call:
# routing + gate_up + act + down + affinity scaling, with is_all_expert toggle
# (dense for BS>=16, selective for BS=1) and FP8-ROW quant (TRN2 per-channel).
# Being one opaque kernel, it COMPILES at BS>=16 where our torch-bmm sparse path
# F137'd the host compiler. With MOE_FP8=1 it runs FP8-ROW; else bf16.
# Expose a compatible nki-library checkout through PYTHONPATH when enabling this.
USE_MOE_NKILIB = os.environ.get("MOE_NKILIB", "0") == "1"

# Context-encoding MoE kernel (env MOE_CTE=1). Unlike moe_tkg, moe_cte accepts
# long token dimensions and computes expert work in routed blocks. Routing
# metadata is built at runtime with fixed tensor shapes by moe_cte_adapter.py.
USE_MOE_CTE = os.environ.get("MOE_CTE", "0") == "1"
USE_MOE_CTE_NKI_PACK = os.environ.get("MOE_CTE_NKI_PACK", "0") == "1"
if USE_MOE_CTE:
    if USE_MOE_NKILIB:
        raise RuntimeError("MOE_CTE and MOE_NKILIB are mutually exclusive")

# High-batch decode-only fused W8 experts. FP8 defaults to row-scaled legacy
# E4M3 in nkilib's all-expert scheduler. Block power-of-two conversion preserves
# the official 128x128 scales up to an exact exponent shift; dual planes remain
# the exact accuracy control. INT8 requantizes each source block symmetrically.
MOE_FUSED_W8 = os.environ.get("MOE_FUSED_W8", "").strip().lower()
if MOE_FUSED_W8 not in ("", "fp8", "int8"):
    raise RuntimeError("MOE_FUSED_W8 must be unset, 'fp8', or 'int8'")
USE_MOE_FUSED_W8 = bool(MOE_FUSED_W8)
MOE_FUSED_W8_FP8_IMPL = os.environ.get(
    "MOE_FUSED_W8_FP8_IMPL", "row"
).strip().lower()
if MOE_FUSED_W8_FP8_IMPL not in (
    "row",
    "dual",
    "block_pow2",
    "block_pow2_coalesced",
):
    raise RuntimeError(
        "MOE_FUSED_W8_FP8_IMPL must be 'row', 'dual', 'block_pow2', "
        "or 'block_pow2_coalesced'"
    )
USE_MOE_FUSED_W8_ROW_FP8 = (
    MOE_FUSED_W8 == "fp8" and MOE_FUSED_W8_FP8_IMPL == "row"
)
USE_MOE_FUSED_W8_DUAL_FP8 = (
    MOE_FUSED_W8 == "fp8" and MOE_FUSED_W8_FP8_IMPL == "dual"
)
USE_MOE_FUSED_W8_BLOCK_COALESCED_FP8 = (
    MOE_FUSED_W8 == "fp8"
    and MOE_FUSED_W8_FP8_IMPL == "block_pow2_coalesced"
)
MOE_FUSED_W8_FP8_PROJECTIONS = os.environ.get(
    "MOE_FUSED_W8_FP8_PROJECTIONS", "all"
).strip().lower()
if MOE_FUSED_W8_FP8_PROJECTIONS not in ("all", "gate_up", "down"):
    raise RuntimeError(
        "MOE_FUSED_W8_FP8_PROJECTIONS must be 'all', 'gate_up', or 'down'"
    )
if (
    MOE_FUSED_W8_FP8_PROJECTIONS != "all"
    and not USE_MOE_FUSED_W8_ROW_FP8
):
    raise RuntimeError(
        "MOE_FUSED_W8_FP8_PROJECTIONS requires "
        "MOE_FUSED_W8=fp8 and MOE_FUSED_W8_FP8_IMPL=row"
    )
MOE_FUSED_W8_FP8_LAYER_START = int(
    os.environ.get("MOE_FUSED_W8_FP8_LAYER_START", "0")
)
MOE_FUSED_W8_FP8_LAYER_LIMIT = int(
    os.environ.get("MOE_FUSED_W8_FP8_LAYER_LIMIT", "40")
)
if not (
    0
    <= MOE_FUSED_W8_FP8_LAYER_START
    <= MOE_FUSED_W8_FP8_LAYER_LIMIT
    <= 40
):
    raise RuntimeError(
        "MOE_FUSED_W8_FP8_LAYER_START/LIMIT must define a range in [0, 40]"
    )
if (
    (
        MOE_FUSED_W8_FP8_LAYER_START != 0
        or MOE_FUSED_W8_FP8_LAYER_LIMIT != 40
    )
    and not USE_MOE_FUSED_W8_ROW_FP8
):
    raise RuntimeError(
        "MOE_FUSED_W8_FP8_LAYER_START/LIMIT requires "
        "MOE_FUSED_W8=fp8 and MOE_FUSED_W8_FP8_IMPL=row"
    )
MOE_FUSED_W8_LAYOUT = os.environ.get(
    "MOE_FUSED_W8_LAYOUT", "weight"
).strip().lower()
if MOE_FUSED_W8_LAYOUT not in ("weight", "token"):
    raise RuntimeError("MOE_FUSED_W8_LAYOUT must be 'weight' or 'token'")
if MOE_FUSED_W8_LAYOUT == "token" and MOE_FUSED_W8 != "fp8":
    raise RuntimeError(
        "MOE_FUSED_W8_LAYOUT=token requires MOE_FUSED_W8=fp8"
    )
if MOE_FUSED_W8 == "fp8" and MOE_FUSED_W8_LAYOUT != "weight":
    raise RuntimeError(
        "fused FP8 requires MOE_FUSED_W8_LAYOUT=weight"
    )
USE_MOE_OFFICIAL_FP8_REFERENCE = (
    os.environ.get("MOE_OFFICIAL_FP8_REFERENCE", "0") == "1"
)
USE_MOE_W8_RESIDUAL_FP32 = (
    os.environ.get("MOE_W8_RESIDUAL_FP32", "0") == "1"
)
if USE_MOE_W8_RESIDUAL_FP32 and not (
    USE_MOE_FUSED_W8 or USE_MOE_OFFICIAL_FP8_REFERENCE
):
    raise RuntimeError(
        "MOE_W8_RESIDUAL_FP32=1 requires a fused-W8 candidate or "
        "official-FP8 reference"
    )
MOE_OFFICIAL_FP8_REFERENCE_ROUNDING = frozenset(
    value.strip()
    for value in os.environ.get(
        "MOE_OFFICIAL_FP8_REFERENCE_ROUNDING", ""
    ).split(",")
    if value.strip()
)
_MOE_REFERENCE_ROUNDING_STAGES = frozenset(
    ("affinity_bf16", "activation_bf16", "local_output_bf16")
)
unknown_reference_rounding = (
    MOE_OFFICIAL_FP8_REFERENCE_ROUNDING - _MOE_REFERENCE_ROUNDING_STAGES
)
if unknown_reference_rounding:
    raise RuntimeError(
        "MOE_OFFICIAL_FP8_REFERENCE_ROUNDING contains unknown stages: "
        + ", ".join(sorted(unknown_reference_rounding))
    )
if (
    MOE_OFFICIAL_FP8_REFERENCE_ROUNDING
    and not USE_MOE_OFFICIAL_FP8_REFERENCE
):
    raise RuntimeError(
        "MOE_OFFICIAL_FP8_REFERENCE_ROUNDING requires "
        "MOE_OFFICIAL_FP8_REFERENCE=1"
    )
if USE_MOE_FUSED_W8 and USE_MOE_OFFICIAL_FP8_REFERENCE:
    raise RuntimeError(
        "MOE_FUSED_W8 and MOE_OFFICIAL_FP8_REFERENCE are mutually exclusive"
    )
if USE_MOE_FUSED_W8 or USE_MOE_OFFICIAL_FP8_REFERENCE:
    conflicts = [
        name
        for name, enabled in (
            ("MOE_SPARSE", USE_MOE_SPARSE),
            ("MOE_DECODE_TP", USE_MOE_DECODE_TP),
            ("MOE_CTE", USE_MOE_CTE),
            ("MOE_CTE_NKI_PACK", USE_MOE_CTE_NKI_PACK),
            ("MOE_NKILIB", USE_MOE_NKILIB),
            ("MOE_FP8", USE_MOE_FP8),
        )
        if enabled
    ]
    if conflicts:
        mode_name = (
            "MOE_FUSED_W8"
            if USE_MOE_FUSED_W8
            else "MOE_OFFICIAL_FP8_REFERENCE"
        )
        raise RuntimeError(
            mode_name + " is mutually exclusive with " + ", ".join(conflicts)
        )

if USE_MOE_FP8:
    import fp8_group_matvec_ops  # registers torch.ops.fp8moe.group_matvec
if USE_MOE_NKILIB:
    from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg as _nkilib_moe_tkg
    from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode
if USE_MOE_CTE:
    from torch_neuronx import wrap_nki
    if USE_MOE_CTE_NKI_PACK:
        from moe_cte_35b import nki_moe_cte_routed_35b
        _nkilib_moe_cte_hop = wrap_nki(nki_moe_cte_routed_35b)[LNC_DEGREE]
    else:
        from moe_cte_adapter import pack_local_routes
        from moe_cte_35b import nki_moe_cte_35b
        _nkilib_moe_cte_hop = wrap_nki(nki_moe_cte_35b)[LNC_DEGREE]
if USE_MOE_FUSED_W8:
    if USE_MOE_FUSED_W8_ROW_FP8:
        import moe_tkg_row_fp8_35b_ops  # registers tkg_row_fp8
    else:
        import moe_fused_w8_35b_ops  # registers block-W8 ops


def uses_row_fp8_layer(layer_index):
    """Return whether this layer uses the row-FP8 expert kernel."""
    return (
        USE_MOE_FUSED_W8_ROW_FP8
        and MOE_FUSED_W8_FP8_LAYER_START
        <= layer_index
        < MOE_FUSED_W8_FP8_LAYER_LIMIT
    )


def _load_fused_w8_expert_layer(
    expert_reader,
    layer_index,
    expert_start,
    expert_end,
    mode,
    fp8_impl,
):
    """Load one fused-W8 layer, retaining baseline BF16 outside its FP8 range."""
    if mode == "fp8" and fp8_impl == "row":
        if uses_row_fp8_layer(layer_index):
            return expert_reader.load_layer_row_fp8(
                layer_index,
                expert_start,
                expert_end,
                fp8_projections=MOE_FUSED_W8_FP8_PROJECTIONS,
            )
        return (
            expert_reader.load_layer_bf16(
                layer_index, expert_start, expert_end
            ),
            None,
        )
    return expert_reader.load_layer(
        layer_index,
        expert_start,
        expert_end,
        mode,
        fp8_impl=fp8_impl,
    )


def rw_to_global(rw_top, sel, device):
    """Scatter normalized top-k weights [T,K] back to a dense [T, E_all] affinity
    matrix (0 for unselected) — the global affinities nkilib all-expert mode wants."""
    T, K = sel.shape
    aff = torch.zeros(T, D.NUM_EXPERTS, dtype=torch.float, device=device)
    aff.scatter_(1, sel.to(torch.int64), rw_top.float())
    return aff


def quantize_experts_fp8(w):
    """w: [E, OUT, IN] bf16 -> (w_i8_T [E,IN,OUT] int8, scale [E,OUT] f32).
    Per-(expert, output-channel) symmetric absmax quant. int8 holds the fp8
    e4m3 bytes (dodges the HLO F8E4M3FN verifier; bit-identical for normals).
    Returned PRE-TRANSPOSED to [E,IN,OUT] so the nki_fp8_group_matvec kernel
    DMAs the weight with IN (contraction) on the partition dim directly."""
    absmax = w.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-12)  # [E,OUT,1]
    scale = absmax / FP8_E4M3_MAX                                          # [E,OUT,1]
    w_q = (w.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    w_i8_T = w_q.view(torch.int8).transpose(1, 2).contiguous()            # [E,IN,OUT]
    return w_i8_T, scale.to(torch.float32).contiguous()                   # [E,IN,OUT],[E,OUT,1]

# TP collective cost-probe (matches 27B NOREDUCE lever; default off = correct).
NO_REDUCE = os.environ.get("NOREDUCE", "0") == "1"

# Compile the entire decode step, including embedding, state handling, LM head,
# and token selection, into one graph. The default segmented path remains useful
# when a full-depth graph exceeds a compiler limit.
USE_DECODE_FULLGRAPH = os.environ.get("DECODE_FULLGRAPH", "0") == "1"
USE_DECODE_SHARDED_LM_HEAD = (
    os.environ.get("DECODE_SHARDED_LM_HEAD", "0") == "1"
)
if USE_DN_DIRECT_STATE_OUT and not USE_DECODE_FULLGRAPH:
    raise RuntimeError("DN_DIRECT_STATE_OUT=1 requires DECODE_FULLGRAPH=1")
if USE_GQA_STATEFUL_KV and not USE_DECODE_FULLGRAPH:
    raise RuntimeError("GQA_STATEFUL_KV=1 requires DECODE_FULLGRAPH=1")
if (
    USE_MOE_FUSED_W8 or USE_MOE_OFFICIAL_FP8_REFERENCE
) and not USE_DECODE_FULLGRAPH:
    mode_name = (
        "MOE_FUSED_W8"
        if USE_MOE_FUSED_W8
        else "MOE_OFFICIAL_FP8_REFERENCE"
    )
    raise RuntimeError(f"{mode_name} requires DECODE_FULLGRAPH=1")


def functional_all_reduce(x, op, group):
    # No-op when probing collective cost, or when there is a single rank
    # (CPU correctness path) — nothing to reduce across.
    if NO_REDUCE or len(group) <= 1:
        return x
    return _functional_all_reduce(x, op, group)


# ─── Norm / RoPE primitives (identical math to the 27B) ──────────────────────
def rms_norm(x, weight):
    """RMSNorm with Qwen3.5 residual weight: (1 + weight) * normalize(x)."""
    x_f32 = x.float()
    norm = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + D.RMS_EPS)
    return ((1.0 + weight.float()) * norm).to(x.dtype)


def decode_rms_norm(x, weight):
    normalized = rms_norm(x, weight)
    if USE_MOE_W8_RESIDUAL_FP32:
        return normalized.to(torch.bfloat16)
    return normalized


def rms_norm_gated(x, gate, weight):
    """RMSNormGated: normalize(x) * weight * silu(gate). Per value head."""
    x_f32 = x.float()
    norm = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + D.RMS_EPS)
    return (weight * norm.to(x.dtype)) * F.silu(gate.float()).to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Partial rotary: rotate [..., :rope_dim], pass the rest through."""
    rd = cos.shape[-1]
    q_rot, q_pass = q[..., :rd], q[..., rd:]
    k_rot, k_pass = k[..., :rd], k[..., rd:]
    q_e = torch.cat(((q_rot * cos) + (rotate_half(q_rot) * sin), q_pass), dim=-1)
    k_e = torch.cat(((k_rot * cos) + (rotate_half(k_rot) * sin), k_pass), dim=-1)
    return q_e, k_e


# ─── MoE layer (masked-dense grouped bmm; expert-parallel) ───────────────────
def moe_forward(x, router_w, gate_up, down, e_lo, e_hi,
                sh_gate, sh_up, sh_down, sh_sigmoid, fp8_scales=None,
                return_routing=False):
    """Masked-dense (or true-sparse) MoE for a [T, H] activation block.

    Validated equivalent to HF sparse routing in test_moe_oracle_cpu.py.
    Router runs full top-8 (replicated); only this rank's experts contribute.

    By default, `gate_up`/`down` contain this rank's expert-parallel rows and
    `e_lo`/`e_hi` map global top-k ids to them. MOE_DECODE_TP instead stores all
    expert ids with a rank-local intermediate-width shard and uses global ids
    directly. Routing math is global in both layouts.

    Returns (routed_partial, shared): routed_partial holds only this rank's
    experts (the caller all-reduces it across ranks), and shared is the
    shared-expert output computed from REPLICATED weights (identical on every
    rank) which the caller adds AFTER the reduce so it is counted exactly once.
    """
    T, H = x.shape
    E = e_hi - e_lo
    xf = x.float()

    # Router (replicated): softmax top-8 with norm_topk_prob.
    logits = F.linear(xf, router_w.float())               # [T, E_all]
    rw = F.softmax(logits, dim=1, dtype=torch.float)
    rw_top, sel = torch.topk(rw, D.TOP_K, dim=-1)          # [T, k]
    if D.NORM_TOPK_PROB:
        rw_top = rw_top / rw_top.sum(dim=-1, keepdim=True)

    # Dense per-(token, local-expert) gate, 0 where unselected. Built with a
    # static loop over TOP_K + a one-hot scatter — compile-safe (no data-dep
    # shapes). gate[t, e_local] = sum_j rw_top[t,j] * [sel[t,j]==e_lo+e_local]
    gate = torch.zeros(T, E, dtype=torch.float, device=x.device)
    for j in range(D.TOP_K):
        ej = sel[:, j]                                     # [T] global idx
        local = ej - e_lo
        on = (ej >= e_lo) & (ej < e_hi)
        oh = F.one_hot(local.clamp(0, E - 1), E).float()   # [T, E]
        gate = gate + oh * (rw_top[:, j] * on.float()).unsqueeze(-1)
    if "affinity_bf16" in MOE_OFFICIAL_FP8_REFERENCE_ROUNDING:
        gate = gate.to(torch.bfloat16).float()

    if os.environ.get("MOE_SHARED_ONLY", "0") == "1":
        # DIAGNOSTIC: skip the expert bmm entirely (routed=0). Isolates whether
        # the masked-dense grouped bmm is the neuronx-cc PGTiling trigger.
        routed = torch.zeros(T, H, dtype=torch.float, device=x.device)
    elif USE_MOE_DECODE_TP:
        if T != 1:
            raise RuntimeError(
                f"MOE_DECODE_TP is decode-only and requires one token, found {T}"
            )
        # Every rank owns the same global expert IDs and a disjoint shard of
        # their intermediate width. The selected expert gather is therefore
        # balanced and reads TOP_K/world_size of the old per-rank weight bytes.
        idx = sel.reshape(-1)
        x_sel = xf.unsqueeze(1).expand(T, D.TOP_K, H).reshape(-1, H)
        local_i = gate_up.shape[1] // 2
        gup_g = gate_up.index_select(0, idx).float()
        dn_g = down.index_select(0, idx).float()
        gu = torch.bmm(
            x_sel.unsqueeze(1), gup_g.transpose(1, 2)
        ).squeeze(1)
        hh = F.silu(gu[:, :local_i]) * gu[:, local_i:]
        y = torch.bmm(
            hh.unsqueeze(1), dn_g.transpose(1, 2)
        ).squeeze(1)
        routed = (
            y * rw_top.reshape(-1, 1)
        ).reshape(T, D.TOP_K, H).sum(dim=1)
    elif USE_MOE_SPARSE:
        # TRUE SPARSE dispatch: gather ONLY the selected experts' weights per
        # (token, slot), instead of reading all E local experts. At BS=1 decode
        # this reads K=8 expert matrices vs E=64 — the ~8x expert-FLOP / HBM-
        # bandwidth win (decode is weight-read bound). index_select has a STATIC
        # output shape ([T*K,...]); only the index VALUES are data-dependent, so
        # it compiles under fullgraph (same pattern as the KV-cache scatter).
        # Non-local experts (this rank doesn't own them) are gathered at a clamped
        # dummy index and zeroed via `is_local`; the cross-rank all-reduce in the
        # caller then sums each rank's owned contributions into the full top-8.
        # NOTE: cost scales as T*K, so this beats masked-dense only while T*K < E
        # (great at BS=1; revisit with TP-within-expert / token-permute for T>~8).
        local = (sel - e_lo)                                   # [T,K] global->local
        is_local = ((sel >= e_lo) & (sel < e_hi)).float()      # [T,K]
        idx = local.clamp(0, E - 1).reshape(-1)                # [T*K] LOCAL rows
        x_sel = xf.unsqueeze(1).expand(T, D.TOP_K, H).reshape(-1, H)  # [T*K, H]
        if fp8_scales is not None:
            # FP8 path: gate_up/down are int8 e4m3 bytes PRE-TRANSPOSED to
            # [E,IN,OUT]. Gather only the top-8 experts' int8 weights (HALF the
            # HBM bytes — the win) + their per-channel scales, then run the FP8
            # grouped matvec kernel (FP8 stationary in nc_matmul, dequant fused
            # on the PSUM copy — compiles, and no GPSIMD dequant pass).
            gu_s, dn_s = fp8_scales                            # [E,2I],[E,H]
            I = gate_up.shape[2] // 2                          # gate_up is [E,H,2I]
            gup_i8 = gate_up.index_select(0, idx)              # [T*K,H,2I] int8
            dn_i8 = down.index_select(0, idx)                  # [T*K,I,H] int8
            gu = torch.ops.fp8moe.group_matvec(
                x_sel.to(torch.bfloat16), gup_i8, gu_s.index_select(0, idx))   # [T*K,2I]
            hh = F.silu(gu[:, :I]) * gu[:, I:]                 # [T*K, I]
            y = torch.ops.fp8moe.group_matvec(
                hh.to(torch.bfloat16), dn_i8, dn_s.index_select(0, idx))       # [T*K,H]
        else:
            I = gate_up.shape[1] // 2
            gup_g = gate_up.index_select(0, idx).float()       # [T*K, 2I, H]
            dn_g = down.index_select(0, idx).float()           # [T*K, H, I]
            gu = torch.bmm(x_sel.unsqueeze(1), gup_g.transpose(1, 2)).squeeze(1)  # [T*K,2I]
            hh = F.silu(gu[:, :I]) * gu[:, I:]                 # [T*K, I]
            y = torch.bmm(hh.unsqueeze(1), dn_g.transpose(1, 2)).squeeze(1)       # [T*K,H]
        w = (rw_top * is_local).reshape(-1, 1)                 # [T*K,1] (0 if not local)
        routed = (y * w).reshape(T, D.TOP_K, H).sum(dim=1)     # [T, H]
    else:
        gup = gate_up.float()                                  # [E, 2I, H] (local)
        dn = down.float()                                      # [E, H, I] (local)
        x_exp = xf.unsqueeze(0).expand(E, T, H)                # [E, T, H]
        gu = torch.bmm(x_exp, gup.transpose(1, 2))             # [E, T, 2I]
        I = gu.shape[-1] // 2
        g_, u_ = gu[:, :, :I], gu[:, :, I:]
        h = F.silu(g_) * u_                                    # [E, T, I]
        if "activation_bf16" in MOE_OFFICIAL_FP8_REFERENCE_ROUNDING:
            h = h.to(torch.bfloat16).float()
        y = torch.bmm(h, dn.transpose(1, 2))                   # [E, T, H]
        routed = (y * gate.t().unsqueeze(-1)).sum(dim=0)       # [T, H]

    if "local_output_bf16" in MOE_OFFICIAL_FP8_REFERENCE_ROUNDING:
        routed = routed.to(torch.bfloat16).float()

    # Sigmoid-gated shared expert (replicated weights; computed on each rank).
    sgate = torch.sigmoid(F.linear(xf, sh_sigmoid.float()))   # [T, 1]
    sh = F.linear(F.silu(F.linear(xf, sh_gate.float())) * F.linear(xf, sh_up.float()),
                  sh_down.float())                            # [T, H]
    shared = sgate * sh
    if return_routing:
        return routed, shared, sel, rw_top
    return routed, shared


class StaticDecode35B(nn.Module):
    """35B-A3B static decode/prefill module (per-core, TP-sharded weights)."""

    def __init__(self, weights, max_seq_len, world_size, batch_size=1, rank=0):
        super().__init__()
        if USE_MOE_FUSED_W8 and batch_size not in (32, 64, 128, 256):
            raise RuntimeError(
                "MOE_FUSED_W8 supports batch sizes 32, 64, 128, and 256"
            )
        if (
            USE_MOE_FUSED_W8_BLOCK_COALESCED_FP8
            and batch_size not in (32, 64, 128)
        ):
            raise RuntimeError(
                "block_pow2_coalesced supports batch sizes 32, 64, and 128"
            )
        self.max_seq_len = max_seq_len
        self.world_size = world_size
        self.tp_group = list(range(world_size))
        self.batch_size = batch_size
        td = D.tp_dims(world_size)
        self.td = td
        # Global expert range for the default expert-parallel layout. The
        # decode-only TP-within-expert path stores every expert id and ignores
        # this range when gathering routed weights.
        self.rank = rank
        self.e_lo = rank * td["experts_per_core"]
        self.e_hi = self.e_lo + td["experts_per_core"]
        if USE_MOE_FUSED_W8_ROW_FP8:
            self.register_buffer(
                "moe_row_fp8_rank_id",
                torch.zeros((1, 1), dtype=torch.uint32),
                persistent=False,
            )
        self.nkv = max(1, D.GQA_KV_HEADS // world_size)   # KV heads per core
        if USE_GQA_STATEFUL_KV:
            if self.nkv != 1:
                raise RuntimeError(
                    "GQA_STATEFUL_KV requires one local KV head per rank"
                )
            cache_shape = (
                D.NUM_GQA,
                batch_size,
                self.nkv,
                max_seq_len,
                D.GQA_HEAD_DIM,
            )
            self.register_buffer(
                "decode_kv_k",
                torch.zeros(cache_shape, dtype=torch.bfloat16),
                persistent=False,
            )
            self.register_buffer(
                "decode_kv_v",
                torch.zeros(cache_shape, dtype=torch.bfloat16),
                persistent=False,
            )
        # Layer segments for graph-split compile (default: single segment = whole
        # model). setup_segments() splits the layer range so each compiled NEFF
        # stays under the neuronx-cc PGTiling collective limit.
        self._segments = [(0, D.NUM_LAYERS, self._run_layers)]
        self._diagnostic_layers = ()

        # [C,C] host constants for the chunked-prefill NKI kernel (no iota in this
        # build). m_incl doubles as the cumsum operator. Only needed when the kernel
        # is active, but cheap to always register.
        C = CHUNK_SIZE
        _idx = torch.arange(C); _i = _idx.view(C, 1); _j = _idx.view(1, C)
        self.register_buffer("chunk_m_incl", (_i >= _j).float())
        self.register_buffer("chunk_m_strict", (_i > _j).float())
        self.register_buffer("chunk_eye", torch.eye(C, dtype=torch.float32))

        def reg(name, w, src, key):
            self.register_buffer(name, w)
            src.pop(key, None)

        reg("embed", weights["embed"], weights, "embed")
        reg("final_norm", weights["final_norm"], weights, "final_norm")
        reg("lm_head_w", weights["lm_head"], weights, "lm_head")

        for i, lw in enumerate(weights["layers"]):
            reg(f"l{i}_input_norm", lw["input_norm"], lw, "input_norm")
            reg(f"l{i}_post_norm", lw["post_norm"], lw, "post_norm")
            # MoE (all layers)
            reg(f"l{i}_router", lw["router"], lw, "router")
            if USE_MOE_FUSED_W8:
                if uses_row_fp8_layer(i):
                    reg(
                        f"l{i}_row_gate_up",
                        lw["row_gate_up"],
                        lw,
                        "row_gate_up",
                    )
                    reg(
                        f"l{i}_row_gate_up_scale",
                        lw["row_gate_up_scale"],
                        lw,
                        "row_gate_up_scale",
                    )
                    reg(f"l{i}_row_down", lw["row_down"], lw, "row_down")
                    reg(
                        f"l{i}_row_down_scale",
                        lw["row_down_scale"],
                        lw,
                        "row_down_scale",
                    )
                elif USE_MOE_FUSED_W8_ROW_FP8:
                    reg(
                        f"l{i}_gate_up",
                        lw["gate_up"],
                        lw,
                        "gate_up",
                    )
                    reg(f"l{i}_down", lw["down"], lw, "down")
                else:
                    reg(
                        f"l{i}_w8_gate_up",
                        lw["w8_gate_up"],
                        lw,
                        "w8_gate_up",
                    )
                    if USE_MOE_FUSED_W8_DUAL_FP8:
                        reg(
                            f"l{i}_w8_gate_up_residual",
                            lw["w8_gate_up_residual"],
                            lw,
                            "w8_gate_up_residual",
                        )
                    reg(
                        f"l{i}_w8_gate_up_scale",
                        lw["w8_gate_up_scale"],
                        lw,
                        "w8_gate_up_scale",
                    )
                    reg(f"l{i}_w8_down", lw["w8_down"], lw, "w8_down")
                    if USE_MOE_FUSED_W8_DUAL_FP8:
                        reg(
                            f"l{i}_w8_down_residual",
                            lw["w8_down_residual"],
                            lw,
                            "w8_down_residual",
                        )
                    reg(
                        f"l{i}_w8_down_scale",
                        lw["w8_down_scale"],
                        lw,
                        "w8_down_scale",
                    )
            elif USE_MOE_NKILIB or USE_MOE_CTE:
                # Repack to the nkilib MoE layout: gate_up [E,2I,H]->[E,H,2,I],
                # down [E,H,I]->[E,I,H]. FP8-ROW quant (per-out-channel) when MOE_FP8.
                gu = lw["gate_up"]; dn = lw["down"]              # [E,2I,H],[E,H,I]
                Ec = gu.shape[0]; II = gu.shape[1] // 2; HH = gu.shape[2]
                if USE_MOE_FP8:
                    if USE_MOE_CTE:
                        raise RuntimeError("MOE_CTE does not yet support MOE_FP8")
                    gq, gs = quantize_experts_fp8(gu)           # weights are e4m3 below
                    dq, ds = quantize_experts_fp8(dn)
                    # quantize_experts_fp8 returns int8 [E,IN,OUT] + scale [E,OUT,1];
                    # reconstruct e4m3 weights in [E,2I,H]/[E,H,I] + ROW scales.
                    gu_q = gq.view(torch.float8_e4m3fn).transpose(1, 2).contiguous()  # [E,2I,H] e4m3
                    dn_q = dq.view(torch.float8_e4m3fn).transpose(1, 2).contiguous()  # [E,H,I]  e4m3
                    gate_up_k = gu_q.reshape(Ec, 2, II, HH).permute(0, 3, 1, 2).contiguous()  # [E,H,2,I]
                    down_k = dn_q.permute(0, 2, 1).contiguous()                               # [E,I,H]
                    self.register_buffer(f"l{i}_k_gate_up", gate_up_k)
                    self.register_buffer(f"l{i}_k_down", down_k)
                    self.register_buffer(f"l{i}_k_gu_s", gs.squeeze(-1).reshape(Ec, 2, II).contiguous())  # [E,2,I]
                    self.register_buffer(f"l{i}_k_dn_s", ds.squeeze(-1).contiguous())                     # [E,H]
                else:
                    gate_up_k = gu.reshape(Ec, 2, II, HH).permute(0, 3, 1, 2).contiguous()  # [E,H,2,I]
                    down_k = dn.permute(0, 2, 1).contiguous()                              # [E,I,H]
                    self.register_buffer(f"l{i}_k_gate_up", gate_up_k)
                    self.register_buffer(f"l{i}_k_down", down_k)
                lw.pop("gate_up", None); lw.pop("down", None)
            elif USE_MOE_FP8:
                # Quantize experts to FP8 (int8 bytes) + per-(expert,out-ch) scale.
                gu_i8, gu_s = quantize_experts_fp8(lw["gate_up"])   # [E,2I,H] i8, [E,2I,1] f32
                dn_i8, dn_s = quantize_experts_fp8(lw["down"])      # [E,H,I]  i8, [E,H,1]  f32
                self.register_buffer(f"l{i}_gate_up_q", gu_i8)
                self.register_buffer(f"l{i}_gate_up_s", gu_s)
                self.register_buffer(f"l{i}_down_q", dn_i8)
                self.register_buffer(f"l{i}_down_s", dn_s)
                lw.pop("gate_up", None); lw.pop("down", None)       # free bf16
            else:
                reg(f"l{i}_gate_up", lw["gate_up"], lw, "gate_up")
                reg(f"l{i}_down", lw["down"], lw, "down")
            reg(f"l{i}_sh_gate", lw["sh_gate"], lw, "sh_gate")
            reg(f"l{i}_sh_up", lw["sh_up"], lw, "sh_up")
            reg(f"l{i}_sh_down", lw["sh_down"], lw, "sh_down")
            reg(f"l{i}_sh_sigmoid", lw["sh_sigmoid"], lw, "sh_sigmoid")

            if D.layer_type(i) == "deltanet":
                reg(f"l{i}_dn_qkv", lw["dn_qkv"], lw, "dn_qkv")
                reg(f"l{i}_dn_conv_w", lw["dn_conv_w"], lw, "dn_conv_w")
                reg(f"l{i}_dn_z", lw["dn_z"], lw, "dn_z")
                reg(f"l{i}_dn_a", lw["dn_a"], lw, "dn_a")
                reg(f"l{i}_dn_b", lw["dn_b"], lw, "dn_b")
                reg(f"l{i}_dn_out", lw["dn_out"], lw, "dn_out")
                reg(f"l{i}_dn_A_log", lw["dn_A_log"], lw, "dn_A_log")
                reg(f"l{i}_dn_dt_bias", lw["dn_dt_bias"], lw, "dn_dt_bias")
                reg(f"l{i}_dn_norm", lw["dn_norm"], lw, "dn_norm")
            else:
                reg(f"l{i}_gqa_q", lw["gqa_q"], lw, "gqa_q")
                reg(f"l{i}_gqa_k", lw["gqa_k"], lw, "gqa_k")
                reg(f"l{i}_gqa_v", lw["gqa_v"], lw, "gqa_v")
                reg(f"l{i}_gqa_o", lw["gqa_o"], lw, "gqa_o")
                reg(f"l{i}_gqa_q_norm", lw["gqa_q_norm"], lw, "gqa_q_norm")
                reg(f"l{i}_gqa_k_norm", lw["gqa_k_norm"], lw, "gqa_k_norm")

        self._init_rope(max_seq_len)

    def _lin(self, name, x):
        w = getattr(self, name)
        return F.linear(x.to(w.dtype), w)

    def _init_rope(self, max_seq_len):
        rd = D.ROPE_DIM
        inv_freq = 1.0 / (D.ROPE_THETA ** (torch.arange(0, rd, 2).float() / rd))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)            # [S, rd]
        self.register_buffer("rope_cos", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("rope_sin", emb.sin().unsqueeze(0).unsqueeze(0))

    # ── MoE helper bound to a layer index ──
    def _moe(self, i, x):
        """x: [B, 1, H] (decode) or [1, S, H] (prefill). Returns same shape."""
        lead = x.shape[:-1]
        x2d = x.reshape(-1, D.HIDDEN)
        if USE_MOE_FUSED_W8 and (
            not USE_MOE_FUSED_W8_ROW_FP8 or uses_row_fp8_layer(i)
        ):
            return self._moe_fused_w8(i, x2d, lead).to(x.dtype)
        if USE_MOE_CTE:
            return self._moe_cte(i, x2d, lead).to(x.dtype)
        if USE_MOE_NKILIB:
            # moe_tkg maps tokens to the NKI partition dimension (max 128).
            # Prefill buckets are larger, so invoke the opaque kernel in fixed
            # token chunks and restore the original leading shape afterward.
            T = x2d.shape[0]
            chunk = int(os.environ.get("MOE_PREFILL_CHUNK", "128"))
            if chunk > 0 and T > chunk:
                parts = []
                for cs in range(0, T, chunk):
                    ce = min(cs + chunk, T)
                    parts.append(self._moe_nkilib(i, x2d[cs:ce], (ce - cs,)))
                return torch.cat(parts, dim=0).reshape(*lead, D.HIDDEN).to(x.dtype)
            return self._moe_nkilib(i, x2d, lead).to(x.dtype)
        if USE_MOE_FP8:
            gate_up = getattr(self, f"l{i}_gate_up_q")   # int8 fp8 bytes
            down = getattr(self, f"l{i}_down_q")
            fp8_scales = (getattr(self, f"l{i}_gate_up_s"), getattr(self, f"l{i}_down_s"))
        else:
            gate_up = getattr(self, f"l{i}_gate_up")
            down = getattr(self, f"l{i}_down")
            fp8_scales = None
        # Masked-dense MoE builds [E, T, ...] intermediates (all local experts over
        # all tokens). At prefill T is large so this OOMs (~[64,T,2048] ≈ 0.5MB*T).
        # MoE is per-token independent, so chunk over the token axis when T exceeds
        # MOE_PREFILL_CHUNK (default 512; 0=off). Numerically identical; caps peak.
        T = x2d.shape[0]
        chunk = int(os.environ.get("MOE_PREFILL_CHUNK", "512"))

        def _mf(xin):
            return moe_forward(
                xin, getattr(self, f"l{i}_router"),
                gate_up, down,
                self.e_lo, self.e_hi,
                getattr(self, f"l{i}_sh_gate"), getattr(self, f"l{i}_sh_up"),
                getattr(self, f"l{i}_sh_down"), getattr(self, f"l{i}_sh_sigmoid"),
                fp8_scales=fp8_scales,
            )

        if chunk > 0 and T > chunk:
            r_parts, s_parts = [], []
            for cs in range(0, T, chunk):
                rp, sp = _mf(x2d[cs:cs + chunk])
                r_parts.append(rp); s_parts.append(sp)
            routed = torch.cat(r_parts, dim=0)
            shared = torch.cat(s_parts, dim=0)
        else:
            routed, shared = _mf(x2d)
        # Reduce the routed-expert partials across ranks (expert-parallel), then
        # add the shared expert (replicated, identical on every rank → added
        # exactly once, AFTER the reduce, so no W-fold over-count).
        routed = functional_all_reduce(routed, "sum", self.tp_group)
        out = routed + shared
        return out.reshape(*lead, D.HIDDEN).to(x.dtype)

    def _moe_routing(self, i, x2d):
        xf = x2d.float()
        logits = F.linear(xf, getattr(self, f"l{i}_router").float())
        routing = F.softmax(logits, dim=1, dtype=torch.float)
        top_weights, top_ids = torch.topk(routing, D.TOP_K, dim=-1)
        if D.NORM_TOPK_PROB:
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        return xf, top_ids, top_weights

    def _shared_expert(self, i, xf):
        shared_gate = torch.sigmoid(
            F.linear(xf, getattr(self, f"l{i}_sh_sigmoid").float())
        )
        shared = F.linear(
            F.silu(F.linear(xf, getattr(self, f"l{i}_sh_gate").float()))
            * F.linear(xf, getattr(self, f"l{i}_sh_up").float()),
            getattr(self, f"l{i}_sh_down").float(),
        )
        return shared_gate * shared

    def _moe_fused_w8_parts(self, i, x2d):
        """Return routing plus local/global/shared FP32 fused-W8 stages."""
        if x2d.shape[0] != self.batch_size:
            raise RuntimeError("MOE_FUSED_W8 is restricted to one-token decode")
        local_experts = self.e_hi - self.e_lo
        xf, top_ids, top_weights = self._moe_routing(i, x2d)
        affinities = build_local_affinities(
            top_ids, top_weights, self.e_lo, local_experts
        )

        if USE_MOE_FUSED_W8_ROW_FP8:
            if not uses_row_fp8_layer(i):
                raise RuntimeError(
                    f"layer {i} is outside the configured row-FP8 range"
                )
            local_ids = (top_ids - self.e_lo).clamp(
                0, local_experts - 1
            ).to(torch.int32)
            local_routed = torch.ops.moe_w8.tkg_row_fp8(
                x2d.to(torch.bfloat16),
                getattr(self, f"l{i}_row_gate_up"),
                getattr(self, f"l{i}_row_down"),
                getattr(self, f"l{i}_row_gate_up_scale"),
                getattr(self, f"l{i}_row_down_scale"),
                affinities.to(torch.bfloat16),
                local_ids,
                self.moe_row_fp8_rank_id,
            )
        elif USE_MOE_FUSED_W8_DUAL_FP8:
            local_routed = torch.ops.moe_w8.fused_fp8_dual(
                x2d.to(torch.bfloat16),
                getattr(self, f"l{i}_w8_gate_up"),
                getattr(self, f"l{i}_w8_gate_up_residual"),
                getattr(self, f"l{i}_w8_down"),
                getattr(self, f"l{i}_w8_down_residual"),
                getattr(self, f"l{i}_w8_gate_up_scale"),
                getattr(self, f"l{i}_w8_down_scale"),
                affinities,
            )
        elif USE_MOE_FUSED_W8_BLOCK_COALESCED_FP8:
            local_routed = torch.ops.moe_w8.fused_fp8_block_coalesced(
                x2d.to(torch.bfloat16),
                getattr(self, f"l{i}_w8_gate_up"),
                getattr(self, f"l{i}_w8_down"),
                getattr(self, f"l{i}_w8_gate_up_scale"),
                getattr(self, f"l{i}_w8_down_scale"),
                affinities,
            )
        elif MOE_FUSED_W8 == "fp8":
            local_routed = torch.ops.moe_w8.fused_fp8_native(
                x2d.to(torch.bfloat16),
                getattr(self, f"l{i}_w8_gate_up"),
                getattr(self, f"l{i}_w8_down"),
                getattr(self, f"l{i}_w8_gate_up_scale"),
                getattr(self, f"l{i}_w8_down_scale"),
                affinities,
            )
        else:
            local_routed = torch.ops.moe_w8.fused_int8(
                x2d.to(torch.bfloat16),
                getattr(self, f"l{i}_w8_gate_up"),
                getattr(self, f"l{i}_w8_down"),
                getattr(self, f"l{i}_w8_gate_up_scale"),
                getattr(self, f"l{i}_w8_down_scale"),
                affinities,
            )
        global_routed = functional_all_reduce(
            local_routed, "sum", self.tp_group
        )
        shared = self._shared_expert(i, xf)
        return (
            top_ids,
            top_weights,
            local_routed,
            global_routed,
            shared,
        )

    def _moe_fused_w8(self, i, x2d, lead):
        """High-batch all-expert W8 call; router/shared/TP semantics stay external."""
        _, _, _, global_routed, shared = self._moe_fused_w8_parts(
            i, x2d
        )
        return (global_routed + shared).reshape(*lead, D.HIDDEN)

    def _moe_reference_parts(self, i, x2d):
        """Return the same stages from the official-FP8-dequantized oracle."""
        skipped_row_fp8_layer = (
            USE_MOE_FUSED_W8_ROW_FP8 and not uses_row_fp8_layer(i)
        )
        if (
            not USE_MOE_OFFICIAL_FP8_REFERENCE
            and not skipped_row_fp8_layer
        ):
            raise RuntimeError(
                "reference MoE diagnostics require "
                "MOE_OFFICIAL_FP8_REFERENCE=1"
            )
        local_routed, shared, top_ids, top_weights = moe_forward(
            x2d,
            getattr(self, f"l{i}_router"),
            getattr(self, f"l{i}_gate_up"),
            getattr(self, f"l{i}_down"),
            self.e_lo,
            self.e_hi,
            getattr(self, f"l{i}_sh_gate"),
            getattr(self, f"l{i}_sh_up"),
            getattr(self, f"l{i}_sh_down"),
            getattr(self, f"l{i}_sh_sigmoid"),
            return_routing=True,
        )
        global_routed = functional_all_reduce(
            local_routed, "sum", self.tp_group
        )
        return (
            top_ids,
            top_weights,
            local_routed,
            global_routed,
            shared,
        )

    def _moe_diagnostic_parts(self, i, x2d):
        if USE_MOE_FUSED_W8 and (
            not USE_MOE_FUSED_W8_ROW_FP8 or uses_row_fp8_layer(i)
        ):
            return self._moe_fused_w8_parts(i, x2d)
        return self._moe_reference_parts(i, x2d)

    def _moe_cte(self, i, x2d, lead):
        """Long-token MoE using nkilib's blockwise context-encoding kernel."""
        T = x2d.shape[0]
        E = self.e_hi - self.e_lo
        block_size = int(os.environ.get("MOE_CTE_BLOCK", "512"))
        if block_size % 256:
            raise ValueError("MOE_CTE_BLOCK must be a multiple of 256")
        if USE_MOE_CTE_NKI_PACK and block_size not in (256, 512):
            raise ValueError("MOE_CTE_NKI_PACK supports MOE_CTE_BLOCK=256 or 512")

        xf = x2d.float()
        logits = F.linear(xf, getattr(self, f"l{i}_router").float())
        rw = F.softmax(logits, dim=1, dtype=torch.float)
        rw_top, sel = torch.topk(rw, D.TOP_K, dim=-1)
        if D.NORM_TOPK_PROB:
            rw_top = rw_top / rw_top.sum(dim=-1, keepdim=True)

        local = sel - self.e_lo
        on_rank = (local >= 0) & (local < E)
        local_safe = local.clamp(0, E - 1)
        affinities = torch.zeros(T, E, dtype=torch.float, device=x2d.device)
        affinities.scatter_add_(
            1, local_safe, rw_top * on_rank.to(rw_top.dtype)
        )
        affinities = torch.cat(
            [affinities, torch.zeros(1, E, dtype=affinities.dtype, device=x2d.device)]
        ).to(torch.bfloat16)
        hidden = torch.cat(
            [x2d.to(torch.bfloat16), torch.zeros(1, D.HIDDEN, dtype=torch.bfloat16, device=x2d.device)]
        )
        if USE_MOE_CTE_NKI_PACK:
            routed = _nkilib_moe_cte_hop(
                hidden,
                affinities.reshape(-1, 1),
                getattr(self, f"l{i}_k_gate_up"),
                getattr(self, f"l{i}_k_down"),
                sel.to(torch.int32),
                self.e_lo,
                block_size,
            )
        else:
            token_position_to_id, block_to_expert, conditions = pack_local_routes(
                sel.to(torch.int32), self.e_lo, E, block_size
            )
            routed = _nkilib_moe_cte_hop(
                hidden,
                affinities.reshape(-1, 1),
                getattr(self, f"l{i}_k_gate_up"),
                getattr(self, f"l{i}_k_down"),
                token_position_to_id,
                block_to_expert,
                conditions,
                block_size,
            )
        routed = routed[:T]
        if os.environ.get("MOE_CTE_RETURN_ROUTED", "0") == "1":
            return routed.reshape(*lead, D.HIDDEN)
        if os.environ.get("MOE_CTE_SYNC_BEFORE_SHARED", "0") == "1":
            torch.neuron.synchronize()
        routed = functional_all_reduce(routed.float(), "sum", self.tp_group)

        sg = torch.sigmoid(F.linear(xf, getattr(self, f"l{i}_sh_sigmoid").float()))
        sh = F.linear(
            F.silu(F.linear(xf, getattr(self, f"l{i}_sh_gate").float()))
            * F.linear(xf, getattr(self, f"l{i}_sh_up").float()),
            getattr(self, f"l{i}_sh_down").float(),
        )
        return (routed + sg * sh).reshape(*lead, D.HIDDEN)

    def _moe_nkilib(self, i, x2d, lead):
        """MoE via the nkilib moe_tkg fused kernel. Router computed here; the
        kernel does gate_up+act+down+affinity-scaling internally. is_all_expert
        by batch (dense >=16, selective <16). FP8-ROW when MOE_FP8. Returns [..,H]."""
        T = x2d.shape[0]
        E = self.e_hi - self.e_lo                         # local experts
        xf = x2d.float()
        # Router (global top-k, norm_topk_prob), then map to LOCAL expert space.
        logits = F.linear(xf, getattr(self, f"l{i}_router").float())   # [T, E_all]
        rw = F.softmax(logits, dim=1, dtype=torch.float)
        rw_top, sel = torch.topk(rw, D.TOP_K, dim=-1)                  # [T,K]
        if D.NORM_TOPK_PROB:
            rw_top = rw_top / rw_top.sum(dim=-1, keepdim=True)
        # Local affinities [T, E]: scatter normalized weight to this rank's experts.
        aff = torch.zeros(T, E, dtype=torch.float, device=x2d.device)
        for j in range(D.TOP_K):
            ej = sel[:, j]; loc = ej - self.e_lo
            on = (ej >= self.e_lo) & (ej < self.e_hi)
            aff = aff + F.one_hot(loc.clamp(0, E - 1), E).float() * (rw_top[:, j] * on.float()).unsqueeze(-1)
        # Local top-k index (kernel selective mode); clamp out-of-rank to 0 (aff=0 there).
        idx = (sel - self.e_lo).clamp(0, E - 1).to(torch.int32)        # [T,K]

        is_all = (T >= 16)
        kw = dict(is_all_expert=is_all,
                  expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
                  activation_fn=ActFnType.SiLU)
        if is_all:
            # All-expert mode: kernel wants GLOBAL affinities [T,E_all] (slices to
            # [T,E_L] internally via rank_id) + rank_id telling it which experts
            # this worker owns ([E_L*rank, E_L*(rank+1))). Our EP sharding = rank.
            aff_in = rw_to_global(rw_top, sel, x2d.device)   # [T, E_all]
            kw["rank_id"] = torch.tensor([[self.rank]], dtype=torch.int32, device=x2d.device)
            kw["mask_unselected_experts"] = True
        else:
            aff_in = aff                                     # local [T, E]
        if USE_MOE_FP8:
            routed = _nkilib_moe_tkg(
                x2d.to(torch.bfloat16), getattr(self, f"l{i}_k_gate_up"),
                getattr(self, f"l{i}_k_down"), aff_in, idx,
                expert_gate_up_weights_scale=getattr(self, f"l{i}_k_gu_s"),
                expert_down_weights_scale=getattr(self, f"l{i}_k_dn_s"), **kw)
        else:
            routed = _nkilib_moe_tkg(
                x2d.to(torch.bfloat16), getattr(self, f"l{i}_k_gate_up"),
                getattr(self, f"l{i}_k_down"), aff_in, idx, **kw)
        routed = functional_all_reduce(routed.float(), "sum", self.tp_group)
        # shared expert (replicated, added once after reduce)
        sg = torch.sigmoid(F.linear(xf, getattr(self, f"l{i}_sh_sigmoid").float()))
        sh = F.linear(F.silu(F.linear(xf, getattr(self, f"l{i}_sh_gate").float()))
                      * F.linear(xf, getattr(self, f"l{i}_sh_up").float()),
                      getattr(self, f"l{i}_sh_down").float())
        return (routed + sg * sh).reshape(*lead, D.HIDDEN)

    def _decode_hidden(self, input_id, position, deltanet_states, conv_states,
                       kv_cache_k, kv_cache_v):
        hidden = F.embedding(input_id, self.embed).unsqueeze(1)   # [B,1,H]
        if USE_MOE_W8_RESIDUAL_FP32:
            hidden = hidden.float()
        if USE_DN_DIRECT_STATE_OUT:
            dn_states = torch.empty_like(deltanet_states)
            cv_states = torch.empty_like(conv_states)
        else:
            dn_states = deltanet_states.clone()
            cv_states = conv_states.clone()
        if USE_GQA_STATEFUL_KV:
            kv_k = self.decode_kv_k
            kv_v = self.decode_kv_v
        else:
            kv_k = kv_cache_k.clone()
            kv_v = kv_cache_v.clone()

        cos = self.rope_cos.squeeze(0).squeeze(0).index_select(
            0, position.unsqueeze(0)).unsqueeze(0).unsqueeze(0)    # [1,1,1,rd]
        sin = self.rope_sin.squeeze(0).squeeze(0).index_select(
            0, position.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

        # Run layer segments. Each entry in self._segments is (lo, hi, fn) where
        # fn is either the plain bound _run_layers or a torch.compile'd wrapper.
        for (lo, hi, fn) in self._segments:
            if USE_DN_DIRECT_STATE_OUT:
                hidden = fn(
                    lo, hi, hidden, cos, sin, position,
                    deltanet_states, conv_states, dn_states, cv_states,
                    kv_k, kv_v,
                )
            else:
                hidden = fn(
                    lo, hi, hidden, cos, sin, position,
                    dn_states, cv_states, kv_k, kv_v,
                )

        hidden = rms_norm(hidden, self.final_norm)
        if USE_MOE_W8_RESIDUAL_FP32:
            hidden = hidden.to(torch.bfloat16)
        return hidden, dn_states, cv_states, kv_k, kv_v

    # ── Decode forward (B tokens in, B logits out) ──
    def forward(self, input_id, position, deltanet_states, conv_states,
                kv_cache_k, kv_cache_v):
        hidden, dn_states, cv_states, kv_k, kv_v = self._decode_hidden(
            input_id, position, deltanet_states, conv_states, kv_cache_k, kv_cache_v
        )
        logits = self._lin("lm_head_w", hidden)   # [B,1,V]
        return logits.squeeze(1), dn_states, cv_states, kv_k, kv_v

    def decode_step(self, input_id, position, deltanet_states, conv_states,
                    kv_cache_k, kv_cache_v):
        if USE_DECODE_SHARDED_LM_HEAD:
            hidden, dn_states, cv_states, kv_k, kv_v = self._decode_hidden(
                input_id, position, deltanet_states, conv_states,
                kv_cache_k, kv_cache_v
            )
            if D.VOCAB % self.world_size:
                raise RuntimeError(
                    "DECODE_SHARDED_LM_HEAD requires vocab size divisible by TP"
                )
            vocab_per_rank = D.VOCAB // self.world_size
            vocab_lo = self.rank * vocab_per_rank
            vocab_hi = vocab_lo + vocab_per_rank
            logits = F.linear(
                hidden, self.lm_head_w[vocab_lo:vocab_hi]
            ).squeeze(1)
            local_max, local_id = logits.max(dim=-1)
            global_max = functional_all_reduce(
                local_max, "max", self.tp_group
            )
            global_id = local_id.to(torch.int32) + vocab_lo
            invalid_id = torch.full_like(global_id, -D.VOCAB)
            neg_winner_id = torch.where(
                local_max == global_max, -global_id, invalid_id
            )
            next_id = -functional_all_reduce(
                neg_winner_id, "max", self.tp_group
            )
            return (
                logits, next_id.to(torch.long),
                dn_states, cv_states, kv_k, kv_v,
            )
        outputs = self.forward(
            input_id, position, deltanet_states, conv_states, kv_cache_k, kv_cache_v
        )
        logits = outputs[0]
        next_id = logits.argmax(-1).to(torch.long)
        return logits, next_id, *outputs[1:]

    def decode_step_stateful(self, input_id, position, deltanet_states, conv_states):
        """Decode while carrying aliased K/V buffers outside the public result."""
        if not USE_GQA_STATEFUL_KV:
            raise RuntimeError(
                "decode_step_stateful requires GQA_STATEFUL_KV=1"
            )
        outputs = self.decode_step(
            input_id,
            position,
            deltanet_states,
            conv_states,
            self.decode_kv_k,
            self.decode_kv_v,
        )
        return outputs[0], outputs[1], outputs[2], outputs[3]

    def set_diagnostic_layers(self, layers):
        selected = tuple(int(layer) for layer in layers)
        if not selected:
            raise ValueError("at least one diagnostic layer is required")
        if selected != tuple(sorted(set(selected))):
            raise ValueError("diagnostic layers must be sorted and unique")
        if selected[0] < 0 or selected[-1] >= D.NUM_LAYERS:
            raise ValueError(
                f"diagnostic layers must be in [0, {D.NUM_LAYERS - 1}]"
            )
        self._diagnostic_layers = selected

    def decode_step_diagnostics(
        self, input_id, position, deltanet_states, conv_states
    ):
        """Full decode with selected MoE and residual stages as graph outputs."""
        if not USE_GQA_STATEFUL_KV:
            raise RuntimeError(
                "decode_step_diagnostics requires GQA_STATEFUL_KV=1"
            )
        if not (USE_MOE_FUSED_W8 or USE_MOE_OFFICIAL_FP8_REFERENCE):
            raise RuntimeError(
                "decode_step_diagnostics requires a fused-W8 candidate or "
                "official-FP8 reference"
            )
        if not self._diagnostic_layers:
            raise RuntimeError("call set_diagnostic_layers before compiling")
        if not USE_DECODE_SHARDED_LM_HEAD:
            raise RuntimeError(
                "decode_step_diagnostics requires DECODE_SHARDED_LM_HEAD=1"
            )

        hidden = F.embedding(input_id, self.embed).unsqueeze(1)
        if USE_MOE_W8_RESIDUAL_FP32:
            hidden = hidden.float()
        if USE_DN_DIRECT_STATE_OUT:
            dn_states = torch.empty_like(deltanet_states)
            cv_states = torch.empty_like(conv_states)
        else:
            dn_states = deltanet_states.clone()
            cv_states = conv_states.clone()
        kv_k = self.decode_kv_k
        kv_v = self.decode_kv_v
        cos = self.rope_cos.squeeze(0).squeeze(0).index_select(
            0, position.unsqueeze(0)
        ).unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin.squeeze(0).squeeze(0).index_select(
            0, position.unsqueeze(0)
        ).unsqueeze(0).unsqueeze(0)

        layer_inputs = []
        attention_outputs = []
        post_attention_hiddens = []
        moe_inputs = []
        router_top_ids = []
        router_top_weights = []
        local_routed_outputs = []
        global_routed_outputs = []
        shared_outputs = []
        layer_outputs = []

        for i in range(D.NUM_LAYERS):
            selected = i in self._diagnostic_layers
            if selected:
                layer_inputs.append(hidden.reshape(self.batch_size, D.HIDDEN))
            normed = decode_rms_norm(
                hidden, getattr(self, f"l{i}_input_norm")
            )
            if D.layer_type(i) == "deltanet":
                if USE_DN_DIRECT_STATE_OUT:
                    attention = self._deltanet_layer(
                        i,
                        normed,
                        deltanet_states,
                        conv_states,
                        dn_states,
                        cv_states,
                    )
                else:
                    attention = self._deltanet_layer(
                        i, normed, dn_states, cv_states
                    )
            else:
                attention = self._gqa_layer(
                    i, normed, cos, sin, position, kv_k, kv_v
                )
            hidden = hidden + attention
            moe_input = decode_rms_norm(
                hidden, getattr(self, f"l{i}_post_norm")
            )

            if selected:
                x2d = moe_input.reshape(-1, D.HIDDEN)
                (
                    top_ids,
                    top_weights,
                    local_routed,
                    global_routed,
                    shared,
                ) = self._moe_diagnostic_parts(i, x2d)
                moe_output = (global_routed + shared).reshape_as(
                    hidden
                ).to(moe_input.dtype)
                attention_outputs.append(
                    attention.reshape(self.batch_size, D.HIDDEN)
                )
                post_attention_hiddens.append(
                    hidden.reshape(self.batch_size, D.HIDDEN)
                )
                moe_inputs.append(
                    moe_input.reshape(self.batch_size, D.HIDDEN)
                )
                router_top_ids.append(top_ids)
                router_top_weights.append(top_weights)
                local_routed_outputs.append(local_routed)
                global_routed_outputs.append(global_routed)
                shared_outputs.append(shared)
            else:
                moe_output = self._moe(i, moe_input)
            hidden = hidden + moe_output
            if selected:
                layer_outputs.append(
                    hidden.reshape(self.batch_size, D.HIDDEN)
                )

        hidden = rms_norm(hidden, self.final_norm)
        if USE_MOE_W8_RESIDUAL_FP32:
            hidden = hidden.to(torch.bfloat16)
        if D.VOCAB % self.world_size:
            raise RuntimeError(
                "DECODE_SHARDED_LM_HEAD requires vocab size divisible by TP"
            )
        vocab_per_rank = D.VOCAB // self.world_size
        vocab_lo = self.rank * vocab_per_rank
        vocab_hi = vocab_lo + vocab_per_rank
        logits = F.linear(
            hidden, self.lm_head_w[vocab_lo:vocab_hi]
        ).squeeze(1)
        local_max, local_id = logits.max(dim=-1)
        global_max = functional_all_reduce(
            local_max, "max", self.tp_group
        )
        global_id = local_id.to(torch.int32) + vocab_lo
        invalid_id = torch.full_like(global_id, -D.VOCAB)
        neg_winner_id = torch.where(
            local_max == global_max, -global_id, invalid_id
        )
        next_id = -functional_all_reduce(
            neg_winner_id, "max", self.tp_group
        )
        diagnostics = (
            torch.stack(layer_inputs),
            torch.stack(attention_outputs),
            torch.stack(post_attention_hiddens),
            torch.stack(moe_inputs),
            torch.stack(router_top_ids),
            torch.stack(router_top_weights),
            torch.stack(local_routed_outputs),
            torch.stack(global_routed_outputs),
            torch.stack(shared_outputs),
            torch.stack(layer_outputs),
        )
        return (
            logits,
            next_id.to(torch.long),
            dn_states,
            cv_states,
            *diagnostics,
        )

    def _run_layers(self, lo, hi, hidden, cos, sin, position,
                    dn_states, cv_states, kv_k, kv_v):
        """Decode layers [lo, hi). Compiled per-segment to keep each NEFF's
        collective count under the neuronx-cc PGTiling limit (the full 40-layer
        graph with ~3 all-reduce/layer trips a compiler tiling assertion;
        20-layer segments compile cleanly)."""
        for i in range(lo, hi):
            normed = decode_rms_norm(
                hidden, getattr(self, f"l{i}_input_norm")
            )
            if D.layer_type(i) == "deltanet":
                hidden = hidden + self._deltanet_layer(i, normed, dn_states, cv_states)
            else:
                hidden = hidden + self._gqa_layer(i, normed, cos, sin, position, kv_k, kv_v)
            normed = decode_rms_norm(
                hidden, getattr(self, f"l{i}_post_norm")
            )
            hidden = hidden + self._moe(i, normed)
        return hidden

    def _run_layers_direct_state(
        self, lo, hi, hidden, cos, sin, position,
        dn_states_in, cv_states_in, dn_states_out, cv_states_out, kv_k, kv_v,
    ):
        """Decode layers while keeping recurrent inputs and outputs disjoint."""
        for i in range(lo, hi):
            normed = decode_rms_norm(
                hidden, getattr(self, f"l{i}_input_norm")
            )
            if D.layer_type(i) == "deltanet":
                hidden = hidden + self._deltanet_layer(
                    i, normed, dn_states_in, cv_states_in,
                    dn_states_out, cv_states_out,
                )
            else:
                hidden = hidden + self._gqa_layer(
                    i, normed, cos, sin, position, kv_k, kv_v
                )
            normed = decode_rms_norm(
                hidden, getattr(self, f"l{i}_post_norm")
            )
            hidden = hidden + self._moe(i, normed)
        return hidden

    def setup_segments(self, n_splits, compile_each=True):
        """Split the layer range into n_splits contiguous segments. When
        compile_each, each segment's _run_layers is wrapped in
        torch.compile(backend='neuron', fullgraph=True) so it compiles to its
        own NEFF — keeping per-graph collective count under the PGTiling limit.
        n_splits chosen so each segment <= ~20 layers (20 compiles; 40 doesn't)."""
        NL = D.NUM_LAYERS
        bounds = [round(k * NL / n_splits) for k in range(n_splits + 1)]
        segs = []
        for k in range(n_splits):
            lo, hi = bounds[k], bounds[k + 1]
            if lo == hi:
                continue
            fn = (
                self._run_layers_direct_state
                if USE_DN_DIRECT_STATE_OUT
                else self._run_layers
            )
            if compile_each:
                fn = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
            segs.append((lo, hi, fn))
        self._segments = segs
        return [(lo, hi) for (lo, hi, _) in segs]

    # ── DeltaNet decode layer (pure-torch recurrence, per batch row) ──
    def _deltanet_layer(
        self, i, x, dn_states, cv_states,
        dn_states_out=None, cv_states_out=None,
    ):
        di = D.deltanet_index(i)
        B = self.batch_size
        td = self.td
        KH, VH = td["dn_k_heads"], td["dn_v_heads"]      # 4, 8 @ TP4
        KD, VD = D.DN_K_DIM, D.DN_V_DIM                   # 128, 128
        key_dim = KH * KD                                # 512 @ TP4
        val_dim = VH * VD                                # 1024 @ TP4
        qkv_dim = 2 * key_dim + val_dim                  # 2048 @ TP4

        conv_w = getattr(self, f"l{i}_dn_conv_w")        # [qkv_dim,1,4]
        A_log = getattr(self, f"l{i}_dn_A_log").float()  # [VH]
        dt_bias = getattr(self, f"l{i}_dn_dt_bias").float()
        norm_w = getattr(self, f"l{i}_dn_norm")          # [VD]

        x_2d = x.squeeze(1)                              # [B,H]
        mixed_qkv = self._lin(f"l{i}_dn_qkv", x_2d)      # [B,qkv_dim]
        a_out = self._lin(f"l{i}_dn_a", x_2d).float()    # [B,VH]
        b_out = self._lin(f"l{i}_dn_b", x_2d).float()    # [B,VH]
        z = self._lin(f"l{i}_dn_z", x_2d).reshape(B, VH, VD)   # [B,VH,VD]

        conv_w2 = conv_w.squeeze(1).float()              # [qkv_dim,4]

        if USE_DN_NKI:
            # NKI-kernel path: ONE custom call/layer folds conv+silu + L2norm +
            # Q-scale + gates + recurrence + RMSNormGated, opaque to neuronx-cc's
            # tiler (the pure-torch einsum recurrence below trips PGTiling at
            # >~20 layers; the kernel is what lets the 27B compile 64 layers).
            cb = torch.zeros(qkv_dim, dtype=torch.float32, device=x_2d.device)
            if USE_DN_DIRECT_STATE_OUT:
                state_in = dn_states[di].reshape(B * VH * KD, VD)
            else:
                state_in = dn_states[di].reshape(B * VH * KD, VD).float()
            qkv_in = mixed_qkv.to(torch.bfloat16).reshape(B * qkv_dim)   # [B*qkv_dim]
            conv_in = cv_states[di].reshape(B * qkv_dim, D.DN_CONV_KERNEL - 1)
            a_in = a_out.reshape(B * VH)
            b_in = b_out.reshape(B * VH)
            z_in = z.to(torch.bfloat16).reshape(B * VH, VD)
            if USE_DN_DIRECT_STATE_OUT:
                state_out = dn_states_out[di].reshape(B * VH * KD, VD)
                conv_out = cv_states_out[di].reshape(
                    B * qkv_dim, D.DN_CONV_KERNEL - 1
                )
                new_state, new_cs, attn_flat = (
                    torch.ops.deltanet35b.full_batched_direct(
                        state_in, qkv_in, conv_in, conv_w2, cb,
                        a_in, b_in, z_in, A_log, dt_bias, norm_w.float(),
                        state_out, conv_out,
                    )
                )
                # Keep the custom op's mutated outputs in explicit graph
                # dataflow. Relying only on alias mutation of these parent
                # slices can leave middle-layer state outputs unwritten.
                dn_states_out[di] = new_state.reshape(
                    B, VH * KD, VD
                ).to(dn_states_out.dtype)
                cv_states_out[di] = new_cs.reshape(
                    B, qkv_dim, D.DN_CONV_KERNEL - 1
                ).to(cv_states_out.dtype)
            else:
                new_state, new_cs, attn_flat = torch.ops.deltanet35b.full_batched(
                    state_in, qkv_in, conv_in, conv_w2, cb,
                    a_in, b_in, z_in, A_log, dt_bias, norm_w.float(),
                )
                dn_states[di] = new_state.reshape(
                    B, VH * KD, VD
                ).to(dn_states.dtype)
                cv_states[di] = new_cs.reshape(
                    B, qkv_dim, D.DN_CONV_KERNEL - 1
                ).to(cv_states.dtype)
            gated = attn_flat.reshape(B, VH * VD)                        # already gated
            out = self._lin(f"l{i}_dn_out", gated)
            out = functional_all_reduce(out, "sum", self.tp_group)
            return out.unsqueeze(1)

        outs = []
        for bi in range(B):
            # causal depthwise conv over [history(3) + new(1)] -> last timestep
            hist = cv_states[di, bi].float()             # [qkv_dim,3]
            cur = mixed_qkv[bi].float()                  # [qkv_dim]
            window = torch.cat([hist, cur.unsqueeze(-1)], dim=-1)   # [qkv_dim,4]
            conv_o = (window * conv_w2).sum(-1)          # depthwise, k=4 -> [qkv_dim]
            conv_o = F.silu(conv_o)
            # update conv state: last 3 raw inputs
            cv_states[di, bi] = window[:, 1:].to(cv_states.dtype)

            q = conv_o[:key_dim].reshape(KH, KD)
            k = conv_o[key_dim:2 * key_dim].reshape(KH, KD)
            v = conv_o[2 * key_dim:].reshape(VH, VD)
            # expand k-heads -> v-heads
            grp = VH // KH
            q = q.repeat_interleave(grp, dim=0)          # [VH,KD]
            k = k.repeat_interleave(grp, dim=0)          # [VH,KD]

            beta = b_out[bi].sigmoid()                   # [VH]
            g = -A_log.exp() * F.softplus(a_out[bi] + dt_bias)   # [VH]

            state_in = dn_states[di, bi].reshape(VH, KD, VD)
            attn, new_state = deltanet_recurrent_step(
                state_in, q, k, v, g, beta, use_qk_l2norm=True)   # [VH,VD]
            dn_states[di, bi] = new_state.reshape(VH * KD, VD).to(dn_states.dtype)
            outs.append(attn)

        attn_out = torch.stack(outs, 0)                  # [B,VH,VD]
        gated = rms_norm_gated(attn_out.to(x.dtype), z, norm_w)   # [B,VH,VD]
        out = self._lin(f"l{i}_dn_out", gated.reshape(B, val_dim))
        out = functional_all_reduce(out, "sum", self.tp_group)
        return out.unsqueeze(1)                          # [B,1,H]

    # ── GQA decode layer (KV heads replicated across cores) ──
    def _gqa_layer(self, i, x, cos, sin, position, kv_k, kv_v):
        gi = D.gqa_index(i)
        B = self.batch_size
        td = self.td
        QH = td["gqa_q_heads"]                           # query heads/core
        HD = D.GQA_HEAD_DIM
        NKV = self.nkv                                   # KV heads/core (>=1)
        GRP = QH // NKV                                  # query heads per KV head

        q_norm_w = getattr(self, f"l{i}_gqa_q_norm")
        k_norm_w = getattr(self, f"l{i}_gqa_k_norm")
        x_2d = x.squeeze(1)                              # [B,H]

        q_out = self._lin(f"l{i}_gqa_q", x_2d).reshape(B, QH, HD * 2)
        query, gate = q_out.chunk(2, dim=-1)             # [B,QH,HD] each
        gate = gate.reshape(B, QH * HD)
        key = self._lin(f"l{i}_gqa_k", x_2d).reshape(B, NKV, HD)
        value = self._lin(f"l{i}_gqa_v", x_2d).reshape(B, NKV, HD)

        # k-side norm + rope stay in torch (single head, cheap). For GQATAIL the
        # q-side norm+rope are done INSIDE the kernel (needs PRE-NORM query), so
        # skip them here; otherwise norm+rope q in torch as usual.
        key = rms_norm(key, k_norm_w)                    # [B,NKV,HD]
        key = key.unsqueeze(2)                           # [B,NKV,1,HD]
        if USE_GQA_TAIL:
            _, key = apply_rope(key, key, cos, sin)      # rope key only
        else:
            query = rms_norm(query, q_norm_w)            # [B,QH,HD]
            query = query.unsqueeze(2)                   # [B,QH,1,HD]
            query, key = apply_rope(query, key, cos, sin)
            query = query.to(x.dtype).squeeze(2).reshape(B, QH, HD)
        key_b = key.squeeze(2).reshape(B, NKV, HD).to(kv_k.dtype)
        value_b = value.reshape(B, NKV, HD).to(kv_v.dtype)

        # KV cache: kv_k/v[gi] = [B, NKV, max_seq, HD].
        if not USE_GQA_STATEFUL_KV:
            pos_idx = position.reshape(1, 1, 1, 1).expand(B, NKV, 1, HD)
            kv_k[gi].scatter_(2, pos_idx, key_b.unsqueeze(2))
            kv_v[gi].scatter_(2, pos_idx, value_b.unsqueeze(2))
        cached_k = kv_k[gi]                               # [B,NKV,S,HD]
        cached_v = kv_v[gi]

        pos_range = torch.arange(self.max_seq_len, device=x.device)
        if USE_GQA_STATEFUL_KV:
            mask = pos_range < position
        else:
            mask = pos_range <= position

        if USE_GQA_TAIL:
            # ONE custom call folds q RMSNorm + partial RoPE + scaled scores +
            # masked softmax + weighted-V + output gate. nkv=1 at TP=4 (KV
            # replicated), so all QH query heads attend to the single cached
            # head — the kernel's [B*S,HD] layout. Pass PRE-NORM query, raw gate.
            S = self.max_seq_len
            if USE_GQA_STATEFUL_KV:
                cache_k_arg = kv_k.reshape(D.NUM_GQA * B * S, HD)
                cache_v_arg = kv_v.reshape(D.NUM_GQA * B * S, HD)
            else:
                cache_k_arg = cached_k.reshape(B * S, HD).float()
                cache_v_arg = cached_v.reshape(B * S, HD).float()
            args = (
                query.reshape(B * QH, HD).float(),
                gate.reshape(B * QH, HD).float(),
                q_norm_w.reshape(1, HD).float(),
                cos.reshape(1, D.ROPE_DIM).float(),
                sin.reshape(1, D.ROPE_DIM).float(),
                cache_k_arg,
                cache_v_arg,
                mask.reshape(1, S).float(),
            )
            if USE_GQA_STATEFUL_KV:
                attn_flat = torch.ops.gqa35b.tail_stateful(
                    *args,
                    key_b.reshape(B, HD),
                    value_b.reshape(B, HD),
                    position.reshape(1, 1).to(torch.int32),
                    gi,
                )
            else:
                attn_flat = torch.ops.gqa35b.tail(*args)
            attn_out = attn_flat.reshape(B, QH * HD).to(x.dtype)
        else:
            # Grouped attention: GRP query heads share each KV head.
            qd = query.reshape(B, NKV, GRP, HD).to(cached_k.dtype)        # [B,NKV,GRP,HD]
            scores = torch.matmul(qd, cached_k.transpose(2, 3)) / math.sqrt(HD)  # [B,NKV,GRP,S]
            scores = scores + (1.0 - mask.to(scores.dtype)).reshape(1, 1, 1, -1) * (-1e9)
            attn_w = F.softmax(scores.float(), dim=-1).to(cached_v.dtype)
            attn_out = torch.matmul(attn_w, cached_v)                     # [B,NKV,GRP,HD]
            attn_out = attn_out.reshape(B, QH * HD)
            attn_out = attn_out * torch.sigmoid(gate)
        out = self._lin(f"l{i}_gqa_o", attn_out)
        out = functional_all_reduce(out, "sum", self.tp_group)
        return out.unsqueeze(1)                          # [B,1,H]

    # ── Prefill (eager; homogeneous batch) ──
    def prefill(self, input_ids, deltanet_states, conv_states, kv_cache_k, kv_cache_v):
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        B, S = input_ids.shape
        hidden = F.embedding(input_ids, self.embed).float()                # [B,S,H]
        dn_states = deltanet_states.clone()
        cv_states = conv_states.clone()
        kv_k = kv_cache_k.clone()
        kv_v = kv_cache_v.clone()

        for i in range(D.NUM_LAYERS):
            normed = rms_norm(hidden, getattr(self, f"l{i}_input_norm"))
            if D.layer_type(i) == "deltanet":
                hidden = hidden + self._deltanet_prefill(i, normed, dn_states, cv_states)
            else:
                hidden = hidden + self._gqa_prefill(i, normed, S, kv_k, kv_v)
            normed = rms_norm(hidden, getattr(self, f"l{i}_post_norm"))
            hidden = hidden + self._moe(i, normed)

        hidden = rms_norm(hidden, self.final_norm)
        logits = self._lin("lm_head_w", hidden[:, -1:, :])
        return logits.squeeze(1), dn_states, cv_states, kv_k, kv_v

    def _deltanet_prefill(self, i, x, dn_states, cv_states, valid_len=None):
        di = D.deltanet_index(i)
        td = self.td
        KH, VH = td["dn_k_heads"], td["dn_v_heads"]
        KD, VD = D.DN_K_DIM, D.DN_V_DIM
        key_dim = KH * KD
        val_dim = VH * VD
        qkv_dim = 2 * key_dim + val_dim
        B, S = x.shape[:2]

        conv_w = getattr(self, f"l{i}_dn_conv_w")
        A_log = getattr(self, f"l{i}_dn_A_log").float()
        dt_bias = getattr(self, f"l{i}_dn_dt_bias").float()
        norm_w = getattr(self, f"l{i}_dn_norm")
        x_2d = x.reshape(B * S, D.HIDDEN)                 # [B*S,H]

        mixed_qkv = self._lin(f"l{i}_dn_qkv", x_2d).reshape(B, S, qkv_dim)
        mqf = mixed_qkv.permute(0, 2, 1).float()           # [B,qkv_dim,S]
        conv_input = torch.cat([cv_states[di].float(), mqf], dim=-1)
        conv_out = F.conv1d(conv_input, conv_w.float(), groups=qkv_dim)
        cv_states[di] = mqf[:, :, -3:].to(cv_states.dtype)
        mixed_qkv = F.silu(conv_out.permute(0, 2, 1))       # [B,S,qkv_dim]

        q = mixed_qkv[:, :, :key_dim].reshape(B, S, KH, KD)
        k = mixed_qkv[:, :, key_dim:2 * key_dim].reshape(B, S, KH, KD)
        v = mixed_qkv[:, :, 2 * key_dim:].reshape(B, S, VH, VD)
        grp = VH // KH
        q = q.repeat_interleave(grp, dim=2)               # [B,S,VH,KD]
        k = k.repeat_interleave(grp, dim=2)

        a_out = self._lin(f"l{i}_dn_a", x_2d).reshape(B, S, VH).float()
        b_out = self._lin(f"l{i}_dn_b", x_2d).reshape(B, S, VH)
        beta = b_out.sigmoid()
        g = -A_log.exp() * F.softplus(a_out + dt_bias)

        # PAD MASKING (bucketed prefill): the final chunk is padded to `chunk` with
        # id=0 tokens. DeltaNet is a recurrence with no valid-length mask, so pad
        # rows would corrupt the carried state that decode reads. Zero beta (→ the
        # delta-rule k_beta/v_beta updates vanish = identity recurrence step) and g
        # (→ decay 1.0, harmless once beta=0) for pad rows. Same fix class as the
        # vLLM chunked-prefill pad-token bug. The reusable compiled path passes
        # valid_len as a runtime scalar so a partial final bucket does not create
        # another graph specialization. The attribute remains for static eager
        # controls.
        if valid_len is not None:
            positions = torch.arange(S, dtype=torch.int32, device=beta.device)
            pad = (positions < valid_len.reshape(-1)[0]).to(beta.dtype)[None, :, None]
            beta = beta * pad
            g = g * pad
        else:
            vlen = getattr(self, "_dn_valid_len", None)
        if valid_len is None and vlen is not None and vlen < S:
            pad = torch.zeros(1, S, VH, dtype=beta.dtype, device=beta.device)
            pad[:, :vlen] = 1.0
            beta = beta * pad
            g = g * pad

        if USE_DN_CHUNK_NKI:
            # ONE NKI custom call/layer — compilable under backend="neuron" (the
            # pure-torch loop below trips NCC_IMGN901 under torch.compile). Kernel
            # is head-major (row h*S+t), L2-normalizes q,k + applies 1/sqrt(K) q-scale
            # INTERNALLY, so pass RAW q,k. Takes/returns state -> composes with buckets.
            q_hm = q.float().permute(0, 2, 1, 3).reshape(B * VH * S, KD).contiguous()
            k_hm = k.float().permute(0, 2, 1, 3).reshape(B * VH * S, KD).contiguous()
            v_hm = v.float().permute(0, 2, 1, 3).reshape(B * VH * S, VD).contiguous()
            g_hm = g.permute(0, 2, 1).reshape(B * VH * S, 1).contiguous()
            beta_hm = beta.float().permute(0, 2, 1).reshape(B * VH * S, 1).contiguous()
            state_in = dn_states[di].float()                  # [B,VH*KD,VD]
            capture_here = (
                DN_CAPTURE_DIR
                and i == DN_CAPTURE_LAYER
                and getattr(self, "_prefill_chunk_index", -1) == DN_CAPTURE_CHUNK
                and not getattr(self, "_dn_capture_done", False)
            )
            if capture_here:
                os.makedirs(DN_CAPTURE_DIR, exist_ok=True)
                torch.save(
                    {
                        "layer": i,
                        "chunk_index": self._prefill_chunk_index,
                        "state": state_in.detach().cpu(),
                        "query": q_hm.detach().cpu(),
                        "key": k_hm.detach().cpu(),
                        "value": v_hm.detach().cpu(),
                        "g": g_hm.detach().cpu(),
                        "beta": beta_hm.detach().cpu(),
                        "m_incl": self.chunk_m_incl.detach().cpu(),
                        "m_strict": self.chunk_m_strict.detach().cpu(),
                        "eye": self.chunk_eye.detach().cpu(),
                    },
                    os.path.join(
                        DN_CAPTURE_DIR,
                        f"dn_layer{i}_chunk{self._prefill_chunk_index}"
                        f"_rank{self.rank}.pt",
                    ),
                )
                self._dn_capture_done = True
            out_hm, new_state = torch.ops.deltanet35b.chunked_prefill(
                state_in, q_hm, k_hm, v_hm, g_hm, beta_hm,
                self.chunk_m_incl, self.chunk_m_strict, self.chunk_eye)
            dn_states[di] = new_state.to(dn_states.dtype)
            attn_out = out_hm.reshape(B, VH, S, VD).permute(0, 2, 1, 3)
        else:
            from chunked_prefill import neuron_chunk_gated_delta_rule
            q_in = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
            k_in = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
            v_in = v.float()
            g_in = g
            beta_in = beta.float()
            init_state = dn_states[di].float().reshape(B, VH, KD, VD)
            attn_4d, final_state = neuron_chunk_gated_delta_rule(
                q_in, k_in, v_in, g=g_in, beta=beta_in, chunk_size=64,
                initial_state=init_state, output_final_state=True,
                use_qk_l2norm_in_kernel=False)
            dn_states[di] = final_state.reshape(B, VH * KD, VD).to(dn_states.dtype)
            attn_out = attn_4d                              # [B,S,VH,VD]

        z = self._lin(f"l{i}_dn_z", x_2d).reshape(B, S, VH, VD)
        gated = rms_norm_gated(attn_out.to(x.dtype), z, norm_w)
        out = self._lin(f"l{i}_dn_out", gated.reshape(B * S, val_dim)).reshape(B, S, D.HIDDEN)
        out = functional_all_reduce(out, "sum", self.tp_group)
        return out

    def _gqa_prefill(self, i, x, S, kv_k, kv_v):
        gi = D.gqa_index(i)
        td = self.td
        QH = td["gqa_q_heads"]
        HD = D.GQA_HEAD_DIM
        NKV = self.nkv
        GRP = QH // NKV
        q_norm_w = getattr(self, f"l{i}_gqa_q_norm")
        k_norm_w = getattr(self, f"l{i}_gqa_k_norm")
        B = x.shape[0]
        x_2d = x.reshape(B * S, D.HIDDEN)

        q_out = self._lin(f"l{i}_gqa_q", x_2d).reshape(B, S, QH, HD * 2)
        query, gate = q_out.chunk(2, dim=-1)
        gate = gate.reshape(B, S, QH * HD)
        key = self._lin(f"l{i}_gqa_k", x_2d).reshape(B, S, NKV, HD)
        value = self._lin(f"l{i}_gqa_v", x_2d).reshape(B, S, NKV, HD)
        query = rms_norm(query, q_norm_w)
        key = rms_norm(key, k_norm_w)

        positions = torch.arange(S, device=x.device)
        cos = self.rope_cos.squeeze(0).squeeze(0)[positions].reshape(1, S, 1, D.ROPE_DIM)
        sin = self.rope_sin.squeeze(0).squeeze(0)[positions].reshape(1, S, 1, D.ROPE_DIM)
        query, key = apply_rope(query, key, cos, sin)
        query = query.to(x.dtype); key = key.to(x.dtype)

        kv_k[gi, :, :, :S] = key.permute(0, 2, 1, 3).to(kv_k.dtype)
        kv_v[gi, :, :, :S] = value.permute(0, 2, 1, 3).to(kv_v.dtype)

        if USE_GQA_FLASH_PREFILL:
            # Flash causal-attention kernel. NKV=1/core, so all QH queries attend the
            # single KV head. q [QH,S,HD], k/v [S,HD]. Kernel returns [QH,S,HD].
            if B != 1:
                raise RuntimeError("GQA_FLASH_PREFILL does not support batched prefill; use GQA_CTE_PREFILL")
            q_f = query[0].permute(1, 0, 2).contiguous()
            k_f = key[0].reshape(S, HD).contiguous()
            v_f = value[0].reshape(S, HD).contiguous()
            attn_out = torch.ops.gqa35b.flash_prefill(q_f, k_f, v_f)       # [QH,S,HD]
            attn_out = attn_out.permute(1, 0, 2).reshape(1, S, QH * HD)
        else:
            # grouped causal attention (pure-torch, full [S,S] — OOMs at large S)
            q_g = query.reshape(B, S, NKV, GRP, HD).permute(0, 2, 3, 1, 4)
            k_g = key.permute(0, 2, 1, 3)
            scores = torch.matmul(q_g, k_g.unsqueeze(2).transpose(3, 4)) / math.sqrt(HD)
            causal = torch.tril(torch.ones(S, S, device=x.device, dtype=scores.dtype))
            scores = scores + (1.0 - causal) * (-1e9)
            v_g = value.permute(0, 2, 1, 3)
            attn_w = F.softmax(scores.float(), dim=-1).to(v_g.dtype)
            attn_out = torch.matmul(attn_w, v_g.unsqueeze(2))
            attn_out = attn_out.permute(0, 3, 1, 2, 4).reshape(B, S, QH * HD)
        attn_out = attn_out * torch.sigmoid(gate)
        out = self._lin(f"l{i}_gqa_o", attn_out.reshape(B * S, QH * HD)).reshape(B, S, D.HIDDEN)
        out = functional_all_reduce(out, "sum", self.tp_group)
        return out

    # ── Bucketed/chunked prefill ──
    def _gqa_prefill_chunk(self, i, x, q_base, chunk, kv_k, kv_v):
        """GQA for one prompt chunk of `chunk` tokens starting at global row
        q_base. The established path receives a Python integer; the reusable
        compiled path receives a runtime int32 [1,1] tensor and performs RoPE
        lookup plus cache writes in an aliased NKI custom op."""
        gi = D.gqa_index(i)
        td = self.td
        QH = td["gqa_q_heads"]
        HD = D.GQA_HEAD_DIM
        NKV = self.nkv                                   # =1 at TP4 (KV replicated)
        q_norm_w = getattr(self, f"l{i}_gqa_q_norm")
        k_norm_w = getattr(self, f"l{i}_gqa_k_norm")
        B = x.shape[0]
        x_2d = x.reshape(B * chunk, D.HIDDEN)

        q_out = self._lin(f"l{i}_gqa_q", x_2d).reshape(B, chunk, QH, HD * 2)
        query, gate = q_out.chunk(2, dim=-1)
        gate = gate.reshape(B, chunk, QH * HD)
        key = self._lin(f"l{i}_gqa_k", x_2d).reshape(B, chunk, NKV, HD)
        value = self._lin(f"l{i}_gqa_v", x_2d).reshape(B, chunk, NKV, HD)
        query = rms_norm(query, q_norm_w)
        key = rms_norm(key, k_norm_w)

        if USE_GQA_DYNAMIC_ROPE_KV:
            cache_k = kv_k[gi, :, 0].reshape(B * self.max_seq_len, HD)
            cache_v = kv_v[gi, :, 0].reshape(B * self.max_seq_len, HD)
            q_f, k_active, k_filled, v_filled = torch.ops.gqa35b.rope_kv_dynamic(
                query.permute(0, 2, 1, 3).contiguous(),
                key[:, :, 0].contiguous(),
                value[:, :, 0].contiguous(),
                self.rope_cos.squeeze(0).squeeze(0),
                self.rope_sin.squeeze(0).squeeze(0),
                cache_k,
                cache_v,
                q_base,
            )
            k_filled = k_filled.reshape(B, self.max_seq_len, HD)
            v_filled = v_filled.reshape(B, self.max_seq_len, HD)
            qb = q_base.float()
        else:
            # Static offset path retained for eager controls and old compile caches.
            positions = torch.arange(q_base, q_base + chunk, device=x.device)
            cos = self.rope_cos.squeeze(0).squeeze(0)[positions].reshape(1, chunk, 1, D.ROPE_DIM)
            sin = self.rope_sin.squeeze(0).squeeze(0)[positions].reshape(1, chunk, 1, D.ROPE_DIM)
            query, key = apply_rope(query, key, cos, sin)
            query = query.to(x.dtype); key = key.to(x.dtype)

            # scatter_ was observed to miscompile here; index_copy_ is safe when
            # positions are static.
            key_t = key.permute(0, 2, 1, 3).to(kv_k.dtype)
            val_t = value.permute(0, 2, 1, 3).to(kv_v.dtype)
            kv_k[gi].index_copy_(2, positions, key_t)
            kv_v[gi].index_copy_(2, positions, val_t)
            q_f = query.permute(0, 2, 1, 3).reshape(B * QH, chunk, HD).float().contiguous()
            k_filled = kv_k[gi, :, 0]
            v_filled = kv_v[gi, :, 0]
            qb = torch.full((1, 1), float(q_base), dtype=torch.float32, device=x.device)

        if USE_GQA_CTE_PREFILL:
            attn_out = torch.ops.gqa35b.cte_prefill(
                (q_f.reshape(B * QH, chunk, HD) * (1.0 / math.sqrt(HD))).to(torch.bfloat16),
                k_active.transpose(1, 2),
                value[:, :, 0].to(torch.bfloat16),
                k_filled.transpose(1, 2),
                v_filled,
                q_base.reshape(1),
            )
            attn_out = attn_out.reshape(B, QH, chunk, HD).permute(0, 2, 1, 3).reshape(B, chunk, QH * HD)
        elif USE_GQA_FLASH_PREFILL:
            if B != 1:
                raise RuntimeError("GQA_FLASH_PREFILL does not support batched prefill; use GQA_CTE_PREFILL")
            # Kernel computes in f32 (dma_transpose dst is f32) — cast the (bf16) KV
            # buffer + query to f32 so the kernel's dma_transpose src/dst dtypes match.
            k_f = k_filled.float()
            v_f = v_filled.float()
            attn_out = torch.ops.gqa35b.flash_prefill_chunk(q_f, k_f, v_f, qb)  # [QH,chunk,HD]
            attn_out = attn_out.permute(1, 0, 2).reshape(1, chunk, QH * HD)
        else:
            # Pure-torch grouped causal attention over the KV buffer prefix
            # [0 : q_base+chunk], with the chunk's queries at global rows q_base..
            # (diagnostic / fallback path — no flash kernel).
            klen = q_base + chunk
            kfull = kv_k[gi, :, 0, :klen].float()
            vfull = kv_v[gi, :, 0, :klen].float()
            qd = query.permute(0, 2, 1, 3).float()
            scores = torch.matmul(qd, kfull.transpose(1, 2).unsqueeze(1)) / math.sqrt(HD)
            qpos = torch.arange(q_base, q_base + chunk, device=x.device).reshape(chunk, 1)
            kpos = torch.arange(klen, device=x.device).reshape(1, klen)
            cmask = (qpos >= kpos)                                # [chunk,klen]
            scores = scores + (~cmask).reshape(1, 1, chunk, klen) * (-1e9)
            attn_w = F.softmax(scores.float(), dim=-1)
            attn_out = torch.matmul(attn_w, vfull.unsqueeze(1))
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, chunk, QH * HD).to(x.dtype)
        attn_out = attn_out * torch.sigmoid(gate)
        out = self._lin(f"l{i}_gqa_o", attn_out.reshape(B * chunk, QH * HD)).reshape(B, chunk, D.HIDDEN)
        out = functional_all_reduce(out, "sum", self.tp_group)
        return out

    def _prefill_chunk_layer_range(
        self, hidden, q_base, valid_len, dn_states, cv_states, kv_k, kv_v,
        start, end
    ):
        """Run one fixed layer range for a prompt chunk."""
        chunk = hidden.shape[1]
        for i in range(start, end):
            normed = rms_norm(hidden, getattr(self, f"l{i}_input_norm"))
            if D.layer_type(i) == "deltanet":
                # DeltaNet prefill already carries state via dn_states/cv_states.
                hidden = hidden + self._deltanet_prefill(
                    i, normed, dn_states, cv_states, valid_len
                )
            else:
                hidden = hidden + self._gqa_prefill_chunk(i, normed, q_base, chunk, kv_k, kv_v)
            normed = rms_norm(hidden, getattr(self, f"l{i}_post_norm"))
            hidden = hidden + self._moe(i, normed)
        return hidden

    def _prefill_chunk_layers(
        self, chunk_ids, q_base, valid_len, dn_states, cv_states, kv_k, kv_v
    ):
        """All layers for one prompt chunk."""
        hidden = F.embedding(chunk_ids, self.embed).float()
        return self._prefill_chunk_layer_range(
            hidden, q_base, valid_len, dn_states, cv_states, kv_k, kv_v,
            0, D.NUM_LAYERS,
        )

    def prefill_bucketed(self, input_ids, deltanet_states, conv_states,
                         kv_cache_k, kv_cache_v, chunk=2048, compile_chunk=True,
                         compile_splits=1):
        """Bucketed prefill: split the prompt into fixed `chunk`-size pieces and
        run each through compiled _prefill_chunk_layers instead of one giant eager
        20k graph. With GQA_DYNAMIC_ROPE_KV, q_base and valid length are runtime
        scalars, allowing one graph set to serve every full or partial bucket.
        The KV buffer must be sized >= padded length. Returns last-token logits
        plus updated states."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        B, S = input_ids.shape
        dn_states = deltanet_states.clone()
        cv_states = conv_states.clone()
        kv_k = kv_cache_k.clone()
        kv_v = kv_cache_v.clone()
        n_chunks = (S + chunk - 1) // chunk
        dev = input_ids.device

        if compile_splits < 1 or compile_splits > D.NUM_LAYERS:
            raise ValueError("compile_splits must be in [1, NUM_LAYERS]")
        if compile_chunk and DN_CAPTURE_DIR:
            raise ValueError("DN_CAPTURE_DIR is supported only with --bucket-compile 0")

        segment_fns = None
        fn = self._prefill_chunk_layers
        if compile_chunk and compile_splits == 1:
            fn = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
        elif compile_chunk:
            step = math.ceil(D.NUM_LAYERS / compile_splits)
            segment_fns = []
            for start in range(0, D.NUM_LAYERS, step):
                end = min(start + step, D.NUM_LAYERS)

                def segment(
                    hidden, q_base, valid_len, dn, cv, kk, vv, s=start, e=end
                ):
                    return self._prefill_chunk_layer_range(
                        hidden, q_base, valid_len, dn, cv, kk, vv, s, e
                    )

                segment_fns.append(torch.compile(
                    segment, backend="neuron", fullgraph=True, dynamic=False
                ))
        elif compile_splits > 1:
            step = math.ceil(D.NUM_LAYERS / compile_splits)
            segment_fns = []
            for start in range(0, D.NUM_LAYERS, step):
                end = min(start + step, D.NUM_LAYERS)

                def segment(
                    hidden, q_base, valid_len, dn, cv, kk, vv, s=start, e=end
                ):
                    return self._prefill_chunk_layer_range(
                        hidden, q_base, valid_len, dn, cv, kk, vv, s, e
                    )

                segment_fns.append(segment)

        last_hidden = None
        last_valid = 0
        trace_finite = os.environ.get("PREFILL_TRACE_FINITE", "0") == "1"

        def report_finite(chunk_idx, segment_idx):
            if not trace_finite or self.rank != 0:
                return
            tensors = {
                "hidden": last_hidden,
                "dn": dn_states,
                "conv": cv_states,
            }
            status = []
            for name, tensor in tensors.items():
                host = tensor.detach().cpu().float()
                finite = torch.isfinite(host)
                status.append(
                    f"{name}={bool(finite.all())}"
                    f"/max={float(host[finite].abs().max()) if bool(finite.any()) else float('nan'):.4e}"
                )
            print(
                f"  PREFILL finite chunk={chunk_idx} segment={segment_idx}: "
                + " ".join(status),
                flush=True,
            )

        for c in range(n_chunks):
            self._prefill_chunk_index = c
            cs = c * chunk
            ce = min(cs + chunk, S)
            csz = ce - cs
            # pad the final chunk up to `chunk` (padded rows are masked out of
            # attention by the causal mask and never read as the final token).
            ids = torch.zeros(B, chunk, dtype=input_ids.dtype, device=dev)
            ids[:, :csz] = input_ids[:, cs:ce]
            # Tell DeltaNet how many rows are real so it zeros beta/g for the pad
            # tail (only the final chunk is padded). Prevents pad tokens corrupting
            # the recurrent state that decode inherits.
            if USE_GQA_DYNAMIC_ROPE_KV:
                self._dn_valid_len = None
                q_base_arg = torch.full(
                    (1, 1), cs, dtype=torch.int32, device=dev
                )
                valid_len_arg = torch.full(
                    (1, 1), csz, dtype=torch.int32, device=dev
                )
            else:
                self._dn_valid_len = csz if csz < chunk else None
                q_base_arg = cs
                valid_len_arg = None
            if segment_fns is None:
                last_hidden = fn(
                    ids, q_base_arg, valid_len_arg,
                    dn_states, cv_states, kv_k, kv_v
                )
                report_finite(c, 0)
            else:
                last_hidden = F.embedding(ids, self.embed).float()
                for segment_idx, segment_fn in enumerate(segment_fns):
                    last_hidden = segment_fn(
                        last_hidden, q_base_arg, valid_len_arg,
                        dn_states, cv_states, kv_k, kv_v
                    )
                    report_finite(c, segment_idx)
            last_valid = csz
        self._dn_valid_len = None
        self._prefill_chunk_index = -1

        hidden = rms_norm(last_hidden, self.final_norm)
        logits = self._lin("lm_head_w", hidden[:, last_valid - 1:last_valid, :])
        return logits.squeeze(1), dn_states, cv_states, kv_k, kv_v


# ─── Weight loading from the HF checkpoint (per-rank TP shard) ────────────────
def load_sharded_weights(
    ckpt,
    rank,
    world_size,
    num_layers=None,
    expert_ckpt=None,
    moe_w8_mode=None,
    official_fp8_reference=None,
):
    """Load + TP-shard the 35B-A3B weights for one rank into the per-core dict.

    Sharding (TP=world_size):
      - DeltaNet in_proj_qkv: group-aware [q|k|v] colwise (q,k by key heads; v by
        value heads). conv1d sharded to match. a/b/z colwise by value heads.
        out_proj rowwise. A_log/dt_bias by value heads.
      - GQA q_proj colwise by Q heads (each head = 2*HD: query|gate). k/v_proj:
        2 KV heads REPLICATED across world_size cores -> each core gets the KV
        head assigned to rank // (world_size//2). o_proj rowwise.
      - MoE experts are expert-parallel by default. MOE_DECODE_TP instead keeps
        all expert IDs and shards each expert's intermediate width. Router and
        shared expert are replicated in both layouts.
    """
    from st_reader import SafeReader, build_weight_map

    NL = num_layers if num_layers is not None else D.NUM_LAYERS
    w8_mode = MOE_FUSED_W8 if moe_w8_mode is None else moe_w8_mode
    w8_fp8_impl = MOE_FUSED_W8_FP8_IMPL
    use_official_reference = (
        USE_MOE_OFFICIAL_FP8_REFERENCE
        if official_fp8_reference is None
        else official_fp8_reference
    )
    if w8_mode not in ("", "fp8", "int8"):
        raise ValueError(f"invalid fused W8 mode {w8_mode!r}")
    if w8_mode and use_official_reference:
        raise ValueError("fused W8 and official-FP8 reference loading are exclusive")
    if (w8_mode or use_official_reference) and not expert_ckpt:
        raise ValueError("official FP8 expert loading requires an expert checkpoint")

    # Map every weight key -> shard file (dependency-free; the DLC lacks safetensors).
    wm = build_weight_map(ckpt)

    # Auto-detect the language-model prefix (the real 35B uses
    # "model.language_model."; the tiny-random model triple-nests it).
    sample = next(k for k in wm if "layers.0.input_layernorm.weight" in k)
    pfx = sample[: sample.index("layers.0.input_layernorm.weight")]
    # Embedding / norm / lm_head keys (search since prefix nesting varies).
    def find_key(suffix, contains=None):
        for k in wm:
            if k.endswith(suffix) and (contains is None or contains in k):
                return k
        return None
    EMBED_K = find_key("embed_tokens.weight")
    NORM_K = pfx + "norm.weight" if (pfx + "norm.weight") in wm else find_key(".norm.weight")
    LMHEAD_K = "lm_head.weight" if "lm_head.weight" in wm else find_key("lm_head.weight")

    # Cache open readers lazily, grouped to minimize reopens.
    _handles = {}
    def get(key):
        f = wm[key]
        if f not in _handles:
            _handles[f] = SafeReader(os.path.join(ckpt, f))
        return _handles[f].get_tensor(key)

    EXPERTS_PACKED = (pfx + "layers.0.mlp.experts.gate_up_proj") in wm

    expert_reader = (
        OfficialFP8ExpertReader(expert_ckpt)
        if (w8_mode or use_official_reference)
        else None
    )

    td = D.tp_dims(world_size)
    KH_full, VH_full = D.DN_K_HEADS, D.DN_V_HEADS
    KD, VD = D.DN_K_DIM, D.DN_V_DIM
    key_dim_full = KH_full * KD                  # 2048
    val_dim_full = VH_full * VD                  # 4096
    kh, vh = td["dn_k_heads"], td["dn_v_heads"]  # per core
    key_dim, val_dim = kh * KD, vh * VD

    QH_full = D.GQA_Q_HEADS
    HD = D.GQA_HEAD_DIM
    qh = td["gqa_q_heads"]
    kv_rep = td["gqa_kv_replication"]            # cores sharing one KV head
    # which KV head (of 2) this rank uses
    kv_head_for_rank = rank // kv_rep if kv_rep > 0 else 0

    experts_per = td["experts_per_core"]
    e_lo = rank * experts_per
    e_hi = e_lo + experts_per

    def colwise(w, r=rank, ws=world_size):
        s = w.shape[0] // ws
        return w[r * s:(r + 1) * s].clone()

    def rowwise(w, r=rank, ws=world_size):
        s = w.shape[1] // ws
        return w[:, r * s:(r + 1) * s].clone()

    def load_experts(lp, layer_index):
        """Load this rank's expert-parallel rows or intermediate-width shards."""
        if w8_mode:
            converted, stats = _load_fused_w8_expert_layer(
                expert_reader,
                layer_index,
                e_lo,
                e_hi,
                w8_mode,
                w8_fp8_impl,
            )
            if stats is not None:
                w8_stats.merge(stats)
            return converted
        if use_official_reference:
            return expert_reader.load_layer_bf16(layer_index, e_lo, e_hi)
        if USE_MOE_DECODE_TP:
            if D.MOE_INTER % world_size:
                raise RuntimeError(
                    "MOE_DECODE_TP requires moe_intermediate_size divisible by TP"
                )
            width = D.MOE_INTER // world_size
            i0 = rank * width
            i1 = i0 + width
            if EXPERTS_PACKED:
                gu_full = get(lp + "mlp.experts.gate_up_proj")
                gate = gu_full[:, i0:i1]
                up = gu_full[:, D.MOE_INTER + i0:D.MOE_INTER + i1]
                gu = torch.cat([gate, up], dim=1).clone()
                dn = get(lp + "mlp.experts.down_proj")[:, :, i0:i1].clone()
                return {"gate_up": gu, "down": dn}
            gus, dns = [], []
            for e in range(D.NUM_EXPERTS):
                ep = lp + f"mlp.experts.{e}."
                gate = get(ep + "gate_proj.weight")[i0:i1]
                up = get(ep + "up_proj.weight")[i0:i1]
                gus.append(torch.cat([gate, up], dim=0))
                dns.append(get(ep + "down_proj.weight")[:, i0:i1])
            return {
                "gate_up": torch.stack(gus, 0).clone(),
                "down": torch.stack(dns, 0).clone(),
            }
        if EXPERTS_PACKED:
            gu = get(lp + "mlp.experts.gate_up_proj")[e_lo:e_hi].clone()   # [Ec,2I,H]
            dn = get(lp + "mlp.experts.down_proj")[e_lo:e_hi].clone()      # [Ec,H,I]
            return {"gate_up": gu, "down": dn}
        # unpacked: experts.<e>.{gate,up,down}_proj.weight, each [I,H]/[H,I]
        gus, dns = [], []
        for e in range(e_lo, e_hi):
            ep = lp + f"mlp.experts.{e}."
            g = get(ep + "gate_proj.weight")          # [I,H]
            u = get(ep + "up_proj.weight")            # [I,H]
            gus.append(torch.cat([g, u], dim=0))      # [2I,H]
            dns.append(get(ep + "down_proj.weight"))  # [H,I]
        return {
            "gate_up": torch.stack(gus, 0).clone(),
            "down": torch.stack(dns, 0).clone(),
        }

    w8_stats = QuantizationStats()
    weights = {
        "embed": get(EMBED_K),
        "final_norm": get(NORM_K),
        "lm_head": get(LMHEAD_K) if LMHEAD_K is not None else get(EMBED_K),
        "layers": [],
    }

    for i in range(NL):
        lp = f"{pfx}layers.{i}."
        expert_weights = load_experts(lp, i)
        lw = {
            "input_norm": get(lp + "input_layernorm.weight"),
            "post_norm": get(lp + "post_attention_layernorm.weight"),
            # MoE — expert-parallel slice + replicated router/shared
            "router": get(lp + "mlp.gate.weight"),                       # [E,H] replicated
            "sh_gate": get(lp + "mlp.shared_expert.gate_proj.weight"),
            "sh_up": get(lp + "mlp.shared_expert.up_proj.weight"),
            "sh_down": get(lp + "mlp.shared_expert.down_proj.weight"),
            "sh_sigmoid": get(lp + "mlp.shared_expert_gate.weight"),
        }
        lw.update(expert_weights)
        if D.layer_type(i) == "deltanet":
            ap = lp + "linear_attn."
            qkv = get(ap + "in_proj_qkv.weight")             # [2*key+val, H]
            q_w = qkv[:key_dim_full]
            k_w = qkv[key_dim_full:2 * key_dim_full]
            v_w = qkv[2 * key_dim_full:]
            qs = key_dim // 1  # per-core key dim
            q_part = q_w[rank * key_dim:(rank + 1) * key_dim]
            k_part = k_w[rank * key_dim:(rank + 1) * key_dim]
            v_part = v_w[rank * val_dim:(rank + 1) * val_dim]
            lw["dn_qkv"] = torch.cat([q_part, k_part, v_part], dim=0).clone()
            # conv1d [2*key+val,1,4] sharded to match the q|k|v group slices
            conv = get(ap + "conv1d.weight")
            cq = conv[rank * key_dim:(rank + 1) * key_dim]
            ck = conv[key_dim_full + rank * key_dim: key_dim_full + (rank + 1) * key_dim]
            cv = conv[2 * key_dim_full + rank * val_dim: 2 * key_dim_full + (rank + 1) * val_dim]
            lw["dn_conv_w"] = torch.cat([cq, ck, cv], dim=0).clone()
            lw["dn_z"] = colwise(get(ap + "in_proj_z.weight"))            # [val/ws, H]
            lw["dn_a"] = colwise(get(ap + "in_proj_a.weight"))            # [vh, H]
            lw["dn_b"] = colwise(get(ap + "in_proj_b.weight"))
            lw["dn_out"] = rowwise(get(ap + "out_proj.weight"))           # [H, val/ws]
            lw["dn_A_log"] = colwise(get(ap + "A_log"))                   # [vh]
            lw["dn_dt_bias"] = colwise(get(ap + "dt_bias"))
            lw["dn_norm"] = get(ap + "norm.weight")                       # [VD] replicated
        else:
            ap = lp + "self_attn."
            lw["gqa_q"] = colwise(get(ap + "q_proj.weight"))             # [QH*HD*2/ws, H]
            # KV heads per core = max(1, KV//ws). When ws > KV, each KV head is
            # REPLICATED across (ws//KV) cores. When ws <= KV, split KV heads.
            nkv = max(1, D.GQA_KV_HEADS // world_size)
            kfull = get(ap + "k_proj.weight").reshape(D.GQA_KV_HEADS, HD, D.HIDDEN)
            vfull = get(ap + "v_proj.weight").reshape(D.GQA_KV_HEADS, HD, D.HIDDEN)
            if world_size >= D.GQA_KV_HEADS:
                # replicate: this rank's single KV head
                sel = slice(kv_head_for_rank, kv_head_for_rank + 1)
            else:
                # split: contiguous nkv heads for this rank
                sel = slice(rank * nkv, (rank + 1) * nkv)
            lw["gqa_k"] = kfull[sel].reshape(nkv * HD, D.HIDDEN).clone()   # [nkv*HD, H]
            lw["gqa_v"] = vfull[sel].reshape(nkv * HD, D.HIDDEN).clone()
            lw["gqa_o"] = rowwise(get(ap + "o_proj.weight"))            # [H, QH*HD/ws]
            lw["gqa_q_norm"] = get(ap + "q_norm.weight")                # [HD]
            lw["gqa_k_norm"] = get(ap + "k_norm.weight")
        weights["layers"].append(lw)

    if w8_mode and rank == 0:
        detail = f"/{w8_fp8_impl}" if w8_mode == "fp8" else ""
        if w8_mode == "fp8" and w8_fp8_impl == "row":
            detail += (
                f"/{MOE_FUSED_W8_FP8_PROJECTIONS}"
                f"/layers[{MOE_FUSED_W8_FP8_LAYER_START},"
                f"{MOE_FUSED_W8_FP8_LAYER_LIMIT})"
            )
        print(
            f"  official experts -> {w8_mode}{detail}: "
            f"cosine={w8_stats.cosine:.7f}, "
            f"normalized RMSE={w8_stats.normalized_rmse:.5%}"
            + (
                f", shifted blocks={w8_stats.shifted_block_fraction:.2%}, "
                f"exact values={w8_stats.exact_fraction:.2%}, "
                f"clipped={w8_stats.clipped_count}"
                if w8_mode == "fp8"
                and w8_fp8_impl in (
                    "block_pow2",
                    "block_pow2_coalesced",
                )
                else ""
            )
        )
    if use_official_reference and rank == 0:
        print("  experts loaded from the official FP8 checkpoint and dequantized to BF16")
    if expert_reader is not None:
        expert_reader.close()
    for reader in _handles.values():
        reader.close()
    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seq-len", type=int, default=20000)
    ap.add_argument("--num-tokens", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-layers", type=int, default=None,
                    help="Override layer count with a model prefix for fast bring-up.")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--cpu", action="store_true",
                    help="Run on CPU (single process) for correctness checks "
                         "without the neuron backend.")
    ap.add_argument("--model-path", default=MODEL_PATH,
                    help="Checkpoint dir (config.json + safetensors).")
    ap.add_argument(
        "--expert-model-path",
        default=EXPERT_MODEL_PATH,
        help="Official FP8 checkpoint used only for fused W8 expert tensors "
             "(default: QWEN35_FP8_MODEL_PATH).",
    )
    ap.add_argument("--graph-splits", type=int, default=0,
                    help="Compile the layer loop in N segments (0=auto: ceil(layers/20)). "
                         "Works around the neuronx-cc PGTiling collective limit on the "
                         "full graph.")
    ap.add_argument("--prompt-ids", type=str, default="",
                    help="Comma-separated prompt token ids. Runs prefill then decodes "
                         "(coherence check). Empty = seed token 100 at position 0.")
    ap.add_argument("--prefill-bench", type=int, default=0,
                    help="If >0, time a prefill of a synthetic N-token prompt "
                         "(prompt-ingest throughput) and exit. Reports tok/s.")
    ap.add_argument("--bucket-chunk", type=int, default=0,
                    help="If >0, use bucketed prefill with this chunk size. "
                         "GQA_DYNAMIC_ROPE_KV=1 reuses compiled segments across "
                         "runtime offsets and partial final buckets.")
    ap.add_argument("--bucket-compile", type=int, default=1,
                    help="1 = compile the per-chunk fn; 0 = run the chunk loop "
                         "EAGERLY (bit-exact vs eager prefill, avoids the "
                         "giant-graph compile without the compiled-kernel numerics drift).")
    ap.add_argument("--prefill-splits", type=int, default=1,
                    help="Number of coarse compiled layer segments per bucket. "
                         "Use 4 for 40 layers to stay below the compiler's "
                         "5-million-instruction graph limit.")
    ap.add_argument("--bench", action="store_true",
                    help="After decode, run a synced-TPOT benchmark window.")
    ap.add_argument("--bench-iters", type=int, default=50,
                    help="Iterations for the synced-TPOT benchmark.")
    args = ap.parse_args()
    model_path = args.model_path
    if USE_MOE_FUSED_W8 or USE_MOE_OFFICIAL_FP8_REFERENCE:
        if args.cpu:
            ap.error("official FP8 expert modes are device validation paths")
        if args.skip_compile:
            ap.error("official FP8 expert modes require a compiled full decode graph")
        if not args.expert_model_path:
            ap.error(
                "official FP8 expert modes require --expert-model-path or "
                "QWEN35_FP8_MODEL_PATH"
            )
    if USE_MOE_FUSED_W8 or USE_MOE_OFFICIAL_FP8_REFERENCE:
        if args.batch_size not in (32, 64, 128, 256):
            ap.error(
                "official FP8 expert modes support --batch-size "
                "32, 64, 128, or 256"
            )
        if not args.skip_prefill or args.prompt_ids or args.prefill_bench:
            ap.error(
                "official FP8 expert modes are decode-only; use --skip-prefill without "
                "--prompt-ids or --prefill-bench"
            )

    # Config-driven dims: lets the same harness run the real 35B and the
    # tiny-random debug model. Falls back to the baked-in 35B constants.
    cfg_path = os.path.join(model_path, "config.json")
    if os.path.exists(cfg_path):
        D.load_from_config(cfg_path)

    if args.num_layers is not None:
        if not 1 <= args.num_layers <= D.NUM_LAYERS:
            ap.error(f"--num-layers must be in [1, {D.NUM_LAYERS}]")
        D.NUM_LAYERS = args.num_layers
        D.NUM_GQA = sum(1 for i in range(D.NUM_LAYERS) if D.layer_type(i) == "gqa")
        D.NUM_DELTANET = D.NUM_LAYERS - D.NUM_GQA

    if args.cpu:
        rank, world_size, device = 0, 1, torch.device("cpu")
    else:
        import torch_neuronx
        dist.init_process_group(backend="neuron")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.neuron.current_device()

    if rank == 0:
        print(f"=== 35B-A3B static decode: TP={world_size}, LNC={LNC_DEGREE}, max_seq={args.max_seq_len}, "
              f"layers={D.NUM_LAYERS}, BS={args.batch_size}"
              f"{', fused W8=' + MOE_FUSED_W8 if USE_MOE_FUSED_W8 else ''}"
              f"{'/' + MOE_FUSED_W8_FP8_IMPL if MOE_FUSED_W8 == 'fp8' else ''}"
              f"{'/' + MOE_FUSED_W8_FP8_PROJECTIONS if USE_MOE_FUSED_W8_ROW_FP8 else ''}"
              f"{'/layers[' + str(MOE_FUSED_W8_FP8_LAYER_START) + ',' + str(MOE_FUSED_W8_FP8_LAYER_LIMIT) + ')' if USE_MOE_FUSED_W8_ROW_FP8 else ''} ===")

    t0 = time.time()
    weights = load_sharded_weights(
        model_path,
        rank,
        world_size,
        num_layers=D.NUM_LAYERS,
        expert_ckpt=args.expert_model_path,
    )
    if rank == 0:
        print(f"  weights loaded+sharded: {time.time()-t0:.1f}s")

    mod = StaticDecode35B(weights, args.max_seq_len, world_size,
                          batch_size=args.batch_size, rank=rank)
    del weights
    mod = mod.to(device).eval()
    if rank == 0:
        mem = sum(b.numel() * b.element_size() for b in mod.buffers()) / 1e9
        print(f"  module on device: {mem:.2f} GB/core")

    B = args.batch_size
    dtype = torch.bfloat16 if not args.cpu else torch.float32
    KD, VD = D.DN_K_DIM, D.DN_V_DIM
    td = D.tp_dims(world_size)
    vh = td["dn_v_heads"]
    qkv_dim = 2 * td["dn_k_heads"] * KD + vh * VD
    dn_states = torch.zeros(D.NUM_DELTANET, B, vh * KD, VD, dtype=dtype, device=device)
    conv_states = torch.zeros(D.NUM_DELTANET, B, qkv_dim, D.DN_CONV_KERNEL - 1, dtype=dtype, device=device)
    nkv = max(1, D.GQA_KV_HEADS // world_size)
    if USE_GQA_STATEFUL_KV:
        kv_k = mod.decode_kv_k
        kv_v = mod.decode_kv_v
    else:
        kv_k = torch.zeros(
            D.NUM_GQA, B, nkv, args.max_seq_len, D.GQA_HEAD_DIM,
            dtype=dtype, device=device,
        )
        kv_v = torch.zeros_like(kv_k)

    compiled_decode_step = None
    if not args.skip_compile and not args.cpu and args.prefill_bench == 0:
        # Graph-split: compile each layer-segment to its own NEFF. The full
        # 40-layer graph trips a neuronx-cc PGTiling assertion (~120 collectives);
        # ceil(NL/20) segments keep each NEFF compilable. forward() (embed +
        # segment dispatch + head) stays eager — the segments hold the compute.
        # Skipped for --prefill-bench (decode graph unused there).
        if USE_DECODE_FULLGRAPH:
            segs = mod.setup_segments(1, compile_each=False)
            decode_fn = (
                mod.decode_step_stateful
                if USE_GQA_STATEFUL_KV
                else mod.decode_step
            )
            compiled_decode_step = torch.compile(
                decode_fn, backend="neuron", fullgraph=True, dynamic=False
            )
            if rank == 0:
                lm_mode = "vocab-sharded LM head" if USE_DECODE_SHARDED_LM_HEAD else "replicated LM head"
                print(f"  full decode step compiled as one graph ({lm_mode}): {segs}")
        else:
            n_splits = args.graph_splits if args.graph_splits else max(1, -(-D.NUM_LAYERS // 20))
            segs = mod.setup_segments(n_splits, compile_each=True)
            if rank == 0:
                print(f"  graph-split into {len(segs)} compiled segments: {segs}")

    def run_decode(input_id, pos, dns, cvs, keys, values):
        if USE_GQA_STATEFUL_KV:
            if compiled_decode_step is not None:
                outputs = compiled_decode_step(input_id, pos, dns, cvs)
            else:
                outputs = mod.decode_step_stateful(input_id, pos, dns, cvs)
            return *outputs, mod.decode_kv_k, mod.decode_kv_v
        if compiled_decode_step is not None:
            return compiled_decode_step(input_id, pos, dns, cvs, keys, values)
        outputs = mod(input_id, pos, dns, cvs, keys, values)
        logits = outputs[0]
        return logits, logits.argmax(-1).to(torch.long), *outputs[1:]

    one = torch.tensor(1, dtype=torch.long, device=device)
    if not args.cpu:
        dist.barrier()

    # ── Prefill-throughput bench: time eager prefill of a synthetic N-token
    # prompt (prompt-ingest tok/s = TTFT regime). Prefill is eager; GQA prefill
    # is pure-torch full [S,S] causal, DeltaNet prefill is the chunked NKI kernel.
    if args.prefill_bench > 0:
        N = args.prefill_bench
        base_pid = torch.arange(N, dtype=torch.long) % D.VOCAB
        # Distinct rows exercise KV/state isolation while retaining homogeneous
        # sequence lengths and the same dynamic bucket offsets.
        pid = (
            base_pid.unsqueeze(0) + torch.arange(B).unsqueeze(1)
        ) % D.VOCAB
        pid = pid.to(device)
        bchunk = args.bucket_chunk
        use_bucket = bchunk > 0
        # warmup (compiles any lazily-traced prefill graphs) + timed run
        for w in range(2):
            if not args.cpu:
                dist.barrier()
            t0 = time.time()
            if use_bucket:
                outputs = mod.prefill_bucketed(
                    pid, dn_states, conv_states, kv_k, kv_v,
                    chunk=bchunk,
                    compile_chunk=(args.bucket_compile == 1),
                    compile_splits=args.prefill_splits,
                )
            else:
                outputs = mod.prefill(pid, dn_states, conv_states, kv_k, kv_v)
            logits = outputs[0]
            _ = int(logits[0].argmax())   # force materialization
            if not args.cpu:
                dist.barrier()
            dt = time.time() - t0
            if rank == 0:
                cc = "c" if args.bucket_compile == 1 else "e"
                mode = f"bucket{bchunk}{cc}" if use_bucket else "eager"
                tag = "warmup" if w == 0 else "TIMED "
                print(f"  PREFILL {tag} BS={B} N={N} ({mode}): {dt*1000:.1f} ms  |  {B*N/dt:.1f} aggregate prompt tok/s"
                      f"{' (incl compile)' if w == 0 else ''}")
                if os.environ.get("PREFILL_FINGERPRINT", "0") == "1":
                    lf = logits.float()
                    top = torch.topk(lf[0], 5).indices
                    finite_names = ("logits", "deltanet", "conv", "kv_k", "kv_v")
                    finite = ",".join(
                        f"{name}={bool(torch.isfinite(tensor).all())}"
                        for name, tensor in zip(finite_names, outputs)
                    )
                    print(
                        "  PREFILL fingerprint:"
                        f" sum={float(lf.sum()):.8e}"
                        f" norm={float(torch.linalg.vector_norm(lf)):.8e}"
                        f" top5={[int(v) for v in top]}"
                        f" finite[{finite}]"
                    )
        return

    # Optional real-prompt prefill: --prompt-ids "760,6511,..." runs the (eager)
    # prefill to build DeltaNet/conv/KV state, then decodes from the prompt's
    # last-token logits — a coherence check. Without it, seed token 100 @ pos 0.
    gen = []
    if args.prompt_ids:
        pid = [int(t) for t in args.prompt_ids.split(",") if t != ""]
        in_t = torch.tensor(pid, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1).contiguous()
        if args.bucket_chunk > 0:
            logits, dn_states, conv_states, kv_k, kv_v = mod.prefill_bucketed(
                in_t, dn_states, conv_states, kv_k, kv_v,
                chunk=args.bucket_chunk, compile_chunk=(args.bucket_compile == 1),
                compile_splits=args.prefill_splits)
        else:
            logits, dn_states, conv_states, kv_k, kv_v = mod.prefill(
                in_t, dn_states, conv_states, kv_k, kv_v)
        if USE_GQA_STATEFUL_KV:
            mod.decode_kv_k.copy_(kv_k)
            mod.decode_kv_v.copy_(kv_v)
            kv_k = mod.decode_kv_k
            kv_v = mod.decode_kv_v
        nid0 = logits[0].argmax().to(torch.long)
        next_id = nid0.reshape(1).expand(B).contiguous()
        gen.append(next_id)
        position = torch.tensor(len(pid), dtype=torch.long, device=device)
        if rank == 0:
            print(f"  prefilled {len(pid)} prompt tokens; first gen id={int(nid0)}")
    else:
        next_id = torch.full((B,), 100, dtype=torch.long, device=device)
        position = torch.tensor(0, dtype=torch.long, device=device)
        gen.append(next_id)

    # ── PROFILE_STEPS: minimal isolated decode trace for neuron-explorer ──
    # When set, run 3 warmup + PROFILE_STEPS constant-(token,pos) decode steps
    # (synced) and EXIT before the normal loop/bench. With
    # NEURON_RT_INSPECT_DEVICE_PROFILE=1 this writes an NTFF containing ONLY the
    # decode NEFF executions (same NEFF every step → clean per-op attribution,
    # not buried under warmup/prefill graphs). See [[reference-neuron-explorer-ui]].
    profile_steps = int(os.environ.get("PROFILE_STEPS", "0"))
    if profile_steps > 0:
        with torch.no_grad():
            for _ in range(3):                       # warmup (not the trace of interest)
                logits, _, dn_states, conv_states, kv_k, kv_v = run_decode(
                    next_id, position, dn_states, conv_states, kv_k, kv_v)
            if not args.cpu:
                torch.neuron.synchronize()
        if not args.cpu:
            dist.barrier()
        with torch.no_grad():
            for _ in range(profile_steps):           # constant token+pos → same decode NEFF
                logits, _, dn_states, conv_states, kv_k, kv_v = run_decode(
                    next_id, position, dn_states, conv_states, kv_k, kv_v)
            if not args.cpu:
                torch.neuron.synchronize()
        if rank == 0:
            print(f"  [profile] ran {profile_steps} decode steps for tracing; exiting")
        if not args.cpu:
            dist.barrier()
            dist.destroy_process_group()
        return

    with torch.no_grad():
        t0 = time.time()
        for step in range(args.num_tokens):
            logits, next_id, dn_states, conv_states, kv_k, kv_v = run_decode(
                next_id, position, dn_states, conv_states, kv_k, kv_v)
            gen.append(next_id)
            position = position + one
            if step == 0 and rank == 0:
                print(f"  first decode step (incl compile): {time.time()-t0:.1f}s")

        # ── Synced-TPOT benchmark (avoids the async-enqueue artifact: the neuron
        # backend dispatches async, so timing without a sync measures enqueue,
        # not execution — see the 27B AGENT.md 1.7ms-vs-40ms lesson). We force a
        # device sync by reading a value off each iteration's output.
        if args.bench:
            iters = args.bench_iters
            # warmup (NEFF already hot from the loop above)
            for _ in range(3):
                logits, next_id, dn_states, conv_states, kv_k, kv_v = run_decode(
                    next_id, position, dn_states, conv_states, kv_k, kv_v)
            _ = next_id[0].item()           # sync
            tb = time.time()
            for _ in range(iters):
                logits, next_id, dn_states, conv_states, kv_k, kv_v = run_decode(
                    next_id, position, dn_states, conv_states, kv_k, kv_v)
            _ = next_id[0].item()           # single sync after the batch
            if rank == 0:
                tpot_ms = (time.time() - tb) / iters * 1000.0
                tput = B * 1000.0 / tpot_ms
                print(f"  BENCH BS={B} seq={args.max_seq_len}: TPOT {tpot_ms:.2f} ms/tok "
                      f"(synced, {iters} iter) | throughput {tput:.1f} tok/s")

    if rank == 0:
        ids = [int(g[0].item()) for g in gen]
        print(f"  generated ids (row0): {ids[:8]}")
        print("  DONE")


if __name__ == "__main__":
    main()
