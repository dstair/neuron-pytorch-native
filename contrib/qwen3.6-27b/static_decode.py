#!/usr/bin/env python3
"""
Qwen 3.6 27B — Static decode step for torch.compile(fullgraph=True, backend="neuron").

Eliminates HuggingFace dispatch overhead (97% of latency) by writing a single
compilable function that covers all 64 layers. DeltaNet recurrence handled by
NKI kernel via @nki_op (no graph break).

Usage:
    torchrun --nproc-per-node=4 static_decode.py [--max-seq-len 2048] [--num-tokens 16]
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

_KERNELS_DIR = os.environ.get(
    "DELTANET_KERNELS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels"),
)
sys.path.insert(0, _KERNELS_DIR)
import deltanet_full_ops  # registers torch.ops.deltanet.full
import deltanet_full_batched_ops  # registers torch.ops.deltanet.full_batched (whole batch, 1 call/layer)
import deltanet_chunked_ops  # registers torch.ops.deltanet.chunked_prefill (1 call/layer compiled prefill)
import gqa_tail_ops  # registers torch.ops.gqa.tail (fused GQA attention-tail: norm+rope+attn+gate)
import deltanet_full_fp8_ops  # registers torch.ops.deltanet.full_fp8 (fold of in_proj + deltanet block)
import fp8_matmul_ops  # registers torch.ops.fp8.matmul

# Norm-fusion: free-axis RMSNorm kernel (no on-chip [[0,1]] collective).
# Gated by NORMFUSE=1 so the baseline path stays byte-identical when off.
USE_NKI_NORM = os.environ.get("NORMFUSE", "0") == "1"
if USE_NKI_NORM:
    from rms_norm_nki import nki_rms_norm

# NOTE: the fused-projection decode kernels (MLPFUSE/mlp_full, GQAFUSE/gqa_qkv,
# NKILIBMLP/nkilib_mlp) were RULED OUT — all regress at BS=1 vs F.linear (the
# compiler's GEMM path beats a hand-rolled tiled+transpose kernel). They've been
# moved to legacy/. See [[project-qwen36-native-baseline]]. The MLP/GQA decode
# layers below use F.linear only.

# Chunked-DeltaNet prefill kernel. CHUNKEDPREFILL=1 routes _deltanet_prefill
# through torch.ops.deltanet.chunked_prefill (ONE NKI custom call/layer) instead
# of chunked_prefill.neuron_chunk_gated_delta_rule's torch chunk-loop, whose
# data-dependent Woodbury slice can't compile under backend="neuron". This is
# what makes a COMPILED (fullgraph) prefill graph possible. Chunk size C=64.
# Default off = eager torch path (unchanged baseline). C must divide S.
USE_CHUNKED_PREFILL = os.environ.get("CHUNKEDPREFILL", "0") == "1"
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "64"))

# Lean KV-cache write. The decode KV write puts one token (position p, shared
# across the B batch rows) into kv[gi] = [B, max_seq, 256]. The scatter_ path
# materializes a [B,1,256] index tile (Vector/Scalar COPY) and uses indirect-
# scatter addressing (Sync EVENT_SEMAPHORE) — ~15% of BS=8 TPOT per the profile
# (static_decode :842/843). Since p is shared, this is just a contiguous slice
# write, expressible as index_copy_(dim=1, [p], src) → one dynamic-offset DMA,
# no index materialization, no indirect-scatter semaphore. Numerically identical.
# LEANKV=1 to enable; default off keeps the byte-identical scatter_ baseline.
USE_LEAN_KV = os.environ.get("LEANKV", "0") == "1"

# GQA-tail mega-kernel (Phase 1 BS>1 throughput). GQATAIL=1 routes the GQA
# attention tail (q RMSNorm + partial-64 RoPE + scaled scores + masked softmax +
# weighted-V + output-gate) through ONE torch.ops.gqa.tail custom call/layer instead
# of ~12 framework ops, collapsing the inter-op EVENT_SEMAPHORE barriers that the
# Phase 0 critical-path analysis showed dominate the BS=8 step. k-side norm/rope +
# KV-cache write stay torch; o_proj stays F.linear. bf16 weights only; default off.
USE_GQA_TAIL = os.environ.get("GQATAIL", "0") == "1"

# Phase 2 DeltaNet-surround glue lever. DNF32STATE=1 stores dn_states in f32 instead
# of bf16, so the per-layer state-in `.float()` cast (profile hotspot, Vector/Scalar
# COPY) and the `.to(bf16)` writeback both become no-ops — removing the cast-COPY
# barriers around the deltanet_full_batched call WITHOUT touching the unavoidable
# state-read/writeback DMA (the surface lean-KV/v3/v4 couldn't move). Numerically
# CLEANER (drops a bf16 round-trip; the kernel already computes state in f32). Costs
# ~2x dn_states memory (tiny vs the 12.3GB/core weights). Default off (bf16 baseline).
USE_DN_F32_STATE = os.environ.get("DNF32STATE", "0") == "1"

# Collective-cost probe: NOREDUCE=1 turns every TP all-reduce into a no-op. This
# BREAKS correctness (each rank keeps its partial sum) but isolates the wall-clock
# cost of the 128 all-reduces/token (2 per layer × 64). The TPOT delta vs BASE is
# the CEILING on what any collective-fusion / collective-reduction work can recover
# — measure the lever before building it. Default off = byte-identical baseline.
NO_REDUCE = os.environ.get("NOREDUCE", "0") == "1"


def functional_all_reduce(x, op, group):
    """TP all-reduce, or identity when NOREDUCE=1 (cost-probe only — not correct)."""
    if NO_REDUCE:
        return x
    return _functional_all_reduce(x, op, group)


MODEL_PATH = "/home/ubuntu/models/Qwen3.6-27B"

# Architecture constants
NUM_LAYERS = 64
NUM_DELTANET = 48
NUM_GQA = 16
HIDDEN = 5120
INTERMEDIATE = 17408
VOCAB = 248320
RMS_EPS = 1e-6

# DeltaNet constants (per-core after TP=4)
DN_QKV_DIM = 2560       # in_proj_qkv output dim per core
DN_K_HEADS = 4          # key heads per core
DN_V_HEADS = 12         # value heads per core
DN_K_DIM = 128          # per-head key dim
DN_V_DIM = 128          # per-head value dim
DN_VALUE_DIM = 1536     # total value dim per core (12*128)
DN_CONV_KERNEL = 4

# GQA constants (per-core after TP=4)
GQA_Q_HEADS = 6         # query heads per core
GQA_KV_HEADS = 1        # KV heads per core
GQA_HEAD_DIM = 256      # head dimension
GQA_Q_DIM = 3072        # q_proj output per core (6*256*2 for query+gate)
# RoPE is PARTIAL: config partial_rotary_factor=0.25 -> rotary applies to the
# first 64 of 256 head dims; dims [64:256] pass through unrotated. theta=1e7.
ROPE_DIM = int(GQA_HEAD_DIM * 0.25)  # 64
ROPE_THETA = 10000000.0              # config rope_theta (1e7, NOT 1e6)

# MLP constants (per-core after TP=4)
MLP_DIM = 4352          # intermediate / tp


def layer_type(i: int) -> str:
    """Layer pattern: [DeltaNet×3, GQA×1] × 16."""
    return "gqa" if i % 4 == 3 else "deltanet"


def deltanet_index(layer_idx: int) -> int:
    """Map absolute layer index to DeltaNet state index (0..47)."""
    block = layer_idx // 4
    offset = layer_idx % 4
    return block * 3 + offset


def gqa_index(layer_idx: int) -> int:
    """Map absolute layer index to GQA cache index (0..15)."""
    return layer_idx // 4


# ─── Weight Extraction ────────────────────────────────────────────────────────

def build_random_sharded_weights(num_layers: int, seed: int) -> dict:
    """Build a random, full-width, per-core weight dict WITHOUT loading the real
    27B checkpoint or running shard_model.

    Returns the SAME structure (and per-core shapes) that extract_weights() emits
    after shard_model() — so StaticDecodeModule and the decode graph are
    byte-identical in shape to production for every WIDTH (HIDDEN, DN/GQA/MLP
    dims). Only NUM_LAYERS and VOCAB are shrunk, which slashes compile time
    ~NUM_LAYERS/64× and embed/lm_head memory while FULLY preserving the per-layer
    RMSNorm cross-core reduction + TP all-reduce pattern we're trying to fix.

    Weights are random (small scale) — numerics are self-consistent only
    (kernel-vs-reference), NOT bit-exact vs the real model. That is exactly what
    the norm-fusion correctness + collective-count validation needs.
    """
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale: float = 0.02, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g) * scale).to(dtype)

    w = {
        "embed": rnd(VOCAB, HIDDEN),
        "final_norm": rnd(HIDDEN),
        "lm_head": rnd(VOCAB, HIDDEN),
        "layers": [],
    }
    for i in range(num_layers):
        lw = {
            "input_norm": rnd(HIDDEN),
            "post_norm": rnd(HIDDEN),
            "mlp_gate": rnd(MLP_DIM, HIDDEN),
            "mlp_up": rnd(MLP_DIM, HIDDEN),
            "mlp_down": rnd(HIDDEN, MLP_DIM),
        }
        if layer_type(i) == "deltanet":
            lw.update({
                "dn_qkv": rnd(DN_QKV_DIM, HIDDEN),                 # [2560, 5120]
                "dn_conv_w": rnd(DN_QKV_DIM, 1, DN_CONV_KERNEL),   # [2560, 1, 4]
                "dn_conv_b": None,                                 # zeros path in layer
                "dn_z": rnd(DN_V_HEADS * DN_V_DIM, HIDDEN),        # [1536, 5120]
                "dn_a": rnd(DN_V_HEADS, HIDDEN),                   # [12, 5120]
                "dn_b": rnd(DN_V_HEADS, HIDDEN),                   # [12, 5120]
                "dn_out": rnd(HIDDEN, DN_VALUE_DIM),               # [5120, 1536]
                "dn_A_log": rnd(DN_V_HEADS, scale=0.1, dtype=torch.float32),
                "dn_dt_bias": rnd(DN_V_HEADS, scale=0.1, dtype=torch.float32),
                "dn_norm": rnd(DN_V_DIM),                          # [128]
            })
        else:
            lw.update({
                "gqa_q": rnd(GQA_Q_DIM, HIDDEN),                          # [3072, 5120]
                "gqa_k": rnd(GQA_KV_HEADS * GQA_HEAD_DIM, HIDDEN),        # [256, 5120]
                "gqa_v": rnd(GQA_KV_HEADS * GQA_HEAD_DIM, HIDDEN),        # [256, 5120]
                "gqa_o": rnd(HIDDEN, GQA_Q_HEADS * GQA_HEAD_DIM),         # [5120, 1536]
                "gqa_q_norm": rnd(GQA_HEAD_DIM),                          # [256]
                "gqa_k_norm": rnd(GQA_HEAD_DIM),                          # [256]
            })
        w["layers"].append(lw)
    return w


def extract_weights(model) -> dict:
    """Extract all weight tensors from the TP-sharded model into a flat dict."""
    lang = model.model.language_model
    w = {
        "embed": lang.embed_tokens.weight,            # [248320, 5120]
        "final_norm": lang.norm.weight,               # [5120]
        "lm_head": model.lm_head.weight,              # [248320, 5120]
        "layers": [],
    }

    for i, layer in enumerate(lang.layers):
        lw = {
            "input_norm": layer.input_layernorm.weight,
            "post_norm": layer.post_attention_layernorm.weight,
            "mlp_gate": layer.mlp.gate_proj.weight,
            "mlp_up": layer.mlp.up_proj.weight,
            "mlp_down": layer.mlp.down_proj.weight,
        }

        if layer_type(i) == "deltanet":
            attn = layer.linear_attn
            lw.update({
                "dn_qkv": attn.in_proj_qkv.weight,
                "dn_conv_w": attn.conv1d.weight,         # [C, 1, 4] depthwise
                "dn_conv_b": attn.conv1d.bias if attn.conv1d.bias is not None else None,
                "dn_z": attn.in_proj_z.weight,
                "dn_a": attn.in_proj_a.weight,
                "dn_b": attn.in_proj_b.weight,
                "dn_out": attn.out_proj.weight,
                "dn_A_log": attn.A_log,
                "dn_dt_bias": attn.dt_bias,
                "dn_norm": attn.norm.weight,             # [128] RMSNormGated
            })
        else:
            attn = layer.self_attn
            lw.update({
                "gqa_q": attn.q_proj.weight,
                "gqa_k": attn.k_proj.weight,
                "gqa_v": attn.v_proj.weight,
                "gqa_o": attn.o_proj.weight,
                "gqa_q_norm": attn.q_norm.weight,        # [256]
                "gqa_k_norm": attn.k_norm.weight,        # [256]
            })

        w["layers"].append(lw)

    return w


# ─── Static Forward Components ───────────────────────────────────────────────

# ─── FP8 W8A16 weight quantization ───────────────────────────────────────────
#
# Quantize Linear weights to FP8 (E4M3) with per-output-channel f32 scales.
# Decode reads the FP8 weight (1 byte/elem) instead of bf16 (2 bytes/elem) →
# halves weight bandwidth, the dominant cost in the per-token decode (E2E
# profile shows 84% MBU on 2.5GB/step weight reads).
#
# At forward time we cast FP8 -> bf16, scale by per-row f32, and call F.linear.
# torch.compile's neuron backend should fuse the cast+scale into the matmul's
# operand-load path. If profiling shows it doesn't fuse, we fall back to a
# custom NKI matmul that takes FP8 stationary directly.

# IMPORTANT: 240.0 is the max **legacy** E4M3 representable value
# (exponent < 0xF). The OCP `e4m3fn` variant extends to 448.0 by reusing the
# all-1s exponent for finite values, but Trn2 nc_matmul only supports the
# legacy format. We clamp to 240 so the quantized bytes are valid in BOTH
# encodings — the kernel reinterprets the bytes as legacy e4m3 in SBUF.
FP8_E4M3_MAX = 240.0


def quantize_fp8_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w: [out, in] bf16 -> (w_fp8_T [in, out] int8, scale [out, 1] f32).
    Per-output-channel symmetric absmax quantization.

    Returns weight pre-transposed to [in, out] (= [K, N]) layout so the NKI
    matmul kernel can DMA it directly with K on the partition dim. Returned
    as int8 (bitwise-equivalent to fp8) to dodge the Trn2 HLO verifier that
    rejects F8E4M3FN; the kernel bit-reinterprets in SBUF as nl.float8_e4m3.

    Memory-conscious: never materialize a full f32 copy of w (would double
    memory on a 27B model and OOM across 4 ranks).
    """
    absmax = w.abs().amax(dim=-1, keepdim=False).float().clamp_min(1e-12)  # [out]
    scale = absmax / FP8_E4M3_MAX                                          # [out]
    inv_scale_bf = (1.0 / scale).to(w.dtype).unsqueeze(-1)                 # [out, 1]
    w_q = (w * inv_scale_bf).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    # Pre-transpose to [in, out] = [K, N] for the kernel's expected layout,
    # and view as int8 (same bytes) so the buffer survives torch.compile's
    # HLO type checks.
    w_q_T_i8 = w_q.t().contiguous().view(torch.int8)
    scale_2d = scale.unsqueeze(-1).contiguous()  # [out, 1]
    return w_q_T_i8, scale_2d


def fp8_linear(x: torch.Tensor, w_fp8_T_i8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x [..., in] bf16, w_fp8_T_i8 [in, out] int8 (fp8 bytes), scale [out, 1] f32.
    Returns [..., out] in x.dtype. Calls the NKI fp8 matmul kernel which does
    nc_matmul on TensorE with fp8 stationary and bf16 moving, then applies
    the per-channel scale on the f32 PSUM result.

    The compiler can't see fp8 ops directly (Trn2 doesn't support e4m3fn);
    the kernel hides the type behind a custom-call boundary by accepting
    int8 input and bit-reinterpreting in SBUF.
    """
    flat_x = x.reshape(-1, x.shape[-1])  # [B, K]
    out = torch.ops.fp8.matmul(flat_x, w_fp8_T_i8, scale)
    return out.reshape(*x.shape[:-1], -1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + weight). Qwen3.5 residual norm.

    When NORMFUSE=1 and x is a single-token decode vector ([..,1,H]), route the
    HIDDEN-wide norm through nki_rms_norm: its sum-of-squares is a free-axis
    reduce on one partition, so the compiler emits no on-chip [[0,1]] collective.
    Other shapes (multi-token prefill, per-head q/k norms) keep the torch path.
    """
    if USE_NKI_NORM and x.shape[-1] == HIDDEN and x.numel() == HIDDEN:
        flat = nki_rms_norm(x.reshape(HIDDEN), weight.reshape(HIDDEN))
        return flat.reshape(x.shape)
    x_f32 = x.float()
    norm = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return ((1.0 + weight.float()) * norm).to(x.dtype)


def rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """RMSNormGated: rms_norm(x) * weight * silu(gate). Applied per-head."""
    x_f32 = x.float()
    norm = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return (weight * norm.to(x.dtype)) * F.silu(gate.float()).to(x.dtype)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """L2 normalize along last dim."""
    return F.normalize(x, p=2, dim=-1, eps=1e-6)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    """Apply PARTIAL rotary position embedding (matches HF Qwen3.5).

    cos/sin have last dim == rotary_dim (ROPE_DIM=64), narrower than the head
    dim (256). Rotary is applied to q/k[..., :rotary_dim]; the remaining dims
    [rotary_dim:] pass through unrotated.
    """
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat(((q_rot * cos) + (rotate_half(q_rot) * sin), q_pass), dim=-1)
    k_embed = torch.cat(((k_rot * cos) + (rotate_half(k_rot) * sin), k_pass), dim=-1)
    return q_embed, k_embed


# ─── Static Decode Module ────────────────────────────────────────────────────

class StaticDecodeModule(nn.Module):
    """Holds extracted weights and implements the static decode forward."""

    # Set of attribute prefixes that get FP8-quantized when fp8_weights=True.
    # Each entry "X" means: if attr `X` exists on a layer/module, quantize it
    # and replace with `X_q` (fp8 weight) + `X_s` (f32 per-row scale).
    _FP8_LINEAR_NAMES = {
        # Per-layer
        "mlp_gate", "mlp_up", "mlp_down",
        "dn_qkv", "dn_z", "dn_a", "dn_b", "dn_out",
        "gqa_q", "gqa_k", "gqa_v", "gqa_o",
        # Top-level (handled separately): lm_head_w
    }

    def __init__(self, weights: dict, max_seq_len: int, world_size: int,
                 fp8_weights: bool = False, batch_size: int = 1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.world_size = world_size
        self.tp_group = list(range(world_size))
        self.fp8_weights = fp8_weights
        self.batch_size = batch_size  # decode batch B; layers read this for B>1
        # Names of bf16 buffers stored PRE-TRANSPOSED to [in, out] for the fused
        # decode kernels. The prefill/fallback F.linear path (_lin) must transpose
        # these back on the fly (a cheap view; prefill is not perf-critical) since
        # F.linear expects [out, in]. Kept as a set so _lin can branch by name.
        self._transposed_w: set[str] = set()

        # Register-and-pop helper — drops the source-dict entry as we go so
        # the original bf16 weights get freed during construction. Without
        # this, peak memory ~doubles on a 27B model and OOMs across 4 ranks.
        def reg_linear(name: str, w: torch.Tensor, src: dict, src_key: str,
                       transpose: bool = False):
            if self.fp8_weights:
                w_q, scale = quantize_fp8_per_channel(w)
                self.register_buffer(f"{name}_q", w_q)
                self.register_buffer(f"{name}_s", scale)
            elif transpose:
                # Pre-transpose [out, in] -> [in, out] = [K, N] ONCE at load time,
                # so the fused NKI kernel DMAs the weight straight into the matmul
                # stationary operand with no per-tile nc_transpose (mirrors
                # fp8_matmul's w_fp8_T). The original [out,in] copy is NOT kept —
                # the F.linear fallback transposes back on the fly (see _lin).
                self.register_buffer(name, w.t().contiguous())
                self._transposed_w.add(name)
            else:
                self.register_buffer(name, w)
            # Drop the source-dict reference so refcount goes to zero
            # (assuming no other holders) and the bf16 weight frees.
            src.pop(src_key, None)

        def reg_buf(name: str, w: torch.Tensor, src: dict, src_key: str):
            self.register_buffer(name, w)
            src.pop(src_key, None)

        reg_buf("embed", weights["embed"], weights, "embed")
        reg_buf("final_norm", weights["final_norm"], weights, "final_norm")
        reg_linear("lm_head_w", weights["lm_head"], weights, "lm_head")

        # Per-layer weights
        for i, lw in enumerate(weights["layers"]):
            reg_buf(f"l{i}_input_norm", lw["input_norm"], lw, "input_norm")
            reg_buf(f"l{i}_post_norm", lw["post_norm"], lw, "post_norm")
            reg_linear(f"l{i}_mlp_gate", lw["mlp_gate"], lw, "mlp_gate")
            reg_linear(f"l{i}_mlp_up", lw["mlp_up"], lw, "mlp_up")
            reg_linear(f"l{i}_mlp_down", lw["mlp_down"], lw, "mlp_down")

            if layer_type(i) == "deltanet":
                reg_linear(f"l{i}_dn_qkv", lw["dn_qkv"], lw, "dn_qkv")
                reg_buf(f"l{i}_dn_conv_w", lw["dn_conv_w"], lw, "dn_conv_w")
                if lw.get("dn_conv_b") is not None:
                    reg_buf(f"l{i}_dn_conv_b", lw["dn_conv_b"], lw, "dn_conv_b")
                reg_linear(f"l{i}_dn_z", lw["dn_z"], lw, "dn_z")
                reg_linear(f"l{i}_dn_a", lw["dn_a"], lw, "dn_a")
                reg_linear(f"l{i}_dn_b", lw["dn_b"], lw, "dn_b")
                reg_linear(f"l{i}_dn_out", lw["dn_out"], lw, "dn_out")
                reg_buf(f"l{i}_dn_A_log", lw["dn_A_log"], lw, "dn_A_log")
                reg_buf(f"l{i}_dn_dt_bias", lw["dn_dt_bias"], lw, "dn_dt_bias")
                reg_buf(f"l{i}_dn_norm", lw["dn_norm"], lw, "dn_norm")
            else:
                reg_linear(f"l{i}_gqa_q", lw["gqa_q"], lw, "gqa_q")
                reg_linear(f"l{i}_gqa_k", lw["gqa_k"], lw, "gqa_k")
                reg_linear(f"l{i}_gqa_v", lw["gqa_v"], lw, "gqa_v")
                reg_linear(f"l{i}_gqa_o", lw["gqa_o"], lw, "gqa_o")
                reg_buf(f"l{i}_gqa_q_norm", lw["gqa_q_norm"], lw, "gqa_q_norm")
                reg_buf(f"l{i}_gqa_k_norm", lw["gqa_k_norm"], lw, "gqa_k_norm")

        # Precompute RoPE cos/sin for max_seq_len (head_dim=256)
        self._init_rope(max_seq_len)

        # [C,C] constants for the chunked-prefill NKI kernel (no iota in this NKI
        # build → pass as host constants). m_incl doubles as the cumsum operator.
        C = CHUNK_SIZE
        _idx = torch.arange(C)
        _i = _idx.view(C, 1)
        _j = _idx.view(1, C)
        self.register_buffer("chunk_m_incl", (_i >= _j).float())
        self.register_buffer("chunk_m_strict", (_i > _j).float())
        self.register_buffer("chunk_eye", torch.eye(C, dtype=torch.float32))

    def _lin(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Linear by attribute name. Picks bf16 weight or FP8 weight+scale
        based on self.fp8_weights, set at __init__ time."""
        if self.fp8_weights:
            w_q = getattr(self, f"{name}_q")
            scale = getattr(self, f"{name}_s")
            return fp8_linear(x, w_q, scale)
        else:
            w = getattr(self, name)
            # Buffers fed to the fused kernels are stored [in, out]; F.linear
            # wants [out, in], so transpose back (cheap view, prefill-only path).
            if name in self._transposed_w:
                w = w.t()
            return F.linear(x.to(w.dtype), w)

    def _init_rope(self, max_seq_len: int):
        """Precompute PARTIAL RoPE cos/sin tables (matches HF Qwen3.5).

        Rotary spans only ROPE_DIM=64 of the 256 head dims (partial_rotary_factor
        0.25) at theta=1e7. mRoPE collapses to standard 1D RoPE for text-only
        decode (all 3 position grids equal), so a 1D table is correct here.
        """
        rope_dim = ROPE_DIM  # 64
        inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # [max_seq, rope_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq, rope_dim]
        self.register_buffer("rope_cos", emb.cos().unsqueeze(0).unsqueeze(0))  # [1, 1, max_seq, rope_dim]
        self.register_buffer("rope_sin", emb.sin().unsqueeze(0).unsqueeze(0))  # [1, 1, max_seq, rope_dim]

    def forward(
        self,
        input_id: torch.Tensor,       # [B] - one token id per batch row
        position: torch.Tensor,        # scalar tensor (shared position across batch)
        deltanet_states: torch.Tensor, # [48, B, 1536, 128]
        conv_states: torch.Tensor,     # [48, B, 2560, 3]
        kv_cache_k: torch.Tensor,      # [16, B, max_seq, 256]
        kv_cache_v: torch.Tensor,      # [16, B, max_seq, 256]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Static decode step: B tokens in, B logits out.
        Returns: (logits, new_deltanet_states, new_conv_states, new_kv_cache_k, new_kv_cache_v)
        """
        # Embedding lookup: [B] -> [B, 1, 5120] (one token per batch row)
        hidden = F.embedding(input_id, self.embed).unsqueeze(1)  # [B, 1, 5120]

        # Clone mutable state for functional style
        dn_states = deltanet_states.clone()
        cv_states = conv_states.clone()
        kv_k = kv_cache_k.clone()
        kv_v = kv_cache_v.clone()

        # Get RoPE for current position (use index_select for Dynamo compatibility)
        cos = self.rope_cos.squeeze(0).squeeze(0).index_select(0, position.unsqueeze(0)).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 64]
        sin = self.rope_sin.squeeze(0).squeeze(0).index_select(0, position.unsqueeze(0)).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 64]

        # ─── Layer Loop (unrolled at trace time) ─────────────────────────
        for i in range(NUM_LAYERS):
            # Pre-attention norm
            normed = rms_norm(hidden, getattr(self, f"l{i}_input_norm"))

            if layer_type(i) == "deltanet":
                hidden = hidden + self._deltanet_layer(i, normed, dn_states, cv_states)
            else:
                hidden = hidden + self._gqa_layer(i, normed, cos, sin, position, kv_k, kv_v)

            # Post-attention norm + MLP
            normed = rms_norm(hidden, getattr(self, f"l{i}_post_norm"))
            hidden = hidden + self._mlp_layer(i, normed)

        # Final norm + LM head
        hidden = rms_norm(hidden, self.final_norm)
        logits = self._lin("lm_head_w", hidden.to(torch.bfloat16))  # [B, 1, vocab]

        return logits.squeeze(1), dn_states, cv_states, kv_k, kv_v  # [B, vocab]

    def prefill(
        self,
        input_ids: torch.Tensor,       # [seq_len] token ids
        deltanet_states: torch.Tensor,  # [48, B, 1536, 128] (prefill uses B=1, row 0)
        conv_states: torch.Tensor,      # [48, B, 2560, 3]
        kv_cache_k: torch.Tensor,       # [16, B, max_seq, 256]
        kv_cache_v: torch.Tensor,       # [16, B, max_seq, 256]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prefill: process all prompt tokens at once (eager, not compiled).
        Returns: (last_logits, dn_states, conv_states, kv_k, kv_v)
        """
        seq_len = input_ids.shape[0]
        hidden = F.embedding(input_ids, self.embed).unsqueeze(0).float()  # [1, S, 5120] fp32

        dn_states = deltanet_states.clone()
        cv_states = conv_states.clone()
        kv_k = kv_cache_k.clone()
        kv_v = kv_cache_v.clone()

        for i in range(NUM_LAYERS):
            normed = rms_norm(hidden, getattr(self, f"l{i}_input_norm"))

            if layer_type(i) == "deltanet":
                hidden = hidden + self._deltanet_prefill(i, normed, dn_states, cv_states)
            else:
                hidden = hidden + self._gqa_prefill(i, normed, seq_len, kv_k, kv_v)

            normed = rms_norm(hidden, getattr(self, f"l{i}_post_norm"))
            hidden = hidden + self._mlp_prefill(i, normed)

        hidden = rms_norm(hidden, self.final_norm)
        logits = self._lin("lm_head_w", hidden[:, -1:, :].to(torch.bfloat16))  # [1, 1, vocab]
        return logits.squeeze(0), dn_states, cv_states, kv_k, kv_v

    def _deltanet_prefill(
        self, i: int, x: torch.Tensor,
        dn_states: torch.Tensor, cv_states: torch.Tensor
    ) -> torch.Tensor:
        """DeltaNet prefill: full sequence, exact recurrence."""
        di = deltanet_index(i)
        seq_len = x.shape[1]

        conv_w = getattr(self, f"l{i}_dn_conv_w")
        conv_b = getattr(self, f"l{i}_dn_conv_b", None)
        A_log = getattr(self, f"l{i}_dn_A_log")
        dt_bias = getattr(self, f"l{i}_dn_dt_bias")
        norm_w = getattr(self, f"l{i}_dn_norm")

        x_2d = x.squeeze(0).to(torch.bfloat16)  # [S, 5120]

        # QKV projection
        mixed_qkv = self._lin(f"l{i}_dn_qkv", x_2d)  # [S, 2560]

        # Causal conv1d over full sequence
        # Prepend conv_state [2560, 3] as history. States are batched [di, B, ...];
        # prefill is BS=1 so index batch row 0.
        conv_input = torch.cat([cv_states[di, 0], mixed_qkv.t()], dim=-1)  # [2560, 3+S]
        conv_input_3d = conv_input.unsqueeze(0)  # [1, 2560, 3+S]
        conv_out = F.conv1d(conv_input_3d, conv_w, groups=DN_QKV_DIM)  # [1, 2560, S]
        if conv_b is not None:
            conv_out = conv_out + conv_b.unsqueeze(0).unsqueeze(-1)
        # Update conv_state: last 3 timesteps of the raw input
        cv_states[di, 0] = mixed_qkv.t()[:, -3:]  # [2560, 3]
        mixed_qkv = F.silu(conv_out.squeeze(0).t())  # [S, 2560]

        # Split and reshape
        q = mixed_qkv[:, :DN_K_HEADS * DN_K_DIM].reshape(seq_len, DN_K_HEADS, DN_K_DIM)
        k = mixed_qkv[:, DN_K_HEADS * DN_K_DIM:2 * DN_K_HEADS * DN_K_DIM].reshape(seq_len, DN_K_HEADS, DN_K_DIM)
        v = mixed_qkv[:, 2 * DN_K_HEADS * DN_K_DIM:].reshape(seq_len, DN_V_HEADS, DN_V_DIM)

        # Expand k-heads to v-heads
        q = q.repeat_interleave(DN_V_HEADS // DN_K_HEADS, dim=1)  # [S, 12, 128]
        k = k.repeat_interleave(DN_V_HEADS // DN_K_HEADS, dim=1)  # [S, 12, 128]

        # Gate and beta: [S, 12]
        a_out = self._lin(f"l{i}_dn_a", x_2d).float()  # [S, 12]
        b_out = self._lin(f"l{i}_dn_b", x_2d)          # [S, 12]
        beta = b_out.sigmoid()  # [S, 12]
        g = -A_log.float().exp() * F.softplus(a_out + dt_bias.float())  # [S, 12]

        if USE_CHUNKED_PREFILL:
            # ONE NKI custom call/layer — compilable under backend="neuron".
            # Kernel layout is head-major (row h*S+t) and it L2-normalizes q,k +
            # applies the 1/sqrt(K) q-scale INTERNALLY, so pass RAW q,k.
            H = DN_V_HEADS
            # [S, H, D] -> [H, S, D] -> [H*S, D]
            q_hm = q.float().transpose(0, 1).reshape(H * seq_len, DN_K_DIM).contiguous()
            k_hm = k.float().transpose(0, 1).reshape(H * seq_len, DN_K_DIM).contiguous()
            v_hm = v.float().transpose(0, 1).reshape(H * seq_len, DN_V_DIM).contiguous()
            # g, beta: [S, H] -> [H, S] -> [H*S, 1]
            g_hm = g.transpose(0, 1).reshape(H * seq_len, 1).contiguous()
            beta_hm = beta.float().transpose(0, 1).reshape(H * seq_len, 1).contiguous()
            # state: dn_states[di,0] is already [V_HEADS*K_DIM, V_DIM] = [1536,128]
            state_in = dn_states[di, 0].float()  # [1536, 128]

            out_hm, new_state = torch.ops.deltanet.chunked_prefill(
                state_in, q_hm, k_hm, v_hm, g_hm, beta_hm,
                self.chunk_m_incl, self.chunk_m_strict, self.chunk_eye,
            )
            # out_hm: [H*S, V_DIM] head-major -> [S, H, V_DIM]
            dn_states[di, 0] = new_state.reshape(DN_V_HEADS * DN_K_DIM, DN_V_DIM).to(dn_states.dtype)
            attn_out = out_hm.reshape(H, seq_len, DN_V_DIM).transpose(0, 1)  # [S, 12, 128]
        else:
            # EAGER fallback: torch chunk-loop (does NOT compile under backend="neuron";
            # the data-dependent Woodbury slice breaks neuronx-cc). L2 norm here.
            q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
            k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
            # Note: 1/sqrt(k_dim) scale is applied inside neuron_chunk_gated_delta_rule
            from chunked_prefill import neuron_chunk_gated_delta_rule
            # Input format: [B, T, H, D]
            q_in = q.unsqueeze(0)  # [1, S, 12, 128]
            k_in = k.unsqueeze(0)  # [1, S, 12, 128]
            v_in = v.float().unsqueeze(0)  # [1, S, 12, 128]
            g_in = g.unsqueeze(0)  # [1, S, 12]
            beta_in = beta.float().unsqueeze(0)  # [1, S, 12]
            init_state = dn_states[di, 0].float().reshape(1, DN_V_HEADS, DN_K_DIM, DN_V_DIM)

            attn_out_4d, final_state = neuron_chunk_gated_delta_rule(
                q_in, k_in, v_in, g=g_in, beta=beta_in,
                chunk_size=64,
                initial_state=init_state, output_final_state=True,
                use_qk_l2norm_in_kernel=False  # already normalized + scaled
            )
            # attn_out_4d: [1, S, 12, 128], final_state: [1, 12, 128, 128]
            dn_states[di, 0] = final_state.squeeze(0).reshape(DN_V_HEADS * DN_K_DIM, DN_V_DIM).to(dn_states.dtype)
            attn_out = attn_out_4d.squeeze(0)  # [S, 12, 128]

        # Gated norm + output projection
        z = self._lin(f"l{i}_dn_z", x_2d).reshape(seq_len, DN_V_HEADS, DN_V_DIM)  # [S, 12, 128]
        gated = rms_norm_gated(attn_out.to(x.dtype), z, norm_w)  # [S, 12, 128]
        out = self._lin(f"l{i}_dn_out", gated.reshape(seq_len, DN_VALUE_DIM).to(torch.bfloat16))  # [S, 5120]
        out = functional_all_reduce(out, "sum", self.tp_group)
        return out.unsqueeze(0)  # [1, S, 5120]

    def _gqa_prefill(
        self, i: int, x: torch.Tensor, seq_len: int,
        kv_k: torch.Tensor, kv_v: torch.Tensor,
    ) -> torch.Tensor:
        """GQA prefill: causal attention over full prompt."""
        gi = gqa_index(i)

        q_norm_w = getattr(self, f"l{i}_gqa_q_norm")
        k_norm_w = getattr(self, f"l{i}_gqa_k_norm")

        x_2d = x.squeeze(0).to(torch.bfloat16)  # [S, 5120]

        q_out = self._lin(f"l{i}_gqa_q", x_2d).reshape(seq_len, GQA_Q_HEADS, GQA_HEAD_DIM * 2)
        query, gate = q_out.chunk(2, dim=-1)  # [S, 6, 256] each
        gate = gate.reshape(seq_len, GQA_Q_HEADS * GQA_HEAD_DIM)  # [S, 1536]

        key = self._lin(f"l{i}_gqa_k", x_2d).reshape(seq_len, GQA_KV_HEADS, GQA_HEAD_DIM)
        value = self._lin(f"l{i}_gqa_v", x_2d).reshape(seq_len, GQA_KV_HEADS, GQA_HEAD_DIM)

        # Per-head RMSNorm
        query = rms_norm(query, q_norm_w)  # [S, 6, 256]
        key = rms_norm(key, k_norm_w)      # [S, 1, 256]

        # PARTIAL RoPE (ROPE_DIM=64 of 256): apply to [..., :64], pass [64:] through.
        positions = torch.arange(seq_len, device=x.device)
        cos = self.rope_cos.squeeze(0).squeeze(0)[positions].unsqueeze(1)  # [S, 1, 64]
        sin = self.rope_sin.squeeze(0).squeeze(0)[positions].unsqueeze(1)  # [S, 1, 64]
        query, key = apply_rope(query, key, cos, sin)  # [S, heads, 256]
        query = query.to(x.dtype)
        key = key.to(x.dtype)

        # Store in KV cache. Cache is batched [gi, B, max_seq, 256]; prefill is BS=1 → row 0.
        kv_k[gi, 0, :seq_len] = key.squeeze(1)  # [S, 256]
        kv_v[gi, 0, :seq_len] = value.squeeze(1)  # [S, 256]

        # Causal attention: [S, 6, 256] @ [S, 1, 256].T -> expand KV heads
        # query [S, 6, 256], key [S, 1, 256]
        # scores[s, h, t] = query[s, h] · key[t, 0] (expand 1->6)
        q_t = query.transpose(0, 1)  # [6, S, 256]
        k_t = key.squeeze(1)  # [S, 256]
        scores = torch.matmul(q_t, k_t.t()) / math.sqrt(GQA_HEAD_DIM)  # [6, S, S]

        # Causal mask
        causal = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=scores.dtype))
        scores = scores + (1.0 - causal) * (-1e9)
        attn_weights = F.softmax(scores.float(), dim=-1).to(torch.bfloat16)  # [6, S, S]

        # Weighted values
        v_t = value.squeeze(1)  # [S, 256]
        attn_out = torch.matmul(attn_weights, v_t)  # [6, S, 256]
        attn_out = attn_out.transpose(0, 1).reshape(seq_len, GQA_Q_HEADS * GQA_HEAD_DIM)  # [S, 1536]

        # Gate + output projection
        attn_out = attn_out * torch.sigmoid(gate)
        out = self._lin(f"l{i}_gqa_o", attn_out.to(torch.bfloat16))  # [S, 5120]
        out = functional_all_reduce(out, "sum", self.tp_group)
        return out.unsqueeze(0)  # [1, S, 5120]

    def _mlp_prefill(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """MLP over full sequence."""
        x_2d = x.squeeze(0).to(torch.bfloat16)  # [S, 5120]
        gate = self._lin(f"l{i}_mlp_gate", x_2d)
        up = self._lin(f"l{i}_mlp_up", x_2d)
        hidden = F.silu(gate) * up
        out = self._lin(f"l{i}_mlp_down", hidden)
        out = functional_all_reduce(out, "sum", self.tp_group)
        return out.unsqueeze(0)  # [1, S, 5120]

    def _deltanet_layer(
        self, i: int, x: torch.Tensor,
        dn_states: torch.Tensor, cv_states: torch.Tensor
    ) -> torch.Tensor:
        """DeltaNet layer for decode (seq_len=1, batch B). The deltanet_full NKI
        kernel is single-token/single-batch, so for B>1 we run it per batch row in
        a Python loop (unrolled at trace time). DeltaNet is NOT the decode bottleneck
        (profile: MLP+GQA+norms dominate), so the loop is acceptable; the projection
        GEMMs are still batched. States: dn_states[di]=[B,1536,128], cv=[B,2560,3]."""
        di = deltanet_index(i)
        B = self.batch_size

        out_w_name = f"l{i}_dn_out"
        conv_w = getattr(self, f"l{i}_dn_conv_w")
        conv_b = getattr(self, f"l{i}_dn_conv_b", None)
        A_log = getattr(self, f"l{i}_dn_A_log")
        dt_bias = getattr(self, f"l{i}_dn_dt_bias")
        norm_w = getattr(self, f"l{i}_dn_norm")

        x_2d = x.squeeze(1).to(torch.bfloat16)  # [B, 5120] bf16

        conv_weight_2d = conv_w.squeeze(1).float()           # [2560, 4]
        if conv_b is not None:
            conv_bias_1d = conv_b.float()                    # [2560]
        else:
            conv_bias_1d = torch.zeros(
                conv_weight_2d.shape[0],
                dtype=torch.float32, device=x_2d.device,
            )

        if self.fp8_weights and B == 1:
            # In-proj GEMMs folded into the kernel — single nki_op call (BS=1 only).
            new_state, new_conv_state, attn_out = torch.ops.deltanet.full_fp8(
                x_2d, dn_states[di][0].float(), cv_states[di][0],
                conv_weight_2d, conv_bias_1d,
                A_log.float(), dt_bias.float(), norm_w.float(),
                getattr(self, f"l{i}_dn_qkv_q"), getattr(self, f"l{i}_dn_qkv_s"),
                getattr(self, f"l{i}_dn_z_q"),   getattr(self, f"l{i}_dn_z_s"),
                getattr(self, f"l{i}_dn_a_q"),   getattr(self, f"l{i}_dn_a_s"),
                getattr(self, f"l{i}_dn_b_q"),   getattr(self, f"l{i}_dn_b_s"),
            )
            dn_states[di][0] = new_state.to(dn_states.dtype)
            cv_states[di][0] = new_conv_state
            attn_out = attn_out.reshape(1, DN_VALUE_DIM)         # [1, 1536]
            out = self._lin(out_w_name, attn_out)                # [1, 5120]
            out = functional_all_reduce(out, "sum", self.tp_group)
            return out.unsqueeze(1)

        # bf16 path: batched in-proj GEMMs.
        mixed_qkv = self._lin(f"l{i}_dn_qkv", x_2d)                       # [B, 2560]
        a_out = self._lin(f"l{i}_dn_a", x_2d).float()                     # [B, 12]
        b_out = self._lin(f"l{i}_dn_b", x_2d).float()                     # [B, 12]
        z = self._lin(f"l{i}_dn_z", x_2d).reshape(B, DN_V_HEADS, DN_V_DIM)  # [B,12,128]

        if B == 1:
            # BS=1: original single-row deltanet.full kernel (byte-identical baseline).
            new_state, new_conv_state, attn_out = torch.ops.deltanet.full(
                dn_states[di][0].float(),
                mixed_qkv[0], cv_states[di][0],
                conv_weight_2d, conv_bias_1d,
                a_out[0], b_out[0], z[0],
                A_log.float(), dt_bias.float(), norm_w.float(),
            )
            dn_states[di][0] = new_state.to(dn_states.dtype)
            cv_states[di][0] = new_conv_state
            attn_out = attn_out.reshape(1, DN_VALUE_DIM).to(torch.bfloat16)
            out = self._lin(out_w_name, attn_out)
            out = functional_all_reduce(out, "sum", self.tp_group)
            return out.unsqueeze(1)

        # B>1: ONE batched deltanet kernel call for the whole batch.
        # Flatten batch into dim 0 to match the kernel's B*-strided layout.
        Vd = DN_VALUE_DIM           # 1536 (V_HEADS*K_DIM)
        Qd = DN_QKV_DIM             # 2560
        state_in = dn_states[di].reshape(B * Vd, DN_V_DIM).float()        # [B*1536,128]
        qkv_in = mixed_qkv.to(torch.bfloat16).reshape(B * Qd)            # [B*2560] bf16
        conv_in = cv_states[di].reshape(B * Qd, DN_CONV_KERNEL - 1)       # [B*2560,3]
        a_in = a_out.reshape(B * DN_V_HEADS)                              # [B*12]
        b_in = b_out.reshape(B * DN_V_HEADS)                              # [B*12]
        z_in = z.to(torch.bfloat16).reshape(B * DN_V_HEADS, DN_V_DIM)     # [B*12,128]

        new_state, new_conv_state, attn_flat = torch.ops.deltanet.full_batched(
            state_in, qkv_in, conv_in,
            conv_weight_2d, conv_bias_1d,
            a_in, b_in, z_in,
            A_log.float(), dt_bias.float(), norm_w.float(),
        )
        dn_states[di] = new_state.reshape(B, Vd, DN_V_DIM).to(dn_states.dtype)
        cv_states[di] = new_conv_state.reshape(B, Qd, DN_CONV_KERNEL - 1).to(cv_states.dtype)

        attn_out = attn_flat.reshape(B, DN_VALUE_DIM).to(torch.bfloat16)  # [B, 1536]
        out = self._lin(out_w_name, attn_out)                            # [B, 5120]
        out = functional_all_reduce(out, "sum", self.tp_group)

        return out.unsqueeze(1)  # [B, 1, 5120]

    def _gqa_layer(
        self, i: int, x: torch.Tensor,
        cos: torch.Tensor, sin: torch.Tensor,
        position: torch.Tensor,
        kv_k: torch.Tensor, kv_v: torch.Tensor,
    ) -> torch.Tensor:
        """GQA attention layer for decode (seq_len=1, batch B). KV cache is
        kv_k/kv_v[gi] = [B, max_seq, 256]."""
        gi = gqa_index(i)
        B = self.batch_size

        q_norm_w = getattr(self, f"l{i}_gqa_q_norm")
        k_norm_w = getattr(self, f"l{i}_gqa_k_norm")

        # x: [B, 1, 5120]
        x_2d = x.squeeze(1).to(torch.bfloat16)  # [B, 5120]

        # Q projection: [B, 3072] = 6 heads * 256 * 2 (query + gate)
        q_out = self._lin(f"l{i}_gqa_q", x_2d)  # [B, 3072]
        q_out = q_out.reshape(B, GQA_Q_HEADS, GQA_HEAD_DIM * 2)
        query, gate = q_out.chunk(2, dim=-1)  # each [B, 6, 256]
        gate = gate.reshape(B, GQA_Q_HEADS * GQA_HEAD_DIM)  # [B, 1536]

        key = self._lin(f"l{i}_gqa_k", x_2d).reshape(B, GQA_KV_HEADS, GQA_HEAD_DIM)
        value = self._lin(f"l{i}_gqa_v", x_2d).reshape(B, GQA_KV_HEADS, GQA_HEAD_DIM)

        # k-side norm+rope always in torch (cheap, single head; the GQA-tail kernel
        # only folds the q-side + attention core). q-side: skip here when GQATAIL is
        # on — the kernel does q RMSNorm + partial RoPE internally (needs PRE-NORM q).
        if not USE_GQA_TAIL:
            query = rms_norm(query, q_norm_w)   # [B, 6, 256]
        key = rms_norm(key, k_norm_w)       # [B, 1, 256]

        key = key.unsqueeze(2)      # [B, 1, 1, 256]
        if USE_GQA_TAIL:
            # only rope the key here; query is roped inside the kernel.
            _, key = apply_rope(key, key, cos, sin)
        else:
            query = query.unsqueeze(2)  # [B, 6, 1, 256]
            query, key = apply_rope(query, key, cos, sin)
            query = query.to(x.dtype).squeeze(2)   # [B, 6, 256]
        key = key.to(x.dtype).squeeze(2)       # [B, 1, 256]

        # key/value: [B, 1, 256] -> [B, 256] (single KV head)
        key_b = key.reshape(B, GQA_HEAD_DIM)      # [B, 256]
        value_b = value.reshape(B, GQA_HEAD_DIM)  # [B, 256]
        query = query.reshape(B, GQA_Q_HEADS, GQA_HEAD_DIM)  # [B, 6, 256] (pre-norm if GQATAIL)

        # Update KV cache at `position` for all B rows. kv_k[gi]: [B, max_seq, 256].
        if USE_LEAN_KV:
            # position is shared across B → contiguous slice write along seq dim.
            # index_copy_(dim=1, index=[position], src=[B,1,256]) lowers to one
            # dynamic-offset DMA: no [B,1,256] index tile to materialize, no
            # indirect-scatter EVENT_SEMAPHORE (the :842/843 BS=8 hotspot).
            pos1 = position.reshape(1)  # [1]
            kv_k[gi].index_copy_(1, pos1, key_b.unsqueeze(1))
            kv_v[gi].index_copy_(1, pos1, value_b.unsqueeze(1))
        else:
            # scatter along seq dim (1): index [B,1,256] broadcast of position; src [B,1,256].
            pos_idx = position.reshape(1, 1, 1).expand(B, 1, GQA_HEAD_DIM)  # [B,1,256]
            kv_k[gi].scatter_(1, pos_idx, key_b.unsqueeze(1))
            kv_v[gi].scatter_(1, pos_idx, value_b.unsqueeze(1))

        cached_k = kv_k[gi]  # [B, max_seq, 256]
        cached_v = kv_v[gi]  # [B, max_seq, 256]

        # Causal mask: attend to positions <= current (shared position across batch)
        pos_range = torch.arange(self.max_seq_len, device=x.device)
        mask = (pos_range <= position)  # [max_seq] bool

        if USE_GQA_TAIL:
            # ONE custom call folds q RMSNorm + partial RoPE + scaled scores + masked
            # softmax + weighted-V + output-gate. Pass PRE-NORM query, raw gate, the
            # already-KV-written cache flattened to [B*S,256], 64-dim cos/sin row.
            S = self.max_seq_len
            attn_flat = torch.ops.gqa.tail(
                query.reshape(B * GQA_Q_HEADS, GQA_HEAD_DIM).float(),
                gate.reshape(B * GQA_Q_HEADS, GQA_HEAD_DIM).float(),
                q_norm_w.reshape(1, GQA_HEAD_DIM).float(),
                cos.reshape(1, ROPE_DIM).float(),
                sin.reshape(1, ROPE_DIM).float(),
                cached_k.reshape(B * S, GQA_HEAD_DIM).float(),
                cached_v.reshape(B * S, GQA_HEAD_DIM).float(),
                mask.reshape(1, S).float(),
            )  # [B*6, 256]
            attn_out = attn_flat.reshape(B, GQA_Q_HEADS * GQA_HEAD_DIM).to(torch.bfloat16)  # [B,1536]
        else:
            # Scores: [B,6,256] @ [B,256,max_seq] -> [B,6,max_seq] (batched matmul)
            scores = torch.matmul(query, cached_k.transpose(1, 2))  # [B, 6, max_seq]
            scores = scores / math.sqrt(GQA_HEAD_DIM)
            scores = scores + (1.0 - mask.to(scores.dtype)).reshape(1, 1, -1) * (-1e9)
            attn_weights = F.softmax(scores.float(), dim=-1).to(x.dtype)  # [B, 6, max_seq]
            # Weighted sum: [B,6,max_seq] @ [B,max_seq,256] -> [B,6,256]
            attn_out = torch.matmul(attn_weights, cached_v)  # [B, 6, 256]
            attn_out = attn_out.reshape(B, GQA_Q_HEADS * GQA_HEAD_DIM)  # [B, 1536]
            attn_out = attn_out * torch.sigmoid(gate)

        out = self._lin(f"l{i}_gqa_o", attn_out.to(torch.bfloat16))  # [B, 5120]
        out = functional_all_reduce(out, "sum", self.tp_group)

        return out.unsqueeze(1)  # [B, 1, 5120]

    def _mlp_layer(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU MLP."""
        x_2d = x.squeeze(1).to(torch.bfloat16)  # [1, 5120]

        gate = self._lin(f"l{i}_mlp_gate", x_2d)  # [1, 4352]
        up = self._lin(f"l{i}_mlp_up", x_2d)      # [1, 4352]
        hidden = F.silu(gate) * up
        out = self._lin(f"l{i}_mlp_down", hidden)  # [1, 5120]

        # All-reduce after rowwise down_proj
        out = functional_all_reduce(out, "sum", self.tp_group)

        return out.unsqueeze(1)  # [1, 1, 5120]


# ─── Model Loading & Sharding (reuse from run_qwen3_6_27b.py) ────────────────

def shard_linear_colwise(linear, rank, world_size):
    out_dim = linear.weight.shape[0]
    shard = out_dim // world_size
    w = linear.weight.data[rank * shard:(rank + 1) * shard].clone()
    linear.weight = nn.Parameter(w)
    if linear.bias is not None:
        b = linear.bias.data[rank * shard:(rank + 1) * shard].clone()
        linear.bias = nn.Parameter(b)
    linear.out_features = shard


def shard_linear_rowwise(linear, rank, world_size):
    in_dim = linear.weight.shape[1]
    shard = in_dim // world_size
    w = linear.weight.data[:, rank * shard:(rank + 1) * shard].clone()
    linear.weight = nn.Parameter(w)
    linear.in_features = shard


def shard_conv1d_for_qkv(conv, rank, world_size, key_dim, value_dim):
    """Shard depthwise conv1d matching the [q, k, v] group structure."""
    C = conv.weight.shape[0]  # = key_dim + key_dim + value_dim
    q_shard = key_dim // world_size
    v_shard = value_dim // world_size
    # Pick channels matching the QKV group sharding
    q_idx = slice(rank*q_shard, (rank+1)*q_shard)
    k_idx = slice(key_dim + rank*q_shard, key_dim + (rank+1)*q_shard)
    v_idx = slice(2*key_dim + rank*v_shard, 2*key_dim + (rank+1)*v_shard)
    indices = list(range(C))[q_idx] + list(range(C))[k_idx] + list(range(C))[v_idx]
    indices_t = torch.tensor(indices, dtype=torch.long)
    
    total = q_shard + q_shard + v_shard
    new_conv = nn.Conv1d(total, total, conv.kernel_size[0],
                         padding=conv.padding[0], groups=total,
                         bias=conv.bias is not None)
    new_conv.weight = nn.Parameter(conv.weight.data[indices_t].clone())
    if conv.bias is not None:
        new_conv.bias = nn.Parameter(conv.bias.data[indices_t].clone())
    return new_conv


def shard_qkv_colwise(linear, rank, world_size, key_dim, value_dim):
    """Shard in_proj_qkv respecting [q(key_dim), k(key_dim), v(value_dim)] structure."""
    w = linear.weight.data  # [key_dim + key_dim + value_dim, hidden]
    q_w = w[:key_dim]
    k_w = w[key_dim:2*key_dim]
    v_w = w[2*key_dim:]
    # Shard each group independently
    q_shard = key_dim // world_size
    v_shard = value_dim // world_size
    q_part = q_w[rank*q_shard:(rank+1)*q_shard]
    k_part = k_w[rank*q_shard:(rank+1)*q_shard]
    v_part = v_w[rank*v_shard:(rank+1)*v_shard]
    linear.weight = nn.Parameter(torch.cat([q_part, k_part, v_part], dim=0).clone())
    linear.out_features = q_shard + q_shard + v_shard


def shard_model(model, rank, world_size):
    """Apply TP=4 sharding to the model (on CPU, before weight extraction)."""
    lang = model.model.language_model
    for i, layer in enumerate(lang.layers):
        if layer_type(i) == "deltanet":
            attn = layer.linear_attn
            shard_qkv_colwise(attn.in_proj_qkv, rank, world_size,
                              key_dim=attn.key_dim, value_dim=attn.value_dim)
            attn.conv1d = shard_conv1d_for_qkv(attn.conv1d, rank, world_size,
                                               key_dim=attn.key_dim, value_dim=attn.value_dim)
            shard_linear_colwise(attn.in_proj_z, rank, world_size)
            shard_linear_colwise(attn.in_proj_a, rank, world_size)
            shard_linear_colwise(attn.in_proj_b, rank, world_size)
            shard_linear_rowwise(attn.out_proj, rank, world_size)
            # Shard A_log and dt_bias: [48] → [12] per core
            n_heads = attn.A_log.shape[0]
            shard_h = n_heads // world_size
            attn.A_log = nn.Parameter(attn.A_log.data[rank * shard_h:(rank + 1) * shard_h].clone())
            attn.dt_bias = nn.Parameter(attn.dt_bias.data[rank * shard_h:(rank + 1) * shard_h].clone())
        else:
            attn = layer.self_attn
            shard_linear_colwise(attn.q_proj, rank, world_size)
            shard_linear_colwise(attn.k_proj, rank, world_size)
            shard_linear_colwise(attn.v_proj, rank, world_size)
            shard_linear_rowwise(attn.o_proj, rank, world_size)

        shard_linear_colwise(layer.mlp.gate_proj, rank, world_size)
        shard_linear_colwise(layer.mlp.up_proj, rank, world_size)
        shard_linear_rowwise(layer.mlp.down_proj, rank, world_size)

    return model


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=None,
                        help="Tiny-mode synthetic prompt length (# tokens) for prefill "
                             "benchmarking. Default 5. Must be <= max-seq-len.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Decode batch size B (throughput). B>1 batches the MLP/GQA "
                             "projections natively (same weights serve B tokens — "
                             "bandwidth-bound decode scales sub-linearly) and loops the "
                             "BS=1 deltanet_full kernel B times (not the bottleneck). "
                             "BS=1 keeps the exact original single-token graph.")
    parser.add_argument("--skip-compile", action="store_true", help="Test eager first")
    parser.add_argument("--skip-prefill", action="store_true",
                        help="Skip the (currently broken) chunked prefill compile and "
                             "benchmark decode on zero-seeded state. Decode-step TPOT is "
                             "structural (independent of state values), so this isolates "
                             "the decode lever (NORMFUSE/fp8) from the prefill bug.")
    parser.add_argument("--compile-prefill", action="store_true",
                        help="torch.compile(backend='neuron', fullgraph=True) the prefill "
                             "forward and report COLD compile + WARM synced prompt tok/s. "
                             "Requires CHUNKEDPREFILL=1 (the NKI chunked-DeltaNet kernel) "
                             "so the DeltaNet recurrence is one compilable custom call/layer "
                             "instead of the un-compilable torch chunk-loop. --prompt-len "
                             "must be a multiple of CHUNK_SIZE (64).")
    parser.add_argument("--fp8-weights", action="store_true",
                        help="Quantize Linear weights to FP8 E4M3 (W8A16). "
                             "Halves weight bandwidth, the dominant cost in decode.")
    parser.add_argument("--tiny", action="store_true",
                        help="Fast-iteration mode: random, full-WIDTH, few-LAYER "
                             "checkpoint. Skips the real 27B load+shard+tokenizer. "
                             "Preserves the exact per-layer RMSNorm/TP-collective "
                             "graph pattern for norm-fusion validation. Numerics "
                             "are self-consistent only, not bit-exact vs real model.")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override layer count (tiny mode only). Must be a "
                             "multiple of 4 to keep the [DeltaNet×3, GQA×1] pattern.")
    args = parser.parse_args()

    # ── tiny mode: shrink NUM_LAYERS (and derived counts) before building buffers
    global NUM_LAYERS, NUM_DELTANET, NUM_GQA, VOCAB
    if args.tiny:
        nl = args.num_layers if args.num_layers is not None else 4
        assert nl % 4 == 0, "--num-layers must be a multiple of 4 ([DeltaNet×3, GQA×1])"
        NUM_LAYERS = nl
        NUM_DELTANET = sum(1 for i in range(nl) if layer_type(i) == "deltanet")
        NUM_GQA = sum(1 for i in range(nl) if layer_type(i) == "gqa")
        VOCAB = 4096  # shrink embed/lm_head; decode graph shape is layer-driven

    import torch_neuronx

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.neuron.current_device()

    if rank == 0:
        tag = f"TINY({NUM_LAYERS}L, full-width)" if args.tiny else "Qwen 3.6 27B"
        print(f"=== Static Decode: {tag}, TP={world_size}, max_seq={args.max_seq_len} ===")

    if args.tiny:
        # Build the already-sharded per-core weight dict directly — no HF load,
        # no shard_model, no tokenizer. Per-core shapes match production exactly.
        t0 = time.time()
        weights = build_random_sharded_weights(NUM_LAYERS, seed=1234 + rank)
        if rank == 0:
            print(f"  Random {NUM_LAYERS}-layer weights built: {time.time()-t0:.1f}s")
    else:
        from transformers import Qwen3_5ForConditionalGeneration
        # 1. Load model on CPU
        t0 = time.time()
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        if rank == 0:
            print(f"  Model loaded: {time.time()-t0:.1f}s")

        # 2. Shard on CPU
        t0 = time.time()
        model = shard_model(model, rank, world_size)
        if rank == 0:
            print(f"  Sharded: {time.time()-t0:.1f}s")

        # 3. Extract weights
        weights = extract_weights(model)
        del model  # Free CPU memory

    # 4. Create static decode module and move to device
    t0 = time.time()
    static_module = StaticDecodeModule(
        weights, args.max_seq_len, world_size, fp8_weights=args.fp8_weights,
        batch_size=args.batch_size,
    )
    del weights
    static_module = static_module.to(device)
    static_module.eval()
    if rank == 0:
        mem_mb = sum(b.numel() * b.element_size() for b in static_module.buffers()) / 1e6
        print(f"  Static module on device: {time.time()-t0:.1f}s ({mem_mb:.0f} MB/core)")

    # 5. Initialize state buffers (batched: leading B dim after the layer dim)
    dtype = torch.bfloat16
    B = args.batch_size
    # DNF32STATE: keep DeltaNet recurrent state in f32 so the per-layer cast round-trip
    # around the kernel call vanishes (Phase 2 glue lever; conv_states stay bf16 — they
    # feed the conv, not the f32 recurrence). The kernel reads/writes f32 state already.
    dn_state_dtype = torch.float32 if USE_DN_F32_STATE else dtype
    dn_states = torch.zeros(NUM_DELTANET, B, DN_V_HEADS * DN_K_DIM, DN_V_DIM,
                            dtype=dn_state_dtype, device=device)
    conv_states = torch.zeros(NUM_DELTANET, B, DN_QKV_DIM, DN_CONV_KERNEL - 1,
                              dtype=dtype, device=device)
    kv_k = torch.zeros(NUM_GQA, B, args.max_seq_len, GQA_HEAD_DIM,
                        dtype=dtype, device=device)
    kv_v = torch.zeros(NUM_GQA, B, args.max_seq_len, GQA_HEAD_DIM,
                        dtype=dtype, device=device)

    # 6. Compile
    if not args.skip_compile:
        if rank == 0:
            print("  Compiling with backend='neuron', fullgraph=True...")
        t0 = time.time()
        static_module.forward = torch.compile(
            static_module.forward, backend="neuron", fullgraph=True, dynamic=False
        )
        if rank == 0:
            print(f"  torch.compile setup: {time.time()-t0:.1f}s")

    # 7. Tokenize prompt (tiny mode: synthetic ids, no tokenizer/detokenize)
    tokenizer = None
    if args.tiny:
        plen = getattr(args, "prompt_len", None) or 5
        # arbitrary valid ids < VOCAB; vary length for prefill benchmarking
        input_ids = [(i % (VOCAB - 1)) + 1 for i in range(plen)]
        if rank == 0:
            print(f"  Synthetic prompt ({len(input_ids)} tokens, tiny mode)")
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompt = "The meaning of life is"
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if rank == 0:
            print(f"  Prompt: '{prompt}' ({len(input_ids)} tokens)")

    dist.barrier()

    # 8. Run generation loop
    # Prefill: process all prompt tokens at once (eager, builds state)
    if rank == 0:
        print("  Prefilling prompt...")

    t0 = time.time()
    # On-device greedy: keep the next token id and position as device tensors
    # across the whole loop so there is NO per-token device->host->device sync.
    # We collect each step's argmax (still on device) and sync to host ONCE at
    # the end for detokenization.
    gen_ids_dev = []   # list of [1] device tensors, one per generated token

    with torch.no_grad():
        # Prefill prompt.
        # --skip-prefill: the prefill chunked-DeltaNet kernel currently fails to
        # compile (chunked_prefill.py:135 strided in-place A[...,i,:i]= -> "Failed
        # to merge transformation chain"). The decode-step TPOT we A/B is purely
        # structural: latency is independent of STATE VALUES (states are already
        # zero-allocated at 984-990). So skip prefill, seed a valid token at
        # position 0, and run the decode benchmark on zero-seeded state. This
        # isolates the decode lever (NORMFUSE / fp8-W8A16) from the prefill bug.
        if getattr(args, "skip_prefill", False):
            if rank == 0:
                print("  [skip-prefill] seeding zero state; benchmarking decode only")
            # B-wide seed: same token across batch rows (tiny-mode gen is not scored)
            next_id = torch.full((B,), input_ids[0], dtype=torch.long, device=device)
            gen_ids_dev.append(next_id)
            position = torch.tensor(0, dtype=torch.long, device=device)
        else:
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
            S = len(input_ids)

            # --compile-prefill: wrap prefill in torch.compile(backend="neuron",
            # fullgraph=True) for a REAL (device-compute) prompt-throughput number.
            # Requires the chunked-DeltaNet NKI kernel (CHUNKEDPREFILL=1) so the
            # recurrence is one compilable custom call/layer; the eager torch
            # chunk-loop does NOT survive fullgraph capture under backend="neuron".
            if getattr(args, "compile_prefill", False):
                if not USE_CHUNKED_PREFILL:
                    raise SystemExit(
                        "--compile-prefill requires CHUNKEDPREFILL=1 (the eager torch "
                        "chunk-loop can't compile under backend='neuron')."
                    )
                if S % CHUNK_SIZE != 0:
                    raise SystemExit(
                        f"--compile-prefill needs --prompt-len a multiple of CHUNK_SIZE "
                        f"({CHUNK_SIZE}); got S={S}."
                    )
                if rank == 0:
                    print(f"  Compiling prefill (backend='neuron', fullgraph=True, "
                          f"chunked DeltaNet C={CHUNK_SIZE})...")
                static_module.prefill = torch.compile(
                    static_module.prefill, backend="neuron", fullgraph=True, dynamic=False
                )

            # COLD prefill: first call includes eager-graph trace + neuronx-cc compile.
            logits, dn_c, cv_c, kvk_c, kvv_c = static_module.prefill(
                input_tensor, dn_states.clone(), conv_states.clone(),
                kv_k.clone(), kv_v.clone()
            )
            _ = logits[0, 0].item()  # device->host sync to finish the cold run
            if rank == 0:
                cold_t = time.time() - t0
                print(f"  Prefill COLD (compile+exec): {cold_t:.2f}s  (S={S})")

            # WARM steady-state prefill: re-run on the cached NEFF, synced, averaged.
            import time as _t
            warm_iters = 5
            _w0 = _t.time()
            for _w in range(warm_iters):
                logits, dn_w, cv_w, kvk_w, kvv_w = static_module.prefill(
                    input_tensor, dn_states.clone(), conv_states.clone(),
                    kv_k.clone(), kv_v.clone()
                )
            _ = logits[0, 0].item()  # single sync after the warm batch
            if rank == 0:
                warm_ms = (_t.time() - _w0) / warm_iters * 1000.0
                tput = S / (warm_ms / 1000.0)
                print(f"  Prefill WARM (synced): {warm_ms:.1f} ms/forward  |  "
                      f"{tput:.0f} prompt tok/s  (S={S}, BS=1)")

            # Real prefill to seed decode state.
            logits, dn_states, conv_states, kv_k, kv_v = static_module.prefill(
                input_tensor, dn_states, conv_states, kv_k, kv_v
            )
            # prefill is BS=1; broadcast its next token across the B decode rows
            nid0 = logits[0, :].argmax().to(torch.long)
            next_id = nid0.reshape(1).expand(B).contiguous()  # [B] on device
            gen_ids_dev.append(next_id)

            if rank == 0:
                prefill_time = time.time() - t0
                print(f"  Prefill (total incl cold+warm): {prefill_time:.2f}s")

            # position lives on device and is incremented with a device-side add
            position = torch.tensor(len(input_ids), dtype=torch.long, device=device)
        one = torch.tensor(1, dtype=torch.long, device=device)

        # Decode remaining tokens — next_id [B] flows straight back in, never to host
        for step in range(args.num_tokens - 1):
            logits, dn_states, conv_states, kv_k, kv_v = static_module(
                next_id, position, dn_states, conv_states, kv_k, kv_v
            )
            next_id = logits.argmax(dim=-1).to(torch.long)  # [B] on device
            gen_ids_dev.append(next_id)
            position = position + one  # device-side increment, no host sync

    # Single device->host sync for the entire generation. gen_ids_dev holds [B]
    # tensors; row 0 is the reportable sequence (all rows identical in tiny mode).
    gen_stack = torch.stack(gen_ids_dev)  # [n_gen, B]
    generated = gen_stack[:, 0].reshape(-1).cpu().tolist()
    last_id_b = gen_stack[-1].clone()     # [B] on device — the last generated tokens
    elapsed = time.time() - t0

    if rank == 0:
        n_gen = len(generated)
        print(f"\n  Total: {elapsed:.2f}s for {len(input_ids)} prompt + {n_gen} generated tokens (B={B})")
        print(f"  TPOT (approx): {elapsed / (len(input_ids) + n_gen) * 1000:.1f} ms")
        if tokenizer is not None:
            gen_text = tokenizer.decode(generated, skip_special_tokens=True)
            print(f"  First token: '{tokenizer.decode([generated[0]])}'")
            print(f"  Generated: {gen_text}")
        else:
            print(f"  Generated token ids (tiny mode, row0): {generated}")

        # Steady-state benchmark (just decode tokens, after warmup)
        print("\n  Benchmarking steady-state decode...")

    dist.barrier()

    n_bench = 100
    start_id = last_id_b.to(device)  # [B]
    start_pos = len(input_ids) + len(generated)

    # ── PROFILE_STEPS: minimal isolated decode trace for neuron-explorer ──
    # When set, run a few warmup steps then PROFILE_STEPS constant-token decode
    # steps (synchronized) and EXIT before the 400-step 4-variant bench. With
    # NEURON_RT_INSPECT_DEVICE_PROFILE=1 this writes a small NTFF containing only
    # the decode NEFF executions, so the per-op attribution isn't buried under
    # warmup/prefill graphs. Constant token+pos = same decode NEFF every step.
    profile_steps = int(os.environ.get("PROFILE_STEPS", "0"))
    if profile_steps > 0:
        const_it = torch.full((B,), generated[-1], dtype=torch.long, device=device)
        const_pos = torch.tensor(start_pos, dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(3):  # warmup (not traced meaningfully)
                logits, dn_states, conv_states, kv_k, kv_v = static_module(
                    const_it, const_pos, dn_states, conv_states, kv_k, kv_v
                )
            torch.neuron.synchronize()
        dist.barrier()
        with torch.no_grad():
            for _ in range(profile_steps):
                logits, dn_states, conv_states, kv_k, kv_v = static_module(
                    const_it, const_pos, dn_states, conv_states, kv_k, kv_v
                )
            torch.neuron.synchronize()
        if rank == 0:
            print(f"  [profile] ran {profile_steps} decode steps for tracing; exiting")
        dist.barrier()
        dist.destroy_process_group()
        return

    # ── Variant A: per-token host sync (the OLD loop: .item() + re-tensor) ──
    with torch.no_grad():
        for _ in range(3):  # warm up
            it = torch.full((B,), generated[-1], dtype=torch.long, device=device)
            pos = torch.tensor(start_pos, dtype=torch.long, device=device)
            logits, dn_states, conv_states, kv_k, kv_v = static_module(
                it, pos, dn_states, conv_states, kv_k, kv_v
            )
    dist.barrier()
    t0 = time.time()
    with torch.no_grad():
        tok = generated[-1]
        for step in range(n_bench):
            it = torch.full((B,), tok, dtype=torch.long, device=device)        # H2D [B]
            pos = torch.tensor(start_pos + step, dtype=torch.long, device=device)  # H2D
            logits, dn_states, conv_states, kv_k, kv_v = static_module(
                it, pos, dn_states, conv_states, kv_k, kv_v
            )
            tok = logits[0, :].argmax().item()  # D2H sync every token (row 0)
    elapsed_hostsync = time.time() - t0

    dist.barrier()

    # ── Variant B: on-device greedy (token + position stay on device) ──
    with torch.no_grad():
        nid = start_id.clone()
        pos = torch.tensor(start_pos, dtype=torch.long, device=device)
        one = torch.tensor(1, dtype=torch.long, device=device)
        for _ in range(3):  # warm up
            logits, dn_states, conv_states, kv_k, kv_v = static_module(
                nid, pos, dn_states, conv_states, kv_k, kv_v
            )
    dist.barrier()
    t0 = time.time()
    with torch.no_grad():
        nid = start_id.clone()
        pos = torch.tensor(start_pos, dtype=torch.long, device=device)
        for step in range(n_bench):
            logits, dn_states, conv_states, kv_k, kv_v = static_module(
                nid, pos, dn_states, conv_states, kv_k, kv_v
            )
            nid = logits.argmax(dim=-1).to(torch.long)  # [B] stays on device
            pos = pos + one
    elapsed_ondevice = time.time() - t0

    dist.barrier()

    # ── Variant C: OLD buggy pattern — constant token, output discarded ──
    # Measured twice: (C1) no synchronize (what the old bench did) and
    # (C2) with torch.neuron.synchronize() before stopping the timer.
    # If C1 << C2, the old 2 ms number was async-dispatch enqueue time,
    # not real execution time.
    const_it = torch.full((B,), generated[-1], dtype=torch.long, device=device)
    const_pos = torch.tensor(start_pos, dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(3):  # warm up
            logits, dn_states, conv_states, kv_k, kv_v = static_module(
                const_it, const_pos, dn_states, conv_states, kv_k, kv_v
            )
        torch.neuron.synchronize()
    dist.barrier()

    # C1: no synchronize (reproduces the old measurement)
    t0 = time.time()
    with torch.no_grad():
        for step in range(n_bench):
            logits, dn_states, conv_states, kv_k, kv_v = static_module(
                const_it, const_pos, dn_states, conv_states, kv_k, kv_v
            )
    elapsed_c1_nosync = time.time() - t0

    # C2: same loop, but synchronize before stopping the timer
    dist.barrier()
    t0 = time.time()
    with torch.no_grad():
        for step in range(n_bench):
            logits, dn_states, conv_states, kv_k, kv_v = static_module(
                const_it, const_pos, dn_states, conv_states, kv_k, kv_v
            )
        torch.neuron.synchronize()
    elapsed_c2_sync = time.time() - t0

    if rank == 0:
        tpot_host = elapsed_hostsync / n_bench * 1000
        tpot_dev = elapsed_ondevice / n_bench * 1000
        tpot_c1 = elapsed_c1_nosync / n_bench * 1000
        tpot_c2 = elapsed_c2_sync / n_bench * 1000
        speedup = (tpot_host / tpot_dev) if tpot_dev > 0 else 0.0
        print("\n  === Greedy decode loop: host-sync vs on-device (same NEFF) ===")
        print(f"  A) per-token host sync (.item()):  {tpot_host:.2f} ms/tok  ({n_bench} steps)")
        print(f"  B) on-device greedy (no sync):     {tpot_dev:.2f} ms/tok  ({n_bench} steps)")
        print(f"  Speedup B vs A: {speedup:.2f}x  ({(speedup-1)*100:+.1f}%)")
        print("\n  === Async-dispatch artifact test (constant token, output discarded) ===")
        print(f"  C1) no synchronize (OLD bench style): {tpot_c1:.2f} ms/tok  ({n_bench} steps)")
        print(f"  C2) + torch.neuron.synchronize():     {tpot_c2:.2f} ms/tok  ({n_bench} steps)")
        print(f"  C2/C1 ratio: {tpot_c2/tpot_c1 if tpot_c1>0 else 0:.1f}x  "
              f"(>>1 means C1 measured enqueue time, NOT execution)")
        print(f"\n  TRUE steady-state TPOT (synced): {tpot_c2:.1f} ms — "
              f"{'PASS ✓' if tpot_c2 < 100 else 'NEEDS OPTIMIZATION'} (target <100 ms)")
        # Throughput: B tokens produced per synced step. tok/s = B * 1000 / TPOT_ms.
        thpt_c2 = (B * 1000.0 / tpot_c2) if tpot_c2 > 0 else 0.0
        thpt_host = (B * 1000.0 / tpot_host) if tpot_host > 0 else 0.0
        print(f"  THROUGHPUT (B={B}): {thpt_c2:.1f} tok/s synced  |  "
              f"{thpt_host:.1f} tok/s host-sync  |  per-token synced {tpot_c2/B:.2f} ms")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
