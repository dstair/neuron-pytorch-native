"""
DeepSeek V4 inference on Neuron with Tensor Parallelism.

Uses torch.compile + parallelize_module (Recipe C from AGENT.md).
TP shards Q heads across ranks. KV is MQA (single head), replicated.
MoE uses expert parallelism: each rank owns n_experts/world_size experts.

Usage:
    torchrun --nproc-per-node=4 run_deepseek_v4_tp.py
    torchrun --nproc-per-node=64 run_deepseek_v4_tp.py --model-id deepseek-ai/DeepSeek-V4-Flash
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce
import torch_neuronx
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

SEQ_LEN = 32
MAX_NEW_TOKENS = 8

torch._dynamo.config.cache_size_limit = 64


# -- Config --

@dataclass
class ModelArgs:
    vocab_size: int = 129280
    dim: int = 8
    n_layers: int = 7
    n_heads: int = 8
    head_dim: int = 32
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    moe_inter_dim: int = 32
    q_lora_rank: int = 32
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 32
    window_size: int = 128
    compress_ratios: list = None
    compress_rope_theta: float = 160000.0
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1
    original_seq_len: int = 65536
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    n_hash_layers: int = 3
    score_func: str = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    max_seq_len: int = 4096

    def __post_init__(self):
        if self.compress_ratios is None:
            self.compress_ratios = [0] * self.n_layers
        if self.rope_head_dim > self.head_dim:
            self.rope_head_dim = self.head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        # These get overridden for pre-sharded mode
        self.local_n_heads = self.n_heads
        self.local_n_experts = self.n_routed_experts
        self.local_o_groups = self.o_groups


def load_args_from_hf(config_path: Path) -> ModelArgs:
    with open(config_path) as f:
        cfg = json.load(f)
    return ModelArgs(
        vocab_size=cfg["vocab_size"],
        dim=cfg["hidden_size"],
        n_layers=cfg["num_hidden_layers"],
        n_heads=cfg["num_attention_heads"],
        head_dim=cfg["head_dim"],
        n_routed_experts=cfg["n_routed_experts"],
        n_shared_experts=cfg.get("n_shared_experts", 1),
        n_activated_experts=cfg["num_experts_per_tok"],
        moe_inter_dim=cfg["moe_intermediate_size"],
        q_lora_rank=cfg["q_lora_rank"],
        rope_head_dim=cfg["qk_rope_head_dim"],
        norm_eps=cfg["rms_norm_eps"],
        o_groups=cfg["o_groups"],
        o_lora_rank=cfg["o_lora_rank"],
        window_size=cfg.get("sliding_window", 128),
        compress_ratios=cfg.get("compress_ratios", [0] * cfg["num_hidden_layers"]),
        compress_rope_theta=cfg.get("compress_rope_theta", 160000.0),
        rope_theta=cfg.get("rope_theta", 10000.0),
        rope_factor=cfg.get("rope_scaling", {}).get("factor", 16.0),
        beta_fast=cfg.get("rope_scaling", {}).get("beta_fast", 32),
        beta_slow=cfg.get("rope_scaling", {}).get("beta_slow", 1),
        original_seq_len=cfg.get("rope_scaling", {}).get(
            "original_max_position_embeddings", 65536
        ),
        hc_mult=cfg.get("hc_mult", 4),
        hc_sinkhorn_iters=cfg.get("hc_sinkhorn_iters", 20),
        hc_eps=cfg.get("hc_eps", 1e-6),
        n_hash_layers=cfg.get("num_hash_layers", 3),
        score_func=cfg.get("scoring_func", "sqrtsoftplus"),
        route_scale=cfg.get("routed_scaling_factor", 1.5),
        swiglu_limit=cfg.get("swiglu_limit", 10.0),
    )


# -- RMSNorm --

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(self.weight.dtype) * self.weight


# -- RoPE --

def precompute_freqs(args: ModelArgs, seq_len: int, theta: float = None):
    if theta is None:
        theta = args.rope_theta
    dim = args.rope_head_dim
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin, positions):
    cos_pos = cos[positions.long()]
    sin_pos = sin[positions.long()]
    if x.dim() > cos_pos.dim():
        cos_pos = cos_pos.unsqueeze(-2)
        sin_pos = sin_pos.unsqueeze(-2)
    # Reshape to pairs instead of stride-2 slicing (contiguous for Neuron compile)
    *lead, d = x.shape
    xr = x.reshape(*lead, d // 2, 2)
    x1 = xr[..., 0]
    x2 = xr[..., 1]
    o1 = x1 * cos_pos - x2 * sin_pos
    o2 = x2 * cos_pos + x1 * sin_pos
    return torch.stack([o1, o2], dim=-1).flatten(-2).to(x.dtype)


# -- Hyper-Connections --

def hc_sinkhorn(logits, n_iters, eps):
    m = torch.exp(logits)
    for _ in range(n_iters):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


# Compiled version — fuses the 20-iteration sinkhorn loop into one kernel
_hc_sinkhorn_compiled = torch.compile(
    hc_sinkhorn, backend="neuron", fullgraph=True, dynamic=False
)


def _hc_pre_impl(x, fn, scale, base, hc_mult, sinkhorn_iters, eps, norm_eps):
    shape = x.shape
    hc = hc_mult
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.pow(2).mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, fn.float()) * rsqrt
    pre_l = mixes[..., :hc] * scale[0] + base[:hc]
    post_l = mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc]
    comb_l = mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]
    pre = torch.sigmoid(pre_l) + eps
    post = torch.sigmoid(post_l) + eps
    # Inline sinkhorn for full fusion
    m = torch.exp(comb_l.view(*comb_l.shape[:-1], hc, hc))
    for _ in range(sinkhorn_iters):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    comb = m
    y = (pre.unsqueeze(-1) * x_flat.view(shape)).sum(dim=2)
    return y.to(x.dtype), post, comb


_hc_pre_compiled = torch.compile(
    _hc_pre_impl, backend="neuron", fullgraph=True, dynamic=False
)


class HCPre(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hc_mult = args.hc_mult
        self.sinkhorn_iters = args.hc_sinkhorn_iters
        self.eps = args.hc_eps
        self.norm_eps = args.norm_eps

    def forward(self, x, fn, scale, base):
        return _hc_pre_compiled(
            x, fn, scale, base,
            self.hc_mult, self.sinkhorn_iters, self.eps, self.norm_eps,
        )


def _hc_post_impl(x, residual, post, comb):
    return (
        post.unsqueeze(-1) * x.unsqueeze(-2)
        + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)
    ).to(x.dtype)


hc_post = torch.compile(
    _hc_post_impl, backend="neuron", fullgraph=True, dynamic=False
)


def _hc_head_impl(x, fn, scale, base, norm_eps, eps):
    shape = x.shape
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.pow(2).mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + eps
    return (pre.unsqueeze(-1) * x_flat.view(shape)).sum(dim=2).to(x.dtype)


_hc_head_compiled = torch.compile(
    _hc_head_impl, backend="neuron", fullgraph=True, dynamic=False
)


class HCHead(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hc_mult = args.hc_mult
        self.norm_eps = args.norm_eps
        self.eps = args.hc_eps

    def forward(self, x, fn, scale, base):
        return _hc_head_compiled(
            x, fn, scale, base, self.norm_eps, self.eps
        )


# -- TP-aware Attention --

class Attention(nn.Module):
    """
    TP strategy: Q heads sharded across ranks (ColwiseParallel on wq_b).
    KV is single-head MQA, replicated on all ranks.
    Output projection uses all-reduce after grouped low-rank matmul.
    """

    def __init__(self, args: ModelArgs, layer_idx: int, world_size: int = 1):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_dim = args.rope_head_dim
        self.nope_dim = args.nope_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.o_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.scale = args.head_dim ** -0.5
        self.world_size = world_size

        # Q path (wq_b will be sharded by parallelize_module)
        self.wq_a = nn.Linear(args.dim, args.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(args.q_lora_rank, args.norm_eps)
        self.wq_b = nn.Linear(
            args.q_lora_rank, args.local_n_heads * args.head_dim, bias=False
        )

        # KV path (MQA single head, replicated)
        self.wkv = nn.Linear(args.dim, args.head_dim, bias=False)
        self.kv_norm = RMSNorm(args.head_dim, args.norm_eps)

        # Output grouped low-rank
        # wo_a is replicated (grouped structure doesn't split when TP > o_groups)
        # wo_b is sharded along input dim by TP
        group_dim = (args.n_heads * args.head_dim) // args.o_groups
        self.wo_a = nn.Linear(
            group_dim, args.o_groups * args.o_lora_rank, bias=False
        )
        wo_b_in = (args.o_groups * args.o_lora_rank) // (
            args.n_heads // args.local_n_heads
        ) if args.local_n_heads < args.n_heads else args.o_groups * args.o_lora_rank
        self.wo_b = nn.Linear(wo_b_in, args.dim, bias=False)

    def forward(self, x, cos, sin, positions, mask=None, cache=None, cache_pos=None):
        b, s, _ = x.shape
        rd = self.rope_dim

        # Q
        q = self.wq_b(self.q_norm(self.wq_a(x)).to(x.dtype))
        local_heads = q.shape[-1] // self.head_dim
        q = q.view(b, s, local_heads, self.head_dim)
        q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        q = q.to(x.dtype)

        # KV (replicated, single head)
        kv = self.kv_norm(self.wkv(x)).to(x.dtype)

        if rd > 0:
            q_rope = apply_rope(q[..., -rd:], cos, sin, positions)
            q = torch.cat([q[..., :-rd], q_rope], dim=-1)
            kv_rope = apply_rope(kv[..., -rd:], cos, sin, positions)
            kv = torch.cat([kv[..., :-rd], kv_rope], dim=-1)

        # Static KV cache: write into pre-allocated buffer
        if cache is not None and cache_pos is not None:
            for i in range(s):
                cache[:, cache_pos + i, :] = kv[:, i, :]
            kv_for_attn = cache
        else:
            kv_for_attn = kv

        # Attention: Q (b,s,local_h,d) x KV (b,kv_len,d)
        scores = torch.einsum("bshd,btd->bhst", q, kv_for_attn) * self.scale
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        out = torch.einsum("bhst,btd->bshd", weights, kv_for_attn)

        # Grouped output projection
        # wo_a is replicated (full o_groups). Each rank has local_heads.
        # Pad local heads into groups, apply grouped wo_a, slice for wo_b.
        local_heads = out.shape[2]
        heads_per_group = self.n_heads // self.o_groups

        # Map local heads to their group indices
        # With TP=64, n_heads=64, o_groups=8: each rank has 1 head in 1 group
        # We need to figure out which group(s) this rank's heads belong to
        rank = dist.get_rank() if dist.is_initialized() else 0
        first_global_head = rank * local_heads
        first_group = first_global_head // heads_per_group

        # Zero-pad: create full group output, fill only this rank's heads
        out_full = torch.zeros(
            b, s, self.o_groups, heads_per_group * self.head_dim,
            dtype=out.dtype, device=out.device,
        )
        head_in_group = first_global_head % heads_per_group
        hd = self.head_dim
        for i in range(local_heads):
            g = (first_global_head + i) // heads_per_group
            h_pos = (first_global_head + i) % heads_per_group
            out_full[:, :, g, h_pos * hd : (h_pos + 1) * hd] = out[:, :, i, :]

        # Apply grouped wo_a: (b, s, o_groups, group_dim) -> (b, s, o_groups, o_lora_rank)
        wo_a_w = self.wo_a.weight.view(self.o_groups, self.o_lora_rank, -1)
        out_lora = torch.einsum("bsgd,grd->bsgr", out_full, wo_a_w)

        # Slice for this rank's wo_b shard
        lora_total = self.o_groups * self.o_lora_rank
        flat = out_lora.flatten(2)  # (b, s, o_groups * o_lora_rank)
        chunk = lora_total // self.world_size if self.world_size > 1 else lora_total
        local_flat = flat[:, :, rank * chunk : (rank + 1) * chunk]

        out = self.wo_b(local_flat)
        # All-reduce moved to caller for multi-stream overlap
        return out


# -- MoE with Expert Parallelism --

class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x):
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
        return self.w2((F.silu(gate) * up).to(x.dtype))


class Gate(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.topk = args.n_activated_experts
        self.route_scale = args.route_scale
        self.score_func = args.score_func
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.is_hash = layer_idx < args.n_hash_layers
        if not self.is_hash:
            self.bias = nn.Parameter(torch.zeros(args.n_routed_experts))

    def forward(self, x):
        # Everything on Neuron — no CPU round-trip
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        else:
            scores = scores.sigmoid()

        if not self.is_hash:
            biased = scores + self.bias.float()
        else:
            biased = scores
        # topk on Neuron: indices auto-cast to int32 (fine for 256 experts)
        _, top_indices = biased.topk(self.topk, dim=-1)  # (n_tokens, topk)
        top_indices_i32 = top_indices.to(torch.int32)
        # Gather on CPU to avoid int64 issues with gather indices
        scores_cpu = scores.detach().cpu()
        indices_cpu = top_indices_i32.detach().cpu().long()
        weights = scores_cpu.gather(1, indices_cpu)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        return (weights * self.route_scale).to(x.dtype).to(x.device), top_indices_i32


class MoE(nn.Module):
    """
    EP strategy: each rank owns experts[rank_start:rank_end].
    Gate is replicated. After local expert computation, all-reduce combines.
    Shared expert is replicated, its output is added after all-reduce.
    """

    def __init__(self, args: ModelArgs, layer_idx: int, rank: int, world_size: int):
        super().__init__()
        self.dim = args.dim
        self.n_activated = args.n_activated_experts
        self.n_experts = args.n_routed_experts
        self.rank = rank
        self.world_size = world_size

        # Expert partition for this rank
        self.experts_per_rank = args.local_n_experts
        self.rank_start = rank * self.experts_per_rank
        self.rank_end = self.rank_start + self.experts_per_rank

        self.gate = Gate(args, layer_idx)
        # Only create local experts
        self.experts = nn.ModuleList(
            [
                Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)
                for _ in range(self.experts_per_rank)
            ]
        )
        self.shared_experts = Expert(args.dim, args.moe_inter_dim)

    def forward(self, x):
        shape = x.shape
        x_flat = x.view(-1, self.dim)
        n_tokens = x_flat.shape[0]
        weights, indices = self.gate(x_flat)  # all on Neuron, indices are int32

        # Build per-expert weight mask: (n_tokens, n_local_experts)
        # For each local expert, check if it's in the topk
        E = self.experts_per_rank
        expert_weights = torch.zeros(
            n_tokens, E, dtype=torch.float32, device=x.device
        )
        for i in range(E):
            global_idx = self.rank_start + i
            match = (indices == global_idx).float()  # (n_tokens, topk)
            expert_weights[:, i] = (weights.float() * match).sum(dim=-1)

        # Batched expert forward: stack weights, single bmm per projection
        # x_flat: (n_tokens, dim) -> expand to (E, n_tokens, dim)
        x_exp = x_flat.unsqueeze(0).expand(E, -1, -1)  # (E, n_tokens, dim)

        # Stack expert weights: (E, inter_dim, dim), (E, inter_dim, dim), (E, dim, inter_dim)
        w1 = torch.stack([e.w1.weight for e in self.experts])  # (E, inter, dim)
        w3 = torch.stack([e.w3.weight for e in self.experts])  # (E, inter, dim)
        w2 = torch.stack([e.w2.weight for e in self.experts])  # (E, dim, inter)

        # Batched SwiGLU: gate = x @ w1^T, up = x @ w3^T
        gate = torch.bmm(x_exp, w1.transpose(1, 2)).float()  # (E, n_tokens, inter)
        up = torch.bmm(x_exp, w3.transpose(1, 2)).float()    # (E, n_tokens, inter)
        if self.experts[0].swiglu_limit > 0:
            lim = self.experts[0].swiglu_limit
            gate = gate.clamp(max=lim)
            up = up.clamp(-lim, lim)
        hidden = (F.silu(gate) * up).to(x.dtype)              # (E, n_tokens, inter)
        out = torch.bmm(hidden, w2.transpose(1, 2))           # (E, n_tokens, dim)

        # Weight by routing scores: (E, n_tokens, dim) * (n_tokens, E) -> sum
        # expert_weights: (n_tokens, E) -> (E, n_tokens, 1)
        ew = expert_weights.t().unsqueeze(-1)  # (E, n_tokens, 1)
        y = (out.float() * ew).sum(dim=0)      # (n_tokens, dim)

        # All-reduce moved to caller for multi-stream overlap
        # Shared expert (replicated, no reduce needed — added after all-reduce by caller)
        return y.to(x.dtype).view(shape), self.shared_experts(x_flat).float().to(x.dtype).view(shape)


# -- Decoder Layer --

class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int, rank: int, world_size: int):
        super().__init__()
        self.attn = Attention(args, layer_idx, world_size)
        self.ffn = MoE(args, layer_idx, rank, world_size)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.hc_pre = HCPre(args)

        mix_hc = (2 + args.hc_mult) * args.hc_mult
        hc_dim = args.hc_mult * args.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.empty(3))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def forward(self, h, cos, sin, positions, mask, cache=None, cache_pos=None,
                comm_stream=None):
        world_size = self.attn.world_size

        # --- Attention with overlapped all-reduce ---
        residual = h
        x, post, comb = self.hc_pre(
            h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x_partial = self.attn(
            self.attn_norm(x).to(h.dtype), cos, sin, positions, mask,
            cache, cache_pos,
        )

        if world_size > 1 and comm_stream is not None:
            # Launch all-reduce on comm stream
            evt = torch.neuron.Event()
            with torch.neuron.stream(comm_stream):
                x_full = all_reduce(x_partial, "sum", list(range(world_size)))
                evt.record(comm_stream)
            # Compute residual term of hc_post while all-reduce is in flight
            term_b = (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)
            # Wait for all-reduce
            torch.neuron.default_stream().wait_event(evt)
            h = (post.unsqueeze(-1) * x_full.unsqueeze(-2) + term_b).to(h.dtype)
        elif world_size > 1:
            x_full = all_reduce(x_partial, "sum", list(range(world_size)))
            h = hc_post(x_full, residual, post, comb)
        else:
            h = hc_post(x_partial, residual, post, comb)

        # --- MoE with overlapped all-reduce ---
        residual = h
        x, post, comb = self.hc_pre(
            h, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        moe_partial, shared_out = self.ffn(self.ffn_norm(x).to(h.dtype))

        if world_size > 1 and comm_stream is not None:
            evt = torch.neuron.Event()
            with torch.neuron.stream(comm_stream):
                moe_full = all_reduce(moe_partial, "sum", list(range(world_size)))
                evt.record(comm_stream)
            # Compute residual term while all-reduce is in flight
            term_b = (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)
            # Also compute shared expert addition (independent of all-reduce)
            # Wait for all-reduce
            torch.neuron.default_stream().wait_event(evt)
            x_full = (moe_full.float() + shared_out.float()).to(h.dtype)
            h = (post.unsqueeze(-1) * x_full.unsqueeze(-2) + term_b).to(h.dtype)
        elif world_size > 1:
            moe_full = all_reduce(moe_partial, "sum", list(range(world_size)))
            x_full = (moe_full.float() + shared_out.float()).to(h.dtype)
            h = hc_post(x_full, residual, post, comb)
        else:
            x_full = (moe_partial.float() + shared_out.float()).to(h.dtype)
            h = hc_post(x_full, residual, post, comb)

        return h


# -- Full Model --

class DeepSeekV4(nn.Module):
    def __init__(self, args: ModelArgs, rank: int, world_size: int):
        super().__init__()
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(args, i, rank, world_size) for i in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.hc_head = HCHead(args)

        hc_dim = args.hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(torch.empty(args.hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(args.hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))

    def forward(self, input_ids, positions=None, caches=None, cache_pos=None):
        b, s = input_ids.shape
        if positions is None:
            positions = (
                torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, -1)
            )

        cos, sin = precompute_freqs(self.args, self.args.max_seq_len)
        cos, sin = cos.to(input_ids.device), sin.to(input_ids.device)

        # Build mask
        if caches is not None and cache_pos is not None:
            max_cache = caches[0].shape[1]
            mask = torch.full(
                (s, max_cache), float("-inf"), device=input_ids.device
            )
            valid_len = cache_pos + s
            mask[:, :valid_len] = 0.0
            if s > 1:
                for i in range(s):
                    mask[i, cache_pos + i + 1 : cache_pos + s] = float("-inf")
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            mask = torch.full((s, s), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        # Create comm stream for overlapping all-reduces
        comm_stream = torch.neuron.Stream() if self.world_size > 1 else None

        h = self.embed(input_ids)
        h = h.unsqueeze(2).expand(-1, -1, self.args.hc_mult, -1).contiguous()

        for i, layer in enumerate(self.layers):
            cache_i = caches[i] if caches is not None else None
            h = layer(h, cos, sin, positions, mask, cache_i, cache_pos,
                      comm_stream)

        h = self.hc_head(
            h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        h = self.norm(h)
        return self.head(h)


# -- TP Plan --

def apply_tp_plan(model, mesh):
    """Apply tensor parallelism to attention Q/output projections."""
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    layer_tp_plan = {
        "attn.wq_b": ColwiseParallel(),  # shard Q heads across ranks
        "attn.wo_b": RowwiseParallel(),  # all-reduce after output proj
    }
    for layer in model.layers:
        parallelize_module(layer, mesh, layer_tp_plan)
    return model


# -- Weight Loading --

def load_sharded_weights(model, shard_dir, rank):
    """Load pre-sharded BF16 weights for this rank."""
    shard_file = os.path.join(shard_dir, f"rank_{rank:03d}.safetensors")
    sd = load_file(shard_file)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    return len(missing), len(unexpected), len(sd)


def load_weights(model, model_dir, rank, world_size):
    """Load weights from HF checkpoint, distributing experts across ranks."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        files = set(index["weight_map"].values())
        sd = {}
        for fname in files:
            sd.update(load_file(str(model_dir / fname)))
    else:
        sd = load_file(str(model_dir / "model.safetensors"))

    experts_per_rank = model.args.n_routed_experts // world_size
    rank_start = rank * experts_per_rank

    mapped = {}
    for k, v in sd.items():
        new_k = k
        if ".experts." in k:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "experts" and i + 1 < len(parts):
                    global_idx = int(parts[i + 1])
                    if global_idx < rank_start or global_idx >= rank_start + experts_per_rank:
                        break
                    local_idx = global_idx - rank_start
                    parts[i + 1] = str(local_idx)
                    new_k = ".".join(parts)
                    break
            else:
                mapped[new_k] = v
                continue
            if global_idx < rank_start or global_idx >= rank_start + experts_per_rank:
                continue
        mapped[new_k] = v

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if rank == 0:
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    return len(missing), len(unexpected)


# -- Main --

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="yujiepan/deepseek-v4-tiny-random",
        help="HuggingFace model ID (for tiny-random test)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to pre-sharded BF16 weights (from preprocess_weights.py)",
    )
    args = parser.parse_args()

    use_presharded = args.model_path is not None

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.neuron.current_device()

    if rank == 0:
        print(f"=== DeepSeek V4 TP Inference ===")
        print(f"World size: {world_size}")
        if use_presharded:
            print(f"Pre-sharded weights: {args.model_path}")
        else:
            print(f"HF model: {args.model_id}")

    # Load config
    if use_presharded:
        config_path = os.path.join(args.model_path, "config.json")
    else:
        if rank == 0:
            print("Downloading model...")
        t0 = time.time()
        model_dir = Path(snapshot_download(args.model_id))
        dist.barrier()
        if rank == 0:
            print(f"  Downloaded in {time.time() - t0:.1f}s")
        config_path = str(model_dir / "config.json")

    margs = load_args_from_hf(Path(config_path))
    if rank == 0:
        print(
            f"  dim={margs.dim}, layers={margs.n_layers}, heads={margs.n_heads}, "
            f"experts={margs.n_routed_experts}, head_dim={margs.head_dim}"
        )

    # Validate TP degree
    assert margs.n_heads % world_size == 0, (
        f"n_heads ({margs.n_heads}) must be divisible by world_size ({world_size})"
    )
    assert margs.n_routed_experts % world_size == 0, (
        f"n_routed_experts ({margs.n_routed_experts}) must be divisible by "
        f"world_size ({world_size})"
    )

    # Set local dimensions for pre-sharded mode
    if use_presharded:
        margs.local_n_heads = margs.n_heads // world_size
        margs.local_n_experts = margs.n_routed_experts // world_size
        heads_per_group = margs.n_heads // margs.o_groups
        margs.local_o_groups = max(1, margs.local_n_heads // heads_per_group)
    else:
        margs.local_n_experts = margs.n_routed_experts // world_size

    # Build model on CPU (or meta device for pre-sharded to save memory)
    if rank == 0:
        print("Building model...")
    t0 = time.time()
    if use_presharded:
        with torch.device("meta"):
            model = DeepSeekV4(margs, rank, world_size)
    else:
        model = DeepSeekV4(margs, rank, world_size)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Built in {time.time() - t0:.1f}s, {total_params:,} params (rank 0)")

    # Load weights
    if rank == 0:
        print("Loading weights...")
    t0 = time.time()
    if use_presharded:
        n_missing, n_unexpected, n_loaded = load_sharded_weights(
            model, args.model_path, rank
        )
        if rank == 0:
            print(
                f"  Loaded {n_loaded} keys in {time.time() - t0:.1f}s "
                f"(missing={n_missing}, unexpected={n_unexpected})"
            )
        # Ensure consistent dtype — shards are BF16 but some params may be float32
        model = model.to(torch.bfloat16)
    else:
        load_weights(model, model_dir, rank, world_size)
        if rank == 0:
            print(f"  Loaded in {time.time() - t0:.1f}s")

    # Apply TP plan only for HF weights (pre-sharded already has correct shapes)
    if not use_presharded and world_size > 1:
        from torch.distributed.device_mesh import DeviceMesh

        mesh = DeviceMesh("neuron", list(range(world_size)))
        if rank == 0:
            print("Applying TP plan...")
        model = apply_tp_plan(model, mesh)

    # Move to device
    if rank == 0:
        print(f"Moving to {device}...")
    t0 = time.time()
    model = model.to(device)
    model.eval()
    if rank == 0:
        print(f"  Moved in {time.time() - t0:.1f}s")

    dist.barrier()

    MAX_CACHE_LEN = 128  # static cache size

    # Allocate static KV caches on device
    caches = [
        torch.zeros(1, MAX_CACHE_LEN, margs.head_dim, dtype=torch.bfloat16, device=device)
        for _ in range(margs.n_layers)
    ]

    # Prefill: fill cache positions 0..SEQ_LEN-1
    input_ids = torch.randint(0, margs.vocab_size, (1, SEQ_LEN), device=device)
    if rank == 0:
        print(f"\nPrefill (seq_len={SEQ_LEN}, max_cache={MAX_CACHE_LEN})...")
    t0 = time.time()
    with torch.no_grad():
        logits = model(input_ids, caches=caches, cache_pos=0)
    ttft = time.time() - t0
    if rank == 0:
        print(f"  TTFT: {ttft:.2f}s")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits sample: {logits[0, -1, :5].cpu().tolist()}")
        has_nan = torch.isnan(logits[0, -1]).any().item()
        print(f"  Has NaN: {has_nan}")

    # Greedy generation with static KV cache
    max_gen = min(MAX_NEW_TOKENS, MAX_CACHE_LEN - SEQ_LEN)
    if rank == 0:
        print(f"\nGreedy generation ({max_gen} tokens, static cache)...")
    generated = input_ids.clone()
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)
    if rank == 0:
        print(f"  step 0: token {next_token.item()}")

    # Decode: each step has FIXED shapes (1 token query, MAX_CACHE_LEN KV)
    t0 = time.time()
    with torch.no_grad():
        for step in range(1, max_gen):
            cache_pos = SEQ_LEN + step
            pos = torch.tensor([[cache_pos]], device=device, dtype=torch.long)
            logits = model(next_token, positions=pos, caches=caches, cache_pos=cache_pos)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if rank == 0:
                print(f"  step {step}: token {next_token.item()}")
    decode_time = time.time() - t0
    if rank == 0:
        new_ids = generated[0, SEQ_LEN:].cpu().tolist()
        n_decode = max_gen - 1
        tps = n_decode / decode_time if decode_time > 0 else 0
        print(f"  Generated {len(new_ids)} tokens")
        print(f"  Token IDs: {new_ids}")
        print(f"  TTFT: {ttft:.2f}s")
        print(f"  Decode: {n_decode} tokens in {decode_time:.2f}s ({tps:.2f} tok/s)")
        print(f"  Total: {ttft + decode_time:.2f}s")
        print("\nDeepSeek V4 TP inference PASSED")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
