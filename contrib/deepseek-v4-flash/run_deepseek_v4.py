"""
DeepSeek V4 tiny-random inference on Neuron (eager mode).

Since transformers doesn't have deepseek_v4 support yet, this script
uses a minimal standalone model implementation adapted from the
DeepSeek V4 reference code.

Usage:
    python3 run_deepseek_v4.py
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_neuronx
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


MODEL_ID = "yujiepan/deepseek-v4-tiny-random"
SEQ_LEN = 32
MAX_NEW_TOKENS = 8


# ── Config ──────────────────────────────────────────────────────────────

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
    rope_head_dim: int = 64  # will be clamped to head_dim
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
        # Clamp rope_head_dim to head_dim for tiny models
        if self.rope_head_dim > self.head_dim:
            self.rope_head_dim = self.head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim


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
        original_seq_len=cfg.get("rope_scaling", {}).get("original_max_position_embeddings", 65536),
        hc_mult=cfg.get("hc_mult", 4),
        hc_sinkhorn_iters=cfg.get("hc_sinkhorn_iters", 20),
        hc_eps=cfg.get("hc_eps", 1e-6),
        n_hash_layers=cfg.get("num_hash_layers", 3),
        score_func=cfg.get("scoring_func", "sqrtsoftplus"),
        route_scale=cfg.get("routed_scaling_factor", 1.5),
        swiglu_limit=cfg.get("swiglu_limit", 10.0),
    )


# ── RMSNorm ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


# ── RoPE ────────────────────────────────────────────────────────────────

def precompute_freqs(args: ModelArgs, seq_len: int, theta: float = None):
    if theta is None:
        theta = args.rope_theta
    dim = args.rope_head_dim
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor):
    """Apply rotary embeddings. x: (..., rope_dim), cos/sin: (max_seq, rope_dim//2)"""
    cos_pos = cos[positions]  # (batch, seq, rope_dim//2)
    sin_pos = sin[positions]
    # If x has a heads dim (4D), unsqueeze cos/sin to broadcast over heads
    if x.dim() > cos_pos.dim():
        cos_pos = cos_pos.unsqueeze(-2)
        sin_pos = sin_pos.unsqueeze(-2)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.stack([x1 * cos_pos - x2 * sin_pos, x2 * cos_pos + x1 * sin_pos], dim=-1)
    return out.flatten(-2).to(x.dtype)


# ── Hyper-Connections ───────────────────────────────────────────────────

def hc_sinkhorn(logits: torch.Tensor, n_iters: int, eps: float) -> torch.Tensor:
    m = torch.exp(logits)
    for _ in range(n_iters):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


class HCPre(nn.Module):
    """Reduce hc_mult copies → 1."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hc_mult = args.hc_mult
        self.sinkhorn_iters = args.hc_sinkhorn_iters
        self.eps = args.hc_eps
        self.norm_eps = args.norm_eps
        mix_hc = (2 + args.hc_mult) * args.hc_mult
        hc_dim = args.hc_mult * args.dim

    def forward(self, x, fn, scale, base):
        """x: (b,s,hc,d) → y: (b,s,d), post, comb"""
        shape = x.shape
        hc = self.hc_mult
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, fn) * rsqrt

        pre_l = mixes[..., :hc] * scale[0] + base[:hc]
        post_l = mixes[..., hc:2*hc] * scale[1] + base[hc:2*hc]
        comb_l = mixes[..., 2*hc:] * scale[2] + base[2*hc:]

        pre = torch.sigmoid(pre_l) + self.eps
        post = torch.sigmoid(post_l) + self.eps
        comb = hc_sinkhorn(comb_l.view(*comb_l.shape[:-1], hc, hc), self.sinkhorn_iters, self.eps)

        y = (pre.unsqueeze(-1) * x_flat.view(shape)).sum(dim=2)
        return y.to(x.dtype), post, comb


def hc_post(x, residual, post, comb):
    """x: (b,s,d), residual: (b,s,hc,d) → (b,s,hc,d)"""
    return (post.unsqueeze(-1) * x.unsqueeze(-2)
            + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)).to(x.dtype)


class HCHead(nn.Module):
    """Final reduction hc_mult → 1."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hc_mult = args.hc_mult
        self.norm_eps = args.norm_eps
        self.eps = args.hc_eps

    def forward(self, x, fn, scale, base):
        shape = x.shape
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.pow(2).mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, fn) * rsqrt
        pre = torch.sigmoid(mixes * scale + base) + self.eps
        return (pre.unsqueeze(-1) * x_flat.view(shape)).sum(dim=2).to(x.dtype)


# ── Attention ───────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_dim = args.rope_head_dim
        self.nope_dim = args.nope_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.o_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.scale = args.head_dim ** -0.5

        # Q path
        self.wq_a = nn.Linear(args.dim, args.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(args.q_lora_rank, args.norm_eps)
        self.wq_b = nn.Linear(args.q_lora_rank, args.n_heads * args.head_dim, bias=False)

        # KV path (MQA single head)
        self.wkv = nn.Linear(args.dim, args.head_dim, bias=False)
        self.kv_norm = RMSNorm(args.head_dim, args.norm_eps)

        # Output grouped low-rank
        group_dim = (args.n_heads * args.head_dim) // args.o_groups
        self.wo_a = nn.Linear(group_dim, args.o_groups * args.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(args.o_groups * args.o_lora_rank, args.dim, bias=False)

    def forward(self, x, cos, sin, positions, mask=None):
        b, s, _ = x.shape
        rd = self.rope_dim

        # Q
        q = self.wq_b(self.q_norm(self.wq_a(x)).to(x.dtype))
        q = q.view(b, s, self.n_heads, self.head_dim)
        # Per-head RMSNorm on Q
        q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        q = q.to(x.dtype)

        # KV
        kv = self.kv_norm(self.wkv(x)).to(x.dtype)

        # Apply RoPE to rope portion
        if rd > 0:
            q_rope = apply_rope(q[..., -rd:], cos, sin, positions)
            q = torch.cat([q[..., :-rd], q_rope], dim=-1)
            kv_rope = apply_rope(kv[..., -rd:], cos, sin, positions)
            kv = torch.cat([kv[..., :-rd], kv_rope], dim=-1)

        # Attention: Q (b,s,h,d) × KV (b,s,d) → scores (b,h,s,s)
        scores = torch.einsum("bshd,btd->bhst", q, kv) * self.scale
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        out = torch.einsum("bhst,btd->bshd", weights, kv)

        # Grouped output projection
        out = out.reshape(b, s, self.o_groups, -1)
        wo_a_w = self.wo_a.weight.view(self.o_groups, self.o_lora_rank, -1)
        out = torch.einsum("bsgd,grd->bsgr", out, wo_a_w)
        return self.wo_b(out.flatten(2))


# ── MoE ─────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)  # gate
        self.w3 = nn.Linear(dim, inter_dim, bias=False)  # up
        self.w2 = nn.Linear(inter_dim, dim, bias=False)   # down
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
        # Gate runs on CPU: topk/gather use int64 which Neuron can't handle
        x_cpu = x.detach().cpu().float()
        w_cpu = self.weight.detach().cpu().float()
        scores = F.linear(x_cpu, w_cpu)
        if self.score_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        else:
            scores = scores.sigmoid()
        if not self.is_hash:
            biased = scores + self.bias.detach().cpu()
        else:
            biased = scores
        indices = biased.topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        return (weights * self.route_scale).to(x.dtype), indices


class MoE(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.dim = args.dim
        self.n_activated = args.n_activated_experts
        self.gate = Gate(args, layer_idx)
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim, args.swiglu_limit)
            for _ in range(args.n_routed_experts)
        ])
        self.shared_experts = Expert(args.dim, args.moe_inter_dim)

    def forward(self, x):
        shape = x.shape
        x_flat = x.view(-1, self.dim)
        n_tokens = x_flat.shape[0]
        weights, indices = self.gate(x_flat)  # CPU tensors: [n_tokens, topk]
        # Only iterate over experts that are actually activated
        active = indices.unique().tolist()
        y = torch.zeros(x_flat.shape, dtype=torch.float32, device=x.device)
        for i in active:
            idx, top = torch.where(indices == i)  # CPU
            w = weights[idx, top, None].to(x.device)
            y[idx.to(x.device)] += (self.experts[i](x_flat[idx.to(x.device)]) * w).float()
        y = y + self.shared_experts(x_flat).float()
        return y.to(x.dtype).view(shape)


# ── Decoder Layer ───────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn = Attention(args, layer_idx)
        self.ffn = MoE(args, layer_idx)
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

    def forward(self, h, cos, sin, positions, mask):
        # Attention with HC
        residual = h
        x, post, comb = self.hc_pre(h, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn(self.attn_norm(x).to(h.dtype), cos, sin, positions, mask)
        h = hc_post(x, residual, post, comb)

        # FFN with HC
        residual = h
        x, post, comb = self.hc_pre(h, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn(self.ffn_norm(x).to(h.dtype))
        h = hc_post(x, residual, post, comb)
        return h


# ── Full Model ──────────────────────────────────────────────────────────

class DeepSeekV4(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([DecoderLayer(args, i) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.hc_head = HCHead(args)

        hc_dim = args.hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(torch.empty(args.hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(args.hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor = None):
        b, s = input_ids.shape
        if positions is None:
            positions = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, -1)

        cos, sin = precompute_freqs(self.args, self.args.max_seq_len)
        cos, sin = cos.to(input_ids.device), sin.to(input_ids.device)

        # Causal mask
        mask = torch.full((s, s), float("-inf"), device=input_ids.device)
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        h = self.embed(input_ids)
        h = h.unsqueeze(2).expand(-1, -1, self.args.hc_mult, -1).contiguous()

        for layer in self.layers:
            h = layer(h, cos, sin, positions, mask)

        h = self.hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        h = self.norm(h)
        return self.head(h.float())


# ── Weight Loading ──────────────────────────────────────────────────────

def load_weights(model: DeepSeekV4, model_dir: Path):
    """Load safetensors weights using the inference key naming convention."""
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

    # Map inference keys → our model keys
    mapped = {}
    for k, v in sd.items():
        new_k = k
        # The safetensors use inference naming: embed.weight, layers.N.*, head.weight, etc.
        # Our model uses the same naming convention, so direct load should work
        mapped[new_k] = v

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:10]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")
    return len(missing), len(unexpected)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    device = torch.device("neuron")

    print(f"Downloading model from {MODEL_ID}...")
    t0 = time.time()
    model_dir = Path(snapshot_download(MODEL_ID))
    print(f"  Downloaded to {model_dir} in {time.time() - t0:.1f}s")

    print("Loading config...")
    args = load_args_from_hf(model_dir / "config.json")
    print(f"  dim={args.dim}, n_layers={args.n_layers}, n_heads={args.n_heads}, "
          f"head_dim={args.head_dim}, rope_dim={args.rope_head_dim}, "
          f"n_experts={args.n_routed_experts}")

    print("Building model...")
    t0 = time.time()
    model = DeepSeekV4(args)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Built in {time.time() - t0:.1f}s, {total_params:,} parameters")

    print("Loading weights...")
    t0 = time.time()
    n_missing, n_unexpected = load_weights(model, model_dir)
    print(f"  Loaded in {time.time() - t0:.1f}s (missing={n_missing}, unexpected={n_unexpected})")
    model.eval()

    print(f"Moving model to {device}...")
    t0 = time.time()
    model = model.to(device)
    print(f"  Moved in {time.time() - t0:.1f}s")

    # Forward pass
    input_ids = torch.randint(0, args.vocab_size, (1, SEQ_LEN), device=device)
    print(f"\nRunning forward pass (seq_len={SEQ_LEN})...")
    t0 = time.time()
    with torch.no_grad():
        logits = model(input_ids)
    print(f"  Forward: {time.time() - t0:.2f}s")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits sample: {logits[0, -1, :5].cpu().tolist()}")

    # Greedy generation
    print(f"\nGreedy generation ({MAX_NEW_TOKENS} tokens)...")
    generated = input_ids.clone()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits = model(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    elapsed = time.time() - t0
    new_ids = generated[0, SEQ_LEN:].cpu().tolist()
    print(f"  Generated {len(new_ids)} tokens in {elapsed:.2f}s")
    print(f"  Token IDs: {new_ids}")

    print("\nDeepSeek V4 tiny-random inference PASSED")


if __name__ == "__main__":
    main()
