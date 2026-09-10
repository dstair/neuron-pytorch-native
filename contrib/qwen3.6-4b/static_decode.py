#!/usr/bin/env python3
"""Maximum-throughput static decode for Qwen/Qwen3.5-4B on Trainium2.

The complete 32-layer token step is one ``torch.compile(..., backend="neuron")``
graph. The implementation transfers the qualified 35B-A3B decode structure:

* DMA-coalesced batched DeltaNet with tiled convolution state.
* Direct, distinct output buffers for every DeltaNet layer.
* Stateful GQA-tail kernels with one KV buffer per GQA layer.
* Vocab-sharded greedy LM head.
* Configurable native Trn2 FP8 W8A16 matmuls for all projections or only the
  vocabulary head.

The current implementation is intentionally decode-only. It benchmarks the
steady-state token step from deterministic zero state, which isolates decode
throughput from prompt processing.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce as _all_reduce

import model_dims as D
from st_reader import SafeReader, build_weight_map


# The copied 35B kernels are already exactly the 4B TP=4 tile geometry.
os.environ.setdefault("DN_K_HEADS", "4")
os.environ.setdefault("DN_V_HEADS", "8")
os.environ.setdefault("GQA_Q_HEADS", "4")
os.environ.setdefault("DN_TILED_CONV", "1")

_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
sys.path.insert(0, _KERNELS)
import deltanet_full_batched_ops  # noqa: E402,F401
import fp8_fused_head_ops  # noqa: E402,F401
import fp8_fused_mlp_ops  # noqa: E402,F401
import fp8_matmul_ops  # noqa: E402,F401
import gqa_tail_ops  # noqa: E402,F401


FP8_MAX = 240.0
TILED_PMAX = 128


def functional_all_reduce(x: torch.Tensor, op: str, world_size: int) -> torch.Tensor:
    return _all_reduce(x, op, list(range(world_size)))


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + D.RMS_EPS)
    return (normalized * (1.0 + weight.float())).to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rd = cos.shape[-1]
    qr, qp = q[..., :rd], q[..., rd:]
    kr, kp = k[..., :rd], k[..., rd:]
    return (
        torch.cat((qr * cos + rotate_half(qr) * sin, qp), dim=-1),
        torch.cat((kr * cos + rotate_half(kr) * sin, kp), dim=-1),
    )


def quantize_fp8_per_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert [out,in] BF16 to legacy-compatible E4M3 bytes [in,out]."""
    absmax = weight.abs().amax(dim=-1).float().clamp_min(1e-12)
    scale = absmax / FP8_MAX
    inverse = (1.0 / scale).to(weight.dtype).unsqueeze(1)
    quantized = (
        (weight * inverse)
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
    )
    return quantized.t().contiguous().view(torch.int8), scale.unsqueeze(1)


def fp8_linear(
    x: torch.Tensor,
    weight_fp8_t: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    shape = x.shape
    flat = x.reshape(-1, shape[-1]).to(torch.bfloat16)
    out = torch.ops.qwen4b_fp8.matmul(flat, weight_fp8_t, scale)
    return out.reshape(*shape[:-1], out.shape[-1])


def _find_checkpoint_keys(weight_map: dict[str, str]) -> tuple[str, str, str | None]:
    candidates = [
        k for k in weight_map
        if k.endswith("layers.0.input_layernorm.weight")
        and not k.startswith("mtp.")
    ]
    sample = next(
        (k for k in candidates if "language_model.layers.0" in k),
        candidates[0],
    )
    prefix = sample[: sample.index("layers.0.input_layernorm.weight")]
    embed = next(k for k in weight_map if k.endswith("embed_tokens.weight"))
    norm = prefix + "norm.weight"
    if norm not in weight_map:
        norm = next(k for k in weight_map if k.endswith(".norm.weight"))
    lm_head = next(
        (k for k in weight_map if k == "lm_head.weight" or k.endswith(".lm_head.weight")),
        None,
    )
    return prefix, embed, lm_head


def load_sharded_weights(
    checkpoint: str,
    rank: int,
    world_size: int,
) -> dict:
    """Read only the TP shard needed by one rank from the HF safetensors."""
    td = D.tp_dims(world_size)
    weight_map = build_weight_map(checkpoint)
    prefix, embed_key, lm_head_key = _find_checkpoint_keys(weight_map)
    handles: dict[str, SafeReader] = {}

    def get(key: str) -> torch.Tensor:
        filename = weight_map[key]
        if filename not in handles:
            handles[filename] = SafeReader(os.path.join(checkpoint, filename))
        return handles[filename].get_tensor(key)

    def colwise(weight: torch.Tensor) -> torch.Tensor:
        rows = weight.shape[0] // world_size
        return weight[rank * rows : (rank + 1) * rows].clone()

    def rowwise(weight: torch.Tensor) -> torch.Tensor:
        cols = weight.shape[1] // world_size
        return weight[:, rank * cols : (rank + 1) * cols].clone()

    embed = get(embed_key)
    lm_full = get(lm_head_key) if lm_head_key is not None else embed
    vocab_rows = D.VOCAB // world_size
    weights = {
        "embed": embed,
        "final_norm": get(prefix + "norm.weight"),
        "lm_head": lm_full[rank * vocab_rows : (rank + 1) * vocab_rows].clone(),
        "layers": [],
    }

    key_full = D.DN_KEY_DIM
    value_full = D.DN_VALUE_DIM
    key_local = key_full // world_size
    value_local = value_full // world_size
    kv_heads_local = D.GQA_KV_HEADS // world_size

    for i in range(D.NUM_LAYERS):
        lp = f"{prefix}layers.{i}."
        lw = {
            "input_norm": get(lp + "input_layernorm.weight"),
            "post_norm": get(lp + "post_attention_layernorm.weight"),
            "mlp_gate": colwise(get(lp + "mlp.gate_proj.weight")),
            "mlp_up": colwise(get(lp + "mlp.up_proj.weight")),
            "mlp_down": rowwise(get(lp + "mlp.down_proj.weight")),
        }
        if D.layer_type(i) == "deltanet":
            ap = lp + "linear_attn."
            qkv = get(ap + "in_proj_qkv.weight")
            q_part = qkv[
                rank * key_local : (rank + 1) * key_local
            ]
            k_part = qkv[
                key_full + rank * key_local : key_full + (rank + 1) * key_local
            ]
            v_part = qkv[
                2 * key_full + rank * value_local :
                2 * key_full + (rank + 1) * value_local
            ]
            conv = get(ap + "conv1d.weight")
            cq = conv[rank * key_local : (rank + 1) * key_local]
            ck = conv[
                key_full + rank * key_local : key_full + (rank + 1) * key_local
            ]
            cv = conv[
                2 * key_full + rank * value_local :
                2 * key_full + (rank + 1) * value_local
            ]
            lw.update(
                {
                    "dn_qkv": torch.cat((q_part, k_part, v_part), dim=0).clone(),
                    "dn_conv_w": torch.cat((cq, ck, cv), dim=0).clone(),
                    "dn_z": colwise(get(ap + "in_proj_z.weight")),
                    "dn_a": colwise(get(ap + "in_proj_a.weight")),
                    "dn_b": colwise(get(ap + "in_proj_b.weight")),
                    "dn_out": rowwise(get(ap + "out_proj.weight")),
                    "dn_A_log": colwise(get(ap + "A_log")),
                    "dn_dt_bias": colwise(get(ap + "dt_bias")),
                    "dn_norm": get(ap + "norm.weight"),
                }
            )
        else:
            ap = lp + "self_attn."
            k = get(ap + "k_proj.weight").reshape(
                D.GQA_KV_HEADS, D.GQA_HEAD_DIM, D.HIDDEN
            )
            v = get(ap + "v_proj.weight").reshape(
                D.GQA_KV_HEADS, D.GQA_HEAD_DIM, D.HIDDEN
            )
            kv_slice = slice(rank * kv_heads_local, (rank + 1) * kv_heads_local)
            lw.update(
                {
                    "gqa_q": colwise(get(ap + "q_proj.weight")),
                    "gqa_k": k[kv_slice].reshape(-1, D.HIDDEN).clone(),
                    "gqa_v": v[kv_slice].reshape(-1, D.HIDDEN).clone(),
                    "gqa_o": rowwise(get(ap + "o_proj.weight")),
                    "gqa_q_norm": get(ap + "q_norm.weight"),
                    "gqa_k_norm": get(ap + "k_norm.weight"),
                }
            )
        weights["layers"].append(lw)

    return weights


class StaticDecode4B(nn.Module):
    def __init__(
        self,
        weights: dict,
        max_seq_len: int,
        world_size: int,
        rank: int,
        batch_size: int,
        fp8_scope: str,
        fused_fp8_mlp: bool = False,
        fused_fp8_head: bool = False,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size
        self.fp8_scope = fp8_scope
        self.fused_fp8_mlp = fused_fp8_mlp
        self.fused_fp8_head = fused_fp8_head
        self.td = D.tp_dims(world_size)
        if (fused_fp8_mlp or fused_fp8_head) and world_size != 1:
            raise ValueError("fused FP8 MLP/head kernels currently require TP=1")
        if fused_fp8_mlp and fp8_scope not in ("all", "mlp-head"):
            raise ValueError("fused FP8 MLP requires an FP8 MLP scope")
        if fused_fp8_head and fp8_scope not in ("all", "mlp-head", "lm-head"):
            raise ValueError("fused FP8 head requires an FP8 LM-head scope")

        def reg_buffer(name: str, value: torch.Tensor, source: dict, key: str):
            self.register_buffer(name, value)
            source.pop(key, None)

        def use_fp8(name: str) -> bool:
            if self.fp8_scope == "all":
                return True
            if self.fp8_scope == "lm-head":
                return name == "lm_head"
            if self.fp8_scope == "mlp-head":
                return name == "lm_head" or "_mlp_" in name
            return False

        def reg_linear(name: str, value: torch.Tensor, source: dict, key: str):
            if use_fp8(name):
                q, scale = quantize_fp8_per_channel(value)
                self.register_buffer(name + "_q", q)
                self.register_buffer(name + "_s", scale)
            else:
                self.register_buffer(name, value)
            source.pop(key, None)

        reg_buffer("embed", weights["embed"], weights, "embed")
        reg_buffer("final_norm", weights["final_norm"], weights, "final_norm")
        if self.fused_fp8_head:
            lm_q, lm_s = quantize_fp8_per_channel(weights["lm_head"])
            self.register_buffer("lm_head_q", lm_q)
            self.register_buffer("lm_head_s_row", lm_s.t().contiguous())
            weights.pop("lm_head", None)
        else:
            reg_linear("lm_head", weights["lm_head"], weights, "lm_head")

        for i, lw in enumerate(weights["layers"]):
            reg_buffer(f"l{i}_input_norm", lw["input_norm"], lw, "input_norm")
            reg_buffer(f"l{i}_post_norm", lw["post_norm"], lw, "post_norm")
            if self.fused_fp8_mlp:
                gate_q, gate_s = quantize_fp8_per_channel(lw["mlp_gate"])
                up_q, up_s = quantize_fp8_per_channel(lw["mlp_up"])
                down_q, down_s = quantize_fp8_per_channel(lw["mlp_down"])
                self.register_buffer(
                    f"l{i}_mlp_gate_up_q",
                    torch.stack((gate_q, up_q), dim=1).contiguous(),
                )
                self.register_buffer(
                    f"l{i}_mlp_gate_up_s",
                    torch.cat(
                        (gate_s.t(), up_s.t()),
                        dim=0,
                    ).contiguous(),
                )
                self.register_buffer(f"l{i}_mlp_down_q", down_q)
                self.register_buffer(
                    f"l{i}_mlp_down_s_row",
                    down_s.t().contiguous(),
                )
                lw.pop("mlp_gate", None)
                lw.pop("mlp_up", None)
                lw.pop("mlp_down", None)
            else:
                reg_linear(f"l{i}_mlp_gate", lw["mlp_gate"], lw, "mlp_gate")
                reg_linear(f"l{i}_mlp_up", lw["mlp_up"], lw, "mlp_up")
                reg_linear(f"l{i}_mlp_down", lw["mlp_down"], lw, "mlp_down")
            if D.layer_type(i) == "deltanet":
                reg_linear(f"l{i}_dn_qkv", lw["dn_qkv"], lw, "dn_qkv")
                reg_buffer(f"l{i}_dn_conv_w", lw["dn_conv_w"], lw, "dn_conv_w")
                reg_linear(f"l{i}_dn_z", lw["dn_z"], lw, "dn_z")
                reg_linear(f"l{i}_dn_a", lw["dn_a"], lw, "dn_a")
                reg_linear(f"l{i}_dn_b", lw["dn_b"], lw, "dn_b")
                reg_linear(f"l{i}_dn_out", lw["dn_out"], lw, "dn_out")
                reg_buffer(f"l{i}_dn_A_log", lw["dn_A_log"], lw, "dn_A_log")
                reg_buffer(f"l{i}_dn_dt_bias", lw["dn_dt_bias"], lw, "dn_dt_bias")
                reg_buffer(f"l{i}_dn_norm", lw["dn_norm"], lw, "dn_norm")
            else:
                reg_linear(f"l{i}_gqa_q", lw["gqa_q"], lw, "gqa_q")
                reg_linear(f"l{i}_gqa_k", lw["gqa_k"], lw, "gqa_k")
                reg_linear(f"l{i}_gqa_v", lw["gqa_v"], lw, "gqa_v")
                reg_linear(f"l{i}_gqa_o", lw["gqa_o"], lw, "gqa_o")
                reg_buffer(f"l{i}_gqa_q_norm", lw["gqa_q_norm"], lw, "gqa_q_norm")
                reg_buffer(f"l{i}_gqa_k_norm", lw["gqa_k_norm"], lw, "gqa_k_norm")

        cache_shape = (batch_size, max_seq_len, D.GQA_HEAD_DIM)
        local_kv_heads = self.td["gqa_kv_heads"]
        for gi in range(D.NUM_GQA):
            if local_kv_heads == 1:
                self.register_buffer(
                    f"kv_k{gi}",
                    torch.zeros(cache_shape, dtype=torch.bfloat16),
                    persistent=False,
                )
                self.register_buffer(
                    f"kv_v{gi}",
                    torch.zeros(cache_shape, dtype=torch.bfloat16),
                    persistent=False,
                )
            else:
                for kvh in range(local_kv_heads):
                    self.register_buffer(
                        f"kv_k{gi}_{kvh}",
                        torch.zeros(cache_shape, dtype=torch.bfloat16),
                        persistent=False,
                    )
                    self.register_buffer(
                        f"kv_v{gi}_{kvh}",
                        torch.zeros(cache_shape, dtype=torch.bfloat16),
                        persistent=False,
                    )

        inv = 1.0 / (
            D.ROPE_THETA
            ** (torch.arange(0, D.ROPE_DIM, 2).float() / D.ROPE_DIM)
        )
        positions = torch.arange(max_seq_len).float()
        frequencies = torch.outer(positions, inv)
        rope = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("rope_cos", rope.cos())
        self.register_buffer("rope_sin", rope.sin())
        self.register_buffer(
            "position_range",
            torch.arange(max_seq_len),
            persistent=False,
        )

    def _linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, name + "_q"):
            return fp8_linear(
                x,
                getattr(self, name + "_q"),
                getattr(self, name + "_s"),
            )
        return F.linear(x.to(torch.bfloat16), getattr(self, name))

    def _deltanet(
        self,
        i: int,
        x: torch.Tensor,
        states_in: torch.Tensor,
        conv_in: torch.Tensor,
        states_out: list[torch.Tensor],
        conv_out: list[torch.Tensor],
    ) -> torch.Tensor:
        di = D.deltanet_index(i)
        batch = self.batch_size
        kh = self.td["dn_k_heads"]
        vh = self.td["dn_v_heads"]
        qkv_dim = self.td["dn_qkv_dim"]
        value_dim = self.td["dn_value_dim"]

        x2 = x.squeeze(1)
        mixed = self._linear(f"l{i}_dn_qkv", x2)
        a = self._linear(f"l{i}_dn_a", x2).float()
        b = self._linear(f"l{i}_dn_b", x2).float()
        z = self._linear(f"l{i}_dn_z", x2).reshape(
            batch, vh, D.DN_V_DIM
        )

        state_arg = states_in[di].reshape(
            batch * vh * D.DN_K_DIM, D.DN_V_DIM
        )
        conv_arg = conv_in[di].reshape(
            batch * TILED_PMAX,
            (qkv_dim // TILED_PMAX) * (D.DN_CONV_KERNEL - 1),
        )
        state_target = states_out[di].reshape(
            batch * vh * D.DN_K_DIM, D.DN_V_DIM
        )
        conv_target = conv_out[di].reshape(
            batch * TILED_PMAX,
            (qkv_dim // TILED_PMAX) * (D.DN_CONV_KERNEL - 1),
        )
        conv_weight = getattr(self, f"l{i}_dn_conv_w").squeeze(1).float()
        conv_bias = torch.zeros(
            qkv_dim, dtype=torch.float32, device=x.device
        )
        new_state, new_conv, gated = torch.ops.deltanet4b.full_batched_direct(
            state_arg,
            mixed.to(torch.bfloat16).reshape(batch * qkv_dim),
            conv_arg,
            conv_weight,
            conv_bias,
            a.reshape(batch * vh),
            b.reshape(batch * vh),
            z.to(torch.bfloat16).reshape(batch * vh, D.DN_V_DIM),
            getattr(self, f"l{i}_dn_A_log").float(),
            getattr(self, f"l{i}_dn_dt_bias").float(),
            getattr(self, f"l{i}_dn_norm").float(),
            state_target,
            conv_target,
        )
        states_out[di] = new_state.reshape(batch, value_dim, D.DN_V_DIM)
        conv_out[di] = new_conv.reshape(
            batch,
            TILED_PMAX,
            qkv_dim // TILED_PMAX,
            D.DN_CONV_KERNEL - 1,
        )
        projected = self._linear(
            f"l{i}_dn_out",
            gated.reshape(batch, value_dim),
        )
        if self.world_size > 1:
            projected = functional_all_reduce(projected, "sum", self.world_size)
        return projected.unsqueeze(1)

    def _gqa(
        self,
        i: int,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        gi = D.gqa_index(i)
        batch = self.batch_size
        qh = self.td["gqa_q_heads"]
        kvh = self.td["gqa_kv_heads"]
        q_per_kv = qh // kvh
        hd = D.GQA_HEAD_DIM
        x2 = x.squeeze(1)

        q_proj = self._linear(f"l{i}_gqa_q", x2).reshape(batch, qh, 2 * hd)
        query, gate = q_proj.chunk(2, dim=-1)
        key = self._linear(f"l{i}_gqa_k", x2).reshape(batch, kvh, hd)
        value = self._linear(f"l{i}_gqa_v", x2).reshape(batch, kvh, hd)
        key = rms_norm(key, getattr(self, f"l{i}_gqa_k_norm")).unsqueeze(2)
        _, key = apply_rope(key, key, cos, sin)
        key = key.squeeze(2).to(torch.bfloat16)

        mask = (self.position_range < position).reshape(1, self.max_seq_len)
        attended_groups = []
        for group in range(kvh):
            suffix = str(gi) if kvh == 1 else f"{gi}_{group}"
            cache_k = getattr(self, f"kv_k{suffix}").reshape(
                batch * self.max_seq_len, hd
            )
            cache_v = getattr(self, f"kv_v{suffix}").reshape(
                batch * self.max_seq_len, hd
            )
            q_start = group * q_per_kv
            q_end = q_start + q_per_kv
            attended, _, _ = torch.ops.gqa4b.tail_stateful(
                query[:, q_start:q_end].reshape(
                    batch * q_per_kv, hd
                ).float(),
                gate[:, q_start:q_end].reshape(
                    batch * q_per_kv, hd
                ).float(),
                getattr(self, f"l{i}_gqa_q_norm").reshape(1, hd).float(),
                cos.reshape(1, D.ROPE_DIM).float(),
                sin.reshape(1, D.ROPE_DIM).float(),
                cache_k,
                cache_v,
                mask.float(),
                key[:, group],
                value[:, group].to(torch.bfloat16),
                position.reshape(1, 1).to(torch.int32),
                0,
            )
            attended_groups.append(
                attended.reshape(batch, q_per_kv, hd)
            )
        attended = torch.cat(attended_groups, dim=1)
        projected = self._linear(
            f"l{i}_gqa_o",
            attended.reshape(batch, qh * hd).to(torch.bfloat16),
        )
        if self.world_size > 1:
            projected = functional_all_reduce(projected, "sum", self.world_size)
        return projected.unsqueeze(1)

    def _mlp(self, i: int, x: torch.Tensor) -> torch.Tensor:
        x2 = x.squeeze(1)
        if self.fused_fp8_mlp:
            down = torch.ops.qwen4b_fp8.fused_mlp(
                x2.to(torch.bfloat16),
                getattr(self, f"l{i}_mlp_gate_up_q"),
                getattr(self, f"l{i}_mlp_gate_up_s"),
                getattr(self, f"l{i}_mlp_down_q"),
                getattr(self, f"l{i}_mlp_down_s_row"),
            )
            return down.unsqueeze(1)
        gate = self._linear(f"l{i}_mlp_gate", x2)
        up = self._linear(f"l{i}_mlp_up", x2)
        hidden = F.silu(gate) * up
        down = self._linear(f"l{i}_mlp_down", hidden)
        if self.world_size > 1:
            down = functional_all_reduce(down, "sum", self.world_size)
        return down.unsqueeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        position: torch.Tensor,
        deltanet_states: torch.Tensor,
        conv_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = F.embedding(input_ids, self.embed).unsqueeze(1)
        states_out = [
            torch.empty_like(deltanet_states[di])
            for di in range(D.NUM_DELTANET)
        ]
        conv_out = [
            torch.empty_like(conv_states[di])
            for di in range(D.NUM_DELTANET)
        ]
        cos = self.rope_cos.index_select(0, position.reshape(1)).reshape(
            1, 1, 1, D.ROPE_DIM
        )
        sin = self.rope_sin.index_select(0, position.reshape(1)).reshape(
            1, 1, 1, D.ROPE_DIM
        )

        for i in range(D.NUM_LAYERS):
            normalized = rms_norm(hidden, getattr(self, f"l{i}_input_norm"))
            if D.layer_type(i) == "deltanet":
                hidden = hidden + self._deltanet(
                    i,
                    normalized,
                    deltanet_states,
                    conv_states,
                    states_out,
                    conv_out,
                )
            else:
                hidden = hidden + self._gqa(i, normalized, cos, sin, position)
            normalized = rms_norm(hidden, getattr(self, f"l{i}_post_norm"))
            hidden = hidden + self._mlp(i, normalized)

        hidden = rms_norm(hidden, self.final_norm).squeeze(1)
        if self.fused_fp8_head:
            next_id = torch.ops.qwen4b_fp8.fused_head_argmax(
                hidden.to(torch.bfloat16),
                self.lm_head_q,
                self.lm_head_s_row,
            )
            return (
                next_id.squeeze(1).to(torch.long),
                torch.stack(states_out),
                torch.stack(conv_out),
            )
        local_logits = self._linear("lm_head", hidden)
        if self.world_size == 1:
            return (
                local_logits.argmax(dim=-1).to(torch.long),
                torch.stack(states_out),
                torch.stack(conv_out),
            )
        local_max, local_id = local_logits.max(dim=-1)
        global_max = functional_all_reduce(local_max, "max", self.world_size)
        vocab_offset = self.rank * self.td["vocab_per_rank"]
        global_id = local_id.to(torch.int32) + vocab_offset
        invalid = torch.full_like(global_id, -D.VOCAB)
        negative_winner = torch.where(
            local_max == global_max,
            -global_id,
            invalid,
        )
        next_id = -functional_all_reduce(
            negative_winner, "max", self.world_size
        )
        return next_id.to(torch.long), torch.stack(states_out), torch.stack(conv_out)


def _state_occupancy(states: torch.Tensor) -> list[int]:
    return [
        int((states[i].reshape(-1, states.shape[-1]) != 0).any(-1).sum())
        for i in range(states.shape[0])
    ]


def _kv_occupancy(module: StaticDecode4B) -> list[int]:
    counts = []
    local_kv_heads = module.td["gqa_kv_heads"]
    for gi in range(D.NUM_GQA):
        for kvh in range(local_kv_heads):
            suffix = str(gi) if local_kv_heads == 1 else f"{gi}_{kvh}"
            cache = getattr(module, f"kv_k{suffix}").detach().cpu()
            counts.append(
                int((cache.reshape(-1, cache.shape[-1]) != 0).any(-1).sum())
            )
    return counts


def _file_barrier(
    root: str | None,
    phase: str,
    replica_id: int,
    replica_count: int,
    timeout: float,
) -> None:
    if replica_count == 1:
        return
    if not root:
        raise ValueError("--barrier-dir is required when --replica-count > 1")
    os.makedirs(root, exist_ok=True)
    marker = os.path.join(root, f"{phase}.{replica_id}")
    with open(marker, "w") as f:
        f.write(f"{os.getpid()}\n")
    deadline = time.time() + timeout
    pattern = os.path.join(root, f"{phase}.*")
    while len(glob.glob(pattern)) < replica_count:
        if time.time() > deadline:
            found = sorted(os.path.basename(p) for p in glob.glob(pattern))
            raise TimeoutError(
                f"barrier {phase!r} timed out: found {found}, "
                f"expected {replica_count} replicas"
            )
        time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/models/Qwen3.5-4B")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--bench-iters", type=int, default=20)
    parser.add_argument(
        "--dtype",
        choices=("fp8", "fp8-mlp-head", "fp8-lm-head", "bf16"),
        default="bf16",
        help=(
            "fp8-mlp-head is the measured DP8 throughput winner; "
            "fp8-lm-head quantizes only the vocabulary projection; "
            "fp8 quantizes every projection"
        ),
    )
    parser.add_argument("--replica-id", type=int, default=0)
    parser.add_argument("--replica-count", type=int, default=1)
    parser.add_argument("--barrier-dir")
    parser.add_argument("--barrier-timeout", type=float, default=900.0)
    parser.add_argument("--output-json")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--fused-fp8-mlp", action="store_true")
    parser.add_argument("--fused-fp8-head", action="store_true")
    args = parser.parse_args()

    if args.dtype != "bf16" and args.batch_size > 128:
        raise ValueError("the FP8 kernel currently supports batch sizes <= 128")
    if args.max_seq_len < 2:
        raise ValueError("--max-seq-len must be at least 2")
    if not 0 <= args.replica_id < args.replica_count:
        raise ValueError("--replica-id must be in [0, --replica-count)")

    import torch_neuronx  # noqa: F401

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    D.tp_dims(world_size)
    device = torch.neuron.current_device()

    if rank == 0:
        print(
            f"Qwen3.5-4B static decode: TP={world_size} "
            f"BS={args.batch_size} S={args.max_seq_len} dtype={args.dtype}",
            flush=True,
        )

    load_start = time.time()
    weights = load_sharded_weights(args.model_path, rank, world_size)
    module = StaticDecode4B(
        weights,
        args.max_seq_len,
        world_size,
        rank,
        args.batch_size,
        fp8_scope={
            "fp8": "all",
            "fp8-mlp-head": "mlp-head",
            "fp8-lm-head": "lm-head",
            "bf16": "none",
        }[args.dtype],
        fused_fp8_mlp=args.fused_fp8_mlp,
        fused_fp8_head=args.fused_fp8_head,
    )
    del weights
    module = module.to(device).eval()
    module_bytes = sum(b.numel() * b.element_size() for b in module.buffers())
    if rank == 0:
        print(
            f"module loaded in {time.time() - load_start:.1f}s "
            f"({module_bytes / 2**30:.2f} GiB/rank)",
            flush=True,
        )

    compiled = torch.compile(
        module,
        backend="neuron",
        fullgraph=True,
        dynamic=False,
    )
    batch = args.batch_size
    qkv_dim = D.DN_QKV_DIM // world_size
    value_dim = D.DN_VALUE_DIM // world_size
    states = torch.zeros(
        D.NUM_DELTANET,
        batch,
        value_dim,
        D.DN_V_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    conv = torch.zeros(
        D.NUM_DELTANET,
        batch,
        TILED_PMAX,
        qkv_dim // TILED_PMAX,
        D.DN_CONV_KERNEL - 1,
        dtype=torch.bfloat16,
        device=device,
    )
    input_ids = (
        (
            torch.arange(batch, device=device, dtype=torch.long)
            + args.replica_id * batch
        )
        % 1024
    ) + 1
    position = torch.tensor(1, device=device, dtype=torch.long)

    dist.barrier()
    compile_start = time.time()
    with torch.no_grad():
        next_id, states, conv = compiled(input_ids, position, states, conv)
        torch.neuron.synchronize()
    compile_seconds = time.time() - compile_start
    first_ids = next_id.detach().cpu().tolist()
    if rank == 0:
        print(f"first step/compile: {compile_seconds:.1f}s", flush=True)

    if args.compile_only:
        dist.barrier()
        dist.destroy_process_group()
        return

    with torch.no_grad():
        for _ in range(args.warmup):
            next_id, states, conv = compiled(input_ids, position, states, conv)
        torch.neuron.synchronize()
    dist.barrier()
    if rank == 0:
        _file_barrier(
            args.barrier_dir,
            "ready",
            args.replica_id,
            args.replica_count,
            args.barrier_timeout,
        )
    dist.barrier()

    start = time.time()
    with torch.no_grad():
        for _ in range(args.bench_iters):
            next_id, states, conv = compiled(input_ids, position, states, conv)
        torch.neuron.synchronize()
    elapsed = time.time() - start
    dist.barrier()
    if rank == 0:
        _file_barrier(
            args.barrier_dir,
            "finished",
            args.replica_id,
            args.replica_count,
            args.barrier_timeout,
        )
    dist.barrier()

    tpot_ms = elapsed * 1000.0 / args.bench_iters
    throughput = batch * args.bench_iters / elapsed
    occupancy = _state_occupancy(states.detach().cpu())
    if any(v == 0 for v in occupancy):
        raise RuntimeError(f"empty DeltaNet recurrent state: {occupancy}")
    kv_counts = _kv_occupancy(module)
    if any(v == 0 for v in kv_counts):
        raise RuntimeError(f"empty GQA KV cache: {kv_counts}")

    result = {
        "model": "Qwen/Qwen3.5-4B",
        "instance": "trn2.3xlarge",
        "topology": f"DP{args.replica_count}-TP{world_size}-LNC2",
        "dp": args.replica_count,
        "tp": world_size,
        "replica_id": args.replica_id,
        "batch_size": batch,
        "aggregate_batch_size": batch * args.replica_count,
        "max_seq_len": args.max_seq_len,
        "dtype": args.dtype,
        "fused_fp8_mlp": args.fused_fp8_mlp,
        "fused_fp8_head": args.fused_fp8_head,
        "compile_seconds": compile_seconds,
        "module_gib_per_rank": module_bytes / 2**30,
        "module_gib_per_replica": module_bytes * world_size / 2**30,
        "tpot_ms": tpot_ms,
        "throughput_tok_s": throughput,
        "first_ids": first_ids,
        "final_ids": next_id.detach().cpu().tolist(),
        "dn_nonzero_rows": occupancy,
        "kv_nonzero_rows": kv_counts,
    }
    if rank == 0:
        print(
            f"BENCH BS={batch} seq={args.max_seq_len}: "
            f"TPOT {tpot_ms:.2f} ms/tok | throughput {throughput:.1f} tok/s",
            flush=True,
        )
        print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2, sort_keys=True)
                f.write("\n")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
