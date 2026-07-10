#!/usr/bin/env python3
"""
Qwen3.5-35B-A3B (MoE) architecture dimensions — single source of truth.

All values VERIFIED against the real checkpoint tensor shapes
(s3://${S3_MODEL_BUCKET}/Qwen3.5-35B-A3B, HF Qwen/Qwen3.5-35B-A3B), 2026-06-26.

HF arch class: Qwen3_5MoeForConditionalGeneration  (model_type qwen3_5_moe).
This is the text backbone of a VLM; we serve the text path only.

Layer pattern: [DeltaNet × 3, GQA × 1] × 10 = 40 layers (full_attention_interval=4),
identical *structure* to the 27B but different counts/widths, and EVERY layer's
MLP is replaced by a 256-expert top-8 sparse MoE + a sigmoid-gated shared expert.

This module deliberately mirrors the constants block of the 27B
examples/qwen3_6/static_decode.py so the two stay easy to diff, but the two
model implementations are kept SEPARATE (we expect them to diverge).
"""

# ── Global ────────────────────────────────────────────────────────────────────
NUM_LAYERS = 40
FULL_ATTN_INTERVAL = 4          # every 4th layer (idx % 4 == 3) is GQA; rest DeltaNet
NUM_GQA = NUM_LAYERS // FULL_ATTN_INTERVAL          # 10
NUM_DELTANET = NUM_LAYERS - NUM_GQA                 # 30
HIDDEN = 2048
VOCAB = 248320
RMS_EPS = 1e-6
TIE_WORD_EMBEDDINGS = False

# ── RoPE (matches 27B fix: PARTIAL rotary, theta 1e7) ───────────────────────────
GQA_HEAD_DIM = 256
PARTIAL_ROTARY_FACTOR = 0.25
ROPE_DIM = int(GQA_HEAD_DIM * PARTIAL_ROTARY_FACTOR)   # 64
ROPE_THETA = 10000000.0                                # 1e7
MAX_POSITION_EMBEDDINGS = 262144

# ── DeltaNet (linear_attention) ─────────────────────────────────────────────────
# in_proj_qkv [8192, 2048] = q(16*128) || k(16*128) || v(32*128)
DN_K_HEADS = 16                 # linear_num_key_heads
DN_V_HEADS = 32                 # linear_num_value_heads
DN_K_DIM = 128                  # linear_key_head_dim
DN_V_DIM = 128                  # linear_value_head_dim
DN_KEY_DIM = DN_K_HEADS * DN_K_DIM      # 2048 (q and k each)
DN_VALUE_DIM = DN_V_HEADS * DN_V_DIM    # 4096
DN_QKV_DIM = 2 * DN_KEY_DIM + DN_VALUE_DIM   # 8192  (in_proj_qkv out)
DN_Z_DIM = DN_VALUE_DIM                  # in_proj_z [4096, 2048]
DN_CONV_KERNEL = 4              # linear_conv_kernel_dim
DN_HEAD_GROUP = DN_V_HEADS // DN_K_HEADS  # 2 (v-heads per k-head)
# in_proj_a / in_proj_b: [32, 2048]; A_log/dt_bias: [32]; norm: [128]

# ── GQA (full_attention) ────────────────────────────────────────────────────────
GQA_Q_HEADS = 16                # num_attention_heads
GQA_KV_HEADS = 2                # num_key_value_heads
ATTN_OUTPUT_GATE = True         # q_proj is doubled: [16*256*2, 2048] = [8192, 2048]
GQA_Q_DIM = GQA_Q_HEADS * GQA_HEAD_DIM * 2   # 8192 (query || gate)
GQA_KV_DIM = GQA_KV_HEADS * GQA_HEAD_DIM      # 512 (k_proj, v_proj each)
GQA_O_IN = GQA_Q_HEADS * GQA_HEAD_DIM         # 4096 (o_proj in)
# q_norm / k_norm: [256] (per-head RMSNorm over head_dim)

# ── MoE (replaces dense MLP on ALL 40 layers) ───────────────────────────────────
NUM_EXPERTS = 256               # num_experts
TOP_K = 8                       # num_experts_per_tok
MOE_INTER = 512                 # moe_intermediate_size (per routed expert)
NORM_TOPK_PROB = True           # normalize_top_k_affinities
ROUTER_ACT = "softmax"          # router_config.act_fn
# Packed expert weights (checkpoint layout):
#   experts.gate_up_proj : [E, 2*MOE_INTER, H] = [256, 1024, 2048]  (gate||up fused)
#   experts.down_proj    : [E, H, MOE_INTER]   = [256, 2048, 512]
#   gate.weight (router) : [E, H]              = [256, 2048]
# Shared expert (sigmoid-gated, Qwen3.5-specific):
SHARED_INTER = 512              # shared_expert_intermediate_size
#   shared_expert.{gate,up}_proj : [SHARED_INTER, H] = [512, 2048]
#   shared_expert.down_proj      : [H, SHARED_INTER] = [2048, 512]
#   shared_expert_gate           : [1, H]            = [1, 2048]  -> sigmoid(x@g.T)*mlp(x)

# ── MTP (multi-token-prediction head; optional spec-decode draft) ────────────────
MTP_NUM_LAYERS = 1              # mtp_num_hidden_layers (a FULL MoE layer; heavy)
MTP_USE_DEDICATED_EMBEDDINGS = False


def layer_type(i: int) -> str:
    """Layer pattern: [DeltaNet×3, GQA×1] × 10."""
    return "gqa" if i % FULL_ATTN_INTERVAL == (FULL_ATTN_INTERVAL - 1) else "deltanet"


def deltanet_index(layer_idx: int) -> int:
    """Map absolute layer index to DeltaNet state index (0..29)."""
    block = layer_idx // FULL_ATTN_INTERVAL
    offset = layer_idx % FULL_ATTN_INTERVAL
    return block * (FULL_ATTN_INTERVAL - 1) + offset


def gqa_index(layer_idx: int) -> int:
    """Map absolute layer index to GQA cache index (0..9)."""
    return layer_idx // FULL_ATTN_INTERVAL


# ── Per-core (TP) sharding helper ───────────────────────────────────────────────
def tp_dims(world_size: int) -> dict:
    """Return per-core dimensions for a given TP degree, plus sharding notes.

    KV-head wrinkle: GQA_KV_HEADS=2 does NOT divide TP=4. KV heads must be
    REPLICATED across cores when world_size > GQA_KV_HEADS (each KV head shared
    by world_size//GQA_KV_HEADS cores). This differs from the 27B (4 KV heads /
    4 cores = clean 1:1). DeltaNet K/V heads and MoE experts DO divide cleanly.
    """
    assert NUM_EXPERTS % world_size == 0, "experts must divide TP"
    assert DN_K_HEADS % world_size == 0 and DN_V_HEADS % world_size == 0
    kv_replication = max(1, world_size // GQA_KV_HEADS)   # cores sharing one KV head
    kv_heads_per_core = max(1, GQA_KV_HEADS // world_size)
    return {
        "world_size": world_size,
        # MoE: expert-parallel (validated in test_moe_oracle_cpu.py)
        "experts_per_core": NUM_EXPERTS // world_size,    # 64 @ TP4
        # DeltaNet
        "dn_k_heads": DN_K_HEADS // world_size,           # 4 @ TP4
        "dn_v_heads": DN_V_HEADS // world_size,           # 8 @ TP4
        "dn_qkv_dim": DN_QKV_DIM // world_size,           # 2048 @ TP4
        "dn_value_dim": DN_VALUE_DIM // world_size,       # 1024 @ TP4
        "dn_head_group": DN_HEAD_GROUP,                   # 2
        # GQA
        "gqa_q_heads": GQA_Q_HEADS // world_size,         # 4 @ TP4
        "gqa_q_dim": GQA_Q_DIM // world_size,             # 2048 @ TP4 (query||gate)
        "gqa_kv_heads_per_core": kv_heads_per_core,       # 0 -> see replication
        "gqa_kv_replication": kv_replication,             # 2 @ TP4 (replicate KV)
        "gqa_o_in_per_core": GQA_O_IN // world_size,      # 1024 @ TP4
    }


def load_from_config(config_path: str):
    """Override this module's globals from a HF config.json (text_config).

    Lets the SAME harness run the real 35B and the tiny-random debug model
    (which has hidden 8, head_dim 32, 4 KV heads, etc.). Mutates module globals
    in place; call once at startup before building the module.
    """
    import json as _json
    g = globals()
    c = _json.load(open(config_path))
    tc = c.get("text_config", c)
    g["NUM_LAYERS"] = tc["num_hidden_layers"]
    g["FULL_ATTN_INTERVAL"] = tc["full_attention_interval"]
    g["NUM_GQA"] = sum(1 for i in range(g["NUM_LAYERS"]) if layer_type(i) == "gqa")
    g["NUM_DELTANET"] = g["NUM_LAYERS"] - g["NUM_GQA"]
    g["HIDDEN"] = tc["hidden_size"]
    g["VOCAB"] = tc["vocab_size"]
    g["RMS_EPS"] = tc.get("rms_norm_eps", 1e-6)
    g["GQA_HEAD_DIM"] = tc["head_dim"]
    rp = tc.get("rope_parameters", tc)
    g["PARTIAL_ROTARY_FACTOR"] = rp.get("partial_rotary_factor",
                                        tc.get("partial_rotary_factor", 0.25))
    g["ROPE_DIM"] = int(g["GQA_HEAD_DIM"] * g["PARTIAL_ROTARY_FACTOR"])
    g["ROPE_THETA"] = float(rp.get("rope_theta", 1e7))
    # DeltaNet
    g["DN_K_HEADS"] = tc["linear_num_key_heads"]
    g["DN_V_HEADS"] = tc["linear_num_value_heads"]
    g["DN_K_DIM"] = tc["linear_key_head_dim"]
    g["DN_V_DIM"] = tc["linear_value_head_dim"]
    g["DN_KEY_DIM"] = g["DN_K_HEADS"] * g["DN_K_DIM"]
    g["DN_VALUE_DIM"] = g["DN_V_HEADS"] * g["DN_V_DIM"]
    g["DN_QKV_DIM"] = 2 * g["DN_KEY_DIM"] + g["DN_VALUE_DIM"]
    g["DN_Z_DIM"] = g["DN_VALUE_DIM"]
    g["DN_CONV_KERNEL"] = tc["linear_conv_kernel_dim"]
    g["DN_HEAD_GROUP"] = g["DN_V_HEADS"] // g["DN_K_HEADS"]
    # GQA
    g["GQA_Q_HEADS"] = tc["num_attention_heads"]
    g["GQA_KV_HEADS"] = tc["num_key_value_heads"]
    g["ATTN_OUTPUT_GATE"] = tc.get("attn_output_gate", True)
    g["GQA_Q_DIM"] = g["GQA_Q_HEADS"] * g["GQA_HEAD_DIM"] * (2 if g["ATTN_OUTPUT_GATE"] else 1)
    g["GQA_KV_DIM"] = g["GQA_KV_HEADS"] * g["GQA_HEAD_DIM"]
    g["GQA_O_IN"] = g["GQA_Q_HEADS"] * g["GQA_HEAD_DIM"]
    # MoE
    g["NUM_EXPERTS"] = tc["num_experts"]
    g["TOP_K"] = tc["num_experts_per_tok"]
    g["MOE_INTER"] = tc["moe_intermediate_size"]
    g["NORM_TOPK_PROB"] = tc.get("normalize_top_k_affinities", True)
    g["SHARED_INTER"] = tc["shared_expert_intermediate_size"]
    g["TIE_WORD_EMBEDDINGS"] = c.get("tie_word_embeddings", False)
    return c


if __name__ == "__main__":
    import json
    print("Qwen3.5-35B-A3B dims")
    for w in (1, 4, 8):
        print(f"\nTP={w}:")
        print(json.dumps(tp_dims(w), indent=2))
