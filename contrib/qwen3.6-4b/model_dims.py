"""Qwen3.5-4B / Qwen3.6-4B text-backbone dimensions and TP layouts."""

NUM_LAYERS = 32
FULL_ATTN_INTERVAL = 4
NUM_GQA = 8
NUM_DELTANET = 24

HIDDEN = 2560
INTERMEDIATE = 9216
VOCAB = 248320
RMS_EPS = 1e-6

DN_K_HEADS = 16
DN_V_HEADS = 32
DN_K_DIM = 128
DN_V_DIM = 128
DN_KEY_DIM = DN_K_HEADS * DN_K_DIM
DN_VALUE_DIM = DN_V_HEADS * DN_V_DIM
DN_QKV_DIM = 2 * DN_KEY_DIM + DN_VALUE_DIM
DN_CONV_KERNEL = 4

GQA_Q_HEADS = 16
GQA_KV_HEADS = 4
GQA_HEAD_DIM = 256
ROPE_DIM = 64
ROPE_THETA = 10_000_000.0


def layer_type(i: int) -> str:
    return "gqa" if i % FULL_ATTN_INTERVAL == FULL_ATTN_INTERVAL - 1 else "deltanet"


def deltanet_index(layer_idx: int) -> int:
    return (layer_idx // FULL_ATTN_INTERVAL) * 3 + layer_idx % FULL_ATTN_INTERVAL


def gqa_index(layer_idx: int) -> int:
    return layer_idx // FULL_ATTN_INTERVAL


def tp_dims(world_size: int) -> dict[str, int]:
    if world_size not in (1, 2, 4):
        raise ValueError(
            "Qwen3.5-4B is currently supported at TP=1, TP=2, or TP=4, "
            f"got TP={world_size}"
        )
    return {
        "dn_k_heads": DN_K_HEADS // world_size,
        "dn_v_heads": DN_V_HEADS // world_size,
        "dn_qkv_dim": DN_QKV_DIM // world_size,
        "dn_value_dim": DN_VALUE_DIM // world_size,
        "gqa_q_heads": GQA_Q_HEADS // world_size,
        "gqa_kv_heads": GQA_KV_HEADS // world_size,
        "mlp_dim": INTERMEDIATE // world_size,
        "vocab_per_rank": VOCAB // world_size,
    }
