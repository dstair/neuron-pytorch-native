# DeepSeek V4 Flash on Trainium2 (PyTorch Native)

A PyTorch-Native (eager + `torch.compile(backend="neuron")`) inference port of
**DeepSeek V4 Flash** — a 284B-total / 13B-active sparse-MoE model with MLA-style
attention and Hyper-Connections (HC) — to AWS Trainium2. It runs the full model
via expert- and tensor-parallelism across a `trn2.48xlarge`, plus a single-device
path for the tiny-random config.

DeepSeek V4 is not yet in a released `transformers`, so the model is implemented
standalone in these scripts (config + weights loaded from a HuggingFace-format
checkpoint).

## Model

```
284B total params, 13B active (46-shard checkpoint, ~160GB FP4/FP8 mixed)
MLA-style attention (low-rank Q/KV projections, decoupled RoPE)
Hyper-Connections (HC): learned sinkhorn-normalized residual mixing pre/post block
MoE: 256 routed experts + shared expert, top-k gating with softplus scoring
Vocab 129280
```

The published checkpoint stores expert weights as FP4 (packed int8) and other
params as FP8. FP8 is not supported in this pre-GA PyTorch-Native build, so
`preprocess_weights.py` dequantizes everything to BF16 and pre-shards per rank.

## Layout

```
run_deepseek_v4.py           single-device inference (tiny-random config; smoke test)
run_deepseek_v4_tp.py        full model, TP=64 expert+tensor parallel (trn2.48xlarge)
run_deepseek_v4_hybrid.py    experimental TP=4 / EP=16 hybrid parallelism
preprocess_weights.py        dequantize FP4/FP8 -> BF16 and shard for TP=64
preprocess_weights_hybrid.py same, for the TP=4/EP=16 hybrid layout
```

## Parallelism

Uses PyTorch DTensor `parallelize_module` + `torch.compile` + `torchrun`:

- **Attention:** Q up-projection ColwiseParallel (shards heads), output projection
  RowwiseParallel (all-reduce); low-rank down-projections and KV norm replicated.
- **MoE:** experts partitioned across ranks (each rank owns `n_experts / world_size`),
  local expert outputs combined with an all-reduce; shared expert replicated.
- **Gate, embeddings, LM head, HC:** replicated (small relative to experts).

The MoE gate + dispatch runs entirely on Neuron (scores, softplus, top-k with int32
indices, mask-based dispatch) — no per-layer CPU round-trips.

## Running

Full model on `trn2.48xlarge` (TP=64), from pre-sharded BF16 weights:

```bash
# 1. Dequantize + shard the HF checkpoint for TP=64
python3 preprocess_weights.py \
    --input-path /scratch/DeepSeek-V4-Flash \
    --output-path /scratch/DeepSeek-V4-Flash-BF16-TP64 \
    --tp-degree 64

# 2. Run TP inference
torchrun --nproc-per-node 64 run_deepseek_v4_tp.py \
    --model-path /scratch/DeepSeek-V4-Flash-BF16-TP64
```

Single-device smoke test with the tiny-random config (no real weights needed):

```bash
python3 run_deepseek_v4.py
# or the TP script against the tiny HF model:
torchrun --nproc-per-node 4 run_deepseek_v4_tp.py --model-id yujiepan/deepseek-v4-tiny-random
```

`preprocess_weights.py` prints an `s5cmd sync ... s3://${S3_MODEL_BUCKET}/...`
command to back up the sharded weights; `source ../../.env` to set the bucket
(see the repo root README).

## Results (trn2.48xlarge, TP=64, BF16, 8-token generation, NEFF cached)

Optimization progression on the full model:

| Step | TTFT | Decode tok/s | Total (8 tok) | Change |
|---|---|---|---|---|
| No cache | 34s | — (128s/tok) | ~1060s | full recompute each step |
| Dynamic KV cache | 12s | 0.19 | 48s | NEFF cache reuse |
| Neuron-native MoE | 6.3s | 0.22 | 38.5s | zero CPU round-trips |
| Batched experts (bmm) | 6.5s | 0.23 | 37.5s | no per-expert Python loop |
| Compiled HC pipeline | 5.3s | 0.40 | 22.8s | `torch.compile` fuses sinkhorn/HC |
| **Multi-stream overlap** | **4.3s** | **0.50** | **18.3s** | comm/compute overlap |

Cumulative from the first cached baseline: **decode 0.19 → 0.50 tok/s (2.6×),
total 48s → 18.3s (2.6×)**.

### Notable findings

- **HC sinkhorn was the biggest single win** (+57% decode): the 20-iteration
  normalization loop launched ~3,440 tiny kernels/forward; `torch.compile`
  collapses it to one fused kernel per call.
- **Decode is all-reduce-bound**, not compute- or launch-bound: at TP=64 each token
  incurs 86 all-reduces (attention + MoE per layer × 43 layers). Batching experts
  and moving the gate on-device removed the earlier bottlenecks; the collectives
  now dominate. Lower TP would cut collectives but must still fit memory.
- **TP=4 / EP=16 hybrid OOMs** at model load: experts are EP-partitioned but
  replicated across the 4 TP ranks within a device (4 × 43.6GB ≫ 96GB/device). The
  hybrid scripts are kept for reference; TP=64 is the working configuration. See
  `run_deepseek_v4_hybrid.py` for the layout and the OOM analysis in-code.

## Status

Research bring-up. Correctness is validated structurally (real weights produce
non-NaN, well-scaled logits; tiny-random produces the expected all-zero argmax).
Decode throughput is modest — eager-mode collectives dominate — and the remaining
levers are NKI attention/RoPE/RMSNorm kernels and reduced-TP configurations.
