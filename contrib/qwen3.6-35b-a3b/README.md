# Qwen3.6-35B-A3B (MoE) on Trainium2 — PyTorch Native

## 1. Overview

A PyTorch-Native inference implementation of the sparse-MoE **Qwen3.6-35B-A3B**
(~3B active parameters of 35B) on a single Trainium2 device (`trn2.3xlarge`).
It shares the DeltaNet + GQA backbone of the dense sibling but replaces the dense
MLP with a 256-expert, top-8 mixture of experts, and targets a fixed long-context
(20,000-token) regime.

```
40 layers = [DeltaNet × 3, GQA × 1] × 10   (full attention every 4th layer)
hidden 2048, vocab 248320, RMSNorm eps 1e-6, RoPE partial-64 @ theta 1e7
  DeltaNet (30 layers): 16 K-heads, 32 V-heads, k/v dim 128, depthwise conv1d k=4
  GQA (10 layers):      16 Q-heads, 2 KV-heads, head_dim 256, sigmoid output gate
  MoE (all 40 layers):  256 experts, top-8, moe_inter 512, + shared expert
```

> **Naming.** Published as [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B);
> its architecture class is `Qwen3_5MoeForConditionalGeneration` (`model_type:
> qwen3_5_moe`) — "3.5" names the architecture family, "3.6" the release. Same
> architecture, so this runs on the HF checkpoint unchanged.

The whole 40-layer decode model compiles to a single NEFF via
`torch.compile(fullgraph=True, backend="neuron")`; the long-context prefill path
uses four coarse 10-layer regions. Entry point: `static_decode_35b.py`; verified
architecture constants and the TP sharding plan live in `model_dims.py`.

**Quick start** (BS=1 decode):
```bash
DN_NKI=1 MOE_SPARSE=1 MOE_DECODE_TP=1 GQATAIL=1 DNBATCHED_V2=1 \
  torchrun --nproc-per-node=4 static_decode_35b.py \
    --num-layers 40 --max-seq-len 2048 --batch-size 1 --bench
```

## 2. Best throughput (trn2.3xlarge)

| Phase | Best | Config | Recipe |
|---|---|---|---|
| **Prefill** | **2,276.9 agg prompt tok/s** | stable **C32**, BS=2, N=20,000, bucket 1024, TP=4/LNC=2, O1 (+8.5% over paired-C16 @ 2,089.7) | [PREFILL_RECIPE.md](PREFILL_RECIPE.md) |
| **Decode** | **343.6 tok/s @ BS=128** | FP8 `block_pow2_coalesced` MoE + tiled DeltaNet conv, TP=8/LNC=1, O2 (bit-identical to untiled) | [DECODE_RECIPE.md](DECODE_RECIPE.md) |

Other reference points: latency-optimal decode **48.9 tok/s @ BS=1** (true-sparse
MoE + `MOE_DECODE_TP`); BF16 full-graph decode **320.6 tok/s @ BS=32**.

Full methodology, ablations, per-optimization progression, HBM/DMA attribution,
and the NxDI (XLA) reference comparison are in **[BENCHMARK.md](BENCHMARK.md)**.

The two recipes are reproducible end-to-end (host requirements, compile, bench,
kickoff); all environment-specific values are read from an ignored `.env` (copy
`.env.example`).

## 3. Kernel flags

All are environment variables read at import/compile time. Defaults are off/`0`
unless noted; combine per the recipes.

### DeltaNet (linear attention)
| Flag | Effect |
|---|---|
| `DN_NKI=1` | DeltaNet NKI kernel — **required past ~20 layers** (pure-torch recurrence trips a compiler tiling assertion) |
| `DNBATCHED_V2=1` | DMA-coalesced batched DeltaNet decode (batches over heads) |
| `DN_DIRECT_STATE_OUT=1` | Full-graph decode: write BF16 DeltaNet/conv state directly to disjoint output buffers (skips whole-state clone + FP32→BF16 copy) |
| `DN_TILED_CONV=1` | Tiled conv-state layout + coalesced `mixed_qkv` DMA (decode) — ~+15–19% at BS=32/128, bit-identical |
| `DN_CHUNK_NKI=1` | Chunked DeltaNet **prefill** NKI kernel (stable long-context) |
| `CHUNK_SIZE=16\|32` | DeltaNet prefill chunk size (default 16). `32` = the faster stable-C32 path (pair with `DN_STABLE_C32=0`) |
| `DN_STABLE_C32=0` | Use the numerically-stable block-diagonal C32 inverse (the +8.5% prefill path). Default `1` |
| `DN_PAIRED_BATCH=1` | Paired two-prompt C16 DeltaNet batching (the C16 prefill baseline path) |
| `DN_WIDE_CONV=1` | Wide-convolution DeltaNet variant |
| `DN_K_HEADS`, `DN_V_HEADS` | Override per-rank DeltaNet K/V head counts (topology/sharding) |
| `DN_PASSTHROUGH=1` | Diagnostic: replace DeltaNet with identity |

### GQA (full attention)
| Flag | Effect |
|---|---|
| `GQATAIL=1` | Fused GQA attention-tail kernel (decode) |
| `GQA_CTE_PREFILL=1` | Prefix-aware nkilib CTE attention for prefill (requires `GQA_DYNAMIC_ROPE_KV=1`; needs a head-dim-256 `attention_cte` nkilib) |
| `GQA_DYNAMIC_ROPE_KV=1` | Dynamic RoPE + KV update for the CTE prefill path |
| `GQA_FLASH_PREFILL=1` | Local fixed-KMAX flash-GQA prefill kernel (older path) |
| `GQA_STATEFUL_KV=1` | Full-graph decode: keep BF16 K/V caches as aliased module state, append only current rows |
| `GQA_Q_HEADS` | Override per-rank GQA query-head count |

### MoE
| Flag | Effect |
|---|---|
| `MOE_SPARSE=1` | True-sparse dispatch (gathers only top-8 experts) — ~2× at BS=1 |
| `MOE_DECODE_TP=1` | BF16 BS=1 decode: shard each expert's intermediate width across TP ranks (avoids dummy non-local expert reads) |
| `MOE_CTE=1` | Long-token nkilib context-encoding MoE kernel for prefill |
| `MOE_CTE_NKI_PACK=1` | Fused NKI route packer inside the CTE call (replaces compiled `one_hot().cumsum()`); used by the validated BS=2/4 prefill |
| `MOE_CTE_BLOCK=512` | CTE MoE block size |
| `MOE_PREFILL_CHUNK` | MoE prefill chunk size (default 128) |
| `MOE_NKILIB=1` | nkilib fused MoE path (BF16) |
| `MOE_FP8=1` | Older per-row FP8 grouped-matvec MoE path |
| `MOE_FUSED_W8=fp8\|int8` | High-batch full-graph decode: fused all-expert path using the official block-scaled FP8 (or symmetric INT8) experts |
| `MOE_FUSED_W8_FP8_IMPL=` | FP8 variant for `MOE_FUSED_W8`: `row` / `dual` / `block_pow2` / **`block_pow2_coalesced`** (the validated high-batch decode kernel) |
| `MOE_FUSED_W8_FP8_LAYER_START`, `_LAYER_LIMIT` | Restrict FP8 experts to a layer range (defaults 0 / 40) — for A/B and layer-limited runs |
| `MOE_W8_TENSOR_SCALE=1` | Experiment (**negative**, default off): dequant to BF16 with per-block scale + PSUM-accumulate; removes Vector scale-adds but is slower |
| `MOE_W8_RESIDUAL_FP32=1` | Keep the routed accumulation residual in FP32 |
| `MOE_CTE_RETURN_ROUTED`, `MOE_CTE_SYNC_BEFORE_SHARED` | CTE MoE variants/diagnostics |
| `MOE_SHARED_ONLY=1` | Diagnostic: run only the shared expert |
| `NOREDUCE=1` | Diagnostic: skip the MoE all-reduce |
| `MOE_OFFICIAL_FP8_REFERENCE=1` | Build the exact official-FP8 reference for correctness comparison |

### Decode graph
| Flag | Effect |
|---|---|
| `DECODE_FULLGRAPH=1` | Compile embedding + all layers + state updates + LM head + greedy token selection into one NEFF |
| `DECODE_SHARDED_LM_HEAD=1` | Vocab-shard the LM head across TP ranks (two all-reduces select the exact global top-1); large HBM reduction |

### Prefill
| Flag | Effect |
|---|---|
| `BUCKET_COMPILE=1` | Compile the bucketed prefill graph (vs eager). Default on |
| `PREFILL_GEN=1` | Iterative-prefill generation (re-prefills the growing sequence each step; used for C32 coherence checks) |
| `PREFILL_FINGERPRINT=1` | Print a per-run token-ID/state fingerprint for correctness comparison |

### Precision / topology / runtime
| Flag | Effect |
|---|---|
| `FP8=1` | Enable FP8 in the standalone kernel tests |
| `NEURON_LOGICAL_NC_CONFIG=2\|1` | Logical NeuronCore config: `2` (default, TP=4, 4 cores) or `1` (TP=8, 8 cores) |
| `QWEN35_MODEL_PATH`, `QWEN35_FP8_MODEL_PATH` | BF16 base and FP8-expert checkpoint directories (or use `--model-path` / `--expert-model-path`) |
| `BATCH_SIZE`, `S`, `T`, `RANK` | Batch size / seq length / token count / TP rank for standalone kernel tests |
| `G_SCALE`, `ZERO_ROWS` | DeltaNet chunk-test knobs (gate scale, zero-row fraction) |

> Additional `PROFILE_*`, `DN_CAPTURE_*`, `PREFILL_TRACE_*`, `BENCH_*`, and
> `CROSS_TARGET_*` variables exist for profiling, capture, and cross-compile
> tooling; see `deploy/`, `debug/`, and `kernels/tests/`.
