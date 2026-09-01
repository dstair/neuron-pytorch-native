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
| **Prefill (FP8 CTE MoE)** | **4,383.9 agg prompt tok/s** | **FP8 CTE MoE** (`MOE_CTE_FP8=1`) with the **output-block scale grid + PSUM hoist** and the **uncapped route packer** (`MOE_CTE_MAX_TOKENS=0` → 6,144 tokens/MoE call), on the same packed **C32 n=4** DeltaNet at **TP=8/LNC=1**; BS=6, **N=10,000**, bucket 1024, O1, beta-5, **5.08 GB/core** (vs 8.94 BF16). Fingerprint `top5=[220,13,197,198,62]`. **Not a like-for-like swap of the row below — different BS *and* N; see the caveat.** | [PREFILL_RECIPE.md](PREFILL_RECIPE.md) §3d |
| Prefill (BF16, the long-context headline) | 4,097.9 agg prompt tok/s | packed **C32 n=4** at **TP=8/LNC=1**, correct per-GQA-layer in-place rope-KV write, both vocab tensors load-time sharded (`PREFILL_SHARDED_LM_HEAD=1 PREFILL_SHARDED_EMBED=1`), `--scratchpad-page-size-mb 256`, BS=2, **N=20,000**, bucket 1024, O1, **beta-5 container** (reproduced 4,095.2, within 0.07%; **+2.4%** over beta-4's 4,002.1; bit-identical fingerprint) | [PREFILL_RECIPE.md](PREFILL_RECIPE.md) §3c |
| Prefill (TP=4/LNC=2) | 2,791.7 agg prompt tok/s | stable **C32 + 4-stream block-diagonal pack, SBUF-resident, hoisted-transpose finish** (`DN_PACK_C32=1 DN_PACK_N=4`), BS=2, N=20,000, bucket 1024, TP=4/LNC=2, O1 (+22.6% over unpacked C32 @ 2,276.9; +33.6% over paired-C16 @ 2,089.7) | [PREFILL_RECIPE.md](PREFILL_RECIPE.md) |
| **Decode** | **681.3 tok/s @ BS=128** | FP8 `block_ob_coalesced` (Reduction B1) MoE + tiled DeltaNet conv + per-layer DeltaNet state (`DN_PERLAYER_STATE=1`), TP=8/LNC=1, O1, **beta-5 container**, seq=256 (**+7.8%** over beta-4's 632.0; **582.3 tok/s** at seq=1024; bit-identical `0cc59fb25112`) | [DECODE_RECIPE.md](DECODE_RECIPE.md) |

> **Read the two prefill rows carefully — they are not the same measurement.** The FP8
> row is BS=6 at N=10,000; the BF16 row is BS=2 at N=20,000, which is the regime this
> model actually targets. Three things follow:
>
> 1. **N differs, and the effect is measured, not assumed:** aggregate tok/s is nearly
>    flat in prompt length (2,803.6 @5k → 2,790.6 @10k → 2,775.6 @20k on the TP=4 packed-C32
>    config), because prefill tiles into fixed 1024-token chunks. So N=10,000 flatters a
>    number by only ~0.5% — small, but it is not zero. **An FP8 run at N=20,000 has not
>    been done.**
> 2. **BS differs, and the batch/tokens-per-call tuning was applied only to FP8.** The
>    nominal +7.0% (4,383.9 vs 4,097.9) is FP8 *plus* the output-block hoist *plus* raising
>    tokens-per-MoE-call to its measured optimum of 6,144 — not FP8 alone.
>    **BF16 CTE has never been run at BS=6/bucket 1024**,
>    so how much of the gap is FP8 rather than tuning is **open**. At the one config where
>    both were measured (BS=2/bucket 1024), BF16 was *faster*: 4,227.7 vs FP8's 3,920.3 —
>    the ~7% FP8 dequant tax, which the output-block hoist then removed.
> 3. **The memory win is unambiguous and needs no caveat:** 5.08 vs 8.94 GB/core, i.e. FP8
>    prefill runs in 57% of the HBM. That is what FP8 reliably buys here — capacity.
>
> Two runs would close this out: FP8 at N=20,000, and BF16 CTE at BS=6/bucket 1024.

Container lineage: the current numbers are on the **beta-5** DLC
(`concourse-release-0461d3b@sha256:94413ce1ffea…`, built 2026-08-05). Both paths are
clean compiler-only wins over beta-4, bit-identical in numerics. Other reference points:
beta-4 headlines were **4,002.1** prefill / **632.0** decode; the pre-beta-4 TP=8/LNC=1
prefill was **3,456.8**; the prior FP8 `block_pow2_coalesced` decode **343.6 tok/s @
BS=128** (B1 is +28.7% over it, same `0cc59fb25112` gen hash); latency-optimal decode
**48.9 tok/s @ BS=1** (true-sparse MoE + `MOE_DECODE_TP`); BF16 full-graph decode
**320.6 tok/s @ BS=32**.

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
| `DN_PERLAYER_STATE=1` | Full-graph decode: one distinct DeltaNet recurrent-state buffer per layer (mutated once, `torch.stack` at return) instead of scattering ~30 writebacks into a shared base — removes the write-after-write serialization. **+96% decode** at O1, bit-identical; requires `DN_DIRECT_STATE_OUT=1` |
| `DN_TILED_CONV=1` | Tiled conv-state layout + coalesced `mixed_qkv` DMA (decode) — ~+15–19% at BS=32/128, bit-identical |
| `DN_CHUNK_NKI=1` | Chunked DeltaNet **prefill** NKI kernel (stable long-context) |
| `CHUNK_SIZE=16\|32` | DeltaNet prefill chunk size (default 16). `32` = the faster stable-C32 path (pair with `DN_STABLE_C32=0`) |
| `DN_STABLE_C32=0` | Use the numerically-stable block-diagonal C32 inverse (the C32 prefill path). Default `1` |
| `DN_PACK_C32=1` | Pack N independent C32 streams into one P=N·32 block-diagonal tile — fewer/larger intra-chunk matmuls, fills the PE partition dim. **The fastest prefill path**; bit-identical to unpacked C32. Requires `DN_STABLE_C32=0` |
| `DN_PACK_N=2\|4` | C32 pack width (default `4` = P=128, the ceiling; `4`→+19.4%, `2`→+12.5% vs unpacked C32) |
| `DN_STREAM_WINDOW=N` | Prefill stream software-pipeline width (default `1`). Kept for experiments; no throughput win at O1/O3 (compiler declines to overlap unrolled streams) |
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
| `MOE_CTE_FP8=1` | Run the CTE MoE experts in **FP8** during prefill — **the fastest prefill path** (§3d of the recipe). Scales are derived at load from the BF16 checkpoint, so no FP8 checkpoint is needed. 5.08 vs 8.94 GB/core |
| `MOE_CTE_FP8_OUTPUT_BLOCK=1` | Reduce the 256×256 FP8 block-scale grid over the **contraction**-block axis, making the scale chunk-independent. Pair with the hoist below; **+7.9%** together. Requires `patches/nkilib-output-block-quant.patch` |
| `MOE_CTE_FP8_OB_HOIST=1` | Apply that single scale **once** at the PSUM→SBUF convergence point instead of per contraction chunk (8 Vector ops → 1 for gate_up, 2 → 1 for down). Defaults to follow `MOE_CTE_FP8_OUTPUT_BLOCK`. Setting the grid **without** the patch silently compiles the old op count |
| `MOE_CTE_MAX_TOKENS=0` | Cap on tokens per MoE call; `0` = uncapped. Now a **tuning knob**, not the old deadlock workaround (that was `nisa.local_gather` in our tiled route packer, since fixed). Optimum is **6,144 tokens/call** and it is **not monotone** — 8,192 is 2.6% *worse* |
| `MOE_CTE_FORCE_TILED=1` | Force the tiled stable-scan route packer regardless of assignment count (diagnostic; the direct `nonzero_with_count` path handles ≤16,384 assignments) |
| `MOE_PREFILL_CHUNK` | MoE prefill chunk size (default 128) |
| `MOE_NKILIB=1` | nkilib fused MoE path (BF16) |
| `MOE_FP8=1` | Older per-row FP8 grouped-matvec MoE path |
| `MOE_FUSED_W8=fp8\|int8` | High-batch full-graph decode: fused all-expert path using the official block-scaled FP8 (or symmetric INT8) experts |
| `MOE_FUSED_W8_FP8_IMPL=` | FP8 variant for `MOE_FUSED_W8`: `row` / `dual` / `block_pow2` / `block_pow2_coalesced` / **`block_ob_coalesced`** (Reduction B1: coarse per-128-output-block scale + PSUM-accumulate — the fastest decode MoE kernel; 681.3 tok/s at BS=128 with `DN_PERLAYER_STATE` on beta-5, bit-identical output) |
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
| `PREFILL_SHARDED_EMBED=1` | Vocab-shard the token embedding at **load** time (~1.02 GB → 0.13 GB per rank). Each rank owns a contiguous id range and masks the rest to zero rows; one sum-all-reduce per chunk reassembles the exact embedding, bit-identical to the replicated path. Pairs with `PREFILL_SHARDED_LM_HEAD` to fit 40-layer prefill at TP=8/LNC=1 without shrinking the query bucket |
| `PREFILL_SHARDED_LM_HEAD=1` | Vocab-shard the LM head at **load** time (~1.02 GB → 0.13 GB per rank); full-vocab logits are rebuilt with one ~2 MB all-reduce, bit-identical to the replicated path. Required to fit 40-layer prefill in the ~12 GB/rank TP=8/LNC=1 budget. Distinct from `DECODE_SHARDED_LM_HEAD`, which keeps the full weight resident and only slices at runtime; the two are mutually exclusive |

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
