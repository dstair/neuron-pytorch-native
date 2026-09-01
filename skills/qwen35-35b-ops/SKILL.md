---
name: qwen35-35b-ops
description: |
  Operating manual for the Qwen3.6-35B-A3B (35B-A3B MoE + DeltaNet/GQA hybrid) model in
  contrib/qwen3.6-35b-a3b. Use when running, benchmarking, or changing this model's prefill
  or decode: choosing MOE_* / DN_* / GQA_* env flags, picking a checkpoint, launching a run,
  or checking a result against the known-good fingerprints and baselines. Covers the model
  dimensions, the flag matrix and which kernel each flag selects, the correctness gates, and
  the current throughput/memory baselines.
---

# Qwen3.6-35B-A3B operating manual

Everything here is specific to `contrib/qwen3.6-35b-a3b`. Generic technique lives in
`neuron-model-profiling`, `trainium-run-discipline`, `nki-perf-patterns`,
`fp8-quantization-perf`, `moe-kernel-perf`.

## Model shape

| dim | value |
|---|---|
| layers | 40 (30 DeltaNet + 10 GQA, `D.layer_type(i)`) |
| hidden | 2048 |
| experts | 256 global, top-K **8**, `norm_topk_prob` |
| expert intermediate | 512 |
| per-expert params | `2·H·I + I·H` = 3.15 M → ~806 M per MoE layer, ~32 B total |
| TP=8 sharding | 32 local experts/rank; DeltaNet K=2/V=4 heads, GQA Q=2 heads |

At TP=8 a token routes to ~**1** on-rank expert on average (`K·E_local/E_global`) — the number
that kills any dense-over-local-experts prefill proposal (see `moe-kernel-perf`).

## Checkpoints

| path | contents | needed by |
|---|---|---|
| `/mnt/nvme/Qwen3.5-35B-A3B` | BF16 | always (`--model-path`) — non-expert weights |
| `/mnt/nvme/Qwen3.5-35B-A3B-FP8` | official FP8 experts + `weight_scale_inv` | `MOE_FUSED_W8` / `MOE_OFFICIAL_FP8_REFERENCE` (`--expert-model-path`) |

`MOE_CTE_FP8` does **not** use the FP8 checkpoint — it quantizes the BF16 experts to 256-block
int8 at load. `MOE_FUSED_W8` reads the official FP8 checkpoint directly.

## MoE flag matrix — which kernel you actually get

`_moe()` dispatches in this order, so an earlier flag wins:

| flags | path | shape it serves |
|---|---|---|
| `MOE_FUSED_W8=fp8\|int8` | `_moe_fused_w8` → token-tiled all-expert kernel | **decode**, batch ∈ {32,64,128,256} |
| `MOE_CTE=1 MOE_CTE_NKI_PACK=1` | routed blockwise CTE | **prefill** |
| … + `MOE_CTE_FP8=1` | FP8×FP8 double_row CTE | prefill, FP8 experts |
| `MOE_CTE=1`, decode | falls through to `moe_tkg` | decode with CTE-packed weights |
| `MOE_NKILIB=1` | `moe_tkg`, chunked by `MOE_PREFILL_CHUNK` (default 128) | T ≤ 128 per call |
| none | torch masked-dense `moe_forward`, chunked (default 512) | reference/oracle only |

`MOE_FUSED_W8` is **mutually exclusive** with all `MOE_CTE*` / `MOE_NKILIB` / `MOE_FP8`.

`MOE_FUSED_W8_FP8_IMPL` selects the kernel entry **and** the weight conversion:

| value | kernel entry | conversion |
|---|---|---|
| `row` (default) | `moe_tkg_row_fp8` | per-output-channel scales |
| `dual` | `nki_moe_fused_w8_fp8_dual` | `split_official_fp8`, two planes |
| `block_pow2` | `nki_moe_fused_w8_fp8_native` | `requantize_official_fp8_pow2` (exact) |
| `block_pow2_coalesced` | `..._block_coalesced` | pow2 + coalesced scale table |
| `block_ob_coalesced` | `..._block_coalesced_ob` | `requantize_official_fp8_output_block` — **the shipped decode path** |

Other MoE knobs: `MOE_CTE_BLOCK` (256 or 512, default 512), `MOE_CTE_MAX_TOKENS` (cap tokens per
CTE call, default 0 = off), `MOE_PREFILL_CHUNK`, and the diagnostics `MOE_DUMP_SEL`,
`MOE_CTE_RETURN_ROUTED`, `MOE_CTE_SYNC_BEFORE_SHARED`.

## Attention / DeltaNet flags

Prefill: `GQA_CTE_PREFILL=1 GQA_DYNAMIC_ROPE_KV=1 DN_CHUNK_NKI=1 CHUNK_SIZE=32 DN_NKI=1
GQATAIL=1 DN_PACK_C32=1 DN_PACK_N=4 PREFILL_SHARDED_LM_HEAD=1 PREFILL_SHARDED_EMBED=1`
(`DN_PACK_N=4` is the hardware max; `DN_STABLE_C32=0` — the stable variant is numerically broken.)

Decode: `DECODE_FULLGRAPH=1 DECODE_SHARDED_LM_HEAD=1 DNBATCHED_V2=1 DN_DIRECT_STATE_OUT=1
DN_PERLAYER_STATE=1 DN_TILED_CONV=1 GQA_STATEFUL_KV=1`.
`DN_PERLAYER_STATE=1` was **+96%**. `DN_TILED_CONV` **flips sign with batch** (+16.8% at BS=8,
−37% at BS=16) — re-measure at your batch.

`GQA_CTE_PREFILL` hard-raises without `GQA_DYNAMIC_ROPE_KV=1`. Both stateful-KV paths require
**one distinct buffer per layer** (a shared `[NUM_GQA,...]` tensor silently keeps only the first
layer's writes).

## Launching

Use the clean-runner shape in `trainium-run-discipline/scripts/run_template.sh`. Compile flags:
`NEURON_CC_FLAGS="--target trn2 --lnc 1 --optlevel 1 --hbm-scratchpad-page-size 256"` plus
`NEURON_LOGICAL_NC_CONFIG=1 QWEN35_LNC=1 NEURON_SCRATCHPAD_PAGE_SIZE=256`, `torchrun
--nproc-per-node=8`, and `PYTHONPATH=/nki-library/src/nkilib_src`.

Gotchas:

- `--prefill-splits 1` hits `UnboundLocalError` (a conditional `import torch._dynamo` makes
  `torch` a local for the whole function). **Use splits ≥ 2.**
- The nkilib checkout needs our two patches (`nkilib-blockquant-hybrid`,
  `nkilib-lnc1-moe-cte`); upstream still asserts `NUM_SHARDS == 2`. Runners check for the
  string "only work on TRN2" and abort if unpatched.
- The DLC ships no `transformers`; if you need it, `pip --target` + `PYTHONPATH`.
- Cap in-flight tokens (`batch × bucket-chunk`) at ~4,096 at 40 layers.

## Correctness gates — use these, not a generation hash

| check | expected |
|---|---|
| real prompt, 4 layers, prefill | top5 = `[1861, 3032, 711, 520, 18876]` |
| `--prefill-bench` synthetic prompt, seq 10k | top5 = `[220, 13, 197, 198, 62]` |
| bit-identical reference (BS=2/chunk1024/4L/FP8) | `sum = -2.77890906e+05`, `norm = 1.29983826e+03` |
| every run | all `finite[...] = True`, and buffer occupancy (`nz=`) full where expected |

A matching `gen hash` agreed across a real defect **and** its fix — never gate on it. `sum`/`norm`
plus occupancy caught what the hash missed.

## Baselines (40 layers, seq 10k, clean flags, TP=8/LNC=1, beta-5 container)

| config | tok/s | GB/core |
|---|---|---|
| BF16 CTE, BS=2 / chunk1024 | 4,227.7 | 8.94 |
| FP8 CTE, BS=2 / chunk1024 | 3,920.3 | 5.08 |
| FP8 CTE, BS=4 / chunk1024 (`MOE_CTE_MAX_TOKENS=2048`) | 3,846.2 | 5.08 |
| FP8 CTE, BS=16 / chunk256 (`MOE_CTE_MAX_TOKENS=2048`) | 3,675.1 | 5.08 |
| published prefill headline (seq 20k) | 4,097.9 | — |
| decode headline, BS=128 | 681.3 tok/s | — |

Throughput is **monotone in bucket size and flat in batch**; FP8 costs ~7% and buys ~44% HBM.
Prefill's largest components are DeltaNet and the MoE; DeltaNet is at its floor.

## Known open items

- A CTE MoE call above ~2048 tokens deadlocks; the leading hypothesis is the packer's
  size-conditional `scatter_barrier`, not a token limit — see
  `moe-kernel-perf/references/cte-blocking.md` (untested).
- The FP8 block-dequant Vector tax (~26% of Vector) has a precedented fix — see
  `fp8-quantization-perf`.
