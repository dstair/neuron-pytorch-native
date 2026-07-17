# Qwen3.6-35B-A3B (MoE) on Trainium2 (PyTorch Native)

A PyTorch-Native inference implementation of the sparse-MoE **Qwen3.6-35B-A3B**
model (~3B active parameters of 35B) on a single Trainium2 device
(`trn2.3xlarge`, TP=4, LNC=2). It shares the DeltaNet + GQA backbone structure of
the [dense 27B](../qwen3.6-27b) but replaces the dense MLP with a 256-expert
top-8 mixture of experts, and targets a fixed long-context (20,000-token) regime.

> **Naming.** The model is published as
> [`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) on Hugging
> Face, but its architecture class is `Qwen3_5MoeForConditionalGeneration`
> (`model_type: qwen3_5_moe`) — "3.5" names the architecture family, "3.6" names
> the release. The two are the same architecture, so this code runs on the HF
> checkpoint unchanged.

The whole 40-layer decode model compiles to a single NEFF via
`torch.compile(fullgraph=True, backend="neuron")`. Long-context prefill uses four
coarse 10-layer NEFFs to stay below the compiler instruction limit.

## Architecture

```
40 layers = [DeltaNet × 3, GQA × 1] × 10   (full-attention every 4th layer)
hidden 2048, vocab 248320, RMSNorm eps 1e-6, RoPE partial-64 @ theta 1e7

DeltaNet (30 layers): 16 K-heads, 32 V-heads, k/v dim 128
  in_proj_qkv [8192,2048] = q(2048)||k(2048)||v(4096); in_proj_z [4096,2048]
  in_proj_a/b [32,2048]; A_log/dt_bias [32]; depthwise conv1d [8192,1,4]

GQA (10 layers): 16 Q-heads, 2 KV-heads, head_dim 256, sigmoid output gate
  q_proj [8192,2048] (query||gate); k/v_proj [512,2048]; o_proj [2048,4096]
  per-head q_norm / k_norm [256]

MoE (all 40 layers): 256 experts, top-8, moe_inter 512, + shared expert
  experts.gate_up_proj [256,1024,2048] (gate||up fused); experts.down_proj [256,2048,512]
  router gate.weight [256,2048]
  shared_expert.{gate,up} [512,2048], down [2048,512]; shared_expert_gate [1,2048]
```

Routing (canonical Qwen3-MoE, validated):

```
logits = x @ router.T;  w = softmax(logits, float);  w, sel = topk(w, 8)
w /= w.sum(-1)                                    # norm_topk_prob = True
routed = sum_j w[:,j] * expert_sel[:,j](x)
out = routed + sigmoid(x @ shared_gate.T) * SwiGLU_shared(x)
```

Verified architecture constants and the `tp_dims(world_size)` sharding plan live in
`model_dims.py`. KV heads (2) don't divide TP=4, so each KV head is replicated
across `world_size // 2` cores.

## Layout

```
static_decode_35b.py    the static decode/prefill forward (all 40 layers) + compile
                        harness + manual TP sharding + benchmark entry point
model_dims.py           verified architecture constants and TP sharding dims
deltanet_decode.py      DeltaNet recurrent decode step
chunked_prefill.py      chunked DeltaNet prefill reference
st_reader.py            safetensors weight reader / sharder
kernels/                NKI kernels + torch.ops registrations (*_ops.py)
  deltanet_full_batched_35b*   batched DeltaNet decode (8 v-heads/core)
  deltanet_chunked_prefill_35b*  chunked-prefill DeltaNet
  gqa_tail_35b*, gqa_flash_prefill_35b*  fused GQA decode tail + local flash prefill
  gqa_cte_35b*, gqa_rope_kv_35b*  nkilib CTE attention + dynamic RoPE/KV update
  fp8_group_matvec*    FP8 grouped matvec for MoE experts
  tests/               repeatable CPU/device checks with pass/fail assertions
debug/                  reusable isolation, capture, and numerical diagnostics
deploy/profile/         device-profiling capture + neuron-explorer UI scripts
experiments/            ignored local journals, resume notes, and benchmark tools
```

Keep active runtime implementations in `kernels/`. A script belongs in
`kernels/tests/` when it is a repeatable regression with an explicit pass/fail
result; diagnostics that print intermediate evidence for manual interpretation
belong in `debug/`. The ignored `experiments/` directory is for machine-specific
records and temporary investigation tooling. Superseded code should normally
remain in Git history rather than accumulating in a `legacy/` directory.

## De-risking: the MoE CPU oracle

The #1 risk — sparse MoE routing under `torch.compile` static shapes — is retired
before any device time by `kernels/tests/test_moe_oracle_cpu.py`. On the real
layer-0 weights, the **masked-dense grouped-bmm** formulation we run on Neuron is
numerically identical (to ~3e-9) to a HF-sparse reference, to the canonical
`transformers` `Qwen3MoeSparseMoeBlock`, and to the expert-parallel sharded form.

```bash
python3 kernels/tests/test_moe_oracle_cpu.py --tokens 8
python3 kernels/tests/test_moe_sparse_eq.py
python3 kernels/tests/test_moe_decode_tp_cpu.py
```

## Running

```bash
# Full 40-layer decode benchmark, BS=1, recommended decode flags
DN_NKI=1 MOE_SPARSE=1 MOE_DECODE_TP=1 GQATAIL=1 DNBATCHED_V2=1 \
  torchrun --nproc-per-node=4 static_decode_35b.py \
    --num-layers 40 --max-seq-len 2048 --batch-size 1 --bench

# Validated compiled prefill, N=20000, BS=1
PYTHONPATH=<nki-library>/src/nkilib_src \
MOE_CTE=1 GQA_CTE_PREFILL=1 GQA_DYNAMIC_ROPE_KV=1 \
DN_CHUNK_NKI=1 CHUNK_SIZE=16 DN_NKI=1 GQATAIL=1 \
  torchrun --nproc-per-node=4 static_decode_35b.py \
    --num-layers 40 --max-seq-len 20480 --prefill-bench 20000 \
    --bucket-chunk 1024 --bucket-compile 1 --prefill-splits 4 --skip-compile

# CPU correctness (no device)
python3 static_decode_35b.py --cpu --num-layers 40
```

Set `QWEN35_MODEL_PATH` (or `--model-path`) to the weights directory.

### Reusable compiler cache

The Native compiler stores the reusable prefill artifacts beneath the complete
host directory mounted as `/tmp` in the container. That directory contains
`hlo_cache`, `neff_cache`, and NKI compiler subtrees. Do not archive or restore
only a `.neff` file, and do not point the runtime at an arbitrary one.

Set these ignored `.env` values on the compile and reuse hosts:

```bash
export QWEN35_COMPILER_CACHE_DIR=/mnt/nvme/qwen35-prefill-cache
export QWEN35_COMPILER_CACHE_S3_URI=s3://YOUR-BUCKET/neuron-compile-cache/qwen35
```

After a successful cold compile, stage an immutable cache key:

```bash
deploy/cache/push.sh bs4-c16-s20000-direct512
```

On a future host, restore it before starting the Native container:

```bash
deploy/cache/pull.sh bs4-c16-s20000-direct512
deploy/cache/inspect.sh
```

Mount `QWEN35_COMPILER_CACHE_DIR` at `/tmp` inside the container exactly as in
the producing run. A persistent-cache hit requires the same captured graph,
input shapes, compiler/image version, compiler flags, Neuron generation, and
TP/LNC topology. The scripts refuse to mix cache contents by default; use
`--replace` only to discard an existing target or deliberately refresh a key.
The optional second argument to `inspect.sh` scans a run log for cache-hit,
cache-miss, and backend-compiler markers. Native logs are sometimes
inconclusive; corroborate a cache hit by confirming no active `neuronx-cc` or
`walrus_driver` process appears during the run.

The `trn2-3xl-bs4-c16-s20000-tp4-b512-fused-direct512` artifact was validated
as a complete 3.4 GiB cache root (664 files, 66 NEFFs). A separately restored
copy ran the matching BS=4 S=20,000 graph without backend codegen, retained the
same finite fingerprint, and measured 39,788.3 ms / 2,010.6 aggregate prompt
tok/s.

### Optimization levers (environment flags)

| Flag | Effect |
|---|---|
| `DN_NKI=1` | DeltaNet NKI kernel — **required past ~20 layers** (the pure-torch recurrence trips a compiler tiling assertion) |
| `MOE_SPARSE=1` | True-sparse MoE dispatch (gathers only the top-8 experts) — ~2× at BS=1 |
| `MOE_DECODE_TP=1` | BF16 BS=1 decode only: shard each expert's intermediate width across TP ranks, avoiding dummy non-local expert reads |
| `GQATAIL=1` | Fused GQA attention-tail kernel |
| `DNBATCHED_V2=1` | DMA-coalesced batched DeltaNet decode |
| `DN_DIRECT_STATE_OUT=1` | Full-graph decode: write BF16 DeltaNet/conv state directly to disjoint output buffers |
| `GQA_STATEFUL_KV=1` | Full-graph decode: keep BF16 K/V caches as aliased module state and append only the current rows |
| `MOE_CTE=1` | Long-token nkilib context-encoding MoE kernel for prefill |
| `GQA_CTE_PREFILL=1` | Prefix-aware nkilib CTE attention; requires `GQA_DYNAMIC_ROPE_KV=1` |
| `DN_CHUNK_NKI=1`, `CHUNK_SIZE=16` | Stable long-context DeltaNet prefill kernel |
| `MOE_FP8=1` | FP8 MoE experts |
| `MOE_SHARED_ONLY`, `NOREDUCE`, `DN_PASSTHROUGH` | Diagnostics (default off) |

## Performance summary

All numbers are `trn2.3xlarge`, TP=4, LNC=2 unless stated otherwise, measured
with a `torch.neuron`-synchronized 30-50-iter timer. "PyTorch Native" = this
repo's `static_decode_35b.py` (one compiled decode NEFF or four coarse prefill
NEFFs). The "XLA" reference is the
[NxDI](https://github.com/aws-neuron/neuronx-distributed-inference) implementation
of the same model (PR #60) on the torch-xla stack.

### Decode — BS=1 optimization progression (seq=2048)

| Config | Framework | TPOT (ms) | tok/s |
|---|---|---|---|
| masked-dense MoE (start) | PyTorch Native | 66.2 | 15.1 |
| + true-sparse MoE (`MOE_SPARSE=1`) | PyTorch Native | 33.4 | 30.0 |
| + DeltaNet micro-opt | PyTorch Native | 32.8 | 30.5 |
| + `GQATAIL=1` | PyTorch Native | 24.4 | 40.9 |
| + `DNBATCHED_V2=1` | PyTorch Native | 23.2 | 43.2 |
| **+ TP within routed experts (`MOE_DECODE_TP=1`)** | PyTorch Native | **20.46** | **48.9** |
| NxDI reference (PR #60) | XLA | 18.4 | 54.3 |

3.24× total from these levers. True-sparse MoE gives ~2× (not ~8×) because the MoE
expert GEMMs are only about half the step — DeltaNet / GQA / projections / norms /
all-reduces are the rest. The NxDI (XLA) reference is faster at BS=1; it is the
validated oracle (100% token-match vs CPU) but its MoE uses a non-portable NxDI
library module.

The TP-expert layout keeps all 256 expert ids on every rank but stores one quarter
of each expert's intermediate width. Resident weights remain 19.09 GB/core, while
each rank gathers eight quarter-experts instead of eight full experts with roughly
six clamped dummy routes. The existing TP all-reduce reconstructs the full down
projection. A matched S=16, 10-layer Explorer replay estimated 758→380 MB HBM
reads, 754→377 MB software-dynamic DMA, and 270k→181k dynamic DMA packets per
rank; trace time fell from 2.95-3.14 to 2.258-2.261 ms. These traffic values are
directional because the profile has missing dynamic-DMA metadata. A production
S=2048, 40-layer replay was consistent across ranks at about 1.77 GB estimated
HBM reads.

`MOE_DECODE_TP` is deliberately restricted to BF16 and one decode token. For
higher batches, leave it disabled and use masked-dense MoE: once many experts
are active, dense grouped GEMMs amortize better than per-route gathers.

### Decode — batch sweep (seq=256, masked-dense MoE + `DN_NKI+GQATAIL+DNBATCHED_V2`)

| BS | Framework | TPOT (ms) | tok/s | scale |
|--|--|--|--|--|
| 1 | PyTorch Native | 54.9 | 18.2 | 1.0× |
| 4 | PyTorch Native | 70.1 | 57.0 | 3.1× |
| 8 | PyTorch Native | 84.3 | 94.9 | 5.2× |
| 16 | PyTorch Native | 120.4 | 132.9 | 7.3× |
| 32 | PyTorch Native | 188.3 | 170.0 | 9.3× |

Near-linear throughput scaling to BS=32 with no OOM (weights fixed ~19 GB/core; KV +
DeltaNet state are tiny at seq=256). Throughput-optimal is high-BS masked-dense;
latency-optimal is BS=1 true-sparse (sparse only wins at BS≤4, since it gathers
`T·K` experts and `T·K ≥ 64` once batch grows).

### Decode - BS=32 full graph on TP=8, LNC=1 (seq=256)

For high-batch decode, compiling embedding, all layers, state updates, the LM
head, and exact greedy token selection into one graph removes eager boundaries.
The LM head is vocab-sharded across the eight TP ranks; two all-reduces select
the exact global top-1 token, including lowest-id tie breaking, without
materializing full-vocabulary logits on every rank.

| Layers | Path | TPOT (ms) | aggregate tok/s |
|--|--|--:|--:|
| 2 | segmented, replicated LM head | 56.01 | 571.3 |
| 2 | full graph, replicated LM head | 10.26 | 3,119.4 |
| 2 | full graph, vocab-sharded LM head | **8.03** | **3,986.7** |
| 40 | full graph, vocab-sharded LM head | 108.86 | 293.9 |
| 40 | + direct recurrent-state output | **105.31** | **303.9** |
| 40 | + stateful K/V cache | **99.80** | **320.6** |

The 40-layer results are cache-hot, synchronized 30-iteration runs with masked-
dense MoE. `DN_DIRECT_STATE_OUT=1` keeps recurrent inputs read-only and has the
DeltaNet NKI kernel convert its final FP32 tiles directly into separate BF16
output buffers. This removes the whole-state input clone and the per-layer FP32
state output followed by a BF16 copy. Two real-weight decode steps matched the
control's greedy IDs and local logits on all ranks; DeltaNet and convolution
state were bit-identical. The earlier matched DGE profile showed that sharding
the LM head reduced HBM reads from 1500.1 to 610.1 MB, software DMA from 1507.5
to 603.6 MB, and device execution from 10.266 to 6.324 ms.

On a matched two-layer profile, direct output reduced device execution from
6.324 to 6.105 ms, estimated HBM reads from 610.1 to 589.2 MB, HBM writes from
46.3 to 29.5 MB, and combined dynamic DMA from 648.1 to 623.0 MB. These no-DGE
traffic estimates are directional because the profile reports missing dynamic
DMA metadata. The full graph has roughly 968,000 instructions and uses about
10.72 GB HBM per rank. A full-depth inline capture cannot allocate its trace
buffers beside the model, so traffic was measured on the two-layer graph only.

`GQA_STATEFUL_KV=1` removes K/V from the compiled step's inputs and outputs.
Each GQA call reads the prior BF16 cache, includes the current K/V row in FP32
attention math, then appends that row to aliased module buffers after attention.
This avoids cloning and returning the full `batch * sequence` cache while
preserving one graph and the established arithmetic path. It requires
`GQATAIL=1 DECODE_FULLGRAPH=1` and one local KV head per rank.

A paired 100-step four-layer run (one GQA layer) measured 12.79 to 12.68 ms.
Two real-weight steps matched every greedy ID on all eight ranks; DeltaNet,
convolution, K, and V state were bit-identical. A matched no-DGE replay reduced
device execution from 10.236 to 10.096 ms, estimated HBM reads from 1079.7 to
1046.2 MB, HBM writes from 77.5 to 44.0 MB, and combined dynamic DMA from
1131.9 to 1094.2 MB. Treat those traffic values as directional because Explorer
reports missing dynamic-DMA metadata. The production 40-layer run improved
105.31 to 99.80 ms/token, or 303.9 to 320.6 aggregate tok/s.

Use:

```bash
DN_NKI=1 DN_K_HEADS=2 DN_V_HEADS=4 \
GQATAIL=1 GQA_Q_HEADS=2 DNBATCHED_V2=1 \
DN_DIRECT_STATE_OUT=1 GQA_STATEFUL_KV=1 \
DECODE_FULLGRAPH=1 DECODE_SHARDED_LM_HEAD=1 \
NEURON_LOGICAL_NC_CONFIG=1 NEURON_CC_FLAGS="--target trn2 --lnc 1" \
  torchrun --nproc-per-node=8 static_decode_35b.py \
    --model-path <weights> --max-seq-len 256 --num-layers 40 \
    --graph-splits 1 --batch-size 32 --num-tokens 2 \
    --bench --bench-iters 30
```

### Decode — long-context batch sweep (seq=10000 and 20000, masked-dense MoE + `DN_NKI+GQATAIL`)

At long context the KV cache is no longer negligible, so the batch ceiling is set by
**device HBM** (24 GB/core), not by throughput scaling. Weights are a fixed
~19.1 GB/core; each sequence's KV cache grows with batch × seq, and past a point the
NEFF fails to load (`NRT_RESOURCE: Failed to allocate resource`).

| Seq | BS | TPOT (ms) | tok/s | scale | notes |
|--|--|--|--|--|--|
| 10000 | 1 | 62.2 | 16.1 | 1.0× | 19.1 GB/core |
| 10000 | 4 | 131.2 | 30.5 | 1.9× | |
| 10000 | 8 | 164.3 | 48.7 | **3.0×** | **peak that fits** |
| 10000 | 16 | — | — | — | OOM on device load |
| 20000 | 1 | 122.5 | 8.2 | 1.0× | 19.1 GB/core |
| 20000 | 4 | — | — | — | OOM on device load |

At seq=10000 the throughput knee is **BS=8 (48.7 tok/s)**; BS=16 exceeds HBM and
fails to load. At seq=20000 the per-sequence KV cache is 2× larger, so the ceiling
drops to **BS=1** — even BS=4 fails to load. This is a memory ceiling, not a compute
one, and it is exactly where FP8 experts help (see below): halving the expert weights
(~19→11 GB/core) frees the headroom to push the long-context batch ceiling higher
(e.g. BS=16 at 10k, or BS>1 at 20k, both of which OOM in bf16 today).

### FP8 experts — a memory/capacity lever, not a decode-latency win

FP8 (e4m3, per-output-channel row scales) on the MoE experts is wired in behind
`MOE_FP8=1`. It is **CPU-validated coherent** (fp8-vs-bf16 cosine 0.9991) and delivers
its headline benefit as a **memory saving: expert weights 16→8 GB/core, total
~19→11 GB/core**.

However, across three independent attempts it did **not** improve decode latency:

| Path | Result |
|---|---|
| Hand-rolled FP8 grouped-matvec (`MOE_FP8=1`) | BS=1 72.3 ms vs 32.8 ms bf16 — **2.2× slower** |
| `nkilib` fused MoE, bf16 (`MOE_NKILIB=1`) | BS=1 **28.2 ms (best bf16)**; FP8 path blocked |
| `nkilib` fused MoE, FP8-row | compile/dtype wall on this toolchain (legacy-e4m3 vs torch e4m3fn) — not reachable |

The reason is the BS=1 GEMV regime: FP8 replaces wide fused GEMMs with many tiny
per-expert matvecs (moving-free=1), and the per-instruction dispatch overhead dwarfs
the bandwidth saved. FP8's real value here is **capacity** — the ~8 GB/core it frees is
what would let the long-context batch ceiling above go higher (e.g. BS=16 at 10k, which
currently OOMs in bf16). Making FP8 also win latency would need the dequant fused into
one wide kernel, or the BS≫1 regime where matvecs become GEMMs. All FP8 paths are
default-off; **bf16 is the recommended decode default.**

### Prefill (prompt throughput)

| Test | Framework | Config | Latency | Prompt tok/s |
|---|---|---|---|---|
| **Batched compiled CTE-GQA + fused NKI-routed CTE-MoE + paired DeltaNet C16** | PyTorch Native | BS=2, N=20000 each, bucket=1024 | **19.141 s** | **2,089.7 aggregate** |
| Batched compiled CTE-GQA + fused NKI-routed CTE-MoE + paired DeltaNet C16 | PyTorch Native | BS=4, N=20000 each, bucket=512 | 39.788 s | 2010.6 aggregate |
| **Compiled CTE-GQA + fused NKI-routed CTE-MoE + DeltaNet C16** | PyTorch Native | BS=1, N=20000, bucket=1024 | **13.488 s** | **1482.8** |
| Compiled CTE-GQA + Torch-routed CTE-MoE + DeltaNet C16 | PyTorch Native | BS=1, N=20000, bucket=1024 | 17.374 s | 1151.1 |
| Batched compiled CTE-GQA + Torch-routed CTE-MoE + DeltaNet C16 | PyTorch Native | BS=2, N=20000 each, bucket=1024 | 41.069 s | 974.0 aggregate |
| Compiled CTE-GQA + CTE-MoE + DeltaNet C16 | PyTorch Native | N=20000, bucket=512 | 17.855 s | 1120.1 |
| Compiled CTE-GQA + CTE-MoE + DeltaNet C16 | PyTorch Native | N=20000, bucket=2048 | 20.632 s | 969.4 |
| Compiled flash-GQA + CTE-MoE + DeltaNet C16 | PyTorch Native | N=20000, bucket=512 | 20.886 s | 957.6 |
| Bucketed prefill, flash-GQA + DeltaNet-chunk kernels | PyTorch Native | N=20000 | 77.2 s (warm) | **259.2** |
| Eager prefill (pre-kernelization) | PyTorch Native | N=4000 | 146.7 s | 27.3 |
| Eager prefill (pre-kernelization) | PyTorch Native | N=2000 | 68.4 s | 29.3 |

The highest validated aggregate throughput uses BS=2, 1024-token buckets,
paired-C16 DeltaNet, four compiled 10-layer segments, runtime bucket
offsets/valid lengths, and fused NKI-routed CTE MoE. A cache-reuse run measured
19.1411 seconds / 2,089.7 aggregate prompt tok/s, with identical finite warm and
timed fingerprints. BS=4 also fits with 512-token buckets and measured 39.7883
seconds / 2,010.6 aggregate prompt tok/s after restoring its compiler cache,
matching the original 2,012.2 tok/s median.

The fastest validated single-prompt path uses 1024-token buckets, the fused
NKI-routed CTE MoE kernel (`MOE_CTE_NKI_PACK=1`), and `CHUNK_SIZE=16`.
The nkilib CTE attention kernel only visits the used KV prefix; it measured
0.77-0.81 ms per production-shape GQA call versus 11.66-11.69 ms for the local
fixed-KMAX flash kernel. At full depth this improves 957.6 to 1120.1 tok/s and
preserves the validated real-prompt continuation. The warm and timed synthetic
fingerprints were identical.

CTE also removes the descriptor ceiling at bucket 2048: all four segments
compiled and loaded. That configuration is nevertheless slower (969.4 tok/s).
The CTE attention kernel itself scales efficiently from active-query sizes 512
to 1024 to 2048 (about 0.79, 1.28, and 2.11 ms), so the regression is in the
larger surrounding compiled graph. The matched bucket-1024 Torch-route run
measured 1151.1 tok/s.

The fused route path replaces the compiled `one_hot().cumsum()` metadata packer
with an NKI stable compaction inside the existing CTE custom call. It passed 96
exact metadata cases across both production token counts, both block sizes, and
all four TP expert ranges. Distributed fused CTE output matched the
precomputed-metadata path exactly, and the four-layer BS=2 isolation test still
matched independent BS=1 executions.

Three synchronized cache-hot BS=1 runs measured 13.4834, 13.4967, and
13.4878 seconds; the 13.4878-second median is 1482.8 tok/s. A matched
8-DeltaNet/2-GQA segment improved from 188.89 to 122.26 ms, total HBM traffic
fell from 31.96 to 3.30 GB, route HBM traffic fell from 24.94 GB to 45.9 MB,
and the old `reduce-window` instruction disappeared. Standalone route packing
measured 2.420 ms for 8,192 assignments and 3.777 ms for 16,384 assignments
(1.56x scaling). The fallback remains available with
`MOE_CTE_NKI_PACK=0`; the validated BS=2 and BS=4 throughput paths use the
fused packer.

Homogeneous batching is implemented with independent DeltaNet, convolution, and
KV state per prompt while retaining one custom call per layer. A four-layer
BS=2 isolation test with distinct prompts and a partial final bucket matched
independent BS=1 runs with cosine >=0.999936 across logits and all carried
states. Full S=20000 BS=2 loaded successfully and all returned states were
finite on the Torch-route baseline, but latency increased 2.36x for 2x the
tokens: 41.069 s / 974.0 aggregate tok/s. This is 15.4% below its matched
BS=1 baseline. Fused NKI route packing plus paired-C16 DeltaNet changed the
BS=2 result to 19.141 s / 2,089.7 aggregate tok/s. The later BS=4 run remained
finite and repeatable at 39.788 s / 2,010.6 aggregate tok/s.

Isolated production-shape profiles explain part of the scaling limit. The C16
DeltaNet call increased from 11.57-11.93 ms at B=1 to 22.58-22.92 ms at B=2
because its independent recurrent streams execute sequentially. The opaque CTE
expert kernel itself stayed near 6.7-7.0 ms for 1024 versus 2048 flattened
tokens.

Full 10-layer segment traces locate the superlinear regression in the routing
wrapper around that expert kernel. For the matched 8-DeltaNet/2-GQA segment,
BS=1 to BS=2 increased from 188.9 to 471.0 ms. The
`pack_local_routes()` prefix scan at `moe_cte_adapter.py:50` increased from
61.3 to 258.3 ms and its attributed HBM traffic increased from 24.9 to
105.7 GB. Neuron lowers the `group_hot.cumsum(dim=0)` to HLO
`reduce-window` backed by TensorE MATMUL/LDWEIGHTS. DeltaNet increased from
82.4 to 165.6 ms. Those two changes explain 99% of the matched segment
regression. The alternating 7-DeltaNet/3-GQA segment showed the same result:
485.7 ms total, 263.7 ms in the route scan, and 145.6 ms in DeltaNet.
The fused NKI route path removes this scan and is used by the validated BS=2
and BS=4 throughput configurations above.

`GQA_CTE_PREFILL=1` needs a recent nkilib with `attention_cte` support for
head-dim 256 and runtime `prior_used_len`; the nkilib bundled in the current DLC
rejects head dimensions above 128. Point `PYTHONPATH` at a compatible
`nki-library/src/nkilib_src`.

The original 259.2 tok/s path used eager sequence bucketing
(`--bucket-chunk 2048 --bucket-compile 0`) with local flash GQA and chunked
DeltaNet. Two kernel bugs had to be fixed before compilation was trustworthy:
pad-token DeltaNet-state corruption and an L2-norm epsilon-semantics mismatch on
near-zero rows (see `kernels/tests/`). DeltaNet C32 is faster but becomes
non-finite at long context, so C16 remains a correctness requirement.

**20k context:** BS=1 memory fits at ~19.1 GB/core; the optimized compiled
prefill path loads through BS=4. BS=2 uses about 0.42 GB/core of persistent
K/V cache. Preserve
`NEURON_COMPILE_CACHE_URL` on NVMe: the first CTE-GQA run, including four segment
compiles and the 20k warm pass, took 861.6 seconds; cached execution is 17.9
seconds.

## Reference

The validated NxDI implementation
(`aws-neuron/neuronx-distributed-inference` PR #60,
`jimburtoft:contrib/qwen3.5-35b-a3b`, torch-xla) is the correctness oracle: 100%
token-match vs CPU, BS=1 54.3 tok/s / 18.4 ms/tok on the same hardware. Its MoE uses
an NxDI library module and is not portable, which is why this implementation carries
its own MoE kernels and CPU oracle.
