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

The whole 40-layer model — DeltaNet, GQA, and MoE — compiles to a single NEFF via
`torch.compile(fullgraph=True, backend="neuron")`.

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
  gqa_tail_35b*, gqa_flash_prefill_35b*  fused GQA decode tail + flash prefill
  fp8_group_matvec*    FP8 grouped matvec for MoE experts
  tests/               CPU MoE oracle (vs canonical Qwen3Moe), device smoke tests
deploy/profile/         device-profiling capture + neuron-explorer UI scripts
```

## De-risking: the MoE CPU oracle

The #1 risk — sparse MoE routing under `torch.compile` static shapes — is retired
before any device time by `kernels/tests/test_moe_oracle_cpu.py`. On the real
layer-0 weights, the **masked-dense grouped-bmm** formulation we run on Neuron is
numerically identical (to ~3e-9) to a HF-sparse reference, to the canonical
`transformers` `Qwen3MoeSparseMoeBlock`, and to the expert-parallel sharded form.

```bash
python3 kernels/tests/test_moe_oracle_cpu.py --tokens 8
```

## Running

```bash
# Full 40-layer decode benchmark, BS=1, recommended decode flags
DN_NKI=1 MOE_SPARSE=1 GQATAIL=1 DNBATCHED_V2=1 \
  torchrun --nproc-per-node=4 static_decode_35b.py \
    --num-layers 40 --max-seq-len 2048 --batch-size 1 --bench

# CPU correctness (no device)
python3 static_decode_35b.py --cpu --num-layers 40
```

Set `QWEN35_MODEL_PATH` (or `--model-path`) to the weights directory.

### Optimization levers (environment flags)

| Flag | Effect |
|---|---|
| `DN_NKI=1` | DeltaNet NKI kernel — **required past ~20 layers** (the pure-torch recurrence trips a compiler tiling assertion) |
| `MOE_SPARSE=1` | True-sparse MoE dispatch (gathers only the top-8 experts) — ~2× at BS=1 |
| `GQATAIL=1` | Fused GQA attention-tail kernel |
| `DNBATCHED_V2=1` | DMA-coalesced batched DeltaNet decode |
| `MOE_FP8=1` | FP8 MoE experts |
| `MOE_SHARED_ONLY`, `NOREDUCE`, `DN_PASSTHROUGH` | Diagnostics (default off) |

## Performance summary

All numbers are `trn2.3xlarge`, TP=4, LNC=2, measured with a `torch.neuron`-
synchronized 50-iter timer. "PyTorch Native" = this repo's `static_decode_35b.py`
(single compiled NEFF). The "XLA" reference is the
[NxDI](https://github.com/aws-neuron/neuronx-distributed-inference) implementation
of the same model (PR #60) on the torch-xla stack.

### Decode — BS=1 optimization progression (seq=2048)

| Config | Framework | TPOT (ms) | tok/s |
|---|---|---|---|
| masked-dense MoE (start) | PyTorch Native | 66.2 | 15.1 |
| + true-sparse MoE (`MOE_SPARSE=1`) | PyTorch Native | 33.4 | 30.0 |
| + DeltaNet micro-opt | PyTorch Native | 32.8 | 30.5 |
| + `GQATAIL=1` | PyTorch Native | 24.4 | 40.9 |
| **+ `DNBATCHED_V2=1`** | PyTorch Native | **23.2** | **43.2** |
| NxDI reference (PR #60) | XLA | 18.4 | 54.3 |

2.86× total from these levers. True-sparse MoE gives ~2× (not ~8×) because the MoE
expert GEMMs are only about half the step — DeltaNet / GQA / projections / norms /
all-reduces are the rest. The NxDI (XLA) reference is faster at BS=1; it is the
validated oracle (100% token-match vs CPU) but its MoE uses a non-portable NxDI
library module.

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

At seq=10000 the throughput knee is **BS=8 (48.7 tok/s)**; BS=16 exceeds HBM and
fails to load. At seq=20000 the per-sequence KV cache is 2× larger, lowering the
ceiling further. This is a memory ceiling, not a compute one — which is exactly where
FP8 experts help (see below): halving the expert weights frees the headroom to push
the batch ceiling higher. (20k batch sweep numbers pending; BS=1 shown.)

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
| Bucketed prefill, flash-GQA + DeltaNet-chunk kernels | PyTorch Native | N=20000 | 77.2 s (warm) | **259.2** |
| Eager prefill (pre-kernelization) | PyTorch Native | N=4000 | 146.7 s | 27.3 |
| Eager prefill (pre-kernelization) | PyTorch Native | N=2000 | 68.4 s | 29.3 |

The fast 20k path uses **eager sequence-bucketing** (`--bucket-chunk 2048
--bucket-compile 0`) with the flash-GQA (`GQA_FLASH_PREFILL=1`) and chunked-DeltaNet
(`DN_CHUNK_NKI=1`) NKI kernels plus pad-token masking, giving coherent output with no
OOM and a ~9 min one-time cold compile. This was a ~9× improvement over the original
eager path (which OOM'd at 20k and, when it fit, ran ~29 tok/s with a 2.7-hour
compile). Two kernel bugs had to be fixed to make the fast path both correct and
fast — pad-token DeltaNet-state corruption and an L2-norm epsilon-semantics mismatch
on near-zero rows (see `kernels/tests/`).

**20k context:** memory fits at ~19.1 GB/core, but cold-compile of the single fixed
20k-decode graph is very long (attention-over-20000 tiling) — use a persistent
`NEURON_COMPILE_CACHE_URL` mount so it is a one-time cost, or seq-bucketing.

## Reference

The validated NxDI implementation
(`aws-neuron/neuronx-distributed-inference` PR #60,
`jimburtoft:contrib/qwen3.5-35b-a3b`, torch-xla) is the correctness oracle: 100%
token-match vs CPU, BS=1 54.3 tok/s / 18.4 ms/tok on the same hardware. Its MoE uses
an NxDI library module and is not portable, which is why this implementation carries
its own MoE kernels and CPU oracle.
