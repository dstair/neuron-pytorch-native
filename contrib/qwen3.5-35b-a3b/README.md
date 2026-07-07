# Qwen3.5-35B-A3B (MoE) on Trainium2 (PyTorch Native)

A PyTorch-Native inference implementation of the sparse-MoE **Qwen3.5-35B-A3B**
model (~3B active parameters of 35B) on a single Trainium2 device
(`trn2.3xlarge`, TP=4, LNC=2). It shares the DeltaNet + GQA backbone structure of
the [dense 27B](../qwen3.6-27b) but replaces the dense MLP with a 256-expert
top-8 mixture of experts, and targets a fixed long-context (20,000-token) regime.

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

## Results (trn2.3xlarge, TP=4, LNC=2, seq=2048, synced 50-iter)

**Decode TPOT progression (BS=1):**

| Config | TPOT | tok/s |
|---|---|---|
| masked-dense MoE (start) | 66.2 ms | 15.1 |
| + true-sparse MoE (`MOE_SPARSE=1`) | 33.4 ms | 30.0 |
| + DeltaNet micro-opt | 32.8 ms | 30.5 |
| + `GQATAIL=1` | 24.4 ms | 40.9 |
| **+ `DNBATCHED_V2=1`** | **23.2 ms** | **43.2** |

2.86× total. True-sparse MoE gives ~2× (not ~8×) because the MoE expert GEMMs were
only about half the step — DeltaNet/GQA/projections/norms/all-reduces are the rest.

**Batch sweep (seq=256, masked-dense MoE + `DN_NKI+GQATAIL+DNBATCHED_V2`):**

| BS | TPOT (ms) | tok/s | scale |
|--|--|--|--|
| 1 | 54.9 | 18.2 | 1.0× |
| 4 | 70.1 | 57.0 | 3.1× |
| 8 | 84.3 | 94.9 | 5.2× |
| 16 | 120.4 | 132.9 | 7.3× |
| 32 | 188.3 | 170.0 | 9.3× |

Near-linear throughput scaling to BS=32 with no OOM (weights fixed ~19 GB/core; KV +
DeltaNet state are tiny at seq=256). Throughput-optimal is high-BS masked-dense;
latency-optimal is BS=1 true-sparse (sparse only wins at BS≤4, since it gathers
`T·K` experts and `T·K ≥ 64` once batch grows).

**20k context:** memory fits at ~19.1 GB/core, but cold-compile of the single fixed
20k-decode graph is very long (attention-over-20000 tiling) — use a persistent
`NEURON_COMPILE_CACHE_URL` mount so it is a one-time cost, or seq-bucketing.

## Reference

The validated NxDI implementation
(`aws-neuron/neuronx-distributed-inference` PR #60,
`jimburtoft:contrib/qwen3.5-35b-a3b`) is the correctness oracle: 100% token-match
vs CPU, BS=1 54.3 tok/s / 18.4 ms/tok on the same hardware. Its MoE uses an NxDI
library module and is not portable, which is why this implementation carries its
own MoE kernels and CPU oracle.
