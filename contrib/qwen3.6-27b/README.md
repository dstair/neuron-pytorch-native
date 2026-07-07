# Qwen3.6-27B on Trainium2 (PyTorch Native)

A PyTorch-Native inference implementation of the dense **Qwen3.6-27B** model on a
single Trainium2 device (`trn2.3xlarge`, TP=4, LNC=2, bf16). The whole decode step
compiles to a single NEFF via `torch.compile(fullgraph=True, backend="neuron")`,
with custom NKI kernels for the DeltaNet recurrence and the GQA attention tail.

## Architecture

```
64 layers = [DeltaNet × 3, GQA × 1] × 16   (full-attention every 4th layer)
hidden 5120, intermediate 17408, vocab 248320
RMSNorm eps 1e-6 with (1 + weight) scaling; RoPE partial-64 @ theta 1e7

DeltaNet (48 layers): 16 key heads, 48 value heads, k/v dim 128
  in_proj_qkv [5120,10240] = q(2048)||k(2048)||v(6144); depthwise conv1d k=4
  in_proj_z [5120,6144]; in_proj_a/b [5120,48]; A_log/dt_bias [48]; out_proj [6144,5120]
  recurrence: state *= exp(g); state += outer(k, delta); out = state.T @ q

GQA (16 layers): 24 Q-heads (+ output gate), 4 KV-heads, head_dim 256
  q_proj [5120,12288] (query||gate); k/v_proj [5120,1024]; o_proj [5120,6144]
  per-head q_norm / k_norm (RMSNorm)

MLP (all 64 layers): SwiGLU, gate/up [5120,17408], down [17408,5120]
```

TP=4 sharding is manual and per-core (colwise for q/k/v/gate/up, rowwise for the
output/down projections) with an all-reduce after each `out_proj` and `down_proj`.
See the sharding helpers in `static_decode.py`.

## Layout

```
static_decode.py        the static decode/prefill forward (all 64 layers), the compile
                        harness, manual TP sharding, and the benchmark entry point
chunked_prefill.py      Neuron-compatible chunked DeltaNet prefill reference (matches HF)
kernels/                NKI kernels + their torch.ops registrations (*_ops.py)
  deltanet_full*        fused DeltaNet inner block (conv + gates + recurrence + gated norm)
  deltanet_chunked_v2*  chunked-prefill DeltaNet kernel (makes prefill compilable)
  deltanet_full_batched*  batched-decode DeltaNet (BS>1 throughput; v2 = DMA-coalesced)
  gqa_tail*             fused GQA attention tail (norm + RoPE + scores + softmax + gate)
  fp8_matmul*           FP8 W8A16 matmul kernel
  rms_norm_nki*         RMSNorm kernel
  tests/                CPU / simulator correctness oracles + device smoke tests
deploy/                 vLLM-Neuron serving scripts (baseline, chunked-prefill, EAGLE3)
```

## Running

The decode/prefill harness runs inside the PyTorch-Native container on a Trainium2
box. It is launched with `torchrun` across the 4 logical cores:

```bash
# Decode benchmark, full 64 layers, BS=1
torchrun --nproc-per-node=4 static_decode.py --num-layers 64 --max-seq-len 2048 --batch-size 1

# BS=8 throughput with the banked-best decode flags
DNBATCHED_V2=1 GQATAIL=1 \
  torchrun --nproc-per-node=4 static_decode.py --num-layers 64 --batch-size 8

# Compiled chunked prefill
CHUNKEDPREFILL=1 \
  torchrun --nproc-per-node=4 static_decode.py --num-layers 64 --prompt-len 128 --compile-prefill
```

Useful flags: `--tiny` (small config for quick iteration), `--num-layers N`,
`--skip-prefill`, `--fp8-weights`, `--batch-size N`, `PROFILE_STEPS=N` (emit a
device profile over N decode steps).

### Optimization levers (environment flags)

All are **default-off**; the baseline path is byte-identical without them.

| Flag | Effect |
|---|---|
| `CHUNKEDPREFILL=1` | Compiled chunked-DeltaNet prefill (~82× faster prefill compile) |
| `DNBATCHED_V2=1` | DMA-coalesced batched-DeltaNet decode kernel (BS>1) |
| `GQATAIL=1` | Fused GQA attention-tail mega-kernel |
| `NORMFUSE=1` | NKI RMSNorm |
| `LEANKV=1`, `NOREDUCE=1`, `DNF32STATE=1` | Reference levers, ruled out (kept for A/B) |

## Results (trn2.3xlarge, TP=4, LNC=2, bf16)

**Decode throughput** — batched decode is the product surface:

| Config | BS=1 | BS=8 | BS=16 |
|---|---|---|---|
| TPOT (ms, synced) | 35.5 | 127.6 | 253.5 |
| Throughput (tok/s) | 28.2 | **62.7** | ~63 |

BS=8 improved 55.6 → 58.8 → 62.7 tok/s across the batched-DeltaNet (`DNBATCHED_V2`)
and GQA-tail (`GQATAIL`) kernels.

**Prefill** — compiled chunked prefill = **83.7 ms/forward @ S=128 BS=1
(1530 prompt tok/s)** vs the eager-path artifact 6847 ms / 19 tok/s (~82×).

### The core finding: BS=8 decode is barrier-bound, not compute-bound

A critical-path decomposition of the BS=8 decode NEFF found GEMM-exclusive time at
**0.7%** of the step; the step is dominated by engines waiting on each other at
inter-op sync barriers (~58% GLUE+SYNC not behind GEMM). So the lever that moves
TPOT is the **number of serialized sync-points per token**, not the cost of any
single op — which is why the GQA-tail kernel (it removes ~12 barriers/GQA-layer)
wins while optimizations that only cut overlapped per-op cost were flat.

Levers that were tried and **ruled out** at BS=1 latency: MLP/GQA projection
fusion (hand-rolled tiled+transpose GEMM can't beat `F.linear`), FP8 W8A16 /
W8A8 weights (per-Linear op boundaries + B=1 GEMV mapping regress), and DMA
coalescing v3/v4 (targeted DMAs are off the critical path). FP8's value is memory
headroom for larger batch, not per-step TPOT.

## Serving (vLLM-Neuron)

`deploy/` contains scripts for serving the model through a vLLM-Neuron fork in
three modes — baseline bf16, chunked-prefill, and EAGLE3 speculative decode
(~3–3.8× decode speedup at BS=1). These target a container image and model weights
referenced via `${ECR_REGISTRY}` / model mount paths; `source ../../.env` first
(see the repo root README). The serving-side model plugin lives in a separate
vLLM-Neuron fork and is not included here — these scripts document the runtime
configuration (env flags, TP, mounts, EAGLE3 config).
