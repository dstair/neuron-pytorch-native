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
| `KERNEL_TRN1=1` | Native trn1 (NeuronCore-v2) portability gate — see below |

### Native trn1 execution (`KERNEL_TRN1`)

The model is a Trainium2 (NeuronCore-v3) program by design, but it **also compiles
and runs natively on Trainium1 (`trn1.32xlarge`, NeuronCore-v2)** for customers where
trn1 is the more widely available part. This runs the **full 64-layer dense model,
real weights, compiled (non-lazy) decode at TP=8**: **37.0 tok/s (B=1), 27.0 ms
steady-state TPOT** (`--target trn1 --optlevel 1`, `KERNEL_TRN1=1`, vocab-sharded).
The chunked-DeltaNet prefill (`CHUNKEDPREFILL=1`) also compiles and executes
(cold compile+exec ~11 s for S=64).

> **trn1 uses TP=8, not TP=4.** A trn1 NeuronLink collective is valid only over
> {1, 4, 8, 16} Neuron *devices*, and each trn1 device has 2 cores — so valid TP
> core-counts are 2/8/16/32. **TP=4 (= 2 devices) is an invalid topology** (NCCL
> `Unsupported topology … got 2` → `BuildGlobalComm failed`). The per-core dims in
> `static_decode.py` derive from `WORLD_SIZE` (byte-identical at TP=4), and the 4 GQA
> KV heads are **replicated** across `WORLD_SIZE//4` ranks (`shard_gqa_kv`: rank r
> holds KV head `r // rep`, the one its contiguous query-head slice attends to).

Two trn1 codegen limitations had to be worked around in the DeltaNet kernels:

1. **`KERNEL_TRN1=1` (gated, default-off).** trn1 lacks the trn2 on-chip
   shared-memory ISA, so any HBM *scratch* buffer allocated in `nl.shared_hbm` trips
   the `neuronx-cc` verifier (`Shared memory is only supported on trn2`). With the
   gate on, the intermediate scratch buffers (`silu_z_hbm` / `exp_g_hbm` / `beta_hbm`
   / `z_hbm`) move to per-core private HBM (`nl.hbm`). Kernel *returned outputs* stay
   `nl.shared_hbm` — the NKI frontend requires that, and trn1 accepts shared_hbm as an
   output DMA target. Default-off keeps the trn2 path unchanged.
2. **Copy-activation immediate bias (unconditional).** trn1 requires an *immediate*
   (scalar) bias for `nisa.activation(op=nl.copy, ...)` (`[NCC_IBVF043]`: "Activation
   function with type Copy only supports immediate bias"). The DeltaNet kernels passed
   a zero-bias SBUF tensor (`zb_p` / `zb1` / `zb_k1` / `zb_1`, each `memset(0.0)`), so
   the fix is `bias=<zero tensor>` → `bias=0.0`. Applied unconditionally because it is
   numerically a no-op on trn2 (adding a zero immediate vs a zero tensor), mirroring
   the 35B `trn1-prefill` branch. **Not re-verified byte-identical on trn2** — gate it
   behind `KERNEL_TRN1` before a trn2 landing if strict NEFF byte-identity is required.

Runtime path: trn1 multi-core uses the host DLAMI `torch_xla` 2.9 / PJRT stack
(`PJRT_DEVICE=NEURON`, `NEURONCORE_NUM_DEVICES=8`, `backend="xla"`, `xm.xla_device()`,
`torch.compile(backend="openxla")`), **not** the container's `backend="neuron"` c10d
path (that SIGSEGVs on the trn1 kernel driver). A `nki_op` compat shim
(`kernels/nki_op_compat.py`) lets the unmodified `*_ops.py` register under 2.9. Three
torch-neuronx-2.9-vs-2.11 version-gap fixes make the **compiled** (not lazy) path work:
(1) `silu(x) = x·sigmoid(x)` on the XLA path (native `F.silu` crashes Dynamo
fake-tensor tracing under `backend="openxla"`); (2) the shim's `functional_decomp`
routes AOTAutograd fake-tracing to the registered fake via
`torch._guards.detect_fake_mode` (a type-based `isinstance(FakeTensor)` check misses
it — args arrive as `FunctionalTensor`s); (3) `xm.wait_device_ops()` in place of the
absent `torch.neuron.synchronize()`. The full 64-layer dense model (~54 GB bf16) fits
comfortably at TP=8 (~7 GB/core with vocab sharding).

## Performance summary

All numbers are `trn2.3xlarge`, TP=4, LNC=2, bf16, measured with a
`torch.neuron`-synchronized timer (async-enqueue artifacts excluded). "PyTorch
Native" = this repo's `static_decode.py` (single compiled NEFF). "XLA" reference
points are the [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference)
implementation of the same model on the torch-xla stack, for comparison.

### Decode (TPOT / throughput)

| Test | Framework | BS | TPOT (ms, synced) | Throughput (tok/s) |
|---|---|---|---|---|
| Decode-only, `static_decode.py` | PyTorch Native | 1 | 35.9 | 28.2 |
| Decode-only, `DNBATCHED_V2` + `GQATAIL` | PyTorch Native | 8 | **127.6** | **62.7** |
| Decode-only, `DNBATCHED_V2` + `GQATAIL` | PyTorch Native | 16 | 253.5 | 63.1 |
| Decode, NxDI (single-stream, published) | XLA | 1 | 54.2 | 18.5 |
| Decode, NxDI offline `llm.generate` | XLA | 8 | 361 | 22.2 |
| Decode, NxDI offline `llm.generate` | XLA | 16 | 525 | 30.5 |
| Decode-only, `static_decode.py` **on trn1** † | PyTorch Native | 1 | 27.0 | **37.0** |

† trn1 row is `trn1.32xlarge`, **TP=8**, `--target trn1 --optlevel 1`, `KERNEL_TRN1=1`,
PJRT/2.9 — NOT the trn2/TP=4/LNC=2 config of the other rows (see *Native trn1
execution*). It is a portability + real-throughput result on the more widely available
part, not a strict head-to-head with the trn2 rows.

- BS=1 synced steady-state TPOT is **35.9 ms**; the realistic host-synchronized
  greedy loop (a `.item()` D2H per token) is ~43.6 ms.
- BS=8 improved 55.6 → 58.8 → 62.7 tok/s across the batched-DeltaNet
  (`DNBATCHED_V2`) and GQA-tail (`GQATAIL`) kernels.
- Throughput plateaus ~63 tok/s: 8→16 adds only ~0.4 tok/s (per-step work grows
  nearly linearly with batch; see the barrier-bound analysis below).
- The XLA/NxDI rows are not a strict head-to-head (different serving overhead and
  the offline path carries per-step Python scheduler + sampling cost) — treat them
  as an independent reference, not an A/B against the Native numbers.

### Prefill (TTFT / prompt throughput)

| Test | Framework | Config | Latency | Prompt tok/s |
|---|---|---|---|---|
| Prefill, compiled chunked-DeltaNet kernel | PyTorch Native | S=128, BS=1 | **83.7 ms** (warm) | **1530** |
| Prefill, eager fallback (unoptimized) | PyTorch Native | S=128, BS=1 | 6847 ms | 19 |

Wiring the validated chunked-DeltaNet NKI kernel makes the prefill graph
compilable (one custom call/layer instead of an un-compilable data-dependent torch
chunk-loop), giving a **~82×** speedup over the eager fallback. Cold compile of the
prefill graph is ~391 s (one-time; NEFF cached after). Enabling chunked prefill
does not regress decode (BS=1 TPOT stays 36.7 ms).

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
three modes — baseline bf16, chunked-prefill, and EAGLE3 speculative decode. They
load the repository's ignored `.env` automatically; configure the image, model,
plugin, and compile-cache locations using the `QWEN27_*` variables documented in
`.env.example`. The serving-side model plugin lives in a separate vLLM-Neuron
fork and is not included here.

> The performance numbers above are all from the PyTorch-Native `static_decode.py`
> harness. vLLM-Neuron serving throughput numbers are intentionally omitted.
