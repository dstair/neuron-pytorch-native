# Qwen3.6-27B on Trainium1

This directory contains the Trainium1 implementation and validation harness for
the dense Qwen3.6-27B model. It is validated on `trn1.32xlarge` with TP=8,
bf16 weights, the torch-xla 2.9 PJRT runtime, and compiler flags
`--target trn1 --optlevel 1`.

The original PyTorch Native Trainium2 implementation is maintained separately
in [`../qwen3.6-27b-trn2`](../qwen3.6-27b-trn2).

## Validated Path

The production decode path uses:

- Manual TP=8 model sharding with replicated GQA KV heads.
- Segmented real prefill because monolithic TP8 prefill is not coherent.
- One compiled decode entry point covering embedding, all 64 layers, final
  norm, LM head, greedy token selection, and position advancement.
- The validated chunked DeltaNet recurrence instead of the incoherent
  `deltanet_full` TP8 decode kernel.
- Contiguous `[K,N]` bf16 MLP and LM-head weights.
- A truly batched DeltaNet recurrence for BS=8: batch and head are flattened
  into 48 independent streams handled by one NKI custom call per layer.

The acceptance environment is Trainium1-specific. Do not use the Trainium2
`backend="neuron"` runtime here; on Trn1 this model uses
`torch.compile(backend="openxla")`, `backend="xla"` collectives, and
`xm.xla_device()`.

## Layout

```text
static_decode.py        model, TP sharding, prefill/decode, and benchmark entry point
dev_stepcompile.py      enforceable TP8 prefill-to-decode acceptance gate
dev_layerdiff.py        eager-versus-compiled layer diagnostic
chunked_prefill.py      host reference for chunked DeltaNet prefill
kernels/                NKI kernels and torch.ops registrations
  deltanet_chunked_v2*  coherent prefill and decode recurrence
  deltanet_full*        retained kernel variants and correctness references
  gqa_tail*             fused GQA tail
  fp8_matmul*           experimental FP8 path
  nki_op_compat.py      torch-neuronx 2.9 custom-op compatibility layer
  tests/                CPU, simulator, and device tests
```

Trainium2 deployment scripts and historical transfer notes intentionally remain
only in the Trn2 directory.

## Architecture

```text
64 layers = [DeltaNet x 3, GQA x 1] x 16
hidden 5120, intermediate 17408, vocab 248320

DeltaNet: 48 layers, 16 key heads, 48 value heads, K/V dim 128
GQA:      16 layers, 24 query heads, 4 KV heads, head dim 256
MLP:      SwiGLU in every layer
```

At TP=8 each rank owns six DeltaNet value heads and two key heads. The four GQA
KV heads are replicated across pairs of ranks so that each contiguous query-head
shard sees the correct KV head.

## Environment

Run from a Trainium1 DLAMI with the Neuron torch-xla 2.9 environment:

```bash
cd contrib/qwen3.6-27b-trn1
source ../../.env

export PJRT_DEVICE=NEURON
export NEURONCORE_NUM_DEVICES=8
export NEURON_CC_FLAGS="--target trn1 --optlevel 1"
export KERNEL_TRN1=1
```

Set `QWEN27_MODEL_PATH`, `TORCH_NEURONX_NEFF_CACHE_DIR`, and
`PREFILL_SNAPSHOT_PATH` in the repository's gitignored `.env`; the corresponding
empty entries in `.env.example` document the required names.

Use a fresh cache when testing compile convergence. `dev_stepcompile.py` appends
the local rank to the cache path so all eight ranks compile independently.

## Acceptance Runs

### BS=1 snapshot resume

The fast default resumes a rank-local BS=1 prefill snapshot from
`PREFILL_SNAPSHOT_PATH`, which may contain `{rank}`.

```bash
DECODE_VIA_CHUNKED=1 \
DECODE_COMPILE_STEP=1 \
DECODE_STEP_FULLGRAPH=1 \
BF16_PREPACKED_LINEAR=1 \
torchrun --nproc-per-node=8 dev_stepcompile.py
```

### BS=1 real prefill

This is the end-to-end correctness gate. It initializes empty state, runs the
exact 128-token George prompt through segmented prefill, selects the first
token, and continues through compiled decode.

```bash
RESUME_PREFILL_SNAPSHOT=0 \
PREFILL_SEGMENTED=1 \
DECODE_VIA_CHUNKED=1 \
DECODE_COMPILE_STEP=1 \
DECODE_STEP_FULLGRAPH=1 \
BF16_PREPACKED_LINEAR=1 \
torchrun --nproc-per-node=8 dev_stepcompile.py
```

### BS=8 compiled decode

BS=8 repeats the BS=1 snapshot across batch rows and validates every row on
every TP rank. Bounded graph spans are required for this path.

```bash
BATCH_SIZE=8 \
RESUME_PREFILL_SNAPSHOT=1 \
DECODE_VIA_CHUNKED=1 \
DECODE_COMPILE_STEP=1 \
DECODE_STEP_FULLGRAPH=0 \
DECODE_STEP_BREAK_EVERY=8 \
BF16_PREPACKED_LINEAR=1 \
torchrun --nproc-per-node=8 dev_stepcompile.py
```

The gate exits nonzero unless all ranks pass:

- Exact 17-token golden continuation.
- Agreement between the returned token and `argmax(logits)`.
- Agreement across every batch row.
- Stable rank-local cache counts after step 0.
- Zero new NEFFs on every later decode step.

`BF16_PREPACKED_LINEAR=1` and `--fp8-weights` are intentionally incompatible.

## Results

All numbers are synchronized worst-rank warm measurements on
`trn1.32xlarge`, TP=8.

| Path | BS | TPOT | Throughput |
|---|---:|---:|---:|
| Fullgraph decode | 1 | 42.7 ms | 23.4 tok/s |
| Fullgraph + bf16 prepacked MLP/head | 1 | 39.9 ms | 25.1 tok/s |
| Real segmented prefill + prepacked decode | 1 | 40.3 ms | 24.8 tok/s |
| Batched recurrence + prepacked decode | 8 | 123.6 ms | 64.7 tok/s |

The BS=8 path replaced 48 row-level recurrence calls per DeltaNet layer span
with six batched calls. In the profiled eight-layer span this reduced latency
from 13.284 to 12.921 ms and collective active time from 1.343 to 0.997 ms.

## Current Bottleneck

DMA remains the longest-active engine in the optimized BS=8 span:

- Span latency: 12.921 ms.
- DMA active time: 7.850 ms.
- Total DMA traffic: 1.587 GB.
- Dense model-matrix inputs: 766 MB.
- Recurrent/cache state input and output: about 370 MB.
- Physical compiler spill traffic: about 444 MB.

`dynamic-update-slice` state materialization accounts for roughly 75% of
compiler-attributed spill traffic. Small transfers are not the dominant byte
source: packets at or below 1 KiB are about 8% of packet count but only 1.1% of
bytes. The next throughput target is eliminating full-state update
materialization and duplicate state copies before pursuing packet-level tuning.

## Trn1 Compatibility

`KERNEL_TRN1=1` moves NKI scratch allocations from `nl.shared_hbm` to private
HBM where required because NeuronCore-v2 does not support the Trn2 shared-memory
scratch ISA. Returned NKI outputs remain in `nl.shared_hbm`, as required by the
frontend and accepted as output DMA targets on Trn1.

`kernels/nki_op_compat.py` provides the torch-neuronx 2.9 custom-op registration
needed to preserve opaque NKI calls through AOT functionalization. The XLA path
also lowers SiLU as `x * sigmoid(x)` and synchronizes with
`xm.wait_device_ops()`.

The coherent fallback is `DECODE_SEGMENTED=1`, but it is approximately
1465 ms/token and is retained for diagnosis rather than throughput.
