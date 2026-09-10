# Qwen3.5-4B on Trainium2

This folder contains a PyTorch Native static-decode implementation for
`Qwen/Qwen3.5-4B`, the 4B dense model in the Qwen 3.6 release family.

The implementation supports TP=4/LNC=2, eight independent TP=1/LNC=1
replicas, and four independent TP=1/LNC=2 replicas on one `trn2.3xlarge`.
It transfers the qualified 35B decode stack: one full graph, DMA-coalesced
DeltaNet, tiled convolution state, distinct per-layer recurrent buffers,
fused stateful GQA, vocabulary sharding for TP=4, and native FP8 W8A16
projection kernels. The TP=1 path additionally fuses gate/up/SwiGLU/down and
fuses vocabulary projection/scaling/argmax.

Architecture:

```text
32 layers = [DeltaNet ×3, GQA ×1] ×8
hidden 2560, intermediate 9216, vocab 248320
DeltaNet: 16 K heads, 32 V heads, head dim 128
GQA: 16 Q heads, 4 KV heads, head dim 256, partial-64 RoPE
```

See [PORT_PLAN.md](PORT_PLAN.md) for the transfer and validation plan,
[RESULTS.md](RESULTS.md) for the device measurements, and
[PROFILE_RESULTS.md](PROFILE_RESULTS.md) for the LNC=1 profile.

## Run

Inside the Native Neuron container:

```bash
NEURON_LOGICAL_NC_CONFIG=2 \
NEURON_RT_NUM_CORES=4 \
NEURON_CC_FLAGS="--target trn2 --lnc 2 --optlevel 1" \
torchrun --nproc-per-node=4 static_decode.py \
  --model-path /models/Qwen3.5-4B \
  --dtype bf16 \
  --batch-size 128 \
  --max-seq-len 256 \
  --warmup 3 \
  --bench-iters 20
```

The benchmark uses deterministic token IDs and zero-initialized state, times
only synchronized steady-state decode steps, and checks that all 24 DeltaNet
layers have nonzero recurrent state.

Use `--dtype fp8-mlp-head` for the selective TP=4 variant,
`--dtype fp8-lm-head` to quantize only the vocabulary projection, or
`--dtype fp8` to quantize every dense projection.

For maximum aggregate throughput, first create an LNC=1 TP=1 cache and then
launch eight replicas:

```bash
export QWEN4B_NATIVE_IMAGE=<native-neuron-image>
export QWEN4B_MODEL_DIR=<Qwen3.5-4B-checkpoint>
export QWEN4B_RUNTIME_TEMPLATE=<persistent-cache-directory>
export QWEN4B_CACHE_DIR=<compiler-cache-directory>
deploy/compile_dp1_trn2.sh

export QWEN4B_RUN_ROOT=<benchmark-output-directory>
deploy/run_dp8_trn2.sh
```

The DP8 launchers enable both fused FP8 kernels by default. Set
`QWEN4B_FUSED_FP8_MLP=0` or `QWEN4B_FUSED_FP8_HEAD=0` to run an ablation.

To run four complete models with LNC=2, use the matching cache and launcher:

```bash
export QWEN4B_RUNTIME_TEMPLATE=<lnc2-cache-directory>
export QWEN4B_BATCH_SIZE=64
deploy/compile_dp4_lnc2_trn2.sh

export QWEN4B_RUN_ROOT=<benchmark-output-directory>
deploy/run_dp4_lnc2_trn2.sh
```

## Results

The current maximum-throughput result is eight TP=1/LNC=1 replicas, BS=32
each, with fused FP8 MLP and vocabulary-head kernels and BF16 attention
projections: **935.1 aggregate tok/s** on a 50-iteration confirmation run.
This is 25.0% above the previous generic-FP8 result.

Four TP=1/LNC=2 copies at BS=64 each measured **513.0 aggregate tok/s** using
the earlier generic FP8 path.

The original TP=4/LNC=2 results at BS=128/seq=256 are:

- BF16: **679.6 aggregate tok/s**
- FP8 vocabulary head with BF16 backbone: **658.5 tok/s**
- FP8 on every projection: **573.3 tok/s**

See [RESULTS.md](RESULTS.md) for the topology comparison, memory numbers,
numerical gates, and FP8 follow-up. See
[PROFILE_RESULTS.md](PROFILE_RESULTS.md) for the representative-core
performance bounds and optimization priorities.
