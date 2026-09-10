#!/usr/bin/env bash
set -euo pipefail

image="${QWEN4B_NATIVE_IMAGE:?set QWEN4B_NATIVE_IMAGE}"
model_dir="${QWEN4B_MODEL_DIR:?set QWEN4B_MODEL_DIR}"
runtime_dir="${QWEN4B_RUNTIME_TEMPLATE:?set QWEN4B_RUNTIME_TEMPLATE}"
cache_dir="${QWEN4B_CACHE_DIR:-$runtime_dir/ccache}"
batch_size="${QWEN4B_BATCH_SIZE:-32}"
fp8_scope="${QWEN4B_FP8_SCOPE:-mlp-head}"
fused_fp8_mlp="${QWEN4B_FUSED_FP8_MLP:-1}"
fused_fp8_head="${QWEN4B_FUSED_FP8_HEAD:-1}"
visible_core="${QWEN4B_VISIBLE_CORE:-0}"
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fused_args=""
[[ "$fused_fp8_mlp" == "1" ]] && fused_args+=" --fused-fp8-mlp"
[[ "$fused_fp8_head" == "1" ]] && fused_args+=" --fused-fp8-head"

mkdir -p "$runtime_dir" "$cache_dir"

docker run --rm --entrypoint /bin/bash \
  --privileged --device=/dev/neuron0 --shm-size=16g \
  -v /opt/aws/neuron:/opt/aws/neuron:ro \
  -v "$model_dir":/models/Qwen3.5-4B:ro \
  -v "$source_dir":/work:ro \
  -v "$runtime_dir":/tmp \
  -v "$cache_dir":/ccache \
  -w /work \
  -e NEURON_LOGICAL_NC_CONFIG=1 \
  -e NEURON_RT_NUM_CORES=1 \
  -e NEURON_RT_VISIBLE_CORES="$visible_core" \
  -e "NEURON_CC_FLAGS=--target trn2 --lnc 1 --optlevel 1" \
  -e NEURON_COMPILE_CACHE_URL=/ccache \
  "$image" -c \
  "python3 static_decode_dp8.py \
    --model-path /models/Qwen3.5-4B \
    --batch-size $batch_size \
    --max-seq-len 256 \
    --warmup 3 \
    --bench-iters 5 \
    --fp8-scope $fp8_scope \
    $fused_args \
    --replica-id 0 \
    --replica-count 1"
