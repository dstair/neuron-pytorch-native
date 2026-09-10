#!/usr/bin/env bash
set -euo pipefail

image="${QWEN4B_NATIVE_IMAGE:?set QWEN4B_NATIVE_IMAGE}"
model_dir="${QWEN4B_MODEL_DIR:?set QWEN4B_MODEL_DIR}"
cache_dir="${QWEN4B_CACHE_DIR:?set QWEN4B_CACHE_DIR}"
runtime_dir="${QWEN4B_RUNTIME_DIR:-$cache_dir/runtime}"
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$cache_dir" "$runtime_dir"

docker run --rm --entrypoint /bin/bash \
  --privileged --device=/dev/neuron0 --shm-size=32g \
  -v /opt/aws/neuron:/opt/aws/neuron:ro \
  -v "$model_dir":/models/Qwen3.5-4B:ro \
  -v "$source_dir":/work:ro \
  -v "$cache_dir":/ccache \
  -v "$runtime_dir":/tmp \
  -w /work \
  -e NEURON_LOGICAL_NC_CONFIG=2 \
  -e NEURON_RT_NUM_CORES=4 \
  -e "NEURON_CC_FLAGS=--target trn2 --lnc 2 --optlevel 1" \
  -e NEURON_COMPILE_CACHE_URL=/ccache \
  "$image" -c \
  'torchrun --nproc-per-node=4 static_decode.py "$@"' bash "$@"
