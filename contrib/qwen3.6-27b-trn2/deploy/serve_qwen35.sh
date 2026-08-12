#!/bin/bash
# Serve the qwen3_5 plugin fork in the configured vLLM-Neuron image.
# Args: $1=max_num_seqs (BS)  $2="extra env" (e.g. QWEN3_5_DELTANET_BATCHED=1)
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

BS="${1:-1}"
EXTRA_ENV="${2:-}"
IMG="$QWEN27_VLLM_IMAGE"
PLUGIN=/opt/conda/lib/python3.12/site-packages/vllm_neuron/model
FORK="$QWEN27_VLLM_PLUGIN_DIR"
require_env QWEN27_COMPILE_CACHE_DIR
mkdir -p "$QWEN27_COMPILE_CACHE_DIR"
CACHE_DIR="$(cd "$QWEN27_COMPILE_CACHE_DIR" && pwd)"

EXTRA_ENV_ARGS=()
if [[ -n "$EXTRA_ENV" ]]; then
  EXTRA_ENV_ARGS=(-e "$EXTRA_ENV")
fi

docker rm -f qwen35_serve 2>/dev/null
docker run -d --name qwen35_serve \
  --privileged --device=/dev/neuron0 --network host \
  -e NEURON_SKIP_EFA_AFFINITY=1 \
  -e NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  -e VLLM_PLUGINS=neuron \
  -e QWEN3_5_KERNELS_DIR=/kernels \
  -e VLLM_NEURON_COMPILATION_TIMEOUT=1800 \
  "${EXTRA_ENV_ARGS[@]}" \
  -v "$QWEN27_MODEL_DIR":/models/Qwen3.6-27B:ro \
  -v "$QWEN27_SOURCE_DIR/kernels":/kernels:ro \
  -v "$FORK/qwen3_5":"$PLUGIN/qwen3_5":ro \
  -v "$FORK/registry.py":"$PLUGIN/registry.py":ro \
  -v "$CACHE_DIR":/root/.cache/vllm/neuron/compile_cache \
  "$IMG" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3.6-27B \
    --tensor-parallel-size 4 \
    --max-model-len 256 \
    --max-num-seqs "$BS" \
    --no-enable-prefix-caching \
    --port 8000
echo "launched qwen35_serve BS=$BS"
sleep 5
docker logs qwen35_serve 2>&1 | tail -8
