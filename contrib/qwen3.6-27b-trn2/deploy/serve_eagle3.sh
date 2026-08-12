#!/bin/bash
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
require_dir QWEN27_EAGLE3_MODEL_DIR

IMG="$QWEN27_VLLM_IMAGE"
PLUGIN=/opt/conda/lib/python3.12/site-packages/vllm_neuron/model
FORK="$QWEN27_VLLM_PLUGIN_DIR"
require_env QWEN27_EAGLE3_COMPILE_CACHE_DIR
mkdir -p "$QWEN27_EAGLE3_COMPILE_CACHE_DIR"
CACHE_DIR="$(cd "$QWEN27_EAGLE3_COMPILE_CACHE_DIR" && pwd)"
CONFIG="$SCRIPT_DIR/eagle3_config.yaml"
docker rm -f qwen35_serve 2>/dev/null >/dev/null
# Pass all server args via a YAML --config file. Avoids the JSON-on-CLI quoting
# that the default dockerd-entrypoint.py shlex round-trip mangles, and keeps the
# normal module entrypoint (so vLLM multiprocessing worker spawn works).
docker run -d --name qwen35_serve \
  --privileged --device=/dev/neuron0 --network host \
  -e NEURON_SKIP_EFA_AFFINITY=1 -e NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  -e VLLM_PLUGINS=neuron -e QWEN3_5_KERNELS_DIR=/kernels \
  -e QWEN3_5_CHUNKED_PREFILL=1 -e QWEN3_5_SPEC_VERIFY_LEN=4 \
  -v "$QWEN27_MODEL_DIR":/models/Qwen3.6-27B:ro \
  -v "$QWEN27_EAGLE3_MODEL_DIR":/models/eagle3-draft:ro \
  -v "$QWEN27_SOURCE_DIR/kernels":/kernels:ro \
  -v "$CONFIG":/eagle3_config.yaml:ro \
  -v "$FORK/qwen3_5":"$PLUGIN/qwen3_5":ro \
  -v "$FORK/registry.py":"$PLUGIN/registry.py":ro \
  -v "$CACHE_DIR":/root/.cache/vllm/neuron/compile_cache \
  "$IMG" \
  python3 -m vllm.entrypoints.openai.api_server --config /eagle3_config.yaml >/dev/null
echo "launched eagle3 spec-decode serve (yaml config)"
