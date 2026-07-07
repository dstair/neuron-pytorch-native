#!/bin/bash
set -u
IMG=${ECR_REGISTRY}/concourse-release-28ce3c3:latest
PLUGIN=/opt/conda/lib/python3.12/site-packages/vllm_neuron/model
FORK=/home/ubuntu/qwen3_5_fork/vllm_neuron/model
mkdir -p /mnt/nvme/vllm_compile_cache_eagle3
docker rm -f qwen35_serve 2>/dev/null >/dev/null
# Pass all server args via a YAML --config file. Avoids the JSON-on-CLI quoting
# that the default dockerd-entrypoint.py shlex round-trip mangles, and keeps the
# normal module entrypoint (so vLLM multiprocessing worker spawn works).
docker run -d --name qwen35_serve \
  --privileged --device=/dev/neuron0 --network host \
  -e NEURON_SKIP_EFA_AFFINITY=1 -e NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  -e VLLM_PLUGINS=neuron -e QWEN3_5_KERNELS_DIR=/kernels \
  -e QWEN3_5_CHUNKED_PREFILL=1 -e QWEN3_5_SPEC_VERIFY_LEN=4 \
  -v /mnt/nvme/Qwen3.6-27B:/models/Qwen3.6-27B:ro \
  -v /mnt/nvme/Qwen3.6-27B-EAGLE3/full:/models/eagle3-draft:ro \
  -v /home/ubuntu/qwen3_6/kernels:/kernels:ro \
  -v /home/ubuntu/eagle3_config.yaml:/eagle3_config.yaml:ro \
  -v $FORK/qwen3_5:$PLUGIN/qwen3_5:ro \
  -v $FORK/registry.py:$PLUGIN/registry.py:ro \
  -v /mnt/nvme/vllm_compile_cache_eagle3:/root/.cache/vllm/neuron/compile_cache \
  "$IMG" \
  python3 -m vllm.entrypoints.openai.api_server --config /eagle3_config.yaml >/dev/null
echo "launched eagle3 spec-decode serve (yaml config)"
