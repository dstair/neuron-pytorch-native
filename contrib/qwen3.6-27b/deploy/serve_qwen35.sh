#!/bin/bash
# Serve the qwen3_5 plugin (fork) in the 28ce3c3 vLLM-Neuron image.
# Args: $1=max_num_seqs (BS)  $2="extra env" (e.g. QWEN3_5_DELTANET_BATCHED=1)
set -u
BS="${1:-1}"
EXTRA_ENV="${2:-}"
IMG=${ECR_REGISTRY}/concourse-release-28ce3c3:latest
PLUGIN=/opt/conda/lib/python3.12/site-packages/vllm_neuron/model
FORK=/home/ubuntu/qwen3_5_fork/vllm_neuron/model
mkdir -p /mnt/nvme/vllm_compile_cache

docker rm -f qwen35_serve 2>/dev/null
docker run -d --name qwen35_serve \
  --privileged --device=/dev/neuron0 --network host \
  -e NEURON_SKIP_EFA_AFFINITY=1 \
  -e NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  -e VLLM_PLUGINS=neuron \
  -e QWEN3_5_KERNELS_DIR=/kernels \
  -e VLLM_NEURON_COMPILATION_TIMEOUT=1800 \
  $( [ -n "$EXTRA_ENV" ] && echo "-e $EXTRA_ENV" ) \
  -v /mnt/nvme/Qwen3.6-27B:/models/Qwen3.6-27B:ro \
  -v /home/ubuntu/qwen3_6/kernels:/kernels:ro \
  -v $FORK/qwen3_5:$PLUGIN/qwen3_5:ro \
  -v $FORK/registry.py:$PLUGIN/registry.py:ro \
  -v /mnt/nvme/vllm_compile_cache:/root/.cache/vllm/neuron/compile_cache \
  "$IMG" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3.6-27B \
    --tensor-parallel-size 4 \
    --max-model-len 256 \
    --max-num-seqs $BS \
    --no-enable-prefix-caching \
    --port 8000
echo "launched qwen35_serve BS=$BS"
sleep 5
docker logs qwen35_serve 2>&1 | tail -8
