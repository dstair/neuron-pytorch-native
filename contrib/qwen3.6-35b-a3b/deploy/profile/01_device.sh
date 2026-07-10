#!/bin/bash
# Step 1: emit the BS=1 decode NEFF + per-rank device NTFF.
# NEURON_RT_INSPECT_DEVICE_PROFILE=1 inline + PROFILE_STEPS=5 → traces only the
# decode NEFF (warmup + 5 constant-token steps), not warmup/prefill graphs.
# Banked-best decode config: DN_NKI + MOE_SPARSE + GQATAIL + DNBATCHED_V2.
set -u
IMG=${ECR_REGISTRY}/concourse-release-0461d3b:latest
OUT=/mnt/nvme/prof_bs1
docker rm -f q35_prof 2>/dev/null; docker ps -q | xargs -r docker kill 2>/dev/null; sleep 2
rm -rf "$OUT"; mkdir -p "$OUT"
docker run --rm --name q35_prof --privileged --device=/dev/neuron0 \
  -e QWEN35_MODEL_PATH=/models/Qwen3.5-35B-A3B \
  -e DN_NKI=1 -e MOE_SPARSE=1 -e GQATAIL=1 -e DNBATCHED_V2=1 \
  -e PROFILE_STEPS=5 \
  -e NEURON_RT_INSPECT_ENABLE=1 -e NEURON_RT_INSPECT_DEVICE_PROFILE=1 \
  -e NEURON_RT_INSPECT_OUTPUT_DIR=/out \
  -v /home/ubuntu/qwen35:/work -v /mnt/nvme/Qwen3.5-35B-A3B:/models/Qwen3.5-35B-A3B \
  -v "$OUT":/out -v /home/ubuntu/logs:/logs \
  -w /work/Qwen3.6-35B-A3B "$IMG" bash -c '
source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null
torchrun --nproc-per-node=4 static_decode_35b.py --model-path /models/Qwen3.5-35B-A3B \
  --max-seq-len 2048 --num-tokens 4 --num-layers 40 --graph-splits 1 --batch-size 1 \
  > /logs/prof_device.log 2>&1 || true
echo "--- profile artifacts ---"; find /out -name "*.neff" -o -name "*.ntff" 2>/dev/null | head
'
echo "=== device profile artifacts ==="
find "$OUT" -name "*.neff" -o -name "*.ntff" 2>/dev/null | xargs -r ls -la
grep -iE "profile|BENCH|Error" /home/ubuntu/logs/prof_device.log 2>/dev/null | grep -vE "nccl|CCOM|OFI" | tail -4
