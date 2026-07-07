#!/bin/bash
# Step: capture device+system profile of the BS=1 banked-best decode via the
# `inspect` subcommand (wraps the workload, traces the whole process tree).
# Needs the neuron cores FREE (kill stragglers first). Writes per-rank dirs with
# the device NEFF/NTFF + system ntrace.pb on workload EXIT.
set -u
IMG=${ECR_REGISTRY}/concourse-release-0461d3b:latest
OUT=/mnt/nvme/prof_bs1
docker rm -f q35_prof q35_ne_ui 2>/dev/null; docker ps -q | xargs -r docker kill 2>/dev/null; sleep 2
rm -rf "$OUT"; mkdir -p "$OUT"
docker run --rm --name q35_prof --privileged --device=/dev/neuron0 \
  -v /home/ubuntu/qwen35:/work -v /mnt/nvme/Qwen3.5-35B-A3B:/models/Qwen3.5-35B-A3B \
  -v "$OUT":/out \
  -w /work/Qwen3.6-35B-A3B "$IMG" bash -lc '
  /opt/aws/neuron/bin/neuron-explorer inspect -o /out bash /work/Qwen3.6-35B-A3B/deploy/profile/workload_bs1.sh
' 2>&1 | grep -ivE "nccl|CCOM|OFI|Warning|registered at|dispatch key|previous kernel|new kernel|operator:" | tail -20
echo "=== profile artifacts (per-rank) ==="
find "$OUT" -maxdepth 3 -type d | head
find "$OUT" -name "*.neff" -o -name "*.ntff" -o -name "ntrace.pb" 2>/dev/null | xargs -r ls -la | head
