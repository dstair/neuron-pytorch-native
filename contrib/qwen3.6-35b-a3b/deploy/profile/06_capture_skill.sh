#!/bin/bash
# Device capture using the neuron-nki-profiling skill's flags:
#   --profile-nth-exec=2 (skip warmup) + --enable-dge-notifs (DMA detail).
# These were MISSING before — likely why capture emitted no NTFF. Replays the
# real decode NEFF (identified by size/identify-neffs) with 4-worker collectives.
set -u
IMG=${ECR_REGISTRY}/concourse-release-0461d3b:latest
DEV=/mnt/nvme/prof_dev
docker rm -f q35_cap q35_ne_ui 2>/dev/null; docker ps -q | xargs -r docker kill 2>/dev/null; sleep 2
NEFF=$(find "$DEV" -name "neff_537715910768540_vnc_0.neff" | head -1)
echo "=== identify NEFFs (confirm which is the decode kernel) ==="
docker run --rm -v /mnt/nvme:/mnt/nvme "$IMG" bash -lc \
  "source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null; python3 /work/Qwen3.6-35B-A3B/deploy/profile/identify-neffs.py $(dirname "$NEFF") 2>&1 | head -20" \
  2>/dev/null || echo "(identify skipped — /tmp workdirs gone)"
echo "=== capture WITH skill flags (--profile-nth-exec=2 --enable-dge-notifs) ==="
docker run --rm --name q35_cap --privileged --device=/dev/neuron0 -v /mnt/nvme:/mnt/nvme "$IMG" \
  bash -lc "/opt/aws/neuron/bin/neuron-explorer capture -n '$NEFF' -s $DEV/device.ntff \
    --collectives-worker-count 4 -r 4 --profile-nth-exec=2 --enable-dge-notifs --ignore-exec-errors 2>&1 \
    | grep -ivE 'nccl|CCOM|OFI' | tail -15"
ls -la "$DEV"/device.ntff 2>/dev/null && echo "NTFF OK" || echo "still no NTFF"
