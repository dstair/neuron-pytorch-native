#!/usr/bin/env bash
# Capture one region/model NEFF with COLLECTIVE REPLAY (TP=N graphs).
#
#   usage: capture_region.sh <tag> [nranks]
#   expects $W/<tag>.neff ; produces $W/<tag>_rank_0_exec_2.ntff
#
# Why each flag:
#   --collectives-worker-count/-r/-i : replay the all-reduces with N workers, profile worker 0.
#                                     Without this, any graph with a collective will not run.
#   --single-io                      : do not materialize every input (a 40L graph will not fit).
#   --profile-nth-exec=2             : skip warmup; note the OUTPUT gets an _exec_2 suffix.
#   --ignore-exec-errors             : synthetic inputs routinely produce garbage/errors; we
#                                      want the trace regardless.
#   DGE notifications are deliberately NOT enabled: at 40 layers one host notification per
#   indirect DMA overflows the queue (NRT status 1204). Cost: DmaPacket tables are incomplete.
#
# NEURON_LOGICAL_NC_CONFIG=1 is required for 8 logical cores, else:
#   "Logical NC not available 8/4".
set -uo pipefail
TAG="${1:?usage: capture_region.sh <tag> [nranks]}"
NRANKS="${2:-8}"

W=${W:-/mnt/nvme/lnc1-work/profile}
IMAGE=${IMAGE:-421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest}
LOG=$W/cap_${TAG}.log

echo "START $(date -u +%FT%TZ) capture tag=$TAG ranks=$NRANKS" > "$LOG"
docker run --rm --privileged \
  --device=/dev/neuron0 \
  -v /opt/aws/neuron/lib:/host_neuron_lib:ro \
  -e NEURON_LOGICAL_NC_CONFIG=1 \
  -e NEURON_SCRATCHPAD_PAGE_SIZE=256 \
  -v "$W":/work -w /work \
  "$IMAGE" bash -lc "
    export LD_LIBRARY_PATH=/host_neuron_lib:\$LD_LIBRARY_PATH
    source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true
    neuron-explorer capture -n /work/${TAG}.neff -s /work/${TAG}.ntff \
      --collectives-worker-count ${NRANKS} -r ${NRANKS} -i 0 --single-io \
      --profile-nth-exec=2 --ignore-exec-errors
  " >> "$LOG" 2>&1
echo "DOCKER_EXIT=$? $(date -u +%FT%TZ)" >> "$LOG"
ls -la "$W/${TAG}"*ntff* >> "$LOG" 2>&1 || true
tail -3 "$LOG"
