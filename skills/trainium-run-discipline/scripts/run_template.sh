#!/usr/bin/env bash
# CLEAN runner template for a TP=8 / LNC=1 model run on trn2. Every number you quote should
# come from a script shaped like this one.
#
#   usage: run_template.sh <bs> <ntokens> <layers> <splits> <chunk>
#
# DELIBERATELY ABSENT (add ONLY to a separate debug copy, never here):
#   -e NEURON_LAUNCH_BLOCKING=1
#   -e NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1     # NRT status 1204 NQ overflow at 40 layers
#   -e XLA_IR_DEBUG=1 -e XLA_HLO_DEBUG=1 -e NEURON_FRAMEWORK_DEBUG=1
#   -e TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1  -v ./nbackend:/tmp/neuron_backend
# Measured cost of those together: ~15x. Any timing taken with them is not a number.
set -uo pipefail
BS="${1:-2}"; N="${2:-10000}"; LAYERS="${3:-40}"; SPLITS="${4:-4}"; CHUNK="${5:-1024}"

IMAGE=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
MODEL=/mnt/nvme/Qwen3.5-35B-A3B
SRC=/mnt/nvme/lnc1-work/src/contrib/qwen3.6-35b-a3b
NKILIB=/mnt/nvme/lnc1-work/nki-library
WORK=/mnt/nvme/lnc1-work

# TAG must encode EVERY parameter that changes the graph, or caches collide silently.
TAG=bs${BS}-n${N}-l${LAYERS}-s${SPLITS}-bc${CHUNK}
LOG=$WORK/logs/run_${TAG}.log
CACHE=$WORK/cache-${TAG}
NAME=run-${TAG}
MAXSEQ=$(( ( (N + CHUNK - 1) / CHUNK ) * CHUNK ))

docker rm -f "$NAME" >/dev/null 2>&1 || true
# SUDO is load-bearing: the privileged container makes these root-owned, and a plain rm fails
# SILENTLY, serving a stale NEFF on every later run.
sudo rm -rf "$CACHE" "$WORK/nbackend"
mkdir -p "$WORK/logs" "$CACHE"

echo "START $(date -u +%FT%TZ) tag=$TAG maxseq=$MAXSEQ" | tee "$LOG"
docker run --rm --name "$NAME" --privileged --network host --ipc host \
  -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
  -e NEURON_LOGICAL_NC_CONFIG=1 \
  -e NEURON_SCRATCHPAD_PAGE_SIZE=256 \
  -e NEURON_CC_FLAGS="--target trn2 --lnc 1 --optlevel 1 --hbm-scratchpad-page-size 256" \
  -e NKI_ENABLE_TRACE_CACHE=0 \
  -e PYTHONPATH=/nki-library/src/nkilib_src \
  -v /opt/aws/neuron:/opt/aws/neuron:ro \
  -v "$SRC":/work:ro -v "$NKILIB":/nki-library:ro \
  -v "$MODEL":/models/model:ro \
  -v "$CACHE":/tmp \
  -w /work "$IMAGE" bash -lc "
    source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true
    torchrun --nproc-per-node=8 --max-restarts=0 static_decode_35b.py \
      --model-path /models/model \
      --batch-size ${BS} --num-layers ${LAYERS} --max-seq-len ${MAXSEQ} \
      --prefill-bench ${N} --bucket-chunk ${CHUNK} \
      --bucket-compile 1 --prefill-splits ${SPLITS} --skip-compile
  " >> "$LOG" 2>&1
RC=$?
echo "DOCKER_EXIT=$RC $(date -u +%FT%TZ)" | tee -a "$LOG"

# Gate on the APP's marker, not $RC: teardown SIGABRT/SIGSEGV fires AFTER valid results.
echo "--- verdict ---"
grep -E "PREFILL TIMED|GB/core" "$LOG" || echo "  NO RESULT MARKER"
grep -qE "NRT status 5|execution timeout" "$LOG" && echo "  !! HANG (NRT status 5)"
grep -qE "Failed to allocate DEVICE memory" "$LOG" && echo "  !! DEVICE OOM AT LOAD (shrink batch x chunk)"
grep -qE "status 1204|notification queue overflow" "$LOG" && echo "  !! NQ overflow -- debug DGE flags leaked in"
grep -q "0 compile activity" "$LOG" && echo "  !! STALE NEFF -- cache was not cleared as root"
grep -E "fingerprint" "$LOG" | tail -1
exit 0
