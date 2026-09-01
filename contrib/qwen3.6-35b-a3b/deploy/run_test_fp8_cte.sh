#!/usr/bin/env bash
# Standalone device unit test for the FP8xFP8 CTE prefill MoE kernel.
# No model weights needed (synthetic inputs). Runs single-process at LNC=1.
# usage: run_test_fp8_cte.sh [tokens] [hidden] [intermediate] [block_size] [min_cosine]
set -uo pipefail
TOK="${1:-512}"; HID="${2:-512}"; INT="${3:-512}"; BLK="${4:-256}"; COS="${5:-0.95}"; LNC="${6:-1}"
IMAGE=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
NKILIB=/mnt/nvme/lnc1-work/nki-library
SRC=/mnt/nvme/lnc1-work/src/contrib/qwen3.6-35b-a3b
LOG=/mnt/nvme/lnc1-work/logs/test_fp8_cte.log
mkdir -p /mnt/nvme/lnc1-work/logs
NAME=q35-test-fp8-cte
docker rm -f "$NAME" >/dev/null 2>&1 || true
# fail loudly if the nkilib LNC=1 patch is missing (the baseline is_block_quant path
# needs the same NUM_SHARDS==1 relaxations as the BF16 CTE path).
if grep -q "only work on TRN2" "$NKILIB/src/nkilib_src/nkilib/core/moe/moe_cte/bwmm_shard_on_I.py"; then
  echo "FATAL: $NKILIB is NOT patched for LNC=1" | tee "$LOG"; exit 1
fi
echo "START $(date -u +%FT%TZ) tok=$TOK hid=$HID int=$INT blk=$BLK" | tee "$LOG"
docker run --rm --name "$NAME" --privileged --network host --ipc host \
  -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
  -e NEURON_LOGICAL_NC_CONFIG="$LNC" \
  -e NEURON_CC_FLAGS="--target trn2 --lnc $LNC" \
  -e PYTHONPATH=/nki-library/src/nkilib_src \
  -e NEURON_LAUNCH_BLOCKING=1 \
  -e TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1 \
  -v /opt/aws/neuron:/opt/aws/neuron:ro \
  -v "$SRC":/work:ro \
  -v "$NKILIB":/nki-library:ro \
  -v /mnt/nvme/lnc1-work/nbackend:/tmp/neuron_backend \
  -w /work "$IMAGE" bash -lc "
    source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true
    python3 kernels/tests/test_moe_cte_fp8_device.py \
      --tokens $TOK --hidden-size $HID --intermediate-size $INT \
      --block-size $BLK --min-cosine $COS
  " >> "$LOG" 2>&1
RC=$?
echo "DOCKER_EXIT=$RC $(date -u +%FT%TZ)" | tee -a "$LOG"
exit $RC
