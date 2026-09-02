#!/usr/bin/env bash
# Standalone device unit test for the FP8xFP8 CTE prefill MoE kernel.
# No model weights needed (synthetic inputs). Runs single-process at LNC=1.
# usage: run_test_fp8_cte.sh [tokens] [hidden] [intermediate] [block_size] [min_cosine]
set -uo pipefail
TOK="${1:-512}"; HID="${2:-512}"; INT="${3:-512}"; BLK="${4:-256}"; COS="${5:-0.95}"; LNC="${6:-1}"
. "$(dirname "$0")/bench_env.sh"   # IMAGE/MODEL/NKILIB/SRC/WORK from .env or derived
LOG=$WORK/logs/test_fp8_cte.log
mkdir -p $WORK/logs
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
  -v $WORK/nbackend:/tmp/neuron_backend \
  -w /work "$IMAGE" bash -lc "
    source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true
    python3 kernels/tests/test_moe_cte_fp8_device.py \
      --tokens $TOK --hidden-size $HID --intermediate-size $INT \
      --block-size $BLK --min-cosine $COS
  " >> "$LOG" 2>&1
RC=$?
echo "DOCKER_EXIT=$RC $(date -u +%FT%TZ)" | tee -a "$LOG"
exit $RC
