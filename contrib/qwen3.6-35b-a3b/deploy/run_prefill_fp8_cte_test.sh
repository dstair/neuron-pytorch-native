#!/usr/bin/env bash
# Small-model FP8xFP8 CTE prefill compile + fingerprint check (Step 5).
# Compiles a few layers at BS=1 with MOE_CTE_FP8 on/off and prints the prefill
# fingerprint for a drift comparison. Needs the BF16 checkpoint + patched nkilib.
# usage: run_prefill_fp8_cte_test.sh [layers] [bs] [bucket] [fp8=0|1]
set -uo pipefail
LAYERS="${1:-4}"; BS="${2:-1}"; BUCKET="${3:-1024}"; FP8="${4:-1}"
. "$(dirname "$0")/bench_env.sh"   # IMAGE/MODEL/NKILIB/SRC/WORK from .env or derived
TAG=l${LAYERS}-bs${BS}-b${BUCKET}-fp8${FP8}
LOG=$WORK/logs/prefill_fp8cte_${TAG}.log
mkdir -p $WORK/logs
NAME=q35-prefill-fp8cte-${TAG}
docker rm -f "$NAME" >/dev/null 2>&1 || true
if grep -q "only work on TRN2" "$NKILIB/src/nkilib_src/nkilib/core/moe/moe_cte/bwmm_shard_on_I.py"; then
  echo "FATAL: $NKILIB is NOT patched for LNC=1" | tee "$LOG"; exit 1
fi
echo "START $(date -u +%FT%TZ) tag=$TAG" | tee "$LOG"
docker run --rm --name "$NAME" --privileged --network host --ipc host \
  -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
  -e NEURON_LOGICAL_NC_CONFIG=1 -e QWEN35_LNC=1 \
  -e NEURON_SCRATCHPAD_PAGE_SIZE=256 \
  -e NEURON_CC_FLAGS="--target trn2 --lnc 1 --optlevel 1 --hbm-scratchpad-page-size 256" \
  -e TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1 \
  -e MOE_CTE=1 -e MOE_CTE_NKI_PACK=1 -e MOE_CTE_BLOCK=512 -e MOE_CTE_FP8=${FP8} \
  -e GQA_CTE_PREFILL=1 -e GQA_DYNAMIC_ROPE_KV=1 \
  -e DN_CHUNK_NKI=1 -e CHUNK_SIZE=32 -e DN_STABLE_C32=0 -e DN_PACK_C32=1 -e DN_PACK_N=4 \
  -e DN_NKI=1 -e GQATAIL=1 -e PREFILL_FINGERPRINT=1 \
  -e DN_K_HEADS=2 -e DN_V_HEADS=4 -e GQA_Q_HEADS=2 \
  -e PYTHONPATH=/nki-library/src/nkilib_src \
  -v /opt/aws/neuron:/opt/aws/neuron:ro \
  -v "$SRC":/work:ro -v "$NKILIB":/nki-library:ro \
  -v $WORK/nbackend:/tmp/neuron_backend \
  -v "$MODEL":/models/Qwen3.5-35B-A3B:ro \
  -w /work "$IMAGE" bash -lc "
    source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true
    torchrun --nproc-per-node=8 --max-restarts=0 static_decode_35b.py \
      --model-path /models/Qwen3.5-35B-A3B \
      --batch-size ${BS} --num-layers ${LAYERS} --max-seq-len 2048 \
      --prefill-bench ${BUCKET} --bucket-chunk ${BUCKET} \
      --bucket-compile 1 --prefill-splits 2 --skip-compile
  " >> "$LOG" 2>&1
RC=$?
echo "DOCKER_EXIT=$RC $(date -u +%FT%TZ)" | tee -a "$LOG"
exit $RC
