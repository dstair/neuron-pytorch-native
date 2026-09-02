#!/usr/bin/env bash
# Step 6: full-model prefill at higher batch to test the FP8 memory win.
# FP8 (fp8=1) should LOAD BS=16 @ seq=10k at 40 layers, which OOMs in BF16.
# Same packed-C32 / CTE / sharded-vocab config as run_prefill_lnc1.sh, plus BS/seq
# params and MOE_CTE_FP8 passthrough. TP=8/LNC=1.
# usage: run_prefill_fp8_bs16_10k.sh [bs] [n_tokens] [fp8=0|1] [layers] [splits]
set -uo pipefail
BS="${1:-16}"; N="${2:-10000}"; FP8="${3:-1}"; LAYERS="${4:-40}"; SPLITS="${5:-4}"
BUCKET=1024
# max-seq-len rounded up to a bucket multiple that covers N.
MAXSEQ=$(( ( (N + BUCKET - 1) / BUCKET ) * BUCKET ))
. "$(dirname "$0")/bench_env.sh"   # IMAGE/MODEL/NKILIB/SRC/WORK from .env or derived
TAG=bs${BS}-n${N}-l${LAYERS}-s${SPLITS}-fp8${FP8}
LOG=$WORK/logs/prefill_fp8bs_${TAG}.log
CACHE=$WORK/cache-fp8bs-${TAG}
mkdir -p $WORK/logs "$CACHE"
NAME=q35-prefill-fp8bs-${TAG}
docker rm -f "$NAME" >/dev/null 2>&1 || true
if grep -q "only work on TRN2" "$NKILIB/src/nkilib_src/nkilib/core/moe/moe_cte/bwmm_shard_on_I.py"; then
  echo "FATAL: $NKILIB is NOT patched for LNC=1" | tee "$LOG"; exit 1
fi
echo "START $(date -u +%FT%TZ) tag=$TAG maxseq=$MAXSEQ" | tee "$LOG"
docker run --rm --name "$NAME" --privileged --network host --ipc host \
  -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
  -e NEURON_LOGICAL_NC_CONFIG=1 -e QWEN35_LNC=1 \
  -e NEURON_SCRATCHPAD_PAGE_SIZE=256 \
  -e NEURON_CC_FLAGS="--target trn2 --lnc 1 --optlevel 1 --hbm-scratchpad-page-size 256" \
  -e MOE_CTE=1 -e MOE_CTE_NKI_PACK=1 -e MOE_CTE_BLOCK=512 -e MOE_CTE_FP8=${FP8} \
  -e GQA_CTE_PREFILL=1 -e GQA_DYNAMIC_ROPE_KV=1 \
  -e DN_CHUNK_NKI=1 -e CHUNK_SIZE=32 -e DN_STABLE_C32=0 -e DN_PACK_C32=1 -e DN_PACK_N=4 \
  -e DN_NKI=1 -e GQATAIL=1 -e PREFILL_FINGERPRINT=1 \
  -e DN_K_HEADS=2 -e DN_V_HEADS=4 -e GQA_Q_HEADS=2 \
  -e PREFILL_SHARDED_LM_HEAD=1 -e PREFILL_SHARDED_EMBED=1 \
  -e PYTHONPATH=/nki-library/src/nkilib_src \
  -v /opt/aws/neuron:/opt/aws/neuron:ro \
  -v "$SRC":/work:ro -v "$NKILIB":/nki-library:ro \
  -v "$MODEL":/models/Qwen3.5-35B-A3B:ro \
  -v "$CACHE":/tmp \
  -w /work "$IMAGE" bash -lc "
    source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true
    torchrun --nproc-per-node=8 --max-restarts=0 static_decode_35b.py \
      --model-path /models/Qwen3.5-35B-A3B \
      --batch-size ${BS} --num-layers ${LAYERS} --max-seq-len ${MAXSEQ} \
      --prefill-bench ${N} --bucket-chunk ${BUCKET} \
      --bucket-compile 1 --prefill-splits ${SPLITS} --skip-compile
  " >> "$LOG" 2>&1
RC=$?
echo "DOCKER_EXIT=$RC $(date -u +%FT%TZ)" | tee -a "$LOG"
exit $RC
