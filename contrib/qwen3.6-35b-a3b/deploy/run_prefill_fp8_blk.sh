#!/usr/bin/env bash
# Block-size discriminator: same clean-flag prefill runner as
# run_prefill_fp8_bucket_clean.sh, but MOE_CTE_BLOCK is a parameter (was hardcoded
# 512). Lets block COUNT, tokens-per-MoE-call and packed_len be varied independently
# to isolate which one owns the CTE ">64-block" hang.
#   max_blocks = ceil((BS*BUCKET*8 - 32)/BLK) + 32 ; packed_len = max_blocks*BLK
# NO debug flags (LAUNCH_BLOCKING / DGE notifications cost ~15x).
# usage: run_prefill_fp8_blk.sh [bs] [n_tokens] [fp8=0|1] [layers] [splits] [bucket] [blk]
set -uo pipefail
BS="${1:-2}"; N="${2:-1024}"; FP8="${3:-1}"; LAYERS="${4:-4}"; SPLITS="${5:-2}"
BUCKET="${6:-1024}"; BLK="${7:-256}"
MAXSEQ=$(( ( (N + BUCKET - 1) / BUCKET ) * BUCKET ))
IMAGE=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
MODEL=/mnt/nvme/Qwen3.5-35B-A3B
NKILIB=/mnt/nvme/lnc1-work/nki-library
SRC=/mnt/nvme/lnc1-work/src/contrib/qwen3.6-35b-a3b
TAG=bs${BS}-n${N}-l${LAYERS}-s${SPLITS}-bc${BUCKET}-blk${BLK}-fp8${FP8}
LOG=/mnt/nvme/lnc1-work/logs/prefill_fp8bs_${TAG}.log
CACHE=/mnt/nvme/lnc1-work/cache-fp8bs-${TAG}
NAME=q35-prefill-fp8bs-${TAG}
docker rm -f "$NAME" >/dev/null 2>&1 || true
# Caches are created root-owned by the privileged container; a non-sudo rm fails
# SILENTLY and a stale NEFF gets served. Always sudo-clear.
sudo rm -rf "$CACHE" /mnt/nvme/lnc1-work/nbackend
mkdir -p /mnt/nvme/lnc1-work/logs "$CACHE"
if grep -q "only work on TRN2" "$NKILIB/src/nkilib_src/nkilib/core/moe/moe_cte/bwmm_shard_on_I.py"; then
  echo "FATAL: $NKILIB is NOT patched for LNC=1" | tee "$LOG"; exit 1
fi
echo "START $(date -u +%FT%TZ) tag=$TAG maxseq=$MAXSEQ blk=$BLK" | tee "$LOG"
docker run --rm --name "$NAME" --privileged --network host --ipc host \
  -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
  -e NEURON_LOGICAL_NC_CONFIG=1 -e QWEN35_LNC=1 \
  -e NEURON_SCRATCHPAD_PAGE_SIZE=256 \
  -e NEURON_CC_FLAGS="--target trn2 --lnc 1 --optlevel 1 --hbm-scratchpad-page-size 256" \
  -e NKI_ENABLE_TRACE_CACHE=0 \
  -e MOE_CTE=1 -e MOE_CTE_NKI_PACK=1 -e MOE_CTE_BLOCK=${BLK} -e MOE_CTE_FP8=${FP8} \
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
