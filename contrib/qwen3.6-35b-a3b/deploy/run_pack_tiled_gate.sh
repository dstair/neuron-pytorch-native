#!/usr/bin/env bash
# Exact metadata gate for the route packer, on ONE core, with the packer path
# selectable. The existing matrix in test_moe_cte_nki_pack.py only covers
# 1024/2048 tokens = 8,192/16,384 assignments, both <= the direct-path limit, so
# the TILED stable scan has never been checked against the CPU reference.
# MOE_CTE_FORCE_TILED=1 routes the same matrix through it.
#   No MoE, no collectives, no 40-layer compile -> a correctness answer in
#   minutes instead of a 6-minute hang per guess.
# usage: run_pack_tiled_gate.sh [tiled=0|1] [lnc]
set -uo pipefail
TILED="${1:-1}"; LNC="${2:-1}"
IMAGE=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
NKILIB=/mnt/nvme/lnc1-work/nki-library
SRC=/mnt/nvme/lnc1-work/src/contrib/qwen3.6-35b-a3b
TAG=packgate-tl${TILED}-lnc${LNC}
CACHE=/mnt/nvme/lnc1-work/cache-${TAG}
LOG=/mnt/nvme/lnc1-work/logs/${TAG}.log
NAME=q35-${TAG}
docker rm -f "$NAME" >/dev/null 2>&1 || true
sudo rm -rf "$CACHE"
mkdir -p "$CACHE" /mnt/nvme/lnc1-work/logs
echo "START $(date -u +%FT%TZ) tag=$TAG force_tiled=$TILED" | tee "$LOG"
docker run --rm --name "$NAME" --privileged --network host --ipc host \
  -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
  -e NEURON_LOGICAL_NC_CONFIG="$LNC" -e QWEN35_LNC="$LNC" \
  -e NEURON_CC_FLAGS="--target trn2 --lnc $LNC --optlevel 1" \
  -e MOE_CTE=1 -e MOE_CTE_NKI_PACK=1 -e MOE_CTE_BLOCK=512 \
  -e MOE_CTE_FORCE_TILED="${TILED}" \
  -e PYTHONPATH=/nki-library/src/nkilib_src \
  -v /opt/aws/neuron:/opt/aws/neuron:ro \
  -v "$SRC":/work:ro -v "$NKILIB":/nki-library:ro \
  -v "$CACHE":/tmp \
  -w /work "$IMAGE" bash -lc "
    source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true
    python3 kernels/tests/test_moe_cte_nki_pack.py --backend device
  " >> "$LOG" 2>&1
RC=$?
echo "DOCKER_EXIT=$RC $(date -u +%FT%TZ)" | tee -a "$LOG"
exit $RC
