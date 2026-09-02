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
# 8192 tokens/call -> max_blocks 160 (block 512) and 288 (block 256), i.e. 2 and 3
# block-metadata tiles, which is the only way to cover the >128-block path.
TOKENS="${3:-1024,2048,8192}"
# The tiled scan is barrier-free in production (its per-slice core_barrier is
# illegal at LNC=1), so force barriers OFF whenever we force the tiled path.
if [ "$TILED" = "1" ]; then BARRIER="${4:-0}"; else BARRIER="${4:-}"; fi
. "$(dirname "$0")/bench_env.sh"   # IMAGE/MODEL/NKILIB/SRC/WORK from .env or derived
TAG=packgate-tl${TILED}-lnc${LNC}
CACHE=$WORK/cache-${TAG}
LOG=$WORK/logs/${TAG}.log
NAME=q35-${TAG}
docker rm -f "$NAME" >/dev/null 2>&1 || true
sudo rm -rf "$CACHE"
mkdir -p "$CACHE" $WORK/logs
echo "START $(date -u +%FT%TZ) tag=$TAG force_tiled=$TILED tokens=$TOKENS barrier=${BARRIER:-default}" | tee "$LOG"
docker run --rm --name "$NAME" --privileged --network host --ipc host \
  -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
  -e NEURON_LOGICAL_NC_CONFIG="$LNC" -e QWEN35_LNC="$LNC" \
  -e NEURON_CC_FLAGS="--target trn2 --lnc $LNC --optlevel 1" \
  -e MOE_CTE=1 -e MOE_CTE_NKI_PACK=1 -e MOE_CTE_BLOCK=512 \
  -e MOE_CTE_FORCE_TILED="${TILED}" -e MOE_CTE_SCATTER_BARRIER="${BARRIER}" \
  -e PYTHONPATH=/nki-library/src/nkilib_src \
  -v /opt/aws/neuron:/opt/aws/neuron:ro \
  -v "$SRC":/work:ro -v "$NKILIB":/nki-library:ro \
  -v "$CACHE":/tmp \
  -w /work "$IMAGE" bash -lc "
    source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true
    python3 kernels/tests/test_moe_cte_nki_pack.py --backend device \
      --metadata-tokens ${TOKENS}
  " >> "$LOG" 2>&1
RC=$?
echo "DOCKER_EXIT=$RC $(date -u +%FT%TZ)" | tee -a "$LOG"
exit $RC
