#!/bin/bash
# Cross-compile the 27B decode NEFF for trn2/LNC=1 ON A trn1.32xlarge, then ship the
# compiler cache to S3 for a native Trn2 run to pick up as a CACHE HIT.
#
# Why this exists: a 64-layer TP=8 decode compile is ~15-20 min of the ~17-20 min
# "cold leg" on the trn2.3xlarge, and that box has far less host RAM than a
# trn1.32xlarge (495 GB). Compile is the dev-loop bottleneck; this moves it off the
# scarce device host.
#
# How it works (three cooperating pieces):
#   1. NEURON_CC_FLAGS="--target trn2 --lnc 1" makes neuronx-cc emit trn2 code.
#   2. The LD_PRELOAD shim (contrib/qwen3.6-35b-a3b/deploy/cross_compile/) intercepts
#      nrt_get_instance_info and reports TRN2 *only* when called from the compiler-
#      cache-key constructor. Without it the key says trn1 and the Trn2 run misses.
#   3. CROSS_TARGET_COMPILE_ONLY=1 (static_decode.py step 6b) catches the inevitable
#      "Invalid NEFF" device load -- the local hardware really is trn1 -- AFTER the
#      cache write, then exits clean via filesystem rank markers.
#
# --tiny --tiny-full-vocab: full 64 layers and the REAL 248320 vocab, but random
# weights. Shape-identical graph to production, so the cache entry is a hit, and it
# skips the 51.8 GB checkpoint download entirely.
#
# Usage (on the trn1 host):  ./cross_compile_trn1.sh <batch-size> [extra-env-csv]
#   ./cross_compile_trn1.sh 12 "DN_CONV_GROUP=12"
set -euo pipefail

BS="${1:?usage: cross_compile_trn1.sh <batch-size> [extra-env-csv]}"
EXTRA_ENV="${2:-}"
LAYERS="${LAYERS:-64}"
SEQ="${SEQ:-1024}"
TP="${TP:-8}"
# Host-RAM throttle: ranks compile in waves. 8 concurrent 64-layer neuronx-cc runs
# is what produced the [F137] "forcibly killed" OOMs on smaller hosts; 495 GB should
# take all 8, but this is the knob to turn down first if a compile gets killed.
CONCURRENCY="${CONCURRENCY:-8}"

IMG=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
WORK=/opt/xc/q27b            # source snapshot
SHIM=/opt/xc/shim            # built libnrt_platform_target_override.so lives here
CACHE=/opt/xc/cache          # persistent compiler cache (the artifact we ship)
LOG=/opt/xc/log
TAG="bs${BS}-${LAYERS}L-s${SEQ}-tp${TP}${EXTRA_ENV:+-$(echo "$EXTRA_ENV" | tr ',=' '--')}"

mkdir -p "$CACHE" "$LOG"
rm -rf /opt/xc/markers && mkdir -p /opt/xc/markers

ex=()
if [ -n "$EXTRA_ENV" ]; then
  IFS=, read -ra kvs <<< "$EXTRA_ENV"
  for kv in "${kvs[@]}"; do ex+=(-e "$kv"); done
fi

echo "=== cross-compile $TAG (concurrency=$CONCURRENCY) ==="
set -x
docker run --rm --name "q27b-xc-$TAG" --privileged \
  -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
  -e LD_PRELOAD=/opt/xc/libnrt_platform_target_override.so \
  -e QWEN35_CACHE_PLATFORM_TARGET=trn2 \
  -e NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  -e NEURON_CC_FLAGS="--target trn2 --lnc 1" \
  -e NEURON_COMPILE_CACHE_URL=/xc-cache \
  -e CROSS_TARGET_COMPILE_ONLY=1 \
  -e CROSS_TARGET_MARKER_DIR=/xc-markers \
  -e CROSS_TARGET_COMPILE_CONCURRENCY="$CONCURRENCY" \
  -e NEURON_LOGICAL_NC_CONFIG=1 \
  -e TP_DEGREE="$TP" \
  -e SHARDED_EMBED=1 -e SHARDED_LM_HEAD=1 \
  -e DNBATCHED_V2=1 -e GQATAIL=1 -e DN_TILED_CONV=1 "${ex[@]}" \
  -v /opt/aws/neuron:/opt/aws/neuron:ro \
  -v "$SHIM/libnrt_platform_target_override.so":/opt/xc/libnrt_platform_target_override.so:ro \
  -v "$WORK":/work:ro \
  -v "$CACHE":/xc-cache \
  -v /opt/xc/markers:/xc-markers \
  -w /work "$IMG" \
  torchrun --nproc-per-node="$TP" --max-restarts=0 static_decode.py \
    --tiny --tiny-full-vocab --num-layers "$LAYERS" \
    --batch-size "$BS" --max-seq-len "$SEQ" --skip-prefill \
  > "$LOG/xc-$TAG.log" 2>&1
set +x

echo "=== done: $(du -sh "$CACHE" | cut -f1) in cache ==="
grep -E "cross-target|Invalid NEFF|Error|Traceback" "$LOG/xc-$TAG.log" | tail -20
