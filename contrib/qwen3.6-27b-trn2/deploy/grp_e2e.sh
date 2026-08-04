#!/bin/bash
# DN_CONV_GROUP end-to-end decode A/B at TP=8 / LNC=1, 64 layers, beta-4.
#
# The gate (grp_gate.sh) proved numerics are digit-identical. This measures whether
# widening Phase-1 buys throughput, and it is designed around the divisor constraint
# G | B, which is what actually binds:
#
#   BS=8  -> G=8  -> GNT=80    (not 128-aligned)
#   BS=12 -> G=12 -> GNT=120   (128-aligned target -- the discriminating leg)
#
# So BS=8 vs BS=12 is the alignment experiment: if the win is much larger at BS=12,
# free-dim alignment is the lever; if both move the same, it is just batching the DMAs.
# Every leg carries T1a (DN_TILED_CONV=1), so the delta isolates GROUP on top of T1a.
#
# --max-seq-len 1024 throughout: BS=16 needs it at TP=8 and it keeps all legs
# comparable. Each cold leg is ~15-20 min (new NEFF per G), so budget ~1.5 h.
set -u
IMG=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
LOG=/mnt/nvme/tmp/runlog
mkdir -p "$LOG"
: > "$LOG/grp_e2e.log"

runx() {  # runx <tag> <extra-env-csv> <harness args...>
  local tag="$1" envs="$2"; shift 2
  docker rm -f "q27b-$tag" >/dev/null 2>&1
  local ex=()
  if [ -n "$envs" ]; then
    local IFS=,
    for kv in $envs; do ex+=(-e "$kv"); done
  fi
  timeout 5400 docker run --rm --name "q27b-$tag" --privileged \
    -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
    -e NEURON_LOGICAL_NC_CONFIG=1 -e PYTHONPATH=/pysite \
    -e TP_DEGREE=8 \
    -e SHARDED_EMBED=1 -e SHARDED_LM_HEAD=1 "${ex[@]}" \
    -v /opt/aws/neuron:/opt/aws/neuron:ro -v /mnt/nvme/tmp/pysite:/pysite:ro \
    -v /mnt/nvme/code/q27b-grp:/work:ro \
    -v /mnt/nvme/Qwen3.6-27B:/models/Qwen3.6-27B:ro \
    -v /mnt/nvme/tmp/q27b-cache:/tmp \
    -w /work "$IMG" \
    torchrun --nproc-per-node=8 --max-restarts=0 static_decode.py "$@" \
    > "$LOG/$tag.log" 2>&1
  echo "=== $tag (env=$envs args=$*) rc=$? ===" >> "$LOG/grp_e2e.log"
  grep -E "MB/core|Generated:|host sync|on-device greedy|ms/tok|Error|Traceback" \
    "$LOG/$tag.log" | tail -12 >> "$LOG/grp_e2e.log"
  echo "--- leg $tag done ---" >> "$LOG/grp_e2e.log"
}

BASE="DNBATCHED_V2=1,GQATAIL=1,DN_TILED_CONV=1"
D="--num-layers 64 --max-seq-len 1024"

# BS=12: the 128-aligned leg (G=12 -> GNT=120). Control first.
runx grp-bs12-t1a  "$BASE"                  $D --batch-size 12
runx grp-bs12-g12  "$BASE,DN_CONV_GROUP=12" $D --batch-size 12

# BS=8: the unaligned leg (G=8 -> GNT=80), against the already-published 92.0 tok/s.
runx grp-bs8-t1a   "$BASE"                  $D --batch-size 8
runx grp-bs8-g8    "$BASE,DN_CONV_GROUP=12" $D --batch-size 8

echo "--- ALL GRP E2E DONE ---" >> "$LOG/grp_e2e.log"
