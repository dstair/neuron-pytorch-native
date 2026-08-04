#!/bin/bash
# DN_CONV_GROUP numerics gate. Kernel-only (single NKI compile, ~30 s at BS=8) --
# the cheap pre-check before any 64-layer end-to-end compile (~15 min each).
#
# Runs at the DEFAULT TP=4 head shape (K=4/V=12, NT=20), because
# test_deltanet_full_batched.py builds its reference from its own module constants
# and SystemExits on a DN_K_HEADS/DN_V_HEADS mismatch. NT=20 caps G at
# 128//20 = 6, so:
#   BS=8  -> largest divisor of 8 that is <=6 is 4  -> GNT = 80
#   BS=12 -> largest divisor of 12 that is <=6 is 6 -> GNT = 120  (128-aligned target)
#
# PASS CRITERION is not the test's own 1e-2/5e-2 tolerance -- it is that the group
# diffs match the DN_CONV_GROUP=0 diffs digit for digit. A silent fallback to G=1
# would also "pass", which is why the resolved G is echoed per leg.
set -u
IMG=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
LOG=/mnt/nvme/tmp/runlog
mkdir -p "$LOG"
: > "$LOG/grp_gate.log"

run() {  # run <bs> <group>
  local bs="$1" g="$2"
  echo "=== BS=$bs DN_CONV_GROUP=$g (TP=4 shape, NT=20) ===" >> "$LOG/grp_gate.log"
  docker rm -f "q27b-grp-$bs-$g" >/dev/null 2>&1
  timeout 2400 docker run --rm --name "q27b-grp-$bs-$g" --privileged \
    -e LD_LIBRARY_PATH=/opt/aws/neuron/lib \
    -e NEURON_LOGICAL_NC_CONFIG=2 -e PYTHONPATH=/pysite \
    -e KERNEL_MODULE=deltanet_full_batched_v2 \
    -e DN_TILED_CONV=1 -e DN_CONV_GROUP="$g" -e BS="$bs" \
    -v /opt/aws/neuron:/opt/aws/neuron:ro \
    -v /mnt/nvme/tmp/pysite:/pysite:ro \
    -v /mnt/nvme/code/q27b-grp:/work:ro \
    -v /mnt/nvme/tmp/q27b-cache:/tmp \
    -w /work/kernels/tests "$IMG" \
    python3 test_deltanet_full_batched.py >> "$LOG/grp_gate.log" 2>&1
  echo "--- rc=$? ---" >> "$LOG/grp_gate.log"
}

run 8 0
run 8 4
run 12 0
run 12 6
echo "GRP GATE DONE" >> "$LOG/grp_gate.log"
