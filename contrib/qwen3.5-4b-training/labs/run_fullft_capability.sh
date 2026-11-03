#!/usr/bin/env bash
# The capability claim: full fine-tuning of all 4.206 B params does NOT fit on one core,
# and FSDP sharding the AdamW state is what makes it fit. Two runs, full 32 layers:
#
#   1. full-FT, single core        -> expected to OOM (47 GiB of state vs 24 GiB/core)
#   2. full-FT, FSDP2 world=4      -> expected to run  (~12 GiB of state per rank)
#
# Run 1 FAILING is the result, not a problem. The script reports it as EXPECTED OOM and
# keeps the log, because "FSDP is a speedup" and "FSDP is the only way this runs" are
# different claims and only the second one is true here.
#
#   labs/run_fullft_capability.sh [seq] [steps]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
LOGS="$(cd "$SRC/.." && pwd)/runs"
mkdir -p "$LOGS"

SEQ="${1:-512}"; STEPS="${2:-4}"
# Hard cap on the OOM run. Measured: the allocator does not fail fast -- it retries with
# progressively smaller requests (48 MB -> 2 MB -> ...), and the driver writes THREE memory
# dump files per failure into /tmp. One unattended run produced 669 dump files and ~400 MB
# of junk while making no progress. The OOM is proven within the first minute; the rest is
# the allocator refusing to give up.
OOM_TIMEOUT="${OOM_TIMEOUT:-420}"

echo "=== 1/2  full-FT, single core, full 32L  (expecting device OOM)"
LOG1="$LOGS/lab04_fullft_1core.log"
NAME="fullft_1core_$$"
NAME="$NAME" LNC=2 WORLD=1 timeout --signal=KILL "$OOM_TIMEOUT" \
  "$SRC/env/container.sh" exec \
  python3 -m sqlft.train --seq "$SEQ" --bs 1 --steps "$STEPS" \
  --tag lab04_fullft_1core > "$LOG1" 2>&1
# `timeout` kills the docker CLI, not the container it started, so reap it explicitly or it
# keeps holding the device and the next run fails with a confusing allocation error.
sudo docker kill "$NAME" >/dev/null 2>&1 || true
sudo find "$(cd "$SRC/.." && pwd)/cache" -maxdepth 1 \
  \( -name 'neuron_mem_table_*' -o -name 'nrt_mem_log_*' -o -name 'neuron_mem_log_*' \) \
  -delete 2>/dev/null || true

if grep -q "TRAIN OK" "$LOG1"; then
  echo "    RAN -- did NOT OOM. The 47 GiB estimate or the 24 GiB/core budget is wrong;"
  echo "    reconcile with sqlft/memory.py before quoting either."
  grep -E "warm|loss " "$LOG1" | tail -2
elif grep -qiE "Failed to allocate DEVICE memory|out of memory|OOM" "$LOG1"; then
  echo "    EXPECTED OOM: $(grep -oiE 'Failed to allocate DEVICE memory.*' "$LOG1" | head -1)"
else
  echo "    FAILED for some OTHER reason -- this does not support the claim:"
  tail -8 "$LOG1" | sed 's/^/      /'
fi

echo "=== 2/2  full-FT, FSDP2 world=4, full 32L  (expecting success)"
LOG2="$LOGS/lab04_fullft_fsdp2_w4_32L.log"
LNC=2 WORLD=4 HOST_CC=1 "$SRC/env/container.sh" exec \
  torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:29555 \
  labs/lab04_ddp_vs_fsdp.py --strategy fsdp2 --full-ft --seq "$SEQ" --steps "$STEPS" \
  --tag lab04_fullft_fsdp2_w4_32L > "$LOG2" 2>&1
if grep -q "LAB04 OK" "$LOG2"; then
  echo "    OK  $(grep -E 'agg tok/s' "$LOG2" | tail -1)"
else
  echo "    FAILED: $(grep -oE 'RuntimeError.*|Failed to allocate DEVICE memory.*' "$LOG2" | head -1)"
fi

echo "=== capability test done"
