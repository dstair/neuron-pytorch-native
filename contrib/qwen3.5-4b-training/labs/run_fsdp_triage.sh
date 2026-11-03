#!/usr/bin/env bash
# Timeboxed triage for the FSDP2 in-graph allgather failure.
#
#   NMGR:dlr_build_and_load_cc_resources  Failed to build and load collectives resources
#       for module.neff, exec_id 469
#   RuntimeError: NRT Execution error ... operation=allgather
#       "NRT EXECUTION FAILED: NRT model scheduling failed, Non-specific failure"
#
# Bare eager all_gather / reduce_scatter both pass (labs/lab_collectives.py), so the problem
# is specific to how FSDP2 issues them. Three hypotheses, one run each, cheapest first:
#
#   A  FSDP2 + HOST_CC=1  -- host-side collectives instead of in-NEFF. If the NEFF cannot
#                            carry collective resources, moving them to the host should fix it.
#   B  FSDP1              -- flat-parameter FSDP issuing plain c10d collectives rather than
#                            DTensor functional ones. This is the path the prior bring-up used.
#   C  FSDP1 + HOST_CC=1  -- both, to separate "FSDP2 is the problem" from "collectives are".
#
#   labs/run_fsdp_triage.sh [world] [layers] [seq] [steps]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
LOGS="$(cd "$SRC/.." && pwd)/runs"
mkdir -p "$LOGS"

WORLD="${1:-4}"; LAYERS="${2:-8}"; SEQ="${3:-512}"; STEPS="${4:-6}"
LNC=2; [ "$WORLD" -gt 4 ] && LNC=1

run_one() {
  local tag="$1" hostcc="$2"; shift 2
  local log="$LOGS/$tag.log"
  echo "=== $tag  (HOST_CC=$hostcc world=$WORLD LNC=$LNC)"
  LNC="$LNC" WORLD="$WORLD" HOST_CC="$hostcc" "$SRC/env/container.sh" exec \
    torchrun --nproc_per_node "$WORLD" --rdzv_backend c10d --rdzv_endpoint "localhost:2954$WORLD" \
    labs/lab04_ddp_vs_fsdp.py --layers "$LAYERS" --seq "$SEQ" --steps "$STEPS" \
    --tag "$tag" "$@" > "$log" 2>&1
  if grep -q "LAB04 OK" "$log"; then
    echo "    OK  $(grep -E 'agg tok/s' "$log" | tail -1)"
  else
    echo "    FAILED: $(grep -oE 'RuntimeError.*|Failed to build and load collectives.*|NotImplementedError.*|AssertionError.*' "$log" | head -1)"
  fi
}

run_one "fsdp_triage_A_fsdp2_hostcc"  1 --strategy fsdp2 --lora
run_one "fsdp_triage_B_fsdp1"         0 --strategy fsdp1 --lora
run_one "fsdp_triage_C_fsdp1_hostcc"  1 --strategy fsdp1 --lora

echo "=== triage done"
