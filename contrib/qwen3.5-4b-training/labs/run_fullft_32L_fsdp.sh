#!/usr/bin/env bash
# The other half of the capability claim: full fine-tune of ALL 4.206 B params, full 32
# layers, under FSDP2 at world=4. Single-core OOMs at 24 GiB (see run_fullft_capability.sh);
# sharding the AdamW moments across 4 ranks is what makes this fit.
#
#   labs/run_fullft_32L_fsdp.sh [world] [seq] [steps]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
LOGS="$(cd "$SRC/.." && pwd)/runs"
mkdir -p "$LOGS"

WORLD="${1:-4}"; SEQ="${2:-512}"; STEPS="${3:-4}"
LNC=2; [ "$WORLD" -gt 4 ] && LNC=1
TAG="lab04_fullft_fsdp2_w${WORLD}_32L"
LOG="$LOGS/$TAG.log"

echo "=== $TAG (world=$WORLD LNC=$LNC HOST_CC=1 seq=$SEQ) -> $LOG"
# HOST_CC=1 is MANDATORY for any FSDP on beta6: without it the first forward dies with
# "Failed to build and load collectives resources for module.neff". See NOTES.md.
LNC="$LNC" WORLD="$WORLD" HOST_CC=1 "$SRC/env/container.sh" exec \
  torchrun --nproc_per_node "$WORLD" --rdzv_backend c10d --rdzv_endpoint localhost:29557 \
  labs/lab04_ddp_vs_fsdp.py --strategy fsdp2 --full-ft --layers 32 \
  --seq "$SEQ" --steps "$STEPS" --tag "$TAG" > "$LOG" 2>&1

if grep -q "LAB04 OK" "$LOG"; then
  echo "    OK  $(grep -E 'agg tok/s' "$LOG" | tail -1)"
  grep -E "^loss " "$LOG" | tail -1 | sed 's/^/    /'
else
  echo "    FAILED: $(grep -oE 'RuntimeError.*|Failed to allocate DEVICE memory.*|Failed to build and load collectives.*' "$LOG" | head -1)"
fi
