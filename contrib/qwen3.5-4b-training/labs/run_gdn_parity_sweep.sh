#!/usr/bin/env bash
# The GDN kernel passes parity at (T=256, H=2) with cos=1.000000 but makes the 32-layer model
# produce a different loss than the reference (1.0299 vs 0.5031 on the same batch) and NaN on
# other batches. So the narrow parity test was not representative. Sweep toward the real
# configuration -- Qwen3.5-4B uses H=32 value heads, D=128, and we train at T=512 -- to find
# which axis breaks it.
#
#   labs/run_gdn_parity_sweep.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
LOGS="$(cd "$SRC/.." && pwd)/runs"
mkdir -p "$LOGS"

for cfg in "256 2" "256 8" "256 32" "512 2" "512 32" "1024 32"; do
  set -- $cfg
  SEQ=$1; H=$2
  LOG="$LOGS/gdn_parity_T${SEQ}_H${H}.log"
  printf "T=%-5s H=%-3s " "$SEQ" "$H"
  LNC=2 WORLD=1 "$SRC/env/container.sh" exec \
    python3 -m sqlft.gdn --parity --device neuron --seq "$SEQ" --heads "$H" \
    > "$LOG" 2>&1
  if grep -q "GDN PARITY OK" "$LOG"; then
    echo "OK    $(grep -E '^fwd ' "$LOG" | head -1)"
  else
    echo "FAIL  $(grep -E '^fwd |^loss |PARITY FAILED|Error' "$LOG" | head -2 | tr '\n' ' ')"
  fi
done
