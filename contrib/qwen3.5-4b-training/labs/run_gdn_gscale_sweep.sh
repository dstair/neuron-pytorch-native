#!/usr/bin/env bash
# Shape is not the problem -- parity holds at every (T, H) up to T=1024/H=32. So sweep the
# INPUT DISTRIBUTION instead. `g` is the raw log-decay; the real model computes
#     g = -A_log.exp() * softplus(a + dt_bias),   A in [0.01, 16]
# which reaches far below the -1..0 that `-rand()` gives. Chunked gated-delta-rule maths forms
# exp(g_i - g_j) terms, so once consecutive g are very negative one direction of that
# exponential overflows. This finds the magnitude at which the kernel departs from the
# reference, which is the minimal reproducer for the 32-layer NaN.
#
#   labs/run_gdn_gscale_sweep.sh [seq] [heads]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
LOGS="$(cd "$SRC/.." && pwd)/runs"
mkdir -p "$LOGS"
SEQ="${1:-512}"; H="${2:-32}"

for GS in 1 4 16 64 256; do
  LOG="$LOGS/gdn_gscale_${GS}.log"
  printf "g_scale=%-5s " "$GS"
  LNC=2 WORLD=1 "$SRC/env/container.sh" exec \
    python3 -m sqlft.gdn --parity --device neuron --seq "$SEQ" --heads "$H" --g-scale "$GS" \
    > "$LOG" 2>&1
  if grep -q "GDN PARITY OK" "$LOG"; then
    echo "OK     $(grep -E '^fwd ' "$LOG" | head -1)"
  else
    echo "DIVERGES  $(grep -E '^loss |^fwd ' "$LOG" | head -2 | tr '\n' ' | ')"
  fi
done
