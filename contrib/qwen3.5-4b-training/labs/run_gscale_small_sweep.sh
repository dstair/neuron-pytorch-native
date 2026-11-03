#!/usr/bin/env bash
# The earlier g_scale sweep went the WRONG WAY. `g` is log-decay, so more-negative g means
# MORE decay, hence a smaller intra-chunk matrix A and a BETTER-conditioned inverse. Sweeping
# g_scale up (1 -> 256) therefore made the kernel more accurate, which is why it never broke.
#
# Real Qwen3.5-4B g has mean -0.19 -- weakly damped, i.e. LESS damped than even g_scale=1
# (mean -0.5). So the failure regime is g_scale << 1. If upstream breaks here on purely
# synthetic inputs, that is both a confirmation of the mechanism and a reproducer that needs
# no model weights at all.
#
#   labs/run_gscale_small_sweep.sh [seq] [heads]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
LOGS="$(cd "$SRC/.." && pwd)/runs"
mkdir -p "$LOGS"
SEQ="${1:-512}"; H="${2:-32}"

printf "%-12s %-24s %s\n" "g_scale" "mean g (synthetic)" "result"
for GS in 0.02 0.05 0.1 0.25 0.5 1; do
  LOG="$LOGS/gscale_small_${GS}.log"
  MEAN=$(python3 -c "print(f'{-0.5*$GS:.3f}')")
  printf "%-12s %-24s " "$GS" "$MEAN"
  LNC=2 WORLD=1 "$SRC/env/container.sh" exec \
    python3 -m sqlft.gdn --parity --device neuron --seq "$SEQ" --heads "$H" --g-scale "$GS" \
    > "$LOG" 2>&1
  if grep -q "GDN PARITY OK" "$LOG"; then
    echo "OK        $(grep -E '^fwd ' "$LOG" | head -1)"
  else
    echo "DIVERGES  $(grep -E '^fwd |^loss ' "$LOG" | head -2 | tr '\n' ' ')"
  fi
done
echo
echo "(real model g has mean -0.19, i.e. between g_scale 0.25 and 0.5)"
