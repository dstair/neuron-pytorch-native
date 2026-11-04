#!/bin/bash
# run29.sh NAME CMD...   -- runs CMD in the plugin checkout under the 0.29 stack.
# Writes /data/omni29/runs/NAME/run.log, ending with "=== exit <code>" (what the watcher keys on).
set -u
NAME=$1; shift
source /data/omni29/env29.sh
OUT=$OMNI/runs/$NAME
mkdir -p "$OUT"
export MINIMAX_H3_OUT=$OUT
cd "$OMNI/plugin"
echo "=== $NAME $(date -u +%FT%TZ) cmd: $*" > "$OUT/run.log"
/usr/bin/time -v "$@" >> "$OUT/run.log" 2>&1
code=$?
grep -E "passed|failed|error" "$OUT/run.log" | grep -E "^(=+ |[0-9]+ (passed|failed))" | tail -2 > "$OUT/compare.txt"
sed -i "s/^/SUMMARY /" "$OUT/compare.txt"
echo "=== exit $code $(date -u +%FT%TZ)" >> "$OUT/run.log"
