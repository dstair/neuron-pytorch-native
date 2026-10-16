#!/bin/bash
# sweep.sh [bs ...] -- Whisper batch-size sweep, one process per BS (each compiles its own graphs).
# Summary lines -> $ASR_WORK/sweep.log, per-BS logs -> $ASR_WORK/sweep_bs<B>.log, JSON -> $ASR_WORK/ref/.
cd "$(dirname "${BASH_SOURCE[0]}")" && source ./env.sh
for b in ${@:-1 2 4 8 16 32}; do
  python whisper_batch.py --bs "$b" > "$ASR_WORK/sweep_bs$b.log" 2>&1
  grep -E "^SWEEP" "$ASR_WORK/sweep_bs$b.log" >> "$ASR_WORK/sweep.log" \
    || echo "bs=$b FAILED: $(grep -E 'Error|error' "$ASR_WORK/sweep_bs$b.log" | tail -1 | cut -c1-200)" >> "$ASR_WORK/sweep.log"
done
echo SWEEP_DONE >> "$ASR_WORK/sweep.log"
