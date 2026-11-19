#!/bin/bash
# ablate.sh -- decoder-step throughput with round-2 choices switched one at a time (ASR_ABLATE switches in
# whisper_static.py: index = scatter cache write, nofuse = separate q/k/v, noargmax = return logits).
# Results -> $ASR_WORK/ablate.log
cd "$(dirname "${BASH_SOURCE[0]}")" && source ./env.sh
for a in "" index nofuse noargmax; do
  r=$(ASR_ABLATE=$a python whisper_step_timing.py 2>&1 | grep -E "^A |Error" | tail -1)
  echo "ABLATE=[${a}] ${r}" >> "$ASR_WORK/ablate.log"
done
echo ABLATE_DONE >> "$ASR_WORK/ablate.log"
