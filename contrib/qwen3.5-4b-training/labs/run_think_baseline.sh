#!/usr/bin/env bash
# Qwen3.5-9B untuned, three prompt formats on the SAME 128-question stratified dev subset
# (seed 0), so every pair is row-aligned for eval_sql --compare --key exec_ok.
#   plain        the fine-tune's completion format, thinking bypassed (seq 2048, 128 new)
#   chat-nothink chat template + teacher instructions, enable_thinking=False
#   chat-think   chat template + teacher instructions, thinking on, 2048-token budget
# Runs ON THE BOX:  nohup labs/run_think_baseline.sh > ../runs/think_baseline.status 2>&1 &
# Precondition: lab10 --warmup-only at seq 2048 and at seq 4096 --warmup-max 2688 (9B).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SQLFT_MODEL=/work/models/Qwen3.5-9B
LIMIT=128 SEQ=2048 MAX_NEW=128  FORMAT=plain        "$HERE/run_bird_eval_sharded.sh" b128_9b_base_plain   || exit 1
LIMIT=128 SEQ=4096 MAX_NEW=256  FORMAT=chat-nothink "$HERE/run_bird_eval_sharded.sh" b128_9b_base_nothink || exit 1
LIMIT=128 SEQ=4096 MAX_NEW=2048 FORMAT=chat-think   "$HERE/run_bird_eval_sharded.sh" b128_9b_base_think   || exit 1
echo "THINK BASELINE DONE $(date -u +%H:%M)"
