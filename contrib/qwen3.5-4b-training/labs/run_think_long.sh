#!/usr/bin/env bash
# Qwen3.5-9B untuned, thinking ON with the largest budget that fits one core: seq 4096,
# max_new 3968 (cached_generate clamps each row to seq - prompt_bucket, i.e. ~1.4k-3.8k
# tokens). seq 8192 OOMs in single-core prefill attention at ~2.2k-token prompts; a
# 2048 budget truncated every early row. 32-question stratified subset (8 per core).
# Always ends with "THINK BASELINE END" so run_trace_ft.sh (waiting on it) proceeds.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"; RUNS="$(cd "$SRC/.." && pwd)/runs"
trap 'echo "THINK BASELINE END $(date -u +%H:%M)"' EXIT
export SQLFT_MODEL=/work/models/Qwen3.5-9B
LIMIT=32 SEQ=4096 MAX_NEW=3968 FORMAT=chat-think "$HERE/run_bird_eval_sharded.sh" b32_9b_base_think4k
