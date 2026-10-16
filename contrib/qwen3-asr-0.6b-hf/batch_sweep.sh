#!/usr/bin/env bash
# Batch-size sweep for the Qwen3-ASR-0.6B static-shape port.
#
# This port's three compiled graphs (Encoder/Prefill/DecodeK) all have a hardcoded leading batch dim of 1 --
# there is no in-graph batching (see README "Known limits"). So "batch size B" here means "transcribe the
# first B utterances of librispeech_asr_dummy sequentially, in one process, with the existing batch=1 graphs
# reused across them" -- it measures how dataset-level throughput behaves given no batching dimension, not a
# scaling benefit from batching.
#
# Usage: ./batch_sweep.sh 1 2 4 8 16 32
# Needs ref/ref.json with at least max(sizes) samples: regenerate with `python cpu_ref.py 32` first.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
for n in "$@"; do
  echo "=== batch size (sequential utterances) = $n ==="
  QASR_RUNTAG="bs${n}" python run.py -n "$n" --repeats 3
done
