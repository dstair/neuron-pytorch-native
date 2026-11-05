#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Restore the Pika-H3-Flex environment from the S3 mirror (see RESTORE.md).
# Runs on the fresh trn2.48xlarge. s5cmd must be installed first.
set -euo pipefail
B=s3://pika-h3-transfer-933591302813-apse4

echo "=== dirs"
sudo mkdir -p /data/omni29 /data/models /data/nvme/minimax_h3
sudo chown -R ubuntu:ubuntu /data

echo "=== venv (12 GB)"
s5cmd --numworkers 64 sync "$B/venv/*" /data/omni29/venv/
echo "=== patched plugin"
s5cmd --numworkers 64 sync "$B/plugin-48xl-patched/*" /data/omni29/plugin/
echo "=== graph caches"
s5cmd --numworkers 64 sync "$B/48xl/vllm-cache/*" /data/omni29/vllm-cache/
s5cmd --numworkers 64 sync "$B/48xl/nki-cache/*" /data/omni29/nki-cache/
echo "=== fl2va + configs"
s5cmd --numworkers 64 sync "$B/48xl/fl2va/*" /data/omni29/fl2va/
s5cmd --numworkers 16 sync "/data/omni29/nonexistent-prefix" /tmp/ 2>/dev/null || true
for f in env29.sh run29.sh pika_h3_flex_stage_tp8.yaml pika_h3_flex_stage_tp8_enc.yaml pika_h3_flex_stage_tp8cp2.yaml pika_h3_flex_ref2va_stage_tp8.yaml pika_h3_flex_ref2va_stage_tp8_768.yaml; do
  s5cmd cp "$B/48xl/$f" "/data/omni29/$f" || true
done

echo "=== weights (146 GB)"
s5cmd --numworkers 64 sync "$B/MiniMax-H3/*" /data/models/MiniMax-H3/
s5cmd --numworkers 64 sync "$B/Pika-H3-Flex/*" /data/models/Pika-H3-Flex/

echo "=== execute bits (s5cmd drops them)"
cd /data/omni29/venv && chmod +x bin/* || true

echo "=== symlink tree"
MM=/data/models/MiniMax-H3; PK=/data/models/Pika-H3-Flex; NE=/data/models/Pika-H3-Flex-Neuron
mkdir -p "$NE"
for f in LICENSE NOTICE README.md SHA256SUMS THIRD_PARTY_NOTICES.md; do ln -sfn "$PK/$f" "$NE/$f"; done
for d in audio_scheduler fl2va ref2va processor scheduler tokenizer; do ln -sfn "$PK/$d" "$NE/$d"; done
for d in audio_vae text_encoder transformer transformer_ref vae model_index.json; do ln -sfn "$MM/$d" "$NE/$d"; done
ln -sfn "$PK/modular_model_index.json" "$NE/modular_model_index.json"

echo "=== env + scripts"
s5cmd cp "$B/48xl/env29.sh" /data/omni29/env29.sh || true
s5cmd cp "$B/48xl/run29.sh" /data/omni29/run29.sh || true
chmod +x /data/omni29/run29.sh || true

echo "RESTORE_DONE"
