#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Bootstrap a fresh trn2.48xlarge (Neuron DLAMI, Ubuntu 24.04) for the Pika-H3-Flex /
# MiniMax-H3 vllm-omni 0.29 stack, mirroring the trn2.3xlarge's /data layout.
#
# What it does:
#   1. /data + /data/nvme dirs (host-weight maps on EBS; NVMe instance store optional)
#   2. rsync the venv (12 GB), plugin (3.2 GB) and omni29 support files from the 3xl
#   3. hf download MiniMax-H3 (135 GB, public) + Pika-H3-Flex small files (2.6 GB)
#   4. Rebuild the Pika-H3-Flex-Neuron symlink tree (MiniMax shared weights + Pika configs)
#
# Usage (from the workstation):
#   scp bootstrap_48xl.sh ubuntu@<new-box>: && ssh ubuntu@<new-box>
#   BOX3XL=ubuntu@<3xl-ip> bash bootstrap_48xl.sh
#
set -euo pipefail

BOX3XL=${BOX3XL:-ubuntu@16.51.211.117}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/trn2-ap-southeast-4-20261001}

echo "=== 1/4: dirs"
sudo mkdir -p /data/omni29 /data/models /data/nvme/minimax_h3
sudo chown -R ubuntu:ubuntu /data

echo "=== 2/4: rsync venv + plugin + support files from $BOX3XL (~15.5 GB)"
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes)
rsync -a --info=progress2 -e "${SSH[*]}" \
  "$BOX3XL:/data/omni29/venv" \
  "$BOX3XL:/data/omni29/plugin" \
  /data/omni29/
rsync -a -e "${SSH[*]}" \
  "$BOX3XL:/data/omni29/env29.sh" \
  "$BOX3XL:/data/omni29/run29.sh" \
  "$BOX3XL:/data/omni29/pika_h3_flex_stage.yaml" \
  /data/omni29/
# Point host_weights_dir at /data/nvme instead of /mnt/nvme in the copied configs.
sed -i 's|/mnt/nvme/minimax_h3|/data/nvme/minimax_h3|' /data/omni29/pika_h3_flex_stage.yaml

echo "=== 3/4: weights via HF (MiniMax-H3 135 GB + Pika-H3-Flex 2.6 GB, public)"
source /data/omni29/env29.sh
hf download MiniMaxAI/MiniMax-H3 --local-dir /data/models/MiniMax-H3
hf download Pika-Labs/Pika-H3-Flex --local-dir /data/models/Pika-H3-Flex \
  --exclude "transformer/*" "transformer_ref/*" "text_encoder/*" "vae/*" "audio_vae/*" "comfy/*"

echo "=== 4/4: Pika-H3-Flex-Neuron symlink tree"
MM=/data/models/MiniMax-H3
PK=/data/models/Pika-H3-Flex
NE=/data/models/Pika-H3-Flex-Neuron
mkdir -p "$NE"
for f in LICENSE NOTICE README.md SHA256SUMS THIRD_PARTY_NOTICES.md; do
  [ -e "$PK/$f" ] && ln -sfn "$PK/$f" "$NE/$f"
done
for d in audio_scheduler fl2va processor scheduler tokenizer; do
  [ -e "$PK/$d" ] && ln -sfn "$PK/$d" "$NE/$d"
done
for d in audio_vae text_encoder transformer vae model_index.json; do
  [ -e "$MM/$d" ] && ln -sfn "$MM/$d" "$NE/$d"
done
[ -e "$PK/modular_model_index.json" ] && ln -sfn "$PK/modular_model_index.json" "$NE/modular_model_index.json"

echo "=== done. Next:"
echo "  source /data/omni29/env29.sh"
echo "  # apply FL2VA patch to the plugin pipeline, then run TP=8:"
echo "  python /data/omni29/fl2va/apply_fl2va.py /data/omni29/plugin/vllm_omni_neuron/diffusion/models/minimax_h3/pipeline_minimax_h3.py"
