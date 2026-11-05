# Restoring the Pika-H3-Flex environment on a fresh trn2.48xlarge

Everything needed lives in `s3://pika-h3-transfer-933591302813-apse4/` and this repo
(branch `pika-h3-flex`). The S3 bucket policy grants the two instance roles; a fresh box
needs its role added (see the bucket policy) or use admin credentials.

## 1. Box

CloudFormation: `dev/trainium-model-dev/cloudformation/trn_capacity_block.yml`
(VolumeSize 200 -> 500 GB), region ap-south-2, AZ matching the reservation,
InstanceType trn2.48xlarge. Add your egress IP to the SG for SSH (the corp prefix list
does not exist in ap-south-2).

## 2. Data (s5cmd, ~10 min)

```bash
sudo mkdir -p /data/omni29 /data/models /data/nvme/minimax_h3 && sudo chown -R ubuntu:ubuntu /data
B=s3://pika-h3-transfer-933591302813-apse4
s5cmd --numworkers 64 sync "$B/MiniMax-H3/*" /data/models/MiniMax-H3/          # 144 GB
s5cmd --numworkers 64 sync "$B/Pika-H3-Flex/*" /data/models/Pika-H3-Flex/      # adapters + configs
s5cmd --numworkers 64 sync "$B/venv/*" /data/omni29/venv/                      # 12 GB
s5cmd --numworkers 64 sync "$B/plugin-48xl-patched/*" /data/omni29/plugin/     # fully patched plugin
s5cmd --numworkers 64 sync "$B/48xl/vllm-cache/*" /data/omni29/vllm-cache/     # graph caches (warm start)
s5cmd --numworkers 64 sync "$B/48xl/nki-cache/*" /data/omni29/nki-cache/
s5cmd --numworkers 64 sync "$B/48xl/fl2va/*" /data/omni29/fl2va/               # runners + keyframes
s5cmd cp "$B/48xl/pika_h3_flex_*.yaml" /data/omni29/
# s5cmd drops execute bits:
cd /data/omni29/venv && chmod +x bin/* || true

# Model symlink tree (shared MiniMax weights + Pika configs/adapters):
MM=/data/models/MiniMax-H3; PK=/data/models/Pika-H3-Flex; NE=/data/models/Pika-H3-Flex-Neuron
mkdir -p $NE
for f in LICENSE NOTICE README.md SHA256SUMS THIRD_PARTY_NOTICES.md; do ln -sfn $PK/$f $NE/$f; done
for d in audio_scheduler fl2va ref2va processor scheduler tokenizer; do ln -sfn $PK/$d $NE/$d; done
for d in audio_vae text_encoder transformer transformer_ref vae model_index.json; do ln -sfn $MM/$d $NE/$d; done
ln -sfn $PK/modular_model_index.json $NE/modular_model_index.json
```

## 3. Run

```bash
source /data/omni29/env29.sh
# T2V 6-step with audio:
/data/omni29/run29.sh t2v python /data/omni29/plugin/examples/minimax_h3/run.py \
  --model-path /data/models/Pika-H3-Flex-Neuron \
  --stage-config /data/omni29/pika_h3_flex_stage_tp8_enc.yaml \
  --num-inference-steps 6 --seed 42 --output out.mp4
# FL2VA: fl2va/run_fl2va.py --first-frame k.png ...
# Ref2VA/lip-sync: fl2va/run_ref2va.py --stage-config .../pika_h3_flex_ref2va_stage_tp8_768.yaml \
#   --image subject.png --audio-track voice.wav ...
```

The WAV soundtrack lands at `$MINIMAX_H3_OUT/request_NNN.wav` (= the run dir).

## 4. If the plugin ever needs re-patching from scratch

Apply, in order (all idempotent-guarded, anchor-based):
`apply_fl2va.py apply_audio_decode.py apply_step_count.py apply_ref2va.py
apply_device_vae_encode.py apply_devenc_fix.py apply_ref_short_edge.py
apply_encode_fallback.py apply_encode_broadcast.py apply_encode_paths.py`
