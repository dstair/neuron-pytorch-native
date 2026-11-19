#!/usr/bin/env bash
# One-time (idempotent) setup on the box. Runs ON THE BOX.
#
#   ./setup_box.sh all        # image + weights + data + kernel
#   ./setup_box.sh image      # derived DLC image with transformers/datasets/duckdb/kernels
#   ./setup_box.sh weights    # Qwen/Qwen3.5-4B -> /work/models/Qwen3.5-4B  (~8.4 GB bf16)
#   ./setup_box.sh data       # motherduckdb/duckdb-text2sql-25k -> /work/data  (~36 MB)
#   ./setup_box.sh kernel     # the trainable GDN NKI kernels from the HF kernels hub
#   ./setup_box.sh check      # assert everything imports and the model class exists
#
# Everything lands under /mnt/nvme (instance store). It survives a REBOOT but a
# stop/start destroys it -- this script is the recovery path, so keep it runnable.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/load_env.sh"
CONTAINER="$HERE/container.sh"
WORK_ROOT="${WORK_ROOT:-$(cd "$HERE/../.." && pwd)}"

MODEL_ID="Qwen/Qwen3.5-4B"
DATASET_ID="motherduckdb/duckdb-text2sql-25k"
# Two pinned kernel builds (see sqlft/gdn.py KERNEL_ROOTS):
#   v15  -- `--gdn nki-v15`, and the base of `nki-v15-split16`, the default. There is no v15
#           tag, so it is pinned by commit.
#   v3.0 -- `--gdn nki` / `nki-patched`, kept to reproduce the original numerics failure.
# The repo was renamed from qwen35-deltanet-tkg-full; the old name still redirects.
KERNEL_ID="kernels/jburtoft/qwen35-gdn-deltanet-neuron-kernels"
KERNEL_REV="${KERNEL_REV:-36d9feb87ad76e3ee2c8c7674c97219625f682b1}"
KERNEL_LEGACY_REV="${KERNEL_LEGACY_REV:-v3.0}"

mkdir -p "$WORK_ROOT"/{models,data,runs,hf,cache}

step_image() {
  # STACK=public (default) or beta6; container.sh resolves the same names.
  local stack="${STACK:-public}" base
  case "$stack" in
    public) base="public.ecr.aws/neuron/torch-neuronx@sha256:b4648437b8c7dbfa2acde18301a4bb4ae9658da5a412da75c59693ed4fd308ec" ;;
    beta6)  require_env QWEN35_TRAIN_BETA6_IMAGE "pre-GA DLC reference, pinned by digest"
            base="$QWEN35_TRAIN_BETA6_IMAGE" ;;
    *) echo "unknown STACK=$stack (public|beta6)" >&2; exit 2 ;;
  esac
  sudo docker build \
    --build-arg BASE="$base" \
    -t "${TRAIN_IMAGE:-qwen35-train:$stack}" \
    -f "$HERE/Dockerfile" "$HERE"
}

step_weights() {
  "$CONTAINER" bash "hf download '$MODEL_ID' --local-dir /work/models/Qwen3.5-4B
    du -sh /work/models/Qwen3.5-4B"
}

step_data() {
  "$CONTAINER" bash "hf download '$DATASET_ID' --repo-type dataset \
      --local-dir /work/data/duckdb-text2sql-25k
    find /work/data/duckdb-text2sql-25k -type f | head"
}

step_kernel() {
  # Fetch the repo explicitly rather than relying on get_kernel()'s implicit download, so
  # the labs can run with HF_HUB_OFFLINE=1 and so the revision is pinned and auditable.
  "$CONTAINER" bash "hf download '$KERNEL_ID' --revision '$KERNEL_REV' \
      --local-dir /work/kernels/qwen35-deltanet-v15
    hf download '$KERNEL_ID' --revision '$KERNEL_LEGACY_REV' \
      --local-dir /work/kernels/qwen35-deltanet
    find /work/kernels/qwen35-deltanet-v15/build -name '*.py' | head -20"
  # The default build is derived from v15, so it has to exist before anything trains.
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$CONTAINER" exec python3 -m sqlft.kernel_patch.apply_v15_split16
}

step_check() {
  "$CONTAINER" exec python3 -c '
import pathlib, torch, transformers
from transformers import AutoConfig

model_dir = "/work/models/Qwen3.5-4B"
cfg = AutoConfig.from_pretrained(model_dir)
text = getattr(cfg, "text_config", cfg)
print("arch        :", cfg.architectures)
print("layers      :", text.num_hidden_layers)
print("hidden      :", text.hidden_size)
print("vocab       :", text.vocab_size)
print("layer_types :", sorted(set(getattr(text, "layer_types", []) or [])))

data = pathlib.Path("/work/data/duckdb-text2sql-25k")
print("dataset     :", sorted(p.name for p in data.rglob("*") if p.is_file())[:5])

kern = pathlib.Path("/work/kernels/qwen35-deltanet")
print("kernel files:", sorted(p.name for p in kern.rglob("*.py"))[:8])
print("SETUP OK")
'
}

case "${1:-all}" in
  image) step_image ;;
  weights) step_weights ;;
  data) step_data ;;
  kernel) step_kernel ;;
  check) step_check ;;
  all) step_image; step_weights; step_data; step_kernel; step_check ;;
  *) echo "usage: $0 {all|image|weights|data|kernel|check}" >&2; exit 2 ;;
esac
