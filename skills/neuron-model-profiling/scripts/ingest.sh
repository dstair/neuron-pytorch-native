#!/usr/bin/env bash
# Ingest one captured region to parquet for op/source-level attribution.
#   usage: ingest.sh <tag>
#   -> $W/data/profiles/global/<tag>@latest/*.parquet
#
# --ingest-only REQUIRES --display-name, else: "fatal: Missing --display-name".
# Ingest of a ~600MB NTFF is CPU-bound, ~1-2 min. Re-running against the same --data-path
# skips work already done.
set -uo pipefail
TAG="${1:?usage: ingest.sh <tag>}"
W=${W:-/mnt/nvme/lnc1-work/profile}
IMAGE=${IMAGE:-421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest}

docker run --rm --privileged -v /opt/aws/neuron/lib:/host_neuron_lib:ro \
  -v "$W":/work -w /work "$IMAGE" bash -lc "
    export LD_LIBRARY_PATH=/host_neuron_lib:\$LD_LIBRARY_PATH
    source /opt/torch-neuronx/.venv/bin/activate 2>/dev/null || true
    mkdir -p /work/data
    neuron-explorer view --ingest-only --data-path /work/data \
      --display-name ${TAG} \
      -n /work/${TAG}.neff -s /work/${TAG}_rank_0_exec_2.ntff
  " > "$W/ing_${TAG}.log" 2>&1
echo "$TAG ingest exit=$?"
ls "$W/data/profiles/global/${TAG}@latest/" 2>/dev/null | head -5
