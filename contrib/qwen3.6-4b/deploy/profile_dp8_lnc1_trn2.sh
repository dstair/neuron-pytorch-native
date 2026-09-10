#!/usr/bin/env bash
set -euo pipefail

# Capture one representative graph from the DP=8/LNC=1 winner.  The graph is
# replayed alone on core 0 so profiling does not perturb the other replicas.
# Consequently, this captures kernel behavior but not shared-HBM contention.

runtime_template="${QWEN4B_RUNTIME_TEMPLATE:?set QWEN4B_RUNTIME_TEMPLATE}"
profile_root="${QWEN4B_PROFILE_ROOT:?set QWEN4B_PROFILE_ROOT}"
profile_name="${QWEN4B_PROFILE_NAME:-qwen4b-dp8-lnc1}"
profile_exec="${QWEN4B_PROFILE_EXEC:-2}"
profile_data_path="${QWEN4B_PROFILE_DATA_PATH:-$profile_root/ne-data}"
explorer="${QWEN4B_NEURON_EXPLORER:-/opt/aws/neuron/bin/neuron-explorer}"
profile_python="${QWEN4B_PROFILE_PYTHON:-/mnt/nvme/profile-venv/bin/python}"
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="${QWEN4B_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="$profile_root/$run_id"
neff="$run_dir/fullgraph.neff"
session_file="$run_dir/fullgraph.ntff"
display_name="$profile_name-$run_id"

[[ -d "$runtime_template" ]] || {
  echo "error: runtime template does not exist: $runtime_template" >&2
  exit 2
}
[[ -x "$explorer" ]] || {
  echo "error: neuron-explorer is not executable: $explorer" >&2
  exit 2
}
[[ -x "$profile_python" ]] || {
  echo "error: profile Python is not executable: $profile_python" >&2
  exit 2
}
[[ ! -e "$run_dir" ]] || {
  echo "error: run directory already exists: $run_dir" >&2
  exit 2
}

source_neff="$(
  sudo find "$runtime_template" -type f -name '*.neff' -printf '%s %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
)"
[[ -n "$source_neff" ]] || {
  echo "error: no NEFF found under $runtime_template" >&2
  exit 2
}

mkdir -p "$run_dir" "$profile_data_path"
sudo cp "$source_neff" "$neff"
sudo chown "$(id -u):$(id -g)" "$neff"

echo "capturing representative NEFF: $source_neff"
NEURON_LOGICAL_NC_CONFIG=1 \
NEURON_RT_VISIBLE_CORES=0 \
NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1 \
  "$explorer" capture \
    -n "$neff" \
    -s "$session_file" \
    --profile-nth-exec="$profile_exec" \
    --enable-dge-notifs

ntff="${session_file%.ntff}_exec_${profile_exec}.ntff"
[[ -s "$ntff" ]] || ntff="$session_file"
[[ -s "$ntff" ]] || {
  echo "error: capture did not produce an NTFF under $run_dir" >&2
  exit 1
}

"$explorer" view \
  --output-format summary-json \
  -n "$neff" \
  -s "$ntff" \
  >"$run_dir/metrics.json"

"$explorer" view \
  -n "$neff" \
  -s "$ntff" \
  --data-path "$profile_data_path" \
  --display-name "$display_name" \
  --ingest-only \
  >"$run_dir/ingest.log" 2>&1

parquet_dir="$profile_data_path/profiles/global/$display_name@latest"
[[ -d "$parquet_dir" ]] || {
  echo "error: ingested parquet profile not found: $parquet_dir" >&2
  exit 1
}

"$profile_python" "$source_dir/analyze_profile.py" \
  "$parquet_dir" \
  --output-json "$run_dir/analysis.json" \
  >"$run_dir/analysis.stdout"

echo "capture: $ntff"
echo "summary: $run_dir/metrics.json"
echo "analysis: $run_dir/analysis.json"
echo "parquet: $parquet_dir"
