#!/usr/bin/env bash

# Print a compiler-cache summary and optional cache-hit evidence from a run log.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'EOF'
Usage: inspect.sh [cache-directory] [run-log]

The optional run log is scanned for common persistent-cache hit and miss
messages. A log without either marker is inconclusive; verify that no
neuronx-cc backend processes were launched during the run.
EOF
}

[[ $# -le 2 ]] || {
  usage >&2
  exit 2
}

cache_dir="$(resolve_cache_dir "${1:-}")"
assert_cache_root "$cache_dir"

echo "cache root: $cache_dir"
echo "size: $(du -sh "$cache_dir" | awk '{print $1}')"
echo "files: $(cache_file_count "$cache_dir")"
echo "NEFFs: $(cache_neff_count "$cache_dir")"
for subtree in hlo_cache neff_cache local_cache neuron_backend; do
  if [[ -d "$cache_dir/$subtree" ]]; then
    echo "$subtree: $(du -sh "$cache_dir/$subtree" | awk '{print $1}')"
  fi
done

if [[ $# -eq 2 ]]; then
  run_log="$2"
  [[ -f "$run_log" ]] || die "run log does not exist: $run_log"

  hit_count="$(grep -Eic 'using a cached neff|using cached neff|cache hit|cache_hit' "$run_log" || true)"
  miss_count="$(grep -Eic 'no candidate found|cache miss|cache_miss' "$run_log" || true)"
  compile_count="$(grep -Eic 'neuronx-cc compile|walrus_driver' "$run_log" || true)"

  echo "log cache hits: $hit_count"
  echo "log cache misses: $miss_count"
  echo "log backend compiler markers: $compile_count"
  if [[ "$hit_count" -eq 0 && "$miss_count" -eq 0 && "$compile_count" -eq 0 ]]; then
    echo "log diagnosis: no recognized cache marker; corroborate with process monitoring"
  fi
fi
