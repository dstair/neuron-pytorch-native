#!/usr/bin/env bash
# Host-RAM watchdog for the trn1 cross-compile.
#
# Why this is not optional: static_decode_35b.py DOES carry per-rank walrus compile
# staggering (CROSS_TARGET_COMPILE_CONCURRENCY), but that code sits AFTER the
# --prefill-bench early return, so a PREFILL cross-compile runs all 8 ranks' walrus
# drivers concurrently with no throttle. Eight co-resident ranks of the tiled O2
# decode graph once peaked at ~493 GB and wedged this 512 GB box (SSM ConnectionLost,
# recoverable only by reboot). A wedged host is much more expensive than a killed
# compile, so trade the compile away.
#
# Stops the container (does not kill -9 the host's processes) when available RAM falls
# below THRESHOLD_GB, then leaves the log for diagnosis. Retry at higher --prefill-splits:
# more splits = smaller per-segment graph = less walrus RSS per rank.
#
# usage: xc_oom_watchdog.sh <container-name> [threshold_gb] [interval_s]
set -uo pipefail
NAME="${1:?usage: xc_oom_watchdog.sh <container-name> [threshold_gb] [interval_s]}"
THRESHOLD_GB="${2:-45}"
INTERVAL="${3:-20}"
BENCH_ENV_REQUIRE_IMAGE=0 . "$(dirname "$0")/bench_env.sh"   # $WORK
LOG=$WORK/logs/xc_oom_watchdog_${NAME}.log

echo "watchdog START $(date -u +%FT%TZ) container=$NAME threshold=${THRESHOLD_GB}GB" >>"$LOG"
while true; do
  # Wait for the container to appear, and exit once it is gone (compile finished).
  if ! docker inspect --format '{{.State.Running}}' "$NAME" >/dev/null 2>&1; then
    if [[ -f /tmp/wd-seen-$NAME ]]; then
      echo "watchdog STOP $(date -u +%FT%TZ) container gone" >>"$LOG"; rm -f /tmp/wd-seen-$NAME; exit 0
    fi
    sleep "$INTERVAL"; continue
  fi
  touch /tmp/wd-seen-$NAME
  AVAIL=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  # `pgrep -c` already prints 0 on no-match AND exits 1, so a `|| echo 0` fallback
  # appends a SECOND 0 and puts a newline inside the variable, splitting the log line.
  NCC=$(pgrep -c -f 'neuronx-cc|walrus_driver' 2>/dev/null | head -1)
  echo "$(date -u +%FT%TZ) avail=${AVAIL}GB cc_procs=${NCC}" >>"$LOG"
  if (( AVAIL < THRESHOLD_GB )); then
    echo "$(date -u +%FT%TZ) THRESHOLD BREACH avail=${AVAIL}GB -> stopping $NAME" >>"$LOG"
    ps -eo pid,rss,cmd --sort=-rss | head -15 >>"$LOG"
    docker stop -t 5 "$NAME" >>"$LOG" 2>&1
    echo "watchdog STOPPED CONTAINER $(date -u +%FT%TZ) -- retry at higher --prefill-splits" >>"$LOG"
    exit 2
  fi
  sleep "$INTERVAL"
done
