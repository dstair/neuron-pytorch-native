#!/usr/bin/env bash
# Kill the prefill container if host MemAvailable falls toward the wedge point.
# 8,192 in-flight tokens at 40L took this host down on 2026-08-20 during the
# 8-rank parallel compile; a wedge costs a reboot, so trip early instead.
FLOOR_GB="${1:-10}"
# Per-invocation marker: a stale shared marker made an earlier run's watchdog
# exit immediately ("container gone") before the new container had started.
MARK="/tmp/watchdog_seen_$$"
trap 'rm -f "$MARK"' EXIT
BENCH_ENV_REQUIRE_IMAGE=0 . "$(dirname "$0")/bench_env.sh"   # $WORK
LOG=$WORK/logs/oom_watchdog.log
echo "watchdog start $(date -u +%FT%TZ) floor=${FLOOR_GB}GB" >> "$LOG"
while true; do
  AVAIL=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
  NAMES=$(docker ps --format '{{.Names}}' | grep '^q35-prefill-fp8bs-' || true)
  if [ -z "$NAMES" ]; then
    if [ -f $MARK ]; then
      echo "container gone $(date -u +%FT%TZ) avail=${AVAIL}GB - exiting" >> "$LOG"; exit 0
    fi
  else
    touch $MARK
    echo "$(date -u +%FT%TZ) avail=${AVAIL}GB load=$(cut -d' ' -f1 /proc/loadavg)" >> "$LOG"
    if [ "$AVAIL" -lt "$FLOOR_GB" ]; then
      echo "TRIPPED $(date -u +%FT%TZ) avail=${AVAIL}GB < ${FLOOR_GB}GB -> killing $NAMES" >> "$LOG"
      for n in $NAMES; do docker kill "$n" >> "$LOG" 2>&1; done
      exit 1
    fi
  fi
  sleep 10
done
