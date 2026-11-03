#!/usr/bin/env bash
# Every gate, in order of cost. Run this after changing anything in sqlft/.
# Each check gates on its own success marker, never on an exit code.
#
#   labs/run_gates.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
LOGS="$(cd "$SRC/.." && pwd)/runs"
mkdir -p "$LOGS"
FAILED=0

check() {
  local name="$1" marker="$2"; shift 2
  local log="$LOGS/gate_$name.log"
  printf "%-22s " "$name"
  "$@" > "$log" 2>&1
  if grep -q "$marker" "$log"; then
    echo "PASS"
  else
    echo "FAIL  (see $log)"
    tail -4 "$log" | sed 's/^/    /'
    FAILED=1
  fi
}

# CPU-only first: no device needed, so they cannot be blocked by a busy chip.
cpu() { TORCH_DEVICE_BACKEND_AUTOLOAD=0 LNC=2 WORLD=1 "$SRC/env/container.sh" exec "$@"; }
dev() { LNC=2 WORLD=1 "$SRC/env/container.sh" exec "$@"; }

check lora_tests  "passed"            cpu python3 -m pytest sqlft/tests -q
check data        "DATA SELF-TEST OK" cpu python3 -m sqlft.data --self-test
check eval_sql    "EVAL SELF-TEST OK" cpu python3 -m sqlft.eval_sql --self-test
check ckpt        "CKPT SELF-TEST OK" cpu python3 -m sqlft.ckpt --self-test

# The kernel patch must apply cleanly against the pinned upstream, and the CPU-side
# stability argument must hold, before any patched-kernel number is trustworthy.
check kernel_patch "patched kernel written" cpu python3 -m sqlft.kernel_patch.apply_blockdiag_fix
check split16_patch "patched kernel written" cpu python3 -m sqlft.kernel_patch.apply_v15_split16
check neumann      "NEUMANN STABILITY OK"   cpu python3 labs/lab03_neumann_stability.py

check device      "SMOKE OK"          dev python3 labs/lab_smoke.py
check gdn_parity  "GDN PARITY OK"     dev python3 -m sqlft.gdn --parity --device neuron --seq 256 --heads 2
# Layer 19's captured inputs are the ones that separate the builds: v3.0 is wrong outright and
# plain v15 loses 6-7% on dv/dbeta, while the gated builds match to 1e-4. A benign capture
# (e.g. call 5 on this box's data order) passes every build and proves nothing.
check gdn_patched "PATCHED KERNEL PARITY OK" dev python3 labs/lab03_patched_kernel_parity.py --grads --repro /work/runs/gdn_repro_call19.pt
check train_4L    "TRAIN OK"          dev python3 -m sqlft.train --layers 4 --seq 512 --bs 1 --steps 6 --lora --tag gate_train_4L
check train_32L_split16 "TRAIN OK"    dev python3 -m sqlft.train --layers 32 --seq 512 --bs 1 --steps 6 --lora --gdn nki-v15-split16 --tag gate_train_32L_split16
# The fast decode must stay logit-equivalent to the full-width one, or every eval number moves.
check eval_fast   "EVAL SPEED OK"     dev python3 labs/lab08_eval_speed.py --limit 1 --logit-steps 6 --skip-slow
# The cached recurrent decode is a second implementation of the model; it must stay
# logit-equivalent to the re-forward path under teacher forcing (top-1 every step).
check eval_cached "LAB09 OK"          dev python3 labs/lab09_cached_decode_parity.py --rows 4 --steps 8

check collectives_w4 "COLLECTIVES OK world=4" \
  env LNC=2 WORLD=4 "$SRC/env/container.sh" exec \
  torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:29581 \
  labs/lab_collectives.py

echo
[ "$FAILED" -eq 0 ] && echo "ALL GATES PASS" || echo "SOME GATES FAILED"
exit "$FAILED"
