#!/usr/bin/env bash
# Talk to the Trainium box. The IP is always rediscovered from the instance ID --
# a stop/start changes it, and a noted address goes stale silently.
#
#   ./remote.sh status          # instance state, IP, neuron-ls, disk
#   ./remote.sh ssh [cmd...]    # run a command on the host (no args = interactive)
#   ./remote.sh push            # rsync this package to $REMOTE_ROOT on the box
#   ./remote.sh pull            # rsync runs/ back down
#   ./remote.sh ip              # print just the IP
set -euo pipefail

# Instance, region and key are deployment state: set QWEN35_TRAIN_INSTANCE_ID,
# QWEN35_TRAIN_REGION and QWEN35_TRAIN_SSH_KEY in the repo-root .env (see .env.example).
# REMOTE_ROOT defaults to /mnt/nvme/qwen35-training; override it on a box where that
# path is taken.
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$PKG_DIR/env/load_env.sh"
require_env QWEN35_TRAIN_INSTANCE_ID "EC2 instance ID of the Trainium box"
require_env QWEN35_TRAIN_REGION "AWS region of that instance"
require_env QWEN35_TRAIN_SSH_KEY "path to the SSH private key for the box"
INSTANCE_ID="$QWEN35_TRAIN_INSTANCE_ID"
REGION="$QWEN35_TRAIN_REGION"
SSH_KEY="$QWEN35_TRAIN_SSH_KEY"
SSH_USER="${QWEN35_TRAIN_SSH_USER:-ubuntu}"
# /mnt/nvme is instance store. Reboot preserves it; stop/start DESTROYS it.
REMOTE_ROOT="${QWEN35_TRAIN_REMOTE_ROOT:-/mnt/nvme/qwen35-training}"

SSH_OPTS=(-i "$SSH_KEY" -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new
          -o ServerAliveInterval=30 -o LogLevel=ERROR)

describe() {
  aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" --output json
}

instance_ip() {
  describe | python3 -c '
import json, sys
i = json.load(sys.stdin)["Reservations"][0]["Instances"][0]
state = i["State"]["Name"]
ip = i.get("PublicIpAddress")
if state != "running":
    sys.exit(f"instance is {state}, not running")
if not ip:
    sys.exit("instance has no public IP")
print(ip)
'
}

do_ssh() {
  local ip; ip="$(instance_ip)"
  if [[ $# -eq 0 ]]; then
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip"
  else
    # Strip the interactive-shell noise this box's rc files emit on every login.
    # -n: a backgrounded remote job otherwise holds the channel open and the call hangs.
    ssh -n "${SSH_OPTS[@]}" "$SSH_USER@$ip" "$@" 2>&1 \
      | grep -v 'Q_PARENT env variable not set' || true
  fi
}

case "${1:-status}" in
  ip) instance_ip ;;

  status)
    describe | python3 -c '
import json, sys
i = json.load(sys.stdin)["Reservations"][0]["Instances"][0]
print("instance :", i["InstanceId"], i["InstanceType"], i["State"]["Name"])
print("ip       :", i.get("PublicIpAddress", "-"))
print("launched :", i["LaunchTime"])
'
    echo
    do_ssh 'neuron-ls; echo; df -h $REMOTE_ROOT / | tail -3; echo; free -g | head -2'
    ;;

  ssh) shift; do_ssh "$@" ;;

  push)
    ip="$(instance_ip)"
    do_ssh "mkdir -p $REMOTE_ROOT"
    rsync -az --delete --info=stats1 \
      -e "ssh ${SSH_OPTS[*]}" \
      --exclude runs/ --exclude __pycache__/ --exclude '*.pyc' \
      --exclude .pytest_cache/ \
      "$PKG_DIR/" "$SSH_USER@$ip:$REMOTE_ROOT/src/"
    # rsync --delete recreates the scripts without the exec bit; restore it every push.
    # find, not a glob: a glob that matches nothing makes chmod fail and take the rest of
    # the argument list with it, which silently left new lab runners non-executable.
    do_ssh "find $REMOTE_ROOT/src -name '*.sh' -exec chmod +x {} +"
    echo "pushed -> $REMOTE_ROOT/src/"
    ;;

  pull)
    ip="$(instance_ip)"
    mkdir -p "$PKG_DIR/runs"
    # Results land in $REMOTE_ROOT/runs (mounted as /work/runs inside the container), NOT
    # under src/ -- so they survive `push`, which uses --delete on src/.
    #
    # --update: never let an older box-side file overwrite a newer local one (a box that
    # predates the local runs/ would otherwise clobber them by name).
    #
    # Skip *.pt by default: LoRA adapters are 130 MB each and checkpoints accumulate, so a
    # plain pull drags ~800 MB of weights across for the sake of a few KB of json. Set
    # PULL_WEIGHTS=1 when you actually need an adapter locally.
    weight_filter=(--exclude '*.pt')
    [[ "${PULL_WEIGHTS:-0}" == 1 ]] && weight_filter=()
    rsync -az --update --info=stats1 "${weight_filter[@]}" -e "ssh ${SSH_OPTS[*]}" \
      "$SSH_USER@$ip:$REMOTE_ROOT/runs/" "$PKG_DIR/runs/"
    echo "pulled -> $PKG_DIR/runs/"
    ;;

  *) echo "usage: $0 {status|ip|ssh [cmd...]|push|pull}" >&2; exit 2 ;;
esac
