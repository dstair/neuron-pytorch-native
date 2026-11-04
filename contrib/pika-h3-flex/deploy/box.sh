#!/bin/bash
# box.sh run '<cmd>'        run a command on the trn2.3xl
# box.sh put LOCAL REMOTE   copy a file over stdin
HOST=ubuntu@16.51.211.117
SSH=(ssh -F "$HOME/.ssh/config" -i "$HOME/.ssh/trn2-ap-southeast-4-20261001" -o IdentitiesOnly=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 "$HOST")
case "$1" in
  run) shift; "${SSH[@]}" "$@" ;;
  put) "${SSH[@]}" "cat > '$3'" < "$2" ;;
  *) echo "usage: box.sh run CMD | put LOCAL REMOTE" >&2; exit 64 ;;
esac
