#!/bin/bash
# box48.sh run '<cmd>'        run a command on the trn2.48xl (ap-south-2)
# box48.sh put LOCAL REMOTE   copy a file over stdin
HOST=ubuntu@18.61.33.185
SSH=(ssh -F "$HOME/.ssh/config" -i "$HOME/.ssh/trn2-ap-south-2-20261104" -o IdentitiesOnly=yes -o ConnectTimeout=15 -o ServerAliveInterval=30)
case "${1:-}" in
  run) shift; "${SSH[@]}" "$HOST" "$@" ;;
  put) shift; cat "$1" | "${SSH[@]}" "$HOST" "cat > $2" ;;
  *) echo "usage: box48.sh run '<cmd>' | box48.sh put LOCAL REMOTE" >&2; exit 64 ;;
esac
