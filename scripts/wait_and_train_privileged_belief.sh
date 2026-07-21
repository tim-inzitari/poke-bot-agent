#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 MANIFEST PYTHON [TRAINER_ARGS...]" >&2
  exit 2
fi

manifest=$1
python=$2
shift 2
root=$(cd "$(dirname "$0")/.." && pwd)

last_notice=0
while [[ ! -s "$manifest" ]]; do
  now=$(date +%s)
  if (( now - last_notice >= 60 )); then
    echo "[belief-train] waiting for canonical Inzi manifest: $manifest"
    last_notice=$now
  fi
  sleep 15
done

echo "[belief-train] canonical corpus ready; starting bounded 3080 Ti warm-start"
exec "$python" -u "$root/scripts/train_privileged_belief_shards.py" \
  --manifest "$manifest" "$@"
