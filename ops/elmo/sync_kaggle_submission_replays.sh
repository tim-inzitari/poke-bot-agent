#!/usr/bin/env bash
# Incrementally sync Kaggle submission replays into an Elmo archive tree.
set -euo pipefail

archive_dir="${1:-/archive}"
shift || true

minimum_free_gib="${POKEBOT_EXPERT_DOWNLOAD_MIN_FREE_GIB:-200}"
maximum_mbit="${POKEBOT_DOWNLOAD_MAX_MBIT:-12}"
min_submission_id="${POKEBOT_SUBMISSION_REPLAY_MIN_ID:-55315274}"
workers="${POKEBOT_SUBMISSION_REPLAY_WORKERS:-4}"
script_path="${POKEBOT_SUBMISSION_REPLAY_SYNC_PY:-/work/sync_kaggle_submission_replays_elmo.py}"

mkdir -p "$archive_dir"

# Kaggle downloads share Elmo's constrained Ethernet path with production RPC.
# Apply an ingress policer inside the download container when it has NET_ADMIN.
# Production worker traffic is in another container/network namespace and is
# therefore never throttled. Fail closed when a positive limit was requested
# but the launcher did not provide the required capability/tooling.
if [[ "$maximum_mbit" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    && awk -v value="$maximum_mbit" 'BEGIN { exit !(value > 0) }'; then
  if ! command -v tc >/dev/null 2>&1; then
    echo "tc is required for POKEBOT_DOWNLOAD_MAX_MBIT=$maximum_mbit" >&2
    exit 1
  fi
  tc qdisc replace dev eth0 handle ffff: ingress
  tc filter replace dev eth0 parent ffff: protocol ip priority 10 u32 \
    match u32 0 0 \
    police rate "${maximum_mbit}mbit" burst 256k drop flowid :1
  echo "[network] Kaggle ingress capped at ${maximum_mbit} Mbit/s"
fi

free_kib="$(df -Pk "$archive_dir" | awk 'NR == 2 {print $4}')"
if (( free_kib < minimum_free_gib * 1024 * 1024 )); then
  echo "free-space guard failed: ${free_kib} KiB available" >&2
  exit 1
fi

export POKEBOT_SUBMISSION_REPLAY_ARCHIVE="$archive_dir"
export POKEBOT_SUBMISSION_REPLAY_MIN_ID="$min_submission_id"

exec python "$script_path" \
  --archive "$archive_dir" \
  --min-submission-id "$min_submission_id" \
  --min-free-gib "$minimum_free_gib" \
  --workers "$workers" \
  --loss-logs \
  "$@"
