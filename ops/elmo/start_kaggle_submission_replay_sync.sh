#!/usr/bin/env bash
# Launch a bandwidth-contained Kaggle submission-replay sync on Elmo.
set -euo pipefail

root="${POKEBOT_ELMO_ROOT:-/mnt/Main/main/poke-bot-agent}"
archive_dir="${1:-$root/archive/submission-replays}"
image="${POKEBOT_DOWNLOAD_IMAGE:-python:3.11-slim}"
maximum_mbit="${POKEBOT_DOWNLOAD_MAX_MBIT:-12}"
container_name="${POKEBOT_SUBMISSION_REPLAY_CONTAINER:-pokebot-kaggle-submission-replay-sync}"

mkdir -p "$archive_dir"

# Prefer repo/ops copies when present; fall back to durable Elmo root copies.
sync_py="${POKEBOT_SUBMISSION_REPLAY_SYNC_PY:-}"
if [[ -z "$sync_py" ]]; then
  for candidate in \
      "$root/scripts/sync_kaggle_submission_replays_elmo.py" \
      "$root/sync_kaggle_submission_replays_elmo.py" \
      "/home/admin/pokebot-expert-guide-src-v1/scripts/sync_kaggle_submission_replays_elmo.py"; do
    if [[ -f "$candidate" ]]; then
      sync_py="$candidate"
      break
    fi
  done
fi
sync_sh="${POKEBOT_SUBMISSION_REPLAY_SYNC_SH:-}"
if [[ -z "$sync_sh" ]]; then
  for candidate in \
      "$root/ops/elmo/sync_kaggle_submission_replays.sh" \
      "$root/sync_kaggle_submission_replays.sh" \
      "/home/admin/pokebot-expert-guide-src-v1/ops/elmo/sync_kaggle_submission_replays.sh"; do
    if [[ -f "$candidate" ]]; then
      sync_sh="$candidate"
      break
    fi
  done
fi
if [[ -z "${sync_py:-}" || -z "${sync_sh:-}" ]]; then
  echo "missing sync script copies (py/sh) under $root or pokebot-expert-guide-src-v1" >&2
  exit 1
fi

docker_bin=(docker)
if ! docker info >/dev/null 2>&1; then
  docker_bin=(sudo -n docker)
fi

exec "${docker_bin[@]}" run --rm \
  --name "$container_name" \
  --cap-add NET_ADMIN \
  -e "POKEBOT_DOWNLOAD_MAX_MBIT=$maximum_mbit" \
  -e "POKEBOT_EXPERT_DOWNLOAD_MIN_FREE_GIB=${POKEBOT_EXPERT_DOWNLOAD_MIN_FREE_GIB:-200}" \
  -e "POKEBOT_SUBMISSION_REPLAY_MIN_ID=${POKEBOT_SUBMISSION_REPLAY_MIN_ID:-55315274}" \
  -e "POKEBOT_SUBMISSION_REPLAY_WORKERS=${POKEBOT_SUBMISSION_REPLAY_WORKERS:-4}" \
  -v "$archive_dir:/archive" \
  -v "$root/.secrets/kaggle.json:/root/.kaggle/kaggle.json:ro" \
  -v "$sync_sh:/work/sync.sh:ro" \
  -v "$sync_py:/work/sync_kaggle_submission_replays_elmo.py:ro" \
  "$image" \
  bash -lc \
  'apt-get update -qq &&
   apt-get install -y -qq --no-install-recommends iproute2 >/dev/null &&
   rm -rf /var/lib/apt/lists/* &&
   pip install --no-cache-dir --quiet kaggle &&
   exec /work/sync.sh /archive "$@"' \
  sync "${@:2}"
