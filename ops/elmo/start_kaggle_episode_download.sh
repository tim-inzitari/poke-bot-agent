#!/usr/bin/env bash
# Launch a bandwidth-contained, managed Kaggle episode download on Elmo.
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 ARCHIVE_DIR YYYY-MM-DD:EXPECTED_EPISODES [...]" >&2
  exit 2
fi

root="${POKEBOT_ELMO_ROOT:-/mnt/Main/main/poke-bot-agent}"
image="${POKEBOT_DOWNLOAD_IMAGE:-python:3.11-slim}"
maximum_mbit="${POKEBOT_DOWNLOAD_MAX_MBIT:-12}"

exec docker run --rm \
  --name pokebot-expert-episode-download \
  --cap-add NET_ADMIN \
  -e "POKEBOT_DOWNLOAD_MAX_MBIT=$maximum_mbit" \
  -v "$1:/archive" \
  -v "$root/.secrets/kaggle.json:/root/.kaggle/kaggle.json:ro" \
  -v "$root/download_kaggle_episode_days.sh:/work/download.sh:ro" \
  "$image" \
  bash -lc \
  'apt-get update -qq &&
   apt-get install -y -qq --no-install-recommends iproute2 >/dev/null &&
   rm -rf /var/lib/apt/lists/* &&
   pip install --no-cache-dir --quiet kaggle &&
   exec /work/download.sh /archive "$@"' \
  download "${@:2}"
