#!/usr/bin/env bash
# Assemble the canonical latest-20 specialist corpora after daily features land.
set -euo pipefail

SOURCE="${POKEBOT_SOURCE:-/home/admin/pokebot-expert-src-v41}"
MAIN="${POKEBOT_MAIN:-/mnt/Main/main}"
ARCHIVE_RECEIPT="${POKEBOT_ARCHIVE_RECEIPT:-$MAIN/poke-bot-agent/archive/expert-latest20/current.json}"
CURRENT_FEATURES="${POKEBOT_CURRENT_FEATURES:-$MAIN/poke-bot-agent/archive/expert-latest20-derived/daily/roster18-v5}"
EXISTING_FEATURES="${POKEBOT_EXISTING_FEATURES:-$MAIN/poke-core-starmie-corpus-v1/all-recognized/daily}"
OUTPUT="${POKEBOT_OUTPUT:-$MAIN/poke-bot-agent/archive/expert-latest20-derived/windows/2026-07-04_2026-07-23/roster18-v5}"
IMAGE="${POKEBOT_IMAGE:-poke-bot-truenas-worker:matchup-v33-runtime}"
NAME="${POKEBOT_CONTAINER_NAME:-pokebot-expert-latest20-finalizer}"
LOCK="${POKEBOT_LOCK:-/tmp/${NAME}.lock}"
SHARED_GID="${POKEBOT_SHARED_GID:-950}"
CONTAINER_CPUS="${POKEBOT_CPUS:-20}"
CONTAINER_MEMORY="${POKEBOT_MEMORY:-48g}"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "latest-20 specialist finalizer launch already in progress"
  exit 0
fi

test -s "$SOURCE/scripts/finalize_latest20_specialist_corpora.py"
test -s "$SOURCE/state/matchup_adapter_roster.json"
test -s "$ARCHIVE_RECEIPT"
sudo -n mkdir -p "$(dirname "$OUTPUT")"

if [[ -s "$OUTPUT/LATEST20_SPECIALIST_CORPORA_READY.json" ]]; then
  echo "latest-20 specialist corpora already ready: $OUTPUT"
  exit 0
fi

if sudo -n docker inspect "$NAME" >/dev/null 2>&1; then
  state="$(sudo -n docker inspect -f '{{.State.Status}}' "$NAME")"
  if [[ "$state" == "running" ]]; then
    echo "$NAME already running"
    exit 0
  fi
  sudo -n docker rm "$NAME" >/dev/null
fi

sudo -n docker run -d \
  --name "$NAME" \
  --restart on-failure:5 \
  --cpus "$CONTAINER_CPUS" \
  --memory "$CONTAINER_MEMORY" \
  --memory-swap "$CONTAINER_MEMORY" \
  --pids-limit 2048 \
  -e PYTHONUNBUFFERED=1 \
  -v "$SOURCE:/workspace:ro" \
  -v "$MAIN:$MAIN" \
  --entrypoint /bin/bash \
  "$IMAGE" -lc "
set -euo pipefail
cd /workspace
until [[ -s '$CURRENT_FEATURES/MISSING_DAYS_READY.json' ]]; do
  sleep 30
done
python -u scripts/finalize_latest20_specialist_corpora.py \
  --archive-receipt '$ARCHIVE_RECEIPT' \
  --candidate-root '$CURRENT_FEATURES' \
  --candidate-root '$EXISTING_FEATURES' \
  --output-root '$OUTPUT' \
  --roster /workspace/state/matchup_adapter_roster.json \
  --source-repo /workspace \
  --minimum-decisions 1
chgrp -R '$SHARED_GID' '$OUTPUT'
chmod -R g+rX '$OUTPUT'
"

echo "started $NAME"
echo "status: sudo docker logs -f $NAME"
