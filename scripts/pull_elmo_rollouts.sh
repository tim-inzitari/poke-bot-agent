#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ELMO_HOST="${ELMO_HOST:-elmo}"
ELMO_PATH="${ELMO_PATH:-~/poke-bot-agent}"
REMOTE_FILE="${REMOTE_FILE:-data/elmo-rollouts.jsonl}"
LOCAL_DIR="${LOCAL_DIR:-data/elmo}"

mkdir -p "$LOCAL_DIR"
rsync -av "$ELMO_HOST:$ELMO_PATH/$REMOTE_FILE" "$LOCAL_DIR/"

echo "Pulled to $LOCAL_DIR/$(basename "$REMOTE_FILE")"
