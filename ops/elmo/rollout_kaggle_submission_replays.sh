#!/usr/bin/env bash
# Full authoritative visual-trace rollout for Elmo submission-replay archives.
set -euo pipefail

source_root="${POKEBOT_ROLLOUT_SOURCE:-/home/admin/pokebot-expert-guide-src-v1}"
replay_root="${POKEBOT_SUBMISSION_REPLAY_ARCHIVE:-/mnt/Main/main/poke-bot-agent/archive/submission-replays}"
output_root="${POKEBOT_SUBMISSION_REPLAY_ROLLOUT_ROOT:-/mnt/Main/main/poke-bot-agent/archive/submission-replay-rollouts}"
workers="${POKEBOT_SUBMISSION_REPLAY_ROLLOUT_WORKERS:-4}"
archetype="${POKEBOT_SUBMISSION_REPLAY_ROLLOUT_ARCHETYPE:-*}"

script="$source_root/scripts/rollout_kaggle_submission_replays.py"
if [[ ! -f "$script" ]]; then
  # Fallback to durable Elmo root copy.
  script="${POKEBOT_ELMO_ROOT:-/mnt/Main/main/poke-bot-agent}/scripts/rollout_kaggle_submission_replays.py"
fi
if [[ ! -f "$script" ]]; then
  echo "rollout script missing: $script" >&2
  exit 1
fi

mkdir -p "$output_root"
cd "$source_root"
export PYTHONPATH="$source_root${PYTHONPATH:+:$PYTHONPATH}"
export POKEBOT_SUBMISSION_REPLAY_ARCHIVE="$replay_root"
export POKEBOT_SUBMISSION_REPLAY_ROLLOUT_ROOT="$output_root"

# Default: only trace episode ids written by the latest sync NEW_DOWNLOADS.json.
exec python3 "$script" \
  --replay-root "$replay_root" \
  --output-root "$output_root" \
  --workers "$workers" \
  --required-archetype "$archetype" \
  --new-downloads-only \
  "$@"
