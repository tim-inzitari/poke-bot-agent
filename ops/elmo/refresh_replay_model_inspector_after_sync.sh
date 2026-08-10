#!/usr/bin/env bash
# Refresh the read-only inspector only after a successful sync added replays.
set -euo pipefail

root="${POKEBOT_ELMO_ROOT:-/mnt/Main/main/poke-bot-agent}"
manifest="${POKEBOT_SUBMISSION_REPLAY_ARCHIVE:-$root/archive/submission-replays}/NEW_DOWNLOADS.json"
check_only=0
if [[ "${1:-}" == "--check" ]]; then
  check_only=1
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--check]" >&2
  exit 2
fi

new_count=0
if [[ -f "$manifest" ]]; then
  new_count="$(python3 - "$manifest" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
value = payload.get("episode_count")
if isinstance(value, bool) or not isinstance(value, int) or value < 0:
    raise SystemExit("NEW_DOWNLOADS episode_count must be a non-negative integer")
print(value)
PY
)"
fi

if (( new_count <= 0 )); then
  echo "[inspector-refresh] no new replay bytes; managed inspector unchanged"
  exit 0
fi
if (( check_only )); then
  echo "[inspector-refresh] check: would refresh for $new_count new replay(s)"
  exit 0
fi

echo "[inspector-refresh] refreshing managed read-only inspector for $new_count new replay(s)"
/usr/bin/bash "$root/rebuild_replay_model_inspector_provenance.sh"
/usr/bin/systemctl try-restart pokebot-replay-model-inspector.service
