#!/usr/bin/env bash
# Download missing Kaggle submission replays, then full visual-trace rollout.
set -euo pipefail

root="${POKEBOT_ELMO_ROOT:-/mnt/Main/main/poke-bot-agent}"
archive="${POKEBOT_SUBMISSION_REPLAY_ARCHIVE:-$root/archive/submission-replays}"
start_sync="${POKEBOT_SUBMISSION_REPLAY_START_SH:-$root/start_kaggle_submission_replay_sync.sh}"
rollout_sh="${POKEBOT_SUBMISSION_REPLAY_ROLLOUT_SH:-}"

if [[ -z "$rollout_sh" ]]; then
  for candidate in \
      "$root/ops/elmo/rollout_kaggle_submission_replays.sh" \
      "$root/rollout_kaggle_submission_replays.sh" \
      "/home/admin/pokebot-expert-guide-src-v1/ops/elmo/rollout_kaggle_submission_replays.sh"; do
    if [[ -f "$candidate" ]]; then
      rollout_sh="$candidate"
      break
    fi
  done
fi

if [[ ! -x "$start_sync" && ! -f "$start_sync" ]]; then
  echo "missing download launcher: $start_sync" >&2
  exit 1
fi
if [[ -z "${rollout_sh:-}" ]]; then
  echo "missing rollout launcher" >&2
  exit 1
fi

echo "[pipeline] begin download sync archive=$archive"
/usr/bin/bash "$start_sync" "$archive"

manifest="$archive/NEW_DOWNLOADS.json"
new_count=0
if [[ -f "$manifest" ]]; then
  new_count="$(python3 - "$manifest" <<'PY'
import json, sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(int(payload.get("episode_count") or 0))
PY
)"
fi

if (( new_count <= 0 )); then
  echo "[pipeline] no new downloads; skipping rollout"
  echo "[pipeline] complete"
  exit 0
fi

echo "[pipeline] begin rollout for $new_count newly downloaded replay(s)"
/usr/bin/bash "$rollout_sh" --new-downloads-only
echo "[pipeline] complete"
