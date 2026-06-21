#!/usr/bin/env bash
# Poll self-play manifest until Kaggle submission is recorded.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MANIFEST="${MANIFEST:-outputs/checkpoints/self_play/manifest.json}"
POLL_SEC="${POLL_SEC:-30}"
export MANIFEST

echo "watching $MANIFEST for Kaggle submission after self-play..."

while true; do
  if [[ ! -f "$MANIFEST" ]]; then
    sleep "$POLL_SEC"
    continue
  fi

  if grep -q '"kaggle_submission"' "$MANIFEST"; then
    echo ""
    echo "=== Kaggle submission confirmed ==="
    python3 - <<'PY'
import json
from pathlib import Path
import os

path = Path(os.environ["MANIFEST"])
manifest = json.loads(path.read_text())
submission = manifest.get("kaggle_submission") or {}
print("checkpoint:", submission.get("checkpoint", "?"))
print("tarball:   ", submission.get("tarball", "?"))
print("message:   ", submission.get("message", "?"))
output = (submission.get("kaggle_output") or "").strip()
if output:
    print("kaggle:")
    print(output)
PY
    exit 0
  fi

  if grep -q '"kaggle_submission_error"' "$MANIFEST"; then
    echo ""
    echo "=== Kaggle submission failed ===" >&2
    python3 - <<'PY'
import json
from pathlib import Path
import os

path = Path(os.environ["MANIFEST"])
manifest = json.loads(path.read_text())
error = manifest.get("kaggle_submission_error") or {}
print("checkpoint:", error.get("checkpoint", "?"), file=__import__("sys").stderr)
print("returncode:", error.get("returncode", "?"), file=__import__("sys").stderr)
for key in ("stdout", "stderr"):
    text = (error.get(key) or "").strip()
    if text:
        print(f"{key}:", text, file=__import__("sys").stderr)
PY
    exit 1
  fi

  if grep -q '"stop_reason"' "$MANIFEST"; then
    reason=$(python3 -c "import json; print(json.load(open('$MANIFEST')).get('stop_reason',''))")
    echo "$(date '+%H:%M:%S') self-play stopped ($reason); waiting for submission..."
  else
    echo "$(date '+%H:%M:%S') self-play still running..."
  fi

  sleep "$POLL_SEC"
done
