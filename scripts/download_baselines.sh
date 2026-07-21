#!/usr/bin/env bash
# Download baseline agents from Kaggle into gitignored payload dirs.
# See baselines/README.md and scripts/download_baselines.py.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "ERROR: python3 not found" >&2
  exit 1
fi

exec "$PYTHON" "$ROOT/scripts/download_baselines.py" "$@"
