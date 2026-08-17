#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${POKEBOT_PYTHON:-python}"
exec "$PY" "$ROOT/scripts/run_test_profile.py" canary "$@"
