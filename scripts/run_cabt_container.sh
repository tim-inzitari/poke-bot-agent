#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed. Kaggle is the default path; use this only for local container tests." >&2
  exit 1
fi

if [ ! -d "kaggle/input/cg-lib" ]; then
  echo "missing kaggle/input/cg-lib; run scripts/download-kaggle-inputs.sh first" >&2
  exit 1
fi

docker build --platform linux/amd64 -f containers/cabt/Dockerfile -t poke-agent-cabt-sim .
docker run --rm --platform linux/amd64 \
  -v "$PWD:/workspace" \
  -w /workspace \
  poke-agent-cabt-sim \
  python scripts/generate_cabt_data.py "$@"
