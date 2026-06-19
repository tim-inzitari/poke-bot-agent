#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# cg-lib uses PEP 604 union syntax (e.g. Pokemon | None), which requires Python 3.10+.
if command -v python3.11 >/dev/null 2>&1; then
  python3.11 -m venv .venv
elif command -v conda >/dev/null 2>&1; then
  conda create -y -p .venv python=3.11 pip
else
  echo "python3.11 is required for cg-lib." >&2
  echo "Install with: sudo apt install python3.11 python3.11-venv" >&2
  echo "Or ensure conda is available." >&2
  exit 1
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install kaggle numpy tqdm

mkdir -p data
scripts/download-kaggle-inputs.sh

echo "Elmo setup complete."
echo "Benchmark with:"
echo "  scripts/elmo_generate.sh --episodes 100 --workers 16 --out data/elmo-rollouts.jsonl"
