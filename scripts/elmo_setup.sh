#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install kaggle numpy tqdm

mkdir -p data
scripts/download-kaggle-inputs.sh

echo "Elmo setup complete."
echo "Benchmark with:"
echo "  scripts/elmo_generate.sh --episodes 100 --workers 16 --out data/elmo-rollouts.jsonl"
