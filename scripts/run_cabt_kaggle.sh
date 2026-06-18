#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cp scripts/generate_cabt_data.py kaggle/cabt-sim/generate_cabt_data.py
cp submission/deck.csv kaggle/cabt-sim/deck.csv
kaggle kernels push -p kaggle/cabt-sim -t "${KAGGLE_TIMEOUT_SECONDS:-600}"

echo "Kaggle run submitted."
echo "Check status:  kaggle kernels status timinzitari/poke-agent-cabt-simulation"
echo "Get output:    kaggle kernels output timinzitari/poke-agent-cabt-simulation -p data/kaggle-output"
