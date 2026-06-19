#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  echo "missing .venv; run scripts/elmo_setup.sh on Elmo first" >&2
  exit 1
fi

.venv/bin/python scripts/generate_cabt_data.py \
  --episodes "${DATASET_GAMES:-${ELMO_EPISODES:-1000}}" \
  --workers "${ELMO_WORKERS:-16}" \
  --max-steps "${ELMO_MAX_STEPS:-300}" \
  --seed "${ELMO_SEED:-7}" \
  --out "${ELMO_OUT:-data/elmo-rollouts.jsonl}" \
  "$@"
