#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

test -f submission/main.py
test -f submission/deck.csv
test -f submission/policy_runtime.py
test -d submission/cg

CG_LIB="kaggle/input/cg-lib/cg/libcg.so"
if [ ! -f "$CG_LIB" ]; then
  echo "missing $CG_LIB; run scripts/download-kaggle-inputs.sh first" >&2
  exit 1
fi

MODEL_CHECKPOINT="${VALUE_MODEL_PATH:-out/value_model.pt}"
if [ ! -f "$MODEL_CHECKPOINT" ]; then
  echo "missing $MODEL_CHECKPOINT; train with scripts/train_agent.py or notebooks/poke_agent_training.ipynb first" >&2
  exit 1
fi

mkdir -p dist
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

cp submission/main.py submission/deck.csv submission/policy_runtime.py "$STAGING/"
cp poke_agent/model.py poke_agent/features.py "$STAGING/"
cp -r submission/cg "$STAGING/"
cp "$CG_LIB" "$STAGING/cg/libcg.so"
cp "$MODEL_CHECKPOINT" "$STAGING/value_model.pt"

tar -czf dist/submission.tar.gz -C "$STAGING" \
  main.py deck.csv cg model.py features.py policy_runtime.py value_model.pt
tar -tzf dist/submission.tar.gz
echo "built dist/submission.tar.gz with trained model $(basename "$MODEL_CHECKPOINT")"
