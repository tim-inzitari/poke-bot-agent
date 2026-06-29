#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

test -f submission/main.py
test -f submission/deck.csv
test -f submission/policy_runtime.py
test -d submission/cg

# Guard: submission scoring must not have drifted from poke_agent (single source of
# truth for archetype heuristics). Fails the build on any divergence or missing symbol.
echo "==> verifying submission heuristics parity with poke_agent"
python3 -m pytest tests/test_submission_heuristics_parity.py -q

CG_LIB="kaggle/input/cg-lib/cg/libcg.so"
if [ ! -f "$CG_LIB" ]; then
  echo "missing $CG_LIB; run scripts/download-kaggle-inputs.sh first" >&2
  exit 1
fi

MODEL_CHECKPOINT="${VALUE_MODEL_PATH:-outputs/checkpoints/temporal_current.pt}"
if [ ! -f "$MODEL_CHECKPOINT" ] && [ -f "out/value_model.pt" ]; then
  MODEL_CHECKPOINT="out/value_model.pt"
fi
if [ ! -f "$MODEL_CHECKPOINT" ]; then
  echo "missing $MODEL_CHECKPOINT; train with scripts/train_agent.py or notebooks/poke_agent_training.ipynb first" >&2
  exit 1
fi

mkdir -p dist
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

cp submission/main.py submission/deck.csv submission/policy_runtime.py submission/beam_search.py submission/rewards.py submission/archetype_heuristics.py "$STAGING/"
PYTHONPATH=. python3 scripts/extract_archetype_signatures.py --output "$STAGING/archetype_signatures_data.py"
cp poke_agent/models/temporal_transformer.py poke_agent/features.py poke_agent/game_tracker.py "$STAGING/"
mv "$STAGING/temporal_transformer.py" "$STAGING/model.py"
cp -r submission/cg "$STAGING/"
cp "$CG_LIB" "$STAGING/cg/libcg.so"
cp "$MODEL_CHECKPOINT" "$STAGING/value_model.pt"

tar -czf dist/submission.tar.gz -C "$STAGING" \
  main.py deck.csv cg model.py features.py game_tracker.py rewards.py policy_runtime.py beam_search.py archetype_heuristics.py archetype_signatures_data.py value_model.pt

python3 scripts/validate_submission.py dist/submission.tar.gz
echo "built dist/submission.tar.gz with trained model $(basename "$MODEL_CHECKPOINT")"
