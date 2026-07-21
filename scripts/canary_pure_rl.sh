#!/usr/bin/env bash
# Short pure-RL canary: AWR smoke + dual-GPU profile validation wiring.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${POKEBOT_PYTHON:-python3}"

echo "[canary] unit tests"
"$PY" -m pytest -q tests/test_pure_rl_awr.py tests/test_pure_rl_curriculum_gate.py

echo "[canary] smoke core loop"
"$PY" -u scripts/launch_pure_rl.py \
  --mode core \
  --smoke \
  --allow-single-gpu \
  --preflight-profile none \
  --run-name "canary_pure_rl_core" \
  -- \
  --iterations 2 \
  --smoke-games 8 \
  --heldout-games 200 \
  --gate-wr 0.70

echo "[canary] warm-start specialist from smoke ckpt"
CKPT=$(ls -1 outputs/pure_rl/canary_pure_rl_core/checkpoints/iter_*.pt | tail -1)
"$PY" -u scripts/warm_start_pure_rl_specialist.py \
  --core-checkpoint "$CKPT" \
  --run-name canary_pure_rl_hammer \
  --archetype hammer-pult \
  --device cpu

echo "[canary] smoke specialist loop"
"$PY" -u scripts/launch_pure_rl.py \
  --mode specialist \
  --smoke \
  --allow-single-gpu \
  --preflight-profile none \
  --run-name "canary_pure_rl_hammer_loop" \
  -- \
  --specialist-archetype hammer-pult \
  --base-checkpoint "outputs/pure_rl/canary_pure_rl_hammer/checkpoints/hammer-pult_warmstart.pt" \
  --iterations 1 \
  --smoke-games 4 \
  --heldout-games 200

echo "[canary] OK"
