#!/usr/bin/env bash
# Package greedy hammer-pult specialist for Kaggle after SPECIALIST_GATE_PASSED.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_NAME="${1:?usage: submit_pure_rl_greedy.sh <pure_rl_run_name>}"
RUN_DIR="outputs/pure_rl/${RUN_NAME}"
GATE="${RUN_DIR}/SPECIALIST_GATE_PASSED"
if [[ ! -f "$GATE" ]]; then
  echo "error: missing ${GATE} — refuse submit before specialist gate" >&2
  exit 2
fi

CKPT=$(ls -1 "${RUN_DIR}/checkpoints"/iter_*.pt 2>/dev/null | tail -1 || true)
if [[ -z "${CKPT}" ]]; then
  CKPT="${RUN_DIR}/checkpoints/hammer-pult_warmstart.pt"
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "error: no specialist checkpoint under ${RUN_DIR}/checkpoints" >&2
  exit 2
fi

echo "[submit] using checkpoint ${CKPT}"
# Reuse existing submission builder when present; force policy/greedy contract.
export POKEBOT_BLACKWELL_STRATEGY_HEADS=0
export POKEBOT_PRIMARY_ARCHETYPE=hammer-pult
if [[ -x scripts/build_submission.sh ]]; then
  CHECKPOINT_PATH="${CKPT}" AGENT_MODE=policy MCTS_SIMS=0 \
    bash scripts/build_submission.sh
else
  echo "error: scripts/build_submission.sh missing — copy ${CKPT} into submission package manually" >&2
  exit 2
fi
