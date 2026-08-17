#!/usr/bin/env bash
# Package an explicitly selected greedy specialist after its exact gate passes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_NAME="${1:?usage: submit_pure_rl_greedy.sh <pure_rl_run_name> <archetype>}"
ARCHETYPE="${2:?usage: submit_pure_rl_greedy.sh <pure_rl_run_name> <archetype>}"
if [[ ! "${ARCHETYPE}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error: invalid archetype ${ARCHETYPE}" >&2
  exit 2
fi
RUN_DIR="outputs/pure_rl/${RUN_NAME}"
GATE="${RUN_DIR}/SPECIALIST_GATE_PASSED"
if [[ ! -f "$GATE" ]]; then
  echo "error: missing ${GATE} — refuse submit before specialist gate" >&2
  exit 2
fi

PYTHON_BIN="${POKEBOT_PYTHON:-python3}"
CKPT=$("${PYTHON_BIN}" -c 'import json,sys; from pathlib import Path; from scripts.pure_rl_auto_progress import resolve_gate_checkpoint; run=Path(sys.argv[1]); gate=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")); ckpt=resolve_gate_checkpoint(run, gate); sys.exit("error: gate checkpoint identity missing or mismatched") if ckpt is None else print(ckpt)' "${RUN_DIR}" "${GATE}")

echo "[submit] using checkpoint ${CKPT}"
# Reuse existing submission builder when present; force policy/greedy contract.
export POKEBOT_BLACKWELL_STRATEGY_HEADS=0
export POKEBOT_PRIMARY_ARCHETYPE="${ARCHETYPE}"
if [[ -x scripts/build_submission.sh ]]; then
  CHECKPOINT_PATH="${CKPT}" AGENT_MODE=policy MCTS_SIMS=0 \
    bash scripts/build_submission.sh
else
  echo "error: scripts/build_submission.sh missing — copy ${CKPT} into submission package manually" >&2
  exit 2
fi
