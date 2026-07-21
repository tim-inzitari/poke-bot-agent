#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/inzi/poke-bot-agent
PYTHON=/home/inzi/miniconda3/envs/poke-bot-agent/bin/python
RUN=state_core_top_ladder_latest10_20260719
MANIFEST="$ROOT/data/bootstrap/latest10-20260709-20260718/manifest.json"
SEED="$ROOT/outputs/checkpoints/state_core_top_ladder_5day_20260719.latest.pt"
LATEST="$ROOT/outputs/checkpoints/${RUN}.latest.pt"

# Freeze the intended 1.5M stateless evaluator.  Do not let a stale host
# default or inherited PURE_RL_* environment silently restore temporal layers.
export PURE_RL_D_MODEL=96
export PURE_RL_SPATIAL_LAYERS=4
export PURE_RL_TEMPORAL_LAYERS=0
export PURE_RL_OPTION_DECODER_LAYERS=4
export PURE_RL_N_HEADS=8
export PURE_RL_FF_DIM=384
export PURE_RL_MAX_CONTEXT=320
export PURE_RL_TEMPORAL_POS=rope
export PURE_RL_DECISION_CONTEXT=stateless
export PURE_RL_HISTORY_ACTION_SCALE=0.1
export PURE_RL_CARD_EMBED_DIM=48
export PURE_RL_ATTACK_EMBED_DIM=48
export PURE_RL_DENSE_CARD2VEC=1
export PURE_RL_DROPOUT=0.05

AVAILABLE_KIB="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if (( AVAILABLE_KIB < 30 * 1024 * 1024 )); then
  echo "host memory guard failed: MemAvailable=${AVAILABLE_KIB} KiB" >&2
  exit 1
fi
test -s "$MANIFEST"

ARGS=(
  --feature-manifest "$MANIFEST"
  --archetype top-ladder-core
  --run-name "$RUN"
  --model-profile pure-rl
  --epochs 30
  --lr 5e-5
  --games-per-batch 128
  --max-decisions-per-batch 8192
  --device-resident
  --val-frac 0.10
  --split-by-episode
  --patience 4
  --aux-loss-weight 0
  --opp-hand-loss-weight 0
  --opp-remainder-loss-weight 0
  --lethal-threat-loss-weight 0
  --prize-race-loss-weight 0
  --min-usable-record-frac 0.98
  --min-decisions 5500000
  --seed 20260729
)

cd "$ROOT"
SOURCE="$SEED"
if [[ -s "$LATEST" ]]; then
  SOURCE="$LATEST"
fi
"$PYTHON" - "$SOURCE" <<'PY'
import sys

from poke_bot import checkpoint
from poke_bot.pure_rl.model_profile import model_config_dict, pure_rl_model_config

path = sys.argv[1]
saved = checkpoint.load_checkpoint(path, map_location="cpu")
actual = dict(saved.get("model_config") or {})
expected = model_config_dict(pure_rl_model_config())
if actual != expected:
    diff = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    raise SystemExit(f"checkpoint profile mismatch before dataset load: {diff}")
print(f"[bootstrap-preflight] compatible checkpoint profile: {path}", flush=True)
PY
if [[ -s "$LATEST" ]]; then
  echo "[bootstrap-launch] resuming $LATEST"
  exec "$PYTHON" -u scripts/train_bootstrap.py "${ARGS[@]}" --resume auto
fi
test -s "$SEED"
echo "[bootstrap-launch] initializing from $SEED"
exec "$PYTHON" -u scripts/train_bootstrap.py "${ARGS[@]}" --resume 0 --init-checkpoint "$SEED"
