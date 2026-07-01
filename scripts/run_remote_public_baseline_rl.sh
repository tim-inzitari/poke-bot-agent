#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ITERATIONS="${ITERATIONS:-10}"
START_ITERATION="${START_ITERATION:-1}"
GAMES="${GAMES:-20}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
BATCH_GAMES="${BATCH_GAMES:-4}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.01}"
TARGET_WIN_RATE="${TARGET_WIN_RATE:-0.60}"
BASELINES="${BASELINES:-public}"
ROLLOUTS="${ROLLOUTS:-outputs/rollouts/remote_public_baseline_rollouts.jsonl}"
CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/temporal_current.pt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/checkpoints/remote_public_baseline}"
REPORT_DIR="${REPORT_DIR:-outputs/reports}"
DECK="${DECK:-decks/submission.csv}"
POLICY_PORT="${POLICY_PORT:-18765}"
POLICY_URL="${POLICY_URL:-http://host.docker.internal:${POLICY_PORT}}"
POLICY_TIMEOUT="${POLICY_TIMEOUT:-180}"
LOCAL_POLICY_URL="http://127.0.0.1:${POLICY_PORT}"
RUN_UNTIL_EPOCH="${RUN_UNTIL_EPOCH:-0}"

mkdir -p "$CHECKPOINT_DIR" "$(dirname "$ROLLOUTS")" "$REPORT_DIR"

echo "Remote public-baseline RL"
echo "  iterations:      $ITERATIONS"
echo "  start iteration: $START_ITERATION"
echo "  games/iteration: $GAMES"
echo "  train epochs:    $TRAIN_EPOCHS"
echo "  batch games:     $BATCH_GAMES"
echo "  early stop:      patience=$EARLY_STOP_PATIENCE min_delta=$EARLY_STOP_MIN_DELTA"
echo "  target winrate:  $TARGET_WIN_RATE"
echo "  baselines:       $BASELINES"
echo "  rollouts:        $ROLLOUTS"
echo "  checkpoint:      $CHECKPOINT"
echo "  report dir:      $REPORT_DIR"
echo "  deck:            $DECK"
echo "  policy timeout:  ${POLICY_TIMEOUT}s"
if [[ "$RUN_UNTIL_EPOCH" != "0" ]]; then
  echo "  run until epoch: $RUN_UNTIL_EPOCH"
fi
echo

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing checkpoint: $CHECKPOINT" >&2
  echo "train once first or set CHECKPOINT=/path/to/value_model.pt" >&2
  exit 1
fi

cleanup_policy_server() {
  if [[ -n "${POLICY_PID:-}" ]]; then
    kill "$POLICY_PID" >/dev/null 2>&1 || true
    wait "$POLICY_PID" >/dev/null 2>&1 || true
    POLICY_PID=""
  fi
}

start_policy_server() {
  cleanup_policy_server
  echo "starting neural policy server on ${LOCAL_POLICY_URL}"
  .venv/bin/python scripts/policy_server.py \
    --checkpoint "$current_checkpoint" \
    --deck "$DECK" \
    --host 127.0.0.1 \
    --port "$POLICY_PORT" &
  POLICY_PID=$!
  local healthy=0
  for _ in $(seq 1 60); do
    if /usr/bin/curl -fsS "${LOCAL_POLICY_URL}/health" 2>/dev/null | grep -q '"checkpoint"'; then
      /usr/bin/curl -fsS "${LOCAL_POLICY_URL}/health" || true
      echo
      healthy=1
      break
    fi
    sleep 1
  done
  if [[ "$healthy" != "1" ]]; then
    echo "policy server failed to start on ${LOCAL_POLICY_URL}" >&2
    exit 1
  fi
}

trap cleanup_policy_server EXIT

echo "building Linux CABT simulator image"
docker build --platform linux/amd64 -t poke-agent-cabt-sim -f containers/cabt/Dockerfile .

current_checkpoint="$CHECKPOINT"
.venv/bin/python scripts/inspect_model.py --checkpoint "$current_checkpoint" --deck "$DECK"
echo

docker_tty_arg=""
if [[ -t 1 ]]; then
  docker_tty_arg="-t"
fi
for iteration in $(seq "$START_ITERATION" "$ITERATIONS"); do
  if [[ "$RUN_UNTIL_EPOCH" != "0" && "$(date +%s)" -ge "$RUN_UNTIL_EPOCH" ]]; then
    echo "time limit reached before iteration $iteration"
    break
  fi

  echo
  echo "========== iteration ${iteration}/${ITERATIONS}: active simulations =========="
  start_policy_server
  summary_path="${REPORT_DIR}/remote_public_baseline_iter_${iteration}.json"
  append_flag=""
  if [[ "$iteration" != "1" ]]; then
    append_flag="--append"
  fi

  docker run --rm ${docker_tty_arg:+$docker_tty_arg} --platform linux/amd64 \
    -e PYTHONPATH=/workspace:/workspace/kaggle/input/cg-lib \
    -v "$PWD":/workspace \
    -w /workspace \
    poke-agent-cabt-sim \
    python scripts/generate_remote_baseline_rollouts.py \
      --games "$GAMES" \
      --episode-offset "$(( (iteration - 1) * GAMES ))" \
      --out "$ROLLOUTS" \
      --deck "$DECK" \
      --policy-url "$POLICY_URL" \
      --policy-timeout "$POLICY_TIMEOUT" \
      --baselines "$BASELINES" \
      --summary-out "$summary_path" \
      $append_flag
  cleanup_policy_server

  win_rate="$(.venv/bin/python - <<PY
import json
from pathlib import Path
p=Path("$summary_path")
print(json.loads(p.read_text()).get("win_rate", 0.0))
PY
)"
  echo "iteration ${iteration} rollout win_rate=${win_rate}"

  next_checkpoint="${CHECKPOINT_DIR}/iter_$(printf '%03d' "$iteration").pt"
  echo
  echo "========== iteration ${iteration}/${ITERATIONS}: Mac neural training =========="
  .venv/bin/python scripts/train_from_rollouts.py \
    --data "$ROLLOUTS" \
    --checkpoint-in "$current_checkpoint" \
    --checkpoint-out "$next_checkpoint" \
    --deck "$DECK" \
    --epochs "$TRAIN_EPOCHS" \
    --batch-games "$BATCH_GAMES" \
    --early-stop-patience "$EARLY_STOP_PATIENCE" \
    --early-stop-min-delta "$EARLY_STOP_MIN_DELTA"
  current_checkpoint="$next_checkpoint"

  if .venv/bin/python - <<PY
import sys
win=float("$win_rate")
target=float("$TARGET_WIN_RATE")
if win >= target:
    print(f"target reached: {win:.1%} >= {target:.1%}")
    sys.exit(0)
print(f"target not reached yet: {win:.1%} < {target:.1%}")
sys.exit(1)
PY
  then
    break
  fi
done

echo
echo "latest checkpoint: $current_checkpoint"
