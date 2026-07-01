#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

CONFIG_FILE="${1:-configs/ubuntu_two_week.env}"
CONFIG_SOURCE="built-in defaults"
if [[ -f "$CONFIG_FILE" ]]; then
  CONFIG_SOURCE="$CONFIG_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  set +a
elif [[ "$CONFIG_FILE" != "configs/ubuntu_two_week.env" ]]; then
  echo "missing config file: $CONFIG_FILE" >&2
  exit 1
fi

mkdir -p outputs/logs outputs/submissions
LOG_FILE="${LOG_FILE:-outputs/logs/ubuntu_long_run_$(date -u +%Y%m%dT%H%M%SZ).log}"
ln -sfn "$(basename "$LOG_FILE")" outputs/logs/latest_ubuntu_long_run.log
exec > >(tee -a "$LOG_FILE") 2>&1

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export POKE_AGENT_SELECT_LARGEST_CUDA="${POKE_AGENT_SELECT_LARGEST_CUDA:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TENSOR_BUILD_WORKERS="${TENSOR_BUILD_WORKERS:-28}"

export MODEL_OUTPUT_PATH="${MODEL_OUTPUT_PATH:-outputs/checkpoints/temporal_current.pt}"
export BATCH_GAMES="${BATCH_GAMES:-8}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
export EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.01}"
export SELF_PLAY_TRAIN_EPOCHS="${SELF_PLAY_TRAIN_EPOCHS:-100}"
export SELF_PLAY_GAMES="${SELF_PLAY_GAMES:-250}"
export SELF_PLAY_EVAL_GAMES="${SELF_PLAY_EVAL_GAMES:-50}"
export SELF_PLAY_ITERATIONS="${SELF_PLAY_ITERATIONS:-100000}"
export SELF_PLAY_OPPONENT_POOL_SIZE="${SELF_PLAY_OPPONENT_POOL_SIZE:-8}"
export SELF_PLAY_OUTPUT_PATH="${SELF_PLAY_OUTPUT_PATH:-outputs/rollouts/ubuntu_long_run.jsonl}"
export SELF_PLAY_CHECKPOINT_DIR="${SELF_PLAY_CHECKPOINT_DIR:-outputs/checkpoints/ubuntu_long_run}"
export SELF_PLAY_BASELINES="${SELF_PLAY_BASELINES:-public}"
export SELF_PLAY_TRAIN_VS_BASELINES="${SELF_PLAY_TRAIN_VS_BASELINES:-1}"
export SELF_PLAY_USE_BEAM="${SELF_PLAY_USE_BEAM:-1}"
export SELF_PLAY_TARGET_WIN_RATE="${SELF_PLAY_TARGET_WIN_RATE:-0.99}"
export SELF_PLAY_PLATEAU_PATIENCE="${SELF_PLAY_PLATEAU_PATIENCE:-100000}"

RUN_HOURS="${RUN_HOURS:-336}"
BOOTSTRAP_IF_NEEDED="${BOOTSTRAP_IF_NEEDED:-1}"
BOOTSTRAP_GAMES="${BOOTSTRAP_GAMES:-20000}"
BOOTSTRAP_WORKERS="${BOOTSTRAP_WORKERS:-28}"
BOOTSTRAP_ROLLOUTS="${BOOTSTRAP_ROLLOUTS:-data/ubuntu_bootstrap_rollouts.jsonl}"
BOOTSTRAP_MERGED="${BOOTSTRAP_MERGED:-data/ubuntu_training_rollouts_merged.jsonl}"
BOOTSTRAP_TRAIN_EPOCHS="${BOOTSTRAP_TRAIN_EPOCHS:-200}"
AUTO_DOWNLOAD_KAGGLE_INPUTS="${AUTO_DOWNLOAD_KAGGLE_INPUTS:-1}"
BUILD_SUBMISSION_EACH_ITER="${BUILD_SUBMISSION_EACH_ITER:-1}"
MAX_CONSECUTIVE_FAILURES="${MAX_CONSECUTIVE_FAILURES:-20}"
FAILURE_SLEEP_SECONDS="${FAILURE_SLEEP_SECONDS:-300}"

python_bin=".venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "missing .venv; run scripts/ubuntu_setup.sh first" >&2
  exit 1
fi

log_header() {
  echo
  echo "================================================================================"
  echo "$*"
  echo "================================================================================"
}

latest_checkpoint() {
  "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

checkpoint_dir = Path(os.environ["SELF_PLAY_CHECKPOINT_DIR"])
base = Path(os.environ["MODEL_OUTPUT_PATH"])
manifest = checkpoint_dir / "manifest.json"
if manifest.exists():
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for entry in reversed(data.get("iterations", [])):
            candidate = Path(entry.get("saved_checkpoint", ""))
            if candidate.exists():
                print(candidate)
                raise SystemExit
    except Exception:
        pass
candidates = sorted(checkpoint_dir.glob("iter_*.pt"))
if candidates:
    print(candidates[-1])
elif base.exists():
    print(base)
PY
}

next_iteration() {
  "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

manifest = Path(os.environ["SELF_PLAY_CHECKPOINT_DIR"]) / "manifest.json"
if not manifest.exists():
    print(1)
    raise SystemExit
try:
    data = json.loads(manifest.read_text(encoding="utf-8"))
except Exception:
    print(1)
    raise SystemExit
print(int(data.get("next_iteration", len(data.get("iterations", [])) + 1)))
PY
}

link_latest_checkpoint() {
  local checkpoint="$1"
  "$python_bin" - "$checkpoint" <<'PY'
import os
import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
link = Path(os.environ["SELF_PLAY_CHECKPOINT_DIR"]) / "latest.pt"
link.parent.mkdir(parents=True, exist_ok=True)
tmp = link.with_suffix(".tmp")
try:
    tmp.unlink()
except FileNotFoundError:
    pass
tmp.symlink_to(target)
tmp.replace(link)
print(f"latest symlink -> {target}")
PY
}

build_submission_snapshot() {
  local checkpoint="$1"
  local iteration="$2"
  if [[ "$BUILD_SUBMISSION_EACH_ITER" != "1" ]]; then
    return 0
  fi
  VALUE_MODEL_PATH="$checkpoint" scripts/build_submission.sh
  cp dist/submission.tar.gz "outputs/submissions/submission_iter_$(printf '%06d' "$iteration").tar.gz"
}

log_header "Ubuntu long-run starting"
date -u +"%Y-%m-%dT%H:%M:%SZ"
echo "config file: $CONFIG_FILE"
echo "config source: $CONFIG_SOURCE"
echo "log file:    $LOG_FILE"
echo "run hours:   $RUN_HOURS"
echo "checkpoint dir: $SELF_PLAY_CHECKPOINT_DIR"
echo "rollouts:       $SELF_PLAY_OUTPUT_PATH"

if [[ "$AUTO_DOWNLOAD_KAGGLE_INPUTS" == "1" && ! -f kaggle/input/cg-lib/cg/libcg.so ]]; then
  log_header "Downloading Kaggle simulator inputs"
  scripts/download-kaggle-inputs.sh
fi

if [[ ! -f kaggle/input/cg-lib/cg/libcg.so ]]; then
  echo "missing kaggle/input/cg-lib/cg/libcg.so; run scripts/download-kaggle-inputs.sh after configuring Kaggle auth" >&2
  exit 1
fi

log_header "Probe"
"$python_bin" scripts/ubuntu_probe.py

checkpoint="$(latest_checkpoint || true)"
if [[ -z "$checkpoint" && "$BOOTSTRAP_IF_NEEDED" == "1" ]]; then
  log_header "Bootstrap rollout generation"
  "$python_bin" scripts/generate_cabt_data.py \
    --episodes "$BOOTSTRAP_GAMES" \
    --workers "$BOOTSTRAP_WORKERS" \
    --matchups weighted \
    --out "$BOOTSTRAP_ROLLOUTS"

  log_header "Bootstrap rollout merge"
  merge_sources=()
  [[ -f data/scraped_rollouts.jsonl ]] && merge_sources+=(data/scraped_rollouts.jsonl)
  merge_sources+=("$BOOTSTRAP_ROLLOUTS")
  "$python_bin" scripts/merge_rollouts.py "${merge_sources[@]}" --out "$BOOTSTRAP_MERGED"

  log_header "Bootstrap model training"
  DATASET_GAMES=0 \
  PRIMARY_ROLLOUT_DATA="$BOOTSTRAP_MERGED" \
  MERGED_ROLLOUT_DATA="$BOOTSTRAP_MERGED" \
  TRAIN_EPOCHS="$BOOTSTRAP_TRAIN_EPOCHS" \
  "$python_bin" scripts/train_agent.py
  checkpoint="$(latest_checkpoint || true)"
fi

if [[ -z "$checkpoint" ]]; then
  echo "no checkpoint available and bootstrap disabled/failed" >&2
  exit 1
fi

mkdir -p "$SELF_PLAY_CHECKPOINT_DIR" "$(dirname "$SELF_PLAY_OUTPUT_PATH")"
link_latest_checkpoint "$checkpoint"

start_epoch="$(date +%s)"
end_epoch="$((start_epoch + RUN_HOURS * 3600))"
failures=0

while [[ "$(date +%s)" -lt "$end_epoch" ]]; do
  iteration="$(next_iteration)"
  checkpoint="$(latest_checkpoint || true)"
  if [[ -z "$checkpoint" ]]; then
    echo "lost checkpoint; aborting" >&2
    exit 1
  fi

  log_header "Self-play iteration $iteration"
  echo "checkpoint in: $checkpoint"
  echo "time remaining seconds: $((end_epoch - $(date +%s)))"

  if "$python_bin" scripts/run_self_play.py \
    --iterations "$iteration" \
    --games "$SELF_PLAY_GAMES" \
    --eval-games "$SELF_PLAY_EVAL_GAMES" \
    --train-epochs "$SELF_PLAY_TRAIN_EPOCHS" \
    --batch-games "$BATCH_GAMES" \
    --early-stop-patience "$EARLY_STOP_PATIENCE" \
    --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
    --target-win-rate "$SELF_PLAY_TARGET_WIN_RATE" \
    --plateau-patience "$SELF_PLAY_PLATEAU_PATIENCE" \
    --baselines "$SELF_PLAY_BASELINES" \
    --checkpoint "$checkpoint"; then
    failures=0
    checkpoint="$(latest_checkpoint || true)"
    if [[ -n "$checkpoint" ]]; then
      link_latest_checkpoint "$checkpoint"
      build_submission_snapshot "$checkpoint" "$iteration" || true
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu,utilization.gpu \
        --format=csv,noheader,nounits || true
    fi
    df -h . || true
  else
    failures="$((failures + 1))"
    echo "iteration $iteration failed; consecutive failures=$failures/$MAX_CONSECUTIVE_FAILURES" >&2
    if [[ "$failures" -ge "$MAX_CONSECUTIVE_FAILURES" ]]; then
      echo "too many consecutive failures; aborting" >&2
      exit 1
    fi
    sleep "$FAILURE_SLEEP_SECONDS"
  fi
done

log_header "Ubuntu long-run complete"
checkpoint="$(latest_checkpoint || true)"
echo "latest checkpoint: ${checkpoint:-none}"
date -u +"%Y-%m-%dT%H:%M:%SZ"
