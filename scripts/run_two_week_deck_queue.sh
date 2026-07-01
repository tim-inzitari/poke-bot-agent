#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_HOURS="${RUN_HOURS:-336}"
RUN_SECONDS="$((RUN_HOURS * 3600))"
START_EPOCH="$(date +%s)"
END_EPOCH="$((START_EPOCH + RUN_SECONDS))"

LOG_DIR="${LOG_DIR:-outputs/logs}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/two_week_deck_queue_${RUN_ID}.log}"
LATEST_LOG="${LATEST_LOG:-${LOG_DIR}/latest_two_week_deck_queue.log}"
QUEUE_ROOT="${QUEUE_ROOT:-outputs}"
SUBMISSION_DIR="${SUBMISSION_DIR:-${QUEUE_ROOT}/submissions/two_week_queue}"
SUBMISSION_LOG="${SUBMISSION_LOG:-${QUEUE_ROOT}/reports/two_week_queue/submissions.jsonl}"

mkdir -p "$LOG_DIR" "$SUBMISSION_DIR" "$(dirname "$SUBMISSION_LOG")"
touch "$LOG_PATH"
abs_log="$(cd "$(dirname "$LOG_PATH")" && pwd)/$(basename "$LOG_PATH")"
ln -sf "$abs_log" "$LATEST_LOG"
exec > >(tee -a "$LOG_PATH") 2>&1

python_bin=".venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "missing .venv python at $python_bin" >&2
  exit 1
fi

echo "Two-week deck queue"
echo "started:      $(date)"
echo "run hours:    $RUN_HOURS"
echo "ends epoch:   $END_EPOCH"
echo "log:          $abs_log"
echo

CAFFEINATE_PID=""
if [[ "${USE_CAFFEINATE:-1}" == "1" ]] && command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu -w "$$" &
  CAFFEINATE_PID="$!"
  echo "caffeinate enabled: pid=$CAFFEINATE_PID"
  echo
fi

cleanup() {
  if declare -f cleanup_league_policy_servers >/dev/null 2>&1; then
    cleanup_league_policy_servers
  fi
  if [[ -n "$CAFFEINATE_PID" ]]; then
    kill "$CAFFEINATE_PID" >/dev/null 2>&1 || true
    wait "$CAFFEINATE_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

export PYTHONPATH="${PYTHONPATH:-.}"

# Keep local Mac training intentionally small and checkpoint-compatible.
export MODEL_D_MODEL="${MODEL_D_MODEL:-32}"
export MODEL_HEADS="${MODEL_HEADS:-4}"
export MODEL_LAYERS="${MODEL_LAYERS:-2}"
export MODEL_USE_KAN="${MODEL_USE_KAN:-0}"
export WINDOW_SIZE="${WINDOW_SIZE:-128}"
export TENSOR_BUILD_WORKERS="${TENSOR_BUILD_WORKERS:-12}"

export ITERATIONS="${ITERATIONS:-100000}"
export GAMES="${GAMES:-120}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-120}"
export BATCH_GAMES="${BATCH_GAMES:-16}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
export EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.01}"
export BASELINES="${BASELINES:-public}"
export POLICY_TIMEOUT="${POLICY_TIMEOUT:-180}"
export BASE_POLICY_PORT="${BASE_POLICY_PORT:-19250}"
export LEAGUE_POLICY_BASE_PORT="${LEAGUE_POLICY_BASE_PORT:-19600}"
export LEAGUE_GAMES="${LEAGUE_GAMES:-160}"
export LEAGUE_TRAIN_EPOCHS="${LEAGUE_TRAIN_EPOCHS:-120}"
export SUBMIT_WIN_RATE="${SUBMIT_WIN_RATE:-0.50}"
export TARGET_WIN_RATE="${TARGET_WIN_RATE:-0.500001}"
export KAGGLE_COMPETITION="${KAGGLE_COMPETITION:-pokemon-tcg-ai-battle}"
export MAX_KAGGLE_SUBMISSIONS_PER_DAY="${MAX_KAGGLE_SUBMISSIONS_PER_DAY:-5}"

DECK_QUEUE="${DECK_QUEUE:-crustle,starmie,abomasnow,lucario,dragapult}"
if [[ -z "${BASE_CHECKPOINT:-}" ]]; then
  if [[ -f outputs/checkpoints/crustle_local_long/iter_001.pt ]]; then
    BASE_CHECKPOINT="outputs/checkpoints/crustle_local_long/iter_001.pt"
  elif [[ -f outputs/checkpoints/crustle_local_long/seed.pt ]]; then
    BASE_CHECKPOINT="outputs/checkpoints/crustle_local_long/seed.pt"
  else
    BASE_CHECKPOINT="outputs/checkpoints/temporal_current.pt"
  fi
fi
if [[ ! -f "$BASE_CHECKPOINT" && -f outputs/checkpoints/temporal_current.pt ]]; then
  BASE_CHECKPOINT="outputs/checkpoints/temporal_current.pt"
fi
if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  echo "missing base checkpoint: $BASE_CHECKPOINT" >&2
  exit 1
fi

deck_path_for() {
  case "$1" in
    crustle) echo "baselines/kaggle_public/dashimaki_day1_crustle/deck.csv" ;;
    starmie) echo "decks/competitive/high_performing/2026-04_regional-prague-2026_32nd_starmie-froslass.csv" ;;
    abomasnow)
      find decks baselines -type f -iname '*abomasnow*.csv' 2>/dev/null | sort | head -1
      ;;
    lucario) echo "baselines/kaggle_public/kojimar_simple_lucario/deck.csv" ;;
    dragapult) echo "baselines/kaggle_public/skarin_dragapult/deck.csv" ;;
    *)
      echo ""
      ;;
  esac
}

latest_checkpoint_in() {
  local checkpoint_dir="$1"
  "$python_bin" - "$checkpoint_dir" <<'PY'
import sys
from pathlib import Path

checkpoint_dir = Path(sys.argv[1])
candidates = [
    path for pattern in ("iter_*.pt", "league_*.pt", "*_latest.pt")
    for path in checkpoint_dir.glob(pattern)
    if path.is_file()
]
if candidates:
    print(max(candidates, key=lambda path: path.stat().st_mtime))
PY
}

next_iteration_for() {
  local checkpoint_dir="$1"
  "$python_bin" - "$checkpoint_dir" <<'PY'
import re
import sys
from pathlib import Path

latest = 0
for path in Path(sys.argv[1]).glob("iter_*.pt"):
    match = re.search(r"iter_(\d+)\.pt$", path.name)
    if match:
        latest = max(latest, int(match.group(1)))
print(latest + 1)
PY
}

latest_summary_json() {
  local report_dir="$1"
  "$python_bin" - "$report_dir" <<'PY'
import json
import re
import sys
from pathlib import Path

best = None
best_iter = -1
for path in Path(sys.argv[1]).glob("remote_public_baseline_iter_*.json"):
    match = re.search(r"iter_(\d+)\.json$", path.name)
    if not match:
        continue
    iteration = int(match.group(1))
    if iteration > best_iter:
        best_iter = iteration
        best = path
if best is None:
    print(json.dumps({"path": "", "win_rate": 0.0, "iteration": 0}))
else:
    data = json.loads(best.read_text(encoding="utf-8"))
    print(json.dumps({
        "path": str(best),
        "win_rate": float(data.get("win_rate", 0.0)),
        "iteration": best_iter,
    }))
PY
}

today_submission_count() {
  local tmp
  tmp="$(mktemp)"
  if ! kaggle competitions submissions -c "$KAGGLE_COMPETITION" -v >"$tmp" 2>/dev/null; then
    rm -f "$tmp"
    return 1
  fi
  "$python_bin" - "$tmp" <<'PY'
import csv
import datetime as dt
import sys

today_utc = dt.datetime.utcnow().date().isoformat()
today_local = dt.datetime.now().date().isoformat()
count = 0
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        value = (row.get("date") or "").strip()
        if value.startswith(today_utc) or value.startswith(today_local):
            count += 1
print(count)
PY
  rm -f "$tmp"
}

checkpoint_already_submitted() {
  local checkpoint="$1"
  [[ -f "$SUBMISSION_LOG" ]] || return 1
  "$python_bin" - "$SUBMISSION_LOG" "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path

log = Path(sys.argv[1])
checkpoint = str(Path(sys.argv[2]))
for line in log.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if row.get("checkpoint") == checkpoint and row.get("status") == "submitted":
        raise SystemExit(0)
raise SystemExit(1)
PY
}

record_submission_event() {
  local status="$1"
  local deck_key="$2"
  local deck_path="$3"
  local checkpoint="$4"
  local tarball="$5"
  local win_rate="$6"
  local message="$7"
  local output="$8"
  "$python_bin" - "$SUBMISSION_LOG" "$status" "$deck_key" "$deck_path" "$checkpoint" "$tarball" "$win_rate" "$message" "$output" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
row = {
    "time_utc": datetime.now(timezone.utc).isoformat(),
    "status": sys.argv[2],
    "deck": sys.argv[3],
    "deck_path": sys.argv[4],
    "checkpoint": sys.argv[5],
    "tarball": sys.argv[6],
    "win_rate": float(sys.argv[7]),
    "message": sys.argv[8],
    "output": sys.argv[9],
}
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
PY
}

submit_checkpoint_if_allowed() {
  local deck_key="$1"
  local deck_path="$2"
  local checkpoint="$3"
  local win_rate="$4"
  local timestamp
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local tarball="${SUBMISSION_DIR}/${deck_key}_${timestamp}.tar.gz"
  local message="two-week ${deck_key} baseline win $(printf '%.1f' "$(python3 - <<PY
print(float("$win_rate") * 100.0)
PY
)")%"

  if checkpoint_already_submitted "$checkpoint"; then
    echo "submission skipped: checkpoint already submitted ($checkpoint)"
    record_submission_event "skipped_duplicate" "$deck_key" "$deck_path" "$checkpoint" "$tarball" "$win_rate" "$message" ""
    return 0
  fi

  echo "building Kaggle submission snapshot for $deck_key"
  VALUE_MODEL_PATH="$checkpoint" SUBMISSION_DECK_PATH="$deck_path" scripts/build_submission.sh
  cp dist/submission.tar.gz "$tarball"
  echo "snapshot: $tarball"

  local count
  if ! count="$(today_submission_count)"; then
    echo "submission skipped: could not read Kaggle submission count"
    record_submission_event "skipped_quota_unknown" "$deck_key" "$deck_path" "$checkpoint" "$tarball" "$win_rate" "$message" ""
    return 0
  fi
  echo "Kaggle submissions today: $count/$MAX_KAGGLE_SUBMISSIONS_PER_DAY"
  if [[ "$count" -ge "$MAX_KAGGLE_SUBMISSIONS_PER_DAY" ]]; then
    echo "submission skipped: daily Kaggle cap reached"
    record_submission_event "skipped_daily_cap" "$deck_key" "$deck_path" "$checkpoint" "$tarball" "$win_rate" "$message" ""
    return 0
  fi

  local output
  if output="$(kaggle competitions submit -c "$KAGGLE_COMPETITION" -f "$tarball" -m "$message" 2>&1)"; then
    echo "$output"
    record_submission_event "submitted" "$deck_key" "$deck_path" "$checkpoint" "$tarball" "$win_rate" "$message" "$output"
  else
    echo "$output" >&2
    record_submission_event "submit_failed" "$deck_key" "$deck_path" "$checkpoint" "$tarball" "$win_rate" "$message" "$output"
  fi
}

echo "queue:          $DECK_QUEUE"
echo "base checkpoint: $BASE_CHECKPOINT"
echo "submit above:  $SUBMIT_WIN_RATE"
echo "target break:  $TARGET_WIN_RATE"
echo "max submits:   $MAX_KAGGLE_SUBMISSIONS_PER_DAY/day"
echo

IFS=',' read -r -a deck_keys <<< "$DECK_QUEUE"
cycle=0
while [[ "$(date +%s)" -lt "$END_EPOCH" ]]; do
  cycle="$((cycle + 1))"
  echo
  echo "################################################################################"
  echo "# deck queue cycle $cycle"
  echo "################################################################################"

  deck_index=0
  for raw_key in "${deck_keys[@]}"; do
    deck_key="$(echo "$raw_key" | xargs)"
    [[ -z "$deck_key" ]] && continue

    if [[ "$(date +%s)" -ge "$END_EPOCH" ]]; then
      echo "time limit reached before deck $deck_key"
      break
    fi

    deck_path="$(deck_path_for "$deck_key")"
    if [[ -z "$deck_path" || ! -f "$deck_path" ]]; then
      echo "skipping $deck_key: no local deck file found"
      deck_index="$((deck_index + 1))"
      continue
    fi

    checkpoint_dir="${QUEUE_ROOT}/checkpoints/two_week_queue/${deck_key}"
    rollout_dir="${QUEUE_ROOT}/rollouts/two_week_queue/${deck_key}"
    report_dir="${QUEUE_ROOT}/reports/two_week_queue/${deck_key}"
    rollouts="${rollout_dir}/rollouts.jsonl"
    mkdir -p "$checkpoint_dir" "$rollout_dir" "$report_dir"

    checkpoint="$(latest_checkpoint_in "$checkpoint_dir")"
    if [[ -z "$checkpoint" ]]; then
      checkpoint="$BASE_CHECKPOINT"
    fi
    start_iteration="$(next_iteration_for "$checkpoint_dir")"
    port="$((BASE_POLICY_PORT + deck_index))"
    deck_index="$((deck_index + 1))"

    echo
    echo "================================================================================"
    echo "$deck_key"
    echo "deck:          $deck_path"
    echo "checkpoint in: $checkpoint"
    echo "start iter:    $start_iteration"
    echo "port:          $port"
    echo "time left:     $((END_EPOCH - $(date +%s))) seconds"
    echo "================================================================================"

    if POLICY_PORT="$port" \
      RUN_UNTIL_EPOCH="$END_EPOCH" \
      START_ITERATION="$start_iteration" \
      CHECKPOINT="$checkpoint" \
      CHECKPOINT_DIR="$checkpoint_dir" \
      ROLLOUTS="$rollouts" \
      REPORT_DIR="$report_dir" \
      DECK="$deck_path" \
      ./scripts/run_remote_public_baseline_rl.sh; then
      :
    else
      echo "$deck_key run failed; continuing to next deck after a short pause" >&2
      sleep 60
      continue
    fi

    summary="$(latest_summary_json "$report_dir")"
    latest_win_rate="$("$python_bin" - "$summary" <<'PY'
import json
import sys
print(json.loads(sys.argv[1]).get("win_rate", 0.0))
PY
)"
    latest_checkpoint="$(latest_checkpoint_in "$checkpoint_dir")"
    echo "$deck_key latest baseline win_rate=$latest_win_rate"
    echo "$deck_key latest checkpoint=${latest_checkpoint:-none}"

    if "$python_bin" - "$latest_win_rate" "$SUBMIT_WIN_RATE" <<'PY'
import sys
win = float(sys.argv[1])
threshold = float(sys.argv[2])
raise SystemExit(0 if win > threshold else 1)
PY
    then
      if [[ -n "$latest_checkpoint" ]]; then
        submit_checkpoint_if_allowed "$deck_key" "$deck_path" "$latest_checkpoint" "$latest_win_rate"
      else
        echo "submission skipped: no checkpoint for $deck_key"
      fi
      echo "$deck_key cleared submit threshold; moving to next deck"
    else
      echo "$deck_key has not cleared submit threshold; continuing queue if time remains"
    fi
  done
done

echo
echo "Two-week deck queue complete"
echo "ended: $(date)"
