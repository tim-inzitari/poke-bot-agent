#!/usr/bin/env bash
# Unattended foreground launcher for ONE archetype, sized for long (multi-day) runs that
# stay visible in a Cursor terminal panel. It sources tuned env, tees all output to a log,
# and auto-retries on crash (resuming self-play from the manifest once a checkpoint exists).
#
#   bash scripts/run_unattended.sh dragapult --ladder-only   # Blackwell
#   bash scripts/run_unattended.sh lucario                   # 3080 Ti
#
# Runs in the FOREGROUND (no tmux/nohup) so output streams live in Cursor. The GPU is pinned
# by scripts/run_archetype.sh from the archetype profile, so two archetypes can run in two
# terminals across two GPUs at the same time.
#
# Env knobs:
#   UNATTENDED_MAX_RETRIES     (default 3)   crash retries after the first attempt
#   UNATTENDED_RETRY_SLEEP_SEC (default 60)  sleep between retries
#   UNATTENDED_LOOP            (default 1)   after a clean finish, start another self-play segment
#   UNATTENDED_MAX_CYCLES      (default 100) max continuation segments (~2 week unattended)
set -o pipefail
cd "$(dirname "$0")/.."

ARCH="${1:-}"
if [[ -z "$ARCH" ]]; then
  echo "usage: $0 <lucario|dragapult|abomasnow|iono|starmie|crustle> [run_archetype flags]" >&2
  exit 1
fi
shift
ORIGINAL_ARGS=("$@")

MAX_RETRIES="${UNATTENDED_MAX_RETRIES:-3}"
RETRY_SLEEP_SEC="${UNATTENDED_RETRY_SLEEP_SEC:-60}"

mkdir -p outputs/logs
LOG="outputs/logs/${ARCH}_$(date +%Y%m%d_%H%M%S).log"

source_env() {
  local file="$1"
  if [[ -f "$file" ]]; then
    echo "==> sourcing env: $file"
    set -a
    # shellcheck disable=SC1090
    source "$file"
    set +a
  fi
}

kill_phantoms() {
  # Arch-scoped: only this archetype's previous launcher (safe for a concurrent run of a
  # different archetype on the other GPU).
  pkill -9 -f "run_archetype.sh ${ARCH}" 2>/dev/null || true
  # Stopped (T-state) trainers left behind by a prior Ctrl+Z; healthy concurrent runs are in
  # S/R state and are not matched.
  local stopped
  stopped="$(ps -eo pid,state,command | awk '$2 ~ /T/ && /train_agent\.py/ {print $1}')"
  if [[ -n "$stopped" ]]; then
    echo "==> killing stopped trainer pids: ${stopped//$'\n'/ }"
    # shellcheck disable=SC2086
    kill -9 $stopped 2>/dev/null || true
  fi
}

{
  echo "================================================================"
  echo "unattended run: ${ARCH}   args: ${ORIGINAL_ARGS[*]:-<none>}"
  echo "started: $(date)"
  echo "log: ${LOG}"
  echo "================================================================"

  source_env "configs/unattended.env"
  source_env "configs/unattended.${ARCH}.env"

  kill_phantoms

  UNATTENDED_LOOP="${UNATTENDED_LOOP:-1}"
  UNATTENDED_MAX_CYCLES="${UNATTENDED_MAX_CYCLES:-100}"
  attempt_args=("${ORIGINAL_ARGS[@]}")
  status=1
  cycle=1

  while [[ $cycle -le $UNATTENDED_MAX_CYCLES ]]; do
    echo ""
    echo "=== cycle ${cycle}/${UNATTENDED_MAX_CYCLES} args: run_archetype.sh ${ARCH} ${attempt_args[*]:-} ($(date)) ==="
    for ((try = 1; try <= MAX_RETRIES + 1; try++)); do
      echo ""
      echo "=== attempt ${try}/$((MAX_RETRIES + 1)): run_archetype.sh ${ARCH} ${attempt_args[*]:-} ($(date)) ==="
      bash scripts/run_archetype.sh "$ARCH" "${attempt_args[@]}"
      status=$?
      if [[ $status -eq 0 ]]; then
        echo "=== ${ARCH} finished cleanly on attempt ${try} (cycle ${cycle}) ($(date)) ==="
        break
      fi
      echo "=== ${ARCH} attempt ${try} failed (exit ${status}) at $(date) ==="
      if [[ $try -le $MAX_RETRIES ]]; then
        echo "=== sleeping ${RETRY_SLEEP_SEC}s before retry ==="
        sleep "$RETRY_SLEEP_SEC"
        if [[ -f "outputs/checkpoints/${ARCH}_fresh.pt" ]]; then
          attempt_args=(--self-play-only)
          echo "=== checkpoint present -> resuming with --self-play-only ==="
        else
          echo "=== no checkpoint yet -> retrying original args ==="
        fi
      fi
    done

    if [[ $status -ne 0 ]]; then
      break
    fi
    if [[ "$UNATTENDED_LOOP" != "1" ]] || [[ $cycle -ge $UNATTENDED_MAX_CYCLES ]]; then
      break
    fi
    if ! python3 scripts/prepare_self_play_continue.py "$ARCH"; then
      echo "=== ${ARCH}: no continuation prepared; stopping after cycle ${cycle} ==="
      break
    fi
    attempt_args=(--self-play-only)
    cycle=$((cycle + 1))
    echo "=== ${ARCH}: starting continuation cycle ${cycle}/${UNATTENDED_MAX_CYCLES} ($(date)) ==="
  done

  echo "=== ${ARCH} exiting with status ${status} after ${cycle} cycle(s) ($(date)) ==="
  exit $status
} 2>&1 | tee -a "$LOG"

exit "${PIPESTATUS[0]}"
