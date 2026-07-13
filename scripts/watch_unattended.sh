#!/usr/bin/env bash
# Autonomous watchdog for unattended dragapult + lucario runs.
# Polls process health, GPU/RAM, logs; restarts dead runs via run_unattended.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

INTERVAL="${WATCH_INTERVAL_SEC:-300}"
PRUNE_INTERVAL="${PRUNE_INTERVAL_SEC:-21600}"
LOG_DIR="$ROOT/outputs/logs"
STATUS_FILE="$ROOT/outputs/UNATTENDED_STATUS.md"
WATCH_LOG="$LOG_DIR/watchdog.log"
PRUNE_STAMP="$LOG_DIR/.last_prune_epoch"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCH_LOG"; }

is_running() {
  local arch="$1"
  pgrep -f "run_unattended.sh ${arch}" >/dev/null 2>&1 || \
    pgrep -f "run_archetype.sh ${arch}" >/dev/null 2>&1
}

has_stopped_train() {
  local arch="$1"
  pgrep -af "train_agent.py" 2>/dev/null | grep -q "${arch}_fresh" && \
    pgrep -af "train_agent.py" 2>/dev/null | grep "${arch}_fresh" | grep -q " T "
}

latest_log() {
  local arch="$1"
  ls -t "$LOG_DIR"/${arch}_*.log 2>/dev/null | head -1
}

tail_epoch() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  grep -oE 'training:.*epoch [0-9]+/[0-9]+' "$f" 2>/dev/null | tail -1 || true
}

read_segment() {
  local arch="$1"
  local state="$ROOT/outputs/logs/${arch}_continue_state.json"
  if [[ -f "$state" ]]; then
    python3 -c "import json; print(json.load(open('$state')).get('segments', 0))" 2>/dev/null || echo "0"
  else
    echo "0"
  fi
}

maybe_prune_disk() {
  local now last
  now=$(date +%s)
  last=0
  [[ -f "$PRUNE_STAMP" ]] && last=$(cat "$PRUNE_STAMP" 2>/dev/null || echo 0)
  if (( now - last < PRUNE_INTERVAL )); then
    return 0
  fi
  log "disk prune (interval=${PRUNE_INTERVAL}s)"
  if bash "$ROOT/scripts/prune_disk_unattended.sh" --force >>"$WATCH_LOG" 2>&1; then
    echo "$now" >"$PRUNE_STAMP"
  else
    log "WARN: disk prune failed"
  fi
}

write_status() {
  local drag_run luc_run drag_log luc_log drag_epoch luc_epoch drag_seg luc_seg
  drag_run="no"; luc_run="no"
  is_running dragapult && drag_run="yes"
  is_running lucario && luc_run="yes"
  drag_log="$(latest_log dragapult)"
  luc_log="$(latest_log lucario)"
  drag_epoch="$(tail_epoch "$drag_log")"
  luc_epoch="$(tail_epoch "$luc_log")"
  drag_seg="$(read_segment dragapult)"
  luc_seg="$(read_segment lucario)"

  {
    echo "# Unattended run status"
    echo ""
    echo "Updated: $(date -Is)"
    echo ""
    echo "Mode: **full run** — auto-restart on crash; auto-continue self-play segments when a cycle finishes early."
    echo ""
    echo "| Archetype | Running | Segments | Latest log | Progress |"
    echo "|-----------|---------|----------|------------|----------|"
    echo "| dragapult | ${drag_run} | ${drag_seg} | ${drag_log##*/} | ${drag_epoch:-—} |"
    echo "| lucario | ${luc_run} | ${luc_seg} | ${luc_log##*/} | ${luc_epoch:-—} |"
    echo ""
    echo "## GPU"
    echo '```'
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
    echo '```'
    echo ""
    echo "## Memory"
    echo '```'
    free -h | head -2
    echo '```'
    echo ""
    echo "## Watchdog"
    echo "- interval: ${INTERVAL}s"
    echo "- disk prune: every ${PRUNE_INTERVAL}s (tensor caches, old ckpts, rollout obs strip)"
    echo "- log: outputs/logs/watchdog.log"
    echo "- launcher: scripts/run_unattended.sh {dragapult|lucario}"
  } > "$STATUS_FILE"
}

restart_if_needed() {
  local arch="$1"
  local extra_args="${2:-}"
  if is_running "$arch"; then
    if has_stopped_train "$arch"; then
      log "WARN: ${arch} train_agent stopped (T); killing phantoms"
      pkill -9 -f "run_archetype.sh ${arch}" 2>/dev/null || true
      pkill -9 -f "train_agent.py.*${arch}_fresh" 2>/dev/null || true
      sleep 5
    else
      return 0
    fi
  fi
  if is_running "$arch"; then
    return 0
  fi
  log "RESTART: ${arch} not running; launching run_unattended.sh"
  local logf="$LOG_DIR/${arch}_watchdog_$(date +%Y%m%d_%H%M%S).log"
  nohup bash "$ROOT/scripts/run_unattended.sh" "$arch" $extra_args \
    >>"$logf" 2>&1 &
  log "started ${arch} pid=$! log=$logf"
}

dragapult_restart_args() {
  if [[ -s "$ROOT/data/dragapult_training.jsonl" ]]; then
    echo "--skip-merge"
  else
    echo "--ladder-only"
  fi
}

lucario_restart_args() {
  if [[ -f "$ROOT/outputs/checkpoints/lucario_fresh.pt" ]]; then
    echo "--self-play-only"
  fi
}

log "watchdog started (interval=${INTERVAL}s)"
while true; do
  write_status
  maybe_prune_disk
  restart_if_needed dragapult "$(dragapult_restart_args)"
  restart_if_needed lucario "$(lucario_restart_args)"
  sleep "$INTERVAL"
done
