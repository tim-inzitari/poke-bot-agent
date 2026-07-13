#!/usr/bin/env bash
# Start (or ensure) the 2-week unattended stack: watchdog + status polling.
# Does NOT restart healthy run_unattended processes — only fills gaps after crashes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p outputs/logs

if ! pgrep -f 'scripts/watch_unattended.sh' >/dev/null 2>&1; then
  WATCH_INTERVAL_SEC="${WATCH_INTERVAL_SEC:-300}" \
    nohup bash "$ROOT/scripts/watch_unattended.sh" >> outputs/logs/watchdog.log 2>&1 &
  echo "started watchdog pid=$!"
else
  echo "watchdog already running"
fi

dragapult_args="--skip-merge"
[[ ! -s "$ROOT/data/dragapult_training.jsonl" ]] && dragapult_args="--ladder-only"

start_arch() {
  local arch="$1"
  shift
  if pgrep -f "run_unattended.sh ${arch}" >/dev/null 2>&1; then
    echo "${arch}: already running"
    return 0
  fi
  local logf="outputs/logs/${arch}_autostart_$(date +%Y%m%d_%H%M%S).log"
  nohup bash "$ROOT/scripts/run_unattended.sh" "$arch" "$@" >>"$logf" 2>&1 &
  echo "started ${arch} pid=$! log=${logf}"
}

start_arch dragapult $dragapult_args
start_arch lucario

echo ""
echo "Monitor: tail -f outputs/logs/watchdog.log outputs/UNATTENDED_STATUS.md"
