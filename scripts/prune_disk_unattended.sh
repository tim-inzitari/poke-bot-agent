#!/usr/bin/env bash
# Periodic disk cleanup for long unattended runs. Safe to call from the watchdog.
set -euo pipefail
cd "$(dirname "$0")/.."

DISK_USE_TRIGGER="${DISK_USE_TRIGGER:-75}"
USE_PCT=$(df . | tail -1 | awk '{gsub(/%/,"",$5); print $5}')

if [[ "${1:-}" != "--force" ]] && [[ "$USE_PCT" -lt "$DISK_USE_TRIGGER" ]]; then
  echo "disk ${USE_PCT}% used (< ${DISK_USE_TRIGGER}% trigger); skip prune (use --force)"
  exit 0
fi

echo "==> unattended disk prune at $(date) (disk ${USE_PCT}%)"
KEEP_CACHE="${KEEP_CACHE:-2}"
KEEP_NAMED_CACHE="${KEEP_NAMED_CACHE:-1}"
KEEP_CKPT="${KEEP_CKPT:-8}"
PRUNE_MIN_AGE_SEC="${PRUNE_MIN_AGE_SEC:-7200}"
bash scripts/prune_outputs.sh --apply
echo "==> prune complete at $(date)"
