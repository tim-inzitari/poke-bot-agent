#!/usr/bin/env bash
# Live one-line Pure-RL tqdm bar (in-place), plus a held-out win-rate
# progression line above it. Prefer *.progress.status; fall back to the
# last \\r segment of *.progress.log.
#
# Usage:
#   bash scripts/watch_pure_rl_progress.sh
#   bash scripts/watch_pure_rl_progress.sh outputs/logs/pure_rl_core.progress.status
#   bash scripts/watch_pure_rl_progress.sh "" "" outputs/pure_rl/<run_name>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TARGET="${1:-outputs/logs/pure_rl_core.progress.status}"
PROG_FALLBACK="${2:-outputs/logs/pure_rl_core.progress.log}"
RUN_DIR="${3:-}"
[[ "$TARGET" = /* ]] || TARGET="$ROOT/$TARGET"
[[ "$PROG_FALLBACK" = /* ]] || PROG_FALLBACK="$ROOT/$PROG_FALLBACK"

PY="$(command -v python3 || true)"
WR_TREND_SCRIPT="$ROOT/scripts/pure_rl_wr_trend.py"
WR_TREND_ARGS=()
[[ -n "$RUN_DIR" ]] && WR_TREND_ARGS+=(--run-dir "$RUN_DIR")

_wr_trend() {
  if [[ -n "$PY" && -f "$WR_TREND_SCRIPT" ]]; then
    "$PY" "$WR_TREND_SCRIPT" "${WR_TREND_ARGS[@]}" 2>/dev/null || echo "wr_trend: (unavailable)"
  fi
}

if [[ ! -t 1 ]]; then
  # Non-interactive: print once and exit (for scripts / proof).
  _wr_trend
  if [[ -f "$TARGET" ]]; then
    cat "$TARGET"
  elif [[ -f "$PROG_FALLBACK" ]]; then
    tail -c 500 "$PROG_FALLBACK" | tr '\r' '\n' | sed '/^$/d' | tail -1
  else
    echo "no progress status/log yet: $TARGET" >&2
    exit 1
  fi
  exit 0
fi

echo "Watching in-place bar + win-rate progression (Ctrl-C to stop)"
echo "  status: $TARGET"
echo "  log:    $PROG_FALLBACK  (also: less -r +F \"$PROG_FALLBACK\")"
echo

# Prefer watch(1); fall back to a tiny shell loop.
if command -v watch >/dev/null 2>&1; then
  WR_CMD="'$PY' '$WR_TREND_SCRIPT'"
  for a in "${WR_TREND_ARGS[@]}"; do WR_CMD+=" '$a'"; done
  if [[ -f "$TARGET" ]] || [[ ! -f "$PROG_FALLBACK" ]]; then
    exec watch -n1 -t "$WR_CMD 2>/dev/null; echo; cat '$TARGET' 2>/dev/null || echo '(waiting for status…)'"
  fi
  exec watch -n1 -t "$WR_CMD 2>/dev/null; echo; tail -c 500 '$PROG_FALLBACK' | tr '\\r' '\\n' | sed '/^\$/d' | tail -1"
fi

while true; do
  printf '\033[H\033[2J'
  _wr_trend
  echo
  if [[ -f "$TARGET" ]]; then
    cat "$TARGET"
  elif [[ -f "$PROG_FALLBACK" ]]; then
    tail -c 500 "$PROG_FALLBACK" | tr '\r' '\n' | sed '/^$/d' | tail -1
  else
    echo "(waiting for progress…)"
  fi
  sleep 1
done
