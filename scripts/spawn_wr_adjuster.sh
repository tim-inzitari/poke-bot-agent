#!/usr/bin/env bash
# Spawn a Cursor agent to diagnose / fix hammer RL win-rate stagnation.
#
# Called by scripts/winrate_watcher.py on alert. Prefer GPT-5.6 Sol Max Fast,
# falling back to Auto when the preferred model fails due to quota / billing /
# rate-limit / usage-limit errors.
#
# Usage:
#   scripts/spawn_wr_adjuster.sh <prompt_file> [log_file]
#
# Env overrides:
#   WR_ADJUSTER_PREFERRED_MODEL   default: gpt-5.6-sol-max-fast
#   WR_ADJUSTER_FALLBACK_MODEL    default: auto
#   WR_ADJUSTER_WORKSPACE         default: repo root (parent of scripts/)
#   WR_ADJUSTER_AGENT_BIN         default: agent (on PATH) or ~/.local/bin/agent
#   WR_ADJUSTER_LOCK              default: <workspace>/outputs/eval/WR_ADJUSTER.lock
#   CURSOR_API_KEY                optional; agent CLI also uses stored login

set -u

PROMPT_FILE="${1:-}"
LOG_FILE="${2:-}"

die() {
  echo "[spawn_wr_adjuster] ERROR: $*" >&2
  exit 1
}

[[ -n "$PROMPT_FILE" ]] || die "usage: $0 <prompt_file> [log_file]"
[[ -f "$PROMPT_FILE" ]] || die "prompt file missing: $PROMPT_FILE"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="${WR_ADJUSTER_WORKSPACE:-$REPO_ROOT}"
PREFERRED="${WR_ADJUSTER_PREFERRED_MODEL:-gpt-5.6-sol-max-fast}"
FALLBACK="${WR_ADJUSTER_FALLBACK_MODEL:-auto}"
LOCK="${WR_ADJUSTER_LOCK:-$WORKSPACE/outputs/eval/WR_ADJUSTER.lock}"
AGENT_BIN="${WR_ADJUSTER_AGENT_BIN:-}"

if [[ -z "$AGENT_BIN" ]]; then
  if command -v agent >/dev/null 2>&1; then
    AGENT_BIN="$(command -v agent)"
  elif [[ -x "$HOME/.local/bin/agent" ]]; then
    AGENT_BIN="$HOME/.local/bin/agent"
  else
    die "Cursor 'agent' CLI not found on PATH or ~/.local/bin/agent"
  fi
fi

mkdir -p "$(dirname "$LOCK")"
if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
fi

log() {
  local line="[$(date '+%Y-%m-%d %H:%M:%S')] [spawn_wr_adjuster] $*"
  echo "$line"
  if [[ -n "$LOG_FILE" ]]; then
    echo "$line" >>"$LOG_FILE" || true
  fi
}

# Debounce: single in-flight adjuster.
if [[ -f "$LOCK" ]]; then
  old_pid="$(awk '/^pid=/{print substr($0,5); exit}' "$LOCK" 2>/dev/null || true)"
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    log "SKIP: adjuster already in-flight (pid=$old_pid lock=$LOCK)"
    exit 0
  fi
  log "stale lock found (pid=${old_pid:-unknown}); reclaiming $LOCK"
fi

is_quota_error() {
  local text="$1"
  # Case-insensitive match for common billing / quota / rate-limit failures.
  echo "$text" | tr '[:upper:]' '[:lower:]' | grep -Eq \
    'usage limit|rate.?limit|quota|payment required|billing|out of tokens|insufficient.*(credit|quota|balance)|plan limit|too many requests|429|402|403.*(quota|usage|billing)|model.*(unavailable|not available).*quota|exceeded.*limit'
}

run_agent() {
  local model="$1"
  local out_file="$2"
  local prompt
  prompt="$(cat "$PROMPT_FILE")"

  log "spawning agent model=$model workspace=$WORKSPACE bin=$AGENT_BIN"
  # --print + --trust: non-interactive headless. --force: allow shell/write
  # tools so the adjuster can actually intervene (revert ckpt, tweak args, etc.).
  "$AGENT_BIN" -p \
    --model "$model" \
    --trust \
    --force \
    --workspace "$WORKSPACE" \
    --output-format text \
    "$prompt" >"$out_file" 2>&1
}

TMP_OUT="$(mktemp -t wr_adjuster_out.XXXXXX)"
trap 'rm -f "$TMP_OUT"' EXIT

{
  echo "pid=$$"
  echo "started=$(date -Iseconds)"
  echo "preferred_model=$PREFERRED"
  echo "fallback_model=$FALLBACK"
  echo "prompt_file=$PROMPT_FILE"
  echo "status=starting"
} >"$LOCK"

MODEL_USED=""
STATUS="failed"

if run_agent "$PREFERRED" "$TMP_OUT"; then
  MODEL_USED="$PREFERRED"
  STATUS="ok"
  log "SUCCESS: preferred model=$PREFERRED"
else
  rc=$?
  out="$(cat "$TMP_OUT" 2>/dev/null || true)"
  log "preferred model=$PREFERRED failed rc=$rc"
  if [[ -n "$LOG_FILE" ]]; then
    {
      echo "----- preferred model failure ($PREFERRED) -----"
      echo "$out" | tail -n 80
      echo "----- end -----"
    } >>"$LOG_FILE" || true
  fi

  if is_quota_error "$out"; then
    log "quota/billing/rate-limit detected; falling back to model=$FALLBACK"
    if run_agent "$FALLBACK" "$TMP_OUT"; then
      MODEL_USED="$FALLBACK"
      STATUS="ok_fallback"
      log "SUCCESS: fallback model=$FALLBACK (after preferred quota failure)"
    else
      rc2=$?
      out2="$(cat "$TMP_OUT" 2>/dev/null || true)"
      log "FAILURE: fallback model=$FALLBACK also failed rc=$rc2"
      if [[ -n "$LOG_FILE" ]]; then
        {
          echo "----- fallback model failure ($FALLBACK) -----"
          echo "$out2" | tail -n 80
          echo "----- end -----"
        } >>"$LOG_FILE" || true
      fi
      STATUS="failed_fallback"
    fi
  else
    log "FAILURE: preferred model failed for non-quota reason; NOT falling back"
    STATUS="failed_preferred"
  fi
fi

{
  echo "pid=$$"
  echo "finished=$(date -Iseconds)"
  echo "model_used=${MODEL_USED:-none}"
  echo "status=$STATUS"
  echo "prompt_file=$PROMPT_FILE"
} >"$LOCK"

# Keep a short result excerpt for the watcher log.
if [[ -n "$LOG_FILE" && -s "$TMP_OUT" ]]; then
  {
    echo "----- adjuster output (model=${MODEL_USED:-none} status=$STATUS) -----"
    tail -n 40 "$TMP_OUT"
    echo "----- end -----"
  } >>"$LOG_FILE" || true
fi

# Clear lock when finished so a later alert can spawn again.
# (In-flight debounce uses live pid; completed runs leave a status snapshot.)
FINISHED_SNAP="${LOCK}.last"
cp -f "$LOCK" "$FINISHED_SNAP" 2>/dev/null || true
rm -f "$LOCK"

if [[ "$STATUS" == ok || "$STATUS" == ok_fallback ]]; then
  log "done status=$STATUS model_used=$MODEL_USED"
  exit 0
fi
log "done status=$STATUS model_used=${MODEL_USED:-none}"
exit 1
