#!/bin/bash
# Stable launchd entrypoint for Bert's remote self-play worker.
#
# POKEBOT_BERT_MEMORY_GUARD_V2
# Bert's production backend is selected only from completed exact-H10
# whole-game receipts. The current topology uses four CPU leaves with four
# threads each; this beat 1t, 2t, 8t, and optimized MPS in whole-game GPS.
# Keep the whole tree in an isolated process group and stop it before launchd
# is allowed to retry if a guard is crossed.
set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin"

repo="${1:-/Users/example/workspace/poke-bot-agent}"
port="${POKEBOT_BERT_WORKER_PORT:-8766}"
checkpoint="$repo/outputs/checkpoints/pure_rl_bootstrap_current.pt"
python="$repo/.venv/bin/python"
worker="$repo/scripts/run_remote_worker.py"
cg_lib="$repo/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg"

# Export both the POKEBOT-scoped setting and its legacy unprefixed alias so the
# guard survives mixed-version rollback/deployment windows.
export REMOTE_REQUEST_TIMEOUT_S="120"
export POKEBOT_REMOTE_REQUEST_TIMEOUT_S="120"
export POKEBOT_REMOTE_WORKER_SAFETY_VERSION="20260717"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.25"
export PYTORCH_MPS_LOW_WATERMARK_RATIO="0.20"
export POKEBOT_MPS_EMPTY_CACHE_EVERY_BATCHES="0"
export POKEBOT_MPS_AUTOCAST_DTYPE="bfloat16"

# One thread per inference process prevents BLAS oversubscription. Sixteen
# simulator processes and four CPU policy leaves keep the 14-core M4 supplied
# while the supervisor retains the 18 GiB tree cap and 30% free-memory floor.
# Environment overrides make fleet tuning a service-owned configuration change
# instead of another source edit.
cpu_threads="${POKEBOT_BERT_CPU_THREADS:-4}"
export OMP_NUM_THREADS="$cpu_threads"
export MKL_NUM_THREADS="$cpu_threads"
export OPENBLAS_NUM_THREADS="$cpu_threads"
export VECLIB_MAXIMUM_THREADS="$cpu_threads"
export NUMEXPR_NUM_THREADS="1"
sim_workers="${POKEBOT_BERT_SIM_WORKERS:-16}"
default_workers="${POKEBOT_BERT_DEFAULT_WORKERS:-16}"
leaf_servers="${POKEBOT_BERT_LEAF_SERVERS:-4}"
leaf_max_batch="${POKEBOT_BERT_LEAF_MAX_BATCH:-32}"
leaf_queue_depth="${POKEBOT_BERT_LEAF_QUEUE_DEPTH:-32}"
# Keep four complete waves admitted to the local process queue while a second
# bounded set of request handlers may be returning trajectories to Inzi.  The
# launchd plist owns the cap so changing it cannot be silently ignored here.
max_connections="${POKEBOT_REMOTE_MAX_CONNECTIONS:-150}"
max_service_jobs=0

# Recycle each libcg simulation process after a small, bounded number of games.
# Whole-service job-count rotation is disabled: closing admission while active
# games drained made the trainer exhaust retries.  Per-process recycling,
# per-batch MPS cache release, and the hard memory watchdog remain authoritative.
export POKEBOT_WORKER_RECYCLE_GAMES="32"
export WORKER_RECYCLE_GAMES="32"

max_group_rss_mib="${POKEBOT_BERT_MAX_GROUP_RSS_MIB:-18432}"
min_free_percent="${POKEBOT_BERT_MIN_FREE_PERCENT:-30}"
max_group_processes="${POKEBOT_BERT_MAX_GROUP_PROCESSES:-32}"
watchdog_interval_s="${POKEBOT_BERT_WATCHDOG_INTERVAL_S:-5}"
violation_samples="${POKEBOT_BERT_WATCHDOG_VIOLATION_SAMPLES:-2}"
takeover_timeout_s="${POKEBOT_BERT_TAKEOVER_TIMEOUT_S:-120}"
restart_window_s="${POKEBOT_BERT_RESTART_WINDOW_S:-3600}"
restart_limit="${POKEBOT_BERT_RESTART_LIMIT:-3}"
stable_reset_s="${POKEBOT_BERT_STABLE_RESET_S:-1800}"

state_dir="$repo/outputs/state/bert_worker_supervisor"
failure_file="$state_dir/failures.epoch"
active_pgid_file="$state_dir/active.pgid"
active_checkpoint_file="$state_dir/active-checkpoint.json"
seed_checkpoint_script="$repo/scripts/seed_remote_active_checkpoint.py"
# Trainer-staged Bert checkpoints always live in the native checkout.  The
# durable record is authoritative after the first reload, so a whole-service
# MPS rotation cannot silently fall back to the bootstrap pointer.
checkpoint_root="$repo"
export POKEBOT_REMOTE_ACTIVE_CHECKPOINT_FILE="$active_checkpoint_file"
export POKEBOT_REMOTE_CHECKPOINT_ROOT="$checkpoint_root"
runtime_marker_source="${POKEBOT_MATCHUP_RUNTIME_MARKER_SOURCE:-}"
runtime_tree_source="${POKEBOT_PUBLIC_MATCHUP_TREE_SOURCE:-}"
worker_pid=""
worker_pgid=""
stop_requested=0

log() {
  printf '[%s] [bert-launchd] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

# The bounded request pool uses up to 128 trainer sockets in addition to
# multiprocessing queues/pipes. Fail before starting if launchd did not apply
# the plist's explicit descriptor budget.
open_file_limit="$(ulimit -n)"
if [[ "$open_file_limit" != "unlimited" && "$open_file_limit" -lt 1024 ]]; then
  log "open-file limit too low for queued sockets: $open_file_limit < 1024"
  exit 64
fi
log "open-file limit=$open_file_limit (max_connections=$max_connections)"

require_uint() {
  local name="$1"
  local value="$2"
  case "$value" in
    ''|*[!0-9]*) log "$name must be an unsigned integer, got '$value'"; exit 64 ;;
  esac
}

for setting in \
  "max_connections:$max_connections" \
  "max_group_rss_mib:$max_group_rss_mib" \
  "min_free_percent:$min_free_percent" \
  "max_group_processes:$max_group_processes" \
  "watchdog_interval_s:$watchdog_interval_s" \
  "violation_samples:$violation_samples" \
  "takeover_timeout_s:$takeover_timeout_s" \
  "restart_window_s:$restart_window_s" \
  "restart_limit:$restart_limit" \
  "stable_reset_s:$stable_reset_s"; do
  require_uint "${setting%%:*}" "${setting#*:}"
done
if [[ "$min_free_percent" -gt 100 || "$restart_limit" -lt 1 || \
      "$violation_samples" -lt 1 || "$watchdog_interval_s" -lt 1 ]]; then
  log "invalid watchdog/circuit-breaker bounds"
  exit 64
fi

mkdir -p "$state_dir"

prune_failures() {
  local now cutoff tmp
  now="$(date +%s)"
  cutoff=$((now - restart_window_s))
  tmp="$(mktemp "$state_dir/failures.XXXXXX")"
  if [[ -f "$failure_file" ]]; then
    awk -v cutoff="$cutoff" '$1 ~ /^[0-9]+$/ && $1 >= cutoff { print $1 }' \
      "$failure_file" >"$tmp"
  fi
  mv -f "$tmp" "$failure_file"
}

failure_count() {
  awk 'NF { count += 1 } END { print count + 0 }' "$failure_file"
}

record_failure() {
  prune_failures
  date +%s >>"$failure_file"
}

prune_failures
recent_failures="$(failure_count)"
circuit_open=0
if [[ "$recent_failures" -ge "$restart_limit" ]]; then circuit_open=1; fi

group_exists() {
  ps -axo pgid= | awk -v pgid="$worker_pgid" \
    '$1 == pgid { found=1 } END { exit !found }'
}

reap_worker_parent_if_exited() {
  local state=""
  [[ -n "$worker_pid" ]] || return 0
  state="$(ps -o state= -p "$worker_pid" 2>/dev/null | tr -d ' ' || true)"
  if ! kill -0 "$worker_pid" 2>/dev/null || [[ -z "$state" || "$state" == Z* ]]; then
    wait "$worker_pid" 2>/dev/null || true
    worker_pid=""
  fi
}

clear_active_group_record() {
  local recorded=""
  if [[ -f "$active_pgid_file" ]]; then
    recorded="$(awk 'NR == 1 { print $1 }' "$active_pgid_file")"
  fi
  if [[ -n "$worker_pgid" && "$recorded" == "$worker_pgid" ]]; then
    rm -f "$active_pgid_file"
  fi
}

terminate_worker_group() {
  local reason="$1"
  local deadline
  [[ -n "$worker_pgid" ]] || return 0
  log "stopping complete worker process group pgid=$worker_pgid reason=$reason"
  kill -TERM -- "-$worker_pgid" 2>/dev/null || true
  deadline=$(( $(date +%s) + 20 ))
  reap_worker_parent_if_exited
  while group_exists && [[ "$(date +%s)" -lt "$deadline" ]]; do
    sleep 1
    reap_worker_parent_if_exited
  done
  if group_exists; then
    log "forcing surviving worker process group pgid=$worker_pgid"
    kill -KILL -- "-$worker_pgid" 2>/dev/null || true
    sleep 1
  fi
  if group_exists; then
    log "worker process group pgid=$worker_pgid survived SIGKILL"
    return 1
  fi
  clear_active_group_record
}

handle_stop() {
  local signal="$1"
  stop_requested=1
  terminate_worker_group "received $signal" || true
  exit 0
}

# A hard-killed prior supervisor may leave its isolated worker group behind.
# The durable PGID record lets the next supervisor/deployer find that group
# even after run_remote_worker.py itself has died and its leaves were reparented.
if [[ -f "$active_pgid_file" ]]; then
  stale_pgid="$(awk 'NR == 1 { print $1 }' "$active_pgid_file")"
  case "$stale_pgid" in
    ''|*[!0-9]*)
      log "invalid stale worker PGID record; refusing to launch"
      if [[ "$circuit_open" -eq 1 ]]; then
        log "restart circuit open; exiting cleanly so launchd stays down"
        exit 0
      fi
      record_failure
      exit 69
      ;;
  esac
  worker_pgid="$stale_pgid"
  if group_exists; then
    current_pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
    if [[ "$worker_pgid" == "$current_pgid" ]] || ! ps -axo pgid=,command= | \
      awk -v pgid="$worker_pgid" -v repo="$repo" '
        $1 == pgid {
          seen=1
          line=$0
          if (index(line, repo) > 0 ||
              line ~ /multiprocessing[.](spawn|resource_tracker)/) {
            worker=1
          } else {
            foreign=1
          }
        }
        END { exit !(seen && worker && !foreign) }
      '; then
      log "stale PGID $worker_pgid is not an exclusively identifiable Bert worker group; refusing to signal it"
      worker_pgid=""
      if [[ "$circuit_open" -eq 1 ]]; then
        log "restart circuit open; exiting cleanly so launchd stays down"
        exit 0
      fi
      record_failure
      exit 69
    fi
    terminate_worker_group "stale group from prior supervisor" || {
      if [[ "$circuit_open" -eq 1 ]]; then
        log "restart circuit open after stale-group cleanup failure; leaving launchd down"
        exit 0
      fi
      record_failure
      exit 70
    }
  else
    clear_active_group_record
  fi
  worker_pgid=""
fi

if [[ "$circuit_open" -eq 1 ]]; then
  log "restart circuit open: $recent_failures failures in ${restart_window_s}s; exiting cleanly so launchd stays down"
  exit 0
fi

for required in "$python" "$worker" "$cg_lib" "$seed_checkpoint_script" "$checkpoint_root"; do
  if [[ ! -e "$required" ]]; then
    log "required path is missing: $required"
    record_failure
    exit 66
  fi
done

takeover_started="$(date +%s)"
waited_pid=""
while :; do
  listeners="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [[ -z "$listeners" ]]; then
    break
  fi

  listener_count="$(printf '%s\n' "$listeners" | awk 'NF { count += 1 } END { print count + 0 }')"
  if [[ "$listener_count" -ne 1 ]]; then
    log "refusing takeover while $listener_count listeners occupy :$port"
    record_failure
    exit 69
  fi
  listener_pid="$(printf '%s\n' "$listeners" | awk 'NF { print; exit }')"
  listener_command="$(ps -ww -p "$listener_pid" -o command= 2>/dev/null || true)"
  case "$listener_command" in
    *"$worker"*"--port $port"*) ;;
    *)
      log "refusing takeover from unexpected :$port listener pid=$listener_pid"
      record_failure
      exit 69
      ;;
  esac
  if [[ "$waited_pid" != "$listener_pid" ]]; then
    log "waiting for existing worker pid=$listener_pid before launchd takeover"
    waited_pid="$listener_pid"
  fi
  now="$(date +%s)"
  if [[ $((now - takeover_started)) -ge "$takeover_timeout_s" ]]; then
    log "takeover timed out after ${takeover_timeout_s}s; existing worker was left untouched"
    record_failure
    exit 75
  fi
  sleep 5
done

if [[ ! -f "$checkpoint" ]]; then
  log "stable checkpoint pointer is missing or invalid: $checkpoint"
  record_failure
  exit 66
fi

# First boot seeds from the stable bootstrap pointer.  Every later invocation
# validates and preserves the last successfully reloaded path+digest instead
# of overwriting it with the bootstrap checkpoint.
if ! "$python" "$seed_checkpoint_script" --checkpoint "$checkpoint"; then
  log "durable active-checkpoint seed/validation failed"
  record_failure
  exit 78
fi

# The trainer content-addresses Bert checkpoints inside their original run
# directory. Runtime routing is armed by an adjacent, checksum-verified marker,
# so materialize the canonical marker/tree beside the durable active checkpoint
# before the worker imports Torch or spawns any simulator.
if [[ -n "$runtime_marker_source" || -n "$runtime_tree_source" ]]; then
  if [[ -z "$runtime_marker_source" || -z "$runtime_tree_source" ]]; then
    log "both matchup runtime companion sources must be configured"
    record_failure
    exit 78
  fi
  if ! "$python" - "$active_checkpoint_file" "$checkpoint_root" \
      "$runtime_marker_source" "$runtime_tree_source" <<'PY'
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

state_file, root_raw, marker_raw, tree_raw = map(Path, sys.argv[1:])
root = root_raw.expanduser().resolve(strict=True)
state = json.loads(state_file.read_text(encoding="utf-8"))
checkpoint = Path(str(state.get("path") or "")).expanduser().resolve(strict=True)
checkpoint.relative_to(root)
marker_source = marker_raw.expanduser().resolve(strict=True)
tree_source = tree_raw.expanduser().resolve(strict=True)
marker = json.loads(marker_source.read_text(encoding="utf-8"))
tree_digest = "sha256:" + hashlib.sha256(tree_source.read_bytes()).hexdigest()
if not (
    marker.get("schema") == "poke_bot.remote_matchup_runtime_activation/v1"
    and marker.get("runtime_enabled") is True
    and marker.get("continuous_reevaluation") is True
    and marker.get("one_route_per_decision") is True
    and marker.get("tree_digest") == tree_digest
    and marker.get("tree_file") == tree_source.name
):
    raise SystemExit("invalid matchup runtime companion contract")
for source, name in (
    (tree_source, tree_source.name),
    (marker_source, "matchup-runtime-activation.json"),
):
    destination = checkpoint.parent / name
    temporary = destination.with_name(destination.name + f".partial.{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
PY
  then
    log "failed to stage matchup runtime companions beside active checkpoint"
    record_failure
    exit 78
  fi
  log "staged digest-verified matchup runtime companions beside active checkpoint"
fi

trap 'handle_stop TERM' TERM
trap 'handle_stop INT' INT
trap 'handle_stop HUP' HUP
trap 'terminate_worker_group "supervisor exit" || true' EXIT

log "starting validated CPU-4t fleet worker on :$port workers=$sim_workers default=$default_workers leaf_servers=$leaf_servers threads=$cpu_threads checkpoint=$(readlink "$checkpoint" 2>/dev/null || printf '%s' "$checkpoint")"
cd "$repo"

# setsid places the parent, pool workers, resource tracker, manager, and MPS
# leaf in one killable process group. If job control already made the child a
# group leader, that group is already isolated and setsid is unnecessary.
"$python" -c '
import os
import sys
try:
    os.setsid()
except PermissionError:
    if os.getpgrp() != os.getpid():
        raise
os.execv(sys.argv[1], sys.argv[1:])
' "$python" -u "$worker" \
  --host 0.0.0.0 --port "$port" \
  --workers "$sim_workers" --default-workers "$default_workers" \
  --leaf-servers "$leaf_servers" --leaf-gpu cpu \
  --leaf-max-batch "$leaf_max_batch" --leaf-queue-depth "$leaf_queue_depth" \
  --leaf-coalesce-ms "2" \
  --max-connections "$max_connections" \
  --tree-rss-limit-gb "45" --min-free-ram-gb "8" \
  --max-service-jobs "$max_service_jobs" --watchdog-interval-s "5" \
  --checkpoint "$checkpoint" \
  --cg-lib-path "$cg_lib" &
worker_pid=$!
worker_pgid="$worker_pid"
active_pgid_tmp="${active_pgid_file}.tmp.$$"
printf '%s\n' "$worker_pgid" >"$active_pgid_tmp"
mv -f "$active_pgid_tmp" "$active_pgid_file"
started_at="$(date +%s)"
consecutive_violations=0
failure_state_cleared=0

while :; do
  worker_state="$(ps -o state= -p "$worker_pid" 2>/dev/null | tr -d ' ' || true)"
  if ! kill -0 "$worker_pid" 2>/dev/null || \
     [[ -z "$worker_state" || "$worker_state" == Z* ]]; then
    set +e
    wait "$worker_pid"
    child_rc=$?
    set -e
    worker_pid=""
    terminate_worker_group "worker exited rc=$child_rc" || true
    trap - EXIT
    if [[ "$stop_requested" -eq 1 ]]; then
      exit 0
    fi
    if [[ "$child_rc" -eq 0 ]]; then
      runtime_s=$(( $(date +%s) - started_at ))
      if [[ "$runtime_s" -lt 60 ]]; then
        record_failure
        log "worker exited cleanly but too quickly (${runtime_s}s); treating as restart failure"
        # KeepAlive.SuccessfulExit=false deliberately restarts only non-zero exits.
        exit 75
      fi
      log "controlled service rotation after ${runtime_s}s; restarting guarded worker"
      # Avoid launchd's failure throttle for a planned MPS cache rotation. The
      # next invocation still checks the same persisted failure circuit first.
      sleep 5
      exec "$0" "$repo"
    fi
    record_failure
    log "worker failed rc=$child_rc; circuit count=$(failure_count)/$restart_limit"
    exit "$child_rc"
  fi

  stats="$(ps -axo pgid=,rss= | awk -v pgid="$worker_pgid" '
    $1 == pgid { count += 1; rss_kib += $2 }
    END { print count + 0, rss_kib + 0 }
  ')"
  process_count="${stats%% *}"
  rss_kib="${stats#* }"
  rss_mib=$((rss_kib / 1024))
  free_percent="$(memory_pressure -Q 2>/dev/null | awk '
    /System-wide memory free percentage:/ {
      value=$NF; gsub(/%/, "", value); print int(value); exit
    }
  ' || true)"

  violation=""
  if [[ "$rss_mib" -ge "$max_group_rss_mib" ]]; then
    violation="group_rss=${rss_mib}MiB>=${max_group_rss_mib}MiB"
  elif [[ "$process_count" -gt "$max_group_processes" ]]; then
    violation="group_processes=$process_count>${max_group_processes}"
  elif [[ -n "$free_percent" && "$free_percent" -lt "$min_free_percent" ]]; then
    violation="system_free=${free_percent}%<${min_free_percent}%"
  fi

  if [[ -n "$violation" ]]; then
    consecutive_violations=$((consecutive_violations + 1))
    log "memory guard sample $consecutive_violations/$violation_samples: $violation"
  else
    consecutive_violations=0
  fi

  if [[ "$consecutive_violations" -ge "$violation_samples" ]]; then
    terminate_worker_group "memory guard tripped: $violation" || true
    trap - EXIT
    record_failure
    log "memory guard stopped worker; circuit count=$(failure_count)/$restart_limit"
    exit 75
  fi

  now="$(date +%s)"
  if [[ "$failure_state_cleared" -eq 0 && $((now - started_at)) -ge "$stable_reset_s" ]]; then
    : >"$failure_file"
    failure_state_cleared=1
    log "stable window reached; restart circuit history cleared"
  fi
  sleep "$watchdog_interval_s"
done
