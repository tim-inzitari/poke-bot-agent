#!/usr/bin/env bash
# Durable production supervisor for the Elmo remote worker.
#
# The worker owns normal process-tree cleanup. This wrapper adds two lifecycle
# guarantees around it:
#   * the reserved planned-rotation exit resumes without consuming failures;
#   * every other exit, including an unclean container restart, consumes a
#     durable failure slot before another worker may start.
set -euo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

readonly REQUIRED_PLANNED_EXIT_CODE=75
state_dir="${POKEBOT_ELMO_SUPERVISOR_STATE_DIR:-/workspace/runtime-logs/elmo-supervisor}"
failure_file="$state_dir/failures.epoch"
attempt_file="$state_dir/active.attempt"
circuit_file="$state_dir/circuit.state"
rotation_file="$state_dir/last_rotation.epoch"

planned_exit_code="${POKEBOT_REMOTE_PLANNED_ROTATION_EXIT_CODE:-}"
restart_limit="${POKEBOT_ELMO_RESTART_LIMIT:-3}"
restart_window_s="${POKEBOT_ELMO_RESTART_WINDOW_S:-3600}"
failure_backoff_s="${POKEBOT_ELMO_FAILURE_BACKOFF_S:-30}"
rotation_delay_s="${POKEBOT_ELMO_ROTATION_DELAY_S:-10}"
min_rotation_runtime_s="${POKEBOT_ELMO_MIN_ROTATION_RUNTIME_S:-60}"
child_stop_grace_s="${POKEBOT_ELMO_CHILD_STOP_GRACE_S:-75}"
session_launcher_python="${POKEBOT_ELMO_SESSION_LAUNCHER_PYTHON:-python}"

child_pid=""
child_pgid=""
pause_pid=""
stop_requested=0
child_group_forced=0
child_group_survived=0
child_reaped=0
child_rc=0

log() {
  printf '[%s] [elmo-supervisor] %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

require_uint() {
  local name="$1"
  local value="$2"
  case "$value" in
    ''|*[!0-9]*) log "$name must be an unsigned integer, got '$value'"; exit 64 ;;
  esac
}

if [[ "$#" -eq 0 ]]; then
  log "missing remote-worker command"
  exit 64
fi
if [[ "$planned_exit_code" != "$REQUIRED_PLANNED_EXIT_CODE" ]]; then
  log "POKEBOT_REMOTE_PLANNED_ROTATION_EXIT_CODE must be $REQUIRED_PLANNED_EXIT_CODE"
  exit 64
fi
for setting in \
  "restart_limit:$restart_limit" \
  "restart_window_s:$restart_window_s" \
  "failure_backoff_s:$failure_backoff_s" \
  "rotation_delay_s:$rotation_delay_s" \
  "min_rotation_runtime_s:$min_rotation_runtime_s" \
  "child_stop_grace_s:$child_stop_grace_s"; do
  require_uint "${setting%%:*}" "${setting#*:}"
done
if [[ "$restart_limit" -lt 1 || "$restart_window_s" -lt 60 || \
      "$child_stop_grace_s" -lt 1 || "$child_stop_grace_s" -gt 85 ]]; then
  log "restart_limit must be >=1, restart_window_s >=60, and child_stop_grace_s within 1..85"
  exit 64
fi
if ! command -v "$session_launcher_python" >/dev/null 2>&1; then
  log "session launcher Python is unavailable: $session_launcher_python"
  exit 69
fi

mkdir -p "$state_dir"
umask 077

write_atomic() {
  local destination="$1"
  shift
  local temporary="${destination}.tmp.$$"
  printf '%s\n' "$*" >"$temporary"
  mv -f "$temporary" "$destination"
}

prune_failures() {
  local now cutoff temporary
  now="$(date +%s)"
  cutoff=$((now - restart_window_s))
  temporary="$(mktemp "$state_dir/failures.XXXXXX")"
  if [[ -f "$failure_file" ]]; then
    awk -v cutoff="$cutoff" \
      '$1 ~ /^[0-9]+$/ && $1 >= cutoff { print $1 }' \
      "$failure_file" >"$temporary"
  fi
  mv -f "$temporary" "$failure_file"
}

failure_count() {
  awk 'NF { count += 1 } END { print count + 0 }' "$failure_file"
}

record_failure() {
  prune_failures
  date +%s >>"$failure_file"
}

open_circuit() {
  local reason="$1"
  local count
  count="$(failure_count)"
  write_atomic "$circuit_file" \
    "open epoch=$(date +%s) failures=$count limit=$restart_limit reason=$reason"
  log "restart circuit OPEN: failures=$count/$restart_limit reason=$reason"
  # Compose uses on-failure:3 as a final wrapper-crash guard. A clean exit is
  # deliberate: Docker must not be able to restart an open worker circuit.
  exit 0
}

mark_circuit_closed() {
  write_atomic "$circuit_file" \
    "closed epoch=$(date +%s) failures=$(failure_count) limit=$restart_limit"
}

mark_attempt_inactive() {
  local detail="$1"
  write_atomic "$attempt_file" "inactive epoch=$(date +%s) $detail"
}

child_group_exists() {
  [[ -n "$child_pgid" ]] || return 1
  kill -0 -- "-$child_pgid" 2>/dev/null
}

reap_child_parent_if_exited() {
  local process_state=""
  local wait_rc=0
  [[ -n "$child_pid" && "$child_reaped" -eq 0 ]] || return 0
  process_state="$(ps -o state= -p "$child_pid" 2>/dev/null | tr -d ' ' || true)"
  if ! kill -0 "$child_pid" 2>/dev/null || \
     [[ -z "$process_state" || "$process_state" == Z* ]]; then
    if wait "$child_pid" 2>/dev/null; then
      wait_rc=0
    else
      wait_rc=$?
    fi
    if [[ "$wait_rc" -ne 127 ]]; then
      child_rc="$wait_rc"
    fi
    child_reaped=1
  fi
}

terminate_child_group() {
  local reason="$1"
  local deadline
  child_group_forced=0
  child_group_survived=0
  [[ -n "$child_pgid" ]] || return 0
  reap_child_parent_if_exited
  if ! child_group_exists; then
    return 0
  fi

  log "stopping complete worker process group pgid=$child_pgid reason=$reason"
  kill -TERM -- "-$child_pgid" 2>/dev/null || true
  deadline=$(( $(date +%s) + child_stop_grace_s ))
  while child_group_exists && [[ "$(date +%s)" -lt "$deadline" ]]; do
    sleep 1
    reap_child_parent_if_exited
  done
  if child_group_exists; then
    child_group_forced=1
    log "forcing surviving worker process group pgid=$child_pgid"
    kill -KILL -- "-$child_pgid" 2>/dev/null || true
    sleep 1
    reap_child_parent_if_exited
  fi
  if child_group_exists; then
    child_group_survived=1
    log "worker process group pgid=$child_pgid survived SIGKILL"
    return 1
  fi
  return 0
}

handle_stop() {
  local signal_name="$1"
  stop_requested=1
  log "received $signal_name; forwarding shutdown"
  if [[ -n "$child_pgid" ]]; then
    kill -TERM -- "-$child_pgid" 2>/dev/null || true
  elif [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
  if [[ -n "$pause_pid" ]]; then
    kill -TERM "$pause_pid" 2>/dev/null || true
  fi
}

pause_interruptibly() {
  local duration="$1"
  [[ "$duration" -gt 0 ]] || return 0
  sleep "$duration" &
  pause_pid=$!
  if [[ "$stop_requested" -eq 1 ]]; then
    kill -TERM "$pause_pid" 2>/dev/null || true
  fi
  set +e
  wait "$pause_pid"
  set -e
  pause_pid=""
}

trap 'handle_stop TERM' TERM
trap 'handle_stop INT' INT
trap 'handle_stop HUP' HUP

prune_failures

# The attempt marker is written before every child start and cleared only
# after wait(2) returns. If the container, supervisor, or cgroup is killed,
# the next outer Docker restart converts that stale marker into a failure.
if [[ -f "$attempt_file" ]] && \
   awk 'NR == 1 && $1 == "active" { found=1 } END { exit !found }' \
     "$attempt_file"; then
  log "recovering unclean active attempt from a prior container lifetime"
  record_failure
  mark_attempt_inactive "recovered=unclean-container-exit"
fi

prune_failures
recent_failures="$(failure_count)"
if [[ "$recent_failures" -ge "$restart_limit" ]]; then
  open_circuit "failure-window-exhausted-before-start"
fi
mark_circuit_closed

while :; do
  if [[ "$stop_requested" -eq 1 ]]; then
    mark_attempt_inactive "reason=operator-stop"
    exit 0
  fi

  prune_failures
  recent_failures="$(failure_count)"
  if [[ "$recent_failures" -ge "$restart_limit" ]]; then
    open_circuit "failure-window-exhausted"
  fi

  started_at="$(date +%s)"
  child_group_forced=0
  child_group_survived=0
  child_reaped=0
  child_rc=0
  write_atomic "$attempt_file" \
    "active epoch=$started_at supervisor_pid=$$ command=$1"
  log "starting guarded worker failures=$recent_failures/$restart_limit command=$1"
  if [[ "$stop_requested" -eq 1 ]]; then
    mark_attempt_inactive "reason=operator-stop-before-child-start"
    exit 0
  fi

  set +e
  # The worker parent, multiprocessing manager, pool, resource tracker, and
  # CUDA leaves live in one isolated session. This lets TERM and the bounded
  # fallback target the whole service without signaling the supervisor.
  "$session_launcher_python" -c '
import os
import sys
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])
' "$@" &
  child_pid=$!
  child_pgid="$child_pid"
  waited_pid="$child_pid"
  # Close the small launch/trap race: a stop that arrived after the pre-launch
  # check but before PGID publication still reaches the newly isolated group.
  if [[ "$stop_requested" -eq 1 ]]; then
    kill -TERM -- "-$child_pgid" 2>/dev/null || \
      kill -TERM "$child_pid" 2>/dev/null || true
  fi
  write_atomic "$attempt_file" \
    "active epoch=$started_at supervisor_pid=$$ child_pgid=$child_pgid command=$1"
  wait "$child_pid"
  child_rc=$?
  process_state="$(ps -o state= -p "$child_pid" 2>/dev/null | tr -d ' ' || true)"
  if [[ -z "$process_state" ]]; then
    child_reaped=1
  fi
  residual_group_cleanup=0
  if child_group_exists; then
    if [[ "$stop_requested" -eq 0 ]]; then
      residual_group_cleanup=1
    fi
    terminate_child_group "worker parent exited rc=$child_rc" || true
  fi
  # A trapped TERM interrupts Bash's wait before Python necessarily finishes
  # its pool/leaf cleanup. Reap the real parent after group shutdown; 127 means
  # the first wait already reaped it and its original status remains valid.
  if [[ "$child_reaped" -eq 0 ]]; then
    if wait "$waited_pid" 2>/dev/null; then
      reaped_rc=0
    else
      reaped_rc=$?
    fi
    if [[ "$reaped_rc" -ne 127 ]]; then
      child_rc="$reaped_rc"
    fi
    child_reaped=1
  fi
  set -e
  child_pid=""
  child_pgid=""
  runtime_s=$(( $(date +%s) - started_at ))
  mark_attempt_inactive \
    "rc=$child_rc runtime_s=$runtime_s forced_group_kill=$child_group_forced"

  if [[ "$stop_requested" -eq 1 ]]; then
    log "worker stopped after forwarded operator signal"
    exit 0
  fi

  if [[ "$residual_group_cleanup" -eq 0 && \
        "$child_rc" -eq "$planned_exit_code" && \
        "$runtime_s" -ge "$min_rotation_runtime_s" ]]; then
    write_atomic "$rotation_file" \
      "$(date +%s) runtime_s=$runtime_s rc=$child_rc"
    mark_circuit_closed
    log "planned service rotation complete after ${runtime_s}s; resuming"
    pause_interruptibly "$rotation_delay_s"
    continue
  fi

  failure_reason="worker-exit-rc-$child_rc"
  if [[ "$child_group_survived" -eq 1 ]]; then
    failure_reason="worker-process-group-survived-sigkill"
  elif [[ "$residual_group_cleanup" -eq 1 ]]; then
    failure_reason="worker-left-process-group-rc-$child_rc"
  elif [[ "$child_rc" -eq "$planned_exit_code" ]]; then
    failure_reason="planned-code-before-min-runtime-${runtime_s}s"
  fi
  record_failure
  recent_failures="$(failure_count)"
  log "unexpected $failure_reason; circuit count=$recent_failures/$restart_limit"
  if [[ "$recent_failures" -ge "$restart_limit" ]]; then
    open_circuit "$failure_reason"
  fi
  mark_circuit_closed
  pause_interruptibly "$failure_backoff_s"
done
