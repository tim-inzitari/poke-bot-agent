#!/usr/bin/env bash
# Coordinated code + schema-compatible checkpoint deployment to Elmo and Bert.
# Call only while the trainer is stopped or between collection waves.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${POKEBOT_PYTHON:-/home/pokebot/miniconda3/envs/poke-bot-agent/bin/python}"
LOG="${1:-$ROOT/outputs/logs/pure_rl_remote_redeploy.log}"
mkdir -p "$(dirname "$LOG")"
log() { echo "[$(date -Is)] [redeploy-self-play] $*" | tee -a "$LOG"; }

ELMO_HOST="${ELMO_SSH:-root@elmo}"
BERT_HOST="${BERT_SSH:-user@bert.local}"
BERT_REPO="${BERT_REPO:-/Users/example/workspace/poke-bot-agent}"
BERT_SERVICE_LABEL="com.pokebot.remote-worker-8766"
BERT_SERVICE_WRAPPER="scripts/run_bert_remote_worker_supervised.sh"
BERT_SERVICE_PLIST="deploy/launchd/${BERT_SERVICE_LABEL}.plist"
CONTAINER="${ELMO_CONTAINER:-poke-bot-truenas-worker}"
ELMO_CHECKPOINT_HOST="${ELMO_CHECKPOINT_HOST:-/mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker/checkpoint/model.pt}"
BOOTSTRAP="${REMOTE_BOOTSTRAP_CHECKPOINT:-}"
PREFLIGHT_PROFILE="${REMOTE_PREFLIGHT_PROFILE:-quick}"

if [[ -z "$BOOTSTRAP" || ! -f "$BOOTSTRAP" ]]; then
  log "ABORT set REMOTE_BOOTSTRAP_CHECKPOINT to a schema-compatible .pt file"
  exit 2
fi
BOOTSTRAP="$(readlink -f "$BOOTSTRAP")"
BOOT_DIGEST="$($PYTHON - "$BOOTSTRAP" <<'PY'
import sys
from poke_bot.checkpoint import checkpoint_digest
print(checkpoint_digest(sys.argv[1]))
PY
)"
BOOT_SHORT="${BOOT_DIGEST#sha256:}"

case "$PREFLIGHT_PROFILE" in
  none|quick|canary) ;;
  *) log "ABORT REMOTE_PREFLIGHT_PROFILE must be none, quick, or canary"; exit 2 ;;
esac

# This is intentionally before scp, docker cp, process termination, extraction,
# or checkpoint replacement. A digest alone does not prove that the new source
# can reconstruct the model or that its feature schema/profile matches.
log "preflight checkpoint load + trusted schema + pure-RL model profile"
EXPECTED_DIGEST="$BOOT_DIGEST" PREFLIGHT_PROFILE="$PREFLIGHT_PROFILE" \
  "$PYTHON" - "$BOOTSTRAP" <<'PY'
import os
import sys
from dataclasses import asdict

import torch

from poke_bot import checkpoint, features
from poke_bot.pure_rl.model_profile import (
    count_params,
    model_config_dict,
    validate_param_budget,
)
from poke_bot.train import load_model_from_checkpoint

path = sys.argv[1]
expected_digest = os.environ["EXPECTED_DIGEST"]
actual_digest = checkpoint.checkpoint_digest(path)
if actual_digest != expected_digest:
    raise SystemExit(
        f"checkpoint digest changed during preflight: {actual_digest} != {expected_digest}"
    )
trusted = checkpoint.assert_trusted_policy_checkpoint(path)
feature_schema = trusted["provenance"].get("feature_schema")
if feature_schema != features.FEATURE_SCHEMA_VERSION:
    raise SystemExit(
        f"feature schema mismatch: checkpoint={feature_schema!r} "
        f"source={features.FEATURE_SCHEMA_VERSION!r}"
    )
model = load_model_from_checkpoint(path, device=torch.device("cpu"))
actual_profile = asdict(model.cfg)
expected_profile = model_config_dict()
if os.environ["PREFLIGHT_PROFILE"] != "none" and actual_profile != expected_profile:
    keys = sorted(
        key
        for key in set(actual_profile) | set(expected_profile)
        if actual_profile.get(key) != expected_profile.get(key)
    )
    raise SystemExit(f"pure-RL model profile mismatch fields={keys}")
validate_param_budget(count_params(model))
print(
    f"preflight_ok digest={actual_digest} schema={feature_schema} "
    f"params={count_params(model)}"
)
PY
if [[ "$PREFLIGHT_PROFILE" != "none" ]]; then
  log "run preflight test profile=$PREFLIGHT_PROFILE"
  "$PYTHON" "$ROOT/scripts/run_test_profile.py" "$PREFLIGHT_PROFILE" --python "$PYTHON"
fi

sync_paths=(
  poke_bot
  scripts/run_remote_worker.py
  scripts/seed_remote_active_checkpoint.py
  scripts/train_round_robin.py
)
bert_service_paths=(
  "$BERT_SERVICE_WRAPPER"
  "$BERT_SERVICE_PLIST"
)
for path in "${sync_paths[@]}" "${bert_service_paths[@]}"; do
  [[ -e "$ROOT/$path" ]] || { log "ABORT missing $path"; exit 2; }
done
if command -v rg >/dev/null 2>&1; then
  rg -q 'self_play' "$ROOT/scripts/run_remote_worker.py" || {
    log "ABORT run_remote_worker.py lacks self_play support"
    exit 2
  }
else
  grep -q 'self_play' "$ROOT/scripts/run_remote_worker.py" || {
    log "ABORT run_remote_worker.py lacks self_play support"
    exit 2
  }
fi

DEPLOY_ID="$(date -u +%Y%m%dT%H%M%SZ).$$"
BUNDLE="/tmp/pokebot_remote_sync.$DEPLOY_ID.tar"
BERT_SERVICE_BUNDLE="/tmp/pokebot_bert_service.$DEPLOY_ID.tar"
BOOT_COPY="/tmp/pokebot_bootstrap.$DEPLOY_ID.pt"
ELMO_ACTIVATION_ATTEMPTED=0
BERT_ACTIVATION_ATTEMPTED=0
DEPLOY_COMMITTED=0
ELMO_PREVIOUS_DIGEST=""
BERT_PREVIOUS_DIGEST=""

rollback_elmo() {
  log "ROLLBACK Elmo deployment id=$DEPLOY_ID"
  ssh -o BatchMode=yes "$ELMO_HOST" bash -s -- \
    "$CONTAINER" "$DEPLOY_ID" "$ELMO_CHECKPOINT_HOST" <<'ELMOROLLBACK'
set -euo pipefail
container="$1"
deploy_id="$2"
host_checkpoint="$3"
marker="/tmp/pokebot_activation_${deploy_id}.marker"
[[ -f "$marker" ]] || exit 0
checkpoint_backup="${host_checkpoint}.before_${deploy_id}"
code_backup="/tmp/pokebot_workspace_before_${deploy_id}.tar"
active_pointer_backup="/tmp/pokebot_active_checkpoint_before_${deploy_id}.json"
active_pointer_existed="/tmp/pokebot_active_checkpoint_before_${deploy_id}.exists"
[[ -f "$checkpoint_backup" ]] || {
  echo "missing Elmo checkpoint rollback artifact $checkpoint_backup" >&2
  exit 40
}
docker exec "$container" test -f "$code_backup" || {
  echo "missing Elmo code rollback artifact $code_backup" >&2
  exit 41
}
restore_tmp="${host_checkpoint}.rollback.${deploy_id}"
cp -p "$checkpoint_backup" "$restore_tmp"
mv -f "$restore_tmp" "$host_checkpoint"
docker exec "$container" rm -rf /workspace/poke_bot
docker exec "$container" tar -xf "$code_backup" -C /workspace

# The supervisor's durable pointer is authoritative over model.pt at startup.
# Restore it as part of the same transaction, before restarting the old code.
mount_record="$(docker inspect --format '{{range .Mounts}}{{println .Destination "|" .Source "|" .RW}}{{end}}' "$container")"
active_checkpoint_file="$(
  docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$container" |
    sed -n 's/^POKEBOT_REMOTE_ACTIVE_CHECKPOINT_FILE=//p' | tail -n 1
)"
runtime_logs_host="$(
  awk -F ' \\| ' '$1 == "/workspace/runtime-logs" { print $2 }' <<<"$mount_record"
)"
case "$active_checkpoint_file" in
  /workspace/runtime-logs/*)
    [[ -n "$runtime_logs_host" ]] || {
      echo "cannot locate Elmo runtime-logs bind for active pointer rollback" >&2
      exit 42
    }
    active_checkpoint_host="${runtime_logs_host}/${active_checkpoint_file#/workspace/runtime-logs/}"
    ;;
  *)
    echo "unexpected Elmo active checkpoint file: $active_checkpoint_file" >&2
    exit 42
    ;;
esac
mkdir -p "$(dirname "$active_checkpoint_host")"
if [[ -f "$active_pointer_existed" ]]; then
  [[ -f "$active_pointer_backup" ]] || {
    echo "missing Elmo active-pointer rollback artifact $active_pointer_backup" >&2
    exit 43
  }
  active_restore_tmp="${active_checkpoint_host}.rollback.${deploy_id}"
  cp -p "$active_pointer_backup" "$active_restore_tmp"
  mv -f "$active_restore_tmp" "$active_checkpoint_host"
else
  rm -f "$active_checkpoint_host"
fi
docker restart "$container" >/dev/null
rm -f "$marker"
ELMOROLLBACK
}

rollback_bert() {
  log "ROLLBACK Bert deployment id=$DEPLOY_ID"
  ssh -o BatchMode=yes "$BERT_HOST" bash -s -- \
    "$BERT_REPO" "$DEPLOY_ID" <<'BERTROLLBACK'
set -euo pipefail
repo="$1"
deploy_id="$2"
marker="/tmp/pokebot_activation_${deploy_id}.marker"
[[ -f "$marker" ]] || exit 0
code_backup="/tmp/pokebot_workspace_before_${deploy_id}.tar"
cmd_backup="/tmp/pokebot_worker_cmds_before_${deploy_id}.json"
service_snapshot="/tmp/pokebot_bert_service_before_${deploy_id}"
service_label="com.pokebot.remote-worker-8766"
service_domain="gui/$(id -u)"
service_target="${service_domain}/${service_label}"
agent_plist="$HOME/Library/LaunchAgents/${service_label}.plist"
checkpoint_current="$repo/outputs/checkpoints/pure_rl_bootstrap_current.pt"
service_wrapper="$repo/scripts/run_bert_remote_worker_supervised.sh"
service_plist_source="$repo/deploy/launchd/${service_label}.plist"
active_pgid_file="$repo/outputs/state/bert_worker_supervisor/active.pgid"
active_checkpoint_file="$repo/outputs/state/bert_worker_supervisor/active-checkpoint.json"
failure_file="$repo/outputs/state/bert_worker_supervisor/failures.epoch"
arm_file="$repo/outputs/state/REMOTE_WORKER_ARMED"
rollback_worker_groups="$service_snapshot/rollback_worker_pgids"

worker_group_exists() {
  local pgid="$1"
  ps -axo pgid= | awk -v pgid="$pgid" \
    '$1 == pgid { found=1 } END { exit !found }'
}

worker_group_is_exclusive() {
  local pgid="$1"
  ps -axo pgid=,command= | awk -v pgid="$pgid" -v repo="$repo" '
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
  '
}

capture_worker_groups() {
  local destination="$1"
  local current_pgid candidate_pids pid pgid recorded
  current_pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  : >"$destination"
  candidate_pids="$(
    { pgrep -f '[r]un_remote_worker.py' || true; \
      lsof -nP -iTCP:8766 -sTCP:LISTEN -t 2>/dev/null || true; } | \
      awk 'NF && !seen[$1]++ { print $1 }'
  )"
  for pid in $candidate_pids; do
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    [[ -n "$pgid" ]] && printf '%s\n' "$pgid" >>"$destination"
  done
  if [[ -f "$active_pgid_file" ]]; then
    recorded="$(awk 'NR == 1 { print $1 }' "$active_pgid_file")"
    case "$recorded" in
      ''|*[!0-9]*)
        echo "invalid durable Bert worker PGID record: $recorded" >&2
        return 1
        ;;
      *) printf '%s\n' "$recorded" >>"$destination" ;;
    esac
  fi
  sort -nu "$destination" -o "$destination"
  while IFS= read -r pgid; do
    [[ -n "$pgid" ]] || continue
    if [[ "$pgid" == "$current_pgid" ]]; then
      echo "refusing to signal current deployment shell PGID $pgid" >&2
      return 1
    fi
    if worker_group_exists "$pgid" && ! worker_group_is_exclusive "$pgid"; then
      echo "refusing non-exclusive/stale Bert worker PGID $pgid" >&2
      return 1
    fi
  done <"$destination"
}

terminate_worker_groups() {
  local groups_file="$1"
  local context="$2"
  local deadline pgid any_alive remaining recorded
  while IFS= read -r pgid; do
    [[ -n "$pgid" ]] || continue
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done <"$groups_file"
  deadline=$(( $(date +%s) + 20 ))
  while [[ "$(date +%s)" -lt "$deadline" ]]; do
    any_alive=0
    while IFS= read -r pgid; do
      [[ -n "$pgid" ]] || continue
      if worker_group_exists "$pgid"; then any_alive=1; fi
    done <"$groups_file"
    [[ "$any_alive" -eq 0 ]] && break
    sleep 1
  done
  while IFS= read -r pgid; do
    [[ -n "$pgid" ]] || continue
    if worker_group_exists "$pgid"; then
      kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
  done <"$groups_file"
  sleep 1
  remaining=""
  while IFS= read -r pgid; do
    [[ -n "$pgid" ]] || continue
    if worker_group_exists "$pgid"; then remaining="$remaining $pgid"; fi
  done <"$groups_file"
  [[ -z "$remaining" ]] || {
    echo "Bert worker process groups survived $context:$remaining" >&2
    return 1
  }
  remaining="$(pgrep -f '[r]un_remote_worker.py' || true)"
  [[ -z "$remaining" ]] || {
    echo "Bert worker parents appeared during $context: $remaining" >&2
    return 1
  }
  if [[ -f "$active_pgid_file" ]]; then
    recorded="$(awk 'NR == 1 { print $1 }' "$active_pgid_file")"
    if ! worker_group_exists "$recorded"; then rm -f "$active_pgid_file"; fi
  fi
}

[[ -f "$code_backup" && -f "$cmd_backup" && \
   -f "$service_snapshot/mode" && \
   -f "$service_snapshot/enable_state" && \
   -f "$service_snapshot/repo_assets.tar" ]] || {
  echo "missing Bert rollback artifacts" >&2
  exit 50
}

# Capture the whole isolated group before launchd can reap/reparent its parent.
# The durable PGID also catches leaves left by a hard-killed supervisor.
capture_worker_groups "$rollback_worker_groups" || exit 51

# A KeepAlive job must be removed from launchd before killing its worker.  A
# plain kill would let launchd race the code/checkpoint rollback with a restart.
launchctl disable "$service_target"
launchctl bootout "$service_target" >/dev/null 2>&1 || true
terminate_worker_groups "$rollback_worker_groups" "rollback termination" || exit 51
rm -rf "$repo/poke_bot"
tar -xf "$code_backup" -C "$repo"

# Restore the repo-owned launchd assets, the installed plist, and the stable
# checkpoint pointer exactly as they were before this transaction.
rm -f "$service_wrapper" "$service_plist_source"
tar -xf "$service_snapshot/repo_assets.tar" -C "$repo"
mkdir -p "$(dirname "$agent_plist")" "$(dirname "$checkpoint_current")"
rm -f "$agent_plist"
if [[ -f "$service_snapshot/agent_plist.present" ]]; then
  cp -p "$service_snapshot/agent_plist" "$agent_plist"
fi
rm -f "$checkpoint_current"
if [[ -f "$service_snapshot/checkpoint_current.present" ]]; then
  cp -Pp "$service_snapshot/checkpoint_current" "$checkpoint_current"
fi
mkdir -p "$(dirname "$active_checkpoint_file")"
rm -f "$active_checkpoint_file"
if [[ -f "$service_snapshot/active_checkpoint.present" ]]; then
  active_checkpoint_restore="${active_checkpoint_file}.rollback.${deploy_id}"
  cp -p "$service_snapshot/active_checkpoint" "$active_checkpoint_restore"
  mv -f "$active_checkpoint_restore" "$active_checkpoint_file"
fi
rm -f "$failure_file"
if [[ -f "$service_snapshot/failure_file.present" ]]; then
  cp -p "$service_snapshot/failure_file" "$failure_file"
fi
rm -f "$arm_file"
if [[ -f "$service_snapshot/arm_file.present" ]]; then
  cp -p "$service_snapshot/arm_file" "$arm_file"
fi

previous_mode="$(cat "$service_snapshot/mode")"
previous_enable_state="$(cat "$service_snapshot/enable_state")"
case "$previous_enable_state" in
  enabled|disabled) ;;
  *) echo "unknown prior Bert launchd enable state: $previous_enable_state" >&2; exit 54 ;;
esac
if [[ "$previous_mode" == "launchd" ]]; then
  [[ -f "$agent_plist" ]] || {
    echo "cannot restore prior Bert launchd service without $agent_plist" >&2
    exit 52
  }
  # Never revive a pre-guard 20-worker service merely to make rollback look
  # successful. A stopped Bert is safer than rolling back into host-wide OOM.
  if ! grep -Eq 'POKEBOT_BERT_MEMORY_GUARD_V(1|2)' "$service_wrapper" || \
     ! grep -Fq 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' "$agent_plist"; then
    launchctl disable "$service_target"
    echo "refusing to restore Bert launchd service without memory guard" >&2
    exit 55
  fi
  launchctl enable "$service_target"
  launchctl bootstrap "$service_domain" "$agent_plist"
  launchctl print "$service_target" >/dev/null
elif [[ "$previous_mode" == "detached" ]]; then
  # One-time migration fallback: preserve the historical command shape, but
  # clamp it to Bert's guarded canary capacity and allocator limits.
  "$repo/.venv/bin/python" - "$repo" "$cmd_backup" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
commands = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if len(commands) != 1:
    raise SystemExit(f"rollback requires exactly one worker command, got {len(commands)}")


def force_option(command, option, value):
    command = list(command)
    found = False
    for index, item in enumerate(command):
        if item == option:
            if index + 1 >= len(command):
                raise SystemExit(f"rollback command has no value for {option}")
            command[index + 1] = value
            found = True
    if not found:
        command.extend([option, value])
    return command


command = list(commands[0])
for option, value in (
    ("--workers", "4"),
    ("--default-workers", "2"),
    ("--leaf-servers", "1"),
    ("--leaf-max-batch", "32"),
    ("--leaf-queue-depth", "8"),
):
    command = force_option(command, option, value)
log = (repo / "outputs/logs/remote_worker_bert.log").open("ab", buffering=0)
process = subprocess.Popen(
    command,
    cwd=repo,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
    env={
        **os.environ,
        "REMOTE_REQUEST_TIMEOUT_S": "120",
        "POKEBOT_REMOTE_REQUEST_TIMEOUT_S": "120",
        "POKEBOT_REMOTE_WORKER_SAFETY_VERSION": "20260717",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.25",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.20",
        "POKEBOT_WORKER_RECYCLE_GAMES": "32",
        "WORKER_RECYCLE_GAMES": "32",
    },
)
state_dir = repo / "outputs/state/bert_worker_supervisor"
state_dir.mkdir(parents=True, exist_ok=True)
active_tmp = state_dir / f"active.pgid.tmp.{os.getpid()}"
active_tmp.write_text(f"{process.pid}\n", encoding="ascii")
active_tmp.replace(state_dir / "active.pgid")
PY
else
  echo "unknown prior Bert service mode: $previous_mode" >&2
  exit 53
fi
if [[ "$previous_enable_state" == "disabled" ]]; then
  launchctl disable "$service_target"
fi
rm -f "$marker"
BERTROLLBACK
}

wait_for_digest() {
  local endpoint="$1"
  local expected="$2"
  for _ in $(seq 1 60); do
    if ENDPOINT="$endpoint" EXPECTED_DIGEST="$expected" "$PYTHON" - <<'PY' 2>/dev/null
import os
from poke_bot.remote_jobs import RemoteJobClient, parse_endpoint

c = RemoteJobClient(
    *parse_endpoint(os.environ["ENDPOINT"]),
    timeout_s=5,
    connect_timeout_s=5,
    control_timeout_s=5,
)
try:
    info = c.connect()
    health = c.health()
    # Rollback may intentionally restore the pre-hardening worker whose health
    # frame has no top-level `ok`; exact hello/health digest plus live leaves was
    # that protocol's strongest identity proof. Final deployment gates below
    # remain strict and require the new per-leaf identity schema.
    schema_health_ok = health.get("ok") is True or (
        health.get("ok") is None and health.get("leaf_alive") is True
    )
    ok = (
        schema_health_ok
        and info.checkpoint_digest == os.environ["EXPECTED_DIGEST"]
        and health.get("checkpoint_digest") == os.environ["EXPECTED_DIGEST"]
    )
finally:
    c.close()
raise SystemExit(0 if ok else 1)
PY
    then
      return 0
    fi
    sleep 2
  done
  return 1
}

cleanup_transaction() {
  local rc=$?
  trap - EXIT
  set +e
  rm -f "$BUNDLE" "$BERT_SERVICE_BUNDLE"
  if [[ "$rc" -ne 0 && "$DEPLOY_COMMITTED" -ne 1 ]]; then
    bert_rollback_rc=0
    elmo_rollback_rc=0
    if [[ "$BERT_ACTIVATION_ATTEMPTED" -eq 1 ]]; then
      rollback_bert
      bert_rollback_rc=$?
    fi
    if [[ "$ELMO_ACTIVATION_ATTEMPTED" -eq 1 ]]; then
      rollback_elmo
      elmo_rollback_rc=$?
    fi
    if [[ "$BERT_ACTIVATION_ATTEMPTED" -eq 1 && -n "$BERT_PREVIOUS_DIGEST" ]]; then
      wait_for_digest "bert.local:8766" "$BERT_PREVIOUS_DIGEST"
      bert_verify_rc=$?
    else
      bert_verify_rc=0
    fi
    if [[ "$ELMO_ACTIVATION_ATTEMPTED" -eq 1 && -n "$ELMO_PREVIOUS_DIGEST" ]]; then
      wait_for_digest "elmo:8765" "$ELMO_PREVIOUS_DIGEST"
      elmo_verify_rc=$?
    else
      elmo_verify_rc=0
    fi
    log "ROLLBACK_RESULT original_rc=$rc bert=$bert_rollback_rc/$bert_verify_rc elmo=$elmo_rollback_rc/$elmo_verify_rc"
  fi
  exit "$rc"
}
trap cleanup_transaction EXIT
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$BUNDLE" "${sync_paths[@]}"
tar -cf "$BERT_SERVICE_BUNDLE" "${bert_service_paths[@]}"
log "bundle ready id=$DEPLOY_ID checkpoint=$BOOT_DIGEST"

current_remote_digest() {
  local endpoint="$1"
  ENDPOINT="$endpoint" "$PYTHON" - <<'PY'
import os
from poke_bot.remote_jobs import RemoteJobClient, parse_endpoint

c = RemoteJobClient(
    *parse_endpoint(os.environ["ENDPOINT"]),
    timeout_s=10,
    connect_timeout_s=10,
    control_timeout_s=10,
)
try:
    info = c.connect()
    health = c.health()
    schema_health_ok = health.get("ok") is True or (
        health.get("ok") is None and health.get("leaf_alive") is True
    )
    if not schema_health_ok:
        raise SystemExit(f"current endpoint is unhealthy: {health}")
    digest = health.get("checkpoint_digest")
    if not digest or digest != info.checkpoint_digest:
        raise SystemExit(f"current endpoint identity mismatch: hello={info.checkpoint_digest} health={digest}")
    print(digest)
finally:
    c.close()
PY
}
ELMO_PREVIOUS_DIGEST="$(current_remote_digest 'elmo:8765')"
BERT_PREVIOUS_DIGEST="$(current_remote_digest 'bert.local:8766')"
log "transaction snapshot Elmo=$ELMO_PREVIOUS_DIGEST Bert=$BERT_PREVIOUS_DIGEST"

# Bert's MPS worker is a per-user LaunchAgent.  Fail before mutating Elmo if
# the logged-in GUI domain needed for launchd supervision is unavailable.
ssh -o BatchMode=yes "$BERT_HOST" bash -s <<'BERTLAUNCHDPREFLIGHT'
set -euo pipefail
service_domain="gui/$(id -u)"
launchctl print "$service_domain" >/dev/null
[[ -d "$HOME/Library/LaunchAgents" && -w "$HOME/Library/LaunchAgents" ]]
BERTLAUNCHDPREFLIGHT
log "Bert launchd GUI domain preflight ok"

log "stage and preflight coherent code/bootstrap on Elmo container $CONTAINER"
scp -o BatchMode=yes "$BUNDLE" "$ELMO_HOST:$BUNDLE"
scp -o BatchMode=yes "$BOOTSTRAP" "$ELMO_HOST:$BOOT_COPY"
ELMO_ACTIVATION_ATTEMPTED=1
ssh -o BatchMode=yes "$ELMO_HOST" bash -s -- \
  "$CONTAINER" "$BUNDLE" "$BOOT_COPY" "$DEPLOY_ID" \
  "$ELMO_CHECKPOINT_HOST" "$BOOT_DIGEST" "$PREFLIGHT_PROFILE" <<'ELMO'
set -euo pipefail
container="$1"
bundle="$2"
bootstrap="$3"
deploy_id="$4"
host_checkpoint="$5"
expected_digest="$6"
preflight_profile="$7"
stage="/tmp/pokebot_remote_preflight_${deploy_id}"

[[ -f "$host_checkpoint" ]] || {
  echo "Elmo host checkpoint source is missing: $host_checkpoint" >&2
  exit 20
}

# Verify that the configured host path is the source of the container's
# read-only checkpoint bind. Replacing a path inside the container is invalid
# for this deployment and can appear to succeed while leaving the bind intact.
mount_record="$(docker inspect --format '{{range .Mounts}}{{println .Destination "|" .Source "|" .RW}}{{end}}' "$container")"
if ! grep -Fq "/workspace/checkpoint/model.pt | $host_checkpoint | false" <<<"$mount_record"; then
  host_checkpoint_dir="$(dirname "$host_checkpoint")"
  if ! grep -Fq "/workspace/checkpoint | $host_checkpoint_dir | false" <<<"$mount_record"; then
    echo "Elmo checkpoint bind does not match expected read-only host source" >&2
    echo "$mount_record" >&2
    exit 21
  fi
fi

# Snapshot the durable startup pointer before any mutation.  It lives on the
# writable runtime-logs bind and must be committed/rolled back with model.pt.
active_checkpoint_file="$(
  docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$container" |
    sed -n 's/^POKEBOT_REMOTE_ACTIVE_CHECKPOINT_FILE=//p' | tail -n 1
)"
runtime_logs_host="$(
  awk -F ' \\| ' '$1 == "/workspace/runtime-logs" { print $2 }' <<<"$mount_record"
)"
case "$active_checkpoint_file" in
  /workspace/runtime-logs/*)
    [[ -n "$runtime_logs_host" ]] || {
      echo "cannot locate Elmo runtime-logs bind for active pointer activation" >&2
      exit 21
    }
    active_checkpoint_host="${runtime_logs_host}/${active_checkpoint_file#/workspace/runtime-logs/}"
    ;;
  *)
    echo "unexpected Elmo active checkpoint file: $active_checkpoint_file" >&2
    exit 21
    ;;
esac
active_pointer_backup="/tmp/pokebot_active_checkpoint_before_${deploy_id}.json"
active_pointer_existed="/tmp/pokebot_active_checkpoint_before_${deploy_id}.exists"
rm -f "$active_pointer_backup" "$active_pointer_existed"
if [[ -f "$active_checkpoint_host" ]]; then
  cp -p "$active_checkpoint_host" "$active_pointer_backup"
  touch "$active_pointer_existed"
fi

# Target-side preflight uses the staged source, not the currently resident
# /workspace modules. Nothing persistent is changed until this load succeeds.
docker cp "$bundle" "$container:/tmp/remote_sync_${deploy_id}.tar"
docker exec "$container" rm -rf "$stage"
docker exec "$container" mkdir -p "$stage"
docker exec "$container" tar -xf "/tmp/remote_sync_${deploy_id}.tar" -C "$stage"
docker cp "$bootstrap" "$container:$stage/bootstrap.pt"
docker exec "$container" python -m py_compile "$stage/scripts/run_remote_worker.py"
docker exec -i -w "$stage" -e PYTHONPATH="$stage" \
  -e EXPECTED_DIGEST="$expected_digest" -e PREFLIGHT_PROFILE="$preflight_profile" \
  "$container" python - "$stage/bootstrap.pt" <<'PY'
import os
import sys
from dataclasses import asdict
import torch
from poke_bot import checkpoint, features
from poke_bot.pure_rl.model_profile import count_params, model_config_dict, validate_param_budget
from poke_bot.train import load_model_from_checkpoint

path = sys.argv[1]
actual = checkpoint.checkpoint_digest(path)
if actual != os.environ["EXPECTED_DIGEST"]:
    raise SystemExit(f"digest mismatch: {actual}")
trusted = checkpoint.assert_trusted_policy_checkpoint(path)
schema = trusted["provenance"].get("feature_schema")
if schema != features.FEATURE_SCHEMA_VERSION:
    raise SystemExit(f"feature schema mismatch: {schema} != {features.FEATURE_SCHEMA_VERSION}")
model = load_model_from_checkpoint(path, device=torch.device("cpu"))
if os.environ["PREFLIGHT_PROFILE"] != "none" and asdict(model.cfg) != model_config_dict():
    raise SystemExit("pure-RL model profile mismatch")
validate_param_budget(count_params(model))
print(f"elmo_preflight_ok digest={actual} schema={schema}")
PY

# The checkpoint is a read-only container bind. Publish on the host filesystem
# with a same-directory rename, retaining the exact previous source as backup.
backup="${host_checkpoint}.before_${deploy_id}"
tmp="${host_checkpoint}.tmp.${deploy_id}"
code_backup="/tmp/pokebot_workspace_before_${deploy_id}.tar"
marker="/tmp/pokebot_activation_${deploy_id}.marker"
docker exec "$container" tar -cf "$code_backup" -C /workspace \
  poke_bot scripts/run_remote_worker.py scripts/train_round_robin.py
cp -p "$host_checkpoint" "$backup"
cp "$bootstrap" "$tmp"
actual="sha256:$(sha256sum "$tmp" | awk '{print $1}')"
if [[ "$actual" != "$expected_digest" ]]; then
  rm -f "$tmp"
  echo "Elmo staged checkpoint digest mismatch: $actual != $expected_digest" >&2
  exit 22
fi
touch "$marker"
mv -f "$tmp" "$host_checkpoint"

# Publish the already-preflighted staged source and restart exactly once into
# the coherent source/checkpoint pair.
docker exec "$container" rm -rf /workspace/poke_bot
docker exec "$container" tar -xf "/tmp/remote_sync_${deploy_id}.tar" -C /workspace
# Atomically make the new bound checkpoint the supervisor's durable selection.
# _persist_active_checkpoint re-hashes the file and validates that it remains
# inside POKEBOT_REMOTE_CHECKPOINT_ROOT before publishing the JSON pointer.
docker exec -i -w /workspace -e EXPECTED_DIGEST="$expected_digest" \
  "$container" python - /workspace/checkpoint/model.pt <<'PY'
import os
import sys

sys.path.insert(0, os.environ["POKEBOT_REMOTE_CHECKPOINT_ROOT"])
from scripts.run_remote_worker import _persist_active_checkpoint

published = _persist_active_checkpoint(sys.argv[1], os.environ["EXPECTED_DIGEST"])
if published is None:
    raise SystemExit("Elmo did not configure a durable active checkpoint file")
print(f"elmo_active_checkpoint_published={published}")
PY
docker restart "$container" >/dev/null
ELMO

elmo_ok=0
for _ in $(seq 1 120); do
  if EXPECTED_DIGEST="$BOOT_DIGEST" "$PYTHON" - <<'PY' 2>/dev/null
import os
from poke_bot.remote_jobs import RemoteJobClient, parse_endpoint

c = RemoteJobClient(*parse_endpoint("elmo:8765"), timeout_s=10)
info = c.connect()
health = c.health()
c.close()
ok = (
    "self_play" in info.job_kinds
    and health.get("ok") is True
    and health.get("leaf_alive") is True
    and health.get("leaf_identity_ok") is True
    and health.get("checkpoint_digest") == os.environ["EXPECTED_DIGEST"]
    and len(health.get("leaves") or []) == info.leaf_servers
    and info.leaf_servers > 0
    and all(
        leaf.get("healthy") is True
        and leaf.get("checkpoint_digest") == os.environ["EXPECTED_DIGEST"]
        and leaf.get("version") == health.get("checkpoint_version")
        for leaf in (health.get("leaves") or [])
    )
)
raise SystemExit(0 if ok else 1)
PY
  then
    elmo_ok=1
    log "Elmo ready on the requested checkpoint"
    break
  fi
  sleep 2
done
[[ "$elmo_ok" == 1 ]] || { log "ABORT Elmo failed readiness"; exit 3; }

log "stage and preflight coherent code/bootstrap on Bert $BERT_REPO"
scp -o BatchMode=yes "$BUNDLE" "$BERT_HOST:$BUNDLE"
scp -o BatchMode=yes "$BERT_SERVICE_BUNDLE" "$BERT_HOST:$BERT_SERVICE_BUNDLE"
scp -o BatchMode=yes "$BOOTSTRAP" "$BERT_HOST:$BOOT_COPY"
BERT_ACTIVATION_ATTEMPTED=1
ssh -o BatchMode=yes "$BERT_HOST" bash -s -- \
  "$BERT_REPO" "$BUNDLE" "$BERT_SERVICE_BUNDLE" "$BOOT_COPY" \
  "$BOOT_SHORT" "$BOOT_DIGEST" "$DEPLOY_ID" "$PREFLIGHT_PROFILE" <<'BERT'
set -euo pipefail
repo="$1"
bundle="$2"
service_bundle="$3"
bootstrap="$4"
digest_short="$5"
expected_digest="$6"
deploy_id="$7"
preflight_profile="$8"
stage="/tmp/pokebot_remote_preflight_${deploy_id}"

# Load with the staged source before stopping or replacing the healthy worker.
rm -rf "$stage"
mkdir -p "$stage"
tar -xf "$bundle" -C "$stage"
tar -xf "$service_bundle" -C "$stage"
cp "$bootstrap" "$stage/bootstrap.pt"
"$repo/.venv/bin/python" -m py_compile "$stage/scripts/run_remote_worker.py"
/bin/bash -n "$stage/scripts/run_bert_remote_worker_supervised.sh"
plutil -lint "$stage/deploy/launchd/com.pokebot.remote-worker-8766.plist"
(
  cd "$stage"
  # Staged source lives outside the repository, so its normal relative-path
  # discovery cannot see Bert's installed competition runtime. Pin that exact
  # runtime explicitly for the target-side model/schema preflight.
  CG_LIB_PATH="$repo/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg" \
    PYTHONPATH="$stage" EXPECTED_DIGEST="$expected_digest" \
    PREFLIGHT_PROFILE="$preflight_profile" \
    "$repo/.venv/bin/python" - "$stage/bootstrap.pt" <<'PY'
import os
import sys
from dataclasses import asdict
import torch
from poke_bot import checkpoint, features
from poke_bot.pure_rl.model_profile import count_params, model_config_dict, validate_param_budget
from poke_bot.train import load_model_from_checkpoint

path = sys.argv[1]
actual = checkpoint.checkpoint_digest(path)
if actual != os.environ["EXPECTED_DIGEST"]:
    raise SystemExit(f"digest mismatch: {actual}")
trusted = checkpoint.assert_trusted_policy_checkpoint(path)
schema = trusted["provenance"].get("feature_schema")
if schema != features.FEATURE_SCHEMA_VERSION:
    raise SystemExit(f"feature schema mismatch: {schema} != {features.FEATURE_SCHEMA_VERSION}")
model = load_model_from_checkpoint(path, device=torch.device("cpu"))
if os.environ["PREFLIGHT_PROFILE"] != "none" and asdict(model.cfg) != model_config_dict():
    raise SystemExit("pure-RL model profile mismatch")
validate_param_budget(count_params(model))
print(f"bert_preflight_ok digest={actual} schema={schema}")
PY
)

# Save the exact live source, worker command, and launchd-owned state before
# stopping anything.  The outer transaction trap uses these if this host or
# the later farm gate fails after Elmo has already activated.
code_backup="/tmp/pokebot_workspace_before_${deploy_id}.tar"
cmd_backup="/tmp/pokebot_worker_cmds_before_${deploy_id}.json"
marker="/tmp/pokebot_activation_${deploy_id}.marker"
service_snapshot="/tmp/pokebot_bert_service_before_${deploy_id}"
service_label="com.pokebot.remote-worker-8766"
service_domain="gui/$(id -u)"
service_target="${service_domain}/${service_label}"
agent_plist="$HOME/Library/LaunchAgents/${service_label}.plist"
checkpoint_current="$repo/outputs/checkpoints/pure_rl_bootstrap_current.pt"
service_wrapper_rel="scripts/run_bert_remote_worker_supervised.sh"
service_plist_rel="deploy/launchd/${service_label}.plist"
active_pgid_file="$repo/outputs/state/bert_worker_supervisor/active.pgid"
active_checkpoint_file="$repo/outputs/state/bert_worker_supervisor/active-checkpoint.json"
failure_file="$repo/outputs/state/bert_worker_supervisor/failures.epoch"
arm_file="$repo/outputs/state/REMOTE_WORKER_ARMED"
activation_worker_groups="$service_snapshot/activation_worker_pgids"

worker_group_exists() {
  local pgid="$1"
  ps -axo pgid= | awk -v pgid="$pgid" \
    '$1 == pgid { found=1 } END { exit !found }'
}

worker_group_is_exclusive() {
  local pgid="$1"
  ps -axo pgid=,command= | awk -v pgid="$pgid" -v repo="$repo" '
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
  '
}

capture_worker_groups() {
  local destination="$1"
  local current_pgid candidate_pids pid pgid recorded
  current_pgid="$(ps -o pgid= -p $$ | tr -d ' ')"
  : >"$destination"
  candidate_pids="$(
    { pgrep -f '[r]un_remote_worker.py' || true; \
      lsof -nP -iTCP:8766 -sTCP:LISTEN -t 2>/dev/null || true; } | \
      awk 'NF && !seen[$1]++ { print $1 }'
  )"
  for pid in $candidate_pids; do
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    [[ -n "$pgid" ]] && printf '%s\n' "$pgid" >>"$destination"
  done
  if [[ -f "$active_pgid_file" ]]; then
    recorded="$(awk 'NR == 1 { print $1 }' "$active_pgid_file")"
    case "$recorded" in
      ''|*[!0-9]*)
        echo "invalid durable Bert worker PGID record: $recorded" >&2
        return 1
        ;;
      *) printf '%s\n' "$recorded" >>"$destination" ;;
    esac
  fi
  sort -nu "$destination" -o "$destination"
  while IFS= read -r pgid; do
    [[ -n "$pgid" ]] || continue
    if [[ "$pgid" == "$current_pgid" ]]; then
      echo "refusing to signal current deployment shell PGID $pgid" >&2
      return 1
    fi
    if worker_group_exists "$pgid" && ! worker_group_is_exclusive "$pgid"; then
      echo "refusing non-exclusive/stale Bert worker PGID $pgid" >&2
      return 1
    fi
  done <"$destination"
}

terminate_worker_groups() {
  local groups_file="$1"
  local context="$2"
  local deadline pgid any_alive remaining recorded
  while IFS= read -r pgid; do
    [[ -n "$pgid" ]] || continue
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done <"$groups_file"
  deadline=$(( $(date +%s) + 20 ))
  while [[ "$(date +%s)" -lt "$deadline" ]]; do
    any_alive=0
    while IFS= read -r pgid; do
      [[ -n "$pgid" ]] || continue
      if worker_group_exists "$pgid"; then any_alive=1; fi
    done <"$groups_file"
    [[ "$any_alive" -eq 0 ]] && break
    sleep 1
  done
  while IFS= read -r pgid; do
    [[ -n "$pgid" ]] || continue
    if worker_group_exists "$pgid"; then
      kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
  done <"$groups_file"
  sleep 1
  remaining=""
  while IFS= read -r pgid; do
    [[ -n "$pgid" ]] || continue
    if worker_group_exists "$pgid"; then remaining="$remaining $pgid"; fi
  done <"$groups_file"
  [[ -z "$remaining" ]] || {
    echo "Bert worker process groups survived $context:$remaining" >&2
    return 1
  }
  remaining="$(pgrep -f '[r]un_remote_worker.py' || true)"
  [[ -z "$remaining" ]] || {
    echo "Bert worker parents appeared during $context: $remaining" >&2
    return 1
  }
  if [[ -f "$active_pgid_file" ]]; then
    recorded="$(awk 'NR == 1 { print $1 }' "$active_pgid_file")"
    if ! worker_group_exists "$recorded"; then rm -f "$active_pgid_file"; fi
  fi
}

tar -cf "$code_backup" -C "$repo" \
  poke_bot scripts/run_remote_worker.py scripts/train_round_robin.py
"$repo/.venv/bin/python" - "$cmd_backup" <<'PY'
import json
import shlex
import subprocess
import sys
from pathlib import Path

listeners = subprocess.check_output(
    ["lsof", "-nP", "-iTCP:8766", "-sTCP:LISTEN", "-t"], text=True
).split()
listeners = sorted(set(listeners))
if len(listeners) != 1:
    raise SystemExit(f"expected exactly one Bert :8766 listener, got {listeners}")
line = subprocess.check_output(
    ["ps", "-ww", "-p", listeners[0], "-o", "command="], text=True
).strip()
try:
    command = shlex.split(line)
except ValueError as exc:
    raise SystemExit(f"cannot parse Bert rollback command: {exc}") from exc
if not any(Path(arg).name == "run_remote_worker.py" for arg in command):
    raise SystemExit(f"Bert :8766 listener is not run_remote_worker.py: {line}")
Path(sys.argv[1]).write_text(json.dumps([command]), encoding="utf-8")
print(f"saved Bert rollback command pid={listeners[0]}")
PY

rm -rf "$service_snapshot"
mkdir -p "$service_snapshot"
if launchctl print "$service_target" >/dev/null 2>&1; then
  printf 'launchd\n' >"$service_snapshot/mode"
else
  printf 'detached\n' >"$service_snapshot/mode"
fi
if launchctl print-disabled "$service_domain" 2>/dev/null | \
    grep -Fq "\"${service_label}\" => true"; then
  printf 'disabled\n' >"$service_snapshot/enable_state"
else
  printf 'enabled\n' >"$service_snapshot/enable_state"
fi
if [[ -e "$agent_plist" || -L "$agent_plist" ]]; then
  cp -Pp "$agent_plist" "$service_snapshot/agent_plist"
  touch "$service_snapshot/agent_plist.present"
fi
if [[ -e "$checkpoint_current" || -L "$checkpoint_current" ]]; then
  cp -Pp "$checkpoint_current" "$service_snapshot/checkpoint_current"
  touch "$service_snapshot/checkpoint_current.present"
fi
if [[ -e "$active_checkpoint_file" || -L "$active_checkpoint_file" ]]; then
  cp -Pp "$active_checkpoint_file" "$service_snapshot/active_checkpoint"
  touch "$service_snapshot/active_checkpoint.present"
fi
if [[ -e "$failure_file" || -L "$failure_file" ]]; then
  cp -Pp "$failure_file" "$service_snapshot/failure_file"
  touch "$service_snapshot/failure_file.present"
fi
if [[ -e "$arm_file" || -L "$arm_file" ]]; then
  cp -Pp "$arm_file" "$service_snapshot/arm_file"
  touch "$service_snapshot/arm_file.present"
fi
: >"$service_snapshot/repo_assets.list"
for path in "$service_wrapper_rel" "$service_plist_rel"; do
  if [[ -e "$repo/$path" || -L "$repo/$path" ]]; then
    printf '%s\n' "$path" >>"$service_snapshot/repo_assets.list"
  fi
done
(
  cd "$repo"
  tar -cf "$service_snapshot/repo_assets.tar" \
    -T "$service_snapshot/repo_assets.list"
)
touch "$marker"

# Snapshot worker process groups before launchd can reap/reparent their parent.
# The durable record also finds a leaf-only group after a supervisor crash.
capture_worker_groups "$activation_worker_groups" || exit 30

# Remove KeepAlive supervision before terminating its worker, otherwise
# launchd can restart against the old source while this transaction publishes
# the new source/checkpoint pair.  The historical detached mode has no service
# to boot out during its one-time migration.
bert_was_launchd=0
if [[ "$(cat "$service_snapshot/mode")" == "launchd" ]]; then
  bert_was_launchd=1
  # Disabling first prevents a throttled KeepAlive job from remaining in
  # launchd's `spawn scheduled` state after bootout.  The recorded prior enable
  # state is restored below (or by rollback) after publication is complete.
  launchctl disable "$service_target"
  launchctl bootout "$service_target"
fi

# Stop and verify every captured parent/pool/leaf/resource-tracker group. A
# parent-only kill is forbidden because multiprocessing children are reparented.
terminate_worker_groups "$activation_worker_groups" "deployment termination" || exit 30

# launchd retains a SIGTERMed job while its process group is still draining.
# Confirm removal only after the captured worker group has been terminated;
# waiting before that cleanup creates a circular shutdown on macOS.
if [[ "$bert_was_launchd" -eq 1 ]]; then
  for _ in $(seq 1 240); do
    if ! launchctl print "$service_target" >/dev/null 2>&1; then
      break
    fi
    sleep 0.25
  done
  if launchctl print "$service_target" >/dev/null 2>&1; then
    echo "Bert launchd service survived bootout and worker drain: $service_target" >&2
    exit 29
  fi
fi

rm -rf "$repo/poke_bot"
tar -xf "$bundle" -C "$repo"
tar -xf "$service_bundle" -C "$repo"
mkdir -p "$repo/outputs/checkpoints" "$repo/outputs/logs"
checkpoint="$repo/outputs/checkpoints/pure_rl_bootstrap_${digest_short}.pt"
if [[ -f "$checkpoint" ]]; then
  cp -p "$checkpoint" "${checkpoint}.before_${deploy_id}"
fi
checkpoint_tmp="${checkpoint}.tmp.${deploy_id}"
cp "$bootstrap" "$checkpoint_tmp"
mv -f "$checkpoint_tmp" "$checkpoint"

# Publish a stable checkpoint pointer for the immutable LaunchAgent arguments.
# Replacing the symlink in the same directory is atomic; the digest-named file
# remains available for audit and rollback.
checkpoint_link_tmp="${checkpoint_current}.tmp.${deploy_id}"
rm -f "$checkpoint_link_tmp"
ln -s "$checkpoint" "$checkpoint_link_tmp"
mv -f "$checkpoint_link_tmp" "$checkpoint_current"
[[ -L "$checkpoint_current" && "$(readlink "$checkpoint_current")" == "$checkpoint" ]] || {
  echo "failed to publish Bert stable checkpoint pointer" >&2
  exit 31
}

# The supervisor's durable active-checkpoint record is authoritative over the
# stable bootstrap symlink. Publish it transactionally before launchd starts;
# rollback restores the exact prior JSON record above.
POKEBOT_REMOTE_ACTIVE_CHECKPOINT_FILE="$active_checkpoint_file" \
POKEBOT_REMOTE_CHECKPOINT_ROOT="$repo" EXPECTED_DIGEST="$expected_digest" \
PYTHONPATH="$repo" \
  "$repo/.venv/bin/python" - "$checkpoint" <<'PY'
import os
import sys

from scripts.run_remote_worker import _persist_active_checkpoint

published = _persist_active_checkpoint(sys.argv[1], os.environ["EXPECTED_DIGEST"])
if published is None:
    raise SystemExit("Bert durable active checkpoint file was not configured")
print(f"bert_active_checkpoint_published={published}")
PY

# Arm only the exact safety contract already pinned in both the LaunchAgent
# and worker preflight. The token is runtime state, so it is published
# atomically and restored (or removed) by rollback with the other Bert state.
arm_tmp="${arm_file}.tmp.${deploy_id}"
printf %s '20260717' >"$arm_tmp"
chmod 0644 "$arm_tmp"
mv -f "$arm_tmp" "$arm_file"

# Render the repo-path token through plistlib so non-default paths remain
# valid XML, then atomically install a user-owned LaunchAgent.
mkdir -p "$(dirname "$agent_plist")"
agent_plist_tmp="${agent_plist}.tmp.${deploy_id}"
"$repo/.venv/bin/python" - \
  "$repo/$service_plist_rel" "$agent_plist_tmp" "$repo" <<'PY'
import plistlib
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
repo = sys.argv[3]
token = "__POKEBOT_BERT_REPO__"
data = plistlib.loads(source.read_bytes())
replacements = 0


def render(value):
    global replacements
    if isinstance(value, str):
        replacements += value.count(token)
        return value.replace(token, repo)
    if isinstance(value, list):
        return [render(item) for item in value]
    if isinstance(value, dict):
        return {key: render(item) for key, item in value.items()}
    return value


data = render(data)
if replacements == 0:
    raise SystemExit(f"launchd template lacks {token}")
destination.write_bytes(plistlib.dumps(data, sort_keys=False))
PY
plutil -lint "$agent_plist_tmp"
chmod 0644 "$agent_plist_tmp"
mv -f "$agent_plist_tmp" "$agent_plist"

launchctl enable "$service_target"
rm -f "$failure_file"
launchctl bootstrap "$service_domain" "$agent_plist"
launchctl print "$service_target" >/dev/null
echo "restarted Bert worker under $service_target checkpoint=$checkpoint_current"
BERT

bert_ok=0
for _ in $(seq 1 120); do
  if EXPECTED_DIGEST="$BOOT_DIGEST" "$PYTHON" - <<'PY' 2>/dev/null
import os
from poke_bot.remote_jobs import RemoteJobClient, parse_endpoint

c = RemoteJobClient(*parse_endpoint("bert.local:8766"), timeout_s=10)
info = c.connect()
health = c.health()
c.close()
ok = (
    "self_play" in info.job_kinds
    and health.get("ok") is True
    and health.get("leaf_alive") is True
    and health.get("leaf_identity_ok") is True
    and health.get("checkpoint_digest") == os.environ["EXPECTED_DIGEST"]
    and len(health.get("leaves") or []) == info.leaf_servers
    and info.leaf_servers > 0
    and all(
        leaf.get("healthy") is True
        and leaf.get("checkpoint_digest") == os.environ["EXPECTED_DIGEST"]
        and leaf.get("version") == health.get("checkpoint_version")
        for leaf in (health.get("leaves") or [])
    )
)
raise SystemExit(0 if ok else 1)
PY
  then
    bert_ok=1
    log "Bert ready on the requested checkpoint"
    break
  fi
  sleep 2
done
[[ "$bert_ok" == 1 ]] || { log "ABORT Bert failed readiness"; exit 4; }

log "verify both remotes and checkpoint identity"
EXPECTED_DIGEST="$BOOT_DIGEST" "$PYTHON" - <<'PY' | tee -a "$LOG"
import os
from poke_bot.remote_jobs import RemoteWorkerFarm

expected = os.environ["EXPECTED_DIGEST"]
endpoints = ["elmo:8765", "bert.local:8766"]
farm = RemoteWorkerFarm(endpoints, timeout_s=30)
ok = False
try:
    infos = farm.connect(require_all=True)
    ok = (
        len(infos) == 2
        and len(farm.clients) == 2
        and len({info.endpoint for info in infos}) == 2
    )
    for client, info in zip(farm.clients, infos):
        health = client.health()
        leaves = health.get("leaves") or []
        endpoint_ok = (
            "self_play" in info.job_kinds
            and info.checkpoint_digest == expected
            and health.get("ok") is True
            and health.get("leaf_alive") is True
            and health.get("leaf_identity_ok") is True
            and health.get("checkpoint_digest") == expected
            and len(leaves) == info.leaf_servers
            and info.leaf_servers > 0
            and all(
                leaf.get("healthy") is True
                and leaf.get("checkpoint_digest") == expected
                and leaf.get("version") == health.get("checkpoint_version")
                for leaf in leaves
            )
        )
        print(
            f"{info.endpoint} digest={info.checkpoint_digest} "
            f"health_ok={health.get('ok')} leaf_alive={health.get('leaf_alive')} "
            f"leaf_identity_ok={health.get('leaf_identity_ok')} "
            f"leaves={len(leaves)}/{info.leaf_servers} endpoint_ok={endpoint_ok} "
            f"job_kinds={info.job_kinds} workers={info.workers} "
            f"default={info.default_workers} max={info.max_workers}"
        )
        ok = ok and endpoint_ok
finally:
    farm.close()
raise SystemExit(0 if ok else 5)
PY
DEPLOY_COMMITTED=1
ssh -o BatchMode=yes "$ELMO_HOST" rm -f "/tmp/pokebot_activation_${DEPLOY_ID}.marker" || \
  log "WARN could not clear committed Elmo transaction marker"
ssh -o BatchMode=yes "$BERT_HOST" rm -f "/tmp/pokebot_activation_${DEPLOY_ID}.marker" || \
  log "WARN could not clear committed Bert transaction marker"
log "DONE both remotes run the coherent code/checkpoint deployment"
