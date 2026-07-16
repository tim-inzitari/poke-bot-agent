#!/usr/bin/env bash
# Safe promotion/iter-boundary redeploy for host + Elmo (TrueNAS) + bert.
#
# Defaults are NON-DESTRUCTIVE:
#   - never kills live overnight training unless --cutover-host is set
#   - never restarts remotes unless --cutover-remotes / --cutover-all
#   - prep modes (stage/build/sync-code) are safe mid-collect
#
# Version-storm avoidance:
#   1. Prep image + code first (no restarts).
#   2. Wait for iter boundary ([pure_rl] iter=N … after remote reload/heldout).
#   3. Restart remotes + host in one cutover window with matching
#      simulator_version / MultiEnv knobs.
#   4. Fail closed on canary digest / simulator_version mismatch.
#
# Run on the training box (LAN to Elmo/bert).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${POKEBOT_PYTHON:-python3}"
BRANCH="${CUTOVER_BRANCH:-cursor/pure-rl-full-rebuild-2d48}"
BERT_HOST="${BERT_SSH:-tsinzitari@bert.local}"
BERT_DIR="${BERT_REPO:-~/workspace/poke-bot-agent}"
ELMO_EP="${ELMO_ENDPOINT:-192.168.1.143:8765}"
BERT_EP="${BERT_ENDPOINT:-bert.local:8766}"
TRUENAS_SSH="${TRUENAS_SSH:-${TRUENAS_SSH_USER:+${TRUENAS_SSH_USER}@192.168.1.143}}"
ELMO_COMPOSE_DIR="${ELMO_COMPOSE_DIR:-/mnt/Main/main/poke-bot-agent/containers/truenas-worker}"
PURE_RL_LOG="${PURE_RL_LOG:-$ROOT/outputs/logs/pure_rl.log}"
GO_FLAG="${CUTOVER_GO_FLAG:-$ROOT/outputs/state/remote_cutover_go}"
LIBCG_FORK="${LIBCG_FORK:-}"
RUN_NAME="${PURE_RL_RUN_NAME:-}"

DO_WAIT=0
SKIP_WAIT=0
DO_PULL=0
DO_STAGE=0
DO_BUILD=0
DO_EXPORT=0
DO_SYNC_BERT_CODE=0
DO_CUTOVER_REMOTES=0
DO_CUTOVER_HOST=0
SKIP_CANARY=0
DRY_RUN=0
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-7200}"
WAIT_POLL_S="${WAIT_POLL_S:-15}"

usage() {
  cat <<'EOF'
Usage: scripts/redeploy_remote_boundary.sh [options]

Prep (safe anytime — no trainer/remote kills):
  --pull                git fetch/checkout/pull CUTOVER_BRANCH
  --stage-elmo          deploy_truenas_worker.sh --stage (SMB)
  --build-elmo          deploy_truenas_worker.sh --build (needs Docker)
  --export-elmo         deploy_truenas_worker.sh --export-image
  --sync-bert-code      git sync bert checkout ONLY (does not restart worker)
  --libcg-fork PATH     optional fork cg tree for Elmo bake/stage

Boundary wait:
  --wait-boundary       poll pure_rl.log for post-iter marker, or GO flag
  --skip-wait           operator asserts we are already at a safe boundary
  --timeout SEC         wait timeout (default 7200)

Cutover (ONLY with --wait-boundary or --skip-wait):
  --cutover-remotes     restart Elmo compose + bert worker
  --cutover-host        stop pure-RL launch/monitor and relaunch (explicit)
  --cutover-all         --cutover-remotes + --cutover-host
  --run-name NAME       run name for host relaunch

Other:
  --skip-canary         do not fail closed on remote canaries
  --dry-run             print actions only
  --branch NAME         deploy branch (default: cursor/pure-rl-full-rebuild-2d48)

Env:
  POKEBOT_PYTHON, CUTOVER_BRANCH, BERT_SSH, BERT_REPO, ELMO_ENDPOINT,
  BERT_ENDPOINT, TRUENAS_SSH / TRUENAS_SSH_USER, ELMO_COMPOSE_DIR,
  PURE_RL_LOG, LIBCG_FORK, CUTOVER_GO_FLAG
EOF
}

log() { echo "[boundary] $*"; }
run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY-RUN: $*"
    return 0
  fi
  "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pull) DO_PULL=1; shift ;;
    --stage-elmo) DO_STAGE=1; shift ;;
    --build-elmo) DO_BUILD=1; shift ;;
    --export-elmo) DO_EXPORT=1; shift ;;
    --sync-bert-code) DO_SYNC_BERT_CODE=1; shift ;;
    --libcg-fork) LIBCG_FORK="$2"; shift 2 ;;
    --wait-boundary) DO_WAIT=1; shift ;;
    --skip-wait) SKIP_WAIT=1; shift ;;
    --timeout) WAIT_TIMEOUT_S="$2"; shift 2 ;;
    --cutover-remotes) DO_CUTOVER_REMOTES=1; shift ;;
    --cutover-host) DO_CUTOVER_HOST=1; shift ;;
    --cutover-all) DO_CUTOVER_REMOTES=1; DO_CUTOVER_HOST=1; shift ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --skip-canary) SKIP_CANARY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --branch) BRANCH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$DO_CUTOVER_REMOTES" -eq 1 || "$DO_CUTOVER_HOST" -eq 1 ]]; then
  if [[ "$DO_WAIT" -eq 0 && "$SKIP_WAIT" -eq 0 ]]; then
    echo "ERROR: cutover requires --wait-boundary or --skip-wait" >&2
    echo "Refusing mid-collection restart (version-storm / team rule)." >&2
    exit 2
  fi
fi

DEPLOY_ARGS=()
if [[ -n "$LIBCG_FORK" ]]; then
  DEPLOY_ARGS+=(--libcg-fork "$LIBCG_FORK")
fi

echo "== remote boundary redeploy =="
echo "root=$ROOT branch=$BRANCH dry_run=$DRY_RUN"
echo "elmo=$ELMO_EP bert=$BERT_EP"
echo "libcg_fork=${LIBCG_FORK:-none}"

if [[ "$DO_PULL" -eq 1 ]]; then
  log "git pull $BRANCH"
  run git fetch origin "$BRANCH"
  run git checkout "$BRANCH"
  run git pull --ff-only origin "$BRANCH"
  log "HEAD=$(git rev-parse --short HEAD)"
fi

export POKEBOT_MULTI_ENV="${POKEBOT_MULTI_ENV:-1}"
export POKEBOT_MULTI_ENV_PER_WORKER="${POKEBOT_MULTI_ENV_PER_WORKER:-4}"
export PURE_RL_MULTI_ENV="${PURE_RL_MULTI_ENV:-1}"
export PURE_RL_MULTI_ENV_PER_WORKER="${PURE_RL_MULTI_ENV_PER_WORKER:-4}"
export PURE_RL_LEAF_COALESCE_MS="${PURE_RL_LEAF_COALESCE_MS:-0}"
export POKEBOT_LIVE_POOL="${POKEBOT_LIVE_POOL:-1}"
export PURE_RL_REMOTE_WORKER_ENDPOINTS="${PURE_RL_REMOTE_WORKER_ENDPOINTS:-$ELMO_EP,$BERT_EP}"

if [[ "$DO_STAGE" -eq 1 ]]; then
  log "stage Elmo SMB context"
  run bash "$ROOT/scripts/deploy_truenas_worker.sh" "${DEPLOY_ARGS[@]}" --stage
fi
if [[ "$DO_BUILD" -eq 1 ]]; then
  log "build Elmo image"
  run bash "$ROOT/scripts/deploy_truenas_worker.sh" "${DEPLOY_ARGS[@]}" --build
fi
if [[ "$DO_EXPORT" -eq 1 ]]; then
  log "export Elmo image to SMB"
  run bash "$ROOT/scripts/deploy_truenas_worker.sh" "${DEPLOY_ARGS[@]}" --export-image
fi

sync_bert_code() {
  log "bert code sync (no worker restart)"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$BERT_HOST" "echo ok" >/dev/null 2>&1; then
    echo "WARN: cannot SSH $BERT_HOST (BatchMode). Manual:" >&2
    echo "  ssh $BERT_HOST 'cd $BERT_DIR && git fetch && git checkout $BRANCH && git pull'" >&2
    return 1
  fi
  run ssh "$BERT_HOST" "bash -s" <<EOF
set -euo pipefail
cd $BERT_DIR
git fetch origin $BRANCH
git checkout $BRANCH
git pull --ff-only origin $BRANCH
echo "bert HEAD=\$(git rev-parse --short HEAD) (worker NOT restarted)"
EOF
}

restart_bert_worker() {
  log "bert worker restart :8766"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$BERT_HOST" "echo ok" >/dev/null 2>&1; then
    echo "ERROR: cannot SSH $BERT_HOST for worker restart" >&2
    return 1
  fi
  run ssh "$BERT_HOST" "bash -s" <<EOF
set -euo pipefail
cd $BERT_DIR
git fetch origin $BRANCH
git checkout $BRANCH
git pull --ff-only origin $BRANCH
mkdir -p logs
PY_BERT=python3
if [[ -x .venv/bin/python ]]; then PY_BERT=.venv/bin/python; fi
export POKEBOT_MULTI_ENV=${POKEBOT_MULTI_ENV}
export POKEBOT_MULTI_ENV_PER_WORKER=${POKEBOT_MULTI_ENV_PER_WORKER}
# Competition arm64 dylib on bert (not cg-lib / not linux .so)
if [[ -z "\${CG_LIB_PATH:-}" ]]; then
  CG_CAND="\$HOME/workspace/poke-bot-agent/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg"
  if [[ -d "\$CG_CAND" ]]; then export CG_LIB_PATH="\$CG_CAND"; fi
fi
if [[ -z "\${POKEBOT_CHECKPOINT:-}" ]]; then
  for c in \\
    outputs/checkpoints/model.pt \\
    checkpoint/model.pt \\
    outputs/pure_rl/*/checkpoints/*.pt
  do
    # shellcheck disable=SC2086
    if ls \$c >/dev/null 2>&1; then
      export POKEBOT_CHECKPOINT="\$(ls -1t \$c | head -1)"
      break
    fi
  done
fi
if [[ -f logs/remote_worker.8766.pid ]]; then
  kill "\$(cat logs/remote_worker.8766.pid)" 2>/dev/null || true
  rm -f logs/remote_worker.8766.pid
fi
pkill -f 'run_remote_worker.py.*8766' 2>/dev/null || true
sleep 1
nohup "\$PY_BERT" -u scripts/run_remote_worker.py --host 0.0.0.0 --port 8766 \\
  --workers "\${SIM_WORKERS:-10}" --leaf-servers "\${LEAF_SERVERS:-1}" \\
  --leaf-gpu mps --leaf-max-batch 96 --leaf-queue-depth 48 \\
  \${POKEBOT_CHECKPOINT:+--checkpoint "\$POKEBOT_CHECKPOINT"} \\
  \${CG_LIB_PATH:+--cg-lib-path "\$CG_LIB_PATH"} \\
  > logs/remote_worker.8766.log 2>&1 &
echo \$! > logs/remote_worker.8766.pid
echo "bert worker pid=\$(cat logs/remote_worker.8766.pid) HEAD=\$(git rev-parse --short HEAD)"
EOF
}

restart_elmo_worker() {
  log "Elmo/TrueNAS compose recreate"
  if [[ -z "${TRUENAS_SSH:-}" ]]; then
    echo "WARN: TRUENAS_SSH unset — print manual load/restart commands" >&2
    bash "$ROOT/scripts/deploy_truenas_worker.sh" --print-load
    return 1
  fi
  if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$TRUENAS_SSH" "echo ok" >/dev/null 2>&1; then
    echo "ERROR: cannot SSH $TRUENAS_SSH" >&2
    bash "$ROOT/scripts/deploy_truenas_worker.sh" --print-load
    return 1
  fi
  local fork_env=""
  if [[ -n "$LIBCG_FORK" ]]; then
    fork_env="export LIBCG_FORK_HOST=./libcg_fork CG_LIB_PATH=/workspace/libcg_fork;"
  fi
  run ssh "$TRUENAS_SSH" "bash -s" <<EOF
set -euo pipefail
DIR="$ELMO_COMPOSE_DIR"
if [[ ! -d "\$DIR" ]]; then
  DIR=/mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker
fi
cd "\$DIR"
if [[ -f poke-bot-truenas-worker.tar.gz ]]; then
  gunzip -fk poke-bot-truenas-worker.tar.gz || true
fi
if [[ -f poke-bot-truenas-worker.tar ]]; then
  docker load -i poke-bot-truenas-worker.tar
fi
mkdir -p checkpoint runtime-logs libcg_fork
$fork_env
export POKEBOT_MULTI_ENV=${POKEBOT_MULTI_ENV}
export POKEBOT_MULTI_ENV_PER_WORKER=${POKEBOT_MULTI_ENV_PER_WORKER}
docker compose down
docker compose up -d
docker compose ps
docker logs --tail 40 poke-bot-truenas-worker || true
EOF
}

wait_boundary() {
  log "waiting for iter/promotion boundary (timeout=${WAIT_TIMEOUT_S}s)"
  log "markers: '$GO_FLAG' OR log line '[pure_rl] iter=' in $PURE_RL_LOG"
  local start now elapsed
  start=$(date +%s)
  local last_seen=""
  while true; do
    now=$(date +%s)
    elapsed=$((now - start))
    if [[ "$elapsed" -ge "$WAIT_TIMEOUT_S" ]]; then
      echo "ERROR: boundary wait timed out after ${WAIT_TIMEOUT_S}s" >&2
      return 1
    fi
    if [[ -f "$GO_FLAG" ]]; then
      log "GO flag present: $GO_FLAG"
      return 0
    fi
    if [[ -f "$PURE_RL_LOG" ]]; then
      # Prefer completed-iter marker (after remote reload + heldout).
      local hit
      hit=$(rg -N '\[pure_rl\] iter=[0-9]+ ' "$PURE_RL_LOG" | tail -1 || true)
      if [[ -n "$hit" && "$hit" != "$last_seen" ]]; then
        last_seen="$hit"
        log "saw boundary marker: $hit"
        # Brief settle so overlapping next-collect submit is less likely mid-pkill.
        sleep 3
        return 0
      fi
    fi
    sleep "$WAIT_POLL_S"
  done
}

canary_remotes() {
  if [[ "$SKIP_CANARY" -eq 1 ]]; then
    log "skip canary"
    return 0
  fi
  if [[ ! -f scripts/canary_remote_worker.py ]]; then
    echo "ERROR: scripts/canary_remote_worker.py missing" >&2
    return 2
  fi
  log "canary $ELMO_EP (fail-closed match local simulator_version)"
  run "$PY" scripts/canary_remote_worker.py "$ELMO_EP" --require-match-local
  log "canary $BERT_EP (fail-closed match local simulator_version)"
  run "$PY" scripts/canary_remote_worker.py "$BERT_EP" --require-match-local
  if [[ -f scripts/canary_game_accuracy.py ]]; then
    log "game accuracy canary (fail-closed)"
    run "$PY" scripts/canary_game_accuracy.py \
      --num-envs "${POKEBOT_MULTI_ENV_PER_WORKER:-4}" \
      --json-out outputs/state/game_accuracy_canary.json
  fi
}

restart_host() {
  log "host pure-RL restart (explicit --cutover-host)"
  # Prefer promotion/iter boundary; this is an explicit operator request.
  run pkill -f 'scripts/train_pure_rl.py' 2>/dev/null || true
  run pkill -f 'scripts/launch_pure_rl.py' 2>/dev/null || true
  run pkill -f 'unattended_monitor.py.*pure_rl' 2>/dev/null || true
  run pkill -f 'scripts/resource_watcher.py' 2>/dev/null || true
  sleep 2
  local launch=(
    "$PY" -u scripts/launch_pure_rl.py
    --mode core
    --preflight-profile quick
    --multi-env-per-worker "${POKEBOT_MULTI_ENV_PER_WORKER}"
    --leaf-coalesce-ms "${PURE_RL_LEAF_COALESCE_MS}"
    --remote-worker-endpoints "$PURE_RL_REMOTE_WORKER_ENDPOINTS"
  )
  if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="pure_rl_core_boundary_$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  launch+=(--run-name "$RUN_NAME")
  mkdir -p outputs/logs
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY-RUN launch: ${launch[*]}"
    return 0
  fi
  nohup "${launch[@]}" >"outputs/logs/redeploy_boundary_${RUN_NAME}.nohup.log" 2>&1 &
  log "started pid=$! run_name=$RUN_NAME"
}

# --- prep ---
if [[ "$DO_SYNC_BERT_CODE" -eq 1 ]]; then
  sync_bert_code || true
fi

# --- boundary ---
if [[ "$DO_WAIT" -eq 1 ]]; then
  wait_boundary
elif [[ "$SKIP_WAIT" -eq 1 ]]; then
  log "skip-wait: operator-asserted safe boundary"
fi

# --- cutover ---
if [[ "$DO_CUTOVER_REMOTES" -eq 1 ]]; then
  restart_elmo_worker || log "Elmo restart incomplete — see printed LOAD commands"
  restart_bert_worker || log "bert restart incomplete"
  sleep 3
  canary_remotes
fi

if [[ "$DO_CUTOVER_HOST" -eq 1 ]]; then
  # Canary again immediately before host relaunch if remotes were cut over.
  if [[ "$DO_CUTOVER_REMOTES" -eq 0 ]]; then
    canary_remotes || true
  fi
  restart_host
fi

if [[ "$DO_CUTOVER_REMOTES" -eq 0 && "$DO_CUTOVER_HOST" -eq 0 ]]; then
  cat <<EOF

== idle / prep complete ==
No restarts performed. Suggested overnight cutover:

  # 1) Prep anytime (safe mid-collect):
  bash scripts/redeploy_remote_boundary.sh \\
    --pull --stage-elmo --build-elmo --export-elmo --sync-bert-code \\
    ${LIBCG_FORK:+--libcg-fork "$LIBCG_FORK"}

  # 2) At iter boundary (after log line '[pure_rl] iter=N …'):
  bash scripts/redeploy_remote_boundary.sh --wait-boundary --cutover-all

  # Or touch a GO flag from another shell once collect/promote settles:
  mkdir -p outputs/state && touch outputs/state/remote_cutover_go

See docs/REMOTE_WORKER_CUTOVER.md
EOF
fi
