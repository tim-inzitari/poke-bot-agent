#!/usr/bin/env bash
# Redeploy throughput knobs and (optionally) restart pure-RL at the next iter.
#
# Run on the **training box** (LAN to Elmo/bert). Safe defaults:
#   - pull cursor/sim-gpu-multi-game-693f (or THRUPUT_BRANCH)
#   - canary remotes
#   - sync bert checkout + restart remote worker on :8766
#   - print the exact launch command for next-iter restart
#
# Does NOT kill overnight trainers unless you pass --restart-now.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BRANCH="${THRUPUT_BRANCH:-cursor/sim-gpu-multi-game-693f}"
PY="${POKEBOT_PYTHON:-python3}"
BERT_HOST="${BERT_SSH:-tsinzitari@bert.local}"
BERT_DIR="${BERT_REPO:-~/workspace/poke-bot-agent}"
ELMO_EP="${ELMO_ENDPOINT:-192.168.1.143:8765}"
BERT_EP="${BERT_ENDPOINT:-bert.local:8766}"
RUN_NAME="${PURE_RL_RUN_NAME:-}"
RESTART_NOW=0
SKIP_BERT=0
SKIP_CANARY=0

usage() {
  cat <<'EOF'
Usage: scripts/redeploy_throughput_next_iter.sh [options]

  --restart-now     Stop current pure-RL launch/monitor PIDs and start a new run
  --run-name NAME   Run name for --restart-now (default: stamped)
  --skip-bert       Do not SSH/sync/restart bert worker
  --skip-canary     Skip remote hello canaries
  --branch NAME     Git branch to deploy (default: cursor/sim-gpu-multi-game-693f)

Env:
  POKEBOT_PYTHON, THRUPUT_BRANCH, BERT_SSH, BERT_REPO, ELMO_ENDPOINT, BERT_ENDPOINT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart-now) RESTART_NOW=1; shift ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --skip-bert) SKIP_BERT=1; shift ;;
    --skip-canary) SKIP_CANARY=1; shift ;;
    --branch) BRANCH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

echo "== throughput redeploy =="
echo "root=$ROOT branch=$BRANCH"

echo "== git =="
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
echo "HEAD=$(git rev-parse --short HEAD)"

export POKEBOT_MULTI_ENV="${POKEBOT_MULTI_ENV:-1}"
export POKEBOT_MULTI_ENV_PER_WORKER="${POKEBOT_MULTI_ENV_PER_WORKER:-4}"
export PURE_RL_MULTI_ENV="${PURE_RL_MULTI_ENV:-1}"
export PURE_RL_MULTI_ENV_PER_WORKER="${PURE_RL_MULTI_ENV_PER_WORKER:-4}"
export PURE_RL_LEAF_COALESCE_MS="${PURE_RL_LEAF_COALESCE_MS:-0}"
export PURE_RL_REMOTE_WORKER_ENDPOINTS="${PURE_RL_REMOTE_WORKER_ENDPOINTS:-$ELMO_EP,$BERT_EP}"

echo "== knobs =="
echo "POKEBOT_MULTI_ENV=$POKEBOT_MULTI_ENV"
echo "POKEBOT_MULTI_ENV_PER_WORKER=$POKEBOT_MULTI_ENV_PER_WORKER"
echo "PURE_RL_LEAF_COALESCE_MS=$PURE_RL_LEAF_COALESCE_MS"
echo "REMOTE=$PURE_RL_REMOTE_WORKER_ENDPOINTS"

if [[ "$SKIP_CANARY" -eq 0 ]]; then
  echo "== canary remotes =="
  if [[ -f scripts/canary_remote_worker.py ]]; then
    "$PY" scripts/canary_remote_worker.py "$ELMO_EP" || echo "WARN: elmo canary failed ($ELMO_EP)"
    "$PY" scripts/canary_remote_worker.py "$BERT_EP" || echo "WARN: bert canary failed ($BERT_EP)"
  else
    echo "WARN: scripts/canary_remote_worker.py missing"
  fi
fi

if [[ "$SKIP_BERT" -eq 0 ]]; then
  echo "== bert sync + worker restart =="
  if ssh -o BatchMode=yes -o ConnectTimeout=8 "$BERT_HOST" "echo ok" >/dev/null 2>&1; then
    ssh "$BERT_HOST" "bash -s" <<EOF
set -euo pipefail
cd $BERT_DIR
git fetch origin $BRANCH
git checkout $BRANCH
git pull --ff-only origin $BRANCH
mkdir -p logs
# Prefer existing venv; fall back to python3
PY_BERT=python3
if [[ -x .venv/bin/python ]]; then PY_BERT=.venv/bin/python; fi
# Stop prior worker on 8766 if we recorded a pid
if [[ -f logs/remote_worker.8766.pid ]]; then
  kill "\$(cat logs/remote_worker.8766.pid)" 2>/dev/null || true
  rm -f logs/remote_worker.8766.pid
fi
pkill -f 'run_remote_worker.py.*8766' 2>/dev/null || true
sleep 1
nohup "\$PY_BERT" -u scripts/run_remote_worker.py --port 8766 \
  > logs/remote_worker.8766.log 2>&1 &
echo \$! > logs/remote_worker.8766.pid
echo "bert worker pid=\$(cat logs/remote_worker.8766.pid) HEAD=\$(git rev-parse --short HEAD)"
EOF
    sleep 2
    "$PY" scripts/canary_remote_worker.py "$BERT_EP" || echo "WARN: bert canary after restart failed"
  else
    echo "WARN: cannot SSH $BERT_HOST (BatchMode). Sync bert manually:"
    echo "  ssh $BERT_HOST 'cd $BERT_DIR && git fetch && git checkout $BRANCH && git pull'"
    echo "  then restart: python scripts/run_remote_worker.py --port 8766"
  fi
else
  echo "== skip bert =="
fi

echo "== elmo note =="
echo "Elmo/host Docker worker is unchanged by this script."
echo "If you need the new code inside the TrueNAS image, rebuild/load compose"
echo "(see containers/truenas-worker/OPS.md). Whole-game TCP to $ELMO_EP still helps."

LAUNCH_CMD=(
  "$PY" -u scripts/launch_pure_rl.py
  --mode core
  --preflight-profile quick
  --multi-env-per-worker "${POKEBOT_MULTI_ENV_PER_WORKER}"
  --leaf-coalesce-ms "${PURE_RL_LEAF_COALESCE_MS}"
  --remote-worker-endpoints "$PURE_RL_REMOTE_WORKER_ENDPOINTS"
)

if [[ -n "$RUN_NAME" ]]; then
  LAUNCH_CMD+=(--run-name "$RUN_NAME")
fi

echo "== next-iter launch command =="
printf ' %q' "${LAUNCH_CMD[@]}"
echo

if [[ "$RESTART_NOW" -eq 1 ]]; then
  echo "== restart-now =="
  # Prefer promotion/iter boundary; this is an explicit operator request.
  pkill -f 'scripts/train_pure_rl.py' 2>/dev/null || true
  pkill -f 'scripts/launch_pure_rl.py' 2>/dev/null || true
  pkill -f 'unattended_monitor.py.*pure_rl' 2>/dev/null || true
  sleep 2
  if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="pure_rl_core_thruput_$(date -u +%Y%m%dT%H%M%SZ)"
    LAUNCH_CMD+=(--run-name "$RUN_NAME")
  fi
  nohup "${LAUNCH_CMD[@]}" >"outputs/logs/redeploy_throughput_${RUN_NAME}.nohup.log" 2>&1 &
  echo "started pid=$! run_name=$RUN_NAME"
  echo "tail -f outputs/logs/pure_rl.log"
  echo "Expect log lines: multi_env=4 leaf_coalesce_ms=0 leaf_modes=gpu-leaf-*"
else
  echo "== idle =="
  echo "Defaults are on for the next launch_pure_rl / train_pure_rl start."
  echo "At the next iter boundary (or when you choose), run the launch command above,"
  echo "or: bash scripts/redeploy_throughput_next_iter.sh --restart-now"
fi
