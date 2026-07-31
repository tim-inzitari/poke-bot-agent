#!/usr/bin/env bash
# Container entrypoint: validate the bounded Elmo topology, then serve jobs.
set -euo pipefail

echo "[entrypoint] hostname=$(hostname) nproc=$(nproc) date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

REQUIRED_SAFETY_VERSION="20260717"
if [[ "${POKEBOT_REMOTE_WORKER_SAFETY_VERSION:-}" != "$REQUIRED_SAFETY_VERSION" ]]; then
  echo "[entrypoint] ERROR: image predates remote-worker memory safety version $REQUIRED_SAFETY_VERSION; rebuild it before starting Elmo" >&2
  exit 78
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "[entrypoint] WARN: nvidia-smi not found in container PATH" >&2
fi

CKPT="${POKEBOT_CHECKPOINT:-/workspace/checkpoint/model.pt}"
export POKEBOT_CHECKPOINT="$CKPT"
export CG_LIB_PATH="${CG_LIB_PATH:-/workspace/kaggle/input/cg-lib}"
export POKEBOT_WORKER_CPU_ONLY="${POKEBOT_WORKER_CPU_ONLY:-1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

SIM_WORKERS="${SIM_WORKERS:-4}"
SIM_DEFAULT_WORKERS="${SIM_DEFAULT_WORKERS:-4}"
ELMO_SIM_WORKER_CEILING="${ELMO_SIM_WORKER_CEILING:-20}"

for pair in \
  "SIM_WORKERS:$SIM_WORKERS" \
  "SIM_DEFAULT_WORKERS:$SIM_DEFAULT_WORKERS" \
  "ELMO_SIM_WORKER_CEILING:$ELMO_SIM_WORKER_CEILING"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[entrypoint] ERROR: $name must be a positive integer (got $value)" >&2
    exit 64
  fi
done

if (( SIM_WORKERS > ELMO_SIM_WORKER_CEILING )); then
  echo "[entrypoint] ERROR: SIM_WORKERS=$SIM_WORKERS exceeds the Elmo ceiling $ELMO_SIM_WORKER_CEILING" >&2
  exit 64
fi
if (( SIM_DEFAULT_WORKERS > SIM_WORKERS )); then
  echo "[entrypoint] ERROR: SIM_DEFAULT_WORKERS=$SIM_DEFAULT_WORKERS exceeds pool size $SIM_WORKERS" >&2
  exit 64
fi
export SIM_WORKERS SIM_DEFAULT_WORKERS ELMO_SIM_WORKER_CEILING

if [[ "${1:-}" == "seed-active-checkpoint" ]]; then
  shift
  if [[ "$#" -ne 0 ]]; then
    echo "[entrypoint] ERROR: seed-active-checkpoint does not accept arguments" >&2
    exit 64
  fi
  exec python /workspace/scripts/seed_remote_active_checkpoint.py \
    --checkpoint "$CKPT"
fi

if [[ "${1:-}" == "production" ]]; then
  shift
  if [[ "$#" -ne 0 ]]; then
    echo "[entrypoint] ERROR: production mode does not accept command-line overrides" >&2
    exit 64
  fi
  production_jobs="${POKEBOT_REMOTE_MAX_SERVICE_JOBS:-}"
  production_rotation_code="${POKEBOT_REMOTE_PLANNED_ROTATION_EXIT_CODE:-}"
  if [[ "$production_jobs" == "0" ]]; then
    : # Stable production mode: watchdogs/recycling bound memory without admission gaps.
  elif [[ "$production_jobs" =~ ^[1-9][0-9]*$ ]] && \
       (( production_jobs >= 512 && production_jobs <= 1024 )); then
    : # Accepted only for explicit rollback/canary configurations.
  else
    echo "[entrypoint] ERROR: production max-service-jobs must be 0 or within 512..1024 (got $production_jobs)" >&2
    exit 64
  fi
  if [[ "$production_rotation_code" != "75" ]]; then
    echo "[entrypoint] ERROR: production planned-rotation exit code must be 75" >&2
    exit 64
  fi
  for required in \
    "POKEBOT_WORKER_RECYCLE_GAMES:256" \
    "WORKER_RECYCLE_GAMES:256" \
    "POKEBOT_REMOTE_TREE_RSS_LIMIT_GB:45" \
    "POKEBOT_REMOTE_MIN_FREE_RAM_GB:24" \
    "POKEBOT_REMOTE_WATCHDOG_INTERVAL_S:5" \
    "POKEBOT_REMOTE_WORKER_CAPACITY_RECOVERY_GRACE_S:300" \
    "POKEBOT_REMOTE_WORKER_MIN_READY_FRAC:0.80" \
    "POKEBOT_ELMO_SUPERVISOR_STATE_DIR:/workspace/runtime-logs/elmo-supervisor" \
    "POKEBOT_ELMO_RESTART_LIMIT:20" \
    "POKEBOT_ELMO_RESTART_WINDOW_S:3600" \
    "POKEBOT_ELMO_CHILD_STOP_GRACE_S:75" \
    "POKEBOT_REMOTE_ACTIVE_CHECKPOINT_FILE:/workspace/runtime-logs/elmo-supervisor/active-checkpoint.json" \
    "POKEBOT_REMOTE_CHECKPOINT_ROOT:/workspace/checkpoint"; do
    name="${required%%:*}"
    expected="${required#*:}"
    if [[ "${!name:-}" != "$expected" ]]; then
      echo "[entrypoint] ERROR: production $name must be $expected (got ${!name:-unset})" >&2
      exit 64
    fi
  done
  exec /supervise-production.sh \
    python /workspace/scripts/run_remote_worker.py \
    --host 0.0.0.0 \
    --port "${REMOTE_JOB_PORT:-8765}" \
    --workers "$SIM_WORKERS" \
    --leaf-servers "${LEAF_SERVERS:-2}" \
    --leaf-gpu "${LEAF_GPU:-cuda:0}" \
    --leaf-max-batch "${LEAF_MAX_BATCH:-192}" \
    --leaf-queue-depth "${LEAF_QUEUE_DEPTH:-96}" \
    --leaf-coalesce-ms "${LEAF_COALESCE_MS:-2}" \
    --checkpoint "$CKPT" \
    --planned-rotation-exit-code "$production_rotation_code"
fi

if [[ "${1:-}" == "smoke" ]]; then
  exec python /workspace/scripts/run_remote_worker.py --smoke --checkpoint "$CKPT"
fi

if [[ "${1:-}" == "bash" ]] || [[ "${1:-}" == "sh" ]]; then
  exec "$@"
fi

exec python /workspace/scripts/run_remote_worker.py \
  --host 0.0.0.0 \
  --port "${REMOTE_JOB_PORT:-8765}" \
  --workers "$SIM_WORKERS" \
  --leaf-servers "${LEAF_SERVERS:-2}" \
  --leaf-gpu "${LEAF_GPU:-cuda:0}" \
  --leaf-max-batch "${LEAF_MAX_BATCH:-192}" \
  --leaf-queue-depth "${LEAF_QUEUE_DEPTH:-96}" \
  --leaf-coalesce-ms "${LEAF_COALESCE_MS:-2}" \
  --checkpoint "$CKPT" \
  "$@"
