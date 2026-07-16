#!/bin/bash
# Container entrypoint: verify GPU, then serve remote jobs (or smoke).
set -euo pipefail

echo "[entrypoint] hostname=$(hostname) nproc=$(nproc) date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "[entrypoint] WARN: nvidia-smi not found in container PATH" >&2
fi

CKPT="${POKEBOT_CHECKPOINT:-/workspace/checkpoint/model.pt}"
export POKEBOT_CHECKPOINT="$CKPT"
export POKEBOT_WORKER_CPU_ONLY="${POKEBOT_WORKER_CPU_ONLY:-1}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTHONPATH="/workspace:${PYTHONPATH:-}"
export POKEBOT_MULTI_ENV="${POKEBOT_MULTI_ENV:-1}"
export POKEBOT_MULTI_ENV_PER_WORKER="${POKEBOT_MULTI_ENV_PER_WORKER:-4}"

# Resolve CG_LIB_PATH: explicit env > fork mount > competition sample > cg-lib.
if [[ -z "${CG_LIB_PATH:-}" ]]; then
  if [[ -f /workspace/libcg_fork/libcg.so ]] || [[ -f /workspace/libcg_fork/cg/libcg.so ]]; then
    CG_LIB_PATH=/workspace/libcg_fork
  elif [[ -f /workspace/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg/libcg.so ]]; then
    CG_LIB_PATH=/workspace/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg
  else
    CG_LIB_PATH=/workspace/kaggle/input/cg-lib
  fi
fi
export CG_LIB_PATH

echo "[entrypoint] CG_LIB_PATH=$CG_LIB_PATH POKEBOT_MULTI_ENV=$POKEBOT_MULTI_ENV" \
     "POKEBOT_MULTI_ENV_PER_WORKER=$POKEBOT_MULTI_ENV_PER_WORKER"

if [[ "${1:-}" == "smoke" ]]; then
  exec python /workspace/scripts/run_remote_worker.py --smoke --checkpoint "$CKPT"
fi

if [[ "${1:-}" == "bash" ]] || [[ "${1:-}" == "sh" ]]; then
  exec "$@"
fi

# Default: listen for training-box jobs. Extra args are forwarded.
exec python /workspace/scripts/run_remote_worker.py \
  --host 0.0.0.0 \
  --port "${REMOTE_JOB_PORT:-8765}" \
  --workers "${SIM_WORKERS:-20}" \
  --leaf-servers "${LEAF_SERVERS:-2}" \
  --leaf-gpu "${LEAF_GPU:-cuda:0}" \
  --leaf-max-batch "${LEAF_MAX_BATCH:-192}" \
  --leaf-queue-depth "${LEAF_QUEUE_DEPTH:-96}" \
  --leaf-coalesce-ms "${LEAF_COALESCE_MS:-2}" \
  --checkpoint "$CKPT" \
  --cg-lib-path "$CG_LIB_PATH" \
  "$@"
