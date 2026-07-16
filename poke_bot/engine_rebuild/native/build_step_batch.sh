#!/usr/bin/env bash
# Build additive libcg_step_batch.so (dlopen wrapper; no ptcg_engine sources).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SRC="$(cd "$(dirname "$0")" && pwd)/step_batch.cpp"
OUT_DIR="${1:-${ROOT}/poke_bot/engine_rebuild/native/build}"
OUT="${OUT_DIR}/libcg_step_batch.so"

mkdir -p "${OUT_DIR}"
g++ -std=c++20 -O3 -fPIC -shared \
  -Wl,-soname,libcg_step_batch.so \
  -o "${OUT}" \
  "${SRC}" \
  -ldl

echo "built: ${OUT}"
ls -la "${OUT}"
# Optional: stage next to stock cg for easy discovery
STAGE_CG="${ROOT}/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg"
if [[ -d "${STAGE_CG}" ]]; then
  cp -f "${OUT}" "${STAGE_CG}/libcg_step_batch.so"
  echo "staged: ${STAGE_CG}/libcg_step_batch.so"
fi
