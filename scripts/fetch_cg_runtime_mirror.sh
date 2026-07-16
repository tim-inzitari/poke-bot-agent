#!/usr/bin/env bash
# Fetch competition sample_submission/cg (libcg.so + Python bindings) from the
# public PrimeIntellect research-environments PR snapshot when Kaggle creds are
# unavailable. Does NOT fetch ptcg_engine C++ sources (still Kaggle-only).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission"
# Pinned SHA from PR #598 (contains official sample_submission/cg artifacts).
SHA="${CG_MIRROR_SHA:-6d0c449d348d5f67120a0a8a255473b41bbb136f}"
BASE="https://raw.githubusercontent.com/PrimeIntellect-ai/research-environments/${SHA}/environments/pokemon_tcg_ai_battle/official/sample_submission"

mkdir -p "${DEST}/cg"
echo ">> Fetching cg runtime → ${DEST}"
for f in cg/sim.py cg/api.py cg/game.py cg/utils.py cg/__init__.py \
         cg/libcg.so cg/libcg-arm64.so cg/libcg.dylib cg/cg.dll \
         deck.csv main.py; do
  echo "   ${f}"
  curl -fsSL "${BASE}/${f}" -o "${DEST}/${f}" || {
    echo "WARN: failed ${f}" >&2
  }
done
chmod +x "${DEST}/cg/libcg.so" 2>/dev/null || true
ls -la "${DEST}/cg/libcg.so"
echo ">> Done. ptcg_engine C++ still needs: bash scripts/setup_competition_data.sh (Kaggle auth)"
