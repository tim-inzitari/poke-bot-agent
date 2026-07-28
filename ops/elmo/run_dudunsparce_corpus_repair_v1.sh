#!/usr/bin/env bash
set -euo pipefail

IMAGE="${POKEBOT_IMAGE:-poke-bot-truenas-worker:matchup-v38-record-schema-runtime}"
ARCHIVE_ROOT="/mnt/Main/main/poke-bot-agent/archive/episode-days"
LEGACY_ROOT="/mnt/Main/main/poke-bot-agent/archive/rare-route-expert-history-20260626-20260701/specialists-v1/dudunsparce"
STAGED_SOURCE="/mnt/Main/main/poke-bot-agent/staging/dudunsparce-repair-v1/src"
OUTPUT_ROOT="/mnt/Main/main/poke-bot-agent/archive/dudunsparce-verified-visual-schema-v2"
LOG_ROOT="/mnt/Main/main/poke-bot-agent/archive/dudunsparce-verified-visual-schema-v2"

test -f "${STAGED_SOURCE}/scripts/extract_verified_specialist_records.py"
test -f "${STAGED_SOURCE}/scripts/featurize_bootstrap_shard.py"
test -f "${STAGED_SOURCE}/scripts/assemble_feature_manifest.py"
test -f "${STAGED_SOURCE}/poke_bot/archetypes.py"
test -f "${STAGED_SOURCE}/poke_bot/authoritative_visual_trace.py"
test -f "${LEGACY_ROOT}/PROTECTED_EXPERT_CORPUS.json"
for day in 2026-06-26 2026-06-27 2026-06-28; do
  test -f "${ARCHIVE_ROOT}/pokemon-tcg-ai-battle-episodes-${day}.zip"
done

mkdir -p "${OUTPUT_ROOT}"
exec docker run --rm \
  --name pokebot-dudunsparce-corpus-repair-v1 \
  --label pokebot.managed-unit=dudunsparce-corpus-repair-v1 \
  --cpus 20 \
  --memory 48g \
  --entrypoint /bin/bash \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -v "${ARCHIVE_ROOT}:/archive/episode-days:ro" \
  -v "${LEGACY_ROOT}:/legacy/dudunsparce:ro" \
  -v "${STAGED_SOURCE}/scripts/extract_verified_specialist_records.py:/workspace/scripts/extract_verified_specialist_records.py:ro" \
  -v "${STAGED_SOURCE}/scripts/featurize_bootstrap_shard.py:/workspace/scripts/featurize_bootstrap_shard.py:ro" \
  -v "${STAGED_SOURCE}/scripts/assemble_feature_manifest.py:/workspace/scripts/assemble_feature_manifest.py:ro" \
  -v "${STAGED_SOURCE}/poke_bot/archetypes.py:/workspace/poke_bot/archetypes.py:ro" \
  -v "${STAGED_SOURCE}/poke_bot/authoritative_visual_trace.py:/workspace/poke_bot/authoritative_visual_trace.py:ro" \
  -v "${OUTPUT_ROOT}:/output" \
  -w /workspace \
  "${IMAGE}" \
  -lc '
    set -euo pipefail
    python -u scripts/extract_verified_specialist_records.py \
      --source-pointer /legacy/dudunsparce/PROTECTED_EXPERT_CORPUS.json \
      --archive-root /archive/episode-days \
      --output-dir /output \
      --specialist-id dudunsparce \
      --forbid-card-id 646 \
      --forbid-card-id 647 \
      --forbid-card-id 648 \
      --expected-records 29
    for day in 2026-06-26 2026-06-27 2026-06-28; do
      shard="/output/verified-${day}.dudunsparce.features"
      if test ! -f "${shard}"; then
        python -u scripts/featurize_bootstrap_shard.py \
          --jsonl "/output/verified-${day}.dudunsparce.jsonl" \
          --out "${shard}" \
          --source-date "${day}" \
          --workers 8 \
          --max-in-flight 16 \
          --max-context 320 \
          --compact-mode temporal-expert-v1 \
          --required-archetype dudunsparce
      fi
    done
    python -u scripts/assemble_feature_manifest.py \
      --staging-dir /output \
      --out /output/manifest.json \
      --expected-date 2026-06-26 \
      --expected-date 2026-06-27 \
      --expected-date 2026-06-28 \
      --min-free-gib 10 \
      --compact-mode temporal-expert-v1 \
      --required-archetype dudunsparce \
      --expected-max-context 320 \
      --require-target-coverage temporal_action_rows \
      --require-target-coverage opponent_hand_rows \
      --require-target-coverage opponent_remainder_rows \
      --require-target-coverage opponent_private_prize_rows \
      --require-target-coverage lethal_threat_rows \
      --require-target-coverage prize_race_rows \
      --seal-protected
  ' >>"${LOG_ROOT}/build.log" 2>&1
