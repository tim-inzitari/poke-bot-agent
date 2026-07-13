#!/usr/bin/env bash
# Per-archetype training profiles. Source this and call:
#   load_archetype_profile <lucario|dragapult|abomasnow|iono|starmie|crustle>
#
# Each profile exports the env vars that scripts/run_archetype.sh consumes. This
# replaces the six near-identical run_<archetype>_*.sh scripts: the only real
# differences between archetypes are deck paths, GPU, model size, and whether the
# archetype is an official baseline agent (single deck) or a corpus archetype.

load_archetype_profile() {
  local arch="$1"

  # Shared defaults (overridden per archetype below).
  ARCH_GPU="3080"
  ARCH_BATCH_GAMES=12
  ARCH_WORKERS=12   # 3080 Ti default for self-play collect
  ARCH_TRAIN_DATA_DEVICE=auto
  ARCH_MODEL_DIMS=""                 # empty = config default (256d/6L)
  ARCH_BASELINE_DECKS_ONLY=1         # 1 = rotate baseline archetype lists; 0 = use ARCH_AGENT_DECK_DIR
  ARCH_AGENT_DECK_DIR=""             # only used when ARCH_BASELINE_DECKS_ONLY=0
  ARCH_MATCHUP_DIVERSITY=1           # 0 for single-deck archetypes (mirror-only bootstrap)
  ARCH_MERGE_LADDER=0                 # 1 = train on CABT bootstrap + ladder replays
  ARCH_LADDER_DATA="data/scraped_rollouts.jsonl"
  ARCH_CABT_DATA=""                   # defaults to ARCH_BOOTSTRAP_DATA when unset
  ARCH_BOOTSTRAP_EPISODES="${BOOTSTRAP_EPISODES:-5000}"

  case "$arch" in
    lucario)
      ARCH_SLUG="mega-lucario-ex"
      ARCH_DECK="decks/competitive/high_performing/2026-05_regional-melbourne-2026_10th_mega-lucario.csv"
      ARCH_FIELD_DIR="decks/lucario-only"
      ;;
    dragapult)
      ARCH_SLUG="dragapult-ex"
      ARCH_DECK="decks/competitive/high_performing/2026-05_regional-indianapolis-2026_3rd_dragapult-dusknoir.csv"
      ARCH_FIELD_DIR="decks/dragapult-only"
      ARCH_GPU="blackwell"
      ARCH_BATCH_GAMES=32
      ARCH_WORKERS=20
      ARCH_TRAIN_DATA_DEVICE=cuda
      ARCH_MODEL_DIMS="512 8 8 2048"
      ARCH_MERGE_LADDER=1
      ARCH_CABT_DATA="data/dragapult_bootstrap.jsonl"
      ARCH_LADDER_DATA="data/dragapult_ladder.jsonl"
      ARCH_BOOTSTRAP_DATA="data/dragapult_training.jsonl"
      ARCH_LADDER_EPISODES=5000
      ARCH_CABT_EPISODES=1000
      ;;
    abomasnow)
      ARCH_SLUG="mega-abomasnow-ex"
      ARCH_DECK="baselines/official/mega-abomasnow-ex/deck.csv"
      ARCH_FIELD_DIR="decks/abomasnow-only"
      ARCH_MATCHUP_DIVERSITY=0   # official single-deck archetype → mirror-only bootstrap
      ;;
    iono)
      ARCH_SLUG="iono"
      ARCH_DECK="baselines/official/iono/deck.csv"
      ARCH_FIELD_DIR="decks/iono-only"
      ARCH_MATCHUP_DIVERSITY=0
      ;;
    starmie)
      ARCH_SLUG="starmie"
      ARCH_DECK="decks/competitive/high_performing/2026-05_se-lima-2026_7th_starmie-froslass.csv"
      ARCH_FIELD_DIR="decks/starmie-only"
      ARCH_BASELINE_DECKS_ONLY=0
      ARCH_AGENT_DECK_DIR="decks/starmie-only"
      ;;
    crustle)
      ARCH_SLUG="crustle"
      ARCH_DECK="decks/competitive/high_performing/2026-04_regional-prague-2026_7th_crustle.csv"
      ARCH_FIELD_DIR="decks/crustle-only"
      ARCH_BASELINE_DECKS_ONLY=0
      ARCH_AGENT_DECK_DIR="decks/crustle-only"
      ;;
    *)
      echo "unknown archetype: ${arch}" >&2
      echo "valid: lucario dragapult abomasnow iono starmie crustle" >&2
      return 1
      ;;
  esac

  # Derived, identical-shape paths (the bulk of the old copy-paste).
  ARCH_NAME="$arch"
  if [[ -z "${ARCH_BOOTSTRAP_DATA:-}" ]]; then
    ARCH_BOOTSTRAP_DATA="data/${arch}_bootstrap.jsonl"
  fi
  if [[ -z "${ARCH_CABT_DATA:-}" ]]; then
    ARCH_CABT_DATA="$ARCH_BOOTSTRAP_DATA"
  fi
  ARCH_MODEL_OUTPUT="outputs/checkpoints/${arch}_fresh.pt"
  ARCH_TENSOR_CACHE_DIR="outputs/cache/training_tensors/${arch}_fresh"
  ARCH_SELF_PLAY_OUTPUT="outputs/rollouts/${arch}_self_play.jsonl"
  ARCH_SELF_PLAY_CKPT_DIR="outputs/checkpoints/self_play/${arch}"
}
