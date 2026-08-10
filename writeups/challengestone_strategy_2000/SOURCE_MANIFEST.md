# Source manifest

This local write-up is a prose synthesis. It does not redistribute contest
engine code, model weights, replays, private data, external prose, logos, card
images, or other Pokémon artwork.

## Official public sources

- Kaggle Strategy overview and rubric:
  <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/overview>
- Kaggle Strategy rules:
  <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/rules>
- Kaggle Simulation task:
  <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/description>
- TCGplayer Alakazam guide:
  <https://www.tcgplayer.com/content/article/Alakazam-Deck-Guide-Pok%C3%A9mon-TCG/7eb46b82-9dc5-40d8-adf9-28cca05f070f/>
- PTCGO News Alakazam guide:
  <https://ptcgonews.com/tips/ptcgl-alakazam-deck-guide/>

## Repository evidence

- `GOAL.md`, revisions 175, 176, and 182
- `state/alakazam-rtp-owner-hard-swap-r175.json`
- `state/alakazam-public-multi-env-split-r182.json`
- `state/matchup_adapter_roster.json`
- `state/replay-model-inspector-activation-r176.json`
- `ops/elmo/replay-model-inspector-provenance-r176.json`
- `config/deck_guides/alakazam-final-refresh.yaml`
- `docs/deck_guides/alakazam-final-refresh-expert-brief.txt`
- `docs/RL_TRAINING_PROTOCOL.md`
- `docs/PURE_RL_PIPELINE.md`
- `experiments/recursive_turn_planner/README.md`
- `experiments/recursive_turn_planner/SWAP_IN.md`
- `outputs/state/alakazam-rtp-r175-collect-enablement.json`
- `outputs/analysis/kaggle-latest-replay-20260802/report-data.json`
- `decks/archetype-samples/alakazam-owner-rtp-pilot-r175.csv`
- `submission/main.py` and `scripts/build_submission.sh`, inspected as a dirty
  working tree and not treated as the historical submitted bytes

## Claim boundaries

- Public replay rates are descriptive historical observations, not a
  controlled current r175 estimate.
- RTP receipts prove collection wiring, not a strength improvement.
- The normal packaged contest controller is greedy/policy-only; the RTP
  sidecar checkpoint is not included in that package.
- Prior Alakazam completion was ceiling-accepted; it was not recorded as a
  measured terminal-gate pass.
