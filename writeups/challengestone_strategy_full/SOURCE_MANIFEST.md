# Source manifest and claim boundaries

## Official public sources

- Strategy overview and rubric:
  <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/overview>
- Strategy rules:
  <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/rules>
- Simulation task:
  <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/description>
- TCGplayer Alakazam guide:
  <https://www.tcgplayer.com/content/article/Alakazam-Deck-Guide-Pok%C3%A9mon-TCG/7eb46b82-9dc5-40d8-adf9-28cca05f070f/>
- PTCGO News Alakazam guide:
  <https://ptcgonews.com/tips/ptcgl-alakazam-deck-guide/>

External strategy sources were paraphrased and cited. Their prose, tables,
images, and card artwork are not redistributed.

## Canonical design and runtime sources

- `GOAL.md`, especially revisions 175, 176, and 182
- `state/alakazam-rtp-owner-hard-swap-r175.json`
- `state/alakazam-public-multi-env-split-r182.json`
- `state/matchup_adapter_roster.json`
- `config/deck_guides/alakazam-final-refresh.yaml`
- `docs/RL_TRAINING_PROTOCOL.md`
- `docs/PURE_RL_PIPELINE.md`
- `poke_bot/model.py`, `poke_bot/agent.py`, and matchup/RTP modules
- `experiments/recursive_turn_planner/README.md`
- `experiments/recursive_turn_planner/SWAP_IN.md`
- `submission/main.py` and `scripts/build_submission.sh`

The last two files are modified in the working tree and are not treated as the
historical submitted bytes.

## Exact deck and strategy sources

- `decks/archetype-samples/alakazam-owner-rtp-pilot-r175.csv`
- `cards/EN_Card_Data.csv`
- `docs/deck_guides/alakazam-final-refresh-expert-brief.txt`

## Receipts and analysis sources

- `outputs/state/alakazam-rtp-r175-collect-enablement.json`
- `outputs/state/alakazam-rtp-r175-remote-sync.json`
- `state/replay-model-inspector-activation-r176.json`
- `ops/elmo/replay-model-inspector-provenance-r176.json`
- `outputs/analysis/kaggle-latest-replay-20260802/report-data.json`
- associated historical Alakazam analysis/head-attribution JSON
- `state/final_format_alakazam_iteration15_allocator_recovery_completion_r105.json`
- `state/final_format_marnie_iteration0_commit_recovery_r107.json`
- Crustle r167 commit/milestone receipts
- Marnie r138/r150 family-study and rollback receipts
- `state/slowking_failed_experiment_transition_r79.json`
- `ops/ACTIVE_ALAKAZAM_GATE.md`

## Non-claims

- There is no completed current-r175 formal gate in the local evidence.
- There is no source-backed current-r175 Kaggle score or leaderboard rank.
- RTP evidence proves trusted collection wiring, not a strength gain.
- Public replay samples are not controlled estimates and are not training data.
- A physical head/route or adapter parameter does not prove decision-level use.
- The prior Alakazam parent was ceiling-accepted, not a measured terminal pass.
- PokeRLM research-roadmap features are not presented as deployed.

## Excluded material

The package contains no contest engine, Sponsor/Kaggle data, replay bytes,
model weights, private resource, card image, logo, or other Pokémon artwork.
