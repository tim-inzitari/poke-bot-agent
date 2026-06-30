# Kaggle Public Rule-Based Baselines

Research date: 2026-06-23

This note summarizes public Kaggle discussion/notebook signals for high-performing rule-based Pokemon TCG AI Battle agents. I pulled public notebooks with `kaggle kernels pull` where Kaggle allowed it, and used the public 11-agent round-robin matrix from `makimakiai/ptcg-public7-sample4-roundrobin-inputs` as the main empirical signal.

## Best Public Baseline Families

| Rank signal | Agent / family | Public result signal | Main idea | Why it matters for us |
| --- | --- | ---: | --- | --- |
| 1 | Kokinn Lucario, fighting search aggro | 63%, 63-37 in public 7 + sample 4 matrix | Lucario shell with search/anti-Crustle tuning | Strongest public matrix result; use as target behavior for Fighting aggro baseline |
| 2 | Biohack Crustle, grass sustain control | 63%, 63-37 | Crustle wall/control deck with very simple priorities | Deck construction carries the policy; useful opponent baseline |
| 3 | Yaroslav Lucario, anti-wall beat | 62%, 62-38 | Mega Lucario with Crustle-aware rules | Good template for matchup-aware rule guards |
| 4 | Penguin Lucario, public 915+ beatdown | 58%, 58-42 | Lucario beatdown | Another strong Lucario baseline for matchup testing |
| 5 | Kacchan Lucario, anti-wall midrange | 58%, 58-42 | Lucario with anti-wall midrange plan | Useful to avoid overfitting only to aggro Lucario |
| 6 | Roman Crustle+Lucario V7 / LB960+ | 56%, 56-44 | Thick Lucario aggro plus Crustle ideas | Good hybrid target |
| Public notebook | Alakazam, rule-based not psychic | Title claims best 5th | Hand-size damage engine with draw/deckout guards | Good non-Lucario, non-Crustle baseline |
| Public notebook | Dragapult ex, Phantom Dive | Public notebook, meta-counter thesis | Plans multi-prize turns and bench damage counters | Good higher-complexity rule policy to imitate |

The public matrix ran 11 agents, 55 pairs, 550 games, and reported 0 errors. Treat these as local matchup-test signals, not leaderboard guarantees.

## Reusable Rule Patterns

### 1. Action Scoring Beats Flat Priority

All useful public agents score every legal option, sort descending, and return the top `maxCount` options. The simple Crustle policy is the minimal version:

- Attach before evolve/play.
- Evolve before play.
- Use abilities before attack.
- Attack late because it often ends the turn.
- Retreat only when a plan requires it.
- For forced sub-selections, prefer active targets, high-energy targets, high-HP allied Pokemon, or context-specific evolution/search targets.

### 2. Deck Choice Can Outweigh Policy Complexity

The Day-1 Crustle notebook explicitly says the agent is intentionally simple: the deck is stable enough that a plain rule policy performs well. Its key logic is setup first, attack last, plus special cases for healing/tools. This is important for our training data: include simple strong-deck pilots, not only smart generic pilots.

### 3. Add Matchup Guards

The better Lucario baselines include explicit Crustle guards:

- Detect Dwebble/Crustle on opponent board.
- Avoid wasting Mega Lucario ex attacks into Crustle wall.
- Keep a non-ex Hariyama route alive against Crustle.
- Weight Fighting energy toward Hariyama when facing Crustle.

This is exactly the kind of rule feature we should expose to the neural value model as a target prior.

### 4. Use Attack Plans, Not Just Current Active

The stronger Lucario and Dragapult policies build an `AttackPlan`:

- Enumerate possible attackers across active + bench.
- Include switch/retreat/gust availability.
- Estimate energy requirements and one-turn attach bridges.
- Score targets by prize count, energy/tools attached, stage, remaining HP, weakness/resistance, and whether the KO wins the game.
- Then score retreat, attach, gust, attack, and target-selection options according to that plan.

This is the best baseline shape for our beam-search value function.

### 5. Protect Against Deckout

Alakazam and Lucario public agents use low-deck guards:

- Avoid draw/search cards below a deck threshold.
- Avoid abilities that draw when the deck is too thin.
- Account for card effects that draw/search multiple cards, not just "draw is good".

This matters because our current model can learn bad draw greed unless rollouts punish deckout clearly.

### 6. Preserve Bench Slots

Alakazam has an explicit bench-slot concept:

- Early: bench key Basics.
- Later: avoid flooding support Pokemon.
- Keep at least one bench slot open for the engine's next required piece.

This can be generalized into a board-shape heuristic for any deck.

### 7. Track Once-Per-Turn / Global State

Public agents use module-level turn state for:

- Ability-used flags.
- Attack plan reset at new turn.
- Log-derived state in Dragapult.
- Prize/card counts.

For our policy runtime, keep this deterministic and reset on new games/turns.

## Baseline Recommendation

For our next local baseline, implement a "public-style planner" rather than copying public notebook code:

1. Generic option scorer with type priorities and safe forced-selection handling.
2. Deck-specific `AttackPlan` layer for our current deck.
3. Matchup detectors for Crustle wall, Dragapult line, water ramp/Abomasnow, Lucario line, and hand-size Alakazam.
4. Low-deck and bench-slot guards.
5. Training hooks that log the top rule score, action type, target score, and plan fields into rollout rows.

This gives us a high-performing rule baseline and better imitation/RL features without depending on public code wholesale.

## Source Links

- Public round-robin matrix notebook: https://www.kaggle.com/code/makimakiai/ptcg-public-7-plus-sample-4-round-robin-visual
- Round-robin input dataset: https://www.kaggle.com/datasets/makimakiai/ptcg-public7-sample4-roundrobin-inputs
- Crustle Day-1 baseline: https://www.kaggle.com/code/dashimaki360/beating-the-day-1-1-crustle-bot
- Simple Lucario baseline and matchup tests: https://www.kaggle.com/code/kojimar/simple-baseline-matchup-tests
- Alakazam rule baseline: https://www.kaggle.com/code/ryotasueyoshi/rule-based-not-psychic-alakazam-best-5th
- Dragapult rule baseline: https://www.kaggle.com/code/skarin/phantom-dive-or-go-home-a-dragapult-ex-deck
- Heuristic agent/data pipeline: https://www.kaggle.com/code/avikdas567/ptcg-ai-battle-heuristic-agent-data-pipeline
- Public advanced Lucario heuristic/search agent: https://www.kaggle.com/code/nursrijan/pokemon-ai-battle-agent-mega-lucario
