# Official baseline agents

These are **full Kaggle sample models** (each folder has `main.py` + `deck.csv`), not deck lists alone.
They are the rule-based opponents from the competition discussion until we consistently beat them.

Source: [Kaggle discussion #708584](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/708584)

## Install

```bash
bash scripts/setup_baseline_agents.sh
```

This pulls the four official Kaggle notebooks by Kiyota + deck datasets and writes:

| Directory | Agent |
|-----------|--------|
| `baselines/official/iono/` | Iono's Deck |
| `baselines/official/dragapult-ex/` | Dragapult ex Deck |
| `baselines/official/mega-abomasnow-ex/` | Mega Abomasnow ex Deck |
| `baselines/official/mega-lucario-ex/` | Mega Lucario ex Deck |

Each directory contains `main.py` + `deck.csv` extracted from:

- `kiyotah/a-sample-rule-based-agent-iono-s-deck` + `kiyotah/iono-deck`
- `kiyotah/a-sample-rule-based-agent-dragapult-ex-deck` + `kiyotah/dragapult-ex-deck`
- `kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck` + `kiyotah/mega-abomasnow-ex-deck`
- `kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck` + `kiyotah/mega-lucario-ex-deck`

The engine `cg/` package is **not** duplicated here — local play uses `kaggle/input/cg-lib` via `scripts/download-kaggle-inputs.sh`.

## Curriculum

1. **Baseline phase** — fine-tune our transformer vs these agents; rotate **our** decks from `decks/archetype-samples/` and save `outputs/checkpoints/by_deck/<slug>.pt`.
2. **Gate** — when aggregate win rate vs all loaded baselines ≥ `SELF_PLAY_BASELINE_WIN_RATE` (default 60%), switch to…
3. **Transformer self-play** — pure checkpoint-vs-checkpoint training (`run_self_play_loop`).

Run: `python scripts/run_self_play.py --baseline-only --checkpoint outputs/checkpoints/pre_self_train.pt`
