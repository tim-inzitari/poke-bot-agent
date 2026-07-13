# Baseline agents

Rule-based opponents for local evaluation. Each runnable agent folder has
`main.py` + `deck.csv` (Kaggle submission shape). The engine `cg/` package is
**not** duplicated here — use competition `sample_submission/cg` / `kaggle/input`.

**Payloads are gitignored.** Only this README and `manifest.json` are tracked.
Restore agents with:

```bash
bash scripts/download_baselines.sh
# or: python scripts/download_baselines.py
```

Flags: `--force` re-download; `--group official|community|roster`; `--only <id>…`.

## Layout

| Path | Tracked? | Contents |
|------|----------|----------|
| `baselines/manifest.json` | yes | Catalog of agents to download (source kernel refs) |
| `baselines/README.md` | yes | This file |
| `baselines/official/<id>/` | no | Official Kiyota samples |
| `baselines/community/<id>/` | no | Early community samples |
| `baselines/roster/<id>/` | no | Public-28 roster samples ([makimakiai roster notebook](https://www.kaggle.com/code/makimakiai/ptcg-public-28-plus-sample-4-roster-update)) |
| `baselines/decks/<id>/` | no | Deck CSV copies |
| `baselines/kernels/<id>/` | no | Pulled notebooks / metadata when available |

## Field size

`manifest.json` lists **29** agents (4 official + 5 community + 20 roster).
Eight additional roster refs from the Public-28 notebook are currently
inaccessible via the Kaggle API (403) and are recorded under
`field_notes.inaccessible_403` — retry later with `--force` if they open up.

## Official (Kiyota samples)

Source: [Kaggle discussion #708584](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/708584)

| Directory | Agent | Kernel |
|-----------|--------|--------|
| `baselines/official/iono/` | Iono's Deck | `kiyotah/a-sample-rule-based-agent-iono-s-deck` |
| `baselines/official/dragapult-ex/` | Dragapult ex Deck | `kiyotah/a-sample-rule-based-agent-dragapult-ex-deck` |
| `baselines/official/mega-abomasnow-ex/` | Mega Abomasnow ex Deck | `kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck` |
| `baselines/official/mega-lucario-ex/` | Mega Lucario ex Deck | `kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck` |

## Community samples

| Directory | Agent | Kernel |
|-----------|--------|--------|
| `baselines/community/cynthia-garchomp-ex/` | Cynthia's Garchomp ex | `masamikobayashi/a-sample-cynthia-garchomp-ex-deck` |
| `baselines/community/archaludon-ex/` | Archaludon ex / Cinderace | `masamikobayashi/a-sample-archaludon-75-wr-vs-my-1300-starmie` |
| `baselines/community/raging-bolt-ex/` | Raging Bolt ex | `yakitori55/a-sample-agent-raging-bolt-ex-deck` |
| `baselines/community/generic-heuristic/` | Deck-agnostic heuristic | `maximim/ptcg-generic-heuristic-baseline-agent` |
| `baselines/community/heuristic-baseline/` | Heuristic baseline | `serariagomes/heurestic-baseline-agent` |

## Roster samples

Twenty downloadable agents from the Public 28 + Sample 4 roster notebook live
under `baselines/roster/` (see `manifest.json` entries with `"group": "roster"`).
