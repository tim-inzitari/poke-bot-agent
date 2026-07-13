# poke-bot-agent

Clean-slate branch (`A-New-Hope`). Rebuild training/agent code from scratch.

## Kept

| Path | Purpose |
|------|---------|
| `decks/` | Deck-list backlog |
| `baselines/` | Official / sample baseline agents |
| `cards/EN_Card_Data.csv` | Card ID reference |
| `containers/cabt/` | CABT simulator container recipe |
| `kaggle/cabt-sim/` | Kaggle kernel metadata for CABT sim |
| `requirements.txt` | Python deps |
| `scripts/elmo_setup.sh` | Local `.venv` (Python 3.11) |
| `scripts/download-kaggle-inputs.sh` | Fetch `cg-lib` + competition inputs |

Kaggle API credentials: `~/.kaggle/kaggle.json` (outside the repo, not committed).

## Setup

```bash
scripts/elmo_setup.sh
scripts/download-kaggle-inputs.sh   # needs Kaggle CLI + login
```

## Competition submission

Kaggle expects a `.tar.gz` with `main.py` and `deck.csv` at the **top level** of the
archive (not inside a subfolder), plus `cg/libcg.so` and the trained checkpoint.
Competition submissions are limited to 5 per team per day.
Only submit the tarball intentionally.

```bash
# after you have a submission package directory with main.py + deck.csv + cg/ + model:
tar -czf dist/submission.tar.gz -C /path/to/package .
kaggle competitions submit -c pokemon-tcg-ai-battle -f dist/submission.tar.gz -m "message"
```
