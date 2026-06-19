# poke-bot-agent

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system design:
deployment topologies, data flow, model architecture, rollout schema, configuration
reference, and competition submission path. Module-level API detail lives in
[docs/poke-agent-modules.md](docs/poke-agent-modules.md).

Train outside the notebook (settings in `poke_agent/config.py`):

```bash
python scripts/train_agent.py
```

## CABT workflow

Run this notebook in VS Code:

```text
notebooks/poke_agent_training.ipynb
```

The training pipeline lives in the `poke_agent/` package. The notebook imports those
modules step-by-step so you can inspect intermediate state between cells.

The legacy monolithic notebook is still available at
`notebooks/poke_agent_unified.ipynb`.

Default path: run CABT simulation on Kaggle, then train locally on Mac with
Torch/MPS.

```bash
scripts/run_cabt_kaggle.sh
kaggle kernels status timinzitari/poke-agent-cabt-simulation
kaggle kernels output timinzitari/poke-agent-cabt-simulation -p data/kaggle-output
```

Optional local container smoke test, only if Docker is installed:

```bash
scripts/download-kaggle-inputs.sh
scripts/run_cabt_container.sh --episodes 1 --out data/cabt_rollouts.jsonl
```

The container is for CABT simulation only. Keep Torch training native in VS Code
so PyTorch can use Apple Silicon via `mps`.

## Elmo Simulation Worker

Elmo is the preferred CABT rollout worker if it is your TrueNAS box with the
Ryzen 5950X. Run CABT there, then pull the JSONL back to this Mac for Torch/MPS
training.

On Elmo (requires Python 3.11 for `cg-lib`):

```bash
git clone https://github.com/tim-inzitari/poke-bot-agent.git
cd poke-bot-agent
scripts/elmo_setup.sh
scripts/elmo_generate.sh --episodes 1000 --workers 16 --out data/elmo-rollouts.jsonl
```

`scripts/elmo_setup.sh` creates `.venv` with Python 3.11. On Debian/Ubuntu:
`sudo apt install python3.11 python3.11-venv`.

On this Mac:

```bash
ELMO_HOST=elmo ELMO_PATH=~/poke-bot-agent scripts/pull_elmo_rollouts.sh
```

Start with `--workers 16`; try `--workers 24` after benchmarking. The 5950X has
32 threads, but using all of them may not be fastest if the simulator or NAS I/O
gets noisy.

### Hard Deck Pools

Put hard-coded deck lists in `decks/`. Each file can be `.csv`, `.txt`, or
`.deck`, and must contain exactly 60 card IDs, one per line or comma-separated.

Generate sampled matchups from the same pool:

```bash
scripts/run_cabt_container.sh \
  --episodes 100 \
  --deck-dir decks \
  --matchups sample \
  --out data/deckpool-rollouts.jsonl
```

Generate the default local 10k-row-ish full-state training file used by the
notebook. Rows include compact features plus the full current observation,
chosen action, next observation, terminal flag, simulator result, and value:

```bash
scripts/run_cabt_container.sh \
  --episodes 220 \
  --workers 4 \
  --max-steps 300 \
  --deck-dir decks \
  --matchups sample \
  --seed 17 \
  --out data/mac-rollouts-10k-fullstate.jsonl
```

Fetch the official Kaggle submission results. This is the real competition API
score history; it is submission-level, not per-position training data:

```bash
python scripts/fetch_competition_results.py
```

Generate deterministic round-robin matchups:

```bash
scripts/run_cabt_container.sh \
  --episodes 100 \
  --deck-dir decks \
  --matchups round-robin \
  --out data/deckpool-rollouts.jsonl
```

Use separate pools for each side (requires Python 3.11 in `.venv`; see Elmo setup):

```bash
.venv/bin/python scripts/generate_cabt_data.py \
  --episodes 1000 \
  --workers 16 \
  --deck0-dir decks/ours \
  --deck1-dir decks/opponents \
  --matchups sample \
  --out data/elmo-hard-decks.jsonl
```

## Competition submission

Kaggle expects a `.tar.gz` with `main.py` and `deck.csv` at the top level.
Competition submissions are limited to 5 per team per day, so use Kaggle
simulation kernels and local container smoke tests for training/evaluation.
Only submit the tarball intentionally.

Download the simulator library first (the tarball must include `cg/libcg.so`):

```bash
scripts/download-kaggle-inputs.sh
scripts/build_submission.sh
```
