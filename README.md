# poke-bot-agent

## CABT workflow

Run this notebook in VS Code:

```text
notebooks/poke_agent_unified.ipynb
```

It can be run end-to-end. On Mac it uses generated rollout data and trains with
Torch/MPS. On Kaggle/Linux it can also generate CABT rollouts if `cg-lib` is
available.

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

## Competition submission

Kaggle expects a `.tar.gz` with `main.py` and `deck.csv` at the top level.
Competition submissions are limited to 5 per team per day, so use Kaggle
simulation kernels and local container smoke tests for training/evaluation.
Only submit the tarball intentionally.

```bash
scripts/build_submission.sh
```
