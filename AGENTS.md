# AGENTS.md

## Cursor Cloud specific instructions

This is a Python/PyTorch ML project for the Kaggle "Pokemon TCG AI Battle"
competition. It is not a client/server app — there is no web server or database.
The product is an offline pipeline: encode CABT game observations → train a
temporal transformer → package a Kaggle agent that plays TCG battles.

### Environment / interpreter
- The update script creates a `.venv`. Run everything with `.venv/bin/python`
  (or activate it). Do not use the system `python3` directly.
- `.python-version` pins 3.11 (for the Kaggle native `cg-lib`), but 3.11 is not
  available from apt on this Ubuntu 24.04 image. The `.venv` uses Python 3.12,
  which is fine for training and tests — `cg-lib` only needs PEP 604 syntax
  (3.10+). Torch runs on CPU here (no GPU); `poke_agent/device.py` auto-selects
  cuda/mps/cpu.

### Lint / test
- No linter is configured (no `ruff`/`flake8`/`pyproject.toml`/`setup.cfg`), so
  there is no "lint" step to run.
- Tests: `.venv/bin/python -m pytest tests/`. ~6 tests skip when optional Kaggle
  data / `cg-lib` is absent — that is expected, not a failure.

### Running training (the main dev workflow)
- Entry points: `scripts/train_agent.py`, `scripts/smoke_train.py`,
  `python -m poke_agent.main` (see `README.md`). Config lives in
  `poke_agent/config.py`.
- These need rollout JSONL. There is NO checked-in rollout data, and generating
  real data needs the Kaggle native simulator `libcg.so`
  (`scripts/download-kaggle-inputs.sh`, which requires `~/.kaggle/kaggle.json`).
- Non-obvious gotcha for a no-data smoke run: `scripts/smoke_train.py` reads
  `data/scraped_rollouts_smoke.jsonl` and that file must already exist. Rows must
  carry a real CABT-style `observation` dict — those encode to the genuine
  548-dim feature layout the model expects. The built-in synthetic fallback in
  `poke_agent/dataset.py` emits 32-dim features that fail the
  `assert_generic_model_inputs` guard, so it cannot train a real model on its own.

### Running the competition agent
- `submission/main.py`'s `agent()` imports `cg.api`, which loads
  `submission/cg/libcg.so` at import time. That native lib is Kaggle-only and is
  NOT in the repo, so `import main` fails without it.
- The actual decision core, `submission/policy_runtime.TrainedPolicyAgent`, runs
  without `libcg.so`: call `TrainedPolicyAgent(ckpt).choose_action(obs_dict,
  our_deck=deck)` on a plain obs dict to get a legal move. Checkpoint discovery
  order: `submission/value_model.pt`, `$VALUE_MODEL_PATH`, then
  `/kaggle_simulations/agent/value_model.pt`.

### Outputs
- `outputs/` and `data/` are gitignored; checkpoints/reports/caches land under
  `outputs/` at runtime.
