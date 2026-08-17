# Poke Bot Agent

Source release of the agent developed for Kaggle's 2026
`pokemon-tcg-ai-battle` competition. The competition has ended; this repository
is published for engineering reference and reproducibility.

The system combines a causal Transformer policy/value model, factorized legal
action decoding, advantage-weighted self-play training, immutable replay
lineage, and optional remote whole-game simulation workers. The code reflects
the final competition development tree, with private fleet defaults and
operational artifacts removed.

## What is included

- policy, feature, action, training, replay, and simulation source under
  `poke_bot/`;
- competition and evaluation entry points under `scripts/`;
- deterministic tests and fixtures under `tests/`;
- public model/deck configuration and documentation;
- submission packaging source.

Models, replay corpora, checkpoints, Kaggle credentials, native competition
engine binaries, private deployment configuration, and live operational
receipts are intentionally not included.

## Install

Python 3.11 is the supported baseline.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,torch]'
```

The native game runtime and card data were distributed through Kaggle. Obtain
them through your own authorized Kaggle account, then run:

```bash
bash scripts/setup_competition_data.sh
```

Those assets retain their original terms and are not covered by this
repository's MIT license.

## Verify the source checkout

```bash
python -m compileall -q poke_bot scripts
python -m pytest -q tests/test_public_release_smoke.py
```

The full historical test inventory is retained, but many tests intentionally
fail closed without private authority receipts or require the competition
engine, CUDA, downloaded baselines, or a fleet. See
`docs/PURE_RL_PIPELINE.md` for the final training contract and `pytest.ini` for
test markers.

## Main entry points

```bash
# Inspect trainer options without launching a run
python scripts/train_pure_rl.py --help

# Run the deterministic quick test profile
POKEBOT_PYTHON="$(command -v python)" bash scripts/test_quick.sh

# Package a checkpoint after supplying the competition runtime yourself
bash scripts/build_submission.sh /path/to/model.pt dist/package
```

Host-specific fleet execution is opt-in. Configure endpoints and roots through
the documented `POKEBOT_*` environment variables; no private fleet is assumed
by this release.

## Provenance

This is a fresh-history source export from private development commit
`9b3225215a3e06c156916c52218fb16667914e33`. See
`SOURCE_PROVENANCE.md` for exclusions and release-time portability changes.

## License and project status

Owned source is released under the [MIT License](LICENSE). Third-party names,
game data, card text, competition assets, and trademarks remain the property of
their respective owners. This project is unaffiliated with The Pokemon Company,
Nintendo, Creatures, GAME FREAK, or Kaggle.

The competition is over and this repository is provided as-is. Security reports
are still welcome through the process in [SECURITY.md](SECURITY.md).
