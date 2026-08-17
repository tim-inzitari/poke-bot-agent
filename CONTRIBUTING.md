# Contributing

This repository is an archival source release, but focused fixes and
reproducibility improvements are welcome.

1. Create a branch from `main`.
2. Keep competition assets, credentials, checkpoints, replays, and generated
   training output outside Git.
3. Install `.[dev,torch]` and run the relevant deterministic tests.
4. Add a regression test for behavior changes.
5. Keep native-engine, GPU, network, and long-running tests explicitly marked.

Before opening a change:

```bash
python -m compileall -q poke_bot scripts tests
python -m ruff check scripts/audit_public_release.py
python -m pytest -q tests/test_public_release_smoke.py
```

Do not submit private fleet addresses, credentials, downloaded third-party
payloads, models, or replay corpora. See `THIRD_PARTY_NOTICES.md` before adding
any externally sourced file.
