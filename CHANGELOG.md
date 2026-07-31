# Changelog

## 0.1.1

- Code review fixes (see `docs/CODE_REVIEW.md`):
  - SHM: claim-then-publish protocol; no torn slots; safer mapping init
  - Ordered writer: advance durable index only after fsync+state commit
  - Process pool: joinable IO threads, CLOEXEC pipes, 64MiB frame cap
- Tests: ordered-writer crash resume; multi-producer SHM
- Standalone publish path (`lib/rl-libs`, bundle, `PUBLISH.md`)

## 0.1.0

- Initial extract: `rl_io`, `rl_runtime`, `proc_pool` C++ cores with pybind11.
- Python-first: `rl_eval`, `torch_ckpt`, `artifact_registry`.
- Apps: `rl_io_bench`, `rl_shm_bench`, `rl_echo_worker`.
- No production poke-bot rewiring.
