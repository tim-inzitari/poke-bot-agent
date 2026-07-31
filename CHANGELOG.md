# Changelog

## 0.2.0

- Complete remaining generalized extracts (Python-first):
  - `rl_resource` — RAM/CPU/GPU sample, `OomGuard`, advisory knob ratchet
  - `schedule_engine` — opaque staged curriculum + digests
  - `test_profiles` — manifest-driven timed pytest/command profiles
  - `submit_guard` — one-shot fail-closed competition submit grants
  - `log_trim` — age/size directory trim with optional receipts
- Docs/mapping updated; production poke-bot still unwired

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
