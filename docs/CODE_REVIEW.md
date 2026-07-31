# Senior engineer code review — rl-libs

Review date: 2026-07-31. Production poke-bot was not touched.

## 0.2.0 remaining extracts

Added Python-first packages with opaque contracts only. Senior pass:

| Package | Notes |
|---|---|
| `rl_resource` | No poke-bot env names hardcoded in core API; knobs caller-owned |
| `schedule_engine` | No fixed 25-epoch / Pokemon head inventory |
| `test_profiles` | Manifest commands; no repo-rooted canary scripts |
| `submit_guard` | Generic schema; Pokemon turn-order attestation left to `extra_checks` |
| `log_trim` | Uses receipt helper; dry-run supported |

## 0.1.1 C++ core review

## Verdict

Ship as a **v0.1 library** after the correctness fixes below. API shape is good
(opaque payloads, no domain schema). Several concurrency / durability bugs were
not acceptable for a crash-safe or SHM library.

## Findings (pre-fix)

| Sev | Area | Issue |
|---|---|---|
| P0 | `ShmRing::submit` | Published `state=1` **before** writing rid/length/payload → consumer could observe torn slots |
| P0 | `OrderedWriter::drain_ready_` | Advanced `next_index_` **before** durable `commit_` → crash mid-commit could skip indices on resume |
| P0 | `Supervisor::submit` | Detached IO threads + FD close on recycle → use-after-close races |
| P1 | `Supervisor::monitor_` / fork | Child inherited sibling pipe FDs; no frame-size cap |
| P1 | `ShmRing::create` | `memset` over live `std::atomic` storage (UB) |
| P1 | Tests | No crash-resume test for ordered writer; no multi-producer SHM test |
| P2 | `ShmRing` | Busy-wait + 50µs sleep caps throughput (~1.6k rt/s in bench) |
| P2 | `proc_pool` | Far thinner than poke-bot `WorkerPool` (no slot leases / capacity grace) — OK if documented |
| P2 | Python packages | `torch_ckpt` soft-depends on `rl_io` for digests; fine with hashlib fallback |
| P2 | CI | Nested `working-directory: rl-libs` breaks on `lib/rl-libs` root layout |

## Fixes in 0.1.1

1. SHM: claim head via CAS, write payload, **then** release-store ready state.
2. Ordered writer: collect batch without committing index; advance only after fsync+state.
3. Supervisor: joinable per-slot IO thread; `O_CLOEXEC` pipes; close leaked FDs in child; 64 MiB frame cap.
4. SHM create: zero non-atomic bytes, placement-init atomics.
5. Tests: resume-after-abort; multi-client SHM echo.
6. Standalone publish docs/scripts aligned with `wave-dispatch`.

## Residual / follow-ups (not blockers)

- Replace SHM busy-wait with futex/eventfd for speed.
- Optional io_uring path (as in wave_dispatch).
- Stronger `proc_pool` lease model if replacing production `WorkerPool`.
- Wire-up into poke-bot only when explicitly ordered.
