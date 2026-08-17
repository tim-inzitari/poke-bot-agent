# Deterministic parallel replay preparation

Status: source-only, opt-in, not deployed or activated.

Revision 306 fixes production RL-window preparation at exactly 16 local Inzi
workers on subsequent training starts. Other supported counts remain
diagnostic-only. The active revision-304 optimizer is excluded.

## Architecture

`poke_bot/pure_rl/replay_parallel_prepare.py` scans an immutable newline JSON
shard once, hashes it, and records every row boundary. It divides rows into
`workers × ranges_per_worker` canonical ranges (one per worker by default), so
32 workers can consume 32 ranges even when the old replay cache contains only eight
partitions. A range identity binds the source digest, ordinal, row interval and
byte interval.

Each spawned worker receives only paths and scalar range metadata. It validates
the source stat, reads exactly its newline-aligned interval, decodes each game
once, runs the existing compact-game converter, and therefore preserves the
existing OwnDeckLedger reconstruction, tactical/strategic labels, current-deck
guide targets, masks, matchup route labels, action ordering and max-context
rules. It immediately packs its games into the existing contiguous tensor/CSR
representation. Variable-length games use `game_decision_offset` and
`game_sample_offset`; sparse board, option, action, hand and remainder data use
flat arrays plus offsets. Only a small descriptor returns to the parent—never
the Python game corpus.

Fragments are fsync'd and checksum-bound. The parent limits in-flight work,
rejects failed/duplicate/missing ranges, rechecks the source stat, validates
every fragment checksum and tensor inventory, and merges fragments strictly by
range ordinal. Worker scheduling cannot alter ordering. The merger maps one
fragment at a time and writes directly into the final contiguous allocation,
bounding peak memory to final-pack bytes plus the largest mapped fragment and a
configurable reserve.

The completed pack contains CPU tensors and can be page-locked with
`pin_cpu_corpus` before bounded nonblocking GPU transfers. Its semantic digest
excludes worker count, timing and serialization bytes, so builds at 1/2/4/8/16/32
workers must agree. The cached pack is reusable across optimizer epochs and
later refreshes when its source and semantic build key are unchanged. Existing
replay-cache, expert-pack and serial `GameSequence` formats remain readable.

## Repeated pure-RL window packing

`scripts/build_parallel_rl_replay_window.py` is the Inzi-only entrypoint for
future pure-RL starts. It accepts the rolling sealed shard list in exact
iteration order. Each immutable shard receives a content-addressed component
pack, so later updates reuse prior components and prepare only newly sealed
shards. The accumulated window is merged in canonical source/game order.

An aligned `adapter-routing.pt` sidecar stores routes, seats, source rows,
decision counts, episode identities, and canonical training tickets as flat
numeric or UTF-8 byte arrays with offsets. Worker queues contain only paths and
small descriptors. A separate `side-tensors.pt` retains the OwnDeckLedger
inputs and tutor/terminal/tactical option targets in the same game/decision/
stage order. Core, routing, and side-pack checksums all contribute to the
semantic output identity.

## Failure and fallback behavior

- Partial source rows, source stat drift, worker exceptions, corrupt fragment
  checksums, duplicate/missing ranges, changed tensor inventories and
  cross-topology output drift fail closed.
- Staging uses a nonce and is never published as the final output. Interrupted
  builds leave diagnostic staging; a retry uses a new staging directory.
- `prepare_with_serial_fallback` falls back only for backend availability/I/O
  failures. Corrupt data and validation failures never silently fall back.
- Strict parallel mode re-raises backend-unavailable failures.
- The current serial loader remains unchanged and remains the default.

## Verification contract

`validate_corpus_parity` requires identical scalar state, inventory, dtype,
shape and exact tensor values. Integer tensors, masks, offsets, identities and
ordering are always exact. Floating tensors are also exact by default; a caller
must explicitly set and receipt-bind a nonzero absolute tolerance.
`validate_one_step_result` applies the same policy to one identically seeded
optimizer step. The benchmark driver tests cache reuse and requires one
semantic output digest across worker counts.

Focused failure tests cover all supported worker counts in the range planner,
truncated input, duplicate/missing ranges, stale source identity, strict and
non-strict fallback, and refusal to hide corruption. Fragment corruption,
inventory and merge-order validation reuse the existing expert parallel-pack
validators.

## Isolated benchmark (512 games)

The local fixture contained 512 source games, 12,288 decisions and 58.1 MB of
JSONL. All 44 packed tensors were bitwise identical to serial preparation, all
worker counts produced the same semantic digest, and an identically seeded
one-step AdamW probe consumed the same packed value targets and produced
bitwise-identical model and optimizer tensors.

| Workers | Build seconds | Speedup vs serial | Cache reuse |
| ---: | ---: | ---: | ---: |
| serial | 5.09 | 1.00x | n/a |
| 1 | 6.09 | 0.84x | 0.07 s |
| 8 | 3.29 | 1.55x | 0.07 s |
| 16 | 5.22 | 0.98x | 0.07 s |
| 32 | 9.78 | 0.52x | 0.08 s |

The diagnostic fixture favored eight workers, but revision 306 explicitly
selects 16 workers for subsequent Inzi production packing. This preparation
topology choice cannot alter tensors, ordering, or optimizer inputs. Peak
RSS reported here is parent-process RSS (serial 449 MB; eight-worker parent
656 MB), not aggregate child RSS. GPU starvation was not measured because the
current trainer and GPUs were intentionally left untouched.

## Safe activation at the next sealed boundary

Do not activate during the current iteration. After the trainer has committed
and stopped at a sealed boundary:

1. Copy the new module and benchmarked receipt into a new immutable deployment
   root; do not edit the active deployment in place.
2. Add an explicit trainer integration that selects the joined pack only after
   validating the receipt, source digest, checkpoint/deck contract, decision
   counts and exact serial-parity digest. Preserve the serial branch.
3. Add a systemd drop-in for the future managed trainer with
   `PURE_RL_PREP_BACKEND=parallel`, `PURE_RL_PREP_WORKERS=16`,
   `PURE_RL_PREP_STRICT=0`, the immutable pack root and validation-receipt path.
4. Run `systemctl --user daemon-reload`, then start the trainer only at that
   sealed boundary. Never restart an in-flight collection or optimizer.
5. Confirm the dashboard reports pack validation, cache hit/miss, preparation
   range progress, CPU utilization, RSS and GPU feed rate. Roll back by removing
   the drop-in at a later sealed boundary; the serial loader remains intact.

Files that would change during a future activation (not changed now):

- `poke_bot/train.py`: consume verified packed CPU batches in the RL optimizer.
- `scripts/train_pure_rl.py`: resolve and receipt-bind the backend selection.
- `scripts/build_parallel_rl_replay_window.py`: build/reuse the canonical
  16-worker Inzi-local replay-window pack and adapter sidecar.
- `ops/systemd/pokebot-alakazam-r274-rl.service.d/<next-revision>-parallel-prep.conf`:
  opt-in environment only.
- `dashboard/lan/server.py` and `dashboard/lan/index.html`: expose range/cache/GPU
  feed telemetry.
