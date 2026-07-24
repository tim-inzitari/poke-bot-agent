# Canonical Specialist Runtime Configuration

The sequential specialist trainer has exactly three configuration sources:

1. `config/specialist_runtime.env` owns host capacity, fleet endpoints, memory
   limits, and the single active-specialist selector.
2. `ops/specialist_runtime_registry_v1.json` owns every specialist-specific
   checkpoint, checksum, expert corpus, matchup runtime, head-training setting,
   gate, iteration bound, and trainer argument.
3. `ops/systemd/pokebot-pure-rl-specialist.service` owns only process lifecycle
   and resource containment.

Systemd drop-ins are not part of the canonical design. A deployment must replace
the previous unit and its entire drop-in directory atomically while the trainer
is stopped. It must never copy historical `.staging/*.conf` files.

The active specialist's pass handler is also a complete unit, currently
`ops/systemd/pokebot-trevenant-passed-gate-handler.service`. It reads the same
environment file and resolves its command from the selected specialist's
`pass_handler` registry record through
`scripts/launch_active_specialist_gate_handler.py`. It may observe and finalize
a passing checkpoint, but it does not own or duplicate trainer configuration.

## Changing specialists

1. Add or complete exactly one specialist record, including its
   `pass_handler` record, in
   `ops/specialist_runtime_registry_v1.json`.
2. Validate every registered checkpoint, dataset, matchup tree, authorization
   receipt, and checksum.
3. Change only `POKEBOT_ACTIVE_SPECIALIST` in
   `config/specialist_runtime.env`.
4. Run `scripts/launch_active_specialist.py --check`.
5. Start the canonical service.

The selector fails closed when the selected specialist is absent, not ready, or
has mismatched artifacts. Only one canonical service may run at a time.

The current protocol bounds are a floor of 5 completed RL iterations and a
ceiling of 15. The registry expresses the ceiling as `--iterations 16` because
iteration numbering begins at zero.
