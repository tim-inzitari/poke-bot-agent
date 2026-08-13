---
name: update-durable-goal
description: Safely create, update, activate, and chain durable-goals packages when the owner changes an objective, invariant, completion condition, delegation, authority rule, activation state, or says one goal should follow another. Use for prompts such as "update the goal," "record this decision," "after this goal do X," "change the authoritative goal," "activate revision N," or "refresh goal status." Do not use for ordinary implementation progress that supplies no owner goal change.
license: MIT
metadata:
  durable-goals-version: "0.1"
---

# Update Durable Goal

Record owner intent without confusing it with activated reality.

Requires filesystem access and the `dgoal` CLI or the Python 3.11+
`durable_goals` package for deterministic writes.

## Workflow

1. Read `GOAL.md` completely and locate the machine-verifiable `gateway.json`
   it names. Read any canonical sources required for the requested change.
2. Run `dgoal validate <gateway>` and `dgoal resolve <gateway>`. Stop on any
   checksum, revision, precondition, or evidence failure; do not repair by
   guessing.
3. Translate the explicit owner decision into the smallest typed contract
   operations. Use an RFC 6901 pointer and include `--expect` for every changed
   existing value.
4. Append the amendment with `dgoal amend`. Preserve the owner's reason and use
   the contract's declared safe-boundary mode unless the owner explicitly
   specifies another mode.
5. Leave the amendment pending by default. Run `dgoal activate` only when the
   owner explicitly requests immediate activation or when separately verified
   evidence proves the declared activation boundary has occurred.
6. Run `dgoal materialize-status`, then `dgoal validate` and `dgoal resolve`
   again. Inspect the diff and report the new desired revision, active revision,
   pending activation, and validation result.

## Goal chains

When the owner says one goal should follow another:

1. Initialize a separate successor package if it does not exist.
2. Define its real invariants and completion predicate through amendments; the
   initialized `literal: false` predicate deliberately keeps a draft incomplete.
3. Chain the current gateway to the successor. This records an
   `after: completion` transition but does not execute the successor.
4. Activate the chain amendment only in revision order. Never skip an earlier
   pending amendment.
5. Treat a successor as runnable only when it appears in the active source
   status under `ready_transitions`.

```bash
dgoal init goals/next-goal \
  --goal-id next-goal \
  --objective 'Perform the next bounded objective.'

dgoal chain goals/current/gateway.json goals/next-goal/gateway.json \
  --transition-id then-next-goal \
  --reason 'Owner ordered next-goal after current completion'
```

## Commands

Record a scalar change:

```bash
dgoal amend path/to/gateway.json \
  --set /completion/all/0/gte '0.85' \
  --expect /completion/all/0/gte '0.90' \
  --reason 'Owner accepted the measured ceiling' \
  --activation-mode next_safe_boundary
```

JSON strings require JSON quoting, for example:

```bash
dgoal amend path/to/gateway.json \
  --set /objective '"Ship the validated candidate."' \
  --expect /objective '"Train a candidate."' \
  --reason 'Owner advanced the product objective'
```

Activate only the next pending revision:

```bash
dgoal activate path/to/gateway.json 12 --evidence-id boundary-receipt
dgoal materialize-status path/to/gateway.json
```

When `dgoal` is not installed, run it from this repository with
`PYTHONPATH=src python -m durable_goals.cli`.

## Guardrails

- Treat `GOAL.md` as the stable authoritative entry point and follow the source
  precedence it declares.
- Never hand-edit generated `STATUS.json`.
- Never rewrite an amendment or activation record. The writer creates immutable
  history and atomically advances the gateway.
- Do not infer an owner decision from ordinary progress, diagnostics, or an
  implementation detail.
- Do not activate a recorded change merely because it was recorded.
- Do not start, stop, restart, or reconfigure execution workloads while only
  recording a goal change.
- Do not begin a successor merely because it is recorded in desired intent;
  require its active transition to be ready.
- Preserve unrelated worktree changes and evidence files.
