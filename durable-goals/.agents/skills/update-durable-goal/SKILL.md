---
name: update-durable-goal
description: Safely create, update, activate, record evidence for, and connect durable-goals packages in a prompt-loop DAG when the owner changes an objective, invariant, completion condition, delegation, authority rule, activation state, or goal order. Use for prompts such as "update the goal," "record this decision," "after this goal do X," "change the authoritative goal," "activate revision N," "record this receipt," or "refresh workflow status." Do not use for ordinary implementation progress that supplies no owner goal change.
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

## Goal workflows

When the owner says one goal should follow another:

1. Initialize a separate successor package if it does not exist.
2. Define its real invariants and completion predicate through amendments; the
   initialized `literal: false` predicate deliberately keeps a draft incomplete.
3. Add both goal gateways to a workflow and add a dependency edge. Use multiple
   incoming edges for fan-in.
4. Run `dgoal workflow validate` and inspect `dgoal workflow status`.
5. Use `dgoal workflow next` to emit the next prompt. Do not assign or launch a
   model; the surrounding harness owns prompt execution.
6. With multiple agent threads, have each call `dgoal workflow claim` using its
   thread ID. This atomically selects different independent ready goals. Release
   the claim if a thread abandons the work.

```bash
dgoal init goals/next-goal \
  --goal-id next-goal \
  --objective 'Perform the next bounded objective.'

dgoal workflow init workflow.json --workflow-id goal-loop
dgoal workflow add-goal workflow.json goals/current/gateway.json \
  --node-id current
dgoal workflow add-goal workflow.json goals/next-goal/gateway.json \
  --node-id next-goal
dgoal workflow depend workflow.json current next-goal \
  --edge-id current-next
dgoal workflow validate workflow.json
dgoal workflow next workflow.json
dgoal workflow claim workflow.json --claimant thread-123
```

## Commands

Record a scalar change:

```bash
dgoal amend path/to/gateway.json \
  --set /completion/all/0/gte '0.85' \
  --expect /completion/all/0/gte '0.90' \
  --reason 'Owner accepted the measured ceiling' \
  --activation-mode next_safe_boundary \
  --when '{"evidence":"boundary","field":"/safe","equals":true}'
```

JSON strings require JSON quoting, for example:

```bash
dgoal amend path/to/gateway.json \
  --set /objective '"Ship the validated candidate."' \
  --expect /objective '"Train a candidate."' \
  --reason 'Owner advanced the product objective'
```

Record boundary evidence, then activate only the next pending revision:

```bash
dgoal evidence add path/to/gateway.json boundary boundary-receipt.json
dgoal activate path/to/gateway.json 12 --evidence-id boundary
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
- Do not begin a successor unless `dgoal workflow next` or workflow status marks
  it ready.
- Do not add agent, subagent, human, harness, or model assignment to the DAG.
  Durable Goals is a prompt loop, not a scheduler.
- Treat a claim as temporary duplicate-work prevention, not assignment. Do not
  take another thread's claim; release your own claim if abandoning the goal.
- Preserve unrelated worktree changes and evidence files.
