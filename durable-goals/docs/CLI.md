# CLI reference

`dgoal` validates and mutates goal records and emits eligible prompts. It never
executes a prompt, chooses a model, or manages a workload.

## Goal packages

- `dgoal init DIR --goal-id ID --objective TEXT` creates a draft package whose
  completion is deliberately false.
- `dgoal validate GATEWAY` validates checksums, records, predicates, and
  revision ordering.
- `dgoal resolve GATEWAY` prints active and desired contracts plus evidence.
- `dgoal status GATEWAY` prints the derived non-authoritative projection.
- `dgoal materialize-status GATEWAY` atomically regenerates `STATUS.json`.
- `dgoal verify-evidence GATEWAY` verifies every evidence checksum.
- `dgoal evidence add GATEWAY ID FILE [--contract-revision N]` copies a JSON
  receipt into immutable package storage and advances the evidence index.

## Intent and activation

- `dgoal amend GATEWAY --set POINTER JSON ... --reason TEXT` appends intent.
- `--remove POINTER` removes a value.
- `--expect POINTER JSON` supplies an optimistic-concurrency precondition.
- `--activation-mode manual` is the default.
- `--activation-mode immediate` records an immediately activatable decision.
- `--activation-mode next_safe_boundary --when PREDICATE_JSON` requires the
  predicate to pass before activation.
- `dgoal activate GATEWAY REVISION [--evidence-id ID ...]` activates only the
  next pending revision and checksum-binds required or selected evidence.

`dgoal chain` remains a compact single-contract transition helper. Use a
workflow for a real multi-goal DAG.

## Workflow DAG and prompt loop

- `dgoal workflow init WORKFLOW --workflow-id ID`
- `dgoal workflow add-goal WORKFLOW GATEWAY --node-id ID`
- `dgoal workflow remove-goal WORKFLOW NODE`
- `dgoal workflow depend WORKFLOW FROM TO --edge-id ID`
- `dgoal workflow remove-dependency WORKFLOW EDGE`
- `dgoal workflow validate WORKFLOW`
- `dgoal workflow status WORKFLOW`
- `dgoal workflow next WORKFLOW [--all]`
- `dgoal workflow claim WORKFLOW --claimant THREAD_ID`
- `dgoal workflow release WORKFLOW NODE --claimant THREAD_ID`

Each workflow mutation advances its revision and preserves a content-addressed
history record. Validation resolves every node, checks goal identity, confines
references to the workflow package, and rejects unknown nodes, duplicate edges,
self-edges, and cycles. Status is deterministic:

- `completed`: the node's active completion predicate is satisfied;
- `ready`: it is incomplete and every predecessor is completed;
- `blocked`: at least one predecessor is incomplete.

`workflow next` walks stable topological order and emits the first ready
authoritative `GOAL.md` prompt. Until that goal records completion evidence,
the same prompt remains next. This is the entire looping contract.

For multiple agent threads, use `workflow claim` instead of `workflow next`.
Claims are selected under one short filesystem lock, so independent ready
branches go to different callers even when they ask concurrently. Claims are
not preconfigured assignments and contain no harness or model choice. Complete
the goal normally, or explicitly `release` it if the thread abandons the work.
