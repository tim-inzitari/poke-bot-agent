# Goal Gateway

Schema: `durable-goals.goal-gateway/v1`
Status: `authoritative`

This file is the stable entry point for every new or resumed agent. Read it
completely before acting, followed by the canonical sources required for the
current action.

## Objective

> Replace this sentence with the short, durable product objective.

## Canonical sources

- Machine-verifiable gateway: `gateway.json`
- Typed goal contract: `contract.json`
- Owner amendment ledger: `amendments.jsonl`
- Activation ledger: `activations.jsonl`
- Evidence index: `evidence-index.json`
- Generated status projection: `STATUS.json`
- Optional multi-goal prompt DAG: repository `workflow.json`

## Source precedence

When sources disagree, use this order:

1. The latest valid owner amendment for desired intent.
2. The typed contract for normalized goal semantics.
3. Valid activation records for what has entered effect.
4. Checksum-verified evidence for execution facts.
5. Generated status, dashboards, plans, and conversation summaries.

Never let status prose or a conversation summary override a canonical source.
Report an unresolved contradiction instead of guessing.

## Change procedure

1. Record an owner decision as the next append-only amendment.
2. Validate its preconditions and the updated desired contract.
3. Keep it pending until its declared activation condition is met.
4. Append an activation record; never rewrite the amendment.
5. Regenerate status from the resolved contract and evidence.

When this goal is a workflow node, only treat it as eligible when
`dgoal workflow next workflow.json` emits it. The workflow never assigns or
launches a model; the surrounding harness owns prompt execution.
