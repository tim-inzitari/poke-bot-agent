# Model Refresh Goal Gateway

Schema: `durable-goals.goal-gateway/v1`
Status: `authoritative`

Read this file completely before acting on this goal.

## Objective

> Promote a validated model without losing provenance.

## Canonical sources

- Verifiable reference gateway: `gateway.json`
- Typed contract: `contract.json`
- Owner amendments: `amendments.jsonl`
- Adopted amendments: `activations.jsonl`
- Evidence inventory: `evidence-index.json`
- Receipts: `receipts/`

The owner has recorded revision 2, lowering the desired accuracy threshold from
`0.90` to `0.85` at the next safe boundary. Revision 2 has not been activated,
so revision 1 remains active.

## Source precedence

1. Amendments determine desired intent.
2. The contract owns normalized base semantics.
3. Activations determine current adopted intent.
4. Checksum-verified receipts determine factual progress.
5. Generated status and conversation summaries are non-authoritative.

Run `dgoal resolve gateway.json` to verify checksums and derive the active and
desired views. Stop and report any validation failure rather than guessing.
