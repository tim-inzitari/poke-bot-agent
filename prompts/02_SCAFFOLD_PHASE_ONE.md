# Prompt 2 — Contracts and Shadow Instrumentation

```text
Read AGENTS.md, the completed repository mapping, docs/poke_rlm/02_TYPED_PLAN_IR.md, docs/poke_rlm/05_INFERENCE_AND_LATENCY.md, and Phase 1 of the implementation roadmap.

Implement only Phase 1 using the repository's existing patterns:
- separate typed deployment observations from privileged training targets;
- add structured legal-action metadata and canonical fingerprints;
- add TurnComputeBudget shared across the whole turn;
- add typed plan IR/runtime types and schema validation;
- add structured validation/fallback/repair reason codes;
- add PokeRLM feature flags defaulting to current behavior;
- add shadow trace instrumentation without changing selected actions or simulator-call behavior;
- add/update the durable PokeRLM progress state.

Requirements:
- CABT remains authoritative.
- Executed actions still use current legal option indices.
- No hidden fields enter deployment objects.
- No new simulator calls occur when PokeRLM is disabled or shadow-only.
- No arbitrary generated code or natural-language plans.
- Preserve all authoritative RL protocol numbers.

Add focused tests for schema round trips, hidden-field rejection, legal-index resolution, budget enforcement, deterministic fallback, and disabled-mode parity. Run the repository's formatter, linter/type checker, focused tests, existing regression tests, and a short shadow smoke test. Report files changed, commands, results, parameter delta (expected zero or negligible), latency delta, and rollback flag.
```
