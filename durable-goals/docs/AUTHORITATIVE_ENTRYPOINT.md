# The authoritative entry-point convention

Every coding harness receives the same project instruction:

> Before acting on a new or resumed goal, read `GOAL.md` completely, followed
> by every canonical source it identifies as required for the current action.

That is the entire cross-harness integration. It works in repository instruction
files, system prompts, onboarding documentation, or a human's first message.
There is no requirement for an editor extension, agent plugin, MCP server, or
vendor-specific state format.

## Why both Markdown and typed records exist

`GOAL.md` is optimized for agents and humans. It explains the objective,
authority boundaries, source precedence, and how to resume. It should remain
short enough to read completely.

Typed records are optimized for deterministic verification:

- `gateway.json` checksum-binds the selected records.
- `contract.json` owns normalized goal values and completion predicates.
- `amendments.jsonl` records owner decisions without rewriting history.
- `activations.jsonl` records which amendment prefix has entered effect.
- `evidence-index.json` binds factual evidence.
- `STATUS.json`, when present, is only a regenerable projection.

For more than one goal, `workflow.json` owns the DAG. It names goal gateways and
dependency edges; the resolver rejects cycles and reports a node ready only
after every predecessor's active completion predicate is satisfied. The
workflow emits prompts but never assigns or launches a model.

Contracts may also declare a small local `after: completion` transition for
compatibility. Use the workflow DAG for fan-in, graph validation, and prompt
looping.

An agent can operate by reading these files. The `dgoal` command is a verifier,
not a prerequisite for understanding the goal.

## Suggested harness instruction files

Whatever instruction mechanism a harness already supports should contain the
same small rule:

```text
## Goal resumption

Before acting on a new or resumed long-running goal, read `GOAL.md` completely,
then read the canonical sources it identifies for the current action. Treat
generated status and conversation summaries as non-authoritative when they
disagree with those sources.
```

The wording should not duplicate the current objective. Duplicating it creates
another stale source of truth.

## Moving between harnesses

Commit or otherwise transfer the goal package with the repository. The next
harness starts by reading `GOAL.md`; it does not need the previous harness's
conversation history. Conversation summaries remain useful context, but the
goal package carries durable authority.
