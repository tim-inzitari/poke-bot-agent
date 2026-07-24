# Specialist transition task graph

This staged controller wraps the existing checksum-validated handoff without
changing model code, specialist selection, or training behavior. It is a small
local dependency graph rather than Airflow: the transition runs on one host,
already uses systemd for lifecycle control, and does not need a database,
scheduler cluster, webserver, or another credential boundary.

`ops/specialist_transition_graph.json` is the only source of task order. Graph
nodes may select only built-in Python actions. The graph cannot contain shell
commands or arbitrary service names. Its service allowlist resolves exact
service names from the existing cycle contract.

The journal is persistent and separated by active-specialist identity plus
cycle-contract checksum. Each completed node has a checksum-bound receipt. A
retry skips valid completed nodes and resumes at the failed or incomplete node.
A changed cycle contract creates a new transition identity. A changed graph
fails closed and requires an explicit state migration instead of silently
reinterpreting old receipts.

The boundary node refuses to continue while the managed specialist training
service is active. The execution node delegates to the existing idempotent
cycle handoff, which retains its immutable checkpoint checks, inner phase
receipts, selector update, and exact service controls. Kaggle submission
completion remains asynchronous and never blocks the next specialist.

The staged service definition is
`deploy/staging/pokebot-specialist-cycle-handoff-task-graph.service`. It is not
installed or enabled.

Safe inspection commands after an eventual deployment:

```text
python scripts/run_specialist_transition_graph.py \
  --graph ops/specialist_transition_graph.json \
  --cycle-contract ops/specialist_cycle_handoff_v1.json \
  --state /home/inzi/poke-bot-agent/outputs/state/specialist-transition-graph.json \
  --status
```

Use `--dry-run` instead of `--status` to display the same execution plan with
an explicit dry-run marker. Neither mode invokes node actions.
