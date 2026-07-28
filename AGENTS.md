# Pokemon RL controller contract

## Interactive-session safety

- The LAN identity `192.168.1.160` / `fe80::4fd:7b09:c7d6:916d%en0`
  (link-layer address `6a:55:8a:be:7b:c4`) is user-owned and hard-allowed.
- Never terminate, signal, disconnect, replace, rate-limit, quarantine, or
  classify any SSH, Codex, terminal, Cursor, Grok, Claude, editor, or other
  interactive user session as stale.
- Treat concurrent interactive sessions as user-authorized unless the user
  identifies the exact session to terminate in the current turn.
- Never use process-tree termination against an interactive session. Control
  managed training and dashboard workloads only through their declared service
  manager.

## Goal resumption

- Before acting on a resumed long-running goal, read `GOAL.md` completely,
  followed by every canonical source it identifies as required for the current
  action.
- `GOAL.md` supersedes stale product goal prose. The
  `ops/current_goal_requirements.json` file is a compatibility projection for
  older controllers and must not override `GOAL.md`.
- When the owner explicitly defines a design change, record it in `GOAL.md`,
  update the one typed canonical source that owns it, and activate it at the
  next safe receipt-backed boundary unless immediate activation is explicitly
  ordered.
- Runtime truth comes from the one canonical selector, immutable receipts, and
  managed-service state. Planning prose and dashboard labels never override it.
- Never restart or preempt healthy active training merely to reconcile planning
  metadata.
