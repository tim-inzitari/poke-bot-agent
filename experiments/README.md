# Experimentation sandbox

Branch: `cursor/experimentation-026a`  
Forked from: `codex/worksession-20260728` @ `78623d5`

This tree is for **isolated R&D only**. It is not a production training lane.

## Guardrails

- Do not rewrite `GOAL.md` or treat this branch as owner-contract authority.
- Do not restart, preempt, or reconcile healthy Blackwell training
  (`pokebot-final-format-alakazam-r79-h10.service`) for experiment metadata.
- Control managed workloads only through declared service managers. Never use
  process-tree termination against interactive sessions.
- Keep experiment artifacts under `experiments/` (or explicitly labeled
  experiment paths). Do not silently mutate production registries, immutable
  receipts, or sealed specialist corpora.
- Promote nothing to production without a checksum-bound receipt and an
  explicit owner activation boundary.

## Existing lanes

- `experiments/cuda-sim-lab/` — CUDA simulator feasibility / parity slices
- `experiments/apple-optimization/` — Apple GPU telemetry / optimization plists
- `experiments/recursive_turn_planner/` — lightweight Recursive Turn Planner
  (importable under `poke_bot/recursive_turn_planner/`)
- `poke_bot/poke_rlm/` — full PokeRLM kit (phases 1–7) behind
  `POKEBOT_POKE_RLM_*` flags; default disabled. Progress:
  `state/poke_rlm_redesign.yaml`. Docs: `docs/poke_rlm/`.

Add new experiment folders beside these. Prefer small, receipt-backed probes
over broad production-path edits.
