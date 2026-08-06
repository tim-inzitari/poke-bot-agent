# PokeRLM (experimental)

Typed, bounded, latent recursive turn planner attachment for the Pokemon TCG
agent. Implements kit phases 1–7 behind feature flags. Phase 8 specialist
adapters are scaffolded only.

## Defaults (behavior-preserving)

| Flag | Default | Effect |
|---|---|---|
| `POKEBOT_POKE_RLM_ENABLED` | unset / false | Planner off |
| `POKEBOT_POKE_RLM_MODE` | `disabled` | `shadow` / `evaluate` / `active` |
| `POKEBOT_POKE_RLM_PROFILE` | `pure_rl_96` | Width binding |

- **disabled**: zero planner side effects; PolicyAgent uses MCTS / RTP / greedy as before.
- **shadow**: planner runs for traces; selected action stays greedy/RTP/MCTS.
- **evaluate / active**: planner may select actions (illegal → greedy fallback).

## Package map

- `config.py` — flags, profiles, env loader
- `observation.py` / `legal_action.py` — deployment contracts
- `plan_ir.py` + `schemas/turn_plan.schema.json` — typed IR
- `model_core.py` — parallel decoder, dynamics, root proposer, shared recursive cell
- `router.py` / `recursion.py` / `executor.py` / `controller.py` — inference stack
- `agent_hooks.py` — PolicyAgent bridge
- `training/` — traces, labels, losses, hard-state curriculum
- `specialists.py` — phase-8 registry scaffold

## Tests / bench

```bash
pytest -m unit tests/test_poke_rlm.py
python scripts/bench_poke_rlm_turn_latency.py --profile pure_rl_96
```

Progress: `state/poke_rlm_redesign.yaml`.
