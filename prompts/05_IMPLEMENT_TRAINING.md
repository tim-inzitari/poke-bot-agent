# Prompt 5 — Training and Teacher Pipeline

```text
Read AGENTS.md, RL_TRAINING_PROTOCOL.md, config/rl_protocol.yaml, state/specialists.yaml, docs/poke_rlm/04_DATA_AND_TRAINING.md, and Phase 7 of the roadmap.

Implement the PokeRLM training/data additions without changing authoritative schedule numbers:
- versioned plan trace records;
- hard-state selection by entropy, margin, disagreement, order sensitivity, long chains, repairs, and failures;
- reproducible CABT counterfactual plan-fragment generation with seed/policy/simulator/rules/card-db/schema provenance;
- process labels for objectives, plan tokens, subgoals, branch predicates, stop, repair, and plan-prefix values;
- short dynamics/successor targets;
- staged losses with normalization and configurable weights;
- compute-cost and invalid-plan penalties;
- train/holdout leakage guards;
- relabel queue for failed/on-policy hard states;
- fixed teacher audit set and calibration metrics.

Preserve exactly 25 supervised bootstrap epochs, 8,192 games per baseline RL iteration with the established 7,168/1,024 split, the five-RL/five-rehearsal cadence, 3,000-game isolated evaluations, specialist freezing, and research-holdout exclusion.

Add deterministic label-generation tests, provenance tests, leakage tests, loss-shape tests, resume/restart counter tests, and a tiny end-to-end overfit fixture. Do not launch expensive training. Produce exact commands for a small smoke run and the later 5M–10M architecture pilot, including expected storage and checkpoints.
```
