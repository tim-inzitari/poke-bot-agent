# Implementation Roadmap

## Principle

Do not perform a blind rewrite. Build PokeRLM as a sequence of independently testable layers behind flags. Each phase must preserve the current agent as a baseline and fallback.

## Phase 0 — Repository audit and baseline lock

Deliverables:

- completed `09_REPOSITORY_MAPPING_TEMPLATE.md` with exact paths and symbols;
- current model parameter count and checkpoint size;
- current observation/action tensor shapes;
- CABT legal-option and step contract;
- MCTS/model/simulator call flow;
- p50/p95/max decision and turn latency;
- simulator calls per turn and per decision;
- current test/evaluation commands;
- deterministic baseline artifact IDs.

Pass condition:

- exact replay works;
- current tests pass;
- no production code behavior changed;
- baseline evidence is recorded.

## Phase 1 — Contracts, schemas, and shadow instrumentation

Implement:

- typed `DeploymentObservation` and `PrivilegedTargets` separation;
- structured `LegalAction` and canonical fingerprint;
- versioned plan IR types and parser;
- `TurnComputeBudget`;
- structured trace and reason codes;
- feature flags with current behavior as default;
- shadow-only planner hooks.

Tests:

- schema round trip;
- hidden-field rejection;
- current-index resolution;
- budget enforcement;
- deterministic fallback;
- no extra simulator calls in disabled mode.

Pass condition:

- instrumentation runs on real games without selecting actions;
- zero parity regression.

## Phase 2 — Parallel action decoder and amortized heads

Implement:

- structured action embeddings;
- batched cross-attention over all current legal actions;
- policy, Q, horizon, successor, state-value, and uncertainty outputs;
- fused head projection for deployment;
- checkpoint compatibility/migration.

Tests:

- all legal actions scored in one state encode;
- padding/masks correct;
- chosen-action parity baseline;
- no Python loop over legal actions in deployed forward;
- fixed-shape and max-action latency tests.

Pass condition:

- policy-only parity or improvement;
- no per-action backbone calls;
- stable head training and audit ranking.

## Phase 3 — Short latent dynamics

Implement:

- action/macro-conditioned transition cell;
- structured delta heads;
- stochastic outcome distribution;
- symbolic-ledger consistency checks;
- one-step/end-turn value prediction.

Tests:

- predicted shape and mask contracts;
- observed CABT delta reconstruction;
- stochastic calibration fixtures;
- impossible resource delta rejection.

Pass condition:

- beats simple no-change and empirical-mean baselines;
- useful teacher ranking without policy regression.

## Phase 4 — Typed root planner

Implement:

- finite objective and plan-token vocabularies;
- 4–8 batched candidate plans;
- plan compiler and static validator;
- plan scoring using direct heads and latent dynamics;
- no recursive refinement yet.

Tests:

- grammar coverage;
- invalid-token rejection;
- plan-node and deadline caps;
- selector resolution under reordered legal lists;
- branch predicate visibility.

Pass condition:

- high plan parse/static-validity rate;
- root-plan ranking improves nonlinear audit states.

## Phase 5 — Bounded recursive refiner

Implement:

- shared planner cell;
- batched same-depth subgoals;
- depth 1 and depth 2 modes;
- compute-cost and stop heads;
- fallback completion for every unresolved subgoal.

Tests:

- parameter object identity across depth;
- exact depth/node/model-call enforcement;
- termination for adversarial plan inputs;
- deterministic output under fixed seed/eval mode;
- recursion disabled on simple-turn fixtures.

Pass condition:

- depth 2 improves difficult states over root-only and depth 1;
- no unacceptable p95 latency or simple-deck regression.

## Phase 6 — Stateful execution and bounded repair

Implement:

- plan cursor persisted across CABT micro-decisions;
- current legal-list selector resolution;
- resource ledger updates;
- conditional branch evaluation;
- observed/predicted divergence trigger;
- bounded repair and deterministic fallback.

Rollout order:

1. shadow plan generation;
2. shadow plan execution against recorded trajectories;
3. live action selection in evaluation only;
4. controlled feature-flag deployment.

Pass condition:

- zero illegal actions;
- low unresolved-selector and repair rates;
- bounded whole-turn latency;
- exact trace replay.

## Phase 7 — Teacher generation and recursive training

Implement:

- hard-state selector;
- offline counterfactual plan teacher;
- process-value and branch labels;
- staged multi-loss trainer;
- relabel queue;
- calibration and plan-regret audits.

Pass condition:

- reproducible labels;
- no train/holdout contamination;
- stable optimization;
- positive paired strength at deployment latency.

## Phase 8 — Specialist adapters and core distillation

Implement:

- small archetype adapters and plan-priority calibration;
- exact compatibility with shared checkpoints;
- adapter registry and checksums;
- specialist disagreement mining;
- broad replay distillation back into shared core.

Preserve all existing specialist freeze and promotion rules.

Pass condition:

- specialists improve in-domain with bounded memory;
- shared core retains held-out strength after distillation;
- all artifacts are traceable in `state/specialists.yaml` and the PokeRLM status file.

## Pull-request/change-set guidance

Prefer one reviewable change set per phase or subphase. Avoid mixing:

- observation schema changes;
- model architecture changes;
- training objective changes;
- inference behavior changes;
- large data migrations;
- unrelated cleanup.

Every change set should state:

- before/after behavior;
- files and interfaces changed;
- parameter delta;
- latency delta;
- tests run;
- known limitations;
- rollback flag.
