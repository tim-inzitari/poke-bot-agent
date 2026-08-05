# Data and Training Design

## 1. Relationship to the existing training protocol

PokeRLM adds architecture-specific targets and stages inside the established workflow. It does not override the current protocol's phase ordering, specialist isolation, game counts, rehearsal cadence, holdout isolation, or checkpoint registration requirements.

Preserve:

- strongest validated Alakazam-derived shared-core initialization;
- exactly 25 supervised bootstrap epochs for each new specialist;
- exactly 8,192 new games per baseline-phase RL iteration;
- 7,168 public-mix plus 1,024 self-play games;
- five supervised rehearsal epochs after each five completed RL epochs;
- exactly 3,000 isolated research evaluation games per required cycle;
- completed specialists frozen until all required specialists pass;
- population round-robin only after all specialists pass.

## 2. Supervision sources

### 2.1 Ordinary trajectories

Teach:

- acting-player state representation;
- chosen-action policy;
- realized value horizons;
- end-of-turn consequences;
- plan completion/failure traces;
- terminal results.

Limitation: they provide strong evidence mainly for actions actually selected.

### 2.2 Counterfactual root and plan-fragment rollouts

At selected consequential states:

1. clone CABT with a reproducible seed;
2. take 4–8 diverse candidate root actions or short plan fragments;
3. continue with one or more frozen rollout policies;
4. stop at end of turn, one opponent response, or terminal;
5. record return distribution, structured deltas, plan validity, and provenance.

These labels train action Q, dynamics, plan ranking, and uncertainty.

### 2.3 Strong teacher plans

Use expensive offline search, beam enumeration, heuristic solvers, specialist ensembles, or exhaustive enumeration where tractable to produce:

- complete conditional plans;
- alternative plan rankings;
- subgoal decomposition;
- critical branch predicates;
- stop/repair decisions;
- process-value labels for plan prefixes;
- failure traces.

Teacher computation may be much larger than the deployed budget, but the student's deployment inputs must remain legal acting-player observations.

### 2.4 Online correction data

Collect states where:

- the plan becomes invalid;
- predicted and observed consequences diverge;
- the direct policy beats the recursive route;
- recursion stops too early or overthinks;
- repair is frequent;
- uncertainty is miscalibrated;
- specialists disagree;
- a legal winning line was missed.

Relabel these states offline and return them to hard-state replay.

## 3. High-information state selection

Do not expand all ~3,000 protocol messages per game into equal training rows.

Prioritize:

- true branch points with materially different legal actions;
- high policy entropy;
- low top-two value margin;
- high bootstrap or specialist disagreement;
- long within-turn action chains;
- action-order sensitivity;
- stochastic search/draw branches;
- attack, switch, evolution, attachment, and once-per-turn commitments;
- choices that determine whether a knockout or setup objective is reachable;
- states where the current planner repairs or falls back.

Downsample:

- forced confirmations;
- deterministic target selections;
- near-duplicate states in one micro-action chain;
- protocol bookkeeping with no strategic branch.

## 4. Training record additions

Add fields to the existing record schema rather than creating an unversioned parallel format.

```text
plan_id, plan_schema_version, root_objective
candidate_plan_rank, plan_tokens, plan_node_count, recursion_depth
subgoal_codes, branch_predicates, fallback_coverage
plan_valid, validation_reason_codes
predicted_resource_delta, observed_resource_delta
predicted_end_turn_features, observed_end_turn_features
plan_prefix_value, plan_total_value, teacher_plan_value
router_label, direct_route_value, recursive_route_value
repair_used, repair_reason, fallback_used
model_calls_turn, simulator_calls_turn
component_latency_ns, deadline_margin_ns
```

Deployment observation fields and privileged target fields must be separate objects with separate visibility tests.

## 5. Losses

A practical joint objective is:

```text
L = λπ L_policy
  + λQ L_distributional_Q
  + λV L_state_value
  + λH L_multi_horizon
  + λS L_successor
  + λD L_dynamics
  + λP L_plan_imitation
  + λR L_router
  + λB L_branch_prediction
  + λU L_uncertainty_calibration
  + λI L_invalid_plan
  + λC L_compute_cost
  + λK L_consistency
```

Where:

- `L_plan_imitation` supervises plan tokens, subgoals, branch predicates, and stop decisions;
- `L_router` chooses direct/root/recursive paths;
- `L_invalid_plan` penalizes schema, selector, ledger, and visibility failures;
- `L_compute_cost` penalizes unnecessary depth, nodes, calls, and repairs;
- `L_consistency` aligns predicted deltas with observed CABT outcomes and direct/plan values where comparable.

Normalize losses and introduce them in stages. Do not allow a large auxiliary loss to erase a previously strong policy.

## 6. Curriculum

### Stage 0 — deterministic parity

- canonical observation and legal-action serializers;
- exact replay reconstruction;
- current policy parity;
- hidden-information and option-index tests;
- baseline latency and parameter count.

### Stage 1 — direct amortized heads

- parallel action decoder;
- policy, state value, chosen-action horizons;
- scalar then distributional Q;
- successor head;
- core frozen initially.

Exit: no per-action backbone calls, policy parity, stable targets, and positive teacher-ranking signal.

### Stage 2 — short latent dynamics

- one action or macro to end-of-turn delta;
- stochastic outcome distributions;
- uncertainty;
- symbolic-ledger consistency losses.

Exit: predicted deltas and values beat simple baselines on fixed audit sets.

### Stage 3 — typed root-plan imitation

- plan grammar tokens;
- root objectives;
- conditional branches;
- stop and fallback labels;
- no recursion yet.

Exit: high static plan-validity rate and positive plan-ranking signal.

### Stage 4 — recursive subgoal refinement

- depth 1, then depth 2;
- shared planner cell;
- batched same-depth subgoals;
- explicit compute penalty;
- simple-turn direct routing.

Exit: depth 2 improves difficult nonlinear states without unacceptable latency or simple-state regression.

### Stage 5 — on-policy execution and repair

- shadow execution first;
- plan/observation divergence labels;
- repair controller;
- hard-state relabeling;
- deterministic fallback.

Exit: low repair and fallback rates, zero legality regression, stable p95 whole-turn latency.

### Stage 6 — specialist adapters and distillation

- hot-start from shared core;
- train small adapters and plan-priority heads;
- preserve completed-specialist freeze rules;
- distill repeatable gains back to the shared core with broad rehearsal.

## 7. Offline simulator budget

Spend expensive simulator work where it creates reusable labels:

- architecture pilot: roughly 250k–1M difficult states with 4–8 alternatives;
- broader core: uncertainty- and disagreement-gated branch relabeling;
- specialist training: in-domain hard-state queues;
- on-policy correction: failed or repaired plans.

Version every rollout policy, seed, horizon, simulator build, rules version, card database, observation schema, and teacher type.

## 8. Leakage controls

Required tests:

- deployment observation serialization excludes opponent hand and prizes;
- future-revealed cards are absent at the current timestamp;
- privileged teacher channels cannot be imported by deployment modules;
- train/eval split is by game and includes deck/opponent/simulator version controls;
- duplicate canonical states do not cross protected splits where practical;
- rollout seeds are independent of candidate action tokens;
- plan predicates reference only visible fields.
