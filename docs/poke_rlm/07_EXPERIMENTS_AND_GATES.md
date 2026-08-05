# Experiments, Metrics, and Gates

## 1. Core hypothesis

PokeRLM should provide its largest gain on **nonlinear, order-sensitive decks and turns**, while the complexity router preserves the speed and strength of simple turns.

## 2. Required variants

| ID | Variant | Purpose |
|---|---|---|
| A | Current policy-only agent | Strength and latency baseline |
| B | Parallel action decoder + policy | Legal-set comparison benefit |
| C | B + Q/value/successor/uncertainty | Amortized consequence modeling |
| D | Typed root plan, no recursion | Complete-turn planning benefit |
| E | D + recursion depth 1 | One refinement round |
| F | D + recursion depth 2 | Recommended PokeRLM |
| G | D + recursion depth 3 | Overthinking/latency audit only |
| H | F + up to 8 verifier calls | Sparse hybrid benefit |
| I | F + up to 16 verifier calls | Higher verifier budget |
| J | Legacy 75-call MCTS | Existing search baseline |

Use equal deployment deadlines and account for all model and simulator calls.

## 3. Deck/turn complexity stratification

Create a complexity score from observable or trace-derived features:

```text
complexity =
    w1 * meaningful_legal_action_count
  + w2 * expected_remaining_micro_actions
  + w3 * action_order_sensitivity
  + w4 * stochastic_branch_count
  + w5 * policy_entropy
  + w6 * Q_disagreement
  + w7 * resource_interaction_count
  + w8 * objective_ambiguity
```

Report results by score quantile and by archetype. Do not rely only on a manually assigned “complex deck” label.

Useful empirical measures:

- average and p95 strategic decision-chain length;
- fraction of action pairs whose order changes end-turn value;
- number of distinct valid end-turn plans;
- top-plan value margin;
- repair and fallback rates;
- observed branch entropy;
- teacher regret of flat versus recursive decisions.

## 4. Primary metrics

### Playing strength

- paired win rate and Elo against fixed snapshots;
- seat alternation and matched seed schedules where possible;
- strength by archetype, matchup, game phase, and complexity bucket;
- held-out deck lists and opponent policies.

### Operational performance

- p50, p95, and maximum decision latency;
- p50, p95, and maximum whole-turn latency;
- model calls and simulator calls per turn;
- peak CPU/GPU memory;
- legal-action count and plan depth at latency tail events.

### Correctness

- illegal-action rate;
- unresolved-selector rate;
- invalid-plan rate by reason;
- hidden-information leakage test failures;
- deterministic replay mismatch;
- NaN, timeout, OOM, repair, and fallback rates.

### Model quality

- policy top-k accuracy where meaningful;
- action/plan ranking regret against teacher;
- state and Q calibration;
- successor/delta error;
- branch-prediction accuracy;
- plan-prefix value accuracy;
- uncertainty versus realized error;
- router regret: value of chosen route versus best available route.

## 5. Architecture gates

These are architecture-development gates and do not replace established specialist competition gates.

### Gate 0 — Safety/parity

- zero hidden-field exposure;
- zero illegal executed actions;
- exact current-option-index resolution;
- deterministic fallback works;
- disabled mode reproduces current behavior.

### Gate 1 — Direct-head viability

- no per-action backbone calls;
- stable losses and calibration;
- policy parity or improvement;
- p95 latency retains margin.

### Gate 2 — Plan validity

- schema/static validity high enough that invalid plans are exceptional and explainable;
- every subgoal has a fallback;
- all generated plans terminate within hard budgets.

### Gate 3 — Recursive value

- depth 2 beats root-only and depth 1 on high-complexity states;
- depth 2 does not materially harm low-complexity states because the router avoids it;
- depth 3 is rejected unless it adds strength with acceptable tail latency.

### Gate 4 — Deployment value

- positive paired win-rate/Elo result at equal whole-turn deadline;
- p95 and max latency safe;
- no material held-out-deck regression;
- optional verifier calls provide measured marginal value.

### Gate 5 — Scale decision

Move from `base_384` to `strong_512` only if controlled scaling shows capacity-limited improvement after data, labels, optimization, and interfaces are validated.

## 6. Suggested experiment sequence

1. A vs B vs C on all decks.
2. C vs D on nonlinear audit states.
3. D vs E vs F vs G with exact compute accounting.
4. F router on/off to quantify simple-turn protection.
5. F vs H vs I to price simulator verification.
6. F/I vs J under the same whole-turn deadline.
7. `pilot_256` vs `base_384`; scale to `strong_512` only if justified.
8. Shared core vs specialist adapters, followed by distillation audit.

## 7. Statistical reporting

Always provide:

- game count;
- paired/unpaired design;
- confidence interval or standard error;
- exact opponent/checkpoint versions;
- seat and seed balancing method;
- exclusions and failure counts;
- deployment configuration and hardware;
- parameter count and latency distribution.

Do not promote based on a single noisy score or aggregate result that hides archetype regressions.
