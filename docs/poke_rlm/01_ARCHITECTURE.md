# PokeRLM Architecture

## 1. System boundary

PokeRLM owns **decision planning**, not game truth.

```text
CABT / environment owns:
  legal options, exact rules, stochastic outcomes, state transitions,
  current option indices, terminal conditions

PokeRLM owns:
  state representation, action ranking, complexity routing,
  typed plan proposal, recursive subgoal refinement,
  learned consequence/value estimates, repair decisions
```

The boundary is enforced in interfaces and tests. A neural prediction can influence which action is attempted; it can never make an illegal action legal or replace the observed state.

## 2. Top-level modules

### 2.1 Acting-player observation builder

Builds a canonical deployment observation from only information legally visible to the acting player.

Required content:

- public board and zones;
- own hand and other own-private information;
- legally known deck/prize/search information;
- timestamped action and event history;
- deck-list or learned deck-summary context;
- opponent belief features derived only from legal observations;
- turn, phase, once-per-turn flags, and resource ledger inputs.

Required properties:

- deterministic canonicalization;
- schema versioning;
- observation hash;
- explicit visibility mask;
- no privileged fields in the deployment type.

### 2.2 Shared state encoder

Produces `state_memory: [B, S, D]` and a pooled state representation.

Responsibilities:

- transferable card and rule interactions;
- temporal context;
- resource and prize planning;
- current tactical state;
- opponent-belief context;
- deck/archetype conditioning without hard-wiring every behavior into a separate model.

The encoder should run once near the start of the turn. Stable key/value tensors may be cached across micro-decisions only when the observation-update contract proves the cache remains valid.

### 2.3 Structured legal-action encoder

Every current legal option is encoded as a structured token rather than only an opaque integer.

Suggested fields:

| Group | Examples |
|---|---|
| Category | play card, use ability, attach, evolve, retreat, attack, pass, choose target |
| Identity | card ID, effect ID, attack ID, ability ID, rules action |
| Source/destination | hand, active, bench slot, discard, deck, prizes |
| Arguments | target, count, mode, ordering, energy type |
| Control | ends turn, once-per-turn, forced, optional, continuation group |
| Exact resolution | current CABT option index and canonical action fingerprint |

The option index is valid only for the current legal list. Future plan nodes use selectors/fingerprints and must be resolved again at execution time.

### 2.4 Parallel action decoder

All legal actions attend to the shared state memory in one batched call.

Outputs per legal action:

- policy logit;
- distributional Q values;
- multi-horizon values;
- successor features/latent;
- uncertainty;
- optional plan-affordance logits.

No causal mask is required for comparing the current legal set. No full-backbone call is permitted per action.

### 2.5 Complexity router

Chooses one of three paths:

1. **Direct:** select from policy/Q immediately.
2. **Root plan only:** propose complete plans without recursive refinement.
3. **Recursive:** refine unresolved subgoals to bounded depth.

Router features should include:

- policy entropy;
- top-two action margin;
- Q-head disagreement;
- legal-action count and action-group structure;
- estimated remaining action-chain length;
- order-sensitivity score;
- stochastic branch count;
- deck/archetype complexity prior;
- time and compute remaining.

The router receives a compute penalty during training. Simple turns should remain simple.

### 2.6 Root plan proposer

Proposes 4–8 typed candidate plans in parallel. Each plan includes:

- an objective code;
- primitive actions or action selectors;
- unresolved subgoals;
- observation-conditioned branches;
- fallback nodes;
- predicted resource deltas;
- expected end-of-turn features;
- confidence and value distributions;
- an explicit compute budget.

The root proposer does not emit prose. It emits tokens that compile into the plan IR defined in `02_TYPED_PLAN_IR.md`.

### 2.7 Shared recursive subgoal refiner

The same planner cell receives each unresolved subgoal:

```text
planner_cell(
    state_memory,
    current_latent,
    resource_ledger,
    subgoal_token,
    legal_affordances,
    remaining_budget,
) -> refinement
```

A refinement is one of:

- primitive action selector;
- short sequence;
- conditional branch;
- smaller subgoal set;
- fallback;
- stop/fail.

All subgoals at a given depth are batched. The cell is reused across depth; recursion adds sequential compute but not a new parameter set.

### 2.8 Latent dynamics and consequence model

Predicts decision-relevant short-horizon consequences:

```text
z_next, delta, outcome_distribution = dynamics(z, action_or_macro)
```

Suggested structured outputs:

- hand/resource count changes;
- board occupancy and damage changes;
- ability/attachment/retreat flags;
- attack availability;
- expected search/draw outcome classes;
- end-of-turn board summary;
- immediate, end-turn, one-round, two-round, and terminal values;
- epistemic uncertainty and stochastic spread.

Keep latent rollouts short. The main target is the end of the current turn or, selectively, one opponent response. Longer learned rollouts increase model exploitation and compounding error.

### 2.9 Exact validator and symbolic resource ledger

The validator is deterministic code, not a learned head.

It checks:

- node and depth budgets;
- selector resolvability;
- current legality;
- source/target existence;
- once-per-turn usage;
- attachment/evolution/retreat constraints represented by available metadata;
- resource ledger consistency;
- branch predicate visibility;
- plan termination and fallback coverage;
- current deadline.

Invalid plans receive structured reason codes and are removed or repaired. Never “best effort” an invalid option index.

### 2.10 Plan executor

At each primitive step:

1. request the current CABT legal options;
2. encode/fingerprint them;
3. resolve the plan selector against the current list;
4. validate the selected current index;
5. submit the index;
6. observe the true transition;
7. update the resource ledger and plan cursor;
8. follow the matching conditional branch;
9. repair or fall back when required.

### 2.11 Repair controller

Repair is triggered only when:

- the expected selector is not resolvable;
- a stochastic result enters an uncovered branch;
- observed state diverges materially from predicted consequences;
- a plan's value falls below the direct-policy alternative;
- CABT rejects or changes the action group;
- the remaining plan cannot complete within budget.

Repair reuses cached state memory when valid, otherwise performs one bounded re-encode. Maximum repair calls are explicit in configuration.

## 3. Turn-level execution pseudocode

```python
obs = observation_builder.from_cabt(cabt, acting_player)
legal = cabt.legal_options()
state_memory = encoder(obs)                    # once per turn when safe
action_features = action_decoder(state_memory, encode(legal))

route = complexity_router(state_memory, action_features, budget)
if route == DIRECT:
    plan = compile_single_action_plan(select_direct(action_features))
else:
    candidates = root_planner.propose(state_memory, action_features, budget)
    candidates = recursive_refiner.refine_batched(candidates, max_depth=2)
    candidates = latent_evaluator.score(candidates)
    plan = validator.best_valid(candidates)

while not plan.done and not deadline.expired:
    legal = cabt.legal_options()
    resolution = validator.resolve_current(plan.cursor, legal)
    if not resolution.ok:
        plan = repair_or_fallback(plan, obs, legal, budget)
        continue

    cabt.step(resolution.option_index)
    obs = observation_builder.from_cabt(cabt, acting_player)
    plan.advance(obs)

return deterministic_fallback_if_needed()
```

## 4. Recommended base tensor/config shape

```text
D = 384
state encoder layers = 10
attention heads = 8
FFN width = 4D
state tokens = profile after repository audit; target compact 192–384
legal actions = packed or bucketed; preserve all actual legal options
parallel action decoder layers = 2
root plans = 4–8
plan tokens per candidate = up to 24
recursive planner layers = 3, shared across depth
maximum recursion depth = 2
maximum plan nodes = 32
latent dynamics blocks = 4
Q bootstrap heads = 4
Q quantiles = 32
value horizons = 5
```

These are starting values, not a substitute for measuring the current encoder and deployment hardware.

## 5. Architectural invariants to assert in code

- `backbone_calls_per_turn <= configured_max`
- `planner_depth <= max_depth`
- `plan_node_count <= max_plan_nodes`
- `simulator_calls_per_turn <= hard_cap`
- `executed_option_index in current_legal_option_indices`
- `deployment_observation.has_privileged_fields is False`
- `recursive_parameter_ids(depth_i) == recursive_parameter_ids(depth_j)`
- `fallback_is_deterministic is True`
- `all_logged_records.have_version_and_seed_provenance is True`
