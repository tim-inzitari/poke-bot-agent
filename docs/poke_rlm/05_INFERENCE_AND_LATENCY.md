# Inference and Latency Design

## 1. Reframe the budget

The observed legacy capacity—about 75 simulator calls for the whole turn—uses essentially the entire available turn time. It is therefore a **hard empirical ceiling**, not a budget that PokeRLM should routinely consume.

Initial deployment target:

```text
normal simulator calls per turn: 0–16
target p95 simulator calls:      <=16
hard emergency cap:              75
neural planner calls per turn:   <=4
recursion depth:                 <=2
```

The remaining time must cover observation building, tensor transfer, state encoding, plan generation, validation, CABT communication, repair, logging, and safety margin.

## 2. Default turn path

```text
1 state encode
1 parallel legal-action decode
0 or 1 root-plan proposal
0–2 recursive refinement rounds
0 or 1 repair call
0–16 optional simulator verification calls
```

All root candidates are batched. All subgoals at one depth are batched. Heads are fused where practical.

## 3. Simulator allocation

Recommended starting allocation:

| Use | Calls | Rule |
|---|---:|---|
| Routine direct/simple turn | 0 | No simulator verification |
| Finalist verification | 0–8 | Only high uncertainty or small plan-value margin |
| Repair reserve | 0–4 | Unexpected observation or selector failure |
| Emergency reserve | 0–4 | Only while deadline margin remains |
| Normal target cap | 16 | Stop and use best valid plan/fallback |
| Hard measured cap | 75 | Never plan around consuming this |

A verifier call should evaluate a finalist plan fragment or high-uncertainty chance branch, not expand a broad raw tree.

## 4. Deadline-aware budget object

Every turn receives one shared budget object:

```python
class TurnComputeBudget:
    deadline_ns: int
    max_model_calls: int
    max_simulator_calls: int
    max_plan_nodes: int
    max_depth: int
    safety_margin_ns: int
```

All components debit the same object. No submodule may maintain an invisible local budget.

Before every optional operation:

```python
if budget.time_remaining_ns <= budget.safety_margin_ns:
    return best_current_valid_or_fallback()
```

## 5. Cache policy

Safe cache candidates:

- card/effect embedding tables;
- static deck-list/deck-summary tokens;
- stable history-prefix keys/values;
- current turn state memory when only a continuation choice changes no encoded fact;
- action fingerprints and structured metadata for unchanged legal groups.

Invalidate on:

- card movement, draw, discard, search reveal, shuffle, prize change;
- board, damage, energy, status, active/bench changes;
- once-per-turn resource changes;
- hidden-belief update from a new observation;
- phase/turn transition;
- any schema-version uncertainty.

Prefer a conservative invalidation policy first. Optimize only after parity tests.

## 6. Direct route

The direct route must remain first-class, not a degraded fallback.

Use it when:

- one action dominates policy and Q margins;
- legal-action entropy is low;
- the turn is forced or nearly forced;
- estimated action-chain length is short;
- latent and bootstrap uncertainty are low;
- the deadline is tight;
- the planner is unhealthy.

This protects simple linear decks from unnecessary recursion.

## 7. Plan persistence

The plan persists across micro-decisions. After each CABT step:

- compare the observed result to the plan predicate;
- update the resource ledger;
- follow the precomputed branch;
- avoid re-running the root planner when the branch remains valid;
- run repair only when a trigger is met.

This is the primary latency advantage over search-per-decision.

## 8. Telemetry

Log per turn:

```text
observation_tokens, legal_action_count
route: direct | root | recursive
candidate_plan_count, selected_plan_rank
max_depth_used, plan_nodes_generated, plan_nodes_executed
model_calls, simulator_calls
encode_ms, action_decode_ms, plan_ms, refine_ms
dynamics_ms, validate_ms, cabt_ms, repair_ms, total_turn_ms
p50/p95/max aggregation keys
repair_reason, fallback_reason, validation_failures
predicted_vs_observed_delta_error
model_version, adapter_version, schema_version
```

Do not log hidden information in deployment traces.

## 9. Failure behavior

On invalid tensor, NaN, timeout, out-of-memory, unresolved selector, validator exception, or planner crash:

1. record a structured error code;
2. discard the unsafe plan;
3. resolve the deterministic existing-policy action against the current legal list;
4. submit only a validated current option index;
5. preserve the trace for offline diagnosis.

No exception path should submit a guessed index.

## 10. Performance experiments

Benchmark representative buckets:

- simple turns with few actions;
- maximum legal-action sets;
- long combo turns;
- frequent stochastic search/draw turns;
- plan depth 0/1/2/3;
- 0/8/16 simulator verification calls;
- cache hit/miss;
- repair/no repair;
- each deployment precision or quantization mode.

Report p50, p95, and maximum whole-turn latency. Average latency alone is insufficient.
