# Typed Turn-Plan Intermediate Representation

## Purpose

The plan IR converts model outputs into a deterministic, inspectable execution program. It prevents a recursive planner from becoming an open-ended text or code generator.

The JSON interchange form is defined in `schemas/turn_plan.schema.json`. The runtime should use typed dataclasses or equivalent repository-native types.

## Core types

```python
@dataclass(frozen=True)
class TurnPlan:
    plan_id: str
    schema_version: str
    objective: ObjectiveCode
    root: PlanNode
    expected_end_turn: ExpectedOutcome
    value: DistributionalValue
    uncertainty: float
    budget: PlanBudget
    provenance: PlanProvenance

PlanNode = PrimitiveAction | SequenceNode | ConditionalNode | SubgoalNode | StopNode
```

### PrimitiveAction

A primitive node does **not** trust a future raw option index.

```python
@dataclass(frozen=True)
class PrimitiveAction:
    selector: ActionSelector
    planned_option_index: int | None
    expected_group: str
    expected_delta: ResourceDelta
    on_unresolvable: PlanNode
```

`planned_option_index` is useful for the immediate current state and audits. Before execution, the selector must be resolved against the current CABT legal list and return a current option index.

### ActionSelector

Recommended selector order:

1. exact canonical action fingerprint;
2. typed identity + source + destination + arguments;
3. constrained candidate ranking when several legal options are strategically equivalent;
4. fail and invoke fallback.

Selectors must never use hidden information.

### SequenceNode

```python
@dataclass(frozen=True)
class SequenceNode:
    children: tuple[PlanNode, ...]
```

A sequence is interrupted when a child cannot resolve, a branch predicate fires, the deadline is reached, or the plan value falls below a configured repair threshold.

### ConditionalNode

```python
@dataclass(frozen=True)
class ConditionalNode:
    predicate: ObservationPredicate
    if_true: PlanNode
    if_false: PlanNode
```

Predicates are evaluated only against the current acting-player observation. Supported predicate families should remain finite and versioned:

- card/effect observed;
- result class from search or draw;
- legal action fingerprint present;
- board slot occupied/empty;
- resource threshold met;
- attack available;
- target damaged/knocked out;
- once-per-turn resource consumed;
- plan deadline or node budget reached.

### SubgoalNode

```python
@dataclass(frozen=True)
class SubgoalNode:
    goal_code: GoalCode
    constraints: tuple[GoalConstraint, ...]
    local_budget: PlanBudget
    success_predicate: ObservationPredicate
    fallback: PlanNode
```

Suggested initial goal vocabulary:

- `ESTABLISH_ATTACKER`
- `FIND_RESOURCE_CLASS`
- `ENABLE_EVOLUTION`
- `REACH_DAMAGE_THRESHOLD`
- `TAKE_KNOCKOUT`
- `DEVELOP_BOARD`
- `PRESERVE_ESCAPE_LINE`
- `MAXIMIZE_DRAW_BEFORE_COMMIT`
- `DISRUPT_OPPONENT`
- `SET_UP_NEXT_TURN`
- `END_TURN_SAFELY`

Start with human-defined codes. Learned discrete plan codes may be introduced only after traces are interpretable and stable.

### StopNode

```python
@dataclass(frozen=True)
class StopNode:
    reason: StopReason
```

Reasons include `PLAN_COMPLETE`, `ATTACK_DECLARED`, `END_TURN`, `NO_SAFE_CONTINUATION`, `BUDGET_EXHAUSTED`, and `FALLBACK_REQUIRED`.

## Resource ledger

Maintain an exact symbolic ledger alongside latent predictions:

```python
@dataclass
class ResourceLedger:
    attachment_used: bool
    retreat_used_or_cost: int
    once_per_turn_effects_used: frozenset[str]
    cards_committed: Counter[str]
    energy_committed: Counter[str]
    bench_slots_free: int
    observed_search_results: tuple[str, ...]
    plan_nodes_executed: int
```

Populate only fields that can be derived from legal observations and CABT metadata. Extend the ledger as needed, but never replace it with an opaque latent vector.

## Plan budgets

```python
@dataclass(frozen=True)
class PlanBudget:
    max_depth: int
    max_nodes: int
    max_subgoals: int
    max_model_calls: int
    max_simulator_calls: int
    deadline_ns: int
```

Budget exhaustion is a normal branch, not an exception. It should deterministically select the best currently valid continuation or fallback.

## Example conditional plan

```json
{
  "plan_id": "turn-31-plan-2",
  "schema_version": "1.0.0",
  "objective": "TAKE_KNOCKOUT",
  "root": {
    "type": "sequence",
    "children": [
      {
        "type": "primitive",
        "selector": {
          "kind": "fingerprint",
          "fingerprint": "PLAY:SEARCH_CARD:HAND->DISCARD:MODE=RESOURCE"
        },
        "expected_group": "PLAY_CARD"
      },
      {
        "type": "conditional",
        "predicate": {
          "kind": "observed_result_class",
          "value": "TARGET_RESOURCE_FOUND"
        },
        "if_true": {
          "type": "sequence",
          "children": [
            {
              "type": "subgoal",
              "goal_code": "REACH_DAMAGE_THRESHOLD",
              "constraints": ["PRESERVE_RETREAT_OUT"]
            },
            {
              "type": "primitive",
              "selector": {
                "kind": "typed_match",
                "category": "ATTACK",
                "effect_id": "BEST_LETHAL_ATTACK"
              },
              "expected_group": "ATTACK"
            }
          ]
        },
        "if_false": {
          "type": "subgoal",
          "goal_code": "SET_UP_NEXT_TURN",
          "constraints": ["DO_NOT_SPEND_LAST_SWITCH_OUT"]
        }
      }
    ]
  },
  "budget": {
    "max_depth": 2,
    "max_nodes": 32,
    "max_subgoals": 8,
    "max_model_calls": 4,
    "max_simulator_calls": 16
  }
}
```

## Compilation and validation pipeline

```text
planner logits
  -> discrete node tokens
  -> schema-level parse
  -> static budget validation
  -> visibility/predicate validation
  -> resource-ledger simulation
  -> current selector resolution
  -> CABT execution
```

Return structured failures such as:

- `SCHEMA_INVALID`
- `DEPTH_EXCEEDED`
- `NODE_BUDGET_EXCEEDED`
- `HIDDEN_PREDICATE_REFERENCE`
- `SELECTOR_AMBIGUOUS`
- `SELECTOR_UNRESOLVABLE`
- `RESOURCE_LEDGER_CONFLICT`
- `NO_FALLBACK`
- `DEADLINE_EXCEEDED`

These reason codes become training data, monitoring dimensions, and regression-test fixtures.

## Plan trace record

Each executed plan should record:

- plan and model versions;
- root objective and candidate rank;
- plan node sequence;
- resolved option indices and fingerprints;
- branch observations;
- predicted versus observed deltas;
- repairs and reason codes;
- simulator/model calls;
- latency by component;
- terminal outcome and horizon values.

The trace is an audit artifact and a source for hard-state relabeling. It must not include hidden information in its deployment-observation section.
