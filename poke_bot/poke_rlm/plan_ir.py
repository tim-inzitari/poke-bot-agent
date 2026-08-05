"""Typed turn-plan IR aligned with schemas/turn_plan.schema.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from .legal_action import ActionSelector
from .reasons import ReasonCode, StopReason

PLAN_IR_SCHEMA_VERSION = "poke_bot.poke_rlm.turn_plan/v1"
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / "turn_plan.schema.json"
)


class ObjectiveCode(str, Enum):
    ESTABLISH_ATTACKER = "ESTABLISH_ATTACKER"
    FIND_RESOURCE_CLASS = "FIND_RESOURCE_CLASS"
    ENABLE_EVOLUTION = "ENABLE_EVOLUTION"
    REACH_DAMAGE_THRESHOLD = "REACH_DAMAGE_THRESHOLD"
    TAKE_KNOCKOUT = "TAKE_KNOCKOUT"
    DEVELOP_BOARD = "DEVELOP_BOARD"
    PRESERVE_ESCAPE_LINE = "PRESERVE_ESCAPE_LINE"
    MAXIMIZE_DRAW_BEFORE_COMMIT = "MAXIMIZE_DRAW_BEFORE_COMMIT"
    DISRUPT_OPPONENT = "DISRUPT_OPPONENT"
    SET_UP_NEXT_TURN = "SET_UP_NEXT_TURN"
    END_TURN_SAFELY = "END_TURN_SAFELY"


class GoalCode(str, Enum):
    ESTABLISH_ATTACKER = "ESTABLISH_ATTACKER"
    FIND_RESOURCE_CLASS = "FIND_RESOURCE_CLASS"
    ENABLE_EVOLUTION = "ENABLE_EVOLUTION"
    REACH_DAMAGE_THRESHOLD = "REACH_DAMAGE_THRESHOLD"
    TAKE_KNOCKOUT = "TAKE_KNOCKOUT"
    DEVELOP_BOARD = "DEVELOP_BOARD"
    PRESERVE_ESCAPE_LINE = "PRESERVE_ESCAPE_LINE"
    MAXIMIZE_DRAW_BEFORE_COMMIT = "MAXIMIZE_DRAW_BEFORE_COMMIT"
    DISRUPT_OPPONENT = "DISRUPT_OPPONENT"
    SET_UP_NEXT_TURN = "SET_UP_NEXT_TURN"
    END_TURN_SAFELY = "END_TURN_SAFELY"


class PredicateFamily(str, Enum):
    CARD_OR_EFFECT_OBSERVED = "card_or_effect_observed"
    RESULT_CLASS = "result_class"
    LEGAL_FINGERPRINT_PRESENT = "legal_fingerprint_present"
    BOARD_SLOT = "board_slot"
    RESOURCE_THRESHOLD = "resource_threshold"
    ATTACK_AVAILABLE = "attack_available"
    TARGET_STATE = "target_state"
    ONCE_PER_TURN_CONSUMED = "once_per_turn_consumed"
    BUDGET_REACHED = "budget_reached"


_FAMILY_TO_SCHEMA: dict[PredicateFamily, str] = {
    PredicateFamily.CARD_OR_EFFECT_OBSERVED: "observed_result_class",
    PredicateFamily.RESULT_CLASS: "observed_result_class",
    PredicateFamily.LEGAL_FINGERPRINT_PRESENT: "legal_action_present",
    PredicateFamily.BOARD_SLOT: "board_slot_state",
    PredicateFamily.RESOURCE_THRESHOLD: "resource_threshold",
    PredicateFamily.ATTACK_AVAILABLE: "attack_available",
    PredicateFamily.TARGET_STATE: "target_state",
    PredicateFamily.ONCE_PER_TURN_CONSUMED: "once_per_turn_consumed",
    PredicateFamily.BUDGET_REACHED: "budget_reached",
}

_SCHEMA_TO_FAMILY: dict[str, PredicateFamily] = {
    "observed_result_class": PredicateFamily.RESULT_CLASS,
    "legal_action_present": PredicateFamily.LEGAL_FINGERPRINT_PRESENT,
    "board_slot_state": PredicateFamily.BOARD_SLOT,
    "resource_threshold": PredicateFamily.RESOURCE_THRESHOLD,
    "attack_available": PredicateFamily.ATTACK_AVAILABLE,
    "target_state": PredicateFamily.TARGET_STATE,
    "once_per_turn_consumed": PredicateFamily.ONCE_PER_TURN_CONSUMED,
    "budget_reached": PredicateFamily.BUDGET_REACHED,
}


@dataclass(frozen=True)
class ObservationPredicate:
    family: PredicateFamily
    name: str
    expected: Any = True
    requires_public_or_own: bool = True
    constraints: tuple[str, ...] = ()

    def visibility_ok(self, observations: dict[str, Any]) -> bool:
        if not self.requires_public_or_own:
            return False
        if self.name.startswith("opponent.hand") or "hidden" in self.name:
            return False
        return True

    def matches(self, observations: dict[str, Any]) -> bool:
        if not self.visibility_ok(observations):
            return False
        if self.name not in observations:
            return False
        return observations[self.name] == self.expected


@dataclass(frozen=True)
class ResourceDelta:
    hand_delta: int = 0
    energy_delta: int = 0
    bench_delta: int = 0
    prize_delta: int = 0
    once_per_turn_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanBudget:
    max_depth: int = 2
    max_nodes: int = 32
    max_subgoals: int = 8
    max_model_calls: int = 4
    max_simulator_calls: int = 16
    deadline_ns: Optional[int] = None

    def to_json(self) -> dict[str, Any]:
        out = {
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "max_subgoals": self.max_subgoals,
            "max_model_calls": self.max_model_calls,
            "max_simulator_calls": self.max_simulator_calls,
        }
        if self.deadline_ns is not None:
            out["deadline_ns"] = self.deadline_ns
        return out


@dataclass(frozen=True)
class DistributionalValue:
    mean: float = 0.0
    quantiles: tuple[float, ...] = ()


@dataclass(frozen=True)
class PlanProvenance:
    encoder_profile: str = ""
    planner_profile: str = ""
    observation_hash: str = ""
    seed: Optional[int] = None
    notes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class StopNode:
    kind: str = "stop"
    reason: StopReason = StopReason.END_TURN


@dataclass(frozen=True)
class PrimitiveAction:
    kind: str = "primitive"
    selector: ActionSelector = field(default_factory=ActionSelector)
    planned_option_index: Optional[tuple[int, ...]] = None
    expected_group: str = "main"
    expected_delta: ResourceDelta = field(default_factory=ResourceDelta)
    on_unresolvable: Optional["PlanNode"] = None

    @property
    def action_ref(self) -> Optional[int]:
        if self.planned_option_index is None or not self.planned_option_index:
            return None
        return int(self.planned_option_index[0])


@dataclass(frozen=True)
class SequenceNode:
    kind: str = "sequence"
    children: tuple["PlanNode", ...] = ()


@dataclass(frozen=True)
class ConditionalNode:
    kind: str = "conditional"
    predicate: ObservationPredicate = field(
        default_factory=lambda: ObservationPredicate(
            PredicateFamily.RESULT_CLASS, "target_found", True
        )
    )
    if_true: "PlanNode" = field(default_factory=StopNode)
    if_false: "PlanNode" = field(default_factory=StopNode)


@dataclass(frozen=True)
class SubgoalNode:
    kind: str = "subgoal"
    goal_code: GoalCode = GoalCode.DEVELOP_BOARD
    constraints: tuple[str, ...] = ()
    local_budget: PlanBudget = field(default_factory=PlanBudget)
    success_predicate: Optional[ObservationPredicate] = None
    fallback: "PlanNode" = field(default_factory=StopNode)


PlanNode = Union[PrimitiveAction, SequenceNode, ConditionalNode, SubgoalNode, StopNode]


@dataclass(frozen=True)
class TurnPlan:
    plan_id: str
    schema_version: str
    objective: ObjectiveCode
    root: PlanNode
    expected_end_turn: dict[str, Any] = field(default_factory=dict)
    value: DistributionalValue = field(default_factory=DistributionalValue)
    uncertainty: float = 0.0
    budget: PlanBudget = field(default_factory=PlanBudget)
    provenance: PlanProvenance = field(default_factory=PlanProvenance)

    def first_primitive(self) -> Optional[PrimitiveAction]:
        return _first_primitive(self.root)

    def node_count(self) -> int:
        return _count_nodes(self.root)

    @property
    def recursion_depth(self) -> int:
        return _max_subgoal_depth(self.root)

    def iter_nodes(self) -> Iterator[PlanNode]:
        yield from _iter_nodes(self.root)


def _first_primitive(node: PlanNode) -> Optional[PrimitiveAction]:
    if isinstance(node, PrimitiveAction):
        return node
    if isinstance(node, SequenceNode):
        for child in node.children:
            found = _first_primitive(child)
            if found is not None:
                return found
    if isinstance(node, ConditionalNode):
        return _first_primitive(node.if_true)
    if isinstance(node, SubgoalNode):
        return _first_primitive(node.fallback)
    return None


def _count_nodes(node: PlanNode) -> int:
    if isinstance(node, SequenceNode):
        return 1 + sum(_count_nodes(c) for c in node.children)
    if isinstance(node, ConditionalNode):
        return 1 + _count_nodes(node.if_true) + _count_nodes(node.if_false)
    if isinstance(node, SubgoalNode):
        return 1 + _count_nodes(node.fallback)
    if isinstance(node, PrimitiveAction) and node.on_unresolvable is not None:
        return 1 + _count_nodes(node.on_unresolvable)
    return 1


def _max_subgoal_depth(node: PlanNode, depth: int = 0) -> int:
    if isinstance(node, SubgoalNode):
        return max(depth + 1, _max_subgoal_depth(node.fallback, depth + 1))
    if isinstance(node, SequenceNode):
        return max((_max_subgoal_depth(c, depth) for c in node.children), default=depth)
    if isinstance(node, ConditionalNode):
        return max(
            _max_subgoal_depth(node.if_true, depth),
            _max_subgoal_depth(node.if_false, depth),
        )
    if isinstance(node, PrimitiveAction) and node.on_unresolvable is not None:
        return _max_subgoal_depth(node.on_unresolvable, depth)
    return depth


def _iter_nodes(node: PlanNode) -> Iterator[PlanNode]:
    yield node
    if isinstance(node, SequenceNode):
        for child in node.children:
            yield from _iter_nodes(child)
    elif isinstance(node, ConditionalNode):
        yield from _iter_nodes(node.if_true)
        yield from _iter_nodes(node.if_false)
    elif isinstance(node, SubgoalNode):
        yield from _iter_nodes(node.fallback)
    elif isinstance(node, PrimitiveAction) and node.on_unresolvable is not None:
        yield from _iter_nodes(node.on_unresolvable)


def _stop_reason_value(reason: StopReason | str) -> str:
    if isinstance(reason, StopReason):
        return reason.value
    try:
        return StopReason(reason).value
    except ValueError:
        aliases = {
            "end_turn": StopReason.END_TURN.value,
            "objective_met": StopReason.PLAN_COMPLETE.value,
            "budget": StopReason.BUDGET_EXHAUSTED.value,
            "fallback": StopReason.FALLBACK_REQUIRED.value,
            "unsafe": StopReason.NO_SAFE_CONTINUATION.value,
        }
        return aliases.get(str(reason), StopReason.FALLBACK_REQUIRED.value)


def _parse_stop_reason(raw: Any) -> StopReason:
    text = str(raw or StopReason.END_TURN.value)
    try:
        return StopReason(text)
    except ValueError:
        aliases = {
            "end_turn": StopReason.END_TURN,
            "objective_met": StopReason.PLAN_COMPLETE,
            "budget": StopReason.BUDGET_EXHAUSTED,
            "fallback": StopReason.FALLBACK_REQUIRED,
            "unsafe": StopReason.NO_SAFE_CONTINUATION,
        }
        return aliases.get(text, StopReason.FALLBACK_REQUIRED)


def _selector_to_json(selector: ActionSelector) -> dict[str, Any]:
    if selector.fingerprint:
        return {"kind": "fingerprint", "fingerprint": selector.fingerprint}
    category = "option"
    if selector.option_type is not None:
        category = f"type:{selector.option_type}"
    return {
        "kind": "typed_match",
        "category": category,
        "object_id": None if selector.card_id is None else str(selector.card_id),
        "effect_id": None if selector.attack_id is None else str(selector.attack_id),
        "source": selector.source or None,
        "destination": selector.destination or None,
        "arguments": {
            "planned_option_index_path": list(selector.planned_option_index_path),
        },
    }


def _selector_from_json(payload: dict[str, Any]) -> ActionSelector:
    kind = payload.get("kind")
    if kind == "fingerprint":
        return ActionSelector(fingerprint=str(payload.get("fingerprint") or ""))
    args = payload.get("arguments") or {}
    path = args.get("planned_option_index_path") or []
    otype = None
    category = str(payload.get("category") or "")
    if category.startswith("type:"):
        try:
            otype = int(category.split(":", 1)[1])
        except ValueError:
            otype = None
    card_id = payload.get("object_id")
    attack_id = payload.get("effect_id")
    return ActionSelector(
        option_type=otype,
        card_id=int(card_id) if card_id is not None else None,
        attack_id=int(attack_id) if attack_id is not None else None,
        source=str(payload.get("source") or ""),
        destination=str(payload.get("destination") or ""),
        planned_option_index_path=tuple(int(x) for x in path),
    )


def _predicate_to_json(pred: ObservationPredicate) -> dict[str, Any]:
    return {
        "kind": _FAMILY_TO_SCHEMA.get(pred.family, "observed_result_class"),
        "value": {"name": pred.name, "expected": pred.expected},
        "operator": "eq",
        "threshold": None,
    }


def _predicate_from_json(payload: dict[str, Any]) -> ObservationPredicate:
    family = _SCHEMA_TO_FAMILY.get(
        str(payload.get("kind") or ""), PredicateFamily.RESULT_CLASS
    )
    value = payload.get("value")
    if isinstance(value, dict):
        name = str(value.get("name") or "target_found")
        expected = value.get("expected", True)
    else:
        name = "target_found"
        expected = value if value is not None else True
    return ObservationPredicate(family=family, name=name, expected=expected)


def _node_to_json(node: PlanNode) -> dict[str, Any]:
    if isinstance(node, StopNode):
        return {"type": "stop", "reason": _stop_reason_value(node.reason)}
    if isinstance(node, PrimitiveAction):
        planned = node.action_ref
        group = node.expected_group or "main"
        out: dict[str, Any] = {
            "type": "primitive",
            "selector": _selector_to_json(node.selector),
            "planned_option_index": planned,
            "expected_group": group,
            "expected_delta": {
                "hand_delta": node.expected_delta.hand_delta,
                "energy_delta": node.expected_delta.energy_delta,
                "bench_delta": node.expected_delta.bench_delta,
                "prize_delta": node.expected_delta.prize_delta,
                "once_per_turn_flags": list(node.expected_delta.once_per_turn_flags),
                "planned_option_index_path": list(node.planned_option_index or ()),
            },
            "on_unresolvable": _node_to_json(
                node.on_unresolvable
                if node.on_unresolvable is not None
                else StopNode(reason=StopReason.FALLBACK_REQUIRED)
            ),
        }
        return out
    if isinstance(node, SequenceNode):
        return {
            "type": "sequence",
            "children": [_node_to_json(c) for c in node.children],
        }
    if isinstance(node, ConditionalNode):
        return {
            "type": "conditional",
            "predicate": _predicate_to_json(node.predicate),
            "if_true": _node_to_json(node.if_true),
            "if_false": _node_to_json(node.if_false),
        }
    if isinstance(node, SubgoalNode):
        payload = {
            "type": "subgoal",
            "goal_code": node.goal_code.value,
            "constraints": list(node.constraints),
            "local_budget": node.local_budget.to_json(),
            "fallback": _node_to_json(node.fallback),
        }
        if node.success_predicate is not None:
            payload["success_predicate"] = _predicate_to_json(node.success_predicate)
        return payload
    raise TypeError(f"unknown plan node {type(node)!r}")


def turn_plan_to_json(plan: TurnPlan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "schema_version": plan.schema_version,
        "objective": plan.objective.value,
        "root": _node_to_json(plan.root),
        "expected_end_turn": dict(plan.expected_end_turn),
        "value": {"mean": plan.value.mean, "quantiles": list(plan.value.quantiles)},
        "uncertainty": float(plan.uncertainty),
        "budget": plan.budget.to_json(),
        "provenance": {
            "encoder_profile": plan.provenance.encoder_profile,
            "planner_profile": plan.provenance.planner_profile,
            "observation_hash": plan.provenance.observation_hash,
            "seed": plan.provenance.seed,
        },
    }


def _node_kind(payload: dict[str, Any]) -> str:
    return str(payload.get("type") or payload.get("kind") or "")


def _node_from_json(payload: dict[str, Any]) -> PlanNode:
    kind = _node_kind(payload)
    if kind == "stop":
        return StopNode(reason=_parse_stop_reason(payload.get("reason")))
    if kind == "primitive":
        sel_p = payload.get("selector") or {}
        # Support both schema selectors and the older flat selector dict.
        if "kind" in sel_p and sel_p["kind"] in {"fingerprint", "typed_match"}:
            selector = _selector_from_json(sel_p)
        else:
            selector = ActionSelector(
                fingerprint=str(sel_p.get("fingerprint") or ""),
                option_type=sel_p.get("option_type"),
                card_id=sel_p.get("card_id"),
                attack_id=sel_p.get("attack_id"),
                planned_option_index_path=tuple(
                    int(x) for x in (sel_p.get("planned_option_index_path") or [])
                ),
            )
        planned = payload.get("planned_option_index")
        delta_p = payload.get("expected_delta") or {}
        path = delta_p.get("planned_option_index_path")
        if path is None:
            if isinstance(planned, list):
                path = planned
            elif planned is not None:
                path = [planned]
            else:
                path = list(selector.planned_option_index_path)
        return PrimitiveAction(
            selector=selector,
            planned_option_index=tuple(int(x) for x in path) if path else None,
            expected_group=str(payload.get("expected_group") or "main"),
            expected_delta=ResourceDelta(
                hand_delta=int(delta_p.get("hand_delta", 0) or 0),
                energy_delta=int(delta_p.get("energy_delta", 0) or 0),
                bench_delta=int(delta_p.get("bench_delta", 0) or 0),
                prize_delta=int(delta_p.get("prize_delta", 0) or 0),
                once_per_turn_flags=tuple(
                    str(x) for x in (delta_p.get("once_per_turn_flags") or ())
                ),
            ),
            on_unresolvable=_node_from_json(
                payload.get("on_unresolvable")
                or {"type": "stop", "reason": StopReason.FALLBACK_REQUIRED.value}
            ),
        )
    if kind == "sequence":
        children = tuple(_node_from_json(c) for c in (payload.get("children") or []))
        if not children:
            raise ValueError("sequence requires children")
        return SequenceNode(children=children)
    if kind == "conditional":
        pred = _predicate_from_json(payload.get("predicate") or {})
        return ConditionalNode(
            predicate=pred,
            if_true=_node_from_json(payload["if_true"]),
            if_false=_node_from_json(payload["if_false"]),
        )
    if kind == "subgoal":
        success = payload.get("success_predicate")
        return SubgoalNode(
            goal_code=GoalCode(payload.get("goal_code", GoalCode.DEVELOP_BOARD.value)),
            constraints=tuple(str(x) for x in (payload.get("constraints") or ())),
            success_predicate=(
                _predicate_from_json(success) if isinstance(success, dict) else None
            ),
            fallback=_node_from_json(
                payload.get("fallback")
                or {"type": "stop", "reason": StopReason.FALLBACK_REQUIRED.value}
            ),
        )
    raise ValueError(f"unknown plan node kind {kind!r}")


def turn_plan_from_json(payload: dict[str, Any]) -> TurnPlan:
    budget_p = payload.get("budget") or {}
    budget = PlanBudget(
        max_depth=int(budget_p.get("max_depth", 2)),
        max_nodes=int(budget_p.get("max_nodes", 32)),
        max_subgoals=int(budget_p.get("max_subgoals", 8)),
        max_model_calls=int(budget_p.get("max_model_calls", 4)),
        max_simulator_calls=int(budget_p.get("max_simulator_calls", 16)),
        deadline_ns=budget_p.get("deadline_ns"),
    )
    value_p = payload.get("value") or {}
    prov_p = payload.get("provenance") or {}
    return TurnPlan(
        plan_id=str(payload["plan_id"]),
        schema_version=str(payload.get("schema_version") or PLAN_IR_SCHEMA_VERSION),
        objective=ObjectiveCode(payload["objective"]),
        root=_node_from_json(payload["root"]),
        expected_end_turn=dict(payload.get("expected_end_turn") or {}),
        value=DistributionalValue(
            mean=float(value_p.get("mean", 0.0)),
            quantiles=tuple(float(x) for x in (value_p.get("quantiles") or ())),
        ),
        uncertainty=float(payload.get("uncertainty") or 0.0),
        budget=budget,
        provenance=PlanProvenance(
            encoder_profile=str(prov_p.get("encoder_profile") or ""),
            planner_profile=str(prov_p.get("planner_profile") or ""),
            observation_hash=str(prov_p.get("observation_hash") or ""),
            seed=prov_p.get("seed"),
        ),
    )


def validate_plan_static(plan: TurnPlan) -> ReasonCode:
    if plan.node_count() > plan.budget.max_nodes:
        return ReasonCode.BUDGET_NODES
    if plan.budget.max_depth > 3:
        return ReasonCode.BUDGET_DEPTH
    if plan.budget.max_simulator_calls > 75:
        return ReasonCode.BUDGET_SIMULATOR_CALLS

    def walk(node: PlanNode) -> ReasonCode:
        if isinstance(node, ConditionalNode):
            if not node.predicate.requires_public_or_own:
                return ReasonCode.BRANCH_PREDICATE_HIDDEN
            if "hidden" in node.predicate.name or node.predicate.name.startswith(
                "opponent.hand"
            ):
                return ReasonCode.BRANCH_PREDICATE_HIDDEN
            r = walk(node.if_true)
            if r is not ReasonCode.OK:
                return r
            return walk(node.if_false)
        if isinstance(node, SequenceNode):
            if not node.children:
                return ReasonCode.INVALID_PLAN_STATIC
            for child in node.children:
                r = walk(child)
                if r is not ReasonCode.OK:
                    return r
            return ReasonCode.OK
        if isinstance(node, SubgoalNode):
            return walk(node.fallback)
        if isinstance(node, PrimitiveAction) and node.on_unresolvable is not None:
            return walk(node.on_unresolvable)
        return ReasonCode.OK

    return walk(plan.root)


def validate_plan_json_schema(payload: dict[str, Any]) -> ReasonCode:
    """Best-effort JSON Schema validation when jsonschema is installed."""
    if not _SCHEMA_PATH.is_file():
        required = {"plan_id", "schema_version", "objective", "root", "budget"}
        if not required.issubset(payload):
            return ReasonCode.INVALID_PLAN_SCHEMA
        return ReasonCode.OK
    try:
        import jsonschema  # type: ignore
    except Exception:
        required = {"plan_id", "schema_version", "objective", "root", "budget"}
        if not required.issubset(payload):
            return ReasonCode.INVALID_PLAN_SCHEMA
        if _node_kind(payload.get("root") or {}) == "":
            return ReasonCode.INVALID_PLAN_SCHEMA
        return ReasonCode.OK
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(payload, schema)
    except Exception:
        return ReasonCode.INVALID_PLAN_SCHEMA
    return ReasonCode.OK


def round_trip_plan(plan: TurnPlan) -> TurnPlan:
    payload = turn_plan_to_json(plan)
    reason = validate_plan_json_schema(payload)
    if reason is not ReasonCode.OK:
        raise ValueError(reason.value)
    return turn_plan_from_json(payload)
