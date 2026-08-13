from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import ResolutionError
from .pointers import get_pointer


@dataclass(frozen=True)
class PredicateResult:
    satisfied: bool
    explanation: dict[str, Any]


def evaluate_predicate(predicate: dict[str, Any], evidence: dict[str, Any]) -> PredicateResult:
    if "literal" in predicate:
        value = predicate["literal"]
        return PredicateResult(value, {"literal": value})
    if "all" in predicate:
        children = [evaluate_predicate(child, evidence) for child in predicate["all"]]
        return PredicateResult(
            all(child.satisfied for child in children),
            {"all": [child.explanation for child in children]},
        )
    if "any" in predicate:
        children = [evaluate_predicate(child, evidence) for child in predicate["any"]]
        return PredicateResult(
            any(child.satisfied for child in children),
            {"any": [child.explanation for child in children]},
        )
    if "not" in predicate:
        child = evaluate_predicate(predicate["not"], evidence)
        return PredicateResult(not child.satisfied, {"not": child.explanation})

    evidence_id = predicate["evidence"]
    payload = evidence.get(evidence_id)
    field = predicate.get("field", "")
    if payload is None:
        return PredicateResult(False, {"evidence": evidence_id, "reason": "missing"})

    sentinel = object()
    actual = get_pointer(payload, field, default=sentinel)
    comparator = next(key for key in ("equals", "gte", "lte", "exists") if key in predicate)
    expected = predicate[comparator]
    try:
        if comparator == "exists":
            satisfied = (actual is not sentinel) is bool(expected)
        elif actual is sentinel:
            satisfied = False
        elif comparator == "equals":
            satisfied = actual == expected
        elif comparator == "gte":
            satisfied = actual >= expected
        else:
            satisfied = actual <= expected
    except TypeError:
        satisfied = False
    explanation = {
        "evidence": evidence_id,
        "field": field,
        "comparator": comparator,
        "expected": expected,
        "actual": "<missing>" if actual is sentinel else actual,
        "satisfied": satisfied,
    }
    return PredicateResult(satisfied, explanation)


def require_evidence_ids(predicate: dict[str, Any]) -> set[str]:
    if "literal" in predicate:
        return set()
    if "evidence" in predicate:
        return {predicate["evidence"]}
    if "not" in predicate:
        return require_evidence_ids(predicate["not"])
    kind = "all" if "all" in predicate else "any"
    result: set[str] = set()
    for child in predicate[kind]:
        result.update(require_evidence_ids(child))
    return result


def assert_declared_evidence(predicate: dict[str, Any], declared: set[str]) -> None:
    missing = sorted(require_evidence_ids(predicate) - declared)
    if missing:
        raise ResolutionError(
            "completion predicate references undeclared evidence: " + ", ".join(missing)
        )
