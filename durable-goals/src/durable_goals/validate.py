from __future__ import annotations

from typing import Any

from .errors import ValidationError
from .io import is_sha256


GATEWAY_SCHEMA = "durable-goals.gateway/v1"
CONTRACT_SCHEMA = "durable-goals.contract/v1"
AMENDMENT_SCHEMA = "durable-goals.amendment/v1"
ACTIVATION_SCHEMA = "durable-goals.activation/v1"
EVIDENCE_INDEX_SCHEMA = "durable-goals.evidence-index/v1"


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def validate_gateway(value: Any) -> dict[str, Any]:
    gateway = _mapping(value, "gateway")
    if gateway.get("schema") != GATEWAY_SCHEMA:
        raise ValidationError(f"gateway.schema must equal {GATEWAY_SCHEMA}")
    _nonempty_string(gateway.get("goal_id"), "gateway.goal_id")
    _positive_int(gateway.get("current_revision"), "gateway.current_revision")
    for key in ("contract", "amendments", "activations", "evidence_index"):
        _mapping(gateway.get(key), f"gateway.{key}")
    if "status" in gateway:
        status = _mapping(gateway["status"], "gateway.status")
        _nonempty_string(status.get("path"), "gateway.status.path")
    return gateway


def validate_contract(value: Any, *, goal_id: str) -> dict[str, Any]:
    contract = _mapping(value, "contract")
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValidationError(f"contract.schema must equal {CONTRACT_SCHEMA}")
    if contract.get("goal_id") != goal_id:
        raise ValidationError("contract.goal_id disagrees with gateway.goal_id")
    _positive_int(contract.get("revision"), "contract.revision")
    _nonempty_string(contract.get("objective"), "contract.objective")
    invariants = contract.get("invariants")
    if not isinstance(invariants, list):
        raise ValidationError("contract.invariants must be an array")
    seen: set[str] = set()
    for index, invariant in enumerate(invariants):
        item = _mapping(invariant, f"contract.invariants[{index}]")
        invariant_id = _nonempty_string(item.get("id"), f"contract.invariants[{index}].id")
        _nonempty_string(item.get("statement"), f"contract.invariants[{index}].statement")
        if invariant_id in seen:
            raise ValidationError(f"duplicate invariant id: {invariant_id}")
        seen.add(invariant_id)
    validate_predicate(contract.get("completion"), "contract.completion")
    delegations = contract.get("delegations", [])
    if not isinstance(delegations, list):
        raise ValidationError("contract.delegations must be an array")
    for index, delegation in enumerate(delegations):
        item = _mapping(delegation, f"contract.delegations[{index}]")
        _nonempty_string(item.get("goal_id"), f"contract.delegations[{index}].goal_id")
        owns = item.get("owns")
        if not isinstance(owns, list) or not owns or not all(isinstance(x, str) and x for x in owns):
            raise ValidationError(f"contract.delegations[{index}].owns must contain strings")
    transitions = contract.get("transitions", [])
    if not isinstance(transitions, list):
        raise ValidationError("contract.transitions must be an array")
    transition_ids: set[str] = set()
    for index, transition in enumerate(transitions):
        item = _mapping(transition, f"contract.transitions[{index}]")
        transition_id = _nonempty_string(
            item.get("id"), f"contract.transitions[{index}].id"
        )
        if transition_id in transition_ids:
            raise ValidationError(f"duplicate transition id: {transition_id}")
        transition_ids.add(transition_id)
        _nonempty_string(
            item.get("goal_id"), f"contract.transitions[{index}].goal_id"
        )
        _nonempty_string(
            item.get("goal_gateway"),
            f"contract.transitions[{index}].goal_gateway",
        )
        if item.get("after") != "completion":
            raise ValidationError(
                f"contract.transitions[{index}].after must equal completion"
            )
    return contract


def validate_predicate(value: Any, label: str) -> None:
    predicate = _mapping(value, label)
    keys = [
        key for key in ("all", "any", "not", "evidence", "literal") if key in predicate
    ]
    if len(keys) != 1:
        raise ValidationError(f"{label} must contain exactly one predicate operator")
    kind = keys[0]
    if kind == "literal":
        if not isinstance(predicate["literal"], bool):
            raise ValidationError(f"{label}.literal must be a boolean")
    elif kind in {"all", "any"}:
        children = predicate[kind]
        if not isinstance(children, list) or not children:
            raise ValidationError(f"{label}.{kind} must be a non-empty array")
        for index, child in enumerate(children):
            validate_predicate(child, f"{label}.{kind}[{index}]")
    elif kind == "not":
        validate_predicate(predicate[kind], f"{label}.not")
    else:
        evidence_id = _nonempty_string(predicate["evidence"], f"{label}.evidence")
        del evidence_id
        field = predicate.get("field", "")
        if not isinstance(field, str) or (field and not field.startswith("/")):
            raise ValidationError(f"{label}.field must be an RFC 6901 JSON pointer")
        comparators = [key for key in ("equals", "gte", "lte", "exists") if key in predicate]
        if len(comparators) != 1:
            raise ValidationError(f"{label} must contain exactly one comparator")


def validate_amendments(
    values: list[Any], *, goal_id: str, base_revision: int, current_revision: int
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    expected_revision = base_revision + 1
    for index, value in enumerate(values):
        amendment = _mapping(value, f"amendments[{index}]")
        if amendment.get("schema") != AMENDMENT_SCHEMA:
            raise ValidationError(f"amendments[{index}].schema must equal {AMENDMENT_SCHEMA}")
        if amendment.get("goal_id") != goal_id:
            raise ValidationError(f"amendments[{index}].goal_id disagrees with gateway")
        revision = _positive_int(amendment.get("revision"), f"amendments[{index}].revision")
        if revision != expected_revision:
            raise ValidationError(
                f"amendment revisions must be contiguous: expected {expected_revision}, got {revision}"
            )
        expected_revision += 1
        _nonempty_string(amendment.get("recorded_at"), f"amendments[{index}].recorded_at")
        _nonempty_string(amendment.get("authority"), f"amendments[{index}].authority")
        operations = amendment.get("operations")
        if not isinstance(operations, list) or not operations:
            raise ValidationError(f"amendments[{index}].operations must be non-empty")
        for op_index, operation in enumerate(operations):
            item = _mapping(operation, f"amendments[{index}].operations[{op_index}]")
            if item.get("op") not in {"set", "remove"}:
                raise ValidationError("amendment op must be set or remove")
            path = item.get("path")
            if not isinstance(path, str) or not path.startswith("/"):
                raise ValidationError("amendment operation path must be a JSON pointer")
            if path in {"/schema", "/goal_id", "/revision"}:
                raise ValidationError(f"amendments may not modify identity field {path}")
        _nonempty_string(
            amendment.get("activation_mode"),
            f"amendments[{index}].activation_mode",
        )
        result.append(amendment)
    observed_revision = result[-1]["revision"] if result else base_revision
    if observed_revision != current_revision:
        raise ValidationError(
            f"gateway.current_revision={current_revision} but history resolves to {observed_revision}"
        )
    return result


def validate_activations(
    values: list[Any], *, goal_id: str, amendment_revisions: list[int]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    expected_prefix = amendment_revisions[: len(values)]
    observed: list[int] = []
    for index, value in enumerate(values):
        activation = _mapping(value, f"activations[{index}]")
        if activation.get("schema") != ACTIVATION_SCHEMA:
            raise ValidationError(f"activations[{index}].schema must equal {ACTIVATION_SCHEMA}")
        if activation.get("goal_id") != goal_id:
            raise ValidationError(f"activations[{index}].goal_id disagrees with gateway")
        revision = _positive_int(
            activation.get("amendment_revision"),
            f"activations[{index}].amendment_revision",
        )
        observed.append(revision)
        _nonempty_string(
            activation.get("activated_at"), f"activations[{index}].activated_at"
        )
        if "evidence_id" in activation:
            _nonempty_string(
                activation["evidence_id"], f"activations[{index}].evidence_id"
            )
        result.append(activation)
    if observed != expected_prefix:
        raise ValidationError(
            "activations must be an ordered amendment prefix: "
            f"expected {expected_prefix}, observed {observed}"
        )
    return result


def validate_evidence_index(value: Any, *, goal_id: str) -> dict[str, Any]:
    index = _mapping(value, "evidence_index")
    if index.get("schema") != EVIDENCE_INDEX_SCHEMA:
        raise ValidationError(f"evidence_index.schema must equal {EVIDENCE_INDEX_SCHEMA}")
    if index.get("goal_id") != goal_id:
        raise ValidationError("evidence_index.goal_id disagrees with gateway")
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise ValidationError("evidence_index.entries must be an array")
    seen: set[str] = set()
    for entry_index, entry in enumerate(entries):
        item = _mapping(entry, f"evidence_index.entries[{entry_index}]")
        evidence_id = _nonempty_string(item.get("id"), f"evidence_index.entries[{entry_index}].id")
        if evidence_id in seen:
            raise ValidationError(f"duplicate evidence id: {evidence_id}")
        seen.add(evidence_id)
        _nonempty_string(item.get("path"), f"evidence_index.entries[{entry_index}].path")
        checksum = item.get("sha256")
        if not is_sha256(checksum):
            raise ValidationError(f"evidence {evidence_id} lacks a sha256 checksum")
    return index
