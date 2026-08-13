from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .evidence import assert_declared_evidence, evaluate_predicate
from .io import load_json, load_jsonl, resolve_local_path, sha256_file, verify_reference
from .pointers import apply_operations
from .validate import (
    validate_amendments,
    validate_activations,
    validate_contract,
    validate_evidence_index,
    validate_gateway,
)


@dataclass(frozen=True)
class Resolution:
    goal_id: str
    current_revision: int
    active_revision: int
    desired_contract: dict[str, Any]
    active_contract: dict[str, Any]
    pending_activations: list[dict[str, Any]]
    evidence: dict[str, Any]
    status: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_evidence(root: Path, index: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for entry in index["entries"]:
        path = resolve_local_path(root, entry["path"])
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            from .errors import IntegrityError

            raise IntegrityError(
                f"evidence {entry['id']} checksum mismatch: expected "
                f"{entry['sha256']}, observed {actual}"
            )
        result[entry["id"]] = load_json(path)
    return result


def resolve_gateway(gateway_path: str | Path) -> Resolution:
    path = Path(gateway_path).resolve()
    root = path.parent
    gateway = validate_gateway(load_json(path))
    goal_id = gateway["goal_id"]

    contract_path = verify_reference(root, gateway["contract"], label="gateway.contract")
    amendments_path = verify_reference(root, gateway["amendments"], label="gateway.amendments")
    activations_path = verify_reference(root, gateway["activations"], label="gateway.activations")
    evidence_index_path = verify_reference(
        root, gateway["evidence_index"], label="gateway.evidence_index"
    )

    base_contract = validate_contract(load_json(contract_path), goal_id=goal_id)
    amendments = validate_amendments(
        load_jsonl(amendments_path),
        goal_id=goal_id,
        base_revision=base_contract["revision"],
        current_revision=gateway["current_revision"],
    )
    activations = validate_activations(
        load_jsonl(activations_path),
        goal_id=goal_id,
        amendment_revisions=[item["revision"] for item in amendments],
    )
    active_amendment_revisions = {
        item["amendment_revision"] for item in activations
    }
    index = validate_evidence_index(load_json(evidence_index_path), goal_id=goal_id)
    evidence = _load_evidence(root, index)

    desired = deepcopy(base_contract)
    active = deepcopy(base_contract)
    active_revision = base_contract["revision"]
    pending: list[dict[str, Any]] = []
    for amendment in amendments:
        desired = apply_operations(desired, amendment["operations"])
        desired["revision"] = amendment["revision"]
        if amendment["revision"] in active_amendment_revisions:
            active = apply_operations(active, amendment["operations"])
            active["revision"] = amendment["revision"]
            active_revision = amendment["revision"]
        else:
            pending.append(
                {
                    "revision": amendment["revision"],
                    "mode": amendment["activation_mode"],
                    "reason": amendment.get("reason"),
                }
            )

    validate_contract(desired, goal_id=goal_id)
    validate_contract(active, goal_id=goal_id)
    declared = set(evidence)
    assert_declared_evidence(active["completion"], declared)
    assert_declared_evidence(desired["completion"], declared)
    active_completion = evaluate_predicate(active["completion"], evidence)
    desired_completion = evaluate_predicate(desired["completion"], evidence)
    transition_status = [
        {
            "id": item["id"],
            "goal_id": item["goal_id"],
            "goal_gateway": item["goal_gateway"],
            "after": item["after"],
            "ready": active_completion.satisfied,
        }
        for item in active.get("transitions", [])
    ]

    status = {
        "schema": "durable-goals.status/v1",
        "goal_id": goal_id,
        "current_revision": gateway["current_revision"],
        "active_revision": active_revision,
        "pending_activation_revisions": [item["revision"] for item in pending],
        "active_completion": {
            "satisfied": active_completion.satisfied,
            "explanation": active_completion.explanation,
        },
        "desired_completion": {
            "satisfied": desired_completion.satisfied,
            "explanation": desired_completion.explanation,
        },
        "evidence_ids": sorted(evidence),
        "transitions": transition_status,
        "ready_transitions": [
            item["id"] for item in transition_status if item["ready"]
        ],
        "authoritative": False,
        "derived_from": {
            "contract": gateway["contract"]["sha256"],
            "amendments": gateway["amendments"]["sha256"],
            "activations": gateway["activations"]["sha256"],
            "evidence_index": gateway["evidence_index"]["sha256"],
        },
    }
    return Resolution(
        goal_id=goal_id,
        current_revision=gateway["current_revision"],
        active_revision=active_revision,
        desired_contract=desired,
        active_contract=active,
        pending_activations=pending,
        evidence=evidence,
        status=status,
    )
