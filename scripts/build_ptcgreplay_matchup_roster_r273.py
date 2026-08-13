#!/usr/bin/env python3
"""Build an append-only exact-ID Router Format 6 roster candidate for r273."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


BASELINE_SHA256 = "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc"
SNAPSHOT_FILE_SHA256 = "sha256:3bd08735ba9e8d10e16757d3f83aed572ff53273bbd40b69c9488cb3bcdac52e"
SNAPSHOT_SHA256 = "sha256:5d392e6f593aaaab43fd6738586c93d7c1303ee08d1dd2223977812ea1fac5a5"
REGISTRY_SCHEMA = "poke_bot.matchup_adapter_roster/v1"
SLOT_SCHEMA = "poke_bot.matchup_adapter_slot_registry/v1"
ROSTER_RECEIPT_SCHEMA = "poke_bot.ptcgreplay_matchup_roster_candidate_r273/v1"
ROUTABLE = {"active", "dormant"}


class RosterError(RuntimeError):
    """Fail closed on roster, source, or allocation drift."""


def canonical_json(value: object, *, newline: bool = True) -> bytes:
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + ("\n" if newline else "")).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def semantic_digest(payload: dict[str, Any], field: str) -> str:
    detached = dict(payload)
    detached.pop(field, None)
    return "sha256:" + hashlib.sha256(canonical_json(detached)).hexdigest()


def registry_digest(payload: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(payload, newline=False)).hexdigest()


def _load(path: Path, *, expected_sha256: str, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RosterError(f"{label} is not a regular file")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise RosterError(f"{label} digest drifted: {actual}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RosterError(f"{label} is not a JSON object")
    return value


def _validate_baseline(registry: dict[str, Any]) -> None:
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("slot_schema") != SLOT_SCHEMA
        or int(registry.get("slot_capacity") or 0) != 64
    ):
        raise RosterError("baseline is not a Router Format 6 registry")
    slots = list(registry.get("slots") or ())
    if len(slots) != 64:
        raise RosterError("baseline does not expose exactly 64 slots")
    for index, row in enumerate(slots):
        if int(row.get("slot", -1)) != index:
            raise RosterError("baseline slots are reindexed")
        if index < 20:
            if row.get("status") not in ROUTABLE or not row.get("archetype_id"):
                raise RosterError("baseline slots 0 through 19 are not allocated")
        elif row != {
            "slot": index,
            "archetype_id": None,
            "status": "unused",
            "lineage": None,
        }:
            raise RosterError("baseline unused-slot suffix drifted")


def _snapshot_digest_is_valid(snapshot: dict[str, Any]) -> bool:
    claimed = snapshot.get("snapshot_sha256")
    return claimed == semantic_digest(snapshot, "snapshot_sha256")


def _allocate(
    candidate: dict[str, Any],
    *,
    identity: str,
    display_name: str,
    source_namespace: str,
    source_id: int,
    source_name: str,
    lineage: str,
) -> int:
    slots = list(candidate["slots"])
    if any(row.get("archetype_id") == identity for row in slots):
        raise RosterError(f"duplicate candidate identity: {identity}")
    target = next((row for row in slots if row.get("status") == "unused"), None)
    if target is None:
        raise RosterError("Router Format 6 capacity exhausted")
    slot = int(target["slot"])
    target.update(
        archetype_id=identity,
        status="dormant",
        lineage=lineage,
    )
    candidate.setdefault("canonical_display_names", {})[identity] = display_name
    candidate.setdefault("specialist_priority", []).append(identity)
    crosswalk = candidate.setdefault("meta_analysis_source", {}).setdefault(
        "crosswalk", {}
    )
    crosswalk[identity] = {
        "source_namespace": source_namespace,
        "source_id": source_id,
        "source_name": source_name,
        "status": "exact",
    }
    return slot


def build_candidate(
    baseline: dict[str, Any],
    snapshot: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_baseline(baseline)
    if (
        snapshot.get("schema") != "poke_bot.ptcgreplay_matchup_meta_snapshot_r272/v1"
        or snapshot.get("status") != "sealed_completed_meta_snapshot"
        or snapshot.get("snapshot_sha256") != SNAPSHOT_SHA256
        or not _snapshot_digest_is_valid(snapshot)
    ):
        raise RosterError("PTCGReplay snapshot identity or self-digest drifted")
    rows = list(snapshot.get("real_archetypes") or ())
    if len(rows) != 31:
        raise RosterError("PTCGReplay snapshot lacks the exact 31 real archetypes")
    candidate = copy.deepcopy(baseline)
    original_slots = copy.deepcopy(list(baseline["slots"]))
    existing_crosswalk = dict(
        (baseline.get("meta_analysis_source") or {}).get("crosswalk") or {}
    )
    existing_source_ids = {
        int(row["source_id"])
        for row in existing_crosswalk.values()
        if isinstance(row, dict)
        and isinstance(row.get("source_id"), int)
        and not isinstance(row.get("source_id"), bool)
    }
    allocations: list[dict[str, Any]] = []
    seen_snapshot_ids: set[int] = set()
    for row in sorted(rows, key=lambda value: int(value["source_id"])):
        source_id = int(row["source_id"])
        exact_name = str(row["exact_name"])
        if source_id in seen_snapshot_ids:
            raise RosterError("snapshot repeats a numeric source identity")
        seen_snapshot_ids.add(source_id)
        if source_id in existing_source_ids:
            continue
        identity = f"ptcgreplay-source-id-{source_id}"
        slot = _allocate(
            candidate,
            identity=identity,
            display_name=exact_name,
            source_namespace="ptcgreplay",
            source_id=source_id,
            source_name=exact_name,
            lineage=(
                "goal-r273:ptcgreplay-ingest-31:"
                f"sha256:5d392e6f593aaaab43fd6738586c93d7c1303ee08d1dd2223977812ea1fac5a5"
            ),
        )
        allocations.append(
            {
                "slot": slot,
                "archetype_id": identity,
                "source_namespace": "ptcgreplay",
                "source_id": source_id,
                "exact_name": exact_name,
                "appearances": int(row["appearances"]),
                "guide_decklists_pointer": (
                    f"{SNAPSHOT_SHA256}#/real_archetypes/source_id={source_id}/guide_decklists"
                ),
            }
        )

    package_identity = "kaggle-submission-55378392"
    package_slot = _allocate(
        candidate,
        identity=package_identity,
        display_name="Alakazam r195 NO-RTP submission 55378392",
        source_namespace="kaggle_submission",
        source_id=55378392,
        source_name="Alakazam r195 NO-RTP",
        lineage=(
            "goal-r273:kaggle-submission-55378392:"
            "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
        ),
    )
    allocations.append(
        {
            "slot": package_slot,
            "archetype_id": package_identity,
            "source_namespace": "kaggle_submission",
            "source_id": 55378392,
            "exact_name": "Alakazam r195 NO-RTP",
            "bundle_sha256": "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145",
            "must_not_alias_ptcgreplay_source_id_48_or_learner_route": True,
        }
    )

    candidate["active_expert_ids"] = [
        row["archetype_id"]
        for row in candidate["slots"]
        if row["status"] in ROUTABLE
    ]
    candidate["expert_ids"] = list(candidate["active_expert_ids"])
    candidate["required_specialist_count"] = len(candidate["active_expert_ids"])
    candidate["revision"] = int(baseline.get("revision") or 0) + 1
    candidate["status"] = "r273_candidate_exact_zero_dormant_pending_router_fit"
    candidate["runtime_enabled"] = False
    candidate["meta_analysis_source"]["source_snapshot"] = {
        "ingest_id": 31,
        "window_start": "2026-07-10",
        "window_end": "2026-08-10",
        "days": 32,
        "match_facts": 148997,
        "archetypes": 31,
        "snapshot_sha256": SNAPSHOT_SHA256,
        "source_view": "Meta snapshot",
        "decklist_role": "guide_reference_only",
    }

    for index in range(20):
        if candidate["slots"][index] != original_slots[index]:
            raise RosterError(f"candidate changed immutable slot {index}")
    expected_slots = list(range(20, 40))
    actual_slots = [int(row["slot"]) for row in allocations]
    if actual_slots != expected_slots:
        raise RosterError("candidate allocation is not the exact slot 20..39 prefix")
    if any(candidate["slots"][index]["status"] != "unused" for index in range(40, 64)):
        raise RosterError("candidate allocated beyond the required prefix")
    if len(set(candidate["active_expert_ids"])) != 40:
        raise RosterError("candidate does not expose exact 40 distinct routes")

    receipt: dict[str, Any] = {
        "schema": ROSTER_RECEIPT_SCHEMA,
        "goal_revision": 273,
        "status": "append_only_exact_id_candidate_ready_for_router_fit",
        "baseline": {
            "sha256": BASELINE_SHA256,
            "registry_digest": registry_digest(baseline),
        },
        "snapshot": {
            "file_sha256": SNAPSHOT_FILE_SHA256,
            "sha256": SNAPSHOT_SHA256,
            "ingest_id": 31,
            "real_archetypes": 31,
        },
        "candidate_registry_digest": registry_digest(candidate),
        "existing_slots_0_through_19_unchanged": True,
        "allocations": allocations,
        "new_ptcgreplay_source_id_count": 19,
        "distinct_kaggle_package_route_count": 1,
        "new_slots_exact_zero_dormant_required": True,
        "router_fit_and_runtime_activation_pending": True,
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = semantic_digest(receipt, "receipt_sha256")
    return candidate, receipt


def write_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RosterError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        os.unlink(temporary)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    baseline = _load(args.baseline, expected_sha256=BASELINE_SHA256, label="baseline")
    snapshot = _load(
        args.snapshot,
        expected_sha256=SNAPSHOT_FILE_SHA256,
        label="snapshot",
    )
    candidate, receipt = build_candidate(baseline, snapshot)
    write_create_only(args.candidate, candidate)
    write_create_only(args.receipt, receipt)
    print(
        json.dumps(
            {
                "candidate": str(args.candidate),
                "candidate_registry_digest": receipt["candidate_registry_digest"],
                "receipt": str(args.receipt),
                "receipt_sha256": receipt["receipt_sha256"],
                "allocated_slots": [row["slot"] for row in receipt["allocations"]],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
