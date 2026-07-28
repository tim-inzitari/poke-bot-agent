#!/usr/bin/env python3
"""Continue sequential specialist training after any post-Starmie pass."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import yaml

from poke_bot.pure_rl.model_registry import sha256, verify_frozen_model
from scripts.run_sequential_specialist_handoff import (
    run as run_sequential_handoff,
    service_active,
    validate_frozen_predecessor_registry,
    validate_source,
)
from scripts.select_next_specialist import select as select_next_specialist
from scripts.resolve_specialist_assets import resolve_specialist_assets
from scripts.run_post_starmie_core_handoff import run as run_core_refresh_handoff
from scripts.run_starmie_expert_bootstrap import (
    decision_fusion_handoff_contract,
    expanded_handoff_training_contract,
)


SCHEMA = "poke_bot.specialist_cycle_handoff_contract/v1"
SELECTOR = "POKEBOT_ACTIVE_SPECIALIST"
GATE_REVISION = re.compile(
    r"^(?P<prefix>.+\+frozen-specialists-r)(?P<revision>\d+)$"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _compatible_prior_cumulative_contract(
    existing: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    """Accept only known additive controller fields from an older receipt."""

    variants: list[dict[str, Any]] = []
    for remove_transition in (False, True):
        for remove_selection_controls in (False, True):
            for remove_boundary_scope in (False, True):
                for remove_boundary_assets in (False, True):
                    candidate = copy.deepcopy(current)
                    if remove_transition:
                        candidate["trigger"].pop(
                            "threshold_transition_receipt", None
                        )
                    if remove_selection_controls:
                        next_specialist = candidate["next_specialist"]
                        next_specialist.pop(
                            "minimum_decisions_by_specialist", None
                        )
                        next_specialist.pop(
                            "minimum_records_by_specialist", None
                        )
                        next_specialist.pop("strict_priority_prefix", None)
                    if remove_boundary_scope and "runtime" in candidate:
                        candidate["runtime"].pop(
                            "future_assets_scope", None
                        )
                    if remove_boundary_assets and "runtime" in candidate:
                        runtime = candidate["runtime"]
                        runtime["inactive_tree_candidate"] = None
                        runtime["candidate_audit"] = None
                        runtime["future_assets_receipt"] = None
                        runtime.pop("future_assets_scope", None)
                    guide_variants = [candidate]
                    prior_without_required_guide = copy.deepcopy(candidate)
                    prior_without_required_guide["next_specialist"].pop(
                        "current_deck_guide_required", None
                    )
                    guide_variants.append(prior_without_required_guide)
                    for guide_candidate in guide_variants:
                        variants.append(guide_candidate)
                        prior_without_nonblocking_fallback = copy.deepcopy(
                            guide_candidate
                        )
                        prior_without_nonblocking_fallback.pop(
                            "core_failure_fallback", None
                        )
                        variants.append(prior_without_nonblocking_fallback)
                        prior_gate_path = copy.deepcopy(guide_candidate)
                        if (
                            isinstance(existing.get("trigger"), dict)
                            and "gate_contract" in existing["trigger"]
                        ):
                            prior_gate_path["trigger"]["gate_contract"] = (
                                existing["trigger"]["gate_contract"]
                            )
                            variants.append(prior_gate_path)
                    if (
                        isinstance(candidate.get("runtime"), dict)
                        and isinstance(existing.get("runtime"), dict)
                    ):
                        prior_assets = copy.deepcopy(candidate)
                        for key in (
                            "inactive_tree_candidate",
                            "candidate_audit",
                            "future_assets_receipt",
                            "future_assets_scope",
                        ):
                            if key in existing["runtime"]:
                                prior_assets["runtime"][key] = existing[
                                    "runtime"
                                ][key]
                            else:
                                prior_assets["runtime"].pop(key, None)
                        variants.append(prior_assets)
                        if (
                            isinstance(existing.get("trigger"), dict)
                            and "gate_contract" in existing["trigger"]
                        ):
                            prior_assets_and_gate = copy.deepcopy(prior_assets)
                            prior_assets_and_gate["trigger"][
                                "gate_contract"
                            ] = existing["trigger"]["gate_contract"]
                            variants.append(prior_assets_and_gate)
    return existing in variants


def _path(section: dict[str, Any], key: str) -> Path:
    raw = str(section.get(key) or "").strip()
    if not raw:
        raise RuntimeError(f"required path is missing: {key}")
    return Path(raw).expanduser().resolve()


def _accepted_core(family: Path, ready_path: Path) -> dict[str, Any]:
    ready = _read(ready_path)
    core = verify_frozen_model(family)
    if (
        ready.get("status") != "ready"
        or ready.get("gameplay_regression_passed") is not True
        or ready.get("checkpoint_digest") != core.get("checkpoint_digest")
    ):
        raise RuntimeError("shared core is not accepted")
    return {
        **core,
        "ready": str(ready_path),
        "ready_digest": sha256(ready_path),
        "version": int(ready.get("version") or len(ready.get("teacher_checkpoint_digests") or ())),
    }


def _resolve_current_core(section: dict[str, Any]) -> dict[str, Any]:
    """Use the latest accepted cumulative core, falling back only for bootstrap."""

    pointer_raw = str(section.get("latest_pointer") or "").strip()
    if pointer_raw:
        pointer_path = Path(pointer_raw).expanduser().resolve()
        if pointer_path.is_file():
            pointer = _read(pointer_path)
            family = _path(pointer, "family")
            ready_path = _path(pointer, "ready")
            core = _accepted_core(family, ready_path)
            if (
                pointer.get("schema")
                != "poke_bot.latest_cumulative_core_pointer/v1"
                or pointer.get("checkpoint_digest") != core["checkpoint_digest"]
                or pointer.get("ready_digest") != core["ready_digest"]
            ):
                raise RuntimeError("latest cumulative core pointer changed")
            return core
    return _accepted_core(
        _path(section, "family"),
        _path(section, "ready"),
    )


def _publish_latest_core_pointer(
    path: Path,
    *,
    family: Path,
    ready_path: Path,
    previous_digest: str,
) -> dict[str, Any]:
    core = _accepted_core(family, ready_path)
    payload = {
        "schema": "poke_bot.latest_cumulative_core_pointer/v1",
        "family": str(family),
        "ready": str(ready_path),
        "ready_digest": core["ready_digest"],
        "checkpoint_digest": core["checkpoint_digest"],
        "version": int(core["version"]),
        "previous_checkpoint_digest": previous_digest,
    }
    if path.is_file() and _read(path) == payload:
        return payload
    _atomic(path, payload)
    return payload


def _active_specialist(runtime: dict[str, Any]) -> str:
    selector = _path(runtime, "selector_env")
    rows = [
        line.split("=", 1)[1].strip()
        for line in selector.read_text(encoding="utf-8").splitlines()
        if line.startswith(SELECTOR + "=")
    ]
    if len(rows) != 1 or not rows[0]:
        raise RuntimeError("canonical specialist selector is absent or duplicated")
    return rows[0]


def _required_specialist_ids(state_path: Path) -> set[str]:
    state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise RuntimeError("canonical specialist state is not an object")
    expected = int(
        ((state.get("target_registry") or {}).get("required_target_count") or 0)
    )
    roster_path = state_path.parent / "matchup_adapter_roster.json"
    roster = _read(roster_path)
    identifiers = {
        str(value) for value in (roster.get("expert_ids") or []) if str(value)
    }
    rows = {
        str(row.get("id") or ""): dict(row)
        for row in (state.get("specialists") or [])
        if isinstance(row, dict) and str(row.get("id") or "")
    }
    order = [
        str(value)
        for value in (
            (state.get("training_priority") or {}).get(
                "ordered_unfinished_ids_after_active"
            )
            or []
        )
    ]
    progress = dict((state.get("current") or {}).get("program_progress") or {})
    remaining = progress.get("remaining_after_active")
    if remaining is None:
        remaining = progress.get("remaining_unfinished")
    unfinished_statuses = {
        "unstarted",
        "restart_required",
        "bootstrap_partial",
        "bootstrap_complete",
        "blocked",
    }
    unfinished = {
        specialist_id
        for specialist_id, row in rows.items()
        if str(row.get("status") or "") in unfinished_statuses
    }
    if (
        expected != len(identifiers)
        or int(roster.get("required_specialist_count") or 0) != len(identifiers)
        or int(roster.get("physical_checkpoint_rows") or 0) != len(identifiers)
        or "" in identifiers
        or set(rows) != identifiers
        or len(order) != len(set(order))
        or set(order) != unfinished
        or int(remaining if remaining is not None else -1) != len(order)
    ):
        raise RuntimeError(
            "canonical specialist roster or unfinished priority projection changed"
        )
    return identifiers


def population_transition_ready(
    completed_ids: set[str],
    required_ids: set[str],
) -> bool:
    """Require the exact canonical roster before population training can start."""

    if not required_ids:
        raise RuntimeError("population transition requires canonical roster")
    if not completed_ids.issubset(required_ids):
        raise RuntimeError(
            "frozen registry contains specialists outside canonical roster"
        )
    return completed_ids == required_ids


def _is_expected_additive_gate_successor(
    *,
    active_id: str,
    saved_gate: dict[str, Any],
    current_gate: dict[str, Any],
    frozen_registry: dict[str, Any],
) -> bool:
    """Recognize only the idempotent local half of a materialization retry."""

    checkpoint_digest = str(saved_gate.get("checkpoint_digest") or "")
    if not checkpoint_digest.startswith("sha256:"):
        return False
    saved_gate_id = str(saved_gate.get("gate_id") or "")
    current_gate_id = str(current_gate.get("active_gate_id") or "")
    saved_revision = GATE_REVISION.fullmatch(saved_gate_id)
    current_revision = GATE_REVISION.fullmatch(current_gate_id)
    if (
        saved_revision is None
        or current_revision is None
        or saved_revision.group("prefix") != current_revision.group("prefix")
        or int(current_revision.group("revision"))
        != int(saved_revision.group("revision")) + 1
    ):
        return False
    next_gate = dict(current_gate.get("next_gate") or {})
    if str(next_gate.get("id") or "") != current_gate_id:
        return False
    saved_roster = {
        str(value) for value in (saved_gate.get("roster_ids") or []) if str(value)
    }
    current_roster = {
        str(row.get("opponent_id") or "")
        for row in (next_gate.get("roster") or [])
        if str(row.get("opponent_id") or "")
    }
    matching_rows = [
        row
        for row in (next_gate.get("roster") or [])
        if row.get("frozen_specialist") is True
        and str(row.get("archetype_id") or "") == active_id
        and str(row.get("frozen_checkpoint_digest") or "") == checkpoint_digest
    ]
    if (
        len(matching_rows) != 1
        or not saved_roster
        or not saved_roster.issubset(current_roster)
        or len(current_roster) != len(saved_roster) + 1
    ):
        return False
    registry_rows = [
        row
        for row in (frozen_registry.get("specialists") or [])
        if str(row.get("specialist_id") or "") == active_id
        and row.get("frozen") is True
        and str(row.get("checkpoint_digest") or "") == checkpoint_digest
    ]
    return len(registry_rows) == 1


def _start_population_handoff(runtime: dict[str, Any]) -> None:
    service = str(runtime.get("population_handoff_service") or "")
    if not service.startswith("pokebot-") or not service.endswith(".service"):
        raise RuntimeError("population handoff service is not configured")
    completed = subprocess.run(
        ["/usr/bin/systemctl", "--user", "start", service],
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"population handoff service failed to start: {service}"
        )


def _source(
    *,
    contract: dict[str, Any],
    active_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = dict(contract["runtime"])
    registry = _read(_path(runtime, "runtime_registry"))
    row = dict((registry.get("specialists") or {}).get(active_id) or {})
    handler = dict(row.get("pass_handler") or {})
    if row.get("status") != "ready" or not handler:
        raise RuntimeError("active specialist runtime row is not ready")
    handler_state = _read(Path(str(handler["state"])).expanduser().resolve())
    saved_gate = dict(handler_state.get("gate") or {})
    saved_gate_contract = str(saved_gate.get("contract") or "").strip()
    saved_gate_path = Path(saved_gate_contract).expanduser().resolve()
    exact_contract_unchanged = (
        bool(saved_gate_contract)
        and saved_gate_path.is_file()
        and saved_gate.get("contract_sha256") == sha256(saved_gate_path)
    )
    expected_additive_successor = False
    if (
        not exact_contract_unchanged
        and handler_state.get("phase") == "complete_handoff_started"
        and saved_gate_path.is_file()
    ):
        expected_additive_successor = _is_expected_additive_gate_successor(
            active_id=active_id,
            saved_gate=saved_gate,
            current_gate=_read(saved_gate_path),
            frozen_registry=_read(
                _path(runtime, "frozen_specialist_registry")
            ),
        )
    if (
        saved_gate.get("completion_authority")
        == "explicit_owner_ceiling_acceptance"
    ):
        if not exact_contract_unchanged and not expected_additive_successor:
            raise RuntimeError(
                "saved ceiling-acceptance gate contract changed"
            )
        gate_contract = str(saved_gate_path)
    elif expected_additive_successor:
        # Materializing a newly frozen specialist deliberately extends the
        # shared additive S+ roster.  Resume from the handler's checksum-bound
        # passing evidence instead of reinterpreting that historical result
        # under the successor roster that now includes itself.
        gate_contract = str(saved_gate_path)
    else:
        gate_contract = str(
            Path(str(registry["runtime_root"]))
            / str(registry["active_gate_contract"])
        )
    source = {
        "id": active_id,
        "run_dir": (
            "/home/inzi/poke-bot-agent/outputs/pure_rl/" + str(row["run_name"])
        ),
        "training_service": runtime["training_service"],
        "gate_contract": gate_contract,
        "gate_marker_name": row["terminal_gate_marker"],
        "minimum_completed_iteration": int(
            row.get(
                "minimum_terminal_iteration",
                registry["minimum_terminal_iteration"],
            )
        ),
        "matchup_runtime_tree": row["matchup_runtime_tree"],
        "passed_family": str(
            _path(runtime, "registry_root") / str(handler["family"])
        ),
        "handler_state": handler["state"],
    }
    transition_receipt = str(
        handler.get("threshold_transition_receipt") or ""
    ).strip()
    if transition_receipt:
        source["threshold_transition_receipt"] = transition_receipt
    if expected_additive_successor:
        frozen_path = Path(source["passed_family"]).expanduser().resolve()
        frozen = verify_frozen_model(frozen_path)
        queued = [
            dict(value)
            for value in (handler_state.get("queued_submissions") or [])
        ]
        validation = dict(saved_gate.get("validation") or {})
        if (
            handler_state.get("schema")
            != "poke_bot.passed_gate_handler/v1"
            or handler_state.get("phase") != "complete_handoff_started"
            or handler_state.get("submission_mode") != "queue_and_continue"
            or not validation
            or not all(value is True for value in validation.values())
            or handler_state.get("frozen_model") != frozen
            or [int(value.get("copy_number", -1)) for value in queued] != [1]
            or any(
                value.get("checkpoint_checksum")
                != saved_gate.get("checkpoint_digest")
                for value in queued
            )
            or frozen.get("checkpoint_digest")
            != saved_gate.get("checkpoint_digest")
            or int(saved_gate.get("commit_boundary", -1))
            < int(source["minimum_completed_iteration"])
        ):
            raise RuntimeError(
                "saved post-materialization source evidence changed"
            )
        evidence = {
            "specialist_id": active_id,
            "gate": saved_gate,
            "frozen_family": str(frozen_path),
            "frozen_manifest_sha256": sha256(frozen_path / "manifest.json"),
            "checkpoint_digest": frozen["checkpoint_digest"],
            "queued_submission_copies": [
                {
                    "copy_number": int(value["copy_number"]),
                    "label": value["label"],
                    "checkpoint_checksum": value["checkpoint_checksum"],
                    "queued_at": value["queued_at"],
                }
                for value in queued
            ],
        }
    else:
        evidence = validate_source({"source_specialist": source})
    completion_authority = str(
        (evidence.get("gate") or {}).get("completion_authority") or ""
    ).strip()
    if completion_authority:
        source["completion_authority"] = completion_authority
    return source, evidence


def _generated(
    *,
    contract: dict[str, Any],
    source: dict[str, Any],
    selected: dict[str, Any],
    core_digest: str,
    candidate_tree: Path | None = None,
    candidate_audit: Path | None = None,
) -> dict[str, Any]:
    runtime = dict(contract["runtime"])
    gate = dict(contract["gate_materialization"])
    training = dict(contract["training"])
    training["expanded_heads"] = expanded_handoff_training_contract()
    training["decision_fusion"] = decision_fusion_handoff_contract()
    specialist_id = str(selected["specialist_id"])
    source_id = str(source["id"])
    state_root = _path(runtime, "state_root")
    runtime_registration = {
        "runtime_tree": source["matchup_runtime_tree"],
        "runtime_registry": runtime["runtime_registry"],
        "selector_env": runtime["selector_env"],
        "state_root": runtime["state_root"],
        "run_name": f"pure_rl_{specialist_id}_temporal1_8k_v1_20260723",
        "handoff_service": runtime["handoff_service"],
        "gate_handler_service": runtime["gate_handler_service"],
    }
    if runtime.get("matchup_v6") is not None:
        runtime_registration["matchup_v6"] = copy.deepcopy(
            runtime["matchup_v6"]
        )
    if runtime.get("inactive_tree_candidate"):
        candidate_tree = candidate_tree or _path(
            runtime, "inactive_tree_candidate"
        )
        candidate_audit = candidate_audit or _path(runtime, "candidate_audit")
        runtime_registration.update(
            {
                "inactive_tree_candidate": str(candidate_tree),
                "candidate_audit": str(candidate_audit),
                "activated_runtime_tree": str(
                    state_root
                    / f"{specialist_id}-public-matchup-tree-v33.json"
                ),
                "minimum_validation_precision": 0.93,
                "minimum_validation_weighted_support": 10_000,
                "consecutive_required": 2,
                "allow_zero_materialized_adapters": True,
            }
        )
    return {
        "schema": "poke_bot.sequential_specialist_handoff_contract/v1",
        "source_specialist": source,
        "shared_core": {
            "family": contract["shared_core"]["family"],
            "checkpoint_checksum": core_digest,
        },
        "next_specialist": {
            "id": specialist_id,
            "expert_corpus": selected["pointer"],
            "family_name": f"{specialist_id}_expert_bootstrap_from_core_v2",
            "ready": str(
                state_root / f"{specialist_id}-expert-bootstrap-ready-v2.json"
            ),
            "run_name": f"{specialist_id}_expert_bootstrap_from_core_v2_20260723",
            "run_dir": (
                "/home/inzi/poke-bot-agent/outputs/bootstrap/"
                f"{specialist_id}-expert-bootstrap-from-core-v2"
            ),
            "cpu_pack_root": (
                "/home/inzi/poke-bot-agent/outputs/bootstrap/cpu-packs/"
                f"{specialist_id}-expert-bootstrap-v2"
            ),
            "activation_receipt": str(
                state_root / f"{specialist_id}-specialist-rl-activation-v2.json"
            ),
            "training_service": runtime["training_service"],
            "gate_contract": gate["base_gate_contract"],
            "frozen_specialist_registry": gate[
                "base_frozen_specialist_registry"
            ],
        },
        "gate_materialization": {
            **gate,
            "archetype_label": source_id.replace("-", " ").title(),
            "receipt": str(
                state_root / f"{source_id}-splus-gate-materialization-v1.json"
            ),
        },
        "training": training,
        "submission_policy": {
            "required_copies": 1,
            "completion_blocks_handoff": False,
            "queue_order": "oldest_first",
        },
        "runtime_registration": runtime_registration,
        "paths": {
            "python": runtime["python"],
            "registry_root": runtime["registry_root"],
            "state": str(
                state_root / f"post-{source_id}-{specialist_id}-handoff-v1.json"
            ),
            "lock": (
                "/home/inzi/.local/state/pokebot/"
                f"post-{source_id}-{specialist_id}-handoff-v1.lock"
            ),
        },
    }


def _cumulative_core_contract(
    *,
    template: dict[str, Any],
    cycle: dict[str, Any],
    source: dict[str, Any],
    completed_ids: set[str],
    runtime_registry: dict[str, Any],
    current_core: dict[str, Any],
) -> dict[str, Any]:
    """Materialize the next immutable cumulative-core handoff contract."""

    # Every checksum-verified frozen specialist contributes to the next
    # cumulative core. Historical post-Starmie refreshes excluded Alakazam,
    # but that policy was explicitly retired before the post-Lucario rebuild.
    excluded: set[str] = set()
    eligible = sorted(completed_ids)
    if source["id"] not in eligible:
        raise RuntimeError("newly passing specialist is absent from core teachers")
    if len(eligible) < 2:
        raise RuntimeError("cumulative core refresh requires at least two teachers")

    registry_root = _path(dict(cycle["runtime"]), "registry_root")
    core_corpus = _path(dict(cycle["selection"]), "core_corpus")
    if not core_corpus.is_file():
        raise RuntimeError(
            "expanded cumulative-core corpus is not atomically promoted"
        )
    specialist_rows = dict(runtime_registry.get("specialists") or {})
    frozen_family_overrides = dict(
        (cycle.get("shared_core") or {}).get("frozen_family_overrides") or {}
    )
    teachers: list[dict[str, Any]] = []
    for specialist_id in eligible:
        row = dict(specialist_rows.get(specialist_id) or {})
        handler = dict(row.get("pass_handler") or {})
        family_name = str(
            handler.get("family")
            or frozen_family_overrides.get(specialist_id)
            or ""
        ).strip()
        if not family_name:
            raise RuntimeError(
                f"cumulative teacher lacks frozen family: {specialist_id}"
            )
        frozen = verify_frozen_model(registry_root / family_name)
        teachers.append(
            {
                "specialist_id": specialist_id,
                "mode": "frozen_inference_only",
                "checkpoint": str(frozen["model_path"]),
                "checksum": str(frozen["checkpoint_digest"]),
            }
        )

    version = len(teachers)
    state_root = _path(dict(cycle["runtime"]), "state_root")
    # Revision 17 requires the same learned 17-input action path in every
    # successor lineage. The cumulative core therefore has its own explicit
    # fused architecture identity instead of silently evaluating a legacy
    # flat-policy core against fused frozen teachers.
    base_family_name = (
        f"deck_agnostic_core_cumulative_v{version}_fused_v1"
    )
    base_run_stem = (
        f"deck-agnostic-core-cumulative-v{version}-fused-v1"
    )
    teacher_digests = [str(row["checksum"]) for row in teachers]
    attempt = 1
    # A completed regression is authoritative at this boundary. A rejection
    # must immediately select the latest accepted core and advance production;
    # the next completed specialist creates the next cumulative-core version.
    # Never spend inter-deck wall time on same-boundary split-seed retries.
    for completed_attempt in range(1, 6):
        completed_suffix = (
            "" if completed_attempt == 1 else f"-attempt{completed_attempt}"
        )
        regression_path = state_root / (
            base_run_stem
            + completed_suffix
            + "-gameplay-regression.json"
        )
        if not regression_path.is_file():
            attempt = completed_attempt
            break
        regression = _read(regression_path)
        identity = dict(regression.get("identity") or {})
        criteria = dict(regression.get("criteria") or {})
        if (
            regression.get("schema")
            != "poke_bot.multi_teacher_core_gameplay_regression/v1"
            or regression.get("passed") not in {True, False}
            or regression.get("training_eligible") is not False
            or identity.get("teacher_checkpoint_digests") != teacher_digests
            or criteria.get("all_reports_valid") is not True
        ):
            raise RuntimeError(
                "existing cumulative-core regression is not a valid "
                "retry authority"
            )
        attempt = completed_attempt
        break
    suffix = "" if attempt == 1 else f"_attempt{attempt}"
    run_suffix = "" if attempt == 1 else f"-attempt{attempt}"
    family_name = base_family_name + suffix
    run_stem = base_run_stem + run_suffix
    contract = copy.deepcopy(template)
    contract["status"] = "staged"
    contract["trigger"] = {
        "specialist_id": source["id"],
        "requires_training_complete": True,
        "requires_exact_frozen_checkpoint": True,
        "completion_authority": (
            source.get("completion_authority")
            or (
                "explicit_owner_ceiling_acceptance"
                if source.get("ceiling_acceptance_receipt")
                else "measured_gate_pass"
            )
        ),
        "requires_exact_frozen_passing_checkpoint": (
            source.get("completion_authority")
            != "explicit_owner_ceiling_acceptance"
        ),
        "run_dir": source["run_dir"],
        "training_service": source["training_service"],
        "gate_contract": source["gate_contract"],
        "gate_marker_name": source["gate_marker_name"],
        "minimum_completed_iteration": int(
            source["minimum_completed_iteration"]
        ),
        "passed_family": source["passed_family"],
        "handler_state": source["handler_state"],
    }
    transition_receipt = str(
        source.get("threshold_transition_receipt") or ""
    ).strip()
    if transition_receipt:
        contract["trigger"]["threshold_transition_receipt"] = transition_receipt
    core = dict(contract["core_refresh"])
    core.update(
        {
            "version": version,
            "family": str(registry_root / family_name),
            "initialization": {
                "checkpoint": str(current_core["model_path"]),
                "checksum": str(current_core["checkpoint_digest"]),
            },
            "teachers": teachers,
            "ready_receipt": str(state_root / f"{run_stem}-ready.json"),
            "run_name": run_stem.replace("-", "_") + "_20260723",
            "run_dir": (
                f"/home/inzi/poke-bot-agent/outputs/bootstrap/{run_stem}"
            ),
            "cpu_pack_root": (
                "/home/inzi/poke-bot-agent/outputs/bootstrap/cpu-packs/"
                + f"deck-agnostic-core-cumulative-v{version}"
            ),
            "balanced_corpus": {
                "pointer": str(core_corpus),
                "checksum": sha256(core_corpus),
            },
            "expanded_heads": expanded_handoff_training_contract(),
            "decision_fusion": decision_fusion_handoff_contract(),
            "direct_checkpoint_tensor_sources_exclude": [],
            "refresh_attempt": attempt,
            "split_seed": 20260723 + attempt - 1,
        }
    )
    if attempt == 5:
        diagnostic_names = (
            f"{base_run_stem}-attempt4-parent-diagnostic-"
            "gameplay-regression.json",
            f"{base_run_stem}-attempt4-grim-anchor-diagnostic-"
            "gameplay-regression.json",
        )
        diagnostic_receipts: list[dict[str, Any]] = []
        for name in diagnostic_names:
            path = state_root / name
            if not path.is_file():
                raise RuntimeError(
                    "teacher-behavior repair requires both failed "
                    f"parameter-space diagnostics: missing={path}"
                )
            payload = _read(path)
            criteria = dict(payload.get("criteria") or {})
            if (
                payload.get("schema")
                != "poke_bot.multi_teacher_core_gameplay_regression/v1"
                or payload.get("passed") is not False
                or payload.get("training_eligible") is not False
                or criteria.get("all_reports_valid") is not True
            ):
                raise RuntimeError(
                    "parameter-space diagnostic is not valid repair evidence"
                )
            diagnostic_receipts.append(
                {"path": str(path), "checksum": sha256(path)}
            )
        core["teacher_behavior_distillation"] = {
            "schema": "poke_bot.teacher_behavior_distillation/v1",
            "enabled": True,
            "target": (
                "matching_archetype_frozen_teacher_greedy_action"
            ),
            "causal_inputs_only": True,
            "loss_weight": 0.5,
            "parameter_space_diagnostics": diagnostic_receipts,
        }
    contract["core_refresh"] = core
    contract["core_failure_fallback"] = {
        "enabled": True,
        "behavior": "continue_with_latest_accepted_core",
        "continue_refresh_after_each_specialist": True,
        "version": int(current_core["version"]),
        "family": str(Path(str(current_core["model_path"])).parent),
        "checkpoint_digest": str(current_core["checkpoint_digest"]),
        "ready_receipt": str(current_core["ready"]),
        "owner_decision": "GOAL.md#/decision-ledger/revision-19",
    }
    contract["acceptance"]["regression_result"] = str(
        state_root / f"{run_stem}-gameplay-regression.json"
    )
    contract["next_specialist"].update(
        {
            "hot_start_core_version": version,
            "state": cycle["selection"]["state"],
            "corpus_root": cycle["selection"]["corpus_root"],
            "minimum_decisions": cycle["selection"]["minimum_decisions"],
            "minimum_decisions_by_specialist": dict(
                cycle["selection"].get(
                    "minimum_decisions_by_specialist", {}
                )
            ),
            "minimum_records_by_specialist": dict(
                cycle["selection"].get(
                    "minimum_records_by_specialist", {}
                )
            ),
            "strict_priority_prefix": list(
                cycle["selection"].get("strict_priority_prefix", [])
            ),
            "selection_receipt": str(
                state_root / f"post-{source['id']}-next-specialist-selection.json"
            ),
            "generated_handoff_contract": str(
                state_root
                / f"post-{source['id']}-generated-sequential-handoff.json"
            ),
            "prestage_receipt": str(
                _path(dict(cycle["prestage"]), "receipt")
            ),
            "current_deck_guide_required": bool(
                cycle["prestage"].get("current_deck_guide_required", False)
            ),
        }
    )
    contract["gate_materialization"] = copy.deepcopy(
        cycle["gate_materialization"]
    )
    contract["gate_materialization"]["receipt"] = str(
        state_root / f"{source['id']}-splus-gate-materialization.json"
    )
    contract["runtime"].update(
        {
            **cycle["runtime"],
            "next_handoff_service": cycle["runtime"]["handoff_service"],
            "state": str(
                state_root / f"post-{source['id']}-core-v{version}-handoff.json"
            ),
            "lock": (
                "/home/inzi/.local/state/pokebot/"
                f"post-{source['id']}-core-v{version}-handoff.lock"
            ),
        }
    )
    return contract


def run(contract_path: Path) -> int:
    contract = _read(contract_path.expanduser().resolve())
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("specialist cycle contract schema changed")
    runtime = dict(contract["runtime"])
    lock_path = _path(runtime, "lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        active_id = _active_specialist(runtime)
        if service_active(str(runtime["training_service"])):
            raise RuntimeError("active specialist trainer has not terminated")
        source, source_evidence = _source(
            contract=contract, active_id=active_id
        )
        frozen_registry_path = _path(runtime, "frozen_specialist_registry")
        frozen_registry = _read(frozen_registry_path)
        validate_frozen_predecessor_registry(source_evidence, frozen_registry)
        core_section = dict(contract["shared_core"])
        core = _resolve_current_core(core_section)
        frozen_registry = _read(frozen_registry_path)
        validate_frozen_predecessor_registry(
            source_evidence,
            frozen_registry,
        )
        runtime_registry = _read(_path(runtime, "runtime_registry"))
        completed_ids = {
            str(row["specialist_id"])
            for row in (frozen_registry.get("specialists") or [])
            if row.get("frozen") is True
        }
        completed_ids.add(active_id)
        required_ids = _required_specialist_ids(
            _path(dict(contract["selection"]), "state")
        )
        if population_transition_ready(completed_ids, required_ids):
            _start_population_handoff(runtime)
            return 0
        template = _read(_path(contract, "core_refresh_template"))
        cumulative = _cumulative_core_contract(
            template=template,
            cycle=contract,
            source=source,
            completed_ids=completed_ids,
            runtime_registry=runtime_registry,
            current_core=core,
        )
        core_version = int(cumulative["core_refresh"]["version"])
        generated_path = (
            _path(runtime, "state_root")
            / (
                f"post-{active_id}-cumulative-core-v{core_version}"
                + "-fused-v1"
                + (
                    ""
                    if int(cumulative["core_refresh"].get("refresh_attempt", 1))
                    == 1
                    else (
                        "-attempt"
                        + str(
                            int(
                                cumulative["core_refresh"][
                                    "refresh_attempt"
                                ]
                            )
                        )
                    )
                )
                + "-handoff.json"
            )
        )
        if generated_path.is_file():
            existing = _read(generated_path)
            if existing != cumulative:
                if not _compatible_prior_cumulative_contract(
                    existing, cumulative
                ):
                    raise RuntimeError("existing cumulative core handoff differs")
                _atomic(generated_path, cumulative)
        else:
            _atomic(generated_path, cumulative)
        run_core_refresh_handoff(generated_path)
        pointer_raw = str(core_section.get("latest_pointer") or "").strip()
        if not pointer_raw:
            raise RuntimeError("latest cumulative core pointer is not configured")
        refreshed = dict(cumulative["core_refresh"])
        refreshed_ready_path = _path(refreshed, "ready_receipt")
        refreshed_ready = _read(refreshed_ready_path)
        if (
            refreshed_ready.get("status") == "ready"
            and refreshed_ready.get("gameplay_regression_passed") is True
        ):
            _publish_latest_core_pointer(
                Path(pointer_raw).expanduser().resolve(),
                family=_path(refreshed, "family"),
                ready_path=refreshed_ready_path,
                previous_digest=str(core["checkpoint_digest"]),
            )
        else:
            # A rejected refresh remains immutable diagnostic evidence.  The
            # handoff has already continued with the checksum-accepted fallback,
            # so the canonical latest-core pointer must remain unchanged.
            resolved_after_fallback = _resolve_current_core(core_section)
            if (
                resolved_after_fallback["checkpoint_digest"]
                != core["checkpoint_digest"]
            ):
                raise RuntimeError(
                    "latest accepted core changed during fallback handoff"
                )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    return run(args.contract)


if __name__ == "__main__":
    raise SystemExit(main())
