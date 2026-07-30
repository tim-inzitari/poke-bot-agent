#!/usr/bin/env python3
"""Continue sequential specialist training after any post-Starmie pass."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
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
POST_FLEET_REFRESH_SCHEMA = "poke_bot.post_fleet_specialist_refresh/v1"
POST_FLEET_REFRESH_COMPLETION_SCHEMA = (
    "poke_bot.post_fleet_specialist_refresh_completion/v1"
)
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
    """Accept only known controller-only changes from an older contract."""

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
    # A generated boundary contract may predate a validated canonical protocol
    # refresh performed while the outgoing gate handler is still completing.
    # Both fused-policy sections bind the whole protocol file, so even a
    # Hammer-only validation receipt changes both digests. Permit exactly that
    # paired checksum refresh while requiring every structural field to remain
    # identical. Unequal or malformed paired digests remain fail-closed.
    try:
        existing_core = existing["core_refresh"]
        current_core = current["core_refresh"]
        existing_digests = (
            existing_core["decision_fusion"]["canonical_config_sha256"],
            existing_core["expanded_heads"]["canonical_config_sha256"],
        )
        current_digests = (
            current_core["decision_fusion"]["canonical_config_sha256"],
            current_core["expanded_heads"]["canonical_config_sha256"],
        )
    except (KeyError, TypeError):
        existing_digests = ()
        current_digests = ()
    valid_paired_refresh = (
        len(existing_digests) == 2
        and len(current_digests) == 2
        and existing_digests[0] == existing_digests[1]
        and current_digests[0] == current_digests[1]
        and all(
            isinstance(value, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None
            for value in (*existing_digests, *current_digests)
        )
    )
    if valid_paired_refresh:
        for candidate in list(variants):
            prior_protocol = copy.deepcopy(candidate)
            prior_protocol["core_refresh"]["decision_fusion"][
                "canonical_config_sha256"
            ] = existing_digests[0]
            prior_protocol["core_refresh"]["expanded_heads"][
                "canonical_config_sha256"
            ] = existing_digests[1]
            variants.append(prior_protocol)
    try:
        old_fleet_config = copy.deepcopy(
            existing["runtime"]["matchup_v6"]["fleet"]
        )
        new_fleet_config = copy.deepcopy(
            current["runtime"]["matchup_v6"]["fleet"]
        )
        old_fleet_receipt = Path(
            old_fleet_config["receipt"]
        ).expanduser().resolve()
        new_fleet_receipt = Path(
            new_fleet_config["receipt"]
        ).expanduser().resolve()
    except (KeyError, TypeError, ValueError):
        old_fleet_config = None
        new_fleet_config = None
        old_fleet_receipt = None
        new_fleet_receipt = None
    prior_fleet_receipt_valid = False
    pending_same_receipt = False
    if (
        old_fleet_receipt is not None
        and new_fleet_receipt is not None
    ):
        if old_fleet_receipt != new_fleet_receipt and old_fleet_receipt.is_file():
            old_fleet = _read(old_fleet_receipt)
            prior_fleet_receipt_valid = (
                old_fleet.get("schema")
                == "poke_bot.matchup_adapter_v6_fleet_activation/v1"
                and old_fleet.get("status") == "active"
            )
        pending_same_receipt = (
            old_fleet_receipt == new_fleet_receipt
            and not new_fleet_receipt.exists()
        )
    if prior_fleet_receipt_valid or pending_same_receipt:
        old_fleet_identity = copy.deepcopy(old_fleet_config)
        new_fleet_identity = copy.deepcopy(new_fleet_config)
        for fleet in (old_fleet_identity, new_fleet_identity):
            fleet.pop("receipt", None)
            elmo = dict(fleet.get("elmo") or {})
            for key in ("image", "build_context", "dockerfile"):
                elmo.pop(key, None)
            fleet["elmo"] = elmo
        new_elmo_config = dict(new_fleet_config.get("elmo") or {})
        if (
            old_fleet_identity == new_fleet_identity
            and str(new_elmo_config.get("image") or "").startswith(
                "poke-bot-truenas-worker:"
            )
            and all(
                Path(str(new_elmo_config.get(key) or ""))
                .expanduser()
                .is_absolute()
                for key in ("build_context", "dockerfile")
            )
        ):
            for candidate in list(variants):
                prior_fleet_receipt = copy.deepcopy(candidate)
                prior_fleet_receipt["runtime"]["matchup_v6"][
                    "fleet"
                ] = old_fleet_config
                variants.append(prior_fleet_receipt)
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
    route_ids = {
        str(value) for value in (roster.get("expert_ids") or []) if str(value)
    }
    slot_route_ids = {
        str(row.get("archetype_id") or "")
        for row in (roster.get("slots") or [])
        if isinstance(row, dict)
        and str(row.get("status") or "") in {"active", "dormant"}
        and str(row.get("archetype_id") or "")
    }
    legacy_prefix_length = int(
        roster.get("legacy_v5_prefix_length") or 0
    )
    all_rows = {
        str(row.get("id") or ""): dict(row)
        for row in (state.get("specialists") or [])
        if isinstance(row, dict) and str(row.get("id") or "")
    }
    owner_removed_ids = {
        str(value)
        for value in (
            ((state.get("training_priority") or {}).get("owner_removal") or {})
            .get("specialist_ids")
            or []
        )
        if str(value)
    }
    rows = {
        specialist_id: row
        for specialist_id, row in all_rows.items()
        if specialist_id not in owner_removed_ids
    }
    identifiers = set(rows)
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
        or int(progress.get("required_specialists_total") or 0) != expected
        or not route_ids
        or int(roster.get("required_specialist_count") or 0) != len(route_ids)
        or int(roster.get("physical_checkpoint_rows") or 0)
        != legacy_prefix_length
        or legacy_prefix_length <= 0
        or slot_route_ids != route_ids
        or len(roster.get("slots") or []) != int(
            roster.get("slot_capacity") or 0
        )
        or len(route_ids) != len(roster.get("expert_ids") or [])
        or "" in identifiers
        or len(order) != len(set(order))
        or set(order) & owner_removed_ids
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
    *,
    completed_refresh_ids: Sequence[str],
    required_refresh_order: Sequence[str],
) -> bool:
    """Require the exact fleet and ordered refresh phase before population."""

    if not required_ids:
        raise RuntimeError("population transition requires canonical roster")
    if not completed_ids.issubset(required_ids):
        raise RuntimeError(
            "frozen registry contains specialists outside canonical roster"
        )
    refresh_order = [str(value) for value in required_refresh_order]
    completed_refreshes = [str(value) for value in completed_refresh_ids]
    if (
        not refresh_order
        or len(refresh_order) != len(set(refresh_order))
        or len(completed_refreshes) != len(set(completed_refreshes))
        or completed_refreshes
        != refresh_order[: len(completed_refreshes)]
    ):
        raise RuntimeError("post-fleet refresh order or completion changed")
    return (
        completed_ids == required_ids
        and completed_refreshes == refresh_order
    )


def _validated_post_fleet_refresh_progress(
    *,
    state_path: Path,
    cycle_contract: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Validate checksum-bound post-fleet refresh completion evidence."""

    contract = dict(cycle_contract.get("post_fleet_refresh") or {})
    state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise RuntimeError("canonical specialist state is not an object")
    phase = dict(state.get("post_fleet_refresh") or {})
    required_order = [
        str(value)
        for value in (contract.get("ordered_specialist_ids") or [])
    ]
    state_order = [
        str(value)
        for value in (phase.get("ordered_specialist_ids") or [])
    ]
    completed_ids = [
        str(value)
        for value in (
            phase.get("completed_refresh_specialist_ids") or []
        )
    ]
    receipt_rows = list(phase.get("completed_refresh_receipts") or [])
    pending_ids = [
        str(value)
        for value in (phase.get("pending_specialist_ids") or [])
    ]
    contract_first_refresh = dict(contract.get("first_refresh") or {})
    state_first_refresh = dict(phase.get("first_refresh") or {})
    contract_release_gates = dict(contract.get("release_gates") or {})
    state_release_gates = dict(phase.get("release_gates") or {})
    final_alakazam_gate = dict(
        contract_release_gates.get("final_alakazam_model_computation") or {}
    )
    broader_capacity_gate = dict(
        contract_release_gates.get(
            "broader_multi_archetype_capacity_program"
        )
        or {}
    )
    migration = dict(
        contract_first_refresh.get("preferred_parent_migration") or {}
    )
    migration_fallback = dict(migration.get("failure_fallback") or {})
    turn_order = dict(contract_first_refresh.get("turn_order") or {})
    originals = dict(phase.get("original_checkpoint_identities") or {})
    phase_complete = len(completed_ids) == len(required_order)
    if (
        contract.get("schema") != POST_FLEET_REFRESH_SCHEMA
        or contract.get("completion_receipt_schema")
        != POST_FLEET_REFRESH_COMPLETION_SCHEMA
        or int(contract.get("owner_decision_revision") or 0) != 64
        or int(contract.get("required_fleet_count") or 0) != 15
        or contract.get("terminal_required_specialist_id") != "slowking"
        or contract.get(
            "slowking_freeze_and_registration_"
            "immediately_triggers_first_refresh"
        )
        is not True
        or required_order
        != ["alakazam", "marnie-s-grimmsnarl-ex"]
        or phase.get("schema") != POST_FLEET_REFRESH_SCHEMA
        or phase.get("completion_receipt_schema")
        != POST_FLEET_REFRESH_COMPLETION_SCHEMA
        or int(phase.get("owner_decision_revision") or 0) != 64
        or int(phase.get("required_fleet_count") or 0) != 15
        or dict(phase.get("trigger") or {}).get(
            "terminal_required_specialist_id"
        )
        != "slowking"
        or dict(phase.get("trigger") or {}).get(
            "slowking_freeze_and_registration_"
            "immediately_triggers_first_refresh"
        )
        is not True
        or contract_release_gates != state_release_gates
        or final_alakazam_gate.get("required_receipts")
        != [
            "required_specialist_fleet_complete_for_final_alakazam_v1",
            "capacity_research_resource_lease_v1",
        ]
        or final_alakazam_gate.get(
            "all_required_before_model_computation"
        )
        is not True
        or final_alakazam_gate.get("authorization_scope")
        != "final_format_alakazam_refresh_only"
        or final_alakazam_gate.get("no_receipt_no_model_work") is not True
        or broader_capacity_gate.get("required_receipt")
        != "post_refresh_sequence_complete_for_capacity_v2"
        or broader_capacity_gate.get(
            "requires_final_format_alakazam_and_marnie_refresh_complete"
        )
        is not True
        or broader_capacity_gate.get("no_receipt_no_model_work") is not True
        or contract_first_refresh != state_first_refresh
        or contract_first_refresh.get("specialist_id") != "alakazam"
        or contract_first_refresh.get("model_format")
        != "final_submission_format"
        or turn_order.get("training_seat_split")
        != {"first": 0.5, "second": 0.5}
        or turn_order.get("exact_even_split_required") is not True
        or turn_order.get("deterministic_assignment_required") is not True
        or turn_order.get("seat_count_parity_receipt_required") is not True
        or turn_order.get("seat_count_parity_receipt_schema")
        != "poke_bot.alakazam_refresh_seat_split/v1"
        or turn_order.get("seat_count_receipt_required_stages")
        != ["assigned", "actual", "consumed"]
        or turn_order.get(
            "equal_first_second_counts_required_at_each_stage"
        )
        is not True
        or turn_order.get("package_preference") != "first_if_allowed"
        or turn_order.get("second_focus_1_to_7_allowed") is not False
        or turn_order.get("always_second_arm_allowed") is not False
        or turn_order.get("second_preferring_refresh_copy_allowed") is not False
        or migration.get("parent_checkpoint_sha256")
        != (
            "sha256:270b5156781b0a95f703abe3e8fe13866"
            "d2fbb4c85a8f32534f99af74aece2ea"
        )
        or migration_fallback.get(
            "migration_failure_receipt_preserved"
        )
        is not True
        or migration_fallback.get(
            "ordinary_same_archetype_alakazam_refresh_initialized_from"
        )
        != "then_latest_checksum_accepted_core"
        or migration_fallback.get(
            "expand_only_that_completed_alakazam_derivative_to_final_format"
        )
        is not True
        or migration_fallback.get(
            "latest_core_direct_final_format_tensor_parent_allowed"
        )
        is not False
        or migration_fallback.get(
            "partial_old_alakazam_core_overlay_allowed"
        )
        is not False
        or state_order != required_order
        or len(completed_ids) != len(receipt_rows)
        or completed_ids != required_order[: len(completed_ids)]
        or pending_ids != required_order[len(completed_ids) :]
        or set(originals) != set(required_order)
        or dict(originals.get("alakazam") or {}).get("immutable") is not True
        or dict(originals.get("alakazam") or {}).get(
            "may_satisfy_new_refresh_gate"
        )
        is not False
        or phase.get("population_transition_blocked_until_complete")
        is not True
        or (
            phase_complete
            and (
                phase.get("status") != "complete"
                or phase.get("active_refresh_specialist_id") is not None
                or phase.get("next_refresh_specialist_id") is not None
                or dict(phase.get("trigger") or {}).get(
                    "all_required_specialists_training_complete"
                )
                is not True
                or dict(phase.get("trigger") or {}).get(
                    "all_required_specialists_frozen_and_registered"
                )
                is not True
            )
        )
    ):
        raise RuntimeError("post-fleet refresh state contract changed")

    repository_root = state_path.parent.parent
    for specialist_id, receipt_row in zip(completed_ids, receipt_rows):
        if not isinstance(receipt_row, dict):
            raise RuntimeError("post-fleet refresh receipt row is invalid")
        receipt_path = Path(str(receipt_row.get("receipt") or "")).expanduser()
        if not receipt_path.is_absolute():
            receipt_path = (repository_root / receipt_path).resolve()
        receipt_digest = str(receipt_row.get("receipt_sha256") or "")
        if (
            str(receipt_row.get("specialist_id") or "") != specialist_id
            or not receipt_path.is_file()
            or not receipt_digest.startswith("sha256:")
            or sha256(receipt_path) != receipt_digest
        ):
            raise RuntimeError(
                f"post-fleet refresh receipt identity failed: {specialist_id}"
            )
        receipt = _read(receipt_path)
        original_digest = str(
            dict(originals[specialist_id]).get("checksum") or ""
        )
        checkpoint_digest = str(
            receipt.get("refresh_checkpoint_checksum") or ""
        )
        core = dict(receipt.get("resolved_core") or {})
        training = dict(receipt.get("training_contract") or {})
        if (
            receipt.get("schema")
            != POST_FLEET_REFRESH_COMPLETION_SCHEMA
            or receipt.get("status") != "passed_frozen_registered"
            or str(receipt.get("specialist_id") or "") != specialist_id
            or not str(receipt.get("refresh_model_version") or "")
            or not checkpoint_digest.startswith("sha256:")
            or checkpoint_digest == original_digest
            or receipt.get("original_checkpoint_checksum")
            != original_digest
            or receipt.get("current_gate_pass") is not True
            or receipt.get("frozen") is not True
            or receipt.get("registered") is not True
            or not str(receipt.get("gate_receipt_sha256") or "").startswith(
                "sha256:"
            )
            or not str(receipt.get("freeze_receipt_sha256") or "").startswith(
                "sha256:"
            )
            or not str(
                receipt.get("registration_receipt_sha256") or ""
            ).startswith("sha256:")
            or core.get("status") != "checksum_accepted"
            or not str(core.get("checkpoint_checksum") or "").startswith(
                "sha256:"
            )
            or not str(core.get("ready_receipt_sha256") or "").startswith(
                "sha256:"
            )
            or training.get("canonical_source")
            != "config/rl_protocol.yaml#/specialist_training"
            or not str(training.get("sha256") or "").startswith("sha256:")
        ):
            raise RuntimeError(
                f"post-fleet refresh completion failed: {specialist_id}"
            )
    return required_order, completed_ids


def _is_expected_additive_gate_successor(
    *,
    active_id: str,
    saved_gate: dict[str, Any],
    current_gate: dict[str, Any],
    frozen_registry: dict[str, Any],
    materialization_receipt: dict[str, Any] | None = None,
    current_gate_sha256: str | None = None,
) -> bool:
    """Recognize only the idempotent local half of a materialization retry."""

    checkpoint_digest = str(saved_gate.get("checkpoint_digest") or "")
    if not checkpoint_digest.startswith("sha256:"):
        return False
    saved_gate_id = str(
        saved_gate.get("base_gate_id")
        or saved_gate.get("gate_id")
        or ""
    )
    current_gate_id = str(current_gate.get("active_gate_id") or "")
    saved_revision = GATE_REVISION.fullmatch(saved_gate_id)
    current_revision = GATE_REVISION.fullmatch(current_gate_id)
    sequential_revision = (
        saved_revision is not None
        and current_revision is not None
        and int(current_revision.group("revision"))
        == int(saved_revision.group("revision")) + 1
        and (
            saved_revision.group("prefix")
            == current_revision.group("prefix")
            or saved_gate.get("completion_authority")
            == "explicit_owner_ceiling_acceptance"
        )
    )
    receipt = dict(materialization_receipt or {})
    receipt_bound_revision = (
        receipt.get("schema")
        == "poke_bot.frozen_specialist_gate_materialization/v1"
        and receipt.get("specialist_id") == active_id
        and receipt.get("checkpoint_digest") == checkpoint_digest
        and receipt.get("gate_id") == current_gate_id
        and str(receipt.get("gate_contract_sha256") or "")
        == str(current_gate_sha256 or "")
    )
    if not sequential_revision and not receipt_bound_revision:
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
    if receipt_bound_revision and (
        receipt.get("opponent_id")
        != matching_rows[0].get("opponent_id")
        or active_id
        not in set(receipt.get("frozen_specialist_ids") or ())
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
        state_root_raw = str(runtime.get("state_root") or "").strip()
        materialization_path = (
            Path(state_root_raw).expanduser().resolve()
            / f"{active_id}-splus-gate-materialization.json"
            if state_root_raw
            else None
        )
        materialization_receipt = (
            _read(materialization_path)
            if materialization_path is not None
            and materialization_path.is_file()
            else None
        )
        expected_additive_successor = _is_expected_additive_gate_successor(
            active_id=active_id,
            saved_gate=saved_gate,
            current_gate=_read(saved_gate_path),
            frozen_registry=_read(
                _path(runtime, "frozen_specialist_registry")
            ),
            materialization_receipt=materialization_receipt,
            current_gate_sha256=sha256(saved_gate_path),
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
        approved_submission_count = int(
            handler_state.get("approved_submission_count", 1)
        )
        expected_copy_numbers = list(
            range(1, approved_submission_count + 1)
        )
        validation = dict(saved_gate.get("validation") or {})
        if (
            handler_state.get("schema")
            != "poke_bot.passed_gate_handler/v1"
            or handler_state.get("phase") != "complete_handoff_started"
            or handler_state.get("submission_mode") != "queue_and_continue"
            or approved_submission_count not in (1, 2)
            or not validation
            or not all(value is True for value in validation.values())
            or handler_state.get("frozen_model") != frozen
            or [int(value.get("copy_number", -1)) for value in queued]
            != expected_copy_numbers
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
    if runtime.get("future_guide_weight_policy") is not None:
        runtime_registration["future_guide_weight_policy"] = copy.deepcopy(
            runtime["future_guide_weight_policy"]
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
        state_path = _path(dict(contract["selection"]), "state")
        required_ids = _required_specialist_ids(state_path)
        if completed_ids == required_ids:
            (
                required_refresh_order,
                completed_refresh_ids,
            ) = _validated_post_fleet_refresh_progress(
                state_path=state_path,
                cycle_contract=contract,
            )
            if population_transition_ready(
                completed_ids,
                required_ids,
                completed_refresh_ids=completed_refresh_ids,
                required_refresh_order=required_refresh_order,
            ):
                _start_population_handoff(runtime)
                return 0
            next_refresh = required_refresh_order[len(completed_refresh_ids)]
            raise RuntimeError(
                "post-fleet specialist refresh is pending before population: "
                f"{next_refresh}"
            )
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
