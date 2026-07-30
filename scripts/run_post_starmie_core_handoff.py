#!/usr/bin/env python3
"""Refresh the shared core and continue from Starmie to specialist four."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poke_bot import archetypes, checkpoint
from poke_bot.pure_rl.model_registry import sha256, verify_frozen_model
from scripts.run_sequential_specialist_handoff import (
    load_contract as load_sequential_contract,
    path_value as sequential_path_value,
    save_state as save_sequential_state,
    service_active,
    run as run_sequential_handoff,
    validate_core as validate_sequential_core,
    validate_corpus as validate_sequential_corpus,
    validate_frozen_predecessor_registry,
    validate_source,
)
from scripts.select_next_specialist import select as select_next_specialist
from scripts.resolve_specialist_assets import resolve_specialist_assets
from scripts.run_starmie_expert_bootstrap import (
    current_deck_guide_handoff_contract,
    decision_fusion_handoff_contract,
    expanded_handoff_training_contract,
)


SCHEMA = "poke_bot.post_specialist_core_refresh_handoff/v1"
STATE_SCHEMA = "poke_bot.post_starmie_core_handoff_state/v1"


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


def _path(section: dict[str, Any], key: str) -> Path:
    raw = str(section.get(key) or "").strip()
    if not raw:
        raise RuntimeError(f"contract path missing: {key}")
    return Path(raw).expanduser().resolve()


def _optional_path(section: dict[str, Any], key: str) -> Path | None:
    raw = str(section.get(key) or "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _command(argv: list[str]) -> None:
    result = subprocess.run(argv, check=False)
    if result.returncode:
        raise RuntimeError(
            f"command failed rc={result.returncode}: {' '.join(argv)}"
        )


def _save_state(path: Path, phase: str, **values: Any) -> None:
    current = _read(path) if path.is_file() else {}
    _atomic(
        path,
        {
            **current,
            "schema": STATE_SCHEMA,
            "phase": phase,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **values,
        },
    )


def _late_matchup_v6_migration_is_safe(
    changes: dict[str, Any],
    sequential_state: dict[str, Any],
) -> bool:
    """Allow the runtime-only V6 expansion after an exact frozen bootstrap."""

    if (
        str(sequential_state.get("phase") or "")
        not in {
            "next_specialist_bootstrap_frozen",
            "next_specialist_runtime_checkpoint_frozen",
            "matchup_v6_fleet_activated",
        }
        or "matchup_v6" not in changes
        or not set(changes).issubset(
            {"matchup_v6", "required_target_coverage"}
        )
    ):
        return False
    coverage_change = changes.get("required_target_coverage")
    if coverage_change is None:
        return True
    before_coverage = coverage_change.get("before")
    if before_coverage is not None and list(before_coverage) != []:
        return False
    required = list(coverage_change.get("after") or ())
    bootstrap = dict(sequential_state.get("next_specialist_bootstrap") or {})
    ready_raw = str(bootstrap.get("ready") or "").strip()
    ready_path = Path(ready_raw).expanduser().resolve() if ready_raw else None
    if (
        not required
        or ready_path is None
        or not ready_path.is_file()
        or bootstrap.get("ready_sha256") != sha256(ready_path)
    ):
        return False
    ready = _read(ready_path)
    return (
        ready.get("schema") == "poke_bot.specialist_expert_bootstrap_ready/v1"
        and ready.get("status") == "ready"
        and ready.get("checkpoint_digest")
        == bootstrap.get("checkpoint_digest")
        and list(ready.get("trained_target_coverage") or ()) == required
    )


def _upgrade_selected_handoff_contract(
    payload: dict[str, Any],
    selected: dict[str, Any],
    *,
    current_deck_guide: dict[str, Any] | None = None,
    matchup_v6: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply selection-derived fields before any target bootstrap has begun."""

    specialist_id = str(selected.get("specialist_id") or "")
    specialist = archetypes.ARCHETYPES.get(specialist_id)
    source_id = str((payload.get("source_specialist") or {}).get("id") or "")
    source_archetype = archetypes.ARCHETYPES.get(source_id)
    minimum_decisions = int(selected.get("minimum_decisions") or 0)
    if (
        payload.get("schema")
        != "poke_bot.sequential_specialist_handoff_contract/v1"
        or str((payload.get("next_specialist") or {}).get("id") or "")
        != specialist_id
        or specialist is None
        or source_archetype is None
        or minimum_decisions <= 0
        or int(selected.get("decisions") or 0) < minimum_decisions
    ):
        raise RuntimeError("selected handoff upgrade identity changed")
    upgraded = json.loads(json.dumps(payload))
    training = dict(upgraded.get("training") or {})
    gate_materialization = dict(upgraded.get("gate_materialization") or {})
    source_specialist = dict(upgraded.get("source_specialist") or {})
    runtime_registration = dict(
        upgraded.get("runtime_registration") or {}
    )
    expected_expanded = expanded_handoff_training_contract()
    existing_guide = dict(training.get("current_deck_guide") or {})
    effective_guide = current_deck_guide or existing_guide
    expected_fusion = decision_fusion_handoff_contract(
        strategic_curriculum=(
            isinstance(effective_guide, dict)
            and effective_guide.get("training_mode")
            == "strategic_curriculum_v1"
        )
    )
    if current_deck_guide is not None and existing_guide:
        saved_guide = dict(existing_guide)
        refreshed_guide = dict(current_deck_guide)
        saved_contract_raw = str(saved_guide.pop("contract", "")).strip()
        refreshed_contract_raw = str(
            refreshed_guide.pop("contract", "")
        ).strip()
        expected_guide_digest = str(
            existing_guide.get("contract_sha256") or ""
        )
        saved_contract = (
            Path(saved_contract_raw).expanduser().resolve()
            if saved_contract_raw
            else None
        )
        refreshed_contract = (
            Path(refreshed_contract_raw).expanduser().resolve()
            if refreshed_contract_raw
            else None
        )
        if (
            saved_guide == refreshed_guide
            and expected_guide_digest.startswith("sha256:")
            and saved_contract is not None
            and refreshed_contract is not None
            and saved_contract.is_file()
            and refreshed_contract.is_file()
            and sha256(saved_contract) == expected_guide_digest
            and sha256(refreshed_contract) == expected_guide_digest
        ):
            # A stable deployment pointer may change the absolute repository
            # prefix without changing the checksum-bound guide contract.
            current_deck_guide = existing_guide
    if training.get("expanded_heads") != expected_expanded:
        target = dict(upgraded.get("next_specialist") or {})
        ready_raw = str(target.get("ready") or "").strip()
        run_raw = str(target.get("run_dir") or "").strip()
        run_dir = Path(run_raw).expanduser() if run_raw else None
        bootstrap_started = bool(
            (ready_raw and Path(ready_raw).expanduser().is_file())
            or (
                run_dir is not None
                and (
                    (run_dir / "state.json").is_file()
                    or (
                        (run_dir / "checkpoints").is_dir()
                        and any((run_dir / "checkpoints").iterdir())
                    )
                )
            )
        )
        if bootstrap_started:
            raise RuntimeError(
                "cannot add expanded-head schedule after specialist bootstrap "
                "has started"
            )
    if (
        current_deck_guide is not None
        and training.get("current_deck_guide") != current_deck_guide
    ):
        target = dict(upgraded.get("next_specialist") or {})
        ready_raw = str(target.get("ready") or "").strip()
        run_raw = str(target.get("run_dir") or "").strip()
        run_dir = Path(run_raw).expanduser() if run_raw else None
        bootstrap_started = bool(
            (ready_raw and Path(ready_raw).expanduser().is_file())
            or (
                run_dir is not None
                and (
                    (run_dir / "state.json").is_file()
                    or (
                        (run_dir / "checkpoints").is_dir()
                        and any((run_dir / "checkpoints").iterdir())
                    )
                )
            )
        )
        if bootstrap_started:
            raise RuntimeError(
                "cannot bind current-deck guide after specialist bootstrap "
                "has started"
            )
    pointer = Path(str(selected.get("pointer") or "")).expanduser().resolve()
    required_targets: list[str] | None = None
    if pointer.is_file():
        protected = _read(pointer)
        manifest_path = (
            pointer.parent / str(protected.get("manifest") or "")
        ).resolve()
        manifest = _read(manifest_path)
        coverage = dict(
            (manifest.get("totals") or {}).get("target_coverage") or {}
        )
        required_targets = [
            target
            for target in (
                "temporal_action_rows",
                "opponent_hand_rows",
                "opponent_remainder_rows",
                "opponent_private_prize_rows",
                "lethal_threat_rows",
                "prize_race_rows",
            )
            if int(coverage.get(target, 0))
            == int(selected.get("decisions") or 0)
        ]
        if "temporal_action_rows" not in required_targets:
            raise RuntimeError("selected specialist corpus lacks action targets")
    changes = {
        "minimum_decisions": {
            "before": training.get("minimum_decisions"),
            "after": minimum_decisions,
        },
        "source_archetype_label": {
            "before": gate_materialization.get("archetype_label"),
            "after": source_archetype.name,
        },
        "expanded_heads": {
            "before": training.get("expanded_heads"),
            "after": expected_expanded,
        },
        "decision_fusion": {
            "before": training.get("decision_fusion"),
            "after": expected_fusion,
        },
    }
    if current_deck_guide is not None:
        changes["current_deck_guide"] = {
            "before": training.get("current_deck_guide"),
            "after": current_deck_guide,
        }
    if required_targets is not None:
        changes["required_target_coverage"] = {
            "before": training.get("required_target_coverage"),
            "after": required_targets,
        }
    if matchup_v6 is not None:
        if (
            matchup_v6.get("enabled") is not True
            or str(matchup_v6.get("family_suffix") or "")
            != "_matchup_v6"
            or not all(
                Path(str(matchup_v6.get(key) or "")).expanduser().is_absolute()
                for key in ("registry", "staging_root", "receipt_root")
            )
            or not isinstance(matchup_v6.get("fleet"), dict)
        ):
            raise RuntimeError("selected handoff Matchup V6 contract changed")
        changes["matchup_v6"] = {
            "before": runtime_registration.get("matchup_v6"),
            "after": matchup_v6,
        }
    handler_raw = str(source_specialist.get("handler_state") or "").strip()
    if handler_raw:
        handler_path = Path(handler_raw).expanduser().resolve()
        handler = _read(handler_path)
        transition = str(
            handler.get("threshold_transition_receipt") or ""
        ).strip()
        transition_digest = str(
            handler.get("threshold_transition_receipt_sha256") or ""
        ).strip()
        if transition:
            transition_path = Path(transition).expanduser().resolve()
            if (
                not transition_path.is_file()
                or not transition_digest
                or sha256(transition_path) != transition_digest
            ):
                raise RuntimeError(
                    "selected handoff threshold-transition identity changed"
                )
            changes["threshold_transition_receipt"] = {
                "before": source_specialist.get(
                    "threshold_transition_receipt"
                ),
                "after": transition,
            }
            source_specialist["threshold_transition_receipt"] = transition
    training["minimum_decisions"] = minimum_decisions
    training["expanded_heads"] = changes["expanded_heads"]["after"]
    training["decision_fusion"] = changes["decision_fusion"]["after"]
    if current_deck_guide is not None:
        training["current_deck_guide"] = current_deck_guide
    if required_targets is not None:
        training["required_target_coverage"] = required_targets
    if matchup_v6 is not None:
        runtime_registration["matchup_v6"] = matchup_v6
    gate_materialization["archetype_label"] = source_archetype.name
    upgraded["training"] = training
    upgraded["gate_materialization"] = gate_materialization
    upgraded["source_specialist"] = source_specialist
    upgraded["runtime_registration"] = runtime_registration
    return upgraded, {
        key: value
        for key, value in changes.items()
        if value["before"] != value["after"]
    }


def _reusable_core_candidate(
    *,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return an immutable existing core candidate without rematerializing it."""

    core = dict(contract["core_refresh"])
    ready_path = _path(core, "ready_receipt")
    family_path = _path(core, "family")
    if not ready_path.is_file() or not family_path.is_dir():
        return None
    ready = _read(ready_path)
    frozen = verify_frozen_model(family_path)
    expected_teacher_digests = [
        str(row["checksum"]) for row in core.get("teachers", ())
    ]
    expanded = dict(core.get("expanded_heads") or {})
    fallback_enabled = bool(
        (contract.get("core_failure_fallback") or {}).get("enabled")
    )
    if (
        expanded
        and expanded != expanded_handoff_training_contract()
        and not fallback_enabled
    ):
        raise RuntimeError("cumulative-core expanded-head contract changed")
    status = str(ready.get("status") or "")
    if (
        ready.get("schema") != "poke_bot.multi_teacher_core_ready/v1"
        or status
        not in {"candidate_ready_for_gameplay_regression", "ready"}
        or (
            status == "ready"
            and ready.get("gameplay_regression_passed") is not True
        )
        or ready.get("checkpoint_digest") != frozen.get("checkpoint_digest")
        or ready.get("teacher_checkpoint_digests")
        != expected_teacher_digests
        or (
            bool(expanded)
            and (
                ready.get("expanded_target_schema_digest")
                != expanded["target_schema_digest"]
                or ready.get("expanded_schedule_digest")
                != expanded["schedule_digest"]
                or set(ready.get("expanded_heads_trained") or ())
                != set(
                    expanded["schedule"]["stages"][-1][
                        "enabled_heads"
                    ]
                )
                or ready.get("runtime_enabled_heads") != []
                or int(ready.get("epochs_completed") or 0)
                != int(expanded["schedule"]["total_epochs"])
            )
        )
    ):
        raise RuntimeError("existing refreshed-core readiness identity changed")
    return ready, frozen


def _reusable_core_regression(
    *,
    path: Path,
    candidate_digest: str,
    teacher_digests: list[str],
) -> dict[str, Any] | None:
    """Reuse an immutable regression after controller-only changes.

    A valid rejection is also authoritative: the non-blocking fallback policy
    must consume it instead of rerunning evaluation or retraining the rejected
    boundary candidate.
    """

    if not path.is_file():
        return None
    result = _read(path)
    identity = dict(result.get("identity") or {})
    candidate = dict(identity.get("candidate") or {})
    if (
        result.get("schema")
        != "poke_bot.multi_teacher_core_gameplay_regression/v1"
        or result.get("passed") not in {True, False}
        or result.get("training_eligible") is not False
        or candidate.get("digest") != candidate_digest
        or identity.get("teacher_checkpoint_digests") != teacher_digests
    ):
        return None
    return result


def _validated_nonblocking_fallback_core(
    *,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Resolve the latest accepted core after a rejected refresh candidate."""

    policy = dict(contract.get("core_failure_fallback") or {})
    if not policy:
        return None
    if (
        policy.get("enabled") is not True
        or policy.get("behavior")
        != "continue_with_latest_accepted_core"
        or policy.get("continue_refresh_after_each_specialist") is not True
    ):
        raise RuntimeError("nonblocking core-fallback policy changed")
    family = _path(policy, "family")
    ready_path = _path(policy, "ready_receipt")
    ready = _read(ready_path)
    frozen = verify_frozen_model(family)
    expected_digest = str(policy.get("checkpoint_digest") or "")
    version = int(policy.get("version") or 0)
    if (
        version <= 0
        or ready.get("schema") != "poke_bot.multi_teacher_core_ready/v1"
        or ready.get("status") != "ready"
        or ready.get("gameplay_regression_passed") is not True
        or ready.get("checkpoint_digest") != expected_digest
        or frozen.get("checkpoint_digest") != expected_digest
    ):
        raise RuntimeError("latest accepted fallback core identity changed")
    return ready, frozen, {
        **dict(contract["core_refresh"]),
        "version": version,
        "family": str(family),
        "ready_receipt": str(ready_path),
    }


def _resolve_boundary_core(
    *,
    contract_path: Path,
    contract: dict[str, Any],
    runtime: dict[str, Any],
    state_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Attempt one cumulative refresh without blocking specialist production."""

    attempted_core = dict(contract["core_refresh"])
    attempted_ready_path = _path(attempted_core, "ready_receipt")
    contract_digest = sha256(contract_path)
    if attempted_ready_path.is_file():
        prior_rejection = _read(attempted_ready_path)
        if (
            prior_rejection.get("schema")
            == "poke_bot.multi_teacher_core_refresh_rejection/v1"
            and prior_rejection.get("status")
            == "rejected_pretraining_validation"
            and prior_rejection.get("contract_sha256") == contract_digest
            and prior_rejection.get("training_eligible") is False
            and prior_rejection.get("candidate_checkpoint_created") is False
        ):
            fallback = _validated_nonblocking_fallback_core(contract=contract)
            if fallback is None:
                raise RuntimeError(
                    "pretraining-rejected core lacks a validated fallback"
                )
            ready, frozen, core = fallback
            updated_contract = {**contract, "core_refresh": core}
            _save_state(
                state_path,
                "core_pretraining_validation_failed_fallback_selected",
                rejected_core_refresh=prior_rejection,
                fallback_core={
                    "version": int(core["version"]),
                    "family": str(core["family"]),
                    "checkpoint_digest": str(frozen["checkpoint_digest"]),
                    "ready_receipt": str(core["ready_receipt"]),
                },
                production_continues=True,
            )
            return ready, frozen, core, updated_contract

    reusable = _reusable_core_candidate(contract=contract)
    if reusable is None:
        try:
            _command(_core_refresh_command(contract=contract, runtime=runtime))
        except RuntimeError as exc:
            fallback = _validated_nonblocking_fallback_core(contract=contract)
            if fallback is None:
                raise
            ready, frozen, core = fallback
            rejection = {
                "schema": "poke_bot.multi_teacher_core_refresh_rejection/v1",
                "status": "rejected_pretraining_validation",
                "contract": str(contract_path),
                "contract_sha256": contract_digest,
                "core_version": int(attempted_core.get("version") or 0),
                "family": str(attempted_core.get("family") or ""),
                "initialization": dict(
                    attempted_core.get("initialization") or {}
                ),
                "teacher_checkpoint_digests": [
                    str(row.get("checksum") or "")
                    for row in attempted_core.get("teachers") or ()
                ],
                "failure": str(exc),
                "training_eligible": False,
                "gameplay_regression_run": False,
                "candidate_checkpoint_created": False,
                "immutable": True,
                "fallback_core": {
                    "version": int(core["version"]),
                    "family": str(core["family"]),
                    "checkpoint_digest": str(frozen["checkpoint_digest"]),
                    "ready_receipt": str(core["ready_receipt"]),
                },
                "owner_decision": "GOAL.md#/decision-ledger/revision-19",
            }
            if attempted_ready_path.is_file():
                if _read(attempted_ready_path) != rejection:
                    raise RuntimeError(
                        "existing pretraining core rejection differs"
                    )
            else:
                _atomic(attempted_ready_path, rejection)
            updated_contract = {**contract, "core_refresh": core}
            _save_state(
                state_path,
                "core_pretraining_validation_failed_fallback_selected",
                rejected_core_refresh=rejection,
                fallback_core=rejection["fallback_core"],
                production_continues=True,
            )
            return ready, frozen, core, updated_contract
        ready = _read(attempted_ready_path)
        frozen = verify_frozen_model(_path(attempted_core, "family"))
    else:
        ready, frozen = reusable

    ready_status = str(ready.get("status") or "")
    resumable_ready = (
        ready_status == "candidate_ready_for_gameplay_regression"
        or (
            ready_status == "ready"
            and ready.get("gameplay_regression_passed") is True
        )
    )
    if (
        not resumable_ready
        or ready.get("checkpoint_digest") != frozen.get("checkpoint_digest")
    ):
        raise RuntimeError("refreshed core candidate identity changed")
    _save_state(state_path, "core_candidate_ready", core_candidate=ready)

    acceptance = dict(contract["acceptance"])
    regression_path = _path(acceptance, "regression_result")
    expected_teacher_digests = [
        str(row["checksum"]) for row in attempted_core.get("teachers", ())
    ]
    regression = _reusable_core_regression(
        path=regression_path,
        candidate_digest=str(frozen["checkpoint_digest"]),
        teacher_digests=expected_teacher_digests,
    )
    if regression is None:
        _save_state(
            state_path,
            "core_gameplay_regression_running",
            core_candidate=ready,
            core_gameplay_regression={
                "status": "running",
                "candidate_digest": str(frozen["checkpoint_digest"]),
                "teachers_total": len(expected_teacher_digests),
                "games_per_teacher": int(acceptance["games_per_teacher"]),
            },
        )
        result = subprocess.run(
            [
                str(runtime["python"]),
                "-u",
                str(
                    Path(__file__).resolve().parent
                    / "run_core_teacher_regression.py"
                ),
                "--contract",
                str(contract_path),
                "--candidate",
                str(frozen["model_path"]),
                "--output",
                str(regression_path),
            ],
            check=False,
        )
        regression = (
            _read(regression_path) if regression_path.is_file() else {}
        )
        regression_returncode = result.returncode
    else:
        regression_returncode = 0
    _save_state(
        state_path,
        "core_gameplay_regression_complete",
        core_gameplay_regression=regression,
    )
    if regression_returncode or regression.get("passed") is not True:
        fallback = _validated_nonblocking_fallback_core(contract=contract)
        if fallback is None:
            raise RuntimeError(
                "refreshed core failed established gameplay regression"
            )
        ready, frozen, core = fallback
        updated_contract = {**contract, "core_refresh": core}
        _save_state(
            state_path,
            "core_gameplay_regression_failed_fallback_selected",
            rejected_core_gameplay_regression=regression,
            fallback_core={
                "version": int(core["version"]),
                "family": str(core["family"]),
                "checkpoint_digest": str(frozen["checkpoint_digest"]),
                "ready_receipt": str(core["ready_receipt"]),
            },
            production_continues=True,
        )
        return ready, frozen, core, updated_contract

    ready["gameplay_regression_passed"] = True
    ready["gameplay_regression_result"] = str(regression_path)
    ready["status"] = "ready"
    _atomic(attempted_ready_path, ready)
    return ready, frozen, attempted_core, contract


def _resume_selected_handoff(
    state_path: Path,
    contract: dict[str, Any],
) -> bool:
    if not state_path.is_file():
        return False
    state = _read(state_path)
    phase = str(state.get("phase") or "")
    if phase == "next_specialist_started":
        return True
    if phase != "next_specialist_selected":
        return False
    generated_raw = str(state.get("generated_handoff_contract") or "").strip()
    expected_digest = str(
        state.get("generated_handoff_contract_sha256") or ""
    ).strip()
    selection = dict(state.get("selection") or {})
    selected = dict(selection.get("selected") or {})
    generated = Path(generated_raw).expanduser().resolve()
    if (
        not generated_raw
        or not generated.is_file()
        or not expected_digest
        or sha256(generated) != expected_digest
        or not str(selected.get("specialist_id") or "")
    ):
        raise RuntimeError("selected specialist resume identity changed")
    payload = _read(generated)
    expected_assets = str(
        (contract.get("runtime") or {}).get("future_assets_receipt") or ""
    ).strip()
    saved_assets = str(
        (payload.get("asset_generation") or {}).get("promotion_receipt") or ""
    ).strip()
    if expected_assets and Path(saved_assets).expanduser().resolve() != Path(
        expected_assets
    ).expanduser().resolve():
        # Continue through normal selection so the controller can validate and
        # receipt-bind the newer boundary router before resuming this bootstrap.
        return False
    guide_handoff = None
    if bool(
        dict(contract["next_specialist"]).get(
            "current_deck_guide_required", False
        )
    ):
        runtime = dict(contract["runtime"])
        prestage = _validated_prestage(
            contract=contract,
            selected=selected,
            assets={
                "candidate_tree": _optional_path(
                    runtime, "inactive_tree_candidate"
                ),
                "candidate_audit": _optional_path(
                    runtime, "candidate_audit"
                ),
            },
        )
        if prestage is None:
            raise RuntimeError(
                "required current-deck guide pre-stage receipt is not ready"
            )
        guide = dict(prestage.get("current_deck_guide") or {})
        if (
            guide.get("status") != "ready"
            or guide.get("specialist_id")
            != str(selected["specialist_id"])
            or guide.get("implementation_ready") is not True
            or guide.get("corpus_binding_ready") is not True
            or guide.get("targets_ready") is not True
        ):
            raise RuntimeError(
                "pre-staged current-deck guide is not handoff-ready"
            )
        guide_handoff = current_deck_guide_handoff_contract(
            specialist_id=str(selected["specialist_id"]),
            contract_path=Path(str(guide["path"])),
            expected_contract_sha256=str(guide["sha256"]),
            guide_version=str(guide["guide_version"]),
            corpus_ready_receipt=Path(
                str(guide["corpus_ready_receipt"])
            ),
            expected_corpus_ready_sha256=str(
                guide["corpus_ready_receipt_sha256"]
            ),
            strategic_curriculum_spec=(
                Path(str((guide.get("strategic_curriculum") or {}).get(
                    "curriculum_spec"
                )))
                if guide.get("strategic_curriculum") is not None
                else None
            ),
            expected_strategic_curriculum_spec_sha256=str(
                (guide.get("strategic_curriculum") or {}).get(
                    "curriculum_spec_sha256"
                )
                or ""
            ),
            strategic_head_role_map=(
                Path(str((guide.get("strategic_curriculum") or {}).get(
                    "head_role_map"
                )))
                if guide.get("strategic_curriculum") is not None
                else None
            ),
            expected_strategic_head_role_map_sha256=str(
                (guide.get("strategic_curriculum") or {}).get(
                    "head_role_map_sha256"
                )
                or ""
            ),
            strategic_validation_receipt=(
                Path(str((guide.get("strategic_curriculum") or {}).get(
                    "validation_receipt"
                )))
                if guide.get("strategic_curriculum") is not None
                else None
            ),
            expected_strategic_validation_receipt_sha256=str(
                (guide.get("strategic_curriculum") or {}).get(
                    "validation_receipt_sha256"
                )
                or ""
            ),
            protocol_path=Path(__file__).resolve().parents[1]
            / "config/rl_protocol.yaml",
        )
    # Recover a controller crash that occurred after the immutable selected
    # contract was upgraded but before its source-only preflight state could be
    # rebound.  No bootstrap/model mutation has happened at this phase, and all
    # three immutable inputs must still validate exactly under the new contract.
    sequential_contract, sequential_digest = load_sequential_contract(generated)
    sequential_state_path = sequential_path_value(
        sequential_contract, "paths", "state"
    )
    if sequential_state_path.is_file():
        sequential_state = _read(sequential_state_path)
        if sequential_state.get("contract_sha256") != sequential_digest:
            source_contract = dict(
                sequential_contract.get("source_specialist") or {}
            )
            saved_source = dict(
                sequential_state.get("source_specialist") or {}
            )
            frozen_source = verify_frozen_model(
                Path(str(source_contract.get("passed_family") or ""))
                .expanduser()
                .resolve()
            )
            handler = _read(
                Path(str(source_contract.get("handler_state") or ""))
                .expanduser()
                .resolve()
            )
            handler_frozen = dict(handler.get("frozen_model") or {})
            immutable_source_matches = (
                saved_source.get("specialist_id")
                == source_contract.get("id")
                and Path(str(saved_source.get("frozen_family") or ""))
                .expanduser()
                .resolve()
                == Path(str(source_contract.get("passed_family") or ""))
                .expanduser()
                .resolve()
                and saved_source.get("checkpoint_digest")
                == frozen_source.get("checkpoint_digest")
                == handler_frozen.get("checkpoint_digest")
                and (saved_source.get("gate") or {}).get("contract")
                == source_contract.get("gate_contract")
            )
            if (
                sequential_state.get("schema")
                != "poke_bot.sequential_specialist_handoff_state/v1"
                or sequential_state.get("phase") != "source_preflight_verified"
                or not immutable_source_matches
                or sequential_state.get("shared_deck_agnostic_core")
                != validate_sequential_core(sequential_contract)
                or sequential_state.get("next_specialist_corpus")
                != validate_sequential_corpus(sequential_contract)
            ):
                raise RuntimeError(
                    "selected handoff state cannot be rebound safely"
                )
            _atomic(
                sequential_state_path,
                {
                    **sequential_state,
                    "contract_sha256": sequential_digest,
                    "contract_migration": {
                        "schema":
                            "poke_bot.selected_handoff_crash_recovery/v1",
                        "new_contract_sha256": sequential_digest,
                        "bootstrap_started_before_migration": False,
                    },
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
    upgraded, changes = _upgrade_selected_handoff_contract(
        payload,
        selected,
        current_deck_guide=guide_handoff,
        matchup_v6=dict(
            (contract.get("runtime") or {}).get("matchup_v6") or {}
        )
        or None,
    )
    if changes:
        old_digest = expected_digest
        old_sequential_contract, old_sequential_digest = (
            load_sequential_contract(generated)
        )
        sequential_state_path = sequential_path_value(
            old_sequential_contract, "paths", "state"
        )
        sequential_state = (
            _read(sequential_state_path)
            if sequential_state_path.is_file()
            else {}
        )
        sequential_phase = str(sequential_state.get("phase") or "")
        late_matchup_v6_only = _late_matchup_v6_migration_is_safe(
            changes,
            sequential_state,
        )
        if sequential_state and (
            sequential_state.get("schema")
            != "poke_bot.sequential_specialist_handoff_state/v1"
            or sequential_state.get("contract_sha256")
            != old_sequential_digest
            or (
                sequential_phase
                not in {"source_preflight_verified", "preflight_verified"}
                and not late_matchup_v6_only
            )
        ):
            raise RuntimeError(
                "selected handoff contract cannot migrate active bootstrap state"
            )
        _atomic(generated, upgraded)
        expected_digest = sha256(generated)
        _, new_sequential_digest = load_sequential_contract(generated)
        if sequential_state:
            _atomic(
                sequential_state_path,
                {
                    **sequential_state,
                    "contract_sha256": new_sequential_digest,
                    "contract_migration": {
                        "schema":
                            "poke_bot.selected_handoff_contract_migration/v1",
                        "old_contract_sha256": old_sequential_digest,
                        "new_contract_sha256": new_sequential_digest,
                        "changes": changes,
                        "bootstrap_started_before_migration":
                            late_matchup_v6_only,
                    },
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        _save_state(
            state_path,
            "next_specialist_selected",
            generated_handoff_contract=str(generated),
            generated_handoff_contract_sha256=expected_digest,
            generated_handoff_contract_migration={
                "schema": "poke_bot.selected_handoff_contract_migration/v1",
                "old_digest": old_digest,
                "new_digest": expected_digest,
                "changes": changes,
                "bootstrap_started_before_migration":
                    late_matchup_v6_only,
            },
        )
    sequential_contract, sequential_digest = load_sequential_contract(
        generated
    )
    sequential_state = sequential_path_value(
        sequential_contract, "paths", "state"
    )
    if not sequential_state.is_file():
        saved_source = dict(state.get("source") or {})
        if not saved_source:
            raise RuntimeError(
                "selected specialist resume lost immutable source evidence"
            )
        save_sequential_state(
            sequential_contract,
            "source_preflight_verified",
            sequential_digest,
            source_specialist=saved_source,
            shared_deck_agnostic_core=validate_sequential_core(
                sequential_contract
            ),
            next_specialist_corpus=validate_sequential_corpus(
                sequential_contract
            ),
        )
    run_sequential_handoff(generated)
    _save_state(
        state_path,
        "next_specialist_started",
        selected_specialist=selected,
    )
    return True


def _compatible_selected_asset_upgrade(
    existing: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    """Permit only a boundary router generation change after bootstrap."""

    if existing.get("schema") != current.get("schema"):
        return False
    old = json.loads(json.dumps(existing))
    new = json.loads(json.dumps(current))
    old["asset_generation"] = {}
    new["asset_generation"] = {}
    for payload in (old, new):
        source = dict(payload.get("source_specialist") or {})
        source.pop("gate_contract", None)
        payload["source_specialist"] = source
        registration = dict(payload.get("runtime_registration") or {})
        registration.pop("inactive_tree_candidate", None)
        registration.pop("candidate_audit", None)
        payload["runtime_registration"] = registration
    return old == new


def _migrate_selected_asset_contract(
    *,
    generated_path: Path,
    existing: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """Rebind a resumable handoff state to a validated router-only upgrade."""

    if not _compatible_selected_asset_upgrade(existing, current):
        raise RuntimeError("generated next-specialist handoff identity changed")
    old_digest = sha256(generated_path)
    state_path = _path(dict(existing["paths"]), "state")
    state = _read(state_path)
    allowed = {
        "preflight_verified",
        "next_specialist_bootstrap_frozen",
        "next_specialist_runtime_checkpoint_frozen",
        "matchup_v6_fleet_activated",
    }
    if (
        state.get("schema")
        != "poke_bot.sequential_specialist_handoff_state/v1"
        or state.get("contract_sha256") != old_digest
        or str(state.get("phase") or "") not in allowed
    ):
        raise RuntimeError(
            "router upgrade cannot rebind the current sequential handoff phase"
        )
    _atomic(generated_path, current)
    new_digest = sha256(generated_path)
    migration = {
        "schema": "poke_bot.selected_handoff_router_upgrade/v1",
        "old_contract_sha256": old_digest,
        "new_contract_sha256": new_digest,
        "phase_preserved": state["phase"],
        "bootstrap_checkpoint_preserved": (
            (state.get("next_specialist_bootstrap") or {}).get(
                "checkpoint_digest"
            )
        ),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic(
        state_path,
        {
            **state,
            "contract_sha256": new_digest,
            "router_contract_migration": migration,
            "updated_at_utc": migration["updated_at_utc"],
        },
    )
    return migration


def _source_contract(contract: dict[str, Any]) -> dict[str, Any]:
    trigger = dict(contract["trigger"])
    source = {
        "id": trigger["specialist_id"],
        "run_dir": trigger["run_dir"],
        "training_service": trigger["training_service"],
        "gate_contract": trigger["gate_contract"],
        "gate_marker_name": trigger["gate_marker_name"],
        "minimum_completed_iteration": trigger[
            "minimum_completed_iteration"
        ],
        "passed_family": trigger["passed_family"],
        "handler_state": trigger["handler_state"],
    }
    transition_receipt = str(
        trigger.get("threshold_transition_receipt") or ""
    ).strip()
    if transition_receipt:
        source["threshold_transition_receipt"] = transition_receipt
    return {
        "source_specialist": {
            **source,
        }
    }


def _core_refresh_command(
    *,
    contract: dict[str, Any],
    runtime: dict[str, Any],
) -> list[str]:
    core = dict(contract["core_refresh"])
    initialization = dict(core["initialization"])
    teachers = list(core["teachers"])
    command = [
        str(runtime["python"]),
        "-u",
        str(Path(__file__).resolve().parent / "run_multi_teacher_core_refresh.py"),
        "--core-corpus",
        str(_path(dict(core["balanced_corpus"]), "pointer")),
        "--initialization-family",
        str(_path(initialization, "checkpoint").parent),
    ]
    for teacher in teachers:
        command.extend(
            ["--teacher-family", str(_path(dict(teacher), "checkpoint").parent)]
        )
    command.extend(
        [
            "--registry-root",
            str(_path(runtime, "registry_root")),
            "--output-family",
            _path(core, "family").name,
            "--ready",
            str(_path(core, "ready_receipt")),
            "--run-name",
            str(core["run_name"]),
            "--run-dir",
            str(_path(core, "run_dir")),
            "--epochs",
            str(int(core["max_epochs"])),
            "--patience",
            str(int(core["early_stop_patience"])),
            "--min-delta",
            str(float(core["early_stop_min_delta"])),
            "--min-decisions",
            str(int(core["minimum_decisions"])),
            "--batch-size",
            str(int(core["requested_decisions_per_batch"])),
            "--split-seed",
            str(int(core.get("split_seed", 20260723))),
            "--cpu-pack-root",
            str(_path(core, "cpu_pack_root")),
        ]
    )
    expanded = dict(core.get("expanded_heads") or {})
    expected_expanded = expanded_handoff_training_contract()
    fusion = dict(core.get("decision_fusion") or {})
    expected_fusion = decision_fusion_handoff_contract()
    if expanded:
        if expanded != expected_expanded:
            raise RuntimeError(
                "cumulative-core expanded-head contract changed"
            )
        if fusion != expected_fusion:
            raise RuntimeError(
                "cumulative-core decision-fusion contract changed"
            )
        command.extend(
            [
                "--expanded-heads",
                "--decision-fusion",
                "--rl-protocol",
                str(
                    Path(__file__).resolve().parents[1]
                    / "config/rl_protocol.yaml"
                ),
                "--expected-expanded-schedule-digest",
                str(expanded["schedule_digest"]),
                "--expected-expanded-target-digest",
                str(expanded["target_schema_digest"]),
            ]
        )
    teacher_behavior = dict(
        core.get("teacher_behavior_distillation") or {}
    )
    if teacher_behavior:
        if (
            teacher_behavior.get("schema")
            != "poke_bot.teacher_behavior_distillation/v1"
            or teacher_behavior.get("enabled") is not True
            or teacher_behavior.get("target")
            != "matching_archetype_frozen_teacher_greedy_action"
            or teacher_behavior.get("causal_inputs_only") is not True
            or float(teacher_behavior.get("loss_weight") or 0.0) <= 0.0
        ):
            raise RuntimeError(
                "cumulative-core teacher behavior contract changed"
            )
        command.extend(
            [
                "--teacher-behavior-distillation",
                "--teacher-policy-weight",
                str(float(teacher_behavior["loss_weight"])),
            ]
        )
    return command


def _generated_contract(
    *,
    contract: dict[str, Any],
    source: dict[str, Any],
    selected: dict[str, Any],
    core_digest: str,
    assets: dict[str, Any] | None = None,
    prestage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trigger = dict(contract["trigger"])
    core = dict(contract["core_refresh"])
    gate = dict(contract["gate_materialization"])
    runtime = dict(contract["runtime"])
    if assets is None:
        candidate_tree = _optional_path(runtime, "inactive_tree_candidate")
        candidate_audit = _optional_path(runtime, "candidate_audit")
        assets = {
            "source": "contract_defaults",
            "corpus_root": _path(dict(contract["next_specialist"]), "corpus_root"),
            "candidate_tree": candidate_tree,
            "candidate_audit": candidate_audit,
            "promotion_receipt": None,
        }
    specialist_id = str(selected["specialist_id"])
    specialist = archetypes.ARCHETYPES.get(specialist_id)
    source_id = str(source.get("specialist_id") or trigger["specialist_id"])
    source_archetype = archetypes.ARCHETYPES.get(source_id)
    if specialist is None:
        raise RuntimeError(
            f"selected specialist is not registered: {specialist_id}"
        )
    if source_archetype is None:
        raise RuntimeError(
            f"source specialist is not registered: {source_id}"
        )
    core_version = int(core["version"])
    bootstrap_run = (
        f"{specialist_id}_expert_bootstrap_from_core_v{core_version}_20260723"
    )
    rl_run = f"pure_rl_{specialist_id}_temporal1_8k_v1_20260723"
    state_root = _path(runtime, "state_root")
    staged_pack_root = str(
        ((prestage or {}).get("cpu_pack") or {}).get("root") or ""
    ).strip()
    guide_handoff = None
    guide_required = bool(
        dict(contract["next_specialist"]).get(
            "current_deck_guide_required", False
        )
    )
    if guide_required and prestage is None:
        raise RuntimeError(
            "required current-deck guide pre-stage receipt is not ready"
        )
    if prestage is not None:
        guide = dict(prestage.get("current_deck_guide") or {})
        if (
            guide.get("status") != "ready"
            or guide.get("specialist_id") != specialist_id
            or guide.get("implementation_ready") is not True
            or guide.get("corpus_binding_ready") is not True
            or guide.get("targets_ready") is not True
        ):
            raise RuntimeError(
                "pre-staged current-deck guide is not handoff-ready"
            )
        guide_handoff = current_deck_guide_handoff_contract(
            specialist_id=specialist_id,
            contract_path=Path(str(guide["path"])),
            expected_contract_sha256=str(guide["sha256"]),
            guide_version=str(guide["guide_version"]),
            corpus_ready_receipt=Path(str(guide["corpus_ready_receipt"])),
            expected_corpus_ready_sha256=str(
                guide["corpus_ready_receipt_sha256"]
            ),
            strategic_curriculum_spec=(
                Path(str((guide.get("strategic_curriculum") or {}).get(
                    "curriculum_spec"
                )))
                if guide.get("strategic_curriculum") is not None
                else None
            ),
            expected_strategic_curriculum_spec_sha256=str(
                (guide.get("strategic_curriculum") or {}).get(
                    "curriculum_spec_sha256"
                )
                or ""
            ),
            strategic_head_role_map=(
                Path(str((guide.get("strategic_curriculum") or {}).get(
                    "head_role_map"
                )))
                if guide.get("strategic_curriculum") is not None
                else None
            ),
            expected_strategic_head_role_map_sha256=str(
                (guide.get("strategic_curriculum") or {}).get(
                    "head_role_map_sha256"
                )
                or ""
            ),
            strategic_validation_receipt=(
                Path(str((guide.get("strategic_curriculum") or {}).get(
                    "validation_receipt"
                )))
                if guide.get("strategic_curriculum") is not None
                else None
            ),
            expected_strategic_validation_receipt_sha256=str(
                (guide.get("strategic_curriculum") or {}).get(
                    "validation_receipt_sha256"
                )
                or ""
            ),
            protocol_path=Path(__file__).resolve().parents[1]
            / "config/rl_protocol.yaml",
        )
    runtime_registration = {
        "runtime_tree": runtime["runtime_tree"],
        "runtime_registry": runtime["runtime_registry"],
        "selector_env": runtime["selector_env"],
        "state_root": runtime["state_root"],
        "run_name": rl_run,
        "handoff_service": runtime["next_handoff_service"],
        "gate_handler_service": runtime["gate_handler_service"],
    }
    if runtime.get("matchup_v6") is not None:
        runtime_registration["matchup_v6"] = json.loads(
            json.dumps(runtime["matchup_v6"])
        )
    if runtime.get("inactive_tree_candidate"):
        runtime_registration.update(
            {
                "inactive_tree_candidate": str(assets["candidate_tree"]),
                "candidate_audit": str(assets["candidate_audit"]),
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
    generated = {
        "schema": "poke_bot.sequential_specialist_handoff_contract/v1",
        "source_specialist": {
            "id": trigger["specialist_id"],
            "run_dir": trigger["run_dir"],
            "training_service": trigger["training_service"],
            "gate_contract": trigger["gate_contract"],
            "gate_marker_name": trigger["gate_marker_name"],
            "minimum_completed_iteration": int(
                trigger["minimum_completed_iteration"]
            ),
            "matchup_runtime_tree": str(_path(runtime, "runtime_tree")),
            "passed_family": trigger["passed_family"],
            "handler_state": trigger["handler_state"],
        },
        "shared_core": {
            "family": str(_path(core, "family")),
            "checkpoint_checksum": core_digest,
        },
        "next_specialist": {
            "id": specialist_id,
            "expert_corpus": selected["pointer"],
            "family_name": (
                f"{specialist_id}_expert_bootstrap_from_core_v{core_version}"
            ),
            "ready": str(
                state_root
                / f"{specialist_id}-expert-bootstrap-ready-v{core_version}.json"
            ),
            "run_name": bootstrap_run,
            "run_dir": (
                "/home/inzi/poke-bot-agent/outputs/bootstrap/"
                f"{specialist_id}-expert-bootstrap-from-core-v{core_version}"
            ),
            "cpu_pack_root": (
                staged_pack_root
                or (
                    "/home/inzi/poke-bot-agent/outputs/bootstrap/cpu-packs/"
                    f"{specialist_id}-expert-bootstrap-v{core_version}"
                )
            ),
            "activation_receipt": str(
                state_root
                / f"{specialist_id}-specialist-rl-activation-v{core_version}.json"
            ),
            "training_service": runtime["training_service"],
            "gate_contract": gate["base_gate_contract"],
            "frozen_specialist_registry": gate[
                "base_frozen_specialist_registry"
            ],
            "current_deck_guide_required": guide_required,
        },
        "gate_materialization": {
            **gate,
            "archetype_label": source_archetype.name,
        },
        "training": {
            "supervised_epochs": 25,
            "patience_diagnostic_only": 5,
            "minimum_decisions": int(
                selected.get(
                    "minimum_decisions",
                    contract["next_specialist"]["minimum_decisions"],
                )
            ),
            "requested_decisions_per_batch": int(
                core["requested_decisions_per_batch"]
            ),
            "expanded_heads": expanded_handoff_training_contract(),
            "decision_fusion": decision_fusion_handoff_contract(
                strategic_curriculum=(
                    guide_handoff is not None
                    and guide_handoff.get("training_mode")
                    == "strategic_curriculum_v1"
                )
            ),
            **(
                {"current_deck_guide": guide_handoff}
                if guide_handoff is not None
                else {}
            ),
        },
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
                state_root
                / f"post-{trigger['specialist_id']}-{specialist_id}-handoff-v1.json"
            ),
            "lock": (
                "/home/inzi/.local/state/pokebot/"
                f"post-{trigger['specialist_id']}-{specialist_id}-handoff-v1.lock"
            ),
        },
    }
    transition_receipt = str(
        source.get("threshold_transition_receipt")
        or trigger.get("threshold_transition_receipt")
        or ""
    ).strip()
    if transition_receipt:
        generated["source_specialist"][
            "threshold_transition_receipt"
        ] = transition_receipt
    completion_authority = str(
        source.get("completion_authority")
        or trigger.get("completion_authority")
        or ""
    ).strip()
    if completion_authority:
        generated["source_specialist"][
            "completion_authority"
        ] = completion_authority
    generated["asset_generation"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in assets.items()
    }
    if prestage:
        generated["prestage"] = {
            "receipt": str(
                (contract.get("next_specialist") or {}).get(
                    "prestage_receipt"
                )
            ),
            "receipt_sha256": sha256(
                _path(dict(contract["next_specialist"]), "prestage_receipt")
            ),
            "selected_specialist": specialist_id,
            "expert_cpu_pack_reused": True,
            "live_training_modified": False,
        }
    return generated


def _validated_prestage(
    *,
    contract: dict[str, Any],
    selected: dict[str, Any],
    assets: dict[str, Any],
) -> dict[str, Any] | None:
    """Reuse a staged CPU pack only when every immutable input still matches."""

    next_config = dict(contract["next_specialist"])
    raw = str(next_config.get("prestage_receipt") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        return None
    receipt = _read(path)
    corpus = dict(receipt.get("expert_corpus") or {})
    runtime = dict(receipt.get("runtime_assets") or {})
    representative = dict(receipt.get("representative") or {})
    pack = dict(receipt.get("cpu_pack") or {})
    payload = Path(str(pack.get("payload") or ""))
    manifest = Path(str(pack.get("manifest") or ""))
    pointer = Path(str(selected["pointer"])).resolve()
    candidate_tree = assets.get("candidate_tree")
    candidate_audit = assets.get("candidate_audit")
    runtime_assets_match = (
        True
        if candidate_tree is None and candidate_audit is None
        else (
            candidate_tree is not None
            and candidate_audit is not None
            and runtime.get("candidate_tree_sha256")
            == sha256(Path(candidate_tree))
            and runtime.get("candidate_audit_sha256")
            == sha256(Path(candidate_audit))
            and runtime.get("selected_route_accepted") is True
        )
    )
    if (
        receipt.get("schema") != "poke_bot.next_specialist_prestage/v1"
        or receipt.get("status") != "ready"
        or receipt.get("live_training_modified") is not False
        or receipt.get("selected_specialist") != selected["specialist_id"]
        or receipt.get("active_specialist")
        != str(contract["trigger"]["specialist_id"])
        or corpus.get("pointer") != str(pointer)
        or corpus.get("pointer_sha256") != sha256(pointer)
        or not runtime_assets_match
        or representative.get("ready") is not True
        or pack.get("status") != "ready"
        or not str(pack.get("root") or "").strip()
        or not payload.is_file()
        or not manifest.is_file()
        or pack.get("payload_sha256") != sha256(payload)
        or pack.get("manifest_sha256") != sha256(manifest)
    ):
        return None
    return receipt


def run(contract_path: Path) -> int:
    contract_path = contract_path.expanduser().resolve()
    contract = _read(contract_path)
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("post-Starmie handoff contract schema changed")
    runtime = dict(contract["runtime"])
    state_path = _path(runtime, "state")
    lock_path = _path(runtime, "lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _resume_selected_handoff(state_path, contract):
            return 0
        source_service = str(contract["trigger"]["training_service"])
        if service_active(source_service):
            raise RuntimeError("Starmie trainer is still active; refusing handoff")
        source = validate_source(_source_contract(contract))
        frozen_registry_path = _path(
            dict(contract["gate_materialization"]),
            "base_frozen_specialist_registry",
        )
        frozen_registry = _read(frozen_registry_path)
        predecessor_evidence = validate_frozen_predecessor_registry(
            source,
            frozen_registry,
        )
        _save_state(
            state_path,
            "starmie_pass_verified",
            source=source,
            frozen_predecessors=predecessor_evidence,
        )

        ready, frozen, core, contract = _resolve_boundary_core(
            contract_path=contract_path,
            contract=contract,
            runtime=runtime,
            state_path=state_path,
        )

        next_config = dict(contract["next_specialist"])
        promotion_raw = str(runtime.get("future_assets_receipt") or "").strip()
        default_tree = _optional_path(runtime, "inactive_tree_candidate")
        default_audit = _optional_path(runtime, "candidate_audit")
        promotion_receipt = (
            Path(promotion_raw).expanduser().resolve()
            if promotion_raw
            else None
        )
        if default_tree is None or default_audit is None:
            if promotion_receipt is not None:
                raise RuntimeError(
                    "future asset promotion requires default candidate assets"
                )
            assets = {
                "source": "current_runtime_tree",
                "corpus_root": _path(next_config, "corpus_root"),
                "candidate_tree": None,
                "candidate_audit": None,
                "promotion_receipt": None,
            }
        else:
            assets = resolve_specialist_assets(
                default_corpus_root=_path(next_config, "corpus_root"),
                default_candidate_tree=default_tree,
                default_candidate_audit=default_audit,
                promotion_receipt=promotion_receipt,
                promotion_scope=str(
                    runtime.get("future_assets_scope") or "full_bundle"
                ),
            )
        candidate_audit = (
            _read(Path(assets["candidate_audit"]))
            if assets.get("candidate_audit") is not None
            else {}
        )
        if candidate_audit:
            if (
                candidate_audit.get("schema")
                != "poke_bot.public_matchup_tree_candidate_audit/v1"
                or candidate_audit.get("runtime_enabled") is not False
                or candidate_audit.get("artifact_sha256")
                != sha256(Path(assets["candidate_tree"]))
                or float(candidate_audit.get("minimum_precision") or 0.0)
                != 0.93
                or int(candidate_audit.get("minimum_weighted_support") or 0)
                != 10_000
            ):
                raise RuntimeError("staged runtime tree candidate audit changed")
            routable_ids = {
                str(value)
                for value in candidate_audit.get(
                    "accepted_specialist_ids", ()
                )
            }
        else:
            routable_ids = {
                str(value)
                for value in (
                    _read(_path(runtime, "runtime_tree")).get(
                        "runtime_contract"
                    )
                    or {}
                ).get("accepted_archetype_ids", ())
            }
        frozen_registry = _read(frozen_registry_path)
        predecessor_evidence = validate_frozen_predecessor_registry(
            source,
            frozen_registry,
        )
        completed_ids = {
            str(row.get("specialist_id") or "")
            for row in (frozen_registry.get("specialists") or [])
            if row.get("frozen") is True
        }
        completed_ids.add(str(contract["trigger"]["specialist_id"]))
        selection = select_next_specialist(
            state_path=_path(next_config, "state"),
            corpus_root=Path(assets["corpus_root"]),
            minimum_decisions=int(next_config["minimum_decisions"]),
            minimum_decisions_by_specialist=dict(
                next_config.get("minimum_decisions_by_specialist", {})
            ),
            minimum_records_by_specialist=dict(
                next_config.get("minimum_records_by_specialist", {})
            ),
            strict_priority_prefix=list(
                next_config.get("strict_priority_prefix", [])
            ),
            completed_ids=completed_ids,
            active_id=str(contract["trigger"]["specialist_id"]),
            routable_ids=routable_ids,
        )
        selection_path = _path(next_config, "selection_receipt")
        _atomic(selection_path, selection)
        selected = dict(selection["selected"])
        prestage = _validated_prestage(
            contract=contract,
            selected=selected,
            assets=assets,
        )
        generated = _generated_contract(
            contract=contract,
            source=source,
            selected=selected,
            core_digest=str(frozen["checkpoint_digest"]),
            assets=assets,
            prestage=prestage,
        )
        generated_path = _path(next_config, "generated_handoff_contract")
        generated_migration = None
        if generated_path.is_file() and _read(generated_path) != generated:
            generated_migration = _migrate_selected_asset_contract(
                generated_path=generated_path,
                existing=_read(generated_path),
                current=generated,
            )
        elif not generated_path.is_file():
            _atomic(generated_path, generated)
        _save_state(
            state_path,
            "next_specialist_selected",
            selection=selection,
            generated_handoff_contract=str(generated_path),
            generated_handoff_contract_sha256=sha256(generated_path),
            generated_handoff_contract_migration=generated_migration,
        )
        run_sequential_handoff(generated_path)
        _save_state(
            state_path,
            "next_specialist_started",
            selected_specialist=selected,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    return run(args.contract)


if __name__ == "__main__":
    raise SystemExit(main())
