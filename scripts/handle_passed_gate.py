#!/usr/bin/env python3
"""Archive one exact passed gate, queue the approved copy, then continue.

The handler is deliberately marker-driven.  It cannot act on dashboard
telemetry, sampled practice, research controls, a partial gate, or an
uncommitted evaluation. Each Kaggle upload consumes its own digest/message
bound authorization before network I/O, and an interrupted attempt is never
retried automatically. The canonical path requires one copy; explicit
two-copy handling remains only for replaying historical audit records.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint
from poke_bot.model import (
    DECISION_FUSION_REQUIRED_HEADS,
    DECISION_FUSION_SCHEMA,
    DECISION_FUSION_V2_OPTIONAL_HEADS,
    DECISION_FUSION_V2_SCHEMA,
    DECISION_FUSION_V3_MAX_RELIABILITY,
    DECISION_FUSION_V3_MIN_RELIABILITY,
    DECISION_FUSION_V3_ROUTE_SCHEMA,
    DECISION_FUSION_V3_SCHEMA,
)
from poke_bot.pure_rl.model_registry import (
    freeze_model,
    sha256,
    verify_frozen_model,
)
from poke_bot.matchup_adapters import EXPERT_IDS
from poke_bot.pure_rl.strong_public_gate import (
    GATE_PROGRAM_SCHEMA,
    GATE_RESULT_SCHEMA,
    materialize_fallback_gate_contract,
)
from poke_bot.rtp_evaluation_promotion import (
    RTPPromotionEvidenceError,
    read_r198_immutable_json_object,
    validate_r198_evaluation_receipt,
)


AUTH_SCHEMA = "poke_bot.kaggle_submission_authorization/v1"
ATTEMPT_SCHEMA = "poke_bot.kaggle_submission_attempt/v1"
HANDLER_SCHEMA = "poke_bot.passed_gate_handler/v1"
SUBMISSION_QUEUE_SCHEMA = "poke_bot.kaggle_submission_queue/v1"
APPROVAL_TEXT = "Each solid submit will only submit one copy to Kaggle"
BASE_ACTIVE_GATE_CHECKS = frozenset(
    {
        "audit",
        "skill_weighted_win_rate",
        "skill_weighted_confidence_lower",
        "s_tier_mean_floor",
        "individual_opponent_floor",
    }
)
_SUBMISSION_RTP_MODES = frozenset(
    {"default_off", "disabled", "enabled", "off", "direct", "recursive"}
)
_RTP_PROMOTION_SCHEMA = "poke_bot.rtp_promotion/v1"
_RTP_R197_SPECIALIST_ID = "alakazam"
_RTP_R198_EXACT_MAX_NEURAL_PASSES = 256
_RTP_R198_EXACT_MAX_ACTION_COMBOS = 1024
_RTP_R197_REQUIRED_NEURAL_PASSES = {
    "normal": 6,
    "forced_replan": 5,
}
_RTP_PROMOTION_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "status",
        "specialist_id",
        "parent_checkpoint_sha256",
        "sidecar_sha256",
        "sidecar_config_sha256",
        "max_neural_passes",
        "max_action_combos",
        "required_neural_passes",
        "deck_file_sha256",
        "deck_cards_sha256",
        "matchup_tree_sha256",
        "evaluation_receipt_path",
        "evaluation_receipt_sha256",
        "identity_gate_passed",
        "planner_activation_gate_passed",
        "reliability_gate_passed",
        "heldout_efficacy_gate_passed",
        "robustness_gate_passed",
        "latency_gate_passed",
        "serving_eligible",
        "action_authority_enabled",
        "created_at_utc",
    }
)
_RTP_PROMOTION_REQUIRED_TRUE_FIELDS = (
    "identity_gate_passed",
    "planner_activation_gate_passed",
    "reliability_gate_passed",
    "heldout_efficacy_gate_passed",
    "robustness_gate_passed",
    "latency_gate_passed",
    "serving_eligible",
    "action_authority_enabled",
)
_RTP_SUBMISSION_ENV_KEYS = (
    "POKEBOT_SUBMISSION_RTP_CHECKPOINT",
    "POKEBOT_SUBMISSION_RTP_PARENT_CHECKPOINT_SHA256",
    "POKEBOT_SUBMISSION_RTP_PROMOTION_RECEIPT",
)


def _required_active_gate_checks(criteria: dict[str, Any]) -> frozenset[str]:
    if "s_plus_individual_floor" in criteria:
        return BASE_ACTIVE_GATE_CHECKS | {"s_plus_matchup_floor_allowance"}
    return BASE_ACTIVE_GATE_CHECKS


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _verified_effective_design_fingerprint(
    run_dir: Path,
    *,
    manifest: dict[str, Any],
    ledger: dict[str, Any],
) -> tuple[str | None, str]:
    """Verify the immutable manifest plus its append-only migration chain."""

    contract = manifest.get("design_contract")
    fingerprint = str(manifest.get("design_fingerprint") or "")
    if (
        not isinstance(contract, dict)
        or not fingerprint.startswith("sha256:")
        or _canonical_digest(contract) != fingerprint
    ):
        return None, "successor manifest design contract is corrupt"

    migration_paths = sorted(
        Path(run_dir).glob("design_migrations/migration_*.json")
    )
    history = list(ledger.get("design_migration_history") or [])
    if len(history) != len(migration_paths):
        return None, "successor design migration ledger is incomplete"

    for index, migration_path in enumerate(migration_paths):
        receipt = _read_json(migration_path)
        previous = receipt.get("previous_contract")
        current = receipt.get("current_contract")
        previous_fingerprint = str(receipt.get("previous_fingerprint") or "")
        current_fingerprint = str(receipt.get("current_fingerprint") or "")
        receipt_path = Path(str(receipt.get("receipt") or "")).expanduser()
        boundary_next_iteration = int(
            receipt.get("boundary_next_iteration", -1)
        )
        last_completed_iteration = int(
            receipt.get("last_completed_iteration", -2)
        )
        reason = str(receipt.get("reason") or "")
        boundary_is_valid = (
            boundary_next_iteration > 0
            and last_completed_iteration == boundary_next_iteration - 1
        ) or (
            boundary_next_iteration == 0
            and last_completed_iteration == -1
            and reason == "receipt_backed_completed_collection_resume_v1"
        )
        if not (
            int(receipt.get("schema", -1)) == 1
            and isinstance(previous, dict)
            and isinstance(current, dict)
            and receipt_path.is_file()
            and receipt_path.resolve() == migration_path.resolve()
            and previous_fingerprint == fingerprint
            and previous == contract
            and _canonical_digest(previous) == previous_fingerprint
            and _canonical_digest(current) == current_fingerprint
            and boundary_is_valid
        ):
            return None, "successor design migration chain is corrupt"

        history_row = history[index]
        if not isinstance(history_row, dict):
            return None, "successor design migration ledger is corrupt"
        history_receipt_path = Path(
            str(history_row.get("receipt") or "")
        ).expanduser()
        if not (
            history_receipt_path.is_file()
            and history_receipt_path.resolve() == migration_path.resolve()
            and str(history_row.get("fingerprint") or "")
            == current_fingerprint
            and (
                history_row.get("recovered") is True
                or (
                    int(history_row.get("boundary_next_iteration", -1))
                    == boundary_next_iteration
                    and str(history_row.get("reason") or "")
                    == reason
                )
            )
        ):
            return None, "successor design migration ledger is corrupt"

        contract = current
        fingerprint = current_fingerprint

    if str(ledger.get("design_fingerprint") or "") != fingerprint:
        return None, "successor effective design fingerprint is not committed"
    return fingerprint, "verified"


def _successor_decision_fusion_runtime_ready(
    run_dir: Path,
    *,
    state: dict[str, Any],
    learner: dict[str, Any],
    allow_terminal_gate_evidence: bool = False,
) -> tuple[bool, str]:
    """Verify a generated successor's runtime-fused descendant.

    Generated successors start from an immutable bootstrap whose fused action
    path is authorized by the specialist activation receipt.  Their later RL
    checkpoints inherit that path, so they do not have the one-off loop
    activation receipt used by the original Dudunsparce migration.  Require
    the bootstrap receipt, run manifest, immutable descendant commit, local
    and remote publication, checkpoint tensors/config, and exact learner
    identity to agree before accepting that equivalent authorization.
    """

    run_dir = Path(run_dir).expanduser().resolve()
    state_learner = dict(state.get("learner") or {})
    learner_path = Path(str(learner.get("path") or "")).expanduser().resolve()
    learner_digest = str(learner.get("digest") or "")
    if (
        not learner_digest.startswith("sha256:")
        or not learner_path.is_file()
        or checkpoint.checkpoint_digest(learner_path) != learner_digest
    ):
        return False, "successor runtime learner identity changed"
    payload = checkpoint.load_checkpoint(learner_path, map_location="cpu")
    model_config = dict(payload.get("model_config") or {})
    fusion = dict((payload.get("provenance") or {}).get("decision_fusion") or {})
    core_required_heads = list(DECISION_FUSION_REQUIRED_HEADS)
    supported_schemas = {
        DECISION_FUSION_SCHEMA,
        DECISION_FUSION_V2_SCHEMA,
        DECISION_FUSION_V3_SCHEMA,
    }

    def valid_fusion_contract(row: dict[str, Any]) -> bool:
        schema = str(row.get("schema") or "")
        heads = list(row.get("required_heads") or [])
        if schema == DECISION_FUSION_SCHEMA:
            return heads == core_required_heads
        if schema not in {DECISION_FUSION_V2_SCHEMA, DECISION_FUSION_V3_SCHEMA}:
            return False
        optional = heads[len(core_required_heads) :]
        head_inventory_is_valid = (
            heads[: len(core_required_heads)] == core_required_heads
            and len(optional) == len(set(optional))
            and all(name in DECISION_FUSION_V2_OPTIONAL_HEADS for name in optional)
            and optional
            == [
                name
                for name in DECISION_FUSION_V2_OPTIONAL_HEADS
                if name in optional
            ]
        )
        if not head_inventory_is_valid or schema == DECISION_FUSION_V2_SCHEMA:
            return head_inventory_is_valid
        routes = dict(row.get("dedicated_routes") or {})
        return (
            row.get("typed_output_centered_routes") is True
            or routes.get("typed_output_centered") is True
        ) and (
            routes.get("schema") == DECISION_FUSION_V3_ROUTE_SCHEMA
            and routes.get("enabled") is True
            and routes.get("runtime_enabled") is True
            and routes.get("positive_bounded_reliability") is True
            and list(routes.get("reliability_bounds") or [])
            == [
                DECISION_FUSION_V3_MIN_RELIABILITY,
                DECISION_FUSION_V3_MAX_RELIABILITY,
            ]
            and float(routes.get("action_type_reliability_cap", -1.0)) == 0.25
            and int(routes.get("route_count", -1)) == len(heads)
            and list(routes.get("route_names") or []) == heads
        )

    specialist_id = str(payload.get("archetype_id") or "").strip().casefold()
    v3_config_is_valid = fusion.get("schema") != DECISION_FUSION_V3_SCHEMA or (
        model_config.get("decision_fusion_dedicated_routes_enabled") is True
        and model_config.get("decision_fusion_dedicated_routes_runtime_enabled")
        is True
        and model_config.get(
            "decision_fusion_typed_output_centered_routes_enabled"
        )
        is True
        and float(
            model_config.get("decision_fusion_action_type_reliability_cap", -1.0)
        )
        == 0.25
    )
    if not (
        specialist_id
        and model_config.get("decision_fusion_enabled") is True
        and model_config.get("decision_fusion_runtime_enabled") is True
        and fusion.get("schema") in supported_schemas
        and fusion.get("runtime_enabled") is True
        and valid_fusion_contract(fusion)
        and v3_config_is_valid
    ):
        return False, "successor checkpoint is not the complete fused runtime"

    manifest = _read_json(run_dir / "manifest.json")
    initial = dict(manifest.get("initial_learner_checkpoint") or {})
    initial_path = Path(str(initial.get("path") or "")).expanduser().resolve()
    initial_digest = str(initial.get("digest") or "")

    def bootstrap_lineage_matches(bootstrap_digest: str) -> bool:
        if bootstrap_digest == initial_digest:
            return True
        if not (
            bootstrap_digest.startswith("sha256:")
            and initial_digest.startswith("sha256:")
            and initial_path.is_file()
        ):
            return False
        try:
            frozen = verify_frozen_model(initial_path.parent)
        except RuntimeError:
            return False
        provenance = dict(frozen.get("provenance") or {})
        return bool(
            str(frozen.get("checkpoint_digest") or "") == initial_digest
            and Path(str(frozen.get("model_path") or "")).expanduser().resolve()
            == initial_path
            and provenance.get("kind") == "decision_fusion_v2_hot_start"
            and provenance.get("all_legacy_tensors_bit_identical") is True
            and str(provenance.get("source_checkpoint_digest") or "")
            == bootstrap_digest
        )

    design_fingerprint, design_reason = _verified_effective_design_fingerprint(
        run_dir,
        manifest=manifest,
        ledger=state,
    )
    if design_fingerprint is None:
        return False, design_reason
    if not (
        str(manifest.get("specialist_archetype") or "").strip().casefold()
        == specialist_id
        and str(initial.get("digest") or "").startswith("sha256:")
    ):
        return False, "successor run manifest is not lineage-bound"

    state_root = run_dir.parent.parent / "state"
    activation_candidates = sorted(
        state_root.glob(f"{specialist_id}-specialist-rl-activation-v*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    activation_path: Path | None = None
    for candidate_path in activation_candidates:
        receipt = _read_json(candidate_path)
        identity = dict(receipt.get("identity") or {})
        bootstrap = dict(identity.get("next_specialist_bootstrap") or {})
        bootstrap_fusion = dict(bootstrap.get("decision_fusion") or {})
        registration = dict(identity.get("runtime_registration") or {})
        runtime_row = dict(registration.get("runtime_row") or {})
        runtime_fusion = dict(runtime_row.get("decision_fusion") or {})
        runtime_initial_digest = "sha256:" + str(
            runtime_row.get("initial_checkpoint_sha256") or ""
        ).removeprefix("sha256:")
        bootstrap_digest = str(bootstrap.get("checkpoint_digest") or "")
        if (
            receipt.get("schema") == "poke_bot.specialist_rl_activation/v2"
            and receipt.get("status") == "ready"
            and str(bootstrap.get("specialist_id") or "").casefold()
            == specialist_id
            and str(registration.get("specialist_id") or "").casefold()
            == specialist_id
            and bootstrap_lineage_matches(bootstrap_digest)
            and runtime_initial_digest == bootstrap_digest
            and bootstrap_fusion.get("schema") in supported_schemas
            and bootstrap_fusion.get("runtime_enabled") is True
            and valid_fusion_contract(bootstrap_fusion)
            and runtime_fusion.get("schema") in supported_schemas
            and runtime_fusion.get("required") is True
            and runtime_fusion.get("runtime_enabled") is True
            and valid_fusion_contract(runtime_fusion)
        ):
            activation_path = candidate_path
            break
    if activation_path is None:
        return False, "successor bootstrap activation receipt is absent or invalid"

    completed_iteration = int(state.get("last_completed_iteration", -1))
    commit_path = run_dir / "commits" / f"iter_{completed_iteration:05d}.json"
    commit = _read_json(commit_path)
    commit_design_fingerprint, commit_design_reason = (
        _verified_effective_design_fingerprint(
            run_dir,
            manifest=manifest,
            ledger=commit,
        )
    )
    if commit_design_fingerprint is None:
        return False, commit_design_reason
    commit_state_learner = dict(commit.get("learner") or {})
    terminal_candidate_mode = bool(
        allow_terminal_gate_evidence and learner != state_learner
    )
    if not (
        completed_iteration >= 0
        and int(state.get("next_iteration", -1)) == completed_iteration + 1
        and commit_design_fingerprint == design_fingerprint
        and int(commit.get("last_completed_iteration", -1))
        == completed_iteration
        and int(commit.get("next_iteration", -1)) == completed_iteration + 1
        and commit_state_learner == state_learner
        and (terminal_candidate_mode or commit_state_learner == learner)
    ):
        return False, "successor immutable commit does not bind the active learner"
    lineage_rows = [
        dict(row)
        for row in (commit.get("history") or [])
        if isinstance(row, dict)
        and int(row.get("iteration", -1)) == completed_iteration
    ]
    if len(lineage_rows) != 1:
        return False, "successor commit has no unique terminal lineage row"
    lineage = lineage_rows[0]
    candidate = dict(lineage.get("candidate") or {})
    learner_after = dict(lineage.get("learner_after") or {})
    publish = dict(lineage.get("next_collection_publish") or {})
    published = (
        lineage.get("completed") is True
        and candidate == learner
        and learner_after == learner
        and str(publish.get("checkpoint") or "") == str(learner_path)
        and str(publish.get("digest") or "") == learner_digest
        and publish.get("local_ok") is True
        and publish.get("remote_ok") is True
    )
    if not published:
        result = dict(lineage.get("active_gate_result") or {})
        audit = dict(result.get("audit") or {})
        runtime = dict(audit.get("matchup_runtime") or {})
        games = int(result.get("games") or 0)
        promotion = dict(lineage.get("promotion") or {})
        terminal_rollback_is_bound = bool(
            learner_after == state_learner
            and promotion.get("passed") is False
            and dict(promotion.get("candidate") or {}) == learner
        )
        terminal_gate_audited = bool(
            allow_terminal_gate_evidence
            and not publish
            and lineage.get("completed") is True
            and candidate == learner
            and (
                learner_after == learner
                or (terminal_candidate_mode and terminal_rollback_is_bound)
            )
            and result.get("schema") == GATE_RESULT_SCHEMA
            and int(result.get("iteration", -1)) == completed_iteration
            and str(result.get("checkpoint") or "") == str(learner_path)
            and str(result.get("checkpoint_digest") or "") == learner_digest
            and games > 0
            and audit.get("passed") is True
            and audit.get("exact_distribution") is True
            and audit.get("exact_weights") is True
            and audit.get("greedy_required") is True
            and audit.get("greedy") is True
            and audit.get("both_seats") is True
            and int(audit.get("valid_games", -1)) == games
            and int(audit.get("rows", -1)) == games
            and int(audit.get("requested_games", -1)) == games
            and str(audit.get("checkpoint_digest") or "") == learner_digest
            and runtime.get("schema")
            == "poke_bot.matchup_runtime_collection_audit/v1"
            and runtime.get("all_games_audited") is True
            and runtime.get("all_runtime_enabled") is True
            and runtime.get("contract_clean") is True
            and int(runtime.get("games", -1)) == games
            and int(runtime.get("audited_games", -1)) == games
            and int(runtime.get("runtime_enabled_games", -1)) == games
            and int(runtime.get("runtime_disabled_games", -1)) == 0
            and int(runtime.get("missing_games", -1)) == 0
            and int(runtime.get("malformed_games", -1)) == 0
            and int(runtime.get("transition_contract_violations", -1)) == 0
        )
        if not terminal_gate_audited:
            return False, "successor terminal learner was not published fleet-wide"
        return (
            True,
            "verified successor fused terminal descendant via complete audited "
            f"active gate in {commit_path.name}",
        )
    return (
        True,
        "verified successor fused descendant via "
        f"{activation_path.name} and {commit_path.name}",
    )


def _terminal_gate_candidate(state: dict[str, Any]) -> dict[str, Any]:
    """Return the uniquely gate-audited terminal candidate, if one is bound."""
    completed_iteration = int(state.get("last_completed_iteration", -1))
    terminal_rows = [
        dict(row)
        for row in (state.get("history") or [])
        if isinstance(row, dict)
        and int(row.get("iteration", -1)) == completed_iteration
    ]
    if len(terminal_rows) != 1:
        return {}
    terminal_row = terminal_rows[0]
    terminal_candidate = dict(terminal_row.get("candidate") or {})
    terminal_result = dict(terminal_row.get("active_gate_result") or {})
    if (
        terminal_candidate
        and str(terminal_result.get("checkpoint") or "")
        == str(terminal_candidate.get("path") or "")
        and str(terminal_result.get("checkpoint_digest") or "")
        == str(terminal_candidate.get("digest") or "")
    ):
        return terminal_candidate
    return {}


def _decision_fusion_runtime_ready(
    run_dir: Path,
    *,
    allow_terminal_gate_evidence: bool = False,
) -> tuple[bool, str]:
    """Verify the exact learner is the receipted all-head serving checkpoint."""
    state = _read_json(Path(run_dir) / "loop_state.json")
    learner = dict(state.get("learner") or {})
    if allow_terminal_gate_evidence:
        learner = _terminal_gate_candidate(state) or learner
    activation = dict(state.get("decision_fusion_activation") or {})
    digest = str(learner.get("digest") or "")
    receipt_path = Path(str(activation.get("receipt") or "")).expanduser()
    loop_activation_ready = bool(
        activation.get("phase") == "runtime_active"
        and activation.get("runtime_enabled") is True
        and activation.get("serving_eligible") is True
        and str(activation.get("learner_digest") or "") == digest
        and digest.startswith("sha256:")
        and receipt_path.is_file()
    )
    if not loop_activation_ready:
        successor_ready, successor_reason = (
            _successor_decision_fusion_runtime_ready(
                run_dir,
                state=state,
                learner=learner,
                allow_terminal_gate_evidence=allow_terminal_gate_evidence,
            )
        )
        if successor_ready:
            return successor_ready, successor_reason
        return (
            False,
            "runtime activation receipt is not published; "
            f"{successor_reason}",
        )
    if sha256(receipt_path) != str(activation.get("receipt_digest") or ""):
        return False, "runtime activation receipt checksum changed"
    receipt = _read_json(receipt_path)
    if not (
        receipt.get("schema") == "poke_bot.causal_decision_fusion_runtime_boundary/v1"
        and str((receipt.get("runtime_learner") or {}).get("digest") or "")
        == digest
        and (receipt.get("decision_fusion") or {}).get("runtime_enabled") is True
        and (receipt.get("decision_fusion") or {}).get("serving_eligible") is True
    ):
        return False, "runtime activation receipt contract is incomplete"
    learner_path = Path(str(learner.get("path") or "")).expanduser()
    if not learner_path.is_file() or checkpoint.checkpoint_digest(learner_path) != digest:
        return False, "runtime learner checkpoint identity changed"
    payload = checkpoint.load_checkpoint(learner_path, map_location="cpu")
    model_config = dict(payload.get("model_config") or {})
    if not (
        model_config.get("decision_fusion_enabled") is True
        and model_config.get("decision_fusion_runtime_enabled") is True
    ):
        return False, "runtime learner checkpoint is not serving-enabled"
    return True, "verified"


def _canonical_digest(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _same_number(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except (TypeError, ValueError):
        return False


def validate_exact_pass(
    run_dir: Path,
    contract_path: Path,
    marker_name: str = "SPECIALIST_GATE_PASSED",
) -> dict[str, Any]:
    """Return a frozen plan only for one immutable, committed exact pass.

    ``loop_state.json`` is deliberately not an authority here.  It is a
    mutable recovery pointer and can be ahead of, behind, or inconsistent
    with append-only history.  The terminal marker is only a wake-up signal;
    every semantic field is rebound to the immutable iteration commit and to
    the canonical digest recorded in the published exact-result pointer.
    """

    run_dir = Path(run_dir).expanduser().resolve()
    contract_path = Path(contract_path).expanduser().resolve()
    if (
        not marker_name
        or Path(marker_name).name != marker_name
        or not marker_name.startswith("SPECIALIST_GATE_PASSED")
    ):
        raise RuntimeError("terminal marker name is invalid")
    marker_path = run_dir / marker_name
    marker = _read_json(marker_path)
    contract = _read_json(contract_path)
    gate = dict(contract.get("next_gate") or {})
    evaluation = dict(gate.get("evaluation") or {})
    roster = list(gate.get("roster") or [])
    if not marker:
        raise RuntimeError(f"{marker_name} is absent or invalid")
    if (
        contract.get("schema") != GATE_PROGRAM_SCHEMA
        or str(contract.get("active_gate_id") or "")
        != str(gate.get("id") or "")
        or not gate
        or not evaluation
        or not roster
    ):
        raise RuntimeError("active gate contract is absent/invalid")

    iteration = int(marker.get("iteration", -1))
    if iteration < 0:
        raise RuntimeError("terminal marker has no valid iteration")
    commit_path = (
        run_dir / "commits" / f"iter_{iteration:05d}.json"
    ).resolve()
    commit = _read_json(commit_path)
    if not commit:
        raise RuntimeError("immutable iteration commit is absent/invalid")
    commit_digest = _canonical_digest(commit)
    rows = [
        dict(row)
        for row in (commit.get("history") or [])
        if isinstance(row, dict) and int(row.get("iteration", -2)) == iteration
    ]
    if len(rows) != 1:
        raise RuntimeError("terminal iteration is not unique in committed history")
    row = rows[0]
    result = dict(row.get("active_gate_result") or {})
    audit = dict(result.get("audit") or {})
    candidate = dict(row.get("candidate") or {})
    checkpoint = Path(str(marker.get("checkpoint") or "")).expanduser().resolve()
    digest = str(marker.get("checkpoint_digest") or "")

    # Training may complete the one owner-authorized child gate staged in the
    # canonical contract.  Reconstruct that effective contract from immutable
    # history instead of trusting a mutable marker or accepting an arbitrary
    # gate-id substitution.  This intentionally mirrors the trainer's
    # iteration-boundary rule.
    base_gate_id = str(gate.get("id") or "")
    result_gate_id = str(result.get("gate_id") or "")
    if result_gate_id != base_gate_id:
        prior_gate_passed = any(
            isinstance(item, dict)
            and int(item.get("iteration", -1)) < iteration
            and isinstance(item.get("active_gate_result"), dict)
            and str(item["active_gate_result"].get("gate_id") or "")
            == base_gate_id
            and item["active_gate_result"].get("passed") is True
            for item in (commit.get("history") or [])
        )
        effective = materialize_fallback_gate_contract(
            contract,
            completed_iteration=iteration - 1,
            prior_gate_passed=prior_gate_passed,
        )
        if (
            effective is not None
            and str((effective.get("next_gate") or {}).get("id") or "")
            == result_gate_id
        ):
            contract = effective
            gate = dict(contract["next_gate"])
            evaluation = dict(gate.get("evaluation") or {})
            roster = list(gate.get("roster") or [])

    raw_result_pointer = str(gate.get("exact_result_pointer") or "").strip()
    if not raw_result_pointer:
        raise RuntimeError("active gate has no exact-result pointer")
    result_pointer_path = Path(raw_result_pointer).expanduser().resolve()
    result_pointer = _read_json(result_pointer_path)
    pointer_core = {
        key: value
        for key, value in result_pointer.items()
        if key not in {"committed", "commit", "commit_digest", "created_at_utc"}
    }

    expected_total = int(evaluation.get("games_total", 0))
    expected_per = int(evaluation.get("games_per_opponent", 0))
    expected_first = int(evaluation.get("seat0_games_per_opponent", 0))
    expected_second = int(evaluation.get("seat1_games_per_opponent", 0))
    expected_ids = {str(item.get("opponent_id") or "") for item in roster}
    audit_rows = dict(audit.get("per_opponent") or {})
    matchup_rows = {
        str(item.get("opponent_id") or ""): dict(item)
        for item in (result.get("matchups") or [])
        if isinstance(item, dict)
    }
    result_checks = dict(result.get("checks") or {})
    required_active_gate_checks = _required_active_gate_checks(
        dict(gate.get("pass_criteria") or {})
    )
    checks = {
        "commit_boundary": int(commit.get("last_completed_iteration", -1))
        == iteration
        and int(commit.get("next_iteration", -1)) == iteration + 1,
        "committed": row.get("completed") is True,
        "result_schema": result.get("schema") == GATE_RESULT_SCHEMA,
        "result_pointer_schema": result_pointer.get("schema")
        == GATE_RESULT_SCHEMA,
        "result_pointer_committed": result_pointer.get("committed") is True,
        "result_pointer_commit": str(result_pointer.get("commit") or "")
        == str(commit_path),
        "result_pointer_commit_digest": str(
            result_pointer.get("commit_digest") or ""
        )
        == commit_digest,
        "result_pointer_payload": pointer_core == result,
        "active_gate_passed": result.get("passed") is True,
        "gate_criteria_set": set(result_checks)
        == required_active_gate_checks,
        "all_gate_criteria": set(result_checks)
        == required_active_gate_checks
        and all(
            result_checks[name] is True
            for name in required_active_gate_checks
        ),
        "gate_id": str(result.get("gate_id") or "") == str(gate.get("id") or ""),
        "iteration": int(result.get("iteration", -1)) == iteration,
        "games": int(result.get("games", -1)) == expected_total,
        "marker_games": int(marker.get("games", -1)) == expected_total,
        "marker_wr": _same_number(marker.get("wr"), result.get("skill_weighted_wr")),
        "marker_lower": _same_number(
            marker.get("confidence_lower"), result.get("confidence_lower")
        ),
        "candidate_path": str(candidate.get("path") or "") == str(checkpoint),
        "candidate_digest": str(candidate.get("digest") or "") == digest,
        "result_path": str(result.get("checkpoint") or "") == str(checkpoint),
        "result_digest": str(result.get("checkpoint_digest") or "") == digest,
        "audit_passed": audit.get("passed") is True,
        "audit_distribution": audit.get("exact_distribution") is True,
        "audit_weights": audit.get("exact_weights") is True,
        "audit_greedy": audit.get("greedy_required") is True
        and audit.get("greedy") is True,
        "audit_both_seats": audit.get("both_seats") is True,
        "audit_games": int(audit.get("valid_games", -1))
        == int(audit.get("rows", -2))
        == int(audit.get("requested_games", -3))
        == expected_total,
        "audit_digest": str(audit.get("checkpoint_digest") or "") == digest,
        "audit_roster": set(audit_rows) == expected_ids,
        "matchup_roster": set(matchup_rows) == expected_ids,
        "checkpoint_exists": checkpoint.is_file(),
    }
    if checks["checkpoint_exists"]:
        checks["checkpoint_bytes"] = sha256(checkpoint) == digest
    else:
        checks["checkpoint_bytes"] = False
    for opponent_id in sorted(expected_ids):
        expected_tuple = (expected_per, expected_first, expected_second)
        audit_row = audit_rows.get(opponent_id) or {}
        matchup = matchup_rows.get(opponent_id) or {}
        checks[f"audit_allocation:{opponent_id}"] = (
            int(audit_row.get("games", -1)),
            int(audit_row.get("seat0", -1)),
            int(audit_row.get("seat1", -1)),
        ) == expected_tuple
        checks[f"matchup_allocation:{opponent_id}"] = (
            int(matchup.get("games", -1)),
            int(matchup.get("seat0", -1)),
            int(matchup.get("seat1", -1)),
        ) == expected_tuple
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError("exact gate pass validation failed: " + ",".join(failed))

    return {
        "schema": "poke_bot.exact_pass_archive_plan/v1",
        "run_dir": str(run_dir),
        "iteration": iteration,
        "commit_boundary": int(commit["last_completed_iteration"]),
        "checkpoint": str(checkpoint),
        "checkpoint_digest": digest,
        "gate_id": str(gate["id"]),
        "base_gate_id": base_gate_id,
        "effective_contract_digest": _canonical_digest(contract),
        "games": expected_total,
        "games_per_opponent": expected_per,
        "candidate_first_per_opponent": expected_first,
        "candidate_second_per_opponent": expected_second,
        "roster_ids": sorted(expected_ids),
        "skill_weighted_wr": float(result["skill_weighted_wr"]),
        "confidence_lower": float(result["confidence_lower"]),
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "commit": str(commit_path),
        "commit_digest": commit_digest,
        "commit_file_sha256": sha256(commit_path),
        "exact_result_pointer": str(result_pointer_path),
        "exact_result_pointer_sha256": sha256(result_pointer_path),
        "marker": marker,
        "result": result,
        "validation": checks,
    }


def validate_ceiling_completion(
    run_dir: Path,
    contract_path: Path,
    ceiling_iteration: int,
) -> dict[str, Any]:
    """Bind an owner-authorized ceiling transition to one durable checkpoint.

    Unlike :func:`validate_exact_pass`, this deliberately does not turn a
    failed gate into a pass.  It requires the complete audited evaluation and
    preserves its measured result verbatim.
    """

    run_dir = Path(run_dir).expanduser().resolve()
    contract_path = Path(contract_path).expanduser().resolve()
    iteration = int(ceiling_iteration)
    commit_path = run_dir / "commits" / f"iter_{iteration:05d}.json"
    commit = _read_json(commit_path)
    rows = [
        dict(row)
        for row in (commit.get("history") or [])
        if isinstance(row, dict) and int(row.get("iteration", -1)) == iteration
    ]
    if len(rows) != 1:
        raise RuntimeError("ceiling iteration is not unique in committed history")
    row = rows[0]
    result = dict(row.get("active_gate_result") or {})
    audit = dict(result.get("audit") or {})
    candidate = dict(row.get("candidate") or {})
    checkpoint = Path(str(candidate.get("path") or "")).expanduser().resolve()
    digest = str(candidate.get("digest") or "")
    matchups = [dict(item) for item in (result.get("matchups") or [])]
    roster_ids = sorted(str(item.get("opponent_id") or "") for item in matchups)
    audited = dict(audit.get("per_opponent") or {})
    validation = {
        "commit_boundary": int(commit.get("last_completed_iteration", -1))
        == iteration
        and int(commit.get("next_iteration", -1)) == iteration + 1,
        "committed": row.get("completed") is True,
        "result_schema": result.get("schema") == GATE_RESULT_SCHEMA,
        "iteration": int(result.get("iteration", -1)) == iteration,
        "audit_passed": audit.get("passed") is True,
        "audit_distribution": audit.get("exact_distribution") is True,
        "audit_weights": audit.get("exact_weights") is True,
        "audit_both_seats": audit.get("both_seats") is True,
        "roster": bool(roster_ids)
        and len(roster_ids) == len(set(roster_ids))
        and set(roster_ids) == set(audited),
        "checkpoint_exists": checkpoint.is_file(),
        "candidate_digest": bool(digest) and sha256(checkpoint) == digest,
        "result_digest": result.get("checkpoint_digest") == digest,
        "result_path": Path(str(result.get("checkpoint") or "")).resolve()
        == checkpoint,
    }
    failed = sorted(name for name, passed in validation.items() if not passed)
    if failed:
        raise RuntimeError(
            "ceiling completion validation failed: " + ",".join(failed)
        )
    contract = _read_json(contract_path)
    return {
        "schema": "poke_bot.ceiling_acceptance_archive_plan/v1",
        "completion_authority": "explicit_owner_ceiling_acceptance",
        "measured_gate_passed": result.get("passed") is True,
        "preserves_measured_gate_result": True,
        "run_dir": str(run_dir),
        "iteration": iteration,
        "commit_boundary": iteration,
        "checkpoint": str(checkpoint),
        "checkpoint_digest": digest,
        "gate_id": str(result.get("gate_id") or ""),
        "base_gate_id": str(result.get("gate_id") or ""),
        "effective_contract_digest": _canonical_digest(contract),
        "games": int(result.get("games", 0)),
        "games_per_opponent": (
            int(result.get("games", 0)) // len(roster_ids)
        ),
        "candidate_first_per_opponent": 125,
        "candidate_second_per_opponent": 125,
        "roster_ids": roster_ids,
        "skill_weighted_wr": float(result.get("skill_weighted_wr", 0.0)),
        "confidence_lower": float(result.get("confidence_lower", 0.0)),
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "commit": str(commit_path),
        "commit_digest": _canonical_digest(commit),
        "commit_file_sha256": sha256(commit_path),
        "exact_result_pointer": str(commit_path),
        "exact_result_pointer_sha256": sha256(commit_path),
        "marker": None,
        "result": result,
        "validation": validation,
    }


def validate_runtime_exact_gate(
    run_dir: Path,
    contract_path: Path,
    receipt_path: Path,
    *,
    accept_ceiling: bool,
    ceiling_iteration: int,
) -> dict[str, Any]:
    """Validate both exact gates for the serving-enabled fusion child.

    Runtime activation intentionally creates a checksum-distinct child after
    the immutable RL commit.  Its gate evidence therefore lives in a separate
    append-only receipt and must never be substituted with the flat parent's
    committed result.
    """

    run_dir = Path(run_dir).expanduser().resolve()
    contract_path = Path(contract_path).expanduser().resolve()
    receipt_path = Path(receipt_path).expanduser().resolve()
    receipt = _read_json(receipt_path)
    contract = _read_json(contract_path)
    gate = dict(contract.get("next_gate") or {})
    evaluation = dict(gate.get("evaluation") or {})
    result = dict(receipt.get("result") or {})
    checkpoint_row = dict(receipt.get("checkpoint") or {})
    activation_row = dict(receipt.get("activation_receipt") or {})
    boundary = dict(receipt.get("boundary") or {})
    checkpoint_path = Path(
        str(checkpoint_row.get("path") or "")
    ).expanduser().resolve()
    checkpoint_digest = str(checkpoint_row.get("digest") or "")
    activation_path = Path(
        str(activation_row.get("path") or "")
    ).expanduser().resolve()
    activation = _read_json(activation_path)
    commit_path = Path(str(boundary.get("commit") or "")).expanduser().resolve()
    commit = _read_json(commit_path)
    premium_rows = {
        str(row.get("opponent_id") or ""): dict(row)
        for row in result.get("matchups") or []
        if isinstance(row, dict)
    }
    official = dict(result.get("research_controls") or {})
    official_rows = {
        str(row.get("opponent_id") or ""): dict(row)
        for row in official.get("matchups") or []
        if isinstance(row, dict)
    }
    premium_ids = {
        str(row.get("opponent_id") or "")
        for row in gate.get("roster") or []
    }
    official_ids = {
        str(row.get("opponent_id") or "")
        for row in gate.get("research_measurements") or []
    }
    premium_per = int(evaluation.get("games_per_opponent") or 0)
    premium_total = int(evaluation.get("games_total") or 0)
    official_total = sum(
        int(row.get("games") or 0)
        for row in gate.get("research_measurements") or []
    )
    result_checks = dict(result.get("checks") or {})
    research_checks = dict(result.get("research_checks") or {})
    required_checks = _required_active_gate_checks(
        dict(gate.get("pass_criteria") or {})
    )
    premium_audit = dict(result.get("audit") or {})
    official_audit = dict(official.get("audit") or {})
    both_passed = bool(receipt.get("both_gates_passed"))
    measured_passed = bool(
        receipt.get("premium_gate_passed") is True
        and receipt.get("official_gate_passed") is True
        and result.get("passed") is True
        and all(research_checks.get(name) is True for name in (
            "research_control_audit",
            "accepted_official_holdout_non_regression",
        ))
    )
    completion_authority = str(receipt.get("completion_authority") or "")
    ceiling_accepted = bool(
        not measured_passed
        and accept_ceiling
        and int(receipt.get("iteration", -1)) == int(ceiling_iteration)
        and completion_authority == "explicit_owner_ceiling_acceptance"
    )
    activation_runtime = dict(activation.get("runtime_learner") or {})
    activation_decision = dict(activation.get("decision_fusion") or {})
    checks = {
        "receipt_schema": receipt.get("schema")
        == "poke_bot.causal_decision_fusion_exact_gate/v1",
        "receipt_complete": receipt.get("complete") is True,
        "training_ineligible": receipt.get("training_eligible") is False,
        "replay_ineligible": receipt.get("replay_eligible") is False,
        "run_dir": Path(str(receipt.get("run_dir") or "")).resolve() == run_dir,
        "iteration": int(receipt.get("iteration", -1))
        == int(ceiling_iteration),
        "checkpoint_exists": checkpoint_path.is_file(),
        "checkpoint_bytes": checkpoint_path.is_file()
        and sha256(checkpoint_path) == checkpoint_digest,
        "activation_exists": activation_path.is_file(),
        "activation_digest": activation_path.is_file()
        and sha256(activation_path) == str(activation_row.get("digest") or ""),
        "activation_schema": activation.get("schema")
        == "poke_bot.causal_decision_fusion_runtime_boundary/v1",
        "activation_checkpoint": (
            str(activation_runtime.get("path") or "") == str(checkpoint_path)
            and str(activation_runtime.get("digest") or "")
            == checkpoint_digest
        ),
        "activation_serving": (
            activation_decision.get("runtime_enabled") is True
            and activation_decision.get("serving_eligible") is True
        ),
        "commit_exists": commit_path.is_file(),
        "commit_file_digest": commit_path.is_file()
        and sha256(commit_path) == str(boundary.get("commit_digest") or ""),
        "commit_boundary": int(commit.get("last_completed_iteration", -1))
        == int(receipt.get("iteration", -2))
        and int(commit.get("next_iteration", -1))
        == int(receipt.get("iteration", -2)) + 1,
        "contract_path": str(
            (receipt.get("contract") or {}).get("path") or ""
        )
        == str(contract_path),
        "contract_digest": sha256(contract_path)
        == str((receipt.get("contract") or {}).get("digest") or ""),
        "gate_id": str(result.get("gate_id") or "")
        == str(gate.get("id") or ""),
        "result_schema": result.get("schema") == GATE_RESULT_SCHEMA,
        "result_digest": _canonical_digest(result)
        == str(receipt.get("result_digest") or ""),
        "result_checkpoint": (
            str(result.get("checkpoint") or "") == str(checkpoint_path)
            and str(result.get("checkpoint_digest") or "")
            == checkpoint_digest
        ),
        "premium_games": int(result.get("games", -1)) == premium_total,
        "premium_roster": set(premium_rows) == premium_ids,
        "premium_allocation": all(
            int(row.get("games", -1)) == premium_per
            and int(row.get("seat0", -1)) == premium_per // 2
            and int(row.get("seat1", -1)) == premium_per // 2
            for row in premium_rows.values()
        ),
        "premium_audit": (
            premium_audit.get("passed") is True
            and premium_audit.get("exact_distribution") is True
            and premium_audit.get("exact_weights") is True
            and premium_audit.get("greedy_required") is True
            and premium_audit.get("greedy") is True
        ),
        "official_games": int(official.get("games", -1))
        == official_total
        == 1000,
        "official_roster": set(official_rows) == official_ids,
        "official_allocation": all(
            int(row.get("games", -1)) == 250
            and int(row.get("seat0", -1)) == 125
            and int(row.get("seat1", -1)) == 125
            for row in official_rows.values()
        ),
        "official_audit": (
            official_audit.get("passed") is True
            and official_audit.get("exact_distribution") is True
            and official_audit.get("exact_weights") is True
            and official_audit.get("greedy_required") is True
        ),
        "premium_check_set": set(result_checks) == required_checks,
        "pass_identity": both_passed == measured_passed,
        "completion_authority": measured_passed or ceiling_accepted,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            "runtime exact gate validation failed: " + ",".join(failed)
        )
    return {
        "schema": "poke_bot.exact_pass_archive_plan/v1",
        "completion_authority": (
            "measured_both_gates_pass"
            if measured_passed
            else "explicit_owner_ceiling_acceptance"
        ),
        "measured_gate_passed": measured_passed,
        "preserves_measured_gate_result": True,
        "run_dir": str(run_dir),
        "iteration": int(receipt["iteration"]),
        "commit_boundary": int(receipt["iteration"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_digest": checkpoint_digest,
        "gate_id": str(gate["id"]),
        "base_gate_id": str(gate["id"]),
        "effective_contract_digest": _canonical_digest(contract),
        "games": premium_total,
        "games_per_opponent": premium_per,
        "candidate_first_per_opponent": premium_per // 2,
        "candidate_second_per_opponent": premium_per // 2,
        "roster_ids": sorted(premium_ids),
        "skill_weighted_wr": float(result["skill_weighted_wr"]),
        "confidence_lower": float(result["confidence_lower"]),
        "contract": str(contract_path),
        "contract_sha256": sha256(contract_path),
        "commit": str(commit_path),
        "commit_digest": _canonical_digest(commit),
        "commit_file_sha256": sha256(commit_path),
        "exact_result_pointer": str(receipt_path),
        "exact_result_pointer_sha256": sha256(receipt_path),
        "marker": None,
        "result": result,
        "complete_holdouts": {
            "premium": {
                "games": premium_total,
                "passed": bool(receipt["premium_gate_passed"]),
            },
            "official": {
                "games": official_total,
                "passed": bool(receipt["official_gate_passed"]),
            },
        },
        "validation": checks,
    }


def freeze_exact_pass(
    plan: dict[str, Any],
    *,
    registry_root: Path,
    family: str,
    display_name: str,
) -> dict[str, Any]:
    manifest = freeze_model(
        registry_root=registry_root,
        family=family,
        display_name=display_name,
        checkpoint=Path(plan["checkpoint"]),
        expected_digest=str(plan["checkpoint_digest"]),
        provenance={
            "run_dir": plan["run_dir"],
            "iteration": plan["iteration"],
            "gate_id": plan["gate_id"],
            "base_gate_id": plan["base_gate_id"],
            "effective_contract_digest": plan["effective_contract_digest"],
            "contract": plan["contract"],
            "contract_sha256": plan["contract_sha256"],
            "commit": plan["commit"],
            "commit_digest": plan["commit_digest"],
            "commit_file_sha256": plan["commit_file_sha256"],
            "exact_result_pointer": plan["exact_result_pointer"],
            "exact_result_pointer_sha256": plan[
                "exact_result_pointer_sha256"
            ],
            "archive_trigger": (
                "explicit_owner_ceiling_acceptance"
                if plan.get("completion_authority")
                == "explicit_owner_ceiling_acceptance"
                else "committed_exact_active_gate_pass"
            ),
        },
        evidence=plan["result"],
        require_exact_heldout=False,
        harden_permissions=True,
    )
    return verify_frozen_model(Path(manifest["model_path"]).parent)


def materialize_pinned_specialist_deck(
    *,
    run_dir: Path,
    representatives_path: Path,
    archetype: str,
    output_path: Path,
) -> dict[str, Any]:
    run_manifest = _read_json(Path(run_dir) / "manifest.json")
    representatives = _read_json(representatives_path)
    raw = ((representatives.get("decks") or {}).get(archetype) or {}).get(
        "card_ids"
    )
    cards = [int(card) for card in (raw or [])]
    if len(cards) != 60:
        raise RuntimeError("pinned specialist representative is not a 60-card deck")
    if (
        run_manifest.get("mode") != "specialist"
        or run_manifest.get("specialist_archetype") != archetype
        or run_manifest.get("our_decks") != [archetype]
    ):
        raise RuntimeError("run manifest is not the requested single-deck specialist")
    expected_entries = (
        ((run_manifest.get("design_contract") or {}).get("measurement_deck_distribution") or {})
        .get("entries")
        or []
    )
    if len(expected_entries) != 1:
        raise RuntimeError("run has no unique pinned measurement deck")
    cards_digest = _canonical_digest(cards)
    if (
        expected_entries[0].get("name") != archetype
        or expected_entries[0].get("cards_sha256") != cards_digest
    ):
        raise RuntimeError("submission deck differs from the exact trained deck")
    body = "".join(f"{card}\n" for card in cards)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_text(encoding="utf-8") != body:
        raise RuntimeError("refusing to replace an existing pinned submission deck")
    if not output_path.exists():
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        temporary.write_text(body, encoding="utf-8")
        os.replace(temporary, output_path)
    return {
        "path": str(output_path.resolve()),
        "cards": len(cards),
        "cards_sha256": cards_digest,
        "file_sha256": sha256(output_path),
        "representatives": str(Path(representatives_path).resolve()),
        "representatives_sha256": sha256(representatives_path),
    }


def _require_submission_sha256(value: object, *, field: str) -> str:
    """Return one canonical SHA-256 identity or reject the package input."""

    digest = str(value or "")
    if (
        len(digest) != 71
        or not digest.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise RuntimeError(f"RTP submission has invalid {field}")
    return digest


def _prepare_submission_rtp_environment(
    *,
    rtp_mode: str,
    rtp_promotion_receipt: Path | None,
    frozen_checkpoint: Path,
    frozen_model_digest: str,
    expected_archetype: str,
    deck_file_digest: str,
    deck_cards_digest: str,
    matchup_tree_digest: str | None,
) -> tuple[dict[str, str], dict[str, Any] | None]:
    """Resolve the only RTP inputs that may enter a submission build.

    Historical ``disabled``/``enabled`` calls retain their ambient sidecar
    interface. The r197 direct and recursive arms bind the same frozen parent,
    deck, tree, and inert sidecar; recursive alone additionally requires its
    separately supplied immutable promotion receipt before the build script is
    even invoked.
    """

    if rtp_mode not in _SUBMISSION_RTP_MODES:
        raise RuntimeError("invalid submission RTP mode")
    if rtp_promotion_receipt is not None and rtp_mode != "recursive":
        raise RuntimeError(
            "a RTP promotion receipt is valid only for recursive submissions"
        )
    if rtp_mode in {"default_off", "disabled", "off"}:
        return {}, None

    raw_sidecar = str(os.environ.get("POKEBOT_SUBMISSION_RTP_CHECKPOINT") or "")
    sidecar = Path(raw_sidecar).expanduser().resolve() if raw_sidecar else None
    if sidecar is None or not sidecar.is_file():
        raise RuntimeError(f"{rtp_mode} submission RTP checkpoint is unavailable")
    if rtp_mode == "enabled":
        return {"POKEBOT_SUBMISSION_RTP_CHECKPOINT": str(sidecar)}, None

    if rtp_mode == "direct":
        if expected_archetype != _RTP_R197_SPECIALIST_ID:
            raise RuntimeError("direct RTP submission is only defined for alakazam r197")
        frozen_checkpoint = Path(frozen_checkpoint).expanduser().resolve()
        if not frozen_checkpoint.is_file():
            raise RuntimeError("direct RTP frozen parent checkpoint is unavailable")
        expected_parent_digest = _require_submission_sha256(
            frozen_model_digest,
            field="frozen model digest",
        )
        if sha256(frozen_checkpoint) != expected_parent_digest:
            raise RuntimeError("direct RTP frozen parent checkpoint digest changed")
        expected_deck_cards_digest = _require_submission_sha256(
            deck_cards_digest,
            field="submission deck cards digest",
        )
        if matchup_tree_digest is None:
            raise RuntimeError("direct RTP submission requires a matchup tree")
        expected_tree_digest = _require_submission_sha256(
            matchup_tree_digest,
            field="submission matchup tree digest",
        )
        return (
            {
                "POKEBOT_SUBMISSION_RTP_CHECKPOINT": str(sidecar),
                "POKEBOT_SUBMISSION_RTP_PARENT_CHECKPOINT_SHA256": expected_parent_digest,
            },
            {
                "sidecar_sha256": sha256(sidecar),
                "parent_checkpoint_sha256": expected_parent_digest,
                "max_neural_passes": _RTP_R198_EXACT_MAX_NEURAL_PASSES,
                "max_action_combos": _RTP_R198_EXACT_MAX_ACTION_COMBOS,
                "required_neural_passes": dict(_RTP_R197_REQUIRED_NEURAL_PASSES),
                "deck_cards_sha256": expected_deck_cards_digest,
                "matchup_tree_sha256": expected_tree_digest,
            },
        )

    # ``recursive`` is an r197-only production candidate.  It must never
    # inherit receipt or parent values from the launcher environment.
    if expected_archetype != _RTP_R197_SPECIALIST_ID:
        raise RuntimeError("recursive RTP submission is only defined for alakazam r197")
    if rtp_promotion_receipt is None:
        raise RuntimeError("recursive submission requires a RTP promotion receipt")
    try:
        promotion_identity, receipt = read_r198_immutable_json_object(
            rtp_promotion_receipt,
            label="recursive RTP promotion receipt",
        )
    except RTPPromotionEvidenceError as exc:
        raise RuntimeError(
            f"recursive RTP promotion receipt is not immutable evidence: {exc}"
        ) from exc
    receipt_path = Path(str(promotion_identity["path"]))
    missing = sorted(_RTP_PROMOTION_REQUIRED_FIELDS.difference(receipt))
    if missing:
        raise RuntimeError(
            "recursive RTP promotion receipt is incomplete: " + ", ".join(missing)
        )
    if (
        receipt.get("schema") != _RTP_PROMOTION_SCHEMA
        or receipt.get("status") != "accepted"
        or receipt.get("specialist_id") != _RTP_R197_SPECIALIST_ID
    ):
        raise RuntimeError("recursive RTP promotion receipt is not an accepted r197 receipt")
    for field in _RTP_PROMOTION_REQUIRED_TRUE_FIELDS:
        if receipt.get(field) is not True:
            raise RuntimeError(f"recursive RTP promotion receipt does not pass {field}")
    if not isinstance(receipt.get("created_at_utc"), str) or not str(
        receipt["created_at_utc"]
    ).strip():
        raise RuntimeError("recursive RTP promotion receipt has no creation time")

    frozen_checkpoint = Path(frozen_checkpoint).expanduser().resolve()
    if not frozen_checkpoint.is_file():
        raise RuntimeError("recursive RTP frozen parent checkpoint is unavailable")
    expected_parent_digest = _require_submission_sha256(
        frozen_model_digest,
        field="frozen model digest",
    )
    if sha256(frozen_checkpoint) != expected_parent_digest:
        raise RuntimeError("recursive RTP frozen parent checkpoint digest changed")
    if _require_submission_sha256(
        receipt.get("parent_checkpoint_sha256"),
        field="promotion.parent_checkpoint_sha256",
    ) != expected_parent_digest:
        raise RuntimeError("recursive RTP promotion receipt parent digest mismatch")
    if _require_submission_sha256(
        receipt.get("sidecar_sha256"), field="promotion.sidecar_sha256"
    ) != sha256(sidecar):
        raise RuntimeError("recursive RTP promotion receipt sidecar digest mismatch")
    _require_submission_sha256(
        receipt.get("sidecar_config_sha256"),
        field="promotion.sidecar_config_sha256",
    )
    if (
        type(receipt.get("max_neural_passes")) is not int
        or receipt.get("max_neural_passes") != _RTP_R198_EXACT_MAX_NEURAL_PASSES
        or type(receipt.get("max_action_combos")) is not int
        or receipt.get("max_action_combos") != _RTP_R198_EXACT_MAX_ACTION_COMBOS
        or receipt.get("required_neural_passes")
        != _RTP_R197_REQUIRED_NEURAL_PASSES
    ):
        raise RuntimeError(
            "recursive RTP receipt does not bind exact 256-pass/1024-action r198"
        )

    expected_deck_cards_digest = _require_submission_sha256(
        deck_cards_digest,
        field="submission deck cards digest",
    )
    expected_deck_file_digest = _require_submission_sha256(
        deck_file_digest,
        field="submission deck file digest",
    )
    if matchup_tree_digest is None:
        raise RuntimeError("recursive RTP submission requires a matchup tree")
    expected_tree_digest = _require_submission_sha256(
        matchup_tree_digest,
        field="submission matchup tree digest",
    )
    if _require_submission_sha256(
        receipt.get("deck_file_sha256"), field="promotion.deck_file_sha256"
    ) != expected_deck_file_digest:
        raise RuntimeError("recursive RTP promotion receipt deck-file digest mismatch")
    if _require_submission_sha256(
        receipt.get("deck_cards_sha256"), field="promotion.deck_cards_sha256"
    ) != expected_deck_cards_digest:
        raise RuntimeError("recursive RTP promotion receipt deck digest mismatch")
    if _require_submission_sha256(
        receipt.get("matchup_tree_sha256"), field="promotion.matchup_tree_sha256"
    ) != expected_tree_digest:
        raise RuntimeError("recursive RTP promotion receipt matchup-tree digest mismatch")

    evaluation_digest = _require_submission_sha256(
        receipt.get("evaluation_receipt_sha256"),
        field="promotion.evaluation_receipt_sha256",
    )
    evaluation_path = Path(str(receipt.get("evaluation_receipt_path") or ""))
    try:
        validate_r198_evaluation_receipt(
            evaluation_path,
            expected_sha256=evaluation_digest,
            require_local_evidence=True,
            expected_parent_checkpoint_sha256=expected_parent_digest,
            expected_sidecar_sha256=sha256(sidecar),
            expected_sidecar_config_sha256=_require_submission_sha256(
                receipt.get("sidecar_config_sha256"),
                field="promotion.sidecar_config_sha256",
            ),
            expected_deck_file_sha256=expected_deck_file_digest,
            expected_deck_cards_sha256=expected_deck_cards_digest,
            expected_matchup_tree_sha256=expected_tree_digest,
        )
    except RTPPromotionEvidenceError as exc:
        raise RuntimeError(
            f"recursive RTP promotion evaluation receipt is non-promotable: {exc}"
        ) from exc
    receipt_digest = str(promotion_identity["sha256"])
    return (
        {
            "POKEBOT_SUBMISSION_RTP_CHECKPOINT": str(sidecar),
            "POKEBOT_SUBMISSION_RTP_PARENT_CHECKPOINT_SHA256": expected_parent_digest,
            "POKEBOT_SUBMISSION_RTP_PROMOTION_RECEIPT": str(receipt_path),
        },
        {
            "path": str(receipt_path),
            "sha256": receipt_digest,
            "parent_checkpoint_sha256": expected_parent_digest,
            "sidecar_sha256": sha256(sidecar),
            "sidecar_config_sha256": str(receipt["sidecar_config_sha256"]),
            "max_neural_passes": _RTP_R198_EXACT_MAX_NEURAL_PASSES,
            "max_action_combos": _RTP_R198_EXACT_MAX_ACTION_COMBOS,
            "required_neural_passes": dict(_RTP_R197_REQUIRED_NEURAL_PASSES),
            "deck_file_sha256": expected_deck_file_digest,
            "deck_cards_sha256": expected_deck_cards_digest,
            "matchup_tree_sha256": expected_tree_digest,
            "evaluation_receipt_path": str(evaluation_path),
            "evaluation_receipt_sha256": evaluation_digest,
        },
    )


def build_submission_bundle(
    *,
    repo_root: Path,
    frozen_manifest: dict[str, Any],
    deck_receipt: dict[str, Any],
    output_dir: Path,
    python: Path,
    archetype: str,
    matchup_tree: Path | None = None,
    matchup_roster: Path | None = None,
    cg_root: Path | None = None,
    turn_order_preference: str = "first_if_allowed",
    rtp_mode: str = "default_off",
    rtp_promotion_receipt: Path | None = None,
    direct_no_search_assets: bool = False,
) -> dict[str, Any]:
    from poke_bot import checkpoint as checkpoint_mod

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = output_dir / "submission.tar.gz"
    frozen_checkpoint = Path(str(frozen_manifest["model_path"])).resolve()
    checkpoint_payload = checkpoint_mod.load_checkpoint(
        frozen_checkpoint, map_location="cpu"
    )
    checkpoint_registry = dict(
        (checkpoint_payload.get("model_config") or {}).get(
            "matchup_adapter_registry"
        )
        or {}
    )
    checkpoint_expert_ids = tuple(
        str(value)
        for value in (
            checkpoint_registry.get("expert_ids")
            or checkpoint_registry.get("active_expert_ids")
            or EXPERT_IDS
        )
    )
    if (
        not checkpoint_expert_ids
        or len(checkpoint_expert_ids) != len(set(checkpoint_expert_ids))
    ):
        raise RuntimeError(
            "frozen checkpoint has no unique matchup-router identity"
        )
    checkpoint_archetype = str(
        checkpoint_payload.get("archetype_id") or ""
    ).strip().casefold()
    expected_archetype = str(archetype).strip().casefold()
    if turn_order_preference not in {
        "first_if_allowed",
        "second_if_allowed",
    }:
        raise RuntimeError("invalid submission turn-order preference")
    if rtp_mode not in _SUBMISSION_RTP_MODES:
        raise RuntimeError("invalid submission RTP mode")
    if checkpoint_archetype != expected_archetype:
        raise RuntimeError(
            "frozen checkpoint archetype does not match the submission "
            f"specialist: checkpoint={checkpoint_archetype!r} "
            f"requested={expected_archetype!r}"
        )
    env = dict(os.environ)
    env.update(
        {
            "POKEBOT_PYTHON": str(python),
            "POKEBOT_PRIMARY_ARCHETYPE": archetype,
            "POKEBOT_SUBMISSION_DECK": str(deck_receipt["path"]),
            "POKEBOT_BLACKWELL_STRATEGY_HEADS": "0",
            "POKEBOT_SEARCH_MODE": "policy",
            "POKEBOT_ALLOW_ORACLE_DECK": "0",
            "POKEBOT_SUBMISSION_TURN_ORDER": turn_order_preference,
            "POKEBOT_SUBMISSION_RTP_MODE": rtp_mode,
            "POKEBOT_SUBMISSION_DIRECT_NO_SEARCH_ASSETS": (
                "1" if direct_no_search_assets else "0"
            ),
        }
    )
    if matchup_roster is not None:
        matchup_roster = Path(matchup_roster).expanduser().resolve()
        if not matchup_roster.is_file():
            raise RuntimeError("submission matchup roster does not exist")
        env["POKEBOT_SUBMISSION_MATCHUP_ROSTER"] = str(matchup_roster)
    if cg_root is not None:
        cg_root = Path(cg_root).expanduser().resolve()
        if not cg_root.is_dir() or not (cg_root / "libcg.so").is_file():
            raise RuntimeError("submission cg root lacks libcg.so")
        env["POKEBOT_SUBMISSION_CG_ROOT"] = str(cg_root)
    # Never allow an ambient launcher value to arm a different package mode.
    # The resolver below re-adds only mode-authorized RTP inputs.
    for name in _RTP_SUBMISSION_ENV_KEYS:
        env.pop(name, None)
    matchup_tree_digest: str | None = None
    if matchup_tree is not None:
        matchup_tree = Path(matchup_tree).expanduser().resolve()
        tree = _read_json(matchup_tree)
        runtime = dict(tree.get("runtime_contract") or {})
        targets = [str(value) for value in tree.get("targets") or []]
        accepted = {
            str(value)
            for value in runtime.get("accepted_archetype_ids") or []
        }
        if (
            not matchup_tree.is_file()
            or tree.get("runtime_enabled") is not True
            or tuple(targets) != checkpoint_expert_ids
            or len(set(targets)) != len(checkpoint_expert_ids)
            or expected_archetype not in targets
            or expected_archetype not in accepted
            or runtime.get("one_route_per_decision") is not True
            or runtime.get("unknown_route_exact_bypass") is not True
        ):
            raise RuntimeError(
                "submission matchup tree lacks the active specialist's "
                "validated canonical runtime route"
            )
        matchup_tree_digest = sha256(matchup_tree)
        env["POKEBOT_SUBMISSION_MATCHUP_TREE"] = str(matchup_tree)
    rtp_env, rtp_binding = _prepare_submission_rtp_environment(
        rtp_mode=rtp_mode,
        rtp_promotion_receipt=rtp_promotion_receipt,
        frozen_checkpoint=frozen_checkpoint,
        frozen_model_digest=str(frozen_manifest.get("checkpoint_digest") or ""),
        expected_archetype=expected_archetype,
        deck_file_digest=str(deck_receipt.get("file_sha256") or ""),
        deck_cards_digest=str(deck_receipt.get("cards_sha256") or ""),
        matchup_tree_digest=matchup_tree_digest,
    )
    env.update(rtp_env)
    completed = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "build_submission.sh"),
            str(frozen_manifest["model_path"]),
            str(output_dir),
        ],
        cwd=repo_root,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"submission build failed rc={completed.returncode}")
    attestation = Path(str(bundle) + ".go-first-verified.json")
    attested = _read_json(attestation)
    digest = sha256(bundle)
    if (
        not bundle.is_file()
        or attested.get("file_sha256") != digest
        or attested.get("schema")
        not in {
            "poke_bot.submission_go_first_attestation/v1",
            "poke_bot.submission_turn_order_attestation/v1",
        }
        or str(
            attested.get("turn_order_preference")
            or (
                "first_if_allowed"
                if attested.get("go_first_if_offered") is True
                else ""
            )
        )
        != turn_order_preference
    ):
        raise RuntimeError(
            "submission bundle lacks its digest-bound turn-order proof"
        )
    with tarfile.open(bundle, "r:gz") as archive:
        members = {
            member.name.removeprefix("./"): member
            for member in archive.getmembers()
            if member.isfile()
        }

        def member_bytes(name: str) -> bytes:
            member = members.get(name)
            stream = archive.extractfile(member) if member is not None else None
            if stream is None:
                raise RuntimeError(f"submission bundle is missing {name}")
            return stream.read()

        model_bytes = member_bytes("model.pt")
        deck_bytes = member_bytes("deck.csv")
        member_bytes("main.py")
        member_bytes("cg/api.py")
        search_config_bytes = (
            None
            if direct_no_search_assets
            else member_bytes("search_config.json")
        )
        belief_decks_bytes = (
            None
            if direct_no_search_assets
            else member_bytes("belief_decks.json")
        )
        if direct_no_search_assets and (
            "search_config.json" in members or "belief_decks.json" in members
        ):
            raise RuntimeError("direct-policy bundle contains forbidden search assets")
        packaged_tree_bytes = (
            member_bytes("matchup_tree.json")
            if matchup_tree is not None
            else None
        )
        runtime_profile_bytes = (
            member_bytes("runtime_profile.json")
            if rtp_mode in {"disabled", "enabled", "off", "direct", "recursive"}
            else None
        )
        rtp_sidecar_bytes = (
            member_bytes("rtp_shadow_planner.pt")
            if rtp_mode in {"enabled", "direct", "recursive"}
            else None
        )
        rtp_promotion_bytes = (
            member_bytes("rtp_promotion_receipt.json")
            if rtp_mode == "recursive"
            else None
        )
        rtp_evaluation_bytes = (
            member_bytes("rtp_evaluation_receipt.json")
            if rtp_mode == "recursive"
            else None
        )
    model_digest = "sha256:" + hashlib.sha256(model_bytes).hexdigest()
    deck_digest = "sha256:" + hashlib.sha256(deck_bytes).hexdigest()
    if model_digest != frozen_manifest["checkpoint_digest"]:
        raise RuntimeError("submission bundle contains the wrong frozen model")
    if deck_digest != deck_receipt["file_sha256"]:
        raise RuntimeError("submission bundle contains the wrong specialist deck")
    runtime_profile: dict[str, Any] = {}
    if rtp_mode == "disabled":
        try:
            runtime_profile = json.loads(runtime_profile_bytes or b"")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("NO RTP runtime profile is invalid JSON") from exc
        if (
            runtime_profile.get("schema")
            != "poke_bot.submission_runtime_profile/v1"
            or runtime_profile.get("recursive_turn_planner") != "disabled"
            or runtime_profile.get("display") != "NO RTP"
            or runtime_profile.get("rtp_sidecar_packaged") is not False
            or any(
                name.endswith("rtp_shadow_planner.pt")
                or name.endswith("rtp_checkpoint.pt")
                for name in members
            )
            or attested.get("recursive_turn_planner") != "disabled"
            or attested.get("submission_message_required_literal") != "NO RTP"
        ):
            raise RuntimeError("submission bundle failed its NO RTP contract")
    elif rtp_mode == "off":
        try:
            runtime_profile = json.loads(runtime_profile_bytes or b"")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("explicit NO RTP runtime profile is invalid JSON") from exc
        if (
            runtime_profile.get("schema")
            != "poke_bot.submission_runtime_profile/v1"
            or runtime_profile.get("rtp_mode") != "off"
            or runtime_profile.get("recursive_turn_planner") != "disabled"
            or runtime_profile.get("display") != "NO RTP"
            or runtime_profile.get("rtp_sidecar_packaged") is not False
            or runtime_profile.get("model_checkpoint_sha256") != model_digest
            or any(
                name.endswith("rtp_shadow_planner.pt")
                or name.endswith("rtp_checkpoint.pt")
                for name in members
            )
            or attested.get("rtp_mode") != "off"
            or attested.get("recursive_turn_planner") != "disabled"
            or attested.get("submission_message_required_literal") != "NO RTP"
            or attested.get("model_checkpoint_sha256") != model_digest
        ):
            raise RuntimeError("submission bundle failed its explicit NO RTP contract")
    elif rtp_mode == "direct":
        if rtp_binding is None:
            raise RuntimeError("direct RTP submission lost its sidecar binding")
        try:
            runtime_profile = json.loads(runtime_profile_bytes or b"")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("direct RTP runtime profile is invalid JSON") from exc
        sidecar_digest = "sha256:" + hashlib.sha256(
            rtp_sidecar_bytes or b""
        ).hexdigest()
        direct_config_digest = _require_submission_sha256(
            runtime_profile.get("rtp_config_sha256"),
            field="direct RTP config digest",
        )
        if (
            runtime_profile.get("schema")
            != "poke_bot.submission_runtime_profile/v1"
            or runtime_profile.get("rtp_mode") != "direct"
            or runtime_profile.get("recursive_turn_planner") != "enabled"
            or runtime_profile.get("display") != "DIRECT RTP"
            or runtime_profile.get("rtp_sidecar_packaged") is not True
            or runtime_profile.get("rtp_direct_bridge_only") is not True
            or runtime_profile.get("rtp_sizing_profile") != "pure_rl_r197"
            or runtime_profile.get("specialist_id") != _RTP_R197_SPECIALIST_ID
            or runtime_profile.get("model_checkpoint_sha256") != model_digest
            or runtime_profile.get("parent_checkpoint_sha256") != model_digest
            or runtime_profile.get("rtp_checkpoint_sha256") != sidecar_digest
            or runtime_profile.get("rtp_checkpoint_sha256")
            != rtp_binding["sidecar_sha256"]
            or runtime_profile.get("deck_cards_sha256")
            != rtp_binding["deck_cards_sha256"]
            or runtime_profile.get("matchup_tree_sha256")
            != rtp_binding["matchup_tree_sha256"]
            or type(runtime_profile.get("max_neural_passes")) is not int
            or runtime_profile.get("max_neural_passes")
            != _RTP_R198_EXACT_MAX_NEURAL_PASSES
            or type(runtime_profile.get("max_action_combos")) is not int
            or runtime_profile.get("max_action_combos")
            != _RTP_R198_EXACT_MAX_ACTION_COMBOS
            or runtime_profile.get("max_action_combos")
            != rtp_binding["max_action_combos"]
            or runtime_profile.get("required_neural_passes")
            != rtp_binding["required_neural_passes"]
            or any(
                name.endswith("rtp_promotion_receipt.json")
                or name.endswith("rtp_evaluation_receipt.json")
                for name in members
            )
            or attested.get("rtp_mode") != "direct"
            or attested.get("recursive_turn_planner") != "enabled"
            or attested.get("submission_message_required_literal") != "DIRECT RTP"
            or attested.get("model_checkpoint_sha256") != model_digest
            or attested.get("rtp_checkpoint_sha256") != sidecar_digest
            or attested.get("rtp_config_sha256") != direct_config_digest
            or attested.get("max_neural_passes")
            != _RTP_R198_EXACT_MAX_NEURAL_PASSES
            or attested.get("max_action_combos")
            != _RTP_R198_EXACT_MAX_ACTION_COMBOS
            or attested.get("required_neural_passes")
            != _RTP_R197_REQUIRED_NEURAL_PASSES
        ):
            raise RuntimeError("submission bundle failed its direct RTP contract")
    elif rtp_mode == "enabled":
        try:
            runtime_profile = json.loads(runtime_profile_bytes or b"")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("RTP runtime profile is invalid JSON") from exc
        sidecar_digest = "sha256:" + hashlib.sha256(
            rtp_sidecar_bytes or b""
        ).hexdigest()
        if (
            runtime_profile.get("schema")
            != "poke_bot.submission_runtime_profile/v1"
            or runtime_profile.get("recursive_turn_planner") != "enabled"
            or runtime_profile.get("display") != "RTP"
            or runtime_profile.get("rtp_sidecar_packaged") is not True
            or runtime_profile.get("rtp_checkpoint_sha256") != sidecar_digest
            or attested.get("recursive_turn_planner") != "enabled"
            or attested.get("submission_message_required_literal") != "RTP"
        ):
            raise RuntimeError("submission bundle failed its RTP contract")
    elif rtp_mode == "recursive":
        if rtp_binding is None:
            raise RuntimeError("recursive RTP submission lost its promotion receipt")
        try:
            runtime_profile = json.loads(runtime_profile_bytes or b"")
            packaged_promotion = json.loads(rtp_promotion_bytes or b"")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("recursive RTP package receipts are invalid JSON") from exc
        sidecar_digest = "sha256:" + hashlib.sha256(
            rtp_sidecar_bytes or b""
        ).hexdigest()
        promotion_digest = "sha256:" + hashlib.sha256(
            rtp_promotion_bytes or b""
        ).hexdigest()
        evaluation_digest = "sha256:" + hashlib.sha256(
            rtp_evaluation_bytes or b""
        ).hexdigest()
        if (
            runtime_profile.get("schema")
            != "poke_bot.submission_runtime_profile/v1"
            or runtime_profile.get("rtp_mode") != "recursive"
            or runtime_profile.get("recursive_turn_planner") != "enabled"
            or runtime_profile.get("display") != "RTP"
            or runtime_profile.get("rtp_sidecar_packaged") is not True
            or runtime_profile.get("rtp_sizing_profile") != "pure_rl_r197"
            or runtime_profile.get("specialist_id") != _RTP_R197_SPECIALIST_ID
            or runtime_profile.get("model_checkpoint_sha256") != model_digest
            or runtime_profile.get("parent_checkpoint_sha256") != model_digest
            or runtime_profile.get("rtp_checkpoint_sha256") != sidecar_digest
            or runtime_profile.get("rtp_checkpoint_sha256")
            != rtp_binding["sidecar_sha256"]
            or runtime_profile.get("rtp_config_sha256")
            != rtp_binding["sidecar_config_sha256"]
            or type(runtime_profile.get("max_neural_passes")) is not int
            or runtime_profile.get("max_neural_passes")
            != _RTP_R198_EXACT_MAX_NEURAL_PASSES
            or type(runtime_profile.get("max_action_combos")) is not int
            or runtime_profile.get("max_action_combos")
            != _RTP_R198_EXACT_MAX_ACTION_COMBOS
            or runtime_profile.get("max_action_combos")
            != rtp_binding["max_action_combos"]
            or runtime_profile.get("required_neural_passes")
            != rtp_binding["required_neural_passes"]
            or runtime_profile.get("deck_file_sha256")
            != rtp_binding["deck_file_sha256"]
            or runtime_profile.get("deck_cards_sha256")
            != rtp_binding["deck_cards_sha256"]
            or runtime_profile.get("matchup_tree_sha256")
            != rtp_binding["matchup_tree_sha256"]
            or runtime_profile.get("rtp_promotion_receipt_file")
            != "rtp_promotion_receipt.json"
            or runtime_profile.get("rtp_promotion_receipt_sha256") != promotion_digest
            or promotion_digest != rtp_binding["sha256"]
            or runtime_profile.get("rtp_evaluation_receipt_file")
            != "rtp_evaluation_receipt.json"
            or runtime_profile.get("rtp_evaluation_receipt_sha256")
            != evaluation_digest
            or evaluation_digest != rtp_binding["evaluation_receipt_sha256"]
            or packaged_promotion.get("parent_checkpoint_sha256") != model_digest
            or packaged_promotion.get("sidecar_sha256") != sidecar_digest
            or packaged_promotion.get("sidecar_config_sha256")
            != rtp_binding["sidecar_config_sha256"]
            or packaged_promotion.get("max_neural_passes")
            != _RTP_R198_EXACT_MAX_NEURAL_PASSES
            or packaged_promotion.get("max_action_combos")
            != _RTP_R198_EXACT_MAX_ACTION_COMBOS
            or packaged_promotion.get("required_neural_passes")
            != _RTP_R197_REQUIRED_NEURAL_PASSES
            or packaged_promotion.get("deck_file_sha256")
            != rtp_binding["deck_file_sha256"]
            or packaged_promotion.get("deck_cards_sha256")
            != rtp_binding["deck_cards_sha256"]
            or packaged_promotion.get("matchup_tree_sha256")
            != rtp_binding["matchup_tree_sha256"]
            or packaged_promotion.get("evaluation_receipt_sha256")
            != evaluation_digest
            or attested.get("rtp_mode") != "recursive"
            or attested.get("recursive_turn_planner") != "enabled"
            or attested.get("submission_message_required_literal") != "RTP"
            or attested.get("model_checkpoint_sha256") != model_digest
            or attested.get("rtp_checkpoint_sha256") != sidecar_digest
            or attested.get("rtp_config_sha256")
            != rtp_binding["sidecar_config_sha256"]
            or attested.get("rtp_promotion_receipt_sha256") != promotion_digest
            or attested.get("max_neural_passes")
            != _RTP_R198_EXACT_MAX_NEURAL_PASSES
            or attested.get("max_action_combos")
            != _RTP_R198_EXACT_MAX_ACTION_COMBOS
            or attested.get("required_neural_passes")
            != _RTP_R197_REQUIRED_NEURAL_PASSES
        ):
            raise RuntimeError("submission bundle failed its recursive RTP contract")
    search_config: dict[str, Any] = {}
    belief_decks: dict[str, Any] = {}
    deck_hypotheses: list[Any] = []
    if not direct_no_search_assets:
        try:
            search_config = json.loads(search_config_bytes or b"")
            belief_decks = json.loads(belief_decks_bytes or b"")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("submission belief-MCTS assets are invalid JSON") from exc
        deck_hypotheses = belief_decks.get("deck_lists") or []
    if not direct_no_search_assets and (
        search_config.get("schema")
        != "poke_bot.submission_search_config/v1"
        or search_config.get("enabled") is not False
        or search_config.get("algorithm")
        != "public_history_root_sampled_belief_mcts"
        or search_config.get("leaf_evaluator")
        != "trained_checkpoint_policy_value_head"
        or search_config.get("leaf_evaluator_checkpoint")
        != "submission_model_pt"
        or search_config.get("require_trained_state_evaluator") is not True
        or float(search_config.get("hard_cap_s") or 0) != 600.0
        or float(search_config.get("internal_deadline_s") or 0) != 540.0
        or float(search_config.get("final_greedy_reserve_s") or 0) != 20.0
        or float(search_config.get("total_search_budget_s") or 0) != 400.0
        or float(search_config.get("baseline_call_s") or 0) != 0.2
        or int(search_config.get("maximum_calls") or 0) != 340
        or int(search_config.get("expected_search_decisions") or 0) != 64
        or float(search_config.get("maximum_move_s") or 0) != 4.0
        or float(search_config.get("minimum_move_s") or 0) != 0.5
        or int(search_config.get("minimum_sims") or 0) != 50
        or int(search_config.get("maximum_sims") or 0) != 50
        or search_config.get("search_failure_behavior")
        != "greedy_current_decision_then_retry"
        or search_config.get("game_wide_greedy_only_for_time_budget") is not True
        or float(search_config.get("safety_factor") or 0) != 0.8
        or search_config.get("fallback") != "frozen_model_greedy_policy"
        or search_config.get("oracle_inputs_allowed") is not False
        or belief_decks.get("schema")
        != "poke_bot.submission_belief_decks/v1"
        or belief_decks.get("anonymous") is not True
        or belief_decks.get("contains_opponent_identity") is not False
        or belief_decks.get("deck_count") != len(deck_hypotheses)
        or len(deck_hypotheses) < 8
        or any(
            len(deck) != 60 or any(int(card) <= 0 for card in deck)
            for deck in deck_hypotheses
        )
    ):
        raise RuntimeError("submission belief-MCTS contract changed")
    search_config_digest = (
        None
        if search_config_bytes is None
        else "sha256:" + hashlib.sha256(search_config_bytes).hexdigest()
    )
    belief_decks_digest = (
        None
        if belief_decks_bytes is None
        else "sha256:" + hashlib.sha256(belief_decks_bytes).hexdigest()
    )
    if matchup_tree is not None:
        packaged_tree_digest = (
            "sha256:" + hashlib.sha256(packaged_tree_bytes or b"").hexdigest()
        )
        if packaged_tree_digest != matchup_tree_digest:
            raise RuntimeError(
                "submission bundle contains the wrong specialist matchup tree"
            )
    else:
        packaged_tree_digest = None
    return {
        "path": str(bundle.resolve()),
        "sha256": digest,
        "bytes": bundle.stat().st_size,
        "attestation": str(attestation.resolve()),
        "specialist_id": expected_archetype,
        "checkpoint_archetype_id": checkpoint_archetype,
        "turn_order_preference": turn_order_preference,
        "rtp_mode": rtp_mode,
        "runtime_profile": runtime_profile,
        "deck": deck_receipt,
        "contents": {
            "model_sha256": model_digest,
            "deck_sha256": deck_digest,
            "deck_cards_sha256": str(deck_receipt["cards_sha256"]),
            "matchup_tree_sha256": packaged_tree_digest,
            "search_config_sha256": search_config_digest,
            "belief_decks_sha256": belief_decks_digest,
            "belief_deck_count": len(deck_hypotheses),
            "belief_mcts_default": False,
            "search_assets_packaged": not direct_no_search_assets,
            "main_present": True,
            "vendored_cg_present": True,
            "rtp_enabled": (
                False
                if rtp_mode in {"disabled", "off"}
                else True
                if rtp_mode in {"enabled", "direct", "recursive"}
                else None
            ),
            "rtp_sidecar_sha256": (
                str((rtp_binding or {}).get("sidecar_sha256") or "")
                or None
            ),
            "rtp_promotion_receipt_sha256": (
                str((rtp_binding or {}).get("sha256") or "") or None
            ),
        },
    }


def _copy_submission_slot(bundle: dict[str, Any], root: Path, slot: int) -> dict[str, Any]:
    destination_dir = root / f"copy-{slot}"
    destination_dir.mkdir(parents=True, exist_ok=True)
    source = Path(bundle["path"])
    target = destination_dir / "submission.tar.gz"
    target_attestation = Path(str(target) + ".go-first-verified.json")
    if not target.exists():
        shutil.copy2(source, target)
        shutil.copy2(bundle["attestation"], target_attestation)
    if sha256(target) != bundle["sha256"]:
        raise RuntimeError(f"submission copy {slot} digest mismatch")
    proof = _read_json(target_attestation)
    if proof.get("file_sha256") != bundle["sha256"]:
        raise RuntimeError(f"submission copy {slot} attestation mismatch")
    return {
        "slot": slot,
        "path": str(target.resolve()),
        "sha256": bundle["sha256"],
        "attestation": str(target_attestation.resolve()),
        "specialist_id": str(bundle["specialist_id"]),
        "turn_order_preference": str(
            bundle.get("turn_order_preference") or "first_if_allowed"
        ),
        "rtp_mode": str(bundle.get("rtp_mode") or "default_off"),
        "runtime_profile": dict(bundle.get("runtime_profile") or {}),
        "model_sha256": str(bundle["contents"]["model_sha256"]),
        "deck_sha256": str(bundle["contents"]["deck_sha256"]),
        "deck_cards_sha256": str(bundle["contents"]["deck_cards_sha256"]),
        "representatives_sha256": str(
            bundle["deck"]["representatives_sha256"]
        ),
        "matchup_tree_sha256": str(
            bundle["contents"]["matchup_tree_sha256"]
        ),
        "search_config_sha256": str(
            bundle["contents"]["search_config_sha256"] or ""
        ),
        "belief_decks_sha256": str(
            bundle["contents"]["belief_decks_sha256"] or ""
        ),
        "search_assets_packaged": bool(
            bundle["contents"].get("search_assets_packaged", True)
        ),
    }


def queue_submission_copies(
    *,
    queue_path: Path,
    copies: list[dict[str, Any]],
    gate_plan: dict[str, Any],
    specialist_id: str,
    competition: str,
) -> list[dict[str, Any]]:
    """Persist ordered original-checkpoint copies without performing network I/O."""

    required_count = len(copies)
    if required_count not in {1, 2} or [
        int(row.get("slot", -1)) for row in copies
    ] != list(range(1, required_count + 1)):
        raise RuntimeError("submission queue requires exact ordered copy slots")
    checkpoint_digest = str(gate_plan.get("checkpoint_digest") or "")
    gate_id = str(gate_plan.get("gate_id") or "")
    owner_decision_source = str(
        gate_plan.get("owner_decision_source")
        or "GOAL.md#/decision-ledger/revision-18"
    )
    label_suffix = str(gate_plan.get("label_suffix") or "").strip()
    iteration = int(gate_plan.get("iteration", -1))
    if (
        not checkpoint_digest.startswith("sha256:")
        or not gate_id
        or iteration < 0
    ):
        raise RuntimeError(
            "submission queue requires an exact gate/checkpoint identity"
        )
    queue_path = Path(queue_path).expanduser().resolve()
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
    queued_at = dt.datetime.now(dt.timezone.utc).isoformat()
    with lock_path.open("a+", encoding="utf-8") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX)
        payload = _read_json(queue_path)
        if payload and payload.get("schema") != SUBMISSION_QUEUE_SCHEMA:
            raise RuntimeError("existing Kaggle submission queue schema changed")
        entries = [dict(row) for row in (payload.get("queue") or [])]
        by_identity = {
            (
                str(row.get("specialist_id") or ""),
                int(row.get("copy_number", -1)),
                str(row.get("checkpoint_checksum") or ""),
            ): row
            for row in entries
        }
        selected: list[dict[str, Any]] = []
        for copy in copies:
            slot = int(copy["slot"])
            copy_label_suffix = str(copy.get("label_suffix") or label_suffix).strip()
            if copy_label_suffix and (
                len(copy_label_suffix) > 80
                or any(character in copy_label_suffix for character in "\r\n\t")
            ):
                raise RuntimeError("submission label suffix is invalid")
            turn_order_preference = str(
                copy.get("turn_order_preference") or "first_if_allowed"
            )
            search_assets_packaged = bool(
                copy.get("search_assets_packaged", True)
            )
            search_assets_valid = (
                str(copy.get("search_config_sha256") or "").startswith("sha256:")
                and str(copy.get("belief_decks_sha256") or "").startswith("sha256:")
                if search_assets_packaged
                else copy.get("search_config_sha256") in {None, ""}
                and copy.get("belief_decks_sha256") in {None, ""}
            )
            if (
                str(copy.get("specialist_id") or "") != specialist_id
                or turn_order_preference
                not in {"first_if_allowed", "second_if_allowed"}
                or str(copy.get("model_sha256") or "") != checkpoint_digest
                or not str(copy.get("deck_sha256") or "").startswith("sha256:")
                or not str(copy.get("deck_cards_sha256") or "").startswith(
                    "sha256:"
                )
                or not str(copy.get("representatives_sha256") or "").startswith(
                    "sha256:"
                )
                or not str(copy.get("matchup_tree_sha256") or "").startswith(
                    "sha256:"
                )
                or not search_assets_valid
            ):
                raise RuntimeError(
                    "submission copy is not bound to the requested specialist, "
                    "frozen checkpoint, and pinned deck"
                )
            identity = (specialist_id, slot, checkpoint_digest)
            completion_authority = gate_plan.get("completion_authority")
            outcome_label = (
                "ceiling accepted"
                if completion_authority == "explicit_owner_ceiling_acceptance"
                else (
                    "training milestone"
                    if completion_authority == "training_milestone_snapshot"
                    else "exact gate"
                )
            )
            label = (
                f"{specialist_id} {outcome_label} iter "
                f"{int(gate_plan['iteration'])} "
                f"copy {slot}/{required_count} "
                f"{turn_order_preference.replace('_if_allowed', '')} "
                f"{checkpoint_digest[7:19]}"
                f"{(' ' + copy_label_suffix) if copy_label_suffix else ''}"
            )
            expected = {
                "specialist_id": specialist_id,
                "copy_number": slot,
                "turn_order_preference": turn_order_preference,
                "label": label,
                "checkpoint_checksum": checkpoint_digest,
                "model_checksum": str(copy["model_sha256"]),
                "deck_file_checksum": str(copy["deck_sha256"]),
                "deck_cards_checksum": str(copy["deck_cards_sha256"]),
                "representatives_checksum": str(
                    copy["representatives_sha256"]
                ),
                "matchup_tree_checksum": str(
                    copy["matchup_tree_sha256"]
                ),
                "search_config_checksum": str(
                    copy["search_config_sha256"]
                ),
                "belief_decks_checksum": str(
                    copy["belief_decks_sha256"]
                ),
                "search_assets_packaged": search_assets_packaged,
                "gate_id": gate_id,
                "iteration": iteration,
                "queued_at": queued_at,
                "queue_status": "pending",
                "competition": competition,
                "file": str(copy["path"]),
                "file_sha256": str(copy["sha256"]),
                "rtp_mode": str(copy.get("rtp_mode") or "default_off"),
                "retry_count": 0,
                "submitted_at": None,
                "submission_id": None,
                "returned_score": None,
                "failure_reason": None,
            }
            existing = by_identity.get(identity)
            if existing is not None:
                immutable = (
                    "specialist_id",
                    "copy_number",
                    "turn_order_preference",
                    "label",
                    "checkpoint_checksum",
                    "model_checksum",
                    "deck_file_checksum",
                    "deck_cards_checksum",
                    "representatives_checksum",
                    "matchup_tree_checksum",
                    "search_config_checksum",
                    "belief_decks_checksum",
                    "search_assets_packaged",
                    "gate_id",
                    "iteration",
                    "competition",
                    "file",
                    "file_sha256",
                    "rtp_mode",
                )
                if any(existing.get(key) != expected.get(key) for key in immutable):
                    raise RuntimeError("existing pending Kaggle copy identity changed")
                selected.append(existing)
                continue
            entries.append(expected)
            by_identity[identity] = expected
            selected.append(expected)
        _atomic_json(
            queue_path,
            {
                "schema": SUBMISSION_QUEUE_SCHEMA,
                "daily_submission_limit": 5,
                "minimum_hours_between_submissions": 4,
                "automatic_one_shot_authorization_on_training_complete": True,
                "one_shot_authorization_uses": 1,
                "standing_owner_decision_source": (
                    owner_decision_source
                ),
                "queue_order": "oldest_first",
                "retry_while_quota_exhausted": False,
                "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "quota": dict(payload.get("quota") or {}),
                "queue": entries,
            },
        )
    return selected


def _nonce(digest: str, slot: int) -> str:
    return f"exact-gate-{digest.removeprefix('sha256:')[:20]}-copy-{slot}"


def _consumed_receipts(receipts_dir: Path, nonce: str) -> list[Path]:
    if not receipts_dir.is_dir():
        return []
    return sorted(receipts_dir.glob(f"*{nonce}*.authorization-consumed.json"))


def _attempt_receipts(receipts_dir: Path, nonce: str) -> list[Path]:
    if not receipts_dir.is_dir():
        return []
    return sorted(
        path
        for path in receipts_dir.glob(f"*{nonce}*.json")
        if not path.name.endswith(".authorization-consumed.json")
    )


def _receipt_digests(paths: list[Path]) -> dict[str, str]:
    return {
        str(path.resolve()): sha256(path)
        for path in paths
        if path.is_file()
    }


def _evaluate_submission_receipts(
    *,
    copy_path: Path,
    copy_digest: str,
    nonce: str,
    consumed_paths: list[Path],
    attempt_paths: list[Path],
    process_returncode: int | None,
    timed_out: bool,
) -> tuple[dict[str, bool], int | None]:
    """Validate the guard's durable evidence for one upload invocation."""

    consumed = _read_json(consumed_paths[0]) if len(consumed_paths) == 1 else {}
    attempt = _read_json(attempt_paths[0]) if len(attempt_paths) == 1 else {}
    identity = dict(attempt.get("identity") or {})
    guard_returncode = attempt.get("returncode")
    try:
        guard_returncode_int = int(guard_returncode)
    except (TypeError, ValueError):
        guard_returncode_int = None
    process_returncode_supplied = process_returncode is not None
    try:
        process_returncode_int = (
            int(process_returncode) if process_returncode_supplied else None
        )
    except (TypeError, ValueError):
        process_returncode_int = None
    expected_consumed = (
        str(consumed_paths[0].resolve()) if len(consumed_paths) == 1 else ""
    )
    checks = {
        "copy_exists": copy_path.is_file(),
        "copy_digest": copy_path.is_file() and sha256(copy_path) == copy_digest,
        "one_consumed_authorization": len(consumed_paths) == 1,
        "one_guard_attempt_receipt": len(attempt_paths) == 1,
        "consumed_schema": consumed.get("schema") == AUTH_SCHEMA,
        "consumed_nonce": str(consumed.get("nonce") or "") == nonce,
        "consumed_before_upload": consumed.get("consumed_before_upload") is True,
        "consumed_one_shot": int(consumed.get("remaining_uses", -1)) == 0,
        "guard_attempt_schema": attempt.get("schema") == ATTEMPT_SCHEMA,
        "guard_attempt_nonce": str(attempt.get("nonce") or "") == nonce,
        "guard_authorization_identity": str(
            attempt.get("authorization_consumed") or ""
        )
        == expected_consumed,
        "guard_returncode_zero": guard_returncode_int == 0,
        "process_returncode_matches_guard": (
            not process_returncode_supplied
            or (
                process_returncode_int is not None
                and guard_returncode_int == process_returncode_int
            )
        ),
        "not_timed_out": timed_out is False,
        "retry_requires_new_approval": attempt.get(
            "retry_requires_new_explicit_user_approval"
        )
        is True,
        "identity_nonce": str(identity.get("nonce") or "") == nonce,
        "identity_file_sha256": str(identity.get("file_sha256") or "")
        == copy_digest,
    }
    return checks, guard_returncode_int


def _submission_attempt_record(
    *,
    copy: dict[str, Any],
    nonce: str,
    consumed_paths: list[Path],
    attempt_paths: list[Path],
    recovered: bool,
    process_returncode: int | None,
    timed_out: bool,
    message: str | None = None,
    started_at_epoch: float | None = None,
    completed_at_epoch: float | None = None,
) -> dict[str, Any]:
    copy_path = Path(str(copy["path"])).expanduser().resolve()
    copy_digest = str(copy["sha256"])
    checks, guard_returncode = _evaluate_submission_receipts(
        copy_path=copy_path,
        copy_digest=copy_digest,
        nonce=nonce,
        consumed_paths=consumed_paths,
        attempt_paths=attempt_paths,
        process_returncode=process_returncode,
        timed_out=timed_out,
    )
    succeeded = all(checks.values())
    return {
        "slot": int(copy["slot"]),
        "nonce": nonce,
        "file": str(copy_path),
        "file_sha256": copy_digest,
        "attempted": True,
        "succeeded": succeeded,
        "outcome": "success" if succeeded else "failed_or_unknown",
        "recovered_from_consumed_authorization": bool(recovered),
        "started_at_epoch": started_at_epoch,
        "completed_at_epoch": completed_at_epoch,
        "returncode": guard_returncode,
        "timed_out": bool(timed_out),
        "message": message,
        "automatic_retry": False,
        "success_checks": checks,
        "consumed_authorizations": [
            str(path.resolve()) for path in consumed_paths
        ],
        "consumed_authorization_digests": _receipt_digests(consumed_paths),
        "guard_attempt_receipts": [
            str(path.resolve()) for path in attempt_paths
        ],
        "guard_attempt_receipt_digests": _receipt_digests(attempt_paths),
    }


def _submission_attempts_by_slot(value: Any) -> dict[int, dict[str, Any]]:
    rows = [dict(row) for row in (value or []) if isinstance(row, dict)]
    if len(rows) != len(value or []):
        raise RuntimeError("submission attempt state contains a non-object")
    by_slot: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            slot = int(row.get("slot", -1))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("submission attempt has an invalid slot") from exc
        if (
            slot not in {1, 2}
            or slot in by_slot
            or row.get("attempted") is not True
            or row.get("automatic_retry") is not False
        ):
            raise RuntimeError("submission attempt state is duplicate or invalid")
        by_slot[slot] = row
    return by_slot


def validate_two_successful_submission_attempts(
    attempts: Any,
    *,
    expected_bundle_sha256: str,
) -> list[dict[str, Any]]:
    """Require exactly two uniquely receipted, successful upload invocations."""

    by_slot = _submission_attempts_by_slot(attempts)
    if set(by_slot) != {1, 2}:
        raise RuntimeError("exactly two submission attempts are required")
    nonces = {str(row.get("nonce") or "") for row in by_slot.values()}
    if len(nonces) != 2 or "" in nonces:
        raise RuntimeError("submission attempts do not have two unique nonces")
    validated: list[dict[str, Any]] = []
    for slot in (1, 2):
        row = by_slot[slot]
        copy_path = Path(str(row.get("file") or "")).expanduser().resolve()
        copy_digest = str(row.get("file_sha256") or "")
        nonce = str(row.get("nonce") or "")
        consumed_paths = [
            Path(str(path)).expanduser().resolve()
            for path in (row.get("consumed_authorizations") or [])
        ]
        attempt_paths = [
            Path(str(path)).expanduser().resolve()
            for path in (row.get("guard_attempt_receipts") or [])
        ]
        checks, guard_returncode = _evaluate_submission_receipts(
            copy_path=copy_path,
            copy_digest=copy_digest,
            nonce=nonce,
            consumed_paths=consumed_paths,
            attempt_paths=attempt_paths,
            process_returncode=row.get("returncode"),
            timed_out=bool(row.get("timed_out")),
        )
        try:
            recorded_returncode = int(row.get("returncode", -1))
        except (TypeError, ValueError):
            recorded_returncode = None
        recorded_consumed_digests = dict(
            row.get("consumed_authorization_digests") or {}
        )
        recorded_attempt_digests = dict(
            row.get("guard_attempt_receipt_digests") or {}
        )
        checks.update(
            {
                "expected_bundle_digest": copy_digest
                == str(expected_bundle_sha256),
                "expected_nonce": nonce == _nonce(copy_digest, slot),
                "recorded_consumed_digests": recorded_consumed_digests
                == _receipt_digests(consumed_paths),
                "recorded_attempt_digests": recorded_attempt_digests
                == _receipt_digests(attempt_paths),
                "recorded_success": row.get("succeeded") is True,
                "recorded_outcome": row.get("outcome") == "success",
                "recorded_returncode": guard_returncode == 0
                and recorded_returncode == 0,
            }
        )
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            raise RuntimeError(
                f"submission slot {slot} is not proven successful: "
                + ",".join(failed)
            )
        validated.append(row)
    return validated


def validate_successful_submission_attempts(
    attempts: Any,
    *,
    expected_bundle_sha256: str,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Validate the configured one-copy policy while retaining legacy audits."""

    if int(expected_count) == 2:
        return validate_two_successful_submission_attempts(
            attempts,
            expected_bundle_sha256=expected_bundle_sha256,
        )
    if int(expected_count) != 1:
        raise RuntimeError("submission count must be one under the active protocol")
    by_slot = _submission_attempts_by_slot(attempts)
    if set(by_slot) != {1}:
        raise RuntimeError("exactly one submission attempt is required")
    row = by_slot[1]
    copy_path = Path(str(row.get("file") or "")).expanduser().resolve()
    copy_digest = str(row.get("file_sha256") or "")
    nonce = str(row.get("nonce") or "")
    consumed_paths = [
        Path(str(path)).expanduser().resolve()
        for path in (row.get("consumed_authorizations") or [])
    ]
    attempt_paths = [
        Path(str(path)).expanduser().resolve()
        for path in (row.get("guard_attempt_receipts") or [])
    ]
    checks, guard_returncode = _evaluate_submission_receipts(
        copy_path=copy_path,
        copy_digest=copy_digest,
        nonce=nonce,
        consumed_paths=consumed_paths,
        attempt_paths=attempt_paths,
        process_returncode=row.get("returncode"),
        timed_out=bool(row.get("timed_out")),
    )
    try:
        recorded_returncode = int(row.get("returncode", -1))
    except (TypeError, ValueError):
        recorded_returncode = None
    checks.update(
        {
            "expected_bundle_digest": copy_digest
            == str(expected_bundle_sha256),
            "expected_nonce": nonce == _nonce(copy_digest, 1),
            "recorded_consumed_digests": dict(
                row.get("consumed_authorization_digests") or {}
            )
            == _receipt_digests(consumed_paths),
            "recorded_attempt_digests": dict(
                row.get("guard_attempt_receipt_digests") or {}
            )
            == _receipt_digests(attempt_paths),
            "recorded_success": row.get("succeeded") is True,
            "recorded_outcome": row.get("outcome") == "success",
            "recorded_returncode": guard_returncode == 0
            and recorded_returncode == 0,
        }
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            "submission slot 1 is not proven successful: " + ",".join(failed)
        )
    return [row]


def submit_one_approved_copy(
    *,
    copy: dict[str, Any],
    gate_plan: dict[str, Any],
    competition: str,
    kaggle: Path,
    authorization_path: Path,
    receipts_dir: Path,
    upload_timeout_seconds: float,
    approved_submission_count: int = 1,
) -> dict[str, Any]:
    slot = int(copy["slot"])
    nonce = _nonce(str(copy["sha256"]), slot)
    consumed = _consumed_receipts(receipts_dir, nonce)
    if consumed:
        attempts = _attempt_receipts(receipts_dir, nonce)
        recovered_payload = _read_json(attempts[0]) if len(attempts) == 1 else {}
        return _submission_attempt_record(
            copy=copy,
            nonce=nonce,
            consumed_paths=consumed,
            attempt_paths=attempts,
            recovered=True,
            process_returncode=None,
            timed_out=False,
            message=str((recovered_payload.get("identity") or {}).get("message") or ""),
            started_at_epoch=recovered_payload.get("started_at_epoch"),
            completed_at_epoch=recovered_payload.get("completed_at_epoch"),
        )
    message = (
        f"Exact gate iter {gate_plan['iteration']} "
        f"copy {slot}/{int(approved_submission_count)} "
        f"{str(gate_plan['checkpoint_digest'])[7:19]}"
    )
    authorization = {
        "schema": AUTH_SCHEMA,
        "explicit_user_approval": True,
        "approval_text": APPROVAL_TEXT,
        "approved_submission_ordinal": slot,
        "approved_submission_count": int(approved_submission_count),
        "conditional_gate_id": gate_plan["gate_id"],
        "conditional_checkpoint_digest": gate_plan["checkpoint_digest"],
        "remaining_uses": 1,
        "nonce": nonce,
        "expires_at_epoch": time.time() + 3600.0,
        "competition": competition,
        "file_sha256": copy["sha256"],
        "message": message,
        "turn_order_preference": str(
            copy.get("turn_order_preference") or "first_if_allowed"
        ),
    }
    if authorization_path.exists():
        existing = _read_json(authorization_path)
        identity_fields = (
            "schema",
            "explicit_user_approval",
            "remaining_uses",
            "nonce",
            "competition",
            "file_sha256",
            "message",
        )
        if (
            any(existing.get(key) != authorization.get(key) for key in identity_fields)
            or float(existing.get("expires_at_epoch", 0.0)) < time.time()
        ):
            raise RuntimeError("another Kaggle authorization is already present")
        authorization = existing
    else:
        _atomic_json(authorization_path, authorization)
    started = time.time()
    process = subprocess.Popen(
        [
            str(kaggle),
            "competitions",
            "submit",
            "-c",
            competition,
            "-f",
            str(copy["path"]),
            "-m",
            message,
        ],
        start_new_session=True,
    )
    timed_out = False
    try:
        returncode = process.wait(timeout=max(float(upload_timeout_seconds), 30.0))
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            returncode = process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait(timeout=10.0)
    completed_at = time.time()
    return _submission_attempt_record(
        copy=copy,
        nonce=nonce,
        consumed_paths=_consumed_receipts(receipts_dir, nonce),
        attempt_paths=_attempt_receipts(receipts_dir, nonce),
        recovered=False,
        process_returncode=int(returncode),
        timed_out=timed_out,
        message=message,
        started_at_epoch=started,
        completed_at_epoch=completed_at,
    )


def _service_active(service: str) -> bool:
    completed = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", service],
        check=False,
    )
    return completed.returncode == 0


def _service_main_pid(service: str) -> int:
    completed = subprocess.run(
        ["systemctl", "--user", "show", service, "-p", "MainPID", "--value"],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        return int(completed.stdout.strip()) if completed.returncode == 0 else 0
    except ValueError:
        return 0


def _service_properties(service: str) -> dict[str, str]:
    completed = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            service,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "Result",
            "-p",
            "ExecMainStatus",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {}
    properties: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    return properties


def _managed_maintenance_lock(service: str) -> dict[str, Any] | None:
    """Return a valid short-lived managed-boundary lock for this service."""

    raw_path = str(os.environ.get("POKEBOT_MANAGED_MAINTENANCE_LOCK") or "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not (
        payload.get("schema") == "poke_bot.managed_training_maintenance/v1"
        and payload.get("training_service") == service
        and float(payload.get("expires_at_epoch") or 0.0) > time.time()
        and int(payload.get("owner_pid") or 0) > 0
    ):
        return None
    return payload


def _recover_status_143_stop(service: str) -> dict[str, Any]:
    """Restart only an externally stopped pre-gate trainer.

    A real trainer exception must remain failed for diagnosis. The explicit
    stop signature observed from systemd is inactive/failed with
    ``ExecMainStatus=143``; no other exit status is automatically recovered.
    """

    properties = _service_properties(service)
    should_recover = (
        properties.get("ActiveState") in {"failed", "inactive"}
        and properties.get("ExecMainStatus") == "143"
    )
    if not should_recover:
        return {"recovered": False, "properties": properties}
    maintenance = _managed_maintenance_lock(service)
    if maintenance is not None:
        return {
            "recovered": False,
            "properties": properties,
            "managed_maintenance": maintenance,
        }
    reset = subprocess.run(
        ["systemctl", "--user", "reset-failed", service],
        check=False,
    )
    if reset.returncode != 0:
        raise RuntimeError(
            f"could not reset externally stopped {service}: rc={reset.returncode}"
        )
    start = subprocess.run(
        ["systemctl", "--user", "start", "--no-block", service],
        check=False,
    )
    if start.returncode != 0:
        raise RuntimeError(
            f"could not recover externally stopped {service}: rc={start.returncode}"
        )
    return {"recovered": True, "properties": properties}


def _resume_training(service: str) -> None:
    completed = subprocess.run(
        ["systemctl", "--user", "start", service],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not restart {service}: rc={completed.returncode}")


def _start_handoff(service: str) -> None:
    """Start the post-pass pipeline without resuming the terminal RL run."""

    completed = subprocess.run(
        ["systemctl", "--user", "start", "--no-block", service],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"could not start post-pass handoff {service}: "
            f"rc={completed.returncode}"
        )


def _install_continue_drop_in(source: Path, target: Path) -> dict[str, Any]:
    source = Path(source).expanduser().resolve()
    target = Path(target).expanduser()
    if not source.is_file():
        raise RuntimeError(f"continue-after-gate drop-in is missing: {source}")
    body = source.read_text(encoding="utf-8")
    if "Environment=PURE_RL_CONTINUE_AFTER_GATE=1" not in body:
        raise RuntimeError("continue drop-in does not arm the exact trainer flag")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_text(encoding="utf-8") != body:
        raise RuntimeError("refusing to replace a different active systemd drop-in")
    if not target.exists():
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_text(body, encoding="utf-8")
        os.replace(temporary, target)
    completed = subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("systemd daemon-reload failed for continuation drop-in")
    return {
        "source": str(source),
        "target": str(target.resolve()),
        "sha256": sha256(target),
        "armed": True,
    }


def _save_state(path: Path, state: dict[str, Any], **updates: Any) -> dict[str, Any]:
    state = {**state, **updates}
    state["schema"] = HANDLER_SCHEMA
    state["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _atomic_json(path, state)
    return state


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--marker-name",
        default="SPECIALIST_GATE_PASSED",
        help="Run-directory marker to poll and bind to the immutable gate commit.",
    )
    parser.add_argument(
        "--minimum-completed-iteration",
        type=int,
        default=-1,
        help=(
            "Reject otherwise valid pass markers before this immutable commit "
            "boundary without freezing, submitting, or starting handoff."
        ),
    )
    parser.add_argument(
        "--ceiling-completed-iteration",
        type=int,
        default=-1,
        help="Freeze, queue, and hand off this exact durable ceiling boundary.",
    )
    parser.add_argument(
        "--accept-ceiling-and-continue",
        action="store_true",
        help=(
            "Apply the standing owner rule at the configured ceiling while "
            "preserving a failed measured gate result."
        ),
    )
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--family", default="alakazam-strong-public-gate-v1")
    parser.add_argument(
        "--display-name", default="Alakazam Exact Strong-Public Gate Champion"
    )
    parser.add_argument(
        "--representatives",
        type=Path,
        default=ROOT / "data/training_mixes/top_ladder_representatives.v1.json",
    )
    parser.add_argument("--archetype", default="alakazam")
    parser.add_argument(
        "--matchup-tree",
        type=Path,
        default=None,
        help=(
            "Validated causal runtime tree to package with the frozen "
            "specialist. Required for routable specialist handoffs."
        ),
    )
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument(
        "--submission-count",
        type=int,
        default=1,
        help=(
            "Copies required for this passing specialist. The canonical "
            "protocol requires one; two remains accepted only for historical "
            "audit/replay of explicitly configured legacy handlers."
        ),
    )
    parser.add_argument(
        "--submission-turn-order-preference",
        action="append",
        choices=("first_if_allowed", "second_if_allowed"),
        default=None,
        help=(
            "Ordered per-copy turn-order profile. Repeat exactly once per "
            "submission copy. Omitted means first_if_allowed for every copy."
        ),
    )
    parser.add_argument(
        "--submission-mode",
        choices=("immediate", "queue_and_continue"),
        default="immediate",
    )
    parser.add_argument(
        "--submission-queue",
        type=Path,
        default=Path("/home/pokebot/poke-bot-agent/outputs/state/kaggle-submission-queue.json"),
    )
    parser.add_argument(
        "--kaggle",
        type=Path,
        default=Path("/home/pokebot/miniconda3/envs/poke-bot-agent/bin/kaggle"),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/pokebot/miniconda3/envs/poke-bot-agent/bin/python"),
    )
    parser.add_argument(
        "--authorization",
        type=Path,
        default=Path(
            "/home/pokebot/.config/pokebot/kaggle-submission-authorization.json"
        ),
    )
    parser.add_argument(
        "--submission-receipts",
        type=Path,
        default=Path("/home/pokebot/.local/state/pokebot/kaggle-submission-attempts"),
    )
    parser.add_argument(
        "--training-service", default="pokebot-pure-rl-alakazam.service"
    )
    parser.add_argument(
        "--recover-status-143-before-gate",
        action="store_true",
        help=(
            "While the exact marker is absent, restart this training service "
            "only when systemd proves an external SIGTERM/stop "
            "(ExecMainStatus=143). Real trainer failures remain stopped."
        ),
    )
    parser.add_argument(
        "--handoff-service",
        default="",
        help=(
            "Optional post-pass systemd unit. When set, the handler waits for "
            "the terminal trainer to stop, starts this unit, and deliberately "
            "does not arm/resume the completed Alakazam run."
        ),
    )
    parser.add_argument("--continue-drop-in-source", type=Path, required=True)
    parser.add_argument("--continue-drop-in-target", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--upload-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--require-decision-fusion-runtime",
        action="store_true",
        help=(
            "Do not freeze or hand off until the exact learner has a verified "
            "all-head serving-activation receipt."
        ),
    )
    parser.add_argument(
        "--runtime-exact-gate-receipt",
        type=Path,
        help=(
            "Append-only two-gate receipt for the serving-enabled fusion child. "
            "When configured, flat-parent markers and ceiling results are never "
            "eligible for freeze or handoff."
        ),
    )
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    submission_count = int(args.submission_count)
    if submission_count not in {1, 2}:
        raise RuntimeError("submission count must be one (or two for legacy audit)")
    turn_order_preferences = list(
        getattr(args, "submission_turn_order_preference", None) or ()
    )
    if not turn_order_preferences:
        turn_order_preferences = ["first_if_allowed"] * submission_count
    if (
        len(turn_order_preferences) != submission_count
        or any(
            value not in {"first_if_allowed", "second_if_allowed"}
            for value in turn_order_preferences
        )
    ):
        raise RuntimeError(
            "submission turn-order profiles must match submission count"
        )
    if (
        len(set(turn_order_preferences)) > 1
        and str(getattr(args, "submission_mode", "immediate"))
        != "queue_and_continue"
    ):
        raise RuntimeError(
            "distinct turn-order bundles require queue_and_continue mode"
        )
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = _read_json(args.state)
        submission_mode = str(getattr(args, "submission_mode", "immediate"))
        if (
            state.get("phase")
            in {"complete_training_resumed", "complete_handoff_started"}
        ):
            if state.get("submission_mode") == "queue_and_continue":
                if len(state.get("queued_submissions") or []) != submission_count:
                    raise RuntimeError("completed queued handoff lost required copies")
                return 0
            bundle = dict(state.get("submission_bundle") or {})
            validate_successful_submission_attempts(
                state.get("submission_attempts"),
                expected_bundle_sha256=str(bundle.get("sha256") or ""),
                expected_count=submission_count,
            )
            if state.get("all_submissions_succeeded") is not True:
                raise RuntimeError(
                    "completed handler state does not prove required successful uploads"
                )
            return 0
        if state.get("phase") in {
            "submission_failed_or_unknown",
            "two_submissions_failed_or_unknown",
        }:
            existing = _submission_attempts_by_slot(
                state.get("submission_attempts")
            )
            if set(existing) != set(range(1, submission_count + 1)):
                raise RuntimeError(
                    "terminal failed submission state has wrong attempt count"
                )
            return 0
        while True:
            marker_name = str(
                getattr(args, "marker_name", "SPECIALIST_GATE_PASSED")
            )
            marker = args.run_dir / marker_name
            plan = None
            runtime_gate_path = getattr(args, "runtime_exact_gate_receipt", None)
            if runtime_gate_path is not None:
                runtime_gate_path = Path(runtime_gate_path).expanduser().resolve()
                if runtime_gate_path.is_file():
                    plan = validate_runtime_exact_gate(
                        args.run_dir,
                        args.contract,
                        runtime_gate_path,
                        accept_ceiling=bool(
                            getattr(args, "accept_ceiling_and_continue", False)
                        ),
                        ceiling_iteration=int(args.ceiling_completed_iteration),
                    )
                else:
                    state = _save_state(
                        args.state,
                        state,
                        phase="waiting_for_runtime_exact_gate",
                        runtime_exact_gate_receipt=str(runtime_gate_path),
                        flat_parent_marker_eligible=False,
                        flat_parent_ceiling_result_eligible=False,
                        observed_training_pid=_service_main_pid(
                            args.training_service
                        ),
                    )
                    if args.once:
                        return 0
                    time.sleep(max(float(args.poll_seconds), 1.0))
                    continue
            elif (
                bool(getattr(args, "accept_ceiling_and_continue", False))
                and _service_main_pid(args.training_service) == 0
            ):
                try:
                    plan = validate_ceiling_completion(
                        args.run_dir,
                        args.contract,
                        int(args.ceiling_completed_iteration),
                    )
                except RuntimeError:
                    plan = None
            if runtime_gate_path is None and not marker.is_file() and plan is None:
                recovery = {"recovered": False, "properties": {}}
                if bool(
                    getattr(args, "recover_status_143_before_gate", False)
                ):
                    recovery = _recover_status_143_stop(
                        args.training_service
                    )
                state = _save_state(
                    args.state,
                    state,
                    phase="waiting_for_exact_gate_pass",
                    approval_text=APPROVAL_TEXT,
                        approved_submission_count=submission_count,
                        observed_training_pid=_service_main_pid(
                            args.training_service
                        ),
                        marker_name=marker_name,
                        minimum_completed_iteration=int(
                            getattr(args, "minimum_completed_iteration", -1)
                        ),
                        rejected_marker=None,
                        rejected_marker_sha256=None,
                        rejected_marker_reason=None,
                        external_stop_recovery_count=(
                            int(state.get("external_stop_recovery_count") or 0)
                            + int(recovery["recovered"] is True)
                        ),
                        last_external_stop_recovery=(
                            {
                                "recovered_at_utc": dt.datetime.now(
                                    dt.timezone.utc
                                ).isoformat(),
                                "service_properties": recovery["properties"],
                            }
                            if recovery["recovered"] is True
                            else state.get("last_external_stop_recovery")
                        ),
                    )
                if args.once:
                    return 0
                time.sleep(max(float(args.poll_seconds), 1.0))
                continue
            if bool(getattr(args, "require_decision_fusion_runtime", False)):
                fusion_ready, fusion_reason = _decision_fusion_runtime_ready(
                    args.run_dir,
                    allow_terminal_gate_evidence=bool(
                        plan is not None
                        and plan.get("schema")
                        == "poke_bot.ceiling_acceptance_archive_plan/v1"
                        and plan.get("completion_authority")
                        == "explicit_owner_ceiling_acceptance"
                    ),
                )
                if not fusion_ready:
                    state = _save_state(
                        args.state,
                        state,
                        phase="waiting_for_decision_fusion_runtime",
                        decision_fusion_runtime_required=True,
                        decision_fusion_runtime_ready=False,
                        decision_fusion_runtime_reason=fusion_reason,
                    )
                    if args.once:
                        return 0
                    time.sleep(max(float(args.poll_seconds), 1.0))
                    continue
                state = _save_state(
                    args.state,
                    state,
                    decision_fusion_runtime_required=True,
                    decision_fusion_runtime_ready=True,
                    decision_fusion_runtime_reason=fusion_reason,
                )
                loop_state = _read_json(args.run_dir / "loop_state.json")
                active_learner = dict(loop_state.get("learner") or {})
                if (
                    plan is not None
                    and plan.get("schema")
                    == "poke_bot.ceiling_acceptance_archive_plan/v1"
                    and plan.get("completion_authority")
                    == "explicit_owner_ceiling_acceptance"
                ):
                    active_learner = (
                        _terminal_gate_candidate(loop_state) or active_learner
                    )
                if plan is not None and (
                    str(plan.get("checkpoint") or "")
                    != str(active_learner.get("path") or "")
                    or str(plan.get("checkpoint_digest") or "")
                    != str(active_learner.get("digest") or "")
                ):
                    raise RuntimeError(
                        "gate plan is not the exact serving-enabled fusion learner"
                    )

            try:
                if plan is None:
                    plan = validate_exact_pass(
                        args.run_dir,
                        args.contract,
                        marker_name,
                    )
            except RuntimeError as exc:
                state = _save_state(
                    args.state,
                    state,
                    phase="waiting_for_current_exact_gate_pass",
                    rejected_marker=str(marker.resolve()),
                    rejected_marker_sha256=(
                        sha256(marker) if marker.is_file() else None
                    ),
                    rejected_marker_reason=str(exc),
                )
                if args.once:
                    return 0
                time.sleep(max(float(args.poll_seconds), 1.0))
                continue
            minimum_completed_iteration = int(
                getattr(args, "minimum_completed_iteration", -1)
            )
            if (
                minimum_completed_iteration >= 0
                and int(plan.get("commit_boundary", -1))
                < minimum_completed_iteration
            ):
                state = _save_state(
                    args.state,
                    state,
                    phase="waiting_for_minimum_terminal_iteration",
                    rejected_marker=str(marker.resolve()),
                    rejected_marker_sha256=(
                        sha256(marker) if marker.is_file() else None
                    ),
                    rejected_marker_reason=(
                        "exact pass predates minimum completed iteration: "
                        f"commit={int(plan.get('commit_boundary', -1))} "
                        f"minimum={minimum_completed_iteration}"
                    ),
                    minimum_completed_iteration=minimum_completed_iteration,
                )
                if args.once:
                    return 0
                time.sleep(max(float(args.poll_seconds), 1.0))
                continue
            state = _save_state(
                args.state,
                state,
                phase=(
                    "ceiling_acceptance_validated"
                    if plan.get("completion_authority")
                    == "explicit_owner_ceiling_acceptance"
                    else "exact_gate_validated"
                ),
                gate=plan,
                approval_text=APPROVAL_TEXT,
                approved_submission_count=submission_count,
            )
            frozen = freeze_exact_pass(
                plan,
                registry_root=args.registry_root,
                family=args.family,
                display_name=(
                    f"{args.archetype} Ceiling-Accepted Iteration "
                    f"{int(plan['iteration'])} Champion"
                    if plan.get("completion_authority")
                    == "explicit_owner_ceiling_acceptance"
                    else args.display_name
                ),
            )
            state = _save_state(
                args.state,
                state,
                phase="model_frozen",
                frozen_model=frozen,
            )

            deck = materialize_pinned_specialist_deck(
                run_dir=args.run_dir,
                representatives_path=args.representatives,
                archetype=args.archetype,
                output_path=(
                    args.submission_root
                    / f"pinned-{str(args.archetype).strip().casefold()}.deck.csv"
                ),
            )
            submission_repo_root = Path(
                os.environ.get("POKEBOT_SPECIALIST_RUNTIME_ROOT") or ROOT
            ).expanduser().resolve()
            if not (
                submission_repo_root / "scripts" / "build_submission.sh"
            ).is_file():
                raise RuntimeError(
                    "submission source root lacks scripts/build_submission.sh: "
                    f"{submission_repo_root}"
                )
            bundles = [
                build_submission_bundle(
                    repo_root=submission_repo_root,
                    frozen_manifest=frozen,
                    deck_receipt=deck,
                    output_dir=(
                        args.submission_root / "build"
                        if submission_count == 1
                        else args.submission_root / f"build-copy-{slot}"
                    ),
                    python=args.python,
                    archetype=args.archetype,
                    matchup_tree=getattr(args, "matchup_tree", None),
                    turn_order_preference=preference,
                )
                for slot, preference in enumerate(
                    turn_order_preferences,
                    start=1,
                )
            ]
            bundle = bundles[0]
            state = _save_state(
                args.state,
                state,
                phase="submission_bundle_verified",
                submission_bundle=bundle,
                submission_bundles=bundles,
            )

            if submission_mode == "queue_and_continue":
                copies = [
                    _copy_submission_slot(
                        bundles[slot - 1],
                        args.submission_root,
                        slot,
                    )
                    for slot in range(1, submission_count + 1)
                ]
                queued = queue_submission_copies(
                    queue_path=args.submission_queue,
                    copies=copies,
                    gate_plan=plan,
                    specialist_id=str(args.archetype),
                    competition=str(args.competition),
                )
                state = _save_state(
                    args.state,
                    state,
                    phase="submissions_queued",
                    submission_mode="queue_and_continue",
                    submission_queue=str(args.submission_queue.resolve()),
                    queued_submissions=queued,
                    successful_submission_count=0,
                    all_submissions_succeeded=False,
                    automatic_retries=False,
                )
            else:
                existing_attempts = _submission_attempts_by_slot(
                    state.get("submission_attempts")
                )
                attempts: list[dict[str, Any]] = []
                for slot in range(1, submission_count + 1):
                    prior = existing_attempts.get(slot)
                    if prior and prior.get("attempted") is True:
                        attempts.append(prior)
                        continue
                    copy = _copy_submission_slot(bundle, args.submission_root, slot)
                    attempt = submit_one_approved_copy(
                        copy=copy,
                        gate_plan=plan,
                        competition=args.competition,
                        kaggle=args.kaggle,
                        authorization_path=args.authorization,
                        receipts_dir=args.submission_receipts,
                        upload_timeout_seconds=args.upload_timeout_seconds,
                        approved_submission_count=submission_count,
                    )
                    attempts.append(attempt)
                    state = _save_state(
                        args.state,
                        state,
                        phase=f"submission_{slot}_attempted",
                        submission_attempts=attempts,
                    )
                state = _save_state(
                    args.state,
                    state,
                    phase="submission_attempts_recorded",
                    submission_attempts=attempts,
                    automatic_retries=False,
                )
                try:
                    successful_attempts = validate_successful_submission_attempts(
                        attempts,
                        expected_bundle_sha256=str(bundle["sha256"]),
                        expected_count=submission_count,
                    )
                except RuntimeError as exc:
                    _save_state(
                        args.state,
                        state,
                        phase=(
                            "two_submissions_failed_or_unknown"
                            if submission_count == 2
                            else "submission_failed_or_unknown"
                        ),
                        submission_attempts=attempts,
                        submission_outcome="failed_or_unknown",
                        submission_failure=str(exc),
                        successful_submission_count=sum(
                            item.get("succeeded") is True for item in attempts
                        ),
                        all_submissions_succeeded=False,
                        automatic_retries=False,
                        handoff_started=False,
                        training_resumed=False,
                    )
                    # A successful exit is intentional. Restart=on-failure must
                    # not relaunch an upload whose one-shot grant was consumed.
                    return 0
                state = _save_state(
                    args.state,
                    state,
                    phase="submissions_succeeded",
                    submission_mode="immediate",
                    submission_attempts=successful_attempts,
                    submission_outcome="success",
                    successful_submission_count=submission_count,
                    all_submissions_succeeded=True,
                    automatic_retries=False,
                )

            if str(args.handoff_service).strip():
                original_pid = int(state.get("observed_training_pid") or 0)
                state = _save_state(
                    args.state,
                    state,
                    phase="waiting_for_terminal_trainer_before_handoff",
                    handoff_service=str(args.handoff_service),
                    training_resumed=False,
                )
                while _service_active(args.training_service) and (
                    original_pid <= 0
                    or _service_main_pid(args.training_service) == original_pid
                ):
                    if args.once:
                        return 0
                    time.sleep(max(float(args.poll_seconds), 1.0))
                if _service_active(args.training_service):
                    raise RuntimeError(
                        "terminal trainer changed identity before post-pass handoff"
                    )
                _start_handoff(str(args.handoff_service))
                _save_state(
                    args.state,
                    state,
                    phase="complete_handoff_started",
                    handoff_service=str(args.handoff_service),
                    handoff_started=True,
                    training_service=args.training_service,
                    training_resumed=False,
                )
                return 0

            continuation = _install_continue_drop_in(
                args.continue_drop_in_source,
                args.continue_drop_in_target,
            )
            state = _save_state(
                args.state,
                state,
                phase="continuation_armed_after_archive_and_submissions",
                continuation=continuation,
            )

            original_pid = int(state.get("observed_training_pid") or 0)
            while _service_active(args.training_service) and (
                original_pid <= 0
                or _service_main_pid(args.training_service) == original_pid
            ):
                if args.once:
                    return 0
                time.sleep(max(float(args.poll_seconds), 1.0))
            if not _service_active(args.training_service):
                state = _save_state(
                    args.state,
                    state,
                    phase="resuming_training",
                    resume_requested=True,
                )
                _resume_training(args.training_service)
            if not _service_active(args.training_service):
                raise RuntimeError("training service did not become active after resume")
            _save_state(
                args.state,
                state,
                phase="complete_training_resumed",
                training_service=args.training_service,
                training_resumed=True,
            )
            return 0


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
