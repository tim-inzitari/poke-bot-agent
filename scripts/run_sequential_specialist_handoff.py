#!/usr/bin/env python3
"""Start one specialist only after the prior specialist is exactly frozen.

The source gate handler owns gate validation, freezing, and the asynchronous
Kaggle queue. This handoff only verifies that immutable evidence, verifies the
shared deck-agnostic core and next acting-seat corpus, runs the exact 25-epoch
bootstrap, writes a checksum-bound activation receipt, and starts the next
specialist. It never mutates the completed source specialist.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.expert_rehearsal import resolve_expert_manifest  # noqa: E402
from poke_bot.pure_rl.model_registry import (  # noqa: E402
    FAMILY_MARKER,
    sha256,
    verify_frozen_model,
)
from scripts.handle_passed_gate import (  # noqa: E402
    HANDLER_SCHEMA,
    validate_ceiling_completion,
    validate_exact_pass,
)
from scripts.accept_gate_threshold_transition import (  # noqa: E402
    validate_threshold_transition_receipt,
)
from scripts.materialize_frozen_specialist_gate import (  # noqa: E402
    materialize_from_contract,
)
from scripts.activate_public_matchup_tree import (  # noqa: E402
    activate as activate_public_matchup_tree,
)
from scripts.register_next_specialist_runtime import (  # noqa: E402
    register as register_specialist_runtime,
)
from scripts.run_starmie_expert_bootstrap import TARGETS  # noqa: E402


CONTRACT_SCHEMA = "poke_bot.sequential_specialist_handoff_contract/v1"
STATE_SCHEMA = "poke_bot.sequential_specialist_handoff_state/v1"
ACTIVATION_SCHEMA = "poke_bot.specialist_rl_activation/v2"
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
SAFE_SERVICE = re.compile(r"pokebot-[A-Za-z0-9_.@-]+\.service")
ALLOWED_HANDLER_PHASES = {
    "submissions_queued",
    "waiting_for_terminal_trainer_before_handoff",
    "complete_handoff_started",
}
RESUMABLE_PREFLIGHT_PHASES = {
    "preflight_verified",
    "next_specialist_bootstrap_frozen",
    "next_specialist_runtime_tree_bound",
    "next_specialist_runtime_registered",
    "next_specialist_rl_armed",
    "next_specialist_rl_started",
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"required JSON is missing/corrupt: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON is not an object: {path}")
    return value


def canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: dict[str, Any], *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if immutable:
        path.chmod(0o444)


def path_value(contract: dict[str, Any], section: str, key: str) -> Path:
    raw = str((contract.get(section) or {}).get(key) or "").strip()
    if not raw:
        raise RuntimeError(f"contract path is missing: {section}.{key}")
    return Path(raw).expanduser().resolve()


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path).expanduser().resolve()
    contract = read_json(path)
    source = dict(contract.get("source_specialist") or {})
    target = dict(contract.get("next_specialist") or {})
    training = dict(contract.get("training") or {})
    submission = dict(contract.get("submission_policy") or {})
    source_id = str(source.get("id") or "")
    target_id = str(target.get("id") or "")
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or not SAFE_ID.fullmatch(source_id)
        or not SAFE_ID.fullmatch(target_id)
        or source_id == target_id
        or not SAFE_SERVICE.fullmatch(str(source.get("training_service") or ""))
        or not SAFE_SERVICE.fullmatch(str(target.get("training_service") or ""))
        or not str(source.get("gate_marker_name") or "").startswith(
            "SPECIALIST_GATE_PASSED"
        )
        or int(training.get("supervised_epochs", -1)) != 25
        or int(training.get("patience_diagnostic_only", -1)) != 5
        or int(training.get("minimum_decisions", 0)) <= 0
        or int(training.get("requested_decisions_per_batch", 0)) <= 0
        or not 5 <= int(source.get("minimum_completed_iteration", -1)) <= 15
        or int(submission.get("required_copies", -1)) != 1
        or submission.get("completion_blocks_handoff") is not False
        or submission.get("queue_order") != "oldest_first"
    ):
        raise RuntimeError("sequential specialist handoff contract changed")
    for section, keys in {
        "source_specialist": (
            "run_dir",
            "gate_contract",
            "passed_family",
            "handler_state",
        ),
        "shared_core": ("family",),
        "next_specialist": (
            "expert_corpus",
            "ready",
            "run_dir",
            "cpu_pack_root",
            "activation_receipt",
            "gate_contract",
            "frozen_specialist_registry",
        ),
        "gate_materialization": (
            "base_gate_contract",
            "base_frozen_specialist_registry",
            "baseline_root",
            "baseline_manifest",
            "receipt",
        ),
        "paths": ("python", "registry_root", "state", "lock"),
    }.items():
        for key in keys:
            path_value(contract, section, key)
    registration = contract.get("runtime_registration")
    if registration is not None:
        if not isinstance(registration, dict):
            raise RuntimeError("runtime_registration must be an object")
        for key in (
            "runtime_tree",
            "runtime_registry",
            "selector_env",
            "state_root",
        ):
            path_value(contract, "runtime_registration", key)
        candidate = registration.get("inactive_tree_candidate")
        if candidate is not None:
            for key in (
                "inactive_tree_candidate",
                "candidate_audit",
                "activated_runtime_tree",
            ):
                path_value(contract, "runtime_registration", key)
            if (
                float(registration.get("minimum_validation_precision") or 0)
                != 0.93
                or int(
                    registration.get("minimum_validation_weighted_support") or 0
                )
                != 10_000
                or int(registration.get("consecutive_required") or 0) != 2
                or registration.get("allow_zero_materialized_adapters") is not True
            ):
                raise RuntimeError("runtime tree activation contract changed")
        if (
            not str(registration.get("run_name") or "").strip()
            or not SAFE_SERVICE.fullmatch(
                str(registration.get("handoff_service") or "")
            )
            or not SAFE_SERVICE.fullmatch(
                str(registration.get("gate_handler_service") or "")
            )
        ):
            raise RuntimeError("runtime registration contract changed")
    return contract, sha256(path)


def prepare_runtime_tree(
    contract: dict[str, Any],
    bootstrap: dict[str, Any],
) -> Path:
    registration = dict(contract["runtime_registration"])
    if registration.get("inactive_tree_candidate") is None:
        return path_value(contract, "runtime_registration", "runtime_tree")

    candidate = path_value(
        contract, "runtime_registration", "inactive_tree_candidate"
    )
    audit_path = path_value(contract, "runtime_registration", "candidate_audit")
    output = path_value(
        contract, "runtime_registration", "activated_runtime_tree"
    )
    audit = read_json(audit_path)
    if (
        audit.get("schema")
        != "poke_bot.public_matchup_tree_candidate_audit/v1"
        or audit.get("runtime_enabled") is not False
        or audit.get("artifact_sha256") != sha256(candidate)
        or int(audit.get("target_count") or 0) != 22
        or int(audit.get("accepted_count") or 0) < 1
        or float(audit.get("minimum_precision") or 0.0) != 0.93
        or int(audit.get("minimum_weighted_support") or 0) != 10_000
        or bootstrap["specialist_id"]
        not in set(audit.get("accepted_specialist_ids") or ())
    ):
        raise RuntimeError("inactive runtime tree candidate audit changed")
    frozen = verify_frozen_model(Path(str(bootstrap["family"])))
    checkpoint = Path(str(frozen["model_path"])).resolve()
    if output.is_file():
        existing = read_json(output)
        runtime = dict(existing.get("runtime_contract") or {})
        if (
            existing.get("runtime_enabled") is not True
            or runtime.get("source_tree_digest") != sha256(candidate)
            or runtime.get("checkpoint_digest") != bootstrap["checkpoint_digest"]
            or bootstrap["specialist_id"]
            not in set(runtime.get("accepted_archetype_ids") or ())
        ):
            raise RuntimeError("existing activated runtime tree identity changed")
        return output

    activate_public_matchup_tree(
        candidate,
        checkpoint,
        output,
        min_precision=0.93,
        min_support=10_000,
        min_leaf_confidence=0.90,
        consecutive_required=2,
        allow_zero_materialized_adapters=True,
    )
    activated = read_json(output)
    runtime = dict(activated.get("runtime_contract") or {})
    if (
        activated.get("runtime_enabled") is not True
        or runtime.get("source_tree_digest") != sha256(candidate)
        or runtime.get("checkpoint_digest") != bootstrap["checkpoint_digest"]
        or bootstrap["specialist_id"]
        not in set(runtime.get("accepted_archetype_ids") or ())
    ):
        raise RuntimeError("activated runtime tree failed identity verification")
    return output


def service_active(name: str) -> bool:
    return (
        subprocess.run(
            ["/usr/bin/systemctl", "--user", "is-active", "--quiet", name],
            check=False,
        ).returncode
        == 0
    )


def run_checked(argv: list[str]) -> None:
    completed = subprocess.run(argv, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(argv)}"
        )


def validate_source(contract: dict[str, Any]) -> dict[str, Any]:
    source = dict(contract["source_specialist"])
    handler_path = path_value(contract, "source_specialist", "handler_state")
    handler = read_json(handler_path)
    transition_receipt = str(source.get("threshold_transition_receipt") or "").strip()
    if transition_receipt:
        plan = validate_threshold_transition_receipt(
            Path(transition_receipt),
            specialist_id=str(source["id"]),
        )
    else:
        try:
            plan = validate_exact_pass(
                path_value(contract, "source_specialist", "run_dir"),
                path_value(contract, "source_specialist", "gate_contract"),
                str(source["gate_marker_name"]),
            )
        except RuntimeError:
            saved_plan = dict(handler.get("gate") or {})
            if (
                saved_plan.get("completion_authority")
                != "explicit_owner_ceiling_acceptance"
            ):
                raise
            plan = validate_ceiling_completion(
                path_value(contract, "source_specialist", "run_dir"),
                path_value(contract, "source_specialist", "gate_contract"),
                int(saved_plan.get("commit_boundary", -1)),
            )
            if saved_plan != plan:
                raise RuntimeError("saved ceiling-acceptance plan changed")
    frozen_path = path_value(contract, "source_specialist", "passed_family")
    frozen = verify_frozen_model(frozen_path)
    queued = [dict(row) for row in (handler.get("queued_submissions") or [])]
    minimum_completed_iteration = int(source["minimum_completed_iteration"])
    if (
        handler.get("schema") != HANDLER_SCHEMA
        or handler.get("phase") not in ALLOWED_HANDLER_PHASES
        or handler.get("submission_mode") != "queue_and_continue"
        or handler.get("gate") != plan
        or handler.get("frozen_model") != frozen
        or [int(row.get("copy_number", -1)) for row in queued] != [1]
        or any(
            row.get("checkpoint_checksum") != plan["checkpoint_digest"]
            for row in queued
        )
        or frozen.get("checkpoint_digest") != plan["checkpoint_digest"]
        or int(plan.get("commit_boundary", -1)) < minimum_completed_iteration
    ):
        raise RuntimeError("source pass/freeze/submission-queue identity changed")
    return {
        "specialist_id": source["id"],
        "gate": plan,
        "frozen_family": str(frozen_path),
        "frozen_manifest_sha256": sha256(frozen_path / "manifest.json"),
        "checkpoint_digest": frozen["checkpoint_digest"],
        "queued_submission_copies": [
            {
                "copy_number": int(row["copy_number"]),
                "label": row["label"],
                "checkpoint_checksum": row["checkpoint_checksum"],
                "queued_at": row["queued_at"],
            }
            for row in queued
        ],
    }


def validate_saved_preflight_source(
    contract: dict[str, Any],
    contract_digest: str,
) -> dict[str, Any] | None:
    """Recover source evidence committed before the next gate was expanded."""

    state_path = path_value(contract, "paths", "state")
    if not state_path.is_file():
        return None
    state = read_json(state_path)
    phase = str(state.get("phase") or "")
    if (
        state.get("schema") != STATE_SCHEMA
        or state.get("contract_sha256") != contract_digest
        or phase not in RESUMABLE_PREFLIGHT_PHASES
    ):
        raise RuntimeError("existing sequential handoff state is not resumable")
    saved = dict(state.get("source_specialist") or {})
    plan = dict(saved.get("gate") or {})
    frozen_path = path_value(contract, "source_specialist", "passed_family")
    frozen = verify_frozen_model(frozen_path)
    handler = read_json(
        path_value(contract, "source_specialist", "handler_state")
    )
    queued = [dict(row) for row in (handler.get("queued_submissions") or [])]
    projected_queue = [
        {
            "copy_number": int(row["copy_number"]),
            "label": row["label"],
            "checkpoint_checksum": row["checkpoint_checksum"],
            "queued_at": row["queued_at"],
        }
        for row in queued
    ]
    validation = dict(plan.get("validation") or {})
    if (
        saved.get("specialist_id")
        != str(contract["source_specialist"]["id"])
        or saved.get("frozen_family") != str(frozen_path)
        or saved.get("frozen_manifest_sha256")
        != sha256(frozen_path / "manifest.json")
        or saved.get("checkpoint_digest") != frozen.get("checkpoint_digest")
        or not validation
        or not all(value is True for value in validation.values())
        or handler.get("schema") != HANDLER_SCHEMA
        or handler.get("phase") not in ALLOWED_HANDLER_PHASES
        or handler.get("submission_mode") != "queue_and_continue"
        or handler.get("gate") != plan
        or handler.get("frozen_model") != frozen
        or saved.get("queued_submission_copies") != projected_queue
        or int(plan.get("commit_boundary", -1))
        < int(contract["source_specialist"]["minimum_completed_iteration"])
    ):
        raise RuntimeError("saved source preflight evidence changed")
    return saved


def validate_frozen_predecessor_registry(
    source: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed if a source gate's frozen S+ predecessors disappeared.

    A registry that silently drops an earlier frozen specialist can otherwise
    make selection and next-gate materialization appear internally consistent
    while forgetting a model that the just-passed gate actually contained.
    The immutable source-gate result is therefore the authority for the
    predecessor opponent IDs that must still be registered.
    """

    gate = dict(source.get("gate") or {})
    roster_ids = [str(value) for value in (gate.get("roster_ids") or ())]
    if not roster_ids or len(roster_ids) != len(set(roster_ids)):
        raise RuntimeError("source gate predecessor roster is missing or duplicated")
    required = {
        opponent_id
        for opponent_id in roster_ids
        if opponent_id.startswith("specialist-")
    }
    rows = [
        dict(row)
        for row in (registry.get("specialists") or ())
        if isinstance(row, dict)
    ]
    frozen_by_opponent = {
        str(row.get("opponent_id") or ""): row
        for row in rows
        if row.get("frozen") is True
    }
    missing = sorted(required - set(frozen_by_opponent))
    if (
        registry.get("schema") != "poke_bot.frozen_specialist_registry/v1"
        or "" in frozen_by_opponent
        or missing
    ):
        suffix = ", ".join(missing) if missing else "invalid registry identity"
        raise RuntimeError(
            "frozen predecessor registry dropped source-gate S+ opponent(s): "
            + suffix
        )
    return {
        "required_opponent_ids": sorted(required),
        "registered_opponent_ids": sorted(frozen_by_opponent),
        "required_count": len(required),
    }


def validate_core(contract: dict[str, Any]) -> dict[str, Any]:
    family = path_value(contract, "shared_core", "family")
    frozen = verify_frozen_model(family)
    expected = str(contract["shared_core"].get("checkpoint_checksum") or "")
    if (
        not expected.startswith("sha256:")
        or frozen.get("checkpoint_digest") != expected
        or not (family / FAMILY_MARKER).is_file()
        or sha256(family / "model.pt") != expected
    ):
        raise RuntimeError("shared deck-agnostic core identity changed")
    return {
        "family": str(family),
        "family_manifest_sha256": sha256(family / "manifest.json"),
        "protection_sha256": sha256(family / FAMILY_MARKER),
        "checkpoint_digest": expected,
    }


def validate_corpus(contract: dict[str, Any]) -> dict[str, Any]:
    target = dict(contract["next_specialist"])
    training = dict(contract["training"])
    pointer = path_value(contract, "next_specialist", "expert_corpus")
    corpus = resolve_expert_manifest(
        pointer,
        min_decisions=int(training["minimum_decisions"]),
        require_protected=True,
        required_archetype=str(target["id"]),
        required_compact_mode="temporal-expert-v1",
        required_max_context=320,
        required_target_coverage=tuple(
            training.get("required_target_coverage") or TARGETS
        ),
    )
    return {
        "pointer": str(pointer),
        "pointer_sha256": sha256(pointer),
        "manifest": corpus.path,
        "manifest_sha256": corpus.digest,
        "records": corpus.records,
        "decisions": corpus.decisions,
        "acting_seat_archetype": target["id"],
    }


def validate_next_specialist_gate(
    contract: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    """Require the next gate to contain every frozen specialist at S+.

    In particular, the source specialist row must reference the exact
    checksum-bound checkpoint that passed at or after the iteration floor.
    A stale pre-floor package therefore cannot authorize the next specialist.
    """
    gate_path = path_value(contract, "next_specialist", "gate_contract")
    registry_path = path_value(
        contract, "next_specialist", "frozen_specialist_registry"
    )
    gate_contract = read_json(gate_path)
    registry = read_json(registry_path)
    rows = [dict(row) for row in (registry.get("specialists") or [])]
    roster = [
        dict(row)
        for row in ((gate_contract.get("next_gate") or {}).get("roster") or [])
    ]
    source_id = str(source["specialist_id"])
    checkpoint_digest = str(source["checkpoint_digest"])
    source_rows = [
        row for row in rows if str(row.get("specialist_id") or "") == source_id
    ]
    gate_by_id = {
        str(row.get("opponent_id") or ""): row
        for row in roster
        if str(row.get("opponent_id") or "")
    }
    frozen_rows = [
        row
        for row in rows
        if row.get("frozen") is True and row.get("public_mix_eligible") is True
    ]
    evaluation = dict((gate_contract.get("next_gate") or {}).get("evaluation") or {})
    if (
        registry.get("schema") != "poke_bot.frozen_specialist_registry/v1"
        or len(source_rows) != 1
        or source_rows[0].get("checkpoint_digest") != checkpoint_digest
        or int(evaluation.get("games_per_opponent", -1)) != 250
        or int(evaluation.get("seat0_games_per_opponent", -1)) != 125
        or int(evaluation.get("seat1_games_per_opponent", -1)) != 125
        or int(evaluation.get("games_total", -1)) != 250 * len(roster)
    ):
        raise RuntimeError("next specialist frozen S+ gate identity changed")
    for row in frozen_rows:
        opponent_id = str(row.get("opponent_id") or "")
        gate_row = gate_by_id.get(opponent_id)
        if (
            gate_row is None
            or gate_row.get("tier") != "S+"
            or gate_row.get("frozen_specialist") is not True
            or gate_row.get("frozen_checkpoint_digest")
            != row.get("checkpoint_digest")
        ):
            raise RuntimeError(
                f"frozen specialist missing or not S+ in next gate: {opponent_id}"
            )
    return {
        "gate_contract": str(gate_path),
        "gate_contract_sha256": sha256(gate_path),
        "frozen_specialist_registry": str(registry_path),
        "frozen_specialist_registry_sha256": sha256(registry_path),
        "frozen_specialist_ids": [
            str(row["specialist_id"]) for row in frozen_rows
        ],
        "source_checkpoint_digest": checkpoint_digest,
        "games_total": int(evaluation["games_total"]),
    }


def bootstrap_command(contract: dict[str, Any]) -> list[str]:
    target = dict(contract["next_specialist"])
    training = dict(contract["training"])
    command = [
        str(path_value(contract, "paths", "python")),
        "-u",
        str(ROOT / "scripts/run_specialist_expert_bootstrap.py"),
        "--expert-corpus",
        str(path_value(contract, "next_specialist", "expert_corpus")),
        "--archetype",
        str(target["id"]),
        "--family",
        str(target["family_name"]),
        "--core-family",
        str(path_value(contract, "shared_core", "family")),
        "--registry-root",
        str(path_value(contract, "paths", "registry_root")),
        "--ready",
        str(path_value(contract, "next_specialist", "ready")),
        "--run-name",
        str(target["run_name"]),
        "--run-dir",
        str(path_value(contract, "next_specialist", "run_dir")),
        "--epochs",
        str(int(training["supervised_epochs"])),
        "--patience",
        str(int(training["patience_diagnostic_only"])),
        "--min-decisions",
        str(int(training["minimum_decisions"])),
        "--batch-size",
        str(int(training["requested_decisions_per_batch"])),
        "--cpu-pack-root",
        str(path_value(contract, "next_specialist", "cpu_pack_root")),
    ]
    for target_name in training.get("required_target_coverage") or TARGETS:
        command.extend(["--required-target", str(target_name)])
    return command


def validate_bootstrap(
    contract: dict[str, Any],
    core: dict[str, Any],
    corpus: dict[str, Any],
) -> dict[str, Any]:
    target = dict(contract["next_specialist"])
    family = path_value(contract, "paths", "registry_root") / str(
        target["family_name"]
    )
    frozen = verify_frozen_model(family)
    ready_path = path_value(contract, "next_specialist", "ready")
    ready = read_json(ready_path)
    provenance = dict(frozen.get("provenance") or {})
    if (
        ready.get("schema") != "poke_bot.specialist_expert_bootstrap_ready/v1"
        or ready.get("status") != "ready"
        or ready.get("acting_seat_archetype") != target["id"]
        or ready.get("checkpoint_digest") != frozen.get("checkpoint_digest")
        or ready.get("core_checkpoint_digest") != core["checkpoint_digest"]
        or ready.get("expert_manifest_sha256") != corpus["manifest_sha256"]
        or provenance.get("initialized_from_digest") != core["checkpoint_digest"]
        or (provenance.get("expert_manifest") or {}).get("digest")
        != corpus["manifest_sha256"]
        or provenance.get("trained_target_coverage")
        != list(contract["training"].get("required_target_coverage") or TARGETS)
    ):
        raise RuntimeError("next specialist bootstrap identity changed")
    return {
        "specialist_id": target["id"],
        "family": str(family),
        "family_manifest_sha256": sha256(family / "manifest.json"),
        "protection_sha256": sha256(family / FAMILY_MARKER),
        "checkpoint_digest": frozen["checkpoint_digest"],
        "core_checkpoint_digest": core["checkpoint_digest"],
        "ready": str(ready_path),
        "ready_sha256": sha256(ready_path),
        "corpus_manifest_sha256": corpus["manifest_sha256"],
        "supervised_epochs": 25,
    }


def activation_identity(
    contract_path: Path,
    contract_digest: str,
    source: dict[str, Any],
    core: dict[str, Any],
    corpus: dict[str, Any],
    bootstrap: dict[str, Any],
    next_gate: dict[str, Any],
    target_service: str,
    runtime_registration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "contract": str(contract_path.resolve()),
        "contract_sha256": contract_digest,
        "source_specialist": source,
        "shared_deck_agnostic_core": core,
        "next_specialist_corpus": corpus,
        "next_specialist_bootstrap": bootstrap,
        "next_specialist_splus_gate": next_gate,
        "next_specialist_service": target_service,
        "completed_source_remains_frozen": True,
        "kaggle_submission_completion_blocks_handoff": False,
        "runtime_registration": runtime_registration,
    }


def validate_runtime_registration(
    contract: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any] | None:
    registration = contract.get("runtime_registration")
    if registration is None:
        return None
    target = dict(contract["next_specialist"])
    state_root = path_value(contract, "runtime_registration", "state_root")
    receipt_path = state_root / (
        f"{str(target['id'])}-runtime-registration-v1.json"
    )
    receipt = read_json(receipt_path)
    row = dict(receipt.get("runtime_row") or {})
    selector = path_value(contract, "runtime_registration", "selector_env")
    registration_contract = dict(contract["runtime_registration"])
    expected_runtime_tree = (
        path_value(contract, "runtime_registration", "activated_runtime_tree")
        if registration_contract.get("inactive_tree_candidate") is not None
        else path_value(contract, "runtime_registration", "runtime_tree")
    )
    selected_rows = [
        line
        for line in selector.read_text(encoding="utf-8").splitlines()
        if line.startswith("POKEBOT_ACTIVE_SPECIALIST=")
    ]
    if (
        receipt.get("schema") != "poke_bot.specialist_runtime_registration/v1"
        or receipt.get("specialist_id") != target["id"]
        or row.get("initial_checkpoint_sha256")
        != str(bootstrap["checkpoint_digest"]).removeprefix("sha256:")
        or row.get("expert_manifest")
        != str(path_value(contract, "next_specialist", "expert_corpus"))
        or row.get("matchup_runtime_tree") != str(expected_runtime_tree)
        or row.get("matchup_runtime_tree_sha256")
        != sha256(expected_runtime_tree).removeprefix("sha256:")
        or selected_rows != [f"POKEBOT_ACTIVE_SPECIALIST={target['id']}"]
        or sha256(path_value(contract, "runtime_registration", "runtime_registry"))
        != receipt.get("runtime_registry_sha256")
        or sha256(selector) != receipt.get("selector_env_sha256")
    ):
        raise RuntimeError("next specialist runtime registration changed")
    return receipt


def _semantic_activation_identity(identity: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(identity))
    registration = normalized.get("runtime_registration")
    if isinstance(registration, dict):
        registration.pop("created_at_utc", None)
    return normalized


def write_or_validate_activation(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    expected = canonical_digest(identity)
    if path.is_file():
        receipt = read_json(path)
        existing_identity = dict(receipt.get("identity") or {})
        if (
            receipt.get("schema") != ACTIVATION_SCHEMA
            or receipt.get("status") != "ready"
            or _semantic_activation_identity(existing_identity)
            != _semantic_activation_identity(identity)
            or receipt.get("identity_sha256")
            != canonical_digest(existing_identity)
        ):
            raise RuntimeError("existing activation receipt identity changed")
        return receipt
    receipt = {
        "schema": ACTIVATION_SCHEMA,
        "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "identity_sha256": expected,
    }
    atomic_json(path, receipt, immutable=True)
    return receipt


def save_state(
    contract: dict[str, Any],
    phase: str,
    contract_digest: str,
    **extra: Any,
) -> None:
    path = path_value(contract, "paths", "state")
    previous = read_json(path) if path.is_file() else {}
    if previous and (
        previous.get("schema") != STATE_SCHEMA
        or previous.get("contract_sha256") != contract_digest
    ):
        raise RuntimeError("existing handoff state belongs to another contract")
    atomic_json(
        path,
        {
            **previous,
            "schema": STATE_SCHEMA,
            "contract_sha256": contract_digest,
            "phase": phase,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **extra,
        },
    )


def deterministic_evidence(
    contract_path: Path,
) -> tuple[
    dict[str, Any],
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    contract, digest = load_contract(contract_path)
    source = (
        validate_saved_preflight_source(contract, digest)
        or validate_source(contract)
    )
    core = validate_core(contract)
    corpus = validate_corpus(contract)
    bootstrap = validate_bootstrap(contract, core, corpus)
    return contract, digest, source, core, corpus, bootstrap


def verify(contract_path: Path) -> dict[str, Any]:
    contract_path = Path(contract_path).expanduser().resolve()
    contract, digest = load_contract(contract_path)
    state_path = path_value(contract, "paths", "state")
    state = read_json(state_path) if state_path.is_file() else {}
    recorded = (
        state.get("schema") == STATE_SCHEMA
        and state.get("contract_sha256") == digest
        and isinstance(state.get("source_specialist"), dict)
        and isinstance(state.get("shared_deck_agnostic_core"), dict)
        and isinstance(state.get("next_specialist_corpus"), dict)
        and isinstance(state.get("next_specialist_splus_gate"), dict)
    )
    if recorded:
        source = dict(state["source_specialist"])
        core = dict(state["shared_deck_agnostic_core"])
        corpus = dict(state["next_specialist_corpus"])
        next_gate = dict(state["next_specialist_splus_gate"])
        bootstrap = validate_bootstrap(contract, core, corpus)
    else:
        (
            contract,
            digest,
            source,
            core,
            corpus,
            bootstrap,
        ) = deterministic_evidence(contract_path)
        next_gate = validate_next_specialist_gate(contract, source)
    source_service = str(contract["source_specialist"]["training_service"])
    if service_active(source_service):
        raise RuntimeError("completed source specialist is active; refusing overlap")
    target_service = str(contract["next_specialist"]["training_service"])
    registration = validate_runtime_registration(contract, bootstrap)
    identity = activation_identity(
        contract_path,
        digest,
        source,
        core,
        corpus,
        bootstrap,
        next_gate,
        target_service,
        registration,
    )
    receipt = read_json(path_value(contract, "next_specialist", "activation_receipt"))
    receipt_identity = dict(receipt.get("identity") or {})
    if (
        receipt.get("schema") != ACTIVATION_SCHEMA
        or receipt.get("status") != "ready"
        or _semantic_activation_identity(receipt_identity)
        != _semantic_activation_identity(identity)
        or receipt.get("identity_sha256")
        != canonical_digest(receipt_identity)
    ):
        raise RuntimeError("activation receipt is absent or changed")
    return receipt


def run(contract_path: Path) -> int:
    contract_path = Path(contract_path).expanduser().resolve()
    contract, digest = load_contract(contract_path)
    lock_path = path_value(contract, "paths", "lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("sequential specialist handoff is already running") from exc

        state_path = path_value(contract, "paths", "state")
        previous = read_json(state_path) if state_path.is_file() else {}
        if previous and (
            previous.get("schema") != STATE_SCHEMA
            or previous.get("contract_sha256") != digest
        ):
            raise RuntimeError("existing handoff state belongs to another contract")
        resumable_preflight = (
            str(previous.get("phase") or "")
            in {
                "preflight_verified",
                "next_specialist_bootstrap_frozen",
                "next_specialist_runtime_tree_bound",
                "next_specialist_runtime_registered",
                "next_specialist_rl_armed",
            }
            and isinstance(previous.get("source_specialist"), dict)
            and isinstance(previous.get("shared_deck_agnostic_core"), dict)
            and isinstance(previous.get("next_specialist_corpus"), dict)
            and isinstance(previous.get("next_specialist_gate_materialization"), dict)
            and isinstance(previous.get("next_specialist_splus_gate"), dict)
        )
        source_service = str(contract["source_specialist"]["training_service"])
        if service_active(source_service):
            raise RuntimeError("source specialist is active; refusing overlap")
        if resumable_preflight:
            source = dict(previous["source_specialist"])
            materialized_gate = dict(
                previous["next_specialist_gate_materialization"]
            )
            next_gate = dict(previous["next_specialist_splus_gate"])
            core = dict(previous["shared_deck_agnostic_core"])
            corpus = dict(previous["next_specialist_corpus"])
        else:
            source = validate_source(contract)
            materialized_gate = materialize_from_contract(contract, source)
            next_gate = validate_next_specialist_gate(contract, source)
            core = validate_core(contract)
            corpus = validate_corpus(contract)
            save_state(
                contract,
                "preflight_verified",
                digest,
                source_specialist=source,
                shared_deck_agnostic_core=core,
                next_specialist_corpus=corpus,
                next_specialist_gate_materialization=materialized_gate,
                next_specialist_splus_gate=next_gate,
            )

        run_checked(bootstrap_command(contract))
        bootstrap = validate_bootstrap(contract, core, corpus)
        save_state(
            contract,
            "next_specialist_bootstrap_frozen",
            digest,
            next_specialist_bootstrap=bootstrap,
        )

        registration_contract = contract.get("runtime_registration")
        registration = None
        if registration_contract is not None:
            target = dict(contract["next_specialist"])
            family = path_value(contract, "paths", "registry_root") / str(
                target["family_name"]
            )
            runtime_tree = prepare_runtime_tree(contract, bootstrap)
            save_state(
                contract,
                "next_specialist_runtime_tree_bound",
                digest,
                runtime_tree=str(runtime_tree),
                runtime_tree_sha256=sha256(runtime_tree),
                runtime_tree_checkpoint_digest=bootstrap["checkpoint_digest"],
            )
            registration = register_specialist_runtime(
                specialist_id=str(target["id"]),
                family=family,
                expert=path_value(contract, "next_specialist", "expert_corpus"),
                runtime_tree=runtime_tree,
                runtime_registry=path_value(
                    contract, "runtime_registration", "runtime_registry"
                ),
                selector_env=path_value(
                    contract, "runtime_registration", "selector_env"
                ),
                state_root=path_value(
                    contract, "runtime_registration", "state_root"
                ),
                run_name=str(registration_contract["run_name"]),
                handoff_service=str(
                    registration_contract["handoff_service"]
                ),
                minimum_decisions=int(contract["training"]["minimum_decisions"]),
                required_target_coverage=tuple(
                    str(value)
                    for value in contract["training"].get(
                        "required_target_coverage", ()
                    )
                ),
            )
            registration = validate_runtime_registration(contract, bootstrap)
            save_state(
                contract,
                "next_specialist_runtime_registered",
                digest,
                runtime_registration=registration,
            )

        target_service = str(contract["next_specialist"]["training_service"])
        identity = activation_identity(
            contract_path,
            digest,
            source,
            core,
            corpus,
            bootstrap,
            next_gate,
            target_service,
            registration,
        )
        activation_path = path_value(
            contract, "next_specialist", "activation_receipt"
        )
        activation = write_or_validate_activation(activation_path, identity)
        save_state(
            contract,
            "next_specialist_rl_armed",
            digest,
            activation_receipt=str(activation_path),
            activation_identity_sha256=activation["identity_sha256"],
        )
        verify(contract_path)
        run_checked(["/usr/bin/systemctl", "--user", "start", target_service])
        run_checked(
            ["/usr/bin/systemctl", "--user", "is-active", "--quiet", target_service]
        )
        if registration_contract is not None:
            gate_handler_service = str(
                registration_contract["gate_handler_service"]
            )
            # The watcher resolves its run directory from the same selector and
            # registry that were atomically updated above.  Restart it only
            # after the new trainer is confirmed active so it cannot retain the
            # completed specialist's run directory across a handoff.
            run_checked(
                [
                    "/usr/bin/systemctl",
                    "--user",
                    "restart",
                    gate_handler_service,
                ]
            )
            run_checked(
                [
                    "/usr/bin/systemctl",
                    "--user",
                    "is-active",
                    "--quiet",
                    gate_handler_service,
                ]
            )
        save_state(
            contract,
            "next_specialist_rl_started",
            digest,
            service=target_service,
            gate_handler_service=(
                str(registration_contract["gate_handler_service"])
                if registration_contract is not None
                else None
            ),
            activation_identity_sha256=activation["identity_sha256"],
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.contract), indent=2, sort_keys=True), flush=True)
        return 0
    return run(args.contract)


if __name__ == "__main__":
    raise SystemExit(main())
