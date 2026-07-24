#!/usr/bin/env python3
"""Fail-closed Alakazam pass -> distilled core -> next-specialist handoff.

This process is intentionally separate from ``handle_passed_gate.py``.  It
does not submit anything, mutate the passed model, or select a checkpoint.  It
continues after the exact gate and immutable freeze. Kaggle copies may remain
in the persistent queue; delayed submissions never block core derivation or
the next specialist. The current canonical next specialist is Trevenant.
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
from scripts.archive_passed_system_to_elmo import SCHEMA as ARCHIVE_SCHEMA  # noqa: E402
from scripts.handle_passed_gate import validate_exact_pass  # noqa: E402
from scripts.run_passed_alakazam_core_distillation import (  # noqa: E402
    FAMILY as CORE_FAMILY,
    TARGETS as CORE_TARGETS,
    _validate_balanced_manifest,
)
from scripts.run_starmie_expert_bootstrap import TARGETS as STARMIE_TARGETS  # noqa: E402


CONTRACT_SCHEMA = "poke_bot.post_alakazam_specialist_handoff_contract/v2"
STATE_SCHEMA = "poke_bot.post_alakazam_specialist_handoff_state/v2"
ACTIVATION_SCHEMA = "poke_bot.specialist_rl_activation/v2"
NEXT_SPECIALIST = "hops-trevenant"
STARMIE_FAMILY = f"{NEXT_SPECIALIST}_expert_bootstrap_from_distilled_core_v1"
SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
SAFE_SERVICE_RE = re.compile(r"[a-zA-Z0-9_.@-]+\.service")
SAFE_REMOTE_PATH_RE = re.compile(r"/[a-zA-Z0-9_./-]+")
ALLOWED_HANDLER_PHASES = {
    "two_submissions_queued",
    "two_submissions_succeeded",
    "waiting_for_terminal_trainer_before_handoff",
    "complete_handoff_started",
}
PHASES = (
    "preflight_verified",
    "elmo_archive_verified",
    "protected_corpora_verified",
    "distilled_core_frozen",
    "next_specialist_bootstrap_frozen",
    "next_specialist_rl_armed",
    "next_specialist_rl_started",
)


def read_json(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"required JSON is missing/corrupt: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON is not an object: {path}")
    return value


def atomic_json(
    path: Path, payload: dict[str, Any], *, immutable: bool = False
) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if immutable:
        path.chmod(0o444)


def canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def require_digest(value: Any, label: str) -> str:
    digest = str(value or "")
    if not SHA256_RE.fullmatch(digest):
        raise RuntimeError(f"{label} is missing an exact sha256 digest")
    return digest


def required_path(contract: dict[str, Any], name: str) -> Path:
    raw = str((contract.get("paths") or {}).get(name) or "")
    path = Path(raw).expanduser()
    if not raw or not path.is_absolute():
        raise RuntimeError(f"contract path {name!r} must be absolute")
    return path.resolve()


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path).expanduser().resolve()
    contract = read_json(path)
    scripts = dict(contract.get("scripts") or {})
    families = dict(contract.get("families") or {})
    services = dict(contract.get("services") or {})
    expected_scripts = {
        "archive": "scripts/archive_passed_system_to_elmo.py",
        "core_distillation": "scripts/run_passed_alakazam_core_distillation.py",
        "starmie_bootstrap": "scripts/run_specialist_expert_bootstrap.py",
    }
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("deployment_mode") != "exact_pass_gated_automatic_handoff"
        or contract.get("production_mutation_policy")
        != "forbidden_before_immutable_activation_receipt"
        or contract.get("automatic_next_specialist_start_after_activation_receipt")
        is not True
        or contract.get("next_specialist_id") != NEXT_SPECIALIST
        or int(contract.get("required_kaggle_copies", -1)) != 2
        or contract.get("submission_completion_blocks_handoff") is not False
        or not str(contract.get("gate_marker_name") or "").startswith(
            "SPECIALIST_GATE_PASSED"
        )
        or list(contract.get("phases") or []) != list(PHASES)
        or scripts != expected_scripts
        or families
        != {"distilled_core": CORE_FAMILY, "starmie_bootstrap": STARMIE_FAMILY}
    ):
        raise RuntimeError("post-pass handoff contract semantics changed")
    service = str(services.get("starmie_rl") or "")
    source_service = str(services.get("passed_alakazam_rl") or "")
    if (
        service != "pokebot-pure-rl-trevenant-staged.service"
        or not SAFE_SERVICE_RE.fullmatch(service)
        or not SAFE_SERVICE_RE.fullmatch(source_service)
    ):
        raise RuntimeError("handoff service identity is invalid")
    for name in (
        "python",
        "source_root",
        "handler_state",
        "passed_family",
        "source_run_dir",
        "gate_contract",
        "submission_root",
        "source_deployment_root",
        "engine",
        "archive_staging_root",
        "archive_receipt",
        "registry_root",
        "core_corpus",
        "core_ready",
        "core_run_dir",
        "core_cpu_pack_root",
        "starmie_corpus",
        "starmie_ready",
        "starmie_run_dir",
        "starmie_cpu_pack_root",
        "activation_receipt",
        "state",
        "lock",
    ):
        required_path(contract, name)
    archive = dict(contract.get("archive") or {})
    if archive.get("destination_host") != "elmo" or not SAFE_REMOTE_PATH_RE.fullmatch(
        str(archive.get("destination_root") or "")
    ):
        raise RuntimeError("Elmo archive destination contract is invalid")
    training = dict(contract.get("training") or {})
    for key in ("core", "starmie"):
        row = dict(training.get(key) or {})
        if (
            not str(row.get("run_name") or "").strip()
            or int(row.get("epochs", 0)) <= 0
            or int(row.get("patience", 0)) <= 0
            or int(row.get("min_decisions", 0)) <= 0
            or int(row.get("batch_size", 0)) <= 0
        ):
            raise RuntimeError(f"{key} training contract is incomplete")
    return contract, sha256(path)


def validate_corpora(contract: dict[str, Any]) -> dict[str, Any]:
    core_cfg = dict((contract.get("training") or {}).get("core") or {})
    starmie_cfg = dict((contract.get("training") or {}).get("starmie") or {})
    core_pointer = required_path(contract, "core_corpus")
    starmie_pointer = required_path(contract, "starmie_corpus")
    _validate_balanced_manifest(
        core_pointer, min_decisions=int(core_cfg["min_decisions"])
    )
    core = resolve_expert_manifest(
        core_pointer,
        min_decisions=int(core_cfg["min_decisions"]),
        require_protected=True,
        required_compact_mode="temporal-expert-v1",
        required_max_context=320,
        required_target_coverage=CORE_TARGETS,
    )
    starmie = resolve_expert_manifest(
        starmie_pointer,
        min_decisions=int(starmie_cfg["min_decisions"]),
        require_protected=True,
        required_archetype=NEXT_SPECIALIST,
        required_compact_mode="temporal-expert-v1",
        required_max_context=320,
        required_target_coverage=STARMIE_TARGETS,
    )
    return {
        "core": {
            "pointer": str(core_pointer),
            "pointer_sha256": sha256(core_pointer),
            "manifest": core.path,
            "manifest_sha256": core.digest,
            "records": core.records,
            "decisions": core.decisions,
        },
        "starmie": {
            "pointer": str(starmie_pointer),
            "pointer_sha256": sha256(starmie_pointer),
            "manifest": starmie.path,
            "manifest_sha256": starmie.digest,
            "records": starmie.records,
            "decisions": starmie.decisions,
        },
    }


def validate_pass_and_uploads(contract: dict[str, Any]) -> dict[str, Any]:
    plan = validate_exact_pass(
        required_path(contract, "source_run_dir"),
        required_path(contract, "gate_contract"),
        marker_name=str(contract["gate_marker_name"]),
    )
    frozen = verify_frozen_model(required_path(contract, "passed_family"))
    handler_path = required_path(contract, "handler_state")
    handler = read_json(handler_path)
    bundle = dict(handler.get("submission_bundle") or {})
    bundle_digest = require_digest(bundle.get("sha256"), "submission bundle")
    queued_mode = handler.get("submission_mode") == "queue_and_continue"
    queued = [dict(row) for row in (handler.get("queued_submissions") or [])]
    if (
        handler.get("phase") not in ALLOWED_HANDLER_PHASES
        or handler.get("gate") != plan
        or handler.get("approved_submission_count") != 2
        or handler.get("automatic_retries") is not False
        or frozen.get("checkpoint_digest") != plan.get("checkpoint_digest")
        or (frozen.get("provenance") or {}).get("contract_sha256")
        != plan.get("contract_sha256")
        or (frozen.get("provenance") or {}).get("commit_digest")
        != plan.get("commit_digest")
        or (bundle.get("contents") or {}).get("model_sha256")
        != frozen.get("checkpoint_digest")
    ):
        raise RuntimeError("exact passed model / handler identity mismatch")
    if queued_mode:
        if (
            len(queued) != 2
            or [int(row.get("copy_number", -1)) for row in queued] != [1, 2]
            or any(row.get("queue_status") != "pending" for row in queued)
            or any(
                row.get("checkpoint_checksum") != frozen.get("checkpoint_digest")
                for row in queued
            )
            or handler.get("all_submissions_succeeded") is not False
        ):
            raise RuntimeError("pending Kaggle copies are not exact/checkpoint-bound")
        for row in queued:
            copy_path = Path(str(row.get("file") or "")).expanduser().resolve()
            if (
                not copy_path.is_file()
                or sha256(copy_path) != row.get("file_sha256")
                or row.get("file_sha256") != bundle_digest
            ):
                raise RuntimeError("pending Kaggle copy bytes changed")
        attempts: list[dict[str, Any]] = []
    else:
        if (
            handler.get("all_submissions_succeeded") is not True
            or handler.get("successful_submission_count") != 2
        ):
            raise RuntimeError("completed submission mode lacks two successes")
        from scripts.handle_passed_gate import (  # local to keep queue path inert
            validate_two_successful_submission_attempts,
        )

        attempts = validate_two_successful_submission_attempts(
            handler.get("submission_attempts"),
            expected_bundle_sha256=bundle_digest,
        )
    bundle_path = Path(str(bundle.get("path") or "")).expanduser().resolve()
    attestation_path = Path(str(bundle.get("attestation") or "")).expanduser().resolve()
    if not bundle_path.is_file() or sha256(bundle_path) != bundle_digest:
        raise RuntimeError("submission bundle bytes are missing or changed")
    attestation = read_json(attestation_path)
    if (
        attestation.get("file_sha256") != bundle_digest
        or attestation.get("go_first_if_offered") is not True
    ):
        raise RuntimeError("submission bundle attestation is missing or changed")
    evidence = {
        "checkpoint_digest": frozen["checkpoint_digest"],
        "passed_family_manifest_sha256": sha256(
            required_path(contract, "passed_family") / "manifest.json"
        ),
        "gate_id": plan["gate_id"],
        "gate_contract_sha256": plan["contract_sha256"],
        "commit_digest": plan["commit_digest"],
        "commit_file_sha256": plan["commit_file_sha256"],
        "exact_result_pointer_sha256": plan["exact_result_pointer_sha256"],
        "handler_state": str(handler_path),
        "submission_bundle_sha256": bundle_digest,
        "submission_attestation_sha256": sha256(attestation_path),
        "submission_completion_blocks_handoff": False,
        "submission_status": "pending" if queued_mode else "complete",
        "pending_submissions": queued if queued_mode else [],
        "successful_uploads": [
            {
                "slot": int(row["slot"]),
                "nonce": str(row["nonce"]),
                "file_sha256": str(row["file_sha256"]),
                "consumed_authorization_digests": dict(
                    row["consumed_authorization_digests"]
                ),
                "guard_attempt_receipt_digests": dict(
                    row["guard_attempt_receipt_digests"]
                ),
            }
            for row in attempts
        ],
    }
    evidence["identity_sha256"] = canonical_digest(evidence)
    return evidence


def run_checked(argv: list[str]) -> None:
    completed = subprocess.run(argv, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: " + " ".join(argv)
        )


def capture_checked(argv: list[str]) -> str:
    completed = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(argv)}\n"
            f"{completed.stdout or ''}"
        )
    return completed.stdout or ""


def archive_command(contract: dict[str, Any]) -> list[str]:
    archive = dict(contract["archive"])
    return [
        str(required_path(contract, "python")),
        "-u",
        str(ROOT / "scripts/archive_passed_system_to_elmo.py"),
        "--handler-state",
        str(required_path(contract, "handler_state")),
        "--frozen-family",
        str(required_path(contract, "passed_family")),
        "--run-dir",
        str(required_path(contract, "source_run_dir")),
        "--submission-root",
        str(required_path(contract, "submission_root")),
        "--source-root",
        str(required_path(contract, "source_root")),
        "--deployment-root",
        str(required_path(contract, "source_deployment_root")),
        "--engine",
        str(required_path(contract, "engine")),
        "--staging-root",
        str(required_path(contract, "archive_staging_root")),
        "--receipt",
        str(required_path(contract, "archive_receipt")),
        "--destination-host",
        str(archive["destination_host"]),
        "--destination-root",
        str(archive["destination_root"]),
    ]


def validate_archive(
    contract: dict[str, Any],
    pass_evidence: dict[str, Any],
    *,
    verify_remote: bool,
) -> dict[str, Any]:
    path = required_path(contract, "archive_receipt")
    receipt = read_json(path)
    destination = str(receipt.get("destination") or "")
    expected_root = str(contract["archive"]["destination_root"]).rstrip("/") + "/"
    remote_manifest_digest = require_digest(
        receipt.get("local_manifest_sha256"), "Elmo archive manifest"
    )
    if (
        receipt.get("schema") != ARCHIVE_SCHEMA
        or receipt.get("status") != "verified"
        or receipt.get("destination_host") != contract["archive"]["destination_host"]
        or not destination.startswith(expected_root)
        or not SAFE_REMOTE_PATH_RE.fullmatch(destination)
        or receipt.get("checkpoint_digest") != pass_evidence["checkpoint_digest"]
        or receipt.get("submitted_slots") != [1, 2]
        or receipt.get("successful_submission_slots") != [1, 2]
        or receipt.get("all_submissions_succeeded") is not True
        or receipt.get("automatic_retries") is not False
    ):
        raise RuntimeError("Elmo archive receipt is incomplete or identity-mismatched")
    if verify_remote:
        host = str(receipt["destination_host"])
        manifest_path = destination + "/ARCHIVE_MANIFEST.json"
        output = capture_checked(
            ["/usr/bin/ssh", host, "sha256sum", manifest_path]
        ).strip()
        actual = "sha256:" + output.split()[0] if output else ""
        if actual != remote_manifest_digest:
            raise RuntimeError("Elmo archive manifest digest mismatch")
        checkpoint = capture_checked(
            ["/usr/bin/ssh", host, "cat", destination + "/VERIFIED_CHECKPOINT_DIGEST"]
        ).strip()
        if checkpoint != pass_evidence["checkpoint_digest"]:
            raise RuntimeError("Elmo archive checkpoint identity mismatch")
    return {
        "receipt": str(path),
        "receipt_sha256": sha256(path),
        "destination_host": receipt["destination_host"],
        "destination": destination,
        "remote_manifest_sha256": remote_manifest_digest,
        "checkpoint_digest": receipt["checkpoint_digest"],
    }


def nonblocking_archive_evidence(
    contract: dict[str, Any],
    pass_evidence: dict[str, Any],
    *,
    verify_remote: bool,
) -> dict[str, Any]:
    """Return archive proof, or an explicit non-blocking pending record."""

    if pass_evidence.get("submission_status") == "pending":
        return {
            "status": "pending_until_submission_queue_completes",
            "blocks_core_or_specialist": False,
            "checkpoint_digest": pass_evidence["checkpoint_digest"],
        }
    return validate_archive(
        contract,
        pass_evidence,
        verify_remote=verify_remote,
    )


def core_command(contract: dict[str, Any]) -> list[str]:
    cfg = dict(contract["training"]["core"])
    return [
        str(required_path(contract, "python")),
        "-u",
        str(ROOT / "scripts/run_passed_alakazam_core_distillation.py"),
        "--core-corpus",
        str(required_path(contract, "core_corpus")),
        "--passed-family",
        str(required_path(contract, "passed_family")),
        "--registry-root",
        str(required_path(contract, "registry_root")),
        "--ready",
        str(required_path(contract, "core_ready")),
        "--run-name",
        str(cfg["run_name"]),
        "--run-dir",
        str(required_path(contract, "core_run_dir")),
        "--epochs",
        str(int(cfg["epochs"])),
        "--patience",
        str(int(cfg["patience"])),
        "--min-decisions",
        str(int(cfg["min_decisions"])),
        "--batch-size",
        str(int(cfg["batch_size"])),
        "--cpu-pack-root",
        str(required_path(contract, "core_cpu_pack_root")),
    ]


def starmie_command(contract: dict[str, Any]) -> list[str]:
    cfg = dict(contract["training"]["starmie"])
    return [
        str(required_path(contract, "python")),
        "-u",
        str(ROOT / "scripts/run_specialist_expert_bootstrap.py"),
        "--expert-corpus",
        str(required_path(contract, "starmie_corpus")),
        "--archetype",
        NEXT_SPECIALIST,
        "--family",
        STARMIE_FAMILY,
        "--core-family",
        str(required_path(contract, "registry_root") / CORE_FAMILY),
        "--registry-root",
        str(required_path(contract, "registry_root")),
        "--ready",
        str(required_path(contract, "starmie_ready")),
        "--run-name",
        str(cfg["run_name"]),
        "--run-dir",
        str(required_path(contract, "starmie_run_dir")),
        "--epochs",
        str(int(cfg["epochs"])),
        "--patience",
        str(int(cfg["patience"])),
        "--min-decisions",
        str(int(cfg["min_decisions"])),
        "--batch-size",
        str(int(cfg["batch_size"])),
        "--cpu-pack-root",
        str(required_path(contract, "starmie_cpu_pack_root")),
    ]


def validate_core_ready(
    contract: dict[str, Any], corpora: dict[str, Any], passed_digest: str
) -> dict[str, Any]:
    family = required_path(contract, "registry_root") / CORE_FAMILY
    frozen = verify_frozen_model(family)
    ready_path = required_path(contract, "core_ready")
    ready = read_json(ready_path)
    provenance = dict(frozen.get("provenance") or {})
    if (
        ready.get("schema") != "poke_bot.distilled_deck_agnostic_core_ready/v1"
        or ready.get("status") != "ready"
        or frozen.get("family") != CORE_FAMILY
        or ready.get("checkpoint_digest") != frozen.get("checkpoint_digest")
        or ready.get("passed_alakazam_digest") != passed_digest
        or ready.get("core_manifest_sha256") != corpora["core"]["manifest_sha256"]
        or provenance.get("distillation_source_digest") != passed_digest
        or (provenance.get("core_manifest") or {}).get("digest")
        != corpora["core"]["manifest_sha256"]
        or provenance.get("all_auxiliary_heads_trained") is not True
    ):
        raise RuntimeError("protected distilled core readiness identity mismatch")
    return {
        "family": str(family),
        "family_manifest_sha256": sha256(family / "manifest.json"),
        "protection_sha256": sha256(family / FAMILY_MARKER),
        "checkpoint_digest": frozen["checkpoint_digest"],
        "ready": str(ready_path),
        "ready_sha256": sha256(ready_path),
        "corpus_manifest_sha256": corpora["core"]["manifest_sha256"],
    }


def validate_starmie_ready(
    contract: dict[str, Any], corpora: dict[str, Any], core_digest: str
) -> dict[str, Any]:
    family = required_path(contract, "registry_root") / STARMIE_FAMILY
    frozen = verify_frozen_model(family)
    ready_path = required_path(contract, "starmie_ready")
    ready = read_json(ready_path)
    provenance = dict(frozen.get("provenance") or {})
    if (
        ready.get("schema") != "poke_bot.specialist_expert_bootstrap_ready/v1"
        or ready.get("status") != "ready"
        or frozen.get("family") != STARMIE_FAMILY
        or ready.get("checkpoint_digest") != frozen.get("checkpoint_digest")
        or ready.get("core_checkpoint_digest") != core_digest
        or ready.get("expert_manifest_sha256")
        != corpora["starmie"]["manifest_sha256"]
        or ready.get("acting_seat_archetype") != NEXT_SPECIALIST
        or provenance.get("initialized_from_digest") != core_digest
        or (provenance.get("expert_manifest") or {}).get("digest")
        != corpora["starmie"]["manifest_sha256"]
        or provenance.get("acting_seat_archetype") != NEXT_SPECIALIST
        or provenance.get("all_auxiliary_heads_trained") is not True
    ):
        raise RuntimeError("protected next-specialist readiness identity mismatch")
    return {
        "family": str(family),
        "family_manifest_sha256": sha256(family / "manifest.json"),
        "protection_sha256": sha256(family / FAMILY_MARKER),
        "checkpoint_digest": frozen["checkpoint_digest"],
        "core_checkpoint_digest": core_digest,
        "ready": str(ready_path),
        "ready_sha256": sha256(ready_path),
        "corpus_manifest_sha256": corpora["starmie"]["manifest_sha256"],
    }


def activation_identity(
    contract_path: Path,
    contract_sha256: str,
    contract: dict[str, Any],
    pass_evidence: dict[str, Any],
    corpora: dict[str, Any],
    archive: dict[str, Any],
    core: dict[str, Any],
    starmie: dict[str, Any],
) -> dict[str, Any]:
    return {
        "contract": str(Path(contract_path).expanduser().resolve()),
        "contract_sha256": contract_sha256,
        "deployment_mode": "exact_pass_gated_automatic_handoff",
        "exact_pass_and_submission_queue": pass_evidence,
        "corpora": corpora,
        "elmo_archive": archive,
        "distilled_core": core,
        "next_specialist_id": NEXT_SPECIALIST,
        "next_specialist_bootstrap": starmie,
        "service": contract["services"]["starmie_rl"],
    }


def write_or_validate_activation(
    contract_path: Path,
    contract_sha256: str,
    contract: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    path = required_path(contract, "activation_receipt")
    expected_digest = canonical_digest(identity)
    if path.exists():
        receipt = read_json(path)
        if (
            receipt.get("schema") != ACTIVATION_SCHEMA
            or receipt.get("status") != "ready"
            or receipt.get("identity") != identity
            or receipt.get("identity_sha256") != expected_digest
        ):
            raise RuntimeError("existing specialist activation receipt identity changed")
        return receipt
    receipt = {
        "schema": ACTIVATION_SCHEMA,
        "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "identity_sha256": expected_digest,
    }
    atomic_json(path, receipt, immutable=True)
    return receipt


def deterministic_evidence(
    contract_path: Path, *, verify_remote: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract, contract_sha256 = load_contract(contract_path)
    passed = validate_pass_and_uploads(contract)
    corpora = validate_corpora(contract)
    archive = nonblocking_archive_evidence(
        contract, passed, verify_remote=verify_remote
    )
    core = validate_core_ready(contract, corpora, passed["checkpoint_digest"])
    starmie = validate_starmie_ready(contract, corpora, core["checkpoint_digest"])
    return contract, activation_identity(
        contract_path,
        contract_sha256,
        contract,
        passed,
        corpora,
        archive,
        core,
        starmie,
    )


def assert_source_inactive(contract: dict[str, Any]) -> None:
    service = str(contract["services"]["passed_alakazam_rl"])
    completed = subprocess.run(
        ["/usr/bin/systemctl", "--user", "is-active", "--quiet", service],
        check=False,
    )
    if completed.returncode == 0:
        raise RuntimeError("passed Alakazam trainer is still active; refusing overlap")


def verify_rl_start(
    contract_path: Path, *, check_source: bool = True
) -> dict[str, Any]:
    contract, identity = deterministic_evidence(contract_path, verify_remote=False)
    receipt = read_json(required_path(contract, "activation_receipt"))
    if (
        receipt.get("schema") != ACTIVATION_SCHEMA
        or receipt.get("status") != "ready"
        or receipt.get("identity") != identity
        or receipt.get("identity_sha256") != canonical_digest(identity)
    ):
        raise RuntimeError("specialist RL activation evidence is missing or changed")
    if check_source:
        assert_source_inactive(contract)
    return receipt


def save_state(
    contract: dict[str, Any], phase: str, input_identity: dict[str, Any], **extra: Any
) -> None:
    path = required_path(contract, "state")
    previous = read_json(path) if path.is_file() else {}
    digest = canonical_digest(input_identity)
    if previous and (
        previous.get("schema") != STATE_SCHEMA
        or previous.get("input_identity_sha256") != digest
    ):
        raise RuntimeError("existing handoff state belongs to different inputs")
    atomic_json(
        path,
        {
            **previous,
            "schema": STATE_SCHEMA,
            "phase": phase,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_identity": input_identity,
            "input_identity_sha256": digest,
            **extra,
        },
    )


def run_pipeline(contract_path: Path) -> int:
    contract_path = Path(contract_path).expanduser().resolve()
    contract, contract_sha256 = load_contract(contract_path)
    lock_path = required_path(contract, "lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("post-pass handoff is already running") from exc

        passed = validate_pass_and_uploads(contract)
        inputs = {
            "contract_sha256": contract_sha256,
            "pass_and_submission_queue_identity_sha256": passed["identity_sha256"],
        }
        save_state(contract, "preflight_verified", inputs)

        if passed.get("submission_status") == "complete":
            run_checked(archive_command(contract))
        archive = nonblocking_archive_evidence(
            contract, passed, verify_remote=True
        )
        save_state(contract, "elmo_archive_verified", inputs, elmo_archive=archive)

        corpora = validate_corpora(contract)
        save_state(
            contract,
            "protected_corpora_verified",
            inputs,
            elmo_archive=archive,
            corpora=corpora,
        )

        run_checked(core_command(contract))
        core = validate_core_ready(contract, corpora, passed["checkpoint_digest"])
        save_state(contract, "distilled_core_frozen", inputs, distilled_core=core)

        run_checked(starmie_command(contract))
        starmie = validate_starmie_ready(contract, corpora, core["checkpoint_digest"])
        save_state(
            contract,
            "next_specialist_bootstrap_frozen",
            inputs,
            distilled_core=core,
            next_specialist_id=NEXT_SPECIALIST,
            next_specialist_bootstrap=starmie,
        )

        identity = activation_identity(
            contract_path,
            contract_sha256,
            contract,
            passed,
            corpora,
            archive,
            core,
            starmie,
        )
        activation = write_or_validate_activation(
            contract_path, contract_sha256, contract, identity
        )
        save_state(
            contract,
            "next_specialist_rl_armed",
            inputs,
            activation_receipt=str(required_path(contract, "activation_receipt")),
            activation_identity_sha256=activation["identity_sha256"],
        )
        verify_rl_start(contract_path)
        service = str(contract["services"]["starmie_rl"])
        run_checked(["/usr/bin/systemctl", "--user", "start", service])
        run_checked(["/usr/bin/systemctl", "--user", "is-active", "--quiet", service])
        save_state(
            contract,
            "next_specialist_rl_started",
            inputs,
            service=service,
            activation_identity_sha256=activation["identity_sha256"],
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "verify-rl-start"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "verify-rl-start":
        receipt = verify_rl_start(args.contract)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        return 0
    return run_pipeline(args.contract)


if __name__ == "__main__":
    raise SystemExit(main())
