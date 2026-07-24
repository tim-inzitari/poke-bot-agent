#!/usr/bin/env python3
"""Owner-accepted Alakazam -> shared core -> Trevenant bootstrap/RL.

This path preserves the measured gate miss. It accepts only the immutable
``accepted_with_waiver`` receipt and the one explicitly authorized Kaggle
submission, then reuses the ordinary protected core and specialist bootstrap
implementations. It never fabricates an exact-gate marker.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.model_registry import sha256, verify_frozen_model  # noqa: E402
from scripts.run_post_alakazam_starmie_handoff import (  # noqa: E402
    core_command,
    starmie_command,
    validate_core_ready,
    validate_corpora,
    validate_starmie_ready,
)


STATE_SCHEMA = "poke_bot.owner_accepted_alakazam_handoff_state/v1"
ACTIVATION_SCHEMA = "poke_bot.specialist_rl_activation/v2"
SOURCE_DIGEST = (
    "sha256:270b5156781b0a95f703abe3e8fe13866d2fbb4c85a8f32534f99af74aece2ea"
)
NEXT_SPECIALIST = "hops-trevenant"
SERVICE = "pokebot-pure-rl-trevenant-staged.service"


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"required JSON is missing/corrupt: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"required JSON is not an object: {path}")
    return value


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any], *, immutable: bool = False) -> None:
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


def _contract() -> dict[str, Any]:
    source = Path("/home/inzi/poke-bot-agent")
    return {
        "training": {
            "core": {
                "run_name": "owner_accepted_alakazam_balanced_core_v1_20260723",
                "epochs": 25,
                "patience": 5,
                "min_decisions": 500_000,
                "batch_size": 12_288,
            },
            "starmie": {
                "run_name": "hops_trevenant_expert_bootstrap_from_core_v1_20260723",
                "epochs": 25,
                "patience": 5,
                "min_decisions": 20_000,
                "batch_size": 12_288,
            },
        },
        "services": {
            "passed_alakazam_rl": "pokebot-pure-rl-alakazam.service",
            "starmie_rl": SERVICE,
        },
        "paths": {
            "python": "/home/inzi/miniconda3/envs/poke-bot-agent/bin/python",
            "passed_family": str(
                source
                / "outputs/pure_rl/_protected/models/"
                "alakazam-owner-accepted-iter39-v1"
            ),
            "registry_root": str(source / "outputs/pure_rl/_protected/models"),
            "core_corpus": str(
                source
                / "data/bootstrap/expert-latest20-20260702-20260721/"
                "core-balanced-v2/PROTECTED_CORE_CORPUS.json"
            ),
            "core_ready": str(
                source
                / "outputs/state/"
                "owner-accepted-alakazam-core-distillation-ready.json"
            ),
            "core_run_dir": str(
                source
                / "outputs/bootstrap/"
                "owner-accepted-alakazam-balanced-core-v1"
            ),
            "core_cpu_pack_root": str(
                source
                / "outputs/bootstrap/cpu-packs/"
                "owner-accepted-alakazam-balanced-core-v1"
            ),
            "starmie_corpus": str(
                source
                / "data/bootstrap/expert-latest20-20260702-20260721/"
                "hops-trevenant-v2/PROTECTED_EXPERT_CORPUS.json"
            ),
            "starmie_ready": str(
                source
                / "outputs/state/hops-trevenant-expert-bootstrap-ready-v1.json"
            ),
            "starmie_run_dir": str(
                source
                / "outputs/bootstrap/"
                "hops-trevenant-expert-bootstrap-from-core-v1"
            ),
            "starmie_cpu_pack_root": str(
                source
                / "outputs/bootstrap/cpu-packs/"
                "hops-trevenant-expert-bootstrap-v1"
            ),
            "activation_receipt": str(
                source
                / "outputs/state/hops-trevenant-specialist-rl-activation-v1.json"
            ),
            "state": str(
                source
                / "outputs/state/post-alakazam-specialist-handoff-v2.json"
            ),
        },
    }


def _path(contract: dict[str, Any], name: str) -> Path:
    return Path(str(contract["paths"][name])).expanduser().resolve()


def _validate_owner_inputs(contract: dict[str, Any]) -> dict[str, Any]:
    source = Path("/home/inzi/poke-bot-agent")
    owner_path = source / "outputs/state/alakazam-owner-acceptance-iter39.json"
    submission_path = source / "outputs/state/alakazam-owner-single-submission.json"
    owner = _read(owner_path)
    submission = _read(submission_path)
    frozen = verify_frozen_model(_path(contract, "passed_family"))
    selected = dict(owner.get("selected_strongest_checkpoint") or {})
    truth = dict(owner.get("truthfulness") or {})
    if (
        owner.get("schema") != "poke_bot.specialist_owner_acceptance/v1"
        or owner.get("status") != "accepted_with_waiver"
        or owner.get("specialist_id") != "alakazam"
        or truth.get("does_not_assert_measured_gate_pass") is not True
        or truth.get("measured_terminal_gate_passed") is not False
        or selected.get("digest") != SOURCE_DIGEST
        or frozen.get("checkpoint_digest") != SOURCE_DIGEST
        or submission.get("schema")
        != "poke_bot.owner_accepted_single_submission/v1"
        or submission.get("status") != "submitted"
        or submission.get("required_copies_for_this_owner_acceptance") != 1
        or submission.get("copy_number") != 1
        or int(submission.get("submission_id") or 0) != 54927679
        or submission.get("checkpoint_checksum") != SOURCE_DIGEST
        or submission.get("automatic_retry") is not False
    ):
        raise RuntimeError("owner acceptance/frozen model/submission identity changed")
    for path_field, digest_field in (
        ("bundle", "bundle_checksum"),
        ("go_first_attestation", "go_first_attestation_checksum"),
        ("guard_attempt_receipt", "guard_attempt_receipt_checksum"),
        ("owner_acceptance_receipt", "owner_acceptance_receipt_checksum"),
        ("frozen_manifest", "frozen_manifest_checksum"),
    ):
        path = Path(str(submission.get(path_field) or "")).expanduser().resolve()
        if not path.is_file() or sha256(path) != submission.get(digest_field):
            raise RuntimeError(f"submission evidence changed: {path_field}")
    attempt = _read(Path(submission["guard_attempt_receipt"]))
    if (
        attempt.get("schema") != "poke_bot.kaggle_submission_attempt/v1"
        or int(attempt.get("returncode", -1)) != 0
        or str((attempt.get("identity") or {}).get("file_sha256") or "")
        != str(submission["bundle_checksum"])
    ):
        raise RuntimeError("Kaggle guard attempt is not a proven success")
    identity = {
        "completion_authority": "accepted_with_waiver",
        "does_not_assert_measured_gate_pass": True,
        "checkpoint_digest": SOURCE_DIGEST,
        "frozen_family": str(_path(contract, "passed_family")),
        "frozen_manifest_sha256": sha256(
            _path(contract, "passed_family") / "manifest.json"
        ),
        "owner_acceptance_receipt": str(owner_path),
        "owner_acceptance_receipt_sha256": sha256(owner_path),
        "single_submission_receipt": str(submission_path),
        "single_submission_receipt_sha256": sha256(submission_path),
        "submission_id": 54927679,
    }
    identity["identity_sha256"] = _canonical_digest(identity)
    return identity


def _save_state(
    contract: dict[str, Any],
    phase: str,
    identity: dict[str, Any],
    **extra: Any,
) -> None:
    path = _path(contract, "state")
    previous = _read(path) if path.is_file() else {}
    if previous and (
        previous.get("schema") != STATE_SCHEMA
        or previous.get("input_identity_sha256") != identity["identity_sha256"]
    ):
        raise RuntimeError("existing owner handoff state belongs to other inputs")
    _atomic_json(
        path,
        {
            **previous,
            "schema": STATE_SCHEMA,
            "phase": phase,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_identity": identity,
            "input_identity_sha256": identity["identity_sha256"],
            **extra,
        },
    )


def _run(argv: list[str]) -> None:
    completed = subprocess.run(argv, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(argv)}"
        )


def _source_inactive() -> None:
    completed = subprocess.run(
        [
            "/usr/bin/systemctl",
            "--user",
            "is-active",
            "--quiet",
            "pokebot-pure-rl-alakazam.service",
        ],
        check=False,
    )
    if completed.returncode == 0:
        raise RuntimeError("Alakazam trainer is active; refusing specialist overlap")


def _activation_identity(
    owner: dict[str, Any],
    corpora: dict[str, Any],
    core: dict[str, Any],
    specialist: dict[str, Any],
) -> dict[str, Any]:
    return {
        "deployment_mode": "owner_accepted_waiver_without_gate_falsification",
        "owner_acceptance_and_single_submission": owner,
        "corpora": corpora,
        "distilled_core": core,
        "next_specialist_id": NEXT_SPECIALIST,
        "next_specialist_bootstrap": specialist,
        "service": SERVICE,
    }


def _write_or_validate_activation(
    contract: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    path = _path(contract, "activation_receipt")
    digest = _canonical_digest(identity)
    if path.is_file():
        receipt = _read(path)
        if (
            receipt.get("schema") != ACTIVATION_SCHEMA
            or receipt.get("status") != "ready"
            or receipt.get("identity") != identity
            or receipt.get("identity_sha256") != digest
        ):
            raise RuntimeError("existing Trevenant activation identity changed")
        return receipt
    receipt = {
        "schema": ACTIVATION_SCHEMA,
        "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "identity_sha256": digest,
    }
    _atomic_json(path, receipt, immutable=True)
    return receipt


def verify() -> dict[str, Any]:
    contract = _contract()
    owner = _validate_owner_inputs(contract)
    corpora = validate_corpora(contract)
    core = validate_core_ready(contract, corpora, SOURCE_DIGEST)
    specialist = validate_starmie_ready(
        contract, corpora, str(core["checkpoint_digest"])
    )
    identity = _activation_identity(owner, corpora, core, specialist)
    receipt = _read(_path(contract, "activation_receipt"))
    if (
        receipt.get("schema") != ACTIVATION_SCHEMA
        or receipt.get("status") != "ready"
        or receipt.get("identity") != identity
        or receipt.get("identity_sha256") != _canonical_digest(identity)
    ):
        raise RuntimeError("Trevenant activation receipt is absent or changed")
    _source_inactive()
    return receipt


def run() -> int:
    contract = _contract()
    owner = _validate_owner_inputs(contract)
    _source_inactive()
    _save_state(contract, "owner_acceptance_verified", owner)

    corpora = validate_corpora(contract)
    _save_state(contract, "protected_corpora_verified", owner, corpora=corpora)

    _run(core_command(contract))
    core = validate_core_ready(contract, corpora, SOURCE_DIGEST)
    _save_state(contract, "distilled_core_frozen", owner, distilled_core=core)

    _run(starmie_command(contract))
    specialist = validate_starmie_ready(
        contract, corpora, str(core["checkpoint_digest"])
    )
    _save_state(
        contract,
        "next_specialist_bootstrap_frozen",
        owner,
        distilled_core=core,
        next_specialist_id=NEXT_SPECIALIST,
        next_specialist_bootstrap=specialist,
    )

    activation_identity = _activation_identity(
        owner, corpora, core, specialist
    )
    activation = _write_or_validate_activation(contract, activation_identity)
    _save_state(
        contract,
        "next_specialist_rl_armed",
        owner,
        activation_receipt=str(_path(contract, "activation_receipt")),
        activation_identity_sha256=activation["identity_sha256"],
    )
    verify()
    _run(["/usr/bin/systemctl", "--user", "start", SERVICE])
    _run(["/usr/bin/systemctl", "--user", "is-active", "--quiet", SERVICE])
    _save_state(
        contract,
        "next_specialist_rl_started",
        owner,
        service=SERVICE,
        activation_identity_sha256=activation["identity_sha256"],
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(), indent=2, sort_keys=True), flush=True)
        return 0
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
