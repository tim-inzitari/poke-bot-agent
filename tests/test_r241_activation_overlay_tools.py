"""Focused unit coverage for r241's create-only activation helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUTH_SCRIPT = ROOT / "scripts/generate_r241_owner_start_authorization.py"
MIRROR_SCRIPT = ROOT / "scripts/install_r241_activation_overlay_mirror.py"
PUBLISHER = ROOT / "scripts/publish_r241_activation_overlay.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object, *, readonly: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    if readonly:
        path.chmod(0o444)
    return path


def _owner_and_manifest(tmp_path: Path) -> tuple[Path, Path, str]:
    owner_sha = "sha256:" + "1" * 64
    owner = _write_json(
        tmp_path / "owner.json",
        {
            "schema": "poke_bot.alakazam_new_list_direct_policy_r241/v1",
            "candidate_id": "alakazam-new-list-direct-policy-r241",
            "latest_owner_clarification_revision": 251,
            "submission": {
                "exact_count": 1,
                "checkpoint_source": "expert_before_iter_00010.pt",
                "intermediate_iteration_5_submission_allowed": False,
                "retry_copy_or_duplicate_allowed": False,
            },
        },
    )
    # The authorizer deliberately verifies the caller's checksum rather than
    # relying on this fixture's arbitrary digest.
    owner_sha = _sha(owner)
    manifest = _write_json(
        tmp_path / "r241-source-snapshot-manifest.json",
        {
            "schema": "poke_bot.alakazam_new_list_direct_r241_source_snapshot/v1",
            "candidate_id": "alakazam-new-list-direct-policy-r241",
            "owner_contract_sha256": owner_sha,
            "source_tree_sha256": "sha256:" + "2" * 64,
            "external_outputs_required": True,
            "baseline_payloads_separate_and_receipted": True,
            "authenticated": True,
            "status": "authenticated_immutable_source_snapshot",
            "files": [{"path": "scripts/example.py"}],
        },
    )
    return owner, manifest, owner_sha


def test_owner_authorization_requires_explicit_intent_and_is_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module(AUTH_SCRIPT, "r241_owner_authorization_test")
    owner, manifest, owner_sha = _owner_and_manifest(tmp_path)
    canonical = _write_json(tmp_path / "canonical-roster.json", {"fixture": True})
    canonical_sha = _sha(canonical)

    monkeypatch.setattr(
        module.baseline_payload,
        "validate_canonical_roster_receipt",
        lambda path, **_kwargs: (
            Path(path),
            {
                "baseline_manifest_sha256": "sha256:" + "3" * 64,
                "baseline_roster_sha256": "sha256:" + "4" * 64,
            },
        ),
    )
    output = tmp_path / "controller" / "owner-start-authorization.json"
    output.parent.mkdir()
    with pytest.raises(module.R241OwnerStartAuthorizationError, match="authorize-managed-start"):
        module.stage_authorization(
            owner_contract=owner,
            owner_contract_sha256=owner_sha,
            source_snapshot_manifest=manifest,
            source_snapshot_manifest_sha256=_sha(manifest),
            canonical_roster_receipt=canonical,
            canonical_roster_receipt_sha256=canonical_sha,
            output=output,
            authorize_managed_start=False,
        )

    first = module.stage_authorization(
        owner_contract=owner,
        owner_contract_sha256=owner_sha,
        source_snapshot_manifest=manifest,
        source_snapshot_manifest_sha256=_sha(manifest),
        canonical_roster_receipt=canonical,
        canonical_roster_receipt_sha256=canonical_sha,
        output=output,
        authorize_managed_start=True,
    )
    second = module.stage_authorization(
        owner_contract=owner,
        owner_contract_sha256=owner_sha,
        source_snapshot_manifest=manifest,
        source_snapshot_manifest_sha256=_sha(manifest),
        canonical_roster_receipt=canonical,
        canonical_roster_receipt_sha256=canonical_sha,
        output=output,
        authorize_managed_start=True,
    )
    assert first["sha256"] == second["sha256"] == _sha(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["authorization_provenance"] == {
        "schema": module.GENERATOR_SCHEMA,
        "create_only": True,
        "explicit_operator_intent": "authorize_managed_r241_training_start",
    }
    assert payload["submission_boundary"]["exact_count"] == 1
    assert not output.stat().st_mode & stat.S_IWUSR


def test_mirror_installer_copies_one_canonical_overlay_and_authorization(
    tmp_path: Path,
) -> None:
    module = _module(MIRROR_SCRIPT, "r241_overlay_mirror_test")
    controller = tmp_path / "controller"
    controller.mkdir()
    outputs = tmp_path / "inzi-outputs"
    outputs.mkdir()
    owner_target = outputs / "state" / "owner-start-authorization.json"
    canonical_owner = _write_json(
        controller / "owner-start-authorization.json",
        {
            "schema": module.OWNER_AUTH_SCHEMA,
            "revision": 241,
            "candidate_id": "alakazam-new-list-direct-policy-r241",
            "status": "authorized",
            "authorized": True,
        },
    )
    canonical_overlay = _write_json(
        controller / "activation-overlay.json",
        {
            "schema": module.OVERLAY_SCHEMA,
            "revision": 241,
            "candidate_id": "alakazam-new-list-direct-policy-r241",
            "status": "ready",
            "passed": True,
            "mirrors": {
                "schema": module.MIRRORS_SCHEMA,
                "hosts": ["inzi", "elmo"],
                "byte_identical_required": True,
            },
            "owner_start_authorization": {
                "schema": module.OWNER_AUTH_SCHEMA,
                "sha256": _sha(canonical_owner),
                "byte_identical_mirrors_required": True,
                "hosts": {
                    "inzi": {"path": str(owner_target)},
                    "elmo": {"path": "/srv/poke-bot-agent/outputs/state/owner.json"},
                },
            },
        },
    )
    overlay_target = outputs / "state" / "activation-overlay.json"
    receipt_target = outputs / "state" / "activation-overlay-mirror.json"
    first = module.install_mirror(
        host="inzi",
        outputs_root=outputs,
        canonical_overlay=canonical_overlay,
        canonical_overlay_sha256=_sha(canonical_overlay),
        canonical_owner_start_authorization=canonical_owner,
        canonical_owner_start_authorization_sha256=_sha(canonical_owner),
        overlay_output=overlay_target,
        owner_start_authorization_output=owner_target,
        receipt_output=receipt_target,
        install_byte_identical_mirror=True,
    )
    second = module.install_mirror(
        host="inzi",
        outputs_root=outputs,
        canonical_overlay=canonical_overlay,
        canonical_overlay_sha256=_sha(canonical_overlay),
        canonical_owner_start_authorization=canonical_owner,
        canonical_owner_start_authorization_sha256=_sha(canonical_owner),
        overlay_output=overlay_target,
        owner_start_authorization_output=owner_target,
        receipt_output=receipt_target,
        install_byte_identical_mirror=True,
    )
    assert overlay_target.read_bytes() == canonical_overlay.read_bytes()
    assert owner_target.read_bytes() == canonical_owner.read_bytes()
    assert first["receipt_sha256"] == second["receipt_sha256"] == _sha(receipt_target)
    receipt = json.loads(receipt_target.read_text(encoding="utf-8"))
    assert receipt["logical_overlay"]["sha256"] == _sha(canonical_overlay)
    assert receipt["byte_identical_copy_verified"] is True
    assert not overlay_target.stat().st_mode & stat.S_IWUSR


def test_mirror_target_rejects_traversal_and_symlink_ancestors(tmp_path: Path) -> None:
    module = _module(MIRROR_SCRIPT, "r241_overlay_mirror_path_safety_test")
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    traversal = outputs / "state" / ".." / ".." / "escaped.json"
    with pytest.raises(
        module.R241ActivationOverlayMirrorError, match="traversal components"
    ):
        module._target_under_outputs(
            traversal, outputs_root=outputs.resolve(), label="traversal target"
        )
    assert not (tmp_path / "escaped.json").exists()
    assert not (outputs / "state").exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    (outputs / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(
        module.R241ActivationOverlayMirrorError, match="escapes|unsafe existing parent"
    ):
        module._target_under_outputs(
            outputs / "linked" / "escaped.json",
            outputs_root=outputs.resolve(),
            label="symlink target",
        )
    assert not (outside / "escaped.json").exists()


def test_overlay_publisher_parser_has_no_duplicate_options() -> None:
    completed = subprocess.run(
        [sys.executable, str(PUBLISHER), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("--canonical-roster-receipt") >= 1
    assert "--elmo-worker-image-receipt" in completed.stdout
    assert "--elmo-worker-image-receipt-sha256" in completed.stdout
    assert "--elmo-worker-image-receipt-declared-path" in completed.stdout


def test_overlay_preservation_uses_typed_owner_contract_file_identity(
    tmp_path: Path,
) -> None:
    module = _module(PUBLISHER, "r241_overlay_preservation_identity_test")
    owner_sha = "sha256:" + "1" * 64

    def receipt_for(contract: object, name: str) -> Path:
        return _write_json(
            tmp_path / name,
            {
                "schema": module.PRESERVATION_SCHEMA,
                "revision": 241,
                "candidate_id": "alakazam-new-list-direct-policy-r241",
                "status": "passed",
                "passed": True,
                "derived_not_self_asserted": True,
                "contract": contract,
            },
        )

    valid = receipt_for(
        {
            "path": "/immutable/state/alakazam-new-list-direct-policy-r241.json",
            "sha256": owner_sha,
            "size_bytes": module.OWNER_CONTRACT_SIZE_BYTES,
        },
        "peak-valid.json",
    )
    module._passed_preservation(
        valid,
        _sha(valid),
        host="elmo",
        owner_contract_sha256=owner_sha,
    )

    nested_legacy = receipt_for(
        {
            "path": "/immutable/state/alakazam-new-list-direct-policy-r241.json",
            "sha256": owner_sha,
            "owner_contract_sha256": owner_sha,
            "size_bytes": module.OWNER_CONTRACT_SIZE_BYTES,
        },
        "peak-nested-legacy.json",
    )
    missing_identity_field = receipt_for(
        {
            "path": "/immutable/state/alakazam-new-list-direct-policy-r241.json",
            "sha256": owner_sha,
        },
        "peak-missing-size.json",
    )
    wrong_identity = receipt_for(
        {
            "path": "/immutable/state/alakazam-new-list-direct-policy-r241.json",
            "sha256": "sha256:" + "2" * 64,
            "size_bytes": module.OWNER_CONTRACT_SIZE_BYTES,
        },
        "peak-wrong-sha.json",
    )
    zero_size = receipt_for(
        {
            "path": "/immutable/state/alakazam-new-list-direct-policy-r241.json",
            "sha256": owner_sha,
            "size_bytes": 0,
        },
        "peak-zero-size.json",
    )
    non_integer_size = receipt_for(
        {
            "path": "/immutable/state/alakazam-new-list-direct-policy-r241.json",
            "sha256": owner_sha,
            "size_bytes": str(module.OWNER_CONTRACT_SIZE_BYTES),
        },
        "peak-non-integer-size.json",
    )
    boolean_size = receipt_for(
        {
            "path": "/immutable/state/alakazam-new-list-direct-policy-r241.json",
            "sha256": owner_sha,
            "size_bytes": True,
        },
        "peak-boolean-size.json",
    )
    negative_size = receipt_for(
        {
            "path": "/immutable/state/alakazam-new-list-direct-policy-r241.json",
            "sha256": owner_sha,
            "size_bytes": -module.OWNER_CONTRACT_SIZE_BYTES,
        },
        "peak-negative-size.json",
    )
    for rejected in (
        nested_legacy,
        missing_identity_field,
        wrong_identity,
        zero_size,
        non_integer_size,
        boolean_size,
        negative_size,
    ):
        with pytest.raises(module.R241ActivationOverlayError):
            module._passed_preservation(
                rejected,
                _sha(rejected),
                host="elmo",
                owner_contract_sha256=owner_sha,
            )


def test_overlay_remote_quartet_requires_schema_specific_no_work_statuses(
    tmp_path: Path,
) -> None:
    """Admit the actual four preflight states without making status generic."""

    module = _module(PUBLISHER, "r241_overlay_remote_quartet_status_test")
    # These are the status-bearing top-level shapes emitted by the immutable
    # Elmo preflight quartet.  The publisher intentionally validates the
    # receipt headers here; the worker's own validator checks their deep body
    # bindings before these copied, checksum-pinned artifacts are accepted.
    actual_quartet = {
        "manifest": {
            "schema": module.REMOTE_MANIFEST_SCHEMA,
            "status": "passed",
            "body": {"receipts": {"host": {}, "runtime": {}, "gameplay": {}}},
        },
        "host": {
            "schema": module.REMOTE_HOST_SCHEMA,
            "status": "passed",
            "body": {"host": {"declared_role": "elmo"}},
        },
        "runtime": {
            "schema": module.REMOTE_RUNTIME_SCHEMA,
            "status": "passed",
            "body": {"direct_policy": {"direct_policy_only": True}},
        },
        "gameplay": {
            "schema": module.REMOTE_GAMEPLAY_SCHEMA,
            "status": "ready_no_games_started",
            "body": {
                "observation_scope": "preflight_no_games_started",
                "promotion_jobs_allowed": False,
            },
        },
    }

    for name, expected in actual_quartet.items():
        receipt = _write_json(
            tmp_path / f"{name}.json",
            {
                "schema": expected["schema"],
                "revision": module.REVISION,
                "status": expected["status"],
                "passed": True,
                "deployment_action": "not_started",
                **expected["body"],
            },
        )
        module._passed_remote_receipt(
            receipt,
            _sha(receipt),
            schema=str(expected["schema"]),
            label=f"actual {name} preflight receipt",
        )

        wrong_status = (
            "passed"
            if expected["status"] == "ready_no_games_started"
            else "ready_no_games_started"
        )
        rejected_status = _write_json(
            tmp_path / f"{name}-wrong-status.json",
            {
                "schema": expected["schema"],
                "revision": module.REVISION,
                "status": wrong_status,
                "passed": True,
                "deployment_action": "not_started",
                **expected["body"],
            },
        )
        with pytest.raises(module.R241ActivationOverlayError, match="canonical r241 preflight"):
            module._passed_remote_receipt(
                rejected_status,
                _sha(rejected_status),
                schema=str(expected["schema"]),
                label=f"wrong-status {name} preflight receipt",
            )

        rejected_action = _write_json(
            tmp_path / f"{name}-wrong-action.json",
            {
                "schema": expected["schema"],
                "revision": module.REVISION,
                "status": expected["status"],
                "passed": True,
                "deployment_action": "started",
                **expected["body"],
            },
        )
        with pytest.raises(module.R241ActivationOverlayError, match="canonical r241 preflight"):
            module._passed_remote_receipt(
                rejected_action,
                _sha(rejected_action),
                schema=str(expected["schema"]),
                label=f"wrong-action {name} preflight receipt",
            )
