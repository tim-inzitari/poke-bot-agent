"""Contract tests for the inert, checksum-bound r241 Inzi service chain."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preflight_alakazam_new_list_direct_r241_service_chain.py"
UNITS = {
    "trainer": ROOT / "deploy/systemd/pokebot-alakazam-new-list-direct-r241.service.template",
    "finalizer": ROOT / "deploy/systemd/pokebot-alakazam-new-list-direct-r241-finalize.service.template",
    "queue": ROOT / "deploy/systemd/pokebot-alakazam-new-list-direct-r241-submission-queue.service.template",
    "uploader": ROOT / "deploy/systemd/pokebot-alakazam-new-list-direct-r241-upload.service.template",
}


@pytest.fixture(scope="module")
def preflight_module():
    spec = importlib.util.spec_from_file_location("r241_inzi_service_chain_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object, *, readonly: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))
    path.chmod(0o444 if readonly else 0o644)
    return path


def _write_file(path: Path, body: bytes = b"x\n", *, readonly: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    path.chmod(0o444 if readonly else 0o644)
    return path


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
        else:
            path.chmod(0o444)
    root.chmod(0o555)


def _sha(path: Path, module) -> str:
    return module._sha256_file(path)


def _digest(seed: str) -> str:
    return "sha256:" + seed * 64


def _env(monkeypatch: pytest.MonkeyPatch, values: dict[str, Path | str]) -> None:
    for name in (
        "CG_LIB_PATH",
        "POKEBOT_LIBCG_PATH",
        "POKEBOT_BATCH_LIBCG",
        "POKEBOT_ALLOW_ORACLE_DECK",
        "POKEBOT_BASELINES_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, str(value))


def _snapshot_source(
    *,
    tmp_path: Path,
    module,
    outputs: Path,
    run_root: Path,
    official_cg_dir: Path,
) -> tuple[Path, Path, str, str, Path]:
    deployments = tmp_path / "deployments"
    deployments.mkdir()
    seed = deployments / "source-seed"
    seed.mkdir()

    owner_contract = _write_json(
        seed / module.OWNER_CONTRACT_RELATIVE_PATH,
        {
            "schema": module.OWNER_CONTRACT_SCHEMA,
            "owner_decision_revision": module.R241_REVISION,
            "latest_owner_clarification_revision": 1,
            "candidate_id": module.CANDIDATE_ID,
        },
        readonly=False,
    )
    owner_sha = _sha(owner_contract, module)
    registry = _write_json(
        seed / module.REGISTRY_RELATIVE_PATH,
        {
            "schema": module.RUNTIME_REGISTRY_SCHEMA,
            "revision": module.R241_REVISION,
            "owner_clarification_revision": 1,
            "status": "immutable_pending_external_activation_overlay",
            "owner_contract": {
                "path": module.OWNER_CONTRACT_RELATIVE_PATH,
                "sha256": owner_sha,
            },
            "run": {
                "name": module.RUN_NAME,
                "inzi_root": str(run_root),
                "elmo_root": "/remote/elmo/outputs/pure_rl/r241",
                "external_activation_overlay_required": True,
                "activation_overlay_schema": module.ACTIVATION_OVERLAY_SCHEMA,
            },
            "official_libcg": {
                "hosts": {
                    "inzi": {"runtime_root": str(official_cg_dir)},
                    "elmo": {"runtime_root": "/remote/elmo/outputs/pure_rl/r241/runtime/cg-r236"},
                }
            },
            "source_snapshot": {
                "schema": module.SOURCE_SNAPSHOT_SCHEMA,
                "candidate_id": module.CANDIDATE_ID,
                "owner_contract_sha256": owner_sha,
                "status": "pending_immutable_source_snapshot",
                "manifest_sha256": "",
                "source_tree_sha256": "",
                "hosts": {
                    "inzi": {"root": "", "manifest": "", "outputs_root": str(outputs)},
                    "elmo": {"root": "", "manifest": "", "outputs_root": "/remote/elmo/outputs"},
                },
            },
            "baseline_payloads": {
                "status": "pending_external_baseline_payload_snapshot",
                "source_snapshot_fallback_allowed": False,
                "hosts": {"inzi": {}, "elmo": {}},
            },
        },
        readonly=False,
    )
    for relative in module.REQUIRED_SERVICE_CHAIN_SOURCE_MEMBERS:
        path = seed / relative
        if path in {owner_contract, registry}:
            continue
        _write_file(path, body=(relative + "\n").encode("utf-8"), readonly=False)

    rows = []
    for path in sorted(member for member in seed.rglob("*") if member.is_file()):
        relative = path.relative_to(seed).as_posix()
        rows.append(
            {
                "path": relative,
                "sha256": _sha(path, module),
                "size_bytes": path.stat().st_size,
            }
        )
    source_tree_sha = module._source_tree_digest(rows)
    manifest = _write_json(
        seed / module.SOURCE_MANIFEST_FILENAME,
        {
            "schema": module.SOURCE_SNAPSHOT_SCHEMA,
            "candidate_id": module.CANDIDATE_ID,
            "owner_contract_sha256": owner_sha,
            "source_tree_sha256": source_tree_sha,
            "external_outputs_required": True,
            "baseline_payloads_separate_and_receipted": True,
            "authenticated": True,
            "status": "authenticated_immutable_source_snapshot",
            "files": rows,
        },
        readonly=False,
    )
    manifest_sha = _sha(manifest, module)
    source_root = deployments / (module.SOURCE_ROOT_PREFIX + manifest_sha.removeprefix("sha256:")[:16])
    seed.rename(source_root)
    _seal_tree(source_root)
    return source_root, source_root / module.SOURCE_MANIFEST_FILENAME, manifest_sha, source_tree_sha, owner_contract


def _snapshot_baseline(
    *, tmp_path: Path, module, owner_sha: str
) -> tuple[Path, Path, str, str, str, str, list[dict[str, str]]]:
    baselines = tmp_path / "baselines"
    baselines.mkdir()
    seed = baselines / "baseline-seed"
    seed.mkdir()
    mounted_manifest = _write_json(seed / "manifest.json", {"agents": []}, readonly=False)
    mounted_manifest_sha = _sha(mounted_manifest, module)
    roster = [
        {
            "id": "fixture-opponent",
            "group": "community",
            "dir": "fixture",
            "content_digest": _digest("c"),
        }
    ]
    roster_sha = _digest("d")
    rows = [
        {
            "path": "manifest.json",
            "sha256": mounted_manifest_sha,
            "size_bytes": mounted_manifest.stat().st_size,
        }
    ]
    tree_sha = module._source_tree_digest(rows)
    manifest = _write_json(
        seed / module.BASELINE_MANIFEST_FILENAME,
        {
            "schema": module.BASELINE_PAYLOAD_SNAPSHOT_SCHEMA,
            "revision": module.R241_REVISION,
            "candidate_id": module.CANDIDATE_ID,
            "owner_contract_sha256": owner_sha,
            "baseline_tree_sha256": tree_sha,
            "baseline_manifest_sha256": mounted_manifest_sha,
            "baseline_roster_sha256": roster_sha,
            "baseline_roster": roster,
            "authenticated": True,
            "status": "authenticated_immutable_baseline_payload_snapshot",
            "files": rows,
        },
        readonly=False,
    )
    manifest_sha = _sha(manifest, module)
    root = baselines / (module.BASELINE_ROOT_PREFIX + manifest_sha.removeprefix("sha256:")[:16])
    seed.rename(root)
    _seal_tree(root)
    return (
        root,
        root / module.BASELINE_MANIFEST_FILENAME,
        manifest_sha,
        tree_sha,
        mounted_manifest_sha,
        roster_sha,
        roster,
    )


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module
) -> tuple[dict[str, Path], Path, str, Path, str]:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    state = outputs / "state"
    state.mkdir()
    elmo_outputs = tmp_path / "elmo-outputs"
    elmo_state = elmo_outputs / "state"
    elmo_state.mkdir(parents=True)
    run_root = outputs / "pure_rl" / module.RUN_NAME
    runtime_dir = run_root / "runtime"
    official_cg_dir = runtime_dir / "cg-r236"
    official_cg_dir.mkdir(parents=True)
    source_root, source_manifest, source_manifest_sha, source_tree_sha, _owner_contract = _snapshot_source(
        tmp_path=tmp_path,
        module=module,
        outputs=outputs,
        run_root=run_root,
        official_cg_dir=official_cg_dir,
    )
    owner_sha = _sha(source_root / module.OWNER_CONTRACT_RELATIVE_PATH, module)
    source_staging = _write_json(
        state / "r241-inzi-source-staging.json",
        {
            "schema": module.SOURCE_STAGING_SCHEMA,
            "revision": module.R241_REVISION,
            "candidate_id": module.CANDIDATE_ID,
            "status": "passed",
            "passed": True,
            "source_snapshot": {
                "schema": module.SOURCE_SNAPSHOT_SCHEMA,
                "status": "authenticated_immutable_source_snapshot",
                "authenticated": True,
                "host": "inzi",
                "root": str(source_root),
                "source_execution_root": str(source_root),
                "manifest": str(source_manifest),
                "manifest_sha256": source_manifest_sha,
                "source_tree_sha256": source_tree_sha,
                "owner_contract_sha256": owner_sha,
                "outputs_root": str(outputs),
            },
        },
    )
    (
        baseline_root,
        baseline_manifest,
        baseline_manifest_sha,
        baseline_tree_sha,
        canonical_baseline_manifest_sha,
        canonical_baseline_roster_sha,
        roster,
    ) = _snapshot_baseline(tmp_path=tmp_path, module=module, owner_sha=owner_sha)
    canonical_receipt = _write_json(
        state / "r241-canonical-baseline-roster.json",
        {
            "schema": module.CANONICAL_BASELINE_ROSTER_SCHEMA,
            "revision": module.R241_REVISION,
            "candidate_id": module.CANDIDATE_ID,
            "status": "passed",
            "passed": True,
            "owner_contract_sha256": owner_sha,
            "baseline_manifest_sha256": canonical_baseline_manifest_sha,
            "baseline_roster_sha256": canonical_baseline_roster_sha,
            "baseline_roster": roster,
            "public_contract_sha256s": {
                "active_gate_contract": _digest("e"),
                "frozen_specialist_registry": _digest("f"),
                "research_control_registry": _digest("1"),
            },
        },
    )
    canonical_receipt_sha = _sha(canonical_receipt, module)
    baseline_staging = _write_json(
        state / "r241-inzi-baseline-staging.json",
        {
            "schema": module.BASELINE_PAYLOAD_STAGING_SCHEMA,
            "revision": module.R241_REVISION,
            "candidate_id": module.CANDIDATE_ID,
            "status": "passed",
            "passed": True,
            "receipt_outside_source_and_baseline_snapshot": True,
            "baseline_payload_snapshot": {
                "schema": module.BASELINE_PAYLOAD_SNAPSHOT_SCHEMA,
                "revision": module.R241_REVISION,
                "candidate_id": module.CANDIDATE_ID,
                "status": "authenticated_immutable_baseline_payload_snapshot",
                "authenticated": True,
                "host": "inzi",
                "root": str(baseline_root),
                "manifest": str(baseline_manifest),
                "manifest_sha256": baseline_manifest_sha,
                "baseline_tree_sha256": baseline_tree_sha,
                "baseline_manifest_sha256": canonical_baseline_manifest_sha,
                "baseline_roster_sha256": canonical_baseline_roster_sha,
                "baseline_roster": roster,
                "owner_contract_sha256": owner_sha,
            },
            "canonical_roster_receipt": {
                "path": str(canonical_receipt),
                "sha256": canonical_receipt_sha,
                "baseline_manifest_sha256": canonical_baseline_manifest_sha,
                "baseline_roster_sha256": canonical_baseline_roster_sha,
                "public_contract_sha256s": {
                    "active_gate_contract": _digest("e"),
                    "frozen_specialist_registry": _digest("f"),
                    "research_control_registry": _digest("1"),
                },
            },
        },
    )
    authorization = _write_json(
        state / "r241-owner-start-authorization.json",
        {
            "schema": module.OWNER_START_AUTHORIZATION_SCHEMA,
            "revision": module.R241_REVISION,
            "candidate_id": module.CANDIDATE_ID,
            "status": "authorized",
            "authorized": True,
            "owner_contract_sha256": owner_sha,
            "allowed_actions": ["managed_r241_training_start"],
            "source_snapshot_manifest_sha256": source_manifest_sha,
            "source_tree_sha256": source_tree_sha,
            "canonical_baseline_manifest_sha256": canonical_baseline_manifest_sha,
            "canonical_baseline_roster_sha256": canonical_baseline_roster_sha,
            "submission_boundary": {
                "exact_count": 1,
                "checkpoint_source": "expert_before_iter_00010.pt",
                "intermediate_iteration_5_submission_allowed": False,
                "retry_copy_or_duplicate_allowed": False,
            },
            "authorization_provenance": {
                "schema": module.OWNER_START_AUTHORIZATION_GENERATOR_SCHEMA,
                "create_only": True,
                "explicit_operator_intent": "authorize_managed_r241_training_start",
            },
        },
    )
    elmo_authorization = _write_file(
        elmo_state / "r241-owner-start-authorization.json",
        authorization.read_bytes(),
    )
    assert _sha(authorization, module) == _sha(elmo_authorization, module)

    matchups = {
        "matchup_tree": _write_file(runtime_dir / "matchup-tree.json", b"{}\n"),
        "matchup_runtime_activation": _write_file(
            runtime_dir / "matchup-runtime-activation.json", b"{}\n"
        ),
        "model_runtime_activation": _write_file(
            runtime_dir / "model-runtime-activation.json", b"{}\n"
        ),
    }
    finalizer_output = run_root / "terminal-package"
    finalizer_receipt = state / "r241-finalizer.json"
    queue_authorization = state / "r241-queue-authorization.json"
    queue = state / "r241-submission-queue.json"
    queue_receipts = state / "r241-submission-receipts"
    paths = {
        "run_root": run_root,
        "runtime_dir": runtime_dir,
        "official_cg_dir": official_cg_dir,
        **matchups,
        "finalizer_output_dir": finalizer_output,
        "finalizer_receipt": finalizer_receipt,
        "queue_authorization": queue_authorization,
        "queue": queue,
        "queue_receipts_dir": queue_receipts,
        "baseline_root": baseline_root,
        "baseline_manifest": baseline_manifest,
        "source_root": source_root,
        "source_manifest": source_manifest,
        "elmo_overlay": elmo_state / "r241-activation-overlay.json",
    }
    overlay = state / "r241-inzi-activation-overlay.json"
    _write_json(
        overlay,
        {
            "schema": module.ACTIVATION_OVERLAY_SCHEMA,
            "revision": module.R241_REVISION,
            "candidate_id": module.CANDIDATE_ID,
            "status": "ready",
            "passed": True,
            "owner_contract_sha256": owner_sha,
            "base_registry": {
                "path": module.REGISTRY_RELATIVE_PATH,
                "sha256": _sha(source_root / module.REGISTRY_RELATIVE_PATH, module),
            },
            "source_snapshot": {
                "status": "ready",
                "owner_contract_sha256": owner_sha,
                "manifest_sha256": source_manifest_sha,
                "source_tree_sha256": source_tree_sha,
                "hosts": {
                    "inzi": {
                        "root": str(source_root),
                        "manifest": str(source_manifest),
                        "outputs_root": str(outputs),
                        "staging_receipt": str(source_staging),
                        "staging_receipt_sha256": _sha(source_staging, module),
                    },
                    "elmo": {
                        "root": "/remote/elmo/source",
                        "manifest": "/remote/elmo/source/r241-source-snapshot-manifest.json",
                        "outputs_root": "/remote/elmo/outputs",
                        "staging_receipt": "/remote/elmo/outputs/state/r241-source-staging.json",
                        "staging_receipt_sha256": _digest("2"),
                    },
                },
            },
            "baseline_payloads": {
                "status": "ready",
                "canonical_roster_receipt_sha256": canonical_receipt_sha,
                "canonical_baseline_manifest_sha256": canonical_baseline_manifest_sha,
                "canonical_baseline_roster_sha256": canonical_baseline_roster_sha,
                "hosts": {
                    "inzi": {
                        "root": str(baseline_root),
                        "manifest": str(baseline_manifest),
                        "manifest_sha256": baseline_manifest_sha,
                        "baseline_tree_sha256": baseline_tree_sha,
                        "staging_receipt": str(baseline_staging),
                        "staging_receipt_sha256": _sha(baseline_staging, module),
                        "canonical_roster_receipt": str(canonical_receipt),
                    },
                    "elmo": {
                        "root": "/remote/elmo/baselines/r241",
                        "manifest": "/remote/elmo/baselines/r241/r241-baseline-payload-manifest.json",
                        "manifest_sha256": _digest("3"),
                        "baseline_tree_sha256": _digest("4"),
                        "staging_receipt": "/remote/elmo/outputs/state/r241-baseline-staging.json",
                        "staging_receipt_sha256": _digest("5"),
                        "canonical_roster_receipt": "/remote/elmo/outputs/state/r241-canonical-baseline-roster.json",
                    },
                },
            },
            "peak_r195_preservation": {
                "receipt_sha256_inzi": _digest("6"),
                "receipt_sha256_elmo": _digest("7"),
            },
            "worker_image": {
                "schema": module.WORKER_IMAGE_SCHEMA,
                "image_id_sha256": _digest("e"),
                "receipt": {
                    "path": "/remote/elmo/outputs/pure_rl/alakazam_new_list_direct_policy_r241/runtime/elmo-8767/r241-elmo-official-r236-worker-image.json",
                    "sha256": _digest("f"),
                },
                "source_snapshot": {
                    "owner_contract_sha256": owner_sha,
                    "manifest_sha256": source_manifest_sha,
                    "source_tree_sha256": source_tree_sha,
                },
                "tag": "pokebot-r241-elmo-official-r236-worker:test",
            },
            "remote_collection": {
                "endpoint_id": "elmo-r241-official-r236-direct-policy-8767",
                "manifest_sha256": _digest("8"),
                "host_receipt_sha256": _digest("9"),
                "runtime_receipt_sha256": _digest("a"),
                "gameplay_receipt_sha256": _digest("b"),
                "checkpoint_transport": {
                    "schema": module.CHECKPOINT_TRANSPORT_SCHEMA,
                    "status": "ready",
                    "endpoint_id": "elmo-r241-official-r236-direct-policy-8767",
                    "host_role": "elmo",
                    "verification_endpoint": "elmo:8767",
                    "verification_port": 8767,
                    "host_root": "/remote/elmo/checkpoints/r241",
                    "trainer_visible_root": str(outputs / "checkpoint-mount"),
                    "container_root": "/workspace/checkpoint",
                    "environment_key": "POKEBOT_REMOTE_CHECKPOINT_ROOT",
                    "remote_path_prefix": "/workspace/checkpoint/",
                    "content_addressing": {
                        "algorithm": "sha256",
                        "filename_scheme": "poke_bot.remote_jobs.digest_addressed_basename/v1",
                    },
                    "read_only_container_mount": True,
                    "same_absolute_source_and_baseline_paths_preserved": True,
                    "staging_receipt": "/remote/elmo/outputs/state/r241-checkpoint-transport.json",
                    "staging_receipt_sha256": _digest("c"),
                    "initial_checkpoint": {
                        "container_path": "/workspace/checkpoint/" + "d" * 64 + ".pt",
                        "sha256": _digest("d"),
                    },
                },
            },
            "mirrors": {
                "schema": module.OVERLAY_MIRRORS_SCHEMA,
                "hosts": ["inzi", "elmo"],
                "byte_identical_required": True,
            },
            "owner_start_authorization": {
                "schema": module.OWNER_START_AUTHORIZATION_SCHEMA,
                "sha256": _sha(authorization, module),
                "byte_identical_mirrors_required": True,
                "hosts": {
                    "inzi": {"path": str(authorization)},
                    "elmo": {"path": str(elmo_authorization)},
                },
            },
        },
    )
    _write_file(paths["elmo_overlay"], overlay.read_bytes())
    assert overlay.read_bytes() == paths["elmo_overlay"].read_bytes()
    overlay_sha = _sha(overlay, module)
    mirror_receipt = _write_json(
        state / "r241-inzi-activation-overlay-mirror.json",
        {
            "schema": module.ACTIVATION_OVERLAY_MIRROR_SCHEMA,
            "revision": module.R241_REVISION,
            "candidate_id": module.CANDIDATE_ID,
            "status": "passed",
            "passed": True,
            "host": "inzi",
            "logical_overlay": {"path": str(overlay), "sha256": overlay_sha},
            "owner_start_authorization": {
                "path": str(authorization),
                "sha256": _sha(authorization, module),
            },
            "outputs_root": str(outputs),
            "byte_identical_copy_verified": True,
        },
    )
    mirror_receipt_sha = _sha(mirror_receipt, module)
    paths["activation_overlay_mirror_receipt"] = mirror_receipt
    python = _write_file(tmp_path / "bin" / "python", b"#!/bin/sh\nexit 0\n", readonly=False)
    python.chmod(0o555)
    kaggle = _write_file(tmp_path / "bin" / "kaggle", b"#!/bin/sh\nexit 0\n", readonly=False)
    kaggle.chmod(0o555)
    environment: dict[str, Path | str] = {
        "R241_INZI_PYTHON": python,
        "R241_INZI_SOURCE_SNAPSHOT_ROOT": source_root,
        "R241_INZI_SOURCE_SNAPSHOT_MANIFEST": source_manifest,
        "R241_INZI_SOURCE_SNAPSHOT_MANIFEST_SHA256": source_manifest_sha,
        "R241_INZI_SOURCE_TREE_SHA256": source_tree_sha,
        "R241_INZI_OUTPUTS_ROOT": outputs,
        "R241_INZI_BASELINES_ROOT": baseline_root,
        "R241_INZI_BASELINE_PAYLOAD_MANIFEST": baseline_manifest,
        "R241_INZI_BASELINE_PAYLOAD_MANIFEST_SHA256": baseline_manifest_sha,
        "R241_INZI_BASELINE_TREE_SHA256": baseline_tree_sha,
        "R241_INZI_ACTIVATION_OVERLAY": overlay,
        "R241_INZI_ACTIVATION_OVERLAY_SHA256": overlay_sha,
        "R241_INZI_ACTIVATION_OVERLAY_MIRROR_RECEIPT_SHA256": mirror_receipt_sha,
        "R241_INZI_RUNTIME_REGISTRY": source_root / module.REGISTRY_RELATIVE_PATH,
        "R241_INZI_KAGGLE_BIN": kaggle,
    }
    environment.update({"R241_INZI_" + name.upper(): path for name, path in paths.items() if name not in {"source_root", "source_manifest", "baseline_root", "baseline_manifest"}})
    _env(monkeypatch, environment)
    return paths, overlay, overlay_sha, mirror_receipt, mirror_receipt_sha


def _replace_overlay(
    overlay: Path, monkeypatch: pytest.MonkeyPatch, module, change
) -> str:
    overlay.chmod(0o644)
    payload = json.loads(overlay.read_text(encoding="utf-8"))
    change(payload)
    overlay.write_bytes(_canonical(payload))
    overlay.chmod(0o444)
    digest = _sha(overlay, module)
    monkeypatch.setenv("R241_INZI_ACTIVATION_OVERLAY_SHA256", digest)
    return digest


def _validate(
    module,
    *,
    stage: str,
    overlay: Path,
    overlay_sha: str,
    mirror_receipt: Path,
    mirror_receipt_sha: str,
) -> dict[str, object]:
    return module.validate_activation_overlay(
        stage=stage,
        overlay_path=overlay,
        overlay_sha256=overlay_sha,
        overlay_mirror_receipt=mirror_receipt,
        overlay_mirror_receipt_sha256=mirror_receipt_sha,
    )


def test_r241_service_templates_form_exact_inert_terminal_chain() -> None:
    texts = {name: path.read_text(encoding="utf-8") for name, path in UNITS.items()}
    assert "OnSuccess=pokebot-alakazam-new-list-direct-r241-finalize.service" in texts["trainer"]
    assert "OnSuccess=pokebot-alakazam-new-list-direct-r241-submission-queue.service" in texts["finalizer"]
    assert "OnSuccess=pokebot-alakazam-new-list-direct-r241-upload.service" in texts["queue"]
    assert "OnSuccess=" not in texts["uploader"]

    forbidden = (
        "pokebot-kaggle-submission-queue.service",
        "pokebot-final-format-alakazam-rtp-r175",
        "pokebot-alakazam-terminal-expert-bootstrap-no-rtp-r195",
        "pokebot-r229",
        "pokebot-r228",
        "Conflicts=",
        "Requires=",
        "PartOf=",
        "BindsTo=",
        "systemctl",
        ".timer",
        "Restart=always",
        "Restart=on-failure",
        "[Install]",
    )
    for text in texts.values():
        assert "EnvironmentFile=/etc/pokebot/alakazam-new-list-direct-r241-inzi-activation.env" in text
        assert "scripts/preflight_alakazam_new_list_direct_r241_service_chain.py" in text
        assert "R241_INZI_SOURCE_SNAPSHOT_ROOT" in text
        assert "R241_INZI_ACTIVATION_OVERLAY" in text
        assert "R241_INZI_ACTIVATION_OVERLAY_SHA256" in text
        assert "R241_INZI_ACTIVATION_OVERLAY_MIRROR_RECEIPT" in text
        assert "R241_INZI_ACTIVATION_OVERLAY_MIRROR_RECEIPT_SHA256" in text
        assert "R241_INZI_RUNTIME_REGISTRY" in text
        assert text.count("--overlay-mirror-receipt ") == 1
        assert text.count("--overlay-mirror-receipt-sha256 ") == 1
        assert (
            '--overlay-mirror-receipt "$R241_INZI_ACTIVATION_OVERLAY_MIRROR_RECEIPT"'
            in text
        )
        assert (
            '--overlay-mirror-receipt-sha256 '
            '"$R241_INZI_ACTIVATION_OVERLAY_MIRROR_RECEIPT_SHA256"' in text
        )
        assert text.count("Restart=no") == 1
        assert all(item not in text for item in forbidden)

    assert texts["trainer"].count("--activation-overlay ") == 2
    assert texts["trainer"].count("--activation-overlay-sha256 ") == 2
    assert texts["trainer"].count("--activation-overlay-mirror-receipt ") == 2
    assert texts["trainer"].count("--activation-overlay-mirror-receipt-sha256 ") == 2
    assert (
        '--activation-overlay-mirror-receipt "$R241_INZI_ACTIVATION_OVERLAY_MIRROR_RECEIPT"'
        in texts["trainer"]
    )
    assert (
        '--activation-overlay-mirror-receipt-sha256 '
        '"$R241_INZI_ACTIVATION_OVERLAY_MIRROR_RECEIPT_SHA256"' in texts["trainer"]
    )
    assert texts["trainer"].count("--check") == 1
    assert texts["trainer"].count("--execute") == 1
    assert "launch_alakazam_new_list_direct_r241.py" in texts["trainer"]
    assert "finalize_alakazam_new_list_direct_r241.py" in texts["finalizer"]
    assert "--enqueue" in texts["queue"]
    assert "--upload" not in texts["queue"]
    assert "upload_alakazam_new_list_direct_r241_submission_queue.py" in texts["uploader"]
    assert texts["uploader"].count("--upload") == 1


def test_preflight_accepts_only_checksum_bound_external_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight_module
) -> None:
    paths, overlay, overlay_sha, mirror_receipt, mirror_receipt_sha = _fixture(
        tmp_path, monkeypatch, preflight_module
    )
    baseline = json.loads(overlay.read_text(encoding="utf-8"))["baseline_payloads"]
    assert "canonical_roster_receipt" not in baseline
    assert (
        baseline["hosts"]["inzi"]["canonical_roster_receipt"]
        != baseline["hosts"]["elmo"]["canonical_roster_receipt"]
    )

    trainer = _validate(
        preflight_module,
        stage="trainer",
        overlay=overlay,
        overlay_sha=overlay_sha,
        mirror_receipt=mirror_receipt,
        mirror_receipt_sha=mirror_receipt_sha,
    )
    finalizer = _validate(
        preflight_module,
        stage="finalizer",
        overlay=overlay,
        overlay_sha=overlay_sha,
        mirror_receipt=mirror_receipt,
        mirror_receipt_sha=mirror_receipt_sha,
    )
    assert trainer["status"] == "passed"
    assert trainer["unit"] == preflight_module.TRAINER_UNIT
    assert trainer["next_on_success_unit"] == preflight_module.FINALIZER_UNIT
    assert trainer["performed_training"] is False
    assert finalizer["unit"] == preflight_module.FINALIZER_UNIT
    assert finalizer["run_root"] == str(paths["run_root"])
    assert not hasattr(preflight_module, "OWNER_CONTRACT_SHA256")
    assert overlay.read_bytes() == paths["elmo_overlay"].read_bytes()

    _write_file(paths["finalizer_receipt"])
    _write_file(paths["queue_authorization"])
    queue = _validate(
        preflight_module,
        stage="queue",
        overlay=overlay,
        overlay_sha=overlay_sha,
        mirror_receipt=mirror_receipt,
        mirror_receipt_sha=mirror_receipt_sha,
    )
    assert queue["unit"] == preflight_module.QUEUE_UNIT
    assert not paths["queue"].exists()

    _write_file(paths["queue"])
    uploader = _validate(
        preflight_module,
        stage="uploader",
        overlay=overlay,
        overlay_sha=overlay_sha,
        mirror_receipt=mirror_receipt,
        mirror_receipt_sha=mirror_receipt_sha,
    )
    assert uploader["unit"] == preflight_module.UPLOADER_UNIT
    assert uploader["next_on_success_unit"] is None
    assert uploader["performed_submission"] is False


@pytest.mark.parametrize(
    "case", ("missing", "wrong", "cross_host", "unscoped", "null_unscoped")
)
def test_preflight_rejects_non_host_scoped_canonical_roster_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preflight_module,
    case: str,
) -> None:
    _paths, overlay, _sealed_sha, mirror_receipt, mirror_receipt_sha = _fixture(
        tmp_path, monkeypatch, preflight_module
    )
    payload = json.loads(overlay.read_text(encoding="utf-8"))
    baseline = payload["baseline_payloads"]
    inzi = baseline["hosts"]["inzi"]
    if case == "missing":
        inzi.pop("canonical_roster_receipt")
    elif case == "wrong":
        original = Path(inzi["canonical_roster_receipt"])
        wrong = _write_file(overlay.parent / "wrong-canonical-roster.json", original.read_bytes())
        inzi["canonical_roster_receipt"] = str(wrong)
    elif case == "cross_host":
        inzi["canonical_roster_receipt"] = baseline["hosts"]["elmo"][
            "canonical_roster_receipt"
        ]
    elif case == "unscoped":
        baseline["canonical_roster_receipt"] = inzi["canonical_roster_receipt"]
    else:
        baseline["canonical_roster_receipt"] = None
    overlay.chmod(0o644)
    overlay.write_bytes(_canonical(payload))
    overlay.chmod(0o444)
    changed_sha = _sha(overlay, preflight_module)
    monkeypatch.setenv("R241_INZI_ACTIVATION_OVERLAY_SHA256", changed_sha)

    with pytest.raises(
        preflight_module.R241ServiceChainPreflightError, match="baseline|canonical"
    ):
        _validate(
            preflight_module,
            stage="trainer",
            overlay=overlay,
            overlay_sha=changed_sha,
            mirror_receipt=mirror_receipt,
            mirror_receipt_sha=mirror_receipt_sha,
        )


def test_preflight_rejects_changed_overlay_even_if_it_is_readonly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight_module
) -> None:
    _paths, overlay, sealed_sha, mirror_receipt, mirror_receipt_sha = _fixture(
        tmp_path, monkeypatch, preflight_module
    )
    overlay.chmod(0o644)
    payload = json.loads(overlay.read_text(encoding="utf-8"))
    payload["passed"] = False
    overlay.write_bytes(_canonical(payload))
    overlay.chmod(0o444)

    with pytest.raises(preflight_module.R241ServiceChainPreflightError, match="checksum drifted"):
        _validate(
            preflight_module,
            stage="trainer",
            overlay=overlay,
            overlay_sha=sealed_sha,
            mirror_receipt=mirror_receipt,
            mirror_receipt_sha=mirror_receipt_sha,
        )


def test_preflight_rejects_overlay_owner_contract_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight_module
) -> None:
    _paths, overlay, _sealed_sha, mirror_receipt, mirror_receipt_sha = _fixture(
        tmp_path, monkeypatch, preflight_module
    )
    changed_sha = _replace_overlay(
        overlay,
        monkeypatch,
        preflight_module,
        lambda payload: payload.__setitem__("owner_contract_sha256", _digest("0")),
    )
    with pytest.raises(preflight_module.R241ServiceChainPreflightError, match="source snapshot identity drifted"):
        _validate(
            preflight_module,
            stage="trainer",
            overlay=overlay,
            overlay_sha=changed_sha,
            mirror_receipt=mirror_receipt,
            mirror_receipt_sha=mirror_receipt_sha,
        )


def test_preflight_rejects_worker_image_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight_module
) -> None:
    _paths, overlay, _sealed_sha, mirror_receipt, mirror_receipt_sha = _fixture(
        tmp_path, monkeypatch, preflight_module
    )
    changed_sha = _replace_overlay(
        overlay,
        monkeypatch,
        preflight_module,
        lambda payload: payload["worker_image"]["source_snapshot"].update(
            {"source_tree_sha256": _digest("0")}
        ),
    )
    with pytest.raises(
        preflight_module.R241ServiceChainPreflightError,
        match="worker image does not bind this immutable source snapshot",
    ):
        _validate(
            preflight_module,
            stage="trainer",
            overlay=overlay,
            overlay_sha=changed_sha,
            mirror_receipt=mirror_receipt,
            mirror_receipt_sha=mirror_receipt_sha,
        )


def test_preflight_rejects_actual_baseline_snapshot_schema_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight_module
) -> None:
    paths, overlay, _sealed_sha, mirror_receipt, mirror_receipt_sha = _fixture(
        tmp_path, monkeypatch, preflight_module
    )
    old_root = paths["baseline_root"]
    old_manifest = paths["baseline_manifest"]
    old_manifest.chmod(0o644)
    payload = json.loads(old_manifest.read_text(encoding="utf-8"))
    payload["schema"] = "poke_bot.r241_baseline_payload_snapshot/v1"
    old_manifest.write_bytes(_canonical(payload))
    old_manifest.chmod(0o444)
    new_manifest_sha = _sha(old_manifest, preflight_module)
    new_root = old_root.parent / (
        preflight_module.BASELINE_ROOT_PREFIX + new_manifest_sha.removeprefix("sha256:")[:16]
    )
    old_root.chmod(0o755)
    old_root.rename(new_root)
    new_root.chmod(0o555)
    new_manifest = new_root / preflight_module.BASELINE_MANIFEST_FILENAME
    monkeypatch.setenv("R241_INZI_BASELINES_ROOT", str(new_root))
    monkeypatch.setenv("R241_INZI_BASELINE_PAYLOAD_MANIFEST", str(new_manifest))
    monkeypatch.setenv("R241_INZI_BASELINE_PAYLOAD_MANIFEST_SHA256", new_manifest_sha)

    changed_sha = _replace_overlay(
        overlay,
        monkeypatch,
        preflight_module,
        lambda activation: activation["baseline_payloads"]["hosts"]["inzi"].update(
            {"root": str(new_root), "manifest": str(new_manifest), "manifest_sha256": new_manifest_sha}
        ),
    )
    with pytest.raises(preflight_module.R241ServiceChainPreflightError, match="baseline payload snapshot schema"):
        _validate(
            preflight_module,
            stage="trainer",
            overlay=overlay,
            overlay_sha=changed_sha,
            mirror_receipt=mirror_receipt,
            mirror_receipt_sha=mirror_receipt_sha,
        )


def test_preflight_rejects_a_symlinked_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight_module
) -> None:
    _paths, overlay, overlay_sha, mirror_receipt, mirror_receipt_sha = _fixture(
        tmp_path, monkeypatch, preflight_module
    )
    link = overlay.with_name("r241-inzi-activation-overlay-link.json")
    link.symlink_to(overlay.name)
    monkeypatch.setenv("R241_INZI_ACTIVATION_OVERLAY", str(link))
    with pytest.raises(preflight_module.R241ServiceChainPreflightError, match="must not be a symlink"):
        _validate(
            preflight_module,
            stage="trainer",
            overlay=link,
            overlay_sha=overlay_sha,
            mirror_receipt=mirror_receipt,
            mirror_receipt_sha=mirror_receipt_sha,
        )


def test_preflight_rejects_a_mirror_receipt_that_does_not_bind_the_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflight_module
) -> None:
    _paths, overlay, overlay_sha, mirror_receipt, _mirror_receipt_sha = _fixture(
        tmp_path, monkeypatch, preflight_module
    )
    mirror_receipt.chmod(0o644)
    payload = json.loads(mirror_receipt.read_text(encoding="utf-8"))
    payload["byte_identical_copy_verified"] = False
    mirror_receipt.write_bytes(_canonical(payload))
    mirror_receipt.chmod(0o444)
    changed_mirror_sha = _sha(mirror_receipt, preflight_module)
    monkeypatch.setenv(
        "R241_INZI_ACTIVATION_OVERLAY_MIRROR_RECEIPT_SHA256", changed_mirror_sha
    )

    with pytest.raises(
        preflight_module.R241ServiceChainPreflightError,
        match="mirror receipt does not bind",
    ):
        _validate(
            preflight_module,
            stage="trainer",
            overlay=overlay,
            overlay_sha=overlay_sha,
            mirror_receipt=mirror_receipt,
            mirror_receipt_sha=changed_mirror_sha,
        )


def test_templates_remain_regular_inert_artifacts() -> None:
    for path in UNITS.values():
        mode = path.stat().st_mode
        assert stat.S_ISREG(mode)
        assert os.access(path, os.R_OK)
