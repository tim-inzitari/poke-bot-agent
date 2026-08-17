"""Focused tests for the local-only r260 pre-start supersession gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_r260_prestart_supersession_gate.py"


def _module():
    spec = importlib.util.spec_from_file_location("r260_prestart_gate_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    return path


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _unit(unit_id: str) -> dict[str, object]:
    return {
        "id": unit_id,
        "load_state": "loaded",
        "active_state": "inactive",
        "sub_state": "dead",
        "exec_main_start_timestamp": "",
        "active_enter_timestamp": "",
        "invocation_id": "",
        "n_restarts": 0,
    }


def _artifacts() -> dict[str, list[object]]:
    return {
        "training_updates": [],
        "checkpoints": [],
        "commits": [],
        "loop_state": [],
        "logs": [],
        "packages": [],
        "finalizers": [],
        "queues": [],
        "uploads": [],
        "submissions": [],
    }


def _observation(module, host: str) -> dict[str, object]:
    if host == "inzi":
        units = [
            _unit("pokebot-alakazam-new-list-direct-r241.service"),
            _unit("pokebot-alakazam-new-list-direct-r241-finalize.service"),
            _unit("pokebot-alakazam-new-list-direct-r241-submission-queue.service"),
            _unit("pokebot-alakazam-new-list-direct-r241-upload.service"),
        ]
        return {
            "schema": module.OBSERVATION_SCHEMA,
            "host": host,
            "observed_at_utc": "2026-08-11T12:51:08Z",
            "scope": "managed_r241_execution_topology",
            "units": units,
            "arm_paths": [{"path": "/not-an-arm-on-inzi", "exists": False}],
            "execution_artifacts": _artifacts(),
            "processes": {"matching_count": 0},
            "container_route": {
                "managed_service_started": False,
                "arm_present": False,
                "listener_count": 0,
                "direct_daemon_inventory": "verified_no_current_or_retained_r241_container",
                "events_since_r241_staging": "verified_zero_matches",
            },
            "journal_history": {
                "readable": True,
                "coverage_start_utc": "2026-08-10T00:00:00Z",
                "matching_r241_unit_entries": 0,
            },
            "listener_count": None,
            "selector_r241_active_reference": False,
        }
    return {
        "schema": module.OBSERVATION_SCHEMA,
        "host": host,
        "observed_at_utc": "2026-08-11T12:51:08Z",
        "scope": "managed_r241_execution_topology",
        "units": [_unit("pokebot-r241-elmo-official-r236-remote-worker.service")],
        "arm_paths": [{"path": "/r241/REMOTE_WORKER_ARMED", "exists": False}],
        "execution_artifacts": _artifacts(),
        "processes": {"matching_count": 0},
        "container_route": {
            "managed_service_started": False,
            "arm_present": False,
            "listener_count": 0,
            "direct_daemon_inventory": "verified_no_current_or_retained_r241_container",
            "events_since_r241_staging": "verified_zero_matches",
        },
        "journal_history": {
            "readable": True,
            "coverage_start_utc": "2026-08-10T00:00:00Z",
            "matching_r241_unit_entries": 0,
        },
        "listener_count": 0,
        "selector_r241_active_reference": None,
    }


def _inputs(tmp_path: Path, module):
    r241 = _write_json(
        tmp_path / "r241.json",
        {
            "schema": module.R241_SCHEMA,
            "candidate_id": module.CANDIDATE_ID,
            "latest_owner_clarification_revision": 262,
            "own_deck_head_structure_import": {
                "owner_revision": 260,
                "provenance_manifest": "state/alakazam-own-deck-ledger-successor-r258.json",
                "pre_start_override": {
                    "supersedes_r258_wait_for_r241_completion_only_for_this_successor_boundary": True
                },
                "migration": {
                    "zero_safe": True,
                    "typed_parent_file_identity_keys": ["path", "sha256", "size_bytes"],
                },
                "training_placement": {
                    "owner_revision": 262,
                    "sole_managed_training_host": "inzi",
                    "elmo_may_train_learner": False,
                    "canonical_inzi_training_root": module.INZI_TRAINING_ROOT,
                    "inzi_prefix_staging_root": module.INZI_PREFIX_STAGING_ROOT,
                    "prefix_transfer_while_elmo_builder_runs": True,
                    "partial_staging_root_training_eligible": False,
                    "trainer_may_consume_elmo_mnt_main_path": False,
                    "healthy_r259_service_may_be_stopped_restarted_or_reconfigured": False,
                },
            },
        },
    )
    r258 = _write_json(
        tmp_path / "r258.json",
        {
            "schema": module.R258_SCHEMA,
            "canonical": True,
            "candidate_id": "alakazam-own-deck-ledger-successor-r258",
        },
    )
    r195 = _write_json(
        tmp_path / "r195.json",
        {
            "schema": module.R195_SCHEMA,
            "completion": {
                "expert_checkpoint": "/immutable/r195.pt",
                "expert_checkpoint_sha256": "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a",
                "expert_checkpoint_bytes": 127914385,
            },
        },
    )
    ledger = _write_json(
        tmp_path / "legacy.json",
        {
            "schema": module.LEGACY_LEDGER_SCHEMA,
            "candidate_id": module.CANDIDATE_ID,
            "status": "activation_overlay_published_and_mirrored_services_not_started",
            "execution_boundary": {
                "environment_installed": False,
                "unit_installed": False,
                "service_started": False,
                "worker_armed": False,
                "listener_started": False,
                "training_started": False,
                "submission_started": False,
            },
            "h10_v8": {
                "inzi_receipt_sha256": "sha256:" + "1" * 64,
                "elmo_receipt_sha256": "sha256:" + "2" * 64,
            },
            "peak_v6": {
                "inzi_receipt_sha256": "sha256:" + "3" * 64,
                "elmo_receipt_sha256": "sha256:" + "4" * 64,
            },
            "worker_image_r13": {
                "image_id_sha256": "sha256:" + "5" * 64,
                "receipt_sha256": "sha256:" + "6" * 64,
            },
            "canonical_worker_preflight": {
                "host_sha256": "sha256:" + "7" * 64,
                "runtime_sha256": "sha256:" + "8" * 64,
                "gameplay_sha256": "sha256:" + "9" * 64,
                "manifest_sha256": "sha256:" + "a" * 64,
                "deployment_action": "not_started",
            },
        },
    )
    inzi = _write_json(tmp_path / "inzi.json", _observation(module, "inzi"))
    elmo = _write_json(tmp_path / "elmo.json", _observation(module, "elmo"))
    return {
        "r241_contract": r241,
        "r258_contract": r258,
        "r195_contract": r195,
        "legacy_execution_ledger": ledger,
        "inzi_observation": inzi,
        "elmo_observation": elmo,
    }


def test_builds_create_only_receipt_for_clean_managed_prestart_boundary(tmp_path: Path) -> None:
    module = _module()
    inputs = _inputs(tmp_path, module)
    receipt = module.build_receipt(**inputs)
    assert receipt["status"] == "passed_prestart_supersession_for_managed_r241_topology"
    assert receipt["latest_owner_clarification_revision"] == 262
    assert receipt["parent_file_identity"]["size_bytes"] == 127914385
    assert receipt["host_observations"]["elmo"]["listener_count"] == 0
    output = tmp_path / "receipt.json"
    first = module._create_only_json(output, receipt)
    second = module._create_only_json(output, receipt)
    assert first == second == output.resolve()
    assert _sha(output) == module._sha256_file(output)
    reopened = module.validate_sealed_receipt(
        receipt_path=output,
        expected_sha256=_sha(output),
        expected_r241_contract_sha256=_sha(inputs["r241_contract"]),
    )
    assert reopened["candidate_id"] == module.CANDIDATE_ID
    assert not output.stat().st_mode & stat.S_IWUSR


def test_rejects_started_managed_unit(tmp_path: Path) -> None:
    module = _module()
    inputs = _inputs(tmp_path, module)
    observation = json.loads(inputs["elmo_observation"].read_text(encoding="utf-8"))
    observation["units"][0]["exec_main_start_timestamp"] = "2026-08-11T12:00:00Z"
    inputs["elmo_observation"].chmod(0o644)
    inputs["elmo_observation"].write_text(json.dumps(observation), encoding="utf-8")
    inputs["elmo_observation"].chmod(0o444)
    with pytest.raises(module.R260PrestartGateError, match="was started"):
        module.build_receipt(**inputs)


def test_rejects_any_r241_execution_artifact_or_active_selector(tmp_path: Path) -> None:
    module = _module()
    inputs = _inputs(tmp_path, module)
    observation = json.loads(inputs["inzi_observation"].read_text(encoding="utf-8"))
    observation["execution_artifacts"]["commits"] = ["commits/iter_00000.json"]
    inputs["inzi_observation"].chmod(0o644)
    inputs["inzi_observation"].write_text(json.dumps(observation), encoding="utf-8")
    inputs["inzi_observation"].chmod(0o444)
    with pytest.raises(module.R260PrestartGateError, match="execution artifacts"):
        module.build_receipt(**inputs)


def test_fails_closed_without_retained_docker_and_journal_evidence(tmp_path: Path) -> None:
    module = _module()
    inputs = _inputs(tmp_path, module)
    observation = json.loads(inputs["elmo_observation"].read_text(encoding="utf-8"))
    observation["container_route"]["direct_daemon_inventory"] = (
        "unavailable_unprivileged_read_only"
    )
    inputs["elmo_observation"].chmod(0o644)
    inputs["elmo_observation"].write_text(json.dumps(observation), encoding="utf-8")
    inputs["elmo_observation"].chmod(0o444)
    with pytest.raises(module.R260PrestartGateError, match="container route"):
        module.build_receipt(**inputs)


def test_sealed_receipt_rejects_different_owner_contract_digest(tmp_path: Path) -> None:
    module = _module()
    inputs = _inputs(tmp_path, module)
    output = tmp_path / "receipt.json"
    module._create_only_json(output, module.build_receipt(**inputs))
    with pytest.raises(module.R260PrestartGateError, match="different r241 contract"):
        module.validate_sealed_receipt(
            receipt_path=output,
            expected_sha256=_sha(output),
            expected_r241_contract_sha256="sha256:" + "f" * 64,
        )
