from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_alakazam_stall_reward_iteration_r333.py"


def _load():
    spec = importlib.util.spec_from_file_location("stall_iteration", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path):
    module = _load()
    contract = tmp_path / "contract.json"
    required = [
        "activation_boundary_iteration",
        "source_module_sha256",
        "maximum_complete_turns_without_win_progress",
        *module.EXPECTED_FIELDS,
    ]
    _write(
        contract,
        {
            "goal_revision": 31,
            "root_handoff_revision": 333,
            "revision_31_stall_reward_and_submission_mirror_gate": {
                "training_only_stall_reward": {"required_receipt_fields": required}
            },
        },
    )
    deployment = tmp_path / "deployment"
    source = deployment / "poke_bot/pure_rl/no_progress.py"
    source.parent.mkdir(parents=True)
    source.write_text("# exact source\n", encoding="utf-8")
    commit = tmp_path / "commits/iter_00005.json"
    _write(commit, {"next_iteration": 6})
    checkpoint = tmp_path / "iter_00005.pt"
    checkpoint.write_bytes(b"activated learner checkpoint")
    activation = tmp_path / "activation.json"
    _write(
        activation,
        {
            "schema": "poke_bot.alakazam_stall_reward_boundary_activation/v1",
            "status": "passed_clean_iteration_5_boundary_before_iteration_6",
            "goal_contract_sha256": module.sha256_file(contract),
            "activation_boundary_iteration": 5,
            "first_eligible_collection_iteration": 6,
            "maximum_complete_turns_without_win_progress": 64,
            "source_module_sha256": module.sha256_file(source),
            "deployment_root": str(deployment),
            "boundary_commit_path": str(commit),
            "boundary_commit_sha256": module.sha256_file(commit),
            "learner_checkpoint_sha256": module.sha256_file(checkpoint),
        },
    )
    shard = tmp_path / "iter_00006.jsonl"
    shard.write_text("{}\n", encoding="utf-8")
    stat = shard.stat()
    collection = tmp_path / "collection.json"
    _write(
        collection,
        {
            "schema": "poke_bot.completed_collection/v1",
            "iteration": 6,
            "requested_games": 8196,
            "source_games": 8196,
            "checkpoint": str(checkpoint),
            "checkpoint_digest": module.sha256_file(checkpoint),
            "shard": {
                "path": str(shard),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": module.sha256_file(shard),
            },
            "stats": {
                "no_progress_stall_receipt": {
                    "maximum_complete_turns_without_win_progress": 64,
                    "stall_games": 1,
                    "stall_rows_by_seat": {"0": 1, "1": 1},
                    "stagnant_turn_distribution": {"64": 1},
                    "training_return_override_counts": {"-1.0": 2},
                    "ordinary_completed_result_counts": {"0": 8195},
                    "disabled_path_parity": False,
                }
            },
        },
    )
    return module, contract, activation, collection


def test_validates_complete_active_iteration(tmp_path: Path) -> None:
    module, contract, activation, collection = _fixture(tmp_path)
    result = module.validate(
        contract_path=contract,
        expected_contract_sha256=module.sha256_file(contract),
        activation_path=activation,
        collection_path=collection,
    )
    assert result["status"] == "passed_iteration_6_stall_reward_collection"
    assert result["stall_rows_by_seat"] == {"0": 1, "1": 1}


def test_rejects_incomplete_stall_telemetry(tmp_path: Path) -> None:
    module, contract, activation, collection = _fixture(tmp_path)
    value = json.loads(collection.read_text(encoding="utf-8"))
    del value["stats"]["no_progress_stall_receipt"]["stagnant_turn_distribution"]
    _write(collection, value)
    try:
        module.validate(
            contract_path=contract,
            expected_contract_sha256=module.sha256_file(contract),
            activation_path=activation,
            collection_path=collection,
        )
    except RuntimeError as exc:
        assert "active stall contract" in str(exc)
    else:
        raise AssertionError("incomplete telemetry was accepted")


def test_validates_later_active_iteration_checkpoint(tmp_path: Path) -> None:
    module, contract, activation, collection = _fixture(tmp_path)
    value = json.loads(collection.read_text(encoding="utf-8"))
    checkpoint = tmp_path / "iter_00006.pt"
    checkpoint.write_bytes(b"later learner checkpoint")
    value["iteration"] = 7
    value["checkpoint"] = str(checkpoint)
    value["checkpoint_digest"] = module.sha256_file(checkpoint)
    _write(collection, value)
    result = module.validate(
        contract_path=contract,
        expected_contract_sha256=module.sha256_file(contract),
        activation_path=activation,
        collection_path=collection,
        expected_iteration=7,
    )
    assert result["status"] == "passed_iteration_7_stall_reward_collection"
    assert result["collection_checkpoint_sha256"] == module.sha256_file(checkpoint)


def test_rejects_later_iteration_checkpoint_drift(tmp_path: Path) -> None:
    module, contract, activation, collection = _fixture(tmp_path)
    value = json.loads(collection.read_text(encoding="utf-8"))
    checkpoint = tmp_path / "iter_00006.pt"
    checkpoint.write_bytes(b"later learner checkpoint")
    value["iteration"] = 7
    value["checkpoint"] = str(checkpoint)
    value["checkpoint_digest"] = "sha256:" + "b" * 64
    _write(collection, value)
    try:
        module.validate(
            contract_path=contract,
            expected_contract_sha256=module.sha256_file(contract),
            activation_path=activation,
            collection_path=collection,
            expected_iteration=7,
        )
    except RuntimeError as exc:
        assert "checkpoint identity drifted" in str(exc)
    else:
        raise AssertionError("drifted later-iteration checkpoint was accepted")
