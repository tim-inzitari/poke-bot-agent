"""Focused create-only tests for the r241 Elmo checkpoint transport producer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import stage_r241_elmo_checkpoint_transport as stage


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.encode("utf-8") if isinstance(payload, str) else payload)
    return path


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    checkpoint = _write(tmp_path / "expert_before_iter_00021.pt", b"exact r195 checkpoint")
    tree_payload = {
        "runtime_contract": {
            "accepted_archetype_ids": ["alakazam", "crustle"],
            "one_route_per_decision": True,
            "unknown_route_exact_bypass": True,
            "consecutive_required": 1,
            "adapter_format": "poke-bot-matchup-adapter-bank-v6",
            "route_target_ids": [f"slot-{index}" for index in range(20)],
            "route_physical_slots": list(range(20)),
            "physical_slot_capacity": 64,
            "slot_registry_digest": "sha256:" + "a" * 64,
            "zero_materialized_adapters_allowed": False,
        }
    }
    tree = _write(
        tmp_path / "pinned-checkpoint-compatible-matchup-tree-v2.json",
        json.dumps(tree_payload, sort_keys=True),
    )
    monkeypatch.setattr(stage, "R195_CHECKPOINT_SHA256", _sha(checkpoint))
    monkeypatch.setattr(stage, "R195_CHECKPOINT_BYTES", checkpoint.stat().st_size)
    monkeypatch.setattr(stage, "R195_E60_TREE_SHA256", _sha(tree))
    owner_payload = {
        "schema": stage.R241_OWNER_SCHEMA,
        "latest_owner_clarification_revision": stage.R241_OWNER_CLARIFICATION_REVISION,
        "parent": {
            "checkpoint_sha256": _sha(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "immutable": True,
        },
    }
    owner = _write(tmp_path / "owner-contract.json", json.dumps(owner_payload))
    return {"checkpoint": checkpoint, "tree": tree, "owner": owner}


def test_validate_only_derives_exact_flat_transport_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    result = stage.stage_r241_elmo_checkpoint_transport(
        checkpoint=inputs["checkpoint"],
        matchup_tree=inputs["tree"],
        owner_contract=inputs["owner"],
        owner_contract_sha256=_sha(inputs["owner"]),
        ssh_target="admin@elmo",
        controller_receipt=tmp_path / "controller-receipt.json",
        execute=False,
    )

    assert result["status"] == "validated_not_staged"
    assert result["worker_started"] is False
    assert result["service_started"] is False
    assert result["remote_listener_started"] is False
    assert result["initial_checkpoint"] == {
        "container_path": "/workspace/checkpoint/expert_before_iter_00021."
        + _sha(inputs["checkpoint"]).removeprefix("sha256:")[:16]
        + ".pt",
        "sha256": _sha(inputs["checkpoint"]),
    }
    assert result["runtime_companions"]["learner_matchup_tree"] == {
        "container_path": "/workspace/checkpoint/"
        "pinned-checkpoint-compatible-matchup-tree-v2.json",
        "sha256": _sha(inputs["tree"]),
    }
    assert not (tmp_path / "controller-receipt.json").exists()


def test_execute_writes_create_only_controller_receipt_and_stages_only_three_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    published: list[tuple[str, str, str]] = []
    verified: dict[str, object] = {}
    remote_receipt_digest: dict[str, str] = {}
    controller_receipt = tmp_path / "controller-receipt.json"

    def publish(**kwargs: object) -> None:
        source = kwargs["source"]
        destination = kwargs["destination"]
        digest = kwargs["digest"]
        assert isinstance(source, Path)
        published.append((source.name, str(destination), str(digest)))
        if destination == stage.DEFAULT_ELMO_STAGING_RECEIPT:
            assert not controller_receipt.exists()
            remote_receipt_digest["value"] = str(digest)

    def verify(**kwargs: object) -> None:
        verified.update(kwargs)

    monkeypatch.setattr(stage, "_stage_remote_file_create_only", publish)
    monkeypatch.setattr(stage, "_verify_remote_transport_directory", verify)
    monkeypatch.setattr(
        stage,
        "_remote_file_state",
        lambda _target, _path: remote_receipt_digest["value"],
    )
    result = stage.stage_r241_elmo_checkpoint_transport(
        checkpoint=inputs["checkpoint"],
        matchup_tree=inputs["tree"],
        owner_contract=inputs["owner"],
        owner_contract_sha256=_sha(inputs["owner"]),
        ssh_target="admin@elmo",
        controller_receipt=controller_receipt,
        execute=True,
    )

    assert result["status"] == "passed"
    assert _sha(controller_receipt) == result["staging_receipt_sha256"]
    receipt = json.loads(controller_receipt.read_text(encoding="utf-8"))
    assert receipt["schema"] == stage.R241_CHECKPOINT_TRANSPORT_STAGING_SCHEMA
    assert receipt["checkpoint_transport"]["verification_endpoint"] == "192.168.1.143:8767"
    assert receipt["checkpoint_transport"]["container_root"] == "/workspace/checkpoint"
    assert receipt["runtime_companions"]["matchup_runtime_activation"]["schema"] == (
        stage.R241_MATCHUP_RUNTIME_MARKER_SCHEMA
    )
    assert [name for name, _destination, _digest in published] == [
        inputs["checkpoint"].name,
        inputs["tree"].name,
        stage.R241_MATCHUP_RUNTIME_MARKER,
        stage.DEFAULT_ELMO_STAGING_RECEIPT.name,
    ]
    assert verified["expected_names"] == {
        stage._digest_addressed_basename(inputs["checkpoint"], _sha(inputs["checkpoint"])),
        inputs["tree"].name,
        stage.R241_MATCHUP_RUNTIME_MARKER,
    }


def test_transport_rejects_noncanonical_output_paths(tmp_path: Path) -> None:
    with pytest.raises(stage.R241CheckpointTransportStageError, match="dedicated Elmo"):
        stage._require_expected_elmo_paths(
            host_root="/srv/checkpoints",
            receipt_path=stage.DEFAULT_ELMO_STAGING_RECEIPT,
        )


def test_conflicting_remote_member_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write(tmp_path / "source.pt", b"immutable")
    digest = _sha(source)
    monkeypatch.setattr(stage, "_remote_file_state", lambda *_args: "sha256:" + "f" * 64)

    with pytest.raises(stage.R241CheckpointTransportStageError, match="conflicting identity"):
        stage._stage_remote_file_create_only(
            target="admin@elmo",
            source=source,
            destination=stage.DEFAULT_ELMO_TRANSPORT_ROOT / "source.immutable.pt",
            digest=digest,
        )


def test_producer_uses_hard_link_create_only_publication_not_force_move() -> None:
    source = (Path(stage.__file__).read_text(encoding="utf-8"))
    assert 'ln -- "$tmp" "$dst"' in source
    assert "mv -f" not in source
    assert "BatchMode=yes" in source
