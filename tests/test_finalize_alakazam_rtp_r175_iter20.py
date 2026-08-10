from scripts.finalize_alakazam_rtp_r175_iter20 import (
    H10_MARNIE_ARCHETYPE,
    H10_MARNIE_CONTENT_DIGEST,
    H10_MARNIE_CHECKPOINT_DIGEST,
    H10_MARNIE_FLOOR,
    H10_MARNIE_ID,
    _canonical_digest,
    _completion_disposition,
    _service_inactive,
    _validate_collection_plan_and_h10_rows,
    _validate_r192_contract,
    sha256,
)

import json


def test_terminal_source_identity_digest_is_order_independent() -> None:
    left = {"commit": "x", "evaluation": "y"}
    right = {"evaluation": "y", "commit": "x"}
    assert _canonical_digest(left) == _canonical_digest(right)


def test_terminal_source_identity_digest_changes_with_evidence() -> None:
    assert _canonical_digest({"commit": "x"}) != _canonical_digest({"commit": "y"})


def test_measured_gate_pass_is_registered_as_pass() -> None:
    result = _completion_disposition(True)
    assert result["completion_authority"] == "measured_gate_pass"
    assert result["measured_gate_pass"] is True
    assert result["failed_gate_results_preserved"] is False


def test_owner_ceiling_never_claims_measured_pass() -> None:
    result = _completion_disposition(False)
    assert result["completion_authority"] == "explicit_owner_ceiling_acceptance"
    assert result["measured_gate_pass"] is False
    assert result["current_gate_pass"] is False
    assert result["failed_gate_results_preserved"] is True


def test_terminal_service_must_be_inactive_with_no_main_pid(monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = "MainPID=0\nActiveState=inactive\n"

    monkeypatch.setattr(
        "scripts.finalize_alakazam_rtp_r175_iter20.subprocess.run",
        lambda *args, **kwargs: Result(),
    )
    assert _service_inactive("trainer.service") is True


def test_terminal_service_rejects_live_main_pid(monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = "MainPID=123\nActiveState=active\n"

    monkeypatch.setattr(
        "scripts.finalize_alakazam_rtp_r175_iter20.subprocess.run",
        lambda *args, **kwargs: Result(),
    )
    assert _service_inactive("trainer.service") is False


def _h10_collection_fixture(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan = {
        "schema": "poke_bot.strong_public_practice_plan/v1",
        "iteration": 20,
        "games": 4586,
        "minimum_games_by_opponent": {H10_MARNIE_ID: H10_MARNIE_FLOOR},
        "per_opponent": {
            H10_MARNIE_ID: {
                "archetype_id": H10_MARNIE_ARCHETYPE,
                "games": H10_MARNIE_FLOOR,
                "minimum_games": H10_MARNIE_FLOOR,
                "seat0": H10_MARNIE_FLOOR // 2,
                "seat1": H10_MARNIE_FLOOR // 2,
            }
        },
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    shard_path = tmp_path / "shard.jsonl"
    row = {
        "source": "pure_rl",
        "target_provenance": {
            "collect": "strong_public_practice",
            "opponent_training_group": "strong_public_practice",
            "opponent_id": H10_MARNIE_ID,
            "opponent_archetype_id": H10_MARNIE_ARCHETYPE,
            "opponent_content_digest": H10_MARNIE_CONTENT_DIGEST,
            "strong_public_practice_floor_games": H10_MARNIE_FLOOR,
            "target_source": "recursive_turn_planner",
            "trusted": True,
            "matchup_runtime_audit": {"runtime_enabled": True},
        },
    }
    shard_path.write_text(
        "".join(json.dumps(row) + "\n" for _ in range(H10_MARNIE_FLOOR)),
        encoding="utf-8",
    )
    collection = {
        "shard": {
            "path": str(shard_path),
            "size": shard_path.stat().st_size,
            "sha256": sha256(shard_path),
        }
    }
    stats = {
        "strong_public_practice_plan": str(plan_path),
        "strong_public_practice_record_receipt": {
            "passed": True,
            "expected_results": 4586,
            "successful_results": 4586,
            "canonical_records_written": 4586,
        },
    }
    return collection, stats


def test_terminal_h10_floor_binds_plan_shard_digest_and_provenance(tmp_path) -> None:
    collection, stats = _h10_collection_fixture(tmp_path)
    proof = _validate_collection_plan_and_h10_rows(collection, stats)
    assert proof["h10_marnie_games"] == H10_MARNIE_FLOOR
    assert proof["shard_sha256"] == collection["shard"]["sha256"]


def test_terminal_h10_floor_rejects_changed_content_digest(tmp_path) -> None:
    collection, stats = _h10_collection_fixture(tmp_path)
    shard_path = tmp_path / "shard.jsonl"
    body = shard_path.read_text(encoding="utf-8").replace(
        H10_MARNIE_CONTENT_DIGEST, "sha256:" + "0" * 64
    )
    shard_path.write_text(body, encoding="utf-8")
    collection["shard"]["size"] = shard_path.stat().st_size
    collection["shard"]["sha256"] = sha256(shard_path)

    try:
        _validate_collection_plan_and_h10_rows(collection, stats)
    except RuntimeError as exc:
        assert "retained provenance" in str(exc)
    else:
        raise AssertionError("changed H10 content digest was accepted")


def test_terminal_r192_contract_binds_checkpoint_package_floor_and_transport(
    tmp_path,
) -> None:
    path = tmp_path / "r192.json"
    path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.alakazam_marnie_splusplus_opponent_r192/v1",
                "owner_decision_revision": 192,
                "status": "activated_iteration_17_runtime_dispatch_verified",
                "opponent": {
                    "opponent_id": H10_MARNIE_ID,
                    "archetype_id": H10_MARNIE_ARCHETYPE,
                    "checkpoint_sha256": H10_MARNIE_CHECKPOINT_DIGEST,
                    "content_digest": H10_MARNIE_CONTENT_DIGEST,
                    "tier": "S++",
                    "weight": 4.0,
                    "floor_games_per_set": H10_MARNIE_FLOOR,
                },
                "collection_contract": {
                    "games_per_iteration": 8196,
                    "self_play_mirrors": 1024,
                    "public_mix_games": 7172,
                    "strong_public_practice_games": 4586,
                },
                "transport": {
                    "activation_training_group": "strong_public_practice",
                    "dispatch_mode": "singleton_remote_play",
                    "r182_default_deny_unchanged": True,
                },
            }
        ),
        encoding="utf-8",
    )
    assert _validate_r192_contract(path) == sha256(path)

    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["opponent"]["checkpoint_sha256"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(changed), encoding="utf-8")
    try:
        _validate_r192_contract(path)
    except RuntimeError as exc:
        assert "r192" in str(exc)
    else:
        raise AssertionError("changed r192 checkpoint identity was accepted")
