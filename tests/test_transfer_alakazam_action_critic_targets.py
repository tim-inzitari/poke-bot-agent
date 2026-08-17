"""Safety coverage for the portable revision-21 target-only transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import transfer_alakazam_action_critic_targets as transfer


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _sha(value: bytes | str) -> str:
    body = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _write(path: Path, body: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return _sha(body)


def _write_content_addressed(directory: Path, body: bytes, suffix: str) -> tuple[Path, str]:
    digest = _sha(body)
    path = directory / f"sha256-{digest.removeprefix('sha256:')}{suffix}"
    _write(path, body)
    return path, digest


def _contract() -> dict[str, object]:
    return {
        "schema": "poke_bot.alakazam_elmo_rule_derivative_goal/v1",
        "goal_revision": 22,
        "revision_21_draw_safe_critic_actor_canary": {
            "owner_goal_revision": 21,
            "target_overlay": {
                "schema": transfer.TARGET_OVERLAY_SCHEMA,
                "manifest_schema": transfer.TARGET_SET_MANIFEST_SCHEMA,
                "row_join_identity": [
                    "utc_day",
                    "source_archive_sha256",
                    "source_member",
                    "episode_id",
                    "acting_seat",
                    "env_step",
                    "program_identity",
                ],
                "group_key": ["source_archive_sha256", "episode_id", "acting_seat"],
                "group_order": "strictly_increasing_env_step_no_duplicates",
                "required_terminal_fields": [
                    "z",
                    "z_mask",
                    "win_target_one_only_for_z_plus1",
                    "win_target_mask",
                ],
                "required_per_horizon_fields": [
                    "h",
                    "mask",
                    "unavailable_reason",
                    "future_program_identity",
                    "future_env_step",
                    "own_remaining_before",
                    "own_remaining_after",
                    "opponent_remaining_before",
                    "opponent_remaining_after",
                    "own_taken",
                    "opponent_taken",
                    "differential",
                ],
                "prize_count": {
                    "valid_inclusive_range": [1, 6],
                    "zero_behavior": "mask_as_setup_or_uninitialized_never_treat_as_real_zero_progress",
                },
                "horizon_definition": {
                    "values": [1, 2, 3],
                    "target": "clip((own_taken-opponent_taken)/3,-1,+1)",
                },
                "hidden_information_simulator_search_rtp_mcts_or_unchosen_targets_allowed": False,
            },
            "actor_advantage": {
                "enabled_formula": "(z-V_existing(s))+0.05*m1*(Q_prize^1(s,a)-V_prize^1(s))",
                "complete_action_value_broadcast_identically_across_selected_factorized_stages": True,
                "actor_gradient_into_sidecar_allowed": False,
            },
        },
        "revision_20_conservative_critic_actor_canary": {
            "elmo_to_bert_bootstrap_transfer": {
                "source_host": "elmo",
                "destination_host": "bert",
                "source_read_only": True,
                "parallel_lanes_exact": 4,
                "unique_source_object_per_lane": True,
                "raw_episode_zip_transfer_required": False,
                "bert_disk_free_floor_bytes": transfer.DEFAULT_DISK_FLOOR_BYTES,
            }
        },
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    """Build a sealed portable 20-day target set without any raw ZIP object."""

    source_root = tmp_path / "elmo-target-set"
    destination_root = tmp_path / "bert-target-view"
    contract_body = _canonical(_contract())
    contract_path, contract_sha = _write_content_addressed(
        source_root / "bindings", contract_body, ".goal-contract.json"
    )
    base_body = _canonical({"schema": transfer.BASE_COMPLETION_SCHEMA, "packs": []})
    base_path, base_sha = _write_content_addressed(
        source_root / "bindings", base_body, ".base-pack-completion.json"
    )
    overlay_shards: list[dict[str, object]] = []
    raw_zip_identities: list[dict[str, object]] = []
    target_shards: list[dict[str, object]] = []
    days: list[dict[str, object]] = []
    for index, day in enumerate(transfer.WINDOW_DAYS):
        split = transfer.SPLIT_BY_DAY[day]
        overlay_sha = _sha(f"overlay:{day}")
        raw_sha = _sha(f"raw-provenance:{day}")
        overlay_shards.append(
            {"utc_day": day, "split": split, "sha256": overlay_sha, "size_bytes": index + 1}
        )
        raw_zip_identities.append(
            {"utc_day": day, "sha256": raw_sha, "size_bytes": 1000 + index}
        )
    overlay_body = _canonical(
        {"schema": transfer.OVERLAY_MANIFEST_SCHEMA, "overlay_shards": overlay_shards}
    )
    overlay_path, overlay_manifest_sha = _write_content_addressed(
        source_root / "bindings", overlay_body, ".complete-action-overlay-manifest.json"
    )
    for index, day in enumerate(transfer.WINDOW_DAYS):
        split = transfer.SPLIT_BY_DAY[day]
        root_relative = f"days/{day}"
        day_root = source_root / root_relative
        shard_body = _canonical(
            {
                "schema": transfer.TARGET_OVERLAY_SCHEMA,
                "owner_goal_revision": 21,
                "utc_day": day,
                "split": split,
                "program_identity": f"program:{day}",
                "target_only": True,
            }
        )
        shard_path, shard_sha = _write_content_addressed(
            day_root / "objects", shard_body, ".action-critic-targets.jsonl"
        )
        shard = {
            "path": str(shard_path.relative_to(day_root)),
            "sha256": shard_sha,
            "size_bytes": shard_path.stat().st_size,
            "row_count": 1,
        }
        target_shards.append({"utc_day": day, "split": split, **shard})
        schema_body = _canonical(
            {
                "schema": "poke_bot.alakazam_action_critic_target_schema/v1",
                "owner_goal_revision": 21,
                "day": day,
            }
        )
        schema_path, schema_sha = _write_content_addressed(
            day_root / "schemas", schema_body, ".target-schema.json"
        )
        overlay = overlay_shards[index]
        raw = raw_zip_identities[index]
        coverage = {"counts": {"complete_action_programs": 1, "prize_h1_masked": 1}}
        day_manifest = {
            "schema": transfer.TARGET_DAY_MANIFEST_SCHEMA,
            "owner_goal_revision": 21,
            "goal_contract": {
                "sha256": contract_sha,
                "goal_revision": 22,
                "critic_semantic_owner_goal_revision": 21,
                "required_authority": "revision_21_draw_safe_critic_actor_canary",
            },
            "utc_day": day,
            "split": split,
            "complete_action_overlay": {"sha256": overlay["sha256"]},
            "raw_episode_zip": {
                "sha256": raw["sha256"],
                "size_bytes": raw["size_bytes"],
                "source_archive_sha256_verified": True,
            },
            "target_shard": shard,
            "target_schema_path": str(schema_path.relative_to(day_root)),
            "target_schema_sha256": schema_sha,
            "coverage": coverage,
        }
        manifest_path, manifest_sha = _write_content_addressed(
            day_root / "manifests", _canonical(day_manifest), ".target-manifest.json"
        )
        day_receipt = {
            "schema": transfer.TARGET_DAY_RECEIPT_SCHEMA,
            "owner_goal_revision": 21,
            "goal_contract_goal_revision": 22,
            "critic_semantic_owner_goal_revision": 21,
            "goal_contract_sha256": contract_sha,
            "manifest_path": str(manifest_path.relative_to(day_root)),
            "manifest_sha256": manifest_sha,
            "complete_action_overlay_sha256": overlay["sha256"],
            "raw_episode_zip_sha256": raw["sha256"],
            "target_shard_sha256": shard_sha,
            "target_schema_sha256": schema_sha,
            "target_shard_size_bytes": shard["size_bytes"],
            "target_row_count": 1,
            "coverage": coverage,
        }
        receipt_path, receipt_sha = _write_content_addressed(
            day_root / "receipts", _canonical(day_receipt), ".target-receipt.json"
        )
        days.append(
            {
                "utc_day": day,
                "split": split,
                "day_artifact_root": root_relative,
                "day_manifest_path": str(manifest_path.relative_to(source_root)),
                "day_manifest_sha256": manifest_sha,
                "day_receipt_path": str(receipt_path.relative_to(source_root)),
                "day_receipt_sha256": receipt_sha,
                "raw_episode_zip": {
                    "sha256": raw["sha256"],
                    "size_bytes": raw["size_bytes"],
                    "source_archive_sha256_verified": True,
                },
                "complete_action_overlay": {"sha256": overlay["sha256"]},
                "target_shard": shard,
                "coverage": coverage,
            }
        )
    set_coverage = {"counts": {"complete_action_programs": 20, "prize_h1_masked": 20}}
    target_set = {
        "schema": transfer.TARGET_SET_MANIFEST_SCHEMA,
        "owner_goal_revision": 21,
        "goal_contract_goal_revision": 22,
        "critic_semantic_owner_goal_revision": 21,
        "required_critic_authority": "revision_21_draw_safe_critic_actor_canary",
        "goal_contract": {
            "path": str(contract_path.relative_to(source_root)),
            "sha256": contract_sha,
            "size_bytes": contract_path.stat().st_size,
        },
        "base_pack_completion": {
            "path": str(base_path.relative_to(source_root)),
            "sha256": base_sha,
            "size_bytes": base_path.stat().st_size,
        },
        "complete_action_overlay_manifest": {
            "path": str(overlay_path.relative_to(source_root)),
            "sha256": overlay_manifest_sha,
            "size_bytes": overlay_path.stat().st_size,
        },
        "source_days": list(transfer.WINDOW_DAYS),
        "split_days": {
            split: [day for day in transfer.WINDOW_DAYS if transfer.SPLIT_BY_DAY[day] == split]
            for split in ("train", "validation", "evaluation")
        },
        "target_days": days,
        "all_20_raw_episode_zip_sha256s": raw_zip_identities,
        "all_20_target_shards": target_shards,
        "coverage": set_coverage,
        "episode_and_seat_group_split_disjoint": True,
        "information_boundary": {
            "hidden_information_simulator_search_rtp_mcts_or_unchosen_targets_allowed": False
        },
    }
    set_path, set_sha = _write_content_addressed(
        source_root / "manifests", _canonical(target_set), ".target-set-manifest.json"
    )
    set_receipt = {
        "schema": transfer.TARGET_SET_RECEIPT_SCHEMA,
        "owner_goal_revision": 21,
        "goal_contract_goal_revision": 22,
        "critic_semantic_owner_goal_revision": 21,
        "required_critic_authority": "revision_21_draw_safe_critic_actor_canary",
        "target_set_manifest_path": str(set_path.relative_to(source_root)),
        "target_set_manifest_sha256": set_sha,
        "goal_contract_sha256": contract_sha,
        "base_pack_completion_sha256": base_sha,
        "complete_action_overlay_manifest_sha256": overlay_manifest_sha,
        "day_count": 20,
        "coverage": set_coverage,
        "episode_and_seat_group_split_disjoint": True,
    }
    receipt_path, receipt_sha = _write_content_addressed(
        source_root / "receipts", _canonical(set_receipt), ".target-set-receipt.json"
    )
    return {
        "source_root": source_root,
        "destination_root": destination_root,
        "contract": contract_path,
        "contract_sha": contract_sha,
        "base_sha": base_sha,
        "overlay_sha": overlay_manifest_sha,
        "set_relative": str(set_path.relative_to(source_root)),
        "receipt_relative": str(receipt_path.relative_to(source_root)),
        "set_body": set_path.read_bytes(),
    }


def _plan(fixture: dict[str, object]) -> tuple[dict[str, object], str]:
    return transfer.build_target_transfer_plan(
        source=transfer.LocalSourceReader(),
        source_root=fixture["source_root"],
        destination_root=fixture["destination_root"],
        target_set_manifest_relative=str(fixture["set_relative"]),
        target_set_receipt_relative=str(fixture["receipt_relative"]),
        local_contract_path=fixture["contract"],
        expected_contract_sha256=str(fixture["contract_sha"]),
        expected_base_completion_sha256=str(fixture["base_sha"]),
        expected_overlay_manifest_sha256=str(fixture["overlay_sha"]),
        disk_floor_bytes=64,
    )


def test_dry_run_builds_exact_four_lane_target_only_plan(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan, plan_sha = _plan(fixture)

    assert transfer.validate_target_transfer_plan(plan) == plan_sha
    assert plan["parallel_lanes_exact"] == 4
    assert len(plan["lanes"]) == 4
    # target-set manifest + receipt + 3 portable binding docs + 20 day quartets.
    assert len(plan["entries"]) == 85
    assert {entry["lane_id"] for entry in plan["entries"]} == {0, 1, 2, 3}
    assert all("raw" not in entry["role"] for entry in plan["entries"])
    assert not Path(fixture["destination_root"]).exists()


def test_execute_keeps_target_set_byte_identical_and_emits_local_pointer(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan, plan_sha = _plan(fixture)
    result = transfer.execute_target_transfer_plan(
        plan,
        source=transfer.LocalSourceReader(),
        plan_sha256=plan_sha,
        free_bytes=lambda _path: 10**12,
    )
    destination = Path(fixture["destination_root"])
    copied_manifest = destination / str(fixture["set_relative"])
    assert copied_manifest.read_bytes() == fixture["set_body"]
    pointer = json.loads(Path(result["target_view_path"]).read_text(encoding="utf-8"))
    assert pointer["schema"] == transfer.TARGET_VIEW_SCHEMA
    assert pointer["canonical_target_set_manifest"]["relative_path"] == fixture["set_relative"]
    assert pointer["canonical_target_set_manifest"]["remains_byte_identical"] is True
    assert len(pointer["file_receipts"]) == 85
    assert pointer["raw_episode_zip_transferred"] is False
    assert not list(destination.rglob("*.zip"))
    assert Path(result["completion_path"]).is_file()


def test_partial_prefix_conflict_fails_closed_without_touching_final(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan, plan_sha = _plan(fixture)
    destination = Path(fixture["destination_root"])
    shard = next(item for item in plan["entries"] if item["role"] == "target_shard")
    partial = transfer._part_path(destination, plan_sha, shard)
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"not-the-source-prefix")

    with pytest.raises(transfer.TargetTransferError, match="prefix"):
        transfer.execute_target_transfer_plan(
            plan,
            source=transfer.LocalSourceReader(),
            plan_sha256=plan_sha,
            free_bytes=lambda _path: 10**12,
        )
    assert not (destination / shard["destination_relative"]).exists()


def test_existing_final_conflict_is_not_overwritten(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan, plan_sha = _plan(fixture)
    destination = Path(fixture["destination_root"])
    shard = next(item for item in plan["entries"] if item["role"] == "target_shard")
    final = destination / shard["destination_relative"]
    final.parent.mkdir(parents=True)
    final.write_bytes(b"wrong-final")

    with pytest.raises(transfer.TargetTransferError, match="conflicts"):
        transfer.execute_target_transfer_plan(
            plan,
            source=transfer.LocalSourceReader(),
            plan_sha256=plan_sha,
            free_bytes=lambda _path: 10**12,
        )
    assert final.read_bytes() == b"wrong-final"


def test_capacity_floor_blocks_before_destination_creation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan, plan_sha = _plan(fixture)
    needed = int(plan["total_size_bytes"])
    with pytest.raises(transfer.TargetTransferError, match="free space"):
        transfer.execute_target_transfer_plan(
            plan,
            source=transfer.LocalSourceReader(),
            plan_sha256=plan_sha,
            free_bytes=lambda _path: 64 + needed - 1,
        )
    assert not Path(fixture["destination_root"]).exists()
