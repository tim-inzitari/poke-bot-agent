"""Safety tests for the r23 portable Prize-plan-v2 Bert transfer."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import transfer_alakazam_prize_plan_v2_targets as transfer


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _sha(body: bytes | str) -> str:
    data = body.encode() if isinstance(body, str) else body
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _write(path: Path, body: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return _sha(body)


def _write_ca(directory: Path, body: bytes, suffix: str) -> tuple[Path, str]:
    digest = _sha(body)
    path = directory / f"sha256-{digest.removeprefix('sha256:')}{suffix}"
    _write(path, body)
    return path, digest


def _contract() -> dict[str, object]:
    target = {
        "row_schema": "poke_bot.alakazam_prize_plan_target_overlay/v2",
        "manifest_schema": "poke_bot.alakazam_prize_plan_target_set_manifest/v2",
        "day_manifest_schema": "poke_bot.alakazam_prize_plan_target_day_manifest/v2",
        "day_materialization_receipt_schema": "poke_bot.alakazam_prize_plan_target_day_materialization_receipt/v2",
        "target_set_materialization_receipt_schema": "poke_bot.alakazam_prize_plan_target_set_materialization_receipt/v2",
        "row_unit": "one_complete_recorded_chosen_action_program",
        "row_join_identity": [
            "utc_day",
            "source_archive_sha256",
            "source_member",
            "episode_id",
            "acting_seat",
            "env_step",
            "program_identity",
        ],
        "horizon_definition": "next_complete_same_seat_actions_with_all_intervening_opponent_activity_included",
        "segment_start": "public_pre_action_state_for_complete_same_seat_action_i_plus_k",
        "segment_end": "public_pre_action_state_for_complete_same_seat_action_i_plus_k_plus_1",
        "public_evidence_only": [
            "public_remaining_prize_counts",
            "exact_public_transition_and_event_evidence",
            "sealed_complete_action_and_public_observation_alignment",
        ],
        "hidden_prize_identity_or_other_hidden_information_allowed": False,
        "terminal_after_state_inference_allowed": False,
        "terminal_observed_z_is_direct_plan_reward_or_actor_term": False,
        "prize_race_potential": {
            "fit_manifest_schema": "poke_bot.alakazam_prize_plan_phi_fit_manifest/v2",
            "fit_receipt_schema": "poke_bot.alakazam_prize_plan_phi_fit_receipt/v2",
            "frozen_table_schema": "poke_bot.alakazam_prize_plan_phi_table/v2",
            "definition": "Phi(our_remaining,opponent_remaining)=2*P_iso(win|counts)-1",
            "fit_scope": "sealed_train_split_only",
            "fit_examples": "causally_available_public_count_pairs_with_observed_completed_trajectory_win_indicator",
            "smoothing_required": True,
            "monotone_constraints": {
                "Phi_when_our_remaining_count_falls": "must_not_decrease",
                "Phi_when_opponent_remaining_count_falls": "must_not_increase",
            },
            "fit_input_manifest_sha256_bound": True,
            "fit_configuration_sha256_bound": True,
            "frozen_table_sha256_bound": True,
            "validation_evaluation_or_runtime_refit_allowed": False,
        },
        "segment_shaping_reward": "rP_t=gamma*Phi(s_t_plus_1)-Phi(s_t)",
        "gamma": {
            "must_be_explicit_fixed_and_receipt_bound_before_materialization_or_actor_use": True,
            "may_silently_default": False,
            "fit_or_tune_on_validation_evaluation_or_runtime": False,
        },
        "horizon_return": "sum_{k=0}^{h-1}gamma^k*rP_{t+k}_over_exact_complete_same_seat_segments",
        "H3_return_requires_exact_segment_count": 3,
        "missing_ambiguous_nonmonotone_or_terminal_censored_evidence_behavior": "mask_target_and_interval_never_assign_zero",
        "m3_requires_all_h3_segments_available": True,
        "closest_valid_diagnostic_target_only_if_exact_target_is_impossible": True,
        "materialization_failure_behavior": "record_measured_schema_or_evidence_blocker_keep_legacy_active_never_fabricate_labels",
    }
    return {
        "goal_revision": 24,
        transfer.REQUIRED_AUTHORITY: {
            "owner_goal_revision": 23,
            "public_prize_plan_target": target,
            "sidecar_strategy": {
                "default_safe_implementation": "separately_versioned_prize_plan_v2_sidecar",
                "sidecar_schema": "poke_bot.alakazam_prize_plan_v2_sidecar/v1",
                "plan_horizons_to_train_and_receipt": [1, 3, 6, 12],
            },
            "actor_advantage": {
                "enabled_formula": "(z-V_existing(s))+0.025*m3*c3*(Q_plan_3(s,a)-V_plan_3(s))",
                "selected_nonzero_cumulative_prize_horizon": 3,
                "simultaneous_or_additive_H1_H3_H6_H12_actor_terms_allowed": False,
            },
        },
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "elmo-v2"
    destination = tmp_path / "bert-v2"
    contract_body = _canonical(_contract())
    contract_path, contract_sha = _write_ca(
        root / "bindings/goal_contract", contract_body, ".goal-contract.json"
    )
    fit_config = {
        "algorithm": "alternating_weighted_2d_isotonic_pava/v1",
        "smoothing_prior_strength": 8.0,
        "max_iterations": 100,
        "convergence_tolerance": 1e-10,
    }
    fit_config_sha = transfer.targets.canonical_sha256(fit_config)
    fit_input_doc = {
        "schema": "poke_bot.alakazam_prize_plan_phi_fit_inputs/v2",
        "fit_scope": "sealed_train_split_only",
    }
    fit_input_path, fit_input_sha = _write_ca(
        root / "bindings/phi_fit/objects",
        _canonical(fit_input_doc),
        ".phi-fit-inputs.json",
    )
    table_doc = {
        "schema": transfer.targets.PRIZE_PLAN_POTENTIAL_SCHEMA,
        "owner_goal_revision": 23,
        "fit_input_manifest_sha256": fit_input_sha,
        "fit_configuration": fit_config,
        "fit_configuration_sha256": fit_config_sha,
    }
    table_path, table_sha = _write_ca(
        root / "bindings/phi_fit/objects", _canonical(table_doc), ".phi-table.json"
    )
    phi_prefix = "bindings/phi_fit"
    phi_manifest = {
        "schema": transfer.targets.PRIZE_PLAN_POTENTIAL_MANIFEST_SCHEMA,
        "owner_goal_revision": 23,
        "goal_contract": {
            "sha256": contract_sha,
            "goal_revision": 24,
            "required_authority": transfer.REQUIRED_AUTHORITY,
            "semantic_owner_goal_revision": 23,
        },
        "fit_scope": "sealed_train_split_only",
        "fit_input_manifest": {
            "path": str(fit_input_path.relative_to(root / phi_prefix)),
            "sha256": fit_input_sha,
            "size_bytes": fit_input_path.stat().st_size,
        },
        "fit_configuration": fit_config,
        "fit_configuration_sha256": fit_config_sha,
        "frozen_phi_table": {
            "path": str(table_path.relative_to(root / phi_prefix)),
            "sha256": table_sha,
            "size_bytes": table_path.stat().st_size,
            "schema": transfer.targets.PRIZE_PLAN_POTENTIAL_SCHEMA,
        },
    }
    phi_manifest_path, phi_manifest_sha = _write_ca(
        root / "bindings/phi_fit/manifests",
        _canonical(phi_manifest),
        ".phi-fit-manifest.json",
    )
    phi_receipt = {
        "schema": transfer.targets.PRIZE_PLAN_POTENTIAL_RECEIPT_SCHEMA,
        "owner_goal_revision": 23,
        "goal_contract_sha256": contract_sha,
        "goal_contract_goal_revision": 24,
        "required_authority": transfer.REQUIRED_AUTHORITY,
        "phi_fit_manifest_sha256": phi_manifest_sha,
        "fit_input_manifest_sha256": fit_input_sha,
        "fit_configuration_sha256": fit_config_sha,
        "frozen_phi_table_sha256": table_sha,
        "fit_scope": "sealed_train_split_only",
        "validation_evaluation_or_runtime_refit": False,
        "terminal_z_used_only_for_train_phi_fit": True,
    }
    phi_receipt_path, phi_receipt_sha = _write_ca(
        root / "bindings/phi_fit/receipts",
        _canonical(phi_receipt),
        ".phi-fit-receipt.json",
    )
    overlay_shards: list[dict[str, object]] = []
    target_shards: list[dict[str, object]] = []
    raw_provenance: list[dict[str, object]] = []
    days: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []

    def add_inventory(path: Path, role: str, day: str | None = None) -> None:
        row: dict[str, object] = {
            "relative_path": str(path.relative_to(root)),
            "sha256": _sha(path.read_bytes()),
            "size_bytes": path.stat().st_size,
            "role": role,
        }
        if day is not None:
            row["utc_day"] = day
            row["split"] = transfer.SPLIT_BY_DAY[day]
        inventory.append(row)

    add_inventory(contract_path, "goal_contract")
    add_inventory(fit_input_path, "phi_fit_input_manifest")
    add_inventory(table_path, "phi_table")
    add_inventory(phi_manifest_path, "phi_fit_manifest")
    add_inventory(phi_receipt_path, "phi_fit_receipt")
    for index, day in enumerate(transfer.WINDOW_DAYS):
        split = transfer.SPLIT_BY_DAY[day]
        raw_sha = _sha(f"raw:{day}")
        overlay_sha = _sha(f"overlay:{day}")
        raw_size = 1000 + index
        overlay_size = 2000 + index
        raw_provenance.append(
            {"utc_day": day, "sha256": raw_sha, "size_bytes": raw_size}
        )
        overlay_shards.append(
            {
                "utc_day": day,
                "split": split,
                "sha256": overlay_sha,
                "size_bytes": overlay_size,
            }
        )
        day_root = root / "days" / day
        shard_path, shard_sha = _write_ca(
            day_root / "objects", (json.dumps({"day": day}) + "\n").encode(), ".prize-plan-targets.jsonl"
        )
        schema_path, schema_sha = _write_ca(
            day_root / "schemas",
            _canonical({"schema": transfer.targets.PRIZE_PLAN_TARGET_SCHEMA}),
            ".target-schema.json",
        )
        shard = {
            "path": str(shard_path.relative_to(day_root)),
            "sha256": shard_sha,
            "size_bytes": shard_path.stat().st_size,
            "row_count": 1,
        }
        schema = {
            "path": str(schema_path.relative_to(day_root)),
            "sha256": schema_sha,
        }
        coverage = {"counts": {"complete_action_programs": 1}}
        transform = {
            "formula": "model_target_value=raw_return_value/(1+gamma**h)",
            "gamma": 1.0,
            "data_dependent_train_fit": False,
            "clipping": False,
            "expected_model_target_range": [-1.0, 1.0],
            "actor_advantage_scaling": "separate_train_split_only_frozen_sidecar_or_actor_receipt_not_this_target_transform",
        }
        day_manifest = {
            "schema": transfer.targets.PRIZE_PLAN_DAY_MANIFEST_SCHEMA,
            "owner_goal_revision": 23,
            "goal_contract": {
                "sha256": contract_sha,
                "goal_revision": 24,
                "required_authority": transfer.REQUIRED_AUTHORITY,
                "semantic_owner_goal_revision": 23,
            },
            "utc_day": day,
            "split": split,
            "gamma": 1.0,
            "target_value_transform": transform,
            "phi_fit_manifest": {
                "sha256": phi_manifest_sha,
                "frozen_phi_table_sha256": table_sha,
                "fit_input_manifest_sha256": fit_input_sha,
                "fit_configuration_sha256": fit_config_sha,
            },
            "complete_action_overlay": {
                "sha256": overlay_sha,
                "size_bytes": overlay_size,
            },
            "raw_episode_zip": {
                "sha256": raw_sha,
                "size_bytes": raw_size,
                "source_archive_sha256_verified": True,
            },
            "target_shard": shard,
            "target_schema": schema,
            "coverage": coverage,
        }
        day_manifest_path, day_manifest_sha = _write_ca(
            day_root / "manifests",
            _canonical(day_manifest),
            ".prize-plan-target-day-manifest.json",
        )
        day_receipt = {
            "schema": transfer.targets.PRIZE_PLAN_DAY_RECEIPT_SCHEMA,
            "owner_goal_revision": 23,
            "goal_contract_sha256": contract_sha,
            "goal_contract_goal_revision": 24,
            "required_authority": transfer.REQUIRED_AUTHORITY,
            "day_manifest_sha256": day_manifest_sha,
            "phi_fit_manifest_sha256": phi_manifest_sha,
            "frozen_phi_table_sha256": table_sha,
            "gamma": 1.0,
            "target_value_transform": transform,
            "complete_action_overlay_sha256": overlay_sha,
            "raw_episode_zip_sha256": raw_sha,
            "target_shard_sha256": shard_sha,
            "target_shard_size_bytes": shard_path.stat().st_size,
            "target_row_count": 1,
            "coverage": coverage,
        }
        day_receipt_path, day_receipt_sha = _write_ca(
            day_root / "receipts",
            _canonical(day_receipt),
            ".prize-plan-target-day-receipt.json",
        )
        for path, role in (
            (shard_path, "target_shard"),
            (schema_path, "target_schema"),
            (day_manifest_path, "target_day_manifest"),
            (day_receipt_path, "target_day_receipt"),
        ):
            add_inventory(path, role, day)
        portable_shard = {
            "path": str(shard_path.relative_to(root)),
            "sha256": shard_sha,
            "size_bytes": shard_path.stat().st_size,
            "row_count": 1,
        }
        target_shards.append({"utc_day": day, "split": split, **portable_shard})
        days.append(
            {
                "utc_day": day,
                "split": split,
                "day_artifact_root": f"days/{day}",
                "day_manifest": {
                    "path": str(day_manifest_path.relative_to(root)),
                    "sha256": day_manifest_sha,
                },
                "day_receipt": {
                    "path": str(day_receipt_path.relative_to(root)),
                    "sha256": day_receipt_sha,
                },
                "target_schema": {
                    "path": str(schema_path.relative_to(root)),
                    "sha256": schema_sha,
                },
                "target_shard": portable_shard,
                "raw_episode_zip": {"sha256": raw_sha, "size_bytes": raw_size},
                "complete_action_overlay": {
                    "sha256": overlay_sha,
                    "size_bytes": overlay_size,
                },
                "coverage": coverage,
            }
        )
    overlay_doc = {
        "schema": "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1",
        "overlay_shards": overlay_shards,
    }
    overlay_path, overlay_sha = _write_ca(
        root / "bindings/complete_action_overlay_manifest",
        _canonical(overlay_doc),
        ".complete-action-overlay-manifest.json",
    )
    add_inventory(overlay_path, "complete_action_overlay_manifest")
    transform_doc = {
        "schema": transfer.TARGET_VALUE_TRANSFORM_SCHEMA,
        "owner_goal_revision": 23,
        "formula": "model_target_value=raw_return_value/(1+gamma**h)",
        "gamma": 1.0,
        "horizons": [1, 3, 6, 12],
        "expected_model_target_range": [-1.0, 1.0],
        "clipping": False,
        "data_dependent_train_fit": False,
        "actor_advantage_scaling": "separate_train_split_only_frozen_sidecar_or_actor_receipt_not_this_target_transform",
        "source_target_shards": target_shards,
    }
    transform_path, transform_sha = _write_ca(
        root / "bindings", _canonical(transform_doc), ".target-value-transform.json"
    )
    add_inventory(transform_path, "target_value_transform")
    inventory.sort(key=lambda row: str(row["relative_path"]))
    coverage = {"counts": {"complete_action_programs": 20}}
    target_set = {
        "schema": transfer.targets.PRIZE_PLAN_TARGET_SET_MANIFEST_SCHEMA,
        "owner_goal_revision": 23,
        "goal_contract_goal_revision": 24,
        "required_authority": transfer.REQUIRED_AUTHORITY,
        "goal_contract": {
            "path": str(contract_path.relative_to(root)),
            "sha256": contract_sha,
            "goal_revision": 24,
            "required_authority": transfer.REQUIRED_AUTHORITY,
            "semantic_owner_goal_revision": 23,
        },
        "complete_action_overlay_manifest": {
            "path": str(overlay_path.relative_to(root)),
            "sha256": overlay_sha,
            "schema": overlay_doc["schema"],
        },
        "phi_fit": {
            "portable_root": phi_prefix,
            "fit_manifest": {
                "path": str(phi_manifest_path.relative_to(root)),
                "sha256": phi_manifest_sha,
            },
            "fit_receipt": {
                "path": str(phi_receipt_path.relative_to(root)),
                "sha256": phi_receipt_sha,
            },
            "frozen_phi_table": {
                "path": str(table_path.relative_to(root)),
                "sha256": table_sha,
                "schema": transfer.targets.PRIZE_PLAN_POTENTIAL_SCHEMA,
            },
            "fit_input_manifest_sha256": fit_input_sha,
            "fit_configuration_sha256": fit_config_sha,
            "fit_scope": "sealed_train_split_only",
        },
        "target_value_transform": {
            "path": str(transform_path.relative_to(root)),
            "sha256": transform_sha,
            "schema": transfer.TARGET_VALUE_TRANSFORM_SCHEMA,
        },
        "source_days": list(transfer.WINDOW_DAYS),
        "split_days": {
            split: [day for day in transfer.WINDOW_DAYS if transfer.SPLIT_BY_DAY[day] == split]
            for split in ("train", "validation", "evaluation")
        },
        "target_days": days,
        "all_20_raw_episode_zip_sha256s": raw_provenance,
        "all_20_complete_action_overlay_sha256s": overlay_shards,
        "all_20_target_shards": target_shards,
        "coverage": coverage,
        "whole_day_episode_and_group_split_disjoint": True,
        "portable_objects": inventory,
        "information_boundary": {
            "raw_zip_or_feature_or_complete_action_overlay_payload_copied": False,
            "hidden_information_simulator_search_rtp_mcts_or_unchosen_targets_allowed": False,
            "terminal_z_is_direct_plan_target_or_actor_term": False,
        },
        "publication": {
            "create_only": True,
            "atomic_root_no_replace": True,
            "portable_relative_paths_only": True,
        },
    }
    set_path, set_sha = _write_ca(
        root / "manifests", _canonical(target_set), ".prize-plan-target-set-manifest.json"
    )
    set_receipt = {
        "schema": transfer.targets.PRIZE_PLAN_TARGET_SET_RECEIPT_SCHEMA,
        "owner_goal_revision": 23,
        "goal_contract_sha256": contract_sha,
        "goal_contract_goal_revision": 24,
        "required_authority": transfer.REQUIRED_AUTHORITY,
        "target_set_manifest_path": str(set_path.relative_to(root)),
        "target_set_manifest_sha256": set_sha,
        "phi_fit_manifest_sha256": phi_manifest_sha,
        "phi_fit_receipt_sha256": phi_receipt_sha,
        "frozen_phi_table_sha256": table_sha,
        "target_value_transform_sha256": transform_sha,
        "complete_action_overlay_manifest_sha256": overlay_sha,
        "day_count": 20,
        "coverage": coverage,
        "whole_day_episode_and_group_split_disjoint": True,
        "portable_object_count": len(inventory),
        "raw_zip_or_feature_or_complete_action_overlay_payload_copied": False,
        "terminal_z_used_as_direct_plan_target_or_actor_term": False,
        "atomic_root_no_replace": True,
        "sealed_at_unix_seconds": 1.0,
    }
    receipt_path, receipt_sha = _write_ca(
        root / "receipts", _canonical(set_receipt), ".prize-plan-target-set-receipt.json"
    )
    local_contract = tmp_path / "current-contract.json"
    local_contract.write_bytes(contract_body)
    return {
        "root": root,
        "destination": destination,
        "contract": local_contract,
        "contract_sha": contract_sha,
        "manifest_relative": str(set_path.relative_to(root)),
        "manifest_sha": set_sha,
        "receipt_relative": str(receipt_path.relative_to(root)),
        "receipt_sha": receipt_sha,
        "manifest_body": set_path.read_bytes(),
    }


def _plan(fixture: dict[str, object]) -> tuple[dict[str, object], str]:
    return transfer.build_prize_plan_v2_transfer_plan(
        source=transfer.LocalSourceReader(),
        source_root=fixture["root"],
        destination_root=fixture["destination"],
        target_set_manifest_relative=str(fixture["manifest_relative"]),
        expected_target_set_manifest_sha256=str(fixture["manifest_sha"]),
        target_set_receipt_relative=str(fixture["receipt_relative"]),
        expected_target_set_receipt_sha256=str(fixture["receipt_sha"]),
        local_contract_path=fixture["contract"],
        expected_contract_sha256=str(fixture["contract_sha"]),
        disk_floor_bytes=transfer.DEFAULT_DISK_FLOOR_BYTES,
        metadata_reserve_bytes=transfer.DEFAULT_METADATA_RESERVE_BYTES,
        test_allow_non_elmo_source=True,
    )


def test_dry_run_is_exact_four_lane_target_only(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan, plan_sha = _plan(fixture)
    assert transfer.validate_prize_plan_v2_transfer_plan(plan) == plan_sha
    assert plan["parallel_lanes_exact"] == 4
    assert {lane["lane_id"] for lane in plan["lanes"]} == {0, 1, 2, 3}
    assert all(lane["entry_count"] > 0 for lane in plan["lanes"])
    assert all(entry["role"] not in {"raw_zip", "feature_pack"} for entry in plan["entries"])
    assert not any(entry["destination_relative"].endswith(".zip") for entry in plan["entries"])
    assert plan["bert_side_c3_or_advantage_scaling_produced_or_transferred"] is False
    assert not Path(fixture["destination"]).exists()


def test_execute_copies_byte_identical_set_and_emits_bound_view(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan, plan_sha = _plan(fixture)
    result = transfer.execute_prize_plan_v2_transfer_plan(
        plan,
        source=transfer.LocalSourceReader(),
        plan_sha256=plan_sha,
        free_bytes=lambda _path: 10**12,
        test_allow_non_elmo_source=True,
    )
    destination = Path(fixture["destination"])
    assert (destination / str(fixture["manifest_relative"])).read_bytes() == fixture["manifest_body"]
    view = json.loads(Path(result["target_view_path"]).read_text())
    completion = json.loads(Path(result["completion_path"]).read_text())
    assert view["schema"] == transfer.TARGET_VIEW_SCHEMA
    assert view["canonical_target_set_manifest"]["remains_byte_identical"] is True
    assert view["source_identity_graph"]["goal_contract_sha256"] == fixture["contract_sha"]
    assert view["source_identity_graph"]["phi_fit"]["frozen_phi_table_sha256"]
    assert view["source_identity_graph"]["target_value_transform"]["gamma"] == 1.0
    assert view["bert_side_c3_or_advantage_scaling_produced_or_transferred"] is False
    assert view["private_partials_not_training_eligible"] is True
    assert len(view["file_receipts"]) == len(plan["entries"])
    assert completion["parallel_lanes_exact"] == 4
    assert completion["private_partials_not_training_eligible"] is True
    assert not list(destination.rglob("*.zip"))

    # A clean second execution must reuse immutable finals and byte-identical
    # receipts rather than conflict on a transient copied/skipped disposition.
    repeated = transfer.execute_prize_plan_v2_transfer_plan(
        plan,
        source=transfer.LocalSourceReader(),
        plan_sha256=plan_sha,
        free_bytes=lambda _path: 10**12,
        test_allow_non_elmo_source=True,
    )
    assert repeated["target_view_sha256"] == result["target_view_sha256"]
    assert repeated["completion_sha256"] == result["completion_sha256"]


def test_corrupt_private_prefix_fails_without_final(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan, plan_sha = _plan(fixture)
    entry = next(item for item in plan["entries"] if item["role"] == "target_shard")
    partial = transfer._part_path(Path(fixture["destination"]), plan_sha, entry)
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"wrong-prefix")
    with pytest.raises(transfer.TargetTransferError, match="prefix"):
        transfer.execute_prize_plan_v2_transfer_plan(
            plan,
            source=transfer.LocalSourceReader(),
            plan_sha256=plan_sha,
            free_bytes=lambda _path: 10**12,
            test_allow_non_elmo_source=True,
        )
    assert not (Path(fixture["destination"]) / entry["destination_relative"]).exists()


def test_existing_final_conflict_is_preserved(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan, plan_sha = _plan(fixture)
    entry = next(item for item in plan["entries"] if item["role"] == "target_shard")
    final = Path(fixture["destination"]) / entry["destination_relative"]
    final.parent.mkdir(parents=True)
    final.write_bytes(b"conflict")
    with pytest.raises(transfer.TargetTransferError, match="conflicts"):
        transfer.execute_prize_plan_v2_transfer_plan(
            plan,
            source=transfer.LocalSourceReader(),
            plan_sha256=plan_sha,
            free_bytes=lambda _path: 10**12,
            test_allow_non_elmo_source=True,
        )
    assert final.read_bytes() == b"conflict"


def test_disk_floor_and_metadata_reserve_block_before_write(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    plan, plan_sha = _plan(fixture)
    required = int(plan["bert_disk_free_floor_bytes"]) + int(plan["total_size_bytes"]) + int(plan["metadata_reserve_bytes"])
    with pytest.raises(transfer.TargetTransferError, match="free space"):
        transfer.execute_prize_plan_v2_transfer_plan(
            plan,
            source=transfer.LocalSourceReader(),
            plan_sha256=plan_sha,
            free_bytes=lambda _path: required - 1,
            test_allow_non_elmo_source=True,
        )
    assert not Path(fixture["destination"]).exists()


def test_expected_contract_sha_and_embedded_r23_are_required(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(transfer.TargetTransferError, match="contract SHA"):
        transfer.build_prize_plan_v2_transfer_plan(
            source=transfer.LocalSourceReader(),
            source_root=fixture["root"],
            destination_root=fixture["destination"],
            target_set_manifest_relative=str(fixture["manifest_relative"]),
            expected_target_set_manifest_sha256=str(fixture["manifest_sha"]),
            target_set_receipt_relative=str(fixture["receipt_relative"]),
            expected_target_set_receipt_sha256=str(fixture["receipt_sha"]),
            local_contract_path=fixture["contract"],
            expected_contract_sha256=_sha("different"),
            test_allow_non_elmo_source=True,
        )
    broken = copy.deepcopy(_contract())
    authority = broken[transfer.REQUIRED_AUTHORITY]
    assert isinstance(authority, dict)
    actor = authority["actor_advantage"]
    assert isinstance(actor, dict)
    actor["enabled_formula"] = "wrong"
    broken_contract = tmp_path / "broken-contract.json"
    broken_sha = _write(broken_contract, _canonical(broken))
    with pytest.raises(transfer.TargetTransferError, match="semantics drifted"):
        transfer._load_current_contract(broken_contract, expected_sha256=broken_sha)
