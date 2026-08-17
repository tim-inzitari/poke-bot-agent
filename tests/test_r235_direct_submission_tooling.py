from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tarfile
import time
from pathlib import Path
from typing import Any

import pytest

from scripts import download_r235_kaggle_evidence as evidence
from scripts import kaggle_cli_guard as guard
from scripts import r235_direct_submission_tooling as support


def _sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_archive(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, body in sorted(files.items()):
            member = tarfile.TarInfo("./" + name)
            member.size = len(body)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(body))


def _r240_manifest_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    return dict(support.R246_LOCAL_SCHEDULER), dict(support.R240_MANIFEST_SCHEDULER)


def _inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    lane_count: int = 2,
    phase1_vcpus: int = 2,
    direct_entrypoint: bytes | None = None,
    publish_binding: bool = True,
) -> dict[str, Any]:
    native_bodies = {
        "cg/libcg.so": b"canonical-r236-linux-x86_64",
        "cg/libcg-arm64.so": b"canonical-r236-linux-aarch64",
        "cg/libcg.dylib": b"canonical-r236-macos-arm64",
        "cg/cg.dll": b"canonical-r236-windows-x86_64",
    }
    platform_by_path = {
        "cg/libcg.so": "linux_x86_64",
        "cg/libcg-arm64.so": "linux_aarch64",
        "cg/libcg.dylib": "macos_arm64",
        "cg/cg.dll": "windows_x86_64",
    }
    wheel_member_by_path = {
        "cg/libcg.so": "kaggle_environments/envs/cabt/cg/libcg.so",
        "cg/libcg-arm64.so": "kaggle_environments/envs/cabt/cg/libcg-arm64.so",
        "cg/libcg.dylib": "kaggle_environments/envs/cabt/cg/libcg.dylib",
        "cg/cg.dll": "kaggle_environments/envs/cabt/cg/cg.dll",
    }
    native_libraries = {
        platform_by_path[path]: {
            "wheel_member": wheel_member_by_path[path],
            "package_relative_path": path,
            "sha256": _sha(body),
            "size_bytes": len(body),
            "format": platform_by_path[path],
        }
        for path, body in native_bodies.items()
    }
    model = b"r195-model"
    tree = b"r195-tree"
    r236 = tmp_path / "r236.json"
    _write_json(
        r236,
        {
            "schema": support.R236_SCHEMA,
            "owner_decision_revision": 236,
            "upstream_provenance": {
                "package_version": "1.32.6",
                "wheel_filename": "kaggle_environments-1.32.6-py3-none-any.whl",
                "wheel_sha256": _sha(b"synthetic-wheel"),
            },
            "canonical_native_libraries": native_libraries,
            "required_native_exports": [
                "AgentStart", "BattleStart", "SearchBegin", "SearchStep", "SearchRelease", "SearchEnd"
            ],
        },
    )
    local_r240, manifest_r240 = _r240_manifest_contract()
    continuation = dict(support.MANIFEST_DETERMINISTIC_CONTINUATION)
    replacement_r240 = dict(support.R246_REPLACEMENT_SCHEDULER)
    replacement_continuation = dict(support.R242_REPLACEMENT_DETERMINISTIC_CONTINUATION)
    r244_projection = {
        "per_lane_handle_identities": ["handle-a", "handle-b"],
        "per_lane_search_id_chains": [[0, 7], [0, 8]],
        "per_lane_first_search_ids": [0, 0],
        "handle_scoped_first_search_id_composite_states": [
            {"lane_id": 0, "handle_identity": "handle-a", "first_search_id": 0},
            {"lane_id": 1, "handle_identity": "handle-b", "first_search_id": 0},
        ],
        "distinct_handle_identity_count": 2,
        "distinct_handle_scoped_first_search_id_composite_state_count": 2,
        "globally_distinct_raw_search_id_integers_required": False,
    }
    r246_projection = {
        "owner_proven_deterministic_terminal_win_this_turn_revision": 246,
        "proven_deterministic_terminal_win_this_turn_regression_passed": True,
        "stop_reason": support.R246_TERMINAL_WIN_STOP_REASON,
        "two_lane_topology_initialized_before_terminal_win_override": True,
        "requested_simulator_lane_count": 2,
        "active_simulator_lane_count": 2,
        "arena_count": 2,
        "unique_handle_count": 2,
        "search_begin_calls": 2,
        "search_release_calls": 2,
        "search_end_calls": 2,
        "completed_root_backup_count": 1,
        "terminal_win_proof_count": 1,
        "proven_deterministic_terminal_win_this_turn_stop_count": 1,
        "terminal_win_proof_backed_up_into_shared_root_tree": True,
        "terminal_leaf_returned_by_exact_stock_simulator": True,
        "parent_validated_current_root_observation_legal_fingerprint_and_actor": True,
        support.R246_CLEANUP_COMPLETE_FIELD: True,
        "outstanding_virtual_loss": 0,
        "two_independent_lane_proofs_required": False,
        "exhaustive_legal_action_scan_required": False,
        "standard_adaptive_min_backups_leader_observations_and_both_lanes_progressed_required_after_valid_proof": False,
        "terminal_win_proof": {
            "proof_kind": support.R246_TERMINAL_WIN_PROOF_KIND,
            "root_observation_fingerprint": "root-observation-fingerprint",
            "root_legal_order_fingerprint": "root-legal-order-fingerprint",
            "root_actor_seat": 0,
            "root_action": [0],
            "selected_action": [0],
            "terminal_result": "win",
            "terminal_winner_seat": 0,
            "terminal_leaf_reached": True,
            "proof_path_action_count": 1,
            "discovering_lane_id": 1,
            "path_actor_seats": [0],
            "path_no_chance_boundary": True,
            "path_no_actor_change_boundary": True,
            "path_no_opponent_boundary_crossing": True,
            "path_no_unresolved_randomness": True,
            "proof_is_deterministic": True,
        },
    }
    r238_receipt_contract = {
        "normal_mcts_decision_required_fields": list(support.R244_TWO_LANE_RECEIPT_REQUIRED_FIELDS),
        "normal_mcts_decision_exact_counts": {
            "requested_simulator_lane_count": 2,
            "active_simulator_lane_count": 2,
            "arena_count": 2,
            "unique_handle_count": 2,
            "distinct_handle_identity_count": 2,
            "distinct_handle_scoped_first_search_id_composite_state_count": 2,
            "search_begin_calls": 2,
            "search_end_calls": 2,
        },
        "normal_mcts_decision_minimum_counts": {"search_release_calls": 2},
        "normal_mcts_decision_per_lane_vectors_exact_length": {
            "per_lane_depth": 2,
            "per_lane_search_id_chains": 2,
            "per_lane_handle_identities": 2,
            "per_lane_first_search_ids": 2,
            "handle_scoped_first_search_id_composite_states": 2,
        },
        "normal_mcts_decision_lane_ids": [0, 1],
        "normal_mcts_decision_microbatch_size_range": [1, 2],
        "normal_mcts_decision_max_simulator_calls_in_flight_range": [1, 2],
        "search_id_identity_contract": dict(support.R244_SEARCH_ID_IDENTITY_CONTRACT),
        "single_lane_serial_fallback_or_eight_lane_receipt_authority_allowed": False,
    }
    replacement_topology = {
        "exactly_two_simulator_search_lanes_required": True,
        "required_simulator_search_lane_count": 2,
        "required_active_simulator_search_lane_count": 2,
        "required_internal_agent_start_simulator_search_arena_count": 2,
        "required_unique_raw_handle_count": 2,
        "required_search_begin_call_count": 2,
        "required_distinct_per_lane_handle_identity_count": 2,
        "per_lane_handle_scoped_search_id_chains_required": True,
        "search_id_numeric_namespace_is_per_distinct_agent_start_handle": True,
        "globally_distinct_raw_search_id_integers_required": False,
        "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
        "required_distinct_handle_identity_first_search_id_composite_state_count": 2,
        "first_search_id_identity_composite": "(handle_identity, first_search_id)",
        "maximum_frontier_leaves_per_frozen_evaluator_batch": 2,
        "one_shared_logical_mcts_tree_required": True,
        "one_lane_serial_fallback_eight_lane_topology_or_partial_lane_mcts_authority_allowed": False,
    }
    r225 = tmp_path / "r225.json"
    _write_json(
        r225,
        {
            "schema": support.R225_SCHEMA,
            "owner_decision_revision": support.R225_OWNER_DECISION_REVISION,
            "owner_kaggle_replacement_diagnostic_revision": 235,
            "owner_canonical_libcg_revision": 236,
            "owner_phase1_submission_resources_and_two_lane_revision": 238,
            "owner_hybrid_confidence_bounded_mcts_revision": 240,
            "owner_high_confidence_frozen_direct_threshold_revision": 242,
            "owner_handle_scoped_search_id_revision": 244,
            "owner_proven_deterministic_terminal_win_this_turn_revision": 246,
            "relationship_to_existing_work": {
                "r242_supersedes_only_the_r240_high_confidence_frozen_direct_threshold_for_the_new_r235_replacement_package": True,
                "r242_uses_inclusive_0_80_at_every_selected_factorized_stage_and_makes_the_historical_0_90_draft_preflight_ineligible": True,
                "r242_does_not_modify_r234_r236_r238_r235_continuation_or_r229_bo1000": True,
                "r244_supersedes_only_global_raw_search_id_integer_distinctness_for_official_libcg_handle_scoped_search_states_in_r225_and_r229": True,
                "r244_preserves_r242_kaggle_hybrid_containment_and_r239_bo1000_lifecycle_boundaries": True,
                "r246_supersedes_only_ambiguous_r235_mcts_root_selection_after_a_valid_deterministic_terminal_win_this_turn_proof": True,
                "r246_does_not_change_r242_high_confidence_direct_before_child_or_any_r229_bo1000_lifecycle": True,
            },
            "replacement_kaggle_diagnostic": {
                "revision": 235,
                "phase1_two_lane_resource_revision": 238,
                "hybrid_confidence_bounded_mcts_revision": 240,
                "handle_scoped_search_id_correction_revision": 244,
                "proven_deterministic_terminal_win_this_turn_revision": 246,
                "replacement_submission_count_limit": 1,
                "replacement_submission_consumed": False,
                "competition": support.COMPETITION,
                "submission_message_required_literal": support.LABEL,
                "submission_message_must_be_unique_and_exact": True,
                "queue_or_batch_submission_allowed": False,
                "automatic_retry_allowed": False,
                "automatic_copy_or_resubmission_allowed": False,
                "second_upload_allowed": False,
                "complete_ordered_legal_action_ceiling": support.ACTION_CAP,
                "all_required_local_gates_and_immutable_binding_must_pass_before_direct_api_upload": True,
                "kaggle_api_call_permitted_now_before_gates": False,
                "kaggle_upload_permitted_now_before_gates": False,
                "required_local_gate_receipts": [
                    receipt_name
                    for receipt_name in support.R235_BINDING_GATE_RECEIPT_NAMES.values()
                    if receipt_name != "go_first_receipt"
                ],
                "deterministic_continuation": replacement_continuation,
                "r240_hybrid_scheduler": replacement_r240,
                "phase1_simulator_search": replacement_topology,
            },
            "phase1_submission_environment": {
                "owner_decision_revision": 238,
                **support.PHASE1_RESOURCES,
                "vcpus": phase1_vcpus,
                "resource_probe_and_archive_size_receipt_required": True,
                "resource_mismatch_or_archive_over_limit_behavior": "hard_fail_closed_and_do_not_upload",
                "gpu_or_os_python_environment_is_not_inferred_from_the_reported_submission_resource_values": True,
            },
            "local_preflight": {
                "required_simulator_search_lane_count": 2,
                "required_internal_agent_start_simulator_search_arena_count_per_child": 2,
                "required_search_begin_call_count_per_ambiguous_mcts_decision": 2,
                "required_distinct_internal_agent_start_handle_identity_count": 2,
                "required_distinct_handle_identity_first_search_id_composite_state_count": 2,
                "search_id_numeric_namespace_is_per_distinct_agent_start_handle": True,
                "globally_distinct_raw_search_id_integers_required": False,
                "first_raw_search_id_may_be_zero_on_each_distinct_handle": True,
                "per_lane_handle_scoped_search_id_chains_required": True,
                "r240_hybrid_scheduler": local_r240,
                "r238_two_lane_receipt_contract": r238_receipt_contract,
                "full_gameplay_shared_tree_contract": {
                    "retain_exact_lane_handle_identity_search_id_tuple_across_repeated_depth_waves": True,
                    "per_lane_handle_scoped_search_id_chains_required": True,
                    "two_distinct_handle_identity_first_search_id_composite_states_required": True,
                    "global_raw_search_id_integer_distinctness_required": False,
                    "r246_proven_deterministic_terminal_win_this_turn_is_an_in_search_early_stop_not_a_r242_direct_bypass": True,
                    "r246_valid_terminal_win_proof_may_bypass_only_the_normal_adaptive_stop_thresholds_after_two_lane_initialization": True,
                    "r246_valid_terminal_win_proof_requires_at_least_one_exact_terminal_backup_but_not_the_normal_eight_backup_leader_or_both_lane_progress_threshold": True,
                },
                "deterministic_continuation": continuation,
            },
            "complete_ordered_action_space_contract": {
                "complete_ordered_legal_action_ceiling": support.ACTION_CAP,
                "ceiling_applies_at_root_and_every_private_leaf": True,
                "sampling_pruning_or_reinterpretation_of_legal_choices_allowed": False,
                "root_over_cap_behavior": "hard_fail_nonzero",
                "private_leaf_over_cap_behavior": (
                    "contained_degraded_parent_direct_fallback_only_after_validated_"
                    "precomputed_root_direct_action_and_exact_child_reap"
                ),
            },
            "exact_frozen_base": {
                "r195_bundle_sha256": _sha(b"r195-bundle"),
                "r195_checkpoint_sha256": _sha(model),
                "r195_matchup_tree_sha256": _sha(tree),
                "stock_libcg_sha256": _sha(native_bodies["cg/libcg.so"]),
                "stock_libcg_size_bytes": len(native_bodies["cg/libcg.so"]),
            },
            "authority": {
                field: False
                for field in (
                    "kaggle_api_call_permitted_now_before_preconditions",
                    "kaggle_upload_permitted_now_before_preconditions",
                    "kaggle_api_call_permitted_now",
                    "kaggle_upload_permitted_now",
                    "kaggle_queue_submission_permitted",
                    "automatic_kaggle_submission_allowed",
                    "kaggle_retry_or_copy_permitted",
                    "second_kaggle_upload_permitted",
                    "training_or_gradient_updates_authorized",
                    "selector_change_authorized",
                    "promotion_authorized",
                )
            },
        },
    )
    monkeypatch.setattr(support, "CANONICAL_R225_PATH", r225)
    monkeypatch.setattr(support, "CANONICAL_R236_PATH", r236)
    monkeypatch.setattr(support, "CANONICAL_R225_R246_SHA256", support.sha256_file(r225))
    monkeypatch.setattr(support, "CANONICAL_R236_SHA256", support.sha256_file(r236))
    main = b"def agent(observation):\n    return [0]\n"
    direct = direct_entrypoint or '''import json
from pathlib import Path
def _turn_order_choice(obs):
    select = obs.get("select") if isinstance(obs, dict) else None
    if not isinstance(select, dict) or select.get("context") not in (41, "IS_FIRST", "IsFirst"):
        return None
    for index, option in enumerate(select.get("option") or []):
        if option.get("type") in (1, "Yes"):
            return [index]
    return []
def agent(obs):
    choice = _turn_order_choice(obs)
    return choice if choice is not None else [0]
'''.encode("utf-8")
    manifest_member = "r238-manifest.json"
    manifest = {
        "schema": support.R238_MANIFEST_SCHEMA,
        "role": support.R238_MANIFEST_ROLE,
        "required_label": support.LABEL,
        "complete_action_cap": support.ACTION_CAP,
        "lane_count": lane_count,
        "r225_typed_contract": {
            "path": "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json",
            "schema": support.R225_SCHEMA,
            "sha256": support.sha256_file(r225),
        },
        "canonical_libcg_contract": {
            "schema": support.R236_SCHEMA,
            "typed_source": "state/canonical-libcg-r236.json",
            "owner_decision_revision": 236,
            "upstream_provenance": {
                "package_version": "1.32.6",
                "wheel_filename": "kaggle_environments-1.32.6-py3-none-any.whl",
                "wheel_sha256": _sha(b"synthetic-wheel"),
            },
            "canonical_native_libraries": native_libraries,
            "required_native_exports": [
                "AgentStart", "BattleStart", "SearchBegin", "SearchStep", "SearchRelease", "SearchEnd"
            ],
            "scope": {
                "r235_kaggle_replacement_must_overlay_and_bind_the_exact_linux_x86_64_binary": True,
                "all_four_platform_native_members_must_be_bound": True,
                "mixed_old_and_new_library_sets_allowed": False,
                "frozen_r195_python_cg_wrapper_retained_while_only_four_canonical_native_members_are_overlaid": True,
            },
            "r225_package_preflight": {
                "frozen_r195_python_cg_wrapper_retained_while_only_four_canonical_native_members_are_overlaid": True,
                "all_four_canonical_native_members_checksum_and_size_verified": True,
                "old_or_mixed_native_members_rejected": True,
                "required_native_exports": [
                    "AgentStart", "BattleStart", "SearchBegin", "SearchStep", "SearchRelease", "SearchEnd"
                ],
            },
        },
        "canonical_native_members": {
            payload["package_relative_path"]: {
                "platform": platform,
                "wheel_member": payload["wheel_member"],
                "sha256": payload["sha256"],
                "size_bytes": payload["size_bytes"],
            }
            for platform, payload in native_libraries.items()
        },
        "canonical_native_member_sha256": {
            payload["package_relative_path"]: payload["sha256"]
            for payload in native_libraries.values()
        },
        "exact_frozen_base": {
            "r195_bundle_sha256": _sha(b"r195-bundle"),
            "r195_checkpoint_sha256": _sha(model),
            "r195_matchup_tree_sha256": _sha(tree),
            "stock_libcg_sha256": _sha(native_bodies["cg/libcg.so"]),
            "stock_libcg_size_bytes": len(native_bodies["cg/libcg.so"]),
        },
        "phase1_kaggle_resource_bounds": {
            **support.PHASE1_MANIFEST_RESOURCE_BOUNDS,
            "vcpus": phase1_vcpus,
        },
        "required_search_lifecycle_counts": {
            "search_begin_calls": 2,
            "search_end_calls": 2,
            "search_release_calls": 2,
        },
        "r240_hybrid_scheduler": manifest_r240,
        "deterministic_continuation": dict(support.MANIFEST_DETERMINISTIC_CONTINUATION),
        "r240_required_preflight_receipts": list(support.R240_REQUIRED_REGRESSION_RECEIPTS),
        "owner_proven_deterministic_terminal_win_this_turn_revision": (
            support.R246_PROVEN_TERMINAL_WIN_REVISION
        ),
        "r246_proven_deterministic_terminal_win_this_turn": (
            support.R246_LOCAL_TERMINAL_WIN_CONTRACT
        ),
        "r246_required_preflight_receipts": list(
            support.R246_REQUIRED_REGRESSION_RECEIPTS
        ),
        "broker_contract": {
            "complete_action_cap": support.ACTION_CAP,
            "degraded_fallback_marker": "R234_KAGGLE_NATIVE_CONTAINMENT_DEGRADED",
            "search_seconds": 2.0,
            "action_timeout_seconds": 4.0,
        },
        "entrypoint_sha256": _sha(main),
        "direct_entrypoint": {"path": "r195_direct_main.py", "sha256": _sha(direct)},
    }
    manifest_path = tmp_path / manifest_member
    _write_json(manifest_path, manifest)
    archive = tmp_path / "r238-two-lane.tar.gz"
    _write_archive(
        archive,
        {
            "main.py": main,
            "r195_direct_main.py": direct,
            "turn_order_profile.json": json.dumps(
                {
                    "schema": "poke_bot.submission_turn_order_profile/v1",
                    "turn_order_preference": "first_if_allowed",
                }
            ).encode("utf-8"),
            "model.pt": model,
            "matchup_tree.json": tree,
            **native_bodies,
            "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json": r225.read_bytes(),
            manifest_member: manifest_path.read_bytes(),
        },
    )
    members = support._archive_members(archive)
    computed = {
        "schema": "poke_bot.r235_r236_archive_member_manifest/v1",
        "archive_sha256": support.sha256_file(archive),
        "members": members,
    }
    gate_names = support.R235_BINDING_GATE_NAMES
    binding = {
        "schema": support.R235_BINDING_SCHEMA,
        "status": "local_gates_bound_not_submitted",
        "immutable": True,
        "write_once": True,
        "owner_decision_revision": support.R225_OWNER_DECISION_REVISION,
        "candidate_package": {
            "archive_sha256": support.sha256_file(archive),
            "archive_size_bytes": archive.stat().st_size,
            "member_manifest_member": manifest_member,
            "member_manifest_sha256": support.sha256_file(manifest_path),
            "manifest_schema": support.R238_MANIFEST_SCHEMA,
            "manifest_role": support.R238_MANIFEST_ROLE,
            "entrypoint_member": "main.py",
            "entrypoint_sha256": _sha(main),
            "computed_archive_member_manifest_sha256": support._sha256_bytes(
                support._canonical_json(computed)
            ),
            "computed_archive_member_manifest": computed,
        },
        "typed_contracts": {
            "r225_historical_r246_typed_contract": {
                "canonical_relative_path": "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json",
                "resolved_path": str(r225.resolve()),
                "schema": support.R225_SCHEMA,
                "sha256": support.sha256_file(r225),
                "expected_sha256": support.sha256_file(r225),
                "archive_member_bound": True,
            },
            "r236_canonical_libcg_typed_contract": {
                "canonical_relative_path": "state/canonical-libcg-r236.json",
                "resolved_path": str(r236.resolve()),
                "schema": support.R236_SCHEMA,
                "sha256": support.sha256_file(r236),
                "expected_sha256": support.sha256_file(r236),
            },
            "r225_r246_replacement_summary": {
                "owner_decision_revision": support.R225_OWNER_DECISION_REVISION,
                "replacement_submission_count_limit": 1,
                "replacement_submission_consumed": False,
                "competition": support.COMPETITION,
                "submission_message_required_literal": support.LABEL,
                "complete_ordered_action_cap": support.ACTION_CAP,
                "canonical_libcg_contract_sha256": support.sha256_file(r236),
            },
        },
        "canonical_libcg_r236": {
            "package_version": "1.32.6",
            "wheel_filename": "kaggle_environments-1.32.6-py3-none-any.whl",
            "official_wheel_sha256": _sha(b"synthetic-wheel"),
            "required_native_exports": [
                "AgentStart", "BattleStart", "SearchBegin", "SearchStep", "SearchRelease", "SearchEnd"
            ],
            "members": native_libraries,
        },
        "frozen_r195_identity": {
            "bundle_sha256": _sha(b"r195-bundle"),
            "checkpoint_sha256": _sha(model),
            "matchup_tree_sha256": _sha(tree),
        },
        "action_space": {
            "complete_ordered_action_cap": 65_536,
            "applies_at_root_and_every_private_leaf": True,
            "sampling_pruning_or_reinterpretation_allowed": False,
            "root_over_cap_behavior": "hard_fail_nonzero",
            "private_leaf_over_cap_behavior": (
                "contained_degraded_parent_direct_fallback_only_after_validated_"
                "precomputed_root_direct_action_and_exact_child_reap"
            ),
        },
        "simulator_search_topology": {
            "lane_count": lane_count,
            "historical_eight_lane_manifest_or_receipt_accepted": False,
            "phase1_submission_environment": {
                **support.PHASE1_RESOURCES,
                "archive_max_bytes": support.PHASE1_ARCHIVE_MAX_BYTES,
            },
            "phase1_manifest_resource_bounds": {
                **support.PHASE1_MANIFEST_RESOURCE_BOUNDS,
                "vcpus": phase1_vcpus,
            },
        },
        "r240_hybrid_scheduler": {
            **support.R240_NORMALIZED_SCHEDULER,
            "high_confidence_mode": "high_confidence_frozen_direct",
            "legacy_fixed_eight_second_window_accepted": False,
        },
        "r240_manifest_scheduler": support.R240_MANIFEST_SCHEDULER,
        "r244_handle_scoped_search_id_identity": support.R244_SEARCH_ID_IDENTITY_CONTRACT,
        "r246_proven_deterministic_terminal_win_this_turn": (
            support.R246_LOCAL_TERMINAL_WIN_CONTRACT
        ),
        "deterministic_continuation": {
            "max_depth": 8,
            "exact_observation_fingerprint_required": True,
            "both_lanes_same_fingerprint_and_backed_action_required": True,
            "same_root_actor_required": True,
            "chance_or_boundary_forbidden": True,
            "no_new_search_on_valid_match": True,
            "mismatch_clears_entire_plan": True,
        },
        "manifest_deterministic_continuation": support.MANIFEST_DETERMINISTIC_CONTINUATION,
        "required_submission": {
            "competition": support.COMPETITION,
            "label": support.LABEL,
            "go_first_preference": "first_if_allowed",
        },
        "local_gate_receipts": {
            name: {
                "receipt_name": support.R235_BINDING_GATE_RECEIPT_NAMES[name],
                "passed": True,
                "sha256": "sha256:" + "0" * 64,
                **(
                    {"validated_counter_projection": r244_projection}
                    if name
                    in {
                        "two_lane_topology",
                        "official_libcg_handle_scoped_search_id_identity",
                    }
                    else {"validated_counter_projection": r246_projection}
                    if name == support.R246_TERMINAL_WIN_GATE
                    else {}
                ),
            }
            for name in gate_names
        },
        "authorization": {
            "replacement_submission_count_limit": 1,
            "replacement_submission_consumed": False,
            "kaggle_api_call_permitted_now": False,
            "kaggle_upload_permitted_now": False,
            "operator_directed_direct_upload_permitted_only_after_all_gates_and_binding": True,
            "queue_or_batch_submission_allowed": False,
            "automatic_retry_allowed": False,
            "automatic_copy_or_resubmission_allowed": False,
            "second_upload_allowed": False,
            "validation_failure_behavior": (
                "preserve_and_download_logs_do_not_retry_queue_copy_or_upload_"
                "without_new_owner_order"
            ),
            "training_authority": False,
            "gpu_authority": False,
            "service_authority": False,
            "bo1000_authority": False,
        },
        "builder": {
            "network_accessed": False,
            "kaggle_api_called": False,
            "kaggle_queue_used": False,
            "kaggle_upload_used": False,
            "gpu_used": False,
            "service_modified": False,
            "bo1000_modified": False,
        },
    }
    binding_path = tmp_path / "binding.json"
    if publish_binding:
        _write_json(binding_path, binding)
        os.chmod(binding_path, 0o444)
    return {
        "archive": archive,
        "manifest": manifest_path,
        "manifest_member": manifest_member,
        "binding": binding_path,
        "binding_payload": binding,
        "r225": r225,
        "r236": r236,
    }


def _candidate_context(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "archive_path": values["archive"],
        "manifest_path": values["manifest"],
        "manifest_member": values["manifest_member"],
        "r225_contract": values["r225"],
        "r236_contract": values["r236"],
    }


def _context(values: dict[str, Any]) -> dict[str, Any]:
    return {**_candidate_context(values), "binding_path": values["binding"]}


def _publish_binding(values: dict[str, Any]) -> None:
    assert not values["binding"].exists()
    _write_json(values["binding"], values["binding_payload"])
    os.chmod(values["binding"], 0o444)


def _bind_go_first_sidecar_digest(values: dict[str, Any], sidecar: Path) -> None:
    """Model the builder's gate-digest binding without changing its owner code."""

    digest = support.sha256_file(sidecar)
    if values["binding"].exists():
        os.chmod(values["binding"], 0o644)
        payload = json.loads(values["binding"].read_text(encoding="utf-8"))
    else:
        payload = values["binding_payload"]
    payload["local_gate_receipts"]["go_first"]["sha256"] = digest
    values["binding_payload"] = payload
    if values["binding"].exists():
        _write_json(values["binding"], payload)
        os.chmod(values["binding"], 0o444)


def _authorized(
    values: dict[str, Any], *, now_epoch: float = 1_000.0
) -> tuple[dict[str, Any], Path, Path]:
    context = _context(values)
    sidecar = Path(str(values["archive"]) + ".go-first-verified.json")
    support.verify_go_first(**_candidate_context(values), output_path=sidecar)
    _bind_go_first_sidecar_digest(values, sidecar)
    if not values["binding"].exists():
        _publish_binding(values)
    consumption = values["archive"].parent / "r235-consumed.json"
    authorization = support.build_one_use_authorization(
        **context,
        go_first_receipt_path=sidecar,
        consumption_path=consumption,
        output_path=values["archive"].parent / "authorization.json",
        nonce="r235-test-nonce",
        expires_at_epoch=now_epoch + 1_000.0,
        now_epoch=now_epoch,
    )
    return authorization, sidecar, consumption


def test_verify_go_first_cli_requires_no_binding_path() -> None:
    parsed = support._args().parse_args(
        [
            "verify-go-first",
            "--archive",
            "candidate.tar.gz",
            "--manifest",
            "candidate-manifest.json",
            "--manifest-member",
            "r238-manifest.json",
            "--output",
            "candidate.tar.gz.go-first-verified.json",
        ]
    )
    assert parsed.command == "verify-go-first"
    assert not hasattr(parsed, "binding")


def test_go_first_receipt_and_one_use_authorization_are_digest_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _inputs(tmp_path, monkeypatch)
    context = support.validate_r235_binding(**_context(values))
    assert context["package"]["manifest"]["lane_count"] == 2
    authorization, sidecar, consumption = _authorized(values)
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
    assert (
        authorization["r246_proven_deterministic_terminal_win_this_turn_revision"]
        == support.R246_PROVEN_TERMINAL_WIN_REVISION
    )
    valid, _reason, details = support.validate_r235_authorization(
        authorization,
        file_path=values["archive"],
        file_sha256=support.sha256_file(values["archive"]),
        competition=support.COMPETITION,
        message=support.LABEL,
    )
    assert valid is True
    assert details["consumption_path"] == str(consumption.resolve())
    support.consume_r235_authority(
        authorization,
        identity={
            "file": str(values["archive"]),
            "file_sha256": support.sha256_file(values["archive"]),
            "competition": support.COMPETITION,
            "message": support.LABEL,
        },
    )
    assert consumption.is_file()
    assert stat.S_IMODE(consumption.stat().st_mode) == 0o444
    assert support.validate_r235_authorization(
        authorization,
        file_path=values["archive"],
        file_sha256=support.sha256_file(values["archive"]),
        competition=support.COMPETITION,
        message=support.LABEL,
    )[0] is False


def test_candidate_then_prebinding_sidecar_then_binding_then_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _inputs(tmp_path, monkeypatch, publish_binding=False)
    sidecar = Path(str(values["archive"]) + ".go-first-verified.json")

    # The candidate can be verified before a binding file exists.  This is the
    # required order because the builder consumes this exact sidecar as a gate.
    assert not values["binding"].exists()
    verified = support.verify_go_first(
        **_candidate_context(values), output_path=sidecar
    )
    assert verified["candidate_archive_sha256"] == support.sha256_file(values["archive"])
    assert "r235_binding" not in verified
    assert set(json.loads(sidecar.read_text(encoding="utf-8"))) == support.GO_FIRST_PREBINDING_RECEIPT_FIELDS

    # The synthetic binding stands in for the builder output and receives the
    # same immutable gate digest that the real builder records.
    _bind_go_first_sidecar_digest(values, sidecar)
    _publish_binding(values)
    authorization = support.build_one_use_authorization(
        **_context(values),
        go_first_receipt_path=sidecar,
        consumption_path=tmp_path / "r235-consumed.json",
        output_path=tmp_path / "authorization.json",
        nonce="r235-order-test",
        expires_at_epoch=2_000.0,
        now_epoch=1_000.0,
    )
    assert authorization["r235_go_first_receipt_sha256"] == support.sha256_file(sidecar)


def test_stale_or_mismatched_prebinding_sidecar_is_rejected_after_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _inputs(tmp_path, monkeypatch, publish_binding=False)
    sidecar = Path(str(values["archive"]) + ".go-first-verified.json")
    support.verify_go_first(**_candidate_context(values), output_path=sidecar)
    _bind_go_first_sidecar_digest(values, sidecar)
    _publish_binding(values)

    # Any post-binding sidecar rewrite invalidates both its digest and its
    # archive identity.  It must never mint a new authorization.
    os.chmod(sidecar, 0o644)
    stale = json.loads(sidecar.read_text(encoding="utf-8"))
    stale["candidate_archive_sha256"] = "sha256:" + "f" * 64
    _write_json(sidecar, stale)
    os.chmod(sidecar, 0o444)
    with pytest.raises(support.R235SupportError):
        support.build_one_use_authorization(
            **_context(values),
            go_first_receipt_path=sidecar,
            consumption_path=tmp_path / "r235-consumed.json",
            output_path=tmp_path / "authorization.json",
            nonce="r235-stale-test",
            expires_at_epoch=2_000.0,
            now_epoch=1_000.0,
        )
    assert not (tmp_path / "authorization.json").exists()


def test_prebinding_verifier_rejects_embedded_r236_manifest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _inputs(tmp_path, monkeypatch, publish_binding=False)
    manifest = json.loads(values["manifest"].read_text(encoding="utf-8"))
    manifest["canonical_libcg_contract"]["canonical_native_libraries"]["linux_x86_64"][
        "sha256"
    ] = "sha256:" + "e" * 64
    _write_json(values["manifest"], manifest)
    archive_files = {
        member: support._archive_member_bytes(values["archive"], member)
        for member in support._archive_members(values["archive"])
    }
    archive_files[values["manifest_member"]] = values["manifest"].read_bytes()
    _write_archive(values["archive"], archive_files)

    with pytest.raises(support.R235SupportError):
        support.verify_go_first(
            **_candidate_context(values),
            output_path=Path(str(values["archive"]) + ".go-first-verified.json"),
        )


def test_guard_requires_the_r235_binding_before_consuming_its_generic_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _inputs(tmp_path, monkeypatch)
    authorization, _sidecar, _consumption = _authorized(values, now_epoch=time.time())
    monkeypatch.setattr(guard, "_r235_tooling", lambda: support)
    valid, reason, details = guard._validate_authorization(
        authorization,
        [
            "competitions",
            "submit",
            "-c",
            support.COMPETITION,
            "-f",
            str(values["archive"]),
            "-m",
            support.LABEL,
        ],
    )
    assert valid is True
    assert reason == "authorized"
    assert details["checks"]["r235_binding"] is True


def test_r244_marker_without_the_complete_r235_authority_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _inputs(tmp_path, monkeypatch)
    authorization, _sidecar, _consumption = _authorized(values, now_epoch=time.time())
    monkeypatch.setattr(guard, "_r235_tooling", lambda: support)
    malformed = {
        "schema": support.AUTH_SCHEMA,
        "explicit_user_approval": True,
        "remaining_uses": 1,
        "nonce": authorization["nonce"],
        "expires_at_epoch": authorization["expires_at_epoch"],
        "competition": support.COMPETITION,
        "file_sha256": support.sha256_file(values["archive"]),
        "message": support.LABEL,
        "turn_order_preference": "first_if_allowed",
        # R244 is part of the R235 package contract; it may not be used as a
        # way to evade the complete immutable binding and one-use authority.
        "r244_handle_scoped_search_id_revision": 244,
    }
    valid, _reason, details = guard._validate_authorization(
        malformed,
        [
            "competitions",
            "submit",
            "-c",
            support.COMPETITION,
            "-f",
            str(values["archive"]),
            "-m",
            support.LABEL,
        ],
    )
    assert valid is False
    assert details["checks"]["r235_binding"] is False


def test_r246_marker_is_also_fail_closed_without_r235_authority() -> None:
    assert guard._has_r235_marker(
        {"r246_proven_deterministic_terminal_win_this_turn_revision": 246}
    ) is True
    assert support._r235_marker(
        {"r246_proven_deterministic_terminal_win_this_turn_revision": 246}
    ) is True


def test_go_first_probe_blocks_process_or_network_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _inputs(
        tmp_path,
        monkeypatch,
        direct_entrypoint=b"import subprocess\n\ndef agent(_observation):\n    return [0]\n",
    )
    with pytest.raises(support.R235SupportError):
        support.verify_go_first(
            **_candidate_context(values),
            output_path=Path(str(values["archive"]) + ".go-first-verified.json"),
        )


def test_stale_gpu_availability_claim_in_r238_manifest_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _inputs(tmp_path, monkeypatch)
    manifest = json.loads(values["manifest"].read_text(encoding="utf-8"))
    manifest["phase1_kaggle_resource_bounds"] = {
        "vcpus": 2,
        "ram_gib": 12.2,
        "hdd_gib": 11.8,
        "archive_mib": 197.7,
        "gpu_available": False,
        "archive_max_bytes": support.PHASE1_ARCHIVE_MAX_BYTES,
    }
    _write_json(values["manifest"], manifest)
    archive_files = {
        member: support._archive_member_bytes(values["archive"], member)
        for member in support._archive_members(values["archive"])
    }
    archive_files[values["manifest_member"]] = values["manifest"].read_bytes()
    _write_archive(values["archive"], archive_files)

    with pytest.raises(support.R235SupportError):
        support.validate_r235_candidate(**_candidate_context(values))


@pytest.mark.parametrize("lane_count,phase1_vcpus", [(8, 2), (2, 8)])
def test_legacy_eight_lane_or_phase1_resource_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lane_count: int,
    phase1_vcpus: int,
) -> None:
    values = _inputs(tmp_path, monkeypatch, lane_count=lane_count, phase1_vcpus=phase1_vcpus)
    with pytest.raises(support.R235SupportError):
        support.validate_r235_binding(**_context(values))


def test_r244_allows_repeated_raw_ids_only_across_distinct_handle_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _inputs(tmp_path, monkeypatch)

    # The valid fixture deliberately proves the R244 correction: both native
    # handles can start at raw SearchId 0 because the owning handles differ.
    assert support.validate_r235_binding(**_context(values))["payload"]["local_gate_receipts"][
        "two_lane_topology"
    ]["validated_counter_projection"]["per_lane_first_search_ids"] == [0, 0]

    # Reusing the raw ID is not a waiver for reusing a handle.  The immutable
    # binding must retain two distinct (handle_identity, first_search_id)
    # composite states at both the topology and official-libcg gates.
    os.chmod(values["binding"], 0o644)
    binding = json.loads(values["binding"].read_text(encoding="utf-8"))
    projection = binding["local_gate_receipts"][
        "official_libcg_handle_scoped_search_id_identity"
    ]["validated_counter_projection"]
    projection["per_lane_handle_identities"] = ["handle-a", "handle-a"]
    projection["handle_scoped_first_search_id_composite_states"][1]["handle_identity"] = "handle-a"
    _write_json(values["binding"], binding)
    os.chmod(values["binding"], 0o444)

    with pytest.raises(support.R235SupportError):
        support.validate_r235_binding(**_context(values))


def test_r246_terminal_win_contract_and_one_proof_projection_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _inputs(tmp_path, monkeypatch)
    binding = support.validate_r235_binding(**_context(values))["payload"]
    assert (
        binding["r246_proven_deterministic_terminal_win_this_turn"]
        == support.R246_LOCAL_TERMINAL_WIN_CONTRACT
    )
    projection = binding["local_gate_receipts"][
        support.R246_TERMINAL_WIN_GATE
    ]["validated_counter_projection"]
    assert projection["terminal_win_proof"]["selected_action"] == [0]
    too_many_backups = json.loads(json.dumps(projection))
    too_many_backups["completed_root_backup_count"] = 33
    with pytest.raises(support.R235SupportError):
        support._validate_r246_terminal_win_projection(
            too_many_backups, label="synthetic R246 projection"
        )
    too_long_path = json.loads(json.dumps(projection))
    too_long_path["terminal_win_proof"]["proof_path_action_count"] = 2
    too_long_path["terminal_win_proof"]["path_actor_seats"] = [0, 0]
    with pytest.raises(support.R235SupportError):
        support._validate_r246_terminal_win_projection(
            too_long_path, label="synthetic R246 projection"
        )

    # The historical ``..._completed`` spelling is not an alias.  It must
    # fail closed so an older receipt cannot authorize this R246 exception.
    legacy_cleanup = json.loads(json.dumps(projection))
    legacy_cleanup[support.R246_LEGACY_CLEANUP_COMPLETED_FIELD] = legacy_cleanup.pop(
        support.R246_CLEANUP_COMPLETE_FIELD
    )
    with pytest.raises(
        support.R235SupportError,
        match=support.R246_LEGACY_CLEANUP_COMPLETED_FIELD,
    ):
        support._validate_r246_terminal_win_projection(
            legacy_cleanup, label="synthetic R246 projection"
        )

    # A passed gate digest is not terminal-win authority by itself: the sealed
    # projection must still prove an exact stock-simulator win for the root
    # actor with no boundary crossing.
    os.chmod(values["binding"], 0o644)
    malformed = json.loads(values["binding"].read_text(encoding="utf-8"))
    malformed["local_gate_receipts"][support.R246_TERMINAL_WIN_GATE][
        "validated_counter_projection"
    ]["terminal_win_proof"]["terminal_result"] = "draw"
    _write_json(values["binding"], malformed)
    os.chmod(values["binding"], 0o444)
    with pytest.raises(support.R235SupportError):
        support.validate_r235_binding(**_context(values))


def test_r246_legacy_cleanup_spelling_is_rejected_by_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _inputs(tmp_path, monkeypatch)
    os.chmod(values["binding"], 0o644)
    binding = json.loads(values["binding"].read_text(encoding="utf-8"))
    projection = binding["local_gate_receipts"][support.R246_TERMINAL_WIN_GATE][
        "validated_counter_projection"
    ]
    projection[support.R246_LEGACY_CLEANUP_COMPLETED_FIELD] = projection.pop(
        support.R246_CLEANUP_COMPLETE_FIELD
    )
    _write_json(values["binding"], binding)
    os.chmod(values["binding"], 0o444)

    with pytest.raises(
        support.R235SupportError,
        match=support.R246_LEGACY_CLEANUP_COMPLETED_FIELD,
    ):
        support.validate_r235_binding(**_context(values))


def test_capture_submission_id_and_seal_both_seat_evidence_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _inputs(tmp_path, monkeypatch)
    authorization, _sidecar, consumption = _authorized(values)
    support.consume_r235_authority(
        authorization,
        identity={
            "file": str(values["archive"]),
            "file_sha256": support.sha256_file(values["archive"]),
            "competition": support.COMPETITION,
            "message": support.LABEL,
        },
    )
    consumed = {
        **authorization,
        "remaining_uses": 0,
        "consumed_before_upload": True,
        "r235_authority_consumption": str(consumption.resolve()),
    }
    consumed_path = tmp_path / "authorization-consumed.json"
    _write_json(consumed_path, consumed)
    attempt_path = tmp_path / "attempt.json"
    _write_json(
        attempt_path,
        {
            "schema": "poke_bot.kaggle_submission_attempt/v1",
            "nonce": authorization["nonce"],
            "authorization_consumed": str(consumed_path.resolve()),
            "returncode": 0,
            "identity": {
                "competition": support.COMPETITION,
                "message": support.LABEL,
                "file_sha256": support.sha256_file(values["archive"]),
                "r235_authority_consumption": str(consumption.resolve()),
            },
        },
    )
    guard_output = tmp_path / "guard.out"
    guard_output.write_text("Submission ID: 765432\n", encoding="utf-8")
    snapshot = tmp_path / "api.json"
    _write_json(
        snapshot,
        {"competition": support.COMPETITION, "rows": [{"description": support.LABEL, "ref": "765432"}]},
    )
    id_receipt = support.capture_submission_id(
        **_context(values),
        authorization_consumed_path=consumed_path,
        attempt_path=attempt_path,
        guard_output_path=guard_output,
        api_snapshot_path=snapshot,
        output_path=tmp_path / "submission-id.json",
    )
    assert id_receipt["submission"]["id"] == 765432
    metadata = tmp_path / "episode.json"
    _write_json(
        metadata,
        {
            "episode_id": 917_999,
            "agents": [
                {"index": 0, "submission_id": 765432},
                {"index": 1, "submission_id": 123},
            ],
        },
    )
    replay = tmp_path / "replay.json"
    replay.write_text('{"replay": true}\n', encoding="utf-8")
    seat_0 = tmp_path / "seat-0.json"
    seat_1 = tmp_path / "seat-1.json"
    seat_0.write_text('{"seat": 0}\n', encoding="utf-8")
    seat_1.write_text('{"seat": 1}\n', encoding="utf-8")
    evidence_receipt = evidence.materialize_evidence(
        **_context(values),
        submission_id_receipt_path=Path(id_receipt["path"]),
        episode_metadata_path=metadata,
        replay_path=replay,
        seat_0_log_path=seat_0,
        seat_1_log_path=seat_1,
        output_path=tmp_path / "evidence",
    )
    assert evidence_receipt["episode"]["submitted_seat"] == 0
    assert set(evidence_receipt["files"]) == {
        "episode_metadata",
        "replay",
        "seat_0_logs",
        "seat_1_logs",
    }
    assert all(
        row["source_sha256"] == row["sha256"] and row["bytes"] > 0
        for row in evidence_receipt["files"].values()
    )
    assert stat.S_IMODE(Path(evidence_receipt["path"]).stat().st_mode) == 0o444


def test_submission_id_capture_rejects_disagreeing_guard_and_api_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _inputs(tmp_path, monkeypatch)
    authorization, _sidecar, consumption = _authorized(values)
    support.consume_r235_authority(
        authorization,
        identity={
            "file": str(values["archive"]),
            "file_sha256": support.sha256_file(values["archive"]),
            "competition": support.COMPETITION,
            "message": support.LABEL,
        },
    )
    consumed_path = tmp_path / "authorization-consumed.json"
    _write_json(
        consumed_path,
        {
            **authorization,
            "remaining_uses": 0,
            "consumed_before_upload": True,
            "r235_authority_consumption": str(consumption.resolve()),
        },
    )
    attempt_path = tmp_path / "attempt.json"
    _write_json(
        attempt_path,
        {
            "schema": "poke_bot.kaggle_submission_attempt/v1",
            "nonce": authorization["nonce"],
            "authorization_consumed": str(consumed_path.resolve()),
            "returncode": 0,
            "identity": {
                "competition": support.COMPETITION,
                "message": support.LABEL,
                "file_sha256": support.sha256_file(values["archive"]),
                "r235_authority_consumption": str(consumption.resolve()),
            },
        },
    )
    guard_output = tmp_path / "guard.out"
    guard_output.write_text("Submission ID: 765432\n", encoding="utf-8")
    snapshot = tmp_path / "api.json"
    _write_json(
        snapshot,
        {
            "competition": support.COMPETITION,
            "rows": [{"description": support.LABEL, "ref": "765433"}],
        },
    )
    output = tmp_path / "submission-id.json"
    with pytest.raises(support.R235SupportError):
        support.capture_submission_id(
            **_context(values),
            authorization_consumed_path=consumed_path,
            attempt_path=attempt_path,
            guard_output_path=guard_output,
            api_snapshot_path=snapshot,
            output_path=output,
        )
    assert not output.exists()


def test_download_path_uses_one_call_per_replay_and_each_seat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _inputs(tmp_path, monkeypatch)
    authorization, _sidecar, consumption = _authorized(values)
    support.consume_r235_authority(
        authorization,
        identity={
            "file": str(values["archive"]),
            "file_sha256": support.sha256_file(values["archive"]),
            "competition": support.COMPETITION,
            "message": support.LABEL,
        },
    )
    consumed_path = tmp_path / "authorization-consumed.json"
    _write_json(
        consumed_path,
        {
            **authorization,
            "remaining_uses": 0,
            "consumed_before_upload": True,
            "r235_authority_consumption": str(consumption.resolve()),
        },
    )
    attempt_path = tmp_path / "attempt.json"
    _write_json(
        attempt_path,
        {
            "schema": "poke_bot.kaggle_submission_attempt/v1",
            "nonce": authorization["nonce"],
            "authorization_consumed": str(consumed_path.resolve()),
            "returncode": 0,
            "identity": {
                "competition": support.COMPETITION,
                "message": support.LABEL,
                "file_sha256": support.sha256_file(values["archive"]),
                "r235_authority_consumption": str(consumption.resolve()),
            },
        },
    )
    guard_output = tmp_path / "guard.out"
    guard_output.write_text("submission=765432\n", encoding="utf-8")
    id_receipt = support.capture_submission_id(
        **_context(values),
        authorization_consumed_path=consumed_path,
        attempt_path=attempt_path,
        guard_output_path=guard_output,
        output_path=tmp_path / "submission-id.json",
    )
    calls: list[tuple[str, int | None]] = []

    class FakeApi:
        def competition_list_episodes(self, submission_id: int) -> list[dict[str, Any]]:
            calls.append(("list", submission_id))
            return [
                {
                    "id": 918_000,
                    "agents": [
                        {"index": 0, "submission_id": 765432},
                        {"index": 1, "submission_id": 1},
                    ],
                }
            ]

        def competition_episode_replay(self, episode_id: int, destination: str) -> None:
            calls.append(("replay", episode_id))
            (Path(destination) / f"episode-{episode_id}-replay.json").write_text("{}\n")

        def competition_episode_agent_logs(self, episode_id: int, seat: int, destination: str) -> None:
            calls.append((f"logs-{seat}", episode_id))
            (Path(destination) / f"episode-{episode_id}-agent-{seat}-logs.json").write_text("{}\n")

    receipt = evidence.download_evidence_once(
        FakeApi(),
        **_context(values),
        submission_id_receipt_path=Path(id_receipt["path"]),
        episode_id=918_000,
        output_path=tmp_path / "downloaded-evidence",
    )
    assert receipt["acquisition"]["network_calls"] == 4
    assert calls == [("list", 765432), ("replay", 918_000), ("logs-0", 918_000), ("logs-1", 918_000)]
