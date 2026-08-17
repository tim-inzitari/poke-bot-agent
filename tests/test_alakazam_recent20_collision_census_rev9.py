import json
from pathlib import Path

import pytest

from poke_bot.alakazam_recent20_collision_census_rev9 import (
    _ROW,
    _analyse_sorted,
    _semantic_diff_paths,
    extract_unresolved_repaired_key_inventory,
    reseal_with_permutation_evidence,
    reseal_with_targeted_witness_evidence,
    seal_permutation_equivalence_receipt,
    seal_targeted_transition_witness_receipt,
    sha256_file,
)


def _row(public: int, legacy: int, semantic: int, action: int, *, selected: bool = False) -> bytes:
    b = lambda value: value.to_bytes(32, "big")
    return _ROW.pack(b(public), b(legacy), b(semantic), b(action), int(selected), 0, b(0), b(0))


def test_streaming_merge_detects_legacy_alias_and_repaired_key(tmp_path: Path):
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"".join(sorted([_row(1, 2, 10, 100, selected=True), _row(8, 9, 80, 800)])))
    second.write_bytes(b"".join(sorted([_row(1, 2, 11, 101), _row(7, 9, 70, 700)])))
    result = _analyse_sorted([first, second])
    assert result["all_option_record_count"] == 4
    assert result["decision_count"] == 1
    assert result["legacy_public_semantic_collision_group_count"] == 1
    assert result["legacy_collision_record_count"] == 2
    assert result["unresolved_repaired_semantic_key_collision_group_count"] == 0
    assert result["incomplete_pinned_transition_evidence_record_count"] == 4


def test_same_repaired_semantic_key_with_different_raw_actions_is_held(tmp_path: Path):
    spool = tmp_path / "a.bin"
    spool.write_bytes(b"".join(sorted([_row(1, 2, 10, 100), _row(1, 2, 10, 101)])))
    result = _analyse_sorted([spool])
    assert result["legacy_public_semantic_collision_group_count"] == 0
    assert result["unresolved_repaired_semantic_key_collision_group_count"] == 1
    inventory = extract_unresolved_repaired_key_inventory([spool])
    assert len(inventory) == 1
    assert len(inventory[0]["candidate_action_sha256s"]) == 2


def _targeted_payload(*, invalid_field: bool = False) -> dict:
    records = []
    inventory = []
    for group in range(99):
        public = f"sha256:{group + 1:064x}"
        legacy = f"sha256:{group + 101:064x}"
        semantic = f"sha256:{group + 201:064x}"
        inventory.append(
            {
                "canonical_public_observation_hash": public,
                "legacy_current_feature_token_hash": legacy,
                "new_complete_semantic_option_key_sha256": semantic,
                "candidate_action_sha256s": [f"sha256:{group + 301:064x}", f"sha256:{group + 401:064x}"],
            }
        )
        seats = (0, 1) if group < 10 else (0, 0)
        for offset, seat in enumerate(seats):
            raw = {"type": 3, "area": 2, "index": group, "playerIndex": seat}
            if invalid_field and group == 0 and offset == 1:
                raw["index"] += 1
            records.append(
                {
                    "canonical_public_observation_hash": public,
                    "legacy_current_feature_token_hash": legacy,
                    "new_complete_semantic_option_key_sha256": semantic,
                    "normalized_canonical_option_multiset_hash": f"sha256:{group + 501:064x}",
                    "normalized_public_rule_semantic_token_hash": f"sha256:{group + 601:064x}",
                    "candidate_action_sha256": f"sha256:{group + 301 + 100 * offset:064x}",
                    "candidate_action": [offset],
                    "complete_raw_option_payload": [raw],
                    "complete_raw_option_payload_audit_only": True,
                    "source": {"acting_seat": seat},
                }
            )
    records.append(dict(records[-1], candidate_action=[2], candidate_action_sha256=f"sha256:{999:064x}"))
    return {
        "schema": "poke_bot.alakazam_recent20_targeted_permutation_equivalence_rows/v1",
        "goal_revision": 9,
        "goal_contract_sha256": "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8",
        "recent20_manifest_sha256": f"sha256:{777:064x}",
        "unresolved_group_count": 99,
        "target_record_count": 199,
        "inventory": inventory,
        "records": records,
        "raw_zip_reopened": False,
    }


def test_seals_only_candidate_order_and_actor_relative_player_index(tmp_path: Path):
    targeted = tmp_path / "targeted.json"
    targeted.write_text(json.dumps(_targeted_payload()))
    result = seal_permutation_equivalence_receipt(
        targeted_rows_path=targeted,
        output_path=tmp_path / "receipt.json",
        collision_census_receipt_sha256=f"sha256:{888:064x}",
    )
    assert result["group_count"] == 99
    assert result["record_count"] == 199
    assert result["equivalence_reason_counts"] == {
        "candidate_order_and_actor_relative_player_index_only": 10,
        "candidate_order_only_identical_raw_payload": 89,
    }


def test_rejects_other_raw_option_difference(tmp_path: Path):
    targeted = tmp_path / "targeted.json"
    targeted.write_text(json.dumps(_targeted_payload(invalid_field=True)))
    with pytest.raises(RuntimeError, match="not harmless permutation-equivalent"):
        seal_permutation_equivalence_receipt(
            targeted_rows_path=targeted,
            output_path=tmp_path / "receipt.json",
            collision_census_receipt_sha256=f"sha256:{888:064x}",
        )


def test_reseals_branch_after_permutation_evidence_without_rescan(tmp_path: Path):
    manifest_sha = f"sha256:{777:064x}"
    report_path = tmp_path / "old-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.alakazam_recent20_collision_census_report/v1",
                "goal_revision": 9,
                "goal_contract_sha256": "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8",
                "recent20_manifest_sha256": manifest_sha,
                "analysis": {
                    "all_option_record_count": 199,
                    "decision_count": 99,
                    "legacy_public_semantic_collision_group_count": 7,
                    "unresolved_repaired_semantic_key_collision_group_count": 99,
                },
            }
        )
    )
    old_receipt_path = tmp_path / "old-receipt.json"
    old_receipt_path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.alakazam_recent20_collision_census_receipt/v1",
                "collision_census_report_sha256": sha256_file(report_path),
            }
        )
    )
    permutation_path = tmp_path / "permutation.json"
    permutation_path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.alakazam_recent20_permutation_equivalence_receipt/v1",
                "goal_revision": 9,
                "goal_contract_sha256": "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8",
                "collision_census_receipt_sha256": sha256_file(old_receipt_path),
                "recent20_manifest_sha256": manifest_sha,
                "targeted_equivalence_group_count": 99,
                "equivalence_reason_counts": {
                    "candidate_order_and_actor_relative_player_index_only": 10,
                    "candidate_order_only_identical_raw_payload": 89,
                },
                "harmless_permutation_equivalence_passed": True,
            }
        )
    )
    result = reseal_with_permutation_evidence(
        previous_report_path=report_path,
        previous_census_receipt_path=old_receipt_path,
        permutation_receipt_path=permutation_path,
        output_root=tmp_path / "resealed",
    )
    receipt = json.loads((tmp_path / "resealed" / "collision-census-receipt.json").read_text())
    branch = json.loads((tmp_path / "resealed" / "branch-adjudication-receipt.json").read_text())
    assert receipt["actionable_unresolved_repaired_collision_group_count"] == 0
    assert receipt["permutation_equivalence_evidence_complete"] is True
    assert branch["targeted_permutation_equivalence_evidence_required"] is False
    assert branch["representationally_supported_but_not_yet_trainable_branches"] == [
        "public_rule_semantic_projection"
    ]
    assert result["status"] == "complete_fail_closed_branch_adjudication_pending_targeted_engine_evidence"


def test_semantic_delta_paths_are_field_specific_and_list_order_stable():
    left = {
        "factorized_stage": {
            "candidate_semantics": [{"option": {"number": 4, "source": {"area": "deck"}}}]
        },
        "normalized_selection": {"remain_energy_cost": 1},
    }
    right = {
        "factorized_stage": {
            "candidate_semantics": [{"option": {"number": 5, "source": {"area": "hand"}}}]
        },
        "normalized_selection": {"remain_energy_cost": 2},
    }
    assert _semantic_diff_paths(left, right) == {
        "factorized_stage.candidate_semantics[].option.number",
        "factorized_stage.candidate_semantics[].option.source.area",
        "normalized_selection.remain_energy_cost",
    }


def test_targeted_witness_receipt_keeps_unselected_outcomes_unclaimed(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "poke_bot.alakazam_recent20_refeaturized_corpus_manifest/v1",
        "goal_revision": 9,
        "goal_contract_sha256": "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8",
        "finalized_shard_count": 20,
        "record_count": 18412973,
    }))
    permutation = tmp_path / "permutation.json"
    permutation.write_text(json.dumps({
        "schema": "poke_bot.alakazam_recent20_permutation_equivalence_receipt/v1",
        "harmless_permutation_equivalence_passed": True,
    }))
    libcg = tmp_path / "libcg.so"
    libcg.write_bytes(b"not-the-pinned-binary")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "schema": "poke_bot.alakazam_collision_census_r298_frozen_schema_manifest/v1",
        "canonical_simulator": {
            "linux_x86_64_sha256": sha256_file(libcg),
            "linux_x86_64_size_bytes": libcg.stat().st_size,
        },
    }))
    class_rows = []
    for index in range(6):
        selected = 1 if index < 5 else 0
        class_rows.append({
            "semantic_delta_paths": [f"field{index}"],
            "group_count": 127636 if index == 0 else 1,
            "record_count": 2,
            "selected_record_count": selected,
            "examples": [{
                "known_selected_successor_count": selected,
                "selected_record_count": selected,
            }],
        })
    classes = tmp_path / "classes.json"
    classes.write_text(json.dumps({
        "schema": "poke_bot.alakazam_recent20_legacy_alias_semantic_delta_classes/v1",
        "goal_revision": 9,
        "goal_contract_sha256": "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8",
        "recent20_manifest_sha256": sha256_file(manifest),
        "legacy_alias_group_count": 127641,
        "legacy_alias_record_count": 255398,
        "semantic_delta_class_count": 6,
        "selected_records_missing_adjacent_public_successor": 0,
        "classes": class_rows,
        "raw_zip_reopened": False,
    }))
    with pytest.raises(RuntimeError, match="canonical libcg bytes differ"):
        seal_targeted_transition_witness_receipt(
            semantic_class_report_path=classes,
            permutation_receipt_path=permutation,
            recent20_manifest_path=manifest,
            frozen_schema_manifest_path=freeze,
            canonical_libcg_path=libcg,
            output_path=tmp_path / "witness.json",
        )


def test_targeted_reseal_promotes_only_public_semantic_projection(tmp_path: Path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "schema": "poke_bot.alakazam_recent20_collision_census_report/v1",
        "goal_revision": 9,
        "goal_contract_sha256": "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8",
        "recent20_manifest_sha256": f"sha256:{777:064x}",
        "analysis": {
            "all_option_record_count": 10,
            "decision_count": 5,
            "legacy_public_semantic_collision_group_count": 3,
        },
    }))
    census = tmp_path / "census.json"
    census.write_text(json.dumps({
        "schema": "poke_bot.alakazam_recent20_collision_census_receipt/v1",
        "collision_census_report_sha256": sha256_file(report),
        "permutation_equivalence_evidence_complete": True,
        "actionable_unresolved_repaired_collision_group_count": 0,
        "permutation_equivalence_receipt_sha256": f"sha256:{701:064x}",
    }))
    branch = tmp_path / "branch.json"
    branch.write_text(json.dumps({
        "schema": "poke_bot.alakazam_recent20_branch_adjudication_receipt/v1",
        "collision_census_receipt_sha256": sha256_file(census),
    }))
    witness = tmp_path / "witness.json"
    witness.write_text(json.dumps({
        "schema": "poke_bot.alakazam_recent20_targeted_transition_witness_receipt/v1",
        "goal_revision": 9,
        "goal_contract_sha256": "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8",
        "recent20_manifest_sha256": f"sha256:{777:064x}",
        "public_rule_semantic_projection_corpus_supported": True,
        "other_derivative_branches_training_supported_by_this_receipt": False,
        "semantic_delta_class_count": 6,
        "covered_legacy_alias_group_count": 127641,
        "selected_transition_witness_class_count": 5,
        "legality_only_witness_class_count": 1,
    }))
    result = reseal_with_targeted_witness_evidence(
        previous_report_path=report,
        previous_census_receipt_path=census,
        previous_branch_receipt_path=branch,
        targeted_witness_receipt_path=witness,
        output_root=tmp_path / "final",
    )
    final_branch = json.loads((tmp_path / "final" / "branch-adjudication-receipt.json").read_text())
    assert final_branch["eligible_trainable_branches"] == ["public_rule_semantic_projection"]
    assert final_branch["candidate_training_allowed"] is False
    assert set(final_branch["unsupported_or_unproven_branches"]) == {
        "public_rule_metadata_residual",
        "r298_repaired_auxiliary_heads",
        "eight_checklist_provenance_gates",
    }
    assert result["status"] == "passed_branch_adjudication_public_rule_semantic_projection_only"
