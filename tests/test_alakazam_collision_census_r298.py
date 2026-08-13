"""Focused fail-closed coverage for the offline r298 collision census."""

from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from poke_bot.alakazam_collision_census_r298 import (
    CANONICAL_R236_LIBCG_SHA256,
    CollisionCensusError,
    MECHANICS_ATTACHMENT_SHA256,
    OWNER_GOAL_SHA256,
    R298_OWNER_REVISION,
    R298_ENGINE_EVIDENCE_SCHEMA,
    R298_FROZEN_SCHEMA_MANIFEST_SCHEMA,
    R298_REV5_CENSUS_VALIDATION_RECEIPT_SCHEMA,
    R298_ZERO_BYPASS_RECEIPT_SCHEMA,
    RAW_CORPUS_SOURCE_RECEIPT_SHA256S,
    RULE_DERIVATIVE_CONTRACT_SHA256,
    RULE_DERIVATIVE_GATEWAY_SHA256,
    REVISION_4_CONTRACT_SHA256,
    REVISION_4_GATEWAY_SHA256,
    REVISION_5_GOAL_REVISION,
    REVISION_5_ROOT_HANDOFF_REVISION,
    action_key_sha256,
    analyze_collision_records,
    build_raw_corpus_manifest,
    build_stage_option_records,
    canonical_json_bytes,
    canonical_public_current_hash,
    canonical_public_observation_hash,
    canonical_sha256,
    complete_semantic_option_key,
    make_raw_corpus_receipt,
    make_receipt,
    make_revision_5_census_validation_receipt,
    inventory_raw_observations,
    raw_observations_from_recorded_episode,
    recorded_episode_frame_coverage,
    revision_5_predecessor_classification,
    stage_descriptors_from_recorded_episode,
    validate_frozen_schema_gate,
    validate_revision_5_census_validation_receipt,
    validate_revision_5_predecessor_classification,
    validate_raw_corpus_manifest,
    validate_phase_a_inventory,
)
from scripts import run_alakazam_collision_census_r298 as runner


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _card(card_id: int, serial: int) -> dict[str, int]:
    return {"id": card_id, "serial": serial}


def _player(*, hand: list[dict[str, int]] | None = None) -> dict:
    return {
        "active": [],
        "bench": [],
        "discard": [],
        "hand": hand,
        "handCount": 0 if hand is None else len(hand),
        "deckCount": 20,
        "prize": [None] * 6,
    }


def _observation() -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "turn": 4,
            "turnActionCount": 1,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "players": [_player(hand=[_card(10, 100)]), _player(hand=[_card(20, 200)])],
        },
        "select": {
            "context": "Main",
            "type": "Number",
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": "Number", "number": 4},
                {"type": "Number", "number": 5},
            ],
        },
    }


def _token_builder(_observation: dict, actions: list[list[int]]) -> SimpleNamespace:
    return SimpleNamespace(
        index=[0] * len(actions),
        value=[1.0] * len(actions),
        offset=list(range(len(actions))),
    )


def test_public_hash_uses_r298_projection_not_legacy_mask() -> None:
    first = _observation()
    second = copy.deepcopy(first)
    second.update(
        {
            "logs": {"private_opponent_deck": [901, 902]},
            "search_begin_input": {"opponent_hand_ids": [903, 904]},
            "transition_after": {"future_prizes": [905]},
            "privateState": {"hidden": True},
        }
    )
    actor = second["current"]["players"][0]
    actor["deckOrder"] = [1, 2, 3]
    actor["privateState"] = {"own_prize_ids": [4]}
    actor["deck"] = [_card(5, 500)]
    second["current"]["players"][1]["hand"] = [_card(999, 9999), _card(998, 9998)]

    assert canonical_public_observation_hash(first) == canonical_public_observation_hash(second)
    assert canonical_public_current_hash(first["current"], actor=0) == canonical_public_current_hash(
        second["current"], actor=0
    )


def test_public_collision_identity_is_invariant_to_consistent_visible_serial_renumbering() -> None:
    """Raw serials may be audited, but cannot partition semantic collisions."""

    first = _observation()
    first["current"]["players"][0]["bench"] = [_card(77, 1001)]
    first["select"] = {
        "context": "Main",
        "type": "Skill",
        "minCount": 1,
        "maxCount": 1,
        "option": [{"type": "Skill", "cardId": 77, "serial": 1001}],
    }
    renumbered = copy.deepcopy(first)
    renumbered["current"]["players"][0]["bench"][0]["serial"] = 9001
    renumbered["select"]["option"][0]["serial"] = 9001

    first_key = complete_semantic_option_key(first, [0])
    renumbered_key = complete_semantic_option_key(renumbered, [0])
    assert canonical_public_observation_hash(first) == canonical_public_observation_hash(renumbered)
    assert canonical_public_current_hash(first["current"], actor=0) == canonical_public_current_hash(
        renumbered["current"], actor=0
    )
    assert first_key == renumbered_key
    assert b"serial" not in canonical_json_bytes(first_key)

    first_record = build_stage_option_records(
        first,
        [[0]],
        token_builder=_token_builder,
    )[0]
    renumbered_record = build_stage_option_records(
        renumbered,
        [[0]],
        token_builder=_token_builder,
    )[0]
    assert first_record["canonical_public_observation_hash"] == renumbered_record[
        "canonical_public_observation_hash"
    ]
    assert first_record["complete_semantic_option_key_sha256"] == renumbered_record[
        "complete_semantic_option_key_sha256"
    ]
    assert first_record["complete_raw_option_payload_audit_only"] is True
    assert first_record["complete_raw_option_payload"][0]["serial"] == 1001
    assert renumbered_record["complete_raw_option_payload"][0]["serial"] == 9001


def _complete_transition(*, successor: str, outcome: int) -> dict:
    return {
        "schema": R298_ENGINE_EVIDENCE_SCHEMA,
        "source_kind": "pinned_simulator_public_deterministic_branch",
        "pinned_simulator_binary_sha256": CANONICAL_R236_LIBCG_SHA256,
        "source_receipt_sha256": _sha("e"),
        "transition_scope": "public_deterministic_branch",
        "hidden_branching_used": False,
        "successor_public_observation_hash": successor,
        "public_event_chain": {"events": [outcome]},
        "public_outcome_distribution": {"outcome": outcome},
    }


def test_equal_current_token_with_distinct_number_semantics_fails_closed() -> None:
    observation = _observation()
    candidates = [[0], [1]]
    transitions = {
        action_key_sha256([0]): _complete_transition(successor=_sha("1"), outcome=1),
        action_key_sha256([1]): _complete_transition(successor=_sha("2"), outcome=2),
    }
    records = build_stage_option_records(
        observation,
        candidates,
        selected_candidate_index=0,
        transition_by_action=transitions,
        token_builder=_token_builder,
    )

    assert records[0]["current_feature_token_hash"] == records[1]["current_feature_token_hash"]
    assert records[0]["new_complete_semantic_option_key_sha256"] != records[1][
        "new_complete_semantic_option_key_sha256"
    ]
    assert records[0]["complete_raw_option_payload"][0]["number"] == 4
    assert records[1]["complete_raw_option_payload"][0]["number"] == 5
    assert records[0]["complete_raw_option_payload_audit_only"] is True

    report = analyze_collision_records(records, decision_count=1)
    assert report["status"] == "failed_actionable_public_semantic_collision"
    assert report["actionable_failure_group_count"] == 1
    assert report["selected_action_in_divergent_collision_record_count"] == 1


def test_forced_empty_selection_emits_one_explicit_stop_stage() -> None:
    observation = _observation()
    observation["select"] = {
        "context": "Main",
        "type": "End",
        "minCount": 0,
        "maxCount": 0,
        "option": [],
    }
    current = copy.deepcopy(observation["current"])
    payload = {
        "id": "forced-empty",
        "steps": [
            [
                {
                    "visualize": [
                        {"action": [[], []]},
                        {"current": current, "obs": observation, "action": [[], []]},
                    ]
                },
                {},
            ],
            [{"observation": observation}, {}],
            [{"action": []}, {"action": []}],
        ],
    }
    coverage = recorded_episode_frame_coverage(payload)
    stages = stage_descriptors_from_recorded_episode(payload, source={"fixture": True})
    assert coverage == {
        "actor_visible_selection_frame_count": 1,
        "forced_selection_frame_count": 1,
    }
    assert len(stages) == 1
    assert stages[0]["candidates"] == [[]]
    assert stages[0]["selected_candidate_index"] == 0


def test_stage_source_marks_any_alakazam_list_variant_by_acting_deck() -> None:
    observation = _observation()
    observation["select"] = {
        "context": "Main",
        "type": "End",
        "minCount": 0,
        "maxCount": 0,
        "option": [],
    }
    current = copy.deepcopy(observation["current"])
    alakazam_variant = [743] + [1] * 59
    non_alakazam = [2] * 60
    payload = {
        "id": "alakazam-list-variant",
        "steps": [
            [
                {
                    "visualize": [
                        {"action": [alakazam_variant, non_alakazam]},
                        {"current": current, "obs": observation, "action": [[], []]},
                    ]
                },
                {},
            ],
            [{"observation": observation}, {}],
            [{"action": []}, {"action": []}],
        ],
    }
    stages = stage_descriptors_from_recorded_episode(payload, source={"fixture": True})
    assert len(stages) == 1
    assert stages[0]["source"]["acting_deck_is_alakazam_archetype"] is True
    # Revision 6 materialization is deliberately literal and acting-seat
    # scoped.  Keep the historical archetype flag only as audit provenance;
    # eligibility must not infer an exact pilot list or classifier label.
    assert stages[0]["source"]["acting_seat_setup_deck_contains_card_743"] is True
    assert stages[0]["source"]["acting_deck_multiset_sha256"].startswith("sha256:")


def test_non_selection_trace_row_with_null_select_is_not_a_policy_stage() -> None:
    current = _observation()["current"]
    observation = {"current": copy.deepcopy(current), "select": None}
    payload = {
        "id": "no-select-row",
        "steps": [
            [
                {
                    "visualize": [
                        {"action": [[], []]},
                        {"current": current, "obs": observation, "action": [[], []]},
                    ]
                },
                {},
            ],
            [{"observation": observation}, {}],
            [{"action": []}, {"action": []}],
        ],
    }
    assert recorded_episode_frame_coverage(payload) == {
        "actor_visible_selection_frame_count": 0,
        "forced_selection_frame_count": 0,
    }
    assert stage_descriptors_from_recorded_episode(payload, source={"fixture": True}) == []


def test_visible_selection_without_post_action_current_fails_closed() -> None:
    """A malformed trace must not lower frame coverage by skipping a decision."""

    observation = _observation()
    payload = {
        "id": "selection-missing-current",
        "steps": [
            [
                {
                    "visualize": [
                        {"action": [[], []]},
                        {"current": None, "obs": observation, "action": [[1], []]},
                    ]
                },
                {},
            ]
        ],
    }

    with pytest.raises(CollisionCensusError, match="selection lacks post-action current"):
        recorded_episode_frame_coverage(payload)
    with pytest.raises(CollisionCensusError, match="selection lacks post-action current"):
        stage_descriptors_from_recorded_episode(payload, source={"fixture": True})


def test_trace_alignment_uses_pre_action_masked_observation_not_post_action_actor() -> None:
    """Setup/automatic handoffs can change ``current.yourIndex`` after action.

    The raw actor observation contains only the legacy private search token in
    addition to the visualizer's masked input.  It must join at the same trace
    index and select seat zero even though the visualizer's post-action state
    now names seat one.
    """

    visual_observation = _observation()
    raw_observation = copy.deepcopy(visual_observation)
    raw_observation["search_begin_input"] = {"private_search_state": [1, 2, 3]}
    post_action_current = copy.deepcopy(visual_observation["current"])
    post_action_current["yourIndex"] = 1
    payload = {
        "id": "post-action-handoff",
        "steps": [
            [
                {
                    "visualize": [
                        {"action": [[], []]},
                        {
                            "current": post_action_current,
                            "obs": visual_observation,
                            "action": [[1], []],
                        },
                    ]
                },
                {},
            ],
            [{"observation": raw_observation}, {}],
            [{"action": [1]}, {"action": []}],
        ],
    }

    assert recorded_episode_frame_coverage(payload) == {
        "actor_visible_selection_frame_count": 1,
        "forced_selection_frame_count": 0,
    }
    stages = stage_descriptors_from_recorded_episode(payload, source={"fixture": True})
    assert len(stages) == 1
    assert stages[0]["source"]["acting_seat"] == 0
    assert stages[0]["selected_candidate_index"] == 1
    assert "search_begin_input" not in stages[0]["observation"]


def test_raw_schema_inventory_covers_every_outer_including_setup_null_current() -> None:
    """Phase A inventories raw outer schema without retaining raw values."""

    visible = _observation()
    raw_visible = copy.deepcopy(visible)
    raw_visible["search_begin_input"] = {"private_search_state": [1, 2, 3]}
    setup_outer = {
        "current": None,
        "select": None,
        "search_begin_input": {"private_setup": [4, 5]},
    }
    payload = {
        "id": "raw-schema-all-outers",
        "steps": [
            [
                {"observation": copy.deepcopy(setup_outer)},
                {"observation": copy.deepcopy(setup_outer)},
            ],
            [
                {"observation": raw_visible},
                {"observation": copy.deepcopy(setup_outer)},
            ],
        ],
    }

    inventory = inventory_raw_observations(
        raw_observations_from_recorded_episode(payload)
    )
    validate_phase_a_inventory(inventory)

    assert inventory["raw_observation_count"] == 4
    assert inventory["actor_context_observation_count"] == 1
    assert inventory["no_actor_context_observation_count"] == 3
    assert inventory["rejected_observation_count"] == 0
    assert inventory["field_classification_occurrences"]["intentionally_hidden_information"] > 0
    hand_id_field = next(
        field
        for field in inventory["fields"]
        if field["path"] == "current.players.[].hand.[].id"
    )
    # One raw actor observation contains both hands.  The normalized schema
    # path is shared, while the transient concrete seat binding classifies the
    # actor's visible hand independently from the opponent's hidden hand.
    assert hand_id_field["classification_occurrences"] == {
        "direct_public_libcg_observation_or_option": 1,
        "intentionally_hidden_information": 1,
    }
    assert all("[1, 2, 3]" not in str(row) for row in inventory["fields"])


def test_streaming_phase_a_accumulator_reconciles_raw_and_masked_scopes() -> None:
    raw_accumulator = runner._PhaseAInventoryAccumulator(
        inventory_scope="all_raw_replay_observations"
    )
    actor_accumulator = runner._PhaseAInventoryAccumulator(
        inventory_scope="all_masked_actor_visible_selection_observations"
    )
    raw_accumulator.add(
        {
            "current": None,
            "select": None,
            "search_begin_input": {"setup_private": [1]},
        }
    )
    raw_accumulator.add(_observation())
    actor_accumulator.add(_observation())

    raw_inventory = raw_accumulator.final()
    actor_inventory = actor_accumulator.final()
    validate_phase_a_inventory(raw_inventory)
    validate_phase_a_inventory(actor_inventory)
    assert raw_inventory["raw_observation_count"] == 2
    assert raw_inventory["actor_context_observation_count"] == 1
    assert raw_inventory["no_actor_context_observation_count"] == 1
    assert actor_inventory["raw_observation_count"] == 1
    assert actor_inventory["actor_context_observation_count"] == 1


def _raw_days() -> list[str]:
    return [f"2026-07-{day:02d}" for day in range(13, 32)] + [
        f"2026-08-{day:02d}" for day in range(1, 12)
    ]


def _strict_raw_manifest() -> dict:
    archives: list[dict] = []
    member_counts: dict[str, int] = {}
    byte_counts: dict[str, int] = {}
    for index, date in enumerate(_raw_days(), start=1):
        digest = "sha256:" + f"{index:064x}"
        archives.append(
            {
                "date": date,
                "dataset_slug": f"source-{date}",
                "path": f"/immutable/{date}.zip",
                "sha256": digest,
                "bytes": index,
                "validated": True,
                "validated_episode_count": 1,
                "index_episode_count": 1,
                "source_discrepancy": None,
                "source_receipt_sha256s": [
                    RAW_CORPUS_SOURCE_RECEIPT_SHA256S[0]
                    if date <= "2026-07-23"
                    else RAW_CORPUS_SOURCE_RECEIPT_SHA256S[1]
                ],
            }
        )
        member_counts[digest] = 1
        byte_counts[digest] = index
    return build_raw_corpus_manifest(
        archives,
        archive_member_counts=member_counts,
        archive_bytes_actual=byte_counts,
        source_manifest_provenance=[
            {
                "path": "/immutable/source-first.json",
                "sha256": RAW_CORPUS_SOURCE_RECEIPT_SHA256S[0],
                "schema": "source/v1",
                "status": "ready",
            },
            {
                "path": "/immutable/source-second.json",
                "sha256": RAW_CORPUS_SOURCE_RECEIPT_SHA256S[1],
                "schema": "source/v1",
                "status": "ready",
            },
        ],
        episode_deduplication={
            "episode_identity_algorithm": "payload.id_plus_canonical_content_sha256",
            "unique_episode_identity_count": 30,
            "unique_episode_id_count": 30,
            "duplicate_episode_identity_count": 0,
            "duplicate_episode_id_with_distinct_content_count": 0,
            "excluded_duplicate_mapping": [],
            "raw_zip_member_count_observed": 30,
            "episode_identity_inventory_sha256": _sha("d"),
        },
    )


def test_raw_manifest_requires_per_day_source_and_full_episode_dedup() -> None:
    manifest = _strict_raw_manifest()
    validate_raw_corpus_manifest(manifest)

    broken = copy.deepcopy(manifest)
    broken["source_receipt_day_coverage"][0]["source_receipt_sha256s"] = []
    with pytest.raises(CollisionCensusError, match="coverage"):
        validate_raw_corpus_manifest(broken)

    broken = copy.deepcopy(manifest)
    broken["episode_deduplication"]["raw_zip_member_count_observed"] = 29
    with pytest.raises(CollisionCensusError, match="deduplication"):
        validate_raw_corpus_manifest(broken)


def _frozen_schema_manifest() -> dict:
    return {
        "schema": R298_FROZEN_SCHEMA_MANIFEST_SCHEMA,
        "status": "frozen_zero_inert_before_refeaturization",
        "owner_revision": R298_OWNER_REVISION,
        "goal_revision": REVISION_5_GOAL_REVISION,
        "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
        "owner_goal_sha256": OWNER_GOAL_SHA256,
        "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
        "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
        "revision_5_predecessor_classification": revision_5_predecessor_classification(),
        "mechanics_attachment_sha256": MECHANICS_ATTACHMENT_SHA256,
        "feature_schema_sha256": _sha("a"),
        "target_schema_sha256": _sha("b"),
        "checklist_provenance_schema_sha256": _sha("c"),
        "new_branch_inventory": ["public_rule_adapter", "rule_heads", "checklist_provenance"],
        "runtime_wired": False,
        "default_zero_and_inert_attestation": True,
        "layer_off_bit_identical_baseline_logits": True,
        "layer_off_identical_legal_choice": True,
    }


def test_frozen_schema_gate_requires_runtime_unwired_and_exact_parity() -> None:
    manifest = _frozen_schema_manifest()
    bypass = {
        "schema": R298_ZERO_BYPASS_RECEIPT_SCHEMA,
        "status": "passed_exact_baseline_logits",
        "owner_revision": R298_OWNER_REVISION,
        "goal_revision": REVISION_5_GOAL_REVISION,
        "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
        "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
        "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
        "revision_5_predecessor_classification": revision_5_predecessor_classification(),
        "frozen_schema_manifest_sha256": canonical_sha256(manifest),
        "all_new_routes_exact_zero_or_bypassed": True,
        "checklist_provenance_schema_frozen": True,
        "layer_off_bit_identical_baseline_logits": True,
        "layer_off_identical_legal_choice": True,
        "runtime_wired": False,
    }
    assert validate_frozen_schema_gate(manifest, bypass) == (
        canonical_sha256(manifest),
        canonical_sha256(bypass),
    )

    bypass["runtime_wired"] = True
    with pytest.raises(CollisionCensusError, match="runtime wiring"):
        validate_frozen_schema_gate(manifest, bypass)


def test_revision_5_predecessor_classifier_cannot_promote_revision_4_receipts() -> None:
    predecessor = revision_5_predecessor_classification(
        [
            {
                "receipt_sha256": _sha("4"),
                "schema": "poke_bot.fixture_revision_4_receipt/v1",
                "classification": "immutable_predecessor_evidence_not_revision_5_authority",
                "checksum_identical_bytes_reused": True,
                "satisfies_revision_5_schema_freeze": False,
            }
        ]
    )
    assert predecessor["predecessor_gateway_sha256"] == REVISION_4_GATEWAY_SHA256
    assert predecessor["predecessor_contract_sha256"] == REVISION_4_CONTRACT_SHA256
    assert validate_revision_5_predecessor_classification(predecessor) == predecessor

    promoted = copy.deepcopy(predecessor)
    promoted["consumed_predecessor_receipts"][0][
        "satisfies_revision_5_schema_freeze"
    ] = True
    with pytest.raises(CollisionCensusError, match="cannot satisfy"):
        validate_revision_5_predecessor_classification(promoted)


def test_typed_revision_5_census_validation_binds_fresh_authority_only() -> None:
    raw_manifest = _strict_raw_manifest()
    raw_receipt = make_raw_corpus_receipt(
        raw_manifest,
        run_identity_sha256=_sha("f"),
        resource_observation={"fixture": True},
    )
    frozen = _frozen_schema_manifest()
    historical = revision_5_predecessor_classification(
        [
            {
                "receipt_sha256": _sha("4"),
                "schema": "poke_bot.fixture_revision_4_freeze/v1",
                "classification": "immutable_predecessor_evidence_not_revision_5_authority",
                "checksum_identical_bytes_reused": True,
                "satisfies_revision_5_schema_freeze": False,
            }
        ]
    )
    frozen["revision_5_predecessor_classification"] = historical
    bypass = {
        "schema": R298_ZERO_BYPASS_RECEIPT_SCHEMA,
        "status": "passed_exact_baseline_logits",
        "owner_revision": R298_OWNER_REVISION,
        "goal_revision": REVISION_5_GOAL_REVISION,
        "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
        "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
        "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
        "revision_5_predecessor_classification": historical,
        "frozen_schema_manifest_sha256": canonical_sha256(frozen),
        "all_new_routes_exact_zero_or_bypassed": True,
        "checklist_provenance_schema_frozen": True,
        "layer_off_bit_identical_baseline_logits": True,
        "layer_off_identical_legal_choice": True,
        "runtime_wired": False,
    }
    raw_inventory = inventory_raw_observations([_observation()])
    actor_selection_inventory = inventory_raw_observations(
        [_observation()],
        inventory_scope="all_masked_actor_visible_selection_observations",
    )
    frame_coverage = {
        "actor_visible_selection_frame_count": 1,
        "processed_actor_visible_selection_frame_count": 1,
        "forced_selection_frame_count": 0,
        "all_actor_visible_and_forced_frames_included": True,
    }
    re_featurization = {
        "raw_outer_observation_count": raw_inventory["raw_observation_count"],
        "raw_schema_inventory_sha256": canonical_sha256(raw_inventory),
        "raw_schema_inventory_scope": "all_raw_replay_observations",
        "raw_schema_inventory_rejected_observation_count": 0,
        "actor_visible_selection_inventory_sha256": canonical_sha256(
            actor_selection_inventory
        ),
        "actor_visible_selection_inventory_scope": "all_masked_actor_visible_selection_observations",
        "actor_visible_selection_inventory_observation_count": 1,
        "raw_observation_values_persisted": False,
    }
    census = make_receipt(
        report=analyze_collision_records([], decision_count=1),
        inventory=raw_inventory,
        raw_expert_corpus_manifest=raw_manifest,
        raw_expert_corpus_receipt=raw_receipt,
        frozen_schema_manifest=frozen,
        zero_bypass_receipt=bypass,
        current_token_abi_source_sha256=(
            "sha256:ce0eb08ca74e337fe6ee1eaeb678eb42bd7f413bbfab04f8a228c3cfd3ce3db5"
        ),
        run_identity_sha256=_sha("c"),
        raw_episode_count=30,
        public_matchup_distribution={},
        acting_deck_distribution={},
        frame_coverage=frame_coverage,
        re_featurization=re_featurization,
    )
    validation = make_revision_5_census_validation_receipt(
        census_receipt=census,
        raw_expert_corpus_manifest=raw_manifest,
        raw_expert_corpus_receipt=raw_receipt,
        frozen_schema_manifest=frozen,
        zero_bypass_receipt=bypass,
    )
    assert validation["schema"] == R298_REV5_CENSUS_VALIDATION_RECEIPT_SCHEMA
    assert validate_revision_5_census_validation_receipt(validation) == validation
    assert validation["revision_5_predecessor_classification"]["consumed_predecessor_receipts"] == [
        historical["consumed_predecessor_receipts"][0]
    ]

    incomplete_raw_inventory = copy.deepcopy(census)
    incomplete_raw_inventory["phase_a_raw_observation_inventory"][
        "rejected_observation_count"
    ] = 1
    with pytest.raises(CollisionCensusError, match="raw Phase A inventory"):
        make_revision_5_census_validation_receipt(
            census_receipt=incomplete_raw_inventory,
            raw_expert_corpus_manifest=raw_manifest,
            raw_expert_corpus_receipt=raw_receipt,
            frozen_schema_manifest=frozen,
            zero_bypass_receipt=bypass,
        )

    incomplete_frame_coverage = copy.deepcopy(census)
    incomplete_frame_coverage["frame_coverage"][
        "all_actor_visible_and_forced_frames_included"
    ] = False
    with pytest.raises(CollisionCensusError, match="actor-visible/forced frame coverage"):
        make_revision_5_census_validation_receipt(
            census_receipt=incomplete_frame_coverage,
            raw_expert_corpus_manifest=raw_manifest,
            raw_expert_corpus_receipt=raw_receipt,
            frozen_schema_manifest=frozen,
            zero_bypass_receipt=bypass,
        )

    stale = copy.deepcopy(validation)
    stale["rule_derivative_contract_sha256"] = REVISION_4_CONTRACT_SHA256
    with pytest.raises(CollisionCensusError, match="authority drifted"):
        validate_revision_5_census_validation_receipt(stale)


def test_refeatured_records_are_content_addressed_bounded_shards(tmp_path) -> None:
    writer = runner._ContentAddressedShardWriter(
        tmp_path / "records",
        bucket_count=2,
        raw_manifest_sha256=_sha("1"),
        frozen_schema_manifest_sha256=_sha("2"),
        zero_bypass_receipt_sha256=_sha("3"),
    )
    writer.write(
        {
            "canonical_public_observation_hash": _sha("a"),
            "current_feature_token_hash": _sha("b"),
            "schema": "fixture",
        }
    )
    writer.close()
    manifest = writer.manifest()
    assert manifest["shard_count"] == 1
    shard = writer.shard_paths()[0]
    assert shard.name == f"sha256-{runner.sha256_file(shard)[7:]}.refeaturization-census.shard"
    assert shard.stat().st_size <= runner.MAX_TRANSFER_SHARD_BYTES
    assert runner._read_refeatured_shard(
        shard,
        raw_manifest_sha256=_sha("1"),
        frozen_schema_manifest_sha256=_sha("2"),
        zero_bypass_receipt_sha256=_sha("3"),
    )[0]["schema"] == "fixture"


def test_shard_routing_uses_full_hash_pair_not_short_digest_prefix(tmp_path) -> None:
    writer = runner._ContentAddressedShardWriter(
        tmp_path / "records",
        bucket_count=4096,
        raw_manifest_sha256=_sha("1"),
        frozen_schema_manifest_sha256=_sha("2"),
        zero_bypass_receipt_sha256=_sha("3"),
    )
    # These differ only beyond the four hex characters the old router used.
    left = {"canonical_public_observation_hash": "sha256:aaaa" + "0" * 60, "current_feature_token_hash": "sha256:bbbb" + "0" * 60}
    right = {"canonical_public_observation_hash": "sha256:aaaa" + "1" * 60, "current_feature_token_hash": "sha256:bbbb" + "1" * 60}
    assert writer._bucket(left) != writer._bucket(right)
    writer.abort()


def test_private_24_day_lanes_merge_only_through_parent_writer(tmp_path) -> None:
    """Workers may spool records, but only the parent can publish a shard."""

    spool_root = tmp_path / "private-lanes"
    spool_root.mkdir()
    lane_results = []
    for lane_index in range(runner.PHASE_A_VALIDATION_WORKERS):
        spool = runner._PrivateDayLaneSpool(
            spool_root / f"lane-{lane_index:02d}",
            lane_index=lane_index,
            bucket_count=2,
        )
        if lane_index == 0:
            spool.write(
                {
                    "canonical_public_observation_hash": _sha("a"),
                    "current_feature_token_hash": _sha("b"),
                    "schema": "fixture",
                }
            )
            spool.write(
                {
                    "canonical_public_observation_hash": _sha("a"),
                    "current_feature_token_hash": _sha("b"),
                    "schema": "fixture-second-record-same-bucket",
                }
            )
        lane_results.append(
            {"lane_index": lane_index, "private_spool_shards": spool.close()}
        )
    writer = runner._ContentAddressedShardWriter(
        tmp_path / "records",
        bucket_count=2,
        raw_manifest_sha256=_sha("1"),
        frozen_schema_manifest_sha256=_sha("2"),
        zero_bypass_receipt_sha256=_sha("3"),
    )
    assert runner._merge_private_day_lane_spools(
        lane_results,
        spool_root=spool_root,
        writer=writer,
    ) == 2
    writer.close()
    assert writer.manifest()["record_count"] == 2


def test_revision_7_parallel_plan_matches_exact_census_day_lanes(tmp_path) -> None:
    """The authoritative 24-lane schedule must match the consumed 30 days."""

    from poke_bot.alakazam_rule_derivative_predecessor_compat_rev7 import (
        revision_7_parallel_execution_plan,
    )

    dates = runner._expected_dates()
    archives = [
        ({"date": date}, Path(tmp_path / f"{date}.zip"))
        for date in dates
    ]
    plan = revision_7_parallel_execution_plan(workers=24)
    runner._validate_revision_7_parallel_day_plan(plan, archives)

    wrong = copy.deepcopy(plan)
    wrong["day_lanes"][0] = list(reversed(wrong["day_lanes"][0]))
    with pytest.raises(runner.RunnerError, match="day plan drifted"):
        runner._validate_revision_7_parallel_day_plan(wrong, archives)


def test_collision_audit_shards_are_separate_from_card_743_materialized_rows(tmp_path) -> None:
    audit_writer = runner._ContentAddressedShardWriter(
        tmp_path / "collision-audit",
        bucket_count=2,
        raw_manifest_sha256=_sha("1"),
        frozen_schema_manifest_sha256=_sha("2"),
        zero_bypass_receipt_sha256=_sha("3"),
        record_scope=runner.RECORD_SCOPE_COLLISION_AUDIT_ALL_ACTOR_VISIBLE,
    )
    audit_writer.write(
        {
            "canonical_public_observation_hash": _sha("a"),
            "current_feature_token_hash": _sha("b"),
            "schema": "all-seat-audit-fixture",
        }
    )
    audit_writer.close()
    shard = audit_writer.shard_paths()[0]
    assert audit_writer.manifest()["record_scope"] == runner.RECORD_SCOPE_COLLISION_AUDIT_ALL_ACTOR_VISIBLE
    assert runner._read_refeatured_shard(
        shard,
        raw_manifest_sha256=_sha("1"),
        frozen_schema_manifest_sha256=_sha("2"),
        zero_bypass_receipt_sha256=_sha("3"),
        record_scope=runner.RECORD_SCOPE_COLLISION_AUDIT_ALL_ACTOR_VISIBLE,
    )[0]["schema"] == "all-seat-audit-fixture"
    with pytest.raises(runner.RunnerError, match="record_scope"):
        runner._read_refeatured_shard(
            shard,
            raw_manifest_sha256=_sha("1"),
            frozen_schema_manifest_sha256=_sha("2"),
            zero_bypass_receipt_sha256=_sha("3"),
        )


def test_episode_iterator_excludes_daily_manifest_csv(tmp_path) -> None:
    archive_path = tmp_path / "day.zip"
    with zipfile.ZipFile(archive_path, "w") as bundle:
        bundle.writestr("episode.json", json.dumps({"id": "fixture"}))
        bundle.writestr("manifest.csv", "id\nfixture\n")
    archive = {
        "date": "2026-07-13",
        "sha256": _sha("f"),
        "bytes": archive_path.stat().st_size,
        "zip_json_member_count": 1,
    }
    rows = list(
        runner._iter_episode_payloads(
            [(archive, archive_path)],
            day_shard_index=0,
            day_shard_count=1,
            max_episodes=None,
            telemetry=runner._ResourceTelemetry(probe_gpu=False),
        )
    )
    assert [(member, payload["id"]) for _archive, member, payload in rows] == [
        ("episode.json", "fixture")
    ]


def test_execute_rejects_non_elmo_host_before_config_or_output(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "workstation")

    def must_not_read_config(_path):
        raise AssertionError("host gate must run before config/output/archive work")

    monkeypatch.setattr(runner, "_load_config", must_not_read_config)
    assert runner.main(["--execute", "--output-root", str(tmp_path)]) == 2


def test_raw_receipt_consumption_requires_matching_elmo_execution_identity(
    monkeypatch,
) -> None:
    manifest = _strict_raw_manifest()
    receipt = {
        "schema": runner.R298_RAW_CORPUS_RECEIPT_SCHEMA,
        "status": "passed",
        "owner_revision": R298_OWNER_REVISION,
        "goal_revision": REVISION_5_GOAL_REVISION,
        "root_handoff_revision": REVISION_5_ROOT_HANDOFF_REVISION,
        "raw_expert_corpus_manifest_sha256": canonical_sha256(manifest),
        "source_manifest_provenance_sha256": canonical_sha256(
            manifest["source_manifest_provenance"]
        ),
        "source_receipt_day_coverage_sha256": canonical_sha256(
            manifest["source_receipt_day_coverage"]
        ),
        "episode_deduplication_sha256": canonical_sha256(
            manifest["episode_deduplication"]
        ),
        "completed_raw_zip_member_count": manifest["total_raw_zip_json_members"],
        "completed_validated_episode_count": manifest["total_validated_episodes"],
        "source_disjointness": {
            "archive_date_source_sha256_unique": True,
            "episode_identity_unique": True,
            "episode_id_content_unique": True,
            "source_window_blending_permitted": False,
            "training_eligible": False,
        },
        "owner_goal_sha256": OWNER_GOAL_SHA256,
        "rule_derivative_contract_sha256": RULE_DERIVATIVE_CONTRACT_SHA256,
        "rule_derivative_gateway_sha256": RULE_DERIVATIVE_GATEWAY_SHA256,
        "revision_5_predecessor_classification": revision_5_predecessor_classification(),
        "mechanics_attachment_sha256": MECHANICS_ATTACHMENT_SHA256,
        "recollection_authorized": False,
    }
    identity = {
        "execution_host_role": "elmo",
        "canonical_execution_hostname": "truenas",
        "execution_hostname": "truenas",
        "execution_fqdn": "truenas",
        "host_verification": "exact_socket_hostname_and_systemd_detect_virt_non_container",
        "container_execution_permitted": False,
    }
    receipt["execution_identity"] = identity
    receipt["resource_observation"] = {"execution_identity": copy.deepcopy(identity)}
    # The source receipt re-open is separately covered by the materializer;
    # isolate the new host-binding consumption rule here.
    monkeypatch.setattr(runner, "_verify_manifest_source_receipts", lambda _manifest: None)
    runner._validate_raw_corpus_binding(manifest, receipt)

    receipt["resource_observation"]["execution_identity"]["execution_hostname"] = "other"
    with pytest.raises(runner.RunnerError, match="execution identity"):
        runner._validate_raw_corpus_binding(manifest, receipt)


def test_execution_identity_requires_exact_truenas_elmo_host(monkeypatch) -> None:
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "truenas")
    monkeypatch.setattr(runner.socket, "getfqdn", lambda: "truenas")
    monkeypatch.setattr(runner.Path, "exists", lambda _path: False)

    class _Probe:
        returncode = 1
        stdout = "none\n"

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: _Probe())
    identity = runner._verified_elmo_execution_identity()
    assert identity["execution_host_role"] == "elmo"
    assert identity["execution_hostname"] == "truenas"
    assert identity["container_execution_permitted"] is False


def test_post_census_revision_7_bridge_is_read_only_and_has_no_preflight(tmp_path, monkeypatch, capsys) -> None:
    """An already-running r5 census is consumed, never retroactively changed."""

    inputs = {
        "raw_manifest": tmp_path / "raw-manifest.json",
        "raw_receipt": tmp_path / "raw-receipt.json",
        "schema": tmp_path / "schema.json",
        "bypass": tmp_path / "bypass.json",
        "bridge": tmp_path / "revision-5-bridge.json",
        "census": tmp_path / "receipt.json",
    }
    for name, path in inputs.items():
        path.write_text(json.dumps({"fixture": name}, sort_keys=True) + "\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_completion_validator(**kwargs):
        captured.update(kwargs)
        return {
            "materialization_preflight_claimed_or_required": False,
            "historical_receipts_retagged_or_rewritten": False,
            "training_runtime_service_transfer_or_activation_authority": False,
            "revision_5_collision_census_canonical_sha256": _sha("a"),
            "revision_5_collision_census_physical_file_sha256": _sha("b"),
        }

    monkeypatch.setattr(
        runner,
        "validate_revision_5_census_completion_under_revision_7",
        fake_completion_validator,
    )
    identity = {
        "execution_host_role": "elmo",
        "canonical_execution_hostname": "truenas",
        "execution_hostname": "truenas",
        "execution_fqdn": "truenas",
        "host_verification": "exact_socket_hostname_and_systemd_detect_virt_non_container",
        "container_execution_permitted": False,
    }
    args = SimpleNamespace(
        raw_corpus_manifest=inputs["raw_manifest"],
        raw_corpus_receipt=inputs["raw_receipt"],
        frozen_schema_manifest=inputs["schema"],
        zero_bypass_receipt=inputs["bypass"],
        revision_5_census_validation_receipt=inputs["bridge"],
        collision_census_receipt=inputs["census"],
    )
    before = sorted(path.name for path in tmp_path.iterdir())
    assert runner._run_census_completion_validation(args, identity) == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == before
    assert set(captured) == {
        "raw_manifest",
        "raw_receipt",
        "schema_manifest",
        "zero_bypass_receipt",
        "census_validation_receipt",
        "collision_census_receipt_path",
    }
    assert "materialization_preflight" not in captured
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "validated_read_only_revision_7_census_completion"
    assert result["completion"]["revision_5_collision_census_physical_file_sha256"] == _sha("b")
