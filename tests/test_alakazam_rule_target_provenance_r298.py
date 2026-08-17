"""Focused integrity tests for the isolated r298 target compiler."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from poke_bot.alakazam_simulator_rule_targets_r298 import (
    R298_CANONICAL_CONTRACT_PATH,
    R298_CANONICAL_CONTRACT_SHA256,
    R298_CANONICAL_GOAL_SHA256,
    R298_CANONICAL_GOAL_REVISION,
    R298_CANONICAL_SIMULATOR,
    R298_PREDECESSOR_CONTRACT_SHA256,
    R298_PREDECESSOR_GOAL_SHA256,
    R298_PRODUCTION_TYPED_SOURCE_SHA256,
    R298_PRIVILEGED_BELIEF_TARGET_SCHEMA,
    R298_PRIVILEGED_BELIEF_TARGET_SCHEMA_VERSION,
    R298_PROMPT_CHAIN_SCHEMA,
    R298_PROMPT_CHAIN_SCHEMA_VERSION,
    R298_RULE_TARGET_SCHEMA,
    R298_RULE_TARGET_SCHEMA_DIGEST,
    R298_R5_DERIVATIVE_LINEAGE_ID,
    R298_R5_HANDOFF_ACTIVATION_RECEIPT_SCHEMA,
    R298_R5_SCHEMA_FREEZE_RECEIPT_SCHEMA,
    R298_R5_TRAJECTORY_LEDGER_STATUS,
    R298_R5_TRAJECTORY_TRAINING_AUTHORIZATION_STATUS,
    R298_R5_TRAINING_HOST,
    R298_ROOT_OWNER_REVISION,
    R298_SEALED_TRAJECTORY_VALIDATOR_KIND,
    R298_TRAJECTORY_LEDGER_ENTRY_SCHEMA,
    R298_TRAJECTORY_LEDGER_SCHEMA,
    R298_TRAJECTORY_LEDGER_SCHEMA_VERSION,
    R298_TRAJECTORY_TRAINING_AUTHORIZATION_SCHEMA,
    R298_TRAJECTORY_TRAINING_AUTHORIZATION_SCHEMA_VERSION,
    SimulatorRuleTargetError,
    compile_simulator_rule_targets,
    load_sealed_selected_action_trajectory_validator,
    public_observation_fingerprint,
    public_target_fingerprint,
    rule_head_target_vectors,
)
from poke_bot.alakazam_simulator_rule_targets_r298 import (
    _ChainEvent,
    _NormalizedChain,
    _forced_promotion_count,
    _opponent_knockout_count,
    _prize_yield_target,
    _turn_resource_target,
)


def _card(card_id: int, serial: int) -> dict[str, Any]:
    return {
        "id": card_id,
        "serial": serial,
        "hp": 100,
        "maxHp": 100,
        "energyCards": [],
        "tools": [],
    }


def _player(*, active: list[dict[str, Any]], bench: list[dict[str, Any]], hand: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "active": active,
        "bench": bench,
        "hand": hand,
        "handCount": len(hand),
        "discard": [],
        "prize": [None] * 6,
        "prizeCount": 6,
        "deckCount": 30,
        "benchMax": 5,
    }


def _observation() -> dict[str, Any]:
    return {
        "current": {
            "yourIndex": 0,
            "turn": 3,
            "turnActionCount": 1,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "players": [
                _player(
                    active=[_card(10, 100)],
                    bench=[_card(77, 111)],
                    hand=[_card(30, 300)],
                ),
                _player(
                    active=[_card(11, 101)],
                    bench=[],
                    hand=[_card(901, 9001)],
                ),
            ],
            "stadium": [],
            "looking": [],
        },
        "select": {
            "context": "Main",
            "type": "Card",
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": "Skill", "cardId": 77, "serial": 111}],
        },
    }


def _chain(observation: dict[str, Any]) -> dict[str, Any]:
    after = copy.deepcopy(observation)
    after["current"]["turnActionCount"] = 2
    return {
        "schema": R298_PROMPT_CHAIN_SCHEMA,
        "version": R298_PROMPT_CHAIN_SCHEMA_VERSION,
        "simulator": copy.deepcopy(R298_CANONICAL_SIMULATOR),
        "source": "observed_selected_action_trajectory",
        "root_action": [0],
        "event_log_complete": True,
        "complete_to_next_strategic_decision": True,
        "events": [
            {
                "before": copy.deepcopy(observation),
                "after": after,
                "event_kind": "skill",
                "action": [0],
                "forced": False,
                "strategic_decision": True,
                "facts": {},
            }
        ],
    }


def _decision(observation: dict[str, Any]) -> dict[str, Any]:
    return {"observation": observation, "action": [0], "prompt_chain": _chain(observation)}


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _sealed_trajectory_validator_for_test(
    *,
    receipt: dict[str, Any],
    compiled: dict[str, Any],
) -> tuple[Any, tempfile.TemporaryDirectory[str]]:
    """Build a fully pinned temporary receipt chain for target ABI coverage."""

    workspace = tempfile.TemporaryDirectory()
    root = Path(workspace.name)
    raw_corpus_sha = receipt["corpus_receipt_sha256"]
    frozen_schema_sha = "sha256:" + "1" * 64
    zero_bypass_sha = "sha256:" + "4" * 64
    branch_support_sha = "sha256:" + "2" * 64
    training_gate_sha = "sha256:" + "3" * 64
    blackwell_preflight_sha = "sha256:" + "5" * 64
    rollback_plan_sha = "sha256:" + "6" * 64
    schema_freeze = {
        "schema": R298_R5_SCHEMA_FREEZE_RECEIPT_SCHEMA,
        "goal_contract_path": R298_CANONICAL_CONTRACT_PATH,
        "goal_contract_sha256": R298_CANONICAL_CONTRACT_SHA256,
        "goal_revision": R298_CANONICAL_GOAL_REVISION,
        "feature_schema_id": "fixture.feature.schema/v1",
        "feature_schema_sha256": "sha256:" + "7" * 64,
        "target_schema_id": R298_RULE_TARGET_SCHEMA,
        "target_schema_sha256": R298_RULE_TARGET_SCHEMA_DIGEST,
        "checklist_provenance_schema_id": "fixture.checklist.schema/v1",
        "checklist_provenance_schema_sha256": "sha256:" + "8" * 64,
        "public_catalog_manifest_sha256": "sha256:" + "9" * 64,
        "canonical_simulator_sha256": _sha256(R298_CANONICAL_SIMULATOR),
        "new_branch_inventory": ["fixture"],
        "census_supported_branch_inventory": ["fixture"],
        "unsupported_zero_inert_branch_inventory": [],
        "q3_bench_only": True,
        "q5_q6_trace_only_zero": True,
        "public_information_contract_passed": True,
        "zero_bypass_receipt_sha256": zero_bypass_sha,
        "layer_off_bit_identical_baseline_logits": True,
        "frozen_at_utc": "2026-08-12T00:00:00Z",
    }
    schema_freeze_path = root / "schema-freeze-receipt.json"
    _write_canonical_json(schema_freeze_path, schema_freeze)
    schema_freeze_sha = _file_sha256(schema_freeze_path)
    handoff = {
        "schema": R298_R5_HANDOFF_ACTIVATION_RECEIPT_SCHEMA,
        "goal_contract_path": R298_CANONICAL_CONTRACT_PATH,
        "goal_contract_sha256": R298_CANONICAL_CONTRACT_SHA256,
        "goal_revision": R298_CANONICAL_GOAL_REVISION,
        "root_owner_revision": R298_ROOT_OWNER_REVISION,
        "readiness_receipt_sha256": "sha256:" + "a" * 64,
        "activation_boundary_id": "fixture-clean-boundary",
        "old_lineage_id": "alakazam-new-list-direct-policy-r274",
        "old_trainer_service": "pokebot-alakazam-r274-rl.service",
        "old_submission_boundary_service": "pokebot-alakazam-r274-rl-submission-boundaries.service",
        "old_services_paused_via_systemd_user": True,
        "old_services_inactive_verified": True,
        "shared_kaggle_queue_service": "pokebot-kaggle-submission-queue.service",
        "shared_kaggle_queue_service_unchanged": True,
        "old_parent_checkpoint_sha256": "sha256:" + "b" * 64,
        "old_optimizer_state_sha256": "sha256:" + "c" * 64,
        "old_collection_manifest_sha256": "sha256:" + "d" * 64,
        "new_lineage_id": R298_R5_DERIVATIVE_LINEAGE_ID,
        "new_managed_bootstrap_service": "pokebot-alakazam-rule-derivative-g5-bootstrap.service",
        "new_run_root": "/fixture/r5",
        "new_runtime_registry_path": "/fixture/r5/registry.json",
        "new_runtime_registry_sha256": "sha256:" + "e" * 64,
        "staged_corpus_receipt_sha256": raw_corpus_sha,
        "staged_shards_training_eligible": True,
        "training_eligibility_activated_at_utc": "2026-08-12T00:00:00Z",
        "blackwell_preflight_receipt_sha256": blackwell_preflight_sha,
        "rollback_plan_receipt_sha256": rollback_plan_sha,
        "no_concurrent_r274_training_or_collection": True,
        "no_serving_selector_or_submission_activation": True,
        "activated_at_utc": "2026-08-12T00:00:00Z",
    }
    handoff_path = root / "handoff-activation-receipt.json"
    _write_canonical_json(handoff_path, handoff)
    handoff_sha = _file_sha256(handoff_path)
    entry = {
        "schema": R298_TRAJECTORY_LEDGER_ENTRY_SCHEMA,
        "source": "observed_selected_action_trajectory",
        "canonical_simulator_sha256": _sha256(R298_CANONICAL_SIMULATOR),
        "normalized_prompt_chain_sha256": compiled["prompt_chain"][
            "realized_target_chain_hash"
        ],
        "public_observation_sha256": compiled["public_observation_hash"],
        "selected_action_semantic_sha256": receipt[
            "selected_action_semantic_sha256"
        ],
        "trajectory_receipt_sha256": receipt["trajectory_receipt_sha256"],
        "corpus_receipt_sha256": raw_corpus_sha,
        "raw_frame_receipt_sha256": receipt["raw_frame_receipt_sha256"],
    }
    ledger = {
        "schema": R298_TRAJECTORY_LEDGER_SCHEMA,
        "version": R298_TRAJECTORY_LEDGER_SCHEMA_VERSION,
        "status": R298_R5_TRAJECTORY_LEDGER_STATUS,
        "goal_sha256": R298_CANONICAL_GOAL_SHA256,
        "contract_sha256": R298_CANONICAL_CONTRACT_SHA256,
        "goal_revision": R298_CANONICAL_GOAL_REVISION,
        "root_owner_revision": R298_ROOT_OWNER_REVISION,
        "production_typed_source_sha256": R298_PRODUCTION_TYPED_SOURCE_SHA256,
        "target_schema": R298_RULE_TARGET_SCHEMA,
        "target_schema_digest": R298_RULE_TARGET_SCHEMA_DIGEST,
        "raw_corpus_receipt_sha256": raw_corpus_sha,
        "frozen_schema_manifest_sha256": frozen_schema_sha,
        "schema_freeze_receipt_sha256": schema_freeze_sha,
        "complete_30_utc_days": True,
        "target_only": True,
        "may_drive_runtime": False,
        "runtime_wired": False,
        "training_authorization": False,
        "revision_4_predecessor_evidence_only": True,
        "blind_revision_4_substitution_allowed": False,
        "entries": [entry],
        "entries_sha256": _sha256([entry]),
    }
    ledger["ledger_payload_sha256"] = _sha256(ledger)
    ledger_path = root / "trajectory-ledger.json"
    _write_canonical_json(ledger_path, ledger)
    authorization = {
        "schema": R298_TRAJECTORY_TRAINING_AUTHORIZATION_SCHEMA,
        "version": R298_TRAJECTORY_TRAINING_AUTHORIZATION_SCHEMA_VERSION,
        "status": R298_R5_TRAJECTORY_TRAINING_AUTHORIZATION_STATUS,
        "goal_sha256": ledger["goal_sha256"],
        "contract_sha256": ledger["contract_sha256"],
        "goal_revision": R298_CANONICAL_GOAL_REVISION,
        "root_owner_revision": R298_ROOT_OWNER_REVISION,
        "production_typed_source_sha256": R298_PRODUCTION_TYPED_SOURCE_SHA256,
        "target_only": True,
        "may_drive_runtime": False,
        "target_schema": R298_RULE_TARGET_SCHEMA,
        "target_schema_digest": R298_RULE_TARGET_SCHEMA_DIGEST,
        "raw_corpus_receipt_sha256": raw_corpus_sha,
        "frozen_schema_manifest_sha256": frozen_schema_sha,
        "schema_freeze_receipt_sha256": schema_freeze_sha,
        "branch_support_receipt_sha256": branch_support_sha,
        "training_gate_report_sha256": training_gate_sha,
        "training_handoff_activation_receipt_sha256": handoff_sha,
        "trajectory_ledger_file_sha256": _file_sha256(ledger_path),
        "trajectory_ledger_payload_sha256": ledger["ledger_payload_sha256"],
        "candidate_training_allowed": True,
        "selected_action_target_training_allowed": True,
        "training_host": R298_R5_TRAINING_HOST,
        "revision_5_handoff_activation_bound": True,
        "runtime_wired": False,
        "production_serving_authority": False,
        "revision_4_predecessor_evidence_only": True,
        "blind_revision_4_substitution_allowed": False,
    }
    authorization["authorization_payload_sha256"] = _sha256(authorization)
    authorization_path = root / "trajectory-authorization.json"
    _write_canonical_json(authorization_path, authorization)
    validator = load_sealed_selected_action_trajectory_validator(
        ledger_path,
        authorization_path,
        schema_freeze_receipt_path=schema_freeze_path,
        training_handoff_activation_receipt_path=handoff_path,
        expected_authorization_file_sha256=_file_sha256(authorization_path),
        expected_raw_corpus_receipt_sha256=raw_corpus_sha,
        expected_frozen_schema_manifest_sha256=frozen_schema_sha,
        expected_schema_freeze_receipt_sha256=schema_freeze_sha,
        expected_branch_support_receipt_sha256=branch_support_sha,
        expected_training_gate_report_sha256=training_gate_sha,
        expected_training_handoff_activation_receipt_sha256=handoff_sha,
    )
    return validator, workspace


def _renumber_serials(value: Any, *, offset: int) -> Any:
    if isinstance(value, list):
        return [_renumber_serials(item, offset=offset) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for name, nested in value.items():
        if name.casefold().endswith("serial") and isinstance(nested, int):
            result[name] = nested + offset
        else:
            result[name] = _renumber_serials(nested, offset=offset)
    return result


def test_public_snapshot_identity_is_invariant_to_global_serial_renumbering() -> None:
    original = _observation()
    renumbered = _renumber_serials(copy.deepcopy(original), offset=50_000)

    assert public_observation_fingerprint(original) == public_observation_fingerprint(
        renumbered
    )


def test_public_target_identity_uses_selected_semantics_not_candidate_ordinal() -> None:
    left_observation = _observation()
    left_observation["select"]["option"].append({"type": "Number", "number": 1})
    left = _decision(left_observation)
    left_compiled = compile_simulator_rule_targets(left, strict=True)

    right_observation = copy.deepcopy(left_observation)
    right_observation["select"]["option"] = list(
        reversed(right_observation["select"]["option"])
    )
    right = _decision(right_observation)
    right["action"] = [1]
    right["prompt_chain"]["root_action"] = [1]
    right["prompt_chain"]["events"][0]["action"] = [1]
    right_compiled = compile_simulator_rule_targets(right, strict=True)

    assert left_compiled["selected_action"] == [0]
    assert right_compiled["selected_action"] == [1]
    assert public_target_fingerprint(left_compiled) == public_target_fingerprint(
        right_compiled
    )


def test_belief_sidecar_requires_chain_binding_and_external_receipt_validator() -> None:
    observation = _observation()
    decision = _decision(observation)
    compiled = compile_simulator_rule_targets(decision, strict=True)
    assert compiled["status"] == "available"

    receipt = {
        "schema": R298_PRIVILEGED_BELIEF_TARGET_SCHEMA,
        "version": R298_PRIVILEGED_BELIEF_TARGET_SCHEMA_VERSION,
        "target_only": True,
        "may_drive_runtime": False,
        "normalized_prompt_chain_sha256": compiled["prompt_chain"][
            "realized_target_chain_hash"
        ],
        "public_observation_sha256": compiled["public_observation_hash"],
        "selected_action_sha256": _sha256({"selected_action": [0]}),
        "trajectory_receipt_sha256": "sha256:" + "a" * 64,
        "corpus_receipt_sha256": "sha256:" + "b" * 64,
    }
    decision["privileged_belief_targets"] = {
        "provenance": "privileged_target_only_authoritative_trace",
        "receipt": receipt,
        "opponent_hand": [901, 901],
        "opponent_remainder": [[902, 1]],
    }

    # SHA-shaped caller data alone never makes hidden-card labels trainable.
    unverified = compile_simulator_rule_targets(decision, strict=True)
    assert unverified["status"] == "available"
    assert unverified["privileged_belief_targets"] == {
        "schema": "poke_bot.alakazam_opponent_belief_targets_r298/v1",
        "target_only": True,
        "available": False,
        "hand_count_distribution": {"pairs": [], "mask": False, "reason": "absent"},
        "remainder_count_distribution": {
            "pairs": [],
            "mask": False,
            "reason": "absent",
        },
        "policy_feature_eligible": False,
        "reason": "immutable_receipt_unverified",
    }

    class AcceptingValidator:
        def validate_privileged_belief_receipt(self, **kwargs: Any) -> bool:
            return kwargs["receipt"] is receipt

    # Even a validator cannot waive the compiler's direct normalized-chain
    # binding; it only attests the external immutable trajectory/corpus link.
    receipt["normalized_prompt_chain_sha256"] = "sha256:" + "c" * 64
    wrong_chain = compile_simulator_rule_targets(
        decision,
        privileged_belief_receipt_validator=AcceptingValidator(),
        strict=True,
    )
    assert wrong_chain["privileged_belief_targets"]["available"] is False

    receipt["normalized_prompt_chain_sha256"] = compiled["prompt_chain"][
        "realized_target_chain_hash"
    ]
    verified = compile_simulator_rule_targets(
        decision,
        privileged_belief_receipt_validator=AcceptingValidator(),
        strict=True,
    )
    belief = verified["privileged_belief_targets"]
    # A generic callback cannot convert caller-supplied hidden card counts
    # into a supervised sidecar.  The sealed public trajectory ledger has no
    # hidden-payload extension yet, so this remains explicitly unavailable.
    assert belief["available"] is False
    assert belief["policy_feature_eligible"] is False
    assert belief["reason"] == "immutable_receipt_unverified"

    # A separately validated belief sidecar cannot bypass the independent raw
    # trajectory/corpus receipt gate for any supervised vector.
    vectors = rule_head_target_vectors(verified)
    assert vectors["target_training_eligible"] is False
    assert vectors["lethal_threat"]["mask"] == [False]
    assert vectors["opponent_belief"]["hand_count_distribution"]["mask"] is False

    selected_keys = [
        row["semantic_key_sha256"]
        for row in compiled["legal_option_semantics"]["selected"]
    ]
    decision["prompt_chain"]["observed_trajectory_receipt"] = {
        "schema": "poke_bot.alakazam_observed_selected_action_trajectory_receipt/v1",
        "version": 1,
        "target_only": True,
        "may_drive_runtime": False,
        "normalized_prompt_chain_sha256": compiled["prompt_chain"][
            "realized_target_chain_hash"
        ],
        "public_observation_sha256": compiled["public_observation_hash"],
        "selected_action_semantic_sha256": _sha256(
            {"selected_option_semantic_keys": sorted(selected_keys)}
        ),
        "trajectory_receipt_sha256": "sha256:" + "e" * 64,
        "corpus_receipt_sha256": "sha256:" + "f" * 64,
        "raw_frame_receipt_sha256": "sha256:" + "0" * 64,
    }

    class AcceptingTrajectoryValidator:
        def validate_selected_action_trajectory_receipt(self, **kwargs: Any) -> bool:
            return kwargs["receipt"] is decision["prompt_chain"][
                "observed_trajectory_receipt"
            ]

    # A generic object that returns True is only a diagnostic seam.  It can
    # prove wiring in a fixture, never authorize an actual target mask.
    unsealed_callback = compile_simulator_rule_targets(
        decision,
        privileged_belief_receipt_validator=AcceptingValidator(),
        trajectory_receipt_validator=AcceptingTrajectoryValidator(),
        strict=True,
    )
    unsealed_provenance = unsealed_callback["provenance"]["trajectory_receipt"]
    assert unsealed_provenance["externally_validated"] is True
    assert unsealed_provenance["validator_kind"] == "unsealed_callback_diagnostic_only"
    assert unsealed_provenance["trainable_target_eligible"] is False
    assert rule_head_target_vectors(unsealed_callback)["target_training_eligible"] is False
    try:
        compile_simulator_rule_targets(
            decision,
            trajectory_receipt_validator=AcceptingTrajectoryValidator(),
            require_trainable_trajectory_receipt=True,
            strict=True,
        )
    except SimulatorRuleTargetError:
        pass
    else:  # pragma: no cover - assertion protects the training authority wall.
        raise AssertionError("a generic callback unexpectedly armed target training")

    # Only a validator loaded from a content-addressed exact-30-day ledger
    # plus a post-census authorization receipt can arm target masks.
    validator, workspace = _sealed_trajectory_validator_for_test(
        receipt=decision["prompt_chain"]["observed_trajectory_receipt"],
        compiled=unsealed_callback,
    )
    try:
        trainable = compile_simulator_rule_targets(
            decision,
            privileged_belief_receipt_validator=AcceptingValidator(),
            trajectory_receipt_validator=validator,
            strict=True,
        )
    finally:
        workspace.cleanup()
    provenance = trainable["provenance"]["trajectory_receipt"]
    assert provenance["trainable_target_eligible"] is True
    assert provenance["validator_kind"] == R298_SEALED_TRAJECTORY_VALIDATOR_KIND
    assert provenance["validator_provenance"]["target_only"] is True
    # A compiled mapping alone is intentionally not a training capability;
    # vectorization must receive the same sealed receipt validator.
    assert rule_head_target_vectors(trainable)["target_training_eligible"] is False
    trainable_vectors = rule_head_target_vectors(
        trainable,
        trajectory_receipt_validator=validator,
    )
    assert trainable_vectors["target_training_eligible"] is True
    assert trainable_vectors["attack_readiness"]["mask"][:2] == [True, True]


def test_complete_chain_zeroes_absent_ko_and_promotion_but_masks_cross_turn_resources() -> None:
    before = _observation()
    after = copy.deepcopy(before)
    after["current"]["turn"] += 1
    event = _ChainEvent(
        before=before,
        after=after,
        before_terminal={},
        after_terminal={},
        event_kind="skill",
        action=(0,),
        forced=False,
        strategic_decision=True,
        facts={},
    )
    chain = _NormalizedChain(
        root_action=(0,),
        root_before=before,
        root_public_observation_hash=public_observation_fingerprint(before),
        final_after=after,
        root_terminal={},
        final_terminal={},
        events=(event,),
        simulator=R298_CANONICAL_SIMULATOR,
        source="observed_selected_action_trajectory",
        restoration_provenance=None,
        event_log_complete=True,
        complete_to_next_strategic_decision=True,
        chain_hash="sha256:" + "d" * 64,
    )

    assert _opponent_knockout_count(chain.events, actor=0) == (0, True)
    assert _forced_promotion_count(chain.events, actor=0) == (0, True)
    resources = _turn_resource_target(chain, actor=0)
    assert resources["mask"] == [False] * len(resources["mask"])
    prizes = _prize_yield_target(chain, actor=0, metadata_cards={})
    assert prizes["public_predicted_yield"] == 0
    assert prizes["public_predicted_yield_mask"] is True
