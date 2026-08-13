"""Simulator-derived, target-only rule labels for the isolated Alakazam r298 path.

This is deliberately a *derivative* compiler.  It does not change the r195,
r241, or r274 model, their features, existing auxiliary heads, Fusion,
OwnDeck, Matchup Adapter, legal actions, or runtime selection.  Its input is
one selected legal action plus the public, deterministic simulator prompt
chain that actually followed that action.  It never invents labels for an
unchosen option and it never calls search with a guessed hidden opponent deck.

The companion :mod:`alakazam_public_rule_adapter_r298` owns the public option
representation.  This module owns only target construction:

* selected-action attack / KO / Prize / promotion credit across forced prompts;
* repaired target definitions for legacy strategic families;
* exact public forced-draw and deck-out facts when the chain records them; and
* a separately typed, target-only opponent-belief payload.

All failures are fail-closed.  ``compile_simulator_rule_targets`` returns a
fully masked target by default; callers that materialize a corpus can request
``strict=True`` to reject the row instead.  Nothing returned here is suitable
as a serving policy feature.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


R298_RULE_TARGET_SCHEMA = "poke_bot.alakazam_simulator_rule_targets/v1"
R298_RULE_TARGET_SCHEMA_VERSION = 1
R298_PROMPT_CHAIN_SCHEMA = "poke_bot.alakazam_selected_action_prompt_chain/v1"
R298_PROMPT_CHAIN_SCHEMA_VERSION = 1
R298_TARGET_CONFIG_SCHEMA = "poke_bot.alakazam_simulator_rule_targets_config/v1"
R298_TARGET_PROVENANCE_SCHEMA = "poke_bot.alakazam_selected_action_target_provenance/v1"
R298_TARGET_PROVENANCE_SCHEMA_VERSION = 1
R298_PRIVILEGED_BELIEF_TARGET_SCHEMA = (
    "poke_bot.alakazam_privileged_belief_target_receipt/v1"
)
R298_PRIVILEGED_BELIEF_TARGET_SCHEMA_VERSION = 1
R298_PUBLIC_SNAPSHOT_IDENTITY_SCHEMA = (
    "poke_bot.alakazam_public_semantic_snapshot_identity/v1"
)
R298_PUBLIC_SNAPSHOT_IDENTITY_SCHEMA_VERSION = 1
R298_OBSERVED_TRAJECTORY_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_observed_selected_action_trajectory_receipt/v1"
)
R298_OBSERVED_TRAJECTORY_RECEIPT_SCHEMA_VERSION = 1
R298_TRAJECTORY_LEDGER_SCHEMA = (
    "poke_bot.alakazam_selected_action_trajectory_ledger/v1"
)
R298_TRAJECTORY_LEDGER_SCHEMA_VERSION = 1
R298_TRAJECTORY_LEDGER_ENTRY_SCHEMA = (
    "poke_bot.alakazam_selected_action_trajectory_ledger_entry/v1"
)
R298_TRAJECTORY_TRAINING_AUTHORIZATION_SCHEMA = (
    "poke_bot.alakazam_selected_action_target_training_authorization/v1"
)
R298_TRAJECTORY_TRAINING_AUTHORIZATION_SCHEMA_VERSION = 1
R298_SEALED_TRAJECTORY_VALIDATOR_KIND = (
    "poke_bot.alakazam_sealed_selected_action_trajectory_validator/v1"
)
R298_REVISION = 298
R298_CANONICAL_GOAL_REVISION = 5
R298_ROOT_OWNER_REVISION = 303
R298_R5_SCHEMA_FREEZE_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_rule_derivative_schema_freeze_receipt/v1"
)
R298_R5_HANDOFF_ACTIVATION_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_rule_derivative_inzi_training_handoff_activation_receipt/v1"
)
R298_R5_DERIVATIVE_LINEAGE_ID = "alakazam-rule-derivative-g5"
R298_R5_TRAINING_HOST = "inzi_blackwell"
R298_R5_TRAJECTORY_LEDGER_STATUS = (
    "sealed_complete_30_day_selected_action_trajectory_ledger_r5"
)
R298_R5_TRAJECTORY_TRAINING_AUTHORIZATION_STATUS = (
    "passed_r5_census_supported_selected_action_target_training_after_handoff"
)

# r298 is historical provenance; the dedicated r303/revision-5 experiment
# contract is the live authority for this derivative.  The revision-4
# identities below are retained only as immutable predecessor evidence.  A
# caller cannot substitute them for revision 5 merely because a catalog or
# schema byte stream happens to be identical.
R298_CANONICAL_GOAL_PATH = "goals/alakazam-elmo-rule-derivative/GOAL.md"
R298_CANONICAL_GOAL_SHA256 = (
    "sha256:7a829abebd348d0ffdf0a73c8b559fe9c799af3d3aff49a64efdfa85a08051b6"
)
R298_CANONICAL_CONTRACT_PATH = "goals/alakazam-elmo-rule-derivative/contract.json"
R298_CANONICAL_CONTRACT_SHA256 = (
    "sha256:dbbd4dbcc057b631d61fa867e45c393d594550b3b45f306f465b6ee5b4428891"
)
R298_PRODUCTION_TYPED_SOURCE_PATH = "state/alakazam-new-list-direct-policy-r241.json"
R298_PRODUCTION_TYPED_SOURCE_SHA256 = (
    "sha256:8d83f5e9eafc8e554f33dcbfbda7e1b337b8f65dc2e49109be4acdd829850c1e"
)
R298_PREDECESSOR_GOAL_REVISION = 4
R298_PREDECESSOR_GOAL_SHA256 = (
    "sha256:2af67560510ca7ffd9fe0bc6ff37cdbbd74f5a78d6c5237091bb527d49ce4ed8"
)
R298_PREDECESSOR_CONTRACT_SHA256 = (
    "sha256:f65e023d454375cfd59324306044da10a116201a187415f0534e24c239bd2dc2"
)
R298_RAW_CORPUS_COMPATIBILITY = {
    "window_start_utc": "2026-07-13",
    "window_end_utc": "2026-08-11",
    "utc_partition_count_exact": 30,
    "validated_deduplicated_manifest_sha256_required": True,
    "split_disjoint_dimensions": ("source", "utc_day_partition", "group"),
    "twenty_day_or_subset_fallback_allowed": False,
}

# The pinned source is intentionally recorded in the target schema instead of
# relying on whichever ``cg`` package happens to be importable in a collector.
# The compiler consumes a trace emitted by that simulator; it does not turn a
# policy-visible observation into a hidden-state simulator search root.
R298_CANONICAL_SIMULATOR = {
    "typed_source": "state/canonical-libcg-r236.json",
    "typed_source_sha256": (
        "sha256:d75ff752808ead08f3ae20f7f2f8a034c9e6163109188a46d3b877bf1910ae2d"
    ),
    "kaggle_environments_version": "1.32.6",
    "linux_x86_64_sha256": (
        "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7"
    ),
    "legal_option_authority": "pinned_competition_simulator",
}

LETHAL_THREAT_LAYOUT: tuple[str, ...] = ("selected_action_prize_conversion",)
PRIZE_RACE_LAYOUT: tuple[str, ...] = (
    "acting_prizes_remaining",
    "opponent_prizes_remaining",
    "acting_visible_prize_liability",
    "opponent_visible_prize_liability",
    "acting_known_prize_modifier",
    "opponent_known_prize_modifier",
)
ACTION_UTILITY_LAYOUT: tuple[str, ...] = (
    "damage_dealt",
    "damage_counters_placed",
    "cards_drawn",
    "attached_energy_delta",
    "open_bench_delta",
    "own_prize_delta",
    "opponent_knockout",
    "forced_promotion_count",
    "terminal_own_win",
)
TURN_RESOURCE_LAYOUT: tuple[str, ...] = (
    "supporter_played",
    "stadium_played",
    "energy_attached",
    "retreated",
    "turn_action_count_delta",
)
GAME_PHASE_CLASSES: tuple[str, ...] = (
    "setup",
    "stabilize",
    "pressure",
    "prize_race",
    "closeout",
    "terminal",
)
TERMINAL_CONVERSION_CLASSES: tuple[str, ...] = (
    "nonterminal",
    "own_win",
    "own_loss",
    "draw",
)
TERMINAL_CONVERSION_LAYOUT: tuple[str, ...] = (
    "terminal_class.nonterminal",
    "terminal_class.own_win",
    "terminal_class.own_loss",
    "terminal_class.draw",
    "prize_closeout_after_forced_chain",
    "opponent_knockout_after_forced_chain",
)
ATTACK_READINESS_LAYOUT: tuple[str, ...] = (
    "selected_option_is_attack",
    "attack_legal_in_simulator_option_list",
    "typed_cost_known",
    "typed_cost_satisfied",
)

_SCHEMA_DEFINITION = {
    "schema": R298_RULE_TARGET_SCHEMA,
    "version": R298_RULE_TARGET_SCHEMA_VERSION,
    "revision": R298_REVISION,
    "canonical_goal": {
        "path": R298_CANONICAL_GOAL_PATH,
        "sha256": R298_CANONICAL_GOAL_SHA256,
        "revision": R298_CANONICAL_GOAL_REVISION,
    },
    "canonical_contract": {
        "path": R298_CANONICAL_CONTRACT_PATH,
        "sha256": R298_CANONICAL_CONTRACT_SHA256,
    },
    "root_handoff": {
        "root_owner_revision": R298_ROOT_OWNER_REVISION,
        "production_typed_source": R298_PRODUCTION_TYPED_SOURCE_PATH,
        "production_typed_source_sha256": R298_PRODUCTION_TYPED_SOURCE_SHA256,
        "semantic_owner": "dedicated_goal_contract",
        "production_transition_owner": "production_typed_source_only",
        "immediate_runtime_or_service_authority": False,
    },
    "revision_4_predecessor": {
        "goal_revision": R298_PREDECESSOR_GOAL_REVISION,
        "goal_sha256": R298_PREDECESSOR_GOAL_SHA256,
        "contract_sha256": R298_PREDECESSOR_CONTRACT_SHA256,
        "historical_evidence_only": True,
        "schema_or_catalog_bytes_may_be_reused_only_if_checksum_identical": True,
        "blind_hash_substitution_allowed": False,
        "satisfies_revision_5_schema_freeze_alone": False,
    },
    "raw_corpus_compatibility": dict(R298_RAW_CORPUS_COMPATIBILITY),
    "simulator": R298_CANONICAL_SIMULATOR,
    "source": "selected_legal_action_plus_deterministic_public_prompt_chain_only",
    "counterfactual_labels": False,
    "restored_trajectory_requires": {
        "schema": R298_TARGET_PROVENANCE_SCHEMA,
        "targeted_libcg_corpus": True,
        "public_information_seed_only": True,
        "immutable_state_seed_and_trajectory_receipts": True,
        "bind_selected_public_observation_and_action": True,
    },
    "observed_trajectory_requires": {
        "receipt_schema": R298_OBSERVED_TRAJECTORY_RECEIPT_SCHEMA,
        "sealed_ledger_schema": R298_TRAJECTORY_LEDGER_SCHEMA,
        "training_authorization_schema": R298_TRAJECTORY_TRAINING_AUTHORIZATION_SCHEMA,
        "revision_5_schema_freeze_receipt_schema": R298_R5_SCHEMA_FREEZE_RECEIPT_SCHEMA,
        "revision_5_handoff_activation_receipt_schema": R298_R5_HANDOFF_ACTIVATION_RECEIPT_SCHEMA,
        "bind_normalized_selected_action_chain": True,
        "bind_public_observation_and_selected_semantics": True,
        "bind_external_immutable_trajectory_and_corpus_receipts": True,
        "external_immutable_receipt_validator_required_for_trainable_masks": True,
        "generic_callback_can_enable_trainable_masks": False,
        "sealed_validator_requires": (
            "immutable_ledger_post_census_authorization_revision_5_schema_freeze_and_handoff_activation_receipts"
        ),
        "revision_5_handoff_activation_required_for_trainable_masks": True,
        "candidate_training_host_after_handoff": R298_R5_TRAINING_HOST,
        "current_materialization_behavior": (
            "all_trainable_masks_false_without_a_revision_5_sealed_ledger_validator"
        ),
        "unverified_behavior": "diagnostic_target_only_all_trainable_masks_false",
    },
    "event_fact_boundary": {
        "retain_only": (
            "typed_damage_draw_discard_bench_prize_counts",
            "public_ko_card_class",
            "public_player_seat",
            "structured_visible_prize_modifier",
            "forced_marker",
        ),
        "typed_fact_values": "canonicalized_and_type_checked_before_target_hash_or_output",
        "opaque_or_private_fact_fields": "dropped_before_target_hash_or_output",
    },
    "public_snapshot_identity": {
        "schema": R298_PUBLIC_SNAPSHOT_IDENTITY_SCHEMA,
        "version": R298_PUBLIC_SNAPSHOT_IDENTITY_SCHEMA_VERSION,
        "source": "public_rule_state_selection_and_option_semantics",
        "raw_global_serials": "internal_join_only_never_hashed_or_emitted",
        "terminal_and_future_fields": "excluded_from_predecision_identity",
        "option_order": "canonical_multiset_not_incidental_candidate_ordinal",
        "selected_action_identity": "selected_semantic_keys_not_numeric_option_indices",
    },
    "catalog_metadata_boundary": {
        "materialization_requires": "receipt_sealed_public_catalog",
        "unsealed_catalog_behavior": "metadata_dependent_labels_masked_unavailable",
        "test_only_opt_in": "compile_simulator_rule_targets.allow_test_catalog",
        "revision_4_catalog_receipt_alone_eligible": False,
        "revision_5_schema_freeze_consumer_receipt_required": True,
    },
    "privileged_belief_boundary": {
        "target_only": True,
        "immutable_trajectory_receipt_required": True,
        "receipt_schema": R298_PRIVILEGED_BELIEF_TARGET_SCHEMA,
        "receipt_binds_normalized_selected_action_chain": True,
        "external_immutable_receipt_validator_required": True,
        "validator_must_bind_trajectory_and_corpus_receipts": True,
        "generic_callback_can_enable_hidden_targets": False,
        "current_behavior": "masked_unavailable_until_a_sealed_ledger_extension_binds_privileged_target_payloads",
        "restored_source_requires_public_seed_and_restoration_receipts": True,
        "may_drive_runtime": False,
    },
    "target_semantics": {
        "action_utility": {
            "credit": "selected_action_through_complete_deterministic_forced_prompt_chain",
            "unchosen_options": "never_labeled",
            "no_named_knockout_or_promotion_in_complete_chain": "exact_zero",
            "numeric_effect_without_typed_event_fact": "masked_unavailable",
        },
        "prize_yield": {
            "source_of_truth": "simulator_prompt_chain_prize_delta",
            "public_prediction_requires": "every_knockout_victim_and_applicable_yield_typed",
            "no_knockout_exact_zero_requires_observed_zero_prize_delta": True,
            "mega_ex_precedence": "three_prizes_before_ordinary_ex",
        },
        "turn_resources": {
            "snapshot_comparison": "complete_chain_root_to_final_same_turn_only",
            "cross_turn_or_missing_turn_identity": "masked_unavailable_not_false_zero",
        },
        "game_phase": {
            "kind": "deterministic_public_target_taxonomy_not_a_simulator_rule",
            "terminal": "terminal_result_after_complete_chain",
            "closeout": "selected_chain_reaches_own_prize_zero",
            "remaining_classes": "public_turn_and_prize_precedence_only",
            "policy_feature_eligible": False,
        },
    },
    "policy_feature_eligible": False,
    "candidate_training_requires": {
        "revision_5_schema_freeze_receipt": True,
        "revision_5_handoff_activation_receipt": True,
        "frozen_parent_and_existing_surfaces": True,
        "production_serving_selector_authority": False,
    },
    "layouts": {
        "lethal_threat": list(LETHAL_THREAT_LAYOUT),
        "prize_race": list(PRIZE_RACE_LAYOUT),
        "action_utility": list(ACTION_UTILITY_LAYOUT),
        "turn_resources": list(TURN_RESOURCE_LAYOUT),
        "game_phase": list(GAME_PHASE_CLASSES),
        "terminal_conversion": list(TERMINAL_CONVERSION_LAYOUT),
        "attack_readiness": list(ATTACK_READINESS_LAYOUT),
    },
    "option_conditioned_target_binding": {
        "families": (
            "action_utility",
            "terminal_conversion",
            "turn_resources",
            "attack_readiness",
        ),
        "selected_option_indices_field": "selected_option_indices",
        "semantics": "pool_exact_selected_action_rows_only_no_unchosen_option_labels",
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


R298_RULE_TARGET_SCHEMA_DIGEST = _digest(_SCHEMA_DEFINITION)


def r298_rule_target_schema_manifest() -> dict[str, Any]:
    """Return the immutable, receipt-bindable target/provenance schema.

    Re-featurization binds this payload and its digest before consuming the
    exact 30-day archive.  It intentionally describes only schemas, fixed
    layouts, source boundaries, and zero-authority contracts—not any trained
    weights, target rows, or runtime activation state.
    """

    return {
        "schema": R298_RULE_TARGET_SCHEMA,
        "version": R298_RULE_TARGET_SCHEMA_VERSION,
        "revision": R298_REVISION,
        "schema_digest": R298_RULE_TARGET_SCHEMA_DIGEST,
        "canonical_goal": {
            "path": R298_CANONICAL_GOAL_PATH,
            "sha256": R298_CANONICAL_GOAL_SHA256,
            "revision": R298_CANONICAL_GOAL_REVISION,
        },
        "canonical_contract": {
            "path": R298_CANONICAL_CONTRACT_PATH,
            "sha256": R298_CANONICAL_CONTRACT_SHA256,
        },
        "root_handoff": {
            "root_owner_revision": R298_ROOT_OWNER_REVISION,
            "production_typed_source": R298_PRODUCTION_TYPED_SOURCE_PATH,
            "production_typed_source_sha256": R298_PRODUCTION_TYPED_SOURCE_SHA256,
            "semantic_owner": "dedicated_goal_contract",
            "production_transition_owner": "production_typed_source_only",
            "immediate_runtime_or_service_authority": False,
        },
        "revision_4_predecessor": {
            "goal_revision": R298_PREDECESSOR_GOAL_REVISION,
            "goal_sha256": R298_PREDECESSOR_GOAL_SHA256,
            "contract_sha256": R298_PREDECESSOR_CONTRACT_SHA256,
            "historical_evidence_only": True,
            "schema_or_catalog_bytes_may_be_reused_only_if_checksum_identical": True,
            "blind_hash_substitution_allowed": False,
            "satisfies_revision_5_schema_freeze_alone": False,
        },
        "raw_corpus_compatibility": copy.deepcopy(R298_RAW_CORPUS_COMPATIBILITY),
        "schema_definition": copy.deepcopy(_SCHEMA_DEFINITION),
        "prompt_chain": {
            "schema": R298_PROMPT_CHAIN_SCHEMA,
            "version": R298_PROMPT_CHAIN_SCHEMA_VERSION,
            "selected_action_only": True,
            "unchosen_counterfactual_labels": False,
            "restored_source_requires_provenance_schema": R298_TARGET_PROVENANCE_SCHEMA,
            "restored_source_requires_public_information_seed": True,
            "restored_source_binds_selected_public_observation_and_action": True,
        },
        "observed_trajectory": {
            "schema": R298_OBSERVED_TRAJECTORY_RECEIPT_SCHEMA,
            "version": R298_OBSERVED_TRAJECTORY_RECEIPT_SCHEMA_VERSION,
            "sealed_ledger_schema": R298_TRAJECTORY_LEDGER_SCHEMA,
            "sealed_ledger_version": R298_TRAJECTORY_LEDGER_SCHEMA_VERSION,
            "training_authorization_schema": R298_TRAJECTORY_TRAINING_AUTHORIZATION_SCHEMA,
            "training_authorization_version": R298_TRAJECTORY_TRAINING_AUTHORIZATION_SCHEMA_VERSION,
            "sealed_validator_kind": R298_SEALED_TRAJECTORY_VALIDATOR_KIND,
            "revision_5_schema_freeze_receipt_schema": R298_R5_SCHEMA_FREEZE_RECEIPT_SCHEMA,
            "revision_5_handoff_activation_receipt_schema": R298_R5_HANDOFF_ACTIVATION_RECEIPT_SCHEMA,
            "external_immutable_receipt_validator_required_for_trainable_masks": True,
            "generic_callback_can_enable_trainable_masks": False,
            "sealed_validator_requires_post_census_authorization": True,
            "revision_5_handoff_activation_required_for_trainable_masks": True,
            "candidate_training_host_after_handoff": R298_R5_TRAINING_HOST,
            "unverified_behavior": "diagnostic_target_only_all_trainable_masks_false",
        },
        "public_snapshot_identity": {
            "schema": R298_PUBLIC_SNAPSHOT_IDENTITY_SCHEMA,
            "version": R298_PUBLIC_SNAPSHOT_IDENTITY_SCHEMA_VERSION,
            "serial_invariant": True,
            "raw_global_serials": "internal_join_only_never_hashed_or_emitted",
            "terminal_and_future_fields": "excluded_from_predecision_identity",
            "option_order": "canonical_multiset_not_incidental_candidate_ordinal",
        },
        "privileged_belief": {
            "schema": R298_PRIVILEGED_BELIEF_TARGET_SCHEMA,
            "version": R298_PRIVILEGED_BELIEF_TARGET_SCHEMA_VERSION,
            "target_only": True,
            "external_immutable_receipt_validator_required": True,
            "generic_callback_can_enable_hidden_targets": False,
            "current_behavior": "masked_unavailable_until_sealed_ledger_extension",
            "policy_feature_eligible": False,
        },
        "default_zero_and_inert": True,
        "runtime_wired": False,
        "policy_feature_eligible": False,
        "training_or_bo250_before_30_day_census": False,
        "candidate_training_requires_revision_5_handoff_activation": True,
    }


def assert_r298_rule_target_schema_binding(
    manifest: Mapping[str, Any],
    *,
    require_current_goal_files: bool = False,
) -> None:
    """Fail closed when a materialization receipt does not bind this schema."""

    row = _mapping(manifest, field="r298 target schema manifest")
    if row.get("schema") != R298_RULE_TARGET_SCHEMA:
        raise SimulatorRuleTargetError("r298 target schema manifest schema mismatch")
    if row.get("version") != R298_RULE_TARGET_SCHEMA_VERSION:
        raise SimulatorRuleTargetError("r298 target schema manifest version mismatch")
    if row.get("revision") != R298_REVISION:
        raise SimulatorRuleTargetError("r298 target schema manifest revision mismatch")
    if row.get("schema_digest") != R298_RULE_TARGET_SCHEMA_DIGEST:
        raise SimulatorRuleTargetError("r298 target schema manifest digest mismatch")
    contract = _mapping(row.get("canonical_contract"), field="manifest.canonical_contract")
    goal = _mapping(row.get("canonical_goal"), field="manifest.canonical_goal")
    if contract.get("path") != R298_CANONICAL_CONTRACT_PATH or contract.get("sha256") != R298_CANONICAL_CONTRACT_SHA256:
        raise SimulatorRuleTargetError("r298 target schema has stale canonical contract binding")
    if (
        goal.get("path") != R298_CANONICAL_GOAL_PATH
        or goal.get("sha256") != R298_CANONICAL_GOAL_SHA256
        or goal.get("revision") != R298_CANONICAL_GOAL_REVISION
    ):
        raise SimulatorRuleTargetError("r298 target schema has stale canonical goal binding")
    root_handoff = _mapping(row.get("root_handoff"), field="manifest.root_handoff")
    if (
        root_handoff.get("root_owner_revision") != R298_ROOT_OWNER_REVISION
        or root_handoff.get("production_typed_source") != R298_PRODUCTION_TYPED_SOURCE_PATH
        or root_handoff.get("production_typed_source_sha256")
        != R298_PRODUCTION_TYPED_SOURCE_SHA256
        or root_handoff.get("semantic_owner") != "dedicated_goal_contract"
        or root_handoff.get("production_transition_owner")
        != "production_typed_source_only"
        or root_handoff.get("immediate_runtime_or_service_authority") is not False
    ):
        raise SimulatorRuleTargetError("r298 target schema has stale revision-5 handoff binding")
    predecessor = _mapping(row.get("revision_4_predecessor"), field="manifest.revision_4_predecessor")
    if (
        predecessor.get("goal_revision") != R298_PREDECESSOR_GOAL_REVISION
        or predecessor.get("goal_sha256") != R298_PREDECESSOR_GOAL_SHA256
        or predecessor.get("contract_sha256") != R298_PREDECESSOR_CONTRACT_SHA256
        or predecessor.get("historical_evidence_only") is not True
        or predecessor.get("schema_or_catalog_bytes_may_be_reused_only_if_checksum_identical")
        is not True
        or predecessor.get("blind_hash_substitution_allowed") is not False
        or predecessor.get("satisfies_revision_5_schema_freeze_alone") is not False
    ):
        raise SimulatorRuleTargetError("r298 target schema predecessor boundary is not fail-closed")
    if row.get("default_zero_and_inert") is not True or row.get("runtime_wired") is not False:
        raise SimulatorRuleTargetError("r298 target schema is not zero-inert")
    snapshot = _mapping(
        row.get("public_snapshot_identity"), field="manifest.public_snapshot_identity"
    )
    if (
        snapshot.get("schema") != R298_PUBLIC_SNAPSHOT_IDENTITY_SCHEMA
        or snapshot.get("version") != R298_PUBLIC_SNAPSHOT_IDENTITY_SCHEMA_VERSION
        or snapshot.get("serial_invariant") is not True
    ):
        raise SimulatorRuleTargetError("r298 target schema lacks serial-invariant public identity")
    belief = _mapping(row.get("privileged_belief"), field="manifest.privileged_belief")
    if (
        belief.get("schema") != R298_PRIVILEGED_BELIEF_TARGET_SCHEMA
        or belief.get("version") != R298_PRIVILEGED_BELIEF_TARGET_SCHEMA_VERSION
        or belief.get("target_only") is not True
        or belief.get("external_immutable_receipt_validator_required") is not True
        or belief.get("generic_callback_can_enable_hidden_targets") is not False
        or belief.get("current_behavior")
        != "masked_unavailable_until_sealed_ledger_extension"
        or belief.get("policy_feature_eligible") is not False
    ):
        raise SimulatorRuleTargetError("r298 target schema belief boundary is not fail-closed")
    observed = _mapping(row.get("observed_trajectory"), field="manifest.observed_trajectory")
    if (
        observed.get("schema") != R298_OBSERVED_TRAJECTORY_RECEIPT_SCHEMA
        or observed.get("version") != R298_OBSERVED_TRAJECTORY_RECEIPT_SCHEMA_VERSION
        or observed.get("sealed_ledger_schema") != R298_TRAJECTORY_LEDGER_SCHEMA
        or observed.get("sealed_ledger_version") != R298_TRAJECTORY_LEDGER_SCHEMA_VERSION
        or observed.get("training_authorization_schema")
        != R298_TRAJECTORY_TRAINING_AUTHORIZATION_SCHEMA
        or observed.get("training_authorization_version")
        != R298_TRAJECTORY_TRAINING_AUTHORIZATION_SCHEMA_VERSION
        or observed.get("sealed_validator_kind")
        != R298_SEALED_TRAJECTORY_VALIDATOR_KIND
        or observed.get("revision_5_schema_freeze_receipt_schema")
        != R298_R5_SCHEMA_FREEZE_RECEIPT_SCHEMA
        or observed.get("revision_5_handoff_activation_receipt_schema")
        != R298_R5_HANDOFF_ACTIVATION_RECEIPT_SCHEMA
        or observed.get("external_immutable_receipt_validator_required_for_trainable_masks")
        is not True
        or observed.get("generic_callback_can_enable_trainable_masks") is not False
        or observed.get("sealed_validator_requires_post_census_authorization")
        is not True
        or observed.get("revision_5_handoff_activation_required_for_trainable_masks")
        is not True
        or observed.get("candidate_training_host_after_handoff")
        != R298_R5_TRAINING_HOST
        or observed.get("unverified_behavior")
        != "diagnostic_target_only_all_trainable_masks_false"
    ):
        raise SimulatorRuleTargetError("r298 observed trajectory boundary is not fail-closed")
    if row.get("candidate_training_requires_revision_5_handoff_activation") is not True:
        raise SimulatorRuleTargetError("r298 target schema lacks revision-5 handoff gate")
    if require_current_goal_files:
        root = Path(__file__).resolve().parents[1]
        for relative, expected in (
            (R298_CANONICAL_GOAL_PATH, R298_CANONICAL_GOAL_SHA256),
            (R298_CANONICAL_CONTRACT_PATH, R298_CANONICAL_CONTRACT_SHA256),
            (R298_PRODUCTION_TYPED_SOURCE_PATH, R298_PRODUCTION_TYPED_SOURCE_SHA256),
        ):
            candidate = root / relative
            if not candidate.is_file():
                raise SimulatorRuleTargetError(f"r298 canonical authority file is absent: {relative}")
            digest = "sha256:" + hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest != expected:
                raise SimulatorRuleTargetError(f"r298 canonical authority digest is stale: {relative}")


class SimulatorRuleTargetError(ValueError):
    """A selected-action simulator target cannot be proven from public facts."""


def _norm_token(value: Any) -> str:
    if hasattr(value, "name"):
        value = getattr(value, "name")
    return "".join(char for char in str(value).casefold() if char.isalnum())


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SimulatorRuleTargetError(f"{field} must be an object")
    return value


def _rows(value: Any, *, field: str, optional: bool = False) -> list[Any]:
    if value is None and optional:
        return []
    if not isinstance(value, (list, tuple)):
        raise SimulatorRuleTargetError(f"{field} must be a list")
    return list(value)


def _exact_int(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
    optional: bool = False,
) -> int | None:
    if value is None:
        if optional:
            return None
        raise SimulatorRuleTargetError(f"{field} is required")
    if isinstance(value, bool):
        raise SimulatorRuleTargetError(f"{field} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SimulatorRuleTargetError(f"{field} is not an integer") from exc
    try:
        exact = value == result
    except Exception:
        exact = False
    if not exact:
        raise SimulatorRuleTargetError(f"{field} is not an exact integer")
    if minimum is not None and result < minimum:
        raise SimulatorRuleTargetError(f"{field} is below {minimum}")
    if maximum is not None and result > maximum:
        raise SimulatorRuleTargetError(f"{field} exceeds {maximum}")
    return result


def _optional_bool(value: Any, *, field: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    parsed = _exact_int(value, field=field, minimum=0, maximum=1)
    assert parsed is not None
    return bool(parsed)


def _finite_number(value: Any, *, field: str, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise SimulatorRuleTargetError(f"{field} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SimulatorRuleTargetError(f"{field} is not numeric") from exc
    if not math.isfinite(result):
        raise SimulatorRuleTargetError(f"{field} is not finite")
    return result


def _public_adapter() -> tuple[Any, Any, Any]:
    """Import the parallel representation lazily.

    Keeping this import lazy lets corpus validation use this target contract
    without importing torch until a caller actually asks for semantic-option
    reconstruction.  The representation itself stays the one source of truth
    for the r298 public option keys.
    """

    from .alakazam_public_rule_adapter_r298 import (
        build_public_rule_representation,
        extract_public_terminal_target,
        sanitize_public_observation,
    )

    return (
        build_public_rule_representation,
        sanitize_public_observation,
        extract_public_terminal_target,
    )


def _sanitize_public_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    _build, sanitize, _terminal = _public_adapter()
    try:
        result = sanitize(observation)
    except Exception as exc:
        raise SimulatorRuleTargetError(f"cannot sanitize public observation: {exc}") from exc
    if not isinstance(result, Mapping):
        raise SimulatorRuleTargetError("public observation sanitizer returned no mapping")
    # ``sanitize_public_observation`` owns the hidden-information boundary.
    # The r298 representation intentionally keeps terminal result/reason out
    # of policy features as well, even though they are public after a game has
    # ended.  Keep those facts in the companion target-only sidecar below;
    # never let them perturb a state/option fingerprint used by a policy.
    public = copy.deepcopy(dict(result))
    current = public.get("current")
    if isinstance(current, Mapping):
        trimmed_current = dict(current)
        for name in (
            "result",
            "resultReason",
            "reason",
            "future",
            "futureState",
            "future_state",
            "futureTransitions",
            "future_transitions",
            "nextState",
            "next_state",
        ):
            trimmed_current.pop(name, None)
        public["current"] = trimmed_current
    for name in (
        "future",
        "futureState",
        "future_state",
        "futureTransitions",
        "future_transitions",
        "nextState",
        "next_state",
        "successor",
    ):
        public.pop(name, None)
    return public


def _extract_terminal_target(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Read the adapter's target-only terminal sidecar for one snapshot."""

    _build, _sanitize, extract = _public_adapter()
    try:
        result = extract(observation)
    except Exception as exc:
        raise SimulatorRuleTargetError(f"cannot extract terminal target: {exc}") from exc
    if not isinstance(result, Mapping) or result.get("target_only") is not True:
        raise SimulatorRuleTargetError("terminal extractor returned no target-only mapping")
    return copy.deepcopy(dict(result))


def public_observation_fingerprint(observation: Mapping[str, Any]) -> str:
    """Hash the serial-invariant public semantic identity of one decision.

    The raw snapshot reaches the representation adapter only long enough for
    it to resolve a visible SKILL/source into a stable *within-observation*
    physical locator.  Its sanitizer then removes raw serials entirely.
    Hashing an earlier copied snapshot would make a globally consistent serial
    renumbering look like a new public state and would make restoration
    receipts needlessly brittle.

    Build the adapter's normalized state/selection/options first, then hash
    the canonical option-semantic multiset.  This preserves real simulator
    distinctions (including a stable discriminator after a true semantic
    collision) without encoding raw serials or incidental legal-option order.
    Terminal/future fields remain absent because the adapter representation is
    intentionally pre-decision only.
    """

    build, _sanitize, _terminal = _public_adapter()
    try:
        representation = build(observation, metadata_catalog=None)
    except Exception as exc:
        raise SimulatorRuleTargetError(
            f"cannot build semantic public observation identity: {exc}"
        ) from exc
    state = getattr(representation, "state", None)
    selection = getattr(representation, "selection", None)
    options = getattr(representation, "options", None)
    if not isinstance(state, Mapping) or not isinstance(selection, Mapping):
        raise SimulatorRuleTargetError("public representation has no semantic state/selection")
    if not isinstance(options, (tuple, list)):
        raise SimulatorRuleTargetError("public representation has no semantic option rows")
    option_keys: list[str] = []
    for index, option in enumerate(options):
        key = getattr(option, "semantic_key_sha256", None)
        if not isinstance(key, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", key):
            raise SimulatorRuleTargetError(
                f"public representation option[{index}] has no canonical semantic key"
            )
        option_keys.append(key)
    return _digest(
        {
            "schema": R298_PUBLIC_SNAPSHOT_IDENTITY_SCHEMA,
            "version": R298_PUBLIC_SNAPSHOT_IDENTITY_SCHEMA_VERSION,
            "representation_schema": getattr(representation, "schema", None),
            "representation_revision": getattr(representation, "revision", None),
            "state": copy.deepcopy(dict(state)),
            "selection": copy.deepcopy(dict(selection)),
            "canonical_option_semantic_keys": sorted(option_keys),
        }
    )


def _current(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(observation.get("current"), field="observation.current")


def _actor(observation: Mapping[str, Any]) -> int:
    current = _current(observation)
    actor = _exact_int(
        current.get("yourIndex"),
        field="current.yourIndex",
        minimum=0,
        maximum=1,
    )
    assert actor is not None
    return actor


def _players(observation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = _rows(_current(observation).get("players"), field="current.players")
    if len(rows) != 2 or not all(isinstance(row, Mapping) for row in rows):
        raise SimulatorRuleTargetError("current.players must contain two player objects")
    return [dict(row) for row in rows]


def _player_count(
    observation: Mapping[str, Any],
    *,
    seat: int,
    names: Sequence[str],
    zone: str,
) -> int | None:
    player = _players(observation)[seat]
    for name in names:
        if player.get(name) is not None:
            return _exact_int(player.get(name), field=f"players[{seat}].{name}", minimum=0)
    value = player.get(zone)
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _prize_count(observation: Mapping[str, Any], *, seat: int) -> int | None:
    return _player_count(
        observation,
        seat=seat,
        names=("prizeCount", "prize_count", "remainingPrizes"),
        zone="prize",
    )


def _deck_count(observation: Mapping[str, Any], *, seat: int) -> int | None:
    return _player_count(
        observation,
        seat=seat,
        names=("deckCount", "deck_count"),
        zone="deck",
    )


def _hand_count(observation: Mapping[str, Any], *, seat: int) -> int | None:
    return _player_count(
        observation,
        seat=seat,
        names=("handCount", "hand_count"),
        zone="hand",
    )


def _bench_slots(observation: Mapping[str, Any], *, seat: int) -> int | None:
    player = _players(observation)[seat]
    bench = player.get("bench")
    bench_max = _exact_int(
        player.get("benchMax"),
        field=f"players[{seat}].benchMax",
        minimum=0,
        optional=True,
    )
    if bench_max is None or not isinstance(bench, (list, tuple)):
        return None
    value = bench_max - len(bench)
    return value if value >= 0 else None


def _attached_energy_count(observation: Mapping[str, Any], *, seat: int) -> int | None:
    player = _players(observation)[seat]
    count = 0
    for zone in ("active", "bench"):
        cards = player.get(zone)
        if not isinstance(cards, (list, tuple)):
            return None
        for card in cards:
            if card is None:
                continue
            if not isinstance(card, Mapping):
                return None
            energy = card.get("energyCards")
            if not isinstance(energy, (list, tuple)):
                return None
            count += len(energy)
    return count


def _terminal_result_from_target(target: Mapping[str, Any]) -> tuple[int | None, str | None]:
    """Normalize terminal facts after the policy projection has stripped them."""

    result = _exact_int(
        target.get("result"),
        field="terminal_target.result",
        minimum=-1,
        maximum=2,
        optional=True,
    )
    reason_raw = target.get("reason", target.get("resultReason"))
    reason = None if reason_raw is None else _norm_token(reason_raw)
    return result, reason


def _legal_action(observation: Mapping[str, Any], action: Sequence[Any]) -> list[int]:
    select = _mapping(observation.get("select"), field="observation.select")
    options = _rows(select.get("option"), field="select.option")
    minimum = _exact_int(select.get("minCount"), field="select.minCount", minimum=0)
    maximum = _exact_int(select.get("maxCount"), field="select.maxCount", minimum=0)
    assert minimum is not None and maximum is not None
    if minimum > maximum or maximum > len(options):
        raise SimulatorRuleTargetError("select bounds are invalid")
    selected: list[int] = []
    for index, value in enumerate(action):
        parsed = _exact_int(value, field=f"action[{index}]", minimum=0)
        assert parsed is not None
        if parsed >= len(options):
            raise SimulatorRuleTargetError("selected action references an absent legal option")
        selected.append(parsed)
    if len(selected) != len(set(selected)):
        raise SimulatorRuleTargetError("selected action repeats an option")
    if not minimum <= len(selected) <= maximum:
        raise SimulatorRuleTargetError("selected action violates current selection bounds")
    return selected


def _simulator_identity_matches(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    for key, expected in R298_CANONICAL_SIMULATOR.items():
        if value.get(key) != expected:
            return False
    return True


@dataclass(frozen=True)
class PromptChainStep:
    """One simulator-observed transition in a selected-action prompt chain.

    ``before`` and ``after`` may contain raw visual snapshots; the compiler
    sanitizes them before any use.  The first step carries the selected action.
    Later steps must be deterministic forced prompts and are credited back to
    that initiating action only until the next genuine strategic decision.
    """

    before: Mapping[str, Any]
    after: Mapping[str, Any]
    event_kind: str
    action: tuple[int, ...] | None = None
    forced: bool = False
    strategic_decision: bool = False
    facts: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "before": copy.deepcopy(dict(self.before)),
            "after": copy.deepcopy(dict(self.after)),
            "event_kind": str(self.event_kind),
            "forced": bool(self.forced),
            "strategic_decision": bool(self.strategic_decision),
        }
        if self.action is not None:
            result["action"] = [int(value) for value in self.action]
        if self.facts is not None:
            result["facts"] = copy.deepcopy(dict(self.facts))
        return result


@dataclass(frozen=True)
class DeterministicPromptChain:
    """A canonical, simulator-bound selected-action chain fixture/receipt."""

    root_action: tuple[int, ...]
    events: tuple[PromptChainStep, ...]
    simulator: Mapping[str, Any]
    source: str = "observed_selected_action_trajectory"
    event_log_complete: bool = True
    complete_to_next_strategic_decision: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": R298_PROMPT_CHAIN_SCHEMA,
            "version": R298_PROMPT_CHAIN_SCHEMA_VERSION,
            "simulator": copy.deepcopy(dict(self.simulator)),
            "source": str(self.source),
            "root_action": [int(value) for value in self.root_action],
            "event_log_complete": bool(self.event_log_complete),
            "complete_to_next_strategic_decision": bool(
                self.complete_to_next_strategic_decision
            ),
            "events": [event.to_dict() for event in self.events],
        }


@runtime_checkable
class PromptChainSimulator(Protocol):
    """A deliberately narrow simulator bridge for a *selected* action only."""

    canonical_simulator_identity: Mapping[str, Any]

    def resolve_selected_action_prompt_chain(
        self,
        *,
        public_observation: Mapping[str, Any],
        selected_action: Sequence[int],
    ) -> Mapping[str, Any] | DeterministicPromptChain:
        """Return an exact public deterministic chain for this selected action."""


@runtime_checkable
class PrivilegedBeliefReceiptValidator(Protocol):
    """Reserved seam for a future sealed privileged-target ledger extension.

    The target compiler can prove that a receipt refers to its normalized
    selected-action chain, but it cannot attest an arbitrary caller-provided
    SHA string against the sealed corpus.  A generic Python callback therefore
    has no authority here.  The current r298 ledger intentionally does not
    bind hidden target payloads, so all belief labels remain unavailable until
    a separately schema-pinned sealed extension is implemented.  The future
    extension has no policy, search, legal-action, or runtime authority.
    """

    def validate_privileged_belief_receipt(
        self,
        *,
        receipt: Mapping[str, Any],
        normalized_prompt_chain_sha256: str,
        public_observation_sha256: str,
        selected_action_sha256: str,
        source: str,
    ) -> bool:
        """Return true only for a sealed external receipt binding this row."""


@runtime_checkable
class SelectedActionTrajectoryReceiptValidator(Protocol):
    """Reserved verification seam for selected simulator-chain receipts.

    An observed trace is useful diagnostic evidence, but its self-declared
    simulator fields are not an immutable provenance proof.  A future
    materializer may use this narrow interface only after it has matched the
    raw-frame/trajectory object and corpus receipt.  This module deliberately
    treats any generic Python callback as *diagnostic only*: it cannot make a
    target mask trainable.  The sole trainable implementation is
    :class:`SealedSelectedActionTrajectoryValidator`, loaded through its
    checksum-bound ledger/authorization factory.
    """

    def validate_selected_action_trajectory_receipt(
        self,
        *,
        receipt: Mapping[str, Any],
        normalized_prompt_chain_sha256: str,
        public_observation_sha256: str,
        selected_action_semantic_sha256: str,
        simulator: Mapping[str, Any],
        source: str,
    ) -> bool:
        """Return true only when an immutable raw/corpus receipt binds this row."""


def _chain_from_simulator(
    simulator: Any,
    *,
    public_observation: Mapping[str, Any],
    selected_action: Sequence[int],
) -> Any:
    identity = getattr(simulator, "canonical_simulator_identity", None)
    if not _simulator_identity_matches(identity):
        raise SimulatorRuleTargetError("selected-action simulator is not bound to pinned libcg")
    for name in (
        "resolve_selected_action_prompt_chain",
        "resolve_selected_action",
        "selected_action_prompt_chain",
    ):
        method = getattr(simulator, name, None)
        if callable(method):
            try:
                return method(
                    public_observation=public_observation,
                    selected_action=list(selected_action),
                )
            except TypeError:
                # A test fixture may expose positional-only arguments.  Do not
                # broaden this into a generic simulator/search call.
                return method(public_observation, list(selected_action))
    raise SimulatorRuleTargetError("selected-action simulator exposes no prompt-chain method")


@dataclass(frozen=True)
class _ChainEvent:
    before: Mapping[str, Any]
    after: Mapping[str, Any]
    before_terminal: Mapping[str, Any]
    after_terminal: Mapping[str, Any]
    event_kind: str
    action: tuple[int, ...] | None
    forced: bool
    strategic_decision: bool
    facts: Mapping[str, Any]


@dataclass(frozen=True)
class _NormalizedChain:
    root_action: tuple[int, ...]
    root_before: Mapping[str, Any]
    root_public_observation_hash: str
    final_after: Mapping[str, Any]
    root_terminal: Mapping[str, Any]
    final_terminal: Mapping[str, Any]
    events: tuple[_ChainEvent, ...]
    simulator: Mapping[str, Any]
    source: str
    restoration_provenance: Mapping[str, Any] | None
    event_log_complete: bool
    complete_to_next_strategic_decision: bool
    chain_hash: str


def _event_mapping(value: Any, *, index: int) -> Mapping[str, Any]:
    if isinstance(value, PromptChainStep):
        return value.to_dict()
    return _mapping(value, field=f"prompt_chain.events[{index}]")


def _event_snapshot(
    event: Mapping[str, Any],
    *,
    names: Sequence[str],
    field: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    for name in names:
        if isinstance(event.get(name), Mapping):
            raw = _mapping(event.get(name), field=field)
            # The raw frame is needed only momentarily by the semantic public
            # representation to resolve a visible source/skill locator.  It
            # is never retained in the normalized chain, target hash, or
            # output record.  The adapter then removes raw serials entirely.
            return (
                raw,
                _sanitize_public_observation(raw),
                _extract_terminal_target(raw),
            )
    raise SimulatorRuleTargetError(f"{field} is missing")


def _event_action(value: Any, *, field: str, optional: bool = True) -> tuple[int, ...] | None:
    if value is None and optional:
        return None
    if not isinstance(value, (list, tuple)):
        raise SimulatorRuleTargetError(f"{field} must be an action list")
    result: list[int] = []
    for index, item in enumerate(value):
        parsed = _exact_int(item, field=f"{field}[{index}]", minimum=0)
        assert parsed is not None
        result.append(parsed)
    if len(result) != len(set(result)):
        raise SimulatorRuleTargetError(f"{field} contains duplicate option indexes")
    return tuple(result)


_EVENT_CARD_FACT_NAMES = frozenset(
    {
        "knockedoutcard",
        "targetcard",
        "pokemon",
        "card",
    }
)
_EVENT_PLAYER_FACT_NAMES = frozenset(
    {
        "victimplayerindex",
        "targetplayerindex",
        "playerindex",
    }
)
_EVENT_NUMBER_FACT_NAMES = frozenset(
    {
        "damage",
        "damagedealt",
        "damagecounters",
        "counters",
        "drawcount",
        "draws",
        "count",
        "discardcount",
        "discards",
        "benchcount",
        "benches",
        "prizeyield",
        "prizes",
    }
)
_EVENT_MODIFIER_FACT_NAMES = frozenset(
    {
        "visibleprizemodifier",
        "prizemodifier",
    }
)

_EVENT_CARD_FACT_CANONICAL = {
    "knockedoutcard": "knockedOutCard",
    "targetcard": "targetCard",
    "pokemon": "pokemon",
    "card": "card",
}
_EVENT_PLAYER_FACT_CANONICAL = {
    "victimplayerindex": "victimPlayerIndex",
    "targetplayerindex": "targetPlayerIndex",
    "playerindex": "playerIndex",
}
_EVENT_NUMBER_FACT_CANONICAL = {
    "damage": "damage",
    "damagedealt": "damageDealt",
    "damagecounters": "damageCounters",
    "counters": "counters",
    "drawcount": "drawCount",
    "draws": "draws",
    "count": "count",
    "discardcount": "discardCount",
    "discards": "discards",
    "benchcount": "benchCount",
    "benches": "benches",
    "prizeyield": "prizeYield",
    "prizes": "prizes",
}
_EVENT_MODIFIER_FACT_CANONICAL = {
    "visibleprizemodifier": "visiblePrizeModifier",
    "prizemodifier": "prizeModifier",
}


def _public_event_card(value: Any) -> Mapping[str, Any] | None:
    """Keep only card-class fields used by public Prize computation."""

    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {}
    for name in ("id", "cardId"):
        if name in value:
            parsed = _exact_int(value[name], field=f"event card.{name}", minimum=0)
            assert parsed is not None
            result[name] = parsed
    for name in ("ex", "megaEx", "mega_ex"):
        if name in value:
            parsed = _optional_bool(value[name], field=f"event card.{name}")
            if parsed is not None:
                result[name] = parsed
    for name in ("prizeYield", "prize_yield", "prizes"):
        if name in value:
            parsed = _exact_int(
                value[name], field=f"event card.{name}", minimum=0
            )
            assert parsed is not None
            result[name] = parsed
    return result


def _public_event_modifier(value: Any) -> Any:
    """Keep structured public modifier fields and discard opaque text/state."""

    if not isinstance(value, Mapping):
        parsed = _exact_int(value, field="event visible prize modifier")
        assert parsed is not None
        return parsed
    result: dict[str, Any] = {}
    for name in (
        "exact_yield", "exactYield", "yield", "delta", "prize_delta",
        "prizeDelta", "reduction", "prize_reduction", "prizeReduction",
    ):
        if name in value:
            minimum = 0 if name in {
                "exact_yield", "exactYield", "yield", "reduction",
                "prize_reduction", "prizeReduction",
            } else None
            parsed = _exact_int(
                value[name], field=f"event visible prize modifier.{name}", minimum=minimum
            )
            assert parsed is not None
            result[name] = parsed
    return result


def _public_event_facts(value: Mapping[str, Any]) -> dict[str, Any]:
    """Drop all non-public/unconsumed facts before target materialization.

    Event receipts may carry richer engine diagnostics.  This compiler only
    retains the narrow typed facts needed for its declared labels.  In
    particular, an opponent hand/deck/prize identity cannot hide in a target
    hash, output payload, or future materializer side channel.
    """

    result: dict[str, Any] = {}

    def retain(canonical_name: str, public_value: Any) -> None:
        existing = result.get(canonical_name)
        if existing is not None and existing != public_value:
            raise SimulatorRuleTargetError(
                f"event facts disagree on canonical public field {canonical_name}"
            )
        result[canonical_name] = public_value

    for raw_name, raw_value in value.items():
        name = _norm_token(raw_name)
        if name in _EVENT_CARD_FACT_NAMES:
            card = _public_event_card(raw_value)
            if card is not None:
                retain(_EVENT_CARD_FACT_CANONICAL[name], card)
        elif name in _EVENT_PLAYER_FACT_NAMES:
            player = _exact_int(
                raw_value, field=f"event fact {raw_name}", minimum=0, maximum=1
            )
            assert player is not None
            retain(_EVENT_PLAYER_FACT_CANONICAL[name], player)
        elif name in _EVENT_NUMBER_FACT_NAMES:
            number = _finite_number(raw_value, field=f"event fact {raw_name}")
            assert number is not None
            retain(_EVENT_NUMBER_FACT_CANONICAL[name], number)
        elif name in _EVENT_MODIFIER_FACT_NAMES:
            retain(
                _EVENT_MODIFIER_FACT_CANONICAL[name],
                _public_event_modifier(raw_value),
            )
        elif name == "forced":
            forced = _optional_bool(raw_value, field="event fact forced")
            if forced is not None:
                retain("forced", forced)
    return result


def _sha256_digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise SimulatorRuleTargetError(f"{field} must be a sha256: hexadecimal digest")
    return value


def _immutable_json_file(
    path: Path | str,
    *,
    field: str,
) -> tuple[dict[str, Any], str]:
    """Read one regular, content-addressed JSON receipt without mutation."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise SimulatorRuleTargetError(f"{field} must be a regular immutable JSON file")
    try:
        raw = candidate.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SimulatorRuleTargetError(f"cannot read {field}") from exc
    row = _mapping(payload, field=field)
    return copy.deepcopy(dict(row)), "sha256:" + hashlib.sha256(raw).hexdigest()


def _sealed_trajectory_entry_key(
    value: Mapping[str, Any],
    *,
    field: str,
) -> tuple[str, ...]:
    """Normalize one ledger entry without retaining a raw observation."""

    row = _mapping(value, field=field)
    if row.get("schema") != R298_TRAJECTORY_LEDGER_ENTRY_SCHEMA:
        raise SimulatorRuleTargetError(f"{field} schema mismatch")
    source = row.get("source")
    if source not in {
        "observed_selected_action_trajectory",
        "restored_public_seed_simulator",
    }:
        raise SimulatorRuleTargetError(f"{field} has an unsupported source")
    simulator_hash = _sha256_digest(
        row.get("canonical_simulator_sha256"),
        field=f"{field}.canonical_simulator_sha256",
    )
    if simulator_hash != _digest(R298_CANONICAL_SIMULATOR):
        raise SimulatorRuleTargetError(f"{field} does not bind the pinned simulator")
    common = (
        str(source),
        simulator_hash,
        _sha256_digest(
            row.get("normalized_prompt_chain_sha256"),
            field=f"{field}.normalized_prompt_chain_sha256",
        ),
        _sha256_digest(
            row.get("public_observation_sha256"),
            field=f"{field}.public_observation_sha256",
        ),
        _sha256_digest(
            row.get("selected_action_semantic_sha256"),
            field=f"{field}.selected_action_semantic_sha256",
        ),
        _sha256_digest(
            row.get("trajectory_receipt_sha256"),
            field=f"{field}.trajectory_receipt_sha256",
        ),
        _sha256_digest(
            row.get("corpus_receipt_sha256"),
            field=f"{field}.corpus_receipt_sha256",
        ),
    )
    if source == "observed_selected_action_trajectory":
        return common + (
            _sha256_digest(
                row.get("raw_frame_receipt_sha256"),
                field=f"{field}.raw_frame_receipt_sha256",
            ),
        )
    return common + (
        _sha256_digest(
            row.get("state_restoration_receipt_sha256"),
            field=f"{field}.state_restoration_receipt_sha256",
        ),
        _sha256_digest(
            row.get("public_seed_receipt_sha256"),
            field=f"{field}.public_seed_receipt_sha256",
        ),
    )


def _sealed_trajectory_key_from_receipt(
    receipt: Mapping[str, Any],
    *,
    normalized_prompt_chain_sha256: str,
    public_observation_sha256: str,
    selected_action_semantic_sha256: str,
    simulator: Mapping[str, Any],
    source: str,
) -> tuple[str, ...]:
    """Build the ledger lookup key from an already normalized chain receipt."""

    if source not in {
        "observed_selected_action_trajectory",
        "restored_public_seed_simulator",
    } or not _simulator_identity_matches(simulator):
        raise SimulatorRuleTargetError("sealed trajectory validation has foreign simulator/source")
    common = (
        source,
        _digest(R298_CANONICAL_SIMULATOR),
        _sha256_digest(
            normalized_prompt_chain_sha256,
            field="sealed trajectory normalized_prompt_chain_sha256",
        ),
        _sha256_digest(
            public_observation_sha256,
            field="sealed trajectory public_observation_sha256",
        ),
        _sha256_digest(
            selected_action_semantic_sha256,
            field="sealed trajectory selected_action_semantic_sha256",
        ),
        _sha256_digest(
            receipt.get("trajectory_receipt_sha256"),
            field="sealed trajectory receipt.trajectory_receipt_sha256",
        ),
        _sha256_digest(
            receipt.get("corpus_receipt_sha256"),
            field="sealed trajectory receipt.corpus_receipt_sha256",
        ),
    )
    if source == "observed_selected_action_trajectory":
        return common + (
            _sha256_digest(
                receipt.get("raw_frame_receipt_sha256"),
                field="sealed trajectory receipt.raw_frame_receipt_sha256",
            ),
        )
    return common + (
        _sha256_digest(
            receipt.get("state_restoration_receipt_sha256"),
            field="sealed trajectory receipt.state_restoration_receipt_sha256",
        ),
        _sha256_digest(
            receipt.get("public_seed_receipt_sha256"),
            field="sealed trajectory receipt.public_seed_receipt_sha256",
        ),
    )


def _payload_sha256(
    value: Mapping[str, Any],
    *,
    claim_field: str,
    field: str,
) -> str:
    """Verify a canonical payload checksum stored outside its own preimage."""

    row = _mapping(value, field=field)
    claimed = _sha256_digest(row.get(claim_field), field=f"{field}.{claim_field}")
    payload = copy.deepcopy(dict(row))
    payload.pop(claim_field, None)
    if claimed != _digest(payload):
        raise SimulatorRuleTargetError(f"{field} payload checksum mismatch")
    return claimed


def _require_receipt_fields(
    value: Mapping[str, Any],
    *,
    fields: Sequence[str],
    field: str,
) -> None:
    """Require the r5 fields that establish an external authority boundary.

    The dedicated contract owns the complete closed receipt inventories.  This
    narrow target compiler validates the subset that establishes its own
    safety boundary instead of accepting a SHA-shaped reference to an
    uninspected external payload.
    """

    missing = [name for name in fields if name not in value]
    if missing:
        raise SimulatorRuleTargetError(
            f"{field} lacks required r5 fields: {', '.join(sorted(missing))}"
        )


def _load_r5_schema_freeze_receipt(
    path: Path | str,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Verify the r5 schema-freeze receipt relevant to target training.

    Revision 4's static schema bytes may remain historical evidence, but the
    revision-5 contract requires a new freeze receipt that binds this exact
    target digest and the zero-bypass/public-information constraints.
    """

    expected_sha256 = _sha256_digest(
        expected_sha256, field="expected revision-5 schema-freeze receipt sha256"
    )
    receipt, actual_sha256 = _immutable_json_file(
        path, field="revision-5 schema-freeze receipt"
    )
    if actual_sha256 != expected_sha256:
        raise SimulatorRuleTargetError("revision-5 schema-freeze receipt digest mismatch")
    _require_receipt_fields(
        receipt,
        fields=(
            "schema",
            "goal_contract_path",
            "goal_contract_sha256",
            "goal_revision",
            "target_schema_id",
            "target_schema_sha256",
            "q3_bench_only",
            "q5_q6_trace_only_zero",
            "public_information_contract_passed",
            "zero_bypass_receipt_sha256",
            "layer_off_bit_identical_baseline_logits",
        ),
        field="revision-5 schema-freeze receipt",
    )
    if (
        receipt.get("schema") != R298_R5_SCHEMA_FREEZE_RECEIPT_SCHEMA
        or receipt.get("goal_contract_path") != R298_CANONICAL_CONTRACT_PATH
        or receipt.get("goal_contract_sha256") != R298_CANONICAL_CONTRACT_SHA256
        or receipt.get("goal_revision") != R298_CANONICAL_GOAL_REVISION
        or receipt.get("target_schema_id") != R298_RULE_TARGET_SCHEMA
        or receipt.get("target_schema_sha256") != R298_RULE_TARGET_SCHEMA_DIGEST
        or receipt.get("q3_bench_only") is not True
        or receipt.get("q5_q6_trace_only_zero") is not True
        or receipt.get("public_information_contract_passed") is not True
        or receipt.get("layer_off_bit_identical_baseline_logits") is not True
    ):
        raise SimulatorRuleTargetError("revision-5 schema-freeze receipt is not target-safe")
    _sha256_digest(
        receipt.get("zero_bypass_receipt_sha256"),
        field="revision-5 schema-freeze receipt.zero_bypass_receipt_sha256",
    )
    return receipt, actual_sha256


def _load_r5_handoff_activation_receipt(
    path: Path | str,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Verify that r5—not a caller string—authorizes candidate training.

    This deliberately does not grant policy/serving authority.  It only
    proves that the exact r5 clean-boundary activation made staged corpus
    bytes training eligible for the derivative's frozen-parent candidate.
    """

    expected_sha256 = _sha256_digest(
        expected_sha256,
        field="expected revision-5 handoff activation receipt sha256",
    )
    receipt, actual_sha256 = _immutable_json_file(
        path, field="revision-5 handoff activation receipt"
    )
    if actual_sha256 != expected_sha256:
        raise SimulatorRuleTargetError("revision-5 handoff activation receipt digest mismatch")
    _require_receipt_fields(
        receipt,
        fields=(
            "schema",
            "goal_contract_path",
            "goal_contract_sha256",
            "goal_revision",
            "root_owner_revision",
            "old_services_paused_via_systemd_user",
            "old_services_inactive_verified",
            "shared_kaggle_queue_service_unchanged",
            "new_lineage_id",
            "staged_shards_training_eligible",
            "blackwell_preflight_receipt_sha256",
            "rollback_plan_receipt_sha256",
            "no_concurrent_r274_training_or_collection",
            "no_serving_selector_or_submission_activation",
        ),
        field="revision-5 handoff activation receipt",
    )
    if (
        receipt.get("schema") != R298_R5_HANDOFF_ACTIVATION_RECEIPT_SCHEMA
        or receipt.get("goal_contract_path") != R298_CANONICAL_CONTRACT_PATH
        or receipt.get("goal_contract_sha256") != R298_CANONICAL_CONTRACT_SHA256
        or receipt.get("goal_revision") != R298_CANONICAL_GOAL_REVISION
        or receipt.get("root_owner_revision") != R298_ROOT_OWNER_REVISION
        or receipt.get("old_services_paused_via_systemd_user") is not True
        or receipt.get("old_services_inactive_verified") is not True
        or receipt.get("shared_kaggle_queue_service_unchanged") is not True
        or receipt.get("new_lineage_id") != R298_R5_DERIVATIVE_LINEAGE_ID
        or receipt.get("staged_shards_training_eligible") is not True
        or receipt.get("no_concurrent_r274_training_or_collection") is not True
        or receipt.get("no_serving_selector_or_submission_activation") is not True
    ):
        raise SimulatorRuleTargetError("revision-5 handoff activation receipt is not target-safe")
    for name in ("blackwell_preflight_receipt_sha256", "rollback_plan_receipt_sha256"):
        _sha256_digest(receipt.get(name), field=f"revision-5 handoff activation receipt.{name}")
    return receipt, actual_sha256


_R298_SEALED_VALIDATOR_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True)
class SealedSelectedActionTrajectoryValidator:
    """Receipt-bound authority that alone can arm selected-action targets.

    The constructor is intentionally private by convention: callers obtain an
    instance from :func:`load_sealed_selected_action_trajectory_validator`,
    which verifies regular files, externally pinned file hashes, the exact
    r298 schema, the 30-day corpus binding, post-census authorization, the
    revision-5 schema freeze, and the revision-5 clean-boundary handoff
    activation.  The instance stores only digest tuples, never raw frames,
    cards, hidden state, or policy features.
    """

    ledger_file_sha256: str
    authorization_file_sha256: str
    raw_corpus_receipt_sha256: str
    frozen_schema_manifest_sha256: str
    schema_freeze_receipt_sha256: str
    branch_support_receipt_sha256: str
    training_gate_report_sha256: str
    training_handoff_activation_receipt_sha256: str
    _entry_keys: frozenset[tuple[str, ...]]
    _construction_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._construction_token is not _R298_SEALED_VALIDATOR_CONSTRUCTION_TOKEN:
            raise SimulatorRuleTargetError(
                "sealed trajectory validator must be created by its receipt loader"
            )

    @property
    def validator_kind(self) -> str:
        return R298_SEALED_TRAJECTORY_VALIDATOR_KIND

    def provenance(self) -> dict[str, Any]:
        return {
            "validator_kind": self.validator_kind,
            "ledger_file_sha256": self.ledger_file_sha256,
            "authorization_file_sha256": self.authorization_file_sha256,
            "raw_corpus_receipt_sha256": self.raw_corpus_receipt_sha256,
            "frozen_schema_manifest_sha256": self.frozen_schema_manifest_sha256,
            "schema_freeze_receipt_sha256": self.schema_freeze_receipt_sha256,
            "branch_support_receipt_sha256": self.branch_support_receipt_sha256,
            "training_gate_report_sha256": self.training_gate_report_sha256,
            "training_handoff_activation_receipt_sha256": (
                self.training_handoff_activation_receipt_sha256
            ),
            "training_host": R298_R5_TRAINING_HOST,
            "root_owner_revision": R298_ROOT_OWNER_REVISION,
            "entry_count": len(self._entry_keys),
            "target_only": True,
            "may_drive_runtime": False,
        }

    def validate_selected_action_trajectory_receipt(
        self,
        *,
        receipt: Mapping[str, Any],
        normalized_prompt_chain_sha256: str,
        public_observation_sha256: str,
        selected_action_semantic_sha256: str,
        simulator: Mapping[str, Any],
        source: str,
    ) -> bool:
        try:
            key = _sealed_trajectory_key_from_receipt(
                receipt,
                normalized_prompt_chain_sha256=normalized_prompt_chain_sha256,
                public_observation_sha256=public_observation_sha256,
                selected_action_semantic_sha256=selected_action_semantic_sha256,
                simulator=simulator,
                source=source,
            )
        except SimulatorRuleTargetError:
            return False
        return key in self._entry_keys


def load_sealed_selected_action_trajectory_validator(
    ledger_path: Path | str,
    authorization_path: Path | str,
    *,
    schema_freeze_receipt_path: Path | str,
    training_handoff_activation_receipt_path: Path | str,
    expected_authorization_file_sha256: str,
    expected_raw_corpus_receipt_sha256: str,
    expected_frozen_schema_manifest_sha256: str,
    expected_schema_freeze_receipt_sha256: str,
    expected_branch_support_receipt_sha256: str,
    expected_training_gate_report_sha256: str,
    expected_training_handoff_activation_receipt_sha256: str,
) -> SealedSelectedActionTrajectoryValidator:
    """Load the sole r298 validator permitted to arm target masks.

    The expected digests and immutable receipt paths are supplied by the
    receipt-owning materializer/training gate, not inferred from temporary
    caller data.  This factory is deliberately unavailable to raw mappings or
    callback objects: an authorization file must content-address the ledger
    bytes and bind the current target schema, exact raw-corpus receipt, frozen
    schema, revision-5 schema freeze, branch-support receipt, passed
    training-gate report, and revision-5 handoff activation.  It grants no
    runtime/selector authority.
    """

    expected_authorization_file_sha256 = _sha256_digest(
        expected_authorization_file_sha256,
        field="expected authorization file sha256",
    )
    expected_raw_corpus_receipt_sha256 = _sha256_digest(
        expected_raw_corpus_receipt_sha256,
        field="expected raw corpus receipt sha256",
    )
    expected_frozen_schema_manifest_sha256 = _sha256_digest(
        expected_frozen_schema_manifest_sha256,
        field="expected frozen schema manifest sha256",
    )
    expected_schema_freeze_receipt_sha256 = _sha256_digest(
        expected_schema_freeze_receipt_sha256,
        field="expected revision-5 schema-freeze receipt sha256",
    )
    expected_branch_support_receipt_sha256 = _sha256_digest(
        expected_branch_support_receipt_sha256,
        field="expected branch support receipt sha256",
    )
    expected_training_gate_report_sha256 = _sha256_digest(
        expected_training_gate_report_sha256,
        field="expected training gate report sha256",
    )
    expected_training_handoff_activation_receipt_sha256 = _sha256_digest(
        expected_training_handoff_activation_receipt_sha256,
        field="expected revision-5 handoff activation receipt sha256",
    )
    _load_r5_schema_freeze_receipt(
        schema_freeze_receipt_path,
        expected_sha256=expected_schema_freeze_receipt_sha256,
    )
    _load_r5_handoff_activation_receipt(
        training_handoff_activation_receipt_path,
        expected_sha256=expected_training_handoff_activation_receipt_sha256,
    )
    ledger, ledger_file_sha256 = _immutable_json_file(
        ledger_path, field="sealed selected-action trajectory ledger"
    )
    authorization, authorization_file_sha256 = _immutable_json_file(
        authorization_path, field="selected-action target training authorization"
    )
    if authorization_file_sha256 != expected_authorization_file_sha256:
        raise SimulatorRuleTargetError("trajectory authorization file digest does not match Phase-D pin")

    if ledger.get("schema") != R298_TRAJECTORY_LEDGER_SCHEMA:
        raise SimulatorRuleTargetError("trajectory ledger schema mismatch")
    if ledger.get("version") != R298_TRAJECTORY_LEDGER_SCHEMA_VERSION:
        raise SimulatorRuleTargetError("trajectory ledger version mismatch")
    if ledger.get("status") != R298_R5_TRAJECTORY_LEDGER_STATUS:
        raise SimulatorRuleTargetError("trajectory ledger is not sealed for the exact 30-day corpus")
    if ledger.get("target_only") is not True or ledger.get("may_drive_runtime") is not False:
        raise SimulatorRuleTargetError("trajectory ledger has runtime authority")
    if ledger.get("runtime_wired") is not False or ledger.get("training_authorization") is not False:
        raise SimulatorRuleTargetError("trajectory ledger may not authorize training itself")
    if (
        ledger.get("goal_sha256") != R298_CANONICAL_GOAL_SHA256
        or ledger.get("contract_sha256") != R298_CANONICAL_CONTRACT_SHA256
        or ledger.get("goal_revision") != R298_CANONICAL_GOAL_REVISION
        or ledger.get("root_owner_revision") != R298_ROOT_OWNER_REVISION
        or ledger.get("production_typed_source_sha256")
        != R298_PRODUCTION_TYPED_SOURCE_SHA256
        or ledger.get("target_schema") != R298_RULE_TARGET_SCHEMA
        or ledger.get("target_schema_digest") != R298_RULE_TARGET_SCHEMA_DIGEST
        or ledger.get("raw_corpus_receipt_sha256") != expected_raw_corpus_receipt_sha256
        or ledger.get("frozen_schema_manifest_sha256") != expected_frozen_schema_manifest_sha256
        or ledger.get("schema_freeze_receipt_sha256")
        != expected_schema_freeze_receipt_sha256
        or ledger.get("complete_30_utc_days") is not True
        or ledger.get("revision_4_predecessor_evidence_only") is not True
        or ledger.get("blind_revision_4_substitution_allowed") is not False
    ):
        raise SimulatorRuleTargetError("trajectory ledger has stale or incomplete provenance")
    entries = _rows(ledger.get("entries"), field="trajectory ledger.entries")
    if not entries:
        raise SimulatorRuleTargetError("trajectory ledger may not be empty")
    entry_keys = [
        _sealed_trajectory_entry_key(
            _mapping(entry, field=f"trajectory ledger.entries[{index}]"),
            field=f"trajectory ledger.entries[{index}]",
        )
        for index, entry in enumerate(entries)
    ]
    if len(set(entry_keys)) != len(entry_keys):
        raise SimulatorRuleTargetError("trajectory ledger has duplicate receipt bindings")
    canonical_entries = [
        copy.deepcopy(dict(_mapping(entry, field="trajectory ledger entry")))
        for entry in entries
    ]
    if ledger.get("entries_sha256") != _digest(
        sorted(canonical_entries, key=_canonical_json)
    ):
        raise SimulatorRuleTargetError("trajectory ledger entry inventory checksum mismatch")
    ledger_payload_sha256 = _payload_sha256(
        ledger,
        claim_field="ledger_payload_sha256",
        field="trajectory ledger",
    )

    if authorization.get("schema") != R298_TRAJECTORY_TRAINING_AUTHORIZATION_SCHEMA:
        raise SimulatorRuleTargetError("trajectory training authorization schema mismatch")
    if authorization.get("version") != R298_TRAJECTORY_TRAINING_AUTHORIZATION_SCHEMA_VERSION:
        raise SimulatorRuleTargetError("trajectory training authorization version mismatch")
    if authorization.get("status") != R298_R5_TRAJECTORY_TRAINING_AUTHORIZATION_STATUS:
        raise SimulatorRuleTargetError("trajectory training authorization is not passed")
    if (
        authorization.get("target_only") is not True
        or authorization.get("may_drive_runtime") is not False
    ):
        raise SimulatorRuleTargetError("trajectory training authorization has runtime authority")
    if (
        authorization.get("goal_sha256") != R298_CANONICAL_GOAL_SHA256
        or authorization.get("contract_sha256") != R298_CANONICAL_CONTRACT_SHA256
        or authorization.get("goal_revision") != R298_CANONICAL_GOAL_REVISION
        or authorization.get("root_owner_revision") != R298_ROOT_OWNER_REVISION
        or authorization.get("production_typed_source_sha256")
        != R298_PRODUCTION_TYPED_SOURCE_SHA256
        or authorization.get("target_schema") != R298_RULE_TARGET_SCHEMA
        or authorization.get("target_schema_digest") != R298_RULE_TARGET_SCHEMA_DIGEST
        or authorization.get("raw_corpus_receipt_sha256")
        != expected_raw_corpus_receipt_sha256
        or authorization.get("frozen_schema_manifest_sha256")
        != expected_frozen_schema_manifest_sha256
        or authorization.get("schema_freeze_receipt_sha256")
        != expected_schema_freeze_receipt_sha256
        or authorization.get("branch_support_receipt_sha256")
        != expected_branch_support_receipt_sha256
        or authorization.get("training_gate_report_sha256")
        != expected_training_gate_report_sha256
        or authorization.get("training_handoff_activation_receipt_sha256")
        != expected_training_handoff_activation_receipt_sha256
        or authorization.get("trajectory_ledger_file_sha256") != ledger_file_sha256
        or authorization.get("trajectory_ledger_payload_sha256") != ledger_payload_sha256
        or authorization.get("revision_4_predecessor_evidence_only") is not True
        or authorization.get("blind_revision_4_substitution_allowed") is not False
    ):
        raise SimulatorRuleTargetError("trajectory training authorization has stale receipt bindings")
    if (
        authorization.get("candidate_training_allowed") is not True
        or authorization.get("selected_action_target_training_allowed") is not True
        or authorization.get("training_host") != R298_R5_TRAINING_HOST
        or authorization.get("revision_5_handoff_activation_bound") is not True
        or authorization.get("runtime_wired") is not False
        or authorization.get("production_serving_authority") is not False
        or authorization.get("elmo_only_nonproduction") is True
    ):
        raise SimulatorRuleTargetError("trajectory training authorization violates revision-5 authority")
    _payload_sha256(
        authorization,
        claim_field="authorization_payload_sha256",
        field="trajectory training authorization",
    )
    return SealedSelectedActionTrajectoryValidator(
        ledger_file_sha256=ledger_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
        raw_corpus_receipt_sha256=expected_raw_corpus_receipt_sha256,
        frozen_schema_manifest_sha256=expected_frozen_schema_manifest_sha256,
        schema_freeze_receipt_sha256=expected_schema_freeze_receipt_sha256,
        branch_support_receipt_sha256=expected_branch_support_receipt_sha256,
        training_gate_report_sha256=expected_training_gate_report_sha256,
        training_handoff_activation_receipt_sha256=(
            expected_training_handoff_activation_receipt_sha256
        ),
        _entry_keys=frozenset(entry_keys),
        _construction_token=_R298_SEALED_VALIDATOR_CONSTRUCTION_TOKEN,
    )


def _restoration_provenance(
    raw: Mapping[str, Any],
    *,
    public_observation_hash: str,
    selected_action: Sequence[int],
) -> dict[str, Any]:
    """Validate the extra immutable proof required for restored trajectories.

    A string saying "restored" is not enough: the separate corpus must bind
    the original selected public state/action, a public-information seed, and
    immutable state-restoration/trajectory receipts.  The compiler does not
    consume a hidden state from this object; it verifies only the proof shape
    and serializes its opaque hashes into target-only provenance.
    """

    value = raw.get("restoration_provenance")
    row = _mapping(value, field="prompt_chain.restoration_provenance")
    if row.get("schema") != R298_TARGET_PROVENANCE_SCHEMA:
        raise SimulatorRuleTargetError("restored prompt chain provenance schema mismatch")
    if row.get("version") != R298_TARGET_PROVENANCE_SCHEMA_VERSION:
        raise SimulatorRuleTargetError("restored prompt chain provenance version mismatch")
    if row.get("targeted_libcg_corpus") is not True:
        raise SimulatorRuleTargetError("restored prompt chain is not a targeted libcg corpus")
    if row.get("public_information_seed_only") is not True:
        raise SimulatorRuleTargetError("restored prompt chain seed is not public-information only")
    if row.get("counterfactual") is not False:
        raise SimulatorRuleTargetError("restored prompt chain must still be selected-action only")
    if row.get("public_observation_sha256") != public_observation_hash:
        raise SimulatorRuleTargetError(
            "restored prompt chain provenance does not bind the selected public observation"
        )
    selected_action_hash = _digest({"selected_action": [int(value) for value in selected_action]})
    if row.get("selected_action_sha256") != selected_action_hash:
        raise SimulatorRuleTargetError(
            "restored prompt chain provenance does not bind the selected action"
        )
    return {
        "schema": R298_TARGET_PROVENANCE_SCHEMA,
        "version": R298_TARGET_PROVENANCE_SCHEMA_VERSION,
        "targeted_libcg_corpus": True,
        "public_information_seed_only": True,
        "counterfactual": False,
        "public_observation_sha256": public_observation_hash,
        "selected_action_sha256": selected_action_hash,
        "state_restoration_receipt_sha256": _sha256_digest(
            row.get("state_restoration_receipt_sha256"),
            field="restoration.state_restoration_receipt_sha256",
        ),
        "public_seed_receipt_sha256": _sha256_digest(
            row.get("public_seed_receipt_sha256"),
            field="restoration.public_seed_receipt_sha256",
        ),
        "trajectory_receipt_sha256": _sha256_digest(
            row.get("trajectory_receipt_sha256"),
            field="restoration.trajectory_receipt_sha256",
        ),
        "corpus_receipt_sha256": _sha256_digest(
            row.get("corpus_receipt_sha256"),
            field="restoration.corpus_receipt_sha256",
        ),
    }


def _normalize_chain(
    chain: Any,
    *,
    root_observation: Mapping[str, Any],
    selected_action: Sequence[int],
) -> _NormalizedChain:
    if isinstance(chain, DeterministicPromptChain):
        raw = chain.to_dict()
    else:
        raw = _mapping(chain, field="prompt_chain")
    if raw.get("schema") != R298_PROMPT_CHAIN_SCHEMA:
        raise SimulatorRuleTargetError("prompt chain schema mismatch")
    if raw.get("version") != R298_PROMPT_CHAIN_SCHEMA_VERSION:
        raise SimulatorRuleTargetError("prompt chain version mismatch")
    simulator = _mapping(raw.get("simulator"), field="prompt_chain.simulator")
    if not _simulator_identity_matches(simulator):
        raise SimulatorRuleTargetError("prompt chain is not bound to the pinned libcg identity")
    source = str(raw.get("source") or "")
    if source not in {
        "observed_selected_action_trajectory",
        "restored_public_seed_simulator",
    }:
        raise SimulatorRuleTargetError("prompt chain source is not an allowed selected-action provenance")
    if source == "observed_selected_action_trajectory" and raw.get("restoration_provenance") is not None:
        raise SimulatorRuleTargetError("observed chain may not carry restoration provenance")
    if raw.get("event_log_complete") is not True:
        raise SimulatorRuleTargetError("prompt chain does not prove a complete event log")
    if raw.get("complete_to_next_strategic_decision") is not True:
        raise SimulatorRuleTargetError("prompt chain does not reach the next strategic boundary")
    root_action = _event_action(raw.get("root_action"), field="prompt_chain.root_action", optional=False)
    assert root_action is not None
    selected_tuple = tuple(int(value) for value in selected_action)
    if root_action != selected_tuple:
        raise SimulatorRuleTargetError("prompt chain action differs from selected legal action")

    # Keep the raw root only at this call boundary.  The adapter needs it to
    # resolve a visible source locator before it strips serials; all retained
    # snapshots below are already public/serial-free projections.
    root_public_hash = public_observation_fingerprint(root_observation)
    root_public = _sanitize_public_observation(root_observation)
    restoration = (
        _restoration_provenance(
            raw,
            public_observation_hash=root_public_hash,
            selected_action=root_action,
        )
        if source == "restored_public_seed_simulator"
        else None
    )
    events_raw = _rows(raw.get("events"), field="prompt_chain.events")
    if not events_raw:
        raise SimulatorRuleTargetError("prompt chain has no selected-action event")
    events: list[_ChainEvent] = []
    prior_after: Mapping[str, Any] | None = None
    prior_after_public_hash: str | None = None
    prior_after_terminal: Mapping[str, Any] | None = None
    for index, value in enumerate(events_raw):
        event = _event_mapping(value, index=index)
        before_raw, before, before_terminal = _event_snapshot(
            event,
            names=("before", "before_observation", "observation"),
            field=f"prompt_chain.events[{index}].before",
        )
        after_raw, after, after_terminal = _event_snapshot(
            event,
            names=("after", "after_observation", "successor"),
            field=f"prompt_chain.events[{index}].after",
        )
        before_public_hash = public_observation_fingerprint(before_raw)
        after_public_hash = public_observation_fingerprint(after_raw)
        if index == 0:
            if before_public_hash != root_public_hash:
                raise SimulatorRuleTargetError("prompt-chain root public state differs from selected decision")
        elif prior_after is not None and before_public_hash != prior_after_public_hash:
            raise SimulatorRuleTargetError("prompt-chain public transitions are not contiguous")
        raw_action = _event_action(event.get("action"), field=f"prompt_chain.events[{index}].action")
        if index == 0 and raw_action != root_action:
            raise SimulatorRuleTargetError("first prompt-chain event does not carry the selected action")
        if index > 0 and raw_action is not None:
            raise SimulatorRuleTargetError("forced prompt-chain event may not invent another selected action")
        forced = _optional_bool(event.get("forced"), field=f"prompt_chain.events[{index}].forced")
        strategic = _optional_bool(
            event.get("strategic_decision"),
            field=f"prompt_chain.events[{index}].strategic_decision",
        )
        if index == 0:
            if forced is True:
                raise SimulatorRuleTargetError("selected root action cannot be marked forced")
        else:
            if forced is not True and strategic is not True:
                raise SimulatorRuleTargetError("post-action event is neither forced nor a strategic boundary")
        if strategic is True and index != len(events_raw) - 1:
            raise SimulatorRuleTargetError("prompt chain continues after a genuine strategic decision")
        if event.get("chance") is True or event.get("hidden_information") is True:
            raise SimulatorRuleTargetError("prompt chain crosses an unresolved chance or hidden-information boundary")
        kind = _norm_token(event.get("event_kind", event.get("kind", "")))
        if not kind:
            raise SimulatorRuleTargetError("prompt-chain event has no kind")
        facts_raw = event.get("facts", event.get("effects", {}))
        if facts_raw is None:
            facts_raw = {}
        facts = _public_event_facts(
            _mapping(facts_raw, field=f"prompt_chain.events[{index}].facts")
        )
        events.append(
            _ChainEvent(
                before=before,
                after=after,
                before_terminal=before_terminal,
                after_terminal=after_terminal,
                event_kind=kind,
                action=raw_action,
                forced=bool(forced),
                strategic_decision=bool(strategic),
                facts=facts,
            )
        )
        prior_after = after
        prior_after_public_hash = after_public_hash
        prior_after_terminal = after_terminal
    assert prior_after is not None and prior_after_terminal is not None
    final_result, _reason = _terminal_result_from_target(prior_after_terminal)
    if final_result in {None, -1} and not events[-1].strategic_decision:
        raise SimulatorRuleTargetError("prompt chain ends before terminal state or next strategic decision")
    normalized_payload = {
        "schema": R298_PROMPT_CHAIN_SCHEMA,
        "version": R298_PROMPT_CHAIN_SCHEMA_VERSION,
        "simulator": dict(simulator),
        "source": source,
        "root_action": list(root_action),
        "events": [
            {
                "before": event.before,
                "after": event.after,
                "before_terminal": event.before_terminal,
                "after_terminal": event.after_terminal,
                "event_kind": event.event_kind,
                "action": None if event.action is None else list(event.action),
                "forced": event.forced,
                "strategic_decision": event.strategic_decision,
                "facts": event.facts,
            }
            for event in events
        ],
    }
    return _NormalizedChain(
        root_action=root_action,
        root_before=root_public,
        root_public_observation_hash=root_public_hash,
        final_after=prior_after,
        root_terminal=events[0].before_terminal,
        final_terminal=prior_after_terminal,
        events=tuple(events),
        simulator=dict(simulator),
        source=source,
        restoration_provenance=restoration,
        event_log_complete=True,
        complete_to_next_strategic_decision=True,
        chain_hash=_digest(normalized_payload),
    )


def _masked_vector(layout: Sequence[str]) -> dict[str, list[Any]]:
    return {"layout": list(layout), "values": [0.0] * len(layout), "mask": [False] * len(layout)}


def _set_vector(target: dict[str, list[Any]], name: str, value: Any) -> None:
    try:
        index = target["layout"].index(name)
    except ValueError as exc:
        raise SimulatorRuleTargetError(f"target vector lacks {name}") from exc
    number = _finite_number(value, field=f"target.{name}")
    assert number is not None
    target["values"][index] = number
    target["mask"][index] = True


def _first_present(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping and mapping.get(name) is not None:
            return mapping.get(name)
    return None


def _event_number(
    event: _ChainEvent,
    names: Sequence[str],
    *,
    optional: bool = True,
) -> float | None:
    value = _first_present(event.facts, names)
    if value is None:
        value = _first_present({"event_kind": event.event_kind}, ())
    if value is None:
        return None if optional else _finite_number(value, field="event fact")
    return _finite_number(value, field=f"event.{event.event_kind}.{names[0]}")


def _event_card(event: _ChainEvent) -> Mapping[str, Any] | None:
    value = _first_present(
        event.facts,
        (
            "knocked_out_card",
            "knockedOutCard",
            "target_card",
            "targetCard",
            "pokemon",
            "card",
        ),
    )
    return value if isinstance(value, Mapping) else None


def _event_player(event: _ChainEvent, names: Sequence[str]) -> int | None:
    value = _first_present(event.facts, names)
    if value is None:
        return None
    parsed = _exact_int(value, field=f"event.{event.event_kind}.{names[0]}", minimum=0, maximum=1)
    return parsed


def _card_id(card: Mapping[str, Any] | None) -> int | None:
    if not isinstance(card, Mapping):
        return None
    raw = card.get("id", card.get("cardId"))
    try:
        return _exact_int(raw, field="card.id", minimum=0, optional=True)
    except SimulatorRuleTargetError:
        return None


@dataclass(frozen=True)
class _TestOnlyCatalog:
    """Narrow wrapper that makes fixture metadata opt-in and non-ambient."""

    value: Any


def _catalog_is_eligible(catalog: Any) -> bool:
    """Allow only adapter-sealed catalog metadata or explicit test fixtures."""

    if isinstance(catalog, _TestOnlyCatalog):
        return True
    try:
        from .alakazam_public_rule_adapter_r298 import is_public_catalog_eligible
    except ImportError:
        # The implementation file may be ahead of the adapter worktree during
        # collaborative staging.  No unsealed catalog becomes eligible here.
        return False
    try:
        return bool(is_public_catalog_eligible(catalog))
    except Exception:
        return False


def _catalog_value(catalog: Any) -> Any:
    return catalog.value if isinstance(catalog, _TestOnlyCatalog) else catalog


def _metadata_card_map(catalog: Any) -> dict[int, Mapping[str, Any]]:
    if catalog is None:
        return {}
    # The adapter will supply a receipt-sealed catalog type for corpus
    # materialization.  Until it is available locally, default-deny arbitrary
    # mappings: old guide CSV/IDs and ad-hoc metadata cannot authorize a
    # mechanics-dependent nonzero target.  Narrow fixtures may opt in through
    # a compiler-owned wrapper set by ``allow_test_catalog=True``.
    if not _catalog_is_eligible(catalog):
        return {}
    catalog = _catalog_value(catalog)
    records = getattr(catalog, "cards", None)
    if records is None and isinstance(catalog, Mapping):
        records = catalog.get("cards")
    if not isinstance(records, (list, tuple)):
        return {}
    result: dict[int, Mapping[str, Any]] = {}
    for row in records:
        if not isinstance(row, Mapping):
            continue
        card_id = _card_id(row)
        if card_id is not None:
            result[card_id] = row
    return result


def _metadata_attack_map(catalog: Any) -> dict[int, Mapping[str, Any]]:
    if catalog is None:
        return {}
    if not _catalog_is_eligible(catalog):
        return {}
    catalog = _catalog_value(catalog)
    records = getattr(catalog, "attacks", None)
    if records is None and isinstance(catalog, Mapping):
        records = catalog.get("attacks")
    if not isinstance(records, (list, tuple)):
        return {}
    result: dict[int, Mapping[str, Any]] = {}
    for row in records:
        if not isinstance(row, Mapping):
            continue
        raw = row.get("attackId", row.get("id"))
        try:
            attack_id = _exact_int(raw, field="metadata.attackId", minimum=0, optional=True)
        except SimulatorRuleTargetError:
            continue
        if attack_id is not None:
            result[attack_id] = row
    return result


def _modifier_delta(value: Any) -> int | None:
    """Read only an explicit structured public prize modifier.

    Text, names, and inferred card effects are intentionally not inputs.  A
    caller may provide an exact simulator-exposed yield, signed delta, or
    reduction; ambiguous modifiers remain unavailable.
    """

    if value is None:
        return 0
    if not isinstance(value, Mapping):
        try:
            return _exact_int(value, field="visible prize modifier")
        except SimulatorRuleTargetError:
            return None
    exact_yield = _first_present(value, ("exact_yield", "exactYield", "yield"))
    if exact_yield is not None:
        # An exact yield cannot be converted to a general delta without the
        # defeated card's base class, so callers of this helper handle it.
        return None
    delta = _first_present(value, ("delta", "prize_delta", "prizeDelta"))
    if delta is not None:
        try:
            return _exact_int(delta, field="visible prize modifier delta")
        except SimulatorRuleTargetError:
            return None
    reduction = _first_present(value, ("reduction", "prize_reduction", "prizeReduction"))
    if reduction is not None:
        try:
            parsed = _exact_int(reduction, field="visible prize reduction", minimum=0)
        except SimulatorRuleTargetError:
            return None
        assert parsed is not None
        return -parsed
    return None


def prize_yield_from_public_card(
    card: Mapping[str, Any] | None,
    *,
    metadata_card: Mapping[str, Any] | None = None,
    visible_modifier: Any = None,
) -> int | None:
    """Return a public Prize yield with Mega-ex precedence.

    The base class is intentionally tiny and explicit: Mega Pokémon ex concede
    three Prizes, ordinary Pokémon ex two, and ordinary Pokémon one.  A
    simulator-exposed exact yield or structured visible modifier may override
    this value.  Card text is never parsed.  ``ex=True, megaEx=True`` is
    accepted and deterministically resolves to the Mega three-Prize class.
    """

    record: dict[str, Any] = {}
    if isinstance(metadata_card, Mapping):
        record.update(metadata_card)
    if isinstance(card, Mapping):
        record.update(card)
    if not record:
        return None
    exact_yield = _first_present(record, ("prizeYield", "prize_yield", "prizes"))
    if exact_yield is not None:
        try:
            return _exact_int(exact_yield, field="public card prize yield", minimum=0)
        except SimulatorRuleTargetError:
            return None
    # An opaque card id is not enough to infer an ordinary one-Prize class.
    # The class must be directly public on the event/board card or supplied by
    # the receipt-sealed catalog.  This blocks historical guide-ID fallbacks.
    if not any(name in record for name in ("megaEx", "mega_ex", "ex")):
        return None
    mega = _optional_bool(record.get("megaEx", record.get("mega_ex")), field="card.megaEx")
    ex = _optional_bool(record.get("ex"), field="card.ex")
    base = 3 if mega is True else 2 if ex is True else 1
    if isinstance(visible_modifier, Mapping):
        exact = _first_present(visible_modifier, ("exact_yield", "exactYield", "yield"))
        if exact is not None:
            try:
                return _exact_int(exact, field="visible exact prize yield", minimum=0)
            except SimulatorRuleTargetError:
                return None
    delta = _modifier_delta(visible_modifier)
    if delta is None:
        return None
    return max(0, base + delta)


def _visible_modifier_for_player(
    observation: Mapping[str, Any], *, seat: int
) -> tuple[Any, float | None, bool]:
    """Read a directly serialized public Prize modifier for one player.

    The first return value is retained in its structured form so an explicit
    simulator ``exact_yield`` can be applied to visible board liability.  The
    second is a signed numeric delta when one exists, suitable for the fixed
    prize-race vector.  An exact yield is useful evidence but deliberately
    does not masquerade as a generic delta.
    """

    current = _current(observation)
    player = _players(observation)[seat]
    raw = _first_present(
        player,
        (
            "visiblePrizeModifier",
            "visible_prize_modifier",
            "prizeModifier",
            "prize_modifier",
            "prizeReduction",
            "prize_reduction",
        ),
    )
    if raw is not None:
        delta = _modifier_delta(raw)
        return raw, None if delta is None else float(delta), True
    global_modifiers = _first_present(
        current,
        ("visiblePrizeModifiers", "visible_prize_modifiers", "prizeModifiers"),
    )
    if isinstance(global_modifiers, Mapping):
        raw = global_modifiers.get(str(seat), global_modifiers.get(seat))
        if raw is not None:
            delta = _modifier_delta(raw)
            return raw, None if delta is None else float(delta), True
        # A directly exposed per-seat mapping with no row for this player is
        # affirmative public evidence of no modifier, unlike an absent field.
        return {"delta": 0}, 0.0, True
    if isinstance(global_modifiers, (list, tuple)):
        matching = [
            row for row in global_modifiers
            if isinstance(row, Mapping)
            and _event_player(
                _ChainEvent({}, {}, {}, {}, "modifier", None, False, False, row),
                ("playerIndex", "player_index", "targetPlayerIndex"),
            )
            == seat
        ]
        if len(matching) == 1:
            delta = _modifier_delta(matching[0])
            return matching[0], None if delta is None else float(delta), True
        if len(matching) > 1:
            return None, None, True
        # A directly exposed list with no matching player is likewise an
        # explicit zero modifier, not a schema omission.
        return {"delta": 0}, 0.0, True
    # Do not infer "no modifier" merely because this replay surface omits the
    # structured field.  Card text is not an allowed fallback.
    return None, None, False


def _visible_board_cards(observation: Mapping[str, Any], *, seat: int) -> list[Mapping[str, Any]] | None:
    player = _players(observation)[seat]
    result: list[Mapping[str, Any]] = []
    for zone in ("active", "bench"):
        rows = player.get(zone)
        if not isinstance(rows, (list, tuple)):
            return None
        for card in rows:
            if card is None:
                continue
            if not isinstance(card, Mapping):
                return None
            result.append(card)
    return result


def _visible_prize_liability(
    observation: Mapping[str, Any],
    *,
    seat: int,
    metadata_cards: Mapping[int, Mapping[str, Any]],
) -> tuple[float | None, float | None]:
    cards = _visible_board_cards(observation, seat=seat)
    modifier, modifier_delta, modifier_visible = _visible_modifier_for_player(
        observation, seat=seat
    )
    if cards is None or not modifier_visible or modifier is None:
        return None, None
    total = 0
    for card in cards:
        card_id = _card_id(card)
        # Without a sealed catalog, a visible card with no direct class/yield
        # declaration cannot be safely called ordinary.  Do not use historic
        # guide IDs or heuristics as a prize-rule fallback.
        has_direct_class = any(
            name in card
            for name in (
                "prizeYield",
                "prize_yield",
                "prizes",
                "ex",
                "megaEx",
                "mega_ex",
            )
        )
        if card_id not in metadata_cards and not has_direct_class:
            return None, modifier_delta if modifier_visible else None
        value = prize_yield_from_public_card(
            card,
            metadata_card=None if card_id is None else metadata_cards.get(card_id),
            visible_modifier=modifier if modifier_visible else None,
        )
        if value is None:
            return None, modifier_delta if modifier_visible else None
        total += value
    return float(total), modifier_delta if modifier_visible else 0.0


def _attack_option_type(value: Any) -> bool:
    token = _norm_token(value)
    return token in {"attack", "optiontypeattack", "13"}


def _selected_semantics(
    representation: Any,
    action: Sequence[int],
) -> list[dict[str, Any]]:
    options = getattr(representation, "options", ())
    result: list[dict[str, Any]] = []
    for index in action:
        if not 0 <= int(index) < len(options):
            raise SimulatorRuleTargetError("semantic representation lost a selected legal option")
        row = options[int(index)]
        semantic = getattr(row, "semantic", None)
        key = getattr(row, "semantic_key_sha256", None)
        if not isinstance(semantic, Mapping) or not isinstance(key, str):
            raise SimulatorRuleTargetError("semantic option representation is malformed")
        result.append({"semantic_key_sha256": key, "semantic": copy.deepcopy(dict(semantic))})
    return result


def _selected_action_semantic_hash(selected_semantics: Sequence[Mapping[str, Any]]) -> str:
    """Return an order-invariant identity for the selected public semantics.

    Numeric option indexes remain necessary to replay the concrete legal list,
    but they are not a durable public identity: a harmless simulator list
    permutation changes them.  Receipt binding uses the normalized semantic
    keys instead, while `_normalize_chain` separately proves that the concrete
    selected indexes were legal and consistent throughout the trace.
    """

    keys: list[str] = []
    for index, row in enumerate(selected_semantics):
        key = row.get("semantic_key_sha256")
        if not isinstance(key, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", key):
            raise SimulatorRuleTargetError(
                f"selected semantic[{index}] has no canonical semantic key"
            )
        keys.append(key)
    return _digest({"selected_option_semantic_keys": sorted(keys)})


def _raw_prompt_chain_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, DeterministicPromptChain):
        return _mapping(value.to_dict(), field="prompt_chain")
    return _mapping(value, field="prompt_chain")


def _trajectory_validator_accepts(
    validator: SelectedActionTrajectoryReceiptValidator | Any | None,
    *,
    receipt: Mapping[str, Any],
    chain: _NormalizedChain,
    public_observation_sha256: str,
    selected_action_semantic_sha256: str,
) -> tuple[bool, bool, str, Mapping[str, Any] | None]:
    """Run a receipt verifier and distinguish diagnostics from authorization.

    This is intentionally not an authorization boundary.  An arbitrary object
    can implement a method that returns ``True``; accepting that result must
    never arm a supervised target.  It is retained solely so fixture and
    future materializer code can check callback wiring while the current r298
    path remains zero-inert.  Only the concrete immutable-ledger validator
    returned by ``load_sealed_selected_action_trajectory_validator``—which
    also proves the revision-5 schema freeze and handoff activation—can set
    the second tuple item true.  The compiler records all other callback
    responses as diagnostics and keeps ``trainable_target_eligible`` false.
    """

    if validator is None:
        return False, False, "absent", None
    method = getattr(validator, "validate_selected_action_trajectory_receipt", None)
    if not callable(method):
        return False, False, "unsupported_validator", None
    try:
        accepted = method(
            receipt=receipt,
            normalized_prompt_chain_sha256=chain.chain_hash,
            public_observation_sha256=public_observation_sha256,
            selected_action_semantic_sha256=selected_action_semantic_sha256,
            simulator=chain.simulator,
            source=chain.source,
        ) is True
    except Exception:
        return False, False, "validator_exception", None
    if isinstance(validator, SealedSelectedActionTrajectoryValidator):
        return (
            accepted,
            accepted,
            validator.validator_kind,
            validator.provenance(),
        )
    return accepted, False, "unsealed_callback_diagnostic_only", None


def _observed_trajectory_provenance(
    raw_chain: Any,
    *,
    chain: _NormalizedChain,
    selected_semantics: Sequence[Mapping[str, Any]],
    validator: SelectedActionTrajectoryReceiptValidator | Any | None,
) -> dict[str, Any]:
    """Validate a recorded-chain receipt before exposing trainable masks.

    This intentionally does not reject an unverified trajectory outright:
    its simulator-derived labels can still be inspected by engine tests and
    collision tooling.  The returned `trainable_target_eligible` bit is the
    one loss/materialization boundary may trust, and stays false until an
    immutable external receipt verifier confirms the raw-frame/corpus link.
    """

    public_hash = chain.root_public_observation_hash
    semantic_action_hash = _selected_action_semantic_hash(selected_semantics)
    common = {
        "source": chain.source,
        "normalized_prompt_chain_sha256": chain.chain_hash,
        "public_observation_sha256": public_hash,
        "selected_action_semantic_sha256": semantic_action_hash,
        "target_only": True,
        "may_drive_runtime": False,
        "trainable_target_eligible": False,
    }

    if chain.source == "observed_selected_action_trajectory":
        raw = _raw_prompt_chain_mapping(raw_chain)
        candidate = raw.get("observed_trajectory_receipt")
        if not isinstance(candidate, Mapping):
            return {**common, "available": False, "reason": "observed_trajectory_receipt_absent"}
        receipt = _mapping(candidate, field="prompt_chain.observed_trajectory_receipt")
        if receipt.get("schema") != R298_OBSERVED_TRAJECTORY_RECEIPT_SCHEMA:
            return {**common, "available": False, "reason": "observed_trajectory_receipt_schema_mismatch"}
        if receipt.get("version") != R298_OBSERVED_TRAJECTORY_RECEIPT_SCHEMA_VERSION:
            return {**common, "available": False, "reason": "observed_trajectory_receipt_version_mismatch"}
        if receipt.get("target_only") is not True or receipt.get("may_drive_runtime") is not False:
            return {**common, "available": False, "reason": "observed_trajectory_receipt_runtime_boundary"}
        if receipt.get("normalized_prompt_chain_sha256") != chain.chain_hash:
            return {**common, "available": False, "reason": "observed_trajectory_chain_binding_mismatch"}
        if receipt.get("public_observation_sha256") != public_hash:
            return {**common, "available": False, "reason": "observed_trajectory_public_binding_mismatch"}
        if receipt.get("selected_action_semantic_sha256") != semantic_action_hash:
            return {**common, "available": False, "reason": "observed_trajectory_action_binding_mismatch"}
        try:
            trajectory_sha = _sha256_digest(
                receipt.get("trajectory_receipt_sha256"),
                field="observed trajectory receipt.trajectory_receipt_sha256",
            )
            corpus_sha = _sha256_digest(
                receipt.get("corpus_receipt_sha256"),
                field="observed trajectory receipt.corpus_receipt_sha256",
            )
            raw_frame_sha = _sha256_digest(
                receipt.get("raw_frame_receipt_sha256"),
                field="observed trajectory receipt.raw_frame_receipt_sha256",
            )
        except SimulatorRuleTargetError:
            return {**common, "available": False, "reason": "observed_trajectory_receipt_digest_malformed"}
        accepted, trainable, validator_kind, validator_provenance = _trajectory_validator_accepts(
            validator,
            receipt=receipt,
            chain=chain,
            public_observation_sha256=public_hash,
            selected_action_semantic_sha256=semantic_action_hash,
        )
        return {
            **common,
            "schema": R298_OBSERVED_TRAJECTORY_RECEIPT_SCHEMA,
            "version": R298_OBSERVED_TRAJECTORY_RECEIPT_SCHEMA_VERSION,
            "trajectory_receipt_sha256": trajectory_sha,
            "corpus_receipt_sha256": corpus_sha,
            "raw_frame_receipt_sha256": raw_frame_sha,
            "available": accepted,
            "externally_validated": accepted,
            "validator_kind": validator_kind,
            "validator_provenance": validator_provenance,
            "trainable_target_eligible": trainable,
            "reason": (
                None
                if trainable
                else "unsealed_callback_diagnostic_only"
                if accepted
                else "immutable_receipt_unverified"
            ),
        }

    # Restored chains already require public-seed/state-restoration fields in
    # `_normalize_chain`, but those SHA-shaped values are still not proof that
    # the caller owns a sealed targeted corpus.  Apply the same external gate.
    receipt = chain.restoration_provenance
    if not isinstance(receipt, Mapping):
        return {**common, "available": False, "reason": "restoration_provenance_absent"}
    accepted, trainable, validator_kind, validator_provenance = _trajectory_validator_accepts(
        validator,
        receipt=receipt,
        chain=chain,
        public_observation_sha256=public_hash,
        selected_action_semantic_sha256=semantic_action_hash,
    )
    return {
        **common,
        "schema": R298_TARGET_PROVENANCE_SCHEMA,
        "version": R298_TARGET_PROVENANCE_SCHEMA_VERSION,
        "trajectory_receipt_sha256": receipt.get("trajectory_receipt_sha256"),
        "corpus_receipt_sha256": receipt.get("corpus_receipt_sha256"),
        "state_restoration_receipt_sha256": receipt.get("state_restoration_receipt_sha256"),
        "public_seed_receipt_sha256": receipt.get("public_seed_receipt_sha256"),
        "available": accepted,
        "externally_validated": accepted,
        "validator_kind": validator_kind,
        "validator_provenance": validator_provenance,
        "trainable_target_eligible": trainable,
        "reason": (
            None
            if trainable
            else "unsealed_callback_diagnostic_only"
            if accepted
            else "immutable_receipt_unverified"
        ),
    }


def _typed_energy_units_from_representation(representation: Any) -> tuple[Counter[str] | None, bool]:
    state = getattr(representation, "state", None)
    if not isinstance(state, Mapping):
        return None, False
    players = state.get("players")
    if not isinstance(players, Mapping):
        return None, False
    acting = players.get("acting")
    if not isinstance(acting, Mapping):
        return None, False
    active = acting.get("active")
    if not isinstance(active, list):
        return None, False
    cards = [row.get("card") for row in active if isinstance(row, Mapping) and isinstance(row.get("card"), Mapping)]
    if len(cards) != 1:
        return None, False
    card = cards[0]
    assert isinstance(card, Mapping)
    unknown = _exact_int(
        card.get("unknown_typed_energy_card_count"),
        field="active.unknown_typed_energy_card_count",
        minimum=0,
        optional=True,
    )
    if unknown is None or unknown > 0:
        return None, False
    raw = card.get("typed_energy_units")
    if not isinstance(raw, list):
        return None, False
    units: Counter[str] = Counter()
    for index, pair in enumerate(raw):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None, False
        label = str(pair[0])
        count = _exact_int(pair[1], field=f"typed_energy_units[{index}].count", minimum=0)
        assert count is not None
        units[label] += count
    return units, True


def _attack_cost_satisfied(
    selected_semantics: Sequence[Mapping[str, Any]],
    *,
    representation: Any,
    metadata_attacks: Mapping[int, Mapping[str, Any]],
    catalog_available: bool,
) -> dict[str, Any]:
    target = _masked_vector(ATTACK_READINESS_LAYOUT)
    attacks: list[Mapping[str, Any]] = []
    for row in selected_semantics:
        semantic = row.get("semantic")
        option = semantic.get("option") if isinstance(semantic, Mapping) else None
        if not isinstance(option, Mapping):
            continue
        if _attack_option_type(option.get("option_type")):
            attacks.append(option)
    if not attacks:
        _set_vector(target, "selected_option_is_attack", 0.0)
        _set_vector(target, "attack_legal_in_simulator_option_list", 0.0)
        return target
    # A selected attack is itself proof that libcg emitted it as legal.  We do
    # not synthesize attacks that the legal option list omitted.
    _set_vector(target, "selected_option_is_attack", 1.0)
    _set_vector(target, "attack_legal_in_simulator_option_list", 1.0)
    if len(attacks) != 1:
        return target
    attack_id = _exact_int(attacks[0].get("attack_id"), field="selected attack id", minimum=0, optional=True)
    if attack_id is None:
        return target
    if not catalog_available:
        return target
    metadata = metadata_attacks.get(attack_id)
    if not isinstance(metadata, Mapping):
        return target
    costs = metadata.get("energies", metadata.get("energyCost"))
    if not isinstance(costs, (list, tuple)):
        return target
    units, available = _typed_energy_units_from_representation(representation)
    if not available or units is None:
        return target
    required: Counter[str] = Counter()
    for cost in costs:
        if isinstance(cost, Mapping):
            token_raw = _first_present(cost, ("type", "energyType", "energy_type"))
            count_raw = _first_present(cost, ("count", "units", "amount"))
            if token_raw is None or count_raw is None:
                return target
            count = _exact_int(count_raw, field="attack typed cost count", minimum=0)
            assert count is not None
            required[f"energy_type:{_norm_token(token_raw)}"] += count
        else:
            required[f"energy_type:{_norm_token(cost)}"] += 1
    _set_vector(target, "typed_cost_known", 1.0)
    available_total = sum(units.values())
    needed_total = sum(required.values())
    satisfied = True
    for token, count in required.items():
        if token in {"energy_type:colorless", "energy_type:generic", "energy_type:any"}:
            if available_total < needed_total:
                satisfied = False
        elif units[token] < count:
            satisfied = False
    _set_vector(target, "typed_cost_satisfied", float(satisfied))
    return target


def _is_kind(event: _ChainEvent, *names: str) -> bool:
    return event.event_kind in {_norm_token(name) for name in names}


def _sum_event_fact(
    events: Sequence[_ChainEvent],
    *,
    kinds: Sequence[str],
    fields: Sequence[str],
    fallback_kinds: Sequence[str] = (),
) -> tuple[float, bool]:
    """Read one exact fact family without aggregate/delta double counting.

    Simulators sometimes emit a granular ``draw``/``damage`` event *and* an
    initiating attack summary.  When granular facts exist they are the source
    of truth; attack/effect summaries are used only as a fallback.  A relevant
    event that lacks the numeric fact makes the label unavailable rather than
    inviting a guessed zero or a duplicate total.
    """

    def collect(tokens: set[str]) -> tuple[float, bool, bool]:
        total = 0.0
        seen = False
        incomplete = False
        for event in events:
            if event.event_kind not in tokens:
                continue
            seen = True
            value = _first_present(event.facts, fields)
            if value is None:
                incomplete = True
                continue
            number = _finite_number(value, field=f"{event.event_kind}.{fields[0]}")
            assert number is not None
            total += number
        return total, seen, incomplete

    total, seen, incomplete = collect({_norm_token(kind) for kind in kinds})
    if seen:
        return total, not incomplete
    if fallback_kinds:
        total, seen, incomplete = collect({_norm_token(kind) for kind in fallback_kinds})
        return total, seen and not incomplete
    return 0.0, False


def _opponent_knockout_count(events: Sequence[_ChainEvent], *, actor: int) -> tuple[int, bool]:
    """Count opposing KOs, including exact zero for a complete no-KO chain.

    The normalized chain is already required to have a complete simulator
    event log.  Thus a chain without a named KO event is evidence of zero,
    whereas a KO event without an identified victim is genuinely unavailable
    rather than silently treated as an own or opposing knockout.
    """

    count = 0
    known = True
    for event in events:
        if not _is_kind(event, "knockout", "ko", "pokemon_knockout"):
            continue
        victim = _event_player(event, ("victimPlayerIndex", "victim_player_index", "targetPlayerIndex", "target_player_index", "playerIndex"))
        if victim is None:
            known = False
            continue
        if victim == 1 - actor:
            count += 1
    return count, known


def _forced_promotion_count(events: Sequence[_ChainEvent], *, actor: int) -> tuple[int, bool]:
    """Count opposing forced promotions, including exact zero when absent."""

    count = 0
    known = True
    for event in events:
        if not _is_kind(event, "promotion", "active_promotion", "promote"):
            continue
        player = _event_player(event, ("playerIndex", "player_index", "targetPlayerIndex", "target_player_index"))
        if player is None:
            known = False
            continue
        if player == 1 - actor:
            count += 1
    return count, known


def _utility_target(chain: _NormalizedChain, *, actor: int) -> dict[str, list[Any]]:
    target = _masked_vector(ACTION_UTILITY_LAYOUT)
    damage, damage_known = _sum_event_fact(
        chain.events,
        kinds=("damage", "attack_damage", "effect_damage"),
        fields=("damage", "damageDealt", "damage_dealt"),
        fallback_kinds=("attack",),
    )
    if damage_known:
        _set_vector(target, "damage_dealt", damage)
    counters, counters_known = _sum_event_fact(
        chain.events,
        kinds=("damage_counter", "damage_counters", "effect_damage"),
        fields=("damageCounters", "damage_counters", "counters"),
        fallback_kinds=("attack",),
    )
    if counters_known:
        _set_vector(target, "damage_counters_placed", counters)
    draws, draws_known = _sum_event_fact(
        chain.events,
        kinds=("draw", "forced_draw", "effect_draw"),
        fields=("drawCount", "draw_count", "draws", "count"),
        fallback_kinds=("attack",),
    )
    if draws_known:
        _set_vector(target, "cards_drawn", draws)
    before_energy = _attached_energy_count(chain.root_before, seat=actor)
    after_energy = _attached_energy_count(chain.final_after, seat=actor)
    if before_energy is not None and after_energy is not None:
        _set_vector(target, "attached_energy_delta", after_energy - before_energy)
    before_slots = _bench_slots(chain.root_before, seat=actor)
    after_slots = _bench_slots(chain.final_after, seat=actor)
    if before_slots is not None and after_slots is not None:
        _set_vector(target, "open_bench_delta", after_slots - before_slots)
    before_prize = _prize_count(chain.root_before, seat=actor)
    after_prize = _prize_count(chain.final_after, seat=actor)
    if before_prize is not None and after_prize is not None:
        _set_vector(target, "own_prize_delta", before_prize - after_prize)
    knockouts, knockouts_known = _opponent_knockout_count(chain.events, actor=actor)
    if knockouts_known:
        _set_vector(target, "opponent_knockout", float(knockouts > 0))
    promotions, promotions_known = _forced_promotion_count(chain.events, actor=actor)
    if promotions_known:
        _set_vector(target, "forced_promotion_count", promotions)
    terminal_class, _reason = _terminal_class(chain, actor=actor)
    if terminal_class is not None:
        _set_vector(target, "terminal_own_win", float(terminal_class == "own_win"))
    return target


def _turn_resource_target(chain: _NormalizedChain, *, actor: int) -> dict[str, list[Any]]:
    target = _masked_vector(TURN_RESOURCE_LAYOUT)
    before = _current(chain.root_before)
    after = _current(chain.final_after)
    before_turn = _exact_int(
        before.get("turn"), field="before.turn", minimum=0, optional=True
    )
    after_turn = _exact_int(
        after.get("turn"), field="after.turn", minimum=0, optional=True
    )
    # Current turn-resource flags are scoped to whoever owns the displayed
    # turn.  Once an attack/END sequence has advanced the turn, comparing the
    # final flags would incorrectly read the next player's reset flags as the
    # selected actor's resources.  Leave all five labels unavailable instead.
    if before_turn is None or after_turn is None or before_turn != after_turn:
        return target
    for name, field in (
        ("supporter_played", "supporterPlayed"),
        ("stadium_played", "stadiumPlayed"),
        ("energy_attached", "energyAttached"),
        ("retreated", "retreated"),
    ):
        before_value = _optional_bool(before.get(field), field=f"before.{field}")
        after_value = _optional_bool(after.get(field), field=f"after.{field}")
        if before_value is not None and after_value is not None:
            # Target whether this selected action / its forced chain consumed
            # the resource.  A previously consumed flag remains zero here.
            _set_vector(target, name, float((not before_value) and after_value))
    before_count = _exact_int(before.get("turnActionCount"), field="before.turnActionCount", minimum=0, optional=True)
    after_count = _exact_int(after.get("turnActionCount"), field="after.turnActionCount", minimum=0, optional=True)
    if before_count is not None and after_count is not None:
        _set_vector(target, "turn_action_count_delta", after_count - before_count)
    return target


def _prize_yield_target(
    chain: _NormalizedChain,
    *,
    actor: int,
    metadata_cards: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(chain.events):
        if not _is_kind(event, "knockout", "ko", "pokemon_knockout"):
            continue
        victim = _event_player(event, ("victimPlayerIndex", "victim_player_index", "targetPlayerIndex", "target_player_index", "playerIndex"))
        card = _event_card(event)
        card_id = _card_id(card)
        modifier = _first_present(event.facts, ("visiblePrizeModifier", "visible_prize_modifier", "prizeModifier", "prize_modifier"))
        explicit_yield = _first_present(event.facts, ("prizeYield", "prize_yield", "prizes"))
        if explicit_yield is not None:
            yield_value = _exact_int(explicit_yield, field="knockout.prizeYield", minimum=0)
            source = "simulator_event_explicit"
        else:
            yield_value = prize_yield_from_public_card(
                card,
                metadata_card=None if card_id is None else metadata_cards.get(card_id),
                visible_modifier=modifier,
            )
            source = "public_card_class_and_visible_modifier" if yield_value is not None else "unavailable"
        rows.append(
            {
                "event_index": index,
                "victim_seat": victim,
                "card_id": card_id,
                "yield": yield_value,
                "mask": yield_value is not None,
                "source": source,
            }
        )
    before = _prize_count(chain.root_before, seat=actor)
    after = _prize_count(chain.final_after, seat=actor)
    observed_delta = None if before is None or after is None else before - after
    predicted = None
    applicable_rows = [row for row in rows if row.get("victim_seat") == 1 - actor]
    unknown_victim = any(row.get("victim_seat") is None for row in rows)
    applicable_yields_known = all(row.get("mask") is True for row in applicable_rows)
    if not unknown_victim and applicable_yields_known:
        # A complete chain with a measured zero Prize delta and no KO events
        # proves zero; never synthesize it when the observed delta is missing
        # or conflicts with the structured KO trace.
        candidate = sum(int(row["yield"]) for row in applicable_rows)
        if rows or observed_delta == 0:
            if observed_delta is None or observed_delta == candidate:
                predicted = candidate
    return {
        "knockouts": rows,
        "observed_own_prize_delta": observed_delta,
        "observed_own_prize_delta_mask": observed_delta is not None,
        "public_predicted_yield": predicted,
        "public_predicted_yield_mask": predicted is not None,
        "simulator_delta_matches_public_yield_when_both_available": (
            None if observed_delta is None or predicted is None else observed_delta == predicted
        ),
        "source_of_truth": "simulator_prompt_chain_prize_delta",
    }


def _lethal_target(chain: _NormalizedChain, *, actor: int) -> dict[str, Any]:
    before = _prize_count(chain.root_before, seat=actor)
    after = _prize_count(chain.final_after, seat=actor)
    terminal, _reason = _terminal_result_from_target(chain.final_terminal)
    mask = before is not None and after is not None
    delta = 0 if not mask else int(before - after)
    # Terminal action labels must not be right-censored merely because there is
    # no later own decision frame.  A terminal own win is a conversion even in
    # variants whose winner is set after the final sequential Prize prompt.
    conversion = bool(delta > 0 or terminal == actor) if mask or terminal is not None else False
    return {
        "layout": list(LETHAL_THREAT_LAYOUT),
        "values": [float(conversion) if (mask or terminal is not None) else 0.0],
        "mask": [bool(mask or terminal is not None)],
        "post_chain_own_prize_delta": float(delta) if mask else 0.0,
        "post_chain_own_prize_delta_mask": bool(mask),
        "terminal_action_included": True,
        "provenance": "selected_action_through_deterministic_prompt_chain",
    }


def _prize_race_target(
    observation: Mapping[str, Any],
    *,
    actor: int,
    metadata_cards: Mapping[int, Mapping[str, Any]],
) -> dict[str, list[Any]]:
    target = _masked_vector(PRIZE_RACE_LAYOUT)
    own_prize = _prize_count(observation, seat=actor)
    opponent_prize = _prize_count(observation, seat=1 - actor)
    if own_prize is not None:
        _set_vector(target, "acting_prizes_remaining", own_prize)
    if opponent_prize is not None:
        _set_vector(target, "opponent_prizes_remaining", opponent_prize)
    own_liability, own_modifier = _visible_prize_liability(
        observation, seat=actor, metadata_cards=metadata_cards
    )
    opponent_liability, opponent_modifier = _visible_prize_liability(
        observation, seat=1 - actor, metadata_cards=metadata_cards
    )
    if own_liability is not None:
        _set_vector(target, "acting_visible_prize_liability", own_liability)
    if opponent_liability is not None:
        _set_vector(target, "opponent_visible_prize_liability", opponent_liability)
    if own_modifier is not None:
        _set_vector(target, "acting_known_prize_modifier", own_modifier)
    if opponent_modifier is not None:
        _set_vector(target, "opponent_known_prize_modifier", opponent_modifier)
    return target


def _terminal_class(chain: _NormalizedChain, *, actor: int) -> tuple[str | None, str | None]:
    result, reason = _terminal_result_from_target(chain.final_terminal)
    if result is None:
        # The complete-chain contract proves that the last event is the next
        # real decision boundary.  A recorder that omits its optional
        # nonterminal sentinel still proves the selected action did not end
        # the game; do not recreate the legacy immediate-only censoring.
        if chain.events[-1].strategic_decision:
            return "nonterminal", reason
        return None, reason
    if result == -1:
        return "nonterminal", reason
    if result == 2:
        return "draw", reason
    return ("own_win" if result == actor else "own_loss"), reason


def _terminal_conversion_target(chain: _NormalizedChain, *, actor: int) -> dict[str, Any]:
    target = _masked_vector(TERMINAL_CONVERSION_LAYOUT)
    terminal_class, reason = _terminal_class(chain, actor=actor)
    if terminal_class is not None:
        _set_vector(target, f"terminal_class.{terminal_class}", 1.0)
        # A deterministic chain that reaches the next genuine strategic prompt
        # proves nonterminal, rather than being masked as legacy immediate-only
        # terminal-conversion labels were.
    before = _prize_count(chain.root_before, seat=actor)
    after = _prize_count(chain.final_after, seat=actor)
    if before is not None and after is not None:
        _set_vector(target, "prize_closeout_after_forced_chain", float(before > 0 and after == 0))
    knockouts, known = _opponent_knockout_count(chain.events, actor=actor)
    if known:
        _set_vector(target, "opponent_knockout_after_forced_chain", float(knockouts > 0))
    return {
        **target,
        "terminal_reason": reason,
        "prompt_chain_credit": "selected_action_through_forced_prize_and_promotion_prompts",
    }


def _game_phase_target(chain: _NormalizedChain, *, actor: int) -> dict[str, Any]:
    before = _prize_count(chain.root_before, seat=actor)
    opponent = _prize_count(chain.root_before, seat=1 - actor)
    after = _prize_count(chain.final_after, seat=actor)
    current = _current(chain.root_before)
    turn = _exact_int(current.get("turn"), field="current.turn", minimum=0, optional=True)
    terminal_class, _reason = _terminal_class(chain, actor=actor)
    label: str | None = None
    if terminal_class is not None and terminal_class != "nonterminal":
        label = "terminal"
    # The repair deliberately does not classify a one-Prize state as closeout
    # in isolation.  It needs a selected legal action whose exact prompt chain
    # reaches Prize zero (or terminal own win).
    elif before is not None and after is not None and before > 0 and after == 0:
        label = "closeout"
    elif before is not None and opponent is not None:
        if before <= 3 and opponent <= 3:
            label = "prize_race"
        elif before - opponent >= 2:
            label = "stabilize"
        elif turn is not None and turn <= 2:
            label = "setup"
        else:
            label = "pressure"
    if label is None:
        return {
            "classes": list(GAME_PHASE_CLASSES),
            "class_index": 0,
            "mask": False,
            "reason": "missing_exact_public_prize_or_prompt_chain_evidence",
            "closeout_requires_reachable_exact_prize_conversion": True,
        }
    return {
        "classes": list(GAME_PHASE_CLASSES),
        "class_index": GAME_PHASE_CLASSES.index(label),
        "mask": True,
        "label": label,
        "closeout_requires_reachable_exact_prize_conversion": True,
    }


def _deck_out_target(chain: _NormalizedChain, *, actor: int) -> dict[str, Any]:
    forced_draws = 0.0
    draw_event_indices: list[int] = []
    draw_counts_known = True
    for index, event in enumerate(chain.events):
        is_draw = _is_kind(event, "draw", "forced_draw", "attack_draw", "effect_draw")
        if not is_draw:
            continue
        forced_marker = _optional_bool(event.facts.get("forced"), field=f"draw[{index}].forced")
        forced = event.forced if forced_marker is None else forced_marker
        if not forced:
            continue
        value = _first_present(event.facts, ("drawCount", "draw_count", "draws", "count"))
        if value is None:
            draw_counts_known = False
            continue
        number = _finite_number(value, field=f"draw[{index}].count")
        assert number is not None
        forced_draws += number
        draw_event_indices.append(index)
    before_deck = _deck_count(chain.root_before, seat=actor)
    after_deck = _deck_count(chain.final_after, seat=actor)
    result, reason = _terminal_result_from_target(chain.final_terminal)
    explicit_deck_out = any(_is_kind(event, "deckout", "deck_out", "forced_draw_loss") for event in chain.events)
    reason_deck_out = reason is not None and any(token in reason for token in ("deckout", "deckout", "forceddraw"))
    return {
        "forced_draw_count_before_next_strategic_decision": forced_draws if draw_counts_known else 0.0,
        "forced_draw_count_mask": draw_counts_known,
        "forced_draw_event_indices": draw_event_indices,
        "acting_deck_before": before_deck,
        "acting_deck_before_mask": before_deck is not None,
        "acting_deck_after": after_deck,
        "acting_deck_after_mask": after_deck is not None,
        "deck_out_observed": bool(explicit_deck_out or reason_deck_out),
        "deck_out_observed_mask": bool(explicit_deck_out or reason is not None),
        "terminal_result": result,
        "terminal_reason": reason,
        "source": "simulator_forced_draw_prompt_chain",
    }


def _immediate_effect_target(chain: _NormalizedChain) -> dict[str, Any]:
    damage, damage_known = _sum_event_fact(
        chain.events,
        kinds=("damage", "attack_damage", "effect_damage"),
        fields=("damage", "damageDealt", "damage_dealt"),
        fallback_kinds=("attack",),
    )
    counters, counters_known = _sum_event_fact(
        chain.events,
        kinds=("damage_counter", "damage_counters", "effect_damage"),
        fields=("damageCounters", "damage_counters", "counters"),
        fallback_kinds=("attack",),
    )
    draws, draws_known = _sum_event_fact(
        chain.events,
        kinds=("draw", "forced_draw", "effect_draw"),
        fields=("drawCount", "draw_count", "draws", "count"),
        fallback_kinds=("attack",),
    )
    discards, discards_known = _sum_event_fact(
        chain.events,
        kinds=("discard", "effect_discard"),
        fields=("discardCount", "discard_count", "discards", "count"),
        fallback_kinds=("attack",),
    )
    benches, benches_known = _sum_event_fact(
        chain.events,
        kinds=("bench", "play_to_bench", "effect_bench"),
        fields=("benchCount", "bench_count", "benches", "count"),
        fallback_kinds=("attack",),
    )
    return {
        "damage": {"value": damage if damage_known else 0.0, "mask": damage_known},
        "damage_counters": {"value": counters if counters_known else 0.0, "mask": counters_known},
        "draw": {"value": draws if draws_known else 0.0, "mask": draws_known},
        "discard": {"value": discards if discards_known else 0.0, "mask": discards_known},
        "bench": {"value": benches if benches_known else 0.0, "mask": benches_known},
        "chain_credit": "selected_action_and_deterministic_forced_prompts_only",
    }


def _prompt_chain_target(chain: _NormalizedChain) -> dict[str, Any]:
    terminal, _reason = _terminal_result_from_target(chain.final_terminal)
    return {
        "event_count": len(chain.events),
        "forced_event_count": sum(1 for event in chain.events[1:] if event.forced),
        "next_genuine_strategic_decision_reached": bool(chain.events[-1].strategic_decision),
        "terminal_reached": terminal is not None and terminal != -1,
        "event_log_complete": chain.event_log_complete,
        "complete_to_next_strategic_decision": chain.complete_to_next_strategic_decision,
        # This digest binds the realized target trajectory.  It is target-only
        # and intentionally absent from the public pre-decision fingerprint.
        "realized_target_chain_hash": chain.chain_hash,
        "simulator": copy.deepcopy(dict(chain.simulator)),
    }


def _normalize_card_count_target(value: Any, *, field: str) -> dict[str, Any]:
    """Validate an explicitly privileged card-count distribution.

    It remains a target payload and is intentionally excluded from all public
    fingerprints and all public target construction.  A count distribution is
    richer than the historical binary-presence labels but does not turn hidden
    identities into policy inputs.
    """

    if value is None:
        return {"pairs": [], "mask": False, "reason": "absent"}
    counter: Counter[int] = Counter()
    if isinstance(value, Mapping):
        rows = value.items()
    elif isinstance(value, (list, tuple)):
        # Accept a card-id list as an exact count target and a list of [id,count]
        # pairs.  This is target-only ingestion, never a public feature path.
        if all(not isinstance(item, (list, tuple, Mapping)) for item in value):
            rows = Counter(value).items()
        else:
            parsed: list[tuple[Any, Any]] = []
            for item in value:
                if isinstance(item, Mapping):
                    parsed.append((item.get("card_id", item.get("id")), item.get("count", 1)))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    parsed.append((item[0], item[1]))
                else:
                    raise SimulatorRuleTargetError(f"{field} has an invalid count row")
            rows = parsed
    else:
        raise SimulatorRuleTargetError(f"{field} must be a count mapping or list")
    for raw_card, raw_count in rows:
        card = _exact_int(raw_card, field=f"{field}.card_id", minimum=0)
        count = _exact_int(raw_count, field=f"{field}.count", minimum=0)
        assert card is not None and count is not None
        if count:
            counter[card] += count
    return {
        "pairs": [[card, count] for card, count in sorted(counter.items())],
        "mask": True,
        "reason": None,
    }


def _validate_privileged_belief_receipt(
    validator: PrivilegedBeliefReceiptValidator | Any | None,
    *,
    receipt: Mapping[str, Any],
    normalized_prompt_chain_sha256: str,
    public_observation_sha256: str,
    selected_action_sha256: str,
    source: str,
) -> None:
    """Reject generic hidden-target callbacks until a sealed ledger extension.

    A generic callback may truthfully inspect a receipt, but it cannot prove
    that the *hidden target values* came from an immutable corpus.  Letting it
    return true would silently turn caller-supplied opponent identities into
    supervised data.  The current sealed trajectory ledger is deliberately
    public-chain-only, so it cannot yet validate this sidecar either.
    """

    if not isinstance(validator, SealedSelectedActionTrajectoryValidator):
        raise SimulatorRuleTargetError(
            "privileged belief target requires a future sealed ledger extension"
        )
    method = getattr(validator, "validate_privileged_belief_receipt", None)
    if not callable(method):
        raise SimulatorRuleTargetError(
            "sealed trajectory validator does not bind privileged belief payloads"
        )
    try:
        accepted = method(
            receipt=receipt,
            normalized_prompt_chain_sha256=normalized_prompt_chain_sha256,
            public_observation_sha256=public_observation_sha256,
            selected_action_sha256=selected_action_sha256,
            source=source,
        )
    except Exception as exc:
        raise SimulatorRuleTargetError("privileged belief receipt validator failed") from exc
    if accepted is not True:
        raise SimulatorRuleTargetError("privileged belief receipt is not sealed by validator")


def _belief_target_provenance(
    row: Mapping[str, Any],
    *,
    chain: _NormalizedChain,
    validator: PrivilegedBeliefReceiptValidator | Any | None,
) -> dict[str, Any]:
    """Verify that target-only belief labels bind this exact selected chain."""

    source = str(row.get("provenance") or "")
    if source not in {
        "privileged_target_only_authoritative_trace",
        "privileged_target_only_restored_simulator",
    }:
        raise SimulatorRuleTargetError("privileged belief target has no allowed target-only provenance")
    receipt = _mapping(row.get("receipt"), field="privileged_belief_targets.receipt")
    if receipt.get("schema") != R298_PRIVILEGED_BELIEF_TARGET_SCHEMA:
        raise SimulatorRuleTargetError("privileged belief receipt schema mismatch")
    if receipt.get("version") != R298_PRIVILEGED_BELIEF_TARGET_SCHEMA_VERSION:
        raise SimulatorRuleTargetError("privileged belief receipt version mismatch")
    if receipt.get("target_only") is not True:
        raise SimulatorRuleTargetError("privileged belief receipt is not target-only")
    if receipt.get("may_drive_runtime") is not False:
        raise SimulatorRuleTargetError("privileged belief receipt permits runtime use")
    public_observation_hash = chain.root_public_observation_hash
    selected_action_hash = _digest(
        {"selected_action": [int(value) for value in chain.root_action]}
    )
    if receipt.get("normalized_prompt_chain_sha256") != chain.chain_hash:
        raise SimulatorRuleTargetError(
            "privileged belief receipt does not bind normalized selected-action chain"
        )
    if receipt.get("public_observation_sha256") != public_observation_hash:
        raise SimulatorRuleTargetError(
            "privileged belief receipt does not bind selected public observation"
        )
    if receipt.get("selected_action_sha256") != selected_action_hash:
        raise SimulatorRuleTargetError(
            "privileged belief receipt does not bind selected action"
        )
    trajectory_receipt_sha256 = _sha256_digest(
        receipt.get("trajectory_receipt_sha256"),
        field="belief.trajectory_receipt_sha256",
    )
    corpus_receipt_sha256 = _sha256_digest(
        receipt.get("corpus_receipt_sha256"),
        field="belief.corpus_receipt_sha256",
    )
    _validate_privileged_belief_receipt(
        validator,
        receipt=receipt,
        normalized_prompt_chain_sha256=chain.chain_hash,
        public_observation_sha256=public_observation_hash,
        selected_action_sha256=selected_action_hash,
        source=source,
    )
    provenance: dict[str, Any] = {
        "schema": R298_PRIVILEGED_BELIEF_TARGET_SCHEMA,
        "version": R298_PRIVILEGED_BELIEF_TARGET_SCHEMA_VERSION,
        "target_only": True,
        "may_drive_runtime": False,
        "source": source,
        "normalized_prompt_chain_sha256": chain.chain_hash,
        "public_observation_sha256": public_observation_hash,
        "selected_action_sha256": selected_action_hash,
        "trajectory_receipt_sha256": trajectory_receipt_sha256,
        "corpus_receipt_sha256": corpus_receipt_sha256,
        "externally_validated": True,
    }
    if source == "privileged_target_only_restored_simulator":
        if receipt.get("public_information_seed_only") is not True:
            raise SimulatorRuleTargetError("restored privileged belief seed is not public-only")
        provenance["public_information_seed_only"] = True
        provenance["state_restoration_receipt_sha256"] = _sha256_digest(
            receipt.get("state_restoration_receipt_sha256"),
            field="belief.state_restoration_receipt_sha256",
        )
        provenance["public_seed_receipt_sha256"] = _sha256_digest(
            receipt.get("public_seed_receipt_sha256"),
            field="belief.public_seed_receipt_sha256",
        )
    return provenance


def _unavailable_privileged_belief_target(reason: str) -> dict[str, Any]:
    return {
        "schema": "poke_bot.alakazam_opponent_belief_targets_r298/v1",
        "target_only": True,
        "available": False,
        "hand_count_distribution": _normalize_card_count_target(None, field="belief.hand"),
        "remainder_count_distribution": _normalize_card_count_target(
            None, field="belief.remainder"
        ),
        "policy_feature_eligible": False,
        "reason": reason,
    }


def _privileged_belief_targets(
    decision: Mapping[str, Any],
    *,
    chain: _NormalizedChain | None = None,
    validator: PrivilegedBeliefReceiptValidator | Any | None = None,
) -> dict[str, Any]:
    raw = decision.get("privileged_belief_targets")
    if raw is None:
        return _unavailable_privileged_belief_target("absent")
    if chain is None:
        return _unavailable_privileged_belief_target("selected_action_chain_unavailable")
    try:
        row = _mapping(raw, field="privileged_belief_targets")
        receipt_provenance = _belief_target_provenance(
            row, chain=chain, validator=validator
        )
        return {
            "schema": "poke_bot.alakazam_opponent_belief_targets_r298/v1",
            "target_only": True,
            "available": True,
            "hand_count_distribution": _normalize_card_count_target(
                _first_present(row, ("opponent_hand", "hand", "hand_counts")),
                field="belief.hand",
            ),
            "remainder_count_distribution": _normalize_card_count_target(
                _first_present(row, ("opponent_remainder", "remainder", "remainder_counts")),
                field="belief.remainder",
            ),
            "policy_feature_eligible": False,
            "provenance": receipt_provenance,
        }
    except SimulatorRuleTargetError:
        # An invalid privileged sidecar never poisons public targets; it is
        # simply not a trainable belief label until the materializer proves the
        # chain/corpus receipt relationship.
        return _unavailable_privileged_belief_target("immutable_receipt_unverified")


def _representation_target(
    observation: Mapping[str, Any],
    *,
    metadata_catalog: Any,
    allow_test_catalog: bool,
) -> Any:
    build, _sanitize, _terminal = _public_adapter()
    catalog = metadata_catalog
    if isinstance(catalog, _TestOnlyCatalog):
        catalog = catalog.value
        allow_test_catalog = True
    elif not _catalog_is_eligible(catalog):
        catalog = None
    try:
        try:
            return build(
                observation,
                metadata_catalog=catalog,
                allow_test_catalog=allow_test_catalog,
            )
        except TypeError:
            # Adapter versions before the sealed-catalog hardening accept no
            # explicit fixture flag.  We still pass only a sealed catalog (or
            # None) on production/materialization paths.
            return build(observation, metadata_catalog=catalog)
    except Exception as exc:
        raise SimulatorRuleTargetError(f"cannot build r298 public rule representation: {exc}") from exc


def _target_public_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return only pre-decision public identity fields for metamorphic checks.

    A selected action can truthfully lead to different realized public
    transitions under distinct hidden worlds.  Such later outcomes are labels,
    not an information-set identity, and must not be compared by the
    pre-decision invariance helper.  The separately privileged belief target is
    excluded for the same reason.  This deliberately does *not* claim that a
    realized trajectory is invariant across hidden-state variants.
    """

    row = _mapping(value, field="compiled target")
    semantics = _mapping(row.get("legal_option_semantics"), field="legal_option_semantics")
    selected = _rows(semantics.get("selected"), field="legal_option_semantics.selected")
    selected_keys: list[str] = []
    for index, item in enumerate(selected):
        row_item = _mapping(item, field=f"legal_option_semantics.selected[{index}]")
        key = row_item.get("semantic_key_sha256")
        if not isinstance(key, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", key):
            raise SimulatorRuleTargetError(
                "legal option selected semantic identity is malformed"
            )
        selected_keys.append(key)
    return {
        "schema": row.get("schema"),
        "version": row.get("version"),
        "revision": row.get("revision"),
        "digest": row.get("digest"),
        "status": row.get("status"),
        "target_only": row.get("target_only"),
        "policy_feature_eligible": row.get("policy_feature_eligible"),
        "public_observation_hash": row.get("public_observation_hash"),
        "legal_option_semantics": {
            "available": semantics.get("available"),
            "representation_schema": semantics.get("representation_schema"),
            "representation_revision": semantics.get("representation_revision"),
            "semantic_token_hash": semantics.get("semantic_token_hash"),
            "canonical_option_multiset_hash": semantics.get("canonical_option_multiset_hash"),
            # This pre-decision identity must survive an incidental legal-list
            # permutation.  The full ordered option rows and raw action
            # indices remain target-only trace evidence elsewhere in the
            # record; only their normalized semantic identities belong here.
            "selected_option_semantic_keys": sorted(selected_keys),
            "legal_option_set_authority": semantics.get("legal_option_set_authority"),
        },
    }


def public_target_fingerprint(value: Mapping[str, Any]) -> str:
    """Hash public *pre-decision* identity, never realized future outcomes."""

    return _digest(_target_public_projection(value))


def _masked_result(
    *,
    reason: str,
    decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        belief = (
            _privileged_belief_targets(decision)
            if isinstance(decision, Mapping)
            else _privileged_belief_targets({})
        )
    except SimulatorRuleTargetError:
        # A malformed privileged payload must not turn an otherwise safe
        # public fail-closed result into an exception when strict mode is off.
        belief = _privileged_belief_targets({})
    return {
        "schema": R298_RULE_TARGET_SCHEMA,
        "version": R298_RULE_TARGET_SCHEMA_VERSION,
        "revision": R298_REVISION,
        "digest": R298_RULE_TARGET_SCHEMA_DIGEST,
        "status": "unavailable",
        "unavailable_reason": str(reason),
        "target_only": True,
        "policy_feature_eligible": False,
        "legal_option_semantics": {"available": False, "reason": str(reason), "options": [], "selected": []},
        "attack_readiness": _masked_vector(ATTACK_READINESS_LAYOUT),
        "immediate_effects": {},
        "prize_yield": {},
        "turn_resources": _masked_vector(TURN_RESOURCE_LAYOUT),
        "prompt_chain": {"available": False, "reason": str(reason)},
        "terminal": {"class": None, "mask": False, "reason": str(reason)},
        "deck_out": {"available": False, "reason": str(reason)},
        "lethal_threat": {
            "layout": list(LETHAL_THREAT_LAYOUT),
            "values": [0.0],
            "mask": [False],
            "post_chain_own_prize_delta": 0.0,
            "post_chain_own_prize_delta_mask": False,
            "terminal_action_included": True,
        },
        "prize_race": _masked_vector(PRIZE_RACE_LAYOUT),
        "action_utility": _masked_vector(ACTION_UTILITY_LAYOUT),
        "game_phase": {"classes": list(GAME_PHASE_CLASSES), "class_index": 0, "mask": False, "reason": str(reason)},
        "terminal_conversion": _masked_vector(TERMINAL_CONVERSION_LAYOUT),
        "privileged_belief_targets": belief,
        "provenance": {
            "counterfactual_labels": False,
            "trajectory_receipt": {
                "available": False,
                "target_only": True,
                "may_drive_runtime": False,
                "trainable_target_eligible": False,
                "reason": str(reason),
            },
            "runtime_action_authority": False,
            "production_authority": False,
        },
    }


def compile_simulator_rule_targets(
    decision: Mapping[str, Any],
    *,
    simulator: PromptChainSimulator | Any | None = None,
    metadata_catalog: Any = None,
    allow_test_catalog: bool = False,
    privileged_belief_receipt_validator: PrivilegedBeliefReceiptValidator | Any | None = None,
    trajectory_receipt_validator: SelectedActionTrajectoryReceiptValidator | Any | None = None,
    require_trainable_trajectory_receipt: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Compile selected-action r298 targets from a pinned simulator chain.

    ``decision`` needs the actor-visible ``observation`` and its executed
    complete ``action``.  It may carry an immutable ``prompt_chain`` captured
    from the pinned simulator; alternatively a narrow selected-action bridge
    can be injected through ``simulator``.  The bridge receives only the
    sanitized public observation and the action that was actually selected.

    This function has no unchosen-action branch.  Missing chain evidence
    yields an all-masked public target unless ``strict=True``.

    ``metadata_catalog`` is default-deny: ordinary mappings are ignored and
    mechanics-dependent labels remain masked.  Production/materialization must
    pass the adapter's receipt-sealed public catalog.  ``allow_test_catalog``
    exists solely for isolated deterministic fixtures and stamps provenance;
    it is not a corpus/training/runtime escape hatch.

    ``privileged_belief_receipt_validator`` is an optional, materializer-owned
    verifier for the separately typed belief sidecar.  Without it, that sidecar
    is masked even if a caller supplies SHA-shaped fields.  It cannot affect
    the public targets, legal options, or policy path.

    ``trajectory_receipt_validator`` can arm a row only when it is a
    :class:`SealedSelectedActionTrajectoryValidator` loaded from an immutable
    exact-30-day trajectory ledger, checksum-pinned post-census authorization,
    revision-5 schema-freeze receipt, and revision-5 handoff activation
    receipt.  Any ordinary callback is diagnostic-only and leaves every vector
    mask false.  Set
    ``require_trainable_trajectory_receipt=True`` only in that receipt-owning
    materializer to reject an unverified row outright.
    """

    try:
        row = _mapping(decision, field="decision")
        observation = _mapping(row.get("observation"), field="decision.observation")
        public_observation = _sanitize_public_observation(observation)
        actor = _actor(public_observation)
        action = _legal_action(public_observation, _rows(row.get("action"), field="decision.action"))
        raw_chain = row.get("prompt_chain")
        if raw_chain is None:
            if simulator is None:
                raise SimulatorRuleTargetError("selected action has no simulator prompt-chain evidence")
            raw_chain = _chain_from_simulator(
                simulator,
                public_observation=public_observation,
                selected_action=action,
            )
        chain = _normalize_chain(
            raw_chain,
            root_observation=observation,
            selected_action=action,
        )
        catalog = (
            _TestOnlyCatalog(metadata_catalog)
            if allow_test_catalog and metadata_catalog is not None
            else metadata_catalog
        )
        if metadata_catalog is not None and not allow_test_catalog and not _catalog_is_eligible(metadata_catalog):
            # Metadata-dependent fields stay masked.  This is not an error:
            # a public observation still supports direct trajectory targets.
            catalog = None
        representation = _representation_target(
            observation,
            metadata_catalog=catalog,
            allow_test_catalog=allow_test_catalog,
        )
        selected_semantics = _selected_semantics(representation, action)
        trajectory_provenance = _observed_trajectory_provenance(
            raw_chain,
            chain=chain,
            selected_semantics=selected_semantics,
            validator=trajectory_receipt_validator,
        )
        if (
            require_trainable_trajectory_receipt
            and trajectory_provenance.get("trainable_target_eligible") is not True
        ):
            raise SimulatorRuleTargetError(
                "selected-action trajectory has no externally validated immutable receipt"
            )
        metadata_cards = _metadata_card_map(catalog)
        metadata_attacks = _metadata_attack_map(catalog)
        terminal_class, terminal_reason = _terminal_class(chain, actor=actor)
        result = {
            "schema": R298_RULE_TARGET_SCHEMA,
            "version": R298_RULE_TARGET_SCHEMA_VERSION,
            "revision": R298_REVISION,
            "digest": R298_RULE_TARGET_SCHEMA_DIGEST,
            "status": "available",
            "target_only": True,
            "policy_feature_eligible": False,
            "public_observation_hash": public_observation_fingerprint(observation),
            "selected_action": list(action),
            "legal_option_semantics": {
                "available": True,
                "representation_schema": getattr(representation, "schema", None),
                "representation_revision": getattr(representation, "revision", None),
                "semantic_token_hash": getattr(representation, "semantic_token_hash", None),
                "canonical_option_multiset_hash": getattr(representation, "canonical_option_multiset_hash", None),
                "options": [option.to_dict() for option in getattr(representation, "options", ())],
                "selected": selected_semantics,
                "legal_option_set_authority": "simulator_emitted_only_no_synthesis",
            },
            "attack_readiness": _attack_cost_satisfied(
                selected_semantics,
                representation=representation,
                metadata_attacks=metadata_attacks,
                catalog_available=_catalog_is_eligible(catalog),
            ),
            "immediate_effects": _immediate_effect_target(chain),
            "prize_yield": _prize_yield_target(
                chain,
                actor=actor,
                metadata_cards=metadata_cards,
            ),
            "turn_resources": _turn_resource_target(chain, actor=actor),
            "prompt_chain": _prompt_chain_target(chain),
            "terminal": {
                "class": terminal_class,
                "mask": terminal_class is not None,
                "reason": terminal_reason,
                "simultaneous_closeout_draw_preserved": terminal_class == "draw",
            },
            "deck_out": _deck_out_target(chain, actor=actor),
            "lethal_threat": _lethal_target(chain, actor=actor),
            "prize_race": _prize_race_target(
                public_observation,
                actor=actor,
                metadata_cards=metadata_cards,
            ),
            "action_utility": _utility_target(chain, actor=actor),
            "game_phase": _game_phase_target(chain, actor=actor),
            "terminal_conversion": _terminal_conversion_target(chain, actor=actor),
            "privileged_belief_targets": _privileged_belief_targets(
                row,
                chain=chain,
                validator=privileged_belief_receipt_validator,
            ),
            "provenance": {
                "simulator": copy.deepcopy(dict(chain.simulator)),
                "prompt_chain_schema": R298_PROMPT_CHAIN_SCHEMA,
                "realized_target_chain_hash": chain.chain_hash,
                "trajectory_receipt": trajectory_provenance,
                "source": chain.source,
                "restoration_provenance": copy.deepcopy(chain.restoration_provenance),
                "selected_action_only": True,
                "unchosen_counterfactuals": False,
                "public_information_only_for_public_targets": True,
                "catalog_metadata": (
                    "explicit_test_fixture_only"
                    if isinstance(catalog, _TestOnlyCatalog)
                    else "sealed_public_catalog"
                    if _catalog_is_eligible(catalog)
                    else "unavailable_masked"
                ),
                "privileged_belief_targets_separate": True,
                "runtime_action_authority": False,
                "production_authority": False,
            },
        }
        result["public_target_fingerprint"] = public_target_fingerprint(result)
        return result
    except SimulatorRuleTargetError:
        if strict:
            raise
        return _masked_result(reason="simulator_rule_target_unavailable", decision=decision)
    except Exception as exc:
        if strict:
            raise SimulatorRuleTargetError(f"unexpected target compiler failure: {exc}") from exc
        return _masked_result(reason="simulator_rule_target_unavailable", decision=decision)


def _r5_trainable_trajectory_provenance(
    value: Mapping[str, Any],
    *,
    trajectory_receipt_validator: SealedSelectedActionTrajectoryValidator | None,
) -> bool:
    """Recognize only the concrete r5 authorization shape in a compiled row.

    This is a defense-in-depth check for the corpus-side vector adapter.  The
    receipt loader remains the authority—an arbitrary mapping can never make
    a generic callback trainable—but this prevents a stale r4-shaped or
    incomplete serialized provenance payload from retaining a true mask bit.
    """

    # A serialized mapping is not an authorization token.  The materializer
    # must carry the exact sealed validator it used to compile the row through
    # this adapter.  This prevents a caller from hand-writing a superficially
    # valid provenance mapping and turning masks on merely by setting booleans.
    if not isinstance(trajectory_receipt_validator, SealedSelectedActionTrajectoryValidator):
        return False
    if value.get("trainable_target_eligible") is not True:
        return False
    if (
        value.get("available") is not True
        or value.get("externally_validated") is not True
        or value.get("target_only") is not True
        or value.get("may_drive_runtime") is not False
        or value.get("validator_kind") != R298_SEALED_TRAJECTORY_VALIDATOR_KIND
    ):
        return False
    provenance = value.get("validator_provenance")
    if not isinstance(provenance, Mapping):
        return False
    if (
        provenance.get("target_only") is not True
        or provenance.get("may_drive_runtime") is not False
        or provenance.get("training_host") != R298_R5_TRAINING_HOST
        or provenance.get("root_owner_revision") != R298_ROOT_OWNER_REVISION
    ):
        return False
    try:
        for name in (
            "ledger_file_sha256",
            "authorization_file_sha256",
            "raw_corpus_receipt_sha256",
            "frozen_schema_manifest_sha256",
            "schema_freeze_receipt_sha256",
            "branch_support_receipt_sha256",
            "training_gate_report_sha256",
            "training_handoff_activation_receipt_sha256",
        ):
            _sha256_digest(provenance.get(name), field=f"validator provenance.{name}")
    except SimulatorRuleTargetError:
        return False
    return _canonical_json(dict(provenance)) == _canonical_json(
        trajectory_receipt_validator.provenance()
    )


def rule_head_target_vectors(
    compiled: Mapping[str, Any],
    *,
    trajectory_receipt_validator: SealedSelectedActionTrajectoryValidator | None = None,
) -> dict[str, dict[str, Any]]:
    """Return fixed-width target/mask rows for the isolated r298 head module.

    The helper is intentionally a corpus-side adapter only.  It accepts the
    compiler result and emits no policy features, action filters, or logits.
    Its default is deliberately all-masked: a trainable vector additionally
    requires the exact sealed validator instance used by the post-handoff
    materializer.  Persisted shards must be revalidated by that materializer
    before it recreates this object; a SHA-shaped serialized provenance row is
    never enough by itself.
    """

    row = _mapping(compiled, field="compiled rule targets")
    if row.get("schema") != R298_RULE_TARGET_SCHEMA:
        raise SimulatorRuleTargetError("rule target schema mismatch")
    if row.get("digest") != R298_RULE_TARGET_SCHEMA_DIGEST:
        raise SimulatorRuleTargetError("rule target digest mismatch")
    provenance = _mapping(row.get("provenance"), field="compiled.provenance")
    trajectory_receipt = _mapping(
        provenance.get("trajectory_receipt"), field="compiled.provenance.trajectory_receipt"
    )
    target_training_eligible = _r5_trainable_trajectory_provenance(
        trajectory_receipt,
        trajectory_receipt_validator=trajectory_receipt_validator,
    )

    def vector(name: str, layout: Sequence[str]) -> dict[str, Any]:
        value = _mapping(row.get(name), field=name)
        if list(value.get("layout") or []) != list(layout):
            raise SimulatorRuleTargetError(f"{name} layout mismatch")
        values = _rows(value.get("values"), field=f"{name}.values")
        mask = _rows(value.get("mask"), field=f"{name}.mask")
        if len(values) != len(layout) or len(mask) != len(layout):
            raise SimulatorRuleTargetError(f"{name} width mismatch")
        return {"values": list(values), "mask": list(mask)}

    lethal = _mapping(row.get("lethal_threat"), field="lethal_threat")
    terminal = _mapping(row.get("terminal_conversion"), field="terminal_conversion")
    game_phase = _mapping(row.get("game_phase"), field="game_phase")
    phase_mask = game_phase.get("mask") is True
    phase_values = [0.0] * len(GAME_PHASE_CLASSES)
    phase_index = _exact_int(game_phase.get("class_index"), field="game_phase.class_index", minimum=0, maximum=len(GAME_PHASE_CLASSES) - 1, optional=True)
    if phase_mask and phase_index is not None:
        phase_values[phase_index] = 1.0
    terminal_values = _rows(terminal.get("values"), field="terminal_conversion.values")
    terminal_mask = _rows(terminal.get("mask"), field="terminal_conversion.mask")
    if len(terminal_values) != len(TERMINAL_CONVERSION_LAYOUT) or len(terminal_mask) != len(TERMINAL_CONVERSION_LAYOUT):
        raise SimulatorRuleTargetError("terminal conversion target width mismatch")
    lethal_values = _rows(lethal.get("values"), field="lethal_threat.values")
    lethal_mask = _rows(lethal.get("mask"), field="lethal_threat.mask")
    if len(lethal_values) != len(LETHAL_THREAT_LAYOUT) or len(lethal_mask) != len(LETHAL_THREAT_LAYOUT):
        raise SimulatorRuleTargetError("lethal threat target width mismatch")
    belief = _mapping(row.get("privileged_belief_targets"), field="privileged_belief_targets")
    selected_action = _rows(row.get("selected_action"), field="selected_action")
    selected_indices: list[int] = []
    for index, value in enumerate(selected_action):
        parsed = _exact_int(value, field=f"selected_action[{index}]", minimum=0)
        assert parsed is not None
        selected_indices.append(parsed)

    # Targets in Phase C describe the complete action that was actually
    # selected, not every candidate option.  Option-conditioned heads therefore
    # receive these indices and pool only those rows in the derivative loss.
    # An empty legal selection remains explicit and gets no option-head loss;
    # we never manufacture a label for a hypothetical option.
    def selected_option_vector(name: str, layout: Sequence[str]) -> dict[str, Any]:
        result = vector(name, layout)
        result["selected_option_indices"] = list(selected_indices)
        return result

    result = {
        "lethal_threat": {"values": list(lethal_values), "mask": list(lethal_mask)},
        "prize_race": vector("prize_race", PRIZE_RACE_LAYOUT),
        "action_utility": selected_option_vector("action_utility", ACTION_UTILITY_LAYOUT),
        "game_phase": {
            "values": phase_values,
            "mask": [phase_mask] * len(GAME_PHASE_CLASSES),
        },
        "terminal_conversion": {
            "values": list(terminal_values),
            "mask": list(terminal_mask),
            "selected_option_indices": list(selected_indices),
        },
        "turn_resources": selected_option_vector("turn_resources", TURN_RESOURCE_LAYOUT),
        "attack_readiness": selected_option_vector("attack_readiness", ATTACK_READINESS_LAYOUT),
        "opponent_belief": copy.deepcopy(dict(belief)),
        "target_training_eligible": target_training_eligible,
    }
    if target_training_eligible:
        return result

    # A caller may inspect public transition diagnostics without a corpus
    # receipt, but cannot accidentally train on self-authored future labels.
    # Preserve values for audit while removing every supervised mask, including
    # the privileged sidecar which otherwise has its own sparse count masks.
    for name in (
        "lethal_threat",
        "prize_race",
        "action_utility",
        "game_phase",
        "terminal_conversion",
        "turn_resources",
        "attack_readiness",
    ):
        vector_row = _mapping(result[name], field=f"target vectors.{name}")
        raw_mask = _rows(vector_row.get("mask"), field=f"target vectors.{name}.mask")
        mutable = dict(vector_row)
        mutable["mask"] = [False] * len(raw_mask)
        mutable["unavailable_reason"] = "immutable_trajectory_receipt_unverified"
        result[name] = mutable
    belief_result = copy.deepcopy(
        dict(_mapping(result["opponent_belief"], field="opponent_belief"))
    )
    belief_result["available"] = False
    belief_result["reason"] = "immutable_trajectory_receipt_unverified"
    for name in ("hand_count_distribution", "remainder_count_distribution"):
        distribution = belief_result.get(name)
        if isinstance(distribution, Mapping):
            masked = dict(distribution)
            masked["pairs"] = []
            masked["mask"] = False
            masked["reason"] = "immutable_trajectory_receipt_unverified"
            belief_result[name] = masked
    result["opponent_belief"] = belief_result
    return result


def assert_public_target_invariance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> None:
    """Fail if two records differ in their public pre-decision identity.

    It intentionally permits different selected-action target outcomes after
    a hidden-state fork.  Compare target receipts separately when validating a
    concrete simulator replay.
    """

    left_hash = public_target_fingerprint(left)
    right_hash = public_target_fingerprint(right)
    if left_hash != right_hash:
        raise SimulatorRuleTargetError(
            "public rule targets differ under a privileged-hidden-state change"
        )


__all__ = [
    "ACTION_UTILITY_LAYOUT",
    "ATTACK_READINESS_LAYOUT",
    "R298_CANONICAL_CONTRACT_PATH",
    "R298_CANONICAL_CONTRACT_SHA256",
    "R298_CANONICAL_GOAL_PATH",
    "R298_CANONICAL_GOAL_SHA256",
    "R298_CANONICAL_GOAL_REVISION",
    "DeterministicPromptChain",
    "GAME_PHASE_CLASSES",
    "LETHAL_THREAT_LAYOUT",
    "PRIZE_RACE_LAYOUT",
    "PrivilegedBeliefReceiptValidator",
    "PromptChainSimulator",
    "PromptChainStep",
    "R298_CANONICAL_SIMULATOR",
    "R298_PROMPT_CHAIN_SCHEMA",
    "R298_PROMPT_CHAIN_SCHEMA_VERSION",
    "R298_PRIVILEGED_BELIEF_TARGET_SCHEMA",
    "R298_PRIVILEGED_BELIEF_TARGET_SCHEMA_VERSION",
    "R298_PRODUCTION_TYPED_SOURCE_PATH",
    "R298_PRODUCTION_TYPED_SOURCE_SHA256",
    "R298_PUBLIC_SNAPSHOT_IDENTITY_SCHEMA",
    "R298_PUBLIC_SNAPSHOT_IDENTITY_SCHEMA_VERSION",
    "R298_PREDECESSOR_CONTRACT_SHA256",
    "R298_PREDECESSOR_GOAL_REVISION",
    "R298_PREDECESSOR_GOAL_SHA256",
    "R298_R5_DERIVATIVE_LINEAGE_ID",
    "R298_R5_HANDOFF_ACTIVATION_RECEIPT_SCHEMA",
    "R298_R5_SCHEMA_FREEZE_RECEIPT_SCHEMA",
    "R298_R5_TRAJECTORY_LEDGER_STATUS",
    "R298_R5_TRAJECTORY_TRAINING_AUTHORIZATION_STATUS",
    "R298_R5_TRAINING_HOST",
    "R298_REVISION",
    "R298_ROOT_OWNER_REVISION",
    "R298_RULE_TARGET_SCHEMA",
    "R298_RULE_TARGET_SCHEMA_DIGEST",
    "R298_RULE_TARGET_SCHEMA_VERSION",
    "R298_TARGET_CONFIG_SCHEMA",
    "R298_TARGET_PROVENANCE_SCHEMA",
    "R298_TARGET_PROVENANCE_SCHEMA_VERSION",
    "R298_SEALED_TRAJECTORY_VALIDATOR_KIND",
    "R298_TRAJECTORY_LEDGER_ENTRY_SCHEMA",
    "R298_TRAJECTORY_LEDGER_SCHEMA",
    "R298_TRAJECTORY_LEDGER_SCHEMA_VERSION",
    "R298_TRAJECTORY_TRAINING_AUTHORIZATION_SCHEMA",
    "R298_TRAJECTORY_TRAINING_AUTHORIZATION_SCHEMA_VERSION",
    "SealedSelectedActionTrajectoryValidator",
    "SelectedActionTrajectoryReceiptValidator",
    "SimulatorRuleTargetError",
    "TERMINAL_CONVERSION_CLASSES",
    "TERMINAL_CONVERSION_LAYOUT",
    "TURN_RESOURCE_LAYOUT",
    "assert_public_target_invariance",
    "assert_r298_rule_target_schema_binding",
    "compile_simulator_rule_targets",
    "load_sealed_selected_action_trajectory_validator",
    "prize_yield_from_public_card",
    "public_observation_fingerprint",
    "public_target_fingerprint",
    "rule_head_target_vectors",
    "r298_rule_target_schema_manifest",
]
