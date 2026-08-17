from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from poke_bot.guide2vec_bo1000 import (
    CANDIDATE_GUIDE2VEC_PRESENCE,
    CONTROL_ARM,
    CONTROL_GUIDE2VEC_PRESENCE,
    GUIDE2VEC_ARM,
    GUIDE2VEC_BO1000_GAME_COUNT,
    GUIDE2VEC_BO1000_PAIR_COUNT,
    GUIDE2VEC_BO1000_REPORT_SCHEMA,
    GUIDE2VEC_EVALUATION_ID,
    R195_BUNDLE_SHA256,
    R195_CHECKPOINT_BYTES,
    R195_CHECKPOINT_SHA256,
    R195_DECK_CARDS_SHA256,
    R195_DECK_ID,
    R195_MATCHUP_TREE_SHA256,
    R195_SUBMISSION_ID,
    R195_SUBMISSION_MESSAGE,
    R212_CONTRACT_SHA256,
    FrozenR195RuntimeIdentity,
    Guide2VecBO1000Error,
    Guide2VecBO1000GameReceipt,
    Guide2VecDecisionReceipt,
    Guide2VecExperimentIdentity,
    build_guide2vec_bo1000_schedule,
    compile_guide2vec_bo1000_report,
    expected_control_guide2vec_absence_attestation,
    expected_is_first_attestation,
    expected_matchup_adapter_parity_attestation,
)

ROOT = Path(__file__).resolve().parents[1]


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


def _experiment() -> Guide2VecExperimentIdentity:
    return Guide2VecExperimentIdentity(
        base_runtime=FrozenR195RuntimeIdentity(
            model_config_sha256=_digest("r195-model-config"),
            matchup_tree_sha256=(
                "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
            ),
            matchup_adapter_bank_sha256=_digest("r195-trained-matchup-adapter-bank"),
            matchup_adapter_training_receipt_sha256=_digest(
                "r195-matchup-adapter-training-receipt"
            ),
            matchup_adapter_runtime_graph_sha256=_digest(
                "r195-matchup-adapter-runtime-graph"
            ),
            matchup_adapter_enabled=True,
            matchup_adapter_trained=True,
            matchup_adapter_frozen=True,
            direct_runtime_graph_sha256=_digest("r195-direct-runtime-graph"),
        ),
        guide2vec_checkpoint_sha256=_digest("r212-guide2vec-checkpoint"),
        guide2vec_training_receipt_sha256=_digest("r212-guide2vec-training"),
        guide2vec_runtime_config_sha256=_digest("r212-guide2vec-runtime"),
        guide2vec_parameter_count=192_384,
        candidate_runtime_graph_sha256=_digest(
            "r195-direct-plus-frozen-guide2vec-runtime-graph"
        ),
        control_runtime_graph_sha256=_digest("r195-direct-runtime-graph"),
        candidate_guide2vec_component_graph_sha256=_digest(
            "r212-guide2vec-component-graph"
        ),
        runtime_graph_difference_receipt_sha256=_digest(
            "r212-runtime-graph-difference-receipt"
        ),
        source_snapshot_sha256=_digest("r212-source-snapshot"),
        evaluation_output_identity_sha256=_digest("r212-isolated-bo1000-output"),
    )


def _decision(spec, *, decision_index: int = 0) -> Guide2VecDecisionReceipt:
    eligible = spec.pair_index % 2 == 0
    applied = eligible
    changed = eligible and spec.game_index == 0
    direct_action = _digest(f"{spec.game_nonce_sha256}:direct:{decision_index}")
    final_action = (
        _digest(f"{spec.game_nonce_sha256}:final:{decision_index}")
        if changed
        else direct_action
    )
    return Guide2VecDecisionReceipt(
        game_nonce_sha256=spec.game_nonce_sha256,
        decision_index=decision_index,
        acting_seat=spec.guide2vec_seat,
        legal_option_count=3,
        eligible=eligible,
        abstained=not eligible,
        bonus_applied=applied,
        action_changed_from_direct_policy=changed,
        max_applied_logit_bonus=0.05 if applied else 0.0,
        direct_action_sha256=direct_action,
        final_action_sha256=final_action,
        legal_options_sha256=_digest(f"{spec.game_nonce_sha256}:legal"),
        guide2vec_scores_sha256=_digest(f"{spec.game_nonce_sha256}:scores"),
        guide2vec_action_latency_seconds=0.001,
        total_action_latency_seconds=0.004,
    )


def _receipt(
    spec,
    experiment: Guide2VecExperimentIdentity,
    *,
    observed_first_actor_seat: int | None = None,
    terminal_status: str = "completed",
) -> Guide2VecBO1000GameReceipt:
    first_seat = (
        spec.sealed_initial_first_actor_seat
        if observed_first_actor_seat is None
        else observed_first_actor_seat
    )
    guide_is_first = first_seat == spec.guide2vec_seat
    control_is_first = not guide_is_first
    first_arm = GUIDE2VEC_ARM if guide_is_first else CONTROL_ARM
    control_graph_observation = _digest(
        f"{spec.game_nonce_sha256}:control-runtime-graph-observation"
    )
    return Guide2VecBO1000GameReceipt(
        game_nonce_sha256=spec.game_nonce_sha256,
        pair_id=spec.pair_id,
        game_index=spec.game_index,
        guide2vec_seat=spec.guide2vec_seat,
        control_seat=spec.control_seat,
        pair_initial_rng_sha256=spec.pair_initial_rng_sha256,
        pair_deck_order_rng_sha256=spec.pair_deck_order_rng_sha256,
        sealed_initial_first_actor_seat=spec.sealed_initial_first_actor_seat,
        observed_first_actor_seat=first_seat,
        observed_first_actor_arm=first_arm,
        guide2vec_is_first=guide_is_first,
        control_is_first=control_is_first,
        is_first_attestation_sha256=expected_is_first_attestation(
            game_nonce_sha256=spec.game_nonce_sha256,
            observed_first_actor_seat=first_seat,
            guide2vec_seat=spec.guide2vec_seat,
            control_seat=spec.control_seat,
            observed_first_actor_arm=first_arm,
            guide2vec_is_first=guide_is_first,
            control_is_first=control_is_first,
        ),
        turn_order_observation_sha256=_digest(f"{spec.game_nonce_sha256}:is-first"),
        base_runtime_identity_sha256=experiment.base_runtime.identity_sha256,
        experiment_identity_sha256=experiment.identity_sha256,
        evaluation_output_identity_sha256=(
            experiment.evaluation_output_identity_sha256
        ),
        candidate_runtime_graph_sha256=experiment.candidate_runtime_graph_sha256,
        candidate_base_runtime_graph_sha256=(
            experiment.base_runtime.direct_runtime_graph_sha256
        ),
        control_runtime_graph_sha256=experiment.control_runtime_graph_sha256,
        candidate_matchup_tree_sha256=experiment.base_runtime.matchup_tree_sha256,
        control_matchup_tree_sha256=experiment.base_runtime.matchup_tree_sha256,
        candidate_matchup_adapter_bank_sha256=(
            experiment.base_runtime.matchup_adapter_bank_sha256
        ),
        control_matchup_adapter_bank_sha256=(
            experiment.base_runtime.matchup_adapter_bank_sha256
        ),
        candidate_matchup_adapter_training_receipt_sha256=(
            experiment.base_runtime.matchup_adapter_training_receipt_sha256
        ),
        control_matchup_adapter_training_receipt_sha256=(
            experiment.base_runtime.matchup_adapter_training_receipt_sha256
        ),
        candidate_matchup_adapter_runtime_graph_sha256=(
            experiment.base_runtime.matchup_adapter_runtime_graph_sha256
        ),
        control_matchup_adapter_runtime_graph_sha256=(
            experiment.base_runtime.matchup_adapter_runtime_graph_sha256
        ),
        candidate_matchup_adapter_enabled=True,
        control_matchup_adapter_enabled=True,
        candidate_matchup_adapter_trained=True,
        control_matchup_adapter_trained=True,
        candidate_matchup_adapter_frozen=True,
        control_matchup_adapter_frozen=True,
        matchup_adapter_parity_attestation_sha256=(
            expected_matchup_adapter_parity_attestation(
                game_nonce_sha256=spec.game_nonce_sha256,
                candidate_matchup_tree_sha256=(
                    experiment.base_runtime.matchup_tree_sha256
                ),
                control_matchup_tree_sha256=(
                    experiment.base_runtime.matchup_tree_sha256
                ),
                candidate_matchup_adapter_bank_sha256=(
                    experiment.base_runtime.matchup_adapter_bank_sha256
                ),
                control_matchup_adapter_bank_sha256=(
                    experiment.base_runtime.matchup_adapter_bank_sha256
                ),
                candidate_matchup_adapter_training_receipt_sha256=(
                    experiment.base_runtime.matchup_adapter_training_receipt_sha256
                ),
                control_matchup_adapter_training_receipt_sha256=(
                    experiment.base_runtime.matchup_adapter_training_receipt_sha256
                ),
                candidate_matchup_adapter_runtime_graph_sha256=(
                    experiment.base_runtime.matchup_adapter_runtime_graph_sha256
                ),
                control_matchup_adapter_runtime_graph_sha256=(
                    experiment.base_runtime.matchup_adapter_runtime_graph_sha256
                ),
                candidate_matchup_adapter_enabled=True,
                control_matchup_adapter_enabled=True,
                candidate_matchup_adapter_trained=True,
                control_matchup_adapter_trained=True,
                candidate_matchup_adapter_frozen=True,
                control_matchup_adapter_frozen=True,
            )
        ),
        candidate_guide2vec_component_graph_sha256=(
            experiment.candidate_guide2vec_component_graph_sha256
        ),
        runtime_graph_difference_receipt_sha256=(
            experiment.runtime_graph_difference_receipt_sha256
        ),
        candidate_guide2vec_checkpoint_sha256=(experiment.guide2vec_checkpoint_sha256),
        candidate_guide2vec_training_receipt_sha256=(
            experiment.guide2vec_training_receipt_sha256
        ),
        candidate_guide2vec_runtime_config_sha256=(
            experiment.guide2vec_runtime_config_sha256
        ),
        candidate_guide2vec_presence=CANDIDATE_GUIDE2VEC_PRESENCE,
        candidate_guide2vec_module_instance_count=1,
        candidate_guide2vec_parameter_count=experiment.guide2vec_parameter_count,
        candidate_guide2vec_frozen=True,
        control_guide2vec_presence=CONTROL_GUIDE2VEC_PRESENCE,
        control_guide2vec_module_instance_count=0,
        control_guide2vec_parameter_count=0,
        control_guide2vec_state_dict_key_count=0,
        control_guide2vec_forward_hook_count=0,
        control_guide2vec_linear_transform_count=0,
        control_guide2vec_disabled_or_zeroed=False,
        control_runtime_graph_observation_sha256=control_graph_observation,
        control_guide2vec_absence_attestation_sha256=(
            expected_control_guide2vec_absence_attestation(
                game_nonce_sha256=spec.game_nonce_sha256,
                control_runtime_graph_sha256=(experiment.control_runtime_graph_sha256),
                control_runtime_graph_observation_sha256=(control_graph_observation),
            )
        ),
        guide2vec_execution_mode="bounded_guide_logit_bonus",
        control_execution_mode="frozen_r195_no_rtp_direct_policy",
        terminal_status=terminal_status,
        winner_seat=(
            None
            if terminal_status == "failed_closed"
            else (
                spec.guide2vec_seat if spec.pair_index % 2 == 0 else spec.control_seat
            )
        ),
        illegal_action_count=0,
        forfeit_count=0,
        crash_count=1 if terminal_status == "failed_closed" else 0,
        timeout_count=0,
        guide2vec_decisions=(
            () if terminal_status == "failed_closed" else (_decision(spec),)
        ),
    )


def _schedule_and_receipts():
    experiment = _experiment()
    schedule = build_guide2vec_bo1000_schedule(_digest("sealed-rng-seed"), experiment)
    return experiment, schedule, [_receipt(spec, experiment) for spec in schedule]


def _adapter_parity_attestation(
    receipt: Guide2VecBO1000GameReceipt, **changes: object
) -> str:
    fields = {
        "game_nonce_sha256": receipt.game_nonce_sha256,
        "candidate_matchup_tree_sha256": receipt.candidate_matchup_tree_sha256,
        "control_matchup_tree_sha256": receipt.control_matchup_tree_sha256,
        "candidate_matchup_adapter_bank_sha256": (
            receipt.candidate_matchup_adapter_bank_sha256
        ),
        "control_matchup_adapter_bank_sha256": (
            receipt.control_matchup_adapter_bank_sha256
        ),
        "candidate_matchup_adapter_training_receipt_sha256": (
            receipt.candidate_matchup_adapter_training_receipt_sha256
        ),
        "control_matchup_adapter_training_receipt_sha256": (
            receipt.control_matchup_adapter_training_receipt_sha256
        ),
        "candidate_matchup_adapter_runtime_graph_sha256": (
            receipt.candidate_matchup_adapter_runtime_graph_sha256
        ),
        "control_matchup_adapter_runtime_graph_sha256": (
            receipt.control_matchup_adapter_runtime_graph_sha256
        ),
        "candidate_matchup_adapter_enabled": receipt.candidate_matchup_adapter_enabled,
        "control_matchup_adapter_enabled": receipt.control_matchup_adapter_enabled,
        "candidate_matchup_adapter_trained": receipt.candidate_matchup_adapter_trained,
        "control_matchup_adapter_trained": receipt.control_matchup_adapter_trained,
        "candidate_matchup_adapter_frozen": receipt.candidate_matchup_adapter_frozen,
        "control_matchup_adapter_frozen": receipt.control_matchup_adapter_frozen,
    }
    fields.update(changes)
    return expected_matchup_adapter_parity_attestation(**fields)


def test_schedule_is_exact_500_sealed_pairs_with_candidate_seat_swap_and_actual_order_shape() -> (
    None
):
    experiment, schedule, _ = _schedule_and_receipts()

    assert len(schedule) == GUIDE2VEC_BO1000_GAME_COUNT
    assert len({spec.pair_id for spec in schedule}) == GUIDE2VEC_BO1000_PAIR_COUNT
    assert sum(spec.guide2vec_seat == 0 for spec in schedule) == 500
    assert sum(spec.guide2vec_seat == 1 for spec in schedule) == 500
    assert {spec.sealed_initial_first_actor_seat for spec in schedule} == {0, 1}
    assert all(
        spec.experiment_identity_sha256 == experiment.identity_sha256
        and spec.evaluation_output_identity_sha256
        == experiment.evaluation_output_identity_sha256
        for spec in schedule
    )

    for pair_index in range(GUIDE2VEC_BO1000_PAIR_COUNT):
        pair = [spec for spec in schedule if spec.pair_index == pair_index]
        assert [(spec.game_index, spec.guide2vec_seat) for spec in pair] == [
            (0, 0),
            (1, 1),
        ]
        assert len({spec.pair_initial_rng_sha256 for spec in pair}) == 1
        assert len({spec.pair_deck_order_rng_sha256 for spec in pair}) == 1
        assert len({spec.sealed_initial_first_actor_seat for spec in pair}) == 1
        first_seat = pair[0].sealed_initial_first_actor_seat
        assert sum(spec.guide2vec_seat == first_seat for spec in pair) == 1


def test_compiler_requires_exact_r195_both_arms_and_reports_actual_first_second_balance() -> (
    None
):
    experiment, schedule, receipts = _schedule_and_receipts()
    report = compile_guide2vec_bo1000_report(
        schedule,
        receipts,
        experiment=experiment,
    )

    assert report["schema"] == GUIDE2VEC_BO1000_REPORT_SCHEMA
    assert report["status"] == "complete"
    assert report["support"] == {
        "scheduled_games": 1000,
        "observed_terminal_game_receipts": 1000,
        "rng_matched_pairs": 500,
        "guide2vec_as_seat_0": 500,
        "guide2vec_as_seat_1": 500,
        "guide2vec_actual_first": 500,
        "guide2vec_actual_second": 500,
        "no_rtp_actual_first": 500,
        "no_rtp_actual_second": 500,
        "paired_analysis_eligible_pairs": 500,
        "paired_analysis_excluded_failed_closed_pairs": 0,
    }
    identities = report["identities"]
    assert identities["evaluation_id"] == GUIDE2VEC_EVALUATION_ID
    assert identities["candidate_arm"] == GUIDE2VEC_ARM
    assert identities["control_arm"] == CONTROL_ARM
    assert identities["r212_contract_sha256"] == R212_CONTRACT_SHA256
    frozen = identities["frozen_r195_base_runtime"]
    assert frozen["submission_id"] == R195_SUBMISSION_ID
    assert frozen["submission_message"] == R195_SUBMISSION_MESSAGE
    assert frozen["checkpoint_sha256"] == R195_CHECKPOINT_SHA256
    assert frozen["checkpoint_bytes"] == R195_CHECKPOINT_BYTES
    assert frozen["bundle_sha256"] == R195_BUNDLE_SHA256
    assert frozen["deck_id"] == R195_DECK_ID
    assert frozen["deck_cards_sha256"] == R195_DECK_CARDS_SHA256
    assert frozen["matchup_tree_sha256"] == R195_MATCHUP_TREE_SHA256
    matchup_adapter = identities["matchup_adapter"]
    assert matchup_adapter == {
        "public_matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
        "frozen_trained_adapter_bank_sha256": (
            experiment.base_runtime.matchup_adapter_bank_sha256
        ),
        "training_receipt_sha256": (
            experiment.base_runtime.matchup_adapter_training_receipt_sha256
        ),
        "shared_adapter_runtime_graph_sha256": (
            experiment.base_runtime.matchup_adapter_runtime_graph_sha256
        ),
        "candidate_enabled": True,
        "control_enabled": True,
        "candidate_trained": True,
        "control_trained": True,
        "candidate_frozen": True,
        "control_frozen": True,
        "parity_attested_games": 1000,
    }
    runtime_graphs = identities["runtime_graphs"]
    assert runtime_graphs["candidate_full_graph_sha256"] == (
        experiment.candidate_runtime_graph_sha256
    )
    assert (
        runtime_graphs["candidate_base_graph_sha256"]
        == (runtime_graphs["control_direct_graph_sha256"])
    )
    assert (
        runtime_graphs["candidate_full_graph_sha256"]
        != (runtime_graphs["control_direct_graph_sha256"])
    )
    assert runtime_graphs["candidate_guide2vec_presence"] == (
        CANDIDATE_GUIDE2VEC_PRESENCE
    )
    assert runtime_graphs["control_guide2vec_presence"] == (CONTROL_GUIDE2VEC_PRESENCE)
    assert runtime_graphs["control_guide2vec_module_instance_count"] == 0
    assert runtime_graphs["control_guide2vec_parameter_count"] == 0
    assert runtime_graphs["control_guide2vec_forward_hook_count"] == 0
    assert runtime_graphs["control_guide2vec_linear_transform_count"] == 0
    assert runtime_graphs["control_guide2vec_disabled_or_zeroed"] is False
    assert runtime_graphs["control_absence_attested_games"] == 1000
    assert (
        report["game_outcomes"]["guide2vec_by_actual_turn_order"]["first"]["win"]
        + report["game_outcomes"]["guide2vec_by_actual_turn_order"]["first"]["loss"]
        == 500
    )
    assert report["guide2vec_decisions"] == {
        "candidate_decision_count": 1000,
        "eligible_stage_count": 500,
        "abstain_stage_count": 500,
        "bonus_applied_stage_count": 500,
        "action_change_count": 250,
        "eligibility_rate": 0.5,
        "abstain_rate_over_all_stages": 0.5,
        "abstain_rate_over_eligible_stages": 0.0,
        "bonus_applied_rate_over_eligible_stages": 1.0,
        "action_change_rate_over_eligible_stages": 0.5,
    }
    assert report["latency"]["guide2vec_action_latency_seconds"][
        "mean"
    ] == pytest.approx(0.001)
    assert report["latency"]["total_action_latency_seconds"]["mean"] == pytest.approx(
        0.004
    )
    assert report["receipt_integrity"] == {
        "raw_per_game_and_per_decision_receipts_preserved": True,
        "missing_receipts_may_be_imputed": False,
        "paired_failed_closed_outcomes_imputed": False,
        "only_candidate_delta_is_frozen_bounded_guide2vec_bonus": True,
        "candidate_and_control_runtime_graphs_separately_bound": True,
        "frozen_r195_matchup_adapter_enabled_on_both_arms": True,
        "candidate_and_control_matchup_adapter_graph_identical": True,
        "control_guide2vec_module_absent_in_all_receipts": True,
        "control_guide2vec_parameter_or_state_key_count": 0,
        "control_guide2vec_hook_or_linear_transform_count": 0,
        "control_disabled_or_zeroed_substitute_allowed": False,
        "initial_actor_is_explicit_sealed_pair_material_not_inferred_from_seat": True,
        "all_1000_terminal_receipts_required": True,
    }
    assert report["authority"]["training_eligible"] is False
    assert report["canonical_sha256"].startswith("sha256:")


def test_compiler_fails_closed_for_missing_duplicate_rng_or_actual_first_actor_evidence() -> (
    None
):
    experiment, schedule, receipts = _schedule_and_receipts()

    with pytest.raises(Guide2VecBO1000Error, match="missing 1"):
        compile_guide2vec_bo1000_report(schedule, receipts[:-1], experiment=experiment)
    with pytest.raises(Guide2VecBO1000Error, match="duplicate game receipt"):
        compile_guide2vec_bo1000_report(
            schedule, [*receipts, receipts[0]], experiment=experiment
        )
    rng_mismatch = replace(
        receipts[1],
        pair_initial_rng_sha256=_digest("wrong-initial-rng"),
    )
    with pytest.raises(Guide2VecBO1000Error, match="sealed initial RNG"):
        compile_guide2vec_bo1000_report(
            schedule,
            [receipts[0], rng_mismatch, *receipts[2:]],
            experiment=experiment,
        )

    spec = schedule[0]
    crossed_order = _receipt(
        spec,
        experiment,
        observed_first_actor_seat=1 - spec.sealed_initial_first_actor_seat,
    )
    with pytest.raises(Guide2VecBO1000Error, match="observed first actor"):
        compile_guide2vec_bo1000_report(
            schedule,
            [crossed_order, *receipts[1:]],
            experiment=experiment,
        )


def test_compiler_rejects_base_sidecar_or_output_identity_substitution() -> None:
    experiment, schedule, receipts = _schedule_and_receipts()
    with pytest.raises(Guide2VecBO1000Error, match="base/candidate/output identity"):
        compile_guide2vec_bo1000_report(
            schedule,
            [
                replace(
                    receipts[0],
                    base_runtime_identity_sha256=_digest("not-r195-base"),
                ),
                *receipts[1:],
            ],
            experiment=experiment,
        )
    with pytest.raises(Guide2VecBO1000Error, match="Guide2Vec identity"):
        compile_guide2vec_bo1000_report(
            schedule,
            [
                replace(
                    receipts[0],
                    candidate_guide2vec_checkpoint_sha256=_digest(
                        "different-guide2vec"
                    ),
                ),
                *receipts[1:],
            ],
            experiment=experiment,
        )
    with pytest.raises(Guide2VecBO1000Error, match="base/candidate/output identity"):
        compile_guide2vec_bo1000_report(
            schedule,
            [
                replace(
                    receipts[0],
                    evaluation_output_identity_sha256=_digest("shared-old-output"),
                ),
                *receipts[1:],
            ],
            experiment=experiment,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "control_guide2vec_presence",
            "disabled",
            "absent, not disabled or zeroed",
        ),
        (
            "control_guide2vec_module_instance_count",
            1,
            "module_instance_count must be zero",
        ),
        (
            "control_guide2vec_parameter_count",
            192_384,
            "parameter_count must be zero",
        ),
        (
            "control_guide2vec_state_dict_key_count",
            8,
            "state_dict_key_count must be zero",
        ),
        (
            "control_guide2vec_forward_hook_count",
            1,
            "forward_hook_count must be zero",
        ),
        (
            "control_guide2vec_linear_transform_count",
            1,
            "linear_transform_count must be zero",
        ),
        (
            "control_guide2vec_disabled_or_zeroed",
            True,
            "cannot instantiate a disabled or zeroed",
        ),
    ),
)
def test_control_fails_closed_on_any_guide2vec_component_parameter_or_hook_claim(
    field: str,
    value: object,
    message: str,
) -> None:
    _, _, receipts = _schedule_and_receipts()
    with pytest.raises(Guide2VecBO1000Error, match=message):
        replace(receipts[0], **{field: value})


def test_candidate_alone_loads_frozen_guide2vec_and_runtime_graphs_cannot_cross() -> (
    None
):
    experiment, schedule, receipts = _schedule_and_receipts()
    with pytest.raises(Guide2VecBO1000Error, match="exactly one Guide2Vec"):
        replace(receipts[0], candidate_guide2vec_module_instance_count=0)
    with pytest.raises(Guide2VecBO1000Error, match="must be frozen"):
        replace(receipts[0], candidate_guide2vec_frozen=False)

    wrong_control_graph = _digest("wrong-control-runtime-graph")
    observation = receipts[0].control_runtime_graph_observation_sha256
    crossed = replace(
        receipts[0],
        candidate_base_runtime_graph_sha256=wrong_control_graph,
        control_runtime_graph_sha256=wrong_control_graph,
        control_guide2vec_absence_attestation_sha256=(
            expected_control_guide2vec_absence_attestation(
                game_nonce_sha256=receipts[0].game_nonce_sha256,
                control_runtime_graph_sha256=wrong_control_graph,
                control_runtime_graph_observation_sha256=observation,
            )
        ),
    )
    with pytest.raises(Guide2VecBO1000Error, match="runtime graphs"):
        compile_guide2vec_bo1000_report(
            schedule,
            [crossed, *receipts[1:]],
            experiment=experiment,
        )


@pytest.mark.parametrize(
    "field",
    (
        "candidate_matchup_adapter_enabled",
        "control_matchup_adapter_enabled",
        "candidate_matchup_adapter_trained",
        "control_matchup_adapter_trained",
        "candidate_matchup_adapter_frozen",
        "control_matchup_adapter_frozen",
    ),
)
def test_both_arms_require_the_same_enabled_trained_frozen_r195_adapter(
    field: str,
) -> None:
    _, _, receipts = _schedule_and_receipts()

    with pytest.raises(
        Guide2VecBO1000Error,
        match="both arms must enable the same trained, frozen r195 matchup adapter",
    ):
        replace(receipts[0], **{field: False})


def test_matchup_adapter_tree_bank_graph_and_parity_digest_fail_closed() -> None:
    _, _, receipts = _schedule_and_receipts()
    receipt = receipts[0]

    with pytest.raises(Guide2VecBO1000Error, match="exact public r195 matchup tree"):
        replace(receipt, candidate_matchup_tree_sha256=_digest("private-tree"))
    with pytest.raises(Guide2VecBO1000Error, match="adapter bank digests"):
        replace(
            receipt,
            candidate_matchup_adapter_bank_sha256=_digest("different-adapter-bank"),
        )
    with pytest.raises(Guide2VecBO1000Error, match="adapter runtime graph digests"):
        replace(
            receipt,
            candidate_matchup_adapter_runtime_graph_sha256=_digest(
                "different-adapter-runtime-graph"
            ),
        )
    with pytest.raises(Guide2VecBO1000Error, match="parity attestation"):
        replace(
            receipt,
            matchup_adapter_parity_attestation_sha256=_digest(
                "wrong-adapter-parity-attestation"
            ),
        )


def test_compiler_rejects_a_consistent_adapter_substitution_on_both_arms() -> None:
    experiment, schedule, receipts = _schedule_and_receipts()
    substituted_bank = _digest("substituted-trained-adapter-bank")
    substituted_graph = _digest("substituted-adapter-runtime-graph")
    substituted = replace(
        receipts[0],
        candidate_matchup_adapter_bank_sha256=substituted_bank,
        control_matchup_adapter_bank_sha256=substituted_bank,
        candidate_matchup_adapter_runtime_graph_sha256=substituted_graph,
        control_matchup_adapter_runtime_graph_sha256=substituted_graph,
        matchup_adapter_parity_attestation_sha256=_adapter_parity_attestation(
            receipts[0],
            candidate_matchup_adapter_bank_sha256=substituted_bank,
            control_matchup_adapter_bank_sha256=substituted_bank,
            candidate_matchup_adapter_runtime_graph_sha256=substituted_graph,
            control_matchup_adapter_runtime_graph_sha256=substituted_graph,
        ),
    )

    with pytest.raises(Guide2VecBO1000Error, match="frozen r195 adapter"):
        compile_guide2vec_bo1000_report(
            schedule,
            [substituted, *receipts[1:]],
            experiment=experiment,
        )


def test_base_runtime_identity_requires_adapter_on_trained_and_frozen() -> None:
    base = _experiment().base_runtime

    for field in (
        "matchup_adapter_enabled",
        "matchup_adapter_trained",
        "matchup_adapter_frozen",
    ):
        with pytest.raises(Guide2VecBO1000Error, match="enabled, trained, and frozen"):
            replace(base, **{field: False})


def test_payload_parsers_reject_search_or_rtp_claim_fields_and_non_direct_modes() -> (
    None
):
    experiment, schedule, receipts = _schedule_and_receipts()
    del experiment, schedule
    payload = receipts[0].as_payload()
    payload["mcts_turns"] = []
    with pytest.raises(Guide2VecBO1000Error, match="unknown=.*mcts_turns"):
        Guide2VecBO1000GameReceipt.from_payload(payload)

    payload = receipts[0].as_payload()
    payload["rtp_telemetry"] = {}
    with pytest.raises(Guide2VecBO1000Error, match="unknown=.*rtp_telemetry"):
        Guide2VecBO1000GameReceipt.from_payload(payload)

    payload = receipts[0].as_payload()
    payload["control_guide2vec_component"] = {"mode": "disabled"}
    with pytest.raises(
        Guide2VecBO1000Error, match="unknown=.*control_guide2vec_component"
    ):
        Guide2VecBO1000GameReceipt.from_payload(payload)

    payload = receipts[0].as_payload()
    payload["guide2vec_execution_mode"] = "search_reranker"
    with pytest.raises(Guide2VecBO1000Error, match="bounded direct-policy bonus"):
        Guide2VecBO1000GameReceipt.from_payload(payload)

    payload = receipts[0].as_payload()
    payload["control_execution_mode"] = "rtp_direct_policy"
    with pytest.raises(Guide2VecBO1000Error, match="exact frozen direct policy"):
        Guide2VecBO1000GameReceipt.from_payload(payload)


def test_runtime_failures_preserve_all_1000_receipts_but_fail_closed_outcome_claims() -> (
    None
):
    experiment, schedule, receipts = _schedule_and_receipts()
    failed = replace(
        receipts[0],
        terminal_status="failed_closed",
        winner_seat=None,
        crash_count=1,
        guide2vec_decisions=(),
    )
    report = compile_guide2vec_bo1000_report(
        schedule,
        [failed, *receipts[1:]],
        experiment=experiment,
    )
    assert report["status"] == "failed_closed_complete_runtime_evidence"
    assert report["support"]["observed_terminal_game_receipts"] == 1000
    assert report["game_outcomes"]["paired_guide2vec_score"] is None
    assert report["game_outcomes"]["paired_win_rate_difference"] is None
    assert report["game_outcomes"]["paired_confidence_interval"] is None


def test_typed_contract_digest_and_module_isolation_are_bound_without_an_engine_runner() -> (
    None
):
    contract_path = ROOT / "state/alakazam-guide2vec-no-mcts-bo1000-r212.json"
    assert f"sha256:{hashlib.sha256(contract_path.read_bytes()).hexdigest()}" == (
        R212_CONTRACT_SHA256
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["bo1000_evaluation"]["evaluation_id"] == GUIDE2VEC_EVALUATION_ID
    assert contract["bo1000_evaluation"]["arms"] == [GUIDE2VEC_ARM, CONTROL_ARM]
    assert (
        contract["bo1000_evaluation"]["actual_turn_order_balance"][
            "initial_actor_is_explicit_sealed_pair_material_not_inferred_from_seat"
        ]
        is True
    )

    source = (ROOT / "poke_bot/guide2vec_bo1000.py").read_text(encoding="utf-8")
    assert "recursive_turn_planner" not in source
    assert "HostLocal" not in source
