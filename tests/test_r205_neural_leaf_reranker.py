from __future__ import annotations

import hashlib
import math
from fractions import Fraction

import pytest

from poke_bot.recursive_turn_planner.neural_leaf_reranker import (
    R205_ALAKAZAM_DECK_CARDS_SHA256,
    R205_NO_RTP_BUNDLE_SHA256,
    R205_NO_RTP_CHECKPOINT_BYTES,
    R205_NO_RTP_CHECKPOINT_SHA256,
    BatchedNeuralLeafReranker,
    FrozenPolicyIdentity,
    LeafDeadline,
    LeafKind,
    LeafRequest,
    LeafScoreCalibration,
    NeuralForwardRow,
    NeuralLeafRerankerConfig,
    NeuralLeafRerankerError,
    verify_frozen_checkpoint_bytes,
)


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode()).hexdigest()}"


class _Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _ScriptedClock:
    def __init__(self, *values: float) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        if not self.values:
            raise AssertionError("scripted clock exhausted")
        return self.values.pop(0)


class _Backend:
    def __init__(
        self,
        identity: FrozenPolicyIdentity,
        rows: dict[str, NeuralForwardRow],
        *,
        on_call=None,
    ) -> None:
        self.frozen_policy_identity_sha256 = identity.identity_sha256
        self.rows = rows
        self.on_call = on_call
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, requests):
        self.calls.append(tuple(request.request_id for request in requests))
        if self.on_call is not None:
            self.on_call(len(self.calls))
        return tuple(self.rows[request.request_id] for request in requests)


def _identity() -> FrozenPolicyIdentity:
    return FrozenPolicyIdentity(
        checkpoint_sha256=_digest("checkpoint"),
        checkpoint_bytes=17,
        bundle_sha256=_digest("bundle"),
        deck_cards_sha256=_digest("deck"),
        nonplanner_runtime_sha256=_digest("runtime"),
    )


def _calibration(identity: FrozenPolicyIdentity, *, value: float, outcome: float) -> LeafScoreCalibration:
    return LeafScoreCalibration(
        receipt_sha256=_digest("calibration"),
        frozen_policy_identity_sha256=identity.identity_sha256,
        value_weight=value,
        outcome_weight=outcome,
        value_head_calibrated=value > 0.0,
        outcome_distribution_calibrated=outcome > 0.0,
        source_excluded=True,
        policy_visible_only=True,
    )


def _nonterminal(
    request_id: str,
    *,
    kind: LeafKind = LeafKind.SUCCESSOR,
    root_seat: int = 0,
    actions: tuple[tuple[int, ...], ...] = ((0,), (1,), (2,)),
    direct_action: tuple[int, ...] = (1,),
) -> LeafRequest:
    return LeafRequest(
        request_id=request_id,
        kind=kind,
        simulator_result_sha256=_digest(f"sim:{request_id}"),
        public_state_sha256=_digest(f"state:{request_id}"),
        root_seat=root_seat,
        packet=object(),
        expected_actions=actions,
        direct_action=direct_action,
    )


def _terminal(request_id: str, value: Fraction) -> LeafRequest:
    return LeafRequest(
        request_id=request_id,
        kind=LeafKind.TERMINAL,
        simulator_result_sha256=_digest(f"sim:{request_id}"),
        public_state_sha256=_digest(f"state:{request_id}"),
        root_seat=0,
        terminal_value=value,
    )


def _deadline() -> LeafDeadline:
    return LeafDeadline(turn_deadline_monotonic=20.0, action_deadline_monotonic=5.0)


def test_r205_factory_binds_the_exact_no_rtp_checkpoint_bundle_and_deck() -> None:
    identity = FrozenPolicyIdentity.r205_no_rtp()

    assert identity.checkpoint_sha256 == R205_NO_RTP_CHECKPOINT_SHA256
    assert identity.checkpoint_bytes == R205_NO_RTP_CHECKPOINT_BYTES
    assert identity.bundle_sha256 == R205_NO_RTP_BUNDLE_SHA256
    assert identity.deck_cards_sha256 == R205_ALAKAZAM_DECK_CARDS_SHA256
    assert identity.nonplanner_runtime_sha256 == R205_NO_RTP_BUNDLE_SHA256


def test_terminal_values_bypass_network_and_nonterminals_batch_in_order() -> None:
    identity = _identity()
    backend = _Backend(
        identity,
        {
            "successor": NeuralForwardRow(
                policy_logits=(1.0, 1.0, 0.0),
                value_from_actor=0.4,
                acting_seat=0,
                outcome_distribution_logits_from_actor=(
                    math.log(0.1),
                    math.log(0.2),
                    math.log(0.7),
                ),
            ),
            "boundary": NeuralForwardRow(
                policy_logits=(0.0, 1.0, 2.0),
                value_from_actor=0.4,
                acting_seat=1,
                outcome_distribution_logits_from_actor=(
                    math.log(0.1),
                    math.log(0.2),
                    math.log(0.7),
                ),
            ),
        }
    )
    reranker = BatchedNeuralLeafReranker(
        backend,
        NeuralLeafRerankerConfig(
            frozen_policy_identity=identity,
            max_microbatch_leaves=2,
            calibration=_calibration(identity, value=0.5, outcome=0.5),
        ),
        clock=_Clock(),
    )

    result = reranker.evaluate(
        (
            _terminal("terminal", Fraction(1)),
            _nonterminal("successor"),
            _nonterminal("boundary", kind=LeafKind.BOUNDARY),
        ),
        deadline=_deadline(),
    )

    assert backend.calls == [("successor", "boundary")]
    assert tuple(row.request_id for row in result.evaluations) == (
        "terminal",
        "successor",
        "boundary",
    )
    terminal, successor, boundary = result.evaluations
    assert terminal.exact_terminal_value == Fraction(1)
    assert terminal.selected_leaf_value == 1.0
    assert terminal.value_source == "exact_simulator_terminal"
    assert successor.root_policy_priors is not None
    assert successor.root_policy_priors.ranked_action_indices == (0, 1, 2)
    assert successor.root_policy_priors.direct_action_index == 1
    assert successor.raw_value_from_root == pytest.approx(0.4)
    assert successor.raw_outcome_expected_value_from_root == pytest.approx(0.6)
    assert successor.selected_leaf_value == pytest.approx(0.5)
    assert successor.search_value_eligible is True
    # The leaf belonged to the opponent, so actor-relative value/outcome signs
    # must flip into root-seat perspective before backup.
    assert boundary.raw_value_from_root == pytest.approx(-0.4)
    assert boundary.raw_outcome_expected_value_from_root == pytest.approx(-0.6)
    assert boundary.selected_leaf_value == pytest.approx(-0.5)
    assert boundary.outcome_probabilities_from_root == pytest.approx((0.7, 0.2, 0.1))

    telemetry = result.telemetry
    assert telemetry.simulator_results_seen == 3
    assert telemetry.simulator_terminal_results_seen == 1
    assert telemetry.simulator_boundary_leaves_seen == 1
    assert telemetry.neural_evaluations_started == 2
    assert telemetry.neural_evaluations_accepted == 2
    assert telemetry.root_policy_prior_evaluations == 2
    assert telemetry.outcome_head_evaluations == 2
    assert telemetry.frozen_policy_prior_batches == 1
    assert telemetry.frozen_policy_prior_evaluations == 2
    assert telemetry.batched_frozen_outcome_value_leaf_reranking_batches == 1
    assert telemetry.frozen_outcome_leaf_evaluations == 2
    assert telemetry.frozen_value_leaf_evaluations == 2
    assert telemetry.nonterminal_leaves_reranked == 2
    assert telemetry.terminal_exact_results_not_reranked == 1
    assert telemetry.result_or_leaf_evaluations_seen == 5
    assert telemetry.exact_terminal_values_used == 1
    assert telemetry.requested_leaf_batch_completed is True
    assert telemetry.every_leaf_has_search_eligible_value is True
    assert telemetry.full_tree_completion_owned_by == "simulator_mcts_tree_builder"


def test_uncalibrated_value_and_outcome_are_diagnostics_not_search_values() -> None:
    identity = _identity()
    backend = _Backend(
        identity,
        {
            "leaf": NeuralForwardRow(
                policy_logits=(0.0, 1.0, 2.0),
                value_from_actor=0.75,
                acting_seat=0,
                outcome_distribution_logits_from_actor=(0.0, 0.0, 0.0),
            )
        }
    )
    reranker = BatchedNeuralLeafReranker(
        backend,
        NeuralLeafRerankerConfig(frozen_policy_identity=identity),
        clock=_Clock(),
    )

    result = reranker.evaluate((_nonterminal("leaf"),), deadline=_deadline())

    evaluation = result.evaluations[0]
    assert evaluation.raw_value_from_root == pytest.approx(0.75)
    assert evaluation.raw_outcome_expected_value_from_root == pytest.approx(0.0)
    assert evaluation.selected_leaf_value is None
    assert evaluation.search_value_eligible is False
    assert evaluation.value_source == "untrusted_network_diagnostic"
    assert result.telemetry.requested_leaf_batch_completed is True
    assert result.telemetry.every_leaf_has_search_eligible_value is False
    assert result.telemetry.incomplete_reason == "uncalibrated_nonterminal_leaf_value"
    assert reranker.production_action_authority_enabled is False
    assert reranker.guide_logit_bonus_used is False


def test_deadline_discards_every_partial_microbatch_result() -> None:
    identity = _identity()
    clock = _Clock()
    rows = {
        label: NeuralForwardRow(
            policy_logits=(0.0, 1.0, 2.0),
            value_from_actor=0.2,
            acting_seat=0,
            outcome_distribution_logits_from_actor=(0.0, 0.0, 0.0),
        )
        for label in ("first", "second")
    }

    def _advance_on_second_call(calls: int) -> None:
        if calls == 2:
            clock.value = 5.01

    backend = _Backend(identity, rows, on_call=_advance_on_second_call)
    reranker = BatchedNeuralLeafReranker(
        backend,
        NeuralLeafRerankerConfig(
            frozen_policy_identity=identity,
            max_microbatch_leaves=1,
            calibration=_calibration(identity, value=1.0, outcome=0.0),
        ),
        clock=clock,
    )

    result = reranker.evaluate(
        (_nonterminal("first"), _nonterminal("second")),
        deadline=_deadline(),
    )

    assert backend.calls == [("first",), ("second",)]
    assert result.evaluations == ()
    assert result.telemetry.neural_evaluations_started == 2
    assert result.telemetry.neural_evaluations_accepted == 0
    assert result.telemetry.neural_forward_batches == 2
    assert result.telemetry.deadline_hit is True
    assert result.telemetry.incomplete_reason == "deadline_after_neural_microbatch"
    assert result.telemetry.requested_leaf_batch_completed is False


def test_deadline_charges_neural_output_validation_not_only_the_forward() -> None:
    identity = _identity()
    backend = _Backend(
        identity,
        {
            "leaf": NeuralForwardRow(
                policy_logits=(0.0, 1.0, 2.0),
                value_from_actor=0.2,
                acting_seat=0,
                outcome_distribution_logits_from_actor=(0.0, 0.0, 0.0),
            )
        },
    )
    # start, initial deadline, pre-forward, post-forward, validation, receipt
    # construction: the result is discarded once validation reaches 5 seconds.
    clock = _ScriptedClock(0.0, 0.0, 0.0, 5.01, 5.01)
    reranker = BatchedNeuralLeafReranker(
        backend,
        NeuralLeafRerankerConfig(
            frozen_policy_identity=identity,
            calibration=_calibration(identity, value=1.0, outcome=0.0),
        ),
        clock=clock,
    )

    result = reranker.evaluate((_nonterminal("leaf"),), deadline=_deadline())

    assert result.evaluations == ()
    assert result.telemetry.deadline_hit is True
    assert result.telemetry.incomplete_reason == "deadline_during_neural_output_validation"


def test_backend_row_count_or_policy_shape_mismatch_fails_closed() -> None:
    identity = _identity()
    leaf = _nonterminal("leaf")
    class _ShortBackend:
        frozen_policy_identity_sha256 = identity.identity_sha256

        def __call__(self, _requests):
            return ()

    reranker = BatchedNeuralLeafReranker(
        _ShortBackend(),
        NeuralLeafRerankerConfig(frozen_policy_identity=identity),
        clock=_Clock(),
    )
    short = reranker.evaluate((leaf,), deadline=_deadline())
    assert short.evaluations == ()
    assert short.telemetry.incomplete_reason == "neural_backend_row_count_mismatch"

    bad_shape = _Backend(
        identity,
        {
            "leaf": NeuralForwardRow(
                policy_logits=(0.0, 1.0),
                value_from_actor=0.0,
                acting_seat=0,
            )
        }
    )
    reranker = BatchedNeuralLeafReranker(
        bad_shape,
        NeuralLeafRerankerConfig(frozen_policy_identity=identity),
        clock=_Clock(),
    )
    malformed = reranker.evaluate((leaf,), deadline=_deadline())
    assert malformed.evaluations == ()
    assert malformed.telemetry.incomplete_reason == "invalid_neural_leaf_output"


def test_leaf_and_calibration_contracts_reject_stale_or_unauthorized_values() -> None:
    identity = _identity()
    with pytest.raises(NeuralLeafRerankerError, match="direct-policy action"):
        _nonterminal("bad-direct", direct_action=(99,))
    with pytest.raises(NeuralLeafRerankerError, match="requires outcome calibration"):
        LeafScoreCalibration(
            receipt_sha256=_digest("bad"),
            frozen_policy_identity_sha256=identity.identity_sha256,
            value_weight=0.0,
            outcome_weight=1.0,
            value_head_calibrated=False,
            outcome_distribution_calibrated=False,
            source_excluded=True,
            policy_visible_only=True,
        )
    with pytest.raises(NeuralLeafRerankerError, match="different frozen policy"):
        NeuralLeafRerankerConfig(
            frozen_policy_identity=identity,
            calibration=LeafScoreCalibration(
                receipt_sha256=_digest("other"),
                frozen_policy_identity_sha256=_digest("wrong-identity"),
                value_weight=1.0,
                outcome_weight=0.0,
                value_head_calibrated=True,
                outcome_distribution_calibrated=False,
                source_excluded=True,
                policy_visible_only=True,
            ),
        )
    class _UnboundBackend:
        def __call__(self, _requests):
            return ()

    with pytest.raises(NeuralLeafRerankerError, match="backend frozen policy identity"):
        BatchedNeuralLeafReranker(
            _UnboundBackend(),
            NeuralLeafRerankerConfig(frozen_policy_identity=identity),
            clock=_Clock(),
        )


def test_r207_requires_outcome_logits_for_every_nonterminal_leaf() -> None:
    identity = _identity()
    backend = _Backend(
        identity,
        {
            "leaf": NeuralForwardRow(
                policy_logits=(0.0, 1.0, 2.0),
                value_from_actor=0.0,
                acting_seat=0,
            )
        },
    )
    reranker = BatchedNeuralLeafReranker(
        backend,
        NeuralLeafRerankerConfig(
            frozen_policy_identity=identity,
            calibration=_calibration(identity, value=1.0, outcome=0.0),
        ),
        clock=_Clock(),
    )

    result = reranker.evaluate((_nonterminal("leaf"),), deadline=_deadline())

    assert result.evaluations == ()
    assert result.telemetry.incomplete_reason == "invalid_neural_leaf_output"


def test_checkpoint_verifier_binds_bytes_and_digest(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    identity = FrozenPolicyIdentity(
        checkpoint_sha256=f"sha256:{hashlib.sha256(checkpoint.read_bytes()).hexdigest()}",
        checkpoint_bytes=checkpoint.stat().st_size,
        bundle_sha256=_digest("bundle"),
        deck_cards_sha256=_digest("deck"),
        nonplanner_runtime_sha256=_digest("runtime"),
    )

    assert verify_frozen_checkpoint_bytes(checkpoint, identity) == checkpoint.resolve()
    wrong_size = FrozenPolicyIdentity(
        checkpoint_sha256=identity.checkpoint_sha256,
        checkpoint_bytes=identity.checkpoint_bytes + 1,
        bundle_sha256=identity.bundle_sha256,
        deck_cards_sha256=identity.deck_cards_sha256,
        nonplanner_runtime_sha256=identity.nonplanner_runtime_sha256,
    )
    with pytest.raises(NeuralLeafRerankerError, match="byte count"):
        verify_frozen_checkpoint_bytes(checkpoint, wrong_size)
