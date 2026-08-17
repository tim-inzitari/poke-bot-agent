from __future__ import annotations

import math

import pytest

from poke_bot.frozen_critic_advantage import (
    CANARY_PRIZE_H1_COEFFICIENT,
    CompleteAction,
    CriticCacheIdentity,
    FrozenCriticAdvantageCache,
    FrozenCriticPrediction,
    FrozenCriticValidationError,
    canonical_sha256,
    resolve_stage_advantages,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _identity() -> CriticCacheIdentity:
    return CriticCacheIdentity(
        contract_sha256=_digest("a"),
        source_sha256=_digest("b"),
        split_sha256=_digest("c"),
        feature_schema_sha256=_digest("d"),
        action_schema_sha256=_digest("e"),
        target_schema_sha256=_digest("f"),
        coefficient_sha256=_digest("1"),
        critic_checkpoint_sha256=_digest("2"),
        policy_checkpoint_sha256=_digest("4"),
    )


def _action(
    key: str = "episode-1/seat-0/action-3",
    stages: tuple[str, ...] = ("stage-1", "stage-2"),
    *,
    terminal_return: float = 1.0,
    existing_state_value: float = 0.5,
    alignment: str | None = None,
) -> CompleteAction:
    return CompleteAction(
        action_key=key,
        stage_keys=stages,
        alignment_sha256=alignment or _digest("3"),
        terminal_return=terminal_return,
        existing_state_value=existing_state_value,
    )


def _prediction(
    key: str = "episode-1/seat-0/action-3",
    stages: tuple[str, ...] = ("stage-1", "stage-2"),
    *,
    alignment: str | None = None,
    v_win_probability: float = 0.75,
    q_win_probability: float = 0.8,
    v_prize: tuple[float, float, float] = (0.1, -0.2, 0.3),
    q_prize: tuple[float, float, float] = (0.7, 0.6, -0.4),
    masks: tuple[bool, bool, bool] = (True, True, True),
) -> FrozenCriticPrediction:
    return FrozenCriticPrediction(
        action_key=key,
        stage_keys=stages,
        alignment_sha256=alignment or _digest("3"),
        v_win_probability=v_win_probability,
        q_win_probability=q_win_probability,
        v_prize=v_prize,
        q_prize=q_prize,
        prize_masks=masks,
    )


def _cache(*predictions: FrozenCriticPrediction) -> FrozenCriticAdvantageCache:
    return FrozenCriticAdvantageCache.from_records(
        identity=_identity(), predictions=predictions or (_prediction(),)
    )


def test_enabled_revision_21_formula_broadcasts_one_complete_action_scalar() -> None:
    action = _action(terminal_return=1.0, existing_state_value=-0.2)
    prediction = _prediction(
        v_win_probability=0.99,
        q_win_probability=0.02,
        v_prize=(0.1, -0.8, 0.5),
        q_prize=(0.7, 0.9, -0.9),
        masks=(True, True, True),
    )
    cache = _cache(prediction)
    result = cache.materialize_enabled(
        (action,),
        expected_identity=_identity(),
        expected_payload_sha256=cache.payload_sha256,
    )

    # Revision 21 keeps z - V_existing: 1 - (-0.2) + 0.05 * (0.7 - 0.1) == 1.23.
    # The deliberately extreme binary win probability must not alter the actor term.
    assert result.advantages_by_stage == {"stage-1": 1.23, "stage-2": 1.23}
    assert result.advantages_by_stage["stage-1"] == result.advantages_by_stage["stage-2"]
    diagnostics = result.diagnostics_by_action[action.action_key]
    assert diagnostics.actor_coefficients == (
        0.0,
        0.0,
        CANARY_PRIZE_H1_COEFFICIENT,
        0.0,
        0.0,
    )
    assert diagnostics.legacy_terminal_advantage == pytest.approx(1.2)
    assert diagnostics.q_win_probability == 0.02
    assert diagnostics.prize_advantages == pytest.approx((0.6, 1.7, -1.4))
    assert diagnostics.q_win_minus_v_win_probability == pytest.approx(-0.97)


def test_enabled_canary_masks_only_horizon_one_prize_term() -> None:
    action = _action(terminal_return=0.0, existing_state_value=-0.5)
    masked = _prediction(
        v_win_probability=0.25,
        v_prize=(-1.0, -1.0, -1.0),
        q_prize=(1.0, 1.0, 1.0),
        masks=(False, True, True),
    )
    cache = _cache(masked)
    result = cache.materialize_enabled(
        (action,),
        expected_identity=_identity(),
        expected_payload_sha256=cache.payload_sha256,
    )

    # The h1 predictor values are diagnostic only when the causal interval is absent.
    assert result.advantages_by_stage["stage-1"] == 0.5
    assert result.diagnostics_by_action[action.action_key].prize_advantages == (2.0, 2.0, 2.0)


def test_qwin_and_horizons_two_three_have_zero_actor_effect() -> None:
    action = _action(terminal_return=-1.0, existing_state_value=0.3)
    base = _prediction(
        q_win_probability=0.0,
        v_prize=(0.0, 0.0, 0.0),
        q_prize=(0.0, -1.0, -1.0),
        masks=(True, True, True),
    )
    changed_diagnostics = _prediction(
        v_win_probability=1.0,
        q_win_probability=1.0,
        v_prize=(0.0, -1.0, -1.0),
        q_prize=(0.0, 1.0, 1.0),
        masks=(True, False, False),
    )

    base_cache = _cache(base)
    changed_cache = _cache(changed_diagnostics)
    base_result = base_cache.materialize_enabled(
        (action,),
        expected_identity=_identity(),
        expected_payload_sha256=base_cache.payload_sha256,
    )
    changed_result = changed_cache.materialize_enabled(
        (action,),
        expected_identity=_identity(),
        expected_payload_sha256=changed_cache.payload_sha256,
    )
    assert changed_result.advantages_by_stage == base_result.advantages_by_stage
    assert changed_result.diagnostics_by_action[action.action_key].q_win_probability == 1.0
    assert changed_result.diagnostics_by_action[action.action_key].prize_advantages[1:] == (2.0, 2.0)


def test_disabled_branch_returns_supplied_legacy_mapping_without_touching_invalid_cache() -> None:
    legacy = {"stage-1": object(), "stage-2": float("nan")}
    result = resolve_stage_advantages(
        enabled=False,
        legacy_advantages_by_stage=legacy,
        cache=None,
        actions=(),
    )
    assert result is legacy


def test_enabled_branch_requires_cache_and_full_action_coverage() -> None:
    with pytest.raises(FrozenCriticValidationError, match="requires a cache"):
        resolve_stage_advantages(
            enabled=True,
            legacy_advantages_by_stage={},
            cache=None,
            actions=(_action(),),
            expected_identity=_identity(),
            expected_payload_sha256=_digest("7"),
        )

    with pytest.raises(FrozenCriticValidationError, match="full-coverage mismatch"):
        cache = _cache()
        cache.materialize_enabled(
            (_action(key="a"),),
            expected_identity=_identity(),
            expected_payload_sha256=cache.payload_sha256,
        )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda prediction: FrozenCriticPrediction(
                **{**prediction.as_mapping(), "v_win_probability": float("nan")}
            ),
            "v_win_probability must be finite",
        ),
        (
            lambda prediction: FrozenCriticPrediction(
                **{**prediction.as_mapping(), "q_win_probability": 1.01}
            ),
            "q_win_probability must be in",
        ),
        (
            lambda prediction: FrozenCriticPrediction(
                **{**prediction.as_mapping(), "q_prize": (0.0, 0.0, -1.1)}
            ),
            "q_prize\\[2\\] must be in",
        ),
        (
            lambda prediction: FrozenCriticPrediction(
                **{**prediction.as_mapping(), "prize_masks": (1, False, True)}
            ),
            "prize_masks\\[0\\] must be a boolean",
        ),
    ],
)
def test_prediction_rejects_nonfinite_out_of_range_and_nonboolean_values(
    mutator: object, match: str
) -> None:
    with pytest.raises(FrozenCriticValidationError, match=match):
        mutator(_prediction())  # type: ignore[operator]


@pytest.mark.parametrize("terminal_return", (-1, 0, 1))
def test_terminal_return_accepts_exact_loss_draw_win_values(terminal_return: int) -> None:
    assert _action(terminal_return=terminal_return).terminal_return == float(terminal_return)


@pytest.mark.parametrize("terminal_return", (-0.5, 0.5, -0.999999, 0.999999))
def test_terminal_return_rejects_in_range_nonterminal_values(terminal_return: float) -> None:
    with pytest.raises(
        FrozenCriticValidationError,
        match=r"terminal_return must be one of \{-1\.0, 0\.0, 1\.0\}",
    ):
        _action(terminal_return=terminal_return)


def test_terminal_return_must_be_finite_and_in_terminal_range() -> None:
    with pytest.raises(FrozenCriticValidationError, match="terminal_return must be finite"):
        _action(terminal_return=math.inf)
    with pytest.raises(FrozenCriticValidationError, match="terminal_return must be in"):
        _action(terminal_return=-1.01)
    with pytest.raises(FrozenCriticValidationError, match="existing_state_value must be finite"):
        _action(existing_state_value=math.nan)
    with pytest.raises(FrozenCriticValidationError, match="existing_state_value must be in"):
        _action(existing_state_value=1.01)


def test_strict_alignment_rejects_stage_order_and_digest_mismatches() -> None:
    prediction = _prediction(stages=("stage-1", "stage-2"), alignment=_digest("3"))
    cache = _cache(prediction)
    with pytest.raises(FrozenCriticValidationError, match="factorized stage alignment mismatch"):
        cache.materialize_enabled(
            (_action(stages=("stage-2", "stage-1")),),
            expected_identity=_identity(),
            expected_payload_sha256=cache.payload_sha256,
        )
    with pytest.raises(FrozenCriticValidationError, match="alignment digest mismatch"):
        cache.materialize_enabled(
            (_action(alignment=_digest("4")),),
            expected_identity=_identity(),
            expected_payload_sha256=cache.payload_sha256,
        )


def test_strict_coverage_rejects_duplicate_actions_and_stage_keys() -> None:
    duplicate_action = _prediction(key="a")
    with pytest.raises(FrozenCriticValidationError, match="duplicate prediction action_key"):
        _cache(duplicate_action, duplicate_action)

    with pytest.raises(FrozenCriticValidationError, match="more than one complete action"):
        _cache(
            _prediction(key="a", stages=("shared",)),
            _prediction(key="b", stages=("shared",)),
        )

    with pytest.raises(FrozenCriticValidationError, match="current complete action"):
        cache = _cache(
            _prediction(key="a", stages=("a-stage",)),
            _prediction(key="b", stages=("b-stage",)),
        )
        cache.materialize_enabled(
            (
                _action(key="a", stages=("shared",)),
                _action(key="b", stages=("shared",)),
            ),
            expected_identity=_identity(),
            expected_payload_sha256=cache.payload_sha256,
        )


def test_artifact_digest_and_identity_bindings_are_verified_strictly() -> None:
    cache = _cache()
    artifact = cache.as_artifact()
    assert artifact["payload_sha256"] == canonical_sha256(
        {key: artifact[key] for key in ("schema", "identity", "predictions")}
    )
    restored = FrozenCriticAdvantageCache.from_artifact(artifact)
    assert restored.payload_sha256 == cache.payload_sha256

    corrupted = dict(artifact)
    corrupted["payload_sha256"] = _digest("0")
    with pytest.raises(FrozenCriticValidationError, match="payload digest mismatch"):
        FrozenCriticAdvantageCache.from_artifact(corrupted)

    bad_identity = _identity().as_mapping()
    bad_identity["coefficient_sha256"] = "sha256:not-a-digest"
    with pytest.raises(FrozenCriticValidationError, match="coefficient_sha256"):
        CriticCacheIdentity.from_mapping(bad_identity)


def test_enabled_materialization_requires_exact_current_identity() -> None:
    cache = _cache()
    different_identity = CriticCacheIdentity(
        **{**_identity().as_mapping(), "split_sha256": _digest("9")}
    )
    with pytest.raises(FrozenCriticValidationError, match="critic cache identity mismatch"):
        cache.materialize_enabled(
            (_action(),),
            expected_identity=different_identity,
            expected_payload_sha256=cache.payload_sha256,
        )
    with pytest.raises(FrozenCriticValidationError, match="expected cache identity"):
        resolve_stage_advantages(
            enabled=True,
            legacy_advantages_by_stage={},
            cache=cache,
            actions=(_action(),),
        )

    with pytest.raises(FrozenCriticValidationError, match="payload digest"):
        cache.materialize_enabled(
            (_action(),),
            expected_identity=_identity(),
            expected_payload_sha256=_digest("8"),
        )


def test_enabled_output_is_immutable() -> None:
    cache = _cache()
    result = cache.materialize_enabled(
        (_action(),),
        expected_identity=_identity(),
        expected_payload_sha256=cache.payload_sha256,
    )
    with pytest.raises(TypeError):
        result.advantages_by_stage["stage-1"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        result.diagnostics_by_action["new"] = result.diagnostics_by_action[  # type: ignore[index]
            "episode-1/seat-0/action-3"
        ]
