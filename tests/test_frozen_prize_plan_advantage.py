from __future__ import annotations

import pytest

from poke_bot.frozen_prize_plan_advantage import (
    ACTIVATION_RECEIPT_SCHEMA,
    H3_COEFFICIENT,
    FrozenPrizePlanAdvantageCache,
    FrozenPrizePlanPrediction,
    FrozenPrizePlanValidationError,
    PrizePlanCacheIdentity,
    PrizePlanCompleteAction,
    PortableStageAdvantage,
    bind_portable_stage_advantages,
    canonical_sha256,
    resolve_prize_plan_stage_advantages,
)


def _sha(char: str) -> str:
    return "sha256:" + char * 64


def _identity(activation_sha: str) -> PrizePlanCacheIdentity:
    return PrizePlanCacheIdentity(
        contract_sha256=_sha("1"),
        source_binding_sha256=_sha("2"),
        target_set_manifest_sha256=_sha("3"),
        critic_checkpoint_sha256=_sha("4"),
        h3_scale_support_sha256=_sha("5"),
        validation_receipt_sha256=_sha("6"),
        coefficient_configuration_sha256=_sha("7"),
        policy_checkpoint_sha256=_sha("8"),
        activation_receipt_sha256=activation_sha,
    )


def _sealed() -> tuple[PrizePlanCacheIdentity, FrozenPrizePlanAdvantageCache]:
    base_identity = _identity(_sha("9")).as_mapping()
    base_identity.pop("activation_receipt_sha256")
    receipt: dict[str, object] = {
        "schema": ACTIVATION_RECEIPT_SCHEMA,
        "activation_eligible": True,
        "actor_activation": True,
        "safe_boundary": True,
        "cache_identity_without_activation_receipt_sha256": base_identity,
    }
    receipt["artifact_sha256"] = canonical_sha256(receipt)
    identity = _identity(receipt["artifact_sha256"])
    prediction = FrozenPrizePlanPrediction(
        action_key="a",
        stage_keys=("s1", "s2"),
        alignment_sha256=_sha("a"),
        v_plan=(0.0, 0.1, 0.2, 0.3),
        q_plan=(0.9, 0.4, -0.8, 0.9),
        masks=(True, True, True, True),
        scaled_h3_advantage=2.0,
        c3=0.5,
    )
    return identity, FrozenPrizePlanAdvantageCache.from_records(
        identity=identity,
        predictions=(prediction,),
        activation_receipt=receipt,
    )


def test_disabled_is_exact_legacy_identity_without_cache_inspection() -> None:
    legacy = {"s": object()}
    assert resolve_prize_plan_stage_advantages(
        enabled=False,
        legacy_advantages_by_stage=legacy,
        cache=None,
        actions=(),
    ) is legacy


def test_enabled_formula_uses_only_h3_and_broadcasts() -> None:
    identity, cache = _sealed()
    action = PrizePlanCompleteAction(
        action_key="a",
        stage_keys=("s1", "s2"),
        alignment_sha256=_sha("a"),
        terminal_return=1.0,
        existing_state_value=-0.25,
    )
    result = cache.materialize_enabled(
        (action,), expected_identity=identity, expected_payload_sha256=cache.payload_sha256
    )
    expected = H3_COEFFICIENT * 0.5 * 2.0
    assert result.h3_additive_by_stage == {"s1": expected, "s2": expected}
    diagnostic = result.diagnostics_by_action["a"]
    assert diagnostic.actor_coefficients == (0.0, H3_COEFFICIENT, 0.0, 0.0)
    assert diagnostic.raw_plan_advantages == pytest.approx((0.9, 0.3, -1.0, 0.6))


def test_masked_h3_falls_back_to_exact_legacy_not_h1() -> None:
    identity, cache = _sealed()
    prediction = cache.predictions["a"]
    masked = FrozenPrizePlanPrediction(
        **{**prediction.__dict__, "masks": (True, False, True, True)}
    )
    masked_cache = FrozenPrizePlanAdvantageCache(
        identity=identity,
        predictions={"a": masked},
        payload_sha256=_sha("b"),
    )
    action = PrizePlanCompleteAction("a", ("s1", "s2"), _sha("a"), 0.0, -0.5)
    result = masked_cache.materialize_enabled(
        (action,), expected_identity=identity, expected_payload_sha256=_sha("b")
    )
    assert result.h3_additive_by_stage == {"s1": 0.0, "s2": 0.0}


def test_enabled_requires_safe_activation_receipt() -> None:
    identity = _identity(_sha("9"))
    prediction = FrozenPrizePlanPrediction(
        "a", ("s",), _sha("a"), (0.0,) * 4, (0.0,) * 4,
        (True,) * 4, 0.0, 1.0,
    )
    with pytest.raises(FrozenPrizePlanValidationError, match="not eligible"):
        FrozenPrizePlanAdvantageCache.from_records(
            identity=identity,
            predictions=(prediction,),
            activation_receipt={
                "schema": ACTIVATION_RECEIPT_SCHEMA,
                "activation_eligible": False,
                "actor_activation": False,
                "safe_boundary": False,
                "cache_identity_without_activation_receipt_sha256": {
                    key: value
                    for key, value in identity.as_mapping().items()
                    if key != "activation_receipt_sha256"
                },
            },
        )


def test_alignment_and_identity_fail_closed() -> None:
    identity, cache = _sealed()
    action = PrizePlanCompleteAction("a", ("s2", "s1"), _sha("a"), -1.0, 0.0)
    with pytest.raises(FrozenPrizePlanValidationError, match="stage alignment"):
        cache.materialize_enabled(
            (action,), expected_identity=identity, expected_payload_sha256=cache.payload_sha256
        )
    with pytest.raises(FrozenPrizePlanValidationError, match="identity mismatch"):
        cache.materialize_enabled(
            (PrizePlanCompleteAction("a", ("s1", "s2"), _sha("a"), -1.0, 0.0),),
            expected_identity=_identity(_sha("f")),
            expected_payload_sha256=cache.payload_sha256,
        )


def test_portable_rows_bind_to_process_local_replay_keys_with_exact_coverage() -> None:
    class Stage:
        pass

    class Decision:
        env_step = 7
        policy_stages = [Stage(), Stage()]

    class Sequence:
        episode_id = "episode-7"
        seat = 1
        decisions = [Decision()]

    sequence = Sequence()
    bound = bind_portable_stage_advantages(
        [sequence],
        [
            PortableStageAdvantage("episode-7", 1, 7, 0, 0.125),
            PortableStageAdvantage("episode-7", 1, 7, 1, 0.125),
        ],
    )
    assert bound == {(id(sequence), 0, 0): 0.125, (id(sequence), 0, 1): 0.125}


def test_portable_rows_reject_stage_drift_missing_and_surplus() -> None:
    class Stage:
        pass

    class Decision:
        env_step = 3
        policy_stages = [Stage(), Stage()]

    class Sequence:
        episode_id = "episode-3"
        seat = 0
        decisions = [Decision()]

    with pytest.raises(FrozenPrizePlanValidationError, match="not identical"):
        bind_portable_stage_advantages(
            [Sequence()],
            [
                PortableStageAdvantage("episode-3", 0, 3, 0, 0.1),
                PortableStageAdvantage("episode-3", 0, 3, 1, 0.2),
            ],
        )
    with pytest.raises(FrozenPrizePlanValidationError, match="full replay coverage"):
        bind_portable_stage_advantages(
            [Sequence()],
            [PortableStageAdvantage("episode-3", 0, 3, 0, 0.1)],
        )
