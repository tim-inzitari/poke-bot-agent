from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.train_pure_rl import (
    _population_collect_specs,
    population_cycle_rehearsal_due,
)


def test_population_collection_requires_exact_complete_own_roster() -> None:
    ids = tuple(f"specialist-{index:02d}" for index in range(22))
    by_id = {
        specialist_id: SimpleNamespace(id=specialist_id)
        for specialist_id in ids
    }

    rows = _population_collect_specs(
        enabled=True,
        frozen_specialist_ids=ids,
        by_id=by_id,
    )

    assert rows is not None
    assert [row.id for row in rows] == list(ids)


def test_population_collection_rejects_incomplete_or_external_field() -> None:
    ids = tuple(f"specialist-{index:02d}" for index in range(21))
    by_id = {
        **{
            specialist_id: SimpleNamespace(id=specialist_id)
            for specialist_id in ids
        },
        "public-agent": SimpleNamespace(id="public-agent"),
    }

    with pytest.raises(RuntimeError, match="exactly 22"):
        _population_collect_specs(
            enabled=True,
            frozen_specialist_ids=ids,
            by_id=by_id,
        )


def test_population_collection_accepts_current_and_history_registry() -> None:
    ids = tuple(f"specialist-{index:02d}" for index in range(22))
    by_id = {
        specialist_id: SimpleNamespace(
            id=specialist_id,
            path=f"/packages/{specialist_id}",
        )
        for specialist_id in ids
    }
    registry = {
        "schema": "poke_bot.population_opponent_registry/v1",
        "member_count": 22,
        "specialist_ids": list(ids),
        "external_agents_training_eligible": False,
        "opponents": [
            {
                "specialist_id": specialist_id,
                "opponent_id": specialist_id,
                "external_agent": False,
                "content_digest": f"digest-{specialist_id}",
            }
            for specialist_id in ids
        ],
    }
    from poke_bot import baselines_runtime

    original = baselines_runtime.baseline_content_digest
    baselines_runtime.baseline_content_digest = lambda path: (
        "digest-" + str(path).rsplit("/", 1)[-1]
    )
    try:
        rows = _population_collect_specs(
            enabled=True,
            frozen_specialist_ids=(),
            by_id=by_id,
            opponent_registry=registry,
        )
    finally:
        baselines_runtime.baseline_content_digest = original
    assert rows is not None
    assert [row.id for row in rows] == list(ids)


def test_population_collection_disabled_does_not_change_baseline_path() -> None:
    assert (
        _population_collect_specs(
            enabled=False,
            frozen_specialist_ids=(),
            by_id={},
        )
        is None
    )


def test_population_cycle_closes_after_exact_five_rl_epochs() -> None:
    assert population_cycle_rehearsal_due(
        population_enabled=True,
        next_iteration=5,
        configured_rehearsal_every=5,
        configured_rehearsal_epochs=5,
    )
    assert not population_cycle_rehearsal_due(
        population_enabled=False,
        next_iteration=5,
        configured_rehearsal_every=5,
        configured_rehearsal_epochs=5,
    )


@pytest.mark.parametrize(
    ("every", "epochs"),
    [(4, 5), (5, 4), (10, 5), (5, 10)],
)
def test_population_cycle_rejects_schedule_drift(
    every: int,
    epochs: int,
) -> None:
    with pytest.raises(RuntimeError, match="exact 5-RL/5-rehearsal"):
        population_cycle_rehearsal_due(
            population_enabled=True,
            next_iteration=5,
            configured_rehearsal_every=every,
            configured_rehearsal_epochs=epochs,
        )
