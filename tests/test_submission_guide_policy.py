from __future__ import annotations

from poke_bot.submission_guide_policy import SCHEMA, select_index


def _config(weight: float) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "mode": "guide_logit_bonus",
        "guide_id": "alakazam",
        "rtp_enabled": False,
        "guide_logit_weight": weight,
        "minimum_unique_margin": 1e-8,
    }


def test_guide_unavailable_falls_back_exactly_to_model() -> None:
    selected, audit = select_index(
        observation={},
        candidates=[[0], [1]],
        model_policy=[0.2, 0.8],
        model_index=1,
        deck=[1],
        config=_config(0.05),
        scorer=lambda *_: None,
    )
    assert selected == 1
    assert audit["reason"] == "guide_unavailable"


def test_weighted_guide_applies_exact_bounded_logit_bonus() -> None:
    selected, audit = select_index(
        observation={},
        candidates=[[0], [1]],
        model_policy=[0.501, 0.499],
        model_index=0,
        deck=[1],
        config=_config(0.05),
        scorer=lambda *_: [0.0, 4.0],
    )
    assert selected == 1
    assert audit["guide_logit_weight"] == 0.05
    assert audit["normalized_guide_scores"] == [0.0, 1.0]


def test_weighted_guide_cannot_overcome_a_large_model_margin() -> None:
    selected, audit = select_index(
        observation={},
        candidates=[[0], [1]],
        model_policy=[0.9, 0.1],
        model_index=0,
        deck=[1],
        config=_config(0.05),
        scorer=lambda *_: [0.0, 9.0],
    )
    assert selected == 0
    assert audit["reason"] == "bounded_guide_logit_bonus"
