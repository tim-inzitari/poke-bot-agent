"""Unit tests for opening/clarity search budget and archetype heuristics registry."""

from __future__ import annotations

from types import SimpleNamespace

from poke_bot import config, hammer_heuristics
from poke_bot.heuristics_registry import (
    apply_prior_logit_bias,
    prior_margin,
    resolve_heuristics,
)
from poke_bot.opening_budget import (
    clarity_caps_to_floor,
    is_opening_turn,
    scale_opening_budgets,
    visit_stop_triggered,
)


def test_opening_turn_scales_sims_and_time_above_floor(monkeypatch) -> None:
    monkeypatch.setattr(config.SEARCH, "opening_budget", True)
    monkeypatch.setattr(config.SEARCH, "opening_turn_max", 2)
    monkeypatch.setattr(config.SEARCH, "opening_sims_mult", 0.5)
    monkeypatch.setattr(config.SEARCH, "opening_move_time_mult", 0.5)
    assert is_opening_turn(1)
    assert not is_opening_turn(5)
    sims, move_t, applied = scale_opening_budgets(
        turn=1,
        requested_sims=256,
        move_time_s=8.0,
        min_trusted_sims=128,
    )
    assert applied
    assert sims == 192  # 128 + 0.5*(256-128)
    assert sims >= 128
    assert abs(move_t - 4.0) < 1e-9


def test_opening_never_below_trust_floor(monkeypatch) -> None:
    monkeypatch.setattr(config.SEARCH, "opening_budget", True)
    monkeypatch.setattr(config.SEARCH, "opening_sims_mult", 0.0)
    sims, _, applied = scale_opening_budgets(
        turn=0,
        requested_sims=128,
        move_time_s=8.0,
        min_trusted_sims=128,
    )
    assert applied
    assert sims == 128


def test_clarity_prior_caps_to_floor(monkeypatch) -> None:
    monkeypatch.setattr(config.SEARCH, "opening_budget", True)
    monkeypatch.setattr(config.SEARCH, "clarity_prior_margin", 0.35)
    plan, stopped = clarity_caps_to_floor(
        [0.8, 0.1, 0.1],
        min_trusted_sims=128,
        current_plan=256,
    )
    assert stopped
    assert plan == 128
    plan2, stopped2 = clarity_caps_to_floor(
        [0.4, 0.3, 0.3],
        min_trusted_sims=128,
        current_plan=256,
    )
    assert not stopped2
    assert plan2 == 256


def test_visit_stop_only_after_trust_floor(monkeypatch) -> None:
    monkeypatch.setattr(config.SEARCH, "opening_budget", True)
    monkeypatch.setattr(config.SEARCH, "clarity_visit_stop", True)
    monkeypatch.setattr(config.SEARCH, "clarity_visit_stop_p", 0.5)
    assert not visit_stop_triggered(
        [50, 10],
        sims_run=60,
        sims_plan=256,
        min_trusted_sims=128,
        elapsed_s=1.0,
        move_budget_s=8.0,
    )
    assert visit_stop_triggered(
        [200, 10],
        sims_run=128,
        sims_plan=256,
        min_trusted_sims=128,
        elapsed_s=2.0,
        move_budget_s=8.0,
    )


def test_prior_margin_and_softmax_bias() -> None:
    assert abs(prior_margin([0.7, 0.2, 0.1]) - 0.5) < 1e-9
    assert prior_margin([1.0]) == 1.0
    biased = apply_prior_logit_bias([0.5, 0.5], [2.0, 0.0])
    assert biased[0] > biased[1]
    assert abs(sum(biased) - 1.0) < 1e-9


def test_registry_hammer_and_neutral() -> None:
    # Non-hammer deck → neutral (applies fails or unknown arch).
    neutral = resolve_heuristics(deck_card_ids=[1] * 60)
    assert neutral.ENABLED is False
    bias = neutral.prior_logit_bias(None, [[0], [1]])
    assert bias == [0.0, 0.0]

    hammer = resolve_heuristics(archetype_id="hammer-pult")
    assert hammer is hammer_heuristics
    assert hammer.ENABLED is False
    assert len(hammer.prior_logit_bias(SimpleNamespace(), [[0], [1]])) == 2
