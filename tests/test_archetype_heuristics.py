from __future__ import annotations

from poke_agent.archetype_heuristics import (
    ARCHETYPE_DRAGAPULT,
    ARCHETYPE_LUCARIO,
    ARCHETYPE_UNKNOWN,
    DRAGAPULT_EX,
    DRAKLOAK,
    DREEPY,
    HARIYAMA,
    LUNATONE,
    MEGA_BRAVE_ATTACK,
    MEGA_LUCARIO_EX,
    PHANTOM_DIVE_ATTACK,
    RIOLU,
    SOLROCK,
    classify_archetype,
    game_phase,
    heuristic_for_deck,
    matchup_context,
    resolve_policy_beta,
    resolve_value_shaping_weight,
    score_action,
    value_shaping_bonus,
)
from poke_agent.search_targets import blend_distributions, heuristic_distribution_over_actions


def _obs(*, turn: int, own_prizes: int, opp_prizes: int, opp_in_play=None, options=None, hand=None):
    opp_in_play = opp_in_play or []
    return {
        "current": {
            "turn": turn,
            "players": [
                {
                    "prize": [0] * own_prizes,
                    "active": [{"id": 999, "hp": 1, "maxHp": 1}],
                    "bench": [],
                    "hand": hand or [],
                },
                {
                    "prize": [0] * opp_prizes,
                    "active": [opp_in_play[0]] if opp_in_play else [{"id": 500, "hp": 1, "maxHp": 1}],
                    "bench": [{"id": cid, "hp": 1, "maxHp": 2} for cid in opp_in_play[1:]],
                },
            ],
        },
        "select": {"option": options or []},
    }


def test_classify_archetype_signature_lines():
    assert classify_archetype([DREEPY, DREEPY, DRAKLOAK, DRAGAPULT_EX, DRAGAPULT_EX]) == ARCHETYPE_DRAGAPULT
    assert classify_archetype([RIOLU, MEGA_LUCARIO_EX, MEGA_LUCARIO_EX, SOLROCK, LUNATONE]) == ARCHETYPE_LUCARIO
    assert classify_archetype([1, 2, 3, 4, 5]) == ARCHETYPE_UNKNOWN
    assert classify_archetype(None) == ARCHETYPE_UNKNOWN


def test_resolve_betas_and_weights_are_archetype_asymmetric():
    # Lucario (linear) gets the stronger prior/shaping than Dragapult (non-linear).
    assert resolve_policy_beta(ARCHETYPE_LUCARIO, lucario_beta=0.35, dragapult_beta=0.15) == 0.35
    assert resolve_policy_beta(ARCHETYPE_DRAGAPULT, lucario_beta=0.35, dragapult_beta=0.15) == 0.15
    assert resolve_policy_beta(ARCHETYPE_UNKNOWN, lucario_beta=0.35, dragapult_beta=0.15) == 0.0
    assert resolve_value_shaping_weight(ARCHETYPE_LUCARIO, lucario_weight=0.12, dragapult_weight=0.08) == 0.12
    assert resolve_value_shaping_weight(ARCHETYPE_DRAGAPULT, lucario_weight=0.12, dragapult_weight=0.08) == 0.08


def test_game_phase_transitions():
    assert game_phase(_obs(turn=1, own_prizes=6, opp_prizes=6), 0) == "early"
    assert game_phase(_obs(turn=5, own_prizes=4, opp_prizes=4), 0) == "mid"
    assert game_phase(_obs(turn=9, own_prizes=2, opp_prizes=3), 0) == "late"


def test_lucario_finisher_prefers_mega_brave_over_ending():
    obs = _obs(
        turn=6,
        own_prizes=2,
        opp_prizes=3,
        options=[{"type": 13, "attackId": MEGA_BRAVE_ATTACK}, {"type": 14}],
    )
    phase = game_phase(obs, 0)
    context = matchup_context(obs, 0, ARCHETYPE_LUCARIO)
    mega = score_action([0], obs, 0, archetype=ARCHETYPE_LUCARIO, phase=phase, context=context)
    end = score_action([1], obs, 0, archetype=ARCHETYPE_LUCARIO, phase=phase, context=context)
    assert mega > end


def test_lucario_evolve_into_mega_or_hariyama_scores_high():
    obs = _obs(
        turn=3,
        own_prizes=5,
        opp_prizes=5,
        options=[{"type": 9, "index": 0}, {"type": 14}],
        hand=[{"id": MEGA_LUCARIO_EX}],
    )
    evolve = score_action([0], obs, 0, archetype=ARCHETYPE_LUCARIO, phase="mid", context="default")
    end = score_action([1], obs, 0, archetype=ARCHETYPE_LUCARIO, phase="mid", context="default")
    assert evolve > end


def test_dragapult_mirror_reduces_early_aggression():
    options = [{"type": 13, "attackId": PHANTOM_DIVE_ATTACK}]
    default_obs = _obs(
        turn=1,
        own_prizes=6,
        opp_prizes=6,
        opp_in_play=[{"id": 9000, "hp": 1, "maxHp": 1}, 9001, 9002],
        options=options,
    )
    mirror_obs = _obs(
        turn=1,
        own_prizes=6,
        opp_prizes=6,
        opp_in_play=[{"id": DRAGAPULT_EX, "hp": 1, "maxHp": 1}, DRAKLOAK, 9002],
        options=options,
    )

    default_ctx = matchup_context(default_obs, 0, ARCHETYPE_DRAGAPULT)
    mirror_ctx = matchup_context(mirror_obs, 0, ARCHETYPE_DRAGAPULT)
    assert mirror_ctx == "mirror"
    assert default_ctx != "mirror"

    pd_default = score_action([0], default_obs, 0, archetype=ARCHETYPE_DRAGAPULT, phase="early", context=default_ctx)
    pd_mirror = score_action([0], mirror_obs, 0, archetype=ARCHETYPE_DRAGAPULT, phase="early", context=mirror_ctx)
    assert pd_mirror < pd_default


def test_dragapult_stall_and_spread_favor_different_phases():
    # Itchy Pollen (stall) is favored early; Phantom Dive (spread) holds up in mid.
    stall_opts = [{"type": 13, "attackId": 323}]
    spread_opts = [{"type": 13, "attackId": PHANTOM_DIVE_ATTACK}]
    stall_early = score_action([0], _obs(turn=1, own_prizes=6, opp_prizes=6, options=stall_opts), 0, archetype=ARCHETYPE_DRAGAPULT, phase="early", context="default")
    stall_mid = score_action([0], _obs(turn=5, own_prizes=4, opp_prizes=4, options=stall_opts), 0, archetype=ARCHETYPE_DRAGAPULT, phase="mid", context="default")
    spread_mid = score_action([0], _obs(turn=5, own_prizes=4, opp_prizes=4, opp_in_play=[{"id": 500}, 1, 2], options=spread_opts), 0, archetype=ARCHETYPE_DRAGAPULT, phase="mid", context="default")
    assert stall_early > stall_mid  # stall decays out of the early window
    assert spread_mid > stall_mid   # in mid, spread beats stalling


def test_value_shaping_bonus_is_bounded():
    obs_before = _obs(turn=5, own_prizes=4, opp_prizes=5)
    obs_after = _obs(turn=5, own_prizes=2, opp_prizes=5)  # took 2 of our prizes (multi-KO)
    bonus = value_shaping_bonus(
        obs_before, obs_after, 0, archetype=ARCHETYPE_DRAGAPULT, phase="mid", context="default"
    )
    assert 0.0 < bonus <= 0.05

    none_bonus = value_shaping_bonus(
        obs_before, obs_after, 0, archetype=ARCHETYPE_UNKNOWN, phase="mid", context="default"
    )
    assert none_bonus == 0.0


def test_unknown_archetype_scores_zero():
    obs = _obs(turn=3, own_prizes=5, opp_prizes=5, options=[{"type": 13, "attackId": MEGA_BRAVE_ATTACK}])
    assert score_action([0], obs, 0, archetype=ARCHETYPE_UNKNOWN, phase="mid", context="default") == 0.0


def test_heuristic_for_deck_binds_archetype_and_beta():
    heuristic = heuristic_for_deck(
        [DREEPY, DREEPY, DRAKLOAK, DRAGAPULT_EX, DRAGAPULT_EX],
        lucario_beta=0.35,
        dragapult_beta=0.15,
    )
    assert heuristic.archetype == ARCHETYPE_DRAGAPULT
    assert heuristic.beta == 0.15
    assert heuristic.active


def test_blend_distributions_renormalizes():
    blended = blend_distributions({"a": 0.8, "b": 0.2}, {"a": 0.0, "b": 1.0}, mix=0.5)
    assert abs(sum(blended.values()) - 1.0) < 1e-9
    assert blended["b"] > blended["a"]
    # mix=0 returns the base untouched
    assert blend_distributions({"a": 1.0}, {"b": 1.0}, mix=0.0) == {"a": 1.0}


def test_heuristic_distribution_over_actions_sums_to_one():
    def scorer(action):
        return 1.0 if action == [0] else 0.0

    dist = heuristic_distribution_over_actions([[0], [1]], scorer)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    # higher-scored action gets more mass
    import json

    assert dist[json.dumps([0], separators=(",", ":"))] > dist[json.dumps([1], separators=(",", ":"))]
