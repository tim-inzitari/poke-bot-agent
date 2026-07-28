from __future__ import annotations

import pytest

from poke_bot import deck_guides
from poke_bot import garchomp_heuristics as garchomp
from poke_bot import grimmsnarl_heuristics as grim
from poke_bot import rockets_mewtwo_heuristics as rockets
from scripts.train_pure_rl import _parse_args


GRIM_DECK = (
    [grim.MARNIES_IMPIDIMP] * 2
    + [grim.MARNIES_MORGREM] * 2
    + [grim.MARNIES_GRIMMSNARL_EX] * 2
)


def _player(*, hand: list[int] | None = None, bench: list[int] | None = None) -> dict:
    return {
        "active": [{"id": grim.MARNIES_IMPIDIMP}],
        "bench": [{"id": value} for value in (bench or [])],
        "hand": [{"id": value} for value in (hand or [])],
        "discard": [],
        "deckCount": 30,
        "prize": [None] * 6,
    }


def _obs(me: dict, options: list[dict], *, context: int = 0) -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "players": [me, _player()],
            "stadium": [],
            "looking": [],
        },
        "select": {
            "context": context,
            "option": options,
            "minCount": 1,
            "maxCount": 1,
        },
    }


def test_generic_registry_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POKEBOT_CURRENT_DECK_GUIDE", raising=False)
    monkeypatch.delenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", raising=False)
    monkeypatch.delenv("POKEBOT_ALAKAZAM_GUIDE_TARGETS", raising=False)
    assert not deck_guides.enabled()
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "unknown")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")
    with pytest.raises(RuntimeError, match="unknown current-deck guide"):
        deck_guides.enabled()


def test_grimmsnarl_guide_prefers_core_evolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "marnie-s-grimmsnarl-ex")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")
    me = _player(hand=[grim.MARNIES_GRIMMSNARL_EX])
    obs = _obs(
        me,
        [
            {"type": grim.OPT_EVOLVE, "area": 2, "index": 0},
            {"type": grim.OPT_END},
        ],
    )
    scores = deck_guides.guide_scores(obs, [[0], [1]], deck=GRIM_DECK)
    assert scores is not None
    assert scores[0] > scores[1] + grim.ABSTENTION_MARGIN


def test_grimmsnarl_guide_masks_wrong_deck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "marnie-s-grimmsnarl-ex")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")
    obs = _obs(_player(), [{"type": grim.OPT_END}, {"type": grim.OPT_ATTACK}])
    assert deck_guides.guide_scores(obs, [[0], [1]], deck=[1] * 60) is None


def test_garchomp_guide_prefers_visible_core_evolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "garchomp")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")
    me = {
        **_player(),
        "active": [{"id": garchomp.CYNTHIAS_GIBLE}],
        "hand": [{"id": garchomp.CYNTHIAS_GABITE}],
    }
    obs = _obs(
        me,
        [
            {"type": garchomp.OPT_EVOLVE, "area": 2, "index": 0},
            {"type": grim.OPT_END},
        ],
    )
    deck = (
        [garchomp.CYNTHIAS_GIBLE] * 2
        + [garchomp.CYNTHIAS_GABITE] * 2
        + [garchomp.CYNTHIAS_GARCHOMP_EX] * 2
    )

    scores = deck_guides.guide_scores(obs, [[0], [1]], deck=deck)

    assert scores is not None
    assert scores[0] > scores[1] + garchomp.ABSTENTION_MARGIN


def test_garchomp_guide_masks_wrong_deck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "garchomp")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")
    obs = _obs(_player(), [{"type": grim.OPT_END}, {"type": grim.OPT_ATTACK}])
    assert deck_guides.guide_scores(obs, [[0], [1]], deck=[1] * 60) is None


def test_rockets_mewtwo_guide_prefers_visible_spidops_evolution(
    monkeypatch,
) -> None:
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "rockets-mewtwo")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")
    me = _player()
    me["active"] = [{"id": rockets.TEAM_ROCKETS_TAROUNTULA}]
    me["hand"] = [{"id": rockets.TEAM_ROCKETS_SPIDOPS}]
    obs = _obs(
        me,
        [
            {"type": rockets.OPT_EVOLVE, "area": 2, "index": 0},
            {"type": 14},
        ],
        context=rockets.CTX_MAIN,
    )
    deck = (
        [rockets.TEAM_ROCKETS_TAROUNTULA] * 2
        + [rockets.TEAM_ROCKETS_SPIDOPS] * 2
        + [rockets.TEAM_ROCKETS_MEWTWO_EX] * 2
        + [1] * 54
    )
    scores = deck_guides.guide_scores(obs, [[0], [1]], deck=deck)
    assert scores is not None
    assert scores[0] > scores[1] + rockets.ABSTENTION_MARGIN


def test_rockets_mewtwo_guide_masks_wrong_deck(
    monkeypatch,
) -> None:
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "rockets-mewtwo")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")
    obs = _obs(
        _player(),
        [{"type": 14}, {"type": 14}],
        context=rockets.CTX_MAIN,
    )
    assert deck_guides.guide_scores(obs, [[0], [1]], deck=[1] * 60) is None


def test_rockets_mewtwo_guide_prefers_exact_proton_setup(
    monkeypatch,
) -> None:
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "rockets-mewtwo")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")
    me = _player()
    me["active"] = [{"id": rockets.TEAM_ROCKETS_MEWTWO_EX}]
    obs = _obs(
        me,
        [
            {"type": 3, "area": rockets.AREA_DECK, "index": 0},
            {"type": 3, "area": rockets.AREA_DECK, "index": 1},
        ],
        context=99,
    )
    obs["select"]["effect"] = {"id": rockets.TEAM_ROCKETS_PROTON}
    obs["select"]["deck"] = [
        {"id": rockets.TEAM_ROCKETS_TAROUNTULA},
        {"id": 1},
    ]
    deck = (
        [rockets.TEAM_ROCKETS_TAROUNTULA] * 2
        + [rockets.TEAM_ROCKETS_SPIDOPS] * 2
        + [rockets.TEAM_ROCKETS_MEWTWO_EX] * 2
        + [1] * 54
    )
    scores = deck_guides.guide_scores(obs, [[0], [1]], deck=deck)
    assert scores is not None
    assert scores[0] > scores[1] + rockets.ABSTENTION_MARGIN


def test_rockets_mewtwo_guide_masks_unsupported_prompt(
    monkeypatch,
) -> None:
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "rockets-mewtwo")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")
    obs = _obs(
        _player(),
        [{"type": 3, "area": rockets.AREA_DECK, "index": 0}],
        context=99,
    )
    obs["select"]["effect"] = {"id": 999}
    obs["select"]["deck"] = [{"id": rockets.TEAM_ROCKETS_TAROUNTULA}]
    deck = (
        [rockets.TEAM_ROCKETS_TAROUNTULA] * 2
        + [rockets.TEAM_ROCKETS_SPIDOPS] * 2
        + [rockets.TEAM_ROCKETS_MEWTWO_EX] * 2
    )
    assert deck_guides.guide_scores(obs, [[0]], deck=deck) is None


def test_trainer_accepts_matching_generic_guide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "marnie-s-grimmsnarl-ex")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")
    args = _parse_args(
        [
            "--run-name",
            "guided-grimmsnarl",
            "--mode",
            "specialist",
            "--specialist-archetype",
            "marnie-s-grimmsnarl-ex",
            "--current-deck-guide-loss-weight",
            "0.05",
        ]
    )
    assert args.alakazam_guide_loss_weight == pytest.approx(0.05)


def test_trainer_rejects_mismatched_generic_guide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "marnie-s-grimmsnarl-ex")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--run-name",
                "mismatched-guide",
                "--mode",
                "specialist",
                "--specialist-archetype",
                "dudunsparce",
                "--current-deck-guide-loss-weight",
                "0.05",
            ]
        )


def test_future_spidops_guide_is_registered_but_not_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POKEBOT_CURRENT_DECK_GUIDE", raising=False)
    monkeypatch.delenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", raising=False)

    assert "team-rockets-spidops" in deck_guides.supported_ids()
    assert deck_guides.selected_id() is None
    assert deck_guides.enabled() is False
