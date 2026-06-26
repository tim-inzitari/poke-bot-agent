from collections import Counter
from pathlib import Path

from poke_agent.archetypes import (
    HEURISTIC_ARCHETYPE_SLUG_PATTERNS,
    load_archetype_registry,
    parse_archetype_filename,
    weighted_deck_pool,
)
from poke_agent.archetype_heuristics import (
    ARCHETYPE_DRAGAPULT,
    ARCHETYPE_LUCARIO,
    ARCHETYPE_UNKNOWN,
    BOSS_ORDERS,
    DRAKLOAK,
    DRAGAPULT_EX,
    MEGA_LUCARIO_EX,
    RIOLU,
    SOLROCK,
    opponent_matches_family,
    predict_opponent_archetype,
)


def test_parse_archetype_filename():
    assert parse_archetype_filename("dragapult1") == ("dragapult", "1")
    assert parse_archetype_filename("mega-lucario3") == ("mega-lucario", "3")


def test_registry_loads_samples_and_competitive_decks():
    root = Path(__file__).resolve().parents[1]
    registry = load_archetype_registry(root)
    assert "dragapult" in registry.priors
    assert len(registry.variants) >= 60
    competitive = [variant for variant in registry.variants if "regional" in variant.path.stem]
    assert competitive


def test_classify_dragapult_sample():
    root = Path(__file__).resolve().parents[1]
    registry = load_archetype_registry(root)
    variant = next(item for item in registry.variants if item.slug == "dragapult")
    slug, score = registry.classify_deck(variant.cards)
    assert slug == "dragapult"
    assert score > 0.5


def test_distinctive_weights_downrank_shared_cards():
    root = Path(__file__).resolve().parents[1]
    registry = load_archetype_registry(root)
    dragapult_weights = registry.distinctive_card_weights("dragapult")
    lucario_weights = registry.distinctive_card_weights("lucario-hariyama")
    assert dragapult_weights[DRAGAPULT_EX] > 0
    assert lucario_weights[MEGA_LUCARIO_EX] > 0
    # Engine cards should outweigh generic overlap.
    assert dragapult_weights[DRAGAPULT_EX] > dragapult_weights.get(BOSS_ORDERS, 0)


def test_classify_visible_archetype_returns_meta_slug():
    root = Path(__file__).resolve().parents[1]
    registry = load_archetype_registry(root)
    dragapult_visible = Counter({DRAGAPULT_EX: 1, DRAKLOAK: 1})
    lucario_visible = Counter({RIOLU: 1, SOLROCK: 1, MEGA_LUCARIO_EX: 1})
    drag_slug, drag_score = registry.classify_visible_archetype(dragapult_visible)
    luc_slug, luc_score = registry.classify_visible_archetype(lucario_visible)
    assert drag_slug.startswith("dragapult")
    assert drag_score >= 0.15
    assert luc_slug in {"lucario-hariyama", "mega-lucario"}
    assert luc_score >= 0.15


def test_classify_visible_heuristic_family_mapping():
    root = Path(__file__).resolve().parents[1]
    registry = load_archetype_registry(root)
    dragapult_visible = Counter({DRAGAPULT_EX: 1, DRAKLOAK: 1})
    family, score = registry.classify_visible_heuristic_archetype(
        dragapult_visible,
        HEURISTIC_ARCHETYPE_SLUG_PATTERNS,
    )
    assert family == ARCHETYPE_DRAGAPULT
    assert score >= 0.15


def test_predict_opponent_archetype_returns_meta_slug():
    obs = {
        "current": {
            "turn": 2,
            "players": [
                {"prize": [0] * 6, "active": [{"id": 999}], "bench": []},
                {
                    "prize": [0] * 6,
                    "active": [{"id": RIOLU}],
                    "bench": [{"id": SOLROCK}, {"id": MEGA_LUCARIO_EX}],
                },
            ],
        }
    }
    slug = predict_opponent_archetype(obs, 0)
    assert opponent_matches_family(slug, ARCHETYPE_LUCARIO)


def test_predict_opponent_archetype_unknown_without_signal():
    obs = {
        "current": {
            "turn": 1,
            "players": [
                {"prize": [0] * 6, "active": [{"id": 999}], "bench": []},
                {"prize": [0] * 6, "active": [{"id": 9000}], "bench": []},
            ],
        }
    }
    assert predict_opponent_archetype(obs, 0) == ARCHETYPE_UNKNOWN


def test_shared_visible_card_does_not_beat_full_engine_match():
    root = Path(__file__).resolve().parents[1]
    registry = load_archetype_registry(root)
    # Card 2 is a flex piece in multiple lists; engine cards should dominate.
    ambiguous = Counter({2: 1, RIOLU: 1, SOLROCK: 1, MEGA_LUCARIO_EX: 1})
    slug, _ = registry.classify_visible_archetype(ambiguous)
    assert opponent_matches_family(slug, ARCHETYPE_LUCARIO)


def test_weighted_deck_pool():
    root = Path(__file__).resolve().parents[1]
    pool = weighted_deck_pool(root)
    assert pool
    assert all(len(cards) == 60 for _, cards, _ in pool)
