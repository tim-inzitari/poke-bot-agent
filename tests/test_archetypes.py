from pathlib import Path

from poke_agent.archetypes import load_archetype_registry, parse_archetype_filename, weighted_deck_pool


def test_parse_archetype_filename():
    assert parse_archetype_filename("dragapult1") == ("dragapult", "1")
    assert parse_archetype_filename("mega-lucario3") == ("mega-lucario", "3")


def test_registry_loads_samples():
    root = Path(__file__).resolve().parents[1]
    registry = load_archetype_registry(root)
    assert "dragapult" in registry.priors
    assert len(registry.variants) >= 60


def test_classify_dragapult_sample():
    root = Path(__file__).resolve().parents[1]
    registry = load_archetype_registry(root)
    variant = next(item for item in registry.variants if item.slug == "dragapult")
    slug, score = registry.classify_deck(variant.cards)
    assert slug == "dragapult"
    assert score > 0.5


def test_weighted_deck_pool():
    root = Path(__file__).resolve().parents[1]
    pool = weighted_deck_pool(root)
    assert pool
    assert all(len(cards) == 60 for _, cards, _ in pool)
