from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_agent.baseline_agents import (
    default_baseline_manifest,
    load_baseline_manifest,
    load_kaggle_submission_agent,
    summarize_baseline_eval,
    write_default_baseline_manifest,
)
from poke_agent.self_play import choose_pool_deck


def test_default_baseline_manifest_has_four_agents():
    agents = default_baseline_manifest()
    ids = {entry["id"] for entry in agents}
    assert ids == {"iono", "dragapult-ex", "mega-abomasnow-ex", "mega-lucario-ex"}


def test_write_and_load_baseline_manifest(tmp_path: Path):
    write_default_baseline_manifest(tmp_path)
    loaded = load_baseline_manifest(tmp_path)
    assert len(loaded) == 4


def test_summarize_baseline_eval():
    reports = {
        "a": {"games": 10.0, "wins": 6.0, "losses": 4.0, "draws": 0.0, "win_rate": 0.6},
        "b": {"games": 10.0, "wins": 8.0, "losses": 2.0, "draws": 0.0, "win_rate": 0.8},
    }
    agg = summarize_baseline_eval(reports)
    assert agg["games"] == 20.0
    assert agg["wins"] == 14.0
    assert agg["win_rate"] == 0.7


def test_choose_pool_deck_round_robin():
    pool = [("a", [1] * 60), ("b", [2] * 60)]
    assert choose_pool_deck(pool, 0)[0] == "a"
    assert choose_pool_deck(pool, 1)[0] == "b"


def test_resolve_baseline_archetype_deck_pool_uses_high_performing(tmp_path: Path):
    hp = tmp_path / "decks" / "competitive" / "high_performing"
    hp.mkdir(parents=True)
    (hp / "2026-05_regional_4th_dragapult.csv").write_text("\n".join(str(i) for i in range(60)), encoding="utf-8")
    (hp / "2026-05_regional_2nd_lopunny-dudunsparce.csv").write_text("\n".join(str(i + 1) for i in range(60)), encoding="utf-8")

    official = tmp_path / "baselines" / "official" / "mega-abomasnow-ex"
    official.mkdir(parents=True)
    (official / "deck.csv").write_text("\n".join(str(i + 2) for i in range(60)), encoding="utf-8")

    manifest = tmp_path / "baselines" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({
            "agents": [
                {"id": "dragapult-ex", "name": "Dragapult", "dir": "dragapult-ex"},
                {"id": "iono", "name": "Iono", "dir": "iono"},
                {"id": "mega-abomasnow-ex", "name": "Abomasnow", "dir": "mega-abomasnow-ex"},
            ]
        }),
        encoding="utf-8",
    )

    from poke_agent.baseline_agents import resolve_baseline_archetype_deck_pool

    pool = resolve_baseline_archetype_deck_pool(tmp_path, top_decks_per_archetype=1)
    names = {name for name, _ in pool}
    assert "2026-05_regional_2nd_lopunny-dudunsparce" in names
    assert "2026-05_regional_4th_dragapult" in names
    assert "official-mega-abomasnow-ex" in names
    assert len(pool) == 3


def test_competitive_deck_archetype_slug():
    from poke_agent.deck_pool import competitive_deck_archetype_slug, deck_matches_archetype_patterns

    stem = "2026-05_regional-melbourne-2026_10th_mega-lucario"
    assert competitive_deck_archetype_slug(stem) == "mega-lucario"
    assert deck_matches_archetype_patterns("dragapult-blaziken", ["dragapult"])
    assert deck_matches_archetype_patterns("lopunny-dudunsparce", ["lopunny-dudunsparce"])


def test_load_kaggle_submission_agent(tmp_path: Path):
    agent_dir = tmp_path / "stub"
    agent_dir.mkdir()
    deck = list(range(60))
    (agent_dir / "deck.csv").write_text("\n".join(str(card) for card in deck), encoding="utf-8")
    (agent_dir / "main.py").write_text(
        """
def agent(obs_dict):
    select = obs_dict.get("select")
    if select is None:
        return list(range(60))
    options = select.get("option") or []
    count = min(int(select.get("maxCount", 1)), len(options))
    return list(range(count))
""",
        encoding="utf-8",
    )

    loaded = load_kaggle_submission_agent(
        agent_dir,
        agent_id="stub",
        name="Stub Agent",
        cg_lib_path=None,
    )
    assert loaded.deck == deck
    action = loaded.act({"select": {"option": [0, 1, 2], "minCount": 1, "maxCount": 1}})
    assert action == [0]
