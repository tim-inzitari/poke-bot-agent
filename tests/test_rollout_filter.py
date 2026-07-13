from poke_agent.rollout_filter import episode_involves_archetype


def test_episode_involves_dragapult_family():
    rows = [
        {"step": 0, "deck0": "dragapult-dusknoir", "deck1": "starmie-froslass"},
        {"step": 1, "deck0": "dragapult-dusknoir", "deck1": "starmie-froslass"},
    ]
    assert episode_involves_archetype(rows, "dragapult-ex")
    assert not episode_involves_archetype(rows, "mega-lucario-ex")


def test_episode_matches_either_seat():
    rows = [{"step": 0, "deck0": "starmie-froslass", "deck1": "dragapult-blaziken"}]
    assert episode_involves_archetype(rows, "dragapult-ex")


def test_episode_matches_competitive_deck_stem():
    rows = [
        {
            "step": 0,
            "deck0": "2026-05_regional-melbourne-2026_19th_dragapult-dusknoir",
            "deck1": "2026-05_regional-la-2026_31st_dragapult",
        }
    ]
    assert episode_involves_archetype(rows, "dragapult-ex")


def test_episode_uses_deck_cards_when_slugs_are_player_names():
    rows = [
        {"step": 0, "deck0": "persn", "deck1": "llkarill", "deck0_cards": [1] * 60},
    ]
    # Without real cards this should not match; with dragapult cards it should.
    assert not episode_involves_archetype(rows, "dragapult-ex")
