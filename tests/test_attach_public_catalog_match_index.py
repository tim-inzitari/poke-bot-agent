from __future__ import annotations

import json
from pathlib import Path

from scripts.attach_public_catalog_match_index import _digest, attach_index


def test_attach_index_filters_to_base_window_and_reproduces_counts(
    tmp_path: Path,
) -> None:
    fingerprint = "sha256:" + ("a" * 64)
    base = {
        "schema": "poke_bot.public_deck_archetype_catalog/v1",
        "specialist_id": "teal-mask-ogerpon-ex",
        "source_window": {
            "start": "2026-07-27",
            "end": "2026-07-28",
            "days": 2,
        },
        "observed_by_day": {"2026-07-27": 1, "2026-07-28": 1},
        "observed_acting_seat_games": 2,
        "source_match_rows": 2,
        "deck_fingerprints": [fingerprint],
    }
    facts = [
        ["2026-07-27", "one", 0, fingerprint],
        ["2026-07-28", "two", 1, fingerprint],
        ["2026-07-29", "outside", 0, fingerprint],
    ]
    index = {
        "schema": "poke_bot.public_deck_archetype_catalog/v1",
        "specialist_id": "slowking",
        "source_archives": [
            {
                "date": day,
                "archive": f"pokemon-tcg-ai-battle-episodes-{day}.zip",
                "archive_sha256": "sha256:" + character * 64,
                "manifest_rows": 1,
                "json_replays": 1,
            }
            for day, character in (
                ("2026-07-27", "b"),
                ("2026-07-28", "c"),
                ("2026-07-29", "d"),
            )
        ],
        "auxiliary_source_match_indexes": {
            "teal-mask-ogerpon-ex": {
                "deck_fingerprints": [fingerprint],
                "source_match_rows": 3,
                "source_match_facts": facts,
                "source_match_facts_sha256": _digest(facts),
            }
        },
    }
    base_path = tmp_path / "base.json"
    index_path = tmp_path / "index.json"
    base_path.write_text(json.dumps(base) + "\n", encoding="utf-8")
    index_path.write_text(json.dumps(index) + "\n", encoding="utf-8")

    revised = attach_index(
        base_catalog_path=base_path,
        index_catalog_path=index_path,
        target_id="teal-mask-ogerpon-ex",
    )

    assert revised["source_match_facts"] == facts[:2]
    assert revised["source_match_facts_sha256"] == _digest(facts[:2])
    assert [row["date"] for row in revised["source_archives"]] == [
        "2026-07-27",
        "2026-07-28",
    ]
    assert revised["source_archives_sha256"] == _digest(
        revised["source_archives"]
    )
    assert revised["match_index_attachment"]["rows"] == 2
