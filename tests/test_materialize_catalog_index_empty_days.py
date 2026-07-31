from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.materialize_catalog_index_empty_days import materialize


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_materializes_only_checksum_indexed_zero_days(tmp_path: Path) -> None:
    output = tmp_path / "archive" / "corpus"
    output.mkdir(parents=True)
    catalog = output / "PUBLIC_DECK_ARCHETYPE_CATALOG.json"
    fingerprint = "sha256:" + "a" * 64
    catalog.write_text(
        json.dumps(
            {
                "schema": "poke_bot.public_deck_archetype_catalog/v1",
                "specialist_id": "slowking",
                "observed_by_day": {
                    "2026-07-06": 0,
                    "2026-07-07": 1,
                    "2026-07-08": 0,
                },
                "source_match_facts": [
                    ["2026-07-07", "episode", 0, fingerprint]
                ],
                "source_archives": [
                    {
                        "date": day,
                        "archive": f"episodes-{day}.zip",
                        "archive_sha256": "sha256:" + character * 64,
                        "json_replays": 1,
                    }
                    for day, character in (
                        ("2026-07-06", "b"),
                        ("2026-07-07", "c"),
                        ("2026-07-08", "d"),
                    )
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    template = output / "slowking-2026-07-06.features"
    template.write_bytes(b"empty-feature")
    template_sidecar = template.with_name(template.name + ".json")
    template_sidecar.write_text(
        json.dumps(
            {
                "format": "pokebot-bootstrap-feature-shard",
                "format_version": 1,
                "compact_mode": "temporal-expert-v1",
                "required_archetype": "slowking",
                "guide_id": "slowking",
                "guide_version": "slowking-north-star-v1",
                "max_context": 320,
                "path": template.name,
                "sha256": _digest(template),
                "source_dates": ["2026-07-06"],
                "stats": {
                    "records_total": 0,
                    "records_kept": 0,
                    "decisions_kept": 0,
                    "target_coverage": {},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    template.with_name(template.name + ".receipt.json").write_text(
        json.dumps(
            {
                "format": "pokebot-authoritative-visual-day-receipt",
                "format_version": 1,
                "source_date": "2026-07-06",
                "selection": {
                    "acting_seat_archetype": "slowking",
                    "current_deck_guide": "slowking",
                },
                "schemas": {
                    "guide": "slowking-north-star-v1",
                    "max_context": 320,
                },
                "stats": {
                    "records_total": 0,
                    "records_kept": 0,
                    "decisions_kept": 0,
                    "target_coverage": {},
                },
                "output": {"sha256": _digest(template)},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    created = materialize(
        catalog_path=catalog,
        output_root=output,
        template_day="2026-07-06",
    )

    assert created == ["2026-07-08"]
    assert not (output / "slowking-2026-07-07.features").exists()
    receipt = json.loads(
        (output / "slowking-2026-07-08.features.receipt.json").read_text()
    )
    assert receipt["zero_match_materialization"]["mode"] == (
        "checksum_bound_catalog_zero_match"
    )
    assert receipt["stats"]["records_kept"] == 0
    assert materialize(
        catalog_path=catalog,
        output_root=output,
        template_day="2026-07-06",
    ) == []
