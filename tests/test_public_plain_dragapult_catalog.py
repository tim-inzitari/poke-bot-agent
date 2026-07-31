from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

from scripts.build_public_plain_dragapult_catalog import (
    build_catalog,
    canonical_deck_sha256,
)
from poke_bot.crustle_heuristics import CANONICAL_DECK_COUNTS
from poke_bot.slowking_heuristics import (
    CANONICAL_DECK_COUNTS as SLOWKING_CANONICAL_DECK_COUNTS,
)


ROOT = Path(__file__).resolve().parents[1]
FULL33_CORPUS_SCRIPT = (
    ROOT / "ops" / "elmo" / "build_plain_dragapult_full33_corpus.sh"
)


PLAIN = (
    [119] * 4
    + [120] * 4
    + [121] * 3
    + [112] * 2
    + [235, 140, 141, 272, 689, 791, 1071, 1071]
    + [1086] * 4
    + [1121] * 4
    + [1152] * 4
    + [1097] * 3
    + [2] * 3
    + [5] * 3
    + [7] * 3
    + [9000] * 15
)
HAMMER = list(PLAIN)
HAMMER[-3:] = [1120, 1120, 1120]
DUSKNOIR = list(PLAIN)
DUSKNOIR[-1] = 131
CRUSTLE = [
    card_id
    for card_id, count in CANONICAL_DECK_COUNTS.items()
    for _ in range(count)
]
SLOWKING = [
    card_id
    for card_id, count in SLOWKING_CANONICAL_DECK_COUNTS.items()
    for _ in range(count)
]


def _episode(episode_id: str, seat0: list[int], seat1: list[int]) -> dict:
    return {
        "id": episode_id,
        "steps": [
            [
                {"action": seat0},
                {"action": seat1},
            ]
        ],
    }


def _archive(path: Path, episodes: dict[str, dict]) -> None:
    manifest = io.StringIO()
    writer = csv.DictWriter(manifest, fieldnames=["episode_id"])
    writer.writeheader()
    for episode_id in episodes:
        writer.writerow({"episode_id": episode_id})
    with zipfile.ZipFile(path, "w") as output:
        output.writestr("manifest.csv", manifest.getvalue())
        for episode_id, payload in episodes.items():
            output.writestr(
                f"{episode_id}.json",
                json.dumps(payload),
            )


def test_catalog_keeps_only_exact_plain_acting_seats(tmp_path: Path) -> None:
    assert len(PLAIN) == 60
    archive = (
        tmp_path / "pokemon-tcg-ai-battle-episodes-2026-07-28.zip"
    )
    _archive(
        archive,
        {
            "1": _episode("1", PLAIN, HAMMER),
            "2": _episode("2", DUSKNOIR, PLAIN),
        },
    )

    catalog = build_catalog(
        archive_dir=tmp_path,
        start="2026-07-28",
        end="2026-07-28",
    )

    assert catalog["specialist_id"] == "dragapult"
    assert catalog["observed_acting_seat_games"] == 2
    assert catalog["observed_by_day"] == {"2026-07-28": 2}
    assert catalog["deck_fingerprints"] == [
        canonical_deck_sha256(PLAIN)
    ]
    assert len(catalog["source_deck_rows"]) == 1
    assert catalog["identity_contract"]["excluded_specialist_ids"] == [
        "hammer-pult",
        "dragapult-blaziken",
        "dragapult-dudunsparce",
        "dragapult-dusknoir",
    ]
    assert catalog["identity_contract"][
        "broad_archetype_name_filter_sufficient"
    ] is False


def test_parallel_day_scan_is_identical_to_sequential_scan(
    tmp_path: Path,
) -> None:
    for day in ("2026-07-27", "2026-07-28"):
        _archive(
            tmp_path / f"pokemon-tcg-ai-battle-episodes-{day}.zip",
            {
                f"{day}-plain": _episode(
                    f"{day}-plain", PLAIN, HAMMER
                ),
            },
        )

    sequential = build_catalog(
        archive_dir=tmp_path,
        start="2026-07-27",
        end="2026-07-28",
        workers=1,
    )
    parallel = build_catalog(
        archive_dir=tmp_path,
        start="2026-07-27",
        end="2026-07-28",
        workers=2,
    )
    sequential.pop("created_at_utc")
    parallel.pop("created_at_utc")
    assert parallel == sequential


def test_catalog_supports_crustle_family_without_changing_representative(
    tmp_path: Path,
) -> None:
    assert len(CRUSTLE) == 60
    mutated = list(CRUSTLE)
    mutated[-1] = 1
    archive = (
        tmp_path / "pokemon-tcg-ai-battle-episodes-2026-07-28.zip"
    )
    _archive(
        archive,
        {
            "1": _episode("1", CRUSTLE, mutated),
        },
    )

    catalog = build_catalog(
        archive_dir=tmp_path,
        start="2026-07-28",
        end="2026-07-28",
        specialist_id="crustle",
    )

    assert catalog["specialist_id"] == "crustle"
    assert catalog["source_archetype"] == {"id": 55, "name": "Crustle"}
    assert catalog["observed_acting_seat_games"] == 2
    assert catalog["deck_fingerprints"] == sorted(
        [
            canonical_deck_sha256(CRUSTLE),
            canonical_deck_sha256(mutated),
        ]
    )
    assert catalog["identity_contract"]["predicate"] == (
        "poke_bot.crustle_heuristics.is_crustle_family_deck"
    )
    assert catalog["identity_contract"]["mode"] == (
        "crustle_card_signature_public_replay_identity"
    )
    assert catalog["identity_contract"]["excluded_specialist_ids"] == []


def test_catalog_keeps_only_owner_exact_slowking_acting_seats(
    tmp_path: Path,
) -> None:
    assert len(SLOWKING) == 60
    generic_slowking = list(SLOWKING)
    generic_slowking[-1] = 1
    archive = (
        tmp_path / "pokemon-tcg-ai-battle-episodes-2026-07-28.zip"
    )
    _archive(
        archive,
        {
            "1": _episode("1", SLOWKING, generic_slowking),
            "2": _episode("2", generic_slowking, SLOWKING),
        },
    )

    catalog = build_catalog(
        archive_dir=tmp_path,
        start="2026-07-28",
        end="2026-07-28",
        specialist_id="slowking",
        auxiliary_fingerprints={
            "other-exact-deck": frozenset(
                {canonical_deck_sha256(generic_slowking)}
            )
        },
    )

    assert catalog["specialist_id"] == "slowking"
    assert catalog["source_archetype"] == {"id": 86, "name": "Slowking"}
    assert catalog["observed_acting_seat_games"] == 2
    assert catalog["deck_fingerprints"] == [
        canonical_deck_sha256(SLOWKING)
    ]
    assert catalog["source_match_facts"] == [
        [
            "2026-07-28",
            "1",
            0,
            canonical_deck_sha256(SLOWKING),
        ],
        [
            "2026-07-28",
            "2",
            1,
            canonical_deck_sha256(SLOWKING),
        ],
    ]
    assert catalog["auxiliary_source_match_indexes"][
        "other-exact-deck"
    ]["source_match_facts"] == [
        [
            "2026-07-28",
            "1",
            1,
            canonical_deck_sha256(generic_slowking),
        ],
        [
            "2026-07-28",
            "2",
            0,
            canonical_deck_sha256(generic_slowking),
        ],
    ]
    assert catalog["identity_contract"]["predicate"] == (
        "poke_bot.slowking_heuristics.is_slowking_deck"
    )
    assert catalog["identity_contract"]["mode"] == (
        "owner_exact_60_card_slowking_public_replay_identity"
    )
    assert catalog["identity_contract"]["excluded_specialist_ids"] == []


def test_full33_seal_allows_nonempty_days_with_all_guide_labels_masked() -> None:
    source = FULL33_CORPUS_SCRIPT.read_text(encoding="utf-8")

    assert "or invalid_guide_days" in source
    assert "guide_by_day[day] < 0" in source
    assert "any(guide_by_day.get(day, 0) <= 0" not in source
