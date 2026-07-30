#!/usr/bin/env python3
"""Build a checksum-bound exact-deck catalog from public replay archives.

The broad public ``Dragapult`` label combines several materially different
decks.  This scanner reads each acting seat's submitted 60-card list and keeps
only lists accepted by the canonical straight-Dragapult identity predicate.
Hammer-Pult, Dragapult/Blaziken, Dragapult/Dudunsparce, and
Dragapult/Dusknoir are therefore excluded before feature materialization.
The same scanner also supports the checksum-bound current Crustle
representative without widening either identity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta, timezone, datetime
from pathlib import Path
from typing import Any, Iterable

from poke_bot.crustle_heuristics import is_crustle_family_deck
from poke_bot.dragapult_heuristics import is_plain_dragapult_deck
from poke_bot.replay_import import episode_id_of, extract_setup_decks


SCHEMA = "poke_bot.public_deck_archetype_catalog/v1"
SPECIALIST_ID = "dragapult"
SOURCE_ARCHETYPE_ID = 58
SOURCE_ARCHETYPE_NAME = "Dragapult ex"
SOURCE_URL = (
    "https://www.kaggle.com/datasets/"
    "pokemon-tcg-ai-battle/pokemon-tcg-ai-battle-episodes"
)
EXCLUDED_SPECIALIST_IDS = (
    "hammer-pult",
    "dragapult-blaziken",
    "dragapult-dudunsparce",
    "dragapult-dusknoir",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def canonical_deck_sha256(card_ids: Iterable[int]) -> str:
    """Return the established sorted-60-card multiset digest."""
    return _canonical_digest(tuple(sorted(int(card_id) for card_id in card_ids)))


def _dates(start: str, end: str) -> list[str]:
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    if last < first:
        raise ValueError("--end must not precede --start")
    return [
        (first + timedelta(days=offset)).isoformat()
        for offset in range((last - first).days + 1)
    ]


def _archive_rows(
    archive_and_specialist: tuple[Path, str],
) -> tuple[list[tuple[str, int, tuple[int, ...]]], dict[str, Any]]:
    """Return exact specialist seats plus manifest/checksum evidence for one day."""
    archive, specialist_id = archive_and_specialist
    predicates = {
        "dragapult": is_plain_dragapult_deck,
        "crustle": is_crustle_family_deck,
    }
    try:
        predicate = predicates[specialist_id]
    except KeyError as exc:
        raise ValueError(f"unsupported exact-deck specialist: {specialist_id}") from exc
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if "manifest.csv" not in names:
            raise RuntimeError(f"public replay archive has no manifest: {archive}")
        manifest_rows = list(
            csv.DictReader(
                io.StringIO(source.read("manifest.csv").decode("utf-8-sig"))
            )
        )
        manifest_ids = [str(row.get("episode_id") or "") for row in manifest_rows]
        replay_names = sorted(
            name
            for name in names
            if name.endswith(".json") and not name.endswith("/")
        )
        replay_ids = [Path(name).stem for name in replay_names]
        if (
            not replay_names
            or not manifest_ids
            or "" in manifest_ids
            or len(manifest_ids) != len(set(manifest_ids))
            or len(replay_ids) != len(set(replay_ids))
            or set(replay_ids) - set(manifest_ids)
        ):
            raise RuntimeError(
                f"public replay archive has invalid manifest identity: {archive}"
            )

        matched: list[tuple[str, int, tuple[int, ...]]] = []
        for replay_name in replay_names:
            payload = json.loads(source.read(replay_name))
            episode_id = episode_id_of(payload, Path(replay_name).stem)
            if episode_id != Path(replay_name).stem:
                raise RuntimeError(
                    "episode filename and payload identity disagree: "
                    f"archive={archive} entry={replay_name}"
                )
            decks = extract_setup_decks(payload)
            if len(decks) != 2:
                raise RuntimeError(f"episode did not expose two seats: {episode_id}")
            for seat, deck in enumerate(decks):
                if deck is None:
                    continue
                if len(deck) != 60 or any(
                    isinstance(card_id, bool) or not isinstance(card_id, int)
                    for card_id in deck
                ):
                    raise RuntimeError(
                        f"invalid submitted deck: episode={episode_id} seat={seat}"
                    )
                if predicate(deck):
                    matched.append(
                        (
                            episode_id,
                            seat,
                            tuple(sorted(int(card_id) for card_id in deck)),
                        )
                    )

    return matched, {
        "archive": archive.name,
        "archive_sha256": _sha256(archive),
        "manifest_rows": len(manifest_ids),
        "json_replays": len(replay_ids),
        "manifest_only_episode_ids": sorted(set(manifest_ids) - set(replay_ids)),
    }


def build_catalog(
    *,
    archive_dir: Path,
    start: str,
    end: str,
    minimum_records: int = 1,
    workers: int = 1,
    specialist_id: str = SPECIALIST_ID,
) -> dict[str, Any]:
    days = _dates(start, end)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    observed_by_day: dict[str, int] = {}
    unique_decks: dict[str, tuple[int, ...]] = {}
    match_facts: list[tuple[str, str, int, str]] = []
    source_archives: list[dict[str, Any]] = []

    archives = [
        (
            archive_dir
            / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        )
        for day in days
    ]
    for archive in archives:
        if not archive.is_file() or archive.stat().st_size <= 0:
            raise FileNotFoundError(archive)

    if workers == 1:
        daily_results = [
            _archive_rows((archive, specialist_id)) for archive in archives
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            daily_results = list(
                executor.map(
                    _archive_rows,
                    [(archive, specialist_id) for archive in archives],
                    chunksize=1,
                )
            )

    for day, (rows, archive_evidence) in zip(
        days, daily_results, strict=True
    ):
        observed_by_day[day] = len(rows)
        archive_evidence["date"] = day
        source_archives.append(archive_evidence)
        for episode_id, seat, cards in rows:
            fingerprint = canonical_deck_sha256(cards)
            prior = unique_decks.setdefault(fingerprint, cards)
            if prior != cards:
                raise RuntimeError(
                    f"deck fingerprint collision detected: {fingerprint}"
                )
            match_facts.append((day, episode_id, seat, fingerprint))

    observed = sum(observed_by_day.values())
    if minimum_records < 1 or observed < minimum_records:
        raise RuntimeError(
            f"{specialist_id} public corpus is below its record floor: "
            f"actual={observed} required={minimum_records}"
        )
    if not unique_decks:
        raise RuntimeError(
            f"no exact {specialist_id} deck identities were found"
        )

    specialist_contracts = {
        "dragapult": {
            "source_archetype_id": SOURCE_ARCHETYPE_ID,
            "source_archetype_name": SOURCE_ARCHETYPE_NAME,
            "predicate": (
                "poke_bot.dragapult_heuristics.is_plain_dragapult_deck"
            ),
            "excluded_specialist_ids": list(EXCLUDED_SPECIALIST_IDS),
        },
        "crustle": {
            "source_archetype_id": 55,
            "source_archetype_name": "Crustle",
            "predicate": (
                "poke_bot.crustle_heuristics.is_crustle_family_deck"
            ),
            "identity_mode": (
                "crustle_card_signature_public_replay_identity"
            ),
            "excluded_specialist_ids": [],
        },
    }
    try:
        specialist_contract = specialist_contracts[specialist_id]
    except KeyError as exc:
        raise ValueError(f"unsupported exact-deck specialist: {specialist_id}") from exc

    source_deck_rows = [
        {
            "deck_hash": fingerprint.removeprefix("sha256:")[:16],
            "archetype_id": specialist_contract["source_archetype_id"],
            "card_ids": list(cards),
        }
        for fingerprint, cards in sorted(unique_decks.items())
    ]
    fingerprints = [
        canonical_deck_sha256(row["card_ids"])
        for row in source_deck_rows
    ]
    if fingerprints != sorted(unique_decks):
        raise RuntimeError("source deck rows do not reproduce catalog fingerprints")
    if any(
        not {
            "dragapult": is_plain_dragapult_deck,
            "crustle": is_crustle_family_deck,
        }[specialist_id](row["card_ids"])
        for row in source_deck_rows
    ):
        raise RuntimeError(
            f"catalog contains a non-{specialist_id} deck"
        )

    return {
        "schema": SCHEMA,
        "specialist_id": specialist_id,
        "source": SOURCE_URL,
        "source_access": "public_checksum_bound_daily_replay_archives",
        "source_archetype": {
            "id": specialist_contract["source_archetype_id"],
            "name": specialist_contract["source_archetype_name"],
        },
        "source_window": {
            "start": days[0],
            "end": days[-1],
            "days": len(days),
        },
        "minimum_acting_seat_games": int(minimum_records),
        "observed_acting_seat_games": observed,
        "observed_by_day": observed_by_day,
        "deck_fingerprints": fingerprints,
        "source_deck_rows": source_deck_rows,
        "source_deck_rows_sha256": _canonical_digest(source_deck_rows),
        "source_match_rows": observed,
        "source_match_facts_sha256": _canonical_digest(match_facts),
        "source_archives_sha256": _canonical_digest(source_archives),
        "source_archives": source_archives,
        "identity_contract": {
            "mode": specialist_contract.get(
                "identity_mode",
                "exact_60_card_public_replay_identity",
            ),
            "predicate": specialist_contract["predicate"],
            "required_card_count": 60,
            "excluded_specialist_ids": specialist_contract[
                "excluded_specialist_ids"
            ],
            "broad_archetype_name_filter_sufficient": False,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _comparable(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key != "created_at_utc"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-records", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--specialist-id",
        choices=("dragapult", "crustle"),
        default=SPECIALIST_ID,
    )
    args = parser.parse_args()

    payload = build_catalog(
        archive_dir=args.archive_dir.resolve(),
        start=args.start,
        end=args.end,
        minimum_records=args.minimum_records,
        workers=args.workers,
        specialist_id=args.specialist_id,
    )
    output = args.out.resolve()
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if _comparable(existing) != _comparable(payload):
            raise RuntimeError(f"immutable catalog differs: {output}")
    else:
        _atomic_json(output, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
