#!/usr/bin/env python3
"""Build checksum-bound replay catalogs for the r274 exact-ID router fit.

PTCGReplay supplies only offline labels: exact numeric archetype IDs joined
through its deck-hash table.  Runtime features are produced later from causal
opponent-public observations in immutable replay archives.  Guide decklists,
display names, hidden zones, actions, and outcomes are never routing inputs or
hard deck requirements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

from poke_bot.ladder_replay import canonical_deck_sha256
from scripts.collect_ptcgreplay_matchup_snapshot_r272 import Client


CATALOG_SCHEMA = "poke_bot.public_deck_archetype_catalog/v1"
RECEIPT_SCHEMA = "poke_bot.ptcgreplay_exact_id_catalog_set_r275/v1"
SOURCE_IDS = (50, 51, 52, 53, 54, 56, 60, 65, 66, 67, 69, 71, 165, 167, 175, 182, 189, 207, 218)
SOURCE_NAMES = {
    50: "Barbaracle",
    51: "Brambleghast",
    52: "Chandelure",
    53: "Cinderace",
    54: "Comfey",
    56: "Cubchoo",
    60: "Dusknoir",
    65: "Mega Froslass ex",
    66: "Mega Gardevoir ex",
    67: "Slop Box",
    69: "Mega Sharpedo ex",
    71: "N’s Zoroark ex",
    165: "Hydrapple ex",
    167: "Teal Mask Ogerpon ex",
    175: "Mega Venusaur ex",
    182: "Arboliva ex",
    189: "Mega Lopunny ex",
    207: "Ethan's Typhlosion",
    218: "Galvantula",
}


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def semantic_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def compact_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def write_create_only(path: Path, payload: dict) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"catalog target already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start", default="2026-07-10")
    parser.add_argument("--end", default="2026-08-10")
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise RuntimeError("output root must be create-only")
    args.output_root.mkdir(parents=True, mode=0o755)

    client = Client()
    decks = client.all("decks?select=deck_hash,archetype_id,card_ids")
    matches = client.all(
        "matches?select=episode_id,played_on,deck0_hash,deck1_hash"
        f"&played_on=gte.{args.start}&played_on=lte.{args.end}&order=episode_id"
    )
    target_ids = set(SOURCE_IDS)
    deck_rows: dict[str, tuple[int, str]] = {}
    for row in decks:
        source_id = row.get("archetype_id")
        cards = row.get("card_ids")
        if source_id is None or int(source_id) not in target_ids or not isinstance(cards, list):
            continue
        if len(cards) != 60:
            raise RuntimeError("PTCGReplay exact-ID deck is not 60 cards")
        deck_rows[str(row["deck_hash"])] = (
            int(source_id), canonical_deck_sha256([int(value) for value in cards])
        )

    archive_evidence: dict[str, dict] = {}
    for day in sorted({str(row["played_on"]) for row in matches}):
        archive = args.archive_root / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        if not archive.is_file() or archive.is_symlink():
            raise RuntimeError(f"immutable replay archive missing: {archive}")
        with zipfile.ZipFile(archive, "r") as stream:
            members = [row for row in stream.infolist() if not row.is_dir() and row.filename.endswith(".json")]
        archive_evidence[day] = {
            "date": day,
            "archive": archive.name,
            "archive_sha256": file_digest(archive),
            "manifest_rows": len(members),
            "json_replays": len(members),
        }

    facts: dict[int, list[list]] = {source_id: [] for source_id in SOURCE_IDS}
    fingerprints: dict[int, set[str]] = {source_id: set() for source_id in SOURCE_IDS}
    observed: dict[int, dict[str, int]] = {
        source_id: {day: 0 for day in archive_evidence} for source_id in SOURCE_IDS
    }
    for row in matches:
        day = str(row["played_on"])
        episode_id = str(row["episode_id"])
        for seat, key in ((0, "deck0_hash"), (1, "deck1_hash")):
            mapped = deck_rows.get(str(row.get(key) or ""))
            if mapped is None:
                continue
            source_id, fingerprint = mapped
            facts[source_id].append([day, episode_id, seat, fingerprint])
            fingerprints[source_id].add(fingerprint)
            observed[source_id][day] += 1

    outputs: dict[str, dict] = {}
    for source_id in SOURCE_IDS:
        target_id = f"ptcgreplay-source-id-{source_id}"
        rows = sorted(facts[source_id], key=lambda row: (row[0], row[1], row[2], row[3]))
        if not rows:
            raise RuntimeError(f"exact source ID {source_id} has no checksum-backed replay support")
        payload = {
            "schema": CATALOG_SCHEMA,
            "specialist_id": target_id,
            "source": "https://ptcgreplay.netlify.app/",
            "source_access": "authenticated_read_only",
            "source_namespace": "ptcgreplay",
            "source_id": source_id,
            "source_archetype": {
                "id": source_id,
                "name": SOURCE_NAMES[source_id],
            },
            "source_window": {
                "start": args.start,
                "end": args.end,
                "days": len(archive_evidence),
            },
            "observed_by_day": observed[source_id],
            "observed_acting_seat_games": len(rows),
            "deck_fingerprints": sorted(fingerprints[source_id]),
            "source_match_rows": len(rows),
            "source_match_facts": rows,
            "source_match_facts_sha256": compact_digest(rows),
            "source_archives": [archive_evidence[day] for day in sorted(archive_evidence)],
            "label_provenance": "exact_numeric_ptcgreplay_source_id_joined_by_deck_hash",
            "runtime_feature_authority": "causal_opponent_public_observations_only",
            "guide_decklist_hard_requirement": False,
        }
        path = args.output_root / f"{target_id}.json"
        write_create_only(path, payload)
        outputs[target_id] = {
            "path": str(path),
            "sha256": file_digest(path),
            "size_bytes": path.stat().st_size,
            "source_id": source_id,
            "acting_seat_games": len(rows),
        }

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "source_window": {"start": args.start, "end": args.end, "days": len(archive_evidence)},
        "catalogs": outputs,
        "exact_source_ids": list(SOURCE_IDS),
        "names_used_as_identity_or_alias_authority": False,
        "guide_decklists_used_as_hard_requirements": False,
        "credentials_included": False,
    }
    receipt["receipt_sha256"] = semantic_digest(receipt)
    write_create_only(args.output_root / "catalog-set-receipt.json", receipt)
    os.chmod(args.output_root, 0o555)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
