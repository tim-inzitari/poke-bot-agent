#!/usr/bin/env python3
"""Re-extract checksum-pinned specialist seats from archived episode zips.

The input corpus is used only as an immutable identity index: episode ID,
acting seat, source day, and submitted deck.  Every selected episode is then
converted again from the original daily archive by the current replay
converter, so current auxiliary and expanded-strategic targets are rebuilt
without rescanning every episode in each multi-gigabyte archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
from typing import Any, Iterator
import zipfile

from poke_bot import archetypes
from poke_bot.authoritative_visual_trace import convert_visual_episode
from poke_bot.feature_shards import SHARD_FORMAT, SHARD_FORMAT_VERSION
from poke_bot.replay_import import (
    episode_id_of,
    extract_setup_decks,
)


RECEIPT_SCHEMA = "poke_bot.verified_specialist_record_extraction/v1"


class _RuleClassifier:
    """Minimal current-rule classifier accepted by the visual-trace validator."""

    def classify_episode(
        self, payload: dict[str, Any]
    ) -> tuple[list[list[int] | None], list[SimpleNamespace]]:
        decks = extract_setup_decks(payload)
        labels = [
            SimpleNamespace(
                deck_id=(
                    archetypes.classify_deck(deck)
                    if deck is not None
                    else archetypes.UNKNOWN
                ),
                method="current_rule_classifier",
            )
            for deck in decks
        ]
        return decks, labels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(raw)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_from_pointer(pointer_path: Path) -> tuple[Path, dict[str, Any]]:
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if (
        pointer.get("schema") != "poke_bot.pinned_expert_corpus/v1"
        or pointer.get("protected") is not True
    ):
        raise RuntimeError("source corpus is not an immutable protected corpus")
    manifest = (pointer_path.parent / str(pointer.get("manifest") or "")).resolve()
    if (
        not manifest.is_file()
        or sha256(manifest) != str(pointer.get("manifest_sha256") or "")
    ):
        raise RuntimeError("source corpus manifest identity changed")
    return manifest, json.loads(manifest.read_text(encoding="utf-8"))


def iter_legacy_identities(
    pointer_path: Path,
    *,
    specialist_id: str,
    forbidden_card_ids: frozenset[int] = frozenset(),
) -> Iterator[dict[str, Any]]:
    """Yield identities from a protected feature stream without schema migration.

    The legacy objects are never supplied to a trainer.  Reading them directly
    is intentional: it preserves the exact historical episode/seat selection
    while the original raw replay is reprocessed into the current schema.
    """

    manifest_path, manifest = _manifest_from_pointer(pointer_path)
    seen: set[tuple[str, int]] = set()
    for row in manifest.get("shards") or ():
        shard = (manifest_path.parent / str(row.get("path") or "")).resolve()
        if not shard.is_file() or sha256(shard) != str(row.get("sha256") or ""):
            raise RuntimeError(f"source feature shard identity changed: {shard}")
        source_dates = tuple(str(value) for value in row.get("source_dates") or ())
        if len(source_dates) != 1:
            raise RuntimeError("identity source requires one calendar day per shard")
        with shard.open("rb") as stream:
            header = pickle.load(stream)
            if (
                not isinstance(header, dict)
                or header.get("format") != SHARD_FORMAT
                or int(header.get("format_version", -1)) != SHARD_FORMAT_VERSION
            ):
                raise RuntimeError(f"invalid source feature shard header: {shard}")
            count = 0
            while True:
                try:
                    item = pickle.load(stream)
                except EOFError as exc:
                    raise RuntimeError(f"source shard lacks a footer: {shard}") from exc
                if (
                    isinstance(item, dict)
                    and item.get("format") == SHARD_FORMAT + "-footer"
                ):
                    expected = int((item.get("stats") or {}).get("records_kept", -1))
                    if expected != count or stream.read(1):
                        raise RuntimeError(f"source shard count/trailing-data error: {shard}")
                    break
                episode_id = str(getattr(item, "episode_id", "") or "")
                seat = int(getattr(item, "seat", -1))
                archetype = str(getattr(item, "archetype", "") or "").casefold()
                deck = tuple(int(value) for value in getattr(item, "deck", ()) or ())
                key = (episode_id, seat)
                if (
                    not episode_id
                    or seat not in (0, 1)
                    or archetype != specialist_id
                    or len(deck) != 60
                    or key in seen
                    or forbidden_card_ids.intersection(deck)
                ):
                    raise RuntimeError(
                        "source specialist identity failed validation: "
                        f"episode={episode_id!r} seat={seat} archetype={archetype!r}"
                    )
                seen.add(key)
                count += 1
                yield {
                    "episode_id": episode_id,
                    "seat": seat,
                    "day": source_dates[0],
                    "deck": list(deck),
                    "source_shard": shard.name,
                }


def _write_day_records(
    *,
    archive: Path,
    day: str,
    identities: list[dict[str, Any]],
    specialist_id: str,
    output: Path,
) -> dict[str, Any]:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    records_written = 0
    decisions_written = 0
    opponent_archetypes: dict[str, int] = {}
    transitions_validated = 0
    exact_target_rows = 0
    classifier = _RuleClassifier()
    try:
        with zipfile.ZipFile(archive) as source, temporary.open(
            "x", encoding="utf-8"
        ) as stream:
            members = set(source.namelist())
            for identity in identities:
                member = f"{identity['episode_id']}.json"
                if member not in members:
                    raise RuntimeError(f"episode is absent from source archive: {member}")
                payload = json.loads(source.read(member))
                if episode_id_of(payload, fallback=identity["episode_id"]) != identity[
                    "episode_id"
                ]:
                    raise RuntimeError(f"episode identity mismatch: {member}")
                decks = extract_setup_decks(payload)
                seat = int(identity["seat"])
                if decks[seat] != identity["deck"]:
                    raise RuntimeError(f"submitted deck changed for {member} seat {seat}")
                converted_result = convert_visual_episode(
                    payload,
                    classifier,
                    source=f"pokemon-tcg-ai-battle-episodes-{day}",
                    required_archetype=specialist_id,
                )
                converted = converted_result.records
                matches = [
                    row
                    for row in converted
                    if int(row.get("seat", -1)) == seat
                    and str(row.get("episode_id") or "") == identity["episode_id"]
                ]
                if len(matches) != 1 or matches[0].get("deck") != identity["deck"]:
                    raise RuntimeError(
                        f"current replay conversion lost verified seat: {member} seat {seat}"
                    )
                record = matches[0]
                stream.write(
                    json.dumps(record, separators=(",", ":"), ensure_ascii=False)
                    + "\n"
                )
                records_written += 1
                decisions_written += int(record.get("n_decisions") or 0)
                transitions_validated += int(
                    converted_result.stats.get("transitions_validated") or 0
                )
                exact_target_rows += int(
                    converted_result.stats.get("exact_target_rows") or 0
                )
                opponent = str(record.get("opp_archetype") or archetypes.UNKNOWN)
                opponent_archetypes[opponent] = opponent_archetypes.get(opponent, 0) + 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "day": day,
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "output": str(output),
        "output_sha256": sha256(output),
        "records": records_written,
        "decisions": decisions_written,
        "transitions_validated": transitions_validated,
        "exact_target_rows": exact_target_rows,
        "opponent_archetypes": dict(sorted(opponent_archetypes.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pointer", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--specialist-id", required=True)
    parser.add_argument("--forbid-card-id", type=int, action="append", default=[])
    parser.add_argument("--expected-records", type=int, required=True)
    args = parser.parse_args()

    specialist_id = str(args.specialist_id).strip().casefold()
    output_dir = args.output_dir.expanduser().resolve()
    receipt_path = output_dir / "VERIFIED_RECORD_EXTRACTION.json"
    if receipt_path.exists():
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != RECEIPT_SCHEMA
            or payload.get("specialist_id") != specialist_id
            or int(payload.get("records") or 0) != int(args.expected_records)
        ):
            raise RuntimeError("existing extraction receipt has a different identity")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    existing = (
        [path for path in output_dir.iterdir() if path.name != "build.log"]
        if output_dir.exists()
        else []
    )
    if existing:
        raise RuntimeError(
            "unsealed extraction output is not empty: "
            + ", ".join(sorted(path.name for path in existing))
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    identities = list(
        iter_legacy_identities(
            args.source_pointer.expanduser().resolve(),
            specialist_id=specialist_id,
            forbidden_card_ids=frozenset(args.forbid_card_id),
        )
    )
    if len(identities) != int(args.expected_records):
        raise RuntimeError(
            f"verified identity count mismatch: {len(identities)} "
            f"!= {int(args.expected_records)}"
        )
    by_day: dict[str, list[dict[str, Any]]] = {}
    for identity in identities:
        by_day.setdefault(str(identity["day"]), []).append(identity)
    days = []
    for day, rows in sorted(by_day.items()):
        archive = (
            args.archive_root.expanduser().resolve()
            / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        )
        if not archive.is_file():
            raise FileNotFoundError(archive)
        days.append(
            _write_day_records(
                archive=archive,
                day=day,
                identities=rows,
                specialist_id=specialist_id,
                output=output_dir / f"verified-{day}.{specialist_id}.jsonl",
            )
        )
    payload = {
        "schema": RECEIPT_SCHEMA,
        "specialist_id": specialist_id,
        "source_pointer": str(args.source_pointer.expanduser().resolve()),
        "source_pointer_sha256": sha256(args.source_pointer.expanduser().resolve()),
        "forbidden_card_ids": sorted(set(args.forbid_card_id)),
        "records": len(identities),
        "decisions": sum(int(row["decisions"]) for row in days),
        "days": days,
        "current_classifier_verified": True,
        "current_replay_conversion_verified": True,
        "authoritative_visual_trace_verified": True,
        "exact_public_post_action_transitions_materialized": True,
        "original_raw_episodes_reprocessed": True,
    }
    atomic_json(receipt_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
