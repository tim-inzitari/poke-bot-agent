#!/usr/bin/env python3
"""Seal observational outcomes for the exact public plain-Dragapult corpus.

The broad public ``Dragapult ex`` family is not a valid identity boundary for
the future straight-Dragapult specialist.  This audit replays the exact
checksum-bound catalog fingerprints across the same daily archives, verifies
that it reproduces the catalog's hidden match-fact digest, and reports
first/second plus opponent-archetype outcomes.

The resulting evidence is research-only.  It is never training-, gate-, or
serving-eligible and cannot authorize a specialist transition by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import paths
from poke_bot.ladder_replay import (
    LadderReplayClassifier,
    canonical_deck_sha256,
)
from poke_bot.replay_import import episode_id_of, extract_setup_decks


SCHEMA = "poke_bot.plain_dragapult_public_outcome_evidence/v1"
CATALOG_SCHEMA = "poke_bot.public_deck_archetype_catalog/v1"
SPECIALIST_ID = "dragapult"


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


def _first_player(payload: dict[str, Any]) -> int | None:
    for step in payload.get("steps") or ():
        if not isinstance(step, list):
            continue
        for seat_row in step:
            observation = (
                seat_row.get("observation")
                if isinstance(seat_row, dict)
                else None
            )
            current = (
                observation.get("current")
                if isinstance(observation, dict)
                else None
            )
            value = (
                current.get("firstPlayer")
                if isinstance(current, dict)
                else None
            )
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value in (0, 1)
            ):
                return value
    return None


def _outcome(payload: dict[str, Any], seat: int) -> str:
    rewards = payload.get("rewards")
    statuses = payload.get("statuses")
    if (
        not isinstance(rewards, list)
        or len(rewards) != 2
        or not isinstance(statuses, list)
        or len(statuses) != 2
        or any(str(status) != "DONE" for status in statuses)
    ):
        raise RuntimeError("public episode lacks a completed two-seat outcome")
    value = rewards[seat]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("public episode reward is not numeric")
    numeric = float(value)
    if numeric > 0:
        return "win"
    if numeric < 0:
        return "loss"
    return "draw"


@dataclass(frozen=True)
class OutcomeRow:
    date: str
    episode_id: str
    seat: int
    deck_fingerprint: str
    play_order: str
    opponent_archetype: str
    opponent_label_method: str
    outcome: str

    @property
    def catalog_fact(self) -> tuple[str, str, int, str]:
        return (
            self.date,
            self.episode_id,
            self.seat,
            self.deck_fingerprint,
        )


_WORKER_CLASSIFIER: LadderReplayClassifier | None = None
_WORKER_FINGERPRINTS: frozenset[str] = frozenset()


def _init_worker(
    classifier: LadderReplayClassifier,
    fingerprints: Iterable[str],
) -> None:
    global _WORKER_CLASSIFIER, _WORKER_FINGERPRINTS
    _WORKER_CLASSIFIER = classifier
    _WORKER_FINGERPRINTS = frozenset(str(value) for value in fingerprints)


def _scan_archive(task: tuple[str, str]) -> tuple[list[OutcomeRow], dict[str, Any]]:
    day, raw_archive = task
    archive = Path(raw_archive)
    classifier = _WORKER_CLASSIFIER
    if classifier is None or not _WORKER_FINGERPRINTS:
        raise RuntimeError("plain-Dragapult audit worker was not initialized")
    rows: list[OutcomeRow] = []
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if "manifest.csv" not in names:
            raise RuntimeError(f"public replay archive has no manifest: {archive}")
        manifest = list(
            csv.DictReader(
                io.StringIO(source.read("manifest.csv").decode("utf-8-sig"))
            )
        )
        manifest_ids = {str(row.get("episode_id") or "") for row in manifest}
        replay_names = sorted(
            name
            for name in names
            if name.endswith(".json") and not name.endswith("/")
        )
        if "" in manifest_ids or not replay_names:
            raise RuntimeError(f"public replay manifest is invalid: {archive}")
        for replay_name in replay_names:
            payload = json.loads(source.read(replay_name))
            episode_id = episode_id_of(payload, Path(replay_name).stem)
            if (
                episode_id != Path(replay_name).stem
                or episode_id not in manifest_ids
            ):
                raise RuntimeError(
                    "public replay filename/payload/manifest identity mismatch: "
                    f"{archive}:{replay_name}"
                )
            decks = extract_setup_decks(payload)
            if len(decks) != 2:
                raise RuntimeError(f"episode did not expose two seats: {episode_id}")
            first_player = _first_player(payload)
            labels = [classifier.classify_deck(deck) for deck in decks]
            for seat, deck in enumerate(decks):
                if deck is None:
                    continue
                fingerprint = canonical_deck_sha256(deck)
                if fingerprint not in _WORKER_FINGERPRINTS:
                    continue
                opponent = labels[1 - seat]
                rows.append(
                    OutcomeRow(
                        date=day,
                        episode_id=episode_id,
                        seat=seat,
                        deck_fingerprint=fingerprint,
                        play_order=(
                            "unknown"
                            if first_player is None
                            else "first"
                            if seat == first_player
                            else "second"
                        ),
                        opponent_archetype=str(opponent.deck_id),
                        opponent_label_method=str(opponent.method),
                        outcome=_outcome(payload, seat),
                    )
                )
    return rows, {
        "date": day,
        "archive": archive.name,
        "archive_sha256": _sha256(archive),
        "manifest_rows": len(manifest),
        "json_replays": len(replay_names),
        "matched_acting_seats": len(rows),
    }


def _record(counter: Counter[str]) -> dict[str, Any]:
    games = sum(counter.values())
    wins = int(counter["win"])
    draws = int(counter["draw"])
    losses = int(counter["loss"])
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": (wins / games if games else None),
        "non_loss_rate": ((wins + draws) / games if games else None),
    }


def _group(rows: Iterable[OutcomeRow], attribute: str) -> dict[str, Any]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[str(getattr(row, attribute))][row.outcome] += 1
    return {
        key: _record(value)
        for key, value in sorted(grouped.items())
    }


def build_evidence(
    *,
    catalog_path: Path,
    archive_dir: Path,
    mix_path: Path,
    representatives_path: Path,
    card_csv: Path,
    workers: int,
) -> dict[str, Any]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    window = dict(catalog.get("source_window") or {})
    days = sorted((catalog.get("observed_by_day") or {}).keys())
    fingerprints = tuple(str(value) for value in catalog.get("deck_fingerprints") or ())
    if (
        catalog.get("schema") != CATALOG_SCHEMA
        or catalog.get("specialist_id") != SPECIALIST_ID
        or len(days) != int(window.get("days") or 0)
        or not fingerprints
        or int(catalog.get("observed_acting_seat_games") or 0) <= 0
    ):
        raise ValueError("invalid exact plain-Dragapult catalog")
    classifier = LadderReplayClassifier.from_paths(
        mix_path,
        representatives_path,
        card_csv=card_csv,
        additive_registered_ids=(SPECIALIST_ID,),
        authoritative_deck_catalogs=(catalog_path,),
        authoritative_only_ids=(SPECIALIST_ID,),
    )
    tasks = [
        (
            day,
            str(
                archive_dir
                / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
            ),
        )
        for day in days
    ]
    if workers < 1:
        raise ValueError("workers must be positive")
    if any(not Path(path).is_file() for _day, path in tasks):
        missing = [path for _day, path in tasks if not Path(path).is_file()]
        raise FileNotFoundError(missing[0])
    if workers == 1:
        _init_worker(classifier, fingerprints)
        results = [_scan_archive(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(classifier, fingerprints),
        ) as pool:
            results = list(pool.map(_scan_archive, tasks, chunksize=1))
    rows = sorted(
        (row for daily, _evidence in results for row in daily),
        key=lambda row: (row.date, row.episode_id, row.seat),
    )
    archive_evidence = [evidence for _rows, evidence in results]
    observed_counts = Counter(row.date for row in rows)
    observed_by_day = {
        day: int(observed_counts.get(day, 0))
        for day in days
    }
    expected_by_day = {
        str(day): int(value)
        for day, value in dict(catalog["observed_by_day"]).items()
    }
    catalog_fact_digest = _canonical_digest(
        [row.catalog_fact for row in rows]
    )
    if (
        len(rows) != int(catalog["observed_acting_seat_games"])
        or observed_by_day != expected_by_day
        or catalog_fact_digest != catalog.get("source_match_facts_sha256")
    ):
        raise RuntimeError(
            "outcome audit does not reproduce the exact catalog match facts"
        )
    all_outcomes = Counter(row.outcome for row in rows)
    unknown_order = sum(row.play_order == "unknown" for row in rows)
    return {
        "schema": SCHEMA,
        "status": "passed",
        "specialist_id": SPECIALIST_ID,
        "identity": {
            "catalog": str(catalog_path.resolve()),
            "catalog_sha256": _sha256(catalog_path),
            "source_match_facts_sha256": catalog_fact_digest,
            "deck_fingerprint_count": len(fingerprints),
            "acting_seat_games": len(rows),
            "source_window": window,
            "observed_by_day": expected_by_day,
        },
        "classifier": {
            "contract": classifier.contract,
            "contract_sha256": _canonical_digest(classifier.contract),
        },
        "outcomes": {
            "overall": _record(all_outcomes),
            "by_play_order": _group(rows, "play_order"),
            "by_opponent_archetype": _group(rows, "opponent_archetype"),
            "by_opponent_label_method": _group(
                rows,
                "opponent_label_method",
            ),
            "by_date": _group(rows, "date"),
            "unknown_play_order_games": unknown_order,
        },
        "source_archives": archive_evidence,
        "evidence_contract": {
            "observational_only": True,
            "training_eligible": False,
            "replay_eligible": False,
            "formal_gate_eligible": False,
            "serving_authority": "none",
            "broad_dragapult_family_substitution_allowed": False,
            "opponent_unknown_is_preserved_not_guessed": True,
        },
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument(
        "--mix",
        type=Path,
        default=ROOT / "data/training_mixes/top_ladder.v1.json",
    )
    parser.add_argument(
        "--representatives",
        type=Path,
        default=ROOT / "data/training_mixes/top_ladder_representatives.v1.json",
    )
    parser.add_argument("--card-csv", type=Path, default=paths.en_card_data_path())
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_evidence(
        catalog_path=args.catalog.resolve(),
        archive_dir=args.archive_dir.resolve(),
        mix_path=args.mix.resolve(),
        representatives_path=args.representatives.resolve(),
        card_csv=args.card_csv.resolve(),
        workers=args.workers,
    )
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"immutable evidence already exists: {output}")
    _atomic_json(output, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
