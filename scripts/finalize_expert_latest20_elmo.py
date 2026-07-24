#!/usr/bin/env python3
"""Validate and atomically publish Elmo's canonical latest-20 replay window."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA = "poke_bot.expert_latest20_receipt/v1"
ARCHIVE_PREFIX = "pokemon-tcg-ai-battle-episodes-"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _window(index_path: Path) -> list[dict[str, Any]]:
    with index_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 20:
        raise RuntimeError("Kaggle index contains fewer than 20 days")
    selected = rows[-20:]
    parsed = [date.fromisoformat(str(row["date"])) for row in selected]
    expected = [
        parsed[0] + timedelta(days=offset) for offset in range(20)
    ]
    if parsed != expected:
        raise RuntimeError("latest 20 Kaggle rows are not consecutive calendar days")
    result: list[dict[str, Any]] = []
    for row in selected:
        episodes = int(row["episode_count"])
        if episodes <= 0:
            raise RuntimeError(f"invalid episode count for {row['date']}")
        result.append(
            {
                "date": str(row["date"]),
                "dataset_slug": str(row["daily_dataset_slug"]),
                "episode_count": episodes,
            }
        )
    return result


def _archive_count(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        # Opening the central directory validates the ZIP structure. Do not
        # call testzip(): these archives expand to tens of GiB per day and a
        # metadata refresh must not decompress the replay corpus.
        return sum(
            name.endswith(".json") and not name.endswith("/")
            for name in archive.namelist()
        )


def _validate(path: Path, expected: int) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing replay archive: {path}")
    actual = _archive_count(path)
    if actual != expected:
        raise RuntimeError(
            f"episode-count mismatch for {path.name}: "
            f"actual={actual} expected={expected}"
        )


def _archive_path(archive_root: Path, day: str) -> Path:
    return archive_root / f"{ARCHIVE_PREFIX}{day}.zip"


def _reuse_archive(
    destination: Path,
    *,
    expected: int,
    reuse_roots: list[Path],
) -> bool:
    if destination.is_file():
        _validate(destination, expected)
        return True
    for root in reuse_roots:
        candidate = root / destination.name
        if not candidate.is_file():
            continue
        _validate(candidate, expected)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.reuse")
        temporary.unlink(missing_ok=True)
        try:
            os.link(candidate, temporary)
        except OSError:
            shutil.copyfile(candidate, temporary)
        os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
        return True
    return False


def prepare(
    index_path: Path,
    *,
    archive_root: Path,
    reuse_roots: list[Path],
) -> dict[str, Any]:
    rows = _window(index_path)
    missing: list[dict[str, Any]] = []
    for row in rows:
        destination = _archive_path(archive_root, row["date"])
        if not _reuse_archive(
            destination,
            expected=int(row["episode_count"]),
            reuse_roots=reuse_roots,
        ):
            missing.append(row)
    return {
        "schema": "poke_bot.expert_latest20_prepare/v1",
        "window_start": rows[0]["date"],
        "window_end": rows[-1]["date"],
        "days": 20,
        "missing": missing,
    }


def install(
    index_path: Path,
    *,
    archive_root: Path,
    incoming: Path,
    day: str,
) -> dict[str, Any]:
    rows = {row["date"]: row for row in _window(index_path)}
    if day not in rows:
        raise RuntimeError(f"{day} is not in the current latest-20 window")
    row = rows[day]
    _validate(incoming, int(row["episode_count"]))
    destination = _archive_path(archive_root, day)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.install")
    temporary.unlink(missing_ok=True)
    shutil.copyfile(incoming, temporary)
    os.chmod(temporary, 0o444)
    os.replace(temporary, destination)
    return {
        "date": day,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _prior_days(receipt: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if payload.get("schema") != SCHEMA:
        return {}
    return {
        str(row.get("date")): dict(row)
        for row in payload.get("archives") or ()
        if row.get("validated") is True
    }


def commit(
    index_path: Path,
    *,
    archive_root: Path,
    receipt_root: Path,
) -> dict[str, Any]:
    rows = _window(index_path)
    current = receipt_root / "current.json"
    prior = _prior_days(current)
    archives: list[dict[str, Any]] = []
    for row in rows:
        day = row["date"]
        path = _archive_path(archive_root, day)
        _validate(path, int(row["episode_count"]))
        stat = path.stat()
        prior_row = prior.get(day, {})
        checksum = None
        if (
            prior_row.get("path") == str(path)
            and int(prior_row.get("bytes") or -1) == stat.st_size
            and str(prior_row.get("sha256") or "").startswith("sha256:")
        ):
            checksum = str(prior_row["sha256"])
        if checksum is None:
            checksum = _sha256(path)
        archives.append(
            {
                **row,
                "path": str(path),
                "bytes": stat.st_size,
                "sha256": checksum,
                "validated": True,
            }
        )
    now = datetime.now(timezone.utc).isoformat()
    receipt = {
        "schema": SCHEMA,
        "status": "ready",
        "window_policy": "latest_20_consecutive_calendar_days",
        "window_start": rows[0]["date"],
        "window_end": rows[-1]["date"],
        "days": 20,
        "all_dates_represented": True,
        "filter_applied_after_window_selection": True,
        "ingress": {
            "kaggle_host": "bert",
            "kaggle_interface": "wifi",
            "kaggle_interface_device": "en1",
            "archive_host": "elmo",
            "transfer_interface": "bert_ethernet",
            "transfer_source_address": "192.168.1.158",
        },
        "archives": archives,
        "total_episodes": sum(int(row["episode_count"]) for row in rows),
        "total_bytes": sum(int(row["bytes"]) for row in archives),
        "committed_at": now,
    }
    versioned = (
        receipt_root
        / "windows"
        / f"{rows[0]['date']}_{rows[-1]['date']}.json"
    )
    _atomic_json(versioned, receipt)
    receipt["versioned_receipt"] = str(versioned)
    _atomic_json(current, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "install", "commit"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path, action="append", default=[])
    parser.add_argument("--incoming", type=Path)
    parser.add_argument("--day")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(
            args.index,
            archive_root=args.archive_root,
            reuse_roots=args.reuse_root,
        )
    elif args.action == "install":
        if args.incoming is None or not args.day:
            parser.error("install requires --incoming and --day")
        result = install(
            args.index,
            archive_root=args.archive_root,
            incoming=args.incoming,
            day=args.day,
        )
    else:
        result = commit(
            args.index,
            archive_root=args.archive_root,
            receipt_root=args.receipt_root,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
