#!/usr/bin/env python3
"""Checksum, throttle-copy, and optionally remove Inzi replay archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import zipfile


SCHEMA = "poke_bot.episode_day_archive_transfer/v1"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--remote-host", default="elmo")
    parser.add_argument(
        "--remote-root",
        default="/srv/poke-bot-agent/archive/episode-days",
    )
    parser.add_argument("--bwlimit-kib", type=int, default=6144)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--delete-source-after-verify", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_count(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(
            name.endswith(".json") and not name.endswith("/")
            for name in archive.namelist()
        )


def _atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _remote_digest(host: str, path: str) -> str | None:
    completed = subprocess.run(
        [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            host,
            f"test -f {shlex.quote(path)} && sha256sum -- {shlex.quote(path)}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return None
    return completed.stdout.split()[0]


def _expected_counts(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        return {
            str(row["date"]): int(row["episode_count"])
            for row in rows
            if row.get("date") and row.get("episode_count")
        }


def main() -> int:
    args = _args()
    if args.bwlimit_kib <= 0:
        raise SystemExit("--bwlimit-kib must be positive")
    source_root = args.source_root.expanduser().resolve()
    expected = _expected_counts(args.manifest.expanduser().resolve())
    archives = sorted(source_root.glob("pokemon-tcg-ai-battle-episodes-*.zip"))
    completed_rows: list[dict[str, object]] = []
    started_at = time.time()
    subprocess.run(
        [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            args.remote_host,
            f"mkdir -p -- {shlex.quote(args.remote_root)}",
        ],
        check=True,
    )
    for index, archive in enumerate(archives, 1):
        date = archive.stem.removeprefix("pokemon-tcg-ai-battle-episodes-")
        expected_episodes = expected.get(date)
        if expected_episodes is None:
            raise RuntimeError(f"manifest has no expected count for {date}")
        observed_episodes = _episode_count(archive)
        if observed_episodes != expected_episodes:
            raise RuntimeError(
                f"{archive.name} has {observed_episodes} episodes; "
                f"expected {expected_episodes}"
            )
        local_digest = _sha256(archive)
        remote_path = f"{args.remote_root}/{archive.name}"
        remote_digest = _remote_digest(args.remote_host, remote_path)
        reused = remote_digest == local_digest
        if not reused:
            _atomic(
                args.state,
                {
                    "schema": SCHEMA,
                    "status": "transferring_throttled",
                    "current": index,
                    "total": len(archives),
                    "current_date": date,
                    "bwlimit_kib": args.bwlimit_kib,
                    "completed": completed_rows,
                    "started_at": started_at,
                    "updated_at": time.time(),
                },
            )
            subprocess.run(
                [
                    "/usr/bin/rsync",
                    "-a",
                    "--partial",
                    f"--bwlimit={args.bwlimit_kib}",
                    str(archive),
                    f"{args.remote_host}:{args.remote_root}/",
                ],
                check=True,
            )
            remote_digest = _remote_digest(args.remote_host, remote_path)
        if remote_digest != local_digest:
            raise RuntimeError(
                f"remote checksum mismatch for {archive.name}: "
                f"local={local_digest} remote={remote_digest}"
            )
        bytes_count = archive.stat().st_size
        completed_rows.append(
            {
                "date": date,
                "archive": archive.name,
                "bytes": bytes_count,
                "episodes": observed_episodes,
                "sha256": f"sha256:{local_digest}",
                "reused_existing_remote": reused,
                "source_deleted": bool(args.delete_source_after_verify),
            }
        )
        if args.delete_source_after_verify:
            archive.unlink()

    _atomic(
        args.state,
        {
            "schema": SCHEMA,
            "status": "complete",
            "bwlimit_kib": args.bwlimit_kib,
            "completed": completed_rows,
            "started_at": started_at,
            "completed_at": time.time(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
