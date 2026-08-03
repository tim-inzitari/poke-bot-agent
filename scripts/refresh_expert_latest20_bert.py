#!/usr/bin/env python3
"""Refresh latest-20 expert archives through Bert Wi-Fi and Elmo Ethernet."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

SCHEMA = "poke_bot.expert_latest20_refresh/v1"
ARCHIVE_PREFIX = "pokemon-tcg-ai-battle-episodes-"
KNOWN_SOURCE_DISCREPANCIES = {
    "2026-07-24": {
        "index_episode_count": 4_445,
        "validated_episode_count": 4_444,
        "archive_sha256": (
            "sha256:68a5c1be539bef579f03b5de29b901a1fab1dc4904af78824fbf7666d73bc8ab"
        ),
    },
    "2026-08-02": {
        "index_episode_count": 4_587,
        "validated_episode_count": 4_586,
        "archive_sha256": (
            "sha256:fa91e058a42d5fffab0f3e63f04fba5acc9bfbd2e2225e97aa62f45f5d430eb8"
        ),
    },
}


def _run(
    command: list[str],
    *,
    capture: bool = False,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        timeout=timeout,
    )


def _default_interface() -> str:
    output = _run(
        ["/sbin/route", "-n", "get", "default"], capture=True, timeout=10
    ).stdout
    for line in output.splitlines():
        key, separator, value = line.strip().partition(":")
        if separator and key == "interface":
            return value.strip()
    raise RuntimeError("unable to resolve Bert's default network interface")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _window(index_path: Path) -> list[dict[str, Any]]:
    with index_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))[-20:]
    if len(rows) != 20:
        raise RuntimeError("Kaggle index contains fewer than 20 days")
    parsed = [date.fromisoformat(str(row["date"])) for row in rows]
    if parsed != [parsed[0] + timedelta(days=i) for i in range(20)]:
        raise RuntimeError("latest 20 Kaggle rows are not consecutive calendar days")
    return [
        {
            "date": str(row["date"]),
            "dataset_slug": str(row["daily_dataset_slug"]),
            "episode_count": int(row["episode_count"]),
        }
        for row in rows
    ]


def _validate(path: Path, expected: int, *, day: str) -> str:
    with zipfile.ZipFile(path) as archive:
        # Parsing the central directory validates the ZIP structure. Full CRC
        # verification would decompress tens of GiB and defeat the incremental
        # refresh; transfer integrity is instead enforced with SHA-256.
        actual = sum(
            name.endswith(".json") and not name.endswith("/")
            for name in archive.namelist()
        )
    checksum = _sha256(path)
    exception = KNOWN_SOURCE_DISCREPANCIES.get(day)
    if actual != expected and not (
        exception
        and expected == int(exception["index_episode_count"])
        and actual == int(exception["validated_episode_count"])
        and checksum == exception["archive_sha256"]
    ):
        raise RuntimeError(
            f"episode-count mismatch: actual={actual} expected={expected}"
        )
    return checksum


def _ssh_prefix(host: str, source_address: str) -> list[str]:
    return [
        "ssh",
        "-b",
        source_address,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        host,
    ]


def _remote_json(
    host: str,
    source_address: str,
    command: list[str],
) -> dict[str, Any]:
    result = _run(
        _ssh_prefix(host, source_address) + command,
        capture=True,
        timeout=60 * 60,
    )
    return json.loads(result.stdout)


def _download_index(root: Path) -> Path:
    staging = root / "index-staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    _run(
        [
            sys.executable,
            "-m",
            "kaggle.cli",
            "datasets",
            "download",
            "kaggle/pokemon-tcg-ai-battle-episodes-index",
            "-p",
            str(staging),
            "--unzip",
            "-o",
        ],
        timeout=10 * 60,
    )
    index = staging / "manifest.csv"
    _window(index)
    return index


def _download_day(cache: Path, row: dict[str, Any]) -> tuple[Path, str]:
    day = str(row["date"])
    slug = str(row["dataset_slug"])
    destination = cache / f"{ARCHIVE_PREFIX}{day}.zip"
    if destination.is_file():
        try:
            return destination, _validate(
                destination,
                int(row["episode_count"]),
                day=day,
            )
        except (OSError, RuntimeError, zipfile.BadZipFile):
            destination.unlink(missing_ok=True)
    staging = cache / ".download" / day
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    _run(
        [
            sys.executable,
            "-m",
            "kaggle.cli",
            "datasets",
            "download",
            f"kaggle/{slug}",
            "-p",
            str(staging),
            "-o",
        ],
        timeout=4 * 60 * 60,
    )
    candidate = staging / f"{slug}.zip"
    checksum = _validate(
        candidate,
        int(row["episode_count"]),
        day=day,
    )
    cache.mkdir(parents=True, exist_ok=True)
    os.replace(candidate, destination)
    shutil.rmtree(staging)
    return destination, checksum


def refresh(args: argparse.Namespace) -> dict[str, Any]:
    if _default_interface() != args.wifi_interface:
        raise RuntimeError(
            f"Kaggle ingress requires {args.wifi_interface} as Bert's "
            "default route"
        )
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "refresh.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("expert latest-20 refresh is already running") from exc

        index = _download_index(root)
        rows = _window(index)
        remote_stage = f"{args.remote_stage}/manifest.csv"
        _run(
            _ssh_prefix(args.elmo, args.ethernet_source)
            + ["mkdir", "-p", args.remote_stage],
            timeout=60,
        )
        _run(
            [
                "rsync",
                "-a",
                "-e",
                (
                    f"ssh -b {args.ethernet_source} -o BatchMode=yes "
                    "-o ConnectTimeout=10"
                ),
                str(index),
                f"{args.elmo}:{remote_stage}",
            ],
            timeout=10 * 60,
        )
        finalizer = [
            "sudo",
            "-n",
            "/usr/bin/nice",
            "-n",
            "15",
            "/usr/bin/ionice",
            "-c",
            "3",
            "python3",
            args.elmo_finalizer,
        ]
        common = [
            "--index",
            remote_stage,
            "--archive-root",
            args.elmo_archive_root,
            "--receipt-root",
            args.elmo_receipt_root,
        ]
        prepare = _remote_json(
            args.elmo,
            args.ethernet_source,
            finalizer
            + ["prepare"]
            + common
            + sum(
                (["--reuse-root", value] for value in args.elmo_reuse_root),
                [],
            ),
        )
        transferred: list[dict[str, Any]] = []
        cache = root / "downloads"
        for row in prepare.get("missing") or ():
            archive, checksum = _download_day(cache, row)
            incoming = f"{args.remote_stage}/{archive.name}.incoming"
            _run(
                [
                    "rsync",
                    "-a",
                    "--partial",
                    "-e",
                    (
                        f"ssh -b {args.ethernet_source} -o BatchMode=yes "
                        "-o ConnectTimeout=10"
                    ),
                    str(archive),
                    f"{args.elmo}:{incoming}",
                ],
                timeout=4 * 60 * 60,
            )
            installed = _remote_json(
                args.elmo,
                args.ethernet_source,
                finalizer
                + ["install"]
                + common
                + ["--incoming", incoming, "--day", str(row["date"])],
            )
            if installed.get("sha256") != checksum:
                raise RuntimeError(
                    f"Elmo checksum mismatch after transfer for {row['date']}"
                )
            _run(
                _ssh_prefix(args.elmo, args.ethernet_source)
                + ["rm", "-f", incoming],
                timeout=60,
            )
            transferred.append(installed)

        receipt = _remote_json(
            args.elmo,
            args.ethernet_source,
            finalizer + ["commit"] + common,
        )
        by_day = {
            str(row["date"]): str(row["sha256"])
            for row in receipt.get("archives") or ()
        }
        if (
            receipt.get("status") != "ready"
            or receipt.get("days") != 20
            or set(by_day) != {str(row["date"]) for row in rows}
        ):
            raise RuntimeError("Elmo did not commit the exact latest-20 window")

        # Publish the tiny authoritative receipt to Inzi for selectors and the
        # dashboard. Replay archives never traverse this link.
        inzi_staging = root / ".expert-latest20-current.json"
        inzi_staging.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        inzi_parent = str(Path(args.inzi_receipt).parent)
        _run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                args.inzi,
                "mkdir",
                "-p",
                inzi_parent,
            ],
            timeout=60,
        )
        remote_temporary = args.inzi_receipt + ".tmp"
        _run(
            [
                "rsync",
                "-a",
                "-e",
                "ssh -o BatchMode=yes -o ConnectTimeout=10",
                str(inzi_staging),
                f"{args.inzi}:{remote_temporary}",
            ],
            timeout=60,
        )
        _run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                args.inzi,
                "mv",
                remote_temporary,
                args.inzi_receipt,
            ],
            timeout=60,
        )
        inzi_staging.unlink(missing_ok=True)

        # Bert is ingress, never the long-term archive. Delete only after Elmo
        # has committed the exact remote checksum for every local download.
        removed: list[str] = []
        for archive in cache.glob(f"{ARCHIVE_PREFIX}*.zip"):
            day = archive.stem.removeprefix(ARCHIVE_PREFIX)
            if by_day.get(day) == _sha256(archive):
                archive.unlink()
                removed.append(day)
        shutil.rmtree(cache / ".download", ignore_errors=True)
        local_receipt = {
            "schema": SCHEMA,
            "status": "ready",
            "window_start": receipt["window_start"],
            "window_end": receipt["window_end"],
            "days": 20,
            "elmo_receipt": f"{args.elmo_receipt_root}/current.json",
            "transferred": transferred,
            "bert_downloads_removed": sorted(removed),
            "persistent_archives_on_bert": 0,
        }
        temporary = root / ".current.json.tmp"
        temporary.write_text(
            json.dumps(local_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, root / "current.json")
        return local_receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/tsinzitari/poke-expert-refresh"),
    )
    parser.add_argument("--wifi-interface", default="en1")
    parser.add_argument("--ethernet-source", default="192.168.1.158")
    parser.add_argument("--elmo", default="admin@192.168.1.143")
    parser.add_argument(
        "--remote-stage", default="/home/admin/poke-expert-refresh"
    )
    parser.add_argument(
        "--elmo-finalizer",
        default=(
            "/mnt/Main/main/poke-bot-agent/tools/"
            "finalize_expert_latest20_elmo.py"
        ),
    )
    parser.add_argument(
        "--elmo-archive-root",
        default="/mnt/Main/main/poke-bot-agent/archive/episode-days",
    )
    parser.add_argument(
        "--elmo-receipt-root",
        default="/mnt/Main/main/poke-bot-agent/archive/expert-latest20",
    )
    parser.add_argument("--inzi", default="inzi@192.168.1.151")
    parser.add_argument(
        "--inzi-receipt",
        default=(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "expert-latest20-current.json"
        ),
    )
    parser.add_argument(
        "--elmo-reuse-root",
        action="append",
        default=[
            "/mnt/Main/main/poke-adapter-oracle-v29/data/raw",
            "/mnt/Main/main/poke-feature-latest10/data/episodes/raw",
            "/mnt/Main/main/poke-feature-refresh-20260721/data/episodes/raw",
        ],
    )
    args = parser.parse_args()
    print(json.dumps(refresh(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
