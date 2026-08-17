#!/usr/bin/env python3
"""Download replay days on Bert Wi-Fi and archive each day to Elmo by Ethernet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
import zipfile

from poke_bot.episodes_index import (
    DailyDatasetEntry,
    download_daily_dataset,
    ensure_episodes_index,
    load_daily_manifest,
)


SCHEMA = "poke_bot.bert_kaggle_ingress_to_elmo/v1"
WIFI_INTERFACE = "en1"
ETHERNET_INTERFACE = "en0"
ETHERNET_SOURCE = "bert"
ELMO_HOST = "elmo"
ELMO_ROOT = "/srv/poke-bot-agent/archive/episode-days"
DEFAULT_KAGGLE = Path("/usr/local/bin/kaggle")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path.home() / "poke-bot-kaggle-ingress",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path.home()
        / "Library/Application Support/PokeBot/bert-kaggle-ingress.json",
    )
    parser.add_argument("--kaggle-bin", type=Path, default=DEFAULT_KAGGLE)
    parser.add_argument("--keep-local", action="store_true")
    return parser.parse_args()


def _default_interface() -> str:
    completed = subprocess.run(
        ["/sbin/route", "-n", "get", "default"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        key, separator, value = line.strip().partition(":")
        if separator and key == "interface":
            return value.strip()
    raise RuntimeError("default route did not report an interface")


def _interface_address(interface: str) -> str:
    completed = subprocess.run(
        ["/usr/sbin/ipconfig", "getifaddr", interface],
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if not value:
        raise RuntimeError(f"{interface} has no IPv4 address")
    return value


def _assert_routes() -> None:
    observed_default = _default_interface()
    if observed_default != WIFI_INTERFACE:
        raise RuntimeError(
            "Kaggle ingress must use Bert Wi-Fi: "
            f"required={WIFI_INTERFACE} observed={observed_default}"
        )
    _interface_address(WIFI_INTERFACE)
    observed_ethernet = _interface_address(ETHERNET_INTERFACE)
    if observed_ethernet != ETHERNET_SOURCE:
        raise RuntimeError(
            "Bert wired source address changed: "
            f"required={ETHERNET_SOURCE} observed={observed_ethernet}"
        )
    subprocess.run(
        [
            "/usr/bin/ssh",
            "-b",
            ETHERNET_SOURCE,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            ELMO_HOST,
            "true",
        ],
        check=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_archive(path: Path, entry: DailyDatasetEntry) -> int:
    with zipfile.ZipFile(path, "r") as archive:
        count = sum(
            name.endswith(".json") and not name.endswith("/")
            for name in archive.namelist()
        )
    if count != entry.episode_count:
        raise RuntimeError(
            f"{entry.date} archive has {count} episodes; "
            f"expected {entry.episode_count}"
        )
    return count


def _atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _remote_digest(name: str) -> str:
    completed = subprocess.run(
        [
            "/usr/bin/ssh",
            "-b",
            ETHERNET_SOURCE,
            "-o",
            "BatchMode=yes",
            ELMO_HOST,
            f"sha256sum -- {ELMO_ROOT}/{name}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.split()[0]


def _remote_validated(entry: DailyDatasetEntry) -> dict[str, object] | None:
    name = f"{entry.slug}.zip"
    remote_path = f"{ELMO_ROOT}/{name}"
    code = (
        "import hashlib,json,sys,zipfile;"
        "p=sys.argv[1];expected=int(sys.argv[2]);"
        "z=zipfile.ZipFile(p);"
        "n=sum(x.endswith('.json') and not x.endswith('/') for x in z.namelist());"
        "z.close();"
        "h=hashlib.sha256();"
        "f=open(p,'rb');"
        "[h.update(b) for b in iter(lambda:f.read(8388608),b'')];"
        "f.close();"
        "print(json.dumps({'episodes':n,'sha256':h.hexdigest()}));"
        "raise SystemExit(0 if n==expected else 3)"
    )
    completed = subprocess.run(
        [
            "/usr/bin/ssh",
            "-b",
            ETHERNET_SOURCE,
            "-o",
            "BatchMode=yes",
            ELMO_HOST,
            (
                f"test -f {shlex.quote(remote_path)} && "
                f"python3 -c {shlex.quote(code)} "
                f"{shlex.quote(remote_path)} {entry.episode_count}"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return None
    payload = json.loads(completed.stdout)
    return {
        "date": entry.date,
        "archive": name,
        "episodes": int(payload["episodes"]),
        "sha256": f"sha256:{payload['sha256']}",
        "source": "existing_elmo_archive",
    }


def _send_to_elmo(path: Path) -> None:
    subprocess.run(
        [
            "/usr/bin/ssh",
            "-b",
            ETHERNET_SOURCE,
            "-o",
            "BatchMode=yes",
            ELMO_HOST,
            f"mkdir -p -- {ELMO_ROOT}",
        ],
        check=True,
    )
    subprocess.run(
        [
            "/usr/bin/rsync",
            "-a",
            "--partial",
            "-e",
            f"ssh -b {ETHERNET_SOURCE} -o BatchMode=yes",
            str(path),
            f"{ELMO_HOST}:{ELMO_ROOT}/",
        ],
        check=True,
    )


def main() -> int:
    args = _arguments()
    if args.start_date > args.end_date:
        raise SystemExit("--start-date must not be after --end-date")
    if not args.kaggle_bin.is_file():
        raise FileNotFoundError(args.kaggle_bin)
    _assert_routes()
    rows = [
        row
        for row in load_daily_manifest(
            ensure_episodes_index(kaggle_bin=args.kaggle_bin)
        )
        if args.start_date <= row.date <= args.end_date
    ]
    if not rows or rows[0].date != args.start_date or rows[-1].date != args.end_date:
        raise RuntimeError("episode index does not cover the requested range")

    root = args.local_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    completed_rows: list[dict[str, object]] = []
    started_at = time.time()
    for index, row in enumerate(rows, 1):
        _assert_routes()
        existing_remote = _remote_validated(row)
        if existing_remote is not None:
            completed_rows.append(existing_remote)
            local_archive = root / f"{row.slug}.zip"
            if local_archive.is_file() and not args.keep_local:
                local_archive.unlink()
            continue
        _atomic(
            args.state,
            {
                "schema": SCHEMA,
                "status": "downloading_via_bert_wifi",
                "host": socket.gethostname(),
                "wifi_interface": WIFI_INTERFACE,
                "ethernet_interface": ETHERNET_INTERFACE,
                "current_date": row.date,
                "current": index,
                "total": len(rows),
                "completed": completed_rows,
                "started_at": started_at,
                "updated_at": time.time(),
            },
        )
        archive = Path(
            download_daily_dataset(
                row,
                root=root,
                kaggle_bin=args.kaggle_bin,
                unzip=False,
            )
        )
        episodes = _validate_archive(archive, row)
        digest = _sha256(archive)
        _assert_routes()
        _atomic(
            args.state,
            {
                "schema": SCHEMA,
                "status": "sending_to_elmo_via_wired_lan",
                "host": socket.gethostname(),
                "wifi_interface": WIFI_INTERFACE,
                "ethernet_interface": ETHERNET_INTERFACE,
                "current_date": row.date,
                "current": index,
                "total": len(rows),
                "completed": completed_rows,
                "started_at": started_at,
                "updated_at": time.time(),
            },
        )
        _send_to_elmo(archive)
        remote_digest = _remote_digest(archive.name)
        if remote_digest != digest:
            raise RuntimeError(
                f"Elmo digest mismatch for {archive.name}: "
                f"local={digest} remote={remote_digest}"
            )
        completed_rows.append(
            {
                "date": row.date,
                "archive": archive.name,
                "episodes": episodes,
                "bytes": archive.stat().st_size,
                "sha256": f"sha256:{digest}",
            }
        )
        if not args.keep_local:
            archive.unlink()

    _atomic(
        args.state,
        {
            "schema": SCHEMA,
            "status": "complete",
            "host": socket.gethostname(),
            "wifi_interface": WIFI_INTERFACE,
            "ethernet_interface": ETHERNET_INTERFACE,
            "completed": completed_rows,
            "started_at": started_at,
            "completed_at": time.time(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
