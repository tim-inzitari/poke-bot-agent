#!/usr/bin/env python3
"""Stage historical Kaggle episode days on Elmo without retaining Inzi copies."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import zipfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(argv: list[str]) -> None:
    completed = subprocess.run(argv, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(argv)}"
        )


def stage_day(
    *,
    day: str,
    kaggle: Path,
    host: str,
    remote_root: str,
    temporary_root: Path,
) -> None:
    if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", day) is None:
        raise ValueError(f"invalid episode day: {day!r}")
    name = f"pokemon-tcg-ai-battle-episodes-{day}"
    destination = f"{remote_root.rstrip('/')}/{name}.zip"
    quoted_destination = shlex.quote(destination)
    probe = subprocess.run(
        [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            host,
            f"sudo -n test -s {quoted_destination}",
        ],
        check=False,
    )
    if probe.returncode == 0:
        print(f"[episode-history] ready day={day} source=existing", flush=True)
        return

    with tempfile.TemporaryDirectory(
        prefix=f"{name}.", dir=str(temporary_root)
    ) as raw:
        workspace = Path(raw)
        _run(
            [
                str(kaggle),
                "datasets",
                "download",
                f"kaggle/{name}",
                "--path",
                str(workspace),
                "--force",
            ]
        )
        candidates = list(workspace.glob("*.zip"))
        if len(candidates) != 1 or not zipfile.is_zipfile(candidates[0]):
            raise RuntimeError(f"downloaded Kaggle archive is invalid: {day}")
        archive = candidates[0]
        digest = _sha256(archive)
        remote_temporary = f"/tmp/.{name}.{os.getpid()}.zip"
        quoted_temporary = shlex.quote(remote_temporary)
        _run(
            [
                "/usr/bin/rsync",
                "-a",
                "--partial",
                str(archive),
                f"{host}:{remote_temporary}",
            ]
        )
        _run(
            [
                "/usr/bin/ssh",
                "-o",
                "BatchMode=yes",
                host,
                (
                    f"test \"$(sha256sum {quoted_temporary} | cut -d' ' -f1)\" "
                    f"= {shlex.quote(digest)} && sudo -n install -m 0444 "
                    f"{quoted_temporary} {quoted_destination} && "
                    f"rm -f {quoted_temporary}"
                ),
            ]
        )
        print(
            f"[episode-history] ready day={day} bytes={archive.stat().st_size} "
            f"sha256={digest}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", action="append", required=True)
    parser.add_argument(
        "--kaggle",
        type=Path,
        help="Kaggle CLI executable (preferred over the deprecated --python)",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help=(
            "Deprecated compatibility hint; its sibling kaggle executable "
            "is used because the Kaggle package has no python -m entrypoint"
        ),
    )
    parser.add_argument("--host", default="elmo")
    parser.add_argument(
        "--remote-root",
        default="/mnt/Main/main/poke-bot-agent/archive/episode-days",
    )
    parser.add_argument(
        "--temporary-root",
        type=Path,
        default=Path("/home/inzi/poke-bot-agent/outputs/tmp"),
    )
    args = parser.parse_args()
    kaggle = args.kaggle
    if kaggle is None and args.python is not None:
        sibling = args.python.expanduser().resolve().with_name("kaggle")
        if sibling.is_file():
            kaggle = sibling
    if kaggle is None:
        discovered = shutil.which("kaggle")
        if discovered:
            kaggle = Path(discovered)
    if kaggle is None or not kaggle.expanduser().resolve().is_file():
        parser.error(
            "a Kaggle CLI executable is required via --kaggle, the sibling "
            "of --python, or PATH"
        )
    kaggle = kaggle.expanduser().resolve()
    args.temporary_root.mkdir(parents=True, exist_ok=True)
    for day in args.day:
        stage_day(
            day=str(day),
            kaggle=kaggle,
            host=str(args.host),
            remote_root=str(args.remote_root),
            temporary_root=args.temporary_root.expanduser().resolve(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
