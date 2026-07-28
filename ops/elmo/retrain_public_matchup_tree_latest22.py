#!/usr/bin/env python3
"""Build an inactive 22-day causal matchup-router candidate on Elmo.

The job reuses the validated row shards from v31, adds July 22 and July 23,
and writes a new immutable candidate.  It never activates or deploys the tree.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time


SOURCE_CONTAINER = "pokebot-public-tree-calibrated-v31"
TARGET_CONTAINER = "pokebot-public-tree-latest22-v32"
ROOT = Path("/mnt/Main/main/poke-adapter-oracle-v29")
EPISODE_ARCHIVE = Path("/mnt/Main/main/poke-bot-agent/archive/episode-days")
OLD_OUTPUT = ROOT / "output/public-matchup-tree-calibrated-v31"
NEW_OUTPUT = ROOT / "output/public-matchup-tree-latest22-v32"
LOG = ROOT / "output/public-matchup-tree-latest22-v32.log"
STATUS = ROOT / "output/public-matchup-tree-latest22-v32.status.json"
ARCHETYPE_REGISTRY_OVERLAY = ROOT / "src/poke_bot/archetypes_v33.py"
NEW_DAYS = ("2026-07-22", "2026-07-23")
ARCHIVE_WAIT_TIMEOUT_SECONDS = 4 * 60 * 60


def atomic_status(phase: str, **values: object) -> None:
    payload = {
        "schema": "poke_bot.public_matchup_tree_refresh/v1",
        "phase": phase,
        "runtime_enabled": False,
        "source_days": 22,
        "new_days": list(NEW_DAYS),
        "updated_at_unix": time.time(),
        **values,
    }
    temporary = STATUS.with_name(f".{STATUS.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, STATUS)


def inspect(field: str):
    raw = subprocess.check_output(
        ["docker", "inspect", SOURCE_CONTAINER, "--format", field],
        text=True,
    )
    return json.loads(raw)


def replace_arg(command: list[str], flag: str, value: str) -> None:
    index = command.index(flag)
    command[index + 1] = value


def insert_archives(command: list[str]) -> None:
    target_index = command.index("--target-archetype")
    extra: list[str] = []
    for day in NEW_DAYS:
        source = EPISODE_ARCHIVE / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        if not source.is_file() or source.stat().st_size <= 0:
            raise RuntimeError(f"validated source archive is missing: {source}")
        destination = ROOT / "data/raw" / source.name
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise RuntimeError(f"archive symlink drifted: {destination}")
            # Absolute host paths are not visible through the container's
            # /work bind mount. Replace our verified legacy link with a
            # same-filesystem hard link that is valid on both sides.
            destination.unlink()
        if destination.exists():
            if (
                destination.stat().st_ino != source.stat().st_ino
                or destination.stat().st_dev != source.stat().st_dev
            ):
                raise RuntimeError(f"refusing to replace archive path: {destination}")
        else:
            destination.hardlink_to(source)
        extra.extend(("--archive", f"/work/data/raw/{source.name}"))
    command[target_index:target_index] = extra


def wait_for_archives() -> None:
    deadline = time.monotonic() + ARCHIVE_WAIT_TIMEOUT_SECONDS
    while True:
        missing = [
            day
            for day in NEW_DAYS
            if not (
                EPISODE_ARCHIVE
                / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
            ).is_file()
        ]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for validated archives: "
                + ", ".join(missing)
            )
        atomic_status("waiting_for_validated_archives", missing_days=missing)
        time.sleep(5)


def prepare_output() -> None:
    if (NEW_OUTPUT / "PUBLIC_MATCHUP_TREE_READY.json").is_file():
        return
    NEW_OUTPUT.mkdir(parents=True, exist_ok=True)
    old_rows = OLD_OUTPUT / "row-shards"
    new_rows = NEW_OUTPUT / "row-shards"
    if not new_rows.exists():
        shutil.copytree(old_rows, new_rows, copy_function=os.link)


def main() -> None:
    wait_for_archives()
    atomic_status("preparing")
    if not ARCHETYPE_REGISTRY_OVERLAY.is_file():
        raise RuntimeError(
            "validated additive archetype registry is missing: "
            f"{ARCHETYPE_REGISTRY_OVERLAY}"
        )
    command = [str(value) for value in inspect("{{json .Config.Cmd}}")]
    image = str(inspect("{{json .Config.Image}}"))
    if command[:2] != [
        "-lc",
        'python -m pip install --quiet --no-cache-dir scikit-learn && exec "$@"',
    ]:
        raise RuntimeError("source container command wrapper drifted")
    replace_arg(command, "--output-dir", "/work/output/public-matchup-tree-latest22-v32")
    replace_arg(command, "--jobs", "8")
    insert_archives(command)
    if "--reuse-row-shards" not in command:
        command.append("--reuse-row-shards")
    prepare_output()

    subprocess.run(
        ["docker", "rm", "-f", TARGET_CONTAINER],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atomic_status("training")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("ab", buffering=0) as stream:
        completed = subprocess.run(
            [
                "docker",
                "run",
                "--name",
                TARGET_CONTAINER,
                "--entrypoint",
                "/bin/bash",
                "-v",
                f"{ROOT}:/work",
                "-v",
                (
                    f"{ARCHETYPE_REGISTRY_OVERLAY}:"
                    "/work/src/poke_bot/archetypes.py:ro"
                ),
                "-w",
                "/work/src",
                "-e",
                "PYTHONPATH=/work/src",
                image,
                *command,
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        atomic_status("failed", returncode=completed.returncode, log=str(LOG))
        raise SystemExit(completed.returncode)
    receipt = NEW_OUTPUT / "PUBLIC_MATCHUP_TREE_READY.json"
    if not receipt.is_file():
        atomic_status("failed", reason="ready receipt missing", log=str(LOG))
        raise RuntimeError("router build completed without ready receipt")
    atomic_status(
        "candidate_ready_inactive",
        receipt=str(receipt),
        log=str(LOG),
    )


if __name__ == "__main__":
    main()
