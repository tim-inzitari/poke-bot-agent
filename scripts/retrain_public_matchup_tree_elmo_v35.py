#!/usr/bin/env python3
"""Add independent simulated public-prefix evidence to inactive router v35."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


SOURCE_CONTAINER = "pokebot-public-tree-v29"
TARGET_CONTAINER = "pokebot-public-tree-calibration-v35"
ROOT = Path("/mnt/Main/main/poke-adapter-oracle-v29")
ARCHIVE_ROOT = Path("/mnt/Main/main/poke-bot-agent/archive/episode-days")
CALIBRATION = Path(
    "/mnt/Main/main/poke-bot-agent/archive/router-calibration-v35/"
    "router-calibration-v35.zip"
)
SOURCE_ROWS = ROOT / "output/public-matchup-tree-calibration-v34/row-shards"
OUTPUT_ROOT = ROOT / "output/public-matchup-tree-calibration-v35"
OUTPUT = "/work/output/public-matchup-tree-calibration-v35"
TRAINER = ROOT / "src/scripts/train_public_matchup_tree_v32.py"
STAGED_ARCHETYPES = ROOT / "src/poke_bot/archetypes_v33.py"
EXTRA_DAYS = (
    "2026-06-26",
    "2026-06-27",
    "2026-06-28",
    "2026-06-29",
    "2026-06-30",
    "2026-07-01",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _inspect(field: str):
    raw = subprocess.check_output(
        ["sudo", "docker", "inspect", SOURCE_CONTAINER, "--format", field],
        text=True,
    )
    return json.loads(raw)


def _replace_arg(command: list[str], flag: str, value: str) -> None:
    index = command.index(flag)
    command[index + 1] = value


def main() -> None:
    for required in (
        TRAINER,
        STAGED_ARCHETYPES,
        SOURCE_ROWS,
        CALIBRATION,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    output_rows = OUTPUT_ROOT / "row-shards"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not output_rows.exists():
        shutil.copytree(SOURCE_ROWS, output_rows, copy_function=os.link)

    archives = []
    for day in EXTRA_DAYS:
        archive = ARCHIVE_ROOT / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        archives.append(f"/episode-days/{archive.name}")
    if not _sha256(CALIBRATION).startswith("sha256:"):
        raise RuntimeError("simulated router calibration identity unavailable")
    archives.append("/router-calibration/router-calibration-v35.zip")

    command = [str(value) for value in _inspect("{{json .Config.Cmd}}")]
    image = str(_inspect("{{json .Config.Image}}"))
    if command[:2] != ["-lc", 'exec "$@"']:
        raise RuntimeError("source container command wrapper drifted")
    command[1] = (
        "python -m pip install --quiet --no-cache-dir scikit-learn "
        '&& exec "$@"'
    )
    command[command.index("scripts/train_public_matchup_tree.py")] = (
        "scripts/train_public_matchup_tree_v32.py"
    )
    insert_at = command.index("--target-archetype")
    for archive in reversed(archives):
        command[insert_at:insert_at] = ["--archive", archive]
    _replace_arg(command, "--output-dir", OUTPUT)
    _replace_arg(command, "--jobs", "4")
    _replace_arg(command, "--max-depth", "24")
    _replace_arg(command, "--min-samples-leaf", "20")
    command.extend(("--runtime-precision-floor", "0.93"))
    command.append("--reuse-row-shards")
    if (OUTPUT_ROOT / "PUBLIC_MATCHUP_TREE_READY.json").exists():
        raise FileExistsError("v35 output already exists; refusing replacement")
    subprocess.run(
        ["sudo", "docker", "rm", "-f", TARGET_CONTAINER],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log = OUTPUT_ROOT / "public-matchup-tree-calibration-v35.log"
    with log.open("ab", buffering=0) as stream:
        completed = subprocess.run(
            [
                "sudo",
                "docker",
                "run",
                "--name",
                TARGET_CONTAINER,
                "--cpus",
                "4",
                "--memory",
                "24g",
                "--entrypoint",
                "/bin/bash",
                "-v",
                f"{ROOT}:/work",
                "-v",
                f"{ARCHIVE_ROOT}:/episode-days:ro",
                "-v",
                f"{CALIBRATION.parent}:/router-calibration:ro",
                "-v",
                f"{STAGED_ARCHETYPES}:/work/src/poke_bot/archetypes.py:ro",
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
        raise SystemExit(completed.returncode)
    receipt = json.loads(
        (OUTPUT_ROOT / "PUBLIC_MATCHUP_TREE_READY.json").read_text(
            encoding="utf-8"
        )
    )
    artifact = OUTPUT_ROOT / "public-matchup-tree.json"
    if receipt.get("artifact_sha256") != _sha256(artifact):
        raise RuntimeError("v35 tree checksum mismatch")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    sources = {str(row["archive"]) for row in payload.get("sources") or ()}
    if CALIBRATION.name not in sources:
        raise RuntimeError("v35 calibration archive absent from source proof")


if __name__ == "__main__":
    main()
