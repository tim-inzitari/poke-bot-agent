#!/usr/bin/env python3
"""Extend the corrected 22-route Elmo tree through 2026-07-23.

The 20 validated v32 row shards are hard-linked and verified. Only the two new
daily archives are featurized. Output remains inactive pending precision/support
audit and an explicit safe-boundary activation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


SOURCE_CONTAINER = "pokebot-public-tree-v29"
TARGET_CONTAINER = "pokebot-public-tree-calibration-v33"
ROOT = Path("/mnt/Main/main/poke-adapter-oracle-v29")
ARCHIVE_ROOT = Path("/mnt/Main/main/poke-bot-agent/archive/episode-days")
SOURCE_ROWS = ROOT / "output/public-matchup-tree-calibration-v32/row-shards"
OUTPUT_ROOT = ROOT / "output/public-matchup-tree-calibration-v33"
OUTPUT = "/work/output/public-matchup-tree-calibration-v33"
TRAINER = ROOT / "src/scripts/train_public_matchup_tree_v32.py"
STAGED_ARCHETYPES = ROOT / "src/poke_bot/archetypes_v33.py"
EXTRA_DAYS = ("2026-07-22", "2026-07-23")


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
    if not TRAINER.is_file():
        raise FileNotFoundError(TRAINER)
    if not STAGED_ARCHETYPES.is_file():
        raise FileNotFoundError(STAGED_ARCHETYPES)
    source = TRAINER.read_text(encoding="utf-8")
    if (
        "_expanded_predict_proba" not in source
        or "expanded_to_canonical_class_indexes" not in source
    ):
        raise RuntimeError("corrected canonical probability calibration is missing")
    if not SOURCE_ROWS.is_dir():
        raise FileNotFoundError(SOURCE_ROWS)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    destination_rows = OUTPUT_ROOT / "row-shards"
    if not destination_rows.exists():
        shutil.copytree(SOURCE_ROWS, destination_rows, copy_function=os.link)

    extra_container_archives: list[str] = []
    for day in EXTRA_DAYS:
        source_archive = ARCHIVE_ROOT / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        if not source_archive.is_file():
            raise FileNotFoundError(source_archive)
        if not _sha256(source_archive).startswith("sha256:"):
            raise RuntimeError(f"archive identity unavailable for {day}")
        extra_container_archives.append(f"/episode-days/{source_archive.name}")

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
    for archive in reversed(extra_container_archives):
        command[insert_at:insert_at] = ["--archive", archive]
    _replace_arg(command, "--output-dir", OUTPUT)
    _replace_arg(command, "--jobs", "4")
    command.extend(("--runtime-precision-floor", "0.93"))
    command.append("--reuse-row-shards")

    if (OUTPUT_ROOT / "PUBLIC_MATCHUP_TREE_READY.json").exists():
        raise FileExistsError("v33 output already exists; refusing replacement")
    subprocess.run(
        ["sudo", "docker", "rm", "-f", TARGET_CONTAINER],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log = OUTPUT_ROOT / "public-matchup-tree-calibration-v33.log"
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
        (OUTPUT_ROOT / "PUBLIC_MATCHUP_TREE_READY.json").read_text(encoding="utf-8")
    )
    artifact = OUTPUT_ROOT / "public-matchup-tree.json"
    if receipt.get("artifact_sha256") != _sha256(artifact):
        raise RuntimeError("v33 tree checksum mismatch")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    contract = dict(payload.get("calibration_contract") or {})
    source_archives = {str(row["archive"]) for row in payload.get("sources") or ()}
    if (
        contract.get("probability_columns")
        != "expanded_to_canonical_class_indexes"
        or int(contract.get("canonical_class_count") or 0) != 23
        or any(
            f"pokemon-tcg-ai-battle-episodes-{day}.zip" not in source_archives
            for day in EXTRA_DAYS
        )
    ):
        raise RuntimeError("v33 calibration or source contract invalid")


if __name__ == "__main__":
    main()
