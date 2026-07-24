#!/usr/bin/env python3
"""Build a corrected, inactive 22-route public tree on Elmo.

This stages a new validation artifact only. It never replaces or activates the
tree used by the live specialist trainer.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


SOURCE_CONTAINER = "pokebot-public-tree-v29"
TARGET_CONTAINER = "pokebot-public-tree-calibration-v32"
ROOT = Path("/mnt/Main/main/poke-adapter-oracle-v29")
SOURCE_ROWS = ROOT / "output/public-matchup-tree-calibrated-v31/row-shards"
OUTPUT_ROOT = ROOT / "output/public-matchup-tree-calibration-v32"
OUTPUT = "/work/output/public-matchup-tree-calibration-v32"
TRAINER = ROOT / "src/scripts/train_public_matchup_tree_v32.py"


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
    source = TRAINER.read_text(encoding="utf-8")
    required = (
        "_expanded_predict_proba",
        "expanded_to_canonical_class_indexes",
    )
    if any(token not in source for token in required):
        raise RuntimeError("corrected canonical probability calibration is missing")
    if not SOURCE_ROWS.is_dir():
        raise FileNotFoundError(SOURCE_ROWS)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    destination_rows = OUTPUT_ROOT / "row-shards"
    if not destination_rows.exists():
        shutil.copytree(SOURCE_ROWS, destination_rows, copy_function=os.link)

    command = [str(value) for value in _inspect("{{json .Config.Cmd}}")]
    image = str(_inspect("{{json .Config.Image}}"))
    if command[:2] != ["-lc", 'exec "$@"']:
        raise RuntimeError("source container command wrapper drifted")
    command[1] = (
        "python -m pip install --quiet --no-cache-dir scikit-learn "
        '&& exec "$@"'
    )
    script_index = command.index("scripts/train_public_matchup_tree.py")
    command[script_index] = "scripts/train_public_matchup_tree_v32.py"
    _replace_arg(command, "--output-dir", OUTPUT)
    _replace_arg(command, "--jobs", "4")
    command.extend(("--runtime-precision-floor", "0.93"))
    command.append("--reuse-row-shards")

    if (OUTPUT_ROOT / "PUBLIC_MATCHUP_TREE_READY.json").exists():
        raise FileExistsError("v32 output already exists; refusing replacement")
    subprocess.run(
        ["sudo", "docker", "rm", "-f", TARGET_CONTAINER],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log = OUTPUT_ROOT / "public-matchup-tree-calibration-v32.log"
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
        raise RuntimeError("v32 tree checksum mismatch")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    contract = dict(payload.get("calibration_contract") or {})
    if (
        contract.get("probability_columns")
        != "expanded_to_canonical_class_indexes"
        or int(contract.get("canonical_class_count") or 0) != 23
    ):
        raise RuntimeError("v32 calibration contract missing or invalid")


if __name__ == "__main__":
    main()
