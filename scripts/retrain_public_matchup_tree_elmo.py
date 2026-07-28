#!/usr/bin/env python3
"""Re-run Elmo's pinned 20-day public-tree job with calibrated routing."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


SOURCE_CONTAINER = "pokebot-public-tree-v29"
TARGET_CONTAINER = "pokebot-public-tree-calibrated-v31"
ROOT = Path("/mnt/Main/main/poke-adapter-oracle-v29")
OUTPUT = "/work/output/public-matchup-tree-calibrated-v31"


def inspect(field: str):
    raw = subprocess.check_output(
        ["sudo", "docker", "inspect", SOURCE_CONTAINER, "--format", field],
        text=True,
    )
    return json.loads(raw)


def replace_arg(command: list[str], flag: str, value: str) -> None:
    index = command.index(flag)
    command[index + 1] = value


def main() -> None:
    command = [str(value) for value in inspect("{{json .Config.Cmd}}")]
    image = str(inspect("{{json .Config.Image}}"))
    if command[:2] != ["-lc", 'exec "$@"']:
        raise RuntimeError("source container command wrapper drifted")
    command[1] = (
        "python -m pip install --quiet --no-cache-dir scikit-learn "
        '&& exec "$@"'
    )
    replace_arg(command, "--output-dir", OUTPUT)
    replace_arg(command, "--jobs", "12")
    command.extend(("--runtime-precision-floor", "0.93"))
    command.append("--reuse-row-shards")
    subprocess.run(
        ["sudo", "docker", "rm", "-f", TARGET_CONTAINER],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log = ROOT / "output/public-matchup-tree-calibrated-v31.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab", buffering=0) as stream:
        completed = subprocess.run(
            [
                "sudo",
                "docker",
                "run",
                "--name",
                TARGET_CONTAINER,
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


if __name__ == "__main__":
    main()
