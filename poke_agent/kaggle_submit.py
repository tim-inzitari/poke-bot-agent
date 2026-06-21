from __future__ import annotations

import os
import subprocess
from pathlib import Path

DEFAULT_SUBMISSION_MESSAGE = "Beam Search and Self Play loop added. Lucario Deck"
DEFAULT_COMPETITION = "pokemon-tcg-ai-battle"


def _kaggle_bin() -> str:
    override = os.environ.get("KAGGLE_BIN")
    if override:
        return override
    candidates = [
        Path.home() / "miniconda3/envs/poke-bot-agent/bin/kaggle",
        Path.home() / "anaconda3/envs/poke-bot-agent/bin/kaggle",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return "kaggle"


def build_submission(*, checkpoint: Path, root: Path) -> Path:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    env = os.environ.copy()
    env["VALUE_MODEL_PATH"] = str(checkpoint)
    env["PATH"] = f"{Path(_kaggle_bin()).parent}:{env.get('PATH', '')}"
    subprocess.run(
        ["bash", "scripts/build_submission.sh"],
        cwd=root,
        check=True,
        env=env,
    )
    tarball = root / "dist/submission.tar.gz"
    if not tarball.is_file():
        raise FileNotFoundError(f"submission tarball missing after build: {tarball}")
    return tarball


def submit_tarball_to_kaggle(
    tarball: Path,
    *,
    message: str,
    root: Path,
    competition: str = DEFAULT_COMPETITION,
) -> str:
    tarball = tarball.resolve()
    rel = tarball.relative_to(root.resolve()) if tarball.is_relative_to(root.resolve()) else tarball
    env = os.environ.copy()
    env["PATH"] = f"{Path(_kaggle_bin()).parent}:{env.get('PATH', '')}"
    result = subprocess.run(
        [
            _kaggle_bin(),
            "competitions",
            "submit",
            "-c",
            competition,
            "-f",
            str(rel),
            "-m",
            message,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return (result.stdout or "") + (result.stderr or "")


def champion_checkpoint_from_manifest(manifest: dict, fallback: Path) -> Path:
    champion = manifest.get("champion") or {}
    saved = champion.get("saved_checkpoint")
    if saved:
        return Path(saved)
    return fallback


def submit_champion_checkpoint(
    checkpoint: Path,
    *,
    root: Path,
    message: str = DEFAULT_SUBMISSION_MESSAGE,
    competition: str = DEFAULT_COMPETITION,
) -> dict[str, str]:
    tarball = build_submission(checkpoint=checkpoint, root=root)
    kaggle_output = submit_tarball_to_kaggle(
        tarball,
        message=message,
        root=root,
        competition=competition,
    )
    print(f"submitted {tarball.name} from {checkpoint.name}")
    if kaggle_output.strip():
        print(kaggle_output.strip())
    return {
        "checkpoint": str(checkpoint.resolve()),
        "tarball": str(tarball.resolve()),
        "message": message,
        "kaggle_output": kaggle_output,
    }
