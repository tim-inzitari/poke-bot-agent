"""Generic test-profile runner: budgets + command lists from a JSON manifest."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class ProfileRunner:
    manifest: Mapping[str, Any]
    history_path: Optional[Path] = None
    python: str = sys.executable

    def profile_names(self) -> list[str]:
        return sorted((self.manifest.get("profiles") or {}).keys())

    def budget_seconds(self, profile: str) -> float:
        return float(self.manifest["profiles"][profile]["budget_seconds"])

    def commands(self, profile: str) -> list[list[str]]:
        """Resolve commands for ``profile``.

        Manifest options per profile:
        - ``commands``: explicit argv lists (``{python}`` expanded)
        - ``pytest_expression``: builds ``python -m pytest -q -m <expr>``
        - ``pytest_paths``: additional pytest path args
        """
        row = self.manifest["profiles"][profile]
        if "commands" in row:
            out: list[list[str]] = []
            for cmd in row["commands"]:
                out.append([self.python if tok == "{python}" else tok for tok in cmd])
            return out
        expr = row.get("pytest_expression")
        paths = list(row.get("pytest_paths") or [])
        if expr:
            cmd = [self.python, "-m", "pytest", "-q", "-m", str(expr), *paths]
            return [cmd]
        if paths:
            return [[self.python, "-m", "pytest", "-q", *paths]]
        raise ValueError(f"profile {profile!r} has no commands or pytest_expression")

    def record(self, profile: str, status: str, wall_s: float, budget_s: float) -> None:
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "profile": profile,
                        "status": status,
                        "wall_s": wall_s,
                        "budget_s": budget_s,
                        "recorded_at": time.time(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def run_profile(
    manifest_path: str | Path,
    profile: str,
    *,
    history_path: str | Path | None = None,
    python: str = sys.executable,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    manifest = load_manifest(manifest_path)
    if profile not in (manifest.get("profiles") or {}):
        raise ValueError(f"unknown test profile {profile!r}")
    runner = ProfileRunner(
        manifest,
        history_path=Path(history_path) if history_path else None,
        python=python,
    )
    budget = runner.budget_seconds(profile)
    commands = runner.commands(profile)
    started = time.perf_counter()
    run_env = os.environ.copy()
    run_env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    if env:
        run_env.update(dict(env))
    for index, command in enumerate(commands, start=1):
        remaining = budget - (time.perf_counter() - started)
        if remaining <= 0:
            wall_s = time.perf_counter() - started
            runner.record(profile, "budget_exhausted", wall_s, budget)
            return 124
        try:
            completed = subprocess.run(
                command,
                env=run_env,
                timeout=max(1.0, remaining),
                check=False,
            )
        except subprocess.TimeoutExpired:
            wall_s = time.perf_counter() - started
            runner.record(profile, "budget_exhausted", wall_s, budget)
            return 124
        if completed.returncode != 0:
            wall_s = time.perf_counter() - started
            runner.record(profile, f"stage_{index}_failed", wall_s, budget)
            return int(completed.returncode)
    wall_s = time.perf_counter() - started
    runner.record(profile, "ok", wall_s, budget)
    return 0
