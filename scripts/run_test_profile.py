#!/usr/bin/env python3
"""Run the quick, pre-launch canary, or manual full test profile."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/profile_manifest.json"
HISTORY = ROOT / "outputs/test-runs/suite_history.jsonl"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def profile_commands(profile: str, python: str) -> list[list[str]]:
    manifest = load_manifest()
    if profile not in manifest["profiles"]:
        raise ValueError(f"unknown test profile {profile!r}")
    quick_expr = manifest["profiles"]["quick"]["pytest_expression"]
    quick = [
        python,
        "-m",
        "pytest",
        "-q",
        "-m",
        quick_expr,
    ]
    if profile == "quick":
        return [quick]
    if profile == "canary":
        return [
            quick,
            [
                python,
                "-m",
                "pytest",
                "-q",
                "tests/test_native_canary.py",
                "-m",
                "native and gpu and integration and slow",
            ],
        ]
    return [
        [python, "-m", "pytest", "-q", "-m", "not native"],
        [
            python,
            str(ROOT / "scripts/native_prelaunch_canary.py"),
            "--mode",
            "full",
            "--timeout-seconds",
            str(manifest["profiles"]["full"]["budget_seconds"]),
        ],
    ]


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("quick", "canary", "full"))
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def _record(profile: str, status: str, wall_s: float, budget_s: float) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as stream:
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


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    manifest = load_manifest()
    budget = float(manifest["profiles"][args.profile]["budget_seconds"])
    commands = profile_commands(args.profile, args.python)
    started = time.perf_counter()
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    if args.profile == "canary":
        env["POKEBOT_RUN_NATIVE_CANARY"] = "1"

    for index, command in enumerate(commands, start=1):
        remaining = budget - (time.perf_counter() - started)
        if remaining <= 0:
            wall_s = time.perf_counter() - started
            _record(args.profile, "budget_exhausted", wall_s, budget)
            print(
                f"TEST_PROFILE_FAIL profile={args.profile} "
                f"cause=budget_exhausted stage={index}",
                file=sys.stderr,
            )
            return 124
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                check=False,
                timeout=remaining,
            )
        except subprocess.TimeoutExpired:
            wall_s = time.perf_counter() - started
            _record(args.profile, "timeout", wall_s, budget)
            print(
                f"TEST_PROFILE_FAIL profile={args.profile} "
                f"cause=timeout stage={index} budget_s={budget:.0f}",
                file=sys.stderr,
            )
            return 124
        if completed.returncode != 0:
            wall_s = time.perf_counter() - started
            _record(
                args.profile,
                f"stage_{index}_exit_{completed.returncode}",
                wall_s,
                budget,
            )
            print(
                f"TEST_PROFILE_FAIL profile={args.profile} "
                f"cause=stage_{index}_exit_{completed.returncode}",
                file=sys.stderr,
            )
            return completed.returncode

    wall_s = time.perf_counter() - started
    _record(args.profile, "passed", wall_s, budget)
    print(
        f"TEST_PROFILE_PASS profile={args.profile} "
        f"wall_s={wall_s:.2f} budget_s={budget:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
