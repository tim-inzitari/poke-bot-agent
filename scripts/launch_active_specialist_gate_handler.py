#!/usr/bin/env python3
"""Resolve the active specialist into one validated pass-handler command."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "ops/specialist_runtime_registry_v1.json"
SELECTOR_ENV = "POKEBOT_ACTIVE_SPECIALIST"


def _required_text(row: dict[str, Any], field: str) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise RuntimeError(f"pass-handler configuration lacks {field}")
    return value


def _resolve_path(runtime_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (runtime_root / path).resolve()


def build_command(
    registry: dict[str, Any], specialist_id: str
) -> list[str]:
    specialists = registry.get("specialists")
    if not isinstance(specialists, dict) or specialist_id not in specialists:
        raise RuntimeError(f"unknown {SELECTOR_ENV}={specialist_id!r}")
    specialist = dict(specialists[specialist_id])
    if specialist.get("status") != "ready":
        raise RuntimeError(f"specialist {specialist_id!r} is not ready")
    handler = specialist.get("pass_handler")
    if not isinstance(handler, dict):
        raise RuntimeError(
            f"specialist {specialist_id!r} lacks pass-handler configuration"
        )
    common = registry.get("pass_handler")
    if not isinstance(common, dict):
        raise RuntimeError("registry lacks common pass-handler configuration")

    runtime_root = Path(_required_text(registry, "runtime_root")).resolve()
    python = _required_text(registry, "python")
    launcher = _resolve_path(runtime_root, _required_text(common, "launcher"))
    contract = _resolve_path(
        runtime_root, _required_text(registry, "active_gate_contract")
    )
    representatives = _resolve_path(
        runtime_root, _required_text(common, "representatives")
    )
    matchup_tree = Path(
        _required_text(specialist, "matchup_runtime_tree")
    ).resolve()
    for path in (launcher, contract, representatives, matchup_tree):
        if not path.is_file():
            raise RuntimeError(f"pass-handler input is missing: {path}")

    run_name = _required_text(specialist, "run_name")
    run_dir = Path(
        "/home/inzi/poke-bot-agent/outputs/pure_rl"
    ) / run_name
    minimum = int(
        specialist.get(
            "minimum_terminal_iteration",
            registry.get("minimum_terminal_iteration", -1),
        )
    )
    if minimum < 0:
        raise RuntimeError("invalid minimum_terminal_iteration")
    ceiling = int(
        specialist.get(
            "iteration_ceiling",
            registry.get("iteration_ceiling", minimum),
        )
    )
    if ceiling < minimum:
        raise RuntimeError("invalid iteration_ceiling")
    submission_count = int(common.get("submission_count", -1))
    if submission_count not in {1, 2}:
        raise RuntimeError("invalid pass-handler submission_count")
    ceiling_behavior = str(common.get("ceiling_behavior") or "").strip()
    if ceiling_behavior not in {
        "",
        "freeze_submit_and_continue_without_false_pass",
    }:
        raise RuntimeError("invalid pass-handler ceiling_behavior")

    command = [
        python,
        "-u",
        str(launcher),
        "--run-dir",
        str(run_dir),
        "--contract",
        str(contract),
        "--marker-name",
        _required_text(specialist, "terminal_gate_marker"),
        "--minimum-completed-iteration",
        str(minimum),
        "--ceiling-completed-iteration",
        str(ceiling),
        "--registry-root",
        _required_text(common, "registry_root"),
        "--family",
        _required_text(handler, "family"),
        "--display-name",
        _required_text(handler, "display_name"),
        "--representatives",
        str(representatives),
        "--archetype",
        specialist_id,
        "--matchup-tree",
        str(matchup_tree),
        "--submission-root",
        _required_text(handler, "submission_root"),
        "--state",
        _required_text(handler, "state"),
        "--lock",
        _required_text(handler, "lock"),
        "--competition",
        _required_text(common, "competition"),
        "--submission-count",
        str(submission_count),
        "--submission-mode",
        _required_text(common, "submission_mode"),
        "--submission-queue",
        _required_text(common, "submission_queue"),
        "--kaggle",
        _required_text(common, "kaggle"),
        "--python",
        python,
        "--authorization",
        _required_text(common, "authorization"),
        "--submission-receipts",
        _required_text(common, "submission_receipts"),
        "--training-service",
        _required_text(common, "training_service"),
        "--recover-status-143-before-gate",
        "--handoff-service",
        _required_text(handler, "handoff_service"),
        "--continue-drop-in-source",
        str(
            _resolve_path(
                runtime_root, _required_text(common, "continue_drop_in_source")
            )
        ),
        "--continue-drop-in-target",
        _required_text(common, "continue_drop_in_target"),
        "--poll-seconds",
        str(float(common.get("poll_seconds", 15))),
        "--upload-timeout-seconds",
        str(float(common.get("upload_timeout_seconds", 900))),
    ]
    if (
        ceiling_behavior
        == "freeze_submit_and_continue_without_false_pass"
    ):
        command.insert(
            command.index("--ceiling-completed-iteration") + 2,
            "--accept-ceiling-and-continue",
        )
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    registry = json.loads(args.registry.resolve().read_text(encoding="utf-8"))
    selected = str(os.environ.get(SELECTOR_ENV) or "").strip().lower()
    if not selected:
        raise RuntimeError(f"{SELECTOR_ENV} is required")
    command = build_command(registry, selected)
    print(
        f"SPECIALIST_GATE_HANDLER_OK id={selected} "
        "minimum="
        + str(
            int(
                dict(registry["specialists"][selected]).get(
                    "minimum_terminal_iteration",
                    registry["minimum_terminal_iteration"],
                )
            )
        ),
        flush=True,
    )
    print("SPECIALIST_GATE_HANDLER_COMMAND " + shlex.join(command), flush=True)
    if args.check:
        return 0
    os.execvpe(command[0], command, os.environ.copy())
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
