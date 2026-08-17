#!/usr/bin/env python3
"""Authorize the exact Marnie 52-socket selector repair at a stopped boundary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


SCHEMA = "poke_bot.marnie_selector_env_migration/v1"
SPECIALIST_ID = "marnie-s-grimmsnarl-ex"
RUN_NAME = "final_format_marnie_r104_h10_i_v6_8k"
SERVICE = "pokebot-final-format-marnie-r104-h10-rl.service"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _selector_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        row = raw.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        if key in values:
            raise RuntimeError(f"duplicate selector assignment: {key}")
        values[key] = value
    return values


def _unit_environment_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        row = raw.strip()
        if not row.startswith("Environment=") or "=" not in row.removeprefix(
            "Environment="
        ):
            continue
        key, value = row.removeprefix("Environment=").split("=", 1)
        if key in values:
            raise RuntimeError(f"duplicate service environment assignment: {key}")
        values[key] = value
    return values


def _stopped_service_state(service: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            service,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "MainPID",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "managed service state unavailable")
    state: dict[str, str] = {}
    for row in result.stdout.splitlines():
        if "=" in row:
            key, value = row.split("=", 1)
            state[key] = value
    active_state = state.get("ActiveState", "")
    main_pid = int(state.get("MainPID") or 0)
    if active_state in {"active", "activating", "reloading"} or main_pid > 0:
        raise RuntimeError(
            f"selector migration requires a stopped managed trainer: {state}"
        )
    return {
        "active_state": active_state,
        "sub_state": state.get("SubState"),
        "main_pid": main_pid,
    }


def _atomic_write_once(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = _read(path)
        comparable = dict(existing)
        comparable.pop("activated_at_utc", None)
        expected = dict(payload)
        expected.pop("activated_at_utc", None)
        if comparable != expected:
            raise RuntimeError(f"immutable selector migration receipt changed: {path}")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registration",
        type=Path,
        default=Path(
            "/home/pokebot/poke-bot-agent/outputs/state/"
            "final-format-marnie-r104-h10-runtime-registration.json"
        ),
    )
    parser.add_argument(
        "--selector",
        type=Path,
        default=Path(
            "/home/pokebot/poke-bot-agent/outputs/final_format_marnie_r104/"
            "runtime/specialist_runtime_h10_r104.env"
        ),
    )
    parser.add_argument(
        "--loop-state",
        type=Path,
        default=Path(
            "/home/pokebot/poke-bot-agent/outputs/pure_rl/"
            "final_format_marnie_r104_h10_i_v6_8k/loop_state.json"
        ),
    )
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=Path(
            "/home/pokebot/poke-bot-agent-deployments/final-format-marnie-h10-r104"
        ),
    )
    parser.add_argument(
        "--scheduler-dropin",
        type=Path,
        default=Path(
            "/home/pokebot/.config/systemd/user/"
            "pokebot-final-format-marnie-r104-h10-rl.service.d/"
            "zzzz-self-play-elmo-tail-r119.conf"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/pokebot/poke-bot-agent/outputs/state/"
            "final-format-marnie-r104-selector-prefetch-r125.json"
        ),
    )
    args = parser.parse_args()

    registration_path = args.registration.expanduser().resolve()
    selector_path = args.selector.expanduser().resolve()
    loop_state_path = args.loop_state.expanduser().resolve()
    deployment = args.deployment_root.expanduser().resolve()
    scheduler_dropin = args.scheduler_dropin.expanduser().resolve()
    output = args.output.expanduser().resolve()
    registration = _read(registration_path)
    loop_state = _read(loop_state_path)
    if (
        registration.get("specialist_id") != SPECIALIST_ID
        or registration.get("selector_env") != str(selector_path)
        or not str(registration.get("selector_env_sha256") or "").startswith(
            "sha256:"
        )
    ):
        raise RuntimeError("base Marnie registration receipt is not authoritative")
    if (
        loop_state.get("run_name") != RUN_NAME
        or int(loop_state.get("last_completed_iteration") or -1) != 5
        or int(loop_state.get("next_iteration") or -1) != 6
    ):
        raise RuntimeError("selector migration is not at the stopped iteration-6 boundary")

    exact_environment = {
        "POKEBOT_ACTIVE_SPECIALIST": SPECIALIST_ID,
        "POKEBOT_SPECIALIST_RUNTIME_ROOT": str(deployment),
        "PYTHONPATH": str(deployment),
        "POKEBOT_REMOTE_SOCKET_PREFETCH": "1",
        "POKEBOT_REMOTE_SOCKET_PREFETCH_MAX": "1",
    }
    exact_service_environment = {
        "POKEBOT_REMOTE_SOCKET_PREFETCH": "1",
        "POKEBOT_REMOTE_SOCKET_PREFETCH_MAX": "1",
        "POKEBOT_SELF_PLAY_ELMO_TAIL_ONLY": "1",
        "POKEBOT_SELF_PLAY_TAIL_WORK_STEAL_GAMES": "20",
    }
    selector_values = _selector_values(selector_path)
    mismatched = {
        key: {"expected": value, "actual": selector_values.get(key)}
        for key, value in exact_environment.items()
        if selector_values.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"selector lacks the exact revision-125 values: {mismatched}")
    service_values = _unit_environment_values(scheduler_dropin)
    service_mismatched = {
        key: {"expected": value, "actual": service_values.get(key)}
        for key, value in exact_service_environment.items()
        if service_values.get(key) != value
    }
    if service_mismatched:
        raise RuntimeError(
            "managed service lacks the exact revision-125 scheduler overlay: "
            f"{service_mismatched}"
        )

    payload = {
        "schema": SCHEMA,
        "status": "activated_at_stopped_uncommitted_boundary",
        "goal_revision": 125,
        "specialist_id": SPECIALIST_ID,
        "run_name": RUN_NAME,
        "base_registration_receipt": str(registration_path),
        "base_registration_sha256": _sha256(registration_path),
        "base_selector_env_sha256": registration["selector_env_sha256"],
        "selector_env": str(selector_path),
        "selector_env_sha256": _sha256(selector_path),
        "scheduler_dropin": str(scheduler_dropin),
        "scheduler_dropin_sha256": _sha256(scheduler_dropin),
        "exact_service_environment": exact_service_environment,
        "loop_state": str(loop_state_path),
        "loop_state_sha256": _sha256(loop_state_path),
        "last_completed_iteration": 5,
        "next_iteration": 6,
        "learner_digest": (loop_state.get("learner") or {}).get("digest"),
        "exact_environment": exact_environment,
        "prior_attempt_rejected": True,
        "observed_remote_sockets": 104,
        "maximum_remote_sockets": 52,
        "base_registration_preserved": True,
        "service_state_at_activation": _stopped_service_state(SERVICE),
        "activated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt = _atomic_write_once(output, payload)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
