#!/usr/bin/env python3
"""Apply one evidence-backed guide weight to a future specialist boundary.

This controller is prospective-only. It refuses registry rows without the
revision-44 future-run policy, waits for a committed five-iteration hard
pause, changes only the active specialist's auxiliary guide weight through
the managed trainer service, and binds the restart to both the compiled
schedule and the trainer's append-only design-migration receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.pure_rl.deck_guide_schedule import (  # noqa: E402
    GuideWeightState,
    update_after_evaluation,
)


SCHEMA = "poke_bot.future_specialist_guide_weight_boundary/v1"
SCHEDULE_SCHEMA = "poke_bot.current_deck_guide_weight_schedule/v1"
POLICY_SCHEMA = "poke_bot.current_deck_guide_weight_policy/v1"
MIGRATION_REASON = "receipt_backed_current_deck_guide_weight_curve_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def atomic_bytes(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def selector_value(text: str, key: str) -> str | None:
    values = [
        line.split("=", 1)[1]
        for line in text.splitlines()
        if line.startswith(key + "=")
    ]
    if len(values) > 1:
        raise RuntimeError(f"selector contains duplicate {key}")
    return values[0] if values else None


def set_selector_value(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    matches = [
        index
        for index, line in enumerate(lines)
        if line.startswith(key + "=")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"selector must contain exactly one {key}")
    lines[matches[0]] = f"{key}={value}"
    return "\n".join(lines) + "\n"


def validate_and_stage_registry(
    registry: dict[str, Any],
    selector_text: str,
    schedule: dict[str, Any],
) -> tuple[str, float, float, dict[str, Any]]:
    """Return the only allowed prospective registry mutation."""

    specialist_id = str(
        selector_value(selector_text, "POKEBOT_ACTIVE_SPECIALIST") or ""
    ).strip()
    row = dict((registry.get("specialists") or {}).get(specialist_id) or {})
    policy = dict(row.get("guide_weight_policy") or {})
    previous = dict(schedule.get("previous_state") or {})
    following = dict(schedule.get("next_state") or {})
    overall = dict(schedule.get("overall") or {})
    old_weight = float(row.get("guide_loss_weight", -1.0))
    old_nonpositive = int(
        policy.get("consecutive_nonpositive_evaluations", -1)
    )
    new_weight = float(following.get("weight", -1.0))
    previous_nonpositive = int(
        previous.get("consecutive_nonpositive_evaluations", -1)
    )
    lower_bound = float(
        overall.get(
            "realized_win_rate_delta_lower_confidence_bound",
            float("nan"),
        )
    )
    expected = update_after_evaluation(
        GuideWeightState(
            weight=old_weight,
            consecutive_nonpositive_evaluations=old_nonpositive,
        ),
        realized_win_rate_delta_lower_confidence_bound=lower_bound,
    )
    guide_on = dict(schedule.get("guide_on_checkpoint") or {})
    guide_off = dict(schedule.get("guide_off_checkpoint") or {})
    checkpoint_rows = (guide_on, guide_off)
    if (
        not specialist_id
        or specialist_id == "teal-mask-ogerpon-ex"
        or schedule.get("schema") != SCHEDULE_SCHEMA
        or schedule.get("status")
        not in {"ready_for_clean_boundary", "hold"}
        or bool(schedule.get("changed")) != (old_weight != new_weight)
        or str(schedule.get("specialist_id") or "") != specialist_id
        or int(schedule.get("completed_iteration", -1)) < 5
        or int(schedule.get("completed_iteration", -1)) % 5
        or int(
            schedule.get(
                "earliest_activation_boundary_next_iteration",
                -1,
            )
        )
        != int(schedule.get("completed_iteration", -1)) + 1
        or schedule.get("application_boundary")
        != "first_available_future_five_iteration_hard_pause"
        or schedule.get("training_eligible") is not False
        or schedule.get("replay_eligible") is not False
        or schedule.get("formal_gate") is not False
        or schedule.get("serving_allowed") is not False
        or schedule.get("promotion_allowed") is not False
        or len(str(schedule.get("evidence_sha256") or "")) != 71
        or not str(schedule.get("evidence_sha256") or "").startswith(
            "sha256:"
        )
        or any(
            not Path(str(checkpoint.get("path") or "")).is_file()
            or sha256(Path(str(checkpoint.get("path") or "")))
            != str(checkpoint.get("sha256") or "")
            for checkpoint in checkpoint_rows
        )
        or guide_off.get("shadow_only") is not True
        or guide_off.get("serving_allowed") is not False
        or guide_off.get("promotion_allowed") is not False
        or policy.get("schema") != POLICY_SCHEMA
        or int(policy.get("prospective_scope_revision") or 0) != 44
        or policy.get("scope") != "future_specialist_training_runs_only"
        or policy.get(
            "retroactive_application_to_completed_frozen_or_started_runs"
        )
        is not False
        or policy.get("historical_weight_or_receipt_rewrite_allowed")
        is not False
        or float(previous.get("weight", -1.0)) != old_weight
        or previous_nonpositive != old_nonpositive
        or old_nonpositive < 0
        or not 0.0 <= old_weight <= 0.50
        or not 0.0 <= new_weight <= 0.50
        or (
            old_weight == new_weight
            and old_nonpositive
            == int(
                following.get(
                    "consecutive_nonpositive_evaluations",
                    -1,
                )
            )
        )
        or expected.weight != new_weight
        or expected.consecutive_nonpositive_evaluations
        != int(
            following.get("consecutive_nonpositive_evaluations", -1)
        )
    ):
        raise RuntimeError(
            "guide-weight schedule is not authorized for this future run"
        )
    staged = json.loads(json.dumps(registry))
    staged["specialists"][specialist_id]["guide_loss_weight"] = new_weight
    staged["specialists"][specialist_id]["guide_weight_policy"][
        "consecutive_nonpositive_evaluations"
    ] = expected.consecutive_nonpositive_evaluations
    return specialist_id, old_weight, new_weight, staged


def run(
    argv: list[str],
    *,
    timeout: float = 90.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if check and result.returncode:
        raise RuntimeError(
            f"command exited {result.returncode}: {' '.join(argv)}"
        )
    return result


def systemctl(*args: str, timeout: float = 90.0) -> str:
    return run(
        ["systemctl", "--user", *args],
        timeout=timeout,
    ).stdout.strip()


def service_value(unit: str, key: str) -> str:
    return systemctl("show", unit, "-p", key, "--value", timeout=15.0)


def wait_inactive(unit: str, timeout_seconds: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if (
            service_value(unit, "ActiveState") in {"inactive", "failed"}
            and int(service_value(unit, "MainPID") or 0) == 0
        ):
            return
        time.sleep(0.10)
    raise RuntimeError("managed trainer did not stop at the boundary")


def hard_pause_boundary(
    run_dir: Path,
    log_path: Path,
    *,
    earliest_completed_iteration: int,
) -> tuple[int, Path, dict[str, Any], Path, str] | None:
    loop_path = run_dir / "loop_state.json"
    if not loop_path.is_file() or not log_path.is_file():
        return None
    loop = read_json(loop_path)
    completed = int(loop.get("last_completed_iteration", -1))
    next_iteration = int(loop.get("next_iteration", -1))
    if (
        completed < earliest_completed_iteration
        or completed % 5
        or next_iteration != completed + 1
    ):
        return None
    commit_path = run_dir / "commits" / f"iter_{completed:05d}.json"
    if not commit_path.is_file() or read_json(commit_path) != loop:
        return None
    tail_bytes = min(log_path.stat().st_size, 2 * 1024 * 1024)
    with log_path.open("rb") as stream:
        stream.seek(-tail_bytes, os.SEEK_END)
        tail = stream.read().decode("utf-8", errors="replace")
    started = tail.rfind(
        f"GATE_BOUNDARY_HARD_PAUSE iteration={completed} "
    )
    ended = tail.rfind(
        f"GATE_BOUNDARY_HARD_PAUSE_COMPLETE iteration={completed} "
    )
    if started < 0 or ended > started:
        return None
    checkpoint_row = dict(loop.get("learner") or {})
    checkpoint = Path(
        str(checkpoint_row.get("path") or "")
    ).expanduser().resolve()
    checkpoint_sha = str(checkpoint_row.get("digest") or "")
    if not checkpoint.is_file() or sha256(checkpoint) != checkpoint_sha:
        raise RuntimeError("boundary checkpoint identity changed")
    return completed, commit_path.resolve(), loop, checkpoint, checkpoint_sha


def wait_migration(
    run_dir: Path,
    *,
    boundary_next_iteration: int,
    timeout_seconds: float = 120.0,
) -> Path:
    required = {
        "learner.alakazam_guide_loss_weight",
        "learner.current_deck_guide_loss_weight",
        "expert_rehearsal.loss_weights.alakazam_guide",
    }
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for path in sorted(
            (run_dir / "design_migrations").glob("migration_*.json"),
            reverse=True,
        ):
            receipt = read_json(path)
            changed = set(receipt.get("changed_paths") or ())
            if (
                receipt.get("reason") == MIGRATION_REASON
                and int(receipt.get("boundary_next_iteration", -1))
                == boundary_next_iteration
                and required <= changed
                and changed <= required | {"source.source_tree_sha256"}
            ):
                return path.resolve()
        time.sleep(0.10)
    raise RuntimeError("trainer did not publish guide-weight migration receipt")


def publish(path: Path, **values: Any) -> None:
    atomic_bytes(
        path,
        (
            json.dumps(
                {
                    "schema": SCHEMA,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    **values,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        0o600,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.10)
    parser.add_argument("--timeout-seconds", type=float, default=43200.0)
    args = parser.parse_args()

    schedule_path = args.schedule.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    log_path = args.log.expanduser().resolve()
    selector_path = args.selector.expanduser().resolve()
    registry_path = args.registry.expanduser().resolve()
    status_path = args.status.expanduser().resolve()
    selector_bytes = selector_path.read_bytes()
    registry_bytes = registry_path.read_bytes()
    selector_text = selector_bytes.decode()
    schedule = read_json(schedule_path)
    registry = read_json(registry_path)
    specialist_id, old_weight, new_weight, staged = (
        validate_and_stage_registry(registry, selector_text, schedule)
    )
    next_nonpositive = int(
        staged["specialists"][specialist_id]["guide_weight_policy"][
            "consecutive_nonpositive_evaluations"
        ]
    )
    old_nonpositive = int(
        dict(schedule["previous_state"])[
            "consecutive_nonpositive_evaluations"
        ]
    )
    persistent_selector = set_selector_value(
        selector_text,
        "POKEBOT_GUIDE_CONSECUTIVE_NONPOSITIVE_EVALUATIONS",
        str(next_nonpositive),
    )
    if service_value(args.unit, "ActiveState") != "active":
        raise RuntimeError("managed future-specialist trainer is not active")
    dropin = (
        Path.home()
        / ".config/systemd/user"
        / f"{args.unit}.d/90-future-guide-weight-boundary.conf"
    )
    publish(
        status_path,
        status="waiting_for_future_specialist_hard_pause",
        specialist_id=specialist_id,
        schedule=str(schedule_path),
        schedule_sha256=sha256(schedule_path),
        old_weight=old_weight,
        new_weight=new_weight,
    )
    deadline = time.monotonic() + max(1.0, args.timeout_seconds)
    proof = None
    earliest = int(schedule["completed_iteration"])
    while time.monotonic() < deadline:
        proof = hard_pause_boundary(
            run_dir,
            log_path,
            earliest_completed_iteration=earliest,
        )
        if proof is not None:
            break
        if service_value(args.unit, "ActiveState") != "active":
            raise RuntimeError("trainer stopped before the future boundary")
        time.sleep(max(0.05, args.poll_seconds))
    if proof is None:
        raise RuntimeError("timed out waiting for a future guide-weight boundary")
    completed, commit_path, _, checkpoint, checkpoint_sha = proof
    old_pid = int(service_value(args.unit, "MainPID") or 0)
    migration_path: Path | None = None
    try:
        atomic_bytes(dropin, b"[Unit]\nRefuseManualStop=no\n", 0o644)
        systemctl("daemon-reload")
        if (
            service_value(args.unit, "RefuseManualStop").strip().lower()
            != "no"
        ):
            raise RuntimeError(
                "managed guide-weight stop window did not open"
            )
        systemctl("stop", args.unit)
        wait_inactive(args.unit)
        atomic_bytes(
            registry_path,
            (json.dumps(staged, indent=2, sort_keys=True) + "\n").encode(),
            0o600,
        )
        if old_weight == new_weight:
            atomic_bytes(
                selector_path,
                persistent_selector.encode(),
                0o644,
            )
            dropin.unlink(missing_ok=True)
            systemctl("daemon-reload")
            if (
                service_value(args.unit, "RefuseManualStop").strip().lower()
                != "yes"
            ):
                raise RuntimeError("manual-stop protection was not restored")
            systemctl("start", args.unit)
            new_pid = int(service_value(args.unit, "MainPID") or 0)
            if (
                service_value(args.unit, "ActiveState") != "active"
                or new_pid <= 0
                or new_pid == old_pid
            ):
                raise RuntimeError(
                    "future specialist did not resume after state-only review"
                )
            publish(
                status_path,
                status="state_updated_weight_held",
                specialist_id=specialist_id,
                schedule=str(schedule_path),
                schedule_sha256=sha256(schedule_path),
                evidence_sha256=schedule["evidence_sha256"],
                completed_iteration=completed,
                boundary_next_iteration=completed + 1,
                commit=str(commit_path),
                commit_sha256=sha256(commit_path),
                checkpoint=str(checkpoint),
                checkpoint_sha256=checkpoint_sha,
                old_weight=old_weight,
                new_weight=new_weight,
                old_nonpositive_evaluations=old_nonpositive,
                new_nonpositive_evaluations=next_nonpositive,
                old_pid=old_pid,
                new_pid=new_pid,
                registry_sha256=sha256(registry_path),
                design_migration_receipt=None,
                selector_identity_restored=True,
                stop_protection_restored=True,
                service_control="managed_systemd_user_only",
                historical_specialists_modified=False,
                active_teal_modified=False,
            )
            return 0
        migrated_selector = set_selector_value(
            selector_text,
            "PURE_RL_ALLOW_CLEAN_BOUNDARY_DESIGN_MIGRATION",
            "1",
        )
        migrated_selector = set_selector_value(
            migrated_selector,
            "PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE",
            MIGRATION_REASON,
        )
        atomic_bytes(selector_path, migrated_selector.encode(), 0o644)
        dropin.unlink(missing_ok=True)
        systemctl("daemon-reload")
        if service_value(args.unit, "RefuseManualStop").strip().lower() != "yes":
            raise RuntimeError("manual-stop protection was not restored")
        systemctl("start", args.unit)
        migration_path = wait_migration(
            run_dir,
            boundary_next_iteration=completed + 1,
        )
        atomic_bytes(selector_path, persistent_selector.encode(), 0o644)
        new_pid = int(service_value(args.unit, "MainPID") or 0)
        if (
            service_value(args.unit, "ActiveState") != "active"
            or new_pid <= 0
            or new_pid == old_pid
        ):
            raise RuntimeError("future specialist did not resume after migration")
        publish(
            status_path,
            status="activated",
            specialist_id=specialist_id,
            schedule=str(schedule_path),
            schedule_sha256=sha256(schedule_path),
            evidence_sha256=schedule["evidence_sha256"],
            completed_iteration=completed,
            boundary_next_iteration=completed + 1,
            commit=str(commit_path),
            commit_sha256=sha256(commit_path),
            checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_sha,
            old_weight=old_weight,
            new_weight=new_weight,
            old_pid=old_pid,
            new_pid=new_pid,
            registry_sha256=sha256(registry_path),
            design_migration_receipt=str(migration_path),
            design_migration_receipt_sha256=sha256(migration_path),
            selector_identity_restored=True,
            stop_protection_restored=True,
            service_control="managed_systemd_user_only",
            historical_specialists_modified=False,
            active_teal_modified=False,
        )
    except Exception:
        if migration_path is None:
            atomic_bytes(registry_path, registry_bytes, 0o600)
            atomic_bytes(selector_path, selector_bytes, 0o644)
            if service_value(args.unit, "ActiveState") == "active":
                atomic_bytes(
                    dropin,
                    b"[Unit]\nRefuseManualStop=no\n",
                    0o644,
                )
                systemctl("daemon-reload")
                systemctl("stop", args.unit)
                wait_inactive(args.unit)
            dropin.unlink(missing_ok=True)
            systemctl("daemon-reload")
            systemctl("start", args.unit)
        else:
            atomic_bytes(
                selector_path,
                persistent_selector.encode(),
                0o644,
            )
            dropin.unlink(missing_ok=True)
            systemctl("daemon-reload")
            if service_value(args.unit, "ActiveState") != "active":
                systemctl("start", args.unit)
        publish(
            status_path,
            status=(
                "failed_rolled_back"
                if migration_path is None
                else "failed_after_migration_receipt_preserved"
            ),
            specialist_id=specialist_id,
            old_weight=old_weight,
            requested_weight=new_weight,
            schedule_sha256=sha256(schedule_path),
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
