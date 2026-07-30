#!/usr/bin/env python3
"""Apply Teal's tactical-outcome 0.05 -> 0.01 rebalance after iteration 13."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REASON = "receipt_backed_teal_auxiliary_head_rebalance_v1"
SPECIALIST = "teal-mask-ogerpon-ex"
SCHEMA = "poke_bot.teal_auxiliary_head_rebalance_boundary/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic(path: Path, payload: bytes, mode: int) -> None:
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


def _publish(path: Path, **values: Any) -> None:
    payload = {
        "schema": SCHEMA,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **values,
    }
    _atomic(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        0o644,
    )


def _systemctl(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["systemctl", "--user", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if check and result.returncode:
        raise RuntimeError(
            f"systemctl {' '.join(args)} exited {result.returncode}"
        )
    return result.stdout.strip()


def _service_value(unit: str, field: str) -> str:
    return _systemctl(
        "show", unit, f"--property={field}", "--value"
    ).strip()


def _set_env(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    matches = [i for i, line in enumerate(lines) if line.startswith(key + "=")]
    if len(matches) != 1:
        raise RuntimeError(f"selector must contain exactly one {key}")
    lines[matches[0]] = f"{key}={value}"
    return "\n".join(lines) + "\n"


def _exact_boundary(run_dir: Path) -> tuple[dict[str, Any], Path] | None:
    commit_path = run_dir / "commits/iter_00013.json"
    if not commit_path.is_file():
        return None
    commit = _json(commit_path)
    loop = _json(run_dir / "loop_state.json")
    if (
        int(commit.get("last_completed_iteration", -1)) != 13
        or int(commit.get("next_iteration", -1)) != 14
        or int(loop.get("last_completed_iteration", -1)) != 13
        or int(loop.get("next_iteration", -1)) != 14
        or (run_dir / "commits/iter_00014.json").exists()
    ):
        return None
    if commit != loop:
        changed = {
            key
            for key in set(commit) | set(loop)
            if commit.get(key) != loop.get(key)
        }
        before_history = list(commit.get("design_migration_history") or ())
        after_history = list(loop.get("design_migration_history") or ())
        if (
            changed != {"design_fingerprint", "design_migration_history"}
            or len(after_history) != len(before_history) + 1
            or after_history[:-1] != before_history
        ):
            return None
        migration = dict(after_history[-1] or {})
        receipt_path = Path(
            str(migration.get("receipt") or "")
        ).expanduser().resolve()
        if not receipt_path.is_file():
            return None
        receipt = _json(receipt_path)
        receipt_changed = frozenset(receipt.get("changed_paths") or ())
        if (
            migration.get("reason") != REASON
            or int(migration.get("boundary_next_iteration", -1)) != 14
            or migration.get("fingerprint") != loop.get("design_fingerprint")
            or receipt.get("reason") != REASON
            or int(receipt.get("boundary_next_iteration", -1)) != 14
            or receipt.get("previous_fingerprint")
            != commit.get("design_fingerprint")
            or receipt_changed
            not in {
                frozenset({"learner.expanded_head_loss_weight_overrides"}),
                frozenset(
                    {
                        "learner.expanded_head_loss_weight_overrides",
                        "source.source_tree_sha256",
                    }
                ),
            }
        ):
            return None
    learner = dict(commit.get("learner") or {})
    checkpoint = Path(str(learner.get("path") or "")).expanduser().resolve()
    if (
        not checkpoint.is_file()
        or _sha256(checkpoint) != str(learner.get("digest") or "")
    ):
        raise RuntimeError("iteration-13 boundary checkpoint does not verify")
    return commit, checkpoint


def _wait_inactive(unit: str) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if (
            _service_value(unit, "ActiveState") in {"inactive", "failed"}
            and int(_service_value(unit, "MainPID") or 0) == 0
        ):
            return
        time.sleep(0.2)
    raise RuntimeError("trainer did not stop inside the boundary window")


def _wait_active(unit: str) -> int:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if (
            _service_value(unit, "ActiveState") == "active"
            and _service_value(unit, "SubState") == "running"
        ):
            pid = int(_service_value(unit, "MainPID") or 0)
            if pid > 0:
                return pid
        time.sleep(0.25)
    raise RuntimeError("trainer did not resume after the rebalance")


def _wait_exact_command(unit: str) -> int:
    """Wait through the selector/launcher exec chain for the exact trainer."""

    required = (
        b"--tactical-outcome-loss-weight-override 0.01",
        b"--current-deck-guide-loss-weight 0.05",
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if _service_value(unit, "ActiveState") == "failed":
            raise RuntimeError("resumed trainer failed before exact rebalance")
        pid = int(_service_value(unit, "MainPID") or 0)
        if pid > 0:
            command_path = Path(f"/proc/{pid}/cmdline")
            if command_path.is_file():
                try:
                    command = command_path.read_bytes().replace(b"\0", b" ")
                except FileNotFoundError:
                    command = b""
                if all(value in command for value in required):
                    return pid
        time.sleep(0.25)
    raise RuntimeError("resumed trainer command lacks exact rebalance")


def _wait_receipt(run_dir: Path) -> Path:
    required = {"learner.expanded_head_loss_weight_overrides"}
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        for path in sorted(
            (run_dir / "design_migrations").glob("migration_*.json"),
            reverse=True,
        ):
            receipt = _json(path)
            changed = set(receipt.get("changed_paths") or ())
            if (
                receipt.get("reason") == REASON
                and int(receipt.get("boundary_next_iteration", -1)) == 14
                and required <= changed
                and changed <= required | {"source.source_tree_sha256"}
            ):
                return path
        time.sleep(0.25)
    raise RuntimeError("Teal auxiliary-head design migration receipt not found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    selector_path = args.selector.expanduser().resolve()
    registry_path = args.registry.expanduser().resolve()
    status_path = args.status.expanduser().resolve()
    dropin = (
        Path.home()
        / ".config/systemd/user"
        / f"{args.unit}.d/91-teal-auxiliary-boundary.conf"
    )

    _publish(
        status_path,
        status="waiting_for_iteration_13_commit",
        completed_iteration=13,
        boundary_next_iteration=14,
        tactical_outcome_weight_before=0.05,
        tactical_outcome_weight_after=0.01,
        active_training_preempted=False,
    )
    while _exact_boundary(run_dir) is None:
        if (run_dir / "commits/iter_00014.json").exists():
            raise RuntimeError("iteration-14 boundary already passed")
        time.sleep(max(0.1, args.poll_seconds))

    commit, checkpoint = _exact_boundary(run_dir) or (None, None)
    if commit is None or checkpoint is None:
        raise RuntimeError("iteration-13 boundary disappeared")
    selector_before = selector_path.read_bytes()
    registry_before = registry_path.read_bytes()
    registry = _json(registry_path)
    row = dict((registry.get("specialists") or {}).get(SPECIALIST) or {})
    if not row or float(row.get("guide_loss_weight", -1)) != 0.05:
        raise RuntimeError("runtime registry is not the active Teal 0.05 row")
    existing_override = row.get("tactical_outcome_loss_weight_override")
    partial_activation = existing_override is not None
    if partial_activation and float(existing_override) != 0.01:
        raise RuntimeError("Teal tactical-outcome override is not the exact 0.01")
    if partial_activation:
        selector_text = selector_before.decode()
        if (
            f"PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE={REASON}\n"
            not in selector_text
            or "PURE_RL_ALLOW_CLEAN_BOUNDARY_DESIGN_MIGRATION=1\n"
            not in selector_text
        ):
            raise RuntimeError(
                "partial Teal rebalance lacks its exact selector authorization"
            )
    elif _service_value(args.unit, "ActiveState") != "active":
        raise RuntimeError("managed trainer is not active at the boundary")

    row["tactical_outcome_loss_weight_override"] = 0.01
    registry["specialists"][SPECIALIST] = row
    registry_after = (
        json.dumps(registry, indent=2, sort_keys=True) + "\n"
    ).encode()
    selector = selector_before.decode()
    selector = _set_env(
        selector, "PURE_RL_ALLOW_CLEAN_BOUNDARY_DESIGN_MIGRATION", "1"
    )
    selector = _set_env(
        selector, "PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE", REASON
    )

    stopped = False
    try:
        if partial_activation:
            if _service_value(args.unit, "ActiveState") != "active":
                _systemctl("reset-failed", args.unit, check=False)
                _systemctl("start", args.unit)
            _wait_active(args.unit)
        else:
            _atomic(dropin, b"[Unit]\nRefuseManualStop=no\n", 0o644)
            _systemctl("daemon-reload")
            _systemctl("stop", args.unit)
            stopped = True
            _wait_inactive(args.unit)
            _atomic(registry_path, registry_after, 0o600)
            _atomic(selector_path, selector.encode(), 0o600)
            _systemctl("start", args.unit)
            _wait_active(args.unit)
        new_pid = _wait_exact_command(args.unit)
        receipt = _wait_receipt(run_dir)
        clean_selector = _set_env(
            selector,
            "PURE_RL_ALLOW_CLEAN_BOUNDARY_DESIGN_MIGRATION",
            "0",
        )
        clean_selector = _set_env(
            clean_selector,
            "PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE",
            "",
        )
        _atomic(selector_path, clean_selector.encode(), 0o600)
        dropin.unlink(missing_ok=True)
        _systemctl("daemon-reload")
        if _service_value(args.unit, "RefuseManualStop").casefold() != "yes":
            raise RuntimeError("manual-stop protection was not restored")
        _publish(
            status_path,
            status="activated",
            completed_iteration=13,
            boundary_next_iteration=14,
            iteration_13_checkpoint=str(checkpoint),
            iteration_13_checkpoint_sha256=_sha256(checkpoint),
            tactical_outcome_weight_before=0.05,
            tactical_outcome_weight_after=0.01,
            guide_weight=0.05,
            new_managed_trainer_pid=new_pid,
            optimizer_state_preservation_required=True,
            selector_identity_unchanged=True,
            design_migration_receipt=str(receipt),
            design_migration_receipt_sha256=_sha256(receipt),
            active_training_preempted=False,
            stop_protection_restored=True,
        )
        return 0
    except Exception:
        if (
            not partial_activation
            and stopped
            and _service_value(args.unit, "ActiveState") != "active"
        ):
            _atomic(registry_path, registry_before, 0o600)
            _atomic(selector_path, selector_before, 0o600)
            _systemctl("start", args.unit, check=False)
        raise
    finally:
        if dropin.exists():
            dropin.unlink(missing_ok=True)
            _systemctl("daemon-reload", check=False)


if __name__ == "__main__":
    raise SystemExit(main())
