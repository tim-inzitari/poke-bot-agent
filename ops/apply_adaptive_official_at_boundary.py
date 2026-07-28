#!/usr/bin/env python3
"""Activate an immutable adaptive-official trainer at a committed RL boundary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


MIGRATION_FLAG = "--allow-clean-boundary-design-migration"
MIGRATION_REASON_FLAG = "--boundary-design-migration-reason"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def publish(path: Path, **values: Any) -> None:
    payload = {
        "schema": "poke_bot.adaptive_official_boundary/v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **values,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(
    argv: list[str],
    *,
    timeout: float = 180.0,
    check: bool = True,
) -> str:
    result = subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if check and result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {' '.join(argv)}")
    return result.stdout.strip()


def service_value(unit: str, key: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", key, "--value"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=15,
        check=False,
    )
    return result.stdout.strip()


def install_unit(path: Path, text: str, unit: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)
    run(["systemctl", "--user", "daemon-reload"], timeout=30)
    run(["systemctl", "--user", "reset-failed", unit], timeout=15, check=False)


def migration_unit(stable_text: str, reason: str) -> str:
    if MIGRATION_FLAG in stable_text or MIGRATION_REASON_FLAG in stable_text:
        raise ValueError("stable unit contains persistent migration authority")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", reason):
        raise ValueError("migration reason must be one shell-safe token")
    needle = " --resume auto\n"
    replacement = (
        f" --resume auto {MIGRATION_FLAG} "
        f"{MIGRATION_REASON_FLAG} {reason}\n"
    )
    if stable_text.count(needle) != 1:
        raise ValueError("stable unit must contain exactly one '--resume auto'")
    return stable_text.replace(needle, replacement)


def validate_staged(
    *,
    staged_root: Path,
    staged_unit: Path,
    expected_source_sha256: str,
    expected_unit_sha256: str,
) -> str:
    source = staged_root / "scripts" / "train_pure_rl.py"
    if sha256_file(source) != expected_source_sha256:
        raise RuntimeError("staged trainer digest changed after validation")
    if sha256_file(staged_unit) != expected_unit_sha256:
        raise RuntimeError("staged unit digest changed after validation")
    text = staged_unit.read_text(encoding="utf-8")
    required_once = (
        f"WorkingDirectory={staged_root}",
        "Environment=PURE_RL_DECISION_CONTEXT=history",
        "Environment=PURE_RL_TEMPORAL_LAYERS=1",
        "Environment=PURE_RL_MAX_CONTEXT=320",
        "Environment=PURE_RL_OFFICIAL_ADAPTIVE_TARGETING=1",
        "Environment=PURE_RL_OFFICIAL_ADAPTIVE_MIN_SHARE=0.05",
        "Environment=PURE_RL_OFFICIAL_ADAPTIVE_GAP_POWER=2.0",
        "--games-per-iter 8192",
        "--train-max-decisions-per-batch 8192",
        "--heldout-games 1000",
        "--heldout-per-opponent-floor 0.50",
        "--resume auto",
    )
    for token in required_once:
        if text.count(token) != 1:
            raise RuntimeError(f"staged unit must contain exactly one {token!r}")
    for token in (
        "--base-checkpoint",
        MIGRATION_FLAG,
        MIGRATION_REASON_FLAG,
        "PURE_RL_BOUNDARY_MIGRATION_REASON_OVERRIDE",
    ):
        if token in text:
            raise RuntimeError(f"staged unit contains forbidden token {token!r}")
    source_text = source.read_text(encoding="utf-8")
    for token in (
        "def _latest_official_heldout_win_rates(",
        "def _adaptive_official_target_weights(",
        '"latest_exact_heldout_gap_v1"',
        '"adaptive_exact_heldout_gap_v1"',
        '"collection.official_targeting"',
    ):
        if token not in source_text:
            raise RuntimeError(f"staged trainer lacks adaptive token {token!r}")
    return text


def load_trainer(staged_root: Path) -> ModuleType:
    source = staged_root / "scripts" / "train_pure_rl.py"
    root_text = str(staged_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    spec = importlib.util.spec_from_file_location(
        "adaptive_official_boundary_train_pure_rl", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import staged trainer: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recover_partial_iteration(
    *, staged_root: Path, run_dir: Path, expected_iteration: int
) -> str | None:
    module = load_trainer(staged_root)
    state = module._load_loop_state(run_dir)
    if not isinstance(state, dict):
        raise RuntimeError("loop state vanished at boundary")
    observed = int(state.get("next_iteration", -1))
    completed = int(state.get("last_completed_iteration", -2))
    if observed != expected_iteration or completed != observed - 1:
        raise RuntimeError(
            "not at a completed N+1 boundary: "
            f"last={completed} next={observed} expected={expected_iteration}"
        )
    commit = run_dir / "commits" / f"iter_{completed:05d}.json"
    if not commit.is_file():
        raise RuntimeError(f"boundary commit is missing: {commit}")
    recovered = module._recover_interrupted_iteration(run_dir, state)
    remaining = module._iteration_artifact_paths(run_dir, observed)
    if remaining:
        raise RuntimeError(f"next-iteration artifacts remain: {remaining}")
    return str(recovered) if recovered is not None else None


def find_receipt(
    run_dir: Path,
    *,
    reason: str,
    boundary: int,
    not_before: float,
) -> tuple[Path, dict[str, Any]] | None:
    for path in sorted((run_dir / "design_migrations").glob("migration_*.json")):
        try:
            if path.stat().st_mtime < not_before:
                continue
        except OSError:
            continue
        payload = load_json(path)
        if (
            str(payload.get("reason") or "") == reason
            and int(payload.get("boundary_next_iteration", -1)) == boundary
        ):
            return path, payload
    return None


def wait_for_receipt(
    run_dir: Path,
    *,
    reason: str,
    boundary: int,
    not_before: float,
    unit: str,
    timeout: float = 240.0,
) -> tuple[Path, dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = find_receipt(
            run_dir, reason=reason, boundary=boundary, not_before=not_before
        )
        if found is not None:
            return found
        if service_value(unit, "ActiveState") == "failed":
            break
        time.sleep(0.25)
    raise RuntimeError("adaptive design migration did not record a receipt")


def validate_receipt(receipt: dict[str, Any]) -> None:
    changed = [str(path) for path in receipt.get("changed_paths") or []]
    if not any(
        path == "collection.official_targeting"
        or path.startswith("collection.official_targeting.")
        for path in changed
    ):
        raise RuntimeError(f"migration receipt omitted official targeting: {changed}")
    unexpected = [
        path
        for path in changed
        if not (
            path == "collection.official_targeting"
            or path.startswith("collection.official_targeting.")
            or path == "source"
            or path.startswith("source.")
        )
    ]
    if unexpected:
        raise RuntimeError(f"migration changed unexpected design paths: {unexpected}")


def wait_for_adaptive_collection(
    *,
    run_dir: Path,
    log_path: Path,
    log_offset: int,
    iteration: int,
    unit: str,
    timeout: float = 240.0,
) -> tuple[str, str]:
    deadline = time.monotonic() + timeout
    targeting_line = ""
    collect_line = ""
    while time.monotonic() < deadline:
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(log_offset)
                new_text = handle.read()
        except OSError:
            new_text = ""
        for line in new_text.splitlines():
            if "[pure_rl] adaptive official targeting " in line:
                targeting_line = line
            if f"[pure_rl] collect iter={iteration} " in line:
                collect_line = line
        runtime = load_json(run_dir / "iteration_runtime.json")
        if (
            targeting_line
            and collect_line
            and int(runtime.get("iteration", -1)) == iteration
            and str(runtime.get("phase") or "") == "collect"
        ):
            return targeting_line, collect_line
        if service_value(unit, "ActiveState") == "failed":
            break
        time.sleep(0.5)
    raise RuntimeError("adaptive collection did not publish verified live telemetry")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--service", required=True)
    parser.add_argument("--target-next-iteration", required=True, type=int)
    parser.add_argument("--staged-root", required=True, type=Path)
    parser.add_argument("--staged-unit", required=True, type=Path)
    parser.add_argument("--active-unit", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--migration-reason", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-unit-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    args = parser.parse_args()

    stable_text = validate_staged(
        staged_root=args.staged_root,
        staged_unit=args.staged_unit,
        expected_source_sha256=args.expected_source_sha256,
        expected_unit_sha256=args.expected_unit_sha256,
    )
    migration_text = migration_unit(stable_text, args.migration_reason)
    old_unit_text = args.active_unit.read_text(encoding="utf-8")
    backup = args.status.parent / "adaptive-official-boundary-backup.service"
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(old_unit_text, encoding="utf-8")
    observed = int(load_json(args.run_dir / "loop_state.json").get("next_iteration", -1))
    publish(
        args.status,
        status="armed",
        target_next_iteration=args.target_next_iteration,
        observed_next_iteration=observed,
        staged_root=str(args.staged_root),
        source_sha256=args.expected_source_sha256,
        unit_sha256=args.expected_unit_sha256,
    )

    last_health_check = 0.0
    while True:
        state = load_json(args.run_dir / "loop_state.json")
        observed = int(state.get("next_iteration", -1))
        if observed >= args.target_next_iteration:
            break
        now = time.monotonic()
        if now - last_health_check >= 5.0:
            if service_value(args.service, "ActiveState") not in ("active", "activating"):
                raise RuntimeError("production trainer stopped before boundary")
            last_health_check = now
        time.sleep(max(0.02, args.poll_seconds))

    if (args.run_dir / "SPECIALIST_GATE_PASSED").is_file():
        publish(
            args.status,
            status="skipped_terminal_gate_passed",
            observed_next_iteration=observed,
        )
        return 0

    migrated = False
    receipt_path: Path | None = None
    recovered: str | None = None
    old_pid = int(service_value(args.service, "MainPID") or 0)
    try:
        publish(
            args.status,
            status="stopping_at_committed_boundary",
            observed_next_iteration=observed,
            old_pid=old_pid,
        )
        run(["systemctl", "--user", "stop", args.service], timeout=60)
        state = load_json(args.run_dir / "loop_state.json")
        observed = int(state.get("next_iteration", -1))
        recovered = recover_partial_iteration(
            staged_root=args.staged_root,
            run_dir=args.run_dir,
            expected_iteration=observed,
        )

        # Re-hash after recovery/import and immediately before activation.
        validate_staged(
            staged_root=args.staged_root,
            staged_unit=args.staged_unit,
            expected_source_sha256=args.expected_source_sha256,
            expected_unit_sha256=args.expected_unit_sha256,
        )
        install_unit(args.active_unit, migration_text, args.service)
        log_offset = args.log.stat().st_size if args.log.exists() else 0
        migration_started = time.time() - 1.0
        run(["systemctl", "--user", "start", args.service], timeout=60)
        receipt_path, receipt = wait_for_receipt(
            args.run_dir,
            reason=args.migration_reason,
            boundary=observed,
            not_before=migration_started,
            unit=args.service,
        )
        migrated = True
        validate_receipt(receipt)

        # The running process has consumed the one-time authority. Revoke it
        # from every future start without restarting the healthy trainer.
        install_unit(args.active_unit, stable_text, args.service)
        targeting_line, collect_line = wait_for_adaptive_collection(
            run_dir=args.run_dir,
            log_path=args.log,
            log_offset=log_offset,
            iteration=observed,
            unit=args.service,
        )
        new_pid = int(service_value(args.service, "MainPID") or 0)
        if new_pid <= 0 or new_pid == old_pid:
            raise RuntimeError(f"trainer did not enter a new process: {new_pid}")
        cwd = os.readlink(f"/proc/{new_pid}/cwd")
        if Path(cwd) != args.staged_root:
            raise RuntimeError(f"trainer cwd is not immutable v8: {cwd}")
        restarts = service_value(args.service, "NRestarts")
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if service_value(args.service, "ActiveState") != "active":
                raise RuntimeError("adaptive trainer left active state")
            if int(service_value(args.service, "MainPID") or 0) != new_pid:
                raise RuntimeError("adaptive trainer restarted during stability window")
            if service_value(args.service, "NRestarts") != restarts:
                raise RuntimeError("adaptive trainer restart counter changed")
            time.sleep(1.0)
        publish(
            args.status,
            status="complete",
            observed_next_iteration=observed,
            main_pid=new_pid,
            migration_receipt=str(receipt_path),
            recovered_partial_iteration=recovered,
            targeting_line=targeting_line,
            collect_line=collect_line,
            source_sha256=args.expected_source_sha256,
            unit_sha256=args.expected_unit_sha256,
        )
        return 0
    except BaseException as exc:  # noqa: BLE001 - boundary rollback must be total
        found = find_receipt(
            args.run_dir,
            reason=args.migration_reason,
            boundary=observed,
            not_before=0.0,
        )
        migrated = migrated or found is not None
        if receipt_path is None and found is not None:
            receipt_path = found[0]
        error = f"{type(exc).__name__}: {exc}"
        publish(
            args.status,
            status="recovering_after_error",
            error=error,
            migrated=migrated,
            migration_receipt=str(receipt_path) if receipt_path else None,
            recovered_partial_iteration=recovered,
        )
        run(["systemctl", "--user", "stop", args.service], timeout=60, check=False)
        install_unit(
            args.active_unit,
            stable_text if migrated else old_unit_text,
            args.service,
        )
        run(["systemctl", "--user", "start", args.service], timeout=60, check=False)
        publish(
            args.status,
            status=("adaptive_restarted_after_error" if migrated else "v7_rolled_back"),
            error=error,
            active_state=service_value(args.service, "ActiveState"),
            main_pid=int(service_value(args.service, "MainPID") or 0),
            migrated=migrated,
            migration_receipt=str(receipt_path) if receipt_path else None,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
