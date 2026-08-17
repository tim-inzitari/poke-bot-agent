#!/usr/bin/env python3
"""Add the 3080 Ti to Alakazam policy leaves after a clean RL boundary.

The CUDA simulator lane is allowed to win the device only if it has a complete,
production-eligible, full-engine parity report. Otherwise this watcher waits
for the specialist handoff and its first immutable iteration commit, installs a
small systemd drop-in, and restarts the same append-only lineage at iteration 1.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def simulator_report_is_production_eligible(report: dict[str, Any]) -> bool:
    """Require explicit full-engine parity, never a fast partial rule slice."""
    full_transition_parity = bool(
        report.get("full_engine_transition_coverage") is True
        or report.get("complete_rule_engine_coverage") is True
    )
    return bool(
        report.get("status") == "complete"
        and report.get("production_eligible") is True
        and report.get("full_seeded_game_parity") is True
        and full_transition_parity
    )


def audit_cuda_reports(patterns: Iterable[str]) -> dict[str, Any]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(value) for value in glob.glob(pattern))
    rows: list[dict[str, Any]] = []
    eligible: list[str] = []
    for path in sorted(set(paths)):
        report = read_json(path)
        row = {
            "path": str(path),
            "schema": report.get("schema"),
            "status": report.get("status"),
            "production_eligible": report.get("production_eligible") is True,
            "full_seeded_game_parity": report.get("full_seeded_game_parity") is True,
            "full_engine_transition_coverage": bool(
                report.get("full_engine_transition_coverage") is True
                or report.get("complete_rule_engine_coverage") is True
            ),
        }
        row["accepted"] = simulator_report_is_production_eligible(report)
        if row["accepted"]:
            eligible.append(str(path))
        rows.append(row)
    return {
        "production_eligible": bool(eligible),
        "eligible_reports": eligible,
        "reports": rows,
        "reason": (
            "complete production CUDA simulator report exists"
            if eligible
            else "no complete full-engine seeded-game parity report is production eligible"
        ),
    }


def systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def service_properties(name: str) -> dict[str, str]:
    result = systemctl(
        "show",
        name,
        "-p",
        "ActiveState",
        "-p",
        "MainPID",
        "-p",
        "ExecStart",
    )
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    return values


def service_state(name: str) -> tuple[str, int]:
    values = service_properties(name)
    try:
        pid = int(values.get("MainPID") or 0)
    except ValueError:
        pid = 0
    return str(values.get("ActiveState") or "unknown"), pid


def select_service_loop_state(
    *,
    service: str,
    paths: Iterable[Path],
    patterns: Iterable[str],
) -> tuple[Path | None, dict[str, Any], list[dict[str, Any]]]:
    """Select only the loop ledger named by the service's current ExecStart.

    Specialist retries use versioned run names. Picking the newest ledger by
    mtime can cut over a stale/abandoned lineage, so wildcard discovery is
    deliberately fail-closed unless exactly one ledger's ``run_name`` appears
    in the installed service command.
    """
    candidates = {Path(path) for path in paths}
    for pattern in patterns:
        candidates.update(Path(value) for value in glob.glob(pattern))
    exec_start = service_properties(service).get("ExecStart", "")
    rows: list[dict[str, Any]] = []
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(candidates):
        loop = read_json(path)
        run_name = str(loop.get("run_name") or "")
        matched = bool(run_name and run_name in exec_start)
        rows.append(
            {
                "path": str(path),
                "run_name": run_name,
                "last_completed_iteration": loop.get("last_completed_iteration"),
                "matches_service_exec_start": matched,
            }
        )
        if matched:
            matches.append((path, loop))
    if len(matches) != 1:
        return None, {}, rows
    return matches[0][0], matches[0][1], rows


def write_drop_in(
    path: Path,
    *,
    gpu0_replicas: int,
    gpu0_fraction: float,
    gpu0_client_fraction: float = 0.38,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "[Service]\n"
        f"Environment=PURE_RL_LEAF_GPU0_REPLICAS={int(gpu0_replicas)}\n"
        f"Environment=PURE_RL_LEAF_GPU0_FRAC={float(gpu0_fraction):.4f}\n"
        f"Environment=PURE_RL_GPU0_CLIENT_FRAC={float(gpu0_client_fraction):.4f}\n"
        "Environment=POKEBOT_LIVE_POOL_MAX_LEAF_GPU0=12\n"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition-state", type=Path, required=True)
    parser.add_argument("--loop-state", type=Path, action="append", default=[])
    parser.add_argument("--loop-state-glob", action="append", default=[])
    parser.add_argument("--service", required=True)
    parser.add_argument("--drop-in", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--after-specialist-iteration", type=int, default=0)
    parser.add_argument("--gpu0-replicas", type=int, default=10)
    parser.add_argument("--gpu0-fraction", type=float, default=0.30)
    parser.add_argument("--gpu0-client-fraction", type=float, default=0.38)
    parser.add_argument("--cuda-report", action="append", default=[])
    parser.add_argument("--poll-seconds", type=float, default=0.20)
    parser.add_argument("--timeout-seconds", type=float, default=604800.0)
    args = parser.parse_args()
    if not 1 <= int(args.gpu0_replicas) <= 12:
        raise SystemExit("--gpu0-replicas must be in [1, 12]")
    if not 0.0 < float(args.gpu0_fraction) < 1.0:
        raise SystemExit("--gpu0-fraction must be in (0, 1)")
    if not 0.0 < float(args.gpu0_client_fraction) < 1.0:
        raise SystemExit("--gpu0-client-fraction must be in (0, 1)")
    if not args.loop_state and not args.loop_state_glob:
        raise SystemExit("provide --loop-state or --loop-state-glob")

    reports = list(args.cuda_report) or [
        "/home/pokebot/cuda-sim-lab/outputs/*cuda*.json",
        "/home/pokebot/cuda-sim-lab/outputs/status.json",
    ]
    started = time.time()
    deadline = time.monotonic() + max(1.0, float(args.timeout_seconds))
    base = {
        "schema": "poke_bot.post_transition_3080ti/v1",
        "status": "waiting_for_specialist_transition",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "service": str(args.service),
        "after_specialist_iteration": int(args.after_specialist_iteration),
        "gpu0_replicas": int(args.gpu0_replicas),
        "gpu0_fraction": float(args.gpu0_fraction),
        "gpu0_client_fraction": float(args.gpu0_client_fraction),
        "drop_in": str(args.drop_in),
    }
    atomic_json(args.status, base)
    old_pid = 0
    while time.monotonic() < deadline:
        transition = read_json(args.transition_state)
        loop_path, loop, loop_candidates = select_service_loop_state(
            service=args.service,
            paths=args.loop_state,
            patterns=args.loop_state_glob,
        )
        active, pid = service_state(args.service)
        if pid:
            old_pid = pid
        transition_complete = transition.get("status") == "complete"
        last_completed = int(loop.get("last_completed_iteration", -1) or -1)
        current = {
            **base,
            "status": (
                "waiting_for_first_specialist_commit"
                if transition_complete
                else "waiting_for_specialist_transition"
            ),
            "transition_status": transition.get("status"),
            "specialist_service_state": active,
            "specialist_pid": pid,
            "selected_loop_state": str(loop_path) if loop_path else "",
            "selected_run_name": loop.get("run_name"),
            "loop_state_candidates": loop_candidates,
            "last_completed_iteration": last_completed,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if transition_complete and loop_path is None:
            current["status"] = "waiting_for_current_specialist_ledger"
        if (
            transition_complete
            and last_completed >= int(args.after_specialist_iteration)
        ):
            audit = audit_cuda_reports(reports)
            current["cuda_simulator_audit"] = audit
            if audit["production_eligible"]:
                current["status"] = "cuda_simulator_retains_3080ti"
                current["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_json(args.status, current)
                print(json.dumps(current, indent=2, sort_keys=True), flush=True)
                return 0

            current["status"] = "applying_3080ti_policy_leaves"
            atomic_json(args.status, current)
            write_drop_in(
                args.drop_in,
                gpu0_replicas=int(args.gpu0_replicas),
                gpu0_fraction=float(args.gpu0_fraction),
                gpu0_client_fraction=float(args.gpu0_client_fraction),
            )
            reload_result = systemctl("daemon-reload")
            if reload_result.returncode:
                raise RuntimeError(reload_result.stdout.strip())
            restart_result = systemctl("restart", args.service)
            if restart_result.returncode:
                raise RuntimeError(restart_result.stdout.strip())
            for _ in range(300):
                state, new_pid = service_state(args.service)
                if state == "active" and new_pid > 0 and new_pid != old_pid:
                    current.update(
                        status="complete",
                        applied_at_boundary_after_iteration=last_completed,
                        resumed_iteration=last_completed + 1,
                        old_pid=old_pid,
                        new_pid=new_pid,
                        completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_s=time.time() - started,
                    )
                    atomic_json(args.status, current)
                    print(json.dumps(current, indent=2, sort_keys=True), flush=True)
                    return 0
                time.sleep(0.20)
            raise RuntimeError("specialist service did not become active after 3080 Ti cutover")
        atomic_json(args.status, current)
        time.sleep(max(0.05, float(args.poll_seconds)))
    base.update(
        status="timed_out",
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    atomic_json(args.status, base)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
