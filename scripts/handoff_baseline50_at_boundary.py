#!/usr/bin/env python3
"""Fork the live RL learner into the intended 50/50 curriculum at a commit.

The source-mix fraction is an immutable learning-design field, so changing it
inside the existing lineage would be dishonest.  This watcher waits for an
append-only source commit, stops the old service, installs already-tested
sources, and starts a new lineage from the best exact-heldout checkpoint.  Any
failure restores the previous sources/unit and resumes the old lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def publish(path: Path, **values: Any) -> None:
    atomic_json(
        path,
        {
            "schema": "poke_bot.baseline50_boundary_handoff/v1",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **values,
        },
    )


def command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 180.0,
    check: bool = True,
) -> str:
    result = subprocess.run(
        argv,
        cwd=str(cwd) if cwd is not None else None,
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


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def service_property(unit: str, name: str) -> str:
    return command(
        ["systemctl", "--user", "show", unit, "-p", name, "--value"],
        timeout=15,
        check=False,
    ).strip()


def render_unit(
    text: str,
    *,
    old_run: str,
    new_run: str,
    checkpoint: Path,
    replay_shard: Path,
) -> str:
    if not old_run or text.count(old_run) < 2:
        raise RuntimeError("source unit does not consistently name the old run")
    forbidden = ("--base-checkpoint", "--initial-learner-checkpoint")
    if any(flag in text for flag in forbidden):
        raise RuntimeError("source unit already pins a lineage checkpoint")
    rendered = text.replace(old_run, new_run)
    needle = " --iterations 1000"
    addition = (
        f" --base-checkpoint {checkpoint}"
        f" --initial-learner-checkpoint {checkpoint}"
        f" --initial-replay-shard {replay_shard}"
        + needle
    )
    if rendered.count(needle) != 1:
        raise RuntimeError("source unit has an ambiguous iterations boundary")
    rendered = rendered.replace(needle, addition)
    required = (
        "Environment=PURE_RL_SELF_PLAY_FRAC=0.50",
        "--games-per-iter 8192",
        "--train-max-decisions-per-batch 12288",
        "--heldout-games 1000",
        "MemoryMax=112G",
    )
    missing = [token for token in required if token not in rendered]
    if missing:
        raise RuntimeError(f"rendered unit lost production invariants: {missing}")
    # Provenance inputs deliberately live inside the old run directory, so
    # the old token may remain in --base-checkpoint/--initial-replay-shard.
    # Only the active run identity and its writable log must be replaced.
    if (
        f"--run-name {old_run}" in rendered
        or f"--log outputs/logs/{old_run}.log" in rendered
        or rendered.count(new_run) < 2
    ):
        raise RuntimeError("run-name replacement was incomplete")
    return rendered


def inherited_official_heldout(state: dict[str, Any]) -> dict[str, Any]:
    identity = state.get("heldout_champion")
    evidence = state.get("heldout_champion_evidence")
    if not isinstance(identity, dict) or not isinstance(evidence, dict):
        raise RuntimeError("source lineage lacks an exact heldout champion")
    digest = str(identity.get("digest") or "")
    if digest != str(evidence.get("checkpoint_digest") or ""):
        raise RuntimeError("heldout identity/evidence digest mismatch")
    audit = evidence.get("audit")
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise RuntimeError("heldout champion does not have a passing exact audit")
    iteration = int(evidence.get("iteration", -1))
    history = state.get("history") if isinstance(state.get("history"), list) else []
    record = next(
        (
            row
            for row in reversed(history)
            if isinstance(row, dict) and int(row.get("iteration", -2)) == iteration
        ),
        None,
    )
    gate = record.get("raw_heldout_gate") if isinstance(record, dict) else None
    if not isinstance(gate, dict):
        raise RuntimeError("heldout champion gate record is missing")
    games = int(evidence.get("games", 0))
    if (
        games != int(gate.get("games", -1))
        or games != int(audit.get("valid_games", -2))
        or abs(float(evidence.get("win_rate", -1.0)) - float(gate.get("win_rate", -2.0)))
        > 1e-12
    ):
        raise RuntimeError("heldout evidence/gate/audit do not reconcile")
    return {
        "available": True,
        "kind": "inherited_official_heldout_champion",
        "valid": True,
        "passed": bool(gate.get("passed")),
        "reason": gate.get("reason"),
        "games": games,
        "wr": float(gate["win_rate"]),
        "lower": float(gate["confidence_lower"]),
        "upper": float(gate["confidence_upper"]),
        "lineage_iteration": iteration,
        "checkpoint": str(identity.get("path") or ""),
        "checkpoint_digest": digest,
        "per_opponent": gate.get("per_opponent"),
        "audit": audit,
        "audit_passed": True,
        "exact_distribution": bool(audit.get("exact_distribution")),
        "exact_weights": bool(audit.get("exact_weights")),
        "greedy_required": bool(audit.get("greedy_required")),
    }


def copy_sources(staged: Path, destinations: list[Path]) -> None:
    relative_files = (
        Path("poke_bot/config.py"),
        Path("poke_bot/remote_jobs.py"),
        Path("scripts/train_pure_rl.py"),
    )
    for destination in destinations:
        for relative in relative_files:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged / relative, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-loop-state", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--target-next-iteration", type=int, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--old-run", required=True)
    parser.add_argument("--new-run", required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--active-unit", type=Path, required=True)
    parser.add_argument("--base-unit", type=Path, required=True)
    parser.add_argument("--deployment-unit", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.10)
    args = parser.parse_args()

    publish(
        args.status,
        status="waiting_for_source_commit",
        target_next_iteration=args.target_next_iteration,
        observed_next_iteration=int(read_json(args.source_loop_state).get("next_iteration", -1)),
    )
    while int(read_json(args.source_loop_state).get("next_iteration", -1)) < int(
        args.target_next_iteration
    ):
        time.sleep(max(0.05, float(args.poll_seconds)))

    backup = args.status.parent / "backup"
    backup.mkdir(parents=True, exist_ok=True)
    source_backups: list[tuple[Path, Path]] = []
    for root_name, root in (("deployment", args.deployment_root), ("base", args.base_root)):
        for relative in (
            Path("poke_bot/config.py"),
            Path("poke_bot/remote_jobs.py"),
            Path("scripts/train_pure_rl.py"),
        ):
            saved = backup / root_name / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, saved)
            source_backups.append((saved, root / relative))
    unit_backups: list[tuple[Path, Path]] = []
    for name, path in (
        ("active.service", args.active_unit),
        ("base.service", args.base_unit),
        ("deployment.service", args.deployment_unit),
    ):
        saved = backup / name
        shutil.copy2(path, saved)
        unit_backups.append((saved, path))

    publish(args.status, status="stopping_source_at_commit")
    command(["systemctl", "--user", "stop", args.unit], timeout=60)
    try:
        source_state = read_json(args.source_loop_state)
        last_completed = int(source_state.get("last_completed_iteration", -1))
        if last_completed < int(args.target_next_iteration) - 1:
            raise RuntimeError("source service stopped before the requested commit")
        commit = args.source_run_dir / "commits" / f"iter_{last_completed:05d}.json"
        replay = args.source_run_dir / "shards" / f"iter_{last_completed:05d}.jsonl"
        if not commit.is_file() or not replay.is_file():
            raise RuntimeError("source commit/replay artifacts are incomplete")
        official = inherited_official_heldout(source_state)
        checkpoint = Path(official["checkpoint"])
        if not checkpoint.is_file() or sha256(checkpoint) != official["checkpoint_digest"]:
            raise RuntimeError("heldout checkpoint bytes do not match their identity")

        publish(
            args.status,
            status="validating_staged_sources",
            source_iteration=last_completed,
            checkpoint_digest=official["checkpoint_digest"],
        )
        tests = [
            "tests/test_config_env_resolution.py",
            "tests/test_pure_rl_wr_progress.py",
            "tests/test_remote_checkpoint_staging.py::test_endpoint_credit_keeps_slow_bert_fed_across_a_long_wave",
            "tests/test_remote_checkpoint_staging.py::test_scheduled_fast_elmo_cannot_starve_slow_bert_refills",
        ]
        command(
            [str(args.python), "-m", "py_compile", "poke_bot/config.py", "poke_bot/remote_jobs.py", "scripts/train_pure_rl.py"],
            cwd=args.staged_root,
            timeout=30,
        )
        command(
            [str(args.python), "-m", "pytest", "-q", *tests, "--maxfail=1"],
            cwd=args.staged_root,
            timeout=120,
        )

        new_run_dir = args.base_root / "outputs" / "pure_rl" / args.new_run
        if new_run_dir.exists():
            raise RuntimeError(f"new lineage path already exists: {new_run_dir}")
        rendered = render_unit(
            args.active_unit.read_text(),
            old_run=args.old_run,
            new_run=args.new_run,
            checkpoint=checkpoint,
            replay_shard=replay,
        )
        publish(args.status, status="installing_baseline50_lineage")
        copy_sources(args.staged_root, [args.deployment_root, args.base_root])
        for unit_path in (args.active_unit, args.base_unit, args.deployment_unit):
            unit_path.write_text(rendered)
        command(["systemctl", "--user", "daemon-reload"], timeout=30)
        command(["systemctl", "--user", "reset-failed", args.unit], timeout=15, check=False)
        command(["systemctl", "--user", "start", args.unit], timeout=60)

        loop_path = new_run_dir / "loop_state.json"
        manifest_path = new_run_dir / "manifest.json"
        deadline = time.monotonic() + 150.0
        while time.monotonic() < deadline:
            if service_property(args.unit, "ActiveState") == "failed":
                raise RuntimeError("baseline50 service entered failed state")
            loop = read_json(loop_path)
            manifest = read_json(manifest_path)
            contract = manifest.get("design_contract") if isinstance(manifest, dict) else {}
            collection = contract.get("collection") if isinstance(contract, dict) else {}
            if loop and manifest and float(collection.get("self_play_fraction", -1)) == 0.5:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("new lineage did not publish a 50/50 immutable manifest")

        source_handoff = read_json(args.source_run_dir / "lineage_handoff.json")
        source_offset = int(source_handoff.get("global_iteration_offset") or 0)
        official["iteration"] = source_offset + int(official["lineage_iteration"])
        handoff = {
            "schema": "poke_bot.pure_rl_lineage_handoff/v1",
            "source_run": args.old_run,
            "source_iteration": last_completed,
            "source_global_iteration_offset": source_offset,
            "global_iteration_offset": source_offset + last_completed + 1,
            "champion": {
                "path": str(checkpoint),
                "digest": official["checkpoint_digest"],
            },
            "inherited_heldout": {
                "games": official["games"],
                "win_rate": official["wr"],
                "passed": official["passed"],
                "source": str(args.source_loop_state),
            },
            "inherited_official_heldout": official,
            "curriculum": {
                "self_play_fraction": 0.5,
                "public_mix_fraction": 0.5,
                "reason": "activate intended public-baseline curriculum after exact env fix",
            },
        }
        atomic_json(new_run_dir / "lineage_handoff.json", handoff)

        log = args.base_root / "outputs" / "logs" / f"{args.new_run}.log"
        deadline = time.monotonic() + 150.0
        collect_contract = "self_play=4096 public_mix=4096"
        while time.monotonic() < deadline:
            text = log.read_text(errors="replace") if log.is_file() else ""
            if collect_contract in text:
                break
            if service_property(args.unit, "ActiveState") == "failed":
                raise RuntimeError("baseline50 service failed before collection")
            time.sleep(0.5)
        else:
            raise RuntimeError("new lineage did not start the exact 4096/4096 collect")

        publish(
            args.status,
            status="complete",
            source_iteration=last_completed,
            new_run=args.new_run,
            main_pid=int(service_property(args.unit, "MainPID") or 0),
            checkpoint_digest=official["checkpoint_digest"],
            replay_shard=str(replay),
            self_play_games=4096,
            public_mix_games=4096,
        )
        return 0
    except BaseException as exc:  # noqa: BLE001 - rollback is deliberately broad
        publish(args.status, status="rolling_back", error=f"{type(exc).__name__}: {exc}")
        command(["systemctl", "--user", "stop", args.unit], timeout=60, check=False)
        for saved, destination in source_backups + unit_backups:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(saved, destination)
        command(["systemctl", "--user", "daemon-reload"], timeout=30, check=False)
        command(["systemctl", "--user", "reset-failed", args.unit], timeout=15, check=False)
        command(["systemctl", "--user", "start", args.unit], timeout=60, check=False)
        publish(
            args.status,
            status="rolled_back_and_source_resumed",
            error=f"{type(exc).__name__}: {exc}",
            active_state=service_property(args.unit, "ActiveState"),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
