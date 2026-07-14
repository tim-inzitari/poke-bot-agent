#!/usr/bin/env python3
"""Run isolated GPU0/GPU1 trusted-search launch canaries concurrently."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint as checkpoint_io
from poke_bot.promotion import CheckpointIdentity


FATAL = re.compile(
    r"FAIL-CLOSED|FATAL HEALTH GATE|ActionSpaceTooLarge|"
    r"remote response slot overflow|BeliefSupportError|hidden-state leakage|"
    r"info-set violation|stale.*generation|generation.*mismatch|"
    r"zero_target_games=[1-9]|trust_failures=[1-9]|game_timeouts=[1-9]",
    re.IGNORECASE,
)
HEALTH = re.compile(r"\[rr\] health iter=(\d+): (?P<body>.+)")


@dataclass
class RunningCanary:
    profile: str
    process: subprocess.Popen[bytes]
    log_path: Path
    stream: object
    offset: int = 0


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("canary", "full"), default="canary")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu0-checkpoint", type=Path)
    parser.add_argument("--gpu1-checkpoint", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument(
        "--opponents", nargs="+", default=["dedquoc-rule-based"]
    )
    parser.add_argument("--skip-baseline-compat", action="store_true")
    parser.add_argument("--no-compat-cache", action="store_true")
    return parser.parse_args(argv)


def _checkpoint(explicit: Path | None, candidates: list[str]) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for pattern in candidates:
        matches = sorted((ROOT / "outputs/checkpoints").glob(pattern))
        if matches:
            return matches[-1].resolve()
    raise FileNotFoundError(
        "no immutable checkpoint matched: " + ", ".join(candidates)
    )


def _active_blackwell_checkpoint() -> Path | None:
    latest_files = sorted(
        (ROOT / "outputs/checkpoints").glob("trusted_factorized_*.latest.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for latest in latest_files:
        try:
            payload = checkpoint_io.load_checkpoint(latest, map_location="cpu")
            state = (payload.get("extra") or {}).get("loop_state") or {}
            incumbent = state.get("incumbent") or {}
            path = Path(incumbent.get("path") or "")
            if path.is_file():
                return path.resolve()
        except Exception:
            continue
    return None


def _active_core_checkpoint() -> Path | None:
    states = sorted(
        (ROOT / "outputs/runs").glob(
            "core_kernel_3080ti_*/pipeline_state.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for state_path in states:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            artifacts = state.get("artifacts") or {}
            for key in (
                "core_deep_search_checkpoint",
                "core_bc_checkpoint",
            ):
                path = Path((artifacts.get(key) or {}).get("path") or "")
                if path.is_file():
                    return path.resolve()
        except Exception:
            continue
    return None


def _command(
    args: argparse.Namespace,
    *,
    profile: str,
    checkpoint: Path,
    run_name: str,
) -> list[str]:
    gpu0 = profile == "gpu0-core"
    command = [
        args.python,
        "-u",
        str(ROOT / "scripts/train_round_robin.py"),
        "--archetype",
        "core-canonical" if gpu0 else "hammer-pult",
        "--run-name",
        run_name,
        "--replay-lineage",
        run_name,
        "--bootstrap-ckpt",
        str(checkpoint),
        "--resume",
        "none",
        "--iterations",
        "2",
        "--games-per-opp",
        "2",
        "--min-games-per-opp",
        "2",
        "--max-games-per-opp",
        "2",
        "--target-search-decisions",
        "0",
        "--games-per-opp-late",
        "2",
        "--curriculum-switch-iter",
        "0",
        "--workers",
        "8" if args.mode == "full" else "4",
        "--no-worker-autotune",
        "--agent-mode",
        "belief-mcts",
        "--mcts-sims",
        "128",
        "--mcts-move-time",
        "8",
        "--game-timeout-s",
        "1200" if gpu0 else "600",
        "--expected-search-decisions",
        "128" if gpu0 else "64",
        "--leaf-eval",
        "gpu-server",
        "--leaf-gpu",
        "cuda:0",
        "--leaf-servers",
        "1",
        "--leaf-max-batch",
        "128",
        "--leaf-queue-depth",
        "32",
        "--leaf-coalesce-ms",
        "3",
        "--sim-device",
        "cpu",
        "--train-epochs",
        "0",
        "--replay-fraction",
        "0.5",
        "--policy-anchor-ratio",
        "0",
        "--heldout-fraction",
        "0",
        "--promotion-games",
        "40",
        "--promotion-max-games",
        "40",
        "--promotion-batch-games",
        "20",
        "--promotion-min-pairs",
        "20",
        "--promotion-mcts-sims",
        "128",
        "--promotion-move-time",
        "8",
    ]
    if args.mode == "canary":
        command.extend(["--only", *args.opponents])
    return command


def _start(
    args: argparse.Namespace,
    *,
    profile: str,
    physical_gpu: int,
    checkpoint: Path,
    output_dir: Path,
    stamp: str,
) -> RunningCanary:
    run_name = f"prelaunch_{profile}_{stamp}"
    log_path = output_dir / f"{profile}.log"
    stream = log_path.open("wb", buffering=0)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "POKEBOT_PRIMARY_ARCHETYPE": (
                "core-canonical" if physical_gpu == 0 else "hammer-pult"
            ),
            "POKEBOT_WORKER_CPU_ONLY": "1",
            "POKEBOT_ALLOW_ORACLE_DECK": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    process = subprocess.Popen(
        _command(
            args,
            profile=profile,
            checkpoint=checkpoint,
            run_name=run_name,
        ),
        cwd=ROOT,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return RunningCanary(profile, process, log_path, stream)


def _stop(run: RunningCanary) -> None:
    if run.process.poll() is not None:
        return
    try:
        os.killpg(run.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _validate(run: RunningCanary) -> dict:
    text = run.log_path.read_text(encoding="utf-8", errors="replace")
    fatal = FATAL.search(text)
    if fatal:
        raise RuntimeError(f"{run.profile}: {fatal.group(0)}")
    health = list(HEALTH.finditer(text))
    if len(health) != 2:
        raise RuntimeError(
            f"{run.profile}: expected two health rows, found {len(health)}"
        )
    for match in health:
        body = match.group("body")
        required = (
            "completed_sims_per_decision=128.0",
            "fail_closed_games=0/",
            "trust_failures=0",
            "game_timeouts=0",
            "zero_target_games=0",
            "opponent_seat_coverage=True",
        )
        missing = [token for token in required if token not in body]
        if missing:
            raise RuntimeError(
                f"{run.profile}: unhealthy iteration {match.group(1)} "
                f"missing {missing}"
            )
    return {
        "profile": run.profile,
        "iterations": 2,
        "health": [match.group(0) for match in health],
        "log": str(run.log_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "outputs/test-runs" / f"{args.mode}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    gpu0 = _checkpoint(
        args.gpu0_checkpoint or _active_core_checkpoint(),
        ["core_kernel_3080ti_*.core_bc.accepted.*.pt", "core_kernel.best.pt"],
    )
    gpu1 = _checkpoint(
        args.gpu1_checkpoint or _active_blackwell_checkpoint(),
        [
            "trusted_factorized_*.evaluated.*.pt",
            "trusted_factorized_*.candidate.iter*.pt",
        ],
    )
    identities = {
        "gpu0-core": CheckpointIdentity.from_path(gpu0).as_dict(),
        "gpu1-hammer": CheckpointIdentity.from_path(gpu1).as_dict(),
    }
    runs: list[RunningCanary] = []
    try:
        runs.append(
            _start(
                args,
                profile="gpu0-core",
                physical_gpu=0,
                checkpoint=gpu0,
                output_dir=output_dir,
                stamp=stamp,
            )
        )
        runs.append(
            _start(
                args,
                profile="gpu1-hammer",
                physical_gpu=1,
                checkpoint=gpu1,
                output_dir=output_dir,
                stamp=stamp,
            )
        )
    except BaseException:
        for run in runs:
            _stop(run)
            run.process.wait(timeout=30)
            run.stream.close()
        raise
    started = time.perf_counter()
    gpu_budget = float(args.timeout_seconds) * 0.75
    failure: str | None = None
    try:
        while any(run.process.poll() is None for run in runs):
            if time.perf_counter() - started > gpu_budget:
                failure = (
                    f"GPU canary timeout after {gpu_budget:.0f}s"
                )
                break
            for run in runs:
                text = run.log_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                fatal = FATAL.search(text[run.offset :])
                run.offset = len(text)
                if fatal:
                    failure = f"{run.profile}: {fatal.group(0)}"
                    break
                if run.process.poll() not in (None, 0):
                    failure = (
                        f"{run.profile}: exited {run.process.returncode}; "
                        f"see {run.log_path}"
                    )
                    break
            if failure:
                break
            time.sleep(1.0)
        if failure:
            for run in runs:
                _stop(run)
            deadline = time.monotonic() + 30.0
            for run in runs:
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    run.process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(run.process.pid, signal.SIGKILL)
                    run.process.wait()
            print(f"NATIVE_CANARY_FAIL cause={failure}", file=sys.stderr)
            return 1
        try:
            gpu_reports = [_validate(run) for run in runs]
        except RuntimeError as exc:
            print(f"NATIVE_CANARY_FAIL cause={exc}", file=sys.stderr)
            return 1
    finally:
        for run in runs:
            run.stream.close()

    if not args.skip_baseline_compat:
        remaining = float(args.timeout_seconds) - (
            time.perf_counter() - started
        )
        if remaining <= 30.0:
            print(
                "NATIVE_CANARY_FAIL cause=no compatibility budget remains",
                file=sys.stderr,
            )
            return 124
        compat = [
            args.python,
            str(ROOT / "scripts/baseline_compat_canary.py"),
            "--budget-seconds",
            str(max(1.0, remaining - 20.0)),
        ]
        if args.no_compat_cache:
            compat.append("--no-cache")
        try:
            completed = subprocess.run(
                compat,
                cwd=ROOT,
                check=False,
                timeout=remaining,
            )
        except subprocess.TimeoutExpired:
            print(
                "NATIVE_CANARY_FAIL cause=baseline compatibility timeout",
                file=sys.stderr,
            )
            return 124
        if completed.returncode != 0:
            print(
                "NATIVE_CANARY_FAIL cause=baseline compatibility",
                file=sys.stderr,
            )
            return 1

    wall_s = time.perf_counter() - started
    report = {
        "schema": 1,
        "mode": args.mode,
        "ok": True,
        "wall_s": wall_s,
        "checkpoint_identities": identities,
        "gpu_reports": gpu_reports,
        "output_dir": str(output_dir),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"NATIVE_CANARY_PASS mode={args.mode} wall_s={wall_s:.2f} "
        f"report={output_dir / 'report.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
