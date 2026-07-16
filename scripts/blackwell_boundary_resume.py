#!/usr/bin/env python3
"""Boundary-only Blackwell resume with cheaper promotion budget.

This script is intended to run on the training host from the poke-bot-agent
root. It does not touch the active process until iteration 0 has checkpointed.
After that boundary it stops the old process, relaxes only the promotion search
guard to permit symmetric 64-sim promotion games, and relaunches the same
lineage with cheaper promotion settings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log_line(path: Path, message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def read_monitor_pid(root: Path, run_name: str) -> int | None:
    state = root / "outputs/runs" / run_name / "monitor_state.json"
    try:
        data = json.loads(state.read_text())
        pid = data.get("pid")
        return int(pid) if pid is not None else None
    except Exception:
        return None


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def process_group(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return None


def clean_tail(path: Path, offset: int) -> tuple[int, list[str]]:
    try:
        size = path.stat().st_size
    except OSError:
        return offset, []
    if size < offset:
        offset = 0
    if size == offset:
        return offset, []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.seek(offset)
        chunk = fh.read()
        offset = fh.tell()
    lines = [
        ANSI_RE.sub("", line).strip()
        for line in chunk.replace("\r", "\n").splitlines()
        if line.strip()
    ]
    return offset, lines


def wait_for_iteration_boundary(
    root: Path,
    run_name: str,
    iteration: int,
    pid: int,
    state_log: Path,
    timeout_s: float,
) -> None:
    log_path = root / "outputs/logs/blackwell.log"
    eval_path = root / "outputs/runs" / run_name / "eval" / f"rr_iter{iteration:03d}.json"
    deadline = time.monotonic() + timeout_s
    offset = log_path.stat().st_size if log_path.exists() else 0
    log_line(state_log, f"WAIT boundary iter={iteration} pid={pid}")
    while time.monotonic() < deadline:
        if not process_alive(pid):
            raise RuntimeError(f"training pid {pid} exited before boundary")
        offset, lines = clean_tail(log_path, offset)
        for line in lines:
            if "[rr] checkpoint" in line:
                log_line(state_log, f"BOUNDARY checkpoint_line={line[-240:]}")
                return
            if (
                f"rr_iter{iteration:03d}" in line
                or "PROMOTED candidate" in line
                or "REJECTED candidate" in line
            ):
                log_line(state_log, f"OBSERVED {line[-240:]}")
        if eval_path.exists():
            log_line(state_log, f"BOUNDARY eval_exists={eval_path}")
            return
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for iter {iteration} checkpoint boundary")


def stop_at_boundary(pid: int, state_log: Path, timeout_s: float) -> None:
    pgid = process_group(pid)
    if pgid is None:
        log_line(state_log, f"OLD already exited pid={pid}")
        return
    log_line(state_log, f"SIGTERM boundary pid={pid} pgid={pgid}")
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not process_alive(pid):
            log_line(state_log, f"OLD exited pid={pid}")
            return
        time.sleep(0.5)
    raise TimeoutError(f"old training pid {pid} did not exit after boundary SIGTERM")


def patch_promotion_guard(root: Path, state_log: Path) -> None:
    path = root / "scripts/train_round_robin.py"
    text = path.read_text(encoding="utf-8")
    old = (
        "    if args.agent_mode == \"belief-mcts\" and (\n"
        "        args.promotion_mcts_sims < 128 or args.promotion_move_time <= 0\n"
        "    ):\n"
        "        print(\n"
        "            \"ERROR: trusted belief-MCTS promotion requires >=128 sims and \"\n"
        "            \"a positive identical move deadline for both agents\",\n"
    )
    new = (
        "    if args.agent_mode == \"belief-mcts\" and (\n"
        "        args.promotion_mcts_sims < 64 or args.promotion_move_time <= 0\n"
        "    ):\n"
        "        print(\n"
        "            \"ERROR: trusted belief-MCTS promotion requires >=64 sims and \"\n"
        "            \"a positive identical move deadline for both agents\",\n"
    )
    if new in text:
        log_line(state_log, "PATCH already_applied promotion guard >=64")
        return
    if old not in text:
        raise RuntimeError("promotion guard snippet not found; refusing blind patch")
    backup = root / "outputs/state" / f"train_round_robin.py.before_promo64.{utc_stamp()}"
    shutil.copy2(path, backup)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "py_compile", str(path)], cwd=root, check=True)
    log_line(state_log, f"PATCH applied promotion guard backup={backup}")


def launch_resume(root: Path, run_name: str, state_log: Path) -> subprocess.Popen[bytes]:
    py = root / "miniconda3/envs/poke-bot-agent/bin/python"
    if not py.exists():
        py = Path("/home/inzi/miniconda3/envs/poke-bot-agent/bin/python")
    log_path = root / "outputs/logs/blackwell.log"
    if log_path.exists():
        archived = root / "outputs/logs" / f"blackwell.pre_promo64_after_iter000.{utc_stamp()}.log"
        shutil.copy2(log_path, archived)
        log_line(state_log, f"ARCHIVE_LOG {archived}")

    cmd = [
        str(py),
        "-u",
        "scripts/launch_blackwell.py",
        "--run-name",
        run_name,
        "--preflight-profile",
        "none",
        "--log",
        "outputs/logs/blackwell.log",
        "--log-threshold-mb",
        "256",
        "--log-keep-mb",
        "16",
        "--monitor-interval",
        "30",
        "--stall-minutes",
        "40",
        "--report-minutes",
        "5",
        "--python",
        str(py),
        "--",
        "--archetype",
        "hammer-pult",
        "--resume",
        "auto",
        "--replay-lineage",
        run_name,
        "--iterations",
        "10000",
        "--games-per-opp",
        "16",
        "--min-games-per-opp",
        "12",
        "--max-games-per-opp",
        "24",
        "--target-search-decisions",
        "3664",
        "--games-per-opp-late",
        "16",
        "--curriculum-switch-iter",
        "0",
        "--workers",
        "20",
        "--no-worker-autotune",
        "--agent-mode",
        "belief-mcts",
        "--mcts-sims",
        "128",
        "--mcts-move-time",
        "12",
        "--game-timeout-s",
        "900",
        "--expected-search-decisions",
        "64",
        "--leaf-eval",
        "gpu-server",
        "--leaf-gpu",
        "cuda:0",
        "--leaf-servers",
        "8",
        "--leaf-max-batch",
        "128",
        "--leaf-queue-depth",
        "128",
        "--leaf-coalesce-ms",
        "2",
        "--sim-device",
        "cpu",
        "--train-epochs",
        "1",
        "--bootstrap-mix",
        "0.25",
        "--history-mix",
        "1",
        "--replay-fraction",
        "0.5",
        "--policy-anchor-ratio",
        "1",
        "--max-decisions-per-game",
        "32",
        "--replay-history-iters",
        "4",
        "--heldout-fraction",
        "0",
        "--promotion-games",
        "40",
        "--promotion-max-games",
        "80",
        "--promotion-batch-games",
        "20",
        "--promotion-min-pairs",
        "20",
        "--promotion-workers",
        "16",
        "--promotion-mcts-sims",
        "64",
        "--promotion-move-time",
        "6",
    ]
    launch_log = root / "outputs/logs/blackwell_boundary_resume.launch.log"
    fh = launch_log.open("ab", buffering=0)
    log_line(state_log, "LAUNCH " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=root,
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(5)
    if proc.poll() is not None:
        fh.close()
        raise RuntimeError(f"launch_blackwell exited early with {proc.returncode}")
    log_line(state_log, f"LAUNCHED launch_pid={proc.pid} launch_log={launch_log}")
    fh.close()
    return proc


def write_budget_state(root: Path, run_name: str) -> None:
    payload = {
        "run_name": run_name,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "scope": "promotion_only_after_iter000_boundary",
        "collection": {
            "mcts_sims": 128,
            "mcts_move_time": 12,
            "unchanged": True,
        },
        "promotion": {
            "promotion_mcts_sims": 64,
            "promotion_move_time": 6,
            "promotion_games": 40,
            "promotion_max_games": 80,
            "promotion_batch_games": 20,
            "promotion_min_pairs": 20,
            "candidate_incumbent_budgets_identical": True,
        },
    }
    path = root / "outputs/state/BLACKWELL_PROMO64_AFTER_ITER000.json"
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/inzi/poke-bot-agent"))
    parser.add_argument("--run-name", default="blackwell_hammer_belief_v3_20260715T175151Z")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--boundary-timeout-minutes", type=float, default=240.0)
    parser.add_argument("--stop-timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    state_log = root / "outputs/state/BLACKWELL_PROMO64_AFTER_ITER000.log"
    state_lock = root / "outputs/state/BLACKWELL_PROMO64_AFTER_ITER000.lock"
    if state_lock.exists():
        raise RuntimeError(f"lock exists: {state_lock}")
    state_lock.write_text(
        f"pid={os.getpid()}\nstarted={datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )
    try:
        pid = args.pid or read_monitor_pid(root, args.run_name)
        if not pid:
            raise RuntimeError("could not determine active training pid")
        wait_for_iteration_boundary(
            root,
            args.run_name,
            args.iteration,
            pid,
            state_log,
            args.boundary_timeout_minutes * 60,
        )
        stop_at_boundary(pid, state_log, args.stop_timeout_seconds)
        patch_promotion_guard(root, state_log)
        write_budget_state(root, args.run_name)
        launch_resume(root, args.run_name, state_log)
        return 0
    except Exception as exc:
        log_line(state_log, f"ERROR {type(exc).__name__}: {exc}")
        return 1
    finally:
        state_lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
