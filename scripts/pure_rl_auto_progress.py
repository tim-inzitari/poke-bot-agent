#!/usr/bin/env python3
"""Durable, explicitly armed Pure-RL phase progressor.

Watches ``CORE_GATE_PASSED`` / ``SPECIALIST_GATE_PASSED`` and advances without
manual intervention. Never kills an in-flight collect: the trainer exits on
gate write; this watcher only waits for that clean exit (or a dead PID) before
warm-start / specialist launch / submit.

Arming
------
Standalone (survives chat agents)::

    nohup $POKEBOT_PYTHON -u scripts/pure_rl_auto_progress.py \\
      --core-run-dir outputs/pure_rl/pure_rl_core_overnight_<UTC> \\
      --archetype <explicit-ladder-selected-specialist> \\
      >> outputs/logs/pure_rl_auto_progress.log 2>&1 &

Or via ``launch_pure_rl.py --auto-progress --specialist-archetype <deck>``.

The watcher deliberately has no default specialist and no implicit submission.
Both choices must be explicit after ranking the current ladder mix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _default_state() -> Path:
    return ROOT / "outputs/pure_rl/auto_progress/state.json"


def _default_log() -> Path:
    return ROOT / "outputs/logs/pure_rl_auto_progress.log"


def _default_lock() -> Path:
    return ROOT / "outputs/pure_rl/auto_progress/auto_progress.lock"


DEFAULT_STATE = _default_state()
DEFAULT_LOG = _default_log()
DEFAULT_LOCK = _default_lock()
DEFAULT_PYTHON = os.environ.get(
    "POKEBOT_PYTHON", "/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"
)
DEFAULT_REMOTES = os.environ.get(
    "PURE_RL_REMOTE_WORKER_ENDPOINTS",
    os.environ.get(
        "POKEBOT_REMOTE_WORKER_ENDPOINTS",
        "192.168.1.143:8765,bert.local:8766",
    ),
)


PHASES = (
    "watching_core",
    "warm_start",
    "launch_specialist",
    "watching_specialist",
    "submit",
    "done",
    "failed",
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--core-run-dir",
        type=Path,
        required=True,
        help="outputs/pure_rl/<core_run_name> being watched for CORE_GATE_PASSED",
    )
    p.add_argument("--specialist-run-name", default=None)
    p.add_argument(
        "--archetype",
        required=True,
        help="Explicit specialist selected from the versioned ladder mix",
    )
    p.add_argument("--python", default=DEFAULT_PYTHON)
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--state-path", type=Path, default=None)
    p.add_argument("--lock-path", type=Path, default=None)
    p.add_argument("--log-path", type=Path, default=None)
    p.add_argument(
        "--remote-worker-endpoints",
        default=DEFAULT_REMOTES,
        help="Forwarded to specialist launch; empty string disables remotes",
    )
    p.add_argument("--no-remote-workers", action="store_true")
    p.add_argument("--iterations", type=int, default=int(os.environ.get("PURE_RL_ITERATIONS", "1000")))
    p.add_argument(
        "--games-per-iter",
        type=int,
        default=int(os.environ.get("PURE_RL_GAMES_PER_ITER", "2048")),
    )
    p.add_argument(
        "--heldout-games",
        type=int,
        default=int(os.environ.get("PURE_RL_HELDOUT_GAMES", "200")),
    )
    p.add_argument(
        "--gate-wr",
        type=float,
        default=float(os.environ.get("PURE_RL_GATE_WR", "0.70")),
    )
    p.add_argument(
        "--submit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After SPECIALIST_GATE_PASSED, package the exact gated checkpoint",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned transitions; do not warm-start/launch/submit",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Single poll cycle then exit (for tests / status checks)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing lock PID if stale or take over",
    )
    return p.parse_args(argv)


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


# Optional override for tests / custom log destinations.
_ACTIVE_LOG_PATH: Path | None = None


def log(msg: str, *, state_path: Path | None = None) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}"
    print(line, flush=True)
    dest = _ACTIVE_LOG_PATH or _default_log()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass
    if state_path is not None:
        # Merge into existing state — never replace the whole document.
        st = load_state(state_path)
        st["last_log"] = line
        st["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_state(state_path, st)


def read_gate(run_dir: Path, name: str) -> dict[str, Any] | None:
    marker = run_dir / name
    if not marker.is_file():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"raw": marker.read_text(encoding="utf-8", errors="replace")}


def _checkpoint_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def resolve_gate_checkpoint(run_dir: Path, gate: dict[str, Any] | None) -> Path | None:
    """Resolve only the exact promoted checkpoint identity recorded by a gate."""
    if not isinstance(gate, dict):
        return None
    raw = gate.get("checkpoint")
    expected = str(gate.get("checkpoint_digest") or "")
    if not raw or not expected.startswith("sha256:"):
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    checkpoint_dir = (run_dir / "checkpoints").resolve()
    try:
        path.relative_to(checkpoint_dir)
    except ValueError:
        return None
    if not path.is_file() or _checkpoint_digest(path) != expected:
        return None
    return path


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pure_rl_trainers_alive(*, run_name: str | None = None) -> list[int]:
    """Return PIDs of live train_pure_rl / launch_pure_rl (optional run filter)."""
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,args="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    hits: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1]
        if "train_pure_rl.py" not in cmd and "launch_pure_rl.py" not in cmd:
            continue
        if "pure_rl_auto_progress" in cmd:
            continue
        if run_name and run_name not in cmd:
            continue
        hits.append(pid)
    return hits


def wait_until_trainers_idle(
    *,
    run_name: str,
    poll_seconds: float,
    state_path: Path,
    timeout_sec: float = 3600.0,
) -> bool:
    """Wait for trainers for ``run_name`` to exit (gate path already breaks loop)."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        alive = pure_rl_trainers_alive(run_name=run_name)
        if not alive:
            return True
        log(
            f"WAIT trainer exit run={run_name} pids={alive} "
            f"(no mid-collect kill; gate already written)",
            state_path=state_path,
        )
        time.sleep(max(5.0, poll_seconds))
    return not pure_rl_trainers_alive(run_name=run_name)


def acquire_lock(lock_path: Path, *, force: bool = False) -> Any:
    """Exclusive flock; returns open file handle to keep locked."""
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        old = fh.read().strip()
        old_pid = None
        try:
            old_pid = int(old.split()[0]) if old else None
        except ValueError:
            old_pid = None
        if force or (old_pid is not None and not pid_alive(old_pid)):
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        else:
            fh.close()
            raise SystemExit(
                f"error: another auto-progress holds {lock_path} (pid={old_pid}); "
                f"pass --force if stale"
            )
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()} {_utc_stamp()}\n")
    fh.flush()
    return fh


def _run(
    cmd: list[str],
    *,
    dry_run: bool,
    state_path: Path,
    env: dict[str, str] | None = None,
) -> int:
    log(f"EXEC {' '.join(cmd)}", state_path=state_path)
    if dry_run:
        return 0
    return int(
        subprocess.call(
            cmd,
            cwd=str(ROOT),
            env=env or os.environ.copy(),
        )
    )


def warm_start_specialist(
    *,
    python: str,
    core_ckpt: Path,
    specialist_run: str,
    archetype: str,
    dry_run: bool,
    state_path: Path,
) -> Path:
    rc = _run(
        [
            python,
            "-u",
            str(ROOT / "scripts/warm_start_pure_rl_specialist.py"),
            "--core-checkpoint",
            str(core_ckpt),
            "--run-name",
            specialist_run,
            "--archetype",
            archetype,
            "--device",
            "cpu",
        ],
        dry_run=dry_run,
        state_path=state_path,
    )
    if rc != 0:
        raise RuntimeError(f"warm_start failed rc={rc}")
    out = (
        ROOT
        / "outputs/pure_rl"
        / specialist_run
        / "checkpoints"
        / f"{archetype}_warmstart.pt"
    )
    if not dry_run and not out.is_file():
        raise RuntimeError(f"missing warm-start checkpoint: {out}")
    return out


def launch_specialist(
    *,
    python: str,
    specialist_run: str,
    warm_ckpt: Path,
    args: argparse.Namespace,
    dry_run: bool,
    state_path: Path,
) -> int | None:
    """Spawn launch_pure_rl specialist detached; return launcher PID."""
    log_path = ROOT / "outputs/logs/pure_rl_specialist.log"
    cmd = [
        python,
        "-u",
        str(ROOT / "scripts/launch_pure_rl.py"),
        "--mode",
        "specialist",
        "--run-name",
        specialist_run,
        "--preflight-profile",
        "none",
        "--log",
        str(log_path),
        # Nested auto-progress OFF — this watcher owns the chain.
        "--no-auto-progress",
    ]
    if args.no_remote_workers or not str(args.remote_worker_endpoints).strip():
        cmd.append("--no-remote-workers")
    else:
        cmd.extend(["--remote-worker-endpoints", str(args.remote_worker_endpoints)])
    cmd.extend(
        [
            "--",
            "--base-checkpoint",
            str(warm_ckpt),
            "--iterations",
            str(args.iterations),
            "--games-per-iter",
            str(args.games_per_iter),
            "--heldout-games",
            str(args.heldout_games),
            "--gate-wr",
            str(args.gate_wr),
            "--specialist-archetype",
            str(args.archetype),
        ]
    )
    log(f"EXEC (detached) {' '.join(cmd)}", state_path=state_path)
    if dry_run:
        return None
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env["POKEBOT_BLACKWELL_STRATEGY_HEADS"] = "0"
    # Detach fully so this watcher can outlive / be restarted independently.
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return int(proc.pid)


def run_submit(
    *, specialist_run: str, archetype: str, dry_run: bool, state_path: Path
) -> int:
    script = ROOT / "scripts/submit_pure_rl_greedy.sh"
    return _run(
        ["bash", str(script), specialist_run, archetype],
        dry_run=dry_run,
        state_path=state_path,
    )


def advance_once(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    """One state-machine step. Idempotent across restarts."""
    core_dir = _resolve(args.core_run_dir)
    state_path = _resolve(args.state_path or _default_state())
    phase = state.get("phase") or "watching_core"
    state.setdefault("core_run_dir", str(core_dir))
    state.setdefault("core_run_name", core_dir.name)
    state["phase"] = phase
    state["pid"] = os.getpid()

    if phase == "done":
        return state
    if phase == "failed":
        return state

    if phase == "watching_core":
        gate = read_gate(core_dir, "CORE_GATE_PASSED")
        if gate is None:
            log(
                f"WATCH core_gate missing run={core_dir.name} "
                f"(poll={args.poll_seconds}s)",
                state_path=state_path,
            )
            return state
        state["core_gate"] = gate
        ckpt = resolve_gate_checkpoint(core_dir, gate)
        if ckpt is None:
            state["phase"] = "failed"
            state["error"] = (
                "CORE_GATE_PASSED lacks an exact in-run checkpoint identity "
                "or its digest does not match"
            )
            log(f"FAIL {state['error']}", state_path=state_path)
            return state
        state["core_checkpoint"] = str(ckpt)
        log(
            f"CORE_GATE_PASSED wr={gate.get('wr')} iter={gate.get('iteration')} "
            f"ckpt={ckpt}",
            state_path=state_path,
        )
        # Do not interrupt; wait for clean trainer exit after gate break.
        if not args.dry_run:
            idle = wait_until_trainers_idle(
                run_name=core_dir.name,
                poll_seconds=args.poll_seconds,
                state_path=state_path,
            )
            if not idle:
                state["phase"] = "failed"
                state["error"] = "timeout waiting for core trainer exit"
                return state
        # Refuse to launch specialist while any other pure-RL trainee still holds GPUs.
        # After core idle, box should be empty; if capacity-bump restart races, wait.
        if not args.dry_run:
            stray = pure_rl_trainers_alive()
            if stray:
                log(
                    f"WAIT box idle before specialist; stray pids={stray}",
                    state_path=state_path,
                )
                deadline = time.time() + 1800.0
                while time.time() < deadline and pure_rl_trainers_alive():
                    time.sleep(max(5.0, args.poll_seconds))
                stray = pure_rl_trainers_alive()
                if stray:
                    state["phase"] = "failed"
                    state["error"] = f"box not idle: {stray}"
                    return state
        state["phase"] = "warm_start"
        save_state(state_path, state)

    if state.get("phase") == "warm_start":
        specialist_run = args.specialist_run_name or state.get("specialist_run_name")
        if not specialist_run:
            label = "".join(
                ch if ch.isalnum() or ch in "-_." else "_"
                for ch in str(args.archetype)
            ).strip("-_.")
            if not label:
                raise RuntimeError("specialist archetype has no safe run-name label")
            specialist_run = f"pure_rl_{label}_overnight_{_utc_stamp()}"
        state["specialist_run_name"] = specialist_run
        ckpt = Path(state["core_checkpoint"])
        try:
            warm = warm_start_specialist(
                python=args.python,
                core_ckpt=ckpt,
                specialist_run=specialist_run,
                archetype=args.archetype,
                dry_run=args.dry_run,
                state_path=state_path,
            )
        except Exception as exc:
            state["phase"] = "failed"
            state["error"] = f"warm_start: {exc}"
            log(f"FAIL {state['error']}", state_path=state_path)
            return state
        state["specialist_checkpoint"] = str(warm)
        state["phase"] = "launch_specialist"
        state["warm_started_at"] = datetime.now(timezone.utc).isoformat()
        save_state(state_path, state)
        log(f"WARM_START ok run={specialist_run} ckpt={warm}", state_path=state_path)

    if state.get("phase") == "launch_specialist":
        specialist_run = str(state["specialist_run_name"])
        warm = Path(state["specialist_checkpoint"])
        # Idempotency: if specialist already running or gate already exists, skip.
        spec_dir = ROOT / "outputs/pure_rl" / specialist_run
        if read_gate(spec_dir, "SPECIALIST_GATE_PASSED") is not None:
            state["phase"] = "watching_specialist"
            save_state(state_path, state)
        elif pure_rl_trainers_alive(run_name=specialist_run):
            state["phase"] = "watching_specialist"
            state["specialist_launch_pid"] = pure_rl_trainers_alive(run_name=specialist_run)[0]
            save_state(state_path, state)
            log(
                f"SPECIALIST already live run={specialist_run}",
                state_path=state_path,
            )
        else:
            try:
                launch_pid = launch_specialist(
                    python=args.python,
                    specialist_run=specialist_run,
                    warm_ckpt=warm,
                    args=args,
                    dry_run=args.dry_run,
                    state_path=state_path,
                )
            except Exception as exc:
                state["phase"] = "failed"
                state["error"] = f"launch_specialist: {exc}"
                log(f"FAIL {state['error']}", state_path=state_path)
                return state
            state["specialist_launch_pid"] = launch_pid
            state["specialist_launched_at"] = datetime.now(timezone.utc).isoformat()
            state["phase"] = "watching_specialist"
            save_state(state_path, state)
            log(
                f"SPECIALIST_LAUNCHED run={specialist_run} launch_pid={launch_pid}",
                state_path=state_path,
            )

    if state.get("phase") == "watching_specialist":
        specialist_run = str(state["specialist_run_name"])
        spec_dir = ROOT / "outputs/pure_rl" / specialist_run
        gate = read_gate(spec_dir, "SPECIALIST_GATE_PASSED")
        if gate is None:
            log(
                f"WATCH specialist_gate missing run={specialist_run}",
                state_path=state_path,
            )
            return state
        gated_checkpoint = resolve_gate_checkpoint(spec_dir, gate)
        if gated_checkpoint is None:
            state["phase"] = "failed"
            state["error"] = (
                "SPECIALIST_GATE_PASSED lacks an exact in-run checkpoint "
                "identity or its digest does not match"
            )
            log(f"FAIL {state['error']}", state_path=state_path)
            return state
        state["specialist_gate"] = gate
        state["specialist_gated_checkpoint"] = str(gated_checkpoint)
        log(
            f"SPECIALIST_GATE_PASSED wr={gate.get('wr')} iter={gate.get('iteration')}",
            state_path=state_path,
        )
        if not args.dry_run:
            wait_until_trainers_idle(
                run_name=specialist_run,
                poll_seconds=args.poll_seconds,
                state_path=state_path,
            )
        state["phase"] = "submit" if args.submit else "done"
        save_state(state_path, state)

    if state.get("phase") == "submit":
        specialist_run = str(state["specialist_run_name"])
        rc = run_submit(
            specialist_run=specialist_run,
            archetype=str(args.archetype),
            dry_run=args.dry_run,
            state_path=state_path,
        )
        state["submit_rc"] = rc
        if rc != 0 and not args.dry_run:
            state["phase"] = "failed"
            state["error"] = f"submit rc={rc}"
            log(f"FAIL {state['error']}", state_path=state_path)
            return state
        state["phase"] = "done"
        state["completed_at"] = datetime.now(timezone.utc).isoformat()
        log(
            f"DONE auto-progress core={state.get('core_run_name')} "
            f"specialist={specialist_run} submit_rc={rc}",
            state_path=state_path,
        )

    return state


def main(argv: list[str] | None = None) -> int:
    global _ACTIVE_LOG_PATH
    args = _parse_args(argv)
    core_dir = _resolve(args.core_run_dir)
    if not core_dir.is_dir():
        print(f"error: core run dir missing: {core_dir}", file=sys.stderr)
        return 2
    trainer_source = (ROOT / "scripts/train_pure_rl.py").read_text(
        encoding="utf-8"
    )
    if '"--specialist-archetype"' not in trainer_source:
        print(
            "error: automatic specialist launch is disabled until "
            "train_pure_rl.py consumes --specialist-archetype end-to-end",
            file=sys.stderr,
        )
        return 2
    # Default coordination files belong to this exact run. A global state/lock
    # lets one completed or stale watcher silently rebind another lineage.
    state_path = (
        _resolve(args.state_path)
        if args.state_path is not None
        else core_dir / "auto_progress" / "state.json"
    )
    lock_path = (
        _resolve(args.lock_path)
        if args.lock_path is not None
        else core_dir / "auto_progress" / "auto_progress.lock"
    )
    args.state_path = state_path
    args.lock_path = lock_path
    _ACTIVE_LOG_PATH = _resolve(args.log_path or _default_log())

    lock_fh = acquire_lock(lock_path, force=args.force)
    stop = {"flag": False}

    def _stop(_signum: int, _frame: object) -> None:
        stop["flag"] = True
        log("SIGNAL stop requested", state_path=state_path)

    prev = {sig: signal.signal(sig, _stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        state = load_state(state_path)
        # Re-bind core dir if state empty or matches.
        if state.get("core_run_dir") and Path(state["core_run_dir"]).resolve() != core_dir.resolve():
            if state.get("phase") not in (None, "watching_core", "done", "failed", ""):
                print(
                    f"error: state at {state_path} tracks different core "
                    f"{state.get('core_run_dir')}; pass a fresh --state-path or --force",
                    file=sys.stderr,
                )
                return 2
        state.setdefault("phase", "watching_core")
        state["core_run_dir"] = str(core_dir)
        state["core_run_name"] = core_dir.name
        state["armed_at"] = state.get("armed_at") or datetime.now(timezone.utc).isoformat()
        save_state(state_path, state)
        log(
            f"ARMED auto-progress core={core_dir.name} phase={state['phase']} "
            f"dry_run={args.dry_run} submit={args.submit}",
            state_path=state_path,
        )

        while not stop["flag"]:
            state = advance_once(args, state)
            # Preserve fields log() may have written (last_log/updated_at).
            disk = load_state(state_path)
            for key in ("last_log", "updated_at"):
                if key in disk:
                    state[key] = disk[key]
            save_state(state_path, state)
            if state.get("phase") in ("done", "failed"):
                return 0 if state.get("phase") == "done" else 1
            if args.once:
                return 0
            time.sleep(max(1.0, float(args.poll_seconds)))
        return 0
    finally:
        for sig, handler in prev.items():
            signal.signal(sig, handler)
        try:
            lock_fh.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
