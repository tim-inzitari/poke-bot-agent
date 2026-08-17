#!/usr/bin/env python3
"""Launch one full-hardware pure-RL trainee (core or specialist) + monitor.

Saturates CPU workers and dual-GPU leaf servers for a single active lineage.
Refuses to start when two GPUs are visible but leaf replicas omit GPU0 or GPU1
unless ``PURE_RL_ALLOW_SINGLE_GPU=1``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Immutable source snapshots are code-only.  A managed candidate can bind its
# generated artifacts to a separately attested external root with
# ``POKEBOT_OUTPUTS_DIR``; the legacy source-root layout remains the default
# for every ordinary invocation.
_outputs_override = os.environ.get("POKEBOT_OUTPUTS_DIR", "").strip()
OUTPUTS_ROOT = (
    Path(_outputs_override).expanduser().resolve()
    if _outputs_override
    else ROOT / "outputs"
)
# Preserve checkout-relative artifact paths for ordinary runs, but never let a
# sealed-source invocation write a relative log/arm file inside the snapshot.
RELATIVE_ARTIFACT_ROOT = OUTPUTS_ROOT if _outputs_override else ROOT
DEFAULT_LOG = OUTPUTS_ROOT / "logs/pure_rl.log"
TRAINING_SAFETY_VERSION = "20260717"
LAUNCH_LOCK = OUTPUTS_ROOT / "state/pure_rl_launcher.lock"
DEFAULT_TRAINING_ARM_FILE = OUTPUTS_ROOT / "state/TRAINING_ARMED"


def _r241_snapshot_execution_active(environment: dict[str, str] | None = None) -> bool:
    """Whether this generic launcher is running inside r241's sealed source tree."""

    env = os.environ if environment is None else environment
    return bool(str(env.get("POKEBOT_R241_SOURCE_EXECUTION_ROOT") or "").strip())


def _validate_r241_snapshot_subprocess_closure(
    environment: dict[str, str] | None = None,
) -> None:
    """Reject an incomplete source snapshot before it can skip safety helpers.

    The generic launcher historically treated optional helper scripts as best
    effort.  That is fine for ordinary checkout development, but r241's
    receipt-bound execution closure must not silently omit its canary, live
    resource watcher, or unattended monitor.
    """

    env = os.environ if environment is None else environment
    if not _r241_snapshot_execution_active(env):
        return
    declared_root = Path(str(env["POKEBOT_R241_SOURCE_EXECUTION_ROOT"])).expanduser()
    if declared_root.resolve() != ROOT.resolve():
        raise RuntimeError(
            "r241 source execution root disagrees with launch_pure_rl.py location"
        )
    for relative in (
        "scripts/canary_game_accuracy.py",
        "scripts/resource_watcher.py",
        "scripts/unattended_monitor.py",
    ):
        script = ROOT / relative
        if script.is_symlink() or not script.is_file():
            raise RuntimeError(
                "r241 immutable source snapshot omits required subprocess helper: "
                + relative
            )


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _preflight_environment(
    source: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return a test environment isolated from live trainer tuning knobs."""
    clean = dict(os.environ if source is None else source)
    for key in tuple(clean):
        if key.startswith(("PURE_RL_", "POKEBOT_")):
            clean.pop(key, None)
    return clean


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default=None)
    p.add_argument("--mode", choices=("core", "specialist"), default="core")
    p.add_argument("--log", type=Path, default=Path(os.environ.get("POKEBOT_PURE_RL_LOG", DEFAULT_LOG)))
    p.add_argument("--python", default=os.environ.get("POKEBOT_PYTHON", sys.executable))
    p.add_argument("--preflight-profile", choices=("canary", "quick", "none"), default="quick")
    p.add_argument("--stall-minutes", type=float, default=20.0)
    p.add_argument("--oom-limit", type=int, default=2)
    p.add_argument("--report-minutes", type=float, default=5.0)
    p.add_argument("--monitor-interval", type=float, default=30.0)
    p.add_argument("--log-threshold-mb", type=float, default=_env_float("POKEBOT_LOG_THRESHOLD_MB", 256.0))
    p.add_argument("--log-keep-mb", type=float, default=_env_float("POKEBOT_LOG_KEEP_MB", 16.0))
    p.add_argument("--allow-single-gpu", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--multi-env-per-worker",
        type=int,
        default=None,
        help=(
            "Forward to train_pure_rl: LibcgMultiEnv battles per OS worker. "
            "Also honour POKEBOT_MULTI_ENV=1 in the child env."
        ),
    )
    p.add_argument(
        "--leaf-coalesce-ms",
        type=float,
        default=None,
        help="Forward to train_pure_rl (default via PURE_RL_LEAF_COALESCE_MS=0).",
    )
    p.add_argument(
        "--remote-worker-endpoints",
        default=None,
        help=(
            "Whole-game farms (comma-separated). Default production: "
            "elmo:8765,bert.local:8766. Empty string disables."
        ),
    )
    p.add_argument(
        "--no-remote-workers",
        action="store_true",
        help="Disable Elmo/bert whole-game farms",
    )
    p.add_argument(
        "--no-resource-watcher",
        action="store_true",
        help="Do not spawn resource_watcher --emit-live-pool (auto-rebalance).",
    )
    p.add_argument(
        "--auto-progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Explicitly arm the core→specialist watcher. Default OFF; the "
            "specialist must be selected from a versioned ladder mix."
        ),
    )
    p.add_argument(
        "--specialist-archetype",
        default=None,
        help="Required with --auto-progress; never inferred or hard-coded",
    )
    p.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Args after '--' forwarded to train_pure_rl.py",
    )
    args = p.parse_args(argv)
    if args.train_args[:1] == ["--"]:
        args.train_args = args.train_args[1:]
    if args.run_name is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_name = f"pure_rl_{args.mode}_{stamp}"
    if args.auto_progress and not str(args.specialist_archetype or "").strip():
        p.error("--auto-progress requires --specialist-archetype")
    if args.auto_progress:
        trainer_source = (ROOT / "scripts/train_pure_rl.py").read_text(
            encoding="utf-8"
        )
        if '"--specialist-archetype"' not in trainer_source:
            p.error(
                "--auto-progress is unavailable until train_pure_rl.py "
                "consumes --specialist-archetype end-to-end"
            )
    return args


def open_stable_log(path: Path):
    """Open the stable watch path append-only across launcher restarts."""
    path = path if path.is_absolute() else RELATIVE_ARTIFACT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        path.unlink()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    return os.fdopen(fd, "wb", buffering=0)


def progress_log_path(log_path: Path) -> Path:
    """Sibling file for tqdm bars (stderr), e.g. pure_rl_core.progress.log."""
    log_path = (
        log_path if log_path.is_absolute() else RELATIVE_ARTIFACT_ROOT / log_path
    )
    return log_path.with_name(f"{log_path.stem}.progress.log")


def progress_status_path(log_path: Path) -> Path:
    """Single-line status rewritten each bar tick (in-place watcher)."""
    prog = progress_log_path(log_path)
    return prog.with_name(prog.name.replace(".progress.log", ".progress.status"))


def publish_stable_log_aliases(log_path: Path) -> None:
    """Atomically point the permanent event/tqdm tails at this run.

    Per-run files remain authoritative archives.  These three aliases are the
    stable operator contract across run names and launcher versions.
    """
    log_path = (
        log_path if log_path.is_absolute() else RELATIVE_ARTIFACT_ROOT / log_path
    )
    progress_path = progress_log_path(log_path)
    status_path = progress_status_path(log_path)
    aliases = {
        log_path.parent / "training.log": log_path,
        log_path.parent / "training.progress.log": progress_path,
        log_path.parent / "training.progress.status": status_path,
    }
    for alias, target in aliases.items():
        if alias == target:
            continue
        alias.parent.mkdir(parents=True, exist_ok=True)
        temporary = alias.with_name(f".{alias.name}.link-{os.getpid()}")
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        temporary.symlink_to(os.path.relpath(target, alias.parent))
        os.replace(temporary, alias)


def _normalized_child_returncode(returncode: int) -> int:
    """Map subprocess signal returns to conventional shell exit statuses."""
    code = int(returncode)
    return 128 + abs(code) if code < 0 else code


def _monitor_requested_this_attempt_stop(
    alert_path: Path,
    *,
    training_pid: int,
    attempt_started_at: float,
) -> bool:
    """Return true only for a monitor stop belonging to this exact child.

    The unattended monitor terminates an unsafe or genuinely stalled trainer
    with SIGTERM.  Returning the conventional 143 from the launcher makes a
    ``Restart=on-failure`` unit immediately recollect the same iteration.  A
    PID and timestamp match lets the launcher instead return the unit's
    restart-prevent status (75), while stale alerts and ordinary/manual stops
    retain their normal semantics.
    """
    try:
        payload = json.loads(alert_path.read_text(encoding="utf-8"))
        return (
            int(payload.get("pid", -1)) == int(training_pid)
            and float(payload.get("timestamp", 0.0)) >= float(attempt_started_at)
            and bool(str(payload.get("reason", "")).strip())
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _production_training_arm(
    source: dict[str, str] | None = None,
) -> tuple[bool, Path]:
    """Require independent environment and filesystem production consent."""
    env = os.environ if source is None else source
    configured = env.get(
        "POKEBOT_TRAINING_ARM_FILE",
        str(DEFAULT_TRAINING_ARM_FILE),
    )
    arm_file = Path(configured)
    if not arm_file.is_absolute():
        arm_file = RELATIVE_ARTIFACT_ROOT / arm_file
    try:
        token_matches = arm_file.read_bytes() == TRAINING_SAFETY_VERSION.encode()
    except OSError:
        token_matches = False
    return (
        env.get("POKEBOT_TRAINING_SAFETY_VERSION", "")
        == TRAINING_SAFETY_VERSION
        and token_matches,
        arm_file,
    )


def _acquire_launch_lock(path: Path, *, run_name: str):
    """Hold the one-full-hardware-trainer-per-checkout interlock.

    Detached recovery scripts previously raced systemd and could start another
    launcher while the first trainer was still alive.  ``flock`` is released
    automatically if the launcher crashes or is killed, so a stale text record
    can never keep the machine wedged.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.seek(0)
        owner = stream.read().strip()
        stream.close()
        return None, owner
    stream.seek(0)
    stream.truncate()
    stream.write(
        f"pid={os.getpid()} run={run_name} "
        f"started={datetime.now(timezone.utc).isoformat()}\n"
    )
    stream.flush()
    return stream, ""


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        _validate_r241_snapshot_subprocess_closure()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 78
    production_armed, arm_file = _production_training_arm()
    if not args.smoke and not production_armed:
        print(
            "error: production training requires both memory-safety env "
            f"version {TRAINING_SAFETY_VERSION} and exact token file "
            f"{arm_file}; refusing to spawn workers",
            file=sys.stderr,
        )
        return 78
    launch_lock, lock_owner = _acquire_launch_lock(
        LAUNCH_LOCK,
        run_name=str(args.run_name),
    )
    if launch_lock is None:
        print(
            "error: another full-hardware trainer launcher owns "
            f"{LAUNCH_LOCK} ({lock_owner or 'owner record unavailable'}); "
            "refusing an overlapping start",
            file=sys.stderr,
        )
        return 75
    sys.path.insert(0, str(ROOT))
    from poke_bot.pure_rl.hardware import full_hardware_profile
    from dataclasses import replace

    # Safe no-swap steady default for this box: measured per-worker RSS
    # (~1.3 GiB) x higher worker counts (96/160) does not fit in 124 GiB
    # RAM alongside ~60 leaf servers + parent without swapping. 48 keeps
    # MemAvailable comfortably >20 GiB with 60 leaves. Operators can still
    # override via PURE_RL_SIM_WORKERS in the shell env before launch.
    os.environ.setdefault("PURE_RL_SIM_WORKERS", "48")
    hw = full_hardware_profile()
    if args.allow_single_gpu or args.smoke:
        hw = replace(hw, allow_single_gpu=True)
    try:
        import torch

        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        visible = 0
    if args.smoke:
        visible = max(visible, 1)
    try:
        hw.validate_or_raise(visible_gpu_count=visible if visible else (1 if hw.allow_single_gpu else 0))
    except ValueError as exc:
        # If no CUDA in this environment, require explicit smoke/single-gpu.
        if visible < 2 and not hw.allow_single_gpu:
            print(
                f"error: {exc}; pass --allow-single-gpu or --smoke on non-dual-GPU hosts",
                file=sys.stderr,
            )
            return 2
        if not hw.allow_single_gpu:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Both GPUs visible so leaf servers can bind 0 and 1; train pins device 1.
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1")
    env["POKEBOT_BLACKWELL_STRATEGY_HEADS"] = "0"
    env["POKEBOT_WORKER_CPU_ONLY"] = "1"
    env["PURE_RL_SIM_WORKERS"] = str(hw.sim_workers)
    env["PURE_RL_LEAF_GPU0_REPLICAS"] = str(hw.leaf_gpu0_replicas)
    env["PURE_RL_LEAF_GPU1_REPLICAS"] = str(hw.leaf_gpu1_replicas)
    env["PURE_RL_TORCH_THREADS"] = str(hw.torch_threads)
    # Tiny ~1.6M pure-RL policy: coalesce≈0 beats the RR Hope-large default (4ms).
    # Do not set LEAF_SERVER_COALESCE_MS globally here if already exported (ops override).
    env.setdefault("PURE_RL_LEAF_COALESCE_MS", "0")
    # MultiEnv (N>1) saves RAM but cuts OS workers (procs=sim//N) and starves
    # GPU leaves on this box — measured ~1.5g/s vs ~3.8g/s at N=1. Opt-in only.
    env.setdefault("POKEBOT_MULTI_ENV", "0")
    env.setdefault("POKEBOT_MULTI_ENV_PER_WORKER", "1")
    env.setdefault("PURE_RL_MULTI_ENV", "0")
    env.setdefault("PURE_RL_MULTI_ENV_PER_WORKER", "1")
    # Live pool auto-rebalance ON unless explicitly disabled.
    env.setdefault("POKEBOT_LIVE_POOL", "1")
    # Local-primary soft floor for additive self-play RR split.
    env.setdefault("PURE_RL_REBALANCE_MIN_LOCAL_FRAC", "0.40")
    # Public mix (baseline vs public/roster opponents) finishes fast locally
    # — route it local-only by default so remotes stay free for self-play.
    # Set 0/false to fall back to PURE_RL_PUBLIC_MIX_MIN_LOCAL_FRAC instead.
    env.setdefault("PURE_RL_PUBLIC_MIX_LOCAL_ONLY", "1")
    env.setdefault("PURE_RL_PUBLIC_MIX_MIN_LOCAL_FRAC", "0.95")
    # No-swap RAM cap for THIS box: local worker ceiling == the safe steady
    # target (48), not headroom above it — measured ~1.3 GiB/worker plus ~60
    # leaf servers at ~0.33 GiB each does not fit 96/160 workers in 124 GiB
    # RAM without swapping. Remotes (Elmo/bert) stay additive on top; total
    # max 10000 ≫ peak local+remote (never binding).
    env.setdefault("PURE_RL_REBALANCE_MAX_WORKERS", "48")
    env.setdefault("PURE_RL_REBALANCE_MAX_TOTAL_WORKERS", "10000")
    env.setdefault("PURE_RL_REBALANCE_MIN_REMOTE_FRAC", "0.25")
    env.setdefault("POKEBOT_LIVE_POOL_MAX_WORKERS", "48")
    env.setdefault("POKEBOT_LIVE_POOL_MAX_LEAF_GPU0", "12")
    env.setdefault("POKEBOT_LIVE_POOL_MAX_LEAF_GPU1", "48")
    env.setdefault("POKEBOT_LIVE_POOL_MAX_LEAF_SERVERS", "60")
    # Per-worker RSS budget used by the RAM-fit clamp (apply_live_pool_plan /
    # resource_watcher); measured reality on this box, not the old 0.8 guess.
    env.setdefault("PURE_RL_PER_WORKER_RSS_GB", "1.3")
    if not args.smoke and not args.no_remote_workers:
        # Configured production farms are part of the execution contract. A
        # missing endpoint or exhausted remote retry must stop the wave instead
        # of silently shifting the advertised remote work onto Inzi.
        env.setdefault("POKEBOT_REMOTE_REQUIRE_ALL", "1")
        env.setdefault("POKEBOT_REMOTE_NO_LOCAL_FALLBACK", "1")
    # Keep \\r tqdm bars on stderr → *.progress.log (+ *.progress.status mirror).
    env.setdefault("PURE_RL_TQDM_INPLACE", "1")
    if hw.allow_single_gpu:
        env["PURE_RL_ALLOW_SINGLE_GPU"] = "1"

    if args.preflight_profile != "none" and not args.smoke:
        preflight = subprocess.run(
            [
                args.python,
                str(ROOT / "scripts/run_test_profile.py"),
                args.preflight_profile,
                "--python",
                args.python,
            ],
            cwd=ROOT,
            env=_preflight_environment(),
            check=False,
        )
        if preflight.returncode != 0:
            print(
                f"error: {args.preflight_profile} preflight failed",
                file=sys.stderr,
            )
            return preflight.returncode

    # Live multi-env game accuracy (fail-closed) before saturating the box.
    accuracy_script = ROOT / "scripts/canary_game_accuracy.py"
    skip_acc = str(env.get("POKEBOT_SKIP_GAME_ACCURACY", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if accuracy_script.is_file() and not args.smoke and not skip_acc:
        accuracy_command = [
            args.python,
            str(accuracy_script),
            "--num-envs",
            env.get("POKEBOT_MULTI_ENV_PER_WORKER", "1"),
            "--json-out",
            str(OUTPUTS_ROOT / "state/game_accuracy_canary.json"),
        ]
        if _r241_snapshot_execution_active(env):
            # The immutable source snapshot intentionally does not contain a
            # Kaggle input tree.  Bind the canary to r241's sealed libcg and
            # exact deck rather than letting its checkout-relative defaults
            # read (or write) beneath the snapshot.
            cg_root = str(env.get("CG_LIB_PATH") or "").strip()
            deck = str(env.get("POKEBOT_SPECIALIST_DECK_PATH") or "").strip()
            if not cg_root or not deck:
                print(
                    "error: r241 canary is missing sealed CG_LIB_PATH or specialist deck",
                    file=sys.stderr,
                )
                return 78
            accuracy_command.extend(["--cg-parent", cg_root, "--deck-csv", deck])
        acc = subprocess.run(
            accuracy_command,
            cwd=ROOT,
            env=env,
            check=False,
        )
        if acc.returncode != 0:
            print(
                "error: game accuracy canary failed "
                "(set POKEBOT_SKIP_GAME_ACCURACY=1 to override)",
                file=sys.stderr,
            )
            return acc.returncode or 2

    train_cmd = [
        args.python,
        "-u",
        str(ROOT / "scripts/train_pure_rl.py"),
        "--run-name",
        args.run_name,
        "--mode",
        args.mode,
        *args.train_args,
    ]
    if args.smoke and "--smoke" not in train_cmd:
        train_cmd.append("--smoke")
    if args.allow_single_gpu and "--allow-single-gpu" not in train_cmd:
        train_cmd.append("--allow-single-gpu")
    # Resolve multi-env for the train argv so logs show the knobs even when
    # only env defaults were set (next-iter redeploy path).
    from poke_bot.pure_rl.multi_env_self_play import (
        pure_rl_leaf_coalesce_ms,
        resolve_multi_env_per_worker,
    )

    # Make launch setdefaults visible to resolve_* helpers.
    for k in (
        "POKEBOT_MULTI_ENV",
        "POKEBOT_MULTI_ENV_PER_WORKER",
        "PURE_RL_MULTI_ENV",
        "PURE_RL_MULTI_ENV_PER_WORKER",
        "PURE_RL_LEAF_COALESCE_MS",
    ):
        if k in env:
            os.environ[k] = env[k]
    multi_n = resolve_multi_env_per_worker(
        args.multi_env_per_worker,
        default_when_enabled=4,
    )
    if not any(
        a == "--multi-env-per-worker" or a.startswith("--multi-env-per-worker=")
        for a in train_cmd
    ):
        train_cmd.extend(["--multi-env-per-worker", str(multi_n)])
    coalesce_ms = (
        float(args.leaf_coalesce_ms)
        if args.leaf_coalesce_ms is not None
        else pure_rl_leaf_coalesce_ms(default=0.0)
    )
    if not any(
        a == "--leaf-coalesce-ms" or a.startswith("--leaf-coalesce-ms=")
        for a in train_cmd
    ):
        train_cmd.extend(["--leaf-coalesce-ms", str(coalesce_ms)])
    # Production: remotes ON by default (canary/smoke skips).
    has_remote_flag = any(
        a == "--remote-worker-endpoints" or a.startswith("--remote-worker-endpoints=")
        for a in train_cmd
    )
    if args.no_remote_workers and "--no-remote-workers" not in train_cmd:
        train_cmd.append("--no-remote-workers")
    elif (
        not args.smoke
        and not args.no_remote_workers
        and not has_remote_flag
    ):
        endpoints = args.remote_worker_endpoints
        if endpoints is None:
            endpoints = os.environ.get(
                "PURE_RL_REMOTE_WORKER_ENDPOINTS",
                os.environ.get(
                    "POKEBOT_REMOTE_WORKER_ENDPOINTS",
                    "elmo:8765,bert.local:8766",
                ),
            )
        if str(endpoints).strip():
            train_cmd.extend(["--remote-worker-endpoints", str(endpoints)])

    log_path = (
        args.log if args.log.is_absolute() else RELATIVE_ARTIFACT_ROOT / args.log
    ).resolve()
    prog_path = progress_log_path(log_path).resolve()
    status_path = progress_status_path(log_path).resolve()
    env["PURE_RL_PROGRESS_LOG"] = str(prog_path)
    publish_stable_log_aliases(log_path)
    log_stream = open_stable_log(log_path)
    # tqdm bars (\\r in-place) on stderr → progress.log; status file for watchers.
    progress_stream = open_stable_log(prog_path)
    # The event log is append-only across systemd restarts. Capture the exact
    # boundary before this training attempt starts so its monitor sees every new
    # line but never replays a prior attempt's FAIL-CLOSED marker. Replaying the
    # old tail caused a one-minute SIGTERM/restart loop in production.
    monitor_log_start_offset = log_path.stat().st_size
    print(
        f"PURE_RL_RUN name={args.run_name} mode={args.mode} "
        f"workers={hw.sim_workers} leaves0={hw.leaf_gpu0_replicas} "
        f"leaves1={hw.leaf_gpu1_replicas} "
        f"multi_env={multi_n} leaf_coalesce_ms={coalesce_ms} "
        f"log={log_path} progress_log={prog_path} progress_status={status_path}",
        flush=True,
    )
    print(
        f"PURE_RL_FOLLOW event_log: tail -F {log_path} | "
        f"tqdm_games: bash scripts/watch_pure_rl_progress.sh "
        f"(or: watch -n1 cat {status_path}; less -r +F {prog_path})",
        flush=True,
    )
    attempt_started_at = time.time()
    training = subprocess.Popen(
        train_cmd,
        cwd=ROOT,
        env=env,
        stdout=log_stream,
        stderr=progress_stream,
        start_new_session=True,
    )
    # Fail-loud if a future edit re-merges stderr into the event log.
    try:
        err_target = Path(os.readlink(f"/proc/{training.pid}/fd/2")).resolve()
    except OSError:
        err_target = Path("?")
    print(
        f"PURE_RL_PROGRESS_SPLIT stdout={log_path} stderr={prog_path} "
        f"child_fd2={err_target} status={status_path}",
        flush=True,
    )
    if err_target != prog_path:
        print(
            "warning: train stderr is not progress.log — tqdm/sps will not "
            "land in the progress view; check launch wiring",
            flush=True,
        )

    watcher = None
    watcher_script = ROOT / "scripts/resource_watcher.py"
    if (
        watcher_script.is_file()
        and not args.smoke
        and not args.no_resource_watcher
        and str(env.get("POKEBOT_LIVE_POOL", "1")).strip().lower()
        not in ("0", "false", "no", "off")
    ):
        watcher_log = OUTPUTS_ROOT / "logs/resource_watcher.log"
        watcher_log.parent.mkdir(parents=True, exist_ok=True)
        (OUTPUTS_ROOT / "state").mkdir(parents=True, exist_ok=True)
        watcher = subprocess.Popen(
            [
                args.python,
                "-u",
                str(watcher_script),
                "--interval",
                "30",
                "--emit-live-pool",
                "--log",
                str(watcher_log),
                "--plan",
                str(OUTPUTS_ROOT / "state" / "resource_plan.json"),
                "--live-pool-plan",
                str(OUTPUTS_ROOT / "state" / "live_pool_plan.json"),
            ],
            cwd=ROOT,
            env=env,
            start_new_session=True,
        )
        print(f"PURE_RL_WATCHER pid={watcher.pid} emit_live_pool=1", flush=True)

    run_dir = OUTPUTS_ROOT / "pure_rl" / args.run_name
    monitor_alert_path = run_dir / "MONITOR_STOP_REQUESTED.json"
    monitor = None
    monitor_script = ROOT / "scripts/unattended_monitor.py"
    if monitor_script.is_file() and not args.smoke:
        monitor_cmd = [
            args.python,
            "-u",
            str(monitor_script),
            "--pid",
            str(training.pid),
            "--log",
            str(log_path),
            "--run-dir",
            str(run_dir),
            "--start-offset",
            str(monitor_log_start_offset),
            "--interval",
            str(args.monitor_interval),
            "--stall-minutes",
            str(args.stall_minutes),
            "--oom-limit",
            str(args.oom_limit),
            "--report-minutes",
            str(args.report_minutes),
            "--log-threshold-mb",
            str(args.log_threshold_mb),
            "--log-keep-mb",
            str(args.log_keep_mb),
            "--process-group",
            "--progress-status",
            str(status_path),
            "--progress-log",
            str(prog_path),
        ]
        monitor = subprocess.Popen(monitor_cmd, cwd=ROOT, env=env, start_new_session=True)

    auto_progress = None
    want_auto = args.auto_progress
    auto_script = ROOT / "scripts/pure_rl_auto_progress.py"
    if want_auto and auto_script.is_file():
        run_dir = OUTPUTS_ROOT / "pure_rl" / args.run_name
        auto_log = OUTPUTS_ROOT / "logs/pure_rl_auto_progress.log"
        auto_cmd = [
            args.python,
            "-u",
            str(auto_script),
            "--core-run-dir",
            str(run_dir),
            "--python",
            args.python,
            "--archetype",
            str(args.specialist_archetype),
        ]
        if args.no_remote_workers:
            auto_cmd.append("--no-remote-workers")
        elif args.remote_worker_endpoints is not None:
            auto_cmd.extend(
                ["--remote-worker-endpoints", str(args.remote_worker_endpoints)]
            )
        auto_progress = subprocess.Popen(
            auto_cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(
            f"PURE_RL_AUTO_PROGRESS pid={auto_progress.pid} "
            f"state={run_dir / 'auto_progress/state.json'} log={auto_log}",
            flush=True,
        )

    pid_bits = [f"training={training.pid}"]
    if monitor is not None:
        pid_bits.append(f"monitor={monitor.pid}")
    if watcher is not None:
        pid_bits.append(f"watcher={watcher.pid}")
    if auto_progress is not None:
        pid_bits.append(f"auto_progress={auto_progress.pid}")
    print(f"PURE_RL_PIDS {' '.join(pid_bits)}", flush=True)

    def _stop(_signum: int, _frame: object) -> None:
        try:
            os.killpg(training.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if monitor and monitor.poll() is None:
            monitor.terminate()
        if watcher and watcher.poll() is None:
            watcher.terminate()
        # Do NOT terminate auto_progress — it owns phase transitions after gate.

    prev = {sig: signal.signal(sig, _stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        child_status = _normalized_child_returncode(training.wait())
        if child_status == 143 and _monitor_requested_this_attempt_stop(
            monitor_alert_path,
            training_pid=training.pid,
            attempt_started_at=attempt_started_at,
        ):
            print(
                "MONITOR_STOP_CONFIRMED restart_prevent_status=75 "
                f"training_pid={training.pid}",
                flush=True,
            )
            return 75
        return child_status
    finally:
        log_stream.close()
        progress_stream.close()
        for proc in (monitor, watcher):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for sig, handler in prev.items():
            signal.signal(sig, handler)
        launch_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
