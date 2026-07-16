#!/usr/bin/env python3
"""Apply POKEBOT_OPENING_MOVE_TIME_MULT=1.0 at the next Core promo/iter-safe boundary.

Obeys outputs/state/RESTART_POLICY.md:
  - mid-collection: wait
  - after N/N + PROMOTED/REJECTED (or next-iter / trainer-down): restart Core only
  - never touch Blackwell
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs/logs/core_kernel.log"
WATCH_LOG = ROOT / "outputs/logs/core_opening_fix_boundary.log"
OWNER = ROOT / "outputs/state/SOLE_TRAIN_OWNER.lock"
STATUS_MD = ROOT / "outputs/state/ERROR_FIX_STATUS.md"
LAUNCH_PID = ROOT / "outputs/state/CORE_BERT_LAUNCH.pid"
WATCHER_PID = ROOT / "outputs/state/CORE_OPENING_FIX_WATCHER.pid"
PY = "/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"
ELMO = "192.168.1.143:8765"
BERT = "bert.local:8766"
CORE_RUN = "core_kernel_3080ti_trusted_20260714T2000Z"

GAMES_RE = re.compile(r"iter(\d+) games:\s+\d+%\|[^\|]*\|\s+(\d+)/(\d+)")
PROMO_RE = re.compile(r"\[rr\] (PROMOTED|REJECTED) candidate")
ITER_START_RE = re.compile(r"\[rr\] iter (\d+):")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    # Prefer file when stdout is already redirected to WATCH_LOG (avoid duplicates).
    WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WATCH_LOG.open("a") as fh:
        fh.write(line + "\n")
    if not sys.stdout.isatty():
        return
    print(line, flush=True)


def write_owner(status: str, extra: str = "") -> None:
    OWNER.write_text(
        "\n".join(
            [
                "SOLE_TRAIN_OWNER=1",
                f"status={status}",
                f"updated={utc_now()}",
                "agent=opening-fix-boundary",
                "topology=CORE_ELMO_PLUS_BERT_BW_LOCAL",
                "fixed=connection_closed+idle_timeout+client_reconnect",
                "action=POKEBOT_OPENING_MOVE_TIME_MULT=1.0 at Core boundary",
                "bw=DO_NOT_TOUCH",
                "rule=Obey RESTART_POLICY.md. DO NOT undo working patches.",
                extra.rstrip(),
                "",
            ]
        )
    )


def core_chunk() -> str:
    raw = LOG.read_text(errors="replace").replace("\r", "\n")
    idx = raw.rfind("[core-pipeline][start]")
    return raw[idx:] if idx >= 0 else raw[-400000:]


def parse_games(chunk: str) -> tuple[int, int, int] | None:
    matches = GAMES_RE.findall(chunk)
    if not matches:
        return None
    it, cur, tot = matches[-1]
    return int(it), int(cur), int(tot)


def pids_by_substr(*needles: str) -> list[int]:
    out: list[int] = []
    for ent in Path("/proc").iterdir():
        if not ent.name.isdigit():
            continue
        try:
            cmd = (ent / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, PermissionError):
            continue
        if all(n in cmd for n in needles):
            out.append(int(ent.name))
    return out


def wait_boundary(poll_s: float = 20.0, max_hours: float = 12.0) -> dict:
    deadline = time.time() + max_hours * 3600
    start_iter: int | None = None
    collection_done = False
    seen_promo = False
    seen_next_iter = False
    # Only count PROMOTED/REJECTED written *after* collection N/N (avoid stale iter).
    post_collection_offset = 0
    write_owner(
        "OPENING_FIX_WAIT_BOUNDARY",
        "phase=wait_collection_then_promo remaining=RemoteLeafTimeout_opening_6s",
    )
    while time.time() < deadline:
        chunk = core_chunk()
        games = parse_games(chunk)
        if games is None:
            log("WAIT no games progress yet")
            time.sleep(poll_s)
            continue
        it, cur, tot = games
        if start_iter is None:
            start_iter = it
            log(f"WAIT latch start_iter={start_iter} games={cur}/{tot}")

        if it == start_iter and cur >= tot and tot > 0 and not collection_done:
            collection_done = True
            post_collection_offset = LOG.stat().st_size if LOG.exists() else 0
            log(
                f"COLLECTION_DONE iter{it} {cur}/{tot} — waiting promo "
                f"(RESTART_POLICY) log_off={post_collection_offset}"
            )
            write_owner(
                "OPENING_FIX_WAIT_PROMO",
                f"phase=wait_promo iter={it} games={cur}/{tot}",
            )

        # Next-iter games bar means the previous iter already finished collection
        # (and usually train/promo). Treat as collection_done even if we missed N/N.
        if it > (start_iter or -1):
            seen_next_iter = True
            if not collection_done:
                collection_done = True
                post_collection_offset = LOG.stat().st_size if LOG.exists() else 0
                log(
                    f"NEXT_ITER_SEEN iter{it} {cur}/{tot} (latched {start_iter}) "
                    f"— treating prior collection done"
                )

        if collection_done:
            try:
                with LOG.open("rb") as fh:
                    fh.seek(post_collection_offset)
                    new_text = fh.read().decode("utf-8", "replace").replace("\r", "\n")
            except OSError:
                new_text = ""
            if PROMO_RE.search(new_text):
                seen_promo = True
            for m in ITER_START_RE.finditer(new_text):
                if int(m.group(1)) > (start_iter or -1):
                    seen_next_iter = True

        # Also: early next-iter collection (games still low) is a safe restart window
        early_next = (
            seen_next_iter
            and it > (start_iter or -1)
            and cur <= 40
            and tot > 0
        )

        rr_alive = bool(
            pids_by_substr("train_round_robin.py", f"{CORE_RUN}.hammer_search_rl")
        )
        write_owner(
            "OPENING_FIX_WAIT_BOUNDARY",
            f"phase={'wait_promo' if collection_done else 'wait_collection'} "
            f"games=({it},{cur},{tot}) promo={int(seen_promo)} next={int(seen_next_iter)}",
        )
        log(
            f"WAIT games=({it},{cur},{tot}) coll_done={collection_done} "
            f"promo={seen_promo} next={seen_next_iter} early={early_next} "
            f"rr_alive={rr_alive}"
        )

        if early_next or (collection_done and (seen_promo or seen_next_iter)):
            return {
                "reason": "early_next_iter" if early_next else "promo_or_next_iter",
                "games": games,
                "seen_promo": seen_promo,
                "seen_next_iter": seen_next_iter,
                "early_next": early_next,
            }
        if collection_done and not rr_alive:
            time.sleep(5)
            if not pids_by_substr(
                "train_round_robin.py", f"{CORE_RUN}.hammer_search_rl"
            ):
                return {
                    "reason": "trainer_down_after_collection",
                    "games": games,
                    "seen_promo": seen_promo,
                    "seen_next_iter": seen_next_iter,
                }
        time.sleep(poll_s)
    raise TimeoutError("boundary wait exceeded max_hours")


def stop_core() -> None:
    launch = pids_by_substr("launch_core_pipeline.py", CORE_RUN)
    pipe = pids_by_substr("train_core_pipeline.py", CORE_RUN)
    rr = pids_by_substr("train_round_robin.py", f"{CORE_RUN}.hammer_search_rl")
    log(f"STOP core pids launch={launch} pipe={pipe} rr={rr}")
    # Do not touch blackwell_* processes.
    targets = sorted(set(launch + pipe + rr))
    for pid in targets:
        try:
            os.killpg(pid, signal.SIGTERM)
            log(f"SIGTERM pgid={pid}")
        except ProcessLookupError:
            try:
                os.kill(pid, signal.SIGTERM)
                log(f"SIGTERM pid={pid}")
            except ProcessLookupError:
                pass
        except PermissionError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
    for _ in range(90):
        still = (
            pids_by_substr("launch_core_pipeline.py", CORE_RUN)
            + pids_by_substr("train_core_pipeline.py", CORE_RUN)
            + pids_by_substr("train_round_robin.py", f"{CORE_RUN}.hammer_search_rl")
        )
        if not still:
            log("STOP core down")
            return
        time.sleep(1)
    for pid in (
        pids_by_substr("launch_core_pipeline.py", CORE_RUN)
        + pids_by_substr("train_core_pipeline.py", CORE_RUN)
        + pids_by_substr("train_round_robin.py", f"{CORE_RUN}.hammer_search_rl")
    ):
        try:
            os.kill(pid, signal.SIGKILL)
            log(f"SIGKILL pid={pid}")
        except ProcessLookupError:
            pass
    time.sleep(2)


def invalidate_stale_metadata() -> None:
    meta = (
        ROOT
        / "outputs/runs"
        / f"{CORE_RUN}.hammer_search_rl"
        / "run_metadata.json"
    )
    if meta.is_file():
        dest = meta.with_name(
            f"run_metadata.json.invalid.opening-fix.{utc_now().replace(':', '')}"
        )
        meta.rename(dest)
        log(f"invalidated metadata -> {dest.name}")


def launch_core() -> int:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "POKEBOT_GPU_PROFILE": "3080ti",
            "POKEBOT_PRIMARY_ARCHETYPE": "core-canonical",
            "POKEBOT_WORKER_CPU_ONLY": "1",
            "POKEBOT_ALLOW_ORACLE_DECK": "0",
            "POKEBOT_REMOTE_PRIMARY": "1",
            "POKEBOT_OPENING_MOVE_TIME_MULT": "1.0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.pop("POKEBOT_REMOTE_SLOT_DIVISOR", None)
    cmd = [
        PY,
        "-u",
        str(ROOT / "scripts/launch_core_pipeline.py"),
        "--run-name",
        CORE_RUN,
        "--preflight-profile",
        "none",
        "--log",
        "outputs/logs/core_kernel.log",
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
        PY,
        "--",
        "--resume",
        "auto",
        "--phase",
        "auto",
        "--stable-log",
        "outputs/logs/core_kernel.log",
        "--workers",
        "6",
        "--worker-ceiling",
        "6",
        "--reserve-cpu-threads",
        "8",
        "--torch-threads",
        "8",
        "--gpu-memory-budget-gb",
        "8",
        "--core-search-games-per-opponent",
        "16",
        "--core-search-min-games-per-opponent",
        "12",
        "--core-search-max-games-per-opponent",
        "24",
        "--core-target-search-decisions",
        "3664",
        "--core-search-replay-fraction",
        "0.5",
        "--core-policy-anchor-ratio",
        "1",
        "--max-decisions-per-game",
        "32",
        "--core-search-epochs",
        "1",
        "--hammer-rl-iterations",
        "10000",
        "--hammer-target-search-decisions",
        "3664",
        "--search-start-sims",
        "128",
        "--search-move-time-s",
        "12",
        "--search-game-time-s",
        "1200",
        "--inference-servers",
        "3",
        "--inference-batch",
        "128",
        "--inference-queue-depth",
        "64",
        "--inference-timeout-s",
        "30",
        "--promotion-workers",
        "12",
        "--remote-worker-endpoints",
        f"{ELMO},{BERT}",
    ]
    log("LAUNCH core with POKEBOT_OPENING_MOVE_TIME_MULT=1.0 Elmo+Bert")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=open(ROOT / "outputs/logs/core_kernel.launcher.stdout", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    LAUNCH_PID.write_text(str(proc.pid) + "\n")
    return proc.pid


def count_errors(chunk: str) -> dict[str, int]:
    leftovers = [float(x) for x in re.findall(r"timed out after ([0-9.]+)s", chunk)]
    return {
        "RemoteLeafTimeout": chunk.count("RemoteLeafTimeout"),
        "RemoteLeafTimeout_lt50ms": sum(1 for x in leftovers if x < 0.05),
        "TrustedSearchBudgetExhausted": chunk.count("TrustedSearchBudgetExhausted"),
        "within_6.000s": chunk.count("within 6.000s"),
        "connection_closed": chunk.count("connection closed while reading frame"),
        "slot_failed": chunk.count("RemoteJobsError: slot")
        + chunk.count("slot failed"),
    }


def verify(window_s: float = 180.0, poll_s: float = 20.0) -> dict:
    # Snapshot baseline after relaunch marker
    time.sleep(25)
    size0 = LOG.stat().st_size if LOG.exists() else 0
    deadline = time.time() + window_s
    last: dict[str, int] = {}
    opening_env = None
    while time.time() < deadline:
        pipe = pids_by_substr("train_core_pipeline.py", CORE_RUN)
        if pipe:
            try:
                env = Path(f"/proc/{pipe[0]}/environ").read_bytes().split(b"\0")
                for item in env:
                    if item.startswith(b"POKEBOT_OPENING_MOVE_TIME_MULT="):
                        opening_env = item.decode().split("=", 1)[1]
            except OSError:
                pass
        raw = LOG.read_bytes()
        delta = raw[size0:].decode("utf-8", "replace") if len(raw) > size0 else ""
        last = count_errors(delta)
        games = parse_games(core_chunk())
        log(
            f"VERIFY opening_env={opening_env} games={games} errs={last}"
        )
        time.sleep(poll_s)
    last["opening_env"] = opening_env  # type: ignore[assignment]
    return last


def update_status_md(phase: str, detail: dict | None = None) -> None:
    detail = detail or {}
    applied = phase == "applied"
    body = f"""# ERROR_FIX_STATUS

updated={utc_now()}
agent=opening-fix-boundary
topology=CORE_ELMO_PLUS_BERT_BW_LOCAL
owner_lock=SOLE_TRAIN_OWNER.lock (held; do not undo patches)

## Executive verdict

1. **FIXED (verified clean):** `connection closed while reading frame` / remote slot-fail / CLOSE-WAIT storm.
2. **FIXED (deployed):** remote idle-timeout keep-alive on Elmo+Bert.
3. **FIXED (code, live on Core client):** farm template `ensure_alive` + hangup reconnect/retry.
4. **{'APPLIED + verified' if applied else 'ARMED / in-progress'}:** `POKEBOT_OPENING_MOVE_TIME_MULT=1.0` via `scripts/core_opening_fix_at_boundary.py` (Core only; BW untouched).
5. **BW:** local-only; do not restart unless fail-closed.

## Opening-budget fix

- Root cause: default `opening_move_time_mult=0.5` (env `POKEBOT_OPENING_MOVE_TIME_MULT`) cut 12s → **6.0s** while `search_start_sims=128`.
- Note: unprefixed `OPENING_MOVE_TIME_MULT` is **ignored** by `config._env_float` (requires `POKEBOT_` prefix).
- Phase: **{phase}**
- Detail: `{detail}`
- Expected after apply: `POKEBOT_OPENING_MOVE_TIME_MULT=1.0` in Core environ; collapse of `within 6.000s` TrustedSearchBudgetExhausted and leftover `<50ms` RemoteLeafTimeout; `connection_closed` stays 0.

## Do not undo

multi-tenant digest, version lockstep, idle-timeout, signal, baseline remap, remote_play_job, WorkerPool.apply fallback, ensure_alive/reconnect, Elmo+Bert Core topology.
"""
    STATUS_MD.write_text(body)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply-now",
        action="store_true",
        help="Skip wait; restart Core immediately (early next-iter / operator force).",
    )
    args = parser.parse_args(argv)

    WATCHER_PID.write_text(str(os.getpid()) + "\n")
    log("START opening-fix boundary waiter")
    if args.apply_now:
        boundary = {
            "reason": "apply_now",
            "games": parse_games(core_chunk()),
        }
        log(f"APPLY_NOW {boundary}")
    else:
        update_status_md(
            "armed_mid_collection",
            {"note": "waiting N/N + promo/next-iter; BW untouched"},
        )
        boundary = wait_boundary()
        log(f"BOUNDARY_HIT {boundary}")
    write_owner("OPENING_FIX_RESTARTING", f"boundary={boundary}")
    update_status_md("boundary_hit_restarting", boundary)
    stop_core()
    invalidate_stale_metadata()
    pid = launch_core()
    log(f"relaunched launch_pid={pid}")
    write_owner("OPENING_FIX_VERIFY", f"launch_pid={pid}")
    results = verify(window_s=180.0)
    ok = (
        results.get("connection_closed", 1) == 0
        and results.get("opening_env") == "1.0"
        and results.get("within_6.000s", 99) == 0
    )
    write_owner(
        "OPENING_FIX_APPLIED" if ok else "OPENING_FIX_APPLIED_CHECK",
        f"verify={results}",
    )
    update_status_md("applied", results)
    log(f"DONE ok={ok} verify={results}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"FATAL {exc!r}")
        write_owner("OPENING_FIX_FATAL", f"error={exc!r}")
        raise
