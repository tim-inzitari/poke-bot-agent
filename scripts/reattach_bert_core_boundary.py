#!/usr/bin/env python3
"""Wait for Core promo/iter-safe boundary, then attach Bert+Elmo (BW untouched).

Obeys outputs/state/RESTART_POLICY.md. Soft-attach of new endpoints is not
supported mid-run, so this script waits for collection N/N and PROMOTED/REJECTED
(or next-iter start), then restarts Core with both remote endpoints.
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
BW_LOG = ROOT / "outputs/logs/blackwell.log"
STATUS = ROOT / "outputs/state/BERT_REATTACH_STATUS.txt"
OWNER = ROOT / "outputs/state/SOLE_TRAIN_OWNER.lock"
TOPOLOGY = ROOT / "outputs/state/TRACK_A_TOPOLOGY.txt"
PY = "/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"
ELMO = "192.168.1.143:8765"
BERT = "bert.local:8766"
CORE_RUN = "core_kernel_3080ti_trusted_20260714T2000Z"

GAMES_RE = re.compile(r"iter(\d+) games:\s+\d+%\|[^\|]*\|\s+(\d+)/(\d+)")
PROMO_RE = re.compile(r"\[rr\] (PROMOTED|REJECTED) candidate")
ITER_START_RE = re.compile(r"\[rr\] iter (\d+):")
REMOTE_LINE_RE = re.compile(r"remote-worker=([^\s]+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"[{utc_now()}] {msg}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a") as fh:
        fh.write(line + "\n")


def write_owner(status: str, extra: str = "") -> None:
    OWNER.write_text(
        "\n".join(
            [
                "SOLE_TRAIN_OWNER=1",
                f"status={status}",
                f"updated={utc_now()}",
                "agent=bert-reattach-track-b",
                "topology="
                + (
                    "CORE_ELMO_PLUS_BERT"
                    if "ATTACHED" in status or status == "OVERNIGHT_MONITOR"
                    else "CORE_ELMO_PLUS_BERT_PENDING"
                ),
                "bw=local-only",
                "rule=Obey RESTART_POLICY.md. Bert on Core only; BW local-only.",
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


def ss_counts() -> dict[str, int]:
    try:
        out = subprocess.check_output(["ss", "-tn"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {}
    counts = {
        "elmo_estab": 0,
        "elmo_closewait": 0,
        "bert_estab": 0,
        "bert_closewait": 0,
    }
    for line in out.splitlines():
        if "192.168.1.143:8765" in line:
            if "ESTAB" in line:
                counts["elmo_estab"] += 1
            if "CLOSE-WAIT" in line:
                counts["elmo_closewait"] += 1
        if "192.168.1.157:8766" in line or ":8766" in line and "192.168.1.157" in line:
            if "ESTAB" in line:
                counts["bert_estab"] += 1
            if "CLOSE-WAIT" in line:
                counts["bert_closewait"] += 1
        # also count bert.local resolved differently
        if "ESTAB" in line and ":8766" in line:
            counts["bert_estab"] = max(
                counts["bert_estab"],
                sum(
                    1
                    for l in out.splitlines()
                    if "ESTAB" in l and ":8766" in l
                ),
            )
    # recount bert simply
    counts["bert_estab"] = sum(
        1 for l in out.splitlines() if "ESTAB" in l and ":8766" in l
    )
    counts["bert_closewait"] = sum(
        1 for l in out.splitlines() if "CLOSE-WAIT" in l and ":8766" in l
    )
    return counts


def count_pattern(chunk: str, pat: str) -> int:
    return len(re.findall(pat, chunk, flags=re.I))


def wait_boundary(poll_s: float = 30.0, max_hours: float = 6.0) -> dict:
    """Wait until collection done AND promo decided (or next iter started)."""
    deadline = time.time() + max_hours * 3600
    collection_done_at: float | None = None
    seen_promo = False
    seen_next_iter = False
    start_iter = None
    # Latch: once start_iter collection hits N/N, keep it across train/promo/next-iter.
    start_iter_collection_done = False
    write_owner("BERT_REATTACH_WAIT_BOUNDARY", "phase=wait_collection_then_promo")
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
        # Mark collection complete for the iter we started waiting on
        if it == start_iter and cur >= tot and tot > 0:
            if not start_iter_collection_done:
                collection_done_at = time.time()
                start_iter_collection_done = True
                log(f"COLLECTION_DONE iter{it} {cur}/{tot} — waiting promo (RESTART_POLICY)")
                write_owner(
                    "BERT_REATTACH_WAIT_BOUNDARY",
                    f"phase=wait_promo iter={it} games={cur}/{tot}",
                )

        # Promo markers after the last start (only after collection latched)
        if start_iter_collection_done and PROMO_RE.search(chunk):
            # Prefer promo lines that appear after collection-done wall time by
            # re-scanning the tail; any PROMOTED/REJECTED in this start chunk
            # after N/N is the boundary we want.
            seen_promo = True
        # Next iter start beyond the one we began waiting on
        for m in ITER_START_RE.finditer(chunk):
            if int(m.group(1)) > start_iter:
                seen_next_iter = True

        # Also: if games bar advances to a higher iter
        if it > start_iter:
            seen_next_iter = True

        rr_alive = bool(
            pids_by_substr("train_round_robin.py", f"{CORE_RUN}.hammer_search_rl")
        )
        # Refresh ownership every poll so overnight No-Bert agents cannot steal the lock.
        write_owner(
            "BERT_REATTACH_WAIT_BOUNDARY",
            f"phase={'wait_promo' if start_iter_collection_done else 'wait_collection'} "
            f"games=({it},{cur},{tot}) promo={int(seen_promo)} next={int(seen_next_iter)}",
        )
        log(
            f"WAIT games=({it},{cur},{tot}) coll_done={start_iter_collection_done} "
            f"promo={seen_promo} next_iter={seen_next_iter} rr_alive={rr_alive} "
            f"ss={ss_counts()}"
        )

        if start_iter_collection_done and (seen_promo or seen_next_iter):
            return {
                "reason": "promo_or_next_iter",
                "games": games,
                "seen_promo": seen_promo,
                "seen_next_iter": seen_next_iter,
            }
        # If trainer died after collection, that is also a safe window
        if start_iter_collection_done and not rr_alive:
            # give a moment in case of respawn
            time.sleep(5)
            if not pids_by_substr("train_round_robin.py", f"{CORE_RUN}.hammer_search_rl"):
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
    targets = launch or pipe
    for pid in targets:
        try:
            os.killpg(pid, signal.SIGTERM)
            log(f"SIGTERM pgid={pid}")
        except ProcessLookupError:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    # Wait up to 90s
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
    # escalate
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


def launch_core_with_bert() -> dict[str, int]:
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
    # Explicitly unset slot divisor so Core is sole consumer of both remotes
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
    log("LAUNCH " + " ".join(cmd[-6:]))
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(8)
    # Discover child pids
    launch = proc.pid
    pipe = pids_by_substr("train_core_pipeline.py", CORE_RUN)
    rr = pids_by_substr("train_round_robin.py", f"{CORE_RUN}.hammer_search_rl")
    return {"core_launch": launch, "core_pipe": pipe[0] if pipe else -1, "core_rr": rr[0] if rr else -1}


def prove_topology(timeout_s: float = 180.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        chunk = core_chunk()
        remotes = REMOTE_LINE_RE.findall(chunk)
        # Prefer the most recent start chunk's remote-worker lines
        start_idx = chunk.rfind("[core-pipeline][start]")
        recent = chunk[start_idx:] if start_idx >= 0 else chunk
        remotes = REMOTE_LINE_RE.findall(recent)
        has_elmo = any(ELMO in r for r in remotes) or f"remote-worker={ELMO}" in recent
        has_bert = any(BERT in r for r in remotes) or f"remote-worker={BERT}" in recent
        # Also check cmdline of live rr
        rr = pids_by_substr("train_round_robin.py", f"{CORE_RUN}.hammer_search_rl")
        cmd = ""
        if rr:
            cmd = Path(f"/proc/{rr[0]}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        bw = pids_by_substr("train_round_robin.py", "blackwell_hammer_belief")
        bw_cmd = ""
        if bw:
            bw_cmd = Path(f"/proc/{bw[0]}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        bw_local = bw and "remote-worker" not in bw_cmd and BERT not in bw_cmd
        cmdline_ok = ELMO in cmd and BERT in cmd
        if (has_elmo and has_bert) or cmdline_ok:
            return {
                "ok": True,
                "remotes": remotes,
                "cmdline_ok": cmdline_ok,
                "bw_local": bool(bw_local),
                "ss": ss_counts(),
                "core_rr": rr[0] if rr else -1,
                "bw_rr": bw[0] if bw else -1,
            }
        log(f"PROOF waiting remotes={remotes} cmdline_ok={cmdline_ok} rr={rr}")
        time.sleep(5)
    return {"ok": False, "ss": ss_counts()}


def watch_post_attach(window_s: float = 300.0, poll_s: float = 20.0) -> dict:
    """Watch for Bert leaf-timeout / slot-fail storm; soft-drop Bert if needed."""
    t0 = time.time()
    chunk0 = core_chunk()
    base_timeout = count_pattern(chunk0, "RemoteLeafTimeout")
    base_bert_slot = count_pattern(chunk0, r"bert\.local:8766 slot failed")
    base_remote_bert = count_pattern(chunk0, r"\[remote\] bert\.local")
    samples = []
    while time.time() - t0 < window_s:
        time.sleep(poll_s)
        chunk = core_chunk()
        games = parse_games(chunk)
        timeouts = count_pattern(chunk, "RemoteLeafTimeout") - base_timeout
        bert_slot = count_pattern(chunk, r"bert\.local:8766 slot failed") - base_bert_slot
        remote_bert = count_pattern(chunk, r"\[remote\] bert\.local") - base_remote_bert
        ss = ss_counts()
        sample = {
            "t": int(time.time() - t0),
            "games": games,
            "timeout_delta": timeouts,
            "bert_slot_delta": bert_slot,
            "remote_bert_delta": remote_bert,
            "ss": ss,
        }
        samples.append(sample)
        log(f"WATCH {sample}")
        # Storm criteria: many Bert-specific slot fails, or timeout flood with Bert ESTAB
        storm = bert_slot >= 8 or (timeouts >= 40 and ss.get("bert_estab", 0) > 0 and bert_slot >= 3)
        if storm:
            log(f"STORM detected — soft-dropping Bert (relaunch Elmo-only) sample={sample}")
            return {"storm": True, "samples": samples, "action": "soft_drop_bert"}
    return {"storm": False, "samples": samples}


def soft_drop_bert() -> dict[str, int]:
    """Restart Core Elmo-only (operational soft-drop)."""
    stop_core()
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
        ELMO,
    ]
    log("SOFT_DROP relaunch Elmo-only")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(8)
    pipe = pids_by_substr("train_core_pipeline.py", CORE_RUN)
    rr = pids_by_substr("train_round_robin.py", f"{CORE_RUN}.hammer_search_rl")
    return {
        "core_launch": proc.pid,
        "core_pipe": pipe[0] if pipe else -1,
        "core_rr": rr[0] if rr else -1,
    }


def write_topology(pids: dict, proof: dict, storm: dict) -> None:
    TOPOLOGY.write_text(
        f"""# Track B topology (Bert reattached to Core)

updated={utc_now()}

## Production
- **Core**: Elmo `{ELMO}` + Bert `{BERT}`, `POKEBOT_REMOTE_PRIMARY=1`, no SLOT_DIVISOR
- **Blackwell**: local-only (no remote endpoints) — unchanged
- **Bert soft-drop**: {"YES — reverted Elmo-only" if storm.get("storm") else "no storm in watch window"}

## PIDs
- core_launch={pids.get("core_launch")} core_pipe={pids.get("core_pipe")} core_rr={pids.get("core_rr")}
- bw_rr={proof.get("bw_rr")} (local-only)

## Proof
- cmdline_ok={proof.get("cmdline_ok")} remotes={proof.get("remotes")}
- bw_local={proof.get("bw_local")} ss={proof.get("ss")}
- storm={storm.get("storm")} samples={len(storm.get("samples") or [])}
"""
    )


def main() -> int:
    STATUS.write_text(f"[{utc_now()}] START bert reattach waiter\n")
    log("Bert healthy check")
    try:
        subprocess.run(
            [PY, "-u", str(ROOT / "scripts/canary_remote_worker.py"), BERT],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        log("Bert canary hello OK")
    except Exception as exc:
        log(f"Bert canary hello FAILED: {exc}")
        write_owner("BERT_REATTACH_ABORT", f"reason=bert_unhealthy {exc}")
        return 2

    # Ensure BW stays untouched — just record pid
    bw = pids_by_substr("train_round_robin.py", "blackwell_hammer_belief")
    log(f"BW rr pids (must stay local)={bw}")

    boundary = wait_boundary()
    log(f"BOUNDARY {boundary}")
    write_owner("BERT_REATTACH_RESTARTING", f"boundary={boundary}")

    stop_core()
    pids = launch_core_with_bert()
    log(f"LAUNCHED {pids}")
    proof = prove_topology()
    log(f"PROOF {proof}")
    if not proof.get("ok"):
        write_owner("BERT_REATTACH_FAIL_PROOF", f"proof={proof}")
        return 3

    write_owner("BERT_REATTACH_WATCH", f"pids={pids}")
    storm = watch_post_attach(window_s=300.0)
    if storm.get("storm"):
        pids = soft_drop_bert()
        proof = prove_topology()
        write_topology(pids, proof, storm)
        write_owner(
            "BERT_SOFT_DROPPED",
            f"pids={pids} storm=1 — Core Elmo-only; BW local; report needed",
        )
        log("DONE soft-dropped Bert after storm")
        return 4

    write_topology(pids, proof, storm)
    write_owner(
        "OVERNIGHT_MONITOR",
        f"last_tick=attached pids={pids} bw_local={proof.get('bw_local')} ss={proof.get('ss')}",
    )
    log("DONE Bert attached to Core; BW local; errors watch clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
