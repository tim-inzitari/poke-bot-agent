#!/usr/bin/env python3
"""Temporary live progress mirror: shard JSONL → *.progress.status.

Used when launch forgot stderr→progress.log split. Safe to kill once the
trainer itself writes tqdm bars. Does not touch the training process.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


class _ShardCursor:
    """Incremental games/decisions counter (avoids full reparse each tick)."""

    __slots__ = ("pos", "games", "decisions")

    def __init__(self) -> None:
        self.pos = 0
        self.games = 0
        self.decisions = 0

    def refresh(self, path: Path) -> None:
        if not path.is_file():
            self.pos = 0
            self.games = 0
            self.decisions = 0
            return
        size = path.stat().st_size
        if size < self.pos:
            self.pos = 0
            self.games = 0
            self.decisions = 0
        with path.open("rb") as f:
            f.seek(self.pos)
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    break
                self.pos = f.tell()
                self.games += 1
                try:
                    obj = json.loads(line)
                    self.decisions += len(obj.get("decisions") or [])
                except Exception:
                    # Fallback: each CompactDecision carries selected_index.
                    self.decisions += line.count(b'"selected_index"')


def parse_collect_iter(event_log: Path) -> int:
    """Return latest ``collect iter=N`` from the event log (0 if unknown)."""
    if not event_log.is_file():
        return 0
    text = event_log.read_text(errors="replace")
    for ln in reversed(text.splitlines()):
        m = re.search(r"collect iter=(\d+)", ln)
        if m:
            return int(m.group(1))
    return 0


def parse_totals(event_log: Path) -> tuple[int, int]:
    """Return (self_play, public_mix) from collect iter line."""
    sp, pm = 3482, 614
    if not event_log.is_file():
        return sp, pm
    text = event_log.read_text(errors="replace")
    for ln in reversed(text.splitlines()):
        if "collect iter=" not in ln:
            continue
        m_sp = re.search(r"self_play=(\d+)", ln)
        m_pm = re.search(r"public_mix=(\d+)", ln)
        if m_sp:
            sp = int(m_sp.group(1))
        if m_pm:
            pm = int(m_pm.group(1))
        break
    return sp, pm


def parse_remote_workers_from_event(event_log: Path) -> int:
    """Best-effort farm size from the latest collect/remote lines."""
    if not event_log.is_file():
        return 0
    text = event_log.read_text(errors="replace")
    for ln in reversed(text.splitlines()):
        m = re.search(r"remote_workers=(\d+)", ln)
        if m:
            return int(m.group(1))
        m = re.search(r"remote additive capacity=(\d+)", ln)
        if m:
            return int(m.group(1))
    return 0


def probe_live_remotes(endpoints: list[str], *, timeout_s: float = 3.0) -> int:
    """Sum advertised workers from hello; 0 if none reachable."""
    if not endpoints:
        return 0
    import sys

    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from poke_bot.remote_jobs import RemoteJobClient, parse_endpoint
    except Exception as exc:
        print(f"[shard-mirror] WARN cannot import remote_jobs: {exc}", flush=True)
        return 0
    total = 0
    alive = 0
    for ep in endpoints:
        ep = ep.strip()
        if not ep:
            continue
        try:
            client = RemoteJobClient(
                *parse_endpoint(ep),
                timeout_s=timeout_s,
                connect_timeout_s=timeout_s,
                control_timeout_s=timeout_s,
            )
            info = client.connect()
            total += int(info.workers or 0)
            alive += 1
            client.close()
        except Exception as exc:
            print(f"[shard-mirror] WARN hello failed {ep}: {exc}", flush=True)
            continue
    if alive:
        print(f"[shard-mirror] live remotes={total} endpoints_up={alive}", flush=True)
    return total


def bar_line(
    done: int,
    total: int,
    stage: str,
    t0: float,
    remotes: int,
    *,
    games_delta: int,
    decisions_delta: int,
    iteration: int = 0,
) -> str:
    total = max(total, 1)
    done = min(max(done, 0), total)
    pct = 100.0 * done / total
    filled = int(pct / 10)
    bar = "#" * filled + "." * (10 - filled)
    elapsed = max(time.time() - t0, 1e-6)
    # Rates from progress since mirror attach (not absolute / short wall).
    gps = max(games_delta, 0) / elapsed
    sps = max(decisions_delta, 0) / elapsed
    remain = max(total - done, 0)
    eta = remain / gps if gps > 1e-9 else 0.0

    def fmt(s: float) -> str:
        m, sec = divmod(int(s), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

    return (
        f"pure_rl {stage} iter={int(iteration)}: {pct:3.0f}%|{bar}| {done}/{total} "
        f"[{fmt(elapsed)}<{fmt(eta)}, {gps:.2f}game/s, sps={sps:.1f}, "
        f"remotes={remotes}] (shard-mirror)"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--event-log", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--progress-log", required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--train-pid", type=int, default=0)
    ap.add_argument(
        "--remote-worker-endpoints",
        default="",
        help="Comma-separated endpoints to probe for live remotes= count",
    )
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    status = Path(args.status)
    prog = Path(args.progress_log)
    event = Path(args.event_log)
    endpoints = [
        p.strip()
        for p in str(args.remote_worker_endpoints or "").split(",")
        if p.strip()
    ]
    if not endpoints:
        # Fall back to the overnight pair when launch omitted the flag.
        endpoints = ["192.168.1.143:8765", "bert.local:8766"]
    status.parent.mkdir(parents=True, exist_ok=True)
    cursor = _ShardCursor()
    base_games: int | None = None
    base_decisions = 0
    t0 = time.time()
    last_remotes = parse_remote_workers_from_event(event)
    last_probe = 0.0
    last_iter = -1
    while True:
        if args.train_pid and not Path(f"/proc/{args.train_pid}").exists():
            break
        it = parse_collect_iter(event)
        shard = run_dir / "shards" / f"iter_{it:05d}.jsonl"
        if it != last_iter:
            cursor = _ShardCursor()
            base_games = None
            base_decisions = 0
            t0 = time.time()
            last_iter = it
        sp, pm = parse_totals(event)
        cursor.refresh(shard)
        done = cursor.games
        decisions = cursor.decisions
        if base_games is None:
            base_games = done
            base_decisions = decisions
            t0 = time.time()
        now = time.time()
        # Probe live hellos every ~10s so a bert cut/redeploy updates the bar.
        if now - last_probe >= 10.0:
            live = probe_live_remotes(endpoints)
            if live > 0:
                last_remotes = live
            elif last_remotes <= 0:
                last_remotes = parse_remote_workers_from_event(event)
            last_probe = now
        remotes = int(last_remotes)
        if done < sp:
            stage, stage_done, stage_total = "collect:self_play", done, sp
        else:
            stage, stage_done, stage_total = (
                "collect:public_mix",
                done - sp,
                pm,
            )
        # Refresh every tick so remotes=/sps update even if shard count stalls.
        line = bar_line(
            stage_done,
            stage_total,
            stage,
            t0,
            remotes,
            games_delta=done - base_games,
            decisions_delta=decisions - base_decisions,
            iteration=it,
        )
        status.write_text(line + "\n", encoding="utf-8")
        with prog.open("ab") as f:
            f.write((line + "\r").encode())
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
