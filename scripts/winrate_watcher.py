#!/usr/bin/env python3
"""Win-rate progression watcher for hammer (or any) round-robin RL.

Asserts that RR training keeps improving overall win rate. Separately from any
resource / log-trim watcher::

  * Track best overall WR seen across ``outputs/eval/rr_iter*.json`` (also
    scrapes ``[rr] iter=N ... wr=X%`` lines from the RR log as a backup).
  * Each poll: log trending WR, best WR, iters-since-improve.
  * Two alert lanes (both can spawn the Sol Max Fast / Auto adjuster):

      1. **Sharp collapse (immediate):** overall WR drops by ≥3 absolute points
         over the last 1–2 iterations (e.g. 7.4%→3.5%). Fire right away.
      2. **Low-WR plateau (~≤10%):** only when WR **and** best WR stay at/below
         the low band (default ``--low-wr-threshold 0.10``) **and** there are
         ``--stagnation-iters`` consecutive iterations with no best-WR
         improvement of ≥ε (defaults: **5** iters, ε = 0.5 WR points). A
         brief dip while still climbing does not count.

On alert the watcher:
  1. Appends to ``outputs/logs/winrate_watcher.log``
  2. Writes ``outputs/eval/WINRATE_STAGNATION_ALERT`` (metrics)
  3. Writes ``outputs/eval/STALL_ACTION_REQUESTED`` (operator restart hint +
     recommended good checkpoint)
  4. Spawns a Cursor adjuster agent via ``scripts/spawn_wr_adjuster.sh``
     (preferred model ``gpt-5.6-sol-max-fast``, quota → ``auto``). Debounced
     with a lock so only one adjuster flies at a time.

Does NOT kill the RR process. Safe intervention is delegated to the spawned
agent (revert champion / tweak hyperparams / fix empty experience) plus the
flag files for an operator.

Defaults
--------
``--stagnation-iters 5 --low-wr-threshold 0.10 --min-improve 0.005
--collapse-drop 0.03 --interval 60``

Launch (nohup)::

    nohup /home/pokebot/miniconda3/envs/poke-bot-agent/bin/python \\
        scripts/winrate_watcher.py > outputs/logs/winrate_watcher.nohup.out 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from poke_bot import paths as _paths

    OUTPUTS_DIR = _paths.OUTPUTS_DIR
    CHECKPOINTS_DIR = _paths.CHECKPOINTS_DIR
except Exception:  # pragma: no cover - keep watcher stand-alone
    OUTPUTS_DIR = REPO_ROOT / "outputs"
    CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

EVAL_DIR = OUTPUTS_DIR / "eval"
LOG_DIR = OUTPUTS_DIR / "logs"
DEFAULT_RR_LOG = LOG_DIR / "round_robin_hammer.log"
DEFAULT_SELF_LOG = LOG_DIR / "winrate_watcher.log"
DEFAULT_STATE = EVAL_DIR / "winrate_watcher_state.json"
ALERT_FLAG = EVAL_DIR / "WINRATE_STAGNATION_ALERT"
STALL_FLAG = EVAL_DIR / "STALL_ACTION_REQUESTED"
SPAWN_SCRIPT = REPO_ROOT / "scripts" / "spawn_wr_adjuster.sh"
ADJUSTER_LOCK = EVAL_DIR / "WR_ADJUSTER.lock"
PROMPT_DIR = EVAL_DIR / "wr_adjuster_prompts"

RR_ITER_RE = re.compile(
    r"\[rr\]\s+iter=(?P<it>\d+)\s+games=(?P<games>\d+)\s+wr=(?P<wr>[0-9.]+)%"
)

DEFAULT_STAGNATION_ITERS = 5  # consecutive no-improve iters while in low-WR band
DEFAULT_LOW_WR = 0.10  # ~10% — low-level plateau band
DEFAULT_MIN_IMPROVE = 0.005  # 0.5 WR points
DEFAULT_COLLAPSE_DROP = 0.03  # 3 absolute points (immediate collapse interrupt)
DEFAULT_INTERVAL = 60.0


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ActionLogger:
    """Append logger; never raises into the poll loop."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def __call__(self, msg: str) -> None:
        line = f"[{_now()}] {msg}"
        print(line, flush=True)
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass


@dataclass
class IterWR:
    iteration: int
    wr: float
    n_games: int = 0
    n_passing_wilson: int = 0
    n_evaluated: int = 0
    champion: str = ""
    best_champion: str = ""
    source: str = "eval"  # eval | log
    path: str = ""


@dataclass
class WatchState:
    best_wr: float = -1.0
    best_iter: int = -1
    last_seen_iter: int = -1
    iters_since_improve: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    last_alert_kind: str = ""
    last_alert_iter: int = -1
    last_alert_ts: float = 0.0
    adjuster_spawns: int = 0


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _overall_wr_from_summary(summary: dict[str, Any]) -> tuple[float, int]:
    matchups = summary.get("matchups") or []
    total_g = 0
    total_score = 0.0
    for m in matchups:
        g = int(m.get("games") or 0)
        if g <= 0:
            continue
        total_g += g
        if "wins" in m:
            total_score += float(m.get("wins") or 0.0) + 0.5 * float(m.get("draws") or 0.0)
        elif m.get("wr") is not None:
            total_score += float(m["wr"]) * g
    if total_g <= 0:
        return 0.0, 0
    return total_score / total_g, total_g


def load_eval_iters(eval_dir: Path) -> list[IterWR]:
    out: list[IterWR] = []
    if not eval_dir.exists():
        return out
    for path in sorted(eval_dir.glob("rr_iter*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        loop = data.get("loop") or {}
        summary = data.get("summary") or {}
        # Prefer top-level summary for this file's iteration; fall back to
        # history entry matching the filename iteration.
        m = re.search(r"rr_iter(\d+)\.json$", path.name)
        it = int(m.group(1)) if m else int(loop.get("iteration") or -1)
        wr = None
        n_games = 0
        evaluated_champion = str(loop.get("last_evaluated_champion") or "")
        for h in reversed(loop.get("history") or []):
            if int(h.get("iteration", -1)) == it and h.get("wr") is not None:
                wr = float(h["wr"])
                hs = h.get("summary") or {}
                n_games = sum(int(x.get("games") or 0) for x in (hs.get("matchups") or []))
                evaluated_champion = str(h.get("champion") or evaluated_champion)
                if not summary:
                    summary = hs
                break
        if wr is None:
            wr, n_games = _overall_wr_from_summary(summary)
        out.append(
            IterWR(
                iteration=it,
                wr=float(wr),
                n_games=n_games,
                n_passing_wilson=int(
                    summary.get("n_passing_draw_aware")
                    or summary.get("n_passing_wilson")
                    or 0
                ),
                n_evaluated=int(summary.get("n_evaluated") or 0),
                # ``loop.champion`` is the candidate trained *after* this eval.
                # Prefer the policy that actually played the recorded games.
                champion=evaluated_champion or str(loop.get("champion") or ""),
                best_champion=str(loop.get("best_champion") or ""),
                source="eval",
                path=str(path),
            )
        )
    out.sort(key=lambda r: r.iteration)
    return out


def scrape_log_iters(log_path: Path) -> list[IterWR]:
    if not log_path.exists():
        return []
    found: dict[int, IterWR] = {}
    try:
        # Read only the last ~2 MiB so we stay cheap on huge RR logs.
        size = log_path.stat().st_size
        with open(log_path, "rb") as fh:
            if size > 2 * 1024 * 1024:
                fh.seek(size - 2 * 1024 * 1024)
                fh.readline()  # discard partial line
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    for m in RR_ITER_RE.finditer(text):
        it = int(m.group("it"))
        wr = float(m.group("wr")) / 100.0
        games = int(m.group("games"))
        found[it] = IterWR(
            iteration=it,
            wr=wr,
            n_games=games,
            source="log",
            path=str(log_path),
        )
    return [found[k] for k in sorted(found)]


def merge_iters(eval_rows: list[IterWR], log_rows: list[IterWR]) -> list[IterWR]:
    """Prefer eval JSON; fill gaps from log scrape."""
    by_it: dict[int, IterWR] = {r.iteration: r for r in log_rows}
    for r in eval_rows:
        by_it[r.iteration] = r
    return [by_it[k] for k in sorted(by_it)]


def resolve_good_checkpoint(
    rows: list[IterWR],
    archetype: str,
) -> Path:
    """Pick the safest known-good weights for a stall restart hint."""
    # 1) best_champion recorded on the latest eval that saw an improvement.
    for r in reversed(rows):
        if r.best_champion:
            p = Path(r.best_champion)
            if not p.is_absolute():
                p = REPO_ROOT / p
            if p.exists():
                return p
    bootstrap = CHECKPOINTS_DIR / f"{archetype}_bootstrap.best.pt"
    if bootstrap.exists():
        return bootstrap
    warm = CHECKPOINTS_DIR / f"{archetype}_round_robin_warm.latest.pt"
    if warm.exists():
        return warm
    latest = CHECKPOINTS_DIR / f"{archetype}_round_robin.latest.pt"
    return latest


def load_state(path: Path) -> WatchState:
    if not path.exists():
        return WatchState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return WatchState(
            best_wr=float(raw.get("best_wr", -1.0)),
            best_iter=int(raw.get("best_iter", -1)),
            last_seen_iter=int(raw.get("last_seen_iter", -1)),
            iters_since_improve=int(raw.get("iters_since_improve", 0)),
            history=list(raw.get("history") or []),
            last_alert_kind=str(raw.get("last_alert_kind") or ""),
            last_alert_iter=int(raw.get("last_alert_iter", -1)),
            last_alert_ts=float(raw.get("last_alert_ts") or 0.0),
            adjuster_spawns=int(raw.get("adjuster_spawns") or 0),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return WatchState()


def save_state(path: Path, state: WatchState) -> None:
    _atomic_write(path, json.dumps(asdict(state), indent=2) + "\n")


def rebuild_progress(rows: list[IterWR], min_improve: float) -> WatchState:
    """Recompute best / iters-since-improve from the full trajectory."""
    st = WatchState()
    for r in rows:
        st.last_seen_iter = r.iteration
        st.history.append(
            {
                "iteration": r.iteration,
                "wr": r.wr,
                "n_games": r.n_games,
                "n_passing_wilson": r.n_passing_wilson,
                "source": r.source,
            }
        )
        # Keep history bounded.
        if len(st.history) > 500:
            st.history = st.history[-500:]
        if st.best_wr < 0:
            st.best_wr = r.wr
            st.best_iter = r.iteration
            st.iters_since_improve = 0
        elif r.wr > st.best_wr + min_improve:
            st.best_wr = r.wr
            st.best_iter = r.iteration
            st.iters_since_improve = 0
        else:
            st.iters_since_improve = max(0, r.iteration - st.best_iter)
    return st


def detect_collapse(
    rows: list[IterWR],
    drop: float,
) -> Optional[dict[str, Any]]:
    """Sharp WR collapse over the last 1–2 iterations."""
    if len(rows) < 2:
        return None
    cur = rows[-1]
    prev = rows[-2]
    d1 = prev.wr - cur.wr
    if d1 >= drop:
        return {
            "kind": "sharp_collapse",
            "from_iter": prev.iteration,
            "to_iter": cur.iteration,
            "from_wr": prev.wr,
            "to_wr": cur.wr,
            "drop": d1,
            "window": 1,
        }
    if len(rows) >= 3:
        older = rows[-3]
        d2 = older.wr - cur.wr
        if d2 >= drop:
            return {
                "kind": "sharp_collapse",
                "from_iter": older.iteration,
                "to_iter": cur.iteration,
                "from_wr": older.wr,
                "to_wr": cur.wr,
                "drop": d2,
                "window": 2,
            }
    return None


def detect_stagnation(
    rows: list[IterWR],
    state: WatchState,
    *,
    stagnation_iters: int,
    low_wr: float,
) -> Optional[dict[str, Any]]:
    """Low-WR plateau: stuck at/below ``low_wr`` with no best-WR improvement.

    Unlike sharp collapse (immediate), this only fires after
    ``stagnation_iters`` consecutive iterations with no ε improvement while
    both current WR and best WR remain in the low band (~≤10% by default).
    """
    if not rows:
        return None
    cur = rows[-1]
    if cur.wr >= low_wr:
        return None
    if state.best_wr >= low_wr:
        # Once past the low band we no longer fire the "stuck at ~10%" alarm.
        return None
    if state.iters_since_improve < stagnation_iters:
        return None
    return {
        "kind": "low_wr_stagnation",
        "iteration": cur.iteration,
        "wr": cur.wr,
        "best_wr": state.best_wr,
        "best_iter": state.best_iter,
        "iters_since_improve": state.iters_since_improve,
        "low_wr_threshold": low_wr,
        "stagnation_iters": stagnation_iters,
    }


def find_rr_pid(archetype: str) -> Optional[int]:
    """Best-effort: locate live train_round_robin for this archetype."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "train_round_robin.py"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    needle = f"--archetype {archetype}"
    for line in out.splitlines():
        if "winrate_watcher" in line:
            continue
        if needle in line or archetype in line:
            try:
                return int(line.split(None, 1)[0])
            except (ValueError, IndexError):
                continue
    return None


def write_alert_artifacts(
    *,
    alert: dict[str, Any],
    rows: list[IterWR],
    state: WatchState,
    archetype: str,
    log: ActionLogger,
) -> Path:
    good = resolve_good_checkpoint(rows, archetype)
    rr_pid = find_rr_pid(archetype)
    cur = rows[-1] if rows else None
    trajectory = [
        {
            "iteration": r.iteration,
            "wr": round(r.wr, 6),
            "n_games": r.n_games,
            "pass": f"{r.n_passing_wilson}/{r.n_evaluated}" if r.n_evaluated else "",
            "source": r.source,
        }
        for r in rows[-30:]
    ]
    payload = {
        "ts": _now(),
        "unix": time.time(),
        "alert": alert,
        "current": asdict(cur) if cur else None,
        "best_wr": state.best_wr,
        "best_iter": state.best_iter,
        "iters_since_improve": state.iters_since_improve,
        "trajectory_tail": trajectory,
        "recommended_checkpoint": str(good),
        "bootstrap_best": str(CHECKPOINTS_DIR / f"{archetype}_bootstrap.best.pt"),
        "rr_pid": rr_pid,
        "operator_hint": (
            "Do NOT kill mid-eval lightly. Prefer: pause after current iter, "
            f"resume from recommended_checkpoint with "
            f"`--resume auto` / `--bootstrap-ckpt {good}` once the adjuster "
            "or operator confirms. Spawned Cursor agent should diagnose and "
            "apply a safe fix (hyperparams, empty-experience bugs, champion "
            "rollback) rather than blindly SIGKILL."
        ),
        "restart_hint": (
            f"CUDA_DEVICE_ORDER=PCI_BUS_ID POKEBOT_PRIMARY_ARCHETYPE={archetype} "
            f"python scripts/train_round_robin.py --archetype {archetype} "
            f"--bootstrap-ckpt {good} --resume auto "
            f"# only after stopping RR cleanly at an iter boundary"
        ),
    }
    text = json.dumps(payload, indent=2) + "\n"
    _atomic_write(ALERT_FLAG, text)
    _atomic_write(STALL_FLAG, text)
    log(
        f"ALERT artifacts written: {ALERT_FLAG.name} + {STALL_FLAG.name} "
        f"kind={alert.get('kind')} recommended_ckpt={good}"
    )
    return good


def build_adjuster_prompt(
    *,
    alert: dict[str, Any],
    rows: list[IterWR],
    state: WatchState,
    archetype: str,
    rr_log: Path,
    good_ckpt: Path,
) -> str:
    traj_lines = []
    for r in rows[-25:]:
        traj_lines.append(
            f"  iter={r.iteration:03d} wr={r.wr:.1%} games={r.n_games} "
            f"draw_aware_pass={r.n_passing_wilson}/{r.n_evaluated or '?'} "
            f"src={r.source}"
        )
    log_snip = ""
    if rr_log.exists():
        try:
            size = rr_log.stat().st_size
            with open(rr_log, "rb") as fh:
                if size > 12000:
                    fh.seek(size - 12000)
                    fh.readline()
                log_snip = fh.read().decode("utf-8", errors="replace")
        except OSError:
            log_snip = ""
    recent_eval = ""
    for r in reversed(rows):
        if r.source == "eval" and r.path:
            try:
                # Keep prompt bounded — first/last bits of the JSON only.
                raw = Path(r.path).read_text(encoding="utf-8")
                if len(raw) > 8000:
                    recent_eval = raw[:4000] + "\n...\n" + raw[-3000:]
                else:
                    recent_eval = raw
            except OSError:
                recent_eval = ""
            break

    rr_pid = find_rr_pid(archetype)
    return f"""You are an on-call RL adjuster for the poke-bot-agent repo at {REPO_ROOT}.

ALERT: hammer / {archetype} round-robin training is failing the win-rate progression assertion.
Kind: {alert.get('kind')}
Alert details (JSON):
{json.dumps(alert, indent=2)}

Metrics:
- current best WR: {state.best_wr:.4f} (iter {state.best_iter})
- iters since last improvement (≥ε): {state.iters_since_improve}
- latest: {"iter=%d WR=%.4f" % (rows[-1].iteration, rows[-1].wr) if rows else "n/a"}
- recommended good checkpoint: {good_ckpt}
- bootstrap.best: {CHECKPOINTS_DIR / f'{archetype}_bootstrap.best.pt'}
- RR log: {rr_log}
- eval dumps: {EVAL_DIR}/rr_iter*.json
- live RR pid (best-effort): {rr_pid}

WR trajectory (tail):
{chr(10).join(traj_lines) if traj_lines else '  (none yet)'}

YOUR JOB (do this now, do not just write notes and stop):
1. Diagnose WHY win rate is stagnating / collapsing (empty experience, fake games,
   MCTS/leaf-server wiring, overly aggressive train step, wrong resume, broken baselines, etc.).
2. Apply a SAFE fix so training resumes maximizing win rate:
   - Prefer rolling the champion / restart base back to bootstrap.best or last-good
     best_champion ({good_ckpt}) if the policy collapsed.
   - Adjust hyperparams (train-epochs, mcts-sims, bootstrap-mix, regression margin,
     games-per-opp) if clearly warranted by evidence.
   - Fix code bugs if experience is empty / games are fail-closed / WR logging is wrong.
3. Do NOT needlessly kill a healthy mid-iteration collect. If a restart is required,
   stop RR cleanly at an iter boundary and relaunch with `--resume auto` from the
   good checkpoint. Document what you did in outputs/logs/winrate_watcher.log (append)
   and clear or update {ALERT_FLAG} / {STALL_FLAG} when healthy progress resumes.
4. Leave the run progressive: subsequent iterations must be able to beat best WR.

Recent RR log tail:
```
{log_snip[-8000:]}
```

Recent eval JSON (truncated):
```
{recent_eval[:7000]}
```

Python ONLY: /home/pokebot/miniconda3/envs/poke-bot-agent/bin/python
Do not commit. Coordinate with any live RR / resource watchers — additive/safe interventions only.
"""


def adjuster_inflight() -> Optional[int]:
    if not ADJUSTER_LOCK.exists():
        return None
    try:
        text = ADJUSTER_LOCK.read_text(encoding="utf-8")
    except OSError:
        return None
    pid = None
    for line in text.splitlines():
        if line.startswith("pid="):
            try:
                pid = int(line.split("=", 1)[1])
            except ValueError:
                pid = None
    if pid is not None and _pid_alive(pid):
        return pid
    return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def spawn_adjuster(
    *,
    prompt: str,
    log: ActionLogger,
    state: WatchState,
) -> bool:
    existing = adjuster_inflight()
    if existing is not None:
        log(f"adjuster SKIP: already in-flight pid={existing}")
        return False

    if not SPAWN_SCRIPT.exists():
        log(f"adjuster BLOCKED: spawn script missing at {SPAWN_SCRIPT}")
        return False

    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_path = PROMPT_DIR / f"prompt_{ts}.txt"
    _atomic_write(prompt_path, prompt)

    # Detach: watcher must keep polling; adjuster can run for a long time.
    cmd = [
        "bash",
        str(SPAWN_SCRIPT),
        str(prompt_path),
        str(DEFAULT_SELF_LOG),
    ]
    try:
        # Open nohup-style sink so the child isn't tied to our stdio.
        sink = open(LOG_DIR / "wr_adjuster.spawn.out", "a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "WR_ADJUSTER_PREFERRED_MODEL": os.environ.get(
                    "WR_ADJUSTER_PREFERRED_MODEL", "gpt-5.6-sol-max-fast"
                ),
                "WR_ADJUSTER_FALLBACK_MODEL": os.environ.get(
                    "WR_ADJUSTER_FALLBACK_MODEL", "auto"
                ),
                "WR_ADJUSTER_WORKSPACE": str(REPO_ROOT),
                "WR_ADJUSTER_LOCK": str(ADJUSTER_LOCK),
            },
        )
        sink.close()
    except OSError as exc:
        log(f"adjuster SPAWN FAILED: {exc!r}")
        return False

    state.adjuster_spawns += 1
    log(
        f"adjuster SPAWNED pid={proc.pid} prompt={prompt_path.name} "
        f"preferred=gpt-5.6-sol-max-fast fallback=auto "
        f"(spawn_count={state.adjuster_spawns})"
    )
    return True


def maybe_alert(
    *,
    alert: dict[str, Any],
    rows: list[IterWR],
    state: WatchState,
    archetype: str,
    rr_log: Path,
    log: ActionLogger,
    spawn: bool,
    alert_cooldown_iters: int,
) -> None:
    kind = str(alert.get("kind"))
    cur_it = rows[-1].iteration if rows else -1
    # Dedup: same kind on the same iter, or within cooldown of last alert.
    if (
        state.last_alert_kind == kind
        and state.last_alert_iter >= 0
        and cur_it >= 0
        and (cur_it - state.last_alert_iter) < alert_cooldown_iters
    ):
        log(
            f"alert suppressed (cooldown): kind={kind} cur_iter={cur_it} "
            f"last_alert_iter={state.last_alert_iter}"
        )
        return

    banner = "!" * 72
    log(banner)
    log(f"!!! WINRATE ALERT kind={kind} iter={cur_it} details={json.dumps(alert)}")
    log(banner)

    good = write_alert_artifacts(
        alert=alert, rows=rows, state=state, archetype=archetype, log=log
    )
    state.last_alert_kind = kind
    state.last_alert_iter = cur_it
    state.last_alert_ts = time.time()

    if not spawn:
        log("adjuster spawn disabled (--no-spawn)")
        return

    prompt = build_adjuster_prompt(
        alert=alert,
        rows=rows,
        state=state,
        archetype=archetype,
        rr_log=rr_log,
        good_ckpt=good,
    )
    spawn_adjuster(prompt=prompt, log=log, state=state)


def poll_once(
    *,
    eval_dir: Path,
    rr_log: Path,
    state_path: Path,
    archetype: str,
    stagnation_iters: int,
    low_wr: float,
    min_improve: float,
    collapse_drop: float,
    spawn: bool,
    alert_cooldown_iters: int,
    log: ActionLogger,
) -> WatchState:
    rows = merge_iters(load_eval_iters(eval_dir), scrape_log_iters(rr_log))
    # Recompute from trajectory so watcher restarts stay consistent.
    fresh = rebuild_progress(rows, min_improve)
    prev = load_state(state_path)
    # Preserve alert / spawn bookkeeping across recomputes.
    fresh.last_alert_kind = prev.last_alert_kind
    fresh.last_alert_iter = prev.last_alert_iter
    fresh.last_alert_ts = prev.last_alert_ts
    fresh.adjuster_spawns = prev.adjuster_spawns

    if not rows:
        log("poll: no rr_iter*.json / log WR lines yet — waiting")
        save_state(state_path, fresh)
        return fresh

    cur = rows[-1]
    trend = " → ".join(f"{r.iteration}:{r.wr:.1%}" for r in rows[-8:])
    log(
        f"poll: iter={cur.iteration} wr={cur.wr:.1%} "
        f"(games={cur.n_games} draw-aware-pass="
        f"{cur.n_passing_wilson}/{cur.n_evaluated}) "
        f"best={fresh.best_wr:.1%}@iter{fresh.best_iter} "
        f"since_improve={fresh.iters_since_improve} "
        f"trend[{trend}]"
    )

    collapse = detect_collapse(rows, collapse_drop)
    if collapse is not None:
        maybe_alert(
            alert=collapse,
            rows=rows,
            state=fresh,
            archetype=archetype,
            rr_log=rr_log,
            log=log,
            spawn=spawn,
            alert_cooldown_iters=alert_cooldown_iters,
        )
    else:
        stagn = detect_stagnation(
            rows,
            fresh,
            stagnation_iters=stagnation_iters,
            low_wr=low_wr,
        )
        if stagn is not None:
            maybe_alert(
                alert=stagn,
                rows=rows,
                state=fresh,
                archetype=archetype,
                rr_log=rr_log,
                log=log,
                spawn=spawn,
                alert_cooldown_iters=alert_cooldown_iters,
            )

    save_state(state_path, fresh)
    return fresh


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archetype", default="hammer-pult")
    p.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    p.add_argument("--rr-log", type=Path, default=DEFAULT_RR_LOG)
    p.add_argument("--self-log", type=Path, default=DEFAULT_SELF_LOG)
    p.add_argument("--state", type=Path, default=DEFAULT_STATE)
    p.add_argument(
        "--stagnation-iters",
        type=int,
        default=DEFAULT_STAGNATION_ITERS,
        help="Consecutive no-improve iters in the low-WR band before alerting "
             f"(default: {DEFAULT_STAGNATION_ITERS}). Sharp collapse is independent.",
    )
    p.add_argument(
        "--low-wr-threshold",
        type=float,
        default=DEFAULT_LOW_WR,
        help="Low-WR plateau band: fire stagnation only when WR and best WR are "
             f"below this (default: {DEFAULT_LOW_WR} ≈ 10%).",
    )
    p.add_argument("--min-improve", type=float, default=DEFAULT_MIN_IMPROVE)
    p.add_argument(
        "--collapse-drop",
        type=float,
        default=DEFAULT_COLLAPSE_DROP,
        help="Immediate sharp-collapse alert: absolute WR drop over 1–2 iters "
             f"(default: {DEFAULT_COLLAPSE_DROP}).",
    )
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    p.add_argument(
        "--alert-cooldown-iters",
        type=int,
        default=5,
        help="Suppress re-alerting the same kind within this many new iterations.",
    )
    p.add_argument("--once", action="store_true", help="Single poll then exit.")
    p.add_argument(
        "--no-spawn",
        action="store_true",
        help="Write alert flags but do not spawn the Cursor adjuster.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    log = ActionLogger(args.self_log)
    stop = {"flag": False}

    def _stop(signum, _frame):  # noqa: ANN001
        stop["flag"] = True
        log(f"received signal {signum}; stopping after current poll")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    log(
        f"winrate_watcher start pid={os.getpid()} archetype={args.archetype} "
        f"eval={args.eval_dir} rr_log={args.rr_log} "
        f"stagnation_iters={args.stagnation_iters} low_wr={args.low_wr_threshold} "
        f"min_improve={args.min_improve} collapse_drop={args.collapse_drop} "
        f"interval={args.interval}s spawn={not args.no_spawn} "
        f"preferred_model=gpt-5.6-sol-max-fast fallback=auto"
    )

    while not stop["flag"]:
        try:
            poll_once(
                eval_dir=args.eval_dir,
                rr_log=args.rr_log,
                state_path=args.state,
                archetype=args.archetype,
                stagnation_iters=args.stagnation_iters,
                low_wr=args.low_wr_threshold,
                min_improve=args.min_improve,
                collapse_drop=args.collapse_drop,
                spawn=not args.no_spawn,
                alert_cooldown_iters=args.alert_cooldown_iters,
                log=log,
            )
        except Exception as exc:  # never die on a poll error
            log(f"poll error (continuing): {exc!r}")
        if args.once or stop["flag"]:
            break
        # Sleep in chunks so SIGTERM is responsive.
        end = time.time() + max(1.0, float(args.interval))
        while time.time() < end and not stop["flag"]:
            time.sleep(min(1.0, end - time.time()))

    log("winrate_watcher stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
