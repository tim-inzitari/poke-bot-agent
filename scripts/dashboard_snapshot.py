#!/usr/bin/env python3
"""Emit one lightweight JSON snapshot for the LAN training dashboard."""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


SERVICE = "pokemon-state-bootstrap.service"
EXACT_SERVICE = "pokemon-privileged-belief-full-blackwell-v1.service"
LATEST10_BOOTSTRAP_SERVICE = "pokemon-latest10-bootstrap.service"
LATEST10_FINALIZER_SERVICE = "pokemon-latest10-finalize.service"
CORE_RL_SERVICE = "pokebot-pure-rl-continuous-rehearsal.service"
ALAKAZAM_BOOTSTRAP_SERVICE = "pokebot-pure-rl-alakazam-bootstrap.service"
ALAKAZAM_SPECIALIST_SERVICE = "pokebot-pure-rl-alakazam.service"
ROOT = Path("/home/inzi/poke-bot-agent")
BOOTSTRAP_LOG = ROOT / "outputs/logs/bootstrap.log"
ALAKAZAM_BOOTSTRAP_LOG = ROOT / "outputs/logs/alakazam_expert_bootstrap.log"
ALAKAZAM_TRANSITION_LOG = ROOT / "outputs/logs/deck_agnostic_core_transition.log"
ALAKAZAM_TRANSITION_STATE = (
    ROOT / "outputs/state/deck-agnostic-core-transition.json"
)
ALAKAZAM_BUILD_READY = ROOT / "outputs/state/alakazam-specialist-build-ready.json"
ALAKAZAM_BOOTSTRAP_READY = (
    ROOT / "outputs/state/alakazam-expert-bootstrap-ready.json"
)
EXACT_LOG = ROOT / "outputs/logs/privileged-belief-full-blackwell.log"
EXACT_ROOT = ROOT / "outputs/privileged_belief/exact_core_20k_v1"
EXACT_RESIDENT_STATUS = EXACT_ROOT / "resident_train.status.json"
EXACT_STREAM_STATUS = EXACT_ROOT / "full_train.status.json"
LATEST10_FINALIZER_LOG = ROOT / "outputs/logs/latest10-finalize.log"
LATEST10_READY = (
    ROOT / "data/bootstrap/latest10-20260709-20260718/READY.json"
)
LATEST10_BERT_STATUS = (
    ROOT / "data/bootstrap/latest10-20260709-20260718/bert-staging-status.json"
)
TRAINING_STATUS = ROOT / "outputs/logs/training.progress.status"
TRAINING_LOG = ROOT / "outputs/logs/training.log"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LATEST10_STATUS = ROOT / "scripts/latest10_status.py"
DASHBOARD_ITERATION_TIMER = ROOT / "outputs/state/dashboard_iteration_timer.json"
MODEL_PROFILE_REGISTRY = ROOT / "outputs/state/pure_rl_model_profiles.json"
PUBLIC_MIX_LIVE_WR = ROOT / "outputs/state/public_mix_live_wr.json"


def run(argv: list[str], timeout: float = 3.0) -> str:
    try:
        result = subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def as_number(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def process_rows() -> dict[int, tuple[int, float, int, str]]:
    """Return pid -> (ppid, cpu%, rss KiB, command) for one cheap snapshot."""
    raw = run(["ps", "-eo", "pid=,ppid=,pcpu=,rss=,args="], timeout=5)
    rows: dict[int, tuple[int, float, int, str]] = {}
    for line in raw.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) != 5:
            continue
        try:
            rows[int(parts[0])] = (
                int(parts[1]),
                float(parts[2]),
                int(parts[3]),
                parts[4],
            )
        except ValueError:
            continue
    return rows


def _unit_values(name: str, *, user: bool = False) -> dict[str, str]:
    argv = ["systemctl"]
    if user:
        argv.append("--user")
    argv.extend(
        [
            "show",
            name,
            "--property=MainPID,ControlGroup,MemoryCurrent,TasksCurrent,Environment",
        ]
    )
    values: dict[str, str] = {}
    for line in run(argv, timeout=4).splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    return values


def _environment_values(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()
    for token in tokens:
        key, sep, value = token.partition("=")
        if sep:
            values[key] = value
    return values


def _cgroup_pids(control_group: str) -> set[int]:
    if not control_group.startswith("/") or ".." in control_group.split("/"):
        return set()
    path = Path("/sys/fs/cgroup") / control_group.lstrip("/") / "cgroup.procs"
    try:
        return {int(line) for line in path.read_text().splitlines() if line.strip()}
    except (OSError, ValueError):
        return set()


def curriculum_worker_state(
    active_units: list[str], active_pids: list[int]
) -> dict[str, Any]:
    """Aggregate the live user-service cgroup rather than guessing names.

    Pure-RL workers are multiprocessing children whose command line is only
    ``spawn_main``.  A process-name filter therefore misses almost the whole
    tree; systemd's control group is the authoritative membership boundary.
    """
    rows = process_rows()
    selected: set[int] = set()
    memory_current = 0
    tasks_current = 0
    environment: dict[str, str] = {}
    for unit in active_units:
        values = _unit_values(unit, user=True)
        selected.update(_cgroup_pids(values.get("ControlGroup", "")))
        memory_current += as_number(values.get("MemoryCurrent", "")) or 0
        tasks_current += as_number(values.get("TasksCurrent", "")) or 0
        environment.update(_environment_values(values.get("Environment", "")))

    # Older/non-systemd test environments may not expose cgroup.procs. Fall
    # back to a complete descendant closure from the unit MainPID(s).
    if not selected:
        selected.update(pid for pid in active_pids if pid > 0)
        changed = True
        while changed:
            changed = False
            for pid, (ppid, _cpu, _rss, _command) in rows.items():
                if ppid in selected and pid not in selected:
                    selected.add(pid)
                    changed = True

    cpu_percent = sum(rows[pid][1] for pid in selected if pid in rows)
    rss_bytes = sum(rows[pid][2] for pid in selected if pid in rows) * 1024
    root_pid = next((pid for pid in active_pids if pid in rows), 0)
    command = rows[root_pid][3] if root_pid else ""
    workers = as_number(environment.get("PURE_RL_SIM_WORKERS", ""))
    leaves0 = as_number(environment.get("PURE_RL_LEAF_GPU0_REPLICAS", "")) or 0
    leaves1 = as_number(environment.get("PURE_RL_LEAF_GPU1_REPLICAS", "")) or 0
    multi_env = as_number(environment.get("POKEBOT_MULTI_ENV_PER_WORKER", ""))
    return {
        "active": bool(active_units and selected),
        "listening": None,
        "controller_pids": list(active_pids),
        "processes": len(selected),
        "tasks": tasks_current or None,
        "workers": workers,
        "multi_env_per_worker": multi_env,
        "leaf_servers": leaves0 + leaves1,
        "leaf_gpu0_replicas": leaves0,
        "leaf_gpu1_replicas": leaves1,
        "cpu_percent": cpu_percent,
        "rss_bytes": memory_current or rss_bytes,
        "command": command or ", ".join(active_units),
        "source": "systemd-user-cgroup",
    }


def process_rss_bytes(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def unit_state(name: str, *, user: bool = False) -> dict[str, Any]:
    argv = ["systemctl"]
    if user:
        argv.append("--user")
    argv.extend(
        [
            "show",
            name,
            "--property=ActiveState,SubState,MainPID,MemoryCurrent,MemoryPeak,CPUUsageNSec,ExecMainStartTimestamp",
        ]
    )
    raw = run(argv)
    values: dict[str, str] = {}
    for line in raw.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    return {
        "name": name,
        "active": values.get("ActiveState") in {"active", "activating"},
        "active_state": values.get("ActiveState", "not-found"),
        "sub_state": values.get("SubState", "dead"),
        "pid": as_number(values.get("MainPID", "0")) or 0,
        "memory_bytes": as_number(values.get("MemoryCurrent", "")),
        "memory_peak_bytes": as_number(values.get("MemoryPeak", "")),
        "cpu_ns": as_number(values.get("CPUUsageNSec", "")),
        "started": values.get("ExecMainStartTimestamp", ""),
    }


def service_state() -> dict[str, Any]:
    candidates = [
        unit_state(ALAKAZAM_SPECIALIST_SERVICE, user=True),
        unit_state(ALAKAZAM_BOOTSTRAP_SERVICE, user=True),
        unit_state(CORE_RL_SERVICE, user=True),
        unit_state(EXACT_SERVICE),
        unit_state(LATEST10_BOOTSTRAP_SERVICE),
        unit_state(SERVICE),
    ]
    values = next((row for row in candidates if row["active"]), candidates[-1])
    service_pid = int(values["pid"] or 0)
    raw_pid = run(
        [
            "pgrep",
            "-fo",
            "train_privileged_belief_resident.py|train_privileged_belief_shards.py|train_bootstrap.py",
        ]
    )
    trainer_pid = as_number(raw_pid) or 0
    pid = trainer_pid or service_pid
    fallback = not service_pid and bool(trainer_pid)
    command = ""
    if pid:
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace"
            ).strip()
        except OSError:
            pass
    return {
        "name": values["name"],
        "active": values["active"] or fallback,
        "active_state": "process-fallback" if fallback else values["active_state"],
        "sub_state": "running" if fallback else values["sub_state"],
        "pid": pid,
        "supervisor_pid": service_pid,
        "memory_bytes": process_rss_bytes(pid) if fallback else values["memory_bytes"],
        "memory_peak_bytes": values["memory_peak_bytes"],
        "cpu_ns": values["cpu_ns"],
        "started": values["started"],
        "command": command,
    }


def transition_state() -> dict[str, Any]:
    """Return a compact, current view of the core-to-Alakazam handoff."""
    raw = read_json(ALAKAZAM_TRANSITION_STATE)
    status = str(raw.get("status") or "waiting")
    bootstrap = unit_state(ALAKAZAM_BOOTSTRAP_SERVICE, user=True)
    specialist = unit_state(ALAKAZAM_SPECIALIST_SERVICE, user=True)
    core = unit_state(CORE_RL_SERVICE, user=True)
    labels = {
        "training_alakazam_expert_bootstrap_blackwell_device_resident": (
            "Deck Agnostic Core → Alakazam · expert bootstrap on Blackwell"
        ),
        "alakazam_specialist_bootstrap_ready_launching": (
            "Alakazam bootstrap ready · launching specialist RL"
        ),
        "launching_alakazam_specialist": "Launching Alakazam specialist RL fleet",
        "complete": "Alakazam specialist RL · transition complete",
    }
    decision = raw.get("decision") if isinstance(raw.get("decision"), dict) else {}
    best = decision.get("best") if isinstance(decision.get("best"), dict) else {}
    triggered = raw.get("triggered") is True or decision.get("triggered") is True
    updated = None
    try:
        updated = ALAKAZAM_TRANSITION_STATE.stat().st_mtime
    except OSError:
        pass
    bootstrap_running = bool(
        bootstrap.get("active")
        and (
            int(bootstrap.get("pid") or 0) > 0
            or bootstrap.get("sub_state") == "running"
        )
    )
    active = bool(
        bootstrap_running
        or (
            triggered
            and status not in {"complete", "specialist_preparation_failed_core_continues"}
        )
    )
    return {
        "available": bool(raw),
        "active": active,
        "triggered": triggered,
        "status": status,
        "label": labels.get(status, status.replace("_", " ").strip().title()),
        "reason": decision.get("reason") or raw.get("handoff_wait_reason"),
        "source": str(ALAKAZAM_TRANSITION_STATE),
        "updated_at": updated,
        "core_iteration": best.get("iteration"),
        "core_win_rate": best.get("win_rate"),
        "bootstrap": bootstrap,
        "specialist": specialist,
        "core": core,
    }


def read_tail(path: Path, max_bytes: int = 1_000_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def parse_metric(line: str, name: str) -> float | None:
    match = re.search(rf"(?:^|[ ,]){re.escape(name)}=(-?[0-9.]+)%?", line)
    return float(match.group(1)) if match else None


def bootstrap_progress() -> dict[str, Any]:
    raw = read_tail(BOOTSTRAP_LOG)
    clean = ANSI_RE.sub("", raw).replace("\r", "\n")
    marker = clean.rfind("== train_bootstrap")
    if marker >= 0:
        clean = clean[marker:]
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    latest = ""
    for line in reversed(lines):
        if "train ep" in line and re.search(r"\d+/\d+", line):
            latest = line
            break
    if not latest:
        for line in reversed(lines):
            if "featurize " in line and "seq" in line:
                latest = line
                break
    if not latest and lines:
        latest = lines[-1]

    result: dict[str, Any] = {
        "log": str(BOOTSTRAP_LOG),
        "latest_line": latest,
        "log_exists": BOOTSTRAP_LOG.exists(),
        "log_bytes": BOOTSTRAP_LOG.stat().st_size if BOOTSTRAP_LOG.exists() else 0,
        "updated_at": BOOTSTRAP_LOG.stat().st_mtime if BOOTSTRAP_LOG.exists() else None,
        "epoch": None,
        "current": None,
        "total": None,
        "percent": None,
        "batch_per_second": None,
        "eta": None,
        "metrics": {},
        "phase": "loading",
        "sequences": None,
        "sequences_per_second": None,
    }
    match = re.search(r"train ep(\d+):\s*(\d+)%.*?\s(\d+)/(\d+)\s*\[([^]]*)\]", latest)
    if match:
        epoch, percent, current, total, timing = match.groups()
        result.update(
            epoch=int(epoch) + 1,
            current=int(current),
            total=int(total),
            percent=float(percent),
        )
        rate = re.search(r"([0-9.]+)batch/s", timing)
        if rate:
            result["batch_per_second"] = float(rate.group(1))
        eta = re.search(r"<([^,]+),", timing)
        if eta:
            result["eta"] = eta.group(1)
        result["phase"] = "training"
    featurize = re.search(r"featurize\s+.*?:\s*(\d+)seq\s+\[[^,]+,\s*([0-9.]+)seq/s\]", latest)
    if featurize:
        result["phase"] = "featurizing"
        result["sequences"] = int(featurize.group(1))
        result["sequences_per_second"] = float(featurize.group(2))
    result["metrics"] = {
        name: parse_metric(latest, name)
        for name in ("acc", "loss", "p", "v", "step")
    }
    return result


def alakazam_bootstrap_progress() -> dict[str, Any]:
    """Parse the live, device-resident Alakazam expert bootstrap."""
    service = unit_state(ALAKAZAM_BOOTSTRAP_SERVICE, user=True)
    raw = read_tail(ALAKAZAM_BOOTSTRAP_LOG, 2_000_000)
    clean = ANSI_RE.sub("", raw).replace("\r", "\n")
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    latest = ""
    for line in reversed(lines):
        if re.search(r"(?:train|val) ep\d+:.*?\d+/\d+", line):
            latest = line
            break
    if not latest:
        for line in reversed(lines):
            if "pack Blackwell corpus" in line or line.startswith("[train] device="):
                latest = line
                break
    if not latest and lines:
        latest = lines[-1]

    build = read_json(ALAKAZAM_BUILD_READY)
    corpus = build.get("expert_corpus") if isinstance(build.get("expert_corpus"), dict) else {}
    corpus_games = as_number(str(corpus.get("records") or "")) or 39_467
    corpus_decisions = as_number(str(corpus.get("decisions") or "")) or 2_579_178
    train_games = int(round(corpus_games * 0.90))
    train_decisions = int(round(corpus_decisions * 0.90))
    phase = "loading"
    epoch = None
    current = None
    total = None
    percent = None
    batch_rate = None
    eta = None

    match = re.search(
        r"(train|val) ep(\d+):\s*(\d+)%.*?\s(\d+)/(\d+)\s*\[([^]]*)\]",
        latest,
    )
    split = "train"
    if match:
        split, epoch_raw, percent_raw, current_raw, total_raw, timing = match.groups()
        phase = "training" if split == "train" else "validation"
        epoch = int(epoch_raw) + 1
        percent = float(percent_raw)
        current = int(current_raw)
        total = int(total_raw)
        rate = re.search(r"([0-9.]+)batch/s", timing)
        if rate:
            batch_rate = float(rate.group(1))
        eta_match = re.search(r"<([^,]+),", timing)
        if eta_match:
            eta = eta_match.group(1)
    elif "pack Blackwell corpus" in latest:
        phase = "packing"
        match = re.search(r"(\d+)%.*?\s(\d+)/(\d+)\s", latest)
        if match:
            percent, current, total = (
                float(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )

    # Derive rates only from the measured batch rate and the exact corpus
    # partition. This is an epoch-average conversion, not a fabricated game
    # simulator rate.
    samples_per_second = None
    game_equivalents_per_second = None
    if batch_rate is not None and total:
        denominator = int(total)
        split_games = train_games if split == "train" else corpus_games - train_games
        split_decisions = (
            train_decisions if split == "train" else corpus_decisions - train_decisions
        )
        samples_per_second = batch_rate * split_decisions / denominator
        game_equivalents_per_second = batch_rate * split_games / denominator

    updated = None
    try:
        updated = ALAKAZAM_BOOTSTRAP_LOG.stat().st_mtime
    except OSError:
        pass
    ready = ALAKAZAM_BOOTSTRAP_READY.is_file()
    status = "complete" if ready else "running" if service.get("active") else "waiting"
    metrics = {
        name: parse_metric(latest, name)
        for name in ("acc", "loss", "p", "v", "step")
    }
    return {
        "authoritative": True,
        "source": str(ALAKAZAM_BOOTSTRAP_LOG),
        "log": str(ALAKAZAM_BOOTSTRAP_LOG),
        "latest_line": latest,
        "updated_at": updated,
        "fresh": bool(updated and time.time() - updated < 30),
        "status": status,
        "mode": "alakazam_expert_bootstrap_device_resident",
        "phase": "complete" if ready else phase,
        "epoch": epoch,
        "epochs_target": 25,
        "current": current,
        "total": total,
        "percent": 100.0 if ready else percent,
        "batch_per_second": batch_rate,
        "samples_per_second": samples_per_second,
        "game_equivalents_per_second": game_equivalents_per_second,
        "acting_sequences_per_second": game_equivalents_per_second,
        "corpus_games": corpus_games,
        "corpus_records": corpus_games,
        "corpus_decisions": corpus_decisions,
        "eta": eta,
        "metrics": metrics,
        "all_training_tensors_device_resident": phase in {"training", "validation"},
        "gpu_name": "NVIDIA RTX PRO 5000 Blackwell",
        "service": service,
    }


def exact_training_state() -> dict[str, Any]:
    """Return the authoritative active exact-replay training state."""
    candidates = [
        (EXACT_RESIDENT_STATUS, read_json(EXACT_RESIDENT_STATUS)),
        (EXACT_STREAM_STATUS, read_json(EXACT_STREAM_STATUS)),
    ]
    candidates = [row for row in candidates if row[1]]
    if not candidates:
        legacy = bootstrap_progress()
        legacy.update(authoritative=False, source=str(BOOTSTRAP_LOG))
        return legacy
    status_path, status = max(
        candidates,
        key=lambda row: float(row[1].get("updated_unix") or 0),
    )
    raw = ANSI_RE.sub("", read_tail(EXACT_LOG)).replace("\r", "\n")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    latest = lines[-1] if lines else ""
    resident = "resident" in str(status.get("mode") or "")
    phase = str(status.get("phase") or "training")
    current = status.get("current")
    total = status.get("total")
    percent = status.get("percent")
    if resident and phase == "packing":
        packing = next(
            (line for line in reversed(lines) if "pack exact Blackwell corpus" in line),
            "",
        )
        match = re.search(r"(\d+)%.*?\s(\d+)/(\d+)\s", packing)
        if match:
            percent, current, total = (
                float(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
            latest = packing
    if not resident:
        current = status.get("shard_cursor")
        total = status.get("train_shards")
        percent = (
            100.0 * float(current) / float(total)
            if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total
            else None
        )
    raw_metrics = status.get("metrics") or status.get("last_train") or {}
    corpus_manifest = read_json(EXACT_ROOT / "manifest.json")
    corpus_totals = corpus_manifest.get("totals") or {}
    corpus_games = corpus_totals.get("games")
    corpus_records = corpus_totals.get("records")
    batch_rate = status.get("batch_per_second")
    batch_size = status.get("batch_size")
    samples_per_second = (
        float(batch_rate) * float(batch_size)
        if isinstance(batch_rate, (int, float))
        and isinstance(batch_size, (int, float))
        else None
    )
    total_samples = (
        int(status.get("train_samples") or 0)
        + int(status.get("val_samples") or 0)
    )
    game_equivalents_per_second = (
        samples_per_second * float(corpus_games) / total_samples
        if samples_per_second is not None
        and isinstance(corpus_games, (int, float))
        and total_samples > 0
        else None
    )
    acting_sequences_per_second = (
        samples_per_second * float(corpus_records) / total_samples
        if samples_per_second is not None
        and isinstance(corpus_records, (int, float))
        and total_samples > 0
        else None
    )
    metrics = {
        "acc": (
            100.0 * float(raw_metrics["policy_accuracy"])
            if isinstance(raw_metrics.get("policy_accuracy"), (int, float))
            else None
        ),
        "loss": raw_metrics.get("total") or raw_metrics.get("objective"),
        "p": raw_metrics.get("policy"),
        "v": raw_metrics.get("value"),
        "aux": raw_metrics.get("aux"),
        "hand": raw_metrics.get("hand"),
        "remainder": raw_metrics.get("remainder"),
        "lethal": raw_metrics.get("lethal"),
        "prize_race": raw_metrics.get("prize_race"),
        "step": status.get("global_step"),
    }

    status_updated = float(status.get("updated_unix") or 0)
    try:
        log_updated = EXACT_LOG.stat().st_mtime
    except OSError:
        log_updated = 0.0
    updated = max(status_updated, log_updated) or None
    eta_seconds = status.get("eta_seconds")
    return {
        "authoritative": True,
        "source": str(status_path),
        "log": str(EXACT_LOG),
        "latest_line": latest,
        "updated_at": updated,
        "fresh": bool(updated and time.time() - updated < 30),
        "status": status.get("status"),
        "mode": status.get("mode"),
        "phase": phase,
        "epoch": status.get("total_epoch_index"),
        "epochs_target": status.get("epochs_target", 26),
        "current": current,
        "total": total,
        "percent": percent,
        "batch_per_second": batch_rate,
        "samples_per_second": samples_per_second,
        "game_equivalents_per_second": game_equivalents_per_second,
        "acting_sequences_per_second": acting_sequences_per_second,
        "corpus_games": corpus_games,
        "corpus_records": corpus_records,
        "eta_seconds": eta_seconds,
        "eta": (
            time.strftime("%H:%M:%S", time.gmtime(float(eta_seconds)))
            if isinstance(eta_seconds, (int, float))
            else None
        ),
        "metrics": metrics,
        "all_training_tensors_device_resident": status.get(
            "all_training_tensors_device_resident", False
        ),
        "resident_bytes": status.get("resident_bytes"),
        "train_samples": status.get("train_samples"),
        "val_samples": status.get("val_samples"),
        "batch_size": batch_size,
        "gpu_name": status.get("gpu_name"),
    }


def baseline_eval_state() -> dict[str, Any]:
    candidates = sorted(
        (ROOT / "outputs/eval").glob(
            "state_core_resident_epoch*_official_core17.json"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return {"available": False}
    path = candidates[-1]
    report = read_json(path)
    pooled = report.get("pooled_formal") or {}
    checkpoint_info = report.get("checkpoint") or {}
    matchups = [
        {
            "opponent_id": row.get("opponent_id"),
            "games": row.get("games"),
            "wr": row.get("wr"),
            "lower": (row.get("draw_aware_score_interval") or {}).get("lower"),
        }
        for row in report.get("matchups") or []
    ]
    return {
        "available": True,
        "source": str(path),
        "updated_at": path.stat().st_mtime,
        "valid": report.get("valid"),
        "passed": report.get("all_pass"),
        "promotion_eligible": report.get("promotion_eligible"),
        "games": pooled.get("games"),
        "wr": pooled.get("wr"),
        "lower": pooled.get("interval_lower"),
        "upper": pooled.get("interval_upper"),
        "checkpoint": checkpoint_info.get("path"),
        "checkpoint_digest": checkpoint_info.get("digest"),
        "matchups": matchups,
        "deck_count": (report.get("deck_agnostic_gate") or {}).get("deck_count"),
    }


def committed_official_heldout_state(
    loop: dict[str, Any],
    run_dir: Path | None,
    *,
    global_iteration_offset: int = 0,
    handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the best append-only, exact official heldout result.

    The curriculum's ``metrics/latest.json`` describes the latest candidate,
    which may be worse than the protected heldout champion.  The Official
    Gateline card instead follows ``heldout_champion_evidence`` and reconciles
    it to the matching committed history row before exposing matchup results.
    This keeps a live/partial heldout wave from replacing audited evidence.
    """

    evidence = loop.get("heldout_champion_evidence")
    identity = loop.get("heldout_champion")
    if not isinstance(evidence, dict) or not isinstance(identity, dict):
        inherited = (
            handoff.get("inherited_official_heldout")
            if isinstance(handoff, dict)
            else None
        )
        inherited_identity = (
            loop.get("heldout_champion") or loop.get("champion")
            if isinstance(loop, dict)
            else None
        )
        if not isinstance(inherited, dict) or not isinstance(inherited_identity, dict):
            return {"available": False}
        digest = str(inherited.get("checkpoint_digest") or "")
        if not digest or digest != str(inherited_identity.get("digest") or ""):
            return {"available": False, "reason": "inherited heldout identity mismatch"}
        audit = inherited.get("audit")
        gate_matchups = inherited.get("per_opponent")
        audit_matchups = audit.get("per_opponent") if isinstance(audit, dict) else None
        games = int(inherited.get("games") or 0)
        if (
            not isinstance(audit, dict)
            or audit.get("passed") is not True
            or audit.get("exact_distribution") is not True
            or audit.get("exact_weights") is not True
            or audit.get("greedy_required") is not True
            or int(audit.get("valid_games") or 0) != games
            or not isinstance(gate_matchups, dict)
            or not isinstance(audit_matchups, dict)
        ):
            return {"available": False, "reason": "inherited heldout audit mismatch"}
        matchups: list[dict[str, Any]] = []
        for opponent_id, row in gate_matchups.items():
            if not isinstance(row, dict):
                return {"available": False, "reason": "inherited heldout matchup mismatch"}
            seats = audit_matchups.get(opponent_id)
            if not isinstance(seats, dict):
                return {"available": False, "reason": "inherited heldout matchup mismatch"}
            matchup_games = int(row.get("games") or 0)
            seat0 = int(seats.get("seat0") or row.get("seat0_games") or 0)
            seat1 = int(seats.get("seat1") or row.get("seat1_games") or 0)
            if matchup_games <= 0 or seat0 + seat1 != matchup_games:
                return {"available": False, "reason": "inherited heldout seat mismatch"}
            matchups.append(
                {
                    "opponent_id": str(opponent_id),
                    "games": matchup_games,
                    "wr": as_float(row.get("win_rate")),
                    "wins": as_float(row.get("wins")),
                    "draws": as_float(row.get("draws")),
                    "losses": as_float(row.get("losses")),
                    "seat0": seat0,
                    "seat1": seat1,
                }
            )
        if sum(int(row["games"]) for row in matchups) != games:
            return {"available": False, "reason": "inherited heldout game mismatch"}
        matchups.sort(key=lambda row: str(row["opponent_id"]))
        lineage_iteration = int(inherited.get("lineage_iteration") or 0)
        display_iteration = inherited.get("iteration")
        if not isinstance(display_iteration, int):
            source_offset = int((handoff or {}).get("source_global_iteration_offset") or 0)
            display_iteration = source_offset + lineage_iteration
        return {
            "available": True,
            "kind": "inherited_official_heldout_champion",
            "valid": True,
            "passed": inherited.get("passed") is True,
            "reason": inherited.get("reason"),
            "games": games,
            "wr": as_float(inherited.get("wr")),
            "lower": as_float(inherited.get("lower")),
            "upper": as_float(inherited.get("upper")),
            "iteration": int(display_iteration),
            "lineage_iteration": lineage_iteration,
            "checkpoint": inherited_identity.get("path"),
            "checkpoint_digest": digest,
            "matchups": matchups,
            "opponent_count": len(matchups),
            "audit_passed": True,
            "exact_distribution": True,
            "exact_weights": True,
            "greedy_required": True,
            "source": f"lineage handoff from {(handoff or {}).get('source_run') or 'prior run'}",
            "updated_at": None,
        }

    iteration = evidence.get("iteration")
    digest = str(evidence.get("checkpoint_digest") or "")
    identity_digest = str(identity.get("digest") or "")
    if not isinstance(iteration, int) or not digest or digest != identity_digest:
        return {"available": False, "reason": "heldout champion identity mismatch"}

    matching: dict[str, Any] | None = None
    for row in reversed(loop.get("history") or []):
        if not isinstance(row, dict) or row.get("iteration") != iteration:
            continue
        candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        if str(candidate.get("digest") or "") == digest:
            matching = row
            break
    if matching is None:
        return {"available": False, "reason": "heldout champion commit is missing"}

    gate = matching.get("raw_heldout_gate")
    audit = matching.get("heldout_audit")
    if not isinstance(gate, dict) or not isinstance(audit, dict):
        return {"available": False, "reason": "heldout gate or audit is missing"}

    games = int(evidence.get("games") or 0)
    audit_games = int(audit.get("valid_games") or 0)
    gate_games = int(gate.get("games") or 0)
    evidence_wr = as_float(evidence.get("win_rate"))
    gate_wr = as_float(gate.get("win_rate"))
    audit_passed = audit.get("passed") is True
    reconciled = bool(
        audit_passed
        and games > 0
        and games == audit_games == gate_games
        and evidence_wr is not None
        and gate_wr is not None
        and abs(evidence_wr - gate_wr) < 1e-12
        and str(audit.get("checkpoint_digest") or "") == digest
    )
    if not reconciled:
        return {"available": False, "reason": "heldout evidence failed reconciliation"}

    audit_matchups = audit.get("per_opponent")
    gate_matchups = gate.get("per_opponent")
    if not isinstance(audit_matchups, dict) or not isinstance(gate_matchups, dict):
        return {"available": False, "reason": "heldout matchup evidence is missing"}
    matchups: list[dict[str, Any]] = []
    for opponent_id, row in gate_matchups.items():
        if not isinstance(row, dict):
            continue
        seats = audit_matchups.get(opponent_id)
        seats = seats if isinstance(seats, dict) else {}
        matchups.append(
            {
                "opponent_id": str(opponent_id),
                "games": int(row.get("games") or 0),
                "wr": as_float(row.get("win_rate")),
                "wins": as_float(row.get("wins")),
                "draws": as_float(row.get("draws")),
                "losses": as_float(row.get("losses")),
                "seat0": int(seats.get("seat0") or row.get("seat0_games") or 0),
                "seat1": int(seats.get("seat1") or row.get("seat1_games") or 0),
            }
        )
    matchups.sort(key=lambda row: str(row["opponent_id"]))

    source = (
        run_dir / "commits" / f"iter_{iteration:05d}.json"
        if run_dir is not None
        else None
    )
    if source is None or not source.is_file():
        source = run_dir / "loop_state.json" if run_dir is not None else None
    return {
        "available": True,
        "kind": "official_heldout_champion",
        "valid": True,
        "passed": gate.get("passed") is True,
        "reason": gate.get("reason"),
        "games": games,
        "wr": evidence_wr,
        "lower": as_float(evidence.get("confidence_lower")),
        "upper": as_float(evidence.get("confidence_upper")),
        "iteration": int(iteration) + int(global_iteration_offset),
        "lineage_iteration": int(iteration),
        "checkpoint": identity.get("path"),
        "checkpoint_digest": digest,
        "matchups": matchups,
        "opponent_count": len(matchups),
        "audit_passed": True,
        "exact_distribution": audit.get("exact_distribution") is True,
        "exact_weights": audit.get("exact_weights") is True,
        "greedy_required": audit.get("greedy_required") is True,
        "source": str(source) if source is not None else None,
        "updated_at": source.stat().st_mtime if source is not None and source.is_file() else None,
    }


def gpu_state() -> list[dict[str, Any]]:
    raw = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=4,
    )
    gpus: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue

        def number(value: str) -> float | None:
            try:
                return float(value)
            except ValueError:
                return None

        gpus.append(
            {
                "index": as_number(parts[0]),
                "name": parts[1],
                "utilization": number(parts[2]),
                "memory_used_mib": number(parts[3]),
                "memory_total_mib": number(parts[4]),
                "power_w": number(parts[5]),
                "power_limit_w": number(parts[6]),
                "temperature_c": number(parts[7]),
            }
        )
    return gpus


def system_state() -> dict[str, Any]:
    mem: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if value:
                mem[key] = int(value.strip().split()[0]) * 1024
    except OSError:
        pass
    loads = os.getloadavg()
    return {
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "load_1m": loads[0],
        "load_5m": loads[1],
        "load_15m": loads[2],
        "memory_total_bytes": mem.get("MemTotal"),
        "memory_available_bytes": mem.get("MemAvailable"),
    }


def recent_events(run_name: str | None = None) -> list[str]:
    # Keep the active run last: callers retain the final lines, so placing a
    # completed bootstrap after it silently hid live scheduler GPS/SPS and
    # result-buffer telemetry.
    active_log = (
        ROOT / "outputs/logs" / f"{run_name}.log" if run_name else None
    )
    raw = (
        read_tail(EXACT_LOG, 40_000)
        + "\n"
        + read_tail(TRAINING_LOG, 40_000)
        + "\n"
        + read_tail(ALAKAZAM_TRANSITION_LOG, 40_000)
        + "\n"
        + read_tail(ALAKAZAM_BOOTSTRAP_LOG, 80_000)
        + "\n"
        + (read_tail(active_log, 120_000) if active_log is not None else "")
    )
    lines = [ANSI_RE.sub("", line).strip() for line in raw.replace("\r", "\n").splitlines()]
    return [line for line in lines if line][-12:]


def scheduler_queue_state(run_name: str | None) -> dict[str, Any]:
    """Read the active dispatch's endpoint-owned queue contract.

    The trainer emits the protected controller depths before request threads
    start.  Remote worker snapshots provide the changing server-side queue;
    keeping these two grains separate prevents a full protected reserve from
    being mislabeled as an idle worker.
    """
    if not run_name:
        return {"available": False, "mode": "waiting"}
    log_path = ROOT / "outputs/logs" / f"{run_name}.log"
    raw = ANSI_RE.sub("", read_tail(log_path, 240_000)).replace("\r", "\n")
    matches = list(
        re.finditer(
            r"\[remote\] endpoint_owned_queues depths=(\{[^\n]+?\}) "
            r"(?:caps|high_water)=(\{[^\n]+?\})(?: safety_ceiling=\{[^\n]+?\})? "
            r"shared_endpoint_race=disabled",
            raw,
        )
    )
    if not matches:
        return {"available": False, "mode": "legacy_or_starting"}
    latest = matches[-1]
    try:
        depths_raw = ast.literal_eval(latest.group(1))
        caps_raw = ast.literal_eval(latest.group(2))
    except (SyntaxError, ValueError):
        return {"available": False, "mode": "invalid_telemetry"}
    if not isinstance(depths_raw, dict) or not isinstance(caps_raw, dict):
        return {"available": False, "mode": "invalid_telemetry"}

    # Socket-prefetch lines are emitted immediately before the owned-queue
    # reservation line, so include a bounded prefix as well as live updates.
    dispatch_tail = raw[max(0, latest.start() - 8_000) :]
    # A bounded prefix is useful for the socket-prefetch lines emitted just
    # before this queue generation. Remaining-job counters are phase-local,
    # however: carrying the preceding self-play ``remaining=0`` into a newly
    # started public-mix wave falsely reports that all public work is assigned.
    phase_tail = raw[latest.end() :]
    socket_prefetch: dict[str, int] = {}
    for endpoint, count in re.findall(
        r"\[remote\]\s+(\S+)\s+socket_prefetch=(\d+)", dispatch_tail
    ):
        socket_prefetch[str(endpoint)] = int(count)
    rebalance = list(re.finditer(r"\bremaining=(\d+)\b", phase_tail))
    unassigned = int(rebalance[-1].group(1)) if rebalance else None
    controller_contract = list(
        re.finditer(
            r"\[remote\] queue_refill_controller interval=([\d.]+)s "
            r"low_water=(\d+)% action=(\S+) endpoints=(\S+) "
            r"ingest_coupled=(\S+)",
            phase_tail,
        )
    )
    refill_events = list(
        re.finditer(
            r"\[remote\]\s+(\S+)\s+LOW_WATER_REFILL "
            r"active=(\d+) queued=(\d+)<(\d+) added=(\d+) "
            r"fill=high_water target_active=(\d+) high_water=(\d+)",
            phase_tail,
        )
    )

    def host_key(endpoint: str) -> str:
        lowered = endpoint.lower()
        if "bert" in lowered or "192.168.1.158" in lowered:
            return "bert"
        if "elmo" in lowered or "192.168.1.143" in lowered:
            return "elmo"
        return endpoint

    endpoints: dict[str, dict[str, Any]] = {}
    for endpoint, cap_value in caps_raw.items():
        endpoint = str(endpoint)
        cap = max(0, int(cap_value))
        sockets = max(0, int(socket_prefetch.get(endpoint, 0)))
        endpoints[host_key(endpoint)] = {
            "endpoint": endpoint,
            "dispatch_reserved": max(0, int(depths_raw.get(endpoint, 0))),
            "protected_high_water": cap,
            "socket_capacity": sockets or None,
            "controller_reserve_target": max(0, cap - sockets) if sockets else None,
        }
    for event in refill_events:
        row = endpoints.get(host_key(event.group(1)))
        if row is None:
            continue
        row["last_refill"] = {
            "sampled_active": int(event.group(2)),
            "sampled_queued": int(event.group(3)),
            "low_water": int(event.group(4)),
            "added": int(event.group(5)),
            "target_active": int(event.group(6)),
            "high_water": int(event.group(7)),
            "action": "fill_to_high_water",
        }
    contract: dict[str, Any] = {
        "probe_interval_s": 0.2,
        "low_water_fraction": 0.5,
        "action": "fill_to_high_water",
        "endpoints_parallel": True,
        "ingest_coupled": False,
    }
    if controller_contract:
        event = controller_contract[-1]
        contract.update(
            probe_interval_s=float(event.group(1)),
            low_water_fraction=float(event.group(2)) / 100.0,
            action=str(event.group(3)),
            endpoints_parallel=str(event.group(4)).lower() == "parallel",
            ingest_coupled=str(event.group(5)).lower() == "true",
        )
    return {
        "available": True,
        "mode": "endpoint_owned",
        "shared_endpoint_race_disabled": True,
        "unassigned": unassigned,
        "refill_contract": contract,
        "endpoints": endpoints,
        "source": str(log_path),
    }


def learner_model_state(
    manifest: dict[str, Any],
    loop: dict[str, Any] | None = None,
    *,
    iteration: int | None = None,
) -> dict[str, Any]:
    """Describe the exact live model plus explicitly non-live staged profiles.

    The old dashboard hard-coded one parameter count. That looked current even
    after a model-profile change. Prefer immutable manifest metadata, then an
    independently deployed profile registry whose full config must match. If
    neither source matches, report an unknown count instead of a plausible lie.
    """
    learner = ((manifest.get("design_contract") or {}).get("learner") or {})
    profile = learner.get("profile") if isinstance(learner.get("profile"), dict) else {}
    loop = loop if isinstance(loop, dict) else {}

    registry = read_json(MODEL_PROFILE_REGISTRY)
    registry_profiles = registry.get("profiles") or []
    matched_profile: dict[str, Any] = {}
    planned_profile: dict[str, Any] = {}
    for candidate in registry_profiles:
        if not isinstance(candidate, dict):
            continue
        candidate_profile = candidate.get("profile")
        if isinstance(candidate_profile, dict) and candidate_profile == profile:
            matched_profile = candidate
        status = str(candidate.get("status") or "")
        # A registry entry can retain its historical ``staged_*`` label after
        # that exact profile has become the immutable live manifest profile.
        # Never surface the active profile as a future plan: the manifest is
        # authoritative for what the trainer is actually running.
        if (
            not planned_profile
            and status.startswith("staged")
            and candidate_profile != profile
        ):
            planned_profile = candidate

    parameter_count = as_number(str(learner.get("trainable_parameters") or ""))
    parameter_source = None
    if parameter_count is not None:
        parameter_source = "manifest.design_contract.learner"
    if parameter_count is None:
        base_contract = manifest.get("base_checkpoint_contract") or {}
        parameter_count = as_number(
            str(base_contract.get("trainable_parameters") or "")
        )
        if parameter_count is not None:
            parameter_source = "manifest.base_checkpoint_contract"
    if parameter_count is None:
        parameter_count = as_number(
            str(manifest.get("trainable_parameters") or "")
        )
        if parameter_count is not None:
            parameter_source = "manifest"
    if parameter_count is None and matched_profile:
        parameter_count = as_number(
            str(matched_profile.get("trainable_parameters") or "")
        )
        if parameter_count is not None:
            parameter_source = f"profile_registry:{matched_profile.get('id') or 'matched'}"

    steady_cap = as_number(str(learner.get("max_decisions_per_batch") or ""))
    warmup_cap = as_number(
        str(learner.get("warmup_max_decisions_per_batch") or "")
    )
    warmup_iterations = as_number(str(learner.get("warmup_iterations") or "")) or 0
    active_cap = steady_cap
    schedule_phase = "steady"
    if (
        warmup_cap is not None
        and iteration is not None
        and int(iteration) < int(warmup_iterations)
    ):
        active_cap = warmup_cap
        schedule_phase = "head_focus"

    active_checkpoint = loop.get("learner")
    if not isinstance(active_checkpoint, dict):
        active_checkpoint = {}

    def weighted_head(weight_key: str, *, outputs: int | None = None) -> dict[str, Any]:
        weight = as_float(learner.get(weight_key)) or 0.0
        row: dict[str, Any] = {"enabled": weight > 0.0, "loss_weight": weight}
        if outputs is not None:
            row["outputs"] = outputs
        return row

    initial = learner.get("initial_checkpoint")
    if not isinstance(initial, dict):
        initial = {}
    temporal_layers = as_number(str(profile.get("temporal_layers") or 0)) or 0
    architecture = (
        "full-game temporal state evaluator"
        if temporal_layers > 0 or profile.get("decision_context") == "history"
        else "stateless state evaluator"
    )
    return {
        "implementation": "TemporalCabtTransformer",
        "architecture": architecture,
        "run": manifest.get("run_name"),
        "profile": profile,
        "profile_id": matched_profile.get("id"),
        "trainable_parameters": parameter_count,
        "parameter_source": parameter_source,
        "active_checkpoint": active_checkpoint.get("path"),
        "active_checkpoint_digest": active_checkpoint.get("digest"),
        "training_schedule": {
            "iteration": iteration,
            "phase": schedule_phase,
            "active_max_decisions_per_batch": active_cap,
            "warmup_max_decisions_per_batch": warmup_cap,
            "warmup_iterations": warmup_iterations,
            "steady_max_decisions_per_batch": steady_cap,
        },
        "planned_profile": planned_profile,
        "heads": {
            "policy": {"enabled": True},
            "value": {"enabled": True},
            "archetype": weighted_head("archetype_aux_loss_weight", outputs=20),
            "opponent_hand": weighted_head("opp_hand_loss_weight", outputs=1268),
            "opponent_remainder": weighted_head(
                "opp_remainder_loss_weight", outputs=1268
            ),
            "lethal_threat": weighted_head("lethal_threat_loss_weight", outputs=1),
            "prize_race": weighted_head("prize_race_loss_weight", outputs=2),
        },
        "training_targets": {
            "alakazam_guide": {
                "enabled": bool(learner.get("alakazam_guide_targets_enabled"))
                and (as_float(learner.get("alakazam_guide_loss_weight")) or 0.0) > 0.0,
                "loss_weight": as_float(learner.get("alakazam_guide_loss_weight")) or 0.0,
                "shared_head": "policy",
                "parameterized_head": False,
            }
        },
        "seed_checkpoint": initial.get("path"),
        "seed_checkpoint_digest": initial.get("digest"),
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def replay_window_state(
    run_dir: Path | None,
    loop: dict[str, Any],
    manifest: dict[str, Any],
    progress: dict[str, Any],
    raw_training_log: str,
) -> dict[str, Any]:
    """Describe the live rolling replay window without touching trainer state.

    During JSONL ingestion the trainer has the current shard open. Linux
    ``fdinfo`` exposes its byte position, giving the dashboard a real loading
    percentage without adding logging or allocations to the training process.
    """
    if run_dir is None:
        return {"available": False, "stage": "waiting", "percent": None}
    design = manifest.get("design_contract") or {}
    collection = design.get("collection") or {}
    training_design = manifest.get("training_design") or {}
    window = as_number(str(collection.get("replay_window_shards", "")))
    if window is None:
        window = as_number(str(training_design.get("replay_window_shards", "")))
    window = max(1, int(window or 2))
    iteration = progress.get("iteration")
    if not isinstance(iteration, int):
        iteration = as_number(str(loop.get("next_iteration", "")))
    if iteration is None:
        return {
            "available": False,
            "stage": "waiting",
            "percent": None,
            "window_shards": window,
        }
    first = max(0, int(iteration) - window + 1)
    indices = list(range(first, int(iteration) + 1))
    shard_rows: list[dict[str, Any]] = []
    learner = design.get("learner") or {}
    game_contract = design.get("games") or {}
    per_iteration = int(as_number(str(game_contract.get("per_iteration", ""))) or 0)
    if int(iteration) == 0 and window > 1:
        inherited = list(learner.get("initial_replay_shards") or [])[-(window - 1) :]
        for offset, identity in enumerate(inherited, start=1):
            if not isinstance(identity, dict) or not identity.get("path"):
                continue
            path = Path(str(identity["path"])).expanduser()
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            shard_rows.append(
                {
                    "iteration": -offset,
                    "path": str(path),
                    "name": path.name,
                    "bytes": size,
                    "inherited": True,
                }
            )
    for index in indices:
        path = run_dir / "shards" / f"iter_{index:05d}.jsonl"
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        # A recovery/gate-only lineage may intentionally collect zero fresh
        # games and train exclusively from its immutable handoff shard.
        if per_iteration == 0 and size == 0:
            continue
        shard_rows.append(
            {"iteration": index, "path": str(path), "name": path.name, "bytes": size}
        )
    total_bytes = sum(int(row["bytes"]) for row in shard_rows)
    stage = str(progress.get("stage") or "")
    collecting = stage.startswith("collect:")
    ready_shards = sum(
        1
        for row in shard_rows
        if int(row["bytes"]) > 0
        and (int(row["iteration"]) < int(iteration) or not collecting)
    )
    percent: float | None = None
    current: int | None = None
    total: int | None = None
    unit = "shards"
    state = "READY"
    detail = f"{ready_shards}/{len(shard_rows)} shards ready"

    clean_log = ANSI_RE.sub("", raw_training_log).replace("\r", "\n")
    train_begin = re.findall(
        rf"\[pure_rl\] train begin iter={int(iteration)} seqs=(\d+)", clean_log
    )
    sequences = int(train_begin[-1]) if train_begin else None
    cache: dict[str, Any] = {}
    status_candidates = [run_dir / "replay_window.cache.status.json"]
    status_candidates.extend(
        Path(str(row["path"])).parent.parent / "replay_window.cache.status.json"
        for row in shard_rows
    )
    row_sources = {
        str(Path(str(row["path"])).resolve()) for row in shard_rows
    }
    for status_path in dict.fromkeys(status_candidates):
        cache_raw = read_json(status_path)
        try:
            cache_source = Path(str(cache_raw.get("source_shard") or "")).resolve()
            cache_age = max(
                0.0, time.time() - float(cache_raw.get("updated_at") or 0.0)
            )
            if str(cache_source) not in row_sources or cache_age > 300.0:
                continue
            candidate = {
                "stage": cache_raw.get("stage"),
                "source_shard": str(cache_source),
                "workers": cache_raw.get("workers"),
                "parts_complete": cache_raw.get("parts_complete"),
                "parts_total": cache_raw.get("parts_total"),
                "bytes_complete": cache_raw.get("bytes_complete"),
                "bytes_total": cache_raw.get("bytes_total"),
                "percent": cache_raw.get("percent"),
                "sequences": cache_raw.get("sequences")
                or cache_raw.get("sequences_loaded"),
                "age_s": cache_age,
            }
            if not cache or cache_age < float(cache.get("age_s") or float("inf")):
                cache = candidate
        except (OSError, TypeError, ValueError):
            continue

    if collecting:
        cache_stage = str(cache.get("stage") or "")
        state = (
            "BUILDING + FEATURIZING WINDOW"
            if cache_stage == "streaming_featurize"
            else "BUILDING WINDOW"
        )
        if isinstance(progress.get("percent"), (int, float)):
            percent = float(progress["percent"])
        current = progress.get("current") if isinstance(progress.get("current"), int) else None
        total = progress.get("total") if isinstance(progress.get("total"), int) else None
        unit = str(progress.get("unit") or "games")
        detail = (
            f"writing iter_{int(iteration):05d}.jsonl"
            + (f" · {current}/{total} {unit}" if current is not None and total else "")
        )
        if cache_stage == "streaming_featurize":
            done = int(cache.get("parts_complete") or 0)
            submitted = int(cache.get("parts_total") or 0)
            workers = int(cache.get("workers") or 0)
            detail += (
                f" · stream cache {done}/{submitted} chunks complete"
                f" on {workers} CPU workers"
            )
    elif stage == "train:preparing":
        state = "LOADING WINDOW"
        unit = "bytes"
        open_row: dict[str, Any] | None = None
        loaded_bytes = 0
        pid_text = run(["pgrep", "-f", "scripts/train_pure_rl.py"], timeout=2)
        by_path = {
            str(Path(str(row["path"])).resolve()): (offset, row)
            for offset, row in enumerate(shard_rows)
        }
        for pid_raw in pid_text.splitlines():
            pid = as_number(pid_raw.strip())
            if not pid:
                continue
            fd_root = Path(f"/proc/{pid}/fd")
            try:
                fds = list(fd_root.iterdir())
            except OSError:
                continue
            for fd in fds:
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                try:
                    resolved_target = str(Path(target).resolve())
                except OSError:
                    resolved_target = target
                matched = by_path.get(resolved_target)
                if matched is None:
                    continue
                offset, row = matched
                pos = 0
                try:
                    for line in Path(f"/proc/{pid}/fdinfo/{fd.name}").read_text().splitlines():
                        if line.startswith("pos:"):
                            pos = int(line.split()[1])
                            break
                except (OSError, ValueError, IndexError):
                    pass
                loaded_bytes = sum(int(item["bytes"]) for item in shard_rows[:offset])
                loaded_bytes += min(max(0, pos), int(row["bytes"]))
                open_row = row
                break
            if open_row is not None:
                break
        if open_row is not None and total_bytes > 0:
            current = loaded_bytes
            total = total_bytes
            percent = min(100.0, 100.0 * loaded_bytes / total_bytes)
            detail = f"reading {open_row['name']} · {percent:.1f}% of window bytes"
        elif sequences is not None:
            percent = 100.0
            current = total_bytes
            total = total_bytes
            state = "WINDOW READY"
            detail = f"assembled {sequences:,} train sequences · preparing AWR"
        else:
            detail = f"opening {len(shard_rows)}-shard window · measuring byte position"
        cache_stage = str(cache.get("stage") or "")
        if cache_stage in {"parallel_featurize", "cache_load", "stream_cache_ready"}:
            state = {
                "parallel_featurize": "PARALLEL FEATURIZING",
                "cache_load": "LOADING FEATURE CACHE",
                "stream_cache_ready": "STREAM CACHE READY",
            }[cache_stage]
            if isinstance(cache.get("percent"), (int, float)):
                percent = float(cache["percent"])
            done = int(cache.get("parts_complete") or 0)
            count = int(cache.get("parts_total") or 0)
            detail = f"{state.lower()} · {done}/{count} chunks"
    elif stage.startswith("train:") or stage in {"heldout", "promotion"}:
        percent = 100.0
        current = total_bytes
        total = total_bytes
        unit = "bytes"
        state = "WINDOW READY"
        detail = (
            f"{sequences:,} train sequences · {len(shard_rows)} shard window"
            if sequences is not None
            else f"{len(shard_rows)} shard window retained on disk"
        )

    return {
        "available": True,
        "iteration": int(iteration),
        "stage": state,
        "detail": detail,
        "percent": percent,
        "current": current,
        "total": total,
        "unit": unit,
        "window_shards": window,
        "target_shards": len(shard_rows),
        "ready_shards": ready_shards,
        "bytes_total": total_bytes,
        "sequences": sequences,
        "shards": shard_rows,
        "cache": cache,
    }


def _tqdm_rate(timing: str, units: tuple[str, ...]) -> tuple[float | None, str | None]:
    unit_pattern = "|".join(re.escape(unit) for unit in units)
    match = re.search(rf"([0-9.]+)({unit_pattern})", timing)
    if not match:
        return None, None
    return float(match.group(1)), match.group(2)


def _tqdm_eta(timing: str) -> str | None:
    match = re.search(r"<([^,\]]+)", timing)
    return match.group(1).strip() if match else None


def parse_curriculum_progress(
    raw_status: str,
    raw_progress_log: str,
    *,
    iteration_hint: int | None = None,
) -> dict[str, Any]:
    """Parse the newest collect, policy-epoch, or validation tqdm frame.

    ``*.progress.status`` is intentionally written by the game collector only.
    Mid-iteration training uses tqdm directly, so its outer epoch and nested
    validation bars live in the run-specific ``*.progress.log`` instead.  We
    preserve stream order and take the newest recognized frame, never a bar
    from a different run or an older global alias.
    """
    progress: dict[str, Any] = {
        "line": raw_status.strip(),
        "stage": None,
        "iteration": iteration_hint,
        "epoch": None,
        "percent": None,
        "current": None,
        "total": None,
        "unit": None,
        "rate": None,
        "rate_unit": None,
        "eta": None,
        "gps": None,
        "sps": None,
        "remotes": None,
        "metrics": {},
    }
    clean_log = ANSI_RE.sub("", raw_progress_log).replace("\r", "\n")
    lines = []
    for raw_line in clean_log.splitlines():
        # A forced service stop can append Python's resource-tracker warning
        # directly after a truncated tqdm frame. Preserve the valid progress
        # prefix and discard the unrelated shutdown warning.
        line = re.split(
            r"(?=/[^\s]*multiprocessing/resource_tracker\.py:|UserWarning:\s*resource_tracker:)",
            raw_line,
            maxsplit=1,
        )[0].strip()
        if line and not line.startswith("warnings.warn("):
            lines.append(line)
    # A just-created progress log can be empty for its first instant. Only in
    # that case use the already run-bound single-line status mirror.
    if not lines and raw_status.strip():
        lines = [ANSI_RE.sub("", raw_status).strip()]

    last_train_metrics: dict[str, float | None] = {}
    for line in lines:
        resident_pack = re.search(
            r"pack Blackwell corpus:\s*(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)\]",
            line,
        )
        if resident_pack:
            percent, current, total, timing = resident_pack.groups()
            rate, rate_unit = _tqdm_rate(timing, ("game/s", "s/game"))
            progress.update(
                line=line,
                stage="train:packing",
                iteration=iteration_hint,
                epoch=0,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="games",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=None,
                remotes=0,
                metrics={},
            )
            continue

        replay_cache = re.search(
            r"replay-cache load\s+(\S+):\s*(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)\]",
            line,
        )
        if replay_cache:
            shard_name, percent, current, total, timing = replay_cache.groups()
            rate, rate_unit = _tqdm_rate(timing, ("part/s", "s/part"))
            progress.update(
                line=line,
                stage="train:preparing",
                iteration=iteration_hint,
                epoch=0,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="parts",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=None,
                remotes=0,
                metrics={"replay_shard": shard_name},
            )
            continue

        collect = re.search(
            r"pure_rl\s+(\S+)\s+iter=(\d+):\s*(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)(?:\]|$)",
            line,
        )
        if collect:
            stage, iteration, percent, current, total, timing = collect.groups()
            if stage == "train:expert":
                progress.update(
                    line=line,
                    stage="train:expert:loading",
                    iteration=int(iteration),
                    epoch=0,
                    epochs=None,
                    percent=float(percent),
                    current=int(current),
                    total=int(total),
                    unit="expert pass",
                    rate=None,
                    rate_unit=None,
                    eta="loading corpus",
                    gps=None,
                    sps=None,
                    remotes=0,
                    metrics={},
                )
                last_train_metrics = {}
                continue
            rate, rate_unit = _tqdm_rate(timing, ("game/s", "s/game"))
            gps = None
            if rate is not None:
                gps = rate if rate_unit == "game/s" else 1.0 / max(rate, 1e-9)
            request_sockets = parse_metric(timing, "remotes")
            remote_demand = parse_metric(timing, "rdmd")
            # With socket prefetch, ``remotes`` is the number of admitted TCP
            # requests, while ``rdmd`` remains execution-worker demand.  Keep
            # the public ``remotes`` metric at worker grain so the dashboard
            # never labels queued requests as extra simulator workers.
            remote_workers = (
                int(remote_demand)
                if remote_demand is not None
                else int(request_sockets)
                if request_sockets is not None
                else None
            )
            remote_queue_capacity = (
                max(0, int(request_sockets) - int(remote_workers))
                if request_sockets is not None and remote_workers is not None
                else None
            )
            sps = parse_metric(timing, "sps")
            progress.update(
                line=line,
                stage=stage,
                iteration=int(iteration),
                epoch=None,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="games",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=gps,
                sps=sps,
                remotes=remote_workers,
                metrics={
                    key: value
                    for key, value in {
                        "remote_request_sockets": (
                            int(request_sockets)
                            if request_sockets is not None
                            else None
                        ),
                        "remote_queue_capacity": remote_queue_capacity,
                    }.items()
                    if value is not None
                },
            )
            last_train_metrics = {}
            continue

        expert_loading = re.search(
            r"pure_rl train:expert iter=(\d+):\s*(\d+)%.*?"
            r"(\d+)/(\d+)\s+\[([^]]*)(?:\]|$)",
            line,
        )
        if expert_loading:
            iteration, percent, current, total, timing = expert_loading.groups()
            progress.update(
                line=line,
                stage="train:expert:loading",
                iteration=int(iteration),
                epoch=0,
                epochs=None,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="expert pass",
                rate=None,
                rate_unit=None,
                eta="loading corpus",
                gps=None,
                sps=None,
                remotes=0,
                metrics={},
            )
            continue

        expert_batch = re.search(
            r"expert rehearsal before iter(\d+) ep(\d+)/(\d+):\s*"
            r"(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)(?:\]|$)",
            line,
        )
        if expert_batch:
            (
                iteration,
                epoch,
                epochs,
                percent,
                current,
                total,
                timing,
            ) = expert_batch.groups()
            rate, rate_unit = _tqdm_rate(timing, ("batch/s", "s/batch"))
            metrics = {
                name: parse_metric(timing, name)
                for name in ("acc", "loss", "policy", "value", "step")
            }
            metrics = {
                key: value for key, value in metrics.items() if value is not None
            }
            progress.update(
                line=line,
                stage="train:expert",
                iteration=int(iteration),
                epoch=int(epoch),
                epochs=int(epochs),
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="batches",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=None,
                remotes=0,
                metrics=metrics,
            )
            continue

        expert_validation = re.search(
            r"expert validation before iter(\d+):\s*(\d+)%.*?"
            r"(\d+)/(\d+)\s+\[([^]]*)(?:\]|$)",
            line,
        )
        if expert_validation:
            iteration, percent, current, total, timing = expert_validation.groups()
            rate, rate_unit = _tqdm_rate(timing, ("batch/s", "s/batch"))
            progress.update(
                line=line,
                stage="train:expert:validation",
                iteration=int(iteration),
                epoch=None,
                epochs=None,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="batches",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=None,
                remotes=0,
                metrics={},
            )
            continue

        training_batch = re.search(
            r"rl-train\s+ep(\d+):\s*(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)\]",
            line,
        )
        if training_batch:
            epoch, percent, current, total, timing = training_batch.groups()
            rate, rate_unit = _tqdm_rate(timing, ("batch/s", "s/batch"))
            optimizer_sps = parse_metric(timing, "sps")
            metrics = {
                name: parse_metric(timing, name)
                for name in (
                    "acc",
                    "loss",
                    "p",
                    "v",
                    "hand",
                    "rem",
                    "aux",
                    "lethal",
                    "prize",
                    "guide",
                )
            }
            metrics = {key: value for key, value in metrics.items() if value is not None}
            if metrics:
                last_train_metrics = metrics
            progress.update(
                line=line,
                stage="train:policy",
                iteration=iteration_hint,
                epoch=int(epoch) + 1,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="batches",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=optimizer_sps,
                remotes=0,
                metrics=dict(last_train_metrics),
            )
            continue

        preparation = re.search(
            r"rl-(prep|agreement)\s+(baseline|parent|candidate):\s*(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)\]",
            line,
        )
        if preparation:
            family, phase, percent, current, total, timing = preparation.groups()
            rate, rate_unit = _tqdm_rate(timing, ("batch/s", "s/batch"))
            progress.update(
                line=line,
                stage=f"train:{family}:{phase}",
                iteration=iteration_hint,
                epoch=0,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="batches",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=None,
                remotes=0,
                metrics={},
            )
            continue

        train = re.search(
            r"rl-train:\s*(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)\]",
            line,
        )
        if train:
            percent, current, total, timing = train.groups()
            rate, rate_unit = _tqdm_rate(timing, ("ep/s", "s/ep"))
            metrics = {
                name: parse_metric(timing, name)
                for name in (
                    "acc",
                    "loss",
                    "p",
                    "v",
                    "hand",
                    "rem",
                    "aux",
                    "lethal",
                    "prize",
                    "guide",
                    "best",
                    "pat",
                )
            }
            metrics = {key: value for key, value in metrics.items() if value is not None}
            if metrics:
                last_train_metrics = metrics
            progress.update(
                line=line,
                stage="train:policy",
                iteration=iteration_hint,
                epoch=int(current),
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="epochs",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=None,
                remotes=0,
                metrics=dict(last_train_metrics),
            )
            continue

        validation = re.search(
            r"rl-val\s+ep(\d+):\s*(\d+)%.*?\s(\d+)/(\d+)\s+\[([^]]*)\]",
            line,
        )
        if validation:
            epoch, percent, current, total, timing = validation.groups()
            rate, rate_unit = _tqdm_rate(timing, ("batch/s", "s/batch"))
            progress.update(
                line=line,
                stage="train:validation",
                iteration=iteration_hint,
                epoch=int(epoch) + 1,
                percent=float(percent),
                current=int(current),
                total=int(total),
                unit="batches",
                rate=rate,
                rate_unit=rate_unit,
                eta=_tqdm_eta(timing),
                gps=None,
                sps=None,
                remotes=0,
                metrics=dict(last_train_metrics),
            )
    return progress


def infer_between_bar_progress(
    progress: dict[str, Any],
    raw_training_log: str,
    *,
    iteration_hint: int | None,
    train_epochs: int = 2,
) -> dict[str, Any]:
    """Expose CPU replay/AWR preparation between collect and tqdm epochs."""
    if iteration_hint is None or str(progress.get("stage") or "").startswith("train"):
        return progress
    clean = ANSI_RE.sub("", raw_training_log).replace("\r", "\n")
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    marker = ""
    marker_kind = ""
    for line in lines:
        patterns = (
            ("collect_done", rf"\[pure_rl\] collect done iter={iteration_hint}(?:\s|$)"),
            ("train_begin", rf"\[pure_rl\] train begin iter={iteration_hint}(?:\s|$)"),
            ("train_released", rf"\[pure_rl\] replay memory released iter={iteration_hint}(?:\s|$)"),
            ("promotion", rf"\[pure_rl\] promotion begin iter={iteration_hint}(?:\s|$)"),
        )
        for kind, pattern in patterns:
            if re.search(pattern, line):
                marker, marker_kind = line, kind
    if marker_kind not in {"collect_done", "train_begin"}:
        return progress
    # Only replace the just-completed collection bar for this same iteration.
    if progress.get("iteration") != iteration_hint or float(progress.get("percent") or 0) < 100:
        return progress
    updated = dict(progress)
    updated.update(
        stage="train:preparing",
        epoch=0,
        percent=None,
        current=0,
        total=max(1, int(train_epochs)),
        unit="epochs",
        rate=None,
        rate_unit=None,
        eta="preparing",
        gps=None,
        sps=None,
        remotes=0,
        metrics={},
        line=(
            marker
            if marker_kind == "train_begin"
            else f"[pure_rl] train preparing iter={iteration_hint}: "
            "assembling rolling replay window + AWR baselines"
        ),
    )
    return updated


def _metric_iteration_wall_seconds(payload: dict[str, Any]) -> float | None:
    """Return full collect-to-commit wall time, including legacy metrics."""
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    explicit = as_float(extra.get("iteration_wall_sec"))
    if explicit is not None and explicit >= 0:
        return explicit
    post_collect = as_float(
        extra.get("post_collect_elapsed_sec", extra.get("elapsed_sec"))
    )
    collect_stats = (
        extra.get("collect_stats")
        if isinstance(extra.get("collect_stats"), dict)
        else {}
    )
    collect = as_float(collect_stats.get("collect_elapsed_sec"))
    if post_collect is not None and collect is not None:
        return max(0.0, post_collect) + max(0.0, collect)
    return max(0.0, post_collect) if post_collect is not None else None


def iteration_timing_state(
    run_dir: Path | None,
    *,
    active: bool,
    global_iteration_offset: int,
    next_iteration: int | None = None,
    progress_iteration: int | None = None,
    progress_stage: str | None = None,
) -> dict[str, Any]:
    """Source-backed current/latest/rolling iteration timing telemetry."""
    if run_dir is None:
        return {
            "available": False,
            "current_seconds": None,
            "latest_seconds": None,
            "rolling5_seconds": None,
            "history": [],
        }
    history: list[dict[str, Any]] = []
    metrics_dir = run_dir / "metrics"
    for path in sorted(metrics_dir.glob("iter_*.json")):
        payload = read_json(path)
        iteration = payload.get("iteration")
        seconds = _metric_iteration_wall_seconds(payload)
        if not isinstance(iteration, int) or seconds is None:
            continue
        history.append(
            {
                "iteration": iteration + int(global_iteration_offset),
                "lineage_iteration": iteration,
                "seconds": seconds,
            }
        )
    history.sort(key=lambda row: int(row["lineage_iteration"]))
    history = history[-20:]
    latest = history[-1] if history else None
    rolling = history[-5:]
    runtime = read_json(run_dir / "iteration_runtime.json")
    current_iteration = runtime.get("iteration")
    started_at = as_float(runtime.get("started_at"))
    phase = str(runtime.get("phase") or "")
    current_seconds: float | None = None
    display_current_iteration: int | None = None
    current_source: str | None = None
    if (
        active
        and isinstance(current_iteration, int)
        and (next_iteration is None or current_iteration == next_iteration)
        and started_at is not None
        and phase != "completed"
        and 0 <= started_at <= time.time() + 5
    ):
        current_seconds = max(0.0, time.time() - started_at)
        display_current_iteration = current_iteration + int(global_iteration_offset)
        current_source = "trainer runtime"
    elif (
        active
        and isinstance(next_iteration, int)
        and progress_iteration == next_iteration
        and bool(progress_stage)
    ):
        # A trainer already running when this telemetry feature is deployed
        # cannot import the new runtime writer until its next natural restart.
        # Persist the first dashboard observation of each new progress-bound
        # iteration so browser refreshes and snapshot subprocesses do not
        # reset the live timer. Committed metrics remain the exact authority.
        observed = read_json(DASHBOARD_ITERATION_TIMER)
        observed_started = as_float(observed.get("started_at"))
        if (
            observed.get("run") != run_dir.name
            or observed.get("iteration") != next_iteration
            or observed_started is None
            or observed_started > time.time() + 5
        ):
            observed_started = time.time()
            payload = {
                "schema": "poke_bot.dashboard_iteration_timer/v1",
                "run": run_dir.name,
                "iteration": next_iteration,
                "started_at": observed_started,
                "source": "first run-bound progress observation",
            }
            DASHBOARD_ITERATION_TIMER.parent.mkdir(parents=True, exist_ok=True)
            tmp = DASHBOARD_ITERATION_TIMER.with_name(
                f".{DASHBOARD_ITERATION_TIMER.name}.{os.getpid()}.tmp"
            )
            tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, DASHBOARD_ITERATION_TIMER)
        current_seconds = max(0.0, time.time() - float(observed_started))
        display_current_iteration = next_iteration + int(global_iteration_offset)
        current_source = "dashboard progress observation"
    return {
        "available": bool(history or current_seconds is not None),
        "current_iteration": display_current_iteration,
        "current_seconds": current_seconds,
        "current_source": current_source,
        "latest_iteration": latest.get("iteration") if latest else None,
        "latest_seconds": latest.get("seconds") if latest else None,
        "rolling5_seconds": (
            sum(float(row["seconds"]) for row in rolling) / len(rolling)
            if rolling
            else None
        ),
        "rolling5_samples": len(rolling),
        "history": history,
        "source": "committed metrics + persisted live iteration timer",
    }


def _run_name_from_command(command: str) -> str | None:
    """Extract the launcher's explicit run identity from argv/systemd text."""
    match = re.search(r"(?:^|\s)--run-name(?:=|\s+)([^\s;\]}]+)", str(command))
    return match.group(1).strip("'\"") if match else None


def _active_curriculum_services() -> tuple[list[str], list[int], str | None]:
    """Return active units/PIDs and their authoritative ``--run-name``."""
    units = run(
        [
            "systemctl",
            "--user",
            "--no-legend",
            "--plain",
            "list-units",
            "--type=service",
            "--state=active",
        ]
    )
    active_units: list[str] = []
    active_pids: list[int] = []
    # RemainAfterExit bootstrap units intentionally stay ``active (exited)``.
    # They are useful history, but they are not live trainers.  A no-PID unit
    # must never make the dashboard active or select its historical run.  A
    # newly starting simple service will publish MainPID by the next dashboard
    # sample; showing one brief inactive sample is safer than lying about the
    # lineage/model contract.
    live_run_name: str | None = None
    for line in units.splitlines():
        parts = line.split()
        if not parts:
            continue
        unit = parts[0]
        if "pure-rl" not in unit.lower() and "curriculum" not in unit.lower():
            continue
        pid = as_number(
            run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "--property=MainPID",
                    "--value",
                ]
            )
        )
        if not pid:
            continue
        active_units.append(unit)
        command = ""
        active_pids.append(pid)
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(
                b"\0", b" "
            ).decode("utf-8", errors="replace")
        except OSError:
            command = ""
        if not command:
            command = run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "--property=ExecStart",
                    "--value",
                ]
            )
        candidate_run_name = _run_name_from_command(command)
        if candidate_run_name:
            live_run_name = live_run_name or candidate_run_name
    return active_units, active_pids, live_run_name


def _select_curriculum_run_dir(
    root: Path,
    candidates: set[Path],
    active_run_name: str | None,
) -> Path | None:
    """Select the active service's run; mtime is fallback-only history."""
    if active_run_name:
        return root / active_run_name
    return (
        max(
            candidates,
            key=lambda path: max(
                (child.stat().st_mtime for child in path.glob("*.json")),
                default=path.stat().st_mtime,
            ),
        )
        if candidates
        else None
    )


def expert_rehearsal_state(
    run_dir: Path | None,
    contract: dict[str, Any],
    loop: dict[str, Any],
    progress: dict[str, Any],
    *,
    global_iteration_offset: int,
    trainer_active: bool,
) -> dict[str, Any]:
    """Describe the recurring expert tune-up separately from bootstrap."""
    every = int(contract.get("every_iterations") or 0)
    epochs = int(contract.get("epochs") or 0)
    lineage_iteration = loop.get("next_iteration")
    due = bool(
        every > 0
        and isinstance(lineage_iteration, int)
        and lineage_iteration > 0
        and lineage_iteration % every == 0
    )
    receipts: list[Path] = []
    if run_dir is not None:
        receipts = sorted(
            (run_dir / "rehearsals").glob("before_iter_*.json"),
            key=lambda path: path.name,
        )
    latest_receipt = read_json(receipts[-1]) if receipts else {}
    latest_before = latest_receipt.get("before_iteration")
    completed_current = bool(
        due
        and isinstance(lineage_iteration, int)
        and latest_before == lineage_iteration
    )
    progress_stage = str(progress.get("stage") or "")
    running = bool(trainer_active and progress_stage.startswith("train:expert"))
    if every <= 0:
        state = "disabled"
    elif running:
        state = "running"
    elif due and not completed_current:
        state = "due · waiting/retry"
    elif completed_current:
        state = "complete for this boundary"
    else:
        state = "scheduled"
    next_lineage: int | None = None
    if every > 0 and isinstance(lineage_iteration, int):
        if due and not completed_current:
            next_lineage = lineage_iteration
        else:
            next_lineage = ((lineage_iteration // every) + 1) * every
    current = dict(progress) if progress_stage.startswith("train:expert") else {}
    return {
        "available": bool(contract),
        "kind": "periodic_expert_tune_up",
        "is_bootstrap": False,
        "state": state,
        "active": running,
        "due": due and not completed_current,
        "every_iterations": every,
        "epochs": epochs,
        "learning_rate": as_float(contract.get("learning_rate")),
        "requested_batch_size": as_number(
            str(contract.get("requested_batch_size") or "")
        ),
        "minimum_decisions": as_number(
            str(contract.get("minimum_decisions") or "")
        ),
        "manifest": contract.get("rolling_manifest_pointer"),
        "current": current,
        "latest_completed_iteration": (
            int(latest_before) + global_iteration_offset
            if isinstance(latest_before, int)
            else None
        ),
        "latest_receipt": str(receipts[-1]) if receipts else None,
        "next_iteration": (
            next_lineage + global_iteration_offset
            if isinstance(next_lineage, int)
            else None
        ),
    }


def curriculum_state() -> dict[str, Any]:
    root = ROOT / "outputs/pure_rl"
    active_units, active_pids, active_run_name = _active_curriculum_services()
    candidates = {
        p.parent
        for pattern in ("*/manifest.json", "*/loop_state.json")
        for p in root.glob(pattern)
        if p.is_file()
    }
    run_dir = _select_curriculum_run_dir(
        root,
        candidates,
        active_run_name,
    )
    metrics = read_json(run_dir / "metrics/latest.json") if run_dir else {}
    loop = read_json(run_dir / "loop_state.json") if run_dir else {}
    manifest = read_json(run_dir / "manifest.json") if run_dir else {}
    handoff = read_json(run_dir / "lineage_handoff.json") if run_dir else {}
    # A fail-closed heldout repair can finish after the trainer intentionally
    # stops but before the normal append-only iteration commit.  That exact
    # audit is newer and more authoritative than inherited historical WR, so
    # surface it explicitly instead of leaving the dashboard on a stale tqdm.
    recovery_path: Path | None = None
    recovery: dict[str, Any] = {}
    if run_dir is not None:
        recovery_candidates = sorted(
            (run_dir / "eval").glob("iter_*.heldout_recovery.json"),
            key=lambda path: path.stat().st_mtime,
        )
        if recovery_candidates:
            recovery_path = recovery_candidates[-1]
            candidate = read_json(recovery_path)
            audit = candidate.get("audit")
            gate = candidate.get("gate")
            if (
                isinstance(audit, dict)
                and audit.get("passed") is True
                and isinstance(gate, dict)
                and int(gate.get("games") or 0) == int(audit.get("valid_games") or -1)
            ):
                recovery = candidate
    global_iteration_offset = int(handoff.get("global_iteration_offset") or 0)
    official_heldout = committed_official_heldout_state(
        loop,
        run_dir,
        global_iteration_offset=global_iteration_offset,
        handoff=handoff,
    )
    run_name = (
        active_run_name
        or loop.get("run_name")
        or (run_dir.name if run_dir else None)
    )
    run_status = (
        ROOT / "outputs/logs" / f"{run_name}.progress.status"
        if run_name
        else None
    )
    run_progress_log = (
        ROOT / "outputs/logs" / f"{run_name}.progress.log"
        if run_name
        else None
    )
    # Once a run identity exists, never fall back to the global alias: it may
    # still point at a previous lineage during the first seconds of launch.
    status_path = run_status if run_status is not None else TRAINING_STATUS
    raw_status = read_tail(status_path, 20_000).strip()
    raw_progress_log = (
        read_tail(run_progress_log, 500_000)
        if run_progress_log is not None
        else ""
    )
    iteration_hint = as_number(str(loop.get("next_iteration", "")))
    progress = parse_curriculum_progress(
        raw_status,
        raw_progress_log,
        iteration_hint=iteration_hint,
    )
    raw_training_log = read_tail(TRAINING_LOG, 250_000)
    progress = infer_between_bar_progress(
        progress,
        raw_training_log,
        iteration_hint=iteration_hint,
        train_epochs=2,
    )
    replay_window = replay_window_state(
        run_dir,
        loop,
        manifest,
        progress,
        raw_training_log,
    )
    lineage_iteration = progress.get("iteration")
    display_progress = dict(progress)
    if global_iteration_offset and isinstance(lineage_iteration, int):
        display_iteration = lineage_iteration + global_iteration_offset
        display_progress["lineage_iteration"] = lineage_iteration
        display_progress["iteration"] = display_iteration
        display_progress["line"] = re.sub(
            rf"\biter={lineage_iteration}\b",
            f"iter={display_iteration}",
            str(display_progress.get("line") or ""),
        )
    if global_iteration_offset and isinstance(replay_window.get("iteration"), int):
        replay_window["lineage_iteration"] = replay_window["iteration"]
        replay_window["iteration"] = (
            int(replay_window["iteration"]) + global_iteration_offset
        )
    expert_contract = (
        (manifest.get("design_contract") or {}).get("expert_rehearsal") or {}
    )
    expert_receipt = (
        run_dir / "rehearsals" / "before_iter_00000.json"
        if run_dir is not None
        else None
    )
    expert_startup_pending = bool(
        active_units
        and int(loop.get("next_iteration") or 0) == 0
        and expert_contract.get("before_first_iteration") is True
        and expert_receipt is not None
        and not expert_receipt.is_file()
        and not raw_status
        and progress.get("stage") is None
    )
    if expert_startup_pending:
        lineage_it = int(loop.get("next_iteration") or 0)
        display_it = lineage_it + global_iteration_offset
        progress.update(
            {
                "line": (
                    f"pure_rl train:expert iter={display_it}: loading exact "
                    "top-ladder corpus onto Blackwell"
                ),
                "stage": "train:expert",
                "iteration": lineage_it,
                "percent": None,
                "current": 0,
                "total": 1,
                "unit": "expert pass",
                "eta": "loading corpus",
            }
        )
        display_progress = dict(progress)
        display_progress["lineage_iteration"] = lineage_it
        display_progress["iteration"] = display_it
    expert_rehearsal = expert_rehearsal_state(
        run_dir,
        expert_contract,
        loop,
        display_progress,
        global_iteration_offset=global_iteration_offset,
        trainer_active=bool(active_units),
    )
    worker = curriculum_worker_state(active_units, active_pids)
    iteration_timing = iteration_timing_state(
        run_dir,
        active=bool(active_units),
        global_iteration_offset=global_iteration_offset,
        next_iteration=(
            int(loop.get("next_iteration"))
            if isinstance(loop.get("next_iteration"), int)
            else None
        ),
        progress_iteration=(
            int(progress.get("iteration"))
            if isinstance(progress.get("iteration"), int)
            else None
        ),
        progress_stage=str(progress.get("stage") or "") or None,
    )
    public_mix_live = read_json(PUBLIC_MIX_LIVE_WR)
    public_mix_age = (
        max(0.0, time.time() - float(public_mix_live.get("updated_at") or 0.0))
        if public_mix_live
        else None
    )
    public_mix_iteration = public_mix_live.get("iteration")
    if (
        not public_mix_live
        or public_mix_live.get("run") != run_name
        or not isinstance(public_mix_iteration, int)
        or public_mix_age is None
        or public_mix_age > 15.0
    ):
        public_mix_live = {
            "available": False,
            "active": False,
            "reason": "live public-mix outcome sidecar is unavailable or stale",
        }
    else:
        public_mix_live = dict(public_mix_live)
        public_mix_live["lineage_iteration"] = public_mix_iteration
        public_mix_live["iteration"] = public_mix_iteration + global_iteration_offset
        public_mix_live["age_s"] = public_mix_age
    extra = metrics.get("extra") if isinstance(metrics.get("extra"), dict) else {}
    promotion = extra.get("promotion") if isinstance(extra.get("promotion"), dict) else {}
    champion = loop.get("champion") if isinstance(loop.get("champion"), dict) else {}
    inherited_heldout = (
        handoff.get("inherited_heldout")
        if isinstance(handoff.get("inherited_heldout"), dict)
        else {}
    )
    metrics_iteration = metrics.get("iteration")
    recovery_iteration = recovery.get("iteration")
    recovery_is_latest = bool(recovery) and (
        not isinstance(metrics_iteration, int)
        or not isinstance(recovery_iteration, int)
        or recovery_iteration >= metrics_iteration
    )
    heldout_inherited = (
        metrics.get("heldout_wr") is None
        and not recovery_is_latest
        and bool(inherited_heldout)
    )
    recovery_gate = recovery.get("gate") if recovery_is_latest else {}
    heldout_wr = (
        recovery_gate.get("win_rate")
        if recovery_is_latest
        else (
            inherited_heldout.get("win_rate")
            if heldout_inherited
            else metrics.get("heldout_wr")
        )
    )
    heldout_games = (
        recovery_gate.get("games")
        if recovery_is_latest
        else (
            inherited_heldout.get("games")
            if heldout_inherited
            else metrics.get("heldout_games")
        )
    )
    gate_passed = (
        recovery_gate.get("passed")
        if recovery_is_latest
        else (
            inherited_heldout.get("passed")
            if heldout_inherited
            else metrics.get("gate_passed")
        )
    )
    if recovery_is_latest and not active_units:
        display_iteration = (
            int(recovery_iteration) + global_iteration_offset
            if isinstance(recovery_iteration, int)
            else display_progress.get("iteration")
        )
        display_progress.update(
            {
                "line": (
                    f"pure_rl heldout COMPLETE iter={display_iteration}: "
                    f"{int(heldout_games or 0)}/{int(heldout_games or 0)} "
                    f"[WR={float(heldout_wr or 0.0) * 100:.1f}% · exact audit PASS]"
                ),
                "stage": "heldout:complete",
                "iteration": display_iteration,
                "lineage_iteration": recovery_iteration,
                "percent": 100.0,
                "current": int(heldout_games or 0),
                "total": int(heldout_games or 0),
                "unit": "games",
                "eta": "done",
            }
        )
    status_updated_at = status_path.stat().st_mtime if status_path.is_file() else None
    log_updated_at = (
        run_progress_log.stat().st_mtime
        if run_progress_log is not None and run_progress_log.is_file()
        else None
    )
    progress_updated_at = max(
        (value for value in (status_updated_at, log_updated_at) if value is not None),
        default=None,
    )
    status_age_s = time.time() - progress_updated_at if progress_updated_at else None
    progress_current = bool(
        run_name
        and run_status is not None
        and status_path == run_status
        and status_path.is_file()
    ) or bool(
        run_progress_log is not None
        and run_progress_log.is_file()
        and progress.get("stage") is not None
    ) or expert_startup_pending
    assertions = {
        "active_service_has_pid": not active_units or bool(active_pids),
        "run_identity_present": not active_units or bool(run_name),
        "active_run_is_authoritative": (
            not active_run_name
            or bool(run_dir is not None and run_dir.name == active_run_name)
        ),
        "progress_bound_to_run": not active_units or progress_current,
        "progress_not_cross_run": run_status is None or status_path == run_status,
        "progress_log_bound_to_run": (
            not active_units
            or bool(run_progress_log is not None and run_progress_log.is_file())
        ),
    }
    progress_source = (
        run_dir / "manifest.json"
        if expert_startup_pending and run_dir is not None
        else run_progress_log
        if str(progress.get("stage") or "").startswith("train")
        and run_progress_log is not None
        else status_path
    )
    return {
        "active": bool(active_units),
        "active_units": active_units,
        "active_pids": active_pids,
        "run": run_name,
        "last_completed_iteration": (
            int(loop.get("last_completed_iteration")) + global_iteration_offset
            if isinstance(loop.get("last_completed_iteration"), int)
            else loop.get("last_completed_iteration")
        ),
        "next_iteration": (
            int(loop.get("next_iteration")) + global_iteration_offset
            if isinstance(loop.get("next_iteration"), int)
            else loop.get("next_iteration")
        ),
        "global_iteration_offset": global_iteration_offset,
        "lineage_iteration": lineage_iteration,
        "stage": (
            "heldout:complete"
            if recovery_is_latest and not active_units
            else progress["stage"] or metrics.get("stage") or ("starting" if active_units else None)
        ),
        "iteration": (
            display_progress["iteration"]
            if display_progress["iteration"] is not None
            else (
                int(metrics.get("iteration", loop.get("next_iteration")))
                + global_iteration_offset
                if isinstance(metrics.get("iteration", loop.get("next_iteration")), int)
                else metrics.get("iteration", loop.get("next_iteration"))
            )
        ),
        "progress": display_progress,
        "replay_window": replay_window,
        "iteration_timing": iteration_timing,
        "expert_rehearsal": expert_rehearsal,
        "public_mix_live": public_mix_live,
        "progress_source": str(progress_source),
        "progress_status_source": str(status_path),
        "progress_log_source": str(run_progress_log) if run_progress_log else None,
        "progress_updated_at": progress_updated_at,
        "progress_age_s": status_age_s,
        "source_assertions": assertions,
        "source_current": all(assertions.values()),
        "worker": worker,
        "games": metrics.get("games"),
        "gps": progress["gps"] if progress["gps"] is not None else metrics.get("games_per_sec"),
        "sps": progress["sps"] if progress["sps"] is not None else metrics.get("decisions_per_sec"),
        "heldout_wr": heldout_wr,
        "heldout_games": heldout_games,
        "heldout_inherited": heldout_inherited,
        "heldout_recovery": recovery_is_latest,
        "heldout_audit_passed": bool(
            recovery_is_latest and (recovery.get("audit") or {}).get("passed")
        ),
        "heldout_matchups": (
            recovery_gate.get("per_opponent") if recovery_is_latest else None
        ),
        "heldout_source": str(recovery_path) if recovery_is_latest else None,
        "heldout_source_run": handoff.get("source_run") if heldout_inherited else run_name,
        "heldout_source_iteration": (
            recovery_iteration
            if recovery_is_latest
            else (
                handoff.get("source_iteration")
                if heldout_inherited
                else metrics.get("iteration")
            )
        ),
        "official_heldout": official_heldout,
        "gate_passed": gate_passed,
        "promotion_wr": promotion.get("wr"),
        "promotion_passed": promotion.get("passed"),
        "remote_workers": extra.get("remote_workers", progress.get("remotes")),
        "remote_request_sockets": (
            (progress.get("metrics") or {}).get("remote_request_sockets")
        ),
        "remote_queue_capacity": (
            (progress.get("metrics") or {}).get("remote_queue_capacity")
        ),
        "remote_dispatch": (
            handoff.get("remote_dispatch")
            if isinstance(handoff.get("remote_dispatch"), dict)
            else {}
        ),
        "scheduler_queues": scheduler_queue_state(run_name),
        "model_contract": learner_model_state(
            manifest,
            loop,
            iteration=(
                int(lineage_iteration)
                if isinstance(lineage_iteration, int)
                else None
            ),
        ),
        "champion": champion.get("path"),
        "updated_at": (
            recovery_path.stat().st_mtime
            if recovery_is_latest and recovery_path is not None
            else (
                (run_dir / "metrics/latest.json").stat().st_mtime
                if run_dir and (run_dir / "metrics/latest.json").is_file()
                else status_updated_at
            )
        ),
        "remote_endpoints": (
            (((manifest.get("design_contract") or {}).get("remotes") or {}).get("endpoints"))
            or (((manifest.get("contract") or {}).get("remotes") or {}).get("endpoints"))
            or []
        ),
    }


def elmo_state() -> dict[str, Any]:
    raw = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=3",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=60",
            "-o",
            "ControlPath=/tmp/pokebot-dashboard-elmo-ssh",
            "elmo",
            "/mnt/Main/Elmo/poke-bot-agent/dashboard/fleet_host_snapshot.py",
            "--role",
            "simulator",
            "--name",
            "Elmo",
        ],
        timeout=6,
    )
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {
            "reachable": False,
            "name": "Elmo",
            "role": "simulator",
            "error": "telemetry unavailable",
        }


def expert_refresh_state() -> dict[str, Any]:
    """Read the isolated latest-ten expert refresh running on Elmo."""
    raw = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=3",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=60",
            "-o",
            "ControlPath=/tmp/pokebot-dashboard-expert-refresh-ssh",
            "elmo",
            "/mnt/Main/main/poke-feature-refresh-20260721/expert_refresh_status.py",
            "--root",
            "/mnt/Main/main/poke-feature-refresh-20260721",
            "--host",
            "Elmo",
        ],
        timeout=6,
    )
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except (TypeError, json.JSONDecodeError):
        pass
    return {
        "available": False,
        "active": False,
        "host": "Elmo",
        "reason": "expert refresh telemetry unavailable",
    }


def latest10_state() -> dict[str, Any]:
    local_raw = run(
        [
            str(LATEST10_STATUS),
            "--root",
            str(ROOT),
            "--host",
            "Inzi",
        ],
        timeout=5,
    )
    remote_raw = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=3",
            "elmo",
            "/mnt/Main/main/poke-feature-latest10/scripts/latest10_status.py",
            "--root",
            "/mnt/Main/main/poke-feature-latest10",
            "--host",
            "Elmo",
        ],
        timeout=6,
    )

    def decoded(raw: str) -> dict[str, Any]:
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except (TypeError, json.JSONDecodeError):
            return {}

    candidates: dict[str, list[dict[str, Any]]] = {}
    for payload in (decoded(local_raw), decoded(remote_raw)):
        for row in payload.get("days") or []:
            if isinstance(row, dict) and row.get("day"):
                candidates.setdefault(str(row["day"]), []).append(row)
    days: list[dict[str, Any]] = []
    for day in [f"2026-07-{value:02d}" for value in range(9, 19)]:
        options = candidates.get(day) or [
            {"day": day, "host": "—", "stage": "waiting", "percent": 0.0}
        ]
        days.append(
            max(
                options,
                key=lambda row: (
                    float(row.get("percent") or 0.0),
                    bool((row.get("service") or {}).get("active")),
                ),
            )
        )
    bert_status = read_json(LATEST10_BERT_STATUS)
    bert_days = dict(bert_status.get("days") or {})
    for row in days:
        staged = bert_days.get(str(row.get("day")))
        if isinstance(staged, dict):
            row["bert"] = staged
    active_days = [
        row for row in days if bool((row.get("service") or {}).get("active"))
    ]
    ready_days = [row for row in days if row.get("stage") == "ready"]
    ready_marker = LATEST10_READY.is_file()
    if ready_marker:
        for row in days:
            row.setdefault(
                "bert",
                {
                    "day": row.get("day"),
                    "stage": "ready",
                    "host": "Bert",
                    "message": "Shard is included in the Bert-verified final manifest.",
                },
            )
    bert_ready_days = [
        row for row in days if (row.get("bert") or {}).get("stage") == "ready"
    ]
    shards_ready = len(ready_days) == 10
    finalizer = unit_state(LATEST10_FINALIZER_SERVICE)
    bootstrap = unit_state(LATEST10_BOOTSTRAP_SERVICE)
    finalizer_log = ANSI_RE.sub("", read_tail(LATEST10_FINALIZER_LOG)).replace(
        "\r", "\n"
    )
    finalizer_lines = [line.strip() for line in finalizer_log.splitlines() if line.strip()]

    current = max(
        active_days
        or [row for row in days if float(row.get("percent") or 0) < 100]
        or days,
        key=lambda row: float(row.get("percent") or 0.0),
    )
    stage = current.get("stage")
    host = current.get("host")
    latest_line = current.get("latest_line")
    current_value = current.get("current")
    total_value = current.get("total")
    unit = current.get("unit")
    rate = current.get("rate")
    current_service = current.get("service")

    if bootstrap["active"]:
        stage = "training on Blackwell"
        host = "Inzi"
        latest_line = "Latest-ten manifest validated; Blackwell bootstrap is active."
        current_value = len(ready_days)
        total_value = 10
        unit = "validated shards"
        rate = None
        current_service = bootstrap
    elif finalizer["active"]:
        last_run = next(
            (line for line in reversed(finalizer_lines) if line.startswith("[run]")),
            "",
        )
        if "assemble_feature_manifest.py" in last_run:
            stage = "assembling manifest"
            host = "Bert"
            latest_line = "Bert is hashing ten compact feature shards and assembling the manifest."
        elif "rsync" in last_run and re.search(r"bert:\S+/\s+/home/inzi/", last_run):
            stage = "returning manifest"
            host = "Bert → Inzi"
            latest_line = "Bert assembly is returning to Inzi for final digest verification."
        elif "rsync" in last_run and re.search(r"\sbert:\S+/?$", last_run):
            stage = "staging on Bert"
            host = "Inzi → Bert"
            latest_line = "Ten validated compact feature shards are staging on Bert."
        else:
            stage = "finalizing"
            host = "Inzi"
            latest_line = "All day shards are ready; the finalizer is validating the bundle."
        current_value = len(ready_days)
        total_value = 10
        unit = "validated shards"
        rate = None
        current_service = finalizer
    elif ready_marker:
        stage = "ready"
        host = "Inzi"
        latest_line = "Latest-ten manifest and post-transfer digests are validated."

    percent = sum(float(row.get("percent") or 0.0) for row in days) / 10.0
    if shards_ready and not ready_marker:
        percent = (
            min(99.0, 90.0 + len(bert_ready_days))
            if bert_days
            else 99.0
        )
    return {
        "active": bool(active_days) or finalizer["active"] or bootstrap["active"],
        "complete": ready_marker,
        "shards_ready": shards_ready,
        "started": any(float(row.get("percent") or 0.0) > 0 for row in days),
        "completed_days": len(ready_days),
        "bert_ready_days": len(bert_ready_days),
        "total_days": 10,
        "percent": percent,
        "stage": stage,
        "host": host,
        "current_day": current.get("day"),
        "current": current_value,
        "total": total_value,
        "unit": unit,
        "rate": rate,
        "latest_line": latest_line,
        "current_service": current_service,
        "finalizer_service": finalizer,
        "bootstrap_service": bootstrap,
        "days": days,
    }


def main() -> None:
    # Elmo is an independent host. Fetch its three views concurrently so a
    # slow SSH handshake cannot serialize into the outer Bert→Inzi timeout.
    with ThreadPoolExecutor(max_workers=3) as remote_pool:
        elmo_future = remote_pool.submit(elmo_state)
        latest10_future = remote_pool.submit(latest10_state)
        expert_refresh_future = remote_pool.submit(expert_refresh_state)
        system = system_state()
        service = service_state()
        transition = transition_state()
        curriculum = curriculum_state()
        gpus = gpu_state()
        elmo = elmo_future.result()
        latest10 = latest10_future.result()
        expert_refresh = expert_refresh_future.result()
    curriculum_worker = curriculum.get("worker") or {}
    for gpu in gpus:
        index = int(gpu.get("index") or 0)
        if (
            index == 0
            and curriculum.get("active")
            and int(curriculum_worker.get("leaf_gpu0_replicas") or 0) == 0
        ):
            gpu["production_active"] = False
            gpu["assignment"] = "OUT OF FLEET · CUDA simulator testing"
        elif index == 1 and curriculum.get("active"):
            gpu["production_active"] = True
            gpu["assignment"] = "PRODUCTION · policy leaves + trainer"
    training = (
        alakazam_bootstrap_progress()
        if ALAKAZAM_BOOTSTRAP_READY.is_file()
        or (
            transition.get("bootstrap", {}).get("active")
            and (
                int(transition.get("bootstrap", {}).get("pid") or 0) > 0
                or transition.get("bootstrap", {}).get("sub_state") == "running"
            )
        )
        or transition.get("status")
        == "training_alakazam_expert_bootstrap_blackwell_device_resident"
        else exact_training_state()
    )
    baseline_eval = baseline_eval_state()
    print(
        json.dumps(
            {
                "ok": True,
                "observed_at": time.time(),
                "system": system,
                "service": service,
                "transition": transition,
                "training": training,
                "bootstrap": training,
                "baseline_eval": baseline_eval,
                "latest10": latest10,
                "expert_refresh": expert_refresh,
                "curriculum": curriculum,
                "gpus": gpus,
                "fleet": {
                    "inzi": {
                        "reachable": True,
                        "observed_at": time.time(),
                        "name": "Inzi",
                        "role": "trainer + simulator",
                        "platform": "linux",
                        "system": system,
                        "gpus": gpus,
                        "worker": {
                            **(curriculum.get("worker") or {}),
                            "active": service["active"] or curriculum["active"],
                            "command": (
                                (curriculum.get("worker") or {}).get("command")
                                or service.get("command")
                                or ", ".join(curriculum["active_units"])
                            ),
                        },
                    },
                    "elmo": elmo,
                },
                "model": {
                    **(curriculum.get("model_contract") or {}),
                    "run": curriculum.get("run"),
                },
                "pure_rl_status": curriculum.get("progress", {}).get("line", ""),
                "recent_events": recent_events(curriculum.get("run")),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
