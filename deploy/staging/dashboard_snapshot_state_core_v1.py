#!/usr/bin/env python3
"""Emit one lightweight JSON snapshot for the LAN training dashboard."""

from __future__ import annotations

import ast
import hashlib
import json
import math
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
STRONG_PUBLIC_GATE_SERVICE = "pokebot-alakazam-strong-public-gate.service"
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
DORMANT_MODEL_MODULES = ROOT / "outputs/state/alakazam_dormant_model_modules_v1.json"
PUBLIC_MIX_LIVE_WR = ROOT / "outputs/state/public_mix_live_wr.json"
COMPETITION_GATE_PROGRAM = ROOT / "ops/alakazam_gate_program_v1.json"
RESEARCH_CONTROL_REGISTRY = ROOT / "ops/research_control_registry_v1.json"
RESEARCH_CONTROL_REGISTRY_LATEST = (
    ROOT / "outputs/state/research_control_registry_latest.json"
)
STRONG_PUBLIC_GATE_PROGRESS = (
    ROOT / "outputs/logs/alakazam_strong_public_gate.progress.status"
)
STRONG_PUBLIC_GATE_LOG = ROOT / "outputs/logs/alakazam_strong_public_gate.progress.log"
PROTECTED_BASELINE_GATE = Path(
    "/home/inzi/poke-bot-model-registry/alakazam_baseline_gate/manifest.json"
)
LEGACY_RESEARCH_CONTROL_DIGESTS = {
    "iono": "sha256:6ba8e818b698774b6e437364e9457600eda950fbefb663d8e4ad39cdaf0371e2",
    "dragapult-ex": "sha256:835dcbcc26366faa04d902db727620d4b12618b6a66d000dccb9c9b86e9d62a0",
    "mega-abomasnow-ex": "sha256:57a9499b2bee493a830abaf5a3e19b8a73faea200faee87aeeb2864bab25c2fb",
    "mega-lucario-ex": "sha256:98f20936d430c6cc60f3eb1da8230392bf6dce8ecacf97773bda4db63f56376a",
}
OFFICIAL_BASELINE_IDS = tuple(LEGACY_RESEARCH_CONTROL_DIGESTS)


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


def _is_sha256_digest(value: object) -> bool:
    return re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")) is not None


def _canonical_json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


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
    optimizer_runtime = {
        "awr_beta": as_float(environment.get("PURE_RL_AWR_BETA")),
        "awr_weight_max": as_float(
            environment.get("PURE_RL_AWR_WEIGHT_MAX")
        ),
    }
    optimizer_runtime = {
        key: value for key, value in optimizer_runtime.items() if value is not None
    }
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
        "optimizer_runtime": optimizer_runtime,
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


def checkpoint_parameter_telemetry(log_path: Path) -> dict[str, Any]:
    """Return the latest parameter count produced by an actual checkpoint load."""
    raw = ANSI_RE.sub("", read_tail(log_path, 512_000)).replace("\r", "\n")
    matches = list(
        re.finditer(
            r"\[pure_rl\] loaded checkpoint params=(\d+) path=(\S+)",
            raw,
        )
    )
    if not matches:
        return {}
    latest = matches[-1]
    count = int(latest.group(1))
    if count <= 0:
        return {}
    return {
        "trainable_parameters": count,
        "checkpoint": latest.group(2),
        "source": str(log_path),
    }


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
        audit = evidence.get("audit")
        report_identity = audit.get("report") if isinstance(audit, dict) else None
        if (
            iteration == -1
            and isinstance(audit, dict)
            and audit.get("passed") is True
            and audit.get("source") == "trusted_external_new_lineage_anchor"
            and audit.get("terminal_gate_eligible") is False
            and isinstance(report_identity, dict)
        ):
            report_path = Path(str(report_identity.get("path") or ""))
            try:
                payload = report_path.read_bytes()
                report_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
                report = json.loads(payload)
            except (OSError, ValueError, TypeError):
                return {"available": False, "reason": "seed audit report is unreadable"}
            expected_digest = str(report_identity.get("digest") or "")
            checkpoint = report.get("checkpoint")
            pooled = report.get("pooled_formal")
            deck_gate = report.get("deck_agnostic_gate")
            report_matchups = report.get("matchups")
            evidence_matchups = evidence.get("per_opponent")
            if (
                report_digest != expected_digest
                or not isinstance(checkpoint, dict)
                or str(checkpoint.get("digest") or "") != digest
                or report.get("valid") is not True
                or report.get("trusted_formal") is not True
                or report.get("formal_mode") != "policy"
                or list(report.get("failures") or [])
                or not isinstance(pooled, dict)
                or not isinstance(deck_gate, dict)
                or deck_gate.get("exact_deck_seat_balance") is not True
                or not isinstance(report_matchups, list)
                or not isinstance(evidence_matchups, dict)
            ):
                return {"available": False, "reason": "seed audit report failed reconciliation"}
            games = int(evidence.get("games") or 0)
            pooled_wr = as_float(pooled.get("wr"))
            evidence_wr = as_float(evidence.get("win_rate"))
            if (
                games <= 0
                or int(report.get("scheduled_jobs") or 0) != games
                or int(report.get("completed_jobs") or 0) != games
                or int(pooled.get("games") or 0) != games
                or pooled_wr is None
                or evidence_wr is None
                or abs(pooled_wr - evidence_wr) > 1e-12
            ):
                return {"available": False, "reason": "seed audit totals mismatch"}
            by_id = {
                str(row.get("opponent_id") or ""): row
                for row in report_matchups
                if isinstance(row, dict)
            }
            if set(by_id) != set(OFFICIAL_BASELINE_IDS):
                return {"available": False, "reason": "seed audit opponent mismatch"}
            matchups: list[dict[str, Any]] = []
            for opponent_id in OFFICIAL_BASELINE_IDS:
                row = by_id[opponent_id]
                anchored = evidence_matchups.get(opponent_id)
                if not isinstance(anchored, dict):
                    return {"available": False, "reason": "seed audit matchup missing"}
                matchup_games = int(row.get("games") or 0)
                wins = as_float(row.get("wins"))
                draws = as_float(row.get("draws"))
                anchored_wins = as_float(anchored.get("wins"))
                if wins is None or draws is None or anchored_wins is None:
                    return {"available": False, "reason": "seed audit matchup mismatch"}
                score = wins + 0.5 * draws
                if (
                    matchup_games <= 0
                    or matchup_games % 2
                    or int(anchored.get("games") or 0) != matchup_games
                    or abs(anchored_wins - score) > 1e-12
                ):
                    return {"available": False, "reason": "seed audit matchup mismatch"}
                matchups.append(
                    {
                        "opponent_id": opponent_id,
                        "games": matchup_games,
                        "wr": score / matchup_games,
                        "wins": score,
                        "draws": draws,
                        "losses": float(row.get("losses") or 0.0),
                        "seat0": matchup_games // 2,
                        "seat1": matchup_games // 2,
                    }
                )
            return {
                "available": True,
                "kind": "external_seed_official_heldout_anchor",
                "valid": True,
                "passed": False,
                "reason": "nonterminal_seed_audit",
                "games": games,
                "wr": as_float(evidence.get("win_rate")),
                "lower": as_float(evidence.get("confidence_lower")),
                "upper": as_float(evidence.get("confidence_upper")),
                "iteration": -1,
                "lineage_iteration": -1,
                "checkpoint": identity.get("path"),
                "checkpoint_digest": digest,
                "matchups": matchups,
                "opponent_count": len(matchups),
                "audit_passed": True,
                "exact_distribution": True,
                "exact_weights": True,
                "greedy_required": True,
                "terminal_gate_eligible": False,
                "source": str(report_path),
                "updated_at": report_path.stat().st_mtime,
            }
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


def latest_committed_official_heldout_state(
    loop: dict[str, Any],
    run_dir: Path | None,
    *,
    global_iteration_offset: int = 0,
) -> dict[str, Any]:
    """Return the newest fully audited official-baseline holdout attempt.

    This is deliberately separate from :func:`committed_official_heldout_state`.
    The latter tracks the protected best checkpoint; this view answers whether
    the most recent candidate actually ran its exact holdout, even when that
    candidate was rejected and the protected checkpoint therefore stayed put.
    """

    if run_dir is None:
        return {"available": False, "reason": "run directory is unavailable"}
    heldout_identity = (
        loop.get("heldout_champion")
        if isinstance(loop.get("heldout_champion"), dict)
        else {}
    )
    heldout_digest = str(heldout_identity.get("digest") or "")
    for history_row in reversed(loop.get("history") or []):
        if not isinstance(history_row, dict) or history_row.get("completed") is not True:
            continue
        iteration = history_row.get("iteration")
        candidate = history_row.get("candidate")
        audit = history_row.get("heldout_audit")
        gate = history_row.get("raw_heldout_gate")
        if (
            not isinstance(iteration, int)
            or not isinstance(candidate, dict)
            or not isinstance(audit, dict)
            or not isinstance(gate, dict)
        ):
            continue
        source = run_dir / "commits" / f"iter_{iteration:05d}.json"
        if not source.is_file():
            continue
        digest = str(candidate.get("digest") or "")
        audit_games = int(audit.get("valid_games") or 0)
        gate_games = int(gate.get("games") or 0)
        gate_wr = as_float(gate.get("win_rate"))
        audit_matchups = audit.get("per_opponent")
        gate_matchups = gate.get("per_opponent")
        if (
            not digest
            or audit.get("passed") is not True
            or audit.get("exact_distribution") is not True
            or audit.get("exact_weights") is not True
            or audit.get("greedy_required") is not True
            or str(audit.get("checkpoint_digest") or "") != digest
            or audit_games <= 0
            or audit_games != gate_games
            or gate_wr is None
            or not isinstance(audit_matchups, dict)
            or not isinstance(gate_matchups, dict)
            or set(audit_matchups) != set(OFFICIAL_BASELINE_IDS)
            or set(gate_matchups) != set(OFFICIAL_BASELINE_IDS)
        ):
            continue
        matchups: list[dict[str, Any]] = []
        valid = True
        for opponent_id in OFFICIAL_BASELINE_IDS:
            audit_row = audit_matchups.get(opponent_id)
            gate_row = gate_matchups.get(opponent_id)
            if not isinstance(audit_row, dict) or not isinstance(gate_row, dict):
                valid = False
                break
            games = int(gate_row.get("games") or 0)
            audit_row_games = int(audit_row.get("games") or 0)
            seat0 = int(audit_row.get("seat0") or gate_row.get("seat0_games") or 0)
            seat1 = int(audit_row.get("seat1") or gate_row.get("seat1_games") or 0)
            wr = as_float(gate_row.get("win_rate"))
            if (
                games <= 0
                or games != audit_row_games
                or seat0 + seat1 != games
                or wr is None
            ):
                valid = False
                break
            matchups.append(
                {
                    "opponent_id": opponent_id,
                    "games": games,
                    "wr": wr,
                    "wins": as_float(gate_row.get("wins")),
                    "draws": as_float(gate_row.get("draws")),
                    "losses": as_float(gate_row.get("losses")),
                    "seat0": seat0,
                    "seat1": seat1,
                }
            )
        if not valid or sum(int(row["games"]) for row in matchups) != gate_games:
            continue
        learner_after = (
            history_row.get("learner_after")
            if isinstance(history_row.get("learner_after"), dict)
            else {}
        )
        return {
            "available": True,
            "kind": "latest_committed_official_heldout_attempt",
            "valid": True,
            "passed": gate.get("passed") is True,
            "reason": gate.get("reason"),
            "games": gate_games,
            "wr": gate_wr,
            "lower": as_float(gate.get("confidence_lower")),
            "upper": as_float(gate.get("confidence_upper")),
            "iteration": iteration + int(global_iteration_offset),
            "lineage_iteration": iteration,
            "checkpoint": candidate.get("path"),
            "checkpoint_digest": digest,
            "matchups": matchups,
            "opponent_count": len(matchups),
            "audit_passed": True,
            "exact_distribution": True,
            "exact_weights": True,
            "greedy_required": True,
            "protected_champion": digest == heldout_digest,
            "heldout_champion_updated": history_row.get("heldout_champion_updated")
            is True,
            "learner_retained": str(learner_after.get("digest") or "") == digest,
            "source": str(source),
            "updated_at": source.stat().st_mtime,
        }
    return {"available": False, "reason": "no committed exact holdout attempt"}


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
    runtime_optimizer: dict[str, Any] | None = None,
    runtime_parameter_contract: dict[str, Any] | None = None,
    dormant_modules_path: Path = DORMANT_MODEL_MODULES,
) -> dict[str, Any]:
    """Describe the exact live model plus explicitly non-live staged profiles.

    The old dashboard hard-coded one parameter count. That looked current even
    after a model-profile change. Prefer immutable manifest metadata, then an
    independently deployed profile registry whose full config must match. If
    neither source matches, report an unknown count instead of a plausible lie.
    """
    design_contract = manifest.get("design_contract") or {}
    learner = design_contract.get("learner") or {}
    expert = design_contract.get("expert_rehearsal") or {}
    profile = learner.get("profile") if isinstance(learner.get("profile"), dict) else {}
    loop = loop if isinstance(loop, dict) else {}
    runtime_optimizer = (
        runtime_optimizer if isinstance(runtime_optimizer, dict) else {}
    )
    runtime_parameter_contract = (
        runtime_parameter_contract
        if isinstance(runtime_parameter_contract, dict)
        else {}
    )

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

    runtime_parameter_count = as_number(
        str(runtime_parameter_contract.get("trainable_parameters") or "")
    )
    parameter_count = (
        runtime_parameter_count
        if runtime_parameter_count is not None and runtime_parameter_count > 0
        else as_number(str(learner.get("trainable_parameters") or ""))
    )
    parameter_source = None
    if runtime_parameter_count is not None and runtime_parameter_count > 0:
        parameter_source = "runtime checkpoint load"
    elif parameter_count is not None:
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
    dormant_contract = read_json(dormant_modules_path)
    dormant_modules: list[dict[str, Any]] = []
    if dormant_contract.get("schema") == "poke_bot.dormant_model_modules/v1":
        for candidate in dormant_contract.get("modules") or []:
            if not isinstance(candidate, dict):
                continue
            expert_count = int(candidate.get("expert_count") or 0)
            hidden_dim = int(candidate.get("hidden_dim") or 0)
            bottleneck_dim = int(candidate.get("bottleneck_dim") or 0)
            expected_parameters = expert_count * (
                hidden_dim * bottleneck_dim
                + bottleneck_dim
                + bottleneck_dim * hidden_dim
                + hidden_dim
            )
            candidate_parameters = int(candidate.get("parameter_count") or 0)
            if (
                candidate.get("status") != "staged_non_active"
                or candidate.get("runtime_enabled") is not False
                or candidate.get("optimizer_active") is not False
                or candidate.get("present_in_active_checkpoint") is not False
                or candidate_parameters <= 0
                or candidate_parameters != expected_parameters
            ):
                continue
            dormant_modules.append(dict(candidate))
    staged_non_active_parameters = sum(
        int(row.get("parameter_count") or 0) for row in dormant_modules
    )
    current_checkpoint_parameters = (
        int(parameter_count) if parameter_count is not None else None
    )
    parameter_breakdown = {
        "current_checkpoint_total": current_checkpoint_parameters,
        "optimizer_active_current": current_checkpoint_parameters,
        "current_non_active": 0 if current_checkpoint_parameters is not None else None,
        "staged_non_active": staged_non_active_parameters,
        "staged_architecture_total": (
            current_checkpoint_parameters + staged_non_active_parameters
            if current_checkpoint_parameters is not None
            else None
        ),
        "staged_modules": len(dormant_modules),
        "source": str(dormant_modules_path),
    }
    return {
        "implementation": "TemporalCabtTransformer",
        "architecture": architecture,
        "run": manifest.get("run_name"),
        "profile": profile,
        "profile_id": matched_profile.get("id"),
        "trainable_parameters": parameter_count,
        "parameter_source": parameter_source,
        "parameter_evidence_checkpoint": runtime_parameter_contract.get("checkpoint"),
        "parameter_evidence_source": runtime_parameter_contract.get("source"),
        "parameter_breakdown": parameter_breakdown,
        "dormant_modules": dormant_modules,
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
        "optimizer": {
            "curriculum": {
                "name": "AdamW",
                "learning_rate": as_float(learner.get("learning_rate")) or 3e-4,
                "weight_decay": as_float(learner.get("weight_decay")) or 1e-4,
                "gradient_clip_norm": as_float(learner.get("gradient_clip_norm"))
                or 1.0,
                "scheduler": "constant_per_iteration",
                "precision": "bf16_autocast",
                "optimizer_state_restored": True,
                "epochs": as_number(str(learner.get("epochs") or "")),
                "games_per_batch": as_number(
                    str(learner.get("games_per_batch") or "")
                ),
                "max_decisions_per_batch": active_cap,
                "awr_frozen_baseline": True,
                "awr_beta": as_float(runtime_optimizer.get("awr_beta"))
                or as_float(learner.get("awr_beta"))
                or 0.5,
                "awr_weight_max": as_float(
                    runtime_optimizer.get("awr_weight_max")
                )
                or as_float(learner.get("awr_weight_max"))
                or 20.0,
                "entropy_bonus": as_float(learner.get("entropy_bonus")) or 0.01,
            },
            "expert_rehearsal": {
                "name": "AdamW",
                "learning_rate": as_float(expert.get("learning_rate")),
                "weight_decay": 1e-4,
                "gradient_clip_norm": 1.0,
                "scheduler": "constant",
                "precision": "bf16_autocast",
                "epochs": as_number(str(expert.get("epochs") or "")),
                "requested_batch_size": as_number(
                    str(expert.get("requested_batch_size") or "")
                ),
            },
            "source": (
                "live systemd environment + immutable manifest contract"
                if runtime_optimizer
                else "live trainer implementation + immutable manifest contract"
            ),
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


def latest_committed_active_gate_result(
    run_dir: Path | None,
    *,
    mutable_result_pointer: Path | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Return the newest active-gate result bound to immutable commit history.

    Eval files are written before the immutable iteration commit and therefore
    are never evidence by themselves.  The history row inside the commit is
    authoritative.  A mutable compact pointer may supply the returned payload
    only when its core, commit path, and canonical commit digest exactly match
    that immutable row; a stale or conflicting pointer is ignored.
    """

    if run_dir is None:
        return {}, None
    candidates: list[tuple[int, Path]] = []
    for path in (run_dir / "commits").glob("iter_*.json"):
        match = re.fullmatch(r"iter_(\d+)\.json", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    for iteration, commit_path in sorted(candidates, reverse=True):
        commit = read_json(commit_path)
        if (
            commit.get("last_completed_iteration") != iteration
            or commit.get("next_iteration") != iteration + 1
        ):
            continue
        history = commit.get("history")
        matching_rows = (
            [
                row
                for row in history
                if isinstance(row, dict) and row.get("iteration") == iteration
            ]
            if isinstance(history, list)
            else []
        )
        if len(matching_rows) != 1 or matching_rows[0].get("completed") is not True:
            continue
        result = matching_rows[0].get("active_gate_result")
        if not isinstance(result, dict) or result.get("iteration") != iteration:
            continue

        if mutable_result_pointer is not None:
            pointer_path = Path(mutable_result_pointer).expanduser().resolve()
            pointer = read_json(pointer_path)
            pointer_core = {
                key: value
                for key, value in pointer.items()
                if key
                not in {"committed", "commit", "commit_digest", "created_at_utc"}
            }
            raw_commit_path = str(pointer.get("commit") or "").strip()
            pointer_commit_path = (
                Path(raw_commit_path).expanduser().resolve()
                if raw_commit_path
                else None
            )
            if (
                pointer.get("committed") is True
                and pointer_core == result
                and pointer_commit_path == commit_path.resolve()
                and str(pointer.get("commit_digest") or "")
                == _canonical_json_digest(commit)
            ):
                return dict(pointer), pointer_path
        return dict(result), commit_path
    return {}, None


def latest_committed_research_control_result(
    run_dir: Path | None,
) -> tuple[dict[str, Any], Path | None]:
    """Return only a dedicated control artifact bound to iteration commit history."""
    if run_dir is None:
        return {}, None
    candidates: list[tuple[int, Path]] = []
    for path in (run_dir / "commits").glob("iter_*.json"):
        match = re.fullmatch(r"iter_(\d+)\.json", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    for iteration, commit_path in sorted(candidates, reverse=True):
        commit = read_json(commit_path)
        if (
            commit.get("last_completed_iteration") != iteration
            or commit.get("next_iteration") != iteration + 1
        ):
            continue
        history = commit.get("history")
        matches = (
            [
                row
                for row in history
                if isinstance(row, dict)
                and row.get("iteration") == iteration
                and row.get("completed") is True
            ]
            if isinstance(history, list)
            else []
        )
        if len(matches) != 1:
            continue
        result = matches[0].get("research_control_result")
        result_rows = (
            result.get("matchups") if isinstance(result, dict) else None
        )
        result_audit = (
            result.get("audit") if isinstance(result, dict) else None
        )
        if (
            not isinstance(result, dict)
            or result.get("schema")
            != "poke_bot.research_control_measurement_result/v1"
            or result.get("iteration") != iteration
            or result.get("training_eligible") is not False
            or result.get("replay_eligible") is not False
            or result.get("diagnostic_only") is not True
            or result.get("formal_eval") is not False
            or result.get("included_in_gate_pass") is not False
            or result.get("gate_weight") != 0.0
            or result.get("action_selection") != "greedy"
            or result.get("seed_namespace")
            != "eval/research-controls-fixed-manifest-v1"
            or not _is_sha256_digest(result.get("checkpoint_digest"))
            or not _is_sha256_digest(result.get("schedule_digest"))
            or not isinstance(result_rows, list)
            or not result_rows
            or result.get("games") != 250 * len(result_rows)
            or any(
                not isinstance(row, dict)
                or row.get("games") != 250
                or row.get("seat0") != 125
                or row.get("seat1") != 125
                or not _is_sha256_digest(row.get("content_digest"))
                for row in result_rows
            )
            or not isinstance(result_audit, dict)
            or result_audit.get("passed") is not True
            or result_audit.get("exact_distribution") is not True
            or result_audit.get("exact_weights") is not True
            or result_audit.get("seed_disjoint") is not True
            or result_audit.get("package_disjoint_from_active_gate") is not True
            or result_audit.get("replay_records_written") != 0
        ):
            continue
        expected_path = (
            run_dir / "research_controls" / f"iter_{iteration:05d}.json"
        ).resolve()
        raw_path = str(result.get("result_path") or "").strip()
        if not raw_path or Path(raw_path).expanduser().resolve() != expected_path:
            continue
        artifact = read_json(expected_path)
        if artifact != result:
            continue
        return dict(result), expected_path
    return {}, None


def research_control_registry_state(
    public_mix_live: dict[str, Any],
    *,
    registry_path: Path | None = None,
    measurement_result: dict[str, Any] | None = None,
    measurement_source: Path | None = None,
) -> dict[str, Any]:
    """Expose committed additive controls independently of training and gate state."""
    path = Path(
        registry_path
        or (
            RESEARCH_CONTROL_REGISTRY_LATEST
            if RESEARCH_CONTROL_REGISTRY_LATEST.is_file()
            else RESEARCH_CONTROL_REGISTRY
        )
    )
    registry = read_json(path)
    raw_controls = registry.get("controls")
    raw_retirements = registry.get("retirements")
    controls = (
        [dict(row) for row in raw_controls if isinstance(row, dict)]
        if isinstance(raw_controls, list)
        else []
    )
    ids = [str(row.get("opponent_id") or "") for row in controls]
    digests = [str(row.get("content_digest") or "") for row in controls]
    version = registry.get("version")

    def zero_gate_weight(row: dict[str, Any]) -> bool:
        value = row.get("gate_weight")
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value) == 0.0
        )

    def nonnegative_int(value: object) -> int | None:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None

    def finite_number(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        parsed = as_float(value)
        return parsed if parsed is not None and math.isfinite(parsed) else None

    valid = bool(
        registry.get("schema") == "poke_bot.research_control_registry/v1"
        and str(registry.get("registry_id") or "")
        and isinstance(version, int)
        and not isinstance(version, bool)
        and version >= 1
        and isinstance(raw_controls, list)
        and len(controls) == len(raw_controls)
        and controls
        and all(ids)
        and len(ids) == len(set(ids))
        and len(digests) == len(set(digests))
        and all(_is_sha256_digest(digest) for digest in digests)
        and all(
            bool(str(row.get("source_gate_id") or "").strip())
            and zero_gate_weight(row)
            and row.get("included_in_gate_pass") is False
            and row.get("formal_eval") is False
            and row.get("training_eligible") is False
            for row in controls
        )
        and isinstance(raw_retirements, list)
    )
    if not valid:
        return {
            "available": False,
            "reason": "research-control registry failed validation",
            "source": str(path),
        }

    controls_by_id = {
        str(row["opponent_id"]): row for row in controls
    }
    retirement_gate_ids: set[str] = set()
    retired_by_opponent: dict[str, dict[str, Any]] = {}
    retirement_valid = True
    for raw in raw_retirements:
        if not isinstance(raw, dict):
            retirement_valid = False
            break
        gate_id = str(raw.get("gate_id") or "")
        opponent_ids = raw.get("opponent_ids")
        if (
            not gate_id
            or gate_id in retirement_gate_ids
            or not isinstance(opponent_ids, list)
            or not opponent_ids
            or any(not isinstance(value, str) or not value for value in opponent_ids)
            or len(opponent_ids) != len(set(opponent_ids))
            or not set(opponent_ids).issubset(controls_by_id)
            or not _is_sha256_digest(raw.get("exact_result_digest"))
            or not _is_sha256_digest(raw.get("checkpoint_digest"))
            or not isinstance(raw.get("iteration"), int)
            or isinstance(raw.get("iteration"), bool)
            or int(raw["iteration"]) < 0
        ):
            retirement_valid = False
            break
        retirement_gate_ids.add(gate_id)
        for opponent_id in opponent_ids:
            if opponent_id in retired_by_opponent:
                retirement_valid = False
                break
            control = controls_by_id[opponent_id]
            if (
                str(control.get("source_gate_id") or "") != gate_id
                or str(control.get("retired_exact_result_digest") or "")
                != str(raw["exact_result_digest"])
                or str(control.get("retired_checkpoint_digest") or "")
                != str(raw["checkpoint_digest"])
                or not str(control.get("retired_at_utc") or "").strip()
            ):
                retirement_valid = False
                break
            retired_by_opponent[opponent_id] = raw
        if not retirement_valid:
            break
    if retirement_valid:
        legacy_controls = {
            str(control["opponent_id"]): str(control.get("content_digest") or "")
            for control in controls
            if str(control.get("source_gate_id") or "")
            == "legacy-original-four"
        }
        if legacy_controls != LEGACY_RESEARCH_CONTROL_DIGESTS:
            retirement_valid = False
    if retirement_valid:
        for control in controls:
            opponent_id = str(control["opponent_id"])
            source_gate_id = str(control.get("source_gate_id") or "")
            if source_gate_id == "legacy-original-four":
                continue
            elif opponent_id not in retired_by_opponent:
                retirement_valid = False
                break
    if not retirement_valid:
        return {
            "available": False,
            "reason": "research-control retirement history failed validation",
            "source": str(path),
        }

    exact_measurement = (
        dict(measurement_result)
        if isinstance(measurement_result, dict) and measurement_result
        else None
    )
    if exact_measurement is not None:
        native = {
            **exact_measurement,
            "available": int(exact_measurement.get("games") or 0) > 0,
            "active": False,
            "stage": "measure:research_controls:complete",
        }
        dedicated_telemetry = True
    else:
        native = public_mix_live.get("research_controls")
        dedicated_telemetry = isinstance(native, dict)
    if not dedicated_telemetry:
        # Migration compatibility for a sidecar written before schema v3.
        legacy_rows = [
            dict(row)
            for row in (public_mix_live.get("matchups") or [])
            if isinstance(row, dict)
            and str(row.get("opponent_id") or "") in set(ids)
        ]
        legacy_game_counts = [nonnegative_int(row.get("games")) for row in legacy_rows]
        if any(value is None for value in legacy_game_counts):
            return {
                "available": False,
                "reason": "research telemetry game totals are malformed",
                "source": str(path),
            }
        legacy_games = sum(value or 0 for value in legacy_game_counts)
        legacy_wins = sum(finite_number(row.get("wins")) or 0.0 for row in legacy_rows)
        legacy_draws = sum(nonnegative_int(row.get("draws")) or 0 for row in legacy_rows)
        legacy_losses = sum(
            nonnegative_int(row.get("losses")) or 0 for row in legacy_rows
        )
        native = {
            **public_mix_live,
            "available": legacy_games > 0,
            "games": legacy_games,
            "wins": legacy_wins,
            "draws": legacy_draws,
            "losses": legacy_losses,
            "win_rate": legacy_wins / legacy_games if legacy_games else None,
            "matchups": legacy_rows,
        }
    raw_measured = native.get("matchups")
    if raw_measured is None:
        raw_measured = []
    if not isinstance(raw_measured, list) or any(
        not isinstance(row, dict) for row in raw_measured
    ):
        return {
            "available": False,
            "reason": "research telemetry matchup rows are malformed",
            "source": str(path),
        }
    measured = [dict(row) for row in raw_measured]
    measured_ids = [str(row.get("opponent_id") or "") for row in measured]
    unexpected = sorted(
        {
            opponent_id
            for opponent_id in measured_ids
            if opponent_id not in set(ids)
        }
    )
    if dedicated_telemetry and unexpected:
        return {
            "available": False,
            "reason": "research telemetry contains an unregistered opponent",
            "unexpected_opponents": unexpected,
            "source": str(path),
        }
    if any(not opponent_id for opponent_id in measured_ids) or len(measured_ids) != len(
        set(measured_ids)
    ):
        return {
            "available": False,
            "reason": "research telemetry contains a missing or duplicate opponent",
            "source": str(path),
        }

    for row in measured:
        control = controls_by_id[str(row["opponent_id"])]
        reported_digests = {
            str(value)
            for value in (
                row.get("content_digest"),
                row.get("opponent_content_digest"),
            )
            if value is not None and str(value)
        }
        if reported_digests and reported_digests != {
            str(control["content_digest"])
        }:
            return {
                "available": False,
                "reason": "research telemetry package digest does not match registry",
                "opponent_id": str(row["opponent_id"]),
                "source": str(path),
            }

    measured_game_counts = [nonnegative_int(row.get("games")) for row in measured]
    if any(value is None for value in measured_game_counts):
        return {
            "available": False,
            "reason": "research telemetry game totals are malformed",
            "source": str(path),
        }
    measured_games = sum(value or 0 for value in measured_game_counts)
    measured_wins: list[float] = []
    measured_draws: list[int] = []
    measured_losses: list[int] = []
    measured_seat0: list[int] = []
    measured_seat1: list[int] = []
    for row, games in zip(measured, measured_game_counts):
        wins = finite_number(row.get("wins"))
        draws = nonnegative_int(row.get("draws"))
        losses = nonnegative_int(row.get("losses"))
        seat0 = nonnegative_int(row.get("seat0"))
        seat1 = nonnegative_int(row.get("seat1"))
        win_rate = finite_number(row.get("win_rate"))
        true_wins = None if wins is None or draws is None else wins - 0.5 * draws
        if (
            games is None
            or wins is None
            or draws is None
            or losses is None
            or seat0 is None
            or seat1 is None
            or win_rate is None
            or not 0.0 <= win_rate <= 1.0
            or true_wins is None
            or true_wins < 0.0
            or abs(true_wins - round(true_wins)) > 1e-9
            or abs((true_wins + draws + losses) - games) > 1e-9
            or seat0 + seat1 != games
            or abs(win_rate - (wins / games if games else 0.0)) > 1e-9
        ):
            return {
                "available": False,
                "reason": "research telemetry matchup aggregates do not reconcile",
                "source": str(path),
            }
        measured_wins.append(wins)
        measured_draws.append(draws)
        measured_losses.append(losses)
        measured_seat0.append(seat0)
        measured_seat1.append(seat1)
    native_games = native.get("games")
    native_wins = finite_number(native.get("wins"))
    native_draws = nonnegative_int(native.get("draws"))
    native_losses = nonnegative_int(native.get("losses"))
    native_win_rate = finite_number(native.get("win_rate"))
    if (
        nonnegative_int(native_games) is None
        or native_games != measured_games
        or native_wins is None
        or abs(native_wins - sum(measured_wins)) > 1e-9
        or native_draws is None
        or native_draws != sum(measured_draws)
        or native_losses is None
        or native_losses != sum(measured_losses)
        or (native.get("available") is True) != (measured_games > 0)
        or (
            measured_games > 0
            and (
                native_win_rate is None
                or abs(native_win_rate - native_wins / measured_games) > 1e-9
            )
        )
    ):
        return {
            "available": False,
            "reason": "research telemetry game totals do not reconcile",
            "source": str(path),
        }
    checkpoint_digest = str(native.get("checkpoint_digest") or "")
    raw_checkpoint_digests = native.get("checkpoint_digests")
    known_checkpoint_digests: set[str] = set()
    checkpoint_counts_valid = raw_checkpoint_digests is None or isinstance(
        raw_checkpoint_digests, dict
    )
    unknown_checkpoint_games = 0
    checkpoint_games = 0
    if isinstance(raw_checkpoint_digests, dict):
        for digest, count in raw_checkpoint_digests.items():
            parsed_count = nonnegative_int(count)
            if parsed_count is None:
                checkpoint_counts_valid = False
                break
            if parsed_count <= 0:
                continue
            checkpoint_games += parsed_count
            if str(digest) == "unknown":
                unknown_checkpoint_games += parsed_count
            else:
                known_checkpoint_digests.add(str(digest))
    digest_invalid = bool(
        measured_games > 0
        and (
            not checkpoint_counts_valid
            or (
                isinstance(raw_checkpoint_digests, dict)
                and checkpoint_games != measured_games
            )
            or unknown_checkpoint_games > 0
            or native.get("checkpoint_mixed") is True
            or len(known_checkpoint_digests) > 1
            or not _is_sha256_digest(checkpoint_digest)
            or (
                known_checkpoint_digests
                and known_checkpoint_digests != {checkpoint_digest}
            )
        )
    )
    if digest_invalid:
        return {
            "available": False,
            "reason": "research telemetry checkpoint digest is missing or mixed",
            "source": str(path),
        }

    measured_by_id = {
        str(row.get("opponent_id") or ""): row for row in measured
    }
    rows: list[dict[str, Any]] = []
    total_games = 0
    weighted_wins = 0.0
    for control in controls:
        opponent_id = str(control["opponent_id"])
        measurement = measured_by_id.get(opponent_id, {})
        games = nonnegative_int(measurement.get("games")) or 0
        win_rate = as_float(measurement.get("win_rate"))
        total_games += games
        if win_rate is not None:
            weighted_wins += win_rate * games
        rows.append(
            {
                **control,
                "games": games,
                "win_rate": win_rate,
                "wins": as_float(measurement.get("wins")),
                "draws": nonnegative_int(measurement.get("draws")) or 0,
                "losses": nonnegative_int(measurement.get("losses")) or 0,
                "seat0": nonnegative_int(measurement.get("seat0")) or 0,
                "seat1": nonnegative_int(measurement.get("seat1")) or 0,
            }
        )
    return {
        "available": True,
        "schema": "poke_bot.dashboard_research_controls/v1",
        "registry_id": registry.get("registry_id"),
        "registry_version": int(registry["version"]),
        "source": str(path),
        "result_source": (
            str(Path(measurement_source).resolve())
            if measurement_source is not None
            else None
        ),
        "controls": rows,
        "control_count": len(rows),
        "games": total_games,
        "win_rate": weighted_wins / total_games if total_games else None,
        "checkpoint_digest": checkpoint_digest or None,
        "checkpoint_mixed": False,
        "iteration": native.get("iteration"),
        "active": native.get("active") is True,
        "stage": native.get("stage") or "waiting",
        "definition": (
            "per-iteration additive greedy diagnostic controls; excluded from "
            "training/replay and active-gate pass/fail"
        ),
    }


def _offset_public_mix_iterations(
    public_mix_live: dict[str, Any],
    global_iteration_offset: int,
) -> dict[str, Any]:
    """Apply a lineage handoff offset to public and nested research telemetry.

    ``lineage_iteration`` makes this idempotent so a dashboard retry cannot
    accidentally add the handoff offset twice.
    """

    shifted = dict(public_mix_live)

    def shift(payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        lineage_iteration = result.get("lineage_iteration")
        if not isinstance(lineage_iteration, int) or isinstance(
            lineage_iteration, bool
        ):
            lineage_iteration = result.get("iteration")
        if isinstance(lineage_iteration, int) and not isinstance(
            lineage_iteration, bool
        ):
            result["lineage_iteration"] = lineage_iteration
            result["iteration"] = lineage_iteration + int(global_iteration_offset)
        return result

    shifted = shift(shifted)
    nested = shifted.get("research_controls")
    if isinstance(nested, dict):
        shifted["research_controls"] = shift(nested)
    return shifted


def competition_gate_program_state(
    official_heldout: dict[str, Any],
    public_mix_live: dict[str, Any],
    *,
    contract_path: Path = COMPETITION_GATE_PROGRAM,
    registry_path: Path = PROTECTED_BASELINE_GATE,
    exact_result_override: dict[str, Any] | None = None,
    exact_result_source: Path | None = None,
) -> dict[str, Any]:
    """Reconcile the accepted gate and the next public-agent gate fail-closed.

    The exact heldout result, protected model registry, and owner decision are
    distinct facts.  The dashboard may call the prior milestone ``accepted``
    only when all three point to the same checkpoint and exact game totals.
    Sampled public-mix trajectories are kept as a separately labeled progress
    diagnostic; they can never populate the exact gate result.
    """

    contract = read_json(contract_path)
    registry = read_json(registry_path)
    if contract.get("schema") != "poke_bot.competition_gate_program/v1":
        return {
            "available": False,
            "reason": "competition gate program is missing or has the wrong schema",
            "source": str(contract_path),
        }

    active_gate_id = str(contract.get("active_gate_id") or "")
    active_semantics = (
        contract.get("active_gate_semantics")
        if isinstance(contract.get("active_gate_semantics"), dict)
        else {}
    )
    accepted_contract = contract.get("accepted_gate")
    next_contract = contract.get("next_gate")
    if not isinstance(accepted_contract, dict) or not isinstance(next_contract, dict):
        return {
            "available": False,
            "reason": "competition gate program is incomplete",
            "source": str(contract_path),
        }

    accepted_digest = str(accepted_contract.get("checkpoint_digest") or "")
    exact_expected = accepted_contract.get("exact_holdout")
    if not isinstance(exact_expected, dict):
        exact_expected = {}
    registry_digest = str(registry.get("checkpoint_digest") or "")
    registry_evidence = (
        registry.get("evidence")
        if isinstance(registry.get("evidence"), dict)
        else {}
    )
    registry_audit = (
        registry_evidence.get("audit")
        if isinstance(registry_evidence.get("audit"), dict)
        else {}
    )
    expected_games = int(exact_expected.get("games") or 0)
    registry_games = int(registry_evidence.get("games") or 0)
    expected_wr = as_float(exact_expected.get("win_rate"))
    registry_wr = as_float(registry_evidence.get("win_rate"))
    expected_lower = as_float(exact_expected.get("confidence_lower"))
    registry_lower = as_float(registry_evidence.get("confidence_lower"))
    exact_values_match = bool(
        expected_wr is not None
        and registry_wr is not None
        and abs(expected_wr - registry_wr) <= 1e-12
        and expected_lower is not None
        and registry_lower is not None
        and abs(expected_lower - registry_lower) <= 1e-12
    )
    accepted_reconciled = bool(
        accepted_contract.get("status") == "accepted"
        and accepted_digest
        and accepted_digest == registry_digest
        and str(registry_evidence.get("checkpoint_digest") or "")
        == accepted_digest
        and str(registry_audit.get("checkpoint_digest") or "")
        == accepted_digest
        and registry_audit.get("passed") is True
        and registry_audit.get("exact_distribution") is True
        and registry_audit.get("exact_weights") is True
        and registry_audit.get("greedy_required") is True
        and expected_games > 0
        and registry_games == expected_games
        and int(registry_audit.get("valid_games") or 0) == expected_games
        and exact_values_match
        and registry.get("immutable") is True
        and registry.get("automatic_pruning_allowed") is False
    )
    submissions = [
        row
        for row in (accepted_contract.get("submissions") or [])
        if isinstance(row, dict)
    ]
    accepted = {
        "available": True,
        "accepted": accepted_reconciled,
        "status": "accepted" if accepted_reconciled else "identity mismatch",
        "id": accepted_contract.get("id"),
        "label": accepted_contract.get("label"),
        "checkpoint_digest": accepted_digest,
        "decision_basis": accepted_contract.get("decision_basis"),
        "raw_legacy_gate": accepted_contract.get("raw_legacy_gate") or {},
        "submissions": submissions,
        "submission_bundle_sha256": accepted_contract.get(
            "submission_bundle_sha256"
        ),
        "identity_reconciled": accepted_reconciled,
        "registry_protected": bool(
            registry.get("immutable") is True
            and registry.get("automatic_pruning_allowed") is False
        ),
    }

    roster = [
        dict(row)
        for row in (next_contract.get("roster") or [])
        if isinstance(row, dict)
    ]
    evaluation = (
        next_contract.get("evaluation")
        if isinstance(next_contract.get("evaluation"), dict)
        else {}
    )
    pass_criteria = (
        next_contract.get("pass_criteria")
        if isinstance(next_contract.get("pass_criteria"), dict)
        else {}
    )
    roster_ids = [str(row.get("opponent_id") or "") for row in roster]
    content_digests = [str(row.get("content_digest") or "") for row in roster]
    research_measurements = [
        dict(row)
        for row in (next_contract.get("research_measurements") or [])
        if isinstance(row, dict)
    ]
    research_ids = [
        str(row.get("opponent_id") or "") for row in research_measurements
    ]
    research_valid = bool(
        len(research_measurements) == 4
        and set(research_ids) == set(OFFICIAL_BASELINE_IDS)
        and len(research_ids) == len(set(research_ids))
        and sum(int(row.get("games") or 0) for row in research_measurements) == 1000
        and all(
            int(row.get("games") or 0) == 250
            and int(row.get("seat0_games") or 0) == 125
            and int(row.get("seat1_games") or 0) == 125
            and bool(str(row.get("archetype_id") or "").strip())
            and bool(str(row.get("archetype_label") or "").strip())
            and (as_float(row.get("gate_weight")) or 0.0) == 0.0
            and row.get("diagnostic_only") is True
            and row.get("included_in_gate_pass") is False
            for row in research_measurements
        )
    )
    per_opponent_games = int(evaluation.get("games_per_opponent") or 0)
    seat0 = int(evaluation.get("seat0_games_per_opponent") or 0)
    seat1 = int(evaluation.get("seat1_games_per_opponent") or 0)
    total_games = int(evaluation.get("games_total") or 0)
    original_four_gate_weight = as_float(
        active_semantics.get("original_four_gate_weight")
    )
    semantics_valid = bool(
        active_gate_id
        and active_gate_id == str(next_contract.get("id") or "")
        and int(active_semantics.get("gate_roster_size") or 0) == len(roster)
        and int(active_semantics.get("games_per_opponent") or 0)
        == per_opponent_games
        and int(active_semantics.get("gate_games_total") or 0) == total_games
        and active_semantics.get("original_four_role") == "research_control_only"
        and original_four_gate_weight is not None
        and original_four_gate_weight == 0.0
    )
    roster_valid = bool(
        roster
        and semantics_valid
        and all(roster_ids)
        and all(str(row.get("archetype_id") or "").strip() for row in roster)
        and all(str(row.get("archetype_label") or "").strip() for row in roster)
        and len(roster_ids) == len(set(roster_ids))
        and set(roster_ids).isdisjoint(OFFICIAL_BASELINE_IDS)
        and set(roster_ids).isdisjoint(research_ids)
        and all(content_digests)
        and len(content_digests) == len(set(content_digests))
        and all((as_float(row.get("weight")) or 0.0) > 0.0 for row in roster)
        and per_opponent_games > 0
        and seat0 + seat1 == per_opponent_games
        and total_games == len(roster) * per_opponent_games
        and int(evaluation.get("minimum_games_per_opponent") or 0)
        == per_opponent_games
        and evaluation.get("all_matchups_must_complete") is True
        and evaluation.get("partial_results_gate_eligible") is False
        and evaluation.get("sequential_early_stop") is False
        and evaluation.get("mode") == "greedy"
        and evaluation.get("fixed_seed_manifest_required") is True
        and evaluation.get("formal_eval_disjoint_from_training") is True
        and evaluation.get("checkpoint_digest_required") is True
        and evaluation.get("package_digest_deduplicated") is True
        and research_valid
    )

    sampled_rows = {
        str(row.get("opponent_id") or ""): row
        for row in (public_mix_live.get("matchups") or [])
        if isinstance(row, dict)
    }
    diagnostic_rows: list[dict[str, Any]] = []
    weighted_score = 0.0
    covered_weight = 0.0
    diagnostic_games = 0
    for member in roster:
        opponent_id = str(member.get("opponent_id") or "")
        sampled = sampled_rows.get(opponent_id) or {}
        games = int(sampled.get("games") or 0)
        wr = as_float(sampled.get("win_rate"))
        weight = as_float(member.get("weight")) or 0.0
        if games > 0 and wr is not None and weight > 0:
            weighted_score += weight * wr
            covered_weight += weight
            diagnostic_games += games
        diagnostic_rows.append(
            {
                "opponent_id": opponent_id,
                "tier": member.get("tier"),
                "weight": weight,
                "games": games,
                "wr": wr,
                "seat0": int(sampled.get("seat0") or 0),
                "seat1": int(sampled.get("seat1") or 0),
                "content_digest": member.get("content_digest"),
            }
        )
    matchup_games = sum(
        int(row.get("games") or 0)
        for row in (public_mix_live.get("matchups") or [])
        if isinstance(row, dict)
    )
    public_games = int(public_mix_live.get("games") or 0)
    diagnostic_valid = bool(
        public_mix_live.get("available") is True
        and public_mix_live.get("checkpoint_mixed") is not True
        and public_mix_live.get("checkpoint_digest")
        and matchup_games == public_games
        and covered_weight > 0
    )
    diagnostic = {
        "available": diagnostic_valid,
        "definition": next_contract.get("diagnostic_definition"),
        "iteration": public_mix_live.get("iteration"),
        "checkpoint_digest": public_mix_live.get("checkpoint_digest"),
        "games": diagnostic_games,
        "roster_coverage": (
            sum(1 for row in diagnostic_rows if int(row["games"]) > 0)
            / len(diagnostic_rows)
            if diagnostic_rows
            else 0.0
        ),
        "skill_weighted_wr": (
            weighted_score / covered_weight if diagnostic_valid else None
        ),
        "rows": diagnostic_rows,
        "source": str(next_contract.get("diagnostic_pointer") or ""),
    }

    if exact_result_override is not None:
        exact_result = dict(exact_result_override)
        result_path = exact_result_source
    else:
        configured_result_path = str(next_contract.get("exact_result_pointer") or "")
        result_path = Path(configured_result_path) if configured_result_path else None
        exact_result = read_json(result_path) if result_path is not None else {}
    result_matchups = [
        row
        for row in (exact_result.get("matchups") or [])
        if isinstance(row, dict)
    ]
    result_ids = [str(row.get("opponent_id") or "") for row in result_matchups]
    result_distribution_valid = bool(
        len(result_matchups) == len(roster)
        and len(result_ids) == len(set(result_ids))
        and set(result_ids) == set(roster_ids)
        and all(
            int(row.get("games") or 0) == per_opponent_games
            and int(row.get("seat0") or 0) == seat0
            and int(row.get("seat1") or 0) == seat1
            for row in result_matchups
        )
    )
    result_audit = (
        exact_result.get("audit")
        if isinstance(exact_result.get("audit"), dict)
        else {}
    )
    fixed_seed_manifest = (
        result_audit.get("fixed_seed_manifest")
        if isinstance(result_audit.get("fixed_seed_manifest"), dict)
        else {}
    )
    fixed_seed_evidence = bool(
        result_audit.get("fixed_seeds") is True
        or (
            int(fixed_seed_manifest.get("gate_games") or 0) == total_games
            and bool(str(fixed_seed_manifest.get("mapping") or "").strip())
            and bool(str(result_audit.get("fixed_seed_manifest_digest") or "").strip())
        )
    )
    result_checkpoint_digest = str(exact_result.get("checkpoint_digest") or "")
    exact_attempt_valid = bool(
        exact_result.get("schema") == "poke_bot.public_agent_gate_result/v1"
        and exact_result.get("gate_id") == next_contract.get("id")
        and result_checkpoint_digest
        and int(exact_result.get("games") or 0) == total_games
        and result_distribution_valid
        and result_audit.get("passed") is True
        and str(result_audit.get("checkpoint_digest") or "")
        == result_checkpoint_digest
        and result_audit.get("exact_distribution") is True
        and result_audit.get("both_seats") is True
        and result_audit.get("greedy") is True
        and fixed_seed_evidence
    )
    exact_result_valid = exact_attempt_valid
    result_checks = (
        exact_result.get("checks")
        if isinstance(exact_result.get("checks"), dict)
        else {}
    )
    required_result_checks = (
        "audit",
        "skill_weighted_win_rate",
        "skill_weighted_confidence_lower",
        "s_tier_mean_floor",
        "individual_opponent_floor",
    )
    exact_passed = bool(
        exact_result_valid
        and exact_result.get("passed") is True
        and all(result_checks.get(name) is True for name in required_result_checks)
    )
    next_gate = {
        "available": roster_valid,
        "status": (
            "passed"
            if exact_passed
            else "failed"
            if exact_result_valid
            else str(next_contract.get("status") or "queued")
        ),
        "id": next_contract.get("id"),
        "label": next_contract.get("label"),
        "purpose": next_contract.get("purpose"),
        "candidate_source": next_contract.get("candidate_source"),
        "evaluation": evaluation,
        "pass_criteria": pass_criteria,
        "milestones": next_contract.get("milestones") or [],
        "roster": roster,
        "excluded_aliases": next_contract.get("excluded_aliases") or [],
        "research_measurements": research_measurements,
        "research_measurements_valid": research_valid,
        "diagnostic": diagnostic,
        "exact_result_available": exact_result_valid,
        "exact_result": exact_result if exact_result_valid else {},
        "latest_exact_attempt_available": exact_attempt_valid,
        "latest_exact_attempt_current": exact_result_valid,
        "latest_exact_attempt": exact_result if exact_attempt_valid else {},
        "exact_result_source": str(result_path) if result_path is not None else None,
        "contract_valid": roster_valid,
        "contract_reason": (
            None
            if roster_valid
            else "active gate identity, semantics, roster, or exact allocation is invalid"
        ),
    }
    return {
        "available": True,
        "active_gate_id": active_gate_id,
        "active_gate_semantics": active_semantics,
        "accepted_gate": accepted,
        "next_gate": next_gate,
        "source": str(contract_path),
        "updated_at_utc": contract.get("updated_at_utc"),
    }


def strong_public_practice_plan_state(
    run_dir: Path | None,
    iteration: int | None,
    active_gate: dict[str, Any] | None,
    *,
    global_iteration_offset: int = 0,
) -> dict[str, Any]:
    """Read one immutable training-only plan and reconcile it to the active gate.

    This deliberately does not scan backward.  A prior iteration's plan is
    useful history, but showing it as the current allocation would make the
    dashboard lie during startup or after a launch regression.
    """

    if run_dir is None or not isinstance(iteration, int) or iteration < 0:
        return {
            "available": False,
            "reason": "current run or iteration is unavailable",
        }
    plan_path = run_dir / "collection_plans" / f"iter_{iteration:05d}.json"
    if not plan_path.is_file():
        return {
            "available": False,
            "iteration": iteration + global_iteration_offset,
            "reason": "current iteration practice plan is not written yet",
            "source": str(plan_path),
        }

    plan = read_json(plan_path)
    gate = active_gate if isinstance(active_gate, dict) else {}
    roster = gate.get("roster") if isinstance(gate.get("roster"), list) else []
    roster_rows = [dict(row) for row in roster if isinstance(row, dict)]
    roster_ids = [str(row.get("opponent_id") or "") for row in roster_rows]
    roster_by_id = {str(row.get("opponent_id") or ""): row for row in roster_rows}
    raw_per_opponent = plan.get("per_opponent")
    per_opponent = raw_per_opponent if isinstance(raw_per_opponent, dict) else {}
    plan_ids = [str(value) for value in per_opponent]
    raw_weights = plan.get("adaptive_weights")
    adaptive_weights = raw_weights if isinstance(raw_weights, dict) else {}
    weight_ids = [str(value) for value in adaptive_weights]

    def finite_positive(value: object) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
        )

    rows: list[dict[str, Any]] = []
    row_validation_ok = True
    for opponent_id in roster_ids:
        raw_row = per_opponent.get(opponent_id)
        row = raw_row if isinstance(raw_row, dict) else {}
        roster_row = roster_by_id.get(opponent_id) or {}
        games = row.get("games")
        seat0 = row.get("seat0")
        seat1 = row.get("seat1")
        expected_archetype = str(roster_row.get("archetype_id") or "")
        actual_archetype = str(row.get("archetype_id") or "")
        integers_valid = all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (games, seat0, seat1)
        )
        row_valid = bool(
            integers_valid
            and games > 0
            and seat0 + seat1 == games
            and abs(seat0 - seat1) <= 1
            and expected_archetype
            and actual_archetype == expected_archetype
            and finite_positive(adaptive_weights.get(opponent_id))
        )
        row_validation_ok = row_validation_ok and row_valid
        rows.append(
            {
                "opponent_id": opponent_id,
                "tier": roster_row.get("tier"),
                "archetype_id": actual_archetype,
                "archetype_label": roster_row.get("archetype_label"),
                "games": games if integers_valid else 0,
                "seat0": seat0 if integers_valid else 0,
                "seat1": seat1 if integers_valid else 0,
                "adaptive_weight": (
                    float(adaptive_weights[opponent_id])
                    if finite_positive(adaptive_weights.get(opponent_id))
                    else None
                ),
            }
        )

    games = plan.get("games")
    temperature = plan.get("temperature")
    seed_namespace = str(plan.get("seed_namespace") or "")
    formal_seed_namespace = str(plan.get("formal_seed_namespace") or "")
    totals_reconcile = bool(
        isinstance(games, int)
        and not isinstance(games, bool)
        and games > 0
        and sum(int(row.get("games") or 0) for row in rows) == games
    )
    contract_aligned = bool(
        gate.get("available") is True
        and gate.get("contract_valid") is True
        and roster_ids
        and all(roster_ids)
        and len(roster_ids) == len(set(roster_ids))
        and set(plan_ids) == set(roster_ids)
        and len(plan_ids) == len(set(plan_ids))
        and set(weight_ids) == set(roster_ids)
        and len(weight_ids) == len(set(weight_ids))
        and str(plan.get("active_gate_id") or "") == str(gate.get("id") or "")
    )
    semantics_valid = bool(
        plan.get("schema") == "poke_bot.strong_public_practice_plan/v1"
        and plan.get("iteration") == iteration
        and plan.get("training_eligible") is True
        and plan.get("formal_eval") is False
        and plan.get("sampled_policy") is True
        and plan.get("seed_disjoint") is True
        and finite_positive(temperature)
        and seed_namespace.startswith("train/")
        and formal_seed_namespace.startswith("eval/")
        and seed_namespace != formal_seed_namespace
    )
    valid = bool(
        contract_aligned
        and semantics_valid
        and row_validation_ok
        and totals_reconcile
    )
    if not valid:
        failed_checks = [
            name
            for name, passed in (
                ("active gate identity/roster", contract_aligned),
                ("training-only sampled semantics", semantics_valid),
                ("per-opponent archetype/seat/weight", row_validation_ok),
                ("game totals", totals_reconcile),
            )
            if not passed
        ]
        return {
            "available": False,
            "iteration": iteration + global_iteration_offset,
            "reason": "practice plan failed: " + ", ".join(failed_checks),
            "source": str(plan_path),
        }

    return {
        "available": True,
        "iteration": iteration + global_iteration_offset,
        "lineage_iteration": iteration,
        "active_gate_id": plan.get("active_gate_id"),
        "games": games,
        "roster_size": len(rows),
        "temperature": float(temperature),
        "sampled_policy": True,
        "training_eligible": True,
        "formal_eval": False,
        "seed_disjoint": True,
        "seed_namespace": seed_namespace,
        "formal_seed_namespace": formal_seed_namespace,
        "per_opponent": rows,
        "source": str(plan_path),
    }


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


def annotate_expert_optimizer_sps(
    progress: dict[str, Any],
    raw_training_log: str,
) -> dict[str, Any]:
    """Convert the live expert batch rate to exact optimizer sample SPS.

    The expert tqdm reports batches/second, while the device-corpus pack line
    records the exact train and validation sample counts.  Combining those
    two run-bound values avoids both a blank SPS card and reuse of rollout SPS
    from the preceding collection phase.
    """
    stage = str(progress.get("stage") or "")
    if stage not in {"train:expert", "train:expert:validation"}:
        return progress
    if isinstance(progress.get("sps"), (int, float)):
        return progress
    rate = as_float(progress.get("rate"))
    total_batches = as_number(str(progress.get("total") or ""))
    rate_unit = str(progress.get("rate_unit") or "")
    if rate is None or rate <= 0.0 or not total_batches or total_batches <= 0:
        return progress
    if rate_unit == "batch/s":
        batches_per_second = rate
    elif rate_unit == "s/batch":
        batches_per_second = 1.0 / rate
    else:
        return progress

    split_rows = list(
        re.finditer(
            r"\[device-corpus\]\s+CPU pack=.*?\bsamples=(\d+)\s+"
            r"train=(\d+)\s+val=(\d+)",
            ANSI_RE.sub("", raw_training_log).replace("\r", "\n"),
        )
    )
    if not split_rows:
        return progress
    _all_samples, train_samples, val_samples = (
        int(value) for value in split_rows[-1].groups()
    )
    split_samples = (
        val_samples if stage == "train:expert:validation" else train_samples
    )
    if split_samples <= 0:
        return progress

    enriched = dict(progress)
    enriched["sps"] = batches_per_second * split_samples / total_batches
    enriched["sps_source"] = "exact device-corpus split × live tqdm batch rate"
    enriched["optimizer_samples"] = split_samples
    return enriched


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


def _expert_rehearsal_exclusion_seconds(
    run_dir: Path | None,
    iteration: int,
    extra: dict[str, Any] | None = None,
) -> float | None:
    """Return expert-only wall time, or ``None`` when it cannot be proven.

    Rehearsal is an out-of-band correction pass rather than part of curriculum
    iteration throughput.  New receipts can carry an exact duration.  Older
    runs are reconstructed conservatively from the immutable completed-
    collection timestamp through the immutable rehearsal-receipt timestamp.
    A missing rehearsal returns ``0``; a known rehearsal without trustworthy
    timing returns ``None`` so it cannot contaminate iteration averages.
    """
    extra = extra if isinstance(extra, dict) else {}
    record = (
        extra.get("expert_rehearsal")
        if isinstance(extra.get("expert_rehearsal"), dict)
        else {}
    )
    receipt_path = (
        run_dir / "rehearsals" / f"before_iter_{int(iteration):05d}.json"
        if run_dir is not None
        else None
    )
    receipt_exists = bool(receipt_path is not None and receipt_path.is_file())
    if not record and not receipt_exists:
        return 0.0

    for candidate in (
        record.get("wall_elapsed_sec"),
        record.get("elapsed_sec"),
        (record.get("rehearsal") or {}).get("elapsed_sec")
        if isinstance(record.get("rehearsal"), dict)
        else None,
    ):
        seconds = as_float(candidate)
        if seconds is not None and seconds >= 0:
            return seconds

    if run_dir is None or not receipt_exists:
        return None
    collection = read_json(
        run_dir / "collection_receipts" / f"iter_{int(iteration):05d}.json"
    )
    rehearsal = read_json(receipt_path)
    started_at = as_float(collection.get("completed_at"))
    completed_at = as_float(rehearsal.get("completed_at"))
    if completed_at is None:
        try:
            completed_at = receipt_path.stat().st_mtime
        except OSError:
            completed_at = None
    if (
        started_at is None
        or completed_at is None
        or completed_at < started_at
    ):
        return None
    return completed_at - started_at


def _metric_iteration_wall_seconds(
    payload: dict[str, Any],
    *,
    run_dir: Path | None = None,
) -> float | None:
    """Return curriculum work time with expert rehearsal removed."""
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    explicit = as_float(extra.get("iteration_wall_sec"))
    if explicit is not None and explicit >= 0:
        wall_seconds = explicit
    else:
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
            wall_seconds = max(0.0, post_collect) + max(0.0, collect)
        elif post_collect is not None:
            wall_seconds = max(0.0, post_collect)
        else:
            return None
    iteration = payload.get("iteration")
    if not isinstance(iteration, int):
        return wall_seconds
    excluded = _expert_rehearsal_exclusion_seconds(run_dir, iteration, extra)
    if excluded is None:
        return None
    return max(0.0, wall_seconds - excluded)


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
    throughput_history: list[dict[str, Any]] = []
    metrics_dir = run_dir / "metrics"
    for path in sorted(metrics_dir.glob("iter_*.json")):
        payload = read_json(path)
        iteration = payload.get("iteration")
        if not isinstance(iteration, int):
            continue
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        collect_stats = (
            extra.get("collect_stats")
            if isinstance(extra.get("collect_stats"), dict)
            else {}
        )
        games = as_float(payload.get("games"))
        decisions = as_float(payload.get("decisions"))
        collect_seconds = as_float(collect_stats.get("collect_elapsed_sec"))
        gps = as_float(payload.get("games_per_sec"))
        sps = as_float(payload.get("decisions_per_sec"))
        if collect_seconds is not None and collect_seconds > 0:
            if games is not None and games >= 0:
                gps = games / collect_seconds
            if decisions is not None and decisions >= 0:
                sps = decisions / collect_seconds
        if gps is not None or sps is not None:
            throughput_history.append(
                {
                    "iteration": iteration + int(global_iteration_offset),
                    "lineage_iteration": iteration,
                    "gps": gps,
                    "sps": sps,
                    "games": games,
                    "decisions": decisions,
                    "collect_seconds": collect_seconds,
                }
            )
        seconds = _metric_iteration_wall_seconds(payload, run_dir=run_dir)
        if seconds is None:
            continue
        excluded = _expert_rehearsal_exclusion_seconds(
            run_dir,
            iteration,
            payload.get("extra") if isinstance(payload.get("extra"), dict) else {},
        )
        history.append(
            {
                "iteration": iteration + int(global_iteration_offset),
                "lineage_iteration": iteration,
                "seconds": seconds,
                "expert_rehearsal_excluded_seconds": excluded or 0.0,
            }
        )
    history.sort(key=lambda row: int(row["lineage_iteration"]))
    history = history[-20:]
    throughput_history.sort(key=lambda row: int(row["lineage_iteration"]))
    throughput_history = throughput_history[-20:]
    latest = history[-1] if history else None
    rolling = history[-5:]
    latest_throughput = throughput_history[-1] if throughput_history else None
    rolling_throughput = throughput_history[-5:]

    def weighted_rate(rows: list[dict[str, Any]], numerator: str) -> float | None:
        exact = [
            row
            for row in rows
            if as_float(row.get(numerator)) is not None
            and as_float(row.get("collect_seconds")) is not None
            and float(row["collect_seconds"]) > 0
        ]
        if exact:
            return sum(float(row[numerator]) for row in exact) / sum(
                float(row["collect_seconds"]) for row in exact
            )
        rate_name = "gps" if numerator == "games" else "sps"
        rates = [
            float(row[rate_name])
            for row in rows
            if as_float(row.get(rate_name)) is not None
        ]
        return sum(rates) / len(rates) if rates else None

    runtime = read_json(run_dir / "iteration_runtime.json")
    current_iteration = runtime.get("iteration")
    started_at = as_float(runtime.get("started_at"))
    phase = str(runtime.get("phase") or "")
    current_seconds: float | None = None
    display_current_iteration: int | None = None
    current_source: str | None = None
    current_paused_for_expert_rehearsal = False
    if (
        active
        and isinstance(current_iteration, int)
        and (next_iteration is None or current_iteration == next_iteration)
        and started_at is not None
        and phase != "completed"
        and 0 <= started_at <= time.time() + 5
    ):
        raw_current_seconds = max(0.0, time.time() - started_at)
        if str(progress_stage or "").startswith("train:expert"):
            collection = read_json(
                run_dir
                / "collection_receipts"
                / f"iter_{int(current_iteration):05d}.json"
            )
            stats = (
                collection.get("stats")
                if isinstance(collection.get("stats"), dict)
                else {}
            )
            collected_seconds = as_float(stats.get("collect_elapsed_sec"))
            current_seconds = (
                max(0.0, collected_seconds)
                if collected_seconds is not None
                else None
            )
            current_paused_for_expert_rehearsal = True
            current_source = "collection receipt; expert rehearsal excluded"
        else:
            excluded = _expert_rehearsal_exclusion_seconds(
                run_dir, current_iteration
            )
            current_seconds = (
                max(0.0, raw_current_seconds - excluded)
                if excluded is not None
                else None
            )
            current_source = "trainer runtime; expert rehearsal excluded"
        display_current_iteration = current_iteration + int(global_iteration_offset)
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
        "current_paused_for_expert_rehearsal": current_paused_for_expert_rehearsal,
        "latest_iteration": latest.get("iteration") if latest else None,
        "latest_seconds": latest.get("seconds") if latest else None,
        "rolling5_seconds": (
            sum(float(row["seconds"]) for row in rolling) / len(rolling)
            if rolling
            else None
        ),
        "rolling5_samples": len(rolling),
        "history": history,
        "latest_throughput_iteration": (
            latest_throughput.get("iteration") if latest_throughput else None
        ),
        "latest_gps": latest_throughput.get("gps") if latest_throughput else None,
        "latest_sps": latest_throughput.get("sps") if latest_throughput else None,
        "rolling5_gps": weighted_rate(rolling_throughput, "games"),
        "rolling5_sps": weighted_rate(rolling_throughput, "decisions"),
        "rolling5_throughput_samples": len(rolling_throughput),
        "throughput_history": throughput_history,
        "source": (
            "committed metrics + persisted live iteration timer; "
            "expert rehearsal excluded"
        ),
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


def strong_public_gate_runtime_state(
    active_gate: dict[str, Any] | None = None,
    *,
    curriculum_progress: dict[str, Any] | None = None,
    curriculum_active: bool = False,
) -> dict[str, Any]:
    """Return live progress normalized to the pinned active-gate contract.

    The service runs the active strong-public gate first and may then run the
    original-four research controls.  Once that second phase starts, its
    1,000-game counter must not replace the completed 2,000-game active gate
    in dashboard telemetry.
    """

    recognized_stages = {
        "heldout:strong_public_gate",
        "measure:research_controls",
    }
    main_progress = (
        curriculum_progress if isinstance(curriculum_progress, dict) else {}
    )
    main_stage = str(main_progress.get("stage") or "")
    if curriculum_active:
        progress = dict(main_progress)
        stage = main_stage
        recognized = stage in recognized_stages
        active = recognized
        updated_at = None
        source = "main curriculum run-bound progress"
    else:
        standalone_active = run(
            ["systemctl", "--user", "is-active", STRONG_PUBLIC_GATE_SERVICE],
            timeout=2,
        ) == "active"
        if standalone_active:
            status = read_tail(STRONG_PUBLIC_GATE_PROGRESS, 20_000).strip()
            log = read_tail(STRONG_PUBLIC_GATE_LOG, 300_000)
            progress = parse_curriculum_progress(status, log)
            stage = str(progress.get("stage") or "")
            recognized = stage in recognized_stages
        else:
            progress = {}
            stage = ""
            recognized = False
        active = bool(standalone_active and recognized)
        updated_at = max(
            (
                path.stat().st_mtime
                for path in (STRONG_PUBLIC_GATE_PROGRESS, STRONG_PUBLIC_GATE_LOG)
                if path.is_file()
            ),
            default=None,
        )
        source = str(STRONG_PUBLIC_GATE_PROGRESS)
    age_s = max(0.0, time.time() - updated_at) if updated_at else None
    gate = active_gate if isinstance(active_gate, dict) else {}
    evaluation = (
        gate.get("evaluation") if isinstance(gate.get("evaluation"), dict) else {}
    )
    roster = gate.get("roster") if isinstance(gate.get("roster"), list) else []
    gate_total = int(evaluation.get("games_total") or 0)
    games_per_opponent = int(evaluation.get("games_per_opponent") or 0)
    roster_size = len(roster)
    contract_aligned = bool(
        gate.get("available") is True
        and gate.get("contract_valid") is True
        and roster_size > 0
        and games_per_opponent > 0
        and gate_total == roster_size * games_per_opponent
    )
    raw_current = int(progress.get("current") or 0) if recognized else 0
    raw_total = int(progress.get("total") or 0) if recognized else 0
    active_phase_aligned = bool(
        stage != "heldout:strong_public_gate"
        or (
            contract_aligned
            and raw_total == gate_total
            and 0 <= raw_current <= gate_total
        )
    )
    if contract_aligned and stage == "heldout:strong_public_gate" and active_phase_aligned:
        gate_current = raw_current
    elif contract_aligned and stage == "measure:research_controls":
        gate_current = gate_total
    else:
        gate_current = 0
    gate_percent = (
        100.0 * gate_current / gate_total
        if contract_aligned and gate_total > 0
        else None
    )
    return {
        "available": contract_aligned,
        "telemetry_available": bool(active or recognized),
        "active": active,
        "current": gate_current,
        "total": gate_total if contract_aligned else 0,
        "percent": gate_percent,
        "roster_size": roster_size if contract_aligned else 0,
        "games_per_opponent": games_per_opponent if contract_aligned else 0,
        "allocation_label": (
            f"{roster_size} x {games_per_opponent}"
            if contract_aligned
            else None
        ),
        "contract_aligned": contract_aligned,
        "progress_aligned": active_phase_aligned,
        "active_gate_complete": bool(contract_aligned and gate_current == gate_total),
        "phase_current": raw_current,
        "phase_total": raw_total,
        "iteration": progress.get("iteration") if recognized else None,
        "stage": stage if recognized else ("starting" if active else "idle"),
        "phase": (
            "active_gate"
            if stage == "heldout:strong_public_gate"
            else "research_controls"
            if stage == "measure:research_controls"
            else "starting"
            if active
            else "idle"
        ),
        "gps": progress.get("gps") if recognized else None,
        "sps": progress.get("sps") if recognized else None,
        "remotes": progress.get("remotes") if recognized else None,
        "line": progress.get("line") if recognized else None,
        "updated_at": updated_at,
        "age_s": age_s,
        "source": source,
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
    latest_official_heldout = latest_committed_official_heldout_state(
        loop,
        run_dir,
        global_iteration_offset=global_iteration_offset,
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
    progress = annotate_expert_optimizer_sps(progress, raw_training_log)
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
        public_mix_live = _offset_public_mix_iterations(
            public_mix_live,
            global_iteration_offset,
        )
        public_mix_live["age_s"] = public_mix_age
    committed_research_result, committed_research_source = (
        latest_committed_research_control_result(run_dir)
    )
    research_controls = research_control_registry_state(
        public_mix_live,
        measurement_result=committed_research_result,
        measurement_source=committed_research_source,
    )
    gate_contract = read_json(COMPETITION_GATE_PROGRAM)
    configured_next_gate = gate_contract.get("next_gate")
    configured_result_pointer: Path | None = None
    if isinstance(configured_next_gate, dict):
        raw_result_pointer = str(
            configured_next_gate.get("exact_result_pointer") or ""
        ).strip()
        if raw_result_pointer:
            configured_result_pointer = Path(raw_result_pointer)
    committed_gate_result, committed_gate_source = (
        latest_committed_active_gate_result(
            run_dir,
            mutable_result_pointer=configured_result_pointer,
        )
    )
    gate_program = competition_gate_program_state(
        official_heldout,
        public_mix_live,
        # Never let a mutable pointer bypass immutable curriculum history.  The
        # helper above returns that pointer only after an exact commit match.
        exact_result_override=committed_gate_result,
        exact_result_source=committed_gate_source,
    )
    if isinstance(gate_program.get("next_gate"), dict):
        active_gate = gate_program["next_gate"]
        active_gate["runtime"] = strong_public_gate_runtime_state(
            active_gate,
            curriculum_progress=display_progress,
            curriculum_active=bool(active_units),
        )
    else:
        active_gate = {}
    practice_iteration = (
        int(progress["iteration"])
        if isinstance(progress.get("iteration"), int)
        else (
            int(loop["next_iteration"])
            if isinstance(loop.get("next_iteration"), int)
            else None
        )
    )
    strong_public_practice = strong_public_practice_plan_state(
        run_dir,
        practice_iteration,
        active_gate,
        global_iteration_offset=global_iteration_offset,
    )
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
        "research_controls": research_controls,
        "strong_public_practice": strong_public_practice,
        "gate_program": gate_program,
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
        "latest_official_heldout": latest_official_heldout,
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
            runtime_optimizer=worker.get("optimizer_runtime"),
            runtime_parameter_contract=checkpoint_parameter_telemetry(
                ROOT / "outputs/logs" / f"{run_name}.log"
            ) if run_name else {},
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


def authoritative_training_state(
    curriculum: dict[str, Any], transition: dict[str, Any]
) -> dict[str, Any]:
    """Select the live trainer before completed bootstrap history.

    ``ALAKAZAM_BOOTSTRAP_READY`` is intentionally durable, so its existence is
    not evidence that bootstrap is still the active workload. Once a
    curriculum run is known, mirror that run into the legacy ``training``
    payload instead of allowing the completed marker to shadow production.
    """
    if curriculum.get("run"):
        progress = curriculum.get("progress") or {}
        worker = curriculum.get("worker") or {}
        active_pids = curriculum.get("active_pids") or []
        return {
            "authoritative": True,
            "source": curriculum.get("progress_source"),
            "log": curriculum.get("progress_log_source"),
            "latest_line": progress.get("line"),
            "updated_at": curriculum.get("progress_updated_at"),
            "fresh": curriculum.get("source_current"),
            "status": "running" if curriculum.get("active") else "stopped",
            "mode": "curriculum_rl",
            "phase": progress.get("stage") or curriculum.get("stage"),
            "epoch": progress.get("epoch"),
            "current": progress.get("current"),
            "total": progress.get("total"),
            "percent": progress.get("percent"),
            "rate": progress.get("rate"),
            "rate_unit": progress.get("rate_unit"),
            "samples_per_second": progress.get("sps"),
            "game_equivalents_per_second": progress.get("gps"),
            "eta": progress.get("eta"),
            "metrics": progress.get("metrics") or {},
            "run": curriculum.get("run"),
            "iteration": progress.get("iteration", curriculum.get("iteration")),
            "service": {
                "active": bool(curriculum.get("active")),
                "pid": active_pids[0] if active_pids else None,
                "memory_bytes": worker.get("rss_bytes"),
                "source": worker.get("source"),
            },
        }

    bootstrap = transition.get("bootstrap") or {}
    bootstrap_live = bool(
        bootstrap.get("active")
        and (
            int(bootstrap.get("pid") or 0) > 0
            or bootstrap.get("sub_state") == "running"
        )
    )
    if (
        ALAKAZAM_BOOTSTRAP_READY.is_file()
        or bootstrap_live
        or transition.get("status")
        == "training_alakazam_expert_bootstrap_blackwell_device_resident"
    ):
        return alakazam_bootstrap_progress()
    return exact_training_state()


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
    training = authoritative_training_state(curriculum, transition)
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
