#!/usr/bin/env python3
"""Small dependency-free LAN dashboard server intended to live on Bert."""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.html"
REMOTE_SNAPSHOT = "/home/inzi/poke-bot-agent-deployments/state-core-v1/scripts/dashboard_snapshot.py"
LOCAL_SNAPSHOT = ROOT / "fleet_host_snapshot.py"
RATE_STATE = ROOT / "fleet_rate_state.json"


class SnapshotCache:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.value: dict[str, Any] = {
            "ok": False,
            "error": "waiting for first telemetry sample",
            "dashboard_host": socket.gethostname(),
        }
        self.stopping = threading.Event()
        self.rate_history: dict[str, list[tuple[float, float, str]]] = {}
        self.decision_density: dict[str, float] = {}
        self.last_valid_rates: dict[str, dict[str, Any]] = {}
        self._last_rate_state_save = 0.0
        self.network_latency: dict[str, Any] = {}
        self._last_latency_sample = 0.0
        try:
            candidate = json.loads(RATE_STATE.read_text())
            if isinstance(candidate, dict):
                self.last_valid_rates = {
                    str(key): item
                    for key, item in candidate.items()
                    if isinstance(item, dict)
                }
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    def _persist_rate_state(self) -> None:
        """Persist phase-bound last-good rates across dashboard restarts."""
        now = time.monotonic()
        if now - self._last_rate_state_save < 5.0:
            return
        self._last_rate_state_save = now
        temporary = RATE_STATE.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(self.last_valid_rates, separators=(",", ":")))
            temporary.replace(RATE_STATE)
        except OSError:
            pass

    @staticmethod
    def _ping_average_ms(command: list[str]) -> float | None:
        try:
            proc = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=6,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        match = re.search(
            r"(?:round-trip|rtt)[^=]*=\s*[^/]+/([^/]+)/", proc.stdout
        )
        return float(match.group(1)) if match else None

    def _refresh_network_latency(self) -> dict[str, Any]:
        now = time.time()
        if self.network_latency and now - self._last_latency_sample < 600.0:
            return dict(self.network_latency)
        commands = {
            "inzi_to_elmo_ms": [
                "/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                "inzi@192.168.1.151", "ping", "-c", "3", "-W", "3", "192.168.1.143",
            ],
            "inzi_to_bert_ms": [
                "/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                "inzi@192.168.1.151", "ping", "-c", "3", "-W", "3", "192.168.1.158",
            ],
            "bert_to_elmo_ms": [
                "/sbin/ping", "-c", "3", "-W", "3000", "192.168.1.143",
            ],
        }
        samples: dict[str, float | None] = {}
        sample_lock = threading.Lock()

        def sample(key: str, command: list[str]) -> None:
            result = self._ping_average_ms(command)
            with sample_lock:
                samples[key] = result

        threads = [
            threading.Thread(target=sample, args=(key, command), daemon=True)
            for key, command in commands.items()
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=7.0)
        self._last_latency_sample = now
        self.network_latency = {
            **samples,
            "sampled_at": now,
            "refresh_interval_s": 600,
        }
        return dict(self.network_latency)

    def _retain_phase_rate(
        self,
        host_key: str,
        *,
        identity: str,
        sampled_at: float,
        gps: float | None,
        sps: float | None,
        source: str,
        estimated: bool,
    ) -> tuple[float | None, float | None, str, bool, bool, float | None]:
        """Hold the last positive rate through counter idle/gaps and phases.

        A fixed dispatch lets a fast host finish its assigned jobs before the
        slower local tail.  Its monotonic counter then correctly stops, but a
        zero/blank dashboard card looks like broken telemetry.  Preserve the
        last measured active rate, explicitly marked held and aged.  The next
        positive counter sample atomically replaces it.
        """
        useful = bool(
            (isinstance(gps, (int, float)) and gps > 0)
            or (isinstance(sps, (int, float)) and sps > 0)
        )
        if useful:
            cached = {
                "identity": identity,
                "sampled_at": float(sampled_at),
                "gps": gps,
                "sps": sps,
                "source": source,
                "estimated": bool(estimated),
            }
            self.last_valid_rates[host_key] = cached
            self._persist_rate_state()
            return gps, sps, source, estimated, True, 0.0

        cached = self.last_valid_rates.get(host_key) or {}
        held_gps = cached.get("gps")
        held_sps = cached.get("sps")
        if not isinstance(held_gps, (int, float)) and not isinstance(
            held_sps, (int, float)
        ):
            return gps, sps, source, estimated, False, None
        age_s = max(0.0, float(sampled_at) - float(cached.get("sampled_at") or sampled_at))
        same_phase = cached.get("identity") == identity
        held_scope = "same-phase" if same_phase else "prior-phase"
        return (
            float(held_gps) if isinstance(held_gps, (int, float)) else gps,
            float(held_sps) if isinstance(held_sps, (int, float)) else sps,
            f"last active {held_scope} rate (telemetry idle; {cached.get('source') or 'worker counters'})",
            bool(cached.get("estimated")),
            False,
            age_s,
        )

    def _counter_rate(
        self,
        key: str,
        *,
        sampled_at: float,
        counter: float | int | None,
        identity: str,
        window_s: float = 15.0,
    ) -> float | None:
        if not isinstance(counter, (int, float)):
            return None
        now = float(sampled_at)
        value = float(counter)
        history = self.rate_history.setdefault(key, [])
        if history and (history[-1][2] != identity or value < history[-1][1]):
            history.clear()
        history.append((now, value, identity))
        cutoff = now - max(2.0, float(window_s))
        while len(history) > 2 and history[1][0] < cutoff:
            history.pop(0)
        first = history[0]
        last = history[-1]
        elapsed = last[0] - first[0]
        if len(history) < 2 or elapsed < 0.75:
            return None
        return max(0.0, (last[1] - first[1]) / elapsed)

    def _annotate_fleet_rates(self, value: dict[str, Any]) -> None:
        """Attach measured per-host GPS and exact/estimated SPS.

        Remote controllers expose monotonic completed-job counters.  The
        trainer's scheduler independently counts completed local games, so
        Inzi GPS comes from its emitted ``local_gps`` rather than subtracting
        asynchronous remote samples from a fleet-wide rate.  Until every path
        emits side-specific decision counters, estimated SPS is explicitly
        marked from the fleet's observed decisions/game.
        """
        curriculum = value.get("curriculum") or {}
        progress = curriculum.get("progress") or {}
        remote_dispatch = curriculum.get("remote_dispatch") or {}
        fleet = value.get("fleet") or {}
        now = float(value.get("observed_at") or time.time())
        run_name = str(curriculum.get("run") or "unknown")
        stage = str(progress.get("stage") or curriculum.get("stage") or "")
        iteration = progress.get("iteration", curriculum.get("iteration"))
        phase_identity = f"{run_name}:{iteration}:{stage}"
        curriculum_active = bool(curriculum.get("active"))
        collecting = bool(
            curriculum_active
            and (stage.startswith("collect:") or stage in {"heldout", "promotion"})
        )
        training = bool(curriculum_active and stage.startswith("train"))
        remote_phase_active = bool(
            collecting
            and (
                stage == "collect:self_play"
                or
                int(progress.get("remotes") or 0) > 0
                or int(curriculum.get("remote_workers") or 0) > 0
                or int(curriculum.get("remote_request_sockets") or 0) > 0
            )
        )

        total_gps = None
        if collecting:
            # The run-bound tqdm rate is measured by the collector itself.
            # Dashboard samples arrive asynchronously and can see the same
            # counter twice, which previously turned a healthy live rate into
            # a false zero after remote subtraction.
            if isinstance(progress.get("gps"), (int, float)):
                total_gps = max(0.0, float(progress["gps"]))
            else:
                total_gps = self._counter_rate(
                    "fleet:games",
                    sampled_at=now,
                    counter=progress.get("current"),
                    identity=phase_identity,
                )
        total_sps = (
            max(0.0, float(progress["sps"]))
            if isinstance(progress.get("sps"), (int, float))
            else None
        )
        if (
            collecting
            and total_gps is not None
            and total_gps > 0
            and total_sps is not None
            and total_sps > 0
        ):
            density = total_sps / total_gps
            # A malformed/tail zero should not erase the last useful estimate.
            if 1.0 <= density <= 10000.0:
                self.decision_density[run_name] = density
        density = self.decision_density.get(run_name)

        remote_gps: dict[str, float | None] = {}
        remote_sps: dict[str, float | None] = {}
        # The Elmo worker can legitimately have every advertised socket in use
        # during a full dispatch. In that state a dashboard-only health
        # connection may not get a control slot, while the trainer still emits
        # measured per-wave local/remote rates. Use that run-bound telemetry as
        # a fallback rather than making the fleet card go blank.
        scheduler_local_gps = None
        scheduler_local_sps = None
        scheduler_remote_gps = None
        scheduler_remote_sps = None
        scheduler_wave_gps = None
        scheduler_wave_sps = None
        buffered_results = 0
        if collecting and bool(curriculum.get("source_current")):
            for event in reversed(value.get("recent_events") or []):
                event_text = str(event)
                remote_match = re.search(
                    r"\bremote_gps=([0-9]+(?:\.[0-9]+)?)", event_text
                )
                local_match = re.search(
                    r"\blocal_gps=([0-9]+(?:\.[0-9]+)?)", event_text
                )
                local_sps_match = re.search(
                    r"\blocal_sps=([0-9]+(?:\.[0-9]+)?)", event_text
                )
                remote_sps_match = re.search(
                    r"\bremote_sps=([0-9]+(?:\.[0-9]+)?)", event_text
                )
                wave_match = re.search(
                    r"\bwave_gps=([0-9]+(?:\.[0-9]+)?)", event_text
                )
                wave_sps_match = re.search(
                    r"\bwave_sps=([0-9]+(?:\.[0-9]+)?)", event_text
                )
                buffer_match = re.search(
                    r"'memory_items':\s*(\d+).*?'spool_files':\s*(\d+)",
                    event_text,
                )
                if buffer_match and buffered_results <= 0:
                    buffered_results = int(buffer_match.group(1)) + int(
                        buffer_match.group(2)
                    )
                if remote_match or local_match or wave_match:
                    if remote_match:
                        scheduler_remote_gps = float(remote_match.group(1))
                    if local_match:
                        scheduler_local_gps = float(local_match.group(1))
                    if local_sps_match:
                        scheduler_local_sps = float(local_sps_match.group(1))
                    if remote_sps_match:
                        scheduler_remote_sps = float(remote_sps_match.group(1))
                    if wave_match:
                        scheduler_wave_gps = float(wave_match.group(1))
                    if wave_sps_match:
                        scheduler_wave_sps = float(wave_sps_match.group(1))
                    break
        for host_key in ("elmo", "bert"):
            host = fleet.get(host_key) or {}
            worker = host.get("worker") or {}
            if (
                host_key == "bert"
                and host.get("production_active") is False
                and worker.get("testing")
            ):
                # Bert's Apple benchmark is deliberately outside the trainer
                # fleet. Preserve its own GPS/SPS/stage instead of replacing it
                # with production counters, and never subtract it from Inzi.
                test_identity = (
                    f"bert:test:{worker.get('command') or worker.get('optimization_variant') or 'idle'}"
                )
                live_gps = self._counter_rate(
                    "bert:test:games",
                    sampled_at=now,
                    counter=worker.get("jobs_completed"),
                    identity=test_identity,
                )
                live_sps = self._counter_rate(
                    "bert:test:decisions",
                    sampled_at=now,
                    counter=worker.get("decisions_completed"),
                    identity=test_identity,
                )
                if live_gps is not None:
                    worker["gps"] = live_gps
                if live_sps is not None:
                    worker["sps"] = live_sps
                worker["rate_stage"] = worker.get("optimization_stage")
                worker["rate_source"] = (
                    "Apple optimization live worker counters"
                    if live_gps is not None or live_sps is not None
                    else worker.get("rate_source")
                    or "Apple optimization latest completed topology"
                )
                worker["sps_estimated"] = False
                remote_gps[host_key] = None
                remote_sps[host_key] = None
                continue
            # Docker health checks can briefly match the worker command and
            # add a transient PID to controller_pids.  Basing identity on that
            # list resets the rate window every health probe.  The monotonic
            # counter rollback below already detects a real worker restart.
            identity = f"{host_key}:{worker.get('command') or 'worker'}"
            gps = (
                self._counter_rate(
                    f"{host_key}:games",
                    sampled_at=now,
                    counter=worker.get("jobs_completed"),
                    identity=f"{identity}:{phase_identity}",
                    window_s=3600.0,
                )
                if collecting
                else None
            )
            gps_from_scheduler = False
            # Aggregate remote scheduler telemetry is never assigned wholly to
            # Elmo: that made Elmo look fast and Bert look idle whenever one
            # health sample was delayed. Per-host cards use host counters only.
            exact_sps = (
                self._counter_rate(
                    f"{host_key}:decisions",
                    sampled_at=now,
                    counter=worker.get("decisions_completed"),
                    identity=f"{identity}:{phase_identity}",
                    window_s=3600.0,
                )
                if collecting
                else None
            )
            sps = exact_sps
            source = "trainer scheduler telemetry" if gps_from_scheduler else "worker-counters"
            estimated = False
            if sps is None and gps is not None and density is not None:
                sps = gps * density
                source = (
                    "trainer scheduler GPS + fleet decisions/game"
                    if gps_from_scheduler
                    else "worker-gps + fleet decisions/game"
                )
                estimated = True
            # Once a remote has drained its assigned/pulled work, its rolling
            # counter window still contains old completions and therefore
            # decays slowly toward zero.  Do not treat those decaying samples
            # as new measurements: retain the final genuinely-active GPS/SPS
            # until this host receives work in a later collection wave.
            active_jobs = worker.get("active_jobs")
            worker_slots = worker.get("workers")
            execution_slots = (
                max(1, int(worker_slots))
                if isinstance(worker_slots, (int, float))
                else None
            )
            request_target = remote_dispatch.get(
                f"{host_key}_request_sockets"
            )
            if not isinstance(request_target, (int, float)):
                request_target = None
            admitted = (
                max(0, int(active_jobs))
                if isinstance(active_jobs, (int, float))
                else None
            )
            fed_workers = (
                min(execution_slots, admitted)
                if execution_slots is not None and admitted is not None
                else None
            )
            queued_jobs = (
                max(0, admitted - execution_slots)
                if execution_slots is not None and admitted is not None
                else None
            )
            worker.update(
                execution_slots=execution_slots,
                admitted_requests=admitted,
                queued_jobs=queued_jobs,
                request_target=(int(request_target) if request_target is not None else None),
                feed_coverage=(
                    float(fed_workers) / float(execution_slots)
                    if execution_slots and fed_workers is not None
                    else None
                ),
            )
            full_sample_floor = (
                max(1, int(float(worker_slots) * 0.25))
                if isinstance(worker_slots, (int, float))
                else 1
            )
            progress_percent = float(progress.get("percent") or 0.0)
            allocation_draining = bool(
                collecting
                and progress_percent >= 90.0
                and isinstance(active_jobs, (int, float))
                and int(active_jobs) < full_sample_floor
            )
            allocation_idle = bool(
                isinstance(active_jobs, (int, float)) and int(active_jobs) <= 0
            )
            if allocation_draining:
                gps = None
                sps = None
                source = "allocation draining; full-concurrency rate frozen"
                estimated = False
            rate_live = gps is not None or sps is not None
            rate_age_s = 0.0 if rate_live else None
            if collecting:
                gps, sps, source, estimated, rate_live, rate_age_s = (
                    self._retain_phase_rate(
                        host_key,
                        identity=f"{identity}:{phase_identity}",
                        sampled_at=now,
                        gps=gps,
                        sps=sps,
                        source=source,
                        estimated=estimated,
                    )
                )
            if training:
                gps, sps, source, estimated, rate_live, rate_age_s = (
                    self._retain_phase_rate(
                        host_key,
                        identity=f"{identity}:{phase_identity}",
                        sampled_at=now,
                        gps=None,
                        sps=None,
                        source="optimizer phase",
                        estimated=False,
                    )
                )
                if gps is None and sps is None:
                    gps, sps, source, estimated = 0.0, 0.0, "optimizer phase", False
                    rate_live, rate_age_s = True, 0.0
            worker.update(
                gps=gps,
                sps=sps,
                rate_source=source,
                sps_estimated=estimated,
                rate_stage=stage,
                rate_live=rate_live,
                rate_age_s=rate_age_s,
            )
            if not bool(host.get("reachable", True)) or not bool(worker.get("active")):
                worker["allocation_state"] = "OFFLINE · no production allocation"
            elif training:
                worker["allocation_state"] = "ALLOCATION COMPLETE · optimizer phase"
            elif remote_phase_active:
                if (
                    execution_slots is not None
                    and admitted is not None
                    and admitted >= execution_slots
                ):
                    worker["allocation_state"] = (
                        f"WORKING · {execution_slots}/{execution_slots} workers fed"
                        f" · {int(queued_jobs or 0)} queued"
                    )
                elif admitted is not None and admitted > 0:
                    worker["allocation_state"] = (
                        f"REFILLING · {int(fed_workers or 0)}/{execution_slots or '?'} "
                        "workers fed · queue empty"
                    )
                elif buffered_results > 0:
                    worker["allocation_state"] = (
                        f"RESULTS BUFFERED · {buffered_results} fleet games awaiting ingest"
                    )
                elif float(progress.get("percent") or 0.0) < 95.0:
                    worker["allocation_state"] = (
                        f"STARVED · 0/{execution_slots or '?'} workers fed · refill pending"
                    )
                elif rate_live is False and (
                    isinstance(gps, (int, float)) or isinstance(sps, (int, float))
                ):
                    worker["allocation_state"] = (
                        "ALLOCATION COMPLETE · waiting for remaining fleet"
                    )
                else:
                    worker["allocation_state"] = "READY · waiting for scheduler work"
            elif curriculum_active and stage in {"collect:public_mix", "heldout", "promotion"}:
                worker["allocation_state"] = "ALLOCATION COMPLETE · local-only phase"
            else:
                worker["allocation_state"] = "READY · next self-play allocation"
            remote_gps[host_key] = gps
            remote_sps[host_key] = sps

        inzi = fleet.get("inzi") or {}
        inzi_worker = inzi.get("worker") or {}
        measured_remote_gps = [rate for rate in remote_gps.values() if rate is not None]
        measured_remote_sps = [rate for rate in remote_sps.values() if rate is not None]
        if density is None and measured_remote_gps and measured_remote_sps:
            remote_gps_sum = sum(measured_remote_gps)
            remote_sps_sum = sum(measured_remote_sps)
            if remote_gps_sum > 0.0 and remote_sps_sum > 0.0:
                remote_density = remote_sps_sum / remote_gps_sum
                if 1.0 <= remote_density <= 10000.0:
                    density = remote_density
                    self.decision_density[run_name] = remote_density
        local_only_phase = bool(
            stage in {"collect:public_mix", "heldout", "promotion"}
            and not remote_phase_active
        )
        if collecting and local_only_phase and total_gps is not None:
            local_gps = max(0.0, float(total_gps))
            local_gps_source = "local-only collector completion counter"
        elif (
            collecting
            and scheduler_local_gps is not None
            and float(scheduler_local_gps) > 0.0
        ):
            local_gps = max(0.0, float(scheduler_local_gps))
            local_gps_source = "trainer scheduler local completion counter"
        elif collecting and total_gps is not None:
            local_gps = max(0.0, float(total_gps) - sum(measured_remote_gps))
            local_gps_source = "collector total minus remote counters"
        else:
            local_gps = None
            local_gps_source = "no active collection"
        local_sps = None
        local_sps_estimated = False
        if (
            collecting
            and scheduler_local_sps is not None
            and float(scheduler_local_sps) > 0.0
        ):
            local_sps = max(0.0, float(scheduler_local_sps))
        elif local_gps is not None and density is not None:
            local_sps = local_gps * density
            local_sps_estimated = True
        elif collecting and total_sps is not None:
            remote_sps_sum = sum(
                rate for rate in remote_sps.values() if rate is not None
            )
            local_sps = max(0.0, total_sps - remote_sps_sum)
            local_sps_estimated = True
        source = (
            f"{local_gps_source}; SPS from fleet decisions/game"
            if local_sps is not None and collecting and local_sps_estimated
            else f"{local_gps_source}; direct local decision counter"
            if local_sps is not None and collecting
            else local_gps_source
        )
        if training:
            local_gps, local_sps, source = 0.0, 0.0, "optimizer phase"
        inzi_worker.update(
            gps=local_gps,
            sps=local_sps,
            rate_source=source,
            sps_estimated=bool(local_sps_estimated),
            rate_stage=stage,
        )
        local_slots = max(1, int(inzi_worker.get("workers") or 96))
        leaf_servers = max(0, int(inzi_worker.get("leaf_servers") or 0))
        if collecting:
            current_games = progress.get("current")
            total_games = progress.get("total")
            remaining_games = (
                max(0, int(total_games) - int(current_games))
                if isinstance(current_games, (int, float))
                and isinstance(total_games, (int, float))
                else local_slots
            )
            active_games = min(local_slots, remaining_games)
            inzi_worker["active_games"] = active_games
            inzi_worker["allocation_state"] = (
                f"WORKING · {active_games} local game"
                f"{'s' if active_games != 1 else ''} active"
            )
            inzi_worker["allocation"] = (
                f"{local_slots} simulator slots · {leaf_servers} Blackwell policy leaves"
            )
        elif stage == "train:preparing":
            inzi_worker["active_games"] = 0
            inzi_worker["allocation_state"] = "REPLAY PREP · 0 simulation games"
            inzi_worker["allocation"] = (
                "CPU replay-window assembly · Blackwell waiting"
            )
        elif training:
            inzi_worker["active_games"] = 0
            inzi_worker["allocation_state"] = "OPTIMIZER · 0 simulation games"
            inzi_worker["allocation"] = "Blackwell full-model optimizer"
        else:
            inzi_worker["active_games"] = 0
            inzi_worker["allocation_state"] = "READY · next local allocation"
            inzi_worker["allocation"] = (
                f"{local_slots} simulator slots · {leaf_servers} Blackwell policy leaves"
            )
        value["fleet_rates"] = {
            "stage": stage,
            "iteration": iteration,
            "total_gps": (
                scheduler_wave_gps
                if scheduler_wave_gps is not None and scheduler_wave_gps > 0
                else total_gps
            ),
            "total_sps": (
                scheduler_wave_sps
                if scheduler_wave_sps is not None and scheduler_wave_sps > 0
                else total_sps
            ),
            "ingest_gps": total_gps,
            "ingest_sps": total_sps,
            "local_gps": scheduler_local_gps,
            "local_sps": scheduler_local_sps,
            "remote_gps": scheduler_remote_gps,
            "remote_sps": scheduler_remote_sps,
            "buffered_results": buffered_results,
            "decisions_per_game": density,
            "window_s": 15.0,
            "rate_source": (
                "trainer scheduler generation telemetry"
                if scheduler_wave_gps is not None
                else "collector ingest telemetry"
            ),
        }

    def _annotate_scheduler_queues(self, value: dict[str, Any]) -> None:
        """Join controller queue targets to live per-host queue flow.

        A remote's monotonic ``jobs_completed + active_jobs`` is the number of
        jobs admitted by that server. Its derivative is measured dispatch GPS;
        ``jobs_completed`` alone is measured drain GPS. This makes outbound
        starvation distinguishable from a healthy queue that is merely being
        consumed quickly.
        """
        curriculum = value.get("curriculum") or {}
        queues = curriculum.get("scheduler_queues") or {}
        if not isinstance(queues, dict):
            queues = {}
        fleet = value.get("fleet") or {}
        rates = value.get("fleet_rates") or {}
        observed_at = float(value.get("observed_at") or time.time())
        progress = curriculum.get("progress") or {}
        identity = (
            f"{curriculum.get('run')}:{progress.get('iteration')}:"
            f"{progress.get('stage')}"
        )
        inzi_worker = ((fleet.get("inzi") or {}).get("worker") or {})
        local_active = inzi_worker.get("active_games")
        local_high_water = inzi_worker.get("workers")
        queues["local"] = {
            "active_or_claimed": local_active,
            "high_water": local_high_water,
            "dispatch_gps": inzi_worker.get("gps"),
            "drain_gps": inzi_worker.get("gps"),
            "source": "trainer local scheduler/generation counter",
        }

        endpoint_rows = queues.get("endpoints")
        if not isinstance(endpoint_rows, dict):
            endpoint_rows = {}
            queues["endpoints"] = endpoint_rows
        for host_key in ("elmo", "bert"):
            host = fleet.get(host_key) or {}
            worker = host.get("worker") or {}
            row = endpoint_rows.get(host_key)
            if not isinstance(row, dict):
                row = {}
                endpoint_rows[host_key] = row
            active_jobs = worker.get("active_jobs")
            completed = worker.get("jobs_completed")
            admitted_counter = (
                float(active_jobs) + float(completed)
                if isinstance(active_jobs, (int, float))
                and isinstance(completed, (int, float))
                else None
            )
            dispatch_gps = self._counter_rate(
                f"{host_key}:scheduler-admitted",
                sampled_at=observed_at,
                counter=admitted_counter,
                identity=identity,
                window_s=15.0,
            )
            execution = worker.get("execution_slots", worker.get("workers"))
            executing = (
                min(max(0, int(execution)), max(0, int(active_jobs)))
                if isinstance(execution, (int, float))
                and isinstance(active_jobs, (int, float))
                else None
            )
            server_queued = worker.get("queued_jobs")
            high_water = row.get("protected_high_water", row.get("protected_cap"))
            sockets = row.get("socket_capacity")
            controller_reserve = (
                max(0, int(high_water) - int(sockets))
                if isinstance(high_water, (int, float))
                and isinstance(sockets, (int, float))
                else row.get("controller_reserve_target")
            )
            row.update(
                executing=executing,
                server_queued=server_queued,
                server_admitted=active_jobs,
                controller_reserve_target=controller_reserve,
                dispatch_gps=dispatch_gps,
                drain_gps=worker.get("gps"),
                queue_delta_gps=(
                    float(dispatch_gps) - float(worker.get("gps"))
                    if isinstance(dispatch_gps, (int, float))
                    and isinstance(worker.get("gps"), (int, float))
                    else None
                ),
                flow_source="remote admitted/completed monotonic counters",
            )
        queues["results"] = {
            "waiting_ingest": rates.get("buffered_results"),
            "generation_gps": rates.get("total_gps"),
            "ingest_gps": rates.get("ingest_gps"),
            "source": "bounded RAM + disk result buffer",
        }
        queues["available"] = bool(
            queues.get("available") or curriculum.get("active")
        )
        queues["updated_at"] = observed_at
        curriculum["scheduler_queues"] = queues
        value["scheduler_queues"] = queues

    def _annotate_replay_progress(self, value: dict[str, Any]) -> None:
        """Promote replay-window loading into the canonical Curriculum bar."""
        curriculum = value.get("curriculum") or {}
        progress = curriculum.get("progress") or {}
        replay = curriculum.get("replay_window") or {}
        if not replay.get("available"):
            return
        now = float(value.get("observed_at") or time.time())
        run_name = str(curriculum.get("run") or "unknown")
        iteration = replay.get("iteration", curriculum.get("iteration"))
        current = replay.get("current")
        total = replay.get("total")
        rate_bps = (
            self._counter_rate(
                "replay-window:bytes",
                sampled_at=now,
                counter=current,
                identity=f"{run_name}:{iteration}:{replay.get('stage')}",
                window_s=12.0,
            )
            if str(replay.get("unit") or "") == "bytes"
            else None
        )
        if isinstance(rate_bps, (int, float)) and rate_bps > 0:
            replay["rate_bytes_per_sec"] = rate_bps
            if isinstance(current, (int, float)) and isinstance(total, (int, float)):
                replay["eta_s"] = max(0.0, (float(total) - float(current)) / rate_bps)
        if (
            str(progress.get("stage") or "") == "train:preparing"
            and str(replay.get("stage") or "") == "LOADING WINDOW"
            and isinstance(replay.get("percent"), (int, float))
            and isinstance(current, (int, float))
            and isinstance(total, (int, float))
        ):
            pct = max(0.0, min(100.0, float(replay["percent"])))
            filled = max(0, min(28, int(round(28.0 * pct / 100.0))))
            bar = "█" * filled + "░" * (28 - filled)
            current_gib = float(current) / (1024.0**3)
            total_gib = float(total) / (1024.0**3)
            rate_mib = (
                float(rate_bps) / (1024.0**2)
                if isinstance(rate_bps, (int, float)) and rate_bps > 0
                else None
            )
            eta_s = replay.get("eta_s")
            eta_text = (
                f"{int(float(eta_s)) // 60:02d}:{int(float(eta_s)) % 60:02d}"
                if isinstance(eta_s, (int, float))
                else "measuring"
            )
            timing = (
                f"{rate_mib:.1f} MiB/s, ETA {eta_text}"
                if rate_mib is not None
                else "measuring throughput"
            )
            progress.update(
                line=(
                    f"pure_rl replay-window iter={iteration}: {pct:5.1f}%|{bar}| "
                    f"{current_gib:.2f}/{total_gib:.2f} GiB [{timing}]"
                ),
                percent=pct,
                current=round(current_gib, 2),
                total=round(total_gib, 2),
                unit="GiB",
                rate=rate_mib,
                rate_unit="MiB/s",
                eta=eta_text,
                gps=None,
                sps=None,
                remotes=0,
            )

    def update(self) -> None:
        command = [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=4",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=60",
            "-o",
            "ControlPath=/tmp/pokebot-dashboard-ssh",
            "inzi@192.168.1.151",
            REMOTE_SNAPSHOT,
        ]
        try:
            proc = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or f"ssh exited {proc.returncode}")
            value = json.loads(proc.stdout)
            local = subprocess.run(
                [
                    "/usr/bin/python3",
                    str(LOCAL_SNAPSHOT),
                    "--role",
                    "simulator",
                    "--name",
                    "Bert",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            try:
                bert = json.loads(local.stdout)
            except json.JSONDecodeError:
                bert = {
                    "reachable": False,
                    "name": "Bert",
                    "role": "simulator",
                    "error": local.stderr.strip() or "local telemetry unavailable",
                }
            trainer_command = str(
                (((value.get("curriculum") or {}).get("worker") or {}).get("command"))
                or ""
            )
            configured_endpoints = [
                str(endpoint)
                for endpoint in ((value.get("curriculum") or {}).get("remote_endpoints") or [])
            ]
            bert_worker = bert.get("worker") or {}
            bert_ready = bool(bert_worker.get("active") and bert_worker.get("listening"))
            bert_in_production = bert_ready and (
                "bert.local:8766" in trainer_command
                or any("bert.local:8766" in endpoint for endpoint in configured_endpoints)
            )
            if bert_in_production:
                bert["role"] = "production simulator"
                bert["production_active"] = True
                bert["assignment"] = (
                    "PRODUCTION · M4 CPU simulator"
                    if bool((value.get("curriculum") or {}).get("active"))
                    else "PRODUCTION READY · M4 CPU simulator"
                )
                bert_worker["testing"] = False
                bert_worker["production_active"] = True
                bert_worker.pop("optimization_stage", None)
                bert_worker.pop("optimization_variant", None)
                bert_worker.pop("optimization_device", None)
                bert["worker"] = bert_worker
            else:
                bert["role"] = "inactive · Apple optimization testing"
                bert["production_active"] = False
                optimization = bert.get("optimization") or {}
                stage = optimization.get("stage") or "CPU/MPS throughput and parity sweep"
                bert["assignment"] = f"M4 Apple optimization · {stage}"
            value.setdefault("fleet", {})["bert"] = bert
            self._annotate_replay_progress(value)
            self._annotate_fleet_rates(value)
            self._annotate_scheduler_queues(value)
            value["network_latency"] = self._refresh_network_latency()
            value["dashboard_host"] = socket.gethostname()
            value["dashboard_sampled_at"] = time.time()
        except Exception as exc:  # keep the last good payload visible
            with self.lock:
                value = dict(self.value)
            if value.get("ok"):
                value.pop("error", None)
                value["telemetry_warning"] = str(exc)
            else:
                value["ok"] = False
                value["error"] = str(exc)
            value["dashboard_sampled_at"] = time.time()
        with self.lock:
            self.value = value

    def loop(self) -> None:
        while not self.stopping.is_set():
            started = time.monotonic()
            self.update()
            self.stopping.wait(max(0.25, 1.0 - (time.monotonic() - started)))

    def get(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.value)


CACHE = SnapshotCache()


class Handler(BaseHTTPRequestHandler):
    server_version = "PokeBotDashboard/1.0"

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.partition("?")[0]
        if path in ("/", "/index.html"):
            self.send_bytes(INDEX.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            body = json.dumps(CACHE.get(), separators=(",", ":")).encode()
            self.send_bytes(body, "application/json; charset=utf-8")
            return
        if path == "/health":
            payload = CACHE.get()
            status = HTTPStatus.OK if payload.get("ok") else HTTPStatus.SERVICE_UNAVAILABLE
            self.send_bytes(json.dumps(payload).encode(), "application/json", status)
            return
        self.send_bytes(b"not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args()
    worker = threading.Thread(target=CACHE.loop, name="telemetry", daemon=True)
    worker.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    finally:
        CACHE.stopping.set()
        server.server_close()


if __name__ == "__main__":
    main()
