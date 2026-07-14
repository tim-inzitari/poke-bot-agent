"""Resumable, GPU-isolated core-to-Hammer training pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import signal
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional
import sys

from . import archetypes, checkpoint, config, paths


PHASE_GRAPH = (
    "core_bc",
    "core_deep_search",
    "core_gate",
    "hammer_warmstart",
    "hammer_search_rl",
)
DEFAULT_CORE_LOG = paths.OUTPUTS_DIR / "logs" / "core_kernel.log"


class UnsafePhaseError(RuntimeError):
    """A phase cannot run without violating the trusted search contract."""


REQUIRED_SEARCH_DIAGNOSTICS = frozenset(
    {
        "sims_run",
        "sims_planned",
        "unique_expanded_nodes",
        "max_depth",
        "mean_depth",
        "mean_branching",
        "leaf_evaluations",
        "chance_samples",
        "unique_particles",
        "root_visits",
        "queue_wait_ms_mean",
        "inference_batch_size_mean",
        "sims_per_s",
        "elapsed_s",
        "trusted",
    }
)


def validate_search_target_identity(
    provenance: dict[str, Any],
    diagnostics: Iterable[dict[str, Any]],
    *,
    expected_checkpoint_digest: str,
    expected_model_generation: int,
) -> None:
    """Reject incomplete, stale, or mixed-generation search targets."""
    required = {
        "checkpoint_digest",
        "model_generation",
        "search_config",
        "belief_config",
        "simulator_version",
    }
    missing = sorted(required - set(provenance))
    if missing:
        raise ValueError(f"search target provenance missing {missing}")
    if provenance["checkpoint_digest"] != expected_checkpoint_digest:
        raise ValueError("stale search target checkpoint digest")
    if int(provenance["model_generation"]) != int(expected_model_generation):
        raise ValueError("mixed/stale search target model generation")
    if not isinstance(provenance["search_config"], dict):
        raise ValueError("search_config must be an exact mapping")
    search_config = provenance["search_config"]
    if (
        search_config.get("algorithm")
        != "public_history_root_sampled_information_set_mcts"
        or search_config.get("tree_reuse") is not False
        or search_config.get("adaptive_sequential_updates") is not True
        or search_config.get("cross_game_batching_only") is not True
    ):
        raise ValueError("search_config violates trusted information-set invariants")
    belief = provenance["belief_config"]
    if not isinstance(belief, dict) or not belief.get("sampler"):
        raise ValueError("belief_config must identify its sampler")
    if belief.get("mode") in (None, "single_world", "oracle"):
        raise ValueError("single-world/oracle targets are not trusted")
    if (
        belief.get("conserves_card_multiplicity") is not True
        or belief.get("uses_baseline_identity") is not False
        or not str(belief.get("model_digest") or "").startswith("sha256:")
    ):
        raise ValueError("belief_config violates trusted particle invariants")
    if not str(provenance["simulator_version"]).startswith(
        "competition-libcg-sha256:"
    ):
        raise ValueError("simulator_version is not immutable competition provenance")
    rows = list(diagnostics)
    if not rows:
        raise ValueError("search target diagnostics are empty")
    for index, row in enumerate(rows):
        missing_diag = sorted(REQUIRED_SEARCH_DIAGNOSTICS - set(row))
        if missing_diag:
            raise ValueError(
                f"search target diagnostics[{index}] missing {missing_diag}"
            )
        if row.get("trusted") is not True:
            raise ValueError(f"search target diagnostics[{index}] is untrusted")
        completed = int(row.get("sims_run") or 0)
        if completed < int(search_config.get("min_trusted_sims") or 128):
            raise ValueError(
                f"search target diagnostics[{index}] has insufficient simulations"
            )
        if int(row.get("root_visits") or -1) != completed:
            raise ValueError(
                f"search target diagnostics[{index}] has partial root backups"
            )


@dataclass(frozen=True)
class SearchSafetyAudit:
    trusted_ready: bool
    belief_mode: str
    chance_mode: str
    history_mode: str
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def search_safety_audit() -> SearchSafetyAudit:
    """Return the currently enforced deep-search capability verdict."""
    return SearchSafetyAudit(
        trusted_ready=True,
        belief_mode="anonymous_empirical_public_history_card_conserving_particles",
        chance_mode="explicit_uniform_coin_sampling",
        history_mode="branch_local_actor_information_history",
        blockers=(),
    )


def require_trusted_search(phase: str) -> None:
    audit = search_safety_audit()
    if not audit.trusted_ready:
        raise UnsafePhaseError(
            f"{phase} blocked by trusted-search guard: "
            + " | ".join(audit.blockers)
        )


def next_phase(phase: str) -> Optional[str]:
    idx = PHASE_GRAPH.index(phase)
    return PHASE_GRAPH[idx + 1] if idx + 1 < len(PHASE_GRAPH) else None


def auto_size_core_workers(
    *,
    cpu_count: Optional[int] = None,
    load_1m: Optional[float] = None,
    reserve_threads: int = 8,
    ceiling: int = 10,
) -> int:
    """Conservatively size core workers around Blackwell and desktop load."""
    cpus = int(cpu_count or os.cpu_count() or 1)
    if load_1m is None:
        try:
            load_1m = os.getloadavg()[0]
        except OSError:
            load_1m = 0.0
    free = cpus - int(math.ceil(max(0.0, float(load_1m)))) - reserve_threads
    return max(2, min(int(ceiling), max(2, free)))


def validate_gpu0_isolation(torch_module=None) -> dict[str, Any]:
    """Fail closed unless this process sees only PCI-order physical GPU 0."""
    order = os.environ.get("CUDA_DEVICE_ORDER")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if order != "PCI_BUS_ID" or visible != "0":
        raise RuntimeError(
            "core pipeline requires CUDA_DEVICE_ORDER=PCI_BUS_ID and "
            "CUDA_VISIBLE_DEVICES=0"
        )
    report: dict[str, Any] = {
        "cuda_device_order": order,
        "cuda_visible_devices": visible,
    }
    if torch_module is not None:
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA unavailable in core pipeline")
        if torch_module.cuda.device_count() != 1:
            raise RuntimeError(
                f"expected exactly one visible CUDA device, got "
                f"{torch_module.cuda.device_count()}"
            )
        name = torch_module.cuda.get_device_name(0)
        if "3080" not in name.lower():
            raise RuntimeError(f"visible GPU is not RTX 3080 Ti: {name}")
        report["device_name"] = name
    return report


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def load_pipeline_state(run_dir: Path, run_name: str) -> dict[str, Any]:
    path = run_dir / "pipeline_state.json"
    if not path.is_file():
        return {
            "run_name": run_name,
            "current_phase": PHASE_GRAPH[0],
            "completed_phases": [],
            "artifacts": {},
            "updated_at": time.time(),
        }
    state = json.loads(path.read_text())
    phase = state.get("current_phase")
    if phase is not None and phase not in PHASE_GRAPH:
        raise ValueError(f"invalid resumed phase {phase!r}")
    return state


def save_pipeline_state(
    run_dir: Path, state: dict[str, Any]
) -> dict[str, Any]:
    state["updated_at"] = time.time()
    _atomic_json(run_dir / "pipeline_state.json", state)
    return state


def advance_pipeline_state(
    run_dir: Path,
    state: dict[str, Any],
    completed_phase: str,
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    if state.get("current_phase") != completed_phase:
        raise ValueError(
            f"cannot complete {completed_phase}: state is "
            f"{state.get('current_phase')}"
        )
    completed = list(state.get("completed_phases") or [])
    if completed_phase not in completed:
        completed.append(completed_phase)
    state = {
        **state,
        "completed_phases": completed,
        "current_phase": next_phase(completed_phase),
        "artifacts": {**dict(state.get("artifacts") or {}), **artifacts},
    }
    return save_pipeline_state(run_dir, state)


def _spec_dict(spec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "name": spec.name,
        "dir_name": spec.dir_name,
        "group": spec.group,
        "source": spec.source,
        "path": str(spec.path),
    }


def build_core_bc_jobs(
    specs: Iterable[Any], games: int, seed: int
) -> list[dict[str, Any]]:
    """Build deterministic, diverse baseline-vs-baseline demonstration jobs."""
    rows = list(specs)
    if len(rows) < 2:
        raise ValueError("core BC collection requires at least two baselines")
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    jobs: list[dict[str, Any]] = []
    for job_id in range(int(games)):
        a_idx = order[job_id % len(order)]
        stride = 1 + ((job_id // len(order)) % (len(order) - 1))
        b_idx = order[(job_id + stride) % len(order)]
        if b_idx == a_idx:
            b_idx = order[(job_id + stride + 1) % len(order)]
        jobs.append(
            {
                "job_id": job_id,
                "seed": seed + job_id,
                "spec0": _spec_dict(rows[a_idx]),
                "spec1": _spec_dict(rows[b_idx]),
            }
        )
    return jobs


def _worker_core_bc_game(job: dict[str, Any]) -> dict[str, Any]:
    """Collect two public-information behavior trajectories from one game."""
    from .agent import install_quiet_stdout, play_game
    from .baselines_runtime import BaselineSpec, load_baseline_agent
    from .replay_import import _strip_opp_private

    install_quiet_stdout(config.agent_verbose())
    job_id = int(job["job_id"])
    seed = int(job["seed"])

    def load(raw: dict[str, Any]):
        spec = BaselineSpec(
            **{**raw, "path": Path(raw["path"])}
        )
        fn, deck = load_baseline_agent(spec)
        return spec, fn, deck

    try:
        spec0, fn0, deck0 = load(job["spec0"])
        spec1, fn1, deck1 = load(job["spec1"])
        captured: list[list[dict[str, Any]]] = [[], []]

        def recorder(seat: int, fn):
            def call(obs: dict) -> list[int]:
                masked, _aux, report = _strip_opp_private(obs)
                if not report.ok:
                    raise RuntimeError(
                        "hidden-state guard violation: "
                        + "; ".join(report.violations)
                    )
                action = [int(x) for x in fn(obs)]
                captured[seat].append(
                    {
                        "observation": masked,
                        "action": action,
                        "env_step": len(captured[seat]),
                    }
                )
                return action

            return call

        def on_timeout(_signum, _frame):
            raise TimeoutError("core BC game exceeded 180s")

        had_alarm = hasattr(signal, "SIGALRM")
        if had_alarm:
            signal.signal(signal.SIGALRM, on_timeout)
            signal.alarm(180)
        t0 = time.perf_counter()
        try:
            outcome = play_game(
                recorder(0, fn0),
                recorder(1, fn1),
                deck0,
                deck1,
            )
        finally:
            if had_alarm:
                signal.alarm(0)
        wall_s = time.perf_counter() - t0
        if outcome.get("failed_seat") is not None or outcome.get("incomplete"):
            return {
                "ok": False,
                "job_id": job_id,
                "error": outcome.get("error") or outcome.get("termination"),
                "wall_s": wall_s,
            }
        winner = int(outcome["winner"])
        specs = (spec0, spec1)
        decks = (deck0, deck1)
        records = []
        for seat in (0, 1):
            value = 0.0 if winner == 2 else (1.0 if winner == seat else -1.0)
            own_deck = decks[seat]
            opp = specs[1 - seat]
            records.append(
                {
                    "episode_id": f"core-bc-{job_id:08d}-seat{seat}",
                    "seat": seat,
                    "archetype": archetypes.classify_deck(own_deck),
                    "opp_archetype": opp.id,
                    "deck": list(own_deck),
                    "value": value,
                    "steps": captured[seat],
                    "policy_targets": [None] * len(captured[seat]),
                    "info_set_ok": True,
                    "source": "core_bc_public_baseline_behavior",
                    "target_provenance": {
                        "trusted": True,
                        "target_source": "observed_public_behavior_action",
                        "search_mode": "none",
                        "job_id": job_id,
                        "simulator": "competition_libcg",
                    },
                }
            )
        return {
            "ok": True,
            "job_id": job_id,
            "records": records,
            "winner": winner,
            "steps": int(outcome.get("steps", 0)),
            "decisions": sum(len(row) for row in captured),
            "wall_s": wall_s,
            "deck_hashes": [
                hashlib.sha256(
                    ",".join(map(str, deck)).encode()
                ).hexdigest()[:16]
                for deck in decks
            ],
            "agents": [spec0.id, spec1.id],
        }
    except BaseException as exc:  # worker must account for every job
        return {
            "ok": False,
            "job_id": job_id,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_s": 0.0,
        }


@dataclass
class CollectionAccounting:
    scheduled_games: int
    completed_games: int = 0
    accepted_games: int = 0
    dropped_games: int = 0
    records: int = 0
    decisions: int = 0
    wall_s: float = 0.0
    deck_hashes: set[str] = field(default_factory=set)
    agents: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)

    def add(self, result: dict[str, Any]) -> None:
        self.completed_games += 1
        if not result.get("ok"):
            self.dropped_games += 1
            self.errors.append(str(result.get("error") or "unknown"))
            return
        records = list(result.get("records") or [])
        if len(records) != 2 or any(not row.get("steps") for row in records):
            self.dropped_games += 1
            self.errors.append("missing complete two-seat records")
            return
        self.accepted_games += 1
        self.records += len(records)
        self.decisions += int(result.get("decisions", 0))
        self.wall_s += float(result.get("wall_s", 0.0))
        self.deck_hashes.update(result.get("deck_hashes") or [])
        self.agents.update(result.get("agents") or [])

    def as_dict(self) -> dict[str, Any]:
        return {
            **{
                key: value
                for key, value in asdict(self).items()
                if key not in ("deck_hashes", "agents")
            },
            "deck_hashes": sorted(self.deck_hashes),
            "agents": sorted(self.agents),
        }


def _resume_complete_job_ids(partial_path: Path) -> set[int]:
    if not partial_path.is_file():
        return set()
    rows: list[tuple[int, str]] = []
    counts: Counter[int] = Counter()
    for raw in partial_path.read_text().splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        job_id = int((record.get("target_provenance") or {}).get("job_id", -1))
        if job_id >= 0:
            rows.append((job_id, raw))
            counts[job_id] += 1
    complete = {job_id for job_id, count in counts.items() if count == 2}
    if len(rows) != sum(counts[job_id] for job_id in complete):
        tmp = partial_path.with_suffix(partial_path.suffix + ".resume.tmp")
        tmp.write_text(
            "".join(raw + "\n" for job_id, raw in rows if job_id in complete)
        )
        os.replace(tmp, partial_path)
    return complete


def collect_core_bc_corpus(
    *,
    out_path: Path,
    games: int,
    workers: int,
    seed: int,
    report_path: Path,
) -> dict[str, Any]:
    """Collect broad public-information baseline demonstrations concurrently."""
    from .baselines_runtime import (
        ensure_baselines_installed,
        filter_loadable_baselines,
        load_manifest,
    )
    from .worker_pool import WorkerPool

    if out_path.is_file():
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        print(
            f"[core-pipeline][core_bc_collect] resume_complete games="
            f"{report.get('accepted_games', '?')} path={out_path}",
            flush=True,
        )
        return report
    specs = ensure_baselines_installed(load_manifest())
    specs, unloadable = filter_loadable_baselines(specs, verbose=False)
    if len(specs) < 4:
        raise RuntimeError(
            f"broad core BC needs >=4 loadable baseline decks; got {len(specs)}"
        )
    jobs = build_core_bc_jobs(specs, games, seed)
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    complete = _resume_complete_job_ids(partial)
    remaining = [job for job in jobs if int(job["job_id"]) not in complete]
    accounting = CollectionAccounting(scheduled_games=len(jobs))
    accounting.completed_games = len(complete)
    accounting.accepted_games = len(complete)
    accounting.records = 2 * len(complete)
    started = time.perf_counter()
    os.environ["POKEBOT_WORKER_CPU_ONLY"] = "1"
    from tqdm.auto import tqdm

    game_bar = tqdm(
        total=len(jobs),
        initial=len(complete),
        desc="core bc games",
        unit="game",
        file=sys.stderr,
        mininterval=0.5,
        ascii=True,
        dynamic_ncols=False,
        leave=True,
    )
    with partial.open("a", encoding="utf-8") as fh:
        with WorkerPool(num_workers=workers) as pool:
            try:
                for result in pool.imap_unordered(
                    _worker_core_bc_game, remaining, chunksize=1
                ):
                    accounting.add(result)
                    if result.get("ok"):
                        chunk = "".join(
                            json.dumps(record) + "\n"
                            for record in result["records"]
                        )
                        fh.write(chunk)
                        fh.flush()
                        if accounting.completed_games % 8 == 0:
                            os.fsync(fh.fileno())
                    game_bar.update(1)
                    game_bar.set_postfix(
                        ok=accounting.accepted_games,
                        drop=accounting.dropped_games,
                        dec=accounting.decisions,
                    )
            finally:
                game_bar.close()
    if accounting.dropped_games or accounting.completed_games != len(jobs):
        raise RuntimeError(
            "FATAL HEALTH GATE: core BC collection had failed or missing games: "
            f"{accounting.as_dict()}"
        )
    if len(accounting.deck_hashes) < 4 or len(accounting.agents) < 4:
        raise RuntimeError(
            "FATAL HEALTH GATE: core BC corpus lacks broad deck diversity"
        )
    os.replace(partial, out_path)
    report = {
        **accounting.as_dict(),
        "elapsed_seconds": time.perf_counter() - started,
        "workers": workers,
        "unloadable_baselines": unloadable,
        "path": str(out_path),
    }
    _atomic_json(report_path, report)
    return report


def snapshot_immutable_checkpoint(source: Path, run_name: str, label: str) -> dict[str, str]:
    digest = checkpoint.checkpoint_digest(source)
    short = digest.split(":", 1)[-1][:16]
    dest = paths.CHECKPOINTS_DIR / f"{run_name}.{label}.{short}.pt"
    if not dest.exists():
        os.link(source, dest)
    elif checkpoint.checkpoint_digest(dest) != digest:
        raise RuntimeError(f"immutable checkpoint collision: {dest}")
    return {"path": str(dest.resolve()), "digest": digest}


def core_to_specialist_transfer_report(
    core_checkpoint: Path,
    archetype_id: str = "hammer-pult",
) -> dict[str, Any]:
    """Report exact tensor compatibility for the eventual specialist transfer."""
    import torch

    from .core_kernel import CoreKernel

    kernel = CoreKernel.load_core_kernel(core_checkpoint, device=torch.device("cpu"))
    specialist = kernel.warm_start_specialist(
        archetype_id, reinit_heads=False, fold_archetype=True
    )
    source = kernel.net.state_dict()
    target = specialist.state_dict()
    loaded = [
        name
        for name, tensor in source.items()
        if name in target and tuple(tensor.shape) == tuple(target[name].shape)
    ]
    skipped = {
        name: {
            "source": list(tensor.shape),
            "target": list(target[name].shape) if name in target else None,
        }
        for name, tensor in source.items()
        if name not in loaded
    }
    return {
        "core_checkpoint": str(core_checkpoint),
        "archetype": archetype_id,
        "loaded_tensors": loaded,
        "loaded_count": len(loaded),
        "skipped_tensors": skipped,
        "target_tensor_count": len(target),
        "complete_shape_transfer": len(loaded) == len(target) and not skipped,
        "excluded_core_only_tensors": ["archetype_embed.weight"],
    }

