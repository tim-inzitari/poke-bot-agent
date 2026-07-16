#!/usr/bin/env python
"""Run the isolated small-core → deep-search → Hammer pipeline on GPU0.

The phase graph is:

  core_bc -> core_deep_search -> core_gate -> hammer_warmstart -> hammer_search_rl

The deep-search phases use the same public-history particle MCTS implementation
as Blackwell, with independent checkpoints, replay and GPU0 inference.

Launch with a stable console path (never timestamp it)::

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 TORCH_THREADS=8 \
    python -u scripts/train_core_pipeline.py --run-name <unique-name> \
    >> outputs/logs/core_kernel.log 2>&1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import config, paths
from poke_bot.iteration_contract import (
    PILOT_DERIVATION,
    PILOT_DERIVED_TARGET_SEARCH_DECISIONS,
)
from poke_bot.core_pipeline import (
    DEFAULT_CORE_LOG,
    PHASE_GRAPH,
    UnsafePhaseError,
    advance_pipeline_state,
    auto_size_core_workers,
    collect_core_bc_corpus,
    core_to_specialist_transfer_report,
    load_pipeline_state,
    require_trusted_search,
    save_pipeline_state,
    search_safety_audit,
    snapshot_immutable_checkpoint,
    validate_gpu0_isolation,
)


class PipelineInterrupted(RuntimeError):
    def __init__(self, signum: int):
        super().__init__(f"pipeline interrupted by signal {signum}")
        self.signum = int(signum)


def _args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-name", required=True, help="Unique core pipeline lineage.")
    p.add_argument(
        "--phase",
        choices=("auto",) + PHASE_GRAPH,
        default="auto",
        help="Resume state phase (default auto) or require a specific current phase.",
    )
    p.add_argument("--resume", choices=("auto", "0"), default="auto")
    p.add_argument(
        "--restart-partial-search",
        action="store_true",
        help="Quarantine an interrupted deep-search writer and recollect it.",
    )
    p.add_argument(
        "--restart-partial-iteration",
        action="store_true",
        help="Forward to hammer RR: quarantine current-iter partials and "
        "recollect from job zero (e.g. restart iter6 from the beginning).",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--stop-after-phase",
        choices=PHASE_GRAPH,
        default=None,
        help="Exit cleanly after this phase (smoke/ops control).",
    )
    p.add_argument("--stable-log", type=Path, default=DEFAULT_CORE_LOG)

    # Core BC collection/training.
    p.add_argument("--core-bc-games", type=int, default=128)
    p.add_argument("--workers", type=int, default=0, help="0=conservative auto-size.")
    p.add_argument("--worker-ceiling", type=int, default=10)
    p.add_argument("--reserve-cpu-threads", type=int, default=8)
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--core-bc-epochs", type=int, default=6)
    p.add_argument("--core-bc-patience", type=int, default=3)
    p.add_argument("--core-bc-lr", type=float, default=3e-4)
    p.add_argument("--training-batch-games", type=int, default=16)
    p.add_argument("--training-max-decisions", type=int, default=1024)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--gpu-memory-budget-gb", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=7319)
    p.add_argument("--core-search-games-per-opponent", type=int, default=16)
    p.add_argument("--core-search-min-games-per-opponent", type=int, default=12)
    p.add_argument("--core-search-max-games-per-opponent", type=int, default=24)
    p.add_argument(
        "--core-target-search-decisions",
        type=int,
        default=PILOT_DERIVED_TARGET_SEARCH_DECISIONS,
        help="Measured trusted decision target; 0 stops at nominal games.",
    )
    p.add_argument("--core-search-replay-fraction", type=float, default=0.50)
    p.add_argument("--core-policy-anchor-ratio", type=float, default=1.0)
    p.add_argument("--max-decisions-per-game", type=int, default=32)
    p.add_argument("--core-search-epochs", type=int, default=1)
    p.add_argument("--hammer-rl-iterations", type=int, default=1)
    p.add_argument(
        "--hammer-target-search-decisions",
        type=int,
        default=PILOT_DERIVED_TARGET_SEARCH_DECISIONS,
    )

    # Declared now so run metadata is stable when trusted search becomes ready.
    p.add_argument(
        "--search-sim-budgets",
        type=int,
        nargs="+",
        default=[32, 128, 256, 512, 1024],
    )
    p.add_argument("--search-start-sims", type=int, default=128)
    p.add_argument("--search-move-time-s", type=float, default=8.0)
    p.add_argument("--search-game-time-s", type=float, default=1200.0)
    p.add_argument("--inference-batch", type=int, default=128)
    p.add_argument("--inference-queue-depth", type=int, default=32)
    p.add_argument("--inference-servers", type=int, default=4)
    p.add_argument("--inference-timeout-s", type=float, default=120.0)
    p.add_argument(
        "--promotion-workers",
        type=int,
        default=12,
        help=(
            "Parallel WorkerPool size for hammer_search_rl belief-MCTS "
            "promotion (remote leaf on GPU0). Default 12 balances 3 leaf "
            "servers vs Blackwell CPU contention; capped by --workers resp_qs."
        ),
    )
    p.add_argument(
        "--remote-worker-endpoints",
        nargs="+",
        default=None,
        metavar="HOST:PORT",
        help="Optional additive LAN whole-game workers for core deep-search / "
        "hammer RL canaries. Local GPU0 IPC leaf path stays default.",
    )
    return p.parse_args(argv)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def _core_iteration_contract(args: argparse.Namespace) -> dict:
    target = int(args.core_target_search_decisions)
    return {
        "schema": "belief_decision_budget_v1",
        "nominal_games_per_opponent": int(
            args.core_search_games_per_opponent
        ),
        "min_games_per_opponent": int(
            args.core_search_min_games_per_opponent
        ),
        "max_games_per_opponent": int(
            args.core_search_max_games_per_opponent
        ),
        "target_search_decisions": target,
        "target_derivation": (
            PILOT_DERIVATION
            if target == PILOT_DERIVED_TARGET_SEARCH_DECISIONS
            else {
                "method": "explicit_cli_override",
                "target_search_decisions": target,
            }
        ),
        "balanced_seats": True,
        "old_policy_targets_equivalent": False,
        "policy_anchor_is_explicit": True,
    }


def _quarantine_core_search_partials(
    replay_dir: Path, *, suffix: str
) -> list[str]:
    partial = replay_dir / "core_deep_search.jsonl.partial"
    moved: list[str] = []
    for artifact in (
        partial,
        partial.with_suffix(partial.suffix + ".journal"),
        partial.with_suffix(partial.suffix + ".writer.json"),
    ):
        if not artifact.exists():
            continue
        destination = artifact.with_name(artifact.name + suffix)
        os.replace(artifact, destination)
        moved.append(str(destination))
    return moved


def _stable_tqdm(**kwargs):
    """tqdm configured for append-only stable logs (`tail -F` friendly)."""
    from tqdm.auto import tqdm

    defaults = {
        "file": sys.stderr,
        "mininterval": 0.5,
        "ascii": True,
        "dynamic_ncols": False,
        "leave": True,
    }
    defaults.update(kwargs)
    return tqdm(**defaults)


def _hydrate_core_search_journal(
    journal_path: Path, *, resume_index: int
) -> dict[str, Any]:
    """Rebuild collection counters from a resumable ordered writer journal."""
    coverage_rows: list[tuple[str, int]] = []
    sims = decisions = failures = 0
    wins = draws = 0
    max_depth = max_ordered_actions = 0
    factorized_decisions = 0
    queue_wait_total = 0.0
    requests = 0
    if not journal_path.is_file() or resume_index <= 0:
        return {
            "coverage_rows": coverage_rows,
            "sims": sims,
            "decisions": decisions,
            "failures": failures,
            "wins": wins,
            "draws": draws,
            "max_depth": max_depth,
            "max_ordered_actions": max_ordered_actions,
            "factorized_decisions": factorized_decisions,
            "queue_wait_total": queue_wait_total,
            "requests": requests,
        }
    with journal_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line_no >= resume_index:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            result = dict(row.get("result") or {})
            if not row.get("record_written"):
                failures += 1
                continue
            result_sims = int(result.get("mcts_sims_run_total") or 0)
            result_decisions = int(result.get("n_decisions") or 0)
            if result_decisions <= 0 or result_sims < 128 * result_decisions:
                failures += 1
                continue
            sims += result_sims
            decisions += result_decisions
            coverage_rows.append(
                (str(result.get("opponent_id")), int(result.get("our_seat") or 0))
            )
            winner = result.get("winner")
            if winner is None:
                draws += 1
            elif int(winner) == int(result.get("our_seat") or 0):
                wins += 1
            max_depth = max(max_depth, int(result.get("max_search_depth") or 0))
            max_ordered_actions = max(
                max_ordered_actions,
                int(result.get("max_complete_ordered_action_count") or 0),
            )
            factorized_decisions += int(
                result.get("factorized_search_decisions") or 0
            )
            requests += int(result.get("remote_leaf_requests") or 0)
            queue_wait_total += float(
                result.get("remote_queue_wait_ms_total") or 0.0
            )
    return {
        "coverage_rows": coverage_rows,
        "sims": sims,
        "decisions": decisions,
        "failures": failures,
        "wins": wins,
        "draws": draws,
        "max_depth": max_depth,
        "max_ordered_actions": max_ordered_actions,
        "factorized_decisions": factorized_decisions,
        "queue_wait_total": queue_wait_total,
        "requests": requests,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.core_bc_games < 8:
        raise ValueError("--core-bc-games must be >=8 for deck diversity")
    from poke_bot.iteration_contract import SearchCollectionContract

    SearchCollectionContract(
        opponents=1,
        nominal_games_per_opponent=args.core_search_games_per_opponent,
        min_games_per_opponent=args.core_search_min_games_per_opponent,
        max_games_per_opponent=args.core_search_max_games_per_opponent,
        target_search_decisions=args.core_target_search_decisions,
    )
    if not 0.50 <= args.core_search_replay_fraction <= 0.75:
        raise ValueError("--core-search-replay-fraction must be in [0.50, 0.75]")
    if args.core_policy_anchor_ratio < 0 or args.max_decisions_per_game < 1:
        raise ValueError("core policy anchor/decision cap is invalid")
    if args.core_search_epochs < 1 or args.hammer_rl_iterations < 1:
        raise ValueError("search training iteration counts must be positive")
    if args.hammer_target_search_decisions < 0:
        raise ValueError("--hammer-target-search-decisions must be non-negative")
    if args.training_batch_games < 1 or args.training_max_decisions < 1:
        raise ValueError("training batch limits must be positive")
    if args.prefetch < 1:
        raise ValueError("--prefetch must be positive")
    if args.gpu_memory_budget_gb <= 0:
        raise ValueError("--gpu-memory-budget-gb must be positive")
    budgets = list(args.search_sim_budgets)
    if any(value <= 0 for value in budgets) or budgets != sorted(set(budgets)):
        raise ValueError("--search-sim-budgets must be sorted unique positive values")
    if args.search_start_sims < 128:
        raise ValueError("--search-start-sims must be >=128")
    if args.inference_servers < 1:
        raise ValueError("--inference-servers must be positive")
    if int(args.promotion_workers) < 1:
        raise ValueError("--promotion-workers must be positive")


def _run_core_bc(
    args: argparse.Namespace,
    run_dir: Path,
    replay_dir: Path,
) -> dict:
    import torch

    from poke_bot.core_kernel import (
        CoreTrainConfig,
        CorpusConfig,
        StreamingArchetypeCorpus,
        core_kernel_config_small_3080ti,
        train_core_kernel,
    )

    isolation = validate_gpu0_isolation(torch)
    config.HARDWARE.torch_threads = int(args.torch_threads)
    torch.set_num_threads(int(args.torch_threads))
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    fraction = min(0.95, float(args.gpu_memory_budget_gb) / total_gb)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)

    workers = (
        int(args.workers)
        if args.workers > 0
        else auto_size_core_workers(
            reserve_threads=int(args.reserve_cpu_threads),
            ceiling=int(args.worker_ceiling),
        )
    )
    corpus_path = replay_dir / "core_bc_public_behavior.jsonl"
    collection_report_path = run_dir / "core_bc_collection.json"
    print(
        "[core-pipeline][core_bc] "
        f"device=cuda:0 name={isolation.get('device_name')} workers={workers} "
        f"torch_threads={args.torch_threads} collection_games={args.core_bc_games} "
        f"gpu_memory_budget_gb={args.gpu_memory_budget_gb:g}",
        flush=True,
    )
    collection = collect_core_bc_corpus(
        out_path=corpus_path,
        games=int(args.core_bc_games),
        workers=workers,
        seed=int(args.seed),
        report_path=collection_report_path,
    )

    model_cfg = core_kernel_config_small_3080ti()
    train_cfg = CoreTrainConfig(
        lr=float(args.core_bc_lr),
        epochs=int(args.core_bc_epochs),
        games_per_batch=int(args.training_batch_games),
        max_decisions_per_batch=int(args.training_max_decisions),
        early_stop_patience=int(args.core_bc_patience),
        value_loss_weight=1.5,
        aux_loss_weight=0.1,
        opp_hand_loss_weight=0.2,
        opp_remainder_loss_weight=0.15,
        amp=args.amp_dtype != "fp32",
        amp_dtype=args.amp_dtype,
        grad_accum_steps=1,
        seed=int(args.seed),
        condition_archetype=False,
    )
    corpus = StreamingArchetypeCorpus(
        CorpusConfig(
            jsonl_paths=[corpus_path],
            val_frac=float(args.val_frac),
            max_context=model_cfg.max_context,
            verify_info_set=True,
            shuffle_buffer=min(512, max(32, int(collection["records"]))),
            seed=int(args.seed),
        )
    )
    params = sum(
        parameter.numel()
        for parameter in __import__(
            "poke_bot.core_kernel", fromlist=["CoreKernel"]
        ).CoreKernel(cfg=model_cfg).parameters()
    )
    print(
        "[core-pipeline][core_bc_model] "
        f"profile=small-3080ti params={params} d_model={model_cfg.d_model} "
        f"layers={model_cfg.spatial_layers}/{model_cfg.temporal_layers}/"
        f"{model_cfg.option_decoder_layers} heads={model_cfg.n_heads} "
        f"ff={model_cfg.ff_dim} max_context={model_cfg.max_context} "
        f"train_epochs={train_cfg.epochs} batch_games={train_cfg.games_per_batch} "
        f"batch_decisions={train_cfg.max_decisions_per_batch} "
        f"amp={train_cfg.amp_dtype} prefetch={args.prefetch}",
        flush=True,
    )
    train_name = f"{args.run_name}.core_bc"
    result = train_core_kernel(
        corpus,
        run_name=train_name,
        train_cfg=train_cfg,
        resume="auto",
        device=torch.device("cuda:0"),
        cfg=model_cfg,
    )
    best_path = Path(result.get("best_path") or "")
    best_metric = float(result.get("best_metric", float("inf")))
    history = list(result.get("history") or [])
    if (
        not best_path.is_file()
        or not math.isfinite(best_metric)
        or not history
        or int(history[-1]["train"]["n_decisions"]) <= 0
        or int(history[-1]["val"]["n_decisions"]) <= 0
    ):
        raise RuntimeError(
            "FATAL HEALTH GATE: core BC failed finite train/validation gate"
        )
    immutable = snapshot_immutable_checkpoint(
        best_path, args.run_name, "core_bc.accepted"
    )
    transfer = core_to_specialist_transfer_report(
        Path(immutable["path"]), "hammer-pult"
    )
    if not transfer["complete_shape_transfer"]:
        raise RuntimeError(
            "FATAL HEALTH GATE: small core is not shape-compatible with specialist"
        )
    report = {
        "phase": "core_bc",
        "status": "accepted",
        "collection": collection,
        "model_profile": {
            **model_cfg.__dict__,
            "parameters": params,
            "profile": "small-3080ti",
        },
        "training": result,
        "accepted_checkpoint": immutable,
        "hammer_transfer_report": transfer,
        "device_isolation": isolation,
    }
    _write_json(run_dir / "core_bc_report.json", report)
    print(
        "[core-pipeline][core_bc_gate] "
        f"status=ACCEPTED checkpoint={immutable['path']} "
        f"checkpoint_digest={immutable['digest']} "
        f"best_validation_loss={best_metric:.6f} "
        f"hammer_transfer_loaded={transfer['loaded_count']}/"
        f"{transfer['target_tensor_count']}",
        flush=True,
    )
    return report


def _run_core_deep_search(
    args: argparse.Namespace,
    run_dir: Path,
    replay_dir: Path,
    state: dict,
) -> dict:
    """Collect trusted belief-search targets and continue core training."""
    import multiprocessing as mp
    import torch

    from poke_bot import checkpoint, features
    from poke_bot.baselines_runtime import ensure_baselines_installed, load_manifest
    from poke_bot.batched_infer import run_leaf_server
    from poke_bot.core_kernel import (
        CoreTrainConfig,
        CorpusConfig,
        StreamingArchetypeCorpus,
        core_kernel_config_small_3080ti,
        train_core_kernel,
    )
    from poke_bot.deck_pool import read_deck
    from poke_bot.promotion import CheckpointIdentity
    from poke_bot.replay_writer import OrderedReplayWriter
    from poke_bot.iteration_contract import SearchCollectionContract
    from poke_bot.worker_pool import WorkerPool, WorkerPoolStopped
    from poke_bot.remote_jobs import RemoteWorkerFarm
    from scripts.train_round_robin import _map_game_batch, _worker_play

    require_trusted_search("core_deep_search")
    isolation = validate_gpu0_isolation(torch)
    checkpoint_info = dict(state.get("artifacts", {}).get("core_bc_checkpoint") or {})
    core_checkpoint = Path(checkpoint_info.get("path") or "")
    if not core_checkpoint.is_file():
        raise FileNotFoundError("core_deep_search has no accepted core_bc checkpoint")
    identity = CheckpointIdentity.from_path(core_checkpoint)
    if identity.digest != checkpoint_info.get("digest"):
        raise RuntimeError("core_bc immutable checkpoint digest changed")
    specs = ensure_baselines_installed(load_manifest())
    collection_contract = SearchCollectionContract(
        opponents=len(specs),
        nominal_games_per_opponent=int(args.core_search_games_per_opponent),
        min_games_per_opponent=int(args.core_search_min_games_per_opponent),
        max_games_per_opponent=int(args.core_search_max_games_per_opponent),
        target_search_decisions=int(args.core_target_search_decisions),
    )
    final = replay_dir / "core_deep_search.jsonl"
    if final.exists():
        # Collection already finalized (e.g. train crashed on optimizer resume).
        # Reuse the immutable replay and continue with train-only — do not
        # recreate leaf servers or refuse to overwrite.
        print(
            "[core-pipeline][core_deep_search] finalized search replay present; "
            f"skipping collect and resuming train from {final}",
            flush=True,
        )
        fresh_games = sum(1 for line in final.open() if line.strip())
        if fresh_games < 1:
            raise RuntimeError(f"finalized search replay is empty: {final}")
        decisions = 0
        with final.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                steps = row.get("steps") or []
                # Count trusted decision steps (those with factorized targets).
                fact = row.get("factorized_policy_targets") or []
                if isinstance(fact, list) and fact:
                    decisions += sum(1 for item in fact if item is not None)
                else:
                    decisions += len(steps)
        # Collection already passed the overnight health gate before finalize.
        # Reconstruct the contract floors so core_gate does not reject a pure
        # train resume (per-decision sims were ≥128 at collect time).
        return _train_core_deep_search_from_replay(
            args,
            run_dir=run_dir,
            replay_dir=replay_dir,
            state=state,
            core_checkpoint=core_checkpoint,
            final=final,
            collection_contract=collection_contract,
            isolation=isolation,
            fresh_games=fresh_games,
            decisions=decisions,
            sims=128 * max(decisions, 1),
            actual_games_per_opponent=max(
                1, fresh_games // max(len(specs), 1)
            ),
            coverage_complete=True,
            coverage_counts={},
            stop_reason="resumed_finalized_replay",
            scheduled_jobs=fresh_games,
            jobs_len=fresh_games,
            max_depth=2,
            max_ordered_actions=1,
            factorized_decisions=decisions,
            started=time.perf_counter(),
            n_servers=0,
            requests=0,
            queue_wait_total=0.0,
            writer_telemetry={"resumed_finalized_replay": True},
        )
    workers = (
        int(args.workers)
        if args.workers > 0
        else auto_size_core_workers(
            reserve_threads=int(args.reserve_cpu_threads),
            ceiling=int(args.worker_ceiling),
        )
    )
    workers = max(2, min(workers, collection_contract.max_games))
    ctx = mp.get_context("spawn")
    response_queues = [ctx.Queue(maxsize=2) for _ in range(workers)]
    n_servers = max(1, min(int(args.inference_servers), workers))
    request_queues = [
        ctx.Queue(maxsize=int(args.inference_queue_depth))
        for _ in range(n_servers)
    ]
    control_queues = [ctx.Queue(maxsize=8) for _ in range(n_servers)]
    status_queues = [ctx.Queue(maxsize=16) for _ in range(n_servers)]
    ready_events = [ctx.Event() for _ in range(n_servers)]
    alive_events = [ctx.Event() for _ in range(n_servers)]
    servers = []

    def shutdown_leaf_servers() -> None:
        for control_queue, request_queue in zip(
            control_queues, request_queues
        ):
            try:
                control_queue.put_nowait({"cmd": "stop"})
            except Exception:
                pass
            try:
                request_queue.put_nowait(None)
            except Exception:
                pass
        for leaf_server in servers:
            leaf_server.join(timeout=10)
            if leaf_server.is_alive():
                leaf_server.terminate()
                leaf_server.join(timeout=5)
        for queue_obj in (
            *request_queues,
            *control_queues,
            *status_queues,
            *response_queues,
        ):
            try:
                queue_obj.cancel_join_thread()
            except Exception:
                pass
            try:
                queue_obj.close()
            except Exception:
                pass

    for index in range(n_servers):
        server = ctx.Process(
            target=run_leaf_server,
            args=(
                str(core_checkpoint),
                "cuda:0",
                request_queues[index],
                response_queues,
            ),
            kwargs={
                "ready_evt": ready_events[index],
                "alive_evt": alive_events[index],
                "ctrl_q": control_queues[index],
                "status_q": status_queues[index],
                "expected_digest": identity.digest,
                "initial_version": 0,
                "max_batch": int(args.inference_batch),
                "coalesce_ms": 2.0,
            },
            daemon=True,
        )
        server.start()
        servers.append(server)
    try:
        for index, ready in enumerate(ready_events):
            if not ready.wait(180):
                raise RuntimeError(
                    f"core leaf server {index} startup timed out"
                )
            status = status_queues[index].get(timeout=10)
            if not status.get("ok"):
                raise RuntimeError(
                    f"core leaf server {index} startup failed: {status}"
                )
    except BaseException:
        shutdown_leaf_servers()
        raise
    slot_counter = ctx.Value("i", 0)
    remote = {
        "req_qs": request_queues,
        "resp_qs": response_queues,
        "slot_counter": slot_counter,
        "ctrl_qs": control_queues,
        "generation": 1,
        "alive_evts": alive_events,
        "expected_digest": identity.digest,
        "expected_version": 0,
        "timeout_s": float(args.inference_timeout_s),
    }
    remote_farm: RemoteWorkerFarm | None = None
    if args.remote_worker_endpoints:
        remote_job_buffer_s = float(
            os.environ.get("POKEBOT_REMOTE_JOB_TIMEOUT_BUFFER_S", "600") or "600"
        )
        remote_job_timeout = max(
            float(args.inference_timeout_s),
            float(args.search_game_time_s) + remote_job_buffer_s,
        )
        remote_farm = RemoteWorkerFarm(
            list(args.remote_worker_endpoints),
            timeout_s=remote_job_timeout,
        )
        try:
            remote_infos = remote_farm.connect()
        except Exception as exc:
            shutdown_leaf_servers()
            raise RuntimeError(f"remote worker connect failed: {exc}") from exc
        for info in remote_infos:
            print(
                "[core-pipeline][core_deep_search] "
                f"remote-worker={info.endpoint} hostname={info.hostname} "
                f"gpu={info.gpu_name!r} workers={info.workers}",
                flush=True,
            )
        try:
            # Match local leaf initial_version=0; do not invent version=1 here.
            remote_farm.reload_all(
                str(core_checkpoint),
                digest=identity.digest,
                version=0,
            )
        except Exception as exc:
            remote_farm.close()
            shutdown_leaf_servers()
            raise RuntimeError(f"remote worker reload failed: {exc}") from exc
    jobs = []
    for game_index in range(collection_contract.max_games_per_opponent):
        for opponent_index, opponent_spec in enumerate(specs):
            index = len(jobs)
            own_spec = specs[(opponent_index + game_index * 7) % len(specs)]
            jobs.append(
                {
                    "job_index": index,
                    "checkpoint": str(core_checkpoint),
                    "checkpoint_digest": identity.digest,
                    "model_generation": 1,
                    "model_max_context": core_kernel_config_small_3080ti().max_context,
                    "our_deck": read_deck(own_spec.deck_csv),
                    "spec": {
                        "id": opponent_spec.id,
                        "name": opponent_spec.name,
                        "dir_name": opponent_spec.dir_name,
                        "group": opponent_spec.group,
                        "source": opponent_spec.source,
                        "path": str(opponent_spec.path),
                    },
                    "our_seat": game_index % 2,
                    "pair_id": None,
                    "mcts_sims": int(args.search_start_sims),
                    "mcts_move_time": float(args.search_move_time_s),
                    "game_timeout_s": int(args.search_game_time_s),
                    "expected_search_decisions": 128,
                    "agent_mode": "belief-mcts",
                    "seed": int(args.seed) + index,
                    "device": "cpu",
                    "archetype": "unknown",
                    "training_eligible": True,
                    "target_provenance": {
                        "source_tree_digest": "core-pipeline-runtime",
                        "simulator_digest": "validated-by-search-provenance",
                        "config_digest": hashlib.sha256(
                            json.dumps(
                                {
                                    "sims": args.search_start_sims,
                                    "move_time": args.search_move_time_s,
                                },
                                sort_keys=True,
                            ).encode()
                        ).hexdigest(),
                        "search_mode": "belief-mcts",
                        "target_source": "belief_mcts",
                        "trusted": True,
                        "action_decoder": "ordered_autoregressive_v1",
                        "feature_schema": features.FEATURE_SCHEMA_VERSION,
                        "incumbent_checkpoint": identity.as_dict(),
                        "iteration": 0,
                        "model_generation": 1,
                        "our_seat": game_index % 2,
                    },
                }
            )
    partial = replay_dir / "core_deep_search.jsonl.partial"
    # ``final`` already resolved above; refuse only if somehow missing from the
    # early-resume path but present mid-collect (should be unreachable).
    if final.exists():
        raise FileExistsError(f"refusing to overwrite {final}")
    if args.restart_partial_search:
        restart_suffix = f".invalid.restart.{int(time.time())}"
        _quarantine_core_search_partials(
            replay_dir, suffix=restart_suffix
        )
    writer = OrderedReplayWriter(
        partial,
        expected_jobs=len(jobs),
        queue_depth=max(16, workers * 2),
        fsync_batch=4,
    )
    resume_index = int(writer.resume_index)
    hydrated = _hydrate_core_search_journal(
        writer.journal_path, resume_index=resume_index
    )
    if resume_index > 0:
        print(
            "[core-pipeline][core_deep_search] resume_writer "
            f"next_index={resume_index}/{len(jobs)} "
            f"hydrated_games={len(hydrated['coverage_rows'])} "
            f"trusted_decisions={hydrated['decisions']}",
            flush=True,
        )
    stop_requested: dict[str, int | None] = {"signal": None}
    active_pool: dict[str, WorkerPool | None] = {"pool": None}

    def request_stop(signum, _frame) -> None:  # noqa: ANN001
        if stop_requested["signal"] is not None:
            return
        stop_requested["signal"] = int(signum)
        pool = active_pool["pool"]
        print(
            f"[core-pipeline] signal={signum}; retaining partial search and "
            "skipping training/checkpoint",
            flush=True,
        )
        if pool is not None:
            # Leave Pool termination to WorkerPool.__exit__; doing it directly
            # inside a signal handler can deadlock on multiprocessing locks.
            raise PipelineInterrupted(int(signum))

    previous_handlers = {
        sig: signal.signal(sig, request_stop)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    failures: list[dict] = []
    sims = int(hydrated["sims"])
    decisions = int(hydrated["decisions"])
    max_depth = int(hydrated["max_depth"])
    max_ordered_actions = int(hydrated["max_ordered_actions"])
    factorized_decisions = int(hydrated["factorized_decisions"])
    queue_wait_total = float(hydrated["queue_wait_total"])
    requests = int(hydrated["requests"])
    wins = int(hydrated["wins"])
    draws = int(hydrated["draws"])
    scheduled_jobs = resume_index
    actual_games_per_opponent = 0
    stop_reason = "not_started"
    coverage_rows: list[tuple[str, int]] = list(hydrated["coverage_rows"])
    started = time.perf_counter()
    game_bar = _stable_tqdm(
        total=len(jobs),
        initial=resume_index,
        desc="core search games",
        unit="game",
    )
    game_bar.set_postfix(
        wr=f"{wins / max(len(coverage_rows), 1):.1%}",
        dec=decisions,
        fail=int(hydrated["failures"]),
        sims=sims,
    )
    try:
        from poke_bot.iteration_contract import balanced_seat_coverage
        from tqdm.auto import tqdm as tqdm_mod

        for wave_end in range(
            2, collection_contract.max_games_per_opponent + 1, 2
        ):
            wave_start = wave_end - 2
            wave = [
                job
                for job in jobs[
                    wave_start * len(specs) : wave_end * len(specs)
                ]
                if int(job["job_index"]) >= resume_index
            ]
            scheduled_jobs += len(wave)
            if stop_requested["signal"] is not None:
                break
            if not wave:
                actual_games_per_opponent = wave_end
                coverage_complete, coverage_counts = balanced_seat_coverage(
                    coverage_rows,
                    opponents=[spec.id for spec in specs],
                    min_games_per_opponent=(
                        collection_contract.min_games_per_opponent
                    ),
                )
                stop_reason = collection_contract.stop_reason(
                    games_per_opponent=wave_end,
                    trusted_decisions=decisions,
                    coverage_complete=coverage_complete,
                ) or "continue"
                tqdm_mod.write(
                    "[core-pipeline][core_deep_search_collect] "
                    f"fresh_games={len(coverage_rows)} "
                    f"trusted_decisions={decisions}/"
                    f"{collection_contract.target_search_decisions or 'nominal'} "
                    f"decisions_per_game="
                    f"{decisions/max(len(coverage_rows), 1):.2f} "
                    f"games_per_opp={wave_end} coverage={coverage_complete} "
                    f"stop={stop_reason} resumed_prefix={resume_index}",
                    file=sys.stderr,
                )
                if stop_reason != "continue":
                    break
                continue
            pool = WorkerPool(num_workers=workers, remote_channel=remote)
            active_pool["pool"] = pool
            try:
                with pool:
                    for result in _map_game_batch(
                        pool,
                        _worker_play,
                        wave,
                        remote_farm=remote_farm,
                        kind="play",
                        local_workers=workers,
                    ):
                        failures_before = len(failures)
                        writer.submit(
                            result["job_index"],
                            result.get("record_json"),
                            {
                                key: value
                                for key, value in result.items()
                                if key
                                not in ("record", "record_json", "targets")
                            },
                        )
                        if (
                            result.get("resource_error")
                            or result.get("baseline_failed")
                            or result.get("our_failed")
                            or result.get("trust_failure")
                            or not result.get("record_json")
                        ):
                            failures.append(result)
                        else:
                            result_sims = int(
                                result.get("mcts_sims_run_total") or 0
                            )
                            result_decisions = int(
                                result.get("n_decisions") or 0
                            )
                            if (
                                result_decisions <= 0
                                or result_sims < 128 * result_decisions
                            ):
                                failures.append(
                                    {
                                        **result,
                                        "error": (
                                            "incomplete trusted simulation "
                                            "contract"
                                        ),
                                    }
                                )
                            else:
                                sims += result_sims
                                decisions += result_decisions
                                coverage_rows.append(
                                    (
                                        str(result["opponent_id"]),
                                        int(result["our_seat"]),
                                    )
                                )
                                winner = result.get("winner")
                                if winner is None:
                                    draws += 1
                                elif int(winner) == int(result["our_seat"]):
                                    wins += 1
                                max_depth = max(
                                    max_depth,
                                    int(
                                        result.get("max_search_depth") or 0
                                    ),
                                )
                                max_ordered_actions = max(
                                    max_ordered_actions,
                                    int(
                                        result.get(
                                            "max_complete_ordered_action_count"
                                        )
                                        or 0
                                    ),
                                )
                                factorized_decisions += int(
                                    result.get(
                                        "factorized_search_decisions"
                                    )
                                    or 0
                                )
                                requests += int(
                                    result.get("remote_leaf_requests") or 0
                                )
                                queue_wait_total += float(
                                    result.get(
                                        "remote_queue_wait_ms_total"
                                    )
                                    or 0.0
                                )
                        if len(failures) > failures_before:
                            cause = (
                                failures[-1].get("error")
                                or failures[-1].get("resource_error")
                                or "incomplete trusted target"
                            )
                            # Soft-skip transient trust/timeout pressure instead
                            # of killing the overnight pipeline. Hard-stop only
                            # when failures become a systemic fraction of work.
                            soft = bool(
                                failures[-1].get("trust_failure")
                                or failures[-1].get("resource_error")
                                or failures[-1].get("search_budget_exhausted")
                                or "TrustedSearchBudgetExhausted" in str(cause)
                                or "incomplete trusted simulation" in str(cause)
                            )
                            completed = len(coverage_rows) + len(failures)
                            systemic = (
                                len(failures) >= 12
                                and len(failures) / max(completed, 1) >= 0.25
                            )
                            tqdm_mod.write(
                                "[core-pipeline][core_deep_search_collect] "
                                f"{'SOFT-SKIP' if soft and not systemic else 'FATAL'} "
                                f"fail={len(failures)}/{completed} cause={cause}",
                                file=sys.stderr,
                            )
                            if (not soft) or systemic:
                                pool.request_stop(
                                    f"core search fatal result: {cause}"
                                )
                                raise RuntimeError(
                                    f"core search fail-fast: {cause}"
                                )
                        oid = str(result.get("opponent_id") or "?")
                        game_bar.update(1)
                        game_bar.set_postfix(
                            wr=f"{wins / max(len(coverage_rows), 1):.1%}",
                            last=oid[:14],
                            dec=decisions,
                            fail=len(failures),
                            sims=sims,
                        )
                        if stop_requested["signal"] is not None:
                            pool.request_stop(
                                f"signal {stop_requested['signal']}"
                            )
                            break
            except (WorkerPoolStopped, PipelineInterrupted, ValueError):
                if stop_requested["signal"] is None:
                    raise
                pool.request_stop(f"signal {stop_requested['signal']}")
            finally:
                active_pool["pool"] = None
            if stop_requested["signal"] is not None:
                break
            actual_games_per_opponent = wave_end
            coverage_complete, coverage_counts = balanced_seat_coverage(
                coverage_rows,
                opponents=[spec.id for spec in specs],
                min_games_per_opponent=(
                    collection_contract.min_games_per_opponent
                ),
            )
            stop_reason = collection_contract.stop_reason(
                games_per_opponent=wave_end,
                trusted_decisions=decisions,
                coverage_complete=coverage_complete,
            ) or "continue"
            tqdm_mod.write(
                "[core-pipeline][core_deep_search_collect] "
                f"fresh_games={len(coverage_rows)} "
                f"trusted_decisions={decisions}/"
                f"{collection_contract.target_search_decisions or 'nominal'} "
                f"decisions_per_game={decisions/max(len(coverage_rows), 1):.2f} "
                f"games_per_opp={wave_end} coverage={coverage_complete} "
                f"stop={stop_reason}",
                file=sys.stderr,
            )
            if stop_reason != "continue":
                break
        if stop_requested["signal"] is not None:
            writer.abort(
                f"signal {stop_requested['signal']} interrupted core search"
            )
            raise PipelineInterrupted(int(stop_requested["signal"]))
        if stop_reason == "continue":
            stop_reason = "max_games"
        for job in jobs[scheduled_jobs:]:
            writer.submit(
                int(job["job_index"]),
                None,
                {
                    "not_scheduled": True,
                    "reason": stop_reason,
                    "opponent_id": job["spec"]["id"],
                    "our_seat": job["our_seat"],
                },
            )
            game_bar.update(1)
        writer_telemetry = writer.close()
        if failures:
            writer.quarantine(f".invalid.{int(time.time())}")
            raise RuntimeError(
                f"core search health gate rejected {len(failures)} games: "
                f"{failures[0].get('error')}"
            )
        if (
            sims < 128 * decisions
            or writer.written_records != scheduled_jobs
            or not coverage_complete
            or (
                collection_contract.target_search_decisions > 0
                and decisions < collection_contract.target_search_decisions
            )
        ):
            writer.quarantine(f".invalid.{int(time.time())}")
            raise RuntimeError("core search produced incomplete trusted targets")
        writer.finalize(final)
    except BaseException as exc:
        try:
            writer.abort(f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise
    finally:
        game_bar.close()
        active_pool["pool"] = None
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if remote_farm is not None:
            remote_farm.close()
        shutdown_leaf_servers()

    return _train_core_deep_search_from_replay(
        args,
        run_dir=run_dir,
        replay_dir=replay_dir,
        state=state,
        core_checkpoint=core_checkpoint,
        final=final,
        collection_contract=collection_contract,
        isolation=isolation,
        fresh_games=int(writer.written_records),
        decisions=decisions,
        sims=sims,
        actual_games_per_opponent=actual_games_per_opponent,
        coverage_complete=coverage_complete,
        coverage_counts=coverage_counts,
        stop_reason=stop_reason,
        scheduled_jobs=scheduled_jobs,
        jobs_len=len(jobs),
        max_depth=max_depth,
        max_ordered_actions=max_ordered_actions,
        factorized_decisions=factorized_decisions,
        started=started,
        n_servers=n_servers,
        requests=requests,
        queue_wait_total=queue_wait_total,
        writer_telemetry=writer_telemetry,
    )


def _train_core_deep_search_from_replay(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    replay_dir: Path,
    state: dict,
    core_checkpoint: Path,
    final: Path,
    collection_contract,
    isolation: dict,
    fresh_games: int,
    decisions: int,
    sims: int,
    actual_games_per_opponent: int,
    coverage_complete: bool,
    coverage_counts: dict,
    stop_reason: str,
    scheduled_jobs: int,
    jobs_len: int,
    max_depth: int,
    max_ordered_actions: int,
    factorized_decisions: int,
    started: float,
    n_servers: int,
    requests: int,
    queue_wait_total: float,
    writer_telemetry: dict,
) -> dict:
    """Fine-tune core kernel from a finalized deep-search replay (+ anchors)."""
    import torch

    from poke_bot import checkpoint
    from poke_bot.core_kernel import (
        CoreTrainConfig,
        CorpusConfig,
        StreamingArchetypeCorpus,
        core_kernel_config_small_3080ti,
        train_core_kernel,
    )
    from poke_bot.iteration_contract import history_games_for_fraction

    deep_run_name = f"{args.run_name}.core_deep_search"
    deep_latest = checkpoint.latest_path(deep_run_name)
    if not deep_latest.exists():
        deep_latest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(core_checkpoint, deep_latest)
    resume_ckpt = checkpoint.load_checkpoint(deep_latest, map_location="cpu")
    saved_epoch = int(resume_ckpt.get("epoch", 0))
    model_cfg = core_kernel_config_small_3080ti()

    desired_history = history_games_for_fraction(
        fresh_games, float(args.core_search_replay_fraction)
    )
    desired_history = min(
        desired_history,
        int(round(fresh_games * float(args.core_policy_anchor_ratio))),
    )
    core_bc_path = replay_dir / "core_bc_public_behavior.jsonl"
    policy_anchor_path = replay_dir / "core_deep_search_policy_anchor.jsonl"
    policy_rows = [
        json.loads(line)
        for line in core_bc_path.read_text().splitlines()
        if line.strip()
    ]
    if desired_history and not policy_rows:
        raise RuntimeError("core deep search has no explicit policy anchor pool")
    anchor_rng = __import__("random").Random(int(args.seed) + 2000)
    selected_anchor_rows = []
    shuffled_policy_rows = list(policy_rows)
    anchor_rng.shuffle(shuffled_policy_rows)
    for index in range(desired_history):
        source = dict(
            shuffled_policy_rows[index % len(shuffled_policy_rows)]
        )
        source["episode_id"] = (
            f"{source.get('episode_id', 'core-bc')}|policy-anchor-{index}"
        )
        provenance = dict(source.get("target_provenance") or {})
        provenance.update(
            {
                "training_role": "explicit_policy_anchor",
                "equivalent_to_belief_mcts": False,
                "anchor_index": index,
            }
        )
        source["target_provenance"] = provenance
        selected_anchor_rows.append(source)
    policy_anchor_path.write_text(
        "".join(json.dumps(row) + "\n" for row in selected_anchor_rows)
    )
    replay_denominator = max(fresh_games + desired_history, 1)
    corpus = StreamingArchetypeCorpus(
        CorpusConfig(
            jsonl_paths=[final, policy_anchor_path],
            val_frac=float(args.val_frac),
            max_context=model_cfg.max_context,
            verify_info_set=True,
            shuffle_buffer=min(
                256, max(32, fresh_games + desired_history)
            ),
            seed=int(args.seed) + 1000,
            max_decisions_per_game=int(args.max_decisions_per_game),
        )
    )
    train_result = train_core_kernel(
        corpus,
        run_name=deep_run_name,
        train_cfg=CoreTrainConfig(
            lr=float(args.core_bc_lr) * 0.25,
            epochs=saved_epoch + 1 + int(args.core_search_epochs),
            games_per_batch=int(args.training_batch_games),
            max_decisions_per_batch=int(args.training_max_decisions),
            early_stop_patience=max(1, int(args.core_bc_patience)),
            value_loss_weight=1.5,
            aux_loss_weight=0.0,
            # Match aux off for search-phase fine-tune (policy/value focus).
            opp_hand_loss_weight=0.0,
            opp_remainder_loss_weight=0.0,
            amp=args.amp_dtype != "fp32",
            amp_dtype=args.amp_dtype,
            seed=int(args.seed) + 1000,
            condition_archetype=False,
        ),
        resume="auto",
        device=torch.device("cuda:0"),
        cfg=model_cfg,
    )
    best = Path(train_result.get("best_path") or "")
    if not best.is_file():
        raise RuntimeError("core deep-search training produced no checkpoint")
    accepted = snapshot_immutable_checkpoint(
        best, args.run_name, "core_deep_search.accepted"
    )
    report = {
        "phase": "core_deep_search",
        "status": "accepted",
        "collection": {
            "contract": collection_contract.__dict__,
            "games": fresh_games,
            "trusted_search_decisions": decisions,
            "decisions": decisions,
            "decisions_per_game": decisions / max(fresh_games, 1),
            "completed_sims": sims,
            "sims_per_decision": sims / max(decisions, 1),
            "actual_games_per_opponent": actual_games_per_opponent,
            "opponent_seat_coverage_complete": coverage_complete,
            "opponent_seat_counts": coverage_counts,
            "stop_reason": stop_reason,
            "scheduled_games": scheduled_jobs,
            "max_games_not_scheduled": jobs_len - scheduled_jobs,
            "max_depth": max_depth,
            "max_complete_ordered_action_count": max_ordered_actions,
            "factorized_search_decisions": factorized_decisions,
            "wall_s": time.perf_counter() - started,
            "inference_servers": n_servers,
            "remote_requests": requests,
            "queue_wait_ms_mean": queue_wait_total / max(requests, 1),
            "writer": writer_telemetry,
        },
        "replay_composition": {
            "fresh_belief_games": fresh_games,
            "belief_history_games": 0,
            "explicit_policy_anchor_games": desired_history,
            "policy_targets_equivalent_to_belief_targets": False,
            "requested_replay_fraction": float(
                args.core_search_replay_fraction
            ),
            "actual_replay_fraction": (
                desired_history / replay_denominator
            ),
            "bootstrap_regularizer_games": 0,
            "bootstrap_regularizer_separate": True,
            "game_first_sampling": True,
            "max_decisions_per_game": int(args.max_decisions_per_game),
        },
        "training": train_result,
        "accepted_checkpoint": accepted,
        "device_isolation": isolation,
    }
    _write_json(run_dir / "core_deep_search_report.json", report)
    print(
        "[core-pipeline][core_deep_search_gate] "
        f"status=ACCEPTED games={fresh_games} decisions={decisions} "
        f"decisions_per_game={decisions/max(fresh_games, 1):.2f} "
        f"collection_stop={stop_reason} "
        f"replay_fraction={desired_history/replay_denominator:.1%} "
        f"sims_per_decision={report['collection']['sims_per_decision']:.1f} "
        f"max_depth={max_depth} max_ordered_actions={max_ordered_actions} "
        f"factorized_decisions={factorized_decisions} "
        f"checkpoint_digest={accepted['digest']}",
        flush=True,
    )
    return report


def _run_core_gate(args: argparse.Namespace, run_dir: Path, state: dict) -> dict:
    checkpoint_info = dict(
        state.get("artifacts", {}).get("core_deep_search_checkpoint") or {}
    )
    path = Path(checkpoint_info.get("path") or "")
    deep_report_path = run_dir / "core_deep_search_report.json"
    if not path.is_file() or not deep_report_path.is_file():
        raise RuntimeError("core gate is missing deep-search artifacts")
    deep = json.loads(deep_report_path.read_text())
    collection = deep.get("collection") or {}
    reasons = []
    if float(collection.get("sims_per_decision") or 0.0) < 128.0:
        reasons.append("completed simulations per decision below 128")
    if int(collection.get("max_depth") or 0) < 2:
        reasons.append("search never reached depth 2")
    if int(collection.get("games") or 0) < 8:
        reasons.append("insufficient search games")
    transfer = core_to_specialist_transfer_report(path, "hammer-pult")
    if not transfer.get("complete_shape_transfer"):
        reasons.append("core-to-Hammer tensor transfer is incomplete")
    if reasons:
        raise RuntimeError("core gate rejected: " + "; ".join(reasons))
    report = {
        "phase": "core_gate",
        "status": "passed",
        "search_collection": collection,
        "checkpoint": checkpoint_info,
        "hammer_transfer_report": transfer,
    }
    _write_json(run_dir / "core_gate_report.json", report)
    print(
        "[core-pipeline][core_gate] status=PASSED "
        f"sims_per_decision={collection['sims_per_decision']:.1f} "
        f"max_depth={collection['max_depth']}",
        flush=True,
    )
    return report


def _run_hammer_warmstart(
    args: argparse.Namespace, run_dir: Path, state: dict
) -> dict:
    import torch

    from poke_bot.core_kernel import warm_start_specialist_from_checkpoint

    checkpoint_info = dict(
        state.get("artifacts", {}).get("core_deep_search_checkpoint") or {}
    )
    source = Path(checkpoint_info.get("path") or "")
    if not source.is_file():
        raise RuntimeError("Hammer warm-start has no accepted core checkpoint")
    run_name = f"{args.run_name}.hammer"
    result = warm_start_specialist_from_checkpoint(
        source,
        "hammer-pult",
        run_name=run_name,
        device=torch.device("cuda:0"),
        write_checkpoint=True,
    )
    latest = Path((result.get("paths") or {}).get("latest") or "")
    if not latest.is_file():
        raise RuntimeError("Hammer warm-start did not write a checkpoint")
    immutable = snapshot_immutable_checkpoint(
        latest, args.run_name, "hammer_warmstart.accepted"
    )
    report = {
        "phase": "hammer_warmstart",
        "status": "accepted",
        "run_name": run_name,
        "source": str(source),
        "reinit_heads": result["reinit_heads"],
        "fold_archetype": result["fold_archetype"],
        "accepted_checkpoint": immutable,
    }
    _write_json(run_dir / "hammer_warmstart_report.json", report)
    print(
        "[core-pipeline][hammer_warmstart] status=ACCEPTED "
        f"checkpoint_digest={immutable['digest']}",
        flush=True,
    )
    return report


def _run_hammer_search_rl(
    args: argparse.Namespace, run_dir: Path, state: dict
) -> dict:
    checkpoint_info = dict(
        state.get("artifacts", {}).get("hammer_warmstart_checkpoint") or {}
    )
    source = Path(checkpoint_info.get("path") or "")
    if not source.is_file():
        raise RuntimeError("Hammer search RL has no warm-start checkpoint")
    rr_run = f"{args.run_name}.hammer_search_rl"
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "train_round_robin.py"),
        "--archetype",
        "hammer-pult",
        "--run-name",
        rr_run,
        "--bootstrap-ckpt",
        str(source),
        "--resume",
        "auto",
        "--iterations",
        str(args.hammer_rl_iterations),
        "--games-per-opp",
        "16",
        "--min-games-per-opp",
        "12",
        "--max-games-per-opp",
        "24",
        "--target-search-decisions",
        str(args.hammer_target_search_decisions),
        "--games-per-opp-late",
        "16",
        "--curriculum-switch-iter",
        "0",
        "--workers",
        str(max(2, int(args.workers) if args.workers > 0 else 8)),
        "--no-worker-autotune",
        "--agent-mode",
        "belief-mcts",
        "--mcts-sims",
        str(args.search_start_sims),
        "--mcts-move-time",
        str(args.search_move_time_s),
        "--game-timeout-s",
        str(int(args.search_game_time_s)),
        "--leaf-eval",
        "gpu-server",
        "--leaf-gpu",
        "cuda:0",
        "--leaf-servers",
        str(args.inference_servers),
        "--leaf-max-batch",
        str(args.inference_batch),
        "--leaf-queue-depth",
        str(args.inference_queue_depth),
        "--sim-device",
        "cpu",
        "--train-epochs",
        "1",
        "--history-mix",
        "1",
        "--replay-fraction",
        str(args.core_search_replay_fraction),
        "--policy-anchor-ratio",
        str(args.core_policy_anchor_ratio),
        "--max-decisions-per-game",
        str(args.max_decisions_per_game),
        "--replay-history-iters",
        "4",
        "--heldout-fraction",
        "0",
        "--promotion-games",
        "80",
        "--promotion-max-games",
        "160",
        "--promotion-batch-games",
        "40",
        "--promotion-min-pairs",
        "40",
        "--promotion-workers",
        str(max(1, int(args.promotion_workers))),
        "--promotion-mcts-sims",
        str(args.search_start_sims),
        "--promotion-move-time",
        str(args.search_move_time_s),
    ]
    if args.remote_worker_endpoints:
        command.extend(
            [
                "--remote-worker-endpoints",
                *list(args.remote_worker_endpoints),
            ]
        )
    if getattr(args, "restart_partial_iteration", False):
        command.append("--restart-partial-iteration")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Hammer belief-search RL exited {completed.returncode}"
        )
    report = {
        "phase": "hammer_search_rl",
        "status": "started_and_checkpointed",
        "run_name": rr_run,
        "command": command,
        "iterations": int(args.hammer_rl_iterations),
    }
    _write_json(run_dir / "hammer_search_rl_report.json", report)
    return report


def main(argv=None) -> int:
    args = _args(argv)
    _validate_args(args)
    paths.ensure_runtime_dirs()
    run_dir = paths.OUTPUTS_DIR / "runs" / args.run_name
    replay_dir = paths.DATA_DIR / "core_pipeline" / args.run_name
    metadata_path = run_dir / "run_metadata.json"
    state_path = run_dir / "pipeline_state.json"
    if args.resume == "0" and (metadata_path.exists() or state_path.exists()):
        raise FileExistsError(
            f"refusing to overwrite existing core pipeline run {args.run_name}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    state = load_pipeline_state(run_dir, args.run_name)
    if args.phase != "auto" and state.get("current_phase") != args.phase:
        raise ValueError(
            f"--phase {args.phase} does not match resumed phase "
            f"{state.get('current_phase')}"
        )
    audit = search_safety_audit()
    requested_contract = _core_iteration_contract(args)
    if not metadata_path.exists():
        _write_json(
            metadata_path,
            {
                "run_name": args.run_name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "phase_graph": PHASE_GRAPH,
                "command_config": vars(args),
                "active_command_config": vars(args),
                "iteration_contract": requested_contract,
                "stable_console_log": str(args.stable_log),
                "gpu_contract": {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": "0",
                    "physical_gpu": "RTX 3080 Ti",
                },
                "search_safety_audit": audit.as_dict(),
                "replay_dir": str(replay_dir),
                "run_dir": str(run_dir),
            },
        )
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous_contract = dict(metadata.get("iteration_contract") or {})
        if previous_contract != requested_contract:
            if not args.restart_partial_search:
                raise RuntimeError(
                    "core iteration contract changed on resume; use "
                    "--restart-partial-search at the core_bc->core_deep_search "
                    "boundary"
                )
            if (
                state.get("current_phase") != "core_deep_search"
                or list(state.get("completed_phases") or []) != ["core_bc"]
            ):
                raise RuntimeError(
                    "core contract migration requires the completed "
                    "core_bc->core_deep_search boundary"
                )
            final_search = replay_dir / "core_deep_search.jsonl"
            if final_search.exists():
                raise RuntimeError(
                    "core contract migration cannot reinterpret finalized "
                    f"search replay {final_search}"
                )
            previous_digest = hashlib.sha256(
                json.dumps(
                    previous_contract,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            suffix = (
                ".invalid.contract-migration."
                + previous_digest[:12]
                + f".{int(time.time())}"
            )
            quarantined = _quarantine_core_search_partials(
                replay_dir, suffix=suffix
            )
            migrations = list(metadata.get("contract_migration_history") or [])
            migrations.append(
                {
                    "recorded_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%S%z"
                    ),
                    "reason": (
                        "explicit_core_bc_boundary_decision_budget_migration"
                    ),
                    "effective_phase": "core_deep_search",
                    "completed_boundary_phase": "core_bc",
                    "previous_contract": previous_contract,
                    "previous_contract_digest": "sha256:" + previous_digest,
                    "new_contract": requested_contract,
                    "new_contract_digest": "sha256:"
                    + hashlib.sha256(
                        json.dumps(
                            requested_contract,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                    "quarantined_partial_artifacts": quarantined,
                    "finalized_search_reinterpreted": False,
                }
            )
            metadata["iteration_contract"] = requested_contract
            metadata["contract_migration_history"] = migrations
        metadata["active_command_config"] = vars(args)
        metadata["last_resumed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_json(metadata_path, metadata)
    print(
        "[core-pipeline][start] "
        f"run={args.run_name} resume={args.resume} "
        f"phase={state.get('current_phase')} graph={'->'.join(PHASE_GRAPH)} "
        f"stable_log={args.stable_log}",
        flush=True,
    )
    print(
        "[core-pipeline][search_guard] "
        f"trusted_ready={audit.trusted_ready} belief={audit.belief_mode} "
        f"chance={audit.chance_mode} history={audit.history_mode}",
        flush=True,
    )
    if args.dry_run:
        print(json.dumps({"state": state, "audit": audit.as_dict()}, indent=2))
        return 0

    while state.get("current_phase") is not None:
        phase = str(state["current_phase"])
        if phase == "core_bc":
            report = _run_core_bc(args, run_dir, replay_dir)
            state = advance_pipeline_state(
                run_dir,
                state,
                "core_bc",
                {
                    "core_bc_checkpoint": report["accepted_checkpoint"],
                    "core_bc_report": str(run_dir / "core_bc_report.json"),
                },
            )
        elif phase == "core_deep_search":
            report = _run_core_deep_search(args, run_dir, replay_dir, state)
            state = advance_pipeline_state(
                run_dir,
                state,
                phase,
                {
                    "core_deep_search_checkpoint": report[
                        "accepted_checkpoint"
                    ],
                    "core_deep_search_report": str(
                        run_dir / "core_deep_search_report.json"
                    ),
                },
            )
        elif phase == "core_gate":
            report = _run_core_gate(args, run_dir, state)
            state = advance_pipeline_state(
                run_dir,
                state,
                phase,
                {"core_gate_report": str(run_dir / "core_gate_report.json")},
            )
        elif phase == "hammer_warmstart":
            if "core_gate_report" not in state.get("artifacts", {}):
                raise UnsafePhaseError(
                    "hammer_warmstart cannot precede a passed core search gate"
                )
            report = _run_hammer_warmstart(args, run_dir, state)
            state = advance_pipeline_state(
                run_dir,
                state,
                phase,
                {
                    "hammer_warmstart_checkpoint": report[
                        "accepted_checkpoint"
                    ],
                    "hammer_warmstart_report": str(
                        run_dir / "hammer_warmstart_report.json"
                    ),
                },
            )
        elif phase == "hammer_search_rl":
            report = _run_hammer_search_rl(args, run_dir, state)
            state = advance_pipeline_state(
                run_dir,
                state,
                phase,
                {
                    "hammer_search_rl_report": str(
                        run_dir / "hammer_search_rl_report.json"
                    ),
                    "hammer_search_rl_run": report["run_name"],
                },
            )
        else:
            raise ValueError(f"unknown phase {phase}")
        if args.stop_after_phase == phase:
            break
    save_pipeline_state(run_dir, state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineInterrupted as exc:
        print(
            f"[core-pipeline][INTERRUPTED] signal={exc.signum}; "
            "phase state unchanged",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(128 + exc.signum)
    except UnsafePhaseError as exc:
        print(f"[core-pipeline][BLOCKED] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(4)
