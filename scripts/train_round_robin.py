#!/usr/bin/env python
"""Trusted history-policy round-robin vs ALL manifest baselines.

Each iteration:
  1. Play seat-stratified games vs every manifest agent (26 after prune).
  2. Train an immutable candidate on current + immutable replay + bootstrap.
  3. Evaluate candidate vs incumbent with draw-aware, seat-stratified
     uncertainty and promote only when the configured gate passes.
  4. Checkpoint the evaluated incumbent and full loop state.

Curriculum (defaults): ``--games-per-opp 24`` until ``--curriculum-switch-iter 25``,
then ``--games-per-opp-late 100``. Leaf eval defaults to one GPU server
(``--leaf-eval gpu-server --leaf-gpu auto`` → same device as the train step,
usually Blackwell) while sim workers stay CPU-only. Formal submit gate prefers
≥100 games/opp (``eval_vs_baselines.py``).

Progress: outer tqdm over iterations, inner over games; running WR postfix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tqdm.auto import tqdm

from poke_bot import checkpoint, config, deck_pool, features, paths
from poke_bot.agent import PolicyAgent, play_game
from poke_bot.baselines_runtime import (
    BaselineSpec,
    ensure_baselines_installed,
    filter_loadable_baselines,
    load_baseline_agent,
    load_manifest,
)
from poke_bot.dataset import BootstrapDataset, DecisionSample, GameSequence, load_bootstrap_dataset
from poke_bot.device import leaf_eval_device, training_device
from poke_bot.eval_metrics import FieldReport
from poke_bot.features import build_board_tokens, build_option_tokens, enumerate_action_combos
from poke_bot.train import (
    TrainConfig,
    load_model_from_checkpoint,
    rl_train_step,
    sequence_losses,
    train_bootstrap,
)
from poke_bot.worker_pool import WorkerPool, WorkerPoolStopped
from poke_bot.promotion import (
    CheckpointIdentity,
    PromotionGateConfig,
    evaluate_candidate_gate,
    next_iteration,
)
from poke_bot.replay_writer import OrderedReplayWriter
from poke_bot.iteration_contract import (
    PILOT_DERIVATION,
    PILOT_DERIVED_TARGET_SEARCH_DECISIONS,
)


class CollectionInterrupted(RuntimeError):
    """Unwind an active worker pool outside the async signal handler."""


# Module-level cache for spawn workers (reloaded after recycle).
_WORKER_STATE: dict[str, Any] = {}


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _source_tree_digest() -> str:
    """Hash runtime source bytes, including dirty/untracked Python edits."""
    h = hashlib.sha256()
    roots = [ROOT / "poke_bot", ROOT / "scripts", ROOT / "submission"]
    for base in roots:
        for path in sorted(base.rglob("*.py")):
            h.update(str(path.relative_to(ROOT)).encode())
            h.update(b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
    return "sha256:" + h.hexdigest()


def _simulator_digest() -> str:
    """Best-effort digest of the native simulator artifact used for targets."""
    try:
        root = paths.cg_runtime_dir()
        candidates = sorted(root.rglob("libcg.so"))
        if not candidates:
            return "unavailable"
        h = hashlib.sha256()
        for path in candidates:
            h.update(path.read_bytes())
        return "sha256:" + h.hexdigest()
    except Exception:
        return "unavailable"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--archetype",
        default=deck_pool.primary_archetype(),
        help="Primary archetype (drives run names, deck, bootstrap JSONL).",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Unique checkpoint/replay lineage. Default preserves the historical "
        "<archetype>_round_robin naming; unattended launches should set this.",
    )
    p.add_argument(
        "--replay-lineage",
        default=None,
        help="Replay directory label. Use a new explicit label when migrating "
        "a checkpoint lineage from policy games to trusted decision budgets.",
    )
    p.add_argument(
        "--policy-anchor-lineage",
        default=None,
        help="Optional legacy policy replay lineage used only as a labeled "
        "anchor; its targets are never treated as belief-MCTS targets.",
    )
    p.add_argument(
        "--bootstrap-ckpt",
        type=Path,
        default=None,
        help="Starting weights (default: <archetype>_bootstrap.best.pt or latest)",
    )
    p.add_argument("--resume", default="auto")
    p.add_argument(
        "--restart-partial-iteration",
        action="store_true",
        help="Quarantine crash-recovery partials for the current iteration and "
        "recollect it from job zero.",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=10_000,
        help="Safety cap on round-robin iterations (default 10000). Primary stop "
        "is the draw-aware field gate, not reaching N.",
    )
    p.add_argument(
        "--games-per-opp",
        type=int,
        default=16,
        help="Nominal fresh games/opponent (default 16, balanced seats). "
        "Belief-MCTS collection may stop/extend within min/max safeguards.",
    )
    p.add_argument(
        "--games-per-opp-late",
        type=int,
        default=16,
        help="Legacy policy curriculum value; belief-MCTS uses the nominal "
        "--games-per-opp decision-budget contract.",
    )
    p.add_argument(
        "--curriculum-switch-iter",
        type=int,
        default=0,
        help="0-indexed iteration at which games/opp bumps from --games-per-opp "
        "to --games-per-opp-late (default 25 → iters 1-25 use 24, iter 26+ use "
        "100).",
    )
    p.add_argument("--min-games-per-opp", type=int, default=12)
    p.add_argument("--max-games-per-opp", type=int, default=24)
    p.add_argument(
        "--target-search-decisions",
        type=int,
        default=PILOT_DERIVED_TARGET_SEARCH_DECISIONS,
        help="Trusted fresh search-decision target (default 3664, derived from "
        "the recorded native 52-game full-roster pilot). 0 stops at nominal games.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Concurrent in-flight games (worker processes). 0 = auto: "
        "config.HARDWARE.rl_games_in_flight (oversubscribed to peg 32 CPU "
        "threads) when the GPU leaf server is active, else sim_workers. Capped "
        "by available RAM (per_worker_rss_gb).",
    )
    p.add_argument(
        "--worker-autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt active belief-MCTS game workers between balanced waves "
        "using inference occupancy and queue pressure.",
    )
    p.add_argument(
        "--mcts-sims",
        type=int,
        default=128,
        help="Completed simulations per decision in MCTS modes (trusted belief "
        "mode requires >=128). Ignored in policy mode.",
    )
    p.add_argument(
        "--mcts-move-time",
        type=float,
        default=8.0,
        help="Per-decision belief-search wall-clock cap in seconds.",
    )
    p.add_argument(
        "--game-timeout-s",
        type=int,
        default=600,
        help="Per-game watchdog; the search clock reserves headroom below it.",
    )
    p.add_argument(
        "--expected-search-decisions",
        type=int,
        default=64,
        help="Public-game clock planning horizon for adaptive move budgets.",
    )
    p.add_argument(
        "--agent-mode",
        choices=("policy", "belief-mcts", "oracle-mcts"),
        default="policy",
        help="belief-mcts is trusted public-history particle search; "
        "oracle-mcts remains diagnostic-only and cannot train/promote/deploy.",
    )
    p.add_argument("--gate", type=float, default=0.55)
    p.add_argument(
        "--train-epochs",
        type=int,
        default=1,
        help="Train-epoch cap on game-first population replay each "
        "iteration (default 1). "
        "Higher caps use early-stopping on val loss (see --patience).",
    )
    p.add_argument(
        "--train-lr",
        type=float,
        default=5e-5,
        help="Candidate learning rate (default 5e-5, deliberately lower than "
        "bootstrap to limit first-update damage).",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stop patience on RL train val-loss each cycle "
        "(default 5; same pattern as train_bootstrap). 0 disables early stop.",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Validation fraction for RL train early-stopping (default 0.1).",
    )
    p.add_argument(
        "--aux-loss-weight",
        type=float,
        default=0.1,
        help="Archetype aux_head CE weight during RL train (default 0.1).",
    )
    p.add_argument(
        "--opp-hand-loss-weight",
        type=float,
        default=0.2,
        help="opp_hand_head BCE weight (default 0.2; masked when labels absent).",
    )
    p.add_argument(
        "--opp-remainder-loss-weight",
        type=float,
        default=0.15,
        help="opp_remainder_head BCE weight (default 0.15; masked when absent).",
    )
    p.add_argument(
        "--lethal-threat-loss-weight",
        type=float,
        default=0.15,
        help=(
            "Scope B lethal_threat_head BCE weight (default 0.15 for Blackwell "
            "Hammer RR; masked when labels absent). Use 0 to disable."
        ),
    )
    p.add_argument(
        "--prize-race-loss-weight",
        type=float,
        default=0.10,
        help=(
            "Scope B prize_race_head SmoothL1 weight (default 0.10 for Blackwell "
            "Hammer RR; masked when absent). Prize-map/KO-path scaffold."
        ),
    )
    p.add_argument(
        "--bootstrap-mix",
        type=float,
        default=0.25,
        help="Bootstrap anchor as a fraction of current games (default 0.25).",
    )
    p.add_argument(
        "--history-mix",
        type=float,
        default=1.0,
        help="Legacy replay/current game ratio (kept for old policy resumes).",
    )
    p.add_argument(
        "--replay-fraction",
        type=float,
        default=0.50,
        help="Old replay share of fresh+old game mixture (0.50-0.75).",
    )
    p.add_argument(
        "--policy-anchor-ratio",
        type=float,
        default=1.0,
        help="Maximum old policy games per fresh game, used only when trusted "
        "belief replay cannot fill the requested replay fraction.",
    )
    p.add_argument(
        "--max-decisions-per-game",
        type=int,
        default=32,
        help="Chronological decision window cap after uniform game sampling.",
    )
    p.add_argument(
        "--replay-history-iters",
        type=int,
        default=4,
        help="How many completed iteration replay files form the history pool.",
    )
    p.add_argument(
        "--heldout-fraction",
        type=float,
        default=0.0,
        help="Legacy in-field holdout fraction. Default 0 because promotion is "
        "a separate candidate-vs-incumbent sample.",
    )
    p.add_argument(
        "--regression-margin",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--promotion-games", type=int, default=80)
    p.add_argument("--promotion-max-games", type=int, default=160)
    p.add_argument("--promotion-batch-games", type=int, default=20)
    p.add_argument("--promotion-min-pairs", type=int, default=40)
    p.add_argument("--promotion-threshold", type=float, default=0.5)
    p.add_argument("--promotion-confidence", type=float, default=0.90)
    p.add_argument("--promotion-bootstrap-resamples", type=int, default=4000)
    p.add_argument("--promotion-wilson-z", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument(
        "--promotion-mcts-sims",
        type=int,
        default=128,
        help="Identical search simulations for candidate and incumbent.",
    )
    p.add_argument(
        "--promotion-move-time",
        type=float,
        default=8.0,
        help="Identical per-move search deadline for candidate and incumbent.",
    )
    p.add_argument(
        "--promotion-workers",
        type=int,
        default=8,
        help="CPU workers for independent seat-stratified promotion games.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", nargs="+")
    p.add_argument(
        "--sim-device",
        default="cpu",
        help="Device for per-worker game rollouts (default: cpu). CPU-only "
        "avoids GPU oversubscription: with N workers each creating a CUDA "
        "context, a 12 GB card OOMs. Baselines always run here (CPU).",
    )
    p.add_argument(
        "--leaf-eval",
        choices=("gpu-server", "cpu"),
        default="gpu-server",
        help="Where OUR agent's MCTS leaf network eval runs. 'gpu-server' "
        "(default) starts ONE persistent GPU-resident inference server that "
        "batches leaves across all concurrent CPU games (fills the GPU during "
        "the game phase). 'cpu' runs the net locally in each CPU worker (slow, "
        "idle GPU — legacy behavior).",
    )
    p.add_argument(
        "--leaf-gpu",
        default="auto",
        help="CUDA device for the persistent leaf-eval server(s). 'auto' "
        "(default) pins them to the SAME Blackwell (GPU1) the train step uses, "
        "so the whole primary pipeline lives on the 48 GB Blackwell and the 3080 "
        "Ti (GPU0) is left free for the core kernel / Phase 6 specialists.",
    )
    p.add_argument(
        "--leaf-servers",
        type=int,
        default=2,
        help="Concurrent GPU eval-server replicas on the Blackwell (default from "
        "2; fewer replicas preserve cross-game batch occupancy). Each keeps a champion replica resident with its own CUDA "
        "context/stream so forwards overlap → higher game-phase util. CPU workers "
        "are sharded across replicas. ~1-2 GB each (bf16) → fits under 40 GB.",
    )
    p.add_argument(
        "--leaf-coalesce-ms",
        type=float,
        default=config.HARDWARE.leaf_server_coalesce_ms,
        help="Server coalesce wait window (ms): fill big leaf batches per forward.",
    )
    p.add_argument(
        "--leaf-max-batch",
        type=int,
        default=config.HARDWARE.leaf_server_max_batch,
        help="Max leaves coalesced into one GPU forward (VRAM-bounded; OomGuard "
        "shrinks further on an actual CUDA OOM).",
    )
    p.add_argument(
        "--leaf-queue-depth",
        type=int,
        default=128,
        help="Bounded requests per inference-server queue.",
    )
    p.add_argument(
        "--agent-verbose",
        action="store_true",
        help="Let baseline/agent print() debug output through to the log "
        "(default: suppressed so the log shows only the tqdm WR bar; also "
        "settable via POKEBOT_AGENT_VERBOSE=1).",
    )
    return p.parse_args(argv)


def _resolve_bootstrap_ckpt(explicit: Path | None, archetype: str) -> Path:
    if explicit is not None:
        return explicit
    best = checkpoint.best_path(f"{archetype}_bootstrap")
    latest = checkpoint.latest_path(f"{archetype}_bootstrap")
    if best.is_file():
        return best
    if latest.is_file():
        return latest
    raise FileNotFoundError(
        "No bootstrap checkpoint found. Run scripts/train_bootstrap.py first "
        f"(looked for {best} / {latest})."
    )


_GAME_TIMEOUT_S = int(os.environ.get("POKEBOT_GAME_TIMEOUT_S", "180"))


def _classify_error(error: str | None) -> str:
    """Classify a failure without blaming an opponent for trusted-search faults.

    Resource / timeout errors are INFRA problems (CUDA/RAM OOM, watchdog
    timeout), never a baseline code fault — they must not blacklist a baseline.
    Only 'code' faults (exceptions, illegal moves) count for skip/removal.
    """
    t = (error or "").lower()
    if "remoteleafcancelled" in t or "workerpoolstopped" in t:
        return "cancelled"
    if (
        "beliefsupporterror" in t
        or "trustedsearchbudgetexhausted" in t
        or "insufficient trusted belief simulations" in t
    ):
        return "trust"
    if (
        "outofmemory" in t
        or "out of memory" in t
        or "cuda error" in t
        or "cublas" in t
        or "cudnn" in t
        or "memoryerror" in t
    ):
        return "resource"
    if "timeouterror" in t or "timed out" in t or "exceeded" in t:
        return "timeout"
    return "code"


def _entry_is_resource(entry: object) -> bool:
    """True if a persisted broken-baseline entry was caused by infra/resource.

    Used to self-heal the blacklist: OOM/timeout entries from a prior
    oversubscribed run are dropped so those (working, rule-based) baselines are
    restored to the field. Genuine code faults (import errors, exceptions) stay.
    """
    if isinstance(entry, dict):
        kind = str(entry.get("kind", "")).lower()
        if kind in ("resource", "timeout"):
            return True
        err = str(entry.get("error", ""))
    else:
        err = str(entry)
    return _classify_error(err) in ("resource", "timeout")


def _worker_play(job: dict) -> dict:
    """Play one trusted policy game in a worker; return outcome + targets.

    Never raises: any failure of the OPPONENT baseline (exception / illegal
    move / timeout / load error) is reported as ``baseline_failed`` so the
    parent can blacklist that agent. Our own agent already fails closed.
    """
    import signal

    from poke_bot import config as _config
    from poke_bot.agent import install_quiet_stdout

    # Once per worker: silence baseline/libcg stdout unless --agent-verbose.
    if (
        not job.get("preserve_stdout")
        and not _WORKER_STATE.get("stdout_done")
    ):
        install_quiet_stdout(_config.agent_verbose())
        _WORKER_STATE["stdout_done"] = True

    opp_id = job["spec"]["id"]
    our_seat = int(job["our_seat"])
    our_deck = job["our_deck"]

    def _base_result(**over) -> dict:
        r = {
            "job_index": int(job["job_index"]),
            "opponent_id": opp_id,
            "our_seat": our_seat,
            "winner": 2,
            "value": 0.0,
            "steps": 0,
            "is_mirror": False,
            "mcts_on": True,  # formal-path compatibility name in FieldReport
            "targets": [],
            "seed": job["seed"],
            "pair_id": job.get("pair_id"),
            "baseline_failed": False,
            "our_failed": False,
            "resource_error": False,
            "trust_failure": False,
            "search_budget_exhausted": False,
            "game_timeout": False,
            "cancelled": False,
            "training_eligible": bool(job.get("training_eligible", True)),
            "record_json": None,
            "error": None,
        }
        r.update(over)
        return r

    def _fail(error: str, *, our: bool = False) -> dict:
        # Classify: infra/resource + timeout errors are RETRYABLE and must NOT
        # blacklist the baseline. Only genuine code faults forfeit the game.
        kind = _classify_error(error)
        if kind == "resource":
            return _base_result(resource_error=True, error=f"[{kind}] {error}")
        if kind == "timeout":
            return _base_result(
                resource_error=True,
                game_timeout=True,
                error=f"[{kind}] {error}",
            )
        if kind == "trust":
            return _base_result(
                trust_failure=True,
                search_budget_exhausted=(
                    "TrustedSearchBudgetExhausted" in error
                ),
                error=f"[{kind}] {error}",
            )
        if kind == "cancelled":
            return _base_result(cancelled=True, error=f"[{kind}] {error}")
        if our:
            # Our agent faulted despite fail-closed → our loss (opponent wins).
            return _base_result(our_failed=True, winner=1 - our_seat, value=-1.0, error=error)
        # Baseline code fault → baseline forfeits (our seat wins).
        return _base_result(baseline_failed=True, winner=our_seat, value=1.0, error=error)

    try:
        import torch

        from poke_bot import batched_infer
        from poke_bot.agent import PolicyAgent, play_game
        from poke_bot.baselines_runtime import BaselineSpec, load_baseline_agent
        from poke_bot.train import load_model_from_checkpoint

        # Remote GPU leaf-eval server active? Then OUR net forwards run on the
        # server (this worker stays CPU-only and holds NO local model — it only
        # featurizes + searches on CPU). Otherwise fall back to a local model.
        leaf_backend = batched_infer.remote_leaf_backend_from_worker()
        if leaf_backend is not None:
            model = None
        else:
            ckpt = job["checkpoint"]
            device = job["device"]
            key = f"{ckpt}|{device}"
            if _WORKER_STATE.get("key") != key:
                _WORKER_STATE["model"] = load_model_from_checkpoint(
                    ckpt, device=torch.device(device)
                )
                _WORKER_STATE["key"] = key
            model = _WORKER_STATE["model"]

        spec_d = dict(job["spec"])
        spec_d["path"] = Path(spec_d["path"])
        spec = BaselineSpec(**spec_d)
        try:
            opp_fn, opp_deck = load_baseline_agent(spec)
        except Exception as exc:  # noqa: BLE001
            return _fail(f"load: {type(exc).__name__}: {exc}")

        rng = random.Random(int(job["seed"]))
        mcts_sims = int(job["mcts_sims"])
        game_timeout_s = int(job.get("game_timeout_s", _GAME_TIMEOUT_S))
        oracle = job.get("agent_mode") == "oracle-mcts"
        belief = job.get("agent_mode") == "belief-mcts"
        belief_posterior = None
        if belief:
            if "belief_posterior" not in _WORKER_STATE:
                from poke_bot.belief import EmpiricalDeckPosterior

                _WORKER_STATE["belief_posterior"] = (
                    EmpiricalDeckPosterior.from_manifest()
                )
            belief_posterior = _WORKER_STATE["belief_posterior"]
        agent = PolicyAgent(
            model=model,
            deck=our_deck,
            opponent_deck=opp_deck if oracle else None,
            use_mcts=oracle or belief,
            oracle_mode=oracle,
            belief_mcts=belief,
            belief_posterior=belief_posterior,
            checkpoint_digest=job.get("checkpoint_digest") if belief else None,
            model_generation=int(job.get("model_generation", 0)),
            max_context_override=job.get("model_max_context"),
            game_time_budget_s=float(game_timeout_s),
            game_watchdog_reserve_s=min(
                120.0, max(30.0, 0.2 * float(game_timeout_s))
            ),
            expected_search_decisions=int(
                job.get("expected_search_decisions", 64)
            ),
            max_sims=mcts_sims,
            move_time_s=float(job.get("mcts_move_time", 30.0)),
            collect_targets=bool(job.get("training_eligible", True)),
            sample_actions=not (oracle or belief),
            rng=rng,
            device=torch.device("cpu") if leaf_backend is not None else None,
            leaf_backend=leaf_backend,
            strict_runtime=True,
        )
        agent.reset_game()

        # Best-effort per-game timeout (spawn worker → SIGALRM on main thread).
        def _on_timeout(signum, frame):
            raise TimeoutError(f"game exceeded {game_timeout_s}s")

        had_alarm = hasattr(signal, "SIGALRM")
        if had_alarm:
            signal.signal(signal.SIGALRM, _on_timeout)
            signal.alarm(game_timeout_s)
        t0 = time.perf_counter()
        try:
            # Worker stdout is already redirected to devnull (see
            # install_quiet_stdout) unless --agent-verbose, so baseline prints
            # never reach the log.
            if our_seat == 0:
                result = play_game(agent, opp_fn, our_deck, opp_deck)
            else:
                result = play_game(opp_fn, agent, opp_deck, our_deck)
        finally:
            if had_alarm:
                signal.alarm(0)
        wall_s = time.perf_counter() - t0

        failed_seat = result.get("failed_seat")
        if failed_seat is not None:
            if failed_seat == our_seat:
                # Our agent failed despite fail-closed — count as our loss, do
                # NOT blacklist the opponent.
                res = _fail(f"our-agent: {result.get('error')}", our=True)
                res["steps"] = result.get("steps", 0)
                res["wall_s"] = wall_s
                res["is_mirror"] = sorted(our_deck) == sorted(opp_deck)
                return res
            return _fail(f"in-game: {result.get('error')}")
        if result.get("incomplete"):
            return _fail(
                f"game exceeded max steps ({result.get('steps', 0)})"
            )

        winner = result["winner"]
        value = 0.0 if winner == 2 else (1.0 if winner == our_seat else -1.0)
        # Aggregate optional experimental-search diagnostics.
        sims_run_total = 0
        sims_plan_peak = 0
        remote_requests = 0
        queue_wait_ms_total = 0.0
        inference_batch_total = 0.0
        batch_occupancy_total = 0.0
        request_queue_depth_max = 0
        server_request_rate_max = 0.0
        max_search_depth = 0
        max_complete_ordered_action_count = 0
        factorized_search_decisions = 0
        leaf_evaluations = 0
        for t in agent.targets:
            d = t.get("diagnostics") or {}
            sims_run_total += int(d.get("sims_run") or 0)
            sims_plan_peak = max(sims_plan_peak, int(d.get("sims_planned") or 0))
            requests = int(d.get("remote_requests") or 0)
            remote_requests += requests
            queue_wait_ms_total += float(d.get("queue_wait_ms_mean") or 0.0) * requests
            inference_batch_total += (
                float(d.get("inference_batch_size_mean") or 0.0) * requests
            )
            batch_occupancy_total += (
                float(d.get("batch_occupancy_mean") or 0.0) * requests
            )
            request_queue_depth_max = max(
                request_queue_depth_max,
                int(d.get("request_queue_depth_max") or 0),
            )
            server_request_rate_max = max(
                server_request_rate_max,
                float(d.get("server_request_rate") or 0.0),
            )
            max_search_depth = max(max_search_depth, int(d.get("max_depth") or 0))
            max_complete_ordered_action_count = max(
                max_complete_ordered_action_count,
                int(d.get("complete_ordered_action_count") or 0),
            )
            factorized_search_decisions += int(
                d.get("action_space_mode")
                == "exact_autoregressive_hierarchical"
            )
            leaf_evaluations += int(d.get("leaf_evaluations") or 0)
        # Build a trusted record from our deployment-visible acting-seat history,
        # behavior-policy distribution, and final outcome.
        record = None
        if job.get("training_eligible", True):
            record = _build_selfplay_record(
                agent.targets,
                our_deck=our_deck,
                our_seat=our_seat,
                value=value,
                opp_id=opp_id,
                archetype=job.get("archetype", "hammer-pult"),
                seed=int(job["seed"]),
                target_provenance=dict(job.get("target_provenance") or {}),
            )
        return {
            "job_index": int(job["job_index"]),
            "opponent_id": opp_id,
            "our_seat": our_seat,
            "winner": winner,
            "value": value,
            "steps": result["steps"],
            "wall_s": wall_s,
            "n_decisions": len(agent.targets),
            "fail_closed": int(agent.fail_closed_count),
            "mcts_sims_planned": (
                (sims_plan_peak or mcts_sims) if (oracle or belief) else 0
            ),
            "mcts_sims_run_total": sims_run_total,
            "remote_leaf_requests": remote_requests,
            "remote_queue_wait_ms_total": queue_wait_ms_total,
            "remote_inference_batch_total": inference_batch_total,
            "remote_batch_occupancy_total": batch_occupancy_total,
            "request_queue_depth_max": request_queue_depth_max,
            "server_request_rate_max": server_request_rate_max,
            "max_search_depth": max_search_depth,
            "max_complete_ordered_action_count": (
                max_complete_ordered_action_count
            ),
            "factorized_search_decisions": factorized_search_decisions,
            "leaf_evaluations": leaf_evaluations,
            "leaf_remote": leaf_backend is not None,
            "is_mirror": sorted(our_deck) == sorted(opp_deck),
            "mcts_on": True,
            "agent_mode": job.get("agent_mode", "policy"),
            "record": None,
            "record_json": (
                json.dumps(record, separators=(",", ":")) if record else None
            ),
            "training_eligible": bool(job.get("training_eligible", True)),
            "seed": job["seed"],
            "pair_id": job.get("pair_id"),
            "baseline_failed": False,
            "our_failed": False,
            "error": None,
        }
    except BaseException as exc:  # noqa: BLE001 - worker must never crash the pool
        # Baseline import/call failures are handled explicitly above or by
        # play_game's failed_seat. Any other worker exception belongs to our
        # model/runtime and must never blacklist the opponent.
        return _fail(f"worker: {type(exc).__name__}: {exc}", our=True)


def _worker_promotion(job: dict) -> dict:
    """Play one strict candidate-vs-parent game; any runtime fault invalidates it."""
    import signal

    from poke_bot import config as _config
    from poke_bot.agent import PolicyAgent, install_quiet_stdout, play_game
    from poke_bot.belief import EmpiricalDeckPosterior
    from poke_bot.train import load_model_from_checkpoint

    if (
        not job.get("preserve_stdout")
        and not _WORKER_STATE.get("stdout_done")
    ):
        install_quiet_stdout(_config.agent_verbose())
        _WORKER_STATE["stdout_done"] = True

    seed = int(job["seed"])
    candidate_seat = int(job["candidate_seat"])
    base = {
        "candidate_seat": candidate_seat,
        "pair_id": None,
        "cluster_id": job.get("cluster_id"),
        "seed": seed,
        "valid": False,
        "winner": 2,
        "steps": 0,
        "error": None,
    }
    try:
        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device(job.get("device", "cpu"))
        key = (
            f"promotion|{job['candidate_checkpoint']}|"
            f"{job['parent_checkpoint']}|{device}"
        )
        if _WORKER_STATE.get("promotion_key") != key:
            _WORKER_STATE["candidate_model"] = load_model_from_checkpoint(
                job["candidate_checkpoint"], device=device
            )
            _WORKER_STATE["parent_model"] = load_model_from_checkpoint(
                job["parent_checkpoint"], device=device
            )
            _WORKER_STATE["promotion_key"] = key
        deck = list(job["deck"])
        belief_mode = job.get("agent_mode") == "belief-mcts"
        posterior = (
            EmpiricalDeckPosterior.from_manifest(extra_deck_lists=[deck])
            if belief_mode
            else None
        )
        promotion_timeout_s = int(job.get("timeout_s", 600))
        watchdog_reserve = min(
            60.0, max(10.0, 0.1 * float(promotion_timeout_s))
        )
        candidate = PolicyAgent(
            model=_WORKER_STATE["candidate_model"],
            deck=deck,
            use_mcts=belief_mode,
            belief_mcts=belief_mode,
            belief_posterior=posterior,
            max_sims=int(job.get("mcts_sims", 0)),
            move_time_s=float(job.get("mcts_move_time", 0.0)),
            checkpoint_digest=job.get("candidate_digest"),
            model_generation=int(job.get("model_generation", 0)),
            max_context_override=int(job.get("model_max_context", 64)),
            game_time_budget_s=float(promotion_timeout_s),
            game_watchdog_reserve_s=watchdog_reserve,
            expected_search_decisions=int(
                job.get("expected_search_decisions", 64)
            ),
            collect_targets=belief_mode,
            rng=random.Random(seed ^ 0xC0FFEE),
            strict_runtime=True,
        )
        parent = PolicyAgent(
            model=_WORKER_STATE["parent_model"],
            deck=deck,
            use_mcts=belief_mode,
            belief_mcts=belief_mode,
            belief_posterior=posterior,
            max_sims=int(job.get("mcts_sims", 0)),
            move_time_s=float(job.get("mcts_move_time", 0.0)),
            checkpoint_digest=job.get("parent_digest"),
            model_generation=int(job.get("model_generation", 0)),
            max_context_override=int(job.get("model_max_context", 64)),
            game_time_budget_s=float(promotion_timeout_s),
            game_watchdog_reserve_s=watchdog_reserve,
            expected_search_decisions=int(
                job.get("expected_search_decisions", 64)
            ),
            collect_targets=belief_mode,
            rng=random.Random(seed ^ 0xFACE),
            strict_runtime=True,
        )
        candidate.reset_game()
        parent.reset_game()

        def _on_timeout(_signum, _frame):
            raise TimeoutError(
                f"promotion game exceeded {promotion_timeout_s}s"
            )

        had_alarm = hasattr(signal, "SIGALRM")
        if had_alarm:
            signal.signal(signal.SIGALRM, _on_timeout)
            signal.alarm(promotion_timeout_s)
        try:
            if candidate_seat == 0:
                result = play_game(candidate, parent, deck, deck)
            else:
                result = play_game(parent, candidate, deck, deck)
        finally:
            if had_alarm:
                signal.alarm(0)
        if result.get("failed_seat") is not None:
            return {
                **base,
                "error": (
                    f"seat {result['failed_seat']} runtime failure: "
                    f"{result.get('error')}"
                ),
                "steps": int(result.get("steps", 0)),
            }
        if result.get("incomplete"):
            return {
                **base,
                "error": f"promotion game incomplete after {result.get('steps', 0)} steps",
                "steps": int(result.get("steps", 0)),
            }
        return {
            **base,
            "valid": True,
            "winner": int(result["winner"]),
            "steps": int(result["steps"]),
            "candidate_search_decisions": len(candidate.targets),
            "parent_search_decisions": len(parent.targets),
            "mcts_sims": int(job.get("mcts_sims", 0)),
            "mcts_move_time": float(job.get("mcts_move_time", 0.0)),
            "agent_mode": job.get("agent_mode", "policy"),
        }
    except BaseException as exc:  # noqa: BLE001
        return {**base, "error": f"{type(exc).__name__}: {exc}"}


def _build_selfplay_record(
    targets: list[dict],
    *,
    our_deck: list[int],
    our_seat: int,
    value: float,
    opp_id: str,
    archetype: str,
    seed: int,
    target_provenance: dict[str, Any],
) -> dict | None:
    """Build one trusted history-policy training record.

    Produces the same JSONL schema the bootstrap/featurizer pipeline consumes
    (``poke_bot.dataset.record_to_sequence``): a per-seat game with ``steps``
    (each carrying the info-set ``observation`` + chosen ``action``),
    ``factorized_policy_targets`` (one complete next-option/STOP distribution
    per teacher-forced action prefix), and a single game ``value`` (final
    outcome from our seat, win=+1 / loss=-1 / draw=0).

    Info-set integrity: only our own observations are recorded — the opponent's
    turns are played by the baseline and never captured here.
    """
    if not targets:
        return None
    from poke_bot import features as _features
    target_sources = {
        str(
            target.get("target_source")
            or (target.get("diagnostics") or {}).get("target_source")
            or ""
        )
        for target in targets
    }
    if len(target_sources) != 1:
        raise ValueError("mixed target sources within one game")
    target_source = next(iter(target_sources))
    if target_source == "belief_mcts":
        from poke_bot.core_pipeline import validate_search_target_identity

        provenances = [dict(target.get("provenance") or {}) for target in targets]
        first_provenance = provenances[0]
        if any(provenance != first_provenance for provenance in provenances[1:]):
            raise ValueError("mixed model/belief/search provenance within game")
        incumbent = dict(target_provenance.get("incumbent_checkpoint") or {})
        validate_search_target_identity(
            first_provenance,
            [dict(target.get("diagnostics") or {}) for target in targets],
            expected_checkpoint_digest=str(incumbent.get("digest") or ""),
            expected_model_generation=int(
                target_provenance.get("model_generation", -1)
            ),
        )
        target_provenance = {
            **target_provenance,
            **first_provenance,
            "target_source": "belief_mcts",
            "trusted": True,
        }
    elif target_source != "history_policy":
        raise ValueError(f"untrusted target source {target_source!r}")

    steps: list[dict] = []
    factorized_policy_targets: list[list[dict]] = []
    for i, t in enumerate(targets):
        obs = t.get("observation")
        if obs is None:
            continue
        action = list(t.get("action") or [])
        canonical_stages = _features.factorized_teacher_forcing_stages(obs, action)
        recorded_stages = list(t.get("factorized_stages") or [])
        rows: list[dict] = []
        for stage_i, (combos, target_index) in enumerate(canonical_stages):
            recorded = (
                dict(recorded_stages[stage_i])
                if stage_i < len(recorded_stages)
                else {}
            )
            recorded_combos = [
                list(c) for c in (recorded.get("action_combos") or [])
            ]
            if recorded_combos and recorded_combos != combos:
                raise ValueError("factorized behavior/canonical ordering mismatch")
            pol = [float(p) for p in (recorded.get("policy") or [])]
            if not pol and len(combos) == 1:
                pol = [1.0]
            if len(pol) != len(combos) or sum(pol) <= 0.0:
                raise ValueError("missing or invalid factorized behavior policy")
            total = sum(pol)
            rows.append(
                {
                    "action_combos": [list(c) for c in combos],
                    "policy": [p / total for p in pol],
                    "selected_index": int(target_index),
                }
            )
        steps.append(
            {
                "observation": obs,
                "action": action,
                "env_step": i,
            }
        )
        factorized_policy_targets.append(rows)

    if not steps:
        return None
    return {
        "episode_id": f"rl-{archetype}-{opp_id}-{our_seat}-{seed}",
        "seat": int(our_seat),
        "archetype": archetype,
        "opp_archetype": opp_id,
        "deck": list(our_deck),
        "value": float(value),
        "steps": steps,
        "policy_targets": [None] * len(steps),
        "factorized_policy_targets": factorized_policy_targets,
        "info_set_ok": True,
        "source": (
            "trusted_belief_mcts_round_robin"
            if target_source == "belief_mcts"
            else "trusted_policy_round_robin"
        ),
        "target_provenance": dict(target_provenance),
    }


def _append_records_jsonl(path: Path, records: list[dict]) -> int:
    """Append self-play training records (one JSON game per line). Returns count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            if not rec:
                continue
            fh.write(json.dumps(rec) + "\n")
            n += 1
    return n


def _snapshot_best_champion(source: Path, run_name: str) -> Path:
    """Preserve an evaluated checkpoint at a digest-addressed immutable path."""
    source = source.expanduser().resolve()
    identity = CheckpointIdentity.from_path(source)
    digest_short = identity.digest.split(":", 1)[-1][:16]
    dest = (
        paths.CHECKPOINTS_DIR / f"{run_name}.evaluated.{digest_short}.pt"
    ).resolve()
    if source == dest:
        return dest
    if not source.is_file():
        raise FileNotFoundError(f"cannot snapshot missing champion: {source}")
    if dest.is_file():
        if checkpoint.checkpoint_digest(dest) != identity.digest:
            raise RuntimeError(f"immutable evaluated checkpoint collision: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".tmp.{os.getpid()}")
    try:
        shutil.copy2(source, tmp)
        try:
            os.link(tmp, dest)
        except FileExistsError:
            if checkpoint.checkpoint_digest(dest) != identity.digest:
                raise
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def _games_per_opp_for_iter(it: int, args) -> int:
    """Curriculum schedule: 0-indexed iters [0, switch) use --games-per-opp,
    iters [switch, ∞) use --games-per-opp-late.

    Default (switch=25): iterations 1-25 → 24 games/opp, iteration 26+ → 100.
    """
    return args.games_per_opp if it < args.curriculum_switch_iter else args.games_per_opp_late


def _iter_of_file(p: Path) -> int:
    """Parse the iteration index out of an ``iterNNN.jsonl`` replay filename."""
    return int(p.stem.replace("iter", ""))


def _contract_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _quarantine_iteration_partials(
    replay_lineage: str,
    iteration: int,
    *,
    suffix: str,
) -> list[str]:
    replay_dir = paths.DATA_DIR / "rl" / replay_lineage
    partial = replay_dir / f"iter{int(iteration):03d}.jsonl.partial"
    artifacts = (
        partial,
        partial.with_suffix(partial.suffix + ".journal"),
        partial.with_suffix(partial.suffix + ".writer.json"),
    )
    moved: list[str] = []
    for artifact in artifacts:
        if not artifact.exists():
            continue
        destination = artifact.with_name(artifact.name + suffix)
        os.replace(artifact, destination)
        moved.append(str(destination))
    return moved


def _validate_contract_migration_boundary(
    *,
    completed_iteration: int,
    effective_iteration: int,
    finalized_paths: list[Path],
) -> None:
    if int(effective_iteration) != int(completed_iteration) + 1:
        raise RuntimeError(
            "contract migration is not at a completed N+1 boundary: "
            f"completed={completed_iteration} effective={effective_iteration}"
        )
    existing = [str(path) for path in finalized_paths if path.exists()]
    if existing:
        raise RuntimeError(
            "contract migration would reinterpret finalized iteration(s): "
            + ", ".join(existing)
        )


def _load_recent_only_dataset(
    iter_replay_path: Path,
    current_iter: int,
    *,
    bootstrap_jsonl: Path | None,
    bootstrap_mix: float,
    replay_dir: Path,
    policy_anchor_dir: Path | None,
    history_mix: float,
    replay_history_iters: int,
    seed: int,
    expected_target_source: str = "history_policy",
    replay_fraction: float = 0.50,
    policy_anchor_ratio: float = 1.0,
    max_decisions_per_game: int = 32,
) -> tuple[BootstrapDataset, dict]:
    """Build current + immutable policy-history + bootstrap population replay."""
    # Invariant: the single source file IS the current iteration's file.
    assert iter_replay_path.name == f"iter{current_iter:03d}.jsonl", (
        f"current replay expected iter{current_iter:03d}.jsonl, "
        f"got {iter_replay_path.name}"
    )
    oldest_game_iter = _iter_of_file(iter_replay_path)
    newest_game_iter = oldest_game_iter
    assert oldest_game_iter == current_iter, (
        f"current replay mismatch: oldest_game_iter={oldest_game_iter} != "
        f"current_iter={current_iter}"
    )

    ds = load_bootstrap_dataset(
        iter_replay_path, max_games=0, use_cache=True, verify_info_set=True
    )
    recent_seqs: list[GameSequence] = list(ds.sequences)
    n_recent = len(recent_seqs)
    untrusted = [
        seq.episode_id
        for seq in recent_seqs
        if not seq.target_provenance.get("trusted")
        or seq.target_provenance.get("target_source") != expected_target_source
    ]
    if untrusted:
        raise ValueError(
            f"current replay contains {len(untrusted)} untrusted target record(s)"
        )

    seqs: list[GameSequence] = list(recent_seqs)
    history_pool: list[GameSequence] = []
    belief_history_pool: list[GameSequence] = []
    policy_history_pool: list[GameSequence] = []
    history_source_counts: Counter[str] = Counter()
    belief_prior_paths = [
        replay_dir / f"iter{idx:03d}.jsonl"
        for idx in range(max(0, current_iter - replay_history_iters), current_iter)
    ]
    policy_prior_paths = (
        [
            policy_anchor_dir / f"iter{idx:03d}.jsonl"
            for idx in range(
                max(0, current_iter - replay_history_iters), current_iter
            )
        ]
        if policy_anchor_dir is not None
        else []
    )
    prior_paths = list(
        dict.fromkeys(belief_prior_paths + policy_prior_paths)
    )
    for prior_path in prior_paths:
        if not prior_path.is_file():
            continue
        prior = load_bootstrap_dataset(
            prior_path, max_games=0, use_cache=True, verify_info_set=True
        )
        for seq in prior.sequences:
            source = str(seq.target_provenance.get("target_source") or "")
            if not seq.target_provenance.get("trusted"):
                continue
            if source not in ("history_policy", "belief_mcts"):
                continue
            history_pool.append(seq)
            history_source_counts[source] += 1
            if source == "belief_mcts":
                belief_history_pool.append(seq)
            else:
                policy_history_pool.append(seq)
    rng = random.Random(seed)
    from poke_bot.iteration_contract import (
        cap_game_decisions,
        game_first_sample,
        history_games_for_fraction,
    )

    if expected_target_source == "belief_mcts":
        desired_history = history_games_for_fraction(n_recent, replay_fraction)
        selected_belief = game_first_sample(
            belief_history_pool,
            count=desired_history,
            rng=rng,
        )
        remaining = desired_history - len(selected_belief)
        max_policy_anchor = int(round(n_recent * policy_anchor_ratio))
        selected_policy = game_first_sample(
            policy_history_pool,
            count=min(remaining, max_policy_anchor),
            rng=rng,
        )
        selected_history = selected_belief + selected_policy
    else:
        desired_history = max(0, int(round(n_recent * history_mix)))
        selected_belief = []
        selected_policy = game_first_sample(
            policy_history_pool,
            count=desired_history,
            rng=rng,
        )
        selected_history = list(selected_policy)
    n_history = len(selected_history)
    seqs.extend(selected_history)
    n_boot = 0
    if bootstrap_jsonl and bootstrap_jsonl.is_file() and bootstrap_mix > 0 and n_recent > 0:
        bds = load_bootstrap_dataset(bootstrap_jsonl, max_games=0, use_cache=True)
        n_boot = min(len(bds.sequences), max(1, int(n_recent * bootstrap_mix)))
        mix = list(bds.sequences)
        rng.shuffle(mix)
        seqs.extend(mix[:n_boot])

    raw_decisions = sum(len(seq.decisions) for seq in seqs)
    seqs = [
        cap_game_decisions(
            seq,
            max_decisions=max_decisions_per_game,
            rng=rng,
        )
        for seq in seqs
    ]
    capped_decisions = sum(len(seq.decisions) for seq in seqs)
    denominator = max(n_recent + n_history, 1)

    info = {
        "current_iter": current_iter,
        "oldest_game_iter": oldest_game_iter,
        "newest_game_iter": newest_game_iter,
        "n_recent_games": n_recent,
        "n_history_games": n_history,
        "n_belief_history_games": len(selected_belief),
        "n_policy_anchor_games": len(selected_policy),
        "requested_replay_fraction": float(replay_fraction),
        "actual_replay_fraction_fresh_plus_history": n_history / denominator,
        "policy_anchor_ratio_cap": float(policy_anchor_ratio),
        "history_pool_games": len(history_pool),
        "history_pool_by_target_source": dict(history_source_counts),
        "history_sampling_weight": float(history_mix),
        "current_target_source": expected_target_source,
        "history_iterations": [str(p) for p in prior_paths if p.is_file()],
        "n_bootstrap_regularizer": n_boot,
        "bootstrap_fraction_total_games": n_boot / max(len(seqs), 1),
        "fresh_fraction_total_games": n_recent / max(len(seqs), 1),
        "history_fraction_total_games": n_history / max(len(seqs), 1),
        "game_first_sampling": True,
        "max_decisions_per_game": int(max_decisions_per_game),
        "raw_decisions": raw_decisions,
        "capped_decisions": capped_decisions,
    }
    return BootstrapDataset(seqs), info


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths.ensure_runtime_dirs()
    stop_requested: dict[str, int | None] = {"signal": None}
    active_pool: dict[str, WorkerPool | None] = {"pool": None}

    def _request_boundary_stop(signum, _frame) -> None:  # noqa: ANN001
        # Cancel collection immediately. A partial writer remains recoverable;
        # an interrupted iteration is never trained, promoted, or checkpointed.
        if stop_requested["signal"] is None:
            stop_requested["signal"] = int(signum)
            pool = active_pool["pool"]
            print(
                f"[rr] received signal {signum}; aborting the current "
                "iteration without training/promotion/checkpoint",
                flush=True,
            )
            if pool is not None:
                # multiprocessing.Pool.terminate() takes locks and joins
                # helper threads, so it is not async-signal-safe.  Raising
                # here unwinds the ``with WorkerPool`` block; __exit__ then
                # performs the one termination from normal control flow.
                raise CollectionInterrupted(f"signal {int(signum)}")

    signal.signal(signal.SIGINT, _request_boundary_stop)
    signal.signal(signal.SIGTERM, _request_boundary_stop)
    # Full-hardware perf toggles (TF32 / cuDNN benchmark / thread pins).
    config.apply_runtime_perf()
    if (
        args.games_per_opp < 2
        or args.games_per_opp % 2
        or args.games_per_opp_late < 2
        or args.games_per_opp_late % 2
    ):
        print(
            "ERROR: games-per-opponent values must be even and >= 2 "
            "(balanced seat strata)",
            file=sys.stderr,
        )
        return 2
    if args.agent_mode == "belief-mcts":
        try:
            from poke_bot.iteration_contract import SearchCollectionContract

            SearchCollectionContract(
                opponents=1,
                nominal_games_per_opponent=int(args.games_per_opp),
                min_games_per_opponent=int(args.min_games_per_opp),
                max_games_per_opponent=int(args.max_games_per_opp),
                target_search_decisions=int(args.target_search_decisions),
            )
        except ValueError as exc:
            print(f"ERROR: invalid collection contract: {exc}", file=sys.stderr)
            return 2
    if (
        not 0.0 <= args.bootstrap_mix <= 1.0
        or not 0.0 <= args.history_mix <= 1.0
        or not 0.0 <= args.heldout_fraction < 1.0
    ):
        print("ERROR: mix fractions must be in range", file=sys.stderr)
        return 2
    if not 0.50 <= args.replay_fraction <= 0.75:
        print("ERROR: --replay-fraction must be in [0.50, 0.75]", file=sys.stderr)
        return 2
    if args.policy_anchor_ratio < 0 or args.max_decisions_per_game < 1:
        print(
            "ERROR: policy anchor must be non-negative and decision cap positive",
            file=sys.stderr,
        )
        return 2
    if (
        args.game_timeout_s < 60
        or args.mcts_move_time <= 0
        or args.expected_search_decisions < 1
    ):
        print(
            "ERROR: game timeout must be >=60s, move time positive, and "
            "expected search decisions positive",
            file=sys.stderr,
        )
        return 2
    if (
        args.promotion_games < 40
        or args.promotion_games % 2
        or args.promotion_max_games < args.promotion_games
        or args.promotion_max_games % 2
        or args.promotion_batch_games < 2
        or args.promotion_batch_games % 2
    ):
        print(
            "ERROR: promotion requires even min>=40, even max>=min, "
            "and an even batch size",
            file=sys.stderr,
        )
        return 2
    if args.replay_history_iters < 0:
        print("ERROR: --replay-history-iters must be >= 0", file=sys.stderr)
        return 2
    if args.agent_mode == "oracle-mcts" and args.train_epochs > 0:
        print(
            "ERROR: oracle-mcts is diagnostic-only and cannot generate trusted "
            "training/promotion targets; use --train-epochs 0",
            file=sys.stderr,
        )
        return 2
    if args.agent_mode == "belief-mcts" and args.mcts_sims < 128:
        print(
            "ERROR: trusted belief-mcts requires --mcts-sims >= 128",
            file=sys.stderr,
        )
        return 2
    if args.agent_mode == "belief-mcts" and (
        args.promotion_mcts_sims < 128 or args.promotion_move_time <= 0
    ):
        print(
            "ERROR: trusted belief-MCTS promotion requires >=128 sims and "
            "a positive identical move deadline for both agents",
            file=sys.stderr,
        )
        return 2
    if args.mcts_move_time <= 0:
        print("ERROR: --mcts-move-time must be > 0", file=sys.stderr)
        return 2
    if args.leaf_queue_depth < 1:
        print("ERROR: --leaf-queue-depth must be >= 1", file=sys.stderr)
        return 2
    if args.agent_mode == "oracle-mcts" and not config.SEARCH.allow_oracle_deck:
        print(
            "ERROR: oracle diagnostics require POKEBOT_ALLOW_ORACLE_DECK=1",
            file=sys.stderr,
        )
        return 2
    if args.regression_margin < 0.0:
        print("ERROR: --regression-margin must be >= 0", file=sys.stderr)
        return 2
    try:
        promotion_cfg = PromotionGateConfig(
            min_games=int(args.promotion_games),
            min_complete_pairs=int(args.promotion_min_pairs),
            threshold=float(args.promotion_threshold),
            confidence=float(args.promotion_confidence),
            bootstrap_resamples=int(args.promotion_bootstrap_resamples),
        )
        promotion_cfg.validate()
    except ValueError as exc:
        print(f"ERROR: invalid promotion gate: {exc}", file=sys.stderr)
        return 2
    if args.train_lr <= 0:
        print("ERROR: --train-lr must be > 0", file=sys.stderr)
        return 2

    import os

    os.environ["POKEBOT_PRIMARY_ARCHETYPE"] = args.archetype
    # Propagate agent-verbosity to spawned workers via env (inherited on spawn).
    if args.agent_verbose:
        os.environ["POKEBOT_AGENT_VERBOSE"] = "1"
    else:
        os.environ.setdefault("POKEBOT_AGENT_VERBOSE", "0")
    # When sim workers run on CPU, force them CPU-only so no worker ever creates
    # a CUDA context (prevents GPU oversubscription → false OOM). Only the parent
    # process (warm-train) touches the GPU → a single CUDA context total.
    sim_device = str(args.sim_device)
    if sim_device == "cpu":
        os.environ["POKEBOT_WORKER_CPU_ONLY"] = "1"
    run_name = args.run_name or f"{args.archetype}_round_robin"
    replay_lineage = args.replay_lineage or args.run_name or args.archetype
    policy_anchor_lineage = args.policy_anchor_lineage
    run_output_dir = paths.OUTPUTS_DIR / "runs" / run_name

    # Resolve concurrent in-flight games (worker processes). Oversubscribe the 32
    # threads when the GPU leaf server is active (IPC/GPU-blocked workers overlap
    # compute workers → CPU pegged), else one-per-thread. Then CAP by available
    # RAM so per_worker_rss_gb × workers stays inside the budget (never RAM-OOM).
    _use_server = args.leaf_eval == "gpu-server" and sim_device == "cpu"
    if int(args.workers) > 0:
        want_workers = int(args.workers)
    elif _use_server:
        want_workers = int(config.HARDWARE.rl_games_in_flight)
    else:
        want_workers = int(config.HARDWARE.sim_workers)
    try:
        avail_gb = float(config.memory_pressure().get("available_gb", 0.0))
        budget_gb = min(config.HARDWARE.ram_cache_gb, max(0.0, avail_gb - config.HARDWARE.free_ram_floor_gb))
        per = max(0.1, float(config.HARDWARE.per_worker_rss_gb))
        ram_cap = max(4, int(budget_gb / per)) if budget_gb > 0 else want_workers
    except Exception:
        ram_cap = want_workers
    workers = max(1, min(want_workers, ram_cap))
    args.workers = workers
    if workers < want_workers:
        print(
            f"[rr] RAM-capped concurrent games {want_workers} → {workers} "
            f"(per_worker≈{config.HARDWARE.per_worker_rss_gb}GB, "
            f"avail≈{config.memory_pressure().get('available_gb', 0):.0f}GB)",
            flush=True,
        )

    specs = ensure_baselines_installed(load_manifest())
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s.id in wanted]
    expected_opponent_ids = {s.id for s in specs}
    specs, dropped = filter_loadable_baselines(specs)
    if dropped:
        for sid, err in dropped:
            print(f"ERROR: expected baseline unavailable: {sid}: {err}", file=sys.stderr)
        print(
            "ERROR: refusing an incomplete round-robin field",
            file=sys.stderr,
        )
        return 2

    # Persistent blacklist of baselines that crashed / timed out / were illegal
    # in this or a prior run. Excluded from the field entirely (no rescheduling,
    # not in the WR matrix), and skipped from the start on subsequent runs.
    broken_path = run_output_dir / "broken_baselines.json"
    broken: dict[str, str] = {}
    if broken_path.is_file():
        try:
            broken = dict(json.loads(broken_path.read_text(encoding="utf-8")))
        except Exception:
            broken = {}

    def _persist_broken() -> None:
        broken_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = broken_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(broken, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(broken_path)

    # Self-heal: drop stale infra/resource (CUDA/RAM OOM, timeout) entries so
    # baselines wrongly removed by a prior oversubscribed run are RESTORED to the
    # field. Only genuine code faults (import errors, exceptions) stay excluded.
    restored = [sid for sid, e in broken.items() if _entry_is_resource(e)]
    if restored:
        for sid in restored:
            broken.pop(sid, None)
        _persist_broken()
        print(
            f"[rr] restored {len(restored)} baseline(s) wrongly removed by "
            f"OOM/timeout (cleared from blacklist): {sorted(restored)}",
            flush=True,
        )

    def _skip_record(error: str, kind: str) -> dict:
        # Ongoing policy is SKIP-ONLY: blacklist + exclude, but leave the agent
        # on disk and in the manifest (deletion is a separate one-time pass —
        # see scripts/prune_broken_baselines.py).
        return {
            "error": error,
            "kind": kind,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "deleted": False,
            "source": "train_round_robin (runtime skip)",
        }

    # Import-time failures found now are recorded as skips (not deleted).
    for sid, err in dropped:
        if sid not in broken:
            broken[sid] = _skip_record(f"import: {err}", "import")
    if dropped:
        _persist_broken()

    # Historical diagnostics never remove an expected opponent from a formal
    # iteration. Runtime failures below invalidate the iteration instead.
    if broken:
        print(
            f"[rr] retrying full expected field despite {len(broken)} historical "
            f"failure record(s) in {broken_path}",
            flush=True,
        )
    if not specs:
        print("ERROR: no loadable baselines", file=sys.stderr)
        return 2
    heldout_count = (
        min(len(specs) - 1, max(1, int(round(len(specs) * args.heldout_fraction))))
        if args.heldout_fraction > 0 and len(specs) > 1
        else 0
    )
    heldout_ids = (
        set(sorted(spec.id for spec in specs)[-heldout_count:])
        if heldout_count > 0
        else set()
    )
    print(
        f"[rr] held-out opponents ({len(heldout_ids)}; evaluation only): "
        f"{sorted(heldout_ids)}",
        flush=True,
    )

    our_deck = deck_pool.primary_deck()
    train_dev = training_device(prefer_name=config.HARDWARE.train_gpu_name)

    loop_state: dict[str, Any] = {
        "iteration": 0,
        "champion": None,
        "incumbent": None,
        "last_completed_iteration": -1,
        "history": [],
        "seed": args.seed,
        "games_per_opp": args.games_per_opp,
        "n_opponents": len(specs),
        "best_wr": -1.0,
        "best_champion": None,
        "n_note": (
            f"curriculum {args.games_per_opp} games/opp (iters <"
            f"{args.curriculum_switch_iter}) → {args.games_per_opp_late} "
            f"(iters ≥{args.curriculum_switch_iter}) × {len(specs)} opps "
            f"(seat-alternated; {args.workers} workers)"
        ),
    }

    resume_path = checkpoint.resolve_resume_path(run_name, args.resume)
    model_ckpt = _resolve_bootstrap_ckpt(args.bootstrap_ckpt, args.archetype)

    if resume_path is not None:
        print(f"[rr] resume loop from {resume_path}", flush=True)
        ckpt = checkpoint.load_checkpoint(resume_path, map_location="cpu")
        extra = ckpt.get("extra") or {}
        loop_state.update(extra.get("loop_state") or {})
        loop_state["iteration"] = next_iteration(
            loop_state,
            checkpoint_iteration=int(ckpt.get("rl_iteration", -1)),
        )
        if args.bootstrap_ckpt is not None:
            # An explicit checkpoint on a resumed loop is an operator rollback:
            # preserve history/best-WR bookkeeping, but do not silently replace
            # the requested weights with the collapsed resumed candidate.
            model_ckpt = model_ckpt.expanduser().resolve()
            loop_state["champion"] = str(model_ckpt)
            print(
                f"[rr] explicit resume champion override → {model_ckpt}",
                flush=True,
            )
        else:
            incumbent = loop_state.get("incumbent") or {}
            history = loop_state.get("history") or []
            last_history_champion = (
                history[-1].get("champion") if history else None
            )
            model_ckpt = Path(
                incumbent.get("path")
                or loop_state.get("last_evaluated_champion")
                or last_history_champion
                or model_ckpt
            )
        print(
            f"[rr] continuing at next iteration={loop_state['iteration']} "
            f"champion={model_ckpt}",
            flush=True,
        )
    else:
        model_ckpt = model_ckpt.expanduser().resolve()

    # Never operate through a mutable bootstrap/latest pointer. Every incumbent
    # is pinned to a digest-addressed evaluated snapshot.
    model_ckpt = _snapshot_best_champion(model_ckpt, run_name)
    incumbent_identity = CheckpointIdentity.from_path(model_ckpt)
    if args.agent_mode in ("policy", "belief-mcts"):
        try:
            checkpoint.assert_trusted_policy_checkpoint(model_ckpt)
        except Exception as exc:
            print(
                f"ERROR: incumbent is not trusted history-policy compatible: {exc}",
                file=sys.stderr,
            )
            return 2
    loop_state["champion"] = incumbent_identity.path  # legacy readers
    loop_state["incumbent"] = incumbent_identity.as_dict()

    loop_state["active_training_config"] = {
        "train_epochs": int(args.train_epochs),
        "train_lr": float(args.train_lr),
        "patience": int(args.patience),
        "val_frac": float(args.val_frac),
        "bootstrap_mix": float(args.bootstrap_mix),
        "history_mix": float(args.history_mix),
        "replay_fraction": float(args.replay_fraction),
        "policy_anchor_ratio": float(args.policy_anchor_ratio),
        "max_decisions_per_game": int(args.max_decisions_per_game),
        "worker_autotune": bool(args.worker_autotune),
        "replay_history_iters": int(args.replay_history_iters),
        "heldout_fraction": float(args.heldout_fraction),
        "heldout_opponents": sorted(heldout_ids),
        "agent_mode": args.agent_mode,
        "replay_lineage": replay_lineage,
        "policy_anchor_lineage": policy_anchor_lineage,
        "engine_seedable": False,
        "mcts_sims": int(args.mcts_sims),
        "mcts_move_time": float(args.mcts_move_time),
        "game_timeout_s": int(args.game_timeout_s),
        "expected_search_decisions": int(
            args.expected_search_decisions
        ),
        "collection_contract": {
            "schema": "belief_decision_budget_v1",
            "supersedes": "policy_fixed_2600_games",
            "nominal_games_per_opponent": int(args.games_per_opp),
            "min_games_per_opponent": int(args.min_games_per_opp),
            "max_games_per_opponent": int(args.max_games_per_opp),
            "target_search_decisions": int(args.target_search_decisions),
            "target_derivation": (
                PILOT_DERIVATION
                if int(args.target_search_decisions)
                == PILOT_DERIVED_TARGET_SEARCH_DECISIONS
                else {
                    "method": "explicit_cli_override",
                    "target_search_decisions": int(
                        args.target_search_decisions
                    ),
                }
            ),
            "balanced_seats": True,
        },
        "leaf_queue_depth": int(args.leaf_queue_depth),
        "promotion_gate": promotion_cfg.__dict__,
        "promotion_max_games": int(args.promotion_max_games),
        "promotion_batch_games": int(args.promotion_batch_games),
        "promotion_agent_mode": args.agent_mode,
        "promotion_mcts_sims_both_agents": int(args.promotion_mcts_sims),
        "promotion_move_time_both_agents": float(args.promotion_move_time),
        "promotion_mcts_sims": int(args.promotion_mcts_sims),
    }
    source_digest = _source_tree_digest()
    simulator_digest = _simulator_digest()
    config_digest = _sha256_bytes(
        json.dumps(
            loop_state["active_training_config"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    loop_state["run_provenance"] = {
        "source_tree_digest": source_digest,
        "git_diff_digest": os.environ.get(
            "POKEBOT_GIT_DIFF_DIGEST", "not_provided"
        ),
        "simulator_digest": simulator_digest,
        "config_digest": config_digest,
        "feature_schema": __import__(
            "poke_bot.features", fromlist=["FEATURE_SCHEMA_VERSION"]
        ).FEATURE_SCHEMA_VERSION,
        "decision_context": "history",
        "search_mode": args.agent_mode,
        "engine_seedable": False,
        "iteration_contract_transition": {
            "from": "policy_fixed_2600_games_per_iteration",
            "to": (
                "belief_decision_budget_v1"
                if args.agent_mode == "belief-mcts"
                else "legacy_policy_game_budget"
            ),
            "old_policy_targets_equivalent": False,
            "policy_anchor_is_explicit": bool(args.policy_anchor_ratio > 0),
        },
    }
    metadata_dir = run_output_dir
    metadata_path = metadata_dir / "run_metadata.json"
    if metadata_path.exists() and resume_path is None:
        print(
            f"ERROR: refusing to overwrite existing run metadata: {metadata_path}",
            file=sys.stderr,
        )
        return 2
    if not metadata_path.exists():
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "command_config": vars(args),
                    "incumbent": incumbent_identity.as_dict(),
                    "provenance": loop_state["run_provenance"],
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.agent_mode == "belief-mcts":
        transition_path = metadata_dir / "belief_iteration_contract.json"
        effective_iteration = int(loop_state.get("iteration", 0))
        completed_boundary = int(
            loop_state.get("last_completed_iteration", effective_iteration - 1)
        )
        transition_payload = {
            "lineage": run_name,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "effective_iteration": effective_iteration,
            "completed_boundary_iteration": completed_boundary,
            "from": "policy_fixed_2600_games_per_iteration",
            "to": loop_state["active_training_config"]["collection_contract"],
            "replay": {
                "target_lineage": replay_lineage,
                "policy_anchor_lineage": policy_anchor_lineage,
                "target_fraction": float(args.replay_fraction),
                "policy_anchor_ratio": float(args.policy_anchor_ratio),
                "policy_targets_equivalent_to_belief_targets": False,
                "bootstrap_mix": float(args.bootstrap_mix),
                "bootstrap_is_separate_regularizer": True,
            },
        }
        if transition_path.exists():
            existing = json.loads(transition_path.read_text())
            changed_keys = [
                key
                for key in ("from", "to", "replay")
                if existing.get(key) != transition_payload.get(key)
            ]
            if changed_keys:
                if not args.restart_partial_iteration:
                    raise RuntimeError(
                        "belief iteration contract changed on resume: "
                        + ",".join(changed_keys)
                    )
                target_final = (
                    paths.DATA_DIR
                    / "rl"
                    / replay_lineage
                    / f"iter{effective_iteration:03d}.jsonl"
                )
                _validate_contract_migration_boundary(
                    completed_iteration=completed_boundary,
                    effective_iteration=effective_iteration,
                    finalized_paths=[target_final],
                )
                old_replay = existing.get("replay") or {}
                old_target_lineage = str(
                    old_replay.get("target_lineage")
                    or existing.get("lineage")
                    or run_name
                )
                previous_contract = {
                    key: existing.get(key)
                    for key in ("from", "to", "replay")
                }
                previous_digest = _contract_digest(previous_contract)
                suffix = (
                    ".invalid.contract-migration."
                    + previous_digest.split(":", 1)[-1][:12]
                    + f".{int(time.time())}"
                )
                quarantined: list[str] = []
                for lineage in dict.fromkeys(
                    (old_target_lineage, replay_lineage)
                ):
                    finalized = (
                        paths.DATA_DIR
                        / "rl"
                        / lineage
                        / f"iter{effective_iteration:03d}.jsonl"
                    )
                    if finalized.exists():
                        raise RuntimeError(
                            "contract migration found finalized boundary replay: "
                            f"{finalized}"
                        )
                    quarantined.extend(
                        _quarantine_iteration_partials(
                            lineage,
                            effective_iteration,
                            suffix=suffix,
                        )
                    )
                migrations = list(
                    existing.get(
                        "migration_history",
                        existing.get("revision_history", []),
                    )
                )
                migrations.append(
                    {
                        "recorded_at": transition_payload["recorded_at"],
                        "reason": "explicit_completed_boundary_contract_migration",
                        "changed_keys": changed_keys,
                        "effective_iteration": effective_iteration,
                        "completed_boundary_iteration": completed_boundary,
                        "previous_contract_digest": previous_digest,
                        "new_contract_digest": _contract_digest(
                            {
                                key: transition_payload.get(key)
                                for key in ("from", "to", "replay")
                            }
                        ),
                        "previous_contract": previous_contract,
                        "quarantined_partial_artifacts": quarantined,
                        "finalized_iteration_reinterpreted": False,
                    }
                )
                transition_payload["first_recorded_at"] = existing.get(
                    "first_recorded_at", existing.get("recorded_at")
                )
                transition_payload["migration_history"] = migrations
                tmp_transition = transition_path.with_suffix(".json.tmp")
                tmp_transition.write_text(
                    json.dumps(transition_payload, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(tmp_transition, transition_path)
                print(
                    "[rr] migrated belief iteration contract at clean boundary "
                    f"{completed_boundary}->{effective_iteration}: "
                    f"changed={changed_keys} quarantined={len(quarantined)} "
                    f"replay_lineage={replay_lineage}",
                    flush=True,
                )
        else:
            transition_path.write_text(
                json.dumps(transition_payload, indent=2) + "\n",
                encoding="utf-8",
            )
    loop_state["run_metadata_path"] = str(metadata_path)

    print(f"== train_round_robin", flush=True)
    print(f"   {loop_state['n_note']}", flush=True)
    print(f"   sim_device={sim_device} (workers CPU-only={os.environ.get('POKEBOT_WORKER_CPU_ONLY', '0')}) "
          f"train_device={train_dev}", flush=True)
    print(f"   sim_workers={args.workers} leaf_batch={config.leaf_batch_for_device(sim_device)} "
          f"train_gpu={config.gpu_kind(train_dev)}", flush=True)
    print(
        f"   agent_mode={args.agent_mode} draw_aware_gate>={args.gate} "
        f"engine_seedable=False",
        flush=True,
    )
    print(
        f"   train_epochs={args.train_epochs} patience={args.patience} "
        f"train_lr={args.train_lr:g} val_frac={args.val_frac:g} "
        f"bootstrap_mix={args.bootstrap_mix:g}",
        flush=True,
    )

    # ---- Persistent GPU-resident batched leaf-eval server -----------------
    # ONE process keeps the champion on the GPU and batches MCTS leaf forwards
    # across ALL concurrent CPU games (fills the GPU during the game phase).
    # CPU workers stay CUDA_VISIBLE_DEVICES="" (no per-worker CUDA context → no
    # OOM); they featurize leaves and offload the forward to this server.
    leaf_servers: list = []
    leaf_req_qs: list = []
    leaf_ctrl_qs: list = []
    leaf_status_qs: list = []
    leaf_alive_evts: list = []
    leaf_resp_qs: list = []
    remote_channel = None
    leaf_slot_counter = None
    leaf_version = 0
    leaf_digest = incumbent_identity.digest
    n_servers = 1
    use_leaf_server = args.leaf_eval == "gpu-server" and sim_device == "cpu"
    if use_leaf_server:
        import multiprocessing as _mp

        from poke_bot.batched_infer import run_leaf_server

        # Pin the leaf servers to the SAME Blackwell device the trainer uses (the
        # whole primary pipeline lives on GPU1). 'auto' → mirror train_dev.
        leaf_gpu = str(train_dev) if args.leaf_gpu == "auto" else args.leaf_gpu
        # Up to N concurrent replica servers on the one card; can't exceed the
        # worker count (each replica needs ≥1 worker feeding it).
        n_servers = max(1, min(int(args.leaf_servers), int(args.workers)))
        mpctx = _mp.get_context("spawn")
        leaf_resp_qs = [mpctx.Queue(maxsize=2) for _ in range(args.workers)]
        leaf_slot_counter = mpctx.Value("i", 0)
        readies = []
        for j in range(n_servers):
            rq = mpctx.Queue(maxsize=int(args.leaf_queue_depth))
            cq = mpctx.Queue(maxsize=8)
            sq = mpctx.Queue(maxsize=16)
            ev = mpctx.Event()
            alive = mpctx.Event()
            proc = mpctx.Process(
                target=run_leaf_server,
                args=(str(model_ckpt), leaf_gpu, rq, leaf_resp_qs),
                kwargs=dict(
                    ready_evt=ev,
                    alive_evt=alive,
                    ctrl_q=cq,
                    status_q=sq,
                    expected_digest=leaf_digest,
                    initial_version=leaf_version,
                    bf16=True,
                    max_batch=int(args.leaf_max_batch),
                    coalesce_ms=float(args.leaf_coalesce_ms),
                ),
                daemon=True,
            )
            proc.start()
            leaf_req_qs.append(rq)
            leaf_ctrl_qs.append(cq)
            leaf_status_qs.append(sq)
            leaf_alive_evts.append(alive)
            leaf_servers.append(proc)
            readies.append(ev)
        startup_ok = True
        for j, ev in enumerate(readies):
            if not ev.wait(timeout=240):
                print("[rr] ERROR: leaf server did not signal ready in 240s", file=sys.stderr)
                startup_ok = False
                continue
            try:
                status = leaf_status_qs[j].get(timeout=5)
            except Exception as exc:
                print(f"[rr] ERROR: missing leaf ready status: {exc}", file=sys.stderr)
                startup_ok = False
                continue
            if (
                not status.get("ok")
                or status.get("checkpoint_digest") != leaf_digest
                or int(status.get("version", -1)) != leaf_version
                or not leaf_servers[j].is_alive()
                or not leaf_alive_evts[j].is_set()
            ):
                print(f"[rr] ERROR: invalid leaf ready ack: {status}", file=sys.stderr)
                startup_ok = False
        if not startup_ok:
            for j, proc in enumerate(leaf_servers):
                try:
                    leaf_ctrl_qs[j].put({"cmd": "stop"})
                    proc.join(timeout=5)
                except Exception:
                    pass
            return 3
        # Workers are sharded across replicas by slot % n_servers (see
        # worker_pool._worker_init).
        remote_channel = {
            "req_qs": leaf_req_qs,
            "resp_qs": leaf_resp_qs,
            "slot_counter": leaf_slot_counter,
            "ctrl_qs": leaf_ctrl_qs,
            "generation": 0,
            "alive_evts": leaf_alive_evts,
            "expected_digest": leaf_digest,
            "expected_version": leaf_version,
            "timeout_s": config.SEARCH.remote_request_timeout_s,
        }
        pids = [p.pid for p in leaf_servers]
        print(
            f"   leaf-eval=gpu-server x{n_servers} replicas on {leaf_gpu} "
            f"({config.gpu_kind(leaf_gpu)}; champions GPU-resident, concurrent "
            f"CUDA contexts; {args.workers} CPU workers sharded across them) "
            f"pids={pids}",
            flush=True,
        )
    else:
        print("   leaf-eval=cpu (net runs locally in each CPU worker; GPU idle)", flush=True)

    mgr = checkpoint.CheckpointManager(run_name)
    rl_replay_dir = paths.DATA_DIR / "rl" / replay_lineage

    start_iter = int(loop_state.get("iteration", 0))
    fatal_health_error: str | None = None
    for it in tqdm(range(start_iter, args.iterations), desc="rr iterations", unit="iter"):
        loop_state["iteration"] = it
        incumbent = loop_state.get("incumbent") or {}
        champion = Path(incumbent.get("path") or loop_state["champion"])

        from poke_bot.iteration_contract import SearchCollectionContract

        nominal_gpo = (
            int(args.games_per_opp)
            if args.agent_mode == "belief-mcts"
            else _games_per_opp_for_iter(it, args)
        )
        collection_contract = SearchCollectionContract(
            opponents=len(specs),
            nominal_games_per_opponent=nominal_gpo,
            min_games_per_opponent=(
                int(args.min_games_per_opp)
                if args.agent_mode == "belief-mcts"
                else nominal_gpo
            ),
            max_games_per_opponent=(
                int(args.max_games_per_opp)
                if args.agent_mode == "belief-mcts"
                else nominal_gpo
            ),
            target_search_decisions=(
                int(args.target_search_decisions)
                if args.agent_mode == "belief-mcts"
                else 0
            ),
        )
        job_gpo = collection_contract.max_games_per_opponent
        report = FieldReport(
            gate_threshold=args.gate,
            expected_opponents=set(expected_opponent_ids),
            min_games_per_opponent=collection_contract.min_games_per_opponent,
        )
        phase = "late" if it >= args.curriculum_switch_iter else "early"
        loop_state["games_per_opp"] = nominal_gpo
        loop_state["curriculum_phase"] = phase
        tqdm.write(
            f"[rr] iter {it}: collection contract="
            f"{'belief_decision_budget_v1' if args.agent_mode == 'belief-mcts' else 'legacy_policy'} "
            f"games_per_opp={collection_contract.min_games_per_opponent}/"
            f"{collection_contract.nominal_games_per_opponent}/"
            f"{collection_contract.max_games_per_opponent} "
            f"(min/nominal/max; balanced seats) × {len(specs)} opponents; "
            f"target_trusted_decisions={collection_contract.target_search_decisions or 'disabled'}",
            file=sys.stderr,
        )

        # The engine exposes no RNG seed API. Schedule balanced independent
        # seat strata and never claim Python RNG seeds create paired trials.
        jobs: list[dict] = []
        seed = int(loop_state.get("seed", args.seed)) + it * 100_000
        target_parent = CheckpointIdentity.from_path(champion)
        # Wave-major ordering means each completed two-game wave adds one game
        # in each seat for every opponent and leaves a contiguous writer prefix.
        for game_i in range(job_gpo):
            for spec in specs:
                our_seat = game_i % 2
                game_seed = seed
                seed += 1
                job_index = len(jobs)
                jobs.append(
                    {
                        "job_index": job_index,
                        "checkpoint": str(champion),
                        "checkpoint_digest": target_parent.digest,
                        "model_generation": it + 1,
                        "model_max_context": int(config.MODEL.max_context),
                        "our_deck": our_deck,
                        "spec": {
                            "id": spec.id,
                            "name": spec.name,
                            "dir_name": spec.dir_name,
                            "group": spec.group,
                            "source": spec.source,
                            "path": str(spec.path),
                        },
                        "our_seat": our_seat,
                        "pair_id": None,
                        "mcts_sims": args.mcts_sims,
                        "mcts_move_time": args.mcts_move_time,
                        "game_timeout_s": int(args.game_timeout_s),
                        "expected_search_decisions": int(
                            args.expected_search_decisions
                        ),
                        "agent_mode": args.agent_mode,
                        "seed": game_seed,
                        "device": sim_device,
                        "archetype": args.archetype,
                        "training_eligible": spec.id not in heldout_ids,
                        "target_provenance": {
                            **loop_state["run_provenance"],
                            "target_source": (
                                "belief_mcts"
                                if args.agent_mode == "belief-mcts"
                                else "history_policy"
                            ),
                            "trusted": args.agent_mode in ("policy", "belief-mcts"),
                            "action_decoder": "ordered_autoregressive_v1",
                            "feature_schema": features.FEATURE_SCHEMA_VERSION,
                            "incumbent_checkpoint": target_parent.as_dict(),
                            "iteration": it,
                            "model_generation": it + 1,
                            **(
                                {"opponent_id": spec.id}
                                if args.agent_mode == "policy"
                                else {}
                            ),
                            "our_seat": our_seat,
                        },
                    }
                )

        rl_dir = paths.DATA_DIR / "rl" / replay_lineage
        rl_dir.mkdir(parents=True, exist_ok=True)
        iter_replay_path = rl_dir / f"iter{it:03d}.jsonl"
        partial_replay_path = iter_replay_path.with_suffix(".jsonl.partial")
        if iter_replay_path.exists():
            fatal_health_error = (
                f"refusing to overwrite iteration {it} artifacts: "
                f"{iter_replay_path}"
            )
            tqdm.write(f"[rr] ABORT: {fatal_health_error}", file=sys.stderr)
            break
        if args.restart_partial_iteration:
            restart_suffix = f".invalid.restart.{int(time.time())}"
            partial_artifacts = (
                partial_replay_path,
                partial_replay_path.with_suffix(
                    partial_replay_path.suffix + ".journal"
                ),
                partial_replay_path.with_suffix(
                    partial_replay_path.suffix + ".writer.json"
                ),
            )
            quarantined_partials = []
            for artifact in partial_artifacts:
                if artifact.exists():
                    destination = artifact.with_name(
                        artifact.name + restart_suffix
                    )
                    os.replace(artifact, destination)
                    quarantined_partials.append(str(destination))
            if quarantined_partials:
                tqdm.write(
                    "[rr] quarantined invalid partial iteration artifacts: "
                    + ", ".join(quarantined_partials),
                    file=sys.stderr,
                )
        # Workers pre-serialize records. This bounded writer reorders by stable
        # job index and commits replay+journal+offsets exactly once.
        replay_writer = OrderedReplayWriter(
            partial_replay_path,
            expected_jobs=len(jobs),
            queue_depth=max(16, min(256, int(args.leaf_queue_depth))),
            fsync_batch=8,
        )
        n_written = 0
        n_expected_targets = 0
        n_scheduled_jobs = 0
        trusted_search_decisions = 0
        completed_search_sims = 0
        completed_coverage_rows: list[tuple[str, int]] = []
        collection_stop_reason = "not_started"
        actual_games_per_opponent = 0
        wins = 0.0
        n = 0
        done_games = 0
        # Health aggregates — detect fake/instant games (no MCTS / fail-closed).
        sum_wall_s = 0.0
        sum_steps = 0
        sum_decisions = 0
        sum_sims_run = 0
        remote_requests_total = 0
        remote_queue_wait_ms_total = 0.0
        remote_inference_batch_total = 0.0
        remote_batch_occupancy_total = 0.0
        request_queue_depth_max = 0
        server_request_rate_max = 0.0
        max_search_depth = 0
        max_complete_ordered_action_count = 0
        factorized_search_decisions = 0
        leaf_evaluations_total = 0
        n_fail_closed = 0
        n_leaf_remote = 0
        n_zero_targets = 0
        n_resource_errors = 0
        n_game_timeouts = 0
        n_trust_failures = 0
        n_cancelled = 0
        n_baseline_failures = 0
        n_our_failures = 0
        n_results_received = 0
        play_t0 = time.perf_counter()
        # Fresh pool each iteration (bounds libcg leak). Reset slot counter so
        # workers re-claim leaf-server slots 0..N-1 deterministically.
        if leaf_slot_counter is not None:
            with leaf_slot_counter.get_lock():
                leaf_slot_counter.value = 0
        if remote_channel is not None:
            dead = [
                str(proc.pid)
                for proc, alive in zip(leaf_servers, leaf_alive_evts)
                if not proc.is_alive() or not alive.is_set()
            ]
            if dead:
                replay_writer.abort("leaf server dead before iteration")
                replay_writer.quarantine(f".invalid.{int(time.time())}")
                fatal_health_error = f"leaf server(s) dead before iteration: {dead}"
                tqdm.write(f"[rr] ABORT: {fatal_health_error}", file=sys.stderr)
                break
            remote_channel["generation"] = it + 1
            remote_channel["expected_digest"] = (
                loop_state.get("incumbent") or {}
            ).get("digest")
            remote_channel["expected_version"] = leaf_version
        def _consume_result(res: dict, game_bar) -> None:
            nonlocal n_results_received, n_resource_errors, n_baseline_failures
            nonlocal n_game_timeouts, n_trust_failures, n_cancelled
            nonlocal n_our_failures, sum_wall_s, sum_steps, sum_decisions
            nonlocal sum_sims_run, remote_requests_total
            nonlocal remote_queue_wait_ms_total, remote_inference_batch_total
            nonlocal remote_batch_occupancy_total, request_queue_depth_max
            nonlocal server_request_rate_max, max_search_depth
            nonlocal max_complete_ordered_action_count
            nonlocal factorized_search_decisions
            nonlocal leaf_evaluations_total, n_fail_closed, n_leaf_remote
            nonlocal n_written, n_zero_targets, n, done_games, wins
            nonlocal trusted_search_decisions, completed_search_sims

            n_results_received += 1
            record_json = res.get("record_json")
            replay_writer.submit(
                int(res["job_index"]),
                record_json,
                {
                    key: value
                    for key, value in res.items()
                    if key not in ("record", "record_json", "targets")
                },
            )
            oid = res["opponent_id"]
            if res.get("cancelled"):
                n_cancelled += 1
                game_bar.update(1)
                return
            if res.get("trust_failure"):
                n_trust_failures += 1
                tqdm.write(
                    f"[rr] trusted-search failure vs {oid}: {res.get('error')}",
                    file=sys.stderr,
                )
                game_bar.update(1)
                return
            if res.get("resource_error"):
                n_resource_errors += 1
                n_game_timeouts += int(bool(res.get("game_timeout")))
                tqdm.write(
                    f"[rr] resource error vs {oid} (retryable, NOT "
                    f"blacklisted): {res.get('error')}",
                    file=sys.stderr,
                )
                game_bar.update(1)
                return
            if res.get("baseline_failed"):
                n_baseline_failures += 1
                if oid not in broken:
                    err = res.get("error") or "unknown failure"
                    broken[oid] = _skip_record(err, "runtime")
                    _persist_broken()
                    tqdm.write(
                        f"[rr] skipped baseline {oid}: {err} "
                        f"(blacklisted; left on disk + in manifest)",
                        file=sys.stderr,
                    )
                game_bar.update(1)
                return
            if res.get("our_failed"):
                n_our_failures += 1
                tqdm.write(
                    f"[rr] our runtime failed vs {oid}: {res.get('error')}",
                    file=sys.stderr,
                )
                game_bar.update(1)
                return

            decisions = int(res.get("n_decisions") or 0)
            completed_sims = int(res.get("mcts_sims_run_total") or 0)
            sum_wall_s += float(res.get("wall_s") or 0.0)
            sum_steps += int(res.get("steps") or 0)
            sum_decisions += decisions
            sum_sims_run += completed_sims
            remote_requests_total += int(res.get("remote_leaf_requests") or 0)
            remote_queue_wait_ms_total += float(
                res.get("remote_queue_wait_ms_total") or 0.0
            )
            remote_inference_batch_total += float(
                res.get("remote_inference_batch_total") or 0.0
            )
            remote_batch_occupancy_total += float(
                res.get("remote_batch_occupancy_total") or 0.0
            )
            request_queue_depth_max = max(
                request_queue_depth_max,
                int(res.get("request_queue_depth_max") or 0),
            )
            server_request_rate_max = max(
                server_request_rate_max,
                float(res.get("server_request_rate_max") or 0.0),
            )
            max_search_depth = max(
                max_search_depth, int(res.get("max_search_depth") or 0)
            )
            max_complete_ordered_action_count = max(
                max_complete_ordered_action_count,
                int(res.get("max_complete_ordered_action_count") or 0),
            )
            factorized_search_decisions += int(
                res.get("factorized_search_decisions") or 0
            )
            leaf_evaluations_total += int(res.get("leaf_evaluations") or 0)
            if int(res.get("fail_closed") or 0) > 0:
                n_fail_closed += 1
            if res.get("leaf_remote"):
                n_leaf_remote += 1
            if record_json:
                n_written += 1
                if (
                    args.agent_mode == "belief-mcts"
                    and res.get("training_eligible", True)
                    and decisions > 0
                    and completed_sims >= 128 * decisions
                ):
                    trusted_search_decisions += decisions
                    completed_search_sims += completed_sims
            elif res.get("training_eligible", True):
                n_zero_targets += 1
            report.merge_game(
                oid,
                our_seat=res["our_seat"],
                winner=res["winner"],
                is_mirror=res["is_mirror"],
                mcts_on=True,
                pair_id=None,
            )
            completed_coverage_rows.append((oid, int(res["our_seat"])))
            n += 1
            done_games += 1
            if res["winner"] == 2:
                wins += 0.5
            elif res["winner"] == res["our_seat"]:
                wins += 1.0
            if done_games % 256 == 0:
                mp_snap = config.memory_pressure()
                if mp_snap.get("under_floor"):
                    tqdm.write(
                        f"[rr] RAM pressure: avail≈"
                        f"{mp_snap.get('available_gb', 0):.0f}GB near floor "
                        f"{config.HARDWARE.free_ram_floor_gb}GB (streamed "
                        f"buffers bounded; concurrency={args.workers})",
                        file=sys.stderr,
                    )
            game_bar.update(1)
            opp_st = report.get(oid)
            game_bar.set_postfix(
                wr=f"{wins / max(n, 1):.1%}",
                last=(
                    f"{oid[:14]}="
                    f"{opp_st.win_rate:.0%}"
                    f"({opp_st.wins + 0.5 * opp_st.draws:g}/{opp_st.games})"
                ),
                pass_=f"{len(report.opponents_passing())}/{len(specs)}",
                removed=len(broken),
            )

        from poke_bot.iteration_contract import balanced_seat_coverage

        game_bar = tqdm(
            total=len(jobs),
            desc=f"iter{it} games",
            unit="game",
            mininterval=0.5,
            file=sys.stderr,
        )
        wave_workers = int(args.workers)
        if args.agent_mode == "belief-mcts" and args.worker_autotune:
            wave_workers = min(
                int(args.workers),
                max(2, int(n_servers) * 3),
            )
        wave_ends = (
            range(2, job_gpo + 1, 2)
            if args.agent_mode == "belief-mcts"
            else [job_gpo]
        )
        collection_interrupted = False
        for wave_end in wave_ends:
            if stop_requested["signal"] is not None:
                collection_interrupted = True
                break
            wave_start = wave_end - (2 if args.agent_mode == "belief-mcts" else job_gpo)
            wave_jobs = jobs[
                wave_start * len(specs) : wave_end * len(specs)
            ]
            n_scheduled_jobs += len(wave_jobs)
            n_expected_targets += sum(
                1 for job in wave_jobs if job.get("training_eligible", True)
            )
            pool = WorkerPool(
                num_workers=wave_workers,
                remote_channel=remote_channel,
            )
            active_pool["pool"] = pool
            try:
                with pool:
                    for res in pool.imap_unordered(_worker_play, wave_jobs):
                        _consume_result(res, game_bar)
                        if stop_requested["signal"] is not None:
                            pool.request_stop(
                                f"signal {stop_requested['signal']}"
                            )
                            collection_interrupted = True
                            break
            except (
                WorkerPoolStopped,
                CollectionInterrupted,
                ValueError,
            ):
                if stop_requested["signal"] is None:
                    raise
                pool.request_stop(f"signal {stop_requested['signal']}")
                collection_interrupted = True
            finally:
                active_pool["pool"] = None
            if collection_interrupted:
                break
            actual_games_per_opponent = wave_end
            coverage_complete, coverage_counts = balanced_seat_coverage(
                completed_coverage_rows,
                opponents=expected_opponent_ids,
                min_games_per_opponent=collection_contract.min_games_per_opponent,
            )
            collection_stop_reason = collection_contract.stop_reason(
                games_per_opponent=actual_games_per_opponent,
                trusted_decisions=trusted_search_decisions,
                coverage_complete=coverage_complete,
            ) or "continue"
            worker_tune = "hold"
            occupancy = (
                remote_batch_occupancy_total / remote_requests_total
                if remote_requests_total
                else 0.0
            )
            queue_wait = (
                remote_queue_wait_ms_total / remote_requests_total
                if remote_requests_total
                else 0.0
            )
            if (
                args.agent_mode == "belief-mcts"
                and args.worker_autotune
                and collection_stop_reason == "continue"
            ):
                old_workers = wave_workers
                if (
                    occupancy < 0.25
                    and request_queue_depth_max <= 2
                    and wave_workers < int(args.workers)
                ):
                    wave_workers = min(
                        int(args.workers),
                        wave_workers + int(n_servers),
                    )
                elif (
                    (
                        request_queue_depth_max
                        >= int(args.leaf_queue_depth) * 0.8
                        or queue_wait > 25.0
                    )
                    and wave_workers > max(2, int(n_servers))
                ):
                    wave_workers = max(
                        max(2, int(n_servers)),
                        wave_workers - int(n_servers),
                    )
                if wave_workers != old_workers:
                    worker_tune = f"{old_workers}->{wave_workers}"
            tqdm.write(
                f"[rr] collection wave: fresh_games={n} "
                f"games_per_opp={actual_games_per_opponent} "
                f"trusted_decisions={trusted_search_decisions}/"
                f"{collection_contract.target_search_decisions or 'nominal'} "
                f"decisions_per_game={trusted_search_decisions/max(n_written, 1):.2f} "
                f"coverage_complete={coverage_complete} "
                f"inference_occupancy={occupancy:.3f} "
                f"queue_wait_ms={queue_wait:.2f} "
                f"next_wave_workers={wave_workers} tune={worker_tune} "
                f"stop={collection_stop_reason}",
                file=sys.stderr,
            )
            if collection_stop_reason != "continue":
                break
        game_bar.close()
        if collection_interrupted or stop_requested["signal"] is not None:
            replay_writer.abort(
                f"signal {stop_requested['signal']} interrupted collection"
            )
            print(
                f"[rr] iteration {it} interrupted; retained partial replay at "
                f"{partial_replay_path}; no train/promotion/checkpoint",
                flush=True,
            )
            break
        if collection_stop_reason == "continue":
            collection_stop_reason = "max_games"
        for job in jobs[n_scheduled_jobs:]:
            replay_writer.submit(
                int(job["job_index"]),
                None,
                {
                    "not_scheduled": True,
                    "reason": collection_stop_reason,
                    "opponent_id": job["spec"]["id"],
                    "our_seat": job["our_seat"],
                },
            )
        writer_telemetry = replay_writer.close()
        n_written = replay_writer.written_records
        if stop_requested["signal"] is not None:
            quarantined = replay_writer.quarantine(
                f".interrupted.signal-{stop_requested['signal']}.{int(time.time())}"
            )
            print(
                f"[rr] iteration {it} interrupted after collection; "
                f"quarantined={quarantined}; no train/promotion/checkpoint",
                flush=True,
            )
            break
        play_wall = time.perf_counter() - play_t0
        gps = n / max(play_wall, 1e-6)
        avg_steps = sum_steps / max(n, 1)
        avg_dec = sum_decisions / max(n, 1)
        avg_sims = sum_sims_run / max(n, 1)
        avg_wall = sum_wall_s / max(n, 1)
        mean_queue_wait_ms = (
            remote_queue_wait_ms_total / remote_requests_total
            if remote_requests_total
            else 0.0
        )
        mean_inference_batch = (
            remote_inference_batch_total / remote_requests_total
            if remote_requests_total
            else 0.0
        )
        mean_batch_occupancy = (
            remote_batch_occupancy_total / remote_requests_total
            if remote_requests_total
            else 0.0
        )
        coverage_complete, coverage_counts = balanced_seat_coverage(
            completed_coverage_rows,
            opponents=expected_opponent_ids,
            min_games_per_opponent=collection_contract.min_games_per_opponent,
        )
        completed_sims_per_trusted_decision = (
            completed_search_sims / trusted_search_decisions
            if trusted_search_decisions
            else 0.0
        )
        tqdm.write(
            f"[rr] health iter={it}: {gps:.2f} games/s wall={play_wall:.1f}s "
            f"fresh_games={n} trusted_decisions={trusted_search_decisions} "
            f"decisions_per_game={trusted_search_decisions/max(n_written, 1):.2f} "
            f"completed_sims_per_decision={completed_sims_per_trusted_decision:.1f} "
            f"games_per_opp={actual_games_per_opponent} "
            f"opponent_seat_coverage={coverage_complete} "
            f"collection_stop={collection_stop_reason} "
            f"avg_steps={avg_steps:.1f} avg_our_decisions={avg_dec:.1f} "
            f"agent_mode={args.agent_mode} avg_search_sims/game={avg_sims:.0f} "
            f"avg_game_wall={avg_wall:.2f}s "
            f"fail_closed_games={n_fail_closed}/{n} leaf_remote={n_leaf_remote}/{n} "
            f"trust_failures={n_trust_failures} game_timeouts={n_game_timeouts} "
            f"zero_target_games={n_zero_targets} "
            f"leaf_requests={remote_requests_total} "
            f"leaf_requests_per_s={remote_requests_total/max(play_wall, 1e-9):.1f} "
            f"queue_wait_ms_mean={mean_queue_wait_ms:.3f} "
            f"inference_batch_mean={mean_inference_batch:.2f} "
            f"batch_occupancy_mean={mean_batch_occupancy:.3f} "
            f"request_queue_depth_max={request_queue_depth_max} "
            f"server_request_rate={server_request_rate_max:.1f}/s "
            f"leaf_evaluations={leaf_evaluations_total} "
            f"max_search_depth={max_search_depth} "
            f"max_ordered_actions={max_complete_ordered_action_count} "
            f"factorized_decisions={factorized_search_decisions} "
            f"writer_queue_max={writer_telemetry['max_queue_depth']} "
            f"writer_records_per_s={writer_telemetry['records_per_s']:.2f}"
        )
        invalid_reasons: list[str] = []
        if n_results_received != n_scheduled_jobs:
            invalid_reasons.append(
                f"received {n_results_received}/{n_scheduled_jobs} scheduled results"
            )
        if n_resource_errors:
            invalid_reasons.append(f"resource/timeout failures={n_resource_errors}")
        if n_game_timeouts:
            invalid_reasons.append(f"game watchdog timeouts={n_game_timeouts}")
        if n_trust_failures:
            invalid_reasons.append(f"trusted-search failures={n_trust_failures}")
        if n_cancelled:
            invalid_reasons.append(f"cancelled games={n_cancelled}")
        if n_baseline_failures:
            invalid_reasons.append(f"baseline failures={n_baseline_failures}")
        if n_our_failures:
            invalid_reasons.append(f"our runtime failures={n_our_failures}")
        if n_fail_closed:
            invalid_reasons.append(f"fail-closed games={n_fail_closed}")
        if n_zero_targets:
            invalid_reasons.append(f"zero-target games={n_zero_targets}")
        if n_written != n_expected_targets:
            invalid_reasons.append(
                f"complete records={n_written}/{n_expected_targets} "
                "training-eligible games"
            )
        if n != n_scheduled_jobs:
            invalid_reasons.append(f"valid games={n}/{n_scheduled_jobs} scheduled")
        if args.agent_mode == "oracle-mcts" and avg_sims < 1 and n > 0:
            invalid_reasons.append(f"average completed simulations={avg_sims:.3f}")
        if (
            args.agent_mode == "belief-mcts"
            and completed_search_sims < 128 * trusted_search_decisions
        ):
            invalid_reasons.append(
                "trusted belief search completed fewer than 128 simulations "
                f"per decision ({completed_search_sims}/{trusted_search_decisions})"
            )
        if args.agent_mode == "belief-mcts" and not coverage_complete:
            invalid_reasons.append(
                "required balanced opponent/seat minimum coverage incomplete"
            )
        if (
            args.agent_mode == "belief-mcts"
            and collection_contract.target_search_decisions > 0
            and trusted_search_decisions < collection_contract.target_search_decisions
        ):
            invalid_reasons.append(
                "maximum games reached before trusted decision target "
                f"({trusted_search_decisions}/"
                f"{collection_contract.target_search_decisions})"
            )
        if use_leaf_server and n_leaf_remote != n:
            invalid_reasons.append(f"remote-leaf games={n_leaf_remote}/{n}")
        for reason in invalid_reasons:
            report.mark_invalid(reason)
        summary = report.summary()
        heldout_rows = [
            row
            for row in summary.get("matchups", [])
            if row.get("opponent_id") in heldout_ids
        ]
        heldout_games = sum(int(row.get("games") or 0) for row in heldout_rows)
        heldout_score = sum(
            float(row.get("wins") or 0) + 0.5 * float(row.get("draws") or 0)
            for row in heldout_rows
        )
        summary["heldout_current_policy"] = {
            "opponents": sorted(heldout_ids),
            "games": heldout_games,
            "wins": sum(float(row.get("wins") or 0) for row in heldout_rows),
            "draws": sum(float(row.get("draws") or 0) for row in heldout_rows),
            "losses": sum(float(row.get("losses") or 0) for row in heldout_rows),
            "score_rate": (
                heldout_score / heldout_games if heldout_games else None
            ),
            "excluded_from_training": True,
        }
        summary["collection"] = {
            "contract": collection_contract.__dict__,
            "fresh_games": n,
            "training_eligible_games": n_written,
            "trusted_search_decisions": trusted_search_decisions,
            "decisions_per_game": trusted_search_decisions / max(n_written, 1),
            "completed_search_sims": completed_search_sims,
            "completed_sims_per_decision": completed_sims_per_trusted_decision,
            "max_complete_ordered_action_count": (
                max_complete_ordered_action_count
            ),
            "factorized_search_decisions": factorized_search_decisions,
            "actual_games_per_opponent": actual_games_per_opponent,
            "opponent_seat_coverage_complete": coverage_complete,
            "opponent_seat_counts": coverage_counts,
            "stop_reason": collection_stop_reason,
            "scheduled_games": n_scheduled_jobs,
            "max_games_not_scheduled": len(jobs) - n_scheduled_jobs,
        }
        if not summary["valid"]:
            fatal_health_error = (
                f"iter {it} invalid: " + "; ".join(
                    summary.get("invalid_reasons")
                    or [
                        f"missing={summary.get('missing_opponents')}",
                        f"insufficient={summary.get('insufficient_opponents')}",
                    ]
                )
            )
            tqdm.write(
                f"[rr] FATAL HEALTH GATE: {fatal_health_error}; rejecting "
                "the replay/eval and stopping before training or checkpointing.",
                file=sys.stderr,
            )
            quarantined = replay_writer.quarantine(
                f".invalid.{int(time.time())}"
            )
            eval_out = run_output_dir / "eval" / f"rr_iter{it:03d}.invalid.json"
            eval_out.parent.mkdir(parents=True, exist_ok=True)
            eval_out.write_text(
                json.dumps(
                    {
                        "iteration": it,
                        "valid": False,
                        "summary": summary,
                        "checkpoint": (loop_state.get("incumbent") or {}),
                    },
                    indent=2,
                )
                + "\n"
            )
            break

        replay_writer.finalize(iter_replay_path)

        # Experience was streamed to iter_replay_path during play (n_written
        # games). Trusted tuples = causal info-set history → behavior policy →
        # game-outcome value.
        wr_iter = wins / max(n, 1)
        tqdm.write(
            f"[rr] iter={it} games={n} wr={wr_iter:.1%} "
            f"experience={n_written} games→{iter_replay_path.name} "
            f"passing={summary['n_passing_draw_aware']}/{summary['n_evaluated']} "
            f"all_pass={summary['all_pass']}"
        )
        for row in summary["matchups"]:
            tqdm.write(
                f"   {row['opponent_id']:32} wr={row['wr']:.1%} "
                f"draw_aware_lo="
                f"{row['draw_aware_score_interval']['lower']:.1%} n={row['games']}"
            )

        played_identity = CheckpointIdentity.from_path(champion)
        expected_incumbent_digest = (loop_state.get("incumbent") or {}).get("digest")
        if (
            expected_incumbent_digest is not None
            and played_identity.digest != expected_incumbent_digest
        ):
            fatal_health_error = (
                "incumbent checkpoint changed after identity was recorded: "
                f"{played_identity.digest} != {expected_incumbent_digest}"
            )
            tqdm.write(f"[rr] ABORT: {fatal_health_error}", file=sys.stderr)
            break
        played_champion = played_identity.path
        loop_state["last_evaluated_champion"] = played_champion
        loop_state["last_evaluated_checkpoint"] = played_identity.as_dict()
        loop_state["seed"] = seed

        best_wr = float(loop_state.get("best_wr", -1.0))
        if wr_iter > best_wr:
            loop_state["best_wr"] = wr_iter
            loop_state["best_champion"] = played_champion

        # A formal field pass belongs to exactly the model that played these
        # games. Preserve that identity and do not train a successor first.
        field_gate_passed = bool(summary.get("all_pass")) and all(
            st.games >= 50 for st in report.matchups.values()
        )
        candidate_identity: CheckpointIdentity | None = None
        training_result: dict[str, Any] | None = None
        promotion_report: dict[str, Any] = {
            "passed": False,
            "valid": True,
            "skipped": True,
            "reason": "field gate already passed" if field_gate_passed else "training disabled",
            "parent": played_identity.as_dict(),
        }
        incumbent_after = played_identity
        stop_after_checkpoint = False

        # ---- Train a separate immutable candidate from the evaluated parent. ----
        bootstrap_jsonl = paths.DATA_DIR / "bootstrap" / f"{args.archetype}.jsonl"
        if args.train_epochs > 0 and n_written > 0 and not field_gate_passed:
            ds, rinfo = _load_recent_only_dataset(
                iter_replay_path,
                it,
                bootstrap_jsonl=bootstrap_jsonl,
                bootstrap_mix=args.bootstrap_mix,
                replay_dir=rl_dir,
                policy_anchor_dir=(
                    paths.DATA_DIR / "rl" / policy_anchor_lineage
                    if policy_anchor_lineage
                    else None
                ),
                history_mix=args.history_mix,
                replay_history_iters=args.replay_history_iters,
                seed=args.seed + it,
                replay_fraction=args.replay_fraction,
                policy_anchor_ratio=args.policy_anchor_ratio,
                max_decisions_per_game=args.max_decisions_per_game,
                expected_target_source=(
                    "belief_mcts"
                    if args.agent_mode == "belief-mcts"
                    else "history_policy"
                ),
            )
            tqdm.write(
                f"[rr] population replay: current={rinfo['n_recent_games']} "
                f"immutable_history={rinfo['n_history_games']}/"
                f"{rinfo['history_pool_games']} "
                f"(belief={rinfo['n_belief_history_games']} "
                f"explicit_policy_anchor={rinfo['n_policy_anchor_games']} "
                f"replay_fraction={rinfo['actual_replay_fraction_fresh_plus_history']:.1%}) "
                f"(+{rinfo['n_bootstrap_regularizer']} bootstrap regularizer games, "
                f"clearly separate; bootstrap_fraction={rinfo['bootstrap_fraction_total_games']:.1%}) "
                f"game_first=True decisions={rinfo['raw_decisions']}→"
                f"{rinfo['capped_decisions']} cap/game={rinfo['max_decisions_per_game']} "
                f"— up to {args.train_epochs} epoch(s) "
                f"(patience={args.patience} val_frac={args.val_frac:g}), "
                f"base={Path(played_champion).name}"
            )
            if rinfo["n_recent_games"] != n_written:
                fatal_health_error = (
                    f"dataset conversion kept {rinfo['n_recent_games']}/"
                    f"{n_written} completed records"
                )
                tqdm.write(f"[rr] ABORT: {fatal_health_error}", file=sys.stderr)
                break
            candidate_path = checkpoint.candidate_path(run_name, it)
            candidate_run_name = f"{run_name}_candidate_iter{it:06d}"
            training_result = rl_train_step(
                ds,
                base_ckpt=played_champion,
                out_run_name=candidate_run_name,
                archetype_id=args.archetype,
                epochs=args.train_epochs,
                device=train_dev,
                cfg=TrainConfig(
                    lr=args.train_lr,
                    epochs=args.train_epochs,
                    games_per_batch=16,
                    max_decisions_per_batch=4096,
                    val_frac=args.val_frac,
                    early_stop_patience=args.patience,
                    aux_loss_weight=float(args.aux_loss_weight),
                    opp_hand_loss_weight=float(args.opp_hand_loss_weight),
                    opp_remainder_loss_weight=float(args.opp_remainder_loss_weight),
                    lethal_threat_loss_weight=float(args.lethal_threat_loss_weight),
                    prize_race_loss_weight=float(args.prize_race_loss_weight),
                    seed=args.seed + it,
                ),
                seed=args.seed + it,
                output_path=candidate_path,
                parent_digest=played_identity.digest,
                training_provenance={
                    **loop_state["run_provenance"],
                    "replay_composition": rinfo,
                    "parent": played_identity.as_dict(),
                },
            )
            if stop_requested["signal"] is not None:
                interrupted_candidate = Path(
                    training_result["candidate_path"]
                )
                if interrupted_candidate.exists():
                    os.replace(
                        interrupted_candidate,
                        interrupted_candidate.with_name(
                            interrupted_candidate.name
                            + f".invalid.interrupted.{int(time.time())}"
                        ),
                    )
                if iter_replay_path.exists():
                    os.replace(
                        iter_replay_path,
                        iter_replay_path.with_name(
                            iter_replay_path.name
                            + f".invalid.interrupted.{int(time.time())}"
                        ),
                    )
                print(
                    f"[rr] iteration {it} interrupted after training; "
                    "candidate/replay retained as invalid; no promotion/checkpoint",
                    flush=True,
                )
                break
            candidate_identity = CheckpointIdentity.from_path(
                training_result["candidate_path"]
            )
            if candidate_identity.digest != training_result.get("candidate_digest"):
                raise RuntimeError("candidate digest changed immediately after save")
            m = training_result.get("metrics") or {}
            vm = training_result.get("validation_metrics") or {}
            epochs_ran = training_result.get("epochs_ran", args.train_epochs)
            tqdm.write(
                f"[rr] immutable candidate={Path(candidate_identity.path).name} "
                f"digest={candidate_identity.digest[:23]} "
                f"(epochs_ran={epochs_ran}/{args.train_epochs} "
                f"policy_loss={m.get('policy_loss', float('nan')):.3f} "
                f"policy_kl={m.get('policy_kl', float('nan')):.3f} "
                f"value_loss={m.get('value_loss', float('nan')):.3f} "
                f"aux_loss={m.get('aux_loss', float('nan')):.3f} "
                f"opp_hand_loss={m.get('opp_hand_loss', float('nan')):.3f} "
                f"opp_remainder_loss={m.get('opp_remainder_loss', float('nan')):.3f} "
                f"lethal_loss={m.get('lethal_threat_loss', float('nan')):.3f} "
                f"prize_race_loss={m.get('prize_race_loss', float('nan')):.3f} "
                f"target_value_mean={m.get('target_value_mean', float('nan')):.3f} "
                f"relative_update={training_result.get('relative_update_norm_l2', float('nan')):.3g} "
                f"validation={training_result.get('validation_source')}:"
                f"{vm.get('total_loss', float('nan')):.3f})"
            )

            # Direct, independent seat-stratified candidate-vs-parent gate. No server is reloaded
            # until this evaluation has completed and passed.
            sequential_looks = 1 + max(
                0,
                (
                    int(args.promotion_max_games) - promotion_cfg.min_games
                )
                // int(args.promotion_batch_games),
            )
            sequential_promotion_cfg = PromotionGateConfig(
                min_games=promotion_cfg.min_games,
                min_complete_pairs=promotion_cfg.min_complete_pairs,
                threshold=promotion_cfg.threshold,
                confidence=1.0
                - (1.0 - promotion_cfg.confidence) / sequential_looks,
                bootstrap_resamples=promotion_cfg.bootstrap_resamples,
            )
            promotion_jobs: list[dict[str, Any]] = []
            promotion_seed = int(seed) + 10_000_000
            for game_i in range(int(args.promotion_max_games)):
                promotion_jobs.append(
                    {
                        "candidate_checkpoint": candidate_identity.path,
                        "parent_checkpoint": played_identity.path,
                        "candidate_seat": game_i % 2,
                        "pair_id": None,
                        "cluster_id": game_i // 2,
                        "seed": promotion_seed + game_i,
                        "deck": our_deck,
                        "device": (
                            str(train_dev)
                            if args.agent_mode == "belief-mcts"
                            else "cpu"
                        ),
                        "agent_mode": args.agent_mode,
                        "mcts_sims": int(args.promotion_mcts_sims),
                        "mcts_move_time": float(args.promotion_move_time),
                        "expected_search_decisions": int(
                            args.expected_search_decisions
                        ),
                        "candidate_digest": candidate_identity.digest,
                        "parent_digest": played_identity.digest,
                        "model_generation": it + 1,
                        "preserve_stdout": (
                            args.agent_mode == "belief-mcts"
                        ),
                        "timeout_s": int(args.game_timeout_s),
                    }
                )
            promotion_results: list[dict[str, Any]] = []
            promotion_stop_reason = "max_games"
            def _run_promotion_batches(map_batch) -> None:
                nonlocal promotion_stop_reason
                promotion_bar = tqdm(
                    total=len(promotion_jobs),
                    desc=f"iter{it} promotion",
                    unit="game",
                    leave=False,
                )
                for batch_start in range(
                    0,
                    len(promotion_jobs),
                    int(args.promotion_batch_games),
                ):
                    batch_end = min(
                        len(promotion_jobs),
                        batch_start + int(args.promotion_batch_games),
                    )
                    for result_row in map_batch(
                        promotion_jobs[batch_start:batch_end]
                    ):
                        promotion_results.append(result_row)
                        promotion_bar.update(1)
                    if len(promotion_results) < promotion_cfg.min_games:
                        continue
                    interim = evaluate_candidate_gate(
                        promotion_results, sequential_promotion_cfg
                    )
                    if not interim["valid"]:
                        promotion_stop_reason = "invalid_sample"
                        break
                    if interim["interval_lower"] > sequential_promotion_cfg.threshold:
                        promotion_stop_reason = "sequential_accept"
                        break
                    if interim["interval_upper"] < sequential_promotion_cfg.threshold:
                        promotion_stop_reason = "sequential_reject"
                        break
                promotion_bar.close()
            if args.agent_mode == "belief-mcts":
                # Run in the CUDA-capable parent. Simulation workers are
                # intentionally GPU-masked; spawning a CUDA promotion pool
                # would violate that isolation and duplicate model contexts.
                _run_promotion_batches(
                    lambda batch: (_worker_promotion(job) for job in batch)
                )
            else:
                with WorkerPool(
                    num_workers=max(
                        1,
                        min(int(args.promotion_workers), len(promotion_jobs)),
                    )
                ) as promotion_pool:
                    _run_promotion_batches(
                        lambda batch: promotion_pool.imap_unordered(
                            _worker_promotion, batch
                        )
                    )
            promotion_report = evaluate_candidate_gate(
                promotion_results, sequential_promotion_cfg
            )
            promotion_report["sequential_stop_reason"] = promotion_stop_reason
            promotion_report["max_games"] = int(args.promotion_max_games)
            promotion_report["batch_games"] = int(args.promotion_batch_games)
            promotion_report["sequential_looks"] = sequential_looks
            promotion_report["overall_confidence"] = promotion_cfg.confidence
            promotion_report["per_look_confidence"] = (
                sequential_promotion_cfg.confidence
            )
            promotion_report["confidence_spending"] = "bonferroni"
            promotion_report["collection_games_included"] = 0
            promotion_report["candidate_incumbent_budgets_identical"] = True
            promotion_report["agent_mode"] = args.agent_mode
            promotion_report["mcts_sims_each"] = int(args.promotion_mcts_sims)
            promotion_report["move_time_each"] = float(args.promotion_move_time)
            promotion_report["candidate"] = candidate_identity.as_dict()
            promotion_report["parent"] = played_identity.as_dict()
            promotion_report["skipped"] = False
            if promotion_report["passed"]:
                incumbent_after = candidate_identity
                tqdm.write(
                    f"[rr] PROMOTED candidate score={promotion_report['wr']:.1%} "
                    f"draw_aware_lo={promotion_report['interval_lower']:.1%} "
                    f"per_seat={promotion_report['games_per_seat']}"
                )
            else:
                tqdm.write(
                    f"[rr] REJECTED candidate valid={promotion_report['valid']} "
                    f"wr={promotion_report.get('wr')} "
                    f"draw_aware_lo={promotion_report.get('interval_lower')} — "
                    f"incumbent remains {Path(played_identity.path).name}"
                )

            # Reload only the evaluated/promoted candidate, with digest/version
            # acknowledgement from every replica.
            if promotion_report["passed"] and leaf_ctrl_qs:
                requested_version = leaf_version + 1
                for cq in leaf_ctrl_qs:
                    cq.put(
                        {
                            "cmd": "reload",
                            "path": candidate_identity.path,
                            "digest": candidate_identity.digest,
                            "version": requested_version,
                        }
                    )
                reload_ok = True
                reload_statuses = []
                for j, sq in enumerate(leaf_status_qs):
                    try:
                        status = sq.get(timeout=240)
                    except Exception as exc:
                        status = {"ok": False, "error": str(exc)}
                    reload_statuses.append(status)
                    if (
                        status.get("type") != "reload"
                        or not status.get("ok")
                        or status.get("checkpoint_digest") != candidate_identity.digest
                        or int(status.get("version", -1)) != requested_version
                        or not leaf_servers[j].is_alive()
                    ):
                        reload_ok = False
                promotion_report["reload_acknowledgements"] = reload_statuses
                if reload_ok:
                    leaf_version = requested_version
                    leaf_digest = candidate_identity.digest
                    if remote_channel is not None:
                        remote_channel["expected_digest"] = leaf_digest
                        remote_channel["expected_version"] = leaf_version
                    tqdm.write(
                        f"[rr] leaf reload acknowledged x{len(leaf_servers)} "
                        f"version={leaf_version} digest={leaf_digest[:23]}"
                    )
                else:
                    # Keep the parent as the deployable incumbent. Some replicas
                    # may have reloaded, so stop after checkpoint and restart the
                    # server set cleanly on resume.
                    incumbent_after = played_identity
                    promotion_report["passed"] = False
                    promotion_report["deployment_failed"] = True
                    stop_after_checkpoint = True
                    fatal_health_error = "leaf reload acknowledgement failed"
        elif args.train_epochs <= 0:
            tqdm.write(
                f"[rr] train_epochs=0 — skip training "
                f"(experience still written: n_written={n_written})"
            )
        else:
            tqdm.write(
                f"[rr] no experience this iter (n_written={n_written}) — "
                f"champion unchanged"
            )

        if stop_requested["signal"] is not None:
            if candidate_identity is not None:
                candidate_path = Path(candidate_identity.path)
                if candidate_path.exists():
                    os.replace(
                        candidate_path,
                        candidate_path.with_name(
                            candidate_path.name
                            + f".invalid.interrupted.{int(time.time())}"
                        ),
                    )
            if iter_replay_path.exists():
                os.replace(
                    iter_replay_path,
                    iter_replay_path.with_name(
                        iter_replay_path.name
                        + f".invalid.interrupted.{int(time.time())}"
                    ),
                )
            print(
                f"[rr] iteration {it} interrupted before parent checkpoint; "
                "artifacts retained as invalid",
                flush=True,
            )
            break

        loop_state["champion"] = incumbent_after.path
        loop_state["incumbent"] = incumbent_after.as_dict()
        loop_state["last_completed_iteration"] = it
        history_row = {
            "iteration": it,
            "completed": True,
            "summary": summary,
            "t": time.time(),
            "wr": wr_iter,
            "champion": played_champion,
            "evaluated_checkpoint": played_identity.as_dict(),
            "candidate": (
                candidate_identity.as_dict() if candidate_identity is not None else None
            ),
            "promotion": promotion_report,
            "incumbent_after": incumbent_after.as_dict(),
            "training": training_result,
        }
        loop_state["history"].append(history_row)
        if summary.get("all_pass"):
            loop_state["formal_best"] = played_identity.as_dict()

        # Save loop checkpoint using only the evaluated incumbent.
        model = load_model_from_checkpoint(
            incumbent_after.path, device=torch.device("cpu")
        )
        ckpt = checkpoint.build_checkpoint(
            model=model,
            step=it,
            epoch=it,
            rl_iteration=it,
            best_metric=summary.get("n_passing_draw_aware"),
            model_config=model.cfg,
            archetype_id=args.archetype,
            model_id=run_name,
            extra={
                "loop_state": loop_state,
                "last_eval": summary,
                "incumbent": incumbent_after.as_dict(),
            },
        )
        saved = mgr.save(ckpt, is_best=False)
        if summary.get("all_pass"):
            # .best.pt is a byte-for-byte copy of the field-evaluated model,
            # never the just-trained successor. The digest-addressed source is
            # retained permanently.
            formal_source = _snapshot_best_champion(champion, run_name)
            best_dest = checkpoint.best_path(run_name)
            tmp_best = best_dest.with_name(best_dest.name + f".tmp.{os.getpid()}")
            try:
                shutil.copy2(formal_source, tmp_best)
                os.replace(tmp_best, best_dest)
            finally:
                tmp_best.unlink(missing_ok=True)
        tqdm.write(f"[rr] checkpoint → {saved}")

        eval_out = run_output_dir / "eval" / f"rr_iter{it:03d}.json"
        eval_out.parent.mkdir(parents=True, exist_ok=True)
        eval_out.write_text(
            json.dumps(
                {
                    "loop": loop_state,
                    "summary": summary,
                    "evaluated_checkpoint": played_identity.as_dict(),
                    "candidate": (
                        candidate_identity.as_dict()
                        if candidate_identity is not None
                        else None
                    ),
                    "promotion": promotion_report,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )

        if stop_after_checkpoint:
            tqdm.write(
                "[rr] stopping after safe parent checkpoint because leaf reload failed",
                file=sys.stderr,
            )
            break

        if stop_requested["signal"] is not None:
            tqdm.write(
                f"[rr] boundary stop complete after iter {it} "
                f"(signal={stop_requested['signal']})"
            )
            break
        if summary.get("all_pass") and all(
            st.games >= 50 for st in report.matchups.values()
        ):
            tqdm.write("[rr] GATE PASSED (wilson≥threshold with ≥50 games/opp) — stop for submit prep")
            break
        if summary.get("all_pass"):
            tqdm.write(
                "[rr] all opponents above draw-aware gate at current N — "
                "increase --games-per-opp (≥100) before Kaggle submit"
            )

    # Shut all leaf server replicas down cleanly.
    for j, proc in enumerate(leaf_servers):
        try:
            leaf_ctrl_qs[j].put_nowait({"cmd": "stop"})
        except Exception:
            pass
        try:
            leaf_req_qs[j].put_nowait(None)
        except Exception:
            pass
    for proc in leaf_servers:
        try:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        except Exception:
            pass
    for queue_obj in (
        *leaf_req_qs,
        *leaf_ctrl_qs,
        *leaf_status_qs,
        *leaf_resp_qs,
    ):
        try:
            queue_obj.cancel_join_thread()
        except Exception:
            pass
        try:
            queue_obj.close()
        except Exception:
            pass

    if stop_requested["signal"] is not None:
        print(
            f">> interrupted_signal={stop_requested['signal']} "
            "no_inflight_iteration_checkpointed=true",
            flush=True,
        )
        return 128 + int(stop_requested["signal"])
    if fatal_health_error is not None:
        print(f">> aborted={fatal_health_error}", flush=True)
        return 2

    print(f">> champion={loop_state.get('champion')}", flush=True)
    print(f">> replay_buffer={rl_replay_dir}", flush=True)
    print(f">> best_wr={loop_state.get('best_wr')} best_champion={loop_state.get('best_champion')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
