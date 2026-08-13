"""Pickle-safe entry points for remote WorkerPool job dispatch.

``scripts/run_remote_worker.py`` loads ``train_round_robin`` via importlib under the
synthetic module name ``train_round_robin_remote``. Multiprocessing (spawn) cannot
unpickle those callables in child workers — they raise
``ModuleNotFoundError: train_round_robin_remote``, die, leak leaf response slots,
and leave the farm stuck at ``jobs_completed=0`` while hello/health still pass.

These wrappers live in an importable package module so ``Pool.apply`` round-trips
cleanly under spawn.
"""

from __future__ import annotations

import gc
import importlib.util
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

_RR = None
_MODULE_NAME = "train_round_robin_remote"
_CONTROLLER_MATCHUP_RUNTIME_KEY = "_controller_matchup_runtime"
_MULTI_ENV_CACHE_FAMILIES = frozenset(
    {
        # ``multi_env`` remains accepted for direct/legacy callers.
        "multi_env",
        # The two runtime transports use incompatible cache-key forms:
        # raw paths for self-play and pipe-delimited paths for public play.
        "multi_env_self_play",
        "multi_env_play",
    }
)
_CHECKPOINT_SCOPED_WORKER_CACHE_KEYS = (
    # Single-game public/self-play local-parent caches.
    "key",
    "model",
    "self_play_key",
    "self_play_us",
    "self_play_them",
    # Promotion keeps two local parents while a packet is active.
    "promotion_key",
    "candidate_model",
    "parent_model",
)


def _checkpoint_scope_tokens(jobs: list[dict[str, Any]]) -> tuple[str, ...]:
    """Return the immutable model identities reachable from one worker task.

    Remote children live through many trainer checkpoint reloads.  Each task is
    stamped by the controller, so these fields are the safe boundary at which
    a child may drop model-keyed caches from an earlier champion.  Include a
    path when a legacy caller lacks a digest; that still prevents an unbounded
    cache keyed by successive checkpoint filenames.
    """

    tokens: set[str] = set()
    for job in jobs:
        for key in (
            "checkpoint_digest",
            "checkpoint",
            "opponent_checkpoint_digest",
            "opponent_checkpoint",
            "candidate_digest",
            "candidate_checkpoint",
            "parent_digest",
            "parent_checkpoint",
        ):
            value = job.get(key)
            if value is not None and str(value):
                tokens.add(f"{key}={value}")
    return tuple(sorted(tokens))


def _primary_checkpoint_identity(jobs: list[dict[str, Any]]) -> tuple[str, ...]:
    """Return the controller-stamped active checkpoint identity for a packet."""

    tokens: set[str] = set()
    for job in jobs:
        for key in ("checkpoint_digest", "checkpoint"):
            value = job.get(key)
            if value is not None and str(value):
                tokens.add(f"{key}={value}")
    if not tokens:
        # Promotion has two explicit parents rather than a server-stamped
        # ``checkpoint`` field. Its candidate is the active model identity for
        # cache invalidation, so a same-path/new-digest promotion cannot reuse
        # an old in-process candidate.
        for job in jobs:
            for key in ("candidate_digest", "candidate_checkpoint"):
                value = job.get(key)
                if value is not None and str(value):
                    tokens.add(f"{key}={value}")
    return tuple(sorted(tokens))


def _required_checkpoint_paths(jobs: list[dict[str, Any]]) -> set[str]:
    """Return models a current packet can legitimately reference."""

    paths: set[str] = set()
    for job in jobs:
        for key in (
            "checkpoint",
            "opponent_checkpoint",
            "candidate_checkpoint",
            "parent_checkpoint",
        ):
            value = job.get(key)
            if value is not None and str(value):
                paths.add(str(value))
    return paths


def _cache_key_mentions_path(key: object, path: str) -> bool:
    """Match the established raw and pipe-delimited worker cache key forms."""

    raw = str(key)
    return raw == path or raw.startswith(f"{path}|") or f"|{path}|" in raw


def _drop_state_entries(state: dict[str, Any], keys: tuple[str, ...]) -> int:
    removed = 0
    for key in keys:
        if key in state:
            state.pop(key, None)
            removed += 1
    return removed


def _is_multi_env_cache_family(cache_family: str) -> bool:
    return str(cache_family) in _MULTI_ENV_CACHE_FAMILIES


def _prune_multi_env_model_cache(
    state: dict[str, Any],
    *,
    required_paths: set[str],
) -> int:
    """Retain only current-packet models in the multi-env cache."""

    model_cache = state.get("multi_env_models")
    if not isinstance(model_cache, dict):
        return 0
    evicted = 0
    for key in list(model_cache):
        if not any(_cache_key_mentions_path(key, path) for path in required_paths):
            model_cache.pop(key, None)
            evicted += 1
    return evicted


def _drop_multi_env_model_cache(state: dict[str, Any]) -> int:
    """Release every multi-env model when another task family becomes active."""

    model_cache = state.get("multi_env_models")
    if not isinstance(model_cache, dict):
        return 0
    evicted = len(model_cache)
    model_cache.clear()
    return evicted


def _prune_single_game_model_caches(
    state: dict[str, Any],
    *,
    required_paths: set[str],
) -> int:
    """Prune single-game/promotion caches without disrupting a current packet."""

    evicted = 0
    key = str(state.get("key") or "")
    if key and not any(_cache_key_mentions_path(key, path) for path in required_paths):
        evicted += _drop_state_entries(state, ("key", "model"))

    self_key = str(state.get("self_play_key") or "")
    if self_key:
        parts = self_key.split("|")
        # Established form: self|us_checkpoint|them_checkpoint|device|mode.
        keep_self = (
            len(parts) >= 3
            and parts[0] == "self"
            and parts[1] in required_paths
            and parts[2] in required_paths
        )
        if not keep_self:
            evicted += _drop_state_entries(
                state,
                ("self_play_key", "self_play_us", "self_play_them"),
            )

    promotion_key = str(state.get("promotion_key") or "")
    if promotion_key:
        parts = promotion_key.split("|")
        # Established form: promotion|candidate_checkpoint|parent_checkpoint|device.
        keep_promotion = (
            len(parts) >= 3
            and parts[0] == "promotion"
            and parts[1] in required_paths
            and parts[2] in required_paths
        )
        if not keep_promotion:
            evicted += _drop_state_entries(
                state,
                ("promotion_key", "candidate_model", "parent_model"),
            )
    return evicted


def _drop_inactive_worker_cache_families(
    state: dict[str, Any],
    *,
    active_family: str,
) -> int:
    """Prevent one child from retaining duplicate models across job transports."""

    evicted = 0
    if not _is_multi_env_cache_family(active_family):
        evicted += _drop_multi_env_model_cache(state)
    if active_family != "play":
        evicted += _drop_state_entries(state, ("key", "model"))
    if active_family != "self_play":
        evicted += _drop_state_entries(
            state,
            ("self_play_key", "self_play_us", "self_play_them"),
        )
    if active_family != "promotion":
        evicted += _drop_state_entries(
            state,
            ("promotion_key", "candidate_model", "parent_model"),
        )
    return evicted


def _advance_remote_worker_checkpoint_scope(
    state: dict[str, Any],
    jobs: list[dict[str, Any]],
    *,
    active_cache_family: str,
) -> bool:
    """Bound a long-lived child's model caches to its current checkpoint scope.

    ``_WORKER_STATE`` is intentionally process-local and persists across all
    remote requests. Prior multi-env code stored models under every seen
    checkpoint key, and the single-game/multi-env paths retained duplicate
    copies of the same parent. Keep exactly one active task-family cache and,
    for a packed task, only its current parent plus models reachable by that
    packet. A fresh controller checkpoint identity flushes all path-keyed
    entries before the new weights can be loaded.
    """

    scope = _checkpoint_scope_tokens(jobs)
    if not scope:
        # Direct unit callers may omit an identity.  Do not evict a known good
        # cache merely because that legacy test/job cannot prove a replacement.
        return False
    previous_raw = state.get("_remote_worker_checkpoint_scope")
    previous = (
        tuple(str(value) for value in previous_raw)
        if isinstance(previous_raw, (list, tuple))
        else ()
    )
    primary = _primary_checkpoint_identity(jobs)
    previous_primary_raw = state.get("_remote_worker_primary_checkpoint_identity")
    previous_primary = (
        tuple(str(value) for value in previous_primary_raw)
        if isinstance(previous_primary_raw, (list, tuple))
        else ()
    )
    scope_changed = previous != scope
    previous_family = str(state.get("_remote_worker_cache_family") or "")
    family_changed = previous_family != str(active_cache_family)
    if not scope_changed and not family_changed:
        return False

    required_paths = _required_checkpoint_paths(jobs)
    if primary != previous_primary:
        # A reloaded active digest may legitimately reuse a stable filename.
        # Path-only cache keys cannot prove byte identity, so drop all model
        # references at that checkpoint boundary before loading the new one.
        model_cache = state.get("multi_env_models")
        evicted = len(model_cache) if isinstance(model_cache, dict) else 0
        if isinstance(model_cache, dict):
            model_cache.clear()
        evicted += _drop_state_entries(state, _CHECKPOINT_SCOPED_WORKER_CACHE_KEYS)
    else:
        evicted = _drop_inactive_worker_cache_families(
            state,
            active_family=active_cache_family,
        )
        if family_changed and _is_multi_env_cache_family(active_cache_family):
            # ``run_self_play_multi`` caches raw checkpoint paths whereas
            # ``run_play_multi`` decorates its key with device/RTP state.
            # Never leave either representation resident when the transport
            # switches; the next task loads exactly the representation it uses.
            evicted += _drop_multi_env_model_cache(state)
        # Same champion, different self-play opponent packet: retain only the
        # active current model plus the at-most-four opponent models reachable
        # from this LibcgMultiEnv packet.  This is the bounded current+needed
        # policy; it does not reload the active parent on every packet.
        if _is_multi_env_cache_family(active_cache_family):
            evicted += _prune_multi_env_model_cache(
                state,
                required_paths=required_paths,
            )
        else:
            evicted += _prune_single_game_model_caches(
                state,
                required_paths=required_paths,
            )
    state["_remote_worker_checkpoint_scope"] = scope
    state["_remote_worker_primary_checkpoint_identity"] = primary
    state["_remote_worker_cache_family"] = str(active_cache_family)
    lifecycle = state.setdefault("_remote_worker_memory_lifecycle", {})
    if isinstance(lifecycle, dict):
        lifecycle["scope_rotations"] = int(lifecycle.get("scope_rotations", 0)) + 1
        lifecycle["cache_entries_evicted"] = int(
            lifecycle.get("cache_entries_evicted", 0)
        ) + int(evicted)
        if family_changed:
            lifecycle["cache_family_transitions"] = int(
                lifecycle.get("cache_family_transitions", 0)
            ) + 1
    # The removed values can include model↔agent cycles created by RTP bridges.
    # Collect at the receipt-safe task boundary, never while a game is active.
    gc.collect()
    return True


def _collect_remote_worker_game_cycles(state: dict[str, Any]) -> int:
    """Collect unreachable per-game RTP/agent cycles after result serialization."""

    collected = int(gc.collect())
    lifecycle = state.setdefault("_remote_worker_memory_lifecycle", {})
    if isinstance(lifecycle, dict):
        lifecycle["post_job_gc_runs"] = int(lifecycle.get("post_job_gc_runs", 0)) + 1
        lifecycle["post_job_gc_collected"] = int(
            lifecycle.get("post_job_gc_collected", 0)
        ) + collected
    return collected


def _prepare_remote_worker_job_scope(
    jobs: list[dict[str, Any]],
    *,
    cache_family: str,
) -> dict[str, Any]:
    """Return the long-lived child state after safe checkpoint-scope pruning."""

    state = load_round_robin_module()._WORKER_STATE
    _advance_remote_worker_checkpoint_scope(
        state,
        jobs,
        active_cache_family=cache_family,
    )
    return state


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_round_robin_module():
    """Load ``scripts/train_round_robin.py`` once per process (lazy)."""
    global _RR
    if _RR is not None:
        return _RR
    existing = sys.modules.get(_MODULE_NAME)
    if existing is not None:
        _RR = existing
        return _RR
    path = _repo_root() / "scripts" / "train_round_robin.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    _RR = mod
    return _RR


def _bind_controller_matchup_runtime(job: dict[str, Any]) -> None:
    """Reassert the controller-owned routing contract before every remote job.

    WorkerPool children are intentionally recycled.  They can also execute
    frozen specialist opponents whose submission packages use the same
    process-global environment variables as the active candidate.  Binding
    from the server-injected contract at each job boundary prevents either
    lifecycle from carrying an opponent's tree into the next candidate game.
    The server overwrites this reserved field, so clients cannot select a tree.
    """

    if _CONTROLLER_MATCHUP_RUNTIME_KEY not in job:
        # Compatibility for direct unit tests and non-server callers.
        return
    runtime = job.get(_CONTROLLER_MATCHUP_RUNTIME_KEY)
    if runtime is None:
        os.environ.pop("POKEBOT_MATCHUP_ADAPTER_RUNTIME", None)
        os.environ.pop("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", None)
        os.environ.pop("POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE", None)
        return
    if not isinstance(runtime, dict):
        raise ValueError("controller matchup runtime contract must be an object")
    tree_path = Path(str(runtime.get("tree") or "")).expanduser().resolve()
    expected_digest = str(runtime.get("tree_digest") or "")
    expected_ids = sorted(
        str(value) for value in runtime.get("accepted_archetype_ids") or ()
    )
    if not tree_path.is_file() or not expected_digest or not expected_ids:
        raise ValueError("controller matchup runtime contract is incomplete")
    raw = tree_path.read_bytes()
    actual_digest = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    if actual_digest != expected_digest:
        raise ValueError(
            "controller matchup runtime tree digest changed: "
            f"expected={expected_digest} actual={actual_digest}"
        )
    payload = json.loads(raw)
    actual_ids = sorted(
        str(value)
        for value in dict(payload.get("runtime_contract") or {}).get(
            "accepted_archetype_ids", ()
        )
    )
    if actual_ids != expected_ids:
        raise ValueError(
            "controller matchup runtime accepted-route roster changed"
        )
    os.environ["POKEBOT_MATCHUP_ADAPTER_RUNTIME"] = "1"
    os.environ["POKEBOT_PUBLIC_MATCHUP_TREE_PATH"] = str(tree_path)
    os.environ["POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE"] = "runtime"


def remote_play_job(job: dict[str, Any]) -> dict[str, Any]:
    """Whole-game collection job (importable; safe to ``Pool.apply``)."""
    _bind_controller_matchup_runtime(job)
    state = _prepare_remote_worker_job_scope([job], cache_family="play")
    try:
        return load_round_robin_module()._worker_play(job)
    finally:
        _collect_remote_worker_game_cycles(state)


def remote_promotion_job(job: dict[str, Any]) -> dict[str, Any]:
    """Promotion evaluation job (importable; safe to ``Pool.apply``)."""
    _bind_controller_matchup_runtime(job)
    state = _prepare_remote_worker_job_scope([job], cache_family="promotion")
    try:
        return load_round_robin_module()._worker_promotion(job)
    finally:
        _collect_remote_worker_game_cycles(state)


def remote_matchup_runtime_probe(_job: dict[str, Any]) -> dict[str, Any]:
    """Report the routing identity actually inherited by one pool child.

    Controller hello metadata lives in the parent process.  A hot marker
    update can therefore make hello look current while long-lived simulator
    children still use the prior tree.  This probe crosses the process-pool
    boundary so the trainer can fail before collecting a mixed-routing shard.
    """

    _bind_controller_matchup_runtime(_job)

    runtime_enabled = os.environ.get(
        "POKEBOT_MATCHUP_ADAPTER_RUNTIME", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    tree_path = Path(
        os.environ.get("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", "")
    ).expanduser()
    if not runtime_enabled or not tree_path.is_file():
        return {
            "runtime_probe": {
                "runtime_enabled": runtime_enabled,
                "tree": str(tree_path),
                "tree_digest": None,
                "accepted_archetype_ids": [],
            },
            "error": None,
        }
    payload = json.loads(tree_path.read_text(encoding="utf-8"))
    accepted = sorted(
        str(value)
        for value in dict(payload.get("runtime_contract") or {}).get(
            "accepted_archetype_ids", ()
        )
    )
    digest = hashlib.sha256(tree_path.read_bytes()).hexdigest()
    return {
        "runtime_probe": {
            "runtime_enabled": True,
            "tree": str(tree_path.resolve()),
            "tree_digest": f"sha256:{digest}",
            "accepted_archetype_ids": accepted,
        },
        "error": None,
    }


def _remote_self_play_multi_job_impl(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Batch self-play or r182-authorized public play via ``LibcgMultiEnv``.

    ``batch`` is ``{"jobs": [job, ...]}``. Self-play jobs use NN vs NN;
    public jobs use our policy vs package opponents only after every child
    passes the immutable r182 id+digest+group+provenance allowlist.  Unrecognized
    and legacy packages stay on the ordinary one-game ``play`` transport.
    """
    jobs = list(batch.get("jobs") or [])
    if not jobs:
        return []
    if not 1 <= len(jobs) <= 4:
        raise ValueError("self_play_multi packet must contain 1..4 children")
    play_flags = [
        isinstance(job, dict) and isinstance(job.get("spec"), dict)
        for job in jobs
    ]
    if any(play_flags) and not all(play_flags):
        raise ValueError("self_play_multi batch mixes play and self-play jobs")
    play_jobs = all(play_flags)
    if play_jobs and not 2 <= len(jobs) <= 4:
        # A public packet is the exact r182 LibcgMultiEnv transport unit, not
        # a generic batch envelope.  Keep singleton public jobs on ordinary
        # ``play`` and reject any oversized forged packet before importing or
        # entering the shared multi-env process.  Pure self-play intentionally
        # retains its established generic batch behavior below.
        raise ValueError(
            "public self_play_multi packet must contain 2..4 children"
        )
    from poke_bot.pure_rl.multi_env_self_play import (
        run_play_multi,
        run_self_play_multi,
    )

    # Match single-game remote_self_play_job: privileged exact hidden zones
    # require the training-fork libcg. Official competition libcg on Bert/Elmo
    # has no GetHiddenSnapshot; keep packing and leave belief labels masked.
    if not os.environ.get("POKEBOT_LIBCG_PATH", "").strip():
        jobs = [
            {**job, "collect_privileged_belief": False}
            if isinstance(job, dict)
            else job
            for job in jobs
        ]
    if play_jobs:
        # Defense in depth: client-side batching is default-deny, but a remote
        # endpoint must independently reject a forged/mixed packet before it
        # can reach the shared LibcgMultiEnv process.
        from poke_bot.public_multi_env_safety import (
            require_public_multi_env_safe_job,
        )

        for job in jobs:
            require_public_multi_env_safe_job(job)
        return run_play_multi(jobs)
    if len(jobs) == 1:
        return [_remote_self_play_job_impl(jobs[0])]
    return run_self_play_multi(jobs)


def _remote_self_play_cache_family(job: dict[str, Any]) -> str:
    """Return the cache family used by one self-play job's actual code path."""

    if bool(job.get("collect_privileged_belief", False)) and os.environ.get(
        "POKEBOT_LIBCG_PATH", ""
    ).strip():
        return "multi_env_self_play"
    return "self_play"


def _remote_self_play_multi_cache_family(jobs: list[dict[str, Any]]) -> str:
    """Return the cache family used by a validated multi-endpoint packet.

    The endpoint also accepts a one-game self-play tail.  Unless that tail
    explicitly needs the private multi-env hidden-zone path, it runs through
    ``_remote_self_play_job_impl`` and therefore owns the ordinary self-play
    cache, not ``multi_env_models``.  Classifying the actual execution path
    keeps a prior packed task from retaining a duplicate H10 model.
    """

    play_flags = [isinstance(job.get("spec"), dict) for job in jobs]
    if any(play_flags) and not all(play_flags):
        raise ValueError("self_play_multi batch mixes play and self-play jobs")
    if all(play_flags):
        if not 2 <= len(jobs) <= 4:
            raise ValueError(
                "public self_play_multi packet must contain 2..4 children"
            )
        return "multi_env_play"
    if len(jobs) != 1:
        return "multi_env_self_play"
    return _remote_self_play_cache_family(jobs[0])


def remote_self_play_multi_job(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Run one bounded r182 LibcgMultiEnv packet with post-result cleanup."""

    jobs = list(batch.get("jobs") or [])
    if not jobs:
        return []
    if not all(isinstance(job, dict) for job in jobs):
        raise ValueError("self_play_multi job.jobs entries must be objects")
    if not 1 <= len(jobs) <= 4:
        raise ValueError("self_play_multi packet must contain 1..4 children")
    cache_family = _remote_self_play_multi_cache_family(jobs)
    _bind_controller_matchup_runtime(jobs[0])
    state = _prepare_remote_worker_job_scope(jobs, cache_family=cache_family)
    try:
        return _remote_self_play_multi_job_impl({**batch, "jobs": jobs})
    finally:
        _collect_remote_worker_game_cycles(state)


def _remote_self_play_job_impl(job: dict[str, Any]) -> dict[str, Any]:
    """Pure self-play: current policy vs recent-self / itself (spawn-safe).

    Primary Stage A collect path (Abhyuday: millions of self-play variations).
    Builds a trusted record from our seat only — no baseline heuristics.

    When the WorkerPool registered a leaf channel, both seats use coalesced
    GPU leaves for same-checkpoint games (``gpu-leaf-both``); recent-self
    opponents stay CPU-local on the opp seat (``gpu-leaf-us-only``).
    """
    # Exact hidden-zone targets require the explicit private training engine.
    # Route even a single game through the multi-handle adapter when that ABI
    # is configured; hosts using the official engine continue normally and
    # their unavailable belief labels are masked during training.
    if bool(job.get("collect_privileged_belief", False)) and os.environ.get(
        "POKEBOT_LIBCG_PATH", ""
    ).strip():
        from poke_bot.pure_rl.multi_env_self_play import run_self_play_multi

        return run_self_play_multi([job])[0]

    import json
    import random
    import signal
    import time

    import torch

    from poke_bot import config as _config
    from poke_bot.agent import PolicyAgent, install_quiet_stdout, play_game
    from poke_bot.train import load_model_from_checkpoint

    our_seat = int(job.get("our_seat", 0))
    opp_id = str(job.get("opponent_id") or "self")
    seed = int(job["seed"])
    deck = list(job["our_deck"])
    timeout_s = int(job.get("game_timeout_s", 600))

    def _base(**over):
        r = {
            "job_index": int(job.get("job_index", 0)),
            "opponent_id": opp_id,
            "our_seat": our_seat,
            "winner": 2,
            "value": 0.0,
            "steps": 0,
            "baseline_failed": False,
            "our_failed": False,
            "resource_error": False,
            "cancelled": False,
            "training_eligible": True,
            "record_json": None,
            "record_jsons": None,
            "error": None,
            "self_play": True,
            "matchup_runtime_audit": None,
            "opponent_matchup_runtime_audit": None,
        }
        r.update(over)
        return r

    try:
        install_quiet_stdout(_config.agent_verbose())
        random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device(job.get("device", "cpu"))
        us = str(job["checkpoint"])
        them = str(job.get("opponent_checkpoint") or us)
        collect_both = bool(
            job.get("training_eligible", True)
            and job.get("collect_both_seats", False)
            and them == us
        )

        # Prefer coalesced GPU leaf servers when WorkerPool registered a channel
        # (same pattern as round-robin ``_worker_play``). Official libcg still
        # steps on CPU; only network eval moves to the leaf farm.
        from poke_bot import batched_infer
        from poke_bot.pure_rl.leaf_self_play import plan_self_play_leaf_wiring

        leaf_backend = batched_infer.remote_leaf_backend_from_worker()
        plan = plan_self_play_leaf_wiring(
            us_checkpoint=us,
            them_checkpoint=them,
            leaf_channel_active=leaf_backend is not None,
        )

        state = load_round_robin_module()._WORKER_STATE
        us_model = None
        them_model = None
        if plan.load_us_local or plan.load_them_local:
            cache_key = f"self|{us}|{them}|{device}|{plan.mode}"
            if state.get("self_play_key") != cache_key:
                if plan.load_us_local:
                    state["self_play_us"] = load_model_from_checkpoint(
                        us, device=device
                    )
                else:
                    state["self_play_us"] = None
                if plan.load_them_local:
                    state["self_play_them"] = (
                        state["self_play_us"]
                        if them == us and plan.load_us_local
                        else load_model_from_checkpoint(them, device=device)
                    )
                else:
                    state["self_play_them"] = None
                state["self_play_key"] = cache_key
            us_model = state.get("self_play_us") if plan.load_us_local else None
            them_model = state.get("self_play_them") if plan.load_them_local else None

        us_leaf = leaf_backend if plan.use_leaf_for_us else None
        them_leaf = leaf_backend if plan.use_leaf_for_them else None
        temp = float(job.get("action_temperature", 1.0))
        sample = bool(job.get("sample_actions", True))
        ctx = int(job.get("model_max_context") or _config.MODEL.max_context)
        us_agent = PolicyAgent(
            model=us_model,
            deck=deck,
            use_mcts=False,
            max_sims=0,
            collect_targets=bool(job.get("training_eligible", True)),
            sample_actions=sample,
            action_temperature=temp,
            rng=random.Random(seed ^ 0xA11CE),
            device=device,
            leaf_backend=us_leaf,
            max_context_override=ctx,
            game_time_budget_s=float(timeout_s),
            game_watchdog_reserve_s=min(60.0, max(10.0, 0.1 * float(timeout_s))),
            strict_runtime=True,
            own_deck_ledger_enabled=bool(
                job.get("own_deck_ledger_enabled", False)
            ),
        )
        them_agent = PolicyAgent(
            model=them_model,
            deck=list(job.get("opp_deck") or deck),
            use_mcts=False,
            max_sims=0,
            collect_targets=collect_both,
            sample_actions=sample,
            action_temperature=temp,
            rng=random.Random(seed ^ 0xBEEF),
            device=device,
            leaf_backend=them_leaf,
            max_context_override=ctx,
            game_time_budget_s=float(timeout_s),
            game_watchdog_reserve_s=min(60.0, max(10.0, 0.1 * float(timeout_s))),
            strict_runtime=True,
            own_deck_ledger_enabled=bool(
                job.get("own_deck_ledger_enabled", False)
            ),
        )
        us_agent.reset_game()
        them_agent.reset_game()

        def _on_timeout(_s, _f):
            raise TimeoutError(f"self-play game exceeded {timeout_s}s")

        had_alarm = hasattr(signal, "SIGALRM")
        if had_alarm:
            signal.signal(signal.SIGALRM, _on_timeout)
            signal.alarm(timeout_s)
        t0 = time.perf_counter()
        try:
            if our_seat == 0:
                result = play_game(us_agent, them_agent, deck, list(job.get("opp_deck") or deck))
            else:
                result = play_game(them_agent, us_agent, list(job.get("opp_deck") or deck), deck)
        finally:
            if had_alarm:
                signal.alarm(0)
        wall_s = time.perf_counter() - t0
        terminal_policy_failure = result.get("failed_seat") is not None
        terminal_failure_error = None
        if terminal_policy_failure:
            failed = int(result["failed_seat"])
            from poke_bot.pure_rl.multi_env_self_play import (
                terminal_policy_failure_outcome,
            )

            winner, value = terminal_policy_failure_outcome(
                failed_seat=failed, our_seat=our_seat
            )
            terminal_failure_error = str(
                result.get("error") or "policy action failure"
            )
        else:
            if result.get("incomplete"):
                return _base(
                    resource_error=True,
                    game_timeout=True,
                    steps=int(result.get("steps", 0)),
                    error="incomplete",
                )
            winner = int(result["winner"])
            value = (
                0.0 if winner == 2 else (1.0 if winner == our_seat else -1.0)
            )
        matchup_runtime_audit = us_agent.matchup_adapter_shadow_snapshot()
        opponent_matchup_runtime_audit = (
            them_agent.matchup_adapter_shadow_snapshot()
        )
        record = None
        records = []
        if job.get("training_eligible", True) and us_agent.targets:
            rr = load_round_robin_module()
            record = rr._build_selfplay_record(
                us_agent.targets,
                our_deck=deck,
                our_seat=our_seat,
                value=value,
                opp_id=opp_id,
                archetype=str(job.get("archetype") or "core"),
                seed=seed,
                target_provenance={
                    **dict(job.get("target_provenance") or {}),
                    "pure_rl": True,
                    "self_play": True,
                    "soft_policy_targets": False,
                    "terminal_policy_failure": terminal_policy_failure,
                    "terminal_failed_seat": (
                        int(result["failed_seat"])
                        if terminal_policy_failure
                        else None
                    ),
                    "matchup_runtime_audit": matchup_runtime_audit,
                },
                opp_archetype=(
                    str(job.get("opp_archetype") or "") or None
                ),
            )
            if record is not None:
                records.append(record)
        if collect_both and them_agent.targets:
            rr = load_round_robin_module()
            opp_deck = list(job.get("opp_deck") or deck)
            opp_value = 0.0 if winner == 2 else -value
            opp_record = rr._build_selfplay_record(
                them_agent.targets,
                our_deck=opp_deck,
                our_seat=1 - our_seat,
                value=opp_value,
                opp_id=f"self:{Path(us).name}",
                archetype=str(
                    job.get("opp_archetype")
                    or job.get("archetype")
                    or "core"
                ),
                seed=seed ^ 0x51DE,
                target_provenance={
                    **dict(job.get("target_provenance") or {}),
                    "pure_rl": True,
                    "self_play": True,
                    "same_policy_second_seat": True,
                    "soft_policy_targets": False,
                    "terminal_policy_failure": terminal_policy_failure,
                    "terminal_failed_seat": (
                        int(result["failed_seat"])
                        if terminal_policy_failure
                        else None
                    ),
                    "matchup_runtime_audit": opponent_matchup_runtime_audit,
                },
                opp_archetype=(
                    str(job.get("archetype") or "") or None
                ),
            )
            if opp_record is not None:
                records.append(opp_record)
        remote_diagnostics = [
            dict(target.get("diagnostics") or {})
            for agent in (us_agent, them_agent)
            for target in agent.targets
        ]
        remote_requests = sum(
            int(row.get("remote_requests") or 0) for row in remote_diagnostics
        )

        def _weighted_total(field: str) -> float:
            return sum(
                float(row.get(field) or 0.0)
                * int(row.get("remote_requests") or 0)
                for row in remote_diagnostics
            )

        return {
            "job_index": int(job.get("job_index", 0)),
            "opponent_id": opp_id,
            "our_seat": our_seat,
            "winner": winner,
            "value": value,
            "steps": int(result.get("steps", 0)),
            "wall_s": wall_s,
            "n_decisions": sum(len(r.get("steps") or []) for r in records),
            "baseline_failed": False,
            "our_failed": False,
            "resource_error": False,
            "cancelled": False,
            "training_eligible": True,
            "self_play": True,
            "leaf_remote": bool(plan.use_leaf_for_us or plan.use_leaf_for_them),
            "leaf_self_play_mode": plan.mode,
            "remote_leaf_requests": remote_requests,
            "remote_queue_wait_ms_total": _weighted_total(
                "queue_wait_ms_mean"
            ),
            "remote_inference_batch_total": _weighted_total(
                "inference_batch_size_mean"
            ),
            "remote_batch_occupancy_total": _weighted_total(
                "batch_occupancy_mean"
            ),
            "remote_coalesced_requests_total": _weighted_total(
                "coalesced_requests_mean"
            ),
            "remote_server_inference_ms_total": _weighted_total(
                "server_inference_ms_mean"
            ),
            "remote_client_roundtrip_ms_total": _weighted_total(
                "client_roundtrip_ms_mean"
            ),
            "remote_request_queue_depth_max": max(
                (
                    int(row.get("request_queue_depth_max") or 0)
                    for row in remote_diagnostics
                ),
                default=0,
            ),
            "matchup_runtime_audit": matchup_runtime_audit,
            "opponent_matchup_runtime_audit": opponent_matchup_runtime_audit,
            "policy_terminal_failure": terminal_policy_failure,
            "failed_seat": (
                int(result["failed_seat"])
                if terminal_policy_failure
                else None
            ),
            "record_json": (
                json.dumps(record, separators=(",", ":")) if record else None
            ),
            "record_jsons": [
                json.dumps(row, separators=(",", ":")) for row in records
            ],
            "error": terminal_failure_error,
        }
    except BaseException as exc:  # noqa: BLE001
        return _base(our_failed=True, error=f"{type(exc).__name__}: {exc}")


def remote_self_play_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one game with checkpoint-scoped caches and post-result collection."""

    _bind_controller_matchup_runtime(job)
    state = _prepare_remote_worker_job_scope(
        [job],
        cache_family=_remote_self_play_cache_family(job),
    )
    try:
        return _remote_self_play_job_impl(job)
    finally:
        _collect_remote_worker_game_cycles(state)
