#!/usr/bin/env python3
"""Superseded exploratory r214 frozen-r195 BeliefMCTS canary scaffold.

The default mode is a pair launcher.  It first asks one fresh interpreter to
seal the native ``BattleStartSeeded`` first-player result, then invokes one
fresh interpreter for each seat-swapped game.  The experimental arm is the
existing public-history/root-sampled ``BeliefMCTS`` around the complete frozen
r195 package; the direct arm is the same package with search disabled.  This
is explicitly *not* r207 exact-chance MCTS.

Revision 215 superseded r214's per-decision execution semantics before this
scaffold launched.  It is retained solely as an unlaunched implementation
reference for package/process isolation; ``main`` deliberately fails closed.
No BO1000 or canary may be started through this file.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib.util
import json
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping


R195_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R195_CHECKPOINT_BYTES = 127_914_385
R195_BUNDLE_SHA256 = (
    "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
)
R195_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
R195_DECK_CARDS_SHA256 = (
    "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
)
R214_REQUIRED_LABEL = "root_sampled_belief_mcts_non_r207_exact_chance"
FORBIDDEN_COMPONENT_WORDS = ("RTP", "GUIDE", "GUIDE2VEC", "POKE_RLM")


class CanaryError(RuntimeError):
    pass


class CanaryInvalidTimeout(CanaryError):
    """A hard r214 planner clock was exceeded; never count a game outcome."""


class SearchWatchdogExpired(Exception):
    """Injected before 5 s so PolicyAgent can transactionally choose direct."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return "sha256:" + hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise CanaryError(f"{label} must be a JSON object")
    return value


def _deck_cards_sha256(path: Path) -> str:
    cards: list[int] = []
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CanaryError(f"cannot read deck: {path}") from exc
    for raw in rows:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cards.append(int(line.split(",", 1)[0]))
        except ValueError as exc:
            raise CanaryError("deck contains a non-integer card row") from exc
    if len(cards) != 60:
        raise CanaryError(f"frozen r195 deck must have 60 cards, got {len(cards)}")
    return _canonical_sha256(cards)


def _assert_python_runtime() -> dict[str, Any]:
    if sys.version_info < (3, 10):
        raise CanaryError("r214 requires Python >= 3.10; do not use the Py3.9 worker")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - no usable model runtime is a hard stop.
        raise CanaryError("r214 requires an importable torch runtime") from exc
    return {
        "python": sys.version.split()[0],
        "python_implementation": sys.implementation.name,
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }


def _iter_files(root: Path) -> dict[str, str]:
    """Return a symlink-free content map relative to one extracted package."""

    if not root.is_dir() or root.is_symlink():
        raise CanaryError(f"package root must be a real directory: {root}")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise CanaryError(f"package may not contain symlinks: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise CanaryError(f"package contains a non-regular file: {relative}")
        files[relative] = _sha256(path)
    return files


def _verify_package_pair(mcts_root: Path, direct_root: Path, bundle: Path) -> dict[str, Any]:
    """Bind both arms to r195 and allow only the evaluator search-config delta."""

    for path, label in ((mcts_root, "MCTS"), (direct_root, "direct")):
        required = (
            "main.py",
            "model.pt",
            "deck.csv",
            "search_config.json",
            "belief_decks.json",
            "matchup_tree.json",
            "runtime_profile.json",
            "turn_order_profile.json",
        )
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise CanaryError(f"{label} package is missing: {', '.join(missing)}")
        if (path / "rtp_shadow_planner.pt").exists():
            raise CanaryError(f"{label} package contains forbidden RTP sidecar")
        if _sha256(path / "model.pt") != R195_CHECKPOINT_SHA256:
            raise CanaryError(f"{label} model is not exact frozen r195 NO-RTP")
        if (path / "model.pt").stat().st_size != R195_CHECKPOINT_BYTES:
            raise CanaryError(f"{label} model byte count is not exact frozen r195")
        if _sha256(path / "matchup_tree.json") != R195_MATCHUP_TREE_SHA256:
            raise CanaryError(f"{label} matchup tree is not exact r195")
        if _deck_cards_sha256(path / "deck.csv") != R195_DECK_CARDS_SHA256:
            raise CanaryError(f"{label} deck is not exact r195 deck order")
        profile = _read_object(path / "runtime_profile.json", label=f"{label} profile")
        if (
            profile.get("display") != "NO RTP"
            or profile.get("recursive_turn_planner") != "disabled"
            or profile.get("rtp_sidecar_packaged") is not False
        ):
            raise CanaryError(f"{label} profile is not explicit NO RTP")
    if not bundle.is_file() or _sha256(bundle) != R195_BUNDLE_SHA256:
        raise CanaryError("--r195-bundle is not the exact immutable r195 NO-RTP bundle")

    mcts_files = _iter_files(mcts_root)
    direct_files = _iter_files(direct_root)
    if set(mcts_files) != set(direct_files):
        raise CanaryError("MCTS/direct extracted package file sets differ")
    changed = sorted(
        relative
        for relative, digest in mcts_files.items()
        if digest != direct_files[relative]
    )
    if changed != ["search_config.json"]:
        raise CanaryError(
            "only search_config.json may differ between direct and r214 MCTS arms"
        )
    direct_search = _read_object(
        direct_root / "search_config.json", label="direct search config"
    )
    mcts_search = _read_object(
        mcts_root / "search_config.json", label="MCTS search config"
    )
    allowed_delta = {"enabled", "maximum_move_s"}
    changed_fields = {
        key
        for key in set(direct_search) | set(mcts_search)
        if direct_search.get(key) != mcts_search.get(key)
    }
    if changed_fields - allowed_delta:
        raise CanaryError(
            "r214 search-config copy changed fields beyond enabled/maximum_move_s"
        )
    if direct_search.get("enabled") is not False:
        raise CanaryError("direct r195 package must retain search_config.enabled=false")
    if mcts_search.get("enabled") is not True:
        raise CanaryError("r214 MCTS copy must set search_config.enabled=true")
    if float(mcts_search.get("maximum_move_s", -1.0)) != 5.0:
        raise CanaryError("r214 MCTS config must declare a 5.0s atomic cap")
    if (
        mcts_search.get("algorithm") != "public_history_root_sampled_belief_mcts"
        or mcts_search.get("leaf_evaluator") != "trained_checkpoint_policy_value_head"
        or mcts_search.get("leaf_evaluator_checkpoint") != "submission_model_pt"
        or mcts_search.get("require_trained_state_evaluator") is not True
        or mcts_search.get("oracle_inputs_allowed") is not False
        or mcts_search.get("search_failure_behavior")
        != "greedy_current_decision_then_retry"
    ):
        raise CanaryError("r214 MCTS copy does not bind the dormant trusted BeliefMCTS")
    if int(mcts_search.get("minimum_sims", 0)) != 50 or int(
        mcts_search.get("maximum_sims", 0)
    ) < 50:
        raise CanaryError("r214 must preserve r195's 50-simulation trust floor")
    return {
        "r195_bundle_sha256": _sha256(bundle),
        "mcts_package_content_sha256": _canonical_sha256(mcts_files),
        "direct_package_content_sha256": _canonical_sha256(direct_files),
        "only_changed_file": "search_config.json",
        "only_changed_search_config_fields": sorted(changed_fields),
        "mcts_search_config_sha256": _sha256(mcts_root / "search_config.json"),
        "direct_search_config_sha256": _sha256(direct_root / "search_config.json"),
        "minimum_trusted_sims": int(mcts_search["minimum_sims"]),
    }


def _scrub_environment(package_root: Path, seeded_lib: Path | None) -> dict[str, str]:
    """Use a clean package-local no-RTP runtime in every fresh game child."""

    for key in list(os.environ):
        upper = key.upper()
        if (
            upper.startswith("BO1000_")
            or any(word in upper for word in FORBIDDEN_COMPONENT_WORDS)
            or upper.startswith("POKEBOT_SLOWKING_DISTILL")
        ):
            os.environ.pop(key, None)
    os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] = "0"
    os.environ["POKEBOT_MATCHUP_ADAPTER_RUNTIME"] = "1"
    os.environ["POKEBOT_PUBLIC_MATCHUP_TREE_PATH"] = str(
        (package_root / "matchup_tree.json").resolve()
    )
    # main.py otherwise uses setdefault and could retain another arm's cg path.
    os.environ["CG_LIB_PATH"] = str(package_root)
    if seeded_lib is not None:
        os.environ["POKEBOT_LIBCG_PATH"] = str(seeded_lib)
    else:
        os.environ.pop("POKEBOT_LIBCG_PATH", None)
    return {
        key: os.environ[key]
        for key in (
            "POKEBOT_USE_RECURSIVE_TURN_PLANNER",
            "POKEBOT_MATCHUP_ADAPTER_RUNTIME",
            "POKEBOT_PUBLIC_MATCHUP_TREE_PATH",
            "CG_LIB_PATH",
            "POKEBOT_LIBCG_PATH",
        )
        if key in os.environ
    }


def _isolate_package_imports(package_root: Path) -> None:
    """Prevent inherited PYTHONPATH/module state from mixing package copies."""

    os.chdir(package_root)
    package_text = str(package_root)
    inherited = [entry for entry in sys.path if entry not in {"", package_text}]
    sys.path[:] = [package_text, *inherited]
    for name in list(sys.modules):
        if name == "poke_bot" or name.startswith("poke_bot.") or name == "cg" or name.startswith("cg."):
            del sys.modules[name]


def _load_controller(controller_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("r214_belief_mcts_controller", controller_path)
    if spec is None or spec.loader is None:
        raise CanaryError("cannot import r214 controller source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_packaged_main(package_root: Path) -> Any:
    spec = importlib.util.spec_from_file_location("r214_packaged_submission", package_root / "main.py")
    if spec is None or spec.loader is None:
        raise CanaryError("cannot import package-local main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._ensure_runtime()
    return module


def _assert_runtime(module: Any, *, min_trusted_sims: int) -> tuple[list[int], Any, Any]:
    deck, model, policy = module._DECK, module._MODEL, module._POLICY
    if not isinstance(deck, list) or len(deck) != 60 or model is None or policy is None:
        raise CanaryError("packaged MCTS runtime did not load deck/model/policy")
    if not bool(getattr(policy, "use_mcts", False)) or not bool(
        getattr(policy, "belief_mcts", False)
    ):
        raise CanaryError("experimental runtime did not activate BeliefMCTS")
    if int(getattr(policy, "min_trusted_sims", 0) or 0) != min_trusted_sims:
        raise CanaryError("live MCTS trust floor drifted from sealed r195 config")
    if int(getattr(policy, "max_sims", 0) or 0) < min_trusted_sims:
        raise CanaryError("live MCTS max simulations are below trust floor")
    if bool(getattr(policy, "use_recursive_turn_planner", False)) or getattr(
        policy, "_rtp_bridge", None
    ) is not None:
        raise CanaryError("experimental runtime unexpectedly enabled RTP")
    if not bool(getattr(policy, "matchup_adapter_runtime", False)):
        raise CanaryError("experimental runtime did not enable matchup adapter routing")
    if not bool(getattr(model, "training", True)) is False:
        raise CanaryError("frozen r195 model is not in eval mode")
    bank = getattr(model, "matchup_adapter_bank", None)
    if bank is None or not bool(getattr(bank, "enabled", False)):
        raise CanaryError("trained r195 matchup adapter bank is not active")
    for config_name in ("poke_rlm_config", "slowking_distill_config"):
        config = getattr(policy, config_name, None)
        if config is not None and bool(getattr(config, "selects_actions", False)):
            raise CanaryError(f"experimental runtime unexpectedly enables {config_name}")
    return deck, model, policy


def _make_direct_policy(model: Any, deck: list[int]) -> Any:
    from poke_bot.agent import PolicyAgent

    direct = PolicyAgent(model=model, deck=list(deck), use_mcts=False)
    direct.strict_runtime = True
    if bool(getattr(direct, "use_recursive_turn_planner", False)) or getattr(
        direct, "_rtp_bridge", None
    ) is not None:
        raise CanaryError("direct arm unexpectedly enabled RTP")
    if not bool(getattr(direct, "matchup_adapter_runtime", False)):
        raise CanaryError("direct arm did not enable r195 matchup adapter runtime")
    for config_name in ("poke_rlm_config", "slowking_distill_config"):
        config = getattr(direct, config_name, None)
        if config is not None and bool(getattr(config, "selects_actions", False)):
            raise CanaryError(f"direct arm unexpectedly enables {config_name}")
    return direct


def _configure_seeded_start(env: Any) -> None:
    """Make the private seeded ABI explicit; ctypes defaults are unsafe here."""

    try:
        seeded = env._lib.BattleStartSeeded
    except AttributeError as exc:
        raise CanaryError("selected libcg lacks BattleStartSeeded; pairing cannot run") from exc
    from cg import sim

    seeded.restype = sim.StartData
    seeded.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint32]


def _new_seeded_env() -> Any:
    from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv

    env = LibcgMultiEnv(1)
    _configure_seeded_start(env)
    return env


def _validate_legal_action(observation: Mapping[str, Any], action: list[int]) -> None:
    select = observation.get("select")
    if not isinstance(select, Mapping):
        raise CanaryError("engine action requested without select payload")
    options = select.get("option")
    if not isinstance(options, list):
        raise CanaryError("engine select payload has no option list")
    lower = int(select.get("minCount", 0) or 0)
    upper = min(int(select.get("maxCount", 0) or 0), len(options))
    if not lower <= len(action) <= upper:
        raise CanaryError("policy returned an invalid action count")
    if len(set(action)) != len(action) or any(
        type(index) is not int or not 0 <= index < len(options) for index in action
    ):
        raise CanaryError("policy returned an invalid action index")


def _packaged_turn_order_action(module: Any, observation: Mapping[str, Any]) -> list[int] | None:
    """Use the exact package ``first_if_allowed`` code before all policy calls."""

    choice = module._turn_order_choice(dict(observation))
    if choice is None:
        return None
    action = list(module._fail_closed(dict(observation), list(choice)))
    if action != list(choice):
        raise CanaryError("packaged first_if_allowed action was not legal/fail-closed exact")
    _validate_legal_action(observation, action)
    return action


@contextlib.contextmanager
def _search_watchdog(seconds: float) -> Iterator[dict[str, bool]]:
    """Raise before 5 s so PolicyAgent's transactional direct fallback runs."""

    if not hasattr(signal, "setitimer") or not hasattr(signal, "ITIMER_REAL"):
        raise CanaryError("r214 requires a Unix monotonic action watchdog")
    if seconds <= 0.0:
        yield {"fired": False}
        return
    prior_timer = signal.getitimer(signal.ITIMER_REAL)
    if prior_timer != (0.0, 0.0):
        raise CanaryError("refusing to overwrite an inherited process watchdog")
    prior_handler = signal.getsignal(signal.SIGALRM)
    fired = {"fired": False}

    def expire(_signum: int, _frame: Any) -> None:
        fired["fired"] = True
        raise SearchWatchdogExpired("r214 pre-fallback search watchdog elapsed")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield fired
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prior_handler)


def _search_diagnostics(policy: Any, prior_result: Any) -> dict[str, Any] | None:
    result = getattr(policy, "last_result", None)
    if result is None or result is prior_result or getattr(policy, "last_search_fallback_reason", None):
        return None
    target = getattr(result, "target", None)
    diagnostics = getattr(target, "diagnostics", None)
    return dict(diagnostics) if isinstance(diagnostics, Mapping) else None


def _make_turn_telemetry(
    *,
    controller: Any,
    timing: Any,
    diagnostics: Mapping[str, Any] | None,
    search_fallback_reason: str | None,
    watchdog_fired: bool,
    adapter_before: Mapping[str, Any],
    adapter_after: Mapping[str, Any],
) -> dict[str, Any]:
    diag = dict(diagnostics or {})
    sims_requested = int(diag.get("sims_requested", 0) or 0)
    sims_run = int(diag.get("sims_run", 0) or 0)
    root_visits = int(diag.get("root_visits", 0) or 0)
    complete_action_count = int(diag.get("complete_ordered_action_count", 0) or 0)
    payload = timing.as_payload()
    payload.update(
        {
            "strategy_label": R214_REQUIRED_LABEL,
            "sims_requested": sims_requested,
            "sims_run": sims_run,
            "sims_per_second": float(diag.get("sims_per_s", 0.0) or 0.0),
            "leaf_evaluations": int(diag.get("leaf_evaluations", 0) or 0),
            "unique_nodes": int(diag.get("unique_nodes", 0) or 0),
            "unique_expanded_nodes": int(diag.get("unique_expanded_nodes", 0) or 0),
            "root_visits": root_visits,
            "max_depth": int(diag.get("max_depth", 0) or 0),
            "mean_depth": float(diag.get("mean_depth", 0.0) or 0.0),
            "mean_branching": float(diag.get("mean_branching", 0.0) or 0.0),
            "particle_bank_size": int(diag.get("particle_bank_size", 0) or 0),
            "particles_sampled": int(diag.get("particles_sampled", 0) or 0),
            "unique_particles": int(diag.get("unique_particles", 0) or 0),
            "particle_support_modes": list(diag.get("particle_support_modes") or []),
            "particle_support_repairs": int(diag.get("particle_support_repairs", 0) or 0),
            "chance_samples": int(diag.get("chance_samples", 0) or 0),
            "complete_ordered_action_count": complete_action_count,
            "action_space_mode": str(diag.get("action_space_mode") or "none_due_to_fallback"),
            "search_stop_reason": str(
                diag.get("stop_reason")
                or search_fallback_reason
                or "packaged_turn_order_profile"
            ),
            "requested_simulation_target_completed_within_budget": bool(
                sims_requested > 0 and sims_run >= sims_requested
            ),
            "full_finite_tree_completion": "not_applicable_root_sampled_stochastic_belief_tree",
            "planner_turn_wall_seconds": payload["planner_wall_seconds"],
            "max_atomic_action_planner_wall_seconds": payload["planner_wall_seconds"],
            "deadline_hit": bool(watchdog_fired),
            "search_fallback_reason": search_fallback_reason,
            "matchup_adapter_before": dict(adapter_before),
            "matchup_adapter_after": dict(adapter_after),
            "matchup_adapter_search_branch": diag.get("matchup_adapter_shadow"),
            "matchup_adapter_enabled_and_route_receipt": bool(adapter_after),
            "tree_config_and_frozen_package_identity_sha256": controller.canonical_sha256(
                {
                    "timing_config_sha256": controller.R214TimingConfig().identity_sha256,
                    "search_semantics": diag.get("search_semantics"),
                    "checkpoint": R195_CHECKPOINT_SHA256,
                    "tree": R195_MATCHUP_TREE_SHA256,
                }
            ),
        }
    )
    return payload


def _mcts_action(*, policy: Any, observation: Mapping[str, Any], ledger: Any, controller: Any) -> tuple[list[int], dict[str, Any]]:
    planner_turn = controller.turn_identity_from_observation(observation)
    budget = ledger.begin_action(planner_turn)
    started = time.monotonic()
    adapter_before = policy.matchup_adapter_shadow_snapshot()
    search_attempted = False
    fallback = False
    fallback_reason: str | None = None
    watchdog_fired = False
    diagnostics: dict[str, Any] | None = None
    try:
        turn_order = None  # The caller intercepts this; defend against regressions.
        if turn_order is not None:
            action = turn_order
        elif not budget.search_allowed:
            action = list(policy.trusted_search_or_greedy_select(dict(observation), search=False))
            fallback = True
            fallback_reason = budget.reason
        else:
            prior_result = policy.last_result
            prior_clock = policy.clock
            prior_move_time = policy.move_time_s
            elapsed_before_search = time.monotonic() - started
            search_seconds = min(
                budget.allowed_search_wall_seconds,
                max(
                    0.0,
                    budget.hard_action_wall_seconds
                    - elapsed_before_search
                    - ledger.config.direct_fallback_reserve_seconds,
                ),
            )
            if search_seconds <= 0.0:
                action = list(policy.trusted_search_or_greedy_select(dict(observation), search=False))
                fallback = True
                fallback_reason = "atomic_action_reserve_exhausted_before_search"
            else:
                policy.clock = None
                policy.move_time_s = search_seconds
                search_attempted = True
                try:
                    with _search_watchdog(search_seconds) as watchdog:
                        action = list(
                            policy.trusted_search_or_greedy_select(
                                dict(observation), search=True
                            )
                        )
                        watchdog_fired = bool(watchdog["fired"])
                except SearchWatchdogExpired:
                    # A signal outside PolicyAgent's transactional catch is
                    # still early enough to ask its exact frozen direct path.
                    action = list(policy.trusted_search_or_greedy_select(dict(observation), search=False))
                    watchdog_fired = True
                    fallback = True
                    fallback_reason = "search_watchdog_escaped_to_direct_policy"
                finally:
                    policy.clock = prior_clock
                    policy.move_time_s = prior_move_time
                if policy.last_search_fallback_reason is not None:
                    fallback = True
                    fallback_reason = str(policy.last_search_fallback_reason)
                diagnostics = _search_diagnostics(policy, prior_result)
    except Exception:
        raise
    finally:
        adapter_after = policy.matchup_adapter_shadow_snapshot()
    _validate_legal_action(observation, action)
    timing = ledger.record_action(
        budget,
        planner_wall_seconds=time.monotonic() - started,
        search_attempted=search_attempted,
        direct_policy_fallback_used=fallback,
        reason=fallback_reason,
    )
    telemetry = _make_turn_telemetry(
        controller=controller,
        timing=timing,
        diagnostics=diagnostics,
        search_fallback_reason=fallback_reason,
        watchdog_fired=watchdog_fired,
        adapter_before=adapter_before,
        adapter_after=adapter_after,
    )
    if not timing.within_hard_budget:
        raise CanaryInvalidTimeout(str(timing.reason))
    return action, telemetry


def _first_player_from_observation(observation: Mapping[str, Any]) -> int | None:
    current = observation.get("current")
    if not isinstance(current, Mapping):
        return None
    first = current.get("firstPlayer")
    return first if type(first) is int and first in {0, 1} else None


def _reset_seeded_game(env: Any, deck: list[int], seed: int) -> Any:
    from poke_bot.engine_rebuild.interfaces import ResetSpec

    return env.reset([ResetSpec(deck0=list(deck), deck1=list(deck), seed=int(seed))]).envs[0]


def _seal_pair_first_player(*, module: Any, deck: list[int], spec: Any, controller: Any) -> dict[str, Any]:
    env = _new_seeded_env()
    try:
        state = _reset_seeded_game(env, deck, spec.engine_seed_u32)
        setup_actions: list[list[int]] = []
        for _ in range(4):
            first = _first_player_from_observation(state.obs)
            if first is not None:
                seal = controller.R214PairFirstPlayerSeal(
                    pair_index=spec.pair_index,
                    pair_id=spec.pair_id,
                    pair_nonce_sha256=spec.pair_nonce_sha256,
                    engine_seed_u32=spec.engine_seed_u32,
                    deck_order_seed_u32=spec.deck_order_seed_u32,
                    first_player_seat=first,
                    post_turn_order_observation_sha256=controller.canonical_sha256(state.obs),
                )
                return {"pair_first_player_seal": seal.as_payload(), "setup_actions": setup_actions}
            action = _packaged_turn_order_action(module, state.obs)
            if action is None:
                raise CanaryError("engine did not expose resolvable packaged IsFirst prompt")
            setup_actions.append(action)
            state = env.step_batch([action]).envs[0]
        raise CanaryError("firstPlayer was not sealed after packaged turn-order action")
    finally:
        env.close()


def _play_one_game(*, module: Any, controller: Any, spec: Any, seal: Any, deck: list[int], mcts_policy: Any, direct_policy: Any, max_steps: int) -> dict[str, Any]:
    env = _new_seeded_env()
    telemetry: list[dict[str, Any]] = []
    turn_order_actions: list[dict[str, Any]] = []
    direct_actions = 0
    try:
        mcts_policy.reset_game()
        direct_policy.reset_game()
        # reset_game recreates the legacy 400 s game allocator. r214 owns the
        # only planner budget, so suppress it after every reset.
        mcts_policy.clock = None
        direct_policy.clock = None
        mcts_policy.strict_runtime = True
        direct_policy.strict_runtime = True
        mcts_policy.rng = random.Random(int(spec.mcts_rng_seed_u32))
        direct_policy.rng = random.Random(int(spec.direct_policy_rng_seed_u32))
        ledger = controller.R214TurnBudgetLedger(controller.R214TimingConfig())
        state = _reset_seeded_game(env, deck, spec.engine_seed_u32)
        steps = 0
        first_player_verified = False
        while not state.done and steps < max_steps:
            observation = state.obs
            first = _first_player_from_observation(observation)
            if first is None:
                action = _packaged_turn_order_action(module, observation)
                if action is None:
                    raise CanaryError("unsealed setup state is not exact packaged IsFirst")
                turn_order_actions.append({"action": action, "acting_seat": (observation.get("current") or {}).get("yourIndex")})
                state = env.step_batch([action]).envs[0]
                steps += 1
                continue
            if first != seal.first_player_seat:
                raise CanaryError("game firstPlayer disagrees with fresh pair seal")
            first_player_verified = True
            turn_order = _packaged_turn_order_action(module, observation)
            if turn_order is not None:
                turn_order_actions.append({"action": turn_order, "acting_seat": (observation.get("current") or {}).get("yourIndex")})
                state = env.step_batch([turn_order]).envs[0]
                steps += 1
                continue
            current = observation.get("current")
            if not isinstance(current, Mapping) or current.get("yourIndex") not in {0, 1}:
                raise CanaryError("engine emitted invalid acting seat")
            seat = int(current["yourIndex"])
            if seat == spec.mcts_seat:
                action, row = _mcts_action(
                    policy=mcts_policy,
                    observation=observation,
                    ledger=ledger,
                    controller=controller,
                )
                telemetry.append(row)
            elif seat == spec.direct_seat:
                action = list(direct_policy.trusted_search_or_greedy_select(dict(observation), search=False))
                _validate_legal_action(observation, action)
                direct_actions += 1
            else:
                raise CanaryError("engine emitted a seat outside this pair")
            state = env.step_batch([action]).envs[0]
            steps += 1
        if not first_player_verified:
            raise CanaryError("game ended before first-player seal was verified")
        if not state.done:
            raise CanaryError("game reached max atomic actions without terminal result")
        if any(not bool(row["within_hard_budget"]) for row in telemetry):
            raise CanaryInvalidTimeout("receipt contains a hard timing breach")
        return {
            "game": spec.as_payload(),
            "pair_first_player_seal_sha256": seal.identity_sha256,
            "first_player_seat": seal.first_player_seat,
            "mcts_actual_turn_order": "first" if spec.mcts_seat == seal.first_player_seat else "second",
            "terminal_status": "completed",
            "outcome_eligible": True,
            "winner_seat": state.winner,
            "steps": steps,
            "packaged_turn_order_actions": turn_order_actions,
            "mcts_turn_telemetry": telemetry,
            "mcts_calls": sum(1 for row in telemetry if row["search_attempted"]),
            "direct_policy_fallbacks": sum(1 for row in telemetry if row["direct_policy_fallback_used"]),
            "direct_atomic_actions": direct_actions,
            "all_mcts_timings_within_hard_budget": True,
            "mcts_real_search_tree_seen": any(int(row["sims_run"]) >= 50 for row in telemetry),
        }
    except CanaryInvalidTimeout as exc:
        return {
            "game": spec.as_payload(),
            "pair_first_player_seal_sha256": seal.identity_sha256,
            "first_player_seat": seal.first_player_seat,
            "mcts_actual_turn_order": "first" if spec.mcts_seat == seal.first_player_seat else "second",
            "terminal_status": "invalid_timeout",
            "outcome_eligible": False,
            "winner_seat": None,
            "error": f"{type(exc).__name__}: {exc}",
            "mcts_turn_telemetry": telemetry,
            "all_mcts_timings_within_hard_budget": False,
        }
    except Exception as exc:  # noqa: BLE001 - game receipts deliberately fail closed.
        return {
            "game": spec.as_payload(),
            "pair_first_player_seal_sha256": seal.identity_sha256,
            "first_player_seat": seal.first_player_seat,
            "mcts_actual_turn_order": "first" if spec.mcts_seat == seal.first_player_seat else "second",
            "terminal_status": "failed_closed",
            "outcome_eligible": False,
            "winner_seat": None,
            "error": f"{type(exc).__name__}: {exc}",
            "mcts_turn_telemetry": telemetry,
            "all_mcts_timings_within_hard_budget": False,
        }
    finally:
        env.close()


def _write_create_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as target:
            target.write(encoded)
    except FileExistsError as exc:
        raise CanaryError(f"refusing to overwrite immutable receipt: {path}") from exc
    path.chmod(0o444)


def _common_payload(*, controller: Any, args: argparse.Namespace, package_identity: Mapping[str, Any], runtime: Mapping[str, Any], env: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "poke_bot.alakazam_r214_belief_mcts_seeded_canary_receipt/v1",
        "evaluation_id": controller.R214_EVALUATION_ID,
        "strategy_label": R214_REQUIRED_LABEL,
        "r207_exact_chance_mcts": False,
        "frozen_r195": {
            "checkpoint_sha256": R195_CHECKPOINT_SHA256,
            "bundle_sha256": R195_BUNDLE_SHA256,
            "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
            "deck_cards_sha256": R195_DECK_CARDS_SHA256,
            "matchup_adapter_required": True,
        },
        "policy_transforms": {
            "guide2vec_enabled": False,
            "guide_logit_transform_enabled": False,
            "guide_linear_transform_enabled": False,
            "legacy_rtp_enabled": False,
        },
        "package_parity": dict(package_identity),
        "interpreter": dict(runtime),
        "sanitized_environment": dict(env),
        "timing": controller.build_r214_canary_plan(args.seed_identity_sha256)["timing"],
        "seed_identity_sha256": args.seed_identity_sha256,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _resolve_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path | None]:
    mcts = args.mcts_package.expanduser().resolve()
    direct = args.direct_package.expanduser().resolve()
    controller = args.controller_source.expanduser().resolve()
    bundle = args.r195_bundle.expanduser().resolve()
    seeded = args.seeded_lib.expanduser().resolve() if args.seeded_lib is not None else None
    if not controller.is_file():
        raise CanaryError("--controller-source is missing")
    if seeded is not None and not seeded.is_file():
        raise CanaryError("--seeded-lib is missing")
    return mcts, direct, controller, bundle, seeded


def _run_seal_child(args: argparse.Namespace) -> int:
    mcts, direct, controller_path, bundle, seeded = _resolve_args(args)
    runtime = _assert_python_runtime()
    package_identity = _verify_package_pair(mcts, direct, bundle)
    env = _scrub_environment(mcts, seeded)
    _isolate_package_imports(mcts)
    controller = _load_controller(controller_path)
    controller.require_sha256(args.seed_identity_sha256, name="seed_identity_sha256")
    schedule = controller.build_r214_canary_schedule(args.seed_identity_sha256)
    if len(schedule) != 2:
        raise CanaryError("r214 canary must have exactly two games")
    module = _load_packaged_main(mcts)
    deck, _model, _policy = _assert_runtime(module, min_trusted_sims=package_identity["minimum_trusted_sims"])
    seal_payload = _seal_pair_first_player(module=module, deck=deck, spec=schedule[0], controller=controller)
    payload = _common_payload(controller=controller, args=args, package_identity=package_identity, runtime=runtime, env=env)
    payload.update({"kind": "pair_first_player_seal", **seal_payload})
    _write_create_once(args.output.expanduser().resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve()), "kind": payload["kind"]}, sort_keys=True))
    return 0


def _run_game_child(args: argparse.Namespace) -> int:
    if args.game_index not in {0, 1} or args.pair_seal is None:
        raise CanaryError("game child requires --game-index 0|1 and --pair-seal")
    mcts, direct, controller_path, bundle, seeded = _resolve_args(args)
    runtime = _assert_python_runtime()
    package_identity = _verify_package_pair(mcts, direct, bundle)
    env = _scrub_environment(mcts, seeded)
    _isolate_package_imports(mcts)
    controller = _load_controller(controller_path)
    controller.require_sha256(args.seed_identity_sha256, name="seed_identity_sha256")
    schedule = controller.build_r214_canary_schedule(args.seed_identity_sha256)
    spec = schedule[args.game_index]
    seal_doc = _read_object(args.pair_seal.expanduser().resolve(), label="pair first-player seal receipt")
    raw_seal = seal_doc.get("pair_first_player_seal")
    if not isinstance(raw_seal, Mapping):
        raise CanaryError("pair seal receipt has no typed pair_first_player_seal")
    seal_values = {key: value for key, value in raw_seal.items() if key != "identity_sha256"}
    seal = controller.R214PairFirstPlayerSeal(**seal_values)
    controller.validate_pair_first_player_seal(schedule, seal)
    module = _load_packaged_main(mcts)
    deck, model, mcts_policy = _assert_runtime(module, min_trusted_sims=package_identity["minimum_trusted_sims"])
    direct_policy = _make_direct_policy(model, deck)
    row = _play_one_game(
        module=module,
        controller=controller,
        spec=spec,
        seal=seal,
        deck=deck,
        mcts_policy=mcts_policy,
        direct_policy=direct_policy,
        max_steps=args.max_steps,
    )
    payload = _common_payload(controller=controller, args=args, package_identity=package_identity, runtime=runtime, env=env)
    payload.update({"kind": "game", "game_result": row})
    _write_create_once(args.output.expanduser().resolve(), payload)
    print(json.dumps({"output": str(args.output.resolve()), "terminal_status": row["terminal_status"]}, sort_keys=True))
    return 0 if row["terminal_status"] == "completed" else 2


def _child_command(args: argparse.Namespace, *, output: Path, mode: str, game_index: int | None = None, pair_seal: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mcts-package", str(args.mcts_package.resolve()),
        "--direct-package", str(args.direct_package.resolve()),
        "--controller-source", str(args.controller_source.resolve()),
        "--r195-bundle", str(args.r195_bundle.resolve()),
        "--seed-identity-sha256", args.seed_identity_sha256,
        "--output", str(output),
        "--max-steps", str(args.max_steps),
        mode,
    ]
    if args.seeded_lib is not None:
        command.extend(["--seeded-lib", str(args.seeded_lib.resolve())])
    if game_index is not None:
        command.extend(["--game-index", str(game_index)])
    if pair_seal is not None:
        command.extend(["--pair-seal", str(pair_seal)])
    return command


def _run_pair_launcher(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise CanaryError(f"refusing to overwrite immutable receipt: {output}")
    _assert_python_runtime()
    # Validate static identities before consuming a fresh engine seed.
    mcts, direct, controller_path, bundle, _seeded = _resolve_args(args)
    package_identity = _verify_package_pair(mcts, direct, bundle)
    controller = _load_controller(controller_path)
    controller.require_sha256(args.seed_identity_sha256, name="seed_identity_sha256")
    schedule = controller.build_r214_canary_schedule(args.seed_identity_sha256)
    if len(schedule) != 2:
        raise CanaryError("r214 canary must be one exact two-game pair")
    seal_path = output.with_name(output.stem + ".pair-seal.json")
    game_paths = [output.with_name(output.stem + f".game-{index:04d}.json") for index in range(2)]
    if seal_path.exists() or any(path.exists() for path in game_paths):
        raise CanaryError("canary child receipt already exists; use a new output identity")
    seal_run = subprocess.run(_child_command(args, output=seal_path, mode="--seal-pair"), check=False)
    child_codes: dict[str, int] = {"seal": int(seal_run.returncode)}
    if seal_run.returncode == 0:
        for index, game_path in enumerate(game_paths):
            result = subprocess.run(
                _child_command(
                    args,
                    output=game_path,
                    mode="--run-game",
                    game_index=index,
                    pair_seal=seal_path,
                ),
                check=False,
            )
            child_codes[f"game_{index}"] = int(result.returncode)
    rows: list[dict[str, Any]] = []
    for path in game_paths:
        if path.is_file():
            child = _read_object(path, label="game child receipt")
            row = child.get("game_result")
            if isinstance(row, Mapping):
                rows.append(dict(row))
    seal_doc = _read_object(seal_path, label="pair seal child receipt") if seal_path.is_file() else None
    first_player_seal = (
        seal_doc.get("pair_first_player_seal") if isinstance(seal_doc, Mapping) else None
    )
    completed = sum(row.get("terminal_status") == "completed" for row in rows)
    payload = {
        "schema": "poke_bot.alakazam_r214_belief_mcts_seeded_canary_pair_receipt/v1",
        "evaluation_id": controller.R214_EVALUATION_ID,
        "strategy_label": R214_REQUIRED_LABEL,
        "r207_exact_chance_mcts": False,
        "fresh_process_required_per_game": True,
        "frozen_r195": {
            "checkpoint_sha256": R195_CHECKPOINT_SHA256,
            "bundle_sha256": R195_BUNDLE_SHA256,
            "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
            "deck_cards_sha256": R195_DECK_CARDS_SHA256,
        },
        "package_parity": package_identity,
        "canary_plan": controller.build_r214_canary_plan(args.seed_identity_sha256),
        "pair_first_player_seal": first_player_seal,
        "games": rows,
        "completed_games": completed,
        "invalid_or_failed_games": len(rows) - completed,
        "child_exit_codes": child_codes,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if len(rows) == 2 and isinstance(first_player_seal, Mapping):
        seal_values = {key: value for key, value in first_player_seal.items() if key != "identity_sha256"}
        seal = controller.R214PairFirstPlayerSeal(**seal_values)
        controller.validate_pair_first_player_seal(schedule, seal)
        actual_orders = [str(row.get("mcts_actual_turn_order")) for row in rows]
        payload["pair_balance_verified"] = sorted(actual_orders) == ["first", "second"]
    else:
        payload["pair_balance_verified"] = False
    _write_create_once(output, payload)
    print(json.dumps({"output": str(output), "completed_games": completed, "pair_balance_verified": payload["pair_balance_verified"]}, sort_keys=True))
    return 0 if completed == 2 and payload["pair_balance_verified"] and all(code == 0 for code in child_codes.values()) else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcts-package", type=Path, required=True)
    parser.add_argument("--direct-package", type=Path, required=True)
    parser.add_argument("--controller-source", type=Path, required=True)
    parser.add_argument("--r195-bundle", type=Path, required=True)
    parser.add_argument("--seed-identity-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--seeded-lib", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--seal-pair", action="store_true")
    mode.add_argument("--run-game", action="store_true")
    parser.add_argument("--game-index", type=int, choices=(0, 1))
    parser.add_argument("--pair-seal", type=Path)
    args = parser.parse_args()
    if args.max_steps < 1:
        raise CanaryError("--max-steps must be positive")
    if (args.game_index is not None or args.pair_seal is not None) and not args.run_game:
        raise CanaryError("--game-index/--pair-seal are only valid with --run-game")
    return args


def main() -> int:
    raise CanaryError(
        "r214 per-decision canary is superseded and disabled; use the r215 "
        "full-actual-turn evaluator after its prerequisites pass"
    )
    args = _parse_args()
    if args.seal_pair:
        return _run_seal_child(args)
    if args.run_game:
        return _run_game_child(args)
    return _run_pair_launcher(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CanaryError as exc:
        print(f"r214 BeliefMCTS canary failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2)
