"""Fresh-process seeded-engine bridge for the r215 local mirror.

This is intentionally evaluator-only.  It imports no Kaggle client, queue,
submission, selector, training, or promotion code.  Both seats load the one
exact archived r195 NO-RTP package; the experimental seat additionally routes
its next-action selection through :class:`R215FullTurnBeliefMCTS`.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.util
import json
import os
import random
import signal
import sys
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .r215_full_turn_belief_mcts import (
    R215_CHANCE_LABEL,
    R215FullTurnBeliefMCTS,
    R215GameClock,
    R215Observation,
    R215PlanRequest,
    R215PlanResult,
    R215TimingConfig,
    R215TurnIdentity,
)
from .r215_bo1000_launch import (
    R216_SEEDED_ENGINE_SHA256,
    verify_r216_seeded_engine,
)
from .seeded_mirror_harness import (
    canonical_sha256,
    configure_battle_start_seeded,
    engine_turn_identity_from_observation,
)


SCHEMA = "poke_bot.alakazam_full_turn_belief_mcts_bo1000_r215_controller/v1"
STRATEGY = "local_approximate_full_turn_public_history_root_sampled_belief_mcts_whole_frozen_model_wrapper"
R195_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)


class R215SeededMirrorRuntimeError(RuntimeError):
    """A fresh r215 engine child cannot truthfully produce a receipt."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _isolate_package_imports(package_root: Path) -> None:
    """Load only the archived r195 runtime for game/model code.

    References imported from this bridge before isolation remain valid, while
    the archived ``main.py`` resolves its own ``poke_bot`` and ``cg`` tree.
    This keeps a developer checkout from mixing modules into the frozen arm.
    """

    package_text = str(package_root)
    # ``main.py`` deliberately resolves package assets from its working
    # directory.  A fresh game child must therefore never inherit the launcher
    # checkout as its asset root, nor an inherited libcg path from another
    # evaluator.
    os.chdir(package_text)
    os.environ["CG_LIB_PATH"] = package_text
    sys.path[:] = [package_text, *[entry for entry in sys.path if entry not in {"", package_text}]]
    for name in list(sys.modules):
        if name == "poke_bot" or name.startswith("poke_bot.") or name == "cg" or name.startswith("cg."):
            del sys.modules[name]


def _load_package_main(package_root: Path) -> Any:
    _isolate_package_imports(package_root)
    spec = importlib.util.spec_from_file_location(
        "r215_frozen_r195_submission", package_root / "main.py"
    )
    if spec is None or spec.loader is None:
        raise R215SeededMirrorRuntimeError("cannot load frozen r195 main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._ensure_runtime()
    return module


def _validate_action(observation: Mapping[str, Any], action: Sequence[int]) -> list[int]:
    select = observation.get("select")
    if not isinstance(select, Mapping):
        raise R215SeededMirrorRuntimeError("engine requested an action without select")
    options = select.get("option")
    if not isinstance(options, list):
        raise R215SeededMirrorRuntimeError("engine select has no ordered option list")
    lower = int(select.get("minCount", 0) or 0)
    upper = min(int(select.get("maxCount", 0) or 0), len(options))
    if isinstance(action, (str, bytes)) or any(type(index) is not int for index in action):
        raise R215SeededMirrorRuntimeError("selected action has a non-integer option index")
    selected = list(action)
    if not lower <= len(selected) <= upper:
        raise R215SeededMirrorRuntimeError("selected action has an invalid count")
    if len(set(selected)) != len(selected) or any(index < 0 or index >= len(options) for index in selected):
        raise R215SeededMirrorRuntimeError("selected action has an invalid option index")
    return selected


def _turn_order_action(module: Any, observation: Mapping[str, Any]) -> list[int] | None:
    choice = module._turn_order_choice(dict(observation))
    if choice is None:
        return None
    action = list(module._fail_closed(dict(observation), list(choice)))
    if action != list(choice):
        raise R215SeededMirrorRuntimeError("r195 first_if_allowed action changed")
    return _validate_action(observation, action)


def _first_player(observation: Mapping[str, Any]) -> int | None:
    current = observation.get("current")
    if not isinstance(current, Mapping):
        return None
    seat = current.get("firstPlayer")
    return int(seat) if type(seat) is int and seat in {0, 1} else None


def _seeded_engine_identity(request: Mapping[str, Any]) -> dict[str, Any]:
    declared = request.get("seeded_engine")
    raw_path = request.get("seeded_engine_lib")
    if not isinstance(declared, Mapping) or not isinstance(raw_path, str) or not raw_path:
        raise R215SeededMirrorRuntimeError("worker request lacks verified seeded-engine identity")
    path = Path(raw_path).resolve()
    if os.environ.get("POKEBOT_LIBCG_PATH") != str(path):
        raise R215SeededMirrorRuntimeError(
            "fresh worker did not inherit the explicit verified seeded-engine path"
        )
    try:
        verified = verify_r216_seeded_engine(engine_lib=path)
    except Exception as exc:
        raise R215SeededMirrorRuntimeError("seeded engine verification failed") from exc
    for key in ("path", "sha256", "bytes", "battle_start_seeded_available"):
        if declared.get(key) != verified.get(key):
            raise R215SeededMirrorRuntimeError(
                f"worker seeded-engine request identity drifted at {key}"
            )
    return verified


def _runtime_receipt(
    *, model: Any, policy: Any, seeded_engine: Mapping[str, Any]
) -> dict[str, Any]:
    bank = getattr(model, "matchup_adapter_bank", None)
    if getattr(policy, "use_recursive_turn_planner", None) is not False or getattr(policy, "_rtp_bridge", None) is not None:
        raise R215SeededMirrorRuntimeError("r195 runtime unexpectedly armed recursive RTP")
    if getattr(policy, "matchup_adapter_runtime", None) is not True:
        raise R215SeededMirrorRuntimeError("r195 matchup adapter runtime is not on")
    if getattr(model, "training", None) is not False:
        raise R215SeededMirrorRuntimeError("r195 frozen model is not in eval mode")
    if bank is None or getattr(bank, "enabled", None) is not True:
        raise R215SeededMirrorRuntimeError("r195 trained adapter bank is not active")
    tree = Path(str(getattr(policy, "matchup_adapter_tree_path", "")))
    if not tree.is_file() or _sha256_file(tree) != R195_MATCHUP_TREE_SHA256:
        raise R215SeededMirrorRuntimeError("r195 runtime did not bind the exact matchup tree")
    for config_name in ("poke_rlm_config", "slowking_distill_config"):
        config = getattr(policy, config_name, None)
        if config is not None and any(
            bool(getattr(config, field, False))
            for field in ("selects_actions", "runs_planner", "runs")
        ):
            raise R215SeededMirrorRuntimeError(
                f"r195 runtime unexpectedly armed non-frozen {config_name}"
            )
    return {
        "seeded_engine": True,
        "seeded_engine_sha256": R216_SEEDED_ENGINE_SHA256,
        "seeded_engine_bytes": seeded_engine.get("bytes"),
        "fresh_process": True,
        "rtp_enabled": False,
        "legacy_rtp_enabled": False,
        "guide_linear_enabled": False,
        "guide_logit_enabled": False,
        "guide2vec_enabled": False,
        "matchup_adapter_enabled": True,
        "adapter_bank_active": True,
        "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
        "training_eligible": False,
        "submission_authority": False,
        "kaggle_authority": False,
        "selector_authority": False,
        "promotion_authority": False,
    }


def _new_seeded_env(seeded_engine: Mapping[str, Any]) -> Any:
    from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv
    from cg import sim

    expected_path = str(seeded_engine.get("path", ""))
    if not expected_path or os.environ.get("POKEBOT_LIBCG_PATH") != expected_path:
        raise R215SeededMirrorRuntimeError("seeded environment lost its b77 engine binding")
    environment = LibcgMultiEnv(1)
    configure_battle_start_seeded(environment._lib, sim.StartData)
    return environment


def _reset_seeded_game(environment: Any, deck: list[int], seed: int) -> Any:
    from poke_bot.engine_rebuild.interfaces import ResetSpec

    return environment.reset(
        [ResetSpec(deck0=list(deck), deck1=list(deck), seed=int(seed))]
    ).envs[0]


class _PackageBeliefPlanner:
    """Expose a package-local ``PolicyAgent.last_result`` to r215 core."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.last_diagnostics: dict[str, Any] = {}
        self.last_selected_from_policy_result = False

    @staticmethod
    @contextmanager
    def _search_watchdog(seconds: float) -> Any:
        """Interrupt private search before its 5-second operation ceiling.

        This only runs in a fresh evaluator worker's main thread.  The
        ``PolicyAgent`` catches the injected exception transactionally and
        returns its frozen direct-policy action, so a near-deadline search is
        reported as fallback and the game continues rather than forfeiting.
        """

        if seconds <= 0.0:
            raise R215SeededMirrorRuntimeError("search watchdog needs positive time")
        if not hasattr(signal, "setitimer") or not hasattr(signal, "ITIMER_REAL"):
            raise R215SeededMirrorRuntimeError(
                "r216 evaluator requires a Unix per-operation search watchdog"
            )
        active_timer = signal.getitimer(signal.ITIMER_REAL)
        if active_timer != (0.0, 0.0):
            raise R215SeededMirrorRuntimeError(
                "fresh r216 worker inherited an active real-time watchdog"
            )
        previous_handler = signal.getsignal(signal.SIGALRM)

        def _expire(_signum: int, _frame: Any) -> None:
            raise TimeoutError("r216 private search watchdog elapsed")

        signal.signal(signal.SIGALRM, _expire)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)

    def plan_turn(self, request: R215PlanRequest) -> R215PlanResult:
        if request.observation.raw_observation is None:
            raise R215SeededMirrorRuntimeError("r215 planner received no raw observation")
        prior_result = getattr(self.policy, "last_result", None)
        prior_move_time = getattr(self.policy, "move_time_s", None)
        prior_clock = getattr(self.policy, "clock", None)
        allowed_seconds = min(
            float(request.operation_allowance_seconds),
            float(request.turn_pool_remaining_seconds),
            float(request.game_clock_remaining_seconds),
        )
        # Reserve enough of the operation allowance to evaluate the frozen
        # direct fallback and return a legal action.  A healthy 5s operation
        # therefore gets a 4.5s private-search window rather than overrunning
        # the ceiling and losing the entire actual-turn pool.
        search_seconds = max(0.0, allowed_seconds - min(0.5, allowed_seconds / 2.0))
        if search_seconds <= 0.0:
            self.last_diagnostics = {
                "search_stop_reason": "operation_fallback_reserve_exhausted",
                "selected_action_from_policy_last_result": False,
                "operation_search_watchdog_seconds": 0.0,
            }
            return R215PlanResult(
                selected_action=request.observation.legal_actions[0],
                sims_run=0,
                diagnostics=self.last_diagnostics,
            )
        self.policy.move_time_s = search_seconds
        # R215 owns the one 600s GameClock.  The package-local policy must not
        # allocate a second per-game pool behind the controller's back.
        self.policy.clock = None
        try:
            with self._search_watchdog(search_seconds):
                selected = list(
                    self.policy.trusted_search_or_greedy_select(
                        dict(request.observation.raw_observation), search=True
                    )
                )
        except TimeoutError:
            # A timeout that escaped the transactional PolicyAgent boundary is
            # still safe: return a zero-simulation plan and let r215 invoke its
            # exact direct policy outside the private search path.
            selected = list(request.observation.legal_actions[0])
            self.last_diagnostics = {
                "search_stop_reason": "operation_search_watchdog_expired",
                "selected_action_from_policy_last_result": False,
                "operation_search_watchdog_seconds": search_seconds,
            }
            return R215PlanResult(
                selected_action=tuple(selected),
                sims_run=0,
                diagnostics=self.last_diagnostics,
            )
        finally:
            self.policy.move_time_s = prior_move_time
            self.policy.clock = prior_clock
        result = getattr(self.policy, "last_result", None)
        fallback_reason = getattr(self.policy, "last_search_fallback_reason", None)
        if result is None or result is prior_result or fallback_reason:
            self.last_diagnostics = {
                "search_stop_reason": str(fallback_reason or "no_valid_simulation"),
                "selected_action_from_policy_last_result": False,
                "operation_search_watchdog_seconds": search_seconds,
            }
            return R215PlanResult(
                selected_action=tuple(selected), sims_run=0, diagnostics=self.last_diagnostics
            )
        target = getattr(result, "target", None)
        diagnostics = dict(getattr(target, "diagnostics", {}) or {})
        self.last_selected_from_policy_result = list(getattr(result, "select", [])) == selected
        diagnostics["selected_action_from_policy_last_result"] = (
            self.last_selected_from_policy_result
        )
        diagnostics["operation_search_watchdog_seconds"] = search_seconds
        self.last_diagnostics = diagnostics
        return R215PlanResult(
            selected_action=tuple(selected),
            sims_run=int(getattr(result, "sims_run", 0)),
            diagnostics=diagnostics,
        )


def _make_search_policy(module: Any, *, deck: list[int], model: Any, rng_seed: int) -> Any:
    from poke_bot.agent import PolicyAgent
    from poke_bot.belief import EmpiricalDeckPosterior

    belief_path = Path(module._agent_dir()) / "belief_decks.json"
    try:
        payload = json.loads(belief_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R215SeededMirrorRuntimeError("r195 package lacks readable belief_decks.json") from exc
    decks = payload.get("deck_lists") if isinstance(payload, Mapping) else None
    if not isinstance(decks, list) or not decks:
        raise R215SeededMirrorRuntimeError("r195 belief deck prior is missing")
    posterior = EmpiricalDeckPosterior(decks)
    policy = PolicyAgent(
        model=model,
        deck=list(deck),
        use_mcts=True,
        belief_mcts=True,
        belief_posterior=posterior,
        checkpoint_digest="sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a",
        model_generation=0,
        # This is only a defensive stop guard.  The r215 controller is
        # deadline-bounded and never reports it as a requested target.
        max_sims=1_000_000,
        min_trusted_sims=1,
        clock=None,
        move_time_s=5.0,
        rng=random.Random(int(rng_seed)),
    )
    if getattr(policy, "matchup_adapter_runtime", None) is not True:
        raise R215SeededMirrorRuntimeError("search policy did not activate matchup adapter runtime")
    if getattr(policy, "use_recursive_turn_planner", None) is not False or getattr(policy, "_rtp_bridge", None) is not None:
        raise R215SeededMirrorRuntimeError("search policy unexpectedly armed RTP")
    return policy


def _observation_for_controller(observation: Mapping[str, Any], policy: Any) -> R215Observation:
    from poke_bot import features
    from poke_bot.belief_mcts import information_state_fingerprint

    try:
        combinations = features.enumerate_action_combos(dict(observation))
    except Exception as exc:
        raise R215SeededMirrorRuntimeError(
            "r215 requires a complete ordered legal action space for a search decision"
        ) from exc
    route = policy.matchup_adapter_shadow_snapshot()
    return R215Observation(
        public_observation_fingerprint=information_state_fingerprint(dict(observation)),
        legal_actions=tuple(tuple(action) for action in combinations),
        raw_observation=dict(observation),
        action_space_mode="complete_ordered",
        matchup_adapter_route_receipt=route,
    )


def _seal_pair(request: Mapping[str, Any]) -> dict[str, Any]:
    pair = request.get("pair")
    if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(item, Mapping) for item in pair):
        raise R215SeededMirrorRuntimeError("seal_pair needs exactly two game specs")
    package_root = Path(str(request["package_root"])).resolve()
    seeded_engine = _seeded_engine_identity(request)
    module = _load_package_main(package_root)
    deck, model, policy = module._ensure_runtime()
    runtime = _runtime_receipt(model=model, policy=policy, seeded_engine=seeded_engine)
    first_spec = pair[0]
    environment = _new_seeded_env(seeded_engine)
    try:
        state = _reset_seeded_game(environment, list(deck), int(first_spec["engine_seed_u32"]))
        setup_actions: list[list[int]] = []
        for _ in range(8):
            seat = _first_player(state.obs)
            if seat is not None:
                return {
                    "schema": SCHEMA,
                    "operation": "seal_pair",
                    "evaluation_id": request["evaluation_id"],
                    "source_identity_sha256": request["source_identity_sha256"],
                    "output_identity_sha256": request["output_identity_sha256"],
                    "pair_id": first_spec["pair_id"],
                    "pair_index": first_spec["pair_index"],
                    "pair_nonce_sha256": first_spec["pair_nonce_sha256"],
                    "engine_seed_u32": first_spec["engine_seed_u32"],
                    "deck_order_seed_u32": first_spec["deck_order_seed_u32"],
                    "first_player_seat": seat,
                    "post_turn_order_observation_sha256": canonical_sha256(state.obs),
                    "setup_actions": setup_actions,
                    "runtime_receipt": runtime,
                }
            action = _turn_order_action(module, state.obs)
            if action is None:
                raise R215SeededMirrorRuntimeError("unsealed setup state has no r195 first_if_allowed action")
            setup_actions.append(action)
            state = environment.step_batch([action]).envs[0]
        raise R215SeededMirrorRuntimeError("native engine did not seal first player")
    finally:
        environment.close()


def _run_game(request: Mapping[str, Any]) -> dict[str, Any]:
    game = request.get("game")
    if not isinstance(game, Mapping):
        raise R215SeededMirrorRuntimeError("run_game needs one game spec")
    pair_seal = request.get("pair_first_player_seal")
    if not isinstance(pair_seal, Mapping):
        raise R215SeededMirrorRuntimeError("run_game needs its sealed pair first-player fact")
    package_root = Path(str(request["package_root"])).resolve()
    seeded_engine = _seeded_engine_identity(request)
    module = _load_package_main(package_root)
    deck, model, direct_policy = module._ensure_runtime()
    runtime = _runtime_receipt(
        model=model, policy=direct_policy, seeded_engine=seeded_engine
    )
    direct_policy.reset_game()
    direct_policy.strict_runtime = True
    direct_policy.rng = random.Random(int(game["control_rng_seed_u32"]))
    search_policy = _make_search_policy(
        module,
        deck=list(deck),
        model=model,
        rng_seed=int(game["experimental_rng_seed_u32"]),
    )
    search_policy.reset_game()
    search_policy.strict_runtime = True
    planner = _PackageBeliefPlanner(search_policy)

    def direct(observation: R215Observation) -> list[int]:
        if observation.raw_observation is None:
            raise R215SeededMirrorRuntimeError("direct fallback lacks raw observation")
        return _validate_action(
            observation.raw_observation,
            direct_policy.trusted_search_or_greedy_select(
                dict(observation.raw_observation), search=False
            ),
        )

    controller = R215FullTurnBeliefMCTS(
        planner,
        direct_policy=direct,
        timing=R215TimingConfig(),
        # Keep r216's one dynamic 600-second controller clock independent of
        # whatever historical GameClock implementation the frozen package
        # happens to expose after package-import isolation.
        game_clock=R215GameClock(),
    )
    environment = _new_seeded_env(seeded_engine)
    experimental_turns: list[dict[str, Any]] = []
    setup_actions: list[dict[str, Any]] = []
    direct_actions = 0
    first_player_verified = False
    steps = 0
    max_steps = int(request.get("max_atomic_steps", 4000))
    try:
        state = _reset_seeded_game(environment, list(deck), int(game["engine_seed_u32"]))
        while not state.done and steps < max_steps:
            observation = state.obs
            first = _first_player(observation)
            if first is None:
                action = _turn_order_action(module, observation)
                if action is None:
                    raise R215SeededMirrorRuntimeError("unsealed setup state has no r195 first_if_allowed action")
                setup_actions.append({"action": action, "acting_seat": (observation.get("current") or {}).get("yourIndex")})
                state = environment.step_batch([action]).envs[0]
                steps += 1
                continue
            if first != pair_seal.get("first_player_seat"):
                raise R215SeededMirrorRuntimeError("game first player disagrees with sealed pair material")
            first_player_verified = True
            turn_order = _turn_order_action(module, observation)
            if turn_order is not None:
                setup_actions.append({"action": turn_order, "acting_seat": (observation.get("current") or {}).get("yourIndex")})
                state = environment.step_batch([turn_order]).envs[0]
                steps += 1
                continue
            identity = engine_turn_identity_from_observation(observation)
            acting_seat = identity.seat
            if acting_seat == int(game["experimental_seat"]):
                controlled_observation = _observation_for_controller(observation, search_policy)
                decision = controller.act(
                    R215TurnIdentity(acting_seat, identity.actual_turn_id),
                    controlled_observation,
                )
                action = _validate_action(observation, decision.selected_action)
                row = dict(decision.receipt)
                diagnostics = dict(planner.last_diagnostics)
                row["root_visits"] = int(diagnostics.get("root_visits", 0) or 0)
                row["selected_action_from_policy_last_result"] = bool(
                    diagnostics.get("selected_action_from_policy_last_result", False)
                )
                row["fixed_simulation_target_or_completion_rate_reported"] = False
                row["chance_label"] = R215_CHANCE_LABEL
                row["selected_action"] = action
                experimental_turns.append(row)
            elif acting_seat == int(game["control_seat"]):
                action = _validate_action(
                    observation,
                    direct_policy.trusted_search_or_greedy_select(dict(observation), search=False),
                )
                direct_actions += 1
            else:
                raise R215SeededMirrorRuntimeError("engine emitted an unknown acting seat")
            state = environment.step_batch([action]).envs[0]
            steps += 1
        controller.finish_actual_turn()
        if not first_player_verified:
            raise R215SeededMirrorRuntimeError("game ended before first-player seal was verified")
        if not state.done:
            raise R215SeededMirrorRuntimeError("game exhausted max atomic steps before terminal state")
        return {
            "schema": SCHEMA,
            "operation": "run_game",
            "evaluation_id": request["evaluation_id"],
            "source_identity_sha256": request["source_identity_sha256"],
            "output_identity_sha256": request["output_identity_sha256"],
            **dict(game),
            "first_player_seat": pair_seal["first_player_seat"],
            "experimental_actual_turn_order": "first" if int(game["experimental_seat"]) == int(pair_seal["first_player_seat"]) else "second",
            "experimental_strategy": STRATEGY,
            "terminal_status": "completed",
            "winner_seat": state.winner,
            "steps": steps,
            "invalid_action": False,
            "crash": False,
            "packaged_turn_order_actions": setup_actions,
            "experimental_turn_receipts": experimental_turns,
            "direct_atomic_actions": direct_actions,
            "runtime_receipt": runtime,
        }
    finally:
        environment.close()


def run_seeded_mirror_operation(request: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one sealed-pair or one game operation in a fresh process."""

    operation = request.get("operation")
    if operation == "seal_pair":
        return _seal_pair(request)
    if operation == "run_game":
        return _run_game(request)
    raise R215SeededMirrorRuntimeError("unsupported r215 seeded mirror operation")
