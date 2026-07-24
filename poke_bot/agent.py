"""Timed MCTS / greedy policy agents for local play and submission parity."""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from . import cg_env, config, features
from .batched_infer import LeafPacket, RemoteLeafCancelled
from .belief import EmpiricalDeckPosterior, PublicBeliefHistory
from .belief_mcts import BeliefMCTS, factorize_visit_policy
from .matchup_adapter_activation import ShadowMatchupAdapterRouter
from .matchup_adapters import UNKNOWN_ROUTE
from .public_matchup_router import RuntimePublicMatchupRouter
from .mcts import GameClock, MCTS, MCTSResult
from .model import TemporalCabtTransformer, TemporalKVCache


def install_quiet_stdout(verbose: bool = False) -> None:
    """Silence this (worker) process's stdout unless ``verbose``.

    Redirects the *file descriptor* 1 to ``os.devnull`` so BOTH Python
    ``print()`` from rule-based baselines AND any C-level ``libcg`` chatter are
    swallowed — a pure ``sys.stdout`` swap would miss the C library. Only stdout
    is redirected; stderr is preserved for genuine crashes, and agent return
    values / inputs are untouched (we only discard their prints). Call once per
    worker process. No-op when ``verbose`` is True.
    """
    if verbose:
        return
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os.dup2(devnull_fd, 1)  # fd-level: catches libcg + python prints
        os.close(devnull_fd)
        # Point Python's high-level stdout at devnull too (buffered writers).
        sys.stdout = open(os.devnull, "w")
    except Exception:
        # Never let stdout policy break a worker.
        pass


def _fail_closed_legal(obs_dict: dict, preferred: list[int], rng: random.Random) -> list[int]:
    """Clamp ``preferred`` into a legal select; never invent illegal indices."""
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return preferred  # deck return path
    options = sel.get("option") or []
    n = len(options)
    if n <= 0:
        return []
    lo = int(sel.get("minCount", 0) or 0)
    hi = min(int(sel.get("maxCount", 0) or 0), n)
    lo = max(0, min(lo, hi))
    # Deduplicate / filter preferred.
    clean: list[int] = []
    for x in preferred:
        try:
            xi = int(x)
        except (TypeError, ValueError):
            continue
        if 0 <= xi < n and xi not in clean:
            clean.append(xi)
    if lo <= len(clean) <= hi and clean:
        return clean[:hi]
    # Fallback: sample a legal count.
    if hi <= 0:
        return []
    k = rng.randint(lo, hi) if hi >= lo else hi
    if k <= 0:
        return []
    return rng.sample(range(n), k)


def _enum_token(value: object) -> str:
    """Normalize competition JSON enums without depending on one cg version."""
    if hasattr(value, "name"):
        value = getattr(value, "name")
    return "".join(character for character in str(value).lower() if character.isalnum())


def forced_go_first_action(obs_dict: dict) -> Optional[list[int]]:
    """Return the legal ``Yes`` choice for the explicit turn-order prompt.

    The competition has represented enums as both integers and strings across
    tools (``41``/``IsFirst`` and ``1``/``Yes``).  A recognized turn-order
    prompt fails closed if it does not contain exactly selectable ``Yes``
    semantics; it must never silently fall through to a learned or random
    choice that could elect to go second.
    """

    select = obs_dict.get("select") if isinstance(obs_dict, dict) else None
    if not isinstance(select, dict):
        return None
    raw_context = select.get("context")
    try:
        is_first = int(raw_context) == 41
    except (TypeError, ValueError):
        is_first = _enum_token(raw_context) == "isfirst"
    if not is_first:
        return None
    options = select.get("option") or []
    yes_indices: list[int] = []
    for index, option in enumerate(options):
        raw_type = option.get("type") if isinstance(option, dict) else getattr(option, "type", None)
        try:
            is_yes = int(raw_type) == 1
        except (TypeError, ValueError):
            is_yes = _enum_token(raw_type) == "yes"
        if is_yes:
            yes_indices.append(index)
    lo = int(select.get("minCount", 0) or 0)
    hi = min(int(select.get("maxCount", 0) or 0), len(options))
    if len(yes_indices) != 1 or not (lo <= 1 <= hi):
        raise RuntimeError(
            "IsFirst prompt does not expose one legal Yes option; refusing "
            "to choose turn order ambiguously"
        )
    return [yes_indices[0]]


@dataclass
class PolicyAgent:
    """NN agent: greedy argmax or timed MCTS."""

    model: Optional[TemporalCabtTransformer]
    deck: list[int]
    #: Privileged opposing deck. Accepted only in explicit oracle diagnostics;
    #: never in trusted training/evaluation or deployment.
    opponent_deck: Optional[list[int]] = None
    use_mcts: bool = False
    oracle_mode: bool = False
    belief_mcts: bool = False
    belief_posterior: Optional[EmpiricalDeckPosterior] = None
    checkpoint_digest: Optional[str] = None
    model_generation: int = 0
    max_context_override: Optional[int] = None
    device: Optional[torch.device] = None
    clock: Optional[GameClock] = None
    game_time_budget_s: Optional[float] = None
    game_watchdog_reserve_s: float = 30.0
    expected_search_decisions: int = 64
    max_sims: Optional[int] = None
    move_time_s: Optional[float] = None
    rng: random.Random = field(default_factory=random.Random)
    last_result: Optional[MCTSResult] = None
    collect_targets: bool = False
    sample_actions: bool = False
    #: Softmax temperature for ``sample_actions`` (>1 explores, <1 sharpens).
    action_temperature: float = 1.0
    targets: list[dict[str, Any]] = field(default_factory=list)
    #: Optional remote leaf backend (persistent GPU inference server). When set,
    #: MCTS leaf forwards run on the server and ``model`` may be None on this
    #: CPU worker (it only featurizes + searches on CPU).
    leaf_backend: Optional[Callable] = None
    #: Formal training/eval must propagate model/runtime errors so the caller
    #: invalidates the game instead of counting random fallback play.
    strict_runtime: bool = False
    #: How many times ``__call__`` had to fall closed to a legal random/empty
    #: select because MCTS/greedy raised. Non-zero is a loud signal that search
    #: / leaf-eval is broken (e.g. remote channel mis-wired).
    fail_closed_count: int = 0
    board_history: list[features.SparseVector] = field(default_factory=list)
    previous_action_history: list[Optional[features.SparseVector]] = field(
        default_factory=list
    )
    _kv_cache: Optional[TemporalKVCache] = None
    _previous_action_token: Optional[features.SparseVector] = None
    _fail_closed_logged: bool = False
    _matchup_shadow_failed_logged: bool = False
    belief_history: PublicBeliefHistory = field(default_factory=PublicBeliefHistory)
    #: Behavior-inert public-prefix audit. Recognized routes are traced per game
    #: and per search branch, but every model invocation remains explicitly -1.
    matchup_adapter_shadow: bool = True
    #: Activated public-tree routing is opt-in and never inferred from a deck
    #: label. The immutable tree artifact is supplied by the deployment.
    matchup_adapter_runtime: bool = False
    matchup_adapter_tree_path: Optional[str] = None
    _matchup_adapter_shadow_router: ShadowMatchupAdapterRouter = field(
        default_factory=ShadowMatchupAdapterRouter,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if int(self.expected_search_decisions) < 1:
            raise ValueError("expected_search_decisions must be positive")
        if self.device is None:
            if self.model is not None:
                self.device = next(self.model.parameters()).device
            else:
                self.device = torch.device("cpu")
        if self.model is not None:
            self.model.eval()
        env_runtime = os.environ.get("POKEBOT_MATCHUP_ADAPTER_RUNTIME", "").strip()
        if env_runtime:
            self.matchup_adapter_runtime = env_runtime.lower() in {
                "1", "true", "yes", "on"
            }
        if self.matchup_adapter_runtime:
            tree_path = (
                self.matchup_adapter_tree_path
                or os.environ.get("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", "").strip()
            )
            if not tree_path:
                raise ValueError(
                    "runtime matchup adapters require an activated public tree"
                )
            self.matchup_adapter_tree_path = str(tree_path)
            self._matchup_adapter_shadow_router = (
                RuntimePublicMatchupRouter.from_path(tree_path)
            )
            if self.model is not None:
                self.model.matchup_adapter_bank.enabled = True
                self.model.matchup_adapter_bank.requires_grad_(False)
        if self.use_mcts and not (self.oracle_mode or self.belief_mcts):
            raise ValueError(
                "single-world MCTS is oracle-only; use trusted belief_mcts "
                "for deployment-safe search"
            )
        if self.oracle_mode and self.belief_mcts:
            raise ValueError("oracle and belief MCTS modes are mutually exclusive")
        if self.belief_mcts:
            if self.belief_posterior is None:
                raise ValueError("belief MCTS requires an anonymous deck posterior")
            if not (self.checkpoint_digest or "").startswith("sha256:"):
                raise ValueError("belief MCTS requires immutable checkpoint digest")
        if self.opponent_deck is not None and not (
            self.use_mcts and self.oracle_mode and config.SEARCH.allow_oracle_deck
        ):
            raise ValueError(
                "exact opponent deck is privileged and forbidden outside "
                "explicit POKEBOT_ALLOW_ORACLE_DECK=1 oracle MCTS"
            )
        if self.clock is None and self.use_mcts:
            self.clock = GameClock(
                total_s=float(
                    self.game_time_budget_s
                    if self.game_time_budget_s is not None
                    else config.SEARCH.game_time_budget_s
                ),
                reserve_s=float(self.game_watchdog_reserve_s),
                    expected_search_decisions=int(
                        self.expected_search_decisions
                    ),
            )

    def reset_game(self) -> None:
        if self.use_mcts:
            self.clock = GameClock(
                total_s=float(
                    self.game_time_budget_s
                    if self.game_time_budget_s is not None
                    else config.SEARCH.game_time_budget_s
                ),
                reserve_s=float(self.game_watchdog_reserve_s),
                    expected_search_decisions=int(
                        self.expected_search_decisions
                    ),
            )
        self.targets.clear()
        self.board_history.clear()
        self.previous_action_history.clear()
        self._kv_cache = None
        self._previous_action_token = None
        self.last_result = None
        self.fail_closed_count = 0
        self._fail_closed_logged = False
        self._matchup_shadow_failed_logged = False
        self.belief_history = PublicBeliefHistory()
        self._matchup_adapter_shadow_router.reset_for_new_game()

    def matchup_adapter_shadow_snapshot(self) -> dict[str, Any]:
        """Return the current shadow or activated causal route audit."""

        router = self._matchup_adapter_shadow_router
        if isinstance(router, RuntimePublicMatchupRouter):
            return router.snapshot()
        return router.audit.snapshot()

    def _matchup_model_route(self) -> int:
        """Return a causal public route; checkpoint config remains final gate."""

        if not (self.matchup_adapter_shadow or self.matchup_adapter_runtime):
            return UNKNOWN_ROUTE
        return self._matchup_adapter_shadow_router.candidate_model_route

    def _history_context_limit(self) -> int:
        """Resolve the checkpoint/job-specific history window.

        Remote GPU-leaf agents intentionally have ``model=None``. Their job's
        immutable ``model_max_context`` override must therefore win over the
        process-global default, especially for game-bounded temporal checkpoints.
        """
        limit = (
            int(self.max_context_override)
            if self.max_context_override is not None
            else int(self.model.max_context)
            if self.model is not None
            else int(config.MODEL.max_context)
        )
        if limit <= 0:
            raise ValueError(f"history context limit must be positive, got {limit}")
        return limit

    @torch.no_grad()
    def greedy_select(self, obs_dict: dict) -> list[int]:
        go_first = forced_go_first_action(obs_dict)
        if go_first is not None:
            # Preserve the setup decision in temporal history when the feature
            # contract is available, but the explicit go-first choice itself
            # must survive any optional telemetry/feature failure.
            try:
                features.assert_info_set(obs_dict)
                board = features.build_board_tokens(obs_dict, self.deck)
                self.board_history.append(board)
                self.previous_action_history.append(self._previous_action_token)
                max_context = self._history_context_limit()
                self.board_history = self.board_history[-max_context:]
                self.previous_action_history = self.previous_action_history[-max_context:]
                self._previous_action_token = features.build_option_tokens(
                    obs_dict, [go_first]
                )
                if self.collect_targets:
                    self.targets.append(
                        {
                            "observation": obs_dict,
                            "action": list(go_first),
                            "action_combos": [list(go_first)],
                            "policy": [1.0],
                            "value": None,
                            "visits": None,
                            "diagnostics": {
                                "target_source": "forced_go_first_contract",
                                "trusted": True,
                                "history_length": len(self.board_history),
                            },
                        }
                    )
            except Exception:
                pass
            return go_first
        features.assert_info_set(obs_dict)
        board = features.build_board_tokens(obs_dict, self.deck)
        self.board_history.append(board)
        self.previous_action_history.append(self._previous_action_token)
        max_context = self._history_context_limit()
        if len(self.board_history) > max_context:
            self.board_history = self.board_history[-max_context:]
            self.previous_action_history = self.previous_action_history[
                -max_context:
            ]
        obs = cg_env.to_observation(obs_dict)
        if obs.current is None:
            raise ValueError("greedy inference requires post-setup observation")
        root_seat = int(obs.current.yourIndex)
        cached_state = None
        cached_spatial = None

        def score(candidates: list[list[int]]) -> list[float]:
            nonlocal cached_state, cached_spatial
            options = features.build_option_tokens(obs_dict, candidates)
            matchup_route = self._matchup_model_route()
            if self.leaf_backend is not None:
                packet = LeafPacket(
                    obs=obs_dict,
                    your_deck=self.deck,
                    root_seat=root_seat,
                    history_boards=list(self.board_history),
                    history_previous_actions=list(self.previous_action_history),
                    action_combos_override=[list(c) for c in candidates],
                    matchup_route=matchup_route,
                )
                out = self.leaf_backend([packet])[0]
                if out.combos != candidates or len(out.priors) != len(candidates):
                    raise RuntimeError("factorized remote policy response mismatch")
                return list(out.priors)
            if self.model is None:
                raise RuntimeError(
                    "greedy inference requires a model or remote leaf backend"
                )
            if cached_state is None or cached_spatial is None:
                if (
                    self.model.decision_context == "history"
                    and self.model.kv_cache_enabled
                ):
                    model_out = self.model.forward(
                        board,
                        options,
                        kv_cache=self._kv_cache,
                        append_cache=True,
                        n_options=[len(candidates)],
                        previous_action=self._previous_action_token,
                        matchup_routes=[matchup_route],
                    )
                    self._kv_cache = model_out["kv_cache"]
                elif self.model.decision_context == "history":
                    model_out = self.model.forward_history_batch(
                        [self.board_history],
                        [options],
                        n_options=[len(candidates)],
                        previous_action_histories=[self.previous_action_history],
                        matchup_routes=[matchup_route],
                    )
                else:
                    model_out = self.model.forward(
                        board,
                        options,
                        append_cache=False,
                        n_options=[len(candidates)],
                        matchup_routes=[matchup_route],
                    )
                cached_state = self.model.matchup_policy_value_state(
                    model_out["state_vec"], [matchup_route]
                )
                cached_spatial = model_out["spatial_memory"]
                logits = model_out["policy_logits"][0, : len(candidates)]
            else:
                logits = self.model.decode_options(
                    options,
                    cached_spatial,
                    cached_state,
                    n_options=[len(candidates)],
                )[0, : len(candidates)]
            probs = torch.softmax(logits.float(), dim=-1)
            return [float(v) for v in probs.cpu().tolist()]

        prefix: list[int] = []
        factorized_stages: list[dict[str, Any]] = []
        while True:
            candidates = features.factorized_action_candidates(obs_dict, prefix)
            if len(candidates) == 1 and candidates[0] == prefix:
                selected = list(prefix)
                break
            policy = score(candidates)
            if self.sample_actions:
                temp = float(self.action_temperature)
                if temp > 0.0 and abs(temp - 1.0) > 1e-6:
                    # Re-temperature from stored probs via log-space.
                    import math

                    logits = [math.log(max(p, 1e-12)) / temp for p in policy]
                    m = max(logits)
                    exps = [math.exp(x - m) for x in logits]
                    z = sum(exps) or 1.0
                    weights = [e / z for e in exps]
                else:
                    weights = policy
                idx = self.rng.choices(
                    range(len(candidates)), weights=weights, k=1
                )[0]
            else:
                idx = max(range(len(candidates)), key=lambda i: policy[i])
            selected = list(candidates[idx])
            factorized_stages.append(
                {
                    "action_combos": [list(c) for c in candidates],
                    "policy": policy,
                    "selected_index": int(idx),
                }
            )
            if selected == prefix:
                break
            prefix = selected
        self._previous_action_token = features.build_option_tokens(
            obs_dict, [selected]
        )
        if self.collect_targets:
            self.targets.append(
                {
                    "observation": obs_dict,
                    "action": selected,
                    "action_combos": [list(selected)],
                    "policy": [1.0],
                    "factorized_stages": factorized_stages,
                    "value": None,
                    "visits": None,
                    "diagnostics": {
                        "target_source": "history_policy",
                        "trusted": True,
                        "history_length": len(self.board_history),
                    },
                }
            )
        return selected

    def mcts_select(self, obs_dict: dict) -> list[int]:
        if self.belief_mcts:
            return self.belief_mcts_select(obs_dict)
        if not self.oracle_mode:
            raise RuntimeError("single-world MCTS requires explicit oracle_mode")
        engine = MCTS(
            self.model,
            self.deck,
            device=self.device,
            opponent_deck_guess=self.opponent_deck,
            leaf_backend=self.leaf_backend,
            oracle_mode=True,
            matchup_shadow_router=(
                self._matchup_adapter_shadow_router.fork()
                if self.matchup_adapter_shadow
                else None
            ),
            matchup_model_route=self._matchup_model_route(),
        )
        result = engine.search(
            obs_dict,
            clock=self.clock,
            max_sims=self.max_sims,
            move_time_s=self.move_time_s,
        )
        self.last_result = result
        if self.collect_targets:
            # Capture our own info-set observation + MCTS visit target so the RL
            # loop can build AlphaZero training tuples (obs → visit-policy →
            # game-outcome value). Only OUR acting-seat decisions are recorded;
            # opponent turns are never seen here (info-set integrity).
            self.targets.append(
                {
                    "observation": obs_dict,
                    "action": list(result.select),
                    "action_combos": result.target.action_combos,
                    "policy": result.target.policy,
                    "value": result.target.value,
                    "visits": result.target.visits,
                    "diagnostics": result.target.diagnostics,
                    "target_source": "oracle_mcts",
                    "trusted": False,
                }
            )
        return list(result.select)

    def belief_mcts_select(self, obs_dict: dict) -> list[int]:
        """Trusted public-history root-sampled information-set search."""
        features.assert_info_set(obs_dict)
        board = features.build_board_tokens(obs_dict, self.deck)
        self.board_history.append(board)
        self.previous_action_history.append(self._previous_action_token)
        max_context = self._history_context_limit()
        self.board_history = self.board_history[-max_context:]
        self.previous_action_history = self.previous_action_history[-max_context:]
        max_sims = int(self.max_sims or max(128, config.SEARCH.sims_per_move))
        configured_move_time = float(
            self.move_time_s
            if self.move_time_s is not None
            else config.SEARCH.move_time_budget_s
        )
        move_time = configured_move_time
        if self.clock is not None:
            move_time = self.clock.next_move_budget(move_time)
        engine = BeliefMCTS(
            self.model,
            self.deck,
            self.belief_posterior,
            checkpoint_digest=str(self.checkpoint_digest),
            model_generation=int(self.model_generation),
            device=self.device,
            leaf_backend=self.leaf_backend,
            rng=self.rng,
            min_trusted_sims=128,
            max_context=max_context,
            matchup_shadow_router=(
                self._matchup_adapter_shadow_router.fork()
                if self.matchup_adapter_shadow
                else None
            ),
            matchup_model_route=self._matchup_model_route(),
        )
        result = engine.search(
            obs_dict,
            belief_history=self.belief_history,
            root_history_boards=self.board_history,
            root_history_previous_actions=self.previous_action_history,
            clock=self.clock,
            max_sims=max_sims,
            move_time_s=move_time,
        )
        self.last_result = result
        selected = list(result.select)
        self.belief_history.record_action(obs_dict, selected)
        self._previous_action_token = features.build_option_tokens(
            obs_dict, [selected]
        )
        if self.collect_targets:
            provenance = engine.target_provenance(
                max_sims=max_sims, move_time_s=configured_move_time
            )
            provenance["search_config"]["clock_allocation"] = (
                "adaptive_per_game_fair_share_with_watchdog_reserve"
            )
            hierarchical_stages = result.target.diagnostics.get(
                "factorized_stages"
            )
            self.targets.append(
                {
                    "observation": obs_dict,
                    "action": selected,
                    "action_combos": result.target.action_combos,
                    "policy": result.target.policy,
                    "factorized_stages": (
                        hierarchical_stages
                        if hierarchical_stages
                        else factorize_visit_policy(
                            obs_dict,
                            result.target.action_combos,
                            result.target.policy,
                            selected,
                        )
                    ),
                    "value": result.target.value,
                    "visits": result.target.visits,
                    "diagnostics": result.target.diagnostics,
                    "target_source": "belief_mcts",
                    "trusted": True,
                    "provenance": provenance,
                }
            )
        return selected

    def __call__(self, obs_dict: dict) -> list[int]:
        """Competition-style agent entry: deck when select is None."""
        if obs_dict is None or obs_dict.get("select") is None:
            return list(self.deck)
        if self.matchup_adapter_shadow or self.matchup_adapter_runtime:
            # The recognizer reads only causal public board/discard evidence.
            # Shadow failures are inert; activated routing is strict so a bad
            # artifact can never silently alter formal gameplay.
            try:
                self._matchup_adapter_shadow_router.observe(
                    obs_dict,
                    scope="game_root",
                    depth=len(self.board_history),
                )
            except Exception as exc:
                if self.matchup_adapter_runtime:
                    raise
                if not self._matchup_shadow_failed_logged:
                    self._matchup_shadow_failed_logged = True
                    print(
                        f"[PolicyAgent] matchup-adapter shadow audit failed closed: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
        try:
            if self.use_mcts:
                action = self.mcts_select(obs_dict)
            else:
                action = self.greedy_select(obs_dict)
        except RemoteLeafCancelled:
            # Parent-driven shutdown is not a search fault and must not emit a
            # fail-closed health violation while the iteration is aborting.
            raise
        except Exception as exc:
            # LOUD: silent random fallback produces instant fake games with
            # empty targets (no train data). Always log to stderr once/game.
            self.fail_closed_count += 1
            if not self._fail_closed_logged:
                self._fail_closed_logged = True
                print(
                    f"[PolicyAgent] FAIL-CLOSED (no search/leaf): "
                    f"{type(exc).__name__}: {exc} — using random legal select. "
                    f"use_mcts={self.use_mcts} leaf_backend="
                    f"{type(self.leaf_backend).__name__ if self.leaf_backend else None}",
                    file=sys.stderr,
                    flush=True,
                )
            if self.strict_runtime:
                raise RuntimeError(
                    f"policy runtime failed closed: {type(exc).__name__}: {exc}"
                ) from exc
            action = []
        return _fail_closed_legal(obs_dict, action, self.rng)


AgentFn = Callable[[dict], list[int]]


def play_game(
    agent0: AgentFn,
    agent1: AgentFn,
    deck0: list[int],
    deck1: list[int],
    *,
    max_steps: int = 4000,
) -> dict[str, Any]:
    """Play one local game; returns winner + length.

    Per-move exception isolation: if an agent raises (crash / illegal move that
    the engine rejects / timeout signal), that seat FORFEITS the game (the
    opponent wins) and the failing seat + error are reported so the caller can
    attribute the failure. An exception never propagates out of ``play_game``,
    so one broken agent can't abort a tournament worker pool.
    """
    obs, _start = cg_env.battle_start(deck0, deck1)
    steps = 0
    failed_seat: Optional[int] = None
    error: Optional[str] = None
    try:
        while obs is not None and not cg_env.is_finished(obs) and steps < max_steps:
            cur = obs.get("current") or {}
            seat = int(cur.get("yourIndex", 0))
            agent = agent0 if seat == 0 else agent1
            try:
                select = agent(obs)
                obs = cg_env.battle_select(select)
            except BaseException as exc:  # noqa: BLE001 - includes TimeoutError/alarm
                failed_seat = seat
                error = f"{type(exc).__name__}: {exc}"
                break
            steps += 1
        incomplete = (
            failed_seat is None
            and obs is not None
            and not cg_env.is_finished(obs)
        )
        if failed_seat is not None:
            winner = 1 - failed_seat  # crashing seat forfeits → opponent wins
        elif incomplete:
            winner = 2
        else:
            winner = cg_env.result_winner(obs) if obs else 2
            if winner is None:
                winner = 2
        return {
            "winner": int(winner),
            "steps": steps,
            "failed_seat": failed_seat,
            "error": error,
            "incomplete": incomplete,
            "termination": (
                "agent_failure"
                if failed_seat is not None
                else "max_steps"
                if incomplete
                else "completed"
            ),
        }
    finally:
        try:
            cg_env.battle_finish()
        except Exception:
            pass
