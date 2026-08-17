"""Timed MCTS / greedy policy agents for local play and submission parity."""

from __future__ import annotations

import copy
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import torch

from . import cg_env, config, features
from .batched_infer import LeafPacket, RemoteLeafCancelled
from .belief import EmpiricalDeckPosterior, PublicBeliefHistory
from .belief_mcts import BeliefMCTS, factorize_visit_policy
from .matchup_adapter_activation import ShadowMatchupAdapterRouter
from .matchup_adapters import UNKNOWN_ROUTE
from .own_deck_ledger import OwnDeckLedger
from .public_matchup_router import RuntimePublicMatchupRouter
from .mcts import GameClock, MCTS, MCTSResult
from .model import TemporalCabtTransformer, TemporalKVCache
from .poke_rlm import (
    PokeRLMAgentBridge,
    PokeRLMConfig,
    load_poke_rlm_config_from_env,
    resolve_poke_rlm_config_for_model,
)
from .recursive_turn_planner.agent_bridge import (
    RTPAgentBridge,
    resolve_rtp_config_for_model,
)
from .recursive_turn_planner.profiles import PURE_RL_R197_MAX_ACTION_COMBOS

# Slowking distill / reverse-engineered policy is Slowking-only research code.
# Never import it at module load — Crustle and other specialists must not depend
# on those packages. Lazy-load only when an explicit Slowking runtime arms it.


def _runtime_is_slowking_specialist() -> bool:
    """True only when the managed specialist/deck identity is Slowking."""
    for key in (
        "POKEBOT_CURRENT_DECK_GUIDE",
        "POKEBOT_SPECIALIST_ARCHETYPE",
        "PURE_RL_SPECIALIST_ARCHETYPE",
        "POKEBOT_SPECIALIST_ID",
    ):
        if "slowking" in os.environ.get(key, "").strip().lower():
            return True
    return False


def _slowking_distill_runtime_requested() -> bool:
    """Any Slowking-distill env arming flag (still Slowking-gated below)."""
    return any(
        os.environ.get(name, "").strip()
        for name in (
            "POKEBOT_SLOWKING_DISTILL_ENABLED",
            "POKEBOT_SLOWKING_DISTILL_MODE",
            "POKEBOT_SLOWKING_DISTILL_ACTOR_CKPT",
        )
    )


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


# Compatibility export for callers that historically imported the external
# turn-order parser from ``agent``.  The implementation is deliberately in
# torch-free ``features`` so isolated runner setup controls do not load torch.
forced_go_first_action = features.forced_go_first_action


def _legacy_ledger_keyword_error(exc: TypeError) -> bool:
    """Whether a pre-ledger model stub rejected an optional ledger keyword.

    The successor ledger must remain behavior-inert for old checkpoints and
    lightweight test doubles.  Only retry on Python's explicit unsupported
    keyword error; any model/runtime error still propagates normally.
    """

    message = str(exc)
    return "unexpected keyword argument" in message and any(
        name in message
        for name in (
            "ledger_snapshots",
            "ledger_histories",
            "ledger_option_features",
        )
    )


def _call_with_optional_ledger(
    method,
    *args,
    ledger_kwargs: dict[str, Any],
    allow_legacy_fallback: bool = True,
    **kwargs,
):
    """Call a model method with ledger inputs, tolerating legacy model stubs.

    An explicitly enabled successor is never allowed to discard its shared
    ledger input merely because a stale model implementation lacks the keyword.
    The retry is exclusively a dormant legacy compatibility path.
    """

    active = {key: value for key, value in ledger_kwargs.items() if value is not None}
    if not active:
        return method(*args, **kwargs)
    try:
        return method(*args, **kwargs, **active)
    except TypeError as exc:
        if not allow_legacy_fallback or not _legacy_ledger_keyword_error(exc):
            raise
        return method(*args, **kwargs)


def _make_own_deck_ledger(deck: list[int]) -> OwnDeckLedger:
    """Construct the successor side-store from the exact 60-card multiset."""

    if len(deck) != 60:
        raise ValueError(
            "own-deck ledger successor requires an exact 60-card starting deck"
        )
    return OwnDeckLedger(deck)


@dataclass
class PolicyAgent:
    """NN agent: greedy argmax, Recursive Turn Planner, or timed MCTS."""

    model: Optional[TemporalCabtTransformer]
    deck: list[int]
    #: Privileged opposing deck. Accepted only in explicit oracle diagnostics;
    #: never in trusted training/evaluation or deployment.
    opponent_deck: Optional[list[int]] = None
    use_mcts: bool = False
    #: Lightweight RTP path. Default off for production RL; enable when possible
    #: via ``POKEBOT_USE_RECURSIVE_TURN_PLANNER=1`` and a readable
    #: ``POKEBOT_RTP_CHECKPOINT``. Missing checkpoint falls back to greedy.
    use_recursive_turn_planner: bool = False
    #: Optional sizing profile override: ``pure_rl`` / ``global_transformer``.
    rtp_sizing_profile: Optional[str] = None
    rtp_online_sim_verify_budget: int = 0
    #: ``None`` preserves the legacy 256-action bridge cap.  The separately
    #: versioned checksum-bound ``pure_rl_r197`` profile instead requires the
    #: exact 1024-action cap so corpus/runtime action support cannot drift.
    rtp_max_action_combos: Optional[int] = None
    #: Isolated three-arm evaluation control.  The bridge remains active and
    #: enumerates the same complete ordered actions, but it is constrained to
    #: the direct-policy branch.  It is intentionally opt-in and has no
    #: serving authority by itself.
    force_direct_bridge_only: bool = False
    #: One-child, immutable evaluator context for the shadow-only r197
    #: sidecar.  It is accepted only by the sealed three-arm factory and is
    #: revalidated by ``RTPAgentBridge`` immediately before each select.
    #: Supplying it cannot grant serving, selector, replay, or submission
    #: authority.
    rtp_evaluation_action_execution: Optional[Mapping[str, Any]] = None
    #: Full PokeRLM kit attachment. Default disabled — preserves greedy/RTP/
    #: MCTS behavior until explicitly enabled via config or env flags.
    poke_rlm_config: Optional[PokeRLMConfig] = None
    poke_rlm_max_action_combos: int = 256
    #: Slowking-only distill research runtime. Never arms for Crustle/other
    #: specialists. Enable only via ``POKEBOT_SLOWKING_DISTILL_*`` on Slowking.
    slowking_distill_config: Optional[Any] = None
    slowking_distill_max_action_combos: int = 256
    oracle_mode: bool = False
    belief_mcts: bool = False
    #: Staged root-parallel search topology.  ``1`` retains the historical
    #: singleton path; ``8`` selects the handle-owned forest only when an
    #: explicitly staged runtime profile requests it.
    belief_mcts_lanes: int = 1
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
    min_trusted_sims: Optional[int] = None
    move_time_s: Optional[float] = None
    rng: random.Random = field(default_factory=random.Random)
    last_result: Optional[MCTSResult] = None
    _belief_forest: Optional[Any] = field(default=None, init=False, repr=False)
    collect_targets: bool = False
    sample_actions: bool = False
    #: Softmax temperature for ``sample_actions`` (>1 explores, <1 sharpens).
    action_temperature: float = 1.0
    targets: list[dict[str, Any]] = field(default_factory=list)
    #: Optional remote leaf backend (persistent GPU inference server). When set,
    #: MCTS leaf forwards run on the server and ``model`` may be None on this
    #: CPU worker (it only featurizes + searches on CPU).
    leaf_backend: Optional[Callable] = None
    #: Successor-only opt-in.  ``None`` resolves only from the checkpoint's
    #: typed capability; remote workers must pass this explicit constructor
    #: argument. Legacy r241/default agents remain completely dormant.
    own_deck_ledger_enabled: Optional[bool] = field(
        default=None,
        kw_only=True,
    )
    #: r287 bounded Alakazam checklist residual.  This is deliberately a
    #: policy-side, parameter-free route: it reads the live legal stage after
    #: the frozen neural policy (including Fusion, OwnDeck and matchup routes)
    #: has produced its score, so enabling it never changes a checkpoint or
    #: state_dict.  ``None`` resolves from the explicit environment arm and
    #: otherwise remains off.
    turn_checklist_logit_layer_enabled: Optional[bool] = field(
        default=None,
        kw_only=True,
    )
    #: Optional immutable r287 layer config.  Supplying a path does not arm
    #: the layer by itself; ``turn_checklist_logit_layer_enabled`` remains the
    #: explicit gate.
    turn_checklist_logit_layer_config_path: Optional[str] = field(
        default=None,
        kw_only=True,
    )
    #: Revision-9 Alakazam public-rule semantic sidecar.  This is supplied
    #: explicitly by the checksum-bound derivative runtime; ordinary/r274
    #: agents never infer or instantiate it from deck identity or ambient
    #: environment state.
    public_rule_semantic_projection: Optional[Any] = field(
        default=None,
        kw_only=True,
        repr=False,
    )
    public_rule_semantic_projection_enabled: bool = field(
        default=False,
        kw_only=True,
    )
    public_rule_semantic_projection_gate: float = field(
        default=1.0,
        kw_only=True,
    )
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
    #: Immutable public own-deck residual snapshots aligned one-for-one with
    #: ``board_history``.  This is a side store rather than an observation
    #: mutation, so it can never expose hidden deck/prize realizations.
    ledger_history: list[Any] = field(default_factory=list, init=False)
    own_deck_ledger: Optional[OwnDeckLedger] = field(init=False, repr=False)
    _kv_cache: Optional[TemporalKVCache] = None
    _previous_action_token: Optional[features.SparseVector] = None
    _fail_closed_logged: bool = False
    _matchup_shadow_failed_logged: bool = False
    #: Set only when trusted deployment search failed and the exact same
    #: information state was re-run through the frozen greedy policy.
    last_search_fallback_reason: Optional[str] = None
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
    _rtp_bridge: Optional[RTPAgentBridge] = field(
        default=None, init=False, repr=False
    )
    #: Stable, read-only-after-selection telemetry for isolated evaluation.
    #: It is intentionally separate from target collection so formal RTP
    #: evidence does not need to create training-eligible examples.
    last_rtp_diagnostics: Optional[dict[str, Any]] = field(
        default=None, init=False, repr=False
    )
    _poke_rlm_bridge: Optional[PokeRLMAgentBridge] = field(
        default=None, init=False, repr=False
    )
    last_poke_rlm_trace: Optional[dict[str, Any]] = field(
        default=None, init=False, repr=False
    )
    _slowking_distill_bridge: Optional[Any] = field(
        default=None, init=False, repr=False
    )
    last_slowking_distill_trace: Optional[dict[str, Any]] = field(
        default=None, init=False, repr=False
    )
    #: Latest factorized-stage audit emitted by the bounded r287 checklist
    #: route.  Kept separate from checkpoint/model traces and cleared per
    #: game/selection so stale advice can never be mistaken for current state.
    last_turn_checklist_logit_trace: Optional[dict[str, Any]] = field(
        default=None, init=False, repr=False
    )
    public_rule_semantic_projection_apply_count: int = field(
        default=0, init=False
    )

    def __post_init__(self) -> None:
        self.public_rule_semantic_projection_enabled = bool(
            self.public_rule_semantic_projection_enabled
        )
        if self.public_rule_semantic_projection_enabled:
            if self.public_rule_semantic_projection is None or not callable(
                getattr(self.public_rule_semantic_projection, "apply_to_logits", None)
            ):
                raise ValueError(
                    "enabled public-rule semantic projection requires a valid sidecar"
                )
            if self.model is None:
                raise ValueError(
                    "public-rule semantic projection requires a local policy model"
                )
            gate = float(self.public_rule_semantic_projection_gate)
            if not math.isfinite(gate) or gate == 0.0:
                raise ValueError(
                    "enabled public-rule semantic projection requires a finite nonzero gate"
                )
            self.public_rule_semantic_projection_gate = gate
        # The checklist route is consciously policy-side and default-off.  It
        # cannot be inferred from deck identity, model metadata, or config
        # presence: an operator must choose the constructor flag or the named
        # environment arm at a receipt-backed boundary.
        if self.turn_checklist_logit_layer_enabled is None:
            checklist_env = os.environ.get(
                "POKEBOT_ALAKAZAM_TURN_CHECKLIST_LOGIT_LAYER", ""
            ).strip().lower()
            self.turn_checklist_logit_layer_enabled = checklist_env in {
                "1", "true", "yes", "on"
            }
        else:
            self.turn_checklist_logit_layer_enabled = bool(
                self.turn_checklist_logit_layer_enabled
            )
        if self.turn_checklist_logit_layer_config_path is None:
            checklist_config_path = os.environ.get(
                "POKEBOT_ALAKAZAM_TURN_CHECKLIST_LOGIT_CONFIG", ""
            ).strip()
            if checklist_config_path:
                self.turn_checklist_logit_layer_config_path = checklist_config_path
        # Keep the ledger local to a game/policy instance.  ``OwnDeckLedger``
        # copies the starting multiset and exposes only immutable public
        # snapshots, so later deck-list mutations cannot retroactively alter a
        # decision input.
        model_ledger_enabled = bool(
            getattr(self.model, "own_deck_ledger_enabled", False)
        )
        model_ledger_runtime_enabled = bool(
            getattr(self.model, "own_deck_ledger_runtime_enabled", False)
        )
        if self.own_deck_ledger_enabled is None:
            self.own_deck_ledger_enabled = model_ledger_enabled
        else:
            self.own_deck_ledger_enabled = bool(self.own_deck_ledger_enabled)
        # A local receipt-enabled successor may never be paired with a neutral
        # PolicyAgent side-store.  That would make the direct local path
        # silently bypass a model input which batched inference already treats
        # as mandatory.  ``model=None`` remains a supported remote-backend
        # worker shape and is checked by its packet receiver instead.
        if model_ledger_runtime_enabled and not self.own_deck_ledger_enabled:
            raise ValueError(
                "own_deck_ledger_runtime_enabled model requires an enabled "
                "PolicyAgent own-deck ledger"
            )
        # Conversely, an explicitly selected local successor cannot be
        # attached to a historical model that lacks the physical adapter.  The
        # model capability marker is intentional here: lightweight remote
        # workers retain ``model=None`` and validate their remote endpoint.
        if (
            self.own_deck_ledger_enabled
            and self.model is not None
            and not model_ledger_enabled
        ):
            raise ValueError(
                "enabled PolicyAgent own-deck ledger requires a model with "
                "physical own-deck ledger capability"
            )
        self.own_deck_ledger = None
        if int(self.expected_search_decisions) < 1:
            raise ValueError("expected_search_decisions must be positive")
        if self.min_trusted_sims is not None:
            if int(self.min_trusted_sims) < 1:
                raise ValueError("min_trusted_sims must be positive")
            if (
                self.max_sims is not None
                and int(self.min_trusted_sims) > int(self.max_sims)
            ):
                raise ValueError("min_trusted_sims cannot exceed max_sims")
        if self.device is None:
            if self.model is not None:
                self.device = next(self.model.parameters()).device
            else:
                self.device = torch.device("cpu")
        if self.model is not None:
            self.model.eval()
        env_rtp = os.environ.get("POKEBOT_USE_RECURSIVE_TURN_PLANNER", "").strip().lower()
        if env_rtp in {"1", "true", "yes", "on"}:
            from pathlib import Path as _Path

            ckpt = os.environ.get("POKEBOT_RTP_CHECKPOINT", "").strip()
            allow_untrained = os.environ.get(
                "POKEBOT_RTP_ALLOW_UNTRAINED", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            # "When possible": only arm RTP if a trained sidecar exists (or
            # untrained is explicitly allowed). Otherwise keep greedy.
            self.use_recursive_turn_planner = bool(
                (ckpt and _Path(ckpt).is_file()) or allow_untrained
            )
        elif env_rtp in {"0", "false", "no", "off"}:
            self.use_recursive_turn_planner = False
        if self.own_deck_ledger_enabled:
            if self.use_mcts:
                raise ValueError(
                    "own-deck ledger successor is direct-policy-only; MCTS is forbidden"
                )
            if self.use_recursive_turn_planner:
                raise ValueError(
                    "own-deck ledger successor forbids recursive turn planning"
                )
            self.own_deck_ledger = _make_own_deck_ledger(self.deck)
        env_rtp_profile = os.environ.get("POKEBOT_RTP_SIZING_PROFILE", "").strip()
        if env_rtp_profile and self.rtp_sizing_profile is None:
            self.rtp_sizing_profile = env_rtp_profile
        if self.use_recursive_turn_planner and self.use_mcts:
            # Search remains an explicit override; RTP is the non-MCTS path.
            pass
        if self.use_recursive_turn_planner and self.model is not None:
            self._init_rtp_bridge()
        if self.rtp_evaluation_action_execution is not None:
            if (
                self._rtp_bridge is None
                or self._rtp_bridge.config.sizing_profile != "pure_rl_r197"
            ):
                raise ValueError(
                    "evaluation_action_execution requires a pure_rl_r197 RTP bridge"
                )
        if self.force_direct_bridge_only and self._rtp_bridge is None:
            raise ValueError(
                "force_direct_bridge_only requires an initialized RTP bridge"
            )
        self.poke_rlm_config = load_poke_rlm_config_from_env(self.poke_rlm_config)
        if (
            self.poke_rlm_config is not None
            and self.poke_rlm_config.runs_planner
            and self.model is not None
        ):
            self._init_poke_rlm_bridge()
        # Slowking policy path: import/load only for Slowking specialists.
        if self.slowking_distill_config is None and (
            _runtime_is_slowking_specialist() and _slowking_distill_runtime_requested()
        ):
            from .slowking_distill import (
                load_runtime_config_from_env as load_slowking_distill_config_from_env,
            )

            self.slowking_distill_config = load_slowking_distill_config_from_env(
                self.slowking_distill_config
            )
        elif (
            self.slowking_distill_config is not None
            and not _runtime_is_slowking_specialist()
        ):
            # Refuse non-Slowking specialist identities (e.g. Crustle).
            self.slowking_distill_config = None
        if (
            self.slowking_distill_config is not None
            and self.slowking_distill_config.runs
            and _runtime_is_slowking_specialist()
        ):
            self._init_slowking_distill_bridge()
        else:
            self._slowking_distill_bridge = None
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
            if int(self.belief_mcts_lanes) not in (1, 8):
                raise ValueError("belief MCTS lanes must be exactly 1 or 8")
            if int(self.belief_mcts_lanes) == 8 and self.model is None:
                raise ValueError("eight-lane belief MCTS requires a local frozen model")
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

    def _init_rtp_bridge(self) -> None:
        if self.model is None:
            self._rtp_bridge = None
            return
        rtp_config = resolve_rtp_config_for_model(
            self.model,
            profile_name=self.rtp_sizing_profile,
            online_sim_verify_budget=self.rtp_online_sim_verify_budget,
        )
        if rtp_config.sizing_profile == "pure_rl_r197":
            if (
                self.rtp_max_action_combos is not None
                and int(self.rtp_max_action_combos) != PURE_RL_R197_MAX_ACTION_COMBOS
            ):
                raise ValueError(
                    "pure_rl_r197 requires rtp_max_action_combos="
                    f"{PURE_RL_R197_MAX_ACTION_COMBOS}"
                )
            rtp_max_action_combos = PURE_RL_R197_MAX_ACTION_COMBOS
        else:
            # Preserve the historical legacy bridge cap.  The r197/r198
            # complete-action expansion is explicit above and must not widen
            # unrelated RTP profiles.
            rtp_max_action_combos = (
                256
                if self.rtp_max_action_combos is None
                else int(self.rtp_max_action_combos)
            )
            if rtp_max_action_combos < 1:
                raise ValueError("rtp_max_action_combos must be positive")
        self._rtp_bridge = RTPAgentBridge(
            model=self.model,
            deck=list(self.deck),
            config=rtp_config,
            get_matchup_route=self._matchup_model_route,
            get_board_history=lambda: self.board_history,
            get_previous_action_history=lambda: self.previous_action_history,
            get_previous_action_token=lambda: self._previous_action_token,
            get_kv_cache=lambda: self._kv_cache,
            set_kv_cache=lambda cache: setattr(self, "_kv_cache", cache),
            max_action_combos=rtp_max_action_combos,
            evaluation_action_execution=self.rtp_evaluation_action_execution,
        )

    def _init_poke_rlm_bridge(self) -> None:
        if self.model is None or self.poke_rlm_config is None:
            self._poke_rlm_bridge = None
            return
        cfg = resolve_poke_rlm_config_for_model(
            self.model, base=self.poke_rlm_config
        )
        self.poke_rlm_config = cfg
        self._poke_rlm_bridge = PokeRLMAgentBridge(
            model=self.model,
            deck=list(self.deck),
            config=cfg,
            get_matchup_route=self._matchup_model_route,
            get_board_history=lambda: self.board_history,
            get_previous_action_history=lambda: self.previous_action_history,
            get_previous_action_token=lambda: self._previous_action_token,
            get_kv_cache=lambda: self._kv_cache,
            set_kv_cache=lambda cache: setattr(self, "_kv_cache", cache),
            max_action_combos=int(self.poke_rlm_max_action_combos),
        )

    def _init_slowking_distill_bridge(self) -> None:
        cfg = self.slowking_distill_config
        if cfg is None or not cfg.runs or not _runtime_is_slowking_specialist():
            self._slowking_distill_bridge = None
            return
        from .slowking_distill import SlowkingDistillAgentBridge
        from .slowking_distill.belief_search_backend import BeliefSearchBundle

        bundle = None
        if (
            cfg.use_belief_mcts
            and self.model is not None
            and self.belief_posterior is not None
            and (self.checkpoint_digest or "").startswith("sha256:")
        ):
            bundle = BeliefSearchBundle(
                model=self.model,
                deck=list(self.deck),
                posterior=self.belief_posterior,
                checkpoint_digest=str(self.checkpoint_digest),
                device=self.device,
                max_sims=int(cfg.search_budget_sims),
                move_time_s=float(cfg.search_move_time_s),
                leaf_backend=self.leaf_backend,
            )
        self._slowking_distill_bridge = SlowkingDistillAgentBridge(
            config=cfg,
            search_bundle=bundle,
            device=self.device,
        )
        if not self._slowking_distill_bridge.ready:
            # Missing checkpoint → leave bridge unset (fail closed to RTP/greedy).
            self._slowking_distill_bridge = None

    def _slowking_distill_legal(
        self, obs_dict: dict
    ) -> list[list[int]]:
        try:
            combos = features.enumerate_action_combos(
                obs_dict, max_combos=int(self.slowking_distill_max_action_combos)
            )
            return [list(c) for c in combos]
        except features.ActionSpaceTooLarge:
            stage = features.factorized_action_candidates(obs_dict, [])
            return [list(c) for c in stage]

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
        if not hasattr(self, "ledger_history"):
            self.ledger_history = []
        self.ledger_history.clear()
        if (
            getattr(self, "own_deck_ledger_enabled", False)
            and getattr(self, "own_deck_ledger", None) is not None
        ):
            self.own_deck_ledger.reset()
        self._kv_cache = None
        self._previous_action_token = None
        self.last_result = None
        self.fail_closed_count = 0
        self._fail_closed_logged = False
        self._matchup_shadow_failed_logged = False
        self.belief_history = PublicBeliefHistory()
        self._matchup_adapter_shadow_router.reset_for_new_game()
        if self._rtp_bridge is not None:
            self._rtp_bridge.reset_game()
        self.last_rtp_diagnostics = None
        if self._poke_rlm_bridge is not None:
            self._poke_rlm_bridge.reset_game()
        self.last_poke_rlm_trace = None
        if (
            self._slowking_distill_bridge is not None
            and self._slowking_distill_bridge.runtime is not None
        ):
            # Runtime has no game-level cache; clear decision counters only.
            self._slowking_distill_bridge.runtime._decisions = 0
            self._slowking_distill_bridge.runtime._search_calls = 0
        self.last_slowking_distill_trace = None
        self.last_turn_checklist_logit_trace = None

    def matchup_adapter_shadow_snapshot(self) -> dict[str, Any]:
        """Return the current shadow or activated causal route audit."""

        router = self._matchup_adapter_shadow_router
        if isinstance(router, RuntimePublicMatchupRouter):
            return router.snapshot()
        return router.audit.snapshot()

    def close_belief_search_lanes(self) -> None:
        """Join staged native lane workers without guessing a handle destructor."""

        forest = self._belief_forest
        self._belief_forest = None
        if forest is not None:
            forest.close()

    def trusted_search_or_greedy_select(
        self,
        obs_dict: dict,
        *,
        search: bool,
    ) -> list[int]:
        """Run trusted search transactionally, falling back to greedy policy.

        Belief search mutates temporal/public-history state before it can know
        whether the minimum trusted simulation floor will fit its deadline.
        Snapshotting that small per-game state lets a deadline or search error
        replay the same information state through the frozen greedy policy,
        rather than returning a random legal action or double-appending the
        observation to temporal history.
        """

        if not search or not self.use_mcts:
            prior_use_mcts = self.use_mcts
            self.use_mcts = False
            self.last_search_fallback_reason = None
            try:
                return self(obs_dict)
            finally:
                self.use_mcts = prior_use_mcts

        snapshot = {
            "board_history": list(self.board_history),
            "previous_action_history": list(self.previous_action_history),
            "ledger_history": list(getattr(self, "ledger_history", [])),
            # Search may mutate the ledger before it knows whether it will be
            # trusted.  Fork its side-store so the greedy fallback replays the
            # exact same public information state once, rather than retaining
            # speculative updates.
            "own_deck_ledger": (
                self.own_deck_ledger.fork()
                if getattr(self, "own_deck_ledger", None) is not None
                else None
            ),
            "kv_cache": self._kv_cache,
            "previous_action_token": copy.deepcopy(self._previous_action_token),
            "last_result": self.last_result,
            "belief_history": copy.deepcopy(self.belief_history),
            "router": self._matchup_adapter_shadow_router.fork(),
            "clock": copy.deepcopy(self.clock),
            "targets_length": len(self.targets),
            "fail_closed_count": self.fail_closed_count,
            "fail_closed_logged": self._fail_closed_logged,
            "matchup_shadow_failed_logged": self._matchup_shadow_failed_logged,
            "rng_state": self.rng.getstate(),
        }
        prior_strict = self.strict_runtime
        self.strict_runtime = True
        self.last_search_fallback_reason = None
        try:
            return self(obs_dict)
        except Exception as exc:
            self.board_history = snapshot["board_history"]
            self.previous_action_history = snapshot[
                "previous_action_history"
            ]
            self.ledger_history = snapshot["ledger_history"]
            self.own_deck_ledger = snapshot["own_deck_ledger"]
            self._kv_cache = snapshot["kv_cache"]
            self._previous_action_token = snapshot["previous_action_token"]
            self.last_result = snapshot["last_result"]
            self.belief_history = snapshot["belief_history"]
            self._matchup_adapter_shadow_router = snapshot["router"]
            self.clock = snapshot["clock"]
            del self.targets[snapshot["targets_length"] :]
            self.fail_closed_count = snapshot["fail_closed_count"]
            self._fail_closed_logged = snapshot["fail_closed_logged"]
            self._matchup_shadow_failed_logged = snapshot[
                "matchup_shadow_failed_logged"
            ]
            self.rng.setstate(snapshot["rng_state"])
            self.last_search_fallback_reason = (
                f"{type(exc).__name__}: {exc}"
            )
            prior_use_mcts = self.use_mcts
            self.use_mcts = False
            try:
                # Keep strict runtime enabled for the frozen-policy fallback.
                # If greedy inference itself fails, propagate to the outer
                # submission fail-safe rather than silently choosing random.
                return self(obs_dict)
            finally:
                self.use_mcts = prior_use_mcts
        finally:
            self.strict_runtime = prior_strict

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

    def _require_own_deck_direct_policy(self, route: str) -> None:
        """Reject planners even if a caller mutates a constructed policy."""

        if getattr(self, "own_deck_ledger_enabled", False):
            raise RuntimeError(
                "own-deck ledger successor is direct-policy-only; "
                f"{route} is forbidden"
            )

    def _observe_own_deck_ledger(self, obs_dict: dict) -> Any:
        """Advance the public ledger for one real root observation.

        Every decision path calls this immediately before appending temporal
        history.  Factorized stages intentionally reuse the returned immutable
        snapshot; speculative search packets never call it.
        """

        # Older pickled agents predate the field.  Lazily restoring the neutral
        # side-store keeps those callers compatible without changing their
        # board/option feature path.
        if not getattr(self, "own_deck_ledger_enabled", False):
            return None
        ledger = getattr(self, "own_deck_ledger", None)
        if ledger is None:
            # An explicitly enabled successor fails closed if its starting
            # multiset cannot form a valid ledger.  Never silently downgrade a
            # selected successor path to stale static-deck features.
            ledger = _make_own_deck_ledger(self.deck)
            self.own_deck_ledger = ledger
        return ledger.observe(obs_dict)

    def _append_history_snapshot(
        self,
        board: features.SparseVector,
        ledger_snapshot: Any,
    ) -> None:
        """Append the three temporally aligned per-decision inputs together."""

        self.board_history.append(board)
        self.previous_action_history.append(self._previous_action_token)
        if getattr(self, "own_deck_ledger_enabled", False):
            self.ledger_history.append(ledger_snapshot)
        max_context = self._history_context_limit()
        if len(self.board_history) > max_context:
            self.board_history = self.board_history[-max_context:]
            self.previous_action_history = self.previous_action_history[
                -max_context:
            ]
            if getattr(self, "own_deck_ledger_enabled", False):
                self.ledger_history = self.ledger_history[-max_context:]

    @staticmethod
    def _ledger_option_features(
        ledger_snapshot: Any,
        obs_dict: dict,
        candidates: list[list[int]],
    ) -> Any:
        """Derive legal-option rows from a frozen public ledger snapshot."""

        if ledger_snapshot is None:
            return None
        option_features = getattr(ledger_snapshot, "option_features", None)
        if not callable(option_features):
            return None
        return option_features(obs_dict, candidates)

    @staticmethod
    def _public_rule_factorized_stage_representation(
        obs_dict: dict,
        candidates: list[list[int]],
        factorized_prefix: list[int],
    ) -> dict[str, Any]:
        """Align r298 public semantics to the live factorized candidate rows.

        The adapter describes the simulator's atomic option menu, whereas the
        policy scores complete factorized prefixes.  At one factorized stage
        every candidate is either the exact STOP suffix or appends exactly one
        atomic option.  Reusing that appended option's normalized public
        semantics gives the sidecar one row per policy logit without encoding
        the candidate ordinal or a private/global serial.
        """

        from .alakazam_public_rule_adapter_r298 import (
            build_public_rule_representation,
        )

        representation = build_public_rule_representation(obs_dict).to_dict()
        raw_options = list(representation.get("options") or [])
        stage_options: list[dict[str, Any]] = []
        prefix = list(factorized_prefix)
        for candidate in candidates:
            row = list(candidate)
            if row == prefix:
                stage_options.append(
                    {
                        "semantic": {
                            "selection": copy.deepcopy(representation["selection"]),
                            "option": {"option_type": "pass"},
                        },
                        "semantic_key_sha256": None,
                        "referenced_card_ids": [],
                        "referenced_attack_ids": [],
                    }
                )
                continue
            if row[: len(prefix)] != prefix or len(row) != len(prefix) + 1:
                raise RuntimeError(
                    "public-rule sidecar received a malformed factorized candidate"
                )
            option_index = int(row[-1])
            if option_index < 0 or option_index >= len(raw_options):
                raise RuntimeError(
                    "public-rule sidecar candidate is outside the simulator option menu"
                )
            stage_options.append(copy.deepcopy(raw_options[option_index]))
        representation["options"] = stage_options
        return representation

    def _record_turn_checklist_not_evaluated(
        self,
        *,
        selected_action: list[int],
        reason: str,
        candidate_rows: Optional[list[list[int]]] = None,
        factorized_prefix: Optional[list[int]] = None,
        factorized_stage_index: Optional[int] = None,
        selected_candidate_index: Optional[int] = None,
    ) -> None:
        """Record an explicit eight-question audit when no residual ran.

        Forced setup controls and non-greedy policy authorities do not expose a
        post-neural factorized score to this layer.  They must therefore not
        look like a zero-evidence evaluation: this compact, action-inert trace
        says that every checklist answer is unavailable instead.  Keeping the
        schema aligned with ordinary stage traces makes an omitted evaluation
        distinguishable from an evaluated-but-neutral stage.
        """

        if not bool(getattr(self, "turn_checklist_logit_layer_enabled", False)):
            return

        channel_names = (
            "ko_hand_threshold",
            "safe_spend_above_threshold",
            "replacement_alakazam_line",
            "unavoidable_draws_before_attack",
            "bench_prize_exposure",
            "immediate_disruption_outcome",
            "unknown_prize_robust_line",
            "terminal_before_forced_draw",
        )
        rows = [list(row) for row in (candidate_rows or [])]
        selected = list(selected_action)
        prefix = list(factorized_prefix or [])
        if selected_candidate_index is None:
            try:
                selected_candidate_index = rows.index(selected)
            except ValueError:
                selected_candidate_index = None
        zeros = [0.0] * len(rows)
        unavailable = [False] * len(rows)

        def channel(name: str) -> dict[str, Any]:
            return {
                "name": name,
                "raw": list(zeros),
                "normalized": list(zeros),
                "option_availability": list(unavailable),
                "available": False,
                "status": "unavailable",
                "reason": reason,
            }

        channels = [channel(name) for name in channel_names]
        guide_support = channel("guide_support")
        stage = {
            "enabled": True,
            "applied": False,
            "evaluation_mode": "not_evaluated",
            "action_authority": False,
            "score_space": "not_evaluated",
            "factorized_stage_index": factorized_stage_index,
            "factorized_prefix": prefix,
            "candidate_rows": rows,
            "selected_candidate_index": selected_candidate_index,
            # ``selected_stage_index`` is the explicit candidate position in
            # this factorized stage.  Keep ``selected_candidate_index`` too so
            # downstream readers do not confuse it with the stage ordinal.
            "selected_stage_index": selected_candidate_index,
            "selected_candidate": selected,
            "whole_decision_budget_initial": 0.10,
            "whole_decision_budget_before": 0.10,
            "whole_decision_budget_consumed": 0.0,
            "whole_decision_budget_remaining": 0.10,
            "base_argmax_index": None,
            "adjusted_argmax_index": None,
            "stage_argmax_changed_at_prefix": False,
            # Deprecated compatibility alias.  This has *never* described a
            # coupled whole-policy/action counterfactual.
            "action_changed_from_base_policy": False,
            "channel_names": list(channel_names),
            "channels": channels,
            "guide_support": guide_support,
            "scalar_gates": {},
            "residuals": list(zeros),
            "facts": {},
            "available": False,
            "active": False,
            "reason": reason,
            "explicit_invocation": False,
            "normalized_channel_vectors": {
                name: list(zeros) for name in channel_names
            },
            "normalized_guide_support_vector": list(zeros),
            "channel_status": {
                name: {
                    "available": False,
                    "status": "unavailable",
                    "reason": reason,
                }
                for name in (*channel_names, "guide_support")
            },
            "channel_option_availability": {
                name: list(unavailable)
                for name in (*channel_names, "guide_support")
            },
        }
        self.last_turn_checklist_logit_trace = {
            "enabled": True,
            "evaluation_mode": "not_evaluated",
            "action_authority": False,
            "selected_action": selected,
            "reason": reason,
            "stages": [stage],
            "whole_decision_budget_initial": 0.10,
            "whole_decision_budget_consumed": 0.0,
            "whole_decision_budget_remaining": 0.10,
        }

    def _record_go_first(self, obs_dict: dict, go_first: list[int]) -> list[int]:
        self._record_turn_checklist_not_evaluated(
            selected_action=list(go_first),
            reason="forced_go_first_contract",
            candidate_rows=[list(go_first)],
            factorized_prefix=[],
            factorized_stage_index=0,
            selected_candidate_index=0,
        )
        try:
            features.assert_info_set(obs_dict)
            ledger_snapshot = self._observe_own_deck_ledger(obs_dict)
            board = features.build_board_tokens(obs_dict, self.deck)
            self._append_history_snapshot(board, ledger_snapshot)
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
            if getattr(self, "own_deck_ledger_enabled", False):
                raise
            pass
        return go_first

    def _append_decision_history(
        self, obs_dict: dict
    ) -> features.SparseVector:
        features.assert_info_set(obs_dict)
        ledger_snapshot = self._observe_own_deck_ledger(obs_dict)
        board = features.build_board_tokens(obs_dict, self.deck)
        self._append_history_snapshot(board, ledger_snapshot)
        return board

    @torch.no_grad()
    def _factorized_greedy_prepared(
        self,
        obs_dict: dict,
        board: features.SparseVector,
        *,
        target_source: str = "history_policy",
        extra_diagnostics: Optional[dict[str, Any]] = None,
    ) -> list[int]:
        """Factorized greedy assuming board history was already appended."""
        # The checklist layer is evaluated at each factorized prefix, but its
        # authority is a *whole-decision* absolute budget.  A selected path may
        # therefore consume at most 0.10 total |logit residual| even when the
        # legal action takes multiple factorized stages.
        whole_decision_budget_initial = 0.10
        whole_decision_budget_remaining = whole_decision_budget_initial
        obs = cg_env.to_observation(obs_dict)
        if obs.current is None:
            raise ValueError("greedy inference requires post-setup observation")
        root_seat = int(obs.current.yourIndex)
        cached_state = None
        cached_spatial = None
        cached_fusion_state = None
        # The root observation is recorded once by ``_append_decision_history``
        # before this method runs.  Keep its immutable ledger snapshot and its
        # aligned temporal history fixed for every factorized stage; only the
        # legal-option rows are recomputed as the prefix changes.
        ledger_snapshot = (
            self.ledger_history[-1] if getattr(self, "ledger_history", []) else None
        )
        ledger_history = list(getattr(self, "ledger_history", []))
        allow_legacy_ledger_fallback = not bool(
            getattr(self, "own_deck_ledger_enabled", False)
        )
        checklist_enabled = bool(
            getattr(self, "turn_checklist_logit_layer_enabled", False)
        )
        checklist_stage_traces: list[dict[str, Any]] = []
        # Never leave an audit from a prior decision visible while the current
        # factorized decision is still being decoded.
        self.last_turn_checklist_logit_trace = None

        def _trace_payload(trace: Any) -> dict[str, Any]:
            """Serialize a layer trace without coupling Agent to its class."""

            to_dict = getattr(trace, "to_dict", None)
            if callable(to_dict):
                payload = to_dict()
            elif isinstance(trace, Mapping):
                payload = dict(trace)
            else:
                payload = {"trace": repr(trace)}
            return payload if isinstance(payload, dict) else {"trace": repr(payload)}

        def _argmax_index(values: list[float]) -> int | None:
            if not values:
                return None
            return max(range(len(values)), key=values.__getitem__)

        def _trace_residuals(trace: Any, *, width: int) -> list[float]:
            """Read the module's aligned raw-logit residual vector strictly."""

            payload = _trace_payload(trace)
            values = payload.get("residuals")
            if not isinstance(values, (list, tuple)) or len(values) != width:
                raise ValueError("checklist trace residual width mismatch")
            residuals = [float(value) for value in values]
            if any(not math.isfinite(value) for value in residuals):
                raise ValueError("checklist trace residual is non-finite")
            if any(abs(value) > 0.100001 for value in residuals):
                raise ValueError("checklist trace residual exceeded module cap")
            return residuals

        def _budgeted_residuals(
            residuals: list[float],
        ) -> tuple[list[float], list[float], float]:
            """Restrict one stage to the unspent selected-path budget."""

            before = max(0.0, float(whole_decision_budget_remaining))
            applied = [max(-before, min(before, value)) for value in residuals]
            return applied, list(residuals), before

        def _remote_priors_with_residual(
            probabilities: list[float], residuals: list[float]
        ) -> list[float]:
            """Apply exact trace logits to remote priors without reviving zeros."""

            if len(probabilities) != len(residuals):
                raise ValueError("remote prior/residual width mismatch")
            if not any(abs(value) > 0.0 for value in residuals):
                # An exact no-op must preserve remote values byte-for-value;
                # this also retains any intentional zero-probability support.
                return list(probabilities)
            total = sum(probabilities)
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError("remote prior mass is invalid")
            supported = [
                index for index, value in enumerate(probabilities) if value > 0.0
            ]
            if not supported:
                raise ValueError("remote prior has no positive support")
            log_scores = [float("-inf")] * len(probabilities)
            for index in supported:
                log_scores[index] = (
                    math.log(probabilities[index] / total) + residuals[index]
                )
            maximum = max(log_scores[index] for index in supported)
            weights = [0.0] * len(probabilities)
            for index in supported:
                weights[index] = math.exp(log_scores[index] - maximum)
            normalizer = sum(weights)
            if not math.isfinite(normalizer) or normalizer <= 0.0:
                raise ValueError("remote checklist normalization is invalid")
            return [value / normalizer for value in weights]

        def _append_checklist_trace(
            trace: Any,
            *,
            base_scores: list[float],
            adjusted_scores: list[float],
            score_space: str,
            candidates: list[list[int]],
            factorized_prefix: list[int],
            factorized_stage_index: int,
            module_residuals: list[float],
            applied_residuals: list[float],
            budget_before: float,
        ) -> None:
            """Attach exact base-vs-adjusted stage choice telemetry.

            Softmax preserves argmax, so raw local logits and remote priors
            both answer whether this bounded route changed the policy's legal
            stage argmax at the prefix actually evaluated.  This is not a
            whole-action counterfactual: a prior-stage change can lead to a
            different later prefix.  Candidate rows and the prefix are carried
            with the vector so every residual component remains auditable.
            """

            payload = _trace_payload(trace)
            # The module's vector is preserved separately from the actual
            # action-authoritative vector when the selected-path budget clips a
            # later stage.  Consumers must read ``residuals`` for what was
            # actually applied to this policy score.
            payload["module_residuals_before_whole_decision_cap"] = list(
                module_residuals
            )
            payload["residuals"] = list(applied_residuals)
            base_index = _argmax_index(base_scores)
            adjusted_index = _argmax_index(adjusted_scores)
            stage_argmax_changed = (
                base_index is not None
                and adjusted_index is not None
                and base_index != adjusted_index
            )
            payload.update(
                {
                    "enabled": True,
                    "applied": True,
                    "evaluation_mode": "evaluated",
                    "action_authority": True,
                    "score_space": score_space,
                    "factorized_stage_index": int(factorized_stage_index),
                    "factorized_prefix": list(factorized_prefix),
                    "candidate_rows": [list(row) for row in candidates],
                    "selected_candidate_index": None,
                    "selected_stage_index": None,
                    "selected_candidate": None,
                    "whole_decision_budget_initial": whole_decision_budget_initial,
                    "whole_decision_budget_before": budget_before,
                    "whole_decision_budget_consumed": None,
                    "whole_decision_budget_remaining": None,
                    "base_argmax_index": base_index,
                    "adjusted_argmax_index": adjusted_index,
                    "stage_argmax_changed_at_prefix": stage_argmax_changed,
                    # Deprecated compatibility alias.  Consumers must use the
                    # canonical name above and must not treat either field as
                    # a final sampled-action or whole-policy comparison.
                    "action_changed_from_base_policy": stage_argmax_changed,
                }
            )
            checklist_stage_traces.append(payload)

        def _apply_checklist_logits(
            logits: torch.Tensor,
            candidates: list[list[int]],
            *,
            factorized_prefix: list[int],
            factorized_stage_index: int,
        ) -> torch.Tensor:
            """Apply the opt-in residual after all frozen model score routes."""

            if not checklist_enabled:
                return logits
            try:
                from .alakazam_turn_checklist_logit_layer import (
                    apply_turn_checklist_logits,
                )

                base_logits = logits.detach().clone()
                adjusted, trace = apply_turn_checklist_logits(
                    observation=obs_dict,
                    candidates=candidates,
                    deck=self.deck,
                    logits=logits,
                    ledger_snapshot=ledger_snapshot,
                    config_path=getattr(
                        self, "turn_checklist_logit_layer_config_path", None
                    ),
                )
                if not isinstance(adjusted, torch.Tensor):
                    raise TypeError("checklist logit layer returned non-tensor logits")
                if adjusted.shape != logits.shape:
                    raise ValueError("checklist logit layer changed policy width")
                if adjusted.device != logits.device:
                    raise ValueError("checklist logit layer changed policy device")
                if not bool(torch.isfinite(adjusted).all().item()):
                    raise ValueError("checklist logit layer returned non-finite logits")
                if adjusted.numel() and float(
                    torch.max(torch.abs(adjusted - logits)).item()
                ) > 0.100001:
                    raise ValueError("checklist logit layer exceeded residual cap")
                module_residuals = _trace_residuals(trace, width=logits.numel())
                applied_residuals, raw_residuals, budget_before = _budgeted_residuals(
                    module_residuals
                )
                residual_tensor = torch.as_tensor(
                    applied_residuals,
                    dtype=logits.dtype,
                    device=logits.device,
                )
                adjusted = logits + residual_tensor
                if not bool(torch.isfinite(adjusted).all().item()):
                    raise ValueError("budgeted checklist logits are non-finite")
                _append_checklist_trace(
                    trace,
                    base_scores=[
                        float(value) for value in base_logits.cpu().tolist()
                    ],
                    adjusted_scores=[
                        float(value)
                        for value in adjusted.detach().cpu().tolist()
                    ],
                    score_space="local_logits",
                    candidates=candidates,
                    factorized_prefix=factorized_prefix,
                    factorized_stage_index=factorized_stage_index,
                    module_residuals=raw_residuals,
                    applied_residuals=applied_residuals,
                    budget_before=budget_before,
                )
                return adjusted
            except Exception as exc:
                # The bounded residual is an optional advisory path.  A bad
                # config or an unavailable fact must reduce exactly to the
                # frozen neural policy, not make an otherwise legal turn fail.
                checklist_stage_traces.append(
                    {
                        "enabled": True,
                        "applied": False,
                        "evaluation_mode": "evaluated",
                        "action_authority": False,
                        "reason": "checklist_layer_unavailable",
                        "error": f"{type(exc).__name__}: {exc}",
                        "score_space": "local_logits",
                        "factorized_stage_index": int(factorized_stage_index),
                        "factorized_prefix": list(factorized_prefix),
                        "candidate_rows": [list(row) for row in candidates],
                        "selected_candidate_index": None,
                        "selected_stage_index": None,
                        "selected_candidate": None,
                        "whole_decision_budget_initial": whole_decision_budget_initial,
                        "whole_decision_budget_before": whole_decision_budget_remaining,
                        "whole_decision_budget_consumed": 0.0,
                        "whole_decision_budget_remaining": (
                            whole_decision_budget_remaining
                        ),
                        "base_argmax_index": None,
                        "adjusted_argmax_index": None,
                        "stage_argmax_changed_at_prefix": False,
                        "action_changed_from_base_policy": False,
                    }
                )
                return logits

        def _apply_checklist_probabilities(
            probabilities: list[float],
            candidates: list[list[int]],
            *,
            factorized_prefix: list[int],
            factorized_stage_index: int,
        ) -> list[float]:
            """Apply the same bounded residual to remote leaf priors.

            The remote server intentionally returns only normalized priors.  Its
            probabilities are therefore converted inside the layer to legal
            log scores, shifted by the same residual, and renormalized before
            this factorized argmax/sampler sees them.
            """

            if not checklist_enabled:
                return probabilities
            try:
                from .alakazam_turn_checklist_logit_layer import (
                    apply_turn_checklist_probabilities,
                )

                base_probabilities = [float(value) for value in probabilities]
                adjusted, trace = apply_turn_checklist_probabilities(
                    observation=obs_dict,
                    candidates=candidates,
                    deck=self.deck,
                    probabilities=base_probabilities,
                    ledger_snapshot=ledger_snapshot,
                    config_path=getattr(
                        self, "turn_checklist_logit_layer_config_path", None
                    ),
                )
                adjusted_list = [float(value) for value in adjusted]
                if len(adjusted_list) != len(base_probabilities):
                    raise ValueError("checklist layer changed remote policy width")
                if (
                    any(
                        not math.isfinite(value) or value < 0.0
                        for value in adjusted_list
                    )
                    or sum(adjusted_list) <= 0.0
                ):
                    raise ValueError(
                        "checklist layer returned invalid remote probabilities"
                    )
                module_residuals = _trace_residuals(
                    trace, width=len(base_probabilities)
                )
                applied_residuals, raw_residuals, budget_before = _budgeted_residuals(
                    module_residuals
                )
                # Do not derive a remote logit adjustment from changed
                # probabilities.  The trace vector is the same raw-logit
                # residual used locally, while zero prior support remains
                # impossible after reweighting.
                adjusted_list = _remote_priors_with_residual(
                    base_probabilities, applied_residuals
                )
                _append_checklist_trace(
                    trace,
                    base_scores=base_probabilities,
                    adjusted_scores=adjusted_list,
                    score_space="remote_priors",
                    candidates=candidates,
                    factorized_prefix=factorized_prefix,
                    factorized_stage_index=factorized_stage_index,
                    module_residuals=raw_residuals,
                    applied_residuals=applied_residuals,
                    budget_before=budget_before,
                )
                return adjusted_list
            except Exception as exc:
                checklist_stage_traces.append(
                    {
                        "enabled": True,
                        "applied": False,
                        "evaluation_mode": "evaluated",
                        "action_authority": False,
                        "reason": "checklist_layer_unavailable",
                        "error": f"{type(exc).__name__}: {exc}",
                        "score_space": "remote_priors",
                        "factorized_stage_index": int(factorized_stage_index),
                        "factorized_prefix": list(factorized_prefix),
                        "candidate_rows": [list(row) for row in candidates],
                        "selected_candidate_index": None,
                        "selected_stage_index": None,
                        "selected_candidate": None,
                        "whole_decision_budget_initial": whole_decision_budget_initial,
                        "whole_decision_budget_before": whole_decision_budget_remaining,
                        "whole_decision_budget_consumed": 0.0,
                        "whole_decision_budget_remaining": (
                            whole_decision_budget_remaining
                        ),
                        "base_argmax_index": None,
                        "adjusted_argmax_index": None,
                        "stage_argmax_changed_at_prefix": False,
                        "action_changed_from_base_policy": False,
                    }
                )
                return probabilities

        def score(
            candidates: list[list[int]],
            *,
            factorized_prefix: list[int],
            factorized_stage_index: int,
        ) -> list[float]:
            nonlocal cached_state, cached_spatial, cached_fusion_state
            options = features.build_option_tokens(obs_dict, candidates)
            ledger_option_features = self._ledger_option_features(
                ledger_snapshot,
                obs_dict,
                candidates,
            )
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
                    ledger_snapshot=ledger_snapshot,
                    history_ledger_snapshots=list(ledger_history),
                    ledger_option_features=ledger_option_features,
                )
                out = self.leaf_backend([packet])[0]
                if out.combos != candidates or len(out.priors) != len(candidates):
                    raise RuntimeError("factorized remote policy response mismatch")
                return _apply_checklist_probabilities(
                    list(out.priors),
                    candidates,
                    factorized_prefix=factorized_prefix,
                    factorized_stage_index=factorized_stage_index,
                )
            if self.model is None:
                raise RuntimeError(
                    "greedy inference requires a model or remote leaf backend"
                )
            semantic_projection_enabled = bool(
                self.public_rule_semantic_projection_enabled
            )
            option_hidden = None
            if cached_state is None or cached_spatial is None:
                if (
                    self.model.decision_context == "history"
                    and self.model.kv_cache_enabled
                ):
                    model_out = _call_with_optional_ledger(
                        self.model.forward,
                        board,
                        options,
                        kv_cache=self._kv_cache,
                        append_cache=True,
                        n_options=[len(candidates)],
                        previous_action=self._previous_action_token,
                        matchup_routes=[matchup_route],
                        ledger_kwargs={
                            "ledger_snapshots": [ledger_snapshot],
                            "ledger_option_features": [ledger_option_features],
                        },
                        allow_legacy_fallback=allow_legacy_ledger_fallback,
                    )
                    self._kv_cache = model_out["kv_cache"]
                elif self.model.decision_context == "history":
                    model_out = _call_with_optional_ledger(
                        self.model.forward_history_batch,
                        [self.board_history],
                        [options],
                        n_options=[len(candidates)],
                        previous_action_histories=[self.previous_action_history],
                        matchup_routes=[matchup_route],
                        ledger_kwargs={
                            "ledger_histories": [ledger_history],
                            "ledger_option_features": [ledger_option_features],
                        },
                        allow_legacy_fallback=allow_legacy_ledger_fallback,
                    )
                else:
                    model_out = _call_with_optional_ledger(
                        self.model.forward,
                        board,
                        options,
                        append_cache=False,
                        n_options=[len(candidates)],
                        matchup_routes=[matchup_route],
                        ledger_kwargs={
                            "ledger_snapshots": [ledger_snapshot],
                            "ledger_option_features": [ledger_option_features],
                        },
                        allow_legacy_fallback=allow_legacy_ledger_fallback,
                    )
                cached_state = self.model.matchup_policy_value_state(
                    model_out["state_vec"], [matchup_route]
                )
                cached_fusion_state = model_out["state_vec"]
                cached_spatial = model_out["spatial_memory"]
                logits = model_out["policy_logits"][0, : len(candidates)]
                if semantic_projection_enabled:
                    # The ordinary forward API deliberately does not retain
                    # shared option states.  Re-decode the same already-
                    # encoded board/state once with return_hidden=True; this
                    # is deterministic and leaves every frozen model tensor
                    # untouched.
                    decoded = _call_with_optional_ledger(
                        self.model.decode_options,
                        options,
                        cached_spatial,
                        cached_state,
                        n_options=[len(candidates)],
                        return_hidden=True,
                        decision_fusion_state_vec=cached_fusion_state,
                        ledger_kwargs={
                            "ledger_option_features": [ledger_option_features],
                        },
                        allow_legacy_fallback=allow_legacy_ledger_fallback,
                    )
                    if not isinstance(decoded, tuple):
                        raise RuntimeError(
                            "public-rule sidecar requires option-hidden decoder output"
                        )
                    decoded_logits, option_hidden = decoded
                    logits = decoded_logits[0, : len(candidates)]
            else:
                decoded = _call_with_optional_ledger(
                    self.model.decode_options,
                    options,
                    cached_spatial,
                    cached_state,
                    n_options=[len(candidates)],
                    return_hidden=semantic_projection_enabled,
                    decision_fusion_state_vec=cached_fusion_state,
                    ledger_kwargs={
                        "ledger_option_features": [ledger_option_features],
                    },
                    allow_legacy_fallback=allow_legacy_ledger_fallback,
                )
                if semantic_projection_enabled:
                    if not isinstance(decoded, tuple):
                        raise RuntimeError(
                            "public-rule sidecar requires option-hidden decoder output"
                        )
                    decoded_logits, option_hidden = decoded
                    logits = decoded_logits[0, : len(candidates)]
                else:
                    logits = decoded[0, : len(candidates)]
            if semantic_projection_enabled:
                if option_hidden is None:
                    raise RuntimeError(
                        "public-rule sidecar option-hidden state is unavailable"
                    )
                representation = self._public_rule_factorized_stage_representation(
                    obs_dict,
                    candidates,
                    factorized_prefix,
                )
                logits = self.public_rule_semantic_projection.apply_to_logits(
                    logits.unsqueeze(0),
                    option_hidden[:, : len(candidates)],
                    [representation],
                    runtime_enabled=True,
                    gate=self.public_rule_semantic_projection_gate,
                )[0]
                self.public_rule_semantic_projection_apply_count += 1
            logits = _apply_checklist_logits(
                logits,
                candidates,
                factorized_prefix=factorized_prefix,
                factorized_stage_index=factorized_stage_index,
            )
            probs = torch.softmax(logits.float(), dim=-1)
            return [float(v) for v in probs.cpu().tolist()]

        prefix: list[int] = []
        factorized_stages: list[dict[str, Any]] = []
        while True:
            candidates = features.factorized_action_candidates(obs_dict, prefix)
            if len(candidates) == 1 and candidates[0] == prefix:
                selected = list(prefix)
                break
            factorized_stage_index = len(factorized_stages)
            trace_count_before_score = len(checklist_stage_traces)
            policy = score(
                candidates,
                factorized_prefix=prefix,
                factorized_stage_index=factorized_stage_index,
            )
            if self.sample_actions:
                temp = float(self.action_temperature)
                if temp > 0.0 and abs(temp - 1.0) > 1e-6:
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
            if (
                checklist_enabled
                and len(checklist_stage_traces) == trace_count_before_score + 1
            ):
                # The residual trace is created while scores are available;
                # selection happens immediately afterwards.  Fill in the
                # exact row it led to without re-running a policy decode.
                stage_trace = checklist_stage_traces[-1]
                residuals = stage_trace.get("residuals", [])
                consumed = 0.0
                if stage_trace.get("applied") is True:
                    if (
                        not isinstance(residuals, list)
                        or len(residuals) != len(candidates)
                    ):
                        raise RuntimeError(
                            "checklist stage trace residuals are misaligned"
                        )
                    try:
                        selected_residual = float(residuals[idx])
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise RuntimeError(
                            "checklist selected residual is malformed"
                        ) from exc
                    if not math.isfinite(selected_residual):
                        raise RuntimeError("checklist selected residual is non-finite")
                    consumed = abs(selected_residual)
                    before = float(
                        stage_trace.get(
                            "whole_decision_budget_before",
                            whole_decision_budget_remaining,
                        )
                    )
                    if consumed > before + 1e-9:
                        raise RuntimeError(
                            "checklist stage exceeded whole-decision budget"
                        )
                    whole_decision_budget_remaining = max(0.0, before - consumed)
                stage_trace.update(
                    {
                        "selected_candidate_index": int(idx),
                        "selected_stage_index": int(idx),
                        "selected_candidate": list(selected),
                        "whole_decision_budget_consumed": consumed,
                        "whole_decision_budget_remaining": (
                            whole_decision_budget_remaining
                        ),
                    }
                )
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
        if checklist_enabled:
            if not checklist_stage_traces:
                # A forced singleton factorized stage has no neural score and
                # therefore no legitimate residual application.  Preserve the
                # eight-channel audit as explicitly unavailable rather than
                # emitting an empty successful-looking checklist trace.
                self._record_turn_checklist_not_evaluated(
                    selected_action=list(selected),
                    reason="factorized_greedy_no_scored_stage",
                    candidate_rows=[list(selected)],
                    factorized_prefix=list(selected),
                    factorized_stage_index=len(factorized_stages),
                    selected_candidate_index=0,
                )
            else:
                self.last_turn_checklist_logit_trace = {
                    "enabled": True,
                    "selected_action": list(selected),
                    "stages": checklist_stage_traces,
                    "whole_decision_budget_initial": whole_decision_budget_initial,
                    "whole_decision_budget_consumed": (
                        whole_decision_budget_initial - whole_decision_budget_remaining
                    ),
                    "whole_decision_budget_remaining": whole_decision_budget_remaining,
                }
        if self.collect_targets:
            diagnostics = {
                "target_source": target_source,
                "trusted": True,
                "history_length": len(self.board_history),
            }
            if extra_diagnostics:
                diagnostics.update(extra_diagnostics)
            if self.last_turn_checklist_logit_trace is not None:
                diagnostics["turn_checklist_logit_layer"] = copy.deepcopy(
                    self.last_turn_checklist_logit_trace
                )
            self.targets.append(
                {
                    "observation": obs_dict,
                    "action": selected,
                    "action_combos": [list(selected)],
                    "policy": [1.0],
                    "factorized_stages": factorized_stages,
                    "value": None,
                    "visits": None,
                    "diagnostics": diagnostics,
                }
            )
        return selected

    @torch.no_grad()
    def greedy_select(self, obs_dict: dict) -> list[int]:
        self.last_turn_checklist_logit_trace = None
        go_first = forced_go_first_action(obs_dict)
        if go_first is not None:
            # Preserve the setup decision in temporal history when the feature
            # contract is available, but the explicit go-first choice itself
            # must survive any optional telemetry/feature failure.
            return self._record_go_first(obs_dict, go_first)
        board = self._append_decision_history(obs_dict)
        return self._factorized_greedy_prepared(obs_dict, board)

    @torch.no_grad()
    def rtp_select(self, obs_dict: dict) -> list[int]:
        """Recursive Turn Planner select with greedy fallback."""
        self._require_own_deck_direct_policy("recursive turn planning")
        go_first = forced_go_first_action(obs_dict)
        if go_first is not None:
            return self._record_go_first(obs_dict, go_first)
        if self._rtp_bridge is None:
            if self.model is not None and self.use_recursive_turn_planner:
                self._init_rtp_bridge()
            if self._rtp_bridge is None:
                return self.greedy_select(obs_dict)

        board = self._append_decision_history(obs_dict)

        def greedy_fallback(_obs: dict) -> list[int]:
            # Keep a single trusted provenance for the game: fallbacks remain
            # recursive_turn_planner with rtp_mode=fallback diagnostics.
            return self._factorized_greedy_prepared(
                obs_dict,
                board,
                target_source="rtp_greedy_fallback",
                extra_diagnostics={
                    "rtp_fallback_reason": (
                        self._rtp_bridge.last_diagnostics.fallback_reason
                        if self._rtp_bridge is not None
                        else "bridge_missing"
                    ),
                },
            )

        selected = self._rtp_bridge.select(
            obs_dict,
            board=board,
            greedy_fallback=greedy_fallback,
            force_direct_bridge_only=self.force_direct_bridge_only,
        )
        diag = self._rtp_bridge.last_diagnostics
        # Fail closed on incomplete/illegal ordered actions instead of
        # submitting them to the engine or teacher-forcing pipeline.
        try:
            features.factorized_teacher_forcing_stages(obs_dict, list(selected))
        except ValueError:
            if diag.mode != "fallback":
                diag.mode = "fallback"
                diag.fallback_code = "illegal_ordered_action"
                diag.fallback_detail = "factorized_teacher_forcing_validation"
                diag.fallback_reason = "illegal_ordered_action"
                self._rtp_bridge.last_diagnostics = diag
            selected = greedy_fallback(obs_dict)
            diag = self._rtp_bridge.last_diagnostics
        # Fallback path already set the previous-action token inside greedy.
        if diag.mode != "fallback":
            self._previous_action_token = features.build_option_tokens(
                obs_dict, [selected]
            )
            if self.collect_targets:
                self.targets.append(
                    {
                        "observation": obs_dict,
                        "action": list(selected),
                        "action_combos": [list(selected)],
                        "policy": [1.0],
                        "value": None,
                        "visits": None,
                        "diagnostics": {
                            "target_source": "recursive_turn_planner",
                            "trusted": False,
                            "history_length": len(self.board_history),
                            "rtp_mode": diag.mode,
                            "rtp_turn_key": list(diag.turn_key),
                            "rtp_legal_count": diag.legal_count,
                            "rtp_used_option_hidden": diag.used_option_hidden,
                        },
                    }
                )
        self.last_rtp_diagnostics = diag.as_dict()
        return selected

    def rtp_diagnostic_snapshot(self) -> Optional[dict[str, Any]]:
        """Return the most recent bridge decision telemetry for evaluation.

        The method deliberately returns a shallow serialized copy instead of
        the bridge object or a mutable planner state.  Callers can audit a
        decision but cannot alter the action path through this API.
        """

        if self.last_rtp_diagnostics is None:
            return None
        return dict(self.last_rtp_diagnostics)

    def _slowking_distill_shadow_after(
        self, obs_dict: dict, selected: list[int]
    ) -> None:
        """Shadow telemetry after another policy chose ``selected``."""
        cfg = self.slowking_distill_config
        if cfg is None or not cfg.shadow_only:
            return
        if self._slowking_distill_bridge is None:
            self._init_slowking_distill_bridge()
        if self._slowking_distill_bridge is None:
            return
        legal = self._slowking_distill_legal(obs_dict)
        try:
            decision = self._slowking_distill_bridge.shadow(
                obs_dict,
                legal,
                committed_action=selected,
            )
            diag = self._slowking_distill_bridge.last_diagnostics
            self.last_slowking_distill_trace = {
                "enabled": True,
                "mode": cfg.mode.value,
                "used_for_selection": False,
                "committed_action": list(selected),
                "would_action": list(decision.action) if decision else [],
                "source": diag.source,
                "critical": diag.critical,
                "search_used": diag.search_used,
                "search_backend": diag.search_backend,
                "reason": diag.reason,
                "trace": diag.trace,
            }
        except Exception as exc:  # noqa: BLE001 — shadow must never raise
            self.last_slowking_distill_trace = {
                "enabled": True,
                "mode": cfg.mode.value,
                "reason": "shadow_failure",
                "error": f"{type(exc).__name__}: {exc}",
                "selected_action": list(selected),
                "used_for_selection": False,
            }

    @torch.no_grad()
    def slowking_distill_select(self, obs_dict: dict) -> list[int]:
        """Slowking distill active select with greedy fallback."""
        go_first = forced_go_first_action(obs_dict)
        if go_first is not None:
            return self._record_go_first(obs_dict, go_first)
        if self._slowking_distill_bridge is None:
            self._init_slowking_distill_bridge()
        if self._slowking_distill_bridge is None:
            return self.greedy_select(obs_dict)

        board = self._append_decision_history(obs_dict)
        legal = self._slowking_distill_legal(obs_dict)
        selected = self._slowking_distill_bridge.select(obs_dict, legal)
        diag = self._slowking_distill_bridge.last_diagnostics
        if selected is None:
            return self._factorized_greedy_prepared(
                obs_dict,
                board,
                target_source="slowking_distill_greedy_fallback",
                extra_diagnostics={
                    "slowking_distill_fallback_reason": diag.fallback_reason
                    or diag.reason
                    or "bridge_unready",
                },
            )
        self._previous_action_token = features.build_option_tokens(
            obs_dict, [selected]
        )
        self.last_slowking_distill_trace = {
            "enabled": True,
            "mode": (
                self.slowking_distill_config.mode.value
                if self.slowking_distill_config is not None
                else "unknown"
            ),
            "used_for_selection": True,
            "action": list(selected),
            "source": diag.source,
            "critical": diag.critical,
            "search_used": diag.search_used,
            "search_backend": diag.search_backend,
            "reason": diag.reason,
            "trace": diag.trace,
        }
        if self.collect_targets:
            self.targets.append(
                {
                    "observation": obs_dict,
                    "action": list(selected),
                    "action_combos": [list(selected)],
                    "policy": [1.0],
                    "value": None,
                    "visits": None,
                    "diagnostics": {
                        "target_source": "slowking_distill",
                        "trusted": False,
                        "history_length": len(self.board_history),
                        "slowking_distill_source": diag.source,
                        "slowking_distill_critical": diag.critical,
                        "slowking_distill_search_backend": diag.search_backend,
                    },
                }
            )
        return selected

    def _poke_rlm_shadow_after(
        self, obs_dict: dict, selected: list[int]
    ) -> None:
        """Shadow telemetry after another policy chose ``selected``."""
        cfg = self.poke_rlm_config
        if cfg is None or not cfg.shadow_only or self.model is None:
            return
        if self._poke_rlm_bridge is None:
            self._init_poke_rlm_bridge()
        if self._poke_rlm_bridge is None:
            return
        board = (
            self.board_history[-1]
            if self.board_history
            else features.build_board_tokens(obs_dict, self.deck)
        )
        try:
            trace = self._poke_rlm_bridge.shadow(
                obs_dict,
                board=board,
                selected_action=selected,
                append_cache=False,
            )
            self.last_poke_rlm_trace = trace.to_json()
        except Exception as exc:
            self.last_poke_rlm_trace = {
                "enabled": True,
                "mode": cfg.mode.value,
                "reason": "encode_failure",
                "error": f"{type(exc).__name__}: {exc}",
                "selected_action": list(selected),
                "used_for_selection": False,
            }

    @torch.no_grad()
    def poke_rlm_select(self, obs_dict: dict) -> list[int]:
        """PokeRLM evaluate/active select with greedy fallback."""
        go_first = forced_go_first_action(obs_dict)
        if go_first is not None:
            return self._record_go_first(obs_dict, go_first)
        if self._poke_rlm_bridge is None:
            if self.model is not None and self.poke_rlm_config is not None:
                self._init_poke_rlm_bridge()
            if self._poke_rlm_bridge is None:
                return self.greedy_select(obs_dict)

        board = self._append_decision_history(obs_dict)

        def greedy_fallback(_obs: dict) -> list[int]:
            return self._factorized_greedy_prepared(
                obs_dict,
                board,
                target_source="poke_rlm_greedy_fallback",
                extra_diagnostics={
                    "poke_rlm_fallback_reason": (
                        self._poke_rlm_bridge.last_diagnostics.fallback_reason
                        if self._poke_rlm_bridge is not None
                        else "bridge_missing"
                    )
                },
            )

        selected = self._poke_rlm_bridge.select(
            obs_dict,
            board=board,
            greedy_fallback=greedy_fallback,
        )
        diag = self._poke_rlm_bridge.last_diagnostics
        if diag.trace is not None:
            self.last_poke_rlm_trace = diag.trace.to_json()
        if diag.used_for_selection:
            self._previous_action_token = features.build_option_tokens(
                obs_dict, [selected]
            )
            if self.collect_targets:
                self.targets.append(
                    {
                        "observation": obs_dict,
                        "action": list(selected),
                        "action_combos": [list(selected)],
                        "policy": [1.0],
                        "value": None,
                        "visits": None,
                        "diagnostics": {
                            "target_source": "poke_rlm",
                            "trusted": False,
                            "history_length": len(self.board_history),
                            "poke_rlm_mode": diag.mode,
                            "poke_rlm_route": diag.route,
                            "poke_rlm_reason": diag.reason,
                            "poke_rlm_legal_count": diag.legal_count,
                        },
                    }
                )
        return selected

    def mcts_select(self, obs_dict: dict) -> list[int]:
        self._require_own_deck_direct_policy("MCTS")
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
        self._require_own_deck_direct_policy("belief MCTS")
        features.assert_info_set(obs_dict)
        ledger_snapshot = self._observe_own_deck_ledger(obs_dict)
        board = features.build_board_tokens(obs_dict, self.deck)
        self._append_history_snapshot(board, ledger_snapshot)
        max_context = self._history_context_limit()
        max_sims = int(self.max_sims or max(128, config.SEARCH.sims_per_move))
        min_trusted_sims = int(
            self.min_trusted_sims
            if self.min_trusted_sims is not None
            else min(128, max_sims)
        )
        configured_move_time = float(
            self.move_time_s
            if self.move_time_s is not None
            else config.SEARCH.move_time_budget_s
        )
        move_time = configured_move_time
        if self.clock is not None:
            move_time = self.clock.next_move_budget(move_time)
        engine: Optional[BeliefMCTS]
        if int(self.belief_mcts_lanes) == 8:
            from .eight_lane_belief_forest import EightLaneBeliefForest

            engine = None
            if self._belief_forest is None:
                self._belief_forest = EightLaneBeliefForest(
                    self.model,
                    self.deck,
                    self.belief_posterior,
                    checkpoint_digest=str(self.checkpoint_digest),
                    model_generation=int(self.model_generation),
                    device=self.device,
                    min_trusted_sims=min_trusted_sims,
                    particle_count=16,
                    max_context=max_context,
                    rng=self.rng,
                )
            result = self._belief_forest.search(
                obs_dict,
                belief_history=self.belief_history,
                root_history_boards=self.board_history,
                root_history_previous_actions=self.previous_action_history,
                matchup_shadow_router=(
                    self._matchup_adapter_shadow_router.fork()
                    if self.matchup_adapter_shadow
                    else None
                ),
                matchup_model_route=self._matchup_model_route(),
                clock=self.clock,
                max_sims=max_sims,
                move_time_s=move_time,
            )
            # Workers receive private snapshots.  Commit the real observation
            # to the live public history exactly once after all eight lanes and
            # their deterministic reduction have succeeded.
            self.belief_history.observe(obs_dict)
        else:
            engine = BeliefMCTS(
                self.model,
                self.deck,
                self.belief_posterior,
                checkpoint_digest=str(self.checkpoint_digest),
                model_generation=int(self.model_generation),
                device=self.device,
                leaf_backend=self.leaf_backend,
                rng=self.rng,
                min_trusted_sims=min_trusted_sims,
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
            provenance = (
                engine.target_provenance(
                    max_sims=max_sims, move_time_s=configured_move_time
                )
                if engine is not None
                else {
                    "schema": "poke_bot.eight_lane_belief_forest_target/v1",
                    "checkpoint_digest": str(self.checkpoint_digest),
                    "model_generation": int(self.model_generation),
                    "search_config": {
                        "requested_lanes": 8,
                        "selected_lanes": 8,
                        "automatic_lane_reduction_allowed": False,
                        "max_sims_per_lane": max_sims,
                        "move_time_s": configured_move_time,
                    },
                }
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
        # A deck request is not a decision.  Clear any prior decision audit
        # before returning so callers cannot mistake it for this turn's trace.
        self.last_turn_checklist_logit_trace = None
        if obs_dict is None or obs_dict.get("select") is None:
            return list(self.deck)
        # Avoid exposing stale RTP diagnostics after a no-RTP selection.  The
        # evaluator samples immediately after each candidate action and treats
        # a missing record as evidence that the bridge was not used.
        self.last_rtp_diagnostics = None
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
            poke_cfg = self.poke_rlm_config
            distill_cfg = self.slowking_distill_config
            selection_path = "factorized_greedy"
            if self.use_mcts:
                selection_path = "mcts"
                action = self.mcts_select(obs_dict)
            elif (
                poke_cfg is not None
                and poke_cfg.selects_actions
                and (self._poke_rlm_bridge is not None or self.model is not None)
            ):
                selection_path = "poke_rlm"
                action = self.poke_rlm_select(obs_dict)
            elif (
                distill_cfg is not None
                and distill_cfg.selects_actions
                and _runtime_is_slowking_specialist()
                and (
                    self._slowking_distill_bridge is not None
                    or bool(distill_cfg.actor_checkpoint)
                )
            ):
                selection_path = "slowking_distill"
                action = self.slowking_distill_select(obs_dict)
            elif self.use_recursive_turn_planner and (
                self._rtp_bridge is not None or self.model is not None
            ):
                selection_path = "recursive_turn_planner"
                action = self.rtp_select(obs_dict)
            else:
                action = self.greedy_select(obs_dict)
            if (
                poke_cfg is not None
                and poke_cfg.shadow_only
                and not poke_cfg.selects_actions
            ):
                self._poke_rlm_shadow_after(obs_dict, action)
            if (
                distill_cfg is not None
                and distill_cfg.shadow_only
                and not distill_cfg.selects_actions
            ):
                self._slowking_distill_shadow_after(obs_dict, action)
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
                fallback = (
                    "raising for transactional frozen-greedy fallback"
                    if self.strict_runtime
                    or getattr(self, "own_deck_ledger_enabled", False)
                    else "using random legal select"
                )
                print(
                    f"[PolicyAgent] FAIL-CLOSED (no search/leaf): "
                    f"{type(exc).__name__}: {exc} — {fallback}. "
                    f"use_mcts={self.use_mcts} leaf_backend="
                    f"{type(self.leaf_backend).__name__ if self.leaf_backend else None}",
                    file=sys.stderr,
                    flush=True,
                )
            if self.strict_runtime or getattr(self, "own_deck_ledger_enabled", False):
                raise RuntimeError(
                    f"policy runtime failed closed: {type(exc).__name__}: {exc}"
                ) from exc
            selection_path = "runtime_failure"
            # A partial factorized audit cannot describe the fallback action.
            # Clear it so the explicit no-evaluation trace below is tied to
            # the action that is actually returned.
            self.last_turn_checklist_logit_trace = None
            action = []
        final_action = _fail_closed_legal(obs_dict, action, self.rng)
        if (
            bool(getattr(self, "turn_checklist_logit_layer_enabled", False))
            and self.last_turn_checklist_logit_trace is None
        ):
            # Do not manufacture a score-space trace for policy paths that did
            # not call the residual.  The forced parser is queried only to
            # improve the audit reason; an unexpected parser failure cannot
            # alter an already chosen legal action.
            try:
                forced = forced_go_first_action(obs_dict)
            except Exception:
                forced = None
            if forced is not None:
                self._record_turn_checklist_not_evaluated(
                    selected_action=final_action,
                    reason="forced_go_first_contract",
                    candidate_rows=[list(forced)],
                    factorized_prefix=[],
                    factorized_stage_index=0,
                    selected_candidate_index=(
                        0 if list(final_action) == list(forced) else None
                    ),
                )
            else:
                reason = (
                    "factorized_greedy_no_scored_stage"
                    if selection_path == "factorized_greedy"
                    else f"{selection_path}_selection_did_not_evaluate_checklist"
                )
                self._record_turn_checklist_not_evaluated(
                    selected_action=final_action,
                    reason=reason,
                )
        return final_action


AgentFn = Callable[[dict], list[int]]


def play_game(
    agent0: AgentFn,
    agent1: AgentFn,
    deck0: list[int],
    deck1: list[int],
    *,
    max_steps: int = 4000,
    max_no_progress_turns: int | None = None,
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
    stalled = False
    stall_turns = 0
    stall_tracker = None
    if max_no_progress_turns is not None:
        from .pure_rl.no_progress import NoProgressTracker

        stall_tracker = NoProgressTracker(int(max_no_progress_turns))
    try:
        while obs is not None and not cg_env.is_finished(obs) and steps < max_steps:
            if stall_tracker is not None and stall_tracker.observe(obs):
                stalled = True
                stall_turns = stall_tracker.stagnant_turns
                break
            cur = obs.get("current") or {}
            seat = int(cur.get("yourIndex", 0))
            agent = agent0 if seat == 0 else agent1
            try:
                target_rows = getattr(agent, "targets", None)
                target_count_before = (
                    len(target_rows) if isinstance(target_rows, list) else None
                )
                select = agent(obs)
                next_obs = cg_env.battle_select(select)
                if (
                    target_count_before is not None
                    and isinstance(target_rows, list)
                    and len(target_rows) == target_count_before + 1
                ):
                    # Training-only transition evidence.  The strategic target
                    # builder consumes this exact post-action public snapshot
                    # before records are serialized; it is never a policy
                    # feature and is deleted after target materialization.
                    from .strategic_heads import public_transition_snapshot

                    target_rows[-1]["transition_after"] = (
                        public_transition_snapshot(next_obs, actor_seat=seat)
                    )
                obs = next_obs
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
            "stall_terminated": stalled,
            "stall_turns": stall_turns,
            "termination": (
                "agent_failure"
                if failed_seat is not None
                else "no_progress_stall"
                if stalled
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
