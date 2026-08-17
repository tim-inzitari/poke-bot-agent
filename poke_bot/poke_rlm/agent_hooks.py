"""PolicyAgent integration for PokeRLM (shadow / evaluate / active)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch
from torch import Tensor

from poke_bot import features
from poke_bot.features import ActionSpaceTooLarge
from poke_bot.model import TemporalCabtTransformer

from .config import PokeRLMConfig, PokeRLMProfile, config_for_profile
from .controller import ControllerResult, PokeRLMController
from .telemetry import ShadowTrace


GreedyFn = Callable[[dict], list[int]]


def resolve_poke_rlm_config_for_model(
    model: Optional[TemporalCabtTransformer],
    *,
    base: Optional[PokeRLMConfig] = None,
) -> PokeRLMConfig:
    """Bind planner width to the attached encoder when possible."""
    cfg = base or PokeRLMConfig()
    if model is None:
        return cfg
    d_model = int(getattr(model, "d_model", cfg.d_model))
    if d_model == 96:
        profile = PokeRLMProfile.PURE_RL_96
    elif d_model == 256:
        profile = PokeRLMProfile.GLOBAL_256
    else:
        return PokeRLMConfig(
            enabled=cfg.enabled,
            mode=cfg.mode,
            profile=cfg.profile,
            d_model=d_model,
            successor_dim=max(32, d_model // 2),
            root_plan_candidates=cfg.root_plan_candidates,
            max_depth=cfg.max_depth,
            max_neural_planner_calls_per_turn=cfg.max_neural_planner_calls_per_turn,
            max_simulator_calls_per_turn=cfg.max_simulator_calls_per_turn,
        )
    bound = config_for_profile(
        profile,
        enabled=cfg.enabled,
        mode=cfg.mode,
        max_depth=cfg.max_depth,
        max_neural_planner_calls_per_turn=cfg.max_neural_planner_calls_per_turn,
        max_simulator_calls_per_turn=cfg.max_simulator_calls_per_turn,
    )
    return bound


@dataclass
class PokeRLMBridgeDiagnostics:
    mode: str = ""
    route: str = ""
    reason: str = ""
    used_for_selection: bool = False
    legal_count: int = 0
    fallback_reason: str = ""
    trace: Optional[ShadowTrace] = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class PokeRLMAgentBridge:
    """Encode-once bridge from PolicyAgent history into PokeRLMController."""

    model: TemporalCabtTransformer
    deck: list[int]
    config: PokeRLMConfig
    get_matchup_route: Callable[[], int]
    get_board_history: Callable[[], list[features.SparseVector]]
    get_previous_action_history: Callable[
        [], list[Optional[features.SparseVector]]
    ]
    get_previous_action_token: Callable[[], Optional[features.SparseVector]]
    get_kv_cache: Callable[[], Any]
    set_kv_cache: Callable[[Any], None]
    max_action_combos: int = 256
    controller: Optional[PokeRLMController] = None
    last_diagnostics: PokeRLMBridgeDiagnostics = field(
        default_factory=PokeRLMBridgeDiagnostics
    )
    last_result: Optional[ControllerResult] = None

    def __post_init__(self) -> None:
        if int(self.model.d_model) != int(self.config.d_model):
            raise ValueError(
                f"PokeRLM d_model={self.config.d_model} does not match model "
                f"d_model={self.model.d_model}"
            )
        if self.controller is None:
            self.controller = PokeRLMController(self.config)

    def reset_game(self) -> None:
        if self.controller is not None:
            self.controller.reset_game()
        self.last_diagnostics = PokeRLMBridgeDiagnostics()
        self.last_result = None

    def _legal_combos(self, obs_dict: dict) -> list[list[int]]:
        try:
            combos = features.enumerate_action_combos(
                obs_dict, max_combos=self.max_action_combos
            )
            return [list(c) for c in combos]
        except ActionSpaceTooLarge:
            # Incomplete factorized prefixes are not complete legal actions.
            # Let the caller fall back to factorized greedy / non-RLM select.
            raise

    @torch.no_grad()
    def encode(
        self,
        obs_dict: dict,
        *,
        board: features.SparseVector,
        legal_actions: Sequence[Sequence[int]],
        append_cache: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Return (state_vec [D], option_hidden [N,D]) for the legal set."""
        if not legal_actions:
            raise ValueError("PokeRLM encode requires legal actions")
        options = features.build_option_tokens(
            obs_dict, [list(a) for a in legal_actions]
        )
        matchup_route = int(self.get_matchup_route())
        model = self.model
        n_options = [len(legal_actions)]
        if model.decision_context == "history" and model.kv_cache_enabled:
            model_out = model.forward(
                board,
                options,
                kv_cache=self.get_kv_cache(),
                append_cache=bool(append_cache),
                n_options=n_options,
                previous_action=self.get_previous_action_token(),
                matchup_routes=[matchup_route],
            )
            if append_cache:
                self.set_kv_cache(model_out["kv_cache"])
        elif model.decision_context == "history":
            model_out = model.forward_history_batch(
                [list(self.get_board_history())],
                [options],
                n_options=n_options,
                previous_action_histories=[list(self.get_previous_action_history())],
                matchup_routes=[matchup_route],
            )
        else:
            model_out = model.forward(
                board,
                options,
                append_cache=False,
                n_options=n_options,
                matchup_routes=[matchup_route],
            )
        state_vec = model.matchup_policy_value_state(
            model_out["state_vec"], [matchup_route]
        )
        spatial = model_out["spatial_memory"]
        decoded = model.decode_options(
            options,
            spatial,
            state_vec,
            n_options=n_options,
            return_hidden=True,
            decision_fusion_state_vec=model_out["state_vec"],
        )
        assert isinstance(decoded, tuple)
        option_hidden = decoded[1][0, : len(legal_actions)]
        if state_vec.dim() == 2:
            state_vec = state_vec[0]
        return state_vec.detach(), option_hidden.detach()

    @torch.no_grad()
    def plan(
        self,
        obs_dict: dict,
        *,
        board: features.SparseVector,
        fallback_action: Optional[Sequence[int]] = None,
        append_cache: bool = True,
        observations: Optional[dict[str, Any]] = None,
    ) -> ControllerResult:
        assert self.controller is not None
        legal = self._legal_combos(obs_dict)
        diag = PokeRLMBridgeDiagnostics(
            mode=self.config.mode.value,
            legal_count=len(legal),
        )
        if not legal:
            result = self.controller.plan_decision(
                obs_dict,
                state_vec=torch.zeros(self.config.d_model),
                option_hidden=torch.zeros(1, self.config.d_model),
                legal_combos=[],
                deck=list(self.deck),
                matchup_route=int(self.get_matchup_route()),
                fallback_action=fallback_action or (),
                observations=observations,
            )
            diag.reason = result.reason.value
            diag.route = result.route
            diag.trace = result.trace
            diag.fallback_reason = "no_legal_actions"
            self.last_diagnostics = diag
            self.last_result = result
            return result

        state_vec, option_hidden = self.encode(
            obs_dict,
            board=board,
            legal_actions=legal,
            append_cache=append_cache,
        )
        # Project encoder hidden to planner width if needed (should match).
        if state_vec.size(-1) != self.config.d_model:
            raise ValueError("encoder/planner d_model mismatch after encode")
        result = self.controller.plan_decision(
            obs_dict,
            state_vec=state_vec,
            option_hidden=option_hidden,
            legal_combos=legal,
            deck=list(self.deck),
            matchup_route=int(self.get_matchup_route()),
            fallback_action=fallback_action,
            observations=observations,
        )
        diag.reason = result.reason.value
        diag.route = result.route
        diag.used_for_selection = result.used_for_selection
        diag.trace = result.trace
        self.last_diagnostics = diag
        self.last_result = result
        return result

    def select(
        self,
        obs_dict: dict,
        *,
        board: features.SparseVector,
        greedy_fallback: GreedyFn,
        append_cache: bool = True,
    ) -> list[int]:
        """Active/evaluate selection with greedy fallback."""
        fallback = greedy_fallback(obs_dict) if not self.config.selects_actions else None
        # For active mode, compute greedy only on failure to avoid double work when
        # possible; still need a concrete fallback list for illegal plans.
        try:
            if fallback is None:
                # Lazy fallback: plan first; if not used, call greedy.
                result = self.plan(
                    obs_dict,
                    board=board,
                    fallback_action=(),
                    append_cache=append_cache,
                )
                if result.used_for_selection and result.action:
                    return list(result.action)
                return greedy_fallback(obs_dict)
            result = self.plan(
                obs_dict,
                board=board,
                fallback_action=fallback,
                append_cache=append_cache,
            )
            if self.config.selects_actions and result.used_for_selection and result.action:
                return list(result.action)
            return list(fallback)
        except Exception as exc:
            self.last_diagnostics = PokeRLMBridgeDiagnostics(
                mode=self.config.mode.value,
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )
            return greedy_fallback(obs_dict)

    def shadow(
        self,
        obs_dict: dict,
        *,
        board: features.SparseVector,
        selected_action: Sequence[int],
        append_cache: bool = False,
    ) -> ShadowTrace:
        """Run planner for telemetry without changing the selected action."""
        result = self.plan(
            obs_dict,
            board=board,
            fallback_action=selected_action,
            append_cache=append_cache,
        )
        return result.trace
