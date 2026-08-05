"""Swap-in bridge from PolicyAgent observations to the Recursive Turn Planner.

Maps the existing encode / legal-action / history path onto:

    obs -> encode-once memory (H, option_hidden)
        -> plan_turn / PlanExecutor
        -> complete legal action

PolicyAgent owns history mutation; this bridge owns plan persistence for the
current ``(seat, turn)`` key and falls back to a provided greedy callback when
RTP cannot act.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch
from torch import Tensor

from poke_bot import cg_env, features
from poke_bot.features import ActionSpaceTooLarge
from poke_bot.model import TemporalCabtTransformer

from .config import RTPConfig
from .dynamics import LookaheadBackedDynamics
from .executor import PlanExecutor
from .memory import PersistentTurnMemory
from .planner import RecursiveTurnPlanner, TurnDecision
from .profiles import get_profile
from .types import TurnProgram


GreedyFn = Callable[[dict], list[int]]


def turn_key_from_obs(obs_dict: dict) -> tuple[int, int]:
    """Stable planning-episode key: (seat, turn)."""
    current = obs_dict.get("current") if isinstance(obs_dict, dict) else None
    if not isinstance(current, dict):
        obs = cg_env.to_observation(obs_dict)
        if obs.current is None:
            return (-1, -1)
        return (int(obs.current.yourIndex), int(getattr(obs.current, "turn", -1)))
    seat = int(current.get("yourIndex", -1))
    turn = int(current.get("turn", -1))
    return (seat, turn)


def resolve_rtp_config_for_model(
    model: Optional[TemporalCabtTransformer],
    *,
    profile_name: Optional[str] = None,
    online_sim_verify_budget: Optional[int] = None,
) -> RTPConfig:
    """Bind RTP widths to the attached model parent."""
    if profile_name:
        profile = get_profile(profile_name)
    elif model is not None and int(getattr(model, "d_model", 0)) == 96:
        profile = get_profile("pure_rl")
    elif model is not None and int(getattr(model, "d_model", 0)) == 256:
        profile = get_profile("global_transformer")
    elif model is not None:
        # Unknown width: still bind exactly to the live encoder.
        d_model = int(model.d_model)
        return RTPConfig(
            sizing_profile=f"model_d{d_model}",
            d_model=d_model,
            dynamics_width=max(32, 2 * d_model),
            prefer_option_hidden=True,
            online_sim_verify_budget=(
                0 if online_sim_verify_budget is None else int(online_sim_verify_budget)
            ),
        )
    else:
        profile = get_profile("global_transformer")
    overrides: dict[str, Any] = {}
    if online_sim_verify_budget is not None:
        overrides["online_sim_verify_budget"] = int(online_sim_verify_budget)
    return profile.to_config(**overrides)


@dataclass
class RTPBridgeDiagnostics:
    mode: str = ""
    turn_key: tuple[int, int] = (-1, -1)
    legal_count: int = 0
    used_option_hidden: bool = False
    fallback_reason: str = ""
    decision: Optional[TurnDecision] = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RTPAgentBridge:
    """Stateful RTP controller for one PolicyAgent instance."""

    model: TemporalCabtTransformer
    deck: list[int]
    config: RTPConfig
    get_matchup_route: Callable[[], int]
    get_board_history: Callable[[], list[features.SparseVector]]
    get_previous_action_history: Callable[
        [], list[Optional[features.SparseVector]]
    ]
    get_previous_action_token: Callable[[], Optional[features.SparseVector]]
    get_kv_cache: Callable[[], Any]
    set_kv_cache: Callable[[Any], None]
    max_action_combos: int = 256
    planner: Optional[RecursiveTurnPlanner] = None
    executor: Optional[PlanExecutor] = None
    memory: Optional[PersistentTurnMemory] = None
    active_turn_key: tuple[int, int] = (-1, -1)
    last_diagnostics: RTPBridgeDiagnostics = field(
        default_factory=RTPBridgeDiagnostics
    )

    def __post_init__(self) -> None:
        if self.planner is None:
            dynamics = None
            lookahead = getattr(self.model, "latent_lookahead", None)
            if (
                lookahead is not None
                and bool(getattr(self.model, "latent_lookahead_enabled", False))
                and int(getattr(lookahead, "d_model", -1)) == self.config.d_model
            ):
                dynamics = LookaheadBackedDynamics(
                    lookahead, d_model=self.config.d_model
                )
            self.planner = RecursiveTurnPlanner(
                self.config, dynamics=dynamics
            )
        if self.executor is None:
            self.executor = PlanExecutor(
                self.config,
                legality=self.planner.legality,
                repair_fn=self._repair_program,
            )
        if int(self.model.d_model) != int(self.config.d_model):
            raise ValueError(
                f"RTP d_model={self.config.d_model} does not match model "
                f"d_model={self.model.d_model}"
            )

    def reset_game(self) -> None:
        self.memory = None
        self.active_turn_key = (-1, -1)
        if self.executor is not None:
            self.executor.clear()
        self.last_diagnostics = RTPBridgeDiagnostics()

    def _repair_program(
        self,
        memory: PersistentTurnMemory,
        _program: TurnProgram,
    ) -> TurnProgram:
        assert self.planner is not None
        decision = self.planner.plan_turn(memory, force_recurse=True)
        if decision.program is None:
            # Minimal legal one-step program.
            action = memory.legal_actions[0] if memory.legal_actions else ()
            from .types import NodeKind, PlanNode

            return TurnProgram(
                root=PlanNode(kind=NodeKind.PRIMITIVE, action=action),
                plan_id="repair-direct",
            )
        return decision.program

    def _legal_actions(self, obs_dict: dict) -> tuple[tuple[int, ...], ...]:
        try:
            combos = features.enumerate_action_combos(
                obs_dict, max_combos=self.max_action_combos
            )
            return tuple(tuple(int(x) for x in combo) for combo in list(combos))
        except ActionSpaceTooLarge:
            # Decision-local fallback: first factorized stage candidates.
            stage = features.factorized_action_candidates(obs_dict, [])
            return tuple(tuple(int(x) for x in combo) for combo in stage)

    @torch.no_grad()
    def encode(
        self,
        obs_dict: dict,
        *,
        board: features.SparseVector,
        legal_actions: Sequence[Sequence[int]],
        append_cache: bool = True,
    ) -> tuple[PersistentTurnMemory, Tensor]:
        """Encode once for the current select; return memory + policy logits."""
        assert self.planner is not None
        if not legal_actions:
            raise ValueError("RTP encode requires legal actions")
        options = features.build_option_tokens(obs_dict, [list(a) for a in legal_actions])
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
            state_vec = model.matchup_policy_value_state(
                model_out["state_vec"], [matchup_route]
            )
            spatial = model_out["spatial_memory"]
            logits = model_out["policy_logits"][0, : len(legal_actions)]
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
        elif model.decision_context == "history":
            model_out = model.forward_history_batch(
                [list(self.get_board_history())],
                [options],
                n_options=n_options,
                previous_action_histories=[list(self.get_previous_action_history())],
                matchup_routes=[matchup_route],
            )
            state_vec = model.matchup_policy_value_state(
                model_out["state_vec"], [matchup_route]
            )
            spatial = model_out["spatial_memory"]
            logits = model_out["policy_logits"][0, : len(legal_actions)]
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
            logits = model_out["policy_logits"][0, : len(legal_actions)]
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
        value = 0.0
        if "value" in model_out:
            value_t = model_out["value"]
            value = float(value_t.reshape(-1)[0].item())
        memory = self.planner.encode_memory(
            state_vec,
            legal_actions=legal_actions,
            spatial_memory=spatial[0] if spatial.dim() == 3 else spatial,
            option_hidden=option_hidden,
            value_estimate=value,
            matchup_route=matchup_route,
        )
        return memory, logits.detach()

    def select(
        self,
        obs_dict: dict,
        *,
        board: features.SparseVector,
        greedy_fallback: GreedyFn,
    ) -> list[int]:
        """Plan or continue a conditional turn program; greedy on failure."""
        assert self.planner is not None
        assert self.executor is not None
        key = turn_key_from_obs(obs_dict)
        diag = RTPBridgeDiagnostics(turn_key=key)
        try:
            legal = self._legal_actions(obs_dict)
            diag.legal_count = len(legal)
            if not legal:
                diag.mode = "fallback"
                diag.fallback_reason = "no_legal_actions"
                self.last_diagnostics = diag
                return greedy_fallback(obs_dict)

            new_turn = key != self.active_turn_key
            if new_turn or self.memory is None or self.executor.active_program is None:
                memory, logits = self.encode(
                    obs_dict,
                    board=board,
                    legal_actions=legal,
                    append_cache=True,
                )
                self.memory = memory
                self.active_turn_key = key
                decision = self.planner.plan_turn(memory, policy_logits=logits)
                diag.decision = decision
                diag.used_option_hidden = memory.option_hidden is not None
                if decision.program is not None and not decision.used_direct_policy:
                    self.executor.load(decision.program)
                    action = decision.action
                    diag.mode = decision.mode
                else:
                    # Direct-policy gate inside RTP: still return its action.
                    action = decision.action
                    diag.mode = decision.mode or "direct_policy"
                    self.executor.clear()
            else:
                # Continue persisted plan for this turn.
                # Refresh option states without appending another KV step.
                memory, _logits = self.encode(
                    obs_dict,
                    board=board,
                    legal_actions=legal,
                    append_cache=False,
                )
                self.memory = memory
                step = self.executor.next_action(memory)
                diag.used_option_hidden = memory.option_hidden is not None
                if step.action is None:
                    decision = self.planner.plan_turn(memory, force_recurse=True)
                    diag.decision = decision
                    if decision.program is not None:
                        self.executor.load(decision.program)
                        action = decision.action
                        diag.mode = "replan"
                    else:
                        action = decision.action
                        diag.mode = "replan_direct"
                else:
                    action = step.action
                    diag.mode = "continue_plan"
                    diag.extras["repaired"] = step.repaired
                    if step.done:
                        self.executor.clear()

            if not action or action not in set(legal):
                diag.mode = "fallback"
                diag.fallback_reason = "planned_action_not_legal"
                self.last_diagnostics = diag
                self.executor.clear()
                return greedy_fallback(obs_dict)

            self.last_diagnostics = diag
            return list(action)
        except Exception as exc:
            diag.mode = "fallback"
            diag.fallback_reason = f"{type(exc).__name__}: {exc}"
            self.last_diagnostics = diag
            if self.executor is not None:
                self.executor.clear()
            return greedy_fallback(obs_dict)
