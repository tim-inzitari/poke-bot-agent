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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch import Tensor

from poke_bot import cg_env, features
from poke_bot.features import ActionSpaceTooLarge
from poke_bot.model import TemporalCabtTransformer

from .config import RTPConfig
from .dynamics import LookaheadBackedDynamics
from .executor import PlanExecutor, PlanStepResult
from .memory import PersistentTurnMemory
from .planner import (
    RTPNeuralPassBudgetExceeded,
    RecursiveTurnPlanner,
    TurnDecision,
)
from .profiles import PURE_RL_R197_MAX_ACTION_COMBOS, get_profile
from .types import TurnProgram


GreedyFn = Callable[[dict], list[int]]
_RTP_SERVING_QUALIFIED_ENV = "POKEBOT_RTP_SERVING_QUALIFIED"
_RTP_EXPECTED_PARENT_DIGEST_ENV = "POKEBOT_RTP_PARENT_CHECKPOINT_SHA256"
_RTP_PROMOTION_RECEIPT_ENV = "POKEBOT_RTP_PROMOTION_RECEIPT"
_RTP_EXPECTED_PROMOTION_RECEIPT_DIGEST_ENV = (
    "POKEBOT_RTP_PROMOTION_RECEIPT_SHA256"
)


class _ProgramFirstActionMismatch(RuntimeError):
    """A planner decision disagreed with its just-loaded executable program."""


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


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
    neural_passes: int = 0
    required_neural_passes: int = 0
    fallback_reason: str = ""
    fallback_code: str = ""
    fallback_detail: str = ""
    planner_config: dict[str, Any] = field(default_factory=dict)
    decision: Optional[TurnDecision] = None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a stable, serializable diagnostic record for telemetry."""

        return {
            "mode": self.mode,
            "turn_key": list(self.turn_key),
            "legal_count": self.legal_count,
            "used_option_hidden": self.used_option_hidden,
            "neural_passes": self.neural_passes,
            "required_neural_passes": self.required_neural_passes,
            "fallback_code": self.fallback_code,
            "fallback_reason": self.fallback_reason,
            "fallback_detail": self.fallback_detail,
            "planner_config": dict(self.planner_config),
            "decision_mode": self.decision.mode if self.decision is not None else "",
            "decision_diagnostics": (
                dict(self.decision.diagnostics) if self.decision is not None else {}
            ),
            "extras": dict(self.extras),
        }


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
    #: ``None`` preserves the historical direct-bridge cap for legacy profiles
    #: while selecting r197's checksum-bound complete-action cap after the
    #: planner configuration has been resolved.
    max_action_combos: Optional[int] = None
    #: Serving-qualified bridges require a checksum-bound parent and promoted
    #: sidecar. Ordinary research/shadow use remains explicitly opt-in.
    serving_qualified: bool = False
    expected_parent_digest: Optional[str] = None
    promotion_receipt: Optional[Path | str] = None
    expected_promotion_receipt_digest: Optional[str] = None
    #: Narrow sealed-evaluator exception for the shadow-only r197 sidecar.
    #: This is intentionally a structured context passed by the evaluator
    #: factory, never a generic environment opt-in or serving authority.
    evaluation_action_execution: Optional[Mapping[str, Any]] = None
    planner: Optional[RecursiveTurnPlanner] = None
    executor: Optional[PlanExecutor] = None
    memory: Optional[PersistentTurnMemory] = None
    active_turn_key: tuple[int, int] = (-1, -1)
    #: The common pre-forcing gate is scoped to a new turn only.  Retain its
    #: immutable diagnostic through continuation/replan selects so evaluation
    #: can distinguish inherited plan execution from a fresh gate decision.
    active_turn_complexity_intent: dict[str, Any] = field(default_factory=dict)
    last_diagnostics: RTPBridgeDiagnostics = field(
        default_factory=RTPBridgeDiagnostics
    )
    _r197_checkpoint_path: Optional[Path] = field(
        default=None, init=False, repr=False
    )
    _r197_serving_promotion_validated: bool = field(
        default=False, init=False, repr=False
    )
    _r197_evaluation_action_execution: Optional[dict[str, Any]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        runtime_config = self.config
        checkpoint_path: Optional[Path] = None
        if _env_enabled(_RTP_SERVING_QUALIFIED_ENV):
            self.serving_qualified = True
        if self.expected_parent_digest is None:
            expected = os.environ.get(_RTP_EXPECTED_PARENT_DIGEST_ENV, "").strip()
            self.expected_parent_digest = expected or None
        if self.promotion_receipt is None:
            receipt = os.environ.get(_RTP_PROMOTION_RECEIPT_ENV, "").strip()
            self.promotion_receipt = receipt or None
        if self.expected_promotion_receipt_digest is None:
            receipt_digest = os.environ.get(
                _RTP_EXPECTED_PROMOTION_RECEIPT_DIGEST_ENV, ""
            ).strip()
            self.expected_promotion_receipt_digest = receipt_digest or None
        if self.serving_qualified:
            missing: list[str] = []
            if not self.expected_parent_digest:
                missing.append("parent checkpoint digest")
            if self.promotion_receipt is None:
                missing.append("promotion receipt")
            if not self.expected_promotion_receipt_digest:
                missing.append("promotion receipt digest")
            if missing:
                raise ValueError(
                    "serving-qualified RTP bridge requires " + ", ".join(missing)
                )
        if self.planner is None:
            ckpt = os.environ.get("POKEBOT_RTP_CHECKPOINT", "").strip()
            if ckpt and Path(ckpt).is_file():
                checkpoint_path = Path(ckpt)
                # Trained sidecar owns dynamics weights; do not wrap parent lookahead.
                from .training.checkpoint import load_rtp_checkpoint

                self.planner = load_rtp_checkpoint(
                    ckpt,
                    device=next(self.model.parameters()).device,
                    expected_parent_digest=self.expected_parent_digest,
                    expected_config=self.config if self.serving_qualified else None,
                    promotion_receipt=self.promotion_receipt,
                    expected_promotion_receipt_digest=(
                        self.expected_promotion_receipt_digest
                    ),
                    serving_qualified=self.serving_qualified,
                )
                self.planner.eval()
                # A sidecar owns the planner configuration. The executor must
                # share that exact config rather than retain a stale bridge
                # profile (previously allowing mismatched repair budgets).
                self.config = self.planner.config
            else:
                if self.serving_qualified:
                    raise ValueError(
                        "serving-qualified RTP bridge requires a readable sidecar"
                    )
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
        elif self.serving_qualified:
            raise ValueError(
                "serving-qualified RTP bridge requires a checkpoint-loaded planner"
            )
        assert self.planner is not None
        if (
            runtime_config.sizing_profile == "pure_rl_r197"
            or self.planner.config.sizing_profile == "pure_rl_r197"
        ) and self.planner.config != runtime_config:
            raise ValueError(
                "pure_rl_r197 planner config must exactly match the runtime profile"
            )
        if self.max_action_combos is None:
            self.max_action_combos = (
                PURE_RL_R197_MAX_ACTION_COMBOS
                if self.planner.config.sizing_profile == "pure_rl_r197"
                else 256
            )
        else:
            self.max_action_combos = int(self.max_action_combos)
        if self.max_action_combos < 1:
            raise ValueError("max_action_combos must be positive")
        if (
            self.planner.config.sizing_profile == "pure_rl_r197"
            and self.max_action_combos != PURE_RL_R197_MAX_ACTION_COMBOS
        ):
            raise ValueError(
                "pure_rl_r197 requires max_action_combos="
                f"{PURE_RL_R197_MAX_ACTION_COMBOS}"
            )
        # Supplied research planners are also made coherent with their
        # executor; serving planners were checked against the runtime profile
        # by the qualified loader above.
        self.config = self.planner.config
        if self.executor is None:
            # r197 evaluator telemetry treats every forced replan as a
            # separately auditable five-pass planning decision.  Letting the
            # generic executor invoke ``_repair_program`` would return a
            # repaired primitive before ``select`` can record that decision.
            # Keep legacy sparse repair behavior unchanged, but route r197
            # invalid-program repair through the explicit bridge replan below.
            repair_fn = (
                None
                if self.planner.config.sizing_profile == "pure_rl_r197"
                else self._repair_program
            )
            self.executor = PlanExecutor(
                self.planner.config,
                legality=self.planner.legality,
                repair_fn=repair_fn,
            )
        elif self.executor.config != self.planner.config:
            raise ValueError("RTP executor config does not match planner config")
        if (
            self.planner.config.sizing_profile == "pure_rl_r197"
            and self.executor.repair_fn is not None
        ):
            raise ValueError(
                "pure_rl_r197 must not use an executor repair_fn; "
                "forced replans require bridge-visible diagnostics"
            )
        if int(self.model.d_model) != int(self.config.d_model):
            raise ValueError(
                f"RTP d_model={self.config.d_model} does not match model "
                f"d_model={self.model.d_model}"
            )
        if self.config.sizing_profile == "pure_rl_r197":
            self._r197_checkpoint_path = checkpoint_path
            # ``load_rtp_checkpoint(... serving_qualified=True)`` above is
            # the only path that can set this marker.  A caller cannot grant
            # serving action authority merely by constructing a bridge with a
            # public boolean or an injected planner.
            self._r197_serving_promotion_validated = bool(
                self.serving_qualified and checkpoint_path is not None
            )
            if self.evaluation_action_execution is not None:
                # Validate/copy at construction, then validate again before
                # every select.  This rejects a forged or mutable evaluator
                # context before any policy state is used.
                from .r197_action_authority import validate_evaluation_action_execution

                self._r197_evaluation_action_execution = (
                    validate_evaluation_action_execution(
                        self.evaluation_action_execution,
                        config=self.config,
                        max_action_combos=self.max_action_combos,
                        expected_parent_digest=self.expected_parent_digest,
                        checkpoint_path=self._r197_checkpoint_path,
                    )
                )

    def _require_r197_action_selection_authority(self) -> dict[str, Any] | None:
        """Refuse r197 action use outside promoted serving or sealed eval.

        Construction, replay inspection, and the evaluator's non-selecting
        complexity instrumentation intentionally remain possible without this
        permission.  The gate lives immediately before ``select`` so an inert
        shadow sidecar can never affect an ordinary policy action.
        """

        if self.config.sizing_profile != "pure_rl_r197":
            return None
        from .r197_action_authority import assert_r197_action_selection_authorized

        return assert_r197_action_selection_authorized(
            serving_qualified=self.serving_qualified,
            serving_promotion_validated=self._r197_serving_promotion_validated,
            evaluation_action_execution=self._r197_evaluation_action_execution,
            config=self.config,
            max_action_combos=self.max_action_combos,
            expected_parent_digest=self.expected_parent_digest,
            checkpoint_path=self._r197_checkpoint_path,
        )

    def _planner_config_diagnostics(self) -> dict[str, Any]:
        assert self.planner is not None
        cfg = self.planner.config
        return {
            "sizing_profile": cfg.sizing_profile,
            "d_model": cfg.d_model,
            "num_plan_candidates": cfg.num_plan_candidates,
            "max_recursion_depth": cfg.max_recursion_depth,
            "max_neural_passes": cfg.max_neural_passes,
            "max_action_combos": self.max_action_combos,
        }

    def _new_diagnostics(self, key: tuple[int, int]) -> RTPBridgeDiagnostics:
        assert self.planner is not None
        return RTPBridgeDiagnostics(
            turn_key=key,
            required_neural_passes=self.planner.required_recursive_passes(),
            planner_config=self._planner_config_diagnostics(),
        )

    def _record_decision(
        self,
        diag: RTPBridgeDiagnostics,
        decision: TurnDecision,
    ) -> None:
        diag.decision = decision
        diag.neural_passes = int(decision.neural_passes)

    def _record_fallback(
        self,
        diag: RTPBridgeDiagnostics,
        *,
        code: str,
        detail: str = "",
    ) -> None:
        assert self.planner is not None
        diag.mode = "fallback"
        diag.fallback_code = str(code)
        diag.fallback_detail = str(detail)
        diag.fallback_reason = str(code) if not detail else f"{code}: {detail}"
        diag.neural_passes = self.planner.neural_passes

    def _load_and_consume_program_first_action(
        self,
        *,
        memory: PersistentTurnMemory,
        decision: TurnDecision,
        diag: RTPBridgeDiagnostics,
        phase: str,
    ) -> PlanStepResult:
        """Load a recursive program and advance its cursor exactly once.

        ``TurnDecision.action`` is the action returned for the current atomic
        select, so merely loading its program leaves the executor cursor at
        that same root action.  Consume the executor's first step now, rather
        than replaying the root on the next same-turn select.  Going through
        :class:`PlanExecutor` intentionally retains its normal legality,
        repair, and terminal-step handling.
        """

        assert self.executor is not None
        if decision.program is None:
            raise ValueError("recursive decision has no program to load")
        if (
            self.config.sizing_profile == "pure_rl_r197"
            and self.executor.repair_fn is not None
        ):
            # Check every program-load boundary, not only continuations. A
            # post-construction callback mutation could otherwise run a
            # hidden forced replan before the bridge records its five-pass
            # decision telemetry.
            self.executor.clear()
            raise _ProgramFirstActionMismatch(
                f"{phase} pure_rl_r197 executor has an untelemetrized "
                "repair callback"
            )
        self.executor.load(decision.program)
        step = self.executor.next_action(memory)
        actual_action = step.action
        expected_action = decision.action
        diag.extras["loaded_program_first_step"] = {
            "phase": str(phase),
            "expected_action": list(expected_action),
            "executor_action": (
                None if actual_action is None else list(actual_action)
            ),
            "done": bool(step.done),
            "repaired": bool(step.repaired),
            "reason": str(step.reason),
        }
        if (
            self.config.sizing_profile == "pure_rl_r197"
            and self.planner.neural_passes != int(decision.neural_passes)
        ):
            self.executor.clear()
            raise _ProgramFirstActionMismatch(
                f"{phase} executor changed planner passes from "
                f"{decision.neural_passes} to {self.planner.neural_passes}"
            )
        # A repair here would have planned a replacement program from inside
        # ``PlanExecutor``.  Its forced-replan passes cannot be represented by
        # this select's original ``TurnDecision``, so accepting it would hide
        # compute/repair telemetry.  Preserve the executor's legality result,
        # but fail closed rather than returning an unaccounted repair action.
        if step.repaired or actual_action != expected_action:
            self.executor.clear()
            reason = "executor repaired the loaded program" if step.repaired else (
                f"executor action {actual_action!r} does not match "
                f"planner decision {expected_action!r}"
            )
            raise _ProgramFirstActionMismatch(
                f"{phase} {reason}"
            )
        return step

    def reset_game(self) -> None:
        self.memory = None
        self.active_turn_key = (-1, -1)
        self.active_turn_complexity_intent = {}
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
            # Never treat incomplete factorized prefixes (e.g. ``[0]`` when
            # bounds require ``[4, 4]``) as complete legal actions — that made
            # RTP submit illegal moves and fail-closed collect. Re-raise so
            # ``select()`` falls back to factorized greedy, which builds a
            # complete ordered action autoregressively.
            raise

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
        force_direct_bridge_only: bool = False,
    ) -> list[int]:
        """Plan or continue a conditional turn program; greedy on failure.

        ``force_direct_bridge_only`` is deliberately a call-scoped evaluation
        control.  It still uses the exact bridge encode + complete ordered
        action enumeration, but tells the planner to take the direct-policy
        branch.  It is not a serving switch and avoids mutating the sidecar or
        planner configuration while a three-arm study compares the bridge
        against recursive planning.
        """
        assert self.planner is not None
        assert self.executor is not None
        # This is deliberately outside the broad fallback handler below.  An
        # unpromoted r197 sidecar must not be allowed to choose an action and
        # then be disguised as an ordinary planner fallback.
        r197_action_execution = self._require_r197_action_selection_authority()
        key = turn_key_from_obs(obs_dict)
        diag = self._new_diagnostics(key)
        if r197_action_execution is not None:
            diag.extras["r197_action_execution"] = dict(r197_action_execution)
        # Diagnostics are per-select, not a stale counter from a prior turn
        # whose plan is merely being executed now.
        self.planner.reset_pass_counter()
        try:
            legal = self._legal_actions(obs_dict)
            diag.legal_count = len(legal)
            if not legal:
                self._record_fallback(diag, code="no_legal_actions")
                self.last_diagnostics = diag
                self.executor.clear()
                self.active_turn_complexity_intent = {}
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
                if force_direct_bridge_only:
                    # The direct comparator must report whether the *same*
                    # bridge inputs would have crossed the recursive
                    # complexity gate.  Run that pre-forcing trace first,
                    # then reset its accounting so it cannot spend planner
                    # budget or alter the direct action path.
                    would_recurse, intent = self.planner.should_recurse(
                        memory,
                        policy_logits=logits,
                    )
                    diag.extras["pre_forcing_complexity_intent"] = {
                        "would_recurse": bool(would_recurse),
                        "gate": dict(intent),
                        "trace_neural_passes": self.planner.neural_passes,
                        "new_turn": bool(new_turn),
                    }
                    self.planner.reset_pass_counter()
                decision = self.planner.plan_turn(
                    memory,
                    policy_logits=logits,
                    force_recurse=False if force_direct_bridge_only else None,
                )
                self._record_decision(diag, decision)
                diag.used_option_hidden = memory.option_hidden is not None
                if force_direct_bridge_only:
                    diag.extras["force_direct_bridge_only"] = True
                else:
                    complexity_gate = decision.diagnostics.get("complexity_gate")
                    if isinstance(complexity_gate, dict):
                        diag.extras["pre_forcing_complexity_intent"] = {
                            "would_recurse": bool(
                                decision.mode
                                in {"recursive_plan", "direct_policy_fallback"}
                            ),
                            "gate": dict(complexity_gate),
                            "trace_neural_passes": int(decision.neural_passes),
                            "new_turn": bool(new_turn),
                        }
                current_intent = diag.extras.get("pre_forcing_complexity_intent")
                self.active_turn_complexity_intent = (
                    dict(current_intent)
                    if isinstance(current_intent, dict)
                    else {}
                )
                if decision.program is not None and not decision.used_direct_policy:
                    step = self._load_and_consume_program_first_action(
                        memory=memory,
                        decision=decision,
                        diag=diag,
                        phase="initial",
                    )
                    assert step.action is not None
                    action = step.action
                    diag.mode = decision.mode
                    if step.done:
                        self.executor.clear()
                        self.active_turn_complexity_intent = {}
                else:
                    # Direct-policy gate inside RTP: still return its action.
                    action = decision.action
                    diag.mode = (
                        "direct_bridge"
                        if force_direct_bridge_only
                        else (decision.mode or "direct_policy")
                    )
                    self.executor.clear()
                    self.active_turn_complexity_intent = {}
            else:
                if force_direct_bridge_only:
                    # Direct bridge never persists a program.  This branch is
                    # defensive for a caller that toggled the evaluation arm
                    # mid-turn; clear state and re-enter the direct path on
                    # the next select rather than allowing a recursive plan to
                    # leak into the direct-only comparator.
                    self.executor.clear()
                    self.memory = None
                    self.active_turn_key = (-1, -1)
                    return self.select(
                        obs_dict,
                        board=board,
                        greedy_fallback=greedy_fallback,
                        force_direct_bridge_only=True,
                    )
                # Continue persisted plan for this turn.
                if self.active_turn_complexity_intent:
                    diag.extras["pre_forcing_complexity_intent"] = {
                        **self.active_turn_complexity_intent,
                        "inherited": True,
                        "new_turn": False,
                    }
                # Refresh option states without appending another KV step.
                memory, _logits = self.encode(
                    obs_dict,
                    board=board,
                    legal_actions=legal,
                    append_cache=False,
                )
                self.memory = memory
                if (
                    self.config.sizing_profile == "pure_rl_r197"
                    and self.executor.repair_fn is not None
                ):
                    # Check before invocation as well as after it below: an
                    # externally mutated callback could otherwise run an
                    # arbitrary number of planner calls before detection.
                    raise RuntimeError(
                        "pure_rl_r197 executor has an untelemetrized repair callback"
                    )
                step = self.executor.next_action(memory)
                diag.used_option_hidden = memory.option_hidden is not None
                if self.config.sizing_profile == "pure_rl_r197" and (
                    step.repaired or self.planner.neural_passes != 0
                ):
                    # The r197 executor is deliberately repair-free.  Guard
                    # against post-construction mutation or an injected
                    # executor silently spending a planner pass before the
                    # bridge can bind it to a forced-replan diagnostic.
                    raise RuntimeError(
                        "pure_rl_r197 executor performed an untelemetrized repair"
                    )
                if step.action is None:
                    diag.required_neural_passes = (
                        self.planner.required_recursive_passes(force_recurse=True)
                    )
                    decision = self.planner.plan_turn(memory, force_recurse=True)
                    self._record_decision(diag, decision)
                    if decision.program is not None:
                        step = self._load_and_consume_program_first_action(
                            memory=memory,
                            decision=decision,
                            diag=diag,
                            phase="replan",
                        )
                        assert step.action is not None
                        action = step.action
                        diag.mode = "replan_with_program"
                        if step.done:
                            self.executor.clear()
                            self.active_turn_complexity_intent = {}
                    else:
                        action = decision.action
                        diag.mode = "replan_direct"
                        self.active_turn_complexity_intent = {}
                else:
                    action = step.action
                    diag.mode = "continue_plan"
                    diag.extras["repaired"] = step.repaired
                    if step.done:
                        self.executor.clear()
                        self.active_turn_complexity_intent = {}

            # An empty tuple is a valid zero-selection action for prompts
            # whose exact legal cardinality admits ``minCount == 0``.  Only a
            # missing action or one outside the complete legal set may fall
            # back here.
            if action is None or action not in set(legal):
                self._record_fallback(diag, code="planned_action_not_legal")
                self.last_diagnostics = diag
                self.executor.clear()
                self.active_turn_complexity_intent = {}
                return greedy_fallback(obs_dict)

            self.last_diagnostics = diag
            return list(action)
        except _ProgramFirstActionMismatch as exc:
            # This is a planner/program integrity failure, never an expected
            # recursive fallback.  The evaluator keeps it in its unexpected
            # fallback budget rather than silently crediting a replayed root.
            self._record_fallback(
                diag,
                code="executor_first_action_mismatch",
                detail=str(exc),
            )
            self.last_diagnostics = diag
            self.executor.clear()
            self.active_turn_complexity_intent = {}
            return greedy_fallback(obs_dict)
        except ActionSpaceTooLarge as exc:
            # r198's 1,024 cap is a complete-ordered-action materialization
            # boundary, not a license to truncate legal support.  The caller
            # uses the existing factorized greedy fallback, while its isolated
            # evaluator records this as planner-ineligible rather than as a
            # recursive/direct/fallback credit.  Capture the exact closed-form
            # summary here so B/C cannot claim the special stratum after some
            # hidden planner work already occurred.
            is_r198_over_cap_path = (
                self.config.sizing_profile == "pure_rl_r197"
                and int(self.max_action_combos or 0)
                == PURE_RL_R197_MAX_ACTION_COMBOS
            )
            if not is_r198_over_cap_path:
                # Preserve the legacy bridge's generic fallback diagnostics.
                # A coincidental ActionSpaceTooLarge under another profile or
                # cap is not evidence for r198's isolated audit stratum.
                self._record_fallback(
                    diag,
                    code="action_space_too_large",
                    detail=str(exc),
                )
                self.last_diagnostics = diag
                self.executor.clear()
                self.active_turn_complexity_intent = {}
                return greedy_fallback(obs_dict)
            action_space = features.complete_ordered_action_space_summary(
                obs_dict,
                max_combos=int(self.max_action_combos or features.MAX_ACTION_COMBOS),
            )
            if action_space.get("over_cap") is not True:
                raise RuntimeError(
                    "ActionSpaceTooLarge disagrees with the strict action-space summary"
                ) from exc
            # `_new_diagnostics` normally records the six-pass recursive
            # requirement.  No complete legal list reached a planner here, so
            # both observed and required passes are truthfully zero.
            diag.required_neural_passes = 0
            diag.extras["over_cap_factorized_fallback"] = {
                "classification": "complete_ordered_action_space_over_cap",
                "action_space": action_space,
                "factorized_greedy_fallback": True,
            }
            self._record_fallback(
                diag,
                code="action_space_too_large",
                detail=str(exc),
            )
            self.last_diagnostics = diag
            self.executor.clear()
            self.active_turn_complexity_intent = {}
            return greedy_fallback(obs_dict)
        except RTPNeuralPassBudgetExceeded as exc:
            diag.extras.update({"used": exc.used, "limit": exc.limit})
            self._record_fallback(
                diag,
                code="neural_pass_budget_exceeded",
                detail=str(exc),
            )
            self.last_diagnostics = diag
            self.executor.clear()
            self.active_turn_complexity_intent = {}
            return greedy_fallback(obs_dict)
        except Exception as exc:
            self._record_fallback(
                diag,
                code="runtime_exception",
                detail=f"{type(exc).__name__}: {exc}",
            )
            self.last_diagnostics = diag
            self.executor.clear()
            self.active_turn_complexity_intent = {}
            return greedy_fallback(obs_dict)
