"""Stock-libcg eight-worker shared-tree action authority for the r228 smoke.

This is intentionally a small viability runtime.  One competition process owns
one frozen model and one shared search tree per decision.  Eight persistent
thread-affine ``AgentStart`` arenas advance independent simulator states and
feed ready leaves to the same model-backed coordinator queue.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from .r228_async_shared_tree_queue import (
    AsyncEightWorkerError,
    DecodedLeaf,
    PersistentAsyncEightWorkerMCTS,
)

SCHEMA = "poke_bot.r228_async_eight_worker_kaggle_viability/v1"
DECISION_PREFIX = "R228_ASYNC_EIGHT_WORKER_DECISION"
# This is deliberately explicit at every r228 complete-action materialization
# site.  The staged r228 package preserves the frozen r195 feature module, so
# changing a workspace-global default does not change the submitted runtime.
R228_COMPLETE_ACTION_CAP = 65_536
STOCK_LIBRARY_SHA256 = {
    "libcg.so": "ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c",
    "libcg.dylib": "77bb978a8129b094452679e0daf0da69593afda7331685f4642c0d4a94d39d82",
    "libcg-arm64.so": "030b4728ce9fb9e90b75830b7cf7236f71859732a05ec4a377078eee0421bbe5",
    "cg.dll": "9ea2b0a751029689bff3ddccb5f29a98edd46961dad264490ed121ef704fb500",
}


class R228GameplayError(RuntimeError):
    """The shared-tree action could not be proven legal and backed."""


@dataclass(frozen=True)
class _Frontier:
    lane_id: int
    raw: dict[str, Any]


def _raw(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        result = asdict(value)
        if isinstance(result, dict):
            return result
    raise R228GameplayError(f"unsupported simulator observation: {type(value).__name__}")


def _state_key(*, lane_id: int | None, raw: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"lane_id": lane_id, "observation": raw},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return "r228:" + hashlib.sha256(encoded).hexdigest()


def _chance_boundary(raw: Mapping[str, Any]) -> bool:
    selection = raw.get("select")
    return isinstance(selection, Mapping) and selection.get("context") == 46


def _positive_seconds_from_env(name: str, default: str) -> float:
    """Read a finite positive runtime deadline without silently disabling it."""

    raw = os.environ.get(name, default)
    try:
        seconds = float(raw)
    except (TypeError, ValueError) as exc:
        raise R228GameplayError(f"{name} must be a positive number") from exc
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise R228GameplayError(f"{name} must be a finite positive number")
    return seconds


class R228AsyncGameplay:
    """Process-wide persistent simulator workers with per-decision shared trees."""

    def __init__(
        self,
        *,
        stage: Path,
        model: Any,
        policy: Any,
        deck: Sequence[int],
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        from .r225_stock_native_lane import (
            R225StockNativeSearchLane,
            prewarm_stock_cg,
        )

        self.stage = stage.resolve()
        self.model = model
        self.policy = policy
        self.deck = tuple(int(card) for card in deck)
        self.cleanup_timeout_seconds = _positive_seconds_from_env(
            "POKEBOT_R228_CLEANUP_TIMEOUT_SECONDS", "1.0"
        )
        if len(self.deck) != 60:
            raise R228GameplayError("frozen r195 deck is not 60 cards")
        api, sim = prewarm_stock_cg()
        loaded = Path(str(getattr(sim.lib, "_name", ""))).resolve()
        cg_root = (self.stage / "cg").resolve()
        if loaded.parent != cg_root or loaded.name not in STOCK_LIBRARY_SHA256:
            raise R228GameplayError(f"stock libcg path is outside the sealed package: {loaded}")
        digest = hashlib.sha256(loaded.read_bytes()).hexdigest()
        if digest != STOCK_LIBRARY_SHA256[loaded.name]:
            raise R228GameplayError(f"stock {loaded.name} digest mismatch")
        self.stock_library_receipt = {
            "path": str(loaded),
            "member": f"cg/{loaded.name}",
            "sha256": f"sha256:{digest}",
            "platform": platform.system().lower(),
            "machine": platform.machine().lower(),
        }

        def arena_factory(lane_id: int) -> Any:
            return R225StockNativeSearchLane(lane_id, lib=sim.lib, api_module=api)

        self._decision: dict[str, Any] | None = None
        self._search = PersistentAsyncEightWorkerMCTS(
            arena_factory=arena_factory,
            make_packet=self._make_packet,
            evaluate_batch=self._evaluate_batch,
            coalesce_seconds=float(os.environ.get("POKEBOT_R228_COALESCE_SECONDS", "0.001")),
            cleanup_timeout_seconds=self.cleanup_timeout_seconds,
            progress_callback=progress_callback,
        )
        self.decision_count = 0
        self.decision_receipts: list[dict[str, Any]] = []

    def close(self) -> None:
        self._search.close()

    def reset_game(self) -> None:
        self.decision_count = 0
        self.decision_receipts.clear()
        reset = getattr(self.policy, "reset_game", None)
        if callable(reset):
            reset()

    def _make_packet(self, lane_id: int, observation: Any) -> _Frontier:
        return _Frontier(lane_id=int(lane_id), raw=_raw(observation))

    def _leaf_packet(
        self,
        raw: dict[str, Any],
        *,
        combos: Sequence[Sequence[int]],
        actor_seat: int,
    ) -> Any:
        from . import features
        from .batched_infer import LeafPacket

        context = self._decision
        if context is None:
            raise R228GameplayError("leaf evaluation has no active decision")
        if actor_seat not in (0, 1):
            raise R228GameplayError("simulator leaf has invalid acting seat")
        acting_deck = (
            self.deck
            if actor_seat == context["root_seat"]
            else context["opponent_deck"]
        )
        board = features.build_board_tokens(raw, list(acting_deck))
        histories = list(context["history_boards"]) + [board]
        previous = list(context["history_previous_actions"]) + [None]
        limit = int(self.policy._history_context_limit())
        return LeafPacket(
            obs=raw,
            your_deck=list(acting_deck),
            root_seat=int(context["root_seat"]),
            history_boards=histories[-limit:],
            history_previous_actions=previous[-limit:],
            action_combos_override=[list(map(int, action)) for action in combos],
            matchup_route=int(context["route"]),
        )

    def _evaluate_batch(self, frontiers: Sequence[_Frontier]) -> Sequence[DecodedLeaf]:
        from . import cg_env, features
        from .batched_infer import forward_leaf_batch

        context = self._decision
        if context is None:
            raise R228GameplayError("model batch has no active decision")
        pending: list[
            tuple[int, _Frontier, tuple[tuple[int, ...], ...], bool, int]
        ] = []
        decoded: list[DecodedLeaf | None] = [None] * len(frontiers)
        packets: list[Any] = []
        for index, frontier in enumerate(frontiers):
            raw = frontier.raw
            if cg_env.is_finished(raw):
                winner = cg_env.result_winner(raw)
                value = 0.0 if winner in (None, 2) else (1.0 if int(winner) == context["root_seat"] else -1.0)
                decoded[index] = DecodedLeaf(
                    state_key=_state_key(lane_id=frontier.lane_id, raw=raw),
                    value=value,
                    legal_actions=(),
                    priors=(),
                    boundary=True,
                    actor_seat=None,
                )
                continue
            current = raw.get("current")
            if not isinstance(current, Mapping):
                raise R228GameplayError("simulator leaf has no current state")
            actor = int(current.get("yourIndex", -1))
            if actor not in (0, 1):
                raise R228GameplayError("simulator leaf has invalid acting seat")
            combos = tuple(
                tuple(int(item) for item in action)
                for action in features.enumerate_action_combos(
                    raw, max_combos=R228_COMPLETE_ACTION_CAP
                )
            )
            if not combos:
                raise R228GameplayError("nonterminal simulator leaf has no legal actions")
            boundary = _chance_boundary(raw)
            packets.append(
                self._leaf_packet(raw, combos=combos, actor_seat=actor)
            )
            pending.append((index, frontier, combos, boundary, actor))
        evaluated = forward_leaf_batch(self.model, packets) if packets else []
        if len(evaluated) != len(pending):
            raise R228GameplayError("frozen model returned a partial leaf batch")
        for (index, frontier, combos, boundary, actor), leaf in zip(
            pending, evaluated
        ):
            priors = tuple(float(value) for value in leaf.priors)
            if len(priors) != len(combos) or any(
                not math.isfinite(value) or value < 0.0 for value in priors
            ):
                raise R228GameplayError("frozen model returned malformed leaf priors")
            decoded[index] = DecodedLeaf(
                state_key=_state_key(lane_id=frontier.lane_id, raw=frontier.raw),
                value=float(leaf.value),
                legal_actions=() if boundary else combos,
                priors=() if boundary else priors,
                boundary=boundary,
                actor_seat=actor,
            )
        if any(row is None for row in decoded):
            raise R228GameplayError("leaf decode was incomplete")
        return tuple(row for row in decoded if row is not None)

    def select(self, obs: dict[str, Any]) -> list[int]:
        """Return the actual shared-tree action for one branching prompt."""

        from . import cg_env, features
        from .batched_infer import LeafPacket, forward_leaf_batch

        legal = tuple(
            tuple(int(item) for item in action)
            for action in features.enumerate_action_combos(
                obs, max_combos=R228_COMPLETE_ACTION_CAP
            )
        )
        if len(legal) < 2:
            raise R228GameplayError("async search requires a branching prompt")
        current = obs.get("current")
        if not isinstance(current, Mapping):
            raise R228GameplayError("branching observation has no current state")
        root_seat = int(current.get("yourIndex", -1))
        if root_seat not in (0, 1):
            raise R228GameplayError("branching observation has invalid acting seat")

        router = getattr(self.policy, "_matchup_adapter_shadow_router", None)
        if router is not None and hasattr(router, "observe"):
            router.observe(obs, scope="game_root", depth=len(self.policy.board_history))
        self.policy._append_decision_history(obs)
        route = int(self.policy._matchup_model_route())
        root_packet = LeafPacket(
            obs=obs,
            your_deck=list(self.deck),
            root_seat=root_seat,
            history_boards=list(self.policy.board_history),
            history_previous_actions=list(self.policy.previous_action_history),
            action_combos_override=[list(action) for action in legal],
            matchup_route=route,
        )
        root_leaf = forward_leaf_batch(self.model, [root_packet])[0]
        root_priors = tuple(float(value) for value in root_leaf.priors)
        if (
            len(root_priors) != len(legal)
            or any(not math.isfinite(value) or value < 0.0 for value in root_priors)
            or math.fsum(root_priors) <= 0.0
        ):
            raise R228GameplayError("frozen model returned malformed root priors")
        direct_index = max(
            range(len(legal)), key=lambda index: (root_priors[index], -index)
        )
        direct_action = legal[direct_index]
        direct_probability = root_priors[direct_index]

        search_inputs = cg_env.build_search_inputs(
            obs,
            list(self.deck),
            opponent_deck_guess=list(self.deck),
        )
        opponent = tuple(int(card) for card in search_inputs.get("opponent_deck", ()))
        if len(opponent) != 60:
            opponent = self.deck
        self._decision = {
            "root_seat": root_seat,
            "route": route,
            "opponent_deck": opponent,
            "history_boards": list(self.policy.board_history),
            "history_previous_actions": list(self.policy.previous_action_history),
        }
        seconds = max(0.25, float(os.environ.get("POKEBOT_R228_DECISION_SECONDS", "8.0")))
        started = time.monotonic()
        try:
            receipt = self._search.run_decision(
                root_observation=obs,
                search_inputs=tuple(dict(search_inputs) for _ in range(8)),
                root_state_key=_state_key(lane_id=None, raw=obs),
                root_actions=legal,
                root_priors=root_priors,
                root_seat=root_seat,
                deadline_monotonic=started + seconds,
            )
            selected = tuple(int(item) for item in receipt.selected_action)
            if selected not in legal or receipt.selected_action_visits < 1:
                raise R228GameplayError("shared tree returned an unbacked or illegal action")
            mode = "shared_tree_mcts"
        except AsyncEightWorkerError as exc:
            # The core emits this exact post-cleanup failure only when the
            # deadline produced zero backups.  Structural/native failures are
            # deliberately not downgraded in this viability submission.
            if "completed no backups" not in str(exc):
                raise
            # The root model pass above has already produced the frozen direct
            # action from this exact complete legal ordering.  Reusing it keeps
            # a clean-deadline fallback bounded and avoids mutating policy
            # history through a second greedy inference path.
            selected = direct_action
            if selected not in legal:
                raise R228GameplayError("clean-deadline frozen fallback was illegal")
            receipt = None
            mode = "clean_deadline_zero_backup_frozen_model_fallback"
        finally:
            self._decision = None

        self.policy._previous_action_token = features.build_option_tokens(
            obs, [list(selected)]
        )
        self.decision_count += 1
        payload = {
            "schema": SCHEMA,
            "decision": self.decision_count,
            "mode": mode,
            "mcts_action_authority": mode == "shared_tree_mcts",
            "selected_action": list(selected),
            "legal_action_count": len(legal),
            "complete_ordered_action_cap": R228_COMPLETE_ACTION_CAP,
            "cleanup_timeout_seconds": self.cleanup_timeout_seconds,
            "direct_action": list(direct_action),
            "direct_action_probability": direct_probability,
            "mcts_action_direct_probability": root_priors[legal.index(selected)],
            "mcts_action_direct_rank": 1 + sum(
                probability > root_priors[legal.index(selected)]
                for probability in root_priors
            ),
            "direct_probability_gap": direct_probability
            - root_priors[legal.index(selected)],
            "action_changed": selected != direct_action,
        }
        if receipt is not None:
            payload.update(
                {
                    "arena_count": receipt.arena_count,
                    "unique_handle_count": receipt.unique_handle_count,
                    "search_begin_calls": receipt.search_begin_calls,
                    "search_step_calls": receipt.search_step_calls,
                    "completed_backups": receipt.completed_backups,
                    "selected_action_visits": receipt.selected_action_visits,
                    "selected_action_value": receipt.selected_action_value,
                    "selected_action_prior": receipt.selected_action_prior,
                    "root_visits": receipt.root_visits,
                    "max_simulator_calls_in_flight": receipt.max_simulator_calls_in_flight,
                    "microbatch_sizes": list(receipt.microbatch_sizes),
                    "per_lane_depth": list(receipt.per_lane_depth),
                    "search_release_calls": receipt.search_release_calls,
                    "search_end_calls": receipt.search_end_calls,
                    "outstanding_virtual_loss": receipt.outstanding_virtual_loss,
                    "elapsed_seconds": receipt.elapsed_seconds,
                    "meaningful_choice_change": (
                        selected != direct_action
                        and receipt.completed_backups > 0
                        and direct_probability - root_priors[legal.index(selected)]
                        > float(os.environ.get("POKEBOT_R229_PROBABILITY_TIE_TOLERANCE", "1e-6"))
                    ),
                }
            )
        payload["stock_library"] = dict(self.stock_library_receipt)
        self.decision_receipts.append(dict(payload))
        print(DECISION_PREFIX + " " + json.dumps(payload, sort_keys=True), flush=True)
        return list(selected)


__all__ = [
    "DECISION_PREFIX",
    "R228_COMPLETE_ACTION_CAP",
    "SCHEMA",
    "R228AsyncGameplay",
    "R228GameplayError",
]
