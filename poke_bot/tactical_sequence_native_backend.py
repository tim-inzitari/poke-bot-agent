"""Official-stock Search adapter for shadow tactical-sequence supervision.

The adapter owns one raw search arena inside the already-owned bounded child.
It never dispatches a real action.  Hidden-card guesses are used only to make
the stock simulator ABI operational; any transition that changes a hidden
zone count is marked nondeterministic and therefore cannot become a proof or
training label.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from .alakazam_tactical_sequence import compile_alakazam_public_tactical_facts
from .tactical_sequence_planner import (
    Action,
    TacticalSearchState,
    TacticalSequenceError,
    TacticalTransition,
)


LaneFactory = Callable[[], Any]

OFFICIAL_R236_LINUX_SHA256 = (
    "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7"
)
OFFICIAL_R236_LINUX_SIZE_BYTES = 1_342_400


def _raw(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        result = asdict(value)
        if isinstance(result, dict):
            return result
    raise TacticalSequenceError(
        f"stock Search observation is not JSON-like: {type(value).__name__}"
    )


def _fingerprint(value: Mapping[str, Any]) -> str:
    try:
        body = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TacticalSequenceError(
            "stock Search observation is not canonical JSON"
        ) from exc
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _semantic_fingerprint(
    raw: Mapping[str, Any], action_history: Sequence[Action]
) -> str:
    body = json.dumps(
        {
            "observation": dict(raw),
            "simulated_action_history": [list(action) for action in action_history],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _current(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    current = raw.get("current")
    if not isinstance(current, Mapping):
        raise TacticalSequenceError("stock Search observation lacks current state")
    return current


def _actor_and_turn(raw: Mapping[str, Any]) -> tuple[int, int | str]:
    current = _current(raw)
    actor = current.get("yourIndex")
    if isinstance(actor, bool) or not isinstance(actor, int) or actor not in (0, 1):
        raise TacticalSequenceError("stock Search observation has invalid actor")
    turn = current.get("turn", -1)
    if isinstance(turn, bool) or not isinstance(turn, (int, str)):
        raise TacticalSequenceError("stock Search observation has invalid turn")
    return int(actor), turn


def _information_boundary(raw: Mapping[str, Any]) -> bool:
    selection = raw.get("select")
    current = raw.get("current")
    return bool(
        isinstance(selection, Mapping) and selection.get("deck") is not None
    ) or bool(isinstance(current, Mapping) and current.get("looking") is not None)


def _chance_boundary(raw: Mapping[str, Any]) -> bool:
    selection = raw.get("select")
    return bool(
        isinstance(selection, Mapping) and selection.get("context") == 46
    )


def _zone_counts(raw: Mapping[str, Any]) -> tuple[tuple[int, int, int], ...]:
    players = _current(raw).get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise TacticalSequenceError("stock Search observation lacks two players")
    result: list[tuple[int, int, int]] = []
    for player in players:
        if not isinstance(player, Mapping):
            raise TacticalSequenceError("stock Search player is malformed")
        try:
            result.append(
                (
                    int(player.get("deckCount", 0) or 0),
                    int(player.get("handCount", 0) or 0),
                    len(player.get("prize") or ()),
                )
            )
        except (TypeError, ValueError) as exc:
            raise TacticalSequenceError("stock Search zone count is malformed") from exc
    return tuple(result)


def _legal_actions(raw: Mapping[str, Any]) -> tuple[tuple[Action, ...], int]:
    from . import cg_env, features

    if cg_env.is_finished(dict(raw)):
        return (), 0
    if _chance_boundary(raw) or _information_boundary(raw):
        return (), 0
    try:
        actions = tuple(
            tuple(int(item) for item in action)
            for action in features.enumerate_action_combos(
                dict(raw), max_combos=65
            )
        )
    except features.ActionSpaceTooLarge:
        return (), 65
    return actions, len(actions)


def tactical_state_from_public_observation(
    raw_observation: Mapping[str, Any],
    *,
    previous_action: Sequence[int] | None = None,
    simulated_action_history: Sequence[Sequence[int]] = (),
    simulated_observation_history: Sequence[Mapping[str, Any]] = (),
) -> TacticalSearchState:
    """Build one public, non-serving planner state from a stock observation."""

    from . import cg_env

    raw = dict(raw_observation)
    actor, turn = _actor_and_turn(raw)
    actions, ordered_count = _legal_actions(raw)
    public_facts: dict[str, Any] = {}
    visible_tutor_cards: tuple[int, ...] = ()
    try:
        facts = compile_alakazam_public_tactical_facts(raw)
        public_facts = facts.to_public_facts()
        visible_tutor_cards = facts.visible_tutor_card_ids
    except Exception:
        # SME facts are optional hints.  Structural simulator state remains
        # usable, while no missing fact can become a positive public goal.
        pass
    action_history = tuple(tuple(int(item) for item in row) for row in simulated_action_history)
    return TacticalSearchState(
        observation_fingerprint=_fingerprint(raw),
        semantic_fingerprint=_semantic_fingerprint(raw, action_history),
        actor=actor,
        turn_id=turn,
        legal_actions=actions,
        ordered_action_count=ordered_count,
        terminal_winner=cg_env.result_winner(raw),
        explicit_chance_boundary=_chance_boundary(raw),
        information_boundary=_information_boundary(raw),
        public_facts=public_facts,
        public_facts_are_observed=True,
        visible_tutor_cards=visible_tutor_cards,
        previous_action_token=(
            None
            if previous_action is None
            else tuple(int(item) for item in previous_action)
        ),
        raw_observation=raw,
        simulated_action_history=action_history,
        simulated_observation_history=tuple(
            dict(row) for row in simulated_observation_history
        ),
    )


class StockSearchTacticalWorker:
    """One thread/process-owned official Search arena for sequential roots."""

    def __init__(self, *, lane_factory: LaneFactory, deck: Sequence[int]) -> None:
        self._lane = lane_factory()
        self._deck = tuple(int(card) for card in deck)
        if len(self._deck) != 60:
            raise TacticalSequenceError("tactical stock worker requires a 60-card deck")
        self._search_ids: dict[str, int] = {}
        self._root_open = False

    def _open_root(self, state: TacticalSearchState) -> int:
        from . import cg_env

        raw = state.raw_observation
        if not isinstance(raw, Mapping):
            raise TacticalSequenceError("tactical root lacks raw public observation")
        if self._root_open:
            self._lane.search_end()
        inputs = cg_env.build_search_inputs(
            dict(raw), list(self._deck), opponent_deck_guess=list(self._deck)
        )
        opened = self._lane.search_begin(dict(raw), inputs, manual_coin=True)
        search_id = int(getattr(opened, "searchId"))
        self._root_open = True
        self._search_ids = {state.semantic_fingerprint: search_id}
        return search_id

    def advance(
        self, state: TacticalSearchState, action: Action
    ) -> TacticalTransition:
        if action not in state.legal_actions:
            raise TacticalSequenceError("tactical stock worker received illegal action")
        search_id = self._search_ids.get(state.semantic_fingerprint)
        if search_id is None:
            if state.simulated_action_history:
                raise TacticalSequenceError("tactical stock Search state was not retained")
            search_id = self._open_root(state)
        stepped = self._lane.search_step(search_id, list(action))
        raw = _raw(getattr(stepped, "observation"))
        action_history = state.simulated_action_history + (tuple(action),)
        observation_history = state.simulated_observation_history + (raw,)
        next_state = tactical_state_from_public_observation(
            raw,
            previous_action=action,
            simulated_action_history=action_history,
            simulated_observation_history=observation_history,
        )
        self._search_ids[next_state.semantic_fingerprint] = int(
            getattr(stepped, "searchId")
        )
        hidden_zone_crossed = _zone_counts(
            dict(state.raw_observation or {})
        ) != _zone_counts(raw)
        deterministic = not hidden_zone_crossed
        if next_state.terminal_winner is not None and (
            next_state.explicit_chance_boundary
            or next_state.information_boundary
        ):
            deterministic = False
        return TacticalTransition(
            next_state=next_state,
            action_token=action,
            deterministic=deterministic,
        )

    def close(self) -> None:
        if self._root_open:
            self._lane.search_end()
            self._root_open = False


@dataclass(frozen=True)
class OfficialStockTacticalWorkerFactory:
    """Pickle-safe factory used only inside an owned spawned child."""

    deck: tuple[int, ...]
    expected_library_sha256: str = OFFICIAL_R236_LINUX_SHA256
    expected_library_size_bytes: int = OFFICIAL_R236_LINUX_SIZE_BYTES

    def __call__(self) -> StockSearchTacticalWorker:
        from pathlib import Path

        from .cg_env import NativeSearchLane, prewarm_native_search_runtime

        api, sim = prewarm_native_search_runtime()
        library_path = Path(str(getattr(sim.lib, "_name", ""))).resolve(strict=True)
        if (
            library_path.stat().st_size != int(self.expected_library_size_bytes)
            or "sha256:" + hashlib.sha256(library_path.read_bytes()).hexdigest()
            != self.expected_library_sha256
        ):
            raise TacticalSequenceError(
                "tactical stock child did not load canonical r236 libcg"
            )

        def lane_factory() -> NativeSearchLane:
            return NativeSearchLane(0, lib=sim.lib, api_module=api)

        return StockSearchTacticalWorker(
            lane_factory=lane_factory,
            deck=self.deck,
        )


__all__ = [
    "StockSearchTacticalWorker",
    "OfficialStockTacticalWorkerFactory",
    "tactical_state_from_public_observation",
]
