"""Thread-affine raw stock-libcg search handles for the r225 viability probe.

This is deliberately separate from the archived r195 package's ``cg_env``
module.  The diagnostic bundle must leave the frozen direct-policy import path
alone while still proving that the stock Search ABI can isolate eight raw
``AgentStart`` handles.  Nothing in this module calls the competition wrapper's
module-global ``cg.api.agent_ptr``.

Only the official ABI is used:

``AgentStart -> SearchBegin -> SearchStep* -> SearchRelease* -> SearchEnd``.

The handle is permanently owned by the thread which created it.  There is no
documented ``AgentFinish`` in the stock library, so the caller retains a handle
until its process exits rather than inventing a native destructor.
"""

from __future__ import annotations

import ctypes
import threading
from typing import Any, Mapping, Sequence


class R225NativeLaneError(RuntimeError):
    """A raw stock Search handle could not safely complete its lifecycle."""


def prewarm_stock_cg() -> tuple[Any, Any]:
    """Import the vendored stock binding before any lane thread starts.

    ``cg.sim`` initializes process-global card metadata on import.  The
    coordinator calls this once before creating worker threads so that lane
    startup measures raw handle isolation, not racy global initialization.
    """

    import cg.api as api
    import cg.sim as sim

    return api, sim


class R225StockNativeSearchLane:
    """One raw ``ApiData*`` bound to exactly one persistent worker thread."""

    def __init__(self, lane_id: int, *, lib: Any, api_module: Any) -> None:
        if int(lane_id) < 0:
            raise ValueError("lane_id must be non-negative")
        self.lane_id = int(lane_id)
        self._lib = lib
        self._api = api_module
        self._owner_thread_id = threading.get_ident()
        self._handle = self._lib.AgentStart()
        if not self._handle:
            raise R225NativeLaneError(
                f"AgentStart failed for r225 search lane {self.lane_id}"
            )
        self._live_search_ids: set[int] = set()
        self._search_end_calls = 0

    @property
    def owner_thread_id(self) -> int:
        return self._owner_thread_id

    @property
    def handle_identity(self) -> int | str:
        try:
            return int(self._handle)
        except (TypeError, ValueError):
            return repr(self._handle)

    @property
    def live_search_ids(self) -> frozenset[int]:
        return frozenset(self._live_search_ids)

    @property
    def search_end_calls(self) -> int:
        return self._search_end_calls

    def _assert_owner(self) -> None:
        current = threading.get_ident()
        if current != self._owner_thread_id:
            raise R225NativeLaneError(
                f"r225 search lane {self.lane_id} belongs to thread "
                f"{self._owner_thread_id}, not {current}"
            )

    @staticmethod
    def _int_array(values: Sequence[int]) -> Any:
        clean = [int(value) for value in values]
        return (ctypes.c_int * len(clean))(*clean)

    def _decode_result(self, payload: Any) -> Any:
        return self._api.json_to_dataclass(payload, self._api.ApiResult)

    @staticmethod
    def _raise_begin_error(error: int) -> None:
        messages = {
            1: "Invalid Card ID.",
            2: "Active card must be the ID of a Pokémon card.",
            30: "stock raw AgentStart handle is broken.",
        }
        raise R225NativeLaneError(
            "SearchBegin failed: " + messages.get(int(error), f"error={error}")
        )

    @staticmethod
    def _raise_step_error(error: int) -> None:
        messages = {
            1: "unknown search id",
            2: "released search id",
            3: "battle ended",
            4: "select count is illegal",
            5: "select option is illegal",
            6: "select has duplicate option",
            30: "stock raw AgentStart handle is broken",
        }
        raise R225NativeLaneError(
            "SearchStep failed: " + messages.get(int(error), f"error={error}")
        )

    @staticmethod
    def _as_observation(api_module: Any, observation: Any) -> Any:
        return (
            api_module.to_observation_class(observation)
            if isinstance(observation, Mapping)
            else observation
        )

    def search_begin(
        self,
        obs_dict: Mapping[str, Any] | Any,
        search_inputs: Mapping[str, Sequence[int]],
        *,
        manual_coin: bool = True,
    ) -> Any:
        """Open one exact raw Search root, exposing chance rather than rolling it."""

        self._assert_owner()
        observation = self._as_observation(self._api, obs_dict)
        encoded_input = getattr(observation, "search_begin_input", None)
        current = getattr(observation, "current", None)
        if encoded_input is None or current is None:
            raise R225NativeLaneError("SearchBegin requires an agent current observation")
        your_index = int(current.yourIndex)
        players = current.players
        if len(players) != 2:
            raise R225NativeLaneError("SearchBegin observation does not have two players")

        def cards(name: str) -> list[int]:
            return [int(value) for value in search_inputs.get(name, ())]

        your_deck = cards("your_deck")
        your_prize = cards("your_prize")
        opponent_deck = cards("opponent_deck")
        opponent_prize = cards("opponent_prize")
        opponent_hand = cards("opponent_hand")
        opponent_active = cards("opponent_active")
        select = getattr(observation, "select", None)
        if select is not None and getattr(select, "deck", None) is not None:
            your_deck = []
        elif len(your_deck) < int(players[your_index].deckCount):
            raise R225NativeLaneError("predicted own deck is shorter than public count")
        if len(your_prize) < len(players[your_index].prize):
            raise R225NativeLaneError("predicted own prizes are shorter than public count")
        opponent_index = 1 - your_index
        if len(opponent_deck) < int(players[opponent_index].deckCount):
            raise R225NativeLaneError("predicted opponent deck is shorter than public count")
        if len(opponent_prize) < len(players[opponent_index].prize):
            raise R225NativeLaneError("predicted opponent prizes are shorter than public count")
        if len(opponent_hand) < int(players[opponent_index].handCount):
            raise R225NativeLaneError("predicted opponent hand is shorter than public count")
        active = players[opponent_index].active
        if active and active[0] is None:
            if not opponent_active:
                raise R225NativeLaneError("predicted opponent active is required")
        else:
            opponent_active = []

        encoded = str(encoded_input).encode("ascii")
        payload = self._lib.SearchBegin(
            self._handle,
            encoded,
            len(encoded),
            self._int_array(your_deck),
            self._int_array(your_prize),
            self._int_array(opponent_deck),
            self._int_array(opponent_prize),
            self._int_array(opponent_hand),
            self._int_array(opponent_active),
            int(bool(manual_coin)),
        )
        result = self._decode_result(payload)
        error = int(result.error)
        if error:
            self._raise_begin_error(error)
        state = result.state
        self._live_search_ids.add(int(state.searchId))
        return state

    def search_step(self, search_id: int, select: Sequence[int]) -> Any:
        self._assert_owner()
        search_id = int(search_id)
        if search_id not in self._live_search_ids:
            raise R225NativeLaneError(
                f"search id {search_id} is not owned by lane {self.lane_id}"
            )
        chosen = [int(value) for value in select]
        payload = self._lib.SearchStep(
            self._handle,
            search_id,
            self._int_array(chosen),
            len(chosen),
        )
        result = self._decode_result(payload)
        error = int(result.error)
        if error:
            self._raise_step_error(error)
        state = result.state
        self._live_search_ids.add(int(state.searchId))
        return state

    def search_release(self, search_id: int) -> None:
        self._assert_owner()
        search_id = int(search_id)
        if search_id not in self._live_search_ids:
            raise R225NativeLaneError(
                f"search id {search_id} is not owned by lane {self.lane_id}"
            )
        self._lib.SearchRelease(self._handle, search_id)
        self._live_search_ids.remove(search_id)

    def search_end(self) -> None:
        self._assert_owner()
        self._lib.SearchEnd(self._handle)
        self._live_search_ids.clear()
        self._search_end_calls += 1


__all__ = [
    "R225NativeLaneError",
    "R225StockNativeSearchLane",
    "prewarm_stock_cg",
]
