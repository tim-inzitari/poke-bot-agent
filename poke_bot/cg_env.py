"""Thin wrapper over the competition ``cg`` Game + Search APIs.

This is the only module that touches the ``cg`` runtime directly. It:
  - puts the correct ``cg`` package on ``sys.path`` (competition copy preferred,
    honouring ``$CG_LIB_PATH``), so ``import cg`` works from anywhere;
  - re-exports the Game API (``battle_start`` / ``battle_select`` /
    ``battle_finish`` / ``visualize_data``) and card metadata;
  - wraps the Search API (``search_begin`` / ``search_step`` / ``search_release``
    / ``search_end``) that the MCTS phase builds on;
  - provides small helpers used by the smoke test and later phases:
    observation parsing, terminal detection, legal-select sampling, and a naive
    predicted-deck builder for ``search_begin``.

Nothing here imports torch or numpy — the simulator path stays lightweight.
"""

from __future__ import annotations

import ctypes
import random
import sys
import threading
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from . import paths

# ---------------------------------------------------------------------------
# cg import bootstrap
# ---------------------------------------------------------------------------

_CG_READY = False


def ensure_cg_importable() -> Path:
    """Ensure ``import cg`` resolves; return the runtime parent directory.

    Idempotent. Prepends the resolved runtime parent (see
    :func:`paths.cg_runtime_dir`) to ``sys.path``.
    """
    global _CG_READY
    parent = paths.cg_runtime_dir()
    sp = str(parent)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    _CG_READY = True
    return parent


def _cg_api():
    if not _CG_READY:
        ensure_cg_importable()
    import cg.api as api

    return api


def _cg_game():
    if not _CG_READY:
        ensure_cg_importable()
    import cg.game as game

    return game


# ---------------------------------------------------------------------------
# Enum / dataclass re-exports (lazy attribute access)
# ---------------------------------------------------------------------------

_EXPORTED = {
    "AreaType", "EnergyType", "CardType", "SpecialConditionType", "SelectType",
    "SelectContext", "OptionType", "LogType", "Card", "Pokemon", "PlayerState",
    "State", "Option", "SelectData", "Log", "Observation", "SearchState",
    "CardData", "Attack",
}


def __getattr__(name: str) -> Any:  # module-level lazy re-export
    if name in _EXPORTED:
        return getattr(_cg_api(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Card / attack metadata
# ---------------------------------------------------------------------------

def all_card_data() -> list:
    """Return every :class:`cg.api.CardData` known to the engine."""
    return _cg_api().all_card_data()


def all_attack() -> list:
    """Return every :class:`cg.api.Attack` known to the engine."""
    return _cg_api().all_attack()


def to_observation(obs_dict: dict):
    """Convert a raw observation dict into a :class:`cg.api.Observation`."""
    return _cg_api().to_observation_class(obs_dict)


# ---------------------------------------------------------------------------
# Game API passthroughs
# ---------------------------------------------------------------------------

def battle_start(deck0: list[int], deck1: list[int]) -> tuple[Optional[dict], Any]:
    """Start a battle between two 60-card decks. Returns ``(obs_dict, StartData)``."""
    return _cg_game().battle_start(deck0, deck1)


def battle_select(select_list: list[int]) -> dict:
    """Apply a selection (list of option indices) and return the next obs dict."""
    return _cg_game().battle_select(select_list)


def battle_finish() -> None:
    """End the current battle and free its memory."""
    _cg_game().battle_finish()


def visualize_data() -> str:
    """Return the visualizer payload for the current battle."""
    return _cg_game().visualize_data()


# ---------------------------------------------------------------------------
# Terminal detection / results
# ---------------------------------------------------------------------------

def is_finished(obs_dict: dict) -> bool:
    """True if the battle in this observation has ended."""
    cur = obs_dict.get("current") if obs_dict else None
    if cur is not None and cur.get("result", -1) != -1:
        return True
    for log in (obs_dict.get("logs") or []):
        # LogType.RESULT == 23
        if log.get("type") == 23:
            return True
    return False


def result_winner(obs_dict: dict) -> Optional[int]:
    """Return the winning player index (0/1), 2 for draw, or None if unfinished."""
    cur = obs_dict.get("current") if obs_dict else None
    if cur is not None and cur.get("result", -1) != -1:
        return cur["result"]
    for log in (obs_dict.get("logs") or []):
        if log.get("type") == 23:
            return log.get("result")
    return None


# ---------------------------------------------------------------------------
# Legal-select sampling (raw dict form, no cg import needed)
# ---------------------------------------------------------------------------

def random_legal_select(obs_dict: dict, rng: Optional[random.Random] = None) -> list[int]:
    """Sample a legal selection for ``obs_dict`` (raw dict).

    Mirrors the sample agent: picks ``k`` distinct option indices where ``k`` is
    a random value in ``[minCount, maxCount]``. Returns ``[]`` when there is no
    active select (deck-selection is handled by the caller).
    """
    rng = rng or random
    sel = obs_dict.get("select") if obs_dict else None
    if sel is None:
        return []
    n = len(sel.get("option", []))
    lo = sel.get("minCount", 0)
    hi = min(sel.get("maxCount", 0), n)
    if hi <= 0:
        return []
    lo = max(0, min(lo, hi))
    k = rng.randint(lo, hi) if hi >= lo else hi
    if k <= 0:
        return []
    return rng.sample(range(n), k)


# ---------------------------------------------------------------------------
# Search API
# ---------------------------------------------------------------------------

def _sized(source: list[int], n: int, fallback: int = 119) -> list[int]:
    """Return a length-``n`` list of card ids by cycling ``source``.

    ``fallback`` (default Dreepy, a Basic Pokémon) is used when ``source`` is
    empty so setup predictions still contain a Basic Pokémon.
    """
    if n <= 0:
        return []
    if not source:
        source = [fallback]
    return [source[i % len(source)] for i in range(n)]


def build_search_inputs(
    obs_dict: dict,
    your_deck: list[int],
    opponent_deck_guess: Optional[list[int]] = None,
    *,
    fallback_basic: int = 119,
) -> dict:
    """Build the predicted-card arguments required by :func:`search_begin`.

    This is a *naive* predictor good enough to instantiate a search tree (the
    MCTS phase will supply belief-aware predictions). It sizes each hidden zone
    correctly from the observation and fills it with plausible, valid card ids.

    Returns a dict with keys matching :func:`search_begin` kwargs.
    """
    obs = to_observation(obs_dict)
    state = obs.current
    if state is None:
        raise ValueError("build_search_inputs requires a post-setup observation (current is None).")
    yi = state.yourIndex
    me = state.players[yi]
    opp = state.players[1 - yi]
    opp_guess = opponent_deck_guess or your_deck

    # your_deck is ignored by the engine when the observation exposes the deck.
    if obs.select is not None and obs.select.deck is not None:
        your_deck_pred: list[int] = []
    else:
        your_deck_pred = _sized(your_deck, me.deckCount, fallback_basic)

    opp_active = opp.active
    needs_active = len(opp_active) > 0 and opp_active[0] is None
    opponent_active = [fallback_basic] if needs_active else []

    return dict(
        your_deck=your_deck_pred,
        your_prize=_sized(your_deck, len(me.prize), fallback_basic),
        opponent_deck=_sized(opp_guess, opp.deckCount, fallback_basic),
        opponent_prize=_sized(opp_guess, len(opp.prize), fallback_basic),
        opponent_hand=_sized(opp_guess, opp.handCount, fallback_basic),
        opponent_active=opponent_active,
    )


def search_begin(obs_dict: dict, search_inputs: dict, manual_coin: bool = False):
    """Begin a search from ``obs_dict`` with predicted decks in ``search_inputs``.

    ``search_inputs`` is the dict returned by :func:`build_search_inputs`.
    Returns a :class:`cg.api.SearchState` (the search root).
    """
    obs = to_observation(obs_dict)
    return _cg_api().search_begin(obs, manual_coin=manual_coin, **search_inputs)


def search_step(search_id: int, select: list[int]):
    """Advance the search node ``search_id`` by choosing option indices ``select``."""
    return _cg_api().search_step(search_id, select)


def search_release(search_id: int) -> None:
    """Release a search node's memory for reuse."""
    _cg_api().search_release(search_id)


def search_end() -> None:
    """End the current search; its memory is reused by the next search."""
    _cg_api().search_end()


# ---------------------------------------------------------------------------
# Handle-scoped Search API (staged multi-lane submission path)
# ---------------------------------------------------------------------------


@runtime_checkable
class SearchBackend(Protocol):
    """The search operations consumed by :class:`BeliefMCTS`.

    The module-level functions above intentionally retain the competition
    wrapper's historical singleton behavior.  A concurrent caller must instead
    inject a backend whose lifecycle is scoped to one native ``ApiData*``.
    """

    def search_begin(
        self,
        obs_dict: dict,
        search_inputs: dict,
        manual_coin: bool = False,
    ) -> Any:
        ...

    def search_step(self, search_id: int, select: list[int]) -> Any:
        ...

    def search_release(self, search_id: int) -> None:
        ...

    def search_end(self) -> None:
        ...


def prewarm_native_search_runtime() -> tuple[Any, Any]:
    """Import and initialize the official binding on the coordinator thread.

    ``cg.sim`` performs the one process-global ``GameInitialize`` call while it
    imports.  Multi-lane workers receive these already initialized modules and
    never race card-table or metadata initialization.
    """

    ensure_cg_importable()
    import cg.api as api
    import cg.sim as sim

    return api, sim


class NativeSearchLane:
    """One thread-affine official Search API arena.

    The shipped Python wrapper stores one module-global ``agent_ptr``.  The
    native ABI is narrower and safer: every Search call accepts the ``ApiData*``
    returned by ``AgentStart``.  This adapter calls that ABI directly, owns one
    pointer for the process lifetime, and refuses calls from any thread other
    than the thread that created it.

    There is no documented ``AgentFinish`` symbol.  ``search_end`` clears only
    this handle's search arena; the fixed lane pool retains the handle until
    process exit rather than guessing that ``BattleFinish`` is a destructor.
    """

    def __init__(
        self,
        lane_id: int,
        *,
        lib: Any | None = None,
        api_module: Any | None = None,
    ) -> None:
        if int(lane_id) < 0:
            raise ValueError("lane_id must be non-negative")
        if lib is None or api_module is None:
            default_api, default_sim = prewarm_native_search_runtime()
            api_module = api_module if api_module is not None else default_api
            lib = lib if lib is not None else default_sim.lib
        self.lane_id = int(lane_id)
        self._lib = lib
        self._api = api_module
        self._owner_thread_id = threading.get_ident()
        handle = self._lib.AgentStart()
        if not handle:
            raise RuntimeError(f"AgentStart failed for search lane {self.lane_id}")
        self._handle = handle
        self._live_search_ids: set[int] = set()
        self._ended_calls = 0

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
        return self._ended_calls

    def _assert_owner(self) -> None:
        current = threading.get_ident()
        if current != self._owner_thread_id:
            raise RuntimeError(
                f"search lane {self.lane_id} belongs to thread "
                f"{self._owner_thread_id}, not {current}"
            )

    @staticmethod
    def _int_array(values: list[int]) -> Any:
        clean = [int(value) for value in values]
        return (ctypes.c_int * len(clean))(*clean)

    def _decode_result(self, payload: Any) -> Any:
        return self._api.json_to_dataclass(payload, self._api.ApiResult)

    @staticmethod
    def _raise_begin_error(error: int) -> None:
        if error == 1:
            raise ValueError("Invalid Card ID.")
        if error == 2:
            raise ValueError("Active card must be the ID of a Pokémon card.")
        if error == 30:
            raise ValueError("agent_ptr broken.")
        raise RuntimeError(f"SearchBegin failed with error {error}")

    @staticmethod
    def _raise_step_error(error: int) -> None:
        messages = {
            1: "There is no element with the specified search_id.",
            2: "Released item.",
            3: "Cannot be selected because the battle has ended.",
            4: (
                "Must be Observation.select.minCount <= len(select) <= "
                "Observation.select.maxCount."
            ),
            5: "Must be 0 <= select elements < len(Observation.select.option).",
            6: "Duplicate select elements.",
            30: "agent_ptr broken.",
        }
        if error in messages:
            raise ValueError(messages[error])
        raise RuntimeError(f"SearchStep failed with error {error}")

    def search_begin(
        self,
        obs_dict: dict,
        search_inputs: dict,
        manual_coin: bool = False,
    ) -> Any:
        self._assert_owner()
        observation = (
            self._api.to_observation_class(obs_dict)
            if isinstance(obs_dict, dict)
            else obs_dict
        )
        search_begin_input = getattr(observation, "search_begin_input", None)
        if search_begin_input is None:
            raise ValueError("Not agent observation.")
        state = getattr(observation, "current", None)
        if state is None:
            raise ValueError("SearchBegin requires a current state.")
        your_index = int(state.yourIndex)
        your_deck = [int(card) for card in search_inputs.get("your_deck", ())]
        your_prize = [int(card) for card in search_inputs.get("your_prize", ())]
        opponent_deck = [
            int(card) for card in search_inputs.get("opponent_deck", ())
        ]
        opponent_prize = [
            int(card) for card in search_inputs.get("opponent_prize", ())
        ]
        opponent_hand = [
            int(card) for card in search_inputs.get("opponent_hand", ())
        ]
        opponent_active = [
            int(card) for card in search_inputs.get("opponent_active", ())
        ]
        select = getattr(observation, "select", None)
        if select is not None and getattr(select, "deck", None) is not None:
            your_deck = []
        elif len(your_deck) < int(state.players[your_index].deckCount):
            raise ValueError("your_deck does not match the number of cards in your deck.")
        if len(your_prize) < len(state.players[your_index].prize):
            raise ValueError("your_prize does not match the number of cards in your prize.")
        if len(opponent_deck) < int(state.players[1 - your_index].deckCount):
            raise ValueError(
                "opponent_deck does not match the number of cards in opponent's deck."
            )
        if len(opponent_prize) < len(state.players[1 - your_index].prize):
            raise ValueError(
                "opponent_prize does not match the number of cards in opponent's prize."
            )
        if len(opponent_hand) < int(state.players[1 - your_index].handCount):
            raise ValueError(
                "opponent_hand does not match the number of cards in opponent's hand."
            )
        active = state.players[1 - your_index].active
        if len(active) > 0 and active[0] is None:
            if not opponent_active:
                raise ValueError("You need to predict the opponent's Active Pokémon.")
        else:
            opponent_active = []

        encoded = str(search_begin_input).encode("ascii")
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
        search_id = int(result.state.searchId)
        self._live_search_ids.add(search_id)
        return result.state

    def search_step(self, search_id: int, select: list[int]) -> Any:
        self._assert_owner()
        search_id = int(search_id)
        if search_id not in self._live_search_ids:
            raise ValueError(
                f"search id {search_id} is not owned by lane {self.lane_id}"
            )
        chosen = [int(index) for index in select]
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
        self._live_search_ids.add(int(result.state.searchId))
        return result.state

    def search_release(self, search_id: int) -> None:
        self._assert_owner()
        search_id = int(search_id)
        if search_id not in self._live_search_ids:
            raise ValueError(
                f"search id {search_id} is not owned by lane {self.lane_id}"
            )
        self._lib.SearchRelease(self._handle, search_id)
        self._live_search_ids.remove(search_id)

    def search_end(self) -> None:
        self._assert_owner()
        self._lib.SearchEnd(self._handle)
        self._live_search_ids.clear()
        self._ended_calls += 1


def legal_select_from_searchstate(search_state, rng: Optional[random.Random] = None) -> list[int]:
    """Sample a legal selection from a :class:`cg.api.SearchState` observation."""
    rng = rng or random
    sel = search_state.observation.select
    if sel is None:
        return []
    n = len(sel.option)
    lo, hi = sel.minCount, min(sel.maxCount, n)
    if hi <= 0:
        return []
    lo = max(0, min(lo, hi))
    k = rng.randint(lo, hi) if hi >= lo else hi
    return rng.sample(range(n), k) if k > 0 else []
