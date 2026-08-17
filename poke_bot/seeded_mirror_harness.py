"""Reusable, strategy-neutral primitives for a seeded seat-swapped mirror.

This module intentionally owns no MCTS policy, simulation target, cache, or
clock semantics.  It gives a later evaluator one safe way to create matched
seed pairs, seal the engine's actual first-player result, and identify native
turn/atomic-action counters.  A strategy-specific runner supplies its own
time budget and action-selection implementation.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

SEEDED_MIRROR_HARNESS_SCHEMA = "poke_bot.seeded_mirror_harness/v1"


class SeededMirrorHarnessError(ValueError):
    """A paired engine schedule or sealed public fact is malformed."""


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_sha256(payload: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise SeededMirrorHarnessError(f"{name} must be a sha256 digest")
    suffix = value[7:]
    if len(suffix) != 64 or any(character not in "0123456789abcdef" for character in suffix):
        raise SeededMirrorHarnessError(f"{name} must be lowercase sha256")
    return value


def _derive_nonzero_u32(payload: object) -> int:
    digest = hashlib.sha256(_canonical_bytes(payload)).digest()
    # Zero is intentionally unavailable so an omitted seed cannot silently
    # become a valid engine/deck-order stream.
    return (int.from_bytes(digest[:4], "big") % 0xFFFFFFFF) + 1


@dataclass(frozen=True, slots=True)
class SeededMirrorGameSpec:
    """One game in a pair sharing engine/deck-order material."""

    evaluation_id: str
    pair_index: int
    pair_id: str
    pair_nonce_sha256: str
    game_index: int
    game_nonce_sha256: str
    engine_seed_u32: int
    deck_order_seed_u32: int
    experimental_rng_seed_u32: int
    control_rng_seed_u32: int
    experimental_seat: int
    control_seat: int

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_id, str) or not self.evaluation_id:
            raise SeededMirrorHarnessError("evaluation_id must be nonempty")
        if type(self.pair_index) is not int or self.pair_index < 0:
            raise SeededMirrorHarnessError("pair_index must be nonnegative")
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise SeededMirrorHarnessError("pair_id must be nonempty")
        require_sha256(self.pair_nonce_sha256, name="pair_nonce_sha256")
        if self.game_index not in {0, 1}:
            raise SeededMirrorHarnessError("game_index must be 0 or 1")
        require_sha256(self.game_nonce_sha256, name="game_nonce_sha256")
        for field_name in (
            "engine_seed_u32",
            "deck_order_seed_u32",
            "experimental_rng_seed_u32",
            "control_rng_seed_u32",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or not 0 < value <= 0xFFFFFFFF:
                raise SeededMirrorHarnessError(f"{field_name} must be a nonzero uint32")
        if self.engine_seed_u32 != self.deck_order_seed_u32:
            raise SeededMirrorHarnessError("engine and deck-order seed must be bound")
        if self.experimental_seat not in {0, 1} or self.control_seat != 1 - self.experimental_seat:
            raise SeededMirrorHarnessError("experimental/control seats must be opposites")
        expected_nonce = canonical_sha256(
            {
                "schema": SEEDED_MIRROR_HARNESS_SCHEMA,
                "pair_nonce_sha256": self.pair_nonce_sha256,
                "game_index": self.game_index,
            }
        )
        if self.game_nonce_sha256 != expected_nonce:
            raise SeededMirrorHarnessError("game nonce does not bind pair/index")

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


def build_seeded_seat_swapped_schedule(
    *,
    evaluation_id: str,
    seed_identity_sha256: str,
    pair_count: int,
) -> tuple[SeededMirrorGameSpec, ...]:
    """Create deterministic two-game pairs with common engine/deck-order RNG.

    The evaluator must not claim the entire post-action random trajectory is
    identical: strategy actions can diverge.  The shared seed is pairing
    material only.  Per-arm decision RNG is separately derived and recorded.
    """

    if not isinstance(evaluation_id, str) or not evaluation_id:
        raise SeededMirrorHarnessError("evaluation_id must be nonempty")
    require_sha256(seed_identity_sha256, name="seed_identity_sha256")
    if type(pair_count) is not int or pair_count < 1:
        raise SeededMirrorHarnessError("pair_count must be positive")
    games: list[SeededMirrorGameSpec] = []
    for pair_index in range(pair_count):
        pair_nonce = canonical_sha256(
            {
                "schema": SEEDED_MIRROR_HARNESS_SCHEMA,
                "evaluation_id": evaluation_id,
                "seed_identity_sha256": seed_identity_sha256,
                "pair_index": pair_index,
            }
        )
        pair_id = f"pair-{pair_index:06d}-{pair_nonce[7:19]}"
        common_seed = _derive_nonzero_u32(
            {
                "schema": SEEDED_MIRROR_HARNESS_SCHEMA,
                "pair_nonce_sha256": pair_nonce,
                "stream": "engine_and_deck_order",
            }
        )
        for game_index in (0, 1):
            game_nonce = canonical_sha256(
                {
                    "schema": SEEDED_MIRROR_HARNESS_SCHEMA,
                    "pair_nonce_sha256": pair_nonce,
                    "game_index": game_index,
                }
            )
            games.append(
                SeededMirrorGameSpec(
                    evaluation_id=evaluation_id,
                    pair_index=pair_index,
                    pair_id=pair_id,
                    pair_nonce_sha256=pair_nonce,
                    game_index=game_index,
                    game_nonce_sha256=game_nonce,
                    engine_seed_u32=common_seed,
                    deck_order_seed_u32=common_seed,
                    experimental_rng_seed_u32=_derive_nonzero_u32(
                        {
                            "schema": SEEDED_MIRROR_HARNESS_SCHEMA,
                            "game_nonce_sha256": game_nonce,
                            "stream": "experimental_policy_rng",
                        }
                    ),
                    control_rng_seed_u32=_derive_nonzero_u32(
                        {
                            "schema": SEEDED_MIRROR_HARNESS_SCHEMA,
                            "game_nonce_sha256": game_nonce,
                            "stream": "control_policy_rng",
                        }
                    ),
                    experimental_seat=game_index,
                    control_seat=1 - game_index,
                )
            )
    return tuple(games)


@dataclass(frozen=True, slots=True)
class PairFirstPlayerSeal:
    """Engine-observed first-player fact, sealed before either game child runs."""

    evaluation_id: str
    pair_index: int
    pair_id: str
    pair_nonce_sha256: str
    engine_seed_u32: int
    deck_order_seed_u32: int
    first_player_seat: int
    post_turn_order_observation_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_id, str) or not self.evaluation_id:
            raise SeededMirrorHarnessError("seal evaluation_id must be nonempty")
        if type(self.pair_index) is not int or self.pair_index < 0:
            raise SeededMirrorHarnessError("seal pair_index must be nonnegative")
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise SeededMirrorHarnessError("seal pair_id must be nonempty")
        require_sha256(self.pair_nonce_sha256, name="seal pair_nonce_sha256")
        require_sha256(
            self.post_turn_order_observation_sha256,
            name="post_turn_order_observation_sha256",
        )
        if self.first_player_seat not in {0, 1}:
            raise SeededMirrorHarnessError("first_player_seat must be 0 or 1")
        for field_name in ("engine_seed_u32", "deck_order_seed_u32"):
            value = getattr(self, field_name)
            if type(value) is not int or not 0 < value <= 0xFFFFFFFF:
                raise SeededMirrorHarnessError(f"seal {field_name} must be nonzero uint32")
        if self.engine_seed_u32 != self.deck_order_seed_u32:
            raise SeededMirrorHarnessError("seal engine/deck-order seed mismatch")

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema": SEEDED_MIRROR_HARNESS_SCHEMA,
                "kind": "pair_first_player_seal",
                "payload": asdict(self),
            }
        )

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["identity_sha256"] = self.identity_sha256
        return payload


def validate_pair_first_player_seal(
    games: Sequence[SeededMirrorGameSpec], seal: PairFirstPlayerSeal
) -> None:
    """Check paired common RNG, exact seat swap, and actual-first/second balance."""

    if len(games) != 2:
        raise SeededMirrorHarnessError("a paired mirror cell needs exactly two games")
    first, second = sorted(games, key=lambda game: game.game_index)
    if (first.game_index, second.game_index) != (0, 1):
        raise SeededMirrorHarnessError("pair must contain game index 0 and 1 exactly once")
    for game in (first, second):
        if (
            game.evaluation_id != seal.evaluation_id
            or game.pair_index != seal.pair_index
            or game.pair_id != seal.pair_id
            or game.pair_nonce_sha256 != seal.pair_nonce_sha256
            or game.engine_seed_u32 != seal.engine_seed_u32
            or game.deck_order_seed_u32 != seal.deck_order_seed_u32
        ):
            raise SeededMirrorHarnessError("first-player seal does not bind this exact pair")
    if first.experimental_seat == second.experimental_seat:
        raise SeededMirrorHarnessError("experimental arm did not swap seats")
    first_orders = [
        game.experimental_seat == seal.first_player_seat for game in (first, second)
    ]
    if first_orders.count(True) != 1 or first_orders.count(False) != 1:
        raise SeededMirrorHarnessError(
            "a seat-swapped pair must give experimental arm first and second once"
        )


@dataclass(frozen=True, slots=True)
class EngineTurnIdentity:
    """Native public turn and within-turn action counters from a libcg observation."""

    seat: int
    actual_turn_id: int
    turn_action_count: int

    @property
    def turn_key(self) -> tuple[int, int]:
        return (self.seat, self.actual_turn_id)


def engine_turn_identity_from_observation(observation: Mapping[str, Any]) -> EngineTurnIdentity:
    current = observation.get("current")
    if not isinstance(current, Mapping):
        raise SeededMirrorHarnessError("observation.current must be an object")
    seat = current.get("yourIndex")
    turn = current.get("turn")
    action_count = current.get("turnActionCount")
    if type(seat) is not int or seat not in {0, 1}:
        raise SeededMirrorHarnessError("current.yourIndex must be seat 0 or 1")
    if type(turn) is not int or turn < 0:
        raise SeededMirrorHarnessError("current.turn must be a nonnegative integer")
    if type(action_count) is not int or action_count < 0:
        raise SeededMirrorHarnessError(
            "current.turnActionCount must be a nonnegative integer"
        )
    return EngineTurnIdentity(seat, turn, action_count)


def configure_battle_start_seeded(lib: Any, start_data_type: Any) -> Any:
    """Bind ctypes ABI before a runner calls a private ``BattleStartSeeded``.

    ctypes otherwise assumes an integer return type, which corrupts the native
    ``StartData`` result.  Callers still need to invoke it and reject a missing
    symbol or invalid native start result themselves.
    """

    try:
        seeded = lib.BattleStartSeeded
    except AttributeError as exc:
        raise SeededMirrorHarnessError("libcg lacks BattleStartSeeded") from exc
    seeded.restype = start_data_type
    seeded.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint32]
    return seeded
