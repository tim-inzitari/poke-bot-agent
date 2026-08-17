"""Typed schedule and hard-clock accounting for the r214 BeliefMCTS mirror.

This module is deliberately narrow.  It owns the pair schedule and monotonic
planner clock for the owner-authorized, testing-only r214 experiment.  It does
*not* implement r207: every BeliefMCTS simulation root-samples a hidden
particle and samples explicit coin outcomes, so its receipts must use
``root_sampled_belief_mcts_non_r207_exact_chance``.

The live launcher uses this module in one fresh Python process per game.  It
must bind an engine-sealed first-player fact before starting either game in a
pair and fail a game closed if any measured MCTS-side action exceeds either
hard monotonic ceiling.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


R214_BELIEF_MCTS_MIRROR_SCHEMA = "poke_bot.alakazam_r214_belief_mcts_mirror/v1"
R214_BELIEF_MCTS_CANARY_SCHEMA = (
    "poke_bot.alakazam_r214_belief_mcts_seeded_canary/v1"
)
R214_EVALUATION_ID = "alakazam-r214-simple-belief-mcts-bo1000"
R214_CANARY_PAIR_COUNT = 1
R214_FULL_PAIR_COUNT = 500
R214_GAMES_PER_PAIR = 2
R214_MAX_TURN_PLANNER_SECONDS = 20.0
R214_MAX_ACTION_PLANNER_SECONDS = 5.0


class R214BeliefMCTSMirrorError(ValueError):
    """An r214 mirror request or timing fact is malformed."""


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
        raise R214BeliefMCTSMirrorError(f"{name} must be a sha256 digest")
    suffix = value[7:]
    if len(suffix) != 64 or any(char not in "0123456789abcdef" for char in suffix):
        raise R214BeliefMCTSMirrorError(
            f"{name} must be a lowercase sha256 digest"
        )
    return value


def _derive_nonzero_u32(payload: object) -> int:
    """Derive a stable nonzero uint32 without relying on process RNG state."""

    digest = hashlib.sha256(_canonical_bytes(payload)).digest()
    return (int.from_bytes(digest[:4], "big") % 0xFFFFFFFF) + 1


@dataclass(frozen=True, slots=True)
class R214TimingConfig:
    """The one easy-to-change typed timing surface for r214.

    The 4.5 s search watchdog leaves half a second inside every 5 s atomic
    action for transactional frozen-policy fallback, validation and receipt
    emission.  The hard reported ceilings remain exactly 20 s / 5 s.
    """

    max_planner_wall_seconds_per_actual_turn: float = R214_MAX_TURN_PLANNER_SECONDS
    max_planner_wall_seconds_before_each_atomic_action: float = (
        R214_MAX_ACTION_PLANNER_SECONDS
    )
    search_watchdog_seconds: float = 4.5
    direct_fallback_reserve_seconds: float = 0.5

    def __post_init__(self) -> None:
        if (
            float(self.max_planner_wall_seconds_per_actual_turn)
            != R214_MAX_TURN_PLANNER_SECONDS
        ):
            raise R214BeliefMCTSMirrorError("r214 turn budget must be exactly 20.0s")
        if (
            float(self.max_planner_wall_seconds_before_each_atomic_action)
            != R214_MAX_ACTION_PLANNER_SECONDS
        ):
            raise R214BeliefMCTSMirrorError("r214 action budget must be exactly 5.0s")
        if not (
            0.0
            < float(self.search_watchdog_seconds)
            < R214_MAX_ACTION_PLANNER_SECONDS
        ):
            raise R214BeliefMCTSMirrorError(
                "search watchdog must be strictly inside the 5.0s action cap"
            )
        if not (
            0.0
            < float(self.direct_fallback_reserve_seconds)
            <= R214_MAX_ACTION_PLANNER_SECONDS
            - float(self.search_watchdog_seconds)
        ):
            raise R214BeliefMCTSMirrorError(
                "fallback reserve must fit after the search watchdog"
            )

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema": R214_BELIEF_MCTS_MIRROR_SCHEMA,
                "timing": asdict(self),
            }
        )


@dataclass(frozen=True, slots=True)
class R214GameSpec:
    """One game of a common-seed, seat-swapped r214 pair."""

    pair_index: int
    pair_id: str
    pair_nonce_sha256: str
    game_index: int
    game_nonce_sha256: str
    engine_seed_u32: int
    deck_order_seed_u32: int
    mcts_rng_seed_u32: int
    direct_policy_rng_seed_u32: int
    mcts_seat: int
    direct_seat: int

    def __post_init__(self) -> None:
        if type(self.pair_index) is not int or self.pair_index < 0:
            raise R214BeliefMCTSMirrorError("pair_index must be a nonnegative integer")
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise R214BeliefMCTSMirrorError("pair_id must be nonempty")
        require_sha256(self.pair_nonce_sha256, name="pair_nonce_sha256")
        if self.game_index not in {0, 1}:
            raise R214BeliefMCTSMirrorError("game_index must be 0 or 1")
        require_sha256(self.game_nonce_sha256, name="game_nonce_sha256")
        for name in (
            "engine_seed_u32",
            "deck_order_seed_u32",
            "mcts_rng_seed_u32",
            "direct_policy_rng_seed_u32",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= 0xFFFFFFFF:
                raise R214BeliefMCTSMirrorError(f"{name} must be a nonzero uint32")
        if self.engine_seed_u32 != self.deck_order_seed_u32:
            raise R214BeliefMCTSMirrorError(
                "r214 binds one common seeded-engine/deck-order stream"
            )
        if self.mcts_seat not in {0, 1} or self.direct_seat != 1 - self.mcts_seat:
            raise R214BeliefMCTSMirrorError("MCTS/direct seats must be exact opposites")
        expected_nonce = canonical_sha256(
            {
                "schema": R214_BELIEF_MCTS_CANARY_SCHEMA,
                "pair_nonce_sha256": self.pair_nonce_sha256,
                "game_index": self.game_index,
            }
        )
        if self.game_nonce_sha256 != expected_nonce:
            raise R214BeliefMCTSMirrorError("game nonce does not bind pair nonce/index")

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class R214PairFirstPlayerSeal:
    """A pre-game engine receipt that seals actual first/second within a pair."""

    pair_index: int
    pair_id: str
    pair_nonce_sha256: str
    engine_seed_u32: int
    deck_order_seed_u32: int
    first_player_seat: int
    post_turn_order_observation_sha256: str

    def __post_init__(self) -> None:
        if type(self.pair_index) is not int or self.pair_index < 0:
            raise R214BeliefMCTSMirrorError("sealed pair_index must be nonnegative")
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise R214BeliefMCTSMirrorError("sealed pair_id must be nonempty")
        require_sha256(self.pair_nonce_sha256, name="sealed pair_nonce_sha256")
        require_sha256(
            self.post_turn_order_observation_sha256,
            name="post_turn_order_observation_sha256",
        )
        if self.first_player_seat not in {0, 1}:
            raise R214BeliefMCTSMirrorError("first_player_seat must be 0 or 1")
        for name in ("engine_seed_u32", "deck_order_seed_u32"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= 0xFFFFFFFF:
                raise R214BeliefMCTSMirrorError(f"sealed {name} must be a nonzero uint32")
        if self.engine_seed_u32 != self.deck_order_seed_u32:
            raise R214BeliefMCTSMirrorError("first-player seal must bind common seed")

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema": R214_BELIEF_MCTS_CANARY_SCHEMA,
                "kind": "pair_first_player_seal",
                "payload": asdict(self),
            }
        )

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["identity_sha256"] = self.identity_sha256
        return payload


def build_r214_seeded_schedule(
    seed_identity_sha256: str,
    *,
    pair_count: int,
    evaluation_id: str = R214_EVALUATION_ID,
) -> tuple[R214GameSpec, ...]:
    """Build common-random-number pairs with MCTS seats swapped inside each pair.

    Engine/deck-order material is identical within a pair.  MCTS particle and
    chance sampling is intentionally separate per game and cannot be claimed
    to preserve the post-divergence engine trajectory.
    """

    require_sha256(seed_identity_sha256, name="seed_identity_sha256")
    if type(pair_count) is not int or pair_count < 1:
        raise R214BeliefMCTSMirrorError("pair_count must be a positive integer")
    if not isinstance(evaluation_id, str) or not evaluation_id:
        raise R214BeliefMCTSMirrorError("evaluation_id must be nonempty")
    specs: list[R214GameSpec] = []
    for pair_index in range(pair_count):
        pair_nonce = canonical_sha256(
            {
                "schema": R214_BELIEF_MCTS_CANARY_SCHEMA,
                "evaluation_id": evaluation_id,
                "seed_identity_sha256": seed_identity_sha256,
                "pair_index": pair_index,
            }
        )
        pair_id = f"r214-pair-{pair_index:06d}-{pair_nonce[7:19]}"
        engine_seed = _derive_nonzero_u32(
            {
                "schema": R214_BELIEF_MCTS_CANARY_SCHEMA,
                "pair_nonce_sha256": pair_nonce,
                "stream": "seeded_engine_and_deck_order",
            }
        )
        for game_index in (0, 1):
            game_nonce = canonical_sha256(
                {
                    "schema": R214_BELIEF_MCTS_CANARY_SCHEMA,
                    "pair_nonce_sha256": pair_nonce,
                    "game_index": game_index,
                }
            )
            specs.append(
                R214GameSpec(
                    pair_index=pair_index,
                    pair_id=pair_id,
                    pair_nonce_sha256=pair_nonce,
                    game_index=game_index,
                    game_nonce_sha256=game_nonce,
                    engine_seed_u32=engine_seed,
                    deck_order_seed_u32=engine_seed,
                    mcts_rng_seed_u32=_derive_nonzero_u32(
                        {
                            "schema": R214_BELIEF_MCTS_CANARY_SCHEMA,
                            "game_nonce_sha256": game_nonce,
                            "stream": "belief_particle_and_explicit_coin_sampling",
                        }
                    ),
                    direct_policy_rng_seed_u32=_derive_nonzero_u32(
                        {
                            "schema": R214_BELIEF_MCTS_CANARY_SCHEMA,
                            "game_nonce_sha256": game_nonce,
                            "stream": "frozen_direct_policy_fail_closed_rng",
                        }
                    ),
                    mcts_seat=game_index,
                    direct_seat=1 - game_index,
                )
            )
    return tuple(specs)


def build_r214_canary_schedule(seed_identity_sha256: str) -> tuple[R214GameSpec, ...]:
    return build_r214_seeded_schedule(
        seed_identity_sha256,
        pair_count=R214_CANARY_PAIR_COUNT,
        evaluation_id=R214_EVALUATION_ID,
    )


def validate_pair_first_player_seal(
    games: tuple[R214GameSpec, ...] | list[R214GameSpec],
    seal: R214PairFirstPlayerSeal,
) -> None:
    """Require one exact seat swap and an engine-bound first-player seal."""

    if len(games) != 2:
        raise R214BeliefMCTSMirrorError("a r214 pair must contain exactly two games")
    ordered = sorted(games, key=lambda game: game.game_index)
    first, second = ordered
    if first.game_index != 0 or second.game_index != 1:
        raise R214BeliefMCTSMirrorError("pair games must be indexed 0 then 1")
    for game in ordered:
        if (
            game.pair_index != seal.pair_index
            or game.pair_id != seal.pair_id
            or game.pair_nonce_sha256 != seal.pair_nonce_sha256
            or game.engine_seed_u32 != seal.engine_seed_u32
            or game.deck_order_seed_u32 != seal.deck_order_seed_u32
        ):
            raise R214BeliefMCTSMirrorError("first-player seal does not bind game pair")
    if first.mcts_seat == second.mcts_seat:
        raise R214BeliefMCTSMirrorError("MCTS must swap seats within each pair")
    actual_first = [game.mcts_seat == seal.first_player_seat for game in ordered]
    if actual_first.count(True) != 1 or actual_first.count(False) != 1:
        raise R214BeliefMCTSMirrorError(
            "seat swap must yield exactly one MCTS-first and one MCTS-second game"
        )


@dataclass(frozen=True, slots=True)
class R214PlannerTurnIdentity:
    """Public engine identity for one actor's actual turn and atomic decision."""

    seat: int
    turn: int
    turn_action_count: int

    @property
    def planner_turn_key(self) -> tuple[int, int]:
        return (self.seat, self.turn)

    def as_payload(self) -> dict[str, int]:
        return {
            "seat": self.seat,
            "actual_turn_id": self.turn,
            "turn_action_count": self.turn_action_count,
        }


def turn_identity_from_observation(observation: Mapping[str, Any]) -> R214PlannerTurnIdentity:
    """Read the native libcg turn and within-turn atomic-action counters verbatim."""

    current = observation.get("current")
    if not isinstance(current, Mapping):
        raise R214BeliefMCTSMirrorError("observation.current must be an object")
    seat = current.get("yourIndex")
    turn = current.get("turn")
    action_count = current.get("turnActionCount")
    if type(seat) is not int or seat not in {0, 1}:
        raise R214BeliefMCTSMirrorError("current.yourIndex must be seat 0 or 1")
    if type(turn) is not int or turn < 0:
        raise R214BeliefMCTSMirrorError("current.turn must be a nonnegative integer")
    if type(action_count) is not int or action_count < 0:
        raise R214BeliefMCTSMirrorError(
            "current.turnActionCount must be a nonnegative integer"
        )
    return R214PlannerTurnIdentity(seat, turn, action_count)


@dataclass(frozen=True, slots=True)
class R214ActionBudget:
    planner_turn: R214PlannerTurnIdentity
    action_index_in_turn: int
    planner_wall_seconds_used_before: float
    planner_wall_seconds_remaining_before: float
    hard_action_wall_seconds: float
    allowed_search_wall_seconds: float
    search_allowed: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class R214ActionTiming:
    planner_turn: R214PlannerTurnIdentity
    action_index_in_turn: int
    search_attempted: bool
    planner_wall_seconds: float
    allowed_search_wall_seconds: float
    hard_action_wall_seconds: float
    direct_policy_fallback_used: bool
    reason: str | None
    within_hard_budget: bool

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        planner_turn = payload.pop("planner_turn")
        payload.update(
            {
                "planner_turn_id": [planner_turn["seat"], planner_turn["turn"]],
                "seat": planner_turn["seat"],
                "actual_turn_id": planner_turn["turn"],
                "turn_action_count": planner_turn["turn_action_count"],
            }
        )
        return payload


class R214TurnBudgetLedger:
    """Charge all MCTS-side planner work against actual public engine turns."""

    def __init__(self, config: R214TimingConfig | None = None) -> None:
        self.config = config or R214TimingConfig()
        self._turn_key: tuple[int, int] | None = None
        self._planner_wall_seconds_used = 0.0
        self._action_count = 0
        self._last_turn_action_count: int | None = None

    @property
    def active_turn_key(self) -> tuple[int, int] | None:
        return self._turn_key

    @property
    def planner_wall_seconds_used(self) -> float:
        return self._planner_wall_seconds_used

    def begin_action(self, planner_turn: R214PlannerTurnIdentity) -> R214ActionBudget:
        key = planner_turn.planner_turn_key
        if self._turn_key != key:
            self._turn_key = key
            self._planner_wall_seconds_used = 0.0
            self._action_count = 0
            self._last_turn_action_count = None
        elif (
            self._last_turn_action_count is not None
            and planner_turn.turn_action_count <= self._last_turn_action_count
        ):
            raise R214BeliefMCTSMirrorError(
                "engine turnActionCount did not advance within an actual turn"
            )
        self._last_turn_action_count = planner_turn.turn_action_count
        remaining = max(
            0.0,
            self.config.max_planner_wall_seconds_per_actual_turn
            - self._planner_wall_seconds_used,
        )
        hard_action = min(
            self.config.max_planner_wall_seconds_before_each_atomic_action,
            remaining,
        )
        allowed_search = min(
            self.config.search_watchdog_seconds,
            max(0.0, hard_action - self.config.direct_fallback_reserve_seconds),
        )
        search_allowed = allowed_search > 0.0
        reason = None if search_allowed else "turn_planner_budget_exhausted"
        return R214ActionBudget(
            planner_turn=planner_turn,
            action_index_in_turn=self._action_count,
            planner_wall_seconds_used_before=self._planner_wall_seconds_used,
            planner_wall_seconds_remaining_before=remaining,
            hard_action_wall_seconds=hard_action,
            allowed_search_wall_seconds=allowed_search,
            search_allowed=search_allowed,
            reason=reason,
        )

    def record_action(
        self,
        budget: R214ActionBudget,
        *,
        planner_wall_seconds: float,
        search_attempted: bool,
        direct_policy_fallback_used: bool,
        reason: str | None = None,
    ) -> R214ActionTiming:
        if budget.planner_turn.planner_turn_key != self._turn_key:
            raise R214BeliefMCTSMirrorError("budget belongs to an inactive turn")
        if type(planner_wall_seconds) not in {int, float}:
            raise R214BeliefMCTSMirrorError("planner wall seconds must be numeric")
        value = float(planner_wall_seconds)
        if value < 0.0 or value != value or value == float("inf"):
            raise R214BeliefMCTSMirrorError("planner wall seconds must be finite")
        self._planner_wall_seconds_used += value
        self._action_count += 1
        within = (
            value <= budget.hard_action_wall_seconds
            and value <= self.config.max_planner_wall_seconds_before_each_atomic_action
            and self._planner_wall_seconds_used
            <= self.config.max_planner_wall_seconds_per_actual_turn
        )
        final_reason = reason
        if not within:
            final_reason = (
                "atomic_action_budget_breach"
                if value > budget.hard_action_wall_seconds
                or value
                > self.config.max_planner_wall_seconds_before_each_atomic_action
                else "actual_turn_budget_breach"
            )
        return R214ActionTiming(
            planner_turn=budget.planner_turn,
            action_index_in_turn=budget.action_index_in_turn,
            search_attempted=bool(search_attempted),
            planner_wall_seconds=value,
            allowed_search_wall_seconds=budget.allowed_search_wall_seconds,
            hard_action_wall_seconds=budget.hard_action_wall_seconds,
            direct_policy_fallback_used=bool(direct_policy_fallback_used),
            reason=final_reason,
            within_hard_budget=within,
        )


def build_r214_canary_plan(seed_identity_sha256: str) -> dict[str, Any]:
    """Serialize the two-game local/remote canary without overclaiming r207."""

    schedule = build_r214_canary_schedule(seed_identity_sha256)
    timing = R214TimingConfig()
    return {
        "schema": R214_BELIEF_MCTS_CANARY_SCHEMA,
        "evaluation_id": R214_EVALUATION_ID,
        "pair_count": 1,
        "game_count": 2,
        "fresh_process_required_per_game": True,
        "seed_identity_sha256": seed_identity_sha256,
        "pairing": {
            "common_seeded_engine_and_deck_order_within_pair": True,
            "first_player_is_engine_sealed_before_game_children": True,
            "same_engine_seed_u32_within_pair": True,
            "same_deck_order_seed_u32_within_pair": True,
            "mcts_seat_0_games": 1,
            "mcts_seat_1_games": 1,
            "mcts_actual_first_games_after_pair_seal": 1,
            "mcts_actual_second_games_after_pair_seal": 1,
            "post_action_divergence_does_not_claim_identical_rng_trajectory": True,
        },
        "strategy": {
            "name": "root_sampled_belief_mcts_non_r207_exact_chance",
            "r207_exact_chance_claim": False,
            "hidden_particle_sampling": True,
            "explicit_coin_sampling": True,
            "frozen_r195_model_and_matchup_adapter_required": True,
            "guide2vec_enabled": False,
            "guide_logit_transform_enabled": False,
            "guide_linear_transform_enabled": False,
            "legacy_rtp_enabled": False,
        },
        "timing": {
            **asdict(timing),
            "timing_config_sha256": timing.identity_sha256,
            "clock": "monotonic_wall_clock",
            "breach_behavior": "invalid_timeout_not_counted_as_game_outcome",
            "deadline_or_insufficient_trusted_search_behavior": (
                "execute_exact_frozen_r195_direct_policy_action"
            ),
        },
        "games": [spec.as_payload() for spec in schedule],
    }
