from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from game_tracker import DERIVED_INFERENCE_DIM, GameEventTracker
except ImportError:
    from poke_agent.game_tracker import DERIVED_INFERENCE_DIM, GameEventTracker

try:
    from rewards import (
        DEFAULT_PRIZE_COUNT,
        blended_value_target,
        episode_outcome_returns,
        prize_count,
    )
except ImportError:
    try:
        from poke_agent.rewards import (
            DEFAULT_PRIZE_COUNT,
            blended_value_target,
            episode_outcome_returns,
            prize_count,
        )
    except ImportError:
        DEFAULT_PRIZE_COUNT = 6

        def prize_count(obs: dict[str, Any] | None, player_index: int) -> int:
            if obs is None:
                return DEFAULT_PRIZE_COUNT
            players = (obs.get("current") or {}).get("players") or [{}, {}]
            if player_index >= len(players):
                return DEFAULT_PRIZE_COUNT
            player = players[player_index] or {}
            prize = player.get("prize")
            if isinstance(prize, list):
                return len(prize)
            raw = player.get("prizeCount")
            if raw is not None:
                return int(raw)
            return DEFAULT_PRIZE_COUNT

        def episode_outcome_returns(*args, **kwargs):
            raise NotImplementedError

        def blended_value_target(*args, **kwargs):
            raise NotImplementedError

FEATURE_SCHEMA_VERSION = 2
CARD_ID_OOV = 0
MAX_BENCH_SLOTS = 8
SELF_HAND_CARD_SLOTS = 10
NUM_ENERGY_TYPES = 12

POKEMON_SLOT_DIM = (
    1  # hp_ratio
    + 1  # hp_scaled
    + 1  # energy_count
    + NUM_ENERGY_TYPES  # energy_by_type
    + 1  # tool_present
    + 3  # evolution stage indicators
    + 1  # appearThisTurn
    + 5  # status flags
    + 1  # slot_present
)

GLOBAL_FEATURE_DIM = 18
SELECT_FEATURE_DIM = 6

POKEMON_SLOT_COUNT = 2 + 2 * MAX_BENCH_SLOTS  # self/opp active + benches
CARD_ID_SLOT_COUNT = (
    1  # self active
    + MAX_BENCH_SLOTS  # self bench
    + 1  # opp active
    + MAX_BENCH_SLOTS  # opp bench
    + SELF_HAND_CARD_SLOTS
    + 1  # stadium
    + 1  # effect
)

STRUCTURED_FEATURE_DIM = (
    POKEMON_SLOT_COUNT * POKEMON_SLOT_DIM
    + GLOBAL_FEATURE_DIM
    + SELECT_FEATURE_DIM
    + DERIVED_INFERENCE_DIM
)

# Backward-compatible aliases used by config/tests.
COARSE_BASE_DIM = GLOBAL_FEATURE_DIM
COARSE_FEATURE_DIM = STRUCTURED_FEATURE_DIM
FEATURE_DIM = STRUCTURED_FEATURE_DIM  # without residual hash; use total_feature_dim()

BASE_FEATURE_NAMES = tuple(
    f"global_{index}" for index in range(GLOBAL_FEATURE_DIM)
) + tuple(f"select_{index}" for index in range(SELECT_FEATURE_DIM))


@dataclass(frozen=True)
class FeatureSpec:
    schema_version: int
    structured_dim: int
    residual_hash_dim: int
    total_dim: int
    pokemon_slot_dim: int
    pokemon_slot_count: int
    card_id_slot_count: int
    derived_dim: int


def feature_spec(*, state_hash_dim: int) -> FeatureSpec:
    total = STRUCTURED_FEATURE_DIM + int(state_hash_dim)
    return FeatureSpec(
        schema_version=FEATURE_SCHEMA_VERSION,
        structured_dim=STRUCTURED_FEATURE_DIM,
        residual_hash_dim=int(state_hash_dim),
        total_dim=total,
        pokemon_slot_dim=POKEMON_SLOT_DIM,
        pokemon_slot_count=POKEMON_SLOT_COUNT,
        card_id_slot_count=CARD_ID_SLOT_COUNT,
        derived_dim=DERIVED_INFERENCE_DIM,
    )


def total_feature_dim(*, state_hash_dim: int) -> int:
    return feature_spec(state_hash_dim=state_hash_dim).total_dim


def card_id_index(card_id: int | None, *, vocab_size: int) -> int:
    if card_id is None or int(card_id) <= 0:
        return CARD_ID_OOV
    cid = int(card_id)
    if cid >= vocab_size:
        return (cid % max(1, vocab_size - 1)) + 1
    return cid


def _is_card_id_key(key: str) -> bool:
    lowered = key.lower()
    return lowered.endswith(".id") or lowered.endswith("id") or lowered.endswith("cardid")


def hashed_deck_composition(deck: list[int] | tuple[int, ...], *, state_hash_dim: int) -> np.ndarray:
    """Hash **your** full 60-card deck list (private info you know; not opponent deck)."""
    vec = np.zeros(state_hash_dim, dtype=np.float32)
    if not deck:
        return vec
    counts: dict[int, int] = {}
    for card_id in deck:
        counts[int(card_id)] = counts.get(int(card_id), 0) + 1
    for card_id, count in sorted(counts.items()):
        token = f"self_deck.card={card_id}"
        vec[stable_hash_index(token, state_hash_dim)] += count / 60.0
    return vec


def seat_deck_from_row(row: dict[str, Any]) -> list[int] | None:
    """Return the seated player's own deck list (never the opponent's hidden deck)."""
    obs = row.get("observation") or {}
    your_index = int((obs.get("current") or {}).get("yourIndex", row.get("player", 0)))
    key = "deck0_cards" if your_index == 0 else "deck1_cards"
    deck = row.get(key)
    if isinstance(deck, list) and deck:
        return [int(card_id) for card_id in deck]
    return None


def going_first_feature(obs: dict[str, Any]) -> float:
    """1.0 if this seat goes first, 0.0 if second, 0.5 if undetermined."""
    current = obs.get("current") or {}
    first_player = int(current.get("firstPlayer", -1))
    your_index = int(current.get("yourIndex", 0))
    if first_player < 0:
        return 0.5
    return 1.0 if your_index == first_player else 0.0


def stable_hash_index(text: str, size: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % size


def iter_state_items(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_state_items(value[key], child_prefix)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            child_prefix = f"{prefix}[{idx}]"
            yield from iter_state_items(item, child_prefix)
    else:
        yield prefix, value


def residual_state_hash(
    observation: Any,
    action: Any = None,
    *,
    state_hash_dim: int,
    our_deck: list[int] | None = None,
) -> np.ndarray:
    vec = np.zeros(state_hash_dim, dtype=np.float32)
    payloads = [("obs", observation, 1.0), ("action", action, 0.5)]
    for label, payload, weight in payloads:
        if payload is None:
            continue
        for key, value in iter_state_items(payload):
            if isinstance(value, bool):
                token = f"{label}.{key}:bool"
                amount = 1.0 if value else -1.0
            elif isinstance(value, (int, float)) and _is_card_id_key(key):
                token = f"{label}.{key}={int(value)}"
                amount = 1.0
            elif isinstance(value, (int, float)):
                token = f"{label}.{key}:num"
                amount = float(np.tanh(float(value) / 100.0))
            elif value is None:
                token = f"{label}.{key}:none"
                amount = 1.0
            else:
                token = f"{label}.{key}={value}"
                amount = 1.0
            vec[stable_hash_index(token, state_hash_dim)] += weight * amount
    if our_deck is not None:
        vec = vec + hashed_deck_composition(our_deck, state_hash_dim=state_hash_dim)
    return vec


def _energy_type_counts(pokemon: dict[str, Any] | None) -> list[float]:
    counts = [0.0] * NUM_ENERGY_TYPES
    if not pokemon:
        return counts
    energies = pokemon.get("energies") or []
    for energy in energies:
        idx = int(energy)
        if 0 <= idx < NUM_ENERGY_TYPES:
            counts[idx] += 1.0
    return counts


def _pokemon_slot_features(
    pokemon: dict[str, Any] | None,
    *,
    status: dict[str, bool] | None = None,
) -> list[float]:
    if not pokemon:
        return [0.0] * POKEMON_SLOT_DIM

    max_hp = max(1, int(pokemon.get("maxHp", pokemon.get("hp", 1)) or 1))
    hp = int(pokemon.get("hp", 0) or 0)
    tools = pokemon.get("tools") or []
    features: list[float] = [
        float(hp / max_hp),
        float(np.tanh(hp / 200.0)),
        float(len(pokemon.get("energies") or [])),
        *_energy_type_counts(pokemon),
        1.0 if tools else 0.0,
        0.0,
        0.0,
        0.0,
        1.0 if pokemon.get("appearThisTurn") else 0.0,
    ]
    status = status or {}
    features.extend(
        [
            1.0 if status.get("poisoned") else 0.0,
            1.0 if status.get("burned") else 0.0,
            1.0 if status.get("asleep") else 0.0,
            1.0 if status.get("paralyzed") else 0.0,
            1.0 if status.get("confused") else 0.0,
            1.0,
        ]
    )
    return features


def _player_status(player: dict[str, Any]) -> dict[str, bool]:
    return {
        "poisoned": bool(player.get("poisoned")),
        "burned": bool(player.get("burned")),
        "asleep": bool(player.get("asleep")),
        "paralyzed": bool(player.get("paralyzed")),
        "confused": bool(player.get("confused")),
    }


def _active_pokemon(player: dict[str, Any]) -> dict[str, Any] | None:
    active = player.get("active") or []
    if not active:
        return None
    first = active[0]
    return first if isinstance(first, dict) else None


def _bench_pokemon(player: dict[str, Any], *, max_slots: int = MAX_BENCH_SLOTS) -> list[dict[str, Any] | None]:
    bench = player.get("bench") or []
    slots: list[dict[str, Any] | None] = []
    for index in range(max_slots):
        if index < len(bench) and isinstance(bench[index], dict):
            slots.append(bench[index])
        else:
            slots.append(None)
    return slots


def _card_id_from_card(card: dict[str, Any] | None) -> int | None:
    if not card:
        return None
    raw = card.get("id")
    if raw is None:
        raw = card.get("cardId")
    return int(raw) if raw is not None else None


def _global_features(obs: dict[str, Any], *, your_index: int, opp_index: int) -> list[float]:
    current = obs.get("current") or {}
    players = current.get("players") or [{}, {}]
    self_player = players[your_index] if your_index < len(players) else {}
    opp_player = players[opp_index] if opp_index < len(players) else {}
    bench_max_self = max(1, int(self_player.get("benchMax", MAX_BENCH_SLOTS) or MAX_BENCH_SLOTS))
    bench_max_opp = max(1, int(opp_player.get("benchMax", MAX_BENCH_SLOTS) or MAX_BENCH_SLOTS))
    self_discard = len(self_player.get("discard") or [])
    opp_discard = len(opp_player.get("discard") or [])
    return [
        float(np.tanh(float(current.get("turn", 0)) / 50.0)),
        float(np.tanh(float(current.get("turnActionCount", 0)) / 10.0)),
        going_first_feature(obs),
        1.0 if current.get("supporterPlayed") else 0.0,
        1.0 if current.get("stadiumPlayed") else 0.0,
        1.0 if current.get("energyAttached") else 0.0,
        1.0 if current.get("retreated") else 0.0,
        float(prize_count(obs, your_index)) / float(DEFAULT_PRIZE_COUNT),
        float(prize_count(obs, opp_index)) / float(DEFAULT_PRIZE_COUNT),
        float(np.tanh(float(self_player.get("deckCount", 0)) / 60.0)),
        float(np.tanh(float(opp_player.get("deckCount", 0)) / 60.0)),
        float(np.tanh(float(self_player.get("handCount", 0)) / 10.0)),
        float(np.tanh(float(opp_player.get("handCount", 0)) / 10.0)),
        float(np.tanh(self_discard / 60.0)),
        float(np.tanh(opp_discard / 60.0)),
        float(len(self_player.get("bench") or []) / bench_max_self),
        float(len(opp_player.get("bench") or []) / bench_max_opp),
        float(your_index),
    ]


def _select_features(obs: dict[str, Any]) -> list[float]:
    select = obs.get("select") or {}
    options = select.get("option") or []
    return [
        float(np.tanh(float(select.get("type", 0)) / 20.0)),
        float(np.tanh(float(select.get("minCount", 0)) / 5.0)),
        float(np.tanh(float(select.get("maxCount", 0)) / 5.0)),
        float(np.tanh(len(options) / 20.0)),
        float(np.tanh(float(select.get("remainDamageCounter", 0)) / 10.0)),
        float(np.tanh(float(select.get("remainEnergyCost", 0)) / 10.0)),
    ]


def _structured_features(
    obs: dict[str, Any],
    tracker: GameEventTracker,
    *,
    card_vocab_size: int,
) -> tuple[list[float], np.ndarray]:
    current = obs.get("current") or {}
    your_index = int(current.get("yourIndex", 0))
    opp_index = 1 - your_index
    players = current.get("players") or [{}, {}]
    self_player = players[your_index] if your_index < len(players) else {}
    opp_player = players[opp_index] if opp_index < len(players) else {}

    dense: list[float] = []
    card_ids = np.zeros(CARD_ID_SLOT_COUNT, dtype=np.int64)
    slot = 0

    self_active = _active_pokemon(self_player)
    dense.extend(_pokemon_slot_features(self_active, status=_player_status(self_player)))
    card_ids[slot] = card_id_index(_card_id_from_card(self_active), vocab_size=card_vocab_size)
    slot += 1

    for pokemon in _bench_pokemon(self_player):
        dense.extend(_pokemon_slot_features(pokemon))
        card_ids[slot] = card_id_index(_card_id_from_card(pokemon), vocab_size=card_vocab_size)
        slot += 1

    opp_active = _active_pokemon(opp_player)
    dense.extend(_pokemon_slot_features(opp_active))
    card_ids[slot] = card_id_index(_card_id_from_card(opp_active), vocab_size=card_vocab_size)
    slot += 1

    for pokemon in _bench_pokemon(opp_player):
        dense.extend(_pokemon_slot_features(pokemon))
        card_ids[slot] = card_id_index(_card_id_from_card(pokemon), vocab_size=card_vocab_size)
        slot += 1

    hand = self_player.get("hand") or []
    for index in range(SELF_HAND_CARD_SLOTS):
        card = hand[index] if index < len(hand) and isinstance(hand[index], dict) else None
        card_ids[slot] = card_id_index(_card_id_from_card(card), vocab_size=card_vocab_size)
        slot += 1

    stadium = (current.get("stadium") or [None])[0] if current.get("stadium") else None
    card_ids[slot] = card_id_index(_card_id_from_card(stadium if isinstance(stadium, dict) else None), vocab_size=card_vocab_size)
    slot += 1

    select = obs.get("select") or {}
    effect = select.get("effect")
    card_ids[slot] = card_id_index(
        _card_id_from_card(effect if isinstance(effect, dict) else None),
        vocab_size=card_vocab_size,
    )

    dense.extend(_global_features(obs, your_index=your_index, opp_index=opp_index))
    dense.extend(_select_features(obs))
    dense.extend(tracker.observe(obs))
    return dense, card_ids


def base_features_from_observation(obs: dict[str, Any]) -> list[float]:
    """Legacy helper: global + select only (no board slots)."""
    current = obs.get("current") or {}
    your_index = int(current.get("yourIndex", 0))
    opp_index = 1 - your_index
    return _global_features(obs, your_index=your_index, opp_index=opp_index) + _select_features(obs)


def pad_coarse_features(stored: list[float] | np.ndarray) -> list[float]:
    values = [float(v) for v in stored[:STRUCTURED_FEATURE_DIM]]
    if len(values) < STRUCTURED_FEATURE_DIM:
        values.extend([0.0] * (STRUCTURED_FEATURE_DIM - len(values)))
    return values


def features_from_observation(
    obs: dict[str, Any],
    tracker: GameEventTracker | None = None,
    *,
    card_vocab_size: int = 2000,
) -> tuple[list[float], np.ndarray]:
    if tracker is None:
        tracker = GameEventTracker()
    return _structured_features(obs, tracker, card_vocab_size=card_vocab_size)


def encode_observation_step(
    observation: dict[str, Any],
    tracker: GameEventTracker,
    *,
    state_hash_dim: int,
    action: Any = None,
    our_deck: list[int] | None = None,
    card_vocab_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    structured, card_ids = _structured_features(
        observation,
        tracker,
        card_vocab_size=card_vocab_size,
    )
    compact = np.array(structured, dtype=np.float32)
    residual = residual_state_hash(
        observation,
        action,
        state_hash_dim=state_hash_dim,
        our_deck=our_deck,
    )
    return np.concatenate([compact, residual]).astype(np.float32), card_ids.astype(np.int64)


def combine_features(
    coarse: list[float],
    observation: Any = None,
    action: Any = None,
    *,
    state_hash_dim: int,
    our_deck: list[int] | None = None,
    card_vocab_size: int = 2000,
    tracker: GameEventTracker | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if observation is not None and tracker is not None:
        return encode_observation_step(
            observation,
            tracker,
            state_hash_dim=state_hash_dim,
            action=action,
            our_deck=our_deck,
            card_vocab_size=card_vocab_size,
        )
    compact = np.array(coarse, dtype=np.float32)
    if observation is None and our_deck is None:
        card_ids = np.zeros(CARD_ID_SLOT_COUNT, dtype=np.int64)
        return compact, card_ids
    residual = residual_state_hash(
        observation,
        action,
        state_hash_dim=state_hash_dim,
        our_deck=our_deck,
    )
    card_ids = np.zeros(CARD_ID_SLOT_COUNT, dtype=np.int64)
    return np.concatenate([compact, residual]).astype(np.float32), card_ids


def row_feature_vector(
    row: dict,
    *,
    state_hash_dim: int,
    tracker: GameEventTracker | None = None,
    card_vocab_size: int = 2000,
) -> np.ndarray:
    features, _ = row_feature_and_cards(
        row,
        state_hash_dim=state_hash_dim,
        tracker=tracker,
        card_vocab_size=card_vocab_size,
    )
    return features


def row_feature_and_cards(
    row: dict,
    *,
    state_hash_dim: int,
    tracker: GameEventTracker | None = None,
    card_vocab_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    our_deck = seat_deck_from_row(row)
    observation = row.get("observation")
    if observation is not None and tracker is not None:
        return encode_observation_step(
            observation,
            tracker,
            state_hash_dim=state_hash_dim,
            action=row.get("action"),
            our_deck=our_deck,
            card_vocab_size=card_vocab_size,
        )
    if observation is not None:
        structured, card_ids = features_from_observation(observation, card_vocab_size=card_vocab_size)
        features, card_ids = combine_features(
            structured,
            observation,
            row.get("action"),
            state_hash_dim=state_hash_dim,
            our_deck=our_deck,
            card_vocab_size=card_vocab_size,
        )
        return features, card_ids
    features, card_ids = combine_features(
        pad_coarse_features(row["features"]),
        None,
        row.get("action"),
        state_hash_dim=state_hash_dim,
    )
    return features, card_ids


def row_next_feature_vector(
    row: dict,
    *,
    state_hash_dim: int,
    tracker: GameEventTracker | None = None,
    card_vocab_size: int = 2000,
) -> np.ndarray:
    features, _ = row_next_feature_and_cards(
        row,
        state_hash_dim=state_hash_dim,
        tracker=tracker,
        card_vocab_size=card_vocab_size,
    )
    return features


def row_next_feature_and_cards(
    row: dict,
    *,
    state_hash_dim: int,
    tracker: GameEventTracker | None = None,
    card_vocab_size: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    if "next_observation" in row and tracker is not None:
        return encode_observation_step(
            row["next_observation"],
            tracker,
            state_hash_dim=state_hash_dim,
            card_vocab_size=card_vocab_size,
        )
    if "next_observation" in row and "next_features" in row:
        return combine_features(
            pad_coarse_features(row["next_features"]),
            row.get("next_observation"),
            None,
            state_hash_dim=state_hash_dim,
            card_vocab_size=card_vocab_size,
        )
    return row_feature_and_cards(
        row,
        state_hash_dim=state_hash_dim,
        tracker=tracker,
        card_vocab_size=card_vocab_size,
    )


def default_tensor_build_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, cpu_count - 2 if cpu_count > 4 else cpu_count))


def _group_rows_by_seat(rows: list[dict]) -> list[list[dict]]:
    by_key: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        key = (int(row["episode"]), int(row["player"]))
        by_key.setdefault(key, []).append(row)
    sequences = [by_key[key] for key in sorted(by_key)]
    for sequence_rows in sequences:
        sequence_rows.sort(key=lambda row: int(row["step"]))
    return sequences


def _build_seat_sequence(
    seat_rows: list[dict],
    *,
    transition_classes: int,
    state_hash_dim: int,
    window_size: int,
    card_vocab_size: int,
    value_gamma: float,
    value_shaping_alpha: float,
    value_win: float,
    value_not_win: float,
    value_timeout: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    if not seat_rows:
        empty = np.zeros((0, total_feature_dim(state_hash_dim=state_hash_dim)), dtype=np.float32)
        return (
            empty,
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            empty,
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, window_size), dtype=np.float32),
            np.zeros((0, window_size), dtype=np.int64),
            np.zeros((0, window_size, CARD_ID_SLOT_COUNT), dtype=np.int64),
            0,
        )

    seat = int(seat_rows[0]["player"])
    terminal_obs = seat_rows[-1].get("next_observation") or seat_rows[-1].get("observation")
    result = int(seat_rows[-1].get("result", -1))
    mc_returns = episode_outcome_returns(
        seat_rows,
        value_gamma,
        seat=seat,
        value_win=value_win,
        value_not_win=value_not_win,
        value_timeout=value_timeout,
        terminal_obs=terminal_obs,
        result=result,
    )

    tracker = GameEventTracker()
    encoded: list[tuple[np.ndarray, np.ndarray]] = []
    for row in seat_rows:
        our_deck = seat_deck_from_row(row)
        if row.get("observation") is not None:
            encoded.append(
                encode_observation_step(
                    row["observation"],
                    tracker,
                    state_hash_dim=state_hash_dim,
                    action=row.get("action"),
                    our_deck=our_deck,
                    card_vocab_size=card_vocab_size,
                )
            )
        else:
            encoded.append(
                combine_features(
                    pad_coarse_features(row["features"]),
                    None,
                    row.get("action"),
                    state_hash_dim=state_hash_dim,
                )
            )

    seq_len = len(encoded)
    keep_len = min(seq_len, window_size)
    start = seq_len - keep_len

    xs = np.stack([encoded[index][0] for index in range(start, seq_len)]).astype(np.float32)
    card_ids = np.stack([encoded[index][1] for index in range(start, seq_len)]).astype(np.int64)
    values = np.zeros((keep_len,), dtype=np.float32)
    returns = np.zeros((keep_len,), dtype=np.float32)
    transition_targets = np.zeros((keep_len,), dtype=np.int64)
    next_features = np.zeros((keep_len, xs.shape[1]), dtype=np.float32)
    terminal_mask = np.zeros((keep_len,), dtype=np.float32)
    seq_mask = np.ones((window_size,), dtype=np.float32)
    padded_x = np.zeros((window_size, xs.shape[1]), dtype=np.float32)
    padded_cards = np.zeros((window_size, CARD_ID_SLOT_COUNT), dtype=np.int64)

    pad_count = window_size - keep_len
    if pad_count > 0:
        seq_mask[:pad_count] = 0.0
        padded_x[pad_count:] = xs
        padded_cards[pad_count:] = card_ids
    else:
        padded_x[:] = xs
        padded_cards[:] = card_ids

    for local_index, row_index in enumerate(range(start, seq_len)):
        row = seat_rows[row_index]
        features, _ = encoded[row_index]
        obs_after = row.get("next_observation") or row.get("observation")
        values[local_index] = blended_value_target(
            mc_returns[row_index],
            obs_after,
            seat,
            shaping_alpha=value_shaping_alpha,
            value_win=value_win,
            value_not_win=value_not_win,
        )
        returns[local_index] = float(mc_returns[row_index])

        if row_index + 1 < seq_len:
            next_feature, _ = encoded[row_index + 1]
            is_terminal = 0.0
        else:
            next_feature = features.copy()
            is_terminal = 1.0

        if "action" in row:
            action_key = json.dumps(row["action"], sort_keys=True, separators=(",", ":"))
            transition_class = stable_hash_index(action_key, transition_classes)
        elif is_terminal:
            transition_class = transition_classes - 1
        else:
            delta = next_feature[:STRUCTURED_FEATURE_DIM] - features[:STRUCTURED_FEATURE_DIM]
            transition_class = int(abs(delta).argmax()) % transition_classes

        transition_targets[local_index] = transition_class
        next_features[local_index] = next_feature
        terminal_mask[local_index] = is_terminal

    y_padded = np.zeros((window_size,), dtype=np.float32)
    returns_padded = np.zeros((window_size,), dtype=np.float32)
    transition_padded = np.zeros((window_size,), dtype=np.int64)
    next_x_padded = np.zeros((window_size, xs.shape[1]), dtype=np.float32)
    terminal_padded = np.zeros((window_size,), dtype=np.float32)
    y_padded[pad_count:] = values
    returns_padded[pad_count:] = returns
    transition_padded[pad_count:] = transition_targets
    next_x_padded[pad_count:] = next_features
    terminal_padded[pad_count:] = terminal_mask

    return (
        padded_x,
        y_padded,
        returns_padded,
        transition_padded,
        next_x_padded,
        terminal_padded,
        seq_mask,
        padded_cards,
        np.array([keep_len], dtype=np.int64),
        keep_len,
    )


def _seat_worker(args: tuple[list[dict], int, int, int, int, float, float, float, float, float]):
    seat_rows, transition_classes, state_hash_dim, window_size, card_vocab_size, value_gamma, value_shaping_alpha, value_win, value_not_win, value_timeout = args
    return _build_seat_sequence(
        seat_rows,
        transition_classes=transition_classes,
        state_hash_dim=state_hash_dim,
        window_size=window_size,
        card_vocab_size=card_vocab_size,
        value_gamma=value_gamma,
        value_shaping_alpha=value_shaping_alpha,
        value_win=value_win,
        value_not_win=value_not_win,
        value_timeout=value_timeout,
    )


def build_training_arrays(
    rows: list[dict],
    *,
    transition_classes: int,
    state_hash_dim: int,
    window_size: int,
    workers: int | None = None,
    card_vocab_size: int = 2000,
    value_gamma: float = 0.997,
    value_shaping_alpha: float = 0.15,
    value_win: float = 1.0,
    value_not_win: float = -1.0,
    value_timeout: float = -2.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    sequences = _group_rows_by_seat(rows)
    worker_count = default_tensor_build_workers() if workers is None else max(1, int(workers))
    common = (
        transition_classes,
        state_hash_dim,
        window_size,
        card_vocab_size,
        value_gamma,
        value_shaping_alpha,
        value_win,
        value_not_win,
        value_timeout,
    )
    if worker_count <= 1 or len(sequences) < worker_count * 2:
        seat_arrays = [
            _build_seat_sequence(
                sequence_rows,
                transition_classes=transition_classes,
                state_hash_dim=state_hash_dim,
                window_size=window_size,
                card_vocab_size=card_vocab_size,
                value_gamma=value_gamma,
                value_shaping_alpha=value_shaping_alpha,
                value_win=value_win,
                value_not_win=value_not_win,
                value_timeout=value_timeout,
            )
            for sequence_rows in sequences
        ]
    else:
        tasks = [(sequence_rows, *common) for sequence_rows in sequences]
        chunksize = max(1, len(tasks) // (worker_count * 4))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            seat_arrays = list(executor.map(_seat_worker, tasks, chunksize=chunksize))
        print(
            f"tensor build: {len(rows)} rows across {len(sequences)} seat-sequences "
            f"using {worker_count} workers"
        )

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    returns_parts: list[np.ndarray] = []
    transition_parts: list[np.ndarray] = []
    next_x_parts: list[np.ndarray] = []
    terminal_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    card_parts: list[np.ndarray] = []
    lengths: list[int] = []

    for x_seq, y_seq, returns_seq, transition_seq, next_x_seq, terminal_seq, seq_mask, card_seq, length_arr, keep_len in seat_arrays:
        if keep_len == 0:
            continue
        x_parts.append(x_seq)
        y_parts.append(y_seq)
        returns_parts.append(returns_seq)
        transition_parts.append(transition_seq)
        next_x_parts.append(next_x_seq)
        terminal_parts.append(terminal_seq)
        mask_parts.append(seq_mask)
        card_parts.append(card_seq)
        lengths.append(int(length_arr[0]))

    if not x_parts:
        feat_dim = total_feature_dim(state_hash_dim=state_hash_dim)
        return (
            np.zeros((0, window_size, feat_dim), dtype=np.float32),
            np.zeros((0, window_size), dtype=np.float32),
            np.zeros((0, window_size), dtype=np.float32),
            np.zeros((0, window_size), dtype=np.int64),
            np.zeros((0, window_size, feat_dim), dtype=np.float32),
            np.zeros((0, window_size), dtype=np.float32),
            np.zeros((0, window_size), dtype=np.float32),
            np.zeros((0, window_size, CARD_ID_SLOT_COUNT), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.stack(x_parts).astype(np.float32),
        np.stack(y_parts).astype(np.float32),
        np.stack(returns_parts).astype(np.float32),
        np.stack(transition_parts).astype(np.int64),
        np.stack(next_x_parts).astype(np.float32),
        np.stack(terminal_parts).astype(np.float32),
        np.stack(mask_parts).astype(np.float32),
        np.stack(card_parts).astype(np.int64),
        np.array(lengths, dtype=np.int64),
        np.arange(len(lengths), dtype=np.int64),
    )
