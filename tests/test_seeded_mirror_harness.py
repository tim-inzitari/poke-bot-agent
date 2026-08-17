from __future__ import annotations

import pytest

from poke_bot.seeded_mirror_harness import (
    PairFirstPlayerSeal,
    SeededMirrorHarnessError,
    build_seeded_seat_swapped_schedule,
    canonical_sha256,
    engine_turn_identity_from_observation,
    validate_pair_first_player_seal,
)

SEED_IDENTITY = canonical_sha256({"test": "seeded-mirror-harness"})


def test_two_game_pair_has_common_engine_material_and_swapped_seats() -> None:
    games = build_seeded_seat_swapped_schedule(
        evaluation_id="test-mirror", seed_identity_sha256=SEED_IDENTITY, pair_count=1
    )

    assert len(games) == 2
    first, second = games
    assert first.engine_seed_u32 == second.engine_seed_u32
    assert first.deck_order_seed_u32 == second.deck_order_seed_u32
    assert (first.experimental_seat, first.control_seat) == (0, 1)
    assert (second.experimental_seat, second.control_seat) == (1, 0)
    assert first.experimental_rng_seed_u32 != second.experimental_rng_seed_u32


def test_full_500_pair_schedule_is_reproducible_and_balanced_by_seat() -> None:
    first = build_seeded_seat_swapped_schedule(
        evaluation_id="test-mirror", seed_identity_sha256=SEED_IDENTITY, pair_count=500
    )
    second = build_seeded_seat_swapped_schedule(
        evaluation_id="test-mirror", seed_identity_sha256=SEED_IDENTITY, pair_count=500
    )

    assert first == second
    assert len(first) == 1_000
    assert sum(game.experimental_seat == 0 for game in first) == 500
    assert sum(game.experimental_seat == 1 for game in first) == 500


@pytest.mark.parametrize("first_player", [0, 1])
def test_engine_sealed_first_player_makes_every_pair_exactly_first_and_second(
    first_player: int,
) -> None:
    games = build_seeded_seat_swapped_schedule(
        evaluation_id="test-mirror", seed_identity_sha256=SEED_IDENTITY, pair_count=1
    )
    game = games[0]
    seal = PairFirstPlayerSeal(
        evaluation_id=game.evaluation_id,
        pair_index=game.pair_index,
        pair_id=game.pair_id,
        pair_nonce_sha256=game.pair_nonce_sha256,
        engine_seed_u32=game.engine_seed_u32,
        deck_order_seed_u32=game.deck_order_seed_u32,
        first_player_seat=first_player,
        post_turn_order_observation_sha256=canonical_sha256({"first": first_player}),
    )

    validate_pair_first_player_seal(games, seal)
    assert [game.experimental_seat == first_player for game in games].count(True) == 1
    assert [game.experimental_seat == first_player for game in games].count(False) == 1


def test_pair_seal_rejects_crossed_seed_material() -> None:
    games = build_seeded_seat_swapped_schedule(
        evaluation_id="test-mirror", seed_identity_sha256=SEED_IDENTITY, pair_count=1
    )
    game = games[0]
    seal = PairFirstPlayerSeal(
        evaluation_id=game.evaluation_id,
        pair_index=game.pair_index,
        pair_id=game.pair_id,
        pair_nonce_sha256=game.pair_nonce_sha256,
        engine_seed_u32=game.engine_seed_u32 + 1,
        deck_order_seed_u32=game.engine_seed_u32 + 1,
        first_player_seat=0,
        post_turn_order_observation_sha256=canonical_sha256({"bad": True}),
    )
    with pytest.raises(SeededMirrorHarnessError, match="does not bind"):
        validate_pair_first_player_seal(games, seal)


def test_engine_turn_identity_requires_native_turn_and_atomic_counters() -> None:
    identity = engine_turn_identity_from_observation(
        {"current": {"yourIndex": 1, "turn": 9, "turnActionCount": 3}}
    )
    assert identity.turn_key == (1, 9)
    assert identity.turn_action_count == 3
    with pytest.raises(SeededMirrorHarnessError, match="turnActionCount"):
        engine_turn_identity_from_observation(
            {"current": {"yourIndex": 1, "turn": 9}}
        )
