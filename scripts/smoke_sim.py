#!/usr/bin/env python
"""Phase 1 foundation smoke test.

Exercises the whole simulator + featurization foundation end-to-end:

  1. Load the Hammer-Pult deck (flat 60-int list) and classify it.
  2. battle_start with two copies; play a full random-legal game to completion
     via battle_select; print the winner.
  3. Build board/option tokens from a live observation (features API).
  4. Separately drive the Search API: build predicted decks -> search_begin ->
     a few search_step -> search_end.

Run with the project's Python (conda env `poke-bot-agent`, which has cg-compatible
Python 3.11 + torch). Exits non-zero on any failure.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

# Make `import poke_bot` work regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from poke_bot import archetypes, cg_env, deck_pool, features  # noqa: E402

MAX_GAME_STEPS = 4000


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def play_full_game(deck: list[int], rng: random.Random) -> tuple[int, int]:
    """Play one random-legal game to completion. Returns (winner, num_steps)."""
    obs, start = cg_env.battle_start(deck, deck)
    if getattr(start, "errorPlayer", -1) >= 0:
        raise RuntimeError(f"battle_start deck error: type={start.errorType}")
    if obs is None:
        raise RuntimeError("battle_start returned no observation.")

    steps = 0
    while not cg_env.is_finished(obs):
        if steps >= MAX_GAME_STEPS:
            raise RuntimeError(f"Game did not finish within {MAX_GAME_STEPS} steps.")
        sel = obs.get("select")
        if sel is None:
            # Local sim shouldn't hit this, but honour the deck-selection contract.
            select = deck
        else:
            select = cg_env.random_legal_select(obs, rng)
        obs = cg_env.battle_select(select)
        steps += 1

    winner = cg_env.result_winner(obs)
    cg_env.battle_finish()
    return winner, steps


def exercise_features(deck: list[int], rng: random.Random) -> None:
    """Capture a mid-game observation and build board + option tokens."""
    obs, start = cg_env.battle_start(deck, deck)
    if obs is None or getattr(start, "errorPlayer", -1) >= 0:
        raise RuntimeError("battle_start failed for feature exercise.")

    # Step a few times to reach an interesting decision point.
    captured = obs
    for _ in range(rng.randint(3, 8)):
        if cg_env.is_finished(captured):
            break
        sel = captured.get("select")
        if sel is None:
            break
        captured = cg_env.battle_select(cg_env.random_legal_select(captured, rng))

    board = features.build_board_tokens(captured, deck)
    combos = features.enumerate_action_combos(captured)
    options = features.build_option_tokens(captured, combos)

    print(f"  board tokens: num_words={board.num_words} (expected {features.NUM_BOARD_TOKENS}), "
          f"nnz={len(board.index)}")
    print(f"  action combos enumerated: {len(combos)}")
    print(f"  option tokens: num_words={options.num_words}, nnz={len(options.index)}")
    print(f"  card vocab={features.card_vocab_size()} attack vocab={features.attack_vocab_size()}")
    print(f"  encoder vocab≈{features.encoder_vocab_size()} decoder vocab={features.decoder_vocab_size()}")

    assert board.num_words == features.NUM_BOARD_TOKENS, "board token count mismatch"
    assert options.num_words == len(combos), "option token count mismatch"
    cg_env.battle_finish()


def exercise_search(deck: list[int], rng: random.Random) -> None:
    """Drive search_begin -> a few search_step -> search_end."""
    obs, start = cg_env.battle_start(deck, deck)
    if obs is None or getattr(start, "errorPlayer", -1) >= 0:
        raise RuntimeError("battle_start failed for search exercise.")

    # Take a couple of real steps so we search from a non-trivial position.
    for _ in range(rng.randint(2, 5)):
        if cg_env.is_finished(obs):
            break
        sel = obs.get("select")
        if sel is None:
            break
        obs = cg_env.battle_select(cg_env.random_legal_select(obs, rng))

    search_inputs = cg_env.build_search_inputs(obs, deck, opponent_deck_guess=deck)
    root = cg_env.search_begin(obs, search_inputs)
    print(f"  search_begin ok: root searchId={root.searchId}")

    node = root
    n_steps = 0
    for _ in range(5):
        if node.observation.current is not None and node.observation.current.result >= 0:
            print(f"  search reached terminal after {n_steps} steps")
            break
        select = cg_env.legal_select_from_searchstate(node, rng)
        node = cg_env.search_step(node.searchId, select)
        n_steps += 1
    print(f"  search_step x{n_steps} ok (last searchId={node.searchId})")

    cg_env.search_end()
    print("  search_end ok")
    cg_env.battle_finish()


def main() -> int:
    rng = random.Random(1234)

    section("cg runtime")
    parent = cg_env.ensure_cg_importable()
    print(f"  cg runtime dir: {parent}")

    section("deck load + classify")
    deck = deck_pool.primary_deck()
    print(f"  deck size: {len(deck)}")
    print(f"  classify_deck -> {archetypes.classify_deck(deck)}")
    print(f"  hammer signature: {archetypes.is_hammer_signature(deck)}")
    assert len(deck) == 60
    assert archetypes.classify_deck(deck) == "dragapult"
    assert not archetypes.is_hammer_signature(deck)

    section("full random-legal game")
    t0 = time.time()
    winner, steps = play_full_game(deck, rng)
    print(f"  game finished: winner={winner} steps={steps} ({time.time()-t0:.2f}s)")
    assert winner in (0, 1, 2), f"unexpected winner {winner}"

    section("features")
    exercise_features(deck, rng)

    section("search API")
    exercise_search(deck, rng)

    print("\nSMOKE TEST PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
