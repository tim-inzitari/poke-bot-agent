# r205 simulator-backed search foundation

This directory is isolated R&D only. It is not imported by a selector, service,
submission, training job, or the existing r198 evaluator.

## What it proves locally

`foundation.py` implements a small, injection-only one-turn
expectimax/MCTS substrate. Every decision edge is expanded through a supplied
native `clone(state)` followed by `branch.step(action)` call; it does not
accept a prebuilt action tree as proof of simulator search. Finite chance is
represented by exact `Fraction` probabilities, requires all outcomes and
future-legality fingerprints, and backs up `Σ p(outcome) × value(outcome)`.
Unknown, private, incomplete, or deadline-limited branches discard the shadow
tree and leave the exact NO-RTP direct action in place.

The isolated API is intentionally shadow-only: `ShadowSearchDecision` always
has `action_authority_enabled=False` and `executed_action == direct_action`.
It can expose a `shadow_recommended_action` after a complete tree only for
diagnostic comparison.

It also generates and validates the exact r205 schedule: 500 immutable start
materials, two games per material, MCTS in seat 0 once and seat 1 once, for
1,000 games total. The two games in a pair use the same sealed initial
material; swapping arms does not falsely claim that later divergent action
histories consume a common RNG path.

Run the focused checks with:

```bash
uv run --with pytest pytest -q tests/test_r205_rng_paired_mcts_foundation.py
```

## Current native-simulator finding

The available primitives do **not** meet r205's arbitrary-midgame branch
requirement:

- `poke_bot.cg_env.search_begin()` requires caller-supplied predicted hidden
  decks, prizes, hands, and active cards. It is not an information-set-safe
  exact successor/future-legality facility.
- `poke_bot.cg_env.battle_select()` advances the one mutable live battle; it
  exposes no clone or restore-at-current-decision operation.
- The private r198 pairing ABI intentionally captures only the exact
  `post_battle_start_first_external_selection` boundary. Its native export
  labels that boundary "not a general mid-game clone facility." Restoring that
  seal gives a fresh root game, not a clone of a later decision.

Accordingly, `MidgameCloneCapability` requires an independently attested native
ABI with all of:

1. `source_kind == "native_midgame_clone"`;
2. arbitrary policy-visible decision scope;
3. complete state, Game RNG, config, and counter cloning;
4. exact future legality and information-set safety; and
5. independent clone handles.

A `source_kind == "sealed_start_replay"` input fails closed before any clone or
simulator step occurs. A later engine overlay must supply this capability and
its receipt before this foundation can contribute to a real r205 MCTS/expectimax
preflight. It must still satisfy r205's separate same-checkpoint parity,
clock, integrity, determinism, and safe-remote gates.

## Explicit non-goals

- No BO1000 launch, remote use, service manipulation, or attempt-10 mutation.
- No selector/action/serving/training/submission authority.
- No claim that a requested seed or BattleStart replay is a midgame clone.
- No sampled or probability-reweighted chance expansion.
