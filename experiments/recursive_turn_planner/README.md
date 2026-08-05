# Recursive Turn Planner (lightweight experiment)

Isolated R&D lane for a typed, bounded, latent Recursive Turn Planner.

## Intent

Replace online whole-turn MCTS as the primary decision mechanism with:

1. Encode-once persistent turn memory `H`
2. Complexity gate → direct policy on simple turns
3. Batched root proposal of typed turn programs
4. Shallow recursive subgoal refinement (depth ≤ 2)
5. Learned latent transitions as the plan evaluator
6. Exact legality / resource checks outside the network
7. Conditional plan persistence with sparse repair

Online simulator calls stay optional and sparse (default budget `0` in the
lightweight config). Teacher-search distillation and archetype adapters are
later stages, not part of this first cut.

## Package

Importable implementation:

```text
poke_bot/recursive_turn_planner/
```

Tests:

```bash
pytest -m unit tests/test_recursive_turn_planner.py
```

## Non-goals for this lane

- No production `PolicyAgent` default switch
- No `GOAL.md` / owner-contract changes
- No healthy training restarts
- No REPL / natural-language recursive prompts
