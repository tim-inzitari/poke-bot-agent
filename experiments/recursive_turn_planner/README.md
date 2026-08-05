# Recursive Turn Planner (lightweight experiment)

Isolated R&D lane for a typed, bounded, latent Recursive Turn Planner.

## Intent

Replace online whole-turn MCTS as the primary decision mechanism with:

1. Encode-once persistent turn memory `H`
2. Complexity gate → direct policy on simple / trivial turns
3. Batched root proposal of typed turn programs
4. Shallow recursive subgoal refinement (depth ≤ 2)
5. Learned latent transitions as the plan evaluator
6. Exact legality / resource checks outside the network
7. Conditional plan persistence with sparse repair

Online simulator calls stay optional and sparse (default budget `0`).

## Sizing profiles

RTP must bind to an encoder parent. Silent width mismatch is a hard failure.

| Profile | `d_model` | `dynamics_width` | Parent in repo | Notes |
|---|---:|---:|---|---|
| `global_transformer` | 256 | 512 | `config.ModelConfig` / Hope-style CABT | Default; matches `latent_lookahead_width=512` |
| `pure_rl` | 96 | 192 | `pure_rl.model_profile` + matchup adapters | Lean production evaluator width |
| `unit_test` | 16 | 32 | tests only | Not deployable |

```python
from poke_bot.recursive_turn_planner import get_profile, RecursiveTurnPlanner

planner = RecursiveTurnPlanner(get_profile("pure_rl").to_config())
```

Shared with existing search code:
- `complexity_option_threshold=8` (`SearchConfig` / `mcts.planned_sims`)
- trivial skip for `n_options <= 1` (`submission_budget` forced/trivial)
- verify ablation budgets: 0 / 8 / 16 / 32 / 50 / 75 (`VERIFY_ABLATONS`)

## Reuse targets (do not reinvent)

| Concern | Existing implementation | RTP use |
|---|---|---|
| Option states | `decode_options(..., return_hidden=True)` | Preferred action embed (`option_hidden`) |
| Latent evaluator | `ActionConditionedLatentLookahead` | Optional via `LookaheadBackedDynamics` |
| Bounded residuals | `CausalDecisionFusion` / route width 16 | Future plan-score residual style |
| Archetype specialization | matchup adapter V6 (`96→8→96`, 64 slots) | Future plan-head adapters, not second trunk |
| Leaf batch hints | Blackwell 512 / 3080 Ti 256 / CPU 16 | `option_batch_hint` on profiles |

## Package

```text
poke_bot/recursive_turn_planner/
  config.py      # hard budgets + sizing fields
  profiles.py    # named parent-bound sizing contracts
  types.py       # typed turn-program AST
  memory.py      # encode-once H + option_hidden
  dynamics.py    # D(z,a) + lookahead adapter
  planner.py     # propose / recurse / score
  legality.py    # exact legal-action prune
  executor.py    # persist / branch / repair
```

```bash
pytest -m unit tests/test_recursive_turn_planner.py
```

## Non-goals for this lane

- No production `PolicyAgent` default switch
- No `GOAL.md` / owner-contract changes
- No healthy training restarts
- No REPL / natural-language recursive prompts
