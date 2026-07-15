# Archetype heuristic conversion contract

Subject-matter expert guides (large PDFs / long text) are **not** loaded at
runtime. Convert them offline (e.g. with an AI model) into an archetype
heuristics **module** that implements the registry surface in
`poke_bot/heuristics_registry.py`.

## Target surface

```python
ENABLED: bool = False  # leave False until human review

def applies(deck_card_ids) -> bool: ...
def prior_logit_bias(obs, action_combos, *, scale=1.0) -> list[float]: ...
def describe() -> str: ...
```

Register the module under its `archetype_id` in
`poke_bot.heuristics_registry._REGISTRY` (or call `register(...)`).

Runtime applies bias as **active book** knowledge: additive logit bias on the
network policy prior inside BeliefMCTS (Baier/Winands-style), never a passive
“always play card X” replacement of search. Trusted belief-MCTS still respects
`min_trusted_sims = 128`.

## Preferred converter output schema

Emit JSON (or generate a module that embeds this table). Each rule:

| Field | Type | Meaning |
| --- | --- | --- |
| `archetype_id` | string | e.g. `hammer-pult` |
| `phase` | string | `opening` \| `setup` \| `mid` \| `prize_race` \| `closing` |
| `turn_min` / `turn_max` | int \| null | Inclusive turn bounds (optional) |
| `prefer` / `avoid` | list | Card ids and/or option types (`PLAY`, `ATTACK`, …) |
| `logit_bias` | float | Positive encourages prefer; negative for avoid |
| `rationale` | string | Short cite back to the guide section |

Map rules into `prior_logit_bias` by inspecting each action combo’s options
(`cg_env.OptionType`, card ids) and summing matching biases × `scale`.

## Workflow

1. Drop SME PDF text for **one** archetype.
2. Convert → structured rules table matching the schema above.
3. Codegen / hand-write `poke_bot/<arch>_heuristics.py` implementing the surface.
4. Register in `heuristics_registry`.
5. Keep `ENABLED=False` until a human signs off; then flip for experiments.
6. Opening-turn clarity in BeliefMCTS uses **network prior + heuristic bias**
   when enabled — sharp SME-shaped openings spend less optional search time.

## Non-goals

- Do not parse PDFs inside training workers.
- Do not lower the trusted simulation floor for heuristic shortcuts.
- Do not ship unvalidated biases with `ENABLED=True` by default.
