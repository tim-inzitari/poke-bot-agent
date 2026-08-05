# Local swap-in: Recursive Turn Planner

This experimental branch wires RTP into `PolicyAgent` as the default non-MCTS
select path.

## Dispatch map

```text
PolicyAgent.__call__(obs)
        │
        ├─ select is None → return deck
        ├─ use_mcts → mcts_select / belief_mcts_select
        ├─ use_recursive_turn_planner + local model
        │         → rtp_select
        │              │
        │              ├─ forced IsFirst → unchanged go-first contract
        │              ├─ append board history (same as greedy)
        │              ├─ RTPAgentBridge.select
        │              │     encode-once (state_vec, option_hidden)
        │              │     plan_turn / continue PlanExecutor
        │              │     legal action or greedy fallback
        │              └─ diagnostics on bridge.last_diagnostics
        └─ else → greedy_select
```

## Enable / disable

On this branch the default is **on** when a local model is attached.

```python
from poke_bot.agent import PolicyAgent

agent = PolicyAgent(
    model=model,
    deck=deck,
    use_mcts=False,
    use_recursive_turn_planner=True,   # default on this branch
    rtp_sizing_profile=None,           # auto: 96→pure_rl, 256→global_transformer
)
```

Disable:

```bash
export POKEBOT_USE_RECURSIVE_TURN_PLANNER=0
```

or:

```python
PolicyAgent(..., use_recursive_turn_planner=False)
```

Force a sizing parent:

```bash
export POKEBOT_RTP_SIZING_PROFILE=pure_rl
```

## Local checklist

1. Checkout `cursor/experimentation-026a`
2. Load your usual checkpoint into `TemporalCabtTransformer`
3. Construct `PolicyAgent(model=..., use_mcts=False)` — RTP is active
4. Play / eval as normal through `agent(obs)`
5. Inspect `agent._rtp_bridge.last_diagnostics` for mode / fallback reason
6. If something looks wrong, set `use_recursive_turn_planner=False` to restore greedy

## Fallback behavior

RTP fails closed to the existing factorized greedy path when:

- no local model / bridge init failed
- action space encoding fails
- planned action is illegal
- any bridge exception

So local runs should keep moving even while the planner is immature.

## Not finished

Teacher distillation, online sim verification, and learned plan-codes are still
absent. This swap-in is the runtime map for the lightweight planner.
