# Multi-Model Catalog

Edit `poke_agent/model_catalog.py` to define neural, heuristic, and hybrid agents
in one place. The legacy single-model path (`poke_agent/config.py` +
`notebooks/poke_agent_training.ipynb`) is unchanged.

## Quick start

```bash
# Edit poke_agent/model_catalog.py (ACTIVE_MODEL, TRAIN_MODELS, MODEL_CATALOG)
python scripts/train_models.py
```

Or open `notebooks/poke_agent_multi_model.ipynb`.

## Model kinds

| Kind | Trains? | Use case |
|---|---|---|
| `neural` | Yes | Transformer checkpoints under `outputs/checkpoints/` |
| `heuristic` | No | Random / first-legal baselines |
| `hybrid` | No* | Combine neural + heuristic at play time |

\*Hybrid entries reference neural + heuristic catalog ids. Train the neural
component via `TRAIN_MODELS`, then wire the checkpoint into `resolve_agent(...)`
later.

## Example catalog entry

```python
"temporal_current": {
    "kind": "neural",
    "description": "Current temporal transformer RL system (mirrors config.py)",
    "architecture": "transformer_rl",
    "window_size": 1024,
    "d_model": 256,
    "model_heads": 4,
    "model_layers": 4,
    "train_epochs": 1000,
    "batch_size": 64,
},
```

Checkpoints default to `outputs/checkpoints/{model_id}.pt` and training reports to
`outputs/reports/{model_id}.json`. Override with `output_path` only when needed.

Neural entries inherit defaults from `poke_agent/config.py` and override only the
keys you specify.

## Choosing models

```python
ACTIVE_MODEL = "hybrid_default_random"   # agent for evaluation / rollouts
TRAIN_MODELS = ["transformer_small", "transformer_default"]
```

## API

```python
from poke_agent.model_registry import describe_catalog, build_model_config
from poke_agent.multi_train import train_catalog_models
from poke_agent.agents import resolve_agent, resolve_active_agent
```
