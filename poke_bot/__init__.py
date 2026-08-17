"""poke_bot — competition-grade Pokémon TCG agent (Hammer-Pult first).

Phase 1 foundation modules:
  - :mod:`poke_bot.paths`      canonical filesystem paths
  - :mod:`poke_bot.device`     GPU selection (Blackwell for training)
  - :mod:`poke_bot.config`     hardware defaults + model/search knobs
  - :mod:`poke_bot.cg_env`     thin wrapper over the competition cg Game+Search APIs
  - :mod:`poke_bot.features`   board/option token builders from an Observation
  - :mod:`poke_bot.worker_pool` recycling sim-worker pool
  - :mod:`poke_bot.archetypes` archetype registry + deck classification
  - :mod:`poke_bot.deck_pool`  deck reader + per-archetype pool
  - :mod:`poke_bot.hammer_heuristics` Hammer-Pult soft-prior stubs

Phase 2A (policy / search):
  - :mod:`poke_bot.model`         TemporalCabtTransformer (RoPE + KV cache)
  - :mod:`poke_bot.matchup_id`    opponent archetype ID net
  - :mod:`poke_bot.mcts`          AlphaZero PUCT on Search API
  - :mod:`poke_bot.search_targets` visit-count policy targets
  - :mod:`poke_bot.checkpoint`    atomic save/load + resume helpers

Later phases add: dataset.py, train.py, self_play.py, agent.py.

Submodules are imported lazily to keep torch-free tooling (sim/features) usable
without a torch install.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = [
    "paths",
    "device",
    "config",
    "cg_env",
    "features",
    "worker_pool",
    "archetypes",
    "deck_pool",
    "hammer_heuristics",
    "model",
    "matchup_id",
    "mcts",
    "search_targets",
    "checkpoint",
]
