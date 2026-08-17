"""BeliefMCTS backend for Stage C / runtime critical-node search."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from .authority import RESEARCH_ONLY
from .critical_search import SearchFn, mock_top_k_search


@dataclass
class BeliefSearchBundle:
    """Optional live search dependencies. Missing pieces → mock fallback."""

    model: Any = None
    deck: Optional[list[int]] = None
    posterior: Any = None
    checkpoint_digest: str = ""
    device: Any = None
    max_sims: int = 32
    move_time_s: float = 2.0
    leaf_backend: Any = None


def belief_mcts_available() -> bool:
    try:
        from poke_bot import paths  # noqa: F401

        paths.cg_runtime_dir()
        from poke_bot.belief_mcts import BeliefMCTS  # noqa: F401

        return True
    except Exception:
        return False


def search_fn_from_bundle(bundle: BeliefSearchBundle) -> SearchFn:
    """Return a SearchFn using BeliefMCTS when possible, else mock."""

    def _mock(obs: dict[str, Any], legal: Sequence[Sequence[int]]) -> dict[str, Any]:
        row = {
            "observation": obs,
            "legal_action_combos": [list(a) for a in legal],
            "action": list(legal[0]) if legal else [],
            "game_id": "runtime",
            "env_step": -1,
        }
        receipt = mock_top_k_search(row)
        payload = dict(receipt.payload)
        payload["search_backend"] = "mock_top_k"
        payload["research_only"] = RESEARCH_ONLY
        return {
            "candidate_actions": payload.get("candidate_actions") or [],
            "visit_counts": payload.get("visit_counts") or [],
            "q_values": payload.get("q_values") or [],
            "chosen_action": payload.get("chosen_action") or [],
            "belief_particles": payload.get("belief_particles") or [],
            "simulator_version": "mock",
            "max_sims": 0,
            "search_backend": "mock_top_k",
        }

    if (
        not bundle.model
        or not bundle.deck
        or not bundle.posterior
        or not str(bundle.checkpoint_digest).startswith("sha256:")
        or not belief_mcts_available()
    ):
        return _mock

    from poke_bot.belief import PublicBeliefHistory
    from poke_bot.belief_mcts import BeliefMCTS

    def _live(obs: dict[str, Any], legal: Sequence[Sequence[int]]) -> dict[str, Any]:
        engine = BeliefMCTS(
            bundle.model,
            list(bundle.deck),
            bundle.posterior,
            checkpoint_digest=str(bundle.checkpoint_digest),
            model_generation=0,
            device=bundle.device,
            leaf_backend=bundle.leaf_backend,
        )
        t0 = time.perf_counter()
        result = engine.search(
            obs,
            belief_history=PublicBeliefHistory(),
            root_history_boards=[],
            root_history_previous_actions=[],
            max_sims=int(bundle.max_sims),
            move_time_s=float(bundle.move_time_s),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        combos = list(getattr(result.target, "action_combos", None) or legal)
        visits = list(getattr(result.target, "visits", None) or [])
        if not visits and combos:
            visits = [0.0] * len(combos)
        total = sum(float(v) for v in visits) or 1.0
        q_values = [float(v) / total for v in visits]
        chosen = list(result.select)
        return {
            "candidate_actions": [list(c) for c in combos],
            "visit_counts": [float(v) for v in visits],
            "q_values": q_values,
            "chosen_action": chosen,
            "belief_particles": [
                {
                    "kind": "belief_mcts_root",
                    "weight": 1.0,
                    "elapsed_ms": elapsed_ms,
                }
            ],
            "simulator_version": "exact_cabt_belief_mcts",
            "max_sims": int(bundle.max_sims),
            "search_backend": "belief_mcts",
        }

    return _live


def resolve_stage_c_search_fn(
    bundle: Optional[BeliefSearchBundle] = None,
) -> tuple[SearchFn, str]:
    if bundle is None:
        return search_fn_from_bundle(BeliefSearchBundle()), "mock_top_k"
    fn = search_fn_from_bundle(bundle)
    backend = "belief_mcts" if belief_mcts_available() and bundle.model else "mock_top_k"
    return fn, backend
