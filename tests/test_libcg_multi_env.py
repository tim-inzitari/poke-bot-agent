"""Live tests against downloaded competition libcg (skipped if missing)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CG_DIR = Path(
    "kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg"
)
DECK_CSV = Path(
    "kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv"
)


pytestmark = pytest.mark.skipif(
    not (CG_DIR / "libcg.so").exists() or not DECK_CSV.exists(),
    reason="competition libcg / deck.csv not downloaded",
)


@pytest.fixture(scope="module")
def deck() -> list[int]:
    return [int(x) for x in DECK_CSV.read_text().splitlines() if x.strip()]


@pytest.fixture(scope="module")
def cg_on_path() -> None:
    """Put competition sample_submission on sys.path once (GameInitialize once)."""
    parent = str(CG_DIR.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    # Import once for the whole module — re-importing calls GameInitialize and aborts.
    import cg.sim  # noqa: F401


def test_libcg_multi_env_concurrent_handles(deck: list[int], cg_on_path: None) -> None:
    from poke_bot.engine_rebuild.interfaces import ResetSpec
    from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv

    env = LibcgMultiEnv(4)
    batch = env.reset([ResetSpec(deck, deck, seed=i) for i in range(4)])
    assert len(batch) == 4
    assert all(not e.done for e in batch.envs)
    assert all((e.obs.get("select") or {}).get("option") for e in batch.envs)

    # Distinct native handles
    ptrs = [p for p in env._ptrs if p is not None]
    assert len(ptrs) == 4
    assert len(set(ptrs)) == 4

    actions = []
    for e in batch.envs:
        opts = (e.obs.get("select") or {}).get("option") or []
        max_c = int((e.obs.get("select") or {}).get("maxCount") or 1)
        actions.append(list(range(min(max_c, len(opts)))))

    batch2 = env.step_batch(actions)
    assert len(batch2) == 4
    assert all(not e.done for e in batch2.envs)
    env.close()
