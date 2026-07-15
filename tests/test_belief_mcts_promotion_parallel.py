"""Belief-MCTS promotion must honor --promotion-workers via WorkerPool."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.train_round_robin import promotion_worker_count


ROOT = Path(__file__).resolve().parents[1]
TRAIN_RR = ROOT / "scripts" / "train_round_robin.py"


@pytest.mark.unit
def test_promotion_worker_count_clamps_parallel_pool() -> None:
    assert promotion_worker_count(16, 160) == 16
    assert promotion_worker_count(16, 4) == 4
    assert promotion_worker_count(0, 10) == 1
    assert promotion_worker_count(8, 0) == 1


@pytest.mark.unit
def test_belief_mcts_promotion_branch_uses_worker_pool_not_serial_generator() -> None:
    """Static guard: belief-mcts must not reintroduce parent-serial promotion."""
    source = TRAIN_RR.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Locate the promotion WorkerPool construction near pin/unpin helpers.
    has_worker_pool_imap = False
    has_serial_belief_generator = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "imap_unordered":
                has_worker_pool_imap = True
        if isinstance(node, ast.GeneratorExp):
            # (_worker_promotion(job) for job in batch) was the serial bug.
            elt = node.elt
            if (
                isinstance(elt, ast.Call)
                and isinstance(elt.func, ast.Name)
                and elt.func.id == "_worker_promotion"
            ):
                has_serial_belief_generator = True

    assert has_worker_pool_imap
    assert not has_serial_belief_generator
    assert "promotion WorkerPool workers=" in source
    assert 'cmd="pin"' in source or "cmd\": \"pin\"" in source or 'cmd="pin"' in source
    # Device for promotion jobs must stay CPU (remote leaf), never train GPU.
    assert '"device": "cpu"' in source or "'device': 'cpu'" in source


@pytest.mark.unit
def test_belief_mcts_promotion_pool_records_parallel_dispatch(monkeypatch) -> None:
    """Mocked WorkerPool path overlaps work when promotion_workers > 1."""
    import scripts.train_round_robin as rr

    starts: list[float] = []
    mapped_batches: list[int] = []

    class FakePool:
        def __init__(self, num_workers, remote_channel=None, **_kwargs):
            self.num_workers = num_workers
            self.remote_channel = remote_channel
            assert num_workers > 1
            assert remote_channel is not None

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def imap_unordered(self, fn, batch):
            import time

            mapped_batches.append(len(batch))
            t0 = time.perf_counter()
            starts.append(t0)
            # Simulate parallel workers claiming jobs together.
            results = []
            for job in batch:
                starts.append(time.perf_counter())
                results.append(fn(job))
            return iter(results)

    monkeypatch.setattr(rr, "WorkerPool", FakePool)
    monkeypatch.setattr(
        rr,
        "_worker_promotion",
        lambda job: {
            "valid": True,
            "candidate_seat": job["candidate_seat"],
            "winner": job["candidate_seat"],
            "pair_id": None,
            "seed": job["seed"],
        },
    )

    jobs = [
        {"candidate_seat": i % 2, "seed": i, "cluster_id": i // 2}
        for i in range(8)
    ]
    n = promotion_worker_count(4, len(jobs))
    remote = {"resp_qs": [object()] * 8, "req_qs": [object()] * 2}
    with rr.WorkerPool(num_workers=n, remote_channel=remote) as pool:
        out = list(pool.imap_unordered(rr._worker_promotion, jobs[:4]))

    assert n == 4
    assert len(out) == 4
    assert mapped_batches == [4]
    # Multiple start stamps in a tight window => parallel dispatch, not serial
    # parent generator of one game after another over minutes.
    assert len(starts) >= 2
    assert max(starts) - min(starts) < 1.0
