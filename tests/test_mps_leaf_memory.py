"""Memory-containment regressions for the persistent Apple MPS leaf."""

from __future__ import annotations

from poke_bot import batched_infer


def test_mps_cache_release_interval_is_safe_by_default(monkeypatch) -> None:
    monkeypatch.delenv("POKEBOT_MPS_EMPTY_CACHE_EVERY_BATCHES", raising=False)
    assert batched_infer._mps_cache_release_interval() == 1

    monkeypatch.setenv("POKEBOT_MPS_EMPTY_CACHE_EVERY_BATCHES", "8")
    assert batched_infer._mps_cache_release_interval() == 8

    monkeypatch.setenv("POKEBOT_MPS_EMPTY_CACHE_EVERY_BATCHES", "0")
    assert batched_infer._mps_cache_release_interval() == 0

    monkeypatch.setenv("POKEBOT_MPS_EMPTY_CACHE_EVERY_BATCHES", "invalid")
    assert batched_infer._mps_cache_release_interval() == 1


def test_release_mps_cache_collects_then_synchronizes_then_empties(
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        batched_infer.gc, "collect", lambda: calls.append("collect") or 0
    )
    monkeypatch.setattr(
        batched_infer.torch.mps,
        "synchronize",
        lambda: calls.append("synchronize"),
    )
    monkeypatch.setattr(
        batched_infer.torch.mps,
        "empty_cache",
        lambda: calls.append("empty_cache"),
    )

    batched_infer._release_mps_cache(collect=True)

    assert calls == ["collect", "synchronize", "empty_cache"]


def test_release_mps_cache_is_best_effort(monkeypatch) -> None:
    monkeypatch.setattr(
        batched_infer.torch.mps,
        "synchronize",
        lambda: (_ for _ in ()).throw(RuntimeError("not initialized")),
    )
    emptied: list[bool] = []
    monkeypatch.setattr(
        batched_infer.torch.mps,
        "empty_cache",
        lambda: emptied.append(True),
    )

    batched_infer._release_mps_cache()

    assert emptied == [True]
