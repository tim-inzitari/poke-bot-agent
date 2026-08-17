from __future__ import annotations

from poke_bot.remote_jobs import endpoint_dispatch_chunk


def test_endpoint_dispatch_chunks_are_host_specific(monkeypatch) -> None:
    monkeypatch.setenv(
        "POKEBOT_REMOTE_ENDPOINT_CHUNKS",
        "elmo:8765=48,bert.local:8766=16",
    )
    assert endpoint_dispatch_chunk("elmo", 8765, default=8) == 48
    assert endpoint_dispatch_chunk("bert.local", 8766, default=8) == 16
    assert endpoint_dispatch_chunk("unknown.local", 9999, default=12) == 12
