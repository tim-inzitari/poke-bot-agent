from __future__ import annotations

from types import SimpleNamespace

import scripts.collect_top_ladder_replays as collector


def test_recognized_only_conversion_keeps_registered_additive_archetypes(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClassifier:
        active_ids = ("starmie",)
        additive_registered_ids = ("walrein",)

        def classify_episode(self, _payload):
            return [list(range(60)), list(range(60))], [
                SimpleNamespace(deck_id="walrein", method="registered_signature"),
                SimpleNamespace(deck_id="unknown", method="unrecognized"),
            ]

    def fake_convert(_payload, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(collector, "_WORKER_CLASSIFIER", FakeClassifier())
    monkeypatch.setattr(collector, "_load_payload", lambda _ref: {"id": "episode"})
    monkeypatch.setattr(collector, "convert_episode_to_records", fake_convert)

    result = collector._convert_ref(("episode.json", "source", True, ""))

    assert result["error"] is None
    assert captured["allowed_archetypes"] == ("starmie", "walrein")
