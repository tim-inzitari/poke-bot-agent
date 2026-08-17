from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.train_pure_rl import _load_frozen_specialist_registry


def _registry(path: Path, *, research_eligible: bool) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "poke_bot.frozen_specialist_registry/v1",
                "version": 1,
                "specialists": [
                    {
                        "opponent_id": "specialist-alakazam",
                        "archetype_id": "alakazam",
                        "checkpoint_digest": "sha256:" + "1" * 64,
                        "content_digest": "sha256:" + "2" * 64,
                        "frozen": True,
                        "public_mix_eligible": True,
                        "research_eligible": research_eligible,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_frozen_specialist_is_excluded_from_research_controls(
    tmp_path: Path,
) -> None:
    loaded = _load_frozen_specialist_registry(
        _registry(tmp_path / "registry.json", research_eligible=False)
    )
    assert loaded["specialists"][0]["research_eligible"] is False


def test_frozen_specialist_cannot_enter_research_control_registry(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="invalid frozen specialist"):
        _load_frozen_specialist_registry(
            _registry(tmp_path / "registry.json", research_eligible=True)
        )
