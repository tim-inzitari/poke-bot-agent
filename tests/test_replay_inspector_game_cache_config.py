from __future__ import annotations

import json
from pathlib import Path

import pytest

from replay_inspector.config import (
    DEFAULT_GAME_TRACE_CACHE_MAX_BYTES,
    DEFAULT_GAME_TRACE_CACHE_MAX_ENTRY_BYTES,
    DEFAULT_GAME_TRACE_CACHE_MAX_GAME_BYTES,
    DEFAULT_GAME_TRACE_CACHE_MIN_FREE_BYTES,
    DEFAULT_GAME_TRACE_CACHE_ROOT,
    InspectorConfig,
)

ROOT = Path(__file__).resolve().parents[1]


def test_game_trace_cache_default_constructor_is_backward_compatible() -> None:
    config = InspectorConfig()

    assert config.game_trace_cache_root == DEFAULT_GAME_TRACE_CACHE_ROOT
    assert config.game_trace_cache_enabled is True
    assert config.game_trace_cache_max_bytes == 128 * 1024 * 1024
    assert config.game_trace_cache_max_game_bytes == 96 * 1024 * 1024
    assert config.game_trace_cache_max_entry_bytes == 8 * 1024 * 1024
    assert config.game_trace_cache_min_free_bytes == 64 * 1024 * 1024


def test_game_trace_cache_defaults_are_bounded_and_public(tmp_path: Path) -> None:
    """Constructing frozen configuration does not create the temporary root."""

    cache_root = Path("/tmp") / "pokebot-inspector-config-tests" / tmp_path.name
    assert not cache_root.exists()

    config = InspectorConfig(game_trace_cache_root=cache_root)

    assert config.game_trace_cache_enabled is True
    assert config.game_trace_cache_root == cache_root
    assert config.game_trace_cache_max_bytes == DEFAULT_GAME_TRACE_CACHE_MAX_BYTES
    assert config.game_trace_cache_max_game_bytes == DEFAULT_GAME_TRACE_CACHE_MAX_GAME_BYTES
    assert config.game_trace_cache_max_entry_bytes == DEFAULT_GAME_TRACE_CACHE_MAX_ENTRY_BYTES
    assert config.game_trace_cache_min_free_bytes == DEFAULT_GAME_TRACE_CACHE_MIN_FREE_BYTES
    assert not cache_root.exists()
    assert config.public_summary()["game_trace_cache"] == {
        "enabled": True,
        "root": str(cache_root),
        "max_bytes": 128 * 1024 * 1024,
        "max_game_bytes": 96 * 1024 * 1024,
        "max_entry_bytes": 8 * 1024 * 1024,
        "min_free_bytes": 64 * 1024 * 1024,
    }


@pytest.mark.parametrize(
    "cache_root",
    (
        Path("/tmp"),
        Path("/var/tmp/pokebot-replay-inspector-game-cache"),
        Path("/tmp/../var/cache/pokebot-replay-inspector-game-cache"),
    ),
)
def test_game_trace_cache_root_must_be_strictly_below_tmp(cache_root: Path) -> None:
    with pytest.raises(ValueError, match="strictly beneath /tmp"):
        InspectorConfig(game_trace_cache_root=cache_root)


def test_game_trace_cache_mapping_and_environment_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "inspector.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(
        "POKEBOT_REPLAY_INSPECTOR_GAME_TRACE_CACHE_ROOT",
        "/tmp/pokebot-inspector-config-env-cache",
    )
    monkeypatch.setenv("POKEBOT_REPLAY_INSPECTOR_GAME_TRACE_CACHE_ENABLED", "false")
    monkeypatch.setenv("POKEBOT_REPLAY_INSPECTOR_GAME_TRACE_CACHE_MAX_BYTES", "1000")
    monkeypatch.setenv(
        "POKEBOT_REPLAY_INSPECTOR_GAME_TRACE_CACHE_MAX_GAME_BYTES", "900"
    )
    monkeypatch.setenv(
        "POKEBOT_REPLAY_INSPECTOR_GAME_TRACE_CACHE_MAX_ENTRY_BYTES", "800"
    )
    monkeypatch.setenv(
        "POKEBOT_REPLAY_INSPECTOR_GAME_TRACE_CACHE_MIN_FREE_BYTES", "700"
    )

    config = InspectorConfig.load(config_path)

    assert config.game_trace_cache_root == Path(
        "/tmp/pokebot-inspector-config-env-cache"
    )
    assert config.game_trace_cache_enabled is False
    assert config.game_trace_cache_max_bytes == 1000
    assert config.game_trace_cache_max_game_bytes == 900
    assert config.game_trace_cache_max_entry_bytes == 800
    assert config.game_trace_cache_min_free_bytes == 700


def test_game_trace_cache_limits_are_nested() -> None:
    with pytest.raises(ValueError, match="max_game_bytes"):
        InspectorConfig(
            game_trace_cache_max_bytes=100,
            game_trace_cache_max_game_bytes=101,
        )
    with pytest.raises(ValueError, match="max_entry_bytes"):
        InspectorConfig(
            game_trace_cache_max_game_bytes=100,
            game_trace_cache_max_entry_bytes=101,
        )


def test_elmo_config_declares_the_tmpfs_safe_game_cache_budget() -> None:
    raw = json.loads(
        (
            ROOT / "ops" / "elmo" / "replay-model-inspector-config-r176.json"
        ).read_text(encoding="utf-8")
    )
    config = InspectorConfig.from_mapping(raw)

    assert config.game_trace_cache_root == DEFAULT_GAME_TRACE_CACHE_ROOT
    assert config.game_trace_cache_enabled is True
    assert config.game_trace_cache_max_bytes == 128 * 1024 * 1024
    assert config.game_trace_cache_max_game_bytes == 96 * 1024 * 1024
    assert config.game_trace_cache_max_entry_bytes == 8 * 1024 * 1024
    assert config.game_trace_cache_min_free_bytes == 64 * 1024 * 1024
