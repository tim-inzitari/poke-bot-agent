"""Configuration for the standalone replay/model inspector."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_REPLAY_ROOT = Path("/srv/poke-bot-agent/archive/submission-replays")
DEFAULT_ROLLOUT_ROOT = Path(
    "/srv/poke-bot-agent/archive/submission-replay-rollouts"
)
DEFAULT_GAME_TRACE_CACHE_ROOT = Path("/tmp/pokebot-replay-inspector-game-cache-v1")
_MEBIBYTE = 1024 * 1024
DEFAULT_GAME_TRACE_CACHE_MAX_BYTES = 128 * _MEBIBYTE
DEFAULT_GAME_TRACE_CACHE_MAX_GAME_BYTES = 96 * _MEBIBYTE
DEFAULT_GAME_TRACE_CACHE_MAX_ENTRY_BYTES = 8 * _MEBIBYTE
DEFAULT_GAME_TRACE_CACHE_MIN_FREE_BYTES = 64 * _MEBIBYTE
_TEMPORARY_CACHE_PARENT = Path("/tmp")


def _path(value: object, *, base: Path | None = None) -> Path:
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute() and base is not None:
        candidate = base / candidate
    return candidate.absolute()


def _optional_path(value: object, *, base: Path | None = None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return _path(value, base=base)


def _list_paths(value: object, *, base: Path | None = None) -> tuple[Path, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, os.PathLike)):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise TypeError("artifact_roots must be a path or list of paths")
    return tuple(_path(item, base=base) for item in values)


def _temporary_cache_root(value: object, *, base: Path | None = None) -> Path:
    """Return a lexical, private cache root strictly below ``/tmp``.

    Configuration must not create a cache directory or resolve it: the latter
    could follow an attacker-controlled symlink before the cache owner applies
    its runtime symlink policy.  ``abspath`` only normalises lexical ``..``
    components and does not inspect the filesystem.
    """

    candidate = _path(value, base=base)
    lexical = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = lexical.relative_to(_TEMPORARY_CACHE_PARENT)
    except ValueError as exc:
        raise ValueError(
            "game_trace_cache_root must be strictly beneath /tmp"
        ) from exc
    if not relative.parts:
        raise ValueError("game_trace_cache_root must be strictly beneath /tmp")
    return lexical


def _cache_byte_limit(value: object, *, field_name: str, minimum: int) -> int:
    """Normalize a byte setting while rejecting bools and invalid bounds."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer byte count")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer byte count") from exc
    if parsed < minimum:
        comparator = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{field_name} must be {comparator}")
    return parsed


def _cache_enabled(value: object) -> bool:
    """Accept JSON booleans and conventional environment boolean values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError("game_trace_cache_enabled must be a boolean")


@dataclass(frozen=True)
class InspectorConfig:
    """One immutable local-service configuration.

    Source roots are never created by this class.  A missing source leaves the
    site available with an explicit diagnostic so an unmounted NAS does not
    turn into a misleading empty success.  ``runtime_source_root`` must point
    at the extracted submitted package used first on ``PYTHONPATH``; it is
    rehashed against a checksum-bound parity receipt before any trace runs.
    """

    bind_host: str = "127.0.0.1"
    port: int = 8791
    replay_root: Path = DEFAULT_REPLAY_ROOT
    rollout_root: Path = DEFAULT_ROLLOUT_ROOT
    provenance_manifest: Path | None = None
    training_recipe_registry: Path | None = None
    artifact_roots: tuple[Path, ...] = field(default_factory=tuple)
    runtime_source_root: Path | None = None
    web_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "web"
    )
    torch_threads: int = 4
    max_parameter_slice: int = 2048
    max_tensor_values: int = 512
    verify_digests: bool = True
    # Derived, disposable inspection payloads only.  Keep these fields last so
    # existing positional ``InspectorConfig`` construction remains compatible.
    game_trace_cache_root: Path = DEFAULT_GAME_TRACE_CACHE_ROOT
    game_trace_cache_enabled: bool = True
    game_trace_cache_max_bytes: int = DEFAULT_GAME_TRACE_CACHE_MAX_BYTES
    game_trace_cache_max_game_bytes: int = DEFAULT_GAME_TRACE_CACHE_MAX_GAME_BYTES
    game_trace_cache_max_entry_bytes: int = DEFAULT_GAME_TRACE_CACHE_MAX_ENTRY_BYTES
    game_trace_cache_min_free_bytes: int = DEFAULT_GAME_TRACE_CACHE_MIN_FREE_BYTES

    def __post_init__(self) -> None:
        if not self.bind_host.strip():
            raise ValueError("bind_host must not be empty")
        if not (1 <= int(self.port) <= 65535):
            raise ValueError("port must be in 1..65535")
        if int(self.torch_threads) < 1 or int(self.torch_threads) > 32:
            raise ValueError("torch_threads must be in 1..32")
        if int(self.max_parameter_slice) < 1 or int(self.max_parameter_slice) > 8192:
            raise ValueError("max_parameter_slice must be in 1..8192")
        if int(self.max_tensor_values) < 1 or int(self.max_tensor_values) > 4096:
            raise ValueError("max_tensor_values must be in 1..4096")

        # Do not resolve or create this path here.  The cache implementation
        # owns runtime directory creation and symlink-safe containment checks.
        object.__setattr__(
            self,
            "game_trace_cache_root",
            _temporary_cache_root(self.game_trace_cache_root),
        )
        object.__setattr__(
            self,
            "game_trace_cache_enabled",
            _cache_enabled(self.game_trace_cache_enabled),
        )
        max_bytes = _cache_byte_limit(
            self.game_trace_cache_max_bytes,
            field_name="game_trace_cache_max_bytes",
            minimum=1,
        )
        max_game_bytes = _cache_byte_limit(
            self.game_trace_cache_max_game_bytes,
            field_name="game_trace_cache_max_game_bytes",
            minimum=1,
        )
        max_entry_bytes = _cache_byte_limit(
            self.game_trace_cache_max_entry_bytes,
            field_name="game_trace_cache_max_entry_bytes",
            minimum=1,
        )
        min_free_bytes = _cache_byte_limit(
            self.game_trace_cache_min_free_bytes,
            field_name="game_trace_cache_min_free_bytes",
            minimum=0,
        )
        if max_game_bytes > max_bytes:
            raise ValueError(
                "game_trace_cache_max_game_bytes must not exceed "
                "game_trace_cache_max_bytes"
            )
        if max_entry_bytes > max_game_bytes:
            raise ValueError(
                "game_trace_cache_max_entry_bytes must not exceed "
                "game_trace_cache_max_game_bytes"
            )
        object.__setattr__(self, "game_trace_cache_max_bytes", max_bytes)
        object.__setattr__(self, "game_trace_cache_max_game_bytes", max_game_bytes)
        object.__setattr__(self, "game_trace_cache_max_entry_bytes", max_entry_bytes)
        object.__setattr__(self, "game_trace_cache_min_free_bytes", min_free_bytes)

    @property
    def source_roots(self) -> tuple[Path, ...]:
        roots = [
            self.replay_root,
            self.rollout_root,
            *self.artifact_roots,
            *(() if self.runtime_source_root is None else (self.runtime_source_root,)),
        ]
        deduped: list[Path] = []
        for root in roots:
            if root not in deduped:
                deduped.append(root)
        return tuple(deduped)

    def public_summary(self) -> dict[str, Any]:
        """Return non-secret status suitable for the localhost health API."""

        return {
            "bind_host": self.bind_host,
            "port": self.port,
            "replay_root": str(self.replay_root),
            "replay_root_available": self.replay_root.is_dir(),
            "rollout_root": str(self.rollout_root),
            "rollout_root_available": self.rollout_root.is_dir(),
            "provenance_manifest": (
                str(self.provenance_manifest)
                if self.provenance_manifest is not None
                else None
            ),
            "provenance_manifest_available": bool(
                self.provenance_manifest is not None
                and self.provenance_manifest.is_file()
            ),
            "training_recipe_registry": (
                str(self.training_recipe_registry)
                if self.training_recipe_registry is not None
                else None
            ),
            "training_recipe_registry_available": bool(
                self.training_recipe_registry is not None
                and self.training_recipe_registry.is_file()
            ),
            "artifact_root_count": len(self.artifact_roots),
            "runtime_source_root_configured": self.runtime_source_root is not None,
            "runtime_source_root_available": bool(
                self.runtime_source_root is not None
                and self.runtime_source_root.is_dir()
            ),
            "web_root_available": self.web_root.is_dir(),
            "torch_threads": self.torch_threads,
            "verify_digests": self.verify_digests,
            "game_trace_cache": {
                "enabled": self.game_trace_cache_enabled,
                "root": str(self.game_trace_cache_root),
                "max_bytes": self.game_trace_cache_max_bytes,
                "max_game_bytes": self.game_trace_cache_max_game_bytes,
                "max_entry_bytes": self.game_trace_cache_max_entry_bytes,
                "min_free_bytes": self.game_trace_cache_min_free_bytes,
            },
        }

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        base: Path | None = None,
    ) -> InspectorConfig:
        allowed = {
            "bind_host",
            "port",
            "replay_root",
            "rollout_root",
            "provenance_manifest",
            "training_recipe_registry",
            "artifact_roots",
            "runtime_source_root",
            "web_root",
            "torch_threads",
            "max_parameter_slice",
            "max_tensor_values",
            "verify_digests",
            "game_trace_cache_root",
            "game_trace_cache_enabled",
            "game_trace_cache_max_bytes",
            "game_trace_cache_max_game_bytes",
            "game_trace_cache_max_entry_bytes",
            "game_trace_cache_min_free_bytes",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown inspector config fields: {unknown}")
        values: dict[str, Any] = dict(raw)
        if "replay_root" in values:
            values["replay_root"] = _path(values["replay_root"], base=base)
        if "rollout_root" in values:
            values["rollout_root"] = _path(values["rollout_root"], base=base)
        if "provenance_manifest" in values:
            values["provenance_manifest"] = _optional_path(
                values["provenance_manifest"], base=base
            )
        if "training_recipe_registry" in values:
            values["training_recipe_registry"] = _optional_path(
                values["training_recipe_registry"], base=base
            )
        if "artifact_roots" in values:
            values["artifact_roots"] = _list_paths(values["artifact_roots"], base=base)
        if "runtime_source_root" in values:
            values["runtime_source_root"] = _optional_path(
                values["runtime_source_root"], base=base
            )
        if "web_root" in values:
            values["web_root"] = _path(values["web_root"], base=base)
        if "game_trace_cache_root" in values:
            values["game_trace_cache_root"] = _temporary_cache_root(
                values["game_trace_cache_root"], base=base
            )
        return cls(**values)

    @classmethod
    def load(cls, path: Path | None = None) -> InspectorConfig:
        raw: dict[str, Any] = {}
        base: Path | None = None
        if path is not None:
            config_path = path.expanduser().absolute()
            base = config_path.parent
            if config_path.is_file():
                parsed = json.loads(config_path.read_text(encoding="utf-8"))
                if not isinstance(parsed, dict):
                    raise ValueError("inspector config must be a JSON object")
                raw.update(parsed)
            else:
                raise FileNotFoundError(config_path)

        env_map = {
            "POKEBOT_REPLAY_INSPECTOR_HOST": "bind_host",
            "POKEBOT_REPLAY_INSPECTOR_PORT": "port",
            "POKEBOT_SUBMISSION_REPLAY_ARCHIVE": "replay_root",
            "POKEBOT_SUBMISSION_REPLAY_ROLLOUT_ROOT": "rollout_root",
            "POKEBOT_REPLAY_INSPECTOR_PROVENANCE": "provenance_manifest",
            "POKEBOT_REPLAY_INSPECTOR_TRAINING_RECIPE_REGISTRY": (
                "training_recipe_registry"
            ),
            "POKEBOT_REPLAY_INSPECTOR_RUNTIME_SOURCE_ROOT": "runtime_source_root",
            "POKEBOT_REPLAY_INSPECTOR_WEB_ROOT": "web_root",
            "POKEBOT_REPLAY_INSPECTOR_TORCH_THREADS": "torch_threads",
            "POKEBOT_REPLAY_INSPECTOR_GAME_TRACE_CACHE_ROOT": (
                "game_trace_cache_root"
            ),
            "POKEBOT_REPLAY_INSPECTOR_GAME_TRACE_CACHE_ENABLED": (
                "game_trace_cache_enabled"
            ),
            "POKEBOT_REPLAY_INSPECTOR_GAME_TRACE_CACHE_MAX_BYTES": (
                "game_trace_cache_max_bytes"
            ),
            "POKEBOT_REPLAY_INSPECTOR_GAME_TRACE_CACHE_MAX_GAME_BYTES": (
                "game_trace_cache_max_game_bytes"
            ),
            "POKEBOT_REPLAY_INSPECTOR_GAME_TRACE_CACHE_MAX_ENTRY_BYTES": (
                "game_trace_cache_max_entry_bytes"
            ),
            "POKEBOT_REPLAY_INSPECTOR_GAME_TRACE_CACHE_MIN_FREE_BYTES": (
                "game_trace_cache_min_free_bytes"
            ),
        }
        for env_name, field_name in env_map.items():
            if env_name in os.environ:
                raw[field_name] = os.environ[env_name]
        if "POKEBOT_REPLAY_INSPECTOR_ARTIFACT_ROOTS" in os.environ:
            raw["artifact_roots"] = [
                item
                for item in os.environ["POKEBOT_REPLAY_INSPECTOR_ARTIFACT_ROOTS"].split(
                    os.pathsep
                )
                if item
            ]
        for integer in (
            "port",
            "torch_threads",
            "max_parameter_slice",
            "max_tensor_values",
            "game_trace_cache_max_bytes",
            "game_trace_cache_max_game_bytes",
            "game_trace_cache_max_entry_bytes",
            "game_trace_cache_min_free_bytes",
        ):
            if integer in raw:
                raw[integer] = int(raw[integer])
        return cls.from_mapping(raw, base=base)
