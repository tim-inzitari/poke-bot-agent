"""Configuration for the standalone replay/model inspector."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_REPLAY_ROOT = Path("/mnt/Main/main/poke-bot-agent/archive/submission-replays")
DEFAULT_ROLLOUT_ROOT = Path(
    "/mnt/Main/main/poke-bot-agent/archive/submission-replay-rollouts"
)


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
        ):
            if integer in raw:
                raw[integer] = int(raw[integer])
        return cls.from_mapping(raw, base=base)
