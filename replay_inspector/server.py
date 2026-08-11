"""Loopback-only HTTP server for the standalone Replay Model Inspector.

This module deliberately has no dependency on :mod:`dashboard`.  It reads the
configured replay archive and explicit provenance records, but never accepts a
filesystem path from an HTTP client and never writes to a replay, checkpoint,
bundle, or provenance root.

The model-runtime import is intentionally lazy.  That lets an operator browse
the causal replay cache on a small host even when the exact submitted package
or CPU PyTorch runtime is not yet available.  In that case the API returns a
precise ``unavailable`` result rather than selecting a similarly named current
checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib
import ipaddress
import json
import logging
import math
import os
import re
import select
import socket
import stat
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from poke_bot import cg_env, features

from .catalog import (
    ReplayArchiveCatalog,
    ReplayCatalogEntry,
    SubmissionCatalogEntry,
    scan_archive,
)
from .config import InspectorConfig
from .provenance import (
    ProvenanceError,
    ProvenanceManifest,
    SubmissionProvenance,
    load_provenance_manifest,
    resolve_contained_path,
    sha256_file,
    sha256_source_tree,
)
from .timeline import (
    DecisionStage,
    ReplayTimelineError,
    causal_observation,
    decision_timeline,
)
from .training_recipe import (
    TrainingRecipeRegistry,
    TrainingRecipeRegistryError,
    load_training_recipe_registry,
)

API_SCHEMA = "poke_bot.replay_model_inspector.api/v1"
SERVICE_NAME = "replay-model-inspector"
GAME_TRACE_MATERIALIZATION_SCHEMA = (
    "poke_bot.replay_model_inspector.physical_game_materialization/v1"
)
GAME_MATERIALIZATION_WORKER_PROTOCOL = (
    "poke_bot.replay_model_inspector.game_materialization_worker/v1"
)
_ISOLATED_GAME_WORKER_ENV = "POKEBOT_REPLAY_INSPECTOR_GAME_WORKER"
_ISOLATED_GAME_WORKER_POLL_SECONDS = 1.0
_ISOLATED_GAME_WORKER_STALL_SECONDS = 240.0
_ISOLATED_GAME_WORKER_MAX_FRAME_BYTES = 32 * 1024 * 1024
_LOG = logging.getLogger(__name__)
_DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
_HEAD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_STATIC_ASSETS: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class InspectorHTTPError(ValueError):
    """A safe, JSON-serializable HTTP problem."""

    def __init__(
        self,
        status: int | HTTPStatus,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = int(status)
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details) if details else None


@dataclass(frozen=True, slots=True)
class ReplayContext:
    """One resolved, causal replay address with no model result attached."""

    submission: SubmissionCatalogEntry
    replay_entry: ReplayCatalogEntry
    replay: dict[str, Any]
    replay_sha256: str
    seat: int
    stage: DecisionStage
    observation: dict[str, Any]


@dataclass(slots=True)
class _GameMaterializationJob:
    """One identity-bound baseline game reconstruction in progress or memory.

    The job deliberately retains only final, server-produced trace payloads.
    It never represents a replay/model artifact and is only an in-process L1
    cache; the bounded private temporary L2 cache is owned separately by the
    game-trace-cache layer.  A waiter can promote an as-yet-uncomputed address
    to the front of ``pending`` but can never cancel an active exact forward.
    """

    identity_key: str
    identity: dict[str, Any]
    addresses: tuple[tuple[int, int], ...]
    pending: list[tuple[int, int]]
    payloads: dict[tuple[int, int], dict[str, Any]]
    active_address: tuple[int, int] | None = None
    complete: bool = False
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class _ImmutablePathSignature:
    """A stat-based identity for one resolved, immutable artifact path.

    The cache is only a performance hint.  Device/inode prevent a replacement
    at the same spelling from looking unchanged, while size, timestamps, mode,
    and link count make ordinary in-place mutation and metadata changes a
    cache miss as well.
    """

    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    nlink: int


@dataclass(frozen=True, slots=True)
class _ImmutableTreeSignature:
    """One resolved runtime tree's checked structure and member signatures."""

    root: _ImmutablePathSignature
    entries: tuple[tuple[str, _ImmutablePathSignature], ...]


class _IntegrityDigestMemo:
    """Bounded digest memo which never skips a current integrity gate.

    A cache lookup resolves the supplied path under the current configured
    roots and restats it before returning a digest.  Runtime trees additionally
    restat every non-ignored member before their digest can be reused.  A
    signature drift during a fresh hash is rejected rather than cached, so a
    caller must fail closed and retry against a stable exact input.

    This class holds only derived digest strings and immutable stat signatures;
    it never retains artifact bytes, opens a writable path, or decides whether
    an expected provenance digest is acceptable.
    """

    _DEFAULT_MAX_ENTRIES = 128

    def __init__(self, *, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self._max_entries = max(1, int(max_entries))
        self._lock = threading.RLock()
        self._file_digests: OrderedDict[
            Path, tuple[_ImmutablePathSignature, str]
        ] = OrderedDict()
        self._tree_digests: OrderedDict[
            Path, tuple[_ImmutableTreeSignature, str]
        ] = OrderedDict()

    @staticmethod
    def _signature(path: Path) -> _ImmutablePathSignature | None:
        try:
            details = path.lstat()
        except OSError:
            return None
        if stat.S_ISLNK(details.st_mode):
            return None
        return _ImmutablePathSignature(
            device=int(details.st_dev),
            inode=int(details.st_ino),
            mode=int(details.st_mode),
            size=int(details.st_size),
            mtime_ns=int(details.st_mtime_ns),
            ctime_ns=int(details.st_ctime_ns),
            nlink=int(details.st_nlink),
        )

    @classmethod
    def _resolved_regular_file(
        cls,
        value: Path,
        *,
        roots: Sequence[Path],
    ) -> tuple[Path, _ImmutablePathSignature] | None:
        try:
            resolved = resolve_contained_path(
                value,
                roots=roots,
                require_exists=True,
            )
        except (FileNotFoundError, OSError, ProvenanceError, RuntimeError):
            return None
        signature = cls._signature(resolved)
        if signature is None or not stat.S_ISREG(signature.mode):
            return None
        return resolved, signature

    @classmethod
    def _resolved_tree(
        cls,
        value: Path,
        *,
        roots: Sequence[Path],
    ) -> tuple[Path, _ImmutableTreeSignature] | None:
        try:
            root = resolve_contained_path(
                value,
                roots=roots,
                require_exists=True,
            )
        except (FileNotFoundError, OSError, ProvenanceError, RuntimeError):
            return None
        root_signature = cls._signature(root)
        if root_signature is None or not stat.S_ISDIR(root_signature.mode):
            return None

        entries: list[tuple[str, _ImmutablePathSignature]] = []
        pending = [root]
        try:
            while pending:
                directory = pending.pop()
                children = sorted(directory.iterdir(), key=lambda item: item.name)
                for candidate in children:
                    relative = candidate.relative_to(root)
                    # Match ``sha256_source_tree`` exactly: Python bytecode is
                    # derived and intentionally excluded from the attested
                    # source tree, so it cannot make this memo stale.
                    if (
                        "__pycache__" in relative.parts
                        or candidate.suffix in {".pyc", ".pyo"}
                    ):
                        continue
                    signature = cls._signature(candidate)
                    if signature is None:
                        return None
                    relative_text = relative.as_posix()
                    if stat.S_ISDIR(signature.mode):
                        entries.append(("dir:" + relative_text, signature))
                        pending.append(candidate)
                    elif stat.S_ISREG(signature.mode):
                        entries.append(("file:" + relative_text, signature))
                    else:
                        return None
        except (OSError, RuntimeError, ValueError):
            return None
        return root, _ImmutableTreeSignature(
            root=root_signature,
            entries=tuple(sorted(entries, key=lambda item: item[0])),
        )

    @staticmethod
    def _remember(
        store: OrderedDict[Any, Any],
        key: Path,
        value: Any,
        *,
        maximum: int,
    ) -> None:
        store[key] = value
        store.move_to_end(key)
        while len(store) > maximum:
            store.popitem(last=False)

    def file_digest(
        self,
        value: Path,
        *,
        roots: Sequence[Path],
    ) -> tuple[Path, str] | None:
        """Return a stable regular-file digest after current containment checks."""

        current = self._resolved_regular_file(value, roots=roots)
        if current is None:
            return None
        resolved, signature = current
        with self._lock:
            cached = self._file_digests.get(resolved)
            if cached is not None and cached[0] == signature:
                self._file_digests.move_to_end(resolved)
                return resolved, cached[1]
            try:
                digest = sha256_file(resolved)
            except OSError:
                return None
            verified = self._resolved_regular_file(resolved, roots=roots)
            if verified is None or verified != current:
                self._file_digests.pop(resolved, None)
                return None
            self._remember(
                self._file_digests,
                resolved,
                (signature, digest),
                maximum=self._max_entries,
            )
            return resolved, digest

    def source_tree_digest(
        self,
        value: Path,
        *,
        roots: Sequence[Path],
    ) -> tuple[Path, str] | None:
        """Return a stable source-tree digest after current containment checks."""

        current = self._resolved_tree(value, roots=roots)
        if current is None:
            return None
        resolved, signature = current
        with self._lock:
            cached = self._tree_digests.get(resolved)
            if cached is not None and cached[0] == signature:
                self._tree_digests.move_to_end(resolved)
                return resolved, cached[1]
            try:
                digest = sha256_source_tree(resolved)
            except (OSError, ProvenanceError, RuntimeError):
                return None
            verified = self._resolved_tree(resolved, roots=roots)
            if verified is None or verified != current:
                self._tree_digests.pop(resolved, None)
                return None
            self._remember(
                self._tree_digests,
                resolved,
                (signature, digest),
                maximum=self._max_entries,
            )
            return resolved, digest


def _is_loopback_host(value: str) -> bool:
    """Accept only a literal loopback address or the conventional localhost."""

    host = str(value).strip()
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _require_loopback_host(value: str) -> str:
    host = str(value).strip()
    if not _is_loopback_host(host):
        raise ValueError(
            "Replay Model Inspector is loopback-only; bind_host must be "
            "127.0.0.1, ::1, or localhost"
        )
    return host


def _host_header_is_loopback(value: str) -> bool:
    """Validate an HTTP Host header without trusting a lexical prefix."""

    if not value or any(character.isspace() for character in value):
        return False
    try:
        parsed = urlsplit("//" + value)
        # Accessing .port also validates malformed/overflowing port syntax.
        _ = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.hostname
        and parsed.username is None
        and parsed.password is None
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
        and _is_loopback_host(parsed.hostname)
    )


def _origin_is_loopback(value: str) -> bool:
    """Accept an explicit browser Origin only when it resolves to loopback."""

    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme in {"http", "https"}
        and parsed.hostname
        and parsed.username is None
        and parsed.password is None
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
        and _is_loopback_host(parsed.hostname)
    )


def _availability(available: bool, reason: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"available": bool(available)}
    if reason:
        payload["reason"] = str(reason)
    return payload


def _reason_from(values: Sequence[str] | None, fallback: str) -> str:
    for value in values or ():
        clean = str(value).strip()
        if clean:
            return clean
    return fallback


def _strict_int(value: str, *, field: str, minimum: int = 0) -> int:
    """Parse an identifier without accepting signs, floats, or whitespace."""

    if not _DECIMAL_RE.fullmatch(value):
        raise InspectorHTTPError(
            HTTPStatus.BAD_REQUEST,
            "invalid_identifier",
            f"{field} must be a decimal integer",
        )
    parsed = int(value)
    if parsed < minimum:
        raise InspectorHTTPError(
            HTTPStatus.BAD_REQUEST,
            "identifier_out_of_range",
            f"{field} must be at least {minimum}",
        )
    return parsed


def _submission_id_text(submission_id: int) -> str:
    """Render an internal submission id in the only public display format.

    JSON numbers beyond a browser's safe integer range can be silently shown
    in scientific notation or rounded.  Keep the numeric field for existing
    callers, but every public selection/summary also carries this exact
    decimal string for display and URL construction.
    """

    return format(int(submission_id), "d")


def _query_values(raw_query: str, *, allowed: set[str]) -> dict[str, str]:
    """Return one value per allowed query key, rejecting ambiguous requests."""

    try:
        parsed = parse_qs(raw_query, keep_blank_values=True, strict_parsing=False)
    except ValueError as exc:
        raise InspectorHTTPError(
            HTTPStatus.BAD_REQUEST,
            "invalid_query",
            "query string is malformed",
        ) from exc
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise InspectorHTTPError(
            HTTPStatus.BAD_REQUEST,
            "unknown_query_parameter",
            "request contains an unsupported query parameter",
            details={"parameters": unknown},
        )
    result: dict[str, str] = {}
    for key, values in parsed.items():
        if len(values) != 1 or values[0] == "":
            raise InspectorHTTPError(
                HTTPStatus.BAD_REQUEST,
                "ambiguous_query_parameter",
                f"{key} must be specified once with a non-empty value",
            )
        result[key] = values[0]
    return result


def _head_scales_query(value: str | None) -> dict[str, float] | None:
    """Parse the bounded GET representation ``head:scale,head:scale``."""

    if value is None:
        return None
    parts = value.split(",")
    if not parts or len(parts) > 32:
        raise InspectorHTTPError(
            HTTPStatus.BAD_REQUEST,
            "invalid_head_scales",
            "head scales must contain between 1 and 32 entries",
        )
    result: dict[str, float] = {}
    for part in parts:
        if part.count(":") != 1:
            raise InspectorHTTPError(
                HTTPStatus.BAD_REQUEST,
                "invalid_head_scales",
                "head scales use the form head:scale,head:scale",
            )
        name, raw_scale = part.split(":", 1)
        if _HEAD_NAME_RE.fullmatch(name) is None or raw_scale != raw_scale.strip():
            raise InspectorHTTPError(
                HTTPStatus.BAD_REQUEST,
                "invalid_head_scales",
                "head scale names or values are malformed",
            )
        if name in result:
            raise InspectorHTTPError(
                HTTPStatus.BAD_REQUEST,
                "duplicate_head_scale",
                "a head scale may be specified only once",
                details={"head": name},
            )
        try:
            scale = float(raw_scale)
        except ValueError as exc:
            raise InspectorHTTPError(
                HTTPStatus.BAD_REQUEST,
                "invalid_head_scales",
                "head scales must be finite numbers from 0 through 2",
            ) from exc
        if not math.isfinite(scale) or not 0.0 <= scale <= 2.0:
            raise InspectorHTTPError(
                HTTPStatus.BAD_REQUEST,
                "head_scale_out_of_range",
                "head scales must be between 0 and 2 inclusive",
                details={"head": name, "minimum": 0.0, "maximum": 2.0},
            )
        result[name] = scale
    return result


def _json_safe(value: Any) -> Any:
    """Keep an inference payload strict-JSON safe without exposing internals.

    Replay files are JSON already, while optional model backends sometimes
    return dataclass-like objects.  This helper handles only ordinary public
    containers and turns non-finite floats into explicit ``null`` values.
    Unknown objects are represented by their type, not ``repr`` (which can
    contain paths, tensor contents, or object addresses).
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    return {"unserializable_type": type(value).__name__}


def _metadata_value(metadata: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = metadata.get(name)
        if value not in (None, ""):
            return value
    return None


_ARCHIVED_SUBMISSION_IDENTITY = "Challengestone"
_ARCHIVED_SUBMISSION_IDENTITY_NORMALIZED = "challengestone"


def _normalized_identity(value: Any) -> str | None:
    """Normalize a source-provided display identity without inventing one."""

    if not isinstance(value, str) or not value.strip():
        return None
    return "".join(character for character in value if character.isalnum()).casefold()


def _decision_owner_payload(
    metadata: Mapping[str, Any] | None,
    *,
    submission_id: int,
    own_seat: int | None,
    seat_reason: str | None = None,
) -> dict[str, Any]:
    """Describe the only seat whose decisions may be selected.

    The seat remains authoritative only when it was explicitly recorded by the
    archive.  ``Challengestone`` is presentation corroboration, never a
    fallback mechanism for assigning a seat: an archive that names the agent
    but omits or contradicts its index remains unavailable for selection.
    """

    if own_seat is None:
        return {
            "availability": _availability(
                False, seat_reason or "own_seat_not_declared"
            ),
            "seat": None,
            "seat_authority": "archived_submission_bound_own_agent_seat",
            "selection_scope": "own_agent_decisions_only",
            "identity": {
                "available": False,
                "name": None,
                "availability_reasons": [
                    "challengestone_identity_not_selectable_without_archived_own_seat"
                ],
            },
        }

    claims: list[tuple[str, str]] = []
    source_metadata = metadata if isinstance(metadata, Mapping) else {}
    own_agent = source_metadata.get("own_agent")
    if isinstance(own_agent, Mapping):
        for field in ("team_name", "teamName", "name", "Name"):
            value = own_agent.get(field)
            if isinstance(value, str) and value.strip():
                claims.append((value, f"archived_episode_metadata.own_agent.{field}"))

    # An agents list is usable only for an exact submission-id and seat match.
    # List order and a text label alone are never identity-to-seat inference.
    agents = source_metadata.get("agents")
    if isinstance(agents, list):
        for agent in agents:
            if not isinstance(agent, Mapping):
                continue
            try:
                matches_submission = int(agent.get("submission_id")) == int(
                    submission_id
                )
                matches_seat = int(agent.get("index")) == int(own_seat)
            except (TypeError, ValueError, OverflowError):
                continue
            if not (matches_submission and matches_seat):
                continue
            for field in ("team_name", "teamName", "name", "Name"):
                value = agent.get(field)
                if isinstance(value, str) and value.strip():
                    claims.append((value, f"archived_episode_metadata.agents.{field}"))

    for value, source in claims:
        if _normalized_identity(value) == _ARCHIVED_SUBMISSION_IDENTITY_NORMALIZED:
            return {
                "availability": _availability(True),
                "seat": own_seat,
                "seat_authority": "archived_submission_bound_own_agent_seat",
                "selection_scope": "own_agent_decisions_only",
                "identity": {
                    "available": True,
                    "name": _ARCHIVED_SUBMISSION_IDENTITY,
                    "source": source,
                },
            }
    return {
        "availability": _availability(True),
        "seat": own_seat,
        "seat_authority": "archived_submission_bound_own_agent_seat",
        "selection_scope": "own_agent_decisions_only",
        "identity": {
            "available": False,
            "name": None,
            "availability_reasons": [
                "challengestone_identity_not_source_backed_by_archived_own_agent_metadata"
            ],
        },
    }


def _public_game_metadata(
    entry: ReplayCatalogEntry,
    *,
    own_seat: int | None,
    seat_reason: str | None,
    step_count: int | None,
) -> dict[str, Any]:
    """Return a compact list-row view rather than the whole replay manifest."""

    metadata = entry.metadata if isinstance(entry.metadata, Mapping) else {}
    own_agent = metadata.get("own_agent")
    opponent_agent = metadata.get("opponent_agent")
    if not isinstance(own_agent, Mapping):
        own_agent = {}
    if not isinstance(opponent_agent, Mapping):
        opponent_agent = {}
        agents = metadata.get("agents")
        if isinstance(agents, list):
            for agent in agents:
                if not isinstance(agent, Mapping):
                    continue
                try:
                    is_ours = int(agent.get("submission_id")) == entry.submission_id
                except (TypeError, ValueError):
                    is_ours = False
                if not is_ours:
                    opponent_agent = agent
                    break
    reward = _metadata_value(metadata, "reward", "score", "return")
    if reward is None:
        reward = _metadata_value(own_agent, "reward", "score", "return")
    opponent = _metadata_value(
        metadata,
        "opponent",
        "opponent_name",
        "opponent_submission_id",
    )
    if opponent is None:
        opponent = _metadata_value(
            opponent_agent,
            "name",
            "team_name",
            "submission_id",
            "id",
        )
    return {
        "submission_id": entry.submission_id,
        "submission_id_text": _submission_id_text(entry.submission_id),
        "episode_id": entry.episode_id,
        "own_seat": own_seat,
        "own_seat_reason": seat_reason,
        "decision_owner": _decision_owner_payload(
            metadata,
            submission_id=entry.submission_id,
            own_seat=own_seat,
            seat_reason=seat_reason,
        ),
        "reward": _json_safe(reward),
        "opponent": _json_safe(opponent),
        "created_at": _json_safe(
            _metadata_value(
                metadata,
                "created_at",
                "createdAt",
                "create_time",
                "createTime",
                "timestamp",
            )
        ),
        "step_count": step_count,
        # These are fixed-seat, source-backed catalog facts.  Do not derive
        # names/ranks from reward, opponent, submission id, or a file name.
        "players": [player.to_dict() for player in entry.players],
        "replay_available": entry.path is not None,
        "availability_reasons": list(entry.availability_reasons),
    }


def _indexed(values: Any, index: int) -> Any:
    """Read one model-output element without manufacturing a missing value."""

    if (
        isinstance(values, Sequence)
        and not isinstance(values, (str, bytes))
        and 0 <= index < len(values)
    ):
        return values[index]
    return None


# These integer values are the public competition OptionType / AreaType wire
# representation.  Textual enum values are accepted too, because archived
# replay JSON can preserve either form.  Keeping the map here avoids asking a
# live engine to reinterpret an archived action just to render its label.
_OPTION_TYPE_NAMES = {
    0: "NUMBER",
    1: "YES",
    2: "NO",
    3: "CARD",
    4: "TOOL_CARD",
    5: "ENERGY_CARD",
    6: "ENERGY",
    7: "PLAY",
    8: "ATTACH",
    9: "EVOLVE",
    10: "ABILITY",
    11: "DISCARD",
    12: "RETREAT",
    13: "ATTACK",
    14: "END",
    15: "SKILL",
    16: "SPECIAL_CONDITION",
}
_AREA_TYPE_NAMES = {
    1: "DECK",
    2: "HAND",
    3: "DISCARD",
    4: "ACTIVE",
    5: "BENCH",
    6: "PRIZE",
    7: "STADIUM",
    8: "ENERGY",
    9: "TOOL",
    10: "PRE_EVOLUTION",
    11: "PLAYER",
    12: "LOOKING",
}
_SPECIAL_CONDITION_NAMES = {
    0: "Poisoned",
    1: "Burned",
    2: "Asleep",
    3: "Paralyzed",
    4: "Confused",
}
_SPECIAL_CONDITION_ALIASES = {
    "poison": "Poisoned",
    "poisoned": "Poisoned",
    "burn": "Burned",
    "burned": "Burned",
    "sleep": "Asleep",
    "asleep": "Asleep",
    "paralyze": "Paralyzed",
    "paralysis": "Paralyzed",
    "paralyzed": "Paralyzed",
    "confuse": "Confused",
    "confused": "Confused",
}


def _plain_enum_name(value: Any, numbered: Mapping[int, str]) -> str | None:
    """Resolve archived integer or enum-name JSON without importing cg enums."""

    if isinstance(value, int) and not isinstance(value, bool):
        return numbered.get(int(value))
    if not isinstance(value, str):
        return None
    normalized = "".join(character for character in value if character.isalnum())
    normalized = normalized.casefold()
    for name in numbered.values():
        candidate = "".join(character for character in name if character.isalnum())
        if normalized == candidate.casefold():
            return name
    return None


def _special_condition_name(value: Any) -> str | None:
    named = _plain_enum_name(value, _SPECIAL_CONDITION_NAMES)
    if named is not None:
        return named
    if not isinstance(value, str):
        return None
    normalized = "".join(character for character in value if character.isalnum())
    return _SPECIAL_CONDITION_ALIASES.get(normalized.casefold())


def _exact_int(value: Any) -> int | None:
    """Accept only exact JSON integer values for option and zone references."""

    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return None


def _mapping_value(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _label_from_metadata(value: Any) -> str | None:
    """Read a source-provided card/attack name, never an inferred nickname."""

    for name in ("name", "cardName", "card_name", "attackName", "title"):
        raw = _mapping_value(value, name)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


@lru_cache(maxsize=1)
def _card_names_by_id() -> dict[int, str]:
    """Read card names by their explicit cardId key when cg metadata exists."""

    try:
        table = features.card_table()
    except Exception:  # noqa: BLE001 - the static replay browser has no cg dependency
        return {}
    if not isinstance(table, Mapping):
        return {}
    names: dict[int, str] = {}
    for raw_id, card in table.items():
        card_id = _exact_int(raw_id)
        if card_id is None:
            continue
        label = _label_from_metadata(card)
        if label is not None:
            names[card_id] = label
    return names


@lru_cache(maxsize=1)
def _attack_names_by_id() -> dict[int, str]:
    """Read attack names by the engine's explicit attack id, never list offset."""

    try:
        attacks = cg_env.all_attack()
    except Exception:  # noqa: BLE001 - render a raw-id transcript when unavailable
        return {}
    if not isinstance(attacks, Sequence) or isinstance(attacks, (str, bytes)):
        return {}
    names: dict[int, str] = {}
    for attack in attacks:
        attack_id = _exact_int(_mapping_value(attack, "id", "attackId"))
        if attack_id is None:
            continue
        label = _label_from_metadata(attack)
        if label is not None:
            names[attack_id] = label
    return names


def _card_id(value: Any) -> int | None:
    return _exact_int(_mapping_value(value, "id", "cardId", "card_id"))


def _card_label(card_id: int | None) -> str | None:
    if card_id is None:
        return None
    return _card_names_by_id().get(card_id)


def _player_label(player_index: int | None, own_index: int | None) -> str:
    if player_index is not None and own_index is not None:
        if player_index == own_index:
            return "your"
        return "opponent's"
    if player_index is None:
        return "the selected player's"
    return f"player {player_index}'s"


def _visible_zone_card(
    observation: Mapping[str, Any],
    *,
    area: str,
    index: int,
    player_index: int | None,
    own_index: int | None,
) -> tuple[Any | None, str | None]:
    """Resolve a card only from a causal, visible zone.

    In particular, no opponent hand/deck/prize identity is resolved even if a
    malformed archive happens to expose it.  The option's raw slot remains
    displayable, but no hidden card name is pulled into the transcript.
    """

    current = observation.get("current")
    current = current if isinstance(current, Mapping) else {}
    select = observation.get("select")
    select = select if isinstance(select, Mapping) else {}
    if area == "DECK":
        collection = select.get("deck")
    elif area == "STADIUM":
        collection = current.get("stadium")
    elif area == "LOOKING":
        collection = current.get("looking")
    else:
        if player_index not in (0, 1):
            return None, "option has no valid playerIndex for its source zone"
        if player_index != own_index and area in {"HAND", "DECK", "PRIZE"}:
            return None, "hidden opponent zone identity is intentionally not resolved"
        players = current.get("players")
        if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
            return None, "causal observation has no player zones"
        if player_index >= len(players) or not isinstance(
            players[player_index], Mapping
        ):
            return None, "causal observation has no referenced player zone"
        player = players[player_index]
        field = {
            "HAND": "hand",
            "DISCARD": "discard",
            "ACTIVE": "active",
            "BENCH": "bench",
            "PRIZE": "prize",
        }.get(area)
        if field is None:
            return None, "option area cannot be resolved as a visible card zone"
        collection = player.get(field)
    if not isinstance(collection, Sequence) or isinstance(collection, (str, bytes)):
        return None, "causal observation has no referenced visible zone"
    if index < 0 or index >= len(collection):
        return None, "option index is outside the referenced visible zone"
    return collection[index], None


def _zone_reference(
    observation: Mapping[str, Any],
    raw_option: Mapping[str, Any],
    *,
    area_field: str = "area",
    index_field: str = "index",
    player_field: str = "playerIndex",
    default_area: str | None = None,
    default_player: int | None = None,
) -> tuple[str | None, str | None]:
    """Return a plain, non-secret source/target zone reference."""

    raw_area = raw_option.get(area_field)
    area = (
        _plain_enum_name(raw_area, _AREA_TYPE_NAMES)
        if raw_area is not None
        else default_area
    )
    index = _exact_int(raw_option.get(index_field))
    current = observation.get("current")
    current = current if isinstance(current, Mapping) else {}
    own_index = _exact_int(current.get("yourIndex"))
    player_index = _exact_int(raw_option.get(player_field))
    if player_index is None:
        player_index = default_player if default_player is not None else own_index
    if area is None or index is None or index < 0:
        return None, "option lacks a valid source/target area and index"
    owner = _player_label(player_index, own_index)
    area_words = area.replace("_", " ").lower()
    reference = f"{owner} {area_words} slot {index}"
    card, resolution_reason = _visible_zone_card(
        observation,
        area=area,
        index=index,
        player_index=player_index,
        own_index=own_index,
    )
    if resolution_reason is not None:
        # The raw option still truthfully identifies the slot.  Do not turn a
        # missing name into a fabricated card identity.
        return f"{reference} (card identity not resolved)", None
    card_id = _card_id(card)
    card_name = _card_label(card_id)
    if card_name is not None:
        return f"{reference} ({card_name})", None
    if card_id is not None:
        return f"{reference} (card ID {card_id})", None
    return reference, None


def _option_description(
    observation: Mapping[str, Any], option_index: int
) -> dict[str, Any]:
    """Describe one raw engine option using only its causal representation."""

    select = observation.get("select")
    select = select if isinstance(select, Mapping) else {}
    options = select.get("option")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        return {
            "availability": _availability(
                False, "causal observation has no select.option list"
            ),
            "raw_option_index": option_index,
            "text": None,
        }
    if option_index < 0 or option_index >= len(options):
        return {
            "availability": _availability(
                False, "raw option index is outside causal select.option"
            ),
            "raw_option_index": option_index,
            "text": None,
        }
    raw_option = options[option_index]
    if not isinstance(raw_option, Mapping):
        return {
            "availability": _availability(False, "causal option is not an object"),
            "raw_option_index": option_index,
            "raw_option": _json_safe(raw_option),
            "text": None,
        }
    option_type = _plain_enum_name(raw_option.get("type"), _OPTION_TYPE_NAMES)
    base = {
        "raw_option_index": option_index,
        "raw_option": _json_safe(raw_option),
        "option_type": option_type,
    }
    if option_type is None:
        return {
            **base,
            "availability": _availability(
                False, "causal option has an unknown or missing OptionType"
            ),
            "text": None,
        }

    def source_reference(**kwargs: Any) -> tuple[str | None, dict[str, Any] | None]:
        text, reason = _zone_reference(observation, raw_option, **kwargs)
        if text is None:
            return None, _availability(False, reason or "option source is unavailable")
        return text, None

    if option_type == "NUMBER":
        number = _exact_int(raw_option.get("number"))
        if number is None:
            return {
                **base,
                "availability": _availability(
                    False, "NUMBER option has no exact number"
                ),
                "text": None,
            }
        text = f"Choose number {number}"
    elif option_type == "YES":
        text = "Answer Yes"
    elif option_type == "NO":
        text = "Answer No"
    elif option_type == "END":
        # This is an actual engine option and must not be conflated with the
        # factorized unchanged-prefix STOP candidate below.
        text = "Choose the engine's End option"
    elif option_type == "SPECIAL_CONDITION":
        condition = _special_condition_name(raw_option.get("specialConditionType"))
        if condition is None:
            return {
                **base,
                "availability": _availability(
                    False, "SPECIAL_CONDITION option has an unknown condition"
                ),
                "text": None,
            }
        text = f"Choose the {condition} special condition"
    elif option_type == "ATTACK":
        attack_id = _exact_int(raw_option.get("attackId"))
        if attack_id is None:
            return {
                **base,
                "availability": _availability(
                    False, "ATTACK option has no exact attackId"
                ),
                "text": None,
            }
        attack_name = _attack_names_by_id().get(attack_id)
        name = attack_name if attack_name is not None else f"attack ID {attack_id}"
        text = f"Use {name}"
    elif option_type == "PLAY":
        source, unavailable = source_reference(default_area="HAND")
        if unavailable is not None:
            return {**base, "availability": unavailable, "text": None}
        text = f"Play {source}"
    elif option_type in {"ATTACH", "EVOLVE"}:
        source, unavailable = source_reference()
        if unavailable is not None:
            return {**base, "availability": unavailable, "text": None}
        target, target_unavailable = source_reference(
            area_field="inPlayArea",
            index_field="inPlayIndex",
            player_field="inPlayPlayerIndex",
        )
        if target_unavailable is not None:
            return {**base, "availability": target_unavailable, "text": None}
        verb = "Attach" if option_type == "ATTACH" else "Evolve"
        preposition = "to" if option_type == "ATTACH" else "onto"
        text = f"{verb} {source} {preposition} {target}"
    elif option_type == "ABILITY":
        source, unavailable = source_reference()
        if unavailable is not None:
            return {**base, "availability": unavailable, "text": None}
        text = f"Use the Ability of {source}"
    elif option_type == "DISCARD":
        source, unavailable = source_reference()
        if unavailable is not None:
            return {**base, "availability": unavailable, "text": None}
        text = f"Discard {source}"
    elif option_type == "RETREAT":
        text = "Retreat your Active Pokémon"
    elif option_type == "CARD":
        source, unavailable = source_reference()
        if unavailable is not None:
            return {**base, "availability": unavailable, "text": None}
        text = f"Choose {source}"
    elif option_type in {"TOOL_CARD", "ENERGY_CARD", "ENERGY"}:
        source, unavailable = source_reference()
        if unavailable is not None:
            return {**base, "availability": unavailable, "text": None}
        if option_type == "TOOL_CARD":
            attachment_index = _exact_int(raw_option.get("toolIndex"))
            kind = "tool"
        else:
            attachment_index = _exact_int(raw_option.get("energyIndex"))
            kind = "energy"
        if attachment_index is None or attachment_index < 0:
            return {
                **base,
                "availability": _availability(
                    False, f"{option_type} option has no exact attachment index"
                ),
                "text": None,
            }
        text = f"Choose {kind} slot {attachment_index} on {source}"
    elif option_type == "SKILL":
        card_id = _exact_int(raw_option.get("cardId"))
        serial = _exact_int(raw_option.get("serial"))
        if card_id is None:
            return {
                **base,
                "availability": _availability(
                    False, "SKILL option has no exact cardId"
                ),
                "text": None,
            }
        serial_text = "" if serial is None else f" #{serial}"
        if card_id == 0:
            text = f"Use special-condition skill{serial_text}"
        else:
            card_name = _card_label(card_id)
            card_text = card_name if card_name is not None else f"card ID {card_id}"
            text = f"Use skill{serial_text} from {card_text}"
    else:
        return {
            **base,
            "availability": _availability(
                False, f"OptionType {option_type} has no transcript formatter"
            ),
            "text": None,
        }
    return {**base, "availability": _availability(True), "text": text}


def _factorized_action_transcript(
    observation: Mapping[str, Any],
    raw_action: Any,
    *,
    factorized_stage: int,
    recorded_action: Sequence[Any],
) -> dict[str, Any]:
    """Render one factorized prefix candidate without treating STOP as END."""

    if not isinstance(raw_action, Sequence) or isinstance(raw_action, (str, bytes)):
        return {
            "availability": _availability(
                False, "legal candidate is not an option-index list"
            ),
            "factorized_stage": factorized_stage,
            "raw_option_indices": _json_safe(raw_action),
            "synthetic_stop": False,
            "text": "Plain-English transcript unavailable; raw candidate is not an option-index list.",
            "components": [],
        }
    indices = [_exact_int(value) for value in raw_action]
    if any(value is None or value < 0 for value in indices):
        return {
            "availability": _availability(
                False, "legal candidate contains a non-integer or negative option index"
            ),
            "factorized_stage": factorized_stage,
            "raw_option_indices": _json_safe(raw_action),
            "synthetic_stop": False,
            "text": "Plain-English transcript unavailable; see the retained raw option path.",
            "components": [],
        }
    option_indices = [int(value) for value in indices if value is not None]
    prefix = [
        value
        for value in (_exact_int(item) for item in recorded_action[:factorized_stage])
        if value is not None
    ]
    synthetic_stop = option_indices == prefix
    components = [_option_description(observation, index) for index in option_indices]
    unavailable = next(
        (
            component.get("availability", {}).get("reason")
            for component in components
            if isinstance(component.get("availability"), Mapping)
            and component["availability"].get("available") is False
        ),
        None,
    )
    if unavailable is not None:
        return {
            "availability": _availability(False, str(unavailable)),
            "factorized_stage": factorized_stage,
            "raw_option_indices": option_indices,
            "synthetic_stop": synthetic_stop,
            "text": "Plain-English transcript unavailable; see the retained raw option path.",
            "components": components,
        }
    component_text = [str(component["text"]) for component in components]
    if synthetic_stop:
        prior = "; then ".join(component_text)
        continuation = (
            "with no selected engine options" if not prior else f"after {prior}"
        )
        text = (
            f"Factorized stage {factorized_stage}: synthetic STOP — finish this "
            f"selection {continuation}."
        )
    elif len(component_text) == 1:
        text = f"Factorized stage {factorized_stage}: {component_text[0]}."
    else:
        text = (
            f"Factorized stage {factorized_stage}: continue the ordered selection "
            + "; then ".join(component_text)
            + "."
        )
    return {
        "availability": _availability(True),
        "factorized_stage": factorized_stage,
        "raw_option_indices": option_indices,
        "synthetic_stop": synthetic_stop,
        "text": text,
        "components": components,
    }


def _with_action_transcripts(
    options: Sequence[Any],
    *,
    observation: Mapping[str, Any],
    factorized_stage: int,
    recorded_action: Sequence[Any],
) -> list[dict[str, Any]]:
    """Preserve raw legal actions while attaching a causal display transcript."""

    rendered: list[dict[str, Any]] = []
    for item in options:
        record = dict(item) if isinstance(item, Mapping) else {"action": item}
        raw_action = record.get("action")
        transcript = _factorized_action_transcript(
            observation,
            raw_action,
            factorized_stage=factorized_stage,
            recorded_action=recorded_action,
        )
        record["raw_action"] = _json_safe(raw_action)
        record["action_transcript"] = transcript["text"]
        record["action_transcript_availability"] = transcript["availability"]
        record["action_transcript_detail"] = transcript
        # ``label`` is retained for the existing local UI, while raw_action
        # remains the source of truth for audits and model reconstruction.
        record["label"] = transcript["text"]
        rendered.append(record)
    return rendered


def _unavailable_policy_influence(reason: str) -> dict[str, Any]:
    return {"availability": _availability(False, reason)}


def _policy_influence_from_head(
    raw_head: Mapping[str, Any], leave_one_out: Any
) -> dict[str, Any]:
    """Keep only supplied LOO final-policy facts; never infer from activations."""

    direct = raw_head.get("policy_influence")
    if isinstance(direct, Mapping):
        return dict(direct)
    if isinstance(leave_one_out, Mapping):
        runtime_path = leave_one_out.get("runtime_path")
        if isinstance(runtime_path, Mapping):
            influence = runtime_path.get("policy_influence")
            if isinstance(influence, Mapping):
                return dict(influence)
            availability = runtime_path.get("availability")
            if (
                isinstance(availability, Mapping)
                and availability.get("available") is False
            ):
                return _unavailable_policy_influence(
                    str(
                        availability.get("reason")
                        or "runtime leave-one-head-out policy is unavailable"
                    )
                )
    availability = raw_head.get("availability")
    if isinstance(availability, Mapping) and availability.get("available") is False:
        return _unavailable_policy_influence(
            str(availability.get("reason") or "head is unavailable")
        )
    return _unavailable_policy_influence(
        "exact leave-one-head-out final-policy influence was not supplied"
    )


def _option_label_for_index(
    legal_options: Sequence[Mapping[str, Any]], index: Any
) -> tuple[str | None, dict[str, Any]]:
    option_index = _exact_int(index)
    if option_index is None or option_index < 0 or option_index >= len(legal_options):
        return None, _availability(
            False, "referenced legal option index is unavailable"
        )
    label = legal_options[option_index].get("label")
    if isinstance(label, str) and label.strip():
        return label, _availability(True)
    return None, _availability(False, "legal option has no action transcript")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    return None


def _signed(value: float) -> str:
    return f"{value:+.6f}"


def _with_policy_influence_labels(
    heads: Sequence[dict[str, Any]], legal_options: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Add labels and a plain-English reading to exact supplied LOO effects."""

    required = (
        "selected_option_index",
        "selected_option_probability_delta",
        "selected_option_logit_delta",
        "maximum_absolute_option_probability_delta",
        "most_helped_option",
        "most_hurt_option",
    )
    result: list[dict[str, Any]] = []
    for head in heads:
        record = dict(head)
        raw_influence = record.get("policy_influence")
        influence = dict(raw_influence) if isinstance(raw_influence, Mapping) else {}
        availability = influence.get("availability")
        if (
            not isinstance(availability, Mapping)
            or availability.get("available") is False
        ):
            reason = (
                availability.get("reason")
                if isinstance(availability, Mapping)
                else "exact leave-one-head-out final-policy influence was not supplied"
            )
            influence["availability"] = _availability(False, str(reason))
            influence["plain_english_interpretation"] = (
                "Exact leave-one-head-out policy influence is unavailable: "
                + str(reason)
            )
            record["policy_influence"] = influence
            result.append(record)
            continue
        missing = [name for name in required if name not in influence]
        if missing:
            reason = (
                "exact leave-one-head-out policy influence omitted required field(s): "
                + ", ".join(missing)
            )
            influence["availability"] = _availability(False, reason)
            influence["plain_english_interpretation"] = (
                "Exact leave-one-head-out policy influence is unavailable: " + reason
            )
            record["policy_influence"] = influence
            result.append(record)
            continue
        selected_label, selected_label_availability = _option_label_for_index(
            legal_options, influence.get("selected_option_index")
        )
        influence["selected_option"] = {
            "index": influence.get("selected_option_index"),
            "label": selected_label,
            "label_availability": selected_label_availability,
        }
        option_records: dict[str, tuple[str, str]] = {
            "most_helped_option": ("most helped", "most_helped_option"),
            "most_hurt_option": ("most hurt", "most_hurt_option"),
        }
        labels: dict[str, str | None] = {}
        for field, (_description, key) in option_records.items():
            raw_option = influence.get(key)
            enriched = dict(raw_option) if isinstance(raw_option, Mapping) else {}
            option_label, label_availability = _option_label_for_index(
                legal_options, enriched.get("index")
            )
            enriched["label"] = option_label
            enriched["label_availability"] = label_availability
            influence[field] = enriched
            labels[field] = option_label

        probability_delta = _finite_number(
            influence.get("selected_option_probability_delta")
        )
        logit_delta = _finite_number(influence.get("selected_option_logit_delta"))
        maximum_shift = _finite_number(
            influence.get("maximum_absolute_option_probability_delta")
        )
        helped_delta = _finite_number(
            influence["most_helped_option"].get("probability_delta")
        )
        hurt_delta = _finite_number(
            influence["most_hurt_option"].get("probability_delta")
        )
        if None in (
            probability_delta,
            logit_delta,
            maximum_shift,
            helped_delta,
            hurt_delta,
        ):
            reason = "exact leave-one-head-out influence has non-finite policy metrics"
            influence["availability"] = _availability(False, reason)
            influence["plain_english_interpretation"] = (
                "Exact leave-one-head-out policy influence is unavailable: " + reason
            )
            record["policy_influence"] = influence
            result.append(record)
            continue
        assert probability_delta is not None
        assert logit_delta is not None
        assert maximum_shift is not None
        assert helped_delta is not None
        assert hurt_delta is not None
        selected_name = (
            selected_label or f"legal option {influence['selected_option_index']}"
        )
        helped_name = labels["most_helped_option"] or "an unlabeled legal option"
        hurt_name = labels["most_hurt_option"] or "an unlabeled legal option"
        direction = "increased" if probability_delta > 0 else "decreased"
        if probability_delta == 0:
            direction = "did not change"
        influence["conditional_policy_scope"] = "factorized_stage_legal_options"
        influence["plain_english_interpretation"] = (
            "Exact leave-one-head-out final-policy recomputation at this conditional "
            "factorized stage (full policy minus policy without this head): keeping "
            f"this head {direction} selected option {selected_name}'s probability by "
            f"{_signed(probability_delta)} and logit by {_signed(logit_delta)}. "
            f"The largest absolute legal-option probability shift was "
            f"{maximum_shift:.6f}. It most helped {helped_name} "
            f"({_signed(helped_delta)}) and most hurt {hurt_name} "
            f"({_signed(hurt_delta)}). Per-head effects are non-additive."
        )
        record["policy_influence"] = influence
        result.append(record)
    return result


def _adapter_status(
    adapter: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalize one decision-specific adapter state without weight guessing."""

    if not isinstance(adapter, Mapping):
        return {
            "status": "unavailable",
            "enabled": None,
            "matched_archetype": None,
            "matched_archetype_availability": _availability(
                False, "adapter runtime payload was not supplied"
            ),
            "route": None,
            "slot": None,
            "route_reliability": _availability(
                False, "adapter runtime payload was not supplied"
            ),
            "policy_influence": _unavailable_policy_influence(
                "adapter runtime payload was not supplied"
            ),
            "reason": "adapter runtime payload was not supplied",
            "plain_english": "Matchup Adapter status is unavailable: adapter runtime payload was not supplied.",
            "evaluation_basis": "checksum_bound_causal_re_evaluation",
            "historical_activation_recorded": False,
            "runtime_activation": {
                "status": "unavailable",
                "applied": False,
                "reason": "adapter runtime payload was not supplied",
            },
        }

    enabled = adapter.get("runtime_enabled")
    enabled = enabled if isinstance(enabled, bool) else None
    route = _exact_int(adapter.get("route"))
    slot = _exact_int(adapter.get("slot"))
    bank_present = adapter.get("bank_present")
    bank_present = bank_present if isinstance(bank_present, bool) else None
    route_routable = adapter.get("route_routable")
    route_routable = route_routable if isinstance(route_routable, bool) else None
    runtime_activation = adapter.get("runtime_activation")
    runtime_activation = (
        dict(runtime_activation)
        if isinstance(runtime_activation, Mapping)
        else {
            "status": "unavailable",
            "applied": False,
            "evaluation_basis": "checksum_bound_causal_re_evaluation",
            "historical_activation_recorded": False,
            "reason": "submitted runtime activation evidence was not supplied",
        }
    )
    actual_active = adapter.get("decision_route_active") is True
    archetype = adapter.get("matched_archetype")
    archetype = archetype if isinstance(archetype, str) and archetype.strip() else None
    archetype_availability = (
        {
            **_availability(True),
            "source": adapter.get("matched_archetype_source"),
        }
        if archetype is not None
        else _availability(False, "no source-backed matched archetype was supplied")
    )
    reliability = adapter.get("route_reliability")
    reliability = (
        dict(reliability)
        if isinstance(reliability, Mapping)
        else _availability(False, "adapter route reliability was not supplied")
    )
    influence = adapter.get("policy_influence")
    influence = (
        dict(influence)
        if isinstance(influence, Mapping)
        else _unavailable_policy_influence(
            "exact no-adapter final-policy effect was not supplied"
        )
    )
    if actual_active:
        if (
            bank_present is not True
            or runtime_activation.get("applied") is not True
            or enabled is not True
            or route is None
            or route < 0
        ):
            status = "unavailable"
            reason = (
                "adapter payload claims an active route without complete runtime facts"
            )
        elif route_routable is not True:
            status = "unavailable"
            reason = (
                "adapter payload does not verify that the selected route is routable"
            )
        else:
            status = "active_for_decision"
            reason = None
    elif runtime_activation.get("applied") is not True:
        status = "unavailable"
        reason = str(
            runtime_activation.get("reason")
            or "checksum-bound submitted startup activation is unavailable"
        )
    elif bank_present is False:
        status = "unavailable"
        reason = "checkpoint has no matchup adapter bank"
    elif route is None:
        status = "bypassed"
        reason = "no router route was bound; policy/value state used exact bypass"
    elif enabled is False:
        status = "bypassed"
        reason = "matchup adapter bank is runtime-disabled for this decision"
    elif route < 0:
        status = "bypassed"
        reason = "router selected its explicit unknown/bypass route"
    elif enabled is None:
        status = "unavailable"
        reason = "adapter runtime enabled state is unavailable"
    elif route_routable is False:
        status = "unavailable"
        reason = "router route is not routable by this adapter bank"
    else:
        status = "unavailable"
        reason = "adapter route application cannot be verified for this decision"
    if status == "active_for_decision":
        matched = archetype or "an unnamed source-backed route"
        plain = (
            f"Matchup Adapter active for this decision on route {route}"
            + (f" (slot {slot})" if slot is not None else "")
            + f" for {matched}."
        )
    elif status == "bypassed":
        plain = f"Matchup Adapter bypassed for this decision: {reason}."
    else:
        plain = f"Matchup Adapter status is unavailable: {reason}."
    return {
        "status": status,
        "enabled": enabled,
        "matched_archetype": archetype,
        "matched_archetype_availability": archetype_availability,
        "matched_archetype_source": adapter.get("matched_archetype_source"),
        "route": route,
        "slot": slot,
        "slot_status": adapter.get("slot_status"),
        "route_reliability": reliability,
        "policy_influence": influence,
        "reason": reason,
        "plain_english": plain,
        "raw_adapter_available": True,
        "evaluation_basis": runtime_activation.get("evaluation_basis"),
        "historical_activation_recorded": runtime_activation.get(
            "historical_activation_recorded"
        ),
        "runtime_activation": runtime_activation,
    }


def _with_adapter_policy_influence_labels(
    status: Mapping[str, Any], legal_options: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Give an exact adapter counterfactual the same legal-action labels."""

    record = _with_policy_influence_labels(
        [{"policy_influence": status.get("policy_influence")}], legal_options
    )[0]
    influence = record["policy_influence"]
    interpretation = influence.get("plain_english_interpretation")
    if isinstance(interpretation, str):
        influence["plain_english_interpretation"] = interpretation.replace(
            "leave-one-head-out", "no-adapter"
        ).replace("this head", "the Matchup Adapter route")
    result = dict(status)
    result["policy_influence"] = influence
    result["on_off_comparison"] = _adapter_on_off_comparison(influence, legal_options)
    return result


def _adapter_on_off_comparison(
    influence: Mapping[str, Any],
    legal_options: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Normalize the exact applied-adapter versus no-adapter policy surface."""

    availability = influence.get("availability")
    if not isinstance(availability, Mapping) or availability.get("available") is False:
        reason = (
            availability.get("reason")
            if isinstance(availability, Mapping)
            else "exact no-adapter policy influence was not supplied"
        )
        return {
            "availability": _availability(False, str(reason)),
            "method": "exact_adapter_on_vs_no_adapter_forward_recomputation",
            "sign_convention": "adapter_on_minus_adapter_off",
            "historical": False,
            "reproduction_status": "recomputed_not_historical",
        }

    raw = influence.get("leave_one_adapter_out")
    if not isinstance(raw, Mapping):
        return {
            "availability": _availability(
                False, "exact adapter ON/OFF policy vectors were not supplied"
            ),
            "method": "exact_adapter_on_vs_no_adapter_forward_recomputation",
            "sign_convention": "adapter_on_minus_adapter_off",
            "historical": False,
            "reproduction_status": "recomputed_not_historical",
        }

    option_count = len(legal_options)

    def finite_vector(name: str) -> list[float] | None:
        value = raw.get(name)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return None
        converted = [_finite_number(item) for item in value]
        if len(converted) != option_count or any(item is None for item in converted):
            return None
        return [float(item) for item in converted if item is not None]

    on_probabilities = finite_vector("full_policy_probabilities")
    off_probabilities = finite_vector("policy_without_head_probabilities")
    on_logits = finite_vector("full_policy_logits")
    off_logits = finite_vector("policy_without_head_logits")
    if any(
        vector is None
        for vector in (on_probabilities, off_probabilities, on_logits, off_logits)
    ):
        return {
            "availability": _availability(
                False,
                "exact adapter ON/OFF vectors are missing, non-finite, or do not match the legal options",
            ),
            "method": "exact_adapter_on_vs_no_adapter_forward_recomputation",
            "sign_convention": "adapter_on_minus_adapter_off",
            "historical": False,
            "reproduction_status": "recomputed_not_historical",
        }

    assert on_probabilities is not None
    assert off_probabilities is not None
    assert on_logits is not None
    assert off_logits is not None
    on_position = max(range(option_count), key=on_probabilities.__getitem__)
    off_position = max(range(option_count), key=off_probabilities.__getitem__)

    def option_record(position: int) -> dict[str, Any]:
        option = legal_options[position]
        option_index = _exact_int(option.get("index"))
        if option_index is None:
            option_index = position
        label, label_availability = _option_label_for_index(legal_options, position)
        return {
            "position": position,
            "index": option_index,
            "label": label,
            "label_availability": label_availability,
        }

    options: list[dict[str, Any]] = []
    for position in range(option_count):
        option = option_record(position)
        option.update(
            {
                "adapter_on_probability": on_probabilities[position],
                "adapter_off_probability": off_probabilities[position],
                "probability_delta": (
                    on_probabilities[position] - off_probabilities[position]
                ),
                "adapter_on_logit": on_logits[position],
                "adapter_off_logit": off_logits[position],
                "logit_delta": on_logits[position] - off_logits[position],
                "adapter_on_choice": position == on_position,
                "adapter_off_choice": position == off_position,
            }
        )
        options.append(option)

    probability_delta = [
        on - off for on, off in zip(on_probabilities, off_probabilities, strict=True)
    ]
    return {
        "availability": _availability(True),
        "method": "exact_adapter_on_vs_no_adapter_forward_recomputation",
        "sign_convention": "adapter_on_minus_adapter_off",
        "historical": False,
        "reproduction_status": "recomputed_not_historical",
        "adapter_on": {
            "meaning": "exact_checksum_bound_submitted_runtime_policy",
            "selected_option": option_record(on_position),
            "probabilities": on_probabilities,
            "logits": on_logits,
        },
        "adapter_off": {
            "meaning": "exact_forward_with_only_matchup_adapter_route_bypassed",
            "selected_option": option_record(off_position),
            "probabilities": off_probabilities,
            "logits": off_logits,
        },
        "choice_changed": on_position != off_position,
        "maximum_absolute_probability_shift": max(
            (abs(value) for value in probability_delta), default=0.0
        ),
        "total_variation_distance": 0.5
        * sum(abs(value) for value in probability_delta),
        "options": options,
    }


def _with_decision_influence_labels(
    raw: Mapping[str, Any] | None,
    legal_options: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach truthful legal-action transcripts to a scale counterfactual."""

    if not isinstance(raw, Mapping):
        return {
            "availability": _availability(
                False, "decision influence payload was not supplied"
            )
        }
    result = dict(raw)

    def option_identity(index: Any) -> dict[str, Any]:
        exact = _exact_int(index)
        label, label_availability = _option_label_for_index(legal_options, exact)
        option = _indexed(legal_options, exact) if exact is not None else None
        option = option if isinstance(option, Mapping) else {}
        return {
            "index": exact,
            "label": label,
            "label_availability": label_availability,
            "raw_action": option.get("raw_action"),
            "action_transcript": option.get("action_transcript"),
        }

    for key in ("baseline", "counterfactual"):
        value = result.get(key)
        if isinstance(value, Mapping):
            enriched = dict(value)
            enriched["selected_option"] = option_identity(
                enriched.get("selected_option_index")
            )
            result[key] = enriched
    effect = result.get("effect")
    if isinstance(effect, Mapping):
        enriched_effect = dict(effect)
        for key in ("most_helped_option", "most_hurt_option"):
            value = enriched_effect.get(key)
            if isinstance(value, Mapping):
                enriched = dict(value)
                enriched.update(option_identity(enriched.get("index")))
                enriched_effect[key] = enriched
        result["effect"] = enriched_effect
    return result


def _route_contributions_for_option(
    fusion: Mapping[str, Any], index: int
) -> dict[str, Any] | None:
    """Project per-route vectors to one legal option for the UI table."""

    routes_wrapper = fusion.get("routes")
    routes = (
        routes_wrapper.get("routes") if isinstance(routes_wrapper, Mapping) else None
    )
    if not isinstance(routes, Mapping):
        return None
    output: dict[str, Any] = {}
    for name, raw_record in routes.items():
        if not isinstance(raw_record, Mapping):
            continue
        contribution = _indexed(raw_record.get("runtime_contribution"), index)
        if contribution is None:
            contribution = _indexed(raw_record.get("reliability_weighted_delta"), index)
        raw_delta = _indexed(raw_record.get("raw_route_delta"), index)
        output[str(name)] = {
            "availability": raw_record.get("availability"),
            "runtime_active": raw_record.get("runtime_active"),
            "reliability": raw_record.get("reliability"),
            "raw_route_delta": raw_delta,
            "contribution": contribution,
        }
    return output or None


def _normalise_heads(
    raw_heads: Any,
    fusion: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Convert the inference map into stable, listable all-head records."""

    if not isinstance(raw_heads, Mapping):
        return []
    routes_wrapper = fusion.get("routes") if isinstance(fusion, Mapping) else None
    routes = routes_wrapper.get("routes") if isinstance(routes_wrapper, Mapping) else {}
    ablations = fusion.get("leave_one_out") if isinstance(fusion, Mapping) else {}
    records: list[dict[str, Any]] = []
    for name in sorted(raw_heads, key=str):
        raw = raw_heads[name]
        if not isinstance(raw, Mapping):
            continue
        availability = raw.get("availability")
        available = not (
            isinstance(availability, Mapping) and availability.get("available") is False
        )
        route = routes.get(name) if isinstance(routes, Mapping) else None
        if not isinstance(route, Mapping):
            route = {}
        leave_one_out = raw.get("leave_one_out")
        if leave_one_out is None and isinstance(ablations, Mapping):
            leave_one_out = ablations.get(name)
        records.append(
            {
                "name": raw.get("name", str(name)),
                "available": available,
                "availability": availability,
                "scope": raw.get("source_kind"),
                "module_name": raw.get("module_name"),
                "shape": raw.get("shape"),
                "raw": raw.get("raw_values"),
                "normalized": raw.get("normalization"),
                "mask": raw.get("mask"),
                "fusion_input_values": raw.get("fusion_input_values"),
                "base_output_values": raw.get("base_output_values"),
                "h10_residual": raw.get("h10_residual"),
                "route_reliability": route.get("reliability"),
                "route_delta": route.get(
                    "runtime_contribution", route.get("reliability_weighted_delta")
                ),
                "ablation_effect": leave_one_out,
                "policy_influence": _policy_influence_from_head(raw, leave_one_out),
            }
        )
    return records


def _simple_value(value: Any) -> Any:
    """Unwrap a singleton model value for a compact decision overview."""

    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


class InspectorApplication:
    """Read-only application state shared by short-lived HTTP handlers.

    The catalog and provenance snapshot are constructed once at startup.  An
    explicit ``refresh_index`` method exists for a future local-only reload
    control, but the HTTP API intentionally has no mutating refresh endpoint.
    At most one model is retained by the lazy backend cache.
    """

    def __init__(self, config: InspectorConfig) -> None:
        self.config = config
        self._state_lock = threading.RLock()
        self._inference_lock = threading.RLock()
        # This is a digest-only performance memo.  Every use still resolves
        # containment and restats the immutable artifact/tree before it can
        # satisfy a current checksum gate.
        self._integrity_digest_memo = _IntegrityDigestMemo()
        # A physical game is the cache boundary for baseline reconstruction.
        # This L1 state makes rapid browser step/stage changes join one job
        # immediately.  The L2 /tmp cache is intentionally kept behind the
        # small adapter methods below so its filesystem policy stays isolated
        # from the HTTP/inference layer.
        self._game_materialization_lock = threading.Lock()
        self._game_materialization_state_lock = threading.RLock()
        self._game_materialization_ready = threading.Condition(
            self._game_materialization_state_lock
        )
        self._game_materialization_jobs: OrderedDict[
            str, _GameMaterializationJob
        ] = OrderedDict()
        # Keep at most one complete raw game in L1.  A full game can contain
        # many large inspector payloads; persistent reuse belongs to the
        # configured compressed /tmp cache rather than process RSS.
        self._game_materialization_memory_limit = 1
        self._game_trace_cache_backend: Any | None = None
        self._game_trace_cache_error: str | None = None
        self._game_materialization_metrics_lock = threading.Lock()
        self._game_materialization_metrics: dict[str, int] = {
            "jobs_started": 0,
            "jobs_completed": 0,
            "backend_game_runs": 0,
            "l2_hits": 0,
            "l1_joins": 0,
        }
        self._provenance: ProvenanceManifest | None = None
        self._provenance_error: str | None = None
        self._training_recipe_registry: TrainingRecipeRegistry | None = None
        self._training_recipe_registry_error: str | None = None
        self._catalog: ReplayArchiveCatalog
        self._web_root = self._validate_web_root(config.web_root)
        self._inference_module: ModuleType | None = None
        self._inference_import_error: str | None = None
        self._model_cache: Any | None = None
        self._model_cache_error: str | None = None
        self.refresh_index(verify_provenance=True)

    @staticmethod
    def _validate_web_root(web_root: Path) -> Path:
        """Resolve only the three fixed static files and reject escaped links."""

        try:
            root = Path(web_root).expanduser().resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"inspector web root is missing: {web_root}") from exc
        except OSError as exc:
            raise ValueError(f"inspector web root is unreadable: {web_root}") from exc
        if not root.is_dir():
            raise ValueError(f"inspector web root is not a directory: {web_root}")
        for filename, _content_type in _STATIC_ASSETS.values():
            try:
                asset = resolve_contained_path(
                    filename,
                    roots=(root,),
                    relative_to=root,
                    require_exists=True,
                )
            except (FileNotFoundError, OSError, ProvenanceError) as exc:
                raise ValueError(
                    f"required inspector static asset is unavailable: {filename}"
                ) from exc
            if not asset.is_file():
                raise ValueError(
                    f"required inspector static asset is not a regular file: {filename}"
                )
        return root

    @property
    def catalog(self) -> ReplayArchiveCatalog:
        with self._state_lock:
            return self._catalog

    @property
    def provenance(self) -> ProvenanceManifest | None:
        with self._state_lock:
            return self._provenance

    def refresh_index(self, *, verify_provenance: bool = True) -> ReplayArchiveCatalog:
        """Rebuild archive rows from configured read-only roots.

        This is deliberately a normal method rather than an HTTP action.  It
        may hash exact model artifacts when ``verify_provenance`` is true, so
        callers should use it only at startup or through an operator-controlled
        lifecycle.
        """

        with self._state_lock:
            provenance = self._provenance
            provenance_error = self._provenance_error
            training_recipe_registry = self._training_recipe_registry
            training_recipe_registry_error = self._training_recipe_registry_error
            if verify_provenance:
                provenance = None
                provenance_error = None
                if self.config.provenance_manifest is None:
                    provenance_error = "provenance_not_configured"
                else:
                    try:
                        provenance = load_provenance_manifest(
                            self.config.provenance_manifest,
                            source_roots=self.config.source_roots,
                        )
                    except (FileNotFoundError, OSError, ProvenanceError, ValueError):
                        # Do not include exception text in a public response:
                        # it can contain a local source path or malformed raw
                        # manifest field.  The operator can validate locally.
                        provenance_error = "provenance_manifest_unavailable_or_invalid"
                self._provenance = provenance
                self._provenance_error = provenance_error
                training_recipe_registry = None
                training_recipe_registry_error = None
                if self.config.training_recipe_registry is None:
                    training_recipe_registry_error = (
                        "training_recipe_registry_not_configured"
                    )
                else:
                    try:
                        training_recipe_registry = load_training_recipe_registry(
                            self.config.training_recipe_registry
                        )
                    except (
                        FileNotFoundError,
                        OSError,
                        TrainingRecipeRegistryError,
                        ValueError,
                    ):
                        # The registry is presentation-only metadata.  It must
                        # never take replay browsing down or trigger a
                        # fallback to a selector/current-training recipe.
                        training_recipe_registry_error = (
                            "training_recipe_registry_unavailable_or_invalid"
                        )
                self._training_recipe_registry = training_recipe_registry
                self._training_recipe_registry_error = training_recipe_registry_error
            self._catalog = scan_archive(self.config.replay_root, provenance=provenance)
            return self._catalog

    def health_payload(self) -> dict[str, Any]:
        with self._state_lock:
            catalog = self._catalog
            provenance = self._provenance
            provenance_error = self._provenance_error
            training_recipe_registry = self._training_recipe_registry
            training_recipe_registry_error = self._training_recipe_registry_error
        archive_available = catalog.available
        verified_record_count = (
            sum(1 for record in provenance.entries if record.available)
            if provenance is not None
            else 0
        )
        model_analysis_submission_count = sum(
            1
            for submission in catalog.submissions
            if submission.model_analysis_available
        )
        provenance_available = (
            provenance is not None
            and not provenance.issues
            and verified_record_count > 0
        )
        degraded = not archive_available or not provenance_available
        return {
            "schema": API_SCHEMA,
            "service": SERVICE_NAME,
            "status": "degraded" if degraded else "ok",
            "read_only": True,
            "network_scope": "loopback_only",
            "config": self.config.public_summary(),
            "archive": {
                "available": archive_available,
                "submission_count": len(catalog.submissions),
                "availability_reasons": list(catalog.availability_reasons),
            },
            "provenance": {
                "configured": self.config.provenance_manifest is not None,
                "available": provenance_available,
                "issue_count": len(provenance.issues) if provenance is not None else 0,
                "verified_record_count": verified_record_count,
                "model_analysis_submission_count": model_analysis_submission_count,
                "reason": provenance_error,
            },
            "training_recipe": {
                "configured": self.config.training_recipe_registry is not None,
                "available": bool(
                    training_recipe_registry is not None
                    and not training_recipe_registry.issues
                    and any(
                        record.available for record in training_recipe_registry.records
                    )
                ),
                "issue_count": (
                    len(training_recipe_registry.issues)
                    if training_recipe_registry is not None
                    else 0
                ),
                "reason": training_recipe_registry_error,
            },
            "model_runtime": {
                "loaded": self._model_cache is not None,
                "availability_reason": self._model_cache_error,
            },
            "game_materialization": self._game_materialization_health_payload(),
        }

    def _game_materialization_health_payload(self) -> dict[str, Any]:
        """Return bounded, non-sensitive cache/job health information.

        Do not create the optional disk cache merely to answer a health
        request: its construction owns filesystem validation and should remain
        demand-driven by an actual baseline trace request.  The counters are
        intentionally aggregate-only, so health cannot disclose replay ids or
        the currently selected game.
        """

        with self._game_materialization_ready:
            jobs = tuple(self._game_materialization_jobs.values())
            active_job_count = sum(1 for job in jobs if not job.complete)
            queued_address_count = sum(len(job.pending) for job in jobs)
            active_address_count = sum(
                1 for job in jobs if job.active_address is not None
            )
        with self._state_lock:
            cache = self._game_trace_cache_backend
            cache_error = self._game_trace_cache_error
        raw_metrics = getattr(cache, "metrics", None) if cache is not None else None
        metric_names = (
            "entry_writes",
            "entry_write_failures",
            "manifest_writes",
            "manifest_write_failures",
            "low_space_rejections",
            "evictions",
        )
        metrics = {
            name: value
            for name in metric_names
            if isinstance((value := getattr(raw_metrics, name, None)), int)
        }
        with self._game_materialization_metrics_lock:
            job_metrics = dict(self._game_materialization_metrics)
        return {
            "enabled": self.config.game_trace_cache_enabled,
            "cache_initialized": cache is not None,
            "cache_error": cache_error,
            "in_memory_job_count": len(jobs),
            "active_job_count": active_job_count,
            "queued_address_count": queued_address_count,
            "active_address_count": active_address_count,
            "counters": job_metrics,
            "cache_metrics": metrics if raw_metrics is not None else None,
        }

    def _record_game_materialization_metric(self, name: str) -> None:
        """Increment one aggregate-only materialization counter."""

        with self._game_materialization_metrics_lock:
            self._game_materialization_metrics[name] = (
                self._game_materialization_metrics.get(name, 0) + 1
            )

    def check_payload(self) -> dict[str, Any]:
        """Return the non-mutating startup validation report used by ``--check``."""

        payload = self.health_payload()
        payload["check"] = {
            "web_root": str(self._web_root),
            "static_assets": sorted(_STATIC_ASSETS),
            "static_assets_available": True,
            "source_roots": [str(root) for root in self.config.source_roots],
        }
        return payload

    def _training_recipe_for_submission(
        self, submission: SubmissionCatalogEntry
    ) -> dict[str, Any]:
        """Return only a checksum-bound read-only recipe for one submission.

        This intentionally consults submission provenance, not a current
        selector, run directory, or checkpoint filename.  A recipe can remain
        useful when a replay lacks dynamic-runtime parity, but it still needs
        a verified exact checkpoint mapping.
        """

        provenance = submission.provenance
        checkpoint_sha256: str | None = None
        if provenance is not None and provenance.checkpoint.available:
            checkpoint_sha256 = provenance.checkpoint.expected_sha256
        base = {
            "submission_id": submission.submission_id,
            "submission_id_text": _submission_id_text(submission.submission_id),
            "checkpoint_sha256": checkpoint_sha256,
            "status": "unavailable",
            "recipe": None,
            "evidence": [],
        }
        if provenance is None or not provenance.available or checkpoint_sha256 is None:
            return {
                **base,
                "availability": _availability(
                    False, "training_recipe_checkpoint_provenance_unavailable"
                ),
            }
        with self._state_lock:
            registry = self._training_recipe_registry
            registry_error = self._training_recipe_registry_error
        if registry is None:
            return {
                **base,
                "availability": _availability(
                    False,
                    registry_error or "training_recipe_registry_unavailable",
                ),
            }
        record, reasons = registry.resolve_checkpoint(checkpoint_sha256)
        if record is None:
            return {
                **base,
                "availability": _availability(
                    False,
                    reasons[0]
                    if reasons
                    else "training_recipe_checkpoint_mapping_unavailable",
                ),
            }
        public = record.to_dict()
        return {
            **base,
            "checkpoint_sha256": record.checkpoint_sha256,
            "status": "available",
            "recipe": public["recipe"],
            "evidence": public["evidence"],
            "availability": _availability(True),
        }

    def training_recipe_payload(self, submission_id: int) -> dict[str, Any]:
        """Serve training-only recipe metadata without loading a model."""

        submission = self._submission_or_404(submission_id)
        return {
            "schema": API_SCHEMA,
            "submission_id": submission_id,
            "submission_id_text": _submission_id_text(submission_id),
            "training_recipe": self._training_recipe_for_submission(submission),
        }

    def submissions_payload(self) -> dict[str, Any]:
        catalog = self.catalog
        rows: list[dict[str, Any]] = []
        for submission in catalog.submissions:
            provenance = submission.provenance
            readiness = self._submission_readiness(submission)
            episode_count = len(submission.replays)
            submission_label = submission.submission_label.to_dict()
            rows.append(
                {
                    "submission_id": submission.submission_id,
                    "submission_id_text": _submission_id_text(submission.submission_id),
                    "label": submission_label["text"],
                    "submission_label": submission_label,
                    "cached_replay_outcomes": submission.cached_replay_outcomes.to_dict(),
                    "status": readiness["status"],
                    "weights": readiness["weights"],
                    "dynamic_trace": readiness["dynamic_trace"],
                    "episode_count": episode_count,
                    "archive_available": submission.archive_directory is not None,
                    "availability_reasons": list(submission.availability_reasons),
                    # Compatibility alias: model_analysis has always meant
                    # exact checkpoint/parameter readiness, not runtime-parity
                    # eligibility for a dynamic replay trace.
                    "model_analysis": readiness["weights"],
                    "provenance": provenance.to_dict() if provenance else None,
                    "training_recipe": self._training_recipe_for_submission(submission),
                }
            )
        return {
            "schema": API_SCHEMA,
            "submissions": rows,
            "archive": {
                "available": catalog.available,
                "availability_reasons": list(catalog.availability_reasons),
            },
        }

    def games_payload(self, submission_id: int) -> dict[str, Any]:
        submission = self._submission_or_404(submission_id)
        if submission.archive_directory is None:
            raise InspectorHTTPError(
                HTTPStatus.NOT_FOUND,
                "submission_archive_unavailable",
                "submission archive is unavailable",
                details={"submission_id": submission_id},
            )
        games: list[dict[str, Any]] = []
        for entry in submission.replays:
            own_seat, seat_reason = self._own_seat(entry.metadata, submission_id)
            step_count = self._replay_step_count(entry, submission)
            games.append(
                _public_game_metadata(
                    entry,
                    own_seat=own_seat,
                    seat_reason=seat_reason,
                    step_count=step_count,
                )
            )
        readiness = self._submission_readiness(submission)
        return {
            "schema": API_SCHEMA,
            "submission_id": submission_id,
            "submission_id_text": _submission_id_text(submission_id),
            "label": (
                submission.submission_label.text
                if submission.submission_label.available
                else None
            ),
            "submission_label": submission.submission_label.to_dict(),
            "cached_replay_outcomes": submission.cached_replay_outcomes.to_dict(),
            "status": readiness["status"],
            "weights": readiness["weights"],
            "dynamic_trace": readiness["dynamic_trace"],
            "model_analysis": readiness["weights"],
            "provenance": (
                submission.provenance.to_dict() if submission.provenance else None
            ),
            "training_recipe": self._training_recipe_for_submission(submission),
            "games": games,
        }

    def steps_payload(self, submission_id: int, episode_id: int) -> dict[str, Any]:
        submission, entry, replay, replay_sha256, seat = self._replay_with_seat(
            submission_id, episode_id
        )
        try:
            stages = self._owner_decision_stages(replay, own_seat=seat)
        except ReplayTimelineError as exc:
            raise InspectorHTTPError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "replay_timeline_unavailable",
                "recorded decisions cannot be reconstructed from this replay",
            ) from exc
        # A single environment decision can expand into more than one
        # factorized choice.  Keep one select-row per environment step and
        # attach every explicit stage so the UI cannot accidentally hide a
        # later target/card choice behind the first stage.
        grouped: dict[int, dict[str, Any]] = {}
        for stage in stages:
            row = grouped.setdefault(
                stage.step_index,
                {
                    "step_index": stage.step_index,
                    "turn": stage.turn,
                    "context": stage.context,
                    "candidate_count": (
                        stage.candidate_count
                        if hasattr(stage, "candidate_count")
                        else len(stage.candidates)
                    ),
                    "recorded_action_summary": list(stage.recorded_action),
                    "factorized_stages": [],
                },
            )
            row["factorized_stages"].append(
                {
                    "stage": stage.factorized_stage,
                    "candidate_count": len(stage.candidates),
                    "recorded_action": list(stage.recorded_action),
                    "recorded_stage_choice": list(stage.candidates[stage.target_index]),
                    "model_forward_expected": stage.model_forward_expected,
                    "status": (
                        "decision"
                        if stage.model_forward_expected
                        else "forced_turn_order_or_runtime_short_circuit"
                    ),
                }
            )
        return {
            "schema": API_SCHEMA,
            "submission_id": submission_id,
            "submission_id_text": _submission_id_text(submission_id),
            "episode_id": episode_id,
            "own_seat": seat,
            "decision_owner": _decision_owner_payload(
                entry.metadata,
                submission_id=submission_id,
                own_seat=seat,
            ),
            "selectable_step_count": len(grouped),
            "selectable_factorized_stage_count": len(stages),
            "model_analysis": _availability(
                self._model_analysis_reason(submission, entry, replay_sha256) is None,
                self._model_analysis_reason(submission, entry, replay_sha256),
            ),
            "steps": [grouped[index] for index in sorted(grouped)],
            "replay": {
                "available": entry.path is not None,
                "availability_reasons": list(entry.availability_reasons),
            },
        }

    def trace_payload(
        self,
        submission_id: int,
        episode_id: int,
        step_index: int,
        factorized_stage: int,
        head_scales: Mapping[str, float] | None = None,
        include_setup_model_forward: bool = True,
    ) -> dict[str, Any]:
        """Return one trace, joining a baseline physical-game materialization.

        A playground request is a distinct counterfactual and must never read
        the submitted-runtime baseline cache.  Baseline requests, on the
        other hand, are keyed by all immutable inputs that can change their
        result.  This method deliberately performs the exact input gates
        before either an in-memory or temporary-cache hit: caching avoids a
        GPU forward, never provenance verification.
        """

        if head_scales:
            return self._trace_payload_uncached(
                submission_id,
                episode_id,
                step_index,
                factorized_stage,
                head_scales=head_scales,
                include_setup_model_forward=include_setup_model_forward,
            )

        prepared = self._prepare_baseline_game_materialization(
            submission_id,
            episode_id,
            step_index,
            factorized_stage,
            include_setup_model_forward=include_setup_model_forward,
        )
        if prepared is None:
            # Preserve the existing detailed error/unavailable envelope for
            # an invalid address or a replay that is not trace-ready.  Only a
            # fully verified physical game gets a materialization identity.
            return self._trace_payload_uncached(
                submission_id,
                episode_id,
                step_index,
                factorized_stage,
                head_scales=None,
                include_setup_model_forward=include_setup_model_forward,
            )
        identity_key, identity, addresses = prepared
        address = (step_index, factorized_stage)

        # The persistent cache adapter intentionally has no authority to
        # select inputs.  It may return a payload only after the preparation
        # above revalidated the replay, submitted runtime, checkpoint, tree,
        # bundle, and parity receipt.
        cached = self._load_materialized_game_trace(
            identity_key, identity, addresses, address
        )
        if cached is not None:
            self._record_game_materialization_metric("l2_hits")
            return cached

        payload = self._join_baseline_game_materialization(
            identity_key,
            identity,
            addresses,
            requested_address=address,
            submission_id=submission_id,
            episode_id=episode_id,
            include_setup_model_forward=include_setup_model_forward,
        )
        if payload is not None:
            return payload

        # A job-level failure is deliberately not cached.  Re-run only the
        # requested trace so callers retain the established fail-closed
        # response instead of receiving a partial game cache as a success.
        return self._trace_payload_uncached(
            submission_id,
            episode_id,
            step_index,
            factorized_stage,
            head_scales=None,
            include_setup_model_forward=include_setup_model_forward,
        )

    def _prepare_baseline_game_materialization(
        self,
        submission_id: int,
        episode_id: int,
        step_index: int,
        factorized_stage: int,
        *,
        include_setup_model_forward: bool,
    ) -> tuple[str, dict[str, Any], tuple[tuple[int, int], ...]] | None:
        """Return a revalidated identity and ordered own-agent addresses.

        ``None`` intentionally means that the normal per-address route must
        produce its precise existing error/status.  It is not a cache hit and
        does not relax any provenance gate.
        """

        submission, entry, replay, replay_sha256, seat = self._replay_with_seat(
            submission_id, episode_id
        )
        try:
            stages = self._owner_decision_stages(replay, own_seat=seat)
        except ReplayTimelineError:
            return None
        addresses = tuple(
            (stage.step_index, stage.factorized_stage) for stage in stages
        )
        if (step_index, factorized_stage) not in addresses:
            return None
        if self._trace_analysis_reason(submission, entry, replay_sha256) is not None:
            return None
        provenance = submission.provenance
        if provenance is None:
            return None
        receipt = provenance.runtime_parity_receipt
        runtime_package = provenance.runtime_package
        artifacts: tuple[Any, ...] = (
            provenance.checkpoint,
            provenance.bundle,
            runtime_package,
            receipt.artifact if receipt is not None else None,
            provenance.matchup_tree,
        )
        # A catalog-time digest is not enough for a cache hit.  Recheck every
        # identity-bearing artifact now; a changed or unreadable artifact
        # simply declines the derived cache and follows the normal exact
        # reconstruction path.  The memo retains only a digest for an exact
        # resolved path plus its stable stat signature, so this still performs
        # current containment and signature validation on every request while
        # avoiding a second 500MiB bundle hash when the bundle and submitted
        # runtime package are the same immutable file.
        for artifact in artifacts:
            if artifact is None:
                continue
            expected = getattr(artifact, "expected_sha256", None)
            path = getattr(artifact, "resolved_path", None)
            if not isinstance(expected, str) or not isinstance(path, Path):
                return None
            verified = self._integrity_digest_memo.file_digest(
                path,
                roots=self.config.source_roots,
            )
            if verified is None or verified[1] != expected:
                return None
        if (
            receipt is None
            or receipt.runtime_source_tree_sha256 is None
            or runtime_package is None
            or runtime_package.expected_sha256 is None
            or provenance.checkpoint.expected_sha256 is None
            or provenance.bundle.expected_sha256 is None
            or receipt.artifact.expected_sha256 is None
        ):
            return None
        tree = provenance.matchup_tree
        identity: dict[str, Any] = {
            "schema": GAME_TRACE_MATERIALIZATION_SCHEMA,
            "trace_semantics_identity": "replay_inspector.server.trace_payload/v1",
            "submission_id": int(submission_id),
            "episode_id": int(episode_id),
            "own_seat": int(seat),
            "replay_sha256": replay_sha256,
            "submitted_bundle_sha256": provenance.bundle.expected_sha256,
            "runtime_package_sha256": runtime_package.expected_sha256,
            "runtime_source_tree_sha256": receipt.runtime_source_tree_sha256,
            "checkpoint_sha256": provenance.checkpoint.expected_sha256,
            "matchup_tree_sha256": (
                tree.expected_sha256 if tree is not None else None
            ),
            "runtime_parity_receipt_sha256": receipt.artifact.expected_sha256,
            "baseline_trace_request": {
                "head_scales": {},
                "include_setup_model_forward": bool(include_setup_model_forward),
            },
        }
        encoded = json.dumps(
            identity,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        identity_key = "sha256:" + hashlib.sha256(encoded).hexdigest()
        return identity_key, identity, addresses

    def _load_materialized_game_trace(
        self,
        identity_key: str,
        identity: Mapping[str, Any],
        addresses: Sequence[tuple[int, int]],
        address: tuple[int, int],
    ) -> dict[str, Any] | None:
        """Read a verified L2 entry when the cache implementation is present.

        The concrete temporary-cache module is intentionally the only layer
        allowed to own paths, compression, eviction, atomic writes, and cache
        checksums.  Server integration is kept here as a narrow adapter while
        that module is introduced; returning ``None`` is always a safe cache
        miss and preserves exact reconstruction.
        """

        _ = identity_key
        cache_identity = self._game_trace_cache_identity(identity)
        cache = self._game_trace_cache()
        if cache_identity is None or cache is None:
            return None
        try:
            from .game_trace_cache import TraceAddress

            expected_addresses = tuple(
                TraceAddress(step_index, factorized_stage)
                for step_index, factorized_stage in addresses
            )
            manifest = cache.read_manifest(cache_identity)
            if manifest is None or set(manifest.addresses) != set(expected_addresses):
                return None
            requested_address = TraceAddress(address[0], address[1])
            if requested_address not in set(expected_addresses):
                return None
            cached = cache.read(cache_identity, requested_address)
            value = cached.value if getattr(cached, "hit", False) else None
            if not isinstance(value, Mapping):
                return None
        except Exception:  # noqa: BLE001 - derived cache failure is a miss
            return None
        return dict(value)

    def _game_trace_cache(self) -> Any | None:
        """Return the configured private derived-payload cache, if usable."""

        if not self.config.game_trace_cache_enabled:
            return None
        with self._state_lock:
            if self._game_trace_cache_backend is not None:
                return self._game_trace_cache_backend
            if self._game_trace_cache_error is not None:
                return None
            try:
                from .game_trace_cache import PhysicalGameTraceCache

                self._game_trace_cache_backend = (
                    PhysicalGameTraceCache.from_inspector_config(self.config)
                )
            except Exception:  # noqa: BLE001 - /tmp cache is optional only
                self._game_trace_cache_error = "game_trace_cache_unavailable"
                _LOG.warning("Replay Inspector temporary game cache is unavailable")
                return None
            return self._game_trace_cache_backend

    @staticmethod
    def _game_trace_cache_identity(identity: Mapping[str, Any]) -> Any | None:
        """Map the server's full identity to the cache module's strict key."""

        try:
            from .game_trace_cache import GameTraceIdentity

            request = identity.get("baseline_trace_request")
            if not isinstance(request, Mapping):
                return None
            provenance = {
                "cache_schema": str(identity["schema"]),
                "trace_semantics_identity": str(
                    identity["trace_semantics_identity"]
                ),
                "own_seat": str(identity["own_seat"]),
                "submitted_bundle_sha256": str(identity["submitted_bundle_sha256"]),
                "runtime_package_sha256": str(identity["runtime_package_sha256"]),
                "runtime_source_tree_sha256": str(
                    identity["runtime_source_tree_sha256"]
                ),
                "checkpoint_sha256": str(identity["checkpoint_sha256"]),
                "matchup_tree_sha256": str(
                    identity.get("matchup_tree_sha256") or "none"
                ),
                "runtime_parity_receipt_sha256": str(
                    identity["runtime_parity_receipt_sha256"]
                ),
                "baseline_head_scales": "empty",
                "include_setup_model_forward": (
                    "true"
                    if bool(request.get("include_setup_model_forward"))
                    else "false"
                ),
            }
            return GameTraceIdentity(
                submission_id=int(identity["submission_id"]),
                episode_id=int(identity["episode_id"]),
                replay_sha256=str(identity["replay_sha256"]),
                provenance=provenance,
            )
        except (ImportError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _cacheable_materialized_trace(payload: Mapping[str, Any]) -> bool:
        """Persist only complete exact baseline results, never failures."""

        top_level = payload.get("availability")
        model = payload.get("model")
        model_availability = (
            model.get("availability") if isinstance(model, Mapping) else None
        )
        status = payload.get("reproduction_status")
        return bool(
            isinstance(top_level, Mapping)
            and top_level.get("available") is True
            and isinstance(model_availability, Mapping)
            and model_availability.get("available") is True
            and status
            in {
                "recomputed_not_historical",
                "exact_runtime_short_circuit",
                "hypothetical_model_forward_not_submitted_runtime",
            }
        )

    def _commit_materialized_game_cache(self, job: _GameMaterializationJob) -> None:
        """Publish one all-address manifest after a complete valid game job."""

        if not self.config.game_trace_cache_enabled:
            return
        if set(job.payloads) != set(job.addresses):
            return
        if not all(
            self._cacheable_materialized_trace(payload)
            for payload in job.payloads.values()
        ):
            return
        cache_identity = self._game_trace_cache_identity(job.identity)
        cache = self._game_trace_cache()
        if cache_identity is None or cache is None:
            return
        try:
            from .game_trace_cache import TraceAddress

            cache_addresses = tuple(
                TraceAddress(step_index, factorized_stage)
                for step_index, factorized_stage in job.addresses
            )
            requested = cache_addresses[0]
            by_address = {
                TraceAddress(step_index, factorized_stage): dict(payload)
                for (step_index, factorized_stage), payload in job.payloads.items()
            }
            cache.get_or_materialize(
                cache_identity,
                cache_addresses,
                requested,
                lambda cache_address: by_address[cache_address],
            )
        except Exception:  # noqa: BLE001 - derived cache write cannot affect trace
            return

    def _join_baseline_game_materialization(
        self,
        identity_key: str,
        identity: dict[str, Any],
        addresses: tuple[tuple[int, int], ...],
        *,
        requested_address: tuple[int, int],
        submission_id: int,
        episode_id: int,
        include_setup_model_forward: bool,
    ) -> dict[str, Any] | None:
        """Start or join one selected physical-game baseline job.

        Only the worker thread drives exact forwards.  Every HTTP request for
        the same game either promotes its pending address or waits for the
        in-flight result, which prevents a rapid step/stage selection from
        launching another independent forward.
        """

        with self._game_materialization_ready:
            job = self._game_materialization_jobs.get(identity_key)
            if job is None:
                pending = list(addresses)
                if requested_address in pending:
                    pending.remove(requested_address)
                    pending.insert(0, requested_address)
                job = _GameMaterializationJob(
                    identity_key=identity_key,
                    identity=dict(identity),
                    addresses=addresses,
                    pending=pending,
                    payloads={},
                )
                self._game_materialization_jobs[identity_key] = job
                self._record_game_materialization_metric("jobs_started")
                thread = threading.Thread(
                    target=self._run_baseline_game_materialization,
                    kwargs={
                        "job": job,
                        "submission_id": submission_id,
                        "episode_id": episode_id,
                        "include_setup_model_forward": include_setup_model_forward,
                    },
                    name="replay-inspector-game-materialization",
                    daemon=True,
                )
                thread.start()
            else:
                self._record_game_materialization_metric("l1_joins")
                self._game_materialization_jobs.move_to_end(identity_key)
                if (
                    requested_address not in job.payloads
                    and requested_address in job.pending
                ):
                    # The current selected step has priority, but never
                    # interrupt an exact forward already under way.
                    job.pending.remove(requested_address)
                    job.pending.insert(0, requested_address)
            while (
                requested_address not in job.payloads
                and not job.complete
                and job.error is None
            ):
                self._game_materialization_ready.wait()
            result = job.payloads.get(requested_address)
            return dict(result) if isinstance(result, Mapping) else None

    def _run_baseline_game_materialization(
        self,
        *,
        job: _GameMaterializationJob,
        submission_id: int,
        episode_id: int,
        include_setup_model_forward: bool,
    ) -> None:
        """Materialize every baseline address once, globally serialized.

        The materializer does not claim that changing causal game states are a
        single neural forward.  It serializes one selected-game job and leaves
        the exact per-game batching implementation behind the raw builder,
        which can coalesce shared causal history without changing this cache
        or HTTP contract.
        """

        try:
            with self._game_materialization_lock:
                self._record_game_materialization_metric("backend_game_runs")
                batched = self._materialize_baseline_game_backend_traces(
                    submission_id,
                    episode_id,
                    job.addresses,
                    include_setup_model_forward=include_setup_model_forward,
                )
                while True:
                    with self._game_materialization_ready:
                        if not job.pending:
                            break
                        address = job.pending.pop(0)
                        job.active_address = address
                    precomputed = (
                        batched.pop(address, None)
                        if isinstance(batched, dict)
                        else batched.get(address)
                        if isinstance(batched, Mapping)
                        else None
                    )
                    disk_payload = (
                        self._load_materialized_game_trace(
                            job.identity_key,
                            job.identity,
                            job.addresses,
                            address,
                        )
                        if precomputed is None
                        else None
                    )
                    if disk_payload is not None:
                        payload = disk_payload
                    else:
                        payload = self._trace_payload_uncached(
                            submission_id,
                            episode_id,
                            address[0],
                            address[1],
                            head_scales=None,
                            include_setup_model_forward=include_setup_model_forward,
                            precomputed_trace=(
                                precomputed
                                if isinstance(precomputed, Mapping)
                                else None
                            ),
                            prevalidated_game=isinstance(precomputed, Mapping),
                        )
                    with self._game_materialization_ready:
                        job.payloads[address] = dict(payload)
                        job.active_address = None
                        self._game_materialization_ready.notify_all()
                # The manifest is the only durable game-cache commit point;
                # do not publish individual rows while materialization is
                # still partial.
                self._commit_materialized_game_cache(job)
        except BaseException as exc:  # noqa: BLE001 - preserve fail-closed trace path
            with self._game_materialization_ready:
                job.error = exc
                job.active_address = None
                job.complete = True
                self._record_game_materialization_metric("jobs_completed")
                self._game_materialization_ready.notify_all()
            return
        with self._game_materialization_ready:
            job.complete = True
            job.active_address = None
            self._record_game_materialization_metric("jobs_completed")
            self._game_materialization_jobs.move_to_end(job.identity_key)
            self._trim_completed_game_materializations_locked()
            self._game_materialization_ready.notify_all()

    def _materialize_baseline_game_backend_traces(
        self,
        submission_id: int,
        episode_id: int,
        addresses: Sequence[tuple[int, int]],
        *,
        include_setup_model_forward: bool,
    ) -> dict[tuple[int, int], dict[str, Any]] | None:
        """Return exact backend traces for one game when its batch API exists.

        This is intentionally an optional accelerator.  A legacy exact
        runtime package without the new game entry point continues through the
        proven single-trace reconstruction path, while a current package gets
        one shared model activation and coalesced causal-history forwards.
        """

        try:
            submission, entry, replay, replay_sha256, seat = self._replay_with_seat(
                submission_id, episode_id
            )
            stages = self._owner_decision_stages(replay, own_seat=seat)
            stage_by_address = {
                (stage.step_index, stage.factorized_stage): stage for stage in stages
            }
            contexts: list[ReplayContext] = []
            for address in addresses:
                stage = stage_by_address.get(address)
                if stage is None:
                    return None
                contexts.append(
                    ReplayContext(
                        submission=submission,
                        replay_entry=entry,
                        replay=replay,
                        replay_sha256=replay_sha256,
                        seat=seat,
                        stage=stage,
                        observation=causal_observation(
                            replay, seat=seat, step_index=stage.step_index
                        ),
                    )
                )
        except (InspectorHTTPError, ReplayTimelineError, KeyError, TypeError, ValueError):
            return None
        return self._inspect_exact_game(
            contexts,
            allow_setup_prompt_model_forward=include_setup_model_forward,
        )

    def _trim_completed_game_materializations_locked(self) -> None:
        """Keep L1 bounded; durable reuse belongs to the temporary L2 cache."""

        while len(self._game_materialization_jobs) > self._game_materialization_memory_limit:
            oldest_key, oldest = next(iter(self._game_materialization_jobs.items()))
            if not oldest.complete:
                return
            self._game_materialization_jobs.pop(oldest_key)

    def _trace_payload_uncached(
        self,
        submission_id: int,
        episode_id: int,
        step_index: int,
        factorized_stage: int,
        head_scales: Mapping[str, float] | None = None,
        include_setup_model_forward: bool = True,
        precomputed_trace: Mapping[str, Any] | None = None,
        prevalidated_game: bool = False,
    ) -> dict[str, Any]:
        submission, entry, replay, replay_sha256, seat = self._replay_with_seat(
            submission_id, episode_id
        )
        try:
            stages = self._owner_decision_stages(replay, own_seat=seat)
        except ReplayTimelineError as exc:
            # An ACTIVE+select row without an exact corroborating actor is an
            # archive integrity failure, not an alternate decision address.
            raise InspectorHTTPError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "replay_timeline_unavailable",
                "recorded decisions cannot be reconstructed from this replay",
            ) from exc
        try:
            stage = next(
                (
                    candidate
                    for candidate in stages
                    if candidate.step_index == step_index
                    and candidate.factorized_stage == factorized_stage
                ),
                None,
            )
            if stage is None:
                raise KeyError("owner decision stage was not found")
            observation = causal_observation(replay, seat=seat, step_index=step_index)
        except KeyError as exc:
            if self._is_opponent_decision_address(
                replay,
                own_seat=seat,
                step_index=step_index,
                factorized_stage=factorized_stage,
            ):
                # The URL has no user-controlled seat parameter.  Still make
                # this rejection explicit when its address is a real opponent
                # decision, rather than ever attributing it to our checkpoint.
                raise InspectorHTTPError(
                    HTTPStatus.NOT_FOUND,
                    "opponent_decision_not_selectable",
                    "only decisions made by the archived submission seat can be inspected",
                    details={
                        "submission_id": submission_id,
                        "episode_id": episode_id,
                        "own_seat": seat,
                        "step_index": step_index,
                        "factorized_stage": factorized_stage,
                    },
                ) from exc
            raise InspectorHTTPError(
                HTTPStatus.NOT_FOUND,
                "decision_stage_not_found",
                "recorded decision stage was not found",
                details={
                    "submission_id": submission_id,
                    "episode_id": episode_id,
                    "step_index": step_index,
                    "factorized_stage": factorized_stage,
                },
            ) from exc
        context = ReplayContext(
            submission=submission,
            replay_entry=entry,
            replay=replay,
            replay_sha256=replay_sha256,
            seat=seat,
            stage=stage,
            observation=observation,
        )
        legal_options = [
            {
                "index": index,
                "action": list(candidate),
                "is_recorded": index == stage.target_index,
                "is_model_choice": False,
            }
            for index, candidate in enumerate(stage.candidates)
        ]
        model: dict[str, Any]
        heads: list[Any]
        fusion: Any = None
        decision_influence: Any = {
            "availability": _availability(
                False, "exact decision influence is unavailable"
            )
        }
        guide_shadow: Any = {
            "availability": _availability(
                False, "guide shadow is unavailable until a neural trace is run"
            ),
            "policy_authority": False,
            "policy_logit_delta": 0.0,
        }
        warnings: list[str] = []
        provenance_payload: Any = (
            submission.provenance.to_dict() if submission.provenance else None
        )
        reproduction_status = "unavailable"
        actual_runtime_decision: dict[str, Any] | None = None
        if not stage.model_forward_expected:
            for option in legal_options:
                is_selected = option["index"] == stage.target_index
                option["probability"] = 1.0 if is_selected else 0.0
                option["is_model_choice"] = is_selected
                option["decision_source"] = (
                    "deterministic_submitted_runtime_short_circuit"
                )
            model = {
                "availability": _availability(True),
                "status": "deterministic_runtime_short_circuit",
                "decision_source": "deterministic_submitted_runtime_short_circuit",
                "neural_model_forward": False,
                "selected_index": stage.target_index,
                "selected_action": list(stage.candidates[stage.target_index]),
                "agreement": True,
                "value": None,
                "route": None,
                "fusion": None,
                "adapter": {
                    "status": "bypassed",
                    "enabled": False,
                    "reason": "setup prompt is answered before model and matchup routing",
                },
            }
            heads = []
            decision_influence = {
                "availability": _availability(
                    False, "deterministic runtime choice has no neural head influence"
                )
            }
            guide_shadow = {
                "availability": _availability(
                    False,
                    "setup prompt is answered before guide or neural policy evaluation",
                ),
                "policy_authority": False,
                "policy_logit_delta": 0.0,
            }
            reproduction_status = "exact_runtime_short_circuit"
            warnings.append(
                "The submitted runtime answers this setup prompt deterministically before the neural model runs. The displayed 100% / 0% choice is exact runtime behavior; no neural logits, value, heads, or fusion routes exist for this step."
            )
            actual_runtime_decision = {
                "availability": _availability(True),
                "status": "exact_runtime_short_circuit",
                "selected_index": stage.target_index,
                "selected_action": list(stage.candidates[stage.target_index]),
                "probabilities": [
                    1.0 if index == stage.target_index else 0.0
                    for index in range(len(stage.candidates))
                ],
                "neural_model_forward": False,
            }
            # A game materializer revalidated this exact identity before its
            # one backend batch and supplied the resulting raw row.  Repeating
            # the full runtime-tree gate for every final presentation row would
            # turn a 124-stage game into 124 identical metadata walks.  This
            # internal flag never applies to an external cache hit: those still
            # enter through ``_prepare_baseline_game_materialization`` first.
            model_reason = (
                None
                if prevalidated_game and precomputed_trace is not None
                else self._trace_analysis_reason(submission, entry, replay_sha256)
            )
            if include_setup_model_forward and model_reason is None:
                hypothetical = (
                    dict(precomputed_trace)
                    if precomputed_trace is not None
                    else self._inspect_exact_trace(
                        context,
                        head_scales=head_scales,
                        allow_setup_prompt_model_forward=True,
                    )
                )
                hypothetical_model = hypothetical.get("model")
                if (
                    isinstance(hypothetical_model, Mapping)
                    and isinstance(hypothetical_model.get("availability"), Mapping)
                    and hypothetical_model["availability"].get("available") is True
                    and isinstance(hypothetical.get("legal_options"), list)
                ):
                    model = dict(hypothetical_model)
                    model["status"] = "hypothetical_neural_rerun"
                    model["historical_execution"] = False
                    model["actual_runtime_short_circuit"] = actual_runtime_decision
                    legal_options = [
                        dict(option) for option in hypothetical["legal_options"]
                    ]
                    for index, option in enumerate(legal_options):
                        option["actual_runtime_probability"] = (
                            1.0 if index == stage.target_index else 0.0
                        )
                        option["probability_semantics"] = (
                            "hypothetical_neural_softmax_if_model_were_called"
                        )
                    heads = list(hypothetical.get("heads") or [])
                    fusion = hypothetical.get("fusion")
                    decision_influence = hypothetical.get(
                        "decision_influence", decision_influence
                    )
                    guide_shadow = hypothetical.get("guide_shadow", guide_shadow)
                    if hypothetical.get("provenance") is not None:
                        provenance_payload = hypothetical["provenance"]
                    reproduction_status = (
                        "hypothetical_model_forward_not_submitted_runtime"
                    )
                    warnings = [
                        "Actual Kaggle runtime: this setup prompt was answered deterministically before the model ran.",
                        "Hypothetical neural rerun: the displayed model probabilities were freshly calculated from the checksum-bound archived model as if it had been called. They are not historical submitted-runtime probabilities.",
                        *list(hypothetical.get("warnings") or []),
                    ]
                else:
                    unavailable_model = hypothetical.get("model")
                    unavailable_availability = (
                        unavailable_model.get("availability")
                        if isinstance(unavailable_model, Mapping)
                        else None
                    )
                    hypothetical_reason = (
                        unavailable_availability.get("reason")
                        if isinstance(unavailable_availability, Mapping)
                        else "hypothetical_setup_model_forward_unavailable"
                    )
                    warnings.append(
                        "Hypothetical neural rerun is unavailable: "
                        f"{hypothetical_reason}."
                    )
        else:
            model_reason = (
                None
                if prevalidated_game and precomputed_trace is not None
                else self._trace_analysis_reason(submission, entry, replay_sha256)
            )
            if model_reason is not None:
                model = {
                    "availability": _availability(False, model_reason),
                    "status": "unavailable",
                    "selected_index": None,
                    "selected_action": None,
                    "agreement": None,
                    "value": None,
                    "route": None,
                    "fusion": None,
                }
                heads = []
                warnings.append(
                    "Exact submitted-model provenance is unavailable; replay data is shown without a substituted model result."
                )
            else:
                trace = (
                    dict(precomputed_trace)
                    if precomputed_trace is not None
                    else self._inspect_exact_trace(context, head_scales=head_scales)
                )
                model = trace["model"]
                heads = trace["heads"]
                fusion = trace["fusion"]
                decision_influence = trace.get("decision_influence", decision_influence)
                guide_shadow = trace.get("guide_shadow", guide_shadow)
                warnings.extend(trace["warnings"])
                if isinstance(trace.get("legal_options"), list):
                    legal_options = trace["legal_options"]
                if trace.get("provenance") is not None:
                    provenance_payload = trace["provenance"]
                reproduction_status = str(
                    trace.get("reproduction_status") or "unavailable"
                )
        legal_options = _with_action_transcripts(
            legal_options,
            observation=observation,
            factorized_stage=stage.factorized_stage,
            recorded_action=stage.recorded_action,
        )
        heads = _with_policy_influence_labels(heads, legal_options)
        decision_influence = _with_decision_influence_labels(
            decision_influence if isinstance(decision_influence, Mapping) else None,
            legal_options,
        )
        if isinstance(model, dict):
            raw_adapter = model.get("adapter")
            model["adapter_status"] = _with_adapter_policy_influence_labels(
                _adapter_status(
                    raw_adapter if isinstance(raw_adapter, Mapping) else None
                ),
                legal_options,
            )
            selected_index = _exact_int(model.get("selected_index"))
            if selected_index is not None:
                selected_option = _indexed(legal_options, selected_index)
                if isinstance(selected_option, Mapping):
                    model["selected_action_raw"] = selected_option.get("raw_action")
                    model["selected_action_transcript"] = selected_option.get(
                        "action_transcript"
                    )
                    model["selected_action_transcript_availability"] = (
                        selected_option.get("action_transcript_availability")
                    )
        recorded_action = stage.to_dict(include_candidates=True)
        recorded_action.update(
            {
                "action": list(stage.recorded_action),
                "raw_action": list(stage.recorded_action),
                "selected_index": stage.target_index,
                "selected_action": list(stage.candidates[stage.target_index]),
            }
        )
        selected_legal_option = _indexed(legal_options, stage.target_index)
        if isinstance(selected_legal_option, Mapping):
            recorded_action["selected_action_raw"] = selected_legal_option.get(
                "raw_action"
            )
            recorded_action["selected_action_transcript"] = selected_legal_option.get(
                "action_transcript"
            )
            recorded_action["selected_action_transcript_availability"] = (
                selected_legal_option.get("action_transcript_availability")
            )
        payload: dict[str, Any] = {
            "schema": API_SCHEMA,
            # The replay itself is addressable even when exact dynamic model
            # values are unavailable, so keep the top-level trace browsable.
            "availability": _availability(True),
            "submission_id": submission_id,
            "submission_id_text": _submission_id_text(submission_id),
            "address": {
                "submission_id": submission_id,
                "submission_id_text": _submission_id_text(submission_id),
                "episode_id": episode_id,
                "own_seat": seat,
                "decision_owner": _decision_owner_payload(
                    entry.metadata,
                    submission_id=submission_id,
                    own_seat=seat,
                ),
                "step_index": step_index,
                "factorized_stage": factorized_stage,
            },
            "provenance": _json_safe(provenance_payload),
            "training_recipe": self._training_recipe_for_submission(submission),
            "reproduction_status": reproduction_status,
            # This is intentionally the masked, acting-seat observation only.
            # Never add replay["steps"][...][other seat] or visualize.current.
            "observation": _json_safe(observation),
            "recorded_action": recorded_action,
            "legal_options": _json_safe(legal_options),
            "model": _json_safe(model),
            "heads": _json_safe(heads),
            "fusion": _json_safe(fusion),
            "decision_influence": _json_safe(decision_influence),
            "guide_shadow": _json_safe(guide_shadow),
            "actual_runtime_decision": _json_safe(actual_runtime_decision),
            "warnings": warnings,
        }
        return payload

    def parameters_payload(self, submission_id: int) -> dict[str, Any]:
        """Inspect one resident model under the full one-model lock."""

        with self._inference_lock:
            return self._parameters_payload_locked(submission_id)

    def _parameters_payload_locked(self, submission_id: int) -> dict[str, Any]:
        submission = self._submission_or_404(submission_id)
        provenance = submission.provenance
        inspector, reason = self._parameter_inspector(submission)
        if inspector is None:
            return {
                "schema": API_SCHEMA,
                "submission_id": submission_id,
                "submission_id_text": _submission_id_text(submission_id),
                "provenance": provenance.to_dict() if provenance else None,
                "availability": _availability(False, reason),
                "parameters": [],
                "inventory": {
                    "tensor_count": 0,
                    "parameter_count": 0,
                    "buffer_count": 0,
                    "element_count": 0,
                },
                "warnings": [
                    "Exact checkpoint parameters are unavailable; no alternate model was selected."
                ],
            }
        try:
            inventory = inspector.inventory()
        except Exception:  # noqa: BLE001 - optional tensor backend boundary
            return self._parameters_runtime_failure(submission_id, provenance)
        raw_tensors = (
            inventory.get("tensors", []) if isinstance(inventory, Mapping) else []
        )
        parameters: list[dict[str, Any]] = []
        if isinstance(raw_tensors, list):
            for raw in raw_tensors:
                if not isinstance(raw, Mapping):
                    continue
                item = dict(raw)
                item["trainable"] = bool(item.get("requires_grad", False))
                parameters.append(item)
        return {
            "schema": API_SCHEMA,
            "submission_id": submission_id,
            "submission_id_text": _submission_id_text(submission_id),
            "provenance": provenance.to_dict() if provenance else None,
            "availability": _availability(True),
            "parameters": _json_safe(parameters),
            "inventory": _json_safe(inventory),
            "warnings": [],
        }

    def parameter_detail_payload(
        self,
        submission_id: int,
        name: str,
        *,
        offset: int,
        limit: int,
        bins: int,
    ) -> dict[str, Any]:
        """Inspect one tensor while preventing cross-submission eviction."""

        with self._inference_lock:
            return self._parameter_detail_payload_locked(
                submission_id,
                name,
                offset=offset,
                limit=limit,
                bins=bins,
            )

    def _parameter_detail_payload_locked(
        self,
        submission_id: int,
        name: str,
        *,
        offset: int,
        limit: int,
        bins: int,
    ) -> dict[str, Any]:
        submission = self._submission_or_404(submission_id)
        provenance = submission.provenance
        inspector, reason = self._parameter_inspector(submission)
        if inspector is None:
            return {
                "schema": API_SCHEMA,
                "submission_id": submission_id,
                "submission_id_text": _submission_id_text(submission_id),
                "name": name,
                "provenance": provenance.to_dict() if provenance else None,
                "availability": _availability(False, reason),
                "parameter": {
                    "availability": _availability(False, reason),
                    "name": name,
                },
                "warnings": [
                    "Exact checkpoint parameters are unavailable; no alternate model was selected."
                ],
            }
        try:
            summary = inspector.summary(name)
            histogram = inspector.histogram(name, bins=bins)
            tensor_slice = inspector.slice(name, offset=offset, limit=limit)
        except Exception as exc:
            # ParameterInspectionError is intentionally imported only after a
            # checksum-bound model is available.  Match its public condition
            # without importing torch on archive-only requests.
            if type(exc).__name__ == "ParameterInspectionError":
                raise InspectorHTTPError(
                    HTTPStatus.NOT_FOUND,
                    "model_tensor_not_found",
                    "requested model tensor is unavailable",
                ) from exc
            return self._parameter_detail_runtime_failure(
                submission_id, name, provenance
            )
        return {
            "schema": API_SCHEMA,
            "submission_id": submission_id,
            "submission_id_text": _submission_id_text(submission_id),
            "name": name,
            "provenance": provenance.to_dict() if provenance else None,
            "availability": _availability(True),
            "parameter": _json_safe(
                {
                    **dict(summary),
                    "trainable": bool(summary.get("requires_grad", False)),
                    "stats": dict(summary),
                    "histogram": histogram,
                    "slice": tensor_slice,
                }
            ),
            "warnings": [],
        }

    @staticmethod
    def _parameters_runtime_failure(
        submission_id: int,
        provenance: SubmissionProvenance | None,
    ) -> dict[str, Any]:
        return {
            "schema": API_SCHEMA,
            "submission_id": submission_id,
            "submission_id_text": _submission_id_text(submission_id),
            "provenance": provenance.to_dict() if provenance else None,
            "availability": _availability(False, "parameter_inspection_failed"),
            "parameters": [],
            "inventory": {
                "tensor_count": 0,
                "parameter_count": 0,
                "buffer_count": 0,
                "element_count": 0,
            },
            "warnings": ["Parameter inspection could not be completed."],
        }

    @staticmethod
    def _parameter_detail_runtime_failure(
        submission_id: int,
        name: str,
        provenance: SubmissionProvenance | None,
    ) -> dict[str, Any]:
        return {
            "schema": API_SCHEMA,
            "submission_id": submission_id,
            "submission_id_text": _submission_id_text(submission_id),
            "name": name,
            "provenance": provenance.to_dict() if provenance else None,
            "availability": _availability(False, "parameter_inspection_failed"),
            "parameter": {
                "availability": _availability(False, "parameter_inspection_failed"),
                "name": name,
            },
            "warnings": ["Parameter inspection could not be completed."],
        }

    def _submission_or_404(self, submission_id: int) -> SubmissionCatalogEntry:
        submission = self.catalog.submission(submission_id)
        if submission is None:
            raise InspectorHTTPError(
                HTTPStatus.NOT_FOUND,
                "submission_not_found",
                "submission was not found in the configured replay archive",
                details={"submission_id": submission_id},
            )
        return submission

    def _replay_with_seat(
        self, submission_id: int, episode_id: int
    ) -> tuple[SubmissionCatalogEntry, ReplayCatalogEntry, dict[str, Any], str, int]:
        submission = self._submission_or_404(submission_id)
        entry = submission.replay(episode_id)
        if entry is None:
            raise InspectorHTTPError(
                HTTPStatus.NOT_FOUND,
                "episode_not_found",
                "episode was not found for this submission",
                details={"submission_id": submission_id, "episode_id": episode_id},
            )
        replay, replay_sha256 = self._read_replay(submission, entry)
        seat, reason = self._own_seat(entry.metadata, submission_id)
        if seat is None:
            raise InspectorHTTPError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "own_seat_unavailable",
                "the replay metadata does not explicitly identify the acting seat",
                details={"reason": reason or "own_seat_not_declared"},
            )
        return submission, entry, replay, replay_sha256, seat

    @staticmethod
    def _owner_decision_stages(
        replay: dict[str, Any], *, own_seat: int
    ) -> list[DecisionStage]:
        """Return only archive-declared own-seat decisions for selection.

        ``decision_timeline`` reads the action at the next Kaggle row and
        requires the requested seat itself to be active.  Crucially, this
        function returns addresses only; callers retain and pass the original
        full replay to causal reconstruction so opponent transitions remain
        part of history rather than becoming candidate model decisions.
        """

        return decision_timeline(replay, own_seat)

    @classmethod
    def _is_opponent_decision_address(
        cls,
        replay: dict[str, Any],
        *,
        own_seat: int,
        step_index: int,
        factorized_stage: int,
    ) -> bool:
        """Tell an invalid own-seat URL from an address owned by another seat.

        This diagnostic path is deliberately best-effort.  A malformed
        opponent row can never create a selectable stage, and a malformed
        archive does not disclose a hidden alternate actor through an error.
        """

        steps = replay.get("steps")
        if (
            not isinstance(steps, list)
            or not (0 <= step_index < len(steps))
            or not isinstance(steps[step_index], list)
        ):
            return False
        for candidate_seat in range(len(steps[step_index])):
            if candidate_seat == own_seat:
                continue
            try:
                stages = cls._owner_decision_stages(replay, own_seat=candidate_seat)
            except ReplayTimelineError:
                continue
            if any(
                stage.step_index == step_index
                and stage.factorized_stage == factorized_stage
                for stage in stages
            ):
                return True
        return False

    @staticmethod
    def _own_seat(
        metadata: Mapping[str, Any] | None, submission_id: int
    ) -> tuple[int | None, str | None]:
        """Extract a declared own seat without inferring it from hidden rows."""

        if not isinstance(metadata, Mapping):
            return None, "episode_metadata_missing"
        candidates: list[int] = []

        def add(value: Any) -> None:
            if isinstance(value, bool):
                return
            try:
                parsed = int(value)
            except (TypeError, ValueError, OverflowError):
                return
            if parsed >= 0 and parsed not in candidates:
                candidates.append(parsed)

        seat_names = (
            "index",
            "agent_index",
            "player_index",
            "seat",
            "position",
        )
        for name in ("own_seat", "own_agent_index", "own_player_index", "player_index"):
            add(metadata.get(name))
        own_agent = metadata.get("own_agent")
        if isinstance(own_agent, Mapping):
            declared_submission = own_agent.get("submission_id")
            if declared_submission is not None:
                if isinstance(declared_submission, int) and not isinstance(
                    declared_submission, bool
                ):
                    parsed_submission = declared_submission
                elif (
                    isinstance(declared_submission, str)
                    and _DECIMAL_RE.fullmatch(declared_submission) is not None
                ):
                    parsed_submission = int(declared_submission)
                else:
                    return None, "own_agent_submission_id_invalid"
                if parsed_submission != int(submission_id):
                    return None, "own_agent_submission_id_mismatch"
            for name in seat_names:
                add(own_agent.get(name))

        # Some archive writers preserve an explicit agents list rather than an
        # own_agent object.  Use it only when its submission id is an exact
        # match; no index is inferred from list order.
        agents = metadata.get("agents")
        if isinstance(agents, list):
            for agent in agents:
                if not isinstance(agent, Mapping):
                    continue
                declared_submission = agent.get("submission_id")
                try:
                    matches = int(declared_submission) == int(submission_id)
                except (TypeError, ValueError):
                    matches = False
                if matches:
                    for name in seat_names:
                        add(agent.get(name))
        if len(candidates) == 1:
            return candidates[0], None
        if len(candidates) > 1:
            return None, "own_seat_ambiguous"
        return None, "own_seat_not_declared"

    def _read_replay(
        self,
        submission: SubmissionCatalogEntry,
        entry: ReplayCatalogEntry,
    ) -> tuple[dict[str, Any], str]:
        if entry.path is None or submission.archive_directory is None:
            raise InspectorHTTPError(
                HTTPStatus.NOT_FOUND,
                "replay_unavailable",
                "replay file is unavailable in the configured archive",
                details={"availability_reasons": list(entry.availability_reasons)},
            )
        try:
            path = resolve_contained_path(
                entry.path,
                roots=(submission.archive_directory,),
                require_exists=True,
            )
        except (FileNotFoundError, OSError, ProvenanceError) as exc:
            raise InspectorHTTPError(
                HTTPStatus.NOT_FOUND,
                "replay_unavailable",
                "replay file is unavailable in the configured archive",
            ) from exc
        if not path.is_file():
            raise InspectorHTTPError(
                HTTPStatus.NOT_FOUND,
                "replay_unavailable",
                "replay path is not a regular file",
            )
        try:
            # Hash the exact byte string that is parsed and later supplied to
            # the trace backend.  A separate hash-then-read sequence would
            # permit a replay replacement between the two operations.
            raw_bytes = path.read_bytes()
            replay_sha256 = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InspectorHTTPError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "replay_invalid_json",
                "replay file cannot be read as a JSON object",
            ) from exc
        if not isinstance(payload, dict):
            raise InspectorHTTPError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "replay_invalid_shape",
                "replay file must contain a JSON object",
            )
        return payload, replay_sha256

    def _replay_step_count(
        self,
        entry: ReplayCatalogEntry,
        submission: SubmissionCatalogEntry,
    ) -> int | None:
        """Best-effort list metadata; failures remain visible on the detail API."""

        try:
            replay, _replay_sha256 = self._read_replay(submission, entry)
        except InspectorHTTPError:
            return None
        steps = replay.get("steps")
        return len(steps) if isinstance(steps, list) else None

    @staticmethod
    def _model_analysis_reason(
        submission: SubmissionCatalogEntry,
        entry: ReplayCatalogEntry | None = None,
        replay_sha256: str | None = None,
    ) -> str | None:
        """Return a provenance-inclusive model eligibility reason, fail closed."""

        if entry is not None and not entry.model_analysis_available:
            return _reason_from(
                entry.model_analysis_availability_reasons,
                "exact_submission_provenance_unavailable",
            )
        if entry is not None and replay_sha256 is not None:
            expected = entry.expected_replay_sha256
            if expected is None:
                return "replay_provenance_missing"
            if replay_sha256 != expected:
                return "replay_sha256_mismatch"
        if not submission.model_analysis_available:
            return _reason_from(
                submission.model_analysis_availability_reasons,
                "exact_submission_provenance_unavailable",
            )
        provenance = submission.provenance
        if provenance is None:
            if len(submission.provenance_candidates) > 1:
                return "provenance_ambiguous"
            return "exact_submission_provenance_unavailable"
        if not provenance.available:
            return _reason_from(
                provenance.availability_reasons,
                "exact_submission_provenance_unavailable",
            )
        if len(submission.provenance_candidates) != 1:
            return "provenance_ambiguous"
        return None

    def _dynamic_trace_summary_reason(
        self,
        submission: SubmissionCatalogEntry,
        *,
        weights_reason: str | None,
    ) -> str | None:
        """Return a cheap, non-executing dynamic-trace readiness reason.

        This deliberately checks only static provenance/runtime-root facts for
        the submission selector.  The selected replay bytes, extracted module
        containment, source-tree digest, runtime-enabled adapter tree, and
        checkpoint are rechecked by :meth:`_trace_analysis_reason` on every
        trace request before anything can load or execute.
        """

        if weights_reason is not None:
            return weights_reason
        provenance = submission.provenance
        if provenance is None:
            return "exact_submission_provenance_unavailable"
        if not provenance.trace_available:
            return _reason_from(
                provenance.trace_availability_reasons,
                "runtime_parity_receipt_unavailable",
            )
        runtime_root = self.config.runtime_source_root
        if runtime_root is None:
            return "runtime_source_root_not_configured"
        try:
            root = runtime_root.expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            return "runtime_source_root_unavailable"
        if not root.is_dir():
            return "runtime_source_root_unavailable"
        return None

    def _submission_readiness(
        self, submission: SubmissionCatalogEntry
    ) -> dict[str, Any]:
        """Keep checkpoint/parameter and dynamic-trace readiness distinct."""

        weights_reason = self._model_analysis_reason(submission)
        dynamic_reason = self._dynamic_trace_summary_reason(
            submission, weights_reason=weights_reason
        )
        weights = _availability(weights_reason is None, weights_reason)
        dynamic_trace = _availability(dynamic_reason is None, dynamic_reason)
        return {
            "status": (
                "replay_only"
                if weights_reason is not None
                else "weights_only"
                if dynamic_reason is not None
                else "trace_ready"
            ),
            "weights": weights,
            "dynamic_trace": dynamic_trace,
        }

    def _trace_analysis_reason(
        self,
        submission: SubmissionCatalogEntry,
        entry: ReplayCatalogEntry | None = None,
        replay_sha256: str | None = None,
    ) -> str | None:
        """Require a verified submitted-runtime identity before executing code.

        Checkpoint provenance is enough for read-only tensor inspection.  It
        is intentionally *not* enough to run a forward pass, because importing
        a similarly shaped model from the service checkout would produce a
        misleading dynamic trace.  The parity receipt binds this submission to
        an exact runtime-package digest and the configured extracted source
        tree is containment-checked and restatted before each trace request;
        its source digest is recomputed on any stable-signature drift.
        """

        reason = self._model_analysis_reason(submission, entry, replay_sha256)
        if reason is not None:
            return reason
        provenance = submission.provenance
        if provenance is None:
            return "exact_submission_provenance_unavailable"
        if not provenance.trace_available:
            return _reason_from(
                provenance.trace_availability_reasons,
                "runtime_parity_receipt_unavailable",
            )
        receipt = provenance.runtime_parity_receipt
        if receipt is None:
            return "runtime_source_root_not_configured"
        root = self._runtime_source_root_for_submission(submission)
        if root is None:
            return "runtime_source_tree_sha256_mismatch"
        try:
            module_candidate = (
                root / "__init__.py"
                if root.name == "poke_bot"
                else root / "poke_bot" / "__init__.py"
            )
            module = resolve_contained_path(
                module_candidate,
                roots=(root,),
                require_exists=True,
            )
            if not module.is_file():
                return "runtime_package_module_path_unavailable"
        except (OSError, ProvenanceError, RuntimeError):
            return "runtime_identity_verification_failed"
        return None

    def _runtime_source_root_for_submission(
        self, submission: SubmissionCatalogEntry
    ) -> Path | None:
        """Resolve the exact extracted runtime attested for one submission.

        A provenance receipt deliberately carries no trusted path.  Resolve a
        path only from configured read-only roots, then require the full source
        tree digest from that receipt.  The submission's verified matchup-tree
        parent is the natural extracted package root; the configured legacy
        root remains a candidate for older records.
        """

        provenance = submission.provenance
        receipt = provenance.runtime_parity_receipt if provenance else None
        expected = receipt.runtime_source_tree_sha256 if receipt else None
        if expected is None:
            return None
        candidates: list[Path] = []
        if self.config.runtime_source_root is not None:
            candidates.append(self.config.runtime_source_root)
        tree = provenance.matchup_tree if provenance else None
        if tree is not None and tree.resolved_path is not None:
            candidates.append(tree.resolved_path.parent)
        seen: set[Path] = set()
        for candidate in candidates:
            verified = self._integrity_digest_memo.source_tree_digest(
                candidate,
                roots=self.config.source_roots,
            )
            if verified is None:
                continue
            root, digest = verified
            if root in seen:
                continue
            seen.add(root)
            if digest == expected:
                return root
        return None

    @staticmethod
    def _imported_runtime_is(root: Path) -> bool:
        """Return true only when the process imported ``poke_bot`` from root."""

        try:
            imported_package = importlib.import_module("poke_bot")
            module_file = getattr(imported_package, "__file__", None)
            if not isinstance(module_file, str) or not module_file:
                return False
            resolved = resolve_contained_path(
                module_file,
                roots=(root,),
                require_exists=True,
            )
            return resolved.is_file()
        except (ImportError, OSError, ProvenanceError, RuntimeError):
            return False

    @staticmethod
    def _runtime_pythonpath_root(runtime_root: Path) -> Path:
        """Return the import parent for either supported extracted layout.

        A receipt can bind the package directory itself (``.../poke_bot``) or
        a source-tree root containing that directory.  Python's import search
        path must always contain the *parent* of the ``poke_bot`` package;
        otherwise a game worker imports the inspector checkout and can recurse
        into another isolated worker instead of using its exact runtime.
        """

        return runtime_root.parent if runtime_root.name == "poke_bot" else runtime_root

    def _clear_resident_model(self) -> None:
        """Release the parent cache before an isolated submitted-runtime trace."""

        cache = self._model_cache
        clear = getattr(cache, "clear", None) if cache is not None else None
        if callable(clear):
            clear()

    def _inspect_exact_trace_isolated(
        self,
        context: ReplayContext,
        *,
        runtime_root: Path,
        head_scales: Mapping[str, float] | None,
        allow_setup_prompt_model_forward: bool = False,
    ) -> dict[str, Any]:
        """Run one trace under its exact submitted Python runtime.

        The enclosing inference lock serializes this with the resident cache.
        The cache is cleared first, so the service and child never retain two
        models concurrently.  The worker receives only a resolved replay
        address and immutable configured roots, never a client filesystem path.
        """

        self._clear_resident_model()
        worker = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / ("replay_inspector_trace_worker.py")
        )
        if not worker.is_file():
            return self._unavailable_trace("isolated_runtime_worker_unavailable")
        request = {
            "config": {
                "bind_host": self.config.bind_host,
                "port": self.config.port,
                "replay_root": str(self.config.replay_root),
                "rollout_root": str(self.config.rollout_root),
                "provenance_manifest": (
                    str(self.config.provenance_manifest)
                    if self.config.provenance_manifest is not None
                    else None
                ),
                "training_recipe_registry": (
                    str(self.config.training_recipe_registry)
                    if self.config.training_recipe_registry is not None
                    else None
                ),
                "artifact_roots": [str(path) for path in self.config.artifact_roots],
                "runtime_source_root": str(runtime_root),
                "web_root": str(self.config.web_root),
                "torch_threads": self.config.torch_threads,
                "max_parameter_slice": self.config.max_parameter_slice,
                "max_tensor_values": self.config.max_tensor_values,
                "verify_digests": self.config.verify_digests,
                "game_trace_cache_root": str(self.config.game_trace_cache_root),
                "game_trace_cache_enabled": self.config.game_trace_cache_enabled,
                "game_trace_cache_max_bytes": self.config.game_trace_cache_max_bytes,
                "game_trace_cache_max_game_bytes": self.config.game_trace_cache_max_game_bytes,
                "game_trace_cache_max_entry_bytes": self.config.game_trace_cache_max_entry_bytes,
                "game_trace_cache_min_free_bytes": self.config.game_trace_cache_min_free_bytes,
            },
            "mode": "trace",
            "submission_id": context.submission.submission_id,
            "episode_id": context.replay_entry.episode_id,
            "step_index": context.stage.step_index,
            "factorized_stage": context.stage.factorized_stage,
            "head_scales": dict(head_scales or {}),
            "allow_setup_prompt_model_forward": allow_setup_prompt_model_forward,
        }
        env = os.environ.copy()
        inspector_root = Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = os.pathsep.join(
            (str(self._runtime_pythonpath_root(runtime_root)), str(inspector_root))
        )
        try:
            completed = subprocess.run(
                [sys.executable, str(worker)],
                input=json.dumps(request, separators=(",", ":")),
                text=True,
                capture_output=True,
                cwd=runtime_root,
                env=env,
                timeout=240,
                check=False,
            )
            if completed.returncode != 0 or len(completed.stdout) > 32 * 1024 * 1024:
                return self._unavailable_trace("isolated_runtime_trace_failed")
            payload = json.loads(completed.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return self._unavailable_trace("isolated_runtime_trace_failed")
        if not isinstance(payload, Mapping):
            return self._unavailable_trace("isolated_runtime_trace_invalid_response")
        return dict(payload)

    @staticmethod
    def _unavailable_trace(reason: str) -> dict[str, Any]:
        return {
            "model": {
                "availability": _availability(False, reason),
                "status": "unavailable",
            },
            "heads": [],
            "fusion": None,
            "decision_influence": {"availability": _availability(False, reason)},
            "provenance": None,
            "reproduction_status": "unavailable",
            "warnings": [
                "Exact submitted-runtime trace execution failed; no alternate runtime was used."
            ],
        }

    def _inference_backend(self) -> ModuleType | None:
        with self._state_lock:
            if self._inference_module is not None:
                return self._inference_module
            if self._inference_import_error is not None:
                return None
            try:
                self._inference_module = importlib.import_module(
                    "replay_inspector.inference"
                )
            except Exception:  # noqa: BLE001 - optional CPU runtime boundary
                # A missing CPU torch build or an optional backend import error
                # must not take archive browsing down.  The raw exception can
                # contain host paths, so retain only a public code.
                self._inference_import_error = "inference_runtime_unavailable"
                return None
            return self._inference_module

    def _get_model_cache(self) -> tuple[Any | None, str | None]:
        """Create the optional verified one-model cache without fallback loads.

        ``replay_inspector.inference`` is owned by the model inspection layer.
        It exposes a strict ``VerifiedCpuModelCache`` specifically so this HTTP
        layer never calls a current-checkout model loader on its own.
        """

        with self._inference_lock:
            if self._model_cache is not None:
                return self._model_cache, None
            if self._model_cache_error is not None:
                return None, self._model_cache_error
            backend = self._inference_backend()
            if backend is None:
                self._model_cache_error = (
                    self._inference_import_error or "inference_runtime_unavailable"
                )
                return None, self._model_cache_error
            cache_type = getattr(backend, "VerifiedCpuModelCache", None)
            if cache_type is None:
                self._model_cache_error = "inference_runtime_contract_unavailable"
                return None, self._model_cache_error
            try:
                # The concrete backend accepts these two conservative controls.
                self._model_cache = cache_type(
                    torch_threads=self.config.torch_threads,
                    verify_digests=self.config.verify_digests,
                )
            except TypeError:
                try:
                    self._model_cache = cache_type(
                        torch_threads=self.config.torch_threads
                    )
                except TypeError:
                    try:
                        # The v1 cache itself is intentionally parameterless;
                        # process-level CPU limits are supplied by config's
                        # managed service environment.
                        self._model_cache = cache_type()
                    except Exception:  # noqa: BLE001 - optional backend construction
                        self._model_cache_error = (
                            "inference_runtime_initialization_failed"
                        )
                        return None, self._model_cache_error
                except Exception:  # noqa: BLE001 - optional backend construction
                    self._model_cache_error = "inference_runtime_initialization_failed"
                    return None, self._model_cache_error
            except Exception:  # noqa: BLE001 - optional backend construction
                self._model_cache_error = "inference_runtime_initialization_failed"
                return None, self._model_cache_error
            return self._model_cache, None

    def _load_exact_model(
        self,
        submission: SubmissionCatalogEntry,
        entry: ReplayCatalogEntry | None = None,
        replay_sha256: str | None = None,
    ) -> tuple[Any | None, str | None]:
        """Load exactly the checksum-bound checkpoint through the backend."""

        reason = self._model_analysis_reason(submission, entry, replay_sha256)
        provenance = submission.provenance
        if reason is not None or provenance is None:
            return None, reason or "exact_submission_provenance_unavailable"
        checkpoint = provenance.checkpoint
        if (
            checkpoint.resolved_path is None
            or checkpoint.expected_sha256 is None
            or not checkpoint.available
        ):
            return None, "checkpoint_provenance_unavailable"
        cache, cache_reason = self._get_model_cache()
        if cache is None:
            return None, cache_reason or "inference_runtime_unavailable"
        try:
            loaded = cache.load(checkpoint.resolved_path, checkpoint.expected_sha256)
        except Exception:  # noqa: BLE001 - checksum-bound loader boundary
            # A failed digest, unsupported package, or model load must not
            # silently leave a previous resident model selected.
            return None, "exact_checkpoint_load_failed"
        model = getattr(loaded, "model", loaded)
        if not callable(getattr(model, "named_parameters", None)):
            return None, "exact_checkpoint_load_failed"
        return model, None

    def _inspect_exact_game(
        self,
        contexts: Sequence[ReplayContext],
        *,
        allow_setup_prompt_model_forward: bool,
    ) -> dict[tuple[int, int], dict[str, Any]] | None:
        """Run one exact baseline materialization under one model residency.

        A submitted source tree that is not imported by this process remains
        isolated.  The isolated game-worker implementation is supplied below;
        returning ``None`` before that contract is available deliberately
        falls back to the existing one-address isolated execution rather than
        importing a foreign runtime into the server process.
        """

        if not contexts:
            return {}
        first = contexts[0]
        if any(
            context.submission.submission_id != first.submission.submission_id
            or context.replay_entry.episode_id != first.replay_entry.episode_id
            or context.replay_sha256 != first.replay_sha256
            or context.seat != first.seat
            for context in contexts[1:]
        ):
            return None
        with self._inference_lock:
            runtime_root = self._runtime_source_root_for_submission(first.submission)
            if runtime_root is None:
                return {
                    (context.stage.step_index, context.stage.factorized_stage): self._unavailable_trace(
                        "runtime_source_tree_sha256_mismatch"
                    )
                    for context in contexts
                }
            if not self._imported_runtime_is(runtime_root):
                # A game worker must never recursively launch another worker
                # if its exact runtime import is misconfigured.  Its parent
                # will treat these structured unavailable rows as a failed
                # exact reconstruction rather than importing an arbitrary
                # checkout or multiplying isolated model loads.
                if os.environ.get(_ISOLATED_GAME_WORKER_ENV) == "1":
                    return {
                        (
                            context.stage.step_index,
                            context.stage.factorized_stage,
                        ): self._unavailable_trace(
                            "isolated_game_worker_runtime_import_mismatch"
                        )
                        for context in contexts
                    }
                return self._inspect_exact_game_isolated(
                    contexts,
                    runtime_root=runtime_root,
                    allow_setup_prompt_model_forward=allow_setup_prompt_model_forward,
                )
            return self._inspect_exact_game_locked(
                contexts,
                allow_setup_prompt_model_forward=allow_setup_prompt_model_forward,
            )

    @staticmethod
    def _stop_owned_isolated_game_worker(child: subprocess.Popen[bytes]) -> None:
        """Contain only the noninteractive direct child created below.

        ``start_new_session=True`` and PIPE stdin mean this is never a user
        terminal, SSH, editor, or service session.  The method operates on the
        one ``Popen`` object it owns, never enumerates processes, and is used
        only after a malformed/stalled worker protocol or parent-side failure.
        """

        try:
            if child.poll() is not None:
                return
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # This remains the same direct noninteractive child; no
                # process tree or session discovery is involved.
                child.kill()
                child.wait(timeout=10)
        except (OSError, subprocess.SubprocessError):
            return

    def _read_isolated_game_worker_stream(
        self,
        child: subprocess.Popen[bytes],
        *,
        addresses: Sequence[tuple[int, int]],
    ) -> dict[tuple[int, int], dict[str, Any]] | None:
        """Consume bounded NDJSON from one exact-runtime game worker.

        Heartbeats bound an *idle* owned worker rather than total elapsed
        work: a legitimate long game can run indefinitely while it continues
        reporting liveness.  Frames are parsed as they arrive, so the parent
        never captures an unbounded whole-game stdout string.
        """

        stdout = child.stdout
        if stdout is None:
            self._stop_owned_isolated_game_worker(child)
            return None
        expected = set(addresses)
        received: dict[tuple[int, int], dict[str, Any]] = {}
        completed = False
        frame = bytearray()
        last_liveness = time.monotonic()
        stream_closed = False
        try:
            descriptor = stdout.fileno()
        except (OSError, ValueError):
            self._stop_owned_isolated_game_worker(child)
            return None

        while True:
            if stream_closed:
                try:
                    returncode = child.wait(timeout=_ISOLATED_GAME_WORKER_POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    if (
                        time.monotonic() - last_liveness
                        > _ISOLATED_GAME_WORKER_STALL_SECONDS
                    ):
                        self._stop_owned_isolated_game_worker(child)
                        return None
                    continue
                break

            try:
                ready, _, _ = select.select(
                    [descriptor], [], [], _ISOLATED_GAME_WORKER_POLL_SECONDS
                )
            except (OSError, ValueError):
                self._stop_owned_isolated_game_worker(child)
                return None
            now = time.monotonic()
            if not ready:
                if now - last_liveness > _ISOLATED_GAME_WORKER_STALL_SECONDS:
                    self._stop_owned_isolated_game_worker(child)
                    return None
                continue
            try:
                block = os.read(descriptor, 64 * 1024)
            except OSError:
                self._stop_owned_isolated_game_worker(child)
                return None
            if not block:
                stream_closed = True
                continue
            frame.extend(block)
            if len(frame) > _ISOLATED_GAME_WORKER_MAX_FRAME_BYTES:
                self._stop_owned_isolated_game_worker(child)
                return None
            while True:
                newline = frame.find(b"\n")
                if newline < 0:
                    break
                raw_line = bytes(frame[:newline])
                del frame[: newline + 1]
                if not raw_line:
                    continue
                if len(raw_line) > _ISOLATED_GAME_WORKER_MAX_FRAME_BYTES:
                    self._stop_owned_isolated_game_worker(child)
                    return None
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._stop_owned_isolated_game_worker(child)
                    return None
                if not isinstance(record, Mapping):
                    self._stop_owned_isolated_game_worker(child)
                    return None
                if record.get("protocol") != GAME_MATERIALIZATION_WORKER_PROTOCOL:
                    self._stop_owned_isolated_game_worker(child)
                    return None
                kind = record.get("kind")
                if kind == "start":
                    if record.get("address_count") != len(expected):
                        self._stop_owned_isolated_game_worker(child)
                        return None
                    last_liveness = now
                    continue
                if kind == "heartbeat":
                    last_liveness = now
                    continue
                if kind == "error":
                    self._stop_owned_isolated_game_worker(child)
                    return None
                if kind == "trace":
                    step_index = record.get("step_index")
                    factorized_stage = record.get("factorized_stage")
                    payload = record.get("payload")
                    if (
                        type(step_index) is not int
                        or type(factorized_stage) is not int
                        or not isinstance(payload, Mapping)
                    ):
                        self._stop_owned_isolated_game_worker(child)
                        return None
                    address = (step_index, factorized_stage)
                    if address not in expected or address in received:
                        self._stop_owned_isolated_game_worker(child)
                        return None
                    received[address] = dict(payload)
                    last_liveness = now
                    continue
                if kind == "complete":
                    if (
                        completed
                        or record.get("address_count") != len(expected)
                        or set(received) != expected
                    ):
                        self._stop_owned_isolated_game_worker(child)
                        return None
                    completed = True
                    last_liveness = now
                    continue
                self._stop_owned_isolated_game_worker(child)
                return None

        if returncode != 0 or not completed or set(received) != expected:
            return None
        return received

    def _inspect_exact_game_isolated(
        self,
        contexts: Sequence[ReplayContext],
        *,
        runtime_root: Path,
        allow_setup_prompt_model_forward: bool,
    ) -> dict[tuple[int, int], dict[str, Any]] | None:
        """Run a nonresident materialization in one owned exact-runtime worker.

        The worker streams bounded raw per-address traces.  The parent owns
        final HTTP-envelope assembly and optional cache commit, so a full,
        disabled, oversized, or unavailable temporary cache cannot turn one
        game worker into a fallback fan-out of isolated single-trace workers.

        There is deliberately no elapsed-total subprocess timeout here.  A
        long game legitimately needs more time than an isolated single trace,
        and the child heartbeat continuously resets an owned-worker idle
        watchdog.  Service lifecycle remains the managed cancellation
        boundary; the legacy one-trace worker retains its bounded watchdog.
        """

        if not contexts:
            return {}

        def unavailable(reason: str) -> dict[tuple[int, int], dict[str, Any]]:
            return {
                (item.stage.step_index, item.stage.factorized_stage): self._unavailable_trace(
                    reason
                )
                for item in contexts
            }

        context = contexts[0]
        worker = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "replay_inspector_trace_worker.py"
        )
        if not worker.is_file():
            return unavailable("isolated_runtime_worker_unavailable")
        addresses = [
            [item.stage.step_index, item.stage.factorized_stage] for item in contexts
        ]
        request = {
            "config": {
                "bind_host": self.config.bind_host,
                "port": self.config.port,
                "replay_root": str(self.config.replay_root),
                "rollout_root": str(self.config.rollout_root),
                "provenance_manifest": (
                    str(self.config.provenance_manifest)
                    if self.config.provenance_manifest is not None
                    else None
                ),
                "training_recipe_registry": (
                    str(self.config.training_recipe_registry)
                    if self.config.training_recipe_registry is not None
                    else None
                ),
                "artifact_roots": [str(path) for path in self.config.artifact_roots],
                "runtime_source_root": str(runtime_root),
                "web_root": str(self.config.web_root),
                "torch_threads": self.config.torch_threads,
                "max_parameter_slice": self.config.max_parameter_slice,
                "max_tensor_values": self.config.max_tensor_values,
                "verify_digests": self.config.verify_digests,
                "game_trace_cache_root": str(self.config.game_trace_cache_root),
                "game_trace_cache_enabled": self.config.game_trace_cache_enabled,
                "game_trace_cache_max_bytes": self.config.game_trace_cache_max_bytes,
                "game_trace_cache_max_game_bytes": self.config.game_trace_cache_max_game_bytes,
                "game_trace_cache_max_entry_bytes": self.config.game_trace_cache_max_entry_bytes,
                "game_trace_cache_min_free_bytes": self.config.game_trace_cache_min_free_bytes,
            },
            "mode": "game",
            "submission_id": context.submission.submission_id,
            "episode_id": context.replay_entry.episode_id,
            "step_index": context.stage.step_index,
            "factorized_stage": context.stage.factorized_stage,
            "addresses": addresses,
            "allow_setup_prompt_model_forward": allow_setup_prompt_model_forward,
        }
        env = os.environ.copy()
        inspector_root = Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = os.pathsep.join(
            (str(self._runtime_pythonpath_root(runtime_root)), str(inspector_root))
        )
        env[_ISOLATED_GAME_WORKER_ENV] = "1"
        # ``_inspect_exact_game`` calls us only while holding
        # ``_inference_lock``.  Clear a parent-resident model immediately
        # before this one owned child is launched so no two model copies are
        # retained concurrently.
        self._clear_resident_model()
        child: subprocess.Popen[bytes] | None = None
        try:
            child = subprocess.Popen(
                [sys.executable, str(worker)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=runtime_root,
                env=env,
                start_new_session=True,
            )
            if child.stdin is None:
                self._stop_owned_isolated_game_worker(child)
                return unavailable("isolated_game_worker_protocol_failed")
            child.stdin.write(json.dumps(request, separators=(",", ":")).encode("utf-8"))
            child.stdin.close()
            streamed = self._read_isolated_game_worker_stream(
                child,
                addresses=tuple((item[0], item[1]) for item in addresses),
            )
            return (
                streamed
                if streamed is not None
                else unavailable("isolated_game_worker_protocol_failed")
            )
        except (OSError, subprocess.SubprocessError):
            if child is not None:
                self._stop_owned_isolated_game_worker(child)
            return unavailable("isolated_game_worker_launch_failed")
        finally:
            if child is not None:
                for stream in (child.stdin, child.stdout):
                    if stream is None or stream.closed:
                        continue
                    try:
                        stream.close()
                    except OSError:
                        pass

    def _inspect_exact_game_locked(
        self,
        contexts: Sequence[ReplayContext],
        *,
        allow_setup_prompt_model_forward: bool,
    ) -> dict[tuple[int, int], dict[str, Any]] | None:
        """Use ``inspect_replay_game`` and retain the exact single-trace ABI."""

        if not contexts:
            return {}
        context = contexts[0]
        submission_provenance = context.submission.provenance
        if submission_provenance is None:
            return {
                (item.stage.step_index, item.stage.factorized_stage): self._unavailable_trace(
                    "exact_submission_provenance_unavailable"
                )
                for item in contexts
            }

        router_factory: Callable[[], Any] | None = None
        # This activation belongs only to ordinary decision forwards.  A
        # hypothetical setup (IsFirst) forward deliberately uses no submitted
        # startup activation; see the explicit grouping below.
        ordinary_runtime_activation: dict[str, Any] | None = None
        tree = submission_provenance.matchup_tree
        if tree is not None:
            if (
                not tree.available
                or tree.resolved_path is None
                or tree.expected_sha256 is None
            ):
                return {
                    (item.stage.step_index, item.stage.factorized_stage): self._unavailable_trace(
                        "matchup_tree_provenance_unavailable"
                    )
                    for item in contexts
                }
            try:
                if sha256_file(tree.resolved_path) != tree.expected_sha256:
                    return {
                        (item.stage.step_index, item.stage.factorized_stage): self._unavailable_trace(
                            "matchup_tree_sha256_mismatch"
                        )
                        for item in contexts
                    }
                from poke_bot.public_matchup_router import (
                    PublicMatchupDecisionTree,
                    RuntimePublicMatchupRouter,
                )

                validated_tree = PublicMatchupDecisionTree.from_path(
                    tree.resolved_path, require_runtime_enabled=True
                )
                if validated_tree.digest != tree.expected_sha256:
                    return {
                        (item.stage.step_index, item.stage.factorized_stage): self._unavailable_trace(
                            "matchup_tree_sha256_mismatch"
                        )
                        for item in contexts
                    }
            except (OSError, RuntimeError, TypeError, ValueError):
                return {
                    (item.stage.step_index, item.stage.factorized_stage): self._unavailable_trace(
                        "matchup_tree_runtime_contract_invalid"
                    )
                    for item in contexts
                }
            router_factory = lambda: RuntimePublicMatchupRouter(validated_tree)

        model, reason = self._load_exact_model(
            context.submission,
            context.replay_entry,
            context.replay_sha256,
        )
        if model is None:
            return {
                (item.stage.step_index, item.stage.factorized_stage): self._unavailable_trace(
                    str(reason or "exact_checkpoint_load_failed")
                )
                for item in contexts
            }
        backend = self._inference_backend()
        inspect_game = getattr(backend, "inspect_replay_game", None) if backend else None
        if not callable(inspect_game):
            return None
        try:
            receipt = submission_provenance.runtime_parity_receipt
            if receipt is None or receipt.runtime_source_tree_sha256 is None:
                raise RuntimeError("verified runtime parity receipt disappeared")
            supplied_provenance = submission_provenance.to_dict()
            supplied_provenance["reproduction_status"] = "recomputed_not_historical"
            supplied_provenance["runtime_parity_verified"] = True
            supplied_provenance["runtime_source_tree_sha256"] = (
                receipt.runtime_source_tree_sha256
            )
            supplied_provenance["runtime_identity_verified"] = True
            if tree is not None and router_factory is not None:
                ordinary_runtime_activation = {
                    "basis": "checksum_bound_submitted_startup",
                    "runtime_parity_verified": True,
                    "runtime_identity_verified": True,
                    "runtime_source_tree_sha256": receipt.runtime_source_tree_sha256,
                    "matchup_tree_verified": True,
                    "matchup_tree_sha256": tree.expected_sha256,
                    "submitted_startup_behavior_verified": True,
                    "submitted_startup_behavior": (
                        "packaged_matchup_tree_enables_policy_agent_runtime"
                    ),
                }
            def inspect_context_group(
                group: Sequence[ReplayContext],
                *,
                activation: Mapping[str, Any] | None,
                setup_forward: bool,
            ) -> Any:
                return inspect_game(
                    model=model,
                    replay=context.replay,
                    acting_seat=context.seat,
                    addresses=[
                        (item.stage.step_index, item.stage.factorized_stage)
                        for item in group
                    ],
                    own_deck=None,
                    router=None,
                    router_factory=router_factory,
                    checkpoint_digest=submission_provenance.checkpoint.expected_sha256,
                    checkpoint_path=submission_provenance.checkpoint.resolved_path,
                    provenance=supplied_provenance,
                    submitted_runtime_activation=activation,
                    allow_setup_prompt_model_forward=setup_forward,
                )

            # ``_trace_payload_uncached`` invokes ordinary decisions through
            # the submitted runtime activation even when its outer response
            # also asks for a hypothetical IsFirst model view.  The IsFirst
            # call itself is forced-bypass.  Preserve that per-address ABI
            # explicitly instead of letting a game-wide boolean alter either
            # activation audit.  This also covers a setup-only short game,
            # where inference's mixed-row convenience split has no ordinary
            # row from which to infer the forced-bypass phase.
            setup_contexts = [
                item for item in contexts if not item.stage.model_forward_expected
            ]
            ordinary_contexts = [
                item for item in contexts if item.stage.model_forward_expected
            ]
            if allow_setup_prompt_model_forward and setup_contexts:
                raw_results = inspect_context_group(
                    setup_contexts,
                    activation=None,
                    setup_forward=True,
                )
                if ordinary_contexts:
                    ordinary_results = inspect_context_group(
                        ordinary_contexts,
                        activation=ordinary_runtime_activation,
                        setup_forward=False,
                    )
                    if isinstance(raw_results, Mapping) and isinstance(
                        ordinary_results, Mapping
                    ):
                        raw_results = {**raw_results, **ordinary_results}
                    else:
                        # A partial/malformed phase must never be treated as
                        # a complete game response.  Returning a non-mapping
                        # here reaches the existing fail-closed envelope
                        # below rather than silently discarding setup rows.
                        raw_results = None
            else:
                raw_results = inspect_context_group(
                    contexts,
                    activation=ordinary_runtime_activation,
                    setup_forward=False,
                )
        except Exception:  # noqa: BLE001 - exact game reconstruction fails closed
            return {
                (item.stage.step_index, item.stage.factorized_stage): self._unavailable_trace(
                    "exact_game_trace_reconstruction_failed"
                )
                for item in contexts
            }
        if not isinstance(raw_results, Mapping):
            return {
                (item.stage.step_index, item.stage.factorized_stage): self._unavailable_trace(
                    "exact_game_trace_invalid_response"
                )
                for item in contexts
            }
        output: dict[tuple[int, int], dict[str, Any]] = {}
        for item in contexts:
            address = (item.stage.step_index, item.stage.factorized_stage)
            raw = raw_results.get(address)
            if raw is None:
                output[address] = self._unavailable_trace(
                    "exact_game_trace_missing_address"
                )
            else:
                output[address] = self._normalise_exact_backend_trace(
                    item, raw, supplied_provenance
                )
        return output

    def _inspect_exact_trace(
        self,
        context: ReplayContext,
        *,
        head_scales: Mapping[str, float] | None = None,
        allow_setup_prompt_model_forward: bool = False,
    ) -> dict[str, Any]:
        """Serialize checkpoint load plus model forward as one CPU residency span."""

        with self._inference_lock:
            runtime_root = self._runtime_source_root_for_submission(context.submission)
            if runtime_root is None:
                return self._unavailable_trace("runtime_source_tree_sha256_mismatch")
            if not self._imported_runtime_is(runtime_root):
                return self._inspect_exact_trace_isolated(
                    context,
                    runtime_root=runtime_root,
                    head_scales=head_scales,
                    allow_setup_prompt_model_forward=allow_setup_prompt_model_forward,
                )
            return self._inspect_exact_trace_locked(
                context,
                head_scales=head_scales,
                allow_setup_prompt_model_forward=allow_setup_prompt_model_forward,
            )

    def _inspect_exact_trace_locked(
        self,
        context: ReplayContext,
        *,
        head_scales: Mapping[str, float] | None = None,
        allow_setup_prompt_model_forward: bool = False,
    ) -> dict[str, Any]:
        """Run the lazy exact backend or return a fail-closed model envelope."""

        def unavailable(reason: str, warning: str) -> dict[str, Any]:
            return {
                "model": {
                    "availability": _availability(False, reason),
                    "status": "unavailable",
                },
                "heads": [],
                "fusion": None,
                "provenance": None,
                "reproduction_status": "unavailable",
                "warnings": [warning],
            }

        submission_provenance = context.submission.provenance
        if submission_provenance is None:
            return unavailable(
                "exact_submission_provenance_unavailable",
                "Exact submission provenance disappeared before trace reconstruction.",
            )

        # The artifact was verified when the catalog loaded, but the exact tree
        # is serving authority for request-local adapter activation.  Rehash and
        # parse the runtime-enabled schema again under the inference lock, then
        # retain the validated immutable in-memory tree for the fresh router.
        router_factory: Callable[[], Any] | None = None
        submitted_runtime_activation: dict[str, Any] | None = None
        tree = submission_provenance.matchup_tree
        if tree is not None:
            if (
                not tree.available
                or tree.resolved_path is None
                or tree.expected_sha256 is None
            ):
                return unavailable(
                    "matchup_tree_provenance_unavailable",
                    "The submitted Matchup Adapter tree is not checksum-verifiable; adapter runtime was not enabled.",
                )
            try:
                if sha256_file(tree.resolved_path) != tree.expected_sha256:
                    return unavailable(
                        "matchup_tree_sha256_mismatch",
                        "The submitted Matchup Adapter tree changed after catalog verification; adapter runtime was not enabled.",
                    )
                from poke_bot.public_matchup_router import (
                    PublicMatchupDecisionTree,
                    RuntimePublicMatchupRouter,
                )

                validated_tree = PublicMatchupDecisionTree.from_path(
                    tree.resolved_path, require_runtime_enabled=True
                )
                if validated_tree.digest != tree.expected_sha256:
                    return unavailable(
                        "matchup_tree_sha256_mismatch",
                        "The parsed Matchup Adapter tree did not match its submitted checksum; adapter runtime was not enabled.",
                    )
            except (OSError, RuntimeError, TypeError, ValueError):
                return unavailable(
                    "matchup_tree_runtime_contract_invalid",
                    "The checksum-bound Matchup Adapter tree failed its runtime-enabled schema validation; adapter runtime was not enabled.",
                )
            router_factory = lambda: RuntimePublicMatchupRouter(validated_tree)

        # Repeat the replay-byte gate immediately before the only path that
        # can call ``torch.load``.  The trace route checks it earlier for a
        # useful response, but this inner guard keeps future call paths from
        # loading a model for substituted cache bytes.
        model, reason = self._load_exact_model(
            context.submission,
            context.replay_entry,
            context.replay_sha256,
        )
        if model is None:
            return unavailable(
                str(reason or "exact_checkpoint_load_failed"),
                "The checksum-bound submitted model could not be loaded; no alternate model was used.",
            )
        backend = self._inference_backend()
        inspect_step = (
            getattr(backend, "inspect_replay_step", None) if backend else None
        )
        if not callable(inspect_step):
            return {
                "model": {
                    "availability": _availability(
                        False, "inference_trace_contract_unavailable"
                    ),
                    "status": "unavailable",
                },
                "heads": [],
                "fusion": None,
                "provenance": None,
                "reproduction_status": "unavailable",
                "warnings": [
                    "The exact model loaded, but the replay-trace backend is unavailable."
                ],
            }
        try:
            # Kept keyword-only so a backend contract change fails closed rather
            # than accidentally swapping seat, step, or factorized stage.
            supplied_provenance = submission_provenance.to_dict()
            # A trace reached this point only after _trace_analysis_reason
            # rehashed the configured extracted submitted runtime and matched
            # it to this receipt.  Raw Kaggle replays still do not retain
            # historical activations/logits, so the result remains a causal
            # re-evaluation rather than a claim of historical capture.
            receipt = submission_provenance.runtime_parity_receipt
            if receipt is None or receipt.runtime_source_tree_sha256 is None:
                raise RuntimeError("verified runtime parity receipt disappeared")
            supplied_provenance["reproduction_status"] = "recomputed_not_historical"
            supplied_provenance["runtime_parity_verified"] = True
            supplied_provenance["runtime_source_tree_sha256"] = (
                receipt.runtime_source_tree_sha256
            )
            supplied_provenance["runtime_identity_verified"] = True
            if (
                tree is not None
                and router_factory is not None
                and not allow_setup_prompt_model_forward
            ):
                submitted_runtime_activation = {
                    "basis": "checksum_bound_submitted_startup",
                    "runtime_parity_verified": True,
                    "runtime_identity_verified": True,
                    "runtime_source_tree_sha256": receipt.runtime_source_tree_sha256,
                    "matchup_tree_verified": True,
                    "matchup_tree_sha256": tree.expected_sha256,
                    "submitted_startup_behavior_verified": True,
                    "submitted_startup_behavior": (
                        "packaged_matchup_tree_enables_policy_agent_runtime"
                    ),
                }
            result = inspect_step(
                model=model,
                replay=context.replay,
                acting_seat=context.seat,
                env_step=context.stage.step_index,
                factorized_stage=context.stage.factorized_stage,
                own_deck=None,
                router=None,
                router_factory=router_factory,
                checkpoint_digest=submission_provenance.checkpoint.expected_sha256,
                checkpoint_path=submission_provenance.checkpoint.resolved_path,
                provenance=supplied_provenance,
                submitted_runtime_activation=submitted_runtime_activation,
                head_scales=head_scales,
                allow_setup_prompt_model_forward=allow_setup_prompt_model_forward,
            )
        except Exception:  # noqa: BLE001 - exact trace must fail closed
            return {
                "model": {
                    "availability": _availability(
                        False, "exact_trace_reconstruction_failed"
                    ),
                    "status": "unavailable",
                },
                "heads": [],
                "fusion": None,
                "provenance": None,
                "reproduction_status": "unavailable",
                "warnings": [
                    "Exact model trace reconstruction failed; no stale or alternate trace is shown."
                ],
            }
        return self._normalise_exact_backend_trace(
            context, result, supplied_provenance
        )

    def _normalise_exact_backend_trace(
        self,
        context: ReplayContext,
        result: Any,
        supplied_provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Translate one raw backend trace into the established HTTP envelope.

        The game batching backend deliberately returns the same raw schema as
        ``inspect_replay_step``.  Keeping the final normalization here means
        cached game rows retain all current legal-option, fusion, adapter, and
        provenance semantics rather than a reduced fast-path representation.
        """

        payload = _json_safe(result)
        if not isinstance(payload, Mapping):
            return self._unavailable_trace("exact_trace_invalid_response")
        availability = payload.get("availability")
        if not isinstance(availability, Mapping):
            availability = _availability(False, "exact_trace_invalid_response")
        backend_provenance = payload.get("provenance")
        if not isinstance(backend_provenance, Mapping):
            backend_provenance = supplied_provenance
        reproduction_status = str(
            backend_provenance.get("reproduction_status")
            or (
                "recomputed_not_historical"
                if availability.get("available")
                else "unavailable"
            )
        )
        if availability.get("available") is False:
            reason = _reason_from(
                [
                    str(availability.get("reason") or ""),
                    str(payload.get("reason") or ""),
                    str(payload.get("unavailable") or ""),
                ],
                "exact_trace_reconstruction_unavailable",
            )
            return {
                "model": {
                    "availability": _availability(False, reason),
                    "status": "unavailable",
                    "selected_index": None,
                    "selected_action": None,
                    "agreement": None,
                    "value": None,
                    "route": None,
                    "fusion": payload.get("fusion"),
                },
                "heads": _normalise_heads(payload.get("heads"), payload.get("fusion")),
                "fusion": payload.get("fusion"),
                "provenance": backend_provenance,
                "reproduction_status": "unavailable",
                "warnings": [
                    "Exact model trace reconstruction is unavailable; no alternate model trace is shown."
                ],
            }

        raw_replay = payload.get("replay")
        raw_replay = raw_replay if isinstance(raw_replay, Mapping) else {}
        policy = payload.get("policy")
        policy = policy if isinstance(policy, Mapping) else {}
        value = payload.get("value")
        value = value if isinstance(value, Mapping) else {}
        adapter = payload.get("adapter")
        adapter = adapter if isinstance(adapter, Mapping) else {}
        fusion = payload.get("fusion")
        fusion_mapping = fusion if isinstance(fusion, Mapping) else None
        selected_index = policy.get("model_choice_index")
        selected_index = selected_index if isinstance(selected_index, int) else None
        selected_action = policy.get("model_choice_candidate")
        if selected_action is None and selected_index is not None:
            selected_action = _indexed(raw_replay.get("legal_candidates"), selected_index)
        legal_candidates = raw_replay.get("legal_candidates")
        if not isinstance(legal_candidates, list):
            legal_candidates = [list(candidate) for candidate in context.stage.candidates]
        legal_options: list[dict[str, Any]] = []
        for index, candidate in enumerate(legal_candidates):
            legal_options.append(
                {
                    "index": index,
                    "action": candidate,
                    "base_logit": _indexed(policy.get("base_logits"), index),
                    "final_logit": _indexed(policy.get("final_logits"), index),
                    "probability": _indexed(policy.get("probabilities"), index),
                    "route_contributions": (
                        _route_contributions_for_option(fusion_mapping, index)
                        if fusion_mapping is not None
                        else None
                    ),
                    "is_recorded": index == context.stage.target_index,
                    "is_model_choice": index == selected_index,
                }
            )
        model_payload = {
            "availability": dict(availability),
            "status": reproduction_status,
            "selected_index": selected_index,
            "selected_action": selected_action,
            "agreement": policy.get("model_matches_recorded_target"),
            "value": _simple_value(value.get("policy_value")),
            "value_detail": value,
            "route": adapter.get("route"),
            "adapter": adapter,
            "fusion": fusion,
            "policy": policy,
            "latent_lookahead": payload.get("latent_lookahead"),
        }
        raw_warnings = payload.get("warnings", [])
        return {
            "model": model_payload,
            "heads": _normalise_heads(payload.get("heads"), fusion_mapping),
            "fusion": fusion,
            "decision_influence": payload.get("decision_influence"),
            "guide_shadow": payload.get("guide_shadow"),
            "legal_options": legal_options,
            "provenance": backend_provenance,
            "reproduction_status": reproduction_status,
            "warnings": [
                "Dynamic model values are checksum-bound causal re-evaluation evidence, not fields stored in the raw replay.",
                *[
                    str(item)
                    for item in raw_warnings
                    if isinstance(item, (str, int, float))
                ],
            ],
        }

    def _parameter_inspector(
        self, submission: SubmissionCatalogEntry
    ) -> tuple[Any | None, str | None]:
        model, reason = self._load_exact_model(submission)
        if model is None:
            return None, reason
        try:
            parameters = importlib.import_module("replay_inspector.parameters")
            inspector_type = parameters.ParameterInspector
            return inspector_type(model), None
        except Exception:  # noqa: BLE001 - optional parameter runtime boundary
            return None, "parameter_runtime_unavailable"

    def static_asset(self, path: str) -> tuple[bytes, str]:
        try:
            filename, content_type = _STATIC_ASSETS[path]
        except KeyError as exc:
            raise InspectorHTTPError(
                HTTPStatus.NOT_FOUND,
                "route_not_found",
                "resource was not found",
            ) from exc
        try:
            asset = resolve_contained_path(
                filename,
                roots=(self._web_root,),
                relative_to=self._web_root,
                require_exists=True,
            )
            if not asset.is_file():
                raise FileNotFoundError(filename)
            return asset.read_bytes(), content_type
        except (FileNotFoundError, OSError, ProvenanceError) as exc:
            raise InspectorHTTPError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "static_asset_unavailable",
                "required inspector static asset is unavailable",
            ) from exc


class _IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def _handler_for(application: InspectorApplication) -> type[BaseHTTPRequestHandler]:
    """Create an isolated handler class bound to one application snapshot."""

    class InspectorRequestHandler(BaseHTTPRequestHandler):
        server_version = "ReplayModelInspector"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *_args: object) -> None:
            # BaseHTTPRequestHandler logs raw URL paths (and query strings) to
            # stderr.  The service is local and has structured endpoint errors;
            # keep journald free of arbitrary request text.
            return

        def end_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header(
                "Permissions-Policy",
                "camera=(), geolocation=(), microphone=(), payment=(), usb=()",
            )
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; base-uri 'none'; form-action 'none'; "
                "frame-ancestors 'none'; object-src 'none'; connect-src 'self'; "
                "img-src 'self' data:; script-src 'self'; style-src 'self'",
            )
            super().end_headers()

        def _write_json(
            self,
            status: int | HTTPStatus,
            payload: Mapping[str, Any],
            *,
            extra_headers: Mapping[str, str] | None = None,
        ) -> None:
            encoded = json.dumps(
                _json_safe(payload),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            for name, value in (extra_headers or {}).items():
                self.send_header(str(name), str(value))
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _write_error(self, error: InspectorHTTPError) -> None:
            body: dict[str, Any] = {
                "schema": API_SCHEMA,
                "error": {"code": error.code, "message": error.message},
            }
            if error.details:
                body["error"]["details"] = _json_safe(error.details)
            self._write_json(error.status, body)

        def _method_not_allowed(self) -> None:
            self._write_json(
                HTTPStatus.METHOD_NOT_ALLOWED,
                {
                    "schema": API_SCHEMA,
                    "error": {
                        "code": "method_not_allowed",
                        "message": "Replay Model Inspector exposes GET-only read-only routes",
                    },
                },
                extra_headers={"Allow": "GET"},
            )

        def _assert_local_browser_request(self) -> None:
            """Avoid cross-site loopback CPU work from a hostile web page.

            The listener is already loopback-only and the API sends no CORS
            grant.  These request checks add a defense against blind browser
            GETs that could otherwise trigger expensive CPU inference even
            though their caller could not read the response.
            """

            host = self.headers.get("Host", "")
            if not _host_header_is_loopback(host):
                raise InspectorHTTPError(
                    HTTPStatus.FORBIDDEN,
                    "non_loopback_host_rejected",
                    "request Host must be a loopback authority",
                )
            origin = self.headers.get("Origin")
            if origin is not None and not _origin_is_loopback(origin):
                raise InspectorHTTPError(
                    HTTPStatus.FORBIDDEN,
                    "cross_origin_request_rejected",
                    "cross-origin requests are not accepted by this localhost service",
                )
            fetch_site = self.headers.get("Sec-Fetch-Site", "").strip().casefold()
            if fetch_site == "cross-site":
                raise InspectorHTTPError(
                    HTTPStatus.FORBIDDEN,
                    "cross_site_request_rejected",
                    "cross-site requests are not accepted by this localhost service",
                )

        def do_GET(self) -> None:
            try:
                self._assert_local_browser_request()
                parsed = urlsplit(self.path)
                path = parsed.path
                if path in _STATIC_ASSETS:
                    if parsed.query:
                        raise InspectorHTTPError(
                            HTTPStatus.BAD_REQUEST,
                            "static_query_not_allowed",
                            "static assets do not accept query parameters",
                        )
                    body, content_type = application.static_asset(path)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Cache-Control", "no-store, max-age=0")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                payload = self._api_payload(path, parsed.query)
                self._write_json(HTTPStatus.OK, payload)
            except InspectorHTTPError as exc:
                self._write_error(exc)
            except (TypeError, ValueError):
                # Keep malformed JSON/query/identifier responses deterministic
                # and never include a Python traceback or source path.
                self._write_error(
                    InspectorHTTPError(
                        HTTPStatus.BAD_REQUEST,
                        "invalid_request",
                        "request could not be parsed",
                    )
                )
            except Exception as exc:  # noqa: BLE001  # pragma: no cover - HTTP boundary
                _LOG.error(
                    "replay inspector internal request failure: %s", type(exc).__name__
                )
                self._write_error(
                    InspectorHTTPError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "internal_error",
                        "request could not be completed",
                    )
                )

        def _api_payload(self, path: str, raw_query: str) -> dict[str, Any]:
            if path in {"/healthz", "/api/health"}:
                _query_values(raw_query, allowed=set())
                return application.health_payload()
            if path == "/api/submissions":
                _query_values(raw_query, allowed=set())
                return application.submissions_payload()
            if not path.startswith("/api/submissions/"):
                raise InspectorHTTPError(
                    HTTPStatus.NOT_FOUND,
                    "route_not_found",
                    "resource was not found",
                )
            parts = path.split("/")
            # Leading slash gives ["", "api", "submissions", ...].  Do not
            # unquote identifiers; encoded separators must not create routes.
            if len(parts) < 4:
                raise InspectorHTTPError(
                    HTTPStatus.NOT_FOUND,
                    "route_not_found",
                    "resource was not found",
                )
            submission_id = _strict_int(parts[3], field="submission_id", minimum=1)
            if len(parts) == 5 and parts[4] == "games":
                _query_values(raw_query, allowed=set())
                return application.games_payload(submission_id)
            if len(parts) == 5 and parts[4] == "parameters":
                _query_values(raw_query, allowed=set())
                return application.parameters_payload(submission_id)
            if len(parts) == 5 and parts[4] == "training-recipe":
                _query_values(raw_query, allowed=set())
                return application.training_recipe_payload(submission_id)
            if len(parts) == 7 and parts[4] == "games" and parts[6] == "steps":
                _query_values(raw_query, allowed=set())
                episode_id = _strict_int(parts[5], field="episode_id", minimum=0)
                return application.steps_payload(submission_id, episode_id)
            if len(parts) == 8 and parts[4] == "games" and parts[6] == "steps":
                query = _query_values(raw_query, allowed={"stage", "scales"})
                episode_id = _strict_int(parts[5], field="episode_id", minimum=0)
                step_index = _strict_int(parts[7], field="step_index", minimum=0)
                stage = _strict_int(query.get("stage", "0"), field="stage", minimum=0)
                return application.trace_payload(
                    submission_id,
                    episode_id,
                    step_index,
                    stage,
                    head_scales=_head_scales_query(query.get("scales")),
                )
            if len(parts) == 6 and parts[4] == "parameters":
                query = _query_values(raw_query, allowed={"offset", "limit", "bins"})
                try:
                    name = unquote(parts[5], encoding="utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    raise InspectorHTTPError(
                        HTTPStatus.BAD_REQUEST,
                        "invalid_tensor_name",
                        "tensor name must be valid UTF-8",
                    ) from exc
                if not name or any(
                    character in name for character in ("/", "\\", "\x00")
                ):
                    raise InspectorHTTPError(
                        HTTPStatus.BAD_REQUEST,
                        "invalid_tensor_name",
                        "tensor name contains an unsupported character",
                    )
                offset = _strict_int(
                    query.get("offset", "0"), field="offset", minimum=0
                )
                requested_limit = _strict_int(
                    query.get(
                        "limit", str(min(256, application.config.max_parameter_slice))
                    ),
                    field="limit",
                    minimum=1,
                )
                configured_limit = min(application.config.max_parameter_slice, 4096)
                if requested_limit > configured_limit:
                    raise InspectorHTTPError(
                        HTTPStatus.BAD_REQUEST,
                        "parameter_slice_limit_exceeded",
                        "tensor slice limit exceeds the configured maximum",
                        details={"maximum": configured_limit},
                    )
                bins = _strict_int(query.get("bins", "64"), field="bins", minimum=1)
                if bins > 256:
                    raise InspectorHTTPError(
                        HTTPStatus.BAD_REQUEST,
                        "histogram_bins_exceeded",
                        "histogram bin count exceeds the supported maximum",
                        details={"maximum": 256},
                    )
                return application.parameter_detail_payload(
                    submission_id,
                    name,
                    offset=offset,
                    limit=requested_limit,
                    bins=bins,
                )
            raise InspectorHTTPError(
                HTTPStatus.NOT_FOUND,
                "route_not_found",
                "resource was not found",
            )

        def do_HEAD(self) -> None:
            self._method_not_allowed()

        def do_POST(self) -> None:
            self._method_not_allowed()

        def do_PUT(self) -> None:
            self._method_not_allowed()

        def do_PATCH(self) -> None:
            self._method_not_allowed()

        def do_DELETE(self) -> None:
            self._method_not_allowed()

        def do_OPTIONS(self) -> None:
            self._method_not_allowed()

    return InspectorRequestHandler


def create_server(
    config: InspectorConfig,
    *,
    application: InspectorApplication | None = None,
    host: str | None = None,
    port: int | None = None,
) -> ThreadingHTTPServer:
    """Construct, but do not start, a loopback-only threaded local server."""

    effective_host = _require_loopback_host(host or config.bind_host)
    effective_port = int(config.port if port is None else port)
    # ``0`` is accepted only by this in-process constructor so tests can use
    # an OS-assigned loopback port.  The public launcher/main path still
    # requires an explicit 1..65535 service port.
    if not 0 <= effective_port <= 65535:
        raise ValueError("port must be in 0..65535")
    app = application or InspectorApplication(config)
    server_type: type[ThreadingHTTPServer]
    try:
        parsed_host = ipaddress.ip_address(effective_host)
    except ValueError:
        parsed_host = None
    server_type = (
        _IPv6ThreadingHTTPServer
        if parsed_host and parsed_host.version == 6
        else ThreadingHTTPServer
    )
    server = server_type((effective_host, effective_port), _handler_for(app))
    server.daemon_threads = True
    # Useful to embedding tests and harmless to BaseHTTPRequestHandler.
    server.inspector_application = app  # type: ignore[attr-defined]
    return server


def main(
    config_path: Path | str | None = None,
    host: str | None = None,
    port: int | None = None,
    check: bool = False,
) -> int:
    """Run the standalone server or validate its local configuration.

    ``scripts/start_replay_model_inspector.py`` owns argument parsing and calls
    this exact signature so systemd can use the same controlled path as a
    manual localhost launch.
    """

    path = Path(config_path) if config_path is not None else None
    config = InspectorConfig.load(path)
    effective_host = _require_loopback_host(host or config.bind_host)
    effective_port = int(config.port if port is None else port)
    if not 1 <= effective_port <= 65535:
        raise ValueError("port must be in 1..65535")
    if effective_host != config.bind_host or effective_port != config.port:
        config = replace(config, bind_host=effective_host, port=effective_port)
    application = InspectorApplication(config)
    if check:
        print(json.dumps(application.check_payload(), sort_keys=True))
        return 0
    server = create_server(config, application=application)
    _LOG.info(
        "Replay Model Inspector listening on %s:%s", effective_host, effective_port
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:  # pragma: no cover - interactive server lifecycle
        pass
    finally:
        server.server_close()
    return 0


__all__ = [
    "API_SCHEMA",
    "InspectorApplication",
    "InspectorHTTPError",
    "ReplayContext",
    "create_server",
    "main",
]
