"""Private, integrity-checked disk cache for reconstructed replay traces.

The replay inspector's expensive work is deterministic for a particular
physical replay *and* a particular verified runtime.  This module keeps the
cache boundary deliberately small: callers must supply both identities, and
the cache stores JSON values only.  It does not know about HTTP, models, or
the replay archive, so it can be used by the inspector without giving the
cache authority over any source of truth.

Typical use::

    identity = GameTraceIdentity(
        submission_id=77001,
        episode_id=88001,
        replay_sha256="sha256:...",
        provenance={
            "checkpoint_sha256": "sha256:...",
            "runtime_package_sha256": "sha256:...",
        },
    )
    cache = GameTraceCache(Path("/tmp/poke-replay-inspector-cache"))
    payload = cache.get_or_compute(
        identity,
        TraceAddress(step_index=12, stage=1),
        reconstruct_trace,
    )

Entries are gzip-compressed JSON envelopes with a payload checksum and an
envelope checksum.  A damaged, insecure, stale, or mismatched entry is always
treated as a cache miss.  The cache is therefore an optimization only: it
never supplies data that was not bound to the requested immutable identity.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar, cast

_CACHE_SCHEMA = "poke_bot.replay_model_inspector_game_trace_cache/v1"
_MANIFEST_SCHEMA = "poke_bot.replay_model_inspector_game_trace_manifest/v1"
_SHA256_RE = re.compile(r"sha256:([0-9a-fA-F]{64})\Z")
_PROVENANCE_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_GAME_DIRECTORY_RE = re.compile(r"[0-9a-f]{64}\Z")
_ENTRY_FILENAME_RE = re.compile(r"s[0-9]+-f[0-9]+\.json\.gz\Z")

_DEFAULT_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_ENTRIES = 4_096
_DEFAULT_MAX_BYTES_PER_GAME = 64 * 1024 * 1024
# Selectable games can legitimately contain more than 512 step/stage traces.
# Byte limits are the production bound; a count cap is opt-in for callers that
# need one, never an invisible completeness limit for a physical game.
_DEFAULT_MAX_ENTRIES_PER_GAME: int | None = None
_DEFAULT_MAX_ENTRY_BYTES = 4 * 1024 * 1024

_JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | list["_JsonValue"]
    | dict[str, "_JsonValue"]
)
_T = TypeVar("_T", bound=_JsonValue)


class UnsafeCacheRootError(ValueError):
    """Raised when a cache root is not a private, owned directory."""


@dataclass(frozen=True)
class GameTraceIdentity:
    """Immutable identity for all inputs that can affect a game trace.

    ``replay_sha256`` binds the cache entry to the physical replay.  The
    non-empty ``provenance`` mapping must bind the relevant verified runtime
    inputs (for example checkpoint, submitted runtime package, source tree,
    and runtime-parity receipt digests).  Requiring it prevents an otherwise
    valid replay entry from being reused after a build changes.
    """

    submission_id: int
    episode_id: int
    replay_sha256: str
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.submission_id, bool) or not isinstance(
            self.submission_id, int
        ):
            raise ValueError("submission_id must be an integer")
        if self.submission_id < 0:
            raise ValueError("submission_id must be non-negative")
        if isinstance(self.episode_id, bool) or not isinstance(self.episode_id, int):
            raise ValueError("episode_id must be an integer")
        if self.episode_id < 0:
            raise ValueError("episode_id must be non-negative")

        object.__setattr__(self, "replay_sha256", _normalise_sha256(self.replay_sha256))
        if not isinstance(self.provenance, Mapping) or not self.provenance:
            raise ValueError(
                "provenance must contain immutable build/runtime identity values"
            )

        normalised: dict[str, str] = {}
        for name, value in self.provenance.items():
            if not isinstance(name, str) or not _PROVENANCE_NAME_RE.fullmatch(name):
                raise ValueError(
                    "provenance names must be lower_snake_case ASCII identifiers"
                )
            if not isinstance(value, str) or not value or len(value) > 512:
                raise ValueError("provenance values must be non-empty strings up to 512 bytes")
            if "\x00" in value:
                raise ValueError("provenance values cannot contain NUL bytes")
            normalised[name] = value
        object.__setattr__(
            self, "provenance", MappingProxyType(dict(sorted(normalised.items())))
        )

    def as_dict(self) -> dict[str, _JsonValue]:
        """Return the canonical, JSON-safe identity representation."""

        return {
            "submission_id": self.submission_id,
            "episode_id": self.episode_id,
            "replay_sha256": self.replay_sha256,
            "provenance": dict(self.provenance),
        }

    @property
    def cache_key(self) -> str:
        """A path-safe SHA-256 key over every identity field."""

        return hashlib.sha256(_canonical_json(self.as_dict())).hexdigest()

    @property
    def fingerprint(self) -> str:
        """The public checksum form of :attr:`cache_key`."""

        return "sha256:" + self.cache_key


# ``GameTraceKey`` reads naturally at integration call sites.  Keep the more
# descriptive class name as the canonical implementation and export both.
GameTraceKey = GameTraceIdentity


@dataclass(frozen=True, init=False)
class TraceAddress:
    """One selectable reconstruction address inside a game.

    ``factorized_stage`` is accepted as a keyword alias for the existing
    inspector terminology.  The serialized form uses the shorter ``stage``
    name and is unambiguous because both components are non-negative ints.
    """

    step_index: int
    stage: int

    def __init__(
        self,
        step_index: int,
        stage: int | None = None,
        *,
        factorized_stage: int | None = None,
    ) -> None:
        if stage is None:
            stage = factorized_stage
        elif factorized_stage is not None and stage != factorized_stage:
            raise ValueError("stage and factorized_stage disagree")
        if stage is None:
            raise TypeError("stage is required")
        _validate_non_negative_int("step_index", step_index)
        _validate_non_negative_int("stage", stage)
        object.__setattr__(self, "step_index", step_index)
        object.__setattr__(self, "stage", stage)

    @property
    def factorized_stage(self) -> int:
        """Compatibility spelling for inspector call sites."""

        return self.stage

    def as_dict(self) -> dict[str, int]:
        return {"step_index": self.step_index, "stage": self.stage}

    @property
    def filename(self) -> str:
        return f"s{self.step_index}-f{self.stage}.json.gz"


GameTraceAddress = TraceAddress


@dataclass(frozen=True)
class CacheRead:
    """Result of a cache read; ``hit`` distinguishes a cached JSON null."""

    hit: bool
    value: _JsonValue | None = None


@dataclass(frozen=True)
class CacheMetrics:
    """Small operational counters suitable for a localhost health payload."""

    entry_writes: int
    entry_write_failures: int
    manifest_writes: int
    manifest_write_failures: int
    low_space_rejections: int
    evictions: int


@dataclass
class _MutableMetrics:
    entry_writes: int = 0
    entry_write_failures: int = 0
    manifest_writes: int = 0
    manifest_write_failures: int = 0
    low_space_rejections: int = 0
    evictions: int = 0


@dataclass(frozen=True)
class GameTraceManifest:
    """Verified commit record for the trace addresses retained for one game."""

    identity_fingerprint: str
    addresses: tuple[TraceAddress, ...]
    trace_sha256: Mapping[TraceAddress, str]
    completed_at_ns: int


@dataclass(frozen=True)
class _Manifest:
    addresses: tuple[TraceAddress, ...]
    trace_sha256: Mapping[TraceAddress, str]
    completed_at_ns: int


@dataclass
class _Flight:
    event: threading.Event = field(default_factory=threading.Event)
    value: _JsonValue | object = field(default_factory=object)
    error: BaseException | None = None


@dataclass
class _GameFlight:
    event: threading.Event = field(default_factory=threading.Event)
    values: dict[TraceAddress, _JsonValue] = field(default_factory=dict)
    error: BaseException | None = None


@dataclass(frozen=True)
class _GameUsage:
    path: Path
    game_key: str
    size: int
    entry_count: int
    access_ns: int


class GameTraceCache:
    """A bounded, private, process-local-singleflight game trace cache.

    The cache root must resolve beneath ``/tmp`` in production.  Tests may
    inject another temporary directory only by explicitly setting
    ``unsafe_test_root=True``.  This deliberate escape hatch keeps production
    callers from accidentally caching replay/model outputs in a durable or
    shared location.

    ``max_*`` byte limits apply to compressed on-disk entries; an entry's
    decoded JSON envelope is separately capped by ``max_entry_bytes`` to avoid
    gzip expansion attacks.  A limit of ``None`` disables that particular
    bound, while zero entry/byte *collection* limits make the cache retain no
    entries after a write.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        unsafe_test_root: bool = False,
        max_total_bytes: int | None = _DEFAULT_MAX_TOTAL_BYTES,
        max_entries: int | None = _DEFAULT_MAX_ENTRIES,
        max_bytes_per_game: int | None = _DEFAULT_MAX_BYTES_PER_GAME,
        max_entries_per_game: int | None = _DEFAULT_MAX_ENTRIES_PER_GAME,
        max_entry_bytes: int = _DEFAULT_MAX_ENTRY_BYTES,
        min_free_bytes: int = 0,
        # Configuration-facing spellings.  They intentionally remain aliases
        # so standalone callers can use the more explicit ``max_total_bytes``
        # and ``max_bytes_per_game`` names above.
        max_bytes: int | None = None,
        max_game_bytes: int | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if max_bytes is not None:
            if (
                max_total_bytes != _DEFAULT_MAX_TOTAL_BYTES
                and max_total_bytes != max_bytes
            ):
                raise ValueError("max_bytes and max_total_bytes disagree")
            max_total_bytes = max_bytes
        if max_game_bytes is not None:
            if (
                max_bytes_per_game != _DEFAULT_MAX_BYTES_PER_GAME
                and max_bytes_per_game != max_game_bytes
            ):
                raise ValueError("max_game_bytes and max_bytes_per_game disagree")
            max_bytes_per_game = max_game_bytes
        self.max_total_bytes = _validate_limit("max_total_bytes", max_total_bytes)
        self.max_entries = _validate_limit("max_entries", max_entries)
        self.max_bytes_per_game = _validate_limit(
            "max_bytes_per_game", max_bytes_per_game
        )
        self.max_entries_per_game = _validate_limit(
            "max_entries_per_game", max_entries_per_game
        )
        if isinstance(max_entry_bytes, bool) or not isinstance(max_entry_bytes, int):
            raise ValueError("max_entry_bytes must be a positive integer")
        if max_entry_bytes <= 0:
            raise ValueError("max_entry_bytes must be a positive integer")
        self.max_entry_bytes = max_entry_bytes
        if isinstance(min_free_bytes, bool) or not isinstance(min_free_bytes, int):
            raise ValueError("min_free_bytes must be a non-negative integer")
        self.min_free_bytes = _validate_limit("min_free_bytes", min_free_bytes)
        self._clock_ns = clock_ns
        self._io_lock = threading.RLock()
        self._flight_lock = threading.Lock()
        self._flights: dict[tuple[str, str], _Flight] = {}
        self._game_flights: dict[str, _GameFlight] = {}
        self._metrics = _MutableMetrics()

        self.root = _prepare_root(Path(root), unsafe_test_root=unsafe_test_root)
        self._ensure_private_directory_tree(self.root)
        self._games_root = self.root / "games"
        self._ensure_private_directory(self._games_root)

    def entry_path(self, identity: GameTraceIdentity, address: TraceAddress) -> Path:
        """Return the deterministic path for an entry without creating it."""

        return self._games_root / identity.cache_key / "steps" / address.filename

    def manifest_path(self, identity: GameTraceIdentity) -> Path:
        """Return the deterministic atomic commit record path for one game."""

        return self._games_root / identity.cache_key / "manifest.json.gz"

    @property
    def metrics(self) -> CacheMetrics:
        """Return counters without exposing mutable cache state."""

        with self._io_lock:
            return CacheMetrics(**self._metrics.__dict__)

    @classmethod
    def from_inspector_config(cls, config: object) -> "GameTraceCache":
        """Construct from the inspector's bounded temporary-cache settings.

        Kept duck-typed to avoid making this generic module import the server
        configuration package.
        """

        return cls(
            getattr(config, "game_trace_cache_root"),
            max_bytes=getattr(config, "game_trace_cache_max_bytes"),
            max_game_bytes=getattr(config, "game_trace_cache_max_game_bytes"),
            max_entry_bytes=getattr(config, "game_trace_cache_max_entry_bytes"),
            min_free_bytes=getattr(config, "game_trace_cache_min_free_bytes"),
        )

    def read(self, identity: GameTraceIdentity, address: TraceAddress) -> CacheRead:
        """Read a verified entry, returning a miss for every unsafe condition."""

        path = self.entry_path(identity, address)
        with self._io_lock:
            return self._read_locked(identity, address, path)

    def read_manifest(self, identity: GameTraceIdentity) -> GameTraceManifest | None:
        """Return the verified completed manifest for ``identity``, if any."""

        with self._io_lock:
            manifest = self._read_manifest_locked(identity)
            if manifest is None:
                return None
            return GameTraceManifest(
                identity_fingerprint=identity.fingerprint,
                addresses=manifest.addresses,
                trace_sha256=MappingProxyType(dict(manifest.trace_sha256)),
                completed_at_ns=manifest.completed_at_ns,
            )

    def get(self, identity: GameTraceIdentity, address: TraceAddress) -> _JsonValue | None:
        """Return a cached value, or ``None`` on a miss.

        Prefer :meth:`read` when a cached JSON ``null`` must be distinguished
        from a miss.
        """

        result = self.read(identity, address)
        return result.value if result.hit else None

    def put(
        self, identity: GameTraceIdentity, address: TraceAddress, value: _JsonValue
    ) -> bool:
        """Atomically cache one JSON value as a one-address completed game.

        ``TypeError``/``ValueError`` means the supplied value is not strict
        JSON.  Operational failures and values over the configured size limit
        simply return ``False`` so caching cannot change reconstruction
        correctness.  Use :meth:`get_or_materialize` for a multi-address game:
        its manifest is committed only after every requested address is ready.
        """

        normalised = _normalise_json_value(value)
        with self._io_lock:
            self._invalidate_manifest_locked(identity)
            if not self._put_normalised_locked(identity, address, normalised):
                return False
            payload_sha256 = _sha256_prefixed(_canonical_json(normalised))
            if not self._read_entry_locked(
                identity,
                address,
                self.entry_path(identity, address),
                expected_payload_sha256=payload_sha256,
            ).hit:
                return False
            return self._write_manifest_locked(
                identity, {address: payload_sha256}
            )

    def get_or_compute(
        self,
        identity: GameTraceIdentity,
        address: TraceAddress,
        compute: Callable[[], _T],
    ) -> _T:
        """Return a cached value or run one process-local singleflight producer.

        The producer runs outside cache I/O locks.  Concurrent callers for the
        same immutable game/address wait for its result, while callers for a
        different game or address proceed independently.  A failed producer is
        propagated to every waiter and is never persisted.
        """

        cached = self.read(identity, address)
        if cached.hit:
            return cast(_T, cached.value)

        flight_key = (identity.cache_key, address.filename)
        with self._flight_lock:
            flight = self._flights.get(flight_key)
            if flight is None:
                flight = _Flight()
                self._flights[flight_key] = flight
                leader = True
            else:
                leader = False

        if not leader:
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            return cast(_T, flight.value)

        try:
            result = _normalise_json_value(compute())
            # A write miss (disk full, transient I/O, or an oversized entry)
            # must not discard the freshly reconstructed result.
            self.put(identity, address, result)
            flight.value = result
            return cast(_T, result)
        except BaseException as exc:
            flight.error = exc
            raise
        finally:
            with self._flight_lock:
                self._flights.pop(flight_key, None)
                flight.event.set()

    def get_or_materialize(
        self,
        identity: GameTraceIdentity,
        addresses: Iterable[TraceAddress],
        requested_address: TraceAddress,
        build_trace: Callable[[TraceAddress], _T],
    ) -> _T:
        """Materialize one game's selectable addresses and return one payload.

        This is the integration-oriented companion to :meth:`get_or_compute`.
        On a cold or partial cache it calls ``build_trace`` once for every
        unique address in ``addresses`` and writes each result independently.
        Concurrent requests for the *same immutable game identity* share one
        process-local flight, so a step/stage selection during materialization
        cannot trigger a second forward pass.  Different games are never
        serialized together.

        The requested address must be included in ``addresses``.  A failed
        builder aborts the current flight; successfully completed earlier
        addresses remain valid cache entries and are reused on retry.
        """

        address_list = _normalise_addresses(addresses, requested_address)
        while True:
            with self._io_lock:
                cached_values = {
                    address: self._read_locked(
                        identity, address, self.entry_path(identity, address)
                    )
                    for address in address_list
                }
            if all(cached.hit for cached in cached_values.values()):
                return cast(_T, cached_values[requested_address].value)

            with self._flight_lock:
                flight = self._game_flights.get(identity.cache_key)
                if flight is None:
                    flight = _GameFlight()
                    self._game_flights[identity.cache_key] = flight
                    leader = True
                else:
                    leader = False

            if not leader:
                flight.event.wait()
                if flight.error is not None:
                    raise flight.error
                # The leader may have received a smaller address set than this
                # caller, or an oversized entry may intentionally have skipped
                # disk persistence.  Recheck and become the next leader only
                # if this caller still needs materialization.
                if requested_address in flight.values:
                    return cast(_T, flight.values[requested_address])
                continue

            try:
                # A manifest is the disk commit point.  Once it is removed,
                # crash/restart readers treat even already-written entries as
                # a partial miss until the new completed manifest replaces it.
                with self._io_lock:
                    self._invalidate_manifest_locked(identity)

                values: dict[TraceAddress, _JsonValue] = {}
                trace_checksums: dict[TraceAddress, str] = {}
                fully_persisted = True
                for address in address_list:
                    value = _normalise_json_value(build_trace(address))
                    with self._io_lock:
                        persisted = self._put_normalised_locked(
                            identity, address, value
                        )
                    values[address] = value
                    trace_checksums[address] = _sha256_prefixed(
                        _canonical_json(value)
                    )
                    fully_persisted = fully_persisted and persisted

                if fully_persisted:
                    with self._io_lock:
                        # Recheck raw entry bytes before publishing the
                        # manifest.  A concurrent LRU prune or file failure
                        # leaves this generation uncommitted instead of
                        # creating a manifest that promises absent traces.
                        verified = all(
                            self._read_entry_locked(
                                identity,
                                address,
                                self.entry_path(identity, address),
                                expected_payload_sha256=trace_checksums[address],
                            ).hit
                            for address in address_list
                        )
                        if verified:
                            self._write_manifest_locked(identity, trace_checksums)
                flight.values = values
                return cast(_T, values[requested_address])
            except BaseException as exc:
                flight.error = exc
                raise
            finally:
                with self._flight_lock:
                    self._game_flights.pop(identity.cache_key, None)
                    flight.event.set()

    def prune(self) -> int:
        """Evict least-recently-used entries until every configured bound holds."""

        with self._io_lock:
            return self._prune_locked()

    def invalidate_game(self, identity: GameTraceIdentity) -> int:
        """Remove every regular cache entry for one exact immutable identity."""

        game_root = self._games_root / identity.cache_key
        with self._io_lock:
            usage = self._measure_game_usage_locked(game_root, identity.cache_key)
            if usage is None:
                return 0
            if not self._evict_game_locked(usage):
                return 0
            removed = max(1, usage.entry_count)
            self._metrics.evictions += removed
            return removed

    def _read_locked(
        self, identity: GameTraceIdentity, address: TraceAddress, path: Path
    ) -> CacheRead:
        """Read only entries committed by a valid manifest generation."""

        manifest = self._read_manifest_locked(identity)
        if manifest is None:
            return CacheRead(hit=False)
        expected_payload_sha256 = manifest.trace_sha256.get(address)
        if expected_payload_sha256 is None:
            return CacheRead(hit=False)
        return self._read_entry_locked(
            identity,
            address,
            path,
            expected_payload_sha256=expected_payload_sha256,
        )

    def _read_entry_locked(
        self,
        identity: GameTraceIdentity,
        address: TraceAddress,
        path: Path,
        *,
        expected_payload_sha256: str | None = None,
    ) -> CacheRead:
        """Read one raw entry; callers decide whether it is manifest-committed."""

        try:
            entry_stat = os.lstat(path)
        except FileNotFoundError:
            return CacheRead(hit=False)
        except OSError:
            return CacheRead(hit=False)

        if not self._is_private_regular_file(entry_stat):
            self._discard_regular_file(path, entry_stat)
            return CacheRead(hit=False)
        if entry_stat.st_size > self.max_entry_bytes:
            self._discard_regular_file(path, entry_stat)
            return CacheRead(hit=False)

        try:
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as raw_file:
                opened_stat = os.fstat(raw_file.fileno())
                if not self._is_private_regular_file(opened_stat):
                    return CacheRead(hit=False)
                with gzip.GzipFile(fileobj=raw_file, mode="rb") as zipped_file:
                    raw_entry = zipped_file.read(self.max_entry_bytes + 1)
                    if len(raw_entry) > self.max_entry_bytes or zipped_file.read(1):
                        raise ValueError("decoded cache entry exceeds size limit")
        except (EOFError, OSError, ValueError, gzip.BadGzipFile):
            self._discard_regular_file(path, entry_stat)
            return CacheRead(hit=False)

        try:
            record = json.loads(raw_entry.decode("utf-8"))
            value, payload_sha256 = self._verify_record(identity, address, record)
            if expected_payload_sha256 is not None and not _constant_time_equal(
                expected_payload_sha256, payload_sha256
            ):
                raise ValueError("manifest trace checksum mismatch")
        except (TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            self._discard_regular_file(path, entry_stat)
            return CacheRead(hit=False)

        self._touch(path)
        return CacheRead(hit=True, value=value)

    def _put_normalised_locked(
        self,
        identity: GameTraceIdentity,
        address: TraceAddress,
        value: _JsonValue,
    ) -> bool:
        now_ns = _clock_value(self._clock_ns)
        record_without_checksum: dict[str, _JsonValue] = {
            "schema": _CACHE_SCHEMA,
            "identity": identity.as_dict(),
            "identity_fingerprint": identity.fingerprint,
            "address": address.as_dict(),
            "created_at_ns": now_ns,
            "payload": value,
            "payload_sha256": _sha256_prefixed(_canonical_json(value)),
        }
        record = dict(record_without_checksum)
        record["entry_sha256"] = _sha256_prefixed(
            _canonical_json(record_without_checksum)
        )
        raw_entry = _canonical_json(record)
        if len(raw_entry) > self.max_entry_bytes:
            self._metrics.entry_write_failures += 1
            return False
        compressed_entry = gzip.compress(raw_entry, mtime=0)
        if len(compressed_entry) > self.max_entry_bytes:
            self._metrics.entry_write_failures += 1
            return False
        if not self._has_minimum_free_space_locked(len(compressed_entry)):
            return False

        game_root = self._games_root / identity.cache_key
        steps_root = game_root / "steps"
        path = steps_root / address.filename
        try:
            self._ensure_private_directory(game_root)
            self._ensure_private_directory(steps_root)
            self._atomic_replace(path, compressed_entry)
            self._touch(path, now_ns=now_ns)
            self._prune_locked()
        except (OSError, UnsafeCacheRootError):
            self._metrics.entry_write_failures += 1
            return False
        self._metrics.entry_writes += 1
        return True

    def _read_manifest_locked(self, identity: GameTraceIdentity) -> _Manifest | None:
        """Read the atomic game commit record, treating every defect as a miss."""

        path = self.manifest_path(identity)
        decoded = self._read_private_gzip_json_locked(path)
        if decoded is None:
            return None
        record, file_stat = decoded
        try:
            return self._verify_manifest_record(identity, record)
        except (TypeError, ValueError):
            self._discard_regular_file(path, file_stat)
            return None

    def _write_manifest_locked(
        self,
        identity: GameTraceIdentity,
        trace_checksums: Mapping[TraceAddress, str],
    ) -> bool:
        """Publish an all-or-nothing set of already-written trace entries."""

        ordered_addresses = tuple(
            sorted(trace_checksums, key=lambda item: (item.step_index, item.stage))
        )
        if not ordered_addresses:
            self._metrics.manifest_write_failures += 1
            return False
        traces: list[dict[str, _JsonValue]] = []
        try:
            for address in ordered_addresses:
                digest = _normalise_sha256(trace_checksums[address])
                traces.append(
                    {
                        "address": address.as_dict(),
                        "payload_sha256": digest,
                    }
                )
            unsigned: dict[str, _JsonValue] = {
                "schema": _MANIFEST_SCHEMA,
                "identity": identity.as_dict(),
                "identity_fingerprint": identity.fingerprint,
                "completed_at_ns": _clock_value(self._clock_ns),
                "addresses": [address.as_dict() for address in ordered_addresses],
                "traces": traces,
            }
            record = dict(unsigned)
            record["manifest_sha256"] = _sha256_prefixed(_canonical_json(unsigned))
            raw_manifest = _canonical_json(record)
        except (TypeError, ValueError):
            self._metrics.manifest_write_failures += 1
            return False
        if len(raw_manifest) > self.max_entry_bytes:
            self._metrics.manifest_write_failures += 1
            return False
        compressed_manifest = gzip.compress(raw_manifest, mtime=0)
        if len(compressed_manifest) > self.max_entry_bytes:
            self._metrics.manifest_write_failures += 1
            return False
        if not self._has_minimum_free_space_locked(len(compressed_manifest)):
            self._metrics.manifest_write_failures += 1
            return False
        try:
            game_root = self._games_root / identity.cache_key
            self._ensure_private_directory(game_root)
            self._atomic_replace(self.manifest_path(identity), compressed_manifest)
            self._touch(self.manifest_path(identity))
            self._prune_locked()
        except (OSError, UnsafeCacheRootError):
            self._metrics.manifest_write_failures += 1
            return False
        manifest_stat = _lstat_or_none(self.manifest_path(identity))
        if manifest_stat is None or not self._is_private_regular_file(manifest_stat):
            # The newly committed game itself may be the LRU victim once the
            # manifest/directory bytes are included in the configured bound.
            self._metrics.manifest_write_failures += 1
            return False
        self._metrics.manifest_writes += 1
        return True

    def _verify_manifest_record(
        self, identity: GameTraceIdentity, record: object
    ) -> _Manifest:
        if not isinstance(record, dict):
            raise ValueError("cache manifest must be an object")
        expected_fields = {
            "schema",
            "identity",
            "identity_fingerprint",
            "completed_at_ns",
            "addresses",
            "traces",
            "manifest_sha256",
        }
        if set(record) != expected_fields:
            raise ValueError("cache manifest fields do not match schema")
        checksum = record["manifest_sha256"]
        if not isinstance(checksum, str):
            raise ValueError("cache manifest checksum is not a string")
        unsigned = {
            name: value for name, value in record.items() if name != "manifest_sha256"
        }
        if not _constant_time_equal(
            checksum, _sha256_prefixed(_canonical_json(unsigned))
        ):
            raise ValueError("cache manifest checksum mismatch")
        if record["schema"] != _MANIFEST_SCHEMA:
            raise ValueError("cache manifest schema mismatch")
        if record["identity"] != identity.as_dict():
            raise ValueError("cache manifest identity mismatch")
        if record["identity_fingerprint"] != identity.fingerprint:
            raise ValueError("cache manifest fingerprint mismatch")
        completed_at_ns = record["completed_at_ns"]
        if isinstance(completed_at_ns, bool) or not isinstance(completed_at_ns, int):
            raise ValueError("cache manifest timestamp is invalid")
        addresses_value = record["addresses"]
        traces_value = record["traces"]
        if not isinstance(addresses_value, list) or not isinstance(traces_value, list):
            raise ValueError("cache manifest addresses/traces are invalid")
        addresses = tuple(_address_from_dict(item) for item in addresses_value)
        if not addresses or len(set(addresses)) != len(addresses):
            raise ValueError("cache manifest address set is invalid")
        if addresses != tuple(sorted(addresses, key=lambda item: (item.step_index, item.stage))):
            raise ValueError("cache manifest address set is not canonical")
        checksums: dict[TraceAddress, str] = {}
        for trace in traces_value:
            if not isinstance(trace, dict) or set(trace) != {"address", "payload_sha256"}:
                raise ValueError("cache manifest trace record is invalid")
            address = _address_from_dict(trace["address"])
            if address in checksums:
                raise ValueError("cache manifest trace address is duplicated")
            checksums[address] = _normalise_sha256(trace["payload_sha256"])
        if tuple(checksums) != addresses:
            raise ValueError("cache manifest traces do not match address set")
        return _Manifest(
            addresses=addresses,
            trace_sha256=MappingProxyType(dict(checksums)),
            completed_at_ns=completed_at_ns,
        )

    def _read_private_gzip_json_locked(
        self, path: Path
    ) -> tuple[object, os.stat_result] | None:
        """Read a bounded private gzip JSON file without following symlinks."""

        file_stat = _lstat_or_none(path)
        if file_stat is None:
            return None
        if not self._is_private_regular_file(file_stat):
            self._discard_regular_file(path, file_stat)
            return None
        if file_stat.st_size > self.max_entry_bytes:
            self._discard_regular_file(path, file_stat)
            return None
        try:
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb") as raw_file:
                opened_stat = os.fstat(raw_file.fileno())
                if not self._is_private_regular_file(opened_stat):
                    return None
                with gzip.GzipFile(fileobj=raw_file, mode="rb") as zipped_file:
                    raw_value = zipped_file.read(self.max_entry_bytes + 1)
                    if len(raw_value) > self.max_entry_bytes or zipped_file.read(1):
                        raise ValueError("decoded cache file exceeds size limit")
            return json.loads(raw_value.decode("utf-8")), file_stat
        except (EOFError, OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._discard_regular_file(path, file_stat)
            return None

    def _invalidate_manifest_locked(self, identity: GameTraceIdentity) -> bool:
        return self._discard_regular_file(self.manifest_path(identity))

    def _has_minimum_free_space_locked(self, pending_bytes: int) -> bool:
        """Preserve configured free space; cache writes are optional."""

        try:
            free_bytes = shutil.disk_usage(self.root).free
        except OSError:
            self._metrics.low_space_rejections += 1
            return False
        if free_bytes < self.min_free_bytes + pending_bytes:
            self._metrics.low_space_rejections += 1
            return False
        return True

    def _verify_record(
        self,
        identity: GameTraceIdentity,
        address: TraceAddress,
        record: object,
    ) -> tuple[_JsonValue, str]:
        if not isinstance(record, dict):
            raise ValueError("cache record must be an object")
        expected_fields = {
            "schema",
            "identity",
            "identity_fingerprint",
            "address",
            "created_at_ns",
            "payload",
            "payload_sha256",
            "entry_sha256",
        }
        if set(record) != expected_fields:
            raise ValueError("cache record fields do not match schema")
        entry_sha256 = record["entry_sha256"]
        if not isinstance(entry_sha256, str):
            raise ValueError("cache record checksum is not a string")
        unsigned = {name: value for name, value in record.items() if name != "entry_sha256"}
        if not _constant_time_equal(
            entry_sha256, _sha256_prefixed(_canonical_json(unsigned))
        ):
            raise ValueError("cache record checksum mismatch")
        if record["schema"] != _CACHE_SCHEMA:
            raise ValueError("cache record schema mismatch")
        if record["identity"] != identity.as_dict():
            raise ValueError("cache record identity mismatch")
        if record["identity_fingerprint"] != identity.fingerprint:
            raise ValueError("cache record fingerprint mismatch")
        if record["address"] != address.as_dict():
            raise ValueError("cache record address mismatch")
        if isinstance(record["created_at_ns"], bool) or not isinstance(
            record["created_at_ns"], int
        ):
            raise ValueError("cache record creation timestamp is invalid")
        payload = _normalise_json_value(record["payload"])
        payload_sha256 = record["payload_sha256"]
        if not isinstance(payload_sha256, str) or not _constant_time_equal(
            payload_sha256, _sha256_prefixed(_canonical_json(payload))
        ):
            raise ValueError("cache payload checksum mismatch")
        return payload, payload_sha256

    def _prune_locked(self) -> int:
        self._cleanup_eviction_tombs_locked()
        games = self._scan_games_locked()
        evicted = 0
        survivors: list[_GameUsage] = []
        for game in games:
            if self._limits_exceeded(
                game.entry_count,
                game.size,
                max_entries=self.max_entries_per_game,
                max_bytes=self.max_bytes_per_game,
            ):
                if self._evict_game_locked(game):
                    evicted += max(1, game.entry_count)
                else:
                    survivors.append(game)
            else:
                survivors.append(game)

        total_entries = sum(game.entry_count for game in survivors)
        total_bytes = sum(game.size for game in survivors)
        survivors.sort(key=lambda game: (game.access_ns, game.game_key))
        while survivors and self._limits_exceeded(
            total_entries,
            total_bytes,
            max_entries=self.max_entries,
            max_bytes=self.max_total_bytes,
        ):
            oldest = survivors[0]
            if not self._evict_game_locked(oldest):
                # The cache is disposable.  If an unexpected filesystem error
                # prevents removal, do not spin or delete a newer game.
                break
            survivors.pop(0)
            total_entries -= oldest.entry_count
            total_bytes -= oldest.size
            evicted += max(1, oldest.entry_count)
        self._metrics.evictions += evicted
        return evicted

    @staticmethod
    def _limits_exceeded(
        entry_count: int,
        byte_count: int,
        *,
        max_entries: int | None,
        max_bytes: int | None,
    ) -> bool:
        return (
            (max_entries is not None and entry_count > max_entries)
            or (max_bytes is not None and byte_count > max_bytes)
        )

    def _scan_games_locked(self) -> list[_GameUsage]:
        games: list[_GameUsage] = []
        games_stat = _lstat_or_none(self._games_root)
        if games_stat is None or not self._is_private_directory(games_stat):
            return games
        try:
            game_directories = list(os.scandir(self._games_root))
        except OSError:
            return games
        for game_directory in game_directories:
            if not _GAME_DIRECTORY_RE.fullmatch(game_directory.name):
                continue
            usage = self._measure_game_usage_locked(
                Path(game_directory.path), game_directory.name
            )
            if usage is not None:
                games.append(usage)
        return games

    def _measure_game_usage_locked(
        self, game_root: Path, game_key: str
    ) -> _GameUsage | None:
        """Measure every private regular file and directory in one game root."""

        root_stat = _lstat_or_none(game_root)
        if root_stat is None or not self._is_private_directory(root_stat):
            return None
        total_bytes = root_stat.st_size
        # Directory atime is creation/lookup metadata, not a user trace
        # selection.  It must count toward storage but must not outrank a
        # trace's explicit LRU touch (especially on filesystems that preserve
        # a later directory creation timestamp).
        access_ns: int | None = None
        entry_count = 0
        pending_directories = [game_root]
        while pending_directories:
            directory = pending_directories.pop()
            try:
                children = list(os.scandir(directory))
            except OSError:
                continue
            for child in children:
                path = Path(child.path)
                try:
                    child_stat = child.stat(follow_symlinks=False)
                except OSError:
                    continue
                if self._is_private_directory(child_stat):
                    total_bytes += child_stat.st_size
                    pending_directories.append(path)
                    continue
                if self._is_private_regular_file(child_stat):
                    if child_stat.st_size > self.max_entry_bytes:
                        self._discard_regular_file(path, child_stat)
                        continue
                    total_bytes += child_stat.st_size
                    access_ns = max(access_ns or 0, child_stat.st_atime_ns)
                    if _ENTRY_FILENAME_RE.fullmatch(child.name):
                        entry_count += 1
                    continue
                # Symlinks and unexpected special files are neither trusted
                # cache state nor part of a published manifest.  They are
                # skipped; a whole-game eviction later removes their parent.
        return _GameUsage(
            path=game_root,
            game_key=game_key,
            size=total_bytes,
            entry_count=entry_count,
            access_ns=root_stat.st_atime_ns if access_ns is None else access_ns,
        )

    def _evict_game_locked(self, game: _GameUsage) -> bool:
        """Atomically hide a whole game before best-effort private cleanup."""

        current = _lstat_or_none(game.path)
        if current is None or not self._is_private_directory(current):
            return False
        tomb = self.root / f".evict-{game.game_key}-{secrets.token_hex(12)}"
        try:
            os.replace(game.path, tomb)
            _fsync_directory(self._games_root)
            _fsync_directory(self.root)
        except OSError:
            return False
        self._remove_private_tree_locked(tomb)
        return True

    def _cleanup_eviction_tombs_locked(self) -> None:
        try:
            candidates = list(os.scandir(self.root))
        except OSError:
            return
        for candidate in candidates:
            if not candidate.name.startswith(".evict-"):
                continue
            path = Path(candidate.path)
            candidate_stat = _lstat_or_none(path)
            if candidate_stat is not None and self._is_private_directory(candidate_stat):
                self._remove_private_tree_locked(path)

    def _remove_private_tree_locked(self, root: Path) -> bool:
        """Remove only our owned, private tree without following symlinks."""

        root_stat = _lstat_or_none(root)
        if root_stat is None or not self._is_private_directory(root_stat):
            return False
        try:
            children = list(os.scandir(root))
        except OSError:
            return False
        for child in children:
            path = Path(child.path)
            child_stat = _lstat_or_none(path)
            if child_stat is None:
                continue
            if stat.S_ISLNK(child_stat.st_mode) or self._is_private_regular_file(
                child_stat
            ):
                try:
                    path.unlink()
                except OSError:
                    return False
                continue
            if self._is_private_directory(child_stat):
                if not self._remove_private_tree_locked(path):
                    return False
                continue
            return False
        try:
            root.rmdir()
        except OSError:
            return False
        return True

    def _ensure_private_directory(self, path: Path) -> None:
        """Create or repair one owned non-symlink directory with mode 0700."""

        try:
            existing = os.lstat(path)
        except FileNotFoundError:
            try:
                os.mkdir(path, mode=0o700)
            except FileExistsError:
                pass
            existing = os.lstat(path)
        if not self._is_owned_directory(existing):
            raise UnsafeCacheRootError(f"cache directory is unsafe: {path}")
        try:
            os.chmod(path, 0o700, follow_symlinks=False)
        except (NotImplementedError, OSError) as exc:
            raise UnsafeCacheRootError(f"cannot secure cache directory: {path}") from exc
        secured = os.lstat(path)
        if not self._is_private_directory(secured):
            raise UnsafeCacheRootError(f"cache directory is not private: {path}")

    def _ensure_private_directory_tree(self, path: Path) -> None:
        """Create missing root ancestors one component at a time, privately."""

        missing: list[Path] = []
        cursor = path
        while True:
            existing = _lstat_or_none(cursor)
            if existing is not None:
                if stat.S_ISLNK(existing.st_mode) or not stat.S_ISDIR(existing.st_mode):
                    raise UnsafeCacheRootError(f"cache root parent is unsafe: {cursor}")
                break
            if cursor.parent == cursor:
                raise UnsafeCacheRootError(f"cache root has no usable parent: {path}")
            missing.append(cursor)
            cursor = cursor.parent
        if not missing:
            self._ensure_private_directory(path)
            return
        for directory in reversed(missing):
            self._ensure_private_directory(directory)

    @staticmethod
    def _is_owned_directory(file_stat: os.stat_result) -> bool:
        return stat.S_ISDIR(file_stat.st_mode) and not stat.S_ISLNK(file_stat.st_mode) and (
            not hasattr(os, "geteuid") or file_stat.st_uid == os.geteuid()
        )

    def _is_private_directory(self, file_stat: os.stat_result) -> bool:
        return self._is_owned_directory(file_stat) and (stat.S_IMODE(file_stat.st_mode) & 0o077) == 0

    @staticmethod
    def _is_private_regular_file(file_stat: os.stat_result) -> bool:
        return stat.S_ISREG(file_stat.st_mode) and not stat.S_ISLNK(file_stat.st_mode) and (
            not hasattr(os, "geteuid") or file_stat.st_uid == os.geteuid()
        ) and (stat.S_IMODE(file_stat.st_mode) & 0o077) == 0

    def _atomic_replace(self, path: Path, data: bytes) -> None:
        previous = _lstat_or_none(path)
        if previous is not None and not self._is_private_regular_file(previous):
            raise UnsafeCacheRootError(f"refusing to replace unsafe cache entry: {path}")
        temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as temporary_file:
                descriptor = None
                temporary_file.write(data)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.chmod(temporary, 0o600, follow_symlinks=False)
            os.replace(temporary, path)
            os.chmod(path, 0o600, follow_symlinks=False)
            _fsync_directory(path.parent)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _touch(self, path: Path, *, now_ns: int | None = None) -> None:
        try:
            entry_stat = os.lstat(path)
            if not self._is_private_regular_file(entry_stat):
                return
            access_ns = _clock_value(self._clock_ns) if now_ns is None else now_ns
            os.utime(path, ns=(access_ns, entry_stat.st_mtime_ns), follow_symlinks=False)
        except OSError:
            # Access-time updates only guide eviction; a failed touch must not
            # turn an otherwise verified hit into an application failure.
            return

    @staticmethod
    def _discard_regular_file(path: Path, known_stat: os.stat_result | None = None) -> bool:
        try:
            current = os.lstat(path)
        except FileNotFoundError:
            return False
        except OSError:
            return False
        if known_stat is not None and (
            current.st_dev != known_stat.st_dev or current.st_ino != known_stat.st_ino
        ):
            # Another process may have atomically replaced the bad entry with a
            # good one after we inspected it.  Never delete that new file.
            return False
        if not stat.S_ISREG(current.st_mode) or stat.S_ISLNK(current.st_mode):
            return False
        try:
            path.unlink()
        except OSError:
            return False
        return True


def _prepare_root(root: Path, *, unsafe_test_root: bool) -> Path:
    if not root.is_absolute():
        raise UnsafeCacheRootError("cache root must be an absolute path")
    try:
        resolved_root = root.resolve(strict=False)
    except OSError as exc:
        raise UnsafeCacheRootError(f"cannot resolve cache root: {root}") from exc
    if not unsafe_test_root:
        try:
            resolved_tmp = Path("/tmp").resolve(strict=True)
        except OSError as exc:
            raise UnsafeCacheRootError("/tmp is unavailable for the cache") from exc
        try:
            resolved_root.relative_to(resolved_tmp)
        except ValueError as exc:
            raise UnsafeCacheRootError(
                "production game-trace cache root must resolve beneath /tmp; "
                "pass unsafe_test_root=True only in tests"
            ) from exc
        if resolved_root == resolved_tmp:
            raise UnsafeCacheRootError("refusing to use /tmp itself as a cache root")

    # ``Path.resolve`` intentionally follows symlinks, so production inspects
    # every existing lexical component.  /tmp itself may be a sanctioned
    # platform symlink (notably on macOS); symlinks below it are not accepted.
    # The explicit test escape hatch permits platform-owned ancestor symlinks
    # such as macOS's /var -> /private/var, but never a symlink *root*.
    if unsafe_test_root:
        root_stat = _lstat_or_none(root)
        if root_stat is not None and stat.S_ISLNK(root_stat.st_mode):
            raise UnsafeCacheRootError(f"cache root cannot be a symlink: {root}")
    else:
        _reject_unsafe_symlink_components(root)
    return resolved_root


def _reject_unsafe_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    tmp_lexical = Path("/tmp")
    for part in path.parts[1:]:
        current /= part
        try:
            component_stat = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise UnsafeCacheRootError(f"cannot inspect cache path: {current}") from exc
        if stat.S_ISLNK(component_stat.st_mode) and current != tmp_lexical:
            raise UnsafeCacheRootError(f"cache root cannot traverse symlink: {current}")
        if (
            not stat.S_ISDIR(component_stat.st_mode)
            and current != path
            and current != tmp_lexical
        ):
            raise UnsafeCacheRootError(f"cache root has non-directory parent: {current}")


def _normalise_sha256(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("SHA-256 digest must be a string")
    match = _SHA256_RE.fullmatch(value)
    if match is None:
        raise ValueError("SHA-256 digest must have sha256:<64-hex> form")
    return "sha256:" + match.group(1).lower()


def _validate_non_negative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_limit(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return value


def _normalise_addresses(
    addresses: Iterable[TraceAddress], requested_address: TraceAddress
) -> tuple[TraceAddress, ...]:
    if not isinstance(requested_address, TraceAddress):
        raise TypeError("requested_address must be a TraceAddress")
    unique: dict[TraceAddress, None] = {}
    for address in addresses:
        if not isinstance(address, TraceAddress):
            raise TypeError("addresses must contain only TraceAddress values")
        unique[address] = None
    if requested_address not in unique:
        raise ValueError("requested_address must be included in addresses")
    return tuple(unique)


def _address_from_dict(value: object) -> TraceAddress:
    if not isinstance(value, dict) or set(value) != {"step_index", "stage"}:
        raise ValueError("cache address must contain step_index and stage")
    return TraceAddress(value["step_index"], value["stage"])


def _normalise_json_value(value: object) -> _JsonValue:
    """Return a detached strict-JSON value, rejecting NaN and custom types."""

    try:
        encoded = _canonical_json(value)
        decoded = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise TypeError("cache values must be strict JSON values") from exc
    return cast(_JsonValue, decoded)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_prefixed(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _constant_time_equal(left: str, right: str) -> bool:
    # ``compare_digest`` is useful even though cache files are not secrets: it
    # keeps checksum comparison behavior conventional and exact.
    import hmac

    return hmac.compare_digest(left, right)


def _clock_value(clock_ns: Callable[[], int]) -> int:
    value = clock_ns()
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("clock_ns must return a non-negative integer")
    return value


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError:
        return None


def _fsync_directory(path: Path) -> None:
    """Best-effort metadata persistence; unsupported filesystems are fine."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "CacheMetrics",
    "CacheRead",
    "GameTraceAddress",
    "GameTraceCache",
    "GameTraceIdentity",
    "GameTraceKey",
    "GameTraceManifest",
    "PhysicalGameTraceCache",
    "TraceAddress",
    "UnsafeCacheRootError",
]


# Integration spelling: the cache is keyed by a physical replay plus immutable
# runtime provenance, never by a transient UI selection.
PhysicalGameTraceCache = GameTraceCache
