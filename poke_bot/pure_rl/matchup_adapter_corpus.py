"""Bounded, checksummed staging for offline matchup-adapter fitting.

The authoritative temporal feature shards intentionally discard
``GameSequence.target_provenance`` during compaction.  Consequently the compact
shard alone cannot authorize an oracle matchup route: the visible
``opp_archetype`` string is useful evidence, but it does not prove which raw
archive member, full opponent deck, classifier contract, or agent package
produced that label.

This module joins each immutable feature stream to an independently
checksummed, row-aligned oracle index rebuilt from the pinned raw archives.  It
uses two streaming passes and a temporary SQLite membership database, then
publishes two immutable feature shards per route (train and validation).
No pass retains ``GameSequence`` objects or an episode-membership table in RAM.

This is an offline training path only.  Every emitted decision receives the
oracle route, while the public/runtime route is forced to ``UNKNOWN_ROUTE``.
Serving still requires its independent public-information router gate.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import pickle
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

from poke_bot.dataset import DATASET_CACHE_SCHEMA_VERSION, GameSequence
from poke_bot.feature_shards import (
    COMPACT_MODE_TEMPORAL_EXPERT,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
    iter_feature_shard,
)
from poke_bot.features import FEATURE_SCHEMA_VERSION
from poke_bot.matchup_adapter_activation import (
    TRAINING_TICKET_SCHEMA,
    adapter_training_ticket,
    gate_exclusions,
    normalize_matchup_identity,
)
from poke_bot.matchup_adapters import (
    EXPERT_IDS,
    UNKNOWN_ROUTE,
    route_for_archetype,
)


ORACLE_INDEX_HEADER_SCHEMA = "poke_bot.matchup_adapter_oracle_index_header/v1"
ORACLE_INDEX_ROW_SCHEMA = "poke_bot.matchup_adapter_oracle_index_row/v1"
ORACLE_INDEX_FOOTER_SCHEMA = "poke_bot.matchup_adapter_oracle_index_footer/v1"
ORACLE_MANIFEST_SCHEMA = "poke_bot.matchup_adapter_oracle_manifest/v1"
PACKAGE_REGISTRY_SCHEMA = "poke_bot.matchup_adapter_package_registry/v1"
STAGED_CORPUS_SCHEMA = "poke_bot.matchup_adapter_staged_corpus/v1"

_SHA256_PREFIX = "sha256:"
_SHA256_LENGTH = len(_SHA256_PREFIX) + 64
_CLASSIFIER_METHODS = frozenset(
    {
        "representative_exact",
        "registered_signature",
        "artifact_signature",
        "derived_primary_ace",
    }
)


def sha256_file(path: Path) -> str:
    """Return a canonical streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return _SHA256_PREFIX + digest.hexdigest()


def full_deck_digest(card_ids: Iterable[int]) -> str:
    """Return a stable multiset digest for one exact 60-card opponent deck."""

    raw_cards = list(card_ids)
    if any(type(card_id) is not int for card_id in raw_cards):
        raise ValueError("full-deck fingerprints require exact integer card IDs")
    cards = sorted(raw_cards)
    if len(cards) != 60 or any(card_id < 0 for card_id in cards):
        raise ValueError("full-deck fingerprints require exactly 60 nonnegative IDs")
    raw = json.dumps(cards, separators=(",", ":")).encode("ascii")
    return _SHA256_PREFIX + hashlib.sha256(raw).hexdigest()


def _require_digest(value: Any, field: str) -> str:
    digest = str(value or "").strip().lower()
    if (
        len(digest) != _SHA256_LENGTH
        or not digest.startswith(_SHA256_PREFIX)
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise ValueError(f"{field} must be a canonical sha256 digest")
    return digest


def _canonical_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = Path(path).stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class OracleEntry:
    """One exact, private offline identity rebuilt from a raw episode."""

    episode_id: str
    seat: int
    acting_archetype: str
    opponent_archetype: str
    opponent_id: str
    opponent_content_digest: str
    opponent_deck_digest: str
    source_archive_digest: str
    source_member_digest: str
    classifier_method: str
    formal_eval: bool
    training_eligible: bool
    public_router_digest: str = ""
    public_route_segments: tuple[tuple[int, int], ...] = ()
    public_decisions: int = 0
    public_confidence_threshold: float = 0.0
    public_consecutive_required: int = 0

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> "OracleEntry":
        if str(payload.get("schema") or "") != ORACLE_INDEX_ROW_SCHEMA:
            raise ValueError("invalid matchup-adapter oracle row schema")
        episode_id = str(payload.get("episode_id") or "").strip()
        acting = normalize_matchup_identity(payload.get("acting_archetype"))
        opponent = normalize_matchup_identity(payload.get("opponent_archetype"))
        opponent_id = normalize_matchup_identity(payload.get("opponent_id"))
        classifier_method = str(payload.get("classifier_method") or "").strip()
        raw_seat = payload.get("seat", -1)
        if type(raw_seat) is not int:
            raise ValueError("oracle row seat must be an exact integer")
        seat = raw_seat
        if not episode_id or seat not in (0, 1):
            raise ValueError("oracle row lacks an exact episode/seat identity")
        if acting != "alakazam":
            raise ValueError("oracle row is not an Alakazam acting seat")
        if (
            not opponent
            or not opponent_id
            or classifier_method not in _CLASSIFIER_METHODS
        ):
            raise ValueError("oracle row lacks opponent/classifier identity")
        # Exact booleans prevent truthy strings or absent fields from silently
        # authorizing a training row.
        formal_eval = payload.get("formal_eval")
        training_eligible = payload.get("training_eligible")
        if not isinstance(formal_eval, bool) or not isinstance(
            training_eligible, bool
        ):
            raise ValueError("oracle row eligibility fields must be booleans")
        public_router_digest = str(payload.get("public_router_digest") or "")
        raw_segments = payload.get("public_route_segments") or ()
        public_decisions = int(payload.get("public_decisions") or 0)
        segments: list[tuple[int, int]] = []
        previous_end = 0
        for raw_segment in raw_segments:
            if not isinstance(raw_segment, (list, tuple)) or len(raw_segment) != 2:
                raise ValueError("public route segment must be [start, end]")
            start, end = int(raw_segment[0]), int(raw_segment[1])
            if start < previous_end or end <= start or end > public_decisions:
                raise ValueError("public route segments are invalid or overlapping")
            segments.append((start, end))
            previous_end = end
        if public_router_digest:
            _require_digest(public_router_digest, "public router digest")
            if public_decisions <= 0:
                raise ValueError("public-routed oracle row has no decisions")
        elif segments or public_decisions:
            raise ValueError("public route evidence lacks a router digest")
        return cls(
            episode_id=episode_id,
            seat=seat,
            acting_archetype=acting,
            opponent_archetype=opponent,
            opponent_id=opponent_id,
            opponent_content_digest=_require_digest(
                payload.get("opponent_content_digest"),
                "oracle opponent package/content digest",
            ),
            opponent_deck_digest=_require_digest(
                payload.get("opponent_deck_digest"),
                "oracle opponent full-deck digest",
            ),
            source_archive_digest=_require_digest(
                payload.get("source_archive_digest"),
                "oracle source archive digest",
            ),
            source_member_digest=_require_digest(
                payload.get("source_member_digest"),
                "oracle source member digest",
            ),
            classifier_method=classifier_method,
            formal_eval=formal_eval,
            training_eligible=training_eligible,
            public_router_digest=public_router_digest,
            public_route_segments=tuple(segments),
            public_decisions=public_decisions,
            public_confidence_threshold=float(
                payload.get("public_confidence_threshold") or 0.0
            ),
            public_consecutive_required=int(
                payload.get("public_consecutive_required") or 0
            ),
        )

    def as_row(self) -> dict[str, Any]:
        row = {
            "schema": ORACLE_INDEX_ROW_SCHEMA,
            "episode_id": self.episode_id,
            "seat": self.seat,
            "acting_archetype": self.acting_archetype,
            "opponent_archetype": self.opponent_archetype,
            "opponent_id": self.opponent_id,
            "opponent_content_digest": self.opponent_content_digest,
            "opponent_deck_digest": self.opponent_deck_digest,
            "source_archive_digest": self.source_archive_digest,
            "source_member_digest": self.source_member_digest,
            "classifier_method": self.classifier_method,
            "formal_eval": self.formal_eval,
            "training_eligible": self.training_eligible,
        }
        if self.public_router_digest:
            row.update(
                {
                    "public_router_digest": self.public_router_digest,
                    "public_route_segments": [
                        [start, end] for start, end in self.public_route_segments
                    ],
                    "public_decisions": self.public_decisions,
                    "public_confidence_threshold": self.public_confidence_threshold,
                    "public_consecutive_required": self.public_consecutive_required,
                }
            )
        return row


def write_oracle_index(
    output_path: Path,
    *,
    source_feature_digest: str,
    classifier_digest: str,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Atomically write one bounded-memory, row-ordered oracle index.

    The caller must produce rows in the exact order of the corresponding
    feature shard.  :func:`stage_matchup_adapter_corpus` verifies that identity
    in lockstep before trusting any route.
    """

    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    source_digest = _require_digest(source_feature_digest, "source feature digest")
    classifier = _require_digest(classifier_digest, "classifier digest")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial.{os.getpid()}")
    row_digest = hashlib.sha256()
    count = 0
    header = {
        "schema": ORACLE_INDEX_HEADER_SCHEMA,
        "source_feature_digest": source_digest,
        "classifier_digest": classifier,
    }
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_line(header))
            for raw in rows:
                entry = OracleEntry.parse(raw)
                encoded = _canonical_line(entry.as_row())
                stream.write(encoded)
                row_digest.update(encoded)
                count += 1
            footer = {
                "schema": ORACLE_INDEX_FOOTER_SCHEMA,
                "rows": count,
                "rows_sha256": _SHA256_PREFIX + row_digest.hexdigest(),
            }
            stream.write(_canonical_line(footer))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        **header,
        "path": output.name,
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "rows": count,
        "rows_sha256": _SHA256_PREFIX + row_digest.hexdigest(),
    }


@dataclass(frozen=True)
class _VerifiedFile:
    path: Path
    digest: str
    identity: tuple[int, int, int, int, int]

    @classmethod
    def open(cls, path: Path, expected_digest: str) -> "_VerifiedFile":
        resolved = Path(path).expanduser().resolve()
        expected = _require_digest(expected_digest, f"digest for {resolved}")
        before = _stat_identity(resolved)
        actual = sha256_file(resolved)
        after = _stat_identity(resolved)
        if before != after:
            raise ValueError(f"file changed during checksum verification: {resolved}")
        if actual != expected:
            raise ValueError(
                f"checksum mismatch for {resolved}: expected={expected} actual={actual}"
            )
        return cls(resolved, actual, after)

    def assert_unchanged(self) -> None:
        if _stat_identity(self.path) != self.identity:
            raise ValueError(f"verified file changed during staging: {self.path}")


@dataclass(frozen=True)
class _SourcePair:
    feature: _VerifiedFile
    oracle: _VerifiedFile
    records: int
    classifier_digest: str
    source_archive_digest: str


def _feature_header(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("rb") as stream:
            header = pickle.load(stream)
    except (OSError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f"invalid source feature header: {path}") from exc
    if (
        not isinstance(header, dict)
        or header.get("format") != SHARD_FORMAT
        or int(header.get("format_version", -1)) != SHARD_FORMAT_VERSION
        or str(header.get("compact_mode") or "")
        != COMPACT_MODE_TEMPORAL_EXPERT
        or normalize_matchup_identity(header.get("required_archetype"))
        != "alakazam"
    ):
        raise ValueError("source feature shard lacks the authoritative Alakazam contract")
    return header


def _iter_oracle_index(pair: _SourcePair) -> Iterator[OracleEntry]:
    """Yield rows and validate the terminal count/digest before exhaustion."""

    pair.oracle.assert_unchanged()
    count = 0
    row_digest = hashlib.sha256()
    saw_footer = False
    with pair.oracle.path.open("rb") as stream:
        first = stream.readline()
        if not first:
            raise ValueError(f"empty oracle index: {pair.oracle.path}")
        try:
            header = json.loads(first)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid oracle index header: {pair.oracle.path}") from exc
        if (
            header.get("schema") != ORACLE_INDEX_HEADER_SCHEMA
            or _require_digest(
                header.get("source_feature_digest"), "oracle source feature digest"
            )
            != pair.feature.digest
            or _require_digest(
                header.get("classifier_digest"), "oracle classifier digest"
            )
            != pair.classifier_digest
        ):
            raise ValueError("oracle index header does not match its source contract")
        for raw_line in stream:
            if not raw_line.strip():
                raise ValueError("oracle index contains an empty line")
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid oracle index row: {pair.oracle.path}") from exc
            schema = str(payload.get("schema") or "")
            if schema == ORACLE_INDEX_FOOTER_SCHEMA:
                if saw_footer:
                    raise ValueError("oracle index has multiple footers")
                saw_footer = True
                if (
                    int(payload.get("rows", -1)) != count
                    or _require_digest(
                        payload.get("rows_sha256"), "oracle row manifest digest"
                    )
                    != _SHA256_PREFIX + row_digest.hexdigest()
                ):
                    raise ValueError("oracle index footer count/digest mismatch")
                if stream.read(1):
                    raise ValueError("oracle index has trailing data after footer")
                break
            if saw_footer:
                raise ValueError("oracle row appears after footer")
            entry = OracleEntry.parse(payload)
            row_digest.update(_canonical_line(entry.as_row()))
            count += 1
            yield entry
    if not saw_footer:
        raise ValueError(f"oracle index is missing its footer: {pair.oracle.path}")
    if count != pair.records:
        raise ValueError(
            f"oracle row count mismatch: expected={pair.records} actual={count}"
        )
    pair.oracle.assert_unchanged()


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload, _SHA256_PREFIX + hashlib.sha256(raw).hexdigest()


def _package_registry(
    path: Path,
    *,
    expected_digest: str,
) -> tuple[dict[str, str], str]:
    payload, digest = _read_json(Path(path).expanduser().resolve())
    if digest != _require_digest(expected_digest, "oracle package registry digest"):
        raise ValueError("package registry digest does not match oracle manifest")
    if payload.get("schema") != PACKAGE_REGISTRY_SCHEMA:
        raise ValueError("invalid matchup-adapter package registry schema")
    identities: dict[str, str] = {}
    canonical_digests: set[str] = set()
    for raw in payload.get("packages") or ():
        row = dict(raw)
        opponent_id = normalize_matchup_identity(row.get("opponent_id"))
        content_digest = _require_digest(
            row.get("content_digest"), "package registry content digest"
        )
        aliases = [
            normalize_matchup_identity(value) for value in row.get("aliases") or ()
        ]
        if not opponent_id or any(not value for value in aliases):
            raise ValueError("package registry contains an empty identity")
        if (
            opponent_id in aliases
            or len(set(aliases)) != len(aliases)
            or content_digest in canonical_digests
        ):
            raise ValueError("package registry repeats an identity/content digest")
        canonical_digests.add(content_digest)
        for identity in (opponent_id, *aliases):
            prior = identities.setdefault(identity, content_digest)
            if prior != content_digest:
                raise ValueError("package registry alias maps to conflicting packages")
    if not identities:
        raise ValueError("package registry is empty")
    return identities, digest


def _open_sources(
    feature_manifest_path: Path,
    oracle_manifest_path: Path,
) -> tuple[list[_SourcePair], str, str, str]:
    feature_path = Path(feature_manifest_path).expanduser().resolve()
    oracle_path = Path(oracle_manifest_path).expanduser().resolve()
    feature_payload, feature_digest = _read_json(feature_path)
    oracle_payload, oracle_digest = _read_json(oracle_path)
    if (
        feature_payload.get("format") != MANIFEST_FORMAT
        or int(feature_payload.get("format_version", -1))
        != MANIFEST_FORMAT_VERSION
        or str(feature_payload.get("compact_mode") or "")
        != COMPACT_MODE_TEMPORAL_EXPERT
    ):
        raise ValueError("source must be a temporal expert feature manifest")
    if oracle_payload.get("schema") != ORACLE_MANIFEST_SCHEMA:
        raise ValueError("invalid matchup-adapter oracle manifest schema")
    if _require_digest(
        oracle_payload.get("source_feature_manifest_digest"),
        "oracle source feature manifest digest",
    ) != feature_digest:
        raise ValueError("oracle manifest does not bind the feature manifest")
    classifier_digest = _require_digest(
        oracle_payload.get("classifier_digest"), "oracle classifier digest"
    )
    package_registry_digest = _require_digest(
        oracle_payload.get("package_registry_digest"),
        "oracle package registry digest",
    )
    oracle_rows: dict[str, dict[str, Any]] = {}
    for raw in oracle_payload.get("shards") or ():
        row = dict(raw)
        source_digest = _require_digest(
            row.get("source_feature_digest"), "oracle shard source digest"
        )
        if source_digest in oracle_rows:
            raise ValueError("oracle manifest repeats a source feature digest")
        oracle_rows[source_digest] = row

    source_rows = list(feature_payload.get("shards") or ())
    if not source_rows or len(source_rows) != len(oracle_rows):
        raise ValueError("feature/oracle manifests do not have one-to-one shards")
    pairs: list[_SourcePair] = []
    for raw in source_rows:
        row = dict(raw)
        feature = _VerifiedFile.open(
            feature_path.parent / str(row.get("path") or ""),
            str(row.get("sha256") or ""),
        )
        source_header = _feature_header(feature.path)
        if _require_digest(
            source_header.get("classifier_sha256"),
            "source feature classifier digest",
        ) != classifier_digest:
            raise ValueError("oracle classifier differs from source feature classifier")
        source_archive_digest = _require_digest(
            source_header.get("source_archive_sha256"),
            "source feature archive digest",
        )
        oracle_row = oracle_rows.pop(feature.digest, None)
        if oracle_row is None:
            raise ValueError("feature shard lacks its exact oracle index")
        records = int((row.get("stats") or {}).get("records_kept", -1))
        if records <= 0 or int(oracle_row.get("records", -1)) != records:
            raise ValueError("feature/oracle manifest record counts disagree")
        oracle = _VerifiedFile.open(
            oracle_path.parent / str(oracle_row.get("path") or ""),
            str(oracle_row.get("sha256") or ""),
        )
        pairs.append(
            _SourcePair(
                feature,
                oracle,
                records,
                classifier_digest,
                source_archive_digest,
            )
        )
    if oracle_rows:
        raise ValueError("oracle manifest contains unreferenced index shards")
    return pairs, feature_digest, oracle_digest, package_registry_digest


def _iter_pairs(pair: _SourcePair) -> Iterator[tuple[GameSequence, OracleEntry]]:
    pair.feature.assert_unchanged()
    sentinel = object()
    count = 0
    for sequence, entry in itertools.zip_longest(
        iter_feature_shard(pair.feature.path),
        _iter_oracle_index(pair),
        fillvalue=sentinel,
    ):
        if sequence is sentinel or entry is sentinel:
            raise ValueError("feature/oracle streams differ in length")
        assert isinstance(sequence, GameSequence)
        assert isinstance(entry, OracleEntry)
        raw_sequence_seat = getattr(sequence, "seat", None)
        if (
            str(sequence.episode_id) != entry.episode_id
            or type(raw_sequence_seat) is not int
            or raw_sequence_seat != entry.seat
            or normalize_matchup_identity(sequence.archetype)
            != entry.acting_archetype
            or normalize_matchup_identity(sequence.opp_archetype)
            != entry.opponent_archetype
        ):
            raise ValueError(
                "feature/oracle row alignment mismatch; refusing positional routing"
            )
        if entry.source_archive_digest != pair.source_archive_digest:
            raise ValueError("oracle row does not bind the feature shard source archive")
        count += 1
        yield sequence, entry
    if count != pair.records:
        raise ValueError(
            f"feature/oracle pair count changed: expected={pair.records} actual={count}"
        )
    pair.feature.assert_unchanged()


def _split_key(seed: int, route: int, episode_id: str) -> str:
    raw = f"{int(seed)}\0{int(route)}\0{episode_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _selection_route(
    entry: OracleEntry,
    *,
    excluded_ids: frozenset[str],
    excluded_digests: frozenset[str],
) -> Optional[int]:
    # Formal evaluation and ambiguous eligibility never enter any output, even
    # if their family itself is supported.
    if (
        entry.formal_eval
        or not entry.training_eligible
        or entry.opponent_id in excluded_ids
        or entry.opponent_content_digest in excluded_digests
    ):
        return None
    route = route_for_archetype(entry.opponent_archetype)
    return None if route == UNKNOWN_ROUTE else int(route)


def _initialize_membership_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE episodes (
            episode_id TEXT PRIMARY KEY,
            route INTEGER NOT NULL,
            split_key TEXT NOT NULL,
            split TEXT
        );
        CREATE TABLE records (
            episode_id TEXT NOT NULL,
            seat INTEGER NOT NULL,
            route INTEGER NOT NULL,
            PRIMARY KEY (episode_id, seat)
        );
        """
    )
    return connection


def _insert_membership(
    connection: sqlite3.Connection,
    *,
    entry: OracleEntry,
    route: int,
    seed: int,
) -> None:
    prior = connection.execute(
        "SELECT route FROM episodes WHERE episode_id = ?", (entry.episode_id,)
    ).fetchone()
    if prior is None:
        connection.execute(
            "INSERT INTO episodes(episode_id, route, split_key) VALUES (?, ?, ?)",
            (entry.episode_id, route, _split_key(seed, route, entry.episode_id)),
        )
    elif int(prior[0]) != int(route):
        raise ValueError("one episode appears in conflicting adapter routes")
    try:
        connection.execute(
            "INSERT INTO records(episode_id, seat, route) VALUES (?, ?, ?)",
            (entry.episode_id, entry.seat, route),
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError("duplicate feature episode/seat in adapter corpus") from exc


def _assign_splits(
    connection: sqlite3.Connection,
    *,
    val_frac: float,
) -> dict[int, dict[str, int]]:
    fraction = float(val_frac)
    if not 0.0 < fraction < 1.0:
        raise ValueError("adapter validation fraction must be in (0, 1)")
    result: dict[int, dict[str, int]] = {}
    for route, archetype_id in enumerate(EXPERT_IDS):
        episode_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM episodes WHERE route = ?", (route,)
            ).fetchone()[0]
        )
        if episode_count == 0:
            result[route] = {
                "episodes": 0,
                "train_episodes": 0,
                "val_episodes": 0,
            }
            continue
        val_count = (
            0
            if episode_count == 1
            else max(1, min(episode_count - 1, round(episode_count * fraction)))
        )
        connection.execute(
            """
            UPDATE episodes SET split = 'val' WHERE episode_id IN (
                SELECT episode_id FROM episodes
                WHERE route = ? ORDER BY split_key, episode_id LIMIT ?
            )
            """,
            (route, val_count),
        )
        connection.execute(
            "UPDATE episodes SET split = 'train' WHERE route = ? AND split IS NULL",
            (route,),
        )
        result[route] = {
            "episodes": episode_count,
            "train_episodes": episode_count - val_count,
            "val_episodes": val_count,
        }
    connection.commit()
    return result


def _membership_digest(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for episode_id, route, split in connection.execute(
        "SELECT episode_id, route, split FROM episodes "
        "ORDER BY route, split, episode_id"
    ):
        digest.update(
            _canonical_line(
                {
                    "episode_id": str(episode_id),
                    "route": int(route),
                    "split": str(split),
                }
            )
        )
    return _SHA256_PREFIX + digest.hexdigest()


@dataclass
class _OutputShard:
    route: int
    archetype_id: str
    split: str
    path: Path
    stream: Any
    sequences: int = 0
    decisions: int = 0

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        route: int,
        archetype_id: str,
        split: str,
        corpus_digest: str,
        gate_digest: str,
        classifier_digest: str,
        package_registry_digest: str,
        membership_digest: str,
    ) -> "_OutputShard":
        path = root / f"route_{route:02d}_{archetype_id}.{split}.features"
        stream = path.open("xb")
        header = {
            "format": SHARD_FORMAT,
            "format_version": SHARD_FORMAT_VERSION,
            "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
            "matchup_adapter_staged": True,
            "offline_oracle_only": True,
            "runtime_routes_enabled": False,
            "route": route,
            "archetype_id": archetype_id,
            "split": split,
            "corpus_manifest_digest": corpus_digest,
            "gate_contract_digest": gate_digest,
            "classifier_digest": classifier_digest,
            "package_registry_digest": package_registry_digest,
            "membership_digest": membership_digest,
        }
        pickle.dump(header, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return cls(route, archetype_id, split, path, stream)

    def append(self, sequence: GameSequence) -> None:
        pickle.dump(sequence, self.stream, protocol=pickle.HIGHEST_PROTOCOL)
        self.sequences += 1
        self.decisions += len(sequence.decisions)

    def finish(self) -> dict[str, Any]:
        stats = {
            "records_total": self.sequences,
            "records_kept": self.sequences,
            "records_dropped": 0,
            "decisions_kept": self.decisions,
        }
        pickle.dump(
            {
                "format": SHARD_FORMAT + "-footer",
                "format_version": SHARD_FORMAT_VERSION,
                "stats": stats,
            },
            self.stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        return {
            "path": self.path.name,
            "sha256": sha256_file(self.path),
            "bytes": self.path.stat().st_size,
            "route": self.route,
            "archetype_id": self.archetype_id,
            "split": self.split,
            "stats": stats,
        }


def _attach_ticket(
    sequence: GameSequence,
    entry: OracleEntry,
    *,
    route: int,
    corpus_digest: str,
    gate_digest: str,
    classifier_digest: str,
    package_registry_digest: str,
    source_feature_digest: str,
    oracle_index_digest: str,
    required_public_router_digest: str | None = None,
) -> None:
    sequence.opp_archetype = EXPERT_IDS[route]
    ticket = {
        "schema": TRAINING_TICKET_SCHEMA,
        "opponent_id": entry.opponent_id,
        "package_digest": entry.opponent_content_digest,
        "archetype_id": entry.opponent_archetype,
        "route": route,
        "corpus_manifest_digest": corpus_digest,
        "gate_contract_digest": gate_digest,
        "episode_id": entry.episode_id,
        "seat": entry.seat,
        # Additional immutable proof retained in the pickle/checksum.  The
        # common ticket parser consumes the routing core; this stager/loader
        # validates the full lineage extension.
        "identity_kind": "raw-archive-full-deck-and-package",
        "opponent_deck_digest": entry.opponent_deck_digest,
        "source_archive_digest": entry.source_archive_digest,
        "source_member_digest": entry.source_member_digest,
        "source_feature_digest": source_feature_digest,
        "oracle_index_digest": oracle_index_digest,
        "classifier_digest": classifier_digest,
        "package_registry_digest": package_registry_digest,
        "classifier_method": entry.classifier_method,
        "formal_eval": False,
        "runtime_route_authorized": False,
    }
    active_rows: set[int] | None = None
    if required_public_router_digest is not None:
        required = _require_digest(
            required_public_router_digest, "required public router digest"
        )
        if entry.public_router_digest != required:
            raise ValueError("oracle row is not bound to the required public router")
        if entry.public_decisions != len(sequence.decisions):
            raise ValueError(
                "public decision count does not align with the compact sequence: "
                f"episode_id={entry.episode_id!r} seat={entry.seat} "
                f"oracle={entry.public_decisions} compact={len(sequence.decisions)}"
            )
        active_rows = {
            index
            for start, end in entry.public_route_segments
            for index in range(start, end)
        }
        ticket.update(
            {
                "public_router_digest": required,
                "public_route_segments": [
                    [start, end] for start, end in entry.public_route_segments
                ],
                "public_confidence_threshold": entry.public_confidence_threshold,
                "public_consecutive_required": entry.public_consecutive_required,
                "oracle_route_rows": len(active_rows),
                "causal_public_alignment": True,
            }
        )
    sequence.matchup_adapter_training_ticket = ticket
    for index, decision in enumerate(sequence.decisions):
        decision.matchup_adapter_oracle_route = (
            route if active_rows is None or index in active_rows else UNKNOWN_ROUTE
        )
        decision.matchup_adapter_public_route = UNKNOWN_ROUTE
    # Exercise the same fail-closed parser used by training before serialization.
    parsed = adapter_training_ticket(sequence)
    if parsed.route != route or parsed.archetype_id != entry.opponent_archetype:
        raise AssertionError("constructed matchup-adapter ticket is inconsistent")


def stage_matchup_adapter_corpus(
    feature_manifest_path: Path,
    oracle_manifest_path: Path,
    package_registry_path: Path,
    gate_contract_path: Path,
    output_dir: Path,
    *,
    val_frac: float = 0.10,
    seed: int = 42,
    min_available_bytes: int = 8 * 1024**3,
    required_public_router_digest: str | None = None,
) -> Path:
    """Publish route-specific offline shards with bounded host memory.

    The function intentionally performs two complete verified passes.  Pass 1
    builds only the on-disk episode membership table; pass 2 writes one
    ``GameSequence`` at a time.  Existing output directories are never
    replaced.  Any failure removes only the new temporary directory.
    """

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    required_router = (
        _require_digest(required_public_router_digest, "required public router digest")
        if required_public_router_digest is not None
        else None
    )
    (
        pairs,
        feature_manifest_digest,
        corpus_digest,
        expected_package_registry_digest,
    ) = _open_sources(feature_manifest_path, oracle_manifest_path)
    package_identities, package_registry_digest = _package_registry(
        package_registry_path,
        expected_digest=expected_package_registry_digest,
    )
    gate_payload, gate_file_digest = _read_json(
        Path(gate_contract_path).expanduser().resolve()
    )
    gate = gate_exclusions(gate_payload)
    available = shutil.disk_usage(output.parent).free
    source_bytes = sum(pair.feature.identity[2] for pair in pairs)
    required = max(int(min_available_bytes), source_bytes + 1024**3)
    if available < required:
        raise RuntimeError(
            f"insufficient staging disk: available={available} required={required}"
        )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial.", dir=output.parent)
    )
    database_path = temporary / ".membership.sqlite3"
    connection = _initialize_membership_db(database_path)
    total_source = selected = excluded = unsupported = 0
    try:
        # Pass 1: exact alignment validation and disk-backed episode membership.
        with connection:
            for pair in pairs:
                for _sequence, entry in _iter_pairs(pair):
                    total_source += 1
                    if required_router is not None and entry.public_router_digest != required_router:
                        raise ValueError(
                            "oracle corpus is not uniformly bound to the required public router"
                        )
                    if required_router is not None and not entry.public_route_segments:
                        unsupported += 1
                        continue
                    registered_digest = package_identities.get(entry.opponent_id)
                    if registered_digest != entry.opponent_content_digest:
                        raise ValueError(
                            "oracle opponent ID/content digest is absent from or "
                            "contradicts the pinned package registry"
                        )
                    route = _selection_route(
                        entry,
                        excluded_ids=gate.opponent_ids,
                        excluded_digests=gate.content_digests,
                    )
                    if route is None:
                        if (
                            entry.formal_eval
                            or not entry.training_eligible
                            or entry.opponent_id in gate.opponent_ids
                            or entry.opponent_content_digest in gate.content_digests
                        ):
                            excluded += 1
                        else:
                            unsupported += 1
                        continue
                    _insert_membership(
                        connection, entry=entry, route=route, seed=int(seed)
                    )
                    selected += 1
        coverage = _assign_splits(connection, val_frac=val_frac)
        membership_digest = _membership_digest(connection)

        classifier_digest = pairs[0].classifier_digest
        if any(pair.classifier_digest != classifier_digest for pair in pairs):
            raise ValueError("oracle corpus mixes classifier contracts")
        writers: dict[tuple[int, str], _OutputShard] = {}
        for route, archetype_id in enumerate(EXPERT_IDS):
            for split in ("train", "val"):
                writers[(route, split)] = _OutputShard.open(
                    temporary,
                    route=route,
                    archetype_id=archetype_id,
                    split=split,
                    corpus_digest=corpus_digest,
                    gate_digest=gate.contract_digest,
                    classifier_digest=classifier_digest,
                    package_registry_digest=package_registry_digest,
                    membership_digest=membership_digest,
                )

        # Pass 2: revalidate source alignment, attach immutable tickets, and
        # immediately serialize.  A source object becomes unreachable after
        # each loop iteration.
        emitted = 0
        for pair in pairs:
            for sequence, entry in _iter_pairs(pair):
                row = connection.execute(
                    """
                    SELECT records.route, episodes.split
                    FROM records JOIN episodes USING (episode_id)
                    WHERE records.episode_id = ? AND records.seat = ?
                    """,
                    (entry.episode_id, entry.seat),
                ).fetchone()
                if row is None:
                    continue
                route, split = int(row[0]), str(row[1])
                if (
                    package_identities.get(entry.opponent_id)
                    != entry.opponent_content_digest
                ):
                    raise ValueError("package registry changed between staging passes")
                if route != route_for_archetype(entry.opponent_archetype):
                    raise ValueError("membership route changed between passes")
                _attach_ticket(
                    sequence,
                    entry,
                    route=route,
                    corpus_digest=corpus_digest,
                    gate_digest=gate.contract_digest,
                    classifier_digest=classifier_digest,
                    package_registry_digest=package_registry_digest,
                    source_feature_digest=pair.feature.digest,
                    oracle_index_digest=pair.oracle.digest,
                    required_public_router_digest=required_router,
                )
                writers[(route, split)].append(sequence)
                emitted += 1
        if emitted != selected:
            raise ValueError(
                f"staged sequence count changed: selected={selected} emitted={emitted}"
            )

        shards: list[dict[str, Any]] = []
        for key in sorted(writers):
            row = writers[key].finish()
            shards.append(row)
        for route, counts in coverage.items():
            for split in ("train", "val"):
                shard = writers[(route, split)]
                counts[f"{split}_sequences"] = shard.sequences
                counts[f"{split}_decisions"] = shard.decisions
            counts["status"] = (
                "ready" if counts["train_sequences"] > 0 else "dormant_no_examples"
            )

        manifest = {
            "schema": STAGED_CORPUS_SCHEMA,
            "offline_oracle_only": True,
            "runtime_routes_enabled": False,
            "public_router_digest": required_router,
            "causal_public_alignment": required_router is not None,
            "source_feature_manifest_digest": feature_manifest_digest,
            "oracle_manifest_digest": corpus_digest,
            "classifier_digest": classifier_digest,
            "package_registry_digest": package_registry_digest,
            "active_gate_contract_digest": gate.contract_digest,
            # Keep both identities. ``active_gate_contract_digest`` is the
            # canonical semantic contract used by every training ticket;
            # this is the exact source-file byte digest used to reproduce the
            # staging run and to fail a resumed adapter fit closed.
            "active_gate_contract_file_digest": gate_file_digest,
            "membership_digest": membership_digest,
            "split": {
                "algorithm": "sha256-ranked-per-route-episode/v1",
                "seed": int(seed),
                "val_frac": float(val_frac),
                "episode_disjoint": True,
            },
            "routes": [
                {
                    "route": route,
                    "archetype_id": EXPERT_IDS[route],
                    **coverage[route],
                }
                for route in range(len(EXPERT_IDS))
            ],
            "shards": shards,
            "totals": {
                "source_sequences": total_source,
                "selected_sequences": selected,
                "gate_or_ineligible_excluded": excluded,
                "unsupported_matchup_excluded": unsupported,
                "decisions": sum(row["stats"]["decisions_kept"] for row in shards),
            },
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        connection.close()
        database_path.unlink(missing_ok=True)
        Path(str(database_path) + "-wal").unlink(missing_ok=True)
        Path(str(database_path) + "-shm").unlink(missing_ok=True)
        _fsync_directory(temporary)
        os.replace(temporary, output)
        _fsync_directory(output.parent)
        return output / "manifest.json"
    except BaseException:
        for writer in locals().get("writers", {}).values():
            try:
                if not writer.stream.closed:
                    writer.stream.close()
            except Exception:
                pass
        try:
            connection.close()
        except Exception:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def iter_staged_split(
    manifest_path: Path,
    split: str,
) -> Iterator[GameSequence]:
    """Verify and stream one staged split without materializing it in RAM."""

    requested = str(split).strip().lower()
    if requested not in {"train", "val"}:
        raise ValueError("staged adapter split must be train or val")
    path = Path(manifest_path).expanduser().resolve()
    payload, corpus_file_digest = _read_json(path)
    if payload.get("schema") != STAGED_CORPUS_SCHEMA:
        raise ValueError("invalid staged matchup-adapter corpus schema")
    if (
        payload.get("offline_oracle_only") is not True
        or payload.get("runtime_routes_enabled") is not False
        or (payload.get("split") or {}).get("episode_disjoint") is not True
    ):
        raise ValueError("staged matchup-adapter corpus weakens its safety contract")
    corpus_digest = _require_digest(
        payload.get("oracle_manifest_digest"), "staged oracle manifest digest"
    )
    gate_digest = _require_digest(
        payload.get("active_gate_contract_digest"), "staged gate contract digest"
    )
    classifier_digest = _require_digest(
        payload.get("classifier_digest"), "staged classifier digest"
    )
    package_registry_digest = _require_digest(
        payload.get("package_registry_digest"), "staged package registry digest"
    )
    membership_digest = _require_digest(
        payload.get("membership_digest"), "staged membership digest"
    )
    del corpus_file_digest  # The caller pins this file digest in its checkpoint.

    selected = [
        dict(row)
        for row in payload.get("shards") or ()
        if str(row.get("split") or "") == requested
    ]
    if len(selected) != len(EXPERT_IDS):
        raise ValueError("staged split does not cover every configured route")
    seen_routes: set[int] = set()
    for row in sorted(selected, key=lambda item: int(item.get("route", -1))):
        route = int(row.get("route", -1))
        archetype_id = normalize_matchup_identity(row.get("archetype_id"))
        if (
            route in seen_routes
            or route < 0
            or route >= len(EXPERT_IDS)
            or EXPERT_IDS[route] != archetype_id
        ):
            raise ValueError("staged shard route identity is invalid")
        seen_routes.add(route)
        verified = _VerifiedFile.open(
            path.parent / str(row.get("path") or ""),
            str(row.get("sha256") or ""),
        )
        count = 0
        for sequence in iter_feature_shard(verified.path):
            ticket = dict(sequence.matchup_adapter_training_ticket or {})
            parsed = adapter_training_ticket(sequence)
            causal_alignment = ticket.get("causal_public_alignment") is True
            oracle_routes = [
                getattr(decision, "matchup_adapter_oracle_route", None)
                for decision in sequence.decisions
            ]
            decision_routes_are_exact = all(
                type(getattr(decision, "matchup_adapter_oracle_route", None))
                is int
                and type(getattr(decision, "matchup_adapter_public_route", None))
                is int
                for decision in sequence.decisions
            )
            if (
                parsed.route != route
                or parsed.archetype_id != archetype_id
                or parsed.corpus_manifest_digest != corpus_digest
                or parsed.gate_contract_digest != gate_digest
                or ticket.get("identity_kind")
                != "raw-archive-full-deck-and-package"
                or _require_digest(
                    ticket.get("classifier_digest"), "ticket classifier digest"
                )
                != classifier_digest
                or _require_digest(
                    ticket.get("package_registry_digest"),
                    "ticket package registry digest",
                )
                != package_registry_digest
                or not _require_digest(
                    ticket.get("opponent_deck_digest"), "ticket deck digest"
                )
                or not _require_digest(
                    ticket.get("source_archive_digest"),
                    "ticket source archive digest",
                )
                or not _require_digest(
                    ticket.get("source_member_digest"),
                    "ticket source member digest",
                )
                or not _require_digest(
                    ticket.get("source_feature_digest"),
                    "ticket source feature digest",
                )
                or not _require_digest(
                    ticket.get("oracle_index_digest"),
                    "ticket oracle index digest",
                )
                or str(ticket.get("classifier_method") or "")
                not in _CLASSIFIER_METHODS
                or ticket.get("formal_eval") is not False
                or ticket.get("runtime_route_authorized") is not False
                or not decision_routes_are_exact
                or any(
                    decision.matchup_adapter_public_route != UNKNOWN_ROUTE
                    for decision in sequence.decisions
                )
                or (
                    causal_alignment
                    and (
                        any(value not in (UNKNOWN_ROUTE, route) for value in oracle_routes)
                        or sum(value == route for value in oracle_routes)
                        != int(ticket.get("oracle_route_rows", -1))
                        or not _require_digest(
                            ticket.get("public_router_digest"),
                            "ticket public router digest",
                        )
                    )
                )
                or (
                    not causal_alignment
                    and any(value != route for value in oracle_routes)
                )
            ):
                raise ValueError("staged sequence lost its exact oracle-only contract")
            count += 1
            yield sequence
        if count != int((row.get("stats") or {}).get("records_kept", -1)):
            raise ValueError("staged shard record count changed")
        verified.assert_unchanged()
    if seen_routes != set(range(len(EXPERT_IDS))):
        raise ValueError("staged split route coverage is incomplete")
    # Retain an explicit read so accidental manifest field removal is caught.
    if not membership_digest:
        raise AssertionError("unreachable empty membership digest")


__all__ = [
    "ORACLE_INDEX_FOOTER_SCHEMA",
    "ORACLE_INDEX_HEADER_SCHEMA",
    "ORACLE_INDEX_ROW_SCHEMA",
    "ORACLE_MANIFEST_SCHEMA",
    "PACKAGE_REGISTRY_SCHEMA",
    "STAGED_CORPUS_SCHEMA",
    "OracleEntry",
    "full_deck_digest",
    "iter_staged_split",
    "sha256_file",
    "stage_matchup_adapter_corpus",
    "write_oracle_index",
]
