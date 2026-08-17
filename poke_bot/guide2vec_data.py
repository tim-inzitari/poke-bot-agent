"""Fail-closed data planning for the r212 Alakazam Guide2Vec experiment.

This module deliberately *does not* train a model or write an artifact.  It
opens the already protected, Alakazam-only expert feature corpus and produces
an in-memory, deterministic split manifest suitable for binding to a later
training receipt.

The split is deliberately stronger than the ordinary expert rehearsal split:

* calendar dates are fixed rather than shuffled;
* every acting-seat record for an ``episode_id`` stays in one partition; and
* exact 60-card deck multiset fingerprints are recorded per partition.

This is deliberately a same-deck Alakazam student, so a deck fingerprint is
allowed to occur in more than one date partition.  A later typed contract may
pass an explicit allowlist; absent that contract the fingerprints are telemetry,
not an exclusion rule.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import pickle
from types import MappingProxyType
from typing import Any, Collection, Iterator, Mapping

from .alakazam_heuristics import is_alakazam_deck
from .dataset import GameSequence
from .feature_shards import (
    COMPACT_MODE_TEMPORAL_EXPERT,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
    iter_feature_shard,
)


SPLIT_SCHEMA = "poke_bot.guide2vec_data_split_r212/v1"
POINTER_SCHEMA = "poke_bot.pinned_expert_corpus/v1"
SPECIALIST_ID = "alakazam"
PARTITION_ORDER = ("train", "validation", "test")


def _date_range(start: str, end: str) -> tuple[str, ...]:
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    values: list[str] = []
    current = first
    while current <= last:
        values.append(current.isoformat())
        current += timedelta(days=1)
    return tuple(values)


R212_PARTITION_DATES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "train": _date_range("2026-07-04", "2026-07-19"),
        "validation": _date_range("2026-07-20", "2026-07-21"),
        "test": _date_range("2026-07-22", "2026-07-23"),
    }
)
R212_SOURCE_DATES = tuple(
    day for partition in PARTITION_ORDER for day in R212_PARTITION_DATES[partition]
)
R212_DATE_TO_PARTITION: Mapping[str, str] = MappingProxyType(
    {
        day: partition
        for partition, days in R212_PARTITION_DATES.items()
        for day in days
    }
)

# The protected r212 source has eleven Alakazam-labelled acting-seat records
# whose list does not meet the current teacher's minimum Alakazam-line check.
# All were independently audited to have zero positive guide rows.  Keeping
# this list exact (rather than silently dropping every incompatible list) makes
# the exception content-addressed and fail-closed.
R212_QUARANTINE_IDENTITIES: tuple[tuple[str, str, int], ...] = (
    ("2026-07-17", "86571471", 1),
    ("2026-07-17", "86571997", 0),
    ("2026-07-17", "86572157", 0),
    ("2026-07-17", "86572717", 0),
    ("2026-07-17", "86573124", 1),
    ("2026-07-17", "86573658", 1),
    ("2026-07-17", "86574195", 1),
    ("2026-07-17", "86574328", 0),
    ("2026-07-17", "86574751", 1),
    ("2026-07-17", "86574899", 1),
    ("2026-07-17", "86575311", 1),
)
_R212_QUARANTINE_IDENTITY_SET = frozenset(R212_QUARANTINE_IDENTITIES)
R212_QUARANTINE_IDENTITIES_SHA256 = (
    "sha256:9794adc8844ff63c6f41ac696c473f32155e530ecb2ca411115ae9126bb7998d"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = Path(path).stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _assert_unchanged(
    path: Path,
    expected: tuple[int, int, int, int, int],
    *,
    label: str,
) -> None:
    if _stat_identity(path) != expected:
        raise ValueError(f"{label} changed during Guide2Vec split validation: {path}")


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before = _stat_identity(path)
    raw = path.read_bytes()
    _assert_unchanged(path, before, label=label)
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value, "sha256:" + hashlib.sha256(raw).hexdigest()


def _is_below(child: Path, root: Path) -> bool:
    try:
        child.relative_to(root)
    except ValueError:
        return False
    return True


def _deck_card_ids(deck: Any) -> tuple[int, ...]:
    """Return the exact deck multiset from list or compact array storage."""

    if isinstance(deck, (str, bytes)):
        raise ValueError("Guide2Vec source row does not carry an exact 60-card deck")
    try:
        cards = tuple(int(value) for value in deck)
    except (TypeError, ValueError) as exc:
        raise ValueError("Guide2Vec source deck has a non-integer card id") from exc
    if len(cards) != 60:
        raise ValueError("Guide2Vec source row does not carry an exact 60-card deck")
    return cards


def _deck_fingerprint(cards: tuple[int, ...]) -> str:
    return _canonical_digest(sorted(cards))


def _guide_rows(sequence: GameSequence) -> int:
    """Validate guide-label alignment and return labelled factorized stages."""

    labelled = 0
    for decision in sequence.decisions:
        stages = list(getattr(decision, "policy_stages", ()) or ())
        if not stages:
            raise ValueError(
                f"Guide2Vec source decision has no factorized policy stage: "
                f"episode={sequence.episode_id!r}"
            )
        for stage in stages:
            try:
                option_count = int(stage.options.num_words)
                target = int(stage.guide_target_index)
                confidence = float(stage.guide_confidence)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Guide2Vec source stage lacks an aligned guide label"
                ) from exc
            if option_count < 1:
                raise ValueError("Guide2Vec source stage has no legal option rows")
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("Guide2Vec guide confidence is not finite in [0, 1]")
            if target == -1:
                if confidence != 0.0:
                    raise ValueError(
                        "masked Guide2Vec guide target carries nonzero confidence"
                    )
                continue
            if target < 0 or target >= option_count:
                raise ValueError("Guide2Vec guide target is outside its legal option stage")
            if option_count < 2 or confidence <= 0.0:
                raise ValueError(
                    "Guide2Vec non-masked guide target is not a comparable legal stage"
                )
            labelled += 1
    return labelled


def _read_shard_header(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("rb") as handle:
            value = pickle.load(handle)
    except (EOFError, OSError, pickle.UnpicklingError) as exc:
        raise ValueError(f"invalid Guide2Vec feature shard header: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Guide2Vec feature shard header is not an object: {path}")
    return value


@dataclass(frozen=True)
class _VerifiedShard:
    path: Path
    digest: str
    stat_identity: tuple[int, int, int, int, int]
    source_date: str
    partition: str
    source_records: int
    source_decisions: int
    source_guide_rows: int
    retained_records: int
    retained_decisions: int
    retained_guide_rows: int
    quarantined_records: int
    quarantined_decisions: int
    quarantined_guide_rows: int

    def assert_unchanged(self) -> None:
        _assert_unchanged(self.path, self.stat_identity, label="feature shard")

    def verify_digest(self) -> None:
        self.assert_unchanged()
        actual = _sha256(self.path)
        self.assert_unchanged()
        if actual != self.digest:
            raise ValueError(
                "Guide2Vec feature shard digest mismatch: "
                f"expected={self.digest} actual={actual} path={self.path}"
            )


@dataclass(frozen=True)
class _QuarantinedRecord:
    source_date: str
    episode_id: str
    seat: int
    decisions: int
    guide_rows: int
    deck_fingerprint: str

    @property
    def identity(self) -> tuple[str, str, int]:
        return (self.source_date, self.episode_id, self.seat)

    def identity_manifest(self) -> dict[str, Any]:
        return {
            "source_date": self.source_date,
            "episode_id": self.episode_id,
            "seat": self.seat,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            **self.identity_manifest(),
            "decisions": self.decisions,
            "guide_rows": self.guide_rows,
            "deck_fingerprint": self.deck_fingerprint,
        }


@dataclass(frozen=True)
class Guide2VecDataSplit:
    """Validated, reiterable view of the fixed r212 source partitions."""

    protected_pointer_path: Path
    protected_pointer_sha256: str
    manifest_path: Path
    manifest_sha256: str
    _shards: tuple[_VerifiedShard, ...]
    _partition_stats: Mapping[str, Mapping[str, Any]]
    _deck_cross_partition_fingerprints: tuple[str, ...]
    _authorized_deck_fingerprints: tuple[str, ...] | None
    _quarantined_records: tuple[_QuarantinedRecord, ...]

    @property
    def partitions(self) -> tuple[str, ...]:
        return PARTITION_ORDER

    def iter_partition(self, partition: str) -> Iterator[GameSequence]:
        """Re-open one immutable partition after re-checking every shard."""

        name = str(partition)
        if name not in PARTITION_ORDER:
            raise ValueError(f"unknown Guide2Vec partition: {partition!r}")
        quarantined = {record.identity for record in self._quarantined_records}
        for shard in self._shards:
            if shard.partition != name:
                continue
            shard.verify_digest()
            source_emitted = 0
            emitted = 0
            skipped = 0
            for sequence in iter_feature_shard(shard.path):
                if str(sequence.archetype).strip().casefold() != SPECIALIST_ID:
                    raise ValueError(
                        "Guide2Vec partition source archetype changed after validation"
                    )
                source_emitted += 1
                identity = (
                    shard.source_date,
                    str(sequence.episode_id or "").strip(),
                    int(sequence.seat),
                )
                if identity in quarantined:
                    skipped += 1
                    continue
                emitted += 1
                yield sequence
            if source_emitted != shard.source_records:
                raise ValueError(
                    "Guide2Vec partition source record count changed after validation: "
                    f"expected={shard.source_records} actual={source_emitted} "
                    f"path={shard.path}"
                )
            if emitted != shard.retained_records or skipped != shard.quarantined_records:
                raise ValueError(
                    "Guide2Vec partition retained/quarantine record accounting changed: "
                    f"expected_retained={shard.retained_records} actual_retained={emitted} "
                    f"expected_quarantined={shard.quarantined_records} "
                    f"actual_quarantined={skipped} path={shard.path}"
                )
            shard.assert_unchanged()

    def manifest(self) -> dict[str, Any]:
        """Return the receipt-ready split description without writing it."""

        partitions: dict[str, dict[str, Any]] = {}
        for name in PARTITION_ORDER:
            stats = dict(self._partition_stats[name])
            source_shards = [
                {
                    "source_date": shard.source_date,
                    "path": str(shard.path),
                    "sha256": shard.digest,
                    "source_records": shard.source_records,
                    "source_decisions": shard.source_decisions,
                    "source_guide_rows": shard.source_guide_rows,
                    "retained_records": shard.retained_records,
                    "retained_decisions": shard.retained_decisions,
                    "retained_guide_rows": shard.retained_guide_rows,
                    "quarantined_records": shard.quarantined_records,
                    "quarantined_decisions": shard.quarantined_decisions,
                    "quarantined_guide_rows": shard.quarantined_guide_rows,
                }
                for shard in self._shards
                if shard.partition == name
            ]
            partitions[name] = {
                "dates": list(R212_PARTITION_DATES[name]),
                **stats,
                "source_shards": source_shards,
            }
        source_accounting = {
            "records": sum(
                int(self._partition_stats[name]["source_records"])
                for name in PARTITION_ORDER
            ),
            "decisions": sum(
                int(self._partition_stats[name]["source_decisions"])
                for name in PARTITION_ORDER
            ),
            "guide_rows": sum(
                int(self._partition_stats[name]["source_guide_rows"])
                for name in PARTITION_ORDER
            ),
        }
        retained_accounting = {
            "records": sum(
                int(self._partition_stats[name]["records"])
                for name in PARTITION_ORDER
            ),
            "decisions": sum(
                int(self._partition_stats[name]["decisions"])
                for name in PARTITION_ORDER
            ),
            "guide_rows": sum(
                int(self._partition_stats[name]["guide_rows"])
                for name in PARTITION_ORDER
            ),
        }
        quarantined_accounting = {
            "records": sum(
                int(self._partition_stats[name]["quarantined_records"])
                for name in PARTITION_ORDER
            ),
            "decisions": sum(
                int(self._partition_stats[name]["quarantined_decisions"])
                for name in PARTITION_ORDER
            ),
            "guide_rows": sum(
                int(self._partition_stats[name]["quarantined_guide_rows"])
                for name in PARTITION_ORDER
            ),
        }
        quarantine_identities = [
            record.identity_manifest() for record in self._quarantined_records
        ]
        return {
            "schema": SPLIT_SCHEMA,
            "specialist_id": SPECIALIST_ID,
            "source": {
                "protected_pointer": str(self.protected_pointer_path),
                "protected_pointer_sha256": self.protected_pointer_sha256,
                "manifest": str(self.manifest_path),
                "manifest_sha256": self.manifest_sha256,
                "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
            },
            "partition_policy": {
                "kind": "fixed_calendar_day_whole_episode_deck_fingerprint",
                "dates": {
                    name: list(R212_PARTITION_DATES[name]) for name in PARTITION_ORDER
                },
                "whole_episode_grouping_required": True,
                "whole_source_day_grouping_required": True,
                "deck_fingerprint": "sha256(canonical_sorted_60_card_multiset)",
                "cross_partition_episode_overlap_policy": "fail_closed",
                "cross_partition_deck_fingerprint_policy": "record_only",
                "authorized_deck_fingerprints": (
                    list(self._authorized_deck_fingerprints)
                    if self._authorized_deck_fingerprints is not None
                    else None
                ),
            },
            "overlap_checks": {
                "whole_episode_groups_partition_disjoint": True,
                "exact_deck_multisets_partition_disjoint": bool(
                    not self._deck_cross_partition_fingerprints
                ),
                "cross_partition_deck_fingerprints": list(
                    self._deck_cross_partition_fingerprints
                ),
                "cross_partition_deck_fingerprint_count": len(
                    self._deck_cross_partition_fingerprints
                ),
                "cross_partition_deck_fingerprint_policy": "record_only",
                "passed": True,
            },
            "accounting": {
                "source": source_accounting,
                "retained": retained_accounting,
                "quarantined": quarantined_accounting,
            },
            "quarantine": {
                "policy": "fail_closed_known_zero_guide_teacher_incompatible_rows_only",
                "expected_identity_count": len(R212_QUARANTINE_IDENTITIES),
                "identity_count": len(quarantine_identities),
                "expected_identities_sha256": R212_QUARANTINE_IDENTITIES_SHA256,
                "identities_sha256": _canonical_digest(quarantine_identities),
                "identities": quarantine_identities,
                "records": [record.manifest() for record in self._quarantined_records],
            },
            "partitions": partitions,
        }


def _resolve_protected_manifest(
    protected_pointer: Path,
) -> tuple[Path, str, Path, str, dict[str, Any], dict[str, Any]]:
    pointer_path = Path(protected_pointer).expanduser().resolve()
    pointer, pointer_digest = _read_json_object(
        pointer_path,
        label="Guide2Vec protected corpus pointer",
    )
    if pointer.get("schema") != POINTER_SCHEMA or pointer.get("protected") is not True:
        raise ValueError("Guide2Vec source is not a protected expert corpus pointer")
    raw_manifest = str(pointer.get("manifest") or "").strip()
    if not raw_manifest:
        raise ValueError("Guide2Vec protected pointer has no manifest path")
    relative_manifest = Path(raw_manifest)
    if relative_manifest.is_absolute():
        raise ValueError("Guide2Vec protected pointer manifest must be relative")
    manifest_path = (pointer_path.parent / relative_manifest).resolve()
    if not _is_below(manifest_path, pointer_path.parent):
        raise ValueError("Guide2Vec protected pointer manifest escapes its corpus root")
    manifest, manifest_digest = _read_json_object(
        manifest_path,
        label="Guide2Vec feature manifest",
    )
    expected = str(pointer.get("manifest_sha256") or "")
    if manifest_digest != expected:
        raise ValueError(
            "Guide2Vec protected pointer manifest digest mismatch: "
            f"expected={expected} actual={manifest_digest}"
        )
    return (
        pointer_path,
        pointer_digest,
        manifest_path,
        manifest_digest,
        pointer,
        manifest,
    )


def _validate_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("Guide2Vec source manifest format mismatch")
    if int(manifest.get("format_version", -1)) != MANIFEST_FORMAT_VERSION:
        raise ValueError("Guide2Vec source manifest version mismatch")
    if manifest.get("compact_mode") != COMPACT_MODE_TEMPORAL_EXPERT:
        raise ValueError("Guide2Vec source must use temporal-expert-v1 compaction")
    dates = tuple(str(value) for value in manifest.get("dates") or ())
    if dates != R212_SOURCE_DATES:
        raise ValueError(
            "Guide2Vec source manifest dates do not match the fixed r212 window"
        )
    if manifest.get("date_start") != R212_SOURCE_DATES[0] or manifest.get(
        "date_end"
    ) != R212_SOURCE_DATES[-1]:
        raise ValueError("Guide2Vec source manifest date bounds do not match r212")
    selection = dict(manifest.get("selection") or {})
    if (
        str(selection.get("value") or "").strip().casefold() != SPECIALIST_ID
        or selection.get("seat_semantics") != "acting_seat_only"
        or selection.get("opponent_routes_only") is not False
    ):
        raise ValueError("Guide2Vec source manifest is not Alakazam acting-seat only")
    quality = dict(manifest.get("quality_gates") or {})
    required_quality = (
        "passed",
        "checksummed",
        "acting_seat_archetype_exact",
        "hidden_targets_are_aux_only",
        "temporal_action_tokens_complete",
    )
    if any(quality.get(name) is not True for name in required_quality):
        raise ValueError("Guide2Vec source manifest quality gates are incomplete")
    rows = list(manifest.get("shards") or ())
    if len(rows) != len(R212_SOURCE_DATES) or not all(
        isinstance(row, dict) for row in rows
    ):
        raise ValueError("Guide2Vec source must contain one checked shard per r212 day")
    return [dict(row) for row in rows]


def _validate_header(
    header: Mapping[str, Any],
    *,
    path: Path,
    source_date: str,
) -> None:
    if header.get("format") != SHARD_FORMAT or int(
        header.get("format_version", -1)
    ) != SHARD_FORMAT_VERSION:
        raise ValueError(f"Guide2Vec shard format mismatch: {path}")
    if header.get("compact_mode") != COMPACT_MODE_TEMPORAL_EXPERT:
        raise ValueError(f"Guide2Vec shard compact mode mismatch: {path}")
    source_dates = tuple(str(value) for value in header.get("source_dates") or ())
    if source_dates != (source_date,):
        raise ValueError(
            "Guide2Vec shard must bind exactly one matching source day: "
            f"path={path} expected={source_date} actual={list(source_dates)}"
        )
    declared = str(header.get("required_archetype") or "").strip().casefold()
    if declared != SPECIALIST_ID:
        raise ValueError(f"Guide2Vec shard required archetype mismatch: {path}")


def _shard_path(manifest_path: Path, row: Mapping[str, Any]) -> Path:
    raw = str(row.get("path") or "").strip()
    if not raw:
        raise ValueError("Guide2Vec manifest shard has no path")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError("Guide2Vec manifest shard path must be relative")
    path = (manifest_path.parent / candidate).resolve()
    if not _is_below(path, manifest_path.parent):
        raise ValueError("Guide2Vec manifest shard path escapes the corpus root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _shard_expected_int(row: Mapping[str, Any], key: str) -> int:
    stats = dict(row.get("stats") or {})
    try:
        return int(stats[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Guide2Vec manifest shard lacks stats.{key}") from exc


def _shard_expected_guide_rows(row: Mapping[str, Any]) -> int:
    stats = dict(row.get("stats") or {})
    coverage = dict(stats.get("target_coverage") or {})
    try:
        return int(coverage["guide_rows"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Guide2Vec manifest shard lacks target_coverage.guide_rows") from exc


def _normalize_authorized_deck_fingerprints(
    fingerprints: Collection[str] | None,
) -> tuple[str, ...] | None:
    """Validate an optional contract-owned deck identity allowlist.

    The r212 data plan itself has no deck exclusion policy. This optional
    allowlist lets a later signed contract constrain accepted deck identities
    without treating same-list lineage across source days as leakage.
    """

    if fingerprints is None:
        return None
    if isinstance(fingerprints, (str, bytes)):
        raise ValueError(
            "authorized Guide2Vec deck fingerprints must be a collection, not a string"
        )
    normalized: set[str] = set()
    for fingerprint in fingerprints:
        value = str(fingerprint).strip().casefold()
        if (
            len(value) != 71
            or not value.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in value[7:])
        ):
            raise ValueError(
                "authorized Guide2Vec deck fingerprint is not a SHA-256 digest"
            )
        normalized.add(value)
    if not normalized:
        raise ValueError("authorized Guide2Vec deck fingerprint allowlist is empty")
    return tuple(sorted(normalized))


def build_r212_guide2vec_split(
    protected_pointer: Path,
    *,
    authorized_deck_fingerprints: Collection[str] | None = None,
) -> Guide2VecDataSplit:
    """Validate and plan the fixed r212 Guide2Vec train/validation/test split.

    The full source is streamed once.  Every pointer, manifest, and shard
    digest is checked before use, then shard identities are checked again after
    scanning. No source file is written or rewritten. Deck reuse across date
    partitions is recorded; an optional future contract may restrict deck
    identities with ``authorized_deck_fingerprints``.
    """

    (
        pointer_path,
        pointer_digest,
        manifest_path,
        manifest_digest,
        pointer,
        manifest,
    ) = _resolve_protected_manifest(protected_pointer)
    rows = _validate_manifest(manifest)

    authorized_decks = _normalize_authorized_deck_fingerprints(
        authorized_deck_fingerprints
    )
    seen_dates: set[str] = set()
    seen_paths: set[Path] = set()
    shards: list[_VerifiedShard] = []
    episode_partition: dict[str, str] = {}
    deck_partitions: dict[str, set[str]] = defaultdict(set)
    partition_episodes: dict[str, set[str]] = defaultdict(set)
    partition_decks: dict[str, set[str]] = defaultdict(set)
    source_partition_counts: dict[str, dict[str, int]] = {
        name: {"records": 0, "decisions": 0, "guide_rows": 0}
        for name in PARTITION_ORDER
    }
    retained_partition_counts: dict[str, dict[str, int]] = {
        name: {"records": 0, "decisions": 0, "guide_rows": 0}
        for name in PARTITION_ORDER
    }
    quarantined_partition_counts: dict[str, dict[str, int]] = {
        name: {"records": 0, "decisions": 0, "guide_rows": 0}
        for name in PARTITION_ORDER
    }
    source_aggregate = {"records": 0, "decisions": 0, "guide_rows": 0}
    retained_aggregate = {"records": 0, "decisions": 0, "guide_rows": 0}
    quarantined_aggregate = {"records": 0, "decisions": 0, "guide_rows": 0}
    quarantined_records: list[_QuarantinedRecord] = []
    seen_quarantine_identities: set[tuple[str, str, int]] = set()

    for row in rows:
        path = _shard_path(manifest_path, row)
        if path in seen_paths:
            raise ValueError(f"Guide2Vec source manifest repeats shard: {path}")
        seen_paths.add(path)
        expected_digest = str(row.get("sha256") or "")
        if not expected_digest.startswith("sha256:"):
            raise ValueError(f"Guide2Vec manifest shard lacks a SHA-256 digest: {path}")
        before = _stat_identity(path)
        actual_digest = _sha256(path)
        _assert_unchanged(path, before, label="feature shard")
        if actual_digest != expected_digest:
            raise ValueError(
                "Guide2Vec feature shard digest mismatch: "
                f"expected={expected_digest} actual={actual_digest} path={path}"
            )
        if "bytes" in row and int(row["bytes"]) != int(before[2]):
            raise ValueError(f"Guide2Vec manifest shard byte count mismatch: {path}")
        header = _read_shard_header(path)
        source_dates = tuple(str(value) for value in header.get("source_dates") or ())
        if len(source_dates) != 1:
            raise ValueError(f"Guide2Vec shard must declare one source day: {path}")
        source_date = source_dates[0]
        if source_date not in R212_DATE_TO_PARTITION:
            raise ValueError(f"Guide2Vec shard has an out-of-window source day: {path}")
        if source_date in seen_dates:
            raise ValueError(f"Guide2Vec source has multiple shards for {source_date}")
        seen_dates.add(source_date)
        _validate_header(header, path=path, source_date=source_date)
        partition = R212_DATE_TO_PARTITION[source_date]
        expected_records = _shard_expected_int(row, "records_kept")
        expected_decisions = _shard_expected_int(row, "decisions_kept")
        expected_guide_rows = _shard_expected_guide_rows(row)
        if min(expected_records, expected_decisions, expected_guide_rows) <= 0:
            raise ValueError(f"Guide2Vec manifest shard has an empty required count: {path}")

        source_records = 0
        source_decisions = 0
        source_guide_rows = 0
        retained_records = 0
        retained_decisions = 0
        retained_guide_rows = 0
        quarantined_records_count = 0
        quarantined_decisions = 0
        quarantined_guide_rows = 0
        for sequence in iter_feature_shard(path):
            if str(sequence.archetype).strip().casefold() != SPECIALIST_ID:
                raise ValueError(
                    "Guide2Vec source contains a non-Alakazam acting-seat record: "
                    f"episode={sequence.episode_id!r} path={path}"
                )
            if sequence.info_set_ok is not True:
                raise ValueError(
                    f"Guide2Vec source contains an invalid information set: {path}"
                )
            try:
                seat = int(sequence.seat)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Guide2Vec source has an invalid acting seat: {path}") from exc
            if seat not in (0, 1):
                raise ValueError(f"Guide2Vec source has an invalid acting seat: {path}")
            episode_id = str(sequence.episode_id or "").strip()
            if not episode_id:
                raise ValueError("Guide2Vec source has no whole-game episode identity")
            previous_episode_partition = episode_partition.setdefault(
                episode_id,
                partition,
            )
            if previous_episode_partition != partition:
                raise ValueError(
                    "Guide2Vec episode crosses date partitions: "
                    f"episode={episode_id!r} first={previous_episode_partition} "
                    f"second={partition}"
                )
            row_decisions = len(sequence.decisions)
            row_guide_rows = _guide_rows(sequence)
            deck_cards = _deck_card_ids(sequence.deck)
            fingerprint = _deck_fingerprint(deck_cards)

            source_records += 1
            source_decisions += row_decisions
            source_guide_rows += row_guide_rows

            if not is_alakazam_deck(deck_cards):
                identity = (source_date, episode_id, seat)
                if row_guide_rows != 0:
                    raise ValueError(
                        "Guide2Vec source deck is incompatible with the Alakazam guide "
                        "teacher and carries positive guide labels: "
                        f"source_date={source_date} episode={episode_id!r} seat={seat} "
                        f"guide_rows={row_guide_rows} path={path}"
                    )
                if identity not in _R212_QUARANTINE_IDENTITY_SET:
                    raise ValueError(
                        "Guide2Vec source has an unexpected zero-guide teacher-incompatible "
                        "record outside the r212 quarantine: "
                        f"source_date={source_date} episode={episode_id!r} seat={seat} "
                        f"path={path}"
                    )
                if identity in seen_quarantine_identities:
                    raise ValueError(
                        "Guide2Vec source repeats an r212 quarantine identity: "
                        f"source_date={source_date} episode={episode_id!r} seat={seat}"
                    )
                seen_quarantine_identities.add(identity)
                quarantined_records.append(
                    _QuarantinedRecord(
                        source_date=source_date,
                        episode_id=episode_id,
                        seat=seat,
                        decisions=row_decisions,
                        guide_rows=row_guide_rows,
                        deck_fingerprint=fingerprint,
                    )
                )
                quarantined_records_count += 1
                quarantined_decisions += row_decisions
                quarantined_guide_rows += row_guide_rows
                continue
            if (source_date, episode_id, seat) in _R212_QUARANTINE_IDENTITY_SET:
                raise ValueError(
                    "Guide2Vec r212 quarantine identity became teacher-compatible: "
                    f"source_date={source_date} episode={episode_id!r} seat={seat} "
                    f"path={path}"
                )
            if authorized_decks is not None and fingerprint not in authorized_decks:
                raise ValueError(
                    "Guide2Vec source deck fingerprint is not authorized by the "
                    f"active contract: fingerprint={fingerprint} partition={partition}"
                )
            deck_partitions[fingerprint].add(partition)
            retained_records += 1
            retained_decisions += row_decisions
            retained_guide_rows += row_guide_rows
            partition_episodes[partition].add(episode_id)
            partition_decks[partition].add(fingerprint)

        _assert_unchanged(path, before, label="feature shard")
        if (source_records, source_decisions, source_guide_rows) != (
            expected_records,
            expected_decisions,
            expected_guide_rows,
        ):
            raise ValueError(
                "Guide2Vec source shard counts disagree with the protected manifest: "
                f"path={path} expected={(expected_records, expected_decisions, expected_guide_rows)} "
                f"actual={(source_records, source_decisions, source_guide_rows)}"
            )
        shards.append(
            _VerifiedShard(
                path=path,
                digest=expected_digest,
                stat_identity=before,
                source_date=source_date,
                partition=partition,
                source_records=source_records,
                source_decisions=source_decisions,
                source_guide_rows=source_guide_rows,
                retained_records=retained_records,
                retained_decisions=retained_decisions,
                retained_guide_rows=retained_guide_rows,
                quarantined_records=quarantined_records_count,
                quarantined_decisions=quarantined_decisions,
                quarantined_guide_rows=quarantined_guide_rows,
            )
        )
        for key, value in (
            ("records", source_records),
            ("decisions", source_decisions),
            ("guide_rows", source_guide_rows),
        ):
            source_partition_counts[partition][key] += value
            source_aggregate[key] += value
        for key, value in (
            ("records", retained_records),
            ("decisions", retained_decisions),
            ("guide_rows", retained_guide_rows),
        ):
            retained_partition_counts[partition][key] += value
            retained_aggregate[key] += value
        for key, value in (
            ("records", quarantined_records_count),
            ("decisions", quarantined_decisions),
            ("guide_rows", quarantined_guide_rows),
        ):
            quarantined_partition_counts[partition][key] += value
            quarantined_aggregate[key] += value

    if seen_dates != set(R212_SOURCE_DATES):
        missing = sorted(set(R212_SOURCE_DATES) - seen_dates)
        extra = sorted(seen_dates - set(R212_SOURCE_DATES))
        raise ValueError(
            "Guide2Vec source day set does not match r212: "
            f"missing={missing} extra={extra}"
        )
    if seen_quarantine_identities != _R212_QUARANTINE_IDENTITY_SET:
        missing = sorted(_R212_QUARANTINE_IDENTITY_SET - seen_quarantine_identities)
        extra = sorted(seen_quarantine_identities - _R212_QUARANTINE_IDENTITY_SET)
        raise ValueError(
            "Guide2Vec r212 quarantine identity set changed: "
            f"missing={missing} extra={extra}"
        )
    frozen_quarantined_records = tuple(
        sorted(quarantined_records, key=lambda record: record.identity)
    )
    quarantine_identity_digest = _canonical_digest(
        [record.identity_manifest() for record in frozen_quarantined_records]
    )
    if quarantine_identity_digest != R212_QUARANTINE_IDENTITIES_SHA256:
        raise ValueError(
            "Guide2Vec r212 quarantine identity digest mismatch: "
            f"expected={R212_QUARANTINE_IDENTITIES_SHA256} "
            f"actual={quarantine_identity_digest}"
        )
    for name in PARTITION_ORDER:
        if not partition_episodes[name] or not partition_decks[name]:
            raise ValueError(f"Guide2Vec {name} partition is empty")
    totals = dict(manifest.get("totals") or {})
    expected_totals = {
        "records": int(totals.get("records_kept") or 0),
        "decisions": int(totals.get("decisions_kept") or 0),
        "guide_rows": int(
            dict(totals.get("target_coverage") or {}).get("guide_rows") or 0
        ),
    }
    if source_aggregate != expected_totals:
        raise ValueError(
            "Guide2Vec aggregate counts disagree with the protected manifest: "
            f"expected={expected_totals} actual={source_aggregate}"
        )
    pointer_totals = dict(pointer.get("totals") or {})
    for key, manifest_key in (("records", "records_kept"), ("decisions", "decisions_kept")):
        if (
            manifest_key in pointer_totals
            and int(pointer_totals[manifest_key]) != source_aggregate[key]
        ):
            raise ValueError("Guide2Vec protected pointer totals disagree with manifest")
    for key in ("records", "decisions", "guide_rows"):
        if source_aggregate[key] != retained_aggregate[key] + quarantined_aggregate[key]:
            raise ValueError(
                "Guide2Vec retained/quarantined accounting does not reconstruct "
                f"the protected source for {key}: source={source_aggregate[key]} "
                f"retained={retained_aggregate[key]} "
                f"quarantined={quarantined_aggregate[key]}"
            )

    frozen_partition_stats = {
        name: {
            **retained_partition_counts[name],
            "source_records": source_partition_counts[name]["records"],
            "source_decisions": source_partition_counts[name]["decisions"],
            "source_guide_rows": source_partition_counts[name]["guide_rows"],
            "quarantined_records": quarantined_partition_counts[name]["records"],
            "quarantined_decisions": quarantined_partition_counts[name]["decisions"],
            "quarantined_guide_rows": quarantined_partition_counts[name]["guide_rows"],
            "whole_game_groups": len(partition_episodes[name]),
            "exact_deck_multisets": len(partition_decks[name]),
            "episode_ids_sha256": _canonical_digest(sorted(partition_episodes[name])),
            "source_days_sha256": _canonical_digest(R212_PARTITION_DATES[name]),
            "deck_fingerprints": sorted(partition_decks[name]),
            "deck_fingerprints_sha256": _canonical_digest(sorted(partition_decks[name])),
        }
        for name in PARTITION_ORDER
    }
    cross_partition_decks = tuple(
        sorted(
            fingerprint
            for fingerprint, partitions in deck_partitions.items()
            if len(partitions) > 1
        )
    )
    return Guide2VecDataSplit(
        protected_pointer_path=pointer_path,
        protected_pointer_sha256=pointer_digest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_digest,
        _shards=tuple(sorted(shards, key=lambda row: R212_SOURCE_DATES.index(row.source_date))),
        _partition_stats=frozen_partition_stats,
        _deck_cross_partition_fingerprints=cross_partition_decks,
        _authorized_deck_fingerprints=authorized_decks,
        _quarantined_records=frozen_quarantined_records,
    )


__all__ = [
    "Guide2VecDataSplit",
    "PARTITION_ORDER",
    "R212_DATE_TO_PARTITION",
    "R212_PARTITION_DATES",
    "R212_QUARANTINE_IDENTITIES",
    "R212_QUARANTINE_IDENTITIES_SHA256",
    "R212_SOURCE_DATES",
    "SPECIALIST_ID",
    "SPLIT_SCHEMA",
    "build_r212_guide2vec_split",
]
