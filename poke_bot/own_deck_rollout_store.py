"""Immutable r259 OwnDeckLedger expert-rollout side store.

This module is deliberately a *side-store builder*.  It never imports a
trainer, changes a selector, writes into a replay corpus, or makes a row
training-eligible.  Its only writable target is a new immutable daily sidecar
directory.  Every accepted input is bound to the sealed r241 archive receipt,
and every listed ZIP is re-hashed before any daily output is published.

The builder supports two equally strict ingestion modes:

``archive_native``
    Read the exact manifest-listed ZIP files directly.  This is the Elmo
    production route; it avoids the generic collector's mutable index/raw
    directories.  It uses the pinned ladder classifier only to retain the
    acting Alakazam seats.

``protected_jsonl``
    Read a protected, already-masked replay-record JSONL(/.gz) stream.  This
    is useful for reproducibility and parity checks.  The archive receipt is
    still checked and every ZIP is still re-hashed before any output is made.

Rows intentionally contain no raw observation, action, source deck list,
opponent/private labels, or transition snapshot.  They retain only public
fingerprints, the immutable public ledger snapshot, target-only supervision,
and typed option/board join fingerprints required by a later, separately
authorized successor dataset join.
"""

from __future__ import annotations

import copy
import fcntl
import gzip
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .own_deck_ledger import OPTION_FEATURE_DIM, OwnDeckLedger
from .own_deck_supervision import (
    OWN_DECK_SUPERVISION_SCHEMA,
    OWN_DECK_SUPERVISION_VERSION,
    TERMINAL_CONVERSION_CLASSES,
    TERMINAL_CONVERSION_OUTPUT_DIM,
    VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM,
    VISIBLE_TUTOR_COMPLETION_SCALAR_TARGET_NAMES,
    build_own_deck_supervision_targets,
    terminal_conversion_target_mask,
    terminal_conversion_target_vector,
    visible_tutor_completion_target_mask,
    visible_tutor_completion_target_vector,
)

OWN_DECK_ROLLOUT_SIDECAR_SCHEMA = "poke_bot.own_deck_rollout_sidecar/v1"
OWN_DECK_ROLLOUT_SIDECAR_VERSION = 1
OWN_DECK_ROLLOUT_DAILY_META_SCHEMA = "poke_bot.own_deck_rollout_daily_meta/v1"
OWN_DECK_ROLLOUT_DAILY_META_VERSION = 1
OWNER_DECISION_REVISION = 259

ARCHIVE_RECEIPT_SCHEMA = "poke_bot.expert_latest20_receipt/v1"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "sha256:09848f04a6c863a02c517fdcd5b7a61a139eceafd3348aa2a08705fd6e971a16"
)
EXPECTED_VERSIONED_RECEIPT_SHA256 = (
    "sha256:d377cd5b4558150588d1461539d50bcfb2ca46898120b4e3ad97e9d95e479551"
)
WINDOW_START = "2026-07-22"
WINDOW_END = "2026-08-10"
WINDOW_DAYS = 20
WINDOW_TOTAL_EPISODES = 91_253
EXPECTED_IMAGE_TAG = "poke-bot-truenas-worker:matchup-v33-runtime"
EXPECTED_IMAGE_ID = (
    "sha256:74d66c41fda841e96ee89e88fab1fa800b82ab8c6a06cabdff146803a1b05a0f"
)
# The r259 image supplies the competition runtime here.  The sealed source is
# intentionally mounted elsewhere so it cannot hide this dependency.
EXPECTED_SMOKE_CG_RUNTIME = "/workspace/kaggle/input/cg-lib"
# This is the exact 2026-07-22 source member that exposed the legacy generic
# winner conversion's ``float(None)`` failure.  Keep the deployment smoke
# bound to the member by name rather than walking earlier archive entries.
SMOKE_UNKNOWN_OUTCOME_MEMBER = "87394115.json"
SMOKE_UNKNOWN_OUTCOME_EPISODE_ID = "87394115"
_SMOKE_UNKNOWN_OUTCOME_STATUSES = ("TIMEOUT", "DONE")
_SMOKE_MAX_RECORDS = 2
_SMOKE_MAX_ACTIVE_DECISIONS_PER_RECORD = 2

DAILY_DIRECTORY_NAME = "daily"
DAILY_SHARD_NAME = "own_deck_rollouts.jsonl.gz"
DAILY_META_NAME = "meta.json"

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DAY_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")
_ARCHIVE_SLUG_PREFIX = "pokemon-tcg-ai-battle-episodes-"
_OUTCOME_PROVENANCE_VERIFIED = "verified_zero_sum_rewards"
_OUTCOME_PROVENANCE_MASKED = "masked_invalid_or_missing_rewards"


class OwnDeckRolloutStoreError(RuntimeError):
    """The protected source or immutable sidecar contract is invalid."""


class SourceManifestError(OwnDeckRolloutStoreError):
    """The sealed archive manifest, versioned receipt, or ZIP binding drifted."""


class SourceRecordError(OwnDeckRolloutStoreError):
    """A record cannot be causally represented without private-state leakage."""


class ImmutableSidecarError(OwnDeckRolloutStoreError):
    """An existing daily sidecar is incomplete, corrupt, or a different identity."""


@dataclass(frozen=True)
class SourceArchive:
    """One checksum-bound date from the protected r241 archive receipt."""

    day: str
    path: Path
    sha256: str
    bytes: int
    validated_episode_count: int
    source_slug: str

    def metadata(self) -> dict[str, Any]:
        return {
            "date": self.day,
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.bytes,
            "validated_episode_count": self.validated_episode_count,
            "source_slug": self.source_slug,
        }


@dataclass(frozen=True)
class SourceWindow:
    """Verified archive window used for one r259 side-store run."""

    manifest_path: Path
    manifest_sha256: str
    original_manifest_path: str
    original_versioned_receipt_path: str | None
    versioned_receipt_path: Path | None
    versioned_receipt_sha256: str | None
    archives: tuple[SourceArchive, ...]

    @property
    def dates(self) -> tuple[str, ...]:
        return tuple(row.day for row in self.archives)

    def archive_for_day(self, day_text: str) -> SourceArchive:
        for row in self.archives:
            if row.day == day_text:
                return row
        raise SourceManifestError(f"day is outside the protected source window: {day_text}")


@dataclass(frozen=True)
class DailySidecarResult:
    """One immutable daily sidecar outcome."""

    day: str
    directory: Path
    shard_path: Path
    meta_path: Path
    shard_sha256: str
    meta_sha256: str
    rows: int
    source_records: int
    skipped_existing: bool


@dataclass
class _ArchiveNativeAccounting:
    """Non-row receipt facts for one archive-native day materialization."""

    episodes_seen: int = 0
    verified_reward_episodes: int = 0
    invalid_or_missing_reward_episodes: int = 0
    invalid_reward_fallback_records: int = 0
    records_emitted: int = 0
    records_skipped_stale_only: int = 0
    invalid_reward_episodes_skipped_unconvertible: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "poke_bot.own_deck_rollout_archive_native_accounting/v1",
            "episodes_seen": self.episodes_seen,
            "verified_reward_episodes": self.verified_reward_episodes,
            "invalid_or_missing_reward_episodes": self.invalid_or_missing_reward_episodes,
            "invalid_reward_fallback_records": self.invalid_reward_fallback_records,
            "records_emitted": self.records_emitted,
            "records_skipped_stale_only": self.records_skipped_stale_only,
            "invalid_reward_episodes_skipped_unconvertible": (
                self.invalid_reward_episodes_skipped_unconvertible
            ),
        }


@dataclass(frozen=True)
class _BuildIdentity:
    """Stable metadata inputs used to decide immutable resume eligibility."""

    mode: str
    source_snapshot_path: str
    source_snapshot_tree_sha256: str
    image_tag: str
    image_id: str
    code_identities: Mapping[str, str]
    classifier: Mapping[str, Any] | None
    protected_stream_sha256: str | None

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source_snapshot": {
                "path": self.source_snapshot_path,
                "tree_sha256": self.source_snapshot_tree_sha256,
            },
            "image": {"tag": self.image_tag, "id": self.image_id},
            "code": dict(self.code_identities),
            "classifier": None if self.classifier is None else dict(self.classifier),
            "protected_stream_sha256": self.protected_stream_sha256,
        }


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON bytes used for every checksum in this side store."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OwnDeckRolloutStoreError("value is not canonical JSON") from exc


def sha256_bytes(value: bytes) -> str:
    """Return the typed SHA-256 digest for bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str, *, label: str = "file") -> str:
    """Hash one regular, non-symlink file without following an unsafe source."""

    source = _regular_file(path, label=label)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def expected_window_dates() -> tuple[str, ...]:
    """Return the exact fixed r241 20-calendar-day archive window."""

    start = date.fromisoformat(WINDOW_START)
    return tuple(
        (start + timedelta(days=index)).isoformat() for index in range(WINDOW_DAYS)
    )


def load_source_window(
    source_manifest: Path | str,
    *,
    expected_manifest_sha256: str = EXPECTED_SOURCE_MANIFEST_SHA256,
    expected_versioned_receipt_sha256: str | None = EXPECTED_VERSIONED_RECEIPT_SHA256,
    original_manifest_path: str | None = None,
    versioned_receipt_lock: Path | str | None = None,
    verify_archives: bool = True,
) -> SourceWindow:
    """Validate the locked manifest and re-hash every exact source ZIP.

    ``source_manifest`` may be the root-created read-only lock copy mounted in
    a non-root container.  ``original_manifest_path`` preserves the protected
    host path in receipts without making the container read that root-only file.
    """

    manifest_path = _regular_file(source_manifest, label="protected archive manifest")
    observed_manifest_sha = sha256_file(manifest_path, label="protected archive manifest")
    _require_sha256(expected_manifest_sha256, label="expected manifest checksum")
    if observed_manifest_sha != expected_manifest_sha256:
        raise SourceManifestError(
            "protected archive manifest checksum mismatch: "
            f"expected {expected_manifest_sha256}, got {observed_manifest_sha}"
        )
    payload = _read_json_object(manifest_path, label="protected archive manifest")
    if payload.get("schema") != ARCHIVE_RECEIPT_SCHEMA or payload.get("status") != "ready":
        raise SourceManifestError("protected archive manifest is not a ready latest20 receipt")
    if payload.get("window_policy") != "exact_20_consecutive_calendar_days":
        raise SourceManifestError("protected archive manifest window policy drifted")
    if payload.get("window_start") != WINDOW_START or payload.get("window_end") != WINDOW_END:
        raise SourceManifestError("protected archive manifest has the wrong exact window")
    if _exact_int(payload.get("days")) != WINDOW_DAYS:
        raise SourceManifestError("protected archive manifest does not contain 20 days")
    if _exact_int(payload.get("total_episodes")) != WINDOW_TOTAL_EPISODES:
        raise SourceManifestError("protected archive manifest total episode identity drifted")

    raw_archives = payload.get("archives")
    if not isinstance(raw_archives, list) or len(raw_archives) != WINDOW_DAYS:
        raise SourceManifestError("protected archive manifest must list exactly 20 archives")
    parsed_archives: list[SourceArchive] = []
    for raw in raw_archives:
        if not isinstance(raw, Mapping):
            raise SourceManifestError("protected archive row is not an object")
        day_text = str(raw.get("date") or "")
        _validate_day(day_text)
        raw_path = raw.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise SourceManifestError(f"archive path is missing for {day_text}")
        size = _exact_int(raw.get("bytes"))
        if size is None or size < 0:
            raise SourceManifestError(f"archive byte size is invalid for {day_text}")
        checksum = raw.get("sha256")
        _require_sha256(checksum, label=f"archive checksum for {day_text}")
        expected_slug = _ARCHIVE_SLUG_PREFIX + day_text
        if raw.get("dataset_slug") != expected_slug:
            raise SourceManifestError(f"archive dataset slug drifted for {day_text}")
        if raw.get("validated") is not True:
            raise SourceManifestError(f"archive is not validated for {day_text}")
        validated_episodes = _exact_int(raw.get("validated_episode_count"))
        if validated_episodes is None or validated_episodes < 0:
            raise SourceManifestError(f"archive validated episode count is invalid for {day_text}")
        archive_path = Path(raw_path)
        if verify_archives:
            archive_file = _regular_file(archive_path, label=f"source ZIP {day_text}")
            if archive_file.stat().st_size != size:
                raise SourceManifestError(f"source ZIP byte size drifted for {day_text}")
            observed = sha256_file(archive_file, label=f"source ZIP {day_text}")
            if observed != checksum:
                raise SourceManifestError(f"source ZIP checksum drifted for {day_text}")
        parsed_archives.append(
            SourceArchive(
                day=day_text,
                path=archive_path,
                sha256=str(checksum),
                bytes=size,
                validated_episode_count=validated_episodes,
                source_slug=expected_slug,
            )
        )
    if tuple(row.day for row in parsed_archives) != expected_window_dates():
        raise SourceManifestError("protected archive rows are not exact, ordered window dates")
    if sum(row.validated_episode_count for row in parsed_archives) != WINDOW_TOTAL_EPISODES:
        raise SourceManifestError("archive validated episode totals do not bind 91,253 episodes")

    original_versioned_path: str | None = None
    versioned_path: Path | None = None
    versioned_sha: str | None = None
    raw_versioned = payload.get("versioned_receipt")
    if expected_versioned_receipt_sha256 is not None:
        _require_sha256(expected_versioned_receipt_sha256, label="expected versioned receipt checksum")
        if not isinstance(raw_versioned, str) or not raw_versioned:
            raise SourceManifestError("protected archive manifest lacks its versioned receipt path")
        original_versioned_path = raw_versioned
        versioned_path = _regular_file(
            versioned_receipt_lock if versioned_receipt_lock is not None else raw_versioned,
            label="root-verified versioned archive receipt lock",
        )
        versioned_sha = sha256_file(versioned_path, label="versioned archive receipt")
        if versioned_sha != expected_versioned_receipt_sha256:
            raise SourceManifestError("versioned archive receipt checksum drifted")
        versioned_payload = _read_json_object(versioned_path, label="versioned archive receipt")
        _validate_versioned_receipt_equivalence(payload, versioned_payload)
    return SourceWindow(
        manifest_path=manifest_path,
        manifest_sha256=observed_manifest_sha,
        original_manifest_path=original_manifest_path or str(manifest_path),
        original_versioned_receipt_path=original_versioned_path,
        versioned_receipt_path=versioned_path,
        versioned_receipt_sha256=versioned_sha,
        archives=tuple(parsed_archives),
    )


def board_feature_fingerprint(observation: Mapping[str, Any], deck: Sequence[int]) -> str:
    """Fingerprint the exact existing sparse board feature representation.

    The hash covers the four ABI fields needed by an r241 compact decision
    sample join.  It never serializes the sparse vector itself into the sidecar.
    """

    try:
        from . import features

        sparse = features.build_board_tokens(dict(observation), list(deck))
    except Exception as exc:
        raise SourceRecordError(
            "could not build canonical board SparseVector for sidecar join"
        ) from exc
    return sha256_bytes(
        canonical_json_bytes(
            {
                "index": list(sparse.index),
                "value": list(sparse.value),
                "offset": list(sparse.offset),
                "num_words": int(sparse.num_words),
            }
        )
    )


def materialize_record_sidecar_rows(
    record: Mapping[str, Any],
    *,
    source_day: str,
    source_manifest_sha256: str,
    transition_resolver: Callable[[int, int], Mapping[str, Any] | None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize every full decision in one protected acting-seat record.

    The ledger is constructed once and advanced in sorted full ``env_step``
    order.  Supervision receives only target-only public transition snapshots;
    those snapshots are discarded after label/vector construction.
    """

    _validate_day(source_day)
    _require_sha256(source_manifest_sha256, label="row source manifest checksum")
    episode_id = _required_text(record.get("episode_id"), label="episode_id")
    seat = _seat(record.get("seat"), label="record seat")
    if str(record.get("archetype") or "").casefold() != "alakazam":
        raise SourceRecordError("side store accepts only exact acting Alakazam records")
    if record.get("info_set_ok") is not True:
        raise SourceRecordError("side store refuses a source record without info_set_ok")
    outcome_provenance = record.get("_r259_outcome_provenance")
    if outcome_provenance is None or outcome_provenance == _OUTCOME_PROVENANCE_VERIFIED:
        outcome_verified = True
    elif outcome_provenance == _OUTCOME_PROVENANCE_MASKED:
        outcome_verified = False
    else:
        raise SourceRecordError("r259 record outcome provenance is invalid")
    deck = _starting_deck(record.get("deck"))
    raw_steps = record.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise SourceRecordError("source record has no full decision steps")

    prepared_steps: list[dict[str, Any]] = []
    prior_env_step = -1
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            raise SourceRecordError(f"source step {index} is not an object")
        env_step = _exact_int(raw_step.get("env_step"))
        if env_step is None or env_step < 0 or env_step <= prior_env_step:
            raise SourceRecordError("source env_step values must be unique and strictly increasing")
        prior_env_step = env_step
        observation = _masked_public_observation(raw_step.get("observation"), expected_seat=seat)
        action = _action_indices(raw_step.get("action"))
        prepared: dict[str, Any] = {
            "env_step": env_step,
            "actor_seat": seat,
            "observation": observation,
            "action": action,
        }
        for name in (
            "is_prefix",
            "prefix_only",
            "is_final_selected_action",
            "is_final_action_stage",
            "stage_is_final",
            "raw_stage_index",
            "raw_stage_count",
            "stage_index",
            "stage_count",
            "chance",
            "is_chance",
            "chance_boundary",
            "stochastic",
            "unresolved_randomness",
            "opponent_intervened",
            "actor_boundary",
            "turn_boundary",
            "boundary",
            "diagnostics",
        ):
            if name in raw_step:
                prepared[name] = copy.deepcopy(raw_step[name])
        transition = (
            transition_resolver(env_step, seat)
            if transition_resolver is not None
            else _coerce_public_transition(raw_step.get("transition_after"), actor=seat)
        )
        if transition is not None:
            prepared["transition_after"] = transition
            if transition.get("transition_after_immediate") is False:
                prepared["transition_after_immediate"] = False
        prepared_steps.append(prepared)

    labels_by_step = build_own_deck_supervision_targets(prepared_steps)
    if len(labels_by_step) != len(prepared_steps):
        raise SourceRecordError("supervision target builder returned an invalid step count")
    ledger = OwnDeckLedger(deck)
    rows: list[dict[str, Any]] = []
    label_counts = _empty_label_counts()
    for step, labels in zip(prepared_steps, labels_by_step, strict=True):
        observation = step["observation"]
        snapshot = ledger.observe(observation)
        snapshot_payload = snapshot.to_dict()
        board_fingerprint = board_feature_fingerprint(observation, deck)
        stages = _policy_stage_option_features(snapshot, observation, step["action"])
        terminal_labels = dict(labels["terminal_conversion"])
        tutor_labels = dict(labels["visible_tutor_completion"])
        if not outcome_verified:
            terminal_labels, tutor_labels = _mask_unverified_outcome_labels(
                terminal_labels,
                tutor_labels,
            )
        terminal_vector = list(terminal_conversion_target_vector(terminal_labels))
        terminal_mask = list(terminal_conversion_target_mask(terminal_labels))
        tutor_vector = list(visible_tutor_completion_target_vector(tutor_labels))
        tutor_mask = list(visible_tutor_completion_target_mask(tutor_labels))
        if len(terminal_vector) != TERMINAL_CONVERSION_OUTPUT_DIM or len(terminal_mask) != TERMINAL_CONVERSION_OUTPUT_DIM:
            raise SourceRecordError("terminal supervision ABI width drifted")
        if len(tutor_vector) != VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM or len(tutor_mask) != VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM:
            raise SourceRecordError("visible tutor supervision ABI width drifted")
        if not outcome_verified:
            if any(terminal_mask) or any(tutor_mask[3:]):
                raise SourceRecordError("unverified rewards produced an outcome supervision label")
            label_counts["outcome_provenance"]["masked_invalid_or_missing_reward_rows"] += 1
        else:
            label_counts["outcome_provenance"]["verified_reward_rows"] += 1
        row = {
            "schema": OWN_DECK_ROLLOUT_SIDECAR_SCHEMA,
            "version": OWN_DECK_ROLLOUT_SIDECAR_VERSION,
            "owner_decision_revision": OWNER_DECISION_REVISION,
            "episode_id": episode_id,
            "seat": seat,
            "env_step": int(step["env_step"]),
            "source_date": source_day,
            "source_manifest_sha256": source_manifest_sha256,
            "deck_fingerprint": snapshot.deck_fingerprint,
            "observation_fingerprint": _public_observation_fingerprint(observation),
            "board_feature_fingerprint": board_fingerprint,
            "ledger_observation_fingerprint": snapshot.observation_fingerprint,
            "ledger_snapshot": snapshot_payload,
            "policy_stage_option_features": stages,
            "supervision": {
                "schema": OWN_DECK_SUPERVISION_SCHEMA,
                "version": OWN_DECK_SUPERVISION_VERSION,
                "target_only": True,
                "terminal_conversion": {
                    "labels": terminal_labels,
                    "vector": terminal_vector,
                    "mask": terminal_mask,
                },
                "visible_tutor_completion": {
                    "labels": tutor_labels,
                    "vector": tutor_vector,
                    "mask": tutor_mask,
                },
            },
            "training_eligibility": {
                "active_r241": False,
                "sidecar_only": True,
                "successor": "pending_refresh_join_parity_receipt",
            },
        }
        _assert_no_private_output(row)
        _accumulate_label_counts(label_counts, row)
        rows.append(row)
    return rows, label_counts


def build_archive_native_sidecar(
    *,
    source_manifest: Path | str,
    output_root: Path | str,
    classifier_mix: Path | str,
    classifier_representatives: Path | str,
    card_csv: Path | str,
    source_snapshot_path: str,
    source_snapshot_tree_sha256: str,
    image_tag: str = EXPECTED_IMAGE_TAG,
    image_id: str = EXPECTED_IMAGE_ID,
    original_manifest_path: str | None = None,
    versioned_receipt_lock: Path | str | None = None,
    expected_manifest_sha256: str = EXPECTED_SOURCE_MANIFEST_SHA256,
    expected_versioned_receipt_sha256: str | None = EXPECTED_VERSIONED_RECEIPT_SHA256,
    only_days: Iterable[str] | None = None,
    classifier: Any | None = None,
) -> list[DailySidecarResult]:
    """Build r259 sidecars by reading only exact manifest-listed ZIP files."""

    window = load_source_window(
        source_manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_versioned_receipt_sha256=expected_versioned_receipt_sha256,
        original_manifest_path=original_manifest_path,
        versioned_receipt_lock=versioned_receipt_lock,
        verify_archives=True,
    )
    selected_days = _selected_days(window, only_days)
    if classifier is None:
        classifier = _load_classifier(classifier_mix, classifier_representatives, card_csv)
    classifier_contract = _classifier_contract(
        classifier,
        mix_path=Path(classifier_mix),
        representatives_path=Path(classifier_representatives),
        card_csv=Path(card_csv),
    )
    identity = _build_identity(
        mode="archive_native",
        source_snapshot_path=source_snapshot_path,
        source_snapshot_tree_sha256=source_snapshot_tree_sha256,
        image_tag=image_tag,
        image_id=image_id,
        classifier=classifier_contract,
        protected_stream_sha256=None,
    )
    results: list[DailySidecarResult] = []
    for day_text in selected_days:
        archive = window.archive_for_day(day_text)
        accounting = _ArchiveNativeAccounting()
        results.append(
            _build_one_day(
                output_root=Path(output_root),
                window=window,
                archive=archive,
                identity=identity,
                records=_iter_archive_native_records(
                    archive,
                    classifier,
                    accounting=accounting,
                ),
                archive_native_accounting=accounting,
            )
        )
    return results


def build_protected_jsonl_sidecar(
    *,
    source_manifest: Path | str,
    protected_records: Path | str,
    output_root: Path | str,
    source_snapshot_path: str,
    source_snapshot_tree_sha256: str,
    image_tag: str = EXPECTED_IMAGE_TAG,
    image_id: str = EXPECTED_IMAGE_ID,
    original_manifest_path: str | None = None,
    versioned_receipt_lock: Path | str | None = None,
    expected_manifest_sha256: str = EXPECTED_SOURCE_MANIFEST_SHA256,
    expected_versioned_receipt_sha256: str | None = EXPECTED_VERSIONED_RECEIPT_SHA256,
    only_days: Iterable[str] | None = None,
) -> list[DailySidecarResult]:
    """Build sidecars from a protected replay JSONL stream, fail-closed."""

    window = load_source_window(
        source_manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_versioned_receipt_sha256=expected_versioned_receipt_sha256,
        original_manifest_path=original_manifest_path,
        versioned_receipt_lock=versioned_receipt_lock,
        verify_archives=True,
    )
    records_path = _regular_file(protected_records, label="protected replay JSONL stream")
    stream_sha = sha256_file(records_path, label="protected replay JSONL stream")
    selected_days = _selected_days(window, only_days)
    identity = _build_identity(
        mode="protected_jsonl",
        source_snapshot_path=source_snapshot_path,
        source_snapshot_tree_sha256=source_snapshot_tree_sha256,
        image_tag=image_tag,
        image_id=image_id,
        classifier=None,
        protected_stream_sha256=stream_sha,
    )
    results: list[DailySidecarResult] = []
    for day_text in selected_days:
        archive = window.archive_for_day(day_text)
        results.append(
            _build_one_day(
                output_root=Path(output_root),
                window=window,
                archive=archive,
                identity=identity,
                records=_iter_jsonl_records_for_day(records_path, day_text),
            )
        )
    return results


def smoke_archive_native_one_record(
    *,
    source_manifest: Path | str,
    classifier_mix: Path | str,
    classifier_representatives: Path | str,
    card_csv: Path | str,
    original_manifest_path: str | None = None,
    versioned_receipt_lock: Path | str | None = None,
    expected_manifest_sha256: str = EXPECTED_SOURCE_MANIFEST_SHA256,
    expected_versioned_receipt_sha256: str | None = EXPECTED_VERSIONED_RECEIPT_SHA256,
    expected_cg_runtime: Path | str = EXPECTED_SMOKE_CG_RUNTIME,
) -> dict[str, Any]:
    """Materialize one normal archive-native Alakazam record in memory only.

    This is deliberately the low-memory (1 GiB) normal-record preflight.
    It never opens or runs the pinned malformed-reward member smoke; that
    regression has a separate 2 GiB container and a separate CLI operation.
    Neither smoke creates an output file.
    """

    window, archive = _load_smoke_window_and_first_archive(
        source_manifest=source_manifest,
        original_manifest_path=original_manifest_path,
        versioned_receipt_lock=versioned_receipt_lock,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_versioned_receipt_sha256=expected_versioned_receipt_sha256,
    )
    runtime, card_vocab_size = _initialize_smoke_cg_runtime(expected_cg_runtime)

    classifier = _load_classifier(classifier_mix, classifier_representatives, card_csv)
    try:
        record = next(
            _iter_archive_native_records(
                archive,
                classifier,
                excluded_members=(SMOKE_UNKNOWN_OUTCOME_MEMBER,),
            )
        )
    except StopIteration as exc:
        raise SourceRecordError(
            f"r259 smoke found no archive-native Alakazam record in {archive.day}"
        ) from exc
    rows, _label_counts = materialize_record_sidecar_rows(
        record,
        source_day=archive.day,
        source_manifest_sha256=window.manifest_sha256,
    )
    if not rows:
        raise SourceRecordError("r259 smoke record produced no sidecar rows")
    first_row = rows[0]
    return {
        "schema": "poke_bot.own_deck_rollout_smoke/v1",
        "smoke_kind": "normal_archive_native_record",
        "status": "passed_in_memory",
        "source_manifest_sha256": window.manifest_sha256,
        "archive": {"day": archive.day, "sha256": archive.sha256},
        "episode_id": str(first_row["episode_id"]),
        "seat": int(first_row["seat"]),
        "row_count": len(rows),
        "first_env_step": int(first_row["env_step"]),
        "card_vocab_size": card_vocab_size,
        "cg_runtime": str(runtime),
    }


def smoke_archive_native_unknown_outcome_member(
    *,
    source_manifest: Path | str,
    classifier_mix: Path | str,
    classifier_representatives: Path | str,
    card_csv: Path | str,
    original_manifest_path: str | None = None,
    versioned_receipt_lock: Path | str | None = None,
    expected_manifest_sha256: str = EXPECTED_SOURCE_MANIFEST_SHA256,
    expected_versioned_receipt_sha256: str | None = EXPECTED_VERSIONED_RECEIPT_SHA256,
    expected_cg_runtime: Path | str = EXPECTED_SMOKE_CG_RUNTIME,
) -> dict[str, Any]:
    """Run only the bounded pinned malformed-reward smoke in memory.

    The launcher isolates this 2 GiB regression from the normal 1 GiB smoke.
    It reads the exact receipt-bound ZIP/member, uses only r259's winner-free
    conversion path, validates retained ACTIVE records, and proves malformed
    reward labels never survive into terminal/tutor supervision.
    """

    window, archive = _load_smoke_window_and_first_archive(
        source_manifest=source_manifest,
        original_manifest_path=original_manifest_path,
        versioned_receipt_lock=versioned_receipt_lock,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_versioned_receipt_sha256=expected_versioned_receipt_sha256,
    )
    runtime, card_vocab_size = _initialize_smoke_cg_runtime(expected_cg_runtime)
    classifier = _load_classifier(classifier_mix, classifier_representatives, card_csv)
    return {
        "schema": "poke_bot.own_deck_rollout_unknown_outcome_smoke/v1",
        "smoke_kind": "pinned_unknown_outcome_member",
        "status": "passed_in_memory",
        "source_manifest_sha256": window.manifest_sha256,
        "archive": {"day": archive.day, "sha256": archive.sha256},
        "card_vocab_size": card_vocab_size,
        "cg_runtime": str(runtime),
        "pinned_unknown_outcome_member": _smoke_pinned_unknown_outcome_member(
            archive=archive,
            classifier=classifier,
            source_manifest_sha256=window.manifest_sha256,
        ),
    }


def _load_smoke_window_and_first_archive(
    *,
    source_manifest: Path | str,
    original_manifest_path: str | None,
    versioned_receipt_lock: Path | str | None,
    expected_manifest_sha256: str,
    expected_versioned_receipt_sha256: str | None,
) -> tuple[SourceWindow, SourceArchive]:
    """Load and checksum the one receipt-bound ZIP used by a smoke process."""

    window = load_source_window(
        source_manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_versioned_receipt_sha256=expected_versioned_receipt_sha256,
        original_manifest_path=original_manifest_path,
        versioned_receipt_lock=versioned_receipt_lock,
        verify_archives=False,
    )
    if not window.archives:  # pragma: no cover - canonical receipt requires 20
        raise SourceManifestError("r259 smoke has no receipt-bound archive")
    archive = window.archives[0]
    archive_path = _regular_file(archive.path, label=f"r259 smoke ZIP {archive.day}")
    if archive_path.stat().st_size != archive.bytes:
        raise SourceManifestError("r259 smoke ZIP byte size drifted")
    if sha256_file(archive_path, label=f"r259 smoke ZIP {archive.day}") != archive.sha256:
        raise SourceManifestError("r259 smoke ZIP checksum drifted")
    return window, archive


def _initialize_smoke_cg_runtime(expected_cg_runtime: Path | str) -> tuple[Path, int]:
    """Verify the image-baked CG runtime used by either isolated smoke."""

    try:
        from . import cg_env, features

        runtime = cg_env.ensure_cg_importable().resolve()
        expected_runtime = Path(expected_cg_runtime).resolve()
        if runtime != expected_runtime:
            raise SourceRecordError(
                "r259 smoke CG runtime differs from the image-baked runtime: "
                f"{runtime} != {expected_runtime}"
            )
        card_vocab_size = features.card_vocab_size()
    except OwnDeckRolloutStoreError:
        raise
    except Exception as exc:
        raise SourceRecordError("r259 smoke could not initialize CG feature runtime") from exc
    if not isinstance(card_vocab_size, int) or card_vocab_size <= 1:
        raise SourceRecordError("r259 smoke CG feature card vocabulary is invalid")
    return runtime, card_vocab_size


def _smoke_pinned_unknown_outcome_member(
    *,
    archive: SourceArchive,
    classifier: Any,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Exercise the known malformed-reward member without generic conversion.

    The member is intentionally read through ``ZipFile.read`` by its exact
    name.  This keeps the preflight bounded even when the member is not near
    the start of the archive's lexicographic listing, and proves the r259-only
    fallback remains safe when a legacy converter cannot derive a winner.
    """

    payload = _read_exact_smoke_member(archive)
    _validate_smoke_unknown_outcome_payload(payload)
    try:
        _decks, labels = classifier.classify_episode(payload)
        label_ids = tuple(str(label.deck_id) for label in labels)
    except Exception as exc:
        raise SourceRecordError(
            "r259 smoke classifier failed for pinned malformed-reward member"
        ) from exc
    if len(label_ids) != 2:
        raise SourceRecordError(
            "r259 smoke classifier did not return two seats for pinned malformed-reward member"
        )

    # Do not call ``_iter_archive_native_records`` here: that route imports
    # the legacy generic converter for valid outcomes.  This exact payload is
    # deliberately unverified, so only the r259-local winner-free converter
    # is allowed to inspect it.
    fallback_records = _convert_unknown_outcome_episode_to_r259_records(
        payload,
        source=archive.source_slug,
        seat_archetypes=label_ids,
        max_active_decisions_per_record=_SMOKE_MAX_ACTIVE_DECISIONS_PER_RECORD,
    )
    if not fallback_records:
        raise SourceRecordError(
            "r259 smoke pinned malformed-reward member has no causal Alakazam fallback record"
        )
    if len(fallback_records) > _SMOKE_MAX_RECORDS:  # pragma: no cover - two seats are canonical
        raise SourceRecordError("r259 smoke fallback exceeded the two-seat record bound")

    retained: list[dict[str, Any]] = []
    stale_only_records = 0
    for record in fallback_records:
        projected = _project_r259_native_record(
            record,
            outcome_provenance=_OUTCOME_PROVENANCE_MASKED,
        )
        filtered = _validate_native_record_decisions(projected, payload)
        if filtered is None:
            stale_only_records += 1
            continue
        attached = _attach_public_transitions(
            filtered,
            payload,
            outcome_verified=False,
        )
        if attached.get("episode_id") != SMOKE_UNKNOWN_OUTCOME_EPISODE_ID:
            raise SourceRecordError("r259 smoke pinned member episode identity drifted")
        retained.append(attached)
    if not retained:
        raise SourceRecordError(
            "r259 smoke pinned malformed-reward member retained no ACTIVE decision record"
        )

    fallback_active_decision_count = _record_step_count(fallback_records)
    retained_active_decision_count = _record_step_count(retained)
    if retained_active_decision_count > (
        len(retained) * _SMOKE_MAX_ACTIVE_DECISIONS_PER_RECORD
    ):
        raise SourceRecordError("r259 smoke retained more decisions than its conversion bound")

    rows: list[dict[str, Any]] = []
    label_counts = _empty_label_counts()
    materialized_records = 0
    for record in retained:
        record_rows, record_counts = materialize_record_sidecar_rows(
            record,
            source_day=archive.day,
            source_manifest_sha256=source_manifest_sha256,
        )
        if not record_rows:
            raise SourceRecordError(
                "r259 smoke pinned malformed-reward record produced no sidecar rows"
            )
        for row in record_rows:
            _assert_smoke_masked_outcome_row(row)
        rows.extend(record_rows)
        _merge_label_counts(label_counts, record_counts)
        materialized_records += 1
    if not rows:  # pragma: no cover - guarded per record above
        raise SourceRecordError("r259 smoke produced no pinned malformed-reward sidecar row")
    if label_counts["outcome_provenance"] != {
        "verified_reward_rows": 0,
        "masked_invalid_or_missing_reward_rows": len(rows),
    }:
        raise SourceRecordError("r259 smoke pinned member outcome accounting drifted")

    return {
        "member": SMOKE_UNKNOWN_OUTCOME_MEMBER,
        "episode_id": SMOKE_UNKNOWN_OUTCOME_EPISODE_ID,
        "fallback_record_count": len(fallback_records),
        "retained_active_record_count": len(retained),
        "stale_only_record_count": stale_only_records,
        "max_active_decisions_per_record": _SMOKE_MAX_ACTIVE_DECISIONS_PER_RECORD,
        "fallback_active_decision_count": fallback_active_decision_count,
        "retained_active_decision_count": retained_active_decision_count,
        "materialized_record_count": materialized_records,
        "materialized_active_decision_count": len(rows),
        "sidecar_row_count": len(rows),
        "terminal_outcome_masked_row_count": len(rows),
        "outcome_provenance": _OUTCOME_PROVENANCE_MASKED,
        "label_counts": label_counts,
    }


def _read_exact_smoke_member(archive: SourceArchive) -> dict[str, Any]:
    """Read only the named regression member from an already-hashed ZIP."""

    archive_path = _regular_file(archive.path, label=f"r259 smoke ZIP {archive.day}")
    try:
        with zipfile.ZipFile(archive_path, "r") as handle:
            try:
                info = handle.getinfo(SMOKE_UNKNOWN_OUTCOME_MEMBER)
                if info.is_dir():
                    raise SourceRecordError("r259 smoke member unexpectedly names a directory")
                raw = handle.read(info)
            except KeyError as exc:
                raise SourceRecordError(
                    "r259 smoke ZIP lacks pinned malformed-reward member "
                    f"{SMOKE_UNKNOWN_OUTCOME_MEMBER}"
                ) from exc
    except zipfile.BadZipFile as exc:
        raise SourceRecordError(f"r259 smoke ZIP is unreadable: {archive.day}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceRecordError("r259 smoke pinned member is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise SourceRecordError("r259 smoke pinned member JSON is not an object")
    return payload


def _validate_smoke_unknown_outcome_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed if the pinned regression no longer has its audited shape."""

    try:
        from . import replay_import

        episode_id = replay_import.episode_id_of(dict(payload))
    except Exception as exc:
        raise SourceRecordError("r259 smoke cannot read pinned member episode identity") from exc
    if episode_id != SMOKE_UNKNOWN_OUTCOME_EPISODE_ID:
        raise SourceRecordError("r259 smoke pinned member episode identity drifted")
    statuses = payload.get("statuses")
    if not isinstance(statuses, (list, tuple)) or tuple(statuses) != _SMOKE_UNKNOWN_OUTCOME_STATUSES:
        raise SourceRecordError("r259 smoke pinned member status envelope drifted")
    rewards = payload.get("rewards")
    if (
        not isinstance(rewards, (list, tuple))
        or len(rewards) != 2
        or rewards[0] is not None
        or isinstance(rewards[1], bool)
        or not isinstance(rewards[1], (int, float))
    ):
        raise SourceRecordError("r259 smoke pinned member reward envelope drifted")
    try:
        right_reward = float(rewards[1])
    except OverflowError as exc:
        raise SourceRecordError("r259 smoke pinned member reward envelope drifted") from exc
    if not math.isfinite(right_reward) or right_reward != 1.0:
        raise SourceRecordError("r259 smoke pinned member reward envelope drifted")
    if _verified_episode_rewards(payload) is not None:
        raise SourceRecordError("r259 smoke pinned member unexpectedly has verified rewards")


def _record_step_count(records: Sequence[Mapping[str, Any]]) -> int:
    """Count already-retained decision steps without looking at raw payloads."""

    total = 0
    for record in records:
        steps = record.get("steps")
        if not isinstance(steps, list):
            raise SourceRecordError("r259 smoke record has no retained ACTIVE decision steps")
        total += len(steps)
    return total


def _assert_smoke_masked_outcome_row(row: Mapping[str, Any]) -> None:
    """Prove no malformed-reward outcome label survives smoke materialization."""

    supervision = row.get("supervision")
    if not isinstance(supervision, Mapping):
        raise SourceRecordError("r259 smoke row lacks supervision")
    terminal = supervision.get("terminal_conversion")
    tutor = supervision.get("visible_tutor_completion")
    if not isinstance(terminal, Mapping) or not isinstance(tutor, Mapping):
        raise SourceRecordError("r259 smoke row has malformed supervision families")
    terminal_mask = terminal.get("mask")
    tutor_mask = tutor.get("mask")
    terminal_labels = terminal.get("labels")
    tutor_labels = tutor.get("labels")
    if (
        not isinstance(terminal_mask, list)
        or len(terminal_mask) != TERMINAL_CONVERSION_OUTPUT_DIM
        or any(terminal_mask)
        or not isinstance(tutor_mask, list)
        or len(tutor_mask) != VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM
        or any(tutor_mask[3:])
        or not isinstance(terminal_labels, Mapping)
        or terminal_labels.get("provenance") != "masked_invalid_or_missing_episode_rewards"
        or not isinstance(tutor_labels, Mapping)
        or not isinstance(tutor_labels.get("same_actor_terminal_class"), Mapping)
        or tutor_labels["same_actor_terminal_class"].get("mask") is not False
    ):
        raise SourceRecordError(
            "r259 smoke pinned malformed-reward member produced an outcome supervision label"
        )


def iter_daily_sidecar_rows(
    output_root: Path | str,
    day_text: str,
    *,
    expected_meta_sha256: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Verify one immutable daily shard before yielding its JSONL rows."""

    _validate_day(day_text)
    root = Path(output_root)
    directory = root / DAILY_DIRECTORY_NAME / day_text
    meta = _validate_daily_directory(directory, expected_meta_sha256=expected_meta_sha256)
    shard = directory / DAILY_SHARD_NAME
    rows_digest = hashlib.sha256()
    count = 0
    with gzip.open(shard, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                raise ImmutableSidecarError(f"blank line in immutable shard at {line_no}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ImmutableSidecarError(f"invalid JSONL row at {line_no}") from exc
            if not isinstance(row, dict):
                raise ImmutableSidecarError(f"non-object JSONL row at {line_no}")
            _validate_sidecar_row_shape(row, day_text=day_text, meta=meta)
            encoded = canonical_json_bytes(row)
            rows_digest.update(encoded)
            rows_digest.update(b"\n")
            count += 1
            yield row
    if "sha256:" + rows_digest.hexdigest() != meta.get("rows_sha256"):
        raise ImmutableSidecarError("daily row digest drifted")
    if count != _exact_int(meta.get("row_count")):
        raise ImmutableSidecarError("daily row count drifted")


def read_daily_meta(output_root: Path | str, day_text: str) -> dict[str, Any]:
    """Return a validated immutable daily sidecar meta object."""

    _validate_day(day_text)
    return _validate_daily_directory(Path(output_root) / DAILY_DIRECTORY_NAME / day_text)


def _build_one_day(
    *,
    output_root: Path,
    window: SourceWindow,
    archive: SourceArchive,
    identity: _BuildIdentity,
    records: Iterable[Mapping[str, Any]],
    archive_native_accounting: _ArchiveNativeAccounting | None = None,
) -> DailySidecarResult:
    output_root = _regular_or_new_directory(output_root, label="r259 output root")
    with _exclusive_day_lock(output_root, archive.day):
        return _build_one_day_locked(
            output_root=output_root,
            window=window,
            archive=archive,
            identity=identity,
            records=records,
            archive_native_accounting=archive_native_accounting,
        )


def _build_one_day_locked(
    *,
    output_root: Path,
    window: SourceWindow,
    archive: SourceArchive,
    identity: _BuildIdentity,
    records: Iterable[Mapping[str, Any]],
    archive_native_accounting: _ArchiveNativeAccounting | None = None,
) -> DailySidecarResult:
    """Build under the per-day flock held by :func:`_build_one_day`."""

    output_root = _regular_or_new_directory(output_root, label="r259 output root")
    daily_root = output_root / DAILY_DIRECTORY_NAME
    if daily_root.exists() and daily_root.is_symlink():
        raise ImmutableSidecarError("daily output root may not be a symlink")
    daily_root.mkdir(parents=True, exist_ok=True)
    target = daily_root / archive.day
    expected_identity = _expected_daily_identity(window, archive, identity)
    if target.exists() or target.is_symlink():
        meta = _validate_daily_directory(target)
        _require_existing_identity(meta, expected_identity)
        return DailySidecarResult(
            day=archive.day,
            directory=target,
            shard_path=target / DAILY_SHARD_NAME,
            meta_path=target / DAILY_META_NAME,
            shard_sha256=str(meta["shard_sha256"]),
            meta_sha256=str(meta["meta_sha256"]),
            rows=int(meta["row_count"]),
            source_records=int(meta["source_record_count"]),
            skipped_existing=True,
        )

    temporary = Path(tempfile.mkdtemp(prefix=f".{archive.day}.r259.", dir=daily_root))
    shard_path = temporary / DAILY_SHARD_NAME
    meta_path = temporary / DAILY_META_NAME
    rows_digest = hashlib.sha256()
    source_digest = hashlib.sha256()
    keys: set[tuple[str, int, int]] = set()
    label_counts = _empty_label_counts()
    source_records = 0
    rows = 0
    try:
        with shard_path.open("xb") as raw_stream:
            with gzip.GzipFile(
                fileobj=raw_stream,
                mode="wb",
                filename="",
                mtime=0,
            ) as compressed:
                for record in records:
                    if not isinstance(record, Mapping):
                        raise SourceRecordError("source record stream contains a non-object row")
                    source_day = _source_day_from_record(record)
                    if source_day != archive.day:
                        raise SourceRecordError(
                            f"source record day {source_day} does not match selected archive {archive.day}"
                        )
                    source_digest.update(canonical_json_bytes(_public_record_identity(record)))
                    source_digest.update(b"\n")
                    source_records += 1
                    materialized, record_counts = materialize_record_sidecar_rows(
                        record,
                        source_day=archive.day,
                        source_manifest_sha256=window.manifest_sha256,
                    )
                    _merge_label_counts(label_counts, record_counts)
                    for row in materialized:
                        key = (str(row["episode_id"]), int(row["seat"]), int(row["env_step"]))
                        if key in keys:
                            raise SourceRecordError(f"duplicate sidecar key: {key}")
                        keys.add(key)
                        encoded = canonical_json_bytes(row)
                        compressed.write(encoded)
                        compressed.write(b"\n")
                        rows_digest.update(encoded)
                        rows_digest.update(b"\n")
                        rows += 1
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        shard_sha = sha256_file(shard_path, label="temporary r259 shard")
        shard_bytes = shard_path.stat().st_size
        meta = {
            "schema": OWN_DECK_ROLLOUT_DAILY_META_SCHEMA,
            "version": OWN_DECK_ROLLOUT_DAILY_META_VERSION,
            "owner_decision_revision": OWNER_DECISION_REVISION,
            "status": "complete_immutable_sidecar",
            "day": archive.day,
            "shard": {
                "path": DAILY_SHARD_NAME,
                "sha256": shard_sha,
                "bytes": shard_bytes,
                "compression": "gzip",
                "format": "jsonl",
                "row_schema": OWN_DECK_ROLLOUT_SIDECAR_SCHEMA,
                "row_version": OWN_DECK_ROLLOUT_SIDECAR_VERSION,
            },
            "shard_sha256": shard_sha,
            "rows_sha256": "sha256:" + rows_digest.hexdigest(),
            "row_count": rows,
            "source_record_count": source_records,
            "source_records_sha256": "sha256:" + source_digest.hexdigest(),
            "source": expected_identity["source"],
            "build": expected_identity["build"],
            "label_counts": label_counts,
            "training_eligibility": {
                "active_r241": False,
                "sidecar_only": True,
                "successor": "pending_refresh_join_parity_receipt",
            },
        }
        if archive_native_accounting is not None:
            if source_records != archive_native_accounting.records_emitted:
                raise SourceRecordError("archive-native accounting record count drifted")
            meta["archive_native_accounting"] = archive_native_accounting.to_dict()
        meta["meta_sha256"] = _meta_digest(meta)
        _write_atomic_file(meta_path, canonical_json_bytes(meta) + b"\n")
        os.chmod(shard_path, 0o444)
        os.chmod(meta_path, 0o444)
        _fsync_directory(temporary)
        os.replace(temporary, target)
        # macOS requires write permission on the source directory for rename;
        # seal the committed directory immediately after the atomic publish.
        os.chmod(target, 0o555)
        _fsync_directory(target)
        _fsync_directory(daily_root)
        return DailySidecarResult(
            day=archive.day,
            directory=target,
            shard_path=target / DAILY_SHARD_NAME,
            meta_path=target / DAILY_META_NAME,
            shard_sha256=shard_sha,
            meta_sha256=str(meta["meta_sha256"]),
            rows=rows,
            source_records=source_records,
            skipped_existing=False,
        )
    except BaseException:
        # Preserve a failed temporary directory for audit rather than deleting
        # it.  It is not a daily target and can never be mistaken for a commit.
        if temporary.exists() and temporary.is_dir():
            _fsync_directory(temporary)
        raise


def _iter_archive_native_records(
    archive: SourceArchive,
    classifier: Any,
    *,
    accounting: _ArchiveNativeAccounting | None = None,
    excluded_members: Collection[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Read one immutable ZIP directly; never touch generic collector paths.

    ``excluded_members`` is only used by the normal 1 GiB preflight to keep
    its workload disjoint from the separately contained malformed-reward
    smoke.  Production never supplies it and processes every receipt member.
    """

    try:
        from .replay_import import convert_episode_to_records
    except Exception as exc:  # pragma: no cover - import failure is deployment failure
        raise SourceRecordError("could not load replay conversion helper") from exc
    archive_path = _regular_file(archive.path, label=f"source ZIP {archive.day}")
    excluded = frozenset(excluded_members or ())
    if any(not isinstance(member, str) or not member for member in excluded):
        raise SourceRecordError("archive-native member exclusion is malformed")
    try:
        with zipfile.ZipFile(archive_path, "r") as handle:
            members = sorted(
                name for name in handle.namelist() if name.lower().endswith(".json") and not name.endswith("/")
            )
            if len(members) != archive.validated_episode_count:
                raise SourceRecordError(
                    "source ZIP JSON episode-member count drifted for "
                    f"{archive.day}: expected {archive.validated_episode_count}, got {len(members)}"
                )
            for member in members:
                if member in excluded:
                    # Do not deserialize, classify, or offer this member to
                    # the generic path.  Its isolated smoke owns it.
                    continue
                try:
                    payload = json.loads(handle.read(member).decode("utf-8"))
                except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SourceRecordError(f"invalid episode JSON in {archive.day}:{member}") from exc
                if not isinstance(payload, dict):
                    raise SourceRecordError(f"episode JSON is not an object in {archive.day}:{member}")
                try:
                    _decks, labels = classifier.classify_episode(payload)
                    label_ids = [str(label.deck_id) for label in labels]
                except Exception as exc:
                    raise SourceRecordError(
                        f"archive-native classifier failed for {archive.day}:{member}"
                    ) from exc
                outcome_verified = _verified_episode_rewards(payload) is not None
                if accounting is not None:
                    accounting.episodes_seen += 1
                    if outcome_verified:
                        accounting.verified_reward_episodes += 1
                    else:
                        accounting.invalid_or_missing_reward_episodes += 1
                try:
                    if outcome_verified:
                        records = convert_episode_to_records(
                            payload,
                            source=archive.source_slug,
                            archetype_filter="alakazam",
                            seat_archetypes=label_ids,
                            allowed_archetypes=("alakazam",),
                            require_complete=True,
                            strict_info_set=True,
                        )
                    else:
                        records = _convert_unknown_outcome_episode_to_r259_records(
                            payload,
                            source=archive.source_slug,
                            seat_archetypes=label_ids,
                        )
                except Exception as exc:
                    raise SourceRecordError(
                        f"archive-native record conversion failed for {archive.day}:{member}"
                    ) from exc
                if not isinstance(records, list) or not all(isinstance(record, Mapping) for record in records):
                    raise SourceRecordError("replay converter returned an invalid record sequence")
                if not outcome_verified and accounting is not None:
                    accounting.invalid_reward_fallback_records += len(records)
                    if not records and "alakazam" in {label.casefold() for label in label_ids}:
                        accounting.invalid_reward_episodes_skipped_unconvertible += 1
                for record in records:
                    projected = _project_r259_native_record(
                        record,
                        outcome_provenance=(
                            _OUTCOME_PROVENANCE_VERIFIED
                            if outcome_verified
                            else _OUTCOME_PROVENANCE_MASKED
                        ),
                    )
                    filtered = _validate_native_record_decisions(projected, payload)
                    if filtered is None:
                        # The shared converter can emit only stale INACTIVE /
                        # DONE echoes for a seat.  They are not decisions and
                        # must not become an empty synthetic r259 record.
                        if accounting is not None:
                            accounting.records_skipped_stale_only += 1
                        continue
                    if accounting is not None:
                        accounting.records_emitted += 1
                    yield _attach_public_transitions(
                        filtered,
                        payload,
                        outcome_verified=outcome_verified,
                    )
    except zipfile.BadZipFile as exc:
        raise SourceRecordError(f"source ZIP is unreadable: {archive.day}") from exc


def _verified_episode_rewards(payload: Mapping[str, Any]) -> tuple[float, float] | None:
    """Return only a finite zero-sum pair from an authoritative DONE envelope."""

    statuses = payload.get("statuses")
    if not isinstance(statuses, (list, tuple)) or list(statuses) != ["DONE", "DONE"]:
        return None
    rewards = payload.get("rewards")
    if not isinstance(rewards, (list, tuple)) or len(rewards) != 2:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in rewards):
        return None
    try:
        left, right = float(rewards[0]), float(rewards[1])
    except OverflowError:
        return None
    if not math.isfinite(left) or not math.isfinite(right) or left != -right:
        return None
    return left, right


def _convert_unknown_outcome_episode_to_r259_records(
    payload: Mapping[str, Any],
    *,
    source: str,
    seat_archetypes: Sequence[str],
    max_active_decisions_per_record: int | None = None,
) -> list[dict[str, Any]]:
    """R259-only causal conversion when the generic winner is unavailable.

    The legacy converter requires a numeric final reward before it can emit a
    record.  This narrow fallback intentionally omits winner/value/opponent
    labels and retains only exact actor-visible action rows needed by the
    ledger and visible-tutor store.  It never repairs a malformed action,
    invents a reward, or synthesizes an env-step.  Production passes no
    decision limit and retains the complete causal record.  The exact-member
    deployment smoke alone may provide a small positive bound; in that mode
    it stops immediately after that many raw ``ACTIVE`` legal decisions per
    selected seat, before copying later full observations.
    """

    if len(seat_archetypes) != 2:
        raise SourceRecordError("archive-native classifier must identify two seats")
    if max_active_decisions_per_record is not None and (
        isinstance(max_active_decisions_per_record, bool)
        or not isinstance(max_active_decisions_per_record, int)
        or max_active_decisions_per_record <= 0
    ):
        raise SourceRecordError("r259 unknown-outcome decision limit must be a positive integer")
    try:
        from . import replay_import

        decks = replay_import.extract_setup_decks(dict(payload))
        episode_id = replay_import.episode_id_of(dict(payload))
    except Exception as exc:
        raise SourceRecordError("r259 could not read unknown-outcome episode setup") from exc
    if not isinstance(episode_id, str) or not episode_id:
        return []
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        return []

    records: list[dict[str, Any]] = []
    for seat, archetype in enumerate(seat_archetypes):
        if str(archetype).casefold() != "alakazam":
            continue
        try:
            deck = _starting_deck(decks[seat])
        except (IndexError, SourceRecordError):
            continue
        steps: list[dict[str, Any]] = []
        info_set_safe = True
        for env_step, frame in enumerate(raw_steps):
            if not isinstance(frame, list) or seat >= len(frame):
                continue
            entry = frame[seat]
            if not isinstance(entry, Mapping):
                continue
            # This raw status gate exists only for the bounded smoke path.
            # The unbounded production fallback deliberately preserves every
            # candidate for the established exact-index stale-row validator.
            if (
                max_active_decisions_per_record is not None
                and entry.get("status") != "ACTIVE"
            ):
                continue
            observation = entry.get("observation")
            action = entry.get("action")
            if not isinstance(observation, Mapping) or not isinstance(action, list):
                continue
            select = observation.get("select")
            current = observation.get("current")
            if not isinstance(select, Mapping) or not isinstance(current, Mapping):
                continue
            if _seat_or_none(current.get("yourIndex")) != seat:
                continue
            options = select.get("option")
            min_count = _exact_int(select.get("minCount", 0))
            max_count = _exact_int(select.get("maxCount", len(options) if isinstance(options, list) else 0))
            if not isinstance(options, list) or min_count is None or max_count is None:
                continue
            try:
                action_is_legal = replay_import._is_option_index_action(  # type: ignore[attr-defined]
                    action,
                    len(options),
                    min_count=min_count,
                    max_count=max_count,
                )
                masked, _aux, report = replay_import._strip_opp_private(  # type: ignore[attr-defined]
                    copy.deepcopy(dict(observation))
                )
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
                info_set_safe = False
                break
            if not action_is_legal:
                continue
            if not isinstance(masked, dict) or (not report.ok and not report.remasked):
                info_set_safe = False
                break
            steps.append(
                {
                    "env_step": env_step,
                    "observation": masked,
                    "action": [int(value) for value in action],
                }
            )
            if (
                max_active_decisions_per_record is not None
                and len(steps) >= max_active_decisions_per_record
            ):
                # Avoid requesting or copying another raw frame.  The raw
                # successor of this exact retained step remains available to
                # target derivation after conversion.
                break
        if info_set_safe and steps:
            records.append(
                {
                    "episode_id": episode_id,
                    "source": source,
                    "seat": seat,
                    "archetype": "alakazam",
                    "deck": deck,
                    "info_set_ok": True,
                    "steps": steps,
                    "n_decisions": len(steps),
                }
            )
    return records


def _project_r259_native_record(
    record: Mapping[str, Any],
    *,
    outcome_provenance: str,
) -> dict[str, Any]:
    """Drop generic converter labels/private fields before r259 processing."""

    steps = record.get("steps")
    if not isinstance(steps, list):
        raise SourceRecordError("native converted record has no step list")
    projected_steps: list[dict[str, Any]] = []
    retained_step_fields = (
        "env_step",
        "observation",
        "action",
        "is_prefix",
        "prefix_only",
        "is_final_selected_action",
        "is_final_action_stage",
        "stage_is_final",
        "raw_stage_index",
        "raw_stage_count",
        "stage_index",
        "stage_count",
        "chance",
        "is_chance",
        "chance_boundary",
        "stochastic",
        "unresolved_randomness",
        "opponent_intervened",
        "actor_boundary",
        "turn_boundary",
        "boundary",
        "diagnostics",
    )
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise SourceRecordError(f"native converted step {index} is malformed")
        projected_steps.append(
            {
                name: copy.deepcopy(step[name])
                for name in retained_step_fields
                if name in step
            }
        )
    return {
        "episode_id": record.get("episode_id"),
        "source": record.get("source"),
        "seat": record.get("seat"),
        "archetype": record.get("archetype"),
        "deck": copy.deepcopy(record.get("deck")),
        "info_set_ok": record.get("info_set_ok"),
        "steps": projected_steps,
        "n_decisions": record.get("n_decisions"),
        "_r259_outcome_provenance": outcome_provenance,
    }


def _iter_jsonl_records_for_day(path: Path, day_text: str) -> Iterator[dict[str, Any]]:
    """Stream one day from a protected JSONL or JSONL.GZ source without writes."""

    opener: Callable[..., Any] = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                raise SourceRecordError(f"blank source JSONL line {line_no}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SourceRecordError(f"invalid source JSONL at line {line_no}") from exc
            if not isinstance(record, dict):
                raise SourceRecordError(f"source JSONL object expected at line {line_no}")
            if _source_day_from_record(record) == day_text:
                yield record


def _attach_public_transitions(
    record: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    outcome_verified: bool,
) -> dict[str, Any]:
    """Attach non-serialized target-only public post-action snapshots.

    The returned mapping is ephemeral.  Its ``transition_after`` values flow
    into the supervision builder and are never copied into JSONL sidecar rows.
    """

    copied = dict(record)
    seat = _seat(copied.get("seat"), label="native record seat")
    raw_steps = copied.get("steps")
    if not isinstance(raw_steps, list):
        raise SourceRecordError("native record lacks a step list")
    updated_steps: list[dict[str, Any]] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, Mapping):
            raise SourceRecordError("native record step is malformed")
        step = dict(raw_step)
        env_step = _exact_int(step.get("env_step"))
        if env_step is None or env_step < 0:
            raise SourceRecordError("native record env_step is invalid")
        transition = _derive_public_transition_from_episode(
            payload,
            env_step,
            seat,
            outcome_verified=outcome_verified,
        )
        if transition is not None:
            step["transition_after"] = transition
            if transition.get("transition_after_immediate") is False:
                step["transition_after_immediate"] = False
        updated_steps.append(step)
    copied["steps"] = updated_steps
    return copied


def _validate_native_record_decisions(
    record: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Filter raw-index-proven stale echoes and validate retained decisions.

    ``convert_episode_to_records`` predates r259 and can retain an action echo
    from a same-index raw ``INACTIVE`` or ``DONE`` row.  Those two explicit
    statuses are the *only* converted rows r259 is allowed to drop.  Every
    other converted row must map without remapping to the exact raw frame and
    seat, be status ``ACTIVE``, and match the actor-visible masked observation
    and action byte-for-byte in their canonical public representations.

    Returning ``None`` means all converted rows were proven stale; callers
    skip that episode/seat rather than creating a synthetic empty decision
    sequence.  The input mapping is never mutated.
    """

    seat = _seat(record.get("seat"), label="native record seat")
    raw_episode_steps = payload.get("steps")
    converted_steps = record.get("steps")
    if not isinstance(raw_episode_steps, list) or not isinstance(converted_steps, list):
        raise SourceRecordError("native replay decision stream is malformed")
    declared_count = record.get("n_decisions")
    if declared_count is not None:
        parsed_count = _exact_int(declared_count)
        if parsed_count is None or parsed_count != len(converted_steps):
            raise SourceRecordError("native converted decision count drifted")

    retained: list[dict[str, Any]] = []
    retained_env_steps: list[int] = []
    active_converted_env_steps: list[int] = []
    seen_env_steps: set[int] = set()
    for index, converted in enumerate(converted_steps):
        if not isinstance(converted, Mapping):
            raise SourceRecordError(f"native converted step {index} is malformed")
        env_step = _exact_int(converted.get("env_step"))
        if env_step is None or not 0 <= env_step < len(raw_episode_steps):
            raise SourceRecordError("native converted env_step is outside raw episode")
        if env_step in seen_env_steps:
            raise SourceRecordError("native converted decision repeats an env_step")
        seen_env_steps.add(env_step)
        raw_rows = raw_episode_steps[env_step]
        if not isinstance(raw_rows, list) or seat >= len(raw_rows):
            raise SourceRecordError("native raw decision frame lacks acting seat")
        raw_entry = raw_rows[seat]
        if not isinstance(raw_entry, Mapping):
            raise SourceRecordError("native raw decision entry is malformed")
        raw_status = raw_entry.get("status")
        if raw_status in {"INACTIVE", "DONE"}:
            # Only exact same-index/same-seat stale rows are removable.  Do
            # not inspect, borrow, or remap a different frame to replace one.
            continue
        if raw_status != "ACTIVE":
            raise SourceRecordError("native converted decision is not a fresh ACTIVE raw frame")
        active_converted_env_steps.append(env_step)
        if "actor_seat" in converted and _seat(
            converted.get("actor_seat"), label="native converted actor_seat"
        ) != seat:
            raise SourceRecordError("native converted actor does not match record seat")
        raw_observation = raw_entry.get("observation")
        if not isinstance(raw_observation, Mapping):
            raise SourceRecordError("native ACTIVE decision has no observation")
        converted_observation = converted.get("observation")
        if not isinstance(converted_observation, Mapping):
            raise SourceRecordError("native converted decision has no observation")
        raw_public = _masked_public_observation(raw_observation, expected_seat=seat)
        converted_public = _masked_public_observation(converted_observation, expected_seat=seat)
        if _public_observation_fingerprint(raw_public) != _public_observation_fingerprint(converted_public):
            raise SourceRecordError("native converted decision public observation drifted from raw ACTIVE frame")
        raw_action = _action_indices(raw_entry.get("action"))
        converted_action = _action_indices(converted.get("action"))
        if raw_action != converted_action:
            raise SourceRecordError("native converted decision action drifted from raw ACTIVE frame")
        retained.append(dict(converted))
        retained_env_steps.append(env_step)

    # This explicit equality proves the filter retained every converted row
    # whose exact raw counterpart was ACTIVE.  It guards future changes from
    # silently broadening the stale-row drop rule.
    if retained_env_steps != active_converted_env_steps:  # pragma: no cover - invariant
        raise SourceRecordError("native ACTIVE decision completeness proof failed")
    if not retained:
        return None
    filtered = dict(record)
    filtered["steps"] = retained
    filtered["n_decisions"] = len(retained)
    return filtered


def _derive_public_transition_from_episode(
    payload: Mapping[str, Any],
    env_step: int,
    actor: int,
    *,
    outcome_verified: bool | None = None,
) -> dict[str, Any] | None:
    """Make a bounded public target snapshot from the next recorded frame.

    The next actor is derived solely from the frame's validated ``ACTIVE`` /
    ``INACTIVE`` statuses.  ``current.yourIndex`` is only a perspective field
    and is never used to infer the next actor.  Ambiguous, chance, opponent,
    or truncated transitions are marked so the supervision module masks them.
    """

    if outcome_verified is None:
        outcome_verified = _verified_episode_rewards(payload) is not None
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or env_step + 1 >= len(raw_steps):
        return None
    next_rows = raw_steps[env_step + 1]
    if not isinstance(next_rows, list) or len(next_rows) < 2:
        return None
    if all(
        isinstance(entry, Mapping) and entry.get("status") == "DONE"
        for entry in next_rows[:2]
    ):
        # DONE observations/actions are stale terminal echoes in the archive.
        # The all-DONE envelope plus top-level rewards is the only admissible
        # terminal evidence, and carries no post-state card/prize/KO facts.
        return (
            _terminal_transition_from_done_envelope(payload, actor=actor)
            if outcome_verified
            else None
        )
    active: list[tuple[int, Mapping[str, Any]]] = []
    for index, entry in enumerate(next_rows[:2]):
        if not isinstance(entry, Mapping):
            return None
        observation = entry.get("observation")
        if entry.get("status") == "ACTIVE" and isinstance(observation, Mapping):
            active.append((index, observation))
    # INACTIVE observations are explicitly stale in the archive contract.
    # The one fresh target surface is the unique status-validated ACTIVE row;
    # without it we cannot prove an immediate post-action state and must mask.
    if len(active) != 1:
        return None
    next_actor, observation = active[0]

    try:
        from .strategic_heads import public_transition_snapshot

        safe_observation = _masked_public_observation(observation, expected_seat=None)
        snapshot = public_transition_snapshot(safe_observation, actor_seat=actor)
    except (AttributeError, KeyError, SourceRecordError, TypeError, ValueError):
        # A malformed target surface is unavailable, never guessed.
        return None
    if not isinstance(snapshot, dict):
        return None
    if not outcome_verified:
        # The next public observation may carry a stale/otherwise unauditable
        # ``current.result``.  Keep the target-only public board available for
        # tutor completion, but remove every outcome fact before supervision.
        snapshot["result"] = None
    snapshot["next_actor_seat"] = next_actor
    boundary = _raw_transition_boundary(next_rows, snapshot, actor=actor, next_actor=next_actor)
    if boundary is not None:
        snapshot["boundary"] = boundary
        snapshot["transition_after_immediate"] = False
    else:
        snapshot["transition_after_immediate"] = True
    if snapshot.get("valid") is not True:
        snapshot["transition_after_immediate"] = False
        snapshot["boundary"] = "unproven_public_transition"
    return snapshot


def _terminal_transition_from_done_envelope(
    payload: Mapping[str, Any], *, actor: int
) -> dict[str, Any] | None:
    """Build a deliberately minimal terminal target from verified rewards."""

    statuses = payload.get("statuses")
    if not isinstance(statuses, (list, tuple)) or list(statuses) != ["DONE", "DONE"]:
        return None
    reward_pair = _verified_episode_rewards(payload)
    if reward_pair is None:
        return None
    left, right = reward_pair
    own, opponent = (left, right) if actor == 0 else (right, left)
    result = actor if own > opponent else (1 - actor if own < opponent else 2)
    return {
        "schema": "poke_bot.public_transition_snapshot/v1",
        "version": 1,
        "actor_seat": actor,
        "next_actor_seat": None,
        "turn": None,
        "result": result,
        "energy_attached": None,
        "retreated": None,
        "players": [],
        "valid": True,
        "terminal_done_envelope": True,
        "transition_after_immediate": True,
    }


def _raw_transition_boundary(
    rows: Sequence[Any],
    snapshot: Mapping[str, Any],
    *,
    actor: int,
    next_actor: int | None,
) -> str | None:
    marker_keys = (
        "chance",
        "is_chance",
        "chance_boundary",
        "stochastic",
        "unresolved_randomness",
        "opponent_intervened",
        "actor_boundary",
        "turn_boundary",
    )
    for entry in rows[:2]:
        if isinstance(entry, Mapping) and any(entry.get(key) is True for key in marker_keys):
            return "chance_boundary"
    result = _exact_int(snapshot.get("result"))
    if next_actor is None:
        return None if result in (0, 1, 2) else "truncated_or_ambiguous_next_actor"
    # An immediately following ACTIVE frame for the other seat is the normal
    # public result of an end-turn action (including attacks).  It is still a
    # causal post-action outcome for terminal/KO/prize labels.  The explicit
    # seat remains in the target-only wrapper for the tutor continuation head;
    # only independently marked intervention/chance/truncation invalidates
    # immediacy here.
    return None


def _coerce_public_transition(value: Any, *, actor: int) -> dict[str, Any] | None:
    """Convert an optional protected-stream transition to target-only public form."""

    if not isinstance(value, Mapping):
        return None
    if value.get("schema") == "poke_bot.public_transition_snapshot/v1":
        result = _public_transition_fields(value)
        if _exact_int(result.get("actor_seat")) != actor:
            return None
        boundary = _explicit_transition_boundary(value)
        if boundary is not None:
            result["boundary"] = boundary
            result["transition_after_immediate"] = False
        elif _seat_or_none(result.get("next_actor_seat")) is None and result.get("result") == -1:
            result["transition_after_immediate"] = False
            result["boundary"] = "unproven_next_actor"
        return result
    nested = value.get("observation") if isinstance(value.get("observation"), Mapping) else value
    if not isinstance(nested, Mapping):
        return None
    try:
        from .strategic_heads import public_transition_snapshot

        snapshot = public_transition_snapshot(
            _masked_public_observation(nested, expected_seat=None), actor_seat=actor
        )
    except (AttributeError, KeyError, SourceRecordError, TypeError, ValueError):
        return None
    if not isinstance(snapshot, dict):
        return None
    supplied_next = _seat_or_none(value.get("next_actor_seat"))
    snapshot["next_actor_seat"] = supplied_next
    boundary = _explicit_transition_boundary(value)
    if boundary is not None:
        snapshot["transition_after_immediate"] = False
        snapshot["boundary"] = boundary
    elif supplied_next is None and _exact_int(snapshot.get("result")) == -1:
        snapshot["transition_after_immediate"] = False
        snapshot["boundary"] = "unproven_next_actor"
    else:
        snapshot["transition_after_immediate"] = bool(value.get("transition_after_immediate", True))
    return _public_transition_fields(snapshot)


def _explicit_transition_boundary(value: Mapping[str, Any]) -> str | None:
    """Normalize only explicit source provenance into a masking boundary."""

    marker_names = (
        ("chance", "chance_boundary"),
        ("is_chance", "chance_boundary"),
        ("chance_boundary", "chance_boundary"),
        ("stochastic", "chance_boundary"),
        ("unresolved_randomness", "chance_boundary"),
        ("opponent_intervened", "opponent_intervened"),
        ("actor_boundary", "actor_boundary"),
        ("turn_boundary", "turn_boundary"),
    )
    for field, boundary in marker_names:
        if value.get(field) is True:
            return boundary
    raw_boundary = value.get("boundary")
    if isinstance(raw_boundary, str) and raw_boundary:
        return raw_boundary
    return None


def _public_transition_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the typed public transition schema, never arbitrary raw state."""

    allowed = {
        "schema",
        "version",
        "actor_seat",
        "next_actor_seat",
        "turn",
        "result",
        "energy_attached",
        "retreated",
        "players",
        "valid",
        "boundary",
        "transition_after_immediate",
    }
    return {name: copy.deepcopy(value[name]) for name in allowed if name in value}


def _policy_stage_option_features(
    snapshot: Any,
    observation: Mapping[str, Any],
    action: list[int],
) -> list[dict[str, Any]]:
    try:
        from . import features

        stages = features.factorized_teacher_forcing_stages(dict(observation), list(action))
    except Exception as exc:
        raise SourceRecordError("could not reconstruct teacher-forced policy stages") from exc
    output: list[dict[str, Any]] = []
    for stage_index, raw_stage in enumerate(stages):
        if not isinstance(raw_stage, tuple) or len(raw_stage) != 2:
            raise SourceRecordError("teacher-forced stage shape is invalid")
        raw_combos, selected_index = raw_stage
        if not isinstance(raw_combos, list):
            raise SourceRecordError("teacher-forced action combos are invalid")
        combos = [_action_indices(combo) for combo in raw_combos]
        selected = _exact_int(selected_index)
        if selected is None or not 0 <= selected < len(combos):
            raise SourceRecordError("teacher-forced selected index is invalid")
        matrix = snapshot.option_features(observation, combos)
        if len(matrix) != len(combos) or any(len(row) != OPTION_FEATURE_DIM for row in matrix):
            raise SourceRecordError("ledger option feature ABI drifted")
        output.append(
            {
                "stage_index": stage_index,
                "action_combos_fingerprint": sha256_bytes(canonical_json_bytes(combos)),
                "candidate_count": len(combos),
                "selected_index": selected,
                "ledger_option_features": [list(row) for row in matrix],
            }
        )
    if not output:
        raise SourceRecordError("teacher-forced stage list is empty")
    return output


def _masked_public_observation(value: Any, *, expected_seat: int | None) -> dict[str, Any]:
    """Return a policy-visible copy with hidden deck/opponent fields stripped."""

    if not isinstance(value, Mapping):
        raise SourceRecordError("observation is not an object")
    try:
        from .replay_import import _strip_opp_private  # type: ignore[attr-defined]

        masked, _aux, report = _strip_opp_private(copy.deepcopy(dict(value)))
    except Exception as exc:
        raise SourceRecordError("could not establish an information-set-safe observation") from exc
    if not isinstance(masked, dict) or not report.ok:
        raise SourceRecordError("observation cannot be remasked into a safe information set")
    current = masked.get("current")
    if not isinstance(current, dict):
        raise SourceRecordError("observation lacks current state")
    actor = _seat(current.get("yourIndex"), label="observation current.yourIndex")
    if expected_seat is not None and actor != expected_seat:
        raise SourceRecordError("observation actor does not match record seat")
    players = current.get("players")
    if not isinstance(players, list) or len(players) != 2 or not all(isinstance(row, dict) for row in players):
        raise SourceRecordError("observation does not contain two player states")
    for index, player in enumerate(players):
        # Deck ordering is never policy-visible, even for the acting player.
        player.pop("deck", None)
        for key in (
            "deckOrder",
            "deck_order",
            "trueDeck",
            "true_deck",
            "hiddenDeck",
            "hidden_deck",
            "prizeOrder",
            "prize_cards",
            "hiddenPrize",
            "truePrize",
        ):
            player.pop(key, None)
        if index != actor:
            # Do not retain an accidental opponent hand/prize identity in a
            # hash input either.  Preserve only public cardinality.
            player["hand"] = None
            prize = player.get("prize")
            if isinstance(prize, list):
                player["prize"] = [None] * len(prize)
    masked.pop("transition_after", None)
    masked.pop("aux_labels", None)
    masked.pop("visualize", None)
    # These names are privileged simulator/search surfaces.  Even an ignored
    # field would be a leak here because the observation fingerprint commits
    # this exact mapping.  Retain only normal policy-observation surfaces.
    for field in (
        "search_begin_input",
        "searchBeginInput",
        "hidden_state",
        "hiddenState",
        "simulator_state",
        "simulatorState",
        "state_snapshot",
        "stateSnapshot",
        "full_state",
        "fullState",
        "private_state",
        "privateState",
        "raw_payload",
        "rawPayload",
        "deck",
        "deckOrder",
        "deck_order",
        "prizeOrder",
        "prize_order",
        "hiddenPrize",
        "hidden_prize",
    ):
        masked.pop(field, None)
    return masked


def _public_observation_fingerprint(observation: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(observation))


def _build_identity(
    *,
    mode: str,
    source_snapshot_path: str,
    source_snapshot_tree_sha256: str,
    image_tag: str,
    image_id: str,
    classifier: Mapping[str, Any] | None,
    protected_stream_sha256: str | None,
) -> _BuildIdentity:
    if mode not in {"archive_native", "protected_jsonl"}:
        raise OwnDeckRolloutStoreError("unknown r259 source mode")
    _require_sha256(source_snapshot_tree_sha256, label="source snapshot tree checksum")
    if image_tag != EXPECTED_IMAGE_TAG:
        raise OwnDeckRolloutStoreError("r259 image tag identity drifted")
    if image_id != EXPECTED_IMAGE_ID:
        raise OwnDeckRolloutStoreError("r259 image ID identity drifted")
    if protected_stream_sha256 is not None:
        _require_sha256(protected_stream_sha256, label="protected stream checksum")
    module_root = Path(__file__).resolve().parent
    code_paths = {
        "own_deck_rollout_store.py": Path(__file__).resolve(),
        "own_deck_ledger.py": module_root / "own_deck_ledger.py",
        "own_deck_supervision.py": module_root / "own_deck_supervision.py",
    }
    code = {name: sha256_file(path, label=f"r259 code {name}") for name, path in code_paths.items()}
    return _BuildIdentity(
        mode=mode,
        source_snapshot_path=str(source_snapshot_path),
        source_snapshot_tree_sha256=source_snapshot_tree_sha256,
        image_tag=image_tag,
        image_id=image_id,
        code_identities=code,
        classifier=classifier,
        protected_stream_sha256=protected_stream_sha256,
    )


def _expected_daily_identity(window: SourceWindow, archive: SourceArchive, identity: _BuildIdentity) -> dict[str, Any]:
    return {
        "source": {
            "manifest": {
                "original_path": window.original_manifest_path,
                "locked_path": str(window.manifest_path),
                "sha256": window.manifest_sha256,
                "schema": ARCHIVE_RECEIPT_SCHEMA,
                "window_start": WINDOW_START,
                "window_end": WINDOW_END,
                "days": WINDOW_DAYS,
                "total_episodes": WINDOW_TOTAL_EPISODES,
            },
            "versioned_receipt": {
                "path": window.original_versioned_receipt_path,
                "original_path": window.original_versioned_receipt_path,
                "locked_path": (
                    None
                    if window.versioned_receipt_path is None
                    else str(window.versioned_receipt_path)
                ),
                "sha256": window.versioned_receipt_sha256,
            },
            "archive": archive.metadata(),
        },
        "build": identity.metadata(),
    }


def _validate_daily_directory(directory: Path, *, expected_meta_sha256: str | None = None) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise ImmutableSidecarError(f"daily sidecar directory is absent or unsafe: {directory}")
    allowed = {DAILY_SHARD_NAME, DAILY_META_NAME}
    actual = {child.name for child in directory.iterdir()}
    if actual != allowed:
        raise ImmutableSidecarError(f"daily sidecar directory is incomplete or has unexpected members: {directory}")
    shard = _regular_file(directory / DAILY_SHARD_NAME, label="daily sidecar shard")
    meta_path = _regular_file(directory / DAILY_META_NAME, label="daily sidecar meta")
    meta = _read_json_object(meta_path, label="daily sidecar meta")
    if meta.get("schema") != OWN_DECK_ROLLOUT_DAILY_META_SCHEMA or _exact_int(meta.get("version")) != OWN_DECK_ROLLOUT_DAILY_META_VERSION:
        raise ImmutableSidecarError("daily sidecar meta schema mismatch")
    expected_digest = _meta_digest(meta)
    if meta.get("meta_sha256") != expected_digest:
        raise ImmutableSidecarError("daily sidecar meta self checksum drifted")
    if expected_meta_sha256 is not None and meta.get("meta_sha256") != expected_meta_sha256:
        raise ImmutableSidecarError("daily sidecar meta checksum does not match requested identity")
    expected_shard_sha = meta.get("shard_sha256")
    _require_sha256(expected_shard_sha, label="daily shard checksum")
    if sha256_file(shard, label="daily sidecar shard") != expected_shard_sha:
        raise ImmutableSidecarError("daily sidecar shard checksum drifted")
    shard_info = meta.get("shard")
    if not isinstance(shard_info, Mapping) or shard_info.get("sha256") != expected_shard_sha:
        raise ImmutableSidecarError("daily sidecar shard metadata drifted")
    if _exact_int(shard_info.get("bytes")) != shard.stat().st_size:
        raise ImmutableSidecarError("daily sidecar shard byte count drifted")
    _validate_outcome_label_accounting(meta)
    build = meta.get("build")
    if isinstance(build, Mapping) and build.get("mode") == "archive_native":
        _validate_archive_native_accounting(meta)
    return meta


def _validate_outcome_label_accounting(meta: Mapping[str, Any]) -> None:
    labels = meta.get("label_counts")
    outcome = labels.get("outcome_provenance") if isinstance(labels, Mapping) else None
    if not isinstance(outcome, Mapping):
        raise ImmutableSidecarError("daily outcome label accounting is missing")
    verified = _exact_int(outcome.get("verified_reward_rows"))
    masked = _exact_int(outcome.get("masked_invalid_or_missing_reward_rows"))
    row_count = _exact_int(meta.get("row_count"))
    if (
        verified is None
        or masked is None
        or row_count is None
        or verified < 0
        or masked < 0
        or verified + masked != row_count
    ):
        raise ImmutableSidecarError("daily outcome label accounting drifted")


def _validate_archive_native_accounting(meta: Mapping[str, Any]) -> None:
    accounting = meta.get("archive_native_accounting")
    if not isinstance(accounting, Mapping):
        raise ImmutableSidecarError("archive-native accounting is missing")
    expected_names = {
        "episodes_seen",
        "verified_reward_episodes",
        "invalid_or_missing_reward_episodes",
        "invalid_reward_fallback_records",
        "records_emitted",
        "records_skipped_stale_only",
        "invalid_reward_episodes_skipped_unconvertible",
    }
    if accounting.get("schema") != "poke_bot.own_deck_rollout_archive_native_accounting/v1" or set(accounting) != expected_names | {"schema"}:
        raise ImmutableSidecarError("archive-native accounting schema drifted")
    values = {name: _exact_int(accounting.get(name)) for name in expected_names}
    if any(value is None or value < 0 for value in values.values()):
        raise ImmutableSidecarError("archive-native accounting has an invalid count")
    if values["episodes_seen"] != (
        values["verified_reward_episodes"] + values["invalid_or_missing_reward_episodes"]
    ):
        raise ImmutableSidecarError("archive-native reward accounting drifted")
    if values["records_emitted"] != _exact_int(meta.get("source_record_count")):
        raise ImmutableSidecarError("archive-native emitted-record accounting drifted")


def _require_existing_identity(meta: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if meta.get("status") != "complete_immutable_sidecar":
        raise ImmutableSidecarError("existing sidecar is not a complete immutable output")
    if meta.get("source") != expected.get("source") or meta.get("build") != expected.get("build"):
        raise ImmutableSidecarError("existing sidecar identity differs; refusing to overwrite it")
    if meta.get("training_eligibility", {}).get("active_r241") is not False:
        raise ImmutableSidecarError("existing sidecar has unsafe r241 eligibility")


def _validate_sidecar_row_shape(row: Mapping[str, Any], *, day_text: str, meta: Mapping[str, Any]) -> None:
    if row.get("schema") != OWN_DECK_ROLLOUT_SIDECAR_SCHEMA or _exact_int(row.get("version")) != OWN_DECK_ROLLOUT_SIDECAR_VERSION:
        raise ImmutableSidecarError("sidecar row schema mismatch")
    if row.get("source_date") != day_text or row.get("source_manifest_sha256") != meta.get("source", {}).get("manifest", {}).get("sha256"):
        raise ImmutableSidecarError("sidecar row source binding drifted")
    if _seat(row.get("seat"), label="sidecar row seat") not in (0, 1):
        raise ImmutableSidecarError("sidecar row seat is invalid")
    if _exact_int(row.get("env_step")) is None:
        raise ImmutableSidecarError("sidecar row env_step is invalid")
    if not isinstance(row.get("ledger_snapshot"), Mapping):
        raise ImmutableSidecarError("sidecar row lacks ledger snapshot")
    if row.get("deck_fingerprint") != row["ledger_snapshot"].get("deck_fingerprint"):
        raise ImmutableSidecarError("sidecar row deck fingerprint does not match ledger")
    supervision = row.get("supervision")
    if not isinstance(supervision, Mapping) or supervision.get("target_only") is not True:
        raise ImmutableSidecarError("sidecar row supervision is not target-only")
    eligibility = row.get("training_eligibility")
    if not isinstance(eligibility, Mapping) or eligibility.get("active_r241") is not False:
        raise ImmutableSidecarError("sidecar row is unsafe for current r241")
    _assert_no_private_output(row)


def _assert_no_private_output(value: Mapping[str, Any]) -> None:
    """Reject serialized names that could carry raw/private replay payloads."""

    forbidden = {
        "observation",
        "action",
        "transition_after",
        "aux_labels",
        "opp_deck",
        "opp_hand",
        "opp_hidden_remainder",
        "deck_order",
        "deckOrder",
        "visualize",
        "raw_payload",
    }

    def visit(item: Any, *, allow_labels: bool = False) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if key in forbidden:
                    raise SourceRecordError(f"sidecar attempted to serialize forbidden field {key!r}")
                visit(nested, allow_labels=allow_labels or key == "labels")
        elif isinstance(item, list):
            for nested in item:
                visit(nested, allow_labels=allow_labels)

    visit(value)


def _empty_label_counts() -> dict[str, Any]:
    return {
        "terminal_conversion": {
            "terminal_class": {name: 0 for name in TERMINAL_CONVERSION_CLASSES},
            "terminal_class_labeled": 0,
            "prize_closeout": {"labeled": 0, "positive": 0},
            "opponent_knockout": {"labeled": 0, "positive": 0},
        },
        "visible_tutor_completion": {
            "visible_tutor_stages": 0,
            "selected_from_visible_deck": {"labeled": 0, "positive": 0},
            "selected_target_observed_after_action": {"labeled": 0, "positive": 0},
            "same_actor_followup": {"labeled": 0, "positive": 0},
            "same_actor_terminal_class": {name: 0 for name in TERMINAL_CONVERSION_CLASSES},
            "same_actor_terminal_class_labeled": 0,
        },
        "ledger": {"integrity_ok": 0, "fail_closed": 0},
        "outcome_provenance": {
            "verified_reward_rows": 0,
            "masked_invalid_or_missing_reward_rows": 0,
        },
        "joinable_policy_stage_count": 0,
    }


def _mask_unverified_outcome_labels(
    terminal_labels: Mapping[str, Any],
    tutor_labels: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mask every reward/outcome target while retaining safe tutor facts.

    A malformed top-level reward pair cannot attest either terminal class or
    terminal-adjacent prize/KO facts.  Visible-menu selection and immediate
    observed-card completion remain causal public facts, so this deliberately
    narrows only the outcome-bearing target fields.
    """

    terminal = dict(terminal_labels)
    terminal["terminal_class"] = {"value": 0, "mask": False}
    terminal["prize_closeout"] = {"value": 0.0, "mask": False}
    terminal["opponent_knockout"] = {"value": 0.0, "mask": False}
    terminal["provenance"] = "masked_invalid_or_missing_episode_rewards"
    tutor = dict(tutor_labels)
    tutor["same_actor_terminal_class"] = {"value": 0, "mask": False}
    return terminal, tutor


def _accumulate_label_counts(counts: dict[str, Any], row: Mapping[str, Any]) -> None:
    snapshot = row["ledger_snapshot"]
    if snapshot.get("integrity_ok") is True:
        counts["ledger"]["integrity_ok"] += 1
    if snapshot.get("fail_closed") is True:
        counts["ledger"]["fail_closed"] += 1
    stages = row.get("policy_stage_option_features")
    if isinstance(stages, list):
        counts["joinable_policy_stage_count"] += len(stages)
    supervision = row["supervision"]
    terminal = supervision["terminal_conversion"]
    tutor = supervision["visible_tutor_completion"]
    _count_categorical(
        counts["terminal_conversion"],
        "terminal_class",
        terminal.get("labels", {}).get("terminal_class"),
    )
    _count_scalar(
        counts["terminal_conversion"],
        "prize_closeout",
        terminal.get("labels", {}).get("prize_closeout"),
    )
    _count_scalar(
        counts["terminal_conversion"],
        "opponent_knockout",
        terminal.get("labels", {}).get("opponent_knockout"),
    )
    tutor_labels = tutor.get("labels", {})
    visible = tutor_labels.get("selected_from_visible_deck")
    if isinstance(visible, Mapping) and visible.get("mask") is True and bool(visible.get("value")):
        counts["visible_tutor_completion"]["visible_tutor_stages"] += 1
    for name in VISIBLE_TUTOR_COMPLETION_SCALAR_TARGET_NAMES:
        _count_scalar(counts["visible_tutor_completion"], name, tutor_labels.get(name))
    _count_categorical(
        counts["visible_tutor_completion"],
        "same_actor_terminal_class",
        tutor_labels.get("same_actor_terminal_class"),
    )


def _count_scalar(target: dict[str, Any], name: str, label: Any) -> None:
    if not isinstance(label, Mapping) or label.get("mask") is not True:
        return
    value = label.get("value")
    if isinstance(value, (bool, int, float)):
        numeric = float(value)
    else:
        return
    target[name]["labeled"] += 1
    if numeric > 0.0:
        target[name]["positive"] += 1


def _count_categorical(target: dict[str, Any], name: str, label: Any) -> None:
    if not isinstance(label, Mapping) or label.get("mask") is not True:
        return
    index = _exact_int(label.get("value"))
    if index is None or not 0 <= index < len(TERMINAL_CONVERSION_CLASSES):
        return
    target[f"{name}_labeled"] += 1
    target[name][TERMINAL_CONVERSION_CLASSES[index]] += 1


def _merge_label_counts(target: dict[str, Any], other: Mapping[str, Any]) -> None:
    for family, value in other.items():
        if isinstance(value, Mapping):
            _merge_label_counts(target[family], value)
        elif isinstance(value, int):
            target[family] += value
        else:  # pragma: no cover - internal count schema is fixed
            raise OwnDeckRolloutStoreError("label-count schema is malformed")


def _source_day_from_record(record: Mapping[str, Any]) -> str:
    explicit = record.get("source_date")
    if isinstance(explicit, str) and _DAY_RE.fullmatch(explicit):
        return explicit
    source = str(record.get("source") or "")
    match = _DAY_RE.search(source)
    if match is None:
        raise SourceRecordError("source record has no exact archive day identity")
    result = match.group(1)
    _validate_day(result)
    return result


def _public_record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Bounded source-row identity used only in daily metadata digests."""

    steps = record.get("steps")
    rows: list[dict[str, Any]] = []
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            observation = _masked_public_observation(
                step.get("observation"),
                expected_seat=_seat(record.get("seat"), label="record seat"),
            )
            action = _action_indices(step.get("action"))
            env_step = _exact_int(step.get("env_step"))
            if env_step is None:
                raise SourceRecordError("source record identity env_step is invalid")
            rows.append(
                {
                    "env_step": env_step,
                    "observation_fingerprint": _public_observation_fingerprint(observation),
                    "action": action,
                }
            )
    return {
        "episode_id": _required_text(record.get("episode_id"), label="episode_id"),
        "seat": _seat(record.get("seat"), label="record seat"),
        "archetype": str(record.get("archetype") or "").casefold(),
        "source_day": _source_day_from_record(record),
        "deck_fingerprint": sha256_bytes(canonical_json_bytes(sorted(_starting_deck(record.get("deck"))))),
        "steps": rows,
    }


def _load_classifier(mix: Path | str, representatives: Path | str, card_csv: Path | str) -> Any:
    try:
        from .ladder_replay import LadderReplayClassifier

        return LadderReplayClassifier.from_paths(mix, representatives, card_csv=card_csv)
    except Exception as exc:
        raise SourceManifestError("could not load the pinned Alakazam replay classifier") from exc


def _classifier_contract(classifier: Any, *, mix_path: Path, representatives_path: Path, card_csv: Path) -> dict[str, Any]:
    contract = getattr(classifier, "contract", None)
    if not isinstance(contract, Mapping):
        raise SourceManifestError("classifier has no immutable contract")
    return {
        "contract": copy.deepcopy(dict(contract)),
        "mix": {"path": str(mix_path), "sha256": sha256_file(mix_path, label="classifier mix")},
        "representatives": {
            "path": str(representatives_path),
            "sha256": sha256_file(representatives_path, label="classifier representatives"),
        },
        "card_csv": {"path": str(card_csv), "sha256": sha256_file(card_csv, label="classifier card CSV")},
    }


def _selected_days(window: SourceWindow, only_days: Iterable[str] | None) -> tuple[str, ...]:
    if only_days is None:
        return window.dates
    requested = tuple(dict.fromkeys(str(day) for day in only_days))
    if not requested:
        raise SourceManifestError("--only-day selection is empty")
    for day_text in requested:
        _validate_day(day_text)
        window.archive_for_day(day_text)
    return requested


def _validate_versioned_receipt_equivalence(current: Mapping[str, Any], versioned: Mapping[str, Any]) -> None:
    for key in (
        "schema",
        "status",
        "window_policy",
        "window_start",
        "window_end",
        "days",
        "archives",
        "total_episodes",
    ):
        if current.get(key) != versioned.get(key):
            raise SourceManifestError("versioned receipt does not exactly bind the protected archive window")


def _meta_digest(meta: Mapping[str, Any]) -> str:
    detached = dict(meta)
    detached.pop("meta_sha256", None)
    return sha256_bytes(canonical_json_bytes(detached))


def _write_atomic_file(path: Path, body: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ImmutableSidecarError(f"refusing to replace an immutable output member: {path}")
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _exclusive_day_lock(output_root: Path, day_text: str) -> Iterator[None]:
    """Serialize validation/build/publish for one immutable daily target.

    The lock lives outside ``daily/<day>`` so a committed directory retains
    exactly its shard and metadata members.  A second managed-service start
    blocks before it can create a temporary directory; when it acquires the
    lock it validates and reuses the first completed immutable shard instead
    of replacing it.
    """

    _validate_day(day_text)
    lock_root = output_root / ".r259-locks"
    try:
        lock_root.mkdir(mode=0o700)
    except FileExistsError:
        # A concurrent managed invocation may have created it first.
        pass
    if lock_root.is_symlink() or not lock_root.is_dir():
        raise ImmutableSidecarError("r259 lock root is unsafe")
    lock_path = lock_root / f"{day_text}.lock"
    try:
        if not lock_path.exists() and not lock_path.is_symlink():
            create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                create_flags |= os.O_NOFOLLOW
            descriptor = os.open(lock_path, create_flags, 0o600)
            os.close(descriptor)
    except FileExistsError:
        # Another process won creation; its regular-file safety is checked
        # below before taking the advisory lock.
        pass
    lock_file = _regular_file(lock_path, label="r259 daily lock")
    open_flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_file, open_flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, OwnDeckRolloutStoreError) as exc:
        raise OwnDeckRolloutStoreError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise OwnDeckRolloutStoreError(f"{label} must be a JSON object")
    return value


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects instead of silently taking the last key."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OwnDeckRolloutStoreError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _regular_file(path: Path | str, *, label: str) -> Path:
    source = Path(path)
    try:
        information = source.lstat()
    except OSError as exc:
        raise OwnDeckRolloutStoreError(f"{label} is missing: {source}") from exc
    if stat.S_ISLNK(information.st_mode) or not stat.S_ISREG(information.st_mode):
        raise OwnDeckRolloutStoreError(f"{label} must be a non-symlink regular file: {source}")
    return source


def _regular_or_new_directory(path: Path, *, label: str) -> Path:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_dir():
            raise ImmutableSidecarError(f"{label} must be a non-symlink directory")
        return path
    path.mkdir(parents=True, exist_ok=False)
    return path


def _validate_day(value: str) -> None:
    if value not in expected_window_dates():
        raise SourceManifestError(f"day is not in exact r241 window: {value}")


def _require_sha256(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OwnDeckRolloutStoreError(f"{label} is not a sha256 digest")


def _exact_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        return parsed if bool(value == parsed) else None
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def _seat(value: Any, *, label: str) -> int:
    parsed = _exact_int(value)
    if parsed not in (0, 1):
        raise SourceRecordError(f"{label} must be exact seat 0 or 1")
    return parsed


def _seat_or_none(value: Any) -> int | None:
    parsed = _exact_int(value)
    return parsed if parsed in (0, 1) else None


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceRecordError(f"{label} must be a non-empty string")
    return value


def _starting_deck(value: Any) -> list[int]:
    if not isinstance(value, list) or len(value) != 60:
        raise SourceRecordError("source record must carry an exact 60-card own starting deck")
    result: list[int] = []
    for card in value:
        parsed = _exact_int(card)
        if parsed is None or parsed < 0:
            raise SourceRecordError("source record starting deck has an invalid card id")
        result.append(parsed)
    return result


def _action_indices(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise SourceRecordError("source action must be an ordered option-index list")
    output: list[int] = []
    for item in value:
        parsed = _exact_int(item)
        if parsed is None or parsed < 0:
            raise SourceRecordError("source action has an invalid option index")
        output.append(parsed)
    if len(output) != len(set(output)):
        raise SourceRecordError("source action repeats an option index")
    return output


__all__ = [
    "ARCHIVE_RECEIPT_SCHEMA",
    "DAILY_DIRECTORY_NAME",
    "DAILY_META_NAME",
    "DAILY_SHARD_NAME",
    "EXPECTED_IMAGE_ID",
    "EXPECTED_IMAGE_TAG",
    "EXPECTED_SMOKE_CG_RUNTIME",
    "EXPECTED_SOURCE_MANIFEST_SHA256",
    "EXPECTED_VERSIONED_RECEIPT_SHA256",
    "OPTION_FEATURE_DIM",
    "OWNER_DECISION_REVISION",
    "OWN_DECK_ROLLOUT_DAILY_META_SCHEMA",
    "OWN_DECK_ROLLOUT_DAILY_META_VERSION",
    "OWN_DECK_ROLLOUT_SIDECAR_SCHEMA",
    "OWN_DECK_ROLLOUT_SIDECAR_VERSION",
    "SMOKE_UNKNOWN_OUTCOME_EPISODE_ID",
    "SMOKE_UNKNOWN_OUTCOME_MEMBER",
    "WINDOW_DAYS",
    "WINDOW_END",
    "WINDOW_START",
    "DailySidecarResult",
    "ImmutableSidecarError",
    "OwnDeckRolloutStoreError",
    "SourceArchive",
    "SourceManifestError",
    "SourceRecordError",
    "SourceWindow",
    "board_feature_fingerprint",
    "build_archive_native_sidecar",
    "build_protected_jsonl_sidecar",
    "canonical_json_bytes",
    "expected_window_dates",
    "iter_daily_sidecar_rows",
    "load_source_window",
    "materialize_record_sidecar_rows",
    "read_daily_meta",
    "sha256_bytes",
    "sha256_file",
    "smoke_archive_native_one_record",
    "smoke_archive_native_unknown_outcome_member",
]
