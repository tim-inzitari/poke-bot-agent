"""Receipt-backed complete-action corpus materialization for Alakazam RTP r197.

This module deliberately sits *before* model encoding and has no parent
checkpoint argument.  It replays only the episode/seat identities already
selected by the protected Alakazam expert pointer, reconstructs the current
information-set-safe visual trace, and stores the complete ordered action set
used by :class:`~poke_bot.recursive_turn_planner.agent_bridge.RTPAgentBridge`.

The output is a small collection of streamable JSONL files plus an immutable
manifest and receipt.  It is intentionally not a general replay importer:
evaluation/Kaggle replay rows are never admitted here, factorized prefixes are
never substituted for a complete action, and a sidecar or policy parent is
never loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from poke_bot import archetypes, features
from poke_bot.authoritative_visual_trace import VisualEpisodeResult, convert_visual_episode
from poke_bot.feature_shards import SHARD_FORMAT, SHARD_FORMAT_VERSION
from poke_bot.replay_import import episode_id_of, extract_setup_decks
from scripts.extract_verified_specialist_records import (
    _RuleClassifier,
    iter_legacy_identities,
)


CORPUS_SCHEMA = "poke_bot.rtp_complete_action_shadow_corpus/v1"
MANIFEST_SCHEMA = "poke_bot.rtp_complete_action_shadow_corpus_manifest/v1"
RECEIPT_SCHEMA = "poke_bot.rtp_complete_action_shadow_corpus_receipt/v1"
ROW_SCHEMA = "poke_bot.rtp_complete_ordered_action_row/v1"
IDENTITY_SCHEMA = "poke_bot.rtp_complete_action_verified_identity/v1"
SPLIT_SCHEMA = "poke_bot.rtp_complete_action_episode_split/v1"
ACTION_SPACE_SCHEMA = "poke_bot.rtp_complete_ordered_legal_actions/v1"
ACTION_SPACE_TOO_LARGE_SCHEMA = "poke_bot.rtp_action_space_too_large/v1"
PROTECTED_IDENTITY_EXACT_SCHEMA = "poke_bot.rtp_protected_identity_exact/v1"

SPECIALIST_ID = "alakazam"
SPLIT_SEED = 5_000_000
MAX_ACTION_COMBOS = 1024
DEFAULT_HELDOUT_FRACTION = "0.20"
PRODUCTION_ARCHIVE_ROOT = Path("/home/pokebot/poke-bot-agent/data/episodes/raw")
# These exact hashes mirror the typed r197 owner contract.  They are kept in
# this new materializer so a caller cannot generate a similarly shaped corpus
# from a nearby archive window.
R197_RAW_ARCHIVE_SHA256_BY_DAY: dict[str, str] = {
    "2026-08-01": "sha256:1ba104fca133b096655ef385b16f630218ad41fe561e14edbc74b79e3f5b2cff",
    "2026-08-02": "sha256:fa91e058a42d5fffab0f3e63f04fba5acc9bfbd2e2225e97aa62f45f5d430eb8",
    "2026-08-03": "sha256:909cbd205f3afcfde6031ae93ef9625b796e8a0c2edf66eeb6edc88469273a04",
    "2026-08-04": "sha256:17cd9cd92f4ae3b293ee3fab3452657316362af134c6d4a7b5dbfda99c3d3d42",
    "2026-08-05": "sha256:ab961e0d98984b611cc4091801b618606cb03cab4413ab7908d3f8c6312030e3",
}
R197_SOURCE_DAYS = tuple(sorted(R197_RAW_ARCHIVE_SHA256_BY_DAY))
SPLIT_RULE = "sha256:r197-episode-split/v1\\0{seed}\\0{episode_id}"
CANONICAL_ACTION_ORDER = (
    "selection_length_ascending_then_itertools_permutations_option_index_ascending"
)

TRAIN_FILENAME = "train.complete-actions.jsonl"
HELDOUT_FILENAME = "heldout.complete-actions.jsonl"
TOO_LARGE_FILENAME = "action-space-too-large.jsonl"
IDENTITIES_FILENAME = "verified-episode-seats.jsonl"
SPLITS_FILENAME = "episode-splits.jsonl"
MANIFEST_FILENAME = "MANIFEST.json"
RECEIPT_FILENAME = "RECEIPT.json"


class R197CorpusError(RuntimeError):
    """The r197 complete-action corpus contract was violated."""


@dataclass(frozen=True)
class StreamFile:
    """One sealed JSONL file described by the manifest."""

    path: str
    sha256: str
    bytes: int
    rows: int

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "rows": self.rows,
        }


class _JsonlWriter:
    """Bounded-memory canonical JSONL writer with a streaming digest."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")
        self._digest = hashlib.sha256()
        self._bytes = 0
        self._rows = 0
        self._closed = False

    def write(self, value: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("cannot write a closed JSONL stream")
        raw = canonical_json_bytes(value)
        self._stream.write(raw.decode("utf-8"))
        self._digest.update(raw)
        self._bytes += len(raw)
        self._rows += 1

    def close(self) -> StreamFile:
        if not self._closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            self._closed = True
        return StreamFile(
            path=self.path.name,
            sha256="sha256:" + self._digest.hexdigest(),
            bytes=int(self._bytes),
            rows=int(self._rows),
        )


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical UTF-8 JSON used for every r197 digest and JSONL line."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the content digest of ``path`` without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R197CorpusError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise R197CorpusError(f"expected JSON object: {path}")
    return value


def _normalise_specialist_id(value: str) -> str:
    result = str(value).strip().casefold()
    if result != SPECIALIST_ID:
        raise R197CorpusError(
            f"r197 complete-action materializer is fixed to {SPECIALIST_ID!r}; "
            f"got {value!r}"
        )
    return result


def _normalise_heldout_fraction(value: str | float | Decimal) -> tuple[Decimal, str]:
    try:
        fraction = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise R197CorpusError(f"invalid heldout fraction: {value!r}") from exc
    if not Decimal("0") < fraction < Decimal("1"):
        raise R197CorpusError("heldout_fraction must be strictly between zero and one")
    # Decimal normalization can produce scientific notation.  The explicit
    # fixed-point string becomes part of the receipt and split identity.
    rendered = format(fraction, "f")
    return fraction, rendered


def _split_threshold(fraction: Decimal) -> int:
    """Convert a decimal fraction into an exact unsigned-64 threshold."""

    numerator, denominator = fraction.as_integer_ratio()
    return (int(numerator) * (1 << 64)) // int(denominator)


def deterministic_episode_split(
    episode_id: str,
    *,
    split_seed: int = SPLIT_SEED,
    heldout_fraction: str | float | Decimal = DEFAULT_HELDOUT_FRACTION,
) -> dict[str, Any]:
    """Return the source-disjoint r197 split assignment for one episode.

    This happens before an episode is converted or any decision is encoded.
    Both acting seats of the same episode consequently get the same split.
    """

    if int(split_seed) != SPLIT_SEED:
        raise R197CorpusError(
            f"r197 split seed is fixed at {SPLIT_SEED}; got {split_seed}"
        )
    normalized_episode_id = str(episode_id)
    if not normalized_episode_id:
        raise R197CorpusError("episode_id cannot be empty for r197 split")
    fraction, rendered_fraction = _normalise_heldout_fraction(heldout_fraction)
    material = (
        f"r197-episode-split/v1\0{SPLIT_SEED}\0{normalized_episode_id}".encode(
            "utf-8"
        )
    )
    digest = hashlib.sha256(material).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False)
    split = "heldout" if bucket < _split_threshold(fraction) else "train"
    return {
        "episode_id": normalized_episode_id,
        "split": split,
        "split_seed": SPLIT_SEED,
        "heldout_fraction": rendered_fraction,
        "bucket_u64": str(bucket),
        "split_hash": "sha256:" + hashlib.sha256(material).hexdigest(),
        "rule": SPLIT_RULE,
    }


def _normalise_actions(actions: Sequence[Sequence[int]]) -> list[list[int]]:
    try:
        return [[int(option) for option in action] for action in actions]
    except (TypeError, ValueError, OverflowError) as exc:
        raise R197CorpusError("legal actions must be sequences of integers") from exc


def _action_space_descriptor(
    observation: Mapping[str, Any],
    legal_actions: Sequence[Sequence[int]],
    *,
    max_action_combos: int,
) -> dict[str, Any]:
    """Build the exact payload consumed by the public fingerprint helper."""

    if int(max_action_combos) != MAX_ACTION_COMBOS:
        raise R197CorpusError(
            f"r197 max_action_combos is fixed at {MAX_ACTION_COMBOS}; "
            f"got {max_action_combos}"
        )
    select = observation.get("select") if isinstance(observation, Mapping) else None
    if not isinstance(select, Mapping):
        raise R197CorpusError("causal observation has no select object")
    options = select.get("option")
    if not isinstance(options, list):
        raise R197CorpusError("causal observation select.option is not a list")
    normalised = _normalise_actions(legal_actions)
    # The bounds are re-derived by the canonical feature implementation.  We
    # deliberately do not trust a caller-provided count or prefix structure.
    try:
        expected = features.enumerate_action_combos(
            dict(observation), max_combos=MAX_ACTION_COMBOS
        )
    except features.ActionSpaceTooLarge:
        raise
    except Exception as exc:
        raise R197CorpusError("cannot enumerate canonical complete action space") from exc
    expected_actions = [list(action) for action in expected]
    if normalised != expected_actions:
        raise R197CorpusError(
            "legal actions do not equal canonical complete ordered action support"
        )
    return {
        "schema": ACTION_SPACE_SCHEMA,
        "builder": "poke_bot.features.enumerate_action_combos",
        "representation": "complete_ordered_action_combinations",
        "canonical_order": CANONICAL_ACTION_ORDER,
        "max_action_combos": MAX_ACTION_COMBOS,
        "observation_fingerprint": canonical_json_sha256(dict(observation)),
        "legal_option_count": len(options),
        "min_count": int(expected.min_count),
        "max_count": int(expected.max_count),
        "complete_ordered_action_count": int(expected.total_count),
        "complete_ordered_actions": normalised,
    }


def complete_action_space_fingerprint(
    observation: Mapping[str, Any],
    legal_actions: Sequence[Sequence[int]],
    *,
    max_action_combos: int = MAX_ACTION_COMBOS,
) -> str:
    """Fingerprint an exact r197 complete ordered action space.

    Public callers (notably the r197 trainer) should recompute the legal
    actions from the observation with ``features.enumerate_action_combos`` and
    pass them here.  This rejects a factorized stage, a reordered action list,
    or a complete-action mismatch rather than normalizing it silently.
    """

    return canonical_json_sha256(
        _action_space_descriptor(
            observation, legal_actions, max_action_combos=max_action_combos
        )
    )


def _source_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        root / "scripts/extract_verified_specialist_records.py",
        root / "poke_bot/authoritative_visual_trace.py",
        root / "poke_bot/features.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise R197CorpusError("required r197 generator source is missing: " + ", ".join(missing))
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in paths
    }


def _pointer_metadata(pointer_path: Path) -> dict[str, Any]:
    pointer = _read_json_object(pointer_path)
    if (
        pointer.get("schema") != "poke_bot.pinned_expert_corpus/v1"
        or pointer.get("protected") is not True
    ):
        raise R197CorpusError("r197 source pointer is not a protected expert corpus")
    manifest_name = str(pointer.get("manifest") or "")
    manifest = (pointer_path.parent / manifest_name).resolve()
    expected_manifest_digest = str(pointer.get("manifest_sha256") or "")
    if not manifest_name or not manifest.is_file() or not expected_manifest_digest:
        raise R197CorpusError("r197 protected pointer lacks a bound manifest")
    actual_manifest_digest = sha256_file(manifest)
    if actual_manifest_digest != expected_manifest_digest:
        raise R197CorpusError("r197 protected pointer manifest digest changed")
    if str(pointer.get("specialist_id") or "").strip().casefold() != SPECIALIST_ID:
        raise R197CorpusError("r197 protected pointer is not fixed to Alakazam")
    return {
        "path": str(pointer_path),
        "sha256": sha256_file(pointer_path),
        "manifest_path": str(manifest),
        "manifest_sha256": actual_manifest_digest,
    }


def _required_sha256(value: Any, *, label: str) -> str:
    """Validate the explicit lower-case SHA-256 receipt representation."""

    digest = str(value or "")
    hex_part = digest.removeprefix("sha256:")
    if (
        not digest.startswith("sha256:")
        or len(hex_part) != 64
        or any(character not in "0123456789abcdef" for character in hex_part)
    ):
        raise R197CorpusError(f"{label} is not a canonical sha256 digest")
    return digest


def _protected_archive_declarations(
    pointer_metadata: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[Path, str]]:
    """Return contract- and feature-bound archive identities by source day.

    The owner contract pins the raw ZIP bytes directly.  When the protected
    feature-manifest row and/or feature-shard header additionally records its
    source archive digest, each such declaration must agree with that exact
    owner digest.  This prevents an archive with matching episode IDs and
    decks from being substituted for the historical five-day source window.
    """

    manifest_path = Path(str(pointer_metadata["manifest_path"])).resolve()
    expected_manifest_sha256 = _required_sha256(
        pointer_metadata.get("manifest_sha256"), label="protected manifest digest"
    )
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise R197CorpusError("protected manifest changed before archive binding")
    manifest = _read_json_object(manifest_path)
    shard_rows = manifest.get("shards")
    if not isinstance(shard_rows, list) or not shard_rows:
        raise R197CorpusError("protected manifest has no feature shards")

    expected_days = set(R197_RAW_ARCHIVE_SHA256_BY_DAY)
    declarations: dict[str, dict[str, Any]] = {}
    shard_digests: dict[Path, str] = {}
    manifest_root = manifest_path.parent.resolve()
    for row in shard_rows:
        if not isinstance(row, Mapping):
            raise R197CorpusError("protected manifest contains a malformed shard row")
        source_dates = [str(value) for value in list(row.get("source_dates") or ())]
        if len(source_dates) != 1 or source_dates[0] not in expected_days:
            raise R197CorpusError(
                "r197 protected manifest must contain exactly one of the five "
                "owner-pinned source days per shard"
            )
        day = source_dates[0]
        expected_archive_sha256 = _required_sha256(
            R197_RAW_ARCHIVE_SHA256_BY_DAY[day],
            label=f"r197 owner raw archive digest for {day}",
        )
        shard_name = str(row.get("path") or "")
        if not shard_name:
            raise R197CorpusError("protected manifest shard has no path")
        shard = (manifest_root / shard_name).resolve()
        if not shard.is_relative_to(manifest_root) or not shard.is_file():
            raise R197CorpusError(f"protected manifest shard is outside its root: {shard_name}")
        shard_sha256 = _required_sha256(
            row.get("sha256"), label=f"protected feature shard digest for {shard.name}"
        )
        if sha256_file(shard) != shard_sha256:
            raise R197CorpusError(f"protected feature shard digest changed: {shard}")
        shard_digests[shard] = shard_sha256

        manifest_archive_sha256 = row.get("source_archive_sha256")
        if manifest_archive_sha256 is not None and _required_sha256(
            manifest_archive_sha256,
            label=f"protected manifest source archive digest for {day}",
        ) != expected_archive_sha256:
            raise R197CorpusError(
                f"protected manifest source archive digest disagrees with r197 contract: {day}"
            )

        try:
            with shard.open("rb") as stream:
                header = pickle.load(stream)
        except (OSError, EOFError, pickle.UnpicklingError) as exc:
            raise R197CorpusError(f"cannot read protected feature shard header: {shard}") from exc
        if (
            not isinstance(header, Mapping)
            or header.get("format") != SHARD_FORMAT
            or int(header.get("format_version", -1)) != SHARD_FORMAT_VERSION
        ):
            raise R197CorpusError(f"invalid protected feature shard header: {shard}")
        header_days = [str(value) for value in list(header.get("source_dates") or ())]
        if header_days != [day]:
            raise R197CorpusError(
                f"protected feature shard source day disagrees with manifest: {shard}"
            )
        header_specialist = header.get("required_archetype")
        if header_specialist is not None and str(header_specialist).strip().casefold() != SPECIALIST_ID:
            raise R197CorpusError(f"protected feature shard is not Alakazam-only: {shard}")
        expected_archive_name = f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        header_archive_name = header.get("source_archive")
        if header_archive_name is not None and str(header_archive_name) != expected_archive_name:
            raise R197CorpusError(
                f"protected feature shard source archive name disagrees with r197 contract: {shard}"
            )
        header_archive_sha256 = header.get("source_archive_sha256")
        if header_archive_sha256 is not None and _required_sha256(
            header_archive_sha256,
            label=f"protected feature shard source archive digest for {shard.name}",
        ) != expected_archive_sha256:
            raise R197CorpusError(
                f"protected feature shard source archive digest disagrees with r197 contract: {shard}"
            )

        declaration = declarations.setdefault(
            day,
            {
                "source_day": day,
                "source_archive": expected_archive_name,
                "expected_sha256": expected_archive_sha256,
                "manifest_source_archive_sha256": (
                    _required_sha256(
                        manifest_archive_sha256,
                        label=f"protected manifest source archive digest for {day}",
                    )
                    if manifest_archive_sha256 is not None
                    else None
                ),
                "feature_shards": [],
            },
        )
        declaration["feature_shards"].append(
            {
                "path": shard.name,
                "sha256": shard_sha256,
                "source_archive_sha256": (
                    _required_sha256(
                        header_archive_sha256,
                        label=f"protected feature shard source archive digest for {shard.name}",
                    )
                    if header_archive_sha256 is not None
                    else None
                ),
            }
        )

    if set(declarations) != expected_days:
        missing = sorted(expected_days.difference(declarations))
        extra = sorted(set(declarations).difference(expected_days))
        raise R197CorpusError(
            "protected manifest does not bind exactly the r197 five-day source window: "
            f"missing={missing!r} extra={extra!r}"
        )
    return (
        {day: declarations[day] for day in sorted(declarations)},
        shard_digests,
    )


def _assert_protected_inputs_unchanged(
    pointer_metadata: Mapping[str, Any],
    shard_digests: Mapping[Path, str],
) -> None:
    """Recheck immutable identity inputs immediately before publication."""

    if sha256_file(Path(str(pointer_metadata["path"]))) != pointer_metadata["sha256"]:
        raise R197CorpusError("protected pointer changed during materialization")
    if (
        sha256_file(Path(str(pointer_metadata["manifest_path"])))
        != pointer_metadata["manifest_sha256"]
    ):
        raise R197CorpusError("protected manifest changed during materialization")
    for shard, expected_sha256 in shard_digests.items():
        if sha256_file(shard) != expected_sha256:
            raise R197CorpusError(f"protected feature shard changed during materialization: {shard}")


def _identity_payload(identity: Mapping[str, Any]) -> dict[str, Any]:
    deck = [int(value) for value in list(identity.get("deck") or ())]
    episode_id = str(identity.get("episode_id") or "")
    day = str(identity.get("day") or "")
    seat = int(identity.get("seat", -1))
    if not episode_id or not day or seat not in (0, 1) or len(deck) != 60:
        raise R197CorpusError("protected identity is malformed")
    return {
        "schema": IDENTITY_SCHEMA,
        "episode_id": episode_id,
        "seat": seat,
        "source_day": day,
        "source_shard": str(identity.get("source_shard") or ""),
        "deck_sha256": canonical_json_sha256(deck),
    }


def _identity_with_fingerprint(identity: Mapping[str, Any]) -> dict[str, Any]:
    row = _identity_payload(identity)
    row["identity_fingerprint"] = canonical_json_sha256(row)
    return row


class _ProtectedIdentityClassifier:
    """Preserve one checksum-verified historical Alakazam identity locally.

    The protected pointer, raw ZIP member name, target seat, and submitted deck
    are already independently verified before this wrapper is constructed.
    A newer rule classifier may no longer recognize an old exact Alakazam list
    and would otherwise make the visual converter silently omit that identity.
    Only that exact target seat may be relabelled, and only from ``unknown``;
    a currently recognized non-Alakazam label remains a hard provenance error.
    """

    def __init__(self, base: Any, identity: Mapping[str, Any]) -> None:
        self._base = base
        self._episode_id = str(identity["episode_id"])
        self._seat = int(identity["seat"])
        self._deck = [int(value) for value in list(identity["deck"])]
        self._cached_payload: object | None = None
        self._cached_result: tuple[list[Any], list[Any]] | None = None
        self._provenance: dict[str, Any] | None = None
        if self._seat not in (0, 1) or len(self._deck) != 60:
            raise R197CorpusError("protected identity classifier received malformed identity")

    def _classify(self, payload: dict[str, Any]) -> tuple[list[Any], list[Any]]:
        try:
            classified_decks, labels = self._base.classify_episode(payload)
        except Exception as exc:
            raise R197CorpusError(
                f"current classifier failed for protected identity {self._episode_id!r} "
                f"seat {self._seat}"
            ) from exc
        if len(classified_decks) != 2 or len(labels) != 2:
            raise R197CorpusError("current classifier did not return two seats")
        try:
            classified_target_deck = [
                int(value) for value in list(classified_decks[self._seat] or ())
            ]
        except (TypeError, ValueError, OverflowError) as exc:
            raise R197CorpusError("current classifier returned a malformed target deck") from exc
        if classified_target_deck != self._deck:
            raise R197CorpusError(
                "current classifier target deck disagrees with exact protected identity: "
                f"episode={self._episode_id!r} seat={self._seat}"
            )
        current_target_label = str(
            getattr(labels[self._seat], "deck_id", "") or ""
        ).strip().casefold()
        opponent_label = str(
            getattr(labels[1 - self._seat], "deck_id", "") or ""
        ).strip().casefold()
        effective_labels = list(labels)
        if current_target_label == SPECIALIST_ID:
            override_applied = False
            classification_mode = "current_rule_alakazam"
        elif current_target_label == archetypes.UNKNOWN:
            override_applied = True
            classification_mode = "protected_identity_exact_unknown_override"
            effective_labels[self._seat] = SimpleNamespace(
                deck_id=SPECIALIST_ID,
                method="r197_protected_identity_exact_unknown_override",
            )
        else:
            raise R197CorpusError(
                "current classifier recognizes a protected Alakazam identity as a "
                "different archetype; refusing override: "
                f"episode={self._episode_id!r} seat={self._seat} "
                f"label={current_target_label!r}"
            )
        self._provenance = {
            "schema": PROTECTED_IDENTITY_EXACT_SCHEMA,
            "selection": "protected_pointer_verified_episode_seat_deck",
            "episode_id": self._episode_id,
            "seat": self._seat,
            "deck_sha256": canonical_json_sha256(self._deck),
            "raw_episode_deck_verified": True,
            "current_rule_target_label": current_target_label,
            "effective_conversion_target_label": SPECIALIST_ID,
            "identity_bound_label_override": override_applied,
            "classification_mode": classification_mode,
            "opponent_current_rule_label": opponent_label,
            "opponent_label_preserved": True,
        }
        return list(classified_decks), effective_labels

    def classify_episode(self, payload: dict[str, Any]) -> tuple[list[Any], list[Any]]:
        if payload is self._cached_payload and self._cached_result is not None:
            return self._cached_result
        result = self._classify(payload)
        self._cached_payload = payload
        self._cached_result = result
        return result

    def prepare(self, payload: dict[str, Any]) -> None:
        """Establish provenance before the converter sees the identity."""

        self.classify_episode(payload)

    def protected_identity_exact_provenance(self) -> dict[str, Any]:
        if self._provenance is None:
            raise R197CorpusError("protected identity classifier was not prepared")
        return dict(self._provenance)


def _validate_protected_identity_exact(
    provenance: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the narrow classifier override evidence before serialization."""

    expected_deck = [int(value) for value in list(identity["deck"])]
    return _validate_protected_identity_exact_fields(
        provenance,
        episode_id=str(identity["episode_id"]),
        seat=int(identity["seat"]),
        deck_sha256=canonical_json_sha256(expected_deck),
    )


def _validate_protected_identity_exact_fields(
    provenance: Mapping[str, Any],
    *,
    episode_id: str,
    seat: int,
    deck_sha256: str,
) -> dict[str, Any]:
    """Validate identity-bound provenance from either source or serialized row."""

    target_label = str(provenance.get("current_rule_target_label") or "").casefold()
    opponent_label = provenance.get("opponent_current_rule_label")
    override = provenance.get("identity_bound_label_override")
    try:
        provenance_seat = int(provenance.get("seat", -1))
    except (TypeError, ValueError):
        provenance_seat = -1
    if (
        provenance.get("schema") != PROTECTED_IDENTITY_EXACT_SCHEMA
        or provenance.get("selection") != "protected_pointer_verified_episode_seat_deck"
        or str(provenance.get("episode_id") or "") != episode_id
        or provenance_seat != seat
        or provenance.get("deck_sha256") != deck_sha256
        or provenance.get("raw_episode_deck_verified") is not True
        or provenance.get("effective_conversion_target_label") != SPECIALIST_ID
        or provenance.get("opponent_label_preserved") is not True
        or not isinstance(opponent_label, str)
        or not opponent_label.strip()
        or target_label not in {SPECIALIST_ID, archetypes.UNKNOWN}
        or (target_label == archetypes.UNKNOWN and override is not True)
        or (target_label == SPECIALIST_ID and override is not False)
    ):
        raise R197CorpusError("protected identity exact provenance is malformed")
    return dict(provenance)


def _assert_preserved_opponent_label(
    record: Mapping[str, Any], provenance: Mapping[str, Any]
) -> None:
    """Ensure conversion did not alter the non-target seat's classifier label."""

    observed = str(record.get("opp_archetype") or "").strip().casefold()
    expected = str(provenance.get("opponent_current_rule_label") or "").strip().casefold()
    if not observed or observed != expected:
        raise R197CorpusError(
            "visual conversion changed the preserved opponent archetype label"
        )


def _validate_protected_identity_exact_summary(
    summary: Mapping[str, Any], counts: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate corpus-level accounting for identity-bound unknown overrides."""

    policy = summary.get("target_label_override_policy")
    summary_counts = summary.get("counts")
    if not isinstance(policy, Mapping) or not isinstance(summary_counts, Mapping):
        raise R197CorpusError("r197 protected identity summary is malformed")
    try:
        verified_records = int(summary_counts.get("verified_records", -1))
        unknown_overrides = int(summary_counts.get("unknown_label_overrides", -1))
        current_rule_alakazam = int(
            summary_counts.get("current_rule_alakazam_records", -1)
        )
        materialized_records = int(counts.get("records", -1))
    except (TypeError, ValueError) as exc:
        raise R197CorpusError("r197 protected identity summary has invalid counts") from exc
    if (
        set(summary)
        != {
            "schema",
            "selection",
            "target_label_override_policy",
            "opponent_label_behavior",
            "counts",
        }
        or set(policy)
        != {
            "allowed_only_when_current_rule_target_label",
            "recognized_non_alakazam_behavior",
            "current_rule_alakazam_behavior",
        }
        or set(summary_counts)
        != {
            "verified_records",
            "unknown_label_overrides",
            "current_rule_alakazam_records",
        }
        or summary.get("schema") != PROTECTED_IDENTITY_EXACT_SCHEMA
        or summary.get("selection") != "protected_pointer_verified_episode_seat_deck"
        or policy.get("allowed_only_when_current_rule_target_label")
        != archetypes.UNKNOWN
        or policy.get("recognized_non_alakazam_behavior") != "fail_closed"
        or policy.get("current_rule_alakazam_behavior")
        != "preserve_without_override"
        or summary.get("opponent_label_behavior") != "preserve_current_rule_label"
        or min(verified_records, unknown_overrides, current_rule_alakazam) < 0
        or verified_records != materialized_records
        or unknown_overrides + current_rule_alakazam != verified_records
    ):
        raise R197CorpusError("r197 protected identity summary contract changed")
    return {
        "schema": PROTECTED_IDENTITY_EXACT_SCHEMA,
        "selection": "protected_pointer_verified_episode_seat_deck",
        "target_label_override_policy": dict(policy),
        "opponent_label_behavior": "preserve_current_rule_label",
        "counts": {
            "verified_records": verified_records,
            "unknown_label_overrides": unknown_overrides,
            "current_rule_alakazam_records": current_rule_alakazam,
        },
    }


def _record_for_identity(
    converted: VisualEpisodeResult,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    episode_id = str(identity["episode_id"])
    seat = int(identity["seat"])
    deck = [int(value) for value in list(identity["deck"])]
    matching = [
        record
        for record in converted.records
        if str(record.get("episode_id") or "") == episode_id
        and int(record.get("seat", -1)) == seat
    ]
    if len(matching) != 1:
        raise R197CorpusError(
            f"current visual conversion lost verified identity {episode_id!r} seat {seat}"
        )
    record = matching[0]
    if (
        str(record.get("archetype") or "").casefold() != SPECIALIST_ID
        or [int(value) for value in list(record.get("deck") or ())] != deck
        or not bool(record.get("info_set_ok", False))
    ):
        raise R197CorpusError(
            f"visual conversion changed verified Alakazam identity {episode_id!r} seat {seat}"
        )
    return record


def _terminal_value(record: Mapping[str, Any]) -> float:
    try:
        value = float(record.get("value"))
    except (TypeError, ValueError) as exc:
        raise R197CorpusError("converted record has no numeric terminal value") from exc
    if not math.isfinite(value) or value not in {-1.0, 0.0, 1.0}:
        raise R197CorpusError("converted record terminal value is not an exact outcome")
    return value


def _complete_action_row(
    *,
    record: Mapping[str, Any],
    identity: Mapping[str, Any],
    protected_identity_exact: Mapping[str, Any],
    split: str,
    step: Mapping[str, Any],
    action_space: Mapping[str, Any],
    selected_action: Sequence[int],
    selected_action_index: int,
) -> dict[str, Any]:
    identity_row = _identity_with_fingerprint(identity)
    identity_provenance = _validate_protected_identity_exact(
        protected_identity_exact, identity
    )
    _assert_preserved_opponent_label(record, identity_provenance)
    observation = step.get("observation")
    if not isinstance(observation, Mapping):
        raise R197CorpusError("validated record step lacks a causal observation")
    legal_actions = [list(action) for action in action_space["complete_ordered_actions"]]
    action = [int(value) for value in selected_action]
    if not 0 <= int(selected_action_index) < len(legal_actions):
        raise R197CorpusError("selected complete action index is out of range")
    if legal_actions[int(selected_action_index)] != action:
        raise R197CorpusError("selected complete action index does not map exactly")
    row = {
        "schema": ROW_SCHEMA,
        "corpus_schema": CORPUS_SCHEMA,
        "split": split,
        "identity": identity_row,
        "protected_identity_exact": identity_provenance,
        "episode_id": str(record.get("episode_id") or ""),
        "seat": int(record.get("seat", -1)),
        "env_step": int(step.get("env_step", -1)),
        "source_day": str(identity["day"]),
        "source": str(record.get("source") or ""),
        "archetype": SPECIALIST_ID,
        "opp_archetype": str(record.get("opp_archetype") or ""),
        # The deck is required later for exact board/token encoding.  It is
        # public submitted-deck metadata, not a policy input replacement.
        "deck": [int(value) for value in list(record.get("deck") or ())],
        "deck_sha256": canonical_json_sha256(list(record.get("deck") or ())),
        "observation": dict(observation),
        "observation_fingerprint": canonical_json_sha256(dict(observation)),
        "action": action,
        "legal_actions": legal_actions,
        "selected_action_index": int(selected_action_index),
        "action_space": dict(action_space),
        "action_space_fingerprint": canonical_json_sha256(dict(action_space)),
        "action_space_source": "runtime_complete_observation",
        "factorized_prefix_substitution": False,
        "unobserved_action_targets_present": False,
        "game_value": _terminal_value(record),
        "outcome_available": True,
        "terminal_complete": True,
        "target_provenance": {
            "schema": "poke_bot.rtp_complete_action_targets/v1",
            "selected_action": "recorded_visual_trace_complete_ordered_action",
            "terminal_value": "recorded_episode_outcome",
            "unobserved_action_targets": "absent_masked",
            "value_of_planning_target": "absent_masked",
            "kaggle_or_evaluation_replay": False,
            "protected_identity_exact": identity_provenance,
        },
    }
    if len(row["deck"]) != 60:
        raise R197CorpusError("converted record deck is not exactly 60 cards")
    row["row_fingerprint"] = canonical_json_sha256(row)
    return row


def _too_large_row(
    *,
    record: Mapping[str, Any],
    identity: Mapping[str, Any],
    protected_identity_exact: Mapping[str, Any],
    split: str,
    step: Mapping[str, Any],
    error: features.ActionSpaceTooLarge,
) -> dict[str, Any]:
    identity_provenance = _validate_protected_identity_exact(
        protected_identity_exact, identity
    )
    _assert_preserved_opponent_label(record, identity_provenance)
    observation = step.get("observation")
    if not isinstance(observation, Mapping):
        raise R197CorpusError("validated overflow step lacks a causal observation")
    select = observation.get("select")
    if not isinstance(select, Mapping) or not isinstance(select.get("option"), list):
        raise R197CorpusError("overflow step lacks a canonical select.option list")
    try:
        total = int(features.ordered_action_count(dict(observation)))
    except Exception as exc:
        raise R197CorpusError("cannot count an overflow complete action space") from exc
    row = {
        "schema": ACTION_SPACE_TOO_LARGE_SCHEMA,
        "corpus_schema": CORPUS_SCHEMA,
        "split": split,
        "identity": _identity_with_fingerprint(identity),
        "protected_identity_exact": identity_provenance,
        "episode_id": str(record.get("episode_id") or ""),
        "seat": int(record.get("seat", -1)),
        "env_step": int(step.get("env_step", -1)),
        "source_day": str(identity["day"]),
        "reason": "action_space_too_large",
        "error": str(error),
        "max_action_combos": MAX_ACTION_COMBOS,
        "legal_option_count": len(select["option"]),
        "complete_ordered_action_count": total,
        "recorded_action": [int(value) for value in list(step.get("action") or ())],
        "observation_fingerprint": canonical_json_sha256(dict(observation)),
        "factorized_prefix_substitution": False,
        "complete_action_mapping": "not_materialized_action_space_too_large",
        "eligible_for_r197_training": False,
    }
    row["row_fingerprint"] = canonical_json_sha256(row)
    return row


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_entry(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _derived_corpus_fingerprint_from_manifest(manifest: Mapping[str, Any]) -> str:
    """Rebuild the timestamp-free corpus identity from manifest content."""

    source_pointer = manifest.get("source_pointer")
    source_archives = manifest.get("source_archives")
    if not isinstance(source_pointer, Mapping) or not isinstance(source_archives, list):
        raise R197CorpusError("r197 manifest cannot form a derived corpus fingerprint")
    if not all(isinstance(row, Mapping) for row in source_archives):
        raise R197CorpusError("r197 manifest has malformed source archive fingerprint input")
    return canonical_json_sha256(
        {
            "schema": CORPUS_SCHEMA,
            "specialist_id": SPECIALIST_ID,
            "source_pointer_sha256": source_pointer.get("sha256"),
            "source_manifest_sha256": source_pointer.get("manifest_sha256"),
            "source_archives": [
                {
                    key: value
                    for key, value in row.items()
                    if key != "path"
                }
                for row in source_archives
            ],
            "protected_identity_exact": manifest.get("protected_identity_exact"),
            "generator_fingerprints": manifest.get("generator_fingerprints"),
            "split": manifest.get("split"),
            "action_space": manifest.get("action_space"),
            "outputs": manifest.get("outputs"),
        }
    )


def _count_jsonl_rows(path: Path) -> int:
    """Count nonblank newline-terminated JSONL rows without retaining them."""

    rows = 0
    with Path(path).open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith(b"\n") or not line.strip():
                raise R197CorpusError(
                    f"r197 JSONL stream has a blank or unterminated row: {path}:{line_number}"
                )
            rows += 1
    return rows


def _archive_path(archive_root: Path, day: str) -> Path:
    return archive_root / f"pokemon-tcg-ai-battle-episodes-{day}.zip"


def _exact_member_index(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Index a raw archive once and reject duplicate file names fail-closed."""

    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        if info.filename in members:
            raise R197CorpusError(
                f"raw source archive contains a duplicate member name: {info.filename!r}"
            )
        members[info.filename] = info
    return members


def _validated_source_payload(
    archive: zipfile.ZipFile,
    *,
    members: Mapping[str, zipfile.ZipInfo],
    episode_id: str,
    seat: int,
    deck: Sequence[int],
) -> dict[str, Any]:
    member_name = f"{episode_id}.json"
    info = members.get(member_name)
    if info is None:
        raise R197CorpusError(
            f"source archive must contain exactly one verified member {member_name!r}"
        )
    try:
        payload = json.loads(archive.read(info))
    except (OSError, json.JSONDecodeError) as exc:
        raise R197CorpusError(f"invalid verified episode member: {member_name}") from exc
    if not isinstance(payload, dict):
        raise R197CorpusError(f"verified episode member is not an object: {member_name}")
    if episode_id_of(payload, fallback=episode_id) != episode_id:
        raise R197CorpusError(f"raw episode identity changed: {member_name}")
    decks = extract_setup_decks(payload)
    if seat not in (0, 1) or [int(value) for value in list(decks[seat] or ())] != [
        int(value) for value in deck
    ]:
        raise R197CorpusError(
            f"raw episode submitted deck changed: {member_name} seat {seat}"
        )
    return payload


def _close_all(writers: Sequence[_JsonlWriter]) -> dict[str, StreamFile]:
    result: dict[str, StreamFile] = {}
    for writer in writers:
        file = writer.close()
        result[file.path] = file
    return result


def _validate_identities(
    pointer_path: Path,
    *,
    specialist_id: str,
) -> list[dict[str, Any]]:
    identities = [
        dict(identity)
        for identity in iter_legacy_identities(
            pointer_path,
            specialist_id=specialist_id,
        )
    ]
    if not identities:
        raise R197CorpusError("protected pointer yielded no verified Alakazam identities")
    seen: set[tuple[str, int]] = set()
    episode_days: dict[str, str] = {}
    for identity in identities:
        row = _identity_payload(identity)
        key = (str(row["episode_id"]), int(row["seat"]))
        if key in seen:
            raise R197CorpusError(f"duplicate verified episode/seat identity: {key!r}")
        seen.add(key)
        previous_day = episode_days.setdefault(str(row["episode_id"]), str(row["source_day"]))
        if previous_day != row["source_day"]:
            raise R197CorpusError(
                "one episode appears under multiple raw archive days: "
                f"episode={row['episode_id']!r} days={previous_day!r},{row['source_day']!r}"
            )
    return sorted(
        identities,
        key=lambda row: (str(row["day"]), str(row["episode_id"]), int(row["seat"])),
    )


def _existing_receipt_if_matching(
    output_dir: Path,
    *,
    pointer_metadata: Mapping[str, Any],
    archive_root: Path,
    heldout_fraction: str,
    generator_fingerprints: Mapping[str, str],
) -> dict[str, Any] | None:
    """Return a verified immutable receipt, or reject an occupied output dir."""

    if not output_dir.exists():
        return None
    if not output_dir.is_dir():
        raise R197CorpusError(f"r197 output exists and is not a directory: {output_dir}")
    receipt_path = output_dir / RECEIPT_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    if not receipt_path.is_file() or not manifest_path.is_file():
        raise R197CorpusError("r197 output directory is occupied but not a sealed corpus")
    receipt = verify_r197_complete_action_manifest(
        manifest_path,
        archive_root=archive_root,
        require_current_generator=True,
    )
    if (
        receipt.get("source_pointer", {}).get("sha256") != pointer_metadata["sha256"]
        or receipt.get("split", {}).get("seed") != SPLIT_SEED
        or receipt.get("split", {}).get("heldout_fraction") != heldout_fraction
        or receipt.get("generator_fingerprints") != dict(generator_fingerprints)
    ):
        raise R197CorpusError("existing r197 corpus receipt has a different identity")
    return receipt


def materialize_r197_complete_action_corpus(
    source_pointer: Path,
    archive_root: Path,
    output_dir: Path,
    *,
    specialist_id: str = SPECIALIST_ID,
    split_seed: int = SPLIT_SEED,
    heldout_fraction: str | float | Decimal = DEFAULT_HELDOUT_FRACTION,
    max_action_combos: int = MAX_ACTION_COMBOS,
    expected_pointer_sha256: str | None = None,
    classifier: Any | None = None,
    convert_episode: Callable[..., VisualEpisodeResult] = convert_visual_episode,
) -> dict[str, Any]:
    """Materialize the parent-independent r197 complete-action corpus.

    The only mutable writes occur below ``output_dir``.  If an output directory
    already contains a matching sealed receipt, it is verified and returned;
    a mismatched or partial directory is rejected rather than overwritten.
    """

    _normalise_specialist_id(specialist_id)
    if int(split_seed) != SPLIT_SEED:
        raise R197CorpusError(f"r197 split seed is fixed at {SPLIT_SEED}")
    if int(max_action_combos) != MAX_ACTION_COMBOS:
        raise R197CorpusError(
            f"r197 max_action_combos is fixed at {MAX_ACTION_COMBOS}"
        )
    fraction, rendered_fraction = _normalise_heldout_fraction(heldout_fraction)
    pointer_path = Path(source_pointer).expanduser().resolve()
    archive_root_path = Path(archive_root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not pointer_path.is_file():
        raise FileNotFoundError(pointer_path)
    if not archive_root_path.is_dir():
        raise NotADirectoryError(archive_root_path)
    if output_path == output_path.parent or output_path.name in {"", "."}:
        raise R197CorpusError("refusing an ambiguous r197 output directory")
    pointer_metadata = _pointer_metadata(pointer_path)
    if expected_pointer_sha256 and pointer_metadata["sha256"] != str(
        expected_pointer_sha256
    ):
        raise R197CorpusError(
            "r197 protected pointer digest mismatch: "
            f"expected={expected_pointer_sha256} actual={pointer_metadata['sha256']}"
        )
    generator_fingerprints = _source_fingerprints()
    existing = _existing_receipt_if_matching(
        output_path,
        pointer_metadata=pointer_metadata,
        archive_root=archive_root_path,
        heldout_fraction=rendered_fraction,
        generator_fingerprints=generator_fingerprints,
    )
    if existing is not None:
        return existing

    pointer_digest_before = pointer_metadata["sha256"]
    archive_declarations, source_shard_digests = _protected_archive_declarations(
        pointer_metadata
    )
    identities = _validate_identities(pointer_path, specialist_id=SPECIALIST_ID)
    if sha256_file(pointer_path) != pointer_digest_before:
        raise R197CorpusError("protected pointer changed while identities were read")

    identity_rows = [_identity_with_fingerprint(identity) for identity in identities]
    identities_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for identity in identities:
        identities_by_episode[str(identity["episode_id"])].append(identity)
        identities_by_day[str(identity["day"])].append(identity)
    if set(identities_by_day) != set(R197_RAW_ARCHIVE_SHA256_BY_DAY):
        raise R197CorpusError(
            "protected verified identities do not cover exactly the r197 five-day source window: "
            f"days={sorted(identities_by_day)!r}"
        )

    split_rows: dict[str, dict[str, Any]] = {}
    for episode_id, episode_identities in sorted(identities_by_episode.items()):
        assignment = deterministic_episode_split(
            episode_id,
            split_seed=SPLIT_SEED,
            heldout_fraction=fraction,
        )
        row = {
            "schema": SPLIT_SCHEMA,
            **assignment,
            "source_day": str(episode_identities[0]["day"]),
            "identity_seats": sorted(int(item["seat"]) for item in episode_identities),
            "identity_count": len(episode_identities),
        }
        row["split_fingerprint"] = canonical_json_sha256(row)
        split_rows[episode_id] = row

    split_counts = {
        split: sum(1 for row in split_rows.values() if row["split"] == split)
        for split in ("train", "heldout")
    }
    if not split_counts["train"] or not split_counts["heldout"]:
        raise R197CorpusError(
            "source-disjoint r197 split is empty; pointer needs at least one "
            "train and one heldout episode under the fixed split rule"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.partial.", dir=output_path.parent)
    )
    writers: list[_JsonlWriter] = []
    try:
        train_writer = _JsonlWriter(work_dir / TRAIN_FILENAME)
        heldout_writer = _JsonlWriter(work_dir / HELDOUT_FILENAME)
        too_large_writer = _JsonlWriter(work_dir / TOO_LARGE_FILENAME)
        identities_writer = _JsonlWriter(work_dir / IDENTITIES_FILENAME)
        split_writer = _JsonlWriter(work_dir / SPLITS_FILENAME)
        writers = [
            train_writer,
            heldout_writer,
            too_large_writer,
            identities_writer,
            split_writer,
        ]
        for row in identity_rows:
            identities_writer.write(row)
        for episode_id in sorted(split_rows):
            split_writer.write(split_rows[episode_id])

        archives: list[dict[str, Any]] = []
        materialized_counts = {
            "records": 0,
            "decisions_seen": 0,
            "complete_action_rows": 0,
            "action_space_too_large_rows": 0,
            "train_complete_action_rows": 0,
            "heldout_complete_action_rows": 0,
            "protected_identity_exact_records": 0,
            "protected_identity_unknown_label_overrides": 0,
            "protected_identity_current_rule_alakazam": 0,
        }
        active_classifier = classifier if classifier is not None else _RuleClassifier()
        for day, day_identities in sorted(identities_by_day.items()):
            declaration = archive_declarations.get(day)
            if declaration is None:
                raise R197CorpusError(
                    f"protected source archive declaration is missing for {day}"
                )
            archive_path = _archive_path(archive_root_path, day)
            if not archive_path.is_file():
                raise FileNotFoundError(archive_path)
            archive_digest_before = sha256_file(archive_path)
            if archive_digest_before != declaration["expected_sha256"]:
                raise R197CorpusError(
                    "raw archive digest disagrees with the exact r197 owner contract: "
                    f"day={day} expected={declaration['expected_sha256']} "
                    f"actual={archive_digest_before}"
                )
            archive_bytes = int(archive_path.stat().st_size)
            archive_records = 0
            archive_decisions = 0
            with zipfile.ZipFile(archive_path, "r") as archive:
                members = _exact_member_index(archive)
                for identity in day_identities:
                    episode_id = str(identity["episode_id"])
                    seat = int(identity["seat"])
                    payload = _validated_source_payload(
                        archive,
                        members=members,
                        episode_id=episode_id,
                        seat=seat,
                        deck=list(identity["deck"]),
                    )
                    identity_classifier = _ProtectedIdentityClassifier(
                        active_classifier, identity
                    )
                    identity_classifier.prepare(payload)
                    converted = convert_episode(
                        payload,
                        identity_classifier,
                        source=f"pokemon-tcg-ai-battle-episodes-{day}",
                        required_archetype=SPECIALIST_ID,
                    )
                    if not isinstance(converted, VisualEpisodeResult):
                        raise R197CorpusError("visual conversion did not return VisualEpisodeResult")
                    protected_identity_exact = _validate_protected_identity_exact(
                        identity_classifier.protected_identity_exact_provenance(),
                        identity,
                    )
                    record = _record_for_identity(converted, identity)
                    materialized_counts["records"] += 1
                    archive_records += 1
                    materialized_counts["protected_identity_exact_records"] += 1
                    if protected_identity_exact["identity_bound_label_override"]:
                        materialized_counts["protected_identity_unknown_label_overrides"] += 1
                    else:
                        materialized_counts["protected_identity_current_rule_alakazam"] += 1
                    split = str(split_rows[episode_id]["split"])
                    for step in list(record.get("steps") or ()):
                        if not isinstance(step, Mapping):
                            raise R197CorpusError("converted record contains a malformed step")
                        materialized_counts["decisions_seen"] += 1
                        archive_decisions += 1
                        observation = step.get("observation")
                        if not isinstance(observation, Mapping):
                            raise R197CorpusError("converted record step lacks an observation")
                        try:
                            combos = features.enumerate_action_combos(
                                dict(observation), max_combos=MAX_ACTION_COMBOS
                            )
                        except features.ActionSpaceTooLarge as exc:
                            too_large_writer.write(
                                _too_large_row(
                                    record=record,
                                    identity=identity,
                                    protected_identity_exact=protected_identity_exact,
                                    split=split,
                                    step=step,
                                    error=exc,
                                )
                            )
                            materialized_counts["action_space_too_large_rows"] += 1
                            continue
                        except Exception as exc:
                            raise R197CorpusError(
                                "canonical complete action enumeration failed"
                            ) from exc
                        legal_actions = [list(action) for action in combos]
                        action = [int(value) for value in list(step.get("action") or ())]
                        try:
                            selected_index = legal_actions.index(action)
                        except ValueError as exc:
                            raise R197CorpusError(
                                "recorded action is absent from canonical complete ordered support: "
                                f"episode={episode_id!r} seat={seat} step={step.get('env_step')!r}"
                            ) from exc
                        action_space = _action_space_descriptor(
                            observation,
                            legal_actions,
                            max_action_combos=MAX_ACTION_COMBOS,
                        )
                        row = _complete_action_row(
                            record=record,
                            identity=identity,
                            protected_identity_exact=protected_identity_exact,
                            split=split,
                            step=step,
                            action_space=action_space,
                            selected_action=action,
                            selected_action_index=selected_index,
                        )
                        (train_writer if split == "train" else heldout_writer).write(row)
                        materialized_counts["complete_action_rows"] += 1
                        materialized_counts[f"{split}_complete_action_rows"] += 1
            archive_digest_after = sha256_file(archive_path)
            if archive_digest_after != archive_digest_before:
                raise R197CorpusError(f"raw archive changed while reprocessing: {archive_path}")
            archives.append(
                {
                    "source_day": day,
                    "path": str(archive_path),
                    "sha256": archive_digest_before,
                    "expected_sha256": declaration["expected_sha256"],
                    "source_archive": declaration["source_archive"],
                    "protected_manifest_source_archive_sha256": declaration[
                        "manifest_source_archive_sha256"
                    ],
                    "protected_feature_shards": declaration["feature_shards"],
                    "bytes": archive_bytes,
                    "verified_identities": len(day_identities),
                    "reprocessed_records": archive_records,
                    "decisions_seen": archive_decisions,
                }
            )

        files = {
            path: file.to_json()
            for path, file in _close_all(writers).items()
        }
        writers = []
        _assert_protected_inputs_unchanged(pointer_metadata, source_shard_digests)
        if _source_fingerprints() != generator_fingerprints:
            raise R197CorpusError("generator source changed during materialization")

        protected_identity_exact_summary = {
            "schema": PROTECTED_IDENTITY_EXACT_SCHEMA,
            "selection": "protected_pointer_verified_episode_seat_deck",
            "target_label_override_policy": {
                "allowed_only_when_current_rule_target_label": archetypes.UNKNOWN,
                "recognized_non_alakazam_behavior": "fail_closed",
                "current_rule_alakazam_behavior": "preserve_without_override",
            },
            "opponent_label_behavior": "preserve_current_rule_label",
            "counts": {
                "verified_records": materialized_counts[
                    "protected_identity_exact_records"
                ],
                "unknown_label_overrides": materialized_counts[
                    "protected_identity_unknown_label_overrides"
                ],
                "current_rule_alakazam_records": materialized_counts[
                    "protected_identity_current_rule_alakazam"
                ],
            },
        }
        protected_identity_exact_summary = _validate_protected_identity_exact_summary(
            protected_identity_exact_summary, materialized_counts
        )

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "corpus_schema": CORPUS_SCHEMA,
            "status": (
                "completed"
                if materialized_counts["action_space_too_large_rows"] == 0
                else "completed_with_action_space_too_large_exclusions"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "specialist_id": SPECIALIST_ID,
            "parent_independent": True,
            "parent_checkpoint": None,
            "source_pointer": pointer_metadata,
            "protected_identity_exact": protected_identity_exact_summary,
            "source_archives": archives,
            "generator_fingerprints": generator_fingerprints,
            "split": {
                "unit": "episode_id",
                "source_disjoint": True,
                "seed": SPLIT_SEED,
                "heldout_fraction": rendered_fraction,
                "rule": SPLIT_RULE,
                "episodes": split_counts,
            },
            "action_space": {
                "schema": ACTION_SPACE_SCHEMA,
                "builder": "poke_bot.features.enumerate_action_combos",
                "representation": "complete_ordered_action_combinations",
                "canonical_order": CANONICAL_ACTION_ORDER,
                "max_action_combos": MAX_ACTION_COMBOS,
                "factorized_policy_stage_substitution_allowed": False,
                "per_decision_action_space_fingerprint_required": True,
                "action_space_too_large_behavior": "separate_nontraining_audit_rows",
            },
            "eligibility": {
                "derived_from_protected_expert_identities": True,
                "training_eligible": True,
                "evaluation_or_kaggle_replays_training_eligible": False,
                "kaggle_replay_eligible": False,
                "serving_eligible": False,
                "action_authority_enabled": False,
            },
            "outputs": {
                "verified_identities": files[IDENTITIES_FILENAME],
                "episode_splits": files[SPLITS_FILENAME],
                "train": files[TRAIN_FILENAME],
                "heldout": files[HELDOUT_FILENAME],
                "action_space_too_large": files[TOO_LARGE_FILENAME],
            },
            "counts": {
                "verified_identities": len(identities),
                "verified_episodes": len(split_rows),
                **materialized_counts,
            },
        }
        # This stable fingerprint deliberately excludes timestamps and the
        # receipt's own digest while binding all byte-bearing input/output
        # identities needed by a later training or promotion receipt.
        manifest["derived_corpus_fingerprint"] = _derived_corpus_fingerprint_from_manifest(
            manifest
        )
        manifest_path = work_dir / MANIFEST_FILENAME
        _write_json(manifest_path, manifest)
        manifest_entry = _file_entry(manifest_path)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": manifest["status"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "corpus_schema": CORPUS_SCHEMA,
            "manifest": manifest_entry,
            "derived_corpus_fingerprint": manifest["derived_corpus_fingerprint"],
            "source_pointer": pointer_metadata,
            "protected_identity_exact": protected_identity_exact_summary,
            "source_archives": archives,
            "generator_fingerprints": generator_fingerprints,
            "split": manifest["split"],
            "action_space": manifest["action_space"],
            "eligibility": manifest["eligibility"],
            "counts": manifest["counts"],
            "parent_independent": True,
        }
        _write_json(work_dir / RECEIPT_FILENAME, receipt)
        _fsync_directory(work_dir)
        # A new directory appearing after the initial reuse check is not ours
        # to replace.  Leave the sealed temporary artifact behind only until
        # the exception handler removes it; never clobber caller data.
        if output_path.exists():
            raise R197CorpusError(
                "r197 output directory appeared during materialization; refusing to replace it"
            )
        os.replace(work_dir, output_path)
        _fsync_directory(output_path.parent)
        result = dict(receipt)
        result["receipt"] = _file_entry(output_path / RECEIPT_FILENAME)
        return result
    except Exception:
        for writer in writers:
            try:
                writer.close()
            except Exception:
                pass
        # The directory was created by this invocation below the exact caller
        # supplied output parent; it has never been promoted to output_path.
        if work_dir.exists():
            shutil.rmtree(work_dir)
        raise


def _manifest_path(path: Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    return candidate / MANIFEST_FILENAME if candidate.is_dir() else candidate


def verify_r197_complete_action_manifest(
    manifest_path: Path,
    *,
    archive_root: Path | None = None,
    require_current_generator: bool = False,
) -> dict[str, Any]:
    """Verify byte identities of a sealed r197 corpus without loading JSONL.

    JSONL rows remain streamable.  Use :func:`iter_complete_action_rows` when
    row-level action-space validation is required by the trainer.
    """

    path = _manifest_path(manifest_path)
    manifest = _read_json_object(path)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("corpus_schema") != CORPUS_SCHEMA:
        raise R197CorpusError("not an r197 complete-action corpus manifest")
    if manifest.get("specialist_id") != SPECIALIST_ID or manifest.get("parent_independent") is not True:
        raise R197CorpusError("r197 corpus has an invalid specialist/parent contract")
    if int((manifest.get("split") or {}).get("seed", -1)) != SPLIT_SEED:
        raise R197CorpusError("r197 corpus split seed changed")
    if int((manifest.get("action_space") or {}).get("max_action_combos", -1)) != MAX_ACTION_COMBOS:
        raise R197CorpusError("r197 corpus complete-action cap changed")
    if (manifest.get("eligibility") or {}).get("kaggle_replay_eligible") is not False:
        raise R197CorpusError("r197 corpus illegally grants Kaggle replay eligibility")
    base = path.parent
    outputs = manifest.get("outputs") or {}
    required_outputs = {
        "verified_identities": IDENTITIES_FILENAME,
        "episode_splits": SPLITS_FILENAME,
        "train": TRAIN_FILENAME,
        "heldout": HELDOUT_FILENAME,
        "action_space_too_large": TOO_LARGE_FILENAME,
    }
    for key, filename in required_outputs.items():
        entry = outputs.get(key) or {}
        candidate = base / str(entry.get("path") or "")
        if candidate.name != filename or not candidate.is_file():
            raise R197CorpusError(f"r197 corpus output is absent: {key}")
        if sha256_file(candidate) != entry.get("sha256"):
            raise R197CorpusError(f"r197 corpus output digest changed: {key}")
        if int(candidate.stat().st_size) != int(entry.get("bytes", -1)):
            raise R197CorpusError(f"r197 corpus output size changed: {key}")
        declared_rows = entry.get("rows")
        if (
            isinstance(declared_rows, bool)
            or not isinstance(declared_rows, int)
            or declared_rows < 0
        ):
            raise R197CorpusError(f"r197 corpus output has no valid declared row count: {key}")
        if _count_jsonl_rows(candidate) != declared_rows:
            raise R197CorpusError(f"r197 corpus output row count changed: {key}")

    counts = manifest.get("counts") or {}
    if not isinstance(counts, Mapping):
        raise R197CorpusError("r197 corpus manifest has no count summary")
    output_rows = {
        key: int((outputs.get(key) or {}).get("rows", -1))
        for key in required_outputs
    }
    expected_counts = {
        "verified_identities": output_rows["verified_identities"],
        "complete_action_rows": output_rows["train"] + output_rows["heldout"],
        "action_space_too_large_rows": output_rows["action_space_too_large"],
    }
    for key, expected in expected_counts.items():
        if int(counts.get(key, -1)) != expected:
            raise R197CorpusError(f"r197 corpus manifest count disagrees with JSONL: {key}")
    protected_identity_exact_summary = _validate_protected_identity_exact_summary(
        manifest.get("protected_identity_exact") or {}, counts
    )

    source_archives = manifest.get("source_archives")
    if not isinstance(source_archives, list):
        raise R197CorpusError("r197 corpus manifest has no source archive binding")
    archives_by_day: dict[str, Mapping[str, Any]] = {}
    for archive in source_archives:
        if not isinstance(archive, Mapping):
            raise R197CorpusError("r197 corpus source archive row is malformed")
        day = str(archive.get("source_day") or "")
        if day in archives_by_day:
            raise R197CorpusError(f"r197 corpus has duplicate source archive day: {day}")
        expected_sha256 = R197_RAW_ARCHIVE_SHA256_BY_DAY.get(day)
        if expected_sha256 is None:
            raise R197CorpusError(f"r197 corpus has an uncontracted source archive day: {day}")
        if (
            archive.get("source_archive")
            != f"pokemon-tcg-ai-battle-episodes-{day}.zip"
            or archive.get("expected_sha256") != expected_sha256
            or archive.get("sha256") != expected_sha256
        ):
            raise R197CorpusError(f"r197 corpus source archive identity changed: {day}")
        manifest_declared = archive.get("protected_manifest_source_archive_sha256")
        if manifest_declared is not None and manifest_declared != expected_sha256:
            raise R197CorpusError(
                f"r197 corpus protected manifest archive declaration changed: {day}"
            )
        feature_shards = archive.get("protected_feature_shards")
        if not isinstance(feature_shards, list) or not feature_shards:
            raise R197CorpusError(f"r197 corpus has no protected feature-shard evidence: {day}")
        for shard in feature_shards:
            if not isinstance(shard, Mapping) or not str(shard.get("path") or ""):
                raise R197CorpusError(f"r197 corpus feature-shard evidence is malformed: {day}")
            declared = shard.get("source_archive_sha256")
            if declared is not None and declared != expected_sha256:
                raise R197CorpusError(
                    f"r197 corpus protected shard archive declaration changed: {day}"
                )
        archives_by_day[day] = archive
    if set(archives_by_day) != set(R197_RAW_ARCHIVE_SHA256_BY_DAY):
        raise R197CorpusError("r197 corpus does not bind exactly the five source archives")
    if (
        manifest.get("derived_corpus_fingerprint")
        != _derived_corpus_fingerprint_from_manifest(manifest)
    ):
        raise R197CorpusError("r197 manifest derived corpus fingerprint mismatch")

    receipt_path = base / RECEIPT_FILENAME
    receipt = _read_json_object(receipt_path)
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise R197CorpusError("r197 receipt schema changed")
    manifest_entry = receipt.get("manifest") or {}
    if (
        manifest_entry.get("path") != MANIFEST_FILENAME
        or manifest_entry.get("sha256") != sha256_file(path)
        or int(manifest_entry.get("bytes", -1)) != int(path.stat().st_size)
    ):
        raise R197CorpusError("r197 receipt no longer binds its manifest")
    if receipt.get("derived_corpus_fingerprint") != manifest.get("derived_corpus_fingerprint"):
        raise R197CorpusError("r197 receipt corpus fingerprint mismatch")
    if receipt.get("source_pointer") != manifest.get("source_pointer"):
        raise R197CorpusError("r197 receipt source pointer binding mismatch")
    if receipt.get("protected_identity_exact") != protected_identity_exact_summary:
        raise R197CorpusError("r197 receipt protected identity provenance mismatch")
    if receipt.get("source_archives") != source_archives:
        raise R197CorpusError("r197 receipt source archive binding mismatch")
    if require_current_generator and receipt.get("generator_fingerprints") != _source_fingerprints():
        raise R197CorpusError("r197 corpus was built by different generator bytes")
    if archive_root is not None:
        root = Path(archive_root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        for archive in source_archives:
            candidate = _archive_path(root, str(archive.get("source_day") or ""))
            if (
                not candidate.is_file()
                or sha256_file(candidate) != archive.get("expected_sha256")
            ):
                raise R197CorpusError("r197 source archive identity changed")
    return receipt


def _validate_complete_action_row(row: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    if row.get("schema") != ROW_SCHEMA or row.get("corpus_schema") != CORPUS_SCHEMA:
        raise R197CorpusError("stream contains a non-r197 complete-action row")
    if row.get("split") != split:
        raise R197CorpusError("stream row appears in the wrong r197 split")
    if row.get("factorized_prefix_substitution") is not False:
        raise R197CorpusError("r197 row permits a factorized-prefix substitution")
    if row.get("action_space_source") != "runtime_complete_observation":
        raise R197CorpusError("r197 row did not use a runtime complete action space")
    identity = row.get("identity")
    deck = row.get("deck")
    if not isinstance(identity, Mapping) or not isinstance(deck, list):
        raise R197CorpusError("r197 row lacks serialized protected identity/deck")
    try:
        row_seat = int(row.get("seat", -1))
        identity_seat = int(identity.get("seat", -1))
        normalized_deck = [int(value) for value in deck]
    except (TypeError, ValueError, OverflowError) as exc:
        raise R197CorpusError("r197 row has malformed protected identity/deck") from exc
    deck_sha256 = canonical_json_sha256(normalized_deck)
    identity_without_fingerprint = dict(identity)
    identity_fingerprint = str(identity_without_fingerprint.pop("identity_fingerprint", ""))
    if (
        identity.get("schema") != IDENTITY_SCHEMA
        or str(identity.get("episode_id") or "") != str(row.get("episode_id") or "")
        or row_seat not in (0, 1)
        or identity_seat != row_seat
        or len(normalized_deck) != 60
        or str(identity.get("source_day") or "") != str(row.get("source_day") or "")
        or identity.get("deck_sha256") != deck_sha256
        or row.get("deck_sha256") != deck_sha256
        or identity_fingerprint != canonical_json_sha256(identity_without_fingerprint)
    ):
        raise R197CorpusError("r197 row protected identity/deck binding mismatch")
    protected_identity_exact = _validate_protected_identity_exact_fields(
        row.get("protected_identity_exact")
        if isinstance(row.get("protected_identity_exact"), Mapping)
        else {},
        episode_id=str(row.get("episode_id") or ""),
        seat=row_seat,
        deck_sha256=deck_sha256,
    )
    if (
        str(row.get("opp_archetype") or "").strip().casefold()
        != str(protected_identity_exact["opponent_current_rule_label"])
        .strip()
        .casefold()
    ):
        raise R197CorpusError("r197 row changed protected opponent archetype label")
    target_provenance = row.get("target_provenance")
    if (
        not isinstance(target_provenance, Mapping)
        or target_provenance.get("protected_identity_exact") != protected_identity_exact
    ):
        raise R197CorpusError("r197 row target provenance lost protected identity binding")
    observation = row.get("observation")
    legal_actions = row.get("legal_actions")
    if not isinstance(observation, Mapping) or not isinstance(legal_actions, list):
        raise R197CorpusError("r197 row lacks observation or legal_actions")
    expected_fingerprint = complete_action_space_fingerprint(
        observation, legal_actions, max_action_combos=MAX_ACTION_COMBOS
    )
    if row.get("action_space_fingerprint") != expected_fingerprint:
        raise R197CorpusError("r197 row action-space fingerprint mismatch")
    if row.get("observation_fingerprint") != canonical_json_sha256(dict(observation)):
        raise R197CorpusError("r197 row observation fingerprint mismatch")
    action = [int(value) for value in list(row.get("action") or ())]
    selected = int(row.get("selected_action_index", -1))
    normalised_actions = _normalise_actions(legal_actions)
    if not 0 <= selected < len(normalised_actions) or normalised_actions[selected] != action:
        raise R197CorpusError("r197 selected complete action cannot be mapped exactly")
    if row.get("evaluator_targets") is not None or row.get("unobserved_action_targets_present") is not False:
        raise R197CorpusError("r197 row fabricates unobserved evaluator targets")
    without_fingerprint = dict(row)
    supplied_fingerprint = str(without_fingerprint.pop("row_fingerprint", ""))
    if supplied_fingerprint != canonical_json_sha256(without_fingerprint):
        raise R197CorpusError("r197 row fingerprint mismatch")
    return dict(row)


def iter_complete_action_rows(
    manifest_path: Path,
    split: str,
    *,
    verify: bool = True,
    episode_ids: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield verified complete-action rows from the requested r197 split.

    The function is bounded-memory: it validates one JSONL row at a time.
    With ``verify=True`` (the default) it first validates manifest/receipt and
    the split file's content hash.  ``episode_ids`` lets a capped trainer scan
    the byte-verified stream but fully validate/yield only a preselected whole
    episode subset, before expensive action-space re-enumeration.
    """

    selected_split = str(split).strip().casefold()
    if selected_split not in {"train", "heldout"}:
        raise R197CorpusError("r197 split must be 'train' or 'heldout'")
    selected_episode_ids = (
        None if episode_ids is None else {str(episode_id) for episode_id in episode_ids}
    )
    path = _manifest_path(manifest_path)
    if verify:
        verify_r197_complete_action_manifest(path)
    manifest = _read_json_object(path)
    entry = (manifest.get("outputs") or {}).get(selected_split) or {}
    stream_path = path.parent / str(entry.get("path") or "")
    if not stream_path.is_file():
        raise R197CorpusError(f"r197 {selected_split} stream is missing")
    with stream_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise R197CorpusError(f"blank r197 JSONL row at {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise R197CorpusError(
                    f"invalid r197 JSONL row at {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise R197CorpusError(f"non-object r197 JSONL row at {line_number}")
            if (
                selected_episode_ids is not None
                and str(row.get("episode_id") or "") not in selected_episode_ids
            ):
                continue
            yield _validate_complete_action_row(row, split=selected_split)
