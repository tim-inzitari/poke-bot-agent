"""Bounded, reiterable views over an immutable expert feature manifest.

The ordinary manifest loader returns a ``BootstrapDataset`` containing every
``GameSequence``.  That is convenient for small corpora but makes the first
expert-pack build scale with the Python object graph rather than with the
compact packed tensors.  This module makes one metadata-only pass, retains only
episode counts and the validation episode ids, and then re-opens the verified
shards for the train and validation passes.

The partition deliberately mirrors ``split_dataset(...,
group_by_episode=True)``: episode ids retain first-seen order, the same Python
``random.Random(seed).shuffle`` is used, validation is filled by whole episodes,
and at least one episode remains in training.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from poke_bot.dataset import GameSequence
from poke_bot.feature_shards import (
    COMPACT_MODE,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    SUPPORTED_COMPACT_MODES,
    iter_feature_shard,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = Path(path).stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


@dataclass(frozen=True)
class _VerifiedShard:
    path: Path
    digest: str
    records: int
    stat_identity: tuple[int, int, int, int, int]

    @classmethod
    def verify(cls, path: Path, digest: str, records: int) -> "_VerifiedShard":
        path = Path(path).resolve()
        expected = str(digest)
        expected_records = int(records)
        if expected_records <= 0:
            raise ValueError(f"feature shard has invalid record count: {path}")
        before = _stat_identity(path)
        if _sha256(path) != expected:
            raise ValueError(f"feature shard digest mismatch: {path}")
        after = _stat_identity(path)
        if after != before:
            raise ValueError(f"feature shard changed during verification: {path}")
        return cls(
            path=path,
            digest=expected,
            records=expected_records,
            stat_identity=after,
        )

    def assert_unchanged(self) -> None:
        actual = _stat_identity(self.path)
        if actual != self.stat_identity:
            raise ValueError(
                f"checksummed feature shard changed after verification: {self.path}"
            )


class FeatureManifestSplitView:
    """A sized iterable that re-opens, validates, and filters every pass."""

    def __init__(
        self,
        plan: "EpisodeGroupedFeatureManifest",
        *,
        validation: bool,
    ) -> None:
        self._plan = plan
        self._validation = bool(validation)

    def __len__(self) -> int:
        return self._plan.val_sequences if self._validation else self._plan.train_sequences

    def __iter__(self) -> Iterator[GameSequence]:
        return self._plan.iter_partition(validation=self._validation)


class EpisodeGroupedFeatureManifest:
    """Metadata-only split plan for one checksummed feature manifest."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        manifest_digest: str,
        shards: tuple[_VerifiedShard, ...],
        validation_episode_ids: frozenset[str],
        sequences: int,
        decisions: int,
        train_sequences: int,
        val_sequences: int,
        max_context: Optional[int],
        truncated_sequences: int,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_digest = str(manifest_digest)
        self._shards = shards
        self._validation_episode_ids = validation_episode_ids
        self.sequences = int(sequences)
        self.decisions = int(decisions)
        self.train_sequences = int(train_sequences)
        self.val_sequences = int(val_sequences)
        self.max_context = (
            int(max_context) if max_context is not None else None
        )
        self.truncated_sequences = int(truncated_sequences)

    @classmethod
    def open(
        cls,
        manifest_path: Path,
        *,
        expected_manifest_digest: str,
        val_frac: float,
        seed: int,
        max_context: Optional[int],
        expected_compact_mode: Optional[str] = None,
    ) -> "EpisodeGroupedFeatureManifest":
        path = Path(manifest_path).expanduser().resolve()
        manifest_bytes = path.read_bytes()
        actual_manifest_digest = (
            "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
        )
        if actual_manifest_digest != str(expected_manifest_digest):
            raise ValueError(
                "expert feature manifest digest mismatch: "
                f"expected={expected_manifest_digest} "
                f"actual={actual_manifest_digest}"
            )
        fraction = float(val_frac)
        if not 0.0 < fraction < 1.0:
            # The production expert contract always has a validation split.
            # Refuse a subtly different ordering for the generic non-grouped
            # fallback used by split_dataset when val_frac <= 0.
            raise ValueError(
                "streamed episode-grouped split requires val_frac in (0, 1)"
            )
        context = int(max_context) if max_context is not None else None
        if context is not None and context <= 0:
            raise ValueError("expert max_context must be positive")

        payload = json.loads(manifest_bytes)
        if payload.get("format") != MANIFEST_FORMAT:
            raise ValueError("invalid feature manifest format")
        if int(payload.get("format_version", -1)) != MANIFEST_FORMAT_VERSION:
            raise ValueError("unsupported feature manifest version")
        compact_mode = str(payload.get("compact_mode") or COMPACT_MODE)
        if compact_mode not in SUPPORTED_COMPACT_MODES:
            raise ValueError(f"unsupported manifest compact mode: {compact_mode}")
        if (
            expected_compact_mode is not None
            and compact_mode != str(expected_compact_mode)
        ):
            raise ValueError(
                "expert manifest compact mode mismatch: "
                f"expected={expected_compact_mode!r} actual={compact_mode!r}"
            )

        rows = list(payload.get("shards") or ())
        if not rows:
            raise ValueError("feature manifest contains no shards")
        verified: list[_VerifiedShard] = []
        for row in rows:
            shard_path = (path.parent / str(row.get("path") or "")).resolve()
            verified.append(
                _VerifiedShard.verify(
                    shard_path,
                    str(row.get("sha256") or ""),
                    int((row.get("stats") or {}).get("records_kept", 0)),
                )
            )

        group_counts: dict[str, int] = {}
        sequences = 0
        decisions = 0
        truncated = 0
        for index, sequence in cls._iter_verified(tuple(verified)):
            episode_id = str(
                sequence.episode_id or f"__missing_episode_{index}"
            )
            group_counts[episode_id] = group_counts.get(episode_id, 0) + 1
            sequences += 1
            decisions += len(sequence.decisions)
            truncated += int(
                context is not None and len(sequence.decisions) > context
            )

        expected_records = sum(shard.records for shard in verified)
        manifest_records = int(
            (payload.get("totals") or {}).get("records_kept", -1)
        )
        if manifest_records != expected_records:
            raise ValueError(
                "feature manifest record total mismatch: "
                f"totals={manifest_records} shards={expected_records}"
            )
        if sequences != expected_records:
            raise ValueError(
                "feature manifest sequence count mismatch: "
                f"expected={expected_records} loaded={sequences}"
            )
        expected_decisions = int(
            (payload.get("totals") or {}).get("decisions_kept", -1)
        )
        if expected_decisions <= 0:
            raise ValueError("feature manifest has invalid decision total")
        if decisions != expected_decisions:
            raise ValueError(
                "feature manifest decision count mismatch: "
                f"expected={expected_decisions} loaded={decisions}"
            )
        if sequences <= 0:
            raise ValueError("feature manifest contains no sequences")

        group_ids = list(group_counts)
        rng = random.Random(int(seed))
        rng.shuffle(group_ids)
        val_ids: set[str] = set()
        if sequences > 1 and len(group_ids) > 1:
            target = max(1, int(sequences * fraction))
            selected = 0
            for episode_id in group_ids[:-1]:
                val_ids.add(episode_id)
                selected += int(group_counts[episode_id])
                if selected >= target:
                    break
        val_sequences = sum(group_counts[value] for value in val_ids)
        return cls(
            manifest_path=path,
            manifest_digest=actual_manifest_digest,
            shards=tuple(verified),
            validation_episode_ids=frozenset(val_ids),
            sequences=sequences,
            decisions=decisions,
            train_sequences=sequences - val_sequences,
            val_sequences=val_sequences,
            max_context=context,
            truncated_sequences=truncated,
        )

    @staticmethod
    def _iter_verified(
        shards: tuple[_VerifiedShard, ...],
    ) -> Iterator[tuple[int, GameSequence]]:
        index = 0
        for shard in shards:
            shard.assert_unchanged()
            loaded = 0
            for sequence in iter_feature_shard(shard.path):
                yield index, sequence
                index += 1
                loaded += 1
            shard.assert_unchanged()
            if loaded != shard.records:
                raise ValueError(
                    f"manifest count mismatch for {shard.path}: "
                    f"expected={shard.records} loaded={loaded}"
                )

    def iter_partition(self, *, validation: bool) -> Iterator[GameSequence]:
        emitted = 0
        for index, sequence in self._iter_verified(self._shards):
            episode_id = str(
                sequence.episode_id or f"__missing_episode_{index}"
            )
            is_validation = episode_id in self._validation_episode_ids
            if is_validation != bool(validation):
                continue
            if self.max_context is not None:
                from poke_bot.train import cap_game_sequence_context

                sequence, _changed = cap_game_sequence_context(
                    sequence, self.max_context
                )
            emitted += 1
            yield sequence
        expected = self.val_sequences if validation else self.train_sequences
        if emitted != expected:
            raise ValueError(
                "streamed feature split count changed: "
                f"expected={expected} emitted={emitted}"
            )

    def splits(self) -> tuple[FeatureManifestSplitView, FeatureManifestSplitView]:
        return (
            FeatureManifestSplitView(self, validation=False),
            FeatureManifestSplitView(self, validation=True),
        )
