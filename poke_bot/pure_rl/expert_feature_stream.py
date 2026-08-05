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
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
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
from poke_bot.strategic_heads import EXPANDED_STRATEGIC_KEY


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


@dataclass(frozen=True)
class _ShardScanResult:
    shard: _VerifiedShard
    group_counts: tuple[tuple[str, int], ...]
    sequence_metadata: tuple[tuple[str, str, int, int, int], ...]
    sequences: int
    decisions: int
    packed_decisions: int
    truncated_sequences: int
    has_expanded_strategic_targets: bool


def _scan_verified_shard(
    task: tuple[Path, str, int, int, Optional[int]],
) -> _ShardScanResult:
    """Verify and scan one shard without retaining any sequence objects."""

    path, digest, records, start_index, max_context = task
    shard = _VerifiedShard.verify(path, digest, records)
    group_counts: dict[str, int] = {}
    sequence_metadata: list[tuple[str, str, int, int, int]] = []
    decisions = 0
    packed_decisions = 0
    truncated = 0
    has_expanded_strategic_targets = False
    loaded = 0
    shard.assert_unchanged()
    for local_index, sequence in enumerate(iter_feature_shard(shard.path)):
        episode_id = str(
            sequence.episode_id
            or f"__missing_episode_{int(start_index) + local_index}"
        )
        group_counts[episode_id] = group_counts.get(episode_id, 0) + 1
        raw_decisions = len(sequence.decisions)
        retained_decisions = (
            sequence.decisions[: int(max_context)]
            if max_context is not None
            else sequence.decisions
        )
        sequence_metadata.append(
            (
                episode_id,
                str(sequence.archetype or ""),
                int(sequence.seat),
                int(raw_decisions),
                int(len(retained_decisions)),
            )
        )
        decisions += raw_decisions
        packed_decisions += (
            min(raw_decisions, int(max_context))
            if max_context is not None
            else raw_decisions
        )
        has_expanded_strategic_targets = (
            has_expanded_strategic_targets
            or any(
                (decision.aux_labels or {}).get(EXPANDED_STRATEGIC_KEY)
                is not None
                for decision in retained_decisions
            )
        )
        truncated += int(
            max_context is not None
            and raw_decisions > int(max_context)
        )
        loaded += 1
    shard.assert_unchanged()
    if loaded != shard.records:
        raise ValueError(
            f"manifest count mismatch for {shard.path}: "
            f"expected={shard.records} loaded={loaded}"
        )
    return _ShardScanResult(
        shard=shard,
        group_counts=tuple(group_counts.items()),
        sequence_metadata=tuple(sequence_metadata),
        sequences=loaded,
        decisions=decisions,
        packed_decisions=packed_decisions,
        truncated_sequences=truncated,
        has_expanded_strategic_targets=has_expanded_strategic_targets,
    )


@dataclass(frozen=True)
class FeatureManifestShardView:
    """One immutable manifest shard with its deterministic split assignment."""

    shard: _VerifiedShard
    start_index: int
    validation_episode_ids: frozenset[str]
    max_context: Optional[int]

    def __len__(self) -> int:
        return int(self.shard.records)

    def __iter__(self) -> Iterator[tuple[bool, GameSequence]]:
        self.shard.assert_unchanged()
        loaded = 0
        for local_index, sequence in enumerate(
            iter_feature_shard(self.shard.path)
        ):
            episode_id = str(
                sequence.episode_id
                or f"__missing_episode_{self.start_index + local_index}"
            )
            if self.max_context is not None:
                from poke_bot.train import cap_game_sequence_context

                sequence, _changed = cap_game_sequence_context(
                    sequence, self.max_context
                )
            loaded += 1
            yield episode_id in self.validation_episode_ids, sequence
        self.shard.assert_unchanged()
        if loaded != self.shard.records:
            raise ValueError(
                f"manifest count mismatch for {self.shard.path}: "
                f"expected={self.shard.records} loaded={loaded}"
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
        return (
            self._plan.val_sequences
            if self._validation
            else self._plan.train_sequences
        )

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
        packed_decisions: int,
        train_sequences: int,
        val_sequences: int,
        max_context: Optional[int],
        truncated_sequences: int,
        shard_starts: tuple[int, ...],
        has_expanded_strategic_targets: bool,
        sequence_metadata: tuple[tuple[str, str, int, int, int], ...],
        source_sequences: int,
        source_decisions: int,
        source_packed_decisions: int,
        selected_indices: frozenset[int],
        exact_seat_split: dict[str, object],
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_digest = str(manifest_digest)
        self._shards = shards
        self._validation_episode_ids = validation_episode_ids
        self.sequences = int(sequences)
        self.decisions = int(decisions)
        self.packed_decisions = int(packed_decisions)
        self.train_sequences = int(train_sequences)
        self.val_sequences = int(val_sequences)
        self.max_context = (
            int(max_context) if max_context is not None else None
        )
        self.truncated_sequences = int(truncated_sequences)
        self._shard_starts = tuple(int(value) for value in shard_starts)
        self.has_expanded_strategic_targets = bool(
            has_expanded_strategic_targets
        )
        self._sequence_metadata = tuple(
            (
                str(episode_id),
                str(archetype_id),
                int(seat),
                int(raw_decisions),
                int(packed_rows),
            )
            for episode_id, archetype_id, seat, raw_decisions, packed_rows
            in sequence_metadata
        )
        self.source_sequences = int(source_sequences)
        self.source_decisions = int(source_decisions)
        self.source_packed_decisions = int(source_packed_decisions)
        self._selected_indices = frozenset(int(value) for value in selected_indices)
        self._exact_seat_split = dict(exact_seat_split)

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
        workers: int = 1,
        require_exact_seat_split: bool = False,
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
        group_counts: dict[str, int] = {}
        sequences = 0
        decisions = 0
        packed_decisions = 0
        truncated = 0
        has_expanded_strategic_targets = False
        sequence_metadata: list[tuple[str, str, int, int, int]] = []
        verified: list[_VerifiedShard] = []
        shard_starts: list[int] = []
        expected_cursor = 0
        scan_tasks: list[tuple[Path, str, int, int, Optional[int]]] = []
        for row in rows:
            shard_path = (path.parent / str(row.get("path") or "")).resolve()
            records = int((row.get("stats") or {}).get("records_kept", 0))
            shard_starts.append(expected_cursor)
            scan_tasks.append(
                (
                    shard_path,
                    str(row.get("sha256") or ""),
                    records,
                    expected_cursor,
                    context,
                )
            )
            expected_cursor += records

        requested_workers = max(1, int(workers))
        if requested_workers > 1 and len(scan_tasks) > 1:
            with ProcessPoolExecutor(
                max_workers=min(requested_workers, len(scan_tasks)),
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                scan_results = list(pool.map(_scan_verified_shard, scan_tasks))
            for result in scan_results:
                verified.append(result.shard)
                for episode_id, count in result.group_counts:
                    group_counts[episode_id] = (
                        group_counts.get(episode_id, 0) + int(count)
                    )
                sequence_metadata.extend(result.sequence_metadata)
                sequences += int(result.sequences)
                decisions += int(result.decisions)
                packed_decisions += int(result.packed_decisions)
                truncated += int(result.truncated_sequences)
                has_expanded_strategic_targets = (
                    has_expanded_strategic_targets
                    or result.has_expanded_strategic_targets
                )
        else:
            for task in scan_tasks:
                result = _scan_verified_shard(task)
                verified.append(result.shard)
                for episode_id, count in result.group_counts:
                    group_counts[episode_id] = (
                        group_counts.get(episode_id, 0) + int(count)
                    )
                sequence_metadata.extend(result.sequence_metadata)
                sequences += int(result.sequences)
                decisions += int(result.decisions)
                packed_decisions += int(result.packed_decisions)
                truncated += int(result.truncated_sequences)
                has_expanded_strategic_targets = (
                    has_expanded_strategic_targets
                    or result.has_expanded_strategic_targets
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
        source_sequences = int(sequences)
        source_decisions = int(decisions)
        source_packed_decisions = int(packed_decisions)
        selected_indices: set[int] = set(range(source_sequences))
        exact_seat_split: dict[str, object] = {}
        if require_exact_seat_split:
            invalid = [
                {"index": index, "seat": seat}
                for index, (_episode, _archetype, seat, _raw, _packed)
                in enumerate(sequence_metadata)
                if seat not in (0, 1)
            ]
            if invalid:
                raise ValueError(
                    "exact expert seat split encountered invalid seats: "
                    f"{invalid[:8]}"
                )

            partition_indices = {
                "train": [
                    index
                    for index, row in enumerate(sequence_metadata)
                    if row[0] not in val_ids
                ],
                "validation": [
                    index
                    for index, row in enumerate(sequence_metadata)
                    if row[0] in val_ids
                ],
            }
            selected_indices = set()
            partition_rows: dict[str, dict[str, object]] = {}
            assignment_rows: list[dict[str, object]] = []
            for partition, indices in partition_indices.items():
                by_seat = {
                    seat: [
                        index
                        for index in indices
                        if sequence_metadata[index][2] == seat
                    ]
                    for seat in (0, 1)
                }
                target = min(len(by_seat[0]), len(by_seat[1]))
                if target <= 0:
                    raise ValueError(
                        "exact expert seat split requires both seats in each "
                        f"partition: partition={partition} "
                        f"seat0={len(by_seat[0])} seat1={len(by_seat[1])}"
                    )

                kept: dict[int, list[int]] = {}
                for seat in (0, 1):
                    ranked = sorted(
                        by_seat[seat],
                        key=lambda index: hashlib.sha256(
                            (
                                f"{int(seed)}\0{partition}\0{index}\0"
                                f"{sequence_metadata[index][0]}\0{seat}"
                            ).encode("utf-8")
                        ).digest(),
                    )
                    kept[seat] = sorted(ranked[:target])
                    selected_indices.update(kept[seat])
                    assignment_rows.extend(
                        {
                            "partition": partition,
                            "source_index": index,
                            "episode_id": sequence_metadata[index][0],
                            "seat": seat,
                            "raw_decisions": sequence_metadata[index][3],
                            "packed_decisions": sequence_metadata[index][4],
                        }
                        for index in kept[seat]
                    )
                partition_rows[partition] = {
                    "source_seat0": len(by_seat[0]),
                    "source_seat1": len(by_seat[1]),
                    "first_games": target,
                    "second_games": target,
                    "total_games": target * 2,
                    "exact_even_split": True,
                }

            selected_order = sorted(selected_indices)
            sequences = len(selected_order)
            decisions = sum(sequence_metadata[index][3] for index in selected_order)
            packed_decisions = sum(
                sequence_metadata[index][4] for index in selected_order
            )
            train_sequences = sum(
                1
                for index in selected_order
                if sequence_metadata[index][0] not in val_ids
            )
            val_sequences = len(selected_order) - train_sequences
            canonical_assignment = json.dumps(
                sorted(
                    assignment_rows,
                    key=lambda row: (
                        str(row["partition"]), int(row["source_index"])
                    ),
                ),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            exact_seat_split = {
                "schema": "poke_bot.expert_rehearsal_seat_selection/v1",
                "policy": "exact_50_50_per_train_and_validation_partition",
                "split_seed": int(seed),
                "source": {
                    "games": source_sequences,
                    "seat0": sum(row[2] == 0 for row in sequence_metadata),
                    "seat1": sum(row[2] == 1 for row in sequence_metadata),
                },
                "partitions": partition_rows,
                "selected_games": len(selected_order),
                "selected_raw_decisions": int(decisions),
                "selected_packed_decisions": int(packed_decisions),
                "deterministic_assignment_manifest_sha256": (
                    "sha256:" + hashlib.sha256(canonical_assignment).hexdigest()
                ),
                "second_focus_1_to_7_used": False,
                "package_preference": "first_if_allowed",
                "passed": True,
            }

        return cls(
            manifest_path=path,
            manifest_digest=actual_manifest_digest,
            shards=tuple(verified),
            validation_episode_ids=frozenset(val_ids),
            sequences=sequences,
            decisions=decisions,
            packed_decisions=packed_decisions,
            train_sequences=sequences - val_sequences,
            val_sequences=val_sequences,
            max_context=context,
            truncated_sequences=truncated,
            shard_starts=tuple(shard_starts),
            has_expanded_strategic_targets=has_expanded_strategic_targets,
            sequence_metadata=tuple(sequence_metadata),
            source_sequences=source_sequences,
            source_decisions=source_decisions,
            source_packed_decisions=source_packed_decisions,
            selected_indices=frozenset(selected_indices),
            exact_seat_split=exact_seat_split,
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
            if index not in self._selected_indices:
                continue
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

    def partition_archetypes(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return acting-archetype IDs in the exact packed split order.

        The device corpus is materialized as every training sequence followed
        by every validation sequence.  Retaining this tiny metadata projection
        during the already-required immutable shard scan lets cumulative-core
        distillation bind each game to its matching frozen teacher without
        reopening the multi-gigabyte feature corpus or changing CPU-pack
        tensor schemas.
        """

        train: list[str] = []
        validation: list[str] = []
        for index, (episode_id, archetype_id, _seat, _raw, _packed) in enumerate(
            self._sequence_metadata
        ):
            if index not in self._selected_indices:
                continue
            target = (
                validation
                if episode_id in self._validation_episode_ids
                else train
            )
            target.append(archetype_id)
        if len(train) != self.train_sequences:
            raise RuntimeError(
                "training archetype metadata count changed: "
                f"expected={self.train_sequences} actual={len(train)}"
            )
        if len(validation) != self.val_sequences:
            raise RuntimeError(
                "validation archetype metadata count changed: "
                f"expected={self.val_sequences} actual={len(validation)}"
            )
        return tuple(train), tuple(validation)

    def partition_identity_rows(
        self,
    ) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
        """Return immutable identity metadata in exact packed split order.

        This deliberately exposes no replay actions or labels.  It is the
        checksum-bound bridge used by external provenance materializers to
        align one training-game weight with the resident temporal corpus.
        """

        train: list[dict[str, object]] = []
        validation: list[dict[str, object]] = []
        for index, (
            episode_id,
            archetype_id,
            seat,
            raw_decisions,
            packed_decisions,
        ) in enumerate(self._sequence_metadata):
            if index not in self._selected_indices:
                continue
            row: dict[str, object] = {
                "source_index": index,
                "episode_id": episode_id,
                "archetype_id": archetype_id,
                "seat": seat,
                "raw_decisions": raw_decisions,
                "packed_decisions": packed_decisions,
            }
            target = (
                validation
                if episode_id in self._validation_episode_ids
                else train
            )
            target.append(row)
        if len(train) != self.train_sequences:
            raise RuntimeError(
                "training identity metadata count changed: "
                f"expected={self.train_sequences} actual={len(train)}"
            )
        if len(validation) != self.val_sequences:
            raise RuntimeError(
                "validation identity metadata count changed: "
                f"expected={self.val_sequences} actual={len(validation)}"
            )
        return tuple(train), tuple(validation)

    def exact_seat_split_evidence(self) -> dict[str, object]:
        """Return the immutable selection projection for receipt binding."""

        return json.loads(json.dumps(self._exact_seat_split, sort_keys=True))

    def shard_views(self) -> tuple[FeatureManifestShardView, ...]:
        """Return manifest-ordered, independently iterable shard views."""

        return tuple(
            FeatureManifestShardView(
                shard=shard,
                start_index=start,
                validation_episode_ids=self._validation_episode_ids,
                max_context=self.max_context,
            )
            for shard, start in zip(
                self._shards, self._shard_starts, strict=True
            )
        )
