"""Bounded deterministic multicore construction of expert CPU packs.

Workers read independent immutable feature shards, build one ordinary packed
corpus in source order, split that packed result into train/validation
fragments, and persist the fragments to a private checksum-backed spool.
The parent merges all train fragments in manifest order followed by all
validation fragments in manifest order.  Worker completion order therefore
cannot change tensor order or split boundaries.

This module is intentionally used only when an explicit worker count greater
than one is supplied.  The serial builder remains the default.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import tempfile
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

from poke_bot.device_corpus import (
    DEVICE_CORPUS_TENSOR_FIELDS,
    DeviceResidentBootstrapCorpus,
)
from poke_bot.pure_rl.expert_cpu_pack import validate_cpu_corpus
from poke_bot.pure_rl.expert_feature_stream import FeatureManifestShardView


FRAGMENT_SCHEMA = "poke_bot.expert_cpu_pack_fragment/v1"
_GIB = 2**30

_SAMPLE_DIRECT_FIELDS = frozenset(
    {
        "n_options",
        "target_index",
        "value_target",
        "sample_aux_class",
        "guide_target_index",
        "guide_confidence",
        "select_context",
        "selected_is_stop",
        "strategic_action_q_target",
        "strategic_action_q_mask",
        "strategic_action_factor_mask",
        "strategic_action_utility_target",
        "strategic_action_utility_mask",
    }
)
_DECISION_DIRECT_FIELDS = frozenset(
    {
        "hand_present",
        "remainder_present",
        "lethal_target",
        "prize_race_target",
        "strategic_tactical_outcome_target",
        "strategic_tactical_outcome_mask",
        "strategic_opponent_response_target",
        "strategic_opponent_response_mask",
        "strategic_resource_forecast_target",
        "strategic_resource_forecast_mask",
        "strategic_game_phase_target",
        "strategic_game_phase_mask",
        "strategic_outcome_class_target",
        "strategic_outcome_class_mask",
        "strategic_remaining_turns_target",
        "strategic_remaining_turns_mask",
    }
)
_FLAT_DIRECT_FIELDS = frozenset(
    {
        "board_index",
        "board_value",
        "option_index",
        "option_value",
        "action_index",
        "action_value",
        "hand_index",
        "remainder_index",
    }
)
_OFFSET_FIELDS = frozenset(
    {
        "board_offset",
        "option_offset",
        "action_offset",
        "hand_offset",
        "remainder_offset",
        "game_decision_offset",
        "game_sample_offset",
    }
)
_REBASED_FIELDS = frozenset({"sample_board", "option_word_start"})
_KNOWN_FIELDS = (
    _SAMPLE_DIRECT_FIELDS
    | _DECISION_DIRECT_FIELDS
    | _FLAT_DIRECT_FIELDS
    | _OFFSET_FIELDS
    | _REBASED_FIELDS
)


@dataclass(frozen=True)
class FragmentDescriptor:
    ordinal: int
    partition: str
    payload: str
    manifest: str
    payload_bytes: int
    payload_sha256: str
    scalar_state: dict[str, Any]
    tensor_specs: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ShardBuildResult:
    ordinal: int
    train: Optional[FragmentDescriptor]
    validation: Optional[FragmentDescriptor]
    source_games: int
    elapsed_sec: float


class _RecordingShardIterable:
    """Yield one shard once while retaining only its validation flags."""

    def __init__(self, view: FeatureManifestShardView) -> None:
        self.view = view
        self.validation_flags: list[bool] = []
        self._iterated = False

    def __len__(self) -> int:
        return len(self.view)

    def __iter__(self) -> Iterator[Any]:
        if self._iterated:
            raise RuntimeError("parallel expert shard iterable is single-use")
        self._iterated = True
        for validation, sequence in self.view:
            self.validation_flags.append(bool(validation))
            yield sequence


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(
        f".{path.name}.partial.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _tensor_specs(
    tensors: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": str(tensor.dtype),
            "shape": [int(value) for value in tensor.shape],
            "nbytes": int(tensor.numel()) * int(tensor.element_size()),
        }
        for name, tensor in sorted(tensors.items())
    }


def _runs(flags: list[bool], *, validation: bool) -> tuple[tuple[int, int], ...]:
    selected = bool(validation)
    result: list[tuple[int, int]] = []
    start: Optional[int] = None
    for index, value in enumerate((*flags, not selected)):
        if bool(value) == selected and start is None:
            start = index
        elif bool(value) != selected and start is not None:
            result.append((start, index))
            start = None
    return tuple(result)


def _cat(
    pieces: list[torch.Tensor],
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    if not pieces:
        return torch.empty(
            (0, *like.shape[1:]),
            dtype=like.dtype,
            device="cpu",
        )
    return torch.cat(pieces, dim=0).contiguous()


def _selected_corpus(
    source: DeviceResidentBootstrapCorpus,
    runs: tuple[tuple[int, int], ...],
    *,
    validation: bool,
) -> Optional[DeviceResidentBootstrapCorpus]:
    """Copy manifest-ordered game runs into one standalone CPU fragment."""

    if not runs:
        return None
    if source.device.type != "cpu":
        raise ValueError("parallel expert fragments must originate on CPU")
    tensors = source.tensor_state()
    unknown = set(tensors) - _KNOWN_FIELDS
    if unknown:
        raise ValueError(
            f"parallel expert selector has unknown tensors: {sorted(unknown)}"
        )
    if source.game_decision_offset is None or source.game_sample_offset is None:
        raise ValueError("parallel expert source lacks temporal game offsets")

    direct_pieces: dict[str, list[torch.Tensor]] = {
        name: [] for name in tensors if name not in _OFFSET_FIELDS
    }
    offset_pieces: dict[str, list[torch.Tensor]] = {
        name: [] for name in tensors if name in _OFFSET_FIELDS
    }
    decisions_out = 0
    samples_out = 0
    games_out = 0
    option_words_out = 0
    flat_bases = {
        "board": 0,
        "option": 0,
        "action": 0,
        "hand": 0,
        "remainder": 0,
    }

    def append_csr(
        prefix: str,
        *,
        row_start: int,
        row_end: int,
    ) -> None:
        index_name = f"{prefix}_index"
        value_name = f"{prefix}_value"
        offset_name = f"{prefix}_offset"
        offset = tensors[offset_name]
        flat_start = int(offset[row_start].item())
        flat_end = int(offset[row_end].item())
        direct_pieces[index_name].append(
            tensors[index_name][flat_start:flat_end]
        )
        if value_name in tensors:
            direct_pieces[value_name].append(
                tensors[value_name][flat_start:flat_end]
            )
        offset_pieces[offset_name].append(
            offset[row_start + 1 : row_end + 1]
            - flat_start
            + flat_bases[prefix]
        )
        flat_bases[prefix] += flat_end - flat_start

    for game_start, game_end in runs:
        decision_start = int(source.game_decision_offset[game_start].item())
        decision_end = int(source.game_decision_offset[game_end].item())
        sample_start = int(source.game_sample_offset[game_start].item())
        sample_end = int(source.game_sample_offset[game_end].item())
        decision_count = decision_end - decision_start
        sample_count = sample_end - sample_start

        append_csr(
            "board",
            row_start=decision_start * 24,
            row_end=decision_end * 24,
        )
        append_csr(
            "action",
            row_start=decision_start,
            row_end=decision_end,
        )
        if sample_count:
            option_word_start = int(
                source.option_word_start[sample_start].item()
            )
            option_words = int(
                source.n_options[sample_start:sample_end]
                .to(dtype=torch.int64)
                .sum()
                .item()
            )
        else:
            option_word_start = int(
                source.option_offset.numel() - 1
                if sample_start == source.total_samples
                else source.option_word_start[sample_start].item()
            )
            option_words = 0
        append_csr(
            "option",
            row_start=option_word_start,
            row_end=option_word_start + option_words,
        )
        if "hand_offset" in tensors:
            append_csr(
                "hand",
                row_start=decision_start,
                row_end=decision_end,
            )
            append_csr(
                "remainder",
                row_start=decision_start,
                row_end=decision_end,
            )

        for name in _DECISION_DIRECT_FIELDS & tensors.keys():
            direct_pieces[name].append(
                tensors[name][decision_start:decision_end]
            )
        for name in _SAMPLE_DIRECT_FIELDS & tensors.keys():
            direct_pieces[name].append(tensors[name][sample_start:sample_end])
        direct_pieces["sample_board"].append(
            tensors["sample_board"][sample_start:sample_end]
            - decision_start
            + decisions_out
        )
        direct_pieces["option_word_start"].append(
            tensors["option_word_start"][sample_start:sample_end]
            - option_word_start
            + option_words_out
        )
        offset_pieces["game_decision_offset"].append(
            tensors["game_decision_offset"][game_start + 1 : game_end + 1]
            - decision_start
            + decisions_out
        )
        offset_pieces["game_sample_offset"].append(
            tensors["game_sample_offset"][game_start + 1 : game_end + 1]
            - sample_start
            + samples_out
        )

        decisions_out += decision_count
        samples_out += sample_count
        option_words_out += option_words
        games_out += game_end - game_start

    selected: dict[str, torch.Tensor] = {}
    for name, pieces in direct_pieces.items():
        selected[name] = _cat(pieces, like=tensors[name])
    for name, pieces in offset_pieces.items():
        zero = torch.zeros(1, dtype=tensors[name].dtype)
        selected[name] = torch.cat([zero, *pieces], dim=0).contiguous()

    tensor_bytes = sum(
        int(value.numel()) * int(value.element_size())
        for value in selected.values()
    )
    train_games = 0 if validation else games_out
    val_games = games_out if validation else 0
    train_samples = 0 if validation else samples_out
    val_samples = samples_out if validation else 0
    corpus = DeviceResidentBootstrapCorpus.from_packed_state(
        tensors=selected,
        scalars={
            "train_samples": train_samples,
            "val_samples": val_samples,
            "train_games": train_games,
            "val_games": val_games,
            "decisions": decisions_out,
            "input_bytes": tensor_bytes,
            "build_seconds": float(source.build_seconds),
            "belief_card_vocab": int(source.belief_card_vocab),
            "expanded_strategic_schema": (
                source.expanded_strategic_schema
            ),
            "expanded_strategic_schema_version": int(
                source.expanded_strategic_schema_version
            ),
            "expanded_strategic_schema_digest": (
                source.expanded_strategic_schema_digest
            ),
        },
    )
    validate_cpu_corpus(corpus, allow_empty_training_fragment=True)
    return corpus


def _write_fragment(
    spool: Path,
    *,
    ordinal: int,
    partition: str,
    corpus: DeviceResidentBootstrapCorpus,
) -> FragmentDescriptor:
    validate_cpu_corpus(corpus, allow_empty_training_fragment=True)
    stem = f"shard-{int(ordinal):05d}-{partition}"
    payload_path = spool / f"{stem}.pt"
    manifest_path = spool / f"{stem}.json"
    partial = payload_path.with_name(
        f".{payload_path.name}.partial.{os.getpid()}.{time.time_ns()}"
    )
    tensors = corpus.tensor_state()
    scalars = corpus.scalar_state()
    payload = {
        "schema": FRAGMENT_SCHEMA,
        "ordinal": int(ordinal),
        "partition": str(partition),
        "scalars": scalars,
        "tensors": tensors,
    }
    try:
        with partial.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        payload_bytes = int(partial.stat().st_size)
        payload_digest = _sha256_file(partial)
        os.replace(partial, payload_path)
        _fsync_dir(spool)
        manifest = {
            "schema": FRAGMENT_SCHEMA,
            "ordinal": int(ordinal),
            "partition": str(partition),
            "payload": payload_path.name,
            "payload_bytes": payload_bytes,
            "payload_sha256": payload_digest,
            "scalar_state": scalars,
            "tensor_specs": _tensor_specs(tensors),
        }
        _atomic_json(manifest_path, manifest)
    finally:
        partial.unlink(missing_ok=True)
    return FragmentDescriptor(
        ordinal=int(ordinal),
        partition=str(partition),
        payload=str(payload_path),
        manifest=str(manifest_path),
        payload_bytes=payload_bytes,
        payload_sha256=payload_digest,
        scalar_state=scalars,
        tensor_specs=_tensor_specs(tensors),
    )


def _build_shard(
    task: tuple[
        int,
        FeatureManifestShardView,
        Path,
        Optional[int],
        bool,
    ],
) -> ShardBuildResult:
    ordinal, view, spool, exact_card_vocab, force_strategic = task
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    started = time.monotonic()
    recording = _RecordingShardIterable(view)
    source = DeviceResidentBootstrapCorpus.from_splits(
        recording,
        (),
        device=torch.device("cpu"),
        exact_card_vocab=exact_card_vocab,
        force_expanded_strategic=bool(force_strategic),
    )
    if len(recording.validation_flags) != len(view):
        raise RuntimeError("parallel expert worker did not cover its full shard")
    train = _selected_corpus(
        source,
        _runs(recording.validation_flags, validation=False),
        validation=False,
    )
    validation = _selected_corpus(
        source,
        _runs(recording.validation_flags, validation=True),
        validation=True,
    )
    train_descriptor = (
        _write_fragment(
            spool,
            ordinal=ordinal,
            partition="train",
            corpus=train,
        )
        if train is not None
        else None
    )
    validation_descriptor = (
        _write_fragment(
            spool,
            ordinal=ordinal,
            partition="validation",
            corpus=validation,
        )
        if validation is not None
        else None
    )
    del source, train, validation
    gc.collect()
    return ShardBuildResult(
        ordinal=int(ordinal),
        train=train_descriptor,
        validation=validation_descriptor,
        source_games=len(view),
        elapsed_sec=time.monotonic() - started,
    )


def _descriptor_from_manifest(path: Path) -> FragmentDescriptor:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != FRAGMENT_SCHEMA:
        raise RuntimeError("parallel expert fragment manifest schema mismatch")
    payload_path = (Path(path).parent / str(raw.get("payload") or "")).resolve()
    return FragmentDescriptor(
        ordinal=int(raw.get("ordinal", -1)),
        partition=str(raw.get("partition") or ""),
        payload=str(payload_path),
        manifest=str(Path(path).resolve()),
        payload_bytes=int(raw.get("payload_bytes", -1)),
        payload_sha256=str(raw.get("payload_sha256") or ""),
        scalar_state=dict(raw.get("scalar_state") or {}),
        tensor_specs=dict(raw.get("tensor_specs") or {}),
    )


def _load_fragment(
    descriptor: FragmentDescriptor,
) -> DeviceResidentBootstrapCorpus:
    persisted = _descriptor_from_manifest(Path(descriptor.manifest))
    if persisted != descriptor:
        raise RuntimeError("parallel expert fragment descriptor changed")
    payload_path = Path(descriptor.payload)
    if (
        not payload_path.is_file()
        or int(payload_path.stat().st_size) != descriptor.payload_bytes
        or _sha256_file(payload_path) != descriptor.payload_sha256
    ):
        raise RuntimeError("parallel expert fragment payload checksum mismatch")
    try:
        try:
            payload = torch.load(
                payload_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except TypeError:
            payload = torch.load(
                payload_path,
                map_location="cpu",
                weights_only=True,
            )
    except Exception as exc:
        raise RuntimeError(
            f"parallel expert fragment cannot be loaded: {exc}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != FRAGMENT_SCHEMA
        or int(payload.get("ordinal", -1)) != descriptor.ordinal
        or str(payload.get("partition") or "") != descriptor.partition
    ):
        raise RuntimeError("parallel expert fragment payload contract mismatch")
    tensors = payload.get("tensors")
    scalars = payload.get("scalars")
    if not isinstance(tensors, dict) or not isinstance(scalars, dict):
        raise RuntimeError("parallel expert fragment payload shape is invalid")
    if _tensor_specs(tensors) != descriptor.tensor_specs:
        raise RuntimeError("parallel expert fragment tensor inventory changed")
    if scalars != descriptor.scalar_state:
        raise RuntimeError("parallel expert fragment scalar state changed")
    corpus = DeviceResidentBootstrapCorpus.from_packed_state(
        tensors=tensors,
        scalars=scalars,
    )
    validate_cpu_corpus(corpus, allow_empty_training_fragment=True)
    return corpus


def _dtype(name: str) -> torch.dtype:
    raw = str(name)
    if not raw.startswith("torch."):
        raise RuntimeError(f"invalid parallel fragment dtype: {raw}")
    value = getattr(torch, raw.removeprefix("torch."), None)
    if not isinstance(value, torch.dtype):
        raise RuntimeError(f"unsupported parallel fragment dtype: {raw}")
    return value


def _validate_descriptor_sequence(
    descriptors: list[FragmentDescriptor],
) -> None:
    if not descriptors:
        raise RuntimeError("parallel expert build produced no fragments")
    seen: set[tuple[int, str]] = set()
    prior_partition = "train"
    prior_ordinal = -1
    for descriptor in descriptors:
        identity = (descriptor.ordinal, descriptor.partition)
        if identity in seen:
            raise RuntimeError("parallel expert build has a duplicate fragment")
        seen.add(identity)
        if descriptor.partition not in {"train", "validation"}:
            raise RuntimeError("parallel expert fragment partition is invalid")
        if (
            prior_partition == "validation"
            and descriptor.partition == "train"
        ):
            raise RuntimeError("parallel expert train fragment follows validation")
        if descriptor.partition != prior_partition:
            prior_partition = descriptor.partition
            prior_ordinal = -1
        if descriptor.ordinal <= prior_ordinal:
            raise RuntimeError("parallel expert fragment order is not canonical")
        prior_ordinal = descriptor.ordinal


def _merged_specs(
    descriptors: list[FragmentDescriptor],
) -> dict[str, dict[str, Any]]:
    first = descriptors[0].tensor_specs
    inventory = set(first)
    if inventory - set(DEVICE_CORPUS_TENSOR_FIELDS):
        raise RuntimeError("parallel expert fragment has unknown tensors")
    merged: dict[str, dict[str, Any]] = {}
    for descriptor in descriptors:
        if set(descriptor.tensor_specs) != inventory:
            raise RuntimeError(
                "parallel expert fragment tensor inventories disagree"
            )
    for name in sorted(inventory):
        rows = []
        dtype = str(first[name].get("dtype") or "")
        tail = list(first[name].get("shape") or [])[1:]
        for descriptor in descriptors:
            spec = dict(descriptor.tensor_specs[name])
            shape = list(spec.get("shape") or ())
            if (
                not shape
                or str(spec.get("dtype") or "") != dtype
                or shape[1:] != tail
            ):
                raise RuntimeError(
                    f"parallel expert fragment shape disagrees for {name}"
                )
            rows.append(int(shape[0]))
        leading = (
            1 + sum(value - 1 for value in rows)
            if name in _OFFSET_FIELDS
            else sum(rows)
        )
        if leading < 0 or leading >= 2**63:
            raise MemoryError(
                f"parallel expert merged shape is invalid for {name}"
            )
        merged[name] = {
            "dtype": dtype,
            "shape": [leading, *tail],
        }
    return merged


def _validate_merge_capacity(
    descriptors: list[FragmentDescriptor],
    specs: dict[str, dict[str, Any]],
) -> None:
    decisions = sum(
        int(descriptor.scalar_state.get("decisions", -1))
        for descriptor in descriptors
    )
    samples = sum(
        int(descriptor.scalar_state.get("train_samples", -1))
        + int(descriptor.scalar_state.get("val_samples", -1))
        for descriptor in descriptors
    )
    games = sum(
        int(descriptor.scalar_state.get("train_games", -1))
        + int(descriptor.scalar_state.get("val_games", -1))
        for descriptor in descriptors
    )
    if min(decisions, samples, games) < 0:
        raise RuntimeError("parallel expert fragment has negative counters")
    if decisions >= 2**31 or samples >= 2**31 or games >= 2**31:
        raise MemoryError("parallel expert merged rows exceed signed int32")
    expected_rows = {
        "board_offset": decisions * 24 + 1,
        "action_offset": decisions + 1,
        "game_decision_offset": games + 1,
        "game_sample_offset": games + 1,
        "sample_board": samples,
        "option_word_start": samples,
        "n_options": samples,
        "target_index": samples,
        "value_target": samples,
        "guide_target_index": samples,
        "guide_confidence": samples,
        "select_context": samples,
        "selected_is_stop": samples,
    }
    if "hand_offset" in specs:
        expected_rows["hand_offset"] = decisions + 1
        expected_rows["remainder_offset"] = decisions + 1
    for name, expected in expected_rows.items():
        actual = int(specs[name]["shape"][0])
        if actual != int(expected):
            raise RuntimeError(
                f"parallel expert merged row contract disagrees for {name}: "
                f"{actual} != {expected}"
            )
        if actual >= 2**31:
            raise MemoryError(
                f"parallel expert merged {name} exceeds signed int32"
            )
    for name in (
        "board_index",
        "option_index",
        "action_index",
        "hand_index",
        "remainder_index",
        "option_offset",
    ):
        if name in specs and int(specs[name]["shape"][0]) >= 2**31:
            raise MemoryError(
                f"parallel expert merged {name} exceeds signed int32"
            )


def _allocate_merged(
    specs: dict[str, dict[str, Any]],
) -> dict[str, torch.Tensor]:
    tensors = {
        name: torch.empty(
            tuple(int(value) for value in spec["shape"]),
            dtype=_dtype(str(spec["dtype"])),
            device="cpu",
        )
        for name, spec in specs.items()
    }
    for name in _OFFSET_FIELDS & tensors.keys():
        tensors[name][0] = 0
    return tensors


def _tensor_spec_bytes(spec: dict[str, Any]) -> int:
    shape = tuple(int(value) for value in spec.get("shape") or ())
    if not shape or any(value < 0 for value in shape):
        raise RuntimeError("parallel expert tensor has an invalid shape")
    elements = math.prod(shape)
    return int(elements) * int(
        torch.empty((), dtype=_dtype(str(spec.get("dtype") or ""))).element_size()
    )


def _required_merge_memory_bytes(
    descriptors: list[FragmentDescriptor],
    specs: dict[str, dict[str, Any]],
    *,
    memory_reserve_gib: float,
) -> int:
    output_bytes = sum(_tensor_spec_bytes(spec) for spec in specs.values())
    largest_fragment_bytes = max(
        sum(
            _tensor_spec_bytes(spec)
            for spec in descriptor.tensor_specs.values()
        )
        for descriptor in descriptors
    )
    return (
        output_bytes
        + largest_fragment_bytes
        + int(float(memory_reserve_gib) * _GIB)
    )


def _merge_fragments(
    descriptors: list[FragmentDescriptor],
    *,
    build_seconds: float,
    memory_reserve_gib: float,
) -> DeviceResidentBootstrapCorpus:
    """Merge verified whole-partition fragments with bounded mapped RSS."""

    _validate_descriptor_sequence(descriptors)
    specs = _merged_specs(descriptors)
    _validate_merge_capacity(descriptors, specs)
    available = _available_memory_bytes()
    required = _required_merge_memory_bytes(
        descriptors,
        specs,
        memory_reserve_gib=memory_reserve_gib,
    )
    if available is not None and available < required:
        raise MemoryError(
            "parallel expert merge cannot preserve its memory reserve: "
            f"available={available} required={required}"
        )
    output = _allocate_merged(specs)
    direct_cursors = {
        name: 0
        for name in output
        if name not in _OFFSET_FIELDS
        and name not in _REBASED_FIELDS
    }
    decision_base = 0
    sample_base = 0
    game_base = 0
    option_word_base = 0
    nnz_bases = {
        "board": 0,
        "option": 0,
        "action": 0,
        "hand": 0,
        "remainder": 0,
    }
    train_games = 0
    val_games = 0
    train_samples = 0
    val_samples = 0
    belief_card_vocab: Optional[int] = None
    strategic_identity: Optional[tuple[str, int, str]] = None

    def copy_direct(name: str, source: torch.Tensor) -> None:
        start = direct_cursors[name]
        end = start + int(source.shape[0])
        output[name][start:end].copy_(source)
        direct_cursors[name] = end

    def copy_offset(
        name: str,
        source: torch.Tensor,
        *,
        row_base: int,
        flat_base: int,
    ) -> None:
        rows = int(source.numel()) - 1
        output[name][row_base + 1 : row_base + rows + 1].copy_(
            source[1:] + int(flat_base)
        )

    for descriptor in descriptors:
        corpus = _load_fragment(descriptor)
        tensors = corpus.tensor_state()
        decisions = int(corpus.decisions)
        samples = int(corpus.total_samples)
        games = int(corpus.train_games + corpus.val_games)
        option_words = int(
            corpus.n_options.to(dtype=torch.int64).sum().item()
        )
        fragment_vocab = int(corpus.belief_card_vocab)
        fragment_strategic = (
            str(corpus.expanded_strategic_schema),
            int(corpus.expanded_strategic_schema_version),
            str(corpus.expanded_strategic_schema_digest),
        )
        if belief_card_vocab is None:
            belief_card_vocab = fragment_vocab
            strategic_identity = fragment_strategic
        elif (
            fragment_vocab != belief_card_vocab
            or fragment_strategic != strategic_identity
        ):
            raise RuntimeError(
                "parallel expert fragment scalar contracts disagree"
            )

        for name in _FLAT_DIRECT_FIELDS & tensors.keys():
            copy_direct(name, tensors[name])
        for name in _SAMPLE_DIRECT_FIELDS & tensors.keys():
            copy_direct(name, tensors[name])
        for name in _DECISION_DIRECT_FIELDS & tensors.keys():
            copy_direct(name, tensors[name])

        output["sample_board"][
            sample_base : sample_base + samples
        ].copy_(tensors["sample_board"] + decision_base)
        output["option_word_start"][
            sample_base : sample_base + samples
        ].copy_(tensors["option_word_start"] + option_word_base)

        copy_offset(
            "board_offset",
            tensors["board_offset"],
            row_base=decision_base * 24,
            flat_base=nnz_bases["board"],
        )
        copy_offset(
            "option_offset",
            tensors["option_offset"],
            row_base=option_word_base,
            flat_base=nnz_bases["option"],
        )
        copy_offset(
            "action_offset",
            tensors["action_offset"],
            row_base=decision_base,
            flat_base=nnz_bases["action"],
        )
        if "hand_offset" in tensors:
            copy_offset(
                "hand_offset",
                tensors["hand_offset"],
                row_base=decision_base,
                flat_base=nnz_bases["hand"],
            )
            copy_offset(
                "remainder_offset",
                tensors["remainder_offset"],
                row_base=decision_base,
                flat_base=nnz_bases["remainder"],
            )
        output["game_decision_offset"][
            game_base + 1 : game_base + games + 1
        ].copy_(tensors["game_decision_offset"][1:] + decision_base)
        output["game_sample_offset"][
            game_base + 1 : game_base + games + 1
        ].copy_(tensors["game_sample_offset"][1:] + sample_base)

        nnz_bases["board"] += int(tensors["board_index"].numel())
        nnz_bases["option"] += int(tensors["option_index"].numel())
        nnz_bases["action"] += int(tensors["action_index"].numel())
        if "hand_index" in tensors:
            nnz_bases["hand"] += int(tensors["hand_index"].numel())
            nnz_bases["remainder"] += int(
                tensors["remainder_index"].numel()
            )
        decision_base += decisions
        sample_base += samples
        game_base += games
        option_word_base += option_words
        train_games += int(corpus.train_games)
        val_games += int(corpus.val_games)
        train_samples += int(corpus.train_samples)
        val_samples += int(corpus.val_samples)
        del corpus, tensors
        gc.collect()

    for name, cursor in direct_cursors.items():
        if cursor != int(output[name].shape[0]):
            raise AssertionError(
                f"parallel expert merge did not fill {name}: "
                f"{cursor} != {output[name].shape[0]}"
            )
    if decision_base >= 2**31 or sample_base >= 2**31:
        raise MemoryError("parallel expert pack exceeds signed int32 row scale")
    if any(value >= 2**31 for value in nnz_bases.values()):
        raise MemoryError("parallel expert pack exceeds signed int32 CSR scale")
    tensor_bytes = sum(
        int(value.numel()) * int(value.element_size())
        for value in output.values()
    )
    assert strategic_identity is not None
    corpus = DeviceResidentBootstrapCorpus.from_packed_state(
        tensors=output,
        scalars={
            "train_samples": train_samples,
            "val_samples": val_samples,
            "train_games": train_games,
            "val_games": val_games,
            "decisions": decision_base,
            "input_bytes": tensor_bytes,
            "build_seconds": float(build_seconds),
            "belief_card_vocab": int(belief_card_vocab or 0),
            "expanded_strategic_schema": strategic_identity[0],
            "expanded_strategic_schema_version": strategic_identity[1],
            "expanded_strategic_schema_digest": strategic_identity[2],
        },
    )
    validate_cpu_corpus(corpus)
    return corpus


def _host_available_memory_bytes() -> Optional[int]:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def _cgroup_available_memory_bytes() -> Optional[int]:
    """Return the tightest finite cgroup-v2 memory allowance for this process."""

    proc = Path("/proc/self/cgroup")
    root = Path("/sys/fs/cgroup")
    if not proc.is_file() or not root.is_dir():
        return None
    relative: Optional[str] = None
    for line in proc.read_text(encoding="utf-8").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            relative = parts[2]
            break
    if relative is None:
        return None
    root = root.resolve()
    current = (root / relative.lstrip("/")).resolve()
    try:
        current.relative_to(root)
    except ValueError:
        return None
    allowances: list[int] = []
    while True:
        maximum_path = current / "memory.max"
        usage_path = current / "memory.current"
        try:
            maximum_raw = maximum_path.read_text(encoding="utf-8").strip()
            usage = int(usage_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pass
        else:
            if maximum_raw != "max":
                try:
                    maximum = int(maximum_raw)
                except ValueError:
                    pass
                else:
                    allowances.append(max(0, maximum - usage))
        if current == root:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return min(allowances) if allowances else None


def _available_memory_bytes() -> Optional[int]:
    candidates = [
        value
        for value in (
            _host_available_memory_bytes(),
            _cgroup_available_memory_bytes(),
        )
        if value is not None
    ]
    return min(candidates) if candidates else None


def _effective_workers(
    requested: int,
    views: tuple[FeatureManifestShardView, ...],
    *,
    memory_reserve_gib: float,
) -> int:
    workers = min(max(1, int(requested)), 4, len(views))
    available = _available_memory_bytes()
    if available is None:
        return workers
    largest = max(int(view.shard.path.stat().st_size) for view in views)
    estimated_worker_peak = max(_GIB, 2 * largest)
    usable = max(
        0,
        available - int(float(memory_reserve_gib) * _GIB),
    )
    if usable < estimated_worker_peak:
        raise MemoryError(
            "parallel expert worker cannot preserve its memory reserve: "
            f"available={available} reserve_gib={memory_reserve_gib} "
            f"estimated_worker_peak={estimated_worker_peak}"
        )
    memory_workers = usable // estimated_worker_peak
    return max(1, min(workers, int(memory_workers)))


def build_parallel_expert_cpu_pack(
    plan: Any,
    *,
    workers: int,
    exact_card_vocab: Optional[int],
    spool_root: Path,
    memory_reserve_gib: float = 12.0,
    disk_reserve_gib: float = 16.0,
) -> DeviceResidentBootstrapCorpus:
    """Build one semantically serial-equivalent CPU corpus with <=4 workers."""

    requested = int(workers)
    if requested <= 1:
        raise ValueError("parallel expert pack requires more than one worker")
    if float(memory_reserve_gib) < 0 or float(disk_reserve_gib) < 0:
        raise ValueError("parallel expert pack reserves must be nonnegative")
    views = tuple(plan.shard_views())
    if not views:
        raise ValueError("parallel expert pack has no feature shards")
    root = Path(spool_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_bytes = sum(int(view.shard.path.stat().st_size) for view in views)
    free_bytes = shutil.disk_usage(root).free
    required_free = 2 * source_bytes + int(float(disk_reserve_gib) * _GIB)
    if free_bytes < required_free:
        raise OSError(
            "parallel expert pack lacks safe spool space: "
            f"free={free_bytes} required={required_free}"
        )
    for stale in root.glob(".parallel-pack-*"):
        if stale.is_dir():
            shutil.rmtree(stale)
    spool = Path(tempfile.mkdtemp(prefix=".parallel-pack-", dir=root))
    started = time.monotonic()
    actual_workers = _effective_workers(
        requested,
        views,
        memory_reserve_gib=memory_reserve_gib,
    )
    tasks = [
        (
            ordinal,
            view,
            spool,
            (
                int(exact_card_vocab)
                if exact_card_vocab is not None
                else None
            ),
            bool(plan.has_expanded_strategic_targets),
        )
        for ordinal, view in enumerate(views)
    ]
    pool: Optional[ProcessPoolExecutor] = None
    futures: list[Future[ShardBuildResult]] = []
    try:
        pool = ProcessPoolExecutor(
            max_workers=actual_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        futures = [pool.submit(_build_shard, task) for task in tasks]
        results = [future.result() for future in futures]
        pool.shutdown(wait=True, cancel_futures=True)
        pool = None
        if [result.ordinal for result in results] != list(range(len(views))):
            raise RuntimeError("parallel expert workers returned wrong ordinals")
        if sum(result.source_games for result in results) != int(
            plan.sequences
        ):
            raise RuntimeError("parallel expert workers omitted source games")
        descriptors = [
            result.train for result in results if result.train is not None
        ] + [
            result.validation
            for result in results
            if result.validation is not None
        ]
        corpus = _merge_fragments(
            [value for value in descriptors if value is not None],
            build_seconds=time.monotonic() - started,
            memory_reserve_gib=float(memory_reserve_gib),
        )
        if int(corpus.decisions) != int(plan.packed_decisions):
            raise RuntimeError(
                "parallel expert decision count differs from split plan"
            )
        if (
            int(corpus.train_games) != int(plan.train_sequences)
            or int(corpus.val_games) != int(plan.val_sequences)
        ):
            raise RuntimeError(
                "parallel expert train/validation counts differ from split plan"
            )
        print(
            "[device-corpus] parallel expert pack "
            f"workers={actual_workers}/{requested} "
            f"shards={len(views)} elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
        return corpus
    finally:
        for future in futures:
            future.cancel()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        shutil.rmtree(spool, ignore_errors=True)
