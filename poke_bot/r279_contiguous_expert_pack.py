"""Revision-279 tensor-only expert pack for the Alakazam bootstrap.

The feature/sidecar materialization boundary may temporarily hold one source
day as Python objects.  The durable artifact and every epoch path are numeric
only: the ordinary resident corpus plus aligned OwnDeckLedger and tactical
target arrays.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .device_corpus import DeviceResidentBootstrapCorpus
from .dataset import GameSequence, PolicyStage
from .own_deck_ledger import OwnDeckLedgerSnapshot
from .own_deck_supervision import (
    TERMINAL_CONVERSION_OUTPUT_DIM,
    VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM,
    terminal_conversion_target_mask,
    terminal_conversion_target_vector,
    visible_tutor_completion_target_mask,
    visible_tutor_completion_target_vector,
)
from .tactical_sequence_supervision import (
    TACTICAL_SEQUENCE_OUTCOME_LABELS,
    TACTICAL_SEQUENCE_OUTCOME_TARGET_SCHEMA,
)


R279_PACK_SCHEMA = "poke_bot.r279_contiguous_expert_pack/v1"
R279_FRAGMENT_SCHEMA = "poke_bot.r279_contiguous_expert_fragment/v1"
LEDGER_CARD_STATS_WIDTH = 5
LEDGER_SCALAR_WIDTH = 10
LEDGER_OPTION_WIDTH = 8
TACTICAL_WIDTH = len(TACTICAL_SEQUENCE_OUTCOME_LABELS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _snapshot(value: object) -> OwnDeckLedgerSnapshot | None:
    if value is None:
        return None
    if isinstance(value, OwnDeckLedgerSnapshot):
        snapshot = value
    elif isinstance(value, Mapping):
        snapshot = OwnDeckLedgerSnapshot.from_dict(value)
    else:
        raise ValueError("own-deck ledger snapshot has an unsupported type")
    if not snapshot.integrity_ok or snapshot.fail_closed:
        return None
    return snapshot


def _supervision_for_stage(
    decision: object,
    *,
    family: str,
    stage_index: int,
    stage_count: int,
) -> Mapping[str, Any] | None:
    labels = getattr(decision, "own_deck_supervision", None)
    if not isinstance(labels, Mapping):
        return None
    mapping = getattr(decision, "own_deck_supervision_stage_indices", {})
    if isinstance(mapping, Mapping) and family in mapping:
        raw = mapping.get(family)
        if not isinstance(raw, (tuple, list)):
            raise ValueError("own-deck supervision stage mapping is malformed")
        indices = {int(value) for value in raw}
        if any(value < 0 or value >= stage_count for value in indices):
            raise ValueError("own-deck supervision stage mapping is out of bounds")
        return labels if stage_index in indices else None
    if family == "terminal_conversion" and stage_index == stage_count - 1:
        return labels
    return None


def _tactical_for_stage(
    decision: object,
    *,
    stage_index: int,
    stage_count: int,
    option_count: int,
) -> Mapping[str, Any] | None:
    value = getattr(decision, "tactical_sequence_supervision", None)
    if not isinstance(value, Mapping):
        return None
    aligned = int(
        getattr(decision, "tactical_sequence_supervision_stage_index", -1)
    )
    if aligned < 0:
        aligned = stage_count - 1
    if not 0 <= aligned < stage_count:
        raise ValueError("tactical supervision stage index is out of bounds")
    if stage_index != aligned:
        return None
    rows = value.get("rows")
    if (
        value.get("schema") != TACTICAL_SEQUENCE_OUTCOME_TARGET_SCHEMA
        or value.get("target_only") is not True
        or value.get("model_input") is not False
        or not isinstance(rows, list)
        or len(rows) != option_count
    ):
        raise ValueError("tactical supervision lost exact option alignment")
    return value


def build_side_tensors(
    sequences: Iterable[GameSequence],
    *,
    card_vocab: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Pack ledger inputs and typed option targets in core-packer row order."""

    decision_present: list[int] = []
    availability_ids: list[int] = []
    availability_stats: list[list[float]] = []
    availability_offsets = [0]
    scalar_rows: list[list[float]] = []
    select_ids: list[int] = []
    select_counts: list[float] = []
    select_offsets = [0]
    looking_ids: list[int] = []
    looking_counts: list[float] = []
    looking_offsets = [0]

    option_features: list[list[float]] = []
    option_present: list[int] = []
    tutor_target: list[list[float]] = []
    tutor_mask: list[list[int]] = []
    terminal_target: list[list[float]] = []
    terminal_mask: list[list[int]] = []
    tactical_target: list[list[float]] = []
    tactical_mask: list[list[int]] = []

    games = decisions = samples = options = 0
    for game in sequences:
        games += 1
        for decision in game.decisions:
            decisions += 1
            snapshot = _snapshot(getattr(decision, "ledger_snapshot", None))
            if snapshot is None:
                decision_present.append(0)
                scalar_rows.append([0.0] * LEDGER_SCALAR_WIDTH)
            else:
                decision_present.append(1)
                for row in snapshot.card_availability:
                    if not 0 <= int(row.card_id) < int(card_vocab):
                        raise ValueError("ledger card id is outside model vocabulary")
                    values = [
                        float(row.lower),
                        float(row.upper),
                        0.0 if row.expected is None else float(row.expected),
                        0.0
                        if row.probability_at_least_one is None
                        else float(row.probability_at_least_one),
                        1.0 if row.exact else 0.0,
                    ]
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError("ledger availability row is non-finite")
                    availability_ids.append(int(row.card_id))
                    availability_stats.append(values)
                scalars = [float(value) for value in snapshot.scalar_vector]
                if len(scalars) != LEDGER_SCALAR_WIDTH or not all(
                    math.isfinite(value) for value in scalars
                ):
                    raise ValueError("ledger scalar vector changed")
                scalar_rows.append(scalars)
                for card_id, count in snapshot.select_deck_counts:
                    select_ids.append(int(card_id))
                    select_counts.append(float(count))
                for card_id, count in snapshot.looking_counts:
                    looking_ids.append(int(card_id))
                    looking_counts.append(float(count))
            availability_offsets.append(len(availability_ids))
            select_offsets.append(len(select_ids))
            looking_offsets.append(len(looking_ids))

            stages = decision.policy_stages or [
                PolicyStage(
                    options=decision.options,
                    action_combos=decision.action_combos,
                    target_index=decision.action_combo_index,
                )
            ]
            for stage_index, stage in enumerate(stages):
                count = int(stage.options.num_words)
                target = int(stage.target_index)
                if count <= 0 or target < 0 or target >= count:
                    continue
                samples += 1
                features = getattr(stage, "ledger_option_features", None)
                if features is None:
                    if snapshot is not None:
                        raise ValueError("joined ledger row lacks option features")
                    parsed_features = [[0.0] * LEDGER_OPTION_WIDTH for _ in range(count)]
                    option_present.append(0)
                else:
                    parsed_features = [
                        [float(value) for value in row] for row in features
                    ]
                    if len(parsed_features) != count or any(
                        len(row) != LEDGER_OPTION_WIDTH
                        or not all(math.isfinite(value) for value in row)
                        for row in parsed_features
                    ):
                        raise ValueError("ledger option features changed shape")
                    option_present.append(1)
                option_features.extend(parsed_features)
                options += count

                tutor = _supervision_for_stage(
                    decision,
                    family="visible_tutor_completion",
                    stage_index=stage_index,
                    stage_count=len(stages),
                )
                terminal = _supervision_for_stage(
                    decision,
                    family="terminal_conversion",
                    stage_index=stage_index,
                    stage_count=len(stages),
                )
                tutor_target.append(list(visible_tutor_completion_target_vector(tutor)))
                tutor_mask.append(
                    [int(value) for value in visible_tutor_completion_target_mask(tutor)]
                )
                terminal_target.append(list(terminal_conversion_target_vector(terminal)))
                terminal_mask.append(
                    [int(value) for value in terminal_conversion_target_mask(terminal)]
                )

                tactical = _tactical_for_stage(
                    decision,
                    stage_index=stage_index,
                    stage_count=len(stages),
                    option_count=count,
                )
                if tactical is None:
                    tactical_target.extend([[0.0] * TACTICAL_WIDTH for _ in range(count)])
                    tactical_mask.extend([[0] * TACTICAL_WIDTH for _ in range(count)])
                else:
                    for row in tactical["rows"]:
                        values = [float(value) for value in row["values"]]
                        mask = [int(value) for value in row["mask"]]
                        if (
                            len(values) != TACTICAL_WIDTH
                            or len(mask) != TACTICAL_WIDTH
                            or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
                            or any(value not in (0, 1) for value in mask)
                        ):
                            raise ValueError("tactical target row changed shape")
                        tactical_target.append(values)
                        tactical_mask.append(mask)

    def tensor(values: Any, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype).contiguous()

    tensors = {
        "ledger_present": tensor(decision_present, torch.uint8),
        "ledger_availability_card_id": tensor(availability_ids, torch.int16),
        "ledger_availability_stats": tensor(availability_stats, torch.float32).reshape(-1, LEDGER_CARD_STATS_WIDTH),
        "ledger_availability_offset": tensor(availability_offsets, torch.int32),
        "ledger_scalar": tensor(scalar_rows, torch.float32).reshape(-1, LEDGER_SCALAR_WIDTH),
        "ledger_select_card_id": tensor(select_ids, torch.int16),
        "ledger_select_count": tensor(select_counts, torch.float32),
        "ledger_select_offset": tensor(select_offsets, torch.int32),
        "ledger_looking_card_id": tensor(looking_ids, torch.int16),
        "ledger_looking_count": tensor(looking_counts, torch.float32),
        "ledger_looking_offset": tensor(looking_offsets, torch.int32),
        "ledger_option_features": tensor(option_features, torch.float32).reshape(-1, LEDGER_OPTION_WIDTH),
        "ledger_option_present": tensor(option_present, torch.uint8),
        "visible_tutor_target": tensor(tutor_target, torch.float32).reshape(-1, VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM),
        "visible_tutor_mask": tensor(tutor_mask, torch.uint8).reshape(-1, VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM),
        "terminal_conversion_target": tensor(terminal_target, torch.float32).reshape(-1, TERMINAL_CONVERSION_OUTPUT_DIM),
        "terminal_conversion_mask": tensor(terminal_mask, torch.uint8).reshape(-1, TERMINAL_CONVERSION_OUTPUT_DIM),
        "tactical_sequence_target": tensor(tactical_target, torch.float32).reshape(-1, TACTICAL_WIDTH),
        "tactical_sequence_mask": tensor(tactical_mask, torch.uint8).reshape(-1, TACTICAL_WIDTH),
    }
    return tensors, {"games": games, "decisions": decisions, "samples": samples, "options": options}


_CSR_PREFIXES = ("board", "option", "action")
_DECISION_CSR_PREFIXES = ("hand", "remainder")


def _merge_offsets(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    merged = [torch.zeros(1, dtype=torch.int32)]
    base = 0
    for value in parts:
        local = value.to(dtype=torch.int64)
        merged.append((local[1:] + base).to(dtype=torch.int32))
        base += int(local[-1].item())
    return torch.cat(merged).contiguous()


def merge_core_corpora(
    fragments: Sequence[DeviceResidentBootstrapCorpus],
    *,
    train_fragment_count: int,
) -> DeviceResidentBootstrapCorpus:
    if not fragments or not 0 < train_fragment_count < len(fragments):
        raise ValueError("r279 merge requires nonempty source-disjoint train and validation fragments")
    states = [fragment.tensor_state() for fragment in fragments]
    names = set(states[0])
    if any(set(state) != names for state in states[1:]):
        raise ValueError("r279 core fragments have different tensor inventories")
    merged: dict[str, torch.Tensor] = {}
    offset_names = {f"{prefix}_offset" for prefix in _CSR_PREFIXES + _DECISION_CSR_PREFIXES}
    offset_names.update({"game_decision_offset", "game_sample_offset"})
    decision_base = option_word_base = 0
    sample_board_parts: list[torch.Tensor] = []
    option_start_parts: list[torch.Tensor] = []
    for fragment in fragments:
        sample_board_parts.append(fragment.sample_board + int(decision_base))
        option_start_parts.append(fragment.option_word_start + int(option_word_base))
        decision_base += int(fragment.decisions)
        option_word_base += int(fragment.n_options.to(dtype=torch.int64).sum().item())
    for name in sorted(names):
        if name in offset_names:
            merged[name] = _merge_offsets([state[name] for state in states])
        elif name == "sample_board":
            merged[name] = torch.cat(sample_board_parts).contiguous()
        elif name == "option_word_start":
            merged[name] = torch.cat(option_start_parts).contiguous()
        else:
            merged[name] = torch.cat([state[name] for state in states], dim=0).contiguous()
    train = fragments[:train_fragment_count]
    val = fragments[train_fragment_count:]
    scalar = fragments[0].scalar_state()
    scalar.update(
        train_samples=sum(int(value.total_samples) for value in train),
        val_samples=sum(int(value.total_samples) for value in val),
        train_games=sum(int(value.train_games + value.val_games) for value in train),
        val_games=sum(int(value.train_games + value.val_games) for value in val),
        decisions=sum(int(value.decisions) for value in fragments),
        build_seconds=sum(float(value.build_seconds) for value in fragments),
    )
    scalar["input_bytes"] = sum(
        int(value.numel()) * int(value.element_size()) for value in merged.values()
    )
    return DeviceResidentBootstrapCorpus.from_packed_state(tensors=merged, scalars=scalar)


def merge_side_tensors(parts: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not parts:
        raise ValueError("no r279 side fragments")
    names = set(parts[0])
    if any(set(part) != names for part in parts[1:]):
        raise ValueError("r279 side fragments have different tensor inventories")
    offset_names = {
        "ledger_availability_offset",
        "ledger_select_offset",
        "ledger_looking_offset",
    }
    return {
        name: (
            _merge_offsets([part[name] for part in parts])
            if name in offset_names
            else torch.cat([part[name] for part in parts], dim=0).contiguous()
        )
        for name in sorted(names)
    }


def validate_r279_pack(
    core: DeviceResidentBootstrapCorpus,
    side: Mapping[str, torch.Tensor],
    *,
    expected_games: int,
    expected_decisions: int,
) -> dict[str, int]:
    games = int(core.train_games + core.val_games)
    decisions = int(core.decisions)
    samples = int(core.total_samples)
    options = int(core.n_options.to(dtype=torch.int64).sum().item())
    if games != int(expected_games) or decisions != int(expected_decisions):
        raise ValueError(
            f"r279 exact corpus count mismatch games={games}/{expected_games} "
            f"decisions={decisions}/{expected_decisions}"
        )
    expected_shapes = {
        "ledger_present": (decisions,),
        "ledger_availability_offset": (decisions + 1,),
        "ledger_scalar": (decisions, LEDGER_SCALAR_WIDTH),
        "ledger_select_offset": (decisions + 1,),
        "ledger_looking_offset": (decisions + 1,),
        "ledger_option_features": (options, LEDGER_OPTION_WIDTH),
        "ledger_option_present": (samples,),
        "visible_tutor_target": (samples, VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM),
        "visible_tutor_mask": (samples, VISIBLE_TUTOR_COMPLETION_OUTPUT_DIM),
        "terminal_conversion_target": (samples, TERMINAL_CONVERSION_OUTPUT_DIM),
        "terminal_conversion_mask": (samples, TERMINAL_CONVERSION_OUTPUT_DIM),
        "tactical_sequence_target": (options, TACTICAL_WIDTH),
        "tactical_sequence_mask": (options, TACTICAL_WIDTH),
    }
    for name, shape in expected_shapes.items():
        value = side.get(name)
        if value is None or tuple(value.shape) != shape or not value.is_contiguous():
            raise ValueError(f"r279 side tensor shape mismatch {name}: {None if value is None else tuple(value.shape)} != {shape}")
    for name in ("ledger_availability_offset", "ledger_select_offset", "ledger_looking_offset"):
        offsets = side[name]
        if int(offsets[0]) != 0 or bool(torch.any(offsets[1:] < offsets[:-1])):
            raise ValueError(f"r279 side offsets are malformed: {name}")
    if bool(torch.any(core.target_index.to(torch.int32) >= core.n_options.to(torch.int32))):
        raise ValueError("r279 selected option is outside its legal row")
    for name in ("ledger_present", "ledger_option_present", "visible_tutor_mask", "terminal_conversion_mask", "tactical_sequence_mask"):
        value = side[name]
        if bool(torch.any((value != 0) & (value != 1))):
            raise ValueError(f"r279 mask is not binary: {name}")
    return {"games": games, "decisions": decisions, "samples": samples, "options": options}


def write_pack_atomic(
    path: Path,
    *,
    core: DeviceResidentBootstrapCorpus,
    side: Mapping[str, torch.Tensor],
    contract: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}.{time.time_ns()}")
    payload = {
        "schema": R279_PACK_SCHEMA,
        "contract": dict(contract),
        "metadata": dict(metadata),
        "core_scalars": core.scalar_state(),
        "core_tensors": core.tensor_state(),
        "side_tensors": dict(side),
    }
    try:
        with partial.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


@dataclass(frozen=True)
class FragmentPayload:
    core: DeviceResidentBootstrapCorpus
    side: dict[str, torch.Tensor]
    metadata: dict[str, Any]


def load_fragment(path: Path) -> FragmentPayload:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or payload.get("schema") != R279_FRAGMENT_SCHEMA:
        raise ValueError(f"invalid r279 fragment: {path}")
    core = DeviceResidentBootstrapCorpus.from_packed_state(
        tensors=dict(payload["core_tensors"]), scalars=dict(payload["core_scalars"])
    )
    return FragmentPayload(core=core, side=dict(payload["side_tensors"]), metadata=dict(payload["metadata"]))


def load_pack(path: Path) -> tuple[DeviceResidentBootstrapCorpus, dict[str, torch.Tensor], dict[str, Any]]:
    """Memory-map a sealed tensor pack without recreating source objects."""

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or payload.get("schema") != R279_PACK_SCHEMA:
        raise ValueError(f"invalid r279 contiguous pack: {path}")
    core = DeviceResidentBootstrapCorpus.from_packed_state(
        tensors=dict(payload["core_tensors"]), scalars=dict(payload["core_scalars"])
    )
    return core, dict(payload["side_tensors"]), {
        "contract": dict(payload["contract"]),
        "metadata": dict(payload["metadata"]),
    }


def _ranges(offsets: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    starts = offsets.index_select(0, ids).to(dtype=torch.long)
    ends = offsets.index_select(0, ids + 1).to(dtype=torch.long)
    return starts, ends


def _ragged_index_batch(
    index: torch.Tensor,
    offset: torch.Tensor,
    row_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts, ends = _ranges(offset, row_ids)
    lengths = ends - starts
    rows = torch.repeat_interleave(
        torch.arange(row_ids.numel(), device=row_ids.device), lengths
    )
    total = int(lengths.sum().item())
    if total == 0:
        values = index.new_empty((0,))
    else:
        local_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=row_ids.device),
                torch.cumsum(lengths, 0),
            )
        )
        relative = (
            torch.arange(total, device=row_ids.device)
            - local_offsets[:-1].index_select(0, rows)
        )
        values = index.index_select(0, starts.index_select(0, rows) + relative)
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=row_ids.device),
            torch.cumsum(lengths, 0).to(torch.int32),
        )
    ).contiguous()
    return values.contiguous(), offsets


def device_game_side_batch(
    core: DeviceResidentBootstrapCorpus,
    side: Mapping[str, torch.Tensor],
    game_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Gather a batch view from a corpus-wide side pack entirely on-device."""

    device = core.device
    if device.type != "cuda":
        raise ValueError("r280 device-side batch gather requires a CUDA corpus")
    ids = game_ids.reshape(-1).to(device=device, dtype=torch.long)
    assert core.game_decision_offset is not None and core.game_sample_offset is not None
    decision_starts, decision_ends = _ranges(core.game_decision_offset, ids)
    sample_starts, sample_ends = _ranges(core.game_sample_offset, ids)
    decision_ids, _ = core._expand_ranges(decision_starts, decision_ends)  # noqa: SLF001
    sample_ids, _ = core._expand_ranges(sample_starts, sample_ends)  # noqa: SLF001
    if not int(decision_ids.numel()) or not int(sample_ids.numel()):
        raise ValueError("r280 device-side batch contains no trainable rows")

    batch: dict[str, torch.Tensor] = {}
    for prefix in ("availability", "select", "looking"):
        source_prefix = f"ledger_{prefix}"
        index_name = f"{source_prefix}_card_id"
        offset_name = f"{source_prefix}_offset"
        values, offsets = _ragged_index_batch(
            side[index_name], side[offset_name], decision_ids
        )
        batch[index_name] = values
        batch[offset_name] = offsets

        starts, ends = _ranges(side[offset_name], decision_ids)
        lengths = ends - starts
        rows = torch.repeat_interleave(
            torch.arange(decision_ids.numel(), device=device), lengths
        )
        total = int(lengths.sum().item())
        if total:
            local_offsets = torch.cat(
                (
                    torch.zeros(1, dtype=torch.long, device=device),
                    torch.cumsum(lengths, 0),
                )
            )
            positions = (
                starts.index_select(0, rows)
                + torch.arange(total, device=device)
                - local_offsets[:-1].index_select(0, rows)
            )
        else:
            positions = torch.empty(0, dtype=torch.long, device=device)
        if prefix == "availability":
            batch["ledger_availability_stats"] = side[
                "ledger_availability_stats"
            ].index_select(0, positions).contiguous()
        else:
            count_name = f"{source_prefix}_count"
            batch[count_name] = side[count_name].index_select(
                0, positions
            ).contiguous()

    batch["ledger_present"] = side["ledger_present"].index_select(
        0, decision_ids
    ).contiguous()
    batch["ledger_scalar"] = side["ledger_scalar"].index_select(
        0, decision_ids
    ).contiguous()
    for name in (
        "ledger_option_present",
        "visible_tutor_target",
        "visible_tutor_mask",
        "terminal_conversion_target",
        "terminal_conversion_mask",
    ):
        batch[name] = side[name].index_select(0, sample_ids).contiguous()

    counts = core.n_options.index_select(0, sample_ids).to(dtype=torch.long)
    starts = core.option_word_start.index_select(0, sample_ids).to(dtype=torch.long)
    max_options = int(counts.max().item())
    columns = torch.arange(max_options, device=device)
    valid = columns.unsqueeze(0) < counts.unsqueeze(1)
    positions = starts.unsqueeze(1) + columns.unsqueeze(0)
    for name in (
        "ledger_option_features",
        "tactical_sequence_target",
        "tactical_sequence_mask",
    ):
        source = side[name]
        if int(source.size(0)) == 0:
            raise ValueError(f"r280 side tensor {name} is unexpectedly empty")
        gathered = source.index_select(
            0, positions.clamp_max(int(source.size(0)) - 1).reshape(-1)
        ).reshape(int(sample_ids.numel()), max_options, int(source.size(-1)))
        batch[name] = torch.where(
            valid.unsqueeze(-1), gathered, torch.zeros_like(gathered)
        ).contiguous()
    return batch


def _to_pinned_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = tensor.contiguous()
    if device.type == "cuda":
        value = value.pin_memory()
    return value.to(device=device, non_blocking=device.type == "cuda")


def pinned_game_batch(
    core: DeviceResidentBootstrapCorpus,
    side: Mapping[str, torch.Tensor],
    game_ids: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[DeviceResidentBootstrapCorpus, dict[str, torch.Tensor], torch.Tensor]:
    """Gather whole games into one pinned CPU batch and stream it to a device."""

    if core.device.type != "cpu":
        raise ValueError("r279 source pack must remain CPU-resident")
    ids = game_ids.reshape(-1).to(device="cpu", dtype=torch.long)
    assert core.game_decision_offset is not None and core.game_sample_offset is not None
    decision_starts, decision_ends = _ranges(core.game_decision_offset, ids)
    sample_starts, sample_ends = _ranges(core.game_sample_offset, ids)
    decision_ids, decision_lengths = core._expand_ranges(decision_starts, decision_ends)  # noqa: SLF001
    sample_ids, sample_lengths = core._expand_ranges(sample_starts, sample_ends)  # noqa: SLF001
    if not int(decision_ids.numel()) or not int(sample_ids.numel()):
        raise ValueError("r279 batch contains no trainable decisions")

    boards = core.board_ids_batch(decision_ids)
    options, counts, targets, values = core._options_batch(sample_ids)  # noqa: SLF001
    assert core.action_index is not None and core.action_value is not None and core.action_offset is not None
    action_starts, action_ends = _ranges(core.action_offset, decision_ids)
    actions = core._gather_segments(  # noqa: SLF001
        core.action_index, core.action_value, action_starts, action_ends - action_starts
    )
    decision_rows = torch.repeat_interleave(torch.arange(ids.numel()), decision_lengths)
    decision_local_offsets = torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(decision_lengths, 0)))
    sample_rows = torch.repeat_interleave(torch.arange(ids.numel()), sample_lengths)
    sample_board_global = core.sample_board.index_select(0, sample_ids).to(torch.long)
    sample_board_local = (
        decision_local_offsets[:-1].index_select(0, sample_rows)
        + sample_board_global
        - decision_starts.index_select(0, sample_rows)
    ).to(torch.int32)
    del decision_rows

    tensors: dict[str, torch.Tensor] = {
        "board_index": boards.index.to(torch.int32),
        "board_value": boards.value.to(torch.float32),
        "board_offset": boards.offset.to(torch.int32),
        "option_index": options.index.to(torch.int32),
        "option_value": options.value.to(torch.float32),
        "option_offset": options.offset.to(torch.int32),
        "sample_board": sample_board_local,
        "option_word_start": torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(counts.to(torch.long), 0)))[:-1].to(torch.int32),
        "n_options": counts.to(torch.int16),
        "target_index": targets.to(torch.int16),
        "value_target": values.to(torch.float32),
        "action_index": actions.index.to(torch.int32),
        "action_value": actions.value.to(torch.float32),
        "action_offset": actions.offset.to(torch.int32),
        "game_decision_offset": torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(decision_lengths, 0))).to(torch.int32),
        "game_sample_offset": torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(sample_lengths, 0))).to(torch.int32),
    }
    decision_fields = (
        "lethal_target", "prize_race_target",
        "strategic_tactical_outcome_target", "strategic_tactical_outcome_mask",
        "strategic_opponent_response_target", "strategic_opponent_response_mask",
        "strategic_resource_forecast_target", "strategic_resource_forecast_mask",
        "strategic_game_phase_target", "strategic_game_phase_mask",
        "strategic_outcome_class_target", "strategic_outcome_class_mask",
        "strategic_remaining_turns_target", "strategic_remaining_turns_mask",
    )
    sample_fields = (
        "sample_aux_class", "guide_target_index", "guide_confidence",
        "select_context", "selected_is_stop", "combo_top_deck_target",
        "combo_top_deck_mask", "combo_seek_source_target",
        "combo_seek_source_mask", "combo_vector_target", "combo_vector_mask",
        "strategic_action_q_target", "strategic_action_q_mask",
        "strategic_action_factor_mask", "strategic_action_utility_target",
        "strategic_action_utility_mask",
    )
    for name in decision_fields:
        value = getattr(core, name)
        if value is not None:
            tensors[name] = value.index_select(0, decision_ids).contiguous()
    for name in sample_fields:
        value = getattr(core, name)
        if value is not None:
            tensors[name] = value.index_select(0, sample_ids).contiguous()
    for prefix in ("hand", "remainder"):
        index = getattr(core, f"{prefix}_index")
        offset = getattr(core, f"{prefix}_offset")
        present = getattr(core, f"{prefix}_present")
        if index is not None and offset is not None and present is not None:
            tensors[f"{prefix}_index"], tensors[f"{prefix}_offset"] = _ragged_index_batch(index, offset, decision_ids)
            tensors[f"{prefix}_present"] = present.index_select(0, decision_ids).contiguous()

    scalars = core.scalar_state()
    scalars.update(
        train_samples=int(sample_ids.numel()), val_samples=0,
        train_games=int(ids.numel()), val_games=0,
        decisions=int(decision_ids.numel()), build_seconds=0.0,
    )
    scalars["input_bytes"] = sum(int(value.numel()) * int(value.element_size()) for value in tensors.values())
    device_tensors = {name: _to_pinned_device(value, device) for name, value in tensors.items()}
    batch_core = DeviceResidentBootstrapCorpus.from_packed_state(tensors=device_tensors, scalars=scalars)

    batch_side: dict[str, torch.Tensor] = {}
    for prefix in ("availability", "select", "looking"):
        source_prefix = f"ledger_{prefix}"
        source_index_name = f"{source_prefix}_card_id"
        source_offset_name = f"{source_prefix}_offset"
        values_out, offsets_out = _ragged_index_batch(side[source_index_name], side[source_offset_name], decision_ids)
        batch_side[source_index_name] = values_out
        batch_side[source_offset_name] = offsets_out
        if prefix == "availability":
            starts, ends = _ranges(side[source_offset_name], decision_ids)
            lengths = ends - starts
            rows = torch.repeat_interleave(torch.arange(decision_ids.numel()), lengths)
            local = torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(lengths, 0)))
            positions = starts.index_select(0, rows) + torch.arange(int(lengths.sum())) - local[:-1].index_select(0, rows)
            batch_side["ledger_availability_stats"] = side["ledger_availability_stats"].index_select(0, positions).contiguous()
        else:
            count_name = f"{source_prefix}_count"
            starts, ends = _ranges(side[source_offset_name], decision_ids)
            lengths = ends - starts
            rows = torch.repeat_interleave(torch.arange(decision_ids.numel()), lengths)
            local = torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(lengths, 0)))
            positions = starts.index_select(0, rows) + torch.arange(int(lengths.sum())) - local[:-1].index_select(0, rows)
            batch_side[count_name] = side[count_name].index_select(0, positions).contiguous()
    batch_side["ledger_present"] = side["ledger_present"].index_select(0, decision_ids).contiguous()
    batch_side["ledger_scalar"] = side["ledger_scalar"].index_select(0, decision_ids).contiguous()
    for name in (
        "ledger_option_present", "visible_tutor_target", "visible_tutor_mask",
        "terminal_conversion_target", "terminal_conversion_mask",
    ):
        batch_side[name] = side[name].index_select(0, sample_ids).contiguous()
    max_options = int(counts.max().item())
    option_features = torch.zeros((sample_ids.numel(), max_options, LEDGER_OPTION_WIDTH), dtype=torch.float32)
    tactical_target = torch.zeros((sample_ids.numel(), max_options, TACTICAL_WIDTH), dtype=torch.float32)
    tactical_mask = torch.zeros((sample_ids.numel(), max_options, TACTICAL_WIDTH), dtype=torch.uint8)
    for row, (sample_id, count) in enumerate(zip(sample_ids.tolist(), counts.tolist(), strict=True)):
        start = int(core.option_word_start[int(sample_id)])
        end = start + int(count)
        option_features[row, : int(count)] = side["ledger_option_features"][start:end]
        tactical_target[row, : int(count)] = side["tactical_sequence_target"][start:end]
        tactical_mask[row, : int(count)] = side["tactical_sequence_mask"][start:end]
    batch_side["ledger_option_features"] = option_features
    batch_side["tactical_sequence_target"] = tactical_target
    batch_side["tactical_sequence_mask"] = tactical_mask
    batch_side = {name: _to_pinned_device(value, device) for name, value in batch_side.items()}
    local_ids = torch.arange(int(ids.numel()), device=device, dtype=torch.long)
    return batch_core, batch_side, local_ids
