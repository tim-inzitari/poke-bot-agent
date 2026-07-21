"""Blackwell-resident sparse bootstrap corpus.

The ordinary training path keeps :class:`GameSequence` objects in host RAM and
rebuilds Python lists plus CUDA tensors for every batch.  Stateless bootstrap
training does not need temporal game grouping, so this module packs every
decision-stage into two CSR stores (board and legal options), copies those
stores to the training device once, and performs all later ragged gathers on
that device.
"""

from __future__ import annotations

import gc
import math
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from . import archetypes
from .dataset import BootstrapDataset, GameSequence, PolicyStage
from .features import SparseVector
from .model import PackedSparse


BOARD_WORDS = 24
DEFAULT_MIN_FREE_GIB = 12.0
# v2 keys the optional exact-card target layout. A v1 policy/value-only pack
# must never be reused after all-head expert rehearsal is enabled.
DEVICE_CORPUS_PACKING_SCHEMA_VERSION = 2

DEVICE_CORPUS_REQUIRED_TENSOR_FIELDS = (
    "board_index",
    "board_value",
    "board_offset",
    "option_index",
    "option_value",
    "option_offset",
    "sample_board",
    "option_word_start",
    "n_options",
    "target_index",
    "value_target",
)
DEVICE_CORPUS_OPTIONAL_TENSOR_FIELDS = (
    "hand_index",
    "hand_offset",
    "hand_present",
    "remainder_index",
    "remainder_offset",
    "remainder_present",
    "lethal_target",
    "prize_race_target",
    "sample_aux_class",
    "guide_target_index",
    "guide_confidence",
    "action_index",
    "action_value",
    "action_offset",
    "game_decision_offset",
    "game_sample_offset",
)
DEVICE_CORPUS_TENSOR_FIELDS = (
    *DEVICE_CORPUS_REQUIRED_TENSOR_FIELDS,
    *DEVICE_CORPUS_OPTIONAL_TENSOR_FIELDS,
)
DEVICE_CORPUS_SCALAR_FIELDS = (
    "train_samples",
    "val_samples",
    "train_games",
    "val_games",
    "decisions",
    "input_bytes",
    "build_seconds",
    "belief_card_vocab",
)


def _card_ids(value: Any) -> list[int] | None:
    """Flatten the exact-hidden card-id fields without importing train.py."""
    if value is None:
        return None
    result: list[int] = []
    if isinstance(value, list):
        for item in value:
            nested = _card_ids(item)
            if nested:
                result.extend(nested)
    elif isinstance(value, dict) and value.get("id") is not None:
        result.append(int(value["id"]))
    elif isinstance(value, int):
        result.append(int(value))
    return result


def _filtered_unique_card_ids(value: Any, card_vocab: int) -> list[int] | None:
    raw = _card_ids(value)
    if raw is None:
        return None
    return sorted({card_id for card_id in raw if 0 <= card_id < card_vocab})


class _CSRBuilder:
    """Chunked array builder that avoids per-nonzero Python integer copies."""

    def __init__(self, *, flush_vectors: int = 4096) -> None:
        if array("I").itemsize != 4 or array("f").itemsize != 4:
            raise RuntimeError("device corpus requires 32-bit I/f array storage")
        self.index = array("I")
        self.value = array("f")
        self.offset = array("I")
        self.words_total = 0
        self._pending: list[SparseVector] = []
        self._flush_vectors = max(1, int(flush_vectors))

    @property
    def vectors_total(self) -> int:
        return getattr(self, "_vectors_total", 0)

    def add(self, sv: SparseVector) -> None:
        if len(sv.index) != len(sv.value):
            raise ValueError("sparse index/value length mismatch")
        self._pending.append(sv)
        self.words_total += int(sv.num_words)
        self._vectors_total = self.vectors_total + 1
        if len(self._pending) >= self._flush_vectors:
            self.flush()

    def flush(self) -> None:
        pending = self._pending
        if not pending:
            return
        nnz = np.fromiter(
            (len(sv.index) for sv in pending), dtype=np.uint64, count=len(pending)
        )
        words = np.fromiter(
            (sv.num_words for sv in pending), dtype=np.uint64, count=len(pending)
        )
        base = np.empty(len(pending), dtype=np.uint64)
        base[0] = len(self.index)
        if len(pending) > 1:
            base[1:] = len(self.index) + np.cumsum(nnz[:-1], dtype=np.uint64)

        total_nnz = int(nnz.sum())
        if total_nnz:
            indices = np.concatenate(
                [np.asarray(sv.index, dtype=np.uint32) for sv in pending]
            )
            values = np.concatenate(
                [np.asarray(sv.value, dtype=np.float32) for sv in pending]
            )
            self.index.frombytes(indices.astype(np.uint32, copy=False).tobytes())
            self.value.frombytes(values.astype(np.float32, copy=False).tobytes())

        total_words = int(words.sum())
        if total_words:
            local_offsets = np.concatenate(
                [np.asarray(sv.offset, dtype=np.uint32) for sv in pending]
            ).astype(np.uint64, copy=False)
            global_offsets = local_offsets + np.repeat(base, words.astype(np.int64))
            if global_offsets.size and int(global_offsets.max()) >= 2**32:
                raise MemoryError("packed CSR offset exceeds uint32 capacity")
            self.offset.frombytes(global_offsets.astype(np.uint32).tobytes())

        self._pending = []

    def finish(self) -> None:
        self.flush()
        # The compact host builder uses unsigned 32-bit storage, but PyTorch's
        # device tensor is signed int32.  Fail before the sign bit could turn a
        # valid offset into a negative CUDA index.
        if len(self.index) >= 2**31:
            raise MemoryError("packed CSR nonzero count exceeds int32 capacity")
        self.offset.append(len(self.index))
        if len(self.offset) != self.words_total + 1:
            raise AssertionError(
                f"CSR offset mismatch: {len(self.offset)} != {self.words_total + 1}"
            )

    @property
    def nbytes(self) -> int:
        return (
            len(self.index) * self.index.itemsize
            + len(self.value) * self.value.itemsize
            + len(self.offset) * self.offset.itemsize
        )


def _to_tensor(values: array, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if not values:
        return torch.empty(0, dtype=dtype, device=device)
    return torch.frombuffer(values, dtype=dtype).to(device=device)


@dataclass
class DeviceResidentBootstrapCorpus:
    """Hard-target bootstrap features resident on one device.

    The core sample tensors remain usable by the stateless fast path.  Corpora
    built from in-memory game splits also retain compact game boundaries and
    realized-action CSR rows so a temporal learner can reconstruct exact
    causal histories without restoring the Python ``GameSequence`` corpus.
    """

    board_index: torch.Tensor
    board_value: torch.Tensor
    board_offset: torch.Tensor
    option_index: torch.Tensor
    option_value: torch.Tensor
    option_offset: torch.Tensor
    sample_board: torch.Tensor
    option_word_start: torch.Tensor
    n_options: torch.Tensor
    target_index: torch.Tensor
    value_target: torch.Tensor
    train_samples: int
    val_samples: int
    train_games: int
    val_games: int
    decisions: int
    input_bytes: int
    build_seconds: float
    # Optional exact-hidden targets.  Card labels use CSR over unique card ids
    # per board/decision; they are expanded only for the active GPU batch.
    belief_card_vocab: int = 0
    hand_index: torch.Tensor | None = None
    hand_offset: torch.Tensor | None = None
    hand_present: torch.Tensor | None = None
    remainder_index: torch.Tensor | None = None
    remainder_offset: torch.Tensor | None = None
    remainder_present: torch.Tensor | None = None
    lethal_target: torch.Tensor | None = None
    prize_race_target: torch.Tensor | None = None
    sample_aux_class: torch.Tensor | None = None
    guide_target_index: torch.Tensor | None = None
    guide_confidence: torch.Tensor | None = None
    action_index: torch.Tensor | None = None
    action_value: torch.Tensor | None = None
    action_offset: torch.Tensor | None = None
    game_decision_offset: torch.Tensor | None = None
    game_sample_offset: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.board_index.device

    @property
    def total_samples(self) -> int:
        return self.train_samples + self.val_samples

    @property
    def has_exact_targets(self) -> bool:
        return bool(
            self.belief_card_vocab > 0
            and self.hand_index is not None
            and self.hand_offset is not None
            and self.hand_present is not None
            and self.remainder_index is not None
            and self.remainder_offset is not None
            and self.remainder_present is not None
            and self.lethal_target is not None
            and self.prize_race_target is not None
            and self.sample_aux_class is not None
        )

    @property
    def has_guide_targets(self) -> bool:
        return bool(
            self.guide_target_index is not None
            and self.guide_confidence is not None
        )

    @property
    def has_temporal_layout(self) -> bool:
        return bool(
            self.action_index is not None
            and self.action_value is not None
            and self.action_offset is not None
            and self.game_decision_offset is not None
            and self.game_sample_offset is not None
            and self.game_decision_offset.numel()
            == self.train_games + self.val_games + 1
            and self.game_sample_offset.numel()
            == self.train_games + self.val_games + 1
        )

    def tensor_state(self) -> dict[str, torch.Tensor]:
        """Return the exact packed tensors, excluding absent optional targets."""
        state: dict[str, torch.Tensor] = {}
        for name in DEVICE_CORPUS_TENSOR_FIELDS:
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"device corpus field {name} is not a tensor")
                state[name] = value
        return state

    def scalar_state(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in DEVICE_CORPUS_SCALAR_FIELDS}

    @classmethod
    def from_packed_state(
        cls,
        *,
        tensors: dict[str, torch.Tensor],
        scalars: dict[str, Any],
    ) -> "DeviceResidentBootstrapCorpus":
        missing = set(DEVICE_CORPUS_REQUIRED_TENSOR_FIELDS) - set(tensors)
        if missing:
            raise ValueError(f"packed device corpus is missing tensors: {sorted(missing)}")
        missing_scalars = set(DEVICE_CORPUS_SCALAR_FIELDS) - set(scalars)
        if missing_scalars:
            raise ValueError(
                f"packed device corpus is missing scalars: {sorted(missing_scalars)}"
            )
        unknown_tensors = set(tensors) - set(DEVICE_CORPUS_TENSOR_FIELDS)
        if unknown_tensors:
            raise ValueError(
                f"packed device corpus has unknown tensors: {sorted(unknown_tensors)}"
            )
        values: dict[str, Any] = {
            name: tensors.get(name) for name in DEVICE_CORPUS_TENSOR_FIELDS
        }
        values.update({name: scalars[name] for name in DEVICE_CORPUS_SCALAR_FIELDS})
        return cls(**values)

    @property
    def tensor_bytes(self) -> int:
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in self.tensor_state().values()
        )

    def to_device(
        self,
        device: torch.device,
        *,
        min_free_gib: float = DEFAULT_MIN_FREE_GIB,
    ) -> "DeviceResidentBootstrapCorpus":
        """Copy one validated CPU pack to its training device exactly once."""
        target = torch.device(device)
        if self.device == target:
            return self
        if self.device.type != "cpu":
            raise ValueError(
                f"durable corpus transfer must originate on CPU, got {self.device}"
            )
        required_bytes = self.tensor_bytes
        if target.type == "cuda":
            free_device, total_device = torch.cuda.mem_get_info(target)
            max_inputs = max(
                1, free_device - int(float(min_free_gib) * 2**30)
            )
            if required_bytes > max_inputs:
                raise MemoryError(
                    f"device corpus needs {required_bytes / 2**30:.2f} GiB, "
                    f"leaving less than {float(min_free_gib):.1f} GiB from "
                    f"current free VRAM {free_device / 2**30:.2f}/"
                    f"{total_device / 2**30:.2f} GiB"
                )
        moved: dict[str, torch.Tensor] = {}
        started = time.monotonic()
        try:
            for name, tensor in self.tensor_state().items():
                moved[name] = tensor.to(device=target)
            corpus = type(self).from_packed_state(
                tensors=moved,
                scalars=self.scalar_state(),
            )
        except BaseException:
            moved.clear()
            if target.type == "cuda":
                torch.cuda.empty_cache()
            raise
        if target.type == "cuda":
            free_device, total_device = torch.cuda.mem_get_info(target)
            if free_device < int(float(min_free_gib) * 2**30):
                del corpus
                moved.clear()
                torch.cuda.empty_cache()
                raise MemoryError(
                    f"post-transfer CUDA free memory {free_device / 2**30:.2f} GiB "
                    f"< required {float(min_free_gib):.1f} GiB"
                )
            print(
                f"[device-corpus] cached H2D={required_bytes / 2**30:.2f} GiB "
                f"elapsed={time.monotonic() - started:.1f}s "
                f"CUDA-free={free_device / 2**30:.2f}/"
                f"{total_device / 2**30:.2f} GiB",
                flush=True,
            )
        return corpus

    @classmethod
    def from_splits(
        cls,
        train: Sequence[GameSequence],
        val: Sequence[GameSequence],
        *,
        device: torch.device,
        min_free_gib: float = DEFAULT_MIN_FREE_GIB,
        exact_card_vocab: int | None = None,
    ) -> "DeviceResidentBootstrapCorpus":
        if exact_card_vocab is not None and (
            int(exact_card_vocab) <= 0 or int(exact_card_vocab) >= 2**15
        ):
            raise ValueError(f"unsupported belief card vocab: {exact_card_vocab}")
        started = time.monotonic()
        boards = _CSRBuilder()
        options = _CSRBuilder()
        actions = _CSRBuilder()
        sample_board = array("I")
        game_decision_offset = array("I", [0])
        game_sample_offset = array("I", [0])
        option_word_start = array("I")
        n_options = array("H")
        target_index = array("H")
        value_target = array("f")
        guide_target_index = array("h")
        guide_confidence = array("f")
        sample_aux_class = array("h")
        hand_index = array("h")
        hand_offset = array("I", [0])
        hand_present = array("B")
        remainder_index = array("h")
        remainder_offset = array("I", [0])
        remainder_present = array("B")
        lethal_target = array("f")
        prize_race_target = array("f")
        archetype_ids = list(archetypes.archetype_ids())
        decisions = 0

        def add_sequences(sequences: Iterable[GameSequence], progress) -> None:
            nonlocal decisions
            for game in sequences:
                game_sample_start = len(sample_board)
                if game.policy_targets is not None or game.factorized_policy_targets is not None:
                    raise ValueError(
                        "device-resident bootstrap currently requires hard targets only"
                    )
                for decision in game.decisions:
                    raw_stages = decision.policy_stages or [
                        PolicyStage(
                            options=decision.options,
                            action_combos=decision.action_combos,
                            target_index=decision.action_combo_index,
                        )
                    ]
                    valid = []
                    for stage in raw_stages:
                        count = int(stage.options.num_words)
                        target = int(stage.target_index)
                        if count > 0 and 0 <= target < count:
                            valid.append((stage, count, target))
                    if decision.board.num_words != BOARD_WORDS:
                        raise ValueError(
                            f"expected {BOARD_WORDS} board words, got "
                            f"{decision.board.num_words}"
                        )
                    board_id = boards.vectors_total
                    boards.add(decision.board)
                    action = decision.action_token
                    if action is None:
                        action = SparseVector()
                        action.word_start()
                    if action.num_words != 1:
                        raise ValueError(
                            "resident temporal action token must contain one word"
                        )
                    actions.add(action)
                    decisions += 1
                    if exact_card_vocab is not None:
                        aux = dict(decision.aux_labels or {})
                        hand_raw = aux.get("opp_hand")
                        deck_raw = aux.get("opp_deck_order")
                        prizes_raw = aux.get("opp_prizes")
                        exact_remainder_raw = aux.get("opp_hidden_remainder")
                        hand_ids = _filtered_unique_card_ids(
                            hand_raw, int(exact_card_vocab)
                        )
                        hand_present.append(1 if hand_raw is not None else 0)
                        if hand_ids:
                            hand_index.extend(hand_ids)
                        hand_offset.append(len(hand_index))

                        rem_present = exact_remainder_raw is not None or any(
                            value is not None
                            for value in (hand_raw, deck_raw, prizes_raw)
                        )
                        rem_raw: list[int] = []
                        remainder_sources = (
                            (exact_remainder_raw,)
                            if exact_remainder_raw is not None
                            else (hand_raw, deck_raw, prizes_raw)
                        )
                        for value in remainder_sources:
                            ids = _card_ids(value)
                            if ids:
                                rem_raw.extend(ids)
                        rem_ids = _filtered_unique_card_ids(
                            rem_raw, int(exact_card_vocab)
                        )
                        remainder_present.append(1 if rem_present else 0)
                        if rem_ids:
                            remainder_index.extend(rem_ids)
                        remainder_offset.append(len(remainder_index))

                        lethal = aux.get("lethal_threat")
                        lethal_target.append(
                            float(lethal) if lethal is not None else math.nan
                        )
                        race = aux.get("prize_race")
                        if isinstance(race, (list, tuple)) and len(race) >= 2:
                            prize_race_target.extend(
                                (float(race[0]), float(race[1]))
                            )
                        else:
                            prize_race_target.extend((math.nan, math.nan))
                    for stage, count, target in valid:
                        if count >= 2**16 or target >= 2**16:
                            raise ValueError("option count/target exceeds uint16 capacity")
                        sample_board.append(board_id)
                        option_word_start.append(options.words_total)
                        n_options.append(count)
                        target_index.append(target)
                        value_target.append(float(game.value))
                        guide_target = int(
                            getattr(stage, "guide_target_index", -1)
                        )
                        if guide_target >= count:
                            raise ValueError(
                                "guide target exceeds resident option count: "
                                f"target={guide_target} options={count}"
                            )
                        guide_target_index.append(guide_target)
                        guide_confidence.append(
                            float(getattr(stage, "guide_confidence", 0.0))
                        )
                        if exact_card_vocab is not None:
                            sample_aux_class.append(-1)
                        options.add(stage.options)
                if (
                    exact_card_vocab is not None
                    and len(sample_board) > game_sample_start
                    and game.opp_archetype in archetype_ids
                ):
                    label = archetype_ids.index(game.opp_archetype)
                    for sample_id in range(game_sample_start, len(sample_board)):
                        sample_aux_class[sample_id] = label
                game_decision_offset.append(decisions)
                game_sample_offset.append(len(sample_board))
                progress.update(1)

        with tqdm(
            total=len(train) + len(val),
            desc="pack Blackwell corpus",
            unit="game",
        ) as progress:
            add_sequences(train, progress)
            train_samples = len(sample_board)
            add_sequences(val, progress)
        val_samples = len(sample_board) - train_samples
        boards.finish()
        options.finish()
        actions.finish()
        if actions.vectors_total != decisions or actions.words_total != decisions:
            raise AssertionError("one resident action row is required per decision")
        if len(game_decision_offset) != len(train) + len(val) + 1:
            raise AssertionError("resident game decision offsets are incomplete")
        if len(game_sample_offset) != len(train) + len(val) + 1:
            raise AssertionError("resident game sample offsets are incomplete")
        if options.vectors_total != len(sample_board):
            raise AssertionError("one option CSR row is required per training sample")
        if options.words_total != sum(n_options):
            raise AssertionError("option word-prefix accounting mismatch")
        if len(guide_target_index) != len(sample_board):
            raise AssertionError("resident guide target shape mismatch")
        if exact_card_vocab is not None:
            if len(hand_offset) != decisions + 1 or len(remainder_offset) != decisions + 1:
                raise AssertionError("exact card-target CSR offset mismatch")
            if len(hand_present) != decisions or len(remainder_present) != decisions:
                raise AssertionError("exact card-target presence mismatch")
            if len(lethal_target) != decisions or len(prize_race_target) != decisions * 2:
                raise AssertionError("exact strategy-target shape mismatch")
            if len(sample_aux_class) != len(sample_board):
                raise AssertionError("exact archetype target shape mismatch")

        exact_arrays = (
            hand_index,
            hand_offset,
            hand_present,
            remainder_index,
            remainder_offset,
            remainder_present,
            lethal_target,
            prize_race_target,
            sample_aux_class,
        )

        cpu_bytes = (
            boards.nbytes
            + options.nbytes
            + actions.nbytes
            + len(sample_board) * sample_board.itemsize
            + len(game_decision_offset) * game_decision_offset.itemsize
            + len(game_sample_offset) * game_sample_offset.itemsize
            + len(option_word_start) * option_word_start.itemsize
            + len(n_options) * n_options.itemsize
            + len(target_index) * target_index.itemsize
            + len(value_target) * value_target.itemsize
            + len(guide_target_index) * guide_target_index.itemsize
            + len(guide_confidence) * guide_confidence.itemsize
            + (
                sum(len(values) * values.itemsize for values in exact_arrays)
                if exact_card_vocab is not None
                else 0
            )
        )
        if cpu_bytes >= 2**31 * 8:
            # This is only a coarse secondary guard; each CSR builder checks its
            # own element/offset range precisely in ``finish``.
            raise MemoryError("device corpus exceeds supported int32 CSR scale")
        if device.type == "cuda":
            free_device, total_device = torch.cuda.mem_get_info(device)
            # Fail before any large H2D transfer; preserve ample activation room.
            max_inputs = max(
                1, free_device - int(float(min_free_gib) * 2**30)
            )
            if cpu_bytes > max_inputs:
                raise MemoryError(
                    f"device corpus needs {cpu_bytes / 2**30:.2f} GiB, leaving "
                    f"less than {float(min_free_gib):.1f} GiB activation headroom "
                    f"from current free VRAM {free_device / 2**30:.2f}/"
                    f"{total_device / 2**30:.2f} GiB"
                )

        print(
            f"[device-corpus] CPU pack={cpu_bytes / 2**30:.2f} GiB "
            f"decisions={decisions} samples={len(sample_board)} "
            f"train={train_samples} val={val_samples}",
            flush=True,
        )
        corpus = cls(
            board_index=_to_tensor(boards.index, torch.int32, device),
            board_value=_to_tensor(boards.value, torch.float32, device),
            board_offset=_to_tensor(boards.offset, torch.int32, device),
            option_index=_to_tensor(options.index, torch.int32, device),
            option_value=_to_tensor(options.value, torch.float32, device),
            option_offset=_to_tensor(options.offset, torch.int32, device),
            action_index=_to_tensor(actions.index, torch.int32, device),
            action_value=_to_tensor(actions.value, torch.float32, device),
            action_offset=_to_tensor(actions.offset, torch.int32, device),
            game_decision_offset=_to_tensor(
                game_decision_offset, torch.int32, device
            ),
            game_sample_offset=_to_tensor(
                game_sample_offset, torch.int32, device
            ),
            sample_board=_to_tensor(sample_board, torch.int32, device),
            option_word_start=_to_tensor(option_word_start, torch.int32, device),
            n_options=_to_tensor(n_options, torch.int16, device),
            target_index=_to_tensor(target_index, torch.int16, device),
            value_target=_to_tensor(value_target, torch.float32, device),
            train_samples=train_samples,
            val_samples=val_samples,
            train_games=len(train),
            val_games=len(val),
            decisions=decisions,
            input_bytes=cpu_bytes,
            build_seconds=time.monotonic() - started,
            belief_card_vocab=int(exact_card_vocab or 0),
            hand_index=(
                _to_tensor(hand_index, torch.int16, device)
                if exact_card_vocab is not None
                else None
            ),
            hand_offset=(
                _to_tensor(hand_offset, torch.int32, device)
                if exact_card_vocab is not None
                else None
            ),
            hand_present=(
                _to_tensor(hand_present, torch.uint8, device)
                if exact_card_vocab is not None
                else None
            ),
            remainder_index=(
                _to_tensor(remainder_index, torch.int16, device)
                if exact_card_vocab is not None
                else None
            ),
            remainder_offset=(
                _to_tensor(remainder_offset, torch.int32, device)
                if exact_card_vocab is not None
                else None
            ),
            remainder_present=(
                _to_tensor(remainder_present, torch.uint8, device)
                if exact_card_vocab is not None
                else None
            ),
            lethal_target=(
                _to_tensor(lethal_target, torch.float32, device)
                if exact_card_vocab is not None
                else None
            ),
            prize_race_target=(
                _to_tensor(prize_race_target, torch.float32, device).reshape(-1, 2)
                if exact_card_vocab is not None
                else None
            ),
            sample_aux_class=(
                _to_tensor(sample_aux_class, torch.int16, device)
                if exact_card_vocab is not None
                else None
            ),
            guide_target_index=_to_tensor(
                guide_target_index, torch.int16, device
            ),
            guide_confidence=_to_tensor(
                guide_confidence, torch.float32, device
            ),
        )
        del boards, options, actions, sample_board, option_word_start, n_options
        del game_decision_offset, game_sample_offset
        del target_index, value_target
        del guide_target_index, guide_confidence
        del hand_index, hand_offset, hand_present
        del remainder_index, remainder_offset, remainder_present
        del lethal_target, prize_race_target, sample_aux_class
        gc.collect()
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            if free < int(float(min_free_gib) * 2**30):
                raise MemoryError(
                    f"post-pack CUDA free memory {free / 2**30:.2f} GiB < "
                    f"required {float(min_free_gib):.1f} GiB"
                )
            print(
                f"[device-corpus] resident={corpus.input_bytes / 2**30:.2f} GiB "
                f"CUDA-free={free / 2**30:.2f}/{total / 2**30:.2f} GiB "
                f"build={corpus.build_seconds:.1f}s",
                flush=True,
            )
        return corpus

    @classmethod
    def from_exact_shards(
        cls,
        train_shards: Sequence[Path],
        val_shards: Sequence[Path],
        *,
        cache_dir: Path,
        max_context: int,
        card_vocab: int,
        device: torch.device,
        min_free_gib: float = DEFAULT_MIN_FREE_GIB,
    ) -> "DeviceResidentBootstrapCorpus":
        """Pack a sharded exact-hidden corpus without retaining host objects.

        Boards, options, outcomes, selected actions, and every auxiliary target
        are copied once.  Hand/remainder labels remain sparse on-device and are
        expanded to multi-hot tensors only for the active optimizer batch.
        """
        if card_vocab <= 0 or card_vocab >= 2**15:
            raise ValueError(f"unsupported belief card vocab: {card_vocab}")
        if not train_shards or not val_shards:
            raise ValueError("exact resident corpus requires train and val shards")
        started = time.monotonic()
        boards = _CSRBuilder()
        options = _CSRBuilder()
        sample_board = array("I")
        option_word_start = array("I")
        n_options = array("H")
        target_index = array("H")
        value_target = array("f")
        guide_target_index = array("h")
        guide_confidence = array("f")
        sample_aux_class = array("h")
        hand_index = array("h")
        hand_offset = array("I", [0])
        hand_present = array("B")
        remainder_index = array("h")
        remainder_offset = array("I", [0])
        remainder_present = array("B")
        lethal_target = array("f")
        prize_race_target = array("f")
        archetype_ids = list(archetypes.archetype_ids())
        decisions = 0
        train_games = 0
        val_games = 0

        def add_game(game: GameSequence) -> None:
            nonlocal decisions
            game_sample_start = len(sample_board)
            for decision in game.decisions:
                raw_stages = decision.policy_stages or [
                    PolicyStage(
                        options=decision.options,
                        action_combos=decision.action_combos,
                        target_index=decision.action_combo_index,
                    )
                ]
                valid: list[tuple[PolicyStage, int, int]] = []
                for stage in raw_stages:
                    count = int(stage.options.num_words)
                    target = int(stage.target_index)
                    if count > 0 and 0 <= target < count:
                        valid.append((stage, count, target))
                if not valid:
                    continue
                if decision.board.num_words != BOARD_WORDS:
                    raise ValueError(
                        f"expected {BOARD_WORDS} board words, got "
                        f"{decision.board.num_words}"
                    )

                board_id = boards.vectors_total
                boards.add(decision.board)
                decisions += 1
                aux = dict(decision.aux_labels or {})
                hand_raw = aux.get("opp_hand")
                deck_raw = aux.get("opp_deck_order")
                prizes_raw = aux.get("opp_prizes")
                exact_remainder_raw = aux.get("opp_hidden_remainder")
                hand_ids = _filtered_unique_card_ids(hand_raw, card_vocab)
                hand_present.append(1 if hand_raw is not None else 0)
                if hand_ids:
                    hand_index.extend(hand_ids)
                hand_offset.append(len(hand_index))

                rem_present = exact_remainder_raw is not None or any(
                    value is not None for value in (hand_raw, deck_raw, prizes_raw)
                )
                rem_raw: list[int] = []
                remainder_sources = (
                    (exact_remainder_raw,)
                    if exact_remainder_raw is not None
                    else (hand_raw, deck_raw, prizes_raw)
                )
                for value in remainder_sources:
                    ids = _card_ids(value)
                    if ids:
                        rem_raw.extend(ids)
                rem_ids = _filtered_unique_card_ids(rem_raw, card_vocab)
                remainder_present.append(1 if rem_present else 0)
                if rem_ids:
                    remainder_index.extend(rem_ids)
                remainder_offset.append(len(remainder_index))

                lethal = aux.get("lethal_threat")
                lethal_target.append(float(lethal) if lethal is not None else math.nan)
                race = aux.get("prize_race")
                if isinstance(race, (list, tuple)) and len(race) >= 2:
                    prize_race_target.extend((float(race[0]), float(race[1])))
                else:
                    prize_race_target.extend((math.nan, math.nan))

                for stage, count, target in valid:
                    if count >= 2**16 or target >= 2**16:
                        raise ValueError("option count/target exceeds uint16 capacity")
                    sample_board.append(board_id)
                    option_word_start.append(options.words_total)
                    n_options.append(count)
                    target_index.append(target)
                    value_target.append(float(game.value))
                    guide_target = int(
                        getattr(stage, "guide_target_index", -1)
                    )
                    if guide_target >= count:
                        raise ValueError(
                            "guide target exceeds resident option count: "
                            f"target={guide_target} options={count}"
                        )
                    guide_target_index.append(guide_target)
                    guide_confidence.append(
                        float(getattr(stage, "guide_confidence", 0.0))
                    )
                    sample_aux_class.append(-1)
                    options.add(stage.options)

            if len(sample_board) > game_sample_start and game.opp_archetype in archetype_ids:
                label = archetype_ids.index(game.opp_archetype)
                for sample_id in range(game_sample_start, len(sample_board)):
                    sample_aux_class[sample_id] = label

        def add_shards(shards: Sequence[Path], progress: tqdm) -> int:
            games = 0
            for shard in shards:
                dataset = BootstrapDataset.from_jsonl(
                    shard,
                    max_context=int(max_context),
                    verify_info_set=True,
                    use_cache=True,
                    cache_dir=Path(cache_dir),
                )
                if not dataset.info_set_ok_all or dataset.n_decisions <= 0:
                    raise RuntimeError(f"invalid exact replay shard: {shard}")
                for game in dataset.sequences:
                    add_game(game)
                    games += 1
                progress.update(1)
                progress.set_postfix(
                    decisions=decisions,
                    samples=len(sample_board),
                )
                del dataset
                gc.collect()
            return games

        with tqdm(
            total=len(train_shards) + len(val_shards),
            desc="pack exact Blackwell corpus",
            unit="shard",
        ) as progress:
            train_games = add_shards(train_shards, progress)
            train_samples = len(sample_board)
            val_games = add_shards(val_shards, progress)
        val_samples = len(sample_board) - train_samples
        boards.finish()
        options.finish()
        if options.vectors_total != len(sample_board):
            raise AssertionError("one option CSR row is required per exact sample")
        if len(hand_offset) != decisions + 1 or len(remainder_offset) != decisions + 1:
            raise AssertionError("exact card-target CSR offset mismatch")
        if len(hand_present) != decisions or len(remainder_present) != decisions:
            raise AssertionError("exact card-target presence mismatch")
        if len(lethal_target) != decisions or len(prize_race_target) != decisions * 2:
            raise AssertionError("exact strategy-target shape mismatch")
        if len(sample_aux_class) != len(sample_board):
            raise AssertionError("exact archetype target shape mismatch")
        if not (
            len(guide_target_index)
            == len(guide_confidence)
            == len(sample_board)
        ):
            raise AssertionError("exact guide target shape mismatch")

        exact_arrays = (
            hand_index,
            hand_offset,
            hand_present,
            remainder_index,
            remainder_offset,
            remainder_present,
            lethal_target,
            prize_race_target,
            sample_aux_class,
        )
        cpu_bytes = (
            boards.nbytes
            + options.nbytes
            + len(sample_board) * sample_board.itemsize
            + len(option_word_start) * option_word_start.itemsize
            + len(n_options) * n_options.itemsize
            + len(target_index) * target_index.itemsize
            + len(value_target) * value_target.itemsize
            + len(guide_target_index) * guide_target_index.itemsize
            + len(guide_confidence) * guide_confidence.itemsize
            + sum(len(values) * values.itemsize for values in exact_arrays)
        )
        if device.type == "cuda":
            free_device, total_device = torch.cuda.mem_get_info(device)
            max_inputs = max(1, free_device - int(float(min_free_gib) * 2**30))
            if cpu_bytes > max_inputs:
                raise MemoryError(
                    f"exact device corpus needs {cpu_bytes / 2**30:.2f} GiB, "
                    f"leaving less than {float(min_free_gib):.1f} GiB from "
                    f"current free VRAM {free_device / 2**30:.2f}/"
                    f"{total_device / 2**30:.2f} GiB"
                )

        print(
            f"[device-corpus-exact] CPU pack={cpu_bytes / 2**30:.2f} GiB "
            f"decisions={decisions} samples={len(sample_board)} "
            f"train={train_samples} val={val_samples}",
            flush=True,
        )
        corpus = cls(
            board_index=_to_tensor(boards.index, torch.int32, device),
            board_value=_to_tensor(boards.value, torch.float32, device),
            board_offset=_to_tensor(boards.offset, torch.int32, device),
            option_index=_to_tensor(options.index, torch.int32, device),
            option_value=_to_tensor(options.value, torch.float32, device),
            option_offset=_to_tensor(options.offset, torch.int32, device),
            sample_board=_to_tensor(sample_board, torch.int32, device),
            option_word_start=_to_tensor(option_word_start, torch.int32, device),
            n_options=_to_tensor(n_options, torch.int16, device),
            target_index=_to_tensor(target_index, torch.int16, device),
            value_target=_to_tensor(value_target, torch.float32, device),
            train_samples=train_samples,
            val_samples=val_samples,
            train_games=train_games,
            val_games=val_games,
            decisions=decisions,
            input_bytes=cpu_bytes,
            build_seconds=time.monotonic() - started,
            belief_card_vocab=int(card_vocab),
            hand_index=_to_tensor(hand_index, torch.int16, device),
            hand_offset=_to_tensor(hand_offset, torch.int32, device),
            hand_present=_to_tensor(hand_present, torch.uint8, device),
            remainder_index=_to_tensor(remainder_index, torch.int16, device),
            remainder_offset=_to_tensor(remainder_offset, torch.int32, device),
            remainder_present=_to_tensor(remainder_present, torch.uint8, device),
            lethal_target=_to_tensor(lethal_target, torch.float32, device),
            prize_race_target=_to_tensor(
                prize_race_target, torch.float32, device
            ).reshape(-1, 2),
            sample_aux_class=_to_tensor(sample_aux_class, torch.int16, device),
            guide_target_index=_to_tensor(
                guide_target_index, torch.int16, device
            ),
            guide_confidence=_to_tensor(
                guide_confidence, torch.float32, device
            ),
        )
        del boards, options, sample_board, option_word_start, n_options
        del target_index, value_target, hand_index, hand_offset, hand_present
        del remainder_index, remainder_offset, remainder_present
        del lethal_target, prize_race_target, sample_aux_class
        del guide_target_index, guide_confidence
        gc.collect()
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            if free < int(float(min_free_gib) * 2**30):
                raise MemoryError(
                    f"post-pack CUDA free memory {free / 2**30:.2f} GiB < "
                    f"required {float(min_free_gib):.1f} GiB"
                )
            print(
                f"[device-corpus-exact] resident={corpus.input_bytes / 2**30:.2f} "
                f"GiB CUDA-free={free / 2**30:.2f}/{total / 2**30:.2f} GiB "
                f"build={corpus.build_seconds:.1f}s",
                flush=True,
            )
        return corpus

    def _gather_segments(
        self,
        index: torch.Tensor,
        value: torch.Tensor,
        starts: torch.Tensor,
        lengths: torch.Tensor,
    ) -> PackedSparse:
        starts64 = starts.reshape(-1).to(dtype=torch.long)
        lengths64 = lengths.reshape(-1).to(dtype=torch.long)
        out_offset64 = torch.cat(
            [
                torch.zeros(1, device=self.device, dtype=torch.long),
                torch.cumsum(lengths64, dim=0),
            ]
        )
        total = int(out_offset64[-1].item())
        if total == 0:
            return PackedSparse(
                index=torch.empty(0, device=self.device, dtype=torch.int32),
                value=torch.empty(0, device=self.device, dtype=torch.float32),
                offset=out_offset64.to(dtype=torch.int32),
            )
        rows = torch.repeat_interleave(
            torch.arange(lengths64.numel(), device=self.device), lengths64
        )
        relative = torch.arange(total, device=self.device) - out_offset64[:-1][rows]
        source = starts64[rows] + relative
        return PackedSparse(
            index=index.index_select(0, source),
            value=value.index_select(0, source),
            offset=out_offset64.to(dtype=torch.int32),
        )

    def _expand_ranges(
        self,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        starts64 = starts.reshape(-1).to(device=self.device, dtype=torch.long)
        ends64 = ends.reshape(-1).to(device=self.device, dtype=torch.long)
        lengths = ends64 - starts64
        if bool((lengths < 0).any()):
            raise ValueError("resident range end precedes start")
        total = int(lengths.sum().item())
        if total <= 0:
            return (
                torch.empty(0, device=self.device, dtype=torch.long),
                lengths,
            )
        rows = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=self.device), lengths
        )
        offsets = torch.cat(
            [
                torch.zeros(1, device=self.device, dtype=torch.long),
                torch.cumsum(lengths, dim=0),
            ]
        )
        relative = torch.arange(total, device=self.device) - offsets[:-1][rows]
        return starts64[rows] + relative, lengths

    def board_ids_batch(self, board_ids: torch.Tensor) -> PackedSparse:
        ids = board_ids.reshape(-1).to(device=self.device, dtype=torch.long)
        board_word_ids = (
            ids.unsqueeze(1) * BOARD_WORDS
            + torch.arange(BOARD_WORDS, device=self.device).unsqueeze(0)
        )
        starts = self.board_offset.index_select(0, board_word_ids.reshape(-1))
        ends = self.board_offset.index_select(0, board_word_ids.reshape(-1) + 1)
        return self._gather_segments(
            self.board_index,
            self.board_value,
            starts,
            ends - starts,
        )

    def _options_batch(
        self, sample_ids: torch.Tensor
    ) -> tuple[PackedSparse, torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = sample_ids.reshape(-1).to(device=self.device, dtype=torch.long)
        counts = self.n_options.index_select(0, ids).to(dtype=torch.long)
        if counts.numel() == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.int32)
            return (
                PackedSparse(
                    index=empty,
                    value=torch.empty(0, device=self.device),
                    offset=torch.zeros(1, device=self.device, dtype=torch.int32),
                ),
                counts,
                self.target_index.index_select(0, ids).to(dtype=torch.long),
                self.value_target.index_select(0, ids),
            )
        max_n = max(1, int(counts.max().item()))
        columns = torch.arange(max_n, device=self.device).unsqueeze(0)
        valid = columns < counts.unsqueeze(1)
        option_start = self.option_word_start.index_select(0, ids).to(
            dtype=torch.long
        )
        option_word_ids = option_start.unsqueeze(1) + columns
        safe_word_ids = torch.where(
            valid, option_word_ids, torch.zeros_like(option_word_ids)
        )
        flat_words = safe_word_ids.reshape(-1)
        option_starts = self.option_offset.index_select(0, flat_words)
        option_ends = self.option_offset.index_select(0, flat_words + 1)
        option_lengths = torch.where(
            valid.reshape(-1),
            option_ends - option_starts,
            torch.zeros_like(option_starts),
        )
        options = self._gather_segments(
            self.option_index,
            self.option_value,
            option_starts,
            option_lengths,
        )
        return (
            options,
            counts,
            self.target_index.index_select(0, ids).to(dtype=torch.long),
            self.value_target.index_select(0, ids),
        )

    def temporal_batch(
        self, game_ids: torch.Tensor
    ) -> tuple[
        PackedSparse,
        PackedSparse,
        PackedSparse,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gather whole games and shifted realized actions entirely on-device."""
        if not self.has_temporal_layout:
            raise ValueError("resident corpus has no temporal game layout")
        assert self.game_decision_offset is not None
        assert self.game_sample_offset is not None
        assert self.action_index is not None
        assert self.action_value is not None
        assert self.action_offset is not None
        ids = game_ids.reshape(-1).to(device=self.device, dtype=torch.long)
        game_count = self.train_games + self.val_games
        if ids.numel() == 0:
            raise ValueError("temporal resident batch requires at least one game")
        if bool(((ids < 0) | (ids >= game_count)).any()):
            raise IndexError("temporal resident game id is outside the corpus")

        decision_start = self.game_decision_offset.index_select(0, ids).to(
            dtype=torch.long
        )
        decision_end = self.game_decision_offset.index_select(0, ids + 1).to(
            dtype=torch.long
        )
        sample_start = self.game_sample_offset.index_select(0, ids).to(
            dtype=torch.long
        )
        sample_end = self.game_sample_offset.index_select(0, ids + 1).to(
            dtype=torch.long
        )
        decision_ids, game_lengths = self._expand_ranges(
            decision_start, decision_end
        )
        sample_ids, sample_lengths = self._expand_ranges(sample_start, sample_end)
        if decision_ids.numel() == 0 or sample_ids.numel() == 0:
            raise ValueError("temporal resident game batch has no trainable decisions")

        boards = self.board_ids_batch(decision_ids)

        # Shift realized own actions right by one timestep inside each game.
        # The first row is an empty CSR bag, exactly matching ``[None] +
        # actions[:-1]`` in the reference GameSequence path.
        decision_rows = torch.repeat_interleave(
            torch.arange(ids.numel(), device=self.device), game_lengths
        )
        decision_offsets = torch.cat(
            [
                torch.zeros(1, device=self.device, dtype=torch.long),
                torch.cumsum(game_lengths, dim=0),
            ]
        )
        within_game = torch.arange(decision_ids.numel(), device=self.device) - (
            decision_offsets[:-1][decision_rows]
        )
        previous_ids = torch.clamp(decision_ids - 1, min=0)
        action_starts = self.action_offset.index_select(0, previous_ids)
        action_ends = self.action_offset.index_select(0, previous_ids + 1)
        action_lengths = action_ends - action_starts
        action_lengths = torch.where(
            within_game > 0, action_lengths, torch.zeros_like(action_lengths)
        )
        previous_actions = self._gather_segments(
            self.action_index,
            self.action_value,
            action_starts,
            action_lengths,
        )

        sample_rows = torch.repeat_interleave(
            torch.arange(ids.numel(), device=self.device), sample_lengths
        )
        local_decision_base = decision_offsets[:-1].index_select(0, sample_rows)
        global_decision_base = decision_start.index_select(0, sample_rows)
        sample_board = self.sample_board.index_select(0, sample_ids).to(
            dtype=torch.long
        )
        sample_state_rows = (
            local_decision_base + sample_board - global_decision_base
        )
        if bool(
            ((sample_state_rows < 0) | (sample_state_rows >= decision_ids.numel())).any()
        ):
            raise AssertionError("temporal sample-to-state mapping escaped its batch")

        options, counts, targets, values = self._options_batch(sample_ids)
        return (
            boards,
            previous_actions,
            options,
            counts,
            targets,
            values,
            game_lengths,
            sample_state_rows,
        )

    def board_batch(
        self, sample_ids: torch.Tensor
    ) -> tuple[PackedSparse, torch.Tensor]:
        ids = sample_ids.reshape(-1).to(device=self.device, dtype=torch.long)
        board_ids = self.sample_board.index_select(0, ids).to(dtype=torch.long)
        return self.board_ids_batch(board_ids), board_ids

    def batch(
        self, sample_ids: torch.Tensor
    ) -> tuple[PackedSparse, PackedSparse, torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = sample_ids.reshape(-1).to(device=self.device, dtype=torch.long)
        board, _board_ids = self.board_batch(ids)

        options, counts, targets, values = self._options_batch(ids)
        return board, options, counts, targets, values

    def _card_multihot(
        self,
        board_ids: torch.Tensor,
        *,
        index: torch.Tensor,
        offset: torch.Tensor,
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        starts = offset.index_select(0, board_ids).to(dtype=torch.long)
        ends = offset.index_select(0, board_ids + 1).to(dtype=torch.long)
        lengths = ends - starts
        rows = torch.repeat_interleave(
            torch.arange(board_ids.numel(), device=self.device), lengths
        )
        total = int(lengths.sum().item())
        labels = torch.zeros(
            (board_ids.numel(), int(self.belief_card_vocab)),
            device=self.device,
            dtype=torch.float32,
        )
        if total:
            row_offsets = torch.cat(
                [
                    torch.zeros(1, device=self.device, dtype=torch.long),
                    torch.cumsum(lengths, dim=0),
                ]
            )
            relative = torch.arange(total, device=self.device) - row_offsets[:-1][rows]
            source = starts[rows] + relative
            cards = index.index_select(0, source).to(dtype=torch.long)
            labels[rows, cards] = 1.0
        mask = present.index_select(0, board_ids).to(dtype=torch.bool)
        return labels, mask

    def exact_targets(self, sample_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.has_exact_targets:
            raise RuntimeError("device corpus has no exact-hidden targets")
        ids = sample_ids.reshape(-1).to(device=self.device, dtype=torch.long)
        board_ids = self.sample_board.index_select(0, ids).to(dtype=torch.long)
        assert self.hand_index is not None and self.hand_offset is not None
        assert self.hand_present is not None
        assert self.remainder_index is not None and self.remainder_offset is not None
        assert self.remainder_present is not None
        assert self.lethal_target is not None and self.prize_race_target is not None
        assert self.sample_aux_class is not None
        hand, hand_mask = self._card_multihot(
            board_ids,
            index=self.hand_index,
            offset=self.hand_offset,
            present=self.hand_present,
        )
        remainder, remainder_mask = self._card_multihot(
            board_ids,
            index=self.remainder_index,
            offset=self.remainder_offset,
            present=self.remainder_present,
        )
        lethal = self.lethal_target.index_select(0, board_ids)
        prize_race = self.prize_race_target.index_select(0, board_ids)
        return {
            "hand": hand,
            "hand_mask": hand_mask,
            "remainder": remainder,
            "remainder_mask": remainder_mask,
            "lethal": lethal,
            "lethal_mask": torch.isfinite(lethal),
            "prize_race": prize_race,
            "prize_race_mask": torch.isfinite(prize_race).all(dim=1),
            "aux_class": self.sample_aux_class.index_select(0, ids).to(
                dtype=torch.long
            ),
        }

    def batches(
        self,
        *,
        train: bool,
        batch_size: int,
        shuffle: bool,
        seed: int,
        epoch: int,
    ) -> list[torch.Tensor]:
        count = self.train_samples if train else self.val_samples
        start = 0 if train else self.train_samples
        if count <= 0:
            return []
        if shuffle:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed) + int(epoch) * 10007)
            order = torch.randperm(count, device=self.device, generator=generator) + start
        else:
            order = torch.arange(start, start + count, device=self.device)
        size = max(1, int(batch_size))
        return list(order.split(size))

    def temporal_batches(
        self,
        *,
        train: bool,
        batch_size: int,
        shuffle: bool,
        seed: int,
        epoch: int,
    ) -> list[torch.Tensor]:
        """Batch whole games under a decision/stage activation budget."""
        if not self.has_temporal_layout:
            raise ValueError("resident corpus has no temporal game layout")
        assert self.game_decision_offset is not None
        assert self.game_sample_offset is not None
        count = self.train_games if train else self.val_games
        start = 0 if train else self.train_games
        if count <= 0:
            return []
        if shuffle:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed) + int(epoch) * 10007)
            order = (
                torch.randperm(count, device=self.device, generator=generator)
                + start
            )
        else:
            order = torch.arange(start, start + count, device=self.device)
        decision_lengths = (
            self.game_decision_offset.index_select(0, order + 1)
            - self.game_decision_offset.index_select(0, order)
        ).to(dtype=torch.long)
        sample_lengths = (
            self.game_sample_offset.index_select(0, order + 1)
            - self.game_sample_offset.index_select(0, order)
        ).to(dtype=torch.long)
        work = torch.maximum(decision_lengths, sample_lengths).cpu().tolist()
        ordered_ids = order.cpu().tolist()
        limit = max(1, int(batch_size))
        chunks: list[torch.Tensor] = []
        current: list[int] = []
        current_work = 0
        for game_id, game_work in zip(ordered_ids, work):
            game_work = int(game_work)
            if game_work <= 0:
                continue
            if current and current_work + game_work > limit:
                chunks.append(
                    torch.tensor(current, device=self.device, dtype=torch.long)
                )
                current = []
                current_work = 0
            current.append(int(game_id))
            current_work += game_work
        if current:
            chunks.append(torch.tensor(current, device=self.device, dtype=torch.long))
        return chunks

    def temporal_probe_batch(self, max_items: int) -> torch.Tensor:
        """Build a representative train batch containing the widest option row."""
        if not self.has_temporal_layout or self.train_games <= 0:
            raise ValueError("temporal probe requires train games and layout")
        assert self.game_decision_offset is not None
        assert self.game_sample_offset is not None
        widest_sample = torch.argmax(self.n_options[: self.train_samples]).to(
            dtype=torch.long
        )
        train_bounds = self.game_sample_offset[1 : self.train_games + 1].to(
            dtype=torch.long
        )
        widest_game = int(
            torch.searchsorted(train_bounds, widest_sample, right=True).item()
        )
        order = [widest_game] + [
            game_id
            for game_id in range(self.train_games)
            if game_id != widest_game
        ]
        decision_lengths = (
            self.game_decision_offset[1 : self.train_games + 1]
            - self.game_decision_offset[: self.train_games]
        ).cpu().tolist()
        sample_lengths = (
            self.game_sample_offset[1 : self.train_games + 1]
            - self.game_sample_offset[: self.train_games]
        ).cpu().tolist()
        limit = max(1, int(max_items))
        selected: list[int] = []
        used = 0
        for game_id in order:
            work = max(
                int(decision_lengths[game_id]), int(sample_lengths[game_id])
            )
            if work <= 0:
                continue
            if selected and used + work > limit:
                break
            selected.append(game_id)
            used += work
        if not selected:
            raise ValueError("temporal probe found no trainable game")
        return torch.tensor(selected, device=self.device, dtype=torch.long)

    def expected_batches(self, *, train: bool, batch_size: int) -> int:
        count = self.train_samples if train else self.val_samples
        return int(math.ceil(count / max(1, int(batch_size))))
