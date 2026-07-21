"""Durable, checksummed CPU packs for periodic expert rehearsal.

The source feature manifest is immutable but expensive to deserialize and
repack into CSR arrays.  This cache stores only the already-packed CPU tensor
state.  A cache hit verifies the complete payload before memory-mapping it, so
the remaining preparation work is one CPU-to-device transfer.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from poke_bot.device_corpus import (
    DEVICE_CORPUS_PACKING_SCHEMA_VERSION,
    DeviceResidentBootstrapCorpus,
)


EXPERT_CPU_PACK_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_PACK_PREFIX = "expert-pack-"


class ExpertCpuPackError(RuntimeError):
    """The derived CPU pack is absent, corrupt, or incompatible."""


@dataclass(frozen=True)
class ExpertCpuPackKey:
    manifest_digest: str
    split_seed: int
    val_frac: float
    max_context: Optional[int]
    belief_card_vocab: Optional[int] = None
    packing_schema: int = DEVICE_CORPUS_PACKING_SCHEMA_VERSION

    def contract(self) -> dict[str, Any]:
        digest = str(self.manifest_digest)
        if not _SHA256.fullmatch(digest):
            raise ValueError(f"invalid expert manifest digest: {digest!r}")
        fraction = float(self.val_frac)
        if not 0.0 <= fraction < 1.0:
            raise ValueError("expert validation fraction must be in [0, 1)")
        context = None if self.max_context is None else int(self.max_context)
        if context is not None and context <= 0:
            raise ValueError("expert max_context must be positive")
        schema = int(self.packing_schema)
        if schema <= 0:
            raise ValueError("expert packing schema must be positive")
        card_vocab = (
            None
            if self.belief_card_vocab is None
            else int(self.belief_card_vocab)
        )
        if card_vocab is not None and not 0 < card_vocab < 2**15:
            raise ValueError(
                f"unsupported expert belief card vocab: {card_vocab}"
            )
        return {
            "manifest_digest": digest,
            "split_seed": int(self.split_seed),
            # Hex is a stable, lossless representation across Python versions.
            "val_frac_hex": fraction.hex(),
            "max_context": context,
            "belief_card_vocab": card_vocab,
            "packing_schema": schema,
            "cache_schema": EXPERT_CPU_PACK_SCHEMA_VERSION,
        }

    @property
    def digest(self) -> str:
        raw = json.dumps(
            self.contract(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


_EXPECTED_DTYPES: dict[str, torch.dtype] = {
    "board_index": torch.int32,
    "board_value": torch.float32,
    "board_offset": torch.int32,
    "option_index": torch.int32,
    "option_value": torch.float32,
    "option_offset": torch.int32,
    "sample_board": torch.int32,
    "option_word_start": torch.int32,
    "n_options": torch.int16,
    "target_index": torch.int16,
    "value_target": torch.float32,
    "hand_index": torch.int16,
    "hand_offset": torch.int32,
    "hand_present": torch.uint8,
    "remainder_index": torch.int16,
    "remainder_offset": torch.int32,
    "remainder_present": torch.uint8,
    "lethal_target": torch.float32,
    "prize_race_target": torch.float32,
    "sample_aux_class": torch.int16,
    "guide_target_index": torch.int16,
    "guide_confidence": torch.float32,
    "action_index": torch.int32,
    "action_value": torch.float32,
    "action_offset": torch.int32,
    "game_decision_offset": torch.int32,
    "game_sample_offset": torch.int32,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(
        f".{path.name}.partial.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with partial.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        _fsync_dir(path.parent)
    finally:
        partial.unlink(missing_ok=True)


def _tensor_specs(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": str(tensor.dtype),
            "shape": [int(value) for value in tensor.shape],
            "nbytes": int(tensor.numel()) * int(tensor.element_size()),
        }
        for name, tensor in sorted(tensors.items())
    }


def _all_nondecreasing(values: torch.Tensor, *, chunk: int = 4_000_000) -> bool:
    flat = values.reshape(-1)
    if flat.numel() <= 1:
        return True
    prior: Optional[int] = None
    for start in range(0, int(flat.numel()), int(chunk)):
        part = flat[start : start + int(chunk)]
        if prior is not None and int(part[0].item()) < prior:
            return False
        if part.numel() > 1 and not bool(torch.all(part[1:] >= part[:-1])):
            return False
        prior = int(part[-1].item())
    return True


def validate_cpu_corpus(corpus: DeviceResidentBootstrapCorpus) -> None:
    """Validate every packed shape/boundary before trusting cache bytes."""
    if corpus.device.type != "cpu":
        raise ExpertCpuPackError(
            f"durable expert pack must be CPU-resident, got {corpus.device}"
        )
    tensors = corpus.tensor_state()
    for name, tensor in tensors.items():
        expected = _EXPECTED_DTYPES.get(name)
        if expected is None or tensor.dtype != expected:
            raise ExpertCpuPackError(
                f"expert pack tensor dtype mismatch {name}: "
                f"got={tensor.dtype} expected={expected}"
            )
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ExpertCpuPackError(
                f"expert pack tensor must be contiguous CPU storage: {name}"
            )

    train_samples = int(corpus.train_samples)
    val_samples = int(corpus.val_samples)
    train_games = int(corpus.train_games)
    val_games = int(corpus.val_games)
    decisions = int(corpus.decisions)
    samples = train_samples + val_samples
    games = train_games + val_games
    if min(train_samples, val_samples, train_games, val_games, decisions) < 0:
        raise ExpertCpuPackError("expert pack has negative counters")
    if samples <= 0 or games <= 0 or decisions <= 0:
        raise ExpertCpuPackError("expert pack has no trainable data")
    if int(corpus.input_bytes) != int(corpus.tensor_bytes):
        raise ExpertCpuPackError(
            "expert pack byte counter mismatch: "
            f"metadata={corpus.input_bytes} tensors={corpus.tensor_bytes}"
        )
    if float(corpus.build_seconds) < 0.0:
        raise ExpertCpuPackError("expert pack has a negative build duration")

    expected_rows = {
        "sample_board": samples,
        "option_word_start": samples,
        "n_options": samples,
        "target_index": samples,
        "value_target": samples,
        "guide_target_index": samples,
        "guide_confidence": samples,
        "board_offset": decisions * 24 + 1,
        "action_offset": decisions + 1,
        "game_decision_offset": games + 1,
        "game_sample_offset": games + 1,
    }
    for name, count in expected_rows.items():
        tensor = tensors.get(name)
        if tensor is None or tensor.ndim != 1 or int(tensor.numel()) != count:
            raise ExpertCpuPackError(
                f"expert pack tensor shape mismatch {name}: "
                f"got={None if tensor is None else list(tensor.shape)} "
                f"expected=[{count}]"
            )

    for prefix in ("board", "option", "action"):
        index = tensors[f"{prefix}_index"]
        value = tensors[f"{prefix}_value"]
        offset = tensors[f"{prefix}_offset"]
        if index.ndim != 1 or value.ndim != 1 or index.numel() != value.numel():
            raise ExpertCpuPackError(f"expert pack {prefix} CSR values disagree")
        if int(offset[0].item()) != 0 or int(offset[-1].item()) != index.numel():
            raise ExpertCpuPackError(f"expert pack {prefix} CSR boundary mismatch")
        if not _all_nondecreasing(offset):
            raise ExpertCpuPackError(f"expert pack {prefix} CSR is not monotonic")
        if bool(torch.any(index < 0)):
            raise ExpertCpuPackError(
                f"expert pack {prefix} CSR contains a negative feature index"
            )

    if bool(torch.any(corpus.n_options <= 0)):
        raise ExpertCpuPackError("expert pack contains a nonpositive option count")
    targets = corpus.target_index.to(dtype=torch.int32)
    counts = corpus.n_options.to(dtype=torch.int32)
    if bool(torch.any(targets < 0)) or bool(torch.any(targets >= counts)):
        raise ExpertCpuPackError("expert pack target is outside its option row")
    if bool(torch.any(corpus.sample_board < 0)) or bool(
        torch.any(corpus.sample_board >= decisions)
    ):
        raise ExpertCpuPackError("expert pack sample references an invalid board")
    starts = corpus.option_word_start.to(dtype=torch.int64)
    if int(starts[0].item()) != 0:
        raise ExpertCpuPackError("expert pack option prefix does not start at zero")
    if starts.numel() > 1 and not bool(
        torch.all(starts[1:] - starts[:-1] == counts[:-1])
    ):
        raise ExpertCpuPackError("expert pack option prefixes are discontinuous")
    if int(starts[-1].item()) + int(counts[-1].item()) != int(counts.sum().item()):
        raise ExpertCpuPackError("expert pack option prefix total is inconsistent")

    for name, final, split_index, split_value in (
        ("game_decision_offset", decisions, None, None),
        ("game_sample_offset", samples, train_games, train_samples),
    ):
        offsets = tensors[name]
        if (
            int(offsets[0].item()) != 0
            or int(offsets[-1].item()) != final
            or not _all_nondecreasing(offsets)
        ):
            raise ExpertCpuPackError(f"expert pack {name} boundaries disagree")
        if split_index is not None and int(offsets[split_index].item()) != split_value:
            raise ExpertCpuPackError(f"expert pack {name} split boundary disagrees")

    temporal = (
        corpus.action_index,
        corpus.action_value,
        corpus.action_offset,
        corpus.game_decision_offset,
        corpus.game_sample_offset,
    )
    if any(value is None for value in temporal):
        raise ExpertCpuPackError("expert CPU pack lacks temporal game layout")

    exact = (
        corpus.hand_index,
        corpus.hand_offset,
        corpus.hand_present,
        corpus.remainder_index,
        corpus.remainder_offset,
        corpus.remainder_present,
        corpus.lethal_target,
        corpus.prize_race_target,
        corpus.sample_aux_class,
    )
    has_any_exact = any(value is not None for value in exact)
    has_all_exact = all(value is not None for value in exact)
    if has_any_exact and not has_all_exact:
        raise ExpertCpuPackError("expert pack has a partial exact-target layout")
    card_vocab = int(corpus.belief_card_vocab)
    if card_vocab < 0 or card_vocab >= 2**15:
        raise ExpertCpuPackError("expert pack belief vocab is outside int16 range")
    if (card_vocab > 0) != has_all_exact:
        raise ExpertCpuPackError(
            "expert pack belief vocab/exact-target layout disagree"
        )
    if has_all_exact:
        assert corpus.hand_index is not None
        assert corpus.hand_offset is not None
        assert corpus.hand_present is not None
        assert corpus.remainder_index is not None
        assert corpus.remainder_offset is not None
        assert corpus.remainder_present is not None
        assert corpus.lethal_target is not None
        assert corpus.prize_race_target is not None
        assert corpus.sample_aux_class is not None
        for prefix, index, offset, present in (
            (
                "hand",
                corpus.hand_index,
                corpus.hand_offset,
                corpus.hand_present,
            ),
            (
                "remainder",
                corpus.remainder_index,
                corpus.remainder_offset,
                corpus.remainder_present,
            ),
        ):
            if (
                offset.ndim != 1
                or int(offset.numel()) != decisions + 1
                or present.ndim != 1
                or int(present.numel()) != decisions
                or int(offset[0].item()) != 0
                or int(offset[-1].item()) != int(index.numel())
                or not _all_nondecreasing(offset)
            ):
                raise ExpertCpuPackError(
                    f"expert pack exact {prefix} CSR boundaries disagree"
                )
            if bool(torch.any(index < 0)) or bool(torch.any(index >= card_vocab)):
                raise ExpertCpuPackError(
                    f"expert pack exact {prefix} card id is outside vocab"
                )
            if bool(torch.any((present != 0) & (present != 1))):
                raise ExpertCpuPackError(
                    f"expert pack exact {prefix} presence mask is not binary"
                )
        if corpus.lethal_target.shape != (decisions,):
            raise ExpertCpuPackError("expert pack lethal target shape disagrees")
        if corpus.prize_race_target.shape != (decisions, 2):
            raise ExpertCpuPackError("expert pack prize-race target shape disagrees")
        if corpus.sample_aux_class.shape != (samples,):
            raise ExpertCpuPackError("expert pack archetype target shape disagrees")


class ExpertCpuPackCache:
    """One-active-pack cache with atomic replacement and strict validation."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def _paths(self, key: ExpertCpuPackKey) -> tuple[Path, Path]:
        stem = f"{_PACK_PREFIX}{key.digest}"
        return self.root / f"{stem}.pt", self.root / f"{stem}.json"

    @property
    def active_path(self) -> Path:
        return self.root / "active.json"

    def _discard(self, key: ExpertCpuPackKey) -> None:
        payload, manifest = self._paths(key)
        payload.unlink(missing_ok=True)
        manifest.unlink(missing_ok=True)
        if self.active_path.is_file():
            try:
                active = json.loads(self.active_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                active = {}
            if str(active.get("key") or "") == key.digest:
                self.active_path.unlink(missing_ok=True)

    def _prune(self, keep: Optional[ExpertCpuPackKey]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        keep_names = set()
        if keep is not None:
            keep_names = {path.name for path in self._paths(keep)}
        for pattern in (f"{_PACK_PREFIX}*.pt", f"{_PACK_PREFIX}*.json"):
            for path in self.root.glob(pattern):
                if path.name not in keep_names:
                    path.unlink(missing_ok=True)
        for partial in self.root.glob(".*.partial.*"):
            partial.unlink(missing_ok=True)
        remove_active = keep is None
        if keep is not None and self.active_path.is_file():
            try:
                active = json.loads(
                    self.active_path.read_text(encoding="utf-8")
                )
                remove_active = str(active.get("key") or "") != keep.digest
            except (OSError, json.JSONDecodeError):
                remove_active = True
        if remove_active:
            self.active_path.unlink(missing_ok=True)
        _fsync_dir(self.root)

    def _load(self, key: ExpertCpuPackKey) -> DeviceResidentBootstrapCorpus:
        payload_path, manifest_path = self._paths(key)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            stat = payload_path.stat()
        except (OSError, json.JSONDecodeError) as exc:
            raise ExpertCpuPackError(f"expert CPU pack metadata unavailable: {exc}") from exc
        if (
            manifest.get("schema") != "poke_bot.expert_cpu_pack/v1"
            or int(manifest.get("schema_version", -1))
            != EXPERT_CPU_PACK_SCHEMA_VERSION
            or str(manifest.get("key") or "") != key.digest
            or manifest.get("contract") != key.contract()
            or int(manifest.get("payload_bytes", -1)) != int(stat.st_size)
        ):
            raise ExpertCpuPackError("expert CPU pack manifest contract mismatch")
        expected_digest = str(manifest.get("payload_sha256") or "")
        if not _SHA256.fullmatch(expected_digest):
            raise ExpertCpuPackError("expert CPU pack has no valid payload checksum")
        actual_digest = _sha256_file(payload_path)
        if actual_digest != expected_digest:
            raise ExpertCpuPackError(
                "expert CPU pack checksum mismatch: "
                f"expected={expected_digest} actual={actual_digest}"
            )
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
        except Exception as exc:  # noqa: BLE001 - wrap untrusted derived bytes
            raise ExpertCpuPackError(f"expert CPU pack cannot be loaded: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("contract") != key.contract():
            raise ExpertCpuPackError("expert CPU pack payload contract mismatch")
        tensors = payload.get("tensors")
        scalars = payload.get("scalars")
        if not isinstance(tensors, dict) or not isinstance(scalars, dict):
            raise ExpertCpuPackError("expert CPU pack payload shape is invalid")
        if _tensor_specs(tensors) != manifest.get("tensor_specs"):
            raise ExpertCpuPackError("expert CPU pack tensor inventory mismatch")
        if scalars != manifest.get("scalar_state"):
            raise ExpertCpuPackError("expert CPU pack scalar metadata mismatch")
        try:
            corpus = DeviceResidentBootstrapCorpus.from_packed_state(
                tensors=tensors,
                scalars=scalars,
            )
            validate_cpu_corpus(corpus)
            expected_vocab = int(key.belief_card_vocab or 0)
            if int(corpus.belief_card_vocab) != expected_vocab:
                raise ExpertCpuPackError(
                    "expert CPU pack belief vocab mismatch: "
                    f"got={corpus.belief_card_vocab} expected={expected_vocab}"
                )
        except (TypeError, ValueError, ExpertCpuPackError) as exc:
            raise ExpertCpuPackError(f"expert CPU pack validation failed: {exc}") from exc
        return corpus

    def _write(
        self,
        key: ExpertCpuPackKey,
        corpus: DeviceResidentBootstrapCorpus,
    ) -> None:
        validate_cpu_corpus(corpus)
        expected_vocab = int(key.belief_card_vocab or 0)
        if int(corpus.belief_card_vocab) != expected_vocab:
            raise ExpertCpuPackError(
                "expert CPU pack builder returned the wrong belief vocab: "
                f"got={corpus.belief_card_vocab} expected={expected_vocab}"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        payload_path, manifest_path = self._paths(key)
        partial = payload_path.with_name(
            f".{payload_path.name}.partial.{os.getpid()}.{time.time_ns()}"
        )
        tensors = corpus.tensor_state()
        scalars = corpus.scalar_state()
        payload = {
            "schema_version": EXPERT_CPU_PACK_SCHEMA_VERSION,
            "contract": key.contract(),
            "scalars": scalars,
            "tensors": tensors,
        }
        try:
            with partial.open("xb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            manifest = {
                "schema": "poke_bot.expert_cpu_pack/v1",
                "schema_version": EXPERT_CPU_PACK_SCHEMA_VERSION,
                "created_at": time.time(),
                "key": key.digest,
                "contract": key.contract(),
                "payload": payload_path.name,
                "payload_bytes": int(partial.stat().st_size),
                "payload_sha256": _sha256_file(partial),
                "scalar_state": scalars,
                "tensor_specs": _tensor_specs(tensors),
            }
            os.replace(partial, payload_path)
            _fsync_dir(self.root)
            _write_atomic_json(manifest_path, manifest)
        finally:
            partial.unlink(missing_ok=True)

    def _activate(
        self,
        key: ExpertCpuPackKey,
        *,
        cache_hit: bool,
    ) -> None:
        payload_path, manifest_path = self._paths(key)
        _write_atomic_json(
            self.active_path,
            {
                "schema": "poke_bot.expert_cpu_pack.active/v1",
                "key": key.digest,
                "contract": key.contract(),
                "payload": str(payload_path),
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "cache_hit": bool(cache_hit),
                "activated_at": time.time(),
            },
        )

    def load_or_build(
        self,
        key: ExpertCpuPackKey,
        builder: Callable[[], DeviceResidentBootstrapCorpus],
    ) -> tuple[DeviceResidentBootstrapCorpus, dict[str, Any]]:
        """Return a validated CPU pack, rebuilding only derived corruption."""
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".build.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                # Delete obsolete full packs before a new build, bounding the
                # cache to approximately one current corpus rather than
                # old+new history. The lock prevents two trainers from
                # interleaving payload/manifest replacement.
                self._prune(keep=key)
                started = time.monotonic()
                try:
                    corpus = self._load(key)
                    hit = True
                except ExpertCpuPackError as exc:
                    print(
                        f"[expert-cpu-pack] MISS/REBUILD key={key.digest[:16]} "
                        f"reason={exc}",
                        flush=True,
                    )
                    self._discard(key)
                    self._prune(keep=None)
                    built = builder()
                    try:
                        validate_cpu_corpus(built)
                        self._write(key, built)
                    finally:
                        del built
                    # Trust only the same strict loader used on a future
                    # restart.
                    corpus = self._load(key)
                    hit = False
                self._activate(key, cache_hit=hit)
                self._prune(keep=key)
                payload_path, manifest_path = self._paths(key)
                info = {
                    "schema": EXPERT_CPU_PACK_SCHEMA_VERSION,
                    "key": key.digest,
                    "cache_hit": hit,
                    "payload": str(payload_path),
                    "manifest": str(manifest_path),
                    "bytes": int(payload_path.stat().st_size),
                    "elapsed_sec": time.monotonic() - started,
                }
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        print(
            f"[expert-cpu-pack] {'HIT' if hit else 'BUILT'} "
            f"key={key.digest[:16]} bytes={info['bytes']} "
            f"elapsed={info['elapsed_sec']:.1f}s",
            flush=True,
        )
        return corpus, info
