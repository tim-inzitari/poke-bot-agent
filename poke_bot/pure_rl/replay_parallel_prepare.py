"""Deterministic, checksum-backed parallel preparation of pure-RL replay.

This backend is deliberately opt-in.  It does not replace the legacy
``dataset_from_shard`` path: callers select it at a sealed boundary and may
fall back to the serial builder unless ``strict=True``.

Workers receive only immutable byte/row ranges and paths.  Each worker decodes
its source rows once, performs the ordinary compact-game conversion (including
OwnDeckLedger, guide and strategic labels), packs the result into contiguous
CPU tensors, and persists one checksum-backed fragment.  The parent receives
small descriptors, verifies every range and fragment, and merges descriptors
in source order regardless of completion order.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from poke_bot.device_corpus import DeviceResidentBootstrapCorpus
from poke_bot.matchup_adapter_activation import (
    adapter_training_ticket,
    training_routes_for_sequence,
)
from poke_bot.pure_rl.dataset_bridge import (
    _compact_game_from_raw,
    compact_game_to_sequence,
)
from poke_bot.pure_rl.expert_cpu_pack import validate_cpu_corpus
from poke_bot.pure_rl.expert_parallel_pack import (
    FragmentDescriptor,
    _load_fragment,
    _merge_fragments,
    _write_fragment,
)
from poke_bot.r279_contiguous_expert_pack import (
    build_side_tensors,
    merge_side_tensors,
)


SCHEMA = "poke_bot.parallel_replay_prepare/v1"
RANGE_SCHEMA = "poke_bot.parallel_replay_range/v1"
RECEIPT_SCHEMA = "poke_bot.parallel_replay_validation_receipt/v1"
ADAPTER_PACK_SCHEMA = "poke_bot.parallel_replay_adapter_routing/v1"
SUPPORTED_WORKERS = frozenset({1, 2, 4, 8, 16, 32})
_SEMANTIC_ENV_KEYS = (
    "POKEBOT_OWN_DECK_LEDGER_RUNTIME",
    "POKEBOT_CURRENT_DECK_GUIDE",
    "POKEBOT_CURRENT_DECK_GUIDE_TARGETS",
    "POKEBOT_CURRENT_DECK_GUIDE_VERSION",
    "POKEBOT_MATCHUP_ADAPTER_RUNTIME",
    "POKEBOT_PUBLIC_MATCHUP_TREE_PATH",
    "POKEBOT_R274_DISABLE_RL_TACTICAL_COTRAIN",
    "COMBO_STATE_ROUTE_ENABLED",
    "POKEBOT_COMBO_STATE_ROUTE_ENABLED",
    "POKEBOT_USE_RECURSIVE_TURN_PLANNER",
    "POKEBOT_SEARCH_MODE",
    "POKEBOT_SUBMISSION_SEARCH_DISABLE",
)


class ParallelReplayUnavailable(RuntimeError):
    """The opt-in backend cannot run in this environment."""


@dataclass(frozen=True)
class SourceIdentity:
    path: str
    size_bytes: int
    mtime_ns: int
    sha256: str
    rows: int


@dataclass(frozen=True)
class RangeIdentity:
    ordinal: int
    row_start: int
    row_end: int
    byte_start: int
    byte_end: int
    identity: str


@dataclass(frozen=True)
class RangeResult:
    range: RangeIdentity
    fragment: Optional[FragmentDescriptor]
    rows: int
    games: int
    decisions: int
    source_sha256: str
    elapsed_sec: float
    adapter_fragment_path: str
    adapter_fragment_sha256: str
    side_fragment_path: str
    side_fragment_sha256: str


@dataclass(frozen=True)
class PackedAdapterRouting:
    """Flat, checksum-bound adapter metadata aligned to packed game order."""

    game_route: torch.Tensor
    game_seat: torch.Tensor
    game_source_row: torch.Tensor
    game_decisions: torch.Tensor
    episode_utf8: torch.Tensor
    episode_offset: torch.Tensor
    ticket_utf8: torch.Tensor
    ticket_offset: torch.Tensor

    @property
    def games(self) -> int:
        return int(self.game_route.numel())

    @property
    def ticketed_games(self) -> int:
        return int((self.game_route >= 0).sum().item())

    @property
    def ticketed_decisions(self) -> int:
        mask = self.game_route >= 0
        return int(self.game_decisions[mask].sum().item()) if bool(mask.any()) else 0

    def tensor_state(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name)
            for name in (
                "game_route",
                "game_seat",
                "game_source_row",
                "game_decisions",
                "episode_utf8",
                "episode_offset",
                "ticket_utf8",
                "ticket_offset",
            )
        }

    def to(self, device: torch.device) -> "PackedAdapterRouting":
        return PackedAdapterRouting(
            **{name: value.to(device=device) for name, value in self.tensor_state().items()}
        )

    def routed_batches(
        self,
        *,
        games_per_batch: int,
        max_decisions: int,
        shuffle: bool,
        seed: int,
        epoch: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Reproduce legacy routed-GameSequence batching with integer IDs only."""

        import random

        routed = torch.nonzero(self.game_route >= 0, as_tuple=False).flatten().tolist()
        order = list(range(len(routed)))
        if shuffle:
            random.Random(int(seed) + int(epoch) * 10007).shuffle(order)
        result: list[torch.Tensor] = []
        current: list[int] = []
        current_decisions = 0
        for index in order:
            game_id = int(routed[index])
            decisions = int(self.game_decisions[game_id].item())
            if current and (
                len(current) >= int(games_per_batch)
                or current_decisions + decisions > int(max_decisions)
            ):
                result.append(torch.tensor(current, dtype=torch.long, device=device))
                current = []
                current_decisions = 0
            current.append(game_id)
            current_decisions += decisions
        if current:
            result.append(torch.tensor(current, dtype=torch.long, device=device))
        return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(dict(value), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def scan_source(path: Path) -> tuple[SourceIdentity, list[int]]:
    """Hash a newline-complete source once and return every row boundary."""

    path = Path(path).resolve()
    stat_before = path.stat()
    digest = hashlib.sha256()
    offsets = [0]
    with path.open("rb") as stream:
        while True:
            raw = stream.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                raise RuntimeError("parallel replay source ends with a partial row")
            digest.update(raw)
            offsets.append(stream.tell())
    stat_after = path.stat()
    if (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
        or offsets[-1] != stat_after.st_size
    ):
        raise RuntimeError("parallel replay source changed while scanning")
    return (
        SourceIdentity(
            path=str(path),
            size_bytes=int(stat_after.st_size),
            mtime_ns=int(stat_after.st_mtime_ns),
            sha256="sha256:" + digest.hexdigest(),
            rows=len(offsets) - 1,
        ),
        offsets,
    )


def deterministic_ranges(
    source: SourceIdentity,
    offsets: Sequence[int],
    *,
    target_ranges: int,
) -> list[RangeIdentity]:
    """Split rows, not cache partitions, into exact canonical ranges."""

    if len(offsets) != source.rows + 1 or offsets[0] != 0:
        raise RuntimeError("parallel replay row index disagrees with source")
    count = max(1, min(int(target_ranges), max(1, source.rows)))
    result: list[RangeIdentity] = []
    for ordinal in range(count):
        row_start = source.rows * ordinal // count
        row_end = source.rows * (ordinal + 1) // count
        payload = {
            "schema": RANGE_SCHEMA,
            "source_sha256": source.sha256,
            "ordinal": ordinal,
            "row_start": row_start,
            "row_end": row_end,
            "byte_start": int(offsets[row_start]),
            "byte_end": int(offsets[row_end]),
        }
        result.append(
            RangeIdentity(
                ordinal=ordinal,
                row_start=row_start,
                row_end=row_end,
                byte_start=int(offsets[row_start]),
                byte_end=int(offsets[row_end]),
                identity=_canonical_digest(payload),
            )
        )
    _validate_ranges(source, result)
    return result


def _validate_ranges(source: SourceIdentity, ranges: Sequence[RangeIdentity]) -> None:
    if not ranges:
        raise RuntimeError("parallel replay has no ranges")
    seen: set[str] = set()
    expected_row = 0
    expected_byte = 0
    for ordinal, item in enumerate(ranges):
        if (
            item.ordinal != ordinal
            or item.identity in seen
            or item.row_start != expected_row
            or item.byte_start != expected_byte
            or item.row_end <= item.row_start
            or item.byte_end <= item.byte_start
        ):
            raise RuntimeError("parallel replay has duplicate, missing, or unordered ranges")
        seen.add(item.identity)
        expected_row = item.row_end
        expected_byte = item.byte_end
    if expected_row != source.rows or expected_byte != source.size_bytes:
        raise RuntimeError("parallel replay ranges do not cover the exact source")


def _tensor_digest(corpus: DeviceResidentBootstrapCorpus) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(corpus.tensor_state().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(contiguous.dtype).encode() + b"\0")
        digest.update(json.dumps(list(contiguous.shape)).encode() + b"\0")
        digest.update(contiguous.numpy().tobytes(order="C"))
    digest.update(
        json.dumps(
            _semantic_scalars(corpus), sort_keys=True, separators=(",", ":")
        ).encode()
    )
    return "sha256:" + digest.hexdigest()


def _semantic_scalars(corpus: DeviceResidentBootstrapCorpus) -> dict[str, Any]:
    # Wall time is diagnostic metadata, not a loss input and cannot be equal
    # across worker topologies. Every count/schema/input-byte scalar remains
    # part of parity and the semantic output digest.
    return {
        key: value
        for key, value in corpus.scalar_state().items()
        if key != "build_seconds"
    }


def _utf8_column(values: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    raw = bytearray()
    offsets = [0]
    for value in values:
        raw.extend(str(value).encode("utf-8"))
        offsets.append(len(raw))
    data = torch.frombuffer(raw, dtype=torch.uint8).clone() if raw else torch.empty(0, dtype=torch.uint8)
    return data, torch.tensor(offsets, dtype=torch.int64)


def _validate_adapter_routing(value: PackedAdapterRouting) -> None:
    games = value.games
    expected = {
        "game_route": (torch.int16, games),
        "game_seat": (torch.int8, games),
        "game_source_row": (torch.int64, games),
        "game_decisions": (torch.int32, games),
        "episode_offset": (torch.int64, games + 1),
        "ticket_offset": (torch.int64, games + 1),
    }
    for name, (dtype, count) in expected.items():
        tensor = getattr(value, name)
        if tensor.device.type != "cpu" or tensor.dtype != dtype or tensor.ndim != 1:
            raise RuntimeError(f"packed adapter routing tensor contract changed: {name}")
        if int(tensor.numel()) != count:
            raise RuntimeError(f"packed adapter routing length changed: {name}")
    for name, data_name in (("episode_offset", "episode_utf8"), ("ticket_offset", "ticket_utf8")):
        offsets = getattr(value, name)
        data = getattr(value, data_name)
        if data.device.type != "cpu" or data.dtype != torch.uint8 or data.ndim != 1:
            raise RuntimeError(f"packed adapter routing byte column changed: {data_name}")
        if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != int(data.numel()):
            raise RuntimeError(f"packed adapter routing offsets changed: {name}")
        if bool((offsets[1:] < offsets[:-1]).any()):
            raise RuntimeError(f"packed adapter routing offsets are unordered: {name}")
    if bool(((value.game_seat != 0) & (value.game_seat != 1)).any()):
        raise RuntimeError("packed adapter routing has an invalid seat")
    if bool((value.game_decisions <= 0).any()):
        raise RuntimeError("packed adapter routing has an empty game")


def _adapter_digest(value: PackedAdapterRouting) -> str:
    _validate_adapter_routing(value)
    digest = hashlib.sha256()
    digest.update(ADAPTER_PACK_SCHEMA.encode())
    for name, tensor in sorted(value.tensor_state().items()):
        contiguous = tensor.contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(contiguous.dtype).encode() + b"\0")
        digest.update(contiguous.numpy().tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def _write_adapter_fragment(
    spool: Path,
    *,
    ordinal: int,
    routes: Sequence[int],
    seats: Sequence[int],
    source_rows: Sequence[int],
    decisions: Sequence[int],
    episodes: Sequence[str],
    tickets: Sequence[str],
) -> tuple[str, str]:
    episode_utf8, episode_offset = _utf8_column(episodes)
    ticket_utf8, ticket_offset = _utf8_column(tickets)
    packed = PackedAdapterRouting(
        game_route=torch.tensor(routes, dtype=torch.int16),
        game_seat=torch.tensor(seats, dtype=torch.int8),
        game_source_row=torch.tensor(source_rows, dtype=torch.int64),
        game_decisions=torch.tensor(decisions, dtype=torch.int32),
        episode_utf8=episode_utf8,
        episode_offset=episode_offset,
        ticket_utf8=ticket_utf8,
        ticket_offset=ticket_offset,
    )
    _validate_adapter_routing(packed)
    path = spool / f"adapter-{ordinal:06d}.pt"
    with path.open("xb") as stream:
        torch.save({"schema": ADAPTER_PACK_SCHEMA, "tensors": packed.tensor_state()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return str(path), _sha256_file(path)


def _write_side_fragment(
    spool: Path,
    *,
    ordinal: int,
    sequences: Sequence[Any],
    card_vocab: Optional[int],
) -> tuple[str, str]:
    if card_vocab is None:
        return "", ""
    side, _counts = build_side_tensors(sequences, card_vocab=int(card_vocab))
    path = spool / f"side-{ordinal:06d}.pt"
    with path.open("xb") as stream:
        torch.save({"schema": "poke_bot.parallel_replay_side/v1", "tensors": side}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return str(path), _sha256_file(path)


def _load_side_fragment(path: Path, expected_sha256: str) -> dict[str, torch.Tensor]:
    if _sha256_file(path) != expected_sha256:
        raise RuntimeError("packed replay side fragment checksum changed")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema") != "poke_bot.parallel_replay_side/v1":
        raise RuntimeError("packed replay side fragment schema changed")
    return dict(payload["tensors"])


def _load_adapter_fragment(path: Path, expected_sha256: str) -> PackedAdapterRouting:
    if _sha256_file(path) != expected_sha256:
        raise RuntimeError("packed adapter routing fragment checksum changed")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema") != ADAPTER_PACK_SCHEMA:
        raise RuntimeError("packed adapter routing fragment schema changed")
    packed = PackedAdapterRouting(**dict(payload["tensors"]))
    _validate_adapter_routing(packed)
    return packed


def _merge_adapter_fragments(results: Sequence[RangeResult]) -> PackedAdapterRouting:
    columns: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("game_route", "game_seat", "game_source_row", "game_decisions")
    }
    byte_columns: dict[str, list[torch.Tensor]] = {"episode_utf8": [], "ticket_utf8": []}
    offsets: dict[str, list[torch.Tensor]] = {"episode_offset": [], "ticket_offset": []}
    byte_bases = {"episode_offset": 0, "ticket_offset": 0}
    for result in results:
        packed = _load_adapter_fragment(
            Path(result.adapter_fragment_path), result.adapter_fragment_sha256
        )
        for name in columns:
            columns[name].append(getattr(packed, name))
        for data_name, offset_name in (("episode_utf8", "episode_offset"), ("ticket_utf8", "ticket_offset")):
            data = getattr(packed, data_name)
            source_offsets = getattr(packed, offset_name)
            byte_columns[data_name].append(data)
            offsets[offset_name].append(source_offsets[:-1] + byte_bases[offset_name])
            byte_bases[offset_name] += int(data.numel())
    merged = PackedAdapterRouting(
        **{name: torch.cat(parts) for name, parts in columns.items()},
        **{name: torch.cat(parts) if parts else torch.empty(0, dtype=torch.uint8) for name, parts in byte_columns.items()},
        **{
            name: torch.cat([*parts, torch.tensor([byte_bases[name]], dtype=torch.int64)])
            for name, parts in offsets.items()
        },
    )
    _validate_adapter_routing(merged)
    return merged


def _range_worker(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        source_raw,
        range_raw,
        spool_raw,
        verify_info_set,
        max_context,
        exact_card_vocab,
        force_strategic,
    ) = task
    torch.set_num_threads(1)
    source = SourceIdentity(**source_raw)
    item = RangeIdentity(**range_raw)
    path = Path(source.path)
    stat = path.stat()
    if stat.st_size != source.size_bytes or stat.st_mtime_ns != source.mtime_ns:
        raise RuntimeError("parallel replay source stat changed before worker read")
    started = time.monotonic()
    sequences = []
    adapter_routes: list[int] = []
    adapter_seats: list[int] = []
    adapter_source_rows: list[int] = []
    adapter_decisions: list[int] = []
    adapter_episodes: list[str] = []
    adapter_tickets: list[str] = []
    rows = 0
    source_digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(item.byte_start)
        while stream.tell() < item.byte_end:
            raw = stream.readline()
            if not raw or not raw.endswith(b"\n") or stream.tell() > item.byte_end:
                raise RuntimeError("parallel replay worker encountered a corrupt range")
            source_digest.update(raw)
            game = _compact_game_from_raw(raw)
            sequence = compact_game_to_sequence(
                game,
                verify_info_set=bool(verify_info_set),
                max_context=int(max_context),
            )
            if sequence is not None:
                sequences.append(sequence)
                raw_ticket = dict(sequence.matchup_adapter_training_ticket or {})
                route = -1
                ticket_json = ""
                if raw_ticket:
                    ticket = adapter_training_ticket(sequence)
                    decision_routes = training_routes_for_sequence(sequence)
                    if not decision_routes or any(
                        candidate != int(ticket.route) for candidate in decision_routes
                    ):
                        raise RuntimeError("packed adapter route changed within one game")
                    route = int(ticket.route)
                    ticket_json = json.dumps(
                        raw_ticket, sort_keys=True, separators=(",", ":")
                    )
                adapter_routes.append(route)
                adapter_seats.append(int(sequence.seat))
                adapter_source_rows.append(item.row_start + rows)
                adapter_decisions.append(len(sequence.decisions))
                adapter_episodes.append(str(sequence.episode_id))
                adapter_tickets.append(ticket_json)
            rows += 1
        final_position = stream.tell()
    if final_position != item.byte_end or rows != item.row_end - item.row_start:
        raise RuntimeError("parallel replay worker range coverage changed")
    corpus = DeviceResidentBootstrapCorpus.from_splits(
        sequences,
        (),
        device=torch.device("cpu"),
        exact_card_vocab=exact_card_vocab,
        force_expanded_strategic=bool(force_strategic),
    )
    descriptor = None
    if int(corpus.decisions) > 0:
        validate_cpu_corpus(corpus, allow_empty_training_fragment=True)
        descriptor = _write_fragment(
            Path(spool_raw),
            ordinal=item.ordinal,
            partition="train",
            corpus=corpus,
        )
    adapter_fragment_path, adapter_fragment_sha256 = _write_adapter_fragment(
        Path(spool_raw),
        ordinal=item.ordinal,
        routes=adapter_routes,
        seats=adapter_seats,
        source_rows=adapter_source_rows,
        decisions=adapter_decisions,
        episodes=adapter_episodes,
        tickets=adapter_tickets,
    )
    side_fragment_path, side_fragment_sha256 = _write_side_fragment(
        Path(spool_raw),
        ordinal=item.ordinal,
        sequences=sequences,
        card_vocab=exact_card_vocab,
    )
    result = RangeResult(
        range=item,
        fragment=descriptor,
        rows=rows,
        games=int(corpus.train_games),
        decisions=int(corpus.decisions),
        source_sha256="sha256:" + source_digest.hexdigest(),
        elapsed_sec=time.monotonic() - started,
        adapter_fragment_path=adapter_fragment_path,
        adapter_fragment_sha256=adapter_fragment_sha256,
        side_fragment_path=side_fragment_path,
        side_fragment_sha256=side_fragment_sha256,
    )
    return {
        **asdict(result),
        "fragment": None if descriptor is None else asdict(descriptor),
    }


def _parse_result(value: Mapping[str, Any]) -> RangeResult:
    return RangeResult(
        range=RangeIdentity(**dict(value["range"])),
        fragment=(
            None
            if value.get("fragment") is None
            else FragmentDescriptor(**dict(value["fragment"]))
        ),
        rows=int(value["rows"]),
        games=int(value["games"]),
        decisions=int(value["decisions"]),
        source_sha256=str(value["source_sha256"]),
        elapsed_sec=float(value["elapsed_sec"]),
        adapter_fragment_path=str(value["adapter_fragment_path"]),
        adapter_fragment_sha256=str(value["adapter_fragment_sha256"]),
        side_fragment_path=str(value.get("side_fragment_path") or ""),
        side_fragment_sha256=str(value.get("side_fragment_sha256") or ""),
    )


def _write_pack(
    output: Path,
    corpus: DeviceResidentBootstrapCorpus,
    adapter_routing: PackedAdapterRouting,
    manifest: Mapping[str, Any],
    side_tensors: Optional[Mapping[str, torch.Tensor]] = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    pack = output / "joined-pack.pt"
    partial = output / f".joined-pack.pt.partial.{os.getpid()}.{time.time_ns()}"
    payload = {
        "schema": SCHEMA,
        "scalars": corpus.scalar_state(),
        "tensors": corpus.tensor_state(),
    }
    try:
        with partial.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, pack)
    finally:
        partial.unlink(missing_ok=True)
    tensor_digest = _tensor_digest(corpus)
    adapter_path = output / "adapter-routing.pt"
    with adapter_path.open("xb") as stream:
        torch.save(
            {"schema": ADAPTER_PACK_SCHEMA, "tensors": adapter_routing.tensor_state()},
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    adapter_digest = _adapter_digest(adapter_routing)
    side_path: Optional[Path] = None
    side_sha256 = ""
    if side_tensors is not None:
        side_path = output / "side-tensors.pt"
        with side_path.open("xb") as stream:
            torch.save(
                {"schema": "poke_bot.parallel_replay_side/v1", "tensors": dict(side_tensors)},
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        side_sha256 = _sha256_file(side_path)
    result = {
        **dict(manifest),
        "pack": pack.name,
        "pack_sha256": _sha256_file(pack),
        "tensor_digest": tensor_digest,
        "adapter_pack": adapter_path.name,
        "adapter_pack_sha256": _sha256_file(adapter_path),
        "adapter_digest": adapter_digest,
        "adapter_ticketed_games": adapter_routing.ticketed_games,
        "adapter_ticketed_decisions": adapter_routing.ticketed_decisions,
        "side_pack": None if side_path is None else side_path.name,
        "side_pack_sha256": side_sha256,
    }
    # The semantic output identity excludes worker count, scheduling, timing,
    # serialization bytes and timestamps.  Rebuilding with any supported
    # topology must produce this exact digest.
    result["output_digest"] = _canonical_digest(
        {
            "build_key": result["build_key"],
            "tensor_digest": tensor_digest,
            "adapter_digest": adapter_digest,
            "side_pack_sha256": side_sha256,
            "rows": result["rows"],
            "games": result["games"],
            "decisions": result["decisions"],
            "scalars": _semantic_scalars(corpus),
        }
    )
    _atomic_json(output / "manifest.json", result)
    return result


def load_packed_side_tensors(
    output: Path, manifest: Optional[Mapping[str, Any]] = None
) -> Optional[dict[str, torch.Tensor]]:
    output = Path(output).resolve()
    metadata = dict(manifest or json.loads((output / "manifest.json").read_text()))
    name = metadata.get("side_pack")
    if name is None:
        return None
    path = output / str(name)
    return _load_side_fragment(path, str(metadata.get("side_pack_sha256") or ""))


def load_packed_adapter_routing(
    output: Path, manifest: Optional[Mapping[str, Any]] = None
) -> PackedAdapterRouting:
    """Load and fully validate the cached flat adapter-routing sidecar."""

    output = Path(output).resolve()
    metadata = dict(manifest or json.loads((output / "manifest.json").read_text()))
    path = output / str(metadata.get("adapter_pack") or "")
    if not path.is_file() or _sha256_file(path) != metadata.get("adapter_pack_sha256"):
        raise RuntimeError("cached packed adapter routing checksum changed")
    packed = _load_adapter_fragment(path, str(metadata["adapter_pack_sha256"]))
    if _adapter_digest(packed) != metadata.get("adapter_digest"):
        raise RuntimeError("cached packed adapter routing semantic digest changed")
    if packed.games != int(metadata.get("games", -1)):
        raise RuntimeError("cached adapter routing lost game alignment")
    return packed


def build_parallel_replay_pack(
    source_path: Path,
    output: Path,
    *,
    workers: int,
    ranges_per_worker: int = 1,
    max_in_flight: Optional[int] = None,
    verify_info_set: bool = False,
    max_context: int,
    exact_card_vocab: Optional[int] = None,
    force_strategic: bool = False,
    memory_reserve_gib: float = 4.0,
    semantic_contract: Optional[Mapping[str, Any]] = None,
) -> tuple[DeviceResidentBootstrapCorpus, dict[str, Any]]:
    """Build and cache one canonical contiguous replay pack."""

    if int(workers) not in SUPPORTED_WORKERS:
        raise ValueError(f"workers must be one of {sorted(SUPPORTED_WORKERS)}")
    source, offsets = scan_source(Path(source_path))
    semantic = {
        "environment": {name: os.environ.get(name) for name in _SEMANTIC_ENV_KEYS},
        **dict(semantic_contract or {}),
    }
    build_key = _canonical_digest(
        {
            "schema": SCHEMA,
            "source": asdict(source),
            "verify_info_set": bool(verify_info_set),
            "max_context": int(max_context),
            "exact_card_vocab": exact_card_vocab,
            "force_strategic": bool(force_strategic),
            "semantic_contract": semantic,
        }
    )
    output = Path(output).resolve()
    existing_path = output / "manifest.json"
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if existing.get("build_key") == build_key:
            pack = (output / str(existing.get("pack") or "")).resolve()
            if pack.is_file() and _sha256_file(pack) == existing.get("pack_sha256"):
                payload = torch.load(pack, map_location="cpu", weights_only=True, mmap=True)
                corpus = DeviceResidentBootstrapCorpus.from_packed_state(
                    tensors=payload["tensors"], scalars=payload["scalars"]
                )
                validate_cpu_corpus(corpus, allow_empty_training_fragment=True)
                if _tensor_digest(corpus) != existing.get("tensor_digest"):
                    raise RuntimeError("cached parallel replay tensor digest changed")
                load_packed_adapter_routing(output, existing)
                load_packed_side_tensors(output, existing)
                return corpus, {**existing, "cache_reused": True}
        raise RuntimeError("parallel replay output exists for a different or corrupt build")

    ranges = deterministic_ranges(
        source,
        offsets,
        target_ranges=max(int(workers), int(workers) * int(ranges_per_worker)),
    )
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}.{time.time_ns()}")
    spool = staging / "fragments"
    spool.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    results: dict[int, RangeResult] = {}
    limit = max(1, min(int(max_in_flight or workers), int(workers)))
    tasks = [
        (
            asdict(source),
            asdict(item),
            str(spool),
            bool(verify_info_set),
            int(max_context),
            exact_card_vocab,
            bool(force_strategic),
        )
        for item in ranges
    ]
    try:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as executor:
            pending: dict[Any, int] = {}
            next_task = 0
            while next_task < len(tasks) or pending:
                while next_task < len(tasks) and len(pending) < limit:
                    pending[executor.submit(_range_worker, tasks[next_task])] = next_task
                    next_task += 1
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    expected = pending.pop(future)
                    result = _parse_result(future.result())
                    if result.range.ordinal != expected or expected in results:
                        raise RuntimeError("parallel replay worker returned a duplicate range")
                    results[expected] = result
        if sorted(results) != list(range(len(ranges))):
            raise RuntimeError("parallel replay build lost one or more ranges")
        stat = Path(source.path).stat()
        if stat.st_size != source.size_bytes or stat.st_mtime_ns != source.mtime_ns:
            raise RuntimeError("parallel replay source changed during build")
        if _sha256_file(Path(source.path)) != source.sha256:
            raise RuntimeError("parallel replay source digest changed during build")
        canonical = [results[index] for index in range(len(ranges))]
        if [row.range for row in canonical] != ranges:
            raise RuntimeError("parallel replay range identities changed")
        descriptors = [row.fragment for row in canonical if row.fragment is not None]
        if not descriptors:
            raise RuntimeError("parallel replay source contains no trainable decisions")
        corpus = _merge_fragments(
            descriptors,
            build_seconds=time.monotonic() - started,
            memory_reserve_gib=float(memory_reserve_gib),
        )
        adapter_routing = _merge_adapter_fragments(canonical)
        if adapter_routing.games != int(corpus.train_games):
            raise RuntimeError("packed adapter routing is not aligned to replay games")
        side_parts = [
            _load_side_fragment(Path(row.side_fragment_path), row.side_fragment_sha256)
            for row in canonical
            if row.side_fragment_path
        ]
        if side_parts and len(side_parts) != len(canonical):
            raise RuntimeError("packed replay side tensors are missing one or more ranges")
        side_tensors = merge_side_tensors(side_parts) if side_parts else None
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "build_key": build_key,
            "source": asdict(source),
            "workers": int(workers),
            "ranges_per_worker": int(ranges_per_worker),
            "max_in_flight": limit,
            "ranges": [asdict(row.range) for row in canonical],
            "fragment_checksums": [
                None if row.fragment is None else row.fragment.payload_sha256
                for row in canonical
            ],
            "adapter_fragment_checksums": [
                row.adapter_fragment_sha256 for row in canonical
            ],
            "side_fragment_checksums": [
                row.side_fragment_sha256 for row in canonical if row.side_fragment_sha256
            ],
            "rows": sum(row.rows for row in canonical),
            "games": sum(row.games for row in canonical),
            "decisions": sum(row.decisions for row in canonical),
            "semantic_contract": semantic,
            "elapsed_sec": time.monotonic() - started,
            "created_at_unix": time.time(),
        }
        if manifest["rows"] != source.rows:
            raise RuntimeError("parallel replay merged row count changed")
        staging_output = staging / "output"
        persisted = _write_pack(
            staging_output, corpus, adapter_routing, manifest, side_tensors
        )
        os.replace(staging_output, output)
        shutil.rmtree(staging, ignore_errors=True)
        persisted = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        return corpus, {**persisted, "cache_reused": False}
    except BaseException:
        # Interrupted or failed builds never publish ``output``.  Preserve the
        # staging directory for diagnosis; a later build uses a fresh nonce.
        raise


def build_parallel_replay_window_pack(
    source_paths: Sequence[Path],
    output: Path,
    *,
    component_root: Path,
    workers: int = 16,
    max_context: int,
    exact_card_vocab: Optional[int] = None,
    force_strategic: bool = False,
    memory_reserve_gib: float = 4.0,
    semantic_contract: Optional[Mapping[str, Any]] = None,
) -> tuple[DeviceResidentBootstrapCorpus, PackedAdapterRouting, dict[str, Any]]:
    """Build/reuse shard packs, then merge an accumulated RL window canonically."""

    sources = [Path(path).resolve() for path in source_paths]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("RL replay window requires unique ordered source shards")
    if int(workers) not in SUPPORTED_WORKERS:
        raise ValueError(f"workers must be one of {sorted(SUPPORTED_WORKERS)}")
    output = Path(output).resolve()
    component_root = Path(component_root).resolve()
    identities = [scan_source(path)[0] for path in sources]
    window_contract = {
        "schema": "poke_bot.parallel_replay_window/v1",
        "sources": [asdict(identity) for identity in identities],
        "max_context": int(max_context),
        "exact_card_vocab": exact_card_vocab,
        "force_strategic": bool(force_strategic),
        "semantic_contract": dict(semantic_contract or {}),
    }
    build_key = _canonical_digest(window_contract)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("build_key") != build_key:
            raise RuntimeError("RL replay window output belongs to a different source set")
        pack_path = output / str(manifest.get("pack") or "")
        if not pack_path.is_file() or _sha256_file(pack_path) != manifest.get("pack_sha256"):
            raise RuntimeError("cached RL replay window pack checksum changed")
        payload = torch.load(pack_path, map_location="cpu", weights_only=True, mmap=True)
        corpus = DeviceResidentBootstrapCorpus.from_packed_state(
            tensors=payload["tensors"], scalars=payload["scalars"]
        )
        validate_cpu_corpus(corpus, allow_empty_training_fragment=True)
        if _tensor_digest(corpus) != manifest.get("tensor_digest"):
            raise RuntimeError("cached RL replay window tensor digest changed")
        routing = load_packed_adapter_routing(output, manifest)
        return corpus, routing, {**manifest, "cache_reused": True}

    nonce = f"{os.getpid()}.{time.time_ns()}"
    staging = output.with_name(f".{output.name}.staging.{nonce}")
    spool = staging / "fragments"
    spool.mkdir(parents=True, exist_ok=False)
    component_manifests: list[dict[str, Any]] = []
    descriptors: list[FragmentDescriptor] = []
    routing_results: list[RangeResult] = []
    window_side_parts: list[dict[str, torch.Tensor]] = []
    try:
        for source_index, (source, identity) in enumerate(zip(sources, identities)):
            component = component_root / identity.sha256.removeprefix("sha256:")
            corpus, component_manifest = build_parallel_replay_pack(
                source,
                component,
                workers=int(workers),
                max_context=int(max_context),
                exact_card_vocab=exact_card_vocab,
                force_strategic=bool(force_strategic),
                memory_reserve_gib=float(memory_reserve_gib),
                semantic_contract={
                    **dict(semantic_contract or {}),
                    "window_component": True,
                },
            )
            component_manifests.append(component_manifest)
            descriptors.append(
                _write_fragment(
                    spool,
                    ordinal=source_index,
                    partition="train",
                    corpus=corpus,
                )
            )
            routing = load_packed_adapter_routing(component, component_manifest)
            component_side = load_packed_side_tensors(component, component_manifest)
            if component_side is not None:
                window_side_parts.append(component_side)
            route_path = spool / f"adapter-{source_index:06d}.pt"
            with route_path.open("xb") as stream:
                torch.save(
                    {"schema": ADAPTER_PACK_SCHEMA, "tensors": routing.tensor_state()},
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
            routing_results.append(
                RangeResult(
                    range=RangeIdentity(
                        ordinal=source_index,
                        row_start=0,
                        row_end=int(identity.rows),
                        byte_start=0,
                        byte_end=int(identity.size_bytes),
                        identity=_canonical_digest(
                            {"source_index": source_index, "source": asdict(identity)}
                        ),
                    ),
                    fragment=None,
                    rows=int(identity.rows),
                    games=int(routing.games),
                    decisions=int(corpus.decisions),
                    source_sha256=identity.sha256,
                    elapsed_sec=0.0,
                    adapter_fragment_path=str(route_path),
                    adapter_fragment_sha256=_sha256_file(route_path),
                    side_fragment_path="",
                    side_fragment_sha256="",
                )
            )
        merged = _merge_fragments(
            descriptors,
            build_seconds=sum(float(row.get("elapsed_sec", 0.0)) for row in component_manifests),
            memory_reserve_gib=float(memory_reserve_gib),
        )
        routing = _merge_adapter_fragments(routing_results)
        if routing.games != int(merged.train_games):
            raise RuntimeError("RL replay window adapter metadata lost alignment")
        if window_side_parts and len(window_side_parts) != len(sources):
            raise RuntimeError("RL replay window side tensors are incomplete")
        window_side = (
            merge_side_tensors(window_side_parts) if window_side_parts else None
        )
        manifest = {
            "schema": "poke_bot.parallel_replay_window/v1",
            "status": "complete",
            "build_key": build_key,
            "sources": [asdict(identity) for identity in identities],
            "component_manifests": [
                {
                    "path": str(component_root / identity.sha256.removeprefix("sha256:")),
                    "source_sha256": identity.sha256,
                    "output_digest": component["output_digest"],
                }
                for identity, component in zip(identities, component_manifests)
            ],
            "workers": int(workers),
            "rows": sum(identity.rows for identity in identities),
            "games": int(merged.train_games),
            "decisions": int(merged.decisions),
            "source_game_offsets": [
                0,
                *list(
                    __import__("itertools").accumulate(
                        int(row.games) for row in routing_results
                    )
                ),
            ],
            "semantic_contract": dict(semantic_contract or {}),
            "created_at_unix": time.time(),
        }
        staging_output = staging / "output"
        persisted = _write_pack(
            staging_output, merged, routing, manifest, window_side
        )
        os.replace(staging_output, output)
        shutil.rmtree(staging, ignore_errors=True)
        return merged, routing, {**persisted, "cache_reused": False}
    except BaseException:
        raise


def pin_cpu_corpus(corpus: DeviceResidentBootstrapCorpus) -> DeviceResidentBootstrapCorpus:
    """Return the same packed corpus in page-locked host tensors."""

    if not torch.cuda.is_available():
        raise ParallelReplayUnavailable("CUDA pin-memory support is unavailable")
    tensors = {name: value.pin_memory() for name, value in corpus.tensor_state().items()}
    return DeviceResidentBootstrapCorpus.from_packed_state(
        tensors=tensors, scalars=corpus.scalar_state()
    )


def validate_corpus_parity(
    serial: DeviceResidentBootstrapCorpus,
    parallel: DeviceResidentBootstrapCorpus,
    *,
    floating_atol: float = 0.0,
) -> dict[str, Any]:
    """Fail closed unless packed serial and parallel loss inputs agree."""

    if _semantic_scalars(serial) != _semantic_scalars(parallel):
        raise RuntimeError("serial/parallel replay scalar states differ")
    left = serial.tensor_state()
    right = parallel.tensor_state()
    if set(left) != set(right):
        raise RuntimeError("serial/parallel replay tensor inventories differ")
    exact = []
    tolerated = []
    for name in sorted(left):
        a = left[name].detach().cpu().contiguous()
        b = right[name].detach().cpu().contiguous()
        if a.dtype != b.dtype or a.shape != b.shape:
            raise RuntimeError(f"serial/parallel tensor shape differs: {name}")
        if torch.equal(a, b):
            exact.append(name)
            continue
        if not (a.is_floating_point() and float(floating_atol) > 0.0):
            raise RuntimeError(f"serial/parallel tensor differs: {name}")
        if not torch.allclose(a, b, rtol=0.0, atol=float(floating_atol)):
            raise RuntimeError(f"serial/parallel floating tensor exceeds tolerance: {name}")
        tolerated.append(name)
    return {
        "passed": True,
        "exact_tensors": exact,
        "tolerated_tensors": tolerated,
        "floating_atol": float(floating_atol),
        "serial_digest": _tensor_digest(serial),
        "parallel_digest": _tensor_digest(parallel),
    }


def validate_one_step_result(
    serial_result: Mapping[str, torch.Tensor],
    parallel_result: Mapping[str, torch.Tensor],
    *,
    floating_atol: float = 0.0,
) -> dict[str, Any]:
    """Compare model/optimizer tensors after one identically seeded step."""

    if set(serial_result) != set(parallel_result):
        raise RuntimeError("one-step optimizer result inventories differ")
    max_abs = 0.0
    for name in sorted(serial_result):
        left = serial_result[name].detach().cpu()
        right = parallel_result[name].detach().cpu()
        if left.dtype != right.dtype or left.shape != right.shape:
            raise RuntimeError(f"one-step optimizer shape differs: {name}")
        if torch.equal(left, right):
            continue
        if not left.is_floating_point():
            raise RuntimeError(f"one-step integer optimizer tensor differs: {name}")
        delta = float((left - right).abs().max().item())
        max_abs = max(max_abs, delta)
        if delta > float(floating_atol):
            raise RuntimeError(
                f"one-step optimizer result differs: {name} max_abs={delta}"
            )
    return {"passed": True, "max_abs": max_abs, "floating_atol": float(floating_atol)}


def prepare_with_serial_fallback(
    parallel: Callable[[], Any],
    serial: Callable[[], Any],
    *,
    strict_parallel: bool,
) -> tuple[Any, str]:
    """Select the opt-in backend while retaining the serial fail-safe."""

    try:
        return parallel(), "parallel"
    except (ParallelReplayUnavailable, ImportError, OSError):
        if strict_parallel:
            raise
        return serial(), "serial_fallback"


def write_validation_receipt(
    path: Path,
    *,
    source: Mapping[str, Any],
    code_sha256: str,
    worker_count: int,
    output_digest: str,
    counts: Mapping[str, int],
    timing: Mapping[str, float],
    memory: Mapping[str, int],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "source": dict(source),
        "code_sha256": str(code_sha256),
        "worker_count": int(worker_count),
        "output_digest": str(output_digest),
        "counts": {key: int(value) for key, value in counts.items()},
        "timing": {key: float(value) for key, value in timing.items()},
        "memory": {key: int(value) for key, value in memory.items()},
        "validation": dict(validation),
        "created_at_unix": time.time(),
    }
    payload["receipt_sha256"] = _canonical_digest(payload)
    _atomic_json(Path(path), payload)
    return payload
