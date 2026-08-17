"""Opaque, fail-closed bindings for the private r207 V3 successor arena.

The module is intentionally separate from the normal ``cg`` interface and
from the r198 snapshot ABI. Python receives opaque arena/node capabilities
only; it has no state, random-generator, replay, or native planner entry
points. A reviewed build may return only a dynamically provenance-clean exact
successor, an exact terminal result, or the two outcomes of one simple fair
coin; every other transition remains audit-required.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

SUCCESS = 0
INVALID_ARGUMENT = 1
UNKNOWN_ARENA = 2
UNKNOWN_NODE = 3
NOT_LIVE_NODE = 4
WRONG_BOUNDARY = 5
INVALID_ACTION = 6
AUDIT_REQUIRED = 7
BUFFER_TOO_SMALL = 8
INITIALIZATION_FAILED = 9

BOUNDARY_CONTROLLED_SELECTION = 1
BOUNDARY_OPPONENT_SELECTION = 2
BOUNDARY_TERMINAL = 3
BOUNDARY_UNSUPPORTED = 4
BOUNDARY_FINITE_CHANCE = 5

CHANCE_OUTCOME_HEADS = 1
CHANCE_OUTCOME_TAILS = 2

RTP_PLANNER_V3_ABI_NAME = "poke_bot.r207_v3.successor_arena"
RTP_PLANNER_V3_ABI_VERSION = 4
RTP_PLANNER_V3_DECK_CARD_COUNT = 120

EXPECTED_V3_EXPORTS = frozenset(
    {
        "RtpPlannerV3AbiVersion",
        "RtpPlannerV3LastError",
        "RtpPlannerV3Initialize",
        "RtpPlannerV3ArenaStart",
        "RtpPlannerV3ArenaDestroy",
        "RtpPlannerV3CaptureLive",
        "RtpPlannerV3CloneNode",
        "RtpPlannerV3RestoreLive",
        "RtpPlannerV3ReleaseNode",
        "RtpPlannerV3NodeInfo",
        "RtpPlannerV3PolicyJsonSize",
        "RtpPlannerV3PolicyJson",
        "RtpPlannerV3ExpandAction",
        "RtpPlannerV3ChanceOutcome",
    }
)

# These are interfaces that would create an accidental bridge to the public
# engine, expose a private-byte route, or invoke the old native planner.
FORBIDDEN_V3_EXPORTS = frozenset(
    {
        "AgentStart",
        "BattleFinish",
        "BattleStart",
        "BattleStartBatchSeeded",
        "BattleStartSeeded",
        "GameInitialize",
        "GetBattleData",
        "GetBattleDataBatch",
        "RtpPairingBattleStartSeededOut",
        "RtpPairingSnapshotCapture",
        "RtpPairingSnapshotFingerprint",
        "RtpPairingSnapshotFingerprintSize",
        "RtpPairingSnapshotGetBattleJsonOut",
        "RtpPairingSnapshotInitialize",
        "RtpPairingSnapshotRelease",
        "RtpPairingSnapshotRestore",
        "RtpPairingSnapshotRestoreSerialized",
        "RtpPairingSnapshotSerialize",
        "RtpPairingSnapshotSerializedSize",
        "SearchBegin",
        "SearchEnd",
        "SearchStep",
        "Select",
        "SetBattleData",
        "StepBatch",
        "VisualizeData",
    }
)


class RtpPlannerSuccessorV3Error(RuntimeError):
    """The private successor arena is unavailable or failed closed."""


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def abi_contract() -> dict[str, Any]:
    """Return the complete, content-addressed public shape of the V3 ABI."""

    return {
        "name": RTP_PLANNER_V3_ABI_NAME,
        "version": RTP_PLANNER_V3_ABI_VERSION,
        "fresh_native_worker_required": True,
        "handles": {
            "arena": "uint64 capability",
            "node": "uint64 capability",
            "no_native_pointer_round_trip": True,
        },
        "operations": {
            "initialize": "RtpPlannerV3Initialize",
            "start": "RtpPlannerV3ArenaStart",
            "destroy": "RtpPlannerV3ArenaDestroy",
            "capture_live": "RtpPlannerV3CaptureLive",
            "clone": "RtpPlannerV3CloneNode",
            "restore_live": "RtpPlannerV3RestoreLive",
            "release": "RtpPlannerV3ReleaseNode",
            "node_info": "RtpPlannerV3NodeInfo",
            "policy_json_size": "RtpPlannerV3PolicyJsonSize",
            "policy_json": "RtpPlannerV3PolicyJson",
            "expand": "RtpPlannerV3ExpandAction",
            "chance_outcome": "RtpPlannerV3ChanceOutcome",
        },
        "policy_visibility": "controlled-selection-only",
        "terminal_visibility": "result-only-through-node-info",
        "transition_policy": "dynamic-provenance-clean-exact-only; default-deny-audit-required",
        "chance_policy": "one-explicitly-armed-single-fair-coin-only",
        "forbids": {
            "public_engine_bridge": True,
            "private_byte_export": True,
            "legacy_native_planner": True,
        },
        "expected_exports": sorted(EXPECTED_V3_EXPORTS),
        "forbidden_exports": sorted(FORBIDDEN_V3_EXPORTS),
    }


def abi_sha256() -> str:
    return canonical_digest(abi_contract())


def validate_native_exports(symbols: Iterable[str]) -> frozenset[str]:
    """Fail unless native exports are exactly the V3 surface plus runtimes.

    Toolchains may publish non-``Rtp`` runtime helpers.  Any additional
    project-named symbol is rejected, and known public/native-planner symbols
    are always rejected regardless of prefix.
    """

    exported = frozenset(str(symbol) for symbol in symbols if str(symbol))
    missing = EXPECTED_V3_EXPORTS - exported
    if missing:
        raise RtpPlannerSuccessorV3Error(
            "V3 native library is missing required exports: " + ", ".join(sorted(missing))
        )
    forbidden = FORBIDDEN_V3_EXPORTS & exported
    if forbidden:
        raise RtpPlannerSuccessorV3Error(
            "V3 native library exports forbidden symbols: "
            + ", ".join(sorted(forbidden))
        )
    unexpected = sorted(
        symbol
        for symbol in exported - EXPECTED_V3_EXPORTS
        if symbol.startswith(("Rtp", "rtp"))
    )
    if unexpected:
        raise RtpPlannerSuccessorV3Error(
            "V3 native library exports an unreviewed project symbol: "
            + ", ".join(unexpected)
        )
    return exported


def _absolute_without_resolving(path: str | Path) -> Path:
    raw = os.path.expanduser(os.fspath(path))
    return Path(raw if os.path.isabs(raw) else os.path.join(os.getcwd(), raw))


def _regular_file_without_symlinks(path: str | Path, *, label: str) -> Path:
    lexical = _absolute_without_resolving(path)
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        if part in ("", "."):
            continue
        if part == "..":
            current = current.parent
            continue
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RtpPlannerSuccessorV3Error(
                f"cannot inspect {label}: {current}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RtpPlannerSuccessorV3Error(f"{label} traverses a symlink: {current}")
    try:
        metadata = os.stat(lexical, follow_symlinks=False)
    except OSError as exc:
        raise RtpPlannerSuccessorV3Error(f"cannot read {label}: {lexical}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RtpPlannerSuccessorV3Error(f"{label} is not a regular file: {lexical}")
    return lexical.resolve(strict=True)


def exported_symbols(library_path: str | Path) -> frozenset[str]:
    """Read dynamic defined symbols without loading the private library."""

    source = _regular_file_without_symlinks(library_path, label="V3 library")
    commands = (("nm", "-D", "--defined-only", str(source)), ("nm", "-gU", str(source)))
    failures: list[str] = []
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            failures.append(f"{' '.join(command[:3])}: {exc}")
            continue
        result: set[str] = set()
        for line in completed.stdout.splitlines():
            fields = line.split()
            if fields:
                result.add(fields[-1])
        return frozenset(result)
    raise RtpPlannerSuccessorV3Error(
        "cannot inspect V3 library exports: " + "; ".join(failures)
    )


class _NativeNodeInfo(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("boundary", ctypes.c_uint32),
        ("select_player", ctypes.c_int32),
        ("terminal_result", ctypes.c_int32),
        ("option_count", ctypes.c_uint32),
        ("generation", ctypes.c_uint64),
    ]


class _NativeTransition(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("status", ctypes.c_int32),
        ("boundary", ctypes.c_uint32),
        ("select_player", ctypes.c_int32),
        ("terminal_result", ctypes.c_int32),
        ("child_node", ctypes.c_uint64),
        ("chance_set", ctypes.c_uint64),
        ("chance_outcome_count", ctypes.c_uint32),
    ]


class _NativeChanceOutcome(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("label", ctypes.c_uint32),
        ("probability_numerator", ctypes.c_uint32),
        ("probability_denominator", ctypes.c_uint32),
        ("child_node", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class NodeInfo:
    boundary: int
    select_player: int
    terminal_result: int
    option_count: int
    generation: int


@dataclass(frozen=True)
class TransitionResult:
    status: int
    boundary: int
    select_player: int
    terminal_result: int
    child_node: int
    chance_set: int
    chance_outcome_count: int


class OpaqueNode:
    """A native node capability; its numeric token is never a native pointer."""

    __slots__ = ("_arena", "_token")

    def __init__(self, arena: OpaqueArena, token: int) -> None:
        self._arena = arena
        self._token = int(token)

    def info(self) -> NodeInfo:
        return self._arena.node_info(self)

    def policy_json(self) -> str:
        return self._arena.policy_json(self)


@dataclass(frozen=True)
class ChanceOutcome:
    """One opaque child of an exactly enumerated finite chance transition."""

    label: int
    probability_numerator: int
    probability_denominator: int
    child: OpaqueNode


class OpaqueArena:
    """An arena capability with no route to engine structs or private bytes."""

    __slots__ = ("_closed", "_engine", "_token")

    def __init__(self, engine: RtpPlannerSuccessorV3Engine, token: int) -> None:
        self._engine = engine
        self._token = int(token)
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RtpPlannerSuccessorV3Error("V3 arena is closed")

    def _check_node(self, node: OpaqueNode) -> int:
        self._require_open()
        if not isinstance(node, OpaqueNode) or node._arena is not self:
            raise RtpPlannerSuccessorV3Error("node does not belong to this V3 arena")
        return node._token

    def close(self) -> None:
        if not self._closed:
            self._engine._destroy_arena(self._token)
            self._closed = True

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def capture_live(self) -> OpaqueNode:
        self._require_open()
        return OpaqueNode(self, self._engine._capture_live(self._token))

    def clone(self, node: OpaqueNode) -> OpaqueNode:
        return OpaqueNode(self, self._engine._clone_node(self._token, self._check_node(node)))

    def restore_live(self, node: OpaqueNode) -> OpaqueNode:
        return OpaqueNode(
            self, self._engine._restore_live(self._token, self._check_node(node))
        )

    def release(self, node: OpaqueNode) -> None:
        self._engine._release_node(self._token, self._check_node(node))

    def node_info(self, node: OpaqueNode) -> NodeInfo:
        return self._engine._node_info(self._token, self._check_node(node))

    def policy_json(self, node: OpaqueNode) -> str:
        return self._engine._policy_json(self._token, self._check_node(node))

    def expand(self, node: OpaqueNode, selected: Sequence[int]) -> TransitionResult:
        return self._engine._expand(self._token, self._check_node(node), selected)

    def chance_outcome(self, chance_set: int, outcome_index: int) -> ChanceOutcome:
        self._require_open()
        if int(chance_set) <= 0:
            raise RtpPlannerSuccessorV3Error("V3 chance-set capability must be positive")
        if int(outcome_index) < 0:
            raise RtpPlannerSuccessorV3Error("V3 chance outcome index must be non-negative")
        label, numerator, denominator, child = self._engine._chance_outcome(
            self._token, int(chance_set), int(outcome_index)
        )
        return ChanceOutcome(
            label=label,
            probability_numerator=numerator,
            probability_denominator=denominator,
            child=OpaqueNode(self, child),
        )


class RtpPlannerSuccessorV3Engine:
    """A ctypes façade for the isolated native V3 build only."""

    def __init__(
        self,
        library_path: str | Path,
        *,
        library: Any | None = None,
        initialize: bool = True,
        inspect_static_exports: bool = True,
    ) -> None:
        lexical_library = _absolute_without_resolving(library_path)
        if ".private" not in lexical_library.parts:
            raise RtpPlannerSuccessorV3Error(
                "V3 library must be beneath a literal .private directory"
            )
        self.library_path = _regular_file_without_symlinks(
            library_path, label="V3 library"
        )
        if ".private" not in self.library_path.parts:
            raise RtpPlannerSuccessorV3Error("V3 library resolved outside .private")
        if inspect_static_exports:
            validate_native_exports(exported_symbols(self.library_path))
        self._library = library if library is not None else ctypes.CDLL(str(self.library_path))
        self._configure_functions()
        version = int(self._library.RtpPlannerV3AbiVersion())
        if version != RTP_PLANNER_V3_ABI_VERSION:
            raise RtpPlannerSuccessorV3Error(
                f"V3 ABI version mismatch: expected {RTP_PLANNER_V3_ABI_VERSION}, got {version}"
            )
        if initialize:
            self._expect(self._library.RtpPlannerV3Initialize(), "initialize")

    def _configure_functions(self) -> None:
        required = sorted(EXPECTED_V3_EXPORTS)
        missing = [name for name in required if not hasattr(self._library, name)]
        if missing:
            raise RtpPlannerSuccessorV3Error(
                "loaded V3 library is missing symbols: " + ", ".join(missing)
            )
        self._library.RtpPlannerV3AbiVersion.restype = ctypes.c_uint32
        self._library.RtpPlannerV3LastError.restype = ctypes.c_char_p
        self._library.RtpPlannerV3Initialize.restype = ctypes.c_int
        self._library.RtpPlannerV3Initialize.argtypes = []
        self._library.RtpPlannerV3ArenaStart.restype = ctypes.c_int
        self._library.RtpPlannerV3ArenaStart.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._library.RtpPlannerV3ArenaDestroy.restype = ctypes.c_int
        self._library.RtpPlannerV3ArenaDestroy.argtypes = [ctypes.c_uint64]
        self._library.RtpPlannerV3CaptureLive.restype = ctypes.c_int
        self._library.RtpPlannerV3CaptureLive.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._library.RtpPlannerV3CloneNode.restype = ctypes.c_int
        self._library.RtpPlannerV3CloneNode.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._library.RtpPlannerV3RestoreLive.restype = ctypes.c_int
        self._library.RtpPlannerV3RestoreLive.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._library.RtpPlannerV3ReleaseNode.restype = ctypes.c_int
        self._library.RtpPlannerV3ReleaseNode.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        self._library.RtpPlannerV3NodeInfo.restype = ctypes.c_int
        self._library.RtpPlannerV3NodeInfo.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(_NativeNodeInfo),
        ]
        self._library.RtpPlannerV3PolicyJsonSize.restype = ctypes.c_int
        self._library.RtpPlannerV3PolicyJsonSize.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._library.RtpPlannerV3PolicyJson.restype = ctypes.c_int
        self._library.RtpPlannerV3PolicyJson.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._library.RtpPlannerV3ExpandAction.restype = ctypes.c_int
        self._library.RtpPlannerV3ExpandAction.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
            ctypes.POINTER(_NativeTransition),
        ]
        self._library.RtpPlannerV3ChanceOutcome.restype = ctypes.c_int
        self._library.RtpPlannerV3ChanceOutcome.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.POINTER(_NativeChanceOutcome),
        ]

    def _last_error(self) -> str:
        value = self._library.RtpPlannerV3LastError()
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _expect(self, status: int, operation: str) -> None:
        if int(status) != SUCCESS:
            detail = self._last_error()
            suffix = f": {detail}" if detail else ""
            raise RtpPlannerSuccessorV3Error(
                f"V3 {operation} failed with status {int(status)}{suffix}"
            )

    def start(
        self, cards: Sequence[int], *, seed: int, controlled_player: int
    ) -> tuple[OpaqueArena, OpaqueNode]:
        if len(cards) != RTP_PLANNER_V3_DECK_CARD_COUNT:
            raise RtpPlannerSuccessorV3Error(
                f"V3 start requires {RTP_PLANNER_V3_DECK_CARD_COUNT} cards"
            )
        if not 0 < int(seed) <= 0xFFFFFFFF:
            raise RtpPlannerSuccessorV3Error("V3 seed must be a non-zero uint32")
        if controlled_player not in (0, 1):
            raise RtpPlannerSuccessorV3Error("controlled_player must be 0 or 1")
        try:
            native_cards = (ctypes.c_int * len(cards))(*(int(card) for card in cards))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RtpPlannerSuccessorV3Error("cards must be signed 32-bit integers") from exc
        arena = ctypes.c_uint64()
        node = ctypes.c_uint64()
        self._expect(
            self._library.RtpPlannerV3ArenaStart(
                native_cards,
                len(cards),
                ctypes.c_uint32(seed),
                controlled_player,
                ctypes.byref(arena),
                ctypes.byref(node),
            ),
            "start",
        )
        opaque_arena = OpaqueArena(self, arena.value)
        return opaque_arena, OpaqueNode(opaque_arena, node.value)

    def _destroy_arena(self, arena: int) -> None:
        self._expect(self._library.RtpPlannerV3ArenaDestroy(arena), "destroy arena")

    def _handle_output(self, operation: str, function: Any, *arguments: Any) -> int:
        output = ctypes.c_uint64()
        self._expect(function(*arguments, ctypes.byref(output)), operation)
        if output.value == 0:
            raise RtpPlannerSuccessorV3Error(f"V3 {operation} returned a zero capability")
        return int(output.value)

    def _capture_live(self, arena: int) -> int:
        return self._handle_output("capture live", self._library.RtpPlannerV3CaptureLive, arena)

    def _clone_node(self, arena: int, node: int) -> int:
        return self._handle_output("clone node", self._library.RtpPlannerV3CloneNode, arena, node)

    def _restore_live(self, arena: int, node: int) -> int:
        return self._handle_output("restore live", self._library.RtpPlannerV3RestoreLive, arena, node)

    def _release_node(self, arena: int, node: int) -> None:
        self._expect(self._library.RtpPlannerV3ReleaseNode(arena, node), "release node")

    def _node_info(self, arena: int, node: int) -> NodeInfo:
        native = _NativeNodeInfo()
        native.struct_size = ctypes.sizeof(_NativeNodeInfo)
        self._expect(
            self._library.RtpPlannerV3NodeInfo(arena, node, ctypes.byref(native)),
            "node info",
        )
        if native.struct_size != ctypes.sizeof(_NativeNodeInfo):
            raise RtpPlannerSuccessorV3Error("V3 node-info ABI size mismatch")
        return NodeInfo(
            boundary=int(native.boundary),
            select_player=int(native.select_player),
            terminal_result=int(native.terminal_result),
            option_count=int(native.option_count),
            generation=int(native.generation),
        )

    def _policy_json(self, arena: int, node: int) -> str:
        size = ctypes.c_size_t()
        self._expect(
            self._library.RtpPlannerV3PolicyJsonSize(arena, node, ctypes.byref(size)),
            "policy JSON size",
        )
        if size.value > 64 * 1024 * 1024:
            raise RtpPlannerSuccessorV3Error("V3 policy JSON exceeds the safety limit")
        output = ctypes.create_string_buffer(size.value + 1)
        copied = ctypes.c_size_t()
        self._expect(
            self._library.RtpPlannerV3PolicyJson(
                arena, node, output, len(output), ctypes.byref(copied)
            ),
            "policy JSON",
        )
        if copied.value != size.value:
            raise RtpPlannerSuccessorV3Error("V3 policy JSON size changed during copy")
        try:
            return output.raw[: copied.value].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RtpPlannerSuccessorV3Error("V3 policy JSON is not UTF-8") from exc

    def _expand(self, arena: int, node: int, selected: Sequence[int]) -> TransitionResult:
        if len(selected) > 256:
            raise RtpPlannerSuccessorV3Error("V3 action has too many selection items")
        try:
            native_selected = (ctypes.c_int * len(selected))(
                *(int(item) for item in selected)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise RtpPlannerSuccessorV3Error(
                "V3 action items must be signed 32-bit integers"
            ) from exc
        native = _NativeTransition()
        native.struct_size = ctypes.sizeof(_NativeTransition)
        status = int(
            self._library.RtpPlannerV3ExpandAction(
                arena,
                node,
                native_selected if selected else None,
                len(selected),
                ctypes.byref(native),
            )
        )
        if status not in (SUCCESS, INVALID_ACTION, WRONG_BOUNDARY, AUDIT_REQUIRED):
            self._expect(status, "expand action")
        if native.struct_size != ctypes.sizeof(_NativeTransition):
            raise RtpPlannerSuccessorV3Error("V3 transition ABI size mismatch")
        if int(native.status) != status:
            raise RtpPlannerSuccessorV3Error("V3 transition status disagrees with return status")
        if status == SUCCESS and int(native.boundary) == BOUNDARY_FINITE_CHANCE:
            if (
                native.child_node != 0
                or native.chance_set == 0
                or native.chance_outcome_count != 2
            ):
                raise RtpPlannerSuccessorV3Error(
                    "V3 finite-chance transition has an invalid capability shape"
                )
        elif status == SUCCESS:
            if (
                native.child_node == 0
                or native.chance_set != 0
                or native.chance_outcome_count != 0
            ):
                raise RtpPlannerSuccessorV3Error(
                    "V3 exact transition has an invalid capability shape"
                )
        elif (
            native.child_node != 0
            or native.chance_set != 0
            or native.chance_outcome_count != 0
        ):
            raise RtpPlannerSuccessorV3Error(
                "V3 rejected transition returned a child capability"
            )
        return TransitionResult(
            status=status,
            boundary=int(native.boundary),
            select_player=int(native.select_player),
            terminal_result=int(native.terminal_result),
            child_node=int(native.child_node),
            chance_set=int(native.chance_set),
            chance_outcome_count=int(native.chance_outcome_count),
        )

    def _chance_outcome(
        self, arena: int, chance_set: int, outcome_index: int
    ) -> tuple[int, int, int, int]:
        native = _NativeChanceOutcome()
        native.struct_size = ctypes.sizeof(_NativeChanceOutcome)
        self._expect(
            self._library.RtpPlannerV3ChanceOutcome(
                arena,
                chance_set,
                outcome_index,
                ctypes.byref(native),
            ),
            "chance outcome",
        )
        if native.struct_size != ctypes.sizeof(_NativeChanceOutcome):
            raise RtpPlannerSuccessorV3Error("V3 chance-outcome ABI size mismatch")
        if native.label not in (CHANCE_OUTCOME_HEADS, CHANCE_OUTCOME_TAILS):
            raise RtpPlannerSuccessorV3Error("V3 chance outcome has an invalid label")
        if native.probability_numerator != 1 or native.probability_denominator != 2:
            raise RtpPlannerSuccessorV3Error("V3 chance outcome is not an exact fair coin")
        if native.child_node == 0:
            raise RtpPlannerSuccessorV3Error("V3 chance outcome returned a zero child")
        return (
            int(native.label),
            int(native.probability_numerator),
            int(native.probability_denominator),
            int(native.child_node),
        )
