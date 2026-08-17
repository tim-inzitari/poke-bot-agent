"""Sealed Kaggle Environments libcg shared-tree action authority for r228.

This is intentionally a small viability runtime.  One competition process owns
one frozen model and one shared search tree per decision.  The configured
thread-affine ``AgentStart`` arenas advance independent simulator states and
feed ready leaves to the same model-backed coordinator queue.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from .r228_async_shared_tree_queue import (
    AsyncDirectFallbackRequired,
    DecodedLeaf,
    PersistentAsyncSharedTreeMCTS,
    PROVEN_TERMINAL_WIN_PROOF_KIND,
    PROVEN_TERMINAL_WIN_REVISION,
    PROVEN_TERMINAL_WIN_STOP_REASON,
)

SCHEMA = "poke_bot.r238_two_lane_kaggle_viability/v1"
DECISION_PREFIX = "R238_TWO_LANE_BOUNDED_MCTS_DECISION"
# This is deliberately explicit at every r228 complete-action materialization
# site.  The staged r228 package preserves the frozen r195 feature module, so
# changing a workspace-global default does not change the submitted runtime.
R228_COMPLETE_ACTION_CAP = 65_536
# Kaggle Phase1 has exactly two vCPUs.  This remains an explicit, fixed
# contract rather than an environment-dependent worker-count guess.
R228_SIMULATOR_LANE_COUNT = 2
R238_SEARCH_SECONDS_ENV = "POKEBOT_R238_SEARCH_SECONDS"
R238_DEFAULT_SEARCH_SECONDS = 2.0
R238_MINIMUM_BACKUPS_BEFORE_STABILITY = 8
R238_STABLE_ROOT_LEADER_OBSERVATIONS = 3
R238_MAXIMUM_BACKUPS_PER_DECISION = 32
KAGGLE_ENVIRONMENTS_VERSION = "1.32.6"
KAGGLE_ENVIRONMENTS_WHEEL_SHA256 = (
    "e70a7d7765b16deb1fcfa00532eb5197f28bc9fbfa07a0eee150a17d67bd77ab"
)
KAGGLE_ENVIRONMENTS_NATIVE_LIBRARY_UPDATE_COMMIT = (
    "03ab2cc235b719e5a3bd0d19e2d2c62c65a4c303"
)
# Revision 236 binds one complete official Kaggle Environments 1.32.6 native
# set.  A newly packaged runtime may not accept a historical member or mix
# members from the old and new sets.
STOCK_LIBRARY_SHA256 = {
    "libcg.so": "d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
    "libcg-arm64.so": "1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2",
    "libcg.dylib": "7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30",
    "cg.dll": "eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771",
}
STOCK_LIBRARY_BYTES = {
    "libcg.so": 1_342_400,
    "libcg-arm64.so": 1_296_464,
    "libcg.dylib": 1_245_544,
    "cg.dll": 1_525_248,
}
STOCK_LIBRARY_PLATFORM_MEMBERS = {
    ("linux", "x86_64"): "libcg.so",
    ("linux", "amd64"): "libcg.so",
    ("linux", "aarch64"): "libcg-arm64.so",
    ("linux", "arm64"): "libcg-arm64.so",
    ("darwin", "arm64"): "libcg.dylib",
    ("darwin", "aarch64"): "libcg.dylib",
    ("windows", "amd64"): "cg.dll",
    ("windows", "x86_64"): "cg.dll",
}
REQUIRED_NATIVE_EXPORTS = (
    "GameInitialize",
    "BattleStart",
    "BattleFinish",
    "GetBattleData",
    "Select",
    "VisualizeData",
    "AgentStart",
    "SearchBegin",
    "SearchStep",
    "SearchRelease",
    "SearchEnd",
    "AllCard",
    "AllAttack",
)
REQUIRED_R225_API_CALLABLES = (
    "json_to_dataclass",
    "to_observation_class",
)
REQUIRED_R225_API_TYPES = ("ApiResult",)
R238_NORMAL_STOP_REASONS = frozenset(
    {
        "stable_root_leader",
        "maximum_backups",
        "decision_deadline",
        "tree_exhausted",
        PROVEN_TERMINAL_WIN_STOP_REASON,
    }
)


class R228GameplayError(RuntimeError):
    """The shared-tree action could not be proven legal and backed."""


@dataclass(frozen=True)
class _Frontier:
    lane_id: int
    raw: dict[str, Any]


def _raw(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        result = asdict(value)
        if isinstance(result, dict):
            return result
    raise R228GameplayError(f"unsupported simulator observation: {type(value).__name__}")


def canonical_observation_fingerprint(raw: Mapping[str, Any]) -> str:
    """Hash the complete raw observation with one lane-independent encoding.

    This is intentionally separate from ``_state_key``: native lanes need
    lane-scoped tree keys, while a parent may reuse a continuation only when
    the exact full public observation matches.  Do not add ``default=str`` or
    any projection here—an observation that is not JSON-native cannot prove a
    deterministic continuation and must fail closed.
    """

    if not isinstance(raw, Mapping):
        raise R228GameplayError("observation fingerprint input is not a mapping")
    try:
        canonical = json.dumps(
            dict(raw),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R228GameplayError(
            "observation is not canonical JSON for continuation fingerprint"
        ) from exc
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def legal_order_fingerprint(
    legal_actions: Sequence[Sequence[int]],
) -> str:
    """Hash the complete ordered root legal set exactly as the parent does."""

    try:
        canonical = json.dumps(
            [[int(item) for item in action] for action in legal_actions],
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R228GameplayError(
            "root legal order is not canonical JSON for its fingerprint"
        ) from exc
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _state_key(*, lane_id: int | None, raw: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"lane_id": lane_id, "observation": raw},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return "r228:" + hashlib.sha256(encoded).hexdigest()


def _chance_boundary(raw: Mapping[str, Any]) -> bool:
    selection = raw.get("select")
    return isinstance(selection, Mapping) and selection.get("context") == 46


def _positive_seconds_from_env(name: str, default: str) -> float:
    """Read a finite positive runtime deadline without silently disabling it."""

    raw = os.environ.get(name, default)
    try:
        seconds = float(raw)
    except (TypeError, ValueError) as exc:
        raise R228GameplayError(f"{name} must be a positive number") from exc
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise R228GameplayError(f"{name} must be a finite positive number")
    return seconds


def _r238_search_seconds() -> float:
    """Read only the r238 child-search deadline; legacy r228 knobs are ignored."""

    return _positive_seconds_from_env(
        R238_SEARCH_SECONDS_ENV, str(R238_DEFAULT_SEARCH_SECONDS)
    )


def _validated_precomputed_direct_action(
    action: Sequence[int] | None,
    *,
    legal: Sequence[tuple[int, ...]],
) -> tuple[int, ...] | None:
    """Accept only the parent's exact, already-complete legal root action.

    The broker has already produced this action from the parent frozen-r195
    root pass.  The child may consume it solely when a typed clean zero-backup
    deadline occurs; it must never turn an unchecked IPC value into action
    authority.
    """

    if action is None:
        return None
    if isinstance(action, (str, bytes)) or not isinstance(action, Sequence):
        raise R228GameplayError("precomputed parent direct action is not a sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in action):
        raise R228GameplayError(
            "precomputed parent direct action has a non-integer token"
        )
    normalized = tuple(int(item) for item in action)
    if normalized not in legal:
        raise R228GameplayError(
            "precomputed parent direct action is outside the complete legal order"
        )
    return normalized


def _stock_library_member_for_host(
    *, system: str | None = None, machine: str | None = None
) -> str:
    """Return the one r236-native member allowed for this runtime host."""

    host = (
        str(platform.system() if system is None else system).strip().lower(),
        str(platform.machine() if machine is None else machine).strip().lower(),
    )
    member = STOCK_LIBRARY_PLATFORM_MEMBERS.get(host)
    if member is None:
        raise R228GameplayError(
            "no canonical Kaggle Environments libcg member for "
            f"platform={host[0]!r}, machine={host[1]!r}"
        )
    return member


def _validate_staged_stock_library(
    stage: Path, *, member: str
) -> tuple[Path, str, int]:
    """Fail closed unless one resolved staged r236 member is exact.

    This runs before importing ``cg.sim`` as well as after the import proves
    which DSO was loaded.  The pre-import pass prevents an arbitrary staged
    native file from being selected before its bytes are bound to the r236
    manifest.
    """

    expected_digest = STOCK_LIBRARY_SHA256.get(member)
    expected_size = STOCK_LIBRARY_BYTES.get(member)
    if expected_digest is None or expected_size is None:
        raise R228GameplayError(f"unbound stock libcg member: {member!r}")
    try:
        cg_root = (Path(stage).resolve(strict=True) / "cg").resolve(strict=True)
        candidate = (cg_root / member).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise R228GameplayError(
            f"sealed stock libcg member is missing: cg/{member}"
        ) from exc
    if candidate.parent != cg_root or candidate.name != member or not candidate.is_file():
        raise R228GameplayError(
            f"stock libcg path is outside the sealed package: {candidate}"
        )
    try:
        observed_size = int(candidate.stat().st_size)
    except OSError as exc:
        raise R228GameplayError(f"unable to stat stock libcg member: {candidate}") from exc
    if observed_size != expected_size:
        raise R228GameplayError(
            f"stock {member} size mismatch: expected {expected_size}, got {observed_size}"
        )
    try:
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    except OSError as exc:
        raise R228GameplayError(f"unable to read stock libcg member: {candidate}") from exc
    if digest != expected_digest:
        raise R228GameplayError(
            f"stock {member} digest mismatch: expected {expected_digest}, got {digest}"
        )
    return candidate, digest, observed_size


def validate_staged_stock_library_identity(stage: Path | str) -> dict[str, Any]:
    """Verify the host r236 DSO before any direct/runtime CG import.

    The submission parent calls this before importing the frozen r195 direct
    entrypoint, which can otherwise map ``cg/libcg`` first.  It deliberately
    has no ``cg``/Torch/model imports: it binds only the resolved staged member
    and its exact r236 identity.  The child separately repeats this check
    before and after its native load, then verifies exports and the retained
    Python wrapper.
    """

    stock_member = _stock_library_member_for_host()
    library, digest, size_bytes = _validate_staged_stock_library(
        Path(stage), member=stock_member
    )
    return {
        "path": str(library),
        "member": f"cg/{stock_member}",
        "sha256": f"sha256:{digest}",
        "size_bytes": size_bytes,
        "expected_sha256": f"sha256:{STOCK_LIBRARY_SHA256[stock_member]}",
        "expected_size_bytes": STOCK_LIBRARY_BYTES[stock_member],
        "kaggle_environments_version": KAGGLE_ENVIRONMENTS_VERSION,
        "kaggle_environments_wheel_sha256": (
            f"sha256:{KAGGLE_ENVIRONMENTS_WHEEL_SHA256}"
        ),
        "native_library_update_commit": KAGGLE_ENVIRONMENTS_NATIVE_LIBRARY_UPDATE_COMMIT,
        "platform": platform.system().lower(),
        "machine": platform.machine().lower(),
    }


def _validate_required_native_exports(lib: Any) -> None:
    """Prove the r236 DSO still provides the frozen r195 ABI surface."""

    missing = tuple(
        name for name in REQUIRED_NATIVE_EXPORTS if not callable(getattr(lib, name, None))
    )
    if missing:
        raise R228GameplayError(
            "stock libcg is missing required exports: " + ", ".join(missing)
        )


def _validate_r225_api_compatibility(api: Any) -> None:
    """Require the retained r195 Python wrapper used by every r225 lane."""

    missing_callables = tuple(
        name
        for name in REQUIRED_R225_API_CALLABLES
        if not callable(getattr(api, name, None))
    )
    missing_types = tuple(
        name for name in REQUIRED_R225_API_TYPES if getattr(api, name, None) is None
    )
    if missing_callables or missing_types:
        missing = missing_callables + missing_types
        raise R228GameplayError(
            "frozen r195 cg.api compatibility is incomplete: " + ", ".join(missing)
        )


def _receipt_exact_int(receipt: Any, field: str) -> int:
    """Read one queue integer without accepting bool/coercion ambiguity."""

    if isinstance(receipt, Mapping):
        if field not in receipt:
            raise R228GameplayError(f"shared tree receipt omitted {field}")
        value = receipt[field]
    else:
        try:
            value = getattr(receipt, field)
        except AttributeError as exc:
            raise R228GameplayError(f"shared tree receipt omitted {field}") from exc
    if isinstance(value, bool) or not isinstance(value, int):
        raise R228GameplayError(
            f"shared tree receipt {field} must be an exact integer"
        )
    return int(value)


def _receipt_action(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise R228GameplayError(f"{label} is not an action sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise R228GameplayError(f"{label} contains a non-integer token")
    return tuple(int(item) for item in value)


def _validate_lane_search_composites(
    receipt: Any, *, allow_zero_depth: bool
) -> int:
    """Validate per-lane native identities without treating SearchIds as global.

    The official ``libcg`` SearchId namespace is scoped to an AgentStart
    handle.  In particular, two correct lanes commonly both begin at id zero.
    A receipt therefore proves lane separation with each
    ``(handle_identity, first_search_id)`` composite, never the raw SearchId
    alone.
    """

    expected = R228_SIMULATOR_LANE_COUNT
    per_lane_depth = tuple(getattr(receipt, "per_lane_depth", ()))
    per_lane_chains = tuple(getattr(receipt, "per_lane_search_id_chains", ()))
    per_lane_handles = tuple(
        getattr(receipt, "per_lane_handle_identities", ())
    )
    if (
        len(per_lane_depth) != expected
        or len(per_lane_chains) != expected
        or len(per_lane_handles) != expected
    ):
        raise R228GameplayError("shared tree receipt has an incomplete lane vector")

    minimum_depth = 0 if allow_zero_depth else 1
    if any(
        isinstance(depth, bool)
        or not isinstance(depth, int)
        or depth < minimum_depth
        for depth in per_lane_depth
    ):
        raise R228GameplayError("shared tree receipt has an invalid lane depth")
    if any(not isinstance(chain, (tuple, list)) or not chain for chain in per_lane_chains):
        raise R228GameplayError(
            "shared tree receipt has a lane without a SearchBegin chain"
        )

    normalized_handles: list[tuple[str, int | str]] = []
    search_begin_ids: list[int] = []
    for handle, chain, depth in zip(
        per_lane_handles, per_lane_chains, per_lane_depth
    ):
        if isinstance(handle, bool) or not isinstance(handle, (int, str)):
            raise R228GameplayError(
                "shared tree receipt has an invalid lane handle identity"
            )
        if isinstance(handle, str) and not handle.strip():
            raise R228GameplayError(
                "shared tree receipt has an empty lane handle identity"
            )
        if len(chain) != depth + 1:
            raise R228GameplayError(
                "shared tree receipt SearchId chain does not match lane depth"
            )
        if any(isinstance(search_id, bool) or not isinstance(search_id, int) for search_id in chain):
            raise R228GameplayError(
                "shared tree receipt has an invalid SearchBegin id"
            )
        if len(set(chain)) != len(chain):
            raise R228GameplayError(
                "shared tree receipt reused a live SearchId within one lane"
            )
        normalized_handles.append((type(handle).__name__, handle))
        search_begin_ids.append(chain[0])

    composites = tuple(
        (handle_type, handle, search_id)
        for (handle_type, handle), search_id in zip(
            normalized_handles, search_begin_ids
        )
    )
    distinct_composite_count = len(set(composites))
    if distinct_composite_count != expected:
        raise R228GameplayError(
            "shared tree receipt has duplicate SearchBegin handle/id composites"
        )
    if len(set(normalized_handles)) != expected:
        raise R228GameplayError(
            "shared tree receipt does not prove distinct lane handles"
        )
    reported_count = _receipt_exact_int(
        receipt, "distinct_search_begin_composite_count"
    )
    if reported_count != distinct_composite_count:
        raise R228GameplayError(
            "shared tree receipt composite SearchBegin count is invalid"
        )
    return distinct_composite_count


def _validate_search_receipt_lanes(receipt: Any) -> tuple[int, str, int]:
    """Require an exact two-lane adaptive receipt before action authority."""

    expected = R228_SIMULATOR_LANE_COUNT
    stop_reason = getattr(receipt, "stop_reason", None)
    if not isinstance(stop_reason, str) or stop_reason not in R238_NORMAL_STOP_REASONS:
        raise R228GameplayError("shared tree receipt has an invalid normal stop reason")
    terminal_win_stop = stop_reason == PROVEN_TERMINAL_WIN_STOP_REASON
    exact_counts = {
        "arena_count": _receipt_exact_int(receipt, "arena_count"),
        "unique_handle_count": _receipt_exact_int(receipt, "unique_handle_count"),
        "search_begin_calls": _receipt_exact_int(receipt, "search_begin_calls"),
        "search_end_calls": _receipt_exact_int(receipt, "search_end_calls"),
    }
    for field, observed in exact_counts.items():
        if observed != expected:
            raise R228GameplayError(
                f"shared tree receipt {field}={observed!r}, expected {expected}"
            )
    if _receipt_exact_int(receipt, "search_release_calls") < expected:
        raise R228GameplayError(
            "shared tree receipt released fewer searches than configured lanes"
        )
    distinct_search_begin_composite_count = _validate_lane_search_composites(
        receipt, allow_zero_depth=terminal_win_stop
    )
    if _receipt_exact_int(receipt, "search_step_calls") < expected:
        raise R228GameplayError(
            "shared tree receipt stepped fewer lanes than configured"
        )
    completed_backups = _receipt_exact_int(receipt, "completed_backups")
    minimum_completed_backups = 1 if terminal_win_stop else expected
    if completed_backups < minimum_completed_backups:
        raise R228GameplayError(
            "shared tree receipt has too few completed root backups for its stop"
        )
    if completed_backups > R238_MAXIMUM_BACKUPS_PER_DECISION:
        raise R228GameplayError("shared tree receipt exceeded its hard backup stop")
    if _receipt_exact_int(receipt, "root_visits") != completed_backups:
        raise R228GameplayError("shared tree receipt root visits do not match backups")
    depths = tuple(getattr(receipt, "per_lane_depth", ()))
    if sum(depths) != completed_backups:
        raise R228GameplayError("shared tree receipt lane depths do not match backups")
    if terminal_win_stop and not any(depth >= 1 for depth in depths):
        raise R228GameplayError("terminal-win receipt has no backed discovering lane")
    if any(
        isinstance(lane, bool)
        or not isinstance(lane, int)
        or lane < 0
        or lane >= expected
        for lane in receipt.completion_order
    ):
        raise R228GameplayError("shared tree receipt contains an invalid lane id")
    if any(
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
        or size > expected
        for size in receipt.microbatch_sizes
    ):
        raise R228GameplayError("shared tree receipt has an invalid microbatch size")
    peak_in_flight = _receipt_exact_int(receipt, "max_simulator_calls_in_flight")
    if peak_in_flight < 1 or peak_in_flight > expected:
        raise R228GameplayError("shared tree receipt has an invalid in-flight lane count")
    if _receipt_exact_int(receipt, "outstanding_virtual_loss") != 0:
        raise R228GameplayError("shared tree receipt leaked virtual loss")

    adaptive_thresholds = {
        "minimum_backups_before_stability": R238_MINIMUM_BACKUPS_BEFORE_STABILITY,
        "stable_root_leader_observations": R238_STABLE_ROOT_LEADER_OBSERVATIONS,
        "maximum_backups_per_decision": R238_MAXIMUM_BACKUPS_PER_DECISION,
    }
    for field, expected_value in adaptive_thresholds.items():
        observed = _receipt_exact_int(receipt, field)
        if observed != expected_value:
            raise R228GameplayError(
                f"shared tree receipt {field}={observed!r}, expected {expected_value}"
            )
    leader_stability_count = _receipt_exact_int(receipt, "leader_stability_count")
    if leader_stability_count < 0:
        raise R228GameplayError("shared tree receipt has a negative leader stability count")
    if stop_reason == "maximum_backups":
        if completed_backups != R238_MAXIMUM_BACKUPS_PER_DECISION:
            raise R228GameplayError(
                "shared tree receipt stopped for maximum backups before the hard cap"
            )
    elif stop_reason == "stable_root_leader":
        if completed_backups < R238_MINIMUM_BACKUPS_BEFORE_STABILITY:
            raise R228GameplayError(
                "shared tree receipt stabilized before the minimum backup count"
            )
        if leader_stability_count < R238_STABLE_ROOT_LEADER_OBSERVATIONS:
            raise R228GameplayError(
                "shared tree receipt stabilized without enough leader observations"
            )
    terminal_win_proof = getattr(receipt, "terminal_win_proof", None)
    if terminal_win_stop:
        if not isinstance(terminal_win_proof, Mapping):
            raise R228GameplayError("terminal-win stop omitted its proof")
    elif terminal_win_proof is not None:
        raise R228GameplayError(
            "shared tree receipt claimed a terminal-win proof without its stop"
        )
    return distinct_search_begin_composite_count, stop_reason, leader_stability_count


_TERMINAL_WIN_PROOF_KEYS = frozenset(
    {
        "proof_kind",
        "root_observation_fingerprint",
        "root_legal_order_fingerprint",
        "root_actor_seat",
        "root_action",
        "selected_action",
        "terminal_result",
        "terminal_winner_seat",
        "terminal_leaf_reached",
        "proof_path_action_count",
        "path_actor_seats",
        "path_no_actor_change_boundary",
        "path_no_opponent_boundary_crossing",
        "path_no_chance_boundary",
        "path_no_unresolved_randomness",
        "proof_is_deterministic",
        "discovering_lane_id",
    }
)


def _validate_terminal_win_proof(
    receipt: Any,
    *,
    root_observation_fingerprint: str,
    root_legal_order_fingerprint: str,
    root_actor_seat: int,
    legal_actions: Sequence[Sequence[int]],
) -> dict[str, object] | None:
    """Bind the exceptional r246 proof to this exact current root."""

    stop_reason = getattr(receipt, "stop_reason", None)
    raw_proof = getattr(receipt, "terminal_win_proof", None)
    if stop_reason != PROVEN_TERMINAL_WIN_STOP_REASON:
        if raw_proof is not None:
            raise R228GameplayError(
                "terminal-win proof lacks its dedicated stop reason"
            )
        return None
    if not isinstance(raw_proof, Mapping) or set(raw_proof) != set(
        _TERMINAL_WIN_PROOF_KEYS
    ):
        raise R228GameplayError("terminal-win proof has an invalid exact schema")
    if _receipt_exact_int(
        receipt,
        "owner_proven_deterministic_terminal_win_this_turn_revision",
    ) != PROVEN_TERMINAL_WIN_REVISION:
        raise R228GameplayError("terminal-win receipt has an invalid owner revision")
    if (
        getattr(receipt, "root_observation_fingerprint", None)
        != root_observation_fingerprint
        or raw_proof.get("root_observation_fingerprint")
        != root_observation_fingerprint
    ):
        raise R228GameplayError("terminal-win proof is stale for the current root")
    if (
        getattr(receipt, "root_legal_order_fingerprint", None)
        != root_legal_order_fingerprint
        or raw_proof.get("root_legal_order_fingerprint")
        != root_legal_order_fingerprint
    ):
        raise R228GameplayError("terminal-win proof has a stale root legal order")
    if (
        _receipt_exact_int(receipt, "root_seat") != root_actor_seat
        or _receipt_exact_int(receipt, "root_actor_seat") != root_actor_seat
        or _receipt_exact_int(raw_proof, "root_actor_seat") != root_actor_seat
    ):
        raise R228GameplayError("terminal-win proof has a stale root actor")
    if raw_proof.get("proof_kind") != PROVEN_TERMINAL_WIN_PROOF_KIND:
        raise R228GameplayError("terminal-win proof kind is invalid")
    if raw_proof.get("terminal_result") != "win":
        raise R228GameplayError("terminal-win proof is not a terminal win")
    if _receipt_exact_int(raw_proof, "terminal_winner_seat") != root_actor_seat:
        raise R228GameplayError("terminal-win proof winner differs from root actor")
    for field in (
        "terminal_leaf_reached",
        "path_no_actor_change_boundary",
        "path_no_opponent_boundary_crossing",
        "path_no_chance_boundary",
        "path_no_unresolved_randomness",
        "proof_is_deterministic",
    ):
        if raw_proof.get(field) is not True:
            raise R228GameplayError(f"terminal-win proof does not prove {field}")

    root_action = _receipt_action(
        raw_proof.get("root_action"), label="terminal-win root action"
    )
    selected_action = _receipt_action(
        raw_proof.get("selected_action"), label="terminal-win selected action"
    )
    receipt_action = tuple(int(item) for item in receipt.selected_action)
    normalized_legal = tuple(
        tuple(int(item) for item in action) for action in legal_actions
    )
    if (
        root_action != selected_action
        or selected_action != receipt_action
        or selected_action not in normalized_legal
        or _receipt_exact_int(receipt, "selected_action_visits") < 1
    ):
        raise R228GameplayError(
            "terminal-win proof does not identify its backed legal selected root action"
        )
    path_action_count = _receipt_exact_int(raw_proof, "proof_path_action_count")
    path_actor_seats = raw_proof.get("path_actor_seats")
    if (
        path_action_count < 1
        or path_action_count > _receipt_exact_int(receipt, "completed_backups")
        or not isinstance(path_actor_seats, (list, tuple))
        or len(path_actor_seats) != path_action_count
        or any(
            isinstance(actor, bool)
            or not isinstance(actor, int)
            or actor != root_actor_seat
            for actor in path_actor_seats
        )
    ):
        raise R228GameplayError("terminal-win proof path is not root-actor-only")
    discovering_lane_id = _receipt_exact_int(raw_proof, "discovering_lane_id")
    if not 0 <= discovering_lane_id < R228_SIMULATOR_LANE_COUNT:
        raise R228GameplayError("terminal-win proof has an invalid discovering lane")
    per_lane_depth = tuple(getattr(receipt, "per_lane_depth", ()))
    if per_lane_depth[discovering_lane_id] < path_action_count:
        raise R228GameplayError(
            "terminal-win proof path exceeds its backed discovering-lane depth"
        )
    if tuple(getattr(receipt, "principal_variation", ())) != ():
        raise R228GameplayError("terminal-win receipt retained a continuation plan")
    return {str(key): value for key, value in raw_proof.items()}


def _receipt_elapsed_seconds(receipt: Any) -> float:
    """Return the literal queue elapsed time or reject the receipt."""

    try:
        elapsed_seconds = float(getattr(receipt, "elapsed_seconds"))
    except (TypeError, ValueError, AttributeError) as exc:
        raise R228GameplayError("shared tree receipt has an invalid elapsed time") from exc
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise R228GameplayError("shared tree receipt has an invalid elapsed time")
    return elapsed_seconds


def _terminal_win_execution_marker_facts(
    receipt: Any,
    *,
    terminal_win_proof: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return literal R246 facts proved by this completed native receipt.

    This is deliberately not a compatibility projection.  It runs only after
    the queue has returned its final receipt and the proof has been bound to
    the current root.  In particular, ``...child_cleanup_complete`` refers to
    the child runtime's owned lane/search/reservation cleanup; the persistent
    broker process itself remains alive for later decisions and is not claimed
    to have been reaped here.
    """

    if terminal_win_proof is None:
        return {}
    if getattr(receipt, "stop_reason", None) != PROVEN_TERMINAL_WIN_STOP_REASON:
        raise R228GameplayError("terminal marker facts lack the terminal-win stop")
    completed_backups = _receipt_exact_int(receipt, "completed_backups")
    if completed_backups < 1:
        raise R228GameplayError("terminal marker facts lack a backed root action")
    if _receipt_exact_int(receipt, "root_visits") != completed_backups:
        raise R228GameplayError("terminal marker facts do not bind root backups")
    if _receipt_exact_int(receipt, "selected_action_visits") < 1:
        raise R228GameplayError("terminal marker facts lack selected-action backup")
    if terminal_win_proof.get("terminal_leaf_reached") is not True:
        raise R228GameplayError("terminal marker facts lack an exact terminal leaf")
    if (
        _receipt_exact_int(receipt, "arena_count") != R228_SIMULATOR_LANE_COUNT
        or _receipt_exact_int(receipt, "unique_handle_count")
        != R228_SIMULATOR_LANE_COUNT
        or _receipt_exact_int(receipt, "search_begin_calls")
        != R228_SIMULATOR_LANE_COUNT
        or _receipt_exact_int(receipt, "search_end_calls")
        != R228_SIMULATOR_LANE_COUNT
        or _receipt_exact_int(receipt, "search_release_calls")
        < R228_SIMULATOR_LANE_COUNT
        or _receipt_exact_int(receipt, "outstanding_virtual_loss") != 0
    ):
        raise R228GameplayError(
            "terminal marker facts lack completed two-lane resource cleanup"
        )
    elapsed_seconds = _receipt_elapsed_seconds(receipt)
    return {
        # These exact counts describe this one completed root receipt; they
        # are never inferred later from marker spelling or a converter.
        "completed_root_backup_count": completed_backups,
        "terminal_win_proof_count": 1,
        "proven_deterministic_terminal_win_this_turn_stop_count": 1,
        "child_search_elapsed_seconds": elapsed_seconds,
        "two_lane_topology_initialized_before_terminal_win_override": True,
        "terminal_win_proof_backed_up_into_shared_root_tree": True,
        "terminal_leaf_returned_by_exact_stock_simulator": True,
        "all_owned_lane_resources_reservations_and_child_cleanup_complete": True,
        # R246 is intentionally the one-proof, non-exhaustive exception once
        # the above exact receipt has been established.
        "two_independent_lane_proofs_required": False,
        "exhaustive_legal_action_scan_required": False,
        (
            "standard_adaptive_min_backups_leader_observations_and_"
            "both_lanes_progressed_required_after_valid_proof"
        ): False,
    }


def _validate_clean_zero_backup_receipt(receipt: Any) -> int:
    """Prove the queue fully cleaned a clean child deadline with zero backups.

    This carries no child action authority.  It is deliberately a narrower
    receipt than a normal MCTS result: every lane must have opened and closed
    cleanly, but a deadline may occur before either lane completes a step.
    """

    expected = R228_SIMULATOR_LANE_COUNT
    exact_counts = {
        "arena_count": _receipt_exact_int(receipt, "arena_count"),
        "unique_handle_count": _receipt_exact_int(receipt, "unique_handle_count"),
        "search_begin_calls": _receipt_exact_int(receipt, "search_begin_calls"),
        "search_end_calls": _receipt_exact_int(receipt, "search_end_calls"),
    }
    for field, observed in exact_counts.items():
        if observed != expected:
            raise R228GameplayError(
                f"clean zero-backup receipt {field}={observed!r}, expected {expected}"
            )
    if _receipt_exact_int(receipt, "search_release_calls") < expected:
        raise R228GameplayError(
            "clean zero-backup receipt released fewer searches than configured lanes"
        )
    if _receipt_exact_int(receipt, "completed_backups") != 0:
        raise R228GameplayError(
            "clean zero-backup receipt unexpectedly completed a backup"
        )
    if _receipt_exact_int(receipt, "search_step_calls") != expected:
        raise R228GameplayError(
            "clean zero-backup receipt did not issue exactly one step per lane"
        )
    if _receipt_exact_int(receipt, "max_simulator_calls_in_flight") != expected:
        raise R228GameplayError(
            "clean zero-backup receipt did not reserve both simulator lanes"
        )
    completion_order = tuple(getattr(receipt, "completion_order", ()))
    microbatch_sizes = tuple(getattr(receipt, "microbatch_sizes", ()))
    if completion_order or microbatch_sizes:
        raise R228GameplayError(
            "clean zero-backup receipt admitted a simulator completion or model batch"
        )
    if tuple(getattr(receipt, "per_lane_depth", ())) != (0,) * expected:
        raise R228GameplayError(
            "clean zero-backup receipt retained an expanded lane path"
        )
    if _receipt_exact_int(receipt, "outstanding_virtual_loss") != 0:
        raise R228GameplayError("clean zero-backup receipt leaked virtual loss")
    if getattr(receipt, "stop_reason", None) != "decision_deadline":
        raise R228GameplayError(
            "clean zero-backup receipt did not stop at the decision deadline"
        )
    if _receipt_exact_int(receipt, "root_seat") not in (0, 1):
        raise R228GameplayError("clean zero-backup receipt has an invalid root seat")
    for field, expected_value in {
        "minimum_backups_before_stability": R238_MINIMUM_BACKUPS_BEFORE_STABILITY,
        "stable_root_leader_observations": R238_STABLE_ROOT_LEADER_OBSERVATIONS,
        "maximum_backups_per_decision": R238_MAXIMUM_BACKUPS_PER_DECISION,
    }.items():
        observed = _receipt_exact_int(receipt, field)
        if observed != expected_value:
            raise R228GameplayError(
                f"clean zero-backup receipt {field}={observed!r}, expected {expected_value}"
            )
    if _receipt_exact_int(receipt, "leader_stability_count") != 0:
        raise R228GameplayError(
            "clean zero-backup receipt retained a leader stability observation"
        )
    try:
        elapsed_seconds = float(getattr(receipt, "elapsed_seconds", float("nan")))
    except (TypeError, ValueError) as exc:
        raise R228GameplayError(
            "clean zero-backup receipt has an invalid elapsed time"
        ) from exc
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise R228GameplayError("clean zero-backup receipt has an invalid elapsed time")
    return _validate_lane_search_composites(receipt, allow_zero_depth=True)


def _validate_principal_variation(
    receipt: Any, *, root_seat: int
) -> tuple[dict[str, object], ...]:
    """Normalize only queue-proved deterministic continuation entries.

    The queue proves the two lane-private paths agree on each emitted leaf
    fingerprint and backed edge.  This boundary makes that proof explicit in
    the child receipt and keeps the parent-facing schema narrow: a later
    consumer must still match the *current* full observation and complete
    legal order before it can execute an entry.
    """

    if root_seat not in (0, 1):
        raise R228GameplayError("principal variation has an invalid root seat")
    if _receipt_exact_int(receipt, "root_seat") != root_seat:
        raise R228GameplayError("principal variation root seat does not match root")
    try:
        raw_entries = receipt.principal_variation
    except AttributeError as exc:
        raise R228GameplayError("shared tree receipt omitted principal variation") from exc
    if not isinstance(raw_entries, tuple) or len(raw_entries) > 8:
        raise R228GameplayError("shared tree receipt has an invalid principal variation")

    normalized: list[dict[str, object]] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, Mapping) or set(entry) != {
            "observation_fingerprint",
            "action",
        }:
            raise R228GameplayError(
                f"shared tree principal variation entry {index} has an invalid schema"
            )
        fingerprint = entry["observation_fingerprint"]
        action = entry["action"]
        if not isinstance(fingerprint, str) or not fingerprint:
            raise R228GameplayError(
                f"shared tree principal variation entry {index} has no fingerprint"
            )
        if not isinstance(action, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in action
        ):
            raise R228GameplayError(
                f"shared tree principal variation entry {index} has an invalid action"
            )
        normalized.append(
            {
                "observation_fingerprint": fingerprint,
                "action": [int(item) for item in action],
            }
        )
    return tuple(normalized)


class R228AsyncGameplay:
    """Process-wide persistent simulator lanes with per-decision shared trees."""

    def __init__(
        self,
        *,
        stage: Path,
        model: Any,
        policy: Any,
        deck: Sequence[int],
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        from .r225_stock_native_lane import (
            R225StockNativeSearchLane,
            prewarm_stock_cg,
        )

        self.stage = stage.resolve()
        self.model = model
        self.policy = policy
        self.deck = tuple(int(card) for card in deck)
        self.cleanup_timeout_seconds = _positive_seconds_from_env(
            "POKEBOT_R228_CLEANUP_TIMEOUT_SECONDS", "1.0"
        )
        if len(self.deck) != 60:
            raise R228GameplayError("frozen r195 deck is not 60 cards")
        # Verify the exact resolved ``stage/cg/<member>`` before importing the
        # binding.  On Kaggle Linux x86-64 this is precisely ``cg/libcg.so``.
        # This avoids loading an unbound staged native member before the
        # r236 checksum/size contract is checked.
        pre_import_identity = validate_staged_stock_library_identity(self.stage)
        staged_library = Path(str(pre_import_identity["path"]))
        staged_digest = str(pre_import_identity["sha256"]).removeprefix("sha256:")
        staged_size = int(pre_import_identity["size_bytes"])
        api, sim = prewarm_stock_cg()
        loaded_name = str(getattr(sim.lib, "_name", ""))
        try:
            loaded = Path(loaded_name).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise R228GameplayError(
                "stock libcg binding did not expose a resolved loaded library"
            ) from exc
        # The binding must map exactly the file which passed the pre-import
        # check, then that member is checked again in case it changed while
        # the native module was being initialized.
        post_import_identity = validate_staged_stock_library_identity(self.stage)
        loaded_library = Path(str(post_import_identity["path"]))
        digest = str(post_import_identity["sha256"]).removeprefix("sha256:")
        size_bytes = int(post_import_identity["size_bytes"])
        if loaded != staged_library or loaded != loaded_library:
            raise R228GameplayError(
                "stock libcg binding loaded a different member than the sealed stage"
            )
        if digest != staged_digest or size_bytes != staged_size:
            raise R228GameplayError(
                "stock libcg member changed during native binding initialization"
            )
        _validate_required_native_exports(sim.lib)
        _validate_r225_api_compatibility(api)
        self.stock_library_receipt = dict(post_import_identity)
        self.stock_library_receipt.update(
            {
                "required_native_exports": list(REQUIRED_NATIVE_EXPORTS),
                "required_r225_api_callables": list(REQUIRED_R225_API_CALLABLES),
                "required_r225_api_types": list(REQUIRED_R225_API_TYPES),
            }
        )

        def arena_factory(lane_id: int) -> Any:
            return R225StockNativeSearchLane(lane_id, lib=sim.lib, api_module=api)

        self._decision: dict[str, Any] | None = None
        self._search = PersistentAsyncSharedTreeMCTS(
            arena_factory=arena_factory,
            make_packet=self._make_packet,
            evaluate_batch=self._evaluate_batch,
            coalesce_seconds=float(os.environ.get("POKEBOT_R228_COALESCE_SECONDS", "0.001")),
            lane_count=R228_SIMULATOR_LANE_COUNT,
            minimum_backups_before_stability=R238_MINIMUM_BACKUPS_BEFORE_STABILITY,
            stable_root_leader_observations=R238_STABLE_ROOT_LEADER_OBSERVATIONS,
            maximum_backups_per_decision=R238_MAXIMUM_BACKUPS_PER_DECISION,
            cleanup_timeout_seconds=self.cleanup_timeout_seconds,
            progress_callback=progress_callback,
        )
        self.decision_count = 0
        self.decision_receipts: list[dict[str, Any]] = []

    def close(self) -> None:
        self._search.close()

    def reset_game(self) -> None:
        self.decision_count = 0
        self.decision_receipts.clear()
        reset = getattr(self.policy, "reset_game", None)
        if callable(reset):
            reset()

    def _make_packet(self, lane_id: int, observation: Any) -> _Frontier:
        return _Frontier(lane_id=int(lane_id), raw=_raw(observation))

    def _leaf_packet(
        self,
        raw: dict[str, Any],
        *,
        combos: Sequence[Sequence[int]],
        actor_seat: int,
    ) -> Any:
        from . import features
        from .batched_infer import LeafPacket

        context = self._decision
        if context is None:
            raise R228GameplayError("leaf evaluation has no active decision")
        if actor_seat not in (0, 1):
            raise R228GameplayError("simulator leaf has invalid acting seat")
        acting_deck = (
            self.deck
            if actor_seat == context["root_seat"]
            else context["opponent_deck"]
        )
        board = features.build_board_tokens(raw, list(acting_deck))
        histories = list(context["history_boards"]) + [board]
        previous = list(context["history_previous_actions"]) + [None]
        limit = int(self.policy._history_context_limit())
        return LeafPacket(
            obs=raw,
            your_deck=list(acting_deck),
            root_seat=int(context["root_seat"]),
            history_boards=histories[-limit:],
            history_previous_actions=previous[-limit:],
            action_combos_override=[list(map(int, action)) for action in combos],
            matchup_route=int(context["route"]),
        )

    def _evaluate_batch(self, frontiers: Sequence[_Frontier]) -> Sequence[DecodedLeaf]:
        from . import cg_env, features
        from .batched_infer import forward_leaf_batch

        context = self._decision
        if context is None:
            raise R228GameplayError("model batch has no active decision")
        for field in (
            "boundary_leaf_count",
            "chance_boundary_leaf_count",
            "actor_change_boundary_leaf_count",
        ):
            current_count = context.get(field, 0)
            if (
                isinstance(current_count, bool)
                or not isinstance(current_count, int)
                or current_count < 0
            ):
                raise R228GameplayError(f"simulator leaf has invalid {field}")
            context[field] = int(current_count)
        pending: list[
            tuple[int, _Frontier, tuple[tuple[int, ...], ...], bool, int]
        ] = []
        decoded: list[DecodedLeaf | None] = [None] * len(frontiers)
        packets: list[Any] = []
        for index, frontier in enumerate(frontiers):
            raw = frontier.raw
            if cg_env.is_finished(raw):
                winner = cg_env.result_winner(raw)
                if isinstance(winner, bool) or winner not in (None, 0, 1, 2):
                    raise R228GameplayError(
                        "finished simulator leaf has an invalid winner seat"
                    )
                terminal_chance_boundary = _chance_boundary(raw)
                if terminal_chance_boundary:
                    context["chance_boundary_leaf_count"] += 1
                    context["boundary_leaf_count"] += 1
                if winner in (None, 2):
                    terminal_result = "draw"
                    terminal_winner_seat = None if winner is None else 2
                    value = 0.0
                elif int(winner) == context["root_seat"]:
                    terminal_result = "win"
                    terminal_winner_seat = int(winner)
                    value = 1.0
                else:
                    terminal_result = "loss"
                    terminal_winner_seat = int(winner)
                    value = -1.0
                decoded[index] = DecodedLeaf(
                    state_key=_state_key(lane_id=frontier.lane_id, raw=raw),
                    observation_fingerprint=canonical_observation_fingerprint(raw),
                    value=value,
                    legal_actions=(),
                    priors=(),
                    boundary=True,
                    actor_seat=None,
                    terminal_result=terminal_result,
                    terminal_winner_seat=terminal_winner_seat,
                    terminal_leaf_reached=True,
                    chance_boundary=terminal_chance_boundary,
                    unresolved_randomness=terminal_chance_boundary,
                )
                continue
            current = raw.get("current")
            if not isinstance(current, Mapping):
                raise R228GameplayError("simulator leaf has no current state")
            raw_actor = current.get("yourIndex")
            if (
                isinstance(raw_actor, bool)
                or not isinstance(raw_actor, int)
                or raw_actor not in (0, 1)
            ):
                raise R228GameplayError("simulator leaf has invalid acting seat")
            actor = int(raw_actor)
            root_seat = context.get("root_seat")
            if (
                isinstance(root_seat, bool)
                or not isinstance(root_seat, int)
                or root_seat not in (0, 1)
            ):
                raise R228GameplayError("simulator leaf has an invalid root seat")

            # A simulated opponent/end-turn transition remains useful value
            # evidence, but never grants this root-owned search authority to
            # choose an opponent action.  Like an unforceable chance prompt,
            # it is a leaf boundary after the model evaluates it.
            chance_boundary = _chance_boundary(raw)
            actor_change_boundary = actor != root_seat
            boundary = chance_boundary or actor_change_boundary
            if chance_boundary:
                context["chance_boundary_leaf_count"] += 1
            if actor_change_boundary:
                context["actor_change_boundary_leaf_count"] += 1
            if boundary:
                context["boundary_leaf_count"] += 1
            if boundary:
                # A chance or actor-change leaf still needs the frozen
                # state/value forward, but cannot expand an action.  Avoid
                # materializing another complete private legal order merely
                # to produce priors that are intentionally discarded.  The
                # one empty option is a deterministic value-only model token.
                combos = ((),)
            else:
                combos = tuple(
                    tuple(int(item) for item in action)
                    for action in features.enumerate_action_combos(
                        raw, max_combos=R228_COMPLETE_ACTION_CAP
                    )
                )
            if not combos:
                raise R228GameplayError("nonterminal simulator leaf has no legal actions")
            packets.append(
                self._leaf_packet(raw, combos=combos, actor_seat=actor)
            )
            pending.append((index, frontier, combos, boundary, actor))
        evaluated = forward_leaf_batch(self.model, packets) if packets else []
        if len(evaluated) != len(pending):
            raise R228GameplayError("frozen model returned a partial leaf batch")
        for (index, frontier, combos, boundary, actor), leaf in zip(
            pending, evaluated
        ):
            priors = tuple(float(value) for value in leaf.priors)
            if len(priors) != len(combos) or any(
                not math.isfinite(value) or value < 0.0 for value in priors
            ):
                raise R228GameplayError("frozen model returned malformed leaf priors")
            decoded[index] = DecodedLeaf(
                state_key=_state_key(lane_id=frontier.lane_id, raw=frontier.raw),
                observation_fingerprint=canonical_observation_fingerprint(
                    frontier.raw
                ),
                value=float(leaf.value),
                legal_actions=() if boundary else combos,
                priors=() if boundary else priors,
                boundary=boundary,
                actor_seat=actor,
                chance_boundary=chance_boundary,
                actor_change_boundary=actor_change_boundary,
                unresolved_randomness=chance_boundary,
            )
        if any(row is None for row in decoded):
            raise R228GameplayError("leaf decode was incomplete")
        return tuple(row for row in decoded if row is not None)

    def select(
        self,
        obs: dict[str, Any],
        *,
        precomputed_direct_action: Sequence[int] | None = None,
    ) -> list[int]:
        """Return one backed MCTS action or the exact typed zero-backup fallback.

        ``precomputed_direct_action`` is optional for standalone compatibility.
        When supplied by the broker parent it is checked against this exact
        complete legal order, retained without another greedy policy pass, and
        used only if the queue proves a clean zero-backup deadline.
        """

        from . import cg_env, features
        from .batched_infer import LeafPacket, forward_leaf_batch

        legal = tuple(
            tuple(int(item) for item in action)
            for action in features.enumerate_action_combos(
                obs, max_combos=R228_COMPLETE_ACTION_CAP
            )
        )
        if len(legal) < 2:
            raise R228GameplayError("async search requires a branching prompt")
        parent_precomputed_direct_action = _validated_precomputed_direct_action(
            precomputed_direct_action, legal=legal
        )
        current = obs.get("current")
        if not isinstance(current, Mapping):
            raise R228GameplayError("branching observation has no current state")
        root_seat = int(current.get("yourIndex", -1))
        if root_seat not in (0, 1):
            raise R228GameplayError("branching observation has invalid acting seat")
        root_observation_fingerprint = canonical_observation_fingerprint(obs)
        root_legal_order_fingerprint = legal_order_fingerprint(legal)

        router = getattr(self.policy, "_matchup_adapter_shadow_router", None)
        if router is not None and hasattr(router, "observe"):
            router.observe(obs, scope="game_root", depth=len(self.policy.board_history))
        self.policy._append_decision_history(obs)
        route = int(self.policy._matchup_model_route())
        root_packet = LeafPacket(
            obs=obs,
            your_deck=list(self.deck),
            root_seat=root_seat,
            history_boards=list(self.policy.board_history),
            history_previous_actions=list(self.policy.previous_action_history),
            action_combos_override=[list(action) for action in legal],
            matchup_route=route,
        )
        root_leaf = forward_leaf_batch(self.model, [root_packet])[0]
        root_priors = tuple(float(value) for value in root_leaf.priors)
        if (
            len(root_priors) != len(legal)
            or any(not math.isfinite(value) or value < 0.0 for value in root_priors)
            or math.fsum(root_priors) <= 0.0
        ):
            raise R228GameplayError("frozen model returned malformed root priors")
        direct_index = max(
            range(len(legal)), key=lambda index: (root_priors[index], -index)
        )
        direct_action = legal[direct_index]
        direct_probability = root_priors[direct_index]
        clean_zero_direct_action = (
            parent_precomputed_direct_action
            if parent_precomputed_direct_action is not None
            else direct_action
        )
        clean_zero_direct_probability = root_priors[
            legal.index(clean_zero_direct_action)
        ]

        search_inputs = cg_env.build_search_inputs(
            obs,
            list(self.deck),
            opponent_deck_guess=list(self.deck),
        )
        opponent = tuple(int(card) for card in search_inputs.get("opponent_deck", ()))
        if len(opponent) != 60:
            opponent = self.deck
        decision_context = {
            "root_seat": root_seat,
            "route": route,
            "opponent_deck": opponent,
            "history_boards": list(self.policy.board_history),
            "history_previous_actions": list(self.policy.previous_action_history),
            # These are incremented only for nonterminal simulator leaves.
            # A leaf with both traits is counted once in the total and once in
            # each reason-specific field, retaining the causal overlap.
            "boundary_leaf_count": 0,
            "chance_boundary_leaf_count": 0,
            "actor_change_boundary_leaf_count": 0,
        }
        self._decision = decision_context
        search_seconds = _r238_search_seconds()
        started = time.monotonic()
        try:
            receipt = self._search.run_decision(
                root_observation=obs,
                search_inputs=tuple(
                    dict(search_inputs) for _ in range(R228_SIMULATOR_LANE_COUNT)
                ),
                root_state_key=_state_key(lane_id=None, raw=obs),
                root_actions=legal,
                root_priors=root_priors,
                root_seat=root_seat,
                root_actor_seat=root_seat,
                root_observation_fingerprint=root_observation_fingerprint,
                root_legal_order_fingerprint=root_legal_order_fingerprint,
                deadline_monotonic=started + search_seconds,
            )
            (
                distinct_search_begin_composite_count,
                stop_reason,
                leader_stability_count,
            ) = _validate_search_receipt_lanes(receipt)
            principal_variation = _validate_principal_variation(
                receipt, root_seat=root_seat
            )
            terminal_win_proof = _validate_terminal_win_proof(
                receipt,
                root_observation_fingerprint=root_observation_fingerprint,
                root_legal_order_fingerprint=root_legal_order_fingerprint,
                root_actor_seat=root_seat,
                legal_actions=legal,
            )
            terminal_win_execution_facts = _terminal_win_execution_marker_facts(
                receipt,
                terminal_win_proof=terminal_win_proof,
            )
            child_search_elapsed_seconds = _receipt_elapsed_seconds(receipt)
            selected = tuple(int(item) for item in receipt.selected_action)
            if selected not in legal or receipt.selected_action_visits < 1:
                raise R228GameplayError("shared tree returned an unbacked or illegal action")
            mode = "shared_tree_mcts"
        except AsyncDirectFallbackRequired as exc:
            # This exact post-cleanup result means the clean deadline produced
            # zero backups.  Structural/native failures remain hard faults and
            # are deliberately not downgraded in this viability submission.
            # The root model pass above has already produced the frozen direct
            # action from this exact complete legal ordering.  Reusing it keeps
            # a clean-deadline fallback bounded and avoids mutating policy
            # history through a second greedy inference path.
            selected = clean_zero_direct_action
            if selected not in legal:
                raise R228GameplayError("clean-deadline frozen fallback was illegal")
            receipt = getattr(exc, "cleanup_receipt", None)
            if receipt is None:
                raise R228GameplayError(
                    "clean-deadline fallback omitted its cleanup receipt"
                ) from exc
            distinct_search_begin_composite_count = (
                _validate_clean_zero_backup_receipt(receipt)
            )
            if _receipt_exact_int(receipt, "root_seat") != root_seat:
                raise R228GameplayError(
                    "clean-deadline receipt root seat differs from the parent root"
                )
            stop_reason = str(receipt.stop_reason)
            leader_stability_count = _receipt_exact_int(
                receipt, "leader_stability_count"
            )
            principal_variation = ()
            terminal_win_proof = None
            terminal_win_execution_facts = {}
            child_search_elapsed_seconds = _receipt_elapsed_seconds(receipt)
            mode = "clean_deadline_zero_backup_frozen_model_fallback"
        finally:
            self._decision = None

        self.policy._previous_action_token = features.build_option_tokens(
            obs, [list(selected)]
        )
        self.decision_count += 1
        payload_direct_action = (
            clean_zero_direct_action
            if mode == "clean_deadline_zero_backup_frozen_model_fallback"
            else direct_action
        )
        payload_direct_probability = (
            clean_zero_direct_probability
            if mode == "clean_deadline_zero_backup_frozen_model_fallback"
            else direct_probability
        )
        payload = {
            "schema": SCHEMA,
            "decision": self.decision_count,
            "mode": mode,
            "mcts_action_authority": mode == "shared_tree_mcts",
            "selected_action": list(selected),
            "legal_action_count": len(legal),
            "complete_ordered_action_cap": R228_COMPLETE_ACTION_CAP,
            "requested_simulator_lane_count": R228_SIMULATOR_LANE_COUNT,
            "owner_proven_deterministic_terminal_win_this_turn_revision": (
                PROVEN_TERMINAL_WIN_REVISION
            ),
            "root_observation_fingerprint": root_observation_fingerprint,
            "root_legal_order_fingerprint": root_legal_order_fingerprint,
            "root_actor_seat": root_seat,
            "configured_search_seconds": search_seconds,
            "minimum_backups_before_stability": R238_MINIMUM_BACKUPS_BEFORE_STABILITY,
            "stable_root_leader_observations": R238_STABLE_ROOT_LEADER_OBSERVATIONS,
            "stable_root_leader_observations_required": (
                R238_STABLE_ROOT_LEADER_OBSERVATIONS
            ),
            "maximum_backups_per_decision": R238_MAXIMUM_BACKUPS_PER_DECISION,
            "cleanup_timeout_seconds": self.cleanup_timeout_seconds,
            "direct_action": list(payload_direct_action),
            "direct_action_probability": payload_direct_probability,
            "mcts_action_direct_probability": root_priors[legal.index(selected)],
            "mcts_action_direct_rank": 1 + sum(
                probability > root_priors[legal.index(selected)]
                for probability in root_priors
            ),
            "direct_probability_gap": payload_direct_probability
            - root_priors[legal.index(selected)],
            "action_changed": selected != payload_direct_action,
            "boundary_leaf_count": int(decision_context["boundary_leaf_count"]),
            "chance_boundary_leaf_count": int(
                decision_context["chance_boundary_leaf_count"]
            ),
            "actor_change_boundary_leaf_count": int(
                decision_context["actor_change_boundary_leaf_count"]
            ),
        }
        if receipt is not None:
            per_lane_handle_identities = list(receipt.per_lane_handle_identities)
            per_lane_first_search_ids = [
                int(chain[0]) for chain in receipt.per_lane_search_id_chains
            ]
            handle_scoped_first_search_id_composite_states = [
                {
                    "lane_id": lane_id,
                    "handle_identity": per_lane_handle_identities[lane_id],
                    "first_search_id": per_lane_first_search_ids[lane_id],
                }
                for lane_id in range(R228_SIMULATOR_LANE_COUNT)
            ]
            payload.update(
                {
                    "arena_count": receipt.arena_count,
                    "active_simulator_lane_count": receipt.arena_count,
                    "unique_handle_count": receipt.unique_handle_count,
                    "search_begin_calls": receipt.search_begin_calls,
                    "search_step_calls": receipt.search_step_calls,
                    "completed_backups": receipt.completed_backups,
                    "max_simulator_calls_in_flight": receipt.max_simulator_calls_in_flight,
                    "microbatch_sizes": list(receipt.microbatch_sizes),
                    "per_lane_depth": list(receipt.per_lane_depth),
                    "per_lane_search_id_chains": [
                        list(chain) for chain in receipt.per_lane_search_id_chains
                    ],
                    "per_lane_handle_identities": per_lane_handle_identities,
                    "per_lane_first_search_ids": per_lane_first_search_ids,
                    "handle_scoped_first_search_id_composite_states": (
                        handle_scoped_first_search_id_composite_states
                    ),
                    "distinct_search_begin_composite_count": (
                        distinct_search_begin_composite_count
                    ),
                    "search_release_calls": receipt.search_release_calls,
                    "search_end_calls": receipt.search_end_calls,
                    "outstanding_virtual_loss": receipt.outstanding_virtual_loss,
                    "stop_reason": stop_reason,
                    "leader_stability_count": leader_stability_count,
                    # The queue's native name remains useful for diagnostics;
                    # these two exact aliases bind it to the r240 receipt
                    # vocabulary used by the parent/stager contract.
                    "observed_stable_root_leader_observations": leader_stability_count,
                    "deterministic_root_leader_observations": leader_stability_count,
                    "root_seat": root_seat,
                    # These are two literal spellings of the same validated
                    # queue timer, emitted at its source rather than added by
                    # an external preflight converter.
                    "elapsed_seconds": child_search_elapsed_seconds,
                    "child_search_elapsed_seconds": child_search_elapsed_seconds,
                }
            )
            if mode == "shared_tree_mcts":
                payload.update(
                    {
                        "selected_action_visits": receipt.selected_action_visits,
                        "selected_action_value": receipt.selected_action_value,
                        "selected_action_prior": receipt.selected_action_prior,
                        "root_visits": receipt.root_visits,
                        "principal_variation": [
                            dict(entry) for entry in principal_variation
                        ],
                        "terminal_win_proof": (
                            None
                            if terminal_win_proof is None
                            else dict(terminal_win_proof)
                        ),
                        "proven_deterministic_terminal_win_this_turn": (
                            terminal_win_proof is not None
                        ),
                        **terminal_win_execution_facts,
                        "meaningful_choice_change": (
                            selected != payload_direct_action
                            and receipt.completed_backups > 0
                            and payload_direct_probability
                            - root_priors[legal.index(selected)]
                            > float(
                                os.environ.get(
                                    "POKEBOT_R229_PROBABILITY_TIE_TOLERANCE",
                                    "1e-6",
                                )
                            )
                        ),
                    }
                )
            else:
                payload.update(
                    {
                        # A child zero-backup result is evidence of its own
                        # bounded cleanup only.  The parent/broker will reap
                        # this exact child and retain sole action authority.
                        "clean_deadline_cleanup_complete": True,
                        "principal_variation": [],
                        "meaningful_choice_change": False,
                    }
                )
        payload["stock_library"] = dict(self.stock_library_receipt)
        self.decision_receipts.append(dict(payload))
        print(DECISION_PREFIX + " " + json.dumps(payload, sort_keys=True), flush=True)
        return list(selected)


__all__ = [
    "DECISION_PREFIX",
    "R228_COMPLETE_ACTION_CAP",
    "R228_SIMULATOR_LANE_COUNT",
    "SCHEMA",
    "R228AsyncGameplay",
    "R228GameplayError",
    "canonical_observation_fingerprint",
    "legal_order_fingerprint",
    "validate_staged_stock_library_identity",
]
