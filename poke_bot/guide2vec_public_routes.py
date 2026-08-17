"""Exact r195 public-matchup route reconstruction for Guide2Vec r212.

The protected temporal feature shards intentionally predate the r195 submitted
runtime's enabled public matchup adapter.  Their
``DecisionSample.matchup_adapter_public_route`` values are therefore all
``UNKNOWN_ROUTE``.  This module never mutates those protected rows.  Instead it
replays the exact *public* router over the header-bound raw daily episode ZIP
and produces one physical V6 route (or exact bypass) for every compact
decision.

Two details are deliberately fail-closed:

* An archived row with a stale ``select`` observation but ``status != ACTIVE``
  is not an agent invocation and must not advance the router.  The r195 agent
  observes immediately before an actual policy call only.
* A compact row is accepted only when its ``episode_id``, seat, environment
  step, frozen board features, and shifted raw action token all agree with the
  raw episode.  This allows a last-320 compact sequence to rebuild the whole
  causal raw prefix without trusting an ordinal row join.

The module can materialize an immutable per-day sidecar.  That permits a host
which owns a protected raw ZIP (for example Elmo for the final heldout days) to
do the reconstruction once; Blackwell then consumes a compact, checksum-bound
projection without requiring that raw archive locally.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import re
import struct
import tarfile
import time
import zipfile
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import features
from .dataset import GameSequence
from .feature_shards import SHARD_FORMAT, SHARD_FORMAT_VERSION, iter_feature_shard
from .public_matchup_router import (
    PublicMatchupDecisionTree,
    RuntimePublicMatchupRouter,
)
from .replay_import import _strip_opp_private, episode_id_of


ROUTE_RECONSTRUCTION_SCHEMA = "poke_bot.guide2vec_r212_r195_public_route_recomputation/v1"
ROUTE_SIDECAR_HEADER_SCHEMA = "poke_bot.guide2vec_r212_r195_public_route_sidecar_header/v1"
ROUTE_SIDECAR_ROW_SCHEMA = "poke_bot.guide2vec_r212_r195_public_route_sidecar_row/v1"
ROUTE_SIDECAR_FOOTER_SCHEMA = "poke_bot.guide2vec_r212_r195_public_route_sidecar_footer/v1"
ROUTE_SIDECAR_FORMAT = "r212-r195-public-routes"
ROUTE_ALGORITHM = "r195_submission_active_select_non_turn_order_public_tree_prefix_v2"
ALIGNMENT_CONTRACT = "episode_seat_env_step_board_and_shifted_action_token_f32/v1"
TURN_ORDER_SHORT_CIRCUIT_CONTRACT = "r195_submission_main_turn_order_choice/v1"
PRODUCER_CODE_SCHEMA = "poke_bot.guide2vec_r212_r195_public_route_producer_code/v1"
# The submitted public router serializes its bypass route as this fixed scalar.
# Keep the raw sidecar producer torch-free so it can run on the archive-owning
# Elmo host; every loaded tree is separately required to report the same value.
UNKNOWN_ROUTE = -1

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class PublicRouteReconstructionError(RuntimeError):
    """The exact raw-to-compact public route join could not be proven."""


@dataclass(frozen=True)
class RuntimeCodeBinding:
    """Exact r195 packaged code identities defining router invocation semantics."""

    submission_bundle_sha256: str
    submission_entrypoint_member: str
    submission_entrypoint_sha256: str
    public_matchup_router_member: str
    public_matchup_router_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "submission_bundle_sha256": self.submission_bundle_sha256,
            "submission_entrypoint_member": self.submission_entrypoint_member,
            "submission_entrypoint_sha256": self.submission_entrypoint_sha256,
            "public_matchup_router_member": self.public_matchup_router_member,
            "public_matchup_router_sha256": self.public_matchup_router_sha256,
            "turn_order_short_circuit_contract": TURN_ORDER_SHORT_CIRCUIT_CONTRACT,
        }


@dataclass(frozen=True)
class ProducerCodeBinding:
    """Byte identities of the code which produced an immutable sidecar.

    This is intentionally separate from :class:`RuntimeCodeBinding`: the
    latter pins the submitted r195 behavior being replayed, while this pins
    the r212 reconstruction glue which decides when that behavior is invoked.
    Paths are deliberately omitted from the serialized form so a sidecar can
    travel from its archive-owning host to Blackwell unchanged.
    """

    guide2vec_public_routes_sha256: str
    materializer_cli_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "schema": PRODUCER_CODE_SCHEMA,
            "guide2vec_public_routes_sha256": self.guide2vec_public_routes_sha256,
            "materializer_cli_sha256": self.materializer_cli_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, label: str) -> "ProducerCodeBinding":
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "guide2vec_public_routes_sha256",
            "materializer_cli_sha256",
        }:
            raise PublicRouteReconstructionError(
                f"{label} does not have the exact r212 producer-code schema"
            )
        if value.get("schema") != PRODUCER_CODE_SCHEMA:
            raise PublicRouteReconstructionError(
                f"{label} producer-code schema differs from r212"
            )
        return cls(
            guide2vec_public_routes_sha256=_require_sha256(
                value.get("guide2vec_public_routes_sha256"),
                label=f"{label} guide2vec-public-routes SHA-256",
            ),
            materializer_cli_sha256=_require_sha256(
                value.get("materializer_cli_sha256"),
                label=f"{label} materializer CLI SHA-256",
            ),
        )


@dataclass(frozen=True)
class BoundProducerCode:
    """Locally verified producer source files plus their portable identity."""

    binding: ProducerCodeBinding
    guide2vec_public_routes_path: Path
    guide2vec_public_routes_stat_identity: tuple[int, int, int, int, int]
    materializer_cli_path: Path
    materializer_cli_stat_identity: tuple[int, int, int, int, int]

    def assert_unchanged(self, *, verify_sha256: bool = True) -> None:
        for label, path, expected_stat, expected_sha256 in (
            (
                "guide2vec public-route producer module",
                self.guide2vec_public_routes_path,
                self.guide2vec_public_routes_stat_identity,
                self.binding.guide2vec_public_routes_sha256,
            ),
            (
                "guide2vec public-route materializer CLI",
                self.materializer_cli_path,
                self.materializer_cli_stat_identity,
                self.binding.materializer_cli_sha256,
            ),
        ):
            if _stat_identity(path) != expected_stat or (
                verify_sha256 and _sha256_file(path) != expected_sha256
            ):
                raise PublicRouteReconstructionError(
                    f"r212 {label} changed during sidecar materialization"
                )


def _bind_runtime_code(
    *,
    submission_bundle_path: Path,
    submission_bundle_sha256: str,
    submission_entrypoint_member: str,
    submission_entrypoint_sha256: str,
    public_matchup_router_member: str,
    public_matchup_router_sha256: str,
) -> tuple[RuntimeCodeBinding, tuple[int, int, int, int, int]]:
    """Hash-bind the exact r195 entrypoint and router implementation.

    The route algorithm alone is insufficient: the submission entrypoint
    short-circuits IsFirst before PolicyAgent, and the router owns the
    consecutive-observation semantics.  Bind both exact files locally before
    scanning raw data; sidecars carry only these host-independent digests.
    """

    bundle = Path(submission_bundle_path).expanduser().resolve()
    expected_bundle = _require_sha256(
        submission_bundle_sha256, label="r212 r195 submission bundle SHA-256"
    )
    expected_entrypoint = _require_sha256(
        submission_entrypoint_sha256, label="r212 r195 submission entrypoint SHA-256"
    )
    expected_router = _require_sha256(
        public_matchup_router_sha256, label="r212 r195 public router SHA-256"
    )
    members = (
        ("submission entrypoint", str(submission_entrypoint_member), expected_entrypoint),
        ("public matchup router", str(public_matchup_router_member), expected_router),
    )
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    bundle_stat = _stat_identity(bundle)
    if _sha256_file(bundle) != expected_bundle:
        raise PublicRouteReconstructionError(
            "r212 r195 submission bundle SHA-256 differs before route reconstruction"
        )
    try:
        with tarfile.open(bundle, "r:gz") as archive:
            for label, member_name, expected_sha256 in members:
                if not member_name.startswith("./") or ".." in Path(member_name).parts:
                    raise PublicRouteReconstructionError(
                        f"r212 r195 {label} member is not canonical"
                    )
                matched = [info for info in archive.getmembers() if info.name == member_name]
                if len(matched) != 1 or not matched[0].isfile():
                    raise PublicRouteReconstructionError(
                        f"r212 r195 submission bundle lacks one exact {label} member"
                    )
                handle = archive.extractfile(matched[0])
                if handle is None or _sha256_bytes(handle.read()) != expected_sha256:
                    raise PublicRouteReconstructionError(
                        f"r212 r195 {label} member SHA-256 differs from its pinned identity"
                    )
    except (OSError, tarfile.TarError) as exc:
        raise PublicRouteReconstructionError(
            "r212 r195 submission bundle is not a readable gzip tar archive"
        ) from exc
    if _stat_identity(bundle) != bundle_stat:
        raise PublicRouteReconstructionError(
            "r212 r195 route-runtime source changed during SHA-256 verification"
        )
    return (
        RuntimeCodeBinding(
            submission_bundle_sha256=expected_bundle,
            submission_entrypoint_member=str(submission_entrypoint_member),
            submission_entrypoint_sha256=expected_entrypoint,
            public_matchup_router_member=str(public_matchup_router_member),
            public_matchup_router_sha256=expected_router,
        ),
        bundle_stat,
    )


def verify_imported_runtime_public_router(*, expected_sha256: str) -> Path:
    """Prove the class used for replay is the exact packaged r195 router."""

    expected = _require_sha256(
        expected_sha256, label="r212 expected imported public router SHA-256"
    )
    source = inspect.getsourcefile(RuntimePublicMatchupRouter)
    if not source:
        raise PublicRouteReconstructionError(
            "r212 cannot locate the imported RuntimePublicMatchupRouter source"
        )
    path = Path(source).resolve()
    if RuntimePublicMatchupRouter.__module__ != "poke_bot.public_matchup_router" or not path.is_file():
        raise PublicRouteReconstructionError(
            "r212 imported RuntimePublicMatchupRouter does not have the canonical module identity"
        )
    if _sha256_file(path) != expected:
        raise PublicRouteReconstructionError(
            "r212 imported RuntimePublicMatchupRouter source differs from the exact r195 bundle member"
        )
    return path


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = Path(path).stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _require_sha256(value: Any, *, label: str) -> str:
    result = str(value or "")
    if _SHA256_RE.fullmatch(result) is None:
        raise PublicRouteReconstructionError(f"{label} is not an exact SHA-256")
    return result


def bind_sidecar_producer_code(*, materializer_cli_path: Path) -> BoundProducerCode:
    """Bind the exact module and CLI bytes used to create a sidecar.

    The raw-resolver code is imported from this module, so hashing an arbitrary
    caller-supplied path would not prove what ran.  Bind ``__file__`` itself,
    and require the dedicated materializer script by its canonical filename.
    Both inode/stat identities and file hashes are rechecked before a sidecar
    is finalized.
    """

    module_path = Path(__file__).resolve()
    cli_path = Path(materializer_cli_path).expanduser().resolve()
    if not module_path.is_file():
        raise PublicRouteReconstructionError(
            "r212 cannot locate the imported guide2vec public-route producer module"
        )
    if (
        not cli_path.is_file()
        or cli_path.name != "materialize_alakazam_guide2vec_r212_public_routes.py"
    ):
        raise PublicRouteReconstructionError(
            "r212 materializer CLI is not the canonical r212 public-route script"
        )
    module_stat = _stat_identity(module_path)
    cli_stat = _stat_identity(cli_path)
    binding = ProducerCodeBinding(
        guide2vec_public_routes_sha256=_sha256_file(module_path),
        materializer_cli_sha256=_sha256_file(cli_path),
    )
    if _stat_identity(module_path) != module_stat or _stat_identity(cli_path) != cli_stat:
        raise PublicRouteReconstructionError(
            "r212 public-route producer code changed during SHA-256 verification"
        )
    return BoundProducerCode(
        binding=binding,
        guide2vec_public_routes_path=module_path,
        guide2vec_public_routes_stat_identity=module_stat,
        materializer_cli_path=cli_path,
        materializer_cli_stat_identity=cli_stat,
    )


def _require_exact_int(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise PublicRouteReconstructionError(f"{label} must be an exact integer")
    return int(value)


def _sparse_f32_equal(left: Any, right: Any) -> bool:
    """Compare a compact ``array('f')`` vector to a freshly rebuilt vector."""

    try:
        if (
            list(left.index) != list(right.index)
            or list(left.offset) != list(right.offset)
            or int(left.pos) != int(right.pos)
            or len(left.value) != len(right.value)
        ):
            return False
        return all(
            struct.pack("<f", float(a)) == struct.pack("<f", float(b))
            for a, b in zip(left.value, right.value)
        )
    except (AttributeError, TypeError, ValueError, struct.error):
        return False


def _sparse_f32_identity(value: Any) -> dict[str, Any]:
    """Canonical byte identity for a frozen sparse feature vector."""

    try:
        index = list(value.index)
        offset = list(value.offset)
        pos = int(value.pos)
        raw_values = b"".join(
            struct.pack("<f", float(item)) for item in value.value
        )
    except (AttributeError, TypeError, ValueError, struct.error) as exc:
        raise PublicRouteReconstructionError(
            "r212 compact feature vector has no exact f32 sparse identity"
        ) from exc
    if (
        any(type(item) is not int for item in index)
        or any(type(item) is not int for item in offset)
        or pos < 0
    ):
        raise PublicRouteReconstructionError(
            "r212 compact sparse feature identity has malformed integer fields"
        )
    return {
        "index": index,
        "offset": offset,
        "pos": pos,
        "f32_le_hex": raw_values.hex(),
    }


def compact_alignment_sha256(
    *,
    env_steps: Sequence[int],
    boards: Sequence[Any],
    action_tokens: Sequence[Any],
) -> str:
    """Hash the exact per-decision board/action-token feature identity."""

    if not (
        len(env_steps) == len(boards) == len(action_tokens)
        and env_steps
        and all(type(step) is int and step >= 0 for step in env_steps)
    ):
        raise PublicRouteReconstructionError(
            "r212 compact alignment digest has malformed decision vectors"
        )
    return _sha256_bytes(
        _canonical_json(
            [
                {
                    "env_step": int(step),
                    "board": _sparse_f32_identity(board),
                    "action_token": _sparse_f32_identity(action_token),
                }
                for step, board, action_token in zip(env_steps, boards, action_tokens)
            ]
        )
    )


def _sequence_compact_alignment_sha256(sequence: GameSequence) -> str:
    decisions = list(sequence.decisions)
    if any(getattr(decision, "action_token", None) is None for decision in decisions):
        raise PublicRouteReconstructionError(
            "r212 compact sequence lacks an exact shifted action-token feature"
        )
    return compact_alignment_sha256(
        env_steps=[int(decision.env_step) for decision in decisions],
        boards=[decision.board for decision in decisions],
        action_tokens=[decision.action_token for decision in decisions],
    )


def _feature_header(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("rb") as handle:
            header = pickle.load(handle)
    except (EOFError, OSError, pickle.UnpicklingError) as exc:
        raise PublicRouteReconstructionError(
            f"cannot read r212 feature-shard header: {path}"
        ) from exc
    if not isinstance(header, dict):
        raise PublicRouteReconstructionError(
            f"r212 feature-shard header is not an object: {path}"
        )
    if (
        header.get("format") != SHARD_FORMAT
        or int(header.get("format_version", -1)) != SHARD_FORMAT_VERSION
    ):
        raise PublicRouteReconstructionError(
            f"r212 feature-shard header format drifted: {path}"
        )
    return header


def _feature_header_raw_archive_contract(
    *, source_date: str, feature_shard_path: Path
) -> tuple[str, str]:
    """Return the header's exact raw archive name and content digest.

    Feature-shard v1 records no archive byte count.  The archive SHA-256 is
    therefore the authoritative immutable identity; callers may additionally
    carry a byte count as a transport/audit invariant, but must never pretend
    that it was authored by the protected feature header.
    """

    day = str(source_date)
    if _SAFE_DAY_RE.fullmatch(day) is None:
        raise PublicRouteReconstructionError("r212 source date is malformed")
    header = _feature_header(Path(feature_shard_path))
    source_dates = tuple(str(value) for value in header.get("source_dates") or ())
    if source_dates != (day,):
        raise PublicRouteReconstructionError(
            "r212 feature shard does not bind exactly one requested source date"
        )
    archive_name = str(header.get("source_archive") or "")
    expected_name = f"pokemon-tcg-ai-battle-episodes-{day}.zip"
    if (
        archive_name != expected_name
        or Path(archive_name).name != archive_name
        or archive_name.startswith(".")
    ):
        raise PublicRouteReconstructionError(
            "r212 feature header raw archive name is not the exact daily archive"
        )
    archive_sha256 = _require_sha256(
        header.get("source_archive_sha256"),
        label="r212 feature-header raw archive SHA-256",
    )
    return archive_name, archive_sha256


@dataclass(frozen=True)
class RawArchiveBinding:
    """One header-bound immutable raw daily episode archive."""

    source_date: str
    archive_name: str
    path: Path
    sha256: str
    byte_count: int
    stat_identity: tuple[int, int, int, int, int]

    def as_dict(self, *, include_path: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "source_date": self.source_date,
            "archive_name": self.archive_name,
            "sha256": self.sha256,
            "bytes": self.byte_count,
        }
        if include_path:
            value["path"] = str(self.path)
        return value

    def assert_unchanged(self) -> None:
        if _stat_identity(self.path) != self.stat_identity:
            raise PublicRouteReconstructionError(
                f"raw r212 source archive changed during reconstruction: {self.path}"
            )


def bind_raw_archive(
    *,
    source_date: str,
    feature_shard_path: Path,
    raw_archive_root: Path,
) -> RawArchiveBinding:
    """Open the source header and bind its one raw daily ZIP by SHA-256."""

    day = str(source_date)
    archive_name, expected_sha256 = _feature_header_raw_archive_contract(
        source_date=day, feature_shard_path=Path(feature_shard_path)
    )
    root = Path(raw_archive_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"r212 raw archive root is unavailable: {root}")
    path = (root / archive_name).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PublicRouteReconstructionError("r212 raw archive escaped its root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    before = _stat_identity(path)
    actual_sha256 = _sha256_file(path)
    after = _stat_identity(path)
    if before != after:
        raise PublicRouteReconstructionError(
            f"r212 raw archive changed during SHA-256 verification: {path}"
        )
    if actual_sha256 != expected_sha256:
        raise PublicRouteReconstructionError(
            "r212 raw archive SHA-256 differs from the protected feature header: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    return RawArchiveBinding(
        source_date=day,
        archive_name=archive_name,
        path=path,
        sha256=actual_sha256,
        byte_count=int(path.stat().st_size),
        stat_identity=after,
    )


@dataclass(frozen=True)
class SequencePublicRoutes:
    """One raw-verified compact decision route projection."""

    episode_id: str
    seat: int
    env_steps: tuple[int, ...]
    routes: tuple[int, ...]
    compact_alignment_sha256: str
    raw_member_sha256: str
    active_observations: int
    turn_order_short_circuits: int
    turn_order_short_circuit_env_steps: tuple[int, ...]
    # r195 invokes ``policy.reset_game()`` on an ACTIVE/select-null entrypoint
    # call.  Keep the full raw-prefix reset evidence, including resets before
    # a last-320 compact window, so a consumer can audit causal state.
    game_resets: int = 0
    game_reset_env_steps: tuple[int, ...] = ()

    def as_sidecar_row(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_SIDECAR_ROW_SCHEMA,
            "episode_id": self.episode_id,
            "seat": self.seat,
            "env_steps": list(self.env_steps),
            "routes": list(self.routes),
            "compact_alignment_sha256": self.compact_alignment_sha256,
            "raw_member_sha256": self.raw_member_sha256,
            "active_observations": self.active_observations,
            "turn_order_short_circuits": self.turn_order_short_circuits,
            "turn_order_short_circuit_env_steps": list(
                self.turn_order_short_circuit_env_steps
            ),
            "game_resets": self.game_resets,
            "game_reset_env_steps": list(self.game_reset_env_steps),
        }


def _public_route_identity(
    episode_id_value: Any, seat_value: Any, env_step_values: Sequence[Any]
) -> tuple[str, int, tuple[int, ...]]:
    """Validate the identity shared by compact and raw-classifier projections."""

    episode_id = str(episode_id_value or "")
    if not episode_id or Path(episode_id).name != episode_id or "/" in episode_id:
        raise PublicRouteReconstructionError("r212 compact sequence has an unsafe episode ID")
    seat = seat_value
    if type(seat) is not int or seat not in (0, 1):
        raise PublicRouteReconstructionError("r212 compact sequence has an invalid seat")
    env_steps = tuple(env_step_values)
    if not env_steps:
        raise PublicRouteReconstructionError("r212 compact sequence has no decisions")
    if (
        any(type(value) is not int or int(value) < 0 for value in env_steps)
        or tuple(sorted(env_steps)) != env_steps
        or len(set(env_steps)) != len(env_steps)
    ):
        raise PublicRouteReconstructionError(
            "r212 compact sequence does not have strictly ordered exact env steps"
        )
    return episode_id, int(seat), tuple(int(value) for value in env_steps)


def _sequence_identity(sequence: GameSequence) -> tuple[str, int, tuple[int, ...]]:
    decisions = list(getattr(sequence, "decisions", ()) or ())
    episode_id, seat, env_steps = _public_route_identity(
        getattr(sequence, "episode_id", ""),
        getattr(sequence, "seat", None),
        [getattr(decision, "env_step", None) for decision in decisions],
    )
    deck = list(getattr(sequence, "deck", ()) or ())
    if len(deck) != 60 or any(type(card) is not int for card in deck):
        raise PublicRouteReconstructionError("r212 compact sequence deck is not exact")
    return episode_id, seat, env_steps


def _active_agent_observation(entry: Any) -> Mapping[str, Any] | None:
    """Return an ACTIVE submitted-agent invocation before r195 entrypoint gates.

    Inactive archived rows can carry stale observations and must never affect
    router state.  The exact r195 entrypoint next tests ``select is None`` and
    the IsFirst context *before* converting to an Observation or requiring
    ``current.yourIndex``.  Preserve that branch ordering in the caller; only
    an ordinary selected policy call is validated for its acting seat.
    """

    if not isinstance(entry, Mapping):
        raise PublicRouteReconstructionError("r212 raw episode seat row is not an object")
    if entry.get("status") != "ACTIVE":
        return None
    observation = entry.get("observation")
    if not isinstance(observation, Mapping):
        raise PublicRouteReconstructionError(
            "r212 ACTIVE raw episode row has no observation object"
        )
    return observation


def _require_selected_policy_acting_seat(
    observation: Mapping[str, Any], *, seat: int
) -> None:
    """Validate only the ordinary r195 PolicyAgent invocation path."""

    current = observation.get("current")
    if not isinstance(current, Mapping):
        raise PublicRouteReconstructionError(
            "r212 ACTIVE raw policy observation has no current state"
        )
    try:
        your_index = int(current.get("yourIndex"))
    except (TypeError, ValueError) as exc:
        raise PublicRouteReconstructionError(
            "r212 ACTIVE raw policy observation has no exact acting seat"
        ) from exc
    if your_index != int(seat):
        raise PublicRouteReconstructionError(
            "r212 ACTIVE raw policy observation has a mismatched acting seat"
        )


def _submission_turn_order_short_circuit(observation: Mapping[str, Any]) -> bool:
    """Match r195's torch-free ``submission.main._turn_order_choice`` gate.

    The submitted entrypoint resolves an ``IsFirst`` prompt *before* it loads
    or invokes ``PolicyAgent``.  Such a row can be ACTIVE and have ``select``
    data, yet it is not a public-router observation.  Keep this deliberately
    literal instead of using feature enum normalization: r195 accepts only
    exact numeric ``41`` or a punctuation/case-insensitive ``isfirst`` token.
    The entrypoint short-circuits even when its options are malformed (then it
    returns its fail-closed empty action), so only the context predicate belongs
    here.
    """

    selection = observation.get("select") if isinstance(observation, Mapping) else None
    if not isinstance(selection, Mapping):
        return False
    context = selection.get("context")
    normalized = "".join(character for character in str(context).lower() if character.isalnum())
    return context == 41 or normalized == "isfirst"


def _raw_action_after(steps: Sequence[Any], *, env_step: int, seat: int) -> list[int]:
    next_index = int(env_step) + 1
    if next_index >= len(steps):
        raise PublicRouteReconstructionError(
            "r212 compact decision has no shifted raw action row"
        )
    row = steps[next_index]
    if not isinstance(row, list) or len(row) != 2 or not isinstance(row[seat], Mapping):
        raise PublicRouteReconstructionError(
            "r212 shifted raw action row is malformed"
        )
    raw_action = row[seat].get("action")
    if not isinstance(raw_action, list) or any(type(value) is not int for value in raw_action):
        raise PublicRouteReconstructionError(
            "r212 shifted raw action is not an exact integer list"
        )
    return [int(value) for value in raw_action]


def _assert_compact_alignment(
    sequence: GameSequence,
    *,
    decision_index: int,
    observation: Mapping[str, Any],
    raw_action: list[int],
) -> None:
    """Prove the raw ACTIVE input is the compact decision's actual input."""

    decision = sequence.decisions[decision_index]
    masked, auxiliary, report = _strip_opp_private(dict(observation))
    if (
        report.remasked
        or not report.ok
        or any(value is not None for value in auxiliary.values())
    ):
        raise PublicRouteReconstructionError(
            "r212 raw policy observation leaks opponent-private state"
        )
    try:
        rebuilt_board = features.build_board_tokens(masked, list(sequence.deck))
        rebuilt_action = features.build_option_tokens(masked, [raw_action])
    except Exception as exc:
        raise PublicRouteReconstructionError(
            "r212 raw policy input cannot rebuild its compact board/action features"
        ) from exc
    if not _sparse_f32_equal(rebuilt_board, decision.board):
        raise PublicRouteReconstructionError(
            "r212 raw ACTIVE observation does not match compact board features"
        )
    if decision.action_token is None or not _sparse_f32_equal(
        rebuilt_action, decision.action_token
    ):
        raise PublicRouteReconstructionError(
            "r212 shifted raw action does not match compact action-token features"
        )


def reconstruct_public_routes_from_raw_member(
    *,
    raw_member: bytes,
    episode_id: str,
    seat: int,
    env_steps: Sequence[int],
    compact_alignment_sha256: str,
    tree: PublicMatchupDecisionTree,
    allowed_physical_slots: Collection[int],
    target_validator: Any | None = None,
) -> SequencePublicRoutes:
    """Causally replay one raw member through the exact r195 router.

    ``target_validator`` is called only for compact target steps as
    ``(decision_index, observation, shifted_raw_action)``.  The direct
    feature-shard resolver uses it to prove board/action equality; raw
    classifier sidecar materialization uses it to prove the authoritative
    visual-recorded action.  The router scan itself is shared so the two paths
    cannot drift on ACTIVE, IsFirst, or consecutive-route semantics.
    """

    normalized_episode, normalized_seat, normalized_steps = _public_route_identity(
        episode_id, seat, env_steps
    )
    alignment_sha256 = _require_sha256(
        compact_alignment_sha256, label="r212 compact board/action alignment SHA-256"
    )
    raw_digest = _sha256_bytes(raw_member)
    try:
        payload = json.loads(raw_member)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicRouteReconstructionError(
            "r212 raw episode member cannot be decoded"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or episode_id_of(dict(payload)) != normalized_episode
    ):
        raise PublicRouteReconstructionError(
            "r212 raw episode member identity differs from its compact sequence"
        )
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or normalized_steps[-1] >= len(raw_steps):
        raise PublicRouteReconstructionError(
            "r212 compact env step is outside the raw episode history"
        )
    slots = list(allowed_physical_slots)
    if any(type(value) is not int or value < 0 for value in slots):
        raise PublicRouteReconstructionError(
            "r212 public runtime tree has malformed allowed physical slots"
        )
    allowed = frozenset(slots)
    if not allowed:
        raise PublicRouteReconstructionError("r212 public runtime tree has no allowed slots")
    target_indexes = {value: index for index, value in enumerate(normalized_steps)}
    router = RuntimePublicMatchupRouter(tree)
    routes: list[int | None] = [None] * len(normalized_steps)
    active_observations = 0
    turn_order_short_circuits = 0
    turn_order_short_circuit_env_steps: list[int] = []
    game_resets = 0
    game_reset_env_steps: list[int] = []
    for env_step in range(normalized_steps[-1] + 1):
        row = raw_steps[env_step]
        if not isinstance(row, list) or len(row) != 2:
            raise PublicRouteReconstructionError(
                "r212 raw episode step is not a two-seat row"
            )
        observation = _active_agent_observation(row[normalized_seat])
        if observation is None:
            if env_step in target_indexes:
                raise PublicRouteReconstructionError(
                    "r212 compact decision is not an ACTIVE raw policy observation"
                )
            continue
        if observation.get("select") is None:
            # Exact r195 entrypoint behavior: after the IsFirst gate (which
            # cannot match a null select), it invokes ``policy.reset_game()``
            # and returns the deck.  Reusing a router over the next game would
            # incorrectly carry public cards and consecutive confirmations.
            router.reset_for_new_game()
            game_resets += 1
            game_reset_env_steps.append(env_step)
            if env_step >= normalized_steps[0]:
                raise PublicRouteReconstructionError(
                    "r212 select-null r195 game reset would span compact temporal history"
                )
            continue
        if _submission_turn_order_short_circuit(observation):
            turn_order_short_circuits += 1
            if env_step in target_indexes:
                turn_order_short_circuit_env_steps.append(env_step)
        else:
            _require_selected_policy_acting_seat(
                observation, seat=normalized_seat
            )
            router.observe(dict(observation))
            active_observations += 1
        decision_index = target_indexes.get(env_step)
        if decision_index is None:
            continue
        raw_action = _raw_action_after(raw_steps, env_step=env_step, seat=normalized_seat)
        if target_validator is not None:
            target_validator(decision_index, observation, raw_action)
        route = int(router.candidate_model_route)
        if route != UNKNOWN_ROUTE and route not in allowed:
            raise PublicRouteReconstructionError(
                "r212 public runtime tree emitted a route outside its V6 binding"
            )
        routes[decision_index] = route
    if any(route is None for route in routes):
        raise PublicRouteReconstructionError(
            "r212 raw prefix did not emit a route for every compact decision"
        )
    return SequencePublicRoutes(
        episode_id=normalized_episode,
        seat=normalized_seat,
        env_steps=normalized_steps,
        routes=tuple(int(route) for route in routes if route is not None),
        compact_alignment_sha256=alignment_sha256,
        raw_member_sha256=raw_digest,
        active_observations=active_observations,
        turn_order_short_circuits=turn_order_short_circuits,
        turn_order_short_circuit_env_steps=tuple(turn_order_short_circuit_env_steps),
        game_resets=game_resets,
        game_reset_env_steps=tuple(game_reset_env_steps),
    )


class RawPublicRouteResolver:
    """One bounded raw-ZIP replay resolver for a single protected source day."""

    def __init__(
        self,
        *,
        source_date: str,
        feature_shard_path: Path,
        feature_shard_sha256: str,
        raw_archive_root: Path,
        matchup_tree_path: Path,
        matchup_tree_sha256: str,
        submission_bundle_path: Path,
        submission_bundle_sha256: str,
        submission_entrypoint_member: str,
        submission_entrypoint_sha256: str,
        public_matchup_router_member: str,
        public_matchup_router_sha256: str,
        allowed_physical_slots: Collection[int],
        sidecar_producer_code: BoundProducerCode | None = None,
    ) -> None:
        self.source_date = str(source_date)
        self.feature_shard_path = Path(feature_shard_path).expanduser().resolve()
        self.feature_shard_sha256 = _require_sha256(
            feature_shard_sha256, label="r212 source feature shard SHA-256"
        )
        if not self.feature_shard_path.is_file():
            raise FileNotFoundError(self.feature_shard_path)
        self._feature_shard_stat_identity = _stat_identity(self.feature_shard_path)
        if _sha256_file(self.feature_shard_path) != self.feature_shard_sha256:
            raise PublicRouteReconstructionError(
                "r212 source feature shard SHA-256 differs before raw route reconstruction"
            )
        if _stat_identity(self.feature_shard_path) != self._feature_shard_stat_identity:
            raise PublicRouteReconstructionError(
                "r212 source feature shard changed during SHA-256 verification"
            )
        self.submission_bundle_path = Path(submission_bundle_path).expanduser().resolve()
        (
            self.runtime_code,
            self._submission_bundle_stat_identity,
        ) = _bind_runtime_code(
            submission_bundle_path=self.submission_bundle_path,
            submission_bundle_sha256=submission_bundle_sha256,
            submission_entrypoint_member=submission_entrypoint_member,
            submission_entrypoint_sha256=submission_entrypoint_sha256,
            public_matchup_router_member=public_matchup_router_member,
            public_matchup_router_sha256=public_matchup_router_sha256,
        )
        self._imported_router_source_path = verify_imported_runtime_public_router(
            expected_sha256=self.runtime_code.public_matchup_router_sha256
        )
        self._imported_router_source_stat_identity = _stat_identity(
            self._imported_router_source_path
        )
        if (
            _stat_identity(self._imported_router_source_path)
            != self._imported_router_source_stat_identity
            or _sha256_file(self._imported_router_source_path)
            != self.runtime_code.public_matchup_router_sha256
        ):
            raise PublicRouteReconstructionError(
                "r212 imported RuntimePublicMatchupRouter source changed during verification"
            )
        if sidecar_producer_code is not None and not isinstance(
            sidecar_producer_code, BoundProducerCode
        ):
            raise PublicRouteReconstructionError(
                "r212 sidecar producer binding has an invalid type"
            )
        self._sidecar_producer_code = sidecar_producer_code
        if self._sidecar_producer_code is not None:
            self._sidecar_producer_code.assert_unchanged()
        self.archive = bind_raw_archive(
            source_date=self.source_date,
            feature_shard_path=self.feature_shard_path,
            raw_archive_root=raw_archive_root,
        )
        self.matchup_tree_path = Path(matchup_tree_path).expanduser().resolve()
        self.matchup_tree_sha256 = _require_sha256(
            matchup_tree_sha256, label="r212 runtime public-tree SHA-256"
        )
        if not self.matchup_tree_path.is_file():
            raise FileNotFoundError(self.matchup_tree_path)
        if _sha256_file(self.matchup_tree_path) != self.matchup_tree_sha256:
            raise PublicRouteReconstructionError(
                "r212 public matchup tree SHA-256 changed before route reconstruction"
            )
        try:
            self.tree = PublicMatchupDecisionTree.from_path(
                self.matchup_tree_path, require_runtime_enabled=True
            )
        except (OSError, TypeError, ValueError) as exc:
            raise PublicRouteReconstructionError(
                "r212 public matchup tree is not an enabled runtime tree"
            ) from exc
        if self.tree.digest != self.matchup_tree_sha256:
            raise PublicRouteReconstructionError(
                "r212 loaded runtime tree digest differs from its pinned identity"
            )
        if int(self.tree.unknown_route) != UNKNOWN_ROUTE:
            raise PublicRouteReconstructionError(
                "r212 runtime tree bypass route differs from submitted r195 semantics"
            )
        slots: list[int] = []
        for value in allowed_physical_slots:
            if type(value) is not int or int(value) < 0:
                raise PublicRouteReconstructionError(
                    "r212 allowed public route is not an exact physical slot"
                )
            slots.append(int(value))
        self.allowed_physical_slots = frozenset(slots)
        if not self.allowed_physical_slots:
            raise PublicRouteReconstructionError("r212 has no runtime-accepted V6 slots")
        actual_allowed = frozenset(
            int(self.tree.route_physical_slots[self.tree.targets.index(target)])
            for target in self.tree.runtime_accepted_archetype_ids
        )
        if actual_allowed != self.allowed_physical_slots:
            raise PublicRouteReconstructionError(
                "r212 allowed V6 slots differ from the exact runtime public tree"
            )
        self._zip: zipfile.ZipFile | None = None
        self._members: dict[str, zipfile.ZipInfo] = {}
        self._seen_keys: set[tuple[str, int]] = set()
        self._member_route_hasher = hashlib.sha256()
        self._records = 0
        self._decisions = 0
        self._active_observations = 0
        self._turn_order_short_circuits = 0
        self._game_resets = 0
        self._routed_decisions = 0
        self._bypassed_decisions = 0
        self._closed = False

    def _assert_runtime_code_unchanged(self, *, verify_sha256: bool = False) -> None:
        """Reject source replacement without hashing the r195 tar per record.

        A source day contains tens of thousands of compact sequences.  Its
        117 MiB submitted tar is immutable-bound at construction and then
        re-hashed at projection/close, while the hot per-sequence path checks
        stable inode/stat identities only.  This preserves fail-closed change
        detection without turning route reconstruction into multi-terabyte IO.
        """

        if (
            _stat_identity(self.submission_bundle_path)
            != self._submission_bundle_stat_identity
            or (
                verify_sha256
                and _sha256_file(self.submission_bundle_path)
                != self.runtime_code.submission_bundle_sha256
            )
        ):
            raise PublicRouteReconstructionError(
                "r212 r195 route-runtime source changed during reconstruction"
            )
        if (
            _stat_identity(self._imported_router_source_path)
            != self._imported_router_source_stat_identity
            or (
                verify_sha256
                and _sha256_file(self._imported_router_source_path)
                != self.runtime_code.public_matchup_router_sha256
            )
        ):
            raise PublicRouteReconstructionError(
                "r212 imported RuntimePublicMatchupRouter source changed during reconstruction"
            )
        if self._sidecar_producer_code is not None:
            self._sidecar_producer_code.assert_unchanged(
                verify_sha256=verify_sha256
            )

    @classmethod
    def open(cls, **kwargs: Any) -> "RawPublicRouteResolver":
        resolver = cls(**kwargs)
        resolver._open_archive()
        return resolver

    def __enter__(self) -> "RawPublicRouteResolver":
        if self._zip is None:
            self._open_archive()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _open_archive(self) -> None:
        if self._zip is not None:
            return
        self.archive.assert_unchanged()
        try:
            archive = zipfile.ZipFile(self.archive.path, "r")
        except (OSError, zipfile.BadZipFile) as exc:
            raise PublicRouteReconstructionError(
                f"r212 raw archive is not a valid ZIP: {self.archive.path}"
            ) from exc
        members: dict[str, zipfile.ZipInfo] = {}
        try:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = str(info.filename)
                # Official daily ZIPs include one ``manifest.csv`` beside the
                # flat episode members.  It is checksum-bound by the archive
                # as a whole but is not an episode identity.
                if not name.endswith(".json"):
                    continue
                if Path(name).name != name:
                    raise PublicRouteReconstructionError(
                        "r212 raw ZIP has a noncanonical episode member name"
                    )
                if name in members:
                    raise PublicRouteReconstructionError(
                        "r212 raw ZIP has a duplicate episode member"
                    )
                members[name] = info
            if not members:
                raise PublicRouteReconstructionError("r212 raw ZIP has no episode members")
        except BaseException:
            archive.close()
            raise
        self._zip = archive
        self._members = members

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._zip is not None:
            self._zip.close()
            self._zip = None
        self.archive.assert_unchanged()
        self._assert_runtime_code_unchanged(verify_sha256=True)
        if _stat_identity(self.feature_shard_path) != self._feature_shard_stat_identity:
            raise PublicRouteReconstructionError(
                "r212 source feature shard changed during route reconstruction"
            )

    def header(self) -> dict[str, Any]:
        header: dict[str, Any] = {
            "schema": ROUTE_SIDECAR_HEADER_SCHEMA,
            "format": ROUTE_SIDECAR_FORMAT,
            "source_date": self.source_date,
            "source_feature_shard_sha256": self.feature_shard_sha256,
            "raw_archive": self.archive.as_dict(include_path=False),
            "runtime_public_tree_sha256": self.matchup_tree_sha256,
            "runtime_code": self.runtime_code.as_dict(),
            "allowed_physical_slots": sorted(self.allowed_physical_slots),
            "algorithm": ROUTE_ALGORITHM,
            "alignment_contract": ALIGNMENT_CONTRACT,
            "compact_source_routes_ignored": True,
            "oracle_route_used": False,
        }
        if self._sidecar_producer_code is not None:
            header["producer_code"] = self._sidecar_producer_code.binding.as_dict()
        return header

    def resolve_sequence(self, sequence: GameSequence) -> SequencePublicRoutes:
        if self._closed:
            raise PublicRouteReconstructionError("r212 raw route resolver is closed")
        self._open_archive()
        assert self._zip is not None
        if _stat_identity(self.feature_shard_path) != self._feature_shard_stat_identity:
            raise PublicRouteReconstructionError(
                "r212 source feature shard changed during route reconstruction"
            )
        self._assert_runtime_code_unchanged()
        episode_id, seat, env_steps = _sequence_identity(sequence)
        key = (episode_id, seat)
        if key in self._seen_keys:
            raise PublicRouteReconstructionError(
                "r212 feature source repeats an episode/seat route identity"
            )
        member_name = f"{episode_id}.json"
        member = self._members.get(member_name)
        if member is None:
            raise PublicRouteReconstructionError(
                f"r212 raw ZIP lacks compact episode member: {member_name}"
            )
        try:
            raw = self._zip.read(member)
            # Decode once here for an archive-specific diagnostic; the shared
            # scanner decodes again so the feature-free query path has exactly
            # the same parsing boundary.
            json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise PublicRouteReconstructionError(
                f"r212 raw episode member cannot be read: {member_name}"
            ) from exc
        # Use the common raw scanner also used by the feature-free sidecar
        # query producer.  Only this direct path adds a sparse board/action
        # proof for each compact target.
        def validate_target(
            decision_index: int,
            observation: Mapping[str, Any],
            raw_action: list[int],
        ) -> None:
            _assert_compact_alignment(
                sequence,
                decision_index=decision_index,
                observation=observation,
                raw_action=raw_action,
            )

        result = reconstruct_public_routes_from_raw_member(
            raw_member=raw,
            episode_id=episode_id,
            seat=seat,
            env_steps=env_steps,
            compact_alignment_sha256=_sequence_compact_alignment_sha256(sequence),
            tree=self.tree,
            allowed_physical_slots=self.allowed_physical_slots,
            target_validator=validate_target,
        )
        row = result.as_sidecar_row()
        self._member_route_hasher.update(_canonical_json(row))
        self._seen_keys.add(key)
        self._records += 1
        self._decisions += len(result.routes)
        self._active_observations += result.active_observations
        self._turn_order_short_circuits += result.turn_order_short_circuits
        self._game_resets += result.game_resets
        self._routed_decisions += sum(
            int(route != UNKNOWN_ROUTE) for route in result.routes
        )
        self._bypassed_decisions += sum(
            int(route == UNKNOWN_ROUTE) for route in result.routes
        )
        return result

    def projection(self, *, expected_records: int) -> dict[str, Any]:
        expected = _require_exact_int(expected_records, label="r212 expected source records")
        if expected <= 0 or self._records != expected:
            raise PublicRouteReconstructionError(
                "r212 raw public-route records do not match the protected source day"
            )
        if self._decisions <= 0 or self._routed_decisions + self._bypassed_decisions != self._decisions:
            raise PublicRouteReconstructionError("r212 raw public-route accounting drifted")
        self.archive.assert_unchanged()
        self._assert_runtime_code_unchanged(verify_sha256=True)
        if _stat_identity(self.feature_shard_path) != self._feature_shard_stat_identity:
            raise PublicRouteReconstructionError(
                "r212 source feature shard changed before route projection"
            )
        projection: dict[str, Any] = {
            "schema": ROUTE_RECONSTRUCTION_SCHEMA,
            "source_date": self.source_date,
            "source_feature_shard_sha256": self.feature_shard_sha256,
            "raw_archive": self.archive.as_dict(include_path=False),
            "runtime_public_tree_sha256": self.matchup_tree_sha256,
            "runtime_code": self.runtime_code.as_dict(),
            "allowed_physical_slots": sorted(self.allowed_physical_slots),
            "algorithm": ROUTE_ALGORITHM,
            "alignment_contract": ALIGNMENT_CONTRACT,
            "records": self._records,
            "decisions": self._decisions,
            "active_observations": self._active_observations,
            "turn_order_short_circuits": self._turn_order_short_circuits,
            "game_resets": self._game_resets,
            "routed_decisions": self._routed_decisions,
            "bypassed_decisions": self._bypassed_decisions,
            "member_route_sha256": "sha256:" + self._member_route_hasher.hexdigest(),
            "compact_source_routes_ignored": True,
            "oracle_route_used": False,
        }
        if self._sidecar_producer_code is not None:
            projection["producer_code"] = self._sidecar_producer_code.binding.as_dict()
        return projection


class SidecarPublicRouteResolver:
    """Sequential validator/reader for an immutable raw-route day sidecar."""

    def __init__(
        self,
        *,
        sidecar_path: Path,
        source_date: str,
        feature_shard_path: Path,
        feature_shard_sha256: str,
        expected_raw_archive: Mapping[str, Any],
        matchup_tree_sha256: str,
        submission_bundle_sha256: str,
        submission_entrypoint_member: str,
        submission_entrypoint_sha256: str,
        public_matchup_router_member: str,
        public_matchup_router_sha256: str,
        allowed_physical_slots: Collection[int],
        expected_sidecar_sha256: str,
        expected_producer_code: Mapping[str, Any],
    ) -> None:
        self.path = Path(sidecar_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.stat_identity = _stat_identity(self.path)
        self.sha256 = _sha256_file(self.path)
        if _stat_identity(self.path) != self.stat_identity:
            raise PublicRouteReconstructionError("r212 route sidecar changed during SHA-256 verification")
        if self.sha256 != _require_sha256(
            expected_sidecar_sha256, label="r212 declared route sidecar SHA-256"
        ):
            raise PublicRouteReconstructionError("r212 declared route sidecar SHA-256 mismatch")
        self.expected_producer_code = ProducerCodeBinding.from_mapping(
            expected_producer_code, label="r212 declared route sidecar producer code"
        )
        self.source_date = str(source_date)
        if _SAFE_DAY_RE.fullmatch(self.source_date) is None:
            raise PublicRouteReconstructionError("r212 sidecar source date is malformed")
        self.feature_shard_path = Path(feature_shard_path).expanduser().resolve()
        self.feature_shard_sha256 = _require_sha256(
            feature_shard_sha256, label="r212 source feature shard SHA-256"
        )
        if not self.feature_shard_path.is_file():
            raise FileNotFoundError(self.feature_shard_path)
        self._feature_shard_stat_identity = _stat_identity(self.feature_shard_path)
        if _sha256_file(self.feature_shard_path) != self.feature_shard_sha256:
            raise PublicRouteReconstructionError(
                "r212 source feature shard SHA-256 differs before sidecar consumption"
            )
        if _stat_identity(self.feature_shard_path) != self._feature_shard_stat_identity:
            raise PublicRouteReconstructionError(
                "r212 source feature shard changed during sidecar SHA-256 verification"
            )
        (
            self._expected_raw_archive_name,
            self._expected_raw_archive_sha256,
        ) = _feature_header_raw_archive_contract(
            source_date=self.source_date, feature_shard_path=self.feature_shard_path
        )
        if not isinstance(expected_raw_archive, Mapping):
            raise PublicRouteReconstructionError(
                "r212 sidecar lacks a declared exact raw archive binding"
            )
        declared_archive = dict(expected_raw_archive)
        if (
            declared_archive.get("source_date") != self.source_date
            or declared_archive.get("archive_name") != self._expected_raw_archive_name
            or _require_sha256(
                declared_archive.get("sha256"),
                label="r212 declared sidecar raw archive SHA-256",
            )
            != self._expected_raw_archive_sha256
            or type(declared_archive.get("bytes")) is not int
            or int(declared_archive["bytes"]) <= 0
            or set(declared_archive) != {"source_date", "archive_name", "sha256", "bytes"}
        ):
            raise PublicRouteReconstructionError(
                "r212 declared sidecar raw archive binding differs from the protected feature header"
            )
        self.expected_raw_archive = declared_archive
        self.matchup_tree_sha256 = _require_sha256(
            matchup_tree_sha256, label="r212 runtime public-tree SHA-256"
        )
        raw_slots = list(allowed_physical_slots)
        if any(type(value) is not int or value < 0 for value in raw_slots):
            raise PublicRouteReconstructionError("r212 sidecar allowed route set is invalid")
        slots = frozenset(raw_slots)
        if not slots:
            raise PublicRouteReconstructionError("r212 sidecar allowed route set is invalid")
        self.allowed_physical_slots = slots
        self.runtime_code = RuntimeCodeBinding(
            submission_bundle_sha256=_require_sha256(
                submission_bundle_sha256, label="r212 r195 submission bundle SHA-256"
            ),
            submission_entrypoint_member=self._require_canonical_tar_member(
                submission_entrypoint_member, label="r212 r195 submission entrypoint"
            ),
            submission_entrypoint_sha256=_require_sha256(
                submission_entrypoint_sha256,
                label="r212 r195 submission entrypoint SHA-256",
            ),
            public_matchup_router_member=self._require_canonical_tar_member(
                public_matchup_router_member, label="r212 r195 public router"
            ),
            public_matchup_router_sha256=_require_sha256(
                public_matchup_router_sha256,
                label="r212 r195 public router SHA-256",
            ),
        )
        self._handle: Any = None
        self._header: dict[str, Any] | None = None
        self._rows = 0
        self._decisions = 0
        self._active_observations = 0
        self._turn_order_short_circuits = 0
        self._game_resets = 0
        self._routed_decisions = 0
        self._bypassed_decisions = 0
        self._hasher = hashlib.sha256()
        self._footer: dict[str, Any] | None = None
        self._closed = False

    @staticmethod
    def _require_canonical_tar_member(value: Any, *, label: str) -> str:
        member = str(value or "")
        if (
            not member.startswith("./")
            or member == "./"
            or ".." in Path(member).parts
            or Path(member).name in {"", ".", ".."}
        ):
            raise PublicRouteReconstructionError(f"{label} member is not canonical")
        return member

    @classmethod
    def open(cls, **kwargs: Any) -> "SidecarPublicRouteResolver":
        resolver = cls(**kwargs)
        resolver._open()
        return resolver

    def __enter__(self) -> "SidecarPublicRouteResolver":
        if self._handle is None:
            self._open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _assert_unchanged(self) -> None:
        if _stat_identity(self.path) != self.stat_identity:
            raise PublicRouteReconstructionError("r212 route sidecar changed while being consumed")
        if _stat_identity(self.feature_shard_path) != self._feature_shard_stat_identity:
            raise PublicRouteReconstructionError(
                "r212 source feature shard changed while sidecar was being consumed"
            )

    def _open(self) -> None:
        if self._handle is not None:
            return
        self._assert_unchanged()
        handle = self.path.open("r", encoding="utf-8")
        try:
            line = handle.readline()
            header = json.loads(line)
        except (OSError, json.JSONDecodeError) as exc:
            handle.close()
            raise PublicRouteReconstructionError("r212 route sidecar header is invalid") from exc
        if not isinstance(header, dict) or header.get("schema") != ROUTE_SIDECAR_HEADER_SCHEMA:
            handle.close()
            raise PublicRouteReconstructionError("r212 route sidecar has no valid header")
        expected = {
            "format": ROUTE_SIDECAR_FORMAT,
            "source_date": self.source_date,
            "source_feature_shard_sha256": self.feature_shard_sha256,
            "runtime_public_tree_sha256": self.matchup_tree_sha256,
            "runtime_code": self.runtime_code.as_dict(),
            "producer_code": self.expected_producer_code.as_dict(),
            "allowed_physical_slots": sorted(self.allowed_physical_slots),
            "algorithm": ROUTE_ALGORITHM,
            "alignment_contract": ALIGNMENT_CONTRACT,
            "compact_source_routes_ignored": True,
            "oracle_route_used": False,
        }
        if any(header.get(key) != value for key, value in expected.items()):
            handle.close()
            raise PublicRouteReconstructionError("r212 route sidecar binding differs from this source day")
        raw_archive = header.get("raw_archive")
        if (
            not isinstance(raw_archive, Mapping)
            or dict(raw_archive) != self.expected_raw_archive
        ):
            handle.close()
            raise PublicRouteReconstructionError(
                "r212 route sidecar raw archive provenance differs from the protected shard"
            )
        self._handle = handle
        self._header = header

    def resolve_sequence(self, sequence: GameSequence) -> SequencePublicRoutes:
        if self._closed:
            raise PublicRouteReconstructionError("r212 route sidecar reader is closed")
        self._open()
        assert self._handle is not None
        episode_id, seat, env_steps = _sequence_identity(sequence)
        line = self._handle.readline()
        if not line:
            raise PublicRouteReconstructionError("r212 route sidecar ended before source rows")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PublicRouteReconstructionError("r212 route sidecar row is invalid JSON") from exc
        if not isinstance(row, dict):
            raise PublicRouteReconstructionError("r212 route sidecar row is not an object")
        if row.get("schema") == ROUTE_SIDECAR_FOOTER_SCHEMA:
            self._footer = row
            raise PublicRouteReconstructionError("r212 route sidecar reached footer before source rows")
        if (
            row.get("schema") != ROUTE_SIDECAR_ROW_SCHEMA
            or str(row.get("episode_id") or "") != episode_id
            or type(row.get("seat")) is not int
            or int(row["seat"]) != seat
            or tuple(row.get("env_steps") or ()) != env_steps
        ):
            raise PublicRouteReconstructionError(
                "r212 route sidecar row does not align with the compact source sequence"
            )
        raw_member_sha256 = _require_sha256(
            row.get("raw_member_sha256"), label="r212 route sidecar raw member SHA-256"
        )
        alignment_sha256 = _require_sha256(
            row.get("compact_alignment_sha256"),
            label="r212 route sidecar compact board/action alignment SHA-256",
        )
        if alignment_sha256 != _sequence_compact_alignment_sha256(sequence):
            raise PublicRouteReconstructionError(
                "r212 route sidecar board/action alignment differs from the compact source"
            )
        routes_raw = row.get("routes")
        if (
            not isinstance(routes_raw, list)
            or len(routes_raw) != len(env_steps)
            or any(type(value) is not int for value in routes_raw)
            or any(value != UNKNOWN_ROUTE and value not in self.allowed_physical_slots for value in routes_raw)
        ):
            raise PublicRouteReconstructionError("r212 route sidecar has invalid physical routes")
        active_observations = _require_exact_int(
            row.get("active_observations"), label="r212 route sidecar active observations"
        )
        turn_order_short_circuits = _require_exact_int(
            row.get("turn_order_short_circuits"),
            label="r212 route sidecar turn-order short circuits",
        )
        turn_order_env_steps_raw = row.get("turn_order_short_circuit_env_steps")
        game_resets = _require_exact_int(
            row.get("game_resets"), label="r212 route sidecar game resets"
        )
        game_reset_env_steps_raw = row.get("game_reset_env_steps")
        if (
            active_observations < 0
            or turn_order_short_circuits < 0
            or not isinstance(turn_order_env_steps_raw, list)
            or any(type(value) is not int for value in turn_order_env_steps_raw)
            or tuple(sorted(turn_order_env_steps_raw)) != tuple(turn_order_env_steps_raw)
            or len(set(turn_order_env_steps_raw)) != len(turn_order_env_steps_raw)
            or any(value not in env_steps for value in turn_order_env_steps_raw)
            or len(turn_order_env_steps_raw) > turn_order_short_circuits
            or active_observations + len(turn_order_env_steps_raw) < len(env_steps)
            or game_resets < 0
            or not isinstance(game_reset_env_steps_raw, list)
            or any(type(value) is not int or value < 0 for value in game_reset_env_steps_raw)
            or tuple(sorted(game_reset_env_steps_raw)) != tuple(game_reset_env_steps_raw)
            or len(set(game_reset_env_steps_raw)) != len(game_reset_env_steps_raw)
            or len(game_reset_env_steps_raw) != game_resets
            or any(value >= env_steps[0] for value in game_reset_env_steps_raw)
        ):
            raise PublicRouteReconstructionError(
                "r212 route sidecar router/turn-order accounting is impossible"
            )
        result = SequencePublicRoutes(
            episode_id=episode_id,
            seat=seat,
            env_steps=env_steps,
            routes=tuple(int(value) for value in routes_raw),
            compact_alignment_sha256=alignment_sha256,
            raw_member_sha256=raw_member_sha256,
            active_observations=active_observations,
            turn_order_short_circuits=turn_order_short_circuits,
            turn_order_short_circuit_env_steps=tuple(turn_order_env_steps_raw),
            game_resets=game_resets,
            game_reset_env_steps=tuple(game_reset_env_steps_raw),
        )
        self._hasher.update(_canonical_json(result.as_sidecar_row()))
        self._rows += 1
        self._decisions += len(result.routes)
        self._active_observations += active_observations
        self._turn_order_short_circuits += turn_order_short_circuits
        self._game_resets += game_resets
        self._routed_decisions += sum(int(value != UNKNOWN_ROUTE) for value in result.routes)
        self._bypassed_decisions += sum(int(value == UNKNOWN_ROUTE) for value in result.routes)
        return result

    def projection(self, *, expected_records: int) -> dict[str, Any]:
        expected = _require_exact_int(expected_records, label="r212 expected source records")
        if self._rows != expected:
            raise PublicRouteReconstructionError("r212 route sidecar source record count differs")
        assert self._handle is not None
        line = self._handle.readline()
        if not line:
            raise PublicRouteReconstructionError("r212 route sidecar has no footer")
        try:
            footer = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PublicRouteReconstructionError("r212 route sidecar footer is invalid") from exc
        if not isinstance(footer, dict) or footer.get("schema") != ROUTE_SIDECAR_FOOTER_SCHEMA:
            raise PublicRouteReconstructionError("r212 route sidecar footer is missing")
        if self._handle.read(1):
            raise PublicRouteReconstructionError("r212 route sidecar has trailing data")
        expected_rows_sha = "sha256:" + self._hasher.hexdigest()
        projection = footer.get("projection")
        expected_provenance = {
            "schema": ROUTE_RECONSTRUCTION_SCHEMA,
            "source_date": self.source_date,
            "source_feature_shard_sha256": self.feature_shard_sha256,
            "raw_archive": dict((self._header or {}).get("raw_archive") or {}),
            "runtime_public_tree_sha256": self.matchup_tree_sha256,
            "runtime_code": self.runtime_code.as_dict(),
            "producer_code": self.expected_producer_code.as_dict(),
            "allowed_physical_slots": sorted(self.allowed_physical_slots),
            "algorithm": ROUTE_ALGORITHM,
            "alignment_contract": ALIGNMENT_CONTRACT,
            "compact_source_routes_ignored": True,
            "oracle_route_used": False,
        }
        if "source_query" in (self._header or {}):
            expected_provenance["source_query"] = (self._header or {}).get(
                "source_query"
            )
        if (
            int(footer.get("rows", -1)) != self._rows
            or str(footer.get("rows_sha256") or "") != expected_rows_sha
            or not isinstance(projection, Mapping)
            or int(projection.get("records", -1)) != self._rows
            or int(projection.get("decisions", -1)) != self._decisions
            or int(projection.get("active_observations", -1)) != self._active_observations
            or int(projection.get("turn_order_short_circuits", -1))
            != self._turn_order_short_circuits
            or int(projection.get("game_resets", -1)) != self._game_resets
            or int(projection.get("routed_decisions", -1)) != self._routed_decisions
            or int(projection.get("bypassed_decisions", -1)) != self._bypassed_decisions
            or str(projection.get("member_route_sha256") or "") != expected_rows_sha
            or any(
                projection.get(key) != value
                for key, value in expected_provenance.items()
            )
        ):
            raise PublicRouteReconstructionError("r212 route sidecar footer accounting differs")
        self._footer = footer
        self._assert_unchanged()
        result = dict(projection)
        result["sidecar"] = {"path": str(self.path), "sha256": self.sha256}
        result["source"] = "immutable_raw_archive_route_sidecar"
        if self._header is not None and "source_query" in self._header:
            result["source_query"] = self._header["source_query"]
        return result

    @property
    def header(self) -> dict[str, Any]:
        """Immutable parsed header for manifest/provenance checks."""

        self._open()
        assert self._header is not None
        return dict(self._header)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._assert_unchanged()


def materialize_day_sidecar(
    *,
    source_date: str,
    feature_shard_path: Path,
    feature_shard_sha256: str,
    raw_archive_root: Path,
    matchup_tree_path: Path,
    matchup_tree_sha256: str,
    submission_bundle_path: Path,
    submission_bundle_sha256: str,
    submission_entrypoint_member: str,
    submission_entrypoint_sha256: str,
    public_matchup_router_member: str,
    public_matchup_router_sha256: str,
    allowed_physical_slots: Collection[int],
    output_dir: Path,
    producer_materializer_cli_path: Path,
) -> tuple[Path, str, dict[str, Any]]:
    """Build one no-clobber day sidecar from its locally available raw ZIP."""

    shard = Path(feature_shard_path).expanduser().resolve()
    expected_shard_sha = _require_sha256(
        feature_shard_sha256, label="r212 source feature shard SHA-256"
    )
    if not shard.is_file():
        raise FileNotFoundError(shard)
    if _sha256_file(shard) != expected_shard_sha:
        raise PublicRouteReconstructionError(
            "r212 feature shard SHA-256 differs before raw-route sidecar materialization"
        )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    producer_code = bind_sidecar_producer_code(
        materializer_cli_path=producer_materializer_cli_path
    )
    partial = destination / f".r212-public-routes-{source_date}.partial.{os.getpid()}.{time.time_ns()}.jsonl"
    try:
        with RawPublicRouteResolver.open(
            source_date=source_date,
            feature_shard_path=shard,
            feature_shard_sha256=expected_shard_sha,
            raw_archive_root=raw_archive_root,
            matchup_tree_path=matchup_tree_path,
            matchup_tree_sha256=matchup_tree_sha256,
            submission_bundle_path=submission_bundle_path,
            submission_bundle_sha256=submission_bundle_sha256,
            submission_entrypoint_member=submission_entrypoint_member,
            submission_entrypoint_sha256=submission_entrypoint_sha256,
            public_matchup_router_member=public_matchup_router_member,
            public_matchup_router_sha256=public_matchup_router_sha256,
            allowed_physical_slots=allowed_physical_slots,
            sidecar_producer_code=producer_code,
        ) as resolver, partial.open("x", encoding="utf-8") as handle:
            handle.write(_canonical_json(resolver.header()).decode("utf-8"))
            row_hasher = hashlib.sha256()
            records = 0
            for sequence in iter_feature_shard(shard):
                result = resolver.resolve_sequence(sequence)
                row = result.as_sidecar_row()
                encoded = _canonical_json(row)
                handle.write(encoded.decode("utf-8"))
                row_hasher.update(encoded)
                records += 1
            projection = resolver.projection(expected_records=records)
            footer = {
                "schema": ROUTE_SIDECAR_FOOTER_SCHEMA,
                "rows": records,
                "rows_sha256": "sha256:" + row_hasher.hexdigest(),
                "projection": projection,
            }
            handle.write(_canonical_json(footer).decode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        sidecar_sha256 = _sha256_file(partial)
        producer_code.assert_unchanged()
        final = destination / (
            f"r212-public-routes-{source_date}-{sidecar_sha256.split(':', 1)[1]}.jsonl"
        )
        try:
            os.link(partial, final)
        except FileExistsError:
            if _sha256_file(final) != sidecar_sha256:
                raise PublicRouteReconstructionError("r212 route sidecar digest collision")
        return final, sidecar_sha256, projection
    finally:
        partial.unlink(missing_ok=True)


__all__ = [
    "ALIGNMENT_CONTRACT",
    "BoundProducerCode",
    "UNKNOWN_ROUTE",
    "PRODUCER_CODE_SCHEMA",
    "ProducerCodeBinding",
    "PublicRouteReconstructionError",
    "ROUTE_ALGORITHM",
    "ROUTE_RECONSTRUCTION_SCHEMA",
    "ROUTE_SIDECAR_FOOTER_SCHEMA",
    "ROUTE_SIDECAR_HEADER_SCHEMA",
    "ROUTE_SIDECAR_ROW_SCHEMA",
    "RawArchiveBinding",
    "RawPublicRouteResolver",
    "RuntimeCodeBinding",
    "SequencePublicRoutes",
    "SidecarPublicRouteResolver",
    "bind_raw_archive",
    "bind_sidecar_producer_code",
    "compact_alignment_sha256",
    "materialize_day_sidecar",
    "reconstruct_public_routes_from_raw_member",
    "TURN_ORDER_SHORT_CIRCUIT_CONTRACT",
    "verify_imported_runtime_public_router",
]
