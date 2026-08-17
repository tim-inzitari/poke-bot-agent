"""Sealed, shadow-only r198 three-arm evaluation factory.

This is intentionally a *factory*, not a production activation path.  It is
loaded by :mod:`poke_bot.rtp_three_arm_evaluation_runner` inside a fresh
``exec`` child for one cell/arm.  The factory accepts only an immutable
snapshot-local input bundle and refuses to consult the live selector, a live
candidate directory, installed baselines, training state, or Kaggle state.

The public surface is deliberately small:

``ProductionR198EvaluationFactory.worker_environment``
    Produce the explicit, sealed child environment.  This method has no model
    or baseline construction side effects, because the runner also invokes it
    in the parent process while preparing the child.
``ProductionR198EvaluationFactory.create_arm_runtime``
    Load one fresh r195 parent model, a fresh official baseline module, and
    one of the three exact evaluation arm configurations.
``ProductionR198EvaluationFactory.create_arm_engine``
    Bind a fresh :class:`RtpPairingSnapshotEngine` to the exact v2 private
    build artifact set.  It deliberately does *not* restore a snapshot; the
    runner performs the already-sealed restore as its last engine mutation.
``run_r198_planner_preflight``
    Execute the real normal and forced-replan planner paths against fixture
    observations supplied by the immutable input producer.  It writes only a
    new receipt at its explicit output path, never a candidate or selector.

All file checks use lexical physical paths and rehash the bytes immediately
before use.  A content digest alone is not enough for this boundary: a later
worker must not be able to import a different symlink target or a writable
baseline module after a prior verifier accepted it.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
import stat
import sys
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

FACTORY_INPUT_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_production_evaluation_factory_inputs/v1"
)
CANDIDATE_SNAPSHOT_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_evaluation_candidate_snapshot/v1"
)
PACKAGE_SNAPSHOT_SCHEMA = (
    "poke_bot.recursive_turn_planner.evaluation_package_tree_snapshot/v1"
)
PREFLIGHT_INPUT_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_planner_pass_preflight_input/v1"
)
PREFLIGHT_RECEIPT_SCHEMA = (
    "poke_bot.recursive_turn_planner.r198_planner_pass_preflight/v1"
)
EVALUATION_AUTHORITY_SCHEMA = (
    "poke_bot.recursive_turn_planner.three_arm_evaluation_authorization/v1"
)
SOURCE_SNAPSHOT_SCHEMA = "poke_bot.alakazam_rtp_r198_eval_source_snapshot/v1"
SOURCE_SNAPSHOT_MANIFEST_NAME = "r198-eval-source-snapshot-manifest.json"
EVALUATION_CG_CLOSURE_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_closure/v1"
)
EVALUATION_CG_CLOSURE_MANIFEST_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_closure_manifest/v1"
)
EVALUATION_CG_METADATA_PARITY_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_metadata_parity/v1"
)
EVALUATION_CG_CLOSURE_FILENAME = "eval-cg-closure.json"
THREE_ARM_EVALUATION_MANIFEST_SCHEMA = (
    "poke_bot.recursive_turn_planner.three_arm_evaluation_manifest/v2"
)
PAIRING_CAPABILITY_SCHEMA = (
    "poke_bot.recursive_turn_planner.true_rng_pairing_capability/v2"
)
RTP_SIDECAR_SCHEMA = "poke_bot.recursive_turn_planner.shadow_train/v1"
RTP_SIDECAR_RECEIPT_SCHEMA = RTP_SIDECAR_SCHEMA + ".receipt"
R197_COMPLETION_RECEIPT_SCHEMA = "poke_bot.alakazam_rtp_r197_shadow_candidate/v1"
R197_COMPLETION_RECEIPT_FILENAME = "r197-completion-receipt.json"

R198_CANDIDATE_ID = (
    "r197-bc31f860b8154549b77f3e414127139b02ad4f4905dd76c78974e599ba868e6e"
)
R198_CANDIDATE_CONTRACT_SHA256 = (
    "sha256:bc31f860b8154549b77f3e414127139b02ad4f4905dd76c78974e599ba868e6e"
)
R195_PARENT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R197_SIDECAR_SHA256 = (
    "sha256:23eb09cbfa5e9e8d3aec3b8af4dc03a71db811ce9b7c32c6c5ece65bc3f3dc31"
)
R197_SIDECAR_RECEIPT_SHA256 = (
    "sha256:2f577d4101b7657d133eac190081ef75fca211435b83dcca8f2e2686d7597d2b"
)
R197_COMPLETION_RECEIPT_SHA256 = (
    "sha256:b0c209257ed401bf9c5fe5a1ee17be1d1cdc01a1f9780e3e0d23ce8fa5f80737"
)
R197_COMPLETION_RECEIPT_BYTES = 113_366
MATCHUP_ADAPTER_REGISTRY_RELATIVE = Path("state/matchup_adapter_roster.json")
MATCHUP_ADAPTER_REGISTRY_SHA256 = (
    "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc"
)
MATCHUP_ADAPTER_REGISTRY_BYTES = 11_899
MATCHUP_ADAPTER_REGISTRY_CANONICAL_SHA256 = (
    "sha256:444c42c1235c19d3d95b10e80a12a84f35c9fb803967096736446eac1a5e225a"
)
MATCHUP_ADAPTER_REGISTRY_SCHEMA = "poke_bot.matchup_adapter_roster/v1"
MATCHUP_ADAPTER_SLOT_REGISTRY_SCHEMA = "poke_bot.matchup_adapter_slot_registry/v1"
RESEARCH_CONTROL_REGISTRY_SHA256 = (
    "sha256:78fd8e52df1464db94e74a49247a67ced41b5d164dc86fafec3229f2c1e47edc"
)
RESEARCH_CONTROL_REGISTRY_BYTES = 2_117
R197_SIDECAR_CONFIG_SHA256 = (
    "sha256:7fb0658f0358c93636524a40ddd52f9f76199de261963a85dbf5946901a9f676"
)
R195_DECK_CSV_SHA256 = (
    "sha256:1705f0f4db0c54b32f297fc9292a417b0c3abc9fdb6edf6a5370af6a635efe65"
)
R195_DECK_CARDS_SHA256 = (
    "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
)
R195_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
R198_EVAL_CG_CLOSURE_RECEIPT_SHA256 = (
    "sha256:419ad46a9b31b9fdc040b851b553108b1bd038b68acadccb4dc9c38bfd35bbe0"
)
R198_EVAL_CG_CLOSURE_RECEIPT_BYTES = 2399
R198_EVAL_CG_CLOSURE_MANIFEST_SHA256 = (
    "sha256:a3c0dea888638d87a2423b437dd4e8dd105423a91a289ad272298de7b5d40da7"
)
R198_EVAL_CG_METADATA_PARITY_SHA256 = (
    "sha256:cbdffe7fe99c9c29d83cc6dd3530b1c406ce7f4d0f99920ca6fc45624e0e25a7"
)

R198_PROFILE = "pure_rl_r197"
R198_MAX_NEURAL_PASSES = 256
R198_MAX_ACTION_COMBOS = 1024
R198_NUM_CANDIDATES = 4
R198_MAX_RECURSION_DEPTH = 2
R198_NORMAL_PASSES = 6
R198_FORCED_REPLAN_PASSES = 5
R198_GPU_UUID = "GPU-79cf504f-6573-0b8c-c90e-eb567b7bcfa6"

OFFICIAL_CONTROL_DIGESTS = {
    "iono": "sha256:6ba8e818b698774b6e437364e9457600eda950fbefb663d8e4ad39cdaf0371e2",
    "dragapult-ex": "sha256:835dcbcc26366faa04d902db727620d4b12618b6a66d000dccb9c9b86e9d62a0",
    "mega-abomasnow-ex": "sha256:57a9499b2bee493a830abaf5a3e19b8a73faea200faee87aeeb2864bab25c2fb",
    "mega-lucario-ex": "sha256:98f20936d430c6cc60f3eb1da8230392bf6dce8ecacf97773bda4db63f56376a",
}
CANONICAL_ARMS = (
    "no_rtp",
    "direct_bridge_recursive_disabled",
    "recursive_rtp",
)
R198_SHARED_ARTIFACT_NAMES = (
    "deck",
    "evaluation_only_cohort",
    "matchup_tree",
    "parent_checkpoint",
    "planner_preflight_receipt",
    "r197_completion_receipt",
    "research_control_registry",
)
_EVALUATION_CG_CLOSURE_BASE_KEYS = frozenset({"receipt", "runtime_library"})
_EVALUATION_CG_CLOSURE_NORMALIZED_KEYS = frozenset(
    {
        *_EVALUATION_CG_CLOSURE_BASE_KEYS,
        "engine_artifact",
        "pairing_build_artifact",
        "cg_source_manifest",
        "closure_manifest",
        "metadata_parity",
        "canonical_abi_sha256",
        "sim_initializer_symbol",
        "snapshot_abi_version",
        "cg_source_tree_sha256",
        "closure_tree_sha256",
        "all_card_canonical_sha256",
        "all_attack_canonical_sha256",
    }
)
_EVALUATION_CG_CLOSURE_TREE_PATHS = (
    "__init__.py",
    "api.py",
    "game.py",
    "libcg.so",
    "sim.py",
    "utils.py",
)


class R198ProductionFactoryError(RuntimeError):
    """Raised before a factory can construct an evaluation-only arm."""


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_digest(value: Any) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R198ProductionFactoryError("value is not canonical JSON") from exc
    return _sha256_bytes(encoded)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _text(value: Any, label: str) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        raise R198ProductionFactoryError(f"{label} is required")
    return rendered


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R198ProductionFactoryError(f"{label} must be an object")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise R198ProductionFactoryError(f"{label} must be a sequence")
    return list(value)


def _lexical_path(value: str | Path, label: str) -> Path:
    raw = Path(_text(value, label)).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    if ".." in raw.parts:
        raise R198ProductionFactoryError(f"{label} may not contain '..'")
    return Path(os.path.abspath(os.fspath(raw)))


def _physical_path(
    value: str | Path,
    label: str,
    *,
    file: bool = False,
    directory: bool = False,
) -> Path:
    """Resolve a lexical absolute path without following any symlink."""

    path = _lexical_path(value, label)
    current = Path(path.anchor)
    for index, component in enumerate(path.parts[1:]):
        current /= component
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise R198ProductionFactoryError(f"{label} is missing: {path}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise R198ProductionFactoryError(
                f"{label} may not traverse symbolic link: {current}"
            )
        if index < len(path.parts[1:]) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise R198ProductionFactoryError(
                f"{label} has a non-directory ancestor: {current}"
            )
    metadata = os.lstat(path)
    if file and not stat.S_ISREG(metadata.st_mode):
        raise R198ProductionFactoryError(f"{label} is not a regular file: {path}")
    if directory and not stat.S_ISDIR(metadata.st_mode):
        raise R198ProductionFactoryError(f"{label} is not a directory: {path}")
    return path


def _inside(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise R198ProductionFactoryError(f"{label} escapes immutable snapshot root") from exc


def _readonly_mode(path: Path, label: str, *, exact: int | None = None) -> int:
    mode = stat.S_IMODE(os.lstat(path).st_mode)
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise R198ProductionFactoryError(f"{label} is writable")
    if exact is not None and mode != exact:
        raise R198ProductionFactoryError(
            f"{label} must use mode {oct(exact)}, not {oct(mode)}"
        )
    return mode


def _identity(
    raw: Any,
    label: str,
    *,
    root: Path | None = None,
    exact_mode: int | None = 0o444,
) -> dict[str, Any]:
    value = _mapping(raw, label)
    path = _physical_path(value.get("path"), f"{label}.path", file=True)
    if root is not None:
        _inside(path, root, label)
    expected = _text(value.get("sha256"), f"{label}.sha256")
    if not _is_sha256(expected):
        raise R198ProductionFactoryError(f"{label}.sha256 is not a SHA-256 digest")
    observed = {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
    if observed["sha256"] != expected:
        raise R198ProductionFactoryError(f"{label} checksum mismatch")
    if "bytes" in value and int(value["bytes"]) != observed["bytes"]:
        raise R198ProductionFactoryError(f"{label} byte count mismatch")
    observed["mode"] = _readonly_mode(path, label, exact=exact_mode)
    return observed


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R198ProductionFactoryError(f"cannot read {label}: {path}") from exc
    return _mapping(loaded, label)


def _assert_readonly_tree(root: Path, label: str) -> None:
    """Require physical 0555 directories and physical 0444 regular files."""

    root = _physical_path(root, label, directory=True)
    _readonly_mode(root, label, exact=0o555)
    for current_raw, directories, files in os.walk(root, topdown=True, followlinks=False):
        current = _physical_path(current_raw, f"{label} directory", directory=True)
        _readonly_mode(current, f"{label} directory", exact=0o555)
        directories.sort()
        files.sort()
        for name in directories:
            child = current / name
            metadata = os.lstat(child)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise R198ProductionFactoryError(f"{label} contains a nonphysical directory")
            _readonly_mode(child, f"{label} directory", exact=0o555)
        for name in files:
            child = _physical_path(current / name, f"{label} file", file=True)
            _readonly_mode(child, f"{label} file", exact=0o444)


def _identity_equal(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> None:
    for key in ("path", "sha256", "bytes"):
        if left.get(key) != right.get(key):
            raise R198ProductionFactoryError(f"{label} differs at {key}")


def _deck_cards_sha256(cards: Sequence[int]) -> str:
    return _sha256_bytes(
        json.dumps(list(cards), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )


def _atomic_readonly_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Create an immutable receipt exactly once; never overwrite evidence."""

    target = _lexical_path(path, "planner preflight output")
    parent = _physical_path(target.parent, "planner preflight output parent", directory=True)
    if target.exists() or target.is_symlink():
        existing = _physical_path(target, "existing planner preflight output", file=True)
        _readonly_mode(existing, "existing planner preflight output", exact=0o444)
        expected = payload.get("preflight_input_sha256")
        observed = _read_json(existing, "existing planner preflight output")
        if observed.get("preflight_input_sha256") == expected:
            return existing
        raise R198ProductionFactoryError("planner preflight output already has different input")
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = parent / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        os.chmod(target, 0o444)
    except FileExistsError:
        existing = _physical_path(target, "racing planner preflight output", file=True)
        observed = _read_json(existing, "racing planner preflight output")
        if observed.get("preflight_input_sha256") != payload.get("preflight_input_sha256"):
            raise R198ProductionFactoryError("planner preflight output raced with different input")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return _physical_path(target, "planner preflight output", file=True)


@dataclass(frozen=True)
class _CandidateAssets:
    source_root: Path
    source_tree_sha256: str
    factory_module: Mapping[str, Any]
    candidate_manifest: Mapping[str, Any]
    package_root: Path
    parent_checkpoint: Mapping[str, Any]
    sidecar: Mapping[str, Any]
    sidecar_receipt: Mapping[str, Any]
    completion_receipt: Mapping[str, Any]
    deck: Mapping[str, Any]
    matchup_tree: Mapping[str, Any]
    sidecar_config: Mapping[str, Any]


@dataclass(frozen=True)
class _CGAssets:
    runtime_root: Path
    closure_manifest: Mapping[str, Any]
    library: Mapping[str, Any]
    closure_payload: Mapping[str, Any]
    closure_evidence: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class _OfficialPackage:
    opponent_id: str
    content_digest: str
    manifest: Mapping[str, Any]
    payload: Mapping[str, Any]
    package_root: Path
    main_py: Mapping[str, Any]
    deck: Mapping[str, Any]


@dataclass(frozen=True)
class _SealedOfficialPackageTree:
    """The verified, snapshot-local two-file official baseline package."""

    manifest: Mapping[str, Any]
    payload: Mapping[str, Any]
    package_root: Path
    entries: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class _FactoryInputs:
    spec: Mapping[str, Any]
    candidate: _CandidateAssets
    cg: _CGAssets
    matchup_adapter_registry: Mapping[str, Any]
    matchup_adapter_registry_digest: str
    authority: Mapping[str, Any]
    factory_identity: Mapping[str, Any]
    source_snapshot_manifest: Mapping[str, Any]
    evaluation_inputs_root: Path | None


def _factory_spec(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("production_factory")
    spec = _mapping(raw, "production_factory")
    if spec.get("schema") != FACTORY_INPUT_SCHEMA:
        raise R198ProductionFactoryError("production_factory schema is invalid")
    if spec.get("status") != "sealed":
        raise R198ProductionFactoryError("production_factory must be sealed")
    return spec


def _source_root(spec: Mapping[str, Any]) -> tuple[Path, str, dict[str, Any]]:
    raw = spec.get("source_snapshot_root")
    if raw is None:
        snapshot = _mapping(spec.get("source_snapshot"), "production_factory.source_snapshot")
        raw = snapshot.get("root")
        tree_sha = snapshot.get("source_tree_sha256")
    else:
        tree_sha = spec.get("source_tree_sha256")
    root = _physical_path(raw, "production_factory.source_snapshot_root", directory=True)
    _assert_readonly_tree(root, "production factory source snapshot")
    digest = _text(tree_sha, "production_factory.source_tree_sha256")
    if not _is_sha256(digest):
        raise R198ProductionFactoryError("production_factory source tree digest is invalid")
    expected_manifest = root / SOURCE_SNAPSHOT_MANIFEST_NAME
    supplied_manifest = _identity(
        spec.get("source_snapshot_manifest"),
        "production_factory.source_snapshot_manifest",
        root=root,
    )
    if Path(supplied_manifest["path"]) != expected_manifest:
        raise R198ProductionFactoryError(
            "production_factory source snapshot manifest is not the canonical snapshot record"
        )
    payload = _read_json(expected_manifest, "production factory source snapshot manifest")
    if payload.get("schema") != SOURCE_SNAPSHOT_SCHEMA:
        raise R198ProductionFactoryError("production_factory source snapshot schema is invalid")
    if payload.get("source_tree_sha256") != digest:
        raise R198ProductionFactoryError(
            "production_factory source snapshot tree digest differs from sealed record"
        )
    return root, digest, supplied_manifest


def _matchup_adapter_registry(
    spec: Mapping[str, Any],
    *,
    source_root: Path,
    source_snapshot_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Bind the public router to the one sealed snapshot-local V6 roster.

    ``PublicMatchupDecisionTree`` resolves its V6 roster relative to the
    imported :mod:`poke_bot.matchup_adapters_v6` module.  A source snapshot
    therefore has to carry that otherwise-transitive JSON input explicitly;
    accepting the mutable checkout's registry would make the same tree bytes
    route through different physical adapter slots.
    """

    raw = _mapping(
        spec.get("matchup_adapter_registry"),
        "production_factory.matchup_adapter_registry",
    )
    if raw.get("mode") != 0o444:
        raise R198ProductionFactoryError(
            "production_factory matchup adapter registry must attest mode 0o444"
        )
    identity = _identity(
        raw,
        "production_factory.matchup_adapter_registry",
        root=source_root,
        exact_mode=0o444,
    )
    expected_path = source_root / MATCHUP_ADAPTER_REGISTRY_RELATIVE
    if Path(identity["path"]) != expected_path:
        raise R198ProductionFactoryError(
            "matchup adapter registry is not the canonical snapshot-local roster"
        )
    if (
        identity["sha256"] != MATCHUP_ADAPTER_REGISTRY_SHA256
        or identity["bytes"] != MATCHUP_ADAPTER_REGISTRY_BYTES
    ):
        raise R198ProductionFactoryError(
            "snapshot-local matchup adapter registry identity changed"
        )

    snapshot_payload = _read_json(
        Path(source_snapshot_manifest["path"]),
        "production factory source snapshot manifest",
    )
    raw_entries = snapshot_payload.get("source_entries")
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise R198ProductionFactoryError(
            "source snapshot manifest has no physical source inventory"
        )
    relative = MATCHUP_ADAPTER_REGISTRY_RELATIVE.as_posix()
    matches = [
        dict(row)
        for row in raw_entries
        if isinstance(row, Mapping) and row.get("path") == relative
    ]
    if len(matches) != 1:
        raise R198ProductionFactoryError(
            "source snapshot manifest does not uniquely bind matchup adapter registry"
        )
    entry = matches[0]
    if set(entry) != {"path", "type", "mode", "size", "sha256"} or entry != {
        "path": relative,
        "type": "file",
        "mode": 0o444,
        "size": MATCHUP_ADAPTER_REGISTRY_BYTES,
        "sha256": MATCHUP_ADAPTER_REGISTRY_SHA256,
    }:
        raise R198ProductionFactoryError(
            "source snapshot matchup adapter registry inventory identity changed"
        )
    required = snapshot_payload.get("required_relative_files")
    if (
        not isinstance(required, Sequence)
        or isinstance(required, (str, bytes))
        or relative not in required
    ):
        raise R198ProductionFactoryError(
            "source snapshot required-file contract omits matchup adapter registry"
        )

    payload = _read_json(expected_path, "snapshot-local matchup adapter registry")
    expected_semantics = {
        "schema": MATCHUP_ADAPTER_REGISTRY_SCHEMA,
        "slot_schema": MATCHUP_ADAPTER_SLOT_REGISTRY_SCHEMA,
        "checkpoint_format": "poke-bot-matchup-adapter-bank-v6",
        "slot_capacity": 64,
        "legacy_v5_prefix_length": 18,
    }
    for key, expected in expected_semantics.items():
        if payload.get(key) != expected:
            raise R198ProductionFactoryError(
                f"snapshot-local matchup adapter registry differs at {key}"
            )
    canonical_digest = _canonical_digest(payload)
    if canonical_digest != MATCHUP_ADAPTER_REGISTRY_CANONICAL_SHA256:
        raise R198ProductionFactoryError(
            "snapshot-local matchup adapter registry canonical digest changed"
        )
    return identity, canonical_digest


def _authority(spec: Mapping[str, Any]) -> dict[str, Any]:
    raw = spec.get("evaluation_authority", spec.get("authority"))
    authority = _mapping(raw, "production_factory.evaluation_authority")
    if authority.get("schema") != EVALUATION_AUTHORITY_SCHEMA:
        raise R198ProductionFactoryError("evaluation authority schema is invalid")
    if authority.get("status") != "authorized_evaluation_only":
        raise R198ProductionFactoryError("evaluation authority is not evaluation-only")
    expected = {
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_change_authorized": False,
        "selector_change_authorized": False,
        "action_authority_authorized": False,
        "kaggle_submission_authorized": False,
    }
    for key, expected_value in expected.items():
        if authority.get(key) is not expected_value:
            raise R198ProductionFactoryError(f"evaluation authority fails {key}")
    if authority.get("scope") != "r198_factory_preparation_and_evaluation_only":
        raise R198ProductionFactoryError("evaluation authority scope is not the r198 factory-only scope")
    return authority


def _candidate_assets(
    spec: Mapping[str, Any],
    source_root: Path,
    tree_sha: str,
    *,
    validate_sidecar_payload: bool = True,
) -> _CandidateAssets:
    factory_module = _identity(
        spec.get("factory_module"), "production_factory.factory_module", root=source_root
    )
    if Path(factory_module["path"]) != source_root / "poke_bot" / "rtp_r198_production_factory.py":
        raise R198ProductionFactoryError("factory module is not the snapshot-local r198 factory")
    actual_factory = _physical_path(__file__, "factory module", file=True)
    actual_identity = {
        "path": str(actual_factory),
        "sha256": _sha256_file(actual_factory),
        "bytes": actual_factory.stat().st_size,
    }
    _identity_equal(factory_module, actual_identity, "sealed factory module")

    candidate_manifest = _identity(
        spec.get("candidate_snapshot"),
        "production_factory.candidate_snapshot",
        root=source_root,
    )
    if Path(candidate_manifest["path"]) != (
        source_root / "evaluation-artifacts" / "r197-candidate" / "manifest.json"
    ):
        raise R198ProductionFactoryError("candidate snapshot is not the canonical snapshot-local manifest")
    payload = _read_json(Path(candidate_manifest["path"]), "candidate snapshot")
    if payload.get("schema") != CANDIDATE_SNAPSHOT_SCHEMA or payload.get("status") != "sealed":
        raise R198ProductionFactoryError("candidate snapshot is not sealed r198 evidence")
    if payload.get("no_symlinks") is not True or payload.get("all_paths_read_only") is not True:
        raise R198ProductionFactoryError("candidate snapshot lacks physical-readonly attestation")
    if payload.get("candidate_id") != R198_CANDIDATE_ID:
        raise R198ProductionFactoryError("candidate snapshot ID is not the completed r197 candidate")
    if payload.get("candidate_contract_sha256") != R198_CANDIDATE_CONTRACT_SHA256:
        raise R198ProductionFactoryError("candidate snapshot contract digest is wrong")
    package_root = _physical_path(payload.get("package_root"), "candidate package root", directory=True)
    _inside(package_root, source_root, "candidate package root")
    if package_root != source_root / "evaluation-artifacts" / "r197-candidate":
        raise R198ProductionFactoryError("candidate package root is not the canonical snapshot-local package")
    _assert_readonly_tree(package_root, "candidate package")
    raw_artifacts = _mapping(payload.get("artifacts"), "candidate snapshot artifacts")
    required = {
        "parent_checkpoint",
        "sidecar",
        "sidecar_receipt",
        "completion_receipt",
        "deck",
        "matchup_tree",
    }
    if set(raw_artifacts) != required:
        raise R198ProductionFactoryError("candidate snapshot artifacts are not the exact required set")
    assets = {
        name: _identity(raw_artifacts[name], f"candidate artifact {name}", root=package_root)
        for name in sorted(required)
    }
    expected_hashes = {
        "parent_checkpoint": R195_PARENT_SHA256,
        "sidecar": R197_SIDECAR_SHA256,
        "sidecar_receipt": R197_SIDECAR_RECEIPT_SHA256,
        "completion_receipt": R197_COMPLETION_RECEIPT_SHA256,
        "deck": R195_DECK_CSV_SHA256,
        "matchup_tree": R195_MATCHUP_TREE_SHA256,
    }
    for name, digest in expected_hashes.items():
        if assets[name]["sha256"] != digest:
            raise R198ProductionFactoryError(f"candidate artifact {name} does not match r198 identity")
    completion_receipt = assets["completion_receipt"]
    if Path(completion_receipt["path"]) != package_root / R197_COMPLETION_RECEIPT_FILENAME:
        raise R198ProductionFactoryError(
            "candidate completion receipt is not the canonical snapshot-local file"
        )
    _validate_r197_completion_receipt(completion_receipt)

    factory_artifacts = _mapping(
        spec.get("artifacts"), "production_factory.artifacts"
    )
    if set(factory_artifacts) != required:
        raise R198ProductionFactoryError(
            "production_factory artifacts are not the exact candidate artifact set"
        )
    for name in sorted(required):
        bound = _identity(
            factory_artifacts.get(name),
            f"production_factory artifact {name}",
            root=package_root,
        )
        _identity_equal(bound, assets[name], f"production_factory/candidate {name}")

    _validate_candidate_deck(assets["deck"])
    # Parent-side environment preparation needs immutable identities only.  It
    # must not deserialize any model/sidecar payload; every fresh child (and
    # the source builder/preflight) does the strict safe sidecar inspection.
    config: Mapping[str, Any]
    if validate_sidecar_payload:
        config = _validate_inert_sidecar(assets["sidecar"], assets["sidecar_receipt"])
    else:
        config = {}
    return _CandidateAssets(
        source_root=source_root,
        source_tree_sha256=tree_sha,
        factory_module=factory_module,
        candidate_manifest=candidate_manifest,
        package_root=package_root,
        parent_checkpoint=assets["parent_checkpoint"],
        sidecar=assets["sidecar"],
        sidecar_receipt=assets["sidecar_receipt"],
        completion_receipt=completion_receipt,
        deck=assets["deck"],
        matchup_tree=assets["matchup_tree"],
        sidecar_config=config,
    )


def _validate_r197_completion_receipt(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate the exact frozen r197 completion boundary inside the snapshot."""

    if (
        identity.get("sha256") != R197_COMPLETION_RECEIPT_SHA256
        or int(identity.get("bytes", -1)) != R197_COMPLETION_RECEIPT_BYTES
        or int(identity.get("mode", -1)) != 0o444
    ):
        raise R198ProductionFactoryError(
            "candidate completion receipt identity is not the exact frozen r197 boundary"
        )
    payload = _read_json(Path(str(identity["path"])), "candidate completion receipt")
    if (
        payload.get("schema") != R197_COMPLETION_RECEIPT_SCHEMA
        or payload.get("status") != "completed_shadow_only"
        or payload.get("candidate_id") != R198_CANDIDATE_ID
        or payload.get("candidate_contract_sha256") != R198_CANDIDATE_CONTRACT_SHA256
    ):
        raise R198ProductionFactoryError(
            "candidate completion receipt does not bind the completed r197 candidate"
        )
    authority = _mapping(
        payload.get("authority"), "candidate completion receipt authority"
    )
    if authority.get("shadow_only") is not True:
        raise R198ProductionFactoryError(
            "candidate completion receipt is not shadow-only"
        )
    for key in (
        "serving_eligible",
        "action_authority_enabled",
        "selector_authority",
        "live_checkpoint_publication",
        "submission_eligible",
    ):
        if authority.get(key) is not False:
            raise R198ProductionFactoryError(
                f"candidate completion receipt unexpectedly grants {key}"
            )
    return payload


def _validate_candidate_deck(identity: Mapping[str, Any]) -> list[int]:
    from .deck_pool import read_deck

    try:
        cards = read_deck(identity["path"])
    except Exception as exc:  # deck parser is intentionally strict about 60 cards.
        raise R198ProductionFactoryError("candidate deck cannot be read as 60 cards") from exc
    if _deck_cards_sha256(cards) != R195_DECK_CARDS_SHA256:
        raise R198ProductionFactoryError("candidate deck card ordering/content identity changed")
    return cards


def _validate_inert_sidecar(
    sidecar: Mapping[str, Any], sidecar_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Safely inspect the exact r197 shadow sidecar before it is attached."""

    try:
        import torch

        payload = torch.load(sidecar["path"], map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - depends on external torch serialization.
        raise R198ProductionFactoryError("inert r197 sidecar safe load failed") from exc
    if not isinstance(payload, Mapping):
        raise R198ProductionFactoryError("inert r197 sidecar payload is invalid")
    expected = {
        "schema": RTP_SIDECAR_SCHEMA,
        "parent_checkpoint_sha256": R195_PARENT_SHA256,
        "shadow_only": True,
        "serving_eligible": False,
        "action_authority_enabled": False,
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise R198ProductionFactoryError(f"inert r197 sidecar fails {key}")
    if payload.get("research_only") is True:
        raise R198ProductionFactoryError("r197 sidecar may not be substituted with a research-only sidecar")
    config = _mapping(payload.get("config"), "inert r197 sidecar config")
    if _canonical_digest(config) != R197_SIDECAR_CONFIG_SHA256:
        raise R198ProductionFactoryError("inert r197 sidecar configuration digest changed")
    fields = {
        "sizing_profile": R198_PROFILE,
        "d_model": 96,
        "dynamics_width": 192,
        "num_plan_candidates": R198_NUM_CANDIDATES,
        "max_recursion_depth": R198_MAX_RECURSION_DEPTH,
        "max_neural_passes": R198_MAX_NEURAL_PASSES,
    }
    for key, expected_value in fields.items():
        if config.get(key) != expected_value:
            raise R198ProductionFactoryError(f"inert r197 sidecar config fails {key}")
    state = payload.get("state_dict", payload.get("planner_state_dict"))
    if not isinstance(state, Mapping) or not state:
        raise R198ProductionFactoryError("inert r197 sidecar has no strict planner state")

    receipt = _read_json(Path(sidecar_receipt["path"]), "inert r197 sidecar receipt")
    if receipt.get("schema") != RTP_SIDECAR_RECEIPT_SCHEMA:
        raise R198ProductionFactoryError("inert r197 sidecar receipt schema is invalid")
    for key, expected_value in expected.items():
        if key == "schema":
            continue
        if receipt.get(key) != expected_value:
            raise R198ProductionFactoryError(f"inert r197 sidecar receipt fails {key}")
    if receipt.get("required_neural_passes_normal") != R198_NORMAL_PASSES:
        raise R198ProductionFactoryError("inert r197 sidecar receipt normal-pass proof changed")
    if receipt.get("required_neural_passes_forced_replan") != R198_FORCED_REPLAN_PASSES:
        raise R198ProductionFactoryError("inert r197 sidecar receipt forced-pass proof changed")
    return config


def _cg_assets(spec: Mapping[str, Any], source_root: Path) -> _CGAssets:
    raw = _mapping(spec.get("evaluation_cg"), "production_factory.evaluation_cg")
    runtime = _physical_path(raw.get("runtime_root"), "evaluation cg runtime root", directory=True)
    _inside(runtime, source_root, "evaluation cg runtime root")
    _assert_readonly_tree(runtime, "evaluation cg runtime")
    closure = _identity(raw.get("closure_manifest"), "evaluation cg closure manifest", root=source_root)
    library = _identity(raw.get("library"), "evaluation cg library", root=runtime)
    if (
        closure["sha256"] != R198_EVAL_CG_CLOSURE_RECEIPT_SHA256
        or closure["bytes"] != R198_EVAL_CG_CLOSURE_RECEIPT_BYTES
    ):
        raise R198ProductionFactoryError("evaluation cg closure is not the final r198 dual-DSO-proof receipt")
    if Path(library["path"]).name != "libcg.so":
        raise R198ProductionFactoryError("evaluation cg library must be libcg.so")
    expected_runtime = source_root / "kaggle" / "input" / "rtp-eval-cg"
    if runtime != expected_runtime:
        raise R198ProductionFactoryError("evaluation cg runtime is not the snapshot-local rtp-eval-cg closure")
    if Path(closure["path"]) != runtime / EVALUATION_CG_CLOSURE_FILENAME:
        raise R198ProductionFactoryError("evaluation cg closure is not the canonical snapshot-local receipt")
    expected_library = runtime / "cg" / "libcg.so"
    if Path(library["path"]) != expected_library:
        raise R198ProductionFactoryError("evaluation cg DSO is not the closure's libcg.so")
    for name in ("__init__.py", "api.py", "game.py", "sim.py", "utils.py", "libcg.so"):
        _physical_path(runtime / "cg" / name, f"evaluation cg {name}", file=True)
    closure_payload = _read_json(Path(closure["path"]), "evaluation cg closure manifest")
    if (
        closure_payload.get("schema") != EVALUATION_CG_CLOSURE_SCHEMA
        or closure_payload.get("status") != "sealed"
        or closure_payload.get("sim_initializer_symbol") != "RtpPairingSnapshotInitialize"
        or closure_payload.get("snapshot_abi_version") != 2
        or closure_payload.get("runtime_or_submission_installation_performed") is not False
    ):
        raise R198ProductionFactoryError("evaluation cg closure record is not the sealed v2 private closure")
    if closure_payload.get("pairing_engine_artifact_sha256") != library["sha256"]:
        raise R198ProductionFactoryError("evaluation cg closure does not bind libcg.so")
    source_manifest_schema = (
        "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_source_manifest/v1"
    )
    evidence: dict[str, Mapping[str, Any]] = {}
    for name, schema, required_status in (
        ("cg_source_manifest", source_manifest_schema, None),
        ("closure_manifest", EVALUATION_CG_CLOSURE_MANIFEST_SCHEMA, None),
        ("metadata_parity", EVALUATION_CG_METADATA_PARITY_SCHEMA, "passed"),
    ):
        identity = _identity(
            closure_payload.get(name), f"evaluation cg closure {name}"
        )
        payload = _read_json(Path(identity["path"]), f"evaluation cg closure {name}")
        if payload.get("schema") != schema:
            raise R198ProductionFactoryError(f"evaluation cg closure {name} schema is invalid")
        if required_status is not None and payload.get("status") != required_status:
            raise R198ProductionFactoryError(f"evaluation cg closure {name} is not passed")
        if name == "closure_manifest" and identity["sha256"] != R198_EVAL_CG_CLOSURE_MANIFEST_SHA256:
            raise R198ProductionFactoryError("evaluation cg closure manifest is not the final r198 record")
        if name == "metadata_parity" and identity["sha256"] != R198_EVAL_CG_METADATA_PARITY_SHA256:
            raise R198ProductionFactoryError("evaluation cg metadata parity is not the final r198 record")
        evidence[name] = identity
    engine = _closure_provenance_engine_identity(
        closure_payload.get("engine_artifact")
    )
    if any(engine.get(key) != library.get(key) for key in ("sha256", "bytes")):
        raise R198ProductionFactoryError("evaluation cg closure engine differs from snapshot-local DSO")
    build = _identity(
        closure_payload.get("pairing_build_artifact"), "evaluation cg closure build artifact"
    )
    if not _is_sha256(closure_payload.get("pairing_source_artifact_sha256")) or not _is_sha256(
        closure_payload.get("pairing_patch_artifact_sha256")
    ):
        raise R198ProductionFactoryError("evaluation cg closure lacks source/patch bindings")
    evidence = {**evidence, "engine_artifact": engine, "pairing_build_artifact": build}
    return _CGAssets(
        runtime_root=runtime,
        closure_manifest=closure,
        library=library,
        closure_payload=closure_payload,
        closure_evidence=evidence,
    )


def _closure_provenance_engine_identity(raw: Any) -> dict[str, Any]:
    """Read the closure receipt's relocatable 0444 DSO provenance record.

    This is intentionally different from the v2 capability's original
    ``PairingArtifactSet.engine_artifact``: that private source artifact is
    0555, while the exact byte-for-byte closure copy is sealed 0444 and is the
    only DSO a worker may load.  Both identities are independently rehashed
    and cross-bound by :func:`_crosscheck_cg_against_pairing`.
    """

    return _identity(
        raw,
        "evaluation cg closure engine artifact",
        exact_mode=0o444,
    )


def _verified_cg_tree_digest(
    identity: Mapping[str, Any],
    *,
    label: str,
    schema: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Re-derive one normalized evaluator CG tree digest from sealed bytes."""

    payload = _read_json(Path(identity["path"]), label)
    if payload.get("schema") != schema:
        raise R198ProductionFactoryError(f"{label} schema is invalid")
    file_count = payload.get("file_count")
    if type(file_count) is not int or file_count != len(_EVALUATION_CG_CLOSURE_TREE_PATHS):
        raise R198ProductionFactoryError(f"{label} file count is invalid")
    raw_files = _sequence(payload.get("files"), f"{label}.files")
    if len(raw_files) != len(_EVALUATION_CG_CLOSURE_TREE_PATHS):
        raise R198ProductionFactoryError(f"{label} file list length is invalid")
    files: list[dict[str, Any]] = []
    for index, raw_file in enumerate(raw_files):
        file_row = _mapping(raw_file, f"{label}.files[{index}]")
        relative_path = _text(
            file_row.get("relative_path"), f"{label}.files[{index}].relative_path"
        )
        sha256 = _text(file_row.get("sha256"), f"{label}.files[{index}].sha256")
        if not _is_sha256(sha256):
            raise R198ProductionFactoryError(f"{label}.files[{index}].sha256 is invalid")
        byte_count = file_row.get("bytes")
        # ``cg/__init__.py`` is intentionally an empty package marker in the
        # sealed closure.  Zero is a valid, checksum-bound file size; booleans
        # (``bool`` subclasses ``int``), negatives, and absent values are not.
        if type(byte_count) is not int or byte_count < 0:
            raise R198ProductionFactoryError(f"{label}.files[{index}].bytes is invalid")
        files.append(
            {
                "relative_path": relative_path,
                "sha256": sha256,
                "bytes": byte_count,
            }
        )
    if tuple(item["relative_path"] for item in files) != _EVALUATION_CG_CLOSURE_TREE_PATHS:
        raise R198ProductionFactoryError(f"{label} does not list the exact curated CG tree")
    material = {"schema": schema, "file_count": len(files), "files": files}
    tree_sha256 = _text(payload.get("tree_sha256"), f"{label}.tree_sha256")
    if not _is_sha256(tree_sha256) or _canonical_digest(material) != tree_sha256:
        raise R198ProductionFactoryError(f"{label} tree digest mismatch")
    return tree_sha256, files


def _verified_cg_metadata_evidence(cg: _CGAssets) -> dict[str, str]:
    """Re-derive normalized metadata evidence from the sealed parity record."""

    metadata_identity = cg.closure_evidence["metadata_parity"]
    metadata = _read_json(Path(metadata_identity["path"]), "evaluation cg metadata parity")
    if (
        metadata.get("schema") != EVALUATION_CG_METADATA_PARITY_SCHEMA
        or metadata.get("status") != "passed"
        or metadata.get("independent_processes") is not True
    ):
        raise R198ProductionFactoryError("evaluation cg metadata parity record is invalid")
    pairing_engine = _identity(
        metadata.get("pairing_engine"),
        "evaluation cg metadata pairing engine",
        exact_mode=None,
    )
    _identity(
        metadata.get("public_cg_engine"),
        "evaluation cg metadata public engine",
        exact_mode=None,
    )
    closure_engine = cg.closure_evidence["engine_artifact"]
    if any(
        pairing_engine.get(key) != closure_engine.get(key) for key in ("sha256", "bytes")
    ):
        raise R198ProductionFactoryError("evaluation cg metadata parity engine differs from closure")
    result: dict[str, str] = {}
    for key in (
        "all_card_canonical_sha256",
        "all_attack_canonical_sha256",
        "public_all_card_raw_sha256",
        "pairing_all_card_raw_sha256",
        "public_all_attack_raw_sha256",
        "pairing_all_attack_raw_sha256",
    ):
        value = _text(metadata.get(key), f"evaluation cg metadata {key}")
        if not _is_sha256(value):
            raise R198ProductionFactoryError(f"evaluation cg metadata {key} is invalid")
        result[key] = value
    if (
        result["public_all_card_raw_sha256"] != result["pairing_all_card_raw_sha256"]
        or result["public_all_attack_raw_sha256"]
        != result["pairing_all_attack_raw_sha256"]
    ):
        raise R198ProductionFactoryError("evaluation cg metadata raw values are not at parity")
    for key in (
        "public_initialized_before_pairing",
        "pairing_private_initialize_after_public_passed",
        "distinct_dso_handles",
    ):
        if metadata.get(key) is not True:
            raise R198ProductionFactoryError(f"evaluation cg metadata does not prove {key}")
    return result


def _verify_normalized_evaluation_cg_closure(
    value: Mapping[str, Any], *, cg: _CGAssets
) -> None:
    """Verify the evaluator-expanded closure evidence against the source snapshot.

    ``prepare_three_arm_manifest`` expands the strict base ``{receipt,
    runtime_library}`` pair with proof derived from the closure receipt.  A
    child receives that expanded, immutable manifest.  Reconstruct every
    added field here rather than trusting a prior normalizer or reducing it
    back to a path-only assertion.
    """

    if set(value) != _EVALUATION_CG_CLOSURE_NORMALIZED_KEYS:
        raise R198ProductionFactoryError(
            "prepared evaluation_cg_closure does not have the exact normalized evidence keys"
        )
    for name in (
        "engine_artifact",
        "pairing_build_artifact",
        "cg_source_manifest",
        "closure_manifest",
        "metadata_parity",
    ):
        expected = cg.closure_evidence[name]
        expected_path = _physical_path(
            expected["path"], f"sealed evaluation CG {name}", file=True
        )
        observed = _identity(
            value.get(name),
            f"evaluation_cg_closure.{name}",
            exact_mode=stat.S_IMODE(os.lstat(expected_path).st_mode),
        )
        _identity_equal(observed, expected, f"normalized evaluation CG {name}")

    source_tree_sha256, _ = _verified_cg_tree_digest(
        cg.closure_evidence["cg_source_manifest"],
        label="evaluation cg source manifest",
        schema=(
            "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_source_manifest/v1"
        ),
    )
    closure_tree_sha256, closure_files = _verified_cg_tree_digest(
        cg.closure_evidence["closure_manifest"],
        label="evaluation cg closure tree manifest",
        schema=EVALUATION_CG_CLOSURE_MANIFEST_SCHEMA,
    )
    closure_library = next(
        item for item in closure_files if item["relative_path"] == "libcg.so"
    )
    if any(closure_library[key] != cg.library[key] for key in ("sha256", "bytes")):
        raise R198ProductionFactoryError("evaluation CG closure tree libcg.so differs from runtime library")
    metadata = _verified_cg_metadata_evidence(cg)
    canonical_abi_sha256 = _text(
        cg.closure_payload.get("canonical_abi_sha256"),
        "evaluation cg closure canonical ABI digest",
    )
    if not _is_sha256(canonical_abi_sha256):
        raise R198ProductionFactoryError("evaluation cg closure canonical ABI digest is invalid")
    expected_scalars = {
        "canonical_abi_sha256": canonical_abi_sha256,
        "sim_initializer_symbol": "RtpPairingSnapshotInitialize",
        "snapshot_abi_version": 2,
        "cg_source_tree_sha256": source_tree_sha256,
        "closure_tree_sha256": closure_tree_sha256,
        "all_card_canonical_sha256": metadata["all_card_canonical_sha256"],
        "all_attack_canonical_sha256": metadata["all_attack_canonical_sha256"],
    }
    for name, expected in expected_scalars.items():
        if value.get(name) != expected:
            raise R198ProductionFactoryError(
                f"normalized evaluation CG closure differs from sealed evidence at {name}"
            )


def _verify_manifest_evaluation_cg_closure(
    manifest: Mapping[str, Any],
    *,
    cg: _CGAssets,
    source_root: Path,
) -> None:
    """Bind the generic evaluator's closure record to the loaded 0444 copy.

    The closure receipt is private provenance, while ``runtime_library`` is
    the relocated snapshot-local DSO that ``CG_LIB_PATH/cg/libcg.so`` and the
    wrapper must actually load.  Both are required so equal bytes at two paths
    cannot silently turn into an ambiguous loader choice.
    """

    value = _mapping(manifest.get("evaluation_cg_closure"), "evaluation_cg_closure")
    key_set = frozenset(value)
    final_manifest = manifest.get("schema") == THREE_ARM_EVALUATION_MANIFEST_SCHEMA
    if key_set not in {
        _EVALUATION_CG_CLOSURE_BASE_KEYS,
        _EVALUATION_CG_CLOSURE_NORMALIZED_KEYS,
    }:
        raise R198ProductionFactoryError(
            "evaluation_cg_closure must be the exact base pair or full evaluator-normalized record"
        )
    if final_manifest and key_set != _EVALUATION_CG_CLOSURE_NORMALIZED_KEYS:
        raise R198ProductionFactoryError(
            "prepared evaluator manifest must contain full normalized evaluation CG evidence"
        )
    if not final_manifest and key_set != _EVALUATION_CG_CLOSURE_BASE_KEYS:
        raise R198ProductionFactoryError(
            "only a strict receipt/runtime_library pair is allowed before evaluator preparation"
        )
    receipt = _identity(
        value.get("receipt"), "evaluation_cg_closure.receipt", root=source_root
    )
    runtime_library = _identity(
        value.get("runtime_library"),
        "evaluation_cg_closure.runtime_library",
        root=cg.runtime_root,
    )
    _identity_equal(receipt, cg.closure_manifest, "evaluation CG closure receipt")
    _identity_equal(
        runtime_library,
        cg.library,
        "evaluation CG closure runtime library",
    )
    if Path(runtime_library["path"]) != cg.runtime_root / "cg" / "libcg.so":
        raise R198ProductionFactoryError(
            "evaluation CG runtime_library is not CG_LIB_PATH/cg/libcg.so"
        )
    if key_set == _EVALUATION_CG_CLOSURE_NORMALIZED_KEYS:
        _verify_normalized_evaluation_cg_closure(value, cg=cg)


def _factory_inputs(
    manifest: Mapping[str, Any],
    *,
    require_evaluation_cg_closure: bool = True,
    validate_sidecar_payload: bool = True,
) -> _FactoryInputs:
    spec = _factory_spec(manifest)
    source_root, tree_sha, source_snapshot_manifest = _source_root(spec)
    authority = _authority(spec)
    candidate = _candidate_assets(
        spec,
        source_root,
        tree_sha,
        validate_sidecar_payload=validate_sidecar_payload,
    )
    matchup_adapter_registry, matchup_adapter_registry_digest = (
        _matchup_adapter_registry(
            spec,
            source_root=source_root,
            source_snapshot_manifest=source_snapshot_manifest,
        )
    )
    cg = _cg_assets(spec, source_root)
    if require_evaluation_cg_closure:
        _verify_manifest_evaluation_cg_closure(
            manifest,
            cg=cg,
            source_root=source_root,
        )
    input_root: Path | None = None
    if spec.get("evaluation_inputs_root") is not None:
        input_root = _physical_path(
            spec.get("evaluation_inputs_root"),
            "production_factory.evaluation_inputs_root",
            directory=True,
        )
        _assert_readonly_tree(input_root, "r198 sealed evaluation inputs")
    return _FactoryInputs(
        spec=spec,
        candidate=candidate,
        cg=cg,
        matchup_adapter_registry=matchup_adapter_registry,
        matchup_adapter_registry_digest=matchup_adapter_registry_digest,
        authority=authority,
        factory_identity=candidate.factory_module,
        source_snapshot_manifest=source_snapshot_manifest,
        evaluation_inputs_root=input_root,
    )


def r198_factory_evaluation_authority_payload() -> dict[str, Any]:
    """Return the non-circular, factory-local r198 evaluation authority.

    This is deliberately *not* the final manifest-bound runner authority: the
    latter cannot exist until the materializer has sealed snapshots, preflight,
    and the final evaluator manifest.  It is only a fixed negative-authority
    declaration embedded in the factory input and is derived here so callers
    cannot hand-author a broader authority during that circular construction.
    """

    return {
        "schema": EVALUATION_AUTHORITY_SCHEMA,
        "status": "authorized_evaluation_only",
        "scope": "r198_factory_preparation_and_evaluation_only",
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_change_authorized": False,
        "selector_change_authorized": False,
        "action_authority_authorized": False,
        "kaggle_submission_authorized": False,
    }


def r198_runtime_profile_payload(arm: str) -> dict[str, Any]:
    """Return the only permitted r198 arm-profile payload for sealing.

    The evaluation-input materializer owns the immutable file publication and
    supplies the resulting file identities to the generic manifest compiler.
    Keeping the payload constructor here prevents a staging script from
    independently hand-authoring subtly different direct/recursive controls.
    """

    if arm not in CANONICAL_ARMS:
        raise R198ProductionFactoryError(f"unknown r198 evaluation arm: {arm}")
    enabled, direct, forced = {
        "no_rtp": (False, False, False),
        "direct_bridge_recursive_disabled": (True, True, True),
        "recursive_rtp": (True, True, False),
    }[arm]
    return {
        "schema": "poke_bot.recursive_turn_planner.r198_arm_runtime_profile/v1",
        "evaluation_arm": arm,
        "sizing_profile": R198_PROFILE,
        "recursive_turn_planner_enabled": enabled,
        "direct_bridge_enabled": direct,
        "force_direct_bridge_only": forced,
        "max_neural_passes": R198_MAX_NEURAL_PASSES,
        "max_action_combos": R198_MAX_ACTION_COMBOS,
        "num_plan_candidates": R198_NUM_CANDIDATES,
        "max_recursion_depth": R198_MAX_RECURSION_DEPTH,
        "max_plan_length": 12,
        "d_model": 96,
        "dynamics_width": 192,
        "complexity_option_threshold": 8,
        "complexity_entropy_threshold": 1.5,
        "online_sim_verify_budget": 0,
        "repair_budget": 1,
        "compute_cost_penalty": 0.01,
        "option_batch_hint": 64,
        "prefer_option_hidden": True,
        "policy_aid_cap": 0.25,
        "normal_recursive_plan_passes": R198_NORMAL_PASSES,
        "forced_replan_passes": R198_FORCED_REPLAN_PASSES,
        "recursive_repairs_enabled": arm == "recursive_rtp",
        "training_eligible": False,
        "replay_eligible": False,
        "serving_eligible": False,
        "action_authority_enabled": False,
    }


def build_r198_evaluator_base_spec(
    *,
    source_snapshot_root: str | Path,
    source_tree_sha256: str,
    pairing_capability: Mapping[str, Any],
    source_snapshot_manifest: Mapping[str, Any] | None = None,
    evaluation_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive the evaluator's frozen base spec from physical snapshot bytes.

    This is the public hand-off API for the r198 input materializer/stage.  It
    does no publication and mutates nothing.  The materializer must seal the
    three returned ``runtime_profile_payload`` values as 0444 files beneath
    its declared ``production_factory.evaluation_inputs_root``, then replace
    each payload field with the resulting immutable ``runtime_profile``
    identity before compiling the final v2 three-arm manifest.

    ``pairing_capability`` is an immutable identity for the independently
    sealed v2 private pairing capability receipt.  Its engine/source/patch/
    build identities are copied into the returned production-factory binding
    only after rehashing/cross-checking the capability receipt.
    """

    source_root = _physical_path(source_snapshot_root, "r198 source snapshot root", directory=True)
    _assert_readonly_tree(source_root, "r198 source snapshot")
    tree_digest = _text(source_tree_sha256, "r198 source snapshot tree digest")
    if not _is_sha256(tree_digest):
        raise R198ProductionFactoryError("r198 source snapshot tree digest is invalid")
    source_manifest_path = source_root / SOURCE_SNAPSHOT_MANIFEST_NAME
    source_manifest_identity = {
        "path": str(_physical_path(source_manifest_path, "r198 source snapshot manifest", file=True)),
        "sha256": _sha256_file(source_manifest_path),
        "bytes": source_manifest_path.stat().st_size,
    }
    _readonly_mode(source_manifest_path, "r198 source snapshot manifest", exact=0o444)
    if source_snapshot_manifest is not None:
        supplied_manifest = _identity(
            source_snapshot_manifest, "r198 supplied source snapshot manifest", root=source_root
        )
        _identity_equal(
            supplied_manifest, source_manifest_identity, "r198 supplied/canonical source snapshot manifest"
        )
    source_manifest_payload = _read_json(source_manifest_path, "r198 source snapshot manifest")
    if (
        source_manifest_payload.get("schema") != SOURCE_SNAPSHOT_SCHEMA
        or source_manifest_payload.get("source_tree_sha256") != tree_digest
    ):
        raise R198ProductionFactoryError("r198 source snapshot manifest is not bound to the requested tree")
    authority = r198_factory_evaluation_authority_payload()
    if evaluation_authority is not None and _mapping(
        evaluation_authority, "r198 supplied evaluation authority"
    ) != authority:
        raise R198ProductionFactoryError(
            "r198 base builder derives the only permitted non-circular factory authority"
        )
    _authority({"evaluation_authority": authority})
    factory_path = _physical_path(__file__, "r198 factory module", file=True)
    _inside(factory_path, source_root, "r198 factory module")
    factory_identity = {
        "path": str(factory_path),
        "sha256": _sha256_file(factory_path),
        "bytes": factory_path.stat().st_size,
    }
    _readonly_mode(factory_path, "r198 factory module", exact=0o444)
    candidate_manifest_path = source_root / "evaluation-artifacts" / "r197-candidate" / "manifest.json"
    candidate_identity = {
        "path": str(_physical_path(candidate_manifest_path, "r198 candidate snapshot", file=True)),
        "sha256": _sha256_file(candidate_manifest_path),
        "bytes": candidate_manifest_path.stat().st_size,
    }
    _readonly_mode(candidate_manifest_path, "r198 candidate snapshot", exact=0o444)
    cg_root = source_root / "kaggle" / "input" / "rtp-eval-cg"
    closure_path = cg_root / EVALUATION_CG_CLOSURE_FILENAME
    library_path = cg_root / "cg" / "libcg.so"
    closure_identity = {
        "path": str(_physical_path(closure_path, "r198 eval cg closure", file=True)),
        "sha256": _sha256_file(closure_path),
        "bytes": closure_path.stat().st_size,
    }
    library_identity = {
        "path": str(_physical_path(library_path, "r198 eval cg DSO", file=True)),
        "sha256": _sha256_file(library_path),
        "bytes": library_path.stat().st_size,
    }
    _readonly_mode(closure_path, "r198 eval cg closure", exact=0o444)
    _readonly_mode(library_path, "r198 eval cg DSO", exact=0o444)
    registry_path = source_root / MATCHUP_ADAPTER_REGISTRY_RELATIVE
    registry_identity = _identity(
        {
            "path": str(registry_path),
            "sha256": _sha256_file(
                _physical_path(
                    registry_path,
                    "r198 snapshot-local matchup adapter registry",
                    file=True,
                )
            ),
            "bytes": registry_path.stat().st_size,
            "mode": 0o444,
        },
        "r198 snapshot-local matchup adapter registry",
        root=source_root,
        exact_mode=0o444,
    )
    pairing = _identity(pairing_capability, "r198 pairing capability")
    capability = _read_json(Path(pairing["path"]), "r198 pairing capability")
    if capability.get("schema") != PAIRING_CAPABILITY_SCHEMA or capability.get("status") != "available":
        raise R198ProductionFactoryError("r198 pairing capability is not an available v2 receipt")
    pairing_artifacts = {
        name: _identity(
            capability.get(name),
            f"r198 pairing {name}",
            exact_mode=0o555 if name == "engine_artifact" else 0o444,
        )
        for name in ("engine_artifact", "source_artifact", "patch_artifact", "build_artifact")
    }
    candidate_payload = _read_json(candidate_manifest_path, "r198 candidate snapshot")
    candidate_artifacts = _mapping(candidate_payload.get("artifacts"), "r198 candidate snapshot artifacts")
    required_candidate_artifacts = {
        "parent_checkpoint",
        "sidecar",
        "sidecar_receipt",
        "completion_receipt",
        "deck",
        "matchup_tree",
    }
    if set(candidate_artifacts) != required_candidate_artifacts:
        raise R198ProductionFactoryError("r198 candidate snapshot artifact set is incomplete")
    frozen_candidate_artifacts = {
        name: _identity(candidate_artifacts[name], f"r198 candidate artifact {name}", root=source_root)
        for name in sorted(required_candidate_artifacts)
    }
    production_factory = {
        "schema": FACTORY_INPUT_SCHEMA,
        "status": "sealed",
        "source_snapshot_root": str(source_root),
        "source_tree_sha256": tree_digest,
        "source_snapshot_manifest": source_manifest_identity,
        "factory_module": factory_identity,
        "candidate_snapshot": candidate_identity,
        "matchup_adapter_registry": registry_identity,
        "evaluation_cg": {
            "runtime_root": str(cg_root),
            "closure_manifest": closure_identity,
            "library": library_identity,
        },
        "evaluation_authority": authority,
        "artifacts": frozen_candidate_artifacts,
        "pairing_artifacts": pairing_artifacts,
    }
    _matchup_adapter_registry(
        production_factory,
        source_root=source_root,
        source_snapshot_manifest=source_manifest_identity,
    )
    snapshot_cg = _mapping(
        source_manifest_payload.get("eval_cg_closure"),
        "r198 source snapshot eval-cg closure",
    )
    for name, expected in (("closure_manifest", closure_identity), ("library", library_identity)):
        observed = _identity(
            snapshot_cg.get(name), f"r198 source snapshot eval-cg {name}", root=source_root
        )
        _identity_equal(observed, expected, f"r198 source snapshot/factory eval-cg {name}")
    candidate_assets = _candidate_assets(production_factory, source_root, tree_digest)
    cg_assets = _cg_assets(production_factory, source_root)
    checked_pairing, pairing_info = _pairing_artifacts(
        {
            "production_factory": production_factory,
            "pairing_capability": {"receipt": pairing},
        },
        source_root,
    )
    _crosscheck_cg_against_pairing(cg_assets, checked_pairing, pairing_info)
    shared = {
        name: dict(getattr(candidate_assets, name))
        for name in (
            "parent_checkpoint",
            "deck",
            "matchup_tree",
            "completion_receipt",
        )
    }
    shared["r197_completion_receipt"] = shared.pop("completion_receipt")
    opponents: list[dict[str, Any]] = []
    for opponent_id, content_digest in OFFICIAL_CONTROL_DIGESTS.items():
        package_manifest_path = (
            source_root / "evaluation-artifacts" / "official-control-manifests" / f"{opponent_id}.json"
        )
        physical_package_manifest = _physical_path(
            package_manifest_path,
            f"r198 {opponent_id} package manifest",
            file=True,
        )
        package_identity = _identity(
            {
                "path": str(physical_package_manifest),
                "sha256": _sha256_file(physical_package_manifest),
                "bytes": physical_package_manifest.stat().st_size,
            },
            f"r198 {opponent_id} package manifest",
            root=source_root,
        )
        sealed_package = _sealed_official_package_tree(
            package_identity,
            source_root=source_root,
            opponent_id=opponent_id,
            content_digest=content_digest,
        )
        opponents.append(
            {
                "id": opponent_id,
                "content_digest": content_digest,
                "artifact": dict(sealed_package.manifest),
                "package_root": str(sealed_package.package_root),
                "deck": dict(sealed_package.entries["deck.csv"]),
            }
        )
    sidecar = frozen_candidate_artifacts["sidecar"]
    binding = {
        "schema": "poke_bot.recursive_turn_planner.r198_candidate_evaluation_binding/v1",
        "status": "bound",
        "candidate_contract_sha256": R198_CANDIDATE_CONTRACT_SHA256,
        "parent_checkpoint_sha256": R195_PARENT_SHA256,
        "sidecar_sha256": R197_SIDECAR_SHA256,
        "sidecar_config_sha256": R197_SIDECAR_CONFIG_SHA256,
        "deck_file_sha256": R195_DECK_CSV_SHA256,
        "deck_cards_sha256": R195_DECK_CARDS_SHA256,
        "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
        "sizing_profile": R198_PROFILE,
        "max_neural_passes": R198_MAX_NEURAL_PASSES,
        "max_action_combos": R198_MAX_ACTION_COMBOS,
        "required_neural_passes": {
            "normal": R198_NORMAL_PASSES,
            "forced_replan": R198_FORCED_REPLAN_PASSES,
        },
    }
    return {
        "factory": "poke_bot.rtp_r198_production_factory:ProductionR198EvaluationFactory",
        "production_factory": production_factory,
        "pairing_capability": {"receipt": pairing},
        "evaluation_cg_closure": {
            "receipt": closure_identity,
            # This is purposefully a separate identity from the private
            # capability engine: same verified bytes, snapshot-local 0444
            # path, and the only library a worker is allowed to dlopen.
            "runtime_library": library_identity,
        },
        "shared_artifacts": shared,
        "candidate_evaluation_binding": binding,
        "arms": {
            arm: {
                "runtime_artifact": dict(factory_identity),
                "runtime_profile_payload": r198_runtime_profile_payload(arm),
                **(
                    {"rtp_sidecar": dict(sidecar)}
                    if arm != "no_rtp"
                    else {}
                ),
            }
            for arm in CANONICAL_ARMS
        },
        "opponents": opponents,
    }


def _arm_spec(
    manifest: Mapping[str, Any], arm: str, inputs: _FactoryInputs
) -> dict[str, Any]:
    if arm not in CANONICAL_ARMS:
        raise R198ProductionFactoryError(f"unknown r198 evaluation arm: {arm}")
    candidate = inputs.candidate
    profile_root = inputs.evaluation_inputs_root
    if profile_root is None:
        raise R198ProductionFactoryError(
            "r198 arm runtime profiles must be sealed beneath evaluation_inputs_root"
        )
    raw_arms = _mapping(manifest.get("arms"), "manifest arms")
    spec = _mapping(raw_arms.get(arm), f"manifest arm {arm}")
    runtime_artifact = _identity(spec.get("runtime_artifact"), f"{arm} runtime artifact", root=candidate.source_root)
    runtime_profile = _identity(spec.get("runtime_profile"), f"{arm} runtime profile", root=profile_root)
    profile_envelope = _read_json(Path(runtime_profile["path"]), f"{arm} runtime profile")
    profile = _mapping(profile_envelope.get("rtp", profile_envelope), f"{arm} runtime profile rtp")
    expected_flags = {
        "no_rtp": (False, False, False),
        "direct_bridge_recursive_disabled": (True, True, True),
        "recursive_rtp": (True, True, False),
    }[arm]
    if profile.get("evaluation_arm") != arm:
        raise R198ProductionFactoryError(f"{arm} runtime profile arm differs")
    flags = tuple(
        profile.get(key)
        for key in (
            "recursive_turn_planner_enabled",
            "direct_bridge_enabled",
            "force_direct_bridge_only",
        )
    )
    if flags != expected_flags:
        raise R198ProductionFactoryError(f"{arm} runtime profile flags differ")
    if profile.get("schema") not in {None, "poke_bot.recursive_turn_planner.r198_arm_runtime_profile/v1"}:
        raise R198ProductionFactoryError(f"{arm} runtime profile schema differs")
    if profile.get("recursive_repairs_enabled") is not (arm == "recursive_rtp"):
        raise R198ProductionFactoryError(f"{arm} runtime profile repair behavior differs")
    if arm == "no_rtp":
        if spec.get("rtp_sidecar") is not None:
            raise R198ProductionFactoryError("no-RTP arm must not carry an RTP sidecar")
        return {
            "runtime_artifact": runtime_artifact,
            "runtime_profile": runtime_profile,
            "profile": profile,
            "rtp_sidecar": None,
        }
    required_profile = {
        "sizing_profile": R198_PROFILE,
        "max_neural_passes": R198_MAX_NEURAL_PASSES,
        "max_action_combos": R198_MAX_ACTION_COMBOS,
        "num_plan_candidates": R198_NUM_CANDIDATES,
        "max_recursion_depth": R198_MAX_RECURSION_DEPTH,
    }
    for key, expected in required_profile.items():
        if profile.get(key) != expected:
            raise R198ProductionFactoryError(f"{arm} runtime profile fails {key}")
    sidecar = _identity(spec.get("rtp_sidecar"), f"{arm} RTP sidecar", root=candidate.package_root)
    _identity_equal(sidecar, candidate.sidecar, f"{arm} RTP sidecar/candidate sidecar")
    return {
        "runtime_artifact": runtime_artifact,
        "runtime_profile": runtime_profile,
        "profile": profile,
        "rtp_sidecar": sidecar,
    }


def _shared_assets(
    manifest: Mapping[str, Any], inputs: _FactoryInputs
) -> dict[str, dict[str, Any]]:
    """Rehash the exact seven shared identities named by the r198 runner.

    These names are receipt fields: the runner deterministically expects
    ``<shared name>_sha256`` in every arm runtime identity.  Candidate-owned
    files are additionally cross-bound to the candidate snapshot, while the
    cohort/preflight and official registry are constrained to their canonical
    sealed roots so an ambient identity cannot satisfy the naming contract.
    """

    raw = _mapping(manifest.get("shared_artifacts"), "manifest shared_artifacts")
    if set(raw) != set(R198_SHARED_ARTIFACT_NAMES):
        raise R198ProductionFactoryError(
            "manifest shared_artifacts are not the exact r198 identity set"
        )
    candidate = inputs.candidate
    result: dict[str, dict[str, Any]] = {}
    for name, expected in (
        ("parent_checkpoint", candidate.parent_checkpoint),
        ("deck", candidate.deck),
        ("matchup_tree", candidate.matchup_tree),
        ("r197_completion_receipt", candidate.completion_receipt),
    ):
        identity = _identity(raw.get(name), f"shared {name}")
        _identity_equal(identity, expected, f"shared {name}/candidate snapshot")
        result[name] = identity

    fixture_root = inputs.evaluation_inputs_root
    if fixture_root is None or fixture_root.name != "preflight-fixture-inputs":
        raise R198ProductionFactoryError(
            "production factory lacks the canonical sealed evaluation-input root"
        )
    evaluation_root = _physical_path(
        fixture_root.parent,
        "r198 materialized evaluation root",
        directory=True,
    )
    _readonly_mode(evaluation_root, "r198 materialized evaluation root", exact=0o555)
    generated = {
        "evaluation_only_cohort": (
            evaluation_root / "cohort" / "evaluation-only-cohort.json",
            "poke_bot.recursive_turn_planner.r197_evaluation_only_cohort/v1",
            "frozen",
        ),
        "planner_preflight_receipt": (
            evaluation_root / "preflight" / "planner-pass-preflight.json",
            PREFLIGHT_RECEIPT_SCHEMA,
            "passed",
        ),
    }
    for name, (expected_path, schema, status) in generated.items():
        identity = _identity(
            raw.get(name),
            f"shared {name}",
            root=evaluation_root,
        )
        if Path(identity["path"]) != expected_path:
            raise R198ProductionFactoryError(
                f"shared {name} is not the canonical materialized artifact"
            )
        payload = _read_json(expected_path, f"shared {name}")
        if payload.get("schema") != schema or payload.get("status") != status:
            raise R198ProductionFactoryError(
                f"shared {name} schema/status is invalid"
            )
        result[name] = identity

    registry = _identity(
        raw.get("research_control_registry"),
        "shared research_control_registry",
        root=candidate.source_root,
    )
    if (
        Path(registry["path"])
        != candidate.source_root / "ops" / "research_control_registry_v1.json"
        or registry["sha256"] != RESEARCH_CONTROL_REGISTRY_SHA256
        or registry["bytes"] != RESEARCH_CONTROL_REGISTRY_BYTES
    ):
        raise R198ProductionFactoryError(
            "shared research_control_registry is not the canonical snapshot artifact"
        )
    registry_payload = _read_json(
        Path(registry["path"]), "shared research_control_registry"
    )
    if (
        registry_payload.get("schema") != "poke_bot.research_control_registry/v1"
        or registry_payload.get("registry_id") != "alakazam-research-controls"
        or registry_payload.get("version") != 1
    ):
        raise R198ProductionFactoryError(
            "shared research_control_registry semantics are invalid"
        )
    result["research_control_registry"] = registry
    if set(result) != set(R198_SHARED_ARTIFACT_NAMES):  # defensive completeness.
        raise R198ProductionFactoryError("shared r198 artifact validation is incomplete")
    return result


def _runtime_shared_artifact_sha256s(
    shared: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    if set(shared) != set(R198_SHARED_ARTIFACT_NAMES):
        raise R198ProductionFactoryError(
            "runtime shared artifacts are not the exact r198 identity set"
        )
    result: dict[str, str] = {}
    for name in R198_SHARED_ARTIFACT_NAMES:
        digest = shared[name].get("sha256")
        if not _is_sha256(digest):
            raise R198ProductionFactoryError(
                f"runtime shared artifact {name} has no valid SHA-256"
            )
        result[f"{name}_sha256"] = str(digest)
    return result


def _safe_official_package_relative_path(raw: Any, label: str) -> tuple[str, Path]:
    """Accept a manifest-relative package file name, never an ambient path."""

    relative = _text(raw, label)
    parsed = PurePosixPath(relative)
    if (
        parsed.is_absolute()
        or relative.startswith(("/", "\\"))
        or any(part in {"", ".", ".."} for part in relative.split("/"))
        or any(part in {".", ".."} for part in parsed.parts)
    ):
        raise R198ProductionFactoryError(f"{label} is unsafe")
    return relative, Path(*parsed.parts)


def _resolve_official_package_entry(
    raw: Any,
    *,
    package_root: Path,
    label: str,
) -> tuple[str, dict[str, Any]]:
    """Resolve one sealed relative manifest entry below its physical package root."""

    entry = _mapping(raw, label)
    if set(entry) != {"path", "sha256", "bytes"}:
        raise R198ProductionFactoryError(f"{label} must be an exact path/SHA-256/byte mapping")
    relative, relative_path = _safe_official_package_relative_path(
        entry.get("path"), f"{label}.path"
    )
    resolved_path = _physical_path(
        package_root / relative_path, f"{label} resolved path", file=True
    )
    _inside(resolved_path, package_root, f"{label} resolved path")
    identity = _identity(
        {
            "path": str(resolved_path),
            "sha256": entry.get("sha256"),
            "bytes": entry.get("bytes"),
        },
        label,
        root=package_root,
    )
    if resolved_path.relative_to(package_root).as_posix() != relative:
        raise R198ProductionFactoryError(f"{label} path differs after physical resolution")
    return relative, identity


def _actual_official_package_entries(package_root: Path, label: str) -> list[dict[str, Any]]:
    """Inventory the full physical package after readonly/no-symlink validation."""

    actual: list[dict[str, Any]] = []
    for current_raw, directories, files in os.walk(
        package_root, topdown=True, followlinks=False
    ):
        current = _physical_path(current_raw, f"{label} directory", directory=True)
        directories.sort()
        files.sort()
        if directories:
            raise R198ProductionFactoryError(f"{label} has an unexpected directory")
        for name in files:
            path = _physical_path(current / name, f"{label} file", file=True)
            actual.append(
                {
                    "path": path.relative_to(package_root).as_posix(),
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    return sorted(actual, key=lambda row: row["path"])


def _sealed_official_package_tree(
    raw_manifest: Any,
    *,
    source_root: Path,
    opponent_id: str,
    content_digest: str,
) -> _SealedOfficialPackageTree:
    """Verify the one canonical read-only baseline package in a source snapshot.

    The publisher intentionally stores ``entries[*].path`` relative to the
    package root.  This resolver converts those names to physical paths only
    after validating the canonical root; it never lets an entry resolve via
    the process CWD or an installed baseline package.
    """

    artifact = _identity(raw_manifest, "official package manifest", root=source_root)
    expected_manifest = (
        source_root
        / "evaluation-artifacts"
        / "official-control-manifests"
        / f"{opponent_id}.json"
    )
    if Path(artifact["path"]) != expected_manifest:
        raise R198ProductionFactoryError("official package manifest is not canonical snapshot evidence")
    payload = _read_json(Path(artifact["path"]), "official package manifest")
    expected_fields = {
        "schema",
        "status",
        "opponent_id",
        "content_digest",
        "no_symlinks",
        "all_paths_read_only",
        "package_root",
        "entries",
        "tree_entries_sha256",
        "deck_sha256",
        "deck_order_sha256",
    }
    if set(payload) != expected_fields:
        raise R198ProductionFactoryError("official package manifest has an unexpected schema")
    if payload.get("schema") != PACKAGE_SNAPSHOT_SCHEMA or payload.get("status") != "sealed":
        raise R198ProductionFactoryError("official package manifest is not sealed")
    if payload.get("no_symlinks") is not True or payload.get("all_paths_read_only") is not True:
        raise R198ProductionFactoryError("official package lacks physical readonly attestation")
    if payload.get("opponent_id") != opponent_id or payload.get("content_digest") != content_digest:
        raise R198ProductionFactoryError("official package identity differs from expected opponent")
    package_root = _physical_path(
        payload.get("package_root"), "official package root", directory=True
    )
    _inside(package_root, source_root, "official package root")
    expected_root = source_root / "baselines" / "official" / opponent_id
    if package_root != expected_root:
        raise R198ProductionFactoryError("official package root is not the sealed snapshot-local package")
    _assert_readonly_tree(package_root, f"official package {opponent_id}")
    raw_entries = _sequence(payload.get("entries"), "official package entries")
    entries: dict[str, dict[str, Any]] = {}
    for raw_entry in raw_entries:
        relative, identity = _resolve_official_package_entry(
            raw_entry,
            package_root=package_root,
            label="official package entry",
        )
        if relative in entries:
            raise R198ProductionFactoryError("official package repeats an entry")
        entries[relative] = identity
    if set(entries) != {"main.py", "deck.csv"}:
        raise R198ProductionFactoryError("official package has an unexpected importable tree")
    canonical_entries = [
        {"path": name, "sha256": entries[name]["sha256"], "bytes": entries[name]["bytes"]}
        for name in sorted(entries)
    ]
    if payload.get("entries") != canonical_entries:
        raise R198ProductionFactoryError("official package entries are not canonical")
    if _canonical_digest(canonical_entries) != payload.get("tree_entries_sha256"):
        raise R198ProductionFactoryError("official package tree digest changed")
    if _actual_official_package_entries(
        package_root, f"official package {opponent_id}"
    ) != canonical_entries:
        raise R198ProductionFactoryError("official package physical tree differs from manifest")
    deck = entries["deck.csv"]
    if (
        payload.get("deck_sha256") != deck["sha256"]
        or payload.get("deck_order_sha256") != deck["sha256"]
    ):
        raise R198ProductionFactoryError("official package deck bindings changed")
    return _SealedOfficialPackageTree(
        manifest=artifact,
        payload=payload,
        package_root=package_root,
        entries=entries,
    )


def _official_package(manifest: Mapping[str, Any], opponent_id: str, source_root: Path) -> _OfficialPackage:
    if opponent_id not in OFFICIAL_CONTROL_DIGESTS:
        raise R198ProductionFactoryError("evaluation cell does not name an official r198 control")
    found: dict[str, Any] | None = None
    for raw in _sequence(manifest.get("opponents"), "manifest opponents"):
        row = _mapping(raw, "manifest opponent")
        if row.get("id") == opponent_id:
            if found is not None:
                raise R198ProductionFactoryError("manifest repeats official opponent")
            found = row
    if found is None:
        raise R198ProductionFactoryError("cell opponent is not in manifest")
    expected_digest = OFFICIAL_CONTROL_DIGESTS[opponent_id]
    if found.get("content_digest") != expected_digest:
        raise R198ProductionFactoryError("official opponent content digest differs")
    sealed = _sealed_official_package_tree(
        found.get("artifact"),
        source_root=source_root,
        opponent_id=opponent_id,
        content_digest=expected_digest,
    )
    if found.get("package_root") != str(sealed.package_root):
        raise R198ProductionFactoryError("official opponent package root differs from sealed manifest")
    deck = sealed.entries["deck.csv"]
    explicit_deck = found.get("deck")
    if explicit_deck is not None:
        # The base spec carries this redundant convenience identity, but the
        # generic evaluator deliberately normalizes opponents down to their
        # sealed package evidence.  When present it must still agree exactly;
        # when omitted, the package tree above is the sole deck authority.
        deck_binding = _identity(
            explicit_deck, "official opponent deck binding", root=sealed.package_root
        )
        _identity_equal(deck_binding, deck, "official opponent deck binding")
    return _OfficialPackage(
        opponent_id=opponent_id,
        content_digest=expected_digest,
        manifest=sealed.manifest,
        payload=sealed.payload,
        package_root=sealed.package_root,
        main_py=sealed.entries["main.py"],
        deck=deck,
    )


_BASELINE_ENV_LOCK = threading.RLock()
_MISSING = object()
_BASELINE_ENV_KEYS = (
    "CG_LIB_PATH",
    "POKEBOT_MATCHUP_ADAPTER_RUNTIME",
    "POKEBOT_PUBLIC_MATCHUP_TREE_PATH",
    "POKEBOT_USE_RECURSIVE_TURN_PLANNER",
    "POKEBOT_RTP_CHECKPOINT",
    "POKEBOT_RTP_SIZING_PROFILE",
    "POKEBOT_RTP_MAX_ACTION_COMBOS",
    "POKEBOT_RTP_SERVING_QUALIFIED",
    "POKEBOT_RTP_PARENT_CHECKPOINT_SHA256",
)


@contextlib.contextmanager
def _baseline_environment_guard() -> Iterator[None]:
    """Restore candidate-owned env if a frozen baseline mutates globals."""

    with _BASELINE_ENV_LOCK:
        prior = {key: os.environ.get(key, _MISSING) for key in _BASELINE_ENV_KEYS}
        try:
            yield
        finally:
            for key, value in prior.items():
                if value is _MISSING:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = str(value)


def _load_snapshot_baseline(package: _OfficialPackage) -> tuple[Callable[[dict[str, Any]], list[int]], list[int], str]:
    """Import exactly one read-only snapshot package under a unique module name."""

    from .deck_pool import read_deck

    try:
        deck = read_deck(package.deck["path"])
    except Exception as exc:
        raise R198ProductionFactoryError("official baseline deck is invalid") from exc
    module_name = "_r198_snapshot_baseline_" + package.opponent_id.replace("-", "_") + "_" + uuid.uuid4().hex
    import_spec = importlib.util.spec_from_file_location(module_name, package.main_py["path"])
    if import_spec is None or import_spec.loader is None:
        raise R198ProductionFactoryError("cannot create snapshot baseline import spec")
    module = importlib.util.module_from_spec(import_spec)
    sys.modules[module_name] = module
    previous_directory = os.getcwd()
    try:
        # Some official packages intentionally use a relative deck.csv.  The
        # package root is sealed and has exactly two entries, so chdir cannot
        # expose an installed/live package.
        with _baseline_environment_guard():
            os.chdir(str(package.package_root))
            import_spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise R198ProductionFactoryError(
            f"snapshot official baseline import failed: {package.opponent_id}"
        ) from exc
    finally:
        os.chdir(previous_directory)
    candidate = getattr(module, "agent", None)
    if not callable(candidate):
        raise R198ProductionFactoryError("snapshot official baseline has no agent()")

    def isolated(observation: dict[str, Any]) -> list[int]:
        with _baseline_environment_guard():
            action = candidate(dict(observation))
        if not isinstance(action, Sequence) or isinstance(action, (str, bytes)):
            raise R198ProductionFactoryError("snapshot official baseline returned invalid action")
        return [int(value) for value in action]

    return isolated, deck, module_name


def _pairing_artifacts(manifest: Mapping[str, Any], source_root: Path) -> tuple[Any, dict[str, Any]]:
    """Rehash the exact v2 capability chain for a new engine instance."""

    from .engine_rebuild.rtp_pairing_snapshot import (
        PairingArtifactSet,
        RTPPairingSnapshotError,
        snapshot_abi_contract,
        snapshot_abi_sha256,
        verify_build_receipt,
    )

    capability_ref = _mapping(manifest.get("pairing_capability"), "pairing capability")
    receipt = _identity(capability_ref.get("receipt"), "pairing capability receipt")
    capability = _read_json(Path(receipt["path"]), "pairing capability receipt")
    if capability.get("schema") != PAIRING_CAPABILITY_SCHEMA or capability.get("status") != "available":
        raise R198ProductionFactoryError("pairing capability is not an available v2 receipt")
    if capability.get("true_rng_pairing_available") is not True or capability.get("supported_rng_kinds") != ["snapshot"]:
        raise R198ProductionFactoryError("pairing capability does not authorize snapshot-only true RNG")
    abi = {**snapshot_abi_contract(), "canonical_abi_sha256": snapshot_abi_sha256()}
    if capability.get("abi") != abi:
        raise R198ProductionFactoryError("pairing capability ABI differs from v2 contract")
    # Private build artifacts must be supplied by the sealed capability.  They
    # are allowed under the separately sealed private capability root, not a
    # mutable live library.  The exact build receipt rebinds all four bytes.
    raw_set = PairingArtifactSet(
        engine_artifact=_identity(
            capability.get("engine_artifact"), "pairing engine artifact", exact_mode=0o555
        ),
        source_artifact=_identity(capability.get("source_artifact"), "pairing source artifact"),
        patch_artifact=_identity(capability.get("patch_artifact"), "pairing patch artifact"),
        build_artifact=_identity(capability.get("build_artifact"), "pairing build receipt"),
    )
    try:
        checked = verify_build_receipt(raw_set)
    except (RTPPairingSnapshotError, TypeError, ValueError) as exc:
        raise R198ProductionFactoryError("pairing build artifact set is invalid") from exc
    # The staging compiler may include redundant bindings.  They must agree if
    # present; this prevents a factory config from quietly choosing another DSO.
    factory_binding = _mapping(
        _factory_spec(manifest).get("pairing_artifacts", {
            "engine_artifact": checked.engine_artifact,
            "source_artifact": checked.source_artifact,
            "patch_artifact": checked.patch_artifact,
            "build_artifact": checked.build_artifact,
        }),
        "production_factory.pairing_artifacts",
    )
    for name, expected in (
        ("engine_artifact", checked.engine_artifact),
        ("source_artifact", checked.source_artifact),
        ("patch_artifact", checked.patch_artifact),
        ("build_artifact", checked.build_artifact),
    ):
        supplied = _identity(
            factory_binding.get(name),
            f"factory pairing {name}",
            exact_mode=0o555 if name == "engine_artifact" else 0o444,
        )
        _identity_equal(supplied, expected, f"factory/capability pairing {name}")
    return checked, {"receipt": receipt, "payload": capability, "abi": abi}


def _crosscheck_cg_against_pairing(cg: _CGAssets, artifacts: Any, capability: Mapping[str, Any]) -> None:
    """Bind the snapshot-local ``cg/libcg.so`` to the v2 private build set.

    The closure JSON is provenance only until this comparison succeeds.  In
    particular, it is not enough for the curated Python package to be
    read-only: its physical DSO must be the very engine supplied by the sealed
    pairing capability and the accompanying source/patch/build identities.
    """

    payload = cg.closure_payload
    engine = getattr(artifacts, "engine_artifact", None)
    source = getattr(artifacts, "source_artifact", None)
    patch = getattr(artifacts, "patch_artifact", None)
    build = getattr(artifacts, "build_artifact", None)
    if not all(isinstance(item, Mapping) for item in (engine, source, patch, build)):
        raise R198ProductionFactoryError("pairing artifact set is incomplete for CG closure")
    if any(cg.library.get(key) != engine.get(key) for key in ("sha256", "bytes")):
        raise R198ProductionFactoryError("snapshot-local CG DSO differs from pairing engine")
    closure_engine = cg.closure_evidence["engine_artifact"]
    if any(closure_engine.get(key) != engine.get(key) for key in ("sha256", "bytes")):
        raise R198ProductionFactoryError("CG closure engine differs from pairing engine")
    closure_build = cg.closure_evidence["pairing_build_artifact"]
    _identity_equal(closure_build, build, "CG closure/pairing build artifact")
    expected = {
        "pairing_engine_artifact_sha256": engine["sha256"],
        "pairing_source_artifact_sha256": source["sha256"],
        "pairing_patch_artifact_sha256": patch["sha256"],
        "canonical_abi_sha256": capability["abi"]["canonical_abi_sha256"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise R198ProductionFactoryError(f"CG closure differs from pairing capability at {key}")


def _require_shared_cg_dso_handle(engine: Any, cg: _CGAssets) -> None:
    """Prove the wrapper and ``cg.sim`` actually share one loaded DSO.

    Digest equality alone is insufficient on ELF platforms: two equal files at
    distinct paths may still receive independent loader handles and global
    tables.  The worker has already pinned ``CG_LIB_PATH`` to this closure;
    import it explicitly and compare the concrete dynamic-loader handles.
    """

    try:
        from . import cg_env

        runtime = cg_env.ensure_cg_importable()
        if _physical_path(runtime, "loaded CG runtime", directory=True) != cg.runtime_root:
            raise R198ProductionFactoryError("loaded CG runtime escaped the snapshot-local closure")
        from cg import sim

        sim_path = _physical_path(Path(sim.__file__), "loaded cg.sim", file=True)
        if sim_path.parent != cg.runtime_root / "cg":
            raise R198ProductionFactoryError("loaded cg.sim is not from the snapshot-local closure")
        sim_library = getattr(sim, "lib", None)
        wrapper_library = getattr(engine, "_library", None)
        sim_handle = getattr(sim_library, "_handle", None)
        wrapper_handle = getattr(wrapper_library, "_handle", None)
        if not isinstance(sim_handle, int) or not isinstance(wrapper_handle, int):
            raise R198ProductionFactoryError("CG closure does not expose native loader handles")
        if sim_handle != wrapper_handle:
            raise R198ProductionFactoryError("snapshot wrapper and cg.sim loaded different DSO handles")
    except R198ProductionFactoryError:
        raise
    except Exception as exc:
        raise R198ProductionFactoryError("could not verify the snapshot-local shared CG DSO handle") from exc


def _require_snapshot_matchup_adapter_registry(inputs: _FactoryInputs) -> str:
    """Load and bind the exact registry that ``PolicyAgent`` will resolve.

    The V6 router currently obtains its registry through
    ``load_slot_registry()``'s module-relative default.  Prove that both the
    imported module and that bound default live in this source snapshot, then
    load the already-verified identity explicitly.  The post-construction
    check below confirms the router retained the same canonical digest.
    """

    try:
        from . import matchup_adapters_v6

        module_path = _physical_path(
            matchup_adapters_v6.__file__,
            "loaded matchup_adapters_v6 module",
            file=True,
        )
        expected_module = inputs.candidate.source_root / "poke_bot" / "matchup_adapters_v6.py"
        if module_path != expected_module:
            raise R198ProductionFactoryError(
                "loaded matchup_adapters_v6 module escaped the source snapshot"
            )
        defaults = getattr(matchup_adapters_v6.load_slot_registry, "__defaults__", None)
        if not isinstance(defaults, tuple) or len(defaults) != 1:
            raise R198ProductionFactoryError(
                "matchup adapter registry loader has no single pinned default"
            )
        default_path = _physical_path(
            defaults[0],
            "matchup adapter registry loader default",
            file=True,
        )
        expected_registry = Path(inputs.matchup_adapter_registry["path"])
        if (
            default_path != expected_registry
            or _physical_path(
                matchup_adapters_v6.DEFAULT_REGISTRY_PATH,
                "matchup adapter registry module default",
                file=True,
            )
            != expected_registry
        ):
            raise R198ProductionFactoryError(
                "PolicyAgent matchup adapter registry default is not snapshot-local"
            )
        rebound = _identity(
            inputs.matchup_adapter_registry,
            "runtime snapshot-local matchup adapter registry",
            root=inputs.candidate.source_root,
            exact_mode=0o444,
        )
        loaded = matchup_adapters_v6.load_slot_registry(rebound["path"])
        digest = str(matchup_adapters_v6.registry_digest(loaded))
        if digest != inputs.matchup_adapter_registry_digest:
            raise R198ProductionFactoryError(
                "runtime matchup adapter registry canonical digest changed"
            )
        return digest
    except R198ProductionFactoryError:
        raise
    except Exception as exc:
        raise R198ProductionFactoryError(
            "could not bind the snapshot-local matchup adapter registry"
        ) from exc


def _require_policy_router_registry(candidate: Any, expected_digest: str) -> None:
    router = getattr(candidate, "_matchup_adapter_shadow_router", None)
    tree = getattr(router, "tree", None)
    if tree is None or getattr(tree, "slot_registry_digest", None) != expected_digest:
        raise R198ProductionFactoryError(
            "PolicyAgent public router did not retain the sealed matchup adapter registry"
        )


def _require_model_matchup_adapter_registry(model: Any, inputs: _FactoryInputs) -> None:
    """Cross-bind the checkpoint's physical V6 slots to the router roster."""

    bank = getattr(model, "matchup_adapter_bank", None)
    config_dict = getattr(bank, "config_dict", None)
    if not callable(config_dict):
        raise R198ProductionFactoryError(
            "r195 parent model has no matchup adapter bank contract"
        )
    config = _mapping(config_dict(), "r195 parent matchup adapter bank contract")
    if config.get("format") != "poke-bot-matchup-adapter-bank-v6":
        raise R198ProductionFactoryError(
            "r195 parent matchup adapter bank is not the exact V6 format"
        )
    expected_digest = inputs.matchup_adapter_registry_digest
    if config.get("slot_registry_digest") != expected_digest:
        raise R198ProductionFactoryError(
            "r195 parent matchup adapter bank registry digest differs from the snapshot roster"
        )
    serialized = _mapping(
        config.get("slot_registry"),
        "r195 parent matchup adapter bank slot_registry",
    )
    if _canonical_digest(serialized) != expected_digest:
        raise R198ProductionFactoryError(
            "r195 parent matchup adapter bank serialized registry differs from the snapshot roster"
        )


def _load_parent_model(candidate: _CandidateAssets) -> Any:
    """Load one frozen parent model in the isolated arm child only."""

    try:
        import torch

        if not torch.cuda.is_available():
            raise R198ProductionFactoryError("r198 production evaluation requires the pinned CUDA worker")
        device = torch.device("cuda:0")
        from .train import load_model_from_checkpoint

        model = load_model_from_checkpoint(candidate.parent_checkpoint["path"], device=device)
        model.eval()
        model.requires_grad_(False)
        observed = _sha256_file(Path(candidate.parent_checkpoint["path"]))
        if observed != R195_PARENT_SHA256:
            raise R198ProductionFactoryError("parent checkpoint changed during model load")
        if int(getattr(model, "d_model", 0)) != 96:
            raise R198ProductionFactoryError("r195 parent model width is not 96")
        return model
    except R198ProductionFactoryError:
        raise
    except Exception as exc:  # pragma: no cover - real runtime/torch error path.
        raise R198ProductionFactoryError("strict r195 parent model load failed") from exc


def _probe_value_fingerprint(value: Any, *, _seen: set[int] | None = None, _depth: int = 0) -> Any:
    """Make a compact, non-mutating structural fingerprint for probe guards."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if _depth > 8:
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}
    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return {"ref_type": f"{type(value).__module__}.{type(value).__qualname__}", "id": identity}
    if isinstance(value, bytes):
        return {"bytes": _sha256_bytes(value), "length": len(value)}
    try:
        import torch

        if isinstance(value, torch.Tensor):
            detached = value.detach()
            raw = detached.cpu().contiguous().numpy().tobytes()
            return {
                "tensor": _sha256_bytes(raw),
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "device": str(detached.device),
            }
    except (AttributeError, ImportError, RuntimeError, TypeError):
        # A lightweight test environment may intentionally lack torch, while a
        # non-tensor value naturally has no tensor-specific byte view.
        pass
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            return {
                str(key): _probe_value_fingerprint(item, _seen=seen, _depth=_depth + 1)
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, (list, tuple)):
            return [
                _probe_value_fingerprint(item, _seen=seen, _depth=_depth + 1)
                for item in value
            ]
        if isinstance(value, set):
            return sorted(
                (_probe_value_fingerprint(item, _seen=seen, _depth=_depth + 1) for item in value),
                key=repr,
            )
        state = getattr(value, "__dict__", None)
        if isinstance(state, Mapping):
            return {
                "object_type": f"{type(value).__module__}.{type(value).__qualname__}",
                "object_id": identity,
                "state": {
                    str(key): _probe_value_fingerprint(item, _seen=seen, _depth=_depth + 1)
                    for key, item in sorted(state.items(), key=lambda item: str(item[0]))
                    if not callable(item)
                },
            }
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}
    finally:
        seen.discard(identity)


def _rng_fingerprint() -> Mapping[str, Any]:
    state: dict[str, Any] = {
        "python": _sha256_bytes(repr(random.getstate()).encode("utf-8")),
    }
    try:
        import torch

        state["torch_cpu"] = _sha256_bytes(bytes(torch.get_rng_state().cpu().tolist()))
        if torch.cuda.is_available():
            state["torch_cuda"] = [
                _sha256_bytes(bytes(item.cpu().tolist()))
                for item in torch.cuda.get_rng_state_all()
            ]
    except (AttributeError, ImportError, RuntimeError, TypeError):
        state["torch"] = "unavailable"
    return state


def _candidate_probe_state_fingerprint(candidate: Any) -> str:
    """Fingerprint every candidate field a complexity probe must not alter."""

    bridge = getattr(candidate, "_rtp_bridge", None)
    executor = getattr(bridge, "executor", None) if bridge is not None else None
    diagnostics = getattr(bridge, "last_diagnostics", None) if bridge is not None else None
    diagnostic_payload = diagnostics.as_dict() if callable(getattr(diagnostics, "as_dict", None)) else diagnostics
    router = getattr(candidate, "_matchup_adapter_shadow_router", None)
    router_snapshot = (
        candidate.matchup_adapter_shadow_snapshot()
        if callable(getattr(candidate, "matchup_adapter_shadow_snapshot", None))
        else _probe_value_fingerprint(router)
    )
    candidate_rng = getattr(candidate, "rng", None)
    candidate_rng_state = (
        _probe_value_fingerprint(candidate_rng.getstate())
        if callable(getattr(candidate_rng, "getstate", None))
        else _probe_value_fingerprint(candidate_rng)
    )
    payload = {
        "board_history": _probe_value_fingerprint(getattr(candidate, "board_history", None)),
        "previous_action_history": _probe_value_fingerprint(
            getattr(candidate, "previous_action_history", None)
        ),
        "previous_action_token": _probe_value_fingerprint(
            getattr(candidate, "_previous_action_token", None)
        ),
        "kv_cache_identity": id(getattr(candidate, "_kv_cache", None)),
        "kv_cache": _probe_value_fingerprint(getattr(candidate, "_kv_cache", None)),
        "router": _probe_value_fingerprint(router_snapshot),
        "bridge": {
            "identity": id(bridge),
            "memory": _probe_value_fingerprint(getattr(bridge, "memory", None)),
            "active_turn_key": _probe_value_fingerprint(
                getattr(bridge, "active_turn_key", None)
            ),
            "active_turn_complexity_intent": _probe_value_fingerprint(
                getattr(bridge, "active_turn_complexity_intent", None)
            ),
            "last_diagnostics": _probe_value_fingerprint(diagnostic_payload),
            "executor": {
                "identity": id(executor),
                "active_program": _probe_value_fingerprint(
                    getattr(executor, "active_program", None)
                ),
                "cursor": _probe_value_fingerprint(getattr(executor, "cursor", None)),
                "repairs_used": getattr(executor, "repairs_used", None),
                "steps_executed": getattr(executor, "steps_executed", None),
                "repair_fn_identity": id(getattr(executor, "repair_fn", None)),
            },
        },
        "candidate_last_rtp_diagnostics": _probe_value_fingerprint(
            getattr(candidate, "last_rtp_diagnostics", None)
        ),
        "candidate_rng": candidate_rng_state,
        "global_rng": _rng_fingerprint(),
    }
    return _canonical_digest(payload)


def _logical_policy_input_value(value: Any, *, label: str) -> Any:
    """Canonicalize a candidate policy input without object identities.

    This deliberately differs from ``_candidate_probe_state_fingerprint``:
    that broader mutation sentinel contains bridge/executor object identities
    and diagnostics, which are useful for detecting an illicit probe mutation
    but would make equal factorized-policy inputs look different across arms.
    The r198 over-cap parity key needs only causal input to the policy itself.
    """

    from .features import SparseVector

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise R198ProductionFactoryError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, SparseVector):
        return {
            "kind": "SparseVector",
            "index": [int(item) for item in value.index],
            "value": [float(item) for item in value.value],
            "offset": [int(item) for item in value.offset],
            "pos": int(value.pos),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise R198ProductionFactoryError(f"{label} mapping has a non-string key")
        return {
            key: _logical_policy_input_value(item, label=f"{label}.{key}")
            for key, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _logical_policy_input_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous().cpu()
            try:
                raw = tensor.numpy().tobytes()
            except RuntimeError as exc:
                raise R198ProductionFactoryError(
                    f"{label} tensor cannot be serialized canonically"
                ) from exc
            return {
                "kind": "torch.Tensor",
                "dtype": str(tensor.dtype),
                "shape": [int(item) for item in tensor.shape],
                "sha256": _sha256_bytes(raw),
            }
    except ImportError:  # pragma: no cover - factory needs torch in production
        pass
    # The only remaining expected object is the temporal cache.  Bind its
    # actual causal tensor contents and counters, never its object address.
    if all(hasattr(value, field) for field in ("layers", "length", "next_position", "input_tokens")):
        return {
            "kind": "temporal_kv_cache",
            "layers": [
                [
                    _logical_policy_input_value(key, label=f"{label}.layers[{index}].key"),
                    _logical_policy_input_value(value_tensor, label=f"{label}.layers[{index}].value"),
                ]
                for index, (key, value_tensor) in enumerate(value.layers)
            ],
            "length": int(value.length),
            "next_position": (
                None if value.next_position is None else int(value.next_position)
            ),
            "input_tokens": _logical_policy_input_value(
                value.input_tokens, label=f"{label}.input_tokens"
            ),
        }
    raise R198ProductionFactoryError(
        f"{label} cannot be represented in a logical policy-input fingerprint"
    )


def _candidate_logical_policy_input_fingerprint(
    candidate: Any, observation: Mapping[str, Any]
) -> str:
    """Return a cross-arm comparable digest of actual factorized inputs.

    The caller invokes this immediately before ``PolicyAgent.__call__``.  The
    agent itself first advances its causal public router and appends the board
    history, so mirror those pure operations on a fork here.  In particular,
    do not use the pre-router snapshot or bridge/executor state: neither is
    the factorized-greedy policy input after an over-cap fallback.
    """

    from . import features

    router = getattr(candidate, "_matchup_adapter_shadow_router", None)
    if router is None or not callable(getattr(router, "fork", None)):
        raise R198ProductionFactoryError(
            "candidate lacks a forkable causal matchup router for over-cap fingerprinting"
        )
    history_before = list(getattr(candidate, "board_history", ()))
    logical_router = router.fork()
    # Shadow-router forks intentionally share their audit sink.  Calling the
    # public ``observe`` method on one would therefore mutate production audit
    # state merely to derive a parity key.  Its game_router fork is independent
    # and has the same recognizer transition that determines the model route.
    game_router = getattr(logical_router, "game_router", None)
    if game_router is not None and callable(getattr(game_router, "observe", None)):
        game_router.observe(dict(observation))
    else:
        observe = getattr(logical_router, "observe", None)
        if not callable(observe):
            raise R198ProductionFactoryError(
                "candidate matchup router lacks a logical over-cap fingerprint surface"
            )
        observe(dict(observation), scope="game_root", depth=len(history_before))
    post_router_route = getattr(logical_router, "candidate_model_route", None)
    if type(post_router_route) is not int:
        raise R198ProductionFactoryError(
            "candidate router has no exact post-observe model route"
        )
    tree = getattr(logical_router, "tree", None)
    tree_digest = getattr(tree, "digest", None)
    if tree_digest is not None and not isinstance(tree_digest, str):
        raise R198ProductionFactoryError("candidate router tree digest is invalid")
    try:
        board = features.build_board_tokens(dict(observation), list(candidate.deck))
    except Exception as exc:
        raise R198ProductionFactoryError(
            "candidate policy observation cannot build a factorized board input"
        ) from exc
    boards = [*history_before, board]
    previous_actions = [
        *list(getattr(candidate, "previous_action_history", ())),
        getattr(candidate, "_previous_action_token", None),
    ]
    history_limit = getattr(candidate, "_history_context_limit", None)
    if not callable(history_limit):
        raise R198ProductionFactoryError(
            "candidate lacks a history-context limit for over-cap fingerprinting"
        )
    limit = int(history_limit())
    if limit <= 0:
        raise R198ProductionFactoryError("candidate history-context limit is invalid")
    boards = boards[-limit:]
    previous_actions = previous_actions[-limit:]
    sampling = bool(getattr(candidate, "sample_actions", False))
    rng = getattr(candidate, "rng", None)
    rng_state = (
        rng.getstate()
        if sampling and callable(getattr(rng, "getstate", None))
        else None
    )
    model = getattr(candidate, "model", None)
    return _canonical_digest(
        {
            "observation": _logical_policy_input_value(
                dict(observation), label="candidate policy observation"
            ),
            "board_history": _logical_policy_input_value(
                boards,
                label="candidate board_history",
            ),
            "previous_action_history": _logical_policy_input_value(
                previous_actions,
                label="candidate previous_action_history",
            ),
            "previous_action_token": _logical_policy_input_value(
                getattr(candidate, "_previous_action_token", None),
                label="candidate previous_action_token",
            ),
            "kv_cache": _logical_policy_input_value(
                getattr(candidate, "_kv_cache", None), label="candidate kv_cache"
            ),
            "matchup_router": {
                "router_type": type(router).__qualname__,
                "tree_digest": tree_digest,
                "post_observe_candidate_model_route": post_router_route,
            },
            "candidate_rng_state": _logical_policy_input_value(
                rng_state, label="candidate rng_state"
            ),
            "sample_actions": sampling,
            "action_temperature": float(getattr(candidate, "action_temperature", 1.0)),
            "deck": [int(card) for card in candidate.deck],
            "model": {
                "type": None if model is None else type(model).__qualname__,
                "d_model": None if model is None else int(getattr(model, "d_model", 0)),
                "max_context": None
                if model is None
                else int(getattr(model, "max_context", 0)),
                "checkpoint_digest": str(getattr(candidate, "checkpoint_digest", "")),
            },
        }
    )


def _require_unchanged_complexity_probe_state(candidate: Any, before: str) -> str:
    """Fail closed when instrumentation changes candidate-causal state.

    The runner independently calls the same digest before and after the probe.
    Keeping the check here too makes the factory safe when the probe is used by
    a focused preflight or a future runner implementation.  The digest includes
    history, cache, router, bridge/executor, diagnostics, and RNG state rather
    than merely the selected-action result.
    """

    after = _candidate_probe_state_fingerprint(candidate)
    if after != before:
        raise R198ProductionFactoryError(
            "complexity probe mutated candidate causal state"
        )
    return after


def _validated_evaluation_action_execution(
    raw: Mapping[str, Any] | None,
    *,
    manifest: Mapping[str, Any],
    cell: Mapping[str, Any],
    arm: str,
    candidate: _CandidateAssets,
) -> Mapping[str, Any] | None:
    """Accept the runner's narrow, process-bound B/C action exception only.

    This is not an ambient action-authority flag.  The action-authority module
    rehashes the manifest, runner authority, candidate snapshot, and per-arm
    fence, then also checks the current process and injected fence environment.
    The returned value is its deep-copied immutable view and is passed only to
    the B/C ``PolicyAgent``/``RTPAgentBridge`` construction path.
    """

    if arm == "no_rtp":
        if raw is not None:
            raise R198ProductionFactoryError(
                "no-RTP arm must not receive evaluation action execution context"
            )
        return None
    if raw is None:
        raise R198ProductionFactoryError(
            "RTP evaluation arm requires an explicit evaluation action execution context"
        )
    try:
        from .recursive_turn_planner.profiles import get_profile
        from .recursive_turn_planner.r197_action_authority import (
            validate_evaluation_action_execution,
        )

        validated = validate_evaluation_action_execution(
            raw,
            config=get_profile(R198_PROFILE).to_config(),
            max_action_combos=R198_MAX_ACTION_COMBOS,
            expected_parent_digest=R195_PARENT_SHA256,
            checkpoint_path=str(candidate.sidecar["path"]),
        )
    except Exception as exc:
        raise R198ProductionFactoryError(
            "RTP evaluation action execution context is invalid"
        ) from exc
    if validated.get("arm") != arm:
        raise R198ProductionFactoryError("evaluation action execution arm differs")
    for key in ("cell_id", "evaluation_case_id", "opponent_id", "candidate_seat"):
        if validated.get(key) != cell.get(key):
            raise R198ProductionFactoryError(
                f"evaluation action execution differs from requested cell at {key}"
            )
    runtime_contract = _identity(
        validated.get("runtime_contract"),
        "evaluation action runtime contract",
        root=candidate.source_root,
    )
    _identity_equal(
        runtime_contract,
        candidate.candidate_manifest,
        "evaluation action runtime contract/candidate snapshot",
    )
    manifest_identity = _identity(
        validated.get("manifest"), "evaluation action manifest"
    )
    context_manifest = _read_json(
        Path(manifest_identity["path"]), "evaluation action manifest"
    )
    if _canonical_digest(context_manifest) != _canonical_digest(manifest):
        raise R198ProductionFactoryError(
            "evaluation action execution references another evaluator manifest"
        )
    return validated


class _ComplexityIntentProbe:
    """A non-selection, non-mutating common r198 complexity predicate.

    The candidate arms must not be used to obtain the probe: calling a bridge
    to ask whether it would recurse would update its pass counter, and calling
    a policy agent would append history.  This object owns a separate strict
    sidecar planner and forks the public matchup router for each observation.
    It uses the same encoder, legal complete-action enumeration, sidecar, and
    input history as the next bridge decision, but its result cannot alter the
    candidate's board history, cached keys/values, planner, executor, or RNG.
    """

    def __init__(self, *, candidate: Any, sidecar_path: str) -> None:
        from .recursive_turn_planner.agent_bridge import RTPAgentBridge
        from .recursive_turn_planner.profiles import get_profile
        from .recursive_turn_planner.training.checkpoint import load_rtp_checkpoint

        self._candidate = candidate
        self._planner = load_rtp_checkpoint(
            sidecar_path,
            device=next(candidate.model.parameters()).device,
            expected_parent_digest=R195_PARENT_SHA256,
        )
        self._planner.eval()
        self._current_boards: list[Any] = []
        self._current_actions: list[Any] = []
        self._current_router: Any = None

        def route() -> int:
            if self._current_router is None:
                return -1
            return int(self._current_router.candidate_model_route)

        self._bridge = RTPAgentBridge(
            model=candidate.model,
            deck=list(candidate.deck),
            config=get_profile(R198_PROFILE).to_config(),
            get_matchup_route=route,
            get_board_history=lambda: list(self._current_boards),
            get_previous_action_history=lambda: list(self._current_actions),
            get_previous_action_token=lambda: candidate._previous_action_token,
            get_kv_cache=lambda: candidate._kv_cache,
            # ``append_cache=False`` is mandatory below; fail if a future code
            # change tries to make this instrumentation mutate candidate state.
            set_kv_cache=lambda _cache: (_ for _ in ()).throw(
                R198ProductionFactoryError("complexity probe attempted to mutate candidate KV cache")
            ),
            max_action_combos=R198_MAX_ACTION_COMBOS,
            planner=self._planner,
        )

    def _is_continuation(self, observation: Mapping[str, Any]) -> bool:
        from .recursive_turn_planner.agent_bridge import turn_key_from_obs

        candidate = self._candidate
        bridge = getattr(candidate, "_rtp_bridge", None)
        if bridge is None:
            # no-RTP has no plan/executor; its counterfactual direct bridge
            # evaluates the new-selection gate on every select.
            return False
        if bool(getattr(candidate, "force_direct_bridge_only", False)):
            # Direct bridge deliberately clears plans after each action.
            return False
        executor = getattr(bridge, "executor", None)
        active = getattr(executor, "active_program", None)
        return bool(
            active is not None
            and getattr(bridge, "memory", None) is not None
            and turn_key_from_obs(dict(observation)) == getattr(bridge, "active_turn_key", (-1, -1))
        )

    def __call__(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        before = _candidate_probe_state_fingerprint(self._candidate)
        try:
            # Turn-order selection is an external, fixed Yes control.  The
            # isolated evaluator excludes it from candidate-policy latency and
            # planner denominators before it calls this probe; keep the same
            # defensive short-circuit here so a future caller cannot classify
            # that control as a neural-complex decision.
            from .features import forced_go_first_action

            if forced_go_first_action(dict(observation)) is not None:
                return {
                    "intended_complex": False,
                    "planner_reason": "forced_go_first_contract",
                }
            from . import features

            action_space = features.complete_ordered_action_space_summary(
                dict(observation), max_combos=R198_MAX_ACTION_COMBOS
            )
            if action_space["over_cap"] is True:
                # The runner owns the only valid over-cap path.  Returning a
                # harmless-looking direct intent here would wrongly make the
                # decision planner-eligible if a future caller invokes this
                # probe out of order.
                raise R198ProductionFactoryError(
                    "runner must preclassify over-cap factorized selection before complexity probing"
                )
            if self._is_continuation(observation):
                return {
                    "intended_complex": False,
                    "planner_reason": "not_new_turn_complexity_gate",
                }

            # Mirror PolicyAgent.__call__'s causal routing without altering its
            # own router/audit.  The cloned router starts from the exact same
            # pre-decision public history and receives this observation once.
            router = self._candidate._matchup_adapter_shadow_router.fork()
            router.observe(
                dict(observation),
                scope="game_root",
                depth=len(self._candidate.board_history),
            )
            board = features.build_board_tokens(dict(observation), self._candidate.deck)
            self._current_router = router
            self._current_boards = [*self._candidate.board_history, board]
            self._current_actions = [
                *self._candidate.previous_action_history,
                self._candidate._previous_action_token,
            ]
            legal = self._bridge._legal_actions(dict(observation))
            if not legal:
                return {"intended_complex": False, "planner_reason": "no_legal_actions"}
            self._planner.reset_pass_counter()
            memory, logits = self._bridge.encode(
                dict(observation), board=board, legal_actions=legal, append_cache=False
            )
            recurse, details = self._planner.should_recurse(memory, policy_logits=logits)
            # The trace itself consumes one pass in its isolated planner.  It
            # is instrumentation, not an action/latency or candidate budget.
            self._planner.reset_pass_counter()
            if not isinstance(details, Mapping):
                raise R198ProductionFactoryError("complexity probe returned invalid planner details")
            if recurse:
                reasons = [
                    name
                    for name in ("by_options", "by_entropy", "by_head")
                    if details.get(name) is True
                ]
                reason = "complexity_gate:" + (",".join(reasons) or "recursive")
            else:
                reason = str(details.get("reason") or "complexity_gate_direct_policy")
            return {"intended_complex": bool(recurse), "planner_reason": reason}
        except R198ProductionFactoryError:
            raise
        except Exception as exc:
            # Do not invent a simple decision.  An instrumentation failure is
            # an evaluation failure, which is safer than changing gate counts.
            raise R198ProductionFactoryError("common complexity intent probe failed") from exc
        finally:
            self._current_boards = []
            self._current_actions = []
            self._current_router = None
            _require_unchanged_complexity_probe_state(self._candidate, before)


class ProductionR198EvaluationFactory:
    """Concrete sealed factory used by the r198 private three-arm runner."""

    def worker_environment(
        self,
        *,
        manifest: Mapping[str, Any],
        cell: Mapping[str, Any],
        arm: str,
        scratch_dir: str,
    ) -> Mapping[str, str]:
        """Return only sealed snapshot bindings; do not load code or models."""

        del cell, scratch_dir  # Parent construction is deliberately side-effect free.
        inputs = _factory_inputs(manifest, validate_sidecar_payload=False)
        arm_spec = _arm_spec(manifest, arm, inputs)
        # Validate the private build chain in the parent too.  This is a pure
        # rehash/read; no engine is loaded or initialized here.
        artifacts, pairing = _pairing_artifacts(manifest, inputs.candidate.source_root)
        _crosscheck_cg_against_pairing(inputs.cg, artifacts, pairing)
        sidecar = arm_spec["rtp_sidecar"]
        rtp_enabled = sidecar is not None
        return {
            # `CG_LIB_PATH` is the parent containing cg/, not libcg.so itself;
            # `paths.cg_runtime_dir()` uses this exact shape and otherwise can
            # fall back to a mutable installed competition runtime.
            "CG_LIB_PATH": str(inputs.cg.runtime_root),
            "POKEBOT_R198_EVAL_SOURCE_SNAPSHOT_ROOT": str(inputs.candidate.source_root),
            "POKEBOT_R198_EVAL_SOURCE_TREE_SHA256": inputs.candidate.source_tree_sha256,
            "POKEBOT_R198_EVAL_RUNTIME_CONTRACT": str(
                inputs.candidate.candidate_manifest["path"]
            ),
            "POKEBOT_R198_EVAL_RUNTIME_CONTRACT_SHA256": inputs.candidate.candidate_manifest[
                "sha256"
            ],
            "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "1" if rtp_enabled else "0",
            "POKEBOT_RTP_CHECKPOINT": str(sidecar["path"]) if sidecar else "",
            "POKEBOT_RTP_SIZING_PROFILE": R198_PROFILE if rtp_enabled else "",
            "POKEBOT_RTP_MAX_ACTION_COMBOS": str(R198_MAX_ACTION_COMBOS) if rtp_enabled else "",
            "POKEBOT_RTP_SERVING_QUALIFIED": "0",
            # Expected parent retains strict sidecar parent validation while
            # deliberately not supplying any promotion receipt/authority.
            "POKEBOT_RTP_PARENT_CHECKPOINT_SHA256": R195_PARENT_SHA256 if rtp_enabled else "",
            "POKEBOT_RTP_PROMOTION_RECEIPT": "",
            "POKEBOT_RTP_PROMOTION_RECEIPT_SHA256": "",
            "POKEBOT_RTP_FORCE_DIRECT_BRIDGE_ONLY": "1" if arm == "direct_bridge_recursive_disabled" else "0",
        }

    def create_arm_engine(
        self, *, manifest: Mapping[str, Any], cell: Mapping[str, Any], arm: str
    ) -> Any:
        """Instantiate/bind a fresh engine; restoration remains runner-owned."""

        del cell
        inputs = _factory_inputs(manifest)
        _arm_spec(manifest, arm, inputs)
        artifacts, capability = _pairing_artifacts(manifest, inputs.candidate.source_root)
        _crosscheck_cg_against_pairing(inputs.cg, artifacts, capability)
        from .engine_rebuild.rtp_pairing_snapshot import RtpPairingSnapshotEngine

        # The pairing receipt names a private immutable engine artifact, while
        # the process imports the snapshot-local curated ``cg`` closure.  Use
        # the latter path so ``cg.sim`` and this wrapper share one dlopen
        # handle/global table; ``require_bound_artifacts`` keeps the private
        # receipt identity authoritative.
        engine = RtpPairingSnapshotEngine(inputs.cg.library["path"])
        engine.require_bound_artifacts(artifacts)
        _require_shared_cg_dso_handle(engine, inputs.cg)
        # Do not call capture/start/restore here.  The runner's next engine
        # mutation is exactly `restore_sealed_snapshot_manifest(seal_path)`.
        return engine

    def create_arm_runtime(
        self,
        *,
        manifest: Mapping[str, Any],
        cell: Mapping[str, Any],
        arm: str,
        evaluation_action_execution: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Load a fresh candidate and fresh snapshot-local official baseline."""

        inputs = _factory_inputs(manifest)
        pairing_artifacts, pairing_capability = _pairing_artifacts(
            manifest, inputs.candidate.source_root
        )
        _crosscheck_cg_against_pairing(inputs.cg, pairing_artifacts, pairing_capability)
        matchup_adapter_registry_digest = _require_snapshot_matchup_adapter_registry(
            inputs
        )
        candidate_assets = inputs.candidate
        arm_spec = _arm_spec(manifest, arm, inputs)
        action_execution = _validated_evaluation_action_execution(
            evaluation_action_execution,
            manifest=manifest,
            cell=cell,
            arm=arm,
            candidate=candidate_assets,
        )
        shared = _shared_assets(manifest, inputs)
        shared_runtime_sha256s = _runtime_shared_artifact_sha256s(shared)
        opponent_id = _text(cell.get("opponent_id"), "evaluation cell opponent_id")
        package = _official_package(manifest, opponent_id, candidate_assets.source_root)
        model = _load_parent_model(candidate_assets)
        _require_model_matchup_adapter_registry(model, inputs)
        candidate_deck = _validate_candidate_deck(candidate_assets.deck)

        from .agent import PolicyAgent

        rtp_enabled = arm != "no_rtp"
        direct_only = arm == "direct_bridge_recursive_disabled"
        policy_kwargs: dict[str, Any] = {
            "model": model,
            "deck": list(candidate_deck),
            "use_recursive_turn_planner": rtp_enabled,
            "rtp_sizing_profile": R198_PROFILE if rtp_enabled else None,
            "rtp_max_action_combos": R198_MAX_ACTION_COMBOS if rtp_enabled else None,
            "force_direct_bridge_only": direct_only,
            "strict_runtime": True,
            "collect_targets": False,
            "sample_actions": False,
            "use_mcts": False,
            "leaf_backend": None,
            "checkpoint_digest": R195_PARENT_SHA256,
            "matchup_adapter_runtime": True,
            "matchup_adapter_tree_path": str(candidate_assets.matchup_tree["path"]),
        }
        # B/C receive a concrete, fully revalidated mapping; A remains
        # bridge-free and intentionally sees no evaluator action exception.
        if rtp_enabled:
            assert action_execution is not None
            policy_kwargs["rtp_evaluation_action_execution"] = action_execution
        candidate = PolicyAgent(
            **policy_kwargs,
        )
        _require_policy_router_registry(
            candidate, matchup_adapter_registry_digest
        )
        if rtp_enabled and getattr(candidate, "_rtp_bridge", None) is None:
            raise R198ProductionFactoryError("RTP arm did not attach the inert r197 sidecar")
        if direct_only:
            bridge = candidate._rtp_bridge
            if bridge is None:
                raise R198ProductionFactoryError("direct bridge arm lacks RTP bridge")
            # The comparison arm must never retain/programmatically repair a
            # recursive plan. `select(... force_direct_bridge_only=True)` also
            # clears this executor after direct selection; clear defensively at
            # construction so no inherited in-memory plan can enter a child.
            bridge.executor.clear()
            bridge.executor.repair_fn = None
            bridge.memory = None
            bridge.active_turn_key = (-1, -1)

        opponent, _opponent_deck, module_name = _load_snapshot_baseline(package)
        probe = _ComplexityIntentProbe(
            candidate=candidate, sidecar_path=str(candidate_assets.sidecar["path"])
        )
        profile = arm_spec["profile"]
        runtime_identity = {
            "arm": arm,
            "runtime_artifact_sha256": arm_spec["runtime_artifact"]["sha256"],
            "runtime_profile_sha256": arm_spec["runtime_profile"]["sha256"],
            "action_attached_rtp_sidecar_sha256": (
                None if arm_spec["rtp_sidecar"] is None else candidate_assets.sidecar["sha256"]
            ),
            "complexity_probe_sidecar_sha256": candidate_assets.sidecar["sha256"],
            "complexity_probe_sidecar_instrumentation_only": True,
            "complexity_probe_latency_excluded": True,
            "rtp_action_attachment_enabled": rtp_enabled,
            "rtp_action_authority_enabled": False,
            **shared_runtime_sha256s,
            "matchup_adapter_registry_sha256": inputs.matchup_adapter_registry[
                "sha256"
            ],
            "matchup_adapter_slot_registry_digest": matchup_adapter_registry_digest,
            "recursive_turn_planner_enabled": profile["recursive_turn_planner_enabled"],
            "direct_bridge_enabled": profile["direct_bridge_enabled"],
            "force_direct_bridge_only": profile["force_direct_bridge_only"],
            "max_neural_passes": profile.get("max_neural_passes"),
            "max_action_combos": profile.get("max_action_combos"),
            "candidate_id": R198_CANDIDATE_ID,
            "candidate_contract_sha256": R198_CANDIDATE_CONTRACT_SHA256,
            "candidate_snapshot_sha256": candidate_assets.candidate_manifest["sha256"],
            "candidate_sidecar_receipt_sha256": candidate_assets.sidecar_receipt["sha256"],
            "candidate_completion_receipt_sha256": candidate_assets.completion_receipt[
                "sha256"
            ],
            "sidecar_config_sha256": R197_SIDECAR_CONFIG_SHA256,
            "factory_module_sha256": inputs.factory_identity["sha256"],
            "evaluation_cg_library_sha256": inputs.cg.library["sha256"],
            "evaluation_cg_closure_sha256": inputs.cg.closure_manifest["sha256"],
            "evaluation_action_execution_sha256": (
                None
                if action_execution is None
                else _canonical_digest(action_execution)
            ),
            "evaluation_only": True,
            "serving_eligible": False,
            "action_authority_enabled": False,
        }
        isolation = {
            "baseline_content_digest": package.content_digest,
            "baseline_package_root": str(package.package_root),
            "baseline_tree_entries_sha256": package.payload["tree_entries_sha256"],
            "baseline_package_manifest_sha256": package.manifest["sha256"],
            "baseline_main_py_sha256": package.main_py["sha256"],
            "baseline_deck_sha256": package.deck["sha256"],
            "baseline_module_name": module_name,
            "package_snapshot_verified_before_import": True,
            "baseline_imported_from_snapshot_local_package": True,
            "candidate_snapshot_sha256": candidate_assets.candidate_manifest["sha256"],
            "candidate_parent_checkpoint_sha256": candidate_assets.parent_checkpoint["sha256"],
            "candidate_sidecar_sha256": candidate_assets.sidecar["sha256"],
            "candidate_sidecar_receipt_sha256": candidate_assets.sidecar_receipt["sha256"],
            "candidate_completion_receipt_sha256": candidate_assets.completion_receipt[
                "sha256"
            ],
            "candidate_deck_sha256": candidate_assets.deck["sha256"],
            "candidate_deck_cards_sha256": R195_DECK_CARDS_SHA256,
            "candidate_matchup_tree_sha256": candidate_assets.matchup_tree["sha256"],
            "matchup_adapter_registry_sha256": inputs.matchup_adapter_registry[
                "sha256"
            ],
            "matchup_adapter_slot_registry_digest": matchup_adapter_registry_digest,
            "factory_module_sha256": inputs.factory_identity["sha256"],
            "source_snapshot_root": str(candidate_assets.source_root),
            "source_snapshot_tree_sha256": candidate_assets.source_tree_sha256,
            "evaluation_cg_library_sha256": inputs.cg.library["sha256"],
            "evaluation_cg_closure_sha256": inputs.cg.closure_manifest["sha256"],
            **shared_runtime_sha256s,
            "action_attached_rtp_sidecar_sha256": (
                None if arm_spec["rtp_sidecar"] is None else candidate_assets.sidecar["sha256"]
            ),
            "complexity_probe_sidecar_sha256": candidate_assets.sidecar["sha256"],
            "complexity_probe_sidecar_instrumentation_only": True,
            "complexity_probe_latency_excluded": True,
            "rtp_action_attachment_enabled": rtp_enabled,
            "rtp_action_authority_enabled": False,
            "candidate_factory_calls": 1,
            "opponent_factory_calls": 1,
            "evaluation_only": True,
            "training_eligible": False,
            "replay_eligible": False,
            "serving_change_authorized": False,
            "selector_change_authorized": False,
            "action_authority_authorized": False,
            "kaggle_submission_authorized": False,
        }
        return {
            "candidate": candidate,
            "opponent": opponent,
            "runtime_identity": runtime_identity,
            "isolation": isolation,
            "complexity_intent": probe,
            "complexity_probe_state_digest": (
                lambda: _candidate_probe_state_fingerprint(candidate)
            ),
            "candidate_policy_input_fingerprint": (
                lambda observation: _candidate_logical_policy_input_fingerprint(
                    candidate, observation
                )
            ),
        }


def _preflight_inputs(raw: Mapping[str, Any]) -> tuple[dict[str, Any], _FactoryInputs, dict[str, Any]]:
    value = _mapping(raw, "planner preflight input")
    if value.get("schema") != PREFLIGHT_INPUT_SCHEMA or value.get("status") != "sealed":
        raise R198ProductionFactoryError("planner preflight input schema/status is invalid")
    # Reuse the exact production factory binding through a minimal pseudo-manifest.
    factory_spec = _mapping(value.get("production_factory"), "planner preflight production_factory")
    pairing = _mapping(value.get("pairing_capability"), "planner preflight pairing_capability")
    pseudo = {"production_factory": factory_spec, "pairing_capability": pairing}
    # Older sealed preflight inputs intentionally contain only the
    # production-factory binding; if the canonical closure pair is supplied,
    # validate it too.  The final scored evaluator always requires the pair.
    if "evaluation_cg_closure" in value:
        pseudo["evaluation_cg_closure"] = value["evaluation_cg_closure"]
    inputs = _factory_inputs(
        pseudo,
        require_evaluation_cg_closure="evaluation_cg_closure" in pseudo,
    )
    candidate = _mapping(value.get("candidate"), "planner preflight candidate")
    if candidate.get("candidate_contract_sha256") != R198_CANDIDATE_CONTRACT_SHA256:
        raise R198ProductionFactoryError(
            "planner preflight candidate contract differs from the frozen r197 candidate"
        )
    completion_receipt = _identity(
        candidate.get("r197_completion_receipt"),
        "planner preflight r197 completion receipt",
        root=inputs.candidate.package_root,
    )
    _identity_equal(
        completion_receipt,
        inputs.candidate.completion_receipt,
        "planner preflight/candidate snapshot r197 completion receipt",
    )
    authority = inputs.authority
    if authority.get("training_eligible") is not False:
        raise R198ProductionFactoryError("planner preflight must remain training-ineligible")
    artifacts, capability = _pairing_artifacts(pseudo, inputs.candidate.source_root)
    _crosscheck_cg_against_pairing(inputs.cg, artifacts, capability)
    return value, inputs, {"artifacts": artifacts, "capability": capability}


def _preflight_arm_bindings(
    preflight: Mapping[str, Any], inputs: _FactoryInputs
) -> dict[str, dict[str, Any]]:
    candidate = inputs.candidate
    profile_root = inputs.evaluation_inputs_root
    if profile_root is None:
        raise R198ProductionFactoryError("planner preflight needs sealed runtime profiles under evaluation_inputs_root")
    raw_arms = _mapping(preflight.get("arms"), "planner preflight arms")
    result: dict[str, dict[str, Any]] = {}
    for arm in ("direct_bridge_recursive_disabled", "recursive_rtp"):
        row = _mapping(raw_arms.get(arm), f"planner preflight arm {arm}")
        runtime_artifact = _identity(row.get("runtime_artifact"), f"planner preflight {arm} runtime artifact", root=candidate.source_root)
        runtime_profile = _identity(row.get("runtime_profile"), f"planner preflight {arm} runtime profile", root=profile_root)
        sidecar = _identity(row.get("rtp_sidecar"), f"planner preflight {arm} sidecar", root=candidate.package_root)
        _identity_equal(sidecar, candidate.sidecar, f"planner preflight {arm} sidecar")
        profile_payload = _read_json(Path(runtime_profile["path"]), f"planner preflight {arm} profile")
        profile = _mapping(profile_payload.get("rtp", profile_payload), f"planner preflight {arm} profile rtp")
        expected = {
            "evaluation_arm": arm,
            "sizing_profile": R198_PROFILE,
            "max_neural_passes": R198_MAX_NEURAL_PASSES,
            "max_action_combos": R198_MAX_ACTION_COMBOS,
            "num_plan_candidates": R198_NUM_CANDIDATES,
            "max_recursion_depth": R198_MAX_RECURSION_DEPTH,
            "recursive_turn_planner_enabled": True,
            "direct_bridge_enabled": True,
            "force_direct_bridge_only": arm == "direct_bridge_recursive_disabled",
        }
        for key, required in expected.items():
            if profile.get(key) != required:
                raise R198ProductionFactoryError(f"planner preflight {arm} profile fails {key}")
        result[arm] = {
            "runtime_artifact": runtime_artifact,
            "runtime_profile": runtime_profile,
            "sidecar": sidecar,
        }
    return result


def _fixture_observation(
    raw: Mapping[str, Any],
    name: str,
    *,
    inputs: _FactoryInputs,
    pairing: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore a sealed native fixture and cross-check its sealed JSON view.

    Materializer records a physical JSON observation to make the preflight
    inputs auditable without a native decoder, but the factory never trusts it
    as an independently authored stimulus.  It restores the exact v2 snapshot
    seal, reads the native observation, and requires canonical equality first.
    """

    fixture = _mapping(raw, f"planner preflight {name} fixture")
    if fixture.get("expected_mode") != (
        "recursive_plan" if name == "normal" else "forced_replan"
    ):
        raise R198ProductionFactoryError(f"planner preflight {name} fixture expected_mode differs")
    root = inputs.evaluation_inputs_root
    if root is None:
        raise R198ProductionFactoryError("planner preflight needs a sealed evaluation_inputs_root")
    observation_identity = _identity(
        fixture.get("observation"), f"planner preflight {name} observation", root=root
    )
    payload = _read_json(Path(observation_identity["path"]), f"planner preflight {name} observation")
    if fixture.get("observation_sha256") not in {None, observation_identity["sha256"]}:
        raise R198ProductionFactoryError(f"planner preflight {name} observation binding differs")
    snapshot_identity = _identity(
        fixture.get("snapshot_artifact"), f"planner preflight {name} snapshot artifact", root=root
    )
    seal_identity = _identity(
        fixture.get("snapshot_seal", fixture.get("seal")),
        f"planner preflight {name} snapshot seal",
        root=root,
    )
    seal = _read_json(Path(seal_identity["path"]), f"planner preflight {name} snapshot seal")
    from .engine_rebuild.rtp_pairing_snapshot import (
        SNAPSHOT_SEAL_SCHEMA,
        RtpPairingSnapshotEngine,
        snapshot_abi_contract,
        snapshot_abi_sha256,
    )

    artifacts = pairing["artifacts"]
    if seal.get("schema") != SNAPSHOT_SEAL_SCHEMA or seal.get("status") != "sealed":
        raise R198ProductionFactoryError(f"planner preflight {name} snapshot seal is not sealed")
    expected_seal = {
        "engine_artifact_sha256": artifacts.engine_artifact["sha256"],
        "source_artifact_sha256": artifacts.source_artifact["sha256"],
        "patch_artifact_sha256": artifacts.patch_artifact["sha256"],
        "build_artifact_sha256": artifacts.build_artifact["sha256"],
        "canonical_abi_sha256": snapshot_abi_sha256(),
        "capture_boundary": snapshot_abi_contract()["capture_boundary"],
        "boundary_tag": snapshot_abi_contract()["boundary_tag"],
        "snapshot_artifact_sha256": snapshot_identity["sha256"],
        "snapshot_artifact_bytes": snapshot_identity["bytes"],
        "requested_seed_is_pairing_proof": False,
    }
    for key, expected in expected_seal.items():
        if seal.get(key) != expected:
            raise R198ProductionFactoryError(f"planner preflight {name} seal fails {key}")
    nested = _identity(seal.get("snapshot_artifact"), f"planner preflight {name} nested snapshot", root=root)
    _identity_equal(nested, snapshot_identity, f"planner preflight {name} seal snapshot")
    candidate_deck = _identity(
        fixture.get("candidate_deck"), f"planner preflight {name} candidate deck", root=inputs.candidate.package_root
    )
    _identity_equal(candidate_deck, inputs.candidate.deck, f"planner preflight {name} candidate deck")
    opponent_deck = _identity(
        fixture.get("opponent_deck"), f"planner preflight {name} opponent deck", root=root
    )
    candidate_seat = fixture.get("candidate_seat")
    if candidate_seat not in {0, 1}:
        raise R198ProductionFactoryError(f"planner preflight {name} candidate seat is invalid")
    opponent_id = _text(fixture.get("opponent_id"), f"planner preflight {name} opponent id")
    if opponent_id not in OFFICIAL_CONTROL_DIGESTS:
        raise R198ProductionFactoryError(f"planner preflight {name} opponent is not official")
    replicate = fixture.get("replicate")
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
        raise R198ProductionFactoryError(f"planner preflight {name} replicate is invalid")
    if seal.get("candidate_deck_sha256") != candidate_deck["sha256"] or seal.get(
        "candidate_deck_order_sha256"
    ) != candidate_deck["sha256"]:
        raise R198ProductionFactoryError(f"planner preflight {name} seal candidate deck differs")
    if seal.get("opponent_deck_sha256") != opponent_deck["sha256"] or seal.get(
        "opponent_deck_order_sha256"
    ) != opponent_deck["sha256"]:
        raise R198ProductionFactoryError(f"planner preflight {name} seal opponent deck differs")
    if seal.get("candidate_seat") != candidate_seat:
        raise R198ProductionFactoryError(f"planner preflight {name} seal candidate seat differs")
    if seal.get("opponent_id") != opponent_id or seal.get("replicate") != replicate:
        raise R198ProductionFactoryError(f"planner preflight {name} seal cell differs")
    if seal.get("purpose") != "r198_planner_pass_preflight_only_non_scored":
        raise R198ProductionFactoryError(f"planner preflight {name} seal has an unsafe purpose")
    for key, value in (
        ("evaluation_only", True),
        ("training_eligible", False),
        ("replay_eligible", False),
        ("serving_eligible", False),
        ("action_authority_enabled", False),
        ("kaggle_submission_authorized", False),
    ):
        if seal.get(key) is not value:
            raise R198ProductionFactoryError(f"planner preflight {name} seal fails {key}")
    # This is deliberately a *fresh* preflight engine.  Its only state
    # mutation is the sealed restore itself; no start/capture/raw-byte restore
    # endpoint is ever exposed to the materializer or preflight caller.
    if any(inputs.cg.library.get(key) != artifacts.engine_artifact.get(key) for key in ("sha256", "bytes")):
        raise R198ProductionFactoryError("planner preflight CG DSO differs from pairing engine")
    engine = RtpPairingSnapshotEngine(inputs.cg.library["path"])
    engine.require_bound_artifacts(artifacts)
    _require_shared_cg_dso_handle(engine, inputs.cg)
    battle = engine.restore_sealed_snapshot_manifest(seal_identity["path"])
    try:
        actual = battle.observation()
    finally:
        battle.close()
    if not isinstance(actual, Mapping):
        raise R198ProductionFactoryError(f"planner preflight {name} native observation is invalid")
    if _canonical_digest(dict(actual)) != _canonical_digest(payload):
        raise R198ProductionFactoryError(
            f"planner preflight {name} sealed observation differs from actual native restore"
        )
    return payload, {
        "snapshot_artifact": snapshot_identity,
        "snapshot_seal": seal_identity,
        "observation": observation_identity,
        "candidate_deck": candidate_deck,
        "opponent_deck": opponent_deck,
        "candidate_seat": candidate_seat,
        "opponent_id": opponent_id,
        "replicate": replicate,
    }


def _run_r198_planner_pass_probes(
    *,
    preflight: Mapping[str, Any],
    inputs: _FactoryInputs,
    pairing: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], int, int]:
    """Execute only non-selecting planner paths for the sealed 6/5 proof.

    The r197 action fence correctly blocks an unpromoted sidecar from selecting
    a game action.  Planner preflight is not an action evaluation: it restores
    sealed fixtures and exercises the bridge encoder plus ``plan_turn`` with
    no action submitted to a battle.  Keeping it non-selecting proves the
    actual normal/forced planner branches without broadening authority.
    """

    arms = _preflight_arm_bindings(preflight, inputs)
    fixtures = _mapping(preflight.get("fixtures"), "planner preflight fixtures")
    # ``_fixture_observation`` imports cg.sim to prove its handle equals the
    # wrapper's.  Pin the snapshot closure before either native import.
    with _temporary_rtp_environment(inputs.candidate, cg_runtime=inputs.cg.runtime_root):
        normal_obs, normal_fixture = _fixture_observation(
            _mapping(fixtures.get("normal"), "planner normal fixture"),
            "normal",
            inputs=inputs,
            pairing=pairing,
        )
        forced_obs, forced_fixture = _fixture_observation(
            _mapping(fixtures.get("forced_replan"), "planner forced fixture"),
            "forced_replan",
            inputs=inputs,
            pairing=pairing,
        )
        matchup_adapter_registry_digest = _require_snapshot_matchup_adapter_registry(
            inputs
        )
        model = _load_parent_model(inputs.candidate)
        _require_model_matchup_adapter_registry(model, inputs)
        deck = _validate_candidate_deck(inputs.candidate.deck)
        from . import features
        from .agent import PolicyAgent

        agent = PolicyAgent(
            model=model,
            deck=list(deck),
            use_recursive_turn_planner=True,
            rtp_sizing_profile=R198_PROFILE,
            rtp_max_action_combos=R198_MAX_ACTION_COMBOS,
            force_direct_bridge_only=False,
            strict_runtime=True,
            collect_targets=False,
            sample_actions=False,
            use_mcts=False,
            leaf_backend=None,
            checkpoint_digest=R195_PARENT_SHA256,
            matchup_adapter_runtime=True,
            matchup_adapter_tree_path=str(inputs.candidate.matchup_tree["path"]),
        )
        _require_policy_router_registry(
            agent, matchup_adapter_registry_digest
        )
        bridge = agent._rtp_bridge
        if bridge is None:
            raise R198ProductionFactoryError(
                "planner preflight could not initialize r197 bridge"
            )

        def encode_fixture(observation: Mapping[str, Any]) -> tuple[Any, Any]:
            # This is the same causal public router update the next policy
            # action would use, but no decision history or action token is
            # appended because planner preflight must not select an action.
            agent.reset_game()
            agent._matchup_adapter_shadow_router.observe(
                dict(observation),
                scope="game_root",
                depth=len(agent.board_history),
            )
            board = features.build_board_tokens(dict(observation), agent.deck)
            legal = bridge._legal_actions(dict(observation))
            if not legal:
                raise R198ProductionFactoryError(
                    "planner preflight fixture has no complete legal action"
                )
            return bridge.encode(
                dict(observation),
                board=board,
                legal_actions=legal,
                append_cache=False,
            )

        normal_memory, normal_logits = encode_fixture(normal_obs)
        normal_decision = bridge.planner.plan_turn(
            normal_memory, policy_logits=normal_logits, force_recurse=None
        )
        normal_passes = int(normal_decision.neural_passes)
        if normal_decision.mode != "recursive_plan":
            raise R198ProductionFactoryError(
                "planner preflight normal fixture did not execute recursive_plan"
            )

        forced_memory, forced_logits = encode_fixture(forced_obs)
        forced_decision = bridge.planner.plan_turn(
            forced_memory, policy_logits=forced_logits, force_recurse=True
        )
        forced_passes = int(forced_decision.neural_passes)
        if forced_decision.program is None:
            raise R198ProductionFactoryError(
                "planner preflight forced fixture did not produce a recursive program"
            )
    if normal_passes != R198_NORMAL_PASSES or forced_passes != R198_FORCED_REPLAN_PASSES:
        raise R198ProductionFactoryError(
            "planner preflight observed pass counts differ from the exact 6/5 contract"
        )
    return arms, normal_fixture, forced_fixture, normal_passes, forced_passes


def run_r198_planner_preflight(
    preflight_input: Mapping[str, Any] | str | Path,
    output_path: str | Path,
) -> Path:
    """Run the real r197 normal and forced-replan paths and seal a receipt.

    The input producer supplies immutable, source-snapshot-local observations
    chosen to exercise the established normal/forced branches.  This function
    does not synthesize a planner result or hand-author a pass count: it loads
    the inert sidecar and invokes ``plan_turn`` for both paths.
    """

    if isinstance(preflight_input, (str, Path)):
        source = _physical_path(preflight_input, "planner preflight input", file=True)
        _readonly_mode(source, "planner preflight input", exact=0o444)
        raw = _read_json(source, "planner preflight input")
        input_identity = {"path": str(source), "sha256": _sha256_file(source), "bytes": source.stat().st_size}
    else:
        raw = _mapping(preflight_input, "planner preflight input")
        input_identity = None
    preflight, inputs, pairing = _preflight_inputs(raw)
    arms, normal_fixture, forced_fixture, normal_passes, forced_passes = (
        _run_r198_planner_pass_probes(
            preflight=preflight,
            inputs=inputs,
            pairing=pairing,
        )
    )
    material = dict(preflight)
    input_digest = _canonical_digest(material)
    payload = {
        "schema": PREFLIGHT_RECEIPT_SCHEMA,
        "status": "passed",
        "preflight_input_sha256": input_digest,
        "preflight_input": input_identity,
        "candidate_id": R198_CANDIDATE_ID,
        "candidate_contract_sha256": R198_CANDIDATE_CONTRACT_SHA256,
        "parent_checkpoint_sha256": inputs.candidate.parent_checkpoint["sha256"],
        "sidecar_sha256": inputs.candidate.sidecar["sha256"],
        "r197_completion_receipt_sha256": inputs.candidate.completion_receipt[
            "sha256"
        ],
        "sidecar_config_sha256": R197_SIDECAR_CONFIG_SHA256,
        "direct_runtime_profile_sha256": arms["direct_bridge_recursive_disabled"]["runtime_profile"]["sha256"],
        "recursive_runtime_profile_sha256": arms["recursive_rtp"]["runtime_profile"]["sha256"],
        "pairing_capability_sha256": pairing["capability"]["receipt"]["sha256"],
        "evaluation_cg_library_sha256": inputs.cg.library["sha256"],
        "evaluation_cg_closure_sha256": inputs.cg.closure_manifest["sha256"],
        "matchup_adapter_registry_sha256": inputs.matchup_adapter_registry["sha256"],
        "matchup_adapter_slot_registry_digest": inputs.matchup_adapter_registry_digest,
        "normal_fixture": normal_fixture,
        "forced_replan_fixture": forced_fixture,
        "max_neural_passes": R198_MAX_NEURAL_PASSES,
        "max_action_combos": R198_MAX_ACTION_COMBOS,
        "num_plan_candidates": R198_NUM_CANDIDATES,
        "max_recursion_depth": R198_MAX_RECURSION_DEPTH,
        "normal_probe_completed": True,
        "normal_probe_observed_neural_passes": normal_passes,
        "normal_probe_mode": "recursive_plan",
        "forced_replan_probe_completed": True,
        "forced_replan_probe_observed_neural_passes": forced_passes,
        "forced_replan_probe_method": "planner.plan_turn(force_recurse=True)",
        "preflight_selection_authority_used": False,
        "neural_budget_failures": 0,
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_change_authorized": False,
        "selector_change_authorized": False,
        "action_authority_authorized": False,
        "kaggle_submission_authorized": False,
    }
    payload["preflight_receipt_input_sha256"] = _canonical_digest(payload)
    return _atomic_readonly_json(Path(output_path), payload)


@contextlib.contextmanager
def _temporary_rtp_environment(
    candidate: _CandidateAssets, *, cg_runtime: Path | None = None
) -> Iterator[None]:
    """Scoped sidecar env for standalone preflight only; always restore it."""

    keys = {
        "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "1",
        "POKEBOT_RTP_CHECKPOINT": str(candidate.sidecar["path"]),
        "POKEBOT_RTP_SIZING_PROFILE": R198_PROFILE,
        "POKEBOT_RTP_MAX_ACTION_COMBOS": str(R198_MAX_ACTION_COMBOS),
        "POKEBOT_RTP_SERVING_QUALIFIED": "0",
        "POKEBOT_RTP_PARENT_CHECKPOINT_SHA256": R195_PARENT_SHA256,
        "POKEBOT_RTP_PROMOTION_RECEIPT": "",
        "POKEBOT_RTP_PROMOTION_RECEIPT_SHA256": "",
    }
    if cg_runtime is not None:
        keys["CG_LIB_PATH"] = str(cg_runtime)
    old = {key: os.environ.get(key, _MISSING) for key in keys}
    try:
        os.environ.update(keys)
        yield
    finally:
        for key, value in old.items():
            if value is _MISSING:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)


__all__ = [
    "CANDIDATE_SNAPSHOT_SCHEMA",
    "EVALUATION_CG_CLOSURE_FILENAME",
    "FACTORY_INPUT_SCHEMA",
    "PREFLIGHT_INPUT_SCHEMA",
    "PREFLIGHT_RECEIPT_SCHEMA",
    "ProductionR198EvaluationFactory",
    "R198ProductionFactoryError",
    "build_r198_evaluator_base_spec",
    "r198_factory_evaluation_authority_payload",
    "r198_runtime_profile_payload",
    "run_r198_planner_preflight",
]
