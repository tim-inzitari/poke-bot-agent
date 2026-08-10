"""Isolated native runtime bridge for the Alakazam Guide2Vec r212 mirror.

The r212 compiler in :mod:`poke_bot.guide2vec_bo1000` deliberately knows
nothing about libcg, package imports, or model loading.  This module is the
small, separate execution bridge that supplies it with receipt-backed games.

It is intentionally direct-policy-only:

* both seats run the exact archived r195 NO-RTP package and its runtime-on V6
  matchup adapter/tree;
* the candidate owns one frozen ``Guide2VecHead`` outside the base model;
* the control factory never imports or attaches that head, and its object graph
  is inspected for Guide2Vec modules, tensors, hooks, and guide transforms;
* native seeded starts seal the actual first player before either game in a
  pair is played; and
* no service, selector, Kaggle, trainer, planner, or simulator-search path is
  imported here.

The public entry points are deliberately split into plan/materialization and
execution phases.  Constructing a plan or preflight receipt does *not* start a
battle.  ``run_guide2vec_bo1000`` is the only function that may create native
battles, and callers must invoke it explicitly after the frozen sidecar has
passed its training gate.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import tarfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from torch import Tensor, nn

from .guide2vec import (
    FrozenBaseIdentity,
    Guide2VecDecision,
    Guide2VecError,
    Guide2VecHead,
    MAX_LOGIT_BONUS,
    load_checkpoint_payload,
    state_dict_sha256,
)
from .guide2vec_bo1000 import (
    CANDIDATE_GUIDE2VEC_PRESENCE,
    CONTROL_ARM,
    CONTROL_GUIDE2VEC_PRESENCE,
    GUIDE2VEC_ARM,
    GUIDE2VEC_BO1000_GAME_COUNT,
    GUIDE2VEC_EVALUATION_ID,
    R195_BUNDLE_SHA256,
    R195_CHECKPOINT_BYTES,
    R195_CHECKPOINT_SHA256,
    R195_DECK_CARDS_SHA256,
    R195_MATCHUP_TREE_SHA256,
    R212_CONTRACT_SHA256,
    FrozenR195RuntimeIdentity,
    Guide2VecBO1000Error,
    Guide2VecBO1000GameReceipt,
    Guide2VecBO1000GameSpec,
    Guide2VecDecisionReceipt,
    Guide2VecExperimentIdentity,
    build_guide2vec_bo1000_schedule,
    compile_guide2vec_bo1000_report,
    expected_control_guide2vec_absence_attestation,
    expected_is_first_attestation,
    expected_matchup_adapter_parity_attestation,
)
from .seeded_mirror_harness import (
    PairFirstPlayerSeal,
    SeededMirrorGameSpec,
    SeededMirrorHarnessError,
    build_seeded_seat_swapped_schedule,
    canonical_sha256 as _native_sha256,
    configure_battle_start_seeded,
    validate_pair_first_player_seal,
)


R212_RUNTIME_PLAN_SCHEMA = "poke_bot.alakazam_guide2vec_bo1000_r212_plan/v1"
R212_RUNTIME_PREFLIGHT_SCHEMA = (
    "poke_bot.alakazam_guide2vec_bo1000_r212_preflight/v1"
)
R212_NATIVE_PAIR_BINDING_SCHEMA = (
    "poke_bot.alakazam_guide2vec_bo1000_r212_native_pair_binding/v1"
)
R212_GAME_TRACE_SCHEMA = "poke_bot.alakazam_guide2vec_bo1000_r212_game_trace/v1"
R212_FINAL_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_guide2vec_bo1000_r212_execution_receipt/v1"
)
R212_RUNTIME_GRAPH_SCHEMA = "poke_bot.alakazam_guide2vec_runtime_graph/v1"

R195_CONTRACT_SHA256 = (
    "sha256:e37cf1d3e638c3aed56230c9fa970c61e6c1ed8b4bd3024de259cb9847c31e48"
)
R195_SUBMISSION_ID = 55_378_392
R195_SUBMISSION_MESSAGE = (
    "alakazam training milestone iter 21 copy 1/2 first 261d367e131e NO RTP"
)
R195_ADAPTER_FORMAT = "poke-bot-matchup-adapter-bank-v6"
R195_ADAPTER_SLOT_CAPACITY = 64
R212_TRAINING_RECEIPT_SCHEMA = "poke_bot.alakazam_guide2vec_r212_training_receipt/v1"
class Guide2VecBO1000RuntimeError(RuntimeError):
    """Raised when a native r212 mirror cannot be truthfully materialized."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Guide2VecBO1000RuntimeError("value is not canonical JSON") from exc


def canonical_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _training_json_sha256(value: object) -> str:
    """Match r212 trainer hashes, whose canonical JSON includes one newline."""

    return "sha256:" + hashlib.sha256(_canonical_bytes(value) + b"\n").hexdigest()


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise Guide2VecBO1000RuntimeError(f"expected a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _require_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise Guide2VecBO1000RuntimeError(f"{label} must be a sha256 digest")
    suffix = value[7:]
    if len(suffix) != 64 or any(char not in "0123456789abcdef" for char in suffix):
        raise Guide2VecBO1000RuntimeError(f"{label} must be a lowercase sha256 digest")
    return value


def _require_exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or int(value) < minimum:
        raise Guide2VecBO1000RuntimeError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Guide2VecBO1000RuntimeError(f"{label} is unreadable: {source}") from exc
    if not isinstance(value, dict):
        raise Guide2VecBO1000RuntimeError(f"{label} must be a JSON object: {source}")
    return value


def _safe_torch_load(
    path: Path,
    *,
    allow_checksum_bound_legacy: bool = False,
) -> Mapping[str, object]:
    """Load tensor payloads safely, allowing legacy pickle only for pinned r195."""

    legacy_fallback = False
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older supported torch builds have no ``weights_only`` argument.  The
        # caller has already checksum-bound the local artifact at this point.
        legacy_fallback = True
    except Exception as exc:  # noqa: BLE001 - normalize unsafe/malformed loads.
        if not allow_checksum_bound_legacy:
            raise Guide2VecBO1000RuntimeError(f"cannot read tensor payload: {path}") from exc
        legacy_fallback = True
    if legacy_fallback:
        try:
            value = torch.load(path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001 - normalize legacy load failure.
            raise Guide2VecBO1000RuntimeError(f"cannot read tensor payload: {path}") from exc
    if not isinstance(value, Mapping):
        raise Guide2VecBO1000RuntimeError(f"tensor payload is not an object: {path}")
    return dict(value)


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    """Create one canonical receipt without overwriting earlier evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    body = _canonical_bytes(dict(payload)) + b"\n"
    if path.exists():
        if path.read_bytes() != body:
            raise Guide2VecBO1000RuntimeError(
                f"immutable r212 artifact already differs: {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != body:
                raise Guide2VecBO1000RuntimeError(
                    f"immutable r212 artifact race differs: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative_member(name: str) -> str:
    relative = PurePosixPath(name.removeprefix("./"))
    if (
        not relative.parts
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise Guide2VecBO1000RuntimeError(f"unsafe package member: {name!r}")
    return relative.as_posix()


def _directory_file_map(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise Guide2VecBO1000RuntimeError(f"package root is not a real directory: {root}")
    files: dict[str, str] = {}
    for entry in sorted(root.rglob("*")):
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise Guide2VecBO1000RuntimeError(f"package contains a symlink: {relative}")
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise Guide2VecBO1000RuntimeError(
                f"package contains a non-regular member: {relative}"
            )
        files[relative] = sha256_file(entry)
    if not files:
        raise Guide2VecBO1000RuntimeError("package root has no files")
    return files


def _tar_file_map(bundle: Path) -> dict[str, str]:
    if not bundle.is_file() or bundle.is_symlink():
        raise Guide2VecBO1000RuntimeError(f"bundle is not a regular file: {bundle}")
    files: dict[str, str] = {}
    try:
        archive = tarfile.open(bundle, "r:*")
    except (OSError, tarfile.TarError) as exc:
        raise Guide2VecBO1000RuntimeError(f"bundle cannot be read as tar: {bundle}") from exc
    with archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            relative = _safe_relative_member(member.name)
            if not member.isfile() or relative in files:
                raise Guide2VecBO1000RuntimeError(
                    f"bundle member is not a unique regular file: {relative}"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise Guide2VecBO1000RuntimeError(f"bundle member is unreadable: {relative}")
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            files[relative] = "sha256:" + digest.hexdigest()
    if not files:
        raise Guide2VecBO1000RuntimeError("bundle has no regular files")
    return files


def _strip_single_archive_root(files: Mapping[str, str]) -> dict[str, str]:
    """Normalize tarballs that wrap the package in exactly one top directory."""

    parts = {PurePosixPath(name).parts[0] for name in files}
    if len(parts) != 1:
        return dict(files)
    root = next(iter(parts))
    if not root:
        return dict(files)
    stripped: dict[str, str] = {}
    for name, digest in files.items():
        path = PurePosixPath(name)
        if len(path.parts) == 1:
            return dict(files)
        new_name = PurePosixPath(*path.parts[1:]).as_posix()
        if new_name in stripped:
            return dict(files)
        stripped[new_name] = digest
    return stripped


def _ordered_deck_cards_sha256(path: Path) -> str:
    cards: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        row = raw.strip()
        if not row or row.startswith("#"):
            continue
        try:
            cards.append(int(row.split(",", 1)[0]))
        except ValueError as exc:
            raise Guide2VecBO1000RuntimeError(
                f"package deck has a non-card row: {path}"
            ) from exc
    if len(cards) != 60:
        raise Guide2VecBO1000RuntimeError(
            f"r195 package deck must have exactly 60 cards, got {len(cards)}"
        )
    return canonical_sha256(cards)


def _tensor_mapping_sha256(state: Mapping[str, Tensor]) -> str:
    try:
        return state_dict_sha256(state)
    except Guide2VecError as exc:
        raise Guide2VecBO1000RuntimeError("tensor state is not canonical") from exc


def _checkpoint_model_config_and_adapter(
    checkpoint_path: Path,
) -> tuple[str, str, str, dict[str, object]]:
    """Read r195's serialized config and frozen V6 adapter provenance."""

    payload = _safe_torch_load(checkpoint_path, allow_checksum_bound_legacy=True)
    model_config = payload.get("model_config")
    model_state = payload.get("model_state_dict")
    extra = payload.get("extra")
    if not isinstance(model_config, Mapping) or not isinstance(model_state, Mapping):
        raise Guide2VecBO1000RuntimeError("r195 checkpoint lacks model config/state")
    if (
        model_config.get("decision_context") != "history"
        or int(model_config.get("d_model", -1)) != 96
        or model_config.get("h10_capacity_enabled") is not True
        or model_config.get("decision_fusion_enabled") is not True
        or model_config.get("combo_state_route_enabled") is not False
    ):
        raise Guide2VecBO1000RuntimeError("checkpoint is not the exact r195 temporal direct model")
    if not isinstance(extra, Mapping):
        raise Guide2VecBO1000RuntimeError("r195 checkpoint lacks extra provenance")
    adapter_config = extra.get("matchup_adapter_config")
    dormant = extra.get("dormant_matchup_adapter_bank")
    fit = extra.get("dormant_matchup_adapter_fit")
    if not isinstance(adapter_config, Mapping) or not isinstance(dormant, Mapping):
        raise Guide2VecBO1000RuntimeError("r195 checkpoint lacks trained adapter provenance")
    if adapter_config.get("format") != R195_ADAPTER_FORMAT:
        raise Guide2VecBO1000RuntimeError("r195 checkpoint does not carry V6 adapters")
    if int(adapter_config.get("slot_capacity", -1)) != R195_ADAPTER_SLOT_CAPACITY:
        raise Guide2VecBO1000RuntimeError("r195 V6 adapter slot capacity drifted")
    if len(list(adapter_config.get("physical_slots") or ())) not in {
        0,
        R195_ADAPTER_SLOT_CAPACITY,
    }:
        raise Guide2VecBO1000RuntimeError("r195 V6 adapter physical slots are malformed")
    if dormant.get("schema") != "poke_bot.trained_dormant_matchup_adapter/v1":
        raise Guide2VecBO1000RuntimeError("r195 adapter is not the trained frozen bank")
    if dormant.get("frozen") is not True or dormant.get("zero_output") is not False:
        raise Guide2VecBO1000RuntimeError("r195 trained adapter frozen/output contract drifted")
    if not isinstance(fit, Mapping) or fit.get("schema") != "poke_bot.dormant_matchup_adapter_fit/v1":
        raise Guide2VecBO1000RuntimeError("r195 adapter fit receipt is missing")
    if fit.get("runtime_enabled") is not False or fit.get("base_frozen") is not True:
        raise Guide2VecBO1000RuntimeError("r195 adapter fit receipt drifted")
    adapter_state: dict[str, Tensor] = {}
    for name, tensor in model_state.items():
        if not isinstance(name, str) or not name.startswith("matchup_adapter_bank."):
            continue
        if not isinstance(tensor, Tensor):
            raise Guide2VecBO1000RuntimeError("r195 adapter state has a non-tensor")
        adapter_state[name.removeprefix("matchup_adapter_bank.")] = tensor
    if not adapter_state:
        raise Guide2VecBO1000RuntimeError("r195 checkpoint has no adapter tensors")
    nonzero_output = any(
        int(tensor.detach().count_nonzero().item()) > 0
        for name, tensor in adapter_state.items()
        if name.endswith(".up.weight") or name.endswith(".up.bias")
    )
    if not nonzero_output:
        raise Guide2VecBO1000RuntimeError("r195 trained adapter has no nonzero output")
    # The frozen Guide2Vec payload was created by the r212 trainer, whose
    # model-config canonicalization deliberately includes a trailing newline.
    # This must not be replaced with this module's no-newline receipt hash.
    model_config_sha = _training_json_sha256(dict(model_config))
    adapter_bank_sha = _tensor_mapping_sha256(adapter_state)
    adapter_fit_sha = canonical_sha256(
        {
            "schema": "poke_bot.alakazam_r195_v6_adapter_training_binding/v1",
            "fit": dict(fit),
            "dormant": dict(dormant),
            "adapter_config": dict(adapter_config),
        }
    )
    return model_config_sha, adapter_bank_sha, adapter_fit_sha, dict(adapter_config)


def _checkpoint_model_state_sha256(checkpoint_path: Path) -> str:
    """Return a canonical digest of every exact r195 model tensor."""

    payload = _safe_torch_load(checkpoint_path, allow_checksum_bound_legacy=True)
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise Guide2VecBO1000RuntimeError("r195 checkpoint lacks full model state")
    typed: dict[str, Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise Guide2VecBO1000RuntimeError("r195 full model state is malformed")
        typed[name] = value
    if not typed:
        raise Guide2VecBO1000RuntimeError("r195 full model state is empty")
    return _tensor_mapping_sha256(typed)


@dataclass(frozen=True, slots=True)
class R212ArtifactIdentity:
    """Immutable inputs needed to construct one r212-only experiment plan."""

    r195_bundle: Path
    r195_package_root: Path
    r195_checkpoint: Path
    guide2vec_checkpoint: Path
    guide2vec_training_receipt: Path
    owner_contract: Path
    r195_contract: Path

    def __post_init__(self) -> None:
        for field in (
            "r195_bundle",
            "r195_package_root",
            "r195_checkpoint",
            "guide2vec_checkpoint",
            "guide2vec_training_receipt",
            "owner_contract",
            "r195_contract",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)).resolve())


@dataclass(frozen=True, slots=True)
class CandidateArtifact:
    checkpoint_sha256: str
    training_receipt_sha256: str
    runtime_config_sha256: str
    parameter_count: int
    state_dict_sha256: str
    base_identity_sha256: str
    source_snapshot_sha256: str
    component_graph_sha256: str
    feature_schema_sha256: str
    model_config_sha256: str


@dataclass(frozen=True, slots=True)
class ControlGraphAudit:
    observation_sha256: str
    module_instance_count: int
    parameter_count: int
    state_dict_key_count: int
    forward_hook_count: int
    linear_transform_count: int

    def as_payload(self) -> dict[str, object]:
        return {
            "schema": R212_RUNTIME_GRAPH_SCHEMA,
            "kind": "control_guide2vec_absence",
            "observation_sha256": self.observation_sha256,
            "guide2vec_presence": CONTROL_GUIDE2VEC_PRESENCE,
            "guide2vec_module_instance_count": self.module_instance_count,
            "guide2vec_parameter_count": self.parameter_count,
            "guide2vec_state_dict_key_count": self.state_dict_key_count,
            "guide2vec_forward_hook_count": self.forward_hook_count,
            "guide2vec_linear_transform_count": self.linear_transform_count,
            "guide2vec_disabled_or_zeroed": False,
        }


@dataclass(frozen=True, slots=True)
class CandidateGraphAudit:
    component_graph_sha256: str
    module_instance_count: int
    parameter_count: int
    frozen: bool

    def as_payload(self) -> dict[str, object]:
        return {
            "schema": R212_RUNTIME_GRAPH_SCHEMA,
            "kind": "candidate_guide2vec_component",
            "component_graph_sha256": self.component_graph_sha256,
            "guide2vec_presence": CANDIDATE_GUIDE2VEC_PRESENCE,
            "guide2vec_module_instance_count": self.module_instance_count,
            "guide2vec_parameter_count": self.parameter_count,
            "guide2vec_frozen": self.frozen,
        }


def _training_candidate_artifact(
    *,
    checkpoint_path: Path,
    receipt_path: Path,
    model_config_sha256: str,
) -> CandidateArtifact:
    """Verify the trained sidecar and recover its immutable r212 identities."""

    checkpoint_sha = sha256_file(checkpoint_path)
    receipt_sha = sha256_file(receipt_path)
    receipt = _read_json_object(receipt_path, label="Guide2Vec training receipt")
    if receipt.get("schema") != R212_TRAINING_RECEIPT_SCHEMA:
        raise Guide2VecBO1000RuntimeError("Guide2Vec receipt schema drifted")
    if receipt.get("status") != "complete_offline_candidate_only":
        raise Guide2VecBO1000RuntimeError("Guide2Vec training did not complete")
    guide = receipt.get("guide2vec")
    heldout = receipt.get("heldout")
    inputs = receipt.get("inputs")
    source_snapshot = receipt.get("source_snapshot")
    if not all(isinstance(value, Mapping) for value in (guide, heldout, inputs, source_snapshot)):
        raise Guide2VecBO1000RuntimeError("Guide2Vec receipt is missing frozen evidence")
    guide_map = dict(guide)
    heldout_map = dict(heldout)
    inputs_map = dict(inputs)
    source_snapshot_map = dict(source_snapshot)
    if source_snapshot_map.get("status") != "validated_published_snapshot":
        raise Guide2VecBO1000RuntimeError("Guide2Vec receipt lacks a validated source snapshot")
    _require_digest(
        source_snapshot_map.get("manifest_sha256"), label="Guide2Vec source snapshot manifest"
    )
    _require_digest(
        source_snapshot_map.get("source_tree_sha256"), label="Guide2Vec source snapshot tree"
    )
    if guide_map.get("checkpoint_sha256") != checkpoint_sha:
        raise Guide2VecBO1000RuntimeError("Guide2Vec receipt checkpoint digest mismatch")
    parameter_count = _require_exact_int(
        guide_map.get("parameter_count"), label="Guide2Vec parameter_count", minimum=100_000
    )
    if parameter_count > 500_000:
        raise Guide2VecBO1000RuntimeError("Guide2Vec sidecar exceeds 500k parameters")
    gate = heldout_map.get("teacher_agreement_gate")
    if not isinstance(gate, Mapping) or gate.get("passed") is not True:
        raise Guide2VecBO1000RuntimeError("Guide2Vec heldout teacher gate did not pass")
    if float(gate.get("minimum", 0.0) or 0.0) != 0.90:
        raise Guide2VecBO1000RuntimeError("Guide2Vec heldout threshold drifted")
    runtime = guide_map.get("runtime_config")
    if not isinstance(runtime, Mapping):
        raise Guide2VecBO1000RuntimeError("Guide2Vec runtime config receipt is absent")
    runtime_map = dict(runtime)
    runtime_sha = _require_digest(
        runtime_map.get("runtime_config_sha256"), label="Guide2Vec runtime config digest"
    )
    config = runtime_map.get("guide2vec_config")
    if not isinstance(config, Mapping) or float(config.get("max_logit_bonus", -1.0)) != MAX_LOGIT_BONUS:
        raise Guide2VecBO1000RuntimeError("Guide2Vec runtime cap is not exactly 0.05")
    checkpoint_info = inputs_map.get("checkpoint")
    protected = inputs_map.get("protected_corpus")
    if not isinstance(checkpoint_info, Mapping) or not isinstance(protected, Mapping):
        raise Guide2VecBO1000RuntimeError("Guide2Vec receipt input identity is missing")
    if checkpoint_info.get("sha256") != R195_CHECKPOINT_SHA256:
        raise Guide2VecBO1000RuntimeError("Guide2Vec receipt parent checkpoint drifted")
    if checkpoint_info.get("bundle_sha256") != R195_BUNDLE_SHA256:
        raise Guide2VecBO1000RuntimeError("Guide2Vec receipt parent bundle drifted")
    if checkpoint_info.get("model_config_sha256") != model_config_sha256:
        raise Guide2VecBO1000RuntimeError("Guide2Vec receipt model config drifted")
    adapter_input = inputs_map.get("matchup_adapter")
    if not isinstance(adapter_input, Mapping):
        raise Guide2VecBO1000RuntimeError("Guide2Vec receipt lacks r195 V6 adapter evidence")
    adapter_input_map = dict(adapter_input)
    runtime_tree = adapter_input_map.get("runtime_tree")
    input_adapter_config = adapter_input_map.get("adapter_config")
    if (
        not isinstance(input_adapter_config, Mapping)
        or input_adapter_config.get("format") != R195_ADAPTER_FORMAT
        or int(input_adapter_config.get("slot_capacity", -1))
        != R195_ADAPTER_SLOT_CAPACITY
    ):
        raise Guide2VecBO1000RuntimeError("Guide2Vec receipt adapter format drifted")
    if not isinstance(runtime_tree, Mapping) or (
        runtime_tree.get("sha256") != R195_MATCHUP_TREE_SHA256
        or runtime_tree.get("runtime_enabled") is not True
    ):
        raise Guide2VecBO1000RuntimeError("Guide2Vec receipt adapter tree/runtime drifted")
    feature_schema_sha = _require_digest(
        protected.get("manifest_sha256"), label="Guide2Vec feature schema digest"
    )

    payload = _safe_torch_load(checkpoint_path)
    try:
        expected_base = FrozenBaseIdentity(
            submission_id=R195_SUBMISSION_ID,
            checkpoint_sha256=R195_CHECKPOINT_SHA256,
            checkpoint_bytes=R195_CHECKPOINT_BYTES,
            bundle_sha256=R195_BUNDLE_SHA256,
            model_config_sha256=model_config_sha256,
            feature_schema_sha256=feature_schema_sha,
        )
        head, base_identity, metadata = load_checkpoint_payload(
            payload,
            expected_base_identity=expected_base,
            map_location="cpu",
        )
    except Guide2VecError as exc:
        raise Guide2VecBO1000RuntimeError("Guide2Vec checkpoint failed strict identity load") from exc
    if base_identity != expected_base or head.parameter_count != parameter_count:
        raise Guide2VecBO1000RuntimeError("Guide2Vec sidecar identity/accounting mismatch")
    if head.config.as_dict() != dict(config):
        raise Guide2VecBO1000RuntimeError("Guide2Vec checkpoint did not serialize runtime calibration")
    if metadata.get("kind") != "frozen_candidate":
        raise Guide2VecBO1000RuntimeError("Guide2Vec sidecar is not the frozen candidate")
    metadata_runtime_sha = metadata.get("guide2vec_runtime_config_sha256")
    if metadata_runtime_sha != runtime_sha:
        raise Guide2VecBO1000RuntimeError("Guide2Vec metadata runtime config drifted")
    metadata_gate = metadata.get("heldout_teacher_agreement_gate")
    if not isinstance(metadata_gate, Mapping) or metadata_gate.get("passed") is not True:
        raise Guide2VecBO1000RuntimeError("Guide2Vec checkpoint omits heldout gate")
    state_sha = _require_digest(
        payload.get("state_dict_sha256"), label="Guide2Vec serialized state digest"
    )
    base_identity_sha = _require_digest(
        payload.get("base_identity_sha256"), label="Guide2Vec serialized base identity digest"
    )
    if base_identity_sha != base_identity.identity_sha256:
        raise Guide2VecBO1000RuntimeError("Guide2Vec serialized base identity drifted")

    component_graph_sha = canonical_sha256(
        {
            "schema": R212_RUNTIME_GRAPH_SCHEMA,
            "kind": "frozen_guide2vec_component",
            "checkpoint_sha256": checkpoint_sha,
            "runtime_config_sha256": runtime_sha,
            "parameter_count": parameter_count,
            "state_dict_sha256": state_sha,
            "base_identity_sha256": base_identity_sha,
            "max_logit_bonus": MAX_LOGIT_BONUS,
            "frozen": True,
        }
    )
    return CandidateArtifact(
        checkpoint_sha256=checkpoint_sha,
        training_receipt_sha256=receipt_sha,
        runtime_config_sha256=runtime_sha,
        parameter_count=parameter_count,
        state_dict_sha256=state_sha,
        base_identity_sha256=base_identity_sha,
        source_snapshot_sha256=canonical_sha256(source_snapshot_map),
        component_graph_sha256=component_graph_sha,
        feature_schema_sha256=feature_schema_sha,
        model_config_sha256=model_config_sha256,
    )


def verify_r212_artifacts(artifacts: R212ArtifactIdentity) -> tuple[
    CandidateArtifact,
    str,
    str,
    str,
    dict[str, object],
    str,
]:
    """Verify immutable package/checkpoint/head inputs without starting a game.

    The returned tuple is ``(candidate, model_config, adapter_bank,
    adapter_training_receipt, adapter_config, package_manifest)``.
    """

    if sha256_file(artifacts.owner_contract) != R212_CONTRACT_SHA256:
        raise Guide2VecBO1000RuntimeError("r212 typed owner contract digest drifted")
    if sha256_file(artifacts.r195_contract) != R195_CONTRACT_SHA256:
        raise Guide2VecBO1000RuntimeError("r195 typed contract digest drifted")
    if sha256_file(artifacts.r195_bundle) != R195_BUNDLE_SHA256:
        raise Guide2VecBO1000RuntimeError("r195 NO-RTP bundle digest drifted")
    if sha256_file(artifacts.r195_checkpoint) != R195_CHECKPOINT_SHA256:
        raise Guide2VecBO1000RuntimeError("r195 checkpoint digest drifted")
    if artifacts.r195_checkpoint.stat().st_size != R195_CHECKPOINT_BYTES:
        raise Guide2VecBO1000RuntimeError("r195 checkpoint byte count drifted")
    package_files = _directory_file_map(artifacts.r195_package_root)
    archive_files = _strip_single_archive_root(_tar_file_map(artifacts.r195_bundle))
    if package_files != archive_files:
        raise Guide2VecBO1000RuntimeError("extracted r195 package differs from bundle")
    package_manifest_sha = canonical_sha256(package_files)
    package_checkpoint = artifacts.r195_package_root / "model.pt"
    if sha256_file(package_checkpoint) != R195_CHECKPOINT_SHA256:
        raise Guide2VecBO1000RuntimeError("r195 package model.pt is not exact checkpoint")
    if package_checkpoint.stat().st_size != R195_CHECKPOINT_BYTES:
        raise Guide2VecBO1000RuntimeError("r195 package model.pt bytes drifted")
    deck_path = artifacts.r195_package_root / "deck.csv"
    if _ordered_deck_cards_sha256(deck_path) != R195_DECK_CARDS_SHA256:
        raise Guide2VecBO1000RuntimeError("r195 package deck order/cards drifted")
    model_config_sha, adapter_bank_sha, adapter_fit_sha, adapter_config = (
        _checkpoint_model_config_and_adapter(artifacts.r195_checkpoint)
    )
    candidate = _training_candidate_artifact(
        checkpoint_path=artifacts.guide2vec_checkpoint,
        receipt_path=artifacts.guide2vec_training_receipt,
        model_config_sha256=model_config_sha,
    )
    return (
        candidate,
        model_config_sha,
        adapter_bank_sha,
        adapter_fit_sha,
        adapter_config,
        package_manifest_sha,
    )


def build_guide2vec_bo1000_plan(
    *,
    artifacts: R212ArtifactIdentity,
    seed_identity_sha256: str,
) -> dict[str, object]:
    """Build a receipt-bound r212 schedule without loading an engine or game."""

    _require_digest(seed_identity_sha256, label="seed_identity_sha256")
    (
        candidate,
        model_config_sha,
        adapter_bank_sha,
        adapter_fit_sha,
        adapter_config,
        package_manifest_sha,
    ) = verify_r212_artifacts(artifacts)
    adapter_runtime_graph_sha = canonical_sha256(
        {
            "schema": R212_RUNTIME_GRAPH_SCHEMA,
            "kind": "r195_runtime_on_frozen_v6_adapter",
            "adapter_bank_sha256": adapter_bank_sha,
            "adapter_training_receipt_sha256": adapter_fit_sha,
            "adapter_config": adapter_config,
            "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
            "enabled": True,
            "trained": True,
            "frozen": True,
            "slot_capacity": R195_ADAPTER_SLOT_CAPACITY,
        }
    )
    direct_graph_sha = canonical_sha256(
        {
            "schema": R212_RUNTIME_GRAPH_SCHEMA,
            "kind": "exact_r195_no_rtp_direct_policy",
            "submission_id": R195_SUBMISSION_ID,
            "submission_message": R195_SUBMISSION_MESSAGE,
            "checkpoint_sha256": R195_CHECKPOINT_SHA256,
            "checkpoint_bytes": R195_CHECKPOINT_BYTES,
            "bundle_sha256": R195_BUNDLE_SHA256,
            "deck_cards_sha256": R195_DECK_CARDS_SHA256,
            "package_manifest_sha256": package_manifest_sha,
            "model_config_sha256": model_config_sha,
            "matchup_adapter_runtime_graph_sha256": adapter_runtime_graph_sha,
            "rtp_enabled": False,
            "mcts_enabled": False,
            "guide_linear_enabled": False,
            "guide_logit_enabled": False,
            "guide2vec_enabled": False,
        }
    )
    base_runtime = FrozenR195RuntimeIdentity(
        model_config_sha256=model_config_sha,
        matchup_tree_sha256=R195_MATCHUP_TREE_SHA256,
        matchup_adapter_bank_sha256=adapter_bank_sha,
        matchup_adapter_training_receipt_sha256=adapter_fit_sha,
        matchup_adapter_runtime_graph_sha256=adapter_runtime_graph_sha,
        matchup_adapter_enabled=True,
        matchup_adapter_trained=True,
        matchup_adapter_frozen=True,
        direct_runtime_graph_sha256=direct_graph_sha,
    )
    output_identity = canonical_sha256(
        {
            "schema": R212_RUNTIME_PLAN_SCHEMA,
            "kind": "isolated_r212_bo1000_output",
            "r212_contract_sha256": R212_CONTRACT_SHA256,
            "base_runtime_identity_sha256": base_runtime.identity_sha256,
            "guide2vec_checkpoint_sha256": candidate.checkpoint_sha256,
            "guide2vec_training_receipt_sha256": candidate.training_receipt_sha256,
            "source_snapshot_sha256": candidate.source_snapshot_sha256,
            "seed_identity_sha256": seed_identity_sha256,
        }
    )
    candidate_graph_sha = canonical_sha256(
        {
            "schema": R212_RUNTIME_GRAPH_SCHEMA,
            "kind": "r195_direct_plus_one_frozen_guide2vec",
            "direct_runtime_graph_sha256": direct_graph_sha,
            "guide2vec_component_graph_sha256": candidate.component_graph_sha256,
            "guide2vec_checkpoint_sha256": candidate.checkpoint_sha256,
            "guide2vec_runtime_config_sha256": candidate.runtime_config_sha256,
            "guide2vec_parameter_count": candidate.parameter_count,
            "maximum_logit_bonus": MAX_LOGIT_BONUS,
            "mcts_enabled": False,
            "rtp_enabled": False,
        }
    )
    difference_receipt_sha = canonical_sha256(
        {
            "schema": R212_RUNTIME_GRAPH_SCHEMA,
            "kind": "candidate_control_exact_difference",
            "candidate_runtime_graph_sha256": candidate_graph_sha,
            "control_runtime_graph_sha256": direct_graph_sha,
            "only_delta": "one_frozen_identity_verified_guide2vec_bounded_logit_bonus",
            "control_guide2vec_absent_not_disabled_or_zeroed": True,
        }
    )
    experiment = Guide2VecExperimentIdentity(
        base_runtime=base_runtime,
        guide2vec_checkpoint_sha256=candidate.checkpoint_sha256,
        guide2vec_training_receipt_sha256=candidate.training_receipt_sha256,
        guide2vec_runtime_config_sha256=candidate.runtime_config_sha256,
        guide2vec_parameter_count=candidate.parameter_count,
        candidate_runtime_graph_sha256=candidate_graph_sha,
        control_runtime_graph_sha256=direct_graph_sha,
        candidate_guide2vec_component_graph_sha256=candidate.component_graph_sha256,
        runtime_graph_difference_receipt_sha256=difference_receipt_sha,
        source_snapshot_sha256=candidate.source_snapshot_sha256,
        evaluation_output_identity_sha256=output_identity,
    )
    schedule = build_guide2vec_bo1000_schedule(seed_identity_sha256, experiment)
    plan: dict[str, object] = {
        "schema": R212_RUNTIME_PLAN_SCHEMA,
        "evaluation_id": GUIDE2VEC_EVALUATION_ID,
        "r212_contract_sha256": R212_CONTRACT_SHA256,
        "r195_contract_sha256": R195_CONTRACT_SHA256,
        "seed_identity_sha256": seed_identity_sha256,
        "experiment": experiment.as_payload(),
        "experiment_identity_sha256": experiment.identity_sha256,
        "evaluation_output_identity_sha256": output_identity,
        "artifacts": {
            "r195_bundle": {
                "path": str(artifacts.r195_bundle),
                "sha256": R195_BUNDLE_SHA256,
            },
            "r195_package_root": {
                "path": str(artifacts.r195_package_root),
                "manifest_sha256": package_manifest_sha,
            },
            "r195_checkpoint": {
                "path": str(artifacts.r195_checkpoint),
                "sha256": R195_CHECKPOINT_SHA256,
                "bytes": R195_CHECKPOINT_BYTES,
                "model_config_sha256": model_config_sha,
            },
            "guide2vec_checkpoint": {
                "path": str(artifacts.guide2vec_checkpoint),
                "sha256": candidate.checkpoint_sha256,
                "parameter_count": candidate.parameter_count,
                "runtime_config_sha256": candidate.runtime_config_sha256,
                "component_graph_sha256": candidate.component_graph_sha256,
            },
            "guide2vec_training_receipt": {
                "path": str(artifacts.guide2vec_training_receipt),
                "sha256": candidate.training_receipt_sha256,
                "source_snapshot_sha256": candidate.source_snapshot_sha256,
            },
            "matchup_adapter": {
                "format": R195_ADAPTER_FORMAT,
                "slot_capacity": R195_ADAPTER_SLOT_CAPACITY,
                "bank_sha256": adapter_bank_sha,
                "training_receipt_sha256": adapter_fit_sha,
                "runtime_graph_sha256": adapter_runtime_graph_sha,
                "runtime_enabled": True,
                "trained": True,
                "frozen": True,
                "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
            },
        },
        "schedule": [spec.as_payload() for spec in schedule],
        "schedule_sha256": canonical_sha256([spec.as_payload() for spec in schedule]),
        "authority": {
            "training_eligible": False,
            "mcts": False,
            "rtp": False,
            "serving": False,
            "selector": False,
            "kaggle": False,
            "promotion": False,
            "service_start": False,
        },
        "execution": {
            "native_seeded_pair_snapshot_required": True,
            "actual_first_second_attestation_required": True,
            "candidate_only_frozen_guide2vec": True,
            "control_guide2vec_module_parameter_state_hook_linear_counts_must_be_zero": True,
            "complete_all_games_without_early_stop": True,
        },
        "output_contract": {
            "content_addressed_subdirectory": f"r212-{output_identity[7:31]}",
            "create_once_files": [
                "PLAN.json",
                "OUTPUT_IDENTITY.json",
                "PREFLIGHT_RECEIPT.json",
                "pair-bindings/<pair-id>.json",
                "games/<game-nonce>/TRACE.json",
                "games/<game-nonce>/RECEIPT.json",
                "REPORT.json",
                "FINAL_RECEIPT.json",
            ],
            "evaluation_games_training_eligible": False,
        },
    }
    plan["canonical_sha256"] = canonical_sha256(plan)
    return plan


def _experiment_from_plan(plan: Mapping[str, object]) -> Guide2VecExperimentIdentity:
    raw = plan.get("experiment")
    if not isinstance(raw, Mapping):
        raise Guide2VecBO1000RuntimeError("r212 plan has no experiment")
    base_raw = raw.get("base_runtime")
    if not isinstance(base_raw, Mapping):
        raise Guide2VecBO1000RuntimeError("r212 plan base runtime is malformed")
    base_fields = {
        name: base_raw.get(name)
        for name in FrozenR195RuntimeIdentity.__dataclass_fields__
    }
    try:
        base = FrozenR195RuntimeIdentity(**base_fields)
        fields = {
            name: raw.get(name)
            for name in Guide2VecExperimentIdentity.__dataclass_fields__
            if name != "base_runtime"
        }
        experiment = Guide2VecExperimentIdentity(base_runtime=base, **fields)
    except (TypeError, Guide2VecBO1000Error) as exc:
        raise Guide2VecBO1000RuntimeError("r212 plan experiment is invalid") from exc
    if plan.get("experiment_identity_sha256") != experiment.identity_sha256:
        raise Guide2VecBO1000RuntimeError("r212 plan experiment digest drifted")
    return experiment


def _schedule_from_plan(plan: Mapping[str, object]) -> tuple[Guide2VecBO1000GameSpec, ...]:
    raw = plan.get("schedule")
    if not isinstance(raw, list):
        raise Guide2VecBO1000RuntimeError("r212 plan schedule is missing")
    try:
        schedule = tuple(Guide2VecBO1000GameSpec.from_payload(item) for item in raw)
    except Guide2VecBO1000Error as exc:
        raise Guide2VecBO1000RuntimeError("r212 plan schedule is invalid") from exc
    if len(schedule) != GUIDE2VEC_BO1000_GAME_COUNT:
        raise Guide2VecBO1000RuntimeError("r212 plan does not contain 1000 games")
    if plan.get("schedule_sha256") != canonical_sha256([spec.as_payload() for spec in schedule]):
        raise Guide2VecBO1000RuntimeError("r212 plan schedule digest drifted")
    return schedule


def verify_guide2vec_bo1000_plan(plan: Mapping[str, object]) -> tuple[
    Guide2VecExperimentIdentity, tuple[Guide2VecBO1000GameSpec, ...]
]:
    """Validate a plan created by :func:`build_guide2vec_bo1000_plan`."""

    if not isinstance(plan, Mapping) or plan.get("schema") != R212_RUNTIME_PLAN_SCHEMA:
        raise Guide2VecBO1000RuntimeError("not an r212 runtime plan")
    if plan.get("r212_contract_sha256") != R212_CONTRACT_SHA256:
        raise Guide2VecBO1000RuntimeError("r212 plan owner contract drifted")
    if plan.get("r195_contract_sha256") != R195_CONTRACT_SHA256:
        raise Guide2VecBO1000RuntimeError("r212 plan r195 contract drifted")
    claimed = plan.get("canonical_sha256")
    core = dict(plan)
    core.pop("canonical_sha256", None)
    if claimed != canonical_sha256(core):
        raise Guide2VecBO1000RuntimeError("r212 plan canonical digest drifted")
    experiment = _experiment_from_plan(plan)
    schedule = _schedule_from_plan(plan)
    if any(spec.experiment_identity_sha256 != experiment.identity_sha256 for spec in schedule):
        raise Guide2VecBO1000RuntimeError("schedule identity differs from experiment")
    return experiment, schedule


def _module_guide_name(name: object) -> bool:
    text = str(name).replace("-", "_").lower()
    return "guide2vec" in text or "guide_linear" in text or "guide_logit" in text


def _hook_guide_name(hook: object) -> bool:
    identity = " ".join(
        (
            type(hook).__module__,
            type(hook).__qualname__,
            getattr(hook, "__module__", ""),
            getattr(hook, "__qualname__", ""),
            repr(hook),
        )
    )
    return _module_guide_name(identity)


def _iter_object_values(
    value: object,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Iterable[tuple[str, object]]:
    """Small bounded object walk for policy-side attached Guide2Vec objects."""

    if depth > 3:
        return
    if seen is None:
        seen = set()
    if isinstance(value, (str, bytes, int, float, bool, type(None), Tensor, nn.Module)):
        return
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _iter_object_values(child, depth=depth + 1, seen=seen)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield str(index), child
            yield from _iter_object_values(child, depth=depth + 1, seen=seen)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            child = getattr(value, field.name)
            yield field.name, child
            yield from _iter_object_values(child, depth=depth + 1, seen=seen)
    elif hasattr(value, "__dict__"):
        attributes = getattr(value, "__dict__", {})
        if isinstance(attributes, Mapping):
            for key, child in attributes.items():
                yield str(key), child
                yield from _iter_object_values(child, depth=depth + 1, seen=seen)


def inspect_control_graph(*, model: nn.Module, policy: object) -> ControlGraphAudit:
    """Prove a direct arm has no Guide2Vec object, state, hook, or transform.

    This deliberately examines the control graph only.  A candidate head may
    exist elsewhere in the evaluator process, but it is not accepted as a
    disabled, zeroed, shared, or latent child of this object graph.
    """

    if not isinstance(model, nn.Module):
        raise Guide2VecBO1000RuntimeError("control model is not a torch module")
    module_names: list[str] = []
    parameter_names: list[str] = []
    state_names: list[str] = []
    hook_names: list[str] = []
    linear_names: list[str] = []
    for name, module in model.named_modules():
        qualified = f"model.{name}" if name else "model"
        if _module_guide_name(qualified) or _module_guide_name(type(module).__qualname__):
            module_names.append(qualified)
            if isinstance(module, nn.Linear):
                linear_names.append(qualified)
        for hook in tuple(getattr(module, "_forward_hooks", {}).values()):
            if _hook_guide_name(hook) or any(
                isinstance(value, Guide2VecHead)
                for _name, value in _iter_object_values(hook)
            ):
                hook_names.append(qualified)
        for hook in tuple(getattr(module, "_forward_pre_hooks", {}).values()):
            if _hook_guide_name(hook) or any(
                isinstance(value, Guide2VecHead)
                for _name, value in _iter_object_values(hook)
            ):
                hook_names.append(qualified)
    for name, _parameter in model.named_parameters(recurse=True):
        if _module_guide_name(name):
            parameter_names.append(name)
    for name in model.state_dict():
        if _module_guide_name(name):
            state_names.append(name)
    policy_values = getattr(policy, "__dict__", {})
    if isinstance(policy_values, Mapping):
        for name, value in _iter_object_values(policy_values):
            if (
                _module_guide_name(name)
                or _module_guide_name(type(value).__qualname__)
                or isinstance(value, Guide2VecHead)
            ):
                module_names.append(f"policy.{name}")
                if isinstance(value, nn.Linear):
                    linear_names.append(f"policy.{name}")
    if module_names or parameter_names or state_names or hook_names or linear_names:
        raise Guide2VecBO1000RuntimeError(
            "control graph contains forbidden Guide2Vec evidence: "
            f"modules={sorted(set(module_names))} parameters={sorted(set(parameter_names))} "
            f"state={sorted(set(state_names))} hooks={sorted(set(hook_names))} "
            f"linear={sorted(set(linear_names))}"
        )
    observation = {
        "schema": R212_RUNTIME_GRAPH_SCHEMA,
        "kind": "control_guide2vec_absence_observation",
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "policy_type": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "module_inventory_sha256": canonical_sha256(
            sorted(name for name, _ in model.named_modules())
        ),
        "parameter_inventory_sha256": canonical_sha256(
            sorted(name for name, _ in model.named_parameters())
        ),
        "state_inventory_sha256": canonical_sha256(sorted(model.state_dict())),
        "guide2vec_module_instance_count": 0,
        "guide2vec_parameter_count": 0,
        "guide2vec_state_dict_key_count": 0,
        "guide2vec_forward_hook_count": 0,
        "guide2vec_linear_transform_count": 0,
        "guide2vec_disabled_or_zeroed": False,
    }
    return ControlGraphAudit(
        observation_sha256=canonical_sha256(observation),
        module_instance_count=0,
        parameter_count=0,
        state_dict_key_count=0,
        forward_hook_count=0,
        linear_transform_count=0,
    )


def inspect_candidate_graph(
    *,
    head: Guide2VecHead,
    expected: CandidateArtifact,
) -> CandidateGraphAudit:
    if not isinstance(head, Guide2VecHead):
        raise Guide2VecBO1000RuntimeError("candidate Guide2Vec object has wrong type")
    parameters = sum(parameter.numel() for parameter in head.parameters())
    frozen = not head.training and not any(parameter.requires_grad for parameter in head.parameters())
    if parameters != expected.parameter_count or not frozen:
        raise Guide2VecBO1000RuntimeError("candidate Guide2Vec is not frozen/identity exact")
    if float(head.config.max_logit_bonus) != MAX_LOGIT_BONUS:
        raise Guide2VecBO1000RuntimeError("candidate Guide2Vec logit cap drifted")
    state = {name: tensor.detach().cpu() for name, tensor in head.state_dict().items()}
    graph = canonical_sha256(
        {
            "schema": R212_RUNTIME_GRAPH_SCHEMA,
            "kind": "frozen_guide2vec_component",
            "checkpoint_sha256": expected.checkpoint_sha256,
            "runtime_config_sha256": expected.runtime_config_sha256,
            "parameter_count": parameters,
            "state_dict_sha256": _tensor_mapping_sha256(state),
            "base_identity_sha256": expected.base_identity_sha256,
            "max_logit_bonus": MAX_LOGIT_BONUS,
            "frozen": True,
        }
    )
    if _tensor_mapping_sha256(state) != expected.state_dict_sha256:
        raise Guide2VecBO1000RuntimeError("candidate Guide2Vec state digest drifted")
    if graph != expected.component_graph_sha256:
        raise Guide2VecBO1000RuntimeError("candidate Guide2Vec component graph drifted")
    return CandidateGraphAudit(
        component_graph_sha256=graph,
        module_instance_count=1,
        parameter_count=parameters,
        frozen=True,
    )


def _is_no_rtp_policy(policy: object) -> bool:
    return (
        getattr(policy, "use_mcts", None) is False
        and getattr(policy, "belief_mcts", None) is False
        and getattr(policy, "use_recursive_turn_planner", None) is False
        and getattr(policy, "_rtp_bridge", None) is None
    )


def _assert_exact_r195_runtime(
    *,
    model: nn.Module,
    policy: object,
    expected_model_state_sha256: str,
    expected_adapter_bank_sha256: str,
    expected_adapter_config: Mapping[str, object],
) -> None:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise Guide2VecBO1000RuntimeError("r195 base model is not frozen eval-only")
    runtime_state = {
        name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
    }
    if _tensor_mapping_sha256(runtime_state) != expected_model_state_sha256:
        raise Guide2VecBO1000RuntimeError("loaded r195 runtime full model state drifted")
    if not _is_no_rtp_policy(policy):
        raise Guide2VecBO1000RuntimeError("r212 arm unexpectedly enables MCTS/RTP")
    if getattr(policy, "oracle_mode", None) is not False:
        raise Guide2VecBO1000RuntimeError("r212 direct arm cannot retain oracle authority")
    if getattr(policy, "sample_actions", None) is not False:
        raise Guide2VecBO1000RuntimeError("r212 direct arm cannot sample actions")
    if getattr(policy, "leaf_backend", None) is not None:
        raise Guide2VecBO1000RuntimeError("r212 direct arm cannot use a leaf backend")
    if getattr(policy, "collect_targets", None) is not False:
        raise Guide2VecBO1000RuntimeError("r212 mirror cannot collect training targets")
    for config_name, bridge_name in (
        ("poke_rlm_config", "_poke_rlm_bridge"),
        ("slowking_distill_config", "_slowking_distill_bridge"),
    ):
        if getattr(policy, config_name, None) is not None or getattr(policy, bridge_name, None) is not None:
            raise Guide2VecBO1000RuntimeError(
                f"r212 direct arm contains an unexpected optional policy sidecar: {config_name}"
            )
    if getattr(policy, "matchup_adapter_runtime", None) is not True:
        raise Guide2VecBO1000RuntimeError("r195 matchup adapter runtime is not on")
    tree_path = Path(str(getattr(policy, "matchup_adapter_tree_path", "")))
    if sha256_file(tree_path) != R195_MATCHUP_TREE_SHA256:
        raise Guide2VecBO1000RuntimeError("r195 runtime matchup tree drifted")
    bank = getattr(model, "matchup_adapter_bank", None)
    if not isinstance(bank, nn.Module) or getattr(bank, "enabled", None) is not True:
        raise Guide2VecBO1000RuntimeError("r195 matchup adapter bank is not active")
    if int(getattr(bank, "slot_capacity", -1)) != R195_ADAPTER_SLOT_CAPACITY:
        raise Guide2VecBO1000RuntimeError("r195 runtime V6 slot capacity drifted")
    experts = getattr(bank, "experts", ())
    if len(experts) != R195_ADAPTER_SLOT_CAPACITY:
        raise Guide2VecBO1000RuntimeError("r195 runtime V6 expert count drifted")
    config_fn = getattr(bank, "config_dict", None)
    if not callable(config_fn) or dict(config_fn()) != dict(expected_adapter_config):
        raise Guide2VecBO1000RuntimeError("r195 runtime adapter config drifted")
    state = {name: tensor.detach().cpu() for name, tensor in bank.state_dict().items()}
    if _tensor_mapping_sha256(state) != expected_adapter_bank_sha256:
        raise Guide2VecBO1000RuntimeError("r195 runtime adapter state drifted")
    if any(parameter.requires_grad for parameter in bank.parameters()):
        raise Guide2VecBO1000RuntimeError("r195 runtime adapter is not frozen")


def _isolate_archived_package(package_root: Path) -> None:
    """Make runtime imports resolve only to the exact archived r195 package."""

    package_text = str(package_root)
    # The package-file manifest is part of the exact r195 identity.  Do not
    # create local bytecode children that would make a later preflight see a
    # mutable package tree rather than the archived submission.
    sys.dont_write_bytecode = True
    os.chdir(package_text)
    os.environ["CG_LIB_PATH"] = package_text
    os.environ["POKEBOT_USE_RECURSIVE_TURN_PLANNER"] = "0"
    os.environ["POKEBOT_COMBO_STATE_ROUTE_ENABLED"] = "0"
    for name in tuple(os.environ):
        if name.startswith("POKEBOT_RTP_"):
            os.environ.pop(name, None)
    # Never let a host guide experiment bleed into an archived direct arm.
    for name in tuple(os.environ):
        if "GUIDE2VEC" in name or "GUIDE_LOGIT" in name or "GUIDE_LINEAR" in name:
            os.environ.pop(name, None)
    sys.path[:] = [package_text, *[entry for entry in sys.path if entry not in {"", package_text}]]
    for name in list(sys.modules):
        if name == "poke_bot" or name.startswith("poke_bot.") or name == "cg" or name.startswith("cg."):
            del sys.modules[name]


def _load_archived_submission(package_root: Path) -> Any:
    _isolate_archived_package(package_root)
    spec = importlib.util.spec_from_file_location(
        "r212_exact_r195_submission", package_root / "main.py"
    )
    if spec is None or spec.loader is None:
        raise Guide2VecBO1000RuntimeError("cannot load exact r195 submission main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    ensure = getattr(module, "_ensure_runtime", None)
    if not callable(ensure):
        raise Guide2VecBO1000RuntimeError("r195 package has no runtime loader")
    return module


def _load_frozen_head(
    *,
    checkpoint_path: Path,
    candidate: CandidateArtifact,
    device: torch.device,
) -> Guide2VecHead:
    if sha256_file(checkpoint_path) != candidate.checkpoint_sha256:
        raise Guide2VecBO1000RuntimeError("Guide2Vec checkpoint changed after verification")
    payload = _safe_torch_load(checkpoint_path)
    expected_base = FrozenBaseIdentity(
        submission_id=R195_SUBMISSION_ID,
        checkpoint_sha256=R195_CHECKPOINT_SHA256,
        checkpoint_bytes=R195_CHECKPOINT_BYTES,
        bundle_sha256=R195_BUNDLE_SHA256,
        model_config_sha256=candidate.model_config_sha256,
        feature_schema_sha256=candidate.feature_schema_sha256,
    )
    try:
        head, identity, _metadata = load_checkpoint_payload(
            payload,
            expected_base_identity=expected_base,
            map_location=device,
        )
    except Guide2VecError as exc:
        raise Guide2VecBO1000RuntimeError("cannot load frozen Guide2Vec checkpoint") from exc
    if identity.identity_sha256 != candidate.base_identity_sha256:
        raise Guide2VecBO1000RuntimeError("loaded Guide2Vec base identity drifted")
    if head.parameter_count != candidate.parameter_count:
        raise Guide2VecBO1000RuntimeError("loaded Guide2Vec parameter count drifted")
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    inspect_candidate_graph(head=head, expected=candidate)
    return head


def _make_candidate_policy(
    *,
    model: nn.Module,
    deck: Sequence[int],
    control_policy: object,
) -> object:
    """Create a second direct r195 policy state machine for the candidate seat."""

    from poke_bot.agent import PolicyAgent

    tree_path = Path(str(getattr(control_policy, "matchup_adapter_tree_path", "")))
    if not tree_path.is_file() or sha256_file(tree_path) != R195_MATCHUP_TREE_SHA256:
        raise Guide2VecBO1000RuntimeError("candidate requires the exact r195 V6 tree")
    policy = PolicyAgent(
        model=model,
        deck=list(deck),
        use_mcts=False,
        use_recursive_turn_planner=False,
        belief_mcts=False,
        matchup_adapter_runtime=True,
        matchup_adapter_tree_path=str(tree_path),
        checkpoint_digest=R195_CHECKPOINT_SHA256,
        sample_actions=False,
        action_temperature=float(getattr(control_policy, "action_temperature", 1.0)),
        oracle_mode=False,
        collect_targets=False,
        strict_runtime=True,
        max_context_override=getattr(control_policy, "max_context_override", None),
        model_generation=getattr(control_policy, "model_generation", 0),
        matchup_adapter_shadow=bool(
            getattr(control_policy, "matchup_adapter_shadow", True)
        ),
    )
    return policy


def _assert_direct_policy_parity(*, control_policy: object, candidate_policy: object, model: nn.Module) -> str:
    """Bind every selection-relevant non-Guide2Vec direct-policy setting."""

    fields = (
        "use_mcts",
        "belief_mcts",
        "use_recursive_turn_planner",
        "oracle_mode",
        "sample_actions",
        "action_temperature",
        "max_context_override",
        "model_generation",
        "matchup_adapter_shadow",
        "matchup_adapter_runtime",
        "matchup_adapter_tree_path",
        "strict_runtime",
    )
    values: dict[str, object] = {}
    for field in fields:
        control_value = getattr(control_policy, field, None)
        candidate_value = getattr(candidate_policy, field, None)
        if field == "matchup_adapter_tree_path":
            control_tree = sha256_file(Path(str(control_value)))
            candidate_tree = sha256_file(Path(str(candidate_value)))
            if control_tree != candidate_tree:
                raise Guide2VecBO1000RuntimeError(
                    "candidate/control direct policy tree path drifted"
                )
            values[field] = control_tree
            continue
        if control_value != candidate_value:
            raise Guide2VecBO1000RuntimeError(
                f"candidate/control direct policy setting drifted: {field}"
            )
        values[field] = control_value
    if getattr(control_policy, "model", None) is not model or getattr(candidate_policy, "model", None) is not model:
        raise Guide2VecBO1000RuntimeError("candidate/control do not share the exact loaded r195 model")
    if list(getattr(control_policy, "deck", ())) != list(getattr(candidate_policy, "deck", ())):
        raise Guide2VecBO1000RuntimeError("candidate/control direct deck order drifted")
    if list(getattr(control_policy, "deck", ())) == []:
        raise Guide2VecBO1000RuntimeError("candidate/control direct deck is empty")
    if canonical_sha256([int(card) for card in getattr(control_policy, "deck", ())]) != R195_DECK_CARDS_SHA256:
        raise Guide2VecBO1000RuntimeError("candidate/control runtime deck differs from exact r195 deck")
    for policy in (control_policy, candidate_policy):
        if getattr(policy, "leaf_backend", None) is not None:
            raise Guide2VecBO1000RuntimeError("r212 direct policy cannot use a leaf backend")
        if getattr(policy, "collect_targets", None) is not False:
            raise Guide2VecBO1000RuntimeError("r212 evaluation cannot collect training targets")
    return canonical_sha256(
        {
            "schema": R212_RUNTIME_GRAPH_SCHEMA,
            "kind": "candidate_control_direct_policy_parity",
            "shared_model": True,
            "deck_cards_sha256": R195_DECK_CARDS_SHA256,
            "settings": values,
        }
    )


@dataclass(slots=True)
class _CandidateOverlay:
    """One candidate-only bounded per-stage Guide2Vec policy wrapper."""

    policy: Any
    head: Guide2VecHead
    expected_base_identity: FrozenBaseIdentity
    decision_rows: list[Guide2VecDecisionReceipt]
    trace_rows: list[dict[str, object]]
    current_game_nonce_sha256: str | None = None
    current_acting_seat: int | None = None

    def reset_game(self, *, game_nonce_sha256: str, acting_seat: int) -> None:
        _require_digest(game_nonce_sha256, label="candidate game nonce")
        if acting_seat not in {0, 1}:
            raise Guide2VecBO1000RuntimeError("candidate acting seat is invalid")
        self.policy.reset_game()
        self.policy.strict_runtime = True
        self.decision_rows.clear()
        self.trace_rows.clear()
        self.current_game_nonce_sha256 = game_nonce_sha256
        self.current_acting_seat = acting_seat

    @staticmethod
    def _action_digest(action: Sequence[int]) -> str:
        return canonical_sha256([int(value) for value in action])

    @staticmethod
    def _tensor_digest(tensor: Tensor) -> str:
        value = tensor.detach().cpu().contiguous()
        return canonical_sha256(
            {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "bytes_sha256": "sha256:" + hashlib.sha256(
                    value.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
        )

    @staticmethod
    def _validate_action(observation: Mapping[str, object], action: Sequence[int]) -> list[int]:
        selection = observation.get("select")
        if not isinstance(selection, Mapping):
            raise Guide2VecBO1000RuntimeError("native engine requested action without select")
        options = selection.get("option")
        if not isinstance(options, list):
            raise Guide2VecBO1000RuntimeError("native select has no ordered options")
        lower = int(selection.get("minCount", 0) or 0)
        upper = min(int(selection.get("maxCount", 0) or 0), len(options))
        result = [int(value) for value in action]
        if not lower <= len(result) <= upper:
            raise Guide2VecBO1000RuntimeError("candidate action count is illegal")
        if len(set(result)) != len(result) or any(value < 0 or value >= len(options) for value in result):
            raise Guide2VecBO1000RuntimeError("candidate action index is illegal")
        return result

    def _observe_route(self, observation: dict[str, object]) -> None:
        if not bool(getattr(self.policy, "matchup_adapter_runtime", False)):
            raise Guide2VecBO1000RuntimeError("candidate adapter runtime is not active")
        router = getattr(self.policy, "_matchup_adapter_shadow_router", None)
        observe = getattr(router, "observe", None)
        if not callable(observe):
            raise Guide2VecBO1000RuntimeError("candidate adapter router is absent")
        observe(observation, scope="game_root", depth=len(self.policy.board_history))

    @torch.no_grad()
    def select(self, observation: Mapping[str, object]) -> list[int]:
        """Run the exact r195 direct decode, adding only the bounded overlay."""

        if self.current_game_nonce_sha256 is None or self.current_acting_seat not in {0, 1}:
            raise Guide2VecBO1000RuntimeError("candidate overlay was not bound to a game")
        obs = dict(observation)
        from poke_bot import features
        from poke_bot.agent import forced_go_first_action

        # Match ``PolicyAgent.__call__``: the public V6 route is observed
        # before the package's deterministic setup shortcut is considered.
        self._observe_route(obs)
        forced = forced_go_first_action(obs)
        if forced is not None:
            # Setup policy is a package-defined deterministic short circuit and
            # has no neural or Guide2Vec action authority.
            return self._validate_action(obs, self.policy._record_go_first(obs, list(forced)))
        board = self.policy._append_decision_history(obs)
        model = self.policy.model
        if not isinstance(model, nn.Module):
            raise Guide2VecBO1000RuntimeError("candidate direct policy has no model")
        prefix: list[int] = []
        cached_state: Tensor | None = None
        cached_spatial: Tensor | None = None
        cached_fusion_state: Tensor | None = None
        while True:
            candidates = features.factorized_action_candidates(obs, prefix)
            if len(candidates) == 1 and candidates[0] == prefix:
                selected = list(prefix)
                break
            if not candidates:
                raise Guide2VecBO1000RuntimeError("candidate stage has no legal options")
            started = time.monotonic()
            options = features.build_option_tokens(obs, candidates)
            route = int(self.policy._matchup_model_route())
            first_decode = (
                cached_state is None
                or cached_spatial is None
                or cached_fusion_state is None
            )
            if first_decode:
                if model.decision_context == "history" and model.kv_cache_enabled:
                    model_out = model.forward(
                        board,
                        options,
                        kv_cache=self.policy._kv_cache,
                        append_cache=True,
                        n_options=[len(candidates)],
                        previous_action=self.policy._previous_action_token,
                        matchup_routes=[route],
                    )
                    self.policy._kv_cache = model_out["kv_cache"]
                elif model.decision_context == "history":
                    model_out = model.forward_history_batch(
                        [self.policy.board_history],
                        [options],
                        n_options=[len(candidates)],
                        previous_action_histories=[self.policy.previous_action_history],
                        matchup_routes=[route],
                    )
                else:
                    model_out = model.forward(
                        board,
                        options,
                        append_cache=False,
                        n_options=[len(candidates)],
                        matchup_routes=[route],
                    )
                raw_state = model_out["state_vec"]
                spatial = model_out["spatial_memory"]
                if not isinstance(raw_state, Tensor) or not isinstance(spatial, Tensor):
                    raise Guide2VecBO1000RuntimeError("r195 decoder state is malformed")
                cached_fusion_state = raw_state
                cached_state = model.matchup_policy_value_state(raw_state, [route])
                cached_spatial = spatial
            decoded = model.decode_options(
                options,
                cached_spatial,
                cached_state,
                n_options=[len(candidates)],
                return_hidden=True,
                decision_fusion_state_vec=cached_fusion_state,
            )
            if not isinstance(decoded, tuple) or len(decoded) != 2:
                raise Guide2VecBO1000RuntimeError("r195 option decoder did not return hidden state")
            decoded_logits, option_hidden = decoded
            if not isinstance(decoded_logits, Tensor) or not isinstance(option_hidden, Tensor):
                raise Guide2VecBO1000RuntimeError("r195 decoder tensors are malformed")
            if first_decode:
                raw_logits = model_out.get("policy_logits")
                if not isinstance(raw_logits, Tensor):
                    raise Guide2VecBO1000RuntimeError("r195 forward policy logits are malformed")
                base_logits = raw_logits
                # The second decode exists solely to obtain the frozen legal
                # option hidden states for Guide2Vec.  It must reproduce the
                # exact package forward logits, otherwise the candidate would
                # no longer be an additive bounded overlay on r195 direct.
                if not torch.equal(
                    decoded_logits[:, : len(candidates)],
                    base_logits[:, : len(candidates)],
                ):
                    raise Guide2VecBO1000RuntimeError(
                        "r195 option-hidden decode does not reproduce direct logits"
                    )
            else:
                base_logits = decoded_logits
            parameter = next(self.head.parameters())
            sidecar_state = cached_fusion_state.to(device=parameter.device, dtype=parameter.dtype)
            sidecar_hidden = option_hidden.to(device=parameter.device, dtype=parameter.dtype)
            sidecar_logits = base_logits.to(device=parameter.device, dtype=parameter.dtype)
            guide_started = time.monotonic()
            result: Guide2VecDecision = self.head.rerank(
                sidecar_state,
                sidecar_hidden,
                sidecar_logits,
                n_options=len(candidates),
                expected_base_identity=self.expected_base_identity,
                observed_base_identity=self.expected_base_identity,
            )
            guide_elapsed = time.monotonic() - guide_started
            applied = bool(result.applied[0].item())
            base_index = int(result.base_indices[0].item())
            selected_index = int(result.selected_indices[0].item())
            direct_action = list(candidates[base_index])
            selected = list(candidates[selected_index])
            total_elapsed = time.monotonic() - started
            legal_scores = result.guide_scores[0, : len(candidates)]
            bonus = result.bonus[0, : len(candidates)]
            observed_max_bonus = float(bonus.max().item()) if applied else 0.0
            if observed_max_bonus > MAX_LOGIT_BONUS + 1e-6:
                raise Guide2VecBO1000RuntimeError(
                    "Guide2Vec produced a bonus above the hard 0.05 cap"
                )
            # Float32/bfloat16 represent the decimal cap a few ULPs above
            # Python's literal ``0.05``.  Receipts bind the contractual cap,
            # while the trace retains the normalized logits/scores that made
            # the selection reproducible.
            max_bonus = min(MAX_LOGIT_BONUS, observed_max_bonus)
            decision_index = len(self.decision_rows)
            receipt = Guide2VecDecisionReceipt(
                game_nonce_sha256=self.current_game_nonce_sha256,
                decision_index=decision_index,
                acting_seat=self.current_acting_seat,
                legal_option_count=len(candidates),
                eligible=applied,
                abstained=not applied,
                bonus_applied=applied,
                action_changed_from_direct_policy=(direct_action != selected),
                max_applied_logit_bonus=max_bonus,
                direct_action_sha256=self._action_digest(direct_action),
                final_action_sha256=self._action_digest(selected),
                legal_options_sha256=canonical_sha256(candidates),
                guide2vec_scores_sha256=self._tensor_digest(legal_scores),
                guide2vec_action_latency_seconds=max(0.0, guide_elapsed),
                total_action_latency_seconds=max(guide_elapsed, total_elapsed),
            )
            # The reusable overlay is explicitly rebound to this nonce/seat
            # before every game; a receipt cannot be retrofitted afterward.
            self.decision_rows.append(receipt)
            self.trace_rows.append(
                {
                    "decision_index": decision_index,
                    "legal_options": [list(value) for value in candidates],
                    "direct_action": direct_action,
                    "final_action": selected,
                    "eligible": applied,
                    "abstained": not applied,
                    "bonus_applied": applied,
                    "reason": result.reasons[0],
                    "max_applied_logit_bonus": max_bonus,
                    "guide_scores_sha256": receipt.guide2vec_scores_sha256,
                    "base_logits_sha256": self._tensor_digest(
                        result.base_logits[0, : len(candidates)]
                    ),
                    "adjusted_logits_sha256": self._tensor_digest(
                        result.adjusted_logits[0, : len(candidates)]
                    ),
                    "matchup_adapter_route": self.policy.matchup_adapter_shadow_snapshot(),
                }
            )
            if selected == prefix:
                break
            prefix = selected
        self.policy._previous_action_token = features.build_option_tokens(obs, [selected])
        return self._validate_action(obs, selected)


def _bound_candidate_decisions(
    overlay: _CandidateOverlay,
    *,
    game_nonce_sha256: str,
    seat: int,
) -> tuple[Guide2VecDecisionReceipt, ...]:
    if overlay.current_game_nonce_sha256 != game_nonce_sha256 or overlay.current_acting_seat != seat:
        raise Guide2VecBO1000RuntimeError("candidate decision rows are bound to another game")
    if any(
        row.game_nonce_sha256 != game_nonce_sha256 or row.acting_seat != seat
        for row in overlay.decision_rows
    ):
        raise Guide2VecBO1000RuntimeError("candidate decision receipt identity drifted")
    return tuple(overlay.decision_rows)


@dataclass(frozen=True, slots=True)
class NativePairBinding:
    """One native seeded-start proof tied to an immutable r212 pair."""

    r212_pair_id: str
    r212_pair_index: int
    r212_pair_nonce_sha256: str
    r212_pair_initial_rng_sha256: str
    r212_pair_deck_order_rng_sha256: str
    sealed_initial_first_actor_seat: int
    seed_attempt: int
    native_seed_identity_sha256: str
    native_games: tuple[SeededMirrorGameSpec, SeededMirrorGameSpec]
    native_first_player_seal: PairFirstPlayerSeal
    setup_actions_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.r212_pair_id, str) or not self.r212_pair_id:
            raise Guide2VecBO1000RuntimeError("r212 pair id is invalid")
        _require_digest(self.r212_pair_nonce_sha256, label="r212 pair nonce")
        _require_digest(self.r212_pair_initial_rng_sha256, label="r212 initial RNG")
        _require_digest(self.r212_pair_deck_order_rng_sha256, label="r212 deck RNG")
        _require_exact_int(self.r212_pair_index, label="r212 pair index")
        _require_exact_int(self.seed_attempt, label="native seed attempt")
        _require_digest(self.native_seed_identity_sha256, label="native seed identity")
        if self.sealed_initial_first_actor_seat not in {0, 1}:
            raise Guide2VecBO1000RuntimeError("sealed first actor seat is invalid")
        if len(self.native_games) != 2:
            raise Guide2VecBO1000RuntimeError("native pair needs two game specs")
        try:
            validate_pair_first_player_seal(self.native_games, self.native_first_player_seal)
        except SeededMirrorHarnessError as exc:
            raise Guide2VecBO1000RuntimeError("native pair seal is invalid") from exc
        if self.native_first_player_seal.first_player_seat != self.sealed_initial_first_actor_seat:
            raise Guide2VecBO1000RuntimeError(
                "native first player does not match r212 sealed pair material"
            )
        if [game.experimental_seat for game in self.native_games] != [0, 1]:
            raise Guide2VecBO1000RuntimeError("native schedule does not seat-swap candidate")
        try:
            expected_native_games = build_seeded_seat_swapped_schedule(
                evaluation_id=f"{GUIDE2VEC_EVALUATION_ID}-native-seed",
                seed_identity_sha256=self.native_seed_identity_sha256,
                pair_count=1,
            )
        except SeededMirrorHarnessError as exc:
            raise Guide2VecBO1000RuntimeError("native seed identity is invalid") from exc
        if tuple(expected_native_games) != self.native_games:
            raise Guide2VecBO1000RuntimeError("native pair schedule drifted from its seed identity")
        _require_digest(self.setup_actions_sha256, label="native setup action digest")

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema": R212_NATIVE_PAIR_BINDING_SCHEMA,
                **self.as_payload(include_identity=False),
            }
        )

    def as_payload(self, *, include_identity: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "r212_pair_id": self.r212_pair_id,
            "r212_pair_index": self.r212_pair_index,
            "r212_pair_nonce_sha256": self.r212_pair_nonce_sha256,
            "r212_pair_initial_rng_sha256": self.r212_pair_initial_rng_sha256,
            "r212_pair_deck_order_rng_sha256": self.r212_pair_deck_order_rng_sha256,
            "sealed_initial_first_actor_seat": self.sealed_initial_first_actor_seat,
            "seed_attempt": self.seed_attempt,
            "native_seed_identity_sha256": self.native_seed_identity_sha256,
            "native_games": [game.as_payload() for game in self.native_games],
            "native_first_player_seal": self.native_first_player_seal.as_payload(),
            "setup_actions_sha256": self.setup_actions_sha256,
        }
        if include_identity:
            payload["schema"] = R212_NATIVE_PAIR_BINDING_SCHEMA
            payload["identity_sha256"] = self.identity_sha256
        return payload

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "NativePairBinding":
        expected_fields = {
            "schema",
            "identity_sha256",
            "r212_pair_id",
            "r212_pair_index",
            "r212_pair_nonce_sha256",
            "r212_pair_initial_rng_sha256",
            "r212_pair_deck_order_rng_sha256",
            "sealed_initial_first_actor_seat",
            "seed_attempt",
            "native_seed_identity_sha256",
            "native_games",
            "native_first_player_seal",
            "setup_actions_sha256",
        }
        if set(raw) != expected_fields:
            raise Guide2VecBO1000RuntimeError("native pair binding fields drifted")
        if raw.get("schema") != R212_NATIVE_PAIR_BINDING_SCHEMA:
            raise Guide2VecBO1000RuntimeError("native pair binding schema drifted")
        games_raw = raw.get("native_games")
        seal_raw = raw.get("native_first_player_seal")
        if not isinstance(games_raw, list) or len(games_raw) != 2 or not isinstance(seal_raw, Mapping):
            raise Guide2VecBO1000RuntimeError("native pair binding is malformed")
        try:
            games = tuple(SeededMirrorGameSpec(**dict(item)) for item in games_raw)
            seal_payload = dict(seal_raw)
            claimed_seal_identity = seal_payload.pop("identity_sha256", None)
            seal = PairFirstPlayerSeal(**seal_payload)
        except (TypeError, SeededMirrorHarnessError) as exc:
            raise Guide2VecBO1000RuntimeError("native pair binding payload is invalid") from exc
        if claimed_seal_identity != seal.identity_sha256:
            raise Guide2VecBO1000RuntimeError("native pair first-player seal digest drifted")
        binding = cls(
            r212_pair_id=str(raw.get("r212_pair_id") or ""),
            r212_pair_index=raw.get("r212_pair_index"),  # type: ignore[arg-type]
            r212_pair_nonce_sha256=raw.get("r212_pair_nonce_sha256"),  # type: ignore[arg-type]
            r212_pair_initial_rng_sha256=raw.get("r212_pair_initial_rng_sha256"),  # type: ignore[arg-type]
            r212_pair_deck_order_rng_sha256=raw.get("r212_pair_deck_order_rng_sha256"),  # type: ignore[arg-type]
            sealed_initial_first_actor_seat=raw.get("sealed_initial_first_actor_seat"),  # type: ignore[arg-type]
            seed_attempt=raw.get("seed_attempt"),  # type: ignore[arg-type]
            native_seed_identity_sha256=raw.get("native_seed_identity_sha256"),  # type: ignore[arg-type]
            native_games=(games[0], games[1]),
            native_first_player_seal=seal,
            setup_actions_sha256=raw.get("setup_actions_sha256"),  # type: ignore[arg-type]
        )
        if raw.get("identity_sha256") != binding.identity_sha256:
            raise Guide2VecBO1000RuntimeError("native pair binding digest drifted")
        return binding


def _new_seeded_environment() -> Any:
    from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv
    from cg import sim

    environment = LibcgMultiEnv(1)
    configure_battle_start_seeded(environment._lib, sim.StartData)
    return environment


def _reset_seeded_game(environment: Any, deck: Sequence[int], seed: int) -> Any:
    from poke_bot.engine_rebuild.interfaces import ResetSpec

    return environment.reset(
        [ResetSpec(deck0=list(deck), deck1=list(deck), seed=int(seed))]
    ).envs[0]


def _first_player(observation: Mapping[str, object]) -> int | None:
    current = observation.get("current")
    if not isinstance(current, Mapping):
        return None
    first = current.get("firstPlayer")
    return int(first) if type(first) is int and first in {0, 1} else None


def _package_turn_order_action(module: Any, observation: Mapping[str, object]) -> list[int] | None:
    choice_fn = getattr(module, "_turn_order_choice", None)
    fail_closed = getattr(module, "_fail_closed", None)
    if not callable(choice_fn) or not callable(fail_closed):
        raise Guide2VecBO1000RuntimeError("r195 package lacks exact turn-order path")
    choice = choice_fn(dict(observation))
    if choice is None:
        return None
    action = list(fail_closed(dict(observation), list(choice)))
    if action != list(choice):
        raise Guide2VecBO1000RuntimeError("r195 packaged turn-order action drifted")
    return _CandidateOverlay._validate_action(observation, action)


def _capture_native_pair_seal(
    *,
    module: Any,
    deck: Sequence[int],
    r212_specs: Sequence[Guide2VecBO1000GameSpec],
    seed_identity_sha256: str,
    max_attempts: int,
) -> NativePairBinding:
    """Find a deterministic native seeded start matching the sealed r212 order.

    The fixed compiler schedule commits an explicit first-actor bit per pair.
    We never infer that bit from the candidate seat.  Instead, independent
    deterministic seed candidates are materialized through libcg until its
    actual observed first actor matches the already sealed bit; the selected
    native seed and setup transcript are then preserved for both mirror games.
    """

    if len(r212_specs) != 2:
        raise Guide2VecBO1000RuntimeError("native pair binding needs two r212 specs")
    first, second = sorted(r212_specs, key=lambda spec: spec.game_index)
    if first.pair_id != second.pair_id or first.pair_index != second.pair_index:
        raise Guide2VecBO1000RuntimeError("r212 pair specs do not agree")
    if max_attempts < 1:
        raise Guide2VecBO1000RuntimeError("native pair seed attempts must be positive")
    for attempt in range(max_attempts):
        native_seed_identity = canonical_sha256(
            {
                "schema": R212_NATIVE_PAIR_BINDING_SCHEMA,
                "r212_pair_nonce_sha256": first.pair_nonce_sha256,
                "r212_pair_initial_rng_sha256": first.pair_initial_rng_sha256,
                "r212_pair_deck_order_rng_sha256": first.pair_deck_order_rng_sha256,
                "seed_identity_sha256": seed_identity_sha256,
                "attempt": attempt,
            }
        )
        native_games = build_seeded_seat_swapped_schedule(
            evaluation_id=f"{GUIDE2VEC_EVALUATION_ID}-native-seed",
            seed_identity_sha256=native_seed_identity,
            pair_count=1,
        )
        native_pair = (native_games[0], native_games[1])
        environment = _new_seeded_environment()
        setup_actions: list[list[int]] = []
        try:
            state = _reset_seeded_game(environment, deck, native_pair[0].engine_seed_u32)
            for _ in range(8):
                observed_first = _first_player(state.obs)
                if observed_first is not None:
                    seal = PairFirstPlayerSeal(
                        evaluation_id=native_pair[0].evaluation_id,
                        pair_index=native_pair[0].pair_index,
                        pair_id=native_pair[0].pair_id,
                        pair_nonce_sha256=native_pair[0].pair_nonce_sha256,
                        engine_seed_u32=native_pair[0].engine_seed_u32,
                        deck_order_seed_u32=native_pair[0].deck_order_seed_u32,
                        first_player_seat=observed_first,
                        post_turn_order_observation_sha256=_native_sha256(state.obs),
                    )
                    if observed_first != first.sealed_initial_first_actor_seat:
                        break
                    return NativePairBinding(
                        r212_pair_id=first.pair_id,
                        r212_pair_index=first.pair_index,
                        r212_pair_nonce_sha256=first.pair_nonce_sha256,
                        r212_pair_initial_rng_sha256=first.pair_initial_rng_sha256,
                        r212_pair_deck_order_rng_sha256=first.pair_deck_order_rng_sha256,
                        sealed_initial_first_actor_seat=first.sealed_initial_first_actor_seat,
                        seed_attempt=attempt,
                        native_seed_identity_sha256=native_seed_identity,
                        native_games=native_pair,
                        native_first_player_seal=seal,
                        setup_actions_sha256=canonical_sha256(setup_actions),
                    )
                action = _package_turn_order_action(module, state.obs)
                if action is None:
                    raise Guide2VecBO1000RuntimeError(
                        "native start did not expose package turn-order action"
                    )
                setup_actions.append(action)
                state = environment.step_batch([action]).envs[0]
            else:
                raise Guide2VecBO1000RuntimeError("native start never sealed first player")
        finally:
            environment.close()
    raise Guide2VecBO1000RuntimeError(
        f"unable to obtain native pair seal after {max_attempts} deterministic attempts"
    )


def _binding_matches_specs(
    binding: NativePairBinding,
    specs: Sequence[Guide2VecBO1000GameSpec],
    *,
    seed_identity_sha256: str,
) -> None:
    if len(specs) != 2:
        raise Guide2VecBO1000RuntimeError("binding requires two r212 specs")
    first, second = sorted(specs, key=lambda spec: spec.game_index)
    if (
        binding.r212_pair_id != first.pair_id
        or binding.r212_pair_index != first.pair_index
        or binding.r212_pair_nonce_sha256 != first.pair_nonce_sha256
        or binding.r212_pair_initial_rng_sha256 != first.pair_initial_rng_sha256
        or binding.r212_pair_deck_order_rng_sha256 != first.pair_deck_order_rng_sha256
        or binding.sealed_initial_first_actor_seat != first.sealed_initial_first_actor_seat
    ):
        raise Guide2VecBO1000RuntimeError("native binding does not bind r212 pair")
    if [game.experimental_seat for game in binding.native_games] != [
        first.guide2vec_seat,
        second.guide2vec_seat,
    ]:
        raise Guide2VecBO1000RuntimeError("native binding candidate seat swap drifted")
    expected_native_seed = canonical_sha256(
        {
            "schema": R212_NATIVE_PAIR_BINDING_SCHEMA,
            "r212_pair_nonce_sha256": first.pair_nonce_sha256,
            "r212_pair_initial_rng_sha256": first.pair_initial_rng_sha256,
            "r212_pair_deck_order_rng_sha256": first.pair_deck_order_rng_sha256,
            "seed_identity_sha256": seed_identity_sha256,
            "attempt": binding.seed_attempt,
        }
    )
    if binding.native_seed_identity_sha256 != expected_native_seed:
        raise Guide2VecBO1000RuntimeError("native binding seed identity is not sealed from r212 pair")


def _runtime_preflight(
    *,
    model: nn.Module,
    control_policy: object,
    candidate_policy: object,
    candidate_head: Guide2VecHead,
    plan: Mapping[str, object],
    candidate: CandidateArtifact,
    model_state_sha256: str,
    adapter_bank_sha: str,
    adapter_config: Mapping[str, object],
) -> dict[str, object]:
    experiment, _schedule = verify_guide2vec_bo1000_plan(plan)
    _assert_exact_r195_runtime(
        model=model,
        policy=control_policy,
        expected_model_state_sha256=model_state_sha256,
        expected_adapter_bank_sha256=adapter_bank_sha,
        expected_adapter_config=adapter_config,
    )
    direct_policy_parity_sha = _assert_direct_policy_parity(
        control_policy=control_policy,
        candidate_policy=candidate_policy,
        model=model,
    )
    _assert_exact_r195_runtime(
        model=model,
        policy=candidate_policy,
        expected_model_state_sha256=model_state_sha256,
        expected_adapter_bank_sha256=adapter_bank_sha,
        expected_adapter_config=adapter_config,
    )
    control = inspect_control_graph(model=model, policy=control_policy)
    # The candidate policy state machine starts as the exact same direct r195
    # graph.  Audit it before the external overlay is constructed so no hidden
    # second Guide2Vec object can be inherited from a policy/environment path.
    candidate_base = inspect_control_graph(model=model, policy=candidate_policy)
    candidate_graph = inspect_candidate_graph(head=candidate_head, expected=candidate)
    if candidate_graph.component_graph_sha256 != experiment.candidate_guide2vec_component_graph_sha256:
        raise Guide2VecBO1000RuntimeError("candidate component graph differs from plan")
    return {
        "schema": R212_RUNTIME_PREFLIGHT_SCHEMA,
        "evaluation_id": GUIDE2VEC_EVALUATION_ID,
        "r212_contract_sha256": R212_CONTRACT_SHA256,
        "r195_contract_sha256": R195_CONTRACT_SHA256,
        "experiment_identity_sha256": experiment.identity_sha256,
        "evaluation_output_identity_sha256": experiment.evaluation_output_identity_sha256,
        "r195_submission_id": R195_SUBMISSION_ID,
        "r195_full_model_state_sha256": model_state_sha256,
        "candidate_control_direct_policy_parity_sha256": direct_policy_parity_sha,
        "r195_no_rtp": True,
        "mcts": False,
        "rtp": False,
        "matchup_adapter": {
            "enabled": True,
            "trained": True,
            "frozen": True,
            "format": R195_ADAPTER_FORMAT,
            "slot_capacity": R195_ADAPTER_SLOT_CAPACITY,
            "tree_sha256": R195_MATCHUP_TREE_SHA256,
            "bank_sha256": adapter_bank_sha,
        },
        "candidate_graph": candidate_graph.as_payload(),
        "candidate_base_graph_without_sidecar": candidate_base.as_payload(),
        "control_graph": control.as_payload(),
        "authority": {
            "training": False,
            "serving": False,
            "selector": False,
            "kaggle": False,
            "promotion": False,
            "service_start": False,
        },
    }


def _game_receipt(
    *,
    spec: Guide2VecBO1000GameSpec,
    experiment: Guide2VecExperimentIdentity,
    control_audit: ControlGraphAudit,
    observed_first_actor_seat: int,
    turn_order_observation_sha256: str,
    terminal_status: str,
    winner_seat: int | None,
    illegal_action_count: int,
    forfeit_count: int,
    crash_count: int,
    timeout_count: int,
    decisions: tuple[Guide2VecDecisionReceipt, ...],
) -> Guide2VecBO1000GameReceipt:
    if observed_first_actor_seat not in {0, 1}:
        raise Guide2VecBO1000RuntimeError("native observed first actor seat is invalid")
    if observed_first_actor_seat != spec.sealed_initial_first_actor_seat:
        raise Guide2VecBO1000RuntimeError("native observed first actor differs from sealed r212 pair")
    _require_digest(turn_order_observation_sha256, label="native turn-order observation")
    observed_first = observed_first_actor_seat
    guide_first = observed_first == spec.guide2vec_seat
    control_first = not guide_first
    first_arm = GUIDE2VEC_ARM if guide_first else CONTROL_ARM
    adapter = experiment.base_runtime
    parity = expected_matchup_adapter_parity_attestation(
        game_nonce_sha256=spec.game_nonce_sha256,
        candidate_matchup_tree_sha256=adapter.matchup_tree_sha256,
        control_matchup_tree_sha256=adapter.matchup_tree_sha256,
        candidate_matchup_adapter_bank_sha256=adapter.matchup_adapter_bank_sha256,
        control_matchup_adapter_bank_sha256=adapter.matchup_adapter_bank_sha256,
        candidate_matchup_adapter_training_receipt_sha256=(
            adapter.matchup_adapter_training_receipt_sha256
        ),
        control_matchup_adapter_training_receipt_sha256=(
            adapter.matchup_adapter_training_receipt_sha256
        ),
        candidate_matchup_adapter_runtime_graph_sha256=(
            adapter.matchup_adapter_runtime_graph_sha256
        ),
        control_matchup_adapter_runtime_graph_sha256=(
            adapter.matchup_adapter_runtime_graph_sha256
        ),
        candidate_matchup_adapter_enabled=True,
        control_matchup_adapter_enabled=True,
        candidate_matchup_adapter_trained=True,
        control_matchup_adapter_trained=True,
        candidate_matchup_adapter_frozen=True,
        control_matchup_adapter_frozen=True,
    )
    absence = expected_control_guide2vec_absence_attestation(
        game_nonce_sha256=spec.game_nonce_sha256,
        control_runtime_graph_sha256=experiment.control_runtime_graph_sha256,
        control_runtime_graph_observation_sha256=control_audit.observation_sha256,
    )
    return Guide2VecBO1000GameReceipt(
        game_nonce_sha256=spec.game_nonce_sha256,
        pair_id=spec.pair_id,
        game_index=spec.game_index,
        guide2vec_seat=spec.guide2vec_seat,
        control_seat=spec.control_seat,
        pair_initial_rng_sha256=spec.pair_initial_rng_sha256,
        pair_deck_order_rng_sha256=spec.pair_deck_order_rng_sha256,
        sealed_initial_first_actor_seat=observed_first,
        observed_first_actor_seat=observed_first,
        observed_first_actor_arm=first_arm,
        guide2vec_is_first=guide_first,
        control_is_first=control_first,
        is_first_attestation_sha256=expected_is_first_attestation(
            game_nonce_sha256=spec.game_nonce_sha256,
            observed_first_actor_seat=observed_first,
            guide2vec_seat=spec.guide2vec_seat,
            control_seat=spec.control_seat,
            observed_first_actor_arm=first_arm,
            guide2vec_is_first=guide_first,
            control_is_first=control_first,
        ),
        turn_order_observation_sha256=turn_order_observation_sha256,
        base_runtime_identity_sha256=experiment.base_runtime.identity_sha256,
        experiment_identity_sha256=experiment.identity_sha256,
        evaluation_output_identity_sha256=experiment.evaluation_output_identity_sha256,
        candidate_runtime_graph_sha256=experiment.candidate_runtime_graph_sha256,
        candidate_base_runtime_graph_sha256=experiment.control_runtime_graph_sha256,
        control_runtime_graph_sha256=experiment.control_runtime_graph_sha256,
        candidate_matchup_tree_sha256=adapter.matchup_tree_sha256,
        control_matchup_tree_sha256=adapter.matchup_tree_sha256,
        candidate_matchup_adapter_bank_sha256=adapter.matchup_adapter_bank_sha256,
        control_matchup_adapter_bank_sha256=adapter.matchup_adapter_bank_sha256,
        candidate_matchup_adapter_training_receipt_sha256=(
            adapter.matchup_adapter_training_receipt_sha256
        ),
        control_matchup_adapter_training_receipt_sha256=(
            adapter.matchup_adapter_training_receipt_sha256
        ),
        candidate_matchup_adapter_runtime_graph_sha256=(
            adapter.matchup_adapter_runtime_graph_sha256
        ),
        control_matchup_adapter_runtime_graph_sha256=(
            adapter.matchup_adapter_runtime_graph_sha256
        ),
        candidate_matchup_adapter_enabled=True,
        control_matchup_adapter_enabled=True,
        candidate_matchup_adapter_trained=True,
        control_matchup_adapter_trained=True,
        candidate_matchup_adapter_frozen=True,
        control_matchup_adapter_frozen=True,
        matchup_adapter_parity_attestation_sha256=parity,
        candidate_guide2vec_component_graph_sha256=(
            experiment.candidate_guide2vec_component_graph_sha256
        ),
        runtime_graph_difference_receipt_sha256=(
            experiment.runtime_graph_difference_receipt_sha256
        ),
        candidate_guide2vec_checkpoint_sha256=experiment.guide2vec_checkpoint_sha256,
        candidate_guide2vec_training_receipt_sha256=(
            experiment.guide2vec_training_receipt_sha256
        ),
        candidate_guide2vec_runtime_config_sha256=(
            experiment.guide2vec_runtime_config_sha256
        ),
        candidate_guide2vec_presence=CANDIDATE_GUIDE2VEC_PRESENCE,
        candidate_guide2vec_module_instance_count=1,
        candidate_guide2vec_parameter_count=experiment.guide2vec_parameter_count,
        candidate_guide2vec_frozen=True,
        control_guide2vec_presence=CONTROL_GUIDE2VEC_PRESENCE,
        control_guide2vec_module_instance_count=control_audit.module_instance_count,
        control_guide2vec_parameter_count=control_audit.parameter_count,
        control_guide2vec_state_dict_key_count=control_audit.state_dict_key_count,
        control_guide2vec_forward_hook_count=control_audit.forward_hook_count,
        control_guide2vec_linear_transform_count=control_audit.linear_transform_count,
        control_guide2vec_disabled_or_zeroed=False,
        control_runtime_graph_observation_sha256=control_audit.observation_sha256,
        control_guide2vec_absence_attestation_sha256=absence,
        guide2vec_execution_mode="bounded_guide_logit_bonus",
        control_execution_mode="frozen_r195_no_rtp_direct_policy",
        terminal_status=terminal_status,
        winner_seat=winner_seat,
        illegal_action_count=illegal_action_count,
        forfeit_count=forfeit_count,
        crash_count=crash_count,
        timeout_count=timeout_count,
        guide2vec_decisions=decisions,
    )


def _play_native_game(
    *,
    module: Any,
    deck: Sequence[int],
    spec: Guide2VecBO1000GameSpec,
    binding: NativePairBinding,
    overlay: _CandidateOverlay,
    control_policy: Any,
    experiment: Guide2VecExperimentIdentity,
    control_audit: ControlGraphAudit,
    max_atomic_actions: int,
) -> tuple[Guide2VecBO1000GameReceipt, dict[str, object]]:
    """Play exactly one direct-vs-overlay game, returning a terminal receipt."""

    if max_atomic_actions < 1:
        raise Guide2VecBO1000RuntimeError("max_atomic_actions must be positive")
    native_game = binding.native_games[spec.game_index]
    if native_game.experimental_seat != spec.guide2vec_seat:
        raise Guide2VecBO1000RuntimeError("native game candidate seat differs from r212 spec")
    environment = _new_seeded_environment()
    overlay.reset_game(
        game_nonce_sha256=spec.game_nonce_sha256,
        acting_seat=spec.guide2vec_seat,
    )
    control_policy.reset_game()
    control_policy.strict_runtime = True
    steps = 0
    setup_actions: list[dict[str, object]] = []
    pre_seal_setup_action_values: list[list[int]] = []
    failure: str | None = None
    terminal_status = "failed_closed"
    winner: int | None = None
    illegal_count = forfeit_count = crash_count = timeout_count = 0
    first_verified = False
    observed_first_actor_seat = binding.native_first_player_seal.first_player_seat
    turn_order_observation_sha256 = (
        binding.native_first_player_seal.post_turn_order_observation_sha256
    )
    try:
        state = _reset_seeded_game(environment, deck, native_game.engine_seed_u32)
        while not state.done and steps < max_atomic_actions:
            observation = state.obs
            observed_first = _first_player(observation)
            if observed_first is None:
                action = _package_turn_order_action(module, observation)
                if action is None:
                    raise Guide2VecBO1000RuntimeError(
                        "unsealed native setup lacks exact package action"
                    )
                setup_actions.append({"acting_seat": (observation.get("current") or {}).get("yourIndex"), "action": action})
                pre_seal_setup_action_values.append(list(action))
                state = environment.step_batch([action]).envs[0]
                steps += 1
                continue
            if observed_first != binding.native_first_player_seal.first_player_seat:
                raise Guide2VecBO1000RuntimeError("game first-player result differs from pair seal")
            if observed_first != spec.sealed_initial_first_actor_seat:
                raise Guide2VecBO1000RuntimeError("game first-player result differs from r212 schedule")
            observed_first_actor_seat = observed_first
            turn_order_observation_sha256 = _native_sha256(observation)
            if (
                turn_order_observation_sha256
                != binding.native_first_player_seal.post_turn_order_observation_sha256
            ):
                raise Guide2VecBO1000RuntimeError(
                    "native turn-order snapshot differs from pre-sealed pair snapshot"
                )
            if canonical_sha256(pre_seal_setup_action_values) != binding.setup_actions_sha256:
                raise Guide2VecBO1000RuntimeError(
                    "native setup actions differ from pre-sealed pair setup"
                )
            first_verified = True
            turn_order = _package_turn_order_action(module, observation)
            if turn_order is not None:
                setup_actions.append({"acting_seat": (observation.get("current") or {}).get("yourIndex"), "action": turn_order})
                state = environment.step_batch([turn_order]).envs[0]
                steps += 1
                continue
            current = observation.get("current")
            if not isinstance(current, Mapping) or current.get("yourIndex") not in {0, 1}:
                raise Guide2VecBO1000RuntimeError("native game emits invalid acting seat")
            seat = int(current["yourIndex"])
            try:
                if seat == spec.guide2vec_seat:
                    action = overlay.select(observation)
                elif seat == spec.control_seat:
                    action = _CandidateOverlay._validate_action(
                        observation,
                        control_policy.trusted_search_or_greedy_select(
                            dict(observation), search=False
                        ),
                    )
                else:
                    raise Guide2VecBO1000RuntimeError("native game seat is outside pair")
            except TimeoutError:
                timeout_count += 1
                raise
            except Guide2VecBO1000RuntimeError:
                illegal_count += 1
                raise
            state = environment.step_batch([action]).envs[0]
            steps += 1
        if not first_verified:
            raise Guide2VecBO1000RuntimeError("native game ended before first-player verification")
        if not state.done:
            timeout_count += 1
            raise Guide2VecBO1000RuntimeError("native game exceeded atomic action cap")
        raw_winner = getattr(state, "winner", None)
        winner = int(raw_winner) if raw_winner in {0, 1} else None
        terminal_status = "completed"
    except TimeoutError as exc:
        failure = f"{type(exc).__name__}: {exc}"
        crash_count += 1
    except Exception as exc:  # noqa: BLE001 - terminal receipt must fail closed.
        failure = f"{type(exc).__name__}: {exc}"
        crash_count += 1
    finally:
        environment.close()
    if terminal_status != "completed":
        winner = None
    decisions = _bound_candidate_decisions(
        overlay, game_nonce_sha256=spec.game_nonce_sha256, seat=spec.guide2vec_seat
    )
    receipt = _game_receipt(
        spec=spec,
        experiment=experiment,
        control_audit=control_audit,
        observed_first_actor_seat=observed_first_actor_seat,
        turn_order_observation_sha256=turn_order_observation_sha256,
        terminal_status=terminal_status,
        winner_seat=winner,
        illegal_action_count=illegal_count,
        forfeit_count=forfeit_count,
        crash_count=crash_count,
        timeout_count=timeout_count,
        decisions=decisions,
    )
    trace: dict[str, object] = {
        "schema": R212_GAME_TRACE_SCHEMA,
        "evaluation_id": GUIDE2VEC_EVALUATION_ID,
        "game_nonce_sha256": spec.game_nonce_sha256,
        "pair_id": spec.pair_id,
        "pair_index": spec.pair_index,
        "native_pair_binding_sha256": binding.identity_sha256,
        "native_engine_seed_u32": native_game.engine_seed_u32,
        "native_deck_order_seed_u32": native_game.deck_order_seed_u32,
        "sealed_first_actor_seat": binding.native_first_player_seal.first_player_seat,
        "observed_first_actor_seat": observed_first_actor_seat,
        "turn_order_observation_sha256": turn_order_observation_sha256,
        "native_first_player_revalidated_in_game": first_verified,
        "setup_actions": setup_actions,
        "steps": steps,
        "terminal_status": terminal_status,
        "winner_seat": winner,
        "error": failure,
        "candidate_matchup_adapter_final_snapshot": overlay.policy.matchup_adapter_shadow_snapshot(),
        "control_matchup_adapter_final_snapshot": control_policy.matchup_adapter_shadow_snapshot(),
        "guide2vec_decisions": overlay.trace_rows,
        "receipt": receipt.as_payload(),
    }
    trace["canonical_sha256"] = canonical_sha256(trace)
    return receipt, trace


def _artifact_paths_from_plan(plan: Mapping[str, object]) -> R212ArtifactIdentity:
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Guide2VecBO1000RuntimeError("plan artifacts are missing")

    def path_at(key: str) -> Path:
        row = artifacts.get(key)
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise Guide2VecBO1000RuntimeError(f"plan has no artifact path for {key}")
        return Path(str(row["path"]))

    # Contracts are deliberately local canonical sources, not package assets.
    root = Path(__file__).resolve().parents[1]
    return R212ArtifactIdentity(
        r195_bundle=path_at("r195_bundle"),
        r195_package_root=path_at("r195_package_root"),
        r195_checkpoint=path_at("r195_checkpoint"),
        guide2vec_checkpoint=path_at("guide2vec_checkpoint"),
        guide2vec_training_receipt=path_at("guide2vec_training_receipt"),
        owner_contract=root / "state/alakazam-guide2vec-no-mcts-bo1000-r212.json",
        r195_contract=root / "state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json",
    )


def _assert_plan_reproducible_from_artifacts(
    plan: Mapping[str, object],
    artifacts: R212ArtifactIdentity,
) -> None:
    """Reject a self-consistent plan unless its bytes rebuild from current inputs."""

    seed_identity = _require_digest(
        plan.get("seed_identity_sha256"), label="r212 plan seed identity"
    )
    rebuilt = build_guide2vec_bo1000_plan(
        artifacts=artifacts,
        seed_identity_sha256=seed_identity,
    )
    if canonical_sha256(dict(plan)) != canonical_sha256(rebuilt):
        raise Guide2VecBO1000RuntimeError(
            "r212 plan no longer reproduces from its exact immutable artifacts"
        )


def materialize_guide2vec_bo1000_plan(
    *,
    plan: Mapping[str, object],
    output_root: str | Path,
) -> Path:
    """Write a create-once plan/index only; no native engine is opened."""

    experiment, _schedule = verify_guide2vec_bo1000_plan(plan)
    root = Path(output_root).resolve()
    output = root / f"r212-{experiment.evaluation_output_identity_sha256[7:31]}"
    _write_immutable_json(output / "PLAN.json", dict(plan))
    _write_immutable_json(
        output / "OUTPUT_IDENTITY.json",
        {
            "schema": R212_RUNTIME_PLAN_SCHEMA,
            "evaluation_id": GUIDE2VEC_EVALUATION_ID,
            "experiment_identity_sha256": experiment.identity_sha256,
            "evaluation_output_identity_sha256": experiment.evaluation_output_identity_sha256,
            "content_addressed_output": True,
            "training_eligible": False,
            "service_start": False,
        },
    )
    return output


def preflight_guide2vec_bo1000_runtime(
    *,
    plan: Mapping[str, object],
    output_root: str | Path | None = None,
) -> dict[str, object]:
    """Load and audit frozen runtime graphs without starting a battle."""

    experiment, _schedule = verify_guide2vec_bo1000_plan(plan)
    artifacts = _artifact_paths_from_plan(plan)
    _assert_plan_reproducible_from_artifacts(plan, artifacts)
    candidate, _model_config, adapter_bank_sha, _adapter_fit, adapter_config, _package = (
        verify_r212_artifacts(artifacts)
    )
    model_state_sha = _checkpoint_model_state_sha256(artifacts.r195_checkpoint)
    if candidate.checkpoint_sha256 != experiment.guide2vec_checkpoint_sha256:
        raise Guide2VecBO1000RuntimeError("preflight Guide2Vec artifact differs from plan")
    module = _load_archived_submission(artifacts.r195_package_root)
    loaded = module._ensure_runtime()
    if not isinstance(loaded, tuple) or len(loaded) != 3:
        raise Guide2VecBO1000RuntimeError("r195 package runtime loader changed")
    deck, model, control_policy = loaded
    if not isinstance(deck, list) or len(deck) != 60 or not isinstance(model, nn.Module):
        raise Guide2VecBO1000RuntimeError("r195 package runtime is malformed")
    control_policy.strict_runtime = True
    candidate_policy = _make_candidate_policy(
        model=model,
        deck=deck,
        control_policy=control_policy,
    )
    device = next(model.parameters()).device
    head = _load_frozen_head(
        checkpoint_path=artifacts.guide2vec_checkpoint,
        candidate=candidate,
        device=device,
    )
    receipt = _runtime_preflight(
        model=model,
        control_policy=control_policy,
        candidate_policy=candidate_policy,
        candidate_head=head,
        plan=plan,
        candidate=candidate,
        model_state_sha256=model_state_sha,
        adapter_bank_sha=adapter_bank_sha,
        adapter_config=adapter_config,
    )
    if output_root is not None:
        output = materialize_guide2vec_bo1000_plan(plan=plan, output_root=output_root)
        _write_immutable_json(output / "PREFLIGHT_RECEIPT.json", receipt)
    return receipt


def run_guide2vec_bo1000(
    *,
    plan: Mapping[str, object],
    output_root: str | Path,
    max_atomic_actions: int = 4000,
    native_seed_attempts: int = 32,
) -> dict[str, object]:
    """Explicitly execute the separate r212 BO1000 after all gates pass.

    This function is intentionally not called by plan creation or preflight.
    It preserves individual bindings/traces/receipts and compiles only after
    all 1,000 terminal receipts exist.  A failed game is recorded as
    ``failed_closed``; it is never imputed as a win/loss/draw.
    """

    experiment, schedule = verify_guide2vec_bo1000_plan(plan)
    artifacts = _artifact_paths_from_plan(plan)
    _assert_plan_reproducible_from_artifacts(plan, artifacts)
    output = materialize_guide2vec_bo1000_plan(plan=plan, output_root=output_root)
    candidate, _model_config, adapter_bank_sha, _adapter_fit, adapter_config, _package = (
        verify_r212_artifacts(artifacts)
    )
    model_state_sha = _checkpoint_model_state_sha256(artifacts.r195_checkpoint)
    module = _load_archived_submission(artifacts.r195_package_root)
    deck, model, control_policy = module._ensure_runtime()
    if not isinstance(deck, list) or not isinstance(model, nn.Module):
        raise Guide2VecBO1000RuntimeError("r195 runtime loader returned malformed values")
    control_policy.strict_runtime = True
    candidate_policy = _make_candidate_policy(
        model=model,
        deck=deck,
        control_policy=control_policy,
    )
    device = next(model.parameters()).device
    head = _load_frozen_head(
        checkpoint_path=artifacts.guide2vec_checkpoint,
        candidate=candidate,
        device=device,
    )
    preflight = _runtime_preflight(
        model=model,
        control_policy=control_policy,
        candidate_policy=candidate_policy,
        candidate_head=head,
        plan=plan,
        candidate=candidate,
        model_state_sha256=model_state_sha,
        adapter_bank_sha=adapter_bank_sha,
        adapter_config=adapter_config,
    )
    _write_immutable_json(output / "PREFLIGHT_RECEIPT.json", preflight)
    control_audit = inspect_control_graph(model=model, policy=control_policy)
    expected_base = FrozenBaseIdentity(
        submission_id=R195_SUBMISSION_ID,
        checkpoint_sha256=R195_CHECKPOINT_SHA256,
        checkpoint_bytes=R195_CHECKPOINT_BYTES,
        bundle_sha256=R195_BUNDLE_SHA256,
        model_config_sha256=candidate.model_config_sha256,
        feature_schema_sha256=candidate.feature_schema_sha256,
    )
    overlay = _CandidateOverlay(
        policy=candidate_policy,
        head=head,
        expected_base_identity=expected_base,
        decision_rows=[],
        trace_rows=[],
    )
    pair_specs: dict[str, list[Guide2VecBO1000GameSpec]] = {}
    for spec in schedule:
        pair_specs.setdefault(spec.pair_id, []).append(spec)
    binding_digest_by_pair: dict[str, str] = {}
    for pair_id, specs in sorted(pair_specs.items(), key=lambda item: item[1][0].pair_index):
        binding_path = output / "pair-bindings" / f"{pair_id}.json"
        if binding_path.exists():
            binding = NativePairBinding.from_payload(_read_json_object(binding_path, label="native pair binding"))
            _binding_matches_specs(
                binding,
                specs,
                seed_identity_sha256=str(plan["seed_identity_sha256"]),
            )
        else:
            binding = _capture_native_pair_seal(
                module=module,
                deck=deck,
                r212_specs=specs,
                seed_identity_sha256=str(plan["seed_identity_sha256"]),
                max_attempts=native_seed_attempts,
            )
            _binding_matches_specs(
                binding,
                specs,
                seed_identity_sha256=str(plan["seed_identity_sha256"]),
            )
            _write_immutable_json(binding_path, binding.as_payload())
        binding_digest_by_pair[pair_id] = binding.identity_sha256
        for spec in sorted(specs, key=lambda value: value.game_index):
            trace_path = output / "games" / spec.game_nonce_sha256[7:] / "TRACE.json"
            if trace_path.exists():
                continue
            receipt, trace = _play_native_game(
                module=module,
                deck=deck,
                spec=spec,
                binding=binding,
                overlay=overlay,
                control_policy=control_policy,
                experiment=experiment,
                control_audit=control_audit,
                max_atomic_actions=max_atomic_actions,
            )
            _write_immutable_json(trace_path, trace)
            _write_immutable_json(
                trace_path.with_name("RECEIPT.json"), receipt.as_payload()
            )
    receipts: list[Guide2VecBO1000GameReceipt] = []
    bindings: list[str] = []
    for spec in schedule:
        trace_path = output / "games" / spec.game_nonce_sha256[7:] / "TRACE.json"
        receipt_path = trace_path.with_name("RECEIPT.json")
        if not trace_path.is_file() or not receipt_path.is_file():
            raise Guide2VecBO1000RuntimeError("r212 execution is missing a terminal game artifact")
        trace = _read_json_object(trace_path, label="r212 game trace")
        if trace.get("canonical_sha256") != canonical_sha256(
            {key: value for key, value in trace.items() if key != "canonical_sha256"}
        ):
            raise Guide2VecBO1000RuntimeError("r212 game trace digest drifted")
        binding_sha = trace.get("native_pair_binding_sha256")
        bound_digest = _require_digest(binding_sha, label="native pair binding digest")
        if (
            trace.get("schema") != R212_GAME_TRACE_SCHEMA
            or trace.get("game_nonce_sha256") != spec.game_nonce_sha256
            or trace.get("pair_id") != spec.pair_id
            or bound_digest != binding_digest_by_pair.get(spec.pair_id)
        ):
            raise Guide2VecBO1000RuntimeError("r212 trace does not bind its scheduled native pair")
        bindings.append(bound_digest)
        try:
            receipt = Guide2VecBO1000GameReceipt.from_payload(
                _read_json_object(receipt_path, label="r212 game receipt")
            )
        except Guide2VecBO1000Error as exc:
            raise Guide2VecBO1000RuntimeError("stored r212 game receipt is invalid") from exc
        if trace.get("receipt") != receipt.as_payload():
            raise Guide2VecBO1000RuntimeError("r212 trace and standalone receipt differ")
        receipts.append(receipt)
    try:
        compiled = compile_guide2vec_bo1000_report(schedule, receipts, experiment=experiment)
    except Guide2VecBO1000Error as exc:
        raise Guide2VecBO1000RuntimeError("r212 BO1000 compiler rejected runtime receipts") from exc
    _write_immutable_json(output / "REPORT.json", compiled)
    final: dict[str, object] = {
        "schema": R212_FINAL_RECEIPT_SCHEMA,
        "evaluation_id": GUIDE2VEC_EVALUATION_ID,
        "r212_contract_sha256": R212_CONTRACT_SHA256,
        "r195_contract_sha256": R195_CONTRACT_SHA256,
        "experiment_identity_sha256": experiment.identity_sha256,
        "evaluation_output_identity_sha256": experiment.evaluation_output_identity_sha256,
        "plan_sha256": plan.get("canonical_sha256"),
        "native_pair_binding_count": len(set(bindings)),
        "native_pair_binding_manifest_sha256": canonical_sha256(sorted(bindings)),
        "compiler_report_sha256": canonical_sha256(compiled),
        "compiler_report": compiled,
        "execution_limits": {
            "max_atomic_actions_per_game": max_atomic_actions,
            "native_seed_attempts_per_pair": native_seed_attempts,
            "early_stop": False,
        },
        "authority": {
            "training": False,
            "mcts": False,
            "rtp": False,
            "serving": False,
            "selector": False,
            "kaggle": False,
            "promotion": False,
            "service_start": False,
        },
    }
    final["canonical_sha256"] = canonical_sha256(final)
    _write_immutable_json(output / "FINAL_RECEIPT.json", final)
    return final


__all__ = [
    "CandidateArtifact",
    "CandidateGraphAudit",
    "ControlGraphAudit",
    "Guide2VecBO1000RuntimeError",
    "NativePairBinding",
    "R212ArtifactIdentity",
    "R212_FINAL_RECEIPT_SCHEMA",
    "R212_GAME_TRACE_SCHEMA",
    "R212_NATIVE_PAIR_BINDING_SCHEMA",
    "R212_RUNTIME_GRAPH_SCHEMA",
    "R212_RUNTIME_PLAN_SCHEMA",
    "R212_RUNTIME_PREFLIGHT_SCHEMA",
    "build_guide2vec_bo1000_plan",
    "canonical_sha256",
    "inspect_candidate_graph",
    "inspect_control_graph",
    "materialize_guide2vec_bo1000_plan",
    "preflight_guide2vec_bo1000_runtime",
    "run_guide2vec_bo1000",
    "sha256_file",
    "verify_guide2vec_bo1000_plan",
    "verify_r212_artifacts",
]
