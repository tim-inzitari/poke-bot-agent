#!/usr/bin/env python3
"""Train the isolated, frozen-base Alakazam Guide2Vec r212 sidecar.

This program is deliberately not part of the pure-RL, RTP, selector, or
submission paths.  It reads the immutable r195 NO-RTP checkpoint and protected
expert corpus, derives causal base-model state/option latents once, and trains
only the small :mod:`poke_bot.guide2vec` sidecar.  Its only writes are
content-addressed artifacts below the explicitly supplied r212 output root.

The data split is by immutable source day, rather than a random row split:

* train: 2026-07-04 through 2026-07-19;
* validation/calibration: 2026-07-20 and 2026-07-21;
* heldout (read only after selection): 2026-07-22 and 2026-07-23.

There is no MCTS, RTP executor, selector mutation, base-model optimization, or
serving activation in this file.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tarfile
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# A content-addressed source snapshot must stay byte-stable while this job is
# running.  In particular, importing the trainer must not create pyc files in
# the published snapshot.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R226_RETIREMENT_CONTRACT_PATH = (
    ROOT / "state/guide2vec-general-training-pipeline-r226.json"
)
R226_RETIREMENT_CONTRACT_SHA256 = (
    "sha256:5a1a3283e3097678c4feb02f468e901caf849548307f132bbddb339977958473"
)
R212_TRAINING_RETIRED_ERROR = (
    "r212 Guide2Vec training is retired by r226; no gradient, GPU, chunk, or output work is authorized"
)


def _reject_r212_training_under_r226() -> None:
    """Fail closed before r212 can touch inputs, CUDA, outputs, or gradients.

    Revision 226 preserves the r212 files as audit evidence, but retires both
    its training and BO1000 launch authority.  This deliberately always raises:
    an absent, changed, or malformed retirement contract is never a path back
    to the retired trainer.
    """

    try:
        raw = R226_RETIREMENT_CONTRACT_PATH.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"{R212_TRAINING_RETIRED_ERROR}: r226 retirement contract is unavailable"
        ) from exc
    observed_sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
    if observed_sha256 != R226_RETIREMENT_CONTRACT_SHA256:
        raise RuntimeError(
            f"{R212_TRAINING_RETIRED_ERROR}: r226 retirement contract digest mismatch"
        )
    try:
        retirement = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{R212_TRAINING_RETIRED_ERROR}: r226 retirement contract is malformed"
        ) from exc
    if not isinstance(retirement, dict):
        raise RuntimeError(
            f"{R212_TRAINING_RETIRED_ERROR}: r226 retirement contract is not an object"
        )
    supersession = retirement.get("supersession")
    authority = retirement.get("authority")
    if (
        retirement.get("schema")
        != "poke_bot.guide2vec_general_training_pipeline_r226/v1"
        or retirement.get("owner_decision_revision") != 226
        or not isinstance(supersession, dict)
        or not isinstance(authority, dict)
        or supersession.get("r212_training_launch_authority_retired") is not True
        or supersession.get("r212_bo1000_launch_authority_retired") is not True
        or supersession.get("r212_service_must_remain_unlinked_and_unstarted")
        is not True
        or authority.get("training_service_start_authorized") is not False
        or authority.get("gradient_updates_authorized") is not False
        or authority.get("candidate_publication_authorized") is not False
        or authority.get("bo1000_authorized") is not False
    ):
        raise RuntimeError(
            f"{R212_TRAINING_RETIRED_ERROR}: r226 retirement contract is incomplete"
        )
    raise RuntimeError(R212_TRAINING_RETIRED_ERROR)


# Execute the retired `--run` fence before importing torch or any project
# module.  The in-process guards below protect direct helper use as well.
if __name__ == "__main__" and "--run" in sys.argv[1:]:
    _reject_r212_training_under_r226()

import torch
import torch.nn.functional as F

from poke_bot import checkpoint
from poke_bot.alakazam_heuristics import is_alakazam_deck
from poke_bot.dataset import PolicyStage
from poke_bot.device_corpus import DeviceResidentBootstrapCorpus
from poke_bot.feature_shards import COMPACT_MODE_TEMPORAL_EXPERT, iter_feature_shard
from poke_bot.guide2vec_public_routes import (
    ROUTE_ALGORITHM,
    ROUTE_RECONSTRUCTION_SCHEMA,
    RawPublicRouteResolver,
    SidecarPublicRouteResolver,
)
from poke_bot.matchup_adapters import UNKNOWN_ROUTE
from poke_bot.train import cap_game_sequence_context, load_model_from_checkpoint


SCHEMA = "poke_bot.alakazam_guide2vec_r212_training/v1"
LATENT_SCHEMA = "poke_bot.alakazam_guide2vec_r212_frozen_latents/v1"
LATENT_EXTRACTOR = "r212_frozen_temporal_state_option_hidden_v1"
SIDE_CAR_SCHEMA = "poke_bot.alakazam_guide2vec_r212_sidecar/v1"
RECEIPT_SCHEMA = "poke_bot.alakazam_guide2vec_r212_training_receipt/v1"
RUNTIME_CONFIG_SCHEMA = "poke_bot.alakazam_guide2vec_r212_runtime_config/v1"
SELECTION_METRIC = "validation_confidence_weighted_listwise_cross_entropy"
SOURCE_SNAPSHOT_SCHEMA = "poke_bot.alakazam_guide2vec_r212_source_snapshot/v1"
SOURCE_SNAPSHOT_MANIFEST = "guide2vec-r212-source-snapshot-manifest.json"
ROUTE_SIDECAR_MANIFEST_SCHEMA = (
    "poke_bot.guide2vec_r212_r195_public_route_sidecar_manifest/v1"
)
ROUTE_SIDECAR_PRODUCER_CODE_SCHEMA = (
    "poke_bot.guide2vec_r212_r195_public_route_producer_code/v1"
)

R195_CHECKPOINT_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
R195_CHECKPOINT_BYTES = 127_914_385
R195_SUBMISSION_ID = "55378392"
R195_SUBMISSION_LABEL = (
    "alakazam training milestone iter 21 copy 1/2 first 261d367e131e NO RTP"
)
R195_BUNDLE_SHA256 = (
    "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
)
R195_DECK_ID = "alakazam-owner-rtp-pilot-r175"
R195_DECK_CARDS_SHA256 = (
    "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
)
R195_DECK_CSV_SHA256 = (
    "sha256:1705f0f4db0c54b32f297fc9292a417b0c3abc9fdb6edf6a5370af6a635efe65"
)
R195_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
R195_RUNTIME_ENTRYPOINT_SHA256 = (
    "sha256:02b6ea8b565e0bb66aed14719cc80636c388742d3af40408a3eb458baa4bd8d7"
)
R195_RUNTIME_ROUTER_SHA256 = (
    "sha256:98b1f6cc871ea56f295aaed9c1fbaad46fbe64036f1ae12d2de31f0f787c4a6a"
)
R195_RUNTIME_ENTRYPOINT_MEMBER = "./main.py"
R195_RUNTIME_ROUTER_MEMBER = "./poke_bot/public_matchup_router.py"
R195_TURN_ORDER_SHORT_CIRCUIT = "r195_submission_main_turn_order_choice/v1"
PROTECTED_POINTER_SHA256 = (
    "sha256:e9d9182eea543e7bfe12ad5c2e7a1784fefa4866059ed153965dfffd12c7da1a"
)
MANIFEST_SHA256 = (
    "sha256:3836852129511fdffd2767f6701dcc562d1723bb0c345c3cf5068ad9774b9acb"
)
BLACKWELL_UUID = "GPU-79cf504f-6573-0b8c-c90e-eb567b7bcfa6"
BLACKWELL_MIN_MEMORY_BYTES = 48_000_000_000
EXPECTED_D_MODEL = 96
EXPECTED_GUIDE2VEC_PARAMETERS = 155_468
MAX_GUIDE2VEC_PARAMETERS = 500_000
MAX_EPOCHS = 5
MAX_GUIDE_LOGIT_BONUS = 0.05
OWNER_CONTRACT_PATH = ROOT / "state/alakazam-guide2vec-no-mcts-bo1000-r212.json"
OWNER_CONTRACT_SHA256 = (
    "sha256:aa9c7b8158c91d183c092b92bab3047c7bd7af705d539c68cdd3e9c206c0c2b9"
)
R195_CONTRACT_PATH = ROOT / "state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json"
R195_CONTRACT_SHA256 = (
    "sha256:e37cf1d3e638c3aed56230c9fa970c61e6c1ed8b4bd3024de259cb9847c31e48"
)
GUIDE_CONTRACT_PATH = ROOT / "config/deck_guides/alakazam-final-refresh.yaml"
GUIDE_CONTRACT_SHA256 = (
    "sha256:f2ce4dfc255ec634e76f8b04b943aecebdd95aad5318bbcd7eab369f938d6798"
)
GUIDE_READY_RECEIPT_PATH = ROOT / "state/final_format_alakazam_guide_ready_r79.json"
GUIDE_READY_RECEIPT_SHA256 = (
    "sha256:634c9db8bd3ff636c277d0ae2c8b8aa1508b7ac9e66b774da383f30f8ca7df84"
)
GUIDE_TEACHER_PATH = ROOT / "poke_bot/alakazam_heuristics.py"
GUIDE_TEACHER_SHA256 = (
    "sha256:c9f8890d5751a70c9a7ce9026545823e92f8384227281f8dba168323d9a13d86"
)

DEFAULT_CHECKPOINT = Path(
    "/home/inzi/poke-bot-agent/outputs/pure_rl/"
    "alakazam_terminal_expert_bootstrap_no_rtp_r195/checkpoints/"
    "expert_before_iter_00021.pt"
)
DEFAULT_PROTECTED_CORPUS = Path(
    "/home/inzi/poke-bot-agent/data/bootstrap/"
    "expert-latest20-2026-07-04-2026-07-23-roster18-v6-strategic/"
    "alakazam/PROTECTED_EXPERT_CORPUS.json"
)
DEFAULT_MATCHUP_TREE = Path(
    "/home/inzi/poke-bot-agent/outputs/submissions/"
    "alakazam-terminal-expert-bootstrap-r195-runtime-comparison/"
    "pinned-checkpoint-compatible-matchup-tree-v2.json"
)
DEFAULT_R195_SUBMISSION_BUNDLE = Path(
    "/home/inzi/poke-bot-agent/outputs/submissions/"
    "alakazam-terminal-expert-bootstrap-r195-runtime-comparison/"
    "build-no-rtp/submission.tar.gz"
)
DEFAULT_RAW_ARCHIVE_ROOT = Path("/home/inzi/poke-bot-agent/data/episodes/raw")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/inzi/poke-bot-agent/outputs/guide2vec/alakazam-r212"
)
ROUTE_MATERIALIZER_PATH = ROOT / "scripts/materialize_alakazam_guide2vec_r212_public_routes.py"

TRAIN_DATES = tuple(f"2026-07-{day:02d}" for day in range(4, 20))
VALIDATION_DATES = ("2026-07-20", "2026-07-21")
HELDOUT_DATES = ("2026-07-22", "2026-07-23")
ALL_DATES = TRAIN_DATES + VALIDATION_DATES + HELDOUT_DATES
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
QUARANTINE_POLICY = "fail_closed_known_zero_guide_teacher_incompatible_rows_only"
EXPECTED_QUARANTINE_RECORDS = 11
QUARANTINE_DATE = "2026-07-17"
EXPECTED_RETAINED_GUIDE_ROWS = 3_162_936
EXPECTED_QUARANTINE_IDENTITIES_SHA256 = (
    "sha256:9794adc8844ff63c6f41ac696c473f32155e530ecb2ca411115ae9126bb7998d"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = Path(path).stat()
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Publish canonical JSON once without replacing an old receipt."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _canonical_json(dict(value))
    if path.exists():
        if path.read_bytes() != body:
            raise RuntimeError(f"immutable artifact already differs: {path}")
        return
    temporary = path.with_name(
        f".{path.name}.partial.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        if not path.exists() or path.read_bytes() != body:
            raise RuntimeError(f"immutable artifact already differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    """Append one fsync'd progress row; no progress row is ever rewritten."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(dict(value))
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def _content_addressed_torch_save(
    value: Mapping[str, Any],
    *,
    directory: Path,
    prefix: str,
) -> tuple[Path, str]:
    """Save a torch object before atomically hard-linking its digest name."""

    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{prefix}.partial.{os.getpid()}.{time.time_ns()}.pt"
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        digest = _sha256(temporary)
        final = directory / f"{prefix}-{digest.split(':', 1)[1]}.pt"
        try:
            os.link(temporary, final)
        except FileExistsError:
            if _sha256(final) != digest:
                raise RuntimeError(f"content-addressed checkpoint collision: {final}")
        return final, digest
    finally:
        temporary.unlink(missing_ok=True)


def _torch_load_verified(path: Path) -> dict[str, Any]:
    """Load a locally-produced, hash-verified latent cache safely."""

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch versions before the safe-load keyword.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise RuntimeError(f"latent artifact is not a mapping: {path}")
    return value


def _model_config_sha256(payload: Mapping[str, Any]) -> str:
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        raise RuntimeError("r195 checkpoint has no serialized model_config")
    return "sha256:" + hashlib.sha256(_canonical_json(model_config)).hexdigest()


@dataclass(frozen=True)
class SourceShard:
    date: str
    path: Path
    sha256: str
    byte_count: int
    records: int
    decisions: int
    stat_identity: tuple[int, int, int, int, int]

    def assert_unchanged(self) -> None:
        if _stat_identity(self.path) != self.stat_identity:
            raise RuntimeError(f"protected shard changed after preflight: {self.path}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.byte_count,
            "records": self.records,
            "decisions": self.decisions,
        }


@dataclass(frozen=True)
class RouteSidecar:
    """One immutable heldout-day raw-route sidecar declared by its manifest."""

    date: str
    path: Path
    sha256: str
    raw_archive: Mapping[str, Any]
    producer_code: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "path": str(self.path),
            "sha256": self.sha256,
            "raw_archive": dict(self.raw_archive),
            "producer_code": dict(self.producer_code),
        }


@dataclass(frozen=True)
class RuntimeRouteCodeBinding:
    """Exact r195 submitted-tar members defining route semantics."""

    submission_bundle_path: Path
    submission_bundle_sha256: str
    entrypoint_member: str
    entrypoint_sha256: str
    router_member: str
    router_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "submission_bundle_sha256": self.submission_bundle_sha256,
            "submission_entrypoint_member": self.entrypoint_member,
            "submission_entrypoint_sha256": self.entrypoint_sha256,
            "public_matchup_router_member": self.router_member,
            "public_matchup_router_sha256": self.router_sha256,
            "turn_order_short_circuit_contract": R195_TURN_ORDER_SHORT_CIRCUIT,
        }

    def raw_resolver_kwargs(self) -> dict[str, Any]:
        return {
            "submission_bundle_path": self.submission_bundle_path,
            "submission_bundle_sha256": self.submission_bundle_sha256,
            "submission_entrypoint_member": self.entrypoint_member,
            "submission_entrypoint_sha256": self.entrypoint_sha256,
            "public_matchup_router_member": self.router_member,
            "public_matchup_router_sha256": self.router_sha256,
        }

    def sidecar_resolver_kwargs(self) -> dict[str, str]:
        return {
            "submission_bundle_sha256": self.submission_bundle_sha256,
            "submission_entrypoint_member": self.entrypoint_member,
            "submission_entrypoint_sha256": self.entrypoint_sha256,
            "public_matchup_router_member": self.router_member,
            "public_matchup_router_sha256": self.router_sha256,
        }


@dataclass(frozen=True)
class AdapterRouteBinding:
    """Exact V6 logical-route to physical-slot contract for frozen r195."""

    adapter_format: str
    target_ids: tuple[str, ...]
    physical_slots: tuple[int, ...]
    slot_capacity: int
    slot_registry_digest: str
    runtime_accepted_target_ids: tuple[str, ...] = ()
    runtime_accepted_physical_slots: tuple[int, ...] = ()
    runtime_accepted_nonzero_output_slots: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter_format": self.adapter_format,
            "route_target_ids": list(self.target_ids),
            "route_physical_slots": list(self.physical_slots),
            "physical_slot_capacity": self.slot_capacity,
            "slot_registry_digest": self.slot_registry_digest,
            "unknown_route": UNKNOWN_ROUTE,
            "target_slot_mapping": [
                {"target_id": target, "physical_slot": slot}
                for target, slot in zip(self.target_ids, self.physical_slots, strict=True)
            ],
            "runtime_accepted_target_ids": list(self.runtime_accepted_target_ids),
            "runtime_accepted_physical_slots": list(
                self.runtime_accepted_physical_slots
            ),
            "runtime_accepted_nonzero_output_slots": list(
                self.runtime_accepted_nonzero_output_slots
            ),
        }

    @property
    def runtime_accepted_slots(self) -> frozenset[int]:
        return frozenset(self.runtime_accepted_physical_slots)

    def checkpoint_contract_dict(self) -> dict[str, Any]:
        """Fields derived solely from the frozen checkpoint's V6 config."""

        return {
            "adapter_format": self.adapter_format,
            "target_ids": self.target_ids,
            "physical_slots": self.physical_slots,
            "slot_capacity": self.slot_capacity,
            "slot_registry_digest": self.slot_registry_digest,
        }


@dataclass(frozen=True)
class ValidatedInputs:
    checkpoint_path: Path
    checkpoint_payload: dict[str, Any]
    checkpoint_sha256: str
    checkpoint_bytes: int
    model_config_sha256: str
    pointer_path: Path
    pointer_sha256: str
    manifest_path: Path
    manifest_sha256: str
    max_context: int
    d_model: int
    shards_by_date: Mapping[str, SourceShard]
    typed_dependencies: Mapping[str, Mapping[str, str]]
    split_manifest: Mapping[str, Any]
    adapter_identity: Mapping[str, Any]
    adapter_route_binding: AdapterRouteBinding
    matchup_tree_path: Path
    matchup_tree_sha256: str
    runtime_route_code: RuntimeRouteCodeBinding
    raw_archive_root: Path
    route_sidecar_manifest_path: Path
    route_sidecar_manifest_sha256: str
    route_sidecars_by_date: Mapping[str, RouteSidecar]
    source_snapshot: Mapping[str, Any]
    quarantine_by_date: Mapping[str, Mapping[tuple[str, int], "QuarantineIdentity"]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": {
                "path": str(self.checkpoint_path),
                "sha256": self.checkpoint_sha256,
                "bytes": self.checkpoint_bytes,
                "submission_id": R195_SUBMISSION_ID,
                "submission_label": R195_SUBMISSION_LABEL,
                "bundle_sha256": R195_BUNDLE_SHA256,
                "deck_id": R195_DECK_ID,
                "deck_cards_sha256": R195_DECK_CARDS_SHA256,
                "deck_csv_sha256": R195_DECK_CSV_SHA256,
                "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
                "rtp_enabled": False,
                "matchup_adapter_runtime_enabled": True,
                "model_config_sha256": self.model_config_sha256,
                "d_model": self.d_model,
                "max_context": self.max_context,
            },
            "protected_corpus": {
                "pointer": str(self.pointer_path),
                "pointer_sha256": self.pointer_sha256,
                "manifest": str(self.manifest_path),
                "manifest_sha256": self.manifest_sha256,
                "teacher_compatible_deck_scope": {
                    "policy": (
                        "all_exact_teacher_compatible_alakazam_60_card_multisets_not_"
                        "single_r195_evaluation_deck_allowlist"
                    ),
                    "fingerprint_distribution_mode": "record_only_exact_60_card_multisets",
                    "expected_guide_rows": EXPECTED_RETAINED_GUIDE_ROWS,
                    "guide_label_source": (
                        "protected_r212_compact_guide_target_index_confidence"
                    ),
                },
                "shards": [
                    self.shards_by_date[date].as_dict() for date in ALL_DATES
                ],
            },
            "typed_dependencies": {
                name: dict(identity)
                for name, identity in sorted(self.typed_dependencies.items())
            },
            "source_snapshot": dict(self.source_snapshot),
            "split": dict(self.split_manifest),
            "matchup_adapter": {
                **dict(self.adapter_identity),
                "route_binding": self.adapter_route_binding.as_dict(),
                "runtime_tree": {
                    "path": str(self.matchup_tree_path),
                    "sha256": self.matchup_tree_sha256,
                    "runtime_enabled": True,
                },
            },
            "raw_public_route_reconstruction": {
                "schema": ROUTE_RECONSTRUCTION_SCHEMA,
                "algorithm": ROUTE_ALGORITHM,
                "raw_archive_root": str(self.raw_archive_root),
                "raw_dates": list(TRAIN_DATES + VALIDATION_DATES),
                "sidecar_manifest": {
                    "path": str(self.route_sidecar_manifest_path),
                    "sha256": self.route_sidecar_manifest_sha256,
                },
                "sidecars": [
                    self.route_sidecars_by_date[date].as_dict()
                    for date in HELDOUT_DATES
                ],
                "runtime_code": {
                    "submission_bundle_path": str(
                        self.runtime_route_code.submission_bundle_path
                    ),
                    **self.runtime_route_code.as_dict(),
                },
                "persisted_compact_routes_accepted": False,
                "oracle_route_used": False,
            },
        }


class _EmptyGames:
    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Iterator[Any]:
        return iter(())


@dataclass(frozen=True)
class QuarantineIdentity:
    """One receipt-approved, unlabeled incompatible source record.

    These records are never turned into eligibility negatives.  The compact
    corpus stays immutable; only the r212 split receipt grants this exact
    pre-packing exclusion.
    """

    source_date: str
    episode_id: str
    seat: int
    decisions: int
    guide_rows: int
    deck_fingerprint: str

    @property
    def key(self) -> tuple[str, int]:
        return self.episode_id, self.seat

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_date": self.source_date,
            "episode_id": self.episode_id,
            "seat": self.seat,
            "decisions": self.decisions,
            "guide_rows": self.guide_rows,
            "deck_fingerprint": self.deck_fingerprint,
        }


def _guide_rows_for_decision(decision: Any) -> int:
    """Strictly count comparable compact teacher labels for one decision."""

    labelled = 0
    stages = list(decision.policy_stages or ())
    if not stages:
        raise RuntimeError("r212 source decision has no factorized guide stage")
    for stage in stages:
        count = int(stage.options.num_words)
        target = int(getattr(stage, "guide_target_index", -1))
        confidence = float(getattr(stage, "guide_confidence", 0.0))
        if count < 1 or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise RuntimeError("r212 source has malformed guide stage")
        if target == -1:
            if confidence != 0.0:
                raise RuntimeError("r212 masked guide stage has nonzero confidence")
            continue
        if target < 0 or target >= count or count < 2 or confidence <= 0.0:
            raise RuntimeError("r212 source has incompatible labeled guide stage")
        labelled += 1
    return labelled


def _guide_rows_for_quarantine(sequence: Any) -> int:
    """Strictly count compact guide labels before a source row is skipped."""

    return sum(_guide_rows_for_decision(decision) for decision in sequence.decisions)


def _deck_fingerprint_for_quarantine(deck: Any) -> str:
    """Hash an exact 60-card multiset; no archetype-level approximation."""

    try:
        raw_cards = list(deck)
    except TypeError as exc:
        raise RuntimeError("r212 quarantine source deck is malformed") from exc
    if any(type(value) is not int for value in raw_cards):
        raise RuntimeError("r212 source deck contains a non-integer card id")
    cards = sorted(raw_cards)
    if len(cards) != 60:
        raise RuntimeError("r212 quarantine source deck is not an exact 60-card list")
    return "sha256:" + hashlib.sha256(
        json.dumps(cards, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _canonical_policy_stages(decision: Any) -> list[Any]:
    """Mirror the resident corpus's legal-stage fallback exactly."""

    stages = list(decision.policy_stages or ())
    if stages:
        return stages
    return [
        PolicyStage(
            options=decision.options,
            action_combos=decision.action_combos,
            target_index=decision.action_combo_index,
        )
    ]


def _valid_resident_stage_count(decision: Any) -> int:
    """Count exactly the policy samples ``DeviceResidentBootstrapCorpus`` keeps."""

    valid = 0
    for stage in _canonical_policy_stages(decision):
        count = int(stage.options.num_words)
        target = int(stage.target_index)
        if count > 0 and 0 <= target < count:
            valid += 1
    return valid


def _manifest_counter(value: Mapping[str, Any], key: str) -> int:
    raw = value.get(key)
    if type(raw) is bool or raw is None:
        raise RuntimeError(f"r212 split manifest lacks exact integer {key}")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"r212 split manifest has invalid integer {key}") from exc


def _quarantine_by_date(split_manifest: Mapping[str, Any]) -> dict[str, dict[tuple[str, int], QuarantineIdentity]]:
    """Parse the data-helper's exact, receipt-bound zero-label quarantine."""

    raw = split_manifest.get("quarantine")
    if not isinstance(raw, Mapping):
        raise RuntimeError("r212 split manifest lacks the mandatory quarantine receipt")
    if raw.get("policy") != QUARANTINE_POLICY:
        raise RuntimeError("r212 split quarantine policy drifted")
    identities = raw.get("identities")
    records = raw.get("records")
    if (
        not isinstance(identities, list)
        or not isinstance(records, list)
        or int(raw.get("expected_identity_count") or -1) != len(identities)
        or int(raw.get("identity_count") or -1) != len(identities)
        or len(records) != len(identities)
    ):
        raise RuntimeError("r212 split quarantine identity count is malformed")
    if len(identities) != EXPECTED_QUARANTINE_RECORDS:
        raise RuntimeError("r212 split quarantine count differs from the receipt-approved 11 rows")
    parsed: list[QuarantineIdentity] = []
    per_date: dict[str, dict[tuple[str, int], QuarantineIdentity]] = {}
    for identity, row in zip(identities, records, strict=True):
        if not isinstance(identity, Mapping) or not isinstance(row, Mapping):
            raise RuntimeError("r212 split quarantine identity/record is not an object")
        identity_source_date = str(identity.get("source_date") or "")
        identity_episode_id = str(identity.get("episode_id") or "")
        identity_seat = int(
            identity.get("seat") if identity.get("seat") is not None else -1
        )
        value = QuarantineIdentity(
            source_date=str(row.get("source_date") or ""),
            episode_id=str(row.get("episode_id") or ""),
            seat=int(row.get("seat") if row.get("seat") is not None else -1),
            decisions=int(row.get("decisions") if row.get("decisions") is not None else -1),
            guide_rows=int(row.get("guide_rows") if row.get("guide_rows") is not None else -1),
            deck_fingerprint=str(row.get("deck_fingerprint") or ""),
        )
        if (
            (value.source_date, value.episode_id, value.seat)
            != (identity_source_date, identity_episode_id, identity_seat)
            or set(identity) != {"source_date", "episode_id", "seat"}
            or set(row)
            != {
                "source_date",
                "episode_id",
                "seat",
                "decisions",
                "guide_rows",
                "deck_fingerprint",
            }
            or value.source_date not in ALL_DATES
            or not value.episode_id
            or value.seat not in (0, 1)
            or value.decisions <= 0
            or value.guide_rows != 0
            or not _SHA256_RE.fullmatch(value.deck_fingerprint)
        ):
            raise RuntimeError("r212 split quarantine identity contract is invalid")
        bucket = per_date.setdefault(value.source_date, {})
        if value.key in bucket:
            raise RuntimeError("r212 split quarantine repeats a source identity")
        bucket[value.key] = value
        parsed.append(value)
    if set(per_date) != {QUARANTINE_DATE}:
        raise RuntimeError("r212 split quarantine source-day scope drifted")
    if [
        (value.source_date, value.episode_id, value.seat) for value in parsed
    ] != sorted((value.source_date, value.episode_id, value.seat) for value in parsed):
        raise RuntimeError("r212 split quarantine identities are not canonically sorted")
    canonical = json.dumps(
        [
            {
                "source_date": value.source_date,
                "episode_id": value.episode_id,
                "seat": value.seat,
            }
            for value in parsed
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_digest = "sha256:" + hashlib.sha256(canonical).hexdigest()
    if (
        expected_digest != EXPECTED_QUARANTINE_IDENTITIES_SHA256
        or str(raw.get("identities_sha256") or "") != expected_digest
        or str(raw.get("expected_identities_sha256") or "") != expected_digest
    ):
        raise RuntimeError("r212 split quarantine identity digest mismatch")
    accounting = dict(split_manifest.get("accounting") or {})
    totals = dict(accounting.get("quarantined") or {})
    if (
        _manifest_counter(totals, "records") != len(parsed)
        or _manifest_counter(totals, "decisions") != sum(value.decisions for value in parsed)
        or _manifest_counter(totals, "guide_rows") != 0
    ):
        raise RuntimeError("r212 split quarantine aggregate mismatch")
    return per_date


class _VerifiedDayGames:
    """A raw-route-verified source day for one resident temporal pack.

    The protected compact corpus was written before the submitted r195 runtime
    enabled its trained V6 adapter bank.  Its persisted route column is thus
    not an admissible source of decode routes.  Reconstruct every source row
    from the header-bound raw archive first, then exclude only the separately
    receipt-approved zero-label quarantine rows before device packing.
    """

    def __init__(
        self,
        shard: SourceShard,
        *,
        max_context: int,
        quarantine: Mapping[tuple[str, int], QuarantineIdentity],
        adapter_route_binding: AdapterRouteBinding,
        raw_archive_root: Path,
        matchup_tree_path: Path,
        matchup_tree_sha256: str,
        runtime_route_code: RuntimeRouteCodeBinding,
        route_sidecar: RouteSidecar | None,
        limit: int = 0,
    ):
        self.shard = shard
        self.max_context = int(max_context)
        self.quarantine = dict(quarantine)
        self.adapter_route_binding = adapter_route_binding
        self.raw_archive_root = Path(raw_archive_root).expanduser().resolve()
        self.matchup_tree_path = Path(matchup_tree_path).expanduser().resolve()
        self.matchup_tree_sha256 = str(matchup_tree_sha256)
        self.runtime_route_code = runtime_route_code
        self.route_sidecar = route_sidecar
        self.limit = int(limit)
        if self.limit < 0:
            raise ValueError("source-day limit must be nonnegative")
        # DeviceResidentBootstrapCorpus retains the legal sample order but not
        # the public adapter route.  Record that aligned side channel while
        # yielding each feature sequence; it is used only to reproduce frozen
        # r195 decoding, never as a Guide2Vec input.
        self.sample_public_routes: list[int] = []
        self._iterated = False
        self._quarantine_seen: set[tuple[str, int]] = set()
        self._retained_records_seen = 0
        self._raw_route_reconstruction: dict[str, Any] | None = None
        self._route_reconstruction_mode = "sidecar" if route_sidecar is not None else "raw"
        self._turn_order_short_circuits = 0
        self._game_resets = 0
        self._turn_order_excluded_decisions = 0
        self._turn_order_excluded_policy_stages = 0
        self._turn_order_excluded_samples = 0
        self._turn_order_excluded_guide_rows = 0
        self._turn_order_exclusion_hasher = hashlib.sha256()
        # The guide accepts multiple exact Alakazam lists; preserve their full
        # 60-card distribution as telemetry rather than inventing a single-list
        # r195 allowlist.
        self._retained_deck_distribution: dict[str, dict[str, int]] = {}
        self._retained_guide_rows_seen = 0
        self._packed_guide_rows_seen = 0
        self._context_cap_excluded_decisions = 0
        self._context_cap_excluded_guide_rows = 0
        self._packed_records_seen = 0
        if self.shard.date in HELDOUT_DATES:
            if route_sidecar is None:
                raise RuntimeError(
                    "r212 heldout source requires an immutable verified public-route sidecar"
                )
            if route_sidecar.date != self.shard.date:
                raise RuntimeError("r212 heldout route sidecar date does not match its shard")
        elif route_sidecar is not None:
            raise RuntimeError(
                "r212 train/validation source must use header-bound raw route reconstruction"
            )

    def __len__(self) -> int:
        retained = self.shard.records - len(self.quarantine)
        if retained <= 0:
            raise RuntimeError("r212 quarantine removed an entire source day")
        return min(retained, self.limit) if self.limit else retained

    def _exclude_turn_order_short_circuits(
        self,
        sequence: Any,
        routes: tuple[int, ...],
        *,
        compact_short_circuit_env_steps: tuple[int, ...],
        raw_member_sha256: str,
    ) -> tuple[Any, tuple[int, ...]]:
        """Remove r195's IsFirst model-bypass decisions before causal packing.

        The submitted ``main.py`` resolves the turn-order choice before it
        invokes ``PolicyAgent``.  Keeping one of these compact decisions would
        both train an unavailable stage and advance temporal history by one
        token.  The raw resolver proves the exact compact env steps; this
        boundary removes the matching decision and route together, before the
        normal context-prefix cap is applied.
        """

        if not compact_short_circuit_env_steps:
            return sequence, routes
        if not _SHA256_RE.fullmatch(raw_member_sha256):
            raise RuntimeError("r212 route resolver lacks a raw-member identity")
        decisions = list(sequence.decisions)
        env_steps: list[int] = []
        for decision in decisions:
            if type(decision.env_step) is not int or int(decision.env_step) < 0:
                raise RuntimeError("r212 compact decision has an invalid environment step")
            env_steps.append(int(decision.env_step))
        if len(routes) != len(decisions) or len(set(env_steps)) != len(env_steps):
            raise RuntimeError("r212 compact decision/route environment-step alignment drifted")
        if (
            tuple(sorted(compact_short_circuit_env_steps))
            != compact_short_circuit_env_steps
            or len(set(compact_short_circuit_env_steps))
            != len(compact_short_circuit_env_steps)
        ):
            raise RuntimeError("r212 turn-order short-circuit steps are not canonical")
        short_steps = set(compact_short_circuit_env_steps)
        indexes = [index for index, step in enumerate(env_steps) if step in short_steps]
        if len(indexes) != len(short_steps):
            raise RuntimeError(
                "r212 raw turn-order proof references a compact decision that is absent"
            )
        retained_indexes = [
            index for index, step in enumerate(env_steps) if step not in short_steps
        ]
        if not retained_indexes:
            raise RuntimeError(
                "r212 source sequence contains only r195 turn-order bypass decisions"
            )

        def _filter_per_decision(value: Any, *, label: str) -> Any:
            if value is None:
                return None
            if len(value) != len(decisions):
                raise RuntimeError(
                    f"r212 {label} does not align with compact decisions before "
                    "turn-order filtering"
                )
            return [value[index] for index in retained_indexes]

        for index in indexes:
            valid_stages = _valid_resident_stage_count(decisions[index])
            exclusion = {
                "source_date": self.shard.date,
                "episode_id": str(sequence.episode_id or ""),
                "seat": int(sequence.seat),
                "env_step": env_steps[index],
                "raw_member_sha256": raw_member_sha256,
                "valid_policy_stages": valid_stages,
            }
            self._turn_order_exclusion_hasher.update(_canonical_json(exclusion))
            self._turn_order_excluded_decisions += 1
            self._turn_order_excluded_policy_stages += valid_stages
            # A valid resident policy stage maps one-to-one to a Guide2Vec
            # sample.  Keep both counts in the receipt to make this boundary
            # independently auditable.
            self._turn_order_excluded_samples += valid_stages
            self._turn_order_excluded_guide_rows += _guide_rows_for_decision(
                decisions[index]
            )
        return (
            dataclasses.replace(
                sequence,
                decisions=[decisions[index] for index in retained_indexes],
                policy_targets=_filter_per_decision(
                    sequence.policy_targets, label="policy targets"
                ),
                factorized_policy_targets=_filter_per_decision(
                    sequence.factorized_policy_targets,
                    label="factorized policy targets",
                ),
            ),
            tuple(routes[index] for index in retained_indexes),
        )

    def __iter__(self) -> Iterator[Any]:
        if self._iterated:
            raise RuntimeError("r212 source day route projection was iterated twice")
        self._iterated = True
        self.shard.assert_unchanged()
        emitted = 0
        retained_seen = 0
        # The raw resolver is intentionally called for *every* protected
        # sequence, including the exact quarantine records.  This keeps the
        # raw ZIP/member order and its projection accounting complete; only
        # the subsequent corpus yield excludes quarantines.
        if self.route_sidecar is None:
            route_resolver_context = RawPublicRouteResolver.open(
                source_date=self.shard.date,
                feature_shard_path=self.shard.path,
                feature_shard_sha256=self.shard.sha256,
                raw_archive_root=self.raw_archive_root,
                matchup_tree_path=self.matchup_tree_path,
                matchup_tree_sha256=self.matchup_tree_sha256,
                allowed_physical_slots=self.adapter_route_binding.runtime_accepted_slots,
                **self.runtime_route_code.raw_resolver_kwargs(),
            )
        else:
            route_resolver_context = SidecarPublicRouteResolver.open(
                sidecar_path=self.route_sidecar.path,
                source_date=self.shard.date,
                feature_shard_path=self.shard.path,
                feature_shard_sha256=self.shard.sha256,
                expected_raw_archive=self.route_sidecar.raw_archive,
                expected_producer_code=self.route_sidecar.producer_code,
                matchup_tree_sha256=self.matchup_tree_sha256,
                allowed_physical_slots=self.adapter_route_binding.runtime_accepted_slots,
                expected_sidecar_sha256=self.route_sidecar.sha256,
                **self.runtime_route_code.sidecar_resolver_kwargs(),
            )
        with route_resolver_context as route_resolver:
            for original_sequence in iter_feature_shard(self.shard.path):
                key = (str(original_sequence.episode_id or ""), int(original_sequence.seat))
                reconstructed_sequence = route_resolver.resolve_sequence(original_sequence)
                resolved_routes = tuple(reconstructed_sequence.routes)
                raw_member_sha256 = str(
                    getattr(reconstructed_sequence, "raw_member_sha256", "")
                )
                short_circuit_count = getattr(
                    reconstructed_sequence, "turn_order_short_circuits", None
                )
                short_circuit_steps = getattr(
                    reconstructed_sequence,
                    "turn_order_short_circuit_env_steps",
                    None,
                )
                game_resets = getattr(reconstructed_sequence, "game_resets", None)
                game_reset_steps = getattr(
                    reconstructed_sequence,
                    "game_reset_env_steps",
                    None,
                )
                if (
                    type(short_circuit_count) is not int
                    or short_circuit_count < 0
                    or not isinstance(short_circuit_steps, tuple)
                    or len(short_circuit_steps) > short_circuit_count
                    or tuple(sorted(short_circuit_steps)) != short_circuit_steps
                    or len(set(short_circuit_steps)) != len(short_circuit_steps)
                    or any(
                        type(step) is not int or step < 0
                        for step in short_circuit_steps
                    )
                    or not _SHA256_RE.fullmatch(raw_member_sha256)
                    or type(game_resets) is not int
                    or game_resets < 0
                    or not isinstance(game_reset_steps, tuple)
                    or len(game_reset_steps) != game_resets
                    or tuple(sorted(game_reset_steps)) != game_reset_steps
                    or len(set(game_reset_steps)) != len(game_reset_steps)
                    or any(type(step) is not int or step < 0 for step in game_reset_steps)
                ):
                    raise RuntimeError(
                        "r212 public-route resolver lacks exact turn-order short-circuit proof"
                    )
                original_env_steps = tuple(
                    int(decision.env_step) for decision in original_sequence.decisions
                )
                if (
                    not original_env_steps
                    or len(set(original_env_steps)) != len(original_env_steps)
                    or any(step not in original_env_steps for step in short_circuit_steps)
                    or any(
                        step >= min(original_env_steps)
                        for step in game_reset_steps
                    )
                ):
                    raise RuntimeError(
                        "r212 turn-order short-circuit proof does not align with compact "
                        "decision environment steps"
                    )
                self._turn_order_short_circuits += short_circuit_count
                self._game_resets += game_resets
                if len(resolved_routes) != len(original_sequence.decisions):
                    raise RuntimeError(
                        "r212 raw public-route reconstruction does not align with "
                        "the compact decision count"
                    )
                if any(
                    type(route) is not int
                    or (
                        route != UNKNOWN_ROUTE
                        and route not in self.adapter_route_binding.runtime_accepted_slots
                    )
                    for route in resolved_routes
                ):
                    raise RuntimeError(
                        "r212 raw public matchup route is not an exact "
                        "runtime-accepted r195 V6 physical slot"
                    )
                quarantined = self.quarantine.get(key)
                if quarantined is not None:
                    if key in self._quarantine_seen:
                        raise RuntimeError("r212 quarantine source identity was duplicated")
                    if (
                        len(original_sequence.decisions) != quarantined.decisions
                        or _guide_rows_for_quarantine(original_sequence)
                        != quarantined.guide_rows
                        or is_alakazam_deck(original_sequence.deck)
                        or _deck_fingerprint_for_quarantine(original_sequence.deck)
                        != quarantined.deck_fingerprint
                    ):
                        raise RuntimeError(
                            "r212 quarantined source record no longer matches its zero-label "
                            "incompatible-deck receipt"
                        )
                    self._quarantine_seen.add(key)
                    continue
                # The canonical split helper proves this for every retained row;
                # repeat it at packing time so an unknown incompatible record can
                # never be converted into an eligibility/coverage negative.
                deck_fingerprint = _deck_fingerprint_for_quarantine(original_sequence.deck)
                guide_rows = _guide_rows_for_quarantine(original_sequence)
                if not is_alakazam_deck(original_sequence.deck):
                    raise RuntimeError(
                        "r212 retained source has an unquarantined incompatible deck"
                    )
                distribution = self._retained_deck_distribution.setdefault(
                    deck_fingerprint,
                    {"records": 0, "decisions": 0, "guide_rows": 0},
                )
                distribution["records"] += 1
                distribution["decisions"] += len(original_sequence.decisions)
                distribution["guide_rows"] += guide_rows
                self._retained_guide_rows_seen += guide_rows
                retained_seen += 1
                if self.limit and emitted >= self.limit:
                    # Resolve it anyway for exact raw-prefix provenance, but do
                    # not let smoke rows enter the resident device pack.
                    continue
                sequence, resolved_routes = self._exclude_turn_order_short_circuits(
                    original_sequence,
                    resolved_routes,
                    compact_short_circuit_env_steps=short_circuit_steps,
                    raw_member_sha256=raw_member_sha256,
                )
                pre_cap_sequence = sequence
                if len(sequence.decisions) > self.max_context:
                    context_tail = sequence.decisions[self.max_context :]
                    self._context_cap_excluded_decisions += len(context_tail)
                    self._context_cap_excluded_guide_rows += sum(
                        _guide_rows_for_decision(decision)
                        for decision in context_tail
                    )
                    sequence, _ = cap_game_sequence_context(sequence, self.max_context)
                    resolved_routes = resolved_routes[: len(sequence.decisions)]
                if (
                    tuple(decision.env_step for decision in sequence.decisions)
                    != tuple(
                        decision.env_step
                        for decision in pre_cap_sequence.decisions[: len(sequence.decisions)]
                    )
                ):
                    raise RuntimeError(
                        "r212 context cap changed the verified raw-route decision prefix"
                    )
                if (
                    sequence.policy_targets is not None
                    or sequence.factorized_policy_targets is not None
                ):
                    raise RuntimeError("r212 source day has unsupported soft policy targets")
                self._packed_guide_rows_seen += _guide_rows_for_quarantine(sequence)
                for decision_index, decision in enumerate(sequence.decisions):
                    route = resolved_routes[decision_index]
                    for _ in range(_valid_resident_stage_count(decision)):
                        self.sample_public_routes.append(int(route))
                emitted += 1
                self._packed_records_seen += 1
                yield sequence
            self._raw_route_reconstruction = dict(
                route_resolver.projection(expected_records=self.shard.records)
            )
        if self._quarantine_seen != set(self.quarantine):
            missing = sorted(set(self.quarantine).difference(self._quarantine_seen))
            raise RuntimeError(f"r212 quarantine identities missing from source shard: {missing}")
        if retained_seen != self.shard.records - len(self.quarantine):
            raise RuntimeError(
                f"r212 retained source record count drift: {self.shard.path} "
                f"expected={self.shard.records - len(self.quarantine)} actual={retained_seen}"
            )
        if (
            sum(row["records"] for row in self._retained_deck_distribution.values())
            != retained_seen
            or sum(row["guide_rows"] for row in self._retained_deck_distribution.values())
            != self._retained_guide_rows_seen
            or any(
                set(row) != {"records", "decisions", "guide_rows"}
                or min(row.values()) < 0
                for row in self._retained_deck_distribution.values()
            )
        ):
            raise RuntimeError("r212 retained exact-deck telemetry accounting drifted")
        if emitted != len(self):
            raise RuntimeError("r212 retained smoke/source limit drifted during packing")
        self._retained_records_seen = retained_seen
        self.shard.assert_unchanged()

    def quarantine_projection(self) -> dict[str, Any]:
        if self._retained_records_seen != self.shard.records - len(self.quarantine):
            raise RuntimeError("r212 quarantine projection requested before source verification")
        identities = [self.quarantine[key].as_dict() for key in sorted(self.quarantine)]
        return {
            "schema": "poke_bot.alakazam_guide2vec_r212_quarantine_projection/v1",
            "date": self.shard.date,
            "records": len(identities),
            "retained_records": self._retained_records_seen,
            "identity_sha256": "sha256:"
            + hashlib.sha256(
                json.dumps(
                    identities,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "identities": identities,
            "never_packed_as_coverage_negatives": True,
        }

    def route_tensor(self, *, device: torch.device, expected_samples: int) -> torch.Tensor:
        if self._raw_route_reconstruction is None:
            raise RuntimeError("r212 public routes were not reconstructed from raw archives")
        if len(self.sample_public_routes) != int(expected_samples):
            raise RuntimeError(
                "r212 public adapter-route projection does not align with resident samples: "
                f"routes={len(self.sample_public_routes)} samples={expected_samples}"
            )
        return torch.tensor(self.sample_public_routes, dtype=torch.long, device=device)

    def route_projection(self, *, expected_samples: int) -> dict[str, Any]:
        reconstructed = self._raw_route_reconstruction
        if not isinstance(reconstructed, Mapping):
            raise RuntimeError("r212 raw public-route reconstruction receipt is missing")
        if len(self.sample_public_routes) != int(expected_samples):
            raise RuntimeError("r212 public adapter-route projection is incomplete")
        if self._packed_records_seen != len(self):
            raise RuntimeError("r212 packed game accounting drifted after route filtering")
        if self._turn_order_excluded_decisions > self._turn_order_short_circuits:
            raise RuntimeError("r212 turn-order exclusion accounting exceeds raw proof")
        if self._retained_records_seen != self.shard.records - len(self.quarantine):
            raise RuntimeError("r212 retained source receipt is incomplete")
        deck_distribution = [
            {"deck_fingerprint": fingerprint, **counts}
            for fingerprint, counts in sorted(self._retained_deck_distribution.items())
        ]
        if (
            not deck_distribution
            or sum(int(row["records"]) for row in deck_distribution)
            != self._retained_records_seen
            or sum(int(row["guide_rows"]) for row in deck_distribution)
            != self._retained_guide_rows_seen
        ):
            raise RuntimeError("r212 exact-deck telemetry cannot be sealed")
        if self.limit == 0 and self._retained_guide_rows_seen != (
            self._packed_guide_rows_seen
            + self._turn_order_excluded_guide_rows
            + self._context_cap_excluded_guide_rows
        ):
            raise RuntimeError(
                "r212 full source guide-row accounting does not match its model "
                "history boundaries"
            )
        deck_telemetry = {
            "schema": "poke_bot.alakazam_guide2vec_r212_exact_deck_telemetry/v1",
            "policy": (
                "all_exact_teacher_compatible_alakazam_60_card_multisets_not_"
                "single_r195_evaluation_deck_allowlist"
            ),
            "card_count": 60,
            "teacher_label_source": "protected_r212_compact_guide_target_index_confidence",
            "retained_records": self._retained_records_seen,
            "retained_guide_rows": self._retained_guide_rows_seen,
            "packed_records": self._packed_records_seen,
            "packed_guide_rows": self._packed_guide_rows_seen,
            "context_cap_excluded_decisions": self._context_cap_excluded_decisions,
            "context_cap_excluded_guide_rows": self._context_cap_excluded_guide_rows,
            "distinct_fingerprints": len(deck_distribution),
            "distribution_sha256": "sha256:"
            + hashlib.sha256(_canonical_json(deck_distribution)).hexdigest(),
            "distribution": deck_distribution,
        }
        values = torch.tensor(self.sample_public_routes, dtype=torch.int32)
        digest = hashlib.sha256(values.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        turn_order_exclusions = {
            "schema": (
                "poke_bot.alakazam_guide2vec_r212_turn_order_short_circuit_exclusions/v1"
            ),
            "packed_records": self._packed_records_seen,
            "excluded_decisions": self._turn_order_excluded_decisions,
            "excluded_policy_stages": self._turn_order_excluded_policy_stages,
            "excluded_samples": self._turn_order_excluded_samples,
            "excluded_guide_rows": self._turn_order_excluded_guide_rows,
            "identity_sha256": "sha256:" + self._turn_order_exclusion_hasher.hexdigest(),
            "admitted_to_model_history": False,
        }
        raw_archive = reconstructed.get("raw_archive")
        expected_sidecar = self.route_sidecar
        if (
            reconstructed.get("schema") != ROUTE_RECONSTRUCTION_SCHEMA
            or reconstructed.get("source_date") != self.shard.date
            or reconstructed.get("source_feature_shard_sha256") != self.shard.sha256
            or reconstructed.get("runtime_public_tree_sha256") != self.matchup_tree_sha256
            or reconstructed.get("allowed_physical_slots")
            != sorted(self.adapter_route_binding.runtime_accepted_slots)
            or int(reconstructed.get("records") or -1) != self.shard.records
            or int(reconstructed.get("decisions") or 0) <= 0
            or type(reconstructed.get("turn_order_short_circuits")) is not int
            or int(reconstructed.get("turn_order_short_circuits"))
            != self._turn_order_short_circuits
            or type(reconstructed.get("game_resets")) is not int
            or int(reconstructed.get("game_resets")) != self._game_resets
            or int(reconstructed.get("routed_decisions") or 0)
            + int(reconstructed.get("bypassed_decisions") or 0)
            != int(reconstructed.get("decisions") or -1)
            or reconstructed.get("compact_source_routes_ignored") is not True
            or reconstructed.get("oracle_route_used") is not False
            or dict(reconstructed.get("runtime_code") or {})
            != self.runtime_route_code.as_dict()
            or not isinstance(raw_archive, Mapping)
            or not _SHA256_RE.fullmatch(str(raw_archive.get("sha256") or ""))
            or int(raw_archive.get("bytes") or 0) <= 0
            or not _SHA256_RE.fullmatch(
                str(reconstructed.get("member_route_sha256") or "")
            )
            or (
                expected_sidecar is None
                and "sidecar" in reconstructed
            )
            or (
                expected_sidecar is not None
                and dict(reconstructed.get("sidecar") or {})
                != {
                    "path": str(expected_sidecar.path),
                    "sha256": expected_sidecar.sha256,
                }
            )
            or (
                expected_sidecar is not None
                and dict(raw_archive or {}) != dict(expected_sidecar.raw_archive)
            )
            or (
                expected_sidecar is not None
                and dict(reconstructed.get("producer_code") or {})
                != dict(expected_sidecar.producer_code)
            )
        ):
            raise RuntimeError("r212 raw public-route reconstruction receipt drifted")
        return {
            "schema": "poke_bot.alakazam_guide2vec_r212_public_adapter_route_projection/v1",
            "sha256": "sha256:" + digest,
            "samples": int(expected_samples),
            "routed_samples": int((values != UNKNOWN_ROUTE).sum().item()),
            "bypassed_samples": int((values == UNKNOWN_ROUTE).sum().item()),
            "source": "r195_raw_public_matchup_route_reconstruction",
            "route_reconstruction_mode": self._route_reconstruction_mode,
            "compact_source_routes_ignored": True,
            "turn_order_short_circuits": self._turn_order_short_circuits,
            "game_resets": self._game_resets,
            "game_reset_history_spans_compact_temporal_sequence": False,
            "compact_turn_order_short_circuits_admitted": False,
            "compact_turn_order_short_circuit_exclusions": turn_order_exclusions,
            "retained_deck_fingerprint_telemetry": deck_telemetry,
            "oracle_route_used": False,
            "adapter_route_binding": self.adapter_route_binding.as_dict(),
            "raw_route_reconstruction": dict(reconstructed),
        }


def _resolve_manifest_path(pointer_path: Path, pointer: Mapping[str, Any]) -> Path:
    raw = str(pointer.get("manifest") or "").strip()
    if not raw:
        raise RuntimeError("protected corpus pointer has no manifest")
    candidate = Path(raw).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (pointer_path.parent / candidate).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _verify_protected_corpus(pointer_path: Path) -> tuple[Path, dict[str, Any], dict[str, SourceShard]]:
    pointer_path = pointer_path.expanduser().resolve()
    if not pointer_path.is_file():
        raise FileNotFoundError(pointer_path)
    if _sha256(pointer_path) != PROTECTED_POINTER_SHA256:
        raise RuntimeError("r212 protected corpus pointer SHA-256 mismatch")
    pointer = _read_json_object(pointer_path)
    if (
        pointer.get("schema") != "poke_bot.pinned_expert_corpus/v1"
        or pointer.get("protected") is not True
        or str(pointer.get("manifest_sha256") or "") != MANIFEST_SHA256
    ):
        raise RuntimeError("r212 protected corpus pointer contract mismatch")
    manifest_path = _resolve_manifest_path(pointer_path, pointer)
    if _sha256(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("r212 protected corpus manifest SHA-256 mismatch")
    manifest = _read_json_object(manifest_path)
    if (
        manifest.get("format") != "pokebot-bootstrap-feature-manifest"
        or str(manifest.get("compact_mode") or "") != COMPACT_MODE_TEMPORAL_EXPERT
        or tuple(str(value) for value in manifest.get("dates") or ()) != ALL_DATES
        or str((manifest.get("selection") or {}).get("value") or "").casefold()
        != "alakazam"
        or (manifest.get("selection") or {}).get("seat_semantics") != "acting_seat_only"
        or (manifest.get("quality_gates") or {}).get("passed") is not True
        or (manifest.get("quality_gates") or {}).get("hidden_targets_are_aux_only")
        is not True
    ):
        raise RuntimeError("r212 protected corpus manifest contract mismatch")
    if int(manifest.get("max_context") or 0) <= 0:
        raise RuntimeError("r212 manifest has no positive temporal context")

    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != len(ALL_DATES):
        raise RuntimeError("r212 manifest does not contain one shard per source day")
    shards: dict[str, SourceShard] = {}
    total_records = total_decisions = total_bytes = 0
    for raw in raw_shards:
        if not isinstance(raw, dict):
            raise RuntimeError("r212 manifest shard is not an object")
        dates = tuple(str(value) for value in raw.get("source_dates") or ())
        if len(dates) != 1 or dates[0] not in ALL_DATES or dates[0] in shards:
            raise RuntimeError("r212 source-day shard layout is ambiguous")
        date = dates[0]
        digest = str(raw.get("sha256") or "")
        if not _SHA256_RE.fullmatch(digest):
            raise RuntimeError(f"r212 shard has invalid SHA-256: {date}")
        raw_path = str(raw.get("path") or "")
        if not raw_path:
            raise RuntimeError(f"r212 shard path missing: {date}")
        path = (manifest_path.parent / raw_path).resolve()
        try:
            path.relative_to(manifest_path.parent.resolve())
        except ValueError as exc:
            raise RuntimeError(f"r212 shard escaped manifest directory: {date}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        byte_count = int(raw.get("bytes") or -1)
        if byte_count <= 0 or path.stat().st_size != byte_count:
            raise RuntimeError(f"r212 shard byte-count mismatch: {date}")
        if _sha256(path) != digest:
            raise RuntimeError(f"r212 protected shard SHA-256 mismatch: {date}")
        stats = dict(raw.get("stats") or {})
        records = int(stats.get("records_kept") or -1)
        decisions = int(stats.get("decisions_kept") or -1)
        if records <= 0 or decisions <= 0:
            raise RuntimeError(f"r212 shard statistics invalid: {date}")
        if str(raw.get("compact_mode") or "") != COMPACT_MODE_TEMPORAL_EXPERT:
            raise RuntimeError(f"r212 shard compact mode mismatch: {date}")
        if int(raw.get("max_context") or 0) != int(manifest["max_context"]):
            raise RuntimeError(f"r212 shard temporal context mismatch: {date}")
        shards[date] = SourceShard(
            date=date,
            path=path,
            sha256=digest,
            byte_count=byte_count,
            records=records,
            decisions=decisions,
            stat_identity=_stat_identity(path),
        )
        total_records += records
        total_decisions += decisions
        total_bytes += byte_count
    if tuple(sorted(shards)) != ALL_DATES:
        raise RuntimeError("r212 source-day set changed after shard verification")
    totals = dict(manifest.get("totals") or {})
    if (
        int(totals.get("records_kept") or -1) != total_records
        or int(totals.get("decisions_kept") or -1) != total_decisions
        or int(totals.get("bytes") or -1) != total_bytes
    ):
        raise RuntimeError("r212 manifest totals do not match verified shards")
    pointer_totals = dict(pointer.get("totals") or {})
    if (
        int(pointer_totals.get("records_kept") or -1) != total_records
        or int(pointer_totals.get("decisions_kept") or -1) != total_decisions
    ):
        raise RuntimeError("r212 protected pointer totals do not match manifest")
    return manifest_path, manifest, shards


def _verify_exact_file(path: Path, expected_sha256: str, *, label: str) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = _sha256(resolved)
    if observed != expected_sha256:
        raise RuntimeError(
            f"r212 {label} SHA-256 mismatch: expected={expected_sha256} actual={observed}"
        )
    return {"path": str(resolved), "sha256": observed}


def _verify_r195_route_runtime_bundle(path: Path) -> RuntimeRouteCodeBinding:
    """Verify the exact submitted tar members that define router invocation.

    The working-tree entrypoint is not acceptable provenance: r195's submitted
    ``main.py`` handles ``IsFirst`` before PolicyAgent.  Bind the uploaded
    NO-RTP bundle itself and hash the two exact member streams before route
    materialization or GPU allocation.
    """

    resolved = Path(path).expanduser().resolve()
    _verify_exact_file(
        resolved,
        R195_BUNDLE_SHA256,
        label="r195 NO-RTP submission bundle",
    )
    before = _stat_identity(resolved)
    required = {
        R195_RUNTIME_ENTRYPOINT_MEMBER: R195_RUNTIME_ENTRYPOINT_SHA256,
        R195_RUNTIME_ROUTER_MEMBER: R195_RUNTIME_ROUTER_SHA256,
    }
    try:
        with tarfile.open(resolved, "r:*") as archive:
            members = archive.getmembers()
            for name, expected_sha256 in required.items():
                matches = [member for member in members if member.name == name]
                if len(matches) != 1 or not matches[0].isfile():
                    raise RuntimeError(
                        f"r212 r195 submission bundle lacks exactly one regular {name}"
                    )
                handle = archive.extractfile(matches[0])
                if handle is None:
                    raise RuntimeError(
                        f"r212 r195 submission bundle cannot read {name}"
                    )
                digest = hashlib.sha256()
                with handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                observed = "sha256:" + digest.hexdigest()
                if observed != expected_sha256:
                    raise RuntimeError(
                        f"r212 r195 submission member SHA-256 mismatch: {name}"
                    )
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError("r212 r195 submission bundle is not a readable tar archive") from exc
    if _stat_identity(resolved) != before or _sha256(resolved) != R195_BUNDLE_SHA256:
        raise RuntimeError("r212 r195 submission bundle changed during runtime-code verification")
    return RuntimeRouteCodeBinding(
        submission_bundle_path=resolved,
        submission_bundle_sha256=R195_BUNDLE_SHA256,
        entrypoint_member=R195_RUNTIME_ENTRYPOINT_MEMBER,
        entrypoint_sha256=R195_RUNTIME_ENTRYPOINT_SHA256,
        router_member=R195_RUNTIME_ROUTER_MEMBER,
        router_sha256=R195_RUNTIME_ROUTER_SHA256,
    )


def _verify_r212_typed_dependencies(owner_contract: Path) -> dict[str, dict[str, str]]:
    """Bind the owner, r195, teacher, and source-ready artifacts before training."""

    identities = {
        "owner_contract": _verify_exact_file(
            owner_contract, OWNER_CONTRACT_SHA256, label="owner contract"
        ),
        "r195_contract": _verify_exact_file(
            R195_CONTRACT_PATH, R195_CONTRACT_SHA256, label="r195 contract"
        ),
        "guide_contract": _verify_exact_file(
            GUIDE_CONTRACT_PATH, GUIDE_CONTRACT_SHA256, label="guide contract"
        ),
        "guide_ready_receipt": _verify_exact_file(
            GUIDE_READY_RECEIPT_PATH,
            GUIDE_READY_RECEIPT_SHA256,
            label="guide source-ready receipt",
        ),
        "guide_teacher": _verify_exact_file(
            GUIDE_TEACHER_PATH, GUIDE_TEACHER_SHA256, label="guide teacher module"
        ),
    }
    owner = _read_json_object(Path(identities["owner_contract"]["path"]))
    frozen = dict(owner.get("frozen_base") or {})
    teacher = dict(owner.get("guide_teacher_and_data") or {})
    head = dict(owner.get("guide2vec_head") or {})
    runtime = dict(owner.get("candidate_runtime") or {})
    if (
        owner.get("schema") != "poke_bot.alakazam_guide2vec_no_mcts_bo1000_r212/v1"
        or int(owner.get("owner_decision_revision") or -1) != 212
        or str(frozen.get("r195_contract_path") or "") != str(R195_CONTRACT_PATH.relative_to(ROOT))
        or str(frozen.get("r195_contract_sha256") or "") != R195_CONTRACT_SHA256
        or str(frozen.get("checkpoint_sha256") or "") != R195_CHECKPOINT_SHA256
        or int(frozen.get("checkpoint_bytes") or -1) != R195_CHECKPOINT_BYTES
        or str(teacher.get("guide_contract_path") or "") != str(GUIDE_CONTRACT_PATH.relative_to(ROOT))
        or str(teacher.get("teacher_module_path") or "") != str(GUIDE_TEACHER_PATH.relative_to(ROOT))
        or str(teacher.get("source_ready_receipt") or "") != str(GUIDE_READY_RECEIPT_PATH.relative_to(ROOT))
        or int(head.get("maximum_epochs") or -1) != MAX_EPOCHS
        or int(head.get("parameter_count_max") or -1) != MAX_GUIDE2VEC_PARAMETERS
        or head.get("selection_metric") != SELECTION_METRIC
        or float(runtime.get("maximum_logit_bonus") or -1.0) != MAX_GUIDE_LOGIT_BONUS
        or runtime.get("mcts_expectimax_rollout_recursive_turn_planner_rtp_or_simulator_leaf_reranking_allowed")
        is not False
    ):
        raise RuntimeError("r212 typed owner dependency contract drifted")
    return identities


def _managed_snapshot_required() -> bool:
    """Return whether this process is the sealed managed r212 job."""

    raw = os.environ.get("POKEBOT_GUIDE2VEC_R212_ISOLATED", "")
    if raw not in {"", "0", "1"}:
        raise RuntimeError("POKEBOT_GUIDE2VEC_R212_ISOLATED must be 0 or 1")
    return raw == "1"


def _assert_combo_state_route_disabled_environment() -> None:
    """Pin the non-architectural combo policy route off for exact r195 parity."""

    raw = os.environ.get("POKEBOT_COMBO_STATE_ROUTE_ENABLED", "")
    if raw not in {"", "0"}:
        raise RuntimeError("r212 requires POKEBOT_COMBO_STATE_ROUTE_ENABLED=0")
    if _managed_snapshot_required() and raw != "0":
        raise RuntimeError("r212 managed runtime must explicitly pin POKEBOT_COMBO_STATE_ROUTE_ENABLED=0")


def _validated_source_snapshot() -> dict[str, Any]:
    """Bind a managed run to the immutable source tree that executes it."""

    manifest_path = ROOT / SOURCE_SNAPSHOT_MANIFEST
    required = _managed_snapshot_required()
    if not manifest_path.is_file():
        if required:
            raise RuntimeError("r212 managed run requires a published source snapshot manifest under ROOT")
        return {
            "schema": SOURCE_SNAPSHOT_SCHEMA,
            "status": "unpublished_local_development_only",
            "required_for_this_run": False,
            "root": str(ROOT),
            "trainer_sha256": _sha256(Path(__file__).resolve()),
        }
    try:
        from scripts.stage_alakazam_guide2vec_r212_source_snapshot import (
            validate_published_root,
        )

        verified = validate_published_root(ROOT)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("r212 source snapshot manifest/inventory verification failed") from exc
    if (
        verified.get("schema") != SOURCE_SNAPSHOT_SCHEMA
        or verified.get("status") != "valid"
        or verified.get("published_root") != str(ROOT)
        or not _SHA256_RE.fullmatch(str(verified.get("source_tree_sha256") or ""))
        or not _SHA256_RE.fullmatch(str(verified.get("manifest_sha256") or ""))
    ):
        raise RuntimeError("r212 source snapshot identity is malformed")
    return {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "status": "validated_published_snapshot",
        "required_for_this_run": required,
        "root": str(ROOT),
        "manifest_path": str(manifest_path),
        "manifest_sha256": str(verified["manifest_sha256"]),
        "source_tree_sha256": str(verified["source_tree_sha256"]),
        "rendered_unit_sha256": str(verified.get("rendered_unit_sha256") or ""),
    }


def _validate_r195_checkpoint(path: Path) -> tuple[dict[str, Any], int, str, int]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    byte_count = int(path.stat().st_size)
    if byte_count != R195_CHECKPOINT_BYTES:
        raise RuntimeError(
            "r212 r195 checkpoint byte-count mismatch: "
            f"expected={R195_CHECKPOINT_BYTES} actual={byte_count}"
        )
    digest = _sha256(path)
    if digest != R195_CHECKPOINT_SHA256:
        raise RuntimeError("r212 r195 checkpoint SHA-256 mismatch")
    payload = checkpoint.load_checkpoint(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError("r212 r195 checkpoint payload is not an object")
    profile = dict(payload.get("model_config") or {})
    if (
        str(payload.get("archetype_id") or "").casefold() != "alakazam"
        or str(profile.get("decision_context") or "") != "history"
        or bool(profile.get("h10_capacity_enabled")) is not True
        or bool(profile.get("decision_fusion_enabled")) is not True
    ):
        raise RuntimeError("r212 parent is not the exact temporal H10 Alakazam r195 model")
    if bool(profile.get("combo_state_route_enabled", True)):
        raise RuntimeError("r212 parent unexpectedly has the combo policy route enabled")
    max_context = int(profile.get("max_context") or 0)
    d_model = int(profile.get("d_model") or 0)
    if max_context <= 0 or d_model != EXPECTED_D_MODEL:
        raise RuntimeError(
            "r212 parent model shape mismatch: "
            f"max_context={max_context} d_model={d_model}"
        )
    return payload, byte_count, digest, max_context


def _resolve_r195_v6_adapter_route_binding(
    adapter_identity: Mapping[str, Any],
) -> AdapterRouteBinding:
    """Resolve the checkpoint's immutable V6 route-to-slot mapping.

    Public trees emit physical slots, not the legacy V5 logical roster index.
    Do not replace this with a range check: retired/unallocated V6 slots are an
    exact bypass in the bank and admitting them here would silently change the
    submitted runtime graph.
    """

    adapter_config = adapter_identity.get("adapter_config")
    if not isinstance(adapter_config, Mapping):
        raise RuntimeError("r212 r195 checkpoint has no matchup adapter config")
    try:
        from poke_bot.matchup_adapter_routes import (
            resolve_matchup_adapter_route_contract,
        )
        from poke_bot.matchup_adapters_v6 import (
            ADAPTER_CHECKPOINT_FORMAT as V6_ADAPTER_CHECKPOINT_FORMAT,
            SLOT_CAPACITY as V6_SLOT_CAPACITY,
        )

        contract = resolve_matchup_adapter_route_contract(adapter_config)
    except (ImportError, TypeError, ValueError) as exc:
        raise RuntimeError("r212 cannot resolve the exact r195 V6 adapter route contract") from exc
    if (
        contract.adapter_format != V6_ADAPTER_CHECKPOINT_FORMAT
        or int(contract.slot_capacity) != V6_SLOT_CAPACITY
        or int(contract.slot_capacity) != 64
        or not contract.target_ids
        or len(contract.target_ids) != len(contract.physical_slots)
        or len(set(contract.target_ids)) != len(contract.target_ids)
        or len(set(contract.physical_slots)) != len(contract.physical_slots)
        or any(slot < 0 or slot >= int(contract.slot_capacity) for slot in contract.physical_slots)
        or not isinstance(contract.slot_registry_digest, str)
        or _SHA256_RE.fullmatch(contract.slot_registry_digest) is None
    ):
        raise RuntimeError("r212 r195 adapter is not the required 64-slot V6 bank")
    return AdapterRouteBinding(
        adapter_format=str(contract.adapter_format),
        target_ids=tuple(str(value) for value in contract.target_ids),
        physical_slots=tuple(int(value) for value in contract.physical_slots),
        slot_capacity=int(contract.slot_capacity),
        slot_registry_digest=contract.slot_registry_digest,
    )


def _validate_r195_runtime_matchup_tree(
    path: Path,
    *,
    adapter_identity: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    route_binding: AdapterRouteBinding,
) -> tuple[Path, str, AdapterRouteBinding]:
    """Verify the exact enabled public tree against the checkpoint V6 contract."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = _sha256(resolved)
    if digest != R195_MATCHUP_TREE_SHA256:
        raise RuntimeError("r212 r195 public matchup tree SHA-256 mismatch")
    adapter_config = adapter_identity.get("adapter_config")
    if not isinstance(adapter_config, Mapping):
        raise RuntimeError("r212 cannot bind a tree without checkpoint adapter config")
    try:
        from poke_bot.matchup_adapter_routes import (
            require_runtime_route_binding,
            resolve_matchup_adapter_route_contract,
        )
        from poke_bot.public_matchup_router import PublicMatchupDecisionTree

        payload = _read_json_object(resolved)
        runtime_contract = payload.get("runtime_contract")
        if not isinstance(runtime_contract, Mapping):
            raise ValueError("tree has no runtime contract")
        checkpoint_routes = resolve_matchup_adapter_route_contract(adapter_config)
        require_runtime_route_binding(
            runtime_contract,
            checkpoint_routes,
            allow_legacy_v5=False,
        )
        tree = PublicMatchupDecisionTree.from_path(
            resolved,
            require_runtime_enabled=True,
        )
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError("r212 r195 public matchup tree is not a valid enabled V6 runtime") from exc
    if (
        tree.digest != digest
        or tree.runtime_enabled is not True
        or tree.adapter_format != route_binding.adapter_format
        or tuple(tree.targets) != route_binding.target_ids
        or tuple(tree.route_physical_slots) != route_binding.physical_slots
        or tree.slot_registry_digest != route_binding.slot_registry_digest
    ):
        raise RuntimeError("r212 public matchup tree differs from r195 checkpoint V6 routing")
    accepted_targets = tuple(
        target
        for target in route_binding.target_ids
        if target in tree.runtime_accepted_archetype_ids
    )
    target_to_slot = dict(
        zip(route_binding.target_ids, route_binding.physical_slots, strict=True)
    )
    accepted_slots = tuple(target_to_slot[target] for target in accepted_targets)
    if (
        not accepted_targets
        or "alakazam" not in accepted_targets
        or len(accepted_targets) != len(tree.runtime_accepted_archetype_ids)
        or len(accepted_slots) != len(set(accepted_slots))
        or any(slot not in route_binding.physical_slots for slot in accepted_slots)
    ):
        raise RuntimeError("r212 public matchup tree has an invalid runtime-accepted V6 route set")
    state = checkpoint_payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("r212 r195 checkpoint has no model state for adapter validation")
    for target, slot in zip(accepted_targets, accepted_slots, strict=True):
        outputs = (
            state.get(f"matchup_adapter_bank.experts.{slot}.up.weight"),
            state.get(f"matchup_adapter_bank.experts.{slot}.up.bias"),
        )
        if not all(isinstance(value, torch.Tensor) for value in outputs):
            raise RuntimeError(
                f"r212 r195 accepted adapter output tensor is missing: {target}@{slot}"
            )
        if not all(bool(torch.isfinite(value).all().item()) for value in outputs):
            raise RuntimeError(
                f"r212 r195 accepted adapter output tensor is non-finite: {target}@{slot}"
            )
        if not any(int(value.count_nonzero().item()) > 0 for value in outputs):
            raise RuntimeError(
                f"r212 r195 tree accepts an exact-zero adapter slot: {target}@{slot}"
            )
    return (
        resolved,
        digest,
        dataclasses.replace(
            route_binding,
            runtime_accepted_target_ids=accepted_targets,
            runtime_accepted_physical_slots=accepted_slots,
            runtime_accepted_nonzero_output_slots=accepted_slots,
        ),
    )


def _assert_blackwell_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("r212 training requires an available CUDA Blackwell device")
    torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device)
    properties = torch.cuda.get_device_properties(device)
    if "blackwell" not in name.casefold():
        raise RuntimeError(f"r212 requires Blackwell, got CUDA device {name!r}")
    if int(properties.total_memory) < BLACKWELL_MIN_MEMORY_BYTES:
        raise RuntimeError(
            "r212 CUDA device has insufficient VRAM: "
            f"{int(properties.total_memory)} < {BLACKWELL_MIN_MEMORY_BYTES}"
        )
    # The service pins CUDA_VISIBLE_DEVICES to the UUID, so logical cuda:0 is
    # unambiguous.  When started manually, require the same UUID or a single
    # visible Blackwell from nvidia-smi rather than guessing an ordinal.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and BLACKWELL_UUID not in visible.split(","):
        raise RuntimeError("r212 CUDA_VISIBLE_DEVICES is not pinned to the Blackwell UUID")
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("r212 cannot verify the Blackwell GPU UUID") from exc
    rows = [tuple(part.strip() for part in line.split(",")) for line in output.splitlines() if line.strip()]
    exact = [row for row in rows if row and row[0] == BLACKWELL_UUID]
    if len(exact) != 1 or "blackwell" not in exact[0][1].casefold():
        raise RuntimeError("r212 expected Blackwell GPU UUID is absent or ambiguous")


def _linux_mem_available_bytes() -> int:
    """Return Linux MemAvailable without treating unrelated CPU work as a conflict."""

    path = Path("/proc/meminfo")
    if not path.is_file():
        return 0
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "MemAvailable:":
            return int(fields[1]) * 1024
    return 0


def _systemd_user_state() -> dict[str, str]:
    """Read the dedicated unit state without starting, stopping, or resetting it."""

    unit = "pokebot-alakazam-guide2vec-r212.service"
    try:
        output = subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "--property=LoadState,ActiveState,SubState,MainPID,ControlPID",
                "--no-pager",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # A direct developer smoke may run before the unit is installed.  The
        # managed service always has this unit, so it is required there.
        if os.environ.get("POKEBOT_GUIDE2VEC_R212_ISOLATED") == "1":
            raise RuntimeError("r212 managed unit cannot be inspected") from exc
        return {"load_state": "unavailable", "active_state": "unavailable"}
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return {
        "load_state": values.get("LoadState", "unknown"),
        "active_state": values.get("ActiveState", "unknown"),
        "sub_state": values.get("SubState", "unknown"),
        "main_pid": values.get("MainPID", "0"),
        "control_pid": values.get("ControlPID", "0"),
    }


def _blackwell_compute_processes() -> list[dict[str, str]]:
    """Return only active compute processes on the pinned GPU UUID."""

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("r212 cannot inspect existing GPU compute processes") from exc
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4 or not fields[0].isdigit() or fields[1] != BLACKWELL_UUID:
            continue
        rows.append(
            {
                "pid": fields[0],
                "gpu_uuid": fields[1],
                "process_name": fields[2],
                "used_memory_mib": fields[3],
            }
        )
    return rows


def _host_preflight(
    *,
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Produce a receipt-ready, non-interfering Blackwell capacity snapshot.

    Only GPU compute users are exclusive.  CPU-only workloads such as the
    unrelated rare-route importer remain permitted when the documented RAM and
    disk floors hold.
    """

    _assert_blackwell_cuda(device)
    run_dir = _resolve_run_dir(args, contract)
    if run_dir.exists():
        contract_path = run_dir / "RUN_CONTRACT.json"
        if contract_path.is_file():
            if _read_json_object(contract_path) != dict(contract):
                raise RuntimeError("r212 output run directory belongs to another contract")
        elif any(run_dir.iterdir()):
            raise RuntimeError("r212 output run directory is nonempty without its contract")
    if os.environ.get("POKEBOT_USE_RECURSIVE_TURN_PLANNER", "").strip().casefold() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    ):
        raise RuntimeError("r212 environment illegally enables recursive turn planning")
    for key in (
        "POKEBOT_GUIDE2VEC_R212_NO_SELECTOR_MUTATION",
        "POKEBOT_GUIDE2VEC_R212_NO_SERVING_ACTIVATION",
    ):
        if key in os.environ and os.environ[key].strip() != "1":
            raise RuntimeError(f"r212 isolation environment is invalid: {key}")
    compute_processes = _blackwell_compute_processes()
    other_gpu_processes = [
        row for row in compute_processes if int(row["pid"]) != os.getpid()
    ]
    if other_gpu_processes:
        raise RuntimeError(
            "r212 Blackwell is already in use by another compute process: "
            f"{other_gpu_processes}"
        )
    free_vram, total_vram = torch.cuda.mem_get_info(device)
    ram_available = _linux_mem_available_bytes()
    disk_path = Path(args.output_root)
    while not disk_path.exists() and disk_path != disk_path.parent:
        disk_path = disk_path.parent
    disk = os.statvfs(disk_path)
    disk_available = int(disk.f_bavail) * int(disk.f_frsize)
    if free_vram < 32 * 2**30:
        raise RuntimeError("r212 Blackwell free VRAM is below the 32 GiB safety floor")
    if ram_available and ram_available < 32 * 2**30:
        raise RuntimeError("r212 host available RAM is below the 32 GiB safety floor")
    if disk_available < 32 * 2**30:
        raise RuntimeError("r212 output filesystem free space is below the 32 GiB safety floor")
    unit = _systemd_user_state()
    if unit.get("active_state") not in ("inactive", "failed", "activating", "unavailable"):
        raise RuntimeError("r212 dedicated systemd unit is already active")
    return {
        "schema": "poke_bot.alakazam_guide2vec_r212_host_preflight/v1",
        "checked_at_utc": _utc_now(),
        "dedicated_unit": {
            "name": "pokebot-alakazam-guide2vec-r212.service",
            **unit,
        },
        "gpu": {
            "uuid": BLACKWELL_UUID,
            "requested_device": str(device),
            "name": torch.cuda.get_device_name(device),
            "free_bytes": int(free_vram),
            "total_bytes": int(total_vram),
            "other_compute_processes": other_gpu_processes,
        },
        "resources": {
            "ram_available_bytes": int(ram_available),
            "output_filesystem_free_bytes": int(disk_available),
        },
        "isolation": {
            "cpu_only_unrelated_workloads_allowed": True,
            "mcts": False,
            "rtp": False,
            "selector": False,
            "serving": False,
            "output_run_dir": str(run_dir),
            "self_cgroup": Path("/proc/self/cgroup").read_text(encoding="utf-8")
            if Path("/proc/self/cgroup").is_file()
            else "unavailable",
        },
    }


def _set_determinism(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    # The base is eval/no-grad and the tiny head has no dropout.  These flags
    # keep the remaining CUDA path reproducible without silently changing its
    # precision or falling back to CPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        pass


def _nested_config_value(config: Mapping[str, Any], *paths: Sequence[str]) -> Any:
    for path in paths:
        cursor: Any = config
        found = True
        for key in path:
            if not isinstance(cursor, Mapping) or key not in cursor:
                found = False
                break
            cursor = cursor[key]
        if found and cursor is not None:
            return cursor
    return None


def _configured_path(config: Mapping[str, Any], *paths: Sequence[str]) -> Path | None:
    value = _nested_config_value(config, *paths)
    if isinstance(value, Mapping):
        value = value.get("path")
    if value is None:
        return None
    text = str(value).strip()
    return Path(text) if text else None


def _expected_route_sidecar_producer_code() -> dict[str, str]:
    """Bind heldout sidecars to the exact route reconstruction implementation."""

    route_module = ROOT / "poke_bot/guide2vec_public_routes.py"
    if not route_module.is_file() or not ROUTE_MATERIALIZER_PATH.is_file():
        raise RuntimeError("r212 public-route sidecar producer sources are unavailable")
    return {
        "schema": ROUTE_SIDECAR_PRODUCER_CODE_SCHEMA,
        "guide2vec_public_routes_sha256": _sha256(route_module),
        "materializer_cli_sha256": _sha256(ROUTE_MATERIALIZER_PATH),
    }


def _validate_route_sidecar_manifest(
    path: Path,
    expected_sha256: str,
) -> tuple[Path, str, dict[str, RouteSidecar]]:
    """Verify the immutable heldout-route manifest before any GPU work.

    The train and validation days are reconstructed directly from their
    header-bound raw ZIPs.  The heldout days are allowed to travel from their
    raw-owning host only through this immutable sidecar manifest, whose member
    paths and bytes are then rechecked by ``SidecarPublicRouteResolver`` while
    every compact record is consumed.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = _sha256(resolved)
    if digest != expected_sha256:
        raise RuntimeError("r212 public-route sidecar manifest SHA-256 mismatch")
    manifest = _read_json_object(resolved)
    if manifest.get("schema") != ROUTE_SIDECAR_MANIFEST_SCHEMA:
        raise RuntimeError("r212 public-route sidecar manifest schema mismatch")
    producer_code = manifest.get("producer_code")
    expected_producer_code = _expected_route_sidecar_producer_code()
    if (
        not isinstance(producer_code, Mapping)
        or dict(producer_code) != expected_producer_code
    ):
        raise RuntimeError(
            "r212 public-route sidecar producer code differs from the sealed "
            "route resolver/materializer implementation"
        )
    days = manifest.get("days")
    if not isinstance(days, Mapping) or set(days) != set(HELDOUT_DATES):
        raise RuntimeError(
            "r212 public-route sidecar manifest must declare exactly the heldout days"
        )
    parsed: dict[str, RouteSidecar] = {}
    for date in HELDOUT_DATES:
        row = days.get(date)
        if not isinstance(row, Mapping):
            raise RuntimeError("r212 public-route sidecar manifest day entry is malformed")
        raw_path = str(row.get("path") or "").strip()
        sidecar_sha256 = str(row.get("sha256") or "")
        if not raw_path or not _SHA256_RE.fullmatch(sidecar_sha256):
            raise RuntimeError("r212 public-route sidecar identity is malformed")
        candidate = Path(raw_path).expanduser()
        sidecar_path = (
            candidate.resolve()
            if candidate.is_absolute()
            else (resolved.parent / candidate).resolve()
        )
        if not sidecar_path.is_file() or _sha256(sidecar_path) != sidecar_sha256:
            raise RuntimeError(
                f"r212 public-route sidecar byte identity mismatch for {date}"
            )
        declared_date = row.get("source_date")
        if declared_date is not None and str(declared_date) != date:
            raise RuntimeError("r212 public-route sidecar source date drifted")
        raw_archive = row.get("raw_archive")
        if (
            not isinstance(raw_archive, Mapping)
            or set(raw_archive)
            != {"source_date", "archive_name", "sha256", "bytes"}
            or raw_archive.get("source_date") != date
            or not isinstance(raw_archive.get("archive_name"), str)
            or str(raw_archive.get("archive_name") or "")
            != f"pokemon-tcg-ai-battle-episodes-{date}.zip"
            or not _SHA256_RE.fullmatch(str(raw_archive.get("sha256") or ""))
            or type(raw_archive.get("bytes")) is not int
            or int(raw_archive.get("bytes") or 0) <= 0
        ):
            raise RuntimeError(
                "r212 public-route sidecar manifest raw archive binding is malformed"
            )
        parsed[date] = RouteSidecar(
            date=date,
            path=sidecar_path,
            sha256=sidecar_sha256,
            raw_archive=dict(raw_archive),
            producer_code=dict(expected_producer_code),
        )
    return resolved, digest, parsed


def _validate_job_spec(job: Mapping[str, Any]) -> None:
    """Accept the typed r212 job document while rejecting authority expansion."""

    if job and job.get("schema") != "poke_bot.alakazam_guide2vec_r212_job/v1":
        # The owner contract itself is allowed as --config for a local audit;
        # a deployment config must be the typed job document below.
        if job.get("schema") != "poke_bot.alakazam_guide2vec_no_mcts_bo1000_r212/v1":
            raise RuntimeError("r212 job spec schema mismatch")
    revision = _nested_config_value(job, ("owner_decision_revision",), ("revision",))
    if revision is not None and int(revision) != 212:
        raise RuntimeError("r212 job spec revision mismatch")
    for path in (("mcts",), ("rtp",), ("selector",), ("serving",), ("kaggle",)):
        value = _nested_config_value(job, path)
        if value is True:
            raise RuntimeError(f"r212 job spec illegally enables {'.'.join(path)}")
    expected = _nested_config_value(
        job,
        ("base_checkpoint_sha256",),
        ("base", "checkpoint_sha256"),
        ("identity", "checkpoint_sha256"),
    )
    if expected is not None and str(expected) != R195_CHECKPOINT_SHA256:
        raise RuntimeError("r212 job spec checkpoint identity mismatch")
    expected = _nested_config_value(
        job,
        ("protected_pointer_sha256",),
        ("data", "protected_pointer_sha256"),
    )
    if expected is not None and str(expected) != PROTECTED_POINTER_SHA256:
        raise RuntimeError("r212 job spec pointer identity mismatch")
    checks = (
        (
            _nested_config_value(
                job,
                ("frozen_base", "checkpoint_sha256"),
                ("frozen_base", "checkpoint", "sha256"),
            ),
            R195_CHECKPOINT_SHA256,
            "frozen base checkpoint",
        ),
        (
            _nested_config_value(
                job,
                ("frozen_base", "checkpoint_bytes"),
                ("frozen_base", "checkpoint", "bytes"),
            ),
            R195_CHECKPOINT_BYTES,
            "frozen base checkpoint bytes",
        ),
        (
            _nested_config_value(job, ("frozen_base", "bundle_sha256")),
            "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145",
            "frozen base bundle",
        ),
        (
            _nested_config_value(
                job,
                ("source_corpus", "protected_pointer", "sha256"),
                ("source_corpus", "protected_pointer_sha256"),
                ("data", "protected_pointer_sha256"),
            ),
            PROTECTED_POINTER_SHA256,
            "source protected pointer",
        ),
        (
            _nested_config_value(
                job,
                ("owner_contract_reference", "sha256"),
                ("owner_contract", "sha256"),
                ("r212_contract", "sha256"),
                ("contract", "sha256"),
            ),
            OWNER_CONTRACT_SHA256,
            "owner contract",
        ),
        (
            _nested_config_value(
                job,
                ("frozen_base", "r195_contract", "sha256"),
                ("frozen_base", "r195_contract_sha256"),
                ("r195_contract", "sha256"),
            ),
            R195_CONTRACT_SHA256,
            "r195 contract",
        ),
        (
            _nested_config_value(
                job,
                ("guide_teacher", "source_ready_receipt", "sha256"),
                ("guide_teacher", "source_ready_receipt", "sha256"),
                ("guide_teacher", "source_ready_sha256"),
            ),
            GUIDE_READY_RECEIPT_SHA256,
            "guide source-ready receipt",
        ),
    )
    for actual, expected_value, label in checks:
        if actual is not None and str(actual) != str(expected_value):
            raise RuntimeError(f"r212 job spec {label} identity mismatch")
    if job.get("schema") == "poke_bot.alakazam_guide2vec_r212_job/v1":
        required = (
            (("frozen_base", "checkpoint", "path"), "frozen base checkpoint path"),
            (("frozen_base", "matchup_tree", "path"), "frozen base matchup tree path"),
            (("frozen_base", "runtime_code", "submission_bundle", "path"), "r195 submission bundle path"),
            (("source_corpus", "protected_pointer", "path"), "source protected pointer path"),
            (("source_corpus", "manifest", "path"), "source manifest path"),
            (
                ("source_corpus", "public_route_reconstruction", "raw_archive_root"),
                "raw public-route archive root",
            ),
            (
                ("source_corpus", "public_route_reconstruction", "sidecar_manifest", "path"),
                "heldout public-route sidecar manifest path",
            ),
            (("owner_contract_reference", "path"), "owner contract path"),
            (("frozen_base", "r195_contract", "path"), "r195 contract path"),
            (("guide_teacher", "guide_contract", "path"), "guide contract path"),
            (("guide_teacher", "teacher_module", "path"), "guide teacher path"),
            (("guide_teacher", "source_ready_receipt", "path"), "guide ready path"),
        )
        for path, label in required:
            if _nested_config_value(job, path) in (None, ""):
                raise RuntimeError(f"r212 job spec lacks explicit {label}")
        required_exact = (
            (_nested_config_value(job, ("source_corpus", "manifest", "sha256")), MANIFEST_SHA256, "source manifest"),
            (_nested_config_value(job, ("guide_teacher", "guide_contract", "sha256")), GUIDE_CONTRACT_SHA256, "guide contract"),
            (_nested_config_value(job, ("guide_teacher", "teacher_module", "sha256")), GUIDE_TEACHER_SHA256, "guide teacher"),
            (_nested_config_value(job, ("frozen_base", "submission_id")), int(R195_SUBMISSION_ID), "submission id"),
            (_nested_config_value(job, ("frozen_base", "bundle_sha256")), R195_BUNDLE_SHA256, "bundle"),
            (_nested_config_value(job, ("frozen_base", "matchup_tree", "sha256")), R195_MATCHUP_TREE_SHA256, "matchup tree"),
            (_nested_config_value(job, ("frozen_base", "runtime_code", "submission_bundle", "sha256")), R195_BUNDLE_SHA256, "r195 submission bundle"),
            (_nested_config_value(job, ("frozen_base", "runtime_code", "submission_entrypoint", "member")), R195_RUNTIME_ENTRYPOINT_MEMBER, "r195 runtime entrypoint member"),
            (_nested_config_value(job, ("frozen_base", "runtime_code", "submission_entrypoint", "sha256")), R195_RUNTIME_ENTRYPOINT_SHA256, "r195 runtime entrypoint"),
            (_nested_config_value(job, ("frozen_base", "runtime_code", "public_matchup_router", "member")), R195_RUNTIME_ROUTER_MEMBER, "r195 runtime router member"),
            (_nested_config_value(job, ("frozen_base", "runtime_code", "public_matchup_router", "sha256")), R195_RUNTIME_ROUTER_SHA256, "r195 runtime router"),
            (_nested_config_value(job, ("frozen_base", "deck", "id")), R195_DECK_ID, "deck id"),
            (_nested_config_value(job, ("frozen_base", "deck", "cards_sha256")), R195_DECK_CARDS_SHA256, "deck cards"),
            (_nested_config_value(job, ("source_corpus", "teacher_compatible_exact_alakazam_deck_required")), True, "exact teacher-compatible deck scope"),
            (_nested_config_value(job, ("source_corpus", "guide_rows_expected")), EXPECTED_RETAINED_GUIDE_ROWS, "protected guide rows"),
            (_nested_config_value(job, ("training", "selection_metric")), SELECTION_METRIC, "selection metric"),
        )
        for actual, expected_value, label in required_exact:
            if actual is None or str(actual) != str(expected_value):
                raise RuntimeError(f"r212 job spec {label} identity missing or mismatched")
        sidecar_manifest_sha = _nested_config_value(
            job,
            ("source_corpus", "public_route_reconstruction", "sidecar_manifest", "sha256"),
        )
        if not _SHA256_RE.fullmatch(str(sidecar_manifest_sha or "")):
            raise RuntimeError(
                "r212 job spec public-route sidecar manifest SHA-256 is missing or malformed"
            )
        if (
            tuple(_nested_config_value(job, ("split", "train_dates")) or ()) != TRAIN_DATES
            or tuple(_nested_config_value(job, ("split", "validation_dates")) or ()) != VALIDATION_DATES
            or tuple(_nested_config_value(job, ("split", "test_dates")) or ()) != HELDOUT_DATES
            or _nested_config_value(job, ("split", "heldout_partition")) != "test"
        ):
            raise RuntimeError("r212 job spec fixed day split mismatch")


def _apply_job_spec(args: argparse.Namespace) -> None:
    job: dict[str, Any] = {}
    if args.config is not None:
        path = args.config.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        job = _read_json_object(path)
        _validate_job_spec(job)
        args.config = path
    args.job_spec = job
    if args.checkpoint is None:
        raw_path = _configured_path(
            job,
            ("frozen_base", "checkpoint", "path"),
            ("frozen_base", "checkpoint_path"),
            ("base", "checkpoint"),
            ("checkpoint",),
            ("base_checkpoint",),
        )
        args.checkpoint = raw_path or DEFAULT_CHECKPOINT
    if args.protected_corpus is None:
        raw_path = _configured_path(
            job,
            ("source_corpus", "protected_pointer", "path"),
            ("source_corpus", "protected_pointer_path"),
            ("protected_corpus",),
            ("protected_pointer",),
            ("data", "protected_corpus"),
            ("data", "protected_pointer"),
        )
        args.protected_corpus = raw_path or DEFAULT_PROTECTED_CORPUS
    configured_matchup_tree = _configured_path(
        job,
        ("frozen_base", "matchup_tree", "path"),
        ("frozen_base", "matchup_tree_path"),
    )
    if args.matchup_tree is None:
        args.matchup_tree = configured_matchup_tree or DEFAULT_MATCHUP_TREE
    configured_submission_bundle = _configured_path(
        job,
        ("frozen_base", "runtime_code", "submission_bundle", "path"),
        ("frozen_base", "submission_bundle", "path"),
    )
    if args.submission_bundle is None:
        args.submission_bundle = configured_submission_bundle or DEFAULT_R195_SUBMISSION_BUNDLE
    configured_submission_bundle_sha256 = _nested_config_value(
        job,
        ("frozen_base", "runtime_code", "submission_bundle", "sha256"),
        ("frozen_base", "submission_bundle", "sha256"),
    )
    if args.submission_bundle_sha256 is None:
        args.submission_bundle_sha256 = configured_submission_bundle_sha256
    configured_raw_archive_root = _configured_path(
        job,
        ("source_corpus", "public_route_reconstruction", "raw_archive_root"),
        ("source_corpus", "raw_archive_root"),
    )
    if args.raw_archive_root is None:
        args.raw_archive_root = (
            configured_raw_archive_root
            or Path(
                os.environ.get(
                    "POKEBOT_GUIDE2VEC_R212_RAW_ARCHIVE_ROOT",
                    DEFAULT_RAW_ARCHIVE_ROOT,
                )
            )
        )
    configured_route_sidecar_manifest = _configured_path(
        job,
        ("source_corpus", "public_route_reconstruction", "sidecar_manifest", "path"),
        ("source_corpus", "route_sidecar_manifest", "path"),
    )
    if args.route_sidecar_manifest is None:
        args.route_sidecar_manifest = configured_route_sidecar_manifest
    configured_route_sidecar_manifest_sha256 = _nested_config_value(
        job,
        ("source_corpus", "public_route_reconstruction", "sidecar_manifest", "sha256"),
        ("source_corpus", "route_sidecar_manifest", "sha256"),
    )
    if args.route_sidecar_manifest_sha256 is None:
        args.route_sidecar_manifest_sha256 = configured_route_sidecar_manifest_sha256
    if args.owner_contract is None:
        raw_path = _configured_path(
            job,
            ("owner_contract_reference", "path"),
            ("owner_contract", "path"),
            ("r212_contract", "path"),
            ("contract", "path"),
        )
        # Passing the owner contract directly as --config is also supported;
        # staged job specs otherwise need an explicit owner-contract binding.
        if raw_path is None and job.get("schema") == "poke_bot.alakazam_guide2vec_no_mcts_bo1000_r212/v1":
            raw_path = args.config
        args.owner_contract = raw_path or OWNER_CONTRACT_PATH
    if args.epochs is None:
        raw = _nested_config_value(job, ("epochs",), ("training", "epochs"))
        args.epochs = int(raw) if raw is not None else MAX_EPOCHS
    if args.learning_rate is None:
        raw = _nested_config_value(job, ("learning_rate",), ("training", "learning_rate"))
        args.learning_rate = float(raw) if raw is not None else 3e-4
    if args.seed is None:
        raw = _nested_config_value(job, ("seed",), ("training", "seed"))
        args.seed = int(raw) if raw is not None else 212_096
    if args.batch_decisions is None:
        raw = _nested_config_value(job, ("batch_decisions",), ("training", "batch_decisions"))
        args.batch_decisions = int(raw) if raw is not None else 2048
    if args.coverage_weight is None:
        raw = _nested_config_value(job, ("coverage_weight",), ("training", "coverage_weight"))
        args.coverage_weight = float(raw) if raw is not None else 0.25
    if args.smoke_max_games is None:
        args.smoke_max_games = 0
    if args.output_root is None:
        args.output_root = Path(os.environ.get("POKEBOT_GUIDE2VEC_R212_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))
    args.checkpoint = Path(args.checkpoint).expanduser().resolve()
    args.protected_corpus = Path(args.protected_corpus).expanduser().resolve()
    args.matchup_tree = Path(args.matchup_tree).expanduser().resolve()
    args.submission_bundle = Path(args.submission_bundle).expanduser().resolve()
    args.raw_archive_root = Path(args.raw_archive_root).expanduser().resolve()
    args.owner_contract = Path(args.owner_contract).expanduser().resolve()
    args.output_root = Path(args.output_root).expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir).expanduser().resolve()
    if args.route_sidecar_manifest is not None:
        args.route_sidecar_manifest = Path(args.route_sidecar_manifest).expanduser().resolve()
    if configured_matchup_tree is not None and args.matchup_tree != configured_matchup_tree.expanduser().resolve():
        raise RuntimeError("r212 --matchup-tree differs from the typed job's exact r195 tree")
    if (
        configured_submission_bundle is not None
        and args.submission_bundle != configured_submission_bundle.expanduser().resolve()
    ):
        raise RuntimeError(
            "r212 --submission-bundle differs from the typed r195 submitted package"
        )
    if str(args.submission_bundle_sha256 or "") != R195_BUNDLE_SHA256:
        raise RuntimeError("r212 submission bundle is not the exact r195 NO-RTP archive")
    if (
        configured_raw_archive_root is not None
        and args.raw_archive_root != configured_raw_archive_root.expanduser().resolve()
    ):
        raise RuntimeError(
            "r212 --raw-archive-root differs from the typed job's exact route source"
        )
    if (
        configured_route_sidecar_manifest is not None
        and args.route_sidecar_manifest
        != configured_route_sidecar_manifest.expanduser().resolve()
    ):
        raise RuntimeError(
            "r212 --route-sidecar-manifest differs from the typed job's immutable manifest"
        )
    if args.route_sidecar_manifest is None:
        raise RuntimeError("r212 requires an immutable heldout public-route sidecar manifest")
    if not _SHA256_RE.fullmatch(str(args.route_sidecar_manifest_sha256 or "")):
        raise RuntimeError(
            "r212 requires an exact SHA-256 for the heldout public-route sidecar manifest"
        )
    if (
        configured_route_sidecar_manifest_sha256 is not None
        and str(args.route_sidecar_manifest_sha256)
        != str(configured_route_sidecar_manifest_sha256)
    ):
        raise RuntimeError(
            "r212 sidecar manifest SHA-256 differs from the typed job's immutable binding"
        )
    args.content_addressed_output_required = _require_content_addressed_output()
    if args.content_addressed_output_required:
        if args.output_dir is not None:
            raise RuntimeError("r212 managed content-addressed output rejects --output-dir")
        if int(args.smoke_max_games):
            raise RuntimeError("r212 managed content-addressed run rejects --smoke-max-games")
    if not 1 <= int(args.epochs) <= MAX_EPOCHS:
        raise RuntimeError(f"r212 epochs must be within [1, {MAX_EPOCHS}]")
    if int(args.batch_decisions) <= 0:
        raise RuntimeError("r212 batch decisions must be positive")
    if int(args.smoke_max_games) < 0:
        raise RuntimeError("r212 smoke max games must be nonnegative")
    if not math.isfinite(float(args.learning_rate)) or float(args.learning_rate) <= 0.0:
        raise RuntimeError("r212 learning rate must be finite and positive")
    if not math.isfinite(float(args.coverage_weight)) or float(args.coverage_weight) < 0.0:
        raise RuntimeError("r212 coverage weight must be finite and nonnegative")


def validate_inputs(args: argparse.Namespace) -> ValidatedInputs:
    """Hash-verify every immutable input before creating any output artifact."""

    _assert_combo_state_route_disabled_environment()
    source_snapshot = _validated_source_snapshot()
    typed_dependencies = _verify_r212_typed_dependencies(args.owner_contract)
    runtime_route_code = _verify_r195_route_runtime_bundle(args.submission_bundle)
    if not args.raw_archive_root.is_dir():
        raise FileNotFoundError(
            f"r212 raw public-route archive root is unavailable: {args.raw_archive_root}"
        )
    (
        route_sidecar_manifest_path,
        route_sidecar_manifest_sha256,
        route_sidecars_by_date,
    ) = _validate_route_sidecar_manifest(
        args.route_sidecar_manifest,
        str(args.route_sidecar_manifest_sha256),
    )
    payload, checkpoint_bytes, checkpoint_sha, checkpoint_context = _validate_r195_checkpoint(
        args.checkpoint
    )
    manifest_path, manifest, shards = _verify_protected_corpus(args.protected_corpus)
    if int(manifest.get("max_context") or 0) != checkpoint_context:
        raise RuntimeError("r212 corpus context does not match frozen r195 checkpoint")
    d_model = int((payload.get("model_config") or {}).get("d_model") or 0)
    if d_model != EXPECTED_D_MODEL:
        raise RuntimeError("r212 checkpoint d_model is not the frozen 96-wide contract")
    # This helper is the canonical whole-episode/day/deck compatibility audit.
    # It deliberately streams the protected source before any device pack is
    # constructed, so an archetype-only manifest cannot admit a non-teacher
    # compatible deck into this same-deck distillation.
    try:
        from poke_bot.dormant_adapter_compat import validate_zero_dormant_checkpoint
        from poke_bot.guide2vec_data import build_r212_guide2vec_split

        split_manifest = build_r212_guide2vec_split(args.protected_corpus).manifest()
        adapter_identity = validate_zero_dormant_checkpoint(
            args.checkpoint, allow_trained=True
        )
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError("r212 split/adapter immutable validation failed") from exc
    if (
        split_manifest.get("source", {}).get("protected_pointer_sha256")
        != PROTECTED_POINTER_SHA256
        or split_manifest.get("source", {}).get("manifest_sha256") != MANIFEST_SHA256
        or tuple(split_manifest.get("partitions", {}).get("train", {}).get("dates") or ())
        != TRAIN_DATES
        or tuple(split_manifest.get("partitions", {}).get("validation", {}).get("dates") or ())
        != VALIDATION_DATES
        or tuple(split_manifest.get("partitions", {}).get("test", {}).get("dates") or ())
        != HELDOUT_DATES
    ):
        raise RuntimeError("r212 fixed data split manifest drifted")
    split_policy = dict(split_manifest.get("partition_policy") or {})
    if split_policy.get("authorized_deck_fingerprints") is not None:
        raise RuntimeError(
            "r212 exact teacher-compatible deck scope must not be silently narrowed "
            "to a single list"
        )
    split_fingerprints: set[str] = set()
    for partition in ("train", "validation", "test"):
        rows = dict(split_manifest.get("partitions", {}).get(partition) or {})
        fingerprints = rows.get("deck_fingerprints")
        if (
            not isinstance(fingerprints, list)
            or any(not _SHA256_RE.fullmatch(str(value)) for value in fingerprints)
        ):
            raise RuntimeError("r212 split exact-deck fingerprint receipt is malformed")
        split_fingerprints.update(str(value) for value in fingerprints)
    retained_accounting = dict(split_manifest.get("accounting", {}).get("retained") or {})
    if (
        not split_fingerprints
        or _manifest_counter(retained_accounting, "guide_rows")
        != EXPECTED_RETAINED_GUIDE_ROWS
    ):
        raise RuntimeError(
            "r212 protected teacher-compatible deck/guide-row distribution drifted"
        )
    quarantine_by_date = _quarantine_by_date(split_manifest)
    for partition, dates in (
        ("train", TRAIN_DATES),
        ("validation", VALIDATION_DATES),
        ("test", HELDOUT_DATES),
    ):
        statistics = dict(split_manifest.get("partitions", {}).get(partition) or {})
        source_records = sum(shards[date].records for date in dates)
        quarantined_records = sum(len(quarantine_by_date.get(date, {})) for date in dates)
        source_shards = list(statistics.get("source_shards") or ())
        if len(source_shards) != len(dates):
            raise RuntimeError("r212 split source-shard accounting is incomplete")
        source_by_date: dict[str, Mapping[str, Any]] = {}
        for row in source_shards:
            if not isinstance(row, Mapping):
                raise RuntimeError("r212 split source-shard accounting is malformed")
            date = str(row.get("source_date") or "")
            if date not in dates or date in source_by_date:
                raise RuntimeError("r212 split source-shard day assignment drifted")
            shard = shards[date]
            if (
                row.get("sha256") != shard.sha256
                or _manifest_counter(row, "source_records") != shard.records
                or _manifest_counter(row, "source_decisions") != shard.decisions
                or _manifest_counter(row, "quarantined_records")
                != len(quarantine_by_date.get(date, {}))
            ):
                raise RuntimeError("r212 split source-shard identity/accounting drifted")
            source_by_date[date] = row
        if set(source_by_date) != set(dates):
            raise RuntimeError("r212 split source-day accounting is incomplete")
        source_decisions = sum(_manifest_counter(source_by_date[date], "source_decisions") for date in dates)
        source_guide_rows = sum(
            _manifest_counter(source_by_date[date], "source_guide_rows")
            for date in dates
        )
        quarantined_decisions = sum(
            _manifest_counter(source_by_date[date], "quarantined_decisions")
            for date in dates
        )
        quarantined_guide_rows = sum(
            _manifest_counter(source_by_date[date], "quarantined_guide_rows")
            for date in dates
        )
        if (
            _manifest_counter(statistics, "source_records") != source_records
            or _manifest_counter(statistics, "source_decisions") != source_decisions
            or _manifest_counter(statistics, "source_guide_rows") != source_guide_rows
            or _manifest_counter(statistics, "quarantined_records") != quarantined_records
            or _manifest_counter(statistics, "quarantined_decisions")
            != quarantined_decisions
            or _manifest_counter(statistics, "quarantined_guide_rows")
            != quarantined_guide_rows
            or _manifest_counter(statistics, "records")
            != source_records - quarantined_records
            or _manifest_counter(statistics, "decisions")
            != source_decisions - quarantined_decisions
            or _manifest_counter(statistics, "guide_rows")
            != source_guide_rows - quarantined_guide_rows
        ):
            raise RuntimeError("r212 split retained/quarantined record accounting drifted")
    if (
        adapter_identity.get("digest") != R195_CHECKPOINT_SHA256
        or adapter_identity.get("trained") is not True
        or adapter_identity.get("runtime_enabled") is not False
        or int(adapter_identity.get("parameter_count") or 0) <= 0
    ):
        raise RuntimeError("r212 r195 trained dormant adapter contract drifted")
    adapter_route_binding = _resolve_r195_v6_adapter_route_binding(adapter_identity)
    (
        matchup_tree_path,
        matchup_tree_sha256,
        adapter_route_binding,
    ) = _validate_r195_runtime_matchup_tree(
        args.matchup_tree,
        adapter_identity=adapter_identity,
        checkpoint_payload=payload,
        route_binding=adapter_route_binding,
    )
    return ValidatedInputs(
        checkpoint_path=args.checkpoint,
        checkpoint_payload=payload,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_bytes=checkpoint_bytes,
        model_config_sha256=_model_config_sha256(payload),
        pointer_path=args.protected_corpus,
        pointer_sha256=PROTECTED_POINTER_SHA256,
        manifest_path=manifest_path,
        manifest_sha256=MANIFEST_SHA256,
        max_context=checkpoint_context,
        d_model=d_model,
        shards_by_date=shards,
        typed_dependencies=typed_dependencies,
        split_manifest=split_manifest,
        adapter_identity=adapter_identity,
        adapter_route_binding=adapter_route_binding,
        matchup_tree_path=matchup_tree_path,
        matchup_tree_sha256=matchup_tree_sha256,
        runtime_route_code=runtime_route_code,
        raw_archive_root=args.raw_archive_root,
        route_sidecar_manifest_path=route_sidecar_manifest_path,
        route_sidecar_manifest_sha256=route_sidecar_manifest_sha256,
        route_sidecars_by_date=route_sidecars_by_date,
        source_snapshot=source_snapshot,
        quarantine_by_date=quarantine_by_date,
    )


def _run_contract(args: argparse.Namespace, inputs: ValidatedInputs) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "owner_decision_revision": 212,
        "mode": "smoke" if int(args.smoke_max_games) else "full",
        "base": inputs.as_dict()["checkpoint"],
        "data": {
            "pointer_sha256": inputs.pointer_sha256,
            "manifest_sha256": inputs.manifest_sha256,
            "train_dates": list(TRAIN_DATES),
            "validation_dates": list(VALIDATION_DATES),
            "heldout_dates": list(HELDOUT_DATES),
            "smoke_max_games": int(args.smoke_max_games),
            "latent_extractor": LATENT_EXTRACTOR,
            "split_manifest_sha256": "sha256:"
            + hashlib.sha256(_canonical_json(inputs.split_manifest)).hexdigest(),
            "public_route_reconstruction": inputs.as_dict()[
                "raw_public_route_reconstruction"
            ],
        },
        "source_snapshot": dict(inputs.source_snapshot),
        "publication": {
            "content_addressed_contract_child": True,
            "managed_content_addressed_output_required": bool(
                getattr(args, "content_addressed_output_required", False)
            ),
            "explicit_output_dir_allowed": not bool(
                getattr(args, "content_addressed_output_required", False)
            ),
        },
        "guide2vec": {
            "d_model": EXPECTED_D_MODEL,
            "score_hidden": 256,
            "bottleneck": 64,
            "eligibility_hidden": 128,
            "expected_parameters": EXPECTED_GUIDE2VEC_PARAMETERS,
            "maximum_parameters": MAX_GUIDE2VEC_PARAMETERS,
            "max_normalized_logit_bonus": MAX_GUIDE_LOGIT_BONUS,
        },
        "submitted_adapter_runtime": {
            "enabled": True,
            "public_routes_only": True,
            "oracle_routes_used": False,
            "matchup_tree": {
                "path": str(inputs.matchup_tree_path),
                "sha256": inputs.matchup_tree_sha256,
                "runtime_enabled": True,
            },
            "adapter_route_binding": inputs.adapter_route_binding.as_dict(),
            "trained_bank_parameter_count": int(
                inputs.adapter_identity.get("parameter_count") or 0
            ),
            "bank_config_sha256": "sha256:"
            + hashlib.sha256(
                _canonical_json(inputs.adapter_identity.get("adapter_config"))
            ).hexdigest(),
        },
        "optimization": {
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "batch_decisions": int(args.batch_decisions),
            "coverage_weight": float(args.coverage_weight),
            "seed": int(args.seed),
            "base_frozen": True,
            "base_eval": True,
            "base_no_grad": True,
            "bf16_latent_extraction": True,
            "rank_loss": "confidence_weighted_cross_entropy_labeled_non_singleton_only",
            "coverage_loss": "binary_cross_entropy_labeled_or_masked_non_singleton",
            "selection_metric": SELECTION_METRIC,
            "selection_uses_coverage_bce": False,
            "nonpositive_confidence_or_out_of_range_target": "mask_entire_stage",
            "early_stop": "validation_only",
            "calibration": "validation_only",
            "heldout_teacher_agreement_minimum": 0.90,
            "heldout_teacher_agreement_scope": "teacher_labeled_legal_stages_only",
        },
        "authority": {
            "mcts": False,
            "rtp": False,
            "selector": False,
            "serving": False,
            "kaggle": False,
            "base_checkpoint_mutation": False,
            "training_eligibility_of_heldout_or_evaluation": False,
        },
    }


def _require_content_addressed_output() -> bool:
    """Read the managed-output policy without silently accepting a typo."""

    raw = os.environ.get("POKEBOT_GUIDE2VEC_R212_REQUIRE_CONTENT_ADDRESSED_OUTPUT", "")
    if raw not in {"", "0", "1"}:
        raise RuntimeError(
            "POKEBOT_GUIDE2VEC_R212_REQUIRE_CONTENT_ADDRESSED_OUTPUT must be 0 or 1"
        )
    return raw == "1"


def _resolve_run_dir(args: argparse.Namespace, contract: Mapping[str, Any]) -> Path:
    digest = hashlib.sha256(_canonical_json(dict(contract))).hexdigest()
    strict_content_addressed = _require_content_addressed_output()
    if strict_content_addressed:
        if args.output_dir is not None:
            raise RuntimeError(
                "r212 managed content-addressed output rejects --output-dir"
            )
        run_dir = args.output_root / f"r212-{digest[:24]}"
        if run_dir.parent != args.output_root:
            raise AssertionError("r212 content-addressed output escaped its root")
    elif args.output_dir is not None:
        run_dir = args.output_dir
    else:
        run_dir = args.output_root / f"r212-{digest[:24]}"
    forbidden = {"pure_rl", "rtp_fleet"}
    if forbidden.intersection(run_dir.parts):
        raise RuntimeError("r212 output may not write into pure_rl or rtp_fleet")
    return run_dir


def _load_frozen_base(inputs: ValidatedInputs, device: torch.device) -> torch.nn.Module:
    # The serialized r195 model keeps the trained bank dormant for training
    # safety, while the exact submitted NO-RTP runtime enabled that same bank
    # with public routes.  Reproduce the latter explicitly; never infer a
    # route or use the oracle route.
    try:
        from poke_bot.guide2vec import assert_base_frozen, freeze_base_model
    except ImportError as exc:
        raise RuntimeError("r212 Guide2Vec frozen-base helpers are unavailable") from exc
    _assert_combo_state_route_disabled_environment()
    base = load_model_from_checkpoint(inputs.checkpoint_path, device=device)
    if int(getattr(base, "d_model", 0)) != inputs.d_model:
        raise RuntimeError("loaded r195 base model d_model drifted after preflight")
    if str(getattr(base, "decision_context", "")) != "history":
        raise RuntimeError("loaded r195 base model is not temporal")
    cfg = getattr(base, "cfg", None)
    fusion = getattr(base, "decision_fusion", None)
    if (
        cfg is None
        or getattr(cfg, "combo_state_route_enabled", True) is not False
        or getattr(base, "combo_state_route_enabled", True) is not False
        or (fusion is not None and getattr(fusion, "combo_state_route_enabled", True) is not False)
    ):
        raise RuntimeError("r212 loaded base has the forbidden combo-state policy route enabled")
    bank = getattr(base, "matchup_adapter_bank", None)
    if bank is None or not hasattr(bank, "config_dict") or not hasattr(bank, "enabled"):
        raise RuntimeError("r212 r195 base lacks its submitted matchup adapter bank")
    expected_config = inputs.adapter_identity.get("adapter_config")
    if not isinstance(expected_config, Mapping) or bank.config_dict() != dict(expected_config):
        raise RuntimeError("r212 loaded matchup adapter bank differs from r195 trained contract")
    loaded_route_binding = _resolve_r195_v6_adapter_route_binding(
        {"adapter_config": bank.config_dict()}
    )
    if (
        loaded_route_binding.checkpoint_contract_dict()
        != inputs.adapter_route_binding.checkpoint_contract_dict()
        or int(getattr(bank, "slot_capacity", -1))
        != inputs.adapter_route_binding.slot_capacity
        or len(getattr(bank, "experts", ()))
        != inputs.adapter_route_binding.slot_capacity
    ):
        raise RuntimeError("r212 loaded bank differs from the exact r195 V6 slot contract")
    if (
        tuple(inputs.adapter_route_binding.runtime_accepted_nonzero_output_slots)
        != tuple(inputs.adapter_route_binding.runtime_accepted_physical_slots)
    ):
        raise RuntimeError("r212 accepted adapter slot output receipt is incomplete")
    for slot in inputs.adapter_route_binding.runtime_accepted_physical_slots:
        expert = bank.experts[slot]
        outputs = (expert.up.weight, expert.up.bias)
        if not any(int(value.detach().count_nonzero().item()) > 0 for value in outputs):
            raise RuntimeError(f"r212 loaded accepted adapter slot is zero-output: {slot}")
    bank.enabled = True
    freeze_base_model(base)
    assert_base_frozen(base)
    if not bool(bank.enabled):
        raise RuntimeError("r212 could not activate the submitted r195 adapter runtime")
    # This marker is receipt/audit-only and not consumed by model inference.
    setattr(base, "_r212_matchup_adapter_runtime_enabled", True)
    return base


def _sample_ids_for_games(
    corpus: DeviceResidentBootstrapCorpus, game_ids: torch.Tensor
) -> torch.Tensor:
    if corpus.game_sample_offset is None:
        raise RuntimeError("r212 temporal corpus has no sample offsets")
    ids = game_ids.reshape(-1).to(device=corpus.device, dtype=torch.long)
    starts = corpus.game_sample_offset.index_select(0, ids).to(dtype=torch.long)
    ends = corpus.game_sample_offset.index_select(0, ids + 1).to(dtype=torch.long)
    sample_ids, _ = corpus._expand_ranges(starts, ends)  # noqa: SLF001
    return sample_ids


@torch.inference_mode()
def extract_temporal_latents(
    base: torch.nn.Module,
    corpus: DeviceResidentBootstrapCorpus,
    game_ids: torch.Tensor,
    sample_public_routes: torch.Tensor,
    adapter_route_binding: AdapterRouteBinding,
) -> dict[str, torch.Tensor]:
    """Return causal frozen r195 state/options/logits for complete legal stages.

    This is intentionally the same temporal reconstruction used by the resident
    trainer: each game receives only its own shifted realized action history,
    then options are decoded exactly once with ``return_hidden=True``.
    """

    (
        board,
        previous_actions,
        options,
        counts,
        _targets,
        _values,
        game_lengths,
        sample_state_rows,
    ) = corpus.temporal_batch(game_ids)
    decisions = int(game_lengths.sum().item())
    samples = int(sample_state_rows.numel())
    if decisions <= 0 or samples <= 0:
        raise RuntimeError("r212 temporal batch has no trainable stages")
    if int(game_lengths.max().item()) > int(getattr(base, "max_context", 0)):
        raise RuntimeError("r212 source game exceeds frozen r195 temporal context")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        spatial = base.encode_board_packed(board, batch_size=decisions)
        action_state = base.encode_previous_actions_packed(
            previous_actions, batch_size=decisions
        )
        cls = base.pool_cls(spatial) + float(base.cfg.history_action_scale) * action_state

        lengths = [int(value) for value in game_lengths.cpu().tolist()]
        cursor = 0
        by_length: dict[int, list[int]] = {}
        for length in lengths:
            by_length.setdefault(length, []).append(cursor)
            cursor += length
        state_parts: list[torch.Tensor] = []
        row_parts: list[torch.Tensor] = []
        for length, starts in by_length.items():
            tokens = torch.stack([cls[start : start + length] for start in starts], dim=0)
            encoded, _ = base.temporal_encode(tokens, append=False, return_all=True)
            state_parts.append(encoded.reshape(-1, int(base.d_model)))
            row_parts.append(
                torch.cat(
                    [
                        torch.arange(
                            start,
                            start + length,
                            device=corpus.device,
                            dtype=torch.long,
                        )
                        for start in starts
                    ]
                )
            )
        grouped_states = torch.cat(state_parts, dim=0)
        grouped_rows = torch.cat(row_parts, dim=0)
        state_by_decision = grouped_states.index_select(0, torch.argsort(grouped_rows))
        state_vec = state_by_decision.index_select(0, sample_state_rows)
        sample_spatial = spatial.index_select(0, sample_state_rows)
        sample_ids = _sample_ids_for_games(corpus, game_ids)
        if int(sample_ids.numel()) != samples:
            raise RuntimeError("r212 sample-id mapping drifted during latent extraction")
        if sample_public_routes.ndim != 1 or int(sample_public_routes.numel()) != int(corpus.total_samples):
            raise RuntimeError("r212 public adapter route vector does not align with corpus samples")
        routes = sample_public_routes.index_select(0, sample_ids).to(dtype=torch.long)
        if bool(
            ((routes < UNKNOWN_ROUTE) | (routes >= adapter_route_binding.slot_capacity)).any()
        ):
            raise RuntimeError("r212 public adapter route escaped the r195 V6 slot capacity")
        routed = routes != UNKNOWN_ROUTE
        if bool(routed.any()):
            exact_slots = torch.tensor(
                adapter_route_binding.runtime_accepted_physical_slots,
                dtype=torch.long,
                device=routes.device,
            )
            if not bool(torch.isin(routes[routed], exact_slots).all()):
                raise RuntimeError(
                    "r212 public adapter route is not in the r195 V6 tree's "
                    "runtime-accepted slot binding"
                )
        # r195 applies the public-prefix route only to its policy/value state.
        # Guide2Vec still receives raw causal state_vec; decision fusion also
        # consumes the raw state exactly as the frozen base forward path does.
        policy_value_state = base.matchup_policy_value_state(
            state_vec,
            routes,
            enabled=True,
        )
        decoded = base.decode_options_packed(
            options,
            sample_spatial,
            policy_value_state,
            n_options=counts,
            batch_size=samples,
            return_hidden=True,
            decision_fusion_state_vec=state_vec,
        )
    if not isinstance(decoded, tuple) or len(decoded) != 2:
        raise RuntimeError("r212 base decoder did not return (logits, option_hidden)")
    base_logits, option_hidden = decoded
    if (
        state_vec.ndim != 2
        or option_hidden.ndim != 3
        or base_logits.ndim != 2
        or state_vec.shape[0] != samples
        or option_hidden.shape[:2] != base_logits.shape
    ):
        raise RuntimeError("r212 frozen temporal latent shape mismatch")
    if corpus.guide_target_index is None or corpus.guide_confidence is None:
        raise RuntimeError("r212 resident corpus unexpectedly lacks guide metadata")
    guide_target = corpus.guide_target_index.index_select(0, sample_ids).to(torch.long)
    guide_confidence = corpus.guide_confidence.index_select(0, sample_ids).to(torch.float32)
    counts = counts.to(dtype=torch.long)
    if not bool(torch.isfinite(guide_confidence).all()) or bool(
        ((guide_confidence < 0.0) | (guide_confidence > 1.0)).any()
    ):
        raise RuntimeError("r212 guide confidence leaves [0, 1]")
    # Preserve the owner-specified masking semantics even if an old compact
    # row is malformed.  The canonical split audit rejects such source rows;
    # this second guard makes the device path fail-safe rather than turning an
    # invalid label into a coverage positive.
    invalid = (guide_target >= 0) & (guide_target >= counts)
    # The owner contract makes a nonpositive confidence an entire-stage mask,
    # not a zero-weight positive eligibility row.  Likewise, a one-option
    # stage carries no ranking decision and is a coverage negative.
    valid_rank_stage = (counts > 1) & (guide_confidence > 0.0) & ~invalid
    guide_target = torch.where(
        valid_rank_stage,
        guide_target,
        torch.full_like(guide_target, -1),
    )
    return {
        "state_vec": state_vec.detach(),
        "option_hidden": option_hidden.detach(),
        "base_logits": base_logits.detach(),
        "n_options": counts.detach(),
        "guide_target_index": guide_target.detach(),
        "guide_confidence": guide_confidence.detach(),
    }


def _flatten_latent_batch(latents: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert padded option tensors into a compact, legal-option-only cache."""

    state = latents["state_vec"]
    hidden = latents["option_hidden"]
    base_logits = latents["base_logits"]
    counts = latents["n_options"].to(dtype=torch.long)
    if state.ndim != 2 or hidden.ndim != 3 or base_logits.ndim != 2:
        raise RuntimeError("r212 cannot flatten malformed latent tensors")
    if hidden.shape[0] != state.shape[0] or hidden.shape[1] != base_logits.shape[1]:
        raise RuntimeError("r212 latent option dimensions disagree")
    if int(counts.numel()) != int(state.shape[0]):
        raise RuntimeError("r212 latent count vector is misaligned")
    max_options = int(hidden.shape[1])
    valid = torch.arange(max_options, device=hidden.device).unsqueeze(0) < counts.unsqueeze(1)
    if not bool(valid.any()):
        raise RuntimeError("r212 latent batch has no legal options")
    offsets = torch.cat(
        [
            torch.zeros(1, device=hidden.device, dtype=torch.long),
            torch.cumsum(counts, dim=0),
        ]
    )
    return {
        "state_vec": state.to(dtype=torch.float16).cpu(),
        "option_hidden": hidden[valid].to(dtype=torch.float16).cpu(),
        "base_logits": base_logits[valid].to(dtype=torch.float16).cpu(),
        "option_offsets": offsets.cpu(),
        "guide_target_index": latents["guide_target_index"].to(dtype=torch.int32).cpu(),
        "guide_confidence": latents["guide_confidence"].to(dtype=torch.float32).cpu(),
    }


@dataclass(frozen=True)
class LatentCache:
    partition: str
    date: str
    root: Path
    manifest: Mapping[str, Any]

    @property
    def chunks(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.manifest["chunks"])


def _latent_identity(
    *,
    partition: str,
    shard: SourceShard,
    inputs: ValidatedInputs,
    smoke_limit: int,
) -> dict[str, Any]:
    return {
        "schema": LATENT_SCHEMA,
        "extractor": LATENT_EXTRACTOR,
        "partition": partition,
        "date": shard.date,
        "source_shard": shard.as_dict(),
        "checkpoint_sha256": inputs.checkpoint_sha256,
        "checkpoint_bytes": inputs.checkpoint_bytes,
        "model_config_sha256": inputs.model_config_sha256,
        "d_model": inputs.d_model,
        "max_context": inputs.max_context,
        "source_snapshot": dict(inputs.source_snapshot),
        "public_route_reconstruction": {
            "schema": ROUTE_RECONSTRUCTION_SCHEMA,
            "algorithm": ROUTE_ALGORITHM,
            "raw_archive_root": str(inputs.raw_archive_root),
            "mode": "sidecar" if shard.date in HELDOUT_DATES else "raw",
            "sidecar_manifest": {
                "path": str(inputs.route_sidecar_manifest_path),
                "sha256": inputs.route_sidecar_manifest_sha256,
            },
            "sidecar": (
                inputs.route_sidecars_by_date[shard.date].as_dict()
                if shard.date in HELDOUT_DATES
                else None
            ),
            "runtime_code": {
                "submission_bundle_path": str(
                    inputs.runtime_route_code.submission_bundle_path
                ),
                **inputs.runtime_route_code.as_dict(),
            },
            "compact_source_routes_ignored": True,
            "oracle_route_used": False,
        },
        "submitted_adapter_runtime": {
            "enabled": True,
            "public_routes_only": True,
            "matchup_tree": {
                "path": str(inputs.matchup_tree_path),
                "sha256": inputs.matchup_tree_sha256,
                "runtime_enabled": True,
            },
            "adapter_route_binding": inputs.adapter_route_binding.as_dict(),
            "bank_config_sha256": "sha256:"
            + hashlib.sha256(
                _canonical_json(inputs.adapter_identity.get("adapter_config"))
            ).hexdigest(),
        },
        "quarantine": {
            "records": len(inputs.quarantine_by_date.get(shard.date, {})),
            "identity_sha256": "sha256:"
            + hashlib.sha256(
                json.dumps(
                    [
                        value.as_dict()
                        for _, value in sorted(
                            inputs.quarantine_by_date.get(shard.date, {}).items()
                        )
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
        "smoke_game_limit": int(smoke_limit),
    }


def _validate_latent_chunk(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise RuntimeError(f"r212 latent chunk checksum mismatch: {path}")
    value = _torch_load_verified(path)
    required = {
        "state_vec",
        "option_hidden",
        "base_logits",
        "option_offsets",
        "guide_target_index",
        "guide_confidence",
    }
    if set(value) != required:
        raise RuntimeError(f"r212 latent chunk fields mismatch: {path}")
    state = value["state_vec"]
    hidden = value["option_hidden"]
    logits = value["base_logits"]
    offsets = value["option_offsets"]
    target = value["guide_target_index"]
    confidence = value["guide_confidence"]
    if not all(isinstance(tensor, torch.Tensor) for tensor in value.values()):
        raise RuntimeError(f"r212 latent chunk has non-tensor payload: {path}")
    rows = int(state.shape[0])
    if (
        state.ndim != 2
        or int(state.shape[1]) != EXPECTED_D_MODEL
        or hidden.ndim != 2
        or int(hidden.shape[1]) != EXPECTED_D_MODEL
        or logits.ndim != 1
        or int(logits.numel()) != int(hidden.shape[0])
        or offsets.ndim != 1
        or int(offsets.numel()) != rows + 1
        or int(offsets[0].item()) != 0
        or int(offsets[-1].item()) != int(hidden.shape[0])
        or target.shape != (rows,)
        or confidence.shape != (rows,)
    ):
        raise RuntimeError(f"r212 latent chunk shape mismatch: {path}")
    if not bool((offsets[1:] >= offsets[:-1]).all()):
        raise RuntimeError(f"r212 latent chunk offsets are not monotonic: {path}")
    counts = offsets[1:] - offsets[:-1]
    if not bool((counts > 0).all()):
        raise RuntimeError(f"r212 latent chunk contains empty legal stage: {path}")
    if not bool(torch.isfinite(state.float()).all()) or not bool(torch.isfinite(hidden.float()).all()):
        raise RuntimeError(f"r212 latent chunk contains non-finite features: {path}")
    if not bool(torch.isfinite(logits.float()).all()):
        raise RuntimeError(f"r212 latent chunk contains non-finite base logits: {path}")
    if not bool(torch.isfinite(confidence).all()) or bool(((confidence < 0) | (confidence > 1)).any()):
        raise RuntimeError(f"r212 latent chunk confidence mismatch: {path}")
    invalid = (target >= 0) & (target.to(torch.long) >= counts)
    if bool(invalid.any()):
        raise RuntimeError(f"r212 latent chunk target outside legal stage: {path}")
    if bool(((confidence <= 0.0) & (target >= 0)).any()):
        raise RuntimeError(
            f"r212 latent chunk retained a nonpositive-confidence guide target: {path}"
        )
    if bool(((counts <= 1) & (target >= 0)).any()):
        raise RuntimeError(
            f"r212 latent chunk retained a singleton guide target: {path}"
        )
    return value


def _validate_cached_public_route_projection(
    route_projection: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    """Reject a latent cache without exact raw/sidecar route provenance."""

    expected_adapter_runtime = dict(identity.get("submitted_adapter_runtime") or {})
    expected_reconstruction = dict(identity.get("public_route_reconstruction") or {})
    source_shard = dict(identity.get("source_shard") or {})
    reconstruction = dict(route_projection.get("raw_route_reconstruction") or {})
    raw_archive = reconstruction.get("raw_archive")
    turn_order_exclusions = route_projection.get(
        "compact_turn_order_short_circuit_exclusions"
    )
    deck_telemetry = route_projection.get("retained_deck_fingerprint_telemetry")
    expected_sidecar = expected_reconstruction.get("sidecar")
    expected_runtime_code = dict(expected_reconstruction.get("runtime_code") or {})
    expected_runtime_code_projection = {
        key: value
        for key, value in expected_runtime_code.items()
        if key != "submission_bundle_path"
    }
    expected_mode = str(expected_reconstruction.get("mode") or "")
    if (
        route_projection.get("schema")
        != "poke_bot.alakazam_guide2vec_r212_public_adapter_route_projection/v1"
        or not _SHA256_RE.fullmatch(str(route_projection.get("sha256") or ""))
        or int(route_projection.get("samples") or -1) <= 0
        or int(route_projection.get("routed_samples") or 0)
        + int(route_projection.get("bypassed_samples") or 0)
        != int(route_projection.get("samples") or -1)
        or route_projection.get("source")
        != "r195_raw_public_matchup_route_reconstruction"
        or route_projection.get("route_reconstruction_mode") != expected_mode
        or route_projection.get("compact_source_routes_ignored") is not True
        or type(route_projection.get("turn_order_short_circuits")) is not int
        or int(route_projection.get("turn_order_short_circuits")) < 0
        or type(route_projection.get("game_resets")) is not int
        or int(route_projection.get("game_resets")) < 0
        or route_projection.get("game_reset_history_spans_compact_temporal_sequence")
        is not False
        or route_projection.get("compact_turn_order_short_circuits_admitted")
        is not False
        or not isinstance(turn_order_exclusions, Mapping)
        or turn_order_exclusions.get("schema")
        != "poke_bot.alakazam_guide2vec_r212_turn_order_short_circuit_exclusions/v1"
        or type(turn_order_exclusions.get("packed_records")) is not int
        or int(turn_order_exclusions.get("packed_records")) <= 0
        or int(turn_order_exclusions.get("packed_records"))
        > int(source_shard.get("records") or 0)
        or any(
            type(turn_order_exclusions.get(field)) is not int
            or int(turn_order_exclusions.get(field)) < 0
            for field in (
                "excluded_decisions",
                "excluded_policy_stages",
                "excluded_samples",
                "excluded_guide_rows",
            )
        )
        or int(turn_order_exclusions.get("excluded_decisions"))
        > int(route_projection.get("turn_order_short_circuits") or 0)
        or int(turn_order_exclusions.get("excluded_policy_stages"))
        != int(turn_order_exclusions.get("excluded_samples"))
        or not _SHA256_RE.fullmatch(
            str(turn_order_exclusions.get("identity_sha256") or "")
        )
        or turn_order_exclusions.get("admitted_to_model_history") is not False
        or not isinstance(deck_telemetry, Mapping)
        or deck_telemetry.get("schema")
        != "poke_bot.alakazam_guide2vec_r212_exact_deck_telemetry/v1"
        or deck_telemetry.get("policy")
        != (
            "all_exact_teacher_compatible_alakazam_60_card_multisets_not_"
            "single_r195_evaluation_deck_allowlist"
        )
        or deck_telemetry.get("card_count") != 60
        or deck_telemetry.get("teacher_label_source")
        != "protected_r212_compact_guide_target_index_confidence"
        or any(
            type(deck_telemetry.get(field)) is not int
            or int(deck_telemetry.get(field)) < 0
            for field in (
                "retained_records",
                "retained_guide_rows",
                "packed_records",
                "packed_guide_rows",
                "context_cap_excluded_decisions",
                "context_cap_excluded_guide_rows",
                "distinct_fingerprints",
            )
        )
        or int(deck_telemetry.get("retained_records"))
        != int(source_shard.get("records") or 0)
        - int(dict(identity.get("quarantine") or {}).get("records") or 0)
        or int(deck_telemetry.get("packed_records"))
        != int(turn_order_exclusions.get("packed_records"))
        or not isinstance(deck_telemetry.get("distribution"), list)
        or int(deck_telemetry.get("distinct_fingerprints"))
        != len(deck_telemetry.get("distribution") or ())
        or not _SHA256_RE.fullmatch(
            str(deck_telemetry.get("distribution_sha256") or "")
        )
        or deck_telemetry.get("distribution_sha256")
        != "sha256:"
        + hashlib.sha256(_canonical_json(deck_telemetry.get("distribution"))).hexdigest()
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"deck_fingerprint", "records", "decisions", "guide_rows"}
            or not _SHA256_RE.fullmatch(str(row.get("deck_fingerprint") or ""))
            or type(row.get("records")) is not int
            or type(row.get("decisions")) is not int
            or type(row.get("guide_rows")) is not int
            or int(row.get("records")) <= 0
            or int(row.get("decisions")) < 0
            or int(row.get("guide_rows")) < 0
            for row in (deck_telemetry.get("distribution") or ())
        )
        or [
            str(row.get("deck_fingerprint"))
            for row in (deck_telemetry.get("distribution") or ())
        ]
        != sorted(
            str(row.get("deck_fingerprint"))
            for row in (deck_telemetry.get("distribution") or ())
        )
        or len(
            {
                str(row.get("deck_fingerprint"))
                for row in (deck_telemetry.get("distribution") or ())
            }
        )
        != int(deck_telemetry.get("distinct_fingerprints"))
        or sum(
            int(row.get("records") or 0)
            for row in (deck_telemetry.get("distribution") or ())
        )
        != int(deck_telemetry.get("retained_records"))
        or sum(
            int(row.get("guide_rows") or 0)
            for row in (deck_telemetry.get("distribution") or ())
        )
        != int(deck_telemetry.get("retained_guide_rows"))
        or int(deck_telemetry.get("packed_guide_rows"))
        + int(turn_order_exclusions.get("excluded_guide_rows"))
        + int(deck_telemetry.get("context_cap_excluded_guide_rows"))
        > int(deck_telemetry.get("retained_guide_rows"))
        or (
            int(identity.get("smoke_game_limit") or 0) == 0
            and int(deck_telemetry.get("packed_guide_rows"))
            + int(turn_order_exclusions.get("excluded_guide_rows"))
            + int(deck_telemetry.get("context_cap_excluded_guide_rows"))
            != int(deck_telemetry.get("retained_guide_rows"))
        )
        or route_projection.get("oracle_route_used") is not False
        or dict(route_projection.get("adapter_route_binding") or {})
        != dict(expected_adapter_runtime.get("adapter_route_binding") or {})
        or reconstruction.get("schema") != ROUTE_RECONSTRUCTION_SCHEMA
        or reconstruction.get("algorithm") != ROUTE_ALGORITHM
        or reconstruction.get("source_date") != identity.get("date")
        or reconstruction.get("source_feature_shard_sha256") != source_shard.get("sha256")
        or int(reconstruction.get("records") or -1) != int(source_shard.get("records") or -1)
        or int(reconstruction.get("decisions") or 0) <= 0
        or type(reconstruction.get("game_resets")) is not int
        or int(reconstruction.get("game_resets"))
        != int(route_projection.get("game_resets"))
        or int(reconstruction.get("routed_decisions") or 0)
        + int(reconstruction.get("bypassed_decisions") or 0)
        != int(reconstruction.get("decisions") or -1)
        or reconstruction.get("runtime_public_tree_sha256")
        != dict(expected_adapter_runtime.get("matchup_tree") or {}).get("sha256")
        or reconstruction.get("allowed_physical_slots")
        != sorted(
            dict(expected_adapter_runtime.get("adapter_route_binding") or {}).get(
                "runtime_accepted_physical_slots"
            )
            or ()
        )
        or reconstruction.get("compact_source_routes_ignored") is not True
        or reconstruction.get("oracle_route_used") is not False
        or dict(reconstruction.get("runtime_code") or {})
        != expected_runtime_code_projection
        or not isinstance(raw_archive, Mapping)
        or not _SHA256_RE.fullmatch(str(raw_archive.get("sha256") or ""))
        or int(raw_archive.get("bytes") or 0) <= 0
        or not _SHA256_RE.fullmatch(
            str(reconstruction.get("member_route_sha256") or "")
        )
    ):
        raise RuntimeError("r212 latent cache lacks a valid raw public-route projection")
    if expected_mode == "raw":
        if expected_sidecar is not None or "sidecar" in reconstruction:
            raise RuntimeError("r212 raw-route cache unexpectedly uses a sidecar")
    elif expected_mode == "sidecar":
        expected_sidecar_binding = (
            {
                "path": str(expected_sidecar.get("path") or ""),
                "sha256": str(expected_sidecar.get("sha256") or ""),
            }
            if isinstance(expected_sidecar, Mapping)
            else None
        )
        if expected_sidecar_binding is None or dict(
            reconstruction.get("sidecar") or {}
        ) != expected_sidecar_binding or dict(raw_archive or {}) != dict(
            expected_sidecar.get("raw_archive") or {}
        ) or dict(reconstruction.get("producer_code") or {}) != dict(
            expected_sidecar.get("producer_code") or {}
        ):
            raise RuntimeError("r212 heldout route-sidecar cache provenance mismatch")
    else:
        raise RuntimeError("r212 latent cache route reconstruction mode is invalid")
    # A full managed candidate may never call an all-bypass projection adapter
    # parity.  Smoke rows deliberately retain their flexibility for narrow
    # diagnostics, but they remain non-candidate/non-promotion artifacts.
    if int(identity.get("smoke_game_limit") or 0) == 0 and (
        int(reconstruction.get("routed_decisions") or 0) <= 0
        or int(route_projection.get("routed_samples") or 0) <= 0
    ):
        raise RuntimeError("r212 full latent cache has no reconstructed active adapter route")


def _open_existing_latent_cache(
    root: Path, identity: Mapping[str, Any]
) -> LatentCache | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema") != LATENT_SCHEMA or dict(manifest.get("identity") or {}) != dict(identity):
        raise RuntimeError(f"r212 latent cache identity mismatch: {root}")
    route_projection = dict(manifest.get("public_adapter_route_projection") or {})
    try:
        _validate_cached_public_route_projection(route_projection, identity)
    except RuntimeError as exc:
        raise RuntimeError(
            f"r212 latent cache lacks a valid public route projection: {root}"
        ) from exc
    quarantine_projection = dict(manifest.get("quarantine_projection") or {})
    expected_quarantine = dict(identity.get("quarantine") or {})
    if (
        quarantine_projection.get("schema")
        != "poke_bot.alakazam_guide2vec_r212_quarantine_projection/v1"
        or int(quarantine_projection.get("records") or -1)
        != int(expected_quarantine.get("records") or -1)
        or quarantine_projection.get("identity_sha256")
        != expected_quarantine.get("identity_sha256")
        or quarantine_projection.get("never_packed_as_coverage_negatives") is not True
    ):
        raise RuntimeError(f"r212 latent cache quarantine projection mismatch: {root}")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise RuntimeError(f"r212 latent cache has no chunks: {root}")
    total_rows = total_options = 0
    for row in chunks:
        if not isinstance(row, dict):
            raise RuntimeError(f"r212 malformed latent chunk entry: {root}")
        relative = str(row.get("path") or "")
        digest = str(row.get("sha256") or "")
        if not relative or not _SHA256_RE.fullmatch(digest):
            raise RuntimeError(f"r212 malformed latent chunk identity: {root}")
        value = _validate_latent_chunk(root / relative, digest)
        total_rows += int(value["state_vec"].shape[0])
        total_options += int(value["option_hidden"].shape[0])
    if total_rows != int(manifest.get("rows") or -1) or total_options != int(manifest.get("option_rows") or -1):
        raise RuntimeError(f"r212 latent cache totals mismatch: {root}")
    if int(route_projection["samples"]) != total_rows:
        raise RuntimeError(f"r212 route projection does not align with latent rows: {root}")
    return LatentCache(
        partition=str(identity["partition"]),
        date=str(identity["date"]),
        root=root,
        manifest=manifest,
    )


def _build_day_latent_cache(
    *,
    cache_root: Path,
    partition: str,
    shard: SourceShard,
    inputs: ValidatedInputs,
    base: torch.nn.Module,
    device: torch.device,
    batch_decisions: int,
    smoke_limit: int,
) -> LatentCache:
    identity = _latent_identity(
        partition=partition,
        shard=shard,
        inputs=inputs,
        smoke_limit=smoke_limit,
    )
    fingerprint = hashlib.sha256(_canonical_json(identity)).hexdigest()[:24]
    root = cache_root / partition / f"{shard.date}-{fingerprint}"
    existing = _open_existing_latent_cache(root, identity)
    if existing is not None:
        return existing
    # Never re-use an incomplete cache directory.  A failed writer leaves its
    # uniquely named partial root as audit evidence; a later attempt builds a
    # fresh temporary tree and publishes it atomically only after full checks.
    temporary = root.with_name(f".{root.name}.partial.{os.getpid()}.{time.time_ns()}")
    temporary.mkdir(parents=True, exist_ok=False)
    public_routes: torch.Tensor | None = None
    try:
        source = _VerifiedDayGames(
            shard,
            max_context=inputs.max_context,
            quarantine=inputs.quarantine_by_date.get(shard.date, {}),
            adapter_route_binding=inputs.adapter_route_binding,
            raw_archive_root=inputs.raw_archive_root,
            matchup_tree_path=inputs.matchup_tree_path,
            matchup_tree_sha256=inputs.matchup_tree_sha256,
            runtime_route_code=inputs.runtime_route_code,
            route_sidecar=inputs.route_sidecars_by_date.get(shard.date),
            limit=int(smoke_limit),
        )
        corpus = DeviceResidentBootstrapCorpus.from_splits(
            source,
            _EmptyGames(),
            device=device,
            min_free_gib=8.0,
        )
        if not corpus.has_temporal_layout or not corpus.has_guide_targets:
            raise RuntimeError("r212 day corpus lost temporal/guide metadata")
        public_routes = source.route_tensor(
            device=device, expected_samples=int(corpus.total_samples)
        )
        route_projection = source.route_projection(
            expected_samples=int(corpus.total_samples)
        )
        quarantine_projection = source.quarantine_projection()
        chunks: list[dict[str, Any]] = []
        rows = options = 0
        try:
            batches = corpus.temporal_batches(
                train=True,
                batch_size=int(batch_decisions),
                shuffle=False,
                seed=0,
                epoch=0,
            )
            if not batches:
                raise RuntimeError(f"r212 source day has no temporal batches: {shard.date}")
            for index, game_ids in enumerate(batches):
                flattened = _flatten_latent_batch(
                    extract_temporal_latents(
                        base,
                        corpus,
                        game_ids,
                        public_routes,
                        inputs.adapter_route_binding,
                    )
                )
                chunk_name = f"chunk-{index:05d}.pt"
                chunk_path = temporary / chunk_name
                checkpoint.immutable_torch_save(flattened, chunk_path)
                digest = _sha256(chunk_path)
                _validate_latent_chunk(chunk_path, digest)
                chunk_rows = int(flattened["state_vec"].shape[0])
                chunk_options = int(flattened["option_hidden"].shape[0])
                chunks.append(
                    {
                        "path": chunk_name,
                        "sha256": digest,
                        "rows": chunk_rows,
                        "option_rows": chunk_options,
                    }
                )
                rows += chunk_rows
                options += chunk_options
        finally:
            del corpus
            if public_routes is not None:
                del public_routes
            torch.cuda.empty_cache()
        if not chunks or rows <= 0 or options <= 0:
            raise RuntimeError(f"r212 day latent extraction produced no rows: {shard.date}")
        manifest = {
            "schema": LATENT_SCHEMA,
            "created_at_utc": _utc_now(),
            "identity": identity,
            "rows": rows,
            "option_rows": options,
            "public_adapter_route_projection": route_projection,
            "quarantine_projection": quarantine_projection,
            "chunks": chunks,
        }
        _write_json_exclusive(temporary / "manifest.json", manifest)
        # Validate the temporary cache before its atomic publication.
        opened = _open_existing_latent_cache(temporary, identity)
        if opened is None:
            raise AssertionError("r212 temporary latent cache disappeared")
        try:
            os.rename(temporary, root)
        except FileExistsError:
            concurrent = _open_existing_latent_cache(root, identity)
            if concurrent is None:
                raise RuntimeError(f"r212 latent cache publication race: {root}")
            return concurrent
        published = _open_existing_latent_cache(root, identity)
        if published is None:
            raise AssertionError("r212 published latent cache disappeared")
        return published
    except BaseException:
        # Do not delete the partial tree: it is immutable diagnostic evidence.
        raise


def _build_partition_caches(
    *,
    cache_root: Path,
    partition: str,
    dates: Sequence[str],
    inputs: ValidatedInputs,
    base: torch.nn.Module,
    device: torch.device,
    batch_decisions: int,
    smoke_max_games: int,
) -> list[LatentCache]:
    caches: list[LatentCache] = []
    remaining = int(smoke_max_games)
    for date in dates:
        if smoke_max_games and remaining <= 0:
            break
        shard = inputs.shards_by_date[date]
        retained_records = int(shard.records) - len(inputs.quarantine_by_date.get(date, {}))
        if retained_records <= 0:
            raise RuntimeError(f"r212 quarantine emptied source day: {date}")
        limit = min(retained_records, remaining) if smoke_max_games else 0
        cache = _build_day_latent_cache(
            cache_root=cache_root,
            partition=partition,
            shard=shard,
            inputs=inputs,
            base=base,
            device=device,
            batch_decisions=batch_decisions,
            smoke_limit=limit,
        )
        caches.append(cache)
        if smoke_max_games:
            remaining -= limit
    if not caches:
        raise RuntimeError(f"r212 {partition} partition has no source games")
    return caches


def _cache_route_projections(
    partitions: Mapping[str, Sequence[LatentCache]],
) -> dict[str, Any]:
    """Seal every cache's raw/sidecar route proof into the final candidate."""

    result: dict[str, Any] = {}
    aggregate_decks: dict[str, dict[str, int]] = {}
    aggregate = {
        "retained_records": 0,
        "retained_guide_rows": 0,
        "packed_records": 0,
        "packed_guide_rows": 0,
        "turn_order_short_circuit_excluded_guide_rows": 0,
        "context_cap_excluded_guide_rows": 0,
    }
    all_dates: set[str] = set()
    all_full_materializations = True
    for partition, caches in sorted(partitions.items()):
        days: list[dict[str, Any]] = []
        seen_dates: set[str] = set()
        for cache in caches:
            if cache.date in seen_dates:
                raise RuntimeError("r212 route projection cache date is duplicated")
            seen_dates.add(cache.date)
            manifest_path = cache.root / "manifest.json"
            route_projection = dict(
                cache.manifest.get("public_adapter_route_projection") or {}
            )
            _validate_cached_public_route_projection(
                route_projection,
                dict(cache.manifest.get("identity") or {}),
            )
            deck_telemetry = dict(
                route_projection.get("retained_deck_fingerprint_telemetry") or {}
            )
            turn_order = dict(
                route_projection.get("compact_turn_order_short_circuit_exclusions")
                or {}
            )
            for field in (
                "retained_records",
                "retained_guide_rows",
                "packed_records",
                "packed_guide_rows",
                "context_cap_excluded_guide_rows",
            ):
                aggregate[field] += int(deck_telemetry[field])
            aggregate["turn_order_short_circuit_excluded_guide_rows"] += int(
                turn_order["excluded_guide_rows"]
            )
            for row in deck_telemetry["distribution"]:
                fingerprint = str(row["deck_fingerprint"])
                target = aggregate_decks.setdefault(
                    fingerprint,
                    {"records": 0, "decisions": 0, "guide_rows": 0},
                )
                for field in ("records", "decisions", "guide_rows"):
                    target[field] += int(row[field])
            all_dates.add(cache.date)
            all_full_materializations = all_full_materializations and (
                int(dict(cache.manifest.get("identity") or {}).get("smoke_game_limit") or 0)
                == 0
            )
            days.append(
                {
                    "date": cache.date,
                    "latent_manifest": {
                        "path": str(manifest_path),
                        "sha256": _sha256(manifest_path),
                    },
                    "public_adapter_route_projection": route_projection,
                }
            )
        if not days:
            raise RuntimeError("r212 route projection partition is empty")
        result[str(partition)] = {"days": days}
    aggregate_distribution = [
        {"deck_fingerprint": fingerprint, **counts}
        for fingerprint, counts in sorted(aggregate_decks.items())
    ]
    if (
        not aggregate_distribution
        or sum(int(row["records"]) for row in aggregate_distribution)
        != aggregate["retained_records"]
        or sum(int(row["guide_rows"]) for row in aggregate_distribution)
        != aggregate["retained_guide_rows"]
    ):
        raise RuntimeError("r212 aggregate exact-deck telemetry accounting drifted")
    if all_full_materializations and (
        all_dates != set(ALL_DATES)
        or aggregate["retained_guide_rows"]
        != aggregate["packed_guide_rows"]
        + aggregate["turn_order_short_circuit_excluded_guide_rows"]
        + aggregate["context_cap_excluded_guide_rows"]
    ):
        raise RuntimeError(
            "r212 full exact-deck/teacher-label distribution does not cover the "
            "authorized 20-day corpus"
        )
    result["exact_deck_telemetry"] = {
        "schema": "poke_bot.alakazam_guide2vec_r212_exact_deck_telemetry/v1",
        "scope": "all_materialized_partitions",
        "fingerprint_distribution_mode": "record_only_exact_60_card_multisets",
        "distinct_fingerprints": len(aggregate_distribution),
        "full_fixed_day_materialization": all_full_materializations,
        "dates": sorted(all_dates),
        **aggregate,
        "distribution_sha256": "sha256:"
        + hashlib.sha256(_canonical_json(aggregate_distribution)).hexdigest(),
        "distribution": aggregate_distribution,
    }
    return result


def _guide2vec_config_and_head(device: torch.device) -> tuple[Any, torch.nn.Module]:
    """Construct only the small, canonical guide head after base extraction."""

    try:
        from poke_bot.guide2vec import Guide2VecConfig, Guide2VecHead
    except ImportError as exc:
        raise RuntimeError("r212 Guide2Vec implementation is unavailable") from exc
    config = Guide2VecConfig(
        d_model=EXPECTED_D_MODEL,
        score_hidden_dim=256,
        score_bottleneck_dim=64,
        eligibility_hidden_dim=128,
        max_logit_bonus=MAX_GUIDE_LOGIT_BONUS,
    )
    head = Guide2VecHead(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in head.parameters())
    if parameter_count != EXPECTED_GUIDE2VEC_PARAMETERS:
        raise RuntimeError(
            "r212 Guide2Vec parameter inventory drifted: "
            f"expected={EXPECTED_GUIDE2VEC_PARAMETERS} actual={parameter_count}"
        )
    if parameter_count > MAX_GUIDE2VEC_PARAMETERS:
        raise RuntimeError("r212 Guide2Vec violates the small-sidecar parameter cap")
    if float(config.max_logit_bonus) != MAX_GUIDE_LOGIT_BONUS:
        raise RuntimeError("r212 Guide2Vec maximum bonus drifted from 0.05")
    if not all(parameter.requires_grad for parameter in head.parameters()):
        raise RuntimeError("r212 Guide2Vec unexpectedly has frozen parameters")
    return config, head


def _config_snapshot(config: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(config):
        value = dataclasses.asdict(config)
    elif isinstance(config, Mapping):
        value = dict(config)
    else:
        value = dict(vars(config))
    return {str(key): value[key] for key in sorted(value)}


def _padded_from_ragged(
    state: torch.Tensor,
    option_hidden: torch.Tensor,
    base_logits: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = offsets.to(device=state.device, dtype=torch.long)
    counts = offsets[1:] - offsets[:-1]
    rows = int(state.shape[0])
    if int(counts.numel()) != rows or bool((counts <= 0).any()):
        raise RuntimeError("r212 ragged latent offsets do not align with state rows")
    maximum = int(counts.max().item())
    padded_hidden = torch.zeros(
        (rows, maximum, EXPECTED_D_MODEL), device=state.device, dtype=state.dtype
    )
    padded_logits = torch.full(
        (rows, maximum), float("-inf"), device=state.device, dtype=state.dtype
    )
    total = int(option_hidden.shape[0])
    if total != int(offsets[-1].item()) or int(base_logits.numel()) != total:
        raise RuntimeError("r212 ragged option tensors do not align with offsets")
    row_ids = torch.repeat_interleave(torch.arange(rows, device=state.device), counts)
    starts = torch.repeat_interleave(offsets[:-1], counts)
    columns = torch.arange(total, device=state.device) - starts
    padded_hidden[row_ids, columns] = option_hidden
    padded_logits[row_ids, columns] = base_logits
    return padded_hidden, padded_logits, counts


def _iter_latent_batches(
    caches: Sequence[LatentCache],
    *,
    device: torch.device,
    batch_rows: int,
    shuffle_chunks: bool,
    seed: int,
) -> Iterator[dict[str, torch.Tensor]]:
    entries: list[tuple[Path, Mapping[str, Any]]] = []
    for cache in caches:
        for entry in cache.chunks:
            entries.append((cache.root / str(entry["path"]), entry))
    if shuffle_chunks:
        random.Random(int(seed)).shuffle(entries)
    for path, entry in entries:
        value = _validate_latent_chunk(path, str(entry["sha256"]))
        rows = int(value["state_vec"].shape[0])
        # Preserve ragged layout and only randomize cache-chunk order.  This
        # keeps gathering linear-time and is deterministic by epoch, while the
        # underlying feature corpus itself is an immutable multi-game stream.
        for start in range(0, rows, int(batch_rows)):
            end = min(rows, start + int(batch_rows))
            offsets_cpu = value["option_offsets"][start : end + 1].to(torch.long)
            option_start = int(offsets_cpu[0].item())
            option_end = int(offsets_cpu[-1].item())
            state = value["state_vec"][start:end].to(device=device, dtype=torch.float32)
            hidden = value["option_hidden"][option_start:option_end].to(
                device=device, dtype=torch.float32
            )
            base_logits = value["base_logits"][option_start:option_end].to(
                device=device, dtype=torch.float32
            )
            offsets = (offsets_cpu - option_start).to(device=device, dtype=torch.long)
            padded_hidden, padded_logits, counts = _padded_from_ragged(
                state, hidden, base_logits, offsets
            )
            yield {
                "state_vec": state,
                "option_hidden": padded_hidden,
                "base_logits": padded_logits,
                "n_options": counts,
                "guide_target_index": value["guide_target_index"][start:end].to(
                    device=device, dtype=torch.long
                ),
                "guide_confidence": value["guide_confidence"][start:end].to(
                    device=device, dtype=torch.float32
                ),
            }


@dataclass
class EpochMetrics:
    rank_nll_sum: float = 0.0
    rank_weight_sum: float = 0.0
    rank_correct: int = 0
    rank_rows: int = 0
    coverage_bce_sum: float = 0.0
    coverage_rows: int = 0
    coverage_correct: int = 0
    total_loss_sum: float = 0.0
    batches: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank_nll": self.rank_nll_sum / max(self.rank_weight_sum, 1e-12),
            "rank_accuracy": self.rank_correct / max(self.rank_rows, 1),
            "rank_rows": self.rank_rows,
            "coverage_bce": self.coverage_bce_sum / max(self.coverage_rows, 1),
            "coverage_accuracy": self.coverage_correct / max(self.coverage_rows, 1),
            "coverage_rows": self.coverage_rows,
            "total_loss": self.total_loss_sum / max(self.batches, 1),
            "batches": self.batches,
        }


def _guide_losses(
    guide_scores: torch.Tensor,
    eligibility_logits: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    counts: torch.Tensor,
    *,
    coverage_weight: float,
) -> tuple[torch.Tensor, EpochMetrics, torch.Tensor, torch.Tensor]:
    if guide_scores.ndim != 2 or eligibility_logits.ndim not in (1, 2):
        raise RuntimeError("r212 Guide2Vec forward output has invalid dimensions")
    if eligibility_logits.ndim == 2:
        if eligibility_logits.shape[1] != 1:
            raise RuntimeError("r212 Guide2Vec eligibility output must be scalar per stage")
        eligibility_logits = eligibility_logits.squeeze(1)
    rows, maximum = guide_scores.shape
    if (
        int(eligibility_logits.numel()) != rows
        or target.shape != (rows,)
        or confidence.shape != (rows,)
        or counts.shape != (rows,)
    ):
        raise RuntimeError("r212 Guide2Vec batch rows are misaligned")
    legal = torch.arange(maximum, device=guide_scores.device).unsqueeze(0) < counts.unsqueeze(1)
    if bool((counts <= 0).any()):
        raise RuntimeError("r212 received an empty legal stage")
    masked_scores = guide_scores.masked_fill(~legal, float("-inf"))
    labeled = (target >= 0) & (counts > 1)
    invalid = labeled & (target >= counts)
    if bool(invalid.any()):
        raise RuntimeError("r212 guide target escaped current legal options")
    coverage_target = labeled.to(dtype=eligibility_logits.dtype)
    positives = int(labeled.sum().item())
    negatives = int((~labeled).sum().item())
    pos_weight = None
    if positives and negatives:
        pos_weight = torch.tensor(
            negatives / positives, device=eligibility_logits.device, dtype=eligibility_logits.dtype
        )
    coverage_bce = F.binary_cross_entropy_with_logits(
        eligibility_logits,
        coverage_target,
        pos_weight=pos_weight,
        reduction="mean",
    )
    if positives:
        rows_index = torch.nonzero(labeled, as_tuple=False).flatten()
        per_row = F.cross_entropy(
            masked_scores.index_select(0, rows_index),
            target.index_select(0, rows_index),
            reduction="none",
        )
        weights = confidence.index_select(0, rows_index).clamp(min=0.0)
        if float(weights.sum().item()) <= 0.0:
            # A zero-confidence target has no rank authority, but all coverage
            # rows still train.  Anchor a finite zero to the head graph.
            rank_loss = guide_scores.sum() * 0.0
        else:
            rank_loss = (per_row * weights).sum() / weights.sum()
        predicted = masked_scores.index_select(0, rows_index).argmax(dim=1)
        rank_correct = int(
            (predicted == target.index_select(0, rows_index)).sum().item()
        )
        rank_nll_sum = float((per_row * weights).detach().sum().item())
        rank_weight_sum = float(weights.detach().sum().item())
    else:
        rank_loss = guide_scores.sum() * 0.0
        rank_correct = 0
        rank_nll_sum = 0.0
        rank_weight_sum = 0.0
    total = rank_loss + float(coverage_weight) * coverage_bce
    coverage_prediction = torch.sigmoid(eligibility_logits) >= 0.5
    metrics = EpochMetrics(
        rank_nll_sum=rank_nll_sum,
        rank_weight_sum=rank_weight_sum,
        rank_correct=rank_correct,
        rank_rows=positives,
        coverage_bce_sum=float(coverage_bce.detach().item()) * rows,
        coverage_rows=rows,
        coverage_correct=int((coverage_prediction == labeled).sum().item()),
        total_loss_sum=float(total.detach().item()),
        batches=1,
    )
    return total, metrics, masked_scores, labeled


def _accumulate(total: EpochMetrics, update: EpochMetrics) -> None:
    total.rank_nll_sum += update.rank_nll_sum
    total.rank_weight_sum += update.rank_weight_sum
    total.rank_correct += update.rank_correct
    total.rank_rows += update.rank_rows
    total.coverage_bce_sum += update.coverage_bce_sum
    total.coverage_rows += update.coverage_rows
    total.coverage_correct += update.coverage_correct
    total.total_loss_sum += update.total_loss_sum
    total.batches += update.batches


def run_epoch(
    head: torch.nn.Module,
    caches: Sequence[LatentCache],
    *,
    device: torch.device,
    batch_rows: int,
    coverage_weight: float,
    optimizer: torch.optim.Optimizer | None,
    seed: int,
    collect_calibration: bool = False,
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    """Run a deterministic cache epoch; optimizer=None is exact eval mode."""

    training = optimizer is not None
    if training:
        _reject_r212_training_under_r226()
    head.train(training)
    aggregate = EpochMetrics()
    calibration: dict[str, list[torch.Tensor]] | None = (
        {"probability": [], "correct": [], "eligible": [], "applicable": []}
        if collect_calibration
        else None
    )
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in _iter_latent_batches(
            caches,
            device=device,
            batch_rows=batch_rows,
            shuffle_chunks=training,
            seed=seed,
        ):
            guide_scores, eligibility_logits = head(
                batch["state_vec"],
                batch["option_hidden"],
                batch["base_logits"],
                n_options=batch["n_options"],
            )
            total, metrics, masked_scores, labeled = _guide_losses(
                guide_scores,
                eligibility_logits,
                batch["guide_target_index"],
                batch["guide_confidence"],
                batch["n_options"],
                coverage_weight=coverage_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
            _accumulate(aggregate, metrics)
            if calibration is not None:
                if eligibility_logits.ndim == 2:
                    eligibility_logits = eligibility_logits.squeeze(1)
                predicted = masked_scores.argmax(dim=1)
                correct = labeled & (predicted == batch["guide_target_index"])
                calibration["probability"].append(torch.sigmoid(eligibility_logits).detach().cpu())
                calibration["correct"].append(correct.detach().cpu())
                calibration["eligible"].append(labeled.detach().cpu())
                calibration["applicable"].append(
                    (batch["n_options"] >= 2).detach().cpu()
                )
    result = aggregate.as_dict()
    tensors = None
    if calibration is not None:
        tensors = {name: torch.cat(values) if values else torch.empty(0) for name, values in calibration.items()}
    return result, tensors


def calibrate_abstention(validation: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Calibrate a runtime threshold against *all* non-singleton stages.

    A guide-unavailable row is an abstention target, not an ignored precision
    denominator.  Otherwise a low threshold could pass validation by only
    measuring teacher-labeled rows while the runtime applies bonuses to masked
    stages.  ``eligible_coverage`` remains the fraction of labeled stages that
    can use the guide, but precision is deliberately over every applied stage.
    """

    probability = validation["probability"].to(dtype=torch.float64)
    correct = validation["correct"].to(dtype=torch.bool)
    eligible = validation["eligible"].to(dtype=torch.bool)
    applicable = validation["applicable"].to(dtype=torch.bool)
    if not (
        probability.shape == correct.shape == eligible.shape == applicable.shape
    ):
        raise RuntimeError("r212 validation calibration rows are misaligned")
    if probability.numel() == 0 or int(eligible.sum().item()) == 0:
        return {
            "schema": "poke_bot.alakazam_guide2vec_abstention_calibration/v1",
            "status": "abstain_all_no_validation_labels",
            "threshold": 1.0,
            "minimum_precision": 0.70,
            "applied_rows": 0,
            "precision": None,
            "eligible_coverage": 0.0,
            "applied_labeled_rows": 0,
        }
    candidates = torch.unique(
        torch.cat(
            [
                torch.linspace(0.0, 1.0, 101, dtype=torch.float64),
                probability.clamp(0.0, 1.0),
            ]
        )
    ).sort().values
    best: tuple[float, float, int, float] | None = None
    for threshold in candidates.tolist():
        applied = (probability >= threshold) & applicable
        applied_rows = int(applied.sum().item())
        if not applied_rows:
            continue
        precision = float(correct[applied].float().mean().item())
        if precision < 0.70:
            continue
        applied_labeled_rows = int((applied & eligible).sum().item())
        coverage = applied_labeled_rows / max(int(eligible.sum().item()), 1)
        candidate = (coverage, precision, applied_rows, float(threshold))
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        return {
            "schema": "poke_bot.alakazam_guide2vec_abstention_calibration/v1",
            "status": "abstain_all_precision_floor_not_met",
            "threshold": 1.0,
            "minimum_precision": 0.70,
            "applied_rows": 0,
            "precision": None,
            "eligible_coverage": 0.0,
            "applied_labeled_rows": 0,
        }
    coverage, precision, applied_rows, threshold = best
    applied = (probability >= threshold) & applicable
    return {
        "schema": "poke_bot.alakazam_guide2vec_abstention_calibration/v1",
        "status": "validation_calibrated",
        "threshold": threshold,
        "minimum_precision": 0.70,
        "applied_rows": applied_rows,
        "precision": precision,
        "eligible_coverage": coverage,
        "validation_eligible_rows": int(eligible.sum().item()),
        "validation_applicable_rows": int(applicable.sum().item()),
        "applied_labeled_rows": int((applied & eligible).sum().item()),
    }


def _runtime_config_from_validation(
    config: Any,
    calibration: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Make the validation-only threshold the immutable runtime gate.

    ``Guide2VecHead.rerank`` reads ``config.min_eligibility`` directly.  Merely
    keeping a calibration note in checkpoint metadata would therefore be a
    provenance-only no-op.  Replace the selected head config before serializing
    the final candidate and bind the complete runtime configuration by digest.
    """

    status = str(calibration.get("status") or "")
    if status not in {
        "validation_calibrated",
        "abstain_all_no_validation_labels",
        "abstain_all_precision_floor_not_met",
    }:
        raise RuntimeError("r212 validation calibration status is invalid")
    raw_threshold = calibration.get("threshold")
    if isinstance(raw_threshold, bool):
        raise RuntimeError("r212 validation eligibility threshold is invalid")
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("r212 validation eligibility threshold is invalid") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise RuntimeError("r212 validation eligibility threshold leaves [0, 1]")
    if status != "validation_calibrated" and threshold != 1.0:
        raise RuntimeError("r212 abstain-all calibration must bind threshold=1.0")
    if status == "validation_calibrated":
        try:
            precision = float(calibration.get("precision"))
            applied_rows = int(calibration.get("applied_rows"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("r212 calibrated eligibility receipt is malformed") from exc
        if not math.isfinite(precision) or precision < 0.70 or applied_rows <= 0:
            raise RuntimeError("r212 calibrated eligibility receipt misses its validation gate")
    if not dataclasses.is_dataclass(config):
        raise RuntimeError("r212 Guide2Vec config cannot bind a calibrated threshold")
    runtime_config = dataclasses.replace(config, min_eligibility=threshold)
    snapshot = _config_snapshot(runtime_config)
    if float(snapshot.get("min_eligibility", -1.0)) != threshold:
        raise RuntimeError("r212 calibrated threshold did not bind the runtime config")
    config_sha256 = "sha256:" + hashlib.sha256(_canonical_json(snapshot)).hexdigest()
    runtime = {
        "schema": RUNTIME_CONFIG_SCHEMA,
        "guide2vec_config": snapshot,
        "guide2vec_config_sha256": config_sha256,
        "eligibility_threshold": threshold,
        "eligibility_threshold_field": "Guide2VecConfig.min_eligibility",
        "threshold_source": "validation_only_calibration",
        "selection_metric": SELECTION_METRIC,
        "validation_calibration": dict(calibration),
    }
    runtime["runtime_config_sha256"] = "sha256:" + hashlib.sha256(
        _canonical_json(runtime)
    ).hexdigest()
    return runtime_config, runtime


def _epoch_checkpoint_payload(
    *,
    config: Any,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    contract: Mapping[str, Any],
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SIDE_CAR_SCHEMA,
        "kind": "epoch_candidate",
        "epoch": int(epoch),
        "guide2vec_config": _config_snapshot(config),
        "guide2vec_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in head.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "base_checkpoint": dict(contract["base"]),
        "training_contract_sha256": "sha256:" + hashlib.sha256(_canonical_json(contract)).hexdigest(),
        "train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
        "validation_calibration": dict(calibration),
        "authority": dict(contract["authority"]),
    }


def _load_state_dict_strict(head: torch.nn.Module, state: Mapping[str, Any]) -> None:
    result = head.load_state_dict(dict(state), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("r212 Guide2Vec selected checkpoint state does not load strictly")


def _run_training(args: argparse.Namespace, inputs: ValidatedInputs) -> dict[str, Any]:
    _reject_r212_training_under_r226()
    device = torch.device(args.device)
    _assert_blackwell_cuda(device)
    _set_determinism(int(args.seed))
    contract = _run_contract(args, inputs)
    preflight = _host_preflight(args=args, contract=contract, device=device)
    run_dir = _resolve_run_dir(args, contract)
    final_receipt = run_dir / "TRAINING_RECEIPT.json"
    if final_receipt.is_file():
        existing = _read_json_object(final_receipt)
        if dict(existing.get("training_contract") or {}) != contract:
            raise RuntimeError("existing r212 receipt has a different immutable contract")
        return existing
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(run_dir / "RUN_CONTRACT.json", contract)
    # A resumed run rechecks live capacity but preserves the original immutable
    # preflight receipt rather than trying to overwrite its timestamp.
    preflight_path = run_dir / "PREFLIGHT_RECEIPT.json"
    if not preflight_path.exists():
        _write_json_exclusive(preflight_path, preflight)
    _append_jsonl(
        run_dir / "progress.jsonl",
        {"event": "started", "at_utc": _utc_now(), "training_contract": contract},
    )

    base = _load_frozen_base(inputs, device)
    try:
        cache_root = run_dir / "latent-cache"
        train_caches = _build_partition_caches(
            cache_root=cache_root,
            partition="train",
            dates=TRAIN_DATES,
            inputs=inputs,
            base=base,
            device=device,
            batch_decisions=int(args.batch_decisions),
            smoke_max_games=int(args.smoke_max_games),
        )
        validation_caches = _build_partition_caches(
            cache_root=cache_root,
            partition="validation",
            dates=VALIDATION_DATES,
            inputs=inputs,
            base=base,
            device=device,
            batch_decisions=int(args.batch_decisions),
            smoke_max_games=int(args.smoke_max_games),
        )
    finally:
        # The base is never needed after its frozen public latent cache has
        # been sealed.  Releasing it before head fitting makes the job safe on
        # the 48 GiB isolated GPU even when later caches are large.
        del base
        torch.cuda.empty_cache()

    config, head = _guide2vec_config_and_head(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)
    best_metric = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_calibration: dict[str, Any] = {}
    patience_left = 1
    epoch_records: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics, _ = run_epoch(
            head,
            train_caches,
            device=device,
            batch_rows=int(args.batch_decisions),
            coverage_weight=float(args.coverage_weight),
            optimizer=optimizer,
            seed=int(args.seed) + epoch * 10_007,
        )
        validation_metrics, calibration_rows = run_epoch(
            head,
            validation_caches,
            device=device,
            batch_rows=int(args.batch_decisions),
            coverage_weight=float(args.coverage_weight),
            optimizer=None,
            seed=0,
            collect_calibration=True,
        )
        assert calibration_rows is not None
        calibration = calibrate_abstention(calibration_rows)
        # The owner-selected criterion is exactly confidence-weighted listwise
        # CE on validation.  Coverage BCE trains the abstention head, but it
        # must not influence checkpoint selection or early stopping.
        metric = float(validation_metrics["rank_nll"])
        payload = _epoch_checkpoint_payload(
            config=config,
            head=head,
            optimizer=optimizer,
            epoch=epoch,
            contract=contract,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            calibration=calibration,
        )
        epoch_path, epoch_digest = _content_addressed_torch_save(
            payload, directory=run_dir / "checkpoints", prefix=f"epoch-{epoch:02d}"
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_calibration": calibration,
            "selection_metric_name": SELECTION_METRIC,
            "selection_metric": metric,
            "checkpoint": {"path": str(epoch_path), "sha256": epoch_digest},
        }
        epoch_records.append(record)
        _append_jsonl(run_dir / "progress.jsonl", {"event": "epoch_complete", "at_utc": _utc_now(), **record})
        if metric < best_metric - 1e-12:
            best_metric = metric
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()}
            best_calibration = calibration
            patience_left = 1
        else:
            patience_left -= 1
            if patience_left < 0:
                _append_jsonl(
                    run_dir / "progress.jsonl",
                    {"event": "early_stop", "at_utc": _utc_now(), "after_epoch": epoch, "best_epoch": best_epoch},
                )
                break
    if best_state is None or best_epoch <= 0:
        raise RuntimeError("r212 training did not select a validation checkpoint")
    _load_state_dict_strict(head, best_state)
    runtime_config, runtime_config_receipt = _runtime_config_from_validation(
        config,
        best_calibration,
    )
    # The final runtime consumes this exact selected head object.  Rebinding
    # its immutable config is what applies the validation threshold to rerank.
    head.config = runtime_config
    if (
        float(head.config.min_eligibility)
        != float(runtime_config_receipt["eligibility_threshold"])
        or sum(parameter.numel() for parameter in head.parameters())
        != EXPECTED_GUIDE2VEC_PARAMETERS
    ):
        raise RuntimeError("r212 selected head did not retain the calibrated runtime config")

    # Only now materialize the sealed heldout dates.  Nothing from these two
    # days has influenced optimizer state, early stopping, or calibration.
    heldout_base = _load_frozen_base(inputs, device)
    try:
        heldout_caches = _build_partition_caches(
            cache_root=run_dir / "latent-cache",
            partition="heldout",
            dates=HELDOUT_DATES,
            inputs=inputs,
            base=heldout_base,
            device=device,
            batch_decisions=int(args.batch_decisions),
            smoke_max_games=int(args.smoke_max_games),
        )
    finally:
        # Explicit lifetime keeps the heldout base from leaking into the
        # post-selection head-only evaluation, and heldout never touches the
        # optimizer/calibration/early-stop path.
        del heldout_base
        torch.cuda.empty_cache()
    heldout_metrics, _ = run_epoch(
        head,
        heldout_caches,
        device=device,
        batch_rows=int(args.batch_decisions),
        coverage_weight=float(args.coverage_weight),
        optimizer=None,
        seed=0,
        collect_calibration=False,
    )
    heldout_agreement = float(heldout_metrics["rank_accuracy"])
    heldout_gate = {
        "minimum": 0.90,
        "scope": "teacher_labeled_legal_stages_only",
        "observed": heldout_agreement,
        "passed": heldout_agreement >= 0.90,
    }
    if not heldout_gate["passed"]:
        raise RuntimeError(
            "r212 heldout teacher-agreement gate failed: "
            f"observed={heldout_agreement:.6f} minimum=0.900000"
        )
    route_projections = _cache_route_projections(
        {
            "train": train_caches,
            "validation": validation_caches,
            "heldout": heldout_caches,
        }
    )
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if head.training or any(parameter.requires_grad for parameter in head.parameters()):
        raise RuntimeError("r212 final Guide2Vec candidate is not frozen")
    _write_json_exclusive(
        run_dir / "GUIDE2VEC_RUNTIME_CONFIG.json",
        runtime_config_receipt,
    )
    try:
        from poke_bot.guide2vec import FrozenBaseIdentity, make_checkpoint_payload
    except ImportError as exc:
        raise RuntimeError("r212 Guide2Vec checkpoint helpers are unavailable") from exc
    base_identity = FrozenBaseIdentity(
        submission_id=int(R195_SUBMISSION_ID),
        checkpoint_sha256=inputs.checkpoint_sha256,
        checkpoint_bytes=inputs.checkpoint_bytes,
        bundle_sha256=R195_BUNDLE_SHA256,
        model_config_sha256=inputs.model_config_sha256,
        feature_schema_sha256=inputs.manifest_sha256,
    )
    final_payload = make_checkpoint_payload(
        head,
        base_identity,
        metadata={
            "kind": "frozen_candidate",
            "training_contract_sha256": "sha256:"
            + hashlib.sha256(_canonical_json(contract)).hexdigest(),
            "selected_epoch": best_epoch,
            "validation_calibration": best_calibration,
            "guide2vec_runtime_config": runtime_config_receipt,
            "guide2vec_runtime_config_sha256": runtime_config_receipt[
                "runtime_config_sha256"
            ],
            "heldout_metrics": heldout_metrics,
            "heldout_teacher_agreement_gate": heldout_gate,
            "matchup_adapter": dict(inputs.adapter_identity),
            "matchup_adapter_route_binding": inputs.adapter_route_binding.as_dict(),
            "matchup_tree": {
                "path": str(inputs.matchup_tree_path),
                "sha256": inputs.matchup_tree_sha256,
                "runtime_enabled": True,
            },
            "public_route_projections": route_projections,
            "authority": dict(contract["authority"]),
        },
    )
    if (
        not isinstance(final_payload.get("config"), Mapping)
        or float(final_payload["config"].get("min_eligibility", -1.0))
        != float(runtime_config_receipt["eligibility_threshold"])
        or dict(final_payload.get("metadata") or {}).get(
            "guide2vec_runtime_config_sha256"
        )
        != runtime_config_receipt["runtime_config_sha256"]
    ):
        raise RuntimeError("r212 final checkpoint failed to bind the calibrated runtime config")
    final_path, final_digest = _content_addressed_torch_save(
        final_payload, directory=run_dir / "checkpoints", prefix="guide2vec-r212"
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete_offline_candidate_only",
        "completed_at_utc": _utc_now(),
        "training_contract": contract,
        "inputs": inputs.as_dict(),
        "source_snapshot": dict(inputs.source_snapshot),
        "public_route_projections": route_projections,
        "guide2vec": {
            "checkpoint": str(final_path),
            "checkpoint_sha256": final_digest,
            "parameter_count": EXPECTED_GUIDE2VEC_PARAMETERS,
            "selected_epoch": best_epoch,
            "validation_calibration": best_calibration,
            "selection_metric_name": SELECTION_METRIC,
            "selection_metric": best_metric,
            "runtime_config": runtime_config_receipt,
        },
        "epochs": epoch_records,
        "heldout": {
            "dates": list(HELDOUT_DATES),
            "metrics": heldout_metrics,
            "teacher_agreement_gate": heldout_gate,
            "training_eligible": False,
        },
        "authority": dict(contract["authority"]),
    }
    _write_json_exclusive(final_receipt, receipt)
    _append_jsonl(
        run_dir / "progress.jsonl",
        {"event": "complete", "at_utc": _utc_now(), "receipt": str(final_receipt), "checkpoint_sha256": final_digest},
    )
    return receipt


def _check_summary(args: argparse.Namespace, inputs: ValidatedInputs) -> dict[str, Any]:
    device = torch.device(args.device)
    _assert_blackwell_cuda(device)
    _set_determinism(int(args.seed))
    contract = _run_contract(args, inputs)
    preflight = _host_preflight(args=args, contract=contract, device=device)
    config, head = _guide2vec_config_and_head(device)
    try:
        # Load/model construction is part of preflight but no corpus data is
        # materialized and no output root is created in --check mode.
        base = _load_frozen_base(inputs, device)
        try:
            base_parameters = sum(parameter.numel() for parameter in base.parameters())
        finally:
            del base
            torch.cuda.empty_cache()
        return {
            "schema": SCHEMA,
            "status": "ready",
            "training_contract": contract,
            "inputs": inputs.as_dict(),
            "run_dir": str(_resolve_run_dir(args, contract)),
            "host_preflight": preflight,
            "device": {
                "requested": str(device),
                "name": torch.cuda.get_device_name(device),
                "blackwell_uuid": BLACKWELL_UUID,
                "base_parameter_count": base_parameters,
            },
            "guide2vec": {
                "config": _config_snapshot(config),
                "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
            },
        }
    finally:
        del head
        torch.cuda.empty_cache()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify inputs/GPU without writing")
    mode.add_argument("--run", action="store_true", help="train only the isolated Guide2Vec sidecar")
    parser.add_argument("--config", type=Path, help="typed r212 job specification")
    parser.add_argument("--output-root", type=Path, help="root for content-addressed r212 artifacts")
    parser.add_argument("--output-dir", type=Path, help="explicit immutable r212 run directory")
    parser.add_argument("--checkpoint", type=Path, help="exact r195 NO-RTP checkpoint")
    parser.add_argument("--matchup-tree", type=Path, help="exact runtime-enabled r195 V6 public matchup tree")
    parser.add_argument(
        "--submission-bundle",
        type=Path,
        help="exact r195 NO-RTP submission.tar.gz used to prove route runtime code",
    )
    parser.add_argument(
        "--submission-bundle-sha256",
        help="exact SHA-256 of --submission-bundle",
    )
    parser.add_argument(
        "--raw-archive-root",
        type=Path,
        help="header-bound raw daily archive root for train/validation route reconstruction",
    )
    parser.add_argument(
        "--route-sidecar-manifest",
        type=Path,
        help="immutable heldout (Jul22--23) public-route sidecar manifest",
    )
    parser.add_argument(
        "--route-sidecar-manifest-sha256",
        help="exact SHA-256 of --route-sidecar-manifest",
    )
    parser.add_argument("--protected-corpus", type=Path, help="r212 protected Jul4--23 corpus pointer")
    parser.add_argument("--owner-contract", type=Path, help="exact r212 owner contract")
    parser.add_argument("--device", default="cuda:0", help="must resolve to the Blackwell UUID")
    parser.add_argument("--epochs", type=int, help=f"1..{MAX_EPOCHS}; default from spec or {MAX_EPOCHS}")
    parser.add_argument("--learning-rate", type=float, help="Guide2Vec AdamW learning rate")
    parser.add_argument("--batch-decisions", type=int, help="bounded temporal extraction/head batch budget")
    parser.add_argument("--coverage-weight", type=float, help="eligibility BCE multiplier")
    parser.add_argument("--seed", type=int, help="deterministic seed")
    parser.add_argument("--smoke-max-games", type=int, help="limit each partition for a fast isolated smoke")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.run:
        # This is intentionally ahead of job-spec parsing, input validation,
        # CUDA discovery, and all output-root handling.  `--check` remains a
        # separately read-only historical verifier.
        _reject_r212_training_under_r226()
    _apply_job_spec(args)
    inputs = validate_inputs(args)
    if args.check:
        print(json.dumps(_check_summary(args, inputs), indent=2, sort_keys=True))
        return 0
    receipt = _run_training(args, inputs)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
