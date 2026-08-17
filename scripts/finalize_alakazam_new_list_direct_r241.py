#!/usr/bin/env python3
"""Build the receipt-gated, direct-policy-only r241 terminal package.

This is intentionally an offline finalizer.  It has no submission client,
does not start a service, and never changes a shared submission queue.  Its
only durable outputs are an immutable archive, one immutable queue
authorization, and one immutable finalizer receipt.

The runtime source is treated as untrusted packaging input.  In particular,
its ``cg/`` directory is *not* copied wholesale.  The archive receives only
the sealed r236 wrapper members and complete checksum-bound native set from
the separately attested official runtime.  This prevents a legacy MCTS/RTP
package, including its old libcg or search sidecars, from leaking into the
direct-policy candidate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from poke_bot import r241_checkpoint_receipts as checkpoint_receipts  # noqa: E402
CONTRACT_PATH = ROOT / "state/alakazam-new-list-direct-policy-r241.json"
CONTRACT_SHA256 = (
    "sha256:2f9ca8fc0d4cb2a7c6acbc12ecce3e96143a2c9e318e198276ea0dd66bb30c7d"
)
OFFICIAL_LIBCG_STAGING_PATH = (
    ROOT / "state/alakazam-new-list-direct-r241-official-libcg-staging.json"
)
OFFICIAL_LIBCG_STAGING_SHA256 = (
    "sha256:894fa309471dcc794dc8e65be69dbceb1bd914bb4b68e141f6794a2b2c6756a0"
)
EXPERT_WINDOW_STAGING_PATH = (
    ROOT / "state/alakazam-new-list-direct-r241-expert-window-staging.json"
)
EXPERT_WINDOW_STAGING_SHA256 = (
    "sha256:508249e6ebc256d8baeacb918e914d309b5afc21a0cbbdb38cf561bc281e8733"
)
EXPERT_WINDOW_CANONICAL_RECEIPT_SHA256 = (
    "sha256:09848f04a6c863a02c517fdcd5b7a61a139eceafd3348aa2a08705fd6e971a16"
)
PEAK_R195_HEAD_ROLE_MAP_PATH = (
    ROOT / "state/alakazam-new-list-direct-r241-strategic-head-roles.json"
)
PEAK_R195_HEAD_ROLE_MAP_SHA256 = (
    "sha256:5b331159ab6e6bced77209f4d8b67a77ebebc78728c7a671384363dc0faaa356"
)

CONTRACT_SCHEMA = "poke_bot.alakazam_new_list_direct_policy_r241/v1"
TERMINAL_REFRESH_SCHEMA = "poke_bot.terminal_expert_soft_refresh/v1"
RUNTIME_PROFILE_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_runtime_profile/v1"
)
PACKAGE_SCHEMA = "poke_bot.alakazam_new_list_direct_policy_r241_package/v1"
QUEUE_AUTHORIZATION_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_queue_authorization/v1"
)
AUTHORIZATION_BINDING_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_queue_authorization_binding/v1"
)
FINALIZER_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_terminal_finalizer/v1"
)
TURN_ORDER_PROFILE_SCHEMA = "poke_bot.submission_turn_order_profile/v1"
MATCHUP_RUNTIME_ACTIVATION_SCHEMA = checkpoint_receipts.MATCHUP_RUNTIME_ACTIVATION_SCHEMA
MODEL_RUNTIME_ACTIVATION_SCHEMA = checkpoint_receipts.MODEL_RUNTIME_ACTIVATION_SCHEMA
OFFICIAL_LIBCG_STAGING_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_official_libcg_staging/v1"
)
OFFICIAL_LIBCG_PREFLIGHT_SCHEMA = "poke_bot.r241_official_libcg_direct_policy_preflight/v1"
EXPERT_WINDOW_STAGING_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_expert_window_staging/v1"
)

OWNER_REVISION = 241
LATEST_OWNER_CLARIFICATION_REVISION = 251
# This is an immutable historical staging receipt.  Its revision deliberately
# remains 245 even though the live r241 owner clarification is now 248.
EXPERT_WINDOW_STAGING_OWNER_CLARIFICATION_REVISION = 245
TERMINAL_ITERATION = 9
NEXT_ITERATION = 10
TERMINAL_CHECKPOINT_NAME = "expert_before_iter_00010.pt"
AUTHORIZATION_BINDING_NAME = "r241-terminal-single-use-queue-authorization-binding.json"
OFFICIAL_LIBCG_PREFLIGHT_NAME = "r241_official_libcg_direct_policy_preflight.json"
PARENT_R195_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
PARENT_R195_TYPED_SOURCE_SHA256 = (
    "sha256:e37cf1d3e638c3aed56230c9fa970c61e6c1ed8b4bd3024de259cb9847c31e48"
)
ARCHIVE_LIMIT_MIB = Decimal("197.7")
ARCHIVE_MAX_BYTES = int(
    (ARCHIVE_LIMIT_MIB * Decimal(1024**2)).to_integral_value(rounding=ROUND_FLOOR)
)
OFFICIAL_LINUX_LIBCG_SHA256 = (
    "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7"
)
OFFICIAL_LINUX_LIBCG_SIZE_BYTES = 1_342_400
LEARNER_R195_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
H10_DIRECT_MATCHUP_TREE_SHA256 = (
    "sha256:da223c4903dd37511e5cb7656fe405bc0baac085be4f131faef136b7056c4588"
)
H10_DIRECT_MATCHUP_TREE_SIZE_BYTES = 2_509_756
_DEFAULT_STAGING_HOST_RECEIPTS = {
    "inzi": "sha256:165af79e33b9851d36c972503012654ed9db86222b7f79e7fac9b8bd00a38965",
    "elmo": "sha256:91506428871349a47cd4fa4af76e45aaafde779a8e462e1b08e3c631d908973b",
}

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_FORBIDDEN_ASSET_COMPONENT = re.compile(
    r"(?:^|[._-])(?:mcts|rtp|belief|search|recursive[_-]turn[_-]planner)(?:$|[._-])",
    re.IGNORECASE,
)
_FORBIDDEN_RUNTIME_SOURCE = re.compile(
    r"\b(?:mcts|rtp|belief(?:_decks)?|search_config|search_tree|"
    r"search(?:begin|step|end)|recursive[_ -]?turn[_ -]?planner)\b",
    re.IGNORECASE,
)
_RESERVED_RUNTIME_MEMBERS = frozenset(
    {
        "deck.csv",
        "model.pt",
        "matchup_tree.json",
        "matchup_runtime_activation.json",
        "model_runtime_activation.json",
        "package_manifest.json",
        "runtime_profile.json",
        "turn_order_profile.json",
    }
)
_OFFICIAL_CG_WRAPPER_MEMBERS = frozenset(
    {"__init__.py", "api.py", "game.py", "sim.py"}
)
_FORBIDDEN_UNSEALED_RUNTIME_SUFFIXES = frozenset(
    {".pyc", ".pyo", ".so", ".dylib", ".dll", ".pt", ".pth", ".ckpt"}
)


class R241FinalizerError(RuntimeError):
    """The direct-policy terminal package cannot safely be finalized."""


@dataclass(frozen=True)
class FileIdentity:
    path: Path
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class TerminalEvidence:
    run_dir: Path
    contract_path: Path
    contract_sha256: str
    terminal_commit: FileIdentity
    terminal_parent: FileIdentity
    terminal_refresh: FileIdentity
    terminal_rehearsal: FileIdentity
    terminal_checkpoint: FileIdentity
    expert_window: "ExpertWindowEvidence"

    def as_dict(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "contract": {
                "path": str(self.contract_path),
                "sha256": self.contract_sha256,
            },
            "durable_iteration_commit": {
                "iteration": TERMINAL_ITERATION,
                **self.terminal_commit.as_dict(),
            },
            "iter_00009_parent": self.terminal_parent.as_dict(),
            "terminal_expert_refresh": self.terminal_refresh.as_dict(),
            "terminal_five_epoch_rehearsal_receipt": self.terminal_rehearsal.as_dict(),
            "terminal_checkpoint": self.terminal_checkpoint.as_dict(),
            "expert_window": self.expert_window.as_dict(),
        }


@dataclass(frozen=True)
class ExpertWindowEvidence:
    staging_receipt: FileIdentity
    canonical_receipt_sha256: str
    immutable_window_receipt_sha256: str
    validated_episodes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "staging_receipt": self.staging_receipt.as_dict(),
            "canonical_receipt_sha256": self.canonical_receipt_sha256,
            "immutable_window_receipt_sha256": self.immutable_window_receipt_sha256,
            "window": {
                "start": "2026-07-22",
                "end": "2026-08-10",
                "days": 20,
                "validated_episodes": self.validated_episodes,
            },
        }


@dataclass(frozen=True)
class PeakR195HeadInventory:
    source: FileIdentity
    head_names: tuple[str, ...]
    fusion_route_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "head_role_map": self.source.as_dict(),
            "head_names": list(self.head_names),
            "fusion_route_ids": list(self.fusion_route_ids),
        }


@dataclass(frozen=True)
class MatchupRuntimeEvidence:
    tree: FileIdentity
    activation: FileIdentity
    accepted_archetype_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "matchup_tree": self.tree.as_dict(),
            "runtime_activation": self.activation.as_dict(),
            "accepted_archetype_ids": list(self.accepted_archetype_ids),
            "adapter_runtime_enabled": True,
            "action_selector": "direct_policy_only",
        }


@dataclass(frozen=True)
class ModelRuntimeEvidence:
    activation: FileIdentity
    active_head_names: tuple[str, ...]
    active_fusion_route_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "runtime_activation": self.activation.as_dict(),
            "active_head_names": list(self.active_head_names),
            "active_fusion_route_ids": list(self.active_fusion_route_ids),
            "all_peak_r195_non_combo_heads_active": True,
            "fusion_routes_active": True,
            "combo_state_loss_weight": 0.0,
            "combo_state_route_enabled": False,
            "matchup_adapter_runtime_enabled": True,
        }


@dataclass(frozen=True)
class OfficialCgStaging:
    staging_receipt: FileIdentity
    canonical_contract: FileIdentity
    native_members: tuple[tuple[str, Mapping[str, object]], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "staging_receipt": self.staging_receipt.as_dict(),
            "canonical_r236_contract": self.canonical_contract.as_dict(),
            "native_members": {
                platform: dict(member) for platform, member in self.native_members
            },
        }


@dataclass(frozen=True)
class RuntimeAudit:
    runtime_root: Path
    official_cg_root: Path
    official_cg_staging: OfficialCgStaging
    official_cg_preflight: FileIdentity
    direct_runtime_files: tuple[tuple[str, FileIdentity], ...]
    official_cg_files: tuple[tuple[str, FileIdentity], ...]
    culled_runtime_cg_members: tuple[str, ...]
    culled_official_cg_members: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "runtime_root": str(self.runtime_root),
            "official_cg_root": str(self.official_cg_root),
            "official_cg_staging": self.official_cg_staging.as_dict(),
            "official_cg_local_preflight": self.official_cg_preflight.as_dict(),
            "direct_runtime_files": [
                {"member": name, **identity.as_dict()}
                for name, identity in self.direct_runtime_files
            ],
            "official_r236_cg_files": [
                {"member": f"cg/{name}", **identity.as_dict()}
                for name, identity in self.official_cg_files
            ],
            "culled_runtime_cg_members": list(self.culled_runtime_cg_members),
            "culled_official_cg_members": list(self.culled_official_cg_members),
            "inherited_cg_copied": False,
        }


@dataclass(frozen=True)
class PackageArtifact:
    archive: FileIdentity
    archive_audit: Mapping[str, object]
    deck: FileIdentity
    matchup_runtime: MatchupRuntimeEvidence
    model_runtime: ModelRuntimeEvidence
    runtime_audit: RuntimeAudit

    def as_dict(self) -> dict[str, object]:
        return {
            "archive": self.archive.as_dict(),
            "deck": self.deck.as_dict(),
            "matchup_runtime": self.matchup_runtime.as_dict(),
            "model_runtime": self.model_runtime.as_dict(),
            "direct_policy_asset_audit": dict(self.archive_audit),
            "runtime_source_audit": self.runtime_audit.as_dict(),
        }


def canonical_json(payload: object) -> bytes:
    """Encode one immutable receipt in a deterministic, finite JSON form."""

    try:
        return (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R241FinalizerError("receipt payload is not canonical JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _validate_checkpoint_receipt_fingerprint(
    payload: Mapping[str, Any], *, label: str
) -> None:
    """Validate the compact v2 fingerprint encoding used outside this script."""

    claimed = _valid_digest(
        payload.get("receipt_fingerprint_sha256"), label=f"{label} fingerprint"
    )
    bare = dict(payload)
    bare.pop("receipt_fingerprint_sha256", None)
    actual = checkpoint_receipts.sha256_bytes(checkpoint_receipts.canonical_json(bare))
    if claimed != actual:
        raise R241FinalizerError(f"{label} v2 fingerprint drifted")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _valid_digest(value: object, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256.fullmatch(digest):
        raise R241FinalizerError(f"{label} is not a SHA-256 digest")
    return digest


def _regular_file(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise R241FinalizerError(f"{label} must be a regular, non-symlink file")
    return candidate.resolve()


def _directory(path: Path | str, *, label: str, create: bool = False) -> Path:
    candidate = Path(path).expanduser()
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    if candidate.is_symlink() or not candidate.is_dir():
        raise R241FinalizerError(f"{label} must be a real directory")
    return candidate.resolve()


def _read_object(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label=label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241FinalizerError(f"{label} is unreadable JSON") from exc
    if not isinstance(value, dict):
        raise R241FinalizerError(f"{label} must be a JSON object")
    return source, value


def _file_identity(path: Path | str, *, label: str) -> FileIdentity:
    source = _regular_file(path, label=label)
    return FileIdentity(
        path=source,
        sha256=sha256_file(source),
        size_bytes=int(source.stat().st_size),
    )


def _checkpoint_identity(
    row: object,
    *,
    label: str,
    expected_path: Path | None = None,
    path_key: str = "path",
    digest_key: str = "digest",
) -> FileIdentity:
    if not isinstance(row, Mapping):
        raise R241FinalizerError(f"{label} is not a checkpoint identity")
    raw_path = str(row.get(path_key) or "").strip()
    expected_digest = _valid_digest(row.get(digest_key), label=f"{label} digest")
    candidate = _regular_file(raw_path, label=f"{label} checkpoint")
    if expected_path is not None and candidate != expected_path.resolve():
        raise R241FinalizerError(f"{label} checkpoint path is not the terminal path")
    actual = _file_identity(candidate, label=f"{label} checkpoint")
    if actual.sha256 != expected_digest:
        raise R241FinalizerError(f"{label} checkpoint bytes do not match its receipt")
    return actual


def _safe_member_name(name: str, *, label: str) -> str:
    result = str(name).removeprefix("./").strip("/")
    candidate = PurePosixPath(result)
    if (
        not result
        or "\\" in result
        or candidate.is_absolute()
        or "." in candidate.parts
        or ".." in candidate.parts
    ):
        raise R241FinalizerError(f"{label} is not a safe archive member path")
    return result


def _relative_path(root: Path, path: Path, *, label: str) -> str:
    try:
        return _safe_member_name(path.resolve().relative_to(root).as_posix(), label=label)
    except ValueError as exc:
        raise R241FinalizerError(f"{label} escapes its declared root") from exc


def _assert_exact_contract(contract: Mapping[str, Any]) -> None:
    """Validate only the r241 fields that authorize packaging, fail closed."""

    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or int(contract.get("owner_decision_revision", -1)) != OWNER_REVISION
        or int(contract.get("latest_owner_clarification_revision", -1))
        != LATEST_OWNER_CLARIFICATION_REVISION
        or contract.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
    ):
        raise R241FinalizerError("r241 typed contract identity is invalid")
    parent = dict(contract.get("parent") or {})
    if (
        parent.get("immutable") is not True
        or parent.get("typed_source")
        != "state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json"
        or parent.get("typed_source_sha256") != PARENT_R195_TYPED_SOURCE_SHA256
        or parent.get("checkpoint_sha256") != PARENT_R195_SHA256
        or int(parent.get("checkpoint_bytes", -1)) != 127_914_385
    ):
        raise R241FinalizerError("r241 is not bound to the immutable r195 parent")
    parent_source, parent_payload = _read_object(
        ROOT / "state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json",
        label="immutable r195 parent typed source",
    )
    if sha256_file(parent_source) != PARENT_R195_TYPED_SOURCE_SHA256:
        raise R241FinalizerError("immutable r195 parent typed source changed")
    parent_bootstrap = dict(parent_payload.get("expert_bootstrap") or {})
    parent_submission = dict(parent_payload.get("submission") or {})
    parent_completion = dict(parent_payload.get("completion") or {})
    if (
        parent_payload.get("schema")
        != "poke_bot.alakazam_terminal_expert_bootstrap_no_rtp_submit_r195/v1"
        or int(parent_payload.get("owner_decision_revision", -1)) != 195
        or parent_bootstrap.get("full_model") is not True
        or parent_bootstrap.get("trainable_heads")
        != "every_architecture_present_non_combo_learner_head"
        or parent_bootstrap.get("combo_state_loss_weight") != 0.0
        or parent_bootstrap.get("combo_state_route_enabled") is not False
        or parent_bootstrap.get("matchup_adapter_fit")
        != "preserve_existing_trained_bank_under_checksum_bound_alakazam_contract"
        or parent_submission.get("matchup_adapter_runtime_required_if_bank_present")
        is not True
        or parent_completion.get("expert_checkpoint_sha256") != PARENT_R195_SHA256
    ):
        raise R241FinalizerError("immutable r195 parent lacks the required preserved runtime")

    cycle = dict(contract.get("training_cycle") or {})
    if (
        int(cycle.get("rl_updates_exact", -1)) != 10
        or list(cycle.get("zero_indexed_iteration_commits") or []) != list(range(10))
        or int(cycle.get("next_iteration_after_loop", -1)) != NEXT_ITERATION
        or cycle.get("iteration_10_collection_allowed") is not False
        or int(cycle.get("games_per_update", -1)) != 8196
        or int(cycle.get("self_play_games_exact", -1)) != 1024
        or int(cycle.get("public_mix_games_exact", -1)) != 7172
        or int(cycle.get("marnie_h10_games_minimum", -1)) != 1024
        or dict(cycle.get("training_seats") or {})
        != {"first": 4098, "second": 4098, "exact_split_required": True}
    ):
        raise R241FinalizerError("r241 fixed ten-update contract is invalid")

    refresh = dict(contract.get("expert_soft_refresh") or {})
    if (
        list(refresh.get("update_boundaries") or []) != [5, 10]
        or int(refresh.get("epochs_each_boundary", -1)) != 5
        or refresh.get("terminal_checkpoint_name") != TERMINAL_CHECKPOINT_NAME
        or refresh.get("terminal_refresh_must_not_collect_an_eleventh_wave") is not True
    ):
        raise R241FinalizerError("r241 terminal five-epoch refresh contract is invalid")
    exact_window_binding = dict(refresh.get("exact_window_evidence_binding") or {})
    if (
        exact_window_binding.get("expected_latest_calendar_date") != "2026-08-10"
        or exact_window_binding.get("staging_receipt")
        != "state/alakazam-new-list-direct-r241-expert-window-staging.json"
        or exact_window_binding.get("canonical_manifest_sha256")
        != EXPERT_WINDOW_CANONICAL_RECEIPT_SHA256
        or exact_window_binding.get("evidence_role")
        != "immutable_exact_window_identity_only; readiness_is_derived_externally"
    ):
        raise R241FinalizerError("r241 exact latest expert window is not contract-bound")
    preservation = dict(contract.get("peak_r195_behavior_preservation") or {})
    if (
        preservation.get("learner_public_matchup_tree_sha256")
        != LEARNER_R195_MATCHUP_TREE_SHA256
        or preservation.get("marnie_public_matchup_tree_sha256")
        != H10_DIRECT_MATCHUP_TREE_SHA256
    ):
        raise R241FinalizerError("r241 learner/H10 Matchup Adapter tree binding is invalid")

    exclusion = dict(contract.get("search_and_planning_exclusion") or {})
    scope = dict(exclusion.get("scope") or {})
    if (
        exclusion.get("mcts") != "forbidden_for_scoped_direct_roles"
        or exclusion.get("recursive_turn_planner") != "forbidden_for_scoped_direct_roles"
        or exclusion.get("search_target_generation") != "forbidden_for_scoped_direct_roles"
        or exclusion.get("training_collector_action_selector") != "direct_policy_only"
        or exclusion.get("marnie_action_selector") != "direct_policy_only"
        or exclusion.get("submission_action_selector") != "direct_policy_only"
        or scope
        != {
            "learner": "direct_policy_only",
            "pinned_h10_marnie_opponent": "direct_policy_only",
            "target_generation": "direct_policy_only",
            "terminal_package_and_submission": "direct_policy_only",
            "frozen_non_h10_diverse_public_opponent_packages_and_selectors": "preserve_unchanged_per_r245",
        }
        or exclusion.get("public_opponent_selector_change") != "forbidden"
        or exclusion.get("public_search_firewall") != "not_introduced"
        or exclusion.get("search_config_or_belief_deck_assets_required") is not False
    ):
        raise R241FinalizerError("r241 direct-policy exclusion contract is invalid")

    simulator = dict(contract.get("canonical_simulator") or {})
    if (
        simulator.get("kaggle_environments_version") != "1.32.6"
        or simulator.get("binding_environment") != "CG_LIB_PATH"
        or simulator.get("linux_x86_64_member") != "cg/libcg.so"
        or not _SHA256.fullmatch(str(simulator.get("linux_x86_64_sha256") or ""))
        or int(simulator.get("linux_x86_64_size_bytes", -1)) <= 0
        or set(simulator.get("forbidden_environment") or [])
        != {"POKEBOT_LIBCG_PATH", "POKEBOT_BATCH_LIBCG"}
    ):
        raise R241FinalizerError("r241 official r236 libcg binding is invalid")

    submission = dict(contract.get("submission") or {})
    try:
        package_limit = Decimal(str(submission.get("package_size_limit_mib")))
    except (InvalidOperation, ValueError) as exc:
        raise R241FinalizerError("r241 package-size limit is invalid") from exc
    if (
        int(submission.get("exact_count", -1)) != 1
        or submission.get("competition") != "pokemon-tcg-ai-battle"
        or submission.get("turn_order_preference") != "first_if_allowed"
        or submission.get("checkpoint_source") != TERMINAL_CHECKPOINT_NAME
        or submission.get("intermediate_iteration_5_submission_allowed") is not False
        or submission.get("retry_copy_or_duplicate_allowed") is not False
        or submission.get("queue_policy")
        != "one_single_use_direct_policy_slot_after_terminal_finalizer"
        or package_limit != ARCHIVE_LIMIT_MIB
        or submission.get("direct_policy_only_bundle_and_queue_schema_required") is not True
        or submission.get("do_not_claim_submitted_without_upload_receipt") is not True
    ):
        raise R241FinalizerError("r241 single-submission contract is invalid")
    authority = dict(contract.get("authority") or {})
    if (
        authority
        != {
            "offline_implementation_staging_validation_and_managed_download": (
                "authorized_by_owner_intent"
            ),
            "managed_training_start": "requires_external_activation_overlay",
            "mcts_or_rtp_work": "not_authorized_by_this_contract",
            "selector_or_production_activation": (
                "requires_external_activation_overlay_and_terminal_finalizer"
            ),
            "kaggle_submission": (
                "requires_terminal_finalizer_and_one_single_use_queue_authorization"
            ),
            "terminal_submission_cardinality": "exactly_one",
        }
    ):
        raise R241FinalizerError("r241 terminal submission authority is invalid")


def _validate_exact_deck(contract: Mapping[str, Any]) -> FileIdentity:
    deck = dict(contract.get("exact_deck") or {})
    deck_path = _regular_file(ROOT / str(deck.get("path") or ""), label="r241 deck")
    identity = _file_identity(deck_path, label="r241 deck")
    if identity.sha256 != _valid_digest(deck.get("file_sha256"), label="deck digest"):
        raise R241FinalizerError("r241 exact deck bytes changed")
    try:
        cards = [
            int(line.split(",", 1)[0])
            for raw in deck_path.read_text(encoding="utf-8").splitlines()
            if (line := raw.strip()) and not line.startswith("#")
        ]
    except (UnicodeDecodeError, ValueError) as exc:
        raise R241FinalizerError("r241 exact deck is unreadable") from exc
    if len(cards) != 60 or int(deck.get("card_count", -1)) != 60:
        raise R241FinalizerError("r241 exact deck does not contain 60 cards")
    ordered = sha256_bytes(json.dumps(cards, separators=(",", ":")).encode("utf-8"))
    multiset = sha256_bytes(
        json.dumps(sorted(cards), separators=(",", ":")).encode("utf-8")
    )
    if (
        ordered != _valid_digest(deck.get("ordered_cards_sha256"), label="deck order")
        or multiset
        != _valid_digest(deck.get("canonical_multiset_sha256"), label="deck multiset")
    ):
        raise R241FinalizerError("r241 exact deck order or multiset changed")
    return identity


def _load_expert_window_evidence(contract: Mapping[str, Any]) -> ExpertWindowEvidence:
    """Bind terminal refreshes to the ready exact July 22--August 10 corpus.

    The finalizer never downloads or trains from this corpus.  It only checks
    the local staging record that proves the latest Kaggle day and the exact
    20-day manifest were available before a receipt may authorize packaging.
    """

    source, staging = _read_object(
        EXPERT_WINDOW_STAGING_PATH, label="r241 exact expert-window staging"
    )
    identity = _file_identity(source, label="r241 exact expert-window staging")
    if identity.sha256 != EXPERT_WINDOW_STAGING_SHA256:
        raise R241FinalizerError("recorded r241 exact expert-window staging changed")
    window = dict(staging.get("window") or {})
    canonical = dict(staging.get("canonical_receipt") or {})
    immutable = dict(staging.get("immutable_window_receipt") or {})
    scope = dict(staging.get("scope") or {})
    if (
        staging.get("schema") != EXPERT_WINDOW_STAGING_SCHEMA
        or int(staging.get("owner_decision_revision", -1)) != OWNER_REVISION
        or int(staging.get("latest_owner_clarification_revision", -1))
        != EXPERT_WINDOW_STAGING_OWNER_CLARIFICATION_REVISION
        or staging.get("status") != "ready"
        or window.get("policy") != "exact_20_consecutive_calendar_days"
        or window.get("start") != "2026-07-22"
        or window.get("end") != "2026-08-10"
        or window.get("days") != 20
        or int(window.get("validated_episodes", -1)) != 91_253
        or canonical.get("schema") != "poke_bot.expert_latest20_receipt/v1"
        or canonical.get("status") != "ready"
        or canonical.get("sha256") != EXPERT_WINDOW_CANONICAL_RECEIPT_SHA256
        or not str(canonical.get("elmo_path") or "").strip()
        or not str(canonical.get("inzi_path") or "").strip()
        or not _SHA256.fullmatch(str(immutable.get("sha256") or ""))
        or scope.get("expert_source_gate_ready") is not True
        or scope.get("training_or_gradient_updates_started") is not False
        or scope.get("managed_training_service_started") is not False
        or scope.get("submission_authority") is not False
    ):
        raise R241FinalizerError("r241 exact latest expert-window staging is invalid")
    contract_refresh = dict(contract.get("expert_soft_refresh") or {})
    exact_window_binding = dict(
        contract_refresh.get("exact_window_evidence_binding") or {}
    )
    if (
        exact_window_binding.get("staging_receipt")
        != "state/alakazam-new-list-direct-r241-expert-window-staging.json"
        or exact_window_binding.get("canonical_manifest_sha256")
        != canonical.get("sha256")
    ):
        raise R241FinalizerError("r241 contract no longer binds the staged expert window")
    return ExpertWindowEvidence(
        staging_receipt=identity,
        canonical_receipt_sha256=str(canonical["sha256"]),
        immutable_window_receipt_sha256=str(immutable["sha256"]),
        validated_episodes=int(window["validated_episodes"]),
    )


def _validate_expert_window_binding(
    value: object, *, evidence: ExpertWindowEvidence, label: str
) -> None:
    binding = dict(value or {})
    _receipt_identity_matches(
        binding.get("staging_receipt"),
        identity=evidence.staging_receipt,
        label=f"{label} expert-window staging",
    )
    window = dict(binding.get("window") or {})
    if (
        binding.get("canonical_receipt_sha256")
        != evidence.canonical_receipt_sha256
        or binding.get("immutable_window_receipt_sha256")
        != evidence.immutable_window_receipt_sha256
        or window.get("start") != "2026-07-22"
        or window.get("end") != "2026-08-10"
        or window.get("days") != 20
        or int(window.get("validated_episodes", -1)) != evidence.validated_episodes
    ):
        raise R241FinalizerError(f"{label} is not bound to the exact expert window")


def _load_peak_r195_head_inventory() -> PeakR195HeadInventory:
    """Read the immutable r241 map of every non-combo r195 head/route."""

    source, payload = _read_object(
        PEAK_R195_HEAD_ROLE_MAP_PATH, label="r241 peak-r195 head-role map"
    )
    identity = _file_identity(source, label="r241 peak-r195 head-role map")
    if identity.sha256 != PEAK_R195_HEAD_ROLE_MAP_SHA256:
        raise R241FinalizerError("r241 peak-r195 head-role map changed")
    names = _strict_name_tuple(
        payload.get("canonical_learned_decision_sources"),
        label="r241 canonical non-combo head inventory",
    )
    heads = dict(payload.get("heads") or {})
    if (
        payload.get("schema") != "poke_bot.future_specialist_strategic_head_roles/v1"
        or payload.get("specialist_id") != "alakazam"
        or payload.get("decision_fusion_schema") != "poke_bot.causal_decision_fusion/v3"
        or set(heads) != set(names)
        or payload.get("one_route_per_learned_source") is not True
        or payload.get("positive_bounded_route_reliability") is not True
    ):
        raise R241FinalizerError("r241 peak-r195 head-role map is incomplete")
    routes: list[str] = []
    for name in names:
        head = dict(heads[name] or {})
        route_id = str(head.get("route_id") or "").strip()
        if (
            head.get("trainable") is not True
            or head.get("enters_decision_fusion") is not True
            or head.get("fusion_role") != "fused_input"
            or head.get("runtime_activation_requirement") != "receipt_backed_validation"
            or not route_id
        ):
            raise R241FinalizerError(
                f"r241 peak-r195 head-role map does not preserve {name}"
            )
        routes.append(route_id)
    if len(set(routes)) != len(routes):
        raise R241FinalizerError("r241 peak-r195 fusion routes are not unique")
    return PeakR195HeadInventory(
        source=identity,
        head_names=names,
        fusion_route_ids=tuple(sorted(routes)),
    )


def _reject_iteration_10_collection(run_dir: Path) -> None:
    """A terminal refresh may exist; a new iteration-10 collection may not."""

    forbidden_roots = (
        "collection_plans",
        "collection_receipts",
        "commits",
        "eval",
        "metrics",
        "shards",
    )
    found: list[str] = []
    for name in forbidden_roots:
        root = run_dir / name
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise R241FinalizerError(f"{name} is not a real terminal artifact directory")
        found.extend(str(path) for path in root.glob("iter_00010*") if path.exists())
    raw_checkpoint = run_dir / "checkpoints" / "iter_00010.pt"
    if raw_checkpoint.exists() or raw_checkpoint.is_symlink():
        found.append(str(raw_checkpoint))
    if found:
        raise R241FinalizerError(
            "iteration 10 collection/training artifacts are forbidden: " + ", ".join(found)
        )


def _validate_commit_chain(
    run_dir: Path, loop_state: Mapping[str, Any]
) -> tuple[FileIdentity, FileIdentity]:
    commits_root = _directory(run_dir / "commits", label="r241 commits")
    expected_names = {f"iter_{iteration:05d}.json" for iteration in range(10)}
    actual_names = {
        child.name
        for child in commits_root.iterdir()
        if child.is_file() or child.is_symlink()
    }
    if actual_names != expected_names:
        raise R241FinalizerError(
            "r241 must have exactly durable commits iter_00000 through iter_00009"
        )

    terminal_commit: FileIdentity | None = None
    terminal_parent: FileIdentity | None = None
    terminal_history: object = None
    for iteration in range(10):
        path, commit = _read_object(
            commits_root / f"iter_{iteration:05d}.json",
            label=f"r241 commit {iteration}",
        )
        if (
            int(commit.get("last_completed_iteration", -1)) != iteration
            or int(commit.get("next_iteration", -1)) != iteration + 1
        ):
            raise R241FinalizerError(f"r241 commit {iteration} is not a durable boundary")
        rows = [
            row
            for row in list(commit.get("history") or [])
            if isinstance(row, Mapping) and int(row.get("iteration", -1)) == iteration
        ]
        if len(rows) != 1 or rows[0].get("completed") is not True:
            raise R241FinalizerError(f"r241 commit {iteration} has no unique completed row")
        if iteration == TERMINAL_ITERATION:
            terminal_commit = _file_identity(path, label="r241 terminal commit")
            terminal_parent = _checkpoint_identity(
                commit.get("learner"),
                label="r241 iter_00009 learner",
                expected_path=(
                    run_dir / "checkpoints" / f"iter_{TERMINAL_ITERATION:05d}.pt"
                ),
            )
            terminal_history = commit.get("history")

    if terminal_commit is None or terminal_parent is None:
        raise R241FinalizerError("r241 terminal commit is absent")
    if loop_state.get("history") != terminal_history:
        raise R241FinalizerError("loop ledger history diverged from durable iter_00009")
    loop_parent = _checkpoint_identity(
        loop_state.get("learner"), label="r241 loop terminal learner"
    )
    if loop_parent != terminal_parent:
        raise R241FinalizerError("loop learner diverged from durable iter_00009 learner")
    return terminal_commit, terminal_parent


def _validate_terminal_refresh(
    *,
    run_dir: Path,
    loop_state: Mapping[str, Any],
    terminal_parent: FileIdentity,
    terminal_refresh_path: Path,
    expert_window: ExpertWindowEvidence,
) -> tuple[FileIdentity, FileIdentity, FileIdentity]:
    refresh_path, refresh = _read_object(
        terminal_refresh_path, label="r241 terminal expert refresh"
    )
    if (
        refresh.get("schema") != TERMINAL_REFRESH_SCHEMA
        or int(refresh.get("before_iteration", -1)) != NEXT_ITERATION
        or int(refresh.get("rl_updates_completed", -1)) != NEXT_ITERATION
        or int(refresh.get("epochs_completed", -1)) != 5
        or refresh.get("next_collection_started") is not False
    ):
        raise R241FinalizerError("terminal refresh is not the exact 5-epoch iter_00010 boundary")

    wrapped_parent = _checkpoint_identity(
        refresh.get("parent"), label="terminal refresh parent"
    )
    if wrapped_parent != terminal_parent:
        raise R241FinalizerError("terminal refresh is not bound to durable iter_00009")
    terminal_checkpoint_path = _regular_file(
        run_dir / "checkpoints" / TERMINAL_CHECKPOINT_NAME,
        label="terminal expert checkpoint",
    )
    refreshed = _checkpoint_identity(
        refresh.get("refreshed"),
        label="terminal refresh output",
        expected_path=terminal_checkpoint_path,
    )

    rehearsal_path = _regular_file(
        run_dir / "rehearsals" / "before_iter_00010.json",
        label="terminal five-epoch rehearsal receipt",
    )
    _path, rehearsal = _read_object(
        rehearsal_path, label="terminal five-epoch rehearsal receipt"
    )
    receipt_checkpoint = _checkpoint_identity(
        {
            "path": rehearsal.get("checkpoint"),
            "digest": rehearsal.get("checkpoint_digest"),
        },
        label="terminal rehearsal output",
        expected_path=terminal_checkpoint_path,
    )
    if (
        int(rehearsal.get("before_iteration", -1)) != NEXT_ITERATION
        or int(rehearsal.get("epochs", -1)) != 5
        or _valid_digest(rehearsal.get("parent_digest"), label="terminal rehearsal parent")
        != terminal_parent.sha256
        or receipt_checkpoint != refreshed
    ):
        raise R241FinalizerError("terminal rehearsal receipt is not the required five-epoch output")
    _validate_expert_window_binding(
        rehearsal.get("expert_window"),
        evidence=expert_window,
        label="terminal rehearsal",
    )

    embedded = dict(refresh.get("expert_rehearsal") or {})
    if (
        int(embedded.get("before_iteration", -1)) != NEXT_ITERATION
        or int(embedded.get("epochs", -1)) != 5
        or _valid_digest(embedded.get("parent_digest"), label="embedded terminal parent")
        != terminal_parent.sha256
        or str(embedded.get("checkpoint_digest") or "") != refreshed.sha256
    ):
        raise R241FinalizerError("terminal refresh lacks its bound rehearsal receipt")
    _validate_expert_window_binding(
        refresh.get("expert_window"),
        evidence=expert_window,
        label="terminal expert refresh",
    )

    state_refresh = dict(loop_state.get("terminal_expert_refresh") or {})
    if (
        _regular_file(
            str(state_refresh.get("path") or ""), label="loop terminal refresh path"
        )
        != refresh_path
        or int(state_refresh.get("epochs", -1)) != 5
        or _checkpoint_identity(state_refresh.get("parent"), label="loop terminal parent")
        != terminal_parent
        or _checkpoint_identity(
            state_refresh.get("refreshed"),
            label="loop terminal refreshed checkpoint",
            expected_path=terminal_checkpoint_path,
        )
        != refreshed
    ):
        raise R241FinalizerError("loop state does not bind the terminal expert refresh")
    return (
        _file_identity(refresh_path, label="terminal refresh"),
        _file_identity(rehearsal_path, label="terminal rehearsal receipt"),
        refreshed,
    )


def validate_terminal_evidence(
    *,
    run_dir: Path,
    contract_path: Path = CONTRACT_PATH,
    terminal_refresh_path: Path | None = None,
) -> tuple[dict[str, Any], TerminalEvidence]:
    """Return only receipt-bound evidence for the completed r241 terminal state."""

    contract_file, contract = _read_object(contract_path, label="r241 typed contract")
    if (
        contract_file == CONTRACT_PATH.resolve()
        and sha256_file(contract_file) != CONTRACT_SHA256
    ):
        raise R241FinalizerError("canonical r241 typed contract changed without finalizer revision")
    _assert_exact_contract(contract)
    expert_window = _load_expert_window_evidence(contract)
    run = _directory(run_dir, label="r241 run directory")
    loop_path, loop_state = _read_object(run / "loop_state.json", label="r241 loop state")
    del loop_path  # The state is rebound through the immutable terminal commit below.
    if (
        int(loop_state.get("last_completed_iteration", -1)) != TERMINAL_ITERATION
        or int(loop_state.get("next_iteration", -1)) != NEXT_ITERATION
    ):
        raise R241FinalizerError("r241 loop is not exactly at the terminal iter_00009 boundary")
    _reject_iteration_10_collection(run)
    terminal_commit, terminal_parent = _validate_commit_chain(run, loop_state)
    refresh_target = (
        Path(terminal_refresh_path).expanduser()
        if terminal_refresh_path is not None
        else run / "terminal_expert_refresh.json"
    )
    refresh, rehearsal, terminal_checkpoint = _validate_terminal_refresh(
        run_dir=run,
        loop_state=loop_state,
        terminal_parent=terminal_parent,
        terminal_refresh_path=refresh_target,
        expert_window=expert_window,
    )
    evidence = TerminalEvidence(
        run_dir=run,
        contract_path=contract_file,
        contract_sha256=sha256_file(contract_file),
        terminal_commit=terminal_commit,
        terminal_parent=terminal_parent,
        terminal_refresh=refresh,
        terminal_rehearsal=rehearsal,
        terminal_checkpoint=terminal_checkpoint,
        expert_window=expert_window,
    )
    return contract, evidence


def _receipt_identity_matches(
    row: object,
    *,
    identity: FileIdentity,
    label: str,
) -> None:
    if not isinstance(row, Mapping):
        raise R241FinalizerError(f"{label} is not a receipt identity")
    declared_digest = _valid_digest(
        row.get("sha256") or row.get("digest"), label=f"{label} digest"
    )
    if declared_digest != identity.sha256:
        raise R241FinalizerError(f"{label} digest does not match its file")
    declared_path = str(row.get("path") or "").strip()
    if declared_path and _regular_file(declared_path, label=f"{label} path") != identity.path:
        raise R241FinalizerError(f"{label} path does not match its file")
    declared_size = row.get("size_bytes")
    if declared_size is not None and int(declared_size) != identity.size_bytes:
        raise R241FinalizerError(f"{label} size does not match its file")


def _checkpoint_audit_environment(official_cg_root: Path) -> dict[str, str]:
    """Use a clean direct-only environment for receipt audit replay."""

    return {
        "CG_LIB_PATH": str(official_cg_root),
        "POKEBOT_R241_DIRECT_POLICY_ONLY": "1",
        "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
        "POKEBOT_SEARCH_MODE": "policy",
        "POKEBOT_SUBMISSION_SEARCH_DISABLE": "1",
        "POKEBOT_COMBO_STATE_ROUTE_ENABLED": "0",
        "POKEBOT_MATCHUP_ADAPTER_RUNTIME": "1",
    }


def _validate_generated_terminal_checkpoint_receipts(
    *,
    model_runtime_activation_path: Path,
    matchup_runtime_activation_path: Path,
    matchup_tree_path: Path,
    terminal: TerminalEvidence,
    contract: Mapping[str, Any],
    official_cg_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay both v2 receipts against checkpoint/model/tree bytes.

    This is intentionally separate from the lightweight package metadata
    checks below.  It is the no-self-assertion boundary: copied JSON is not
    sufficient because the audit module rehashes and reconstructs the actual
    r195/terminal tensors under the sealed r236 direct-policy environment.
    """

    _matchup_path, matchup = _read_object(
        matchup_runtime_activation_path,
        label="r241 matchup runtime activation",
    )
    h10 = dict(dict(matchup.get("h10_training_opponent") or {}).get("matchup_tree") or {})
    h10_path = _regular_file(
        str(h10.get("path") or ""), label="r241 H10 matchup tree"
    )
    parent = dict(contract.get("parent") or {})
    parent_path = _regular_file(
        str(parent.get("checkpoint") or ""), label="r195 immutable parent checkpoint"
    )
    try:
        return checkpoint_receipts.validate_terminal_runtime_receipts(
            model_receipt_path=model_runtime_activation_path,
            matchup_receipt_path=matchup_runtime_activation_path,
            r195_parent_checkpoint=parent_path,
            terminal_parent_checkpoint=terminal.terminal_parent.path,
            terminal_checkpoint=terminal.terminal_checkpoint.path,
            terminal_refresh_receipt=terminal.terminal_refresh.path,
            terminal_rehearsal_receipt=terminal.terminal_rehearsal.path,
            learner_matchup_tree=matchup_tree_path,
            h10_matchup_tree=h10_path,
            official_cg_root=official_cg_root,
            environment=_checkpoint_audit_environment(official_cg_root),
        )
    except checkpoint_receipts.R241CheckpointReceiptError as exc:
        raise R241FinalizerError(
            f"r241 terminal checkpoint-derived runtime receipt failed: {exc}"
        ) from exc


def validate_matchup_runtime_evidence(
    *,
    matchup_tree_path: Path,
    runtime_activation_path: Path,
    terminal: TerminalEvidence,
) -> MatchupRuntimeEvidence:
    """Require the trained direct-policy adapter runtime before packaging.

    Direct policy excludes MCTS/RTP, not the r195 trained matchup-adapter
    bank.  The tree is a separate, checksum-bound policy routing asset and the
    activation receipt proves it was active against this terminal checkpoint.
    """

    tree_path, tree = _read_object(matchup_tree_path, label="r241 matchup tree")
    tree_identity = _file_identity(tree_path, label="r241 matchup tree")
    if (
        tree_identity.sha256 != LEARNER_R195_MATCHUP_TREE_SHA256
        or tree_identity.size_bytes <= 0
    ):
        raise R241FinalizerError("r241 matchup tree is not the pinned r195 learner tree")
    runtime = dict(tree.get("runtime_contract") or {})
    accepted = tuple(sorted(str(value) for value in runtime.get("accepted_archetype_ids") or []))
    if (
        tree.get("runtime_enabled") is not True
        or not accepted
        or runtime.get("one_route_per_decision") is not True
        or runtime.get("unknown_route_exact_bypass") is not True
    ):
        raise R241FinalizerError("r241 matchup tree is not an active direct adapter route")

    activation_path, activation = _read_object(
        runtime_activation_path, label="r241 matchup runtime activation"
    )
    activation_identity = _file_identity(
        activation_path, label="r241 matchup runtime activation"
    )
    runtime_smoke = dict(activation.get("runtime_smoke") or {})
    h10_training_opponent = dict(activation.get("h10_training_opponent") or {})
    if (
        activation.get("schema") != MATCHUP_RUNTIME_ACTIVATION_SCHEMA
        or int(activation.get("owner_decision_revision", -1)) != OWNER_REVISION
        or activation.get("owner_clarification_revision")
        != LATEST_OWNER_CLARIFICATION_REVISION
        or activation.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or activation.get("status") != "active_direct_policy_only"
        or activation.get("derived_not_self_asserted") is not True
        or dict(activation.get("parent_r195_checkpoint") or {}).get("sha256")
        != PARENT_R195_SHA256
        or activation.get("action_selector") != "direct_policy_only"
        or activation.get("mcts_enabled") is not False
        or activation.get("recursive_turn_planner_enabled") is not False
        or activation.get("search_enabled") is not False
        or activation.get("belief_assets_enabled") is not False
        or runtime_smoke.get("model_reconstructed") is not True
        or runtime_smoke.get("adapter_runtime_enabled_for_smoke") is not True
        or runtime_smoke.get("adapter_output_changed") is not True
        or runtime_smoke.get("action_selector") != "direct_policy_only"
        or runtime_smoke.get("mcts_calls") != 0
        or runtime_smoke.get("rtp_calls") != 0
        or runtime_smoke.get("search_calls") != 0
        or h10_training_opponent.get("direct_policy_only") is not True
        or h10_training_opponent.get("mcts_enabled") is not False
        or h10_training_opponent.get("recursive_turn_planner_enabled") is not False
        or h10_training_opponent.get("search_enabled") is not False
    ):
        raise R241FinalizerError("r241 matchup runtime activation is not direct-policy active")
    _receipt_identity_matches(
        activation.get("terminal_checkpoint"),
        identity=terminal.terminal_checkpoint,
        label="r241 matchup runtime terminal checkpoint",
    )
    _receipt_identity_matches(
        activation.get("learner_matchup_tree"),
        identity=tree_identity,
        label="r241 matchup runtime tree",
    )
    _receipt_identity_matches(
        h10_training_opponent.get("matchup_tree"),
        identity=_file_identity(
            _regular_file(
                str(dict(h10_training_opponent.get("matchup_tree") or {}).get("path") or ""),
                label="r241 H10 matchup runtime tree",
            ),
            label="r241 H10 matchup runtime tree",
        ),
        label="r241 matchup runtime H10 tree",
    )
    return MatchupRuntimeEvidence(
        tree=tree_identity,
        activation=activation_identity,
        accepted_archetype_ids=accepted,
    )


def _strict_name_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise R241FinalizerError(f"{label} must be a non-empty list of head names")
    names = tuple(str(item).strip() for item in value)
    if not names or any(not item for item in names) or len(set(names)) != len(names):
        raise R241FinalizerError(f"{label} must contain unique non-empty head names")
    return tuple(sorted(names))


def validate_model_runtime_evidence(
    *,
    runtime_activation_path: Path,
    terminal: TerminalEvidence,
    contract: Mapping[str, Any],
) -> ModelRuntimeEvidence:
    """Require the terminal model's direct r195-derived serving inventory.

    The model checkpoint itself is opaque binary state.  This receipt is the
    fail-closed boundary that proves the terminal model retained every
    architecture-present non-combo head and fusion route from r195 while only
    leaving the explicitly disabled combo-state loss/route off.  It is also
    where the trained matchup-adapter runtime is asserted active.
    """

    activation_path, activation = _read_object(
        runtime_activation_path, label="r241 terminal model runtime activation"
    )
    identity = _file_identity(
        activation_path, label="r241 terminal model runtime activation"
    )
    inventory = _load_peak_r195_head_inventory()
    expected_routes = inventory.fusion_route_ids
    preservation = dict(contract.get("peak_r195_behavior_preservation") or {})
    heads = dict(activation.get("heads") or {})
    combo = dict(heads.get("combo_state") or {})
    active_heads = _strict_name_tuple(
        heads.get("active_non_combo_head_names"),
        label="terminal model active non-combo head inventory",
    )
    active_routes = _strict_name_tuple(
        heads.get("active_non_combo_fusion_route_ids"),
        label="terminal model active non-combo fusion-route inventory",
    )
    runtime_smoke = dict(activation.get("runtime_smoke") or {})
    package_activation = dict(activation.get("runtime_package_activation") or {})
    parent = dict(activation.get("parent_r195_checkpoint") or {})
    if (
        activation.get("schema") != MODEL_RUNTIME_ACTIVATION_SCHEMA
        or int(activation.get("owner_decision_revision", -1)) != OWNER_REVISION
        or activation.get("owner_clarification_revision")
        != LATEST_OWNER_CLARIFICATION_REVISION
        or activation.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or activation.get("status") != "active_peak_r195_non_combo_fusion"
        or activation.get("derived_not_self_asserted") is not True
        or active_heads != inventory.head_names
        or active_routes != expected_routes
        or heads.get("architecture_present_head_count") != 19
        or heads.get("non_combo_head_count") != 18
        or heads.get("non_combo_route_count") != 18
        or heads.get("every_non_combo_head_trainable") is not True
        or heads.get("every_non_combo_fusion_route_enabled") is not True
        or combo.get("head_present") is not True
        or combo.get("physical_route_present") is not True
        or combo.get("loss_weight") != 0.0
        or combo.get("route_enabled") is not False
        or parent.get("sha256") != PARENT_R195_SHA256
        or package_activation.get("matchup_adapters_enabled") is not True
        or package_activation.get("checkpoint_remains_dormant") is not True
        or activation.get("action_selector") != "direct_policy_only"
        or activation.get("mcts_enabled") is not False
        or activation.get("recursive_turn_planner_enabled") is not False
        or activation.get("search_enabled") is not False
        or activation.get("belief_assets_enabled") is not False
        or runtime_smoke.get("model_reconstructed") is not True
        or runtime_smoke.get("adapter_runtime_enabled_for_smoke") is not True
        or runtime_smoke.get("adapter_output_changed") is not True
        or runtime_smoke.get("mcts_calls") != 0
        or runtime_smoke.get("rtp_calls") != 0
        or runtime_smoke.get("search_calls") != 0
        or preservation.get("every_architecture_present_non_combo_head_trainable")
        is not True
        or preservation.get("every_architecture_present_non_combo_fusion_route_enabled")
        is not True
        or preservation.get("combo_state_head_remains_present") is not True
        or preservation.get("combo_state_loss_weight") != 0.0
        or preservation.get("combo_state_fusion_route_enabled") is not False
        or preservation.get("matchup_adapter_bank_preserved") is not True
        or preservation.get("matchup_adapter_runtime_enabled") is not True
    ):
        raise R241FinalizerError("terminal model runtime inventory is not generated r195 v2 evidence")
    _receipt_identity_matches(
        activation.get("terminal_checkpoint"),
        identity=terminal.terminal_checkpoint,
        label="terminal model runtime checkpoint",
    )
    return ModelRuntimeEvidence(
        activation=identity,
        active_head_names=active_heads,
        active_fusion_route_ids=active_routes,
    )


def _canonical_r236_native_members(
    contract: Mapping[str, Any], *, require_official_d162: bool
) -> tuple[FileIdentity, tuple[tuple[str, Mapping[str, object]], ...]]:
    """Read the checksum-bound r236 source for its complete native set."""

    simulator = dict(contract.get("canonical_simulator") or {})
    source_path = _regular_file(
        ROOT / str(simulator.get("typed_source") or ""),
        label="r241 canonical r236 source",
    )
    source_identity = _file_identity(source_path, label="r241 canonical r236 source")
    if source_identity.sha256 != _valid_digest(
        simulator.get("typed_source_sha256"), label="r241 canonical r236 source digest"
    ):
        raise R241FinalizerError("r241 canonical r236 source bytes changed")
    _path, canonical = _read_object(source_path, label="r241 canonical r236 source")
    native = dict(canonical.get("canonical_native_libraries") or {})
    required_platforms = {
        "linux_x86_64",
        "linux_aarch64",
        "macos_arm64",
        "windows_x86_64",
    }
    if (
        canonical.get("schema") != "poke_bot.canonical_libcg_r236/v1"
        or set(native) != required_platforms
    ):
        raise R241FinalizerError("canonical r236 source lacks the complete native set")
    normalized: list[tuple[str, Mapping[str, object]]] = []
    for platform_name in sorted(required_platforms):
        member = dict(native[platform_name])
        package_path = _safe_member_name(
            str(member.get("package_relative_path") or ""),
            label=f"canonical r236 {platform_name} member",
        )
        if not package_path.startswith("cg/"):
            raise R241FinalizerError("canonical r236 member does not live under cg/")
        normalized.append(
            (
                platform_name,
                {
                    "package_relative_path": package_path,
                    "sha256": _valid_digest(
                        member.get("sha256"),
                        label=f"canonical r236 {platform_name} digest",
                    ),
                    "size_bytes": int(member.get("size_bytes", -1)),
                },
            )
        )
    native_map = dict(normalized)
    linux = dict(native_map["linux_x86_64"])
    if linux.get("package_relative_path") != "cg/libcg.so":
        raise R241FinalizerError("canonical r236 source lacks the Linux CG member")
    if require_official_d162 and (
        linux.get("sha256") != OFFICIAL_LINUX_LIBCG_SHA256
        or int(linux.get("size_bytes", -1)) != OFFICIAL_LINUX_LIBCG_SIZE_BYTES
    ):
        raise R241FinalizerError("canonical r236 source no longer identifies D162 libcg")
    return source_identity, tuple(normalized)


def _load_official_cg_staging(
    *,
    contract: Mapping[str, Any],
    staging_path: Path,
) -> OfficialCgStaging:
    """Bind packaging to the recorded Inzi/Elmo official-libcg staging proof."""

    source, staging = _read_object(staging_path, label="r241 official libcg staging")
    identity = _file_identity(source, label="r241 official libcg staging")
    is_default_staging = source == OFFICIAL_LIBCG_STAGING_PATH.resolve()
    if is_default_staging and identity.sha256 != OFFICIAL_LIBCG_STAGING_SHA256:
        raise R241FinalizerError("recorded r241 official-libcg staging receipt changed")
    if (
        staging.get("schema") != OFFICIAL_LIBCG_STAGING_SCHEMA
        or int(staging.get("owner_decision_revision", -1)) != OWNER_REVISION
        or staging.get("status") != "staged_and_export_attested_on_inzi_and_elmo"
    ):
        raise R241FinalizerError("r241 official-libcg staging receipt is invalid")
    canonical_identity, native_members = _canonical_r236_native_members(
        contract,
        require_official_d162=is_default_staging,
    )
    stage_source = dict(staging.get("source") or {})
    simulator = dict(contract.get("canonical_simulator") or {})
    if (
        stage_source.get("canonical_contract") != simulator.get("typed_source")
        or stage_source.get("canonical_contract_sha256") != canonical_identity.sha256
        or stage_source.get("wheel_sha256") != simulator.get("wheel_sha256")
    ):
        raise R241FinalizerError("official-libcg staging is not bound to this r236 source")
    runtime = dict(staging.get("required_runtime") or {})
    linux_member = dict(native_members)["linux_x86_64"]
    if (
        runtime.get("environment") != "CG_LIB_PATH"
        or runtime.get("member") != "cg/libcg.so"
        or runtime.get("member_sha256") != linux_member.get("sha256")
        or int(runtime.get("member_size_bytes", -1))
        != int(linux_member.get("size_bytes", -2))
        or set(runtime.get("forbidden_environment_absent") or [])
        != {"POKEBOT_LIBCG_PATH", "POKEBOT_BATCH_LIBCG"}
        or int(runtime.get("native_function_calls_during_export_attestation", -1)) != 0
        or int(runtime.get("search_calls_during_export_attestation", -1)) != 0
    ):
        raise R241FinalizerError("official-libcg staging runtime evidence is invalid")
    hosts = dict(staging.get("hosts") or {})
    if set(hosts) != {"inzi", "elmo"}:
        raise R241FinalizerError("official-libcg staging lacks the Inzi/Elmo receipts")
    for name, receipt_sha256 in _DEFAULT_STAGING_HOST_RECEIPTS.items():
        host = dict(hosts.get(name) or {})
        if (
            host.get("passed") is not True
            or host.get("loaded_member_sha256") != linux_member.get("sha256")
            or not str(host.get("runtime_root") or "").strip()
            or not str(host.get("receipt") or "").strip()
            or (is_default_staging and host.get("receipt_sha256") != receipt_sha256)
            or (
                not is_default_staging
                and not _SHA256.fullmatch(str(host.get("receipt_sha256") or ""))
            )
        ):
            raise R241FinalizerError(f"official-libcg {name} host receipt is invalid")
    scope = dict(staging.get("scope") or {})
    if (
        scope.get("runtime_roots_created") is not True
        or scope.get("complete_four_platform_native_set_staged_per_root") is not True
        or scope.get("old_wrapper_native_members_discarded") is not True
        or int(scope.get("simulator_battles_started", -1)) != 0
        or scope.get("training_or_gradient_updates_started") is not False
        or scope.get("managed_services_started_or_restarted") is not False
        or scope.get("mcts_or_rtp_authority") is not False
        or scope.get("selector_or_submission_authority") is not False
    ):
        raise R241FinalizerError("official-libcg staging scope is not offline direct-policy only")
    return OfficialCgStaging(
        staging_receipt=identity,
        canonical_contract=canonical_identity,
        native_members=native_members,
    )


def _is_forbidden_member(member: str) -> bool:
    return any(_FORBIDDEN_ASSET_COMPONENT.search(part) for part in member.split("/"))


def _is_unsealed_runtime_member(member: str) -> bool:
    parts = PurePosixPath(member).parts
    return "__pycache__" in parts or Path(member).suffix.casefold() in _FORBIDDEN_UNSEALED_RUNTIME_SUFFIXES


def _check_direct_source(path: Path, *, member: str) -> None:
    """Reject planner/belief code outside the sealed CG API wrapper.

    The official API may expose search symbols from libcg.  It is deliberately
    excluded from this source scan; its byte identity is verified separately.
    """

    if path.suffix.casefold() not in {".py", ".pyi"}:
        return
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise R241FinalizerError(f"direct runtime source is unreadable: {member}") from exc
    if _FORBIDDEN_RUNTIME_SOURCE.search(source):
        raise R241FinalizerError(
            f"direct runtime source contains a forbidden planner/belief token: {member}"
        )


def _walk_runtime_source(
    runtime_root: Path,
) -> tuple[tuple[tuple[str, FileIdentity], ...], tuple[str, ...]]:
    files: list[tuple[str, FileIdentity]] = []
    culled_cg: list[str] = []
    for candidate in sorted(runtime_root.rglob("*")):
        relative = _relative_path(runtime_root, candidate, label="runtime source member")
        if relative.split("/", 1)[0] == "cg":
            if candidate.is_symlink():
                culled_cg.append(relative)
                continue
            if candidate.is_file():
                culled_cg.append(relative)
            continue
        if candidate.is_symlink():
            raise R241FinalizerError(f"direct runtime source has a symlink: {relative}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise R241FinalizerError(f"direct runtime source has a non-regular member: {relative}")
        if relative in _RESERVED_RUNTIME_MEMBERS:
            raise R241FinalizerError(
                f"runtime source attempts to replace generated member: {relative}"
            )
        if _is_forbidden_member(relative):
            raise R241FinalizerError(f"runtime source has a forbidden asset: {relative}")
        if _is_unsealed_runtime_member(relative):
            raise R241FinalizerError(
                f"runtime source has an unsealed executable/model sidecar: {relative}"
            )
        _check_direct_source(candidate, member=relative)
        files.append((relative, _file_identity(candidate, label=f"runtime source {relative}")))
    if not any(name == "main.py" for name, _identity in files):
        raise R241FinalizerError("direct runtime source must supply main.py")
    return tuple(files), tuple(sorted(culled_cg))


def _resolve_official_cg_runtime_root(cg_root: Path) -> tuple[Path, Path]:
    """Accept either a staged CG_LIB_PATH root or its contained ``cg`` dir."""

    supplied = _directory(cg_root, label="sealed official r236 cg directory")
    if supplied.name == "cg":
        runtime_root = _directory(
            supplied.parent, label="sealed official r236 runtime root"
        )
        return runtime_root, supplied
    package = _directory(supplied / "cg", label="sealed official r236 cg package")
    return supplied, package


def _validate_local_official_cg_preflight(
    *,
    runtime_root: Path,
    staging: OfficialCgStaging,
) -> FileIdentity:
    receipt_path, receipt = _read_object(
        runtime_root / OFFICIAL_LIBCG_PREFLIGHT_NAME,
        label="sealed official r236 local preflight",
    )
    members = dict(receipt.get("canonical_native_members") or {})
    expected_members = dict(staging.native_members)
    linux_member = dict(expected_members["linux_x86_64"])
    if (
        receipt.get("schema") != OFFICIAL_LIBCG_PREFLIGHT_SCHEMA
        or int(receipt.get("revision", -1)) != OWNER_REVISION
        or receipt.get("status") != "passed"
        or receipt.get("passed") is not True
        or receipt.get("immutable") is not True
        or receipt.get("write_once") is not True
        or receipt.get("local_only") is not True
        or receipt.get("direct_policy_only") is not True
        or receipt.get("cg_lib_path") != str(runtime_root)
        or set(members) != set(expected_members)
    ):
        raise R241FinalizerError("sealed official r236 local preflight is invalid")
    for platform_name, expected in expected_members.items():
        observed = dict(members.get(platform_name) or {})
        if (
            observed.get("path") != expected.get("package_relative_path")
            or observed.get("sha256") != expected.get("sha256")
            or int(observed.get("size_bytes", -1)) != int(expected.get("size_bytes", -2))
        ):
            raise R241FinalizerError(
                f"sealed official r236 preflight drifted for {platform_name}"
            )
    loaded = dict(receipt.get("loaded_library") or {})
    native_attestation = dict(receipt.get("native_export_attestation") or {})
    environment = dict(receipt.get("environment") or {})
    if (
        loaded.get("target_platform") != "linux_x86_64"
        or loaded.get("path") != str(runtime_root / "cg" / "libcg.so")
        or loaded.get("sha256") != linux_member.get("sha256")
        or int(loaded.get("size_bytes", -1))
        != int(linux_member.get("size_bytes", -2))
        or int(native_attestation.get("native_function_calls", -1)) != 0
        or native_attestation.get("method") != "ctypes_symbol_resolution_only"
        or environment.get("CG_LIB_PATH") != str(runtime_root)
        or environment.get("forbidden_override_keys_absent") is not True
        or set(environment.get("forbidden_override_keys") or [])
        != {"POKEBOT_LIBCG_PATH", "POKEBOT_BATCH_LIBCG"}
    ):
        raise R241FinalizerError("sealed official r236 local preflight is not direct-only")
    wrapper_source = dict(receipt.get("wrapper_source") or {})
    discarded = dict(wrapper_source.get("discarded_native_members") or {})
    if (
        int(wrapper_source.get("copied_member_count", 0))
        < len(_OFFICIAL_CG_WRAPPER_MEMBERS)
        or "cg/libcg.so" not in discarded
    ):
        raise R241FinalizerError("sealed official r236 preflight did not discard old wrapper natives")
    return _file_identity(receipt_path, label="sealed official r236 local preflight")


def _audit_official_cg(
    *,
    cg_root: Path,
    staging: OfficialCgStaging,
) -> tuple[Path, FileIdentity, tuple[tuple[str, FileIdentity], ...], tuple[str, ...]]:
    runtime_root, root = _resolve_official_cg_runtime_root(cg_root)
    local_preflight = _validate_local_official_cg_preflight(
        runtime_root=runtime_root, staging=staging
    )
    expected_native = {
        str(member["package_relative_path"]).removeprefix("cg/"): dict(member)
        for _platform, member in staging.native_members
    }
    allowed_members = _OFFICIAL_CG_WRAPPER_MEMBERS | set(expected_native)
    allowed: list[tuple[str, FileIdentity]] = []
    culled: list[str] = []
    for candidate in sorted(root.rglob("*")):
        relative = _relative_path(root, candidate, label="official cg member")
        if candidate.is_symlink():
            if relative in allowed_members:
                raise R241FinalizerError(
                    f"sealed official cg member is a symlink: {relative}"
                )
            culled.append(relative)
            continue
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise R241FinalizerError(
                f"sealed official cg has a non-regular member: {relative}"
            )
        if relative not in allowed_members:
            culled.append(relative)
            continue
        allowed.append(
            (relative, _file_identity(candidate, label=f"official cg {relative}"))
        )
    identities = dict(allowed)
    missing_wrapper = sorted(_OFFICIAL_CG_WRAPPER_MEMBERS - set(identities))
    missing_native = sorted(set(expected_native) - set(identities))
    if missing_wrapper or missing_native:
        raise R241FinalizerError(
            "sealed official cg is incomplete: "
            + ", ".join(missing_wrapper + missing_native)
        )
    for relative, expected in expected_native.items():
        observed = identities[relative]
        if (
            observed.sha256 != expected.get("sha256")
            or observed.size_bytes != int(expected.get("size_bytes", -1))
        ):
            raise R241FinalizerError(
                f"sealed official cg native member drifted: cg/{relative}"
            )
    return (
        root,
        local_preflight,
        tuple(sorted(allowed)),
        tuple(sorted(culled)),
    )


def audit_runtime_source(
    *,
    runtime_dir: Path,
    official_cg_dir: Path | None,
    contract: Mapping[str, Any],
    official_libcg_staging_path: Path = OFFICIAL_LIBCG_STAGING_PATH,
) -> RuntimeAudit:
    """Audit a direct-policy source tree without executing any of its code."""

    runtime_root = _directory(runtime_dir, label="direct runtime source directory")
    direct_files, culled_runtime_cg = _walk_runtime_source(runtime_root)
    staging = _load_official_cg_staging(
        contract=contract, staging_path=official_libcg_staging_path
    )
    sealed_cg_root = (
        _directory(official_cg_dir, label="sealed official r236 cg directory")
        if official_cg_dir is not None
        else _directory(runtime_root / "cg", label="sealed official r236 cg directory")
    )
    (
        package_cg_root,
        local_preflight,
        official_files,
        culled_official_cg,
    ) = _audit_official_cg(cg_root=sealed_cg_root, staging=staging)
    return RuntimeAudit(
        runtime_root=runtime_root,
        official_cg_root=package_cg_root,
        official_cg_staging=staging,
        official_cg_preflight=local_preflight,
        direct_runtime_files=direct_files,
        official_cg_files=official_files,
        culled_runtime_cg_members=culled_runtime_cg,
        culled_official_cg_members=culled_official_cg,
    )


def _copy_identity(source: FileIdentity, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise R241FinalizerError(f"staging destination already exists: {destination}")
    shutil.copyfile(source.path, destination)
    os.chmod(destination, source.path.stat().st_mode & 0o777)
    copied = _file_identity(destination, label="copied package member")
    if copied.sha256 != source.sha256 or copied.size_bytes != source.size_bytes:
        raise R241FinalizerError("copied package member changed during staging")


def _stage_member_identities(stage: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for candidate in sorted(stage.rglob("*")):
        if candidate.is_symlink() or candidate.is_dir():
            if candidate.is_symlink():
                raise R241FinalizerError("package stage contains a symlink")
            continue
        if not candidate.is_file():
            raise R241FinalizerError("package stage contains a non-regular member")
        member = _relative_path(stage, candidate, label="package stage member")
        result[member] = {
            "sha256": sha256_file(candidate),
            "size_bytes": int(candidate.stat().st_size),
        }
    return result


def _write_stage_json(stage: Path, name: str, payload: Mapping[str, Any]) -> FileIdentity:
    path = stage / name
    if path.exists() or path.is_symlink():
        raise R241FinalizerError(f"generated package member already exists: {name}")
    body = canonical_json(dict(payload))
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    return _file_identity(path, label=f"generated package member {name}")


def _write_deterministic_archive(stage: Path, archive_path: Path) -> None:
    """Create a tar.gz without timestamps, owners, links, or implicit files."""

    with archive_path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for candidate in sorted(path for path in stage.rglob("*") if path.is_file()):
                    if candidate.is_symlink():
                        raise R241FinalizerError("package stage has a symlink")
                    member = _relative_path(stage, candidate, label="package archive member")
                    info = archive.gettarinfo(str(candidate), arcname=member)
                    if not info.isreg():
                        raise R241FinalizerError("package stage has a non-regular member")
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.pax_headers = {}
                    with candidate.open("rb") as source:
                        archive.addfile(info, source)


def _publish_immutable_file(source: Path, target: Path, *, label: str) -> FileIdentity:
    source_identity = _file_identity(source, label=f"temporary {label}")
    target = Path(target).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise R241FinalizerError(f"{label} parent is unsafe")
    if target.exists() or target.is_symlink():
        existing = _file_identity(target, label=label)
        if (
            existing.sha256 != source_identity.sha256
            or existing.size_bytes != source_identity.size_bytes
        ):
            raise R241FinalizerError(f"immutable {label} already exists with different bytes")
        return existing
    try:
        os.link(source, target)
    except FileExistsError:
        existing = _file_identity(target, label=label)
        if (
            existing.sha256 != source_identity.sha256
            or existing.size_bytes != source_identity.size_bytes
        ):
            raise R241FinalizerError(f"immutable {label} raced with different bytes")
        return existing
    except OSError as exc:
        raise R241FinalizerError(f"cannot publish immutable {label}") from exc
    os.chmod(target, 0o444)
    return _file_identity(target, label=label)


def _write_immutable_json(path: Path, payload: Mapping[str, Any], *, label: str) -> FileIdentity:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise R241FinalizerError(f"{label} parent is unsafe")
    body = canonical_json(dict(payload))
    if target.exists() or target.is_symlink():
        existing = _regular_file(target, label=label)
        if existing.read_bytes() != body:
            raise R241FinalizerError(f"immutable {label} already exists with different bytes")
        return _file_identity(existing, label=label)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.r241-", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = _regular_file(target, label=label)
            if existing.read_bytes() != body:
                raise R241FinalizerError(f"immutable {label} raced with different bytes")
            return _file_identity(existing, label=label)
        os.chmod(target, 0o444)
        return _file_identity(target, label=label)
    finally:
        temporary.unlink(missing_ok=True)


def _read_archive_members(
    archive_path: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, bytes]]:
    archive = _regular_file(archive_path, label="r241 direct-policy archive")
    metadata: dict[str, dict[str, object]] = {}
    contents: dict[str, bytes] = {}
    try:
        with tarfile.open(archive, "r:*") as source:
            for member in source.getmembers():
                name = _safe_member_name(member.name, label="r241 archive member")
                if name in metadata:
                    raise R241FinalizerError("r241 archive has duplicate member paths")
                if member.isdir():
                    continue
                if not member.isfile() or member.issym() or member.islnk() or member.isdev():
                    raise R241FinalizerError("r241 archive has a non-regular or linked member")
                stream = source.extractfile(member)
                if stream is None:
                    raise R241FinalizerError("r241 archive member cannot be read")
                body = stream.read()
                if len(body) != member.size:
                    raise R241FinalizerError("r241 archive member is truncated")
                metadata[name] = {
                    "sha256": sha256_bytes(body),
                    "size_bytes": len(body),
                    "mode": member.mode & 0o777,
                }
                contents[name] = body
    except (OSError, tarfile.TarError) as exc:
        raise R241FinalizerError("r241 direct-policy archive is unreadable") from exc
    if not metadata:
        raise R241FinalizerError("r241 direct-policy archive has no files")
    return metadata, contents


def _archive_json(contents: Mapping[str, bytes], name: str, *, label: str) -> dict[str, Any]:
    raw = contents.get(name)
    if raw is None:
        raise R241FinalizerError(f"r241 archive lacks {label}: {name}")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241FinalizerError(f"r241 archive {label} is unreadable JSON") from exc
    if not isinstance(value, dict):
        raise R241FinalizerError(f"r241 archive {label} is not an object")
    return value


def _check_archive_direct_sources(contents: Mapping[str, bytes]) -> None:
    for name, body in contents.items():
        if name.startswith("cg/") or not name.endswith((".py", ".pyi")):
            continue
        try:
            source = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise R241FinalizerError(f"archive source is not UTF-8: {name}") from exc
        if _FORBIDDEN_RUNTIME_SOURCE.search(source):
            raise R241FinalizerError(
                f"archive source contains a forbidden planner/belief token: {name}"
            )


def audit_direct_policy_archive(
    archive_path: Path,
    *,
    contract: Mapping[str, Any],
    expected_model_sha256: str | None = None,
    official_cg_staging: OfficialCgStaging | None = None,
    official_cg_preflight: FileIdentity | None = None,
    expected_matchup_runtime: MatchupRuntimeEvidence | None = None,
    expected_model_runtime: ModelRuntimeEvidence | None = None,
) -> dict[str, object]:
    """Verify that an archive has exactly direct-policy packaging semantics."""

    archive = _regular_file(archive_path, label="r241 direct-policy archive")
    if archive.stat().st_size > ARCHIVE_MAX_BYTES:
        raise R241FinalizerError("r241 archive exceeds the exact 197.7 MiB cap")
    members, contents = _read_archive_members(archive)
    forbidden = sorted(name for name in members if _is_forbidden_member(name))
    if forbidden:
        raise R241FinalizerError(
            "r241 archive contains forbidden MCTS/RTP/search/belief assets: "
            + ", ".join(forbidden)
        )
    unsealed = sorted(
        name
        for name in members
        if name != "model.pt"
        and not name.startswith("cg/")
        and _is_unsealed_runtime_member(name)
    )
    if unsealed:
        raise R241FinalizerError(
            "r241 archive contains an unsealed executable/model sidecar: "
            + ", ".join(unsealed)
        )
    required = {
        "main.py",
        "model.pt",
        "deck.csv",
        "matchup_tree.json",
        "matchup_runtime_activation.json",
        "model_runtime_activation.json",
        "runtime_profile.json",
        "turn_order_profile.json",
        "package_manifest.json",
    }
    staging = official_cg_staging or _load_official_cg_staging(
        contract=contract,
        staging_path=OFFICIAL_LIBCG_STAGING_PATH,
    )
    expected_native = {
        str(member["package_relative_path"])
        for _platform, member in staging.native_members
    }
    required |= {f"cg/{member}" for member in _OFFICIAL_CG_WRAPPER_MEMBERS}
    required |= expected_native
    missing = sorted(required - set(members))
    if missing:
        raise R241FinalizerError("r241 archive is missing required members: " + ", ".join(missing))
    _check_archive_direct_sources(contents)

    simulator = dict(contract.get("canonical_simulator") or {})
    library = dict(members["cg/libcg.so"])
    for _platform, native in staging.native_members:
        member_name = str(native["package_relative_path"])
        observed = dict(members[member_name])
        if (
            observed.get("sha256") != native.get("sha256")
            or int(observed.get("size_bytes", -1))
            != int(native.get("size_bytes", -2))
        ):
            raise R241FinalizerError(
                f"archive native member is not sealed official r236: {member_name}"
            )
    if (
        library.get("sha256") != simulator.get("linux_x86_64_sha256")
        or int(library.get("size_bytes", -1))
        != int(simulator.get("linux_x86_64_size_bytes", -2))
    ):
        raise R241FinalizerError("archive does not contain sealed official r236 D162 libcg")
    deck = _validate_exact_deck(contract)
    expert_window = _load_expert_window_evidence(contract)
    if members["deck.csv"].get("sha256") != deck.sha256:
        raise R241FinalizerError("archive deck does not match the r241 exact list")
    if expected_model_sha256 is not None and (
        members["model.pt"].get("sha256") != expected_model_sha256
    ):
        raise R241FinalizerError("archive model does not match terminal expert checkpoint")
    tree_member = dict(members["matchup_tree.json"])
    if (
        tree_member.get("sha256") != LEARNER_R195_MATCHUP_TREE_SHA256
        or int(tree_member.get("size_bytes", -1)) <= 0
    ):
        raise R241FinalizerError("archive matchup tree is not the pinned r195 learner tree")
    activation = _archive_json(
        contents, "matchup_runtime_activation.json", label="matchup runtime activation"
    )
    _validate_checkpoint_receipt_fingerprint(
        activation, label="archive matchup runtime activation"
    )
    matchup_runtime_smoke = dict(activation.get("runtime_smoke") or {})
    h10_training_opponent = dict(activation.get("h10_training_opponent") or {})
    if (
        activation.get("schema") != MATCHUP_RUNTIME_ACTIVATION_SCHEMA
        or int(activation.get("owner_decision_revision", -1)) != OWNER_REVISION
        or activation.get("owner_clarification_revision")
        != LATEST_OWNER_CLARIFICATION_REVISION
        or activation.get("candidate_id") != "alakazam-new-list-direct-policy-r241"
        or activation.get("status") != "active_direct_policy_only"
        or activation.get("derived_not_self_asserted") is not True
        or activation.get("action_selector") != "direct_policy_only"
        or activation.get("mcts_enabled") is not False
        or activation.get("recursive_turn_planner_enabled") is not False
        or activation.get("search_enabled") is not False
        or activation.get("belief_assets_enabled") is not False
        or dict(activation.get("terminal_checkpoint") or {}).get("sha256")
        != members["model.pt"].get("sha256")
        or dict(activation.get("learner_matchup_tree") or {}).get("sha256")
        != tree_member.get("sha256")
        or matchup_runtime_smoke.get("model_reconstructed") is not True
        or matchup_runtime_smoke.get("adapter_runtime_enabled_for_smoke") is not True
        or matchup_runtime_smoke.get("adapter_output_changed") is not True
        or matchup_runtime_smoke.get("action_selector") != "direct_policy_only"
        or matchup_runtime_smoke.get("mcts_calls") != 0
        or matchup_runtime_smoke.get("rtp_calls") != 0
        or matchup_runtime_smoke.get("search_calls") != 0
        or dict(h10_training_opponent.get("matchup_tree") or {}).get("sha256")
        != H10_DIRECT_MATCHUP_TREE_SHA256
        or dict(h10_training_opponent.get("matchup_tree") or {}).get("size_bytes")
        != H10_DIRECT_MATCHUP_TREE_SIZE_BYTES
        or h10_training_opponent.get("direct_policy_only") is not True
        or h10_training_opponent.get("mcts_enabled") is not False
        or h10_training_opponent.get("recursive_turn_planner_enabled") is not False
        or h10_training_opponent.get("search_enabled") is not False
        or dict(activation.get("adapter_slot_migration") or {}).get("status")
        != "no_slot_change"
        or list(dict(activation.get("adapter_slot_migration") or {}).get("new_slots") or [])
        or list(dict(activation.get("adapter_slot_migration") or {}).get("new_slot_proofs") or [])
    ):
        raise R241FinalizerError("archive matchup adapter runtime is not active/direct-only")
    if expected_matchup_runtime is not None:
        if (
            tree_member.get("sha256") != expected_matchup_runtime.tree.sha256
            or dict(members["matchup_runtime_activation.json"]).get("sha256")
            != expected_matchup_runtime.activation.sha256
        ):
            raise R241FinalizerError("archive matchup runtime identity changed")

    model_activation_member = dict(members["model_runtime_activation.json"])
    model_activation = _archive_json(
        contents, "model_runtime_activation.json", label="model runtime activation"
    )
    _validate_checkpoint_receipt_fingerprint(
        model_activation, label="archive model runtime activation"
    )
    model_runtime_smoke = dict(model_activation.get("runtime_smoke") or {})
    model_heads = dict(model_activation.get("heads") or {})
    combo_state = dict(model_heads.get("combo_state") or {})
    active_heads = _strict_name_tuple(
        model_heads.get("active_non_combo_head_names"),
        label="archive model active non-combo inventory",
    )
    inventory = _load_peak_r195_head_inventory()
    active_routes = inventory.fusion_route_ids
    model_active_routes = _strict_name_tuple(
        model_heads.get("active_non_combo_fusion_route_ids"),
        label="archive model active non-combo fusion-route inventory",
    )
    preservation = dict(contract.get("peak_r195_behavior_preservation") or {})
    if (
        active_heads != inventory.head_names
        or model_active_routes != active_routes
        or model_activation.get("schema") != MODEL_RUNTIME_ACTIVATION_SCHEMA
        or int(model_activation.get("owner_decision_revision", -1)) != OWNER_REVISION
        or model_activation.get("owner_clarification_revision")
        != LATEST_OWNER_CLARIFICATION_REVISION
        or model_activation.get("candidate_id")
        != "alakazam-new-list-direct-policy-r241"
        or model_activation.get("status") != "active_peak_r195_non_combo_fusion"
        or model_activation.get("derived_not_self_asserted") is not True
        or dict(model_activation.get("parent_r195_checkpoint") or {}).get("sha256")
        != PARENT_R195_SHA256
        or model_heads.get("architecture_present_head_count") != 19
        or model_heads.get("non_combo_head_count") != 18
        or model_heads.get("non_combo_route_count") != 18
        or model_heads.get("every_non_combo_head_trainable") is not True
        or model_heads.get("every_non_combo_fusion_route_enabled") is not True
        or combo_state.get("head_present") is not True
        or combo_state.get("physical_route_present") is not True
        or combo_state.get("loss_weight") != 0.0
        or combo_state.get("route_enabled") is not False
        or dict(model_activation.get("runtime_package_activation") or {}).get(
            "matchup_adapters_enabled"
        )
        is not True
        or dict(model_activation.get("runtime_package_activation") or {}).get(
            "checkpoint_remains_dormant"
        )
        is not True
        or model_activation.get("action_selector") != "direct_policy_only"
        or model_activation.get("mcts_enabled") is not False
        or model_activation.get("recursive_turn_planner_enabled") is not False
        or model_activation.get("search_enabled") is not False
        or model_activation.get("belief_assets_enabled") is not False
        or dict(model_activation.get("terminal_checkpoint") or {}).get("sha256")
        != members["model.pt"].get("sha256")
        or model_runtime_smoke.get("model_reconstructed") is not True
        or model_runtime_smoke.get("adapter_runtime_enabled_for_smoke") is not True
        or model_runtime_smoke.get("adapter_output_changed") is not True
        or model_runtime_smoke.get("mcts_calls") != 0
        or model_runtime_smoke.get("rtp_calls") != 0
        or model_runtime_smoke.get("search_calls") != 0
        or dict(model_activation.get("adapter_slot_migration") or {}).get("status")
        != "no_slot_change"
        or list(dict(model_activation.get("adapter_slot_migration") or {}).get("new_slots") or [])
        or list(dict(model_activation.get("adapter_slot_migration") or {}).get("new_slot_proofs") or [])
        or preservation.get("every_architecture_present_non_combo_head_trainable")
        is not True
        or preservation.get("every_architecture_present_non_combo_fusion_route_enabled")
        is not True
    ):
        raise R241FinalizerError("archive model runtime inventory is not r195 direct active")
    if expected_model_runtime is not None and (
        model_activation_member.get("sha256") != expected_model_runtime.activation.sha256
        or tuple(active_heads) != expected_model_runtime.active_head_names
        or tuple(model_active_routes) != expected_model_runtime.active_fusion_route_ids
    ):
        raise R241FinalizerError("archive model runtime identity changed")

    profile = _archive_json(contents, "runtime_profile.json", label="runtime profile")
    expected_profile = {
        "schema": RUNTIME_PROFILE_SCHEMA,
        "action_selector": "direct_policy_only",
        "mcts_enabled": False,
        "recursive_turn_planner_enabled": False,
        "search_enabled": False,
        "belief_assets_enabled": False,
        "matchup_adapter_runtime_enabled": True,
        "peak_r195_non_combo_heads_active": True,
        "fusion_routes_active": True,
        "combo_state_route_enabled": False,
        "inherited_cg_copied": False,
    }
    if any(profile.get(key) != value for key, value in expected_profile.items()):
        raise R241FinalizerError("archive runtime profile is not direct-policy-only")
    if (
        profile.get("matchup_runtime_activation_sha256")
        != dict(members["matchup_runtime_activation.json"]).get("sha256")
        or profile.get("model_runtime_activation_sha256")
        != dict(members["model_runtime_activation.json"]).get("sha256")
    ):
        raise R241FinalizerError("archive runtime profile activation identities drifted")
    profile_cg = dict(profile.get("official_r236_libcg") or {})
    if (
        profile_cg.get("sha256") != simulator.get("linux_x86_64_sha256")
        or profile_cg.get("binding_environment") != "CG_LIB_PATH"
        or set(profile_cg.get("forbidden_environment") or [])
        != {"POKEBOT_LIBCG_PATH", "POKEBOT_BATCH_LIBCG"}
    ):
        raise R241FinalizerError("archive runtime profile does not bind official r236 libcg")
    if profile.get("official_r236_staging_sha256") != staging.staging_receipt.sha256:
        raise R241FinalizerError("archive runtime profile is not bound to r241 libcg staging")
    if official_cg_preflight is not None and (
        profile.get("official_r236_local_preflight_sha256")
        != official_cg_preflight.sha256
    ):
        raise R241FinalizerError("archive runtime profile local preflight identity changed")
    turn_order = _archive_json(
        contents, "turn_order_profile.json", label="turn-order profile"
    )
    if (
        turn_order.get("schema") != TURN_ORDER_PROFILE_SCHEMA
        or turn_order.get("turn_order_preference") != "first_if_allowed"
    ):
        raise R241FinalizerError("archive turn-order profile is not first_if_allowed")
    manifest = _archive_json(contents, "package_manifest.json", label="package manifest")
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or int(manifest.get("owner_decision_revision", -1)) != OWNER_REVISION
        or manifest.get("direct_policy_only") is not True
        or int(manifest.get("submission_count_authorized", -1)) != 1
        or manifest.get("turn_order_preference") != "first_if_allowed"
        or manifest.get("official_r236_staging_sha256")
        != staging.staging_receipt.sha256
        or not _SHA256.fullmatch(
            str(manifest.get("official_r236_local_preflight_sha256") or "")
        )
    ):
        raise R241FinalizerError("archive package manifest is not an r241 direct package")
    model_provenance = dict(manifest.get("terminal_model_provenance") or {})
    if (
        model_provenance.get("source")
        != "terminal_expert_before_iter_00010_five_epoch_receipt"
        or model_provenance.get("terminal_checkpoint_sha256")
        != members["model.pt"].get("sha256")
        or not _SHA256.fullmatch(str(model_provenance.get("iter_00009_parent_sha256") or ""))
        or not _SHA256.fullmatch(
            str(model_provenance.get("terminal_rehearsal_receipt_sha256") or "")
        )
        or model_provenance.get("expert_window_staging_sha256")
        != expert_window.staging_receipt.sha256
        or model_provenance.get("expert_window_canonical_receipt_sha256")
        != expert_window.canonical_receipt_sha256
        or model_provenance.get("expert_window_immutable_receipt_sha256")
        != expert_window.immutable_window_receipt_sha256
    ):
        raise R241FinalizerError("archive lacks terminal model-content provenance")
    manifest_matchup = dict(manifest.get("matchup_runtime") or {})
    if (
        dict(manifest_matchup.get("matchup_tree") or {}).get("sha256")
        != tree_member.get("sha256")
        or dict(manifest_matchup.get("runtime_activation") or {}).get("sha256")
        != dict(members["matchup_runtime_activation.json"]).get("sha256")
        or manifest_matchup.get("adapter_runtime_enabled") is not True
        or manifest_matchup.get("action_selector") != "direct_policy_only"
    ):
        raise R241FinalizerError("archive matchup runtime provenance is incomplete")
    manifest_model = dict(manifest.get("model_runtime") or {})
    if (
        dict(manifest_model.get("runtime_activation") or {}).get("sha256")
        != model_activation_member.get("sha256")
        or tuple(manifest_model.get("active_head_names") or []) != active_heads
        or tuple(manifest_model.get("active_fusion_route_ids") or []) != active_routes
        or manifest_model.get("all_peak_r195_non_combo_heads_active") is not True
        or manifest_model.get("fusion_routes_active") is not True
        or manifest_model.get("combo_state_loss_weight") != 0.0
        or manifest_model.get("combo_state_route_enabled") is not False
        or manifest_model.get("matchup_adapter_runtime_enabled") is not True
    ):
        raise R241FinalizerError("archive model runtime provenance is incomplete")
    return {
        "passed": True,
        "archive_path": str(archive),
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": int(archive.stat().st_size),
        "archive_max_bytes": ARCHIVE_MAX_BYTES,
        "member_count": len(members),
        "members": members,
        "forbidden_members": forbidden,
        "direct_policy_only": True,
        "official_r236_d162_libcg": library,
        "official_r236_local_preflight_sha256": profile.get(
            "official_r236_local_preflight_sha256"
        ),
        "matchup_runtime": manifest_matchup,
        "model_runtime": manifest_model,
        "package_manifest_sha256": members["package_manifest.json"]["sha256"],
    }


def _runtime_profile(
    contract: Mapping[str, Any],
    runtime: RuntimeAudit,
    matchup_runtime: MatchupRuntimeEvidence,
    model_runtime: ModelRuntimeEvidence,
) -> dict[str, object]:
    simulator = dict(contract.get("canonical_simulator") or {})
    library = dict(runtime.official_cg_files).get("libcg.so")
    if library is None:
        raise R241FinalizerError("official libcg audit unexpectedly lacks libcg.so")
    return {
        "schema": RUNTIME_PROFILE_SCHEMA,
        "owner_decision_revision": OWNER_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "action_selector": "direct_policy_only",
        "mcts_enabled": False,
        "recursive_turn_planner_enabled": False,
        "search_enabled": False,
        "belief_assets_enabled": False,
        "matchup_adapter_runtime_enabled": True,
        "peak_r195_non_combo_heads_active": True,
        "fusion_routes_active": True,
        "combo_state_route_enabled": False,
        "matchup_runtime_activation_sha256": matchup_runtime.activation.sha256,
        "model_runtime_activation_sha256": model_runtime.activation.sha256,
        "inherited_cg_copied": False,
        "official_r236_staging_sha256": runtime.official_cg_staging.staging_receipt.sha256,
        "official_r236_local_preflight_sha256": runtime.official_cg_preflight.sha256,
        "official_r236_libcg": {
            "member": "cg/libcg.so",
            "sha256": library.sha256,
            "size_bytes": library.size_bytes,
            "binding_environment": "CG_LIB_PATH",
            "forbidden_environment": list(simulator["forbidden_environment"]),
        },
    }


def build_direct_policy_package(
    *,
    evidence: TerminalEvidence,
    contract: Mapping[str, Any],
    runtime_dir: Path,
    official_cg_dir: Path | None,
    output_dir: Path,
    matchup_tree_path: Path,
    matchup_runtime_activation_path: Path,
    model_runtime_activation_path: Path,
    official_libcg_staging_path: Path = OFFICIAL_LIBCG_STAGING_PATH,
) -> PackageArtifact:
    """Package the terminal expert checkpoint without any queue or upload action."""

    output = _directory(output_dir, label="r241 package output directory", create=True)
    runtime = audit_runtime_source(
        runtime_dir=runtime_dir,
        official_cg_dir=official_cg_dir,
        contract=contract,
        official_libcg_staging_path=official_libcg_staging_path,
    )
    _validate_generated_terminal_checkpoint_receipts(
        model_runtime_activation_path=model_runtime_activation_path,
        matchup_runtime_activation_path=matchup_runtime_activation_path,
        matchup_tree_path=matchup_tree_path,
        terminal=evidence,
        contract=contract,
        official_cg_root=runtime.official_cg_root,
    )
    deck = _validate_exact_deck(contract)
    matchup_runtime = validate_matchup_runtime_evidence(
        matchup_tree_path=matchup_tree_path,
        runtime_activation_path=matchup_runtime_activation_path,
        terminal=evidence,
    )
    model_runtime = validate_model_runtime_evidence(
        runtime_activation_path=model_runtime_activation_path,
        terminal=evidence,
        contract=contract,
    )
    temporary_root = Path(tempfile.mkdtemp(prefix=".r241-package-", dir=output))
    try:
        stage = temporary_root / "stage"
        stage.mkdir()
        for member, identity in runtime.direct_runtime_files:
            _copy_identity(identity, stage / member)
        for member, identity in runtime.official_cg_files:
            _copy_identity(identity, stage / "cg" / member)
        _copy_identity(evidence.terminal_checkpoint, stage / "model.pt")
        _copy_identity(deck, stage / "deck.csv")
        _copy_identity(matchup_runtime.tree, stage / "matchup_tree.json")
        _copy_identity(
            matchup_runtime.activation, stage / "matchup_runtime_activation.json"
        )
        _copy_identity(
            model_runtime.activation, stage / "model_runtime_activation.json"
        )
        _write_stage_json(
            stage,
            "runtime_profile.json",
            _runtime_profile(contract, runtime, matchup_runtime, model_runtime),
        )
        _write_stage_json(
            stage,
            "turn_order_profile.json",
            {
                "schema": TURN_ORDER_PROFILE_SCHEMA,
                "turn_order_preference": "first_if_allowed",
            },
        )
        _write_stage_json(
            stage,
            "package_manifest.json",
            {
                "schema": PACKAGE_SCHEMA,
                "owner_decision_revision": OWNER_REVISION,
                "candidate_id": "alakazam-new-list-direct-policy-r241",
                "direct_policy_only": True,
                "submission_count_authorized": 1,
                "turn_order_preference": "first_if_allowed",
                "model_sha256": evidence.terminal_checkpoint.sha256,
                "deck_sha256": deck.sha256,
                "official_r236_libcg_sha256": dict(runtime.official_cg_files)[
                    "libcg.so"
                ].sha256,
                "official_r236_staging_sha256": runtime.official_cg_staging.staging_receipt.sha256,
                "official_r236_local_preflight_sha256": runtime.official_cg_preflight.sha256,
                "official_r236_native_members": {
                    platform: dict(member)
                    for platform, member in runtime.official_cg_staging.native_members
                },
                "terminal_model_provenance": {
                    "source": "terminal_expert_before_iter_00010_five_epoch_receipt",
                    "terminal_checkpoint_sha256": evidence.terminal_checkpoint.sha256,
                    "iter_00009_parent_sha256": evidence.terminal_parent.sha256,
                    "iter_00009_commit_sha256": evidence.terminal_commit.sha256,
                "terminal_rehearsal_receipt_sha256": evidence.terminal_rehearsal.sha256,
                "terminal_refresh_receipt_sha256": evidence.terminal_refresh.sha256,
                "expert_window_staging_sha256": evidence.expert_window.staging_receipt.sha256,
                "expert_window_canonical_receipt_sha256": evidence.expert_window.canonical_receipt_sha256,
                "expert_window_immutable_receipt_sha256": evidence.expert_window.immutable_window_receipt_sha256,
                },
                "matchup_runtime": matchup_runtime.as_dict(),
                "model_runtime": model_runtime.as_dict(),
                "files_before_manifest": _stage_member_identities(stage),
                "inherited_cg_copied": False,
                "culled_runtime_cg_members": list(runtime.culled_runtime_cg_members),
                "culled_official_cg_members": list(runtime.culled_official_cg_members),
            },
        )
        temporary_archive = temporary_root / "submission.tar.gz"
        _write_deterministic_archive(stage, temporary_archive)
        if temporary_archive.stat().st_size > ARCHIVE_MAX_BYTES:
            raise R241FinalizerError("r241 package exceeds the exact 197.7 MiB cap")
        archive = _publish_immutable_file(
            temporary_archive, output / "submission.tar.gz", label="r241 package archive"
        )
        audit = audit_direct_policy_archive(
            archive.path,
            contract=contract,
            expected_model_sha256=evidence.terminal_checkpoint.sha256,
            official_cg_staging=runtime.official_cg_staging,
            official_cg_preflight=runtime.official_cg_preflight,
            expected_matchup_runtime=matchup_runtime,
            expected_model_runtime=model_runtime,
        )
        return PackageArtifact(
            archive=archive,
            archive_audit=audit,
            deck=deck,
            matchup_runtime=matchup_runtime,
            model_runtime=model_runtime,
            runtime_audit=runtime,
        )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _queue_authorization_payload(
    *, evidence: TerminalEvidence, package: PackageArtifact
) -> dict[str, object]:
    binding = sha256_bytes(
        canonical_json(
            {
                "contract_sha256": evidence.contract_sha256,
                "terminal_checkpoint_sha256": evidence.terminal_checkpoint.sha256,
                "terminal_rehearsal_sha256": evidence.terminal_rehearsal.sha256,
                "archive_sha256": package.archive.sha256,
            }
        )
    )
    entry = {
        "sequence": 1,
        "single_use_nonce": binding,
        "queue_status": "authorized_pending_external_queue",
        "turn_order_preference": "first_if_allowed",
        "maximum_uses": 1,
        "remaining_uses": 1,
        "retry_allowed": False,
        "duplicate_allowed": False,
        "copy_allowed": False,
        "direct_policy_only": True,
    }
    return {
        "schema": QUEUE_AUTHORIZATION_SCHEMA,
        "owner_decision_revision": OWNER_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "competition": "pokemon-tcg-ai-battle",
        "authorization_scope": "one_external_queue_entry_only",
        "submission_count_authorized": 1,
        "submission_count_emitted": 1,
        "turn_order_preference": "first_if_allowed",
        "queue_entries": [entry],
        "terminal_checkpoint": evidence.terminal_checkpoint.as_dict(),
        "terminal_five_epoch_rehearsal_receipt": evidence.terminal_rehearsal.as_dict(),
        "expert_window": evidence.expert_window.as_dict(),
        "archive": package.archive.as_dict(),
        "direct_policy_only": True,
        "direct_submission_performed": False,
        "upload_receipt_present": False,
        "emitter_network_io_performed": False,
        "emitter_queue_mutation_performed": False,
        "submission_execution": "external_queue_processor_only",
    }


def _reserve_single_queue_authorization(
    *,
    evidence: TerminalEvidence,
    authorization_path: Path,
    authorization_payload: Mapping[str, Any],
) -> FileIdentity:
    """Bind this run to one authorization target before that target is written.

    The stable marker lives under the isolated r241 run root rather than in a
    caller-selected output directory.  A later finalizer invocation therefore
    cannot obtain a second slot merely by supplying a new receipt or output
    path.  It is a bookkeeping receipt, not a queue entry and not an upload.
    """

    target = Path(authorization_path).expanduser().absolute()
    expected_authorization_sha256 = sha256_bytes(canonical_json(authorization_payload))
    binding = {
        "schema": AUTHORIZATION_BINDING_SCHEMA,
        "owner_decision_revision": OWNER_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "authorization_path": str(target),
        "authorization_sha256": expected_authorization_sha256,
        "submission_count_authorized": 1,
        "turn_order_preference": "first_if_allowed",
        "direct_submission_performed": False,
        "emitter_network_io_performed": False,
    }
    return _write_immutable_json(
        evidence.run_dir / AUTHORIZATION_BINDING_NAME,
        binding,
        label="r241 single-use queue authorization binding",
    )


def finalize_terminal(
    *,
    run_dir: Path,
    runtime_dir: Path,
    output_dir: Path,
    matchup_tree_path: Path,
    matchup_runtime_activation_path: Path,
    model_runtime_activation_path: Path,
    receipt_path: Path,
    queue_authorization_path: Path,
    contract_path: Path = CONTRACT_PATH,
    terminal_refresh_path: Path | None = None,
    official_cg_dir: Path | None = None,
    official_libcg_staging_path: Path = OFFICIAL_LIBCG_STAGING_PATH,
) -> dict[str, object]:
    """Emit one package and one queue authorization; never submit anything."""

    if Path(receipt_path).expanduser() == Path(queue_authorization_path).expanduser():
        raise R241FinalizerError("finalizer receipt and queue authorization must differ")
    contract, evidence = validate_terminal_evidence(
        run_dir=run_dir,
        contract_path=contract_path,
        terminal_refresh_path=terminal_refresh_path,
    )
    package = build_direct_policy_package(
        evidence=evidence,
        contract=contract,
        runtime_dir=runtime_dir,
        official_cg_dir=official_cg_dir,
        output_dir=output_dir,
        matchup_tree_path=matchup_tree_path,
        matchup_runtime_activation_path=matchup_runtime_activation_path,
        model_runtime_activation_path=model_runtime_activation_path,
        official_libcg_staging_path=official_libcg_staging_path,
    )
    authorization_payload = _queue_authorization_payload(evidence=evidence, package=package)
    authorization_binding = _reserve_single_queue_authorization(
        evidence=evidence,
        authorization_path=queue_authorization_path,
        authorization_payload=authorization_payload,
    )
    authorization = _write_immutable_json(
        queue_authorization_path,
        authorization_payload,
        label="r241 single-use queue authorization",
    )
    receipt = {
        "schema": FINALIZER_RECEIPT_SCHEMA,
        "owner_decision_revision": OWNER_REVISION,
        "candidate_id": "alakazam-new-list-direct-policy-r241",
        "status": "package_built_and_exactly_one_queue_authorization_emitted",
        "terminal_evidence": evidence.as_dict(),
        "package": package.as_dict(),
        "single_use_authorization_binding": authorization_binding.as_dict(),
        "queue_authorization": authorization.as_dict(),
        "queue_authorizations_emitted": 1,
        "submission_count_authorized": 1,
        "submission_count_performed": 0,
        "turn_order_preference": "first_if_allowed",
        "direct_policy_only": True,
        "direct_submission_performed": False,
        "upload_receipt_present": False,
        "network_io_performed": False,
        "shared_queue_mutated": False,
    }
    immutable = _write_immutable_json(
        receipt_path, receipt, label="r241 terminal finalizer receipt"
    )
    return {
        **receipt,
        "receipt": immutable.as_dict(),
    }


def _check_payload(
    *,
    run_dir: Path,
    runtime_dir: Path,
    contract_path: Path,
    terminal_refresh_path: Path | None,
    official_cg_dir: Path | None,
    official_libcg_staging_path: Path,
    matchup_tree_path: Path,
    matchup_runtime_activation_path: Path,
    model_runtime_activation_path: Path,
) -> dict[str, object]:
    contract, evidence = validate_terminal_evidence(
        run_dir=run_dir,
        contract_path=contract_path,
        terminal_refresh_path=terminal_refresh_path,
    )
    runtime = audit_runtime_source(
        runtime_dir=runtime_dir,
        official_cg_dir=official_cg_dir,
        contract=contract,
        official_libcg_staging_path=official_libcg_staging_path,
    )
    _validate_generated_terminal_checkpoint_receipts(
        model_runtime_activation_path=model_runtime_activation_path,
        matchup_runtime_activation_path=matchup_runtime_activation_path,
        matchup_tree_path=matchup_tree_path,
        terminal=evidence,
        contract=contract,
        official_cg_root=runtime.official_cg_root,
    )
    deck = _validate_exact_deck(contract)
    matchup_runtime = validate_matchup_runtime_evidence(
        matchup_tree_path=matchup_tree_path,
        runtime_activation_path=matchup_runtime_activation_path,
        terminal=evidence,
    )
    model_runtime = validate_model_runtime_evidence(
        runtime_activation_path=model_runtime_activation_path,
        terminal=evidence,
        contract=contract,
    )
    return {
        "schema": FINALIZER_RECEIPT_SCHEMA,
        "status": "offline_preflight_passed_no_package_or_authorization_written",
        "terminal_evidence": evidence.as_dict(),
        "deck": deck.as_dict(),
        "matchup_runtime": matchup_runtime.as_dict(),
        "model_runtime": model_runtime.as_dict(),
        "runtime_source_audit": runtime.as_dict(),
        "direct_submission_performed": False,
        "network_io_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--official-cg-dir", type=Path)
    parser.add_argument(
        "--official-libcg-staging",
        type=Path,
        default=OFFICIAL_LIBCG_STAGING_PATH,
        help="checksum-bound r241 official-libcg staging receipt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matchup-tree", type=Path, required=True)
    parser.add_argument("--matchup-runtime-activation", type=Path, required=True)
    parser.add_argument("--model-runtime-activation", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--queue-authorization", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--terminal-refresh", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate receipt/runtime inputs only; write neither package nor authorization",
    )
    args = parser.parse_args(argv)
    if args.check:
        result = _check_payload(
            run_dir=args.run_dir,
            runtime_dir=args.runtime_dir,
            contract_path=args.contract,
            terminal_refresh_path=args.terminal_refresh,
            official_cg_dir=args.official_cg_dir,
            official_libcg_staging_path=args.official_libcg_staging,
            matchup_tree_path=args.matchup_tree,
            matchup_runtime_activation_path=args.matchup_runtime_activation,
            model_runtime_activation_path=args.model_runtime_activation,
        )
    else:
        result = finalize_terminal(
            run_dir=args.run_dir,
            runtime_dir=args.runtime_dir,
            output_dir=args.output_dir,
            receipt_path=args.receipt,
            queue_authorization_path=args.queue_authorization,
            contract_path=args.contract,
            terminal_refresh_path=args.terminal_refresh,
            official_cg_dir=args.official_cg_dir,
            official_libcg_staging_path=args.official_libcg_staging,
            matchup_tree_path=args.matchup_tree,
            matchup_runtime_activation_path=args.matchup_runtime_activation,
            model_runtime_activation_path=args.model_runtime_activation,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
