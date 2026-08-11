"""Checkpoint-derived, fail-closed receipts for the isolated r241 lineage.

The r241 launch and terminal package are deliberately not allowed to trust a
hand-written claim that a checkpoint still has the r195 Fusion-v3 learner or
its trained matchup-adapter bank.  This module reads the immutable checkpoint
bytes, reconstructs the model, inspects the serialized tensors and optimizer
state, and emits small JSON receipts whose assertions can be checked again at
the launch/package boundary.

It performs no collection, optimization, service control, queue mutation, or
network I/O.  The only optional write is a create-only JSON receipt.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

from poke_bot.r241_direct_policy_runtime import (
    R241_H10_ADAPTER_RECEIPT_BASENAME,
    R241_PEAK_R195_PRESERVATION_RECEIPT_BASENAME,
)


ROOT = Path(__file__).resolve().parents[1]

R241_REVISION = 241
R241_OWNER_CLARIFICATION_REVISION = 251
R241_CANDIDATE_ID = "alakazam-new-list-direct-policy-r241"
R241_CONTRACT_SCHEMA = "poke_bot.alakazam_new_list_direct_policy_r241/v1"
R241_CHECKPOINT_AUDIT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_checkpoint_audit/v2"
)
R241_SOURCE_SNAPSHOT_SCHEMA = "poke_bot.alakazam_new_list_direct_r241_source_snapshot/v1"
R241_ADAPTER_SLOT_MIGRATION_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_adapter_slot_migration/v1"
)
TERMINAL_EXPERT_REFRESH_SCHEMA = "poke_bot.terminal_expert_soft_refresh/v1"
R241_PEAK_R195_LIVE_FUSION_SCHEMA = "poke_bot.peak_r195_live_fusion/v1"
PEAK_R195_PRESERVATION_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_peak_r195_preservation/v2"
)
MODEL_RUNTIME_ACTIVATION_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_model_runtime_activation/v2"
)
MATCHUP_RUNTIME_ACTIVATION_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_matchup_runtime_activation/v2"
)

PARENT_R195_SHA256 = (
    "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
)
PARENT_R195_SIZE_BYTES = 127_914_385
PARENT_R195_ADAPTER_ACTIVATION_SHA256 = (
    "sha256:ea38dcb143f3300f0acd6c821aa492e99532e1227e0b8b8c71e80f8fa82d5cc2"
)
PARENT_R195_ADAPTER_ACTIVATION_SIZE_BYTES = 989
PARENT_R195_TYPED_SOURCE_SHA256 = (
    "sha256:e37cf1d3e638c3aed56230c9fa970c61e6c1ed8b4bd3024de259cb9847c31e48"
)
PARENT_R195_TYPED_SOURCE_RELATIVE_PATH = (
    "state/alakazam-terminal-expert-bootstrap-no-rtp-submit-r195.json"
)
PARENT_R195_TYPED_SOURCE_SCHEMA = (
    "poke_bot.alakazam_terminal_expert_bootstrap_no_rtp_submit_r195/v1"
)
LEARNER_R195_MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
H10_DIRECT_MATCHUP_TREE_SHA256 = (
    "sha256:da223c4903dd37511e5cb7656fe405bc0baac085be4f131faef136b7056c4588"
)
H10_DIRECT_MATCHUP_TREE_SIZE_BYTES = 2_509_756
HEAD_ROLE_MAP_PATH = ROOT / "state/alakazam-new-list-direct-r241-strategic-head-roles.json"
HEAD_ROLE_MAP_SHA256 = (
    "sha256:5b331159ab6e6bced77209f4d8b67a77ebebc78728c7a671384363dc0faaa356"
)
BASELINE_ADAPTER_ROSTER_PATH = ROOT / "state/matchup_adapter_roster.json"
BASELINE_ADAPTER_ROSTER_SHA256 = (
    "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc"
)
EXACT_WINDOW_START = "2026-07-22"
EXACT_WINDOW_END = "2026-08-10"
EXACT_WINDOW_DAYS = 20
R241_EXACT20_TRANSFER_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_exact20_transfer/v1"
)
R241_EXACT20_POINTER_SCHEMA = "poke_bot.pinned_expert_corpus/v1"
R241_EXACT20_SOURCE_READY_NAME = "SOURCE_LATEST20_SPECIALIST_CORPORA_READY.json"
R241_EXACT20_SOURCE_POINTER_NAME = "SOURCE_PROTECTED_EXPERT_CORPUS.json"
R241_EXACT20_MANIFEST_NAME = "manifest.json"
R241_EXACT20_ARCHIVE_COPY_NAME = "EXACT20_ARCHIVE_RECEIPT.json"
R241_EXACT20_TRANSFER_RECEIPT_NAME = "R241_EXACT20_CORPUS_TRANSFER_READY.json"
R241_EXACT20_INZI_TRANSFER_RECEIPT_COPY_NAME = (
    "INZI_R241_EXACT20_CORPUS_TRANSFER_READY.json"
)
R241_ELMO_METADATA_HANDOFF_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_r241_elmo_metadata_handoff/v1"
)
R241_EXACT20_INZI_TRANSFER_RECEIPT_PATH = (
    "/home/inzi/poke-bot-agent/outputs/pure_rl/"
    "alakazam_new_list_direct_policy_r241/runtime/expert/"
    "R241_EXACT20_CORPUS_TRANSFER_READY.json"
)
R241_EXACT20_INZI_TRANSFER_RECEIPT_SHA256 = (
    "sha256:b4b9cfdfcca444a8e09ee83db7836c2384075d8d5c4186c642c9423d72f888a3"
)
R241_EXACT20_INZI_TRANSFER_RECEIPT_SIZE_BYTES = 9_890
R241_EXACT20_SOURCE_READY_SHA256 = (
    "sha256:7da3523ede3b065e1335ded4630b810f4cbec3857fe78a7b55f53bf1e3ff8d37"
)
R241_EXACT20_SOURCE_READY_SIZE_BYTES = 8_891
R241_EXACT20_SOURCE_POINTER_SHA256 = (
    "sha256:bfb2f77cc17ba29b450bc9f81e7cca223b035feb64034ba89e6e9985573bacde"
)
R241_EXACT20_SOURCE_POINTER_SIZE_BYTES = 2_132
R241_EXACT20_MANIFEST_SHA256 = (
    "sha256:d23e38ba14e004fbaa74921eea94ce63f96a7ec953342eda69582f7ebcbbccd6"
)
R241_EXACT20_MANIFEST_SIZE_BYTES = 118_026
R241_EXACT20_ARCHIVE_SHA256 = (
    "sha256:09848f04a6c863a02c517fdcd5b7a61a139eceafd3348aa2a08705fd6e971a16"
)
R241_EXACT20_ARCHIVE_SIZE_BYTES = 15_298
R241_EXACT20_SHARD_BYTES = 5_471_162_566
R241_EXACT20_RECORDS = 26_704
R241_EXACT20_DECISIONS = 2_040_911
IMMUTABLE_ADAPTER_SLOT_PREFIX = 20
TERMINAL_REFRESH_BOUNDARY = 10
TERMINAL_REFRESH_EPOCHS = 5

# These fields were introduced after the immutable r195 checkpoint was
# serialized.  ``load_model_from_checkpoint`` supplies the same inert values
# only so a current ``ModelConfig`` can reconstruct the historical tensor
# inventory.  The receipt audit may erase *only* these absent, live-only
# fields, and only when they retain their exact inactive defaults.  This is a
# compatibility boundary, not a normalization of the checkpoint payload:
# the serialized config remains the identity hashed into the audit receipt.
R195_LIVE_ONLY_SUCCESSOR_MODEL_CONFIG_DEFAULTS: dict[str, object] = {
    "own_deck_ledger_enabled": False,
    "own_deck_ledger_runtime_enabled": False,
    "own_deck_ledger_width": 128,
    "own_deck_ledger_option_feature_dim": 8,
    "visible_tutor_completion_head_enabled": False,
    "terminal_conversion_head_enabled": False,
    "visible_tutor_completion_route_enabled": False,
    "visible_tutor_completion_route_runtime_enabled": False,
    "terminal_conversion_route_enabled": False,
    "terminal_conversion_route_runtime_enabled": False,
}

_SHA256_PREFIX = "sha256:"
_ENV_LOCK = threading.RLock()


class R241CheckpointReceiptError(RuntimeError):
    """A checkpoint-derived r241 receipt is incomplete, stale, or forged."""


@dataclasses.dataclass(frozen=True)
class FileIdentity:
    """Immutable file identity used by all derived receipts."""

    path: Path
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclasses.dataclass(frozen=True)
class R241AuditPolicy:
    """The fixed identities a production r241 auditor is allowed to accept."""

    parent_sha256: str = PARENT_R195_SHA256
    parent_size_bytes: int = PARENT_R195_SIZE_BYTES
    parent_typed_source_sha256: str = PARENT_R195_TYPED_SOURCE_SHA256
    learner_matchup_tree_sha256: str = LEARNER_R195_MATCHUP_TREE_SHA256
    h10_matchup_tree_sha256: str = H10_DIRECT_MATCHUP_TREE_SHA256
    h10_matchup_tree_size_bytes: int = H10_DIRECT_MATCHUP_TREE_SIZE_BYTES
    head_role_map_sha256: str = HEAD_ROLE_MAP_SHA256
    baseline_adapter_roster_sha256: str = BASELINE_ADAPTER_ROSTER_SHA256


DEFAULT_POLICY = R241AuditPolicy()


# Semantic names in the Fusion-v3 inventory intentionally differ from a few
# physical module names.  Checking both is what prevents a receipt from
# claiming a route whose source module was removed from the checkpoint.
_HEAD_MODULE_PREFIXES: dict[str, tuple[str, ...]] = {
    "value": ("value_head.",),
    "archetype": ("aux_head.",),
    "opponent_hand": ("opp_hand_head.",),
    "opponent_remainder": ("opp_remainder_head.",),
    "lethal_threat": ("lethal_threat_head.",),
    "prize_race": ("prize_race_head.",),
    "action_q": ("action_q_head.",),
    "action_type": ("action_type_head.",),
    "action_target": ("action_target_head.",),
    "action_resource": ("action_resource_head.",),
    "action_utility": ("action_utility_head.",),
    "tactical_outcomes": ("tactical_outcome_head.",),
    "opponent_response": ("opponent_response_head.",),
    "resource_forecast": ("resource_forecast_head.",),
    "game_phase": ("game_phase_head.",),
    "outcome_distribution": ("outcome_distribution_head.",),
    "remaining_turns": ("remaining_turns_head.",),
    "setup_board_outcome": ("setup_board_outcome_head.",),
    "combo_state": ("combo_state_head.",),
}


def canonical_json(value: object) -> bytes:
    """Return the deterministic encoding used for all receipt fingerprints."""

    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R241CheckpointReceiptError("receipt value is not canonical JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return _SHA256_PREFIX + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return _SHA256_PREFIX + digest.hexdigest()


def _regular_file(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise R241CheckpointReceiptError(
            f"{label} must be a regular, non-symlink file: {candidate}"
        )
    return candidate.resolve()


def _directory(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise R241CheckpointReceiptError(
            f"{label} must be a real directory: {candidate}"
        )
    return candidate.resolve()


def file_identity(
    path: Path | str,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> FileIdentity:
    """Hash a literal immutable file and enforce any supplied identity."""

    source = _regular_file(path, label=label)
    identity = FileIdentity(
        path=source,
        sha256=sha256_file(source),
        size_bytes=int(source.stat().st_size),
    )
    if expected_sha256 is not None and identity.sha256 != str(expected_sha256):
        raise R241CheckpointReceiptError(
            f"{label} checksum mismatch: expected={expected_sha256} actual={identity.sha256}"
        )
    if expected_size_bytes is not None and identity.size_bytes != int(expected_size_bytes):
        raise R241CheckpointReceiptError(
            f"{label} size mismatch: expected={expected_size_bytes} actual={identity.size_bytes}"
        )
    return identity


def _read_object(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label=label)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241CheckpointReceiptError(f"{label} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise R241CheckpointReceiptError(f"{label} must contain a JSON object")
    return source, payload


def _as_exact_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise R241CheckpointReceiptError(f"{label} must be an exact integer")
    return int(value)


def _sha256_text(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 71 or not text.startswith(_SHA256_PREFIX):
        raise R241CheckpointReceiptError(f"{label} is not a SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in text[7:]):
        raise R241CheckpointReceiptError(f"{label} is not a SHA-256 digest")
    return text


def _identity_from_row(row: object, *, label: str) -> FileIdentity:
    if not isinstance(row, Mapping):
        raise R241CheckpointReceiptError(f"{label} must be an identity object")
    path = _regular_file(str(row.get("path") or ""), label=f"{label} path")
    sha = _sha256_text(row.get("sha256") or row.get("digest"), label=f"{label} sha")
    size = _as_exact_int(row.get("size_bytes"), label=f"{label} size")
    return file_identity(
        path,
        label=label,
        expected_sha256=sha,
        expected_size_bytes=size,
    )


def _identity_matches(row: object, identity: FileIdentity, *, label: str) -> None:
    candidate = _identity_from_row(row, label=label)
    if candidate != identity:
        raise R241CheckpointReceiptError(f"{label} does not match the direct file identity")


def _head_contract(
    *,
    head_role_map_path: Path | str = HEAD_ROLE_MAP_PATH,
    policy: R241AuditPolicy = DEFAULT_POLICY,
) -> tuple[FileIdentity, tuple[str, ...], tuple[str, ...]]:
    source, payload = _read_object(head_role_map_path, label="r241 head-role map")
    identity = file_identity(
        source,
        label="r241 head-role map",
        expected_sha256=policy.head_role_map_sha256,
    )
    names_raw = payload.get("canonical_learned_decision_sources")
    if not isinstance(names_raw, list):
        raise R241CheckpointReceiptError("r241 head-role map omits head inventory")
    names = tuple(sorted(str(item).strip() for item in names_raw))
    heads = dict(payload.get("heads") or {})
    if (
        payload.get("schema") != "poke_bot.future_specialist_strategic_head_roles/v1"
        or payload.get("specialist_id") != "alakazam"
        or payload.get("decision_fusion_schema") != "poke_bot.causal_decision_fusion/v3"
        or len(names) != 18
        or len(set(names)) != 18
        or set(names) != set(heads)
        or "combo_state" in names
    ):
        raise R241CheckpointReceiptError("r241 head-role map is not the exact 18-head map")
    routes: list[str] = []
    for name in names:
        row = dict(heads[name] or {})
        route = str(row.get("route_id") or "").strip()
        if (
            name not in _HEAD_MODULE_PREFIXES
            or row.get("trainable") is not True
            or row.get("enters_decision_fusion") is not True
            or row.get("fusion_role") != "fused_input"
            or row.get("runtime_activation_requirement") != "receipt_backed_validation"
            or not route
        ):
            raise R241CheckpointReceiptError(
                f"r241 head-role map does not preserve {name!r}"
            )
        routes.append(route)
    if len(set(routes)) != 18:
        raise R241CheckpointReceiptError("r241 head-role map has duplicate route IDs")
    return identity, names, tuple(sorted(routes))


def _tensor_content_sha256(tensor: Any) -> str:
    """Hash an actual dense CPU tensor without normalizing its bytes."""

    import torch

    if not isinstance(tensor, torch.Tensor):
        raise R241CheckpointReceiptError("checkpoint state contains a non-tensor value")
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise R241CheckpointReceiptError("checkpoint tensor layout is unsupported")
    value = tensor.detach().cpu().contiguous()
    if value.is_floating_point() or value.is_complex():
        if not bool(torch.isfinite(value).all().item()):
            raise R241CheckpointReceiptError("checkpoint tensor contains non-finite values")
    # ``view(dtype)`` is valid for dense tensors but older torch releases
    # reject the dtype overload on a zero-dimensional scalar parameter.
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return sha256_bytes(raw)


def _tensor_inventory(state: Mapping[str, Any]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for name in sorted(str(key) for key in state):
        tensor = state[name]
        content_sha = _tensor_content_sha256(tensor)
        rows.append(
            {
                "name": name,
                "shape": [int(value) for value in tensor.shape],
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "numel": int(tensor.numel()),
                "content_sha256": content_sha,
            }
        )
    if not rows:
        raise R241CheckpointReceiptError("checkpoint has no model tensors")
    structural = [
        {key: row[key] for key in ("name", "shape", "dtype", "numel")}
        for row in rows
    ]
    return {
        "tensor_count": len(rows),
        "structural_sha256": sha256_bytes(canonical_json(structural)),
        "content_sha256": sha256_bytes(canonical_json(rows)),
        "tensors": rows,
    }


def _load_checkpoint_payload(
    checkpoint_path: Path | str,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> tuple[FileIdentity, dict[str, Any]]:
    """Load a known-hash tensor checkpoint through PyTorch's safe loader."""

    identity = file_identity(
        checkpoint_path,
        label=label,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    try:
        import torch

        # r195 deliberately carries RNG/provenance objects which PyTorch's
        # weights-only unpickler cannot decode.  Verify the complete immutable
        # byte identity *before* allowing the established checkpoint loader to
        # deserialize it, then verify it again below.  This is intentionally
        # not a generic untrusted-checkpoint loader.
        payload = torch.load(identity.path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - convert loader detail to fail-closed evidence
        raise R241CheckpointReceiptError(
            f"{label} cannot be safely decoded as a tensor checkpoint"
        ) from exc
    if not isinstance(payload, dict):
        raise R241CheckpointReceiptError(f"{label} payload is not a mapping")
    # A mutable source between the first hash and deserialization is never
    # acceptable, even when the caller thought the pathname was immutable.
    if sha256_file(identity.path) != identity.sha256:
        raise R241CheckpointReceiptError(f"{label} changed while being audited")
    return identity, dict(payload)


@contextlib.contextmanager
def _environment_scope(environment: Mapping[str, str] | None) -> Iterator[None]:
    """Temporarily expose the already validated direct-policy environment."""

    if environment is None:
        yield
        return
    with _ENV_LOCK:
        before = dict(os.environ)
        try:
            os.environ.clear()
            os.environ.update({str(key): str(value) for key, value in environment.items()})
            yield
        finally:
            os.environ.clear()
            os.environ.update(before)


def _assert_direct_environment(
    environment: Mapping[str, str] | None,
    *,
    official_cg_root: Path | str | None,
) -> dict[str, object]:
    """Bind audit execution to a real sealed r236, no-search environment."""

    if environment is None or official_cg_root is None:
        raise R241CheckpointReceiptError(
            "r241 receipt generation requires the actual sealed direct-policy environment"
        )
    from poke_bot.r241_direct_policy_runtime import (
        R241_OFFICIAL_LIBCG_RECEIPT_FILENAME,
        assert_direct_policy_environment,
        validate_sealed_official_libcg,
    )

    env = {str(key): str(value) for key, value in environment.items()}
    try:
        selected = assert_direct_policy_environment(env)
        supplied = _directory(official_cg_root, label="official r236 CG root")
        if selected != supplied:
            raise R241CheckpointReceiptError(
                "direct environment CG_LIB_PATH does not match the supplied r236 root"
            )
        resolved = validate_sealed_official_libcg(supplied, environment=env)
    except R241CheckpointReceiptError:
        raise
    except Exception as exc:  # noqa: BLE001 - preserve the direct-runtime boundary
        raise R241CheckpointReceiptError(
            f"sealed r241 direct-policy environment failed: {exc}"
        ) from exc
    receipt = file_identity(
        resolved / R241_OFFICIAL_LIBCG_RECEIPT_FILENAME,
        label="r241 official libcg local preflight",
    )
    library = file_identity(resolved / "cg" / "libcg.so", label="r241 official libcg")
    return {
        "cg_lib_path": str(resolved),
        "local_preflight": receipt.as_dict(),
        "linux_x86_64_library": library.as_dict(),
        "action_selector": "direct_policy_only",
        "mcts_calls": 0,
        "rtp_calls": 0,
        "search_calls": 0,
    }


def _assert_model_config(config: Mapping[str, Any]) -> None:
    required_true = (
        "expanded_heads_enabled",
        "setup_board_outcome_head_enabled",
        "combo_state_head_enabled",
        "decision_fusion_enabled",
        "decision_fusion_runtime_enabled",
        "decision_fusion_dedicated_routes_enabled",
        "decision_fusion_dedicated_routes_runtime_enabled",
        "decision_fusion_typed_output_centered_routes_enabled",
        "h10_capacity_enabled",
    )
    missing = [name for name in required_true if config.get(name) is not True]
    if missing:
        raise R241CheckpointReceiptError(
            "checkpoint model_config does not preserve H10 Fusion-v3: "
            + ", ".join(missing)
        )
    if (
        config.get("combo_state_route_enabled") is not False
        or config.get("matchup_adapters_enabled") is not False
        or str(config.get("matchup_adapter_format") or "")
        != "poke-bot-matchup-adapter-bank-v6"
    ):
        raise R241CheckpointReceiptError(
            "checkpoint model_config has an invalid combo or adapter activation state"
        )
    registry = config.get("matchup_adapter_registry")
    if not isinstance(registry, dict):
        raise R241CheckpointReceiptError("checkpoint omits the immutable V6 adapter registry")


def _assert_runtime_model_config_matches_serialized(
    *,
    serialized: Mapping[str, Any],
    live: Mapping[str, Any],
) -> None:
    """Allow only the documented inert post-r195 reconstruction backfills.

    ``serialized`` is the exact checkpoint payload and must never be amended
    for receipt identity.  A modern runtime has ten additional configuration
    fields, however.  If (and only if) one is absent from the checkpoint, its
    reconstructed value must be the exact inert default before it is removed
    from the live comparison.  All remaining fields, including any unknown
    runtime additions or a serialized successor field, stay under strict
    equality.
    """

    serialized_config = dict(serialized)
    normalized_live = dict(live)
    for field, expected_default in (
        R195_LIVE_ONLY_SUCCESSOR_MODEL_CONFIG_DEFAULTS.items()
    ):
        if field in serialized_config:
            # It is not a live-only backfill.  The strict comparison below
            # proves it was reconstructed from the immutable checkpoint value.
            continue
        observed = normalized_live.get(field, object())
        if (
            type(observed) is not type(expected_default)
            or observed != expected_default
        ):
            raise R241CheckpointReceiptError(
                "runtime reconstruction changed r195 successor-only "
                f"model_config default: {field}"
            )
        normalized_live.pop(field)
    if normalized_live != serialized_config:
        raise R241CheckpointReceiptError("runtime reconstruction changed model_config")


def _assert_main_optimizer_excludes_adapters(
    payload: Mapping[str, Any],
    *,
    model: Any,
    extra: Mapping[str, Any],
) -> dict[str, object]:
    """Verify the normal learner optimizer did not absorb the isolated bank."""

    optimizer = payload.get("optimizer_state_dict")
    if not isinstance(optimizer, Mapping):
        raise R241CheckpointReceiptError("checkpoint omits its ordinary optimizer state")
    groups = list(optimizer.get("param_groups") or [])
    if len(groups) != 1:
        raise R241CheckpointReceiptError("ordinary learner optimizer must have exactly one group")
    parameter_ids = list(groups[0].get("params") or [])
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in parameter_ids)
        or len(set(parameter_ids)) != len(parameter_ids)
    ):
        raise R241CheckpointReceiptError("ordinary optimizer parameter IDs are malformed")
    expected_non_adapter = [
        name
        for name, parameter in model.named_parameters()
        if not name.startswith("matchup_adapter_bank.") and parameter.requires_grad
    ]
    if parameter_ids != list(range(len(expected_non_adapter))):
        raise R241CheckpointReceiptError(
            "ordinary optimizer does not cover exactly the reconstructed non-adapter parameters"
        )
    if (
        extra.get("matchup_adapters_runtime_enabled") is not False
        or extra.get("matchup_adapter_training_enabled") is not False
        or extra.get("matchup_adapter_optimizer_included") is not False
    ):
        raise R241CheckpointReceiptError(
            "ordinary checkpoint falsely records a live or main-optimizer adapter bank"
        )
    state = dict(optimizer.get("state") or {})
    if not state:
        raise R241CheckpointReceiptError("ordinary optimizer has no continuation state")
    if not set(state).issubset(set(parameter_ids)):
        raise R241CheckpointReceiptError("ordinary optimizer state owns an unknown parameter")
    rows = [
        {"id": int(parameter_id), "has_state": int(parameter_id) in state}
        for parameter_id in parameter_ids
    ]
    return {
        "ordinary_optimizer_present": True,
        "ordinary_optimizer_parameter_count": len(parameter_ids),
        "ordinary_optimizer_state_count": len(state),
        "ordinary_optimizer_parameter_inventory_sha256": sha256_bytes(
            canonical_json(rows)
        ),
        "adapter_parameters_in_ordinary_optimizer": False,
    }


def _isolated_adapter_activation_identity(
    fit: Mapping[str, Any],
    *,
    adapter_training_activation: Path | str | None,
) -> tuple[FileIdentity, dict[str, object]]:
    """Bind the r195 EA38 declaration to an exact local immutable copy.

    The r195 checkpoint records the original Inzi path, which is historical
    provenance rather than a portable runtime location.  The actual path used
    for auditing may therefore be supplied by a host-local provenance copy,
    but its regular-file identity must remain the exact embedded EA38 digest
    and known size.  The raw checkpoint declaration is returned alongside the
    host-local identity so a receipt cannot erase that distinction.
    """

    declared_path = str(fit.get("activation_receipt") or "")
    declared_digest = _sha256_text(
        fit.get("activation_receipt_digest"),
        label="isolated adapter activation digest",
    )
    if declared_digest != PARENT_R195_ADAPTER_ACTIVATION_SHA256:
        raise R241CheckpointReceiptError(
            "checkpoint isolated adapter activation digest is not the exact EA38 proof"
        )
    selected_path = (
        adapter_training_activation
        if adapter_training_activation is not None
        else declared_path
    )
    identity = file_identity(
        selected_path,
        label="isolated adapter fit activation receipt",
        expected_sha256=declared_digest,
        expected_size_bytes=PARENT_R195_ADAPTER_ACTIVATION_SIZE_BYTES,
    )
    return identity, {
        "path": declared_path,
        "sha256": declared_digest,
        "expected_size_bytes": PARENT_R195_ADAPTER_ACTIVATION_SIZE_BYTES,
    }


def _assert_isolated_adapter_training(
    *,
    payload: Mapping[str, Any],
    model: Any,
    state: Mapping[str, Any],
    adapter_training_activation: Path | str | None = None,
) -> dict[str, object]:
    """Validate the trained V6 bank and its separately persisted optimizer."""

    extra = dict(payload.get("extra") or {})
    bank = getattr(model, "matchup_adapter_bank", None)
    if bank is None:
        raise R241CheckpointReceiptError("reconstructed model lacks a matchup adapter bank")
    config = extra.get("matchup_adapter_config")
    if not isinstance(config, dict) or config != bank.config_dict():
        raise R241CheckpointReceiptError("checkpoint adapter config disagrees with the live bank")
    model_config = dict(payload.get("model_config") or {})
    if model_config.get("matchup_adapter_registry") != config.get("slot_registry"):
        raise R241CheckpointReceiptError("model config and adapter checkpoint registry drifted")
    if (
        config.get("format") != "poke-bot-matchup-adapter-bank-v6"
        or _as_exact_int(config.get("slot_capacity"), label="adapter slot capacity") != 64
        or _as_exact_int(config.get("unknown_route"), label="adapter unknown route") != -1
    ):
        raise R241CheckpointReceiptError("adapter config is not the expected V6 bank")
    dormant = dict(extra.get("dormant_matchup_adapter_bank") or {})
    fit = dict(extra.get("dormant_matchup_adapter_fit") or {})
    if (
        dormant.get("schema") != "poke_bot.trained_dormant_matchup_adapter/v1"
        or dormant.get("runtime_enabled") is not False
        or dormant.get("training_enabled") is not False
        or dormant.get("optimizer_included") is not False
        or dormant.get("frozen") is not True
        or dormant.get("zero_output") is not False
        or dormant.get("adapter_config") != config
        or fit.get("schema") != "poke_bot.dormant_matchup_adapter_fit/v1"
        or fit.get("runtime_enabled") is not False
        or fit.get("base_frozen") is not True
        or fit.get("optimizer_scope") != "matchup_adapter_bank_only"
        or _as_exact_int(fit.get("epochs"), label="adapter fit epochs") <= 0
        or _as_exact_int(fit.get("steps"), label="adapter fit steps") <= 0
        or _as_exact_int(fit.get("rows"), label="adapter fit rows") <= 0
    ):
        raise R241CheckpointReceiptError("checkpoint lacks an auditable trained isolated adapter fit")
    route_decisions = dict(fit.get("route_decisions") or {})
    if (
        "alakazam" not in set(str(value) for value in fit.get("trained_archetype_ids") or [])
        or _as_exact_int(route_decisions.get("alakazam"), label="adapter Alakazam rows") <= 0
    ):
        raise R241CheckpointReceiptError("isolated adapter fit did not train the Alakazam route")
    activation_identity, declared_activation = _isolated_adapter_activation_identity(
        fit,
        adapter_training_activation=adapter_training_activation,
    )

    state_adapter = {
        str(name).removeprefix("matchup_adapter_bank."): value
        for name, value in state.items()
        if str(name).startswith("matchup_adapter_bank.")
    }
    live_adapter = bank.state_dict()
    if set(state_adapter) != set(live_adapter):
        raise R241CheckpointReceiptError("checkpoint adapter tensor inventory is incomplete")
    if len(state_adapter) != 256:
        raise R241CheckpointReceiptError("V6 adapter bank must retain all 64x4 tensors")
    for name, tensor in state_adapter.items():
        live = live_adapter[name]
        if tuple(tensor.shape) != tuple(live.shape) or str(tensor.dtype) != str(live.dtype):
            raise R241CheckpointReceiptError(f"adapter tensor shape/dtype drifted: {name}")
    output_tensors = [
        value
        for name, value in state_adapter.items()
        if name.endswith("up.weight") or name.endswith("up.bias")
    ]
    if not output_tensors or not any(int(value.count_nonzero().item()) > 0 for value in output_tensors):
        raise R241CheckpointReceiptError("trained adapter bank has no non-zero output route")

    isolated_optimizer = extra.get("dormant_matchup_adapter_optimizer_state")
    if not isinstance(isolated_optimizer, Mapping):
        raise R241CheckpointReceiptError("checkpoint omits isolated adapter optimizer state")
    groups = list(isolated_optimizer.get("param_groups") or [])
    if len(groups) != 1:
        raise R241CheckpointReceiptError("isolated adapter optimizer must have one group")
    parameters = list(groups[0].get("params") or [])
    adapter_parameters = list(bank.named_parameters())
    if parameters != list(range(len(adapter_parameters))) or len(parameters) != 256:
        raise R241CheckpointReceiptError("isolated adapter optimizer parameter coverage drifted")
    optimizer_state = dict(isolated_optimizer.get("state") or {})
    if not optimizer_state:
        raise R241CheckpointReceiptError("isolated adapter optimizer has no trained moments")
    if not set(optimizer_state).issubset(set(parameters)):
        raise R241CheckpointReceiptError("isolated adapter optimizer owns unknown parameters")
    rows: list[dict[str, object]] = []
    for index, (name, parameter) in enumerate(adapter_parameters):
        slot = optimizer_state.get(index)
        if slot is not None:
            if not isinstance(slot, Mapping):
                raise R241CheckpointReceiptError("adapter optimizer state row is malformed")
            for moment_name in ("exp_avg", "exp_avg_sq"):
                moment = slot.get(moment_name)
                if moment is None:
                    continue
                if tuple(moment.shape) != tuple(parameter.shape):
                    raise R241CheckpointReceiptError(
                        f"adapter optimizer moment shape drifted: {name}/{moment_name}"
                    )
                _tensor_content_sha256(moment)
        rows.append(
            {
                "optimizer_id": index,
                "parameter": name,
                "shape": [int(value) for value in parameter.shape],
                "has_state": slot is not None,
            }
        )
    adapter_tensor_inventory = _tensor_inventory(state_adapter)
    return {
        "checkpoint_dormant_state": {
            "runtime_enabled": False,
            "training_enabled": False,
            "ordinary_optimizer_included": False,
        },
        "adapter_config_sha256": sha256_bytes(canonical_json(config)),
        "adapter_tensor_inventory": adapter_tensor_inventory,
        "fit": {
            # The checkpoint's path is historical Inzi provenance.  A sealed
            # Elmo receipt instead binds the create-only host-local copy whose
            # bytes are checked against the embedded EA38 declaration above.
            "activation_receipt": activation_identity.as_dict(),
            "checkpoint_declared_activation_receipt": declared_activation,
            "epochs": int(fit["epochs"]),
            "steps": int(fit["steps"]),
            "rows": int(fit["rows"]),
            "trained_archetype_ids": [str(value) for value in fit["trained_archetype_ids"]],
            "alakazam_route_decisions": int(route_decisions["alakazam"]),
            "optimizer_scope": "matchup_adapter_bank_only",
        },
        "isolated_optimizer": {
            "parameter_count": len(parameters),
            "state_count": len(optimizer_state),
            "parameter_name_inventory_sha256": sha256_bytes(canonical_json(rows)),
            "state_key_inventory_sha256": sha256_bytes(
                canonical_json(sorted(int(key) for key in optimizer_state))
            ),
        },
    }


def _adapter_slot_tensor_names(slot: int) -> tuple[str, ...]:
    if slot < 0 or slot >= 64:
        raise R241CheckpointReceiptError("adapter slot is outside the V6 physical bank")
    prefix = f"matchup_adapter_bank.experts.{slot}."
    return tuple(prefix + suffix for suffix in ("down.weight", "down.bias", "up.weight", "up.bias"))


def _slot_registry_from_payload(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    extra = dict(payload.get("extra") or {})
    config = dict(extra.get("matchup_adapter_config") or {})
    registry = config.get("slot_registry")
    if not isinstance(registry, Mapping):
        raise R241CheckpointReceiptError(f"{label} does not embed a V6 slot registry")
    try:
        from poke_bot.matchup_adapters_v6 import load_slot_registry_dict, registry_digest

        normalized = load_slot_registry_dict(dict(registry))
        if config.get("slot_registry_digest") != registry_digest(normalized):
            raise R241CheckpointReceiptError(f"{label} slot registry digest drifted")
    except R241CheckpointReceiptError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise R241CheckpointReceiptError(f"{label} slot registry is invalid") from exc
    return dict(normalized)


def _baseline_adapter_roster(
    *,
    path: Path | str = BASELINE_ADAPTER_ROSTER_PATH,
    policy: R241AuditPolicy = DEFAULT_POLICY,
) -> tuple[FileIdentity, dict[str, Any]]:
    """Load the r247 pre-PTCGReplay roster that owns slots 0 through 19."""

    source, payload = _read_object(path, label="r241 baseline adapter roster")
    identity = file_identity(
        source,
        label="r241 baseline adapter roster",
        expected_sha256=policy.baseline_adapter_roster_sha256,
    )
    try:
        from poke_bot.matchup_adapters_v6 import load_slot_registry_dict

        registry = load_slot_registry_dict(payload)
    except Exception as exc:  # noqa: BLE001
        raise R241CheckpointReceiptError("r241 baseline adapter roster is not valid V6") from exc
    slots = list(registry.get("slots") or [])
    if (
        len(slots) != 64
        or any(dict(slots[index] or {}).get("slot") != index for index in range(64))
        or any(
            dict(slots[index] or {}).get("status") == "unused"
            or not str(dict(slots[index] or {}).get("archetype_id") or "")
            for index in range(IMMUTABLE_ADAPTER_SLOT_PREFIX)
        )
        or any(
            dict(slots[index] or {}).get("status") != "unused"
            or dict(slots[index] or {}).get("archetype_id") is not None
            for index in range(IMMUTABLE_ADAPTER_SLOT_PREFIX, 64)
        )
    ):
        raise R241CheckpointReceiptError(
            "r241 baseline roster must own exact immutable slots 0 through 19"
        )
    return identity, dict(registry)


def _slot_proof_identity(
    row: object,
    *,
    label: str,
    archetype_id: str,
    slot: int,
    kind: str,
) -> FileIdentity:
    """Validate a concrete data/fit/activation receipt for a newly trained slot."""

    identity = _identity_from_row(row, label=label)
    _path, payload = _read_object(identity.path, label=label)
    if (
        str(payload.get("archetype_id") or "") != archetype_id
        or _as_exact_int(payload.get("slot"), label=f"{label} slot") != slot
    ):
        raise R241CheckpointReceiptError(f"{label} does not bind its adapter slot")
    status = str(payload.get("status") or "")
    allowed = {
        "data": {"ready", "accepted", "sealed"},
        "fit": {"completed", "passed", "trained"},
        "activation": {"active", "passed"},
    }[kind]
    if status not in allowed:
        raise R241CheckpointReceiptError(f"{label} status cannot prove {kind}")
    return identity


def audit_append_only_adapter_slot_migration(
    *,
    parent_checkpoint: Path | str,
    candidate_checkpoint: Path | str,
    parent_audit: Mapping[str, Any],
    candidate_audit: Mapping[str, Any],
) -> dict[str, object]:
    """Derive append-only V6 migration evidence from checkpoint tensor bytes.

    The PTCGReplay roster can grow without altering the proven r195 routes.
    Existing allocated slots and their tensors are byte-immutable.  A newly
    allocated former-unused slot is admissible only as an exact-zero dormant
    route, or as an explicitly trained/activated route with independent
    data/fit/activation receipt identities.  No claim in the migration record
    substitutes for these direct comparisons.
    """

    parent_identity, parent_payload = _load_checkpoint_payload(
        parent_checkpoint,
        label="r241 adapter-migration parent checkpoint",
        expected_sha256=str(dict(parent_audit.get("checkpoint") or {}).get("sha256") or ""),
        expected_size_bytes=_as_exact_int(
            dict(parent_audit.get("checkpoint") or {}).get("size_bytes"),
            label="adapter-migration parent checkpoint size",
        ),
    )
    candidate_identity, candidate_payload = _load_checkpoint_payload(
        candidate_checkpoint,
        label="r241 adapter-migration candidate checkpoint",
        expected_sha256=str(dict(candidate_audit.get("checkpoint") or {}).get("sha256") or ""),
        expected_size_bytes=_as_exact_int(
            dict(candidate_audit.get("checkpoint") or {}).get("size_bytes"),
            label="adapter-migration candidate checkpoint size",
        ),
    )
    parent_state = dict(parent_payload.get("model_state_dict") or {})
    candidate_state = dict(candidate_payload.get("model_state_dict") or {})
    parent_registry = _slot_registry_from_payload(parent_payload, label="adapter-migration parent")
    candidate_registry = _slot_registry_from_payload(candidate_payload, label="adapter-migration candidate")
    parent_slots = list(parent_registry.get("slots") or [])
    candidate_slots = list(candidate_registry.get("slots") or [])
    if len(parent_slots) != 64 or len(candidate_slots) != 64:
        raise R241CheckpointReceiptError("V6 adapter migration lacks all 64 physical slots")

    retained_rows: list[dict[str, object]] = []
    added_slots: list[dict[str, object]] = []
    for slot in range(64):
        before = dict(parent_slots[slot] or {})
        after = dict(candidate_slots[slot] or {})
        if before.get("slot") != slot or after.get("slot") != slot:
            raise R241CheckpointReceiptError("adapter migration reindexed a physical slot")
        names = _adapter_slot_tensor_names(slot)
        for name in names:
            if name not in parent_state or name not in candidate_state:
                raise R241CheckpointReceiptError(f"adapter migration omits tensor {name}")
        before_status = str(before.get("status") or "")
        if slot < IMMUTABLE_ADAPTER_SLOT_PREFIX or before_status != "unused":
            # Existing identities, logical attributes, and weights are all
            # immutable.  A route activation must use package/tree state, not
            # rewrite the dormant r195 checkpoint row.
            if after != before:
                raise R241CheckpointReceiptError(
                    f"adapter migration changed existing slot metadata: {slot}"
                )
            hashes = {
                name: _tensor_content_sha256(parent_state[name]) for name in names
            }
            if any(_tensor_content_sha256(candidate_state[name]) != hashes[name] for name in names):
                raise R241CheckpointReceiptError(
                    f"adapter migration changed existing slot tensors: {slot}"
                )
            retained_rows.append(
                {
                    "slot": slot,
                    "archetype_id": str(before.get("archetype_id") or ""),
                    "tensor_inventory_sha256": sha256_bytes(canonical_json(hashes)),
                }
            )
            continue
        if after == before:
            # Unused rows must remain exact zero if they stay unused.
            if any(int(candidate_state[name].count_nonzero().item()) != 0 for name in names):
                raise R241CheckpointReceiptError(
                    f"unused adapter slot has non-zero tensor state: {slot}"
                )
            continue
        if (
            slot < IMMUTABLE_ADAPTER_SLOT_PREFIX
            or before.get("archetype_id") is not None
            or after.get("status") not in {"dormant", "active"}
        ):
            raise R241CheckpointReceiptError("adapter migration is not append-only")
        archetype_id = str(after.get("archetype_id") or "")
        if not archetype_id:
            raise R241CheckpointReceiptError("new adapter slot has no archetype identity")
        added_slots.append(
            {
                "slot": slot,
                "archetype_id": archetype_id,
                "status": str(after.get("status")),
                "tensor_inventory_sha256": sha256_bytes(
                    canonical_json(
                        {
                            name: _tensor_content_sha256(candidate_state[name])
                            for name in names
                        }
                    )
                ),
                "all_tensors_zero": all(
                    int(candidate_state[name].count_nonzero().item()) == 0
                    for name in names
                ),
                "output_tensors_nonzero": any(
                    int(candidate_state[name].count_nonzero().item()) > 0
                    for name in names
                    if name.endswith("up.weight") or name.endswith("up.bias")
                ),
            }
        )

    parent_aliases = dict(parent_registry.get("logical_aliases") or {})
    candidate_aliases = dict(candidate_registry.get("logical_aliases") or {})
    if any(candidate_aliases.get(key) != value for key, value in parent_aliases.items()):
        raise R241CheckpointReceiptError("adapter migration rewrote an existing logical alias")
    extra = dict(candidate_payload.get("extra") or {})
    record = extra.get("r241_adapter_slot_migration")
    if not added_slots:
        if record not in (None, {}):
            raise R241CheckpointReceiptError("adapter migration record claims slots that were not added")
        mode = "no_slot_change"
        proof_rows: list[dict[str, object]] = []
    else:
        if not isinstance(record, Mapping):
            raise R241CheckpointReceiptError("new adapter slots lack an append-only migration record")
        migration = dict(record)
        if (
            migration.get("schema") != R241_ADAPTER_SLOT_MIGRATION_SCHEMA
            or migration.get("parent_checkpoint_sha256") != parent_identity.sha256
            or migration.get("parent_adapter_tensor_inventory_sha256")
            != dict(parent_audit.get("matchup_adapter") or {})
            .get("adapter_tensor_inventory", {})
            .get("content_sha256")
        ):
            raise R241CheckpointReceiptError("adapter migration record does not bind the immutable parent")
        declared = {
            _as_exact_int(dict(item).get("slot"), label="new adapter slot"): dict(item)
            for item in list(migration.get("new_slots") or [])
            if isinstance(item, Mapping)
        }
        if set(declared) != {int(row["slot"]) for row in added_slots}:
            raise R241CheckpointReceiptError("adapter migration record does not enumerate new slots exactly")
        proof_rows = []
        for observed in added_slots:
            slot = int(observed["slot"])
            row = declared[slot]
            if row.get("archetype_id") != observed["archetype_id"]:
                raise R241CheckpointReceiptError("adapter migration new-slot identity drifted")
            mode = str(row.get("mode") or "")
            if mode == "zero_dormant":
                if observed["status"] != "dormant" or observed["all_tensors_zero"] is not True:
                    raise R241CheckpointReceiptError("new dormant adapter slot is not exact zero")
                proof_rows.append({"slot": slot, "mode": mode})
            elif mode == "trained_activated":
                if observed["status"] != "active" or observed["output_tensors_nonzero"] is not True:
                    raise R241CheckpointReceiptError("new active adapter slot has no trained output")
                data = _slot_proof_identity(
                    row.get("data_receipt"),
                    label="new adapter slot data receipt",
                    archetype_id=str(observed["archetype_id"]),
                    slot=slot,
                    kind="data",
                )
                fit = _slot_proof_identity(
                    row.get("fit_receipt"),
                    label="new adapter slot fit receipt",
                    archetype_id=str(observed["archetype_id"]),
                    slot=slot,
                    kind="fit",
                )
                activation = _slot_proof_identity(
                    row.get("activation_receipt"),
                    label="new adapter slot activation receipt",
                    archetype_id=str(observed["archetype_id"]),
                    slot=slot,
                    kind="activation",
                )
                proof_rows.append(
                    {
                        "slot": slot,
                        "mode": mode,
                        "data_receipt": data.as_dict(),
                        "fit_receipt": fit.as_dict(),
                        "activation_receipt": activation.as_dict(),
                    }
                )
            else:
                raise R241CheckpointReceiptError("new adapter slot has an unsupported migration mode")
        mode = "append_only_slot_addition"
    return {
        "schema": R241_ADAPTER_SLOT_MIGRATION_SCHEMA,
        "status": mode,
        "parent_checkpoint": parent_identity.as_dict(),
        "candidate_checkpoint": candidate_identity.as_dict(),
        "parent_slot_registry_sha256": sha256_bytes(canonical_json(parent_registry)),
        "candidate_slot_registry_sha256": sha256_bytes(canonical_json(candidate_registry)),
        "retained_slot_count": len(retained_rows),
        "retained_slot_tensor_inventory_sha256": sha256_bytes(canonical_json(retained_rows)),
        "existing_slots_byte_immutable": True,
        "new_slots": added_slots,
        "new_slot_proofs": proof_rows,
    }


def _assert_fusion_and_head_inventory(
    *,
    model: Any,
    state: Mapping[str, Any],
    expected_heads: tuple[str, ...],
) -> dict[str, object]:
    """Inspect the reconstructed physical 19-head / 18-route runtime."""

    fusion_inventory_fn = getattr(model, "decision_fusion_inventory", None)
    expanded_inventory_fn = getattr(model, "expanded_head_inventory", None)
    if not callable(fusion_inventory_fn) or not callable(expanded_inventory_fn):
        raise R241CheckpointReceiptError("reconstructed model has no Fusion-v3 inventory")
    fusion = dict(fusion_inventory_fn() or {})
    routes = dict(fusion.get("dedicated_routes") or {})
    physical_heads = tuple(sorted(str(value) for value in fusion.get("required_heads") or []))
    active_heads = tuple(sorted(str(value) for value in fusion.get("active_required_heads") or []))
    physical_routes = tuple(sorted(str(value) for value in routes.get("route_names") or []))
    active_routes = tuple(sorted(str(value) for value in routes.get("active_route_names") or []))
    disabled_routes = tuple(sorted(str(value) for value in routes.get("disabled_route_names") or []))
    all_expected = tuple(sorted((*expected_heads, "combo_state")))
    if (
        fusion.get("schema") != "poke_bot.causal_decision_fusion/v3"
        or fusion.get("enabled") is not True
        or fusion.get("runtime_enabled") is not True
        or routes.get("enabled") is not True
        or routes.get("runtime_enabled") is not True
        or routes.get("typed_output_centered") is not True
        or routes.get("positive_bounded_reliability") is not True
        or routes.get("combo_state_route_enabled") is not False
        or _as_exact_int(routes.get("route_count"), label="fusion physical route count") != 19
        or physical_heads != all_expected
        or physical_routes != all_expected
        or active_heads != expected_heads
        or active_routes != expected_heads
        or disabled_routes != ("combo_state",)
    ):
        raise R241CheckpointReceiptError(
            "checkpoint does not retain the exact 19 physical / 18 active Fusion-v3 inventory"
        )
    expanded = dict(expanded_inventory_fn() or {})
    modules = dict(expanded.get("modules") or {})
    if (
        expanded.get("enabled") is not True
        or "combo_state_head" not in modules
        or "setup_board_outcome_head" not in modules
        or "combo_state_head" not in set(expanded.get("runtime_disabled_heads") or [])
        or "combo_state_head" in set(expanded.get("runtime_enabled_heads") or [])
    ):
        raise R241CheckpointReceiptError("checkpoint lacks the disabled-but-resident combo head")

    named_parameters = dict(model.named_parameters())
    head_rows: list[dict[str, object]] = []
    for name in expected_heads:
        prefixes = _HEAD_MODULE_PREFIXES[name]
        source_parameters = [
            parameter_name
            for parameter_name in named_parameters
            if any(parameter_name.startswith(prefix) for prefix in prefixes)
        ]
        route_prefix = f"decision_fusion.dedicated_routes.{name}."
        route_parameters = [
            parameter_name
            for parameter_name in named_parameters
            if parameter_name.startswith(route_prefix)
        ]
        if not source_parameters or not route_parameters:
            raise R241CheckpointReceiptError(f"r241 non-combo head/route is missing: {name}")
        if any(not named_parameters[key].requires_grad for key in source_parameters + route_parameters):
            raise R241CheckpointReceiptError(f"r241 non-combo head/route is not trainable: {name}")
        if not any(str(key).startswith(route_prefix) for key in state):
            raise R241CheckpointReceiptError(f"r241 checkpoint omits route tensors: {name}")
        head_rows.append(
            {
                "head": name,
                "source_parameter_names": sorted(source_parameters),
                "route_parameter_names": sorted(route_parameters),
                "trainable": True,
                "runtime_route_enabled": True,
            }
        )
    combo_prefix = "decision_fusion.dedicated_routes.combo_state."
    if not any(str(key).startswith(combo_prefix) for key in state):
        raise R241CheckpointReceiptError("r241 checkpoint omitted physical combo route tensors")
    if not any(str(key).startswith("combo_state_head.") for key in state):
        raise R241CheckpointReceiptError("r241 checkpoint omitted physical combo head tensors")
    return {
        "architecture_present_head_count": 19,
        "non_combo_head_count": 18,
        "non_combo_route_count": 18,
        "physical_head_names": list(all_expected),
        "active_non_combo_head_names": list(expected_heads),
        "active_non_combo_route_names": list(expected_heads),
        "physical_head_names_sha256": sha256_bytes(canonical_json(list(all_expected))),
        "active_non_combo_head_names_sha256": sha256_bytes(
            canonical_json(list(expected_heads))
        ),
        "active_non_combo_route_names_sha256": sha256_bytes(
            canonical_json(list(expected_heads))
        ),
        "head_route_parameter_inventory_sha256": sha256_bytes(
            canonical_json(head_rows)
        ),
        "every_non_combo_head_trainable": True,
        "every_non_combo_fusion_route_enabled": True,
        "combo_state": {
            "head_present": True,
            "physical_route_present": True,
            "loss_weight": 0.0,
            "route_enabled": False,
        },
    }


def _assert_combo_loss_off(payload: Mapping[str, Any]) -> None:
    extra = dict(payload.get("extra") or {})
    rehearsal = dict(extra.get("expert_rehearsal") or {})
    weights = dict(rehearsal.get("loss_weights") or {})
    if float(weights.get("combo_state", -1.0)) != 0.0:
        raise R241CheckpointReceiptError("checkpoint rehearsal did not keep combo-state loss off")
    strategic = dict(extra.get("current_deck_guide_training") or {})
    contract = dict(strategic.get("contract") or {})
    if contract and float(contract.get("combo_state_base_loss_weight", -1.0)) != 0.0:
        raise R241CheckpointReceiptError("checkpoint guide contract re-enabled combo-state loss")


def _reconstruct_runtime_smoke(
    *,
    checkpoint_path: Path,
    expected_state: Mapping[str, Any],
    environment: Mapping[str, str] | None,
) -> tuple[Any, dict[str, object]]:
    """Load the exact tensor checkpoint and execute one in-memory adapter route."""

    try:
        import torch
        from poke_bot.train import load_model_from_checkpoint
    except Exception as exc:  # noqa: BLE001
        raise R241CheckpointReceiptError("r241 runtime smoke cannot import model loader") from exc
    with _environment_scope(environment):
        try:
            model = load_model_from_checkpoint(checkpoint_path, device=torch.device("cpu"))
        except Exception as exc:  # noqa: BLE001
            raise R241CheckpointReceiptError(
                "checkpoint failed direct model reconstruction"
            ) from exc
    live_state = model.state_dict()
    if set(live_state) != set(expected_state):
        raise R241CheckpointReceiptError("runtime reconstruction changed tensor names")
    for name, tensor in expected_state.items():
        if tuple(live_state[name].shape) != tuple(tensor.shape) or not bool(
            torch.equal(live_state[name].detach().cpu(), tensor.detach().cpu())
        ):
            raise R241CheckpointReceiptError(
                f"runtime reconstruction changed tensor bytes: {name}"
            )
    bank = getattr(model, "matchup_adapter_bank", None)
    if bank is None:
        raise R241CheckpointReceiptError("runtime reconstruction lost adapter bank")
    registry = getattr(bank, "registry", None)
    if not isinstance(registry, dict):
        raise R241CheckpointReceiptError("runtime reconstruction lost V6 adapter registry")
    try:
        from poke_bot.matchup_adapters_v6 import route_for_archetype

        route = int(route_for_archetype("alakazam", registry=registry))
    except Exception as exc:  # noqa: BLE001
        raise R241CheckpointReceiptError("runtime reconstruction cannot resolve Alakazam route") from exc
    if route < 0:
        raise R241CheckpointReceiptError("runtime reconstruction has no Alakazam adapter route")
    d_model = int(getattr(model, "d_model", 0))
    if d_model <= 0:
        raise R241CheckpointReceiptError("runtime reconstruction has no model width")
    state = torch.linspace(-1.0, 1.0, d_model, dtype=torch.float32).reshape(1, d_model)
    model.eval()
    previous_enabled = bool(getattr(bank, "enabled", False))
    try:
        bank.enabled = True
        with torch.no_grad():
            adapted = model.matchup_policy_value_state(state, [route], enabled=True)
    finally:
        bank.enabled = previous_enabled
    if (
        tuple(adapted.shape) != tuple(state.shape)
        or not bool(torch.isfinite(adapted).all().item())
        or bool(torch.equal(adapted, state))
    ):
        raise R241CheckpointReceiptError(
            "runtime adapter smoke did not execute a non-zero Alakazam route"
        )
    return model, {
        "method": "checkpoint_reconstruction_and_direct_v6_adapter_forward/v1",
        "model_reconstructed": True,
        "adapter_runtime_enabled_for_smoke": True,
        "adapter_route": route,
        "adapter_output_changed": True,
        "adapter_output_finite": True,
        "action_selector": "direct_policy_only",
        "mcts_calls": 0,
        "rtp_calls": 0,
        "search_calls": 0,
    }


def audit_checkpoint(
    checkpoint_path: Path | str,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    head_role_map_path: Path | str = HEAD_ROLE_MAP_PATH,
    policy: R241AuditPolicy = DEFAULT_POLICY,
    environment: Mapping[str, str] | None = None,
    adapter_training_activation: Path | str | None = None,
) -> dict[str, object]:
    """Derive the full model/head/optimizer evidence from one checkpoint.

    ``environment`` is optional for the pure checkpoint inspection so tests
    and offline checksum audits do not need a host libcg root.  Production
    receipt generators always call :func:`_assert_direct_environment` first
    and pass its exact process environment here.  ``adapter_training_activation``
    is an optional host-local EA38 provenance copy; it cannot change the
    checkpoint declaration and must match its fixed digest and size exactly.
    """

    checkpoint_identity, payload = _load_checkpoint_payload(
        checkpoint_path,
        label="r241 checkpoint",
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    model_config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(model_config, Mapping) or not isinstance(state, Mapping):
        raise R241CheckpointReceiptError("checkpoint lacks model_config or model_state_dict")
    if str(payload.get("archetype_id") or "").casefold() != "alakazam":
        raise R241CheckpointReceiptError("checkpoint archetype is not Alakazam")
    _assert_model_config(model_config)
    _assert_combo_loss_off(payload)
    head_role_map, expected_heads, route_ids = _head_contract(
        head_role_map_path=head_role_map_path,
        policy=policy,
    )
    model, smoke = _reconstruct_runtime_smoke(
        checkpoint_path=checkpoint_identity.path,
        expected_state=state,
        environment=environment,
    )
    live_config = dataclasses.asdict(model.cfg)
    _assert_runtime_model_config_matches_serialized(
        serialized=model_config,
        live=live_config,
    )
    heads = _assert_fusion_and_head_inventory(
        model=model,
        state=state,
        expected_heads=expected_heads,
    )
    heads["active_non_combo_fusion_route_ids"] = list(route_ids)
    heads["active_non_combo_fusion_route_ids_sha256"] = sha256_bytes(
        canonical_json(list(route_ids))
    )
    adapter = _assert_isolated_adapter_training(
        payload=payload,
        model=model,
        state=state,
        adapter_training_activation=adapter_training_activation,
    )
    adapter_registry = _slot_registry_from_payload(payload, label="checkpoint adapter")
    adapter["slot_registry_sha256"] = sha256_bytes(canonical_json(adapter_registry))
    adapter["immutable_slot_prefix"] = IMMUTABLE_ADAPTER_SLOT_PREFIX
    # These values are derived from the serialized dormant bank, its isolated
    # continuation optimizer, and the live adapter-on forward above.  Keeping
    # checkpoint state separate from external runtime state is deliberate:
    # the ordinary r195 checkpoint remains dormant while r241's direct
    # package/collector enables the already-trained bank at runtime.
    adapter["activation_provenance"] = {
        "matchup_adapter_bank_preserved": True,
        "matchup_adapter_training_enabled": True,
        "matchup_adapter_runtime_enabled": True,
        "matchup_adapter_checkpoint_runtime_enabled": False,
        "matchup_adapter_checkpoint_training_enabled": False,
        "matchup_adapter_checkpoint_main_optimizer_included": False,
        "matchup_adapter_isolated_bank_only_optimizer": True,
        "matchup_adapter_isolated_fit_continuation_required": True,
        "matchup_adapter_external_collection_runtime_enabled": True,
        "matchup_adapter_external_terminal_runtime_enabled": True,
    }
    ordinary_optimizer = _assert_main_optimizer_excludes_adapters(
        payload,
        model=model,
        extra=dict(payload.get("extra") or {}),
    )
    model_config_hash = sha256_bytes(canonical_json(dict(model_config)))
    inventory = _tensor_inventory(state)
    result = {
        "schema": R241_CHECKPOINT_AUDIT_SCHEMA,
        "checkpoint": checkpoint_identity.as_dict(),
        "archetype_id": "alakazam",
        "model_config_sha256": model_config_hash,
        "model_config": {
            "expanded_heads_enabled": True,
            "setup_board_outcome_head_enabled": True,
            "combo_state_head_enabled": True,
            "combo_state_route_enabled": False,
            "decision_fusion_enabled": True,
            "decision_fusion_runtime_enabled": True,
            "decision_fusion_dedicated_routes_enabled": True,
            "decision_fusion_dedicated_routes_runtime_enabled": True,
            "decision_fusion_typed_output_centered_routes_enabled": True,
            "h10_capacity_enabled": True,
            "matchup_adapters_checkpoint_dormant": True,
        },
        "sorted_tensor_inventory": inventory,
        "head_role_map": head_role_map.as_dict(),
        "head_role_map_route_ids_sha256": sha256_bytes(canonical_json(list(route_ids))),
        "heads": heads,
        "ordinary_optimizer": ordinary_optimizer,
        "matchup_adapter": adapter,
        "runtime_smoke": smoke,
    }
    result["audit_fingerprint_sha256"] = sha256_bytes(canonical_json(result))
    return result


def _validate_tree(
    path: Path | str,
    *,
    label: str,
    expected_sha256: str,
    expected_size_bytes: int | None = None,
) -> FileIdentity:
    source, payload = _read_object(path, label=label)
    identity = file_identity(
        source,
        label=label,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    runtime = dict(payload.get("runtime_contract") or {})
    if (
        payload.get("runtime_enabled") is not True
        or runtime.get("one_route_per_decision") is not True
        or runtime.get("unknown_route_exact_bypass") is not True
    ):
        raise R241CheckpointReceiptError(f"{label} is not an enabled direct adapter tree")
    return identity


def _validate_expert_window(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
) -> FileIdentity:
    source, payload = _read_object(path, label="r241 exact expert-window receipt")
    identity = file_identity(
        source,
        label="r241 exact expert-window receipt",
        expected_sha256=expected_sha256,
    )
    nested_window = payload.get("window")
    if nested_window is not None and not isinstance(nested_window, Mapping):
        raise R241CheckpointReceiptError("expert-window nested window is not an object")
    window = dict(nested_window or {})
    top_level_window = {
        "start": payload.get("window_start"),
        "end": payload.get("window_end"),
        "days": payload.get("days"),
    }
    has_nested_window = bool(window)
    has_top_level_window = any(value is not None for value in top_level_window.values())
    nested_is_exact = (
        window.get("start") == EXACT_WINDOW_START
        and window.get("end") == EXACT_WINDOW_END
        and _as_exact_int(window.get("days"), label="expert nested window days")
        == EXACT_WINDOW_DAYS
    ) if has_nested_window else False
    top_level_is_exact = (
        top_level_window["start"] == EXACT_WINDOW_START
        and top_level_window["end"] == EXACT_WINDOW_END
        and _as_exact_int(top_level_window["days"], label="expert window days")
        == EXACT_WINDOW_DAYS
    ) if has_top_level_window else False
    if (
        payload.get("schema") != "poke_bot.expert_latest20_receipt/v1"
        or payload.get("status") != "ready"
        or not (nested_is_exact or top_level_is_exact)
        # If a receipt carries both historical representations, neither is
        # allowed to contradict the immutable Jul22-Aug10 identity.
        or (has_nested_window and not nested_is_exact)
        or (has_top_level_window and not top_level_is_exact)
    ):
        raise R241CheckpointReceiptError("expert-window receipt is not the exact r241 window")
    return identity


def _r241_exact20_local_identity(
    root: Path,
    row: object,
    *,
    expected_path: str,
    label: str,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> FileIdentity:
    """Rehash a named non-symlink member and its receipt identity row."""

    if not isinstance(row, Mapping):
        raise R241CheckpointReceiptError(f"{label} must be an identity object")
    if str(row.get("path") or "") != expected_path:
        raise R241CheckpointReceiptError(f"{label} path is not the sealed r241 member")
    member = Path(expected_path)
    if member.is_absolute() or ".." in member.parts or member.name != expected_path:
        raise R241CheckpointReceiptError(f"{label} path is unsafe")
    candidate = (root / member).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise R241CheckpointReceiptError(f"{label} escapes the protected corpus") from exc
    identity = file_identity(
        candidate,
        label=label,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    if (
        _sha256_text(row.get("sha256"), label=f"{label} sha") != identity.sha256
        or _as_exact_int(row.get("size_bytes"), label=f"{label} size")
        != identity.size_bytes
    ):
        raise R241CheckpointReceiptError(f"{label} identity drifted")
    return identity


def _r241_exact20_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R241CheckpointReceiptError(f"{label} must be an object")
    return dict(value)


def _validate_r241_elmo_metadata_handoff(
    root: Path,
    *,
    receipt: Mapping[str, Any],
    receipt_source: Mapping[str, Any],
    corpus: Mapping[str, Any],
) -> None:
    """Validate the small Elmo projection when a transfer declares one.

    The ordinary Inzi handoff retains all twenty local feature shards.  Elmo
    intentionally projects only immutable receipts and an exact copy of the
    already-sealed Inzi transfer receipt, so this branch must prove that it is
    a receipt-only root rather than accidentally treating missing shards as a
    normal complete corpus.
    """

    declared = receipt.get("r241_elmo_metadata_handoff")
    if declared is None:
        return
    projection = _r241_exact20_mapping(
        declared, label="r241 Elmo metadata handoff declaration"
    )
    expected_members = {
        R241_EXACT20_SOURCE_READY_NAME,
        R241_EXACT20_SOURCE_POINTER_NAME,
        R241_EXACT20_MANIFEST_NAME,
        R241_EXACT20_ARCHIVE_COPY_NAME,
        R241_EXACT20_INZI_TRANSFER_RECEIPT_COPY_NAME,
        R241_EXACT20_TRANSFER_RECEIPT_NAME,
        "PROTECTED_EXPERT_CORPUS.json",
    }
    members = tuple(root.iterdir())
    if {member.name for member in members} != expected_members:
        raise R241CheckpointReceiptError(
            "r241 Elmo metadata handoff must contain exactly seven receipt files"
        )
    for member in members:
        _regular_file(member, label="r241 Elmo metadata handoff member")

    origin = _r241_exact20_mapping(
        projection.get("inzi_transfer_receipt"),
        label="r241 Elmo metadata Inzi transfer provenance",
    )
    if (
        projection.get("schema") != R241_ELMO_METADATA_HANDOFF_SCHEMA
        or projection.get("metadata_only") is not True
        or projection.get("feature_shards_copied") is not False
        or projection.get("feature_sidecars_copied") is not False
        or projection.get("source_archive_reused_without_mutation") is not True
        or origin.get("host") != "inzi"
        or origin.get("remote_path") != R241_EXACT20_INZI_TRANSFER_RECEIPT_PATH
    ):
        raise R241CheckpointReceiptError(
            "r241 Elmo metadata handoff declaration is not the sealed receipt-only projection"
        )
    inzi_receipt_identity = _r241_exact20_local_identity(
        root,
        origin.get("local_copy"),
        expected_path=R241_EXACT20_INZI_TRANSFER_RECEIPT_COPY_NAME,
        label="r241 copied sealed Inzi transfer receipt",
        expected_sha256=R241_EXACT20_INZI_TRANSFER_RECEIPT_SHA256,
        expected_size_bytes=R241_EXACT20_INZI_TRANSFER_RECEIPT_SIZE_BYTES,
    )
    _, inzi_receipt = _read_object(
        inzi_receipt_identity.path, label="r241 copied sealed Inzi transfer receipt"
    )
    if (
        inzi_receipt.get("schema") != R241_EXACT20_TRANSFER_SCHEMA
        or inzi_receipt.get("status") != "ready"
        or inzi_receipt.get("candidate_id") != R241_CANDIDATE_ID
        or inzi_receipt.get("source_mutated") is not False
        or inzi_receipt.get("active_training_modified") is not False
        or _r241_exact20_mapping(inzi_receipt.get("source"), label="r241 sealed Inzi source")
        != dict(receipt_source)
        or _r241_exact20_mapping(inzi_receipt.get("corpus"), label="r241 sealed Inzi corpus")
        != dict(corpus)
    ):
        raise R241CheckpointReceiptError(
            "r241 Elmo metadata handoff does not preserve sealed Inzi transfer evidence"
        )


def validate_r241_protected_expert_pointer(
    pointer_path: Path | str,
    *,
    archive_receipt_path: Path | str,
) -> FileIdentity:
    """Validate the create-only r241 Alakazam corpus handoff at launch time.

    Full shard rehashing belongs to the transfer finalizer because it is a
    multi-gigabyte operation.  This boundary rehashes every small receipt and
    checks that the immutable raw Elmo pointer, READY receipt, and manifest
    identities survived unchanged while the local archive binding names the
    exact archive bytes used by runtime and checkpoint receipts.
    """

    archive_identity = file_identity(
        archive_receipt_path,
        label="r241 exact20 bound archive receipt",
        expected_sha256=R241_EXACT20_ARCHIVE_SHA256,
        expected_size_bytes=R241_EXACT20_ARCHIVE_SIZE_BYTES,
    )
    _validate_expert_window(
        archive_identity.path,
        expected_sha256=R241_EXACT20_ARCHIVE_SHA256,
    )
    pointer_file, pointer = _read_object(
        pointer_path, label="r241 transferred protected expert pointer"
    )
    pointer_identity = file_identity(
        pointer_file, label="r241 transferred protected expert pointer"
    )
    root = _directory(pointer_file.parent, label="r241 transferred expert root")
    if (
        pointer.get("schema") != R241_EXACT20_POINTER_SCHEMA
        or pointer.get("protected") is not True
        or pointer.get("manifest") != R241_EXACT20_MANIFEST_NAME
        or _sha256_text(pointer.get("manifest_sha256"), label="r241 top pointer manifest sha")
        != R241_EXACT20_MANIFEST_SHA256
    ):
        raise R241CheckpointReceiptError(
            "r241 transferred top-level pointer does not retain the sealed manifest"
        )
    source_finalization = _r241_exact20_mapping(
        pointer.get("r241_source_finalization"), label="r241 source finalization"
    )

    ready = _r241_exact20_local_identity(
        root,
        _r241_exact20_mapping(
            source_finalization.get("ready"),
            label="r241 source READY provenance",
        ),
        expected_path=R241_EXACT20_SOURCE_READY_NAME,
        label="r241 copied source READY receipt",
        expected_sha256=R241_EXACT20_SOURCE_READY_SHA256,
        expected_size_bytes=R241_EXACT20_SOURCE_READY_SIZE_BYTES,
    )
    raw_pointer = _r241_exact20_local_identity(
        root,
        _r241_exact20_mapping(
            source_finalization.get("source_pointer"),
            label="r241 source pointer provenance",
        ),
        expected_path=R241_EXACT20_SOURCE_POINTER_NAME,
        label="r241 copied source protected pointer",
        expected_sha256=R241_EXACT20_SOURCE_POINTER_SHA256,
        expected_size_bytes=R241_EXACT20_SOURCE_POINTER_SIZE_BYTES,
    )
    manifest = _r241_exact20_local_identity(
        root,
        _r241_exact20_mapping(
            source_finalization.get("source_manifest"),
            label="r241 source manifest provenance",
        ),
        expected_path=R241_EXACT20_MANIFEST_NAME,
        label="r241 copied source manifest",
        expected_sha256=R241_EXACT20_MANIFEST_SHA256,
        expected_size_bytes=R241_EXACT20_MANIFEST_SIZE_BYTES,
    )
    if (
        not str(source_finalization.get("host") or "").strip()
        or not str(source_finalization.get("source_ready_path") or "").strip()
        or not str(source_finalization.get("source_pointer_path") or "").strip()
    ):
        raise R241CheckpointReceiptError("r241 source finalization omits source provenance")

    binding = _r241_exact20_mapping(
        pointer.get("r241_archive_binding"), label="r241 archive binding"
    )
    bound_archive = _regular_file(
        str(binding.get("archive_receipt_path") or ""), label="r241 bound archive path"
    )
    if (
        bound_archive != archive_identity.path
        or _sha256_text(binding.get("archive_receipt_sha256"), label="r241 bound archive sha")
        != archive_identity.sha256
        or _as_exact_int(binding.get("archive_receipt_size_bytes"), label="r241 bound archive size")
        != archive_identity.size_bytes
        or _sha256_text(binding.get("source_archive_receipt_sha256"), label="r241 source archive sha")
        != archive_identity.sha256
        or _as_exact_int(
            binding.get("source_archive_receipt_size_bytes"),
            label="r241 source archive size",
        )
        != archive_identity.size_bytes
        or not str(binding.get("source_host") or "").strip()
        or not str(binding.get("source_archive_receipt_path") or "").strip()
    ):
        raise R241CheckpointReceiptError(
            "r241 protected pointer archive binding does not match actual archive bytes"
        )
    copied_archive = _r241_exact20_local_identity(
        root,
        binding.get("copied_archive_receipt"),
        expected_path=R241_EXACT20_ARCHIVE_COPY_NAME,
        label="r241 copied archive receipt",
        expected_sha256=archive_identity.sha256,
        expected_size_bytes=archive_identity.size_bytes,
    )

    transfer = _r241_exact20_mapping(
        pointer.get("r241_exact20_transfer"), label="r241 exact20 transfer binding"
    )
    if (
        transfer.get("schema") != R241_EXACT20_TRANSFER_SCHEMA
        or transfer.get("source_mutated") is not False
    ):
        raise R241CheckpointReceiptError("r241 exact20 transfer binding is invalid")
    transfer_receipt = _r241_exact20_local_identity(
        root,
        transfer.get("receipt"),
        expected_path=R241_EXACT20_TRANSFER_RECEIPT_NAME,
        label="r241 exact20 transfer receipt",
    )
    _, receipt = _read_object(transfer_receipt.path, label="r241 exact20 transfer receipt")
    if (
        receipt.get("schema") != R241_EXACT20_TRANSFER_SCHEMA
        or receipt.get("status") != "ready"
        or receipt.get("candidate_id") != R241_CANDIDATE_ID
        or receipt.get("source_mutated") is not False
        or receipt.get("active_training_modified") is not False
    ):
        raise R241CheckpointReceiptError("r241 exact20 transfer receipt is not inert ready evidence")
    receipt_source = _r241_exact20_mapping(receipt.get("source"), label="r241 transfer source")
    if (
        receipt_source.get("host") != source_finalization.get("host")
        or not str(receipt_source.get("window_root") or "").strip()
        or receipt_source.get("archive_receipt_path")
        != binding.get("source_archive_receipt_path")
    ):
        raise R241CheckpointReceiptError("r241 transfer receipt source provenance drifted")
    _r241_exact20_local_identity(
        root,
        receipt_source.get("final_ready"),
        expected_path=R241_EXACT20_SOURCE_READY_NAME,
        label="r241 transfer source READY",
        expected_sha256=ready.sha256,
        expected_size_bytes=ready.size_bytes,
    )
    _r241_exact20_local_identity(
        root,
        receipt_source.get("protected_pointer"),
        expected_path=R241_EXACT20_SOURCE_POINTER_NAME,
        label="r241 transfer source pointer",
        expected_sha256=raw_pointer.sha256,
        expected_size_bytes=raw_pointer.size_bytes,
    )
    _r241_exact20_local_identity(
        root,
        receipt_source.get("manifest"),
        expected_path=R241_EXACT20_MANIFEST_NAME,
        label="r241 transfer source manifest",
        expected_sha256=manifest.sha256,
        expected_size_bytes=manifest.size_bytes,
    )
    receipt_destination = _r241_exact20_mapping(
        receipt.get("destination"), label="r241 transfer destination"
    )
    if receipt_destination.get("archive_receipt_path") != str(archive_identity.path):
        raise R241CheckpointReceiptError("r241 transfer receipt binds another archive path")
    _r241_exact20_local_identity(
        root,
        receipt_destination.get("archive_receipt"),
        expected_path=R241_EXACT20_ARCHIVE_COPY_NAME,
        label="r241 transfer copied archive",
        expected_sha256=copied_archive.sha256,
        expected_size_bytes=copied_archive.size_bytes,
    )
    corpus = _r241_exact20_mapping(receipt.get("corpus"), label="r241 transfer corpus")
    if (
        _as_exact_int(corpus.get("records"), label="r241 transfer records")
        != R241_EXACT20_RECORDS
        or _as_exact_int(corpus.get("decisions"), label="r241 transfer decisions")
        != R241_EXACT20_DECISIONS
        or _as_exact_int(corpus.get("shard_bytes"), label="r241 transfer shard bytes")
        != R241_EXACT20_SHARD_BYTES
        or len(list(corpus.get("shards") or ())) != EXACT_WINDOW_DAYS
        or len(list(corpus.get("sidecars") or ())) != EXACT_WINDOW_DAYS
    ):
        raise R241CheckpointReceiptError("r241 transfer receipt corpus totals drifted")

    _validate_r241_elmo_metadata_handoff(
        root,
        receipt=receipt,
        receipt_source=receipt_source,
        corpus=corpus,
    )

    _, raw_payload = _read_object(raw_pointer.path, label="r241 copied raw source pointer")
    extension_keys = {
        "r241_source_finalization",
        "r241_archive_binding",
        "r241_exact20_transfer",
    }
    if (
        set(pointer) - set(raw_payload) != extension_keys
        or any(pointer.get(key) != value for key, value in raw_payload.items())
    ):
        raise R241CheckpointReceiptError(
            "r241 top-level pointer altered raw Elmo pointer provenance"
        )
    return pointer_identity


def _validate_receipt_expert_window(
    row: object,
    *,
    expected_sha256: str,
    label: str,
) -> FileIdentity:
    """Rehash the exact latest-20 receipt named by a generated receipt."""

    identity = _identity_from_row(row, label=f"{label} expert window")
    observed = _validate_expert_window(
        identity.path,
        expected_sha256=expected_sha256,
    )
    if observed != identity:
        raise R241CheckpointReceiptError(f"{label} expert window identity drifted")
    return observed


def _validate_contract(
    path: Path | str,
    *,
    policy: R241AuditPolicy,
) -> tuple[FileIdentity, dict[str, Any]]:
    source, contract = _read_object(path, label="r241 owner contract")
    if (
        contract.get("schema") != R241_CONTRACT_SCHEMA
        or _as_exact_int(contract.get("owner_decision_revision"), label="owner revision")
        != R241_REVISION
        or _as_exact_int(
            contract.get("latest_owner_clarification_revision"),
            label="owner clarification revision",
        )
        != R241_OWNER_CLARIFICATION_REVISION
        or contract.get("candidate_id") != R241_CANDIDATE_ID
    ):
        raise R241CheckpointReceiptError("r241 owner contract identity is invalid")
    parent = dict(contract.get("parent") or {})
    if (
        parent.get("checkpoint_sha256") != policy.parent_sha256
        or _as_exact_int(parent.get("checkpoint_bytes"), label="r195 parent bytes")
        != policy.parent_size_bytes
        or parent.get("typed_source_sha256") != policy.parent_typed_source_sha256
        or parent.get("immutable") is not True
    ):
        raise R241CheckpointReceiptError("r241 owner contract does not bind the immutable r195 parent")
    _validate_r195_parent_typed_source(parent, policy=policy)
    return file_identity(source, label="r241 owner contract"), contract


def _validate_r195_parent_typed_source(
    parent: Mapping[str, Any], *, policy: R241AuditPolicy
) -> FileIdentity:
    """Verify the immutable r195 terminal source named by the owner contract.

    The r241 contract's parent digest is useful, but it is not enough on its
    own: a hand-written contract could otherwise claim that digest while
    severing the r195 completion receipt that fixes its model/tree lineage.
    This intentionally reads the canonical typed source from the repository,
    rather than trusting a path copied into a generated receipt.
    """

    relative = str(parent.get("typed_source") or "").strip()
    if relative != PARENT_R195_TYPED_SOURCE_RELATIVE_PATH:
        raise R241CheckpointReceiptError(
            "r241 owner contract names an unexpected r195 typed source"
        )
    source = ROOT / relative
    identity = file_identity(
        source,
        label="r195 immutable typed source",
        expected_sha256=policy.parent_typed_source_sha256,
    )
    _source_path, typed = _read_object(identity.path, label="r195 immutable typed source")
    completion = dict(typed.get("completion") or {})
    if (
        typed.get("schema") != PARENT_R195_TYPED_SOURCE_SCHEMA
        or _as_exact_int(
            typed.get("owner_decision_revision"), label="r195 typed source revision"
        )
        != 195
        or typed.get("specialist_id") != "alakazam"
        or typed.get("status") != "completed_dual_submission"
        or _sha256_text(
            completion.get("expert_checkpoint_sha256"),
            label="r195 typed source checkpoint",
        )
        != policy.parent_sha256
        or _as_exact_int(
            completion.get("expert_checkpoint_bytes"),
            label="r195 typed source checkpoint bytes",
        )
        != policy.parent_size_bytes
        or _sha256_text(
            completion.get("matchup_tree_sha256"),
            label="r195 typed source learner tree",
        )
        != policy.learner_matchup_tree_sha256
    ):
        raise R241CheckpointReceiptError(
            "r195 immutable typed source does not bind the peak checkpoint/tree"
        )
    return identity


def _validate_receipt_contract(
    row: object,
    *,
    owner_clarification_revision: object,
    policy: R241AuditPolicy,
    label: str,
) -> tuple[FileIdentity, dict[str, Any]]:
    """Re-read the typed owner source named by a receipt, not its claim."""

    identity = _identity_from_row(row, label=f"{label} contract")
    observed, contract = _validate_contract(identity.path, policy=policy)
    if observed != identity:
        raise R241CheckpointReceiptError(f"{label} contract identity drifted")
    expected_revision = _as_exact_int(
        contract.get("latest_owner_clarification_revision"),
        label=f"{label} owner clarification revision",
    )
    if _as_exact_int(
        owner_clarification_revision,
        label=f"{label} receipt owner clarification revision",
    ) != expected_revision:
        raise R241CheckpointReceiptError(
            f"{label} receipt does not bind the current typed owner clarification"
        )
    return observed, contract


def _snapshot_member(root: Path, relative: object, *, label: str) -> Path:
    """Resolve one manifest member without permitting an escape or symlink."""

    text = str(relative or "")
    member = Path(text)
    if (
        not text
        or member.is_absolute()
        or "." in member.parts
        or ".." in member.parts
    ):
        raise R241CheckpointReceiptError(f"{label} has an unsafe relative path")
    candidate = root.joinpath(*member.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise R241CheckpointReceiptError(f"{label} is not a regular snapshot file")
    resolved = candidate.resolve()
    if root not in resolved.parents:
        raise R241CheckpointReceiptError(f"{label} escapes the source snapshot")
    return resolved


def _source_tree_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    canonical_rows = [
        {
            "path": str(row["path"]),
            "sha256": _sha256_text(row["sha256"], label="snapshot member sha"),
            "size_bytes": _as_exact_int(row["size_bytes"], label="snapshot member size"),
        }
        for row in sorted(rows, key=lambda item: str(item["path"]))
    ]
    # Match the source-snapshot publisher/launcher exactly: this digest has no
    # trailing newline, unlike immutable receipt JSON.
    return sha256_bytes(
        json.dumps(
            canonical_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    )


def authenticated_source_snapshot_provenance(
    *,
    source_root: Path | str,
    manifest_path: Path | str,
    outputs_root: Path | str,
    host: str | None = None,
) -> dict[str, object]:
    """Derive immutable code-snapshot provenance without pinning a checkout.

    r241 intentionally has no source-tree checksum in its planning state while
    the PTCGReplay roster migration is being prepared.  A receipt is generated
    only from a later authenticated, read-only manifest.  The manifest's
    content address is therefore an observed runtime fact, not a constant in
    this module or a claim copied from a mutable checkout.
    """

    root = _directory(source_root, label="r241 authenticated source root")
    outputs = _directory(outputs_root, label="r241 external outputs root")
    manifest = _regular_file(manifest_path, label="r241 source snapshot manifest")
    if (
        root not in manifest.parents
        or manifest.name != "r241-source-snapshot-manifest.json"
        or root.stat().st_mode & 0o222
        or manifest.stat().st_mode & 0o222
        or not root.name.startswith("alakazam-new-list-direct-r241-src-")
        or root == outputs
        or root in outputs.parents
        or outputs in root.parents
    ):
        raise R241CheckpointReceiptError(
            "r241 source snapshot is not an authenticated immutable code root"
        )
    _manifest_file, manifest_payload = _read_object(manifest, label="r241 source snapshot manifest")
    if (
        manifest_payload.get("schema") != R241_SOURCE_SNAPSHOT_SCHEMA
        or manifest_payload.get("candidate_id") != R241_CANDIDATE_ID
        or manifest_payload.get("external_outputs_required") is not True
        or manifest_payload.get("baseline_payloads_separate_and_receipted") is not True
        or manifest_payload.get("authenticated") is not True
        or manifest_payload.get("status") != "authenticated_immutable_source_snapshot"
    ):
        raise R241CheckpointReceiptError(
            "r241 source snapshot manifest lacks authenticated immutable provenance"
        )
    source_tree_sha = _sha256_text(
        manifest_payload.get("source_tree_sha256"), label="source snapshot tree"
    )
    owner_contract_sha = _sha256_text(
        manifest_payload.get("owner_contract_sha256"), label="source snapshot owner contract"
    )
    rows_raw = manifest_payload.get("files")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise R241CheckpointReceiptError("r241 source snapshot manifest has no file inventory")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in rows_raw:
        if not isinstance(raw, Mapping):
            raise R241CheckpointReceiptError("r241 source snapshot inventory row is malformed")
        relative = str(raw.get("path") or "")
        if relative in seen:
            raise R241CheckpointReceiptError("r241 source snapshot has duplicate member paths")
        seen.add(relative)
        expected_sha = _sha256_text(raw.get("sha256"), label="source snapshot member sha")
        expected_size = _as_exact_int(raw.get("size_bytes"), label="source snapshot member size")
        if expected_size < 0:
            raise R241CheckpointReceiptError("r241 source snapshot member size is negative")
        member = _snapshot_member(root, relative, label="r241 source snapshot member")
        if member.stat().st_mode & 0o222:
            raise R241CheckpointReceiptError("r241 source snapshot member is writable")
        identity = file_identity(
            member,
            label="r241 source snapshot member",
            expected_sha256=expected_sha,
            expected_size_bytes=expected_size,
        )
        rows.append(
            {
                "path": relative,
                "sha256": identity.sha256,
                "size_bytes": identity.size_bytes,
            }
        )
    if _source_tree_sha256(rows) != source_tree_sha:
        raise R241CheckpointReceiptError("r241 source snapshot tree inventory hash drifted")
    manifest_identity = file_identity(manifest, label="r241 source snapshot manifest")
    result: dict[str, object] = {
        "schema": R241_SOURCE_SNAPSHOT_SCHEMA,
        "status": "authenticated_immutable_source_snapshot",
        "authenticated": True,
        "root": str(root),
        "source_execution_root": str(root),
        "manifest": str(manifest_identity.path),
        "manifest_sha256": manifest_identity.sha256,
        "source_tree_sha256": source_tree_sha,
        "owner_contract_sha256": owner_contract_sha,
        "file_inventory_sha256": sha256_bytes(canonical_json(rows)),
        "outputs_root": str(outputs),
    }
    if host is not None:
        normalized_host = str(host).strip().lower()
        if normalized_host not in {"inzi", "elmo"}:
            raise R241CheckpointReceiptError("r241 source snapshot host is invalid")
        result["host"] = normalized_host
    return result


def _validate_source_snapshot_binding(value: object, *, label: str) -> dict[str, object]:
    """Re-derive a receipt's source provenance from its manifest/root fields."""

    if not isinstance(value, Mapping):
        raise R241CheckpointReceiptError(f"{label} omits authenticated source snapshot")
    row = dict(value)
    host = row.get("host")
    derived = authenticated_source_snapshot_provenance(
        source_root=str(row.get("root") or ""),
        manifest_path=str(row.get("manifest") or ""),
        outputs_root=str(row.get("outputs_root") or ""),
        host=str(host) if host is not None else None,
    )
    if row != derived:
        raise R241CheckpointReceiptError(f"{label} source snapshot identity drifted")
    return derived


def _immutable_json(path: Path | str, payload: Mapping[str, Any], *, label: str) -> FileIdentity:
    target = Path(path).expanduser()
    if target.exists():
        identity = file_identity(target, label=label)
        if target.read_bytes() != canonical_json(dict(payload)):
            raise R241CheckpointReceiptError(f"{label} already exists with different bytes")
        return identity
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json(dict(payload)))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != canonical_json(dict(payload)):
                raise R241CheckpointReceiptError(f"{label} was raced with different bytes")
        return file_identity(target, label=label)
    finally:
        temporary.unlink(missing_ok=True)


def _path_binding(path: Path | str, *, label: str) -> dict[str, object]:
    return file_identity(path, label=label).as_dict()


def validate_r241_h10_adapter_source_binding(
    adapter_receipt: Path | str,
    *,
    source_snapshot: Mapping[str, object],
) -> dict[str, object]:
    """Require an H10 data-only receipt minted from the active source snapshot.

    The adapter receipt is an external, post-snapshot artifact.  Its path and
    checksum therefore cannot participate in the source manifest without
    creating a cycle, but its ``offline_preflight`` must point back to the
    already-sealed source root and manifest.  Re-deriving those fields here
    prevents a byte-valid receipt from an older source snapshot authorizing a
    newer worker, preservation receipt, or launch.
    """

    if not isinstance(source_snapshot, Mapping):
        raise R241CheckpointReceiptError(
            "r241 H10 adapter source binding omits the active source snapshot"
        )
    active = dict(source_snapshot)
    active_root_text = str(active.get("root") or "").strip()
    active_manifest_text = str(active.get("manifest") or "").strip()
    if not active_root_text or not active_manifest_text:
        raise R241CheckpointReceiptError(
            "r241 H10 adapter source binding omits the active source root or manifest"
        )
    active_root = _directory(
        active_root_text, label="active r241 source snapshot root"
    )
    active_manifest = _regular_file(
        active_manifest_text,
        label="active r241 source snapshot manifest",
    )
    active_manifest_sha256 = _sha256_text(
        active.get("manifest_sha256"),
        label="active r241 source snapshot manifest sha256",
    )
    if active_manifest.parent != active_root:
        raise R241CheckpointReceiptError(
            "active r241 source manifest is not rooted in the active snapshot"
        )
    file_identity(
        active_manifest,
        label="active r241 source snapshot manifest",
        expected_sha256=active_manifest_sha256,
    )

    receipt_path, receipt = _read_object(
        adapter_receipt, label="r241 H10 adapter receipt"
    )
    if receipt_path.name != R241_H10_ADAPTER_RECEIPT_BASENAME:
        raise R241CheckpointReceiptError(
            "r241 H10 adapter receipt does not use the predeclared successor path"
        )
    raw_offline = receipt.get("offline_preflight")
    if not isinstance(raw_offline, Mapping):
        raise R241CheckpointReceiptError(
            "r241 H10 adapter receipt omits offline source preflight"
        )
    offline = dict(raw_offline)
    observed_root_text = str(offline.get("source_snapshot_root") or "").strip()
    observed_manifest_text = str(
        offline.get("source_snapshot_manifest") or ""
    ).strip()
    if not observed_root_text or not observed_manifest_text:
        raise R241CheckpointReceiptError(
            "r241 H10 adapter receipt omits its source root or manifest"
        )
    observed_root = Path(observed_root_text).expanduser().resolve()
    observed_manifest = Path(observed_manifest_text).expanduser().resolve()
    observed_manifest_sha256 = _sha256_text(
        offline.get("source_snapshot_manifest_sha256"),
        label="r241 H10 adapter source manifest sha256",
    )
    if (
        observed_root != active_root
        or observed_manifest != active_manifest
        or observed_manifest_sha256 != active_manifest_sha256
    ):
        raise R241CheckpointReceiptError(
            "r241 H10 adapter receipt binds a stale source snapshot"
        )
    if (
        offline.get("model_weights_loaded") is not False
        or offline.get("baseline_package_main_imported") is not False
        or _as_exact_int(
            offline.get("native_function_calls"),
            label="r241 H10 adapter offline native calls",
        )
        != 0
        or _as_exact_int(
            offline.get("search_calls_made"),
            label="r241 H10 adapter offline search calls",
        )
        != 0
        or _as_exact_int(
            offline.get("simulator_battles_started"),
            label="r241 H10 adapter offline simulator battles",
        )
        != 0
    ):
        raise R241CheckpointReceiptError(
            "r241 H10 adapter offline preflight is not data-only/no-search"
        )
    return file_identity(receipt_path, label="r241 H10 adapter receipt").as_dict()


def _checkpoint_identity_matches(
    row: object,
    identity: FileIdentity,
    *,
    label: str,
) -> None:
    """Compare a promotion-style ``digest`` row with a direct file identity.

    The pure-RL boundary receipts predate the JSON receipt convention and use
    ``digest`` rather than ``sha256``.  Requiring both a real path and its
    direct digest here avoids treating that legacy spelling as a weaker claim.
    """

    if not isinstance(row, Mapping):
        raise R241CheckpointReceiptError(f"{label} is not a checkpoint identity")
    raw_path = str(row.get("path") or "").strip()
    if not raw_path:
        raise R241CheckpointReceiptError(f"{label} omits its checkpoint path")
    actual = file_identity(raw_path, label=label)
    digest = _sha256_text(
        row.get("sha256") or row.get("digest"), label=f"{label} digest"
    )
    if actual != identity or digest != identity.sha256:
        raise R241CheckpointReceiptError(f"{label} does not bind the direct checkpoint")
    if row.get("size_bytes") is not None and _as_exact_int(
        row.get("size_bytes"), label=f"{label} size"
    ) != identity.size_bytes:
        raise R241CheckpointReceiptError(f"{label} size does not match its checkpoint")


def _validate_terminal_refresh_boundary(
    *,
    terminal_refresh_receipt: Path | str,
    terminal_rehearsal_receipt: Path | str,
    terminal_parent: FileIdentity,
    terminal_checkpoint: FileIdentity,
    terminal_payload: Mapping[str, Any],
    root_parent_audit: Mapping[str, Any],
    contract_identity: FileIdentity,
    source_snapshot: Mapping[str, object],
    baseline_adapter_roster: FileIdentity,
    slot_migration: Mapping[str, object],
) -> dict[str, object]:
    """Derive the exact terminal five-epoch boundary proof from live files.

    Finalization must not infer that the terminal checkpoint came from the
    update-ten refresh merely from its filename or a claimed parent digest.
    The terminal refresh, durable rehearsal receipt, and the checkpoint's
    own peak-r195 sidecar all need to agree on the exact same boundary.
    """

    refresh_path, refresh = _read_object(
        terminal_refresh_receipt, label="r241 terminal expert refresh"
    )
    rehearsal_path, rehearsal = _read_object(
        terminal_rehearsal_receipt, label="r241 terminal five-epoch rehearsal"
    )
    refresh_identity = file_identity(refresh_path, label="r241 terminal expert refresh")
    rehearsal_identity = file_identity(
        rehearsal_path, label="r241 terminal five-epoch rehearsal"
    )
    if (
        refresh.get("schema") != TERMINAL_EXPERT_REFRESH_SCHEMA
        or _as_exact_int(
            refresh.get("before_iteration"), label="terminal refresh boundary"
        )
        != TERMINAL_REFRESH_BOUNDARY
        or _as_exact_int(
            refresh.get("rl_updates_completed"), label="terminal refresh updates"
        )
        != TERMINAL_REFRESH_BOUNDARY
        or _as_exact_int(
            refresh.get("epochs_completed"), label="terminal refresh epochs"
        )
        != TERMINAL_REFRESH_EPOCHS
        or refresh.get("next_collection_started") is not False
    ):
        raise R241CheckpointReceiptError(
            "terminal refresh is not the exact boundary-10 five-epoch record"
        )
    _checkpoint_identity_matches(
        refresh.get("parent"), terminal_parent, label="terminal refresh parent"
    )
    _checkpoint_identity_matches(
        refresh.get("refreshed"),
        terminal_checkpoint,
        label="terminal refresh output",
    )
    if (
        _as_exact_int(rehearsal.get("schema"), label="terminal rehearsal schema") != 5
        or _as_exact_int(
            rehearsal.get("before_iteration"), label="terminal rehearsal boundary"
        )
        != TERMINAL_REFRESH_BOUNDARY
        or _as_exact_int(rehearsal.get("epochs"), label="terminal rehearsal epochs")
        != TERMINAL_REFRESH_EPOCHS
        or _sha256_text(
            rehearsal.get("parent_digest"), label="terminal rehearsal parent digest"
        )
        != terminal_parent.sha256
        or _sha256_text(
            rehearsal.get("checkpoint_digest"),
            label="terminal rehearsal checkpoint digest",
        )
        != terminal_checkpoint.sha256
    ):
        raise R241CheckpointReceiptError(
            "terminal rehearsal is not the exact boundary-10 five-epoch record"
        )
    rehearsal_checkpoint_path = str(rehearsal.get("checkpoint") or "").strip()
    if not rehearsal_checkpoint_path or file_identity(
        rehearsal_checkpoint_path, label="terminal rehearsal checkpoint"
    ) != terminal_checkpoint:
        raise R241CheckpointReceiptError(
            "terminal rehearsal does not bind the terminal checkpoint path"
        )

    checkpoint_extra = dict(terminal_payload.get("extra") or {})
    checkpoint_fusion = dict(checkpoint_extra.get("peak_r195_live_fusion") or {})
    rehearsal_fusion = dict(rehearsal.get("peak_r195_live_fusion") or {})
    embedded_rehearsal = dict(refresh.get("expert_rehearsal") or {})
    embedded_fusion = dict(embedded_rehearsal.get("peak_r195_live_fusion") or {})
    fusion_migration = dict(checkpoint_fusion.get("adapter_slot_migration") or {})
    if (
        not checkpoint_fusion
        or checkpoint_fusion != rehearsal_fusion
        or checkpoint_fusion != embedded_fusion
    ):
        raise R241CheckpointReceiptError(
            "terminal checkpoint/refresh/rehearsal peak-r195 fusion sidecars drifted"
        )
    if (
        checkpoint_fusion.get("schema") != R241_PEAK_R195_LIVE_FUSION_SCHEMA
        or checkpoint_fusion.get("candidate_id") != R241_CANDIDATE_ID
        or _as_exact_int(
            checkpoint_fusion.get("owner_decision_revision"),
            label="terminal fusion owner revision",
        )
        != R241_REVISION
        or _as_exact_int(
            checkpoint_fusion.get("owner_clarification_revision"),
            label="terminal fusion owner clarification",
        )
        != R241_OWNER_CLARIFICATION_REVISION
        or _as_exact_int(
            checkpoint_fusion.get("fixed_cycle_updates"),
            label="terminal fusion update count",
        )
        != TERMINAL_REFRESH_BOUNDARY
        or checkpoint_fusion.get("phase") != "expert_refresh"
        or _as_exact_int(
            checkpoint_fusion.get("boundary_iteration"),
            label="terminal fusion boundary",
        )
        != TERMINAL_REFRESH_BOUNDARY
        or checkpoint_fusion.get("physical_head_count") != 19
        or checkpoint_fusion.get("active_non_combo_head_count") != 18
        or dict(checkpoint_fusion.get("combo_state") or {})
        != {
            "head_present": True,
            "loss_weight": 0.0,
            "fusion_route_enabled": False,
        }
        or checkpoint_fusion.get("checkpoint_audit_fingerprint_sha256")
        != root_parent_audit.get("audit_fingerprint_sha256")
        or checkpoint_fusion.get("source_snapshot") != dict(source_snapshot)
        or fusion_migration.get("schema") != R241_ADAPTER_SLOT_MIGRATION_SCHEMA
        or fusion_migration.get("status") != "no_slot_change"
        or fusion_migration.get("existing_slots_byte_immutable") is not True
        or _as_exact_int(
            fusion_migration.get("retained_slot_count"),
            label="terminal fusion retained adapter slots",
        )
        != IMMUTABLE_ADAPTER_SLOT_PREFIX
        or list(fusion_migration.get("new_slots") or [])
        or list(fusion_migration.get("new_slot_proofs") or [])
        or fusion_migration.get("parent_slot_registry_sha256")
        != slot_migration.get("parent_slot_registry_sha256")
        or fusion_migration.get("candidate_slot_registry_sha256")
        != slot_migration.get("candidate_slot_registry_sha256")
        or fusion_migration.get("retained_slot_tensor_inventory_sha256")
        != slot_migration.get("retained_slot_tensor_inventory_sha256")
    ):
        raise R241CheckpointReceiptError(
            "terminal peak-r195 live-fusion sidecar is not the exact boundary-10 proof"
        )
    _checkpoint_identity_matches(
        checkpoint_fusion.get("owner_contract"),
        contract_identity,
        label="terminal fusion owner contract",
    )
    _checkpoint_identity_matches(
        checkpoint_fusion.get("baseline_adapter_roster"),
        baseline_adapter_roster,
        label="terminal fusion baseline adapter roster",
    )
    # The refresh embeds the durable receipt for human/recovery inspection;
    # compare its hard provenance fields instead of relying on a timestamp or
    # optional recovery annotation copied by the trainer.
    if (
        _as_exact_int(
            embedded_rehearsal.get("before_iteration"),
            label="embedded terminal rehearsal boundary",
        )
        != TERMINAL_REFRESH_BOUNDARY
        or _as_exact_int(
            embedded_rehearsal.get("epochs"), label="embedded terminal rehearsal epochs"
        )
        != TERMINAL_REFRESH_EPOCHS
        or _sha256_text(
            embedded_rehearsal.get("parent_digest"),
            label="embedded terminal rehearsal parent",
        )
        != terminal_parent.sha256
        or _sha256_text(
            embedded_rehearsal.get("checkpoint_digest"),
            label="embedded terminal rehearsal checkpoint",
        )
        != terminal_checkpoint.sha256
    ):
        raise R241CheckpointReceiptError(
            "terminal refresh embedded rehearsal does not bind boundary ten"
        )
    return {
        "terminal_expert_refresh": refresh_identity.as_dict(),
        "terminal_five_epoch_rehearsal": rehearsal_identity.as_dict(),
        "before_iteration": TERMINAL_REFRESH_BOUNDARY,
        "rl_updates_completed": TERMINAL_REFRESH_BOUNDARY,
        "epochs_completed": TERMINAL_REFRESH_EPOCHS,
        "parent": terminal_parent.as_dict(),
        "refreshed": terminal_checkpoint.as_dict(),
        "peak_r195_live_fusion_sha256": sha256_bytes(
            canonical_json(checkpoint_fusion)
        ),
    }


def generate_peak_r195_preservation_receipt(
    *,
    output_path: Path | str,
    contract_path: Path | str,
    parent_checkpoint: Path | str,
    learner_matchup_tree: Path | str,
    h10_matchup_tree: Path | str,
    official_cg_root: Path | str,
    environment: Mapping[str, str],
    expert_window_receipt: Path | str,
    protected_expert_pointer: Path | str,
    h10_adapter_receipt: Path | str,
    active_gate_contract: Path | str,
    frozen_specialist_registry: Path | str,
    research_control_registry: Path | str,
    adapter_training_activation: Path | str,
    source_snapshot_root: Path | str,
    source_snapshot_manifest: Path | str,
    source_outputs_root: Path | str,
    source_snapshot_host: str,
    baseline_adapter_roster: Path | str = BASELINE_ADAPTER_ROSTER_PATH,
    policy: R241AuditPolicy = DEFAULT_POLICY,
) -> dict[str, object]:
    """Generate the host-local pre-launch preservation receipt from real bytes."""

    if Path(output_path).name != R241_PEAK_R195_PRESERVATION_RECEIPT_BASENAME:
        raise R241CheckpointReceiptError(
            "r241 peak-r195 preservation receipt does not use the predeclared "
            "successor path"
        )
    contract_identity, contract = _validate_contract(contract_path, policy=policy)
    direct_environment = _assert_direct_environment(
        environment, official_cg_root=official_cg_root
    )
    parent = audit_checkpoint(
        parent_checkpoint,
        expected_sha256=policy.parent_sha256,
        expected_size_bytes=policy.parent_size_bytes,
        policy=policy,
        environment=environment,
        adapter_training_activation=adapter_training_activation,
    )
    baseline_roster_identity, baseline_roster = _baseline_adapter_roster(
        path=baseline_adapter_roster, policy=policy
    )
    _parent_identity, parent_payload = _load_checkpoint_payload(
        parent_checkpoint,
        label="r195 parent checkpoint",
        expected_sha256=policy.parent_sha256,
        expected_size_bytes=policy.parent_size_bytes,
    )
    if _slot_registry_from_payload(parent_payload, label="r195 parent") != baseline_roster:
        raise R241CheckpointReceiptError(
            "r195 checkpoint slot registry does not match the immutable 0..19 baseline"
        )
    source_snapshot = authenticated_source_snapshot_provenance(
        source_root=source_snapshot_root,
        manifest_path=source_snapshot_manifest,
        outputs_root=source_outputs_root,
        host=source_snapshot_host,
    )
    if source_snapshot.get("owner_contract_sha256") != contract_identity.sha256:
        raise R241CheckpointReceiptError(
            "r241 source snapshot is not authenticated to this owner contract"
        )
    learner_tree = _validate_tree(
        learner_matchup_tree,
        label="r195 learner matchup tree",
        expected_sha256=policy.learner_matchup_tree_sha256,
    )
    h10_tree = _validate_tree(
        h10_matchup_tree,
        label="H10 Marnie matchup tree",
        expected_sha256=policy.h10_matchup_tree_sha256,
        expected_size_bytes=policy.h10_matchup_tree_size_bytes,
    )
    expert_identity = _validate_expert_window(
        expert_window_receipt,
        expected_sha256=str(
            dict(contract.get("expert_soft_refresh") or {})
            .get("exact_window_evidence_binding", {})
            .get("canonical_manifest_sha256")
            or ""
        ),
    )
    protected = validate_r241_protected_expert_pointer(
        protected_expert_pointer,
        archive_receipt_path=expert_identity.path,
    )
    adapter_receipt = validate_r241_h10_adapter_source_binding(
        h10_adapter_receipt,
        source_snapshot=source_snapshot,
    )
    adapter_training = _path_binding(
        adapter_training_activation, label="r241 isolated adapter training activation"
    )
    public = {
        "active_gate_contract": _path_binding(active_gate_contract, label="r241 active gate contract"),
        "frozen_specialist_registry": _path_binding(
            frozen_specialist_registry, label="r241 frozen specialist registry"
        ),
        "research_control_registry": _path_binding(
            research_control_registry, label="r241 research-control registry"
        ),
        "established_diverse_public_mix_preserved": True,
        "research_control_phase_preserved": True,
        "official_collect_fraction": 0.50,
        "research_control_games_per_iter": 1_000,
    }
    no_slot_change = audit_append_only_adapter_slot_migration(
        parent_checkpoint=parent_checkpoint,
        candidate_checkpoint=parent_checkpoint,
        parent_audit=parent,
        candidate_audit=parent,
    )
    if no_slot_change.get("status") != "no_slot_change":
        raise R241CheckpointReceiptError("r241 peak receipt requires no adapter-slot change")
    receipt: dict[str, object] = {
        "schema": PEAK_R195_PRESERVATION_SCHEMA,
        "revision": R241_REVISION,
        "owner_clarification_revision": _as_exact_int(
            contract.get("latest_owner_clarification_revision"),
            label="owner clarification revision",
        ),
        "candidate_id": R241_CANDIDATE_ID,
        "status": "passed",
        "passed": True,
        "derived_not_self_asserted": True,
        "contract": contract_identity.as_dict(),
        "parent": dict(parent["checkpoint"]),
        "checkpoint_audit": parent,
        "source_snapshot": source_snapshot,
        "baseline_adapter_roster": baseline_roster_identity.as_dict(),
        "adapter_slot_migration": no_slot_change,
        "heads": dict(parent["heads"]),
        "matchup_adapter": {
            **dict(parent["matchup_adapter"]),
            "bank_preserved": True,
            "checkpoint_runtime_enabled": False,
            "checkpoint_training_enabled": False,
            "runtime_package_activation_required": True,
            "training_enabled": True,
            "runtime_enabled": True,
            "epochs_per_rl_update": 1,
            "learner_matchup_tree": learner_tree.as_dict(),
            "training_activation": adapter_training,
        },
        "h10_marnie_matchup_tree": h10_tree.as_dict(),
        "direct_environment": direct_environment,
        "runtime_smoke": dict(parent["runtime_smoke"]),
        "public_mix": public,
        "expert_window": {
            "archive_receipt": expert_identity.as_dict(),
            "archive_receipt_sha256": expert_identity.sha256,
            "protected_pointer": protected.as_dict(),
        },
        "h10_adapter_receipt": adapter_receipt,
        "trainer": {
            "r195_non_combo_arguments": [
                "--archetype-aux-loss-weight", "0.05",
                "--opp-hand-loss-weight", "0.05",
                "--opp-remainder-loss-weight", "0.05",
                "--lethal-threat-loss-weight", "0.025",
                "--prize-race-loss-weight", "0.025",
                "--setup-board-outcome-loss-weight", "0.025",
            ],
        },
    }
    receipt["receipt_fingerprint_sha256"] = sha256_bytes(canonical_json(receipt))
    identity = _immutable_json(output_path, receipt, label="r241 peak-r195 preservation receipt")
    return {**receipt, "receipt": identity.as_dict()}


def generate_terminal_runtime_receipts(
    *,
    model_output_path: Path | str,
    matchup_output_path: Path | str,
    contract_path: Path | str,
    r195_parent_checkpoint: Path | str,
    terminal_parent_checkpoint: Path | str,
    terminal_checkpoint: Path | str,
    learner_matchup_tree: Path | str,
    h10_matchup_tree: Path | str,
    official_cg_root: Path | str,
    environment: Mapping[str, str],
    expert_window_receipt: Path | str,
    terminal_refresh_receipt: Path | str,
    terminal_rehearsal_receipt: Path | str,
    source_snapshot_root: Path | str,
    source_snapshot_manifest: Path | str,
    source_outputs_root: Path | str,
    source_snapshot_host: str,
    baseline_adapter_roster: Path | str = BASELINE_ADAPTER_ROSTER_PATH,
    policy: R241AuditPolicy = DEFAULT_POLICY,
) -> dict[str, object]:
    """Generate terminal model + matchup activation receipts after update ten."""

    contract_identity, contract = _validate_contract(contract_path, policy=policy)
    direct_environment = _assert_direct_environment(
        environment, official_cg_root=official_cg_root
    )
    root_parent = audit_checkpoint(
        r195_parent_checkpoint,
        expected_sha256=policy.parent_sha256,
        expected_size_bytes=policy.parent_size_bytes,
        policy=policy,
        environment=environment,
    )
    baseline_roster_identity, baseline_roster = _baseline_adapter_roster(
        path=baseline_adapter_roster, policy=policy
    )
    _parent_identity, parent_payload = _load_checkpoint_payload(
        r195_parent_checkpoint,
        label="r195 parent checkpoint",
        expected_sha256=policy.parent_sha256,
        expected_size_bytes=policy.parent_size_bytes,
    )
    if _slot_registry_from_payload(parent_payload, label="r195 parent") != baseline_roster:
        raise R241CheckpointReceiptError(
            "r195 checkpoint slot registry does not match immutable 0..19 baseline"
        )
    source_snapshot = authenticated_source_snapshot_provenance(
        source_root=source_snapshot_root,
        manifest_path=source_snapshot_manifest,
        outputs_root=source_outputs_root,
        host=source_snapshot_host,
    )
    if source_snapshot.get("owner_contract_sha256") != contract_identity.sha256:
        raise R241CheckpointReceiptError(
            "r241 source snapshot is not authenticated to this owner contract"
        )
    terminal_parent_identity, _terminal_parent_payload = _load_checkpoint_payload(
        terminal_parent_checkpoint,
        label="r241 terminal iter_00009 parent checkpoint",
    )
    terminal = audit_checkpoint(
        terminal_checkpoint,
        policy=policy,
        environment=environment,
    )
    terminal_payload_identity, terminal_payload = _load_checkpoint_payload(
        terminal_checkpoint,
        label="r241 terminal checkpoint",
        expected_sha256=str(dict(terminal["checkpoint"])["sha256"]),
        expected_size_bytes=int(dict(terminal["checkpoint"])["size_bytes"]),
    )
    extra = dict(terminal_payload.get("extra") or {})
    if _sha256_text(extra.get("parent_digest"), label="terminal checkpoint parent digest") != terminal_parent_identity.sha256:
        raise R241CheckpointReceiptError(
            "terminal checkpoint does not prove the immediate iter_00009 parent"
        )
    if (
        dict(root_parent["heads"])["active_non_combo_head_names_sha256"]
        != dict(terminal["heads"])["active_non_combo_head_names_sha256"]
        or dict(root_parent["heads"])["active_non_combo_route_names_sha256"]
        != dict(terminal["heads"])["active_non_combo_route_names_sha256"]
        or dict(root_parent["model_config"])
        != dict(terminal["model_config"])
    ):
        raise R241CheckpointReceiptError(
            "terminal checkpoint changed the peak-r195 architecture/route contract"
        )
    slot_migration = audit_append_only_adapter_slot_migration(
        parent_checkpoint=r195_parent_checkpoint,
        candidate_checkpoint=terminal_checkpoint,
        parent_audit=root_parent,
        candidate_audit=terminal,
    )
    if slot_migration.get("status") != "no_slot_change":
        raise R241CheckpointReceiptError(
            "r241 terminal receipts forbid PTCGReplay adapter-slot activation this cycle"
        )
    terminal_refresh = _validate_terminal_refresh_boundary(
        terminal_refresh_receipt=terminal_refresh_receipt,
        terminal_rehearsal_receipt=terminal_rehearsal_receipt,
        terminal_parent=terminal_parent_identity,
        terminal_checkpoint=terminal_payload_identity,
        terminal_payload=terminal_payload,
        root_parent_audit=root_parent,
        contract_identity=contract_identity,
        source_snapshot=source_snapshot,
        baseline_adapter_roster=baseline_roster_identity,
        slot_migration=slot_migration,
    )
    learner_tree = _validate_tree(
        learner_matchup_tree,
        label="terminal learner matchup tree",
        expected_sha256=policy.learner_matchup_tree_sha256,
    )
    h10_tree = _validate_tree(
        h10_matchup_tree,
        label="terminal H10 Marnie matchup tree",
        expected_sha256=policy.h10_matchup_tree_sha256,
        expected_size_bytes=policy.h10_matchup_tree_size_bytes,
    )
    expert_identity = _validate_expert_window(
        expert_window_receipt,
        expected_sha256=str(
            dict(contract.get("expert_soft_refresh") or {})
            .get("exact_window_evidence_binding", {})
            .get("canonical_manifest_sha256")
            or ""
        ),
    )
    model_receipt: dict[str, object] = {
        "schema": MODEL_RUNTIME_ACTIVATION_SCHEMA,
        "owner_decision_revision": R241_REVISION,
        "owner_clarification_revision": _as_exact_int(
            contract.get("latest_owner_clarification_revision"),
            label="owner clarification revision",
        ),
        "candidate_id": R241_CANDIDATE_ID,
        "status": "active_peak_r195_non_combo_fusion",
        "derived_not_self_asserted": True,
        "contract": contract_identity.as_dict(),
        "parent_r195_checkpoint": dict(root_parent["checkpoint"]),
        "parent_r195_checkpoint_sha256": policy.parent_sha256,
        "parent_r195_typed_source_sha256": policy.parent_typed_source_sha256,
        "terminal_parent_checkpoint": terminal_parent_identity.as_dict(),
        "terminal_checkpoint": terminal_payload_identity.as_dict(),
        "checkpoint_audit": terminal,
        "peak_r195_checkpoint_audit": root_parent,
        "source_snapshot": source_snapshot,
        "baseline_adapter_roster": baseline_roster_identity.as_dict(),
        "adapter_slot_migration": slot_migration,
        "heads": dict(terminal["heads"]),
        "matchup_adapter": dict(terminal["matchup_adapter"]),
        "direct_environment": direct_environment,
        "runtime_smoke": dict(terminal["runtime_smoke"]),
        "runtime_package_activation": {
            "matchup_adapters_enabled": True,
            "checkpoint_remains_dormant": True,
            "method": "direct_v6_adapter_forward_with_runtime_enable/v1",
        },
        "terminal_refresh": terminal_refresh,
        "expert_window": expert_identity.as_dict(),
        "action_selector": "direct_policy_only",
        "mcts_enabled": False,
        "recursive_turn_planner_enabled": False,
        "search_enabled": False,
        "belief_assets_enabled": False,
    }
    model_receipt["receipt_fingerprint_sha256"] = sha256_bytes(canonical_json(model_receipt))
    model_identity = _immutable_json(
        model_output_path, model_receipt, label="r241 terminal model runtime activation"
    )
    matchup_receipt: dict[str, object] = {
        "schema": MATCHUP_RUNTIME_ACTIVATION_SCHEMA,
        "owner_decision_revision": R241_REVISION,
        "owner_clarification_revision": _as_exact_int(
            contract.get("latest_owner_clarification_revision"),
            label="owner clarification revision",
        ),
        "candidate_id": R241_CANDIDATE_ID,
        "status": "active_direct_policy_only",
        "derived_not_self_asserted": True,
        "contract": contract_identity.as_dict(),
        "parent_r195_checkpoint": dict(root_parent["checkpoint"]),
        "terminal_checkpoint": terminal_payload_identity.as_dict(),
        "learner_matchup_tree": learner_tree.as_dict(),
        "h10_training_opponent": {
            "matchup_tree": h10_tree.as_dict(),
            "direct_policy_only": True,
            "mcts_enabled": False,
            "recursive_turn_planner_enabled": False,
            "search_enabled": False,
        },
        "checkpoint_audit_fingerprint_sha256": str(terminal["audit_fingerprint_sha256"]),
        "model_runtime_activation": model_identity.as_dict(),
        "matchup_adapter": dict(terminal["matchup_adapter"]),
        "source_snapshot": source_snapshot,
        "baseline_adapter_roster": baseline_roster_identity.as_dict(),
        "adapter_slot_migration": slot_migration,
        "terminal_refresh": terminal_refresh,
        "direct_environment": direct_environment,
        "runtime_smoke": dict(terminal["runtime_smoke"]),
        "expert_window": expert_identity.as_dict(),
        "action_selector": "direct_policy_only",
        "mcts_enabled": False,
        "recursive_turn_planner_enabled": False,
        "search_enabled": False,
        "belief_assets_enabled": False,
    }
    matchup_receipt["receipt_fingerprint_sha256"] = sha256_bytes(canonical_json(matchup_receipt))
    matchup_identity = _immutable_json(
        matchup_output_path, matchup_receipt, label="r241 terminal matchup runtime activation"
    )
    return {
        "model_runtime_activation": {**model_receipt, "receipt": model_identity.as_dict()},
        "matchup_runtime_activation": {**matchup_receipt, "receipt": matchup_identity.as_dict()},
    }


def _validate_receipt_fingerprint(receipt: Mapping[str, Any], *, label: str) -> None:
    claimed = _sha256_text(
        receipt.get("receipt_fingerprint_sha256"), label=f"{label} fingerprint"
    )
    bare = dict(receipt)
    bare.pop("receipt_fingerprint_sha256", None)
    actual = sha256_bytes(canonical_json(bare))
    if claimed != actual:
        raise R241CheckpointReceiptError(f"{label} fingerprint does not match contents")


def _validate_checkpoint_audit(
    audit: object,
    *,
    expected_checkpoint: FileIdentity,
    expected_heads: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    if not isinstance(audit, Mapping):
        raise R241CheckpointReceiptError(f"{label} omits checkpoint-derived audit")
    payload = dict(audit)
    if payload.get("schema") != R241_CHECKPOINT_AUDIT_SCHEMA:
        raise R241CheckpointReceiptError(f"{label} uses a self-asserted v1 audit")
    claimed = _sha256_text(
        payload.get("audit_fingerprint_sha256"), label=f"{label} audit fingerprint"
    )
    bare = dict(payload)
    bare.pop("audit_fingerprint_sha256", None)
    if sha256_bytes(canonical_json(bare)) != claimed:
        raise R241CheckpointReceiptError(f"{label} audit fingerprint mismatch")
    _identity_matches(payload.get("checkpoint"), expected_checkpoint, label=f"{label} checkpoint")
    heads = dict(payload.get("heads") or {})
    route_ids = tuple(str(value) for value in heads.get("active_non_combo_fusion_route_ids") or [])
    route_ids_sha256 = sha256_bytes(canonical_json(list(route_ids)))
    if (
        heads.get("architecture_present_head_count") != 19
        or heads.get("non_combo_head_count") != 18
        or heads.get("non_combo_route_count") != 18
        or tuple(heads.get("active_non_combo_head_names") or []) != expected_heads
        or tuple(heads.get("active_non_combo_route_names") or []) != expected_heads
        or len(route_ids) != 18
        or tuple(sorted(route_ids)) != route_ids
        or len(set(route_ids)) != 18
        or any(not value for value in route_ids)
        or heads.get("active_non_combo_fusion_route_ids_sha256") != route_ids_sha256
        or payload.get("head_role_map_route_ids_sha256") != route_ids_sha256
        or heads.get("every_non_combo_head_trainable") is not True
        or heads.get("every_non_combo_fusion_route_enabled") is not True
        or dict(heads.get("combo_state") or {}).get("head_present") is not True
        or dict(heads.get("combo_state") or {}).get("loss_weight") != 0.0
        or dict(heads.get("combo_state") or {}).get("route_enabled") is not False
    ):
        raise R241CheckpointReceiptError(f"{label} does not preserve the 19/18 head inventory")
    inventory = dict(payload.get("sorted_tensor_inventory") or {})
    if not _sha256_text(inventory.get("structural_sha256"), label=f"{label} tensor structural hash"):
        raise AssertionError("unreachable")
    if not _sha256_text(inventory.get("content_sha256"), label=f"{label} tensor content hash"):
        raise AssertionError("unreachable")
    adapter = dict(payload.get("matchup_adapter") or {})
    isolated = dict(adapter.get("isolated_optimizer") or {})
    expected_adapter_activation = {
        "matchup_adapter_bank_preserved": True,
        "matchup_adapter_training_enabled": True,
        "matchup_adapter_runtime_enabled": True,
        "matchup_adapter_checkpoint_runtime_enabled": False,
        "matchup_adapter_checkpoint_training_enabled": False,
        "matchup_adapter_checkpoint_main_optimizer_included": False,
        "matchup_adapter_isolated_bank_only_optimizer": True,
        "matchup_adapter_isolated_fit_continuation_required": True,
        "matchup_adapter_external_collection_runtime_enabled": True,
        "matchup_adapter_external_terminal_runtime_enabled": True,
    }
    if (
        dict(adapter.get("checkpoint_dormant_state") or {}).get("runtime_enabled") is not False
        or dict(adapter.get("checkpoint_dormant_state") or {}).get("training_enabled") is not False
        or dict(adapter.get("checkpoint_dormant_state") or {}).get("ordinary_optimizer_included") is not False
        or _as_exact_int(isolated.get("parameter_count"), label=f"{label} isolated adapter params") != 256
        or _as_exact_int(isolated.get("state_count"), label=f"{label} isolated adapter optimizer state") <= 0
        or not _sha256_text(isolated.get("parameter_name_inventory_sha256"), label=f"{label} adapter optimizer names")
        or dict(adapter.get("activation_provenance") or {})
        != expected_adapter_activation
    ):
        raise R241CheckpointReceiptError(f"{label} adapter optimizer evidence is incomplete")
    smoke = dict(payload.get("runtime_smoke") or {})
    if (
        smoke.get("model_reconstructed") is not True
        or smoke.get("adapter_runtime_enabled_for_smoke") is not True
        or smoke.get("adapter_output_changed") is not True
        or smoke.get("action_selector") != "direct_policy_only"
        or smoke.get("mcts_calls") != 0
        or smoke.get("rtp_calls") != 0
        or smoke.get("search_calls") != 0
    ):
        raise R241CheckpointReceiptError(f"{label} runtime smoke is not direct adapter-on")
    return payload


def _recompute_peak_checkpoint_audit(
    *,
    parent: FileIdentity,
    policy: R241AuditPolicy,
    environment: Mapping[str, str],
    training_activation: FileIdentity,
) -> dict[str, object]:
    """Redo peak evidence with the receipt's checked host-local EA38 copy."""

    return audit_checkpoint(
        parent.path,
        expected_sha256=parent.sha256,
        expected_size_bytes=parent.size_bytes,
        policy=policy,
        environment=environment,
        adapter_training_activation=training_activation.path,
    )


def validate_peak_r195_preservation_receipt(
    *,
    receipt_path: Path | str,
    parent_checkpoint: Path | str,
    learner_matchup_tree: Path | str,
    h10_matchup_tree: Path | str,
    official_cg_root: Path | str,
    environment: Mapping[str, str],
    policy: R241AuditPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Revalidate a v2 pre-launch receipt against direct checkpoint/tree bytes."""

    if Path(receipt_path).name != R241_PEAK_R195_PRESERVATION_RECEIPT_BASENAME:
        raise R241CheckpointReceiptError(
            "r241 peak-r195 preservation receipt does not use the predeclared "
            "successor path"
        )
    receipt_file, receipt = _read_object(receipt_path, label="r241 peak-r195 preservation receipt")
    _validate_receipt_fingerprint(receipt, label="peak-r195 preservation receipt")
    if (
        receipt.get("schema") != PEAK_R195_PRESERVATION_SCHEMA
        or _as_exact_int(receipt.get("revision"), label="preservation revision") != R241_REVISION
        or receipt.get("candidate_id") != R241_CANDIDATE_ID
        or receipt.get("status") != "passed"
        or receipt.get("passed") is not True
        or receipt.get("derived_not_self_asserted") is not True
    ):
        raise R241CheckpointReceiptError("peak-r195 preservation receipt is not generated v2 evidence")
    receipt_contract, _contract = _validate_receipt_contract(
        receipt.get("contract"),
        owner_clarification_revision=receipt.get("owner_clarification_revision"),
        policy=policy,
        label="peak-r195 preservation receipt",
    )
    expected_window_sha = _sha256_text(
        dict(_contract.get("expert_soft_refresh") or {})
        .get("exact_window_evidence_binding", {})
        .get("canonical_manifest_sha256"),
        label="peak-r195 expert window canonical receipt",
    )
    expert_window = dict(receipt.get("expert_window") or {})
    archive_window = _validate_receipt_expert_window(
        expert_window.get("archive_receipt"),
        expected_sha256=expected_window_sha,
        label="peak-r195 preservation receipt",
    )
    if expert_window.get("archive_receipt_sha256") != archive_window.sha256:
        raise R241CheckpointReceiptError(
            "peak-r195 preservation receipt expert window checksum drifted"
        )
    protected_pointer = _identity_from_row(
        expert_window.get("protected_pointer"),
        label="peak-r195 protected expert pointer",
    )
    observed_protected_pointer = validate_r241_protected_expert_pointer(
        protected_pointer.path,
        archive_receipt_path=archive_window.path,
    )
    if observed_protected_pointer != protected_pointer:
        raise R241CheckpointReceiptError(
            "peak-r195 protected expert pointer identity drifted"
        )
    adapter_identity = _identity_from_row(
        receipt.get("h10_adapter_receipt"),
        label="peak-r195 H10 adapter receipt",
    )
    source_snapshot = _validate_source_snapshot_binding(
        receipt.get("source_snapshot"), label="peak-r195 preservation receipt"
    )
    if source_snapshot.get("owner_contract_sha256") != receipt_contract.sha256:
        raise R241CheckpointReceiptError("peak-r195 source snapshot binds another owner contract")
    validate_r241_h10_adapter_source_binding(
        adapter_identity.path,
        source_snapshot=source_snapshot,
    )
    baseline_identity, baseline_registry = _baseline_adapter_roster(policy=policy)
    _identity_matches(
        receipt.get("baseline_adapter_roster"),
        baseline_identity,
        label="peak-r195 baseline adapter roster",
    )
    direct_environment = _assert_direct_environment(environment, official_cg_root=official_cg_root)
    parent = file_identity(
        parent_checkpoint,
        label="r195 parent checkpoint",
        expected_sha256=policy.parent_sha256,
        expected_size_bytes=policy.parent_size_bytes,
    )
    _identity_matches(receipt.get("parent"), parent, label="preservation parent")
    _head_map, expected_heads, _routes = _head_contract(policy=policy)
    audit = _validate_checkpoint_audit(
        receipt.get("checkpoint_audit"),
        expected_checkpoint=parent,
        expected_heads=expected_heads,
        label="peak-r195 preservation",
    )
    receipt_adapter = dict(receipt.get("matchup_adapter") or {})
    audit_adapter = dict(audit.get("matchup_adapter") or {})
    if (
        receipt.get("heads") != audit.get("heads")
        or receipt.get("runtime_smoke") != audit.get("runtime_smoke")
        or any(
            receipt_adapter.get(key) != audit_adapter.get(key)
            for key in (
                "checkpoint_dormant_state",
                "adapter_config_sha256",
                "adapter_tensor_inventory",
                "fit",
                "isolated_optimizer",
                "slot_registry_sha256",
                "immutable_slot_prefix",
                "activation_provenance",
            )
        )
        or receipt_adapter.get("bank_preserved") is not True
        or receipt_adapter.get("checkpoint_runtime_enabled") is not False
        or receipt_adapter.get("checkpoint_training_enabled") is not False
        or receipt_adapter.get("runtime_package_activation_required") is not True
        or receipt_adapter.get("training_enabled") is not True
        or receipt_adapter.get("runtime_enabled") is not True
        or _as_exact_int(
            receipt_adapter.get("epochs_per_rl_update"),
            label="peak-r195 adapter epochs per update",
        )
        != 1
    ):
        raise R241CheckpointReceiptError(
            "peak-r195 receipt does not copy its direct adapter/head evidence"
        )
    training_activation = _identity_from_row(
        receipt_adapter.get("training_activation"),
        label="peak-r195 isolated adapter training activation",
    )
    if (
        dict(audit_adapter.get("fit") or {}).get("activation_receipt")
        != training_activation.as_dict()
    ):
        raise R241CheckpointReceiptError(
            "peak-r195 receipt adapter audit does not bind its host-local activation"
        )
    # Directly redo the expensive checkpoint/model inspection rather than
    # trusting a copied audit object.  This is the critical no-self-assertion
    # boundary the v1 receipt lacked.
    recomputed = _recompute_peak_checkpoint_audit(
        parent=parent,
        policy=policy,
        environment=environment,
        training_activation=training_activation,
    )
    if recomputed.get("audit_fingerprint_sha256") != audit.get("audit_fingerprint_sha256"):
        raise R241CheckpointReceiptError("peak-r195 receipt audit no longer matches checkpoint bytes")
    _parent_identity, parent_payload = _load_checkpoint_payload(
        parent.path,
        label="r195 parent checkpoint",
        expected_sha256=parent.sha256,
        expected_size_bytes=parent.size_bytes,
    )
    if _slot_registry_from_payload(parent_payload, label="r195 parent") != baseline_registry:
        raise R241CheckpointReceiptError("r195 parent slot registry drifted from baseline roster")
    recomputed_migration = audit_append_only_adapter_slot_migration(
        parent_checkpoint=parent.path,
        candidate_checkpoint=parent.path,
        parent_audit=recomputed,
        candidate_audit=recomputed,
    )
    if (
        recomputed_migration.get("status") != "no_slot_change"
        or receipt.get("adapter_slot_migration") != recomputed_migration
    ):
        raise R241CheckpointReceiptError("peak-r195 adapter-slot migration proof drifted")
    learner = _validate_tree(
        learner_matchup_tree,
        label="r195 learner matchup tree",
        expected_sha256=policy.learner_matchup_tree_sha256,
    )
    h10 = _validate_tree(
        h10_matchup_tree,
        label="H10 Marnie matchup tree",
        expected_sha256=policy.h10_matchup_tree_sha256,
        expected_size_bytes=policy.h10_matchup_tree_size_bytes,
    )
    adapter = dict(receipt.get("matchup_adapter") or {})
    _identity_matches(adapter.get("learner_matchup_tree"), learner, label="preservation learner tree")
    _identity_matches(receipt.get("h10_marnie_matchup_tree"), h10, label="preservation H10 tree")
    if receipt.get("direct_environment") != direct_environment:
        raise R241CheckpointReceiptError("peak-r195 receipt direct environment drifted")
    return receipt


def validate_terminal_runtime_receipts(
    *,
    model_receipt_path: Path | str,
    matchup_receipt_path: Path | str,
    r195_parent_checkpoint: Path | str,
    terminal_parent_checkpoint: Path | str,
    terminal_checkpoint: Path | str,
    terminal_refresh_receipt: Path | str,
    terminal_rehearsal_receipt: Path | str,
    learner_matchup_tree: Path | str,
    h10_matchup_tree: Path | str,
    official_cg_root: Path | str,
    environment: Mapping[str, str],
    policy: R241AuditPolicy = DEFAULT_POLICY,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate both terminal v2 receipts against all live file evidence."""

    model_file, model = _read_object(model_receipt_path, label="r241 model runtime activation")
    matchup_file, matchup = _read_object(matchup_receipt_path, label="r241 matchup runtime activation")
    _validate_receipt_fingerprint(model, label="terminal model runtime receipt")
    _validate_receipt_fingerprint(matchup, label="terminal matchup runtime receipt")
    if (
        model.get("schema") != MODEL_RUNTIME_ACTIVATION_SCHEMA
        or matchup.get("schema") != MATCHUP_RUNTIME_ACTIVATION_SCHEMA
        or _as_exact_int(
            model.get("owner_decision_revision"), label="terminal model owner revision"
        )
        != R241_REVISION
        or _as_exact_int(
            matchup.get("owner_decision_revision"), label="terminal matchup owner revision"
        )
        != R241_REVISION
        or model.get("candidate_id") != R241_CANDIDATE_ID
        or matchup.get("candidate_id") != R241_CANDIDATE_ID
        or model.get("status") != "active_peak_r195_non_combo_fusion"
        or matchup.get("status") != "active_direct_policy_only"
        or model.get("derived_not_self_asserted") is not True
        or matchup.get("derived_not_self_asserted") is not True
    ):
        raise R241CheckpointReceiptError("terminal receipt is not generated v2 evidence")
    model_contract, _contract = _validate_receipt_contract(
        model.get("contract"),
        owner_clarification_revision=model.get("owner_clarification_revision"),
        policy=policy,
        label="terminal model receipt",
    )
    matchup_contract, _matchup_contract = _validate_receipt_contract(
        matchup.get("contract"),
        owner_clarification_revision=matchup.get("owner_clarification_revision"),
        policy=policy,
        label="terminal matchup receipt",
    )
    if model_contract != matchup_contract:
        raise R241CheckpointReceiptError("terminal receipt pair binds different owner contracts")
    expected_window_sha = _sha256_text(
        dict(_contract.get("expert_soft_refresh") or {})
        .get("exact_window_evidence_binding", {})
        .get("canonical_manifest_sha256"),
        label="terminal expert window canonical receipt",
    )
    model_window = _validate_receipt_expert_window(
        model.get("expert_window"),
        expected_sha256=expected_window_sha,
        label="terminal model receipt",
    )
    if matchup.get("expert_window") != model.get("expert_window"):
        raise R241CheckpointReceiptError(
            "terminal matchup receipt expert-window identity drifted"
        )
    _validate_receipt_expert_window(
        matchup.get("expert_window"),
        expected_sha256=model_window.sha256,
        label="terminal matchup receipt",
    )
    source_snapshot = _validate_source_snapshot_binding(
        model.get("source_snapshot"), label="terminal model receipt"
    )
    if source_snapshot.get("owner_contract_sha256") != model_contract.sha256:
        raise R241CheckpointReceiptError("terminal source snapshot binds another owner contract")
    if matchup.get("source_snapshot") != model.get("source_snapshot"):
        raise R241CheckpointReceiptError("terminal matchup receipt source snapshot drifted")
    baseline_identity, baseline_registry = _baseline_adapter_roster(policy=policy)
    _identity_matches(
        model.get("baseline_adapter_roster"),
        baseline_identity,
        label="terminal model baseline adapter roster",
    )
    _identity_matches(
        matchup.get("baseline_adapter_roster"),
        baseline_identity,
        label="terminal matchup baseline adapter roster",
    )
    direct_environment = _assert_direct_environment(environment, official_cg_root=official_cg_root)
    root_parent = file_identity(
        r195_parent_checkpoint,
        label="r195 parent checkpoint",
        expected_sha256=policy.parent_sha256,
        expected_size_bytes=policy.parent_size_bytes,
    )
    terminal_parent = file_identity(
        terminal_parent_checkpoint, label="terminal iter_00009 parent checkpoint"
    )
    terminal = file_identity(terminal_checkpoint, label="terminal expert checkpoint")
    _identity_matches(model.get("parent_r195_checkpoint"), root_parent, label="model receipt r195 parent")
    _identity_matches(
        matchup.get("parent_r195_checkpoint"),
        root_parent,
        label="matchup receipt r195 parent",
    )
    if (
        model.get("parent_r195_checkpoint_sha256") != policy.parent_sha256
        or model.get("parent_r195_typed_source_sha256")
        != policy.parent_typed_source_sha256
    ):
        raise R241CheckpointReceiptError(
            "terminal model receipt omits immutable r195 parent/source digests"
        )
    _identity_matches(model.get("terminal_parent_checkpoint"), terminal_parent, label="model receipt terminal parent")
    _identity_matches(model.get("terminal_checkpoint"), terminal, label="model receipt terminal checkpoint")
    _head_map, expected_heads, _routes = _head_contract(policy=policy)
    model_audit = _validate_checkpoint_audit(
        model.get("checkpoint_audit"),
        expected_checkpoint=terminal,
        expected_heads=expected_heads,
        label="terminal model receipt",
    )
    recomputed = audit_checkpoint(
        terminal.path,
        expected_sha256=terminal.sha256,
        expected_size_bytes=terminal.size_bytes,
        policy=policy,
        environment=environment,
    )
    if recomputed.get("audit_fingerprint_sha256") != model_audit.get("audit_fingerprint_sha256"):
        raise R241CheckpointReceiptError("terminal model receipt audit no longer matches checkpoint bytes")
    if (
        model.get("heads") != recomputed.get("heads")
        or model.get("matchup_adapter") != recomputed.get("matchup_adapter")
        or model.get("runtime_smoke") != recomputed.get("runtime_smoke")
        or dict(model.get("runtime_package_activation") or {}).get(
            "matchup_adapters_enabled"
        )
        is not True
        or dict(model.get("runtime_package_activation") or {}).get(
            "checkpoint_remains_dormant"
        )
        is not True
        or model.get("action_selector") != "direct_policy_only"
        or model.get("mcts_enabled") is not False
        or model.get("recursive_turn_planner_enabled") is not False
        or model.get("search_enabled") is not False
        or model.get("belief_assets_enabled") is not False
    ):
        raise R241CheckpointReceiptError(
            "terminal model receipt does not copy its direct checkpoint evidence"
        )
    _parent_identity, parent_payload = _load_checkpoint_payload(
        root_parent.path,
        label="r195 parent checkpoint",
        expected_sha256=root_parent.sha256,
        expected_size_bytes=root_parent.size_bytes,
    )
    if _slot_registry_from_payload(parent_payload, label="r195 parent") != baseline_registry:
        raise R241CheckpointReceiptError("terminal receipt parent roster drifted from baseline")
    root_recomputed = audit_checkpoint(
        root_parent.path,
        expected_sha256=root_parent.sha256,
        expected_size_bytes=root_parent.size_bytes,
        policy=policy,
        environment=environment,
    )
    if model.get("peak_r195_checkpoint_audit") != root_recomputed:
        raise R241CheckpointReceiptError("terminal model receipt peak-r195 audit drifted")
    recomputed_migration = audit_append_only_adapter_slot_migration(
        parent_checkpoint=root_parent.path,
        candidate_checkpoint=terminal.path,
        parent_audit=root_recomputed,
        candidate_audit=recomputed,
    )
    if (
        recomputed_migration.get("status") != "no_slot_change"
        or model.get("adapter_slot_migration") != recomputed_migration
        or matchup.get("adapter_slot_migration") != recomputed_migration
    ):
        raise R241CheckpointReceiptError("terminal adapter-slot migration proof drifted")
    _terminal_identity, terminal_payload = _load_checkpoint_payload(
        terminal.path,
        label="terminal expert checkpoint",
        expected_sha256=terminal.sha256,
        expected_size_bytes=terminal.size_bytes,
    )
    terminal_refresh = _validate_terminal_refresh_boundary(
        terminal_refresh_receipt=terminal_refresh_receipt,
        terminal_rehearsal_receipt=terminal_rehearsal_receipt,
        terminal_parent=terminal_parent,
        terminal_checkpoint=terminal,
        terminal_payload=terminal_payload,
        root_parent_audit=root_recomputed,
        contract_identity=model_contract,
        source_snapshot=source_snapshot,
        baseline_adapter_roster=baseline_identity,
        slot_migration=recomputed_migration,
    )
    if (
        model.get("terminal_refresh") != terminal_refresh
        or matchup.get("terminal_refresh") != terminal_refresh
    ):
        raise R241CheckpointReceiptError("terminal refresh receipt binding drifted")
    learner = _validate_tree(
        learner_matchup_tree,
        label="terminal learner matchup tree",
        expected_sha256=policy.learner_matchup_tree_sha256,
    )
    h10 = _validate_tree(
        h10_matchup_tree,
        label="terminal H10 Marnie matchup tree",
        expected_sha256=policy.h10_matchup_tree_sha256,
        expected_size_bytes=policy.h10_matchup_tree_size_bytes,
    )
    _identity_matches(matchup.get("terminal_checkpoint"), terminal, label="matchup receipt terminal")
    _identity_matches(matchup.get("learner_matchup_tree"), learner, label="matchup receipt learner tree")
    h10_row = dict(matchup.get("h10_training_opponent") or {})
    _identity_matches(h10_row.get("matchup_tree"), h10, label="matchup receipt H10 tree")
    if (
        h10_row.get("direct_policy_only") is not True
        or h10_row.get("mcts_enabled") is not False
        or h10_row.get("recursive_turn_planner_enabled") is not False
        or h10_row.get("search_enabled") is not False
        or matchup.get("checkpoint_audit_fingerprint_sha256")
        != model_audit.get("audit_fingerprint_sha256")
        or matchup.get("model_runtime_activation", {}).get("sha256")
        != file_identity(model_file, label="r241 model runtime activation").sha256
        or matchup.get("matchup_adapter") != recomputed.get("matchup_adapter")
        or matchup.get("runtime_smoke") != recomputed.get("runtime_smoke")
        or matchup.get("action_selector") != "direct_policy_only"
        or matchup.get("mcts_enabled") is not False
        or matchup.get("recursive_turn_planner_enabled") is not False
        or matchup.get("search_enabled") is not False
        or matchup.get("belief_assets_enabled") is not False
        or model.get("direct_environment") != direct_environment
        or matchup.get("direct_environment") != direct_environment
    ):
        raise R241CheckpointReceiptError("terminal matchup receipt is not a direct adapter-on binding")
    return model, matchup
