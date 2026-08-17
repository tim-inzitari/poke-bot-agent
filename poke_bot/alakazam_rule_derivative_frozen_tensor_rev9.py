"""Build the revision-9 composite candidate initialization and frozen-tensor receipt.

This is a non-activating, create-only producer.  It reconstructs the r274
architecture on CPU, fills inherited tensors with the revision-7 r195-first
source order, adds only the census-supported public semantic projection, and
proves that every inherited tensor survives candidate serialization/reload
bit-for-bit.  It never controls a service or starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


GOAL_REVISION = 9
GOAL_PATH = "goals/alakazam-elmo-rule-derivative/GOAL.md"
CONTRACT_PATH = "goals/alakazam-elmo-rule-derivative/contract.json"
GOAL_SHA256 = "sha256:8908c4e8bcf36a089ba7f230c137e259f024125807bdb04b03d77483f533c223"
CONTRACT_SHA256 = "sha256:fd5460fca1ebab8ae0881de33ed7467905b8dbc2839e859a1aad89db83cd5cf8"
R195_SHA256 = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
R274_SHA256 = "sha256:d5ad45faa1d05dd3e2e62ffd2b25867b6ce0285a918aefa622891880f4902b05"
COMPOSITE_SOURCE_MAP_SHA256 = "sha256:8d7b741cf6704840d13fc7d58b1fae0b1706abcac761d28f9b6b47097b518ae3"
COMPOSITE_SOURCE_MAP_RECEIPT_SHA256 = "sha256:cbc8b97d8f81725761f69843867bafdc700fd06267ab4f849ca68c08265a200c"

FROZEN_RECEIPT_SCHEMA = "poke_bot.alakazam_rule_derivative_frozen_tensor_receipt/v1"
CANDIDATE_BUNDLE_SCHEMA = "poke_bot.alakazam_rule_derivative_composite_candidate_initialization/v1"
ARCHITECTURE_SCHEMA = "poke_bot.alakazam_rule_derivative_candidate_architecture/v1"
EVIDENCE_SCHEMA = "poke_bot.alakazam_rule_derivative_frozen_tensor_evidence/v1"
SIDECAR_PREFIX = "public_rule_semantic_projection."
SIDECAR_SEED = 298_009
FROZEN_MODULE_NAMES = (
    "neural_backbone",
    "existing_heads",
    "fusion",
    "own_deck_routes",
    "matchup_adapters",
    "card2vec",
)
SUPPORTED_BRANCHES = ("public_rule_semantic_projection",)
UNSUPPORTED_BRANCHES = (
    "public_rule_metadata_residual",
    "r298_repaired_auxiliary_heads",
    "eight_checklist_provenance_gates",
)


class FrozenTensorRev9Error(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(canonical_bytes(value))


def _verify_regular(path: Path, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FrozenTensorRev9Error(f"missing regular input: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise FrozenTensorRev9Error(
            f"input digest mismatch for {path}: {observed} != {expected_sha256}"
        )


def _tensor_raw_bytes(value: Any) -> bytes:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - execution host supplies torch
        raise FrozenTensorRev9Error("Torch is required") from exc
    if not isinstance(value, torch.Tensor):
        raise FrozenTensorRev9Error("state dictionary contains a non-tensor value")
    tensor = value.detach().cpu().contiguous()
    # ``view(dtype)`` rejects a zero-dimensional multi-byte scalar.  Flatten
    # first so persistent scalar buffers participate in the same exact byte
    # stream as every other tensor.
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_inventory(state: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name in sorted(state):
        tensor = state[name]
        rows.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": [int(item) for item in tensor.shape],
                "bytes_sha256": _sha256_bytes(_tensor_raw_bytes(tensor)),
            }
        )
    inventory = {"tensor_count": len(rows), "tensors": rows}
    inventory["inventory_sha256"] = _canonical_sha256(inventory)
    return inventory


def tensor_stream_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        metadata = canonical_bytes(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": [int(item) for item in tensor.shape],
            }
        )
        raw = _tensor_raw_bytes(tensor)
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return "sha256:" + digest.hexdigest()


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FrozenTensorRev9Error(f"output exists: {path}")
    body = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(body):
            offset += os.write(descriptor, body[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256_file(path)


def _write_torch_create_only(path: Path, value: Mapping[str, Any]) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise FrozenTensorRev9Error("Torch is required") from exc
    if path.exists() or path.is_symlink():
        raise FrozenTensorRev9Error(f"output exists: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            torch.save(dict(value), stream)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return sha256_file(path)


def _load_checkpoint(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    _verify_regular(path, expected_sha256)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise FrozenTensorRev9Error("Torch is required") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state_dict"), Mapping):
        raise FrozenTensorRev9Error(f"checkpoint lacks a model state: {path}")
    return payload


def _build_base_model(model_config: Mapping[str, Any], state: Mapping[str, Any]) -> Any:
    try:
        import torch
        from poke_bot import config, features
        from poke_bot.model import build_model
    except ImportError as exc:  # pragma: no cover
        raise FrozenTensorRev9Error("model runtime is unavailable") from exc
    known = set(config.ModelConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = sorted(set(model_config) - known)
    if unknown:
        raise FrozenTensorRev9Error("checkpoint model_config has unknown fields: " + ", ".join(unknown))
    cfg = config.ModelConfig(**dict(model_config))
    try:
        aux_classes = int(state["aux_head.3.weight"].shape[0])
        encoder_vocab = int(state["card2vec.board_kind"].shape[0])
        decoder_vocab = int(state["card2vec.option_kind"].shape[0])
        belief_widths = {
            int(state[name].shape[0])
            for name in ("opp_hand_head.weight", "opp_remainder_head.weight")
            if name in state
        }
        if len(belief_widths) > 1:
            raise FrozenTensorRev9Error("belief head widths disagree")
        belief_vocab = (
            next(iter(belief_widths))
            if belief_widths
            else int(state["card2vec.card_emb.weight"].shape[0])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FrozenTensorRev9Error("checkpoint lacks architecture-defining tensors") from exc
    model = build_model(
        cfg,
        device=torch.device("cpu"),
        aux_archetype_classes=aux_classes,
        encoder_vocab=encoder_vocab,
        decoder_vocab=decoder_vocab,
        belief_card_vocab=belief_vocab,
    )
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise FrozenTensorRev9Error("strict base-model load did not close")
    model.eval()
    return model


def _fixture() -> tuple[Any, Any, dict[str, Any]]:
    from poke_bot import features

    board = features.SparseVector()
    for _ in range(features.NUM_BOARD_TOKENS):
        board.word_start()
    options = features.SparseVector()
    for index in (3, 10):
        options.word_start()
        options.add(index, 1.0)
    payload = {
        "board": {"index": board.index, "value": board.value, "offset": board.offset},
        "options": {"index": options.index, "value": options.value, "offset": options.offset},
        "n_options": [2],
    }
    return board, options, payload


def _forward_evidence(model: Any) -> tuple[Any, Any, str, str, str]:
    import torch

    board, options, fixture = _fixture()
    with torch.no_grad():
        output = model.forward(board, options, n_options=[2])
    logits = output["policy_logits"].detach().cpu().contiguous()
    spatial = output["spatial_memory"].detach().cpu().contiguous()
    input_schema = {
        "feature_schema_version": int(model._feature_schema_version.detach().cpu()),
        "encoder_vocab": int(model.encoder_vocab),
        "decoder_vocab": int(model.decoder_vocab),
        "dense_card2vec": bool(model.dense_card2vec),
        "card2vec_board_factor_table_shape": list(model.card2vec.board_kind.shape),
        "card2vec_option_factor_table_shape": list(model.card2vec.option_kind.shape),
        "fixture_schema": {
            "board_fields": ["index", "value", "offset"],
            "option_fields": ["index", "value", "offset"],
        },
    }
    return (
        logits,
        spatial,
        _canonical_sha256(input_schema),
        _canonical_sha256(fixture),
        _sha256_bytes(_tensor_raw_bytes(spatial)),
    )


def _source_map_matches(
    source_map: Mapping[str, Any], sources: Mapping[str, str], base_state: Mapping[str, Any]
) -> None:
    rows = source_map.get("tensors")
    if not isinstance(rows, list):
        raise FrozenTensorRev9Error("composite source map is malformed")
    expected = {
        str(row["name"]): (
            "r195" if row["source"] == "r195_exact_name_shape_dtype" else "r274_architecture_addition"
        )
        for row in rows
    }
    if expected != dict(sources) or set(base_state) != set(expected):
        raise FrozenTensorRev9Error("sealed source map does not match assembled composite state")


def _required_frozen_receipt_fields(contract: Mapping[str, Any]) -> set[str]:
    try:
        values = contract["inzi_blackwell_training_handoff"]["receipt_contract"]["frozen_tensors"]["required_fields"]
    except (KeyError, TypeError) as exc:
        raise FrozenTensorRev9Error("contract frozen-tensor receipt declaration is missing") from exc
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise FrozenTensorRev9Error("contract frozen-tensor field inventory is malformed")
    return set(values)


def build_artifacts(
    *,
    repo_root: Path,
    r195_checkpoint: Path,
    r274_checkpoint: Path,
    source_map_path: Path,
    source_map_receipt_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    try:
        import torch
        from poke_bot.alakazam_r195_r274_composite_warmstart_r307 import (
            assemble_composite_warmstart,
        )
        from poke_bot.alakazam_rule_derivative_model_r298 import (
            R298PublicRuleSemanticProjection,
            R298SemanticProjectionConfig,
        )
    except ImportError as exc:  # pragma: no cover
        raise FrozenTensorRev9Error("candidate dependencies are unavailable") from exc

    _verify_regular(repo_root / GOAL_PATH, GOAL_SHA256)
    _verify_regular(repo_root / CONTRACT_PATH, CONTRACT_SHA256)
    _verify_regular(source_map_path, COMPOSITE_SOURCE_MAP_SHA256)
    _verify_regular(source_map_receipt_path, COMPOSITE_SOURCE_MAP_RECEIPT_SHA256)
    contract = json.loads((repo_root / CONTRACT_PATH).read_text())
    source_map = json.loads(source_map_path.read_text())
    source_map_receipt = json.loads(source_map_receipt_path.read_text())
    if source_map_receipt.get("per_tensor_source_map_sha256") != COMPOSITE_SOURCE_MAP_SHA256:
        raise FrozenTensorRev9Error("source-map receipt does not bind the source map")

    r195 = _load_checkpoint(r195_checkpoint, R195_SHA256)
    r274 = _load_checkpoint(r274_checkpoint, R274_SHA256)
    r195_state = r195["model_state_dict"]
    r274_state = r274["model_state_dict"]
    if not isinstance(r274.get("model_config"), Mapping):
        raise FrozenTensorRev9Error("r274 checkpoint lacks model_config")
    d_model = int(r274["model_config"].get("d_model", 0))
    if d_model <= 0:
        raise FrozenTensorRev9Error("r274 d_model is invalid")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SIDECAR_SEED)
        sidecar = R298PublicRuleSemanticProjection(
            R298SemanticProjectionConfig(d_model=d_model)
        ).cpu()
    sidecar_state = {
        SIDECAR_PREFIX + name: value.detach().cpu().clone()
        for name, value in sidecar.state_dict().items()
    }
    target_state = {name: value.detach().cpu().clone() for name, value in r274_state.items()}
    target_state.update(sidecar_state)
    composite = assemble_composite_warmstart(
        target_state=target_state,
        r195_state=r195_state,
        r274_state=r274_state,
        new_parameter_prefixes=(SIDECAR_PREFIX.rstrip("."),),
        r195_checkpoint_sha256=R195_SHA256,
        r274_checkpoint_sha256=R274_SHA256,
    )
    base_state = {name: value for name, value in composite.state_dict.items() if not name.startswith(SIDECAR_PREFIX)}
    initialized_sidecar_state = {
        name: value for name, value in composite.state_dict.items() if name.startswith(SIDECAR_PREFIX)
    }
    _source_map_matches(source_map, {k: v for k, v in composite.source_map.items() if not k.startswith(SIDECAR_PREFIX)}, base_state)
    new_names = sorted(initialized_sidecar_state)
    if set(new_names) != {SIDECAR_PREFIX + name for name, _ in sidecar.named_parameters()}:
        raise FrozenTensorRev9Error("new trainable parameter inventory is not exactly the semantic sidecar")
    zero_keys = {
        SIDECAR_PREFIX + "semantic_projection.2.weight",
        SIDECAR_PREFIX + "semantic_projection.2.bias",
        SIDECAR_PREFIX + "logit_projection.2.weight",
        SIDECAR_PREFIX + "logit_projection.2.bias",
    }
    if not zero_keys <= set(initialized_sidecar_state) or any(
        torch.count_nonzero(initialized_sidecar_state[name]).item() != 0 for name in zero_keys
    ):
        raise FrozenTensorRev9Error("semantic sidecar final additions are not exact zero")

    before_inventory = tensor_inventory(base_state)
    before_bytes_sha = tensor_stream_sha256(base_state)
    new_inventory = tensor_inventory(initialized_sidecar_state)
    base_model = _build_base_model(r274["model_config"], base_state)
    logits_before, card2vec_before, input_schema_sha, input_values_sha, card2vec_output_sha = _forward_evidence(base_model)
    off_logits = sidecar.apply_to_logits(
        logits_before,
        None,
        None,
        runtime_enabled=False,
        gate=0.0,
    )
    if off_logits is not logits_before:
        raise FrozenTensorRev9Error("disabled sidecar did not return baseline logits by object identity")

    output_root.mkdir(parents=True, exist_ok=False)
    bundle_path = output_root / "candidate-initialization.pt"
    bundle = {
        "schema": CANDIDATE_BUNDLE_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_gateway_sha256": GOAL_SHA256,
        "goal_contract_sha256": CONTRACT_SHA256,
        "r195_checkpoint_sha256": R195_SHA256,
        "r274_checkpoint_sha256": R274_SHA256,
        "composite_source_map_sha256": COMPOSITE_SOURCE_MAP_SHA256,
        "model_config": dict(r274["model_config"]),
        "base_model_state_dict": base_state,
        "public_rule_semantic_projection_config": asdict(sidecar.config),
        "public_rule_semantic_projection_state_dict": {
            name.removeprefix(SIDECAR_PREFIX): value for name, value in initialized_sidecar_state.items()
        },
        "eligible_trainable_branches": list(SUPPORTED_BRANCHES),
        "unsupported_zero_inert_branches": list(UNSUPPORTED_BRANCHES),
        "runtime_wired": False,
        "training_eligible_before_activation": False,
    }
    bundle_sha = _write_torch_create_only(bundle_path, bundle)
    loaded = torch.load(bundle_path, map_location="cpu", weights_only=False)
    loaded_base = loaded["base_model_state_dict"]
    loaded_sidecar = {
        SIDECAR_PREFIX + name: value
        for name, value in loaded["public_rule_semantic_projection_state_dict"].items()
    }
    after_inventory = tensor_inventory(loaded_base)
    after_bytes_sha = tensor_stream_sha256(loaded_base)
    if before_inventory != after_inventory or before_bytes_sha != after_bytes_sha:
        raise FrozenTensorRev9Error("inherited tensors changed across candidate serialization/load")
    if tensor_inventory(initialized_sidecar_state) != tensor_inventory(loaded_sidecar):
        raise FrozenTensorRev9Error("new sidecar tensors changed across candidate serialization/load")

    reloaded_model = _build_base_model(loaded["model_config"], loaded_base)
    reloaded_sidecar = R298PublicRuleSemanticProjection(
        R298SemanticProjectionConfig(**loaded["public_rule_semantic_projection_config"])
    ).cpu()
    reloaded_sidecar.load_state_dict(loaded["public_rule_semantic_projection_state_dict"], strict=True)
    logits_after, card2vec_after, input_schema_after, input_values_after, card2vec_output_after = _forward_evidence(reloaded_model)
    if (
        input_schema_after != input_schema_sha
        or input_values_after != input_values_sha
        or card2vec_output_after != card2vec_output_sha
        or not torch.equal(card2vec_before, card2vec_after)
        or not torch.equal(logits_before, logits_after)
    ):
        raise FrozenTensorRev9Error("Card2Vec or baseline policy output changed after candidate load")
    reloaded_off = reloaded_sidecar.apply_to_logits(
        logits_after, None, None, runtime_enabled=False, gate=0.0
    )
    if reloaded_off is not logits_after or not torch.equal(reloaded_off, logits_before):
        raise FrozenTensorRev9Error("layer-off baseline logits are not bit-identical")

    architecture = {
        "schema": ARCHITECTURE_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_gateway_sha256": GOAL_SHA256,
        "goal_contract_sha256": CONTRACT_SHA256,
        "base_architecture": "exact_r274_model_config_and_state_shape",
        "d_model": d_model,
        "composite_source_precedence": ["r195_exact_name_shape", "r274_name_absent_from_r195", "explicit_new_initialization"],
        "composite_source_map_sha256": COMPOSITE_SOURCE_MAP_SHA256,
        "candidate_initialization_sha256": bundle_sha,
        "inherited_tensor_count": len(base_state),
        "new_trainable_parameter_names": new_names,
        "new_trainable_parameter_count": len(new_names),
        "new_module": {
            "class": "R298PublicRuleSemanticProjection",
            "state_prefix": SIDECAR_PREFIX,
            "config": asdict(sidecar.config),
            "seed": SIDECAR_SEED,
            "zero_output_parameter_names": sorted(zero_keys),
        },
        "eligible_trainable_branches": list(SUPPORTED_BRANCHES),
        "unsupported_zero_inert_branches": list(UNSUPPORTED_BRANCHES),
        "card2vec_replaced_or_rewired": False,
        "runtime_wired": False,
        "training_eligible_before_activation": False,
    }
    architecture_path = output_root / "candidate-architecture.json"
    architecture_sha = _write_json_create_only(architecture_path, architecture)

    frozen_receipt = {
        "schema": FROZEN_RECEIPT_SCHEMA,
        "goal_contract_path": CONTRACT_PATH,
        "goal_contract_sha256": CONTRACT_SHA256,
        "goal_revision": GOAL_REVISION,
        "parent_checkpoint_sha256": R274_SHA256,
        "frozen_module_names": list(FROZEN_MODULE_NAMES),
        "frozen_parameter_and_buffer_inventory_sha256": before_inventory["inventory_sha256"],
        "frozen_parameter_and_buffer_bytes_sha256_before": before_bytes_sha,
        "frozen_parameter_and_buffer_bytes_sha256_after_load": after_bytes_sha,
        "card2vec_input_schema_sha256": input_schema_sha,
        "card2vec_output_parity_sha256": card2vec_output_sha,
        "card2vec_parameter_and_buffer_bit_identity": True,
        "backbone_existing_heads_fusion_own_deck_and_matchup_adapter_bit_identity": True,
        "trainable_parameter_names": new_names,
        "trainable_parameter_inventory_sha256": new_inventory["inventory_sha256"],
        "optimizer_parameter_names": new_names,
        "frozen_parameters_absent_from_optimizer": True,
        "new_path_zero_initialized": True,
        "layer_off_bit_identical_baseline_logits": True,
        "validated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if set(frozen_receipt) != _required_frozen_receipt_fields(contract):
        raise FrozenTensorRev9Error("frozen-tensor receipt does not exactly match the contract field inventory")
    frozen_receipt_path = output_root / "frozen-tensor-receipt.json"
    frozen_receipt_sha = _write_json_create_only(frozen_receipt_path, frozen_receipt)

    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "goal_revision": GOAL_REVISION,
        "goal_gateway_sha256": GOAL_SHA256,
        "goal_contract_sha256": CONTRACT_SHA256,
        "candidate_initialization_path": str(bundle_path),
        "candidate_initialization_sha256": bundle_sha,
        "candidate_architecture_sha256": architecture_sha,
        "composite_source_map_sha256": COMPOSITE_SOURCE_MAP_SHA256,
        "composite_source_map_receipt_sha256": COMPOSITE_SOURCE_MAP_RECEIPT_SHA256,
        "inherited_tensor_inventory": before_inventory,
        "new_tensor_inventory": new_inventory,
        "card2vec_input_values_sha256": input_values_sha,
        "card2vec_output_sha256": card2vec_output_sha,
        "baseline_policy_logits_sha256": _sha256_bytes(_tensor_raw_bytes(logits_before)),
        "layer_off_policy_logits_sha256": _sha256_bytes(_tensor_raw_bytes(reloaded_off)),
        "layer_off_returned_same_tensor_object": True,
        "candidate_reload_strict": True,
        "all_inherited_tensors_bit_identical_after_load": True,
        "all_new_tensors_bit_identical_after_load": True,
        "new_path_zero_initialized": True,
        "optimizer_scope": "public_rule_semantic_projection_only",
        "optimizer_parameter_names": new_names,
        "unsupported_branches_present_in_candidate": False,
        "service_control_performed": False,
        "training_or_activation_performed": False,
    }
    evidence_path = output_root / "frozen-tensor-evidence.json"
    evidence_sha = _write_json_create_only(evidence_path, evidence)
    complete = {
        "schema": "poke_bot.alakazam_rule_derivative_frozen_tensor_completion/v1",
        "status": "passed_nonactivating_composite_candidate_initialization_and_frozen_tensor_validation",
        "goal_revision": GOAL_REVISION,
        "candidate_initialization_sha256": bundle_sha,
        "candidate_architecture_sha256": architecture_sha,
        "frozen_tensor_receipt_sha256": frozen_receipt_sha,
        "frozen_tensor_evidence_sha256": evidence_sha,
        "service_control_performed": False,
        "training_or_activation_performed": False,
    }
    complete_path = output_root / "COMPLETE.json"
    complete_sha = _write_json_create_only(complete_path, complete)
    return {
        "output_root": str(output_root),
        "candidate_initialization_sha256": bundle_sha,
        "candidate_architecture_sha256": architecture_sha,
        "frozen_tensor_receipt_sha256": frozen_receipt_sha,
        "frozen_tensor_evidence_sha256": evidence_sha,
        "completion_receipt_sha256": complete_sha,
        "inherited_tensor_count": len(base_state),
        "new_trainable_parameter_count": len(new_names),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--r195-checkpoint", type=Path, required=True)
    parser.add_argument("--r274-checkpoint", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--source-map-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_artifacts(
        repo_root=args.repo_root.resolve(),
        r195_checkpoint=args.r195_checkpoint.resolve(),
        r274_checkpoint=args.r274_checkpoint.resolve(),
        source_map_path=args.source_map.resolve(),
        source_map_receipt_path=args.source_map_receipt.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FrozenTensorRev9Error",
    "build_artifacts",
    "canonical_bytes",
    "sha256_file",
    "tensor_inventory",
    "tensor_stream_sha256",
]
