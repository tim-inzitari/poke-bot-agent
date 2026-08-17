#!/usr/bin/env python3
"""Seal the train-only public support scalar for the frozen H3 critic.

This is an offline artifact builder.  It reads the same sealed chosen-action
train split used by the Prize-plan-v2 sidecar and emits a deterministic lookup
from the current-decision public action signature to ``c3``.  Validation and
evaluation splits are never opened, and an unseen signature is defined as
zero support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.recursive_turn_planner.recent20_overlay import sha256_file  # noqa: E402
from scripts.train_alakazam_prize_plan_v2 import (  # noqa: E402
    H3_SCALE_SUPPORT_SCHEMA,
    PRIZE_AUTHORITY_KEY,
    SEMANTIC_OWNER_REVISION,
    CompleteActionExample,
    PrizePlanV2TrainingError,
    _public_action_signature,
    atomic_write_json,
    canonical_bytes,
    iter_complete_action_examples,
    open_sealed_inputs,
)


C3_SCHEMA = "poke_bot.alakazam_prize_plan_v2_h3_public_support_c3/v1"


class PrizePlanC3Error(ValueError):
    """Raised when a public support artifact cannot be sealed safely."""


def _sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise PrizePlanC3Error(f"{label} must be sha256:<64 lowercase hex>")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise PrizePlanC3Error(f"{label} must be sha256:<64 lowercase hex>") from exc
    if value.lower() != value:
        raise PrizePlanC3Error(f"{label} must be lowercase")
    return value


def _json_sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def public_support_confidence(count: int, unique_signatures: int) -> float:
    """Threshold-free train-support shrinkage; unseen actions remain zero.

    ``sqrt(K)`` is derived solely from the sealed number of public signatures,
    so the curve has no hand-tuned count threshold and cannot inspect a
    validation result or hidden state.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise PrizePlanC3Error("support count must be a nonnegative integer")
    if isinstance(unique_signatures, bool) or not isinstance(unique_signatures, int) or unique_signatures < 1:
        raise PrizePlanC3Error("unique signature count must be positive")
    if count == 0:
        return 0.0
    return count / (count + math.sqrt(unique_signatures))


def build_c3_artifact(
    rows: Iterable[CompleteActionExample],
    *,
    source_binding: Mapping[str, Any],
    historical_contract_sha256: str,
    current_contract_sha256: str,
    h3_scale_support_sha256: str,
    h3_scale_artifact_sha256: str,
    critic_checkpoint_sha256: str,
    validation_receipt_sha256: str,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    opened = 0
    eligible = 0
    for row in rows:
        opened += 1
        if row.plan_masks[1]:
            eligible += 1
            counts[_public_action_signature(row)] += 1
    if not opened or not eligible or not counts:
        raise PrizePlanC3Error("sealed train split supplied no H3 support")
    unique = len(counts)
    table = {
        signature: {
            "train_h3_chosen_action_count": count,
            "c3": public_support_confidence(count, unique),
        }
        for signature, count in sorted(counts.items())
    }
    artifact: dict[str, Any] = {
        "schema": C3_SCHEMA,
        "owner_goal_revision": SEMANTIC_OWNER_REVISION,
        "required_authority": PRIZE_AUTHORITY_KEY,
        "fit_split": "train",
        "full_train_split_consumed": True,
        "validation_or_evaluation_examples_opened": False,
        "recorded_chosen_actions_only": True,
        "public_information_only": True,
        "hidden_information_used": False,
        "signature_definition": {
            "fields": [
                "selected_option_indices",
                "selected_legal_counts",
                "selected_action_programs",
            ],
            "encoding": "canonical_sorted_compact_json_then_sha256",
            "current_decision_only": True,
        },
        "confidence_definition": {
            "formula": "c3=n/(n+sqrt(K))",
            "n": "train_h3_chosen_action_count_for_exact_public_signature",
            "K": "unique_train_h3_public_selected_action_signatures",
            "unseen_signature_c3": 0.0,
            "range": [0.0, 1.0],
            "monotone_in_support_count": True,
            "validation_tuned_threshold": False,
        },
        "train_complete_actions_opened": opened,
        "train_h3_labeled_complete_actions": eligible,
        "unique_public_selected_action_signatures": unique,
        "minimum_signature_count": min(counts.values()),
        "maximum_signature_count": max(counts.values()),
        "support_table": table,
        "source_binding": dict(source_binding),
        "source_binding_sha256": _json_sha(dict(source_binding)),
        "historical_training_contract_sha256": _sha(
            historical_contract_sha256, label="historical contract SHA-256"
        ),
        "current_authority_contract_sha256": _sha(
            current_contract_sha256, label="current contract SHA-256"
        ),
        "h3_scale_support_sha256": _sha(h3_scale_support_sha256, label="H3 scale file SHA-256"),
        "h3_scale_artifact_sha256": _sha(h3_scale_artifact_sha256, label="H3 scale artifact SHA-256"),
        "critic_checkpoint_sha256": _sha(critic_checkpoint_sha256, label="critic checkpoint SHA-256"),
        "validation_receipt_sha256": _sha(validation_receipt_sha256, label="validation receipt SHA-256"),
        "actor_activation": False,
        "activation_eligible": False,
        "requires_ess_clip_noninterference_rollback_and_paired_eval_receipts": True,
    }
    artifact["artifact_sha256"] = _json_sha(artifact)
    return artifact


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PrizePlanC3Error(f"{label} must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PrizePlanC3Error(f"{label} must contain an object")
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--training-view-sha256", required=True)
    parser.add_argument("--target-view", type=Path, required=True)
    parser.add_argument("--target-view-sha256", required=True)
    parser.add_argument("--historical-contract", type=Path, required=True)
    parser.add_argument("--historical-contract-sha256", required=True)
    parser.add_argument("--current-contract", type=Path, required=True)
    parser.add_argument("--current-contract-sha256", required=True)
    parser.add_argument("--h3-scale-support", type=Path, required=True)
    parser.add_argument("--h3-scale-support-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--validation-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    current = _load_json(args.current_contract, label="current contract")
    if sha256_file(args.current_contract) != _sha(args.current_contract_sha256, label="current contract SHA-256"):
        raise PrizePlanC3Error("current contract digest mismatch")
    authority = current.get(PRIZE_AUTHORITY_KEY)
    if not isinstance(authority, Mapping) or authority.get("owner_goal_revision") != SEMANTIC_OWNER_REVISION:
        raise PrizePlanC3Error("current contract lacks embedded revision-23 critic authority")
    for path, digest, label in (
        (args.historical_contract, args.historical_contract_sha256, "historical contract"),
        (args.h3_scale_support, args.h3_scale_support_sha256, "H3 scale support"),
        (args.checkpoint, args.checkpoint_sha256, "critic checkpoint"),
        (args.validation_receipt, args.validation_receipt_sha256, "validation receipt"),
    ):
        if sha256_file(path) != _sha(digest, label=f"{label} SHA-256"):
            raise PrizePlanC3Error(f"{label} digest mismatch")
    scale = _load_json(args.h3_scale_support, label="H3 scale support")
    if (
        scale.get("schema") != H3_SCALE_SUPPORT_SCHEMA
        or scale.get("full_train_split_consumed") is not True
        or scale.get("fit_split") != "train"
        or scale.get("actor_activation") is not False
    ):
        raise PrizePlanC3Error("H3 scale support is not the full frozen train-only artifact")

    sealed_args = SimpleNamespace(
        contract=args.historical_contract,
        contract_sha256=args.historical_contract_sha256,
        training_view=args.training_view,
        training_view_sha256=args.training_view_sha256,
        target_view=args.target_view,
        target_view_sha256=args.target_view_sha256,
        target_set_root=None,
        target_manifest=None,
        target_manifest_sha256=None,
        target_set_receipt=None,
        target_set_receipt_sha256=None,
        test_mode=False,
        test_allow_noncanonical_split=False,
        test_skip_input_shard_sha256=False,
    )
    dataset, targets, _split_days, source_binding = open_sealed_inputs(sealed_args)
    artifact = build_c3_artifact(
        iter_complete_action_examples(dataset, targets, split="train"),
        source_binding=source_binding,
        historical_contract_sha256=args.historical_contract_sha256,
        current_contract_sha256=args.current_contract_sha256,
        h3_scale_support_sha256=args.h3_scale_support_sha256,
        h3_scale_artifact_sha256=str(scale.get("artifact_sha256")),
        critic_checkpoint_sha256=args.checkpoint_sha256,
        validation_receipt_sha256=args.validation_receipt_sha256,
    )
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise PrizePlanC3Error("c3 output is create-only")
    digest = atomic_write_json(output, artifact)
    return {"path": str(output), "sha256": digest, "artifact_sha256": artifact["artifact_sha256"]}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        print(json.dumps(run(args), sort_keys=True), flush=True)
    except (PrizePlanC3Error, PrizePlanV2TrainingError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
