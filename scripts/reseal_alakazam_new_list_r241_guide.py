#!/usr/bin/env python3
"""Rebuild and checksum-seal the four r241 Alakazam guide artifacts.

This is an offline, training-preparation-only operation.  It derives the
guide-readiness receipt from the exact owner deck/guide/module/write-up,
delegates the strategic measurements to the canonical curriculum
materializer, and atomically publishes only the four fixed r241 artifacts.
It never starts training, changes a runtime selector, or submits a package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import alakazam_new_list_heuristics as guide

OWNER_CONTRACT = ROOT / "state/alakazam-new-list-direct-policy-r241.json"
GUIDE_CONTRACT = ROOT / "config/deck_guides/alakazam-new-list-direct-r241.yaml"
DECK = ROOT / "decks/archetype-samples/alakazam-new-list-direct-r241.csv"
TEACHER = ROOT / "poke_bot/alakazam_new_list_heuristics.py"
WRITEUP = ROOT / "docs/deck_guides/alakazam-new-list-direct-r241-expert-brief.md"
TRAINING_IMPLEMENTATION = ROOT / "poke_bot/train.py"

READINESS = ROOT / "state/alakazam-new-list-direct-r241-guide-readiness.json"
HEAD_ROLE_MAP = ROOT / "state/alakazam-new-list-direct-r241-strategic-head-roles.json"
CURRICULUM = ROOT / "state/alakazam-new-list-direct-r241-strategic-curriculum.json"
VALIDATION = (
    ROOT / "state/alakazam-new-list-direct-r241-strategic-curriculum-validation.json"
)

SPECIALIST_ID = "alakazam"
CANDIDATE_ID = "alakazam-new-list-direct-policy-r241"
DECK_ID = "alakazam-new-list-direct-r241"
GUIDE_VERSION = "powerful-hand-new-list-r241-v1"
READINESS_CREATED_AT_UTC = "2026-08-10T23:55:00Z"
ORDINARY_GUIDE_LOSS_WEIGHT = 0.05
EXPERT_REFRESH_GUIDE_LOSS_WEIGHT = 0.0
TRAINING_MODE = "strategic_directional_v2"
GUIDE_ROWS_SCOPE = (
    "one deterministic exact-deck canary stage; this is a scorer-readiness "
    "count, not a future training-corpus cardinality"
)
EXPERT_ZERO_REASON = (
    "historical expert rows do not carry the exact new 60-card multiset"
)
READINESS_SCHEMA = "poke_bot.alakazam_new_list_direct_policy_guide_readiness/v1"


class R241GuideResealError(RuntimeError):
    """The four r241 guide artifacts cannot be safely resealed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise R241GuideResealError(f"unreadable JSON source: {path}") from exc
    if not isinstance(value, dict):
        raise R241GuideResealError(f"JSON object required: {path}")
    return value


def _require_equal(*, label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise R241GuideResealError(
            f"{label} mismatch: actual={actual!r} expected={expected!r}"
        )


def _normalize_sha256(value: str) -> str:
    candidate = str(value or "").strip().lower()
    if candidate.startswith("sha256:"):
        candidate = candidate.removeprefix("sha256:")
    if len(candidate) != 64 or any(
        char not in "0123456789abcdef" for char in candidate
    ):
        raise R241GuideResealError(
            "expected SHA-256 must contain exactly 64 hex digits"
        )
    return "sha256:" + candidate


def _cards() -> list[int]:
    try:
        cards = [
            int(line.strip())
            for line in DECK.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, ValueError) as exc:
        raise R241GuideResealError(f"invalid exact deck: {DECK}") from exc
    if len(cards) != 60:
        raise R241GuideResealError(
            f"r241 exact deck must contain 60 cards, found {len(cards)}"
        )
    return cards


def _ordered_cards_sha256(cards: Sequence[int]) -> str:
    raw = json.dumps([int(card) for card in cards], separators=(",", ":")).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _multiset_sha256(cards: Sequence[int]) -> str:
    raw = json.dumps(sorted(int(card) for card in cards), separators=(",", ":")).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _canary_player(*, hand: Sequence[int] = (), active_id: int = 741) -> dict[str, Any]:
    return {
        "active": [{"id": int(active_id), "hp": 60, "maxHp": 60, "energyCards": []}],
        "bench": [{"id": 741, "hp": 60, "maxHp": 60, "energyCards": []}],
        "deckCount": 30,
        "discard": [],
        "prize": [None] * 6,
        "hand": [{"id": int(card_id)} for card_id in hand],
        "handCount": len(hand),
    }


def _validate_exact_deck_canary(cards: Sequence[int]) -> int:
    observation = {
        "current": {
            "yourIndex": 0,
            "players": [
                _canary_player(hand=[guide.BATTLE_CAGE]),
                _canary_player(active_id=guide.FROSLASS),
            ],
            "stadium": [],
            "looking": [],
        },
        "select": {
            "context": 0,
            "option": [{"type": 7, "index": 0}, {"type": 14}],
            "minCount": 1,
            "maxCount": 1,
        },
    }
    scores = guide.guide_scores(
        observation,
        [[0], [1]],
        deck=cards,
        force_enabled=True,
    )
    if (
        scores is None
        or len(scores) != 2
        or any(not math.isfinite(float(value)) for value in scores)
        or max(scores) - min(scores) < 0.25
    ):
        raise R241GuideResealError(
            "exact-deck deterministic guide canary is not safely nonflat"
        )
    nonexact = [*cards[:-1], -1]
    if (
        guide.guide_scores(
            observation,
            [[0], [1]],
            deck=nonexact,
            force_enabled=True,
        )
        is not None
    ):
        raise R241GuideResealError("guide exact-deck gate accepted a foreign multiset")
    return 1


def build_readiness_payload() -> dict[str, Any]:
    """Derive the readiness receipt without trusting the previous receipt."""

    source = _read_json(OWNER_CONTRACT)
    try:
        contract = yaml.safe_load(GUIDE_CONTRACT.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise R241GuideResealError(
            f"unreadable guide contract: {GUIDE_CONTRACT}"
        ) from exc
    if not isinstance(contract, dict):
        raise R241GuideResealError("guide contract must be a YAML mapping")

    _require_equal(
        label="owner contract schema",
        actual=source.get("schema"),
        expected="poke_bot.alakazam_new_list_direct_policy_r241/v1",
    )
    _require_equal(
        label="owner candidate",
        actual=source.get("candidate_id"),
        expected=CANDIDATE_ID,
    )
    _require_equal(
        label="guide specialist",
        actual=contract.get("specialist_id"),
        expected=SPECIALIST_ID,
    )
    _require_equal(
        label="guide version",
        actual=contract.get("guide_version"),
        expected=GUIDE_VERSION,
    )
    _require_equal(
        label="teacher module selector",
        actual=contract.get("teacher_module"),
        expected="poke_bot.alakazam_new_list_heuristics",
    )
    _require_equal(
        label="teacher implementation version",
        actual=guide.GUIDE_VERSION,
        expected=GUIDE_VERSION,
    )

    cards = _cards()
    ordered_sha256 = _ordered_cards_sha256(cards)
    multiset_sha256 = _multiset_sha256(cards)
    deck_sha256 = _sha256(DECK)
    guide_sha256 = _sha256(GUIDE_CONTRACT)
    teacher_sha256 = _sha256(TEACHER)
    writeup_sha256 = _sha256(WRITEUP)
    try:
        writeup_word_count = len(WRITEUP.read_text(encoding="utf-8").split())
    except (OSError, UnicodeError) as exc:
        raise R241GuideResealError(f"unreadable guide write-up: {WRITEUP}") from exc
    if writeup_word_count <= 0 or writeup_word_count > 10_000:
        raise R241GuideResealError(
            f"guide write-up word count is out of bounds: {writeup_word_count}"
        )

    source_deck = dict(source.get("exact_deck") or {})
    source_guide = dict(source.get("owner_guide") or {})
    representative = dict(contract.get("representative_binding") or {})
    expert_writeup = dict(contract.get("expert_writeup") or {})
    policy_target = dict(contract.get("policy_target") or {})
    curriculum = dict(contract.get("strategic_curriculum") or {})
    expected_deck = {
        "path": "decks/archetype-samples/alakazam-new-list-direct-r241.csv",
        "file_sha256": deck_sha256,
        "ordered_cards_sha256": ordered_sha256,
        "canonical_multiset_sha256": multiset_sha256,
        "card_count": 60,
    }
    for key, expected in expected_deck.items():
        _require_equal(
            label=f"typed source exact_deck.{key}",
            actual=source_deck.get(key),
            expected=expected,
        )
        _require_equal(
            label=f"guide representative_binding.{key}",
            actual=representative.get(key),
            expected=expected,
        )
    _require_equal(
        label="typed source exact deck id",
        actual=source_deck.get("deck_id"),
        expected=DECK_ID,
    )
    _require_equal(
        label="guide exact deck id",
        actual=representative.get("deck_id"),
        expected=DECK_ID,
    )
    _require_equal(
        label="teacher exact deck",
        actual=tuple(cards),
        expected=tuple(guide.EXACT_DECK),
    )
    _require_equal(
        label="teacher canonical multiset",
        actual=guide.CANONICAL_MULTISET_SHA256,
        expected=multiset_sha256,
    )

    source_guide_checks = {
        "guide_contract": "config/deck_guides/alakazam-new-list-direct-r241.yaml",
        "guide_contract_sha256": guide_sha256,
        "guide_version": GUIDE_VERSION,
        "human_guide": "docs/deck_guides/alakazam-new-list-direct-r241-expert-brief.md",
        "human_guide_sha256": writeup_sha256,
        "teacher_module": "poke_bot/alakazam_new_list_heuristics.py",
        "teacher_module_sha256": teacher_sha256,
        "ordinary_rl_guide_loss_weight": ORDINARY_GUIDE_LOSS_WEIGHT,
        "expert_soft_refresh_guide_loss_weight": EXPERT_REFRESH_GUIDE_LOSS_WEIGHT,
    }
    for key, expected in source_guide_checks.items():
        _require_equal(
            label=f"typed source owner_guide.{key}",
            actual=source_guide.get(key),
            expected=expected,
        )
    _require_equal(
        label="guide teacher checksum",
        actual=contract.get("teacher_module_sha256"),
        expected=teacher_sha256,
    )
    _require_equal(
        label="guide write-up path",
        actual=expert_writeup.get("path"),
        expected="docs/deck_guides/alakazam-new-list-direct-r241-expert-brief.md",
    )
    _require_equal(
        label="guide write-up checksum",
        actual=expert_writeup.get("sha256"),
        expected=writeup_sha256,
    )
    _require_equal(
        label="guide write-up word count",
        actual=expert_writeup.get("word_count"),
        expected=writeup_word_count,
    )
    _require_equal(
        label="guide training mode",
        actual=policy_target.get("training_mode"),
        expected=TRAINING_MODE,
    )
    _require_equal(
        label="direct policy cross entropy",
        actual=policy_target.get("direct_policy_cross_entropy_allowed"),
        expected=False,
    )
    _require_equal(
        label="guide runtime logit route",
        actual=policy_target.get("runtime_action_logit_route_allowed"),
        expected=False,
    )
    _require_equal(
        label="ordinary guide loss",
        actual=curriculum.get("ordinary_rl_guide_loss_weight"),
        expected=ORDINARY_GUIDE_LOSS_WEIGHT,
    )
    _require_equal(
        label="expert refresh guide loss",
        actual=curriculum.get("expert_soft_refresh_guide_loss_weight"),
        expected=EXPERT_REFRESH_GUIDE_LOSS_WEIGHT,
    )
    _require_equal(
        label="combo loss",
        actual=curriculum.get("combo_state_loss_weight"),
        expected=0.0,
    )
    _require_equal(
        label="combo route",
        actual=curriculum.get("combo_state_route_enabled"),
        expected=False,
    )
    guide_rows = _validate_exact_deck_canary(cards)

    return {
        "schema": READINESS_SCHEMA,
        "status": "validated",
        "created_at_utc": READINESS_CREATED_AT_UTC,
        "scope": "exact_deck_direct_policy_ordinary_rl_only",
        "specialist_id": SPECIALIST_ID,
        "candidate_id": CANDIDATE_ID,
        "guide_version": GUIDE_VERSION,
        "guide_rows": guide_rows,
        "guide_rows_scope": GUIDE_ROWS_SCOPE,
        "guide_contract": {
            "path": "config/deck_guides/alakazam-new-list-direct-r241.yaml",
            "sha256": guide_sha256,
        },
        "expert_writeup": {
            "path": "docs/deck_guides/alakazam-new-list-direct-r241-expert-brief.md",
            "sha256": writeup_sha256,
            "word_count": writeup_word_count,
            "maximum_words": 10_000,
        },
        "exact_deck": {
            "deck_id": DECK_ID,
            "path": "decks/archetype-samples/alakazam-new-list-direct-r241.csv",
            "file_sha256": deck_sha256,
            "ordered_cards_sha256": ordered_sha256,
            "canonical_multiset_sha256": multiset_sha256,
            "card_count": 60,
            "exact_multiset_required": True,
        },
        "teacher": {
            "module": "poke_bot/alakazam_new_list_heuristics.py",
            "sha256": teacher_sha256,
            "exact_deck_gate_required": True,
            "complete_legal_stage_scoring_required": True,
            "incomplete_or_ambiguous_stage_behavior": "mask_entire_stage",
            "public_state_only": True,
        },
        "ordinary_rl": {
            "guide_loss_weight": ORDINARY_GUIDE_LOSS_WEIGHT,
            "guide_target_source": "exact_new_deck_direct_policy_self_play",
            "guide_target_generation_enabled": True,
            "guide_rows": guide_rows,
            "guide_rows_scope": GUIDE_ROWS_SCOPE,
            "training_mode": TRAINING_MODE,
            "direct_policy_cross_entropy_allowed": False,
            "guide_pairwise_route_heads": [
                "action_q",
                "action_resource",
                "action_utility",
                "setup_board_outcome",
            ],
        },
        "expert_soft_refresh": {
            "guide_loss_weight": EXPERT_REFRESH_GUIDE_LOSS_WEIGHT,
            "guide_target_generation_enabled": False,
            "expert_corpus_guide_rows": 0,
            "reason": EXPERT_ZERO_REASON,
            "historical_expert_rows_may_train_observed_outcomes": True,
            "historical_expert_rows_may_not_supply_r241_guide_targets": True,
        },
        "corpus_readiness": {
            "is_current_deck_guide_corpus_ready_receipt": False,
            "expert_corpus_ready_for_r241_guide_targets": False,
            "reason": (
                "the only guide-ready path in this receipt is checksum-bound "
                "direct-policy self-play"
            ),
        },
        "runtime_exclusions": {
            "runtime_action_authority": False,
            "runtime_input": False,
            "runtime_action_logit_route": False,
            "mcts": False,
            "recursive_turn_planner": False,
            "guide2vec": False,
            "guide_logit_bias": False,
            "hidden_state_or_future_information": False,
        },
        "checks": {
            "contract_module_deck_checksums_match": True,
            "exact_multiset_gate_rejects_non_r241_decks": True,
            "nonflat_complete_stage_canary_is_scorable": True,
            "ordinary_rl_weight_is_exactly_0_05": True,
            "expert_soft_refresh_weight_is_exactly_0_0": True,
            "expert_corpus_mismatch_is_not_relabelled_or_reused": True,
            "historical_r175_or_r79_guide_artifacts_consumed": False,
            "final_policy_logits_are_guide_targets": False,
        },
        "authority": {
            "training_preparation": True,
            "managed_training_start": False,
            "selector": False,
            "production": False,
            "kaggle_submission": False,
        },
    }


def _atomic_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    sort_keys: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=sort_keys)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _r241_execution_block() -> dict[str, Any]:
    return {
        "guide_readiness_receipt": (
            "state/alakazam-new-list-direct-r241-guide-readiness.json"
        ),
        "ordinary_rl": {
            "guide_loss_weight": ORDINARY_GUIDE_LOSS_WEIGHT,
            "guide_target_source": "exact_new_deck_direct_policy_self_play",
            "guide_target_generation_enabled": True,
        },
        "expert_soft_refresh": {
            "guide_loss_weight": EXPERT_REFRESH_GUIDE_LOSS_WEIGHT,
            "guide_target_generation_enabled": False,
            "expert_corpus_guide_rows": 0,
            "reason": EXPERT_ZERO_REASON,
        },
        "historical_r175_or_r79_guide_artifacts_consumed": False,
        "runtime_guide_authority": False,
        "mcts_rtp_guide2vec": False,
    }


def _validate_staged_bundle(
    *,
    readiness: Path,
    role_map: Path,
    curriculum: Path,
    validation: Path,
    expected_training_sha256: str,
) -> None:
    roles = _read_json(role_map)
    spec = _read_json(curriculum)
    receipt = _read_json(validation)
    sources = roles.get("canonical_learned_decision_sources")
    heads = roles.get("heads")
    if not isinstance(sources, list) or not isinstance(heads, dict):
        raise R241GuideResealError("strategic head-role inventory is malformed")
    if len(sources) != 18 or set(sources) != set(heads) or "combo_state" in heads:
        raise R241GuideResealError(
            "r241 must contain exactly 18 live non-combo learned routes"
        )
    for name, row in heads.items():
        if not isinstance(row, dict) or any(
            row.get(field) is not True
            for field in ("trainable", "enters_decision_fusion")
        ):
            raise R241GuideResealError(f"non-combo route is not live/trainable: {name}")
    _require_equal(
        label="curriculum head-role digest",
        actual=spec.get("head_role_map_sha256"),
        expected=_sha256(role_map),
    )
    _require_equal(
        label="validation curriculum digest",
        actual=receipt.get("curriculum_spec_sha256"),
        expected=_sha256(curriculum),
    )
    _require_equal(
        label="validation head-role digest",
        actual=receipt.get("head_role_map_sha256"),
        expected=_sha256(role_map),
    )
    _require_equal(
        label="validation readiness path",
        actual=receipt.get("guide_ready_receipt"),
        expected=str(READINESS.resolve()),
    )
    _require_equal(
        label="validation readiness digest",
        actual=receipt.get("guide_ready_receipt_sha256"),
        expected=_sha256(readiness),
    )
    _require_equal(
        label="r241 guide execution block",
        actual=receipt.get("r241_exact_deck_guide_execution"),
        expected=_r241_execution_block(),
    )
    implementations = receipt.get("implementation_artifacts")
    if not isinstance(implementations, list):
        raise R241GuideResealError("validation implementation inventory is missing")
    training_rows = [
        row
        for row in implementations
        if isinstance(row, dict) and row.get("role") == "training_implementation"
    ]
    if len(training_rows) != 1:
        raise R241GuideResealError(
            "validation must bind exactly one train implementation"
        )
    _require_equal(
        label="validation train.py digest",
        actual=training_rows[0].get("sha256"),
        expected=expected_training_sha256,
    )

    from poke_bot.train import (
        GUIDE_TRAINING_MODE_DIRECTIONAL,
        assert_strategic_curriculum_receipt_contract,
    )

    assert_strategic_curriculum_receipt_contract(
        specialist_id=SPECIALIST_ID,
        curriculum_spec=str(curriculum),
        head_role_map=str(role_map),
        validation_receipt=str(validation),
        expected_training_mode=GUIDE_TRAINING_MODE_DIRECTIONAL,
    )


def reseal(*, expected_training_sha256: str, check_only: bool) -> dict[str, Any]:
    expected_training_sha256 = _normalize_sha256(expected_training_sha256)
    _require_equal(
        label="pre-materialization train.py digest",
        actual=_sha256(TRAINING_IMPLEMENTATION),
        expected=expected_training_sha256,
    )
    owner_sha256 = _sha256(OWNER_CONTRACT)
    readiness_payload = build_readiness_payload()

    with tempfile.TemporaryDirectory(
        prefix=".r241-guide-reseal.", dir=READINESS.parent
    ) as raw_staging:
        staging = Path(raw_staging)
        staged_readiness = staging / READINESS.name
        _atomic_json(staged_readiness, readiness_payload, sort_keys=False)

        from scripts import (
            validate_future_specialist_strategic_curriculum as generator,
        )

        generated = generator.materialize(
            specialist_id=SPECIALIST_ID,
            guide_contract=GUIDE_CONTRACT,
            guide_ready_receipt=staged_readiness,
            output_root=staging / "generic",
            training_implementation=TRAINING_IMPLEMENTATION,
            include_combo_state=False,
            training_mode=generator.DIRECTIONAL_TRAINING_MODE,
        )
        staged_roles = staging / HEAD_ROLE_MAP.name
        staged_curriculum = staging / CURRICULUM.name
        staged_validation = staging / VALIDATION.name
        role_payload = _read_json(generated["head_role_map"])
        curriculum_payload = _read_json(generated["curriculum_spec"])
        validation_payload = _read_json(generated["validation_receipt"])
        validation_payload["guide_ready_receipt"] = str(READINESS.resolve())
        validation_payload["guide_ready_receipt_sha256"] = _sha256(staged_readiness)
        validation_payload["r241_exact_deck_guide_execution"] = _r241_execution_block()
        _atomic_json(staged_roles, role_payload, sort_keys=True)
        _atomic_json(staged_curriculum, curriculum_payload, sort_keys=True)
        _atomic_json(staged_validation, validation_payload, sort_keys=True)

        _require_equal(
            label="post-materialization train.py digest",
            actual=_sha256(TRAINING_IMPLEMENTATION),
            expected=expected_training_sha256,
        )
        _require_equal(
            label="owner contract stability",
            actual=_sha256(OWNER_CONTRACT),
            expected=owner_sha256,
        )
        _require_equal(
            label="readiness source stability",
            actual=build_readiness_payload(),
            expected=readiness_payload,
        )
        _validate_staged_bundle(
            readiness=staged_readiness,
            role_map=staged_roles,
            curriculum=staged_curriculum,
            validation=staged_validation,
            expected_training_sha256=expected_training_sha256,
        )

        staged_paths = {
            "guide_readiness": staged_readiness,
            "strategic_head_roles": staged_roles,
            "strategic_curriculum": staged_curriculum,
            "strategic_curriculum_validation": staged_validation,
        }
        final_paths = {
            "guide_readiness": READINESS,
            "strategic_head_roles": HEAD_ROLE_MAP,
            "strategic_curriculum": CURRICULUM,
            "strategic_curriculum_validation": VALIDATION,
        }
        mismatches = [
            name
            for name in staged_paths
            if not final_paths[name].is_file()
            or staged_paths[name].read_bytes() != final_paths[name].read_bytes()
        ]
        if check_only:
            if mismatches:
                raise R241GuideResealError(
                    "sealed r241 guide artifacts are stale: "
                    + ", ".join(sorted(mismatches))
                )
        else:
            # Publish dependency leaves first and the validation receipt last.
            for name in (
                "guide_readiness",
                "strategic_head_roles",
                "strategic_curriculum",
                "strategic_curriculum_validation",
            ):
                os.replace(staged_paths[name], final_paths[name])

    return {
        "schema": "poke_bot.alakazam_new_list_direct_policy_r241_guide_reseal/v1",
        "status": "matched" if check_only else "resealed",
        "training_implementation": {
            "path": str(TRAINING_IMPLEMENTATION),
            "sha256": expected_training_sha256,
        },
        "artifacts": {
            "guide_readiness": {
                "path": str(READINESS.relative_to(ROOT)),
                "sha256": _sha256(READINESS),
            },
            "strategic_head_roles": {
                "path": str(HEAD_ROLE_MAP.relative_to(ROOT)),
                "sha256": _sha256(HEAD_ROLE_MAP),
            },
            "strategic_curriculum": {
                "path": str(CURRICULUM.relative_to(ROOT)),
                "sha256": _sha256(CURRICULUM),
            },
            "strategic_curriculum_validation": {
                "path": str(VALIDATION.relative_to(ROOT)),
                "sha256": _sha256(VALIDATION),
            },
        },
        "guide_loss_weights": {
            "ordinary_rl": ORDINARY_GUIDE_LOSS_WEIGHT,
            "expert_soft_refresh": EXPERT_REFRESH_GUIDE_LOSS_WEIGHT,
        },
        "managed_training_started": False,
        "runtime_or_selector_activated": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-training-implementation-sha256",
        required=True,
        help="Final trainer-owner SHA-256 for poke_bot/train.py.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate in staging and fail if the four published files differ.",
    )
    args = parser.parse_args(argv)
    result = reseal(
        expected_training_sha256=args.expected_training_implementation_sha256,
        check_only=bool(args.check),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
