"""Read-only, checksum-keyed training recipe metadata for r187.

The Replay Model Inspector must never infer a training recipe from a checkpoint
filename, an active selector, or a submission label.  This small registry is a
separate trust boundary: a recipe is visible only when the exact checkpoint
SHA-256 has one verified, source-evidence-backed record.

Registry evidence paths are repository-relative to the registry's repository
root (``<root>/ops/elmo/<registry>.json``).  They are rehashed while the
registry is loaded, so editing a cited source makes that record unavailable
rather than silently changing the recipe presented for a submitted model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeAlias

TRAINING_RECIPE_REGISTRY_SCHEMA = (
    "poke_bot.replay_model_inspector_training_recipe_registry/v1"
)
PathLike: TypeAlias = str | os.PathLike[str]

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_STRATEGIC_HEAD_IDS = (
    "action_q",
    "action_type",
    "action_target",
    "action_resource",
    "action_utility",
    "tactical_outcome",
    "opponent_response",
    "resource_forecast",
    "game_phase",
    "outcome_distribution",
    "remaining_turns",
)


class TrainingRecipeRegistryError(ValueError):
    """The top-level training-recipe registry is not a usable v1 document."""


def _unique_reasons(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        clean = str(value).strip()
        if clean and clean not in result:
            result.append(clean)
    return tuple(result)


def _canonical_sha256(value: object) -> str | None:
    text = str(value or "").strip()
    return text if _SHA256_RE.fullmatch(text) else None


def _finite_nonnegative_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TrainingRecipeEvidence:
    """One cited source file and the read-only verification outcome."""

    path: str | None
    sha256: str | None
    pointer: str | None
    role: str | None
    availability_reasons: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return (
            self.path is not None
            and self.sha256 is not None
            and self.pointer is not None
            and self.role is not None
            and not self.availability_reasons
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "pointer": self.pointer,
            "role": self.role,
            "available": self.available,
            "availability_reasons": list(self.availability_reasons),
        }


@dataclass(frozen=True, slots=True)
class TrainingRecipeRecord:
    """One exact-checkpoint recipe record from a registry."""

    checkpoint_sha256: str | None
    status: str | None
    recipe: Mapping[str, Any] | None
    evidence: tuple[TrainingRecipeEvidence, ...]
    availability_reasons: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return (
            self.checkpoint_sha256 is not None
            and self.status == "verified"
            and self.recipe is not None
            and bool(self.evidence)
            and all(item.available for item in self.evidence)
            and not self.availability_reasons
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "status": "available" if self.available else "unavailable",
            "record_status": self.status,
            "recipe": dict(self.recipe) if self.available and self.recipe else None,
            "evidence": [item.to_dict() for item in self.evidence],
            "availability": {
                "available": self.available,
                **(
                    {"reason": self.availability_reasons[0]}
                    if self.availability_reasons
                    else {}
                ),
            },
            "availability_reasons": list(self.availability_reasons),
        }


@dataclass(frozen=True, slots=True)
class TrainingRecipeRegistry:
    """Parsed checksum-keyed records and non-sensitive registry diagnostics."""

    path: Path
    evidence_root: Path
    records: tuple[TrainingRecipeRecord, ...]
    issues: tuple[str, ...] = ()
    schema: str = TRAINING_RECIPE_REGISTRY_SCHEMA

    def records_for_checkpoint(
        self, checkpoint_sha256: str
    ) -> tuple[TrainingRecipeRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.checkpoint_sha256 == checkpoint_sha256
        )

    def resolve_checkpoint(
        self, checkpoint_sha256: str | None
    ) -> tuple[TrainingRecipeRecord | None, tuple[str, ...]]:
        """Resolve exactly one verified record or return stable fail-closed codes."""

        digest = _canonical_sha256(checkpoint_sha256)
        if digest is None:
            return None, ("training_recipe_checkpoint_digest_unavailable",)
        candidates = self.records_for_checkpoint(digest)
        if not candidates:
            return None, ("training_recipe_checkpoint_mapping_unavailable",)
        if len(candidates) != 1:
            return None, ("training_recipe_checkpoint_mapping_ambiguous",)
        candidate = candidates[0]
        if not candidate.available:
            return None, _unique_reasons(
                (
                    "training_recipe_checkpoint_mapping_unavailable",
                    *candidate.availability_reasons,
                )
            )
        return candidate, ()


def _evidence_item(
    raw: object,
    *,
    evidence_root: Path,
) -> TrainingRecipeEvidence:
    if not isinstance(raw, Mapping):
        return TrainingRecipeEvidence(
            path=None,
            sha256=None,
            pointer=None,
            role=None,
            availability_reasons=("training_recipe_evidence_not_an_object",),
        )
    path_value = raw.get("path")
    path = path_value.strip() if isinstance(path_value, str) else None
    digest = _canonical_sha256(raw.get("sha256"))
    pointer_value = raw.get("pointer")
    pointer = pointer_value.strip() if isinstance(pointer_value, str) else None
    role_value = raw.get("role")
    role = role_value.strip() if isinstance(role_value, str) else None
    reasons: list[str] = []
    if not path:
        reasons.append("training_recipe_evidence_path_invalid")
    if digest is None:
        reasons.append("training_recipe_evidence_sha256_invalid")
    if not pointer:
        reasons.append("training_recipe_evidence_pointer_invalid")
    if not role:
        reasons.append("training_recipe_evidence_role_invalid")
    if not reasons and path is not None and digest is not None:
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            reasons.append("training_recipe_evidence_path_outside_root")
        else:
            try:
                resolved = (evidence_root / candidate).resolve(strict=True)
                resolved.relative_to(evidence_root)
                if not resolved.is_file():
                    reasons.append("training_recipe_evidence_not_a_regular_file")
                elif _sha256_file(resolved) != digest:
                    reasons.append("training_recipe_evidence_sha256_mismatch")
            except FileNotFoundError:
                reasons.append("training_recipe_evidence_missing")
            except (OSError, RuntimeError, ValueError):
                reasons.append("training_recipe_evidence_path_outside_root")
    return TrainingRecipeEvidence(
        path=path,
        sha256=digest,
        pointer=pointer,
        role=role,
        availability_reasons=_unique_reasons(reasons),
    )


def _recipe(value: object) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return None, ("training_recipe_missing_or_invalid",)
    result = dict(value)
    reasons: list[str] = []
    if result.get("scope") != "source_backed_training_loss_multipliers":
        reasons.append("training_recipe_scope_invalid")
    if result.get("training_only") is not True:
        reasons.append("training_recipe_not_marked_training_only")
    if result.get("fine_tune_execution_authority") is not False:
        reasons.append("training_recipe_fine_tune_authority_invalid")
    if result.get("evaluation_replays_training_eligible") is not False:
        reasons.append("training_recipe_evaluation_replay_safety_invalid")
    weights = result.get("loss_weights")
    if not isinstance(weights, Mapping) or not weights:
        reasons.append("training_recipe_loss_weights_missing_or_invalid")
    else:
        for name, raw_weight in weights.items():
            if not isinstance(name, str) or not name.strip():
                reasons.append("training_recipe_loss_weight_name_invalid")
                break
            if _finite_nonnegative_number(raw_weight) is None:
                reasons.append("training_recipe_loss_weight_invalid")
                break
    strategic = result.get("strategic_head_loss_weights")
    if not isinstance(strategic, Mapping):
        reasons.append("training_recipe_strategic_head_weights_missing_or_invalid")
    else:
        keys = set(strategic)
        if keys != set(_STRATEGIC_HEAD_IDS):
            reasons.append("training_recipe_strategic_head_inventory_invalid")
        elif any(_finite_nonnegative_number(strategic[name]) is None for name in keys):
            reasons.append("training_recipe_strategic_head_weight_invalid")
    return result, _unique_reasons(reasons)


def _record(raw: object, *, evidence_root: Path) -> TrainingRecipeRecord:
    if not isinstance(raw, Mapping):
        return TrainingRecipeRecord(
            checkpoint_sha256=None,
            status=None,
            recipe=None,
            evidence=(),
            availability_reasons=("training_recipe_record_not_an_object",),
        )
    digest = _canonical_sha256(raw.get("checkpoint_sha256"))
    status = raw.get("status") if isinstance(raw.get("status"), str) else None
    recipe, recipe_reasons = _recipe(raw.get("recipe"))
    raw_evidence = raw.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        evidence: tuple[TrainingRecipeEvidence, ...] = ()
        evidence_reasons = ("training_recipe_evidence_missing_or_invalid",)
    else:
        evidence = tuple(
            _evidence_item(item, evidence_root=evidence_root) for item in raw_evidence
        )
        evidence_reasons = ()
    reasons = [*recipe_reasons, *evidence_reasons]
    if digest is None:
        reasons.append("training_recipe_checkpoint_sha256_invalid")
    if status != "verified":
        reasons.append("training_recipe_status_not_verified")
    for item in evidence:
        reasons.extend(item.availability_reasons)
    return TrainingRecipeRecord(
        checkpoint_sha256=digest,
        status=status,
        recipe=recipe,
        evidence=evidence,
        availability_reasons=_unique_reasons(reasons),
    )


def load_training_recipe_registry(
    path: PathLike,
    *,
    evidence_root: PathLike | None = None,
) -> TrainingRecipeRegistry:
    """Load and verify a read-only r187 training-recipe registry.

    Top-level schema errors reject the registry.  Individual records remain in
    the parsed result with explicit reasons; this keeps a submission browseable
    while ensuring no invalid/ambiguous recipe is selected.
    """

    try:
        registry_path = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise TrainingRecipeRegistryError(
            "training recipe registry is unavailable"
        ) from exc
    if not registry_path.is_file():
        raise TrainingRecipeRegistryError("training recipe registry is not a file")
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainingRecipeRegistryError(
            "training recipe registry is invalid JSON"
        ) from exc
    except OSError as exc:
        raise TrainingRecipeRegistryError(
            "training recipe registry cannot be read"
        ) from exc
    if not isinstance(payload, Mapping):
        raise TrainingRecipeRegistryError("training recipe registry must be an object")
    if payload.get("schema") != TRAINING_RECIPE_REGISTRY_SCHEMA:
        raise TrainingRecipeRegistryError("training recipe registry schema is invalid")
    if payload.get("version") != 1 or isinstance(payload.get("version"), bool):
        raise TrainingRecipeRegistryError("training recipe registry version is invalid")
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise TrainingRecipeRegistryError(
            "training recipe registry records are invalid"
        )
    try:
        root = (
            Path(evidence_root).expanduser().resolve(strict=True)
            if evidence_root is not None
            else registry_path.parents[2].resolve(strict=True)
        )
    except (IndexError, OSError, RuntimeError) as exc:
        raise TrainingRecipeRegistryError(
            "training recipe evidence root is unavailable"
        ) from exc
    if not root.is_dir():
        raise TrainingRecipeRegistryError(
            "training recipe evidence root is not a directory"
        )
    records = tuple(_record(row, evidence_root=root) for row in rows)
    duplicate_digests = {
        record.checkpoint_sha256
        for record in records
        if record.checkpoint_sha256 is not None
        and sum(
            other.checkpoint_sha256 == record.checkpoint_sha256 for other in records
        )
        > 1
    }
    if duplicate_digests:
        records = tuple(
            replace(
                record,
                availability_reasons=_unique_reasons(
                    (
                        *record.availability_reasons,
                        "training_recipe_checkpoint_mapping_ambiguous",
                    )
                ),
            )
            if record.checkpoint_sha256 in duplicate_digests
            else record
            for record in records
        )
    issues = tuple(
        f"training_recipe_record_{index}_not_an_object"
        for index, row in enumerate(rows)
        if not isinstance(row, Mapping)
    )
    return TrainingRecipeRegistry(
        path=registry_path,
        evidence_root=root,
        records=records,
        issues=issues,
    )


__all__ = [
    "TRAINING_RECIPE_REGISTRY_SCHEMA",
    "TrainingRecipeEvidence",
    "TrainingRecipeRecord",
    "TrainingRecipeRegistry",
    "TrainingRecipeRegistryError",
    "load_training_recipe_registry",
]
