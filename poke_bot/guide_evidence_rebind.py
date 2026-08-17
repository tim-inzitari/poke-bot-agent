"""Fail-closed current-deck guide evidence-only receipt migration."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_identity(value: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(value)
    normalized.pop("contract_sha256", None)
    strategic = dict(normalized.get("strategic_curriculum") or {})
    for key in (
        "curriculum_spec_sha256",
        "head_role_map_sha256",
        "validation_receipt_sha256",
    ):
        strategic.pop(key, None)
    normalized["strategic_curriculum"] = strategic
    return normalized


def _normalized_contract(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"guide contract must be a mapping: {path}")
    normalized = copy.deepcopy(value)
    coverage = dict(normalized.get("combo_head_coverage") or {})
    coverage.pop("artifact_sha256", None)
    coverage.pop("implementation_validation_receipt_sha256", None)
    normalized["combo_head_coverage"] = coverage
    return normalized


def validation_or_evidence_only_rebind(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    old_contract_snapshot: Path | None = None,
) -> bool:
    """Allow only derived validation/evidence checksums to advance.

    Training behavior, guide schedule, corpus identity, strategic spec, and
    head-role map must remain byte-for-byte equivalent after removing the two
    implementation-evidence hashes whose files were regenerated.
    """

    if not old or not new or _normalized_identity(old) != _normalized_identity(new):
        return False
    old_digest = str(old.get("contract_sha256") or "")
    new_digest = str(new.get("contract_sha256") or "")
    old_validation = str(
        (old.get("strategic_curriculum") or {}).get(
            "validation_receipt_sha256"
        )
        or ""
    )
    new_validation = str(
        (new.get("strategic_curriculum") or {}).get(
            "validation_receipt_sha256"
        )
        or ""
    )
    if not old_validation or not new_validation or old_validation == new_validation:
        return False
    if old_digest == new_digest:
        return True
    current_raw = str(new.get("contract") or "").strip()
    if old_contract_snapshot is None or not current_raw:
        return False
    snapshot = old_contract_snapshot.expanduser().resolve()
    current = Path(current_raw).expanduser().resolve()
    if (
        not snapshot.is_file()
        or not current.is_file()
        or _sha256(snapshot) != old_digest
        or _sha256(current) != new_digest
    ):
        return False
    return _normalized_contract(snapshot) == _normalized_contract(current)
