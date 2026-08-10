"""Exact r182 safety gate for public ``LibcgMultiEnv`` transport packs.

The public worker path is allowed to share a multi-env process only for the
specific package bytes that passed the r175 compatibility audit.  A job that
does not bind every identity/provenance field to that immutable allowlist is
ordinary one-game ``play`` work.  This module is intentionally usable by both
the trainer dispatch code and the remote worker before it enters
``run_play_multi``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


_SAFETY_SCHEMA = "poke_bot.alakazam_public_multi_env_safety_r182/v1"
_DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "state"
    / "alakazam-public-multi-env-split-r182.json"
)
def public_multi_env_safety_manifest_path() -> Path:
    """Return the deployment-local immutable r182 public-pack manifest."""

    # This path is intentionally not environment-overridable.  Both controller
    # and worker must classify against the one typed source committed beside
    # their running code; accepting an arbitrary environment path would let a
    # syntactically valid replacement manifest broaden the exact allowlist.
    return _DEFAULT_MANIFEST


@lru_cache(maxsize=4)
def _load_manifest(
    path_text: str,
) -> tuple[dict[str, str], dict[str, str], frozenset[str], str]:
    path = Path(path_text)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "public multi-env safety manifest is unavailable or invalid: "
            f"{path}: {exc}"
        ) from exc
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != _SAFETY_SCHEMA
        or raw.get("default_singleton") is not True
    ):
        raise RuntimeError(f"public multi-env safety manifest schema mismatch: {path}")
    contract_schema = str(raw.get("portable_baseline_contract_schema") or "")
    groups = frozenset(
        str(value) for value in raw.get("permitted_training_groups") or ()
    )
    pairs: dict[str, str] = {}
    for row in raw.get("safe_public_pairs") or ():
        if not isinstance(row, dict):
            raise RuntimeError("public multi-env safety manifest has a non-object pair")
        opponent_id = str(row.get("opponent_id") or "")
        digest = str(row.get("content_digest") or "")
        if (
            not opponent_id
            or not digest.startswith("sha256:")
            or opponent_id in pairs
        ):
            raise RuntimeError("public multi-env safety manifest has an invalid pair")
        pairs[opponent_id] = digest
    raw_group_ids = raw.get("safe_public_group_ids")
    if not isinstance(raw_group_ids, dict) or set(raw_group_ids) != set(groups):
        raise RuntimeError("public multi-env safety manifest group map mismatch")
    pair_groups: dict[str, str] = {}
    for group, opponent_ids in raw_group_ids.items():
        if not isinstance(opponent_ids, list):
            raise RuntimeError("public multi-env safety group IDs must be a list")
        for raw_opponent_id in opponent_ids:
            opponent_id = str(raw_opponent_id or "")
            if not opponent_id or opponent_id in pair_groups:
                raise RuntimeError("public multi-env safety group IDs are invalid")
            pair_groups[opponent_id] = str(group)
    legacy = frozenset(
        str(value) for value in raw.get("legacy_singleton_opponent_ids") or ()
    )
    if (
        len(pairs) != 27
        or not contract_schema
        or not groups
        or len(legacy) != 10
        or set(pairs) & set(legacy)
        or set(pair_groups) != set(pairs)
    ):
        raise RuntimeError("public multi-env safety manifest is incomplete")
    return pairs, pair_groups, legacy, contract_schema


def _manifest() -> tuple[dict[str, str], dict[str, str], frozenset[str], str]:
    return _load_manifest(str(public_multi_env_safety_manifest_path()))


def public_multi_env_legacy_opponent_ids() -> frozenset[str]:
    """Return the documented ten r182 singleton-only gate packages."""

    return _manifest()[2]


def public_multi_env_safe_job(job: Mapping[str, Any]) -> bool:
    """Whether this exact public job may enter a four-way multi-env pack.

    This is deliberately default-deny.  It accepts only a portable baseline
    job with matching job/spec/provenance identity and digest, one of the two
    training collection groups, and an exact allowlisted ``(id, digest)``.
    Manifest failure is a safe ``False`` here; the remote worker's explicit
    pack admission turns that case into a fail-closed request error.
    """

    try:
        pairs, pair_groups, legacy, contract_schema = _manifest()
    except RuntimeError:
        return False
    if not isinstance(job, Mapping):
        return False
    spec = job.get("spec")
    provenance = job.get("target_provenance")
    if not isinstance(spec, Mapping) or not isinstance(provenance, Mapping):
        return False
    opponent_id = str(job.get("opponent_id") or "")
    spec_id = str(spec.get("id") or "")
    digest = str(spec.get("content_digest") or "")
    if (
        not opponent_id
        or opponent_id != spec_id
        or job.get("require_portable_baseline_contract") is not True
        or str(spec.get("contract_schema") or "") != contract_schema
        or str(provenance.get("opponent_id") or "") != opponent_id
        or str(provenance.get("opponent_content_digest") or "") != digest
        or str(provenance.get("opponent_training_group") or "")
        != pair_groups.get(opponent_id)
        or opponent_id in legacy
    ):
        return False
    return pairs.get(opponent_id) == digest


def require_public_multi_env_safe_job(job: Mapping[str, Any]) -> None:
    """Raise a transport admission error unless ``job`` passes the exact gate."""

    # Force an actionable hard failure for a missing/corrupt manifest rather
    # than making an unaudited multi request look like a normal bad child.
    _manifest()
    if not public_multi_env_safe_job(job):
        opponent_id = ""
        if isinstance(job, Mapping):
            opponent_id = str(job.get("opponent_id") or "")
        raise ValueError(
            "public multi-env pack rejected by exact r182 safety allowlist: "
            f"opponent_id={opponent_id!r}"
        )
