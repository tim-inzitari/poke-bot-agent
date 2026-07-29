#!/usr/bin/env python3
"""Select the next executable specialist from canonical protocol state.

Selection preserves the configured priority order, but a target is executable
only when it is unfinished and its protected expert corpus satisfies the exact
minimum-decision contract. Insufficient corpora remain required targets and
are reported as deferred; they are never silently removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml


RESULT_SCHEMA = "poke_bot.next_specialist_selection/v1"
UNFINISHED = {
    "unstarted",
    "restart_required",
    "bootstrap_partial",
    "bootstrap_complete",
    "blocked",
}
LATEST20_DAYS = 20
PRIMARY_SOURCE_MODE = "latest20_primary"
FALLBACK_SOURCE_MODE = "historical_validated_shard_fallback"
PUBLIC_FULL_HISTORY_SOURCE_MODE = "public_full_history_exact_deck_identity"


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"YAML root is not an object: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _is_sha256(value: Any) -> bool:
    raw = str(value or "")
    return bool(
        raw.startswith("sha256:")
        and len(raw) == len("sha256:") + 64
        and all(character in "0123456789abcdef" for character in raw[7:])
    )


def _manifest_from_pointer(pointer: Path, payload: dict[str, Any]) -> Path:
    raw = str(payload.get("manifest") or "").strip()
    if not raw:
        raise RuntimeError("protected expert corpus manifest is missing")
    manifest = (pointer.parent / raw).resolve()
    if manifest.parent != pointer.parent.resolve() or not manifest.is_file():
        raise RuntimeError("protected expert corpus manifest escaped its directory")
    if payload.get("manifest_sha256") != _sha256(manifest):
        raise RuntimeError("protected expert corpus manifest checksum changed")
    return manifest


def _guide_latest20_evidence(
    manifest_path: Path,
    *,
    specialist_id: str,
) -> dict[str, Any]:
    """Validate the specialized guide-corpus latest-20 receipt chain.

    Guide corpora are assembled from one independently receipted feature shard
    per calendar day. Their compact manifest intentionally omits the ordinary
    ``source_window`` projection, so selection reconstructs that projection
    only from the checksum-bound ready receipt and all 20 authoritative daily
    receipts. This is not a weaker alternate path: every guide shard and every
    source archive must be identified by SHA-256.
    """

    root = manifest_path.parent
    ready_path = root / "CURRENT_DECK_GUIDE_CORPUS_READY.json"
    if not ready_path.is_file():
        raise RuntimeError(
            "expert corpus is not an exact latest-20 calendar-day "
            "window filtered after selection"
        )
    ready = _load_json(ready_path)
    raw_dates = [str(value) for value in (ready.get("dates") or ())]
    try:
        parsed_dates = [date.fromisoformat(value) for value in raw_dates]
    except ValueError as exc:
        raise RuntimeError(
            "guide latest20 ready receipt contains an invalid date"
        ) from exc
    expected_dates = (
        [parsed_dates[0] + timedelta(days=index) for index in range(LATEST20_DAYS)]
        if parsed_dates
        else []
    )
    daily_rows = list(ready.get("daily_shards") or ())
    daily_by_date = {
        str(row.get("date") or ""): dict(row)
        for row in daily_rows
        if isinstance(row, dict)
    }
    if (
        ready.get("schema") != "poke_bot.current_deck_guide_corpus_ready/v1"
        or ready.get("status") != "ready"
        or ready.get("specialist_id") != specialist_id
        or int(ready.get("days") or 0) != LATEST20_DAYS
        or len(raw_dates) != LATEST20_DAYS
        or len(set(raw_dates)) != LATEST20_DAYS
        or parsed_dates != expected_dates
        or set(daily_by_date) != set(raw_dates)
        or ready.get("manifest_sha256") != _sha256(manifest_path)
    ):
        raise RuntimeError("guide corpus latest20 ready receipt is invalid")

    source_days: list[dict[str, Any]] = []
    for source_date in raw_dates:
        row = daily_by_date[source_date]
        receipt_path = (
            root
            / f"{specialist_id}-{source_date}.features.receipt.json"
        )
        feature_path = root / f"{specialist_id}-{source_date}.features"
        if not receipt_path.is_file() or not feature_path.is_file():
            raise RuntimeError(
                f"guide latest20 daily artifact is missing: {source_date}"
            )
        receipt = _load_json(receipt_path)
        selection = dict(receipt.get("selection") or {})
        source_archive = dict(receipt.get("source_archive") or {})
        output = dict(receipt.get("output") or {})
        stats = dict(receipt.get("stats") or {})
        feature_sha256 = _sha256(feature_path)
        archive_sha256 = str(source_archive.get("sha256") or "")
        if (
            receipt.get("format")
            != "pokebot-authoritative-visual-day-receipt"
            or receipt.get("source_date") != source_date
            or selection.get("acting_seat_archetype") != specialist_id
            or output.get("sha256") != feature_sha256
            or row.get("sha256") != feature_sha256
            or row.get("receipt_sha256") != _sha256(receipt_path)
            or not _is_sha256(archive_sha256)
            or int(row.get("records") or 0)
            != int(stats.get("records_kept") or 0)
            or int(row.get("decisions") or 0)
            != int(stats.get("decisions_kept") or 0)
        ):
            raise RuntimeError(
                f"guide latest20 daily receipt is invalid: {source_date}"
            )
        source_days.append(
            {
                "date": source_date,
                "source_feature_validated": True,
                "source_feature_sha256": feature_sha256,
                "source_archive_validated": True,
                "source_archive_sha256": archive_sha256,
                "matching_games": int(row.get("records") or 0),
                "matching_decisions": int(row.get("decisions") or 0),
            }
        )
    return {
        "days": LATEST20_DAYS,
        "dates": raw_dates,
        "date_start": raw_dates[0],
        "date_end": raw_dates[-1],
        "matching_games": sum(row["matching_games"] for row in source_days),
        "matching_decisions": sum(
            row["matching_decisions"] for row in source_days
        ),
        "filter_applied_after_window_selection": True,
        "filter_archetype": specialist_id,
        "all_dates_represented": True,
        "evidence_source": (
            "current_deck_guide_ready_receipt_and_daily_receipts"
        ),
        "ready_receipt": str(ready_path),
        "ready_receipt_sha256": _sha256(ready_path),
    }


def _latest20_evidence(
    manifest: dict[str, Any],
    *,
    specialist_id: str,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    window = dict(manifest.get("source_window") or {})
    rows = list(manifest.get("source_days") or ())
    raw_dates = [str(value) for value in window.get("dates") or ()]
    try:
        parsed_dates = [date.fromisoformat(value) for value in raw_dates]
    except ValueError as exc:
        raise RuntimeError("latest20 source window contains an invalid date") from exc
    expected_dates = (
        [parsed_dates[0] + timedelta(days=index) for index in range(LATEST20_DAYS)]
        if parsed_dates
        else []
    )
    row_by_date = {
        str(row.get("date") or ""): dict(row)
        for row in rows
        if isinstance(row, dict)
    }
    if not window and not rows and manifest_path is not None:
        return _guide_latest20_evidence(
            manifest_path,
            specialist_id=specialist_id,
        )
    if (
        window.get("unit") != "calendar_day"
        or window.get("selection")
        != "latest_available_fully_validated_daily_sources"
        or int(window.get("days") or 0) != LATEST20_DAYS
        or len(raw_dates) != LATEST20_DAYS
        or len(set(raw_dates)) != LATEST20_DAYS
        or parsed_dates != expected_dates
        or window.get("filter_applied_after_window_selection") is not True
        or str(window.get("filter_archetype") or "") != specialist_id
        or len(rows) != LATEST20_DAYS
        or set(row_by_date) != set(raw_dates)
        or str((manifest.get("selection") or {}).get("value") or "")
        != specialist_id
    ):
        raise RuntimeError(
            "expert corpus is not an exact latest-20 calendar-day "
            "window filtered after selection"
        )
    for source_date in raw_dates:
        row = row_by_date[source_date]
        if (
            row.get("source_feature_validated") is not True
            or not _is_sha256(row.get("source_feature_sha256"))
            or row.get("source_archive_validated") is not True
            or not _is_sha256(row.get("source_archive_sha256"))
            or int(row.get("matching_games") or 0) < 0
            or int(row.get("matching_decisions") or 0) < 0
        ):
            raise RuntimeError(
                f"latest20 source day is not checksum-validated: {source_date}"
            )
    return {
        "days": LATEST20_DAYS,
        "dates": raw_dates,
        "date_start": raw_dates[0],
        "date_end": raw_dates[-1],
        "matching_games": sum(
            int(row_by_date[value].get("matching_games") or 0)
            for value in raw_dates
        ),
        "matching_decisions": sum(
            int(row_by_date[value].get("matching_decisions") or 0)
            for value in raw_dates
        ),
        "filter_applied_after_window_selection": True,
        "filter_archetype": specialist_id,
        "all_dates_represented": True,
    }


def _validated_historical_shards(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    specialist_id: str,
) -> int:
    quality = dict(manifest.get("quality_gates") or {})
    shards = list(manifest.get("shards") or ())
    if (
        str((manifest.get("selection") or {}).get("value") or "")
        != specialist_id
        or quality.get("passed") is not True
        or quality.get("checksummed") is not True
        or not shards
    ):
        raise RuntimeError("historical fallback is not a validated shard corpus")
    for row in shards:
        if not isinstance(row, dict):
            raise RuntimeError("historical fallback shard row is invalid")
        raw_path = Path(str(row.get("path") or ""))
        shard = (
            raw_path.resolve()
            if raw_path.is_absolute()
            else (manifest_path.parent / raw_path).resolve()
        )
        if (
            not shard.is_file()
            or row.get("sha256") != _sha256(shard)
            or int(row.get("bytes") or -1) != shard.stat().st_size
        ):
            raise RuntimeError("historical fallback shard checksum changed")
    return len(shards)


def validate_corpus_source_contract(
    pointer: Path,
    *,
    specialist_id: str,
) -> dict[str, Any]:
    """Validate an authorized immutable expert-corpus source contract."""

    pointer = pointer.expanduser().resolve()
    payload = _load_json(pointer)
    if (
        payload.get("schema") != "poke_bot.pinned_expert_corpus/v1"
        or payload.get("protected") is not True
    ):
        raise RuntimeError("protected expert corpus pointer contract changed")
    manifest_path = _manifest_from_pointer(pointer, payload)
    manifest = _load_json(manifest_path)
    policy = dict(payload.get("source_policy") or {})
    mode = str(policy.get("mode") or PRIMARY_SOURCE_MODE)
    if mode == PRIMARY_SOURCE_MODE:
        latest20 = _latest20_evidence(
            manifest,
            specialist_id=specialist_id,
            manifest_path=manifest_path,
        )
        return {
            "mode": PRIMARY_SOURCE_MODE,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "latest20": latest20,
            "historical_fallback": False,
            "masquerades_as_latest20": False,
        }
    if mode == PUBLIC_FULL_HISTORY_SOURCE_MODE:
        validated_shards = _validated_historical_shards(
            manifest_path,
            manifest,
            specialist_id=specialist_id,
        )
        catalog_raw = str(policy.get("public_deck_catalog") or "").strip()
        catalog_path = (pointer.parent / catalog_raw).resolve()
        if (
            not catalog_raw
            or catalog_path.parent != pointer.parent
            or not catalog_path.is_file()
            or policy.get("public_deck_catalog_sha256")
            != _sha256(catalog_path)
        ):
            raise RuntimeError(
                "public full-history corpus lacks its checksum-bound deck catalog"
            )
        catalog = _load_json(catalog_path)
        window = dict(catalog.get("source_window") or {})
        observed_by_day = dict(catalog.get("observed_by_day") or {})
        try:
            start = date.fromisoformat(str(window.get("start") or ""))
            end = date.fromisoformat(str(window.get("end") or ""))
        except ValueError as exc:
            raise RuntimeError(
                "public full-history deck catalog has an invalid date"
            ) from exc
        dates = [
            (start + timedelta(days=index)).isoformat()
            for index in range((end - start).days + 1)
        ]
        minimum_records = int(policy.get("minimum_records") or 0)
        observed_records = int(
            catalog.get("observed_acting_seat_games") or 0
        )
        manifest_records = int(
            (manifest.get("totals") or {}).get("records_kept") or 0
        )
        manifest_dates = [
            str(value) for value in (manifest.get("dates") or ())
        ]
        if not manifest_dates:
            manifest_dates = sorted(
                {
                    str(value)
                    for row in (manifest.get("shards") or ())
                    for value in (row.get("source_dates") or ())
                }
            )
        if (
            catalog.get("schema")
            != "poke_bot.public_deck_archetype_catalog/v1"
            or catalog.get("specialist_id") != specialist_id
            or not str(catalog.get("source") or "").startswith("https://")
            or int(window.get("days") or 0) != len(dates)
            or end < start
            or sorted(observed_by_day) != dates
            or sum(int(value) for value in observed_by_day.values())
            != observed_records
            or observed_records < minimum_records
            or manifest_records < minimum_records
            or manifest_dates != dates
            or dict(policy.get("source_window") or {}) != window
        ):
            raise RuntimeError(
                "public full-history corpus identity or record floor changed"
            )
        return {
            "mode": PUBLIC_FULL_HISTORY_SOURCE_MODE,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "public_deck_catalog": str(catalog_path),
            "public_deck_catalog_sha256": _sha256(catalog_path),
            "source_window": window,
            "observed_public_acting_seat_games": observed_records,
            "minimum_records": minimum_records,
            "manifest_records": manifest_records,
            "validated_shards": validated_shards,
            "historical_fallback": False,
            "masquerades_as_latest20": False,
        }
    if mode != FALLBACK_SOURCE_MODE:
        raise RuntimeError(f"unknown expert corpus source mode: {mode}")
    validated_shards = _validated_historical_shards(
        manifest_path,
        manifest,
        specialist_id=specialist_id,
    )
    evidence_raw = str(policy.get("latest20_zero_match_manifest") or "").strip()
    evidence_path = (pointer.parent / evidence_raw).resolve()
    if (
        not evidence_raw
        or evidence_path.parent != pointer.parent
        or not evidence_path.is_file()
        or policy.get("latest20_zero_match_manifest_sha256")
        != _sha256(evidence_path)
        or policy.get("reason") != "latest20_matching_games_exactly_zero"
        or policy.get("fallback_is_latest20") is not False
    ):
        raise RuntimeError(
            "historical fallback lacks explicit checksum-bound latest20 evidence"
        )
    latest20 = _latest20_evidence(
        _load_json(evidence_path),
        specialist_id=specialist_id,
        manifest_path=evidence_path,
    )
    if int(latest20["matching_games"]) != 0:
        raise RuntimeError(
            "historical fallback is forbidden when latest20 has matching games"
        )
    return {
        "mode": FALLBACK_SOURCE_MODE,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "latest20_zero_match_manifest": str(evidence_path),
        "latest20_zero_match_manifest_sha256": _sha256(evidence_path),
        "latest20": latest20,
        "historical_fallback": True,
        "validated_historical_shards": validated_shards,
        "masquerades_as_latest20": False,
    }


def select(
    *,
    state_path: Path,
    corpus_root: Path,
    minimum_decisions: int,
    minimum_decisions_by_specialist: dict[str, int] | None = None,
    minimum_records_by_specialist: dict[str, int] | None = None,
    strict_priority_prefix: list[str] | None = None,
    completed_ids: set[str] | None = None,
    active_id: str | None = None,
    routable_ids: set[str] | None = None,
) -> dict[str, Any]:
    state_path = state_path.expanduser().resolve()
    corpus_root = corpus_root.expanduser().resolve()
    state = _load_yaml(state_path)
    roster_path = state_path.parent / "matchup_adapter_roster.json"
    roster = _load_json(roster_path)
    roster_ids = [
        str(value)
        for value in (roster.get("expert_ids") or [])
        if str(value)
    ]
    slot_route_ids = {
        str(row.get("archetype_id") or "")
        for row in (roster.get("slots") or [])
        if isinstance(row, dict)
        and str(row.get("status") or "") in {"active", "dormant"}
        and str(row.get("archetype_id") or "")
    }
    legacy_prefix_length = int(
        roster.get("legacy_v5_prefix_length") or 0
    )
    if (
        not roster_ids
        or len(roster_ids) != len(set(roster_ids))
        or int(roster.get("required_specialist_count") or 0) != len(roster_ids)
        or int(roster.get("physical_checkpoint_rows") or 0)
        != legacy_prefix_length
        or legacy_prefix_length <= 0
        or slot_route_ids != set(roster_ids)
        or len(roster.get("slots") or []) != int(
            roster.get("slot_capacity") or 0
        )
    ):
        raise RuntimeError("canonical matchup-adapter roster is invalid")
    rows = {
        str(row.get("id") or ""): dict(row)
        for row in (state.get("specialists") or [])
        if isinstance(row, dict) and str(row.get("id") or "")
    }
    expected_total = int(
        ((state.get("target_registry") or {}).get("required_target_count") or 0)
    )
    order = list(
        ((state.get("training_priority") or {}).get(
            "ordered_unfinished_ids_after_active"
        ) or [])
    )
    staged_successor = str(
        ((state.get("current") or {}).get("staged_successor_specialist") or "")
    ).strip()
    completed = {str(value) for value in (completed_ids or set())}
    active = str(active_id or "")
    # A passing checkpoint is registered before the mutable priority
    # projection is reconciled.  Normalize only checksum-verified completed
    # IDs (and the outgoing active ID) so that this receipt-backed boundary
    # cannot burn a service retry while still rejecting every other roster
    # inconsistency.
    boundary_exclusions = completed | ({active} if active else set())
    effective_order = [
        specialist_id
        for specialist_id in order
        if specialist_id not in boundary_exclusions
    ]
    progress = dict((state.get("current") or {}).get("program_progress") or {})
    remaining_projection = progress.get("remaining_after_active")
    if remaining_projection is None:
        # Canonical state calls this value ``remaining_unfinished`` whenever
        # no specialist is active at a handoff boundary. Older live-state
        # projections used ``remaining_after_active``. They are the same
        # count at that boundary; accept either spelling without weakening
        # the roster/set checks below.
        remaining_projection = progress.get("remaining_unfinished")
    unfinished_ids = {
        specialist_id
        for specialist_id, row in rows.items()
        if str(row.get("status") or "") in UNFINISHED
    }
    effective_unfinished_ids = unfinished_ids - boundary_exclusions
    declared_required_total = progress.get("required_specialists_total")
    if (
        expected_total != len(rows)
        or (
            declared_required_total is not None
            and int(declared_required_total) != expected_total
        )
        or int(remaining_projection if remaining_projection is not None else -1)
        not in {len(order), len(effective_order)}
        or len(set(order)) != len(order)
        or not set(order).issubset(rows)
        or not effective_unfinished_ids.issubset(set(effective_order))
    ):
        raise RuntimeError(
            "canonical specialist plan or unfinished priority projection "
            "drifted"
        )
    order = effective_order
    # Once the controller has staged a concrete successor, both background
    # pre-stage and the terminal handoff must resolve that same identity.
    # Otherwise a newly arriving higher-priority corpus can race an already
    # prepared successor and make the terminal transition disagree with the
    # receipt that was validated while production was live.  A stale pin that
    # names the active or a completed specialist is ignored so the next
    # one-ahead cycle can begin before the mutable projection catches up.
    effective_staged_successor = (
        staged_successor
        if staged_successor
        and staged_successor not in boundary_exclusions
        else None
    )
    if (
        effective_staged_successor is not None
        and effective_staged_successor not in order
    ):
        raise RuntimeError(
            "staged successor is not in the canonical unfinished priority "
            f"projection: {effective_staged_successor}"
        )
    minimum_overrides = {
        str(key): int(value)
        for key, value in (minimum_decisions_by_specialist or {}).items()
    }
    minimum_record_overrides = {
        str(key): int(value)
        for key, value in (minimum_records_by_specialist or {}).items()
    }
    strict_prefix = [str(value) for value in (strict_priority_prefix or [])]
    if (
        any(value <= 0 for value in minimum_overrides.values())
        or any(value <= 0 for value in minimum_record_overrides.values())
        or not set(minimum_overrides).issubset(rows)
        or not set(minimum_record_overrides).issubset(rows)
        or len(set(strict_prefix)) != len(strict_prefix)
        or not set(strict_prefix).issubset(rows)
    ):
        raise RuntimeError("specialist selection override contract changed")

    deferred: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    routable = (
        {str(value) for value in routable_ids}
        if routable_ids is not None
        else None
    )
    for rank, specialist_id in enumerate(order):
        specialist_id = str(specialist_id)
        row = rows.get(specialist_id)
        if row is None:
            raise RuntimeError(f"priority target missing from state: {specialist_id}")
        if specialist_id in completed or specialist_id == active:
            continue
        if routable is not None and specialist_id not in routable:
            deferred.append(
                {
                    "specialist_id": specialist_id,
                    "priority_rank": rank,
                    "reason": "validated_causal_runtime_route_missing",
                }
            )
            continue
        if str(row.get("status") or "") not in UNFINISHED:
            continue
        pointer = corpus_root / specialist_id / "PROTECTED_EXPERT_CORPUS.json"
        if not pointer.is_file():
            deferred.append(
                {
                    "specialist_id": specialist_id,
                    "priority_rank": rank,
                    "reason": "protected_expert_corpus_missing",
                    "pointer": str(pointer),
                }
            )
            continue
        payload = _load_json(pointer)
        decisions = int((payload.get("totals") or {}).get("decisions_kept") or 0)
        records = int((payload.get("totals") or {}).get("records_kept") or 0)
        required_decisions = minimum_overrides.get(
            specialist_id, int(minimum_decisions)
        )
        required_records = minimum_record_overrides.get(specialist_id, 0)
        try:
            source_contract = validate_corpus_source_contract(
                pointer,
                specialist_id=specialist_id,
            )
        except RuntimeError as exc:
            deferred.append(
                {
                    "specialist_id": specialist_id,
                    "priority_rank": rank,
                    "reason": "expert_corpus_source_contract_invalid",
                    "detail": str(exc),
                    "pointer": str(pointer),
                }
            )
            continue
        if (
            payload.get("schema") != "poke_bot.pinned_expert_corpus/v1"
            or payload.get("protected") is not True
            or decisions < required_decisions
            or records < required_records
        ):
            deferred.append(
                {
                    "specialist_id": specialist_id,
                    "priority_rank": rank,
                    "reason": "protected_expert_corpus_below_contract",
                    "decisions": decisions,
                    "minimum_decisions": required_decisions,
                    **(
                        {
                            "records": records,
                            "minimum_records": required_records,
                        }
                        if required_records
                        else {}
                    ),
                    "pointer": str(pointer),
                }
            )
            continue
        eligible.append(
            {
                "specialist_id": specialist_id,
                "priority_rank": rank,
                "decisions": decisions,
                "minimum_decisions": required_decisions,
                **(
                    {
                        "records": records,
                        "minimum_records": required_records,
                    }
                    if required_records
                    else {}
                ),
                "pointer": str(pointer),
                "source_contract": source_contract,
            }
        )

    if effective_staged_successor is not None:
        selected = next(
            (
                row
                for row in eligible
                if row["specialist_id"] == effective_staged_successor
            ),
            None,
        )
        if selected is None:
            reason = next(
                (
                    str(row["reason"])
                    for row in deferred
                    if row["specialist_id"] == effective_staged_successor
                ),
                "not_eligible",
            )
            raise RuntimeError(
                "staged successor "
                f"{effective_staged_successor} is not executable: {reason}"
            )
    else:
        selected = eligible[0] if eligible else None
    if selected is None:
        raise RuntimeError(
            "no unfinished specialist currently has a protocol-valid corpus"
        )
    strict_expected = next(
        (
            specialist_id
            for specialist_id in strict_prefix
            if specialist_id in order
            and specialist_id not in completed
            and specialist_id != active
            and str(rows[specialist_id].get("status") or "") in UNFINISHED
        ),
        None,
    )
    if (
        strict_expected is not None
        and str(selected["specialist_id"]) != strict_expected
    ):
        reason = next(
            (
                str(row["reason"])
                for row in deferred
                if row["specialist_id"] == strict_expected
            ),
            "not_eligible",
        )
        raise RuntimeError(
            f"strict priority specialist {strict_expected} is not executable: "
            f"{reason}"
        )
    return {
        "schema": RESULT_SCHEMA,
        "required_specialists_total": expected_total,
        "remaining_unfinished": len(order),
        # Retained for v1 receipt compatibility. The value is now the live
        # unfinished count rather than a permanent post-Starmie constant.
        "remaining_after_starmie": len(order),
        "minimum_decisions": int(minimum_decisions),
        "minimum_decisions_by_specialist": minimum_overrides,
        "minimum_records_by_specialist": minimum_record_overrides,
        "strict_priority_prefix": strict_prefix,
        "staged_successor_specialist": effective_staged_successor,
        "completed_specialist_ids": sorted(completed),
        "active_specialist_id": active or None,
        "routable_specialist_ids": (
            sorted(routable) if routable is not None else None
        ),
        "selected": selected,
        "deferred_higher_priority": [
            row
            for row in deferred
            if int(row["priority_rank"]) < int(selected["priority_rank"])
        ],
        "eligible_in_priority_order": eligible,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--minimum-decisions", type=int, default=20_000)
    parser.add_argument(
        "--minimum-decisions-override",
        action="append",
        default=[],
        metavar="SPECIALIST_ID=COUNT",
    )
    parser.add_argument(
        "--minimum-records-override",
        action="append",
        default=[],
        metavar="SPECIALIST_ID=COUNT",
    )
    parser.add_argument(
        "--strict-priority-specialist",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--completed-specialist",
        action="append",
        default=[],
        help="Checksum-verified frozen specialist ID to exclude from selection.",
    )
    parser.add_argument("--active-specialist")
    parser.add_argument(
        "--runtime-tree",
        type=Path,
        help="Require selection to have a validated accepted causal route.",
    )
    args = parser.parse_args()
    if int(args.minimum_decisions) <= 0:
        raise ValueError("minimum decisions must be positive")
    minimum_overrides: dict[str, int] = {}
    for raw in args.minimum_decisions_override:
        specialist_id, separator, count = str(raw).partition("=")
        if not separator or not specialist_id or int(count) <= 0:
            raise ValueError(
                "minimum decision override must be SPECIALIST_ID=positive-count"
            )
        minimum_overrides[specialist_id] = int(count)
    minimum_record_overrides: dict[str, int] = {}
    for raw in args.minimum_records_override:
        specialist_id, separator, count = str(raw).partition("=")
        if not separator or not specialist_id or int(count) <= 0:
            raise ValueError(
                "minimum record override must be SPECIALIST_ID=positive-count"
            )
        minimum_record_overrides[specialist_id] = int(count)
    routable_ids = None
    if args.runtime_tree is not None:
        tree = _load_json(args.runtime_tree.expanduser().resolve())
        routable_ids = {
            str(value)
            for value in (tree.get("runtime_contract") or {}).get(
                "accepted_archetype_ids", ()
            )
        }
    print(
        json.dumps(
            select(
                state_path=args.state,
                corpus_root=args.corpus_root,
                minimum_decisions=int(args.minimum_decisions),
                minimum_decisions_by_specialist=minimum_overrides,
                minimum_records_by_specialist=minimum_record_overrides,
                strict_priority_prefix=list(args.strict_priority_specialist),
                completed_ids=set(args.completed_specialist),
                active_id=args.active_specialist,
                routable_ids=routable_ids,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
